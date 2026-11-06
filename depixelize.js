// depixelize.js — Resolution-independent vectorization of pixel art.
// Implementation of Kopf & Lischinski, "Depixelizing Pixel Art" (SIGGRAPH 2011).
// Single file, zero dependencies, browser + Node.
//
// Usage (browser):   const svg = depixelize(imageData, opts);
// Usage (Node):      const svg = depixelize({ data, width, height }, opts);
//
// Options:
//   scale        40      SVG units per input pixel
//   mode         'auto'  'cell'  — a region is a group of similar pixels (as in
//                                  the paper); colour varies smoothly inside it
//                        'shape' — a region is a group of identical pixels;
//                                  every colour boundary becomes a curve
//                        'auto'  — per region: banded shading stays crisp,
//                                  gradients / dither / noise are smoothed
//   smooth       true    optimise the curve control points (paper §3.4)
//   iterations   20      relaxation sweeps
//   guesses      20      random candidates per control point per sweep
//   guessOffset  0.05    max displacement of one candidate (pixels)
//   shading      0.7     σ (pixels) of the edge-aware colour interpolation used
//                        inside multi-colour regions; 0 = flat cells
//   seed         1       PRNG seed — output is deterministic
//
// Pipeline
//   1. Similarity graph (hqx YUV thresholds, alpha aware) and resolution of
//      crossing diagonals (curves / sparse pixels / islands heuristics).
//   2. Reshaped pixel cells: the simplified generalised Voronoi diagram is
//      pasted together from per-corner templates; every node sits on the
//      ¼-pixel lattice and every cell edge knows the pixel on its other side.
//   3. Regions: connected pixels that are rendered as one area.
//   4. Visible edges (between regions, or on the image frame) are chained into
//      curves that are SHARED by the two regions they separate. 3-way
//      junctions become T-junctions (contour edges beat shading edges,
//      otherwise the straightest pair continues).
//   5. Quadratic B-splines through the node lattice, corner patterns (Fig. 7)
//      excluded from the smoothness term, control points optimised by random
//      relaxation of  E = ∫|κ| ds + |p − p̂|⁴  (∫|κ| ds evaluated exactly).
//   6. SVG: every region outline is assembled from the shared curve pieces, so
//      neighbours meet exactly — no gaps and no pixel backdrop. Regions with
//      several colours get an edge-aware Gaussian colour interpolation.

'use strict';

// ============================================================================
// Constants and small helpers
// ============================================================================
const ALPHA_CUT    = 16;    // alpha below this counts as fully transparent
const SHADING_DIST = 100;   // YUV distance up to which a visible edge is a "shading" edge
const FRAGMENT     = 3;     // similar-colour groups up to this size may be absorbed…
const ABSORB_DIST  = 40;    // …into an edge-adjacent group within this YUV distance
const UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3;
const TWO_PI = Math.PI * 2;

// Which node terminates a cell side that arrives at a pixel corner, indexed
// [cornerType * 4 + direction]. Corner types: 0 = no diagonal through the
// corner, 1 = TL–BR diagonal kept, 2 = TR–BL diagonal kept. A chamfered corner
// C is replaced by two nodes (slot 0 / slot 1):
//   type 1: a = C + (¼, −¼), b = C + (−¼, ¼)     type 2: a = C + (−¼, −¼), b = C + (¼, ¼)
const E_SLOT = new Uint8Array([
  0, 0, 0, 0,
  0, 0, 1, 1,
  0, 1, 1, 0,
]);

const DEFAULTS = {
  scale: 40, mode: 'auto', smooth: true, iterations: 20, guesses: 20,
  guessOffset: 0.05, shading: 0.7, seed: 1,
};

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function grow(a, n) { const b = new a.constructor(n); b.set(a); return b; }

// Corner patterns of Fig. 7 in ¼-pixel units relative to the pixel's top-left
// corner, expanded to all 8 symmetries and both traversal directions. These
// are the places where a curve wraps tightly around a single pixel (tips of
// one-pixel-wide features); their curvature is intentional.
const CORNER_PATTERNS = (() => {
  const base = [
    [0, 4, 1, 1, 3, 1, 4, 4],
    [-1, 3, 1, 1, 3, 1, 4, 4],
    [-1, 3, 1, 1, 3, 1, 5, 3],
    [0, 4, 0, 0, 4, 0, 4, 4],
    [3, 5, 1, 3, 1, 1, 3, 1, 5, 3],
  ];
  const seen = new Set(), out = [];
  for (const p of base) for (let s = 0; s < 8; s++) for (let rev = 0; rev < 2; rev++) {
    const L = p.length >> 1, q = new Int32Array(p.length);
    for (let k = 0; k < L; k++) {
      let dx = p[2 * k] - 2, dy = p[2 * k + 1] - 2;
      if (s & 4) dx = -dx;
      for (let r = 0; r < (s & 3); r++) { const t = dx; dx = -dy; dy = t; }
      const d = rev ? L - 1 - k : k;
      q[2 * d] = dx + 2; q[2 * d + 1] = dy + 2;
    }
    const key = q.join(',');
    if (!seen.has(key)) { seen.add(key); out.push(q); }
  }
  return out;
})();

// ============================================================================
// Curves. Control points P0..Pn for an open curve with n edges (P0, Pn sit on
// junctions), P0..P(n-1) for a closed one. The quadratic B-spline is split
// into one Bézier segment per interior control point s: A_s → (P_s) → C_s.
// Every control point k also has a "node point" Q_k on the curve; the piece
// of curve between Q_k and Q_(k+1) belongs to edge k, which is what lets two
// regions share the same geometry for the same edge.
// ============================================================================
function segA(c, s, o) {
  const x = c.x, y = c.y;
  if (!c.closed && s === 1) { o[0] = x[0]; o[1] = y[0]; return; }
  const p = s === 0 ? c.n - 1 : s - 1;
  o[0] = (x[p] + x[s]) * 0.5; o[1] = (y[p] + y[s]) * 0.5;
}
function segC(c, s, o) {
  const x = c.x, y = c.y, n = c.n;
  if (!c.closed && s === n - 1) { o[0] = x[n]; o[1] = y[n]; return; }
  const q = s === n - 1 ? 0 : s + 1;
  o[0] = (x[s] + x[q]) * 0.5; o[1] = (y[s] + y[q]) * 0.5;
}
const _A = new Float64Array(2), _C = new Float64Array(2);
function nodePt(c, k, o) {
  const x = c.x, y = c.y;
  if (c.border || (!c.closed && (k === 0 || k === c.n))) { o[0] = x[k]; o[1] = y[k]; return; }
  segA(c, k, _A); segC(c, k, _C);
  o[0] = (_A[0] + 2 * x[k] + _C[0]) * 0.25;
  o[1] = (_A[1] + 2 * y[k] + _C[1]) * 0.25;
}

// |turning angle| of the control polygon at b. For a uniform quadratic
// B-spline the segment around control point b is a parabola whose tangent
// turns monotonically from (P_b − P_a) to (P_c − P_b), so ∫|κ| ds over that
// segment equals this angle exactly.
function turn(x, y, a, b, c) {
  const ux = x[b] - x[a], uy = y[b] - y[a], vx = x[c] - x[b], vy = y[c] - y[b];
  return Math.abs(Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy));
}

// Energy of control point i (paper eq. 2–4): smoothness of the three segments
// it influences (corner-pattern segments excluded) + |p − p̂|⁴.
function nodeEnergy(c, i) {
  const x = c.x, y = c.y, ex = c.excl, n = c.n;
  let E = 0;
  if (c.closed) {
    for (let d = -1; d <= 1; d++) {
      let j = i + d; if (j < 0) j += n; else if (j >= n) j -= n;
      if (ex[j]) continue;
      E += turn(x, y, j === 0 ? n - 1 : j - 1, j, j === n - 1 ? 0 : j + 1);
    }
  } else {
    const lo = i > 1 ? i - 1 : 1, hi = i < n - 1 ? i + 1 : n - 1;
    for (let j = lo; j <= hi; j++) if (!ex[j]) E += turn(x, y, j - 1, j, j + 1);
  }
  const dx = x[i] - c.ox[i], dy = y[i] - c.oy[i], d2 = dx * dx + dy * dy;
  return E + d2 * d2;
}

// ============================================================================
// PixelData — the whole geometric pipeline.
// ============================================================================
class PixelData {
  constructor(width, height, data, opts = {}) {
    const w = width | 0, h = height | 0;
    if (w <= 0 || h <= 0) throw new RangeError('depixelize: image must be at least 1×1');
    if (!data || data.length < w * h * 3) throw new RangeError('depixelize: pixel data too short');
    this.w = w; this.h = h; this.N = w * h;
    this._readPixels(data);
    this._buildSimilarityGraph();
    this._removeDiagonals();
    this._buildCells();
    this._buildRegions(opts.mode || 'auto');
    this._buildBoundaries();
    this._traceLoops();
    this._buildCurves();
    this._detectCorners();
  }

  // ---- pixels -----------------------------------------------------------------
  _readPixels(data) {
    const N = this.N, stride = data.length >= N * 4 ? 4 : 3;
    const rgb = this.rgb = new Uint32Array(N);
    const alpha = this.alpha = new Uint8Array(N);
    const key = this.key = new Uint32Array(N);      // exact colour identity, 0 = transparent
    const yuv = this.yuv = new Int16Array(N * 3);
    let opaque = true;
    for (let i = 0, j = 0; i < N; i++, j += stride) {
      const a = stride === 4 ? data[j + 3] : 255;
      if (a < 255) opaque = false;
      if (a < ALPHA_CUT) continue;
      const r = data[j], g = data[j + 1], b = data[j + 2];
      const c = (r << 16) | (g << 8) | b;
      rgb[i] = c; alpha[i] = a;
      key[i] = ((c << 8) | a) >>> 0;
      const y = (77 * r + 150 * g + 29 * b) >> 8;     // ~0.299r + 0.587g + 0.114b
      yuv[i * 3] = y;
      yuv[i * 3 + 1] = (126 * (b - y)) >> 8;           // ~0.492 (b − y)
      yuv[i * 3 + 2] = (224 * (r - y)) >> 8;           // ~0.877 (r − y)
    }
    this.opaque = opaque;
  }

  similar(i, j) {
    const A = this.alpha, ai = A[i], aj = A[j];
    if (ai === 0 || aj === 0) return ai === aj;
    const da = ai - aj; if (da > 32 || da < -32) return false;
    const Y = this.yuv, p = i * 3, q = j * 3;
    const dy = Y[p] - Y[q];         if (dy > 48 || dy < -48) return false;
    const du = Y[p + 1] - Y[q + 1]; if (du > 7 || du < -7) return false;
    const dv = Y[p + 2] - Y[q + 2]; if (dv > 6 || dv < -6) return false;
    return true;
  }

  yuvDist(i, j) {
    const A = this.alpha;
    if (A[i] === 0 || A[j] === 0) return A[i] === A[j] ? 0 : 1e9;
    const Y = this.yuv, p = i * 3, q = j * 3;
    const dy = Y[p] - Y[q], du = Y[p + 1] - Y[q + 1], dv = Y[p + 2] - Y[q + 2], da = A[i] - A[j];
    return Math.sqrt(dy * dy + du * du + dv * dv + da * da);
  }

  // ---- similarity graph ------------------------------------------------------
  // 4 forward bits per pixel: 1 = +x, 2 = +y, 4 = +x+y (SE), 8 = +x−y (NE).
  _buildSimilarityGraph() {
    const { w, h } = this;
    const neigh = this.neigh = new Uint8Array(this.N);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let n = 0;
      if (x + 1 < w && this.similar(i, i + 1)) n |= 1;
      if (y + 1 < h && this.similar(i, i + w)) n |= 2;
      if (x + 1 < w && y + 1 < h && this.similar(i, i + w + 1)) n |= 4;
      if (x + 1 < w && y > 0 && this.similar(i, i - w + 1)) n |= 8;
      neigh[i] = n;
    }
  }

  pxNeighbors(i, out) {
    const w = this.w, h = this.h, neigh = this.neigh;
    const x = i % w, y = (i / w) | 0, n = neigh[i];
    let c = 0;
    if (n & 1) out[c++] = i + 1;
    if (n & 2) out[c++] = i + w;
    if (n & 4) out[c++] = i + w + 1;
    if (n & 8) out[c++] = i - w + 1;
    if (x > 0 && (neigh[i - 1] & 1)) out[c++] = i - 1;
    if (y > 0 && (neigh[i - w] & 2)) out[c++] = i - w;
    if (x > 0 && y > 0 && (neigh[i - w - 1] & 4)) out[c++] = i - w - 1;
    if (x > 0 && y + 1 < h && (neigh[i + w - 1] & 8)) out[c++] = i + w - 1;
    return c;
  }

  pxDegree(i) {
    const w = this.w, h = this.h, neigh = this.neigh;
    const x = i % w, y = (i / w) | 0, n = neigh[i];
    let d = (n & 1) + ((n >> 1) & 1) + ((n >> 2) & 1) + ((n >> 3) & 1);
    if (x > 0 && (neigh[i - 1] & 1)) d++;
    if (y > 0 && (neigh[i - w] & 2)) d++;
    if (x > 0 && y > 0 && (neigh[i - w - 1] & 4)) d++;
    if (x > 0 && y + 1 < h && (neigh[i + w - 1] & 8)) d++;
    return d;
  }

  _rmPxEdge(a, b) {
    const w = this.w;
    const dx = (b % w) - (a % w), dy = ((b / w) | 0) - ((a / w) | 0);
    if      (dx ===  1 && dy ===  0) this.neigh[a] &= ~1;
    else if (dx ===  0 && dy ===  1) this.neigh[a] &= ~2;
    else if (dx ===  1 && dy ===  1) this.neigh[a] &= ~4;
    else if (dx ===  1 && dy === -1) this.neigh[a] &= ~8;
    else if (dx === -1 && dy ===  0) this.neigh[b] &= ~1;
    else if (dx ===  0 && dy === -1) this.neigh[b] &= ~2;
    else if (dx === -1 && dy === -1) this.neigh[b] &= ~4;
    else if (dx === -1 && dy ===  1) this.neigh[b] &= ~8;
  }

  // ---- crossing diagonals (paper §3.2) ------------------------------------------
  _removeDiagonals() {
    const { w, h, neigh } = this;
    const ambiguous = [];
    for (let y = 0; y < h - 1; y++) for (let x = 0; x < w - 1; x++) {
      const tl = y * w + x, tr = tl + 1, bl = tl + w, br = bl + 1;
      if (!(neigh[tl] & 4) || !(neigh[bl] & 8)) continue;
      const card = (neigh[tl] & 1 ? 1 : 0) + (neigh[bl] & 1 ? 1 : 0) +
                   (neigh[tl] & 2 ? 1 : 0) + (neigh[tr] & 2 ? 1 : 0);
      if (card === 4) {                 // flat shaded block: both diagonals are redundant
        neigh[tl] &= ~4; neigh[bl] &= ~8;
      } else if (card === 0) {          // genuinely ambiguous: decided by the heuristics
        ambiguous.push(tl, br, bl, tr);
      } else {
        // Partially connected block (possible because similarity is not
        // transitive). Keep the closer pair so the graph stays planar.
        const d1 = this.yuvDist(tl, br), d2 = this.yuvDist(bl, tr);
        if (d1 >= d2) neigh[tl] &= ~4;
        if (d2 >= d1) neigh[bl] &= ~8;
      }
    }
    if (ambiguous.length) this._applyHeuristics(ambiguous);
  }

  _applyHeuristics(flat) {
    this._mark = new Uint32Array(this.N);
    this._stamp = 0;
    this._stk = new Int32Array(this.N + 8);
    this._nbs = new Int32Array(8);
    for (let k = 0; k < flat.length; k += 4) {
      const a1 = flat[k], b1 = flat[k + 1], a2 = flat[k + 2], b2 = flat[k + 3];
      const w1 = this._hCurve(a1, b1) + this._hSparse(a1, b1) + this._hIsland(a1, b1);
      const w2 = this._hCurve(a2, b2) + this._hSparse(a2, b2) + this._hIsland(a2, b2);
      if (w1 <= w2) this._rmPxEdge(a1, b1);      // tie: both removed
      if (w2 <= w1) this._rmPxEdge(a2, b2);
    }
    this._mark = this._stk = null;
  }

  // Length of the valence-2 curve the diagonal belongs to (in nodes).
  _hCurve(a, b) {
    const mark = this._mark, stk = this._stk, nbs = this._nbs, s = ++this._stamp;
    mark[a] = s; mark[b] = s;
    let sp = 0, count = 2;
    stk[sp++] = a; stk[sp++] = b;
    while (sp) {
      const n = stk[--sp];
      if (this.pxDegree(n) !== 2) continue;
      const nc = this.pxNeighbors(n, nbs);
      for (let k = 0; k < nc; k++) {
        const m = nbs[k];
        if (mark[m] !== s) { mark[m] = s; count++; stk[sp++] = m; }
      }
    }
    return count;
  }

  // Minus the size of the component in an 8×8 window centred on the block.
  _hSparse(a, b) {
    const w = this.w, mark = this._mark, stk = this._stk, nbs = this._nbs, s = ++this._stamp;
    const x0 = Math.min(a % w, b % w) - 3, y0 = Math.min((a / w) | 0, (b / w) | 0) - 3;
    mark[a] = s; mark[b] = s;
    let sp = 0, count = 2;
    stk[sp++] = a; stk[sp++] = b;
    while (sp) {
      const n = stk[--sp], nc = this.pxNeighbors(n, nbs);
      for (let k = 0; k < nc; k++) {
        const m = nbs[k];
        if (mark[m] === s) continue;
        const mx = (m % w) - x0, my = ((m / w) | 0) - y0;
        if (mx >= 0 && mx < 8 && my >= 0 && my < 8) { mark[m] = s; count++; stk[sp++] = m; }
      }
    }
    return -count;
  }

  _hIsland(a, b) {
    return (this.pxDegree(a) === 1 || this.pxDegree(b) === 1) ? 5 : 0;
  }

  // ---- reshaped cells (paper §3.2, simplified Voronoi diagram) ------------------
  // Nodes: two slots per pixel corner, id = corner * 2 + slot, coordinates in
  // ¼-pixel units. Cell polygons run clockwise on screen (y down); edge k goes
  // from vertex k to vertex k+1 and cellNb holds the pixel across it (−1 =
  // outside the image).
  _buildCells() {
    const { w, h, N, neigh } = this;
    const CW = w + 1, CH = h + 1;
    const ctype = this.ctype = new Uint8Array(CW * CH);
    for (let cy = 1; cy < h; cy++) for (let cx = 1; cx < w; cx++) {
      const tl = (cy - 1) * w + cx - 1;
      if (neigh[tl] & 4) ctype[cy * CW + cx] = 1;
      else if (neigh[tl + w] & 8) ctype[cy * CW + cx] = 2;
    }
    const NN = this.nodeCount = CW * CH * 2;
    const X = this.nodeX = new Int32Array(NN), Y = this.nodeY = new Int32Array(NN);
    for (let cy = 0, c = 0; cy < CH; cy++) for (let cx = 0; cx < CW; cx++, c++) {
      const t = ctype[c], x4 = cx * 4, y4 = cy * 4, n = c * 2;
      if (t === 0)      { X[n] = x4;     Y[n] = y4;     X[n + 1] = x4;     Y[n + 1] = y4; }
      else if (t === 1) { X[n] = x4 + 1; Y[n] = y4 - 1; X[n + 1] = x4 - 1; Y[n + 1] = y4 + 1; }
      else              { X[n] = x4 - 1; Y[n] = y4 - 1; X[n + 1] = x4 + 1; Y[n + 1] = y4 + 1; }
    }
    const V = this.cellV = new Int32Array(N * 8);
    const NB = this.cellNb = new Int32Array(N * 8);
    const CN = this.cellN = new Uint8Array(N);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, base = i * 8;
      const c00 = y * CW + x, c10 = c00 + 1, c01 = c00 + CW, c11 = c01 + 1;
      const t00 = ctype[c00] * 4, t10 = ctype[c10] * 4, t01 = ctype[c01] * 4, t11 = ctype[c11] * 4;
      const a0 = 2 * c00 + E_SLOT[t00 + DOWN],  a1 = 2 * c00 + E_SLOT[t00 + RIGHT];
      const b0 = 2 * c10 + E_SLOT[t10 + LEFT],  b1 = 2 * c10 + E_SLOT[t10 + DOWN];
      const d0 = 2 * c11 + E_SLOT[t11 + UP],    d1 = 2 * c11 + E_SLOT[t11 + LEFT];
      const e0 = 2 * c01 + E_SLOT[t01 + RIGHT], e1 = 2 * c01 + E_SLOT[t01 + UP];
      let k = 0;
      V[base] = a0;
      if (a1 !== a0) { NB[base + k] = i - w - 1; V[base + ++k] = a1; }   // chamfer, NW neighbour
      NB[base + k++] = y > 0 ? i - w : -1;                               // top side
      V[base + k] = b0;
      if (b1 !== b0) { NB[base + k] = i - w + 1; V[base + ++k] = b1; }   // chamfer, NE
      NB[base + k++] = x < w - 1 ? i + 1 : -1;                           // right side
      V[base + k] = d0;
      if (d1 !== d0) { NB[base + k] = i + w + 1; V[base + ++k] = d1; }   // chamfer, SE
      NB[base + k++] = y < h - 1 ? i + w : -1;                           // bottom side
      V[base + k] = e0;
      if (e1 !== e0) { NB[base + k] = i + w - 1; V[base + ++k] = e1; }   // chamfer, SW
      NB[base + k++] = x > 0 ? i - 1 : -1;                               // left side
      CN[i] = k;
    }
    // Twin slot of every interior cell edge (same edge, opposite direction).
    const TW = this.cellTwin = new Int8Array(N * 8).fill(-1);
    for (let i = 0; i < N; i++) {
      const base = i * 8, n = CN[i];
      for (let k = 0; k < n; k++) {
        const nb = NB[base + k];
        if (nb < 0) continue;
        const u = V[base + k], v = V[base + (k + 1 === n ? 0 : k + 1)];
        const nbase = nb * 8, nn = CN[nb];
        for (let t = 0; t < nn; t++) {
          if (V[nbase + t] === v && V[nbase + (t + 1 === nn ? 0 : t + 1)] === u) { TW[base + k] = t; break; }
        }
        if (TW[base + k] < 0) throw new Error('depixelize: inconsistent cell graph');
      }
    }
  }

  // ---- regions -----------------------------------------------------------------
  _components(exact) {
    const N = this.N, key = this.key;
    const comp = new Int32Array(N).fill(-1), stack = new Int32Array(N), nbs = new Int32Array(8);
    const sizes = [], first = [];
    let id = 0;
    for (let i = 0; i < N; i++) {
      if (comp[i] >= 0) continue;
      const ki = key[i];
      let sp = 0, size = 0;
      stack[sp++] = i; comp[i] = id;
      while (sp) {
        const n = stack[--sp];
        size++;
        const nc = this.pxNeighbors(n, nbs);
        for (let k = 0; k < nc; k++) {
          const m = nbs[k];
          if (comp[m] < 0 && (!exact || key[m] === ki)) { comp[m] = id; stack[sp++] = m; }
        }
      }
      sizes.push(size); first.push(i); id++;
    }
    return { comp, count: id, sizes, first };
  }

  // Noise suppression (auto/cell): tiny groups of pixels whose colour is only
  // slightly off from an edge-adjacent group are merged into it, so they are
  // shaded smoothly instead of becoming isolated cells. Real pixel-art
  // features (an eye, a highlight) are far more distinct and are kept.
  _absorbFragments(sim) {
    const N = this.N, comp = sim.comp, NB = this.cellNb, CN = this.cellN;
    const parent = new Int32Array(sim.count);
    for (let c = 0; c < sim.count; c++) parent[c] = c;
    const find = c => { while (parent[c] !== c) c = parent[c] = parent[parent[c]]; return c; };
    const size = new Int32Array(sim.count);
    for (let c = 0; c < sim.count; c++) size[c] = sim.sizes[c];
    const members = [];   // pixels of the current small groups, per root
    for (let pass = 0; pass < 4; pass++) {
      const bestD = new Float64Array(sim.count).fill(Infinity), bestC = new Int32Array(sim.count).fill(-1);
      let any = false;
      for (let i = 0; i < N; i++) {
        const c = find(comp[i]);
        if (size[c] > FRAGMENT) continue;
        for (let k = 0, base = i * 8; k < CN[i]; k++) {
          const nb = NB[base + k];
          if (nb < 0) continue;
          const o = find(comp[nb]);
          if (o === c) continue;
          const d = this.yuvDist(i, nb);
          if (d < bestD[c] || (d === bestD[c] && size[o] > size[bestC[c]])) { bestD[c] = d; bestC[c] = o; }
        }
      }
      for (let c = 0; c < sim.count; c++) {
        if (parent[c] !== c || size[c] > FRAGMENT || bestC[c] < 0 || bestD[c] > ABSORB_DIST) continue;
        const o = find(bestC[c]);
        if (o === c) continue;
        parent[c] = o; size[o] += size[c]; any = true;
      }
      if (!any) break;
    }
    const map = new Int32Array(sim.count).fill(-1), out = new Int32Array(N), sizes = [], first = [];
    let n = 0;
    for (let i = 0; i < N; i++) {
      const r = find(comp[i]);
      if (map[r] < 0) { map[r] = n++; sizes.push(0); first.push(i); }
      out[i] = map[r]; sizes[map[r]]++;
    }
    return { comp: out, count: n, sizes, first };
  }

  _buildRegions(mode) {
    const N = this.N;
    const sim = mode === 'shape' ? null : this._absorbFragments(this._components(false));
    let region, count;
    if (mode === 'cell') {
      region = sim.comp; count = sim.count;
    } else {
      const ex = this._components(true);
      if (mode === 'shape') {
        region = ex.comp; count = ex.count;
      } else {
        // auto: a group of similar pixels stays one smoothly shaded region
        // unless it is made of large flat patches (banded shading), in which
        // case each patch gets its own crisp outline.
        const S = sim.count, K = new Int32Array(S), parts = new Int32Array(S), small = new Int32Array(S);
        for (let i = 0; i < N; i++) K[sim.comp[i]]++;
        for (let e = 0; e < ex.count; e++) {
          const s = sim.comp[ex.first[e]];
          parts[s]++;
          if (ex.sizes[e] < 3) small[s] += ex.sizes[e];
        }
        region = new Int32Array(N);
        for (let i = 0; i < N; i++) {
          const s = sim.comp[i];
          const crisp = parts[s] === 1 || (K[s] >= 4 * parts[s] && small[s] * 4 <= K[s]);
          region[i] = crisp ? ex.comp[i] : ex.count + s;
        }
        count = ex.count + S;
      }
    }
    const map = new Int32Array(count).fill(-1);
    let R = 0;
    for (let i = 0; i < N; i++) if (map[region[i]] < 0) map[region[i]] = R++;
    const reg = this.region = new Int32Array(N);
    for (let i = 0; i < N; i++) reg[i] = map[region[i]];
    this.R = R;
    const start = this.rStart = new Int32Array(R + 1);
    for (let i = 0; i < N; i++) start[reg[i] + 1]++;
    for (let r = 0; r < R; r++) start[r + 1] += start[r];
    const pix = this.rPix = new Int32Array(N), fill = start.slice(0, R);
    for (let i = 0; i < N; i++) pix[fill[reg[i]]++] = i;
    const multi = this.rMulti = new Uint8Array(R);
    const kind = this.rKind = new Uint8Array(R);      // 0 opaque, 1 translucent, 2 transparent
    const key = this.key, alpha = this.alpha;
    for (let r = 0; r < R; r++) {
      const k0 = key[pix[start[r]]];
      let m = 0, minA = 255;
      for (let p = start[r]; p < start[r + 1]; p++) {
        const i = pix[p];
        if (key[i] !== k0) m = 1;
        if (alpha[i] < minA) minA = alpha[i];
      }
      multi[r] = m;
      kind[r] = minA === 0 ? 2 : minA < 255 ? 1 : 0;
    }
  }

  // ---- visible edges -------------------------------------------------------------
  _buildBoundaries() {
    const N = this.N, V = this.cellV, NB = this.cellNb, CN = this.cellN, TW = this.cellTwin, reg = this.region;
    const edgeOf = this.edgeOf = new Int32Array(N * 8).fill(-1);
    let cap = 256, E = 0;
    let eU = new Int32Array(cap), eV = new Int32Array(cap), eP = new Int32Array(cap), eQ = new Int32Array(cap);
    for (let i = 0; i < N; i++) {
      const base = i * 8, n = CN[i], ri = reg[i];
      for (let k = 0; k < n; k++) {
        const s = base + k;
        if (edgeOf[s] >= 0) continue;
        const nb = NB[s];
        if (nb >= 0 && reg[nb] === ri) continue;
        if (E === cap) { cap *= 2; eU = grow(eU, cap); eV = grow(eV, cap); eP = grow(eP, cap); eQ = grow(eQ, cap); }
        eU[E] = V[s]; eV[E] = V[base + (k + 1 === n ? 0 : k + 1)]; eP[E] = i; eQ[E] = nb;
        edgeOf[s] = E;
        if (nb >= 0) edgeOf[nb * 8 + TW[s]] = E;
        E++;
      }
    }
    this.E = E; this.eU = eU; this.eV = eV; this.eP = eP; this.eQ = eQ;
    const NN = this.nodeCount;
    const deg = this.bdeg = new Uint8Array(NN);
    for (let e = 0; e < E; e++) { deg[eU[e]]++; deg[eV[e]]++; }
    const st = this.bStart = new Int32Array(NN + 1);
    for (let v = 0; v < NN; v++) st[v + 1] = st[v] + deg[v];
    const inc = this.bInc = new Int32Array(2 * E), fp = st.slice(0, NN);
    for (let e = 0; e < E; e++) { inc[fp[eU[e]]++] = e; inc[fp[eV[e]]++] = e; }
  }

  // Next visible half-edge after half-edge s (slot = pixel * 8 + k) along the
  // boundary of s's region: rotate around the end vertex through the region's
  // own cells until a visible edge is found.
  _nextBoundary(s) {
    const NB = this.cellNb, CN = this.cellN, TW = this.cellTwin, EO = this.edgeOf;
    let p = s >> 3, q = (s & 7) + 1;
    if (q === CN[p]) q = 0;
    for (let guard = 0; guard < 32; guard++) {
      const t = (p << 3) + q;
      if (EO[t] >= 0) return t;
      const tw = TW[t];
      p = NB[t]; q = tw + 1;
      if (q === CN[p]) q = 0;
    }
    throw new Error('depixelize: boundary rotation failed');
  }

  _traceLoops() {
    const N = this.N, V = this.cellV, CN = this.cellN, EO = this.edgeOf, X = this.nodeX, Y = this.nodeY;
    const reg = this.region, R = this.R;
    const seen = new Uint8Array(N * 8);
    const loops = this.loops = [];
    const rLoops = this.rLoops = new Array(R);
    for (let r = 0; r < R; r++) rLoops[r] = [];
    const tmp = [], limit = 2 * this.E + 8;
    for (let i = 0; i < N; i++) {
      const n = CN[i];
      for (let k = 0; k < n; k++) {
        const s0 = i * 8 + k;
        if (EO[s0] < 0 || seen[s0]) continue;
        tmp.length = 0;
        let area = 0, s = s0, guard = 0;
        do {
          seen[s] = 1; tmp.push(s);
          const p = s >> 3, q = s & 7;
          const v0 = V[s], v1 = V[(p << 3) + (q + 1 === CN[p] ? 0 : q + 1)];
          area += X[v0] * Y[v1] - X[v1] * Y[v0];
          s = this._nextBoundary(s);
          if (++guard > limit) throw new Error('depixelize: boundary trace failed');
        } while (s !== s0);
        rLoops[reg[i]].push(loops.length);
        loops.push({ region: reg[i], slots: Int32Array.from(tmp), area: area / 32 });  // px²
      }
    }
    // Outer area of each region (sum of its positive loops) for paint order.
    const rArea = this.rArea = new Float64Array(R);
    for (const L of loops) if (L.area > 0) rArea[L.region] += L.area;
  }

  // ---- curves (paper §3.3) --------------------------------------------------------
  _isShading(e) {
    const q = this.eQ[e];
    return q >= 0 && this.yuvDist(this.eP[e], q) <= SHADING_DIST ? 1 : 0;
  }

  // Direction leaving v along visible edge e, looking up to two edges ahead.
  _dirFrom(v, e, D, o) {
    const eU = this.eU, eV = this.eV, X = this.nodeX, Y = this.nodeY;
    const u = eU[e] === v ? eV[e] : eU[e];
    let tx = X[u], ty = Y[u];
    if (this.bdeg[u] === 2) {
      const s = this.bStart[u], a = this.bInc[s], e2 = a === e ? this.bInc[s + 1] : a;
      const u2 = eU[e2] === u ? eV[e2] : eU[e2];
      tx = X[u2]; ty = Y[u2];
    }
    D[o] = tx - X[v]; D[o + 1] = ty - Y[v];
  }

  _buildCurves() {
    const E = this.E, eU = this.eU, eV = this.eV, eQ = this.eQ;
    const NN = this.nodeCount, deg = this.bdeg, st = this.bStart, inc = this.bInc;
    const X = this.nodeX, Y = this.nodeY;

    // T-junctions: at every 3-way junction pick the two edges that continue.
    const j1 = this.join1 = new Int32Array(NN).fill(-1);
    const j2 = this.join2 = new Int32Array(NN).fill(-1);
    const D = new Float64Array(6);
    const cos2 = (a, b) => (D[a] * D[b] + D[a + 1] * D[b + 1]) /
      Math.sqrt((D[a] * D[a] + D[a + 1] * D[a + 1]) * (D[b] * D[b] + D[b + 1] * D[b + 1]) || 1);
    for (let v = 0; v < NN; v++) {
      if (deg[v] !== 3) continue;
      const s = st[v], ea = inc[s], eb = inc[s + 1], ec = inc[s + 2];
      const ba = eQ[ea] < 0, bb = eQ[eb] < 0, bc = eQ[ec] < 0;
      let p, q;
      if (ba || bb || bc) {                           // the image frame continues
        if (ba && bb) { p = ea; q = eb; } else if (ba && bc) { p = ea; q = ec; } else { p = eb; q = ec; }
      } else {
        const sa = this._isShading(ea), sb = this._isShading(eb), sc = this._isShading(ec);
        if (sa + sb + sc === 1) {                     // contour edges continue through
          if (sa) { p = eb; q = ec; } else if (sb) { p = ea; q = ec; } else { p = ea; q = eb; }
        } else {                                      // straightest pair continues
          this._dirFrom(v, ea, D, 0); this._dirFrom(v, eb, D, 2); this._dirFrom(v, ec, D, 4);
          const cab = cos2(0, 2), cac = cos2(0, 4), cbc = cos2(2, 4);
          if (cab <= cac && cab <= cbc) { p = ea; q = eb; }
          else if (cac <= cbc) { p = ea; q = ec; }
          else { p = eb; q = ec; }
        }
      }
      j1[v] = p; j2[v] = q;
    }

    const curveOf = this.curveOf = new Int32Array(E).fill(-1);
    const idxOf = this.idxOf = new Int32Array(E);
    const curves = this.curves = [];
    const terminal = (v, e) => deg[v] !== 2 && j1[v] !== e && j2[v] !== e;
    const nextEdge = (v, e) => {
      if (deg[v] === 2) { const a = inc[st[v]]; return a === e ? inc[st[v] + 1] : a; }
      return j1[v] === e ? j2[v] : j2[v] === e ? j1[v] : -1;
    };
    const nodes = [], edges = [];
    const make = (v0, e0, closed) => {
      nodes.length = 0; edges.length = 0;
      const id = curves.length;
      let v = v0, e = e0;
      nodes.push(v0);
      for (;;) {
        const u = eU[e] === v ? eV[e] : eU[e];
        curveOf[e] = id; idxOf[e] = edges.length; edges.push(e);
        if (closed) {
          const ne = nextEdge(u, e);
          if (ne === e0) break;
          nodes.push(u);
          if (ne < 0 || curveOf[ne] >= 0) { closed = false; break; }   // defensive
          v = u; e = ne;
        } else {
          nodes.push(u);
          if (terminal(u, e)) break;
          const ne = nextEdge(u, e);
          if (ne < 0 || curveOf[ne] >= 0) break;
          v = u; e = ne;
        }
      }
      const M = nodes.length;
      const c = {
        id, closed, border: eQ[edges[0]] < 0, n: edges.length,
        nodes: Int32Array.from(nodes), edges: Int32Array.from(edges),
        x: new Float64Array(M), y: new Float64Array(M),
        ox: null, oy: null, excl: null,
        att0: -1, att0j: 0, att1: -1, att1j: 0,
      };
      for (let k = 0; k < M; k++) { c.x[k] = X[nodes[k]] * 0.25; c.y[k] = Y[nodes[k]] * 0.25; }
      c.ox = c.x.slice(); c.oy = c.y.slice();
      curves.push(c);
    };
    for (let v = 0; v < NN; v++) {
      const d = deg[v];
      if (d === 0 || d === 2) continue;
      for (let k = st[v]; k < st[v + 1]; k++) {
        const e = inc[k];
        if (curveOf[e] < 0 && terminal(v, e)) make(v, e, false);
      }
    }
    for (let e = 0; e < E; e++) if (curveOf[e] < 0) make(eU[e], e, true);

    // A curve that ends on a T-junction is attached to the curve passing
    // through it, so that its end follows that curve exactly.
    this.attached = [];
    for (const c of curves) {
      if (c.closed || c.border) continue;
      for (let end = 0; end < 2; end++) {
        const v = end ? c.nodes[c.n] : c.nodes[0];
        if (deg[v] !== 3 || j1[v] < 0) continue;
        const e = end ? c.edges[c.n - 1] : c.edges[0];
        if (e === j1[v] || e === j2[v]) continue;
        const h = curves[curveOf[j1[v]]];
        if (h.border) continue;
        const je = idxOf[j1[v]];
        let jn = h.nodes[je] === v ? je : je + 1;
        if (h.closed && jn === h.n) jn = 0;
        if (end) { c.att1 = h.id; c.att1j = jn; } else { c.att0 = h.id; c.att0j = jn; }
      }
      if (c.att0 >= 0 || c.att1 >= 0) this.attached.push(c);
    }
  }

  _detectCorners() {
    const X = this.nodeX, Y = this.nodeY;
    for (const c of this.curves) {
      if (c.border) continue;
      const nodes = c.nodes, M = nodes.length, closed = c.closed;
      const ex = c.excl = new Uint8Array(M);
      for (let pi = 0; pi < CORNER_PATTERNS.length; pi++) {
        const pat = CORNER_PATTERNS[pi], L = pat.length >> 1;
        if (L > M) continue;
        const sEnd = closed ? M : M - L + 1;
        for (let s = 0; s < sEnd; s++) {
          const n0 = nodes[s], ox = X[n0] - pat[0], oy = Y[n0] - pat[1];
          if ((ox & 3) !== 0 || (oy & 3) !== 0) continue;
          let k = 1;
          for (; k < L; k++) {
            let idx = s + k; if (idx >= M) idx -= M;
            const nk = nodes[idx];
            if (X[nk] - pat[2 * k] !== ox || Y[nk] - pat[2 * k + 1] !== oy) break;
          }
          if (k === L) for (let t = 1; t < L - 1; t++) { let idx = s + t; if (idx >= M) idx -= M; ex[idx] = 1; }
        }
      }
    }
  }

  // ---- optimisation (paper §3.4) -----------------------------------------------------
  optimize(o) {
    const iters = Math.max(0, o.iterations | 0), guesses = Math.max(0, o.guesses | 0), off = +o.guessOffset;
    if (!iters || !guesses || !(off > 0)) return;
    const rand = mulberry32(o.seed | 0);
    const live = this.curves.filter(c => !c.border && (c.closed ? c.n >= 3 : c.n >= 2));
    for (let it = 0; it < iters; it++) {
      this._updateAttachments();
      for (let ci = 0; ci < live.length; ci++) {
        const c = live[ci], x = c.x, y = c.y, lo = c.closed ? 0 : 1, hi = c.n - 1;
        for (let i = lo; i <= hi; i++) {
          const x0 = x[i], y0 = y[i];
          let best = nodeEnergy(c, i), bx = x0, by = y0;
          for (let g = 0; g < guesses; g++) {
            const r = rand() * off, th = rand() * TWO_PI;
            x[i] = x0 + r * Math.cos(th); y[i] = y0 + r * Math.sin(th);
            const e = nodeEnergy(c, i);
            if (e < best) { best = e; bx = x[i]; by = y[i]; }
          }
          x[i] = bx; y[i] = by;
        }
      }
    }
  }

  _updateAttachments() {
    const C = this.curves, T = _T;
    let delta = 0;
    for (const c of this.attached) {
      if (c.att0 >= 0) {
        nodePt(C[c.att0], c.att0j, T);
        delta = Math.max(delta, Math.abs(T[0] - c.x[0]) + Math.abs(T[1] - c.y[0]));
        c.x[0] = T[0]; c.y[0] = T[1];
      }
      if (c.att1 >= 0) {
        const n = c.n;
        nodePt(C[c.att1], c.att1j, T);
        delta = Math.max(delta, Math.abs(T[0] - c.x[n]) + Math.abs(T[1] - c.y[n]));
        c.x[n] = T[0]; c.y[n] = T[1];
      }
    }
    return delta;
  }

  // Settle attachments and compute the rendered position of every node.
  finalize() {
    for (let p = 0; p < 64; p++) if (this._updateAttachments() < 1e-12) break;
    const NN = this.nodeCount, X = this.nodeX, Y = this.nodeY;
    const qx = this.qx = new Float64Array(NN), qy = this.qy = new Float64Array(NN);
    for (let v = 0; v < NN; v++) { qx[v] = X[v] * 0.25; qy[v] = Y[v] * 0.25; }
    for (const c of this.curves) {
      const M = c.nodes.length;
      for (let k = 0; k < M; k++) { nodePt(c, k, _T); qx[c.nodes[k]] = _T[0]; qy[c.nodes[k]] = _T[1]; }
    }
  }

  // Structural self-check used by the tests.
  validate() {
    const { N, cellV: V, cellN: CN, nodeX: X, nodeY: Y } = this;
    let area = 0;
    for (let i = 0; i < N; i++) {
      const n = CN[i];
      for (let k = 0; k < n; k++) {
        const a = V[i * 8 + k], b = V[i * 8 + (k + 1) % n];
        area += X[a] * Y[b] - X[b] * Y[a];
      }
    }
    const problems = [];
    if (Math.abs(area / 32 - N) > 1e-9) problems.push('cell area ' + area / 32 + ' != ' + N);
    for (let r = 0; r < this.R; r++) {
      const pos = this.rLoops[r].filter(l => this.loops[l].area > 0).length;
      if (pos !== 1) problems.push('region ' + r + ' has ' + pos + ' outer loops');
    }
    return problems;
  }
}
const _T = new Float64Array(2);

// ============================================================================
// SVG writer
// ============================================================================
let FALLBACK_UID = 0;
function hashId(pd, o) {
  let h = 0x811c9dc5;
  const mix = v => { h ^= v; h = Math.imul(h, 0x01000193); };
  mix(pd.w); mix(pd.h);
  const k = pd.key;
  for (let i = 0; i < k.length; i++) mix(k[i]);
  const s = [o.scale, o.mode, o.shading, o.smooth, o.iterations, o.guesses, o.guessOffset, o.seed].join('|');
  for (let i = 0; i < s.length; i++) mix(s.charCodeAt(i));
  return (h >>> 0).toString(36);
}
function hex(c) { return '#' + (c | 0x1000000).toString(16).slice(1); }

class SVGWriter {
  constructor(pd, opts = {}) {
    this.pd = pd;
    this.S = +opts.scale > 0 ? +opts.scale : DEFAULTS.scale;
    this.k10 = this.S * 10;
    this.sigma = Math.max(0, +(opts.shading ?? DEFAULTS.shading) || 0);
    this.id = opts.idPrefix || ('dpx' + (pd.key ? hashId(pd, opts) : ++FALLBACK_UID) + '-');
    // Each opaque fill is also stroked with its own colour, ~1.2 screen pixels
    // wide at any zoom (non-scaling stroke): a region painted later then covers
    // the anti-aliased seam of the one below, so no background shows through.
    this.strokeW = 1.2;
    this.cmd = '';
    this.T = new Float64Array(2); this.U = new Float64Array(2);
  }

  num(v) { return '' + Math.round(v * this.k10) / 10; }        // pixel units → SVG units
  _M(out, x, y) { out.push('M', this.num(x), ' ', this.num(y)); this.cmd = 'M'; }
  _L(out, x, y) { out.push(this.cmd === 'L' ? ' ' : 'L', this.num(x), ' ', this.num(y)); this.cmd = 'L'; }
  _Q(out, cx, cy, x, y) {
    out.push(this.cmd === 'Q' ? ' ' : 'Q', this.num(cx), ' ', this.num(cy), ' ', this.num(x), ' ', this.num(y));
    this.cmd = 'Q';
  }
  _Z(out) { out.push('Z'); this.cmd = ''; }

  render() {
    const pd = this.pd, S = this.S, out = [];
    const W = '' + Math.round(pd.w * S * 10) / 10, H = '' + Math.round(pd.h * S * 10) / 10;
    out.push('<svg xmlns="http://www.w3.org/2000/svg" width="', W, '" height="', H,
             '" viewBox="0 0 ', W, ' ', H, '">');
    const exact = !pd.opaque;            // holes must be cut when something is see-through
    const order = [];
    for (let r = 0; r < pd.R; r++) if (pd.rKind[r] !== 2) order.push(r);
    order.sort((a, b) => pd.rArea[b] - pd.rArea[a] || a - b);
    // Multi-colour regions get smooth shading. Regions whose blurs cannot
    // interact share one mask + filter (a batch), which keeps the number of
    // offscreen layers small; a batch is painted at its last member's slot,
    // with holes cut, so painter's order stays valid.
    const smooth = this.sigma > 0, batchOf = new Int32Array(pd.R).fill(-1), batches = [];
    if (smooth) {
      const D = Math.ceil(3 * this.sigma) + 2;
      for (let oi = 0; oi < order.length; oi++) {
        const r = order[oi];
        if (!pd.rMulti[r] || pd.rKind[r] !== 0) continue;
        const bb = this._bbox(r);
        let b = 0;
        for (; b < batches.length; b++) {
          let ok = true;
          for (const q of batches[b].boxes) {
            if (bb[0] - D <= q[2] && q[0] <= bb[2] + D && bb[1] - D <= q[3] && q[1] <= bb[3] + D) { ok = false; break; }
          }
          if (ok) break;
        }
        if (b === batches.length) batches.push({ members: [], boxes: [], last: -1 });
        batches[b].members.push(r); batches[b].boxes.push(bb); batches[b].last = oi;
        batchOf[r] = b;
      }
      if (batches.length) {
        out.push('<defs><filter id="', this.id, 'f" x="-10%" y="-10%" width="120%" height="120%"',
                 ' color-interpolation-filters="sRGB"><feGaussianBlur stdDeviation="',
                 '' + Math.round(this.sigma * S * 100) / 100,
                 '"/><feComponentTransfer><feFuncA type="linear" slope="40"/></feComponentTransfer></filter></defs>');
      }
    }
    out.push('<g stroke-width="', '' + this.strokeW, '" stroke-linejoin="round"', exact ? ' fill-rule="evenodd"' : '', '>');
    for (let oi = 0; oi < order.length; oi++) {
      const r = order[oi];
      if (batchOf[r] >= 0) {
        const B = batches[batchOf[r]];
        if (B.last === oi) this._smoothBatch(out, B, batchOf[r]);
        continue;
      }
      if (pd.rMulti[r]) { this._cellRegion(out, r); continue; }
      const p0 = pd.rPix[pd.rStart[r]];
      this._paint(out, pd.rgb[p0], pd.alpha[p0]);
      out.push(' d="');
      this._regionPath(out, r, exact);
      out.push('"/>');
    }
    out.push('</g></svg>');
    return out.join('');
  }

  // Opening of a <path> filled (and seam-stroked) with one colour.
  _paint(out, rgb, a) {
    const c = hex(rgb);
    if (a === 255) out.push('<path fill="', c, '" stroke="', c, '" vector-effect="non-scaling-stroke"');
    else out.push('<path fill="', c, '" fill-opacity="', '' + Math.round(a / 2.55) / 100, '"');
  }

  _regionPath(out, r, exact) {
    const pd = this.pd;
    for (const l of pd.rLoops[r]) {
      const L = pd.loops[l];
      if (exact || L.area > 0) this._loop(out, L);
    }
  }

  _bbox(r) {
    const pd = this.pd, w = pd.w, pix = pd.rPix;
    let x0 = w, y0 = pd.h, x1 = 0, y1 = 0;
    for (let p = pd.rStart[r]; p < pd.rStart[r + 1]; p++) {
      const i = pix[p], x = i % w, y = (i / w) | 0;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    return [x0, y0, x1, y1];
  }

  // Smooth multi-colour regions: the pixels are blurred and renormalised
  // (an edge-aware Gaussian interpolation, since only the region's own pixels
  // contribute) and masked by the region outline. The mask is stroked so the
  // region slightly overlaps whatever was painted before it (no seams).
  _smoothBatch(out, B, b) {
    const pd = this.pd, S = this.S, w = pd.w, pix = pd.rPix, rgb = pd.rgb, id = this.id + 'm' + b;
    out.push('<mask id="', id, '"><path fill="#fff" stroke="#fff" vector-effect="non-scaling-stroke" fill-rule="evenodd" d="');
    for (const r of B.members) this._regionPath(out, r, true);
    out.push('"/></mask><g mask="url(#', id, ')"><g filter="url(#', this.id, 'f)" stroke="none">');
    let x0 = w, y0 = pd.h, x1 = 0, y1 = 0;
    for (const q of B.boxes) {
      if (q[0] < x0) x0 = q[0]; if (q[1] < y0) y0 = q[1]; if (q[2] > x1) x1 = q[2]; if (q[3] > y1) y1 = q[3];
    }
    // Invisible margin: makes the filter and mask regions cover the outlines.
    out.push('<rect x="', '' + (x0 - 2) * S, '" y="', '' + (y0 - 2) * S, '" width="', '' + (x1 - x0 + 5) * S,
             '" height="', '' + (y1 - y0 + 5) * S, '" fill="none"/>');
    for (const r of B.members) {
      const a = pd.rStart[r], e = pd.rStart[r + 1];
      for (let p = a; p < e;) {
        const i = pix[p], c = rgb[i], y = (i / w) | 0;
        let q = p + 1;
        while (q < e && pix[q] === pix[q - 1] + 1 && rgb[pix[q]] === c && ((pix[q] / w) | 0) === y) q++;
        out.push('<rect x="', '' + (i % w) * S, '" y="', '' + y * S, '" width="', '' + (q - p) * S,
                 '" height="', '' + S, '" fill="', hex(c), '"/>');
        p = q;
      }
    }
    out.push('</g></g>');
  }

  // Multi-colour region, flat: every pixel cell is drawn, with its visible
  // edges replaced by the shared curve pieces.
  _cellRegion(out, r) {
    const pd = this.pd;
    for (let p = pd.rStart[r]; p < pd.rStart[r + 1]; p++) {
      const i = pd.rPix[p];
      this._paint(out, pd.rgb[i], pd.alpha[i]);
      out.push(' d="');
      this._cellPath(out, i);
      out.push('"/>');
    }
  }

  _cellPath(out, i) {
    const pd = this.pd, base = i * 8, n = pd.cellN[i], V = pd.cellV, EO = pd.edgeOf, qx = pd.qx, qy = pd.qy;
    this._M(out, qx[V[base]], qy[V[base]]);
    for (let k = 0; k < n; k++) {
      const s = base + k, e = EO[s];
      if (e < 0) {
        const v1 = V[base + (k + 1 === n ? 0 : k + 1)];
        this._L(out, qx[v1], qy[v1]);
        continue;
      }
      const c = pd.curves[pd.curveOf[e]], j = pd.idxOf[e], d = V[s] === c.nodes[j] ? 1 : -1;
      let a = d > 0 ? j : j + 1;
      if (c.closed && a === c.n) a = 0;
      this._run(out, c, a, 1, d);
    }
    this._Z(out);
  }

  // One boundary loop: a cyclic list of visible half-edges, emitted as runs
  // along the shared curves.
  _loop(out, L) {
    const pd = this.pd, slots = L.slots, m = slots.length;
    const C = pd.curves, EO = pd.edgeOf, CO = pd.curveOf, IO = pd.idxOf, V = pd.cellV;
    if (!this.hc || this.hc.length < m) {
      const cap = Math.max(m, 64);
      this.hc = new Int32Array(cap); this.hj = new Int32Array(cap); this.hd = new Int8Array(cap);
    }
    const hc = this.hc, hj = this.hj, hd = this.hd;
    for (let t = 0; t < m; t++) {
      const s = slots[t], e = EO[s], c = CO[e], j = IO[e];
      hc[t] = c; hj[t] = j; hd[t] = V[s] === C[c].nodes[j] ? 1 : -1;
    }
    const cont = (a, b) => {
      if (hc[a] !== hc[b] || hd[a] !== hd[b]) return false;
      const c = C[hc[a]];
      let j = hj[a] + hd[a];
      if (c.closed) { if (j < 0) j += c.n; else if (j >= c.n) j -= c.n; }
      return j === hj[b];
    };
    let s0 = -1;
    for (let t = 0; t < m; t++) if (!cont(t === 0 ? m - 1 : t - 1, t)) { s0 = t; break; }
    if (s0 < 0) { this._fullCurve(out, C[hc[0]]); return; }
    let t = s0, first = true;
    do {
      let cnt = 1;
      while (cnt < m) {
        const b = (t + cnt) % m;
        if (b === s0 || !cont((t + cnt - 1) % m, b)) break;
        cnt++;
      }
      const c = C[hc[t]], d = hd[t];
      let a0 = d > 0 ? hj[t] : hj[t] + 1;
      if (c.closed && a0 === c.n) a0 = 0;
      if (first) { nodePt(c, a0, this.T); this._M(out, this.T[0], this.T[1]); first = false; }
      this._run(out, c, a0, cnt, d);
      t = (t + cnt) % m;
    } while (t !== s0);
    this._Z(out);
  }

  // A loop made of one complete closed curve.
  _fullCurve(out, c) {
    if (c.border) {
      this._M(out, c.x[0], c.y[0]);
      this._run(out, c, 0, c.n, 1);
      this._Z(out);
      return;
    }
    const T = this.T, x = c.x, y = c.y;
    segA(c, 0, T);
    this._M(out, T[0], T[1]);
    for (let s = 0; s < c.n; s++) { segC(c, s, T); this._Q(out, x[s], y[s], T[0], T[1]); }
    this._Z(out);
  }

  // Emit `cnt` edges of curve c starting at control point a, direction d (±1).
  _run(out, c, a, cnt, d) {
    const x = c.x, y = c.y, n = c.n, closed = c.closed, M = c.nodes.length;
    if (c.border || (!closed && n === 1)) {
      // Straight pieces (image frame, or a single edge between two junctions);
      // collinear points are merged.
      let lx = x[a], ly = y[a], px = lx, py = ly, has = false;
      for (let t = 1; t <= cnt; t++) {
        let k = a + d * t;
        if (closed) { k %= M; if (k < 0) k += M; }
        const qx = x[k], qy = y[k];
        if (has && Math.abs((px - lx) * (qy - py) - (py - ly) * (qx - px)) > 1e-9) {
          this._L(out, px, py); lx = px; ly = py;
        }
        px = qx; py = qy; has = true;
      }
      if (has) this._L(out, px, py);
      return;
    }
    const T = this.T, U = this.U;
    const has = s => closed || (s >= 1 && s <= n - 1);
    const wrap = s => closed ? ((s % n) + n) % n : s;
    if (d > 0) {
      if (has(a)) { const s = wrap(a); segC(c, s, T); this._Q(out, (x[s] + T[0]) * 0.5, (y[s] + T[1]) * 0.5, T[0], T[1]); }
      for (let t = 1; t < cnt; t++) { const s = wrap(a + t); segC(c, s, T); this._Q(out, x[s], y[s], T[0], T[1]); }
      const b = a + cnt;
      if (has(b)) {
        const s = wrap(b); segA(c, s, T); nodePt(c, s, U);
        this._Q(out, (T[0] + x[s]) * 0.5, (T[1] + y[s]) * 0.5, U[0], U[1]);
      }
    } else {
      if (has(a)) { const s = wrap(a); segA(c, s, T); this._Q(out, (T[0] + x[s]) * 0.5, (T[1] + y[s]) * 0.5, T[0], T[1]); }
      for (let t = 1; t < cnt; t++) { const s = wrap(a - t); segA(c, s, T); this._Q(out, x[s], y[s], T[0], T[1]); }
      const b = a - cnt;
      if (has(b)) {
        const s = wrap(b); segC(c, s, T); nodePt(c, s, U);
        this._Q(out, (x[s] + T[0]) * 0.5, (y[s] + T[1]) * 0.5, U[0], U[1]);
      }
    }
  }
}

// ============================================================================
// Public entry point
// ============================================================================
function depixelize(image, opts = {}) {
  const o = {};
  for (const k in DEFAULTS) o[k] = opts[k] !== undefined && opts[k] !== null ? opts[k] : DEFAULTS[k];
  if (opts.idPrefix) o.idPrefix = opts.idPrefix;
  const pd = new PixelData(image.width, image.height, image.data, o);
  if (o.smooth !== false) pd.optimize(o);
  pd.finalize();
  return new SVGWriter(pd, o).render();
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { depixelize, PixelData, SVGWriter, DEFAULTS };
}
if (typeof window !== 'undefined') window.depixelize = depixelize;
