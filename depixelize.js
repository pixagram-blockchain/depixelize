// depixelize.js — Resolution-independent vectorization of pixel art.
// Cell-mode-only implementation of Kopf & Lischinski (2011).
// Single file, zero dependencies. Optimized with TypedArrays and packed
// integer keys throughout (no string nodes, no Map of Sets where avoidable).
//
// Usage (browser):   const svg = depixelize(imageData);
// Usage (Node):      const svg = depixelize({ data, width, height });
//
// Options:
//   scale        40     SVG output scale
//   smooth       true   run random-relaxation spline smoothing
//   iterations   20     outer smoothing iterations
//   guesses      20     random candidates per control point per iteration
//   guessOffset  0.05   max random displacement in pixel units

'use strict';

// ============================================================================
// Integer key packing
// ============================================================================
// Grid corner coordinates are quantized to quarter-pixel after cell
// deformation, so every corner has coordinates in N/4 where N is an integer.
// We store positions scaled ×4 (= Q4) so keys are plain ints.
//
//   corner key  =  (iy * gw) + ix      where ix,iy in [0..(w+1)*Q4]
//   pixel key   =   y * w + x          plain raster index
const Q4 = 4;

// ============================================================================
// PixelData — full pipeline. All graphs use integer node IDs; adjacency
// stored in TypedArrays where feasible.
// ============================================================================
class PixelData {
  constructor(w, h, rgba) {
    this.w = w; this.h = h;
    const N = w * h;

    // ---- pack RGB (0xRRGGBB) + YUV diffs ---------------------------------
    const rgb = this.rgb = new Uint32Array(N);
    const yuv = this.yuv = new Int16Array(N * 3);
    for (let i = 0, j = 0; i < N; i++, j += 4) {
      const r = rgba[j], g = rgba[j+1], b = rgba[j+2];
      rgb[i] = (r << 16) | (g << 8) | b;
      const y = (77 * r + 150 * g + 29 * b) >> 8;    // ~0.299r + 0.587g + 0.114b
      yuv[i*3    ] = y;
      yuv[i*3 + 1] = (126 * (b - y)) >> 8;           // ~0.492 * (b - y)
      yuv[i*3 + 2] = (224 * (r - y)) >> 8;           // ~0.877 * (r - y)
    }

    // ---- pixel similarity graph -----------------------------------------
    // 4 forward-direction bits per pixel in a Uint8Array.
    //   bit 0 : +x             bit 1 : +y
    //   bit 2 : +x+y (SE)      bit 3 : +x-y (NE)
    // Backward edges read from the neighbor's forward bit — no duplication.
    this.neigh = new Uint8Array(N);
    this._buildSimilarityGraph();
    this._removeDiagonals();

    // ---- corner grid ----------------------------------------------------
    this.gw = (w + 1) * Q4 + 1;
    this.gh = (h + 1) * Q4 + 1;
    this.grid = new FixedGraph(this.gw * this.gh);

    // Per-pixel corner sets. Fixed-stride Uint32Array: cornerBuf[i*8 .. +7],
    // counts in cornerCount[i]. Overflow (rare) spills to cornerOver Map.
    this.cornerBuf   = new Uint32Array(N * 8);
    this.cornerCount = new Uint8Array(N);
    this.cornerOver  = null;

    this._buildGridGraph();
    this._deformGrid();

    // ---- shapes, outlines, paths ---------------------------------------
    // Size the shared localIdxMap for this image.
    if (localIdxMap.length < this.gw * this.gh) {
      localIdxMap = new Int32Array(this.gw * this.gh);
    } else {
      localIdxMap.fill(0);
    }
    this._findShapes();
    this._findOutlines();
    this._assignPaths();
  }

  // ---- similarity test (hot) ------------------------------------------
  similar(ai, bi) {
    const y = this.yuv;
    const dy = y[ai*3  ] - y[bi*3  ]; if (dy >  48 || dy < -48) return false;
    const du = y[ai*3+1] - y[bi*3+1]; if (du >   7 || du <  -7) return false;
    const dv = y[ai*3+2] - y[bi*3+2]; if (dv >   6 || dv <  -6) return false;
    return true;
  }

  _buildSimilarityGraph() {
    const { w, h, neigh } = this;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (x + 1 < w && this.similar(i, i + 1))                  neigh[i] |= 1;
      if (y + 1 < h && this.similar(i, i + w))                  neigh[i] |= 2;
      if (x + 1 < w && y + 1 < h && this.similar(i, i + w + 1)) neigh[i] |= 4;
      if (x + 1 < w && y > 0     && this.similar(i, i - w + 1)) neigh[i] |= 8;
    }
  }

  // Enumerate all pixel neighbors of pixel i. Used in hot loops — avoid
  // generators here by writing into a caller-supplied scratch buffer.
  // Returns count of neighbors written.
  pxNeighbors(i, out) {
    const w = this.w, h = this.h, neigh = this.neigh;
    const x = i % w, y = (i / w) | 0;
    const n = neigh[i];
    let c = 0;
    if (n & 1) out[c++] = i + 1;
    if (n & 2) out[c++] = i + w;
    if (n & 4) out[c++] = i + w + 1;
    if (n & 8) out[c++] = i - w + 1;
    if (x > 0     && (neigh[i - 1]     & 1)) out[c++] = i - 1;
    if (y > 0     && (neigh[i - w]     & 2)) out[c++] = i - w;
    if (x > 0 && y > 0     && (neigh[i - w - 1] & 4)) out[c++] = i - w - 1;
    if (x > 0 && y + 1 < h && (neigh[i + w - 1] & 8)) out[c++] = i + w - 1;
    return c;
  }

  pxDegree(i) {
    const w = this.w, h = this.h, neigh = this.neigh;
    const x = i % w, y = (i / w) | 0;
    const n = neigh[i];
    let d = (n & 1) + ((n >> 1) & 1) + ((n >> 2) & 1) + ((n >> 3) & 1);
    if (x > 0     &&     (neigh[i - 1]     & 1)) d++;
    if (y > 0     &&     (neigh[i - w]     & 2)) d++;
    if (x > 0 && y > 0     && (neigh[i - w - 1] & 4)) d++;
    if (x > 0 && y + 1 < h && (neigh[i + w - 1] & 8)) d++;
    return d;
  }

  _rmPxEdge(a, b) {
    const w = this.w;
    const ax = a % w, ay = (a / w) | 0;
    const bx = b % w, by = (b / w) | 0;
    const dx = bx - ax, dy = by - ay;
    if      (dx ===  1 && dy ===  0) this.neigh[a] &= ~1;
    else if (dx ===  0 && dy ===  1) this.neigh[a] &= ~2;
    else if (dx ===  1 && dy ===  1) this.neigh[a] &= ~4;
    else if (dx ===  1 && dy === -1) this.neigh[a] &= ~8;
    else if (dx === -1 && dy ===  0) this.neigh[b] &= ~1;
    else if (dx ===  0 && dy === -1) this.neigh[b] &= ~2;
    else if (dx === -1 && dy === -1) this.neigh[b] &= ~4;
    else if (dx === -1 && dy ===  1) this.neigh[b] &= ~8;
  }

  // ==========================================================================
  // Diagonal removal + heuristics.
  // ==========================================================================
  _removeDiagonals() {
    const { w, h, neigh } = this;
    const ambiguous = [];
    for (let y = 0; y < h - 1; y++) for (let x = 0; x < w - 1; x++) {
      const tl = y * w + x, tr = tl + 1, bl = tl + w, br = bl + 1;
      const eTL_TR = (neigh[tl] & 1) !== 0;
      const eBL_BR = (neigh[bl] & 1) !== 0;
      const eTL_BL = (neigh[tl] & 2) !== 0;
      const eTR_BR = (neigh[tr] & 2) !== 0;
      const eTL_BR = (neigh[tl] & 4) !== 0;
      const eBL_TR = (neigh[bl] & 8) !== 0;
      if (!eTL_BR || !eBL_TR) continue;
      const card = (eTL_TR?1:0)+(eBL_BR?1:0)+(eTL_BL?1:0)+(eTR_BR?1:0);
      if (card === 4) {
        neigh[tl] &= ~4;
        neigh[bl] &= ~8;
      } else if (card === 0) {
        ambiguous.push(tl, br, bl, tr);   // flat: a1,b1,a2,b2
      }
    }
    if (ambiguous.length) this._applyHeuristics(ambiguous);
  }

  _hCurve(a, b, seenScratch) {
    seenScratch.fill(0);
    seenScratch[a] = 1; seenScratch[b] = 1;
    const stack = [a, b];
    let count = 2;
    const nbs = new Int32Array(8);
    while (stack.length) {
      const n = stack.pop();
      if (this.pxDegree(n) !== 2) continue;
      const nc = this.pxNeighbors(n, nbs);
      for (let k = 0; k < nc; k++) {
        const m = nbs[k];
        if (!seenScratch[m]) { seenScratch[m] = 1; count++; stack.push(m); }
      }
    }
    return count;
  }
  _hSparse(a, b, seenScratch) {
    const w = this.w;
    const ax = a % w, ay = (a / w) | 0;
    const bx = b % w, by = (b / w) | 0;
    const ox = 3 - Math.min(ax, bx), oy = 3 - Math.min(ay, by);
    seenScratch.fill(0);
    seenScratch[a] = 1; seenScratch[b] = 1;
    const stack = [a, b];
    let count = 2;
    const nbs = new Int32Array(8);
    while (stack.length) {
      const n = stack.pop();
      const nc = this.pxNeighbors(n, nbs);
      for (let k = 0; k < nc; k++) {
        const m = nbs[k];
        if (seenScratch[m]) continue;
        const mx = m % w, my = (m / w) | 0;
        const px = mx + ox, py = my + oy;
        if (px >= 0 && px < 8 && py >= 0 && py < 8) {
          seenScratch[m] = 1; count++; stack.push(m);
        }
      }
    }
    return -count;
  }
  _hIsland(a, b) {
    return (this.pxDegree(a) === 1 || this.pxDegree(b) === 1) ? 5 : 0;
  }
  _applyHeuristics(flat) {
    const seen = new Uint8Array(this.w * this.h);
    for (let k = 0; k < flat.length; k += 4) {
      const a1 = flat[k], b1 = flat[k+1], a2 = flat[k+2], b2 = flat[k+3];
      const w1 = this._hCurve(a1, b1, seen) + this._hSparse(a1, b1, seen) + this._hIsland(a1, b1);
      const w2 = this._hCurve(a2, b2, seen) + this._hSparse(a2, b2, seen) + this._hIsland(a2, b2);
      if (w1 <= w2) this._rmPxEdge(a1, b1);
      if (w2 <= w1) this._rmPxEdge(a2, b2);
    }
  }

  // ==========================================================================
  // Grid graph on the quarter-pixel lattice.
  // ==========================================================================
  _buildGridGraph() {
    const { w, h, gw } = this;
    const grid = this.grid;
    // Fast direct writes: during build every edge is unique, skip hasEdge.
    const exists = grid.exists, adj = grid.adj, deg = grid.deg;
    const GW4 = gw * Q4;    // row stride in corner keys
    for (let y = 0; y <= h; y++) {
      const rowBase = y * GW4;
      for (let x = 0; x <= w; x++) {
        const k = rowBase + x * Q4;
        exists[k] = 1;
        if (x < w) {
          const r = rowBase + (x + 1) * Q4;
          adj[k * MAX_DEG + deg[k]++] = r;
          adj[r * MAX_DEG + deg[r]++] = k;
        }
        if (y < h) {
          const d = rowBase + GW4 + x * Q4;
          adj[k * MAX_DEG + deg[k]++] = d;
          adj[d * MAX_DEG + deg[d]++] = k;
        }
      }
    }
    // Initial 4 corners per pixel — direct buffer writes.
    const cBuf = this.cornerBuf, cCnt = this.cornerCount;
    for (let y = 0; y < h; y++) {
      const top = y * Q4 * gw;
      const bot = (y + 1) * Q4 * gw;
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const leftQ  = x * Q4;
        const rightQ = (x + 1) * Q4;
        const base = i * 8;
        cBuf[base    ] = top + leftQ;
        cBuf[base + 1] = top + rightQ;
        cBuf[base + 2] = bot + leftQ;
        cBuf[base + 3] = bot + rightQ;
        cCnt[i] = 4;
      }
    }
  }

  // Write corner keys of pixel pi into out[]; return count.
  cornersArray(pi, out) {
    if (this.cornerOver) {
      const s = this.cornerOver.get(pi);
      if (s) { let i = 0; for (const k of s) out[i++] = k; return i; }
    }
    const base = pi * 8, n = this.cornerCount[pi];
    for (let j = 0; j < n; j++) out[j] = this.cornerBuf[base + j];
    return n;
  }

  // ==========================================================================
  // Cell deformation.
  // ==========================================================================
  _deformGrid() {
    const { w, h, neigh } = this;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const n = neigh[i];
      if (n & 4) this._deformCell(x, y, x + 1, y + 1);
      if (n & 8) this._deformCell(x, y, x + 1, y - 1);
      if (x > 0 && y > 0     && (neigh[i - w - 1] & 4)) this._deformCell(x, y, x - 1, y - 1);
      if (x > 0 && y + 1 < h && (neigh[i + w - 1] & 8)) this._deformCell(x, y, x - 1, y + 1);
    }
    // Collapse valence-2 corner nodes. Preserve image corners.
    const gw = this.gw;
    const keep0 = 0,
          keep1 = w * Q4,
          keep2 = (h * Q4) * gw,
          keep3 = keep2 + keep1;
    const grid = this.grid;
    const nbs = new Uint32Array(MAX_DEG);
    // Walk the exists mask directly — much faster than a generator.
    const exists = grid.exists, deg = grid.deg;
    const cap = grid.capacity;
    // Two-phase: collect nodes to collapse, then process. Collapsing can
    // cascade, so we iterate to a fixed point.
    const removals = [];
    for (let node = 0; node < cap; node++) {
      if (!exists[node]) continue;
      if (node === keep0 || node === keep1 || node === keep2 || node === keep3) continue;
      const d = deg[node];
      if (d <= 2) {
        if (d === 2) {
          grid.neighborsInto(node, nbs);
          grid.addEdge(nbs[0], nbs[1]);
        }
        removals.push(node);
      }
    }
    for (let k = 0; k < removals.length; k++) grid.removeNode(removals[k]);
    // Drop stale corner refs from pixel corner sets.
    for (let i = 0; i < w * h; i++) {
      if (this.cornerOver && this.cornerOver.has(i)) {
        const s = this.cornerOver.get(i);
        for (const c of s) if (!grid.hasNode(c)) s.delete(c);
        continue;
      }
      const base = i * 8; let cnt = this.cornerCount[i];
      for (let j = 0; j < cnt; ) {
        if (!grid.hasNode(this.cornerBuf[base + j])) {
          this.cornerBuf[base + j] = this.cornerBuf[base + cnt - 1];
          cnt--;
        } else j++;
      }
      this.cornerCount[i] = cnt;
    }
  }

  // Chamfer the shared corner between pixel (x,y) and diagonal neighbor
  // (nx,ny) in response to a blocking cardinal. Fully inlined for speed.
  _deformCell(x, y, nx, ny) {
    const { w, h, rgb, gw, cornerBuf, cornerCount } = this;
    const grid = this.grid;
    const gAdj = grid.adj, gDeg = grid.deg, gExi = grid.exists;

    const px = Math.max(x, nx), py = Math.max(y, ny);
    const ox = nx - x, oy = ny - y;
    const Q2 = Q4 >> 1, Q1 = Q4 >> 2;
    const pxQ = px * Q4, pyQ = py * Q4;
    const pixnode = pyQ * gw + pxQ;
    const me = y * w + x;
    const meColor = rgb[me];

    // --- horizontal cardinal at (nx, y) ------------------------------
    if (nx >= 0 && nx < w) {
      const adj = y * w + nx;
      if (rgb[adj] !== meColor) {
        const pn  = (pyQ - oy * Q4) * gw + pxQ;
        const mpn = (pyQ - oy * Q2) * gw + pxQ;
        const npn = (pyQ - oy * Q1) * gw + (pxQ + ox * Q1);
        // corner ops on `adj`
        if (this.cornerOver && this.cornerOver.has(adj)) {
          const s = this.cornerOver.get(adj);
          s.delete(pixnode); s.add(npn);
        } else {
          const base = adj * 8; let cnt = cornerCount[adj];
          for (let j = 0; j < cnt; j++) if (cornerBuf[base + j] === pixnode) {
            cornerBuf[base + j] = cornerBuf[base + cnt - 1];
            cnt--; break;
          }
          // add npn if not present
          let has = false;
          for (let j = 0; j < cnt; j++) if (cornerBuf[base + j] === npn) { has = true; break; }
          if (!has) {
            if (cnt < 8) { cornerBuf[base + cnt++] = npn; }
            else { this._spillAdd(adj, npn); }
          }
          cornerCount[adj] = cnt;
        }
        // corner ops on `me`: add npn if not present
        if (this.cornerOver && this.cornerOver.has(me)) {
          this.cornerOver.get(me).add(npn);
        } else {
          const base = me * 8; let cnt = cornerCount[me];
          let has = false;
          for (let j = 0; j < cnt; j++) if (cornerBuf[base + j] === npn) { has = true; break; }
          if (!has) {
            if (cnt < 8) { cornerBuf[base + cnt++] = npn; }
            else { this._spillAdd(me, npn); }
          }
          cornerCount[me] = cnt;
        }
        // edge ops
        this._deformEdgeInlined(pixnode, pn, mpn, npn);
      }
    }

    // --- vertical cardinal at (x, ny) --------------------------------
    if (ny >= 0 && ny < h) {
      const adj = ny * w + x;
      if (rgb[adj] !== meColor) {
        const pn  = pyQ * gw + (pxQ - ox * Q4);
        const mpn = pyQ * gw + (pxQ - ox * Q2);
        const npn = (pyQ + oy * Q1) * gw + (pxQ - ox * Q1);
        if (this.cornerOver && this.cornerOver.has(adj)) {
          const s = this.cornerOver.get(adj);
          s.delete(pixnode); s.add(npn);
        } else {
          const base = adj * 8; let cnt = cornerCount[adj];
          for (let j = 0; j < cnt; j++) if (cornerBuf[base + j] === pixnode) {
            cornerBuf[base + j] = cornerBuf[base + cnt - 1];
            cnt--; break;
          }
          let has = false;
          for (let j = 0; j < cnt; j++) if (cornerBuf[base + j] === npn) { has = true; break; }
          if (!has) {
            if (cnt < 8) { cornerBuf[base + cnt++] = npn; }
            else { this._spillAdd(adj, npn); }
          }
          cornerCount[adj] = cnt;
        }
        if (this.cornerOver && this.cornerOver.has(me)) {
          this.cornerOver.get(me).add(npn);
        } else {
          const base = me * 8; let cnt = cornerCount[me];
          let has = false;
          for (let j = 0; j < cnt; j++) if (cornerBuf[base + j] === npn) { has = true; break; }
          if (!has) {
            if (cnt < 8) { cornerBuf[base + cnt++] = npn; }
            else { this._spillAdd(me, npn); }
          }
          cornerCount[me] = cnt;
        }
        this._deformEdgeInlined(pixnode, pn, mpn, npn);
      }
    }
  }

  // Fallback when a pixel exceeds 8 corners — spill to a Map<Set>.
  _spillAdd(pi, k) {
    if (!this.cornerOver) this.cornerOver = new Map();
    let s = this.cornerOver.get(pi);
    if (!s) {
      s = new Set();
      const base = pi * 8, n = this.cornerCount[pi];
      for (let j = 0; j < n; j++) s.add(this.cornerBuf[base + j]);
      this.cornerOver.set(pi, s);
    }
    s.add(k);
  }

  _deformEdgeInlined(pixnode, pn, mpn, npn) {
    const grid = this.grid;
    if (grid.exists[mpn]) {
      grid.removeEdge(mpn, pixnode);
    } else {
      grid.removeEdge(pn, pixnode);
      grid.addEdge(pn, mpn);
    }
    grid.addEdge(mpn, npn);
    grid.addEdge(npn, pixnode);
  }

  // ==========================================================================
  // Shapes (connected same-color pixels).
  // ==========================================================================
  _findShapes() {
    const N = this.w * this.h;
    const comp = new Int32Array(N); comp.fill(-1);
    this.shapes = [];
    const stack = new Int32Array(N);
    const nbs   = new Int32Array(8);
    for (let i = 0; i < N; i++) {
      if (comp[i] !== -1) continue;
      const id = this.shapes.length;
      let sp = 0; stack[sp++] = i; comp[i] = id;
      const pixels = [];
      while (sp) {
        const n = stack[--sp];
        pixels.push(n);
        const nc = this.pxNeighbors(n, nbs);
        for (let k = 0; k < nc; k++) {
          const m = nbs[k];
          if (comp[m] === -1) { comp[m] = id; stack[sp++] = m; }
        }
      }
      this.shapes.push({ id, color: this.rgb[pixels[0]], pixels, outer: null, holes: [] });
    }
    this.pixelShape = comp;
  }

  // ==========================================================================
  // Outline graph = grid minus intra-shape edges.
  // Optimization: build outline by copying grid then removing intra-shape
  // boundary edges. Inner loop uses a small Uint8 lookup keyed by corner
  // value (cap-sized; reset via bumping a version).
  // ==========================================================================
  _findOutlines() {
    this.outline = this.grid.copy();
    const N = this.w * this.h;
    const outline = this.outline;
    const nbs = new Int32Array(8);
    const ci  = new Uint32Array(16);
    // Lazy-initialized shared marker sized to grid capacity.
    if (!this._cornerMark || this._cornerMark.length < outline.capacity) {
      this._cornerMark = new Uint32Array(outline.capacity);
    }
    const mark = this._cornerMark;
    let mver = this._cornerVer | 0;

    for (let i = 0; i < N; i++) {
      const ncount = this.pxNeighbors(i, nbs);
      if (ncount === 0) continue;
      // Mark corners of i with a fresh version.
      mver++;
      const ciCount = this.cornersArray(i, ci);
      for (let p = 0; p < ciCount; p++) mark[ci[p]] = mver;

      for (let k = 0; k < ncount; k++) {
        const j = nbs[k];
        if (j < i) continue;
        // Walk corners of j, find the (up to 2) that are also marked.
        // Using the fast Uint32 lookup path directly avoids calling cornersArray(j).
        let a = -1, b = -1;
        if (this.cornerOver && this.cornerOver.has(j)) {
          for (const c of this.cornerOver.get(j)) {
            if (mark[c] === mver) {
              if (a === -1) a = c; else { b = c; break; }
            }
          }
        } else {
          const base = j * 8, cnt = this.cornerCount[j];
          for (let q = 0; q < cnt; q++) {
            const c = this.cornerBuf[base + q];
            if (mark[c] === mver) {
              if (a === -1) a = c; else { b = c; break; }
            }
          }
        }
        if (b !== -1 && outline.hasEdge(a, b)) outline.removeEdge(a, b);
      }
    }
    this._cornerVer = mver;
  }

  // ==========================================================================
  // Per-shape boundary paths + spline fit. Each shape gets its induced
  // subgraph on the outline graph; components become closed curves.
  // Optimization: instead of Set<cornerKey> and Map<cornerKey, array[]> per
  // shape, we use a global Uint8Array mask + per-corner neighbor lists
  // reused across shapes. The induced-subgraph calculation becomes a couple
  // of tight typed-array loops.
  // ==========================================================================
  _assignPaths() {
    this.paths = [];
    const pathIndex = new Map();
    const gw = this.gw;
    const outline = this.outline;
    const cap = outline.capacity;
    const cBuf = new Uint32Array(16);
    // Persistent per-corner mask of "is in current shape's corner set"
    // Bumped version per shape lets us avoid clearing between shapes.
    const inShape = new Uint32Array(cap);
    let shapeVer = 0;

    // Scratch: flat representation of this shape's corner list + induced
    // adjacency as per-corner arrays. Grow on demand.
    let shapeCorners = new Uint32Array(128);
    let inducedAdj = new Uint32Array(128 * MAX_DEG);
    let inducedDeg = new Uint8Array(128);

    // BFS scratch
    const visited = new Uint32Array(cap);      // 0 if unvisited in this shape, shapeVer if visited
    let visitVer = 0;
    const stack = new Int32Array(cap);

    for (const shape of this.shapes) {
      // Render threshold: shapes below MIN_SMOOTH_PIXELS are painted only by
      // the NN backdrop in the SVG, so we don't need paths for them. Skip.
      if (shape.pixels.length < 4) continue;
      shapeVer++;
      // Collect on-outline corners for this shape into shapeCorners[].
      let cn = 0;
      for (let pk = 0; pk < shape.pixels.length; pk++) {
        const pi = shape.pixels[pk];
        const k = this.cornersArray(pi, cBuf);
        for (let j = 0; j < k; j++) {
          const c = cBuf[j];
          if (outline.exists[c] && inShape[c] !== shapeVer) {
            inShape[c] = shapeVer;
            if (cn >= shapeCorners.length) {
              const nb = new Uint32Array(shapeCorners.length * 2);
              nb.set(shapeCorners); shapeCorners = nb;
            }
            shapeCorners[cn++] = c;
          }
        }
      }
      if (cn < 3) continue;

      // Build induced adjacency. For each corner c in shapeCorners, filter
      // its outline neighbors to those also in inShape. Store neighbors as
      // LOCAL INDEX into shapeCorners[] (so subsequent ops work in a dense
      // 0..cn-1 index space rather than sparse corner keys).
      // Growth of induced arrays to match cn.
      if (cn > inducedDeg.length) {
        let cap2 = inducedDeg.length; while (cap2 < cn) cap2 *= 2;
        inducedAdj = new Uint32Array(cap2 * MAX_DEG);
        inducedDeg = new Uint8Array(cap2);
      } else {
        for (let i = 0; i < cn; i++) inducedDeg[i] = 0;
      }
      // Map corner key -> local index: reuse the `inShape` array by writing
      // the local index +1 there (we already know c is in shape because
      // inShape[c] === shapeVer; store (shapeVer << 16) | (localIdx+1)).
      // Simpler: use a separate Int32Array keyed by corner.
      // We'll just reuse the visited buffer — it's safe since shapeVer !== visitVer yet.
      // Local mapping: cornerKey -> local index in [1..cn]
      for (let i = 0; i < cn; i++) localIdxMap[shapeCorners[i]] = i + 1;

      const adjBuf = outline.adj, degBuf = outline.deg;
      let lexMin = shapeCorners[0], lexMinX = lexMin % gw, lexMinY = (lexMin / gw) | 0;
      for (let i = 1; i < cn; i++) {
        const c = shapeCorners[i];
        const cx = c % gw, cy = (c / gw) | 0;
        if (cx < lexMinX || (cx === lexMinX && cy < lexMinY)) { lexMin = c; lexMinX = cx; lexMinY = cy; }
      }
      const lexMinLocal = localIdxMap[lexMin] - 1;

      // Build induced adjacency in local indices.
      for (let i = 0; i < cn; i++) {
        const c = shapeCorners[i];
        const base = c * MAX_DEG, d = degBuf[c];
        const iBase = i * MAX_DEG;
        let ld = 0;
        for (let k = 0; k < d; k++) {
          const m = adjBuf[base + k];
          const li = localIdxMap[m];
          if (li !== 0) {
            // localIdxMap only holds current shape's corners (we wrote it
            // above; other shapes' writes are stale but collide — we have
            // to validate via inShape[]).
            if (inShape[m] === shapeVer) {
              inducedAdj[iBase + ld++] = li - 1;
            }
          }
        }
        inducedDeg[i] = ld;
      }

      // Clear the localIdxMap slots we wrote (must not leak to next shape).
      for (let i = 0; i < cn; i++) localIdxMap[shapeCorners[i]] = 0;

      // Connected components in induced graph, via BFS on local indices.
      visitVer++;
      for (let seed = 0; seed < cn; seed++) {
        if (visited[seed] === visitVer) continue;
        // Collect component; track whether it contains lexMinLocal.
        let sp = 0;
        stack[sp++] = seed;
        visited[seed] = visitVer;
        const compStart = sp === 1 ? 0 : 0;
        // Use a simple JS array for comp (rarely > 100 nodes, allocator OK).
        const comp = [seed];
        let hasLex = (seed === lexMinLocal);
        while (sp) {
          const n = stack[--sp];
          const base = n * MAX_DEG, d = inducedDeg[n];
          for (let k = 0; k < d; k++) {
            const m = inducedAdj[base + k];
            if (visited[m] !== visitVer) {
              visited[m] = visitVer;
              stack[sp++] = m;
              comp.push(m);
              if (m === lexMinLocal) hasLex = true;
            }
          }
        }
        if (comp.length < 3) continue;

        // Dedup paths by sorted-corner-key signature. Build the signature
        // and the key inline.
        const compKeys = new Uint32Array(comp.length);
        for (let i = 0; i < comp.length; i++) compKeys[i] = shapeCorners[comp[i]];
        compKeys.sort();
        let keyStr = '';
        for (let i = 0; i < compKeys.length; i++) keyStr += compKeys[i] + ',';

        let path = pathIndex.get(keyStr);
        if (!path) {
          const nodes = tracePathLocal(inducedAdj, inducedDeg, comp, shapeCorners, gw);
          if (nodes.length < 3) continue;
          path = new Path(nodes, gw);
          pathIndex.set(keyStr, path);
          this.paths.push(path);
        }
        if (hasLex) shape.outer = path; else shape.holes.push(path);
        path.shapes.add(shape);
      }
    }
  }
}

// Global scratch for corner-key → local-index mapping during _assignPaths.
// Sized once to the grid capacity lazily; reset between shapes via clearing.
let localIdxMap = new Int32Array(0);

// ============================================================================
// FixedGraph — dense fixed-slot adjacency for integer-keyed nodes with
// bounded degree (≤ MAX_DEG). Zero-alloc for all ops. Replaces the former
// Map<int, Set<int>> SparseGraph. Nodes are indexed in [0, capacity).
//
//   exists[n]              1 byte: 1 if node n is present
//   adj[n * MAX_DEG + s]   4 bytes: neighbor key, or EMPTY = 0xFFFFFFFF
//   deg[n]                 1 byte: current degree of n
//
// All three arrays are TypedArrays — memory-packed, cache-friendly.
// ============================================================================
const EMPTY   = 0xFFFFFFFF;
const MAX_DEG = 8;               // max degree observed on the quarter-pixel grid

class FixedGraph {
  constructor(capacity) {
    this.capacity = capacity;
    this.exists = new Uint8Array(capacity);
    this.deg    = new Uint8Array(capacity);
    this.adj    = new Uint32Array(capacity * MAX_DEG);
    this.adj.fill(EMPTY);
  }
  addNode(n)      { this.exists[n] = 1; }
  hasNode(n)      { return this.exists[n] === 1; }
  removeNode(n) {
    if (!this.exists[n]) return;
    const base = n * MAX_DEG, d = this.deg[n];
    for (let i = 0; i < d; i++) {
      const m = this.adj[base + i];
      if (m === EMPTY) continue;
      // Remove back-edge m → n
      const mb = m * MAX_DEG, md = this.deg[m];
      for (let j = 0; j < md; j++) if (this.adj[mb + j] === n) {
        this.adj[mb + j] = this.adj[mb + md - 1];
        this.adj[mb + md - 1] = EMPTY;
        this.deg[m] = md - 1;
        break;
      }
      this.adj[base + i] = EMPTY;
    }
    this.deg[n] = 0;
    this.exists[n] = 0;
  }
  addEdge(a, b) {
    this.exists[a] = 1; this.exists[b] = 1;
    if (!this._hasNeighbor(a, b)) {
      this.adj[a * MAX_DEG + this.deg[a]++] = b;
      this.adj[b * MAX_DEG + this.deg[b]++] = a;
    }
  }
  hasEdge(a, b) { return this.exists[a] === 1 && this._hasNeighbor(a, b); }
  removeEdge(a, b) {
    this._removeOne(a, b);
    this._removeOne(b, a);
  }
  _hasNeighbor(n, m) {
    const base = n * MAX_DEG, d = this.deg[n];
    for (let i = 0; i < d; i++) if (this.adj[base + i] === m) return true;
    return false;
  }
  _removeOne(n, m) {
    const base = n * MAX_DEG, d = this.deg[n];
    for (let i = 0; i < d; i++) if (this.adj[base + i] === m) {
      this.adj[base + i] = this.adj[base + d - 1];
      this.adj[base + d - 1] = EMPTY;
      this.deg[n] = d - 1;
      return;
    }
  }
  degreeOf(n)   { return this.deg[n]; }
  // Write neighbors of n into out[]; return count.
  neighborsInto(n, out) {
    const base = n * MAX_DEG, d = this.deg[n];
    for (let i = 0; i < d; i++) out[i] = this.adj[base + i];
    return d;
  }
  // Iterate with a callback — no allocation.
  forEachNeighbor(n, fn) {
    const base = n * MAX_DEG, d = this.deg[n];
    for (let i = 0; i < d; i++) fn(this.adj[base + i]);
  }
  // Iterate all existing nodes into an out array; return count.
  // Typically called once per pass in non-hot code.
  *nodes() {
    const e = this.exists;
    for (let i = 0; i < this.capacity; i++) if (e[i]) yield i;
  }
  copy() {
    const g = new FixedGraph(this.capacity);
    g.exists.set(this.exists);
    g.deg.set(this.deg);
    g.adj.set(this.adj);
    return g;
  }
}

// ============================================================================
// Planar-face trace (local-index variant for _assignPaths).
// inducedAdj/Deg are stride-MAX_DEG arrays indexed by local node index
// (0..cn-1). shapeCorners maps local index → corner key (needed for the
// lex-min start selection and the final coordinate conversion).
// Returns an array of corner keys (not local indices).
// ============================================================================
function tracePathLocal(inducedAdj, inducedDeg, comp, shapeCorners, gw) {
  // Lex-min start among this component's LOCAL indices.
  let startLocal = comp[0];
  let startKey = shapeCorners[startLocal];
  let startX = startKey % gw, startY = (startKey / gw) | 0;
  for (let i = 1; i < comp.length; i++) {
    const lc = comp[i];
    const k = shapeCorners[lc];
    const cx = k % gw, cy = (k / gw) | 0;
    if (cx < startX || (cx === startX && cy < startY)) {
      startLocal = lc; startKey = k; startX = cx; startY = cy;
    }
  }
  const d0 = inducedDeg[startLocal];
  if (d0 === 0) return [startKey];
  const base0 = startLocal * MAX_DEG;
  // Pick neighbor with smallest slope.
  let firstLocal = inducedAdj[base0];
  {
    let firstKey = shapeCorners[firstLocal];
    let bestSlope = slopeIntKeys(startX, startY, firstKey, gw);
    for (let i = 1; i < d0; i++) {
      const lc = inducedAdj[base0 + i];
      const k = shapeCorners[lc];
      const s = slopeIntKeys(startX, startY, k, gw);
      if (s < bestSlope) { bestSlope = s; firstLocal = lc; firstKey = k; }
    }
  }
  const out = [startKey];
  let prev = startLocal, curr = firstLocal;
  const maxSteps = comp.length * 4 + 10;
  for (let step = 0; step < maxSteps; step++) {
    out.push(shapeCorners[curr]);
    const d = inducedDeg[curr];
    if (d === 0) break;
    const next = pickMostCWLocal(prev, curr, inducedAdj, d, shapeCorners, gw);
    if (next === -1) break;
    if (curr === startLocal && next === firstLocal) break;
    prev = curr; curr = next;
  }
  if (out.length > 1 && out[out.length - 1] === startKey) out.pop();
  return out;
}
function slopeIntKeys(x0, y0, nKey, gw) {
  const x1 = nKey % gw, y1 = (nKey / gw) | 0;
  const dx = x1 - x0, dy = y1 - y0;
  return dx === 0 ? dy * 1e14 : dy / dx;
}
const PI2 = Math.PI * 2;
function pickMostCWLocal(prevLocal, currLocal, inducedAdj, d, shapeCorners, gw) {
  const currKey = shapeCorners[currLocal], prevKey = shapeCorners[prevLocal];
  const cx = currKey % gw, cy = (currKey / gw) | 0;
  const px = prevKey % gw, py = (prevKey / gw) | 0;
  const inAng = Math.atan2(py - cy, px - cx);
  const base = currLocal * MAX_DEG;
  let bestLocal = -1, bestTurn = Infinity;
  for (let k = 0; k < d; k++) {
    const nLocal = inducedAdj[base + k];
    if (nLocal === prevLocal && d > 1) continue;
    const nKey = shapeCorners[nLocal];
    const nx = nKey % gw, ny = (nKey / gw) | 0;
    let turn = inAng - Math.atan2(ny - cy, nx - cx);
    while (turn <    0) turn += PI2;
    while (turn >= PI2) turn -= PI2;
    if (turn < bestTurn) { bestTurn = turn; bestLocal = nLocal; }
  }
  if (bestLocal === -1 && d === 1) bestLocal = inducedAdj[base];
  return bestLocal;
}

// ============================================================================
// Path — boundary polyline + its quadratic B-spline fit.
// ============================================================================
class Path {
  constructor(nodes, gw) {
    this.n = nodes.length;
    const p = this.pts = new Float32Array(this.n * 2);
    for (let i = 0; i < this.n; i++) {
      const k = nodes[i];
      p[i*2]     = (k % gw) / Q4;
      p[i*2 + 1] = ((k / gw) | 0) / Q4;
    }
    this.shapes = new Set();
    this.spline = null;
    this.smooth = null;
  }
  fitSpline() {
    if (this.n < 3) return;
    this.spline = ClosedBSpline.fit(this.pts, this.n, 2);
  }
}

// ============================================================================
// Closed quadratic B-spline (Float64Array-backed).
// ============================================================================
class ClosedBSpline {
  constructor(knots, cps, degree, unwrap) {
    this.knots = knots; this.cps = cps; this.degree = degree;
    this.n = cps.length / 2; this.unwrap = unwrap;
    this._d1 = null;
  }
  static fit(pts, nPts, degree) {
    const n = nPts + degree, m = n + degree;
    const knots = new Float64Array(m + 1);
    for (let i = 0; i <= m; i++) knots[i] = i / m;
    const cps = new Float64Array(n * 2);
    for (let i = 0; i < nPts; i++) { cps[i*2] = pts[i*2]; cps[i*2+1] = pts[i*2+1]; }
    for (let i = 0; i < degree; i++) {
      cps[(nPts + i)*2]     = pts[i*2];
      cps[(nPts + i)*2 + 1] = pts[i*2 + 1];
    }
    return new ClosedBSpline(knots, cps, degree, nPts);
  }
  copy() {
    return new ClosedBSpline(
      new Float64Array(this.knots),
      new Float64Array(this.cps),
      this.degree, this.unwrap);
  }
  cpSet(i, x, y) {
    i = ((i % this.unwrap) + this.unwrap) % this.unwrap;
    this.cps[i*2] = x; this.cps[i*2+1] = y;
    if (i < this.degree) {
      const j = i + this.unwrap;
      this.cps[j*2] = x; this.cps[j*2+1] = y;
    }
    this._d1 = null;
  }
  // de Boor — writes into result[0..1]
  evalAt(u, result) {
    const p = this.degree, knots = this.knots, cps = this.cps;
    let k;
    if (u <= knots[p]) k = p;
    else if (u >= knots[this.n]) k = this.n - 1;
    else {
      let lo = p, hi = this.n;
      while (lo < hi - 1) { const mid = (lo + hi) >> 1; (u < knots[mid]) ? hi = mid : lo = mid; }
      k = lo;
    }
    // inlined small allocations: for p<=2 we only need 3 points
    const i0 = 2 * (k - p), i1 = i0 + 2, i2 = i0 + 4;
    let dx0 = cps[i0], dy0 = cps[i0+1];
    let dx1 = cps[i1], dy1 = cps[i1+1];
    let dx2 = p >= 2 ? cps[i2]   : 0;
    let dy2 = p >= 2 ? cps[i2+1] : 0;
    if (p === 2) {
      // r=1: update (1), (2)
      let a = (u - knots[k - 1]) / (knots[k + 1] - knots[k - 1]);
      dx1 = (1 - a) * dx0 + a * dx1; dy1 = (1 - a) * dy0 + a * dy1;
      a   = (u - knots[k    ]) / (knots[k + 2] - knots[k    ]);
      dx2 = (1 - a) * cps[i1] /* old */ + a * dx2; dy2 = (1 - a) * cps[i1+1] + a * dy2;
      // r=2: final
      a   = (u - knots[k    ]) / (knots[k + 1] - knots[k    ]);
      result[0] = (1 - a) * dx1 + a * dx2;
      result[1] = (1 - a) * dy1 + a * dy2;
      return;
    } else if (p === 1) {
      const a = (u - knots[k]) / (knots[k + 1] - knots[k]);
      result[0] = (1 - a) * dx0 + a * dx1;
      result[1] = (1 - a) * dy0 + a * dy1;
      return;
    } else if (p === 0) {
      result[0] = dx0; result[1] = dy0; return;
    }
    // Generic path (not hit for our use).
    const dX = new Float64Array(p + 1), dY = new Float64Array(p + 1);
    for (let i = 0; i <= p; i++) {
      dX[i] = cps[2*(k - p + i)]; dY[i] = cps[2*(k - p + i) + 1];
    }
    for (let r = 1; r <= p; r++) for (let i = p; i >= r; i--) {
      const j = k - p + i;
      const a = (u - knots[j]) / (knots[j + p - r + 1] - knots[j]);
      dX[i] = (1 - a) * dX[i-1] + a * dX[i];
      dY[i] = (1 - a) * dY[i-1] + a * dY[i];
    }
    result[0] = dX[p]; result[1] = dY[p];
  }
  derivative() {
    if (this._d1) return this._d1;
    const p = this.degree, n = this.n;
    const nc = new Float64Array((n - 1) * 2);
    for (let i = 0; i < n - 1; i++) {
      const c = p / (this.knots[i + 1 + p] - this.knots[i + 1]);
      nc[i*2]     = c * (this.cps[(i+1)*2]     - this.cps[i*2]);
      nc[i*2 + 1] = c * (this.cps[(i+1)*2 + 1] - this.cps[i*2 + 1]);
    }
    const nk = this.knots.slice(1, this.knots.length - 1);
    this._d1 = new ClosedBSpline(nk, nc, p - 1, this.unwrap);
    return this._d1;
  }
  // Write beziers into out[] as flat 6-tuples (sx,sy,cx,cy,ex,ey). Returns count.
  toBeziers(out) {
    const p = this.degree;
    const tmp = TMP2;
    this.evalAt(this.knots[p], tmp);
    let sx = tmp[0], sy = tmp[1];
    const count = this.n - p;
    for (let k = 0; k < count; k++) {
      const cx = this.cps[(k + 1) * 2];
      const cy = this.cps[(k + 1) * 2 + 1];
      this.evalAt(this.knots[p + 1 + k], tmp);
      const ex = tmp[0], ey = tmp[1];
      const base = k * 6;
      out[base  ] = sx; out[base+1] = sy;
      out[base+2] = cx; out[base+3] = cy;
      out[base+4] = ex; out[base+5] = ey;
      sx = ex; sy = ey;
    }
    return count;
  }
  reversed() {
    const p = this.degree, n = this.n;
    const nk = new Float64Array(this.knots.length);
    for (let i = 0; i < nk.length; i++) nk[i] = 1 - this.knots[nk.length - 1 - i];
    const nc = new Float64Array(this.cps.length);
    for (let i = 0; i < n; i++) {
      nc[i*2]     = this.cps[(n - 1 - i) * 2];
      nc[i*2 + 1] = this.cps[(n - 1 - i) * 2 + 1];
    }
    return new ClosedBSpline(nk, nc, p, this.unwrap);
  }
}
const TMP2 = new Float64Array(2);

// ============================================================================
// Spline smoother — random-relaxation. Hot loop operates on Float64Arrays
// directly with zero per-iteration allocation.
// ============================================================================
function smoothSpline(original, opts) {
  const iterations  = opts.iterations   ?? 20;
  const guesses     = opts.guesses      ?? 20;
  const guessOffset = opts.guessOffset  ?? 0.05;
  const N_INT = 20;

  const s = original.copy();
  const U = s.unwrap, p = s.degree, knots = s.knots;
  const d1 = new Float64Array(2), d2 = new Float64Array(2);
  const domLo = knots[p], domHi = knots[s.n];

  function curvature(u) {
    const D1 = s.derivative();
    const D2 = D1.derivative();
    D1.evalAt(u, d1); D2.evalAt(u, d2);
    const x1 = d1[0], y1 = d1[1], x2 = d2[0], y2 = d2[1];
    const num = x1 * y2 - y1 * x2;
    const q = x1 * x1 + y1 * y1;
    const den = q * Math.sqrt(q);
    return den === 0 ? 0 : Math.abs(num / den);
  }

  function energyAt(i) {
    const ox = original.cps[i*2], oy = original.cps[i*2 + 1];
    const cx = s.cps[i*2],        cy = s.cps[i*2 + 1];
    const dx = cx - ox, dy = cy - oy;
    const dsq = dx * dx + dy * dy;
    let E = dsq * dsq;
    for (let r = 0; r < p; r++) {
      let lo = knots[i + 1 + r], hi = knots[i + 2 + r];
      if (lo < domLo) lo = domLo; else if (lo > domHi) lo = domHi;
      if (hi < domLo) hi = domLo; else if (hi > domHi) hi = domHi;
      if (lo === hi) continue;
      const step = (hi - lo) / N_INT;
      let acc = 0.5 * (curvature(lo) + curvature(hi));
      for (let j = 1; j < N_INT; j++) acc += curvature(lo + j * step);
      E += acc * step;
    }
    return E;
  }

  for (let it = 0; it < iterations; it++) {
    for (let i = 0; i < U; i++) {
      const sx = s.cps[i*2], sy = s.cps[i*2 + 1];
      let bestE = energyAt(i), bx = sx, by = sy;
      for (let k = 0; k < guesses; k++) {
        const r = Math.random() * guessOffset;
        const th = Math.random() * PI2;
        const cx = sx + r * Math.cos(th), cy = sy + r * Math.sin(th);
        s.cpSet(i, cx, cy);
        const e = energyAt(i);
        if (e < bestE) { bestE = e; bx = cx; by = cy; }
      }
      s.cpSet(i, bx, by);
    }
  }
  return s;
}

// ============================================================================
// SVG writer. Assembles one big string from pre-stringified tokens and
// lookup tables. No per-pixel number-to-string coercion in the hot path.
// ============================================================================
class SVGWriter {
  constructor(pd, scale) {
    this.pd = pd; this.scale = scale;
    // Pre-render strings for all quarter-pixel-aligned coordinates.
    // After deformation every corner is a multiple of Q4; after smoothing
    // they drift slightly but still often land near quarter-pixels, so
    // this table hits often and saves toString() cost.
    const maxQ = Math.max(pd.w, pd.h) * Q4 + Q4;
    this.qStr = new Array(maxQ + 1);
    const S = scale;
    for (let i = 0; i <= maxQ; i++) {
      const v = (i / Q4) * S;
      this.qStr[i] = Number.isInteger(v) ? v.toString() : v.toFixed(1);
    }
    // Pre-render RGB strings per unique color.
    this.colorStr = new Map();
    for (let i = 0; i < pd.rgb.length; i++) {
      const c = pd.rgb[i];
      if (!this.colorStr.has(c)) {
        this.colorStr.set(c,
          'rgb(' + ((c >> 16) & 255) + ',' + ((c >> 8) & 255) + ',' + (c & 255) + ')');
      }
    }
    // Pre-rendered "scale" string used by the NN backdrop.
    this.scaleStr = String(S);
  }
  _col(c) { return this.colorStr.get(c); }
  _num(v) {
    // Hot path: quarter-pixel-aligned values hit the lookup table.
    const q = v * Q4;
    const qi = (q + 0.5) | 0;
    if (Math.abs(q - qi) < 1e-4 && qi >= 0 && qi < this.qStr.length) {
      return this.qStr[qi];
    }
    const scaled = v * this.scale;
    const rounded = Math.round(scaled * 10) / 10;
    return rounded.toString();
  }

  render() {
    const pd = this.pd, S = this.scale;
    const W = pd.w * S, H = pd.h * S;
    const out = [];
    out.push('<svg xmlns="http://www.w3.org/2000/svg" width="', W, '" height="', H,
             '" viewBox="0 0 ', W, ' ', H, '" shape-rendering="geometricPrecision">');

    // 1) Nearest-neighbor backdrop (every pixel as a square).
    this._emitBackdrop(out);

    // 2) Clipped smoothed silhouettes over big shapes.
    this._emitShapes(out);

    out.push('</svg>');
    return out.join('');
  }

  _emitBackdrop(out) {
    const pd = this.pd, S = this.scale, sStr = this.scaleStr;
    const xS = new Array(pd.w + 1);    // pre-stringified pixel x positions
    const yS = new Array(pd.h + 1);
    for (let x = 0; x <= pd.w; x++) xS[x] = (x * S).toString();
    for (let y = 0; y <= pd.h; y++) yS[y] = (y * S).toString();
    const rgb = pd.rgb;
    // Row-merging: contiguous same-color pixels become one rect. Shrinks
    // output size a lot on classic pixel art and photos with large flats.
    for (let y = 0; y < pd.h; y++) {
      let x = 0;
      while (x < pd.w) {
        const c = rgb[y * pd.w + x];
        let x2 = x + 1;
        while (x2 < pd.w && rgb[y * pd.w + x2] === c) x2++;
        const ww = (x2 - x) * S;
        out.push('<rect x="', xS[x], '" y="', yS[y],
                 '" width="', ww, '" height="', sStr,
                 '" fill="', this._col(c), '" shape-rendering="crispEdges"/>');
        x = x2;
      }
    }
  }

  _emitShapes(out) {
    const MIN_SMOOTH_PIXELS = 4;
    const pd = this.pd;
    const shapes = pd.shapes.slice().sort((a, b) => b.pixels.length - a.pixels.length);
    let bezBuf = new Float32Array(256 * 6);

    // Build clip-path definitions for shapes large enough to benefit.
    out.push('<defs>');
    const shapeId = new Map();
    for (let i = 0; i < shapes.length; i++) {
      const s = shapes[i];
      if (!s.outer || !s.outer.smooth || s.pixels.length < MIN_SMOOTH_PIXELS) continue;
      const id = 'c' + i;
      shapeId.set(s, id);
      const splines = [s.outer.smooth.reversed()];
      for (const h of s.holes) if (h.smooth) splines.push(h.smooth);
      out.push('<clipPath id="', id, '" clip-rule="evenodd"><path d="');
      bezBuf = this._appendSplines(out, splines, bezBuf);
      out.push('"/></clipPath>');
    }
    out.push('</defs>');

    // Emit each big shape: backdrop (avg color) + per-cell overlays.
    for (const s of shapes) {
      const id = shapeId.get(s);
      if (!id) continue;
      // Average color.
      let ar = 0, ag = 0, ab = 0;
      const rgb = pd.rgb;
      for (let k = 0; k < s.pixels.length; k++) {
        const c = rgb[s.pixels[k]];
        ar += (c >> 16) & 255; ag += (c >> 8) & 255; ab += c & 255;
      }
      const n = s.pixels.length;
      const avg = 'rgb(' + ((ar/n) | 0) + ',' + ((ag/n) | 0) + ',' + ((ab/n) | 0) + ')';
      const splines = [s.outer.smooth.reversed()];
      for (const h of s.holes) if (h.smooth) splines.push(h.smooth);
      out.push('<g clip-path="url(#', id, ')"><path fill="', avg, '" fill-rule="evenodd" d="');
      bezBuf = this._appendSplines(out, splines, bezBuf);
      out.push('"/>');
      // Per-cell overlays.
      for (let k = 0; k < s.pixels.length; k++) {
        const pi = s.pixels[k];
        const col = this._col(rgb[pi]);
        out.push('<path fill="', col, '" stroke="', col,
                 '" stroke-width="1.5" stroke-linejoin="round" d="');
        this._appendCellPolygon(out, pi);
        out.push('"/>');
      }
      out.push('</g>');
    }
  }

  _appendSplines(out, splines, bezBuf) {
    for (const spline of splines) {
      if (!spline) continue;
      const need = (spline.n - spline.degree) * 6;
      if (need > bezBuf.length) bezBuf = new Float32Array(need);
      const count = spline.toBeziers(bezBuf);
      if (!count) continue;
      out.push('M', this._num(bezBuf[0]), ',', this._num(bezBuf[1]));
      for (let k = 0; k < count; k++) {
        const o = k * 6;
        out.push('Q', this._num(bezBuf[o+2]), ',', this._num(bezBuf[o+3]),
                 ' ', this._num(bezBuf[o+4]), ',', this._num(bezBuf[o+5]));
      }
      out.push('Z');
    }
    return bezBuf;
  }

  _appendCellPolygon(out, pi) {
    const pd = this.pd;
    const buf = CELL_BUF, angBuf = CELL_ANG;
    let n = 0;
    let cx = 0, cy = 0;
    if (pd.cornerOver && pd.cornerOver.has(pi)) {
      for (const k of pd.cornerOver.get(pi)) {
        const vx = (k % pd.gw) / Q4, vy = ((k / pd.gw) | 0) / Q4;
        buf[n*2] = vx; buf[n*2 + 1] = vy; cx += vx; cy += vy; n++;
      }
    } else {
      const base = pi * 8, cnt = pd.cornerCount[pi];
      for (let j = 0; j < cnt; j++) {
        const k = pd.cornerBuf[base + j];
        const vx = (k % pd.gw) / Q4, vy = ((k / pd.gw) | 0) / Q4;
        buf[n*2] = vx; buf[n*2 + 1] = vy; cx += vx; cy += vy; n++;
      }
    }
    if (n < 3) return;
    cx /= n; cy /= n;
    // Angular sort around centroid (insertion sort on small n is fast).
    for (let i = 0; i < n; i++) {
      angBuf[i] = Math.atan2(buf[i*2 + 1] - cy, buf[i*2] - cx);
    }
    for (let i = 1; i < n; i++) {
      const aAng = angBuf[i], ax = buf[i*2], ay = buf[i*2 + 1];
      let j = i - 1;
      while (j >= 0 && angBuf[j] > aAng) {
        angBuf[j+1] = angBuf[j];
        buf[(j+1)*2] = buf[j*2]; buf[(j+1)*2 + 1] = buf[j*2 + 1];
        j--;
      }
      angBuf[j+1] = aAng; buf[(j+1)*2] = ax; buf[(j+1)*2 + 1] = ay;
    }
    out.push('M', this._num(buf[0]), ',', this._num(buf[1]));
    for (let i = 1; i < n; i++) out.push('L', this._num(buf[i*2]), ',', this._num(buf[i*2 + 1]));
    out.push('Z');
  }
}
// Scratch for cell polygon rendering — cells max 8 corners, rarely more.
const CELL_BUF = new Float32Array(32);
const CELL_ANG = new Float32Array(16);

// ============================================================================
// Public entry point
// ============================================================================
function depixelize(image, opts = {}) {
  const { scale = 40, smooth = true } = opts;
  const pd = new PixelData(image.width, image.height, image.data);
  if (smooth !== false) {
    for (const path of pd.paths) path.fitSpline();
    for (const path of pd.paths) {
      if (!path.spline) continue;
      if (path.shapes.size === 1) path.smooth = path.spline.copy();
      else                        path.smooth = smoothSpline(path.spline, opts);
    }
  }
  return new SVGWriter(pd, scale).render();
}

// ============================================================================
// Exports
// ============================================================================
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { depixelize, PixelData, SVGWriter, ClosedBSpline, smoothSpline };
}
if (typeof window !== 'undefined') window.depixelize = depixelize;
