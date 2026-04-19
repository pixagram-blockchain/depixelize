// depixelize.js — Resolution-independent vectorization of pixel art.
// Port of Kopf & Lischinski (2011). Single file, zero dependencies, ~900 lines.
//
// Usage (browser):
//   const svg = depixelize(imageData);            // ImageData -> SVG string
// Usage (Node):
//   const svg = depixelize({ data, width, height });   // data: Uint8Array/Uint8ClampedArray RGBA
//
// Options:
//   scale        40     SVG output scale
//   mode         auto   'shape' | 'cell' | 'auto' (auto picks by palette size)
//                       shape: smooth spline contours per connected region
//                              — best for classic pixel art (Mario, Invaders)
//                       cell:  per-pixel cell polygons under smooth clip silhouettes
//                              — preserves gradients and photo-like input
//   smooth       true   run spline smoothing (random relaxation)
//   iterations   20     outer smoothing iterations
//   guesses      20     random candidates per control point per iteration
//   guessOffset  0.05   max random displacement in pixel units

'use strict';

// ---------- tiny utilities ----------------------------------------------------
const key2 = (x, y) => x + ',' + y;                // node key for 2D coords
const parseKey = s => s.split(',').map(Number);    // "x,y" -> [x, y]

// ---------- Graph: adjacency Map with per-edge attribute Map -----------------
class Graph {
  constructor() {
    this.adj  = new Map();   // node -> Set(neighbor)
    this.attr = new Map();   // unordered-pair-key -> object
  }
  _ek(a, b) { return a < b ? a + '|' + b : b + '|' + a; }
  addNode(n)      { if (!this.adj.has(n)) this.adj.set(n, new Set()); }
  hasNode(n)      { return this.adj.has(n); }
  removeNode(n)   {
    const nb = this.adj.get(n); if (!nb) return;
    for (const m of nb) { this.adj.get(m).delete(n); this.attr.delete(this._ek(n, m)); }
    this.adj.delete(n);
  }
  addEdge(a, b, at = null) {
    this.addNode(a); this.addNode(b);
    this.adj.get(a).add(b); this.adj.get(b).add(a);
    if (at) this.attr.set(this._ek(a, b), at);
  }
  hasEdge(a, b)    { const s = this.adj.get(a); return !!s && s.has(b); }
  removeEdge(a, b) {
    const sa = this.adj.get(a), sb = this.adj.get(b);
    if (sa) sa.delete(b); if (sb) sb.delete(a);
    this.attr.delete(this._ek(a, b));
  }
  edgeAttr(a, b)   { return this.attr.get(this._ek(a, b)); }
  neighbors(n)     { return this.adj.get(n) || new Set(); }
  degree(n)        { const s = this.adj.get(n); return s ? s.size : 0; }
  nodes()          { return this.adj.keys(); }
  copy() {
    const g = new Graph();
    for (const [n, nb] of this.adj) g.adj.set(n, new Set(nb));
    for (const [k, v] of this.attr) g.attr.set(k, { ...v });
    return g;
  }
  // Connected components of this graph restricted to nodeSet (or all nodes).
  components(nodeSet) {
    const pool  = nodeSet ? new Set(nodeSet) : new Set(this.adj.keys());
    const comps = [];
    while (pool.size) {
      const start = pool.values().next().value;
      const comp  = new Set(); const stack = [start];
      while (stack.length) {
        const n = stack.pop();
        if (comp.has(n) || !pool.has(n)) continue;
        comp.add(n); pool.delete(n);
        for (const m of this.adj.get(n)) if (pool.has(m) && !comp.has(m)) stack.push(m);
      }
      comps.push(comp);
    }
    return comps;
  }
  // Induced subgraph: only keeps nodes in nodeSet and edges where both
  // endpoints are in nodeSet. Critical for shape-boundary tracing — within
  // a shape's induced subgraph, every boundary node has degree 2 (simple
  // cycle), so the greedy tracer works correctly.
  induced(nodeSet) {
    const g = new Graph();
    for (const n of nodeSet) if (this.adj.has(n)) g.addNode(n);
    for (const n of nodeSet) {
      const nb = this.adj.get(n); if (!nb) continue;
      for (const m of nb) if (nodeSet.has(m)) {
        g.adj.get(n).add(m);
        const k = this._ek(n, m);
        const a = this.attr.get(k); if (a) g.attr.set(k, a);
      }
    }
    return g;
  }
}

// ---------- Similarity & grid graph construction -----------------------------
class PixelGraph {
  // thresholds from paper (hqx): ΔY>48, ΔU>7, ΔV>6 on 0..255
  static Y_T = 48; static U_T = 7; static V_T = 6;

  constructor(w, h, rgba) {
    this.w = w; this.h = h;
    // Pack RGB into Uint8Array, one row after another, stride = 3*w.
    this.rgb = new Uint8Array(w * h * 3);
    for (let i = 0, j = 0; i < w * h; i++, j += 4) {
      this.rgb[3*i    ] = rgba[j    ];
      this.rgb[3*i + 1] = rgba[j + 1];
      this.rgb[3*i + 2] = rgba[j + 2];
    }
    // Precompute YUV for every pixel as Float32 (cheap memory, skip recompute).
    this.yuv = new Float32Array(w * h * 3);
    for (let i = 0; i < w * h; i++) {
      const r = this.rgb[3*i], g = this.rgb[3*i+1], b = this.rgb[3*i+2];
      const y = 0.299*r + 0.587*g + 0.114*b;
      this.yuv[3*i    ] = y;
      this.yuv[3*i + 1] = 0.492 * (b - y);
      this.yuv[3*i + 2] = 0.877 * (r - y);
    }
    this.graph = new Graph();
    this._build();
  }
  idx(x, y)   { return y * this.w + x; }
  inBounds(x, y, ox = 0, oy = 0) {
    return (x + ox) >= 0 && (x + ox) < this.w && (y + oy) >= 0 && (y + oy) < this.h;
  }
  rgbAt(x, y) {
    const i = 3 * this.idx(x, y);
    return (this.rgb[i] << 16) | (this.rgb[i+1] << 8) | this.rgb[i+2];
  }
  // YUV-based similarity (paper/hqx). Pixels are "equal" iff all channel diffs
  // are within thresholds.
  similar(x0, y0, x1, y1) {
    const a = 3 * this.idx(x0, y0), b = 3 * this.idx(x1, y1);
    return Math.abs(this.yuv[a]   - this.yuv[b])   <= PixelGraph.Y_T
        && Math.abs(this.yuv[a+1] - this.yuv[b+1]) <= PixelGraph.U_T
        && Math.abs(this.yuv[a+2] - this.yuv[b+2]) <= PixelGraph.V_T;
  }
  _build() {
    const g = this.graph;
    for (let y = 0; y < this.h; y++) for (let x = 0; x < this.w; x++) {
      g.addNode(key2(x, y));
      // 4 of 8 neighbors: right, down, down-right diagonal, up-right diagonal.
      this._tryEdge(x, y, x + 1, y    );
      this._tryEdge(x, y, x,     y + 1);
      this._tryEdge(x, y, x + 1, y - 1);
      this._tryEdge(x, y, x + 1, y + 1);
    }
  }
  _tryEdge(x0, y0, x1, y1) {
    if (x1 < 0 || x1 >= this.w || y1 < 0 || y1 >= this.h) return;
    if (!this.similar(x0, y0, x1, y1)) return;
    const diag = x0 !== x1 && y0 !== y1;
    this.graph.addEdge(key2(x0, y0), key2(x1, y1), { diagonal: diag });
  }
}

// ---------- Heuristics for resolving ambiguous diagonal pairs ----------------
class Heuristics {
  static SPARSE_WINDOW = 8;
  constructor(pg) { this.pg = pg; this.g = pg.graph; }

  apply(ambiguousPairs) {
    // Score every diagonal edge in every pair, then remove the lowest-scored.
    for (const pair of ambiguousPairs) for (const e of pair) {
      e.weight = this._curve(e) + this._sparse(e) + this._island(e);
    }
    for (const pair of ambiguousPairs) {
      const min = Math.min(...pair.map(e => e.weight));
      for (const e of pair) {
        if (e.weight === min) this.g.removeEdge(e.a, e.b);
      }
    }
  }
  // Length of the maximal valence-2 curve through this edge.
  _curve(e) {
    const seen = new Set([e.a < e.b ? e.a+'|'+e.b : e.b+'|'+e.a]);
    const stack = [e.a, e.b];
    while (stack.length) {
      const n = stack.pop();
      const nb = this.g.neighbors(n);
      if (nb.size !== 2) continue;                 // junction/endpoint: stop
      for (const m of nb) {
        const ek = n < m ? n+'|'+m : m+'|'+n;
        if (seen.has(ek)) continue;
        seen.add(ek); stack.push(m);
      }
    }
    return seen.size;
  }
  // Negative size of the component within an 8x8 window around the edge.
  _sparse(e) {
    const [ax, ay] = parseKey(e.a), [bx, by] = parseKey(e.b);
    const minx = Math.min(ax, bx), miny = Math.min(ay, by);
    const W = Heuristics.SPARSE_WINDOW;
    const ox = W/2 - 1 - minx, oy = W/2 - 1 - miny;
    const comp = new Set([e.a, e.b]); const stack = [e.a, e.b];
    while (stack.length) {
      const n = stack.pop();
      for (const m of this.g.neighbors(n)) {
        if (comp.has(m)) continue;
        const [mx, my] = parseKey(m);
        if ((mx + ox) >= 0 && (mx + ox) < W && (my + oy) >= 0 && (my + oy) < W) {
          comp.add(m); stack.push(m);
        }
      }
    }
    return -comp.size;
  }
  // +5 if cutting this edge would orphan a valence-1 endpoint.
  _island(e) {
    return (this.g.degree(e.a) === 1 || this.g.degree(e.b) === 1) ? 5 : 0;
  }
}

// ---------- Main pipeline ----------------------------------------------------
class PixelData {
  constructor(w, h, rgba) {
    this.pg = new PixelGraph(w, h, rgba);
    this.w  = w; this.h = h;
    this.gg = new Graph();          // grid (corner) graph
    // Each pixel stores its polygon-cell corner set (initially the 4 square corners).
    this.corners = new Map();       // pixel-key -> Set(corner-key)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      this.corners.set(key2(x, y), new Set([
        key2(x,   y  ), key2(x+1, y  ),
        key2(x,   y+1), key2(x+1, y+1),
      ]));
    }
  }
  run() {
    this._removeDiagonals();
    this._buildGrid();
    this._deformGrid();
    this._findShapes();
    this._findOutlines();
    this._assignBoundariesToShapes();
  }

  // ---- planarize similarity graph --------------------------------------------
  _removeDiagonals() {
    const g = this.pg.graph, ambiguous = [];
    for (let y = 0; y < this.h - 1; y++) for (let x = 0; x < this.w - 1; x++) {
      const nodes = [key2(x,y), key2(x+1,y), key2(x,y+1), key2(x+1,y+1)];
      // Collect edges internal to this 2x2 block.
      const edges = [];
      for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
        if (g.hasEdge(nodes[i], nodes[j])) {
          const at = g.edgeAttr(nodes[i], nodes[j]);
          edges.push({ a: nodes[i], b: nodes[j], diagonal: at.diagonal });
        }
      }
      const diag = edges.filter(e => e.diagonal);
      if (diag.length !== 2) continue;          // no crossing to resolve
      if (edges.length === 6)      diag.forEach(e => g.removeEdge(e.a, e.b));   // flat block
      else if (edges.length === 2) ambiguous.push(diag);                        // X only
    }
    new Heuristics(this.pg).apply(ambiguous);
  }

  // ---- (w+1)x(h+1) square-corner grid ---------------------------------------
  _buildGrid() {
    for (let y = 0; y <= this.h; y++) for (let x = 0; x <= this.w; x++) {
      const k = key2(x, y); this.gg.addNode(k);
      if (x < this.w) this.gg.addEdge(k, key2(x + 1, y));
      if (y < this.h) this.gg.addEdge(k, key2(x,     y + 1));
    }
  }

  // ---- deform cells to follow similarity graph, then collapse valence-2 ------
  _deformGrid() {
    const g = this.pg.graph;
    for (const node of g.nodes()) {
      const [x, y] = parseKey(node);
      for (const nb of g.neighbors(node)) {
        const [nx, ny] = parseKey(nb);
        if (nx === x || ny === y) continue;       // cardinal, skip
        this._deformCell(x, y, nx, ny);
      }
    }
    // Collapse valence-2 corner nodes for smoother cells. Keep the 4 image corners.
    const keep = new Set([key2(0,0), key2(0,this.h), key2(this.w,0), key2(this.w,this.h)]);
    const toRemove = [];
    for (const node of this.gg.nodes()) {
      if (keep.has(node)) continue;
      const nb = [...this.gg.neighbors(node)];
      if (nb.length === 2) this.gg.addEdge(nb[0], nb[1]);
      if (nb.length <= 2)  toRemove.push(node);
    }
    for (const n of toRemove) this.gg.removeNode(n);
    // Drop stale corners from each pixel's set.
    for (const [, cs] of this.corners) for (const c of cs) if (!this.gg.hasNode(c)) cs.delete(c);
  }

  // For one diagonal pair, chamfer the shared corner on each "nibbling" side.
  _deformCell(x, y, nx, ny) {
    const px = Math.max(x, nx), py = Math.max(y, ny);
    const ox = nx - x, oy = ny - y;                    // ±1 each
    const pixnode = key2(px, py);
    // The two adjacent cardinal neighbors of the same shape as (x,y).
    const testChamfer = (adjX, adjY, pn, mpn, npn) => {
      if (adjX < 0 || adjY < 0 || adjX >= this.w || adjY >= this.h) return;
      if (this.pg.rgbAt(x, y) === this.pg.rgbAt(adjX, adjY)) return;  // same color: no chamfer
      const adjK = key2(adjX, adjY);
      this.corners.get(adjK).delete(pixnode);          // remove pre-chamfer corner
      this.corners.get(adjK).add(npn);
      this.corners.get(key2(x, y)).add(npn);
      this._deformEdge(pixnode, pn, mpn, npn);
    };
    // "horizontal" neighbor test
    testChamfer(nx, y,
      key2(px, py - oy),
      key2(px, py - 0.5 * oy),
      key2(px + 0.25 * ox, py - 0.25 * oy),
    );
    // "vertical" neighbor test
    testChamfer(x, ny,
      key2(px - ox,      py),
      key2(px - 0.5 * ox, py),
      key2(px - 0.25 * ox, py + 0.25 * oy),
    );
  }

  _deformEdge(pixnode, pn, mpn, npn) {
    if (this.gg.hasNode(mpn)) {
      this.gg.removeEdge(mpn, pixnode);
    } else {
      this.gg.removeEdge(pn, pixnode);
      this.gg.addEdge(pn, mpn);
    }
    this.gg.addEdge(mpn, npn);
    this.gg.addEdge(npn, pixnode);
  }

  // ---- group similar pixels into Shape instances -----------------------------
  _findShapes() {
    this.shapes = [];
    for (const comp of this.pg.graph.components()) {
      const pixels = [...comp];
      const [x, y] = parseKey(pixels[0]);
      const color  = this.pg.rgbAt(x, y);
      const cset   = new Set();
      for (const p of pixels) for (const c of this.corners.get(p)) cset.add(c);
      this.shapes.push(new Shape(pixels, color, cset));
    }
  }

  // ---- outline graph = grid graph minus intra-shape edges --------------------
  _findOutlines() {
    this.outlines = this.gg.copy();
    const g = this.pg.graph;
    for (const pix of g.nodes()) {
      const cs = this.corners.get(pix);
      for (const nb of g.neighbors(pix)) {
        const shared = [];
        const nbCs = this.corners.get(nb);
        for (const c of cs) if (nbCs.has(c)) shared.push(c);
        if (shared.length === 2 && this.outlines.hasEdge(shared[0], shared[1])) {
          this.outlines.removeEdge(shared[0], shared[1]);
        }
      }
    }
    // drop isolated corners
    for (const n of [...this.outlines.nodes()]) if (this.outlines.degree(n) === 0) this.outlines.removeNode(n);
  }

  // ---- each shape's boundary is the outline subgraph on its corner set ------
  _assignBoundariesToShapes() {
    this.paths = new Map();                            // key -> Path (dedup)
    for (const shape of this.shapes) {
      const allCorners = [...shape.corners].filter(c => this.outlines.hasNode(c));
      if (!allCorners.length) continue;
      // Induced subgraph on this shape's corners only. Within the induced
      // graph each boundary node has degree 2, making each connected
      // component a simple cycle that the greedy tracer can walk cleanly.
      const sg = this.outlines.induced(new Set(allCorners));
      const lexMin = allCorners.reduce((a, b) => minKey(a, b));
      for (const comp of sg.components()) {
        const path = this._makePath(sg, comp);
        if (!path) continue;
        const isOuter = comp.has(lexMin);
        shape.addOutline(path, isOuter);
      }
    }
  }

  _makePath(graph, nodeSet) {
    const nodes = [...nodeSet];
    if (nodes.length < 3) return null;
    const key = nodes.slice().sort().join(';');
    if (this.paths.has(key)) return this.paths.get(key);
    const path = new Path(graph, nodeSet);
    this.paths.set(key, path);
    return path;
  }
}

// numeric comparison on "x,y" keys (parse once, compare tuple)
function minKey(a, b) {
  const [ax, ay] = parseKey(a), [bx, by] = parseKey(b);
  if (ax !== bx) return ax < bx ? a : b;
  return ay < by ? a : b;
}

// ---------- Shape ------------------------------------------------------------
class Shape {
  constructor(pixels, color, corners) {
    this.pixels  = pixels;
    this.color   = color;
    this.corners = corners;
    this.outer   = null;
    this.holes   = [];
  }
  addOutline(path, isOuter) {
    if (isOuter) this.outer = path; else this.holes.push(path);
    path.shapes.add(this);
  }
}

// ---------- Path: cyclic ordered traversal of a boundary subgraph ------------
// Uses angle-based next-edge selection to trace a topological face correctly
// even through degree-3+ junctions (multiple pinch points inside a shape's
// boundary), not just simple cycles.
class Path {
  constructor(graph, nodeSet) {
    this.graph  = graph;
    this.nodes  = this._trace(nodeSet);
    this.shapes = new Set();
    this.spline = null;
    this.smooth = null;
  }

  // Planar face-walking traversal. At every vertex we arrive along a
  // directed edge (prev→curr); to stay on the same face we pick the
  // next outgoing edge that is the MOST CLOCKWISE turn from the reverse
  // of our incoming direction. This gives the canonical "walk along the
  // wall with your right hand touching it" polygon trace.
  _trace(nodeSet) {
    // Pick the lex-min node as start. That node lies on the OUTER hull of
    // the component (smallest x, then smallest y among all corners), so the
    // face we trace starting there is the outer face of that component.
    let start = null;
    for (const n of nodeSet) if (start === null || minKey(n, start) === n) start = n;
    const nbsStart = [...this.graph.neighbors(start)].filter(n => nodeSet.has(n));
    if (nbsStart.length === 0) return [start];
    // From the lex-min corner go to the neighbor with smallest slope.
    // Since lex-min is on the outer hull, all neighbors are to the right
    // or below, so sorting by slope picks a canonical outgoing direction.
    const [sx, sy] = parseKey(start);
    nbsStart.sort((a, b) => this._slope(sx, sy, a) - this._slope(sx, sy, b));

    const path = [start];
    let prev = start, curr = nbsStart[0];
    // Walk until we return to the starting edge (start → nbsStart[0]).
    // Guard with a hard iteration cap as a safety net.
    const maxSteps = nodeSet.size * 4 + 10;
    let steps = 0;
    while (steps++ < maxSteps) {
      path.push(curr);
      const nb = [...this.graph.neighbors(curr)].filter(n => nodeSet.has(n));
      if (nb.length === 0) break;
      const next = this._nextCW(prev, curr, nb);
      if (next === null) break;
      // Termination: we've closed the loop when the oriented edge
      // (curr → next) equals the oriented edge we started with
      // (start → nbsStart[0]).
      if (curr === start && next === nbsStart[0]) break;
      prev = curr; curr = next;
    }
    // Drop duplicate trailing start if present (we re-visit 'start' then
    // detect termination).
    if (path.length > 1 && path[path.length - 1] === start) path.pop();
    return path;
  }

  // Given that we arrived at `curr` from `prev`, pick the next neighbor
  // (from `nbs`) that is the most-clockwise turn from the incoming edge.
  // Angles measured counter-clockwise from +x; we take the neighbor whose
  // angle, measured relative to the incoming reverse direction, is the
  // smallest positive rotation — i.e. the tightest clockwise turn.
  _nextCW(prev, curr, nbs) {
    const [cx, cy] = parseKey(curr);
    const [px, py] = parseKey(prev);
    // Incoming-reverse direction: from curr back toward prev.
    const inAng = Math.atan2(py - cy, px - cx);
    let best = null, bestTurn = Infinity;
    for (const n of nbs) {
      if (n === prev && nbs.length > 1) continue;   // don't U-turn if any alternative exists
      const [nx, ny] = parseKey(n);
      const outAng = Math.atan2(ny - cy, nx - cx);
      // CW turn magnitude from inAng to outAng, in [0, 2π).
      let turn = inAng - outAng;
      while (turn <    0) turn += 2 * Math.PI;
      while (turn >= 2 * Math.PI) turn -= 2 * Math.PI;
      if (turn < bestTurn) { bestTurn = turn; best = n; }
    }
    // If nbs.length === 1, only option is the U-turn — take it.
    if (best === null && nbs.length === 1) best = nbs[0];
    return best;
  }

  _slope(x0, y0, n) {
    const [x1, y1] = parseKey(n);
    const dx = x1 - x0, dy = y1 - y0;
    return dx === 0 ? dy * 1e14 : dy / dx;
  }
  makeSpline() {
    const pts = this.nodes.map(parseKey);
    if (pts.length < 3) { this.spline = null; return; }
    this.spline = ClosedBSpline.fit(pts, 2);
  }
}

// ---------- Quadratic closed B-spline ----------------------------------------
// Control points are plain Float64Array pairs packed [x0,y0,x1,y1,...].
// Closed form: the first `degree` points are duplicated at the end (wrap).
class ClosedBSpline {
  constructor(knots, cps, degree = 2) {
    this.knots  = knots;                 // Float64Array
    this.cps    = cps;                   // Float64Array of length 2*n
    this.degree = degree;
    this.n      = cps.length / 2;
    this.unwrappedLen = this.n - degree; // # of "useful" (non-wrapped) control points
  }
  static fit(points, degree = 2) {
    // Wrap: append first `degree` points to end so spline is closed.
    const all = points.concat(points.slice(0, degree));
    const n = all.length, m = n + degree;
    const knots = new Float64Array(m + 1);
    for (let i = 0; i <= m; i++) knots[i] = i / m;
    const cps = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) { cps[2*i] = all[i][0]; cps[2*i + 1] = all[i][1]; }
    return new ClosedBSpline(knots, cps, degree);
  }
  copy() { return new ClosedBSpline(new Float64Array(this.knots), new Float64Array(this.cps), this.degree); }
  domain() { return [this.knots[this.degree], this.knots[this.n]]; }
  cpGet(i) { return [this.cps[2*i], this.cps[2*i + 1]]; }
  cpSet(i, x, y) {
    i = ((i % this.unwrappedLen) + this.unwrappedLen) % this.unwrappedLen;
    this.cps[2*i] = x; this.cps[2*i + 1] = y;
    if (i < this.degree) {
      const j = i + this.unwrappedLen;
      this.cps[2*j] = x; this.cps[2*j + 1] = y;
    }
  }
  // de Boor's algorithm evaluated at u.
  eval(u) {
    const p = this.degree, k = this._span(u);
    // copy p+1 relevant control points
    const d = new Float64Array(2 * (p + 1));
    for (let i = 0; i <= p; i++) {
      d[2*i]     = this.cps[2 * (k - p + i)];
      d[2*i + 1] = this.cps[2 * (k - p + i) + 1];
    }
    for (let r = 1; r <= p; r++) for (let i = p; i >= r; i--) {
      const j = k - p + i;
      const a = (u - this.knots[j]) / (this.knots[j + p - r + 1] - this.knots[j]);
      d[2*i]     = (1 - a) * d[2*(i-1)]     + a * d[2*i];
      d[2*i + 1] = (1 - a) * d[2*(i-1) + 1] + a * d[2*i + 1];
    }
    return [d[2*p], d[2*p + 1]];
  }
  _span(u) {
    // Locate the knot span index k with knots[k] <= u < knots[k+1], clamped to domain.
    const p = this.degree, n = this.n;
    if (u >= this.knots[n]) return n - 1;
    if (u <= this.knots[p]) return p;
    let lo = p, hi = n;
    while (lo < hi - 1) { const mid = (lo + hi) >> 1; (u < this.knots[mid]) ? hi = mid : lo = mid; }
    return lo;
  }
  // Decompose into successive quadratic Beziers, one per interior knot span.
  // Mirrors Python Quadratic_Bezier_Fit: interior knots are knots[degree..n],
  // interior control points are cps[1..n-degree].
  *toBeziers() {
    const p = this.degree;
    let prev = this.eval(this.knots[p]);
    for (let k = 0; k < this.n - p; k++) {
      const ctrl = this.cpGet(k + 1);
      const end  = this.eval(this.knots[p + 1 + k]);
      yield [prev, ctrl, end];
      prev = end;
    }
  }
  // First-derivative spline (degree p-1).
  derivative() {
    if (this._d) return this._d;
    const p = this.degree, n = this.n;
    const newCps = new Float64Array(2 * (n - 1));
    for (let i = 0; i < n - 1; i++) {
      const c = p / (this.knots[i + 1 + p] - this.knots[i + 1]);
      newCps[2*i]     = c * (this.cps[2*(i+1)]     - this.cps[2*i]);
      newCps[2*i + 1] = c * (this.cps[2*(i+1) + 1] - this.cps[2*i + 1]);
    }
    const newKnots = this.knots.slice(1, this.knots.length - 1);
    this._d = new ClosedBSpline(newKnots, newCps, p - 1);
    return this._d;
  }
  _clearDerivCache() { this._d = null; this._dd = null; }
  curvature(u) {
    const d1s = this.derivative();
    const d2s = (this._dd ||= d1s.derivative());
    const [x1, y1] = d1s.eval(u), [x2, y2] = d2s.eval(u);
    const num = x1 * y2 - y1 * x2, den = Math.pow(x1*x1 + y1*y1, 1.5);
    return den === 0 ? 0 : Math.abs(num / den);
  }
  // Reverse curve direction (for SVG winding when drawing outer + holes).
  reversed() {
    const p = this.degree, n = this.n;
    const newKnots = new Float64Array(this.knots.length);
    for (let i = 0; i < this.knots.length; i++) newKnots[i] = 1 - this.knots[this.knots.length - 1 - i];
    const newCps = new Float64Array(this.cps.length);
    for (let i = 0; i < n; i++) {
      newCps[2*i]     = this.cps[2*(n - 1 - i)];
      newCps[2*i + 1] = this.cps[2*(n - 1 - i) + 1];
    }
    return new ClosedBSpline(newKnots, newCps, p);
  }
}

// ---------- Spline smoother (random-relaxation over energy) ------------------
class SplineSmoother {
  static INTERVALS_PER_SPAN = 20;
  static GUESSES            = 20;
  static OFFSET             = 0.05;
  static ITERATIONS         = 20;
  static POS_MULT           = 1;

  constructor(spline, opts = {}) {
    this.orig   = spline;                                 // reference for E_p
    this.s      = spline.copy();
    this.iters  = opts.iterations   ?? SplineSmoother.ITERATIONS;
    this.guesses= opts.guesses      ?? SplineSmoother.GUESSES;
    this.offset = opts.guessOffset  ?? SplineSmoother.OFFSET;
  }
  smooth() {
    const U = this.s.unwrappedLen;
    for (let it = 0; it < this.iters; it++) {
      for (let i = 0; i < U; i++) {
        const [sx, sy] = this.s.cpGet(i);
        let bestE = this._E(i), bx = sx, by = sy;
        for (let k = 0; k < this.guesses; k++) {
          const r  = Math.random() * this.offset;
          const th = Math.random() * Math.PI * 2;
          const cx = sx + r * Math.cos(th), cy = sy + r * Math.sin(th);
          this.s.cpSet(i, cx, cy); this.s._clearDerivCache();
          const e = this._E(i);
          if (e < bestE) { bestE = e; bx = cx; by = cy; }
        }
        this.s.cpSet(i, bx, by); this.s._clearDerivCache();
      }
    }
    return this.s;
  }
  _E(i) { return this._Ec(i) + this._Ep(i); }
  _Ep(i) {
    const [ox, oy] = this.orig.cpGet(i), [x, y] = this.s.cpGet(i);
    const d = Math.hypot(x - ox, y - oy);
    return d * d * d * d * SplineSmoother.POS_MULT;
  }
  // Integrated |curvature| over knot spans influenced by control point i.
  _Ec(i) {
    const p = this.s.degree, k = this.s.knots;
    let sum = 0;
    // Control point i influences knot spans [k[i+1]..k[i+p+1]], clamped to domain.
    const [d0, d1] = this.s.domain();
    for (let r = 0; r < p; r++) {
      const lo = Math.max(d0, Math.min(d1, k[i + 1 + r]));
      const hi = Math.max(d0, Math.min(d1, k[i + 2 + r]));
      if (lo === hi) continue;
      const N = SplineSmoother.INTERVALS_PER_SPAN;
      const step = (hi - lo) / N;
      let acc = 0.5 * (this.s.curvature(lo) + this.s.curvature(hi));
      for (let j = 1; j < N; j++) acc += this.s.curvature(lo + j * step);
      sum += acc * step;
    }
    return sum;
  }
}

// ---------- SVG writer -------------------------------------------------------
// Two render modes — both are correct implementations of parts of the paper.
//
// "shape" mode  (the classic pipeline — good for hand-drawn pixel art with a
//               small palette, like Yoshi, Invaders, Mario):
//   Emit one filled region per connected same-color component, with outer and
//   inner boundaries smoothed by quadratic B-splines (§3.3–3.4).
//
// "cell" mode   (the correct fallback for photographic / anti-aliased input):
//   Emit one polygon per pixel cell using that pixel's own color. This is the
//   rendering primitive the paper describes in §3.5: "place color diffusion
//   sources at the centroids of the cells". Photo-like input keeps every
//   unique tint instead of collapsing to a handful of flat blobs.
//
// The default is "auto": if the input contains more than 32 distinct colors
// we pick "cell"; otherwise "shape". This recovers classic behaviour for
// small sprites and fixes the Figure-11 failure case for photo input.
class SVGWriter {
  constructor(pd, scale = 40) { this.pd = pd; this.scale = scale; }

  _colorStr(c) {
    return 'rgb(' + ((c >> 16) & 255) + ',' + ((c >> 8) & 255) + ',' + (c & 255) + ')';
  }
  _pt(xy) {
    // Drop trailing ".0" to keep output compact. Fractional quarter-pixel
    // corners are preserved as needed.
    const fmt = v => {
      const s = (v * this.scale).toFixed(1);
      return s.endsWith('.0') ? s.slice(0, -2) : s;
    };
    return [fmt(xy[0]), fmt(xy[1])];
  }

  render(opts = {}) {
    const { w, h } = this.pd;
    const W = w * this.scale, H = h * this.scale;
    const mode = opts.mode || this._autoMode();
    const out = [
      '<svg xmlns="http://www.w3.org/2000/svg" width="', W, '" height="', H,
      '" viewBox="0 0 ', W, ' ', H, '" shape-rendering="geometricPrecision">',
    ];
    if (mode === 'cell') this._renderCells(out);
    else                 this._renderShapes(out);
    out.push('</svg>');
    return out.join('');
  }

  // Heuristic: count distinct colors. Classic pixel art rarely exceeds ~16
  // colors; AI-generated or downsampled inputs easily have 1000+.
  _autoMode() {
    const seen = new Set(); const max = 64;
    for (let i = 0; i < this.pd.pg.rgb.length; i += 3) {
      seen.add((this.pd.pg.rgb[i] << 16) | (this.pd.pg.rgb[i+1] << 8) | this.pd.pg.rgb[i+2]);
      if (seen.size > max) return 'cell';
    }
    return 'shape';
  }

  // ---- per-shape mode ------------------------------------------------------
  _renderShapes(out) {
    // Sort large-first so interior shapes layer on top.
    const shapes = [...this.pd.shapes].sort((a, b) => b.pixels.length - a.pixels.length);
    for (const shape of shapes) {
      if (!shape.outer || !shape.outer.smooth) continue;
      const splines = [shape.outer.smooth.reversed(),
                       ...shape.holes.filter(p => p.smooth).map(p => p.smooth)];
      const d = this._splinesPath(splines);
      if (!d) continue;
      const col = this._colorStr(shape.color);
      out.push('<path fill="', col, '" stroke="', col,
               '" stroke-width="0.5" fill-rule="evenodd" d="', d, '"/>');
    }
  }
  _splinesPath(splines) {
    const parts = [];
    for (const spline of splines) {
      const beziers = [...spline.toBeziers()];
      if (!beziers.length) continue;
      const [sx, sy] = this._pt(beziers[0][0]);
      parts.push('M', sx, ',', sy);
      for (const [, ctrl, end] of beziers) {
        const [cx, cy] = this._pt(ctrl), [ex, ey] = this._pt(end);
        parts.push('Q', cx, ',', cy, ' ', ex, ',', ey);
      }
      parts.push('Z');
    }
    return parts.join('');
  }

  // ---- per-cell mode -------------------------------------------------------
  // Two-layer approach that preserves per-pixel color (gradients, anti-
  // aliasing) WHILE giving smoothly curved region silhouettes:
  //   1. For each connected same-color shape, define a <clipPath> from its
  //      smoothed outer spline boundary (and hole boundaries).
  //   2. Render every pixel cell inside that clip with its own color.
  // The clip absorbs the cell-polygon jaggies at the region boundary; the
  // interior keeps every per-pixel tint.
  _renderCells(out) {
    const MIN_SMOOTH_PIXELS = 4;    // below this, skip spline silhouette
    // Background pass: paint every pixel as a scale×scale square first.
    // This guarantees full coverage with no sub-pixel gaps; subsequent
    // clipped layers will overdraw with smooth silhouettes where valuable.
    const S = this.scale;
    for (let y = 0; y < this.pd.h; y++) for (let x = 0; x < this.pd.w; x++) {
      const color = this._colorStr(this.pd.pg.rgbAt(x, y));
      out.push('<rect x="', x*S, '" y="', y*S, '" width="', S, '" height="', S,
               '" fill="', color, '" shape-rendering="crispEdges"/>');
    }
    // Clipped per-shape cells for silhouette smoothing.
    const shapes = [...this.pd.shapes].sort((a, b) => b.pixels.length - a.pixels.length);
    out.push('<defs>');
    const shapeId = new Map();
    for (let i = 0; i < shapes.length; i++) {
      const s = shapes[i];
      if (!s.outer || !s.outer.smooth || s.pixels.length < MIN_SMOOTH_PIXELS) continue;
      const id = 'c' + i; shapeId.set(s, id);
      const splines = [s.outer.smooth.reversed(),
                       ...s.holes.filter(p => p.smooth).map(p => p.smooth)];
      const d = this._splinesPath(splines);
      out.push('<clipPath id="', id, '" clip-rule="evenodd"><path d="', d, '"/></clipPath>');
    }
    out.push('</defs>');
    for (const s of shapes) {
      const id = shapeId.get(s);
      if (!id) continue;   // tiny shape — nearest-neighbor pass already covered it
      const [avgR, avgG, avgB] = this._avgColor(s);
      const avg = 'rgb(' + avgR + ',' + avgG + ',' + avgB + ')';
      const splines = [s.outer.smooth.reversed(),
                       ...s.holes.filter(p => p.smooth).map(p => p.smooth)];
      const bgD = this._splinesPath(splines);
      // Paint the shape's avg color as silhouette backdrop, then overlay
      // per-pixel cells inside the clipPath. Cells use a wider stroke to
      // bleed across seams; the clipPath hides overflow at the boundary.
      out.push('<g clip-path="url(#', id, ')">',
               '<path fill="', avg, '" fill-rule="evenodd" d="', bgD, '"/>');
      for (const pix of s.pixels) {
        const [x, y] = parseKey(pix);
        const poly = this._cellPolygon(x, y);
        if (!poly || poly.length < 3) continue;
        const color = this._colorStr(this.pd.pg.rgbAt(x, y));
        out.push('<path fill="', color, '" stroke="', color,
                 '" stroke-width="1.5" stroke-linejoin="round" d="',
                 this._polyPath(poly), '"/>');
      }
      out.push('</g>');
    }
  }
  _avgColor(shape) {
    let r = 0, g = 0, b = 0;
    for (const pix of shape.pixels) {
      const c = this.pd.pg.rgbAt(...parseKey(pix));
      r += (c >> 16) & 255; g += (c >> 8) & 255; b += c & 255;
    }
    const n = shape.pixels.length;
    return [Math.round(r/n), Math.round(g/n), Math.round(b/n)];
  }
  _renderCellsPlain(out) {
    for (let y = 0; y < this.pd.h; y++) for (let x = 0; x < this.pd.w; x++) {
      const poly = this._cellPolygon(x, y);
      if (!poly || poly.length < 3) continue;
      const color = this._colorStr(this.pd.pg.rgbAt(x, y));
      out.push('<path fill="', color, '" stroke="', color,
               '" stroke-width="0.5" stroke-linejoin="round" d="',
               this._polyPath(poly), '"/>');
    }
  }
  _cellPolygon(x, y) {
    const cornerSet = this.pd.corners.get(key2(x, y));
    if (!cornerSet || cornerSet.size < 3) return null;
    const verts = [...cornerSet].map(parseKey);
    // Angular sort around centroid: each cell is convex after valence-2
    // collapse, so this reliably recovers polygon order.
    let cx = 0, cy = 0;
    for (const [vx, vy] of verts) { cx += vx; cy += vy; }
    cx /= verts.length; cy /= verts.length;
    verts.sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
    return verts;
  }
  _polyPath(verts) {
    const [x0, y0] = this._pt(verts[0]);
    const parts = ['M', x0, ',', y0];
    for (let i = 1; i < verts.length; i++) {
      const [x, y] = this._pt(verts[i]);
      parts.push('L', x, ',', y);
    }
    parts.push('Z');
    return parts.join('');
  }
}

// ---------- Public entry point -----------------------------------------------
function depixelize(image, opts = {}) {
  const { scale = 40, smooth = true } = opts;
  const w = image.width, h = image.height;
  const rgba = image.data;
  const pd = new PixelData(w, h, rgba);
  pd.run();
  const writer = new SVGWriter(pd, scale);
  const mode = opts.mode || writer._autoMode();
  // Fit splines whenever we'll use them: "shape" mode draws them directly,
  // "cell" mode uses them as clip boundaries to round cell-polygon edges.
  // Only skip if the caller explicitly asked for smooth=false.
  if (smooth !== false) {
    for (const path of pd.paths.values()) path.makeSpline();
    for (const path of pd.paths.values()) {
      if (!path.spline) continue;
      if (path.shapes.size === 1) {
        path.smooth = path.spline.copy();
      } else {
        path.smooth = new SplineSmoother(path.spline, opts).smooth();
      }
    }
  }
  return writer.render({ ...opts, mode });
}

// ---------- exports ----------------------------------------------------------
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { depixelize, PixelData, SVGWriter, ClosedBSpline, SplineSmoother, Graph };
}
if (typeof window !== 'undefined') window.depixelize = depixelize;
