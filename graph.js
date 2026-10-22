/**
 * week6/graph.js - HW6 graph / Personalized PageRank layer.
 *
 * This is a plain (classic) script, NOT an ES module. `index.html` loads the
 * TensorFlow.js runtime from a CDN, then the vendored d3 v7 build, then
 * `data.js`, then this file, then `script.js`:
 *
 *   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
 *   <script src="vendor/d3.min.js"></script>
 *   <script src="data.js"></script>
 *   <script src="graph.js"></script>
 *   <script src="script.js"></script>
 *
 * `week6/data.js` (loaded before this file) exposes the MovieLens 100k data
 * layer as globals: `movies`, `ratings`, `numUsers`, `numMovies`,
 * `ratingMatrix`, `genreNames`, `SPLIT_EXCLUSIONS` and `splitByTimestamp()`.
 * This file does not mutate them.
 *
 * WHAT YOU MUST IMPLEMENT (`TODO(hw6)` - each stub throws until you write it):
 *   1. `buildGraph(train)`            - bipartite graph over training positives.
 *   2. `rowNormalize(graph)`          - row-stochastic transition Q + dangling mask.
 *   3. `pageRank(alpha, opts)`        - uniform-restart PageRank.
 *   4. `personalizedPageRank(userId, alpha, opts)` - one-hot restart at one user.
 *
 * PROVIDED for you (scaffolding): the `REFERENCE_GRAPH` constant with the two
 * reference vectors and both rankings from readme.md section 4.1 (plus the
 * derived `REFERENCE_GRAPH.transition`, see below), the graph
 * accessors (`GRAPH`, `setGraph`, `getGraph`), the d3 graph view
 * (`renderGraphView`), the 4-node fixture teaching demo (`fixtureStep` /
 * `fixtureReset`), and the `#graph-summary-body` renderer.
 *
 * GRACEFUL DEGRADATION: the fixture demo and the graph view work before any
 * stub is implemented, so the page loads and renders with no console errors.
 *
 * SCOPE NOTE: d3 is used ONLY by the graph view. The propagation stubs must use
 * TensorFlow.js tensor operations on `R` and its transpose for the MovieLens
 * graph (readme.md section 3.5); the fixture demo below is the only
 * plain-JavaScript propagation loop in this file.
 *
 * @module week6/graph
 */

// ---------------------------------------------------------------------------
// Typedefs (documentation only - no runtime effect)
// ---------------------------------------------------------------------------

/**
 * A single rating record as produced by `data.js` / `splitByTimestamp()`.
 *
 * @typedef {Object} Rating
 * @property {number} userId
 * @property {number} movieId
 * @property {number} rating    1..5 (MovieLens scale).
 * @property {number} timestamp Unix seconds.
 */

/**
 * The bipartite user-movie graph returned by `buildGraph`.
 *
 * @typedef {Object} Graph
 * @property {number} nUsers        Number of distinct user ids observed in TRAIN.
 * @property {number} nMovies       Number of distinct movie ids observed in TRAIN.
 * @property {number[]} userIds     User ids, ascending; index = local user index.
 * @property {number[]} movieIds    Movie ids, ascending; index = local movie index.
 * @property {Map<number,number>} userIndexOf  userId -> local user index [0, nUsers).
 * @property {Map<number,number>} movieIndexOf movieId -> local movie index [0, nMovies).
 * @property {number} nodeCount     nUsers + nMovies (no padding node).
 * @property {function(number):number} nodeOfUser    userId -> global node index.
 * @property {function(number):number} movieOfNode   global node index -> movieId.
 * @property {function(number):boolean} isUserNode   global node index -> user?
 * @property {Float64Array} degree  Bipartite degree over node index.
 * @property {number[]} zeroDegreeUsers  User ids with degree 0.
 * @property {number[]} zeroDegreeMovies Movie ids with degree 0.
 * @property {number} positiveEdges Number of positive training edges (1 entries in R).
 * @property {number[][]} R         R[u][i] = 1 iff user u rated movie i >= 4.0
 *   in TRAINING, else 0. Shape [nUsers][nMovies].
 */

/**
 * The row-normalized transition returned by `rowNormalize`.
 *
 * @typedef {Object} RowNormalized
 * @property {number[][]} Q             Row-stochastic transition; positive-degree
 *   rows sum to 1, zero-degree rows are all zero.
 * @property {boolean[]} danglingMask   true for zero-degree node indices.
 */

/**
 * A PageRank / PPR result.
 *
 * @typedef {Object} PageRankResult
 * @property {Float64Array} p     Node probabilities, node order, sums to ~1.
 * @property {number} iterations  Number of updates performed.
 * @property {number} residual    Final L1 difference between consecutive distributions.
 * @property {boolean} converged  true when residual < tol before the cap.
 */

/**
 * An explicit transition supplied to `pageRank` / `personalizedPageRank`
 * through `opts.transition`.
 *
 * Use this to propagate on a caller-provided graph instead of the global
 * `GRAPH` (the HW6 harness uses it to run the implementation on the 4-node
 * reference fixture, which has no MovieLens graph). When present it replaces
 * the global-graph lookup for that call.
 *
 * @typedef {Object} TransitionSpec
 * @property {number[][]} Q          Row-normalized transition; `Q[i][j]` is the
 *   probability of moving from node `i` to node `j`. Square (n x n). A
 *   positive-degree row sums to 1; a zero-degree row is all zero. Already
 *   normalized, so the implementation must NOT re-normalize it.
 * @property {Uint8Array} danglingMask Nonzero (1) exactly at the zero-degree
 *   node indices, indexed by position in the same node order as `Q`.
 * @property {string[]} [nodeLabels] Optional labels for display and tests only;
 *   the propagation never reads them. `REFERENCE_GRAPH.transition.nodeLabels`
 *   is `['A', 'B', 'C', 'D']`.
 */

/**
 * Options for `renderGraphView`.
 *
 * @typedef {Object} GraphViewOptions
 * @property {("fixture"|"ego")} [mode]   Which graph to draw.
 * @property {number|null} [seedUserId]   Seed user for the "ego" subgraph.
 * @property {("degree"|"score")} [colorBy] Node fill encoding.
 * @property {number} [maxNodes]          Cap on drawn nodes (default 120).
 * @property {function(number):number} [getScore] Node index -> score, used when
 *   colorBy === "score".
 */

// ---------------------------------------------------------------------------
// Reference graph (provided, read-only by convention)
//
// Exactly the readme.md section 4.1 fixture: node order A, B, C, D; directed
// edges A -> B, A -> C, B -> C, C -> A, D -> C; no dangling nodes. The two
// reference vectors and both rankings are reproduced verbatim from readme.md
// section 4.1.
// ---------------------------------------------------------------------------

/**
 * The 4-node reference graph from readme.md section 4.1, plus its reference
 * results. `nodes` assigns A, B to the left (user) column and C, D to the right
 * (movie) column so the fixture can reuse the same deterministic two-column
 * layout as the data view. Treat this object as read-only.
 */
const REFERENCE_GRAPH = {
  nodeOrder: ["A", "B", "C", "D"],
  nodes: [
    { index: 0, label: "A", kind: "user" },
    { index: 1, label: "B", kind: "user" },
    { index: 2, label: "C", kind: "movie" },
    { index: 3, label: "D", kind: "movie" }
  ],
  // [from, to] pairs in node index space (A = 0, B = 1, C = 2, D = 3).
  edges: [
    [0, 1], // A -> B
    [0, 2], // A -> C
    [1, 2], // B -> C
    [2, 0], // C -> A
    [3, 2]  // D -> C
  ],
  edgeLabels: ["A -> B", "A -> C", "B -> C", "C -> A", "D -> C"],
  hasDanglingNodes: false,
  reference: {
    // alpha = 0.85, uniform restart over all four nodes.
    uniform: {
      alpha: 0.85,
      restart: "uniform",
      p: [0.3725268513, 0.1958239118, 0.3941492369, 0.0375000000],
      ranking: [2, 0, 1, 3],
      rankingLabel: "C > A > B > D"
    },
    // alpha = 0.85, restart only at A.
    restartAtA: {
      alpha: 0.85,
      restart: "A",
      restartIndex: 0,
      p: [0.4522328999, 0.1921989825, 0.3555681176, 0.0000000000],
      ranking: [0, 2, 1, 3],
      rankingLabel: "A > C > B > D"
    }
  }
};

// PROVIDED: the fixture's row-normalized transition, DERIVED from
// `REFERENCE_GRAPH.edges` (never a hand-typed matrix). `Q` is what
// `pageRank` / `personalizedPageRank` must consume when the harness passes
// `opts.transition`; `danglingMask` is a Uint8Array, 1 at zero-degree node
// indices (all zeros here, since A/B/C/D all have an outgoing link).
//
// The load-time self-test below is deliberately NON-FATAL: it records state in
// `REFERENCE_GRAPH.transitionSelfTest` instead of throwing, so a malformed
// fixture can never break page load. It adds no global name.
(function () {
  const fixture = graphFixtureTransition(); // derives P from `edges`
  const n = REFERENCE_GRAPH.nodeOrder.length;
  const Q = fixture.P.map(function (row) { return row.slice(); });
  const danglingMask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (fixture.out[i] === 0) danglingMask[i] = 1;
  }

  REFERENCE_GRAPH.transition = {
    Q: Q,
    danglingMask: danglingMask,
    nodeLabels: REFERENCE_GRAPH.nodeOrder.slice()
  };

  let maxRowSumError = 0;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += Q[i][j];
    maxRowSumError = Math.max(maxRowSumError, Math.abs(sum - 1));
  }
  let danglingCount = 0;
  for (let i = 0; i < n; i++) {
    if (danglingMask[i]) danglingCount += 1;
  }
  REFERENCE_GRAPH.transitionSelfTest = {
    rows: n,
    cols: Q.length ? Q[0].length : 0,
    maxRowSumError: maxRowSumError,
    rowsSumToOne: maxRowSumError < 1e-12,
    danglingCount: danglingCount,
    danglingMaskAllZero: danglingCount === 0
  };
})();

// ---------------------------------------------------------------------------
// Graph-view theme (private)
//
// Canvas is #FFFFFF; the categorical pair is Okabe-Ito blue/orange (red-green
// colour-blind safe), and scores use a single-hue sequential blue ramp.
// ---------------------------------------------------------------------------

/**
 * Fixed visual parameters for the graph view. Private to graph.js.
 */
const graphTheme = {
  width: 1000,
  height: 420,
  margin: { top: 34, right: 110, bottom: 26, left: 110 },
  userColor: "#0072B2",
  movieColor: "#E69F00",
  edgeColor: "#b9c0ca",
  nodeStroke: "#ffffff",
  zeroStroke: "#1c2128",
  labelColor: "#1c2128",
  noteColor: "#5b6472",
  minRadius: 6,
  maxRadius: 22,
  rampLight: [232, 243, 252],
  rampDark: [8, 48, 96],
  tooltipId: "graph-tooltip"
};

// ---------------------------------------------------------------------------
// Module state (public: GRAPH_VIEW_STATE)
// ---------------------------------------------------------------------------

/**
 * Small module-level state for the graph view. Mutated only by the rendering
 * and fixture-demo code in this file.
 */
const GRAPH_VIEW_STATE = {
  mode: "fixture",
  seedUserId: null,
  colorBy: "degree",
  maxNodes: 120,
  lastRender: null,
  fixture: {
    iteration: 0,
    p: null,
    residual: null,
    restart: "uniform",
    alpha: 0.85
  }
};

// ---------------------------------------------------------------------------
// Graph accessors (public)
// ---------------------------------------------------------------------------

/**
 * The current graph, or null before `buildGraph` has been wired in.
 *
 * @type {Graph|null}
 */
let GRAPH = null;

/**
 * Cached TensorFlow.js block tensors for the active graph.
 *
 * @type {{graph:Object, nUsers:number, nMovies:number,
 *         aTranspose:tf.Tensor2D, bTranspose:tf.Tensor2D, mask:tf.Tensor1D}|null}
 */
let GRAPH_BLOCK_CACHE = null;

/**
 * Release the cached block tensors. Called whenever the active graph changes so
 * repeated graph builds cannot accumulate GPU memory.
 *
 * @returns {void}
 */
function graphReleaseBlockCache() {
  if (!GRAPH_BLOCK_CACHE) return;
  try {
    if (GRAPH_BLOCK_CACHE.aTranspose) GRAPH_BLOCK_CACHE.aTranspose.dispose();
    if (GRAPH_BLOCK_CACHE.bTranspose) GRAPH_BLOCK_CACHE.bTranspose.dispose();
    if (GRAPH_BLOCK_CACHE.mask) GRAPH_BLOCK_CACHE.mask.dispose();
  } catch (error) {
    // Disposal is best-effort; the cache is replaced wholesale below.
  }
  GRAPH_BLOCK_CACHE = null;
}

/**
 * Replace the current graph and refresh the dependent UI.
 *
 * `script.js` should call this (rather than assigning `GRAPH` directly) so the
 * graph summary and the graph view re-render whenever the graph changes. A
 * `null` argument clears the graph and restores the empty states.
 *
 * @param {Graph|null} g
 * @returns {Graph|null} the stored graph.
 */
function setGraph(g) {
  if (GRAPH_BLOCK_CACHE && GRAPH_BLOCK_CACHE.graph !== g) graphReleaseBlockCache();
  GRAPH = g || null;
  try {
    graphRenderSummary(GRAPH);
  } catch (error) {
    graphSetLegendMessage("Graph summary failed: " + graphErrorMessage(error));
  }
  try {
    renderGraphView({ mode: GRAPH_VIEW_STATE.mode });
  } catch (error) {
    graphSetLegendMessage("Graph view failed: " + graphErrorMessage(error));
  }
  return GRAPH;
}

/**
 * Read the current graph without changing any state.
 *
 * @returns {Graph|null}
 */
function getGraph() {
  return GRAPH;
}

// ---------------------------------------------------------------------------
// Student stubs (TODO(hw6))
//
// Each stub throws until you implement it. Keep the signature and the return
// shape; update the JSDoc contract notes if your design differs. The graph view
// and the fixture demo keep working while these throw.
// ---------------------------------------------------------------------------

/**
 * Build the bipartite user-movie graph from the TRAINING split.
 *
 * TODO(hw6): implement this. Rules (readme.md section 3.1):
 *  - The node universe is the set of user ids and movie ids observed in
 *    `train` ONLY. Map ids to contiguous indices explicitly; never add a
 *    padding node (no index-0 "dummy" row).
 *  - Node order is users first, then movies:
 *      user u  -> global node userIndexOf.get(u) in [0, nUsers)
 *      movie m -> global node nUsers + movieIndexOf.get(m) in [nUsers, nUsers + nMovies)
 *    The graph view relies on this exact order (it treats node < nUsers as a
 *    user node and indexes degree by the global node index).
 *  - R[u][i] = 1 iff user u rated movie i with rating >= 4.0 in TRAINING, else
 *    0. A zero is "no positive edge", not a negative opinion.
 *  - degree is the bipartite degree over BOTH node kinds.
 *
 * Assumptions: `train` is an array of {userId, movieId, rating, timestamp}
 * records; ratings >= 4.0 are positives. Failure cases: a non-array `train`
 * should be treated as empty (return an empty graph) rather than throwing.
 *
 * @param {Rating[]} train
 * @returns {Graph}
 */
function buildGraph(train) {
  const records = Array.isArray(train) ? train : [];

  // 1. Collect every user and movie id observed in the TRAINING split, including
  //    ids whose ratings are all below 4.0 (they become zero-degree nodes).
  const userSet = new Set();
  const movieSet = new Set();
  for (let r = 0; r < records.length; r++) {
    const row = records[r];
    if (!row) continue;
    const userId = Number(row.userId);
    const movieId = Number(row.movieId);
    if (!Number.isFinite(userId) || !Number.isFinite(movieId)) continue;
    userSet.add(userId);
    movieSet.add(movieId);
  }

  // 2. Map ids to contiguous indices, no padding node, ascending id order.
  const userIds = Array.from(userSet).sort(function (a, b) { return a - b; });
  const movieIds = Array.from(movieSet).sort(function (a, b) { return a - b; });
  const nUsers = userIds.length;
  const nMovies = movieIds.length;
  const userIndexOf = new Map();
  const movieIndexOf = new Map();
  for (let u = 0; u < nUsers; u++) userIndexOf.set(userIds[u], u);
  for (let i = 0; i < nMovies; i++) movieIndexOf.set(movieIds[i], i);

  // 3. R[u][i] = 1 only for a positive (>= 4.0) training rating.
  const R = new Array(nUsers);
  for (let u = 0; u < nUsers; u++) R[u] = new Array(nMovies).fill(0);
  const degree = new Float64Array(nUsers + nMovies);
  let positiveEdges = 0;
  for (let r = 0; r < records.length; r++) {
    const row = records[r];
    if (!row) continue;
    const rating = Number(row.rating);
    if (!(rating >= 4.0)) continue;
    const u = userIndexOf.get(Number(row.userId));
    const i = movieIndexOf.get(Number(row.movieId));
    if (u === undefined || i === undefined) continue;
    if (R[u][i]) continue; // duplicate (user, movie) pair: count the edge once
    R[u][i] = 1;
    degree[u] += 1;
    degree[nUsers + i] += 1;
    positiveEdges += 1;
  }

  const zeroDegreeUsers = [];
  for (let u = 0; u < nUsers; u++) if (degree[u] === 0) zeroDegreeUsers.push(userIds[u]);
  const zeroDegreeMovies = [];
  for (let i = 0; i < nMovies; i++) {
    if (degree[nUsers + i] === 0) zeroDegreeMovies.push(movieIds[i]);
  }

  const movieNodes = new Array(nMovies);
  for (let i = 0; i < nMovies; i++) movieNodes[i] = nUsers + i;

  return {
    nUsers: nUsers,
    nMovies: nMovies,
    userIds: userIds,
    movieIds: movieIds,
    userIndexOf: userIndexOf,
    movieIndexOf: movieIndexOf,
    nodeCount: nUsers + nMovies,
    nodeOfUser: function (userId) {
      const local = userIndexOf.get(Number(userId));
      return local === undefined ? -1 : local;
    },
    movieOfNode: function (node) {
      const local = Number(node) - nUsers;
      return local >= 0 && local < nMovies ? movieIds[local] : -1;
    },
    isUserNode: function (node) {
      const index = Number(node);
      return Number.isFinite(index) && index >= 0 && index < nUsers;
    },
    movieNodes: movieNodes,
    degree: degree,
    zeroDegreeUsers: zeroDegreeUsers,
    zeroDegreeMovies: zeroDegreeMovies,
    positiveEdges: positiveEdges,
    R: R,
  };
}

/**
 * Row-normalize the transition matrix and record the zero-degree nodes.
 *
 * TODO(hw6): for each positive-degree node, divide its row by its degree so the
 * row sums to 1; leave zero-degree rows all-zero and mark them in
 * `danglingMask`. The dangling mass must be handled in the update exactly as in
 * readme.md section 3.3:
 *
 *   p_next = alpha * (Q^T * p + d * v) + (1 - alpha) * v
 *
 * where d is the probability mass currently sitting on zero-degree nodes.
 *
 * INTENDED TENSORFLOW.JS USE (readme.md section 3.5): this helper returns plain
 * JavaScript arrays so it can be inspected and tested, but the student
 * implementation should lift them into tensors for the batched propagation:
 * build the full transition from the two off-diagonal blocks (R row-normalized
 * by user degree and R-transpose row-normalized by movie degree) so the square
 * adjacency matrix is never materialized, and use `tf.tensor2d(Q)`,
 * `tf.transpose`, `tf.matMul`, and a `danglingMask`-derived correction for the
 * dangling term. Q and danglingMask are deliberately plain so tests can verify
 * row sums and the mask without a backend.
 *
 * Assumptions: `graph` comes from `buildGraph` (node order users then movies).
 * Failure cases: a null/missing graph returns { Q: [], danglingMask: [] }.
 *
 * @param {Graph} graph
 * @returns {RowNormalized}
 */
function rowNormalize(graph) {
  if (!graph || !graph.R) return { Q: [], danglingMask: [] };

  const R = graph.R;
  const nUsers = Number.isFinite(graph.nUsers) ? Number(graph.nUsers) : R.length;
  const firstRow = R.length > 0 && R[0] ? R[0].length : 0;
  const nMovies = Number.isFinite(graph.nMovies) ? Number(graph.nMovies) : firstRow;
  const n = nUsers + nMovies;

  // Out-degrees measured from R, so normalization and the dangling mask can
  // never disagree with the edges actually observed.
  const userDegree = new Float64Array(nUsers);
  const movieDegree = new Float64Array(nMovies);
  for (let u = 0; u < nUsers; u++) {
    const row = R[u] || [];
    for (let i = 0; i < nMovies; i++) {
      if (row[i]) {
        userDegree[u] += 1;
        movieDegree[i] += 1;
      }
    }
  }

  const Q = new Array(n);
  for (let i = 0; i < n; i++) Q[i] = new Array(n).fill(0);
  const danglingMask = new Array(n).fill(false);

  for (let u = 0; u < nUsers; u++) {
    if (userDegree[u] === 0) {
      danglingMask[u] = true;
      continue;
    }
    const scale = 1 / userDegree[u];
    const row = R[u] || [];
    for (let i = 0; i < nMovies; i++) {
      if (row[i]) Q[u][nUsers + i] = scale;
    }
  }

  for (let i = 0; i < nMovies; i++) {
    const node = nUsers + i;
    if (movieDegree[i] === 0) {
      danglingMask[node] = true;
      continue;
    }
    const scale = 1 / movieDegree[i];
    for (let u = 0; u < nUsers; u++) {
      const row = R[u] || [];
      if (row[i]) Q[node][u] = scale;
    }
  }

  return { Q: Q, danglingMask: danglingMask };
}

/**
 * Global PageRank with a uniform restart distribution.
 *
 * TODO(hw6): implement the power iteration with a uniform restart vector v
 * (readme.md sections 3.2-3.4), using TensorFlow.js for the batched
 * propagation:
 *
 *   p_0 = v ;  p_next = alpha * (Q^T * p + d * v) + (1 - alpha) * v
 *
 * Initialize p to v. Stop when the L1 difference between consecutive
 * distributions is below `tol`, or after `maxIter` iterations; an iteration cap
 * is NOT proof of convergence, so `converged` must be false when the cap is
 * hit. Do not force any node's score to equal the restart probability.
 *
 * TRANSITION SOURCE - part of the contract you must implement:
 *  - `opts.transition` present (a {@link TransitionSpec}, e.g.
 *    `REFERENCE_GRAPH.transition`): propagate on THAT matrix and do not require
 *    the global `GRAPH`. `Q` is already row-normalized, so do not re-normalize
 *    it; read `danglingMask` by position (`danglingMask[i]` is 1 iff node `i` is
 *    zero-degree). This is how the HW6 harness runs your implementation on the
 *    4-node reference fixture.
 *  - `opts.transition` absent: behave exactly as for MovieLens - require the
 *    global `GRAPH` (from `buildGraph` + `rowNormalize`) and build the
 *    transition from its `R` / `degree`.
 *  - Neither present: throw a clear Error; never silently fall back to a
 *    uniform distribution.
 *
 * NODE-INDEX CONVENTION: node indices are global positions. For the MovieLens
 * graph they follow `buildGraph` (users `[0, nUsers)`, then movies
 * `[nUsers, nUsers + nMovies)`, no padding node). For a supplied
 * `opts.transition` they follow `Q`'s own order. `p[i]` is the probability of
 * node `i` in that same order.
 *
 * The zero-degree policy must be identical here and in `personalizedPageRank`.
 *
 * @param {number} alpha Link-following probability in [0, 1]; 1 - alpha restarts.
 * @param {{tol?: number, maxIter?: number, transition?: TransitionSpec}} [opts]
 *   Defaults tol = 1e-6, maxIter = 500, transition absent (use the global GRAPH).
 * @returns {PageRankResult} `{p, iterations, residual, converged}`: `p` is a
 *   Float64Array over node indices that sums to ~1; `iterations` is the number
 *   of updates performed; `residual` is the final L1 difference
 *   `sum_j |p_next[j] - p[j]|`; `converged` is true only when `residual < tol`
 *   before the iteration cap was reached.
 * @throws {Error} when neither `opts.transition` nor the global `GRAPH` exists.
 */
function pageRank(alpha, opts) {
  const options = opts || {};
  const source = graphTransitionSource(options.transition);
  const n = source.kind === "dense" ? source.Q.length : graphNodeCountOf(source.graph);
  const restart = graphUniformRestart(n);
  return graphPropagate(alpha, restart, source, options);
}

/**
 * Personalized PageRank with a one-hot restart at one user's node.
 *
 * TODO(hw6): identical to `pageRank` except the restart vector v is one-hot at
 * one node. It accepts the same `opts.transition` override, the same stopping
 * rule, the same zero-degree policy and the same return shape
 * `{p, iterations, residual, converged}`.
 *
 * RESTART INDEX (node index, not a raw user id): the first argument is a node
 * INDEX and is used directly as the restart node. This is today's caller
 * convention; keep it.
 *  - default MovieLens path: the index is a global node index from `buildGraph`
 *    (a user node is `GRAPH.nodeOfUser(userId)` or equivalently any
 *    `index < GRAPH.nUsers`).
 *  - `opts.transition` path: the index must be in `[0, Q.length)` and refers to
 *    the same node order as `Q` (`REFERENCE_GRAPH.transition.nodeLabels` names
 *    the fixture's nodes for display).
 *
 * FAILURE CASES: an unknown / out-of-range restart index must throw a clear
 * Error naming the index rather than silently falling back to a uniform
 * restart; and when neither `opts.transition` nor the global `GRAPH` is
 * available, throw instead of guessing.
 *
 * @param {number} userId Node index of the restart node (see RESTART INDEX).
 * @param {number} alpha  Link-following probability in [0, 1]; 1 - alpha restarts.
 * @param {{tol?: number, maxIter?: number, transition?: TransitionSpec}} [opts]
 *   Same options as `pageRank`; defaults tol = 1e-6, maxIter = 500, transition
 *   absent (use the global GRAPH).
 * @returns {PageRankResult} `{p, iterations, residual, converged}` (see `pageRank`).
 * @throws {Error} when the restart index is unknown / out of range, or when
 *   neither `opts.transition` nor the global `GRAPH` exists.
 */
function personalizedPageRank(userId, alpha, opts) {
  const options = opts || {};
  const source = graphTransitionSource(options.transition);
  const n = source.kind === "dense" ? source.Q.length : graphNodeCountOf(source.graph);
  const seed = graphRestartIndex(userId, source);
  const restart = new Float64Array(n);
  restart[seed] = 1;
  return graphPropagate(alpha, restart, source, options);
}

/**
 * Batched Personalized PageRank: one propagation per seed, all seeds updated in
 * the same TensorFlow.js matrix operations.
 *
 * This is the evaluation-path optimization used by `script.js` (Recall@10 is
 * computed for hundreds of users). The single-seed `personalizedPageRank`
 * remains the reference implementation and is what the harness verifies; this
 * function applies exactly the same update, with `p` a `[nodes x seeds]` matrix.
 *
 * @param {number[]} seedNodes Node indices of the restart nodes (not raw ids).
 * @param {number} alpha Link-following probability in [0, 1].
 * @param {{tol?:number, maxIter?:number, transition?:TransitionSpec}} [opts]
 * @returns {{scores:Float32Array, nodeCount:number, columns:number,
 *            iterations:number, residual:number, converged:boolean}}
 *   `scores` is row-major `[node][column]`.
 */
function personalizedPageRankBatch(seedNodes, alpha, opts) {
  if (typeof tf === "undefined" || typeof tf.tensor2d !== "function") {
    throw new Error("TensorFlow.js is required for PPR but is not loaded");
  }
  const options = opts || {};
  const linkProbability = graphValidateAlpha(alpha);
  const limits = graphIterationOptions(options);
  const source = graphTransitionSource(options.transition);
  const n = source.kind === "dense" ? source.Q.length : graphNodeCountOf(source.graph);
  const seeds = Array.isArray(seedNodes) ? seedNodes : [];
  const columns = seeds.length;

  if (columns === 0) {
    return { scores: new Float32Array(0), nodeCount: n, columns: 0, iterations: 0, residual: 0, converged: true };
  }

  const seedData = new Float32Array(n * columns);
  for (let j = 0; j < columns; j++) {
    const seed = Number(seeds[j]);
    if (!Number.isInteger(seed) || seed < 0 || seed >= n) {
      throw new Error("personalizedPageRankBatch: seed node " + seeds[j] + " is out of range [0, " + n + ")");
    }
    seedData[seed * columns + j] = 1;
  }

  if (n === 0) {
    return { scores: seedData, nodeCount: 0, columns: columns, iterations: 0, residual: 0, converged: true };
  }

  const v = tf.tensor2d(seedData, [n, columns], "float32");
  let p = v.clone();
  let preparer = null;

  try {
    preparer = graphCreateBatchPreparer(source, v, linkProbability);

    // With no edges at all, every node is dangling and the restart matrix is
    // already the stationary distribution.
    if (source.kind === "graph") {
      const graph = source.graph;
      const nUsers = Number.isFinite(graph.nUsers) ? Number(graph.nUsers) : 0;
      const nMovies = Number.isFinite(graph.nMovies) ? Number(graph.nMovies) : 0;
      if (nUsers === 0 || nMovies === 0) {
        return {
          scores: Float32Array.from(seedData),
          nodeCount: n,
          columns: columns,
          iterations: 0,
          residual: 0,
          converged: true,
        };
      }
    }

    let iterations = 0;
    let residual = Infinity;
    let converged = false;

    while (iterations < limits.maxIter) {
      const next = tf.tidy(function () { return preparer.step(p); });
      const perColumn = tf.tidy(function () {
        return tf.sum(tf.abs(tf.sub(next, p)), 0).arraySync();
      });
      residual = 0;
      for (let j = 0; j < columns; j++) {
        const value = Number(perColumn[j]);
        if (value > residual) residual = value;
      }
      iterations += 1;
      p.dispose();
      p = next;
      if (residual < limits.tol) {
        converged = true;
        break;
      }
    }

    const flat = p.dataSync();
    return {
      scores: Float32Array.from(flat),
      nodeCount: n,
      columns: columns,
      iterations: iterations,
      residual: residual,
      converged: converged,
    };
  } finally {
    if (preparer) preparer.dispose();
    p.dispose();
    v.dispose();
  }
}

/**
 * Per-iteration TensorFlow.js step for the batched propagation.
 *
 * @param {{kind:string, Q:(number[][]|undefined), danglingMask:(ArrayLike<*>|undefined), graph:(Object|undefined)}} source
 * @param {tf.Tensor2D} v Restart matrix `[nodes x seeds]`.
 * @param {number} alpha
 * @returns {{step:function(tf.Tensor2D):tf.Tensor2D, dispose:function():void}}
 */
function graphCreateBatchPreparer(source, v, alpha) {
  if (source.kind === "dense") {
    const Q = source.Q;
    const n = Q.length;
    const maskData = new Float32Array(n);
    for (let i = 0; i < n; i++) maskData[i] = source.danglingMask && source.danglingMask[i] ? 1 : 0;
    const qTranspose = tf.transpose(tf.tensor2d(Q, [n, n], "float32"));
    const mask = tf.reshape(tf.tensor1d(maskData), [n, 1]);
    return {
      step: function (p) {
        const propagated = tf.matMul(qTranspose, p);
        const dangling = tf.sum(tf.mul(p, mask), 0);
        const base = dangling.mul(alpha).add(tf.scalar(1 - alpha));
        return propagated.mul(alpha).add(v.mul(base));
      },
      dispose: function () {
        qTranspose.dispose();
        mask.dispose();
      },
    };
  }

  const graph = source.graph;
  const R = graph.R;
  if (!R) throw new Error("personalizedPageRankBatch: the active graph has no R matrix");
  const nUsers = Number.isFinite(graph.nUsers) ? Number(graph.nUsers) : R.length;
  const nMovies = Number.isFinite(graph.nMovies)
    ? Number(graph.nMovies)
    : (R.length > 0 && R[0] ? R[0].length : 0);
  const n = nUsers + nMovies;

  // Reuse the same cached block tensors as the single-seed path.
  if (!GRAPH_BLOCK_CACHE || GRAPH_BLOCK_CACHE.graph !== graph) {
    const userDegree = new Float64Array(nUsers);
    const movieDegree = new Float64Array(nMovies);
    for (let u = 0; u < nUsers; u++) {
      const row = R[u] || [];
      for (let i = 0; i < nMovies; i++) {
        if (row[i]) {
          userDegree[u] += 1;
          movieDegree[i] += 1;
        }
      }
    }
    const aData = new Float32Array(nUsers * nMovies);
    const bData = new Float32Array(nMovies * nUsers);
    for (let u = 0; u < nUsers; u++) {
      const row = R[u] || [];
      const userScale = userDegree[u] > 0 ? 1 / userDegree[u] : 0;
      for (let i = 0; i < nMovies; i++) {
        if (!row[i]) continue;
        aData[u * nMovies + i] = userScale;
        bData[i * nUsers + u] = movieDegree[i] > 0 ? 1 / movieDegree[i] : 0;
      }
    }
    const maskData = new Float32Array(n);
    for (let u = 0; u < nUsers; u++) if (userDegree[u] === 0) maskData[u] = 1;
    for (let i = 0; i < nMovies; i++) if (movieDegree[i] === 0) maskData[nUsers + i] = 1;

    graphReleaseBlockCache();
    const a = tf.tensor2d(aData, [nUsers, nMovies], "float32");
    const b = tf.tensor2d(bData, [nMovies, nUsers], "float32");
    GRAPH_BLOCK_CACHE = {
      graph: graph,
      nUsers: nUsers,
      nMovies: nMovies,
      aTranspose: tf.transpose(a),
      bTranspose: tf.transpose(b),
      mask: tf.tensor1d(maskData),
    };
    a.dispose();
    b.dispose();
  }

  const cache = GRAPH_BLOCK_CACHE;
  const mask = tf.reshape(cache.mask, [n, 1]);
  return {
    step: function (p) {
      const userPart = p.slice([0, 0], [nUsers, p.shape[1]]);
      const moviePart = p.slice([nUsers, 0], [nMovies, p.shape[1]]);
      const nextMovie = tf.matMul(cache.aTranspose, userPart);
      const nextUser = tf.matMul(cache.bTranspose, moviePart);
      const propagated = tf.concat([nextUser, nextMovie], 0);
      const dangling = tf.sum(tf.mul(p, mask), 0);
      const base = dangling.mul(alpha).add(tf.scalar(1 - alpha));
      return propagated.mul(alpha).add(v.mul(base));
    },
    dispose: function () { mask.dispose(); },
  };
}

// ---------------------------------------------------------------------------
// Propagation helpers (private)
//
// The square adjacency is never materialized for the MovieLens path. `Q` has
// the bipartite form [[0, A], [B, 0]] with A = R row-normalized by user degree
// and B = R^T row-normalized by movie degree, so `Q^T p` is computed with the
// two off-diagonal blocks and `tf.matMul`.
// ---------------------------------------------------------------------------

/**
 * Resolve the transition source for a propagation call.
 *
 * @param {TransitionSpec|undefined} transition
 * @returns {{kind:("dense"|"graph"), Q:(number[][]|undefined), danglingMask:(ArrayLike<*>|undefined), graph:(Object|undefined)}}
 */
function graphTransitionSource(transition) {
  if (transition) {
    const Q = transition.Q;
    if (!Array.isArray(Q) || Q.length === 0) {
      throw new Error("pageRank: opts.transition.Q must be a non-empty square matrix");
    }
    return { kind: "dense", Q: Q, danglingMask: transition.danglingMask, graph: undefined };
  }
  const graph = typeof getGraph === "function" ? getGraph() : null;
  if (!graph) {
    throw new Error(
      "PageRank/PPR need an active graph: call buildGraph/setGraph first or pass opts.transition",
    );
  }
  return { kind: "graph", Q: undefined, danglingMask: undefined, graph: graph };
}

/**
 * Node count of a graph object.
 *
 * @param {Object} graph
 * @returns {number}
 */
function graphNodeCountOf(graph) {
  if (!graph) return 0;
  if (Number.isFinite(graph.nodeCount)) return Number(graph.nodeCount);
  const users = Array.isArray(graph.userIds) ? graph.userIds.length : 0;
  const movies = Array.isArray(graph.movieIds) ? graph.movieIds.length : 0;
  return users + movies;
}

/**
 * Uniform restart distribution over `n` nodes.
 *
 * @param {number} n
 * @returns {Float64Array}
 */
function graphUniformRestart(n) {
  const v = new Float64Array(n);
  if (n <= 0) return v;
  const value = 1 / n;
  for (let i = 0; i < n; i++) v[i] = value;
  return v;
}

/**
 * Resolve the PPR restart argument to a node index.
 *
 * With `opts.transition` the argument is a node index in `Q`'s own order (the
 * fixture contract). With the global MovieLens graph the caller (script.js's
 * adapter) passes a raw user id, so a known user id is mapped through
 * `userIndexOf`; a value that is not a known user id is treated as a node index.
 *
 * @param {number} seed
 * @param {{kind:string, Q:(number[][]|undefined), graph:(Object|undefined)}} source
 * @returns {number}
 */
function graphRestartIndex(seed, source) {
  const value = Number(seed);

  if (source.kind === "dense") {
    const n = source.Q.length;
    if (!Number.isInteger(value) || value < 0 || value >= n) {
      throw new Error("personalizedPageRank: restart node index " + seed + " is out of range [0, " + n + ")");
    }
    return value;
  }

  const graph = source.graph;
  const n = graphNodeCountOf(graph);
  if (graph.userIndexOf instanceof Map && graph.userIndexOf.has(value)) {
    return Number(graph.userIndexOf.get(value));
  }
  if (!Number.isInteger(value) || value < 0 || value >= n) {
    throw new Error("personalizedPageRank: unknown restart node/user " + seed + " for a graph with " + n + " nodes");
  }
  return value;
}

/**
 * Normalize caller options for the power iteration.
 *
 * @param {{tol?:number, maxIter?:number}} options
 * @returns {{tol:number, maxIter:number}}
 */
function graphIterationOptions(options) {
  const tolRaw = Number(options && options.tol);
  const iterRaw = Number(options && options.maxIter);
  const tol = Number.isFinite(tolRaw) && tolRaw > 0 ? tolRaw : 1e-6;
  const maxIter = Number.isFinite(iterRaw) && iterRaw >= 0 ? Math.floor(iterRaw) : 500;
  return { tol: tol, maxIter: maxIter };
}

/**
 * Validate alpha as a link-following probability.
 *
 * @param {number} alpha
 * @returns {number}
 */
function graphValidateAlpha(alpha) {
  const value = Number(alpha);
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error("alpha must be a link-following probability in [0, 1], got " + alpha);
  }
  return value;
}

/**
 * Build the per-iteration TensorFlow.js step for a transition source.
 *
 * Constant tensors are created here (outside `tf.tidy`) and disposed by the
 * returned `dispose`, so repeated runs cannot accumulate tensors.
 *
 * @param {{kind:string, Q:(number[][]|undefined), danglingMask:(ArrayLike<*>|undefined), graph:(Object|undefined)}} source
 * @param {tf.Tensor1D} v Restart distribution tensor.
 * @param {number} alpha
 * @returns {{step:function(tf.Tensor1D):tf.Tensor1D, dispose:function():void}}
 */
function graphCreatePreparer(source, v, alpha) {
  if (source.kind === "dense") {
    const Q = source.Q;
    const n = Q.length;
    for (let i = 0; i < n; i++) {
      if (!Q[i] || Q[i].length !== n) {
        throw new Error("pageRank: opts.transition.Q must be square; row " + i + " has the wrong length");
      }
    }
    const maskData = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      maskData[i] = source.danglingMask && source.danglingMask[i] ? 1 : 0;
    }
    const q = tf.tensor2d(Q, [n, n], "float32");
    const qTranspose = tf.transpose(q);
    const mask = tf.tensor1d(maskData);

    return {
      step: function (p) {
        const propagated = tf.matMul(qTranspose, p.reshape([n, 1])).reshape([n]);
        const dangling = tf.sum(tf.mul(p, mask));
        const base = tf.scalar(1 - alpha).add(dangling.mul(alpha));
        return propagated.mul(alpha).add(v.mul(base));
      },
      dispose: function () {
        q.dispose();
        qTranspose.dispose();
        mask.dispose();
      },
    };
  }

  const graph = source.graph;
  const R = graph.R;
  if (!R) throw new Error("pageRank: the active graph has no R matrix");
  const nUsers = Number.isFinite(graph.nUsers) ? Number(graph.nUsers) : R.length;
  const nMovies = Number.isFinite(graph.nMovies)
    ? Number(graph.nMovies)
    : (R.length > 0 && R[0] ? R[0].length : 0);
  const n = nUsers + nMovies;

  if (nUsers === 0 || nMovies === 0) {
    // Degenerate graph: no edges, everything is a dangling node.
    const maskData = new Float32Array(n).fill(1);
    const mask = tf.tensor1d(maskData);
    return {
      step: function (p) {
        const dangling = tf.sum(tf.mul(p, mask)).mul(alpha).add(1 - alpha);
        return v.mul(dangling);
      },
      dispose: function () { mask.dispose(); },
    };
  }

  // A[u][i] = 1/outdeg(u) on a positive edge, else 0 ; shape [nUsers, nMovies].
  // B[i][u] = 1/outdeg(movie i) on a positive edge, else 0 ; shape [nMovies, nUsers].
  const aData = new Float32Array(nUsers * nMovies);
  const bData = new Float32Array(nMovies * nUsers);
  const userDegree = new Float64Array(nUsers);
  const movieDegree = new Float64Array(nMovies);
  for (let u = 0; u < nUsers; u++) {
    const row = R[u] || [];
    for (let i = 0; i < nMovies; i++) {
      if (row[i]) {
        userDegree[u] += 1;
        movieDegree[i] += 1;
      }
    }
  }
  for (let u = 0; u < nUsers; u++) {
    const row = R[u] || [];
    const userScale = userDegree[u] > 0 ? 1 / userDegree[u] : 0;
    for (let i = 0; i < nMovies; i++) {
      if (!row[i]) continue;
      aData[u * nMovies + i] = userScale;
      bData[i * nUsers + u] = movieDegree[i] > 0 ? 1 / movieDegree[i] : 0;
    }
  }

  const maskData = new Float32Array(n);
  for (let u = 0; u < nUsers; u++) if (userDegree[u] === 0) maskData[u] = 1;
  for (let i = 0; i < nMovies; i++) if (movieDegree[i] === 0) maskData[nUsers + i] = 1;

  // The block tensors depend only on the graph, not on alpha or the seed, so
  // they are cached for the active graph. Rebuilding them per user would
  // dominate the cost of a full Recall@10 evaluation.
  if (!GRAPH_BLOCK_CACHE || GRAPH_BLOCK_CACHE.graph !== graph) {
    graphReleaseBlockCache();
    const a = tf.tensor2d(aData, [nUsers, nMovies], "float32");
    const b = tf.tensor2d(bData, [nMovies, nUsers], "float32");
    GRAPH_BLOCK_CACHE = {
      graph: graph,
      nUsers: nUsers,
      nMovies: nMovies,
      aTranspose: tf.transpose(a), // [nMovies, nUsers]
      bTranspose: tf.transpose(b), // [nUsers, nMovies]
      mask: tf.tensor1d(maskData),
    };
    a.dispose();
    b.dispose();
  }

  const cache = GRAPH_BLOCK_CACHE;
  return {
    step: function (p) {
      const userPart = p.slice([0], [nUsers]).reshape([nUsers, 1]);
      const moviePart = p.slice([nUsers], [nMovies]).reshape([nMovies, 1]);
      // Incoming mass: users -> movies via A^T, movies -> users via B^T.
      const nextMovie = tf.matMul(cache.aTranspose, userPart).reshape([nMovies]);
      const nextUser = tf.matMul(cache.bTranspose, moviePart).reshape([nUsers]);
      const propagated = tf.concat([nextUser, nextMovie], 0);
      const dangling = tf.sum(tf.mul(p, cache.mask));
      const base = tf.scalar(1 - alpha).add(dangling.mul(alpha));
      return propagated.mul(alpha).add(v.mul(base));
    },
    // Tensors are owned by GRAPH_BLOCK_CACHE and released when the graph changes.
    dispose: function () {},
  };
}

/**
 * Power iteration shared by global PageRank and PPR.
 *
 * p_0 = v ; p_next = alpha * (Q^T p + d * v) + (1 - alpha) * v, where d is the
 * mass currently on zero-degree nodes. Uses TensorFlow.js for every update.
 *
 * @param {number} alpha
 * @param {Float64Array} restart Restart distribution, length = node count.
 * @param {{kind:string, Q:(number[][]|undefined), danglingMask:(ArrayLike<*>|undefined), graph:(Object|undefined)}} source
 * @param {{tol?:number, maxIter?:number}} options
 * @returns {PageRankResult}
 */
function graphPropagate(alpha, restart, source, options) {
  if (typeof tf === "undefined" || typeof tf.tensor1d !== "function") {
    throw new Error("TensorFlow.js is required for PageRank/PPR but is not loaded");
  }
  const linkProbability = graphValidateAlpha(alpha);
  const limits = graphIterationOptions(options);
  const n = restart.length;

  if (n === 0) {
    return { p: new Float64Array(0), iterations: 0, residual: 0, converged: true };
  }

  const v = tf.tensor1d(Float32Array.from(restart), "float32");
  let p = tf.tensor1d(Float32Array.from(restart), "float32");
  let preparer = null;

  try {
    preparer = graphCreatePreparer(source, v, linkProbability);
    let iterations = 0;
    let residual = Infinity;
    let converged = false;

    while (iterations < limits.maxIter) {
      // Keep the next distribution on the backend (one transfer per iteration,
      // for the residual only) and read `p` once at the end.
      const next = tf.tidy(function () {
        return preparer.step(p);
      });
      residual = tf.tidy(function () {
        return tf.sum(tf.abs(tf.sub(next, p))).arraySync();
      });
      iterations += 1;
      p.dispose();
      p = next;
      if (residual < limits.tol) {
        converged = true;
        break;
      }
    }

    return {
      p: Float64Array.from(p.dataSync()),
      iterations: iterations,
      residual: residual,
      converged: converged,
    };
  } finally {
    if (preparer) preparer.dispose();
    p.dispose();
    v.dispose();
  }
}

// ---------------------------------------------------------------------------
// Small helpers (private, all graph-prefixed to avoid collisions)
// ---------------------------------------------------------------------------

/**
 * Readable message from an unknown thrown value.
 *
 * @param {*} error
 * @returns {string}
 */
function graphErrorMessage(error) {
  return String(error && error.message ? error.message : error);
}

/**
 * Escape text before inserting it into HTML (the local copy, so this file does
 * not depend on any helper defined by script.js).
 *
 * @param {*} text
 * @returns {string}
 */
function graphEscape(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Read alpha from the #alpha range input, falling back to 0.85.
 *
 * @returns {number}
 */
function graphReadAlpha() {
  const el = document.getElementById("alpha");
  const value = el ? Number(el.value) : Number.NaN;
  return Number.isFinite(value) ? value : 0.85;
}

/**
 * Map t in [0, 1] onto the single-hue sequential blue ramp.
 *
 * @param {number} t
 * @returns {string} an rgb(...) string.
 */
function graphRampColor(t) {
  const n = Number(t);
  const clamped = Math.max(0, Math.min(1, Number.isFinite(n) ? n : 0));
  const lo = graphTheme.rampLight;
  const hi = graphTheme.rampDark;
  const mix = (a, b, s) => Math.round(a + (b - a) * s);
  const r = mix(lo[0], hi[0], clamped);
  const g = mix(lo[1], hi[1], clamped);
  const b = mix(lo[2], hi[2], clamped);
  return "rgb(" + r + ", " + g + ", " + b + ")";
}

/**
 * Position lookup with a safe fallback for a missing node.
 *
 * @param {Map<number,{x:number,y:number}>} positions
 * @param {number} node
 * @returns {{x:number,y:number}}
 */
function graphPos(positions, node) {
  const pt = positions.get(node);
  return pt || { x: 0, y: 0 };
}

// ---------------------------------------------------------------------------
// Fixture demo helpers (private)
// ---------------------------------------------------------------------------

/**
 * Out-degree of a fixture node (number of outgoing reference edges).
 *
 * @param {number} index
 * @returns {number}
 */
function graphFixtureOutDegree(index) {
  let d = 0;
  for (const edge of REFERENCE_GRAPH.edges) {
    if (edge[0] === index) d += 1;
  }
  return d;
}

/**
 * Build the row-stochastic transition for the 4-node reference fixture.
 *
 * @returns {{P: number[][], out: number[]}} P[from][to] and the out-degrees.
 */
function graphFixtureTransition() {
  const n = REFERENCE_GRAPH.nodes.length;
  const out = new Array(n).fill(0);
  for (const edge of REFERENCE_GRAPH.edges) out[edge[0]] += 1;

  const P = [];
  for (let i = 0; i < n; i++) P.push(new Array(n).fill(0));
  for (const edge of REFERENCE_GRAPH.edges) {
    const from = edge[0];
    const to = edge[1];
    if (out[from] > 0) P[from][to] = 1 / out[from];
  }
  return { P, out };
}

/**
 * Restart distribution for the fixture: uniform, or one-hot at a named node.
 *
 * @param {string} restart "uniform" or one of A/B/C/D.
 * @returns {number[]}
 */
function graphFixtureRestartVector(restart) {
  const n = REFERENCE_GRAPH.nodes.length;
  const v = new Array(n).fill(0);
  if (!restart || restart === "uniform") {
    v.fill(1 / n);
    return v;
  }
  const index = REFERENCE_GRAPH.nodeOrder.indexOf(restart);
  if (index < 0) {
    v.fill(1 / n);
    return v;
  }
  v[index] = 1;
  return v;
}

/**
 * One plain-JavaScript propagation step on the 4-node fixture.
 *
 * The fixture has no dangling nodes, so the d * v term from readme.md
 * section 3.3 is not needed here. Returns the new distribution and the L1
 * change.
 *
 * @param {number} alpha
 * @param {string} restart
 * @param {number[]} pPrev
 * @returns {{p: number[], residual: number}}
 */
function graphFixturePropagate(alpha, restart, pPrev) {
  const transition = graphFixtureTransition();
  const P = transition.P;
  const n = REFERENCE_GRAPH.nodes.length;
  const v = graphFixtureRestartVector(restart);
  const pNext = new Array(n).fill(0);

  // p_next[j] = alpha * sum_i p[i] * P[i][j] + (1 - alpha) * v[j]
  for (let i = 0; i < n; i++) {
    const mass = pPrev[i];
    if (!mass) continue;
    for (let j = 0; j < n; j++) {
      if (P[i][j] !== 0) pNext[j] += alpha * mass * P[i][j];
    }
  }
  for (let j = 0; j < n; j++) pNext[j] += (1 - alpha) * v[j];

  let residual = 0;
  for (let j = 0; j < n; j++) residual += Math.abs(pNext[j] - pPrev[j]);
  return { p: pNext, residual };
}

// ---------------------------------------------------------------------------
// View builders (private)
// ---------------------------------------------------------------------------

/**
 * Build the drawable view for the 4-node fixture.
 *
 * @returns {{kind:string, nodes:Object[], edges:Object[], omitted:number,
 *            nodeCount:number, note:string}}
 */
function graphBuildFixtureView() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  const p = fixture.p || graphFixtureRestartVector(fixture.restart || "uniform");
  const nodes = REFERENCE_GRAPH.nodes.map((nd) => ({
    node: nd.index,
    key: "fixture:" + nd.label,
    label: nd.label,
    kind: nd.kind,
    id: nd.label,
    degree: graphFixtureOutDegree(nd.index),
    score: p[nd.index]
  }));
  const edges = REFERENCE_GRAPH.edges.map((edge) => ({
    source: edge[0],
    target: edge[1],
    directed: true
  }));
  return {
    kind: "fixture",
    nodes: nodes,
    edges: edges,
    omitted: 0,
    nodeCount: nodes.length,
    note: "4-node reference fixture (A, B, C, D) - no dangling nodes"
  };
}

/**
 * Neighbours of a node in the bipartite graph (users <-> movies), or [] when
 * the graph is missing. Uses the users-then-movies node order from `buildGraph`.
 *
 * @param {Graph} graph
 * @param {number} node
 * @returns {number[]}
 */
function graphNeighbors(graph, node) {
  const result = [];
  if (!graph || !graph.R) return result;
  if (graph.isUserNode(node)) {
    const row = graph.R[node];
    if (!row) return result;
    for (let i = 0; i < row.length; i++) {
      if (row[i]) result.push(graph.nUsers + i);
    }
  } else {
    const movieLocal = node - graph.nUsers;
    for (let u = 0; u < graph.nUsers; u++) {
      const row = graph.R[u];
      if (row && row[movieLocal]) result.push(u);
    }
  }
  return result;
}

/**
 * Build the capped ego subgraph for a seed user.
 *
 * Selection: the seed user's node plus its neighbours and their neighbours (a
 * 2-hop walk) in the bipartite graph, then apply the maxNodes cap in
 * deterministic node-index order. `omitted` is how many reachable nodes the cap
 * left out, so the legend can report it.
 *
 * @param {number|null} seedUserId
 * @param {function(number):number} [getScore]
 * @returns {{kind:string, nodes:Object[], edges:Object[], omitted:number,
 *            nodeCount:number, note:string}}
 */
function graphBuildEgoView(seedUserId, getScore) {
  const graph = GRAPH;
  if (!graph || !graph.nodeCount) {
    return {
      kind: "empty",
      nodes: [],
      edges: [],
      omitted: 0,
      nodeCount: 0,
      note: "No graph yet - build the graph (TODO(hw6) buildGraph) to draw the ego subgraph."
    };
  }

  const seed = seedUserId === null || seedUserId === undefined ? null : Number(seedUserId);
  if (seed === null || !graph.userIndexOf || !graph.userIndexOf.has(seed)) {
    return {
      kind: "empty",
      nodes: [],
      edges: [],
      omitted: 0,
      nodeCount: graph.nodeCount,
      note: "Pick a seed user (click a user node) to draw its ego subgraph."
    };
  }

  const maxNodes = Math.max(1, Math.floor(GRAPH_VIEW_STATE.maxNodes) || 120);
  const startNode = graph.nodeOfUser(seed);

  // Unbounded 2-hop reachable set, then cap deterministically.
  const reachable = new Set([startNode]);
  let frontier = [startNode];
  for (let depth = 0; depth < 2; depth++) {
    const next = [];
    for (const node of frontier) {
      for (const nb of graphNeighbors(graph, node)) {
        if (!reachable.has(nb)) {
          reachable.add(nb);
          next.push(nb);
        }
      }
    }
    frontier = next;
    if (reachable.size > 500000) break; // hard safety valve
  }

  const ordered = [startNode].concat(
    Array.from(reachable)
      .filter((node) => node !== startNode)
      .sort((a, b) => a - b)
  );
  const selected = new Set(ordered.slice(0, maxNodes));
  const omitted = reachable.size - selected.size;

  const nodes = [];
  for (const node of ordered) {
    if (!selected.has(node)) continue;
    const isUser = graph.isUserNode(node);
    const id = isUser ? graph.userIds[node] : graph.movieOfNode(node);
    let score = null;
    if (typeof getScore === "function") {
      const raw = Number(getScore(node));
      score = Number.isFinite(raw) ? raw : null;
    }
    nodes.push({
      node: node,
      key: "node:" + node,
      label: isUser ? "U" + id : "M" + id,
      kind: isUser ? "user" : "movie",
      id: id,
      degree: graph.degree ? graph.degree[node] : 0,
      score: score
    });
  }

  const edges = [];
  for (const a of nodes) {
    if (a.kind !== "user") continue;
    const uLocal = graph.userIndexOf.get(a.id);
    const row = graph.R ? graph.R[uLocal] : null;
    if (!row) continue;
    for (const b of nodes) {
      if (b.kind !== "movie") continue;
      const mLocal = graph.movieIndexOf.get(b.id);
      if (mLocal === undefined) continue;
      if (row[mLocal]) edges.push({ source: a.node, target: b.node, directed: false });
    }
  }

  return {
    kind: "ego",
    nodes: nodes,
    edges: edges,
    omitted: omitted,
    nodeCount: graph.nodeCount,
    note: "Ego subgraph of user " + seed + " - drawn " + nodes.length + " of " + graph.nodeCount + " nodes"
  };
}

/**
 * Deterministic two-column bipartite layout: users on the left, movies on the
 * right, evenly spaced top-to-bottom in ascending node order.
 *
 * @param {Object[]} nodes
 * @param {number} width
 * @param {number} height
 * @param {{top:number,right:number,bottom:number,left:number}} margin
 * @returns {{positions: Map<number,{x:number,y:number}>, leftX:number, rightX:number}}
 */
function graphLayoutTwoColumn(nodes, width, height, margin) {
  const positions = new Map();
  const usableWidth = width - margin.left - margin.right;
  const usableHeight = height - margin.top - margin.bottom;
  const leftX = margin.left + usableWidth * 0.2;
  const rightX = margin.left + usableWidth * 0.8;

  const place = (list, x) => {
    const n = list.length;
    list.forEach((item, i) => {
      const y = n <= 1 ? margin.top + usableHeight / 2 : margin.top + (usableHeight * i) / (n - 1);
      positions.set(item.node, { x: x, y: y });
    });
  };

  const users = nodes.filter((n) => n.kind === "user").sort((a, b) => a.node - b.node);
  const movies = nodes.filter((n) => n.kind === "movie").sort((a, b) => a.node - b.node);
  const others = nodes
    .filter((n) => n.kind !== "user" && n.kind !== "movie")
    .sort((a, b) => a.node - b.node);

  place(users, leftX);
  place(movies, rightX);
  place(others, leftX);

  return { positions: positions, leftX: leftX, rightX: rightX };
}

// ---------------------------------------------------------------------------
// d3 drawing (private)
// ---------------------------------------------------------------------------

/**
 * Draw one view (fixture or ego) into the #graph-view SVG element.
 *
 * Read-only with respect to GRAPH and any score arrays: it only reads nodes,
 * edges and score values. Zero-degree nodes are drawn as hollow rings with a
 * label so the dangling case is visible.
 *
 * @param {SVGElement} svgEl
 * @param {Object} view result of `graphBuildFixtureView` / `graphBuildEgoView`.
 * @param {{scoreAccessor: (function(number):number|null)|null, colorByScore: boolean}} opts
 * @returns {void}
 */
function graphDrawView(svgEl, view, opts) {
  const scoreAccessor = opts && typeof opts.scoreAccessor === "function" ? opts.scoreAccessor : null;
  const colorByScore = !!(opts && opts.colorByScore);
  const width = graphTheme.width;
  const height = graphTheme.height;
  const margin = graphTheme.margin;

  const svg = d3.select(svgEl);
  svg.selectAll("*").remove();
  svg.attr("viewBox", "0 0 " + width + " " + height);
  svg.attr("preserveAspectRatio", "xMidYMid meet");

  const defs = svg.append("defs");
  defs
    .append("marker")
    .attr("id", "graph-arrowhead")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 10)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", graphTheme.edgeColor);

  if (!view || view.nodes.length === 0) {
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .attr("font-size", 15)
      .attr("fill", graphTheme.noteColor)
      .text(view && view.note ? view.note : "Nothing to draw yet.");
    return;
  }

  const layout = graphLayoutTwoColumn(view.nodes, width, height, margin);
  const positions = layout.positions;

  svg
    .append("text")
    .attr("x", layout.leftX)
    .attr("y", margin.top - 12)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .attr("fill", graphTheme.noteColor)
    .text("users");
  svg
    .append("text")
    .attr("x", layout.rightX)
    .attr("y", margin.top - 12)
    .attr("text-anchor", "middle")
    .attr("font-size", 12)
    .attr("fill", graphTheme.noteColor)
    .text("movies");

  // Resolve scores and the shared scales.
  const scoreByNode = new Map();
  let maxScore = 0;
  for (const n of view.nodes) {
    let score = null;
    if (scoreAccessor) {
      const raw = Number(scoreAccessor(n.node));
      score = Number.isFinite(raw) ? raw : null;
    } else if (n.score !== null && n.score !== undefined) {
      const raw = Number(n.score);
      score = Number.isFinite(raw) ? raw : null;
    }
    scoreByNode.set(n.node, score);
    if (score !== null && score > maxScore) maxScore = score;
  }
  const maxDegree = view.nodes.reduce((m, n) => Math.max(m, Number(n.degree) || 0), 1);

  const normFor = (n) => {
    if (colorByScore) {
      const score = scoreByNode.get(n.node);
      return score !== null && maxScore > 0 ? Math.min(1, score / maxScore) : 0;
    }
    const deg = Math.max(0, Number(n.degree) || 0);
    return maxDegree > 0 ? Math.min(1, deg / maxDegree) : 0;
  };
  const radiusFor = (n) =>
    graphTheme.minRadius +
    (graphTheme.maxRadius - graphTheme.minRadius) * Math.sqrt(normFor(n));

  // Edges first so nodes sit on top.
  svg
    .append("g")
    .attr("class", "graph-edges")
    .selectAll("line")
    .data(view.edges)
    .join("line")
    .attr("x1", (d) => graphPos(positions, d.source).x)
    .attr("y1", (d) => graphPos(positions, d.source).y)
    .attr("x2", (d) => graphPos(positions, d.target).x)
    .attr("y2", (d) => graphPos(positions, d.target).y)
    .attr("stroke", graphTheme.edgeColor)
    .attr("stroke-width", 1)
    .attr("marker-end", (d) => (d.directed ? "url(#graph-arrowhead)" : null));

  const nodeSel = svg
    .append("g")
    .attr("class", "graph-nodes")
    .selectAll("g.graph-node")
    .data(view.nodes, (d) => d.key)
    .join("g")
    .attr("class", "graph-node")
    .attr("transform", (d) => {
      const pt = graphPos(positions, d.node);
      return "translate(" + pt.x + "," + pt.y + ")";
    })
    .style("cursor", "pointer");

  nodeSel.each(function eachNode(d) {
    const group = d3.select(this);
    const degree = Math.max(0, Number(d.degree) || 0);
    const isZero = degree === 0;
    const radius = radiusFor(d);
    const score = scoreByNode.get(d.node);

    let fill = d.kind === "user" ? graphTheme.userColor : graphTheme.movieColor;
    if (colorByScore) fill = graphRampColor(normFor(d));

    group
      .append("circle")
      .attr("r", radius)
      .attr("fill", isZero ? "#ffffff" : fill)
      .attr("stroke", isZero ? graphTheme.zeroStroke : graphTheme.nodeStroke)
      .attr("stroke-width", isZero ? 2.5 : 1.5);

    // Every fixture node is labelled; in the ego view only zero-degree nodes and
    // the seed user are labelled to keep a dense view readable.
    const showLabel =
      view.kind === "fixture" ||
      isZero ||
      (view.kind === "ego" && d.kind === "user" && d.id === GRAPH_VIEW_STATE.seedUserId);

    if (showLabel) {
      group
        .append("text")
        .attr("x", 0)
        .attr("y", -(radius + 4))
        .attr("text-anchor", "middle")
        .attr("font-size", 11)
        .attr("fill", graphTheme.labelColor)
        .attr("pointer-events", "none")
        .text(d.label);
    }

    group.on("mouseenter", (event) => graphShowTooltip(event, d, score));
    group.on("mousemove", (event) => graphShowTooltip(event, d, score));
    group.on("mouseleave", graphHideTooltip);
    group.on("click", () => graphHandleNodeClick(d, view));
  });
}

// ---------------------------------------------------------------------------
// Tooltip (private)
// ---------------------------------------------------------------------------

/**
 * Lazily create and return the single shared tooltip element.
 *
 * @returns {HTMLElement}
 */
function graphEnsureTooltip() {
  let el = document.getElementById(graphTheme.tooltipId);
  if (el) return el;
  el = document.createElement("div");
  el.id = graphTheme.tooltipId;
  el.setAttribute("role", "tooltip");
  el.style.position = "absolute";
  el.style.pointerEvents = "none";
  el.style.background = "#1c2128";
  el.style.color = "#ffffff";
  el.style.padding = "0.35rem 0.5rem";
  el.style.borderRadius = "6px";
  el.style.fontSize = "0.78rem";
  el.style.lineHeight = "1.35";
  el.style.zIndex = "10";
  el.style.whiteSpace = "nowrap";
  el.style.display = "none";
  document.body.appendChild(el);
  return el;
}

/**
 * Show the hover tooltip for a node: id, kind, degree and current score.
 *
 * @param {MouseEvent} event
 * @param {{id:(number|string), kind:string, node:number, degree:number}} d
 * @param {number|null} score
 * @returns {void}
 */
function graphShowTooltip(event, d, score) {
  const el = graphEnsureTooltip();
  const kind = d.kind === "user" ? "user" : "movie";
  const scoreText = score === null || score === undefined || !Number.isFinite(Number(score))
    ? "-"
    : Number(score).toFixed(6);
  el.textContent =
    kind + " " + d.id + "  |  node " + d.node + "  |  degree " + d.degree + "  |  score " + scoreText;
  el.style.display = "block";
  const left = (event && event.pageX ? event.pageX : 0) + 12;
  const top = (event && event.pageY ? event.pageY : 0) + 12;
  el.style.left = left + "px";
  el.style.top = top + "px";
}

/**
 * Hide the hover tooltip.
 *
 * @returns {void}
 */
function graphHideTooltip() {
  const el = document.getElementById(graphTheme.tooltipId);
  if (el) el.style.display = "none";
}

/**
 * Handle a click on a node: in the fixture mode it moves the restart node; in
 * the ego mode it selects a seed user. Never mutates GRAPH or any score array.
 *
 * @param {{label:string, kind:string, id:(number|string)}} d
 * @param {{kind:string}} view
 * @returns {void}
 */
function graphHandleNodeClick(d, view) {
  if (view.kind === "fixture") {
    GRAPH_VIEW_STATE.fixture.restart = d.label;
    fixtureReset();
    return;
  }
  if (view.kind === "ego" && d.kind === "user") {
    GRAPH_VIEW_STATE.seedUserId = Number(d.id);
    renderGraphView({ mode: "ego", seedUserId: Number(d.id) });
  }
}

// ---------------------------------------------------------------------------
// Legend and controls (private)
// ---------------------------------------------------------------------------

/**
 * A coloured square swatch for the legend.
 *
 * @param {string} color
 * @returns {string} HTML
 */
function graphSwatch(color) {
  return '<span class="legend-swatch" style="background:' + color + '"></span>';
}

/**
 * A hollow-ring swatch for the zero-degree legend entry.
 *
 * @returns {string} HTML
 */
function graphRingSwatch() {
  return (
    '<span class="legend-swatch" style="background:#ffffff;border:2px solid ' +
    graphTheme.zeroStroke +
    ';border-radius:50%"></span>'
  );
}

/**
 * A gradient swatch showing the single-hue sequential ramp.
 *
 * @returns {string} HTML
 */
function graphRampSwatch() {
  return (
    '<span class="legend-swatch" style="background:linear-gradient(90deg, ' +
    graphRampColor(0) +
    ", " +
    graphRampColor(1) +
    ')"></span>'
  );
}

/**
 * One legend row: swatch (may be empty) + text.
 *
 * @param {string} swatchHtml
 * @param {string} text
 * @returns {string} HTML
 */
function graphLegendItem(swatchHtml, text) {
  return (
    '<span class="legend-item">' + swatchHtml + "<span>" + graphEscape(text) + "</span></span>"
  );
}

/**
 * Replace the legend with a plain message (used for errors / unavailable d3).
 *
 * @param {string} message
 * @returns {void}
 */
function graphSetLegendMessage(message) {
  const target = document.getElementById("graph-legend");
  if (!target) return;
  target.innerHTML = '<p class="empty-state">' + graphEscape(message) + "</p>";
}

/**
 * Write one line per colour/size meaning into #graph-legend.
 *
 * @param {Object} view
 * @param {{colorByScore: boolean}} opts
 * @returns {void}
 */
function graphRenderLegend(view, opts) {
  const target = document.getElementById("graph-legend");
  if (!target) return;
  const colorByScore = !!(opts && opts.colorByScore);
  const items = [];

  if (view && view.kind === "fixture") {
    items.push(graphLegendItem(graphSwatch(graphTheme.userColor), "user node (A, B)"));
    items.push(graphLegendItem(graphSwatch(graphTheme.movieColor), "movie node (C, D)"));
    items.push(graphLegendItem(graphRampSwatch(), "fill = current probability mass (single-hue ramp)"));
    items.push(graphLegendItem(graphRingSwatch(), "hollow ring = zero-degree node"));
    items.push(graphLegendItem("", "arrow = link direction; this fixture has no dangling nodes"));
    items.push(graphLegendItem("", "radius = outgoing degree; click a node to restart there"));
    target.innerHTML = items.join("");
    return;
  }

  if (!view || view.kind === "empty") {
    const note = view && view.note ? view.note : "No graph to draw yet.";
    target.innerHTML = '<p class="empty-state">' + graphEscape(note) + "</p>";
    return;
  }

  items.push(graphLegendItem(graphSwatch(graphTheme.userColor), "user node (left column)"));
  items.push(graphLegendItem(graphSwatch(graphTheme.movieColor), "movie node (right column)"));
  if (colorByScore) {
    items.push(graphLegendItem(graphRampSwatch(), "fill = score (single-hue sequential ramp)"));
  } else {
    items.push(graphLegendItem("", "fill = node kind; switch colour-by to score to use the ramp"));
  }
  items.push(graphLegendItem(graphRingSwatch(), "hollow ring with label = zero-degree node"));
  items.push(
    graphLegendItem(
      "",
      colorByScore ? "radius scales with score" : "radius scales with degree"
    )
  );
  if (view.omitted > 0) {
    items.push(
      graphLegendItem(
        "",
        view.omitted + " node(s) omitted by the " + GRAPH_VIEW_STATE.maxNodes + "-node cap"
      )
    );
  }
  target.innerHTML = items.join("");
}

/**
 * Fill #graph-controls with the view-mode controls.
 *
 * @returns {void}
 */
function graphRenderControls() {
  const target = document.getElementById("graph-controls");
  if (!target) return;
  const state = GRAPH_VIEW_STATE;

  const modeOptions = ["fixture", "ego"]
    .map((m) => '<option value="' + m + '"' + (state.mode === m ? " selected" : "") + ">" + m + "</option>")
    .join("");
  const colorOptions = ["degree", "score"]
    .map((c) => '<option value="' + c + '"' + (state.colorBy === c ? " selected" : "") + ">" + c + "</option>")
    .join("");
  const restart = state.fixture.restart || "uniform";
  const restartOptions = ["uniform"].concat(REFERENCE_GRAPH.nodeOrder)
    .map((r) => '<option value="' + r + '"' + (restart === r ? " selected" : "") + ">" + r + "</option>")
    .join("");

  const seedText = state.seedUserId === null ? "- (click a user node)" : String(state.seedUserId);

  target.innerHTML =
    '<label class="legend-item">View <select id="graph-mode-select">' + modeOptions + "</select></label>" +
    '<label class="legend-item">Colour by <select id="graph-colorby-select"' +
    (state.mode === "fixture" ? " disabled" : "") +
    ">" + colorOptions + "</select></label>" +
    '<label class="legend-item">Fixture restart <select id="graph-restart-select"' +
    (state.mode === "fixture" ? "" : " disabled") +
    ">" + restartOptions + "</select></label>" +
    '<span class="hint">Seed user: ' + graphEscape(seedText) + "</span>";

  const modeSelect = document.getElementById("graph-mode-select");
  if (modeSelect) {
    modeSelect.addEventListener("change", () => renderGraphView({ mode: modeSelect.value }));
  }
  const colorSelect = document.getElementById("graph-colorby-select");
  if (colorSelect) {
    colorSelect.addEventListener("change", () => renderGraphView({ colorBy: colorSelect.value }));
  }
  const restartSelect = document.getElementById("graph-restart-select");
  if (restartSelect) {
    restartSelect.addEventListener("change", () => {
      GRAPH_VIEW_STATE.fixture.restart = restartSelect.value;
      fixtureReset();
    });
  }
}

// ---------------------------------------------------------------------------
// Graph summary (private)
// ---------------------------------------------------------------------------

/**
 * Numeric value of a count-or-list field (arrays report their length).
 *
 * @param {*} value
 * @returns {number}
 */
function graphCount(value) {
  if (Array.isArray(value)) return value.length;
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/**
 * Render the #graph-summary-body panel: nodes, edges and zero-degree counts.
 *
 * @param {Graph|null} graph
 * @returns {void}
 */
function graphRenderSummary(graph) {
  const target = document.getElementById("graph-summary-body");
  if (!target) return;
  if (!graph) {
    target.innerHTML =
      '<p class="empty-state">No graph yet. Press "Build graph" to report nodes, edges, ' +
      "zero-degree users, zero-degree movies and positive training interactions.</p>";
    return;
  }
  const nUsers = graphCount(graph.nUsers);
  const nMovies = graphCount(graph.nMovies);
  const nodeCount = graphCount(graph.nodeCount) || nUsers + nMovies;
  const positiveEdges = graphCount(graph.positiveEdges);
  const zeroUsers = graphCount(graph.zeroDegreeUsers);
  const zeroMovies = graphCount(graph.zeroDegreeMovies);

  target.innerHTML =
    '<dl class="summary-list">' +
    "<dt>User nodes</dt><dd>" + nUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Movie nodes</dt><dd>" + nMovies.toLocaleString("en-US") + "</dd>" +
    "<dt>Nodes (no padding node)</dt><dd>" + nodeCount.toLocaleString("en-US") + "</dd>" +
    "<dt>Positive training edges</dt><dd>" + positiveEdges.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree users</dt><dd>" + zeroUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree movies</dt><dd>" + zeroMovies.toLocaleString("en-US") + "</dd>" +
    "</dl>";
}

// ---------------------------------------------------------------------------
// Graph view (public: renderGraphView)
// ---------------------------------------------------------------------------

/**
 * Render the graph view into #graph-view, #graph-legend and #graph-controls.
 *
 * Merges `opts` into the module state, builds either the deterministic 4-node
 * fixture view or the capped ego subgraph of the seed user, and draws it with
 * d3 v7 in a two-column bipartite layout. The view is read-only: it never
 * modifies GRAPH or any score array.
 *
 * Degrades gracefully: with no d3, no graph, or no seed user it writes a plain
 * message instead of throwing, and an unimplemented `buildGraph` stub simply
 * leaves GRAPH null (the fixture demo still draws).
 *
 * @param {GraphViewOptions} [opts]
 * @returns {(Object|null)} a small description of the last render, or null.
 */
function renderGraphView(opts) {
  const o = opts || {};
  if (o.mode === "fixture" || o.mode === "ego") GRAPH_VIEW_STATE.mode = o.mode;
  if (o.seedUserId !== undefined) {
    GRAPH_VIEW_STATE.seedUserId = o.seedUserId === null ? null : Number(o.seedUserId);
  }
  if (o.colorBy === "degree" || o.colorBy === "score") GRAPH_VIEW_STATE.colorBy = o.colorBy;
  if (o.maxNodes !== undefined && o.maxNodes !== null && Number.isFinite(Number(o.maxNodes))) {
    GRAPH_VIEW_STATE.maxNodes = Math.max(1, Math.floor(Number(o.maxNodes)));
  }

  const svgEl = document.getElementById("graph-view");
  if (!svgEl) return null;
  if (typeof d3 === "undefined") {
    graphSetLegendMessage("d3 is not available, so the graph view cannot be drawn. Check that vendor/d3.min.js loaded.");
    return null;
  }

  const mode = GRAPH_VIEW_STATE.mode;
  const external = typeof o.getScore === "function" ? o.getScore : null;
  let view;
  let scoreAccessor = null;
  let colorByScore = false;

  if (mode === "fixture") {
    view = graphBuildFixtureView();
    scoreAccessor = (node) => {
      const p = GRAPH_VIEW_STATE.fixture.p;
      return p ? p[node] : 0;
    };
    colorByScore = true;
  } else {
    view = graphBuildEgoView(GRAPH_VIEW_STATE.seedUserId, external);
    if (external && GRAPH_VIEW_STATE.colorBy === "score") {
      scoreAccessor = external;
      colorByScore = true;
    }
  }

  try {
    graphDrawView(svgEl, view, { scoreAccessor: scoreAccessor, colorByScore: colorByScore });
  } catch (error) {
    try {
      d3.select(svgEl).selectAll("*").remove();
    } catch (ignored) {
      /* the SVG may not be an element we can clear */
    }
    graphSetLegendMessage("Graph view could not be drawn: " + graphErrorMessage(error));
  }

  graphRenderLegend(view, { colorByScore: colorByScore });
  graphRenderControls();

  GRAPH_VIEW_STATE.lastRender = {
    mode: mode,
    seedUserId: GRAPH_VIEW_STATE.seedUserId,
    colorByScore: colorByScore,
    nodeCount: view.nodes.length,
    edgeCount: view.edges.length,
    omitted: view.omitted || 0,
    note: view.note || ""
  };
  return GRAPH_VIEW_STATE.lastRender;
}

// ---------------------------------------------------------------------------
// Fixture teaching demo (public: fixtureStep / fixtureReset)
//
// THIS DEMO EXISTS ONLY FOR THE 4-NODE FIXTURE. It is a tiny plain-JavaScript
// propagation so the animation works before you write any stub, and it is NOT a
// template for the MovieLens implementation. The MovieLens propagation must use
// TensorFlow.js tensor operations on R and R-transpose (readme.md section 3.5),
// as required by pageRank / personalizedPageRank.
// ---------------------------------------------------------------------------

/**
 * Update #convergence-results with the fixture iteration and its L1 residual.
 *
 * @returns {void}
 */
function graphRenderConvergence() {
  const target = document.getElementById("convergence-results");
  if (!target) return;
  const fixture = GRAPH_VIEW_STATE.fixture;
  const p = fixture.p || [];
  const values = p
    .map((value, i) => REFERENCE_GRAPH.nodeOrder[i] + " " + Number(value).toFixed(10))
    .join(", ");
  const residual = fixture.residual === null ? "-" : Number(fixture.residual).toExponential(3);

  target.innerHTML =
    "<p><strong>Fixture demo (4-node reference graph)</strong> - iteration " +
    fixture.iteration +
    ", alpha " +
    Number(fixture.alpha !== undefined ? fixture.alpha : graphReadAlpha()).toFixed(2) +
    ", restart " +
    graphEscape(fixture.restart || "uniform") +
    ".</p>" +
    '<dl class="summary-list">' +
    "<dt>L1 residual (last step)</dt><dd>" + graphEscape(residual) + "</dd>" +
    "<dt>Current p</dt><dd>" + graphEscape(values) + "</dd>" +
    "</dl>" +
    '<p class="hint">This plain-JavaScript demo exists only for the 4-node fixture and is not a ' +
    "template for the MovieLens implementation, which must use TensorFlow.js tensor operations " +
    "on R and R-transpose (readme.md section 3.5).</p>";
}

/**
 * Advance the 4-node fixture demo by one propagation step and redraw it.
 *
 * @returns {void}
 */
function fixtureStep() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  const alpha = graphReadAlpha();
  if (!fixture.p) {
    fixture.p = graphFixtureRestartVector(fixture.restart || "uniform");
    fixture.iteration = 0;
  }
  const result = graphFixturePropagate(alpha, fixture.restart || "uniform", fixture.p);
  fixture.p = result.p;
  fixture.iteration += 1;
  fixture.residual = result.residual;
  fixture.alpha = alpha;
  GRAPH_VIEW_STATE.mode = "fixture";
  renderGraphView({ mode: "fixture" });
  graphRenderConvergence();
}

/**
 * Reset the fixture demo to iteration 0 (the restart distribution) and redraw.
 *
 * @returns {void}
 */
function fixtureReset() {
  const fixture = GRAPH_VIEW_STATE.fixture;
  fixture.alpha = graphReadAlpha();
  fixture.iteration = 0;
  fixture.p = graphFixtureRestartVector(fixture.restart || "uniform");
  fixture.residual = null;
  GRAPH_VIEW_STATE.mode = "fixture";
  renderGraphView({ mode: "fixture" });
  graphRenderConvergence();
}

// ---------------------------------------------------------------------------
// Bootstrap (private)
// ---------------------------------------------------------------------------

/**
 * Wire the fixture step/reset buttons and draw the fixture demo once.
 *
 * @returns {void}
 */
function graphInitGraphView() {
  const stepButton = document.getElementById("step-btn");
  if (stepButton) stepButton.addEventListener("click", fixtureStep);
  const resetButton = document.getElementById("reset-steps");
  if (resetButton) resetButton.addEventListener("click", fixtureReset);
  fixtureReset();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", graphInitGraphView);
} else {
  graphInitGraphView();
}
