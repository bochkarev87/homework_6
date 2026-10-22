# HW6 — Graph-Based Recommendations with Personalized PageRank

Submission for the LLM4Rec course assignment. The original assignment text is
preserved in [`assignment.md`](assignment.md); this file documents the submitted
implementation and how to reproduce its outputs.

## Status

- All seven `TODO(hw6)` stubs are implemented.
- Self-checks: **12 passed, 0 failed, 0 pending**.
- Four-method TEST comparison and the PPR validation sweep completed.
- No run hit the 500-iteration cap; all reported propagation runs converged
  below the L1 tolerance `1e-6`.
- The MF comparison uses a 16-factor model at validation checkpoint epoch 4
  (validation RMSE `0.9779`).

## Requirements

| Component | Version used |
|---|---|
| Browser | Google Chrome `155.0.8059.85` (headless for the recorded run) |
| TensorFlow.js | `4.22.0`, loaded from jsDelivr CDN |
| D3 | `7.9.0`, vendored at `vendor/d3.min.js` |
| Backend | `webgl` (ANGLE Direct3D11, NVIDIA GeForce RTX 5080) |
| Dataset | MovieLens 100K (`u.data`, `u.item`) copied from `week5/` |

No build step is required. The page needs network access for the TensorFlow.js
CDN, and the data layer uses `fetch`, so serve the directory over HTTP rather
than opening `index.html` from `file://`.

## Run

```text
cd week6
python3 -m http.server 8000
```

Open `http://localhost:8000/index.html`. Any static HTTP server works; the
application has no server-side component.

If Python is unavailable, use another static server, for example
`npx serve .` (Node.js) or the Live Server extension in VS Code. Do not rely on
`file://`: Chrome only loads the data files there with special flags, and the
assignment's documented run path is HTTP.

## Reproduce the reported results

1. Wait for `TensorFlow.js backend: webgl` and the dataset summary.
2. Press **Build graph**. Expected graph: 2,554 nodes (943 users + 1,611
   movies), 45,602 positive training edges, 1 zero-degree user and 253
   zero-degree movies, no padding node.
3. Press **Run tests**. Expected: `12 passed, 0 failed, 0 pending`.
4. Press **Run alpha sweep (validation)**. This evaluates PPR at
   `0.50, 0.70, 0.85, 0.95` on the 846-user validation cohort and selects the
   best alpha.
5. Press **Evaluate Recall@10 — all four methods**. This selects the best alpha
   (`0.50`), then evaluates popularity, global PageRank, PPR and MF on the
   shared 861-user TEST cohort.
6. Press **Run two-user demo (alpha / seed)** to reproduce the three
   one-factor-at-a-time settings.
7. Use the three **Export** buttons to download `verification.json`,
   `results.csv` and `recommendations.csv`.

The checked-in exports were produced by the submitted code with the same flow.
The evaluation step also records one list per method for both demo users at the
selected alpha, so `recommendations.csv` contains all four methods. Timings are
single-run evidence on the machine above and must not be compared across
devices.

## Results

Selected alpha: **0.50** (validation Recall@10).

| Method | Alpha | TEST Recall@10 | Mean recommended-list popularity | Iterations |
|---|---:|---:|---:|---:|
| Popularity | — | 0.0760 | 272.0 | — |
| Global PageRank | 0.50 | 0.0797 | 257.8 | 19 |
| PPR | 0.50 | **0.1122** | 244.2 | 22 |
| MF (checkpoint 4) | — | 0.0312 | 94.3 | — |

PPR's macro Recall@10 is 47.6% above popularity and 259.0% above MF on the same
861 users. The mean popularity column is descriptive only: a less popular list
is not automatically a better or more novel recommendation.

Validation alpha sweep (846 users):

| Alpha | Recall@10 |
|---:|---:|
| 0.50 | 0.1145 |
| 0.70 | 0.1048 |
| 0.85 | 0.0900 |
| 0.95 | 0.0738 |

Shared TEST cohort and exclusions:

- 943 candidate users; 861 evaluable; 82 users excluded.
- 13 relevant movie IDs in the test split are absent from the 1,611-movie
  training candidate pool: `913, 1106, 1130, 1458, 1498, 1525, 1613, 1625,
  1645, 1646, 1650, 1651, 1674`.
- One user has no positive training edge. Warm-user evaluation only.

PPR at the selected alpha converged in 22 iterations with residual
`5.01e-7`; global PageRank converged in 19 iterations with residual `5.47e-7`.
The batched cohort path was cross-checked against the single-seed reference:
at alpha `0.85`, 94 vs 91 iterations and maximum absolute score difference
`3.96e-9`.

## Two-user demonstration

Users 1 and 2 are fixed across settings.

| Setting | User | Alpha | Mean list popularity |
|---|---:|---:|---:|
| Baseline | 1 | 0.85 | 245.5 |
| Change seed only | 2 | 0.85 | 254.9 |
| Change alpha only | 1 | 0.50 | 228.4 |

Overlap with the baseline Top-10: changing only the seed gives **0/10** shared
movies; changing only alpha gives **7/10**. Across methods, the two users share
1/10 movies for popularity, 1/10 for PageRank, 1/10 for PPR and 4/10 for MF.

Training histories:

- User 1: 272 ratings total; 217 train, 27 validation, 28 test; 131 positive
  training ratings. Top positive-training genres: Drama 61, Comedy 39, Action
  36, Sci-Fi 27, Thriller 25.
- User 2: 62 ratings total; 49 train, 6 validation, 7 test; 32 positive training
  ratings. Top positive-training genres: Drama 18, Romance 11, Comedy 11,
  Action 5, Crime 5.

Exact Top-10 lists, including titles, genres, scores and training popularity,
are in [`recommendations.csv`](recommendations.csv).

## Implementation notes

- The graph universe is exactly the observed training users and movies. User
  nodes occupy indices `0..nUsers-1`; movie nodes follow. There is no padding
  node.
- Only training ratings `>= 4.0` become edges. Each user row is
  row-normalized over positively rated movies, and each movie row over users
  who rated it positively.
- Propagation uses the two off-diagonal blocks, not a dense `2554 x 2554`
  matrix. Dangling mass is redistributed with the same restart vector in
  global PageRank and PPR; dangling rows are never divided by zero.
- The stopping rule is L1 difference `< 1e-6` or 500 iterations. Iteration
  counts, residuals and convergence flags are exported, never inferred from the
  cap alone.
- PPR is computed once per `(graph, alpha, split)` for the whole evaluation
  cohort with a single batched TensorFlow.js propagation, then reused for each
  user's Top-10. The single-user API remains available for panels and checks.
- `mfScores` keeps the previous assignment's scoring definition: dot product
  plus global training mean, ranked as predicted ratings. MF scores are not
  probabilities and are not compared numerically with PPR scores.

### MF regularization disclosure

The starter MF loop summed L2 over every factor row and divided only by the
matrix size implicitly through Adam, which collapsed all embeddings toward zero
within the first epoch: validation RMSE stayed at `1.1830` and checkpoint 1 was
selected. The loop now uses the week-5 objective, averaging L2 over the batch
rows (`sum(u^2 + i^2) * reg / batchSize`). With that correction, validation RMSE
improves to `0.9779` at epoch 4. This changes regularization, not the MF scoring
rule or the validation-checkpoint rule.

## Files

| File | Purpose |
|---|---|
| `index.html`, `style.css` | Application shell and presentation |
| `data.js` | MovieLens parsing, chronological split, candidate metadata |
| `graph.js` | Graph construction, normalization, PageRank, PPR, batched PPR |
| `script.js` | Scoring, Top-10 filtering, Recall@10, evaluation, exports |
| `vendor/d3.min.js` | Local D3 7.9.0 |
| `u.data`, `u.item` | MovieLens 100K source files |
| `verification.json` | Machine-readable self-check and environment evidence |
| `results.csv` | TEST comparison plus validation alpha sweep |
| `recommendations.csv` | Top-10 lists for the demo users and all four methods |
| `report.pdf` | Implementation evidence and the business analysis |
| `report.html` | Printable source of `report.pdf` |
| `assignment.md` | Original assignment specification, preserved |

## Verification

`verification.json` contains 12 PASS checks covering the two reference-graph
vectors, row normalization, restart distributions, probability conservation,
nonnegative scores, one-extra-update stability, dangling-node mass, matrix
orientation, Top-10 filtering/ties, node mapping and the public API. It also
records the batched-vs-single PPR cross-check, MF settings, backend, dependency
versions and single-run timings. The report also documents a hand-computed
one-step PageRank check on the four-node fixture, with the console command and
the expected vector.

## Limitations

- The split is chronological within each user, not globally time-consistent;
  MovieLens timestamps record rating times, not viewing sessions.
- Evaluation is warm-user only. Results say nothing about cold start.
- Recall@10 measures retrieval against held-out `>= 4.0` ratings. It does not
  measure click-through, completion, satisfaction, retention or revenue.
- MF and PPR use different validation objectives (validation RMSE vs
  validation Recall@10). Their TEST recalls are descriptive, not a controlled
  test of one algorithm's superiority.
- Timings are from one headless WebGL run on the stated GPU.

## Manual steps before submission

1. Add your name and student details to the title block of
   `report.html`, then regenerate `report.pdf`, if the course requires them.
2. Confirm the submission directory, deadline, report format and AI-session
   export requirement with the instructor. Do not rely on this repository for
   those administrative details.
3. Export the unmodified AI session with the course-required tool and add it to
   the submission if still required.
4. Re-run the application once in your own browser and keep the exported
   `verification.json`, `results.csv` and `recommendations.csv` if the course
   requires machine-specific evidence.

## Dataset citation

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. *ACM Transactions on Interactive Intelligent Systems* 5, 4,
Article 19. Dataset: MovieLens 100K, GroupLens, University of Minnesota.
