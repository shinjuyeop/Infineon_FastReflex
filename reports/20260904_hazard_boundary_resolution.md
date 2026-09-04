# Hazard Boundary Resolution Cycle

## Starting state

- Repository and branch: `/d/shin/Infineon_FastReflex`, `main`.
- Starting `HEAD == origin/main`: `f32f6056f23ba05ed447fff7771ee63f789be7b6`.
- The tracked worktree was clean and the expected base was exact.
- Historical Generalization HOLDOUT guard and scientific open count were both 1. No historical HOLDOUT payload was read.
- The prior `FACTOR_VALIDATION` state was `CONSUMED_DEVELOPMENT_VALIDATION`. Its payload was not reopened and its reference/candidate inference was not repeated.

## Scientific boundary and previous failure

The historical V2 status remains `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`; whole-simulation evidence remains `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`; Support simulation evidence remains `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`.

The failed candidate `model_v2_factor_conditioned_gru20_20260904` remains `FROZEN_FAILED_DEVELOPMENT_EVIDENCE`. Saved results showed that it improved strict-Sand specificity from 44/46 to 46/46 and reduced adverse Sand responses from 17/46 to 2/46, but reduced fresh Support from 16/18 to 8/18. The saved V2_VALIDATION regression similarly changed Support from 30/30 to 17/30 and right Support from 12/12 to 4/12. This cycle treated those validation artifacts as immutable summaries only.

## Failure audit

The audit used only saved artifacts, reference/failed checkpoints, and Unified TRAIN + V2_TRAIN + FACTOR_TRAIN. It performed no simulation or optimizer step. The effective prior partition contains 109 actually run-disjoint diagnostic runs with zero fit-run overlap. Seven delayed-Support monitor runs are separately labeled endpoint-disjoint-only because the earlier anchor-refined recipe exposed positive anchors from every delayed-Support TRAIN run; they are not misrepresented as run-disjoint evidence.

| Diagnostic | Reference V2 | Failed factor candidate | Interpretation |
|---|---:|---:|---|
| Round-3 TRAIN endpoints | Different frozen source population | 49,987 negative / 2,810 positive | raw imbalance alone is not causal evidence |
| Effective class loss mass | .50 negative / .50 positive | .50 negative / .50 positive | inverse-frequency CE balanced classes, not physical domains |
| Factor-Sand median maximum probability | .855543 | .691229 | the failed candidate learned Sand suppression |
| Factor ordinary-Support median maximum probability | .998704 | .991881 | shift across the `.99`/5 ms operating boundary |
| Factor delayed-Support median maximum probability | .998767 | .996209 | probability remains high but persistence/onset is lost |
| Factor Support decision | 6/6 | 3/6 | ordinary and delayed controls both degrade |
| Right-Support decision | 9/9 | 4/9 | collapse is not left-side-localized |
| Final-hidden ridge / 1NN balanced accuracy | 1.0000 / 1.0000 | 1.0000 / 1.0000 | recurrent body preserves simple geometry |
| Final-hidden centroid separation | 2.0141 | 2.0519 | no hidden-state separation collapse |
| Logit ridge balanced accuracy | .8333 | .8333 | logit plane retains diagnostic information |
| Frozen-head balanced accuracy at the anchor | .5000 | .5000 | fixed head/operating decision does not recover Support |
| Seed agreement at Round 3 | reference ensemble 6/6 factor Support | ensemble 3/6; seeds 2/6, 6/6, 6/6 | ensemble boundary and one member dominate the failure |
| Round 0/1/2/3 factor Support | reference 6/6 | 5/6 → 6/6 → 5/6 → 3/6 | degradation starts before HNM and strengthens later |

### Optimization pressure

Inverse-frequency CE assigned exactly half of expected loss mass to each class in every round, but it did not balance domains within either class. The table reports total expected objective mass; positive and negative rows each sum to 0.5.

| Domain | Round 0 endpoints / mass | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|---:|
| Fresh Sand negative | 6,368 / .0934 | 7,304 / .0929 | 8,237 / .0922 | 9,165 / .0917 |
| Fresh Support pre-I1 negative | 2,231 / .0327 | 2,560 / .0326 | 2,903 / .0325 | 3,234 / .0323 |
| Historical benign negative | 10,803 / .1584 | 12,528 / .1593 | 14,244 / .1594 | 15,937 / .1594 |
| Historical Hazard-precursor negative | 14,704 / .2156 | 16,920 / .2152 | 19,288 / .2159 | 21,651 / .2166 |
| Slip positive | 1,680 / .2989 | unchanged | unchanged | unchanged |
| Ordinary-Support positive | 800 / .1423 | unchanged | unchanged | unchanged |
| Delayed-Support positive | 330 / .0587 | unchanged | unchanged | unchanged |

The actual fit population grew from 34,106 negative + 2,810 positive endpoints at Round 0 to 49,987 + 2,810 at Round 3. Within the positive half of the loss, Slip received 59.8%, ordinary Support 28.5%, and delayed Support 11.7%. Fresh Support pre-I1 negatives received about 6.5% of the negative half, while Support-positive subdomains were not protected as distinct objective strata. Uniform endpoint shuffling preserved this imbalance.

### HNM progression

Each HNM pass selected 6,888 endpoint records. Only 5,206, 5,360, and 5,315 were newly materialized in rounds 1–3; 1,682, 1,528, and 1,573 selected identities were already in the materialized negative population. This is material duplicate pressure, but HNM is not the sole cause because the Round-0 checkpoint already showed partial Support loss and Round 1 temporarily recovered it.

| Checkpoint | Factor Sand correct | Ordinary Support | Delayed Support | Total factor Support | Right Support |
|---|---:|---:|---:|---:|---:|
| Reference V2 | 18/18 | 4/4 | 2/2 | 6/6 | 9/9 |
| Failed Round 0 | 16/18 | 4/4 | 1/2 | 5/6 | 9/9 |
| Failed Round 1 | 16/18 | 4/4 | 2/2 | 6/6 | 9/9 |
| Failed Round 2 | 18/18 | 3/4 | 2/2 | 5/6 | 6/9 |
| Failed Round 3 | 18/18 | 2/4 | 1/2 | 3/6 | 4/9 |

The frozen collapse-stage classification is `before_hnm`: a material reference-to-Round-0 drop was already present. Later HNM rounds strengthened Sand suppression and contributed to the final trade-off, but a pure `HNM_BOUNDARY_OVERCORRECTION` classification is not supported.

### Seed consistency

The final TRAIN-only diagnostic is not an all-seed collapse. Seed 20260828 suppressed Support most strongly, while the other two members retained these small TRAIN diagnostic controls. Ensemble degradation is nevertheless real and matches the separately saved validation failure.

| Round-3 member | Sand | Ordinary Support | Delayed Support | Right Support |
|---|---:|---:|---:|---:|
| Ensemble | 18/18 | 2/4 | 1/2 | 4/9 |
| 20260828 | 18/18 | 2/4 | 0/2 | 2/9 |
| 20260829 | 14/18 | 4/4 | 2/2 | 9/9 |
| 20260830 | 18/18 | 4/4 | 2/2 | 9/9 |

The ensemble median maximum probabilities were `.691229` for Sand, `.991881` for ordinary Support, and `.996209` for delayed Support. The values cluster around the fixed `.99/5 ms` operating boundary, explaining why averaging and persistence expose the loss/objective trade-off even when individual members differ.

### 80D, hidden, head, and temporal representation

| Diagnostic | Reference V2 | Failed Round 3 | Interpretation |
|---|---:|---:|---|
| Frozen 20x80 input ridge-probe balanced accuracy | .8056 | same input | adequate but not perfect raw causal separation |
| Current-80D ridge-probe balanced accuracy | .8889 | same input | current causal input is informative |
| Final hidden ridge-probe balanced accuracy | 1.0000 | 1.0000 | simple decision geometry exists in the recurrent state |
| Final hidden 1NN accuracy | 1.0000 | 1.0000 | no hidden-space collapse on the diagnostic set |
| Final hidden centroid separation | 2.0141 | 2.0519 | failed body preserves at least reference-level separation |
| Logit ridge-probe balanced accuracy | .8333 | .8333 | logit plane still contains separable structure |
| Frozen head decision at `.99` balanced accuracy | .5000 | .5000 | the fixed operating decision does not recover Support at the anchor |
| Failed hidden-probe minus head gap | — | .5000 | material head/objective mismatch |

Temporal hidden separation is already above the predeclared `.75` threshold at relative step -19 ms for both reference (`.8613`) and failed candidate (`.8395`), and grows monotonically to `2.0141` and `2.0519` at the current endpoint. The distinction does not first appear at the 20 ms horizon edge, so longer history is not justified. Delayed-Support descriptive evidence uses the explicitly disclosed endpoint-disjoint-only supplement.

## Frozen root cause

| Candidate root cause | Evidence for | Evidence against | Decision |
|---|---|---|---|
| Training objective / sampling tension | adequate input probe; perfect failed-hidden probe; .50 probe-to-head gap; unbalanced positive subdomains | not all seeds collapse on the small TRAIN diagnostic | **PRIMARY** |
| HNM boundary overcorrection | Round 1→3 Support falls 6/6→3/6; repeated selected identities | partial collapse exists at Round 0; Round 1 recovers | rejected as primary |
| Recurrent capacity tension | a nonlinear boundary is plausible | failed GRU hidden probe and 1NN are both 1.00 | rejected |
| Temporal representation tension | recurrent separation increases over time | strong separation exists from the first step | rejected |
| Feature representation tension | raw probe is not perfect | input 1NN=.9583 and current-80D probe=.8889 | rejected |

The exactly frozen primary diagnosis is `TRAINING_OBJECTIVE_SAMPLING_TENSION`.

## Selected intervention

The minimum justified Track-A intervention is one coherent loss change: retain 50/50 class mass, then divide positive mass equally among Slip, ordinary Support, and delayed Support and negative mass equally among fresh Sand, fresh Support pre-I1, historical benign, and historical Hazard precursor. Each endpoint receives its domain mass divided by the number of endpoints in that domain. Uniform deterministic shuffling, all endpoints, the three TRAIN-only HNM rounds, Adam settings, stopping rule, normalizer, seeds, architecture, features, and runtime decision remain unchanged.

This intervention was frozen but **not trained**, because the fresh validation corpus failed its predeclared physical gates. Implementing or executing the unused training variant after that stop would create speculative code and an untestable candidate.

| Item | Reference V2 | Failed candidate | New Candidate A | Candidate B |
|---|---|---|---|---|
| Architecture / parameters | GRU32, 11,010 | GRU32, 11,010 | same, frozen hypothesis only | not selected |
| History / features | 20 ms / causal 80D | same | same | — |
| Normalizer | frozen V2 | frozen V2 | frozen V2 | — |
| Training source | Unified + V2 TRAIN | Unified + V2 + FACTOR_TRAIN | same as failed candidate | — |
| Sampler | deterministic endpoint shuffle | same | same | — |
| Loss | inverse-frequency CE | inverse-frequency CE | 50/50 class mass plus equal within-class domain mass | — |
| HNM | three TRAIN-only rounds | three TRAIN-only rounds | unchanged, if training becomes authorized | — |
| Seeds | 20260828–20260830 | 20260828–20260830 | frozen but not executed | — |

## Architecture comparator

No architecture comparator was used. The GRU remains 11,010 parameters with one hidden-32 layer and `[20,80]` input. LSTM was unnecessary: TRAIN-only hidden separation directly contradicted a recurrent-capacity diagnosis. Avoiding it also preserves the current deployment interface and avoids adding recurrent-operator risk before scientific necessity exists.

## Fresh validation design

The fully independent model-blind dataset ID is `hazard_boundary_resolution_validation_20260904`; its semantic split is `BOUNDARY_RESOLUTION_VALIDATION`.

| Population | Planned |
|---|---:|
| Mild strict Sand | 48 |
| Moderate strict Sand | 12 |
| Ordinary Support | 42 |
| Delayed Support | 18 |
| Total | 120 |

Every Concrete/Marble × `.20/.25/.30` cell contains 8 mild Sand, 2 moderate Sand, 7 ordinary Support, and 3 delayed Support runs. The overall denominator is exactly 60 benign Sand and 60 Hazard controls. Standard mild cells contain both adverse transition-left and comparison transition-right manifolds; Concrete/.25 retains the physically supported left-only exception. Planned Support contains 33 left and 27 right cases including delayed left Support.

Before simulation, all 120 run IDs and scenario signatures were unique. Exact overlap, forbidden-near overlap, run-ID reuse, and cross-role overlap were each zero across historical, prior factor, failed, and pilot manifests. Model-output fields were empty.

## Fresh validation generation and stop

All 120 planned simulations completed once in fixed order in 787.626 seconds. Backfill, replacement, rerun, and model inference were zero. The physical result was:

| Physical population | Actual / planned or gate |
|---|---:|
| Objective eligible | 102 / minimum 115 |
| Strict Sand | 43 / minimum 58 |
| Mild Sand | 36 / minimum 46 |
| Moderate Sand | 7 / minimum 11 |
| Ordinary Support | 42 / minimum 40 |
| Delayed Support | 17 / minimum 17 |
| Invalid | 14 / maximum 5 |
| Slip + Dual contamination | 4 / maximum 2 |
| Unique physical signatures | 102/120 = .85 / minimum .80 |

Actual outcomes were 43 strict benign, 59 Support, 3 Slip, 1 Dual Hazard, and 14 invalid. The invalids consist of six pretarget falls and eight insufficient-post-target observations. All 42 ordinary Support controls were valid with the planned 15 left / 27 right distribution. One delayed Marble/.25 control became a Dual Hazard, leaving 17 qualified delayed controls. Three moderate Marble controls became Slip.

Eight of 19 conjunctive gates failed: objective-valid total, strict Sand, mild Sand, moderate Sand, invalid ceiling, Slip/Dual ceiling, per-source-speed minimums, and factor-manifold minimums. Completion, planned and physical uniqueness, ordinary/delayed Support totals, side coverage, all overlap checks, model blindness, and no-backfill/replacement/rerun passed.

The dataset is immutably frozen as `SEALED_FAILED_PHYSICAL_EVIDENCE`, not opened for model inference, and not eligible for candidate selection.

## Candidate training, freezes, and validation authorization

- New candidate families trained: 0.
- Seeds trained: 0.
- Optimizer steps: 0.
- HNM rounds: 0.
- Candidate freezes: 0.
- Fresh-validation authorization records: 0.
- Fresh-validation open count: 0.
- Reference V2, failed candidate, Candidate A, and Candidate B results on this dataset: not computed.
- Historical V2_VALIDATION regression in this cycle: not run.

No Sand, ordinary-Support, delayed-Support, right-Support, factor-manifold, or paired model metric is reportable for the failed physical corpus. Reporting such metrics would violate the predeclared stop.

| Metric | Reference V2 | Failed candidate | Candidate A | Candidate B |
|---|---|---|---|---|
| Sand specificity / FP / adverse / median / p95 | not evaluated | not evaluated | not trained or evaluated | not selected |
| Ordinary Support recall | not evaluated | not evaluated | not trained or evaluated | not selected |
| Delayed Support recall | not evaluated | not evaluated | not trained or evaluated | not selected |
| Total / right Support recall | not evaluated | not evaluated | not trained or evaluated | not selected |
| Premature behavior | not evaluated | not evaluated | not trained or evaluated | not selected |
| Factor-manifold breakdown | not evaluated | not evaluated | not trained or evaluated | not selected |
| Historical V2_VALIDATION regression | not replayed | not replayed | not trained or evaluated | not selected |

The manifest and physical audit are the complete run-level physical ledger for all 120 runs, including source, speed, severity, topology, phase, physical class, I1, Support, Slip, censor reason, and outcome. Model, probability, streak, and Reflex fields are absent because the validation seal never opened.

## Scientific verdict

The single primary outcome is `BOUNDARY_RESOLUTION_INVALID`. This is a protocol-valid fail-closed result: there was no leakage or adaptive repair, but the independently frozen physical corpus did not meet the eligibility gates needed to test the candidate hypothesis.

There is no development candidate. The frozen Track-A hypothesis remains untested, and final generalization remains `NOT_ESTABLISHED`. Historical V2 and deployment-reference status do not change.

## LSTM, sensors, and deployment implication

LSTM was not necessary, was not trained, and should not replace the GRU. The existing GRU hidden state retained simple Sand/Support separation, so architecture expansion lacks evidence.

Current evidence does not require FSR, Foot IMU, or Terrain gating. Pelvis IMU6 and the causal 80D representation remain the only Hazard runtime input.

The existing E84 Float engineering-reference interface remains usable exactly as before: `[1,20,80]`, GRU hidden 32, binary output, `.99` threshold, and 5 ms persistence. This cycle performed no QAT, PTQ, M3, firmware, HIL, export, or deployment-repository modification.

## Required future evidence and next milestone

The next milestone is exactly `HAZARD_BOUNDARY_VALIDATION_PHYSICAL_REDESIGN`. It should inspect this new physical ledger without model access and freeze a wholly new independent model-blind validation matrix; it must not repair, backfill, or reopen the failed corpus. Candidate training and validation remain outside that milestone until a new corpus passes physical gates.

## Counters

| Counter | Value |
|---|---:|
| Historical HOLDOUT payload reads / inference / feature reconstruction / visualization | 0 / 0 / 0 / 0 |
| Consumed FACTOR_VALIDATION new inference | 0 |
| Generalization VALIDATION inference | 0 |
| New validation simulations | 120 |
| New TRAIN simulations | 0 |
| Candidate families / seeds | 0 / 0 |
| Optimizer steps / HNM rounds | 0 / 0 |
| Threshold / persistence / seed searches | 0 / 0 / 0 |
| Unplanned architecture searches / sensor-fusion experiments / QAT | 0 / 0 / 0 |

## Hashes

| Artifact | SHA-256 |
|---|---|
| Failure-audit config | `9bf0d42abd7da13a30c3c40d8cdebc4c569363ee5180e7f082eb5f21bde460a2` |
| TRAIN diagnostic split | `868718ea51d2094c60dfc69805691522ee9a012dfa228d77389658e9f259bf5c` |
| Optimization pressure | `ec5f286c8a2f19ea950aebfc2a05187a3efce61a4b79be8dfc813ee2f04543db` |
| Checkpoint diagnostics | `3c28137f7a542a902cb4fc1e1bdf693a1295cf3903c30fe8a7fd19c4dececcc7` |
| Representation analysis | `df7d2523f4a485273e6088723420baa032166d179ec93ab0c60cdaa539e9a3bb` |
| Root-cause decision | `f28fc2232d5527fe324b5106c11746de0eab2245a8e774e107ee016212c93730` |
| Intervention/result config | `caa5b475b3042c39bc0ff2bd81f17e2b6f344a95aab04e150ba7391d300d1703` |
| Fresh validation design config | `e56bbfaea3541fbc36ad1d4656f9cd10996012a6b85d4daa6a36dfb3f721ea39` |
| Fresh validation matrix | `78d0ad6d3bc5c2232da0b499db25f3153758f367c2c0208aaedfef4ddb746f74` |
| Fresh validation signature matrix | `e9da061898ec99c76d98e4d13e81b836862db6e8628e3a0b4ce2e1774896d9e1` |
| Pre-simulation freeze | `73853f819fa3d6311381cd5a8bfc2e419aeedd28101d071bf90b88e5f35fb2a2` |
| Manifest | `3b1a35049e79ecd90399d4bac8092a24d5053b1c26bd6144854c77a492914934` |
| Physical audit | `e52f7c827597d4b42547a90e1689d2e350a78c13c3832d793027a2645adc320c` |
| Validation seal | `290cfac37ea283881137175240b2a6acf04ce04f15f1c01c8b8446c1f4d8e507` |
| Dataset freeze file / semantic | `2092a4a8ce3a39ce954d573a4da25efa11ddf348e59e4b827287de02c274ba5d` / `ecddb3b01728eaf97a4b5a937df426adbcfa631ec1ff3c4139bb9ef0c09dfc09` |
| NPZ aggregate | `c89accf0fd86a3a700d79ef56c0825caeb5f2947b6141fea70d7fb34ae2a7898` |
| Milestone semantic result / file | `c16e428cb4d009ec010ae2a6812e003f4408ba3b0f8f8dce683586da6b1160d5` / `8089023ecbb404f8be610fa26b350d5157e161aac8e28534546718d23d4d1378` |
| Candidate training configs / checkpoints / freezes | not produced; physical stop preceded training |
| Validation authorization / model results / paired comparisons | not produced; open count remained 0 |
| Final candidate decision | no development candidate; `BOUNDARY_RESOLUTION_INVALID` |

## Tests

The targeted suite proves consumed-validation refusal before NPZ loading, historical HOLDOUT guard preservation, run-disjoint audit identity, deterministic design and hashes, model-blind failed-corpus sealing, zero validation opens, no post-validation training, the zero-candidate budget, and the unchanged sensor/model/runtime contract. Full repository results are recorded in the completion response.
