# Model V2 Candidate Readiness Review

## 1. Purpose

This review determines whether the frozen anchor-refined Model V2 candidate is ready for an external development evaluation despite failing the original internal Slip-recall gate. It is a read-only scientific review: no training, data generation, label or extraction change, model selection, or runtime-parameter search was performed.

The narrow question is whether the five remaining primary Slip failures are genuine absent responses or physically supported early responses that conflict with the historical timing window. The answer does not rewrite the primary score.

## 2. Starting state

The review started from clean `main` at `8a4970cad8778e100751d4f6a8ae4f15b5eb4c03` (`Train anchor-refined Model V2`). `HEAD` equaled `origin/main`. The pre-analysis review contract is [`configs/experiment/20260902_model_v2_candidate_readiness_review.yaml`](../configs/experiment/20260902_model_v2_candidate_readiness_review.yaml), SHA-256 `c8a7f79a916b0d660bd319b0a1f66f9f01c9c76fdf41272b3504521337122b65`.

## 3. Evidence boundary

Authorized evidence was limited to V2_VALIDATION, already-frozen effective-TRAIN context, prior Ice-semantics and failure-audit evidence, and frozen model artifacts. The candidate was replayed only on V2_VALIDATION.

```text
Generalization VALIDATION new-candidate inference: NO
Current Unified HOLDOUT waveform reopened: NO
Current Unified HOLDOUT new inference: NO
Generalization HOLDOUT waveform opened: NO
Generalization HOLDOUT inference: NO
Generalization HOLDOUT guard count: 0
```

## 4. Candidate preservation

All protected candidates and datasets verified before review. No protected artifact was written.

| Protected item | Frozen identity | Verification |
|---|---|---|
| Model V1 | candidate freeze `91834c88...a08c2` | exact/restorable |
| Baseline data-only V2 | candidate freeze `edb16b7d...ed8725` | exact/restorable |
| Extraction-rebalanced V2 | candidate freeze `ca1ba7ab...c4533` | exact/restorable |
| Anchor-refined V2 | candidate freeze `95dab532...bd85f`; evaluation freeze `cad39021...95c07` | exact/restorable |
| Terrain V1 | frozen FSR4/MLP normalizer and all three checkpoints | exact/restorable |
| Unified dataset | 256/256 files; manifest `d023384a...e9d6` | exact |
| V2 dataset | 412/412 files; manifest `7a036d34...8b25` | exact |
| Generalization dataset | 72/72 files; manifest `72f5dd30...6f53` | exact; HOLDOUT sealed |
| Ice-semantics dataset | 48/48 files; manifest `6a472d4b...5759` | exact |

## 5. Historical V2 progression

The anchor refinement is the strongest internal candidate in the frozen four-model comparison.

| Metric | V1 | Baseline V2 | Rebalanced V2 | Anchor-refined V2 |
|---|---:|---:|---:|---:|
| Hazard | 34/64 (53.13%) | 55/64 (85.94%) | 58/64 (90.63%) | **59/64 (92.19%)** |
| Slip | 22/35 (62.86%) | 29/35 (82.86%) | 29/35 (82.86%) | **30/35 (85.71%)** |
| Support | 12/30 (40.00%) | 27/30 (90.00%) | 30/30 (100%) | **30/30 (100%)** |
| Confirmed specificity | 23/26 (88.46%) | 26/26 (100%) | 23/26 (88.46%) | **26/26 (100%)** |
| Premature | 13/64 (20.31%) | 6/64 (9.38%) | 6/64 (9.38%) | **5/64 (7.81%)** |
| Right Support | 0/12 | 12/12 | 12/12 | **12/12** |
| Delayed Support | 0/6 | 3/6 | 6/6 | **6/6** |
| Marble delayed Support | 0/3 | 0/3 | 3/3 | **3/3** |
| Staged Sand benign | 8/8 | 8/8 | 8/8 | **8/8** |
| Speed Sand benign | 9/12 | 12/12 | 9/12 | **12/12** |

This progression shows genuine gains in Support, side coverage, and hard-negative specificity. It does not make the failed Slip gate pass.

## 6. Frozen primary contract

Established Slip remains touchdown-anchor tangential drift `>=0.050 m` for 3 ms under ANY-foot logic. The primary valid alert region remains `Slip-30 ms` through `Slip+40 ms`. Slip recall must remain at least 95% to pass. A response before the window remains `PREMATURE` and a primary failure even when secondary physical evidence supports it.

No definition, denominator, gate, label, or score changed in this review.

## 7. Frozen Ice precursor semantics

The separate `ICE_PRECURSOR_CANDIDATE` is loaded exact-Ice drift in the half-open interval `[0.030, 0.050) m`. Its existing physical verdict is `ICE_PHYSICAL_PRECURSOR_SUPPORTED`. Frozen outcome annotations distinguish same-episode, next-episode, later Slip, benign release, and censoring. The precursor remains secondary evidence and is not established Slip.

The applicable semantics configuration and report hashes are `29ce97d2...46e0` and `535c9e50...0d5d`, respectively. No candidate-specific secondary metric was invented.

## 8. Anchor-refined V2 result

Read-only replay reproduced the frozen V2_VALIDATION result exactly, including the complete result object.

| Metric | Result | Frozen gate | Status |
|---|---:|---:|---|
| Overall Hazard recall | 59/64 (92.19%) | >=90% | PASS |
| Slip recall | 30/35 (85.71%) | >=95% | **FAIL** |
| Support recall | 30/30 (100%) | >=85% | PASS |
| Confirmed no-hazard specificity | 26/26 (100%) | >=95% | PASS |
| Premature rate | 5/64 (7.81%) | <=10% | PASS |
| Slip p95 latency | 27.2 ms | <=40 ms | PASS |
| Support p95 established latency | -17 ms | <=50 ms | PASS |

Delayed Support is 6/6, Marble delayed Support 3/3, right Support 12/12, staged-Sand benign 8/8, and speed-Sand benign 12/12. Slip recall is the only failed primary gate.

## 9. Five primary Slip failures

The table below is the primary review result. Sample indices are milliseconds at the frozen 1 kHz replay rate. `C/M` means Concrete/Marble source.

| Run | Speed | Side | Family/source | Primary result | Precursor onset | Reflex | Slip | Reflex relation | Physical interpretation |
|---|---:|---|---|---|---:|---:|---:|---|---|
| `m2v2_bis_v_m_0200_i08` | .20 | bilateral | baseline immediate-Ice/M | PREMATURE | 2170 | 2170 | 2323 | `INSIDE_FUTURE_SLIP_PRECURSOR` | supported early Hazard response |
| `m2v2_ibc_v_c_0200_b07` | .20 | right-only | Ice benign/C | PREMATURE | 2466 | 2478 | 2627 | `INSIDE_FUTURE_SLIP_PRECURSOR` | supported early Hazard response |
| `m2v2_ibc_v_c_0200_b08` | .20 | right-only | Ice benign/C | PREMATURE | 2466 | 2478 | 2707 | `INSIDE_FUTURE_SLIP_PRECURSOR` | supported early Hazard response |
| `m2v2_inp_v_c_0200_p08` | .20 | right-only | Ice near-hazard/C | PREMATURE | 2466 | 2478 | 2627 | `INSIDE_FUTURE_SLIP_PRECURSOR` | supported early Hazard response |
| `m2v2_odi_v_c_0200_d08` | .20 | bilateral | one-contact delayed-Ice/C | PREMATURE | 2466 | 2478 | 2632 | `INSIDE_FUTURE_SLIP_PRECURSOR` | supported early Hazard response |

All five are delayed physical progressions and carry the frozen `LATER_SLIP` outcome at first Reflex. The detailed physical and runtime values are:

| Run | Target contact | First >=.99 | Reflex | Max p | >=.99 streak | Drift / velocity at Reflex | Loaded phase | Same / next / later |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `m2v2_bis_v_m_0200_i08` | 1491 | 2166 | 2170 | .998968 | 61 ms | .030604 m / .942928 m/s | right support | no / no / **yes** |
| `m2v2_ibc_v_c_0200_b07` | 1504 | 2471 | 2478 | .997492 | 16 ms | .042741 m / 1.018981 m/s | left support | no / no / **yes** |
| `m2v2_ibc_v_c_0200_b08` | 1504 | 2471 | 2478 | .997492 | 18 ms | .042741 m / 1.018981 m/s | left support | no / no / **yes** |
| `m2v2_inp_v_c_0200_p08` | 1504 | 2471 | 2478 | .997492 | 7 ms | .042741 m / 1.018981 m/s | left support | no / no / **yes** |
| `m2v2_odi_v_c_0200_d08` | 1504 | 2471 | 2478 | .997492 | 19 ms | .042741 m / 1.018981 m/s | left support | no / no / **yes** |

For the first run, the first raw threshold crossing is four milliseconds before the precursor boundary, but the frozen 5 ms persistence places the actual `REFLEX_REQUIRED` onset exactly at precursor entry. Classification is based on Reflex, not an unsustained raw crossing.

## 10. Precursor relationship

| Category | Count |
|---|---:|
| Primary Slip successes | 30 |
| Primary Slip failures | 5 |
| Supported future-Slip precursor early response among failures | **5** |
| Before-precursor early response among failures | 0 |
| Benign-release false response among failures | 0 |
| Censored precursor response among failures | 0 |
| True miss among failures | **0** |
| Other primary failure | 0 |

Thus `PRIMARY_METRIC_FAILURE` and `PHYSICAL_HAZARD_RESPONSE_FAILURE` differ for every reviewed run: all five remain official primary failures, while none is a genuine missing physical response under the pre-existing precursor semantics.

The descriptive development physical-response view is therefore 35/35: 30 primary successes plus five supported early future-Slip responses. This observation must not be renamed a new passing metric or used to replace official 30/35 Slip recall.

## 11. Genuine detection failures

`GENUINE_DETECTION_FAILURE = 0/5` among the primary-failed runs. There is no absent Reflex, no low-confidence miss, no unrelated early alert, and no alert tied to a benign or unresolved outcome. Each response is sustained and tied to a later established Slip by the frozen semantics annotation.

The 30 primary successes were also reviewed to avoid failure-only interpretation:

| Successful-Slip view | Count |
|---|---:|
| Alert within the primary region before established Slip | 22 |
| Alert after established Slip but still valid | 7 |
| Alert inside precursor before the primary window | 0 |
| Alert before the Slip window outside precursor | 1 |
| No precursor before Slip (cross-cutting) | 1 |

The one outside-precursor early alert is a dual Slip-and-Support run whose response is valid for the much earlier Support event; it is not a hidden premature Ice alert. Successes by speed are `.20: 9`, `.25: 11`, `.30: 10`; by side, bilateral 27 and left-only 3; by physical timing, immediate 4 and delayed 26. No right-only Slip run passes the primary timing window. Family counts are baseline immediate-Ice 11, delayed-Sand Support 1, Ice benign 2, Ice near-hazard 7, and one-contact delayed-Ice 9.

The reviewed behavior is therefore narrow: five specific long-lead future-Slip responses. The candidate is not claimed to be a general precursor detector.

## 12. Speed breakdown

| Breakdown | N | Primary correct | Primary fail | Supported precursor among fails | True miss |
|---|---:|---:|---:|---:|---:|
| 0.20 m/s | 14 | 9 | 5 | 5 | 0 |
| 0.25 m/s | 11 | 11 | 0 | 0 | 0 |
| 0.30 m/s | 10 | 10 | 0 | 0 | 0 |
| left-only | 3 | 3 | 0 | 0 | 0 |
| right-only | 3 | 0 | 3 | 3 | 0 |
| bilateral | 29 | 27 | 2 | 2 | 0 |
| immediate | 4 | 4 | 0 | 0 | 0 |
| delayed | 31 | 26 | 5 | 5 | 0 |

The residual cluster is entirely 0.20 m/s delayed progression. Anchor refinement removed the former `.30 m/s` failure; it did not change the scientific interpretation of the remaining low-speed cases.

## 13. Side breakdown

All three right-only failures produce strong early, precursor-supported responses, as do both bilateral failures. Right-only Slip remains 0/3 under the original timing contract and must be reported as such, but these cases do not demonstrate loss of Pelvis-IMU observability or absence of Hazard sensitivity. Left-only Slip is 3/3.

The right-only sample is small and no claim of universal side invariance is warranted. External evaluation is more informative than internally fitting these three timing-conflict runs.

## 14. Seed/persistence behavior

| Run | Max ensemble p | Max streak >=.99 | Longest >=.95 / >=.90 | Seed maxima (28 / 29 / 30) | Seed pattern | Response type |
|---|---:|---:|---:|---|---|---|
| `m2v2_bis_v_m_0200_i08` | .998968 | 61 ms | 91 / 97 ms | .998698 / .999817 / .999634 | all 3 high | `SUSTAINED_EARLY` |
| `m2v2_ibc_v_c_0200_b07` | .997492 | 16 ms | 49 / 62 ms | .997748 / .997948 / .998622 | all 3 high | `SUSTAINED_EARLY` |
| `m2v2_ibc_v_c_0200_b08` | .997492 | 18 ms | 41 / 48 ms | .997748 / .997948 / .998107 | all 3 high | `SUSTAINED_EARLY` |
| `m2v2_inp_v_c_0200_p08` | .997492 | 7 ms | 43 / 81 ms | .997748 / .997948 / .998049 | all 3 high | `SUSTAINED_EARLY` |
| `m2v2_odi_v_c_0200_d08` | .997492 | 19 ms | 75 / 78 ms | .998944 / .998478 / .998986 | all 3 high | `SUSTAINED_EARLY` |

Here “high” means a seed maximum at or above 0.99. Every ensemble streak already exceeds the frozen 5 ms persistence. Threshold and persistence are not root causes; changing either would be post-result tuning and would not resolve the timing-contract tension.

## 15. Primary vs secondary interpretation

The previously frozen precursor-aware evaluation is unchanged:

| Outcome | Episodes | Alert inside candidate | Alert before established Slip |
|---|---:|---:|---:|
| Same-episode Slip | 257 | 33 | 30 |
| Next-episode Slip | 25 | 0 | 0 |
| Later Slip | 142 | 11 | 28 |
| **All future Slip** | **424** | **44** | **58** |
| Benign release | 7 | 0 | 0 |
| Censored | 30 | 0 | 0 |

There are 461 total precursor episodes and 380 future-Slip precursor states without an in-candidate alert. This sparse coverage is why the candidate must not be described as a dedicated precursor detector. The secondary evidence answers only whether the five observed early alerts are physically defensible.

Read-only comparison on the same five runs shows that the timing behavior predates the final anchor intervention. Each cell is `first Reflex / max p / primary / precursor relation`.

| Run | V1 | Baseline V2 | Rebalanced V2 | Anchor-refined V2 |
|---|---|---|---|---|
| `m2v2_bis_v_m_0200_i08` | 2164/.999976/PREMATURE/before | 2165/.999936/PREMATURE/before | 2168/.999831/PREMATURE/before | 2170/.998968/PREMATURE/**inside future** |
| `m2v2_ibc_v_c_0200_b07` | 2470/.999980/PREMATURE/inside future | 2478/.998693/PREMATURE/inside future | 2478/.999219/PREMATURE/inside future | 2478/.997492/PREMATURE/inside future |
| `m2v2_ibc_v_c_0200_b08` | 2470/.999984/PREMATURE/inside future | 2478/.998693/PREMATURE/inside future | 2478/.999205/PREMATURE/inside future | 2478/.997492/PREMATURE/inside future |
| `m2v2_inp_v_c_0200_p08` | 2470/.999986/PREMATURE/inside future | 2478/.998693/PREMATURE/inside future | 2478/.999148/PREMATURE/inside future | 2478/.997492/PREMATURE/inside future |
| `m2v2_odi_v_c_0200_d08` | 2470/.999988/PREMATURE/inside future | 2478/.999341/PREMATURE/inside future | 2478/.999682/PREMATURE/inside future | 2478/.997492/PREMATURE/inside future |

Anchor refinement moves the first Marble case's persisted Reflex to the exact precursor boundary and retains the other supported responses. It does not manufacture a new timing exception to rescue its score.

## 16. Specificity/Support readiness

The remaining primary failure does not coexist with an unresolved specificity or Support regression. Confirmed specificity is 26/26, hard normal is 6/6, staged-Sand benign is 8/8, speed-Sand benign is 12/12, Support is 30/30, delayed Support is 6/6, Marble delayed Support is 3/3, and right-only Support is 12/12. Overall Hazard and premature-rate gates also pass.

These solved controls make a broad new internal intervention disproportionate to the evidence.

## 17. Architecture implications

No capacity or memory failure is demonstrated. The 20 ms GRU produces high, persistent responses long before Slip and now solves all Support cases and hard-negative controls. Longer history, LSTM, a larger GRU, and feature redesign have no specific unresolved behavior to target.

```text
ARCHITECTURE_CHANGE_NOT_YET_JUSTIFIED
```

## 18. Sensor implications

The provisional architecture remains Pelvis IMU6 for Hazard plus left FSR4 for Terrain, ten physical channels total. Strong Pelvis-only responses in all five cases argue against immediate sensor expansion. This readiness result is not a final sensor freeze and does not establish hardware realism.

```text
10_CHANNEL_ARCHITECTURE_STILL_PLAUSIBLE
FINAL_SENSOR_ARCHITECTURE_NOT_READY
```

## 19. Risk of further internal optimization

Another internal retraining experiment is not scientifically justified now because no genuine response failure remains in the target set. Training specifically to delay or suppress these five alerts would conflict with the completed Ice physical-semantics study. Repeated intervention after several V2_VALIDATION-informed audits also increases internal-set overfitting risk while yielding little new information.

The higher-value experiment is the untouched external development split. A future internal intervention would become justified only if that evaluation reveals genuine, repeatable recall or specificity failures under the frozen contract.

## 20. Candidate readiness decision

```text
READY_FOR_EXTERNAL_DEVELOPMENT_EVALUATION
```

Rationale: every primary gate except Slip recall passes; specificity and Support regressions are resolved; all five remaining Slip failures are unanimous, sustained responses inside frozen future-Slip precursor states; and genuine missed responses are absent. This is eligibility to test generalization, not evidence that external generalization is supported.

The exact existing candidate is promoted without retraining or replacement in [`configs/model/development_candidate.yaml`](../configs/model/development_candidate.yaml). `MODEL_V2_DEVELOPMENT_PROMOTION_SHA` is `1e4931e35e873cd721b412c6a45f66340f7ee9eebc1900d9c4aa3dc9ab3d092f`.

Promotion preserves candidate freeze `95dab532...bd85f`, evaluation freeze `cad39021...95c07`, normalizer `e0d796e8...e92a`, the three checkpoint hashes, architecture `ae475369...a897`, feature schema `fe5b6c1c...f8adb`, threshold 0.99, and persistence 5 ms. No new model artifact exists.

## 21. External development evaluation contract

The next milestone is predeclared but not executed:

```text
MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION
```

It must compare frozen V1 and the exact promoted V2 on the identical 36 Generalization VALIDATION runs. It must report original Hazard, Slip, Support, confirmed no-hazard and applicable Ice-benign specificity, premature rate, Slip latency, and Support timing, with breakdowns for one-contact delayed Ice, Ice benign, delayed Sand Support, right Sand Support, and speed-stratified scenarios. The existing frozen precursor-aware secondary contract must be reported beside, never instead of, the primary metrics.

Existing gates remain authoritative. No retraining, threshold or persistence search, post-result model selection, or HOLDOUT access is allowed. Predeclared result categories are:

- `GENERALIZATION_DEVELOPMENT_SUPPORTED`
- `GENERALIZATION_DEVELOPMENT_SUPPORTED_WITH_ICE_TIMING_TENSION`
- `GENERALIZATION_DEVELOPMENT_NOT_SUPPORTED`

The exact decision must follow the stricter repository protocol if it applies.

## 22. Generalization HOLDOUT preservation

Generalization HOLDOUT remains 36 sealed runs with guard count 0. Even future Generalization VALIDATION success does not authorize opening it. The required sequence remains Generalization VALIDATION, result audit, final-candidate freeze, explicit HOLDOUT-readiness decision, then one-shot Generalization HOLDOUT.

The current Unified HOLDOUT was not reopened and received no new inference.

## 23. Limitations

- Official Slip recall is still only 30/35 and fails the 95% gate.
- All five reviewed failures are from 0.20 m/s delayed progressions; three are right-only, and several traces share highly similar simulated dynamics.
- V2_VALIDATION contains only three right-only Slip cases, so external side evidence is needed.
- The candidate covers only 44/424 future-Slip precursor episodes inside the candidate region; it is not a general precursor detector.
- Evidence is simulator- and policy-specific; no hardware or deployment claim follows.
- Generalization VALIDATION has not received V2 inference, and no external V2 performance claim is made.

## 24. Verdict

```text
MODEL_V2_CANDIDATE_READINESS_REVIEW_COMPLETE
READY_FOR_EXTERNAL_DEVELOPMENT_EVALUATION

Historical verdict retained separately:
MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED
```

The two verdicts are compatible. The historical internal verdict remains correct under the unchanged primary gate; readiness means only that the physically explained residual is better tested on untouched external development evidence than optimized further on V2_VALIDATION.

All review counters are zero:

```text
optimizer steps = 0
checkpoint writes = 0
normalizer fits = 0
HNM rounds = 0
threshold searches = 0
persistence searches = 0
architecture searches = 0
seed searches = 0
new simulation runs = 0
```

Final verification passed:

| Check | Result |
|---|---|
| Full pytest | 82 passed, 1 skipped |
| `compileall src scripts tests` | PASS |
| Critical Ruff `E9/F63/F7/F82` | PASS |
| `git diff --check` | PASS |
| V1 / Terrain V1 / baseline V2 / rebalanced V2 / anchor-refined V2 verifiers | all PASS; exact hashes |
| Unified / V2 / Generalization / Ice-semantics file rehash | 256/256, 412/412, 72/72, 48/48; zero mismatch |
| Generalization HOLDOUT guard | 0 |

## 25. Recommended next milestone

```text
MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION
```

It was not started in this milestone. Retraining, extraction or dataset changes, Generalization VALIDATION inference, Generalization HOLDOUT access, model/runtime tuning, architecture expansion, Terrain retraining, E84 work, quantization, HIL, and Recovery remain out of scope.
