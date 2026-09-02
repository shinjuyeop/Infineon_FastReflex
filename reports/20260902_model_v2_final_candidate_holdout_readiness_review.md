# Model V2 Final Candidate Freeze and HOLDOUT Readiness Review

## 1. Purpose

This milestone performs the final scientific review before the first and only Model V2 `GENERALIZATION_HOLDOUT` evaluation. It decides whether the exact promoted candidate can be frozen as `FINAL_GENERALIZATION_CANDIDATE`, and freezes the primary, secondary, verdict, comparison, and post-open contracts for that future operation.

The review concludes `HOLDOUT_READY`. This is readiness to run one fresh simulation HOLDOUT evaluation, not evidence that the candidate passes HOLDOUT, transfers to a real robot, is production-ready, or fixes the final sensor architecture.

## 2. Starting state

The review began from clean `main` at `a1c1950b9a55d647d72b509de8e72bd3ee858079` (`Evaluate Model V2 generalization development`). `HEAD` equaled `origin/main`, and the tracked worktree was clean.

Before the review conclusion was finalized, the contract was recorded in [`configs/experiment/20260902_model_v2_final_candidate_holdout_readiness_review.yaml`](../configs/experiment/20260902_model_v2_final_candidate_holdout_readiness_review.yaml), SHA-256 `0206dd12078ffa191cc6424a35cdead889c9614a1ecb38a9a830a1d464368ec8`.

## 3. Evidence boundary

The review reused the committed Generalization VALIDATION evidence frozen by the previous milestone. It did not rerun V1, V2, or Terrain inference. It did not train, tune, generate data, or inspect a model variant.

Generalization HOLDOUT checks were restricted to safe provenance: IDs, count, split membership, file existence, byte hashes, file sizes, and guard metadata. No HOLDOUT NPZ was deserialized; no waveform, IMU, FSR, feature, event, per-run signal, metric, inference, or visualization operation occurred. Unified HOLDOUT also remained sealed.

```text
Generalization VALIDATION reused from frozen evidence: YES
Generalization VALIDATION new candidate inference: NO
Generalization HOLDOUT waveform opened: NO
Generalization HOLDOUT Hazard inference: NO
Generalization HOLDOUT Terrain inference: NO
Generalization HOLDOUT visualization: NO
Generalization HOLDOUT guard count: 0
Current Unified HOLDOUT waveform reopened: NO
Current Unified HOLDOUT new inference: NO
```

Protected data integrity was re-established by byte hashing without payload loading.

| Dataset | Manifest SHA-256 | Declared | Exact files |
|---|---|---:|---:|
| Unified `unified_hazard_reflex_20260829` | `d023384aaaac22076a2be7c2fa242acac91a548d82102640c39868990cb0e9d6` | 256 | 256 |
| Model V2 `model_v2_hazard_reflex_20260901` | `7a036d3485bb19a3570a3dd4a41cc990375028f7e153e85525bd4bf19cff8b25` | 412 | 412 |
| Generalization `generalization_hazard_reflex_20260831` | `72f5dd300bb5be78b22b9a8c1ad9788cdf45e7cda9b4f46d79eb290140146f53` | 72 | 72 |
| Ice semantics `ice_near_hazard_semantics_20260901` | `6a472d4b26e724355c6f2d88d0668c4f91c925037c63ad8e1838f83a53e15759` | 48 | 48 |

Generalization remains 36 VALIDATION plus 36 HOLDOUT. Their canonical run-ID hashes are respectively `2b568dc4ef452307cbb99b027162bf9da5a3d2977d70a36fc32b2de1b901e1a1` and `6c911c33bc7ea1eb89a58f129d44848989ba4f6aea070f9c15084bdcc2b00c1f`. The canonical safe HOLDOUT file/hash/size metadata SHA-256 is `dfbadbb6299a0f946c29f6332fe6095daa80931dcd30389de66e91b71ff07e0f`.

## 4. Candidate integrity

The promoted identity and every referenced file hash resolve exactly. No weights, normalizer, architecture, feature schema, ensemble membership, threshold, or persistence were changed.

| Item | Frozen value | Status |
|---|---|---|
| Candidate ID | `model_v2_anchor_refined_gru20_20260902` | exact |
| Candidate freeze | `95dab53275aa77ded36b479c3633ab879a1331cb2c96bd0eb014de2be99bd85f` | exact |
| Development promotion | `1e4931e35e873cd721b412c6a45f66340f7ee9eebc1900d9c4aa3dc9ab3d092f` | exact |
| Normalizer | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` | exact |
| Checkpoint seed 20260828 | `7094a2dca40e8d3c84554619652d69c17920c8e82765460ff8621c13ef494cb9` | exact |
| Checkpoint seed 20260829 | `3ad298eea4c35eca896afd31f860fd6b44ce35d7d9978e2546bb40b693e62c39` | exact |
| Checkpoint seed 20260830 | `fe96dfeb8461871044de0f8672190680ce46164a1c28efbe6738d22f9d439bbd` | exact |
| Architecture | `ae4753699399aa7e4a93639935e085156ef28ed100a77058d7308eb1549aa897` | exact |
| Feature schema | `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb` | exact |
| Runtime | threshold 0.99; persistence 5 ms; replay stride 1 ms | exact |
| Ensemble | seeds 20260828, 20260829, 20260830 | unchanged |

The architecture is Pelvis IMU6 to causal 80D, `[20,80]` input, GRU hidden 32, one unidirectional layer, linear `32->2`, 11,010 parameters, and a fixed three-seed mean. Artifact copy or mutation is `NO`.

## 5. Historical candidate progression

All historical candidates remain independently exact and restorable. No model was overwritten, combined, or averaged beyond its already frozen ensemble.

| Candidate | Frozen identity | Review result |
|---|---|---|
| Model V1 | supported freeze `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2` | exact/restorable |
| Baseline data-only V2 | candidate freeze `edb16b7d96fb38e680e36dcca5ecd7b1c1682ba410b94d21f40c2a38d4ed8725` | exact/restorable |
| Extraction-rebalanced V2 | candidate freeze `ca1ba7abae1746528cdd098b903ce8f967937a625773bb8029fc486645fc4533`; evaluation freeze `3d415c0d4c635497f6d882218a059c9228f02fdda9936124439d709480d94dbc` | exact/restorable |
| Anchor-refined V2 | candidate freeze `95dab53275aa77ded36b479c3633ab879a1331cb2c96bd0eb014de2be99bd85f` | exact/restorable; selected |
| Terrain V1 | FSR4, 50 ms, seeds 17/29/43, held state | exact/restorable; advisory |

## 6. Internal V2 evidence

The frozen `V2_VALIDATION` result remains Hazard 59/64 (92.19%), Slip 30/35 (85.71%), Support 30/30 (100%), confirmed no-hazard specificity 26/26 (100%), and premature 5/64 (7.81%). The original internal verdict remains `MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED`, principally because strict Slip recall did not meet its gate.

This review does not reinterpret internal failure scores. It uses them together with the external development result to decide whether another intervention is scientifically justified.

## 7. External Generalization VALIDATION evidence

The table below is a direct transcription of frozen evidence, not a recomputation.

| Metric | V1 Generalization VALIDATION | Final V2 Generalization VALIDATION | Delta | Primary gate |
|---|---:|---:|---:|---|
| Hazard recall | 13/26 (50.00%) | 25/26 (96.15%) | +46.15 pp | >=90%: PASS |
| Slip recall | 7/12 (58.33%) | 11/12 (91.67%) | +33.33 pp | >=95%: **FAIL** |
| Support recall | 6/14 (42.86%) | 14/14 (100%) | +57.14 pp | >=85%: PASS |
| Primary specificity | 5/10 (50.00%) | 10/10 (100%) | +50.00 pp | >=95%: PASS |
| Ice-benign specificity | 3/4 (75.00%) | 4/4 (100%) | +25.00 pp | >=95%: PASS |
| Premature rate | 7/26 (26.92%) | 1/26 (3.85%) | -23.08 pp | <=10%: PASS |
| Slip p95 latency | +5.3 ms | +11 ms | +5.7 ms | <=+40 ms: PASS |
| Support p95 established latency | -17.25 ms | -17 ms | +0.25 ms | <=+50 ms: PASS |

The frozen external evidence has SHA-256 `cc559c592f64ed8afd27b039d57ace35d8bbb3efad1cac64e3d10788e0ffa556`; V2 metrics have `291e40aaa96daddd1267f65ed188f205cde5d79d8782379f9bdcb9a95bc5260f`, and run-level evidence has `561d7394dcd3645cbaf64a6ec9089d98106738a37145653fd259c48e33f983a4`.

## 8. Primary gate status

The historical primary verdict remains exactly `GENERALIZATION_PRIMARY_GATES_FAIL`. Slip recall is 11/12 (91.67%), below the frozen 95% minimum. This review does not claim that all development gates passed and does not lower the Slip gate to fit the observed result.

The separate development interpretation remains exactly `GENERALIZATION_DEVELOPMENT_SUPPORTED_WITH_ICE_TIMING_TENSION`; it is not upgraded to an unqualified supported verdict.

## 9. Ice timing limitation

The sole V2 primary failure is `ghr_ocd_v_c020`, family `ONE_CONTACT_DELAYED_ICE_SLIP`, source Concrete, speed 0.20 m/s, actual bilateral Slip. Frozen event times are:

| Event | Time |
|---|---:|
| Loaded exact-Ice `[0.030,0.050) m` precursor onset | 2466 ms |
| Persisted Reflex onset | 2478 ms |
| Established Slip | 2632 ms |

The Reflex is sustained, begins 12 ms after the frozen physical precursor, and is 154 ms before established Slip. All three seeds respond strongly. Under the strict primary event window it is `PREMATURE`; under the independent frozen physical diagnostic it is `SUPPORTED_ICE_PRECURSOR_TIMING_CONFLICT`.

There is no benign-release false alert, censor ambiguity, absent response, late miss, or threshold/persistence instability. `GENUINE_DETECTION_FAILURE = NO`. The official primary scores remain 25/26 Hazard and 11/12 Slip.

## 10. Genuine failure review

Frozen run-level evidence contains 25 primary Hazard successes, one primary failure with a physically supported early response, and zero genuine detection misses. The physical-response diagnostic is 26/26, but it is descriptive and cannot replace the strict primary denominator.

The one primary failure remains visible because readiness is justified despite the known contract tension, not by rewriting it.

## 11. Specificity review

V2 achieves primary no-hazard specificity 10/10 and Ice-benign specificity 4/4. It has zero pre-I1 delayed-Sand alerts, speed-Sand benign specificity 6/6, and zero invalid benign alerts. Sensitivity did not produce a broad external false-positive failure.

## 12. Support review

V2 achieves Support 14/14: delayed Sand Support 4/4, right-only Sand Support 4/4, Concrete Support 7/7, and Marble Support 7/7. The delayed-Sand cases have no pre-I1 response. This is external evidence that corrected extraction and anchor refinement transfer beyond internal data.

Terrain remains advisory and is not used to gate, rescue, or delay Hazard.

## 13. Speed review

Every frozen V2 speed cell is 2/2. These are small descriptive cells and do not predict HOLDOUT performance.

| Speed | Slip | Support | Sand-benign specificity |
|---:|---:|---:|---:|
| 0.20 m/s | 2/2 | 2/2 | 2/2 |
| 0.25 m/s | 2/2 | 2/2 | 2/2 |
| 0.30 m/s | 2/2 | 2/2 | 2/2 |

The prior 0.20 and 0.30 m/s V1 Slip gaps and the speed-Sand benign gap are externally resolved on VALIDATION.

## 14. Side/source review

V2 Support is left-only 10/10 and right-only 4/4. Slip is bilateral 11/12. The external split contains no right-only or left-only Slip denominator, so this review makes no claim of final unilateral Slip generalization.

By source, V2 Support is 7/7 for both Concrete and Marble and specificity is 5/5 for both. Hazard is 12/13 Concrete and 13/13 Marble; the sole Concrete failure is the frozen Ice timing conflict. These are descriptive external-development results, not post-hoc subgroup gates.

## 15. Data-coverage hypothesis

`DATA_COVERAGE_HYPOTHESIS_SUPPORTED` remains the major research conclusion. With the same Pelvis IMU6, causal 80D features, GRU20 hidden 32 architecture, 11,010 parameters, threshold 0.99, and persistence 5 ms, expanded/balanced data plus corrected extraction improved Hazard from 50.00% to 96.15%, Slip from 58.33% to 91.67%, Support from 42.86% to 100%, specificity from 50% to 100%, and premature rate from 26.92% to 3.85%.

| Original V1 gap | V1 external | V2 external | Status |
|---|---:|---:|---|
| Delayed Ice | 3/6 | 5/6 | **PARTIAL PRIMARY / PHYSICALLY EXPLAINED** |
| Ice benign | 3/4 | 4/4 | RESOLVED |
| Delayed Sand pre-I1 | 0/4 valid; 4 premature | 4/4 valid; 0 premature | RESOLVED |
| Right Sand Support | 0/4 | 4/4 | RESOLVED |
| 0.20 Slip | 1/2 | 2/2 | RESOLVED |
| 0.30 Slip | 1/2 | 2/2 | RESOLVED |
| Speed Sand benign | 2/6 | 6/6 | RESOLVED |

Six of seven audited mechanisms are fully resolved. The seventh is improved and its only remaining primary failure is a frozen, physically supported timing conflict rather than a detector miss.

## 16. Need for further intervention

The decision is `NO_FURTHER_INTERNAL_OPTIMIZATION_JUSTIFIED`.

Additional retraining, data changes, extraction changes, threshold/persistence changes, checkpoint/seed selection, or model changes before HOLDOUT would mostly optimize against one already understood Generalization VALIDATION run. There is no genuine miss, specificity is 100%, Support is 100%, every speed cell is correct, and the remaining limitation is already separated by a frozen secondary physical contract. No repository evidence identifies a corrective intervention with a better scientific justification than preserving the candidate and obtaining fresh HOLDOUT evidence.

## 17. Architecture readiness

The decision is `ARCHITECTURE_CHANGE_NOT_JUSTIFIED`, equivalently no architecture experiment before HOLDOUT. GRU20 remains justified. Current evidence does not justify longer history, LSTM, a larger GRU, feature redesign, or sensor expansion: the unchanged compact architecture resolved six gaps and substantially improved the seventh.

## 18. Sensor readiness

The provisional Pelvis IMU6 Hazard plus left-FSR4 Terrain design remains `10_CHANNEL_ARCHITECTURE_STILL_PLAUSIBLE`. Right-only Support is 4/4 and no genuine external Hazard miss exists with Pelvis IMU6.

`FINAL_SENSOR_ARCHITECTURE_FROZEN = NO`. The candidate freeze is only for simulation-generalization HOLDOUT. Hardware realism, deployment resources, and real-sensor evidence remain future work.

## 19. Overfitting risk

Generalization VALIDATION has served its purpose as development evidence: V1 and the promoted V2 have been evaluated, the original mechanisms are known, and several internal data/extraction refinements preceded this final comparison. Further tuning would risk fitting one known run and weakening the independence of the next claim.

The decision is `FURTHER_GENERALIZATION_VALIDATION_TUNING_NOT_JUSTIFIED`.

## 20. Final candidate eligibility

All predeclared readiness requirements pass.

| Readiness check | Result | Evidence |
|---|---|---|
| Candidate integrity | PASS | exact promotion, freeze, architecture, normalizer, checkpoints, runtime |
| Dataset integrity | PASS | Unified 256/256; V2 412/412; Generalization 72/72; Ice 48/48 |
| No leakage | PASS | frozen VALIDATION evidence only; HOLDOUT metadata-only checks |
| No genuine unresolved Hazard miss | PASS | 0 genuine misses |
| Specificity acceptable | PASS | 10/10 primary; 4/4 Ice benign |
| Support acceptable | PASS | 14/14 |
| Speed diversity acceptable | PASS | every external speed/subtype cell 2/2 |
| Known Slip limitation physically explained | PASS | sole failure is sustained frozen Ice precursor response |
| Primary metric retained | PASS | strict metric copied exactly; SHA frozen |
| Secondary metric frozen | PASS | existing Ice precursor semantics copied exactly; SHA frozen |
| No further justified optimization | PASS | no remaining evidence-based intervention |
| HOLDOUT untouched | PASS | no payload access or inference; guard 0 |
| One-shot evaluator ready | PASS | exact future config/command, candidate gate, and atomic 0-to-1 claim predeclared |
| Final verdict predeclared | PASS | primary and scientific hierarchies frozen |

The known 11/12 Slip primary failure remains explicit; its physical explanation makes one-shot evaluation justified but does not turn that failure into a pass.

## 21. Final candidate identity

The exact promoted model is designated `FINAL_GENERALIZATION_CANDIDATE`. [`configs/model/final_generalization_candidate.yaml`](../configs/model/final_generalization_candidate.yaml) is a role/reference record only; it does not contain or duplicate weights.

| Item | Frozen value |
|---|---|
| Role | `final_generalization_candidate` |
| Candidate ID | `model_v2_anchor_refined_gru20_20260902` |
| Normalizer SHA-256 | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` |
| Checkpoint SHA-256, seed 20260828 | `7094a2dca40e8d3c84554619652d69c17920c8e82765460ff8621c13ef494cb9` |
| Checkpoint SHA-256, seed 20260829 | `3ad298eea4c35eca896afd31f860fd6b44ce35d7d9978e2546bb40b693e62c39` |
| Checkpoint SHA-256, seed 20260830 | `fe96dfeb8461871044de0f8672190680ce46164a1c28efbe6738d22f9d439bbd` |
| Architecture SHA-256 | `ae4753699399aa7e4a93639935e085156ef28ed100a77058d7308eb1549aa897` |
| Feature schema SHA-256 | `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb` |
| Threshold / persistence | 0.99 / 5 ms |
| Internal evaluation SHA-256 | `cad3902137b622c3a3d15ecb3d6c3bb31ee9751f3605fb8d43daa6ac81695c07` |
| External evaluation SHA-256 | `cc559c592f64ed8afd27b039d57ace35d8bbb3efad1cac64e3d10788e0ffa556` |
| Development promotion SHA-256 | `1e4931e35e873cd721b412c6a45f66340f7ee9eebc1900d9c4aa3dc9ab3d092f` |
| Final candidate metadata SHA-256 | `52644d3efb0e756002bd43a7f94aea0a16a65eb2ff014c72548f2b3b138b7bc2` |
| Known limitation | Generalization VALIDATION Slip 11/12 < 95%; sole supported Ice precursor timing conflict |

No new weights, checkpoint, normalizer, architecture, or deployment export were created.

`FINAL_GENERALIZATION_CANDIDATE` means only “the exact candidate frozen for the one-shot simulation HOLDOUT.” It does not mean HOLDOUT supported, real-robot supported, final sensor architecture, E84 supported, production-ready, or safety-certified.

## 22. HOLDOUT primary contract

The future primary contract is byte-for-byte structurally equal to Generalization VALIDATION's primary contract. Its canonical SHA-256 is `feabfc4519e8ec28e59710810b6e587b7a8be1a128ecf57a028d32710c1b246e`.

| Metric | Definition | Gate |
|---|---|---:|
| Overall Hazard recall | established Slip OR established Support; strict valid event window | >=0.90 |
| Slip recall | any-foot drift >=0.050 m for 3 ms; first Reflex in `[-30,+40]` ms of established Slip | >=0.95 |
| Support recall | spread >=0.010 m for 20 ms; first Reflex from I1 through established Support +50 ms | >=0.85 |
| Primary no-hazard specificity | no established Slip, I1, or established Support; any Reflex is false positive | >=0.95 |
| Ice-benign specificity | frozen Ice benign subset under original primary scoring | >=0.95 |
| Premature rate | first Hazard response before applicable lower timing bound | <=0.10 |
| Slip p95 latency | established-Slip-relative latency among valid Slip responses | <=+40 ms |
| Support p95 established latency | established-Support-relative latency among valid Support responses | <=+50 ms |

The first premature response cannot be rescued. Terrain is not a Hazard gate. Threshold 0.99, persistence 5 ms, checkpoints, seeds, labels, denominators, and metric code cannot change after opening.

## 23. HOLDOUT secondary contract

The independent secondary contract uses loaded exact-Ice drift in the half-open interval `[0.030,0.050) m`, 1,000 ms future follow-up, and the frozen outcomes `SAME_EPISODE_SLIP`, `NEXT_EPISODE_SLIP`, `LATER_SLIP`, `BENIGN_RELEASE`, and `CENSORED`. Its canonical SHA-256 is `085d6f73156a5618767284faa2ccdcd29d3645694f56155431159d533b77130a`.

The future report will separately describe future-Slip precursor episodes, alerts inside precursor, alerts before established Slip, benign-release alerts, censored alerts, same/next/later Slip, precursor-to-Reflex timing, and Reflex-to-Slip timing.

The secondary metric cannot rescue, replace, or rewrite the primary score. A primary Slip failure with a supported precursor response remains `GENERALIZATION_HOLDOUT_PRIMARY_GATES_FAIL` at the primary layer and may receive a separate physical interpretation.

## 24. HOLDOUT one-shot guard policy

Current Generalization HOLDOUT guard count is 0 and this milestone does not claim it. The next milestone may transition `0 -> 1` exactly once only after resolving this readiness freeze, the exact final candidate, and guard 0. An unapproved candidate or any nonzero pre-open guard fails closed.

The future operation must open the entire 36-run split at once. It cannot inspect a family, decide whether to continue, tune, and resume. The one atomic pass includes frozen V1 and final V2 on identical data. Frozen Terrain V1 is predeclared in that same pass as advisory-only; it cannot gate Hazard or alter V2. A second scientific open is forbidden.

After the open: no training, retuning, model selection, threshold/persistence change, architecture/seed/checkpoint change, metric change, relabeling, family exclusion, or scientific rerun is permitted. The post-open contract canonical SHA-256 is `4baabe28158f7319177a9d5501e2be31eabad110706e2e32811638dfa657175a`.

## 25. Final verdict hierarchy

The predeclared hierarchy canonical SHA-256 is `e86fb11f457734c41cd7b9c66a827a22b587f7a1f95aa91130931f7586c8cba5`.

Primary status is always reported separately as exactly one of:

- `GENERALIZATION_HOLDOUT_PRIMARY_GATES_PASS`
- `GENERALIZATION_HOLDOUT_PRIMARY_GATES_FAIL`

The scientific verdict is exactly one of:

- `MODEL_V2_GENERALIZATION_HOLDOUT_SUPPORTED`: all primary gates pass.
- `MODEL_V2_GENERALIZATION_HOLDOUT_SUPPORTED_WITH_ICE_TIMING_TENSION`: primary failure is solely or predominantly Ice Slip timing with frozen physically supported precursor evidence; genuine misses are absent or isolated; specificity and Support remain strong; no new unexplained failure appears.
- `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`: genuine misses, specificity failure, Support failure, an unexplained speed/side/source failure, or a failure outside the frozen semantics is material.
- `MODEL_V2_GENERALIZATION_HOLDOUT_INCONCLUSIVE`: integrity or insufficient-evidence problem only.

Future primary Hazard failures use the predeclared categories `SUPPORTED_ICE_PRECURSOR_TIMING_CONFLICT`, `GENUINE_DETECTION_MISS`, `BENIGN_FALSE_ALERT`, `PRE_I1_SUPPORT_FALSE_ALERT`, `LATE_DETECTION`, `CENSORED_OR_AMBIGUOUS`, or `OTHER`. No inconvenient result may be repaired with a new scoring category.

## 26. HOLDOUT execution plan

The next milestone is predeclared as `MODEL_V2_GENERALIZATION_HOLDOUT_ONE_SHOT_EVALUATION`. Its exact future config path and command are:

```bash
PYTHONPATH=src python scripts/fastreflex.py evaluate \
  --config configs/experiment/20260902_model_v2_generalization_holdout_one_shot_evaluation.yaml
```

That future config is intentionally not created or run in this milestone. The next milestone must atomically claim guard 0-to-1, open all 36 runs in one scientific operation, run frozen V1 and final V2 in the same pass, run the predeclared frozen Terrain advisory diagnostic in that same pass, compute both frozen contracts, write one result, avoid all trainer/optimizer paths, and refuse a second claim.

Family reporting is predeclared for delayed Ice, Ice benign, delayed Sand Support, right Sand Support, and speed-stratified Slip/Support/Sand-benign. Descriptive speed (0.20/0.25/0.30), side (left/right/bilateral/none), and source (Concrete/Marble) denominators will be reported after opening without new subgroup gates.

## 27. Limitations

- Generalization VALIDATION primary Slip is 11/12 (91.67%), below 95%; historical `GENERALIZATION_PRIMARY_GATES_FAIL` remains binding.
- The sole failure is physically explained but still fails the strict primary metric.
- There is no external right-only Slip denominator, so unilateral Slip generalization is not established.
- Family and speed denominators are small deterministic scenario cells, not statistical population guarantees.
- Generalization VALIDATION is consumed development evidence and cannot establish final simulation generalization.
- Generalization HOLDOUT support, real-robot support, recovery effectiveness, final sensor architecture, E84 compatibility, quantization, HIL parity, production readiness, and safety certification remain unestablished.

Even a future supported HOLDOUT would leave candidate release, deployment packaging, E84 compatibility, INT8/Vela as applicable, HIL/resource/latency parity, sensor review, Recovery study, and real-sensor/hardware domain-gap validation.

## 28. Readiness verdict

```text
Review verdict:
MODEL_V2_FINAL_CANDIDATE_HOLDOUT_READINESS_REVIEW_COMPLETE

Readiness verdict:
HOLDOUT_READY

Final candidate freeze status:
FINAL_GENERALIZATION_CANDIDATE_FROZEN
```

The deterministic gitignored readiness artifact is `artifacts/runs/20260902_model_v2_final_candidate_holdout_readiness_review/readiness_review_freeze.json`, SHA-256 `0167d72942ee402b7bdcb83f5bcd3e69c62f4db8044c1bf62bfd8607487eb7c6`.

All activity counters remain zero:

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
Generalization VALIDATION new inference = NO
Generalization HOLDOUT inference = NO
Generalization HOLDOUT guard count = 0
```

The full repository suite passed 97 tests with one pre-existing skip. Targeted tests cover exact final aliasing and hashes, no artifact duplication/mutation, contract immutability, HOLDOUT fail-closed loading, access prohibition, unauthorized candidate/nonzero guard rejection, one-time atomic claim, absence of trainer/optimizer/model-array imports, deterministic metadata, and all protected datasets.

## 29. Recommended next milestone

The sole recommended next milestone is:

```text
MODEL_V2_GENERALIZATION_HOLDOUT_ONE_SHOT_EVALUATION
```

It is not started here. No additional retraining, data generation, extraction change, tuning, HNM, model architecture experiment, feature redesign, sensor expansion, Terrain retraining, deployment, quantization, HIL, Recovery, or GUI work is authorized by this readiness verdict.
