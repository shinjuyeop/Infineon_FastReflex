# Model V2 HOLDOUT Failure Interpretation

## 1. Purpose

This milestone interprets the already-consumed 36-run Generalization HOLDOUT result for the exact final Model V2 candidate. It does not reopen a HOLDOUT payload, rerun HOLDOUT inference, reconstruct a HOLDOUT feature, or alter a score. HOLDOUT claims below come only from the immutable saved result JSON and allowed manifest metadata; raw waveform, probability, feature, FSR, event, and visualization analysis is development-only.

The interpretation is actionable. Final V2 strongly generalized the Support correction and reduced premature responses, but three fresh Sand-benign false alerts and one genuine 0.30 m/s late Slip response remain. The smallest justified next milestone is a fresh Sand-benign generalization study design, not immediate Model V3 training or an architecture search.

## 2. Starting state

The starting `HEAD` and `origin/main` were both `7fdb61940a6fc60edbd0b2ad5e0726b5eb07d3b6` (`Evaluate Model V2 generalization HOLDOUT`), and the tracked worktree was clean. The immutable historical results were:

- primary: `GENERALIZATION_HOLDOUT_PRIMARY_GATES_FAIL`;
- final candidate: `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`;
- simulation research: `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`;
- data-coverage hypothesis: `DATA_COVERAGE_HYPOTHESIS_NOT_SUPPORTED`.

The Generalization HOLDOUT guard was already permanently consumed at `1`, with one scientific opening recorded at `2026-09-02T15:54:22.042348+09:00`.

Before development waveform analysis, the interpretation contract was frozen in `configs/experiment/20260902_model_v2_holdout_failure_interpretation.yaml`, SHA-256 `b93c6fdf0e3e1b39a1312c3440d2049045103427e0b21ca2bed698d4d0e2180b`.

## 3. Consumed-HOLDOUT evidence boundary

The following frozen artifacts were the only HOLDOUT result sources:

| Object | SHA-256 |
|---|---|
| One-shot execution config | `ec53c761f426aaeba5528916c60a6c3f69550007987cdf5f3754304cd4bbef0a` |
| Evaluation result | `2948449fb818335ec2e03ac0b90c34280714bac53570e05f3e67ae1c9bd839da` |
| Run-level result | `18a7a40205f59dd230ef5cbb2a838027a2bfb5764f460d0c72abf1063539152d` |
| Primary metrics | `a0e4b9436c559df6f6966debdfa17b1e22ca494299db9b9ec742650f354e8615` |
| Secondary metrics | `190e7117dc83c221a34e0002314093cbb2237d6f78603e93bcd89507f9bc3628` |
| Terrain diagnostics | `2fd5e867c7b254dc09011b435c5c1a0202e2dc48373ef98c3a98bce77aaa3730` |
| Durable guard | `0913da8a583cab7834590d669ba9c8d470485b8129fe81f538aba6c461613154` |

All hashes matched. The guard remains `1`; a second scientific open is refused. This milestone performed zero HOLDOUT payload reads and zero HOLDOUT model inference. Where a value was not persisted, this report uses `NOT_AVAILABLE_FROM_FROZEN_HOLDOUT_RESULT` rather than deriving it.

Authorized raw analysis was limited to Unified TRAIN, V2_TRAIN, V2_VALIDATION, Generalization VALIDATION, and the earlier Ice-semantics development corpus. No optimizer, normalizer fit, HNM, threshold search, persistence search, architecture search, seed search, or simulation ran.

## 4. Frozen final result

| Metric | V1 HOLDOUT | Final V2 HOLDOUT | Frozen gate | V2 status |
|---|---:|---:|---:|---|
| Overall Hazard recall | 14/28 = 50.00% | 25/28 = 89.29% | >=90% | **FAIL** |
| Slip recall | 8/14 = 57.14% | 11/14 = 78.57% | >=95% | **FAIL** |
| Support recall | 6/14 = 42.86% | 14/14 = 100% | >=85% | PASS |
| Primary no-hazard specificity | 5/8 = 62.50% | 5/8 = 62.50% | >=95% | **FAIL** |
| Physical Ice-benign specificity | 2/2 = 100% | 2/2 = 100% | >=95% | PASS |
| Premature Hazard rate | 9/28 = 32.14% | 2/28 = 7.14% | <=10% | PASS |
| Slip median / p95 latency | -21 / -17 ms | -13 / +7.5 ms | p95 <=+40 ms | PASS |
| Support median / p95 established latency | -20 / -17.25 ms | -23 / -17 ms | p95 <=+50 ms | PASS |

The exact frozen candidate remains `model_v2_anchor_refined_gru20_20260902`: Pelvis IMU6, causal 80D `[20,80]`, one-layer unidirectional GRU hidden 32, 11,010 parameters, three-seed mean, threshold 0.99, and persistence 5 ms.

## 5. Failure decomposition

The six final-V2 primary failures are not one mechanism:

| Mechanism | Count | Runs | Gate consequence |
|---|---:|---|---|
| Supported Ice precursor timing conflict | 2 | `ghr_ibc_h_c020`, `ghr_ocd_h_c020` | Hazard and Slip strict-primary misses |
| Genuine late detection | 1 | `ghr_ssh_is_h_c030` | Hazard and Slip strict-primary miss |
| Benign Sand false alert | 3 | `ghr_ssh_sb_h_c020`, `ghr_ssh_sb_h_c030`, `ghr_ssh_sb_h_m030` | specificity collapse |
| Pre-I1 Support false alert | 0 | — | none |
| Literal `GENUINE_DETECTION_MISS` category | 0 | — | none |

The overall Hazard gate missed by one correct run: 25/28 is 89.29%, below 90%. The Slip gate is blocked by two known semantic-timing conflicts and one genuine +42 ms late response. Specificity is blocked independently by the three Sand false alerts.

`DIAGNOSTIC DECOMPOSITION ONLY`: describing the two already-supported Ice precursor responses separately would yield 27/28 = 96.43% physical Hazard response and 13/14 = 92.86% physical Slip response. Support remains 14/14, specificity remains 5/8, and the +42 ms late run remains a failure. This is not an alternative score or verdict. The official 25/28 Hazard, 11/14 Slip, gates, and verdicts are unchanged.

## 6. Support branch result

Support is the clear success. Final V2 improved V1 from 6/14 to 14/14 and passed every Support subgroup in the fresh HOLDOUT:

- delayed Sand Support: 4/4, with Reflex at I1 +4 ms and 52 ms before established Support;
- right-only Sand Support: 4/4, improving V1 0/4;
- speed-stratified Sand Support: 6/6 across 0.20, 0.25, and 0.30 m/s;
- pre-I1 Support false alerts: zero.

This is `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`. It is branch-specific evidence and does not replace the failed whole-model verdict.

## 7. Sand benign false alerts

All six speed-Sand HOLDOUT runs are physically no-hazard. Final V2 false-alerted on Concrete/0.20, Concrete/0.30, and Marble/0.30. The saved member maxima were high in all three failures, and ensemble persistence was 13, 7, and 9 ms respectively; these are coherent ensemble responses rather than one-seed spikes.

| Run | Source/speed | V2 max p | First crossing / Reflex | `>=.99` max streak | Seed maxima | Classification |
|---|---|---:|---:|---:|---|---|
| `ghr_ssh_sb_h_c020` | C/.20 | .999232 | 5150 / 5154 | 13 ms | .998859/.999206/.999891 | false alert |
| `ghr_ssh_sb_h_c030` | C/.30 | .993631 | 4029 / 4033 | 7 ms | .992218/.991185/.998743 | false alert |
| `ghr_ssh_sb_h_m030` | M/.30 | .997827 | 4294 / 4298 | 9 ms | .995046/.999046/.999451 | false alert |

HOLDOUT FSR, contact state, physical phase at Reflex, feature distance, and waveform morphology were not persisted: `NOT_AVAILABLE_FROM_FROZEN_HOLDOUT_RESULT`.

## 8. Sand speed/source matrix

The required result table distinguishes frozen-model TRAIN replay from held-out development and saved HOLDOUT results. TRAIN replay is descriptive and is not validation evidence.

| Evidence set | Speed | Source | N | Correct | FP | Model |
|---|---:|---|---:|---:|---:|---|
| V2_TRAIN replay | .20 | Concrete | 6 | 6 | 0 | final V2 |
| V2_TRAIN replay | .20 | Marble | 6 | 6 | 0 | final V2 |
| V2_TRAIN replay | .25 | Concrete | 6 | 6 | 0 | final V2 |
| V2_TRAIN replay | .25 | Marble | 6 | 6 | 0 | final V2 |
| V2_TRAIN replay | .30 | Concrete | 6 | 6 | 0 | final V2 |
| V2_TRAIN replay | .30 | Marble | 6 | 6 | 0 | final V2 |
| V2_VALIDATION | .20 | Concrete | 2 | 2 | 0 | final V2 |
| V2_VALIDATION | .20 | Marble | 2 | 2 | 0 | final V2 |
| V2_VALIDATION | .25 | Concrete | 2 | 2 | 0 | final V2 |
| V2_VALIDATION | .25 | Marble | 2 | 2 | 0 | final V2 |
| V2_VALIDATION | .30 | Concrete | 2 | 2 | 0 | final V2 |
| V2_VALIDATION | .30 | Marble | 2 | 2 | 0 | final V2 |
| Generalization VALIDATION | .20 | Concrete | 1 | 1 | 0 | final V2 |
| Generalization VALIDATION | .20 | Marble | 1 | 1 | 0 | final V2 |
| Generalization VALIDATION | .25 | Concrete | 1 | 1 | 0 | final V2 |
| Generalization VALIDATION | .25 | Marble | 1 | 1 | 0 | final V2 |
| Generalization VALIDATION | .30 | Concrete | 1 | 1 | 0 | final V2 |
| Generalization VALIDATION | .30 | Marble | 1 | 1 | 0 | final V2 |
| Generalization HOLDOUT, saved | .20 | Concrete | 1 | 0 | 1 | final V2 |
| Generalization HOLDOUT, saved | .20 | Marble | 1 | 1 | 0 | final V2 |
| Generalization HOLDOUT, saved | .25 | Concrete | 1 | 1 | 0 | final V2 |
| Generalization HOLDOUT, saved | .25 | Marble | 1 | 1 | 0 | final V2 |
| Generalization HOLDOUT, saved | .30 | Concrete | 1 | 0 | 1 | final V2 |
| Generalization HOLDOUT, saved | .30 | Marble | 1 | 0 | 1 | final V2 |

The decisive pattern is 0.25 m/s 2/2, 0.20 m/s mixed by source, and 0.30 m/s 0/2. This supports `SPEED_CONDITIONED_GENERALIZATION_FAILURE` and a lower-confidence `SOURCE_SPEED_INTERACTION`; it is not a monotonic source-only effect. The family status is `SAND_BENIGN_GENERALIZATION_NOT_ESTABLISHED`.

## 9. V1/V2 false-positive identity comparison

| Run | Family | Speed | Source | V1 result | V2 result | Same FP? | Stored max p, V1 / V2 |
|---|---|---:|---|---|---|---|---:|
| `ghr_ibc_h_c030` | Ice benign | .30 | Concrete | TN | TN | no | .553611 / .540740 |
| `ghr_ibc_h_m030` | Ice benign | .30 | Marble | TN | TN | no | .622182 / .509163 |
| `ghr_ssh_sb_h_c020` | Speed Sand benign | .20 | Concrete | TN | **FP** | no | .996851 / .999232 |
| `ghr_ssh_sb_h_c025` | Speed Sand benign | .25 | Concrete | **FP** | TN | no | .997720 / .971435 |
| `ghr_ssh_sb_h_c030` | Speed Sand benign | .30 | Concrete | TN | **FP** | no | .912050 / .993631 |
| `ghr_ssh_sb_h_m020` | Speed Sand benign | .20 | Marble | **FP** | TN | no | .999145 / .950155 |
| `ghr_ssh_sb_h_m025` | Speed Sand benign | .25 | Marble | TN | TN | no | .697499 / .880940 |
| `ghr_ssh_sb_h_m030` | Speed Sand benign | .30 | Marble | **FP** | **FP** | yes | .999807 / .997827 |

V1 FP set is `{c025, m020, m030}` and V2 FP set is `{c020, c030, m030}`. Their intersection is `{m030}`, union size is five, and Jaccard overlap is `1/5 = 0.20`. Classification: `PARTIAL_OVERLAP`. V2 fixed two V1 false-positive cells, introduced two different cells, and retained Marble/.30. This is evidence of decision-boundary instability plus one persistent difficult cell, not one invariant three-run failure region.

## 10. TRAIN/V2_VAL/Gen-VAL Sand coverage

All six source-speed cells have the same counts. The topology shown is the closest left-transition topology because both Generalization splits use `transition_left` / `left_balanced_deformable`.

| Speed | Source | TRAIN runs | TRAIN fit negative endpoints | HNM selections, 3 rounds | V2_VAL runs | Gen VAL runs | HOLDOUT runs | HOLDOUT V2 FP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| .20 | Concrete | 3 left-transition (6 total) | 157 | 108 | 2 | 1 | 1 | 1 |
| .20 | Marble | 3 left-transition (6 total) | 160 | 108 | 2 | 1 | 1 | 0 |
| .25 | Concrete | 3 left-transition (6 total) | 245 | 108 | 2 | 1 | 1 | 0 |
| .25 | Marble | 3 left-transition (6 total) | 246 | 108 | 2 | 1 | 1 | 0 |
| .30 | Concrete | 3 left-transition (6 total) | 241 | 108 | 2 | 1 | 1 | 1 |
| .30 | Marble | 3 left-transition (6 total) | 241 | 108 | 2 | 1 | 1 | 1 |

HNM scored every one of the 442 effective TRAIN runs and selected 12 windows per run in each round. Therefore each three-run left-transition cell contributed 36 selected windows per round, 108 cumulatively. Across all speed-Sand runs the family contributed 432 windows per round. Exact HNM endpoint/contact-phase provenance was not persisted.

The three failed source-speed cells are coarse-cell covered, but exact Generalization geometry is not in V2_TRAIN or V2_VALIDATION. They are classified `METADATA_CELL_SPARSE`, not `METADATA_CELL_UNSEEN` and not `METADATA_CELL_WELL_COVERED`.

## 11. Sand metadata-domain shift

| Evidence set | Topology | Patch start | Patch width | Runs per source-speed cell | Result |
|---|---|---:|---:|---:|---|
| V2_TRAIN | 3 left + 3 right | .304-.334 m | .718-.738 m | 6 | 36/36 TN in final replay |
| V2_VALIDATION | 1 left + 1 right | .342/.348 m | .742/.746 m | 2 | 12/12 TN |
| Generalization VALIDATION | left | .358 m | .725 m | 1 | 6/6 TN |
| Generalization HOLDOUT, saved metadata | left | .362 m | .735 m | 1 | 3/6 TN |

The HOLDOUT start is 4 mm beyond Generalization VALIDATION and 14 mm beyond the largest V2_VALIDATION start. Its width lies inside the TRAIN width range but differs from the one adjacent Generalization VALIDATION width by 10 mm. All six HOLDOUT cells share that geometry, so geometry novelty alone cannot explain why only three alert. Speed and source-conditioned dynamics must interact with the shift.

Saved/reproducible development metadata also rules out a single target-contact-duration threshold: VALIDATION-to-HOLDOUT target-contact durations changed by -25 ms (C/.20), +541 (C/.25), +75 (C/.30), +152 (M/.20), -21 (M/.25), and -3 (M/.30), while the false-alert pattern was C/.20, C/.30, and M/.30. The strongest supported description is: speed strata were necessary but too coarse because within-speed geometry/contact/phase diversity remained narrow.

## 12. Development Sand probability margins

Final V2 was replayed read-only on authorized development data. A threshold-near run has maximum ensemble probability in `[.95,.99)`; an above-threshold/subpersistent run has max `>=.99` but fewer than five consecutive samples.

| Development group | N | Median max p | p90 | p95 | `[.95,.99)` | `>=.99`, <5 ms | Reflex |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2_TRAIN Speed Sand | 36 | .779207 | .898661 | .935462 | 1 | 0 | 0 |
| V2_VALIDATION Speed Sand | 12 | .953668 | .987605 | .990276 | 5 | 1 | 0 |
| Generalization VALIDATION Speed Sand | 6 | .986838 | .992835 | .993071 | 3 | 2 | 0 |
| V2_TRAIN staged Sand | 26 | .364359 | .387780 | .518881 | 0 | 0 | 0 |
| V2_VALIDATION staged Sand | 8 | .357028 | .387780 | .387780 | 0 | 0 | 0 |
| V2_VALIDATION hard normal | 6 | .652511 | .763261 | .765736 | 0 | 0 | 0 |

In Generalization VALIDATION, Concrete/.30 and Marble/.20 each sustained `>=.99` for 4 ms—one millisecond short of the frozen decision—and four of six runs had maxima at least .98588. Binary 6/6 specificity therefore concealed a progressively thin margin. HOLDOUT moved Concrete/.30 from 4 to 7 ms and introduced sustained responses in Concrete/.20 and Marble/.30.

Classification: `LOW_MARGIN_BENIGN_GENERALIZATION`, confidence HIGH. It supports insufficient fresh negative diversity and a decision-boundary generalization failure; it does not justify raising the threshold or persistence.

Existing-domain HNM was extensive rather than absent. Its selected-probability median/p95 declined from `.8248/.9852` to `.5886/.9280` to `.3427/.7888` over three rounds, while every speed-Sand TRAIN run contributed each time. Simple duplication of the same scenarios is therefore less compelling than fresh physical diversity.

## 13. Support/Sand feature overlap

Development-only windows were compared using raw Pelvis IMU6, the frozen final normalizer, causal 80D features, and flattened `[20,80]` histories. One benign risk window per run used the final-V2 maximum; one Support window per run used the maximum within established Support `[-20,+40]` ms. No classifier or probe was trained.

Across 88 benign Sand and 124 Support windows, selected raw acceleration norms overlapped: speed-Sand median was 11.106 m/s² with range 7.869-23.795, delayed Support was 11.100 with range 11.095-11.105, and ordinary Support median was 12.093 with range 7.733-13.347. Gyro means likewise overlapped around the central region: speed-Sand median .371 rad/s, delayed Support .222, and ordinary Support .265.

In pooled standardized flattened-feature distance, centroid separation was only `.507` within-group RMS units. Leave-one-out nearest-neighbor group agreement was 94.3%, but all Support query windows lay inside the broad 95% benign centroid radius. Exact repeated development waveforms and unequal group sizes limit these descriptive statistics; they nevertheless show an asymmetric benign region broad enough to contain Support-like windows.

The earlier controlled regression audit supplies the strongest local evidence. Regressed benign Sand windows were closest among added delayed-Support anchors to dense Support-local endpoints (normalized current/window median distances `3.885/42.196`) and were physically single-left loaded with zero Support spread. Removing that dense anchor restored V2_VALIDATION specificity, but current Generalization margins remained thin and HOLDOUT failed on a different exact geometry. The supported interpretation is `RESIDUAL_SAND_SUPPORT_FEATURE_OVERLAP` with MODERATE confidence: anchor refinement solved one training-induced overlap mode, not every fresh Sand transient.

## 14. FSR/contact diagnostic separability

Development-only diagnostic comparison used FSR8 mean/max/std plus loaded-contact, target-contact, Support-spread, and displacement summaries over the same 20 ms windows. Its centroid separation rose from `.507` for Pelvis features to `.693`, and local 1-nearest-neighbor group agreement rose from 94.3% to 100%. The global picture was not clean: 81.8% of benign points fell inside the Support 95% radius and 85.5% of Support points fell inside the benign radius.

Prior event-local evidence explains the ambiguity. Regressed benign FSR norms were `202.70-210.73`, overlapping delayed-Support I1 `199.32`, midpoint `236.35`, and Support-local `221.72`. Privileged Support spread cleanly differed—zero for the benign failures versus 2.820/8.419/12.749 mm across delayed-Support anchor stages—but spread is not current runtime input and is not itself proof that FSR will carry the same information robustly.

Conclusion: contact/load diagnostics have descriptive value, but this study does not show a sufficiently clean FSR increment to select sensor fusion ahead of fresh domain coverage. A future Hazard FSR ablation remains a legitimate later question. Terrain must remain advisory; these results do not justify Terrain gating Hazard.

## 15. 0.30 Slip late detection

The failed saved HOLDOUT row is exact under the frozen contract:

| Field | Saved value |
|---|---|
| Run | `ghr_ssh_is_h_c030` |
| Source / speed / side | Concrete / .30 m/s / bilateral |
| Family | `SPEED_STRATIFIED_HAZARD / ICE_SLIP` |
| Target contact / precursor / established Slip | 1227 / 1884 / 2043 |
| First threshold crossing / first Reflex | 1917 / 2085 |
| Established-Slip-relative latency | **+42 ms** |
| Max p / max `>=.99` streak | .999209 / 75 ms |
| Seed maxima | .999451/.999277/.999454 (`ALL_3_HIGH`) |
| Terrain first valid / order | 2143 / Reflex 58 ms before Terrain |
| Frozen result | `OUT_OF_VALID_WINDOW`, `LATE_DETECTION`, genuine failure |

The response is high-confidence and sustained, but the first valid Reflex is 2 ms beyond the immutable +40 ms upper bound. It remains a primary failure.

The closest development evidence is more revealing than an unseen-cell account:

| Evidence | Source | Speed | Side | Family | Result | Latency | Metadata-cell coverage |
|---|---|---:|---|---|---|---:|---|
| V2_TRAIN replay, 6 immediate rows | C | .30 | bilateral | baseline immediate Ice | 6 Reflex; one late | -21,-13,+9,+42,0,0 ms | same family/source/speed/side/topology; starts .310-.326 |
| Closest V2_TRAIN row `...i04` | C | .30 | bilateral | baseline immediate Ice | late | **+42 ms** | same .322 start; .750 width vs .722 HOLDOUT |
| V2_VALIDATION, 2 immediate rows | C | .30 | bilateral | baseline immediate Ice | 2 correct | -21,+20 ms | starts .312/.320; widths .720/.744 |
| Generalization VALIDATION | C | .30 | bilateral | speed Ice Slip | correct | +9 ms | .318/.718 |
| Generalization HOLDOUT, saved | C | .30 | bilateral | speed Ice Slip | **late** | **+42 ms** | .322/.722 |

The closest TRAIN replay and saved HOLDOUT result have the same established Slip, maximum probability, 75 ms streak, and +42 ms latency to displayed precision despite the width difference. This comparison does not read a HOLDOUT waveform; it shows that a dynamically close development analogue already exposed the same strict-boundary weakness.

Classifications: `METADATA_CELL_WELL_COVERED_MODEL_FAILURE` and `STRICT_BOUNDARY_BORDERLINE_FAILURE`, both HIGH confidence. It is not a broad low-confidence detector failure, but it is also not isolated only to one fresh run.

## 16. Right-only Slip question

The one actual right-only HOLDOUT Slip is `ghr_ibc_h_c020`, not the +42 ms late case. It responded inside a frozen future-Slip precursor and is one of the two `SUPPORTED_ICE_PRECURSOR_TIMING_CONFLICT` rows. The late case is bilateral.

Therefore the HOLDOUT does not add a new right-side observability failure mechanism. It also cannot establish right-only robustness from N=1. The correct interpretation is `RIGHT_SLIP_OBSERVABILITY_STILL_NOT_DEMONSTRATED_AS_LIMIT`; a fresh study would still be required before claiming right-only generalization.

## 17. Slip development coverage

Effective TRAIN contained eight right-only Slip runs but only 75 right-only positive endpoints, versus 1,470 bilateral endpoints and 135 left-only endpoints. V2_VALIDATION right-only Slip was 0/3 under the strict primary timing window, but all three were sustained high responses inside future-Slip precursor regions. Generalization VALIDATION contributed no right-only Slip denominator; its speed-Slip and delayed-Ice cases were bilateral.

For 0.30 m/s broadly, final replay found 18/21 Reflex-bearing V2_TRAIN Slip runs, 8/8 V2_VALIDATION runs, and 4/4 Generalization VALIDATION runs. The closest immediate Concrete subgroup contains the development +42 ms analogue described above. This makes event-timing/target behavior the supported issue for the observed HOLDOUT late case, not missing signal amplitude, model confidence, or side observability.

The branch interpretation is `SLIP_PRIMARY_GENERALIZATION_NOT_SUPPORTED` and `SLIP_PHYSICAL_RESPONSE_PARTIALLY_SUPPORTED`. The latter is descriptive only and does not replace the frozen primary score.

## 18. Ice precursor timing conflicts

Both saved timing-conflict runs satisfy the already-frozen secondary interpretation:

| Run | Side | Precursor | Reflex | Slip | V2 streak | Secondary class | Benign/censor ambiguity |
|---|---|---:|---:|---:|---:|---|---|
| `ghr_ibc_h_c020` | right-only | 2466 | 2478 | 2628 | 7 ms | supported future-Slip precursor | none saved |
| `ghr_ocd_h_c020` | bilateral | 2466 | 2478 | 2632 | 19 ms | supported future-Slip precursor | none saved |

Both respond after the 30 mm precursor onset and well before established Slip. The saved secondary aggregate has zero benign-release alerts and zero censored-region alerts. These are known primary-metric versus physical-precursor semantics tension, not a new absence-of-response mechanism. Primary results remain failures and are not relabeled.

## 19. Validation-to-HOLDOUT gap

| Evidence set | Hazard | Slip | Support | Specificity | Premature |
|---|---:|---:|---:|---:|---:|
| V2_VALIDATION | 59/64 | 30/35 | 30/30 | 26/26 | 5/64 |
| Generalization VALIDATION | 25/26 | 11/12 | 14/14 | 10/10 | 1/26 |
| Generalization HOLDOUT | 25/28 | 11/14 | 14/14 | 5/8 | 2/28 |

Support stayed exact and premature behavior stayed within gate. The fresh deterioration is concentrated in Sand specificity and one speed-Slip timing cell. Generalization VALIDATION Sand already showed 4 ms threshold excursions and p95 max `.993071`; the HOLDOUT geometry/speed/source shift converted that hidden margin into three false alerts. Thus the validation-to-HOLDOUT gap is not unexplained binary noise: it is consistent with low-margin negative generalization plus one known timing-edge analogue.

The largest fresh shift is specificity: 100% on Generalization VALIDATION to 62.50% on HOLDOUT, a -37.50 percentage-point change. That family-level collapse is more concerning than the one-run overall-Hazard gate miss because it introduces three false Reflex actions in only eight no-hazard runs and reproduces across both sources at 0.30 m/s.

## 20. Coverage hypothesis reinterpretation

The historical verdict remains exactly `DATA_COVERAGE_HYPOTHESIS_NOT_SUPPORTED`. A more precise interpretation, without rewriting that label, is:

`DATA_COVERAGE_ALONE_INSUFFICIENT`

Coverage and extraction were necessary and highly beneficial: Hazard improved by 39.29 percentage points, Slip by 21.43 points, Support by 57.14 points, and premature rate by 25 points. They were insufficient to guarantee fresh Sand-benign rejection and strict Slip timing. Coarse source/speed coverage did not provide enough within-cell geometry/contact/phase diversity.

## 21. Architecture evidence

| Evidence question | Evidence for | Evidence against | Confidence |
|---|---|---|---|
| Training-domain limitation | exact Sand geometry unseen; narrow within-speed patterns; fresh shift | coarse cells and extensive HNM were present | HIGH |
| Representation/generalization limitation | low-margin Sand, prior Support/Sand overlap, FP identities changed | most development and half of HOLDOUT Sand remain correct | MODERATE |
| Temporal-memory limitation | one strict +42 ms Slip; 20 ms history is short | high-confidence 75 ms response; Sand specificity dominates failures | LOW |
| Capacity limitation | decision boundary fails on fresh cells | 11,010 parameters solve all Support and many new cells | LOW |
| Sensor-observability limitation | Pelvis overlap; contact diagnostics add information | FSR norm overlaps and exact Sand domain remains sparse | LOW/MODERATE |

Architecture verdict: `ARCHITECTURE_EVIDENCE_STILL_FAVORS_DATA_DOMAIN_STUDY`.

Final HOLDOUT does not yet justify LSTM, longer history, or a larger GRU experiment. One +42 ms response does not prove memory deficiency, and changing memory cannot be assumed to solve Sand false alerts. The provisional 10-channel system architecture remains unfrozen.

## 22. Sensor observability evidence

Pelvis-only Hazard conclusion: `PELVIS_ONLY_HAZARD_STILL_PLAUSIBLE`.

Pelvis features exhibit residual benign/Support overlap, but the failures also occupy a fresh exact Sand geometry with known thin development margins. FSR/contact descriptors improve some local descriptive separation but do not cleanly separate the full development groups, and no fusion model was trained. Consequently a Hazard sensor-observability study is reasonable only after the smaller data-domain study determines whether fresh scenario diversity closes the failure mode.

Terrain remains a separate advisory model. Its saved HOLDOUT accuracy was 271/349 = 77.65% over clean evaluable events; target Terrain was available in 35/36 runs. It remains `advisory_only=true`, `hazard_gate=false`.

## 23. Failure attribution matrix

| Failure mechanism | Frozen HOLDOUT evidence | Development evidence | Likely root cause | Confidence | Next-study implication |
|---|---|---|---|---|---|
| Sand FP C/.20 | .999232, 13 ms, all seeds high | coarse cell/HNM present; exact geometry unseen; C/.20 Gen VAL .956 | domain shift plus source-speed decision boundary | MODERATE | add fresh within-speed geometry/phase diversity |
| Sand FP C/.30 | .993631, 7 ms | Gen VAL C/.30 already 4 ms `>=.99`; exact geometry unseen | low-margin speed-conditioned generalization | HIGH | prioritize .30 negative diversity without copying HOLDOUT |
| Sand FP M/.30 | .997827, 9 ms; also V1 FP | Gen VAL M/.30 .987797; coarse cell/HNM present | persistent difficult .30 region plus domain shift | HIGH | require multiple fresh .30 patterns and frozen external validation |
| Slip late C/.30 | bilateral +42 ms, .999209, 75 ms | close TRAIN analogue also +42; validation analogues -21/+20/+9 | strict-boundary timing failure in a represented dynamic mode | HIGH | retain as secondary fresh-development timing question |
| Ice precursor conflict | two early responses inside precursor | repeated known semantics; no benign/censor alert | primary metric versus supported precursor semantics | HIGH | no metric rewrite and no new mechanism claim |
| Right-only Slip | N=1; the timing-conflict row, not late row | sparse 75 positive endpoints; early sustained V2_VAL responses | robustness unknown, observability limit not demonstrated | MODERATE | do not infer side robustness; fresh study required if prioritized |

Sand alert classification is jointly `SAND_SCENARIO_METADATA_COVERAGE_GAP`, `MODEL_DECISION_BOUNDARY_GENERALIZATION_FAILURE`, `SPEED_CONDITIONED_GENERALIZATION_FAILURE`, and `RESIDUAL_SAND_SUPPORT_FEATURE_OVERLAP`. Confidence is HIGH for the first three taken together, MODERATE for residual feature overlap, and MODERATE for `SOURCE_SPEED_INTERACTION` because each cell has N=1.

## 24. Branch-level research status

| Branch | V1 HOLDOUT | V2 HOLDOUT | Status | Main limitation |
|---|---:|---:|---|---|
| Support detection | 6/14 | 14/14 | `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED` | simulator-only evidence |
| Slip detection | 8/14 | 11/14 | `SLIP_SIMULATION_GENERALIZATION_NOT_SUPPORTED` | two strict semantic conflicts plus one genuine late response |
| Benign rejection | 5/8 | 5/8 | `BENIGN_REJECTION_GENERALIZATION_NOT_SUPPORTED` | three Sand false alerts |
| Terrain advisory | 271/349 clean events | same saved advisory | advisory only | 77.65% accuracy; never gates Hazard |

The historical whole-system status remains `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`.

## 25. Old HOLDOUT future-use prohibition

The consumed `generalization_hazard_reflex_20260831 / GENERALIZATION_HOLDOUT` is permanently historical evidence.

| Future use | Status |
|---|---|
| Raw runs for training or hard-negative extraction | FORBIDDEN |
| Run-level signals or reconstructed features for training | FORBIDDEN |
| Saved model scores for tuning/model selection | FORBIDDEN |
| Threshold/persistence behavior for operating-point search | FORBIDDEN |
| Reopening, reinference, visualization, or event recomputation | FORBIDDEN |
| High-level consumed-test failure interpretation | allowed |

Any modified candidate requires a new development corpus and eventually a new independent final HOLDOUT. The old 36 runs may never again be represented as fresh final evidence.

## 26. Smallest justified next study

Exactly one next milestone is selected:

`SAND_BENIGN_GENERALIZATION_STUDY_DESIGN`

That design should predeclare a new development matrix spanning source, speed, left/right transition topology, patch start/width, gait/contact phase, and contact-duration diversity. It must use signatures distinct from every historical corpus, freeze split membership before simulation, and include an external development validation split. It may test whether the current Pelvis-only representation remains viable, but it must not copy old HOLDOUT windows, use old HOLDOUT scores as targets, or tune the 0.99/5 ms operating point.

The 0.30 Slip timing weakness should be retained as a documented secondary question, not used to broaden this next milestone into Model V3 or an architecture search. A future modified candidate must eventually face a newly designed independent final HOLDOUT whose contract is frozen before its first and only open.

## 27. Limitations

- The HOLDOUT contains only one run per Sand source-speed cell; source-speed claims are descriptive, not statistical.
- No HOLDOUT waveform, feature, FSR, contact state, event recomputation, or plot was available by policy.
- Development distance analyses use selected windows, repeated simulator waveforms, unequal group sizes, and privileged diagnostics; they are descriptive and not trained probes.
- Exact HNM endpoints and contact phase were not persisted, although every-run and family exposure counts were.
- The +42 ms development/HOLDOUT comparison uses a close, not exact, metadata signature; equality of stored response values does not prove byte-identical waveforms.
- No real-robot, hardware realism, safety, production, deployment, or final sensor-architecture claim follows.

## 28. Verdict

`MODEL_V2_HOLDOUT_FAILURE_INTERPRETATION_ACTIONABLE`

The next question is identifiable without reopening the HOLDOUT: determine whether fresh within-speed Sand geometry/contact/phase diversity can produce robust benign rejection while preserving solved Support. The historical final verdict remains `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`.

Scientific counters for this milestone are:

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
HOLDOUT scientific opens total = 1
HOLDOUT payload reads this milestone = 0
HOLDOUT model inference this milestone = 0
```

Verification passed:

- the full `pytest` suite ran under a process-wide exact-path block on all 36 Generalization HOLDOUT NPZ files: `106 passed, 1 skipped`;
- saved-result verification performed no HOLDOUT payload deserialization, the raw loader remained fail-closed, and a second guard claim was refused;
- protected dataset integrity was exact for Unified 256/256, Model V2 412/412, Generalization 72/72, and Ice semantics 48/48 using metadata/hash checks only;
- the final candidate record, normalizer, three checkpoints, and all seven consumed-result/guard hashes remained exact;
- `compileall src scripts tests`, critical Ruff `E9,F63,F7,F82`, and `git diff --check` passed.

## 29. Recommended next milestone

`SAND_BENIGN_GENERALIZATION_STUDY_DESIGN`

| Candidate next direction | Evidence for | Evidence against | Recommendation |
|---|---|---|---|
| More data of the same type | known negative family remains difficult | all existing speed-Sand TRAIN runs had three HNM passes | do not duplicate the current domain |
| More scenario diversity | exact Sand geometry unseen; within-speed patterns narrow | coarse cells already present | **select as primary** |
| Better negative diversity | low-margin validation; three fresh FPs | extensive same-domain HNM already ran | include as fresh physical diversity, not duplicate windows |
| Slip-target study | represented +42 ms analogue | only one genuine HOLDOUT late case; Sand has three FPs | retain as secondary question |
| Longer GRU history | one late Slip | sustained high response; no Sand remedy shown | not yet |
| LSTM | possible temporal capacity | no memory-controlled evidence | not yet |
| Larger GRU | possible capacity | all Support solved; capacity not isolated | not yet |
| Feature redesign | residual Pelvis overlap is plausible | exact-domain sparsity has not been controlled | defer pending domain study |
| Hazard FSR fusion study | contact diagnostics add information | FSR overlap; exact Sand domain sparse | defer until domain study |
| Foot IMU study | possible local contact motion | no direct current evidence | defer |
| Threshold tuning | some margins are close | consumed-result fitting and specificity/recall tradeoff | reject |
| Persistence tuning | 4 ms development excursions | consumed-result fitting; would change frozen contract | reject |

This milestone does not start the recommended study.
