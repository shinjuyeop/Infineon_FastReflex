# Controls-Recalibrated Factor-Conditioned Development Generation

## Starting state and scientific boundary

- Repository: `/d/shin/Infineon_FastReflex`
- Branch: `main`
- Starting `HEAD` and `origin/main`: `48e21858f777bac52876848ddff2df9448f08413`
- Expected scientific base ancestor: yes, exact starting commit
- Starting tracked worktree: clean
- Parallel changes: none observed

This milestone performed only physical simulation, censor-aware physical labeling, frozen generation-gate evaluation, and dataset freezing. It performed no V1, V2, or Terrain inference; Hazard-probability or 80D/model analysis; training, optimizer work, HNM, normalizer fitting, tuning, seed search, architecture search, or sensor-fusion experiment. It did not read a historical HOLDOUT payload and did not modify the E84 repository. No generated record was replaced, backfilled, rerun, relabeled for yield, or selectively promoted.

## Frozen design integrity

| Artifact | SHA-256 | Result |
|---|---|---|
| Controls-recalibrated design | `b18be44668f1d0e2c07b6a127c7fe626d42636a002ad023e48721af7c2443fb5` | PASS |
| Ordinary-Support review config | `2e01a24771de138442e30afce3967f63dbbbc45e06e3b8057d8fb87f96c81ee5` | PASS |
| Ordinary-Support readiness contract | `3518bb4b8cdeec8b47b59f9ed2bd8ccaadc02243c27fe17f54f04648a2c88deb` | PASS |
| Ordinary-Support physical ledger | `dc5b9c2149ff5ad04686d09c6d99bce9c6067cf124eb0063e6cbd68c6aeaa696` | PASS |
| Ordinary-Support review report | `8644ecfbe6c0cdcc89d96823d918d46adcf47649ece7806df419da5fc19db18c` | PASS |
| Prior delayed-Support design | `b1f09effbb90313ef8c883db8e0ef376c5f4eb8e83ec66d8ed5fe52ca95ef775` | PASS |
| Historical HOLDOUT guard | `0913da8a583cab7834590d669ba9c8d470485b8129fe81f538aba6c461613154` | PASS (`guard_after=1`, `scientific_open_count=1`) |

The expanded matrix had 198/198 unique run IDs and 198/198 unique planned scenario signatures. Historical exact overlap, historical forbidden-near overlap, historical run-ID reuse, cross-split exact overlap, and cross-split forbidden-near overlap were all zero before run 1. The successful Sand domain and coupled transition-left/right-single versus transition-right/left-single manifolds were preserved, including the Concrete/.25 adverse-only exception. Delayed Support retained the exact LEFT_ONLY, transition-left, staged-lateral-deformable `.324–.332/.825–.833/1.153–1.165 m` family. Ordinary Support used the frozen source-speed-conditioned design, including Concrete/.30 RIGHT_ONLY, the Marble/.30 higher-start left strip, and exclusion of the Concrete/.20 late/long left corner. Physical-label and Support semantics were unchanged.

## Generation

| Item | Result |
|---|---:|
| Dataset ID | `sand_factor_conditioned_development_controls_recalibrated_20260903` |
| Planned / attempted / completed | 198 / 198 / 198 |
| FACTOR_TRAIN / FACTOR_VALIDATION | 132 / 66 |
| Mild / Moderate / ordinary Support / delayed Support | 108 / 36 / 36 / 18 |
| Runtime | 1,342.589 s |
| NPZ files | 198 |
| NPZ bytes | 94,550,565 |
| Total files | 206 |
| Dataset-directory bytes (`du -sb`) | 96,926,762 |
| Replacement / backfill / rerun | 0 / 0 / 0 |

All rows ran once in the frozen split, source-speed, and profile order. Environment provenance was Python 3.10.12 on Linux 6.8.0-136-generic x86_64, Asia/Seoul. The pre-simulation freeze records that no mutation was allowed after generation start.

## Overall physical outcomes

| Outcome | FACTOR_TRAIN | FACTOR_VALIDATION | Total |
|---|---:|---:|---:|
| Planned | 132 | 66 | 198 |
| Completed | 132 | 66 | 198 |
| Objective valid | 132 | 64 | 196 |
| Strict Sand | 96 | 46 | 142 |
| Mild strict | 72 | 35 | 107 |
| Moderate strict | 24 | 11 | 35 |
| Ordinary Support | 24 | 12 | 36 |
| Delayed Support | 12 | 6 | 18 |
| Slip | 0 | 1 | 1 |
| Dual Hazard | 0 | 0 | 0 |
| Pretarget fall | 0 | 0 | 0 |
| Post-target fall/censor | 0 | 1 | 1 |
| Other invalid | 0 | 0 | 0 |

TRAIN was physically eligible 132/132. VALIDATION supplied 64/66 objective-valid records. Mild Sand was TRAIN 72/72 and VALIDATION 35/36 against minima 68 and 34. Moderate Sand was TRAIN 24/24 and VALIDATION 11/12 against minima 22 and 10. The single valid-but-ineligible Hazard was a moderate Sand Slip; the single invalid record was a real post-target fall/censor. Both were retained.

## Source-speed Sand matrix

`Valid` is objective-valid under the frozen Sand role contract. `Invalid` combines pretarget fall, target-following fall/censor, and other invalid.

| Split | Source | Speed | Planned | Valid | Strict | Slip | Dual | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FACTOR_TRAIN | concrete | 0.20 | 16 | 16 | 16 | 0 | 0 | 0 |
| FACTOR_TRAIN | concrete | 0.25 | 16 | 16 | 16 | 0 | 0 | 0 |
| FACTOR_TRAIN | concrete | 0.30 | 16 | 16 | 16 | 0 | 0 | 0 |
| FACTOR_TRAIN | marble | 0.20 | 16 | 16 | 16 | 0 | 0 | 0 |
| FACTOR_TRAIN | marble | 0.25 | 16 | 16 | 16 | 0 | 0 | 0 |
| FACTOR_TRAIN | marble | 0.30 | 16 | 16 | 16 | 0 | 0 | 0 |
| FACTOR_VALIDATION | concrete | 0.20 | 8 | 8 | 8 | 0 | 0 | 0 |
| FACTOR_VALIDATION | concrete | 0.25 | 8 | 7 | 7 | 0 | 0 | 1 |
| FACTOR_VALIDATION | concrete | 0.30 | 8 | 8 | 8 | 0 | 0 | 0 |
| FACTOR_VALIDATION | marble | 0.20 | 8 | 8 | 8 | 0 | 0 | 0 |
| FACTOR_VALIDATION | marble | 0.25 | 8 | 8 | 8 | 0 | 0 | 0 |
| FACTOR_VALIDATION | marble | 0.30 | 8 | 7 | 7 | 1 | 0 | 0 |

All 12 source-speed minima passed: every TRAIN cell was 16 against `>=14`, and every VALIDATION cell was at least 7 against `>=7`. Designed-Sand Slip-plus-Dual was 1 against the frozen maximum 6.

## Factor manifolds

The Concrete/.25 exception rows are a subset of the adverse rows and are not additive. Measured phase is physical ground truth; topology and phase are coupled, and no independent phase-manipulation claim is made.

| Split | Manifold | Planned | Valid | Strict | Slip/Dual | Invalid | Yield | Measured phase truth |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FACTOR_TRAIN | transition-left / adverse | 42 | 42 | 42 | 0 | 0 | 100.0% | RIGHT_SINGLE_SUPPORT 42 |
| FACTOR_TRAIN | transition-right / comparison | 30 | 30 | 30 | 0 | 0 | 100.0% | LEFT_SINGLE_SUPPORT 30 |
| FACTOR_TRAIN | Concrete/.25 adverse exception | 12 | 12 | 12 | 0 | 0 | 100.0% | RIGHT_SINGLE_SUPPORT 12 |
| FACTOR_VALIDATION | transition-left / adverse | 21 | 20 | 20 | 0 | 1 | 95.2% | RIGHT_SINGLE_SUPPORT 21 |
| FACTOR_VALIDATION | transition-right / comparison | 15 | 15 | 15 | 0 | 0 | 100.0% | LEFT_SINGLE_SUPPORT 15 |
| FACTOR_VALIDATION | Concrete/.25 adverse exception | 6 | 5 | 5 | 0 | 1 | 83.3% | RIGHT_SINGLE_SUPPORT 6 |

Both principal topology and phase gates passed in each split, all ten nonexception source-speed/split cells realized both manifolds, and both Concrete/.25 split exceptions realized the required adverse-only direction.

## Ordinary Support

| Split | Source | Speed | Planned | Support | Dual | Slip | Invalid | Yield |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FACTOR_TRAIN | concrete | 0.20 | 4 | 4 | 0 | 0 | 0 | 100% |
| FACTOR_TRAIN | concrete | 0.25 | 4 | 4 | 0 | 0 | 0 | 100% |
| FACTOR_TRAIN | concrete | 0.30 | 4 | 4 | 0 | 0 | 0 | 100% |
| FACTOR_TRAIN | marble | 0.20 | 4 | 4 | 0 | 0 | 0 | 100% |
| FACTOR_TRAIN | marble | 0.25 | 4 | 4 | 0 | 0 | 0 | 100% |
| FACTOR_TRAIN | marble | 0.30 | 4 | 4 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | concrete | 0.20 | 2 | 2 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | concrete | 0.25 | 2 | 2 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | concrete | 0.30 | 2 | 2 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | marble | 0.20 | 2 | 2 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | marble | 0.25 | 2 | 2 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | marble | 0.30 | 2 | 2 | 0 | 0 | 0 | 100% |

The critical gates passed at TRAIN 24/24 against `>=22` and VALIDATION 12/12 against `>=11`. Ordinary Support Slip-plus-Dual was 0 overall and in both splits. Concrete/.30 was physically RIGHT_ONLY with measured LEFT_SINGLE_SUPPORT. Other cells realized LEFT_ONLY/right-single and RIGHT_ONLY/left-single as designed.

| Split | Source | Speed | Actual side → measured phase | Later physical fall ms | Minimum post-Support observation ms |
|---|---|---:|---|---|---:|
| TRAIN | concrete | 0.20 | LEFT→right-single; RIGHT→left-single | 5469 | 2994 |
| TRAIN | concrete | 0.25 | LEFT→right-single; RIGHT→left-single | 5520 | 3045 |
| TRAIN | concrete | 0.30 | RIGHT→left-single | — | 6832 |
| TRAIN | marble | 0.20 | LEFT→right-single; RIGHT→left-single | 5779, 5845 | 3303 |
| TRAIN | marble | 0.25 | LEFT→right-single; RIGHT→left-single | 5172, 5172 | 2698 |
| TRAIN | marble | 0.30 | LEFT→right-single; RIGHT→left-single | — | 5964 |
| VALIDATION | concrete | 0.20 | LEFT→right-single; RIGHT→left-single | — | 6220 |
| VALIDATION | concrete | 0.25 | LEFT→right-single; RIGHT→left-single | 5551 | 3076 |
| VALIDATION | concrete | 0.30 | RIGHT→left-single | — | 6832 |
| VALIDATION | marble | 0.20 | LEFT→right-single; RIGHT→left-single | 5853 | 3377 |
| VALIDATION | marble | 0.25 | LEFT→right-single; RIGHT→left-single | 5172 | 2698 |
| VALIDATION | marble | 0.30 | LEFT→right-single; RIGHT→left-single | — | 5964 |

The listed later falls occurred only after at least 2,698 ms of valid post-Support observation, so the unchanged physical contract correctly retains these rows as Support. Entry was 1220–1810 ms, I1 1239–1829 ms, and established Support 2168–3036 ms; no ordinary row slipped or missed side/phase intent.

## Delayed Support

| Split | Source | Speed | Planned | Support | Dual | Slip | Invalid | Yield |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FACTOR_TRAIN | concrete | 0.20 | 2 | 2 | 0 | 0 | 0 | 100% |
| FACTOR_TRAIN | concrete | 0.25 | 2 | 2 | 0 | 0 | 0 | 100% |
| FACTOR_TRAIN | concrete | 0.30 | 2 | 2 | 0 | 0 | 0 | 100% |
| FACTOR_TRAIN | marble | 0.20 | 2 | 2 | 0 | 0 | 0 | 100% |
| FACTOR_TRAIN | marble | 0.25 | 2 | 2 | 0 | 0 | 0 | 100% |
| FACTOR_TRAIN | marble | 0.30 | 2 | 2 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | concrete | 0.20 | 1 | 1 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | concrete | 0.25 | 1 | 1 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | concrete | 0.30 | 1 | 1 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | marble | 0.20 | 1 | 1 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | marble | 0.25 | 1 | 1 | 0 | 0 | 0 | 100% |
| FACTOR_VALIDATION | marble | 0.30 | 1 | 1 | 0 | 0 | 0 | 100% |

The critical delayed gates passed at TRAIN 12/12 against `>=11` and VALIDATION 6/6 against `>=5`; Slip-plus-Dual was 0 overall and in both splits. All 18 were LEFT_ONLY, transition-left, measured RIGHT_SINGLE_SUPPORT, with two clean target touchdowns before I1 and at least 1,903 ms post-Support observation. Entry was 1220–1810 ms, I1 2421–3609 ms, and established Support 3054–3663 ms.

## Support failure ledger

There were zero ordinary- or delayed-Support misses. Consequently there is no run-level miss row for run ID, source/speed, side, topology, measured phase, geometry, entry, I1, Support, Slip, fall/censor, post-Support observation, or final outcome. All 54 planned controls produced qualified `SUPPORT` and remain in the corpus.

## Invalidity

| Run | Split/group | Source/speed | Geometry start/width/exit m | Topology / measured phase | Entry | Slip | Fall/censor | Post-target | Result |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `sfcocr_v_sml_c_025_03` | VALIDATION Mild | concrete/.25 | .350/.789/1.139 | transition-left / RIGHT_SINGLE_SUPPORT | 1220 | — | 2156 | 0 | target-following physical fall/censor; invalid |
| `sfcocr_v_smd_m_030_01` | VALIDATION Moderate | marble/.30 | .350/.793/1.143 | transition-left / RIGHT_SINGLE_SUPPORT | 1227 | 4687 | 9000 censor | 4279 | genuine Slip |

The invalid Mild row physically fell at 2156 ms, so its `insufficient_post_target_observation` label is not a nominal nine-second horizon defect. The Moderate row remained fully observed, established Slip, and was retained as physical Hazard evidence. Neither result triggered a repair or rerun.

## Physical signatures and anti-contamination

| Metric | Result |
|---|---:|
| Scenario unique / total | 198 / 198 |
| Physical unique / total | 174 / 198 |
| Valid physical unique / valid total | 172 / 196 |
| Valid physical uniqueness fraction | 0.8775510204 |
| Frozen minimum | 0.80 (PASS) |
| Physical duplicate-row excess | 24 |
| Exact physical duplicate-pair combinations | 45 |
| Historical exact / forbidden-near overlap | 0 / 0 |
| Failed 162 exact / forbidden-near overlap | 0 / 0 |
| Failed recalibrated 198 exact / forbidden-near overlap | 0 / 0 |
| Failed support-recalibrated 198 exact / forbidden-near overlap | 0 / 0 |
| All calibration/pilot exact / forbidden-near overlap | 0 / 0 |
| Cross-split exact / forbidden-near overlap | 0 / 0 |
| Historical run-ID reuse | 0 |

Exact physical duplicates are permitted by the frozen protocol because distinct planned coordinates can realize the same measured physical signature. Only the predeclared valid-uniqueness fraction is gated. No physical-near criterion was defined, so none was invented after observing the data.

## Complete frozen gate ledger

| Gate | Threshold | Observed | Result | Source artifact |
|---|---:|---:|---|---|
| censor/pretarget_fall | <=3 | 0 | PASS | manifest.json |
| censor/target_following_fall | <=8 | 1 | PASS | manifest.json |
| contamination/FACTOR_TRAIN/delayed_support_slip_plus_dual | <=1 | 0 | PASS | manifest.json |
| contamination/FACTOR_TRAIN/ordinary_support_slip_plus_dual | <=1 | 0 | PASS | manifest.json |
| contamination/FACTOR_VALIDATION/delayed_support_slip_plus_dual | <=1 | 0 | PASS | manifest.json |
| contamination/FACTOR_VALIDATION/ordinary_support_slip_plus_dual | <=1 | 0 | PASS | manifest.json |
| contamination/delayed_support_slip_plus_dual | <=2 | 0 | PASS | manifest.json |
| contamination/designed_sand_slip_plus_dual | <=6 | 1 | PASS | manifest.json |
| contamination/ordinary_support_slip_plus_dual | <=2 | 0 | PASS | manifest.json |
| diversity/valid_physical_signature_uniqueness_fraction | >=0.8 | 0.8775510204 | PASS | manifest.json |
| execution/adaptive_backfill | 0 | 0 | PASS | manifest.json |
| execution/attempted | 198 | 198 | PASS | manifest.json |
| execution/completed | 198 | 198 | PASS | manifest.json |
| execution/planned | 198 | 198 | PASS | pre_simulation_freeze.json |
| execution/replacement | 0 | 0 | PASS | manifest.json |
| execution/rerun | 0 | 0 | PASS | manifest.json |
| integrity/cross_split_exact_overlap | 0 | 0 | PASS | pre_simulation_freeze.json |
| integrity/cross_split_forbidden_near_overlap | 0 | 0 | PASS | pre_simulation_freeze.json |
| integrity/failed_198_exact_overlap | 0 | 0 | PASS | pre_simulation_freeze.json |
| integrity/failed_198_forbidden_near_overlap | 0 | 0 | PASS | pre_simulation_freeze.json |
| integrity/historical_exact_overlap | 0 | 0 | PASS | pre_simulation_freeze.json |
| integrity/historical_forbidden_near_overlap | 0 | 0 | PASS | pre_simulation_freeze.json |
| integrity/historical_run_id_reuse | 0 | 0 | PASS | pre_simulation_freeze.json |
| integrity/model_outputs | 0 | 0 | PASS | manifest.json |
| integrity/pilot_exact_overlap | 0 | 0 | PASS | pre_simulation_freeze.json |
| integrity/pilot_forbidden_near_overlap | 0 | 0 | PASS | pre_simulation_freeze.json |
| integrity/planned_run_ids_unique | 198 | 198 | PASS | pre_simulation_freeze.json |
| integrity/planned_scenario_signatures_unique | 198 | 198 | PASS | pre_simulation_freeze.json |
| topology_phase/FACTOR_TRAIN/principal_precontact_phases | 2 | 2 | PASS | manifest.json |
| topology_phase/FACTOR_TRAIN/principal_topologies | 2 | 2 | PASS | manifest.json |
| topology_phase/FACTOR_VALIDATION/principal_precontact_phases | 2 | 2 | PASS | manifest.json |
| topology_phase/FACTOR_VALIDATION/principal_topologies | 2 | 2 | PASS | manifest.json |
| topology_phase/all_nonexception_cells_both_manifolds | 10 | 10 | PASS | manifest.json |
| topology_phase/concrete_025_left_right_single_exception | 2 | 2 | PASS | manifest.json |
| yield/FACTOR_TRAIN/concrete/0.20/strict_sand | >=14 | 16 | PASS | manifest.json |
| yield/FACTOR_TRAIN/concrete/0.25/strict_sand | >=14 | 16 | PASS | manifest.json |
| yield/FACTOR_TRAIN/concrete/0.30/strict_sand | >=14 | 16 | PASS | manifest.json |
| yield/FACTOR_TRAIN/delayed_support | >=11 | 12 | PASS | manifest.json |
| yield/FACTOR_TRAIN/marble/0.20/strict_sand | >=14 | 16 | PASS | manifest.json |
| yield/FACTOR_TRAIN/marble/0.25/strict_sand | >=14 | 16 | PASS | manifest.json |
| yield/FACTOR_TRAIN/marble/0.30/strict_sand | >=14 | 16 | PASS | manifest.json |
| yield/FACTOR_TRAIN/mild | >=68 | 72 | PASS | manifest.json |
| yield/FACTOR_TRAIN/mild_adverse_direction | >=39 | 42 | PASS | manifest.json |
| yield/FACTOR_TRAIN/mild_comparison_direction | >=28 | 30 | PASS | manifest.json |
| yield/FACTOR_TRAIN/moderate | >=22 | 24 | PASS | manifest.json |
| yield/FACTOR_TRAIN/ordinary_support | >=22 | 24 | PASS | manifest.json |
| yield/FACTOR_TRAIN/strict_sand | >=91 | 96 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/concrete/0.20/strict_sand | >=7 | 8 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/concrete/0.25/strict_sand | >=7 | 7 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/concrete/0.30/strict_sand | >=7 | 8 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/delayed_support | >=5 | 6 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/marble/0.20/strict_sand | >=7 | 8 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/marble/0.25/strict_sand | >=7 | 8 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/marble/0.30/strict_sand | >=7 | 7 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/mild | >=34 | 35 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/mild_adverse_direction | >=19 | 20 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/mild_comparison_direction | >=14 | 15 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/moderate | >=10 | 11 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/ordinary_support | >=11 | 12 | PASS | manifest.json |
| yield/FACTOR_VALIDATION/strict_sand | >=44 | 46 | PASS | manifest.json |
| yield/objective_valid | >=180 | 196 | PASS | manifest.json |

All 61 frozen gates passed; failed gates: none.

## Dataset freeze and FACTOR_VALIDATION status

| Frozen value | SHA-256 |
|---|---|
| Execution config file | `a9380ddb3f5649183878af7ac6b865bf6e372cbb4954d367072a1ba1366f0377` |
| Pre-simulation freeze file | `ca52fee47b300bf85bb20ccdfd9efb83dc76c1e7344040423dae424da19b65d4` |
| Manifest file | `70f850f22507384a50c81bdd065c7d485f2e37cd7e181fda20431ed2fede2d50` |
| FACTOR_TRAIN split | `039725f3231b2f48daae3f9e0d5f768613fdb71c3b6c15050e4f62308cef45c2` |
| FACTOR_VALIDATION split | `8e4a915347781d1654a9a05ec0e78754a7b7515284abfb6433197978ecce72e0` |
| Scenario-signature ledger | `8f279599fc1379138e38784a1e609275249741ac38d6b2406f58a8c307a5dacc` |
| Physical signatures | `c272793465d5c4a70a4bb97854bd0db5abbd9ff7f25348ca91a11ed3dda3bfeb` |
| Implementation aggregate | `bfed29c425b873bf09e606d0f676df0efe26ca3c16f518d332764aa27ca7229e` |
| NPZ aggregate | `464348e45b20d8c5adac7965e47233b0cae020adc66e3515209e84ddc9da6e21` |
| Physical outcomes | `cbfcab7b5613c6be5d35523a1b2b5b9def8eead9c7d1dc1277c5828c7499d247` |
| Generation-gate results | `02812b58a5d270d64c6cb677a090ffccfb4752d0e478f997ec8e89fc262d0fdc` |
| Physical-audit file | `7081ae25ed20659992392913f50b5001ca7eafa510f92cafb0edda7038f633be` |
| FACTOR_VALIDATION model seal | `761aec9eb466efa2b067cd5f10d13a76482cd66561c316801321bb20870a31b3` |
| Semantic dataset freeze | `e397c78d19386732eb54ba388c551a8fe213a6097ba7d0819ff20f7b9b0255f4` |
| Dataset-freeze file | `c9fee8eed1e75c23ce44c9d9fa4a9d204b150ed69e20b467d0c188e22e8194df` |
| Generation-summary file | `dbbade0cf564f01b49c64037e7735f2b9dc24b00b473159dca26f9898c403ce1` |

FACTOR_VALIDATION is generated but `SEALED_FOR_FUTURE_FACTOR_VALIDATION`: model inference, training, HNM, normalized 80D/model analysis, and visualization are prohibited until a future factor-conditioned candidate is completely trained and frozen. Current counters for all of those uses are zero.

## Model boundary, scientific status, and counters

| Counter | Actual |
|---|---:|
| New full-study simulations | 198 |
| New pilot simulations | 0 |
| Replacement / backfill / adaptive rerun | 0 / 0 / 0 |
| V1 / V2 / Terrain inference | 0 / 0 / 0 |
| Hazard probability / 80D analysis | 0 / 0 |
| Training / optimizer steps / checkpoint writes | 0 / 0 / 0 |
| Normalizer fits / HNM rounds | 0 / 0 |
| Threshold / persistence / architecture / seed searches | 0 / 0 / 0 / 0 |
| Sensor-fusion experiments | 0 |
| Historical HOLDOUT reads / inference | 0 / 0 |
| FACTOR_VALIDATION model inference / training use | 0 / 0 |

Historical scientific status is unchanged: `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`, `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`, and `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`. `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS` remains `NOT_YET_MODEL_TESTED`; the present result establishes a physically valid development corpus, not model-level support.

## Generation verdict and next milestone

`SAND_FACTOR_CONDITIONED_DEVELOPMENT_CONTROLS_RECALIBRATED_GENERATION_READY`

The exact recommended next milestone is `SAND_FACTOR_CONDITIONED_MODEL_TRAINING`. It was not started. Deployment engineering may continue independently with `model_v2_anchor_refined_gru20_20260902` only as `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`; no deployment repository was modified here.
