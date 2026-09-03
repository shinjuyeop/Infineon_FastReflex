# Factor-Conditioned Recalibrated Development Corpus Generation

## Decision

The exact frozen 198-run matrix was generated once and frozen as failed physical-development evidence. The complete predeclared ledger passed 53 of 55 gates. Only delayed Support yield failed: `FACTOR_TRAIN` produced 9 against a minimum of 10, and `FACTOR_VALIDATION` produced 3 against a minimum of 5. No run was replaced, backfilled, rerun, or relabeled.

Canonical verdict:

`SAND_FACTOR_CONDITIONED_DEVELOPMENT_RECALIBRATED_GENERATION_INSUFFICIENT`

Training and all model-side analysis remain prohibited. This result does not test `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS`.

## Starting state and frozen inputs

- Starting `HEAD` and `origin/main`: `dd4c909584741606c965241061a98796ca5fc5db`
- Expected scientific base was the starting `HEAD`; ancestor check passed.
- Tracked worktree was clean and no later parallel commit was discarded.
- Redesign config SHA-256: `dcb1417eb1771b7e02652ab4979024fa145dbe156ff333416e485cd29679e449`
- Scenario matrix SHA-256: `6ffad518466d3082a742787199c038732dea885c7ef508b2585a1dc267e39fc3`
- Scenario-signature ledger SHA-256: `0085a9568c3b30870739792a4cf552699e2dcf4ef45f4f00c3dd4780945e86bf`
- Redesign readiness SHA-256: `49f6897b8282648b94a4989ece90bc1c72b775e7c25ccd2b1c003c065c2e6ea5`
- Execution config SHA-256: `bd2f75f6882ae9e5573d9854ed67628f9c9750d35b24419363fe1c88f345e889`
- Source commit: `dd4c909584741606c965241061a98796ca5fc5db`

All protected redesign, calibration, failed-intervention, reference-model, normalizer, and consumed-HOLDOUT artifacts matched their frozen hashes before run 1. The pre-simulation freeze was written before physical execution.

## Generation

- Dataset: `data/raw/sand_factor_conditioned_development_recalibrated_20260903`
- Planned / attempted / completed: 198 / 198 / 198
- `FACTOR_TRAIN` / `FACTOR_VALIDATION`: 132 / 66
- Runtime: 1,363.184 seconds
- Files: 206 total, including 198 NPZ payloads
- NPZ bytes: 94,565,344
- New pilots / replacements / backfills / adaptive reruns: 0 / 0 / 0 / 0

## Overall physical outcomes

| Outcome | TRAIN | VALIDATION | Total |
|---|---:|---:|---:|
| Planned | 132 | 66 | 198 |
| Completed | 132 | 66 | 198 |
| Objective valid | 127 | 62 | 189 |
| Strict Sand | 96 | 47 | 143 |
| Mild strict Sand | 72 | 36 | 108 |
| Moderate strict Sand | 24 | 11 | 35 |
| Ordinary Support | 22 | 12 | 34 |
| Delayed Support | 9 | 3 | 12 |
| Slip | 0 | 1 | 1 |
| Dual Hazard | 3 | 2 | 5 |
| Pretarget fall | 0 | 0 | 0 |
| Post-target fall/censor | 0 | 0 | 0 |
| Other invalid | 2 | 1 | 3 |

`FACTOR_TRAIN` was 127/132 objective-valid, with 96 strict Sand, 22 ordinary Support, and 9 delayed Support. `FACTOR_VALIDATION` was 62/66 objective-valid, with 47 strict Sand, 12 ordinary Support, and 3 delayed Support. Physical outcome always overrode scenario intent.

## Mild and moderate Sand

| Split | Severity | Planned | Strict | Slip | Dual | Pretarget | Post-target | Other invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FACTOR_TRAIN | Mild | 72 | 72 | 0 | 0 | 0 | 0 | 0 |
| FACTOR_TRAIN | Moderate | 24 | 24 | 0 | 0 | 0 | 0 | 0 |
| FACTOR_VALIDATION | Mild | 36 | 36 | 0 | 0 | 0 | 0 | 0 |
| FACTOR_VALIDATION | Moderate | 12 | 11 | 1 | 0 | 0 | 0 | 0 |

The recalibrated mild domain yielded 108/108 strict Sand. Reduced moderate coverage yielded 35/36 strict Sand; the only exception was `sfcr_v_smd_m_030_02`, which physically became Slip. This is strong evidence that the recalibrated Sand geometry succeeded physically, but it cannot override the failed Support gates.

## Designed-Sand source-speed matrix

| Split | Source | Speed | Planned | Valid | Strict | Slip | Dual | Pretarget | Post-target |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FACTOR_TRAIN | Concrete | 0.20 | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| FACTOR_TRAIN | Concrete | 0.25 | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| FACTOR_TRAIN | Concrete | 0.30 | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| FACTOR_TRAIN | Marble | 0.20 | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| FACTOR_TRAIN | Marble | 0.25 | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| FACTOR_TRAIN | Marble | 0.30 | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| FACTOR_VALIDATION | Concrete | 0.20 | 8 | 8 | 8 | 0 | 0 | 0 | 0 |
| FACTOR_VALIDATION | Concrete | 0.25 | 8 | 8 | 8 | 0 | 0 | 0 | 0 |
| FACTOR_VALIDATION | Concrete | 0.30 | 8 | 8 | 8 | 0 | 0 | 0 | 0 |
| FACTOR_VALIDATION | Marble | 0.20 | 8 | 8 | 8 | 0 | 0 | 0 | 0 |
| FACTOR_VALIDATION | Marble | 0.25 | 8 | 8 | 8 | 0 | 0 | 0 | 0 |
| FACTOR_VALIDATION | Marble | 0.30 | 8 | 7 | 7 | 1 | 0 | 0 | 0 |

Every source-speed strict-yield gate passed.

## Factor manifolds and measured phase

Topology and measured phase remain coupled physical factors; this table does not claim independent phase manipulation.

| Split | Manifold | Planned | Valid | Strict | Slip/Dual | Invalid | Yield | Measured phase |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FACTOR_TRAIN | transition-left / adverse | 42 | 42 | 42 | 0 | 0 | 100% | RIGHT_SINGLE_SUPPORT 42 |
| FACTOR_TRAIN | transition-right / comparison | 30 | 30 | 30 | 0 | 0 | 100% | LEFT_SINGLE_SUPPORT 30 |
| FACTOR_TRAIN | Concrete/.25 left-only exception | 12 | 12 | 12 | 0 | 0 | 100% | RIGHT_SINGLE_SUPPORT 12 |
| FACTOR_VALIDATION | transition-left / adverse | 21 | 21 | 21 | 0 | 0 | 100% | RIGHT_SINGLE_SUPPORT 21 |
| FACTOR_VALIDATION | transition-right / comparison | 15 | 15 | 15 | 0 | 0 | 100% | LEFT_SINGLE_SUPPORT 15 |
| FACTOR_VALIDATION | Concrete/.25 left-only exception | 6 | 6 | 6 | 0 | 0 | 100% | RIGHT_SINGLE_SUPPORT 6 |

Both manifolds were physically realized in all 10 non-exception split/source-speed cells, and the two split-specific Concrete/.25 exception checks passed.

## Support controls

| Split | Kind | Source | Speed | Planned | Valid | Ordinary | Delayed | Side / actual physical outcome |
|---|---|---|---:|---:|---:|---:|---:|---|
| FACTOR_TRAIN | Delayed | Concrete | 0.20 | 2 | 0 | 0 | 0 | LEFT/DUAL_HAZARD 2 |
| FACTOR_TRAIN | Delayed | Concrete | 0.25 | 2 | 2 | 0 | 2 | LEFT/SUPPORT 2 |
| FACTOR_TRAIN | Delayed | Concrete | 0.30 | 2 | 2 | 0 | 2 | LEFT/SUPPORT 2 |
| FACTOR_TRAIN | Delayed | Marble | 0.20 | 2 | 1 | 0 | 1 | LEFT/DUAL_HAZARD 1; LEFT/SUPPORT 1 |
| FACTOR_TRAIN | Delayed | Marble | 0.25 | 2 | 2 | 0 | 2 | LEFT/SUPPORT 2 |
| FACTOR_TRAIN | Delayed | Marble | 0.30 | 2 | 2 | 0 | 2 | LEFT/SUPPORT 2 |
| FACTOR_TRAIN | Ordinary | Concrete | 0.20 | 4 | 4 | 4 | 0 | LEFT/SUPPORT 2; RIGHT/SUPPORT 2 |
| FACTOR_TRAIN | Ordinary | Concrete | 0.25 | 4 | 4 | 4 | 0 | LEFT/SUPPORT 2; RIGHT/SUPPORT 2 |
| FACTOR_TRAIN | Ordinary | Concrete | 0.30 | 4 | 2 | 2 | 0 | LEFT/INVALID 2; RIGHT/SUPPORT 2 |
| FACTOR_TRAIN | Ordinary | Marble | 0.20 | 4 | 4 | 4 | 0 | LEFT/SUPPORT 2; RIGHT/SUPPORT 2 |
| FACTOR_TRAIN | Ordinary | Marble | 0.25 | 4 | 4 | 4 | 0 | LEFT/SUPPORT 2; RIGHT/SUPPORT 2 |
| FACTOR_TRAIN | Ordinary | Marble | 0.30 | 4 | 4 | 4 | 0 | LEFT/SUPPORT 2; RIGHT/SUPPORT 2 |
| FACTOR_VALIDATION | Delayed | Concrete | 0.20 | 1 | 0 | 0 | 0 | LEFT/INVALID 1 |
| FACTOR_VALIDATION | Delayed | Concrete | 0.25 | 1 | 1 | 0 | 1 | LEFT/SUPPORT 1 |
| FACTOR_VALIDATION | Delayed | Concrete | 0.30 | 1 | 0 | 0 | 0 | LEFT/DUAL_HAZARD 1 |
| FACTOR_VALIDATION | Delayed | Marble | 0.20 | 1 | 0 | 0 | 0 | LEFT/DUAL_HAZARD 1 |
| FACTOR_VALIDATION | Delayed | Marble | 0.25 | 1 | 1 | 0 | 1 | LEFT/SUPPORT 1 |
| FACTOR_VALIDATION | Delayed | Marble | 0.30 | 1 | 1 | 0 | 1 | LEFT/SUPPORT 1 |
| FACTOR_VALIDATION | Ordinary | Concrete | 0.20 | 2 | 2 | 2 | 0 | LEFT/SUPPORT 1; RIGHT/SUPPORT 1 |
| FACTOR_VALIDATION | Ordinary | Concrete | 0.25 | 2 | 2 | 2 | 0 | LEFT/SUPPORT 1; RIGHT/SUPPORT 1 |
| FACTOR_VALIDATION | Ordinary | Concrete | 0.30 | 2 | 2 | 2 | 0 | LEFT/SUPPORT 1; RIGHT/SUPPORT 1 |
| FACTOR_VALIDATION | Ordinary | Marble | 0.20 | 2 | 2 | 2 | 0 | LEFT/SUPPORT 1; RIGHT/SUPPORT 1 |
| FACTOR_VALIDATION | Ordinary | Marble | 0.25 | 2 | 2 | 2 | 0 | LEFT/SUPPORT 1; RIGHT/SUPPORT 1 |
| FACTOR_VALIDATION | Ordinary | Marble | 0.30 | 2 | 2 | 2 | 0 | LEFT/SUPPORT 1; RIGHT/SUPPORT 1 |

Ordinary Support passed at TRAIN 22/24 and VALIDATION 12/12. Delayed Support failed at TRAIN 9/12 and VALIDATION 3/6. The five dual-Hazard outcomes are concentrated in delayed controls: TRAIN Concrete/.20 (2), TRAIN Marble/.20 (1), VALIDATION Concrete/.30 (1), and VALIDATION Marble/.20 (1). The remaining VALIDATION delayed miss was an actual insufficient-follow-up invalid at Concrete/.20. Support onset, I1, delayed semantics, side logic, and extraction were not changed.

## Invalidity and physical exceptions

Pretarget fall and target-following fall/censor counts were both zero. The three `OTHER_INVALID` records were retained:

| Run | Split | Source / speed | Intent | Topology / phase | Start / width / exit (m) | Target (ms) | Fall/censor (ms) | Post-target (ms) | Reason |
|---|---|---|---|---|---|---:|---:|---:|---|
| `sfcr_t_osp_c_030_01` | TRAIN | Concrete / .30 | Moderate ordinary Support | transition-left / RIGHT_SINGLE_SUPPORT | .337 / .719 / 1.056 | 1227 | 3599 | 0 | insufficient post-Support observation |
| `sfcr_t_osp_c_030_02` | TRAIN | Concrete / .30 | Moderate ordinary Support | transition-left / RIGHT_SINGLE_SUPPORT | .341 / .727 / 1.068 | 1227 | 3610 | 0 | insufficient post-Support observation |
| `sfcr_v_dsp_c_020_01` | VALIDATION | Concrete / .20 | Moderate delayed Support | transition-left / RIGHT_SINGLE_SUPPORT | .354 / .805 / 1.159 | 2454 | 5118 | 1 | insufficient post-Support observation |

The single designed-Sand Slip was `sfcr_v_smd_m_030_02` (VALIDATION, Marble/.30 moderate, transition-left/right-single, start .326 m, width .799 m, exit 1.125 m). No model score was produced or consulted.

## Physical-signature and anti-contamination audit

- Planned scenario signatures: 198 unique / 198 total
- Physical signatures: 175 unique / 198 total
- Valid physical signatures: 166 unique / 189 valid, uniqueness fraction `0.8783068783068783`
- Exact physical duplicates: 23
- Physical near-pair count: not defined by the frozen protocol; no post-hoc criterion was added
- Historical exact / forbidden-near / run-ID reuse: 0 / 0 / 0
- Pilot exact / forbidden-near reuse: 0 / 0
- Cross-split exact / forbidden-near overlap: 0 / 0

## Complete generation-gate ledger

| Gate | Frozen threshold | Observed | Result | Supporting artifact |
|---|---:|---:|---|---|
| `censor/pretarget_fall` | <=4 | 0 | PASS | `manifest.json` |
| `censor/target_following_fall` | <=14 | 0 | PASS | `manifest.json` |
| `contamination/FACTOR_TRAIN/designed_sand_slip_plus_dual` | <=5 | 0 | PASS | `manifest.json` |
| `contamination/FACTOR_VALIDATION/designed_sand_slip_plus_dual` | <=3 | 1 | PASS | `manifest.json` |
| `contamination/designed_sand_slip_plus_dual` | <=8 | 1 | PASS | `manifest.json` |
| `diversity/valid_physical_signature_uniqueness_fraction` | >=0.8 | 0.8783068783068783 | PASS | `manifest.json` |
| `execution/adaptive_backfill` | 0 | 0 | PASS | `manifest.json` |
| `execution/attempted` | 198 | 198 | PASS | `manifest.json` |
| `execution/completed` | 198 | 198 | PASS | `manifest.json` |
| `execution/planned` | 198 | 198 | PASS | `pre_simulation_freeze.json` |
| `execution/replacement` | 0 | 0 | PASS | `manifest.json` |
| `execution/rerun` | 0 | 0 | PASS | `manifest.json` |
| `integrity/cross_split_exact_overlap` | 0 | 0 | PASS | `pre_simulation_freeze.json` |
| `integrity/cross_split_forbidden_near_overlap` | 0 | 0 | PASS | `pre_simulation_freeze.json` |
| `integrity/historical_exact_overlap` | 0 | 0 | PASS | `pre_simulation_freeze.json` |
| `integrity/historical_forbidden_near_overlap` | 0 | 0 | PASS | `pre_simulation_freeze.json` |
| `integrity/historical_run_id_reuse` | 0 | 0 | PASS | `pre_simulation_freeze.json` |
| `integrity/model_outputs` | 0 | 0 | PASS | `manifest.json` |
| `integrity/pilot_exact_overlap` | 0 | 0 | PASS | `pre_simulation_freeze.json` |
| `integrity/pilot_forbidden_near_overlap` | 0 | 0 | PASS | `pre_simulation_freeze.json` |
| `integrity/planned_run_ids_unique` | 198 | 198 | PASS | `pre_simulation_freeze.json` |
| `integrity/planned_scenario_signatures_unique` | 198 | 198 | PASS | `pre_simulation_freeze.json` |
| `topology_phase/FACTOR_TRAIN/principal_precontact_phases` | 2 | 2 | PASS | `manifest.json` |
| `topology_phase/FACTOR_TRAIN/principal_topologies` | 2 | 2 | PASS | `manifest.json` |
| `topology_phase/FACTOR_VALIDATION/principal_precontact_phases` | 2 | 2 | PASS | `manifest.json` |
| `topology_phase/FACTOR_VALIDATION/principal_topologies` | 2 | 2 | PASS | `manifest.json` |
| `topology_phase/all_nonexception_cells_both_manifolds` | 10 | 10 | PASS | `manifest.json` |
| `topology_phase/concrete_025_left_right_single_exception` | 2 | 2 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/concrete/0.20/strict_sand` | >=12 | 16 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/concrete/0.25/strict_sand` | >=12 | 16 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/concrete/0.30/strict_sand` | >=12 | 16 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/delayed_support` | >=10 | 9 | **FAIL** | `manifest.json` |
| `yield/FACTOR_TRAIN/marble/0.20/strict_sand` | >=12 | 16 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/marble/0.25/strict_sand` | >=12 | 16 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/marble/0.30/strict_sand` | >=12 | 16 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/mild` | >=62 | 72 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/mild_adverse_direction` | >=36 | 42 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/mild_comparison_direction` | >=25 | 30 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/moderate` | >=14 | 24 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/ordinary_support` | >=22 | 22 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/strict_sand` | >=76 | 96 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/concrete/0.20/strict_sand` | >=6 | 8 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/concrete/0.25/strict_sand` | >=6 | 8 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/concrete/0.30/strict_sand` | >=6 | 8 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/delayed_support` | >=5 | 3 | **FAIL** | `manifest.json` |
| `yield/FACTOR_VALIDATION/marble/0.20/strict_sand` | >=6 | 8 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/marble/0.25/strict_sand` | >=6 | 8 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/marble/0.30/strict_sand` | >=6 | 7 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/mild` | >=31 | 36 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/mild_adverse_direction` | >=18 | 21 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/mild_comparison_direction` | >=12 | 15 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/moderate` | >=7 | 11 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/ordinary_support` | >=11 | 12 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/strict_sand` | >=36 | 47 | PASS | `manifest.json` |
| `yield/objective_valid` | >=165 | 189 | PASS | `manifest.json` |

Ledger total: 55; PASS: 53; FAIL: 2.

## Frozen artifact hashes

| Artifact | SHA-256 |
|---|---|
| Pre-simulation freeze | `1b6524ac865afe6c47c96b54f813a465388d66debe13bab0bb541c70d839cb83` |
| Manifest | `776c2a22c8963d5ddcdad49d2f36c109a21af4dfa1c0b5d2a924112b1fb6b19c` |
| FACTOR_TRAIN split | `59fa9edc13c2fbba90c2d54d0b07c2c04e6e716fed88e6646b2bc6d82baf18d6` |
| FACTOR_VALIDATION split | `35fa774081c3153e198a3d81f1ab7e931a7be75b626aac681afff5a6a16d1495` |
| Scenario matrix | `6ffad518466d3082a742787199c038732dea885c7ef508b2585a1dc267e39fc3` |
| Scenario signatures | `0085a9568c3b30870739792a4cf552699e2dcf4ef45f4f00c3dd4780945e86bf` |
| Physical signatures | `a20308a95cb3608e0fdb8ac1ff3d33fdf377b94aabd0816d453353b09e002d09` |
| NPZ aggregate | `c64c934ce51fab1669b1c0badb5b645d2eb15a7ac10f72d43b7dc17295f6b6fa` |
| Physical outcomes | `2f4931e815cc90242ddf1b37546c21ddc246fdf00df0175c1e0245d36cc1ab3d` |
| Generation-gate result | `98839a793ca5f55b61ccfe72312405fb5b0c15b93d4a6428cfc431975f2bbcb1` |
| Physical audit | `64bcf9311946e96da2ed39714a0a1ffb3d6aa765548542cdd2edf3bf413229f8` |
| Implementation bundle | `6933d281ce687c742fd92e45cf71e0b8d050dd770c54cee8a800b98732c4314d` |
| FACTOR_VALIDATION seal | `debeae9d0c4f8f3f0e4e93e20aa82c9dd95eeaaee43fc066261e9b563197cf4d` |
| Semantic dataset freeze | `d7a7b06095ce80e0bfdc5766e9a8265178e8ef184e0ec3251d35eab555588f84` |
| Dataset-freeze file | `7b8ade8aff7dbb9321e1ac7a4474a892b7f58b15ed03f3093b819f3bae551cce` |
| Generation summary | `9db31072b56cd15d4486f4ba6d94bf755d192a5b7add55c7920b933c38ddaffa` |

## Protection and counters

`FACTOR_VALIDATION` is generated but sealed as `SEALED_FAILED_PHYSICAL_EVIDENCE`. Model inference, training use, and HNM are all zero. The loader rejects its split identity before calling `numpy.load`.

V1 inference, V2 inference, Terrain inference, Hazard probability calculation, 80D model analysis, optimizer steps, checkpoint writes, normalizer fits, HNM rounds, threshold searches, persistence searches, architecture searches, sensor-fusion experiments, old HOLDOUT payload reads/inference, and FACTOR_VALIDATION model access are all zero. The historical HOLDOUT guard remains `1`, with one scientific open.

Historical status is unchanged:

- `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`
- `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS`: `NOT YET MODEL-TESTED`

## Interpretation and next milestone

The Sand redesign itself produced 143/144 strict Sand across mild and moderate intent, including 108/108 mild, and every Sand yield/manifold/source-speed gate passed. The insufficient verdict is localized to the unchanged delayed-Support control population, not to the factor-conditioned Sand manifolds. Because the frozen rule is conjunctive, model training cannot begin.

The smallest follow-up is `SAND_FACTOR_CONDITIONED_DELAYED_SUPPORT_PHYSICAL_REVIEW`: a model-blind, saved-evidence-first review of the six delayed-control misses and the frozen delayed-Support geometry/contact sequence. It must not backfill or mutate this corpus, reopen historical HOLDOUT, or perform model science. The E84 deployment repository remains untouched; independent engineering may continue using only the frozen historical reference model under its existing non-final label.
