# Support-Recalibrated Factor-Conditioned Development Generation

## Starting state

- Repository: `/d/shin/Infineon_FastReflex`
- Branch: `main`
- Starting `HEAD`: `5346eca3aeae1b09b6e23ffffe92c61a15c363c9`
- Starting `origin/main`: `5346eca3aeae1b09b6e23ffffe92c61a15c363c9`
- Tracked worktree: clean
- Frozen delayed-Support review verdict: `DELAYED_SUPPORT_PHYSICAL_RECALIBRATION_READY`
- Frozen review/config/pilot/future-design hashes were verified before run 1.

## Scientific boundary

This milestone performed physical simulation, censor-aware physical labeling, frozen generation-gate evaluation, and dataset freezing only. It performed no training, V1/V2/Terrain inference, Hazard-probability calculation, 80D analysis, HNM, normalizer fitting, tuning, or old-HOLDOUT payload access. Scenario intent never overrode physical truth. There was no replacement, backfill, adaptive rerun, or post-start protocol mutation.

The historical scientific status remains unchanged:

- `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`
- `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS: NOT_YET_MODEL-TESTED`

## Frozen support-recalibrated design

- Design config: `configs/experiment/20260903_sand_factor_conditioned_development_support_recalibrated.yaml`
- Design config SHA-256: `b1f09effbb90313ef8c883db8e0ef376c5f4eb8e83ec66d8ed5fe52ca95ef775`
- Execution config: `configs/experiment/20260903_sand_factor_conditioned_development_support_recalibrated_generation.yaml`
- Execution config SHA-256: `c4bde7adc1d4a917fe78da9b2f470c1f6155af63a06f314972609c64ef5aade4`
- Dataset ID: `sand_factor_conditioned_development_support_recalibrated_20260903`
- Matrix: 198/198 unique run IDs and 198/198 unique scenario signatures
- Scenario matrix SHA-256: `6ed2c0c23ae036fd0bc8f523b3f254429cb1c835507d9859dadcbf82af6bd8b2`
- Scenario-signature SHA-256: `0944705e3cb18ff78f4edf68573fbf56477ae9fc7cf7576a2894145549feb4be`
- Split identities: TRAIN `5283046afaffbe4dd41947bc7db792e30d9c71dec13322d30bd9e0801c035a31`; VALIDATION `f81a9acb0b1947d4da0a5c09211c7a96050cdc86d0b5873bf885b8209fe8c901`

The successful Sand and ordinary-Support physical-domain concepts were preserved exactly as frozen in the future design. Their execution coordinates are deliberately fresh rather than literal copies of the failed corpus, satisfying the frozen anti-reuse contract. The only physical-family recalibration was delayed Support.

The delayed-Support envelope was LEFT-only, `transition_left`, expected measured `RIGHT_SINGLE_SUPPORT`, `staged_lateral_deformable`, moderate severity, start 0.324–0.332 m, width 0.825–0.833 m, exit 1.153–1.165 m, 9000 ms horizon, and at least 1000 ms post-Support observation. Canonical Support (10 mm/20 samples), Slip (50 mm/3 samples), I1, side, and two-clean-touchdown semantics were unchanged.

## Generation

| Item | Result |
|---|---:|
| Planned | 198 |
| Attempted | 198 |
| Completed | 198 |
| FACTOR_TRAIN | 132 |
| FACTOR_VALIDATION | 66 |
| Runtime | 1317.767 s |
| NPZ files | 198 |
| Total files | 206 |
| NPZ bytes | 94,465,341 |
| Dataset-directory bytes | 96,844,590 |
| Replacement | 0 |
| Backfill | 0 |
| Rerun | 0 |

Environment provenance was frozen before run 1: Python 3.10.12, Linux 6.8.0-136-generic x86_64, Asia/Seoul. All 198 runs were executed once in frozen split/source-speed/profile order.

## Overall outcomes

| Outcome | TRAIN | VALIDATION | Total |
|---|---:|---:|---:|
| Planned | 132 | 66 | 198 |
| Completed | 132 | 66 | 198 |
| Objective valid | 124 | 64 | 188 |
| Strict Sand | 93 | 47 | 140 |
| Mild strict Sand | 71 | 36 | 107 |
| Moderate strict Sand | 22 | 11 | 33 |
| Ordinary Support | 20 | 11 | 31 |
| Delayed Support | 11 | 6 | 17 |
| Slip | 1 | 1 | 2 |
| Dual Hazard | 1 | 1 | 2 |
| Pretarget fall | 0 | 0 | 0 |
| Post-target fall/censor | 2 | 0 | 2 |
| Other invalid | 4 | 0 | 4 |

TRAIN supplied 124 objective-valid records; VALIDATION supplied 64. Mild Sand produced 107/108 strict records: TRAIN 71/72 and VALIDATION 36/36. Moderate Sand produced 33/36 strict records: TRAIN 22/24 and VALIDATION 11/12. Both mild and moderate frozen split gates passed.

## Source-speed Sand

Here `Valid` is objective-valid under the frozen Sand contract; `Invalid` is the sum of pretarget fall, target-following fall/censor, and other invalid.

| Split | Source | Speed | Planned | Valid | Strict | Slip | Dual | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FACTOR_TRAIN | concrete | 0.20 | 16 | 16 | 16 | 0 | 0 | 0 |
| FACTOR_TRAIN | concrete | 0.25 | 16 | 15 | 15 | 0 | 0 | 1 |
| FACTOR_TRAIN | concrete | 0.30 | 16 | 16 | 16 | 0 | 0 | 0 |
| FACTOR_TRAIN | marble | 0.20 | 16 | 16 | 16 | 0 | 0 | 0 |
| FACTOR_TRAIN | marble | 0.25 | 16 | 15 | 15 | 0 | 0 | 1 |
| FACTOR_TRAIN | marble | 0.30 | 16 | 15 | 15 | 1 | 0 | 0 |
| FACTOR_VALIDATION | concrete | 0.20 | 8 | 8 | 8 | 0 | 0 | 0 |
| FACTOR_VALIDATION | concrete | 0.25 | 8 | 8 | 8 | 0 | 0 | 0 |
| FACTOR_VALIDATION | concrete | 0.30 | 8 | 8 | 8 | 0 | 0 | 0 |
| FACTOR_VALIDATION | marble | 0.20 | 8 | 8 | 8 | 0 | 0 | 0 |
| FACTOR_VALIDATION | marble | 0.25 | 8 | 8 | 8 | 0 | 0 | 0 |
| FACTOR_VALIDATION | marble | 0.30 | 8 | 7 | 7 | 1 | 0 | 0 |

All 12 split/source-speed minimum gates passed. Designed Sand Slip-plus-Dual was 2 against the frozen ceiling of 6. The frozen config defines combined Slip-plus-Dual ceilings, not independent total-Dual or other-invalid ceilings; no post-hoc gates were invented.

## Factor manifolds

The Concrete/.25 exception rows are a subset of adverse-direction rows and are shown separately for auditability, not added again to totals.

| Split | Manifold | Planned | Strict | Slip/Dual | Invalid | Yield | Measured phase truth |
|---|---|---:|---:|---:|---:|---:|---|
| FACTOR_TRAIN | transition-left / adverse | 42 | 42 | 0 | 0 | 100.0% | RIGHT_SINGLE_SUPPORT 42 |
| FACTOR_TRAIN | transition-right / comparison | 30 | 29 | 0 | 1 | 96.7% | LEFT_SINGLE_SUPPORT 30 |
| FACTOR_TRAIN | Concrete/.25 adverse exception | 12 | 12 | 0 | 0 | 100.0% | RIGHT_SINGLE_SUPPORT 12 |
| FACTOR_VALIDATION | transition-left / adverse | 21 | 21 | 0 | 0 | 100.0% | RIGHT_SINGLE_SUPPORT 21 |
| FACTOR_VALIDATION | transition-right / comparison | 15 | 15 | 0 | 0 | 100.0% | LEFT_SINGLE_SUPPORT 15 |
| FACTOR_VALIDATION | Concrete/.25 adverse exception | 6 | 6 | 0 | 0 | 100.0% | RIGHT_SINGLE_SUPPORT 6 |

The measured phase truth exactly followed the designed topology/phase coupling. This is not evidence of independent phase manipulation.

## Ordinary and delayed Support

| Split | Type | Source | Speed | Planned | Support | Dual | Slip | Invalid | Yield |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FACTOR_TRAIN | delayed | concrete | 0.20 | 2 | 2 | 0 | 0 | 0 | 100.0% |
| FACTOR_TRAIN | delayed | concrete | 0.25 | 2 | 2 | 0 | 0 | 0 | 100.0% |
| FACTOR_TRAIN | delayed | concrete | 0.30 | 2 | 2 | 0 | 0 | 0 | 100.0% |
| FACTOR_TRAIN | delayed | marble | 0.20 | 2 | 2 | 0 | 0 | 0 | 100.0% |
| FACTOR_TRAIN | delayed | marble | 0.25 | 2 | 1 | 1 | 0 | 0 | 50.0% |
| FACTOR_TRAIN | delayed | marble | 0.30 | 2 | 2 | 0 | 0 | 0 | 100.0% |
| FACTOR_TRAIN | ordinary | concrete | 0.20 | 4 | 4 | 0 | 0 | 0 | 100.0% |
| FACTOR_TRAIN | ordinary | concrete | 0.25 | 4 | 4 | 0 | 0 | 0 | 100.0% |
| FACTOR_TRAIN | ordinary | concrete | 0.30 | 4 | 1 | 0 | 0 | 3 | 25.0% |
| FACTOR_TRAIN | ordinary | marble | 0.20 | 4 | 4 | 0 | 0 | 0 | 100.0% |
| FACTOR_TRAIN | ordinary | marble | 0.25 | 4 | 4 | 0 | 0 | 0 | 100.0% |
| FACTOR_TRAIN | ordinary | marble | 0.30 | 4 | 3 | 0 | 0 | 1 | 75.0% |
| FACTOR_VALIDATION | delayed | concrete | 0.20 | 1 | 1 | 0 | 0 | 0 | 100.0% |
| FACTOR_VALIDATION | delayed | concrete | 0.25 | 1 | 1 | 0 | 0 | 0 | 100.0% |
| FACTOR_VALIDATION | delayed | concrete | 0.30 | 1 | 1 | 0 | 0 | 0 | 100.0% |
| FACTOR_VALIDATION | delayed | marble | 0.20 | 1 | 1 | 0 | 0 | 0 | 100.0% |
| FACTOR_VALIDATION | delayed | marble | 0.25 | 1 | 1 | 0 | 0 | 0 | 100.0% |
| FACTOR_VALIDATION | delayed | marble | 0.30 | 1 | 1 | 0 | 0 | 0 | 100.0% |
| FACTOR_VALIDATION | ordinary | concrete | 0.20 | 2 | 1 | 1 | 0 | 0 | 50.0% |
| FACTOR_VALIDATION | ordinary | concrete | 0.25 | 2 | 2 | 0 | 0 | 0 | 100.0% |
| FACTOR_VALIDATION | ordinary | concrete | 0.30 | 2 | 2 | 0 | 0 | 0 | 100.0% |
| FACTOR_VALIDATION | ordinary | marble | 0.20 | 2 | 2 | 0 | 0 | 0 | 100.0% |
| FACTOR_VALIDATION | ordinary | marble | 0.25 | 2 | 2 | 0 | 0 | 0 | 100.0% |
| FACTOR_VALIDATION | ordinary | marble | 0.30 | 2 | 2 | 0 | 0 | 0 | 100.0% |

Ordinary Support was TRAIN 20/24 against `>=22` (**FAIL**) and VALIDATION 11/12 against `>=11` (PASS). Delayed Support was TRAIN 11/12 against `>=11` (PASS) and VALIDATION 6/6 against `>=5` (PASS). Delayed contamination was 1/18 against `<=2`, TRAIN 1 against `<=1`, and VALIDATION 0 against `<=1`; all passed. The recalibrated delayed envelope therefore resolved the previously failed delayed-Support gates, but the complete corpus is insufficient because ordinary-Support TRAIN yield failed.

## Delayed-Support physical ledger

Times are simulation milliseconds. `Obs valid` is the frozen censor-aware observation contract.

| Run | Split | Source | Speed | Side | Topology | Measured phase | Entry | I1 | Support | Slip | Dual | Obs valid | Outcome |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---|---|---|
| sfcsr_t_dsp_c_020_01 | FACTOR_TRAIN | concrete | 0.20 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1808 | 3607 | 3661 | — | no | yes | SUPPORT |
| sfcsr_t_dsp_c_020_02 | FACTOR_TRAIN | concrete | 0.20 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1808 | 3607 | 3661 | — | no | yes | SUPPORT |
| sfcsr_t_dsp_c_025_01 | FACTOR_TRAIN | concrete | 0.25 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 3011 | 3067 | — | no | yes | SUPPORT |
| sfcsr_t_dsp_c_025_02 | FACTOR_TRAIN | concrete | 0.25 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 3011 | 3067 | — | no | yes | SUPPORT |
| sfcsr_t_dsp_c_030_01 | FACTOR_TRAIN | concrete | 0.30 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 2421 | 3053 | — | no | yes | SUPPORT |
| sfcsr_t_dsp_c_030_02 | FACTOR_TRAIN | concrete | 0.30 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 2421 | 3072 | — | no | yes | SUPPORT |
| sfcsr_t_dsp_m_020_01 | FACTOR_TRAIN | marble | 0.20 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1809 | 3609 | 3663 | — | no | yes | SUPPORT |
| sfcsr_t_dsp_m_020_02 | FACTOR_TRAIN | marble | 0.20 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1810 | 3609 | 3663 | — | no | yes | SUPPORT |
| sfcsr_t_dsp_m_025_01 | FACTOR_TRAIN | marble | 0.25 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 3012 | 3068 | — | no | yes | SUPPORT |
| sfcsr_t_dsp_m_025_02 | FACTOR_TRAIN | marble | 0.25 | BILATERAL | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 3012 | 3068 | 6515 | yes | yes | DUAL_HAZARD |
| sfcsr_t_dsp_m_030_01 | FACTOR_TRAIN | marble | 0.30 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 2422 | 3058 | — | no | yes | SUPPORT |
| sfcsr_t_dsp_m_030_02 | FACTOR_TRAIN | marble | 0.30 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 2422 | 3099 | — | no | yes | SUPPORT |
| sfcsr_v_dsp_c_020_01 | FACTOR_VALIDATION | concrete | 0.20 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1808 | 3607 | 3661 | — | no | yes | SUPPORT |
| sfcsr_v_dsp_c_025_01 | FACTOR_VALIDATION | concrete | 0.25 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 3011 | 3067 | — | no | yes | SUPPORT |
| sfcsr_v_dsp_c_030_01 | FACTOR_VALIDATION | concrete | 0.30 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 2421 | 3056 | — | no | yes | SUPPORT |
| sfcsr_v_dsp_m_020_01 | FACTOR_VALIDATION | marble | 0.20 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1810 | 3609 | 3663 | — | no | yes | SUPPORT |
| sfcsr_v_dsp_m_025_01 | FACTOR_VALIDATION | marble | 0.25 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 3012 | 3068 | — | no | yes | SUPPORT |
| sfcsr_v_dsp_m_030_01 | FACTOR_VALIDATION | marble | 0.30 | LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 2422 | 3063 | — | no | yes | SUPPORT |

The sole delayed miss, `sfcsr_t_dsp_m_025_02`, used start 0.332 m, width 0.833 m, exit 1.165 m, marble/.25, measured RIGHT_SINGLE_SUPPORT. Entry was 1220 ms, I1 3012 ms, Support 3068 ms, and genuine Slip 6515 ms: Support preceded Slip by 3447 ms, producing fully observed bilateral `DUAL_HAZARD`. Its post-Support observation was 5932 ms. It was retained without recalibration or rerun.

## Invalidity and ordinary-Support failure localization

Six records were physically invalid: zero pretarget falls, two target-following fall/censors, and four other invalid records. The four other-invalid records were ordinary-Support controls with only 563–990 ms after Support before physical fall/censor, below the frozen 1000 ms contract. These four TRAIN misses caused the sole gate failure. One VALIDATION ordinary Support was a genuine late dual hazard rather than an invalid observation.

| Run | Group | Source/speed | Start/width/exit m | Phase | Entry | I1 | Support | Slip | Fall/censor | Post-event ms | Result |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| sfcsr_t_smd_c_025_01 | moderate Sand | concrete/.25 | .328/.846/1.174 | RIGHT_SINGLE | 1220 | — | — | — | 5186 | 3966 after entry | target-following fall/censor |
| sfcsr_t_osp_c_030_01 | ordinary Support | concrete/.30 | .329/.713/1.042 | RIGHT_SINGLE | 1227 | 1246 | 3036 | — | 3599 | 563 after Support | other invalid |
| sfcsr_t_osp_c_030_02 | ordinary Support | concrete/.30 | .337/.723/1.060 | RIGHT_SINGLE | 1227 | 1246 | 3036 | — | 3602 | 566 after Support | other invalid |
| sfcsr_t_osp_c_030_03 | ordinary Support | concrete/.30 | .331/.715/1.046 | LEFT_SINGLE | 1508 | 1527 | 2731 | — | 3581 | 850 after Support | other invalid |
| sfcsr_t_sml_m_025_10 | mild Sand | marble/.25 | .340/.785/1.125 | LEFT_SINGLE | 1509 | — | — | — | 5478 | 3969 after entry | target-following fall/censor |
| sfcsr_t_osp_m_030_01 | ordinary Support | marble/.30 | .329/.713/1.042 | RIGHT_SINGLE | 1227 | 1246 | 3036 | — | 4026 | 990 after Support | other invalid |
| sfcsr_v_osp_c_020_01 | ordinary Support | concrete/.20 | .349/.745/1.094 | RIGHT_SINGLE | 1808 | 1827 | 2475 | 6874 | 9000 | 6525 after Support | DUAL_HAZARD |

The table includes the non-invalid ordinary-Support dual because it contributes to the ordinary-Support yield miss. No generator, label-semantic, or observation-horizon change was made after seeing these outcomes.

## Physical signatures and anti-contamination

| Metric | Result |
|---|---:|
| Scenario unique / total | 198 / 198 |
| Physical unique / total | 177 / 198 |
| Valid physical unique / valid total | 167 / 188 |
| Valid physical uniqueness fraction | 0.8882978723 |
| Frozen minimum | 0.80 (PASS) |
| Duplicate-row excess | 21 |
| Exact physical duplicate pair combinations | 39 |
| Historical exact overlap | 0 |
| Historical forbidden-near overlap | 0 |
| Historical run-ID reuse | 0 |
| Failed-198 exact / forbidden-near overlap | 0 / 0 |
| Pilot exact / forbidden-near overlap | 0 / 0 |
| Pilot run-ID reuse | 0 |
| Cross-split exact / forbidden-near overlap | 0 / 0 |

The 39 exact physical-signature pair combinations were:

```text
sfcsr_t_sml_c_025_02 <> sfcsr_v_sml_c_025_03
sfcsr_t_sml_c_025_03 <> sfcsr_v_sml_c_025_06
sfcsr_t_sml_c_030_04 <> sfcsr_v_sml_c_030_02
sfcsr_t_sml_c_030_07 <> sfcsr_t_sml_c_030_11
sfcsr_t_sml_c_030_07 <> sfcsr_t_sml_c_030_12
sfcsr_t_sml_c_030_07 <> sfcsr_v_sml_c_030_05
sfcsr_t_sml_c_030_11 <> sfcsr_t_sml_c_030_12
sfcsr_t_sml_c_030_11 <> sfcsr_v_sml_c_030_05
sfcsr_t_sml_c_030_12 <> sfcsr_v_sml_c_030_05
sfcsr_t_sml_c_030_08 <> sfcsr_t_sml_c_030_09
sfcsr_t_sml_c_030_08 <> sfcsr_t_sml_c_030_10
sfcsr_t_sml_c_030_08 <> sfcsr_v_sml_c_030_04
sfcsr_t_sml_c_030_08 <> sfcsr_v_sml_c_030_06
sfcsr_t_sml_c_030_09 <> sfcsr_t_sml_c_030_10
sfcsr_t_sml_c_030_09 <> sfcsr_v_sml_c_030_04
sfcsr_t_sml_c_030_09 <> sfcsr_v_sml_c_030_06
sfcsr_t_sml_c_030_10 <> sfcsr_v_sml_c_030_04
sfcsr_t_sml_c_030_10 <> sfcsr_v_sml_c_030_06
sfcsr_v_sml_c_030_04 <> sfcsr_v_sml_c_030_06
sfcsr_t_smd_c_030_02 <> sfcsr_t_smd_c_030_04
sfcsr_t_sml_m_020_04 <> sfcsr_v_sml_m_020_02
sfcsr_t_osp_m_025_01 <> sfcsr_t_osp_m_025_02
sfcsr_t_sml_m_030_04 <> sfcsr_v_sml_m_030_02
sfcsr_t_sml_m_030_07 <> sfcsr_t_sml_m_030_11
sfcsr_t_sml_m_030_07 <> sfcsr_t_sml_m_030_12
sfcsr_t_sml_m_030_07 <> sfcsr_v_sml_m_030_05
sfcsr_t_sml_m_030_11 <> sfcsr_t_sml_m_030_12
sfcsr_t_sml_m_030_11 <> sfcsr_v_sml_m_030_05
sfcsr_t_sml_m_030_12 <> sfcsr_v_sml_m_030_05
sfcsr_t_sml_m_030_08 <> sfcsr_t_sml_m_030_09
sfcsr_t_sml_m_030_08 <> sfcsr_t_sml_m_030_10
sfcsr_t_sml_m_030_08 <> sfcsr_v_sml_m_030_04
sfcsr_t_sml_m_030_08 <> sfcsr_v_sml_m_030_06
sfcsr_t_sml_m_030_09 <> sfcsr_t_sml_m_030_10
sfcsr_t_sml_m_030_09 <> sfcsr_v_sml_m_030_04
sfcsr_t_sml_m_030_09 <> sfcsr_v_sml_m_030_06
sfcsr_t_sml_m_030_10 <> sfcsr_v_sml_m_030_04
sfcsr_t_sml_m_030_10 <> sfcsr_v_sml_m_030_06
sfcsr_v_sml_m_030_04 <> sfcsr_v_sml_m_030_06
```

These are realized physical-signature collisions, not scenario-signature overlap. The frozen protocol gates only the valid uniqueness fraction; it defines no physical-near criterion, so none was added after generation.

## Complete generation-gate ledger

| Gate | Threshold | Observed | Result | Evidence |
|---|---:|---:|---|---|
| `censor/pretarget_fall` | <=3 | 0 | PASS | `manifest.json` |
| `censor/target_following_fall` | <=8 | 2 | PASS | `manifest.json` |
| `contamination/FACTOR_TRAIN/delayed_support_slip_plus_dual` | <=1 | 1 | PASS | `manifest.json` |
| `contamination/FACTOR_VALIDATION/delayed_support_slip_plus_dual` | <=1 | 0 | PASS | `manifest.json` |
| `contamination/delayed_support_slip_plus_dual` | <=2 | 1 | PASS | `manifest.json` |
| `contamination/designed_sand_slip_plus_dual` | <=6 | 2 | PASS | `manifest.json` |
| `diversity/valid_physical_signature_uniqueness_fraction` | >=0.8 | 0.8882978723404256 | PASS | `manifest.json` |
| `execution/adaptive_backfill` | 0 | 0 | PASS | `manifest.json` |
| `execution/attempted` | 198 | 198 | PASS | `manifest.json` |
| `execution/completed` | 198 | 198 | PASS | `manifest.json` |
| `execution/planned` | 198 | 198 | PASS | `pre_simulation_freeze.json` |
| `execution/replacement` | 0 | 0 | PASS | `manifest.json` |
| `execution/rerun` | 0 | 0 | PASS | `manifest.json` |
| `integrity/cross_split_exact_overlap` | 0 | 0 | PASS | `pre_simulation_freeze.json` |
| `integrity/cross_split_forbidden_near_overlap` | 0 | 0 | PASS | `pre_simulation_freeze.json` |
| `integrity/failed_198_exact_overlap` | 0 | 0 | PASS | `pre_simulation_freeze.json` |
| `integrity/failed_198_forbidden_near_overlap` | 0 | 0 | PASS | `pre_simulation_freeze.json` |
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
| `yield/FACTOR_TRAIN/concrete/0.20/strict_sand` | >=14 | 16 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/concrete/0.25/strict_sand` | >=14 | 15 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/concrete/0.30/strict_sand` | >=14 | 16 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/delayed_support` | >=11 | 11 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/marble/0.20/strict_sand` | >=14 | 16 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/marble/0.25/strict_sand` | >=14 | 15 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/marble/0.30/strict_sand` | >=14 | 15 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/mild` | >=68 | 71 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/mild_adverse_direction` | >=39 | 42 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/mild_comparison_direction` | >=28 | 29 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/moderate` | >=22 | 22 | PASS | `manifest.json` |
| `yield/FACTOR_TRAIN/ordinary_support` | >=22 | 20 | **FAIL** | `manifest.json` |
| `yield/FACTOR_TRAIN/strict_sand` | >=91 | 93 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/concrete/0.20/strict_sand` | >=7 | 8 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/concrete/0.25/strict_sand` | >=7 | 8 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/concrete/0.30/strict_sand` | >=7 | 8 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/delayed_support` | >=5 | 6 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/marble/0.20/strict_sand` | >=7 | 8 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/marble/0.25/strict_sand` | >=7 | 8 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/marble/0.30/strict_sand` | >=7 | 7 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/mild` | >=34 | 36 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/mild_adverse_direction` | >=19 | 21 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/mild_comparison_direction` | >=14 | 15 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/moderate` | >=10 | 11 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/ordinary_support` | >=11 | 11 | PASS | `manifest.json` |
| `yield/FACTOR_VALIDATION/strict_sand` | >=44 | 47 | PASS | `manifest.json` |
| `yield/objective_valid` | >=180 | 188 | PASS | `manifest.json` |

Gate result: **57/58 PASS, 1 FAIL**. The only failure is `yield/FACTOR_TRAIN/ordinary_support` (20 observed, at least 22 required).

## Dataset freeze

The insufficient corpus was retained and frozen exactly as generated:

| Artifact/identity | SHA-256 |
|---|---|
| Pre-simulation freeze | `36b0ab505013b23811c23807583164c1bdc6ad9f8e44ca6329ef648414fe05d1` |
| Execution config | `c4bde7adc1d4a917fe78da9b2f470c1f6155af63a06f314972609c64ef5aade4` |
| Manifest | `bda6961f79525df237d440086057aea71a03afc5134a49c50f5e4d4a2193be67` |
| FACTOR_TRAIN split | `5283046afaffbe4dd41947bc7db792e30d9c71dec13322d30bd9e0801c035a31` |
| FACTOR_VALIDATION split | `f81a9acb0b1947d4da0a5c09211c7a96050cdc86d0b5873bf885b8209fe8c901` |
| Scenario signatures | `0944705e3cb18ff78f4edf68573fbf56477ae9fc7cf7576a2894145549feb4be` |
| Physical signatures | `043057dabcabc102fc6ba5ce4d8ff66d1e5a1a489594d63e1f74d0f771bf45ec` |
| NPZ aggregate | `fb64922ae4316c272568362003958e1dc9e2008e591cc353338d56e42394f5d3` |
| Physical outcomes | `811f2900e7b3198a7dd15f28eed3273f1601d617ccb4b38a49414ab95fa334e4` |
| Generation gates | `23af29362e3177e039eb94744f432f67b7b01b4212d7d4cebb9c5975978cae65` |
| Physical audit | `f4861dc18456da28f76caf38257de03abef90e184879378bcbe844301490f9a3` |
| Validation seal | `4bbc5df8d80d3be811117700970ca068f3932edcd10f62feb7f97cc5377dd728` |
| Semantic dataset freeze | `d50602a59d196416825b09a3b49fed297ef8f2bf324eba298aadba99c988ce3b` |
| Dataset-freeze file | `fb575566574ef87bdc6ca8c161cb770c6d16e530b18fda3df0b65d213ad59922` |
| Generation summary | `90b7dc1ab1c3ead5c524a2d20554c68f850e1f7f26695e13715d7828c186f053` |

## FACTOR_VALIDATION status

`FACTOR_VALIDATION` is `SEALED_FAILED_PHYSICAL_EVIDENCE`. Model inference, training use, HNM use, normalized-80D analysis, and visualization were all zero. The canonical loader rejects VALIDATION before NPZ access. This split is not authorized for model science.

## Model boundary and counters

| Counter | Actual |
|---|---:|
| New full-study simulations | 198 |
| New pilot simulations | 0 |
| Replacement / backfill / rerun | 0 / 0 / 0 |
| V1 / V2 / Terrain inference | 0 / 0 / 0 |
| Hazard probability calculations | 0 |
| Training / optimizer steps / checkpoint writes | 0 / 0 / 0 |
| Normalizer fits / HNM rounds | 0 / 0 |
| 80D model analysis | 0 |
| Threshold / persistence searches | 0 / 0 |
| Architecture / sensor-fusion experiments | 0 / 0 |
| Old HOLDOUT reads / inference | 0 / 0 |
| FACTOR_VALIDATION model inference / training / HNM | 0 / 0 / 0 |

## Verdict and next milestone

Canonical verdict:

`SAND_FACTOR_CONDITIONED_DEVELOPMENT_SUPPORT_RECALIBRATED_GENERATION_INSUFFICIENT`

Do not start `SAND_FACTOR_CONDITIONED_MODEL_TRAINING`. The smallest next physical follow-up is a saved-evidence-first review of the five ordinary-Support misses, especially the four .30 m/s post-Support fall/censor cases, without reopening delayed-Support calibration or changing Sand semantics. A suitable next milestone is `SAND_FACTOR_CONDITIONED_ORDINARY_SUPPORT_FAILURE_REVIEW`.

Deployment engineering may continue independently with `model_v2_anchor_refined_gru20_20260902` only as `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`. No E84 repository files were modified.

## Tests

Pre-generation targeted design/integrity tests: 3 passed. Legacy factor-conditioned regression tests: 12 passed. Post-generation targeted tests: 17 passed. Full repository suite: 189 passed, 1 skipped. `compileall`, Ruff E9/F63/F7/F82, configured full Ruff, `git diff --check`, and the independent 13-check frozen-dataset verifier all passed.
