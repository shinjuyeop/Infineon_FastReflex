# Factor-Conditioned Hazard Model Training

## 1. Purpose

This milestone tested exactly one controlled data-only intervention: whether adding the fresh, physically frozen factor-conditioned Sand/Support TRAIN domain lets the unchanged Pelvis-IMU6, causal-80D, GRU20 family reject benign Sand while preserving genuine Support. It trained one three-seed family, consumed the fresh development validation once only after candidate freeze, and stopped without adaptation.

The frozen verdict is `FACTOR_CONDITIONED_DATA_INTERVENTION_NOT_EFFECTIVE`. Sand rejection improved substantially, but genuine Support detection collapsed on both the fresh split and historical `V2_VALIDATION`. The exact next milestone is the read-only `FACTOR_CONDITIONED_DATA_INTERVENTION_FAILURE_AUDIT`; it is not started here.

## 2. Starting state

| Item | Result |
|---|---|
| Expected scientific base | `68ebc1ddebfaec106f5fa85d48eecf9d647c8896` |
| Starting `HEAD` / `origin/main` | exact expected base |
| Expected base ancestor | yes |
| Starting tracked worktree | clean |
| Parallel changes | none observed |
| Training implementation source | `546f9bcd68c5bbd636be48570b4f89a2f636be06` |
| Frozen protocol commit | `5acd6510ac5ea2a3b96dfb7633af8b67c0d3adba` |

## 3. Scientific boundary

The historical final Model V2 remains `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`; whole-simulation status remains `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`; Support status remains `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`. The permanently consumed Generalization HOLDOUT stayed at `guard_after=1`, `scientific_open_count=1` and had zero payload reads, feature reconstruction, inference, visualization, training use, or HNM use. Generalization VALIDATION also received no new-candidate inference. No simulator, dataset, threshold, persistence, architecture, feature, sensor, or normalizer intervention was introduced, and the E84 repository was not modified.

## 4. Fresh dataset

`sand_factor_conditioned_development_controls_recalibrated_20260903` was verified before model work.

| Frozen identity | SHA-256 / result |
|---|---|
| Manifest | `70f850f22507384a50c81bdd065c7d485f2e37cd7e181fda20431ed2fede2d50` |
| NPZ aggregate | `464348e45b20d8c5adac7965e47233b0cae020adc66e3515209e84ddc9da6e21` |
| Physical audit | `7081ae25ed20659992392913f50b5001ca7eafa510f92cafb0edda7038f633be` |
| Validation seal | `761aec9eb466efa2b067cd5f10d13a76482cd66561c316801321bb20870a31b3` |
| Semantic dataset freeze | `e397c78d19386732eb54ba388c551a8fe213a6097ba7d0819ff20f7b9b0255f4` |
| Dataset-freeze file | `c9fee8eed1e75c23ce44c9d9fa4a9d204b150ed69e20b467d0c188e22e8194df` |
| FACTOR_TRAIN split | `039725f3231b2f48daae3f9e0d5f768613fdb71c3b6c15050e4f62308cef45c2` |
| FACTOR_VALIDATION split | `8e4a915347781d1654a9a05ec0e78754a7b7515284abfb6433197978ecce72e0` |

The physical population was 198/198 completed and 196/198 objective-valid, with all 61 gates passing. TRAIN contributed 96 strict Sand, 24 ordinary Support, and 12 delayed Support. VALIDATION retained 46 strict Sand, 12 ordinary Support, 6 delayed Support, one actual Slip for secondary description, and one invalid/censored row for provenance only.

## 5. Training-source contract

| Source dataset | Role | Physical/class population | Used for training? | Used for HNM? | Used for validation? |
|---|---|---|---:|---:|---:|
| Unified TRAIN | historical TRAIN | canonical Hazard and benign | yes | yes | no |
| V2_TRAIN | historical augmentation | valid actual physical outcomes | yes | yes | no |
| FACTOR_TRAIN | intervention TRAIN | 96 strict Sand + 24 ordinary + 12 delayed Support | yes | yes | no |
| V2_VALIDATION | predeclared regression only | historical development | no | no | once, post-freeze |
| FACTOR_VALIDATION | primary fresh development | 46 strict Sand + 18 Support + secondary Slip | no | no | once, post-freeze |
| Generalization VALIDATION | protected historical development | historical only | no | no | no |
| old Sand Discovery | prior evidence | historical only | no | no | no |
| old Sand Confirmation | prior evidence | historical only | no | no | no |
| historical HOLDOUT | permanently consumed final evidence | sealed | no | no | no |

Training-source ledger SHA-256: `86f25596db4481409d4b8eba48b45fc6a60ab145d1d077db6eeb4577a2e0e9f3`.

## 6. Architecture freeze

The exact architecture remained Pelvis IMU6 → causal `[20,80]` → one-layer, unidirectional GRU with hidden size 32 → Linear 32→2. Each model has 11,010 parameters. Architecture SHA-256 is `ae4753699399aa7e4a93639935e085156ef28ed100a77058d7308eb1549aa897`.

## 7. Feature freeze

The ten frozen base signals and eight frozen causal transforms remained the exact 80D schema. No future sample or privileged physical value enters the tensor. Feature schema SHA-256 is `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`.

## 8. Normalizer sanity

The exact V2 normalizer `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` was reused; fit count was zero. The pre-optimizer FACTOR_TRAIN-only audit sampled 132,000 80D rows from all 132 runs: all values were finite, absolute-z p95/p99/p99.9/max were 1.17424/2.49657/7.95764/68.94363, fraction `|z|>20` was 0.000311553, and fraction `|z|>50` was 0.000024811. The frozen max-500 and 1%-above-20 gates passed.

## 9. Training protocol and TRAIN composition

Config SHA-256 `10b723270478b87fae9854cbf5a165386e8317d2dadef9a36f846c13d7218a22` was committed before optimizer step 1. The only intervention variable was fresh data coverage. Training used Adam, learning rate 0.001, zero weight decay, inverse-frequency weighted cross entropy, batch 128, maximum 40 epochs, patience 6, and the deterministic TRAIN-only monitor. Seeds were exactly 20260828/20260829/20260830. Runtime stayed at ensemble mean, threshold 0.99, persistence 5 consecutive 1 ms samples.

The effective population was Unified TRAIN 152 + valid V2_TRAIN 290 + FACTOR_TRAIN 132 = 574 runs. FACTOR_TRAIN contained only actual `STRICT_BENIGN` 96 and actual `SUPPORT` 36; there was no physical mismatch to improvise over.

## 10. HNM

Round 0 was followed by exactly three canonical TRAIN-only HNM passes. Each pass scored 574 runs and added 6,888 endpoints: 5,304 historical and 1,584 new-factor endpoints. The new-factor contribution was Sand 1,152, ordinary Support 288, delayed Support 144; concrete/marble were 792/792; speeds .20/.25/.30 were 528 each; transition-left/right and right-single/left-single were 1,056/528. Every pass had zero duplicate, spacing, future-Slip precursor, censored precursor, I1-positive, or post-censor/fall violation. HNM provenance SHA-256 is `d345e61194d1d102be443f0f2e39bf8e8e965048e89075ecb464ccadfe5b6c0d`.

## 11. Per-seed training

`New Sand`, `New ordinary`, and `New delayed` are fit-window exposures per epoch; `Epoch` is best/completed. Checkpoint paths follow `artifacts/runs/20260904_sand_factor_conditioned_model_training/checkpoints/model_v2_factor_conditioned_gru_history20_round{Round}_seed{Seed}.pt`. No seed was selected or removed.

| Seed | Round | Pos | Neg | New Sand | New ordinary Support | New delayed Support | HNM added | Epoch | Steps | Monitor CE | Checkpoint SHA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20260828` | 0 | 2810 | 34106 | 6368 | 1598 | 925 | 6888 | 4/10 | 2890 | 0.162968 | `dc1d5d9c7bd36f8e02724b06105c4e376ad376946da0627e434d0a84317d2984` |
| `20260829` | 0 | 2810 | 34106 | 6368 | 1598 | 925 | 6888 | 7/13 | 3757 | 0.169344 | `ecab6e14b75a4ba18197a1d136892cd30bc1ebf0796216153611670a62fc2a03` |
| `20260830` | 0 | 2810 | 34106 | 6368 | 1598 | 925 | 6888 | 4/10 | 2890 | 0.197880 | `047af3c3918d516de2e0073bc32e30df2716db4af8f2c8fccdb179e09b043f65` |
| `20260828` | 1 | 2810 | 39312 | 7304 | 1810 | 1042 | 6888 | 6/12 | 3960 | 0.179950 | `8de36e939871aa57df0d2d9aa17b2272ad185f41e0d3393890a2ca9b58d91c30` |
| `20260829` | 1 | 2810 | 39312 | 7304 | 1810 | 1042 | 6888 | 6/12 | 3960 | 0.163870 | `4b8cdffa464346f96e2044306953a9e56ab2ecd14e3279afdedcf4cf71097f7b` |
| `20260830` | 1 | 2810 | 39312 | 7304 | 1810 | 1042 | 6888 | 8/14 | 4620 | 0.172951 | `087805eb230e981b49fa7c9219a63e50a77ef7d23bd3e149cc14f591badd7053` |
| `20260828` | 2 | 2810 | 44672 | 8237 | 2040 | 1155 | 6888 | 4/10 | 3710 | 0.194241 | `d059c14cec44e544f394e99292e121b1c4d8e6853d1ab5ac6a30ecad26baa2e6` |
| `20260829` | 2 | 2810 | 44672 | 8237 | 2040 | 1155 | 6888 | 8/14 | 5194 | 0.217907 | `51c40ba21a7a350a59620d74609a2a8d7394b261504c3c153c1a183b9ca234a9` |
| `20260830` | 2 | 2810 | 44672 | 8237 | 2040 | 1155 | 6888 | 7/13 | 4823 | 0.231176 | `db3e21ef1e391e9599346adeed4298e0c28919bec4f73ed9975b6f0df3e2930a` |
| `20260828` | 3 | 2810 | 49987 | 9165 | 2254 | 1272 | 0 | 3/9 | 3717 | 0.210033 | `58df8b1ceb4bd4ab9a3343506eefc83b49d2d98dd614250229b589def061b8ec` |
| `20260829` | 3 | 2810 | 49987 | 9165 | 2254 | 1272 | 0 | 10/16 | 6608 | 0.202590 | `c990b3f95a009e53ab878302192ba8b20e0971268900786413502c23584fe47e` |
| `20260830` | 3 | 2810 | 49987 | 9165 | 2254 | 1272 | 0 | 4/10 | 4130 | 0.222292 | `f5b1a9be27f1a5b84ef706b9da33227c90d434c2aac152fc1365cb876aea3837` |

Actual optimizer steps were 50,259 and checkpoint writes were 12.

## 12. Candidate freeze and validation authorization

The complete three-seed ensemble `model_v2_factor_conditioned_gru20_20260904` was frozen before validation as a `DEVELOPMENT_FACTOR_CONDITIONED_CANDIDATE`. Candidate SHA-256 is `27314d6d097c7daa3a2da66be1d017c4097787e276412670837046c9296029d6`; final Round-3 checkpoint SHAs are the final three rows above. It records the config, dataset, TRAIN/VALIDATION split, architecture, feature, normalizer, training ledger, HNM, optimizer, seeds, threshold and persistence identities.

Only afterward, at `2026-09-04T12:41:23+09:00`, authorization SHA-256 `aa8876addaf6fe17524dd3ce73faf5607f6393f8893ad69f70301f6d12f2bf15` moved the logical seal from `SEALED_FOR_FUTURE_FACTOR_VALIDATION` to `AUTHORIZED_ONCE_FOR_FROZEN_FACTOR_CONDITIONED_CANDIDATE`. Open count was 0→1. The 66-row split produced one same-memory load of 65 model-eligible payloads; the one invalid row remained metadata provenance. Reference and candidate replayed that same loaded set. The final logical state is `CONSUMED_DEVELOPMENT_VALIDATION`; no candidate mutation followed authorization.

## 13. FACTOR_VALIDATION result

| Metric | Reference V2 | New candidate | Delta | Frozen target | Pass? |
|---|---:|---:|---:|---:|---|
| strict Sand specificity | 44/46 = 0.9565 | 46/46 = 1.0000 | +0.0435 | candidate ≥.95 and gain ≥.05 | candidate PASS; gain FAIL |
| false Reflex | 2 | 0 | -2 | reduction ≥2 | PASS |
| adverse rate (`max p≥.95`) | 17/46 = .3696 | 2/46 = .0435 | -.3261 | reduction ≥.10 | PASS |
| median max p | .9154 | .6823 | -.2331 | lower is descriptive | — |
| p95 max p | .9944 | .9404 | -.0539 | lower is descriptive | — |
| Mild specificity | 33/35 = .9429 | 35/35 = 1.0000 | +.0571 | no severity transfer | PASS |
| Moderate specificity | 11/11 = 1.0000 | 11/11 = 1.0000 | 0 | no severity transfer | PASS |
| adverse manifold specificity | 29/31 = .9355 | 31/31 = 1.0000 | +.0645 | improve | PASS |
| adverse manifold adverse rate | 14/31 = .4516 | 2/31 = .0645 | -.3871 | reduction ≥.10 | PASS |
| comparison manifold specificity | 15/15 = 1.0000 | 15/15 = 1.0000 | 0 | preserve | PASS |
| comparison manifold adverse rate | 3/15 = .2000 | 0/15 = 0 | -.2000 | improve | PASS |
| ordinary Support recall | 12/12 = 1.0000 | 7/12 = .5833 | -.4167 | ≥.90 | **FAIL** |
| delayed Support recall | 4/6 = .6667 | 1/6 = .1667 | -.5000 | ≥.90 | **FAIL** |
| total Support recall | 16/18 = .8889 | 8/18 = .4444 | -.4444 | ≥.95 | **FAIL** |
| right Support recall | 7/7 = 1.0000 | 3/7 = .4286 | -.5714 | ≥.90 | **FAIL** |

Reference margin bins `<.90 / [.90,.95) / [.95,.99) / >=.99 subpersistent / Reflex` were 21/8/10/5/2. Candidate bins were 41/3/2/0/0. This is a real benign-margin improvement, not just removal of five-sample persistence crossings. It nevertheless fails the joint decision because the same downward shift suppressed genuine Support.

The complete strict-Sand maximum-probability distribution was reference median/p75/p90/p95/max `.915365/.973577/.992849/.994375/.995560` versus candidate `.682263/.769758/.901688/.940449/.961164`.

| Severity | N | Ref specific / FP / adverse | New specific / FP / adverse | Ref median / p95 max p | New median / p95 max p |
|---|---:|---:|---:|---:|---:|
| Mild | 35 | 33 / 2 / 15 | 35 / 0 / 2 | .929181 / .994618 | .675029 / .948847 |
| Moderate | 11 | 11 / 0 / 2 | 11 / 0 / 0 | .904211 / .992235 | .711797 / .830628 |

## 14. Source-speed and factor results

| Source | Speed | N | Ref TN/FP | New TN/FP | Ref adverse | New adverse | Ref median p | New median p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| concrete | .20 | 8 | 8/0 | 8/0 | 1 | 1 | .9236 | .6883 |
| concrete | .25 | 7 | 7/0 | 7/0 | 4 | 0 | .9563 | .6823 |
| concrete | .30 | 8 | 8/0 | 8/0 | 2 | 0 | .8991 | .6640 |
| marble | .20 | 8 | 8/0 | 8/0 | 7 | 0 | .9906 | .7063 |
| marble | .25 | 8 | 6/2 | 8/0 | 3 | 1 | .8393 | .8306 |
| marble | .30 | 7 | 7/0 | 7/0 | 0 | 0 | .8343 | .6577 |

Four of the four source-speed cells with reducible reference adverse counts improved; the two unchanged cells were concrete/.20 (1→1 adverse) and marble/.30 (0→0). Thus the Sand improvement was broad rather than confined to one trivial cell.

| Manifold | N | Ref FP | New FP | Ref adverse | New adverse | Improvement |
|---|---:|---:|---:|---:|---:|---|
| transition-left / right-single | 31 | 2 | 0 | 14 | 2 | yes |
| transition-right / left-single | 15 | 0 | 0 | 3 | 0 | yes |
| Concrete/.25 Mild exception | 5 | 0 | 0 | 4 | 0 | yes |

The generated manifest retains the Concrete/.25 rows under the generic `ADVERSE_DIRECTION` field. The frozen evaluator's separate enum filter therefore recorded zero under its convenience `concrete_025_exception` key. The exact table row above is a read-only filter of the already-frozen 46-row paired ledger (`source=concrete`, `speed=.25`, `severity=LOW`, `topology=transition_left`); it performs no payload read, inference, threshold change, or decision rewrite. This reporting correction cannot alter the NOT_EFFECTIVE verdict because Support gates fail decisively.

## 15. Support preservation

Timing is median I1→Reflex among detected runs; a positive delta is later. All fresh pre-I1 false-response counts were zero for both models.

| Group | N | Ref correct | New correct | Ref recall | New recall | Median I1→Reflex delta ms |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 12 | 12 | 7 | 1.0000 | .5833 | +592.0 |
| delayed | 6 | 4 | 1 | .6667 | .1667 | +23.5 |
| concrete | 9 | 8 | 4 | .8889 | .4444 | +300.0 |
| marble | 9 | 8 | 4 | .8889 | .4444 | +298.0 |
| left | 11 | 9 | 5 | .8182 | .4545 | +592.0 |
| right | 7 | 7 | 3 | 1.0000 | .4286 | +4.0 |

The regression spans ordinary/delayed, concrete/marble, left/right, and all speeds: new recall at .20/.25/.30 was 2/6, 5/6, and 1/6. It is not a localized subgroup miss.

For detected runs, reference→candidate median I1→Reflex times were ordinary 627→1219 ms, delayed 5.5→29 ms, concrete 623.5→923.5 ms, marble 625.5→923.5 ms, left 627→1219 ms, and right 624→628 ms. Median Reflex→Support times were respectively 19.5→19, 49.5→27, 21→23, 22→20.5, 21→19, and 23→22 ms. Detection-conditioned timing does not offset the large recall loss.

## 16. Actual Slip descriptive result

One physically valid actual Slip was present (`sfcocr_v_smd_m_030_01`). Reference max p was .995709 with longest .99 streak 4; candidate max p was .978957 with streak 0. Neither produced a five-sample Reflex, so descriptive detection was 0/1 for both. This tiny denominator is not evidence of Slip generalization and was not used to tune or redesign Slip.

## 17. Historical V2_VALIDATION regression

The predeclared candidate-only replay used saved reference V2 results rather than rerunning the reference. Historical no-hazard specificity and speed-Sand specificity stayed 1.0, and Slip recall changed 30/35→29/35 (`-.0286`, within the .03 drop allowance). However overall Hazard recall fell 59/64→45/64, Support fell 30/30→17/30, delayed Support 6/6→3/6, and right Support 12/12→4/12. Premature rate changed .0781→.0938. The Support, delayed, and right preservation gates all fail, independently confirming a major regression.

## 18. Intervention decision and hypothesis

The data intervention genuinely moved the Sand-benign boundary: false Reflex 2→0, adverse rate .3696→.0435, adverse-manifold adverse rate .4516→.0645, and broad source-speed margins improved. But it did so by suppressing the Support response. Fresh total Support was only 8/18 and historical V2_VALIDATION Support only 17/30. The candidate also missed the frozen Sand specificity-gain threshold by one finite-denominator run-equivalent (`+.0435 < +.05`).

Therefore the exact scientific verdict is `FACTOR_CONDITIONED_DATA_INTERVENTION_NOT_EFFECTIVE`, and hypothesis status is `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS_NOT_SUPPORTED_BY_DEVELOPMENT`. This does not promote a sensor, history, architecture, threshold, persistence, LSTM, or fusion hypothesis. The candidate is frozen as a failed development result and is not promoted to final, production, release, generalization-supported, or deployment-reference status. Its final generalization status is `NOT_ESTABLISHED`.

The one permitted next milestone is `FACTOR_CONDITIONED_DATA_INTERVENTION_FAILURE_AUDIT`, a saved-result read-only audit. It should examine why the unchanged extraction/weighting protocol converts the expanded benign population into broad Support suppression. It must not reopen FACTOR_VALIDATION, retrain, tune, or begin an external/final dataset. If a future modified candidate is ever justified, it will require a new independent external validation and then a new independently designed final HOLDOUT; the consumed historical HOLDOUT is permanently ineligible.

## 19. Complete paired strict-Sand ledger

The ledger below contains every one of the 46 strict Sand runs; p delta is candidate minus reference. Its artifact SHA-256 is `6f94e71018cae9974570ae96fafbb430e92cce1c76238f08fa36a862114a79f5`.

| Run | Source | Speed | Severity | Topology | Phase | Manifold | Ref max p | New max p | Delta p | Ref streak | New streak | Ref Reflex | New Reflex |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| `sfcocr_v_smd_c_020_01` | concrete | .20 | MEDIUM | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .917938 | .693752 | -.224186 | 0 | 0 | no | no |
| `sfcocr_v_smd_c_020_02` | concrete | .20 | MEDIUM | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .917938 | .721736 | -.196202 | 0 | 0 | no | no |
| `sfcocr_v_smd_c_025_01` | concrete | .25 | MEDIUM | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .851362 | .665309 | -.186054 | 0 | 0 | no | no |
| `sfcocr_v_smd_c_025_02` | concrete | .25 | MEDIUM | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .880323 | .711797 | -.168526 | 0 | 0 | no | no |
| `sfcocr_v_smd_c_030_01` | concrete | .30 | MEDIUM | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .904211 | .778492 | -.125719 | 0 | 0 | no | no |
| `sfcocr_v_smd_c_030_02` | concrete | .30 | MEDIUM | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .894067 | .686129 | -.207938 | 0 | 0 | no | no |
| `sfcocr_v_smd_m_020_01` | marble | .20 | MEDIUM | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .992235 | .606702 | -.385533 | 1 | 0 | no | no |
| `sfcocr_v_smd_m_020_02` | marble | .20 | MEDIUM | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .992235 | .606702 | -.385533 | 1 | 0 | no | no |
| `sfcocr_v_smd_m_025_01` | marble | .25 | MEDIUM | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .839346 | .784554 | -.054792 | 0 | 0 | no | no |
| `sfcocr_v_smd_m_025_02` | marble | .25 | MEDIUM | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .839346 | .876701 | +.037355 | 0 | 0 | no | no |
| `sfcocr_v_smd_m_030_02` | marble | .30 | MEDIUM | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .928782 | .783401 | -.145381 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_020_01` | concrete | .20 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .898964 | .959022 | +.060058 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_020_02` | concrete | .20 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .898964 | .944486 | +.045522 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_020_03` | concrete | .20 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .929181 | .677358 | -.251824 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_020_04` | concrete | .20 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .953887 | .669198 | -.284689 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_020_05` | concrete | .20 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .941821 | .682869 | -.258952 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_020_06` | concrete | .20 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .942699 | .673478 | -.269221 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_025_01` | concrete | .25 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .956331 | .682263 | -.274068 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_025_02` | concrete | .25 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .995560 | .841086 | -.154474 | 1 | 0 | no | no |
| `sfcocr_v_sml_c_025_04` | concrete | .25 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .956331 | .682263 | -.274068 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_025_05` | concrete | .25 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .805411 | .682263 | -.123149 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_025_06` | concrete | .25 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .973970 | .858315 | -.115655 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_030_01` | concrete | .30 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .954569 | .800114 | -.154455 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_030_02` | concrete | .30 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .986343 | .655123 | -.331220 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_030_03` | concrete | .30 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .912791 | .672847 | -.239944 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_030_04` | concrete | .30 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .704605 | .594878 | -.109727 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_030_05` | concrete | .30 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .704605 | .594878 | -.109727 | 0 | 0 | no | no |
| `sfcocr_v_sml_c_030_06` | concrete | .30 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .704605 | .594878 | -.109727 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_020_01` | marble | .20 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .993649 | .675029 | -.318620 | 3 | 0 | no | no |
| `sfcocr_v_sml_m_020_02` | marble | .20 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .994617 | .739124 | -.255493 | 3 | 0 | no | no |
| `sfcocr_v_sml_m_020_03` | marble | .20 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .988938 | .661181 | -.327758 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_020_04` | marble | .20 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .832427 | .737495 | -.094932 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_020_05` | marble | .20 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .973663 | .743555 | -.230108 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_020_06` | marble | .20 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .973320 | .737495 | -.235825 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_025_01` | marble | .25 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .994623 | .928337 | -.066286 | 5 | 0 | yes | no |
| `sfcocr_v_sml_m_025_02` | marble | .25 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .993464 | .926676 | -.066788 | 7 | 0 | yes | no |
| `sfcocr_v_sml_m_025_03` | marble | .25 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .983406 | .961164 | -.022242 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_025_04` | marble | .25 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .752150 | .601549 | -.150602 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_025_05` | marble | .25 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .748156 | .644587 | -.103568 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_025_06` | marble | .25 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .748127 | .625091 | -.123036 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_030_01` | marble | .30 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .686745 | .636461 | -.050284 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_030_02` | marble | .30 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .705801 | .608270 | -.097531 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_030_03` | marble | .30 | LOW | transition-left | RIGHT_SINGLE_SUPPORT | ADVERSE | .820206 | .621293 | -.198913 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_030_04` | marble | .30 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .834263 | .657659 | -.176604 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_030_05` | marble | .30 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .834263 | .657659 | -.176604 | 0 | 0 | no | no |
| `sfcocr_v_sml_m_030_06` | marble | .30 | LOW | transition-right | LEFT_SINGLE_SUPPORT | COMPARISON | .834263 | .657659 | -.176604 | 0 | 0 | no | no |

## 20. Counters, artifacts, and verification

| Counter | Actual |
|---|---:|
| new simulation / pilot runs | 0 / 0 |
| candidate families / seeds trained | 1 / 3 |
| optimizer steps / checkpoint writes | 50,259 / 12 |
| normalizer fits / HNM rounds | 0 / 3 |
| threshold / persistence / architecture / seed searches | 0 / 0 / 0 / 0 |
| sensor fusion / V1 inference | 0 / 0 |
| reference / candidate FACTOR_VALIDATION inference | 1 split / 1 split |
| FACTOR_VALIDATION opens | 1 |
| old HOLDOUT reads / inference | 0 / 0 |
| Generalization VALIDATION candidate inference | 0 |

| Artifact | SHA-256 |
|---|---|
| Training config | `10b723270478b87fae9854cbf5a165386e8317d2dadef9a36f846c13d7218a22` |
| Training-source ledger | `86f25596db4481409d4b8eba48b45fc6a60ab145d1d077db6eeb4577a2e0e9f3` |
| Normalizer | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` |
| HNM provenance | `d345e61194d1d102be443f0f2e39bf8e8e965048e89075ecb464ccadfe5b6c0d` |
| Seed 20260828 final checkpoint | `58df8b1ceb4bd4ab9a3343506eefc83b49d2d98dd614250229b589def061b8ec` |
| Seed 20260829 final checkpoint | `c990b3f95a009e53ab878302192ba8b20e0971268900786413502c23584fe47e` |
| Seed 20260830 final checkpoint | `f5b1a9be27f1a5b84ef706b9da33227c90d434c2aac152fc1365cb876aea3837` |
| Candidate freeze | `27314d6d097c7daa3a2da66be1d017c4097787e276412670837046c9296029d6` |
| Validation authorization | `aa8876addaf6fe17524dd3ce73faf5607f6393f8893ad69f70301f6d12f2bf15` |
| Reference validation | `c3626071dc688b62dc670021316a32e4795386e44f395acfe6b8e491d9cdeafa` |
| Candidate validation | `480f5e7d0d3b07e37f149801faa77cea3be0597f37c0eb311a3825520dfab43e` |
| Paired strict-Sand ledger | `6f94e71018cae9974570ae96fafbb430e92cce1c76238f08fa36a862114a79f5` |
| Comparison | `e70edbf3ccd91af4d25406080503a56814bdd39e53f40b4d97c5d46068923dc7` |
| Historical regression | `0da1dd8c9e56790f0cd6cea4c3102c667150326d3f16ad3aa4ac16f46052a869` |
| Intervention decision | `8a5f07e6e01c32e745262271b49d975d4fcc7790953c6886e2de3925d6ef5fbd` |
| Failure interpretation | `e50b570c522227e78a8c5affd66ec96f614d87196c8334590b05ad443c1ad2b7` |
| Evaluation freeze | `1691babdc4b75af8e3bba8fba09c1e98d3116729f2f0a70a42f02163e1cfacd5` |
| Overall milestone result | `4cfc440830107e5e268c8f66ae21bd55294bcf43fbc31ea41364a6313887ca92` |

The read-only verifier passed the complete hash chain, all three final checkpoint hashes, the single-family/single-open assertions, and every zero-forbidden counter. The targeted pre-training contract suite passed 13 tests with the post-execution hash test skipped before execution. After execution, the complete safe suite passed 207 tests with one expected user-supplied-policy simulator smoke skip; compileall, configured full Ruff, explicit `E9,F63,F7,F82` Ruff, and `git diff --check` also passed. Final Git parity is recorded at handoff after documentation finalization.

## 21. Limitations and deployment parallelization

This is one simulator policy, one frozen factor-conditioned physical domain, one architecture family, three fixed seeds, one development validation, and one historical regression replay. The single actual Slip is only descriptive. The consumed factor validation cannot be reused adaptively, and the candidate has no external or final evidence.

E84 engineering remains compatible with the unchanged Pelvis-IMU6 `[20,80]`, GRU20, 0.99/5 ms interface and may continue using `model_v2_anchor_refined_gru20_20260902` only as `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`. The failed factor-conditioned candidate is not suitable for deployment-reference handoff. No E84 file was touched.
