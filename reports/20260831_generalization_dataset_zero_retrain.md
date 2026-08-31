# Generalization Dataset Generation and Zero-Retrain Evaluation

## 1. Purpose

이 milestone은 current frozen Hazard/Terrain candidate를 변경하지 않고, calibration pilot과 물리 signature가 겹치지 않는 fresh MuJoCo corpus에서 zero-retrain generalization을 검증했다. 사전 선언한 다섯 family의 physical corpus를 먼저 생성·감사·freeze한 뒤 `GENERALIZATION_VALIDATION`만 replay했다. Validation gate 실패에 따라 `GENERALIZATION_HOLDOUT`은 sealed 상태로 보존했다.

Physical dataset verdict와 model verdict는 분리한다.

- Dataset: `GENERALIZATION_DATASET_READY`
- Model: `ZERO_RETRAIN_GENERALIZATION_NOT_SUPPORTED`

## 2. Protected baseline

Starting HEAD와 `origin/main`은 모두 `fd4a6981ae2c50d0f5ebe596c68f69dcbae45269` (`Resolve Ice generalization gaps`)였고 tracked worktree는 clean이었다.

Hazard는 freeze `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`, Pelvis IMU6, causal 80D, `[20,80]`, GRU hidden 32/one layer/three seeds, 11,010 parameters, threshold `0.99`, persistence `5 ms` 그대로다. Terrain은 FSR4, `[50,4]`, MLP/50 ms/three seeds/left-only/advisory-only 그대로다. Hazard normalizer/checkpoints와 Terrain normalizer/checkpoints는 read-only hash verification을 통과했다.

## 3. Frozen scenario families

사전 선언 family는 다음 다섯 개뿐이다.

| Family | Designed role | Runs | Split allocation |
|---|---|---:|---:|
| `ONE_CONTACT_DELAYED_ICE_SLIP` | complete benign Ice episode 뒤 distinct Slip | 12 | 6 / 6 |
| `ICE_BENIGN_CONTROL` | clean primary no-hazard Ice | 8 | 4 / 4 |
| `DELAYED_SAND_SUPPORT_ONSET` | benign interval 뒤 delayed left Support | 8 | 4 / 4 |
| `RIGHT_SAND_SUPPORT` | right-only Support | 8 | 4 / 4 |
| `SPEED_STRATIFIED_HAZARD` | 0.20/0.25/0.30 m/s Slip, Support, Sand benign | 36 | 18 / 18 |

첫 split 수는 `GENERALIZATION_VALIDATION`, 둘째는 `GENERALIZATION_HOLDOUT`이다. TRAIN split은 없다.

## 4. Fresh signature exclusions

Frozen physical signature fields는 source/target terrain, speed, patch start/width, slip/sink pattern, severity, support pattern이다.

| Reference | Reference signatures | Fresh overlap |
|---|---:|---:|
| Unified corpus | 256 | 0 |
| Generalization calibration pilots | 78 | 0 |
| Ice gap-resolution pilots | 48 | 0 |
| Historical pre-unified exclusions | 447 | 0 |
| Fresh internal duplicates | 72 designed | 0 |
| Validation/Holdout overlap | 36 / 36 | 0 |

Pilot waveform나 pilot model smoke result는 이번 denominator에 포함하지 않았다.

## 5. Pre-simulation design matrix

Exact 72-run matrix는 simulation 전에 config와 `run_matrix_freeze.json`에 기록했다. Matrix artifact SHA-256은 `b6790bdca8d664139fcbd8024fd08610c3a8892a61425b3aa37ac90af29ac68b`다. Scenario, geometry, split, run ID는 outcome을 보기 전에 확정했으며 adaptive backfill과 replacement run은 0이다.

Ice delayed와 benign은 이전 resolution의 bounded local cells를 사용했다. Delayed Sand는 approved opt-in `staged_lateral_deformable`, right Sand는 `transition_right`/`lateral_deformable`, speed family는 각 speed에서 Ice Slip/Sand Support/Sand benign을 Concrete/Marble에 배치했다. Material constant나 default simulator behavior는 변경하지 않았다.

## 6. Split freeze

- Split membership SHA-256: `929c963dff786d74728e1aad4be2457fe1dc6b4ef667fdddb1e5c01825822787`
- Physical signatures SHA-256: `e835ac8c7bc51b5a0925b8ebba1eb7d4e0b29b3015cc6b0bba16cf059f159553`
- Validation: 36
- Holdout: 36
- TRAIN: 0
- Split frozen before simulation: YES
- Outcome/model output used for membership: NO

## 7. Dataset generation

Dataset ID는 `generalization_hazard_reflex_20260831`이다. 72개 run을 current simulator/policy로 8 s, 1 kHz, headless하게 생성했다. Generation 순서는 scenario specification → MuJoCo → physical signals → physical label/diagnostic → NPZ였고, 이때 model output은 load하지 않았다.

| Family | Designed | Valid | Actual Hazard | Actual No Hazard | Validation | Holdout |
|---|---:|---:|---:|---:|---:|---:|
| One-contact delayed Ice Slip | 12 | 12 | 12 | 0 | 6 | 6 |
| Ice benign control | 8 | 8 | 2 | 6 | 4 | 4 |
| Delayed Sand Support | 8 | 8 | 8 | 0 | 4 | 4 |
| Right Sand Support | 8 | 8 | 8 | 0 | 4 | 4 |
| Speed-stratified | 36 | 36 | 24 | 12 | 18 | 18 |
| **Total** | **72** | **72** | **54** | **18** | **36** | **36** |

Design intent와 다른 두 Ice-benign HOLDOUT run은 삭제하지 않았다. 두 run 모두 actual `SLIP_HAZARD`로 기록했다. 따라서 전체 physical label은 Slip 26, Support 28, no-hazard 18이다.

## 8. Dataset integrity

Manifest와 모든 72 NPZ의 SHA-256을 다시 계산했다. Runtime tensor는 모두 `timestamp_us int64[8000]`, `pelvis_imu6 float32[8000,6]`, `foot_fsr8 float32[8000,8]`이며 1,000 us cadence, finite input, nonnegative FSR, sensor drop 0을 확인했다. Invalid/rejected run은 0이다.

Manifest는 source commit과 immutable experiment config SHA를 기록하고, 그 config가 policy SHA `2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`, simulator config/source SHA, mechanics source SHA를 고정한다. Dataset integrity artifact SHA-256은 `5aad40fc06e0afd6a08205f27175924de6256bda4228067ddd9ea467adf26701`이다.

## 9. Physical outcome audit

Primary label은 established Slip OR established Support다. Primary no-hazard는 Slip, I1, Support가 모두 없는 경우에만 부여했다. Terrain, fall, intent는 label이 아니다.

- Delayed Ice: 12/12 Slip, exactly-one 4, multi-contact 8, bilateral 12. Contact→Slip은 688–1,128 ms, median 828.5 ms였다.
- Ice benign: 6/8 primary no-hazard, 2/8 accidental Slip. Clean 50 ms target opportunities는 run당 1–3개였다. Physical success rate는 75%다. Validation은 4/4 no-hazard이고 Holdout은 2/4 no-hazard다.
- Delayed Sand: 8/8 left-only Support. 모든 run에 I1 전 clean target touchdown 2개가 있고 contact→I1 1,791–1,792 ms, I1→Support 56 ms다.
- Right Sand: 8/8 right-only Support. Contact→I1 19 ms, I1→Support 647–648 ms다.
- Speed family: 각 0.20/0.25/0.30 m/s에서 source 두 개씩 Slip 2, Support 2, no-hazard 2를 확보했다.

Fall은 Hazard label이 아니며 censor에만 사용했다. 모든 run은 target encounter 전까지 valid였다.

## 10. Family coverage

각 split별 사전 readiness minimum을 독립적으로 통과했다.

| Family | Validation physical coverage | Holdout physical coverage | Ready |
|---|---|---|---|
| Delayed Ice | viable 6; sources 2; speeds 3 | viable 6; sources 2; speeds 3 | YES |
| Ice benign | no-hazard 4; sources 2 | no-hazard 2; sources 2; accidental Slip 2 retained | YES |
| Delayed Sand | delayed left Support 4; sources 2 | delayed left Support 4; sources 2 | YES |
| Right Sand | right-only Support 4; sources 2 | right-only Support 4; sources 2 | YES |
| Speed | every speed/subtype ≥2 across both sources | same | YES |

Physical readiness artifact SHA-256은 `b596a5738e662e47055e375aba2129ae3140e324aee2ffbf7c08b36c84993a24`다.

## 11. Dataset freeze

Model load 전 `MODEL_BLIND_DATASET_FREEZE_COMPLETE`를 기록했다.

- Config SHA-256: `68c92b9dd7615b8cce6dff59315e51577c2cab75575fb4e37b1588d1ec1758aa`
- Manifest file SHA-256: `72f5dd300bb5be78b22b9a8c1ad9788cdf45e7cda9b4f46d79eb290140146f53`
- Manifest canonical SHA-256: `a9fa921c0e1ee125cfa314d156a19adae6dd58d3fcc876c5f8e4db310abd0f0d`
- Dataset freeze SHA-256: `43d5072dc7b2fed001e76b8d47bfde12934b0696a1a42a096f67e89f92175ae6`
- Model outputs loaded before freeze: NO
- Generalization HOLDOUT opened before freeze: NO
- Current Unified HOLDOUT opened: NO

## 12. Zero-retrain protocol

Freeze 뒤 current Hazard GRU와 current Terrain MLP만 load했다. Hazard replay는 canonical `evaluation/hazard.py`, Terrain replay는 canonical `evaluation/terrain.py`의 left-only scheme을 사용했다. Hazard tensor는 Pelvis IMU6에서 생성한 80D뿐이며 FSR/Terrain/physical clock은 들어가지 않았다.

Retraining, optimizer step, checkpoint write, normalizer refit, HNM, threshold/persistence/architecture search는 모두 0이다. Calibration pilot은 평가 denominator에서 제외했다.

## 13. Generalization VALIDATION

36개 Validation waveform은 dataset freeze 후 열었다. Physical 구성은 Hazard 26, primary no-hazard 10이다.

| Metric | Result | Gate | Pass |
|---|---:|---:|---|
| Overall Hazard recall | 13/26 = 50.00% | ≥90% | NO |
| Slip recall | 7/12 = 58.33% | ≥95% | NO |
| Support recall | 6/14 = 42.86% | ≥85% | NO |
| Primary no-hazard specificity | 5/10 = 50.00% | ≥95% | NO |
| Ice-benign specificity | 3/4 = 75.00% | ≥95% | NO |
| Premature hazard-run rate | 7/26 = 26.92% | ≤10% | NO |
| Slip valid latency p95 | +5.3 ms | ≤+40 ms | YES |
| Support established latency p95 | -17.25 ms | ≤+50 ms | YES |

Slip valid latency는 min -25, median -24, p95 +5.3, max +11 ms였다. Support valid detection 6개의 established-relative latency는 min -559, median -20, p95 -17.25, max -17 ms였고 I1-relative latency는 626–1,231 ms였다. Timing 통계는 valid detection만 포함하므로 낮은 recall을 상쇄하지 않는다.

| Family | Eligible runs | Detected correctly | Recall/Specificity | Premature | Timing |
|---|---:|---:|---:|---:|---|
| Delayed Ice Slip | 6 hazard | 3 | recall 50.00% | 3 | valid Slip latency median -24 ms |
| Ice benign | 4 no-hazard | 3 | specificity 75.00% | — | false reflex contact +966 ms |
| Delayed Sand Support | 4 hazard | 0 | recall 0% | 4 | first reflex I1 -553…-550 ms |
| Right Sand Support | 4 hazard | 0 | recall 0% | 0 | no Reflex onset |
| Speed Slip | 6 hazard | 4 | recall 66.67% | 0 | valid latency median -16 ms |
| Speed Support | 6 hazard | 6 | recall 100% | 0 | established latency median -20 ms |
| Speed Sand benign | 6 no-hazard | 2 | specificity 33.33% | — | four late false reflexes |

## 14. Ice benign specificity

Fresh physical no-hazard Ice controls은 4개였고 false reflex는 1개다.

| Run | Source | Ice contact | Clean Ice touchdown | First Terrain ICE | First Reflex | Max target drift | Margin to 50 mm | Contact phase | Fall/censor |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `ghr_ibc_v_c020` | Concrete | 1504 | 2393 | 2443 | 2470 | 43.569 mm | 6.431 mm below | left support, left on Ice | 2673/2673 |

Reflex는 first correct Terrain ICE state 27 ms 뒤 발생했다. Established Slip, I1, Support는 없었으므로 이는 genuine primary false reflex다. Calibration pilot 결과를 denominator에 넣지 않았고 원인 수정·threshold 변경은 하지 않았다.

## 15. Delayed Ice Slip

Validation 6개 모두 actual bilateral Slip이며 contact→Slip은 min 688, median 828.5, p95 1,091, max 1,128 ms였다.

| Episode class | Runs | Correct | Recall | Premature | Reflex during prior benign episode |
|---|---:|---:|---:|---:|---:|
| Exactly one benign episode | 2 | 2 | 100% | 0 | 0 |
| Multiple benign episodes | 4 | 1 | 25% | 3 | 3 |

전체 recall은 3/6이다. Multi-contact 세 run에서 first reflex가 later Slip valid region보다 일찍, prior benign target episode 중 발생해 canonical premature 처리되었다. Valid run의 Reflex→Slip lead는 17–25 ms였다. Actual side는 6/6 bilateral이고 left-only/right-only Slip은 없었다.

## 16. Delayed Sand Support

Physical order는 네 run 모두 target contact (1220) → target Terrain state (1851–1852) → first Reflex (2459–2461) → I1 (3011–3012) → Support (3067–3068)였다. I1 전 clean target touchdown은 각 2개다.

First Reflex가 I1보다 550–553 ms 빨라 4/4 pre-I1 premature가 되었고, canonical rule상 later onset으로 rescue하지 않아 recall은 0/4다. Terrain→I1은 1,160 ms, Terrain→Reflex는 607–610 ms, Reflex→Support는 606–609 ms였다.

## 17. Right Sand Support

Validation 4/4가 actual right-only Support였으나 Hazard onset은 모두 없어서 recall 0/4다. Premature도 0이다. Physical contact→I1은 19 ms, contact→Support는 666–667 ms다.

Left-only Terrain producer에는 clean target-Sand event가 0개였다. Held SAND state가 나타난 경우도 모두 established Support 뒤였으므로 target Terrain은 event 시점에 0/4 available이었다. Terrain이 unavailable/wrong이어도 Hazard는 독립이어야 하지만, 이 subset에서 성공한 Hazard detection은 0개다.

## 18. Speed robustness

각 cell은 Concrete/Marble 한 run씩 포함한다.

| Speed (m/s) | Hazard type | Source | Runs | Recall | Specificity | Premature |
|---:|---|---|---:|---:|---:|---:|
| 0.20 | Ice Slip | C + M | 2 | 50% | — | 0 |
| 0.20 | Sand Support | C + M | 2 | 100% | — | 0 |
| 0.20 | Sand benign | C + M | 2 | — | 0% | — |
| 0.25 | Ice Slip | C + M | 2 | 100% | — | 0 |
| 0.25 | Sand Support | C + M | 2 | 100% | — | 0 |
| 0.25 | Sand benign | C + M | 2 | — | 50% | — |
| 0.30 | Ice Slip | C + M | 2 | 50% | — | 0 |
| 0.30 | Sand Support | C + M | 2 | 100% | — | 0 |
| 0.30 | Sand benign | C + M | 2 | — | 50% | — |

Speed family Hazard recall은 10/12, no-hazard specificity는 2/6이다. Source별 전체 corpus 결과는 Concrete Hazard 6/13/no-hazard 3/5, Marble Hazard 7/13/no-hazard 2/5였다.

## 19. Affected-foot breakdown

Affected foot는 physical diagnostic metadata이며 Pelvis IMU-only Hazard input이 아니다.

| Physical side | Runs | Hazard runs | Correct | Recall |
|---|---:|---:|---:|---:|
| Bilateral | 12 | 12 | 7 | 58.33% |
| Left-only | 10 | 10 | 6 | 60.00% |
| Right-only | 4 | 4 | 0 | 0% |
| None | 10 | 0 | — | specificity 50.00% |

Right-only Support recall은 0/4, bilateral Slip recall은 7/12다. Naturally occurring left-only Slip은 0개였다.

## 20. Terrain advisory behavior

Terrain은 primary gate가 아니다. Left-only producer에서 clean prediction event는 331개, overall event accuracy는 254/331 = 76.74%였다. Evaluable target events는 171개이고 target accuracy는 153/171 = 89.47%다. Correct target state unavailable run은 2/36 (5.56%)이며 first target state after contact는 min 50, median 640.5, p95 1,489.3, max 1,503 ms였다.

| Breakdown | Runs/events | Target events | Target accuracy | Target-state runs |
|---|---:|---:|---:|---:|
| Concrete source | 18 / 171 | 85 | 90.59% | 17/18 |
| Marble source | 18 / 160 | 86 | 88.37% | 17/18 |
| Ice benign | 4 / 14 | 4 | 50.00% | 2/4 |
| Delayed Ice | 6 / 70 | 48 | 89.58% | 6/6 |
| Delayed Sand | 4 / 44 | 21 | 90.48% | 4/4 |
| Right Support | 4 / 38 | 0 | not evaluable | 0/4 by event |

전체 run order는 Terrain-before-Reflex 25, Reflex-before-Terrain 9, Terrain-unavailable 2다. Missing/wrong Terrain은 Hazard gate나 miss 정의에 사용하지 않았다.

## 21. Temporal-order analysis

| Family | Target contact | Terrain valid | First Reflex | I1 | Physical event | Observed order |
|---|---|---|---|---|---|---|
| Delayed Ice | 1220–1507 | 2127–2444 | 1883–2470 | — | Slip 1908–2632 | mixed; Reflex-before-Terrain 4/6 |
| Ice benign | 1227–1507 | 2443–2444 in 2/4 | 2470 in 1/4 | — | none | one false reflex after Terrain |
| Delayed Sand | 1220 | 1851–1852 | 2459–2461 | 3011–3012 | Support 3067–3068 | Terrain → Reflex → I1 → Support |
| Right Sand | 1507–1509 | 2991–3010, after event | none | 1526–1528 | Support 2173–2176 | I1 → Support → Terrain; no Reflex |
| Speed family | 1220–1810 | 1745–2446 | 1616–6062 when present | 1239–1829 for Support | 1596–3036 | subtype-dependent |

Hazard run 기준 Reflex-before-established-Hazard 19, Reflex-after-established-Hazard 3, Reflex unavailable 4였다. 이 순서는 correctness가 아니다. Pre-I1/Slip-window 규칙을 별도로 적용한 primary metric이 authoritative하다.

## 22. Validation verdict

Validation verdict는 `ZERO_RETRAIN_GENERALIZATION_VALIDATION_FAIL`이다. Eight primary gates 중 recall/specificity/premature 여섯 개가 실패했고, conditional timing gate 두 개만 통과했다. Decision artifact SHA-256은 `a369162b121ca6f8f0dacfdc5ccef92eec5b1fc5a06c0f1cbfde4a362ef095f2`다.

실패 뒤 threshold, persistence, scenario, split, label, model을 변경하지 않았고 validation run을 제거하지 않았다.

## 23. Generalization HOLDOUT access decision

Validation gate가 하나라도 실패하면 Holdout open을 금지한다는 사전 규칙을 적용했다.

- Generalization HOLDOUT waveform opened: NO
- Guard open count: 0
- Holdout model inference: NO
- `holdout_open_freeze.json`: absent
- `holdout_evaluation.json`: absent
- Retune/retrain/rescue: NO

## 24. Fresh HOLDOUT result

없다. Holdout physical metadata는 model-blind dataset readiness freeze에만 사용되었고 waveform/model output은 열지 않았다. Fresh Holdout metric을 추정하거나 Validation result로 대체하지 않는다.

## 25. Historical unified comparison

분모를 합치지 않는다.

| Evidence | Hazard | Slip | Support | No-hazard | Premature | Status |
|---|---:|---:|---:|---:|---:|---|
| Existing Unified fresh HOLDOUT | 26/26 | 13/13 | 13/13 | 26/26 | 0/26 | historical supported baseline |
| Fresh generalization Validation | 13/26 | 7/12 | 6/14 | 5/10 | 7/26 | failed gates |
| Fresh generalization HOLDOUT | sealed | sealed | sealed | sealed | sealed | not opened |

Existing Unified HOLDOUT waveform은 이번 milestone에서 재오픈하지 않았고 new inference도 0이다.

## 26. Failure-mode interpretation

결과는 GRU architecture 전체를 기각하는 증거가 아니라 predeclared family별 current frozen candidate의 generalization gap이다.

- Multi-contact Ice: prior benign Ice episode에서 early Reflex가 반복되어 3/4 premature였다.
- Ice benign: 1/4 false reflex가 fresh data에서 반복되었다. 이 run은 Slip threshold보다 6.431 mm 아래였다.
- Delayed Sand: 4/4 onset은 Support를 예고했지만 I1보다 약 551 ms 빨라 frozen benign interval을 침범했다.
- Right-only Support: 0/4로 pelvis-level observability/side asymmetry gap이 가장 직접적이다.
- Speed controls: Slip/Support보다 Sand benign specificity(0–50%)가 취약했다.

## 27. Limitations

이 결과는 predeclared MuJoCo mechanics와 좁은 local geometry domain에 한정된다. Ice benign은 calibration상 start 0.335 m 주변의 narrow-width domain이며 Holdout physical audit에서도 2/4 accidental Slip이 발생했다. Terrain은 left-only라 right-only target contact를 직접 관측하지 못한다. Speed는 세 discrete value이고 gait phase는 독립적으로 통제한 axis가 아니다. Real robot, E84 latency/resource, quantization, HIL, recovery를 검증하지 않았다.

Post-evaluation visualization selection은 model confidence를 사용하지 않고 validation의 lexicographic rule로 `ghr_dss_v_c300`, `ghr_ibc_v_c020`, `ghr_ocd_v_c020`, `ghr_rss_v_c300`, `ghr_ssh_is_v_c020`, `ghr_ssh_is_v_c030`을 freeze했다. GUI는 실행하지 않았고 generalization Holdout은 visualize하지 않았다.

## 28. Final verdict

Physical corpus는 integrity와 family coverage를 만족하므로 `GENERALIZATION_DATASET_READY`다.

Current frozen candidate는 fresh Validation gate를 통과하지 못했고 Holdout은 sealed이므로 final model verdict는 `ZERO_RETRAIN_GENERALIZATION_NOT_SUPPORTED`다. 이는 “frozen simulation-trained candidate가 이 predeclared fresh MuJoCo scenario family들에 zero retraining으로 generalize했다”는 주장을 지지하지 않는다.

Existing four representative Validation regression은 Ice Slip, Sand Support, Sand benign, hard normal 모두 timestamp/IMU/FSR/physical clock/frozen Hazard/frozen Terrain parity PASS였다. Regression artifact SHA-256은 `94dc9a9cfc737ef0fe777dfcdec81024dffb714f31522f56590d76b7fee6342c`다.

Repository verification은 pytest 71 passed/1 skipped, `compileall` PASS (`src`, `scripts`, `tests`), Ruff critical `E9/F63/F7/F82` PASS였다. Source code는 변경하지 않았다.

## 29. Recommended next step

별도 failure-mode audit에서 다음을 먼저 분석한다.

1. Multi-contact Ice와 Sand benign false-reflex temporal signature
2. Delayed Support의 pre-I1 Pelvis IMU response와 benign-interval 구분
3. Right-only Support의 pelvis observability
4. Speed/source dependency

그 audit 뒤에만 Dataset/Model v2 필요성을 결정한다. 이 milestone에서는 retraining, LSTM, history ablation, Research export, E84, quantization, Recovery, GUI 작업을 시작하지 않는다.
