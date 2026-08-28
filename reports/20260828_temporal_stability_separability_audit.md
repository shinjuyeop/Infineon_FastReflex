# Temporal Walking Stability Separability Audit

Milestone: `TEMPORAL_STABILITY_SEPARABILITY_AUDIT`

## 1. Purpose

이 audit의 목적은 새로운 scalar `t_instability`를 만드는 것이 아니라, current G1 policy에서 observed-stable trajectory와 eventual-fall trajectory가 actual fall 몇 ms 전부터 causal history만으로 reliably 구분되는지 측정하는 것이었다. Primary output은 `EARLIEST_RELIABLE_FALL_RISK_HORIZON`이다.

작업 시작 시 실제 `HEAD`와 `origin/main`은 모두 `54b2c1178b7392271b0b3b6d59f1c91bc3655c02`였고 worktree는 clean이었다. Cohort, split, representation, five offsets, three history windows, GRU, seeds와 gates는 [`20260828_temporal_stability_separability_audit.yaml`](../configs/experiment/20260828_temporal_stability_separability_audit.yaml)에 simulation 전에 고정했다. Config SHA-256은 `6bae3ea8a7374888f1a9043c66bc215f90f37c7e0b4e84c5f2512f2bc128183b`이다.

## 2. Why scalar/single-state clocks were retired

Historical phase-aware XCoM/MoS clock은 fresh stable FP `20.83%`, fall coverage `18.75%`, Ice `42.86%`, Sand `0%`로 `WALKING_STABILITY_GROUND_TRUTH_NEEDS_REVISION`이었다. Stable-only Mahalanobis single-state study도 Lower/Full stable FP `25%`, fall coverage `56.25%`, median lead `35/28 ms`로 세 candidate가 calibration gate를 통과하지 못해 `FULL_STATE_STABILITY_GROUND_TRUTH_NOT_SEPARABLE`이었다.

두 결과는 수정하거나 삭제하지 않았다. 이번 audit은 instantaneous abnormality threshold를 확장하지 않고 recent trajectory가 future fall outcome을 예측할 수 있는지만 별도로 검증했다.

## 3. Fall-risk prediction semantics

Observed fall과 `t_fall`은 supervised label/alignment에 사용했다. Candidate input에는 fall, fall time, time-to-fall, intended role, run/scenario ID, terrain identity, Slip/Sink, deformation, patch geometry와 future/post-fall sample이 없다.

각 positive input은 endpoint `t < t_fall`에서 `[t-history+1, t]`만 포함한다. 이 milestone의 timing quantity는 detector latency가 아니라 `fall prediction lead`, 즉 reliable classification이 가능했던 pre-fall offset이다. Runtime enum `UNSTABLE`은 변경하지 않았으며 `STABLE/FALL_RISK`는 향후 naming recommendation일 뿐이다.

## 4. Scenario cohort

Existing clean transition 30개만으로는 transition stable 14 / fall 16이라 목표에 부족했다. Calibration/prior MoS validation과 physical signature가 겹치지 않도록 이미 사전 설계됐지만 이전 milestone에서 실행하지 않았던 fresh 48개를 추가했다. Primary 78개와 secondary hard-stable control 6개, 총 84 simulations을 실행했다.

| Cohort | Stable | Fall | Total |
|---|---:|---:|---:|
| Primary transitions | 42 | 36 | 78 |
| Hard stable controls | 6 | 0 | 6 |
| Ice primary | 20 | 18 | 38 |
| Sand primary | 22 | 18 | 40 |
| Concrete-origin primary | 21 | 18 | 39 |
| Marble-origin primary | 21 | 18 | 39 |

Invalid run, missing target contact, pre-transition fall과 nonfinite simulation은 0이었다. Four fresh fall-design conditions(`fs_val_c_sand_f05/f06`, `fs_val_m_ice_f05/f06`)은 실제 stable이었다. Intended role을 label로 강제하거나 split을 재배치하지 않았다.

## 5. Stable/fall matching

Primary fixed-offset analysis는 split/offset별로 source terrain과 target terrain을 exact match한 뒤 speed nearest, exact time-since-contact, endpoint support phase, stable run ID 순으로 one-to-one negative를 선택했다. Stable endpoint는 fall endpoint의 exact post-contact elapsed time을 stable contact에 더했고 invalid endpoint를 임의로 앞당기지 않았다.

Train은 각 offset에서 22 pairs/44 independent runs, validation은 8 pairs/16 independent runs이었다. Train Ice strata는 fall 6개 대 stable 5개여서 source별 fall 1개씩 총 2개를 deterministic unmatched exclusion했다. One run × one offset은 최대 one window다.

Sand speed는 exact match됐지만 Ice stable/fall calibrated domain은 각각 `0.25/0.15 m/s`라 0.10 m/s difference가 남는다. 이 speed shortcut 가능성은 결과 해석의 limitation이다.

## 6. Split

Split은 simulation/window construction 전에 calibrated design strata만으로 고정했다.

| Split | Configured | Observed stable | Observed fall | Use |
|---|---:|---:|---:|---|
| Train | 46 | 22 | 24 | normalization + model fit |
| Validation | 16 | 8 | 8 | history/horizon selection |
| Holdout | 16 | 12 | 4 | sealed one-shot only after reliable validation selection |

Train/validation/holdout run overlap은 0이다. Holdout은 모두 fresh conditions다. Observed flips 때문에 holdout fall은 Concrete→Ice 2개와 Marble→Sand 2개에 집중됐지만, 이 사실을 보고 split을 수정하지 않았다.

## 7. Privileged representation

`PRIVILEGED_FULL_STATE`는 40 channels다. 이전 37-D FULL_STATE feature order(pelvis roll/pitch, angular/linear velocity, height, 12 lower-body q/dq, COM velocity xyz, COM height above support)를 재사용하고 current exact support phase Left/Right/Double one-hot 3개를 붙였다. No-support COM height는 zero-before-first-valid + causal last-valid fill만 사용했다.

Absolute yaw와 MoS는 제외했다. q/dq 사용은 upper-bound observability를 위한 simulator privilege이며 runtime sensor 추가가 아니다. GRU parameter count는 7,170이다.

## 8. IMU6 representation

`RUNTIME_IMU6`는 Pelvis runtime trace의 raw `accel_x/y/z + gyro_x/y/z` 6 channels, 1 kHz다. Terrain, fall alignment와 physical diagnostics는 tensor에 없다. GRU parameter count는 3,906이다.

Privileged와 IMU6는 각자 train-only channel mean/std를 fit했다. 각 train run의 target-contact 이후 pre-fall 또는 stable-run-end interval을 evenly-spaced 2,048 samples로 cap했고 representation별 sample count는 87,990, fit run은 46개였다. Validation/holdout sample은 normalization에 쓰지 않았다.

## 9. History windows

Exactly `50/100/200 ms` histories를 비교했다. 모든 window의 마지막 sample은 current endpoint이고 future smoothing/centered filtering은 없다. 동일 pair/offset을 세 history와 두 representation에 사용했다.

History selection rule은 validation에서 reliable horizon을 fall에서 가장 멀리 확보한 candidate를 찾고, near-tie이면 가장 짧은 history를 선택하는 것이었다. 결과를 보고 500 ms history나 다른 window를 추가하지 않았다.

## 10. Pre-fall offsets

Primary Analysis B는 exactly `-500/-300/-200/-100/-50 ms` endpoint의 fall window와 exact elapsed-time matched stable window를 직접 비교했다. Positive history는 전부 strictly pre-fall이고 primary denominator는 B-target contact 이후만 포함한다.

Secondary Analysis A는 같은 horizon boundary score에 더해 `H+200 ms` early-fall window가 horizon label 0으로 유지되는지 모든 30 candidate에서 audit했다. 이는 selection에 영향을 주지 않았다. 20 ms offset은 결과-driven rescue로 사용하지 않았다.

## 11. Training protocol

양 representation에 동일한 existing one-layer unidirectional GRU(hidden 32, bidirectional false, dropout 0), Adam `1e-3`, batch 32, max 30 epochs, patience 5를 사용했다. Seeds는 `20260828/20260829/20260830`이고 primary probability는 three-seed mean, frozen decision threshold는 0.5다. CNN/LSTM/MLP 추가, hyperparameter search와 threshold retuning은 하지 않았다.

Metrics는 dependency-light deterministic implementation의 tie-aware Mann–Whitney AUROC와 average-precision step AUPRC, macro F1, balanced accuracy, fall recall, stable specificity다. Independent count는 overlapping window 수가 아니라 run 수다.

## 12. Privileged separability curve

각 row는 validation 8 stable + 8 fall runs다.

| History | Offset | AUROC | AUPRC | Macro F1 | Balanced acc. | Fall recall | Stable spec. | Reliable |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 50 | 500 | .8594 | .8478 | .5636 | .6250 | 1.000 | .250 | N |
| 50 | 300 | .7031 | .7560 | .5466 | .5625 | .750 | .375 | N |
| 50 | 200 | .7500 | .8396 | .6190 | .6250 | .750 | .500 | N |
| 50 | 100 | .8750 | .8976 | .7460 | .7500 | .875 | .625 | N |
| 50 | 50 | .8594 | .8921 | .8118 | .8125 | .875 | .750 | N |
| 100 | 500 | .8594 | .8478 | .5636 | .6250 | 1.000 | .250 | N |
| 100 | 300 | .7031 | .7560 | .5466 | .5625 | .750 | .375 | N |
| 100 | 200 | .7344 | .8354 | .6190 | .6250 | .750 | .500 | N |
| 100 | 100 | .8750 | .8976 | .7460 | .7500 | .875 | .625 | N |
| 100 | 50 | .8594 | .8921 | .8118 | .8125 | .875 | .750 | N |
| 200 | 500 | .8594 | .8478 | .5636 | .6250 | 1.000 | .250 | N |
| 200 | 300 | .7031 | .7560 | .5466 | .5625 | .750 | .375 | N |
| 200 | 200 | .7344 | .8354 | .6190 | .6250 | .750 | .500 | N |
| 200 | 100 | .8750 | .8976 | .7460 | .7500 | .875 | .625 | N |
| 200 | 50 | .8594 | .8921 | .8118 | .8125 | .875 | .750 | N |

Validation gate는 AUROC ≥.90, AUPRC ≥.85, balanced accuracy/stable specificity/fall recall 각각 ≥.85 모두다. Highest AUROC diagnostic은 50 ms history/-100 ms였지만 AUROC, balanced accuracy와 specificity를 실패했다. -50 ms는 thresholded metrics가 가장 높았지만 AUROC .8594와 specificity .75였다. 더 긴 history는 early separation을 개선하지 않았다.

## 13. IMU separability curve

| History | Offset | AUROC | AUPRC | Macro F1 | Balanced acc. | Fall recall | Stable spec. | Reliable |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 50 | 500 | .8125 | .8097 | .7333 | .7500 | 1.000 | .500 | N |
| 50 | 300 | .3281 | .4342 | .4921 | .5000 | .625 | .375 | N |
| 50 | 200 | .7031 | .7979 | .6863 | .6875 | .625 | .750 | N |
| 50 | 100 | .5781 | .7278 | .6761 | .6875 | .500 | .875 | N |
| 50 | 50 | .8438 | .8757 | .6863 | .6875 | .750 | .625 | N |
| 100 | 500 | .8125 | .8097 | .7333 | .7500 | 1.000 | .500 | N |
| 100 | 300 | .3281 | .4342 | .4921 | .5000 | .625 | .375 | N |
| 100 | 200 | .7031 | .7979 | .6863 | .6875 | .625 | .750 | N |
| 100 | 100 | .5781 | .7278 | .6761 | .6875 | .500 | .875 | N |
| 100 | 50 | .8438 | .8757 | .6863 | .6875 | .750 | .625 | N |
| 200 | 500 | .8125 | .8097 | .7333 | .7500 | 1.000 | .500 | N |
| 200 | 300 | .3281 | .4342 | .4921 | .5000 | .625 | .375 | N |
| 200 | 200 | .7031 | .7979 | .6863 | .6875 | .625 | .750 | N |
| 200 | 100 | .5781 | .7278 | .6761 | .6875 | .500 | .875 | N |
| 200 | 50 | .8438 | .8757 | .6863 | .6875 | .750 | .625 | N |

Highest AUROC diagnostic은 50 ms history/-50 ms였다. AUPRC는 통과했지만 AUROC, balanced accuracy, fall recall과 stable specificity가 모두 gate 아래였다. 50→200 ms history 증가가 metrics를 개선하지 않았다.

## 14. Earliest reliable horizon

| Representation | Validation reliable horizon | Diagnostic best | Interpretation |
|---|---|---|---|
| Privileged full state | None | 50 ms history / -100 ms, AUROC .875 | Even the upper-bound state did not reliably separate |
| Pelvis IMU6 | None | 50 ms history / -50 ms, AUROC .8438 | No supported runtime fall-risk horizon |

따라서 `EARLIEST_RELIABLE_FALL_RISK_HORIZON`은 두 representation 모두 `NOT_ESTABLISHED`다. Privileged가 50 ms에서도 formal gate를 통과하지 못했으므로 physical early separability evidence가 weak하다는 Case C/D pattern이다.

Secondary horizon-fixed early-negative specificity도 Privileged `.125–.625`, IMU6 `.375–.625` 범위였다. Boundary model이 horizon보다 200 ms 더 이른 fall trajectory도 자주 FALL_RISK로 분류해 reliable horizon semantics를 지지하지 못했다.

## 15. Holdout

Validation에서 reliable history/horizon이 하나도 선택되지 않았으므로 holdout guard는 열리지 않았다(`open_count=0`). Holdout waveform construction, model evaluation, threshold/window/horizon reselection은 모두 수행하지 않았다.

따라서 holdout AUROC/accuracy/recall/specificity는 `N/A`다. 이는 failed validation candidate를 holdout에 맞춰 구조하는 것을 방지하는 사전 계약의 결과다.

## 16. Ice/Sand breakdown

아래는 authoritative selection이 아닌 diagnostic-best validation candidate의 breakdown이며 각 terrain support는 4 stable + 4 fall이다.

| Representation / offset | Terrain | AUROC | Fall recall | Stable specificity |
|---|---|---:|---:|---:|
| Privileged 50 ms / -100 ms | Ice | .8125 | .750 | .500 |
| Privileged 50 ms / -100 ms | Sand | 1.000 | 1.000 | .750 |
| IMU6 50 ms / -50 ms | Ice | 1.000 | .750 | 1.000 |
| IMU6 50 ms / -50 ms | Sand | .6875 | .750 | .250 |

Sand는 privileged trajectory에서 ranking separation은 강했지만 threshold specificity가 gate 아래였다. IMU6에서는 Sand AUROC와 specificity가 다시 크게 저하됐다. 어느 breakdown도 overall reliability failure를 구제하지 않는다.

## 17. Concrete/Marble source robustness

| Representation / offset | Source | AUROC | Fall recall | Stable specificity |
|---|---|---:|---:|---:|
| Privileged 50 ms / -100 ms | Concrete | 1.000 | 1.000 | 1.000 |
| Privileged 50 ms / -100 ms | Marble | .7500 | .750 | .250 |
| IMU6 50 ms / -50 ms | Concrete | .8125 | .750 | .750 |
| IMU6 50 ms / -50 ms | Marble | .8750 | .750 | .500 |

Privileged diagnostic의 apparent Concrete separation은 Marble에 generalize되지 않았다. IMU6도 두 source 모두 fall recall과 specificity gate를 만족하지 못했다.

## 18. Stable Slip hard negatives

Ice stable 20/20과 Ice fall 18/18 모두 physical Slip diagnostic이 있었다. Diagnostic-best validation에서 privileged는 Ice stable 2/4를 FALL_RISK로 오인했고, IMU6는 Ice stable 0/4를 오인했지만 Ice fall recall은 3/4였다.

즉 privileged false positives는 recoverable Slip trajectory를 완전히 흡수하지 못했고, IMU6는 stable Slip specificity가 높아도 fall recall이 부족했다. Slip occurrence 자체는 input도 label도 아니다.

## 19. Stable Sand-deformation hard negatives

Sand stable 22개는 deformation `20.14–40.14 mm`를 보였고, outcome-flip stable 2개는 Sink diagnostic까지 있었다. Sand fall 18개는 Sink 18/18, deformation `40.09–40.22 mm`였다.

Diagnostic-best validation에서 privileged는 Sand stable 1/4, IMU6는 3/4를 FALL_RISK로 오인했다. IMU6 Sand specificity .25는 deformation/recoverable motion과 fall trajectory를 minimal sensor에서 분리하지 못한 핵심 failure다.

Unmatched hard-ground controls 6개를 같은 diagnostic model/elapsed endpoint로 audit했을 때 privileged는 6/6 false positive, IMU6는 0/6 false positive였다. 이 control은 primary matching/selection denominator가 아닌 secondary diagnostic이다.

## 20. Representative temporal replay

Generated plots는 Gitignored artifact에만 저장했고 threshold를 변경하는 데 사용하지 않았다. 대표 physical timelines:

| Pair | Stable timeline | Fall timeline |
|---|---|---|
| Concrete→Ice | contact 1221 ms → Slip 2190 ms → non-fall | contact 1803 ms → Slip 3088 ms → fall 5133 ms |
| Concrete→Sand | contact 1508 ms → deformation 20.14 mm → non-fall | contact 1221 ms → Sink 2476 ms → deformation 40.17 mm → fall 5567 ms |

50 ms history로 각 offset에 별도 학습한 model의 representative `(fall / matched-stable)` probability는 다음과 같다.

| Representation | Offset | Ice fall/stable | Sand fall/stable |
|---|---:|---:|---:|
| Privileged | 500 | .700 / .660 | .578 / .499 |
| Privileged | 300 | .462 / .363 | .534 / .529 |
| Privileged | 200 | .507 / .432 | .555 / .537 |
| Privileged | 100 | .549 / .417 | .542 / .415 |
| Privileged | 50 | .681 / .384 | .542 / .446 |
| IMU6 | 500 | .530 / .489 | .528 / .515 |
| IMU6 | 300 | .515 / .568 | .505 / .511 |
| IMU6 | 200 | .627 / .468 | .500 / .500 |
| IMU6 | 100 | .575 / .442 | .453 / .500 |
| IMU6 | 50 | .683 / .374 | .435 / .478 |

대표 pair 자체에서는 일부 late separation이 보이지만 16-run validation 전체의 stable/fall gates를 만족하지 못한다. Four diagnostic plot의 physics mutation은 없으며 첫 execution과 final output-contract execution의 core metrics는 exact 재현됐다.

## 21. Historical oracle comparison

| Study | Primary result | Timing interpretation |
|---|---|---|
| Phase-aware MoS | FP 20.83%, fall coverage 18.75% | authoritative onset unsupported |
| Single-state Full distance | FP 25%, fall coverage 56.25%, median lead 28 ms | instantaneous onset not separable |
| Temporal privileged GRU | no validation-reliable offset; best AUROC .875 | early fall-risk horizon not established |
| Temporal IMU6 GRU | no validation-reliable offset; best AUROC .8438 | runtime horizon not established |

Temporal history는 single-state coverage question을 future-risk classification으로 바꿨지만 formal stable/fall separation을 만들지 못했다. 서로 다른 metric을 직접 동일 수치로 비교하지 않는다.

## 22. Fusion implications

Conceptually 향후 validated `FALL_RISK`가 존재한다면 `ICE→SLIP_RISK`, `SAND→SINK_RISK`, other/unknown→`GENERIC_INSTABILITY`, `RECOVERY_REQUIRED=true`로 해석할 수 있다. 이번 결과에는 validated signal이 없으므로 actual fusion performance를 재평가하거나 runtime enum을 변경하지 않았다.

Existing fusion truth table unit regression은 PASS다. Terrain/Stability producer independence와 Terrain FSR4+MLP+50 ms candidate는 그대로다.

## 23. Limitations

- Fall/no-fall은 current controller와 frozen G1 simulation walking policy에 의존하는 outcome이며 real-world universal fall-risk definition이 아니다. Recovery controller가 바뀌면 label relationship도 바뀐다.
- Validation independent units는 offset별 stable 8 + fall 8로 작다. AUROC와 threshold metrics가 discrete하고 uncertainty가 크다.
- Ice stable/fall speed domain이 다르므로 speed shortcut confound를 완전히 제거하지 못했다. 그 shortcut이 있어도 reliable separation이 나오지 않았지만 absence-of-observability theorem으로 해석할 수 없다.
- Observed flips 때문에 unopened holdout은 stable 12/fall 4이며 source×terrain fall coverage가 불균형하다.
- One fixed GRU architecture와 three seeds만 평가했다. Result를 보고 hyperparameter/model/threshold search를 하지 않았다.
- Privileged 40-D state도 simulator manifold의 가능한 모든 temporal representation은 아니다. 다만 이번 predeclared upper-bound candidate는 early reliability를 지지하지 않았다.
- Holdout을 열지 않았으므로 validation diagnostic-best 수치는 unseen generalization evidence가 아니다.

Causality regression은 PASS다. Window endpoint가 마지막 input sample이고 future suffix 변경은 이미 결정된 window를 바꾸지 않으며 post-fall sample은 positive tensor에 없다. Terrain protected files의 실행 전후 SHA는 동일했고 Terrain 재학습은 없었다.

전체 repository regression은 `138 tests OK`, 기존 policy-dependent end-to-end smoke `1 skipped`로 통과했다.

## 24. Verdict

`TEMPORAL_STABILITY_SEPARABILITY_NOT_SUPPORTED`

| Requirement | Privileged | IMU6 |
|---|---|---|
| Any validation-reliable offset | None | None |
| Reliable at ≥200 ms | FAIL | FAIL |
| Reliable at 50–100 ms | FAIL | FAIL |
| Selected history/horizon | None | None |
| Holdout generalization | Not opened | Not opened |

Privileged full-state trajectory조차 50–500 ms offsets에서 모든 validation gate를 동시에 만족하지 못했다. 따라서 physical temporal separability와 IMU6 sufficiency를 지지할 evidence가 없고 production `FALL_RISK` signal도 채택하지 않는다.

## 25. Recommended next step

이 결과는 Case C에 가까운 architecture-review trigger다. 독립 run을 무작정 100–200개로 확대하거나 IMU6 Stability AI production을 시작하지 않는다. 먼저 deterministic policy에서 stable recovery와 eventual fall이 짧은 local history에서 왜 겹치는지, outcome diversity와 intervention/recovery semantics가 충분한지 검토해야 한다.

이번 milestone에서 q/dq runtime augmentation, FSR Stability input, 다른 sensor 추가, CNN/LSTM, threshold retuning, final sensor freeze, Recovery와 E84를 시작하지 않는다. Additional runtime sensor 검토도 privileged upper bound의 representation/label problem을 별도 승인 하에 재정의한 뒤에만 정당화할 수 있다.
