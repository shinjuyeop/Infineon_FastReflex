# Event-Centric Reflex Trigger Development

## 1. Motivation

이 milestone은 eventual fall prediction을 primary reflex objective에서 종료하고, frozen physical Slip 또는 uneven-support disturbance 자체를 빠르게 검출하는 binary `REFLEX_EVENT` formulation을 검증했다. 시작 시 `HEAD`와 `origin/main`은 모두 `7285978d233a57e86575d6bcaffd42698d1a5c54`, branch는 `main`, worktree는 clean이었다.

Historical dense fall-risk study의 높은 window ranking signal에도 불구하고 recoverable Ice Slip과 Sand deformation을 full-run false alert로 취급한 semantics는 reflex trigger와 맞지 않았다. 이번 질문은 “나중에 넘어지는가?”가 아니라 “지금 frozen engineering severity를 넘은 physical disturbance가 발생했는가?”다.

## 2. Why fall prediction was retired as primary trigger

현재 mechanics에서 Ice stable 60/60과 Ice fall 60/60 모두 established Slip이 발생한다. 이번 새 corpus에서도 Slip event 120개 중 58개는 최종 recovery, 62개는 fall이었다. Support event 60개도 recovery 7, fall 53으로 나뉘었다. 따라서 final outcome은 event의 존재와 동치가 아니다.

Old fall-risk semantics는 `severe Slip + recovered → negative`였지만, event-centric semantics는 `severe Slip + recovered → positive REFLEX_EVENT`다. Fall/no-fall, `t_fall`, intended stable/fall과 recovery success는 model label/input에서 제외하고 subgroup diagnostic으로만 보존했다.

## 3. Reflex event semantics

Primary binary target은 다음 union이다.

```text
REFLEX_EVENT = ANY_SLIP_EVENT OR SUPPORT_REFLEX_EVENT
```

Runtime output 후보는 `NORMAL / REFLEX_EVENT`뿐이다. Event type `SLIP / SUPPORT / SLIP_AND_SUPPORT / NONE`은 분석 metadata이며, detector가 cause를 직접 분류할 필요는 없다. `t_event`는 각 frozen oracle의 persistence가 완료된 현재 sample이다.

Detector latency는 `t_detect - t_event`다. `-20…-1 ms`는 valid early warning, `0…+50 ms`는 valid causal detection, `<-20 ms`는 premature, `>+50 ms`는 late다. Runtime alert는 probability threshold를 5 consecutive 1 kHz samples가 넘은 confirmation sample이다.

## 4. Frozen Slip oracle

Slip oracle는 historical contract를 변경하지 않았다.

```text
touchdown-anchor tangential foot drift >= 0.050 m
for 3 consecutive 1 kHz samples
```

이는 current MuJoCo G1 policy용 engineering reference이며 universal physical injury threshold가 아니다. Runtime tensor에는 drift, event clock 또는 exact contact가 들어가지 않는다.

## 5. Frozen support oracle

Support oracle도 기존 supported deformable-support contract를 재사용했다.

```text
support_surface_spread
= max(cell displacement) - min(cell displacement)
>= 0.010 m for 20 ms
```

Contact episode가 바뀌거나 fall censor가 시작되면 persistence가 reset된다. Mild balanced Sand는 전체 support가 약 20 mm 내려가도 spread가 10 mm 미만이면 negative다. 실제 no-event Sand 60개의 maximum deformation은 `20.139–20.200 mm`, peak spread는 `0 mm`였다.

## 6. ANY/BILATERAL Slip policy

Left 또는 right established Slip 중 먼저 확립된 sample을 primary ANY-Slip clock으로 사용했다. Affected-foot correctness는 gate가 아니다. Corpus breakdown은 다음과 같다.

| Slip diagnostic | Runs |
|---|---:|
| ANY Slip | 120 |
| Left event | 120 |
| Right event | 105 |
| Bilateral/overlapping | 105 |
| Unilateral left | 15 |
| Unilateral right | 0 |

Right-only event가 없다는 사실은 frozen mechanics의 observed result이며 임의 조건을 추가하지 않았다.

## 7. Dataset/replay source

먼저 historical `fall_risk_dense_20260828`을 audit했다. 256 NPZ에는 `timestamp_us`, Pelvis IMU6, privileged full state, gait phase와 fall/censor clocks만 있었다. Bilateral FSR8, tangential drift, support spread와 contact-episode evidence가 없어 Slip/support clocks를 독립 재현할 수 없었다. 기존 dataset은 수정하지 않았다.

새 Gitignored dataset은 `reflex_event_20260828`, local path는 `data/raw/reflex_event_20260828/`다. 기존 dense condition index 1–24를 TRAIN/VALIDATION development에 재생하고, historical dense index 25–30을 재사용하지 않은 index 31–36의 새 physical signatures를 HOLDOUT으로 동결했다. 240 transition + 16 hard controls를 one-NPZ-per-run으로 저장했다.

| Dataset provenance | Value |
|---|---|
| Run files | 256 |
| Valid / invalid | 256 / 0 |
| Stored size | 98,625,958 bytes (94.1 MiB) |
| Manifest SHA-256 | `64f86ffcb6b3291c8f37537a4a732ff61aa700e135b09c60590bff48e83df936` |
| Duplicate signature | 0 |
| Split overlap | 0 |
| Fresh holdout overlap with historical dense | 0 |
| Pre-transition fall invalidation | 0 |

Stored runtime channels은 Pelvis IMU6와 bilateral FSR8이다. Privileged drift/spread traces와 physical event clocks는 label reproduction/diagnostic 전용이며 runtime representation에 포함되지 않는다. Event label construction은 eventual outcome을 읽지 않는다.

## 8. Event coverage

Readiness gate는 전부 PASS했다.

| Physical class | Total | Train | Validation | Sealed holdout | Concrete origin | Marble origin |
|---|---:|---:|---:|---:|---:|---:|
| Slip event | 120 | 72 | 24 | 24 | 60 | 60 |
| Support event | 60 | 36 | 12 | 12 | 30 | 30 |
| No event transition | 60 | 36 | 12 | 12 | 30 | 30 |
| All transitions | 240 | 144 | 48 | 48 | 120 | 120 |

Event-positive outcome diagnostic은 recovered 65 / fall 115다. `SLIP_AND_SUPPORT`는 0이었다.

## 9. Negative/hard-negative coverage

Event-positive run은 target contact부터 `t_event - 30 ms`까지 deterministic negative windows를 제공한다. No-event run은 event-positive elapsed-since-contact 분포에 time-match했다. Sand benign-deformation no-event는 60개, hard-ground Concrete/Marble stable controls는 16개였다. Slip과 Support 모두 train/validation pre-event negative evidence가 존재했다.

현재 Ice mechanics에서는 120/120이 Slip event-positive여서 Ice no-slip run을 인위적으로 만들지 않았다. Within-Ice negative evidence는 각 run의 pre-slip causal interval이다.

## 10. Sensor candidates

Exactly 두 runtime representation만 비교했다.

| Candidate | Channels | Input |
|---|---:|---|
| `PELVIS_IMU6` | 6 | accel xyz + gyro xyz |
| `PELVIS_IMU6_FSR8` | 14 | Pelvis IMU6 + left FSR4 + right FSR4 |

Terrain identity/output, source/target name, patch geometry, design role, fall/outcome, `t_event`, Slip/support traces와 future sample은 tensor에 없다. Foot IMU, q/dq와 torque/current를 추가하지 않았다.

## 11. Models/windows

각 representation에서 MLP/GRU × 20/50 ms의 exactly four candidates를 비교했고 seeds `20260828/20260829/20260830` mean-probability ensemble을 사용했다. Adam 1e-3, batch 128, max 40 epochs, patience 6, validation cross-entropy best epoch를 고정했다.

Positive endpoint는 `t_event-10…t_event+50 ms`, 5 ms stride, per-run maximum 13이다. Event-run early negative와 no-event matched negative도 per-run 13으로 cap했다. Train/validation은 각각 3,380/1,144 windows와 152/52 independent runs(controls 포함)를 사용했다. Window 수는 independent sample count로 주장하지 않는다.

| Representation | Model | History | Parameters | Seed best epochs |
|---|---|---:|---:|---|
| IMU6 | MLP | 20 ms | 9,890 | 10 / 14 / 37 |
| IMU6 | MLP | 50 ms | 21,410 | 11 / 8 / 9 |
| IMU6 | GRU | 20 ms | 3,906 | 18 / 15 / 26 |
| IMU6 | GRU | 50 ms | 3,906 | 15 / 40 / 24 |
| IMU6+FSR8 | MLP | 20 ms | 20,130 | 40 / 40 / 40 |
| IMU6+FSR8 | MLP | 50 ms | 47,010 | 13 / 13 / 10 |
| IMU6+FSR8 | GRU | 20 ms | 4,674 | 40 / 40 / 30 |
| IMU6+FSR8 | GRU | 50 ms | 4,674 | 20 / 40 / 13 |

## 12. Validation threshold calibration

Validation-only grid `0.10…0.90`, step 0.02를 사용했다. Feasibility는 transition no-event FP ≤10%, hard-ground FP ≤5%, event-run premature FP ≤10%다. 결과를 보고 grid, persistence, event window 또는 model을 변경하지 않았다.

8개 candidate 모두 feasible threshold count가 0이었다. 아래 threshold는 **선택된 operating point가 아니라 report용 diagnostic-best grid point**다. 따라서 holdout detector는 freeze되지 않았다.

## 13. IMU6 result

| Model/history | Diagnostic threshold | Window AUROC/AUPRC | Event recall | Slip | Support | No-event spec. | Hard spec. | Premature | Median / p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP 20 | .86 | 1.000 / 1.000 | 41.7% | 50.0% | 25.0% | 16.7% | 0% | 58.3% | -8 / -4 ms |
| MLP 50 | .90 | .997 / .990 | 75.0% | 100% | 25.0% | 0% | 0% | 25.0% | -9 / -7 ms |
| GRU 20 | .86 | 1.000 / 1.000 | 41.7% | 50.0% | 25.0% | 0% | 0% | 58.3% | -7 / -6 ms |
| GRU 50 | .90 | 1.000 / 1.000 | 75.0% | 100% | 25.0% | 0% | 0% | 25.0% | -9 / -8 ms |

Diagnostic-best는 GRU/50 ms/.90이지만 feasible하지 않으며 selected model이 아니다. Timing gates만 PASS했고 recall/type/specificity/premature gates는 실패했다.

## 14. IMU6+FSR8 result

| Model/history | Diagnostic threshold | Window AUROC/AUPRC | Event recall | Slip | Support | No-event spec. | Hard spec. | Premature | Median / p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP 20 | .80 | 1.000 / 1.000 | 66.7% | 75.0% | 50.0% | 0% | 0% | 33.3% | -15.5 / -11 ms |
| MLP 50 | .52 | 1.000 / 1.000 | 83.3% | 100% | 50.0% | 25.0% | 0% | 16.7% | -17 / -12 ms |
| GRU 20 | .90 | 1.000 / 1.000 | 66.7% | 75.0% | 50.0% | 0% | 0% | 33.3% | -11.5 / -8 ms |
| GRU 50 | .66 | 1.000 / 1.000 | 83.3% | 100% | 50.0% | 0% | 0% | 16.7% | -15 / -10 ms |

Diagnostic-best는 MLP/50 ms/.52다. FSR8 추가로 Support recall과 일부 Sand specificity는 개선됐지만 false-alarm feasibility와 primary gates를 통과하지 못했다.

## 15. Slip recall

Diagnostic-best IMU6와 IMU6+FSR8는 모두 Slip validation 24/24를 valid window에서 검출했다. Recovered Ice Slip과 fall Ice Slip 모두 event-positive로 학습한 semantic change는 Slip observability에 유리했다. 그러나 Slip recall 단독 성공으로 detector를 채택할 수 없다. Same threshold에서 Sand/hard-ground false alerts와 Support timing이 실패했다.

## 16. Support recall

IMU6 diagnostic-best는 3/12, IMU6+FSR8 diagnostic-best는 6/12 valid Support detection이었다. Severe uneven Sand의 두 timing modes 중 event가 contact 뒤 약 1,255 ms에 확립된 run은 detector onset 약 2,462–2,464 sample과 맞아 valid했다. Event가 contact 뒤 약 1,818 ms에 확립된 run에서는 같은 alert가 약 `-574/-575 ms` premature였다. Frozen `-20…+50 ms` timing contract와 맞지 않는다.

## 17. False positives

IMU6+FSR8 MLP50 diagnostic-best는 Sand no-event 12개 중 9개에서 sustained FP를 냈고 specificity는 25%였다. IMU6 diagnostic-best는 12/12 FP였다. 모든 candidate의 hard-ground validation controls 4/4가 FP여서 hard specificity는 0%였다.

이 결과는 event-local window AUROC/AUPRC가 continuous replay operating point를 보장하지 않음을 보여준다. Capped negative windows는 full-run gait phase와 benign transients 전체를 충분히 대표하지 못했고, model은 정상 trajectory의 다른 시점에도 event-local score를 높게 냈다. 이번 milestone에서 negative mining/window/threshold를 재설계하지 않았다.

## 18. Detection latency

Valid detection만 보면 diagnostic-best IMU6+FSR8의 median/p95는 `-17/-12 ms`, IMU6는 `-9/-8 ms`로 timing gate를 만족한다. 그러나 이는 valid subset에만 해당한다. Premature, no-event FP, hard FP와 missed/late runs를 제외한 latency이므로 detector acceptance 근거로 독립 사용할 수 없다.

## 19. Ice representative replay

아래는 selected detector가 아니라 diagnostic-best validation replay(IMU6+FSR8, MLP50, 50 ms, threshold .52)다. Terrain output은 clean target touchdown +50 ms로 표시했다.

| Run | Contact / Terrain output | Peak drift | Slip event | Detect | Latency | P(event) / P(detect) | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| `evt_c_ice_s19` | 1220 / 1270 ms | 113.28 mm | 2189 | 2171 | -18 ms | 1.000 / .933 | Recovered |
| `evt_c_ice_f19` | 1220 / 1270 ms | 117.15 mm | 1910 | 1896 | -14 ms | 1.000 / .986 | Fall |
| `evt_c_ice_s19` bilateral | 1220 / 1270 ms | 113.28 mm | 2189 | 2171 | -18 ms | 1.000 / .933 | Recovered |

두 representative 모두 Terrain 50 ms output보다 event/detect가 늦었다. Architecture상 detector는 Terrain을 기다리지 않지만, 이 replay는 “event가 Terrain refinement보다 먼저 왔다”는 evidence가 아니다.

## 20. Sand representative replay

| Run | Contact / Terrain output | Deformation / spread | Support event | Detect | Classification | P(event) / P(detect) | Outcome |
|---|---:|---:|---:|---:|---|---:|---|
| `evt_c_sand_s19` | 1220 / 1270 ms | 20.17 / 0 mm | none | 1917 | No-event FP | n/a / .994 | Recovered |
| `evt_c_sand_f20` | 1220 / 1270 ms | 40.17 / 40.17 mm | 2475 | 2462 | Valid, -13 ms | 1.000 / .968 | Recovered |
| `evt_c_sand_f19` | 1220 / 1270 ms | 40.09 / 40.09 mm | 3038 | 2464 | Premature, -574 ms | .999 / .921 | Fall |

동일 model이 support event의 존재 신호를 강하게 rank하지만 frozen event clock과 benign/full-run state를 안정적으로 구분하지 못했다.

## 21. Recovered-event vs fall-event comparison

| Event | Recovered | Fall | Total |
|---|---:|---:|---:|
| Slip | 58 | 62 | 120 |
| Support | 7 | 53 | 60 |
| All events | 65 | 115 | 180 |

Representative Ice에서는 recovered/fall 모두 valid하게 검출됐다. Sand에서는 recovered event가 valid한 예도 있지만 event timing mode에 따라 recovered/fall 양쪽에 premature 또는 valid 사례가 존재한다. Detector가 fall outcome을 알 필요는 없지만 현재 continuous calibration은 충분하지 않다.

Severity diagnostic은 Slip peak drift `57.37 / 134.77 / 192.26 / 192.26 mm`(min/median/p95/max), peak tangential velocity `2.084 / 2.445 / 6.044 / 6.044 m/s`였다. Support peak spread와 maximum deformation은 각각 `36.55 / 40.11 / 40.17 / 40.17 mm`; pattern은 60/60 `lateral_deformable`이었다. 이 분포로 LOW/MEDIUM/HIGH classifier를 만들지 않았다.

## 22. Terrain interaction

Event detector는 continuous하며 Terrain output을 기다리지 않는 architecture contract를 유지한다. Event가 먼저라면 `UNKNOWN + REFLEX_EVENT → GENERIC_DISTURBANCE → REFLEX_REQUIRED`, 뒤의 Terrain update로 cause를 refine할 수 있다. 다만 이번 representative에서는 clean touchdown 기반 Terrain output 1270 ms가 detector보다 먼저였다.

Terrain protected files의 before/after SHA는 동일했고 retraining은 없었다. Current `FSR4 + MLP + 50 ms`, holdout macro F1 0.9713/worst recall 0.95 evidence는 untouched다. Existing fusion truth-table regression도 PASS했고 production enum은 변경하지 않았다.

## 23. Previous fall-risk comparison

| Study | Primary semantics | Ranking diagnostic | Run-level result |
|---|---|---|---|
| Sparse fall-risk | fixed-offset eventual fall | Privileged .875 / IMU6 .844 AUROC best | reliable horizon not supported |
| Dense fall-risk | fall within H | Privileged .972–.987 / IMU6 .921–.965 AUROC | `DENSE_FALL_RISK_NOT_SUPPORTED` |
| Event-centric | frozen physical event | validation window AUROC .997–1.000 | no feasible continuous threshold |

Event-centric relabeling은 severe recovered Slip을 올바른 positive로 바꾸고 window separation을 명확히 했다. 그러나 high window AUROC와 full-run safety operating point의 간극은 남았다. Historical MoS, full-state Mahalanobis, sparse temporal audit와 dense fall-risk reports는 수정하지 않았다.

## 24. Limitations

- Development condition diversity는 frozen calibrated domains 안에 제한된다.
- Ice no-event가 없어 Ice negative는 pre-event region뿐이다.
- Support event pattern은 `lateral_deformable`에 집중돼 severity/pattern generalization을 주장할 수 없다.
- Window-local capped negatives가 continuous gait phase의 모든 hard negative를 대표하지 못했다.
- Exact Slip/support clocks는 MuJoCo privileged engineering oracles이며 real-world universal thresholds가 아니다.
- 48-run fresh holdout은 생성/integrity/event-count readiness만 확인했고 waveform/model evaluation은 validation selection 부재로 봉인했다. Guard open count는 0이다.
- No detector가 selected되지 않았으므로 runtime sensor recommendation, architecture support와 final sensor freeze를 선언할 수 없다.

## 25. Verdict

`EVENT_CENTRIC_REFLEX_DETECTION_NOT_SUPPORTED`

`EVENT_CENTRIC_REFLEX_ARCHITECTURE_SUPPORTED`는 선언하지 않는다.

근거는 두 representation 모두 validation feasible threshold가 0이고, overall/Support recall, Sand no-event specificity, hard-ground specificity와 premature FP가 primary gates를 크게 위반했기 때문이다. Holdout은 열지 않았으며 holdout metric은 없다.

## 26. Next recommendation

이번 milestone의 제한에 따라 Foot IMU, q/dq, torque/current, severity 3-class, Recovery, final sensor freeze 또는 E84를 자동 시작하지 않는다. 다음 별도 승인 milestone이 있다면 sensor augmentation보다 먼저 event-local training과 continuous replay 사이의 negative-coverage mismatch, hard-ground false alerts와 delayed Support clock timing mode를 protocol 수준에서 재검토해야 한다. 이번 결과를 threshold/persistence sweep으로 소급 수정하지 않는다.
