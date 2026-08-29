# Unified Hazard Reflex System Validation

## 1. Purpose

`UNIFIED_HAZARD_REFLEX_SYSTEM_VALIDATION`은 cause classifier의 완전성보다 현재 실제 reflex-worthy physical hazard가 존재하는지를 control-facing target으로 검증했다. 시작 상태는 `HEAD = origin/main = d3501ed4627bbea66929f6da04dfc032e48c3495`, branch `main`, tracked worktree clean이었다. 최종 primary verdict는 `UNIFIED_HAZARD_REFLEX_SUPPORTED_SINGLE_IMU`다.

## 2. Why cause classification was separated from reflex decision

Frozen Support detector가 Ice Slip에도 반응한다는 사실은 Support cause 분류 관점에서는 cross-trigger지만, controller 관점에서는 실제 Slip이 이미 reflex를 요구하므로 false reflex가 아니다. 따라서 시스템은 먼저 `REFLEX_REQUIRED`를 결정하고 Terrain은 이후 cause refinement에만 사용한다.

## 3. Historical evidence

Frozen continuous Slip은 이전 one-shot holdout에서 recall 24/24, true no-hazard specificity 16/16, premature 0%, latency median 약 -27.5 ms였다. Frozen Support는 development에서 established-event recall TRAIN 36/36, VALIDATION 12/12와 Sand benign specificity 100%를 보였다. 약 -565 ms Support early mode는 gait alias가 아니라 6.5 mm 부근에서 시작한 physical precursor였고 I1이 이를 validation 12/12에서 포착했다. 기존 MoS/full-state/fall-risk/event/fusion 실패 및 Continuous Slip 성공 결과는 수정하지 않았다.

## 4. Physical hazard semantics

Primary label은 `ESTABLISHED_SLIP OR ESTABLISHED_SUPPORT`다. Fall/recovery, scenario role, Terrain output은 label에 들어가지 않는다. Actual physical clocks가 design role을 override했다. Fresh corpus에는 `SLIP_HAZARD` 64, `SUPPORT_HAZARD` 64, `NO_HAZARD` 128이 있었고 `SLIP_AND_SUPPORT_HAZARD`와 `SUPPORT_PRECURSOR_ONLY`는 0이었다.

## 5. Slip reference

Slip reference는 touchdown anchor tangential foot drift 50 mm가 3 ms 지속된 최초 causal sample이다. System alert의 허용 구간은 `t_slip - 30 ms`부터 `t_slip + 40 ms`까지며 바꾸지 않았다.

## 6. Support established/precursor references

Established Support는 support-surface spread 10 mm가 20 ms 지속된 clock이다. I1은 loaded-foot spread의 positive 1 ms derivative가 TRAIN benign q99.5 bound를 20 ms 지속적으로 넘는 privileged precursor다. 기존 benign corpus의 bound가 정확히 0으로 collapse한 사실을 그대로 보존했다. Fresh established Support 64/64에 I1이 established clock 이전에 존재했다. Support alert는 I1 이상, established +50 ms 이하일 때 causal하게 유효하며 I1은 runtime tensor나 deployable threshold가 아니다.

## 7. Fresh dataset design

Dataset ID는 `unified_hazard_reflex_20260829`다. Calibrated mechanics 안에서 source, patch start/width, Sand pattern과 contact phase를 deterministic하게 변화시켰으며 Ice friction, Sand stiffness/damping/travel, policy/controller는 변경하지 않았다.

| Designed group | Runs | Concrete | Marble | Actual physical label |
|---|---:|---:|---:|---|
| Ice Slip hazard | 64 | 32 | 32 | Slip 64 |
| Sand Support hazard | 64 | 32 | 32 | Support 64 |
| Sand benign | 64 | 32 | 32 | No hazard 64 |
| Hard-ground normal | 64 | 32 | 32 | No hazard 64 |

256 NPZ의 총 크기는 101,116,918 bytes(약 96.43 MiB)다. Manifest SHA-256은 `d023384aaaac22076a2be7c2fa242acac91a548d82102640c39868990cb0e9d6`이며 raw data는 Gitignored다.

## 8. Dataset readiness

| Gate | Result |
|---|---:|
| Valid runs | 256/256 |
| Established Slip | 64 |
| Established Support | 64 |
| I1-covered Support | 64/64 = 100% |
| Primary no-hazard | 128 |
| Sand benign no-hazard | 64 |
| Hard-ground no-hazard | 64 |
| Invalid / pre-transition fall | 0 / 0 |
| Duplicate / split overlap / prior signature overlap | 0 / 0 / 0 |

모든 readiness gate가 통과했다. 7개 prior manifest를 integrity-check했고 comparable physical signature 447개와 교집합이 없었다.

## 9. Split and holdout sealing

Simulation 전에 각 64-run group을 TRAIN 38, VALIDATION 13, HOLDOUT 13으로 고정했다. 전체 split은 152/52/52이며 outcome을 보고 membership을 바꾸지 않았다. HOLDOUT 전에는 file existence, SHA, count와 membership만 확인했고 guard open count는 0이었다. Final candidate와 threshold를 freeze한 뒤 guard를 한 번 열었으며 reselection은 없었다.

## 10. Phase A frozen dual-detector system

Phase A는 frozen Slip `Pelvis IMU-derived 80D / GRU / 20 ms / threshold 0.58 / persistence 5 ms`와 frozen Support `Pelvis IMU-derived 60D / GRU / 20 ms / threshold 0.94 / persistence 5 ms`를 각각 Terrain-independent하게 continuous replay한 뒤 alert를 OR했다. Normalizer, checkpoint, feature schema와 Terrain model hash가 모두 일치했다. Terrain은 score, GRU state, persistence 또는 OR을 gate하지 않았다.

## 11. Slip results

Fresh VALIDATION의 Phase A Slip hazard recall은 11/13 = 84.62%였다. 두 miss(`uhr_ice_h_c20`, `uhr_ice_h_c24`)의 frozen Slip alert는 event -32 ms에 발생해 frozen lower boundary -30 ms보다 정확히 2 ms 빨랐다. 이를 뒤의 alert로 소급 rescue하지 않았다. 유효 검출 latency는 median/p95 -27/-27 ms였다.

## 12. Support results

Phase A의 fresh Support system/native branch recall은 모두 13/13 = 100%였다. System detection의 precursor-relative latency median은 1220 ms였고 established-relative latency median/p95는 -17/-15 ms였다. Established Support lead median은 17 ms였으며 최소 15 ms였다.

## 13. Hazard cross-trigger

VALIDATION cause diagnostic에서 Slip hazard 13개 중 Slip branch가 11개, Support branch가 4개를 유효 구간에서 trigger했다. Support hazard 13개 중 Support branch가 13개, Slip branch가 3개를 trigger했다. 중복 trigger가 가능하므로 합은 run 수와 같을 필요가 없다. 다른 branch가 실제 hazard에서 먼저 반응한 경우를 system false positive로 세지 않았다.

## 14. System hazard recall

Phase A overall hazard recall은 24/26 = 92.31%, Slip 84.62%, Support 100%였다. Overall gate는 통과했지만 Slip-specific 95% gate가 실패했다.

## 15. True no-hazard specificity

Phase A fresh VALIDATION에서 primary no-hazard 26/26, Sand benign 13/13, hard-ground 13/13이 모두 false reflex 없이 통과했다. `NO_HAZARD`는 Slip, I1, established Support가 모두 없는 trajectory에만 부여했다.

## 16. Premature behavior

Phase A system premature는 Slip의 -32 ms 두 건, 즉 hazard run 2/26 = 7.69%였다. Support pre-I1 premature와 no-hazard false reflex는 0이었다. Premature rate gate 자체는 통과했지만 두 건은 first-alert Slip recall을 실패시켰다.

## 17. Timing

| System | Slip median / p95 | Support precursor latency median | Support established latency median / p95 | Support lead median |
|---|---:|---:|---:|---:|
| Phase A validation | -27 / -27 ms | 1220 ms | -17 / -15 ms | 17 ms |
| Phase B validation, 20 ms | -25 / -24 ms | 1217 ms | -562 / -19 ms | 562 ms |
| Phase B holdout, 20 ms | -25 / -24 ms | 1217 ms | -563 / -19 ms | 563 ms |

Phase B Support latency distribution은 두 contact-phase mode 때문에 p95가 -19 ms이고 median은 약 -563 ms다. 모든 detection은 I1 이후였으며 미래 state는 사용하지 않았다.

## 18. Phase A verdict

Phase A는 specificity, Support, overall recall, premature와 timing gate를 통과했지만 Slip recall 84.62% < 95% 때문에 FAIL이다. Frozen detector threshold/persistence를 retune하지 않았고 predeclared Phase B만 활성화했다.

## 19. Phase B activation decision

Phase B는 Phase A 실패 후에만 시작했다. TRAIN 152개만 normalization, Round 0와 HNM 3회에 사용했다. 두 history 후보가 모두 Round 3를 완료하기 전에는 Phase B validation threshold evaluation을 수행하지 않았다.

## 20. Unified feature schema

Runtime input은 Pelvis IMU6만이다. Existing Slip80 schema가 Support60의 strict semantic superset임을 확인했다: Support `raw`는 Slip `base`, Support `mean_10ms`/`variance_10ms`는 Slip `causal_mean_10ms`/`causal_variance_10ms`에 정확히 대응한다. 새 transform, Terrain, FSR, Foot IMU, physical clock, fall/recovery는 추가하지 않았다. Selected GRU의 feature dimension은 80, history 20 ms, parameter count는 11,010이다.

## 21. Unified HNM if used

각 round는 TRAIN 152개를 1 ms replay하고 run당 K=12, 최소 30 ms spacing으로 1,824개 hard negatives를 선택했다. I1 이상과 Slip positive 구간은 negative가 될 수 없다.

| History | Round | Fit windows | Monitor windows | New HNM |
|---:|---:|---:|---:|---:|
| 20 ms | 0 | 8,605 | 2,296 | 1,824 |
| 20 ms | 1 | 9,970 | 2,660 | 1,824 |
| 20 ms | 2 | 11,409 | 3,044 | 1,824 |
| 20 ms | 3 | 12,844 | 3,427 | — |
| 50 ms | 0 | 7,180 | 1,916 | 1,824 |
| 50 ms | 1 | 8,555 | 2,284 | 1,824 |
| 50 ms | 2 | 9,946 | 2,652 | 1,824 |
| 50 ms | 3 | 11,364 | 3,030 | — |

## 22. Unified validation if used

Threshold grid 0.10–0.99 step 0.01과 persistence 5 ms를 VALIDATION에서만 평가했다. 20 ms는 0.54–0.99, 50 ms는 0.66–0.99가 모든 gate를 통과했다. Selection priority와 conservative threshold tie-break로 각 후보 threshold 0.99가 선택됐고, 둘 다 26/26 hazard recall, 26/26 no-hazard specificity, premature 0이었다. 두 후보가 pass했으므로 predeclared shorter-history rule로 20 ms를 선택했다. 50 ms의 marginal timing 차이로 복잡도를 늘리지 않았다.

## 23. Final candidate freeze

Final candidate는 `PHASE_B_UNIFIED_HAZARD_DETECTOR`: Pelvis IMU6-derived 80D, GRU, history 20 ms, three-seed mean probability, threshold 0.99, persistence 5 ms다. Freeze SHA-256은 `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`다. Checkpoint/normalizer는 simulation ground-truth output이 아닌 generated research artifact이며 Gitignored다.

## 24. One-shot fresh holdout

Freeze 뒤 fresh HOLDOUT 52개를 정확히 한 번 열었다. 구성은 Slip 13, Support 13, Sand benign 13, hard normal 13이다. Guard open count는 1, policy/model/history/threshold/persistence 변경과 reselection은 0이다.

## 25. Holdout hazard recall

Overall hazard recall은 26/26 = 100%, Slip 13/13 = 100%, Support 13/13 = 100%였다. Concrete-origin hazard 12/12와 Marble-origin hazard 14/14가 모두 검출됐다. Slip latency min/p10/median/p95/max는 -26/-25.8/-25/-24/-24 ms였다. Support established lead min/p10/median/p95/max는 19/19/563/597/597 ms였다.

## 26. Holdout specificity

Primary no-hazard specificity는 26/26 = 100%, Sand benign 13/13 = 100%, hard-ground 13/13 = 100%였다. Premature alert는 Slip, pre-I1 Support, 전체 system 모두 0/26 hazard runs였다. Precursor-only run은 fresh corpus에 0개여서 해당 gray-zone behavior는 추가 증거가 없다.

## 27. Cause-refinement diagnostics

Single unified detector가 Slip 13/13과 Support 13/13을 모두 trigger했으므로 runtime cause branch 정확도를 primary claim으로 삼지 않는다. HOLDOUT reflex 시점의 frozen Terrain state는 ICE 5, SAND 3, CONCRETE 4, MARBLE 14였고 actual target과 일치한 것은 3/26뿐이었다. 이 수치는 current held Terrain state가 reflex authorization에 부적합하지만 이후 갱신되는 advisory cause refinement에는 사용할 수 있음을 보여준다.

## 28. Terrain/reflex timing

HOLDOUT hazard 26개 중 15개에서 reflex가 first valid target Terrain output보다 먼저였다. `reflex - Terrain valid` median은 -172 ms, range는 -574–615 ms였다. Target touchdown부터 Terrain valid까지 median은 1095 ms였다. Terrain은 reflex를 지연·차단하지 않았고 protected FSR4/MLP/50 ms producer는 재학습하지 않았다.

## 29. Sensor-count implication

Hazard reflex는 shared Pelvis IMU6만 사용하고 Terrain advisory는 current minimum FSR4 candidate를 유지한다. Provisional unique set은 10 physical channels다. Foot IMU, q/dq, torque/current 또는 FSR-to-reflex augmentation은 필요하지 않았다. 이는 `10_CHANNEL_REFLEX_SYSTEM_CANDIDATE`이지 `FINAL_SENSOR_ARCHITECTURE_FROZEN` 선언이 아니다.

## 30. Historical comparison

| Evidence | Hazard recall | Slip | Support | No-hazard specificity | Premature |
|---|---:|---:|---:|---:|---:|
| Previous frozen Slip one-shot | Slip 24/24 | 100% | n/a | 16/16 | 0% |
| Fresh Phase A frozen OR validation | 24/26 | 84.62% | 100% | 26/26 | 7.69% |
| Fresh unified Phase B validation | 26/26 | 100% | 100% | 26/26 | 0% |
| Fresh unified Phase B holdout | 26/26 | 100% | 100% | 26/26 | 0% |

Phase A의 failure는 old detectors를 재해석해 숨기지 않았다. Unified target과 dense TRAIN-only HNM이 cause-specific cross-trigger를 safety-positive evidence로 학습한 결과만 새 evidence다.

## 31. Limitations

결과는 current G1 policy, MuJoCo engineering terrain, current physical Slip/Support criteria와 calibrated domain에 한정된다. I1 q99.5 threshold가 benign spread 0 때문에 0으로 collapse하므로 universal physical law나 deployable sensor threshold가 아니다. Fresh precursor-only 사례가 없어 gray-zone generalization은 검증되지 않았다. Threshold 0.99는 calibrated ensemble operating point이지 probability calibration claim이 아니다. Terrain advisory는 reflex 순간 cause correctness가 낮으며 final cause가 필요한 consumer는 later Terrain update를 기다려야 한다. Recovery controller, E84 compute/memory, hardware realism과 integrated runtime은 평가하지 않았다.

## 32. Verdict

`UNIFIED_HAZARD_REFLEX_SUPPORTED_SINGLE_IMU`

Phase A frozen OR은 Slip gate를 실패했지만 predeclared Phase B Pelvis IMU6 unified GRU가 fresh validation과 one-shot holdout의 hazard, specificity, premature와 timing gate를 모두 통과했다. Protected Slip/Support/Terrain artifacts와 fusion regression은 unchanged/PASS이며 causality는 PASS다.

Full repository regression은 256 tests PASS, 1 skipped였다.

## 33. Next recommendation

다음 architecture candidate는 `Pelvis IMU6 → unified continuous HAZARD_REFLEX_REQUIRED`를 먼저 내고 `FSR4 → frozen Terrain → cause refinement`를 비동기 advisory로 적용하는 구조다. Generated candidate를 deployment release로 승격하거나 sensor architecture를 최종 freeze하지 않는다. 다음 작업인 Recovery, E84, quantization, integrated production model은 자동으로 시작하지 않는다.
