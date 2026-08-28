# Continuous Slip Reflex Detector Development

## 1. Purpose

이 milestone은 frozen Terrain 판정을 기다리지 않고 runtime sensor trajectory만으로 reflex-worthy established Slip을 연속 감지할 수 있는지 검증했다. Slip alert는 Terrain이 `UNKNOWN`이어도 즉시 `REFLEX_REQUIRED=true`를 만든다. Frozen Sand Support detector와 frozen Terrain recognizer는 변경하거나 재학습하지 않았다.

시작 상태는 `main`, HEAD와 `origin/main` 모두 `f1228e517879db48cff778786150d2fa7404bc74`, clean worktree였다.

## 2. Why terrain-gated Slip failed

이전 구조에서는 development Slip 96/96이 valid target `ICE` output보다 먼저 발생했다. 따라서 active-ICE positive/negative endpoint가 모두 0이었고 Phase A/B가 fail-closed됐다. 이는 sensor observability 실패가 아니라 `TERRAIN_GATED_SLIP_DETECTION_CAUSALLY_TOO_LATE`였다.

## 3. Asymmetric reflex architecture

```text
Pelvis IMU / FSR / optional Foot IMU
        -> Continuous Slip -> immediate REFLEX_REQUIRED

FSR4 touchdown window -> frozen Terrain -> advisory cause refinement
Pelvis IMU6 -> frozen Support -> active only while frozen Terrain == SAND
```

Slip은 Terrain과 독립적으로 동작한다. Support만 기존대로 `SAND` output에 조건화한다.

## 4. Frozen Slip physical oracle

Oracle은 touchdown anchor 기준 tangential foot drift `>= 0.050 m`가 1 kHz에서 3 samples 지속된 최초 시점이다. LEFT 또는 RIGHT 중 먼저 확립된 시점을 `t_slip`으로 사용했다. threshold, persistence, ANY/BILATERAL semantics를 변경하지 않았다.

## 5. Dataset provenance

`reflex_event_20260828`의 256 runs와 기존 split을 재사용했다. Manifest SHA-256은 `64f86ffcb6b3291c8f37537a4a732ff61aa700e135b09c60590bff48e83df936`, NPZ payload는 98,625,958 bytes이며 모든 file SHA가 일치했다.

| Split | Total | Slip | Support | No event |
|---|---:|---:|---:|---:|
| TRAIN | 152 | 72 | 36 | 44 |
| VALIDATION | 52 | 24 | 12 | 16 |
| HOLDOUT | 52 | 24 | 12 | 16 |

Slip 120 runs는 bilateral 105, left-only 15, right-only 0이며 recovered/fall은 58/62였다. Split별 Slip은 TRAIN bilateral/left 63/9, VALIDATION 21/3, HOLDOUT 21/3이다. Physical signature duplicate와 split overlap은 0이다.

No-event evidence는 76 runs이며 benign Sand 60, hard Concrete 8, hard Marble 8이다.

## 6. Holdout sealing proof

이전 selection artifact는 `holdout_opened=false`, guard open count 0이었다. 이 실험은 TRAIN만 먼저 로드했고 12 Phase A candidates의 Round 3 checkpoint가 모두 생성된 후 VALIDATION을 로드했다. Slip candidate와 frozen Support provenance를 기록한 `selection_before_holdout.json`을 쓴 다음 common HOLDOUT을 정확히 한 번 열었다. 최종 guard open count는 1이며 holdout 이후 reselection은 0이다.

## 7. Phase A physical sensors

정확히 세 representation을 비교했다.

| ID | Runtime input | Raw physical channels | Derived dimensions |
|---|---|---:|---:|
| A1 | Pelvis IMU6 | 6 | 80 |
| A2 | Bilateral FSR8 | 8 | 240 |
| A3 | Pelvis IMU6 + FSR8 | 14 | 320 |

각 representation에 MLP/GRU와 20/50 ms histories를 적용해 총 12 candidates를 평가했다.

## 8. Causal derived features

Pelvis block은 raw accel/gyro xyz와 accel/gyro/horizontal norms 10 base values에 base, delta 1/5/10 ms, trailing mean 5/10 ms, trailing variance 5/10 ms를 순서대로 적용했다. FSR block은 raw8, per-foot spatial/load 18, bilateral 4의 30 base values에 같은 8 transforms를 적용했다. 모든 rolling operation은 current/past samples만 사용한다.

Feature schema에 Terrain, scenario, event clock, fall/recovery, q/dq/torque가 없고 future-suffix regression에서 causal prefix가 동일했다.

## 9. Continuous negative coverage

각 Slip run은 valid walking start부터 `t_slip-40 ms`까지 negative evidence를 제공했다. No-Slip hard controls와 benign Sand는 full pre-censor walking interval을 제공했고 Support runs는 `t_support-30 ms` 이전만 primary no-hazard negative로 사용했다. Contact phase는 touchdown/loading, contact release, left/right/double/no-support와 high contact shock를 sampling diversification에만 사용했으며 tensor에는 넣지 않았다.

Round 0 fit set은 history 20 ms에서 10,322 windows, history 50 ms에서 9,096 windows였다. 각각 positive 870 windows를 보존했고 전체 independent TRAIN unit은 152 runs다.

## 10. Round 0 training

세 fixed seeds `20260828/20260829/20260830`, balanced cross-entropy, canonical MLP/one-layer GRU hidden 32를 사용했다. Epoch는 TRAIN 내부 monitor cross-entropy minimum으로 선택했다. Round 0 no-hazard run FP는 candidate에 따라 0%에서 81.8%였으며 이 값을 validation selection에 사용하지 않았다.

## 11. HNM Round 1

모든 TRAIN run을 1 ms causal replay하고 true no-hazard endpoint에서 run당 최대 12개를 probability 내림차순으로 선택했다. 선택점 간격은 최소 30 ms다. 모든 candidate에서 1,824 windows가 HNM1으로 선택됐다. Support hazard region과 Slip positive region은 제외했다.

## 12. HNM Round 2

동일 규칙으로 두 번째 1,824 hard negatives를 TRAIN에서만 추가했다. A2 MLP 20 ms는 no-hazard run FP가 81.8%→61.4%→0%로 감소했다. A2 MLP 50 ms는 HNM2 뒤에도 81.8%가 남았다.

## 13. HNM Round 3

세 번째 1,824 hard negatives를 같은 규칙으로 추가하고 Round 3 final checkpoint를 생성했다. A2 MLP 50 ms는 HNM3에서 처음 0%에 도달했다. Validation을 보고 round를 추가하지 않았고 Round 4는 수행하지 않았다.

## 14. Train continuous replay

아래 FP는 threshold 0.5의 TRAIN-only diagnostic이다. Window count는 correlated samples이고 independent unit은 152 simulation runs다.

| Candidate | Round 0 FP | HNM1 FP | HNM2 FP | HNM3 FP |
|---|---:|---:|---:|---:|
| A1 MLP 20 | 81.8% | 0% | 0% | 0% |
| A1 MLP 50 | 81.8% | 0% | 0% | 0% |
| A1 GRU 20 | 20.5% | 0% | 0% | 0% |
| A1 GRU 50 | 47.7% | 0% | 0% | 0% |
| A2 MLP 20 | 81.8% | 61.4% | 0% | 0% |
| A2 MLP 50 | 81.8% | 81.8% | 81.8% | 0% |
| A2 GRU 20 | 20.5% | 0% | 0% | 0% |
| A2 GRU 50 | 0% | 0% | 0% | 0% |
| A3 MLP 20 | 81.8% | 0% | 0% | 0% |
| A3 MLP 50 | 81.8% | 0% | 0% | 0% |
| A3 GRU 20 | 70.5% | 0% | 0% | 0% |
| A3 GRU 50 | 50.0% | 0% | 0% | 0% |

Final fit windows는 candidate별 12,834–14,649, monitor windows는 3,129–3,587이었다. Positive windows 870은 모든 round에서 유지됐다.

## 15. Phase A validation

모든 Round 3 checkpoint가 존재한 후 52 VALIDATION runs를 1 ms replay하고 0.10–0.98/0.02 grid와 5 ms persistence를 적용했다.

| Candidate | Threshold | Slip recall | No-hazard specificity | Premature | Negative-time | Latency median/p95 | Result |
|---|---:|---:|---:|---:|---:|---:|:---:|
| A1 MLP 20 | 0.98 | 75% | 100% | 25% | 0.0029% | -29/-28 ms | FAIL |
| A1 MLP 50 | 0.98 | 100% | 100% | 0% | 0% | -29.5/-28 ms | PASS |
| A1 GRU 20 | 0.58 | 100% | 100% | 0% | 0% | -27.5/-27 ms | PASS |
| A1 GRU 50 | 0.98 | 100% | 100% | 0% | 0% | -30/-28 ms | PASS |
| A2 MLP 20 | 0.96 | 100% | 100% | 0% | 0% | -27.5/-27 ms | PASS |
| A2 MLP 50 | 0.82 | 100% | 100% | 0% | 0% | -28.5/-28 ms | PASS |
| A2 GRU 20 | 0.26 | 50% | 100% | 50% | 0.0144% | -29/-29 ms | FAIL |
| A2 GRU 50 | 0.22 | 50% | 100% | 50% | 0.0145% | -29.5/-29 ms | FAIL |
| A3 MLP 20 | 0.78 | 75% | 100% | 25% | 0.0058% | -29/-29 ms | FAIL |
| A3 MLP 50 | 0.98 | 100% | 100% | 0% | 0% | -29.5/-28 ms | PASS |
| A3 GRU 20 | 0.54 | 100% | 100% | 0% | 0% | -29/-29 ms | PASS |
| A3 GRU 50 | 0.98 | 75% | 100% | 25% | 0.0029% | -29/-28 ms | FAIL |

Selected A1 GRU 20 ms는 recovered/fall 12/12와 12/12, bilateral/left-only 21/21과 3/3, Concrete/Marble origin 12/12와 12/12를 검출했다. Right-only support는 0 runs다.

## 16. Phase A candidate selection

A1 Pelvis IMU6-derived + GRU + 20 ms를 선택했다. A1의 multiple candidates와 A2/A3 candidates가 primary gates를 통과했지만 recall/specificity/p95 near-tie 안에서 raw physical channels가 가장 적은 A1을 우선했고, A1 내에서는 shorter history와 smaller model인 GRU 20 ms가 선택됐다.

Selected model은 80 features, 11,010 parameters, threshold 0.58, persistence 5 ms다.

## 17. Phase B activation decision

Phase A candidate가 validation-supported이므로 Phase B는 미활성화했다. Foot IMU checkpoint는 학습하지 않았고 Foot IMU를 sensor recommendation에 포함하지 않았다.

## 18. Foot IMU observer parity

기존 `reflex_event_foot_imu_20260828` 256-run corpus를 metadata/file-integrity audit했다. Manifest SHA-256은 `c206e05dec4f6bb63a667a5fe8f0c1fd0e9c4d45f4c4e8901a571c81c0dba234`, payload는 171,100,805 bytes다. Scenario/split/event clock/policy action/contact/observer-only physics parity가 모두 PASS였고 corpus를 재생성하지 않았다.

## 19. Phase B HNM results

Phase B activation condition이 false이므로 B1–B4의 Round 0/HNM은 수행하지 않았다. 이는 missing result가 아니라 predeclared conditional branch의 정상 종료다.

## 20. Phase B validation

Phase B validation은 수행하지 않았다. Phase A 성공 후 Foot IMU 결과를 보는 sensor expansion은 금지했다.

## 21. Slip candidate freeze

Selected feature order, TRAIN-only normalizer, three Round 3 ensemble checkpoints, threshold 0.58, 5 ms persistence와 `[-30,+40] ms` timing contract를 holdout 전에 freeze했다. Freeze artifact SHA-256은 `df0a232ec242283ef8b25c59421cebde982a7a93febb655cc511fa2fa3de3229`다.

## 22. Frozen Support candidate verification

Pelvis IMU6 derived + GRU + 20 ms + threshold 0.94 + persistence 5 ms의 identity와 feature schema SHA, 세 checkpoint SHA가 이전 milestone과 일치했다. Support normalizer는 이전 canonical TRAIN rule로 147,456 samples에서 결정적으로 재구성했다. 재학습, threshold 변경, event 변경은 0이다.

Support protected hashes before/after가 일치했고 Terrain protected hashes와 기존 Fusion truth table도 PASS했다.

## 23. One-shot holdout

Slip freeze와 Support verification 후 52-run common HOLDOUT을 한 번 열었다. Slip, frozen Support, frozen Terrain과 integrated logic을 같은 opening에서 평가했다. Guard open count는 1이고 model/history/threshold reselection은 하지 않았다.

## 24. Slip holdout

Continuous Slip은 모든 holdout gate를 통과했다.

| Metric | Result | Gate |
|---|---:|---:|
| Slip recall | 24/24 = 100% | >=90% |
| True no-hazard specificity | 16/16 = 100% | >=90% |
| Premature Slip runs | 0/24 = 0% | <=15% |
| Negative-time alert fraction | 0% | <=3% |
| Latency median / p95 | -27.5 / -27 ms | <=20 / <=50 ms |

Recovered/fall recall은 10/10과 14/14, bilateral/left-only는 21/21과 3/3, Concrete/Marble origin은 각각 12/12다. Right-only holdout support는 없다.

## 25. Support holdout

Frozen Support는 recall gate를 통과하지 못했다.

| Metric | Result | Gate |
|---|---:|---:|
| Support recall | 9/12 = 75% | >=85% — FAIL |
| Benign Sand specificity | 12/12 = 100% | >=85% |
| Premature | 0/12 = 0% | <=15% |
| Latency median / p95 | -16 / -15 ms | <=25 / <=50 ms |

Recovered Support는 2/2, fall Support는 7/10이었다. Concrete/Marble origin은 4/6과 5/6이다. 기존 validation 12/12 결과는 수정하지 않으며 이번 one-shot holdout generalization 결과를 추가 evidence로 보존한다.

## 26. Integrated system evaluation

Slip alert는 Terrain을 기다리지 않고 reflex를 발생시키고, Support alert는 frozen `SAND`에서만 유효하게 결합했다.

| Metric | Result | Gate |
|---|---:|---:|
| Physical event union recall | 33/36 = 91.67% | >=87.5% PASS |
| No-hazard specificity | 16/16 = 100% | >=90% PASS |
| Hard-ground specificity | 4/4 = 100% | >=90% PASS |

Integrated gates는 모두 PASS지만 frozen Support branch 자체 gate가 FAIL이므로 final architecture support를 선언하지 않는다.

## 27. Hard-ground false reflex

Holdout hard Concrete/Marble controls 4/4에서 continuous Slip 또는 gated Support sustained alert가 없었다. Hard-ground specificity는 100%, system false-reflex count는 0이다.

## 28. Hazard cross-trigger analysis

Support holdout 12 runs에서 selected Slip detector의 Support-window cross-trigger는 0이었다. 따라서 wrong provisional cause로 event recall을 보충한 case도 0이며, integrated detection 33/36은 native branch detection으로만 구성된다.

## 29. Terrain-vs-Slip timing

Holdout Slip 24/24가 target ICE output보다 먼저 발생했다. `t_slip-t_terrain_valid`의 finite distribution은 min -394 ms, p10 -262.6 ms, median -147 ms이며 일부 fall runs는 censor 전 target ICE output 자체가 없었다. Support 12/12는 Terrain output 뒤 발생했고 Terrain-to-Support margin은 min/p10/median/p95/max 23/23/612/616.9/618 ms다.

## 30. Reflex reaction latency

Selected detector는 Slip 24/24에서 Terrain보다 먼저 reflex를 발생시켰다. `t_detect-t_slip`은 min -28, median -27.5, p95 -27, max -27 ms다. 이는 50 mm/3 ms established Slip clock보다 직전의 causal sensor progression을 사용한 accepted early warning이며, -30 ms 이전 alert는 없었다. Support latency는 median/p95 -16/-15 ms다.

## 31. Recovered vs fall breakdown

Slip detector는 recovered 10/10과 fall 14/14를 모두 감지해 future fall/recovery outcome을 label로 사용하지 않은 contract와 일치한다. Frozen Support는 recovered 2/2, fall 7/10으로 holdout miss가 fall subgroup에 집중됐다.

## 32. Sensor-count tradeoff

Selected Slip과 frozen Support가 같은 Pelvis IMU6를 공유한다. Frozen Terrain LEFT_ONLY FSR4를 더하면 recommended installed set은 10 unique physical channels다.

| Branch | Input | Channels |
|---|---|---:|
| Terrain | Left FSR4 | 4 |
| Continuous Slip | Pelvis IMU6 | 6 |
| Frozen Support | shared Pelvis IMU6 | 6, duplicated install 0 |
| Total unique | Left FSR4 + Pelvis IMU6 | 10 |

Foot IMU는 추가되지 않았다. `FINAL_SENSOR_ARCHITECTURE_FROZEN`은 선언하지 않는다.

## 33. Historical architecture comparison

| Architecture | Result |
|---|---|
| Terrain-gated Slip | causally too late; 96/96 Slip before ICE output |
| Sand-conditioned Support | validation supported; one-shot holdout recall 75% |
| Continuous Slip | validation and holdout fully supported with existing Pelvis IMU6 |
| Final asymmetric system | integrated gates PASS, but frozen Support branch holdout gate FAIL |

Historical report와 result를 수정하지 않았다.

## 34. Limitations

Evidence는 current G1 policy, MuJoCo engineering terrain과 50 mm established-Slip oracle에 한정된다. Slip corpus는 bilateral 105와 left-only 15이며 right-only가 없어 right-only robustness를 주장할 수 없다. Detector onset이 `t_slip`보다 27–28 ms 빠른 것은 accepted sensor precursor evidence이지 universal physical Slip onset 정의가 아니다. Frozen Support는 validation에서 선택됐지만 이번 holdout recall이 75%였고, 이 milestone에서는 재학습·threshold 변경을 금지했다.

## 35. Verdict

`CONTINUOUS_SLIP_REFLEX_PROMISING`

Continuous Slip branch 자체는 existing Pelvis IMU6만으로 validation과 holdout을 완전히 통과했고 Terrain보다 24/24 먼저 reflex를 만들었다. 그러나 final asymmetric architecture는 frozen Support holdout recall 75%가 predeclared 85% gate에 미달해 supported verdict를 사용할 수 없다.

전체 regression suite는 202 tests PASS, 1 environment-dependent policy smoke SKIP, failure 0이다. 신규/인접 reflex tests 30개, formatting, pyflakes, diff whitespace와 viewer/simulator parity regression이 PASS했다.

## 36. Next recommendation

다음 bounded milestone이 승인된다면 frozen Support branch의 holdout misses 세 case를 failure analysis 대상으로 삼아야 한다. 이번 결과를 보고 Slip threshold, Slip oracle, Foot IMU, q/dq/torque 또는 추가 architecture를 자동으로 확장하지 않는다. E84 feasibility와 final sensor freeze도 시작하지 않는다.
