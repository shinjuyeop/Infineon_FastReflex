# Terrain + Walking Stability Integrated Sanity

Milestone: `TERRAIN_STABILITY_INTEGRATED_SANITY`

## 1. Architecture change rationale

Control-facing requirement를 terrain-conditioned instability로 재정의했다. `NORMAL/SLIP/SINK`를 하나의 runtime AI가 직접 분류하는 대신 terrain recognition과 walking stability detection을 독립 producer로 유지하고 마지막에 deterministic fusion한다. `SLIP_RISK`와 `SINK_RISK`는 terrain context가 붙은 recovery advisory이며 물리적 Slip/Sink 원인의 causal diagnosis가 아니다.

```text
runtime sensor streams
  ├─ touchdown-centered Terrain producer ── terrain_state ─┐
  └─ continuous Stability producer ─────── stability_state ├─ fusion
                                                          └─ hazard_state + RECOVERY_REQUIRED
```

작업 시작 HEAD는 요청에 명시된 `3475c2cf050efed0eed2fe973461e5c339755492`와 실제 repository HEAD가 일치했고 worktree는 clean이었다. Experiment matrix, threshold, split과 acceptance는 [`20260827_terrain_stability_integrated_sanity.yaml`](../configs/experiment/20260827_terrain_stability_integrated_sanity.yaml)에 simulation 전에 고정했다.

## 2. Historical evidence preserved

다음 evidence와 기존 report/dataset은 삭제, 수정 또는 소급 relabel하지 않았다.

- frozen Slip `50 mm + 3 ms`
- historical outcome-based Sink
- penetration-spread와 support-loss oracle 결과
- deformable-support Sink proxy와 `sink_observability_20260827`
- `SINK_SENSOR_OBSERVABILITY_PROMISING`
- 기존 Pilot datasets, checkpoints와 reports

이번 report에서 Slip/Sink clock은 scenario physics diagnostic일 뿐 stability label이나 primary Fast Reflex latency reference가 아니다. `physical disturbance clock != walking instability clock != runtime detection clock`을 유지한다.

## 3. Dual-stream runtime contract

Canonical state는 `terrain_state/valid/updated_at_us`, `stability_state/valid/updated_at_us`, `hazard_state`, `recovery_required`를 분리한다. Terrain update는 stability event와 독립이고 latest valid state를 hold한다. Stability producer는 terrain identity를 입력으로 받지 않는다.

Runtime-facing stability input은 1 kHz pelvis IMU6 `[accel xyz, gyro xyz]`뿐이다. COM, XCoM, support polygon, contact, terrain GT, physical Slip/Sink clock, future fall과 scenario name은 runtime detector input에 없다. MuJoCo exact state는 별도 `StabilityDiagnostics`에만 존재한다.

## 4. Terrain runtime migration status

Verdict: `TERRAIN_RUNTIME_MODEL_PENDING`.

Legacy repository `/d/shin/Infineon` HEAD `4194af1e0d8db8d113609c11879713c29a583261`를 read-only로 audit했다. Frozen Terrain v4 evidence는 다음과 같다.

| Item | Audited value |
|---|---|
| Float artifact | `simulation/outputs/terrain_static_reference_v4/selected_model.keras` |
| Float SHA-256 | `adcfa113b679327dc5bac0d4df7dcfceeb2f4dd703960665ecc491581d77df3b` |
| Strict INT8 artifact | `gap_50_seed_20260921_strict_int8.tflite` |
| INT8 SHA-256 | `27d4da4d30c012307c895ea73636c6a17fa2bbb36d9507c0293d2a7fd7f4c943` |
| Class order | Concrete, Marble, Ice, Sand |
| Input | 50 samples × Fusion10, 1 kHz |
| Channels | left foot FSR4 + left foot/ankle accel3/gyro3 |
| Float test | accuracy 0.9514, macro F1 0.9517 |
| INT8 test | accuracy 0.9490, macro F1 0.9494 |

현재 clean repository는 pelvis IMU6와 bilateral virtual FSR8만 제공하고 legacy left-foot/ankle IMU6 runtime contract와 TensorFlow/TFLite dependency가 없다. Exact input/sensor/runtime parity를 보장할 수 없어 artifact를 복사하지 않았다. Integrated plumbing은 exact terrain identity를 `ORACLE_PROXY`로 명시해 사용한다. Proxy contact-to-valid 0 ms는 AI terrain latency가 아니다.

## 5. Frozen scenario design

44개 deterministic 8초 run을 실행했다. Ice는 frozen friction/profile을, Sand는 frozen deformable-support mechanics를 그대로 사용했다. 결과 뒤 friction, support stiffness/damping/travel, patch matrix, intended role, split, MoS quantile, 10 mm margin, 20 ms persistence 또는 IMU rule threshold를 수정하지 않았다.

| Cohort | Runs | Intended design |
|---|---:|---|
| Concrete/Marble stable calibration | 6 | 0.15/0.20/0.25 m/s uniform hard terrain |
| Ice stable-intended | 8 | 0.25 m/s, frozen Ice, predeclared position/width |
| Ice fall-intended | 10 | 0.15/0.20 m/s, frozen Ice, position/width |
| Sand stable-intended | 8 | mild balanced deformable, side/speed/position |
| Sand fall-intended | 12 | moderate lateral/balanced deformable, position/width |

## 6. Stable/fall and prefix coverage

전체 observed outcome은 non-fall `11`, fall `33`이었다. Hard controls는 `6/6` stable이었다. Ice stable-intended는 `1/8`, Sand stable-intended는 `4/8`만 non-fall이었다. Fall-intended는 Ice `10/10`, Sand `12/12` fall이었지만 12개 run은 target terrain contact 전에 이미 fall했다. 모든 runtime IMU는 finite였고 transition contact가 누락된 Ice/Sand run은 없었다.

Predeclared scenario acceptance는 terrain별 observed stable 최소 4, fall-intended observed-fall fraction 최소 0.70, pre-transition fall 0이었다. Ice stable coverage `1/8`과 pre-transition fall `12` 때문에 scenario gate는 FAIL이다. Intended name이 observed outcome을 강제하지 않았다.

## 7. Exact stability ground truth

Pelvis-root robot subtree의 mass-weighted whole-body COM과 `mj_subtreeVel` COM velocity를 1 kHz endpoint에서 읽었다. Exact loaded-contact state로 left single, right single, double support를 구분하고 active foot의 네 named sole point 또는 양발 여덟 point convex hull을 support polygon으로 만들었다.

```text
omega0 = sqrt(9.81 / (COM_z - mean_support_z))
XCoM_xy = COM_xy + COM_velocity_xy / omega0
raw_MoS = signed Euclidean distance(XCoM, support polygon)
```

No-support는 diagnostic-only이고 raw MoS를 만들지 않는다. Positive margin은 polygon inside, negative는 outside다. Exact calculation, velocity causality, left/right mapping, convex hull, XCoM와 signed margin을 unit test로 검증했다.

## 8. Phase-aware normal gait envelope

Predeclared Concrete/Marble stable six run만 사용해 phase별 0.5 percentile lower bound를 fit했다. Fall, Ice/Sand outcome, IMU와 future sample은 calibration에 사용하지 않았다.

| Phase | Lower bound (m) |
|---|---:|
| Left single support | -0.141066 |
| Right single support | -0.119620 |
| Double support | 0.046000 |

Primary candidate는 `raw_MoS - phase_lower_bound < -0.010 m`가 20 consecutive 1 kHz samples 지속되는 시각이다. 이는 real-world universal threshold가 아니라 사전 고정한 MuJoCo/G1 sanity criterion이다.

## 9. t_instability acceptance

Exact-state oracle은 acceptance를 통과하지 못했다.

| Metric | Required | Observed |
|---|---:|---:|
| Stable run firing | ≤ 5% | 5/11 = 45.45% |
| Fall coverage | ≥ 80% | 22/33 = 66.67% |
| Ice fall coverage | detection present | 13/17 |
| Sand fall coverage | detection present | 9/16 |
| Median accepted t_instability→fall lead | ≥ 100 ms | 1,870.5 ms |
| p95 accepted lead | report | 3,595.65 ms |
| Minimum accepted lead | report | 2 ms |

Median lead는 충분했지만 stable FP와 coverage가 primary gates를 크게 위반했다. Stable false firing은 유일한 observed Ice stable run과 observed Sand stable four run 모두에서 발생했다. 일부 fall은 `t_instability` 전에 censor됐거나 target transition 전에 넘어졌다. Threshold를 사후 수정하지 않았고 `STABILITY_GROUND_TRUTH_NEEDS_REVISION` 조건도 성립한다.

## 10. Full oracle table

시간 단위는 ms다. Negative lead는 computed `t_instability`가 fall 뒤여서 unsupported임을 뜻한다. Slip/Sink column은 secondary physical diagnostic onset이다.

| Scenario | Terrain | Speed | Intended | Fall | Transition | t_instability | t_fall | Lead | Slip diag | Sink diag |
|---|---|---:|---|:---:|---:|---:|---:|---:|---:|---:|
| `concrete_stable_s015` | CONCRETE | 0.15 | stable | N | — | — | — | — | — | — |
| `concrete_stable_s020` | CONCRETE | 0.20 | stable | N | — | — | — | — | — | — |
| `concrete_stable_s025` | CONCRETE | 0.25 | stable | N | — | — | — | — | — | — |
| `marble_stable_s015` | MARBLE | 0.15 | stable | N | — | — | — | — | — | — |
| `marble_stable_s020` | MARBLE | 0.20 | stable | N | — | — | — | — | — | — |
| `marble_stable_s025` | MARBLE | 0.25 | stable | N | — | — | — | — | — | — |
| `ice_stable_s025_p030_w065` | ICE | 0.25 | stable | Y | 1221 | 2042 | 4411 | 2369 | 1911 | — |
| `ice_stable_s025_p030_w075` | ICE | 0.25 | stable | Y | 1221 | 2042 | 4531 | 2489 | 1911 | — |
| `ice_stable_s025_p032_w075` | ICE | 0.25 | stable | Y | 1221 | 2042 | 4871 | 2829 | 1911 | — |
| `ice_stable_s025_p035_w065` | ICE | 0.25 | stable | Y | 1221 | 2040 | 3429 | 1389 | 1915 | — |
| `ice_stable_s025_p035_w075` | ICE | 0.25 | stable | N | 1221 | 2040 | — | — | 1915 | — |
| `ice_stable_s025_p038_w075` | ICE | 0.25 | stable | Y | 1231 | 1324 | 1332 | 8 | — | — |
| `ice_stable_s025_p040_w065` | ICE | 0.25 | stable | Y | 1759 | 1516 | 1319 | -197 | — | — |
| `ice_stable_s025_p040_w075` | ICE | 0.25 | stable | Y | 1759 | 1516 | 1319 | -197 | — | — |
| `ice_fall_s015_p030_w075` | ICE | 0.15 | fall | Y | 2060 | 1778 | 2269 | 491 | 2114 | — |
| `ice_fall_s015_p032_w065` | ICE | 0.15 | fall | Y | 1803 | 2904 | 3492 | 588 | 2934 | — |
| `ice_fall_s015_p035_w075` | ICE | 0.15 | fall | Y | 1803 | 3220 | 5133 | 1913 | 3088 | — |
| `ice_fall_s015_p038_w065` | ICE | 0.15 | fall | Y | 2360 | 2088 | 3916 | 1828 | 2431 | — |
| `ice_fall_s015_p040_w075` | ICE | 0.15 | fall | Y | 2099 | 2182 | 2184 | 2 | — | — |
| `ice_fall_s020_p030_w075` | ICE | 0.20 | fall | Y | 1763 | 1311 | 1314 | 3 | — | — |
| `ice_fall_s020_p032_w065` | ICE | 0.20 | fall | Y | 1504 | 2620 | 5294 | 2674 | 2493 | — |
| `ice_fall_s020_p035_w075` | ICE | 0.20 | fall | Y | 1505 | 2620 | 5087 | 2467 | 2634 | — |
| `ice_fall_s020_p038_w065` | ICE | 0.20 | fall | Y | 1725 | 1775 | 1621 | -154 | — | — |
| `ice_fall_s020_p040_w075` | ICE | 0.20 | fall | Y | 1904 | 1640 | 1621 | -19 | — | — |
| `sand_stable_balanced_left_s015_p035` | SAND | 0.15 | stable | N | 1803 | 3534 | — | — | — | — |
| `sand_stable_balanced_right_s015_p035` | SAND | 0.15 | stable | N | 2096 | 3874 | — | — | — | — |
| `sand_stable_balanced_left_s025_p035` | SAND | 0.25 | stable | N | 1221 | 1513 | — | — | — | — |
| `sand_stable_balanced_right_s025_p035` | SAND | 0.25 | stable | N | 1508 | 1836 | — | — | — | — |
| `sand_stable_balanced_left_s020_p030` | SAND | 0.20 | stable | Y | 1211 | 4135 | 2157 | -1978 | — | — |
| `sand_stable_balanced_right_s020_p030` | SAND | 0.20 | stable | Y | 2123 | 1311 | 1314 | 3 | — | — |
| `sand_stable_balanced_left_s020_p040` | SAND | 0.20 | stable | Y | 2457 | 1640 | 1621 | -19 | — | — |
| `sand_stable_balanced_right_s020_p040` | SAND | 0.20 | stable | Y | 1904 | 1640 | 1621 | -19 | — | — |
| `sand_fall_lateral_left_s025_p030_w065` | SAND | 0.25 | fall | Y | 1221 | 2926 | 4231 | 1305 | — | 3038 |
| `sand_fall_lateral_left_s025_p030_w075` | SAND | 0.25 | fall | Y | 1221 | 2886 | 5561 | 2675 | — | 2482 |
| `sand_fall_lateral_left_s025_p035_w065` | SAND | 0.25 | fall | Y | 1221 | 2886 | 4219 | 1333 | — | 2482 |
| `sand_fall_lateral_left_s025_p035_w075` | SAND | 0.25 | fall | Y | 1221 | 2898 | 5567 | 2669 | — | 2476 |
| `sand_fall_lateral_left_s025_p040_w065` | SAND | 0.25 | fall | Y | 1759 | 1516 | 1319 | -197 | — | — |
| `sand_fall_lateral_left_s025_p040_w075` | SAND | 0.25 | fall | Y | 1759 | 1516 | 1319 | -197 | — | — |
| `sand_fall_balanced_left_s025_p030` | SAND | 0.25 | fall | Y | 1221 | 1509 | 4315 | 2806 | — | — |
| `sand_fall_balanced_right_s025_p030` | SAND | 0.25 | fall | Y | 1508 | 1806 | 1869 | 63 | — | — |
| `sand_fall_balanced_left_s025_p035` | SAND | 0.25 | fall | Y | 1221 | 1509 | 5145 | 3636 | — | — |
| `sand_fall_balanced_right_s025_p035` | SAND | 0.25 | fall | Y | 1508 | 1806 | 5723 | 3917 | — | — |
| `sand_fall_balanced_left_s025_p040` | SAND | 0.25 | fall | Y | 1759 | 1516 | 1319 | -197 | — | — |
| `sand_fall_balanced_right_s025_p040` | SAND | 0.25 | fall | Y | 2113 | 1516 | 1319 | -197 | — | — |

## 11. Deterministic pelvis-IMU rule

Stable TRAIN controls `concrete/marble × 0.15/0.20 m/s`만으로 feature envelope를 fit했다.

| Feature | 99.5 percentile threshold |
|---|---:|
| roll/pitch gyro magnitude | 0.812744 rad/s |
| horizontal acceleration magnitude | 6.895642 m/s² |
| total acceleration deviation from stable median | 10.215044 m/s² |

Stable acceleration norm center는 9.562903 m/s²다. Any-feature abnormal이 20 ms 지속되면 UNSTABLE, normal이 50 ms 지속되면 reset한다. Future smoothing, zero-phase filter, exact state와 terrain identity는 쓰지 않았다.

## 12. Rule holdout result

Predeclared holdout에서 exact oracle이 fall 전에 supported된 positive는 Ice 2, Sand 2였다. Rule은 결국 4/4를 감지했지만 두 Ice run에서 `t_instability` 전 false onset도 있었다.

| Metric | Result |
|---|---:|
| Stable FP run rate | 2/3 = 66.67% |
| Stable FP duration | Ice stable 3,907 ms; Sand stable 51 ms |
| Supported instability recall | 4/4 = 100% |
| Recall @0/+10/+20/+50/+100 ms | 0/0/0/0/0 |
| Median / p95 latency | 353.5 / 541.75 ms |
| Median lead before fall | 2,274 ms |
| p05 lead before fall | 1,682.95 ms |
| Pre-instability FP positive runs | 2/4 |

Per-run supported detection latency는 Ice 331/212 ms, Sand lateral 376 ms, Sand balanced 571 ms였다. Rule complexity는 three scalar thresholds, abnormal/reset counters와 one binary state지만 false-positive와 latency가 Fast Reflex baseline으로 부적합하다.

## 13. Optional AI stability baseline

Not performed. Scenario gate와 exact stability ground-truth gate가 모두 실패했으므로 config의 fail-closed rule에 따라 GRU 50/100 ms × 3 seeds training, validation selection과 holdout access를 시작하지 않았다. 따라서 rule-vs-AI verdict는 `NOT_COMPARABLE`, AI parameter/latency/FP 주장은 없다.

## 14. Latency semantics

Primary latency reference는 `t_instability`다. Physical Slip/Sink onset, terrain contact, fall outcome을 runtime detector latency anchor로 대체하지 않았다. Exact oracle이 acceptance를 실패했으므로 rule과 fusion latency는 diagnostic-only이고 system readiness evidence가 아니다.

## 15. Terrain timing

Hard control은 첫 exact touchdown에서 Concrete/Marble proxy state를 valid로 만들고 hold했다. Ice/Sand는 first named patch contact에서 target terrain proxy를 valid로 갱신했다. Proxy update는 같은 sampled endpoint이므로 plumbing latency 0 ms지만 실제 frozen Terrain model의 recognition latency가 아니다. Terrain AI가 integrated됐다는 주장을 하지 않는다.

## 16. Fusion truth table

Unit-tested deterministic fusion은 다음을 만족한다.

| Terrain | Stability | Hazard | Recovery required |
|---|---|---|:---:|
| Any | STABLE | NORMAL | false |
| ICE | UNSTABLE | SLIP_RISK | true |
| SAND | UNSTABLE | SINK_RISK | true |
| CONCRETE/MARBLE | UNSTABLE | GENERIC_INSTABILITY | true |
| UNKNOWN | UNSTABLE | GENERIC_INSTABILITY | true |

Terrain producer update는 stability state를 바꾸지 않고 stability producer update는 held terrain을 바꾸지 않는다. Terrain invalid가 recovery assertion을 막지 않는다.

## 17. Integrated latency diagnostic

Oracle-supported, pre-fall measurable cases에서 rule-based fusion/recovery latency median/p95는 325.5/583.3 ms였다. Rule detection 시 target terrain proxy가 이미 valid였던 diagnostic count는 22, 아직 target terrain이 valid하지 않았던 count는 13이었다. 후자는 transition 전 controller/fall 문제를 포함하므로 target-terrain fusion evidence로 받아들이면 안 된다. Terrain-already-known case는 rule detect가 fusion ready를 지배하고, terrain-not-yet-valid case는 max rule/terrain timing semantics를 구현한다.

## 18. Viewer/status integration

Canonical `evaluate` flow는 Concrete stable, Ice stable-intended, Ice fall, Sand stable, Sand fall representative의 timestamp-synchronized terminal status를 생성했다. `simulate --status-calibration <calibration.json>`으로 동일 formatting/replay를 canonical simulate flow에서 접근할 수 있다. Formatter는 render/physics state를 쓰지 않으며 unit test에서 qpos/qvel 불변을 확인했다.

Status 자체가 failure를 숨기지 않았다. Observed Ice stable run은 rule이 exact oracle보다 26 ms 먼저 false-UNSTABLE이 됐고, observed Sand stable controls는 exact oracle 자체가 fire했다. Representative Sand fall은 `SAND + UNSTABLE → SINK_RISK`, recovery true를 표시했다.

## 19. Sensor implications

- Stability pelvis IMU6 sufficiency: `NO` for the current criterion/baseline. Rule stable FP 66.67%와 Recall@100 ms 0이므로 충분하다고 볼 수 없다.
- 이는 q/dq, torque, FSR 또는 foot IMU를 stability input에 즉시 추가하라는 근거가 아니다. Scenario와 exact ground truth부터 수정해야 한다.
- Terrain reference requirement는 legacy frozen contract의 left foot FSR4 + foot/ankle IMU6다. Current repository와 parity migration은 pending이다.
- Potential integrated sensor architecture와 channel count는 freeze하지 않는다.

## 20. Limitations

- Frozen policy/fixed initial condition의 deterministic 44-run sanity이며 broader locomotion distribution이 아니다.
- Position variation 중 여러 run이 target patch 전에 fall해 scenario geometry/controller prefix contract가 깨졌다.
- Phase-only MoS envelope는 hard stable controls와 Ice/Sand stable controls 사이의 gait/support-domain shift를 흡수하지 못했다.
- Stable Sand에서 MoS false firing이 발생해 `t_instability` label을 AI target으로 사용할 수 없다.
- Terrain path는 ORACLE_PROXY이며 actual classifier, foot sensor stream과 model runtime이 없다.
- Terminal status replay는 state plumbing 검증이지 GUI human-factor validation이 아니다.
- MuJoCo terrain/deformable mechanics는 engineering proxy이며 real-world stability threshold가 아니다.

## 21. Verdict

Primary verdict: `INTEGRATED_SCENARIO_NEEDS_REVISION`.

Scenario gate가 먼저 실패했고 exact-state oracle도 독립적으로 실패했다. Fusion implementation과 truth table은 PASS지만 integrated readiness는 아니다. AI를 생략했고 architecture를 deployment-ready 또는 sensor-frozen으로 승격하지 않는다.

Generated artifact는 Gitignored `artifacts/runs/20260827_terrain_stability_integrated_sanity/`에 있으며 final run의 `results.json` SHA-256은 `0ab918a61a203e81ce6581838166eca5fde617c3e853b9d8ab24cb179db69573`이다.

## 22. Next recommendation

다음 승인된 milestone은 새 model search가 아니라 scenario/oracle revision이어야 한다.

1. Patch position variation이 target contact 전 locomotion을 바꾸는 이유를 exact scene/controller prefix parity로 분리한다.
2. Hard-prefix가 bit/metric-equivalent하고 fall-free인 candidate만 결과를 보지 않는 새 bounded matrix에 사전 선언한다.
3. Stable Ice/Sand controls를 calibration domain에 포함할 수 있는 phase/contact representation을 설계하되 현재 threshold를 사후 sweep하지 않는다.
4. Revised exact clock이 stable FP ≤5%, fall coverage ≥80%, median lead ≥100 ms를 통과한 뒤에만 동일 pelvis IMU6 rule과 optional GRU를 비교한다.

Recovery controller, final sensor freeze, Full Dataset, CNN/LSTM sweep, quantization, E84와 HIL은 시작하지 않는다.
