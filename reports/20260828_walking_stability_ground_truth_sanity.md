# Walking Stability Ground-Truth Sanity

Milestone: `WALKING_STABILITY_GROUND_TRUTH_SANITY`

## 1. Purpose

이 실험의 유일한 목적은 calibrated terrain transition에서 실제 fall trajectory가 observed-stable walking의 동적 안정 범위를 지속적으로 벗어나는 최초 causal simulator-only clock `t_instability`를 검증하는 것이었다. Runtime detector, Pelvis IMU threshold, Stability AI, sensor augmentation, recovery와 E84 작업은 수행하지 않았다.

작업 시작 시 실제 `HEAD`와 `origin/main`은 모두 `d7f846909ae4c5a3de7e24133b7d8b50513d4027`이었고 worktree는 clean이었다. Experiment matrix와 frozen rule은 [`20260828_walking_stability_ground_truth_sanity.yaml`](../configs/experiment/20260828_walking_stability_ground_truth_sanity.yaml)에 simulation 전에 기록했다.

## 2. Why old ground truth failed

Historical integrated sanity의 phase envelope는 Concrete/Marble hard-ground stable 6 runs만 사용했다. 그 결과 stable false firing `5/11`, fall coverage `22/33`, verdict `STABILITY_GROUND_TRUTH_NEEDS_REVISION`이었다. 이 결과와 당시 lower bounds(Left `-0.141066 m`, Right `-0.119620 m`, Double `0.046000 m`)는 소급 수정하지 않았다.

이전 normal 정의는 recoverable Ice Slip과 recoverable Sand deformation을 포함하지 않아 실제 controller가 버틴 상태를 abnormal로 만들었다. 이번 실험은 threshold를 바꾸지 않고 normal-envelope evidence만 올바른 의미로 재구성했다.

## 3. Why clean transition scenarios change the experiment

`TRANSITION_SCENARIOS_CALIBRATED`에서 Concrete/Marble→Ice/Sand matched-prefix parity, target contact, finite run, normal prefix, pre-transition fall 0과 stable/fall operating domains가 검증됐다. 따라서 target contact 전 trajectory와 B-side disturbance를 혼동하지 않고 stability clock을 평가할 수 있다.

Calibration은 기존 clean selected conditions를 재실행했고 36/36 outcome이 prior observed result와 일치했다. Fresh validation은 calibration에 없는 source-inclusive physical signatures 40개를 frozen operating domains 안에서 사용했다. Friction, Sand mechanics, controller와 policy는 바꾸지 않았다.

## 4. Stability semantics

Walking stability의 `STABLE`은 hard ground에서 작은 움직임만 보이는 상태가 아니라 robot/controller가 실제로 넘어지지 않고 통과한 전체 trajectory다.

- Ice stable의 physical Slip은 stable evidence다.
- Sand stable의 약 20 mm deformation은 stable evidence다.
- 한 fresh moderate Sand condition은 Sink diagnostic과 약 40 mm deformation이 있었지만 실제 non-fall이어서 stable denominator에 포함됐다.
- Terrain identity, Slip occurrence와 Sink occurrence는 instability label이 아니다.

Fall은 calibration cohort와 validation outcome을 정하는 데만 사용했다. Sample-time oracle은 future fall, fall time 또는 post-fall state를 입력으로 받지 않는다.

## 5. Calibration cohort

### ORACLE CALIBRATION

36개 deterministic 8 s run을 재실행했다. 실제 `VALID_STABLE` 20개만 phase envelope에 사용했고 `VALID_FALL` 16개는 paired analysis에서만 비교했다.

| Observed-stable evidence | Runs used |
|---|---:|
| Concrete uniform hard | 3 |
| Marble uniform hard | 3 |
| Concrete→Ice | 3 |
| Marble→Ice | 3 |
| Concrete→Sand | 4 |
| Marble→Sand | 4 |
| Total | 20 |

Calibration fall comparison은 Concrete/Marble 각각 Ice 4, Sand 4로 총 16개다. Fall run ID는 normal fit ID에 하나도 포함되지 않았다. Intended role 대신 재실행한 observed outcome을 사용했으며 calibration invalid와 pre-transition firing은 모두 0이었다.

## 6. Stable transition envelope

Stable Ice 6개 모두 physical Slip diagnostic이 있었고 stable Sand 8개는 mild deformable support에서 약 20 mm deformation이 있었다. 이 상태 전체를 hard stable 6개와 함께 normal evidence로 pool했다. Primary envelope는 terrain별로 나누지 않은 하나의 contract다.

이 변경으로 single-support lower bound가 historical hard-only bound보다 크게 낮아졌다. 이는 recoverable transition 중 XCoM이 sole polygon 밖으로 크게 이동한 상태까지 normal range에 포함됐다는 뜻이지, Ice/Sand 자체를 instability로 취급했다는 뜻이 아니다.

## 7. XCoM / MoS definition

각 1 kHz sample의 simulator exact state만 사용했다.

```text
h = COM_z - current_loaded_support_height
omega0 = sqrt(9.81 / h)
XCoM_xy = COM_xy + COM_velocity_xy / omega0
raw_mos_m = signed_distance(XCoM_xy, current support polygon)
```

Whole-body COM/velocity는 pelvis-root robot subtree에서 읽고, exact loaded left/right sole의 named four-point geometry로 current convex hull을 만들었다. Positive MoS는 inside, negative는 outside다. Nonphysical `h <= 0`, no-support와 계산 불가능 sample은 `NaN`으로 fail-closed하며 임의의 0 margin을 만들지 않는다.

## 8. Phase-aware normal envelope

Observed-stable 20개 run의 finite sample을 support phase별로 모아 predeclared linear `0.5 percentile` lower bound를 계산했다.

| Phase | Frozen lower bound (m) |
|---|---:|
| Left single support | -0.302386 |
| Right single support | -0.304408 |
| Double support | -0.054176 |

No-support는 diagnostic-only다. Source terrain과 target terrain 어느 것도 envelope lookup key가 아니다.

## 9. Frozen instability rule

```text
stability_residual_m = raw_mos_m - phase_normal_lower_bound

candidate = stability_residual_m < -0.010 m
t_instability = first sample where candidate has persisted for 20 consecutive samples
```

Transition run에서는 first exact B-terrain contact 전 candidate를 primary clock에서 mask하고 persistence count를 contact에서 새로 시작했다. 별도 ungated replay로 pre-transition firing을 기록했다. Calibration freeze SHA-256은 `2a972952c2131301d034c7443d85d2978e8ca87f991b6c679e7d946cf6561bb8`이며 fresh validation 전후 동일했다. Fresh result를 보고 quantile, 10 mm margin 또는 20 ms persistence를 수정하지 않았다.

## 10. Ice stable vs fall

Calibration paired diagnostics는 threshold selection에 사용하지 않았다.

| Ice observed outcome | Runs | Median per-run min raw MoS (m) | Median per-run min residual (m) | Any oracle firing | Physical Slip |
|---|---:|---:|---:|---:|---:|
| Stable | 6 | -0.389699 | -0.086302 | 2/6 | 6/6 |
| Fall | 8 | -0.312292 | -0.007885 | 8/8 | 8/8 |

Minimum excursion만 보면 stable Ice가 fall Ice보다 오히려 더 negative했다. Persistence가 일부 transient difference를 제거했지만 phase-only scalar MoS가 trajectory outcome을 단조롭게 분리하지 않는다는 calibration warning이었다. Rule은 그대로 freeze했다.

## 11. Sand stable vs fall

| Sand observed outcome | Runs | Median per-run min raw MoS (m) | Median per-run min residual (m) | Any oracle firing | Sink diagnostic |
|---|---:|---:|---:|---:|---:|
| Stable | 8 | -0.237964 | 0.027271 | 2/8 | 0/8 |
| Fall | 8 | -0.251512 | 0.051445 | 3/8 | 8/8 |

Sand fall의 pre-fall residual은 대부분 frozen lower bound 아래로 10 mm 더 악화되지 않았다. Stronger deformation/Sink와 fall outcome이 있었어도 XCoM-to-current-deformed-support-polygon residual 하나로는 instability clock이 형성되지 않았다.

## 12. Fresh validation design

### FRESH ORACLE VALIDATION

Source Concrete/Marble × target Ice/Sand × design stable/fall의 8 groups에 각각 5개, 총 40개를 사전 고정했다. Width는 calibration-unused values였고 start, speed, mechanics, policy는 frozen domains 안에 있었다. Source를 포함한 physical signature 중복과 calibration overlap은 0이었다.

Scenario gate 결과 target contact, finite simulation, minimum normal prefix와 post-contact observation을 모두 만족한 valid run은 40/40, invalid 0, pre-transition fall 0이었다. Observed outcome은 다음과 같았다.

| Source→target | Stable | Fall |
|---|---:|---:|
| Concrete→Ice | 5 | 5 |
| Concrete→Sand | 6 | 4 |
| Marble→Ice | 8 | 2 |
| Marble→Sand | 5 | 5 |
| Total | 24 | 16 |

Design-fall 중 Concrete→Sand 1개와 Marble→Ice 3개가 actual stable이었으므로 observed-stable denominator로 이동했다.

## 13. Stable false positives

Observed stable 24개 중 5개에서 contact 이후 `t_instability`가 발생해 false-instability run rate는 `20.83%`였다. Gate `≤10%`를 실패했다.

| Run | A→B | Speed | Contact (ms) | Slip/Sink diagnostic | Min raw MoS (m) | Min residual (m) | t_instability (ms) | False firing | Fall |
|---|---|---:|---:|---|---:|---:|---:|:---:|:---:|
| `val_c_ice_s01` | Concrete→Ice | 0.25 | 1221 | Slip 2190 | -0.351912 | -0.049526 | — | N | N |
| `val_c_ice_s02` | Concrete→Ice | 0.25 | 1221 | Slip 2190 | -0.351912 | -0.049526 | — | N | N |
| `val_c_ice_s03` | Concrete→Ice | 0.25 | 1221 | Slip 2190 | -0.351912 | -0.049526 | — | N | N |
| `val_c_ice_s04` | Concrete→Ice | 0.25 | 1221 | Slip 2190 | -0.351912 | -0.049526 | — | N | N |
| `val_c_ice_s05` | Concrete→Ice | 0.25 | 1221 | Slip 2190 | -0.351912 | -0.049526 | — | N | N |
| `val_c_sand_s01` | Concrete→Sand | 0.25 | 1221 | Deform 20.17 mm | -0.320118 | -0.023153 | — | N | N |
| `val_c_sand_s02` | Concrete→Sand | 0.25 | 1221 | Deform 20.17 mm | -0.322538 | -0.023098 | — | N | N |
| `val_c_sand_s03` | Concrete→Sand | 0.25 | 1508 | Deform 20.14 mm | -0.150895 | 0.078842 | — | N | N |
| `val_c_sand_s04` | Concrete→Sand | 0.25 | 1508 | Deform 20.14 mm | -0.150895 | 0.078842 | — | N | N |
| `val_c_sand_s05` | Concrete→Sand | 0.25 | 1508 | Deform 20.14 mm | -0.150895 | 0.078842 | — | N | N |
| `val_c_sand_f03` | Concrete→Sand | 0.25 | 1221 | Sink 2476 / 40.14 mm | -0.247735 | 0.049348 | — | N | N |
| `val_m_ice_s01` | Marble→Ice | 0.25 | 1221 | Slip 2187 | -0.395289 | -0.092903 | — | N | N |
| `val_m_ice_s02` | Marble→Ice | 0.25 | 1221 | Slip 2187 | -0.395289 | -0.092903 | 5036 | Y | N |
| `val_m_ice_s03` | Marble→Ice | 0.25 | 1221 | Slip 2187 | -0.395289 | -0.092903 | 5017 | Y | N |
| `val_m_ice_s04` | Marble→Ice | 0.25 | 1221 | Slip 2187 | -0.395289 | -0.092903 | — | N | N |
| `val_m_ice_s05` | Marble→Ice | 0.25 | 1221 | Slip 2187 | -0.395289 | -0.092903 | 5279 | Y | N |
| `val_m_ice_f03` | Marble→Ice | 0.15 | 1803 | Slip 3085 | -0.384109 | -0.079701 | — | N | N |
| `val_m_ice_f04` | Marble→Ice | 0.15 | 1803 | Slip 3085 | -0.384109 | -0.079701 | — | N | N |
| `val_m_ice_f05` | Marble→Ice | 0.15 | 1803 | Slip 3085 | -0.384109 | -0.079701 | — | N | N |
| `val_m_sand_s01` | Marble→Sand | 0.25 | 1221 | Deform 20.19 mm | -0.365780 | -0.061373 | 4696 | Y | N |
| `val_m_sand_s02` | Marble→Sand | 0.25 | 1221 | Deform 20.18 mm | -0.365041 | -0.060634 | 4697 | Y | N |
| `val_m_sand_s03` | Marble→Sand | 0.25 | 1510 | Deform 20.18 mm | -0.158735 | 0.076276 | — | N | N |
| `val_m_sand_s04` | Marble→Sand | 0.25 | 1510 | Deform 20.18 mm | -0.158735 | 0.076276 | — | N | N |
| `val_m_sand_s05` | Marble→Sand | 0.25 | 1510 | Deform 20.18 mm | -0.160320 | 0.076276 | — | N | N |

## 14. Fall coverage

Observed fall 16개 중 valid pre-fall `t_instability`는 Concrete→Ice 3개뿐이었다. Overall coverage `3/16 = 18.75%`로 gate `≥85%`를 실패했다. Fall 뒤의 firing은 success로 세지 않았다.

| Run | A→B | Speed | Contact (ms) | Slip/Sink (ms) | t_instability (ms) | t_fall (ms) | Fall lead (ms) | Valid detection |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| `val_c_ice_f01` | Concrete→Ice | 0.15 | 1803 | Slip 2934 | 3660 | 3492 | — | N, late |
| `val_c_ice_f02` | Concrete→Ice | 0.15 | 1803 | Slip 2934 | 3660 | 3492 | — | N, late |
| `val_c_ice_f03` | Concrete→Ice | 0.15 | 1803 | Slip 3088 | 4515 | 5168 | 653 | Y |
| `val_c_ice_f04` | Concrete→Ice | 0.15 | 1803 | Slip 3088 | 4515 | 5135 | 620 | Y |
| `val_c_ice_f05` | Concrete→Ice | 0.15 | 1803 | Slip 3088 | 4515 | 5133 | 618 | Y |
| `val_c_sand_f01` | Concrete→Sand | 0.25 | 1221 | Sink 3046 | 4000 | 3686 | — | N, late |
| `val_c_sand_f02` | Concrete→Sand | 0.25 | 1221 | Sink 3042 | — | 4225 | — | N |
| `val_c_sand_f04` | Concrete→Sand | 0.25 | 1221 | Sink 2476 | — | 5552 | — | N |
| `val_c_sand_f05` | Concrete→Sand | 0.25 | 1221 | Sink 2476 | — | 5563 | — | N |
| `val_m_ice_f01` | Marble→Ice | 0.15 | 1803 | Slip 2820 | 3652 | 3476 | — | N, late |
| `val_m_ice_f02` | Marble→Ice | 0.15 | 1803 | Slip 2820 | 3652 | 3476 | — | N, late |
| `val_m_sand_f01` | Marble→Sand | 0.25 | 1221 | Sink 3046 | — | 4219 | — | N |
| `val_m_sand_f02` | Marble→Sand | 0.25 | 1221 | Sink 3039 | 4539 | 4205 | — | N, late |
| `val_m_sand_f03` | Marble→Sand | 0.25 | 1221 | Sink 2475 | — | 5162 | — | N |
| `val_m_sand_f04` | Marble→Sand | 0.25 | 1221 | Sink 2475 | — | 5202 | — | N |
| `val_m_sand_f05` | Marble→Sand | 0.25 | 1221 | Sink 2475 | — | 5216 | — | N |

## 15. Ice/Sand coverage

| Terrain | Stable FP | Fall detection | Required | Result |
|---|---:|---:|---:|---|
| Ice | 3/13 = 23.08% | 3/7 = 42.86% | Fall ≥80% | FAIL |
| Sand | 2/11 = 18.18% | 0/9 = 0% | Fall ≥80% | FAIL |

Sand는 catastrophic coverage failure다. Physical Sink diagnostic은 fall 9/9에 존재했지만 primary MoS residual은 pre-fall에 threshold 아래로 지속되지 않았다. Sink를 primary rule에 추가하지 않았다.

## 16. Concrete/Marble robustness

| Source | Stable FP | Fall detection | Meaningful detection |
|---|---:|---:|:---:|
| Concrete | 0/11 = 0% | 3/9 = 33.33% | Y |
| Marble | 5/13 = 38.46% | 0/7 = 0% | N |

Source robustness gate도 실패했다. 같은 terrain-agnostic contract가 Concrete stable에는 specific했지만 Marble stable에는 과민했고 Marble fall에는 meaningful pre-fall onset을 만들지 못했다.

## 17. Fall lead distribution

Valid detection 3개에서 `fall_lead_ms = t_fall - t_instability`는 다음과 같다.

| Statistic | Lead (ms) |
|---|---:|
| Minimum | 618.0 |
| p10 | 618.4 |
| p50 / median | 620.0 |
| p95 | 649.7 |
| Maximum | 653.0 |

Median gate `≥200 ms`는 PASS다. 이 값은 runtime detector가 확보한 시간이 아니라 physical stability degradation clock부터 actual fall까지의 available physical horizon이다. Coverage가 18.75%이므로 전체 fall population의 horizon으로 일반화할 수 없다.

## 18. Causality

Result: `PASS`.

`val_c_ice_f01`에서 primary onset이 결정된 sample 3659(3660 ms) 이후 raw-MoS suffix를 synthetic stable values로 교체했다. Candidate, residual, active와 onset의 이미 계산된 prefix는 exact equality를 유지했다. Future fall metadata와 post-fall state는 oracle function signature에 존재하지 않는다. Unit test도 future suffix를 바꾼 두 trace의 결정 전 output identity를 확인한다.

Transition cleanliness도 PASS다. Fresh valid 40개와 calibration 36개 모두 ungated pre-transition false-instability run은 0이었다. Primary persistence는 first exact target contact에서 reset되므로 contact 이전 onset을 primary success로 만들 수 없다.

## 19. Failure-case diagnostics

Primary 실패는 단순히 lead가 짧은 문제가 아니다.

- Fresh Marble Ice stable 세 run은 약 5.0–5.3 s에 persistent false onset이 생겼고 Marble Sand stable 두 run은 약 4.7 s에 false onset이 생겼다.
- Sand fall 9개는 모두 Sink diagnostic과 약 40 mm deformation이 있었지만 pre-fall coverage는 0이었다. 일부는 fall 이후 firing했고 대부분은 firing이 전혀 없었다.
- Observed stable transition까지 포함하자 single-support normal bound가 약 `-0.30 m`로 넓어졌다. 이 bound는 recoverable large XCoM excursion을 흡수했지만 Sand fall excursion도 함께 normal range 안에 두었다.
- 동시에 phase-pooled 0.5-percentile contract는 source/trajectory별 sustained tail을 모두 흡수하지 못해 Marble stable FP도 남겼다.
- Pelvis roll/pitch, exact angular velocity/height, COM/XCoM speed, support phase, Slip/Sink와 fall clocks는 artifact의 per-run diagnostic에 기록했지만 primary rule 조건에는 사용하지 않았다.

즉 current support-phase-normalized scalar MoS residual은 clean scenarios에서도 terrain-agnostic stable/fall separation을 제공하지 않는다.

## 20. Limitations

- 결과는 frozen Unitree G1 MuJoCo policy와 idealized exact contact/support geometry에 한정된다.
- Normal envelope는 observed non-fall을 recovery-capable evidence로 쓰지만 왜 recovery됐는지는 모델링하지 않는다.
- Phase 외 gait-cycle position, contact quality 또는 multivariate dynamics는 primary contract에 없다.
- Fresh matrix는 calibrated domain 안의 deterministic local variation이며 hardware generalization evidence가 아니다.
- Fall coverage가 낮아 lead distribution은 detected Concrete→Ice 세 run에만 해당한다.
- Viewer는 stored evaluation scalar의 terminal replay이며 별도 visual physics validation을 주장하지 않는다.

Terrain protected dataset/config/source/report SHA는 실행 전후 동일했고 FSR4+MLP+50 ms model을 재학습하지 않았다. Terrain/Stability producer independence와 기존 fusion truth table unit regression은 PASS다.

전체 regression은 `115 tests OK`, policy-dependent end-to-end smoke `1 skipped`로 통과했다.

## 21. Verdict

`WALKING_STABILITY_GROUND_TRUTH_NEEDS_REVISION`

| Acceptance gate | Required | Observed | Result |
|---|---:|---:|:---:|
| Stable false-instability rate | ≤10% | 5/24 = 20.83% | FAIL |
| Overall fall coverage | ≥85% | 3/16 = 18.75% | FAIL |
| Ice fall coverage | ≥80% | 3/7 = 42.86% | FAIL |
| Sand fall coverage | ≥80% | 0/9 = 0% | FAIL |
| Concrete/Marble meaningful detection | both | Concrete only | FAIL |
| Median detected fall lead | ≥200 ms | 620 ms | PASS |
| Pre-transition false instability | 0 | 0 | PASS |
| Causality | PASS | PASS | PASS |

따라서 `t_instability`를 future runtime Stability detector의 authoritative reference clock으로 사용할 수 없다.

## 22. Next recommendation

이번 milestone에서는 실패 패턴만 보존하고 종료한다. Quantile/threshold sweep, margin/persistence retuning, pelvis tilt/gyro AND/OR rule, AI oracle, Pelvis IMU runtime detector, GRU/MLP/CNN/LSTM, q/dq/torque/FSR stability input, recovery와 E84를 시작하지 않는다.

다음 별도 승인 작업에서는 threshold search가 아니라 current-support-polygon scalar MoS가 Sand fall과 Marble recovery를 왜 동일 범위에 놓는지 representation-level 원인을 먼저 검토해야 한다. 그 검토 전 Terrain candidate, fusion contract와 final sensor architecture 상태는 변경하지 않는다.
