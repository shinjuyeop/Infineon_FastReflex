# Full-State Walking Stability Ground-Truth Sanity

Milestone: `FULL_STATE_STABILITY_GROUND_TRUTH_SANITY`

## 1. Purpose

이 실험의 목적은 기존 XCoM/MoS scalar를 더 튜닝하지 않고, observed-stable exact simulator trajectory로 학습한 phase-conditioned 정상 상태 분포에서 현재 privileged state가 지속적으로 이탈하는 최초 causal 시점 `t_instability`를 검증하는 것이었다. Runtime Stability detector, sensor architecture freeze, recovery와 E84 작업은 수행하지 않았다.

작업 시작 시 실제 `HEAD`와 `origin/main`은 모두 `2940c68a58ccb6b132b9be0f316f4fd905949d05`였고 worktree는 clean이었다. 세 candidate, feature order, fitting contract, selection gate와 조건부 fresh matrix는 simulation 전에 [`20260828_full_state_stability_ground_truth_sanity.yaml`](../configs/experiment/20260828_full_state_stability_ground_truth_sanity.yaml)에 고정했다.

## 2. Why MoS was retired

Historical `WALKING_STABILITY_GROUND_TRUTH_SANITY` 결과는 fresh valid 40/40, stable false firing `5/24 = 20.83%`, overall fall coverage `3/16 = 18.75%`, Ice `3/7 = 42.86%`, Sand `0/9 = 0%`, pre-transition FP 0, causality PASS였다. Verdict는 `WALKING_STABILITY_GROUND_TRUTH_NEEDS_REVISION`이었다.

Stable Ice가 fall Ice보다 더 negative MoS를 보이기도 했고 Sand fall은 strong deformation과 Sink에도 residual로 분리되지 않았다. 이 historical result는 수정하거나 삭제하지 않았다. 이번 실험에서 raw MoS와 stability residual은 candidate vector에 넣지 않았고 quantile/margin/persistence retuning도 하지 않았다.

## 3. Full-state normal-envelope concept

각 support phase의 observed-stable state를 정규화하고 하나의 multivariate Gaussian approximation으로 fit했다. 현재 sample `x`와 stable mean/covariance 사이의 regularized Mahalanobis distance가 stable-only q99.5를 20 ms 연속 초과한 첫 시점을 `t_instability` candidate로 정의했다.

이 clock의 정확한 의미는 “현재 full-body state가 observed-stable locomotion distribution에서 비정상적으로 멀어진 상태가 20 ms 지속된 최초 시점”이다. 이는 fall theorem이나 real-world universal boundary가 아니라 frozen G1 simulation 안의 privileged stability-reference 후보다.

## 4. Privileged/runtime boundary

Candidate input은 simulator-only pelvis pose/velocity, lower-body q/dq, whole-body COM velocity와 exact loaded-contact-derived support phase다. Terrain identity, scenario/run ID, intended/observed outcome, fall/fall time, Slip/Sink, deformation, patch geometry와 transition-relative timestamp는 vector에서 제외했다.

이 state는 future runtime sensor proposal이 아니다. Rich privileged state로 ground truth를 만들고 향후 minimal sensor detector가 그 clock을 예측하는 구조를 의도했다. 특히 `PELVIS_STATE`는 향후 Pelvis IMU6와 물리적으로 가까워 circularity limitation이 있지만 exact orientation/velocity/height와 raw IMU tensor는 동일한 numerical input이 아니다. 이번 실험은 runtime IMU를 사용하지 않았다.

## 5. Calibration cohort

### CANDIDATE CALIBRATION

기존 clean calibrated cohort 36개를 8 s, 1 kHz로 다시 실행했고 36/36 observed outcome이 prior result와 일치했다. Observed-stable 20개만 normal fit에 사용했고 fall 16개는 candidate comparison에만 사용했다.

| Stable fitting evidence | Runs |
|---|---:|
| Concrete hard stable | 3 |
| Marble hard stable | 3 |
| Concrete/Marble→Ice stable | 6 |
| Concrete/Marble→Sand stable | 8 |
| Total stable fit | 20 |
| Fall comparison, excluded from fit | 16 |

10 ms global time-grid stride와 run/phase별 256-sample cap을 적용한 candidate별 phase sample 수는 Left single support `5,120`, Right single support `5,120`, Double support `2,448`이었다. 모든 candidate가 동일 run/sample eligibility를 사용했다.

## 6. Candidate A — Pelvis

`PELVIS_STATE`는 9차원이다: roll, pitch, angular velocity xyz, linear velocity xyz, pelvis height. Absolute yaw는 제외했다.

| Metric | Calibration result |
|---|---:|
| Stable FP | 6/20 = 30.00% |
| False-run abnormal candidate duration | 553 ms |
| Overall fall coverage | 2/16 = 12.50% |
| Ice fall coverage | 2/8 = 25.00% |
| Sand fall coverage | 0/8 = 0% |
| Median detected fall lead | 576 ms |

Lead는 길었지만 2개 fall에만 해당한다. Stable specificity, overall/Ice/Sand coverage gate를 모두 실패했다.

## 7. Candidate B — Lower body

`LOWER_BODY_STATE`는 G1 MJCF의 authoritative joint names로 resolve한 12개 leg joint의 q/dq 24차원이다. 순서는 left/right 각각 hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll이며 waist, shoulder, elbow, wrist 등 upper body는 제외했다.

| Metric | Calibration result |
|---|---:|
| Stable FP | 5/20 = 25.00% |
| False-run abnormal candidate duration | 668 ms |
| Overall fall coverage | 9/16 = 56.25% |
| Ice fall coverage | 6/8 = 75.00% |
| Sand fall coverage | 3/8 = 37.50% |
| Median detected fall lead | 35 ms |

MoS보다 coverage는 높았지만 stable specificity, overall coverage, Sand coverage와 median lead를 실패했다.

## 8. Candidate C — Full state

`FULL_STATE`는 Pelvis 9 + lower-body 24 + whole-body COM linear velocity xyz + COM height above current support의 37차원이다. Raw MoS/residual은 포함하지 않았다.

| Metric | Calibration result |
|---|---:|
| Stable FP | 5/20 = 25.00% |
| False-run abnormal candidate duration | 636 ms |
| Overall fall coverage | 9/16 = 56.25% |
| Ice fall coverage | 6/8 = 75.00% |
| Sand fall coverage | 3/8 = 37.50% |
| Median detected fall lead | 28 ms |

LOWER_BODY_STATE와 detected run이 같았고 추가 locomotion state가 qualification을 개선하지 못했다. Stable specificity, overall coverage, Sand coverage와 median lead를 실패했다.

## 9. Stable-only fitting

Mean, standard deviation, covariance와 distance threshold는 candidate/phase별 observed-stable samples만으로 계산했다. Fall run ID 16개는 모든 model의 `fit_run_ids`에서 제외됐다. Long-run dominance를 제한하기 위해 predeclared `10 ms stride + per-run/per-phase 256 cap`을 deterministic evenly-spaced selection으로 적용했다.

Transition stable run은 normal prefix, target contact 이후 recoverable passage와 post-transition stable walking을 포함했다. Post-fall samples는 fit에 들어갈 수 없으며 fall run 자체가 fit cohort에 없다.

## 10. Phase conditioning

Exact loaded contact로 `LEFT_SINGLE_SUPPORT`, `RIGHT_SINGLE_SUPPORT`, `DOUBLE_SUPPORT`를 계산해 phase별 독립 normal distribution을 fit했다. `NO_SUPPORT`는 score/fit 대상이 아닌 diagnostic이다. Phase는 terrain lookup key가 아니며 runtime detector input commitment도 아니다.

## 11. Mahalanobis definition

각 candidate와 phase에서 stable mean `μ`, standard deviation `s`, normalized sample covariance `Σ`를 계산했다. Standard deviation floor는 `1e-8`이었다.

```text
z = (x - μ) / s
Σ_reg = (1 - 0.05) Σ + 0.05 diag(Σ) + 1e-6 I
D = sqrt(zᵀ Σ_reg⁻¹ z)
```

NumPy로 고정된 deterministic implementation을 사용했다. 현재 sample과 frozen stable statistics만 score에 사용하며 smoothing이나 future state는 없다.

## 12. Frozen threshold/persistence

Phase threshold는 stable calibration distance의 linear q99.5, persistence는 1 kHz에서 20 consecutive samples였다.

| Candidate | Left single | Right single | Double |
|---|---:|---:|---:|
| `PELVIS_STATE` | 11.408824 | 9.233617 | 7.631731 |
| `LOWER_BODY_STATE` | 18.586073 | 16.242706 | 11.060074 |
| `FULL_STATE` | 20.695724 | 19.164220 | 13.306059 |

Transition primary persistence는 first exact B-terrain contact에서 reset했다. Ungated trace는 pre-transition diagnostic만 제공한다. Candidate 결과를 본 뒤 λ, q99.5, 20 ms, sample cap 또는 feature schema를 바꾸지 않았다.

## 13. Calibration candidate comparison

| Oracle | Dimensions | Stable FP | Fall coverage | Ice | Sand | Median lead |
|---|---:|---:|---:|---:|---:|---:|
| Pelvis-state distance | 9 | 30.00% | 12.50% | 25.00% | 0% | 576 ms |
| Lower-body distance | 24 | 25.00% | 56.25% | 75.00% | 37.50% | 35 ms |
| Full-state distance | 37 | 25.00% | 56.25% | 75.00% | 37.50% | 28 ms |

Calibration minimum은 stable FP ≤10%, overall fall ≥80%, Ice/Sand 각각 ≥75%, median lead ≥200 ms였다. Pelvis는 lead만 통과했고 Lower/Full은 Ice coverage만 통과했다.

## 14. Candidate selection

선택된 candidate는 없다. 사전 선언한 rule은 qualification을 통과한 candidate만 priority/near-tie 비교에 넣는다. 세 representation 모두 qualification을 실패했으므로 near-tie simplicity rule을 적용할 대상도 없었다.

이 결과를 근거로 threshold sweep, feature addition, representation 조합 탐색이나 neural oracle을 시작하지 않았다.

## 15. Freeze provenance

Experiment config SHA-256은 `b5521fccd8dc9f4068cedd2db8d9c4a859d34e3fc2b73b308cf47b6085054d95`이고 calibration source commit은 `2940c68a58ccb6b132b9be0f316f4fd905949d05`이다. Policy SHA-256 `2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`도 실행 전에 확인했다.

Qualified selection이 없으므로 `selected_ground_truth.json`과 selected artifact SHA는 생성하지 않았다. Gitignored `results.json`에는 candidate statistics와 causality/replay diagnostics만 저장했다. 이는 “selected candidate가 존재할 때만 artifact를 freeze한다”는 contract를 따른 것이다.

## 16. Fresh validation

### FRESH SELECTED-CANDIDATE VALIDATION

Fresh matrix는 Concrete/Marble × Ice/Sand × stable/fall 8 groups에 각 6개, 총 48개로 simulation 전에 고정했다. Calibration 및 이전 MoS fresh signatures와 중복이 없고 frozen operating domains 안에 있음을 static gate로 확인했다.

그러나 calibration candidate가 하나도 qualification을 통과하지 못했으므로 fresh 48개는 실행하지 않았다. 따라서 fresh valid stable/fall, fresh stable FP, fresh fall coverage와 fresh lead distribution은 `N/A`다. Validation 결과를 보며 contract를 수정하는 leakage도 발생하지 않았다.

## 17. Stable false positives

Calibration stable FP는 Pelvis 6/20, Lower 5/20, Full 5/20이었다. Lower/Full false firing은 Ice stable `5/6`에 집중됐고 Sand stable은 `0/8`이었다. Pelvis는 Ice stable `5/6`과 Sand stable `1/8`에서 firing했다. 모든 candidate가 overall ≤10% gate를 실패했다.

Lower/Full의 false run은 `cal_c_ice_s01`, `cal_c_ice_s02`, `cal_c_ice_s03`, `cal_m_ice_02`, `cal_m_ice_03`이다. Recoverable physical Slip 뒤의 정상 trajectory 일부가 pooled stable Gaussian의 q99.5 밖에서 20 ms 이상 지속된 결과다.

## 18. Fall coverage

Pelvis는 fall 16개 중 Ice 2개만 pre-fall detection했고 Sand detection은 없었다. Lower/Full은 동일한 9개 fall을 검출했다: Concrete-origin 5/8, Marble-origin 4/8이다. 그러나 overall `56.25%`는 calibration minimum 80%보다 23.75 percentage points 낮다.

Lower/Full detected IDs는 `cal_c_ice_f01`, `cal_c_ice_f03`, `cal_c_ice_f04`, `cal_c_sand_f01`, `cal_c_sand_f03`, `cal_m_ice_01`, `cal_m_ice_04`, `cal_m_ice_05`, `cal_m_sand_f01`이다. Fall 이후 onset은 detection으로 세지 않았다.

## 19. Ice/Sand coverage

| Candidate | Ice stable FP | Ice fall | Sand stable FP | Sand fall |
|---|---:|---:|---:|---:|
| Pelvis | 5/6 = 83.33% | 2/8 = 25.00% | 1/8 = 12.50% | 0/8 = 0% |
| Lower body | 5/6 = 83.33% | 6/8 = 75.00% | 0/8 = 0% | 3/8 = 37.50% |
| Full state | 5/6 = 83.33% | 6/8 = 75.00% | 0/8 = 0% | 3/8 = 37.50% |

Full-state expansion은 Sand separation을 충분히 만들지 못했다. 반대로 Ice fall coverage가 올라간 representation은 recoverable Ice stable에도 매우 과민했다.

## 20. Concrete/Marble robustness

| Candidate | Concrete stable FP | Concrete fall | Marble stable FP | Marble fall |
|---|---:|---:|---:|---:|
| Pelvis | 4/10 = 40% | 1/8 = 12.50% | 2/10 = 20% | 1/8 = 12.50% |
| Lower body | 3/10 = 30% | 5/8 = 62.50% | 2/10 = 20% | 4/8 = 50.00% |
| Full state | 3/10 = 30% | 5/8 = 62.50% | 2/10 = 20% | 4/8 = 50.00% |

Lower/Full은 두 source에서 meaningful detection 자체는 보였지만 coverage와 specificity가 낮다. Source robustness만으로 authoritative clock을 주장할 수 없다.

## 21. Fall-lead distribution

Calibration valid detection에 한정한 lead distribution이다.

| Candidate | Min | p10 | Median | p95 | Max |
|---|---:|---:|---:|---:|---:|
| Pelvis, n=2 | 454 ms | 478.4 ms | 576 ms | 685.8 ms | 698 ms |
| Lower body, n=9 | 25 ms | 25 ms | 35 ms | 988.4 ms | 1,150 ms |
| Full state, n=9 | 10 ms | 17.2 ms | 28 ms | 979.2 ms | 1,148 ms |

Pelvis의 median은 ≥200 ms이지만 coverage 12.5%에 기반한다. Lower/Full은 coverage가 개선된 대신 절반의 detected fall onset이 fall 약 28–35 ms 전에 형성돼 recovery reference로 쓰기에는 지나치게 늦다. 이는 runtime detector latency가 아니라 privileged divergence부터 fall까지 남은 physical horizon이다.

## 22. Representative replay

선택된 candidate가 없으므로 아래는 가장 높은 overall coverage를 가진 더 단순한 `LOWER_BODY_STATE`의 **non-authoritative calibration diagnostic replay**다. Viewer/status는 stored evaluation 값을 재생하며 physics를 바꾸지 않았다.

| Run | Timeline |
|---|---|
| `cal_c_ice_s01` | contact 1221 ms → Slip 1911 ms → false `t_instability` 4450 ms → stable |
| `cal_c_ice_f01` | contact 1221 ms → Slip 1911 ms → `t_instability` 4019 ms → fall 4049 ms, lead 30 ms |
| `cal_m_ice_01` | contact 1221 ms → Slip 1909 ms → `t_instability` 2029 ms → fall 2064 ms, lead 35 ms |
| `cal_c_sand_s01` | contact 1221 ms → deformation 20.17 mm → no `t_instability` → stable |
| `cal_c_sand_f01` | contact 1221 ms → Sink 3038 ms → `t_instability` 4206 ms → fall 4231 ms, lead 25 ms |
| `cal_m_sand_f01` | contact 1221 ms → Sink 3046 ms → `t_instability` 4204 ms → fall 4229 ms, lead 25 ms |

Concrete→Sand stable replay의 maximum score는 Left single support에서 `18.55996`, threshold `18.58607`로 바로 아래였다. 다른 five replay는 status text의 distance/phase threshold와 evaluator onset/fall 값이 일치했다.

## 23. Physical Slip/Sink relationship

Ice stable 6/6과 Ice fall 8/8 모두 physical Slip이 있었다. Lower/Full은 Ice stable 5/6을 false firing하면서 Ice fall 6/8만 검출했으므로 Slip occurrence가 아니라 walking-state divergence를 안정적으로 분리했다고 볼 수 없다.

Sand stable 8개는 약 `20.14–20.20 mm` deformation이 있었고 Sink diagnostic은 0/8이었다. Sand fall 8개는 약 `40.12–40.19 mm` deformation과 Sink diagnostic 8/8이었지만 Lower/Full detection은 3/8이었다. Sink/deformation 값은 candidate input이 아니며, 결과는 선택한 generic state distance가 Sand fall divergence 대부분을 pre-fall에 포착하지 못했음을 보여준다.

## 24. Historical MoS comparison

| Oracle | Stable FP | Fall coverage | Ice | Sand |
|---|---:|---:|---:|---:|
| Historical MoS residual, fresh | 20.83% | 18.75% | 42.86% | 0% |
| Pelvis-state distance, calibration | 30.00% | 12.50% | 25.00% | 0% |
| Lower-body distance, calibration | 25.00% | 56.25% | 75.00% | 37.50% |
| Full-state distance, calibration | 25.00% | 56.25% | 75.00% | 37.50% |

서로 다른 evaluation split이라는 제한은 있지만 Lower/Full이 MoS보다 fall coverage를 명확히 높인 것은 확인된다. 그러나 stable FP, Sand coverage와 lead가 calibration qualification에 크게 못 미쳐 fresh generalization을 검증할 수준의 separation은 아니다.

## 25. Causality

Result: `PASS` for all three candidates.

각 candidate에서 이미 결정된 onset까지의 distance/candidate/active/onset prefix를 보존한 채 이후 privileged feature suffix를 synthetic zero state로 교체했다. Pelvis는 4727 ms까지, Lower/Full은 4450 ms까지 output이 exact equality를 유지했다. Score function은 future fall metadata를 받지 않는다.

Ungated pre-transition false-instability run은 모든 candidate에서 0이었다. Primary score는 first exact target contact 전 abnormal을 mask하고 persistence를 contact에서 reset했다.

## 26. Limitations

- 결과는 frozen Unitree G1 MuJoCo policy, exact simulator state와 현재 calibrated operating domains에 한정된다.
- Phase-conditioned single Gaussian/Mahalanobis distance는 multimodal gait-cycle manifold를 충분히 표현하지 못할 수 있다. 이 milestone에서는 mixture, temporal embedding, feature search나 neural ground truth를 시도하지 않았다.
- q/dq와 pelvis/COM state를 더한 것만으로 recoverable Ice deviation과 falling deviation이 분리되지 않았고 Sand fall onset은 대부분 너무 늦거나 없었다.
- Pelvis candidate는 runtime Pelvis IMU와 물리적으로 가까워 향후 runtime detector evaluation에서 circularity를 별도로 관리해야 한다.
- Fresh validation을 실행하지 않았으므로 calibration improvement를 unseen generalization evidence로 해석할 수 없다.
- Representative replay는 failed candidate의 diagnostic이며 authoritative selected clock이 아니다.

Terrain protected dataset/config/source/report SHA는 실행 전후 동일했다. FSR4 + MLP + 50 ms candidate를 재학습하거나 변경하지 않았고 Terrain/Stability producer independence와 fusion truth-table regression은 PASS였다. `FINAL_SENSOR_ARCHITECTURE_FROZEN`은 선언하지 않는다.

전체 repository regression은 `126 tests OK`, policy-dependent end-to-end smoke `1 skipped`로 통과했다.

## 27. Verdict

`FULL_STATE_STABILITY_GROUND_TRUTH_NOT_SEPARABLE`

| Calibration gate | Required | Pelvis | Lower | Full |
|---|---:|---:|---:|---:|
| Stable FP | ≤10% | 30.00% FAIL | 25.00% FAIL | 25.00% FAIL |
| Overall fall coverage | ≥80% | 12.50% FAIL | 56.25% FAIL | 56.25% FAIL |
| Ice fall coverage | ≥75% | 25.00% FAIL | 75.00% PASS | 75.00% PASS |
| Sand fall coverage | ≥75% | 0% FAIL | 37.50% FAIL | 37.50% FAIL |
| Median fall lead | ≥200 ms | 576 ms PASS | 35 ms FAIL | 28 ms FAIL |
| Pre-transition false instability | 0 | 0 PASS | 0 PASS | 0 PASS |
| Causality | PASS | PASS | PASS | PASS |

세 representation 모두 stable/fall state divergence를 qualification 수준으로 나타내지 못했다. 따라서 selected privileged state-distance clock은 없고 fresh acceptance verdict를 낼 authoritative `t_instability`도 없다.

## 28. Next recommendation

이번 milestone은 failure pattern을 보존하고 종료한다. Threshold/quantile/persistence sweep, feature AND/OR 조합, neural ground-truth model, Pelvis IMU rule/GRU/MLP, q/dq runtime augmentation, Terrain retraining, final sensor freeze, recovery와 E84를 자동으로 시작하지 않는다.

다음 별도 승인 작업이 있다면 단순 covariance/threshold tuning보다 먼저 stable gait의 multimodality와 gait-cycle conditioning이 왜 pooled phase Gaussian에서 Ice recovery와 Sand fall을 겹치게 하는지 representation-level evidence를 검토해야 한다.
