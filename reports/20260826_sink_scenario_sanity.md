# Sink Hazard Scenario Sanity

## 목적과 결론

`SINK_HAZARD_SCENARIO_SANITY`는 contact penetration 자체와 locomotion/posture hazard를 분리하기 위한 bounded study다. 기존 uniform sand는 physical sink가 많아도 10초 동안 안정적으로 걸었고, same-height asymmetric compliance는 severity에 따라 support-depth asymmetry와 effect metric을 증가시켰다.

- `BENIGN_SINK_PHENOMENON`: uniform sand와 asymmetric mild
- Clean `HAZARD_SINK_CANDIDATE`: `asymmetric_left/severe`
- Strong visual stress candidate: `asymmetric_right/severe`; 후반 disturbance에서 established Slip 105 samples가 함께 발생하므로 향후 dual-phenomenon review 필요
- Primary class gate: `SINK_HAZARD_CRITERIA_NOT_YET_FROZEN`

이 결과는 scenario foundation의 sanity evidence이며 dataset, training label 또는 실제 토양 물성 검증이 아니다.

## Scenario geometry와 contact approximation

Uniform concrete/marble/ice/sand는 기존 `scene.xml`의 infinite plane과 기존 profile을 그대로 사용한다. 비대칭 pattern만 `scene_sink.xml`을 사용한다.

- Left lane: `x=[-10,10]`, `y=[0,10]`, nominal top `z=0`
- Right lane: `x=[-10,10]`, `y=[-10,0]`, nominal top `z=0`
- 두 static box는 `y=0`에서 겹치지 않고 맞닿으며 충분한 substrate depth를 갖는다.
- Blue/orange는 side identification용 visual-only color다.
- 반대 lane은 frozen uniform-sand profile이고 지정 lane만 severity profile을 받는다.
- Hole, step, lowered surface 또는 deformable mesh는 없다.

모든 profile은 synthetic engineering approximation이다. Friction은 sand와 동일하게 유지하고 `solref`/`solimp`만 바꿔 spatial compliance effect를 분리했다.

| Profile | Friction | `solref` | `solimp` |
|---|---|---|---|
| uniform sand | `(0.70, 0.010, 0.0010)` | `(0.050, 1.5)` | `(0.70, 0.90, 0.010, 0.5, 2.0)` |
| mild | same | `(0.055, 1.5)` | `(0.68, 0.89, 0.011, 0.5, 2.0)` |
| moderate | same | `(0.060, 1.5)` | `(0.65, 0.87, 0.013, 0.5, 2.0)` |
| severe | same | `(0.070, 1.5)` | `(0.60, 0.84, 0.016, 0.5, 2.0)` |

## 방법

Config는 [`20260826_sink_scenario_sanity.yaml`](../configs/experiment/20260826_sink_scenario_sanity.yaml)이다. 8개 run은 같은 verified policy, fixed-stand initial condition, 0.15 m/s command, 10초 duration, 2 kHz physics와 1 kHz sensor를 사용했다.

Policy SHA-256:

`2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`

Headless run으로 exact metric을 계산하고 모든 scenario의 max-tilt render state를 비교했다. `asymmetric_right/severe`는 canonical `--viewer` 10초 run도 완료했으며 headless와 동일한 runtime/diagnostic summary를 냈다. Viewer의 patch 색은 physics selection에 사용되지 않았다.

### Baseline 2초 regression

변경 전/후의 기존 scene 결과가 일치했다. 모든 run은 2000 samples, drop 0, finite IMU6였다.

| Terrain | Max penetration | Slip samples | `sink_physical` samples | Fall/censor |
|---|---:|---:|---:|---|
| concrete | 3.579 mm | 0 | 0 | none |
| ice | 14.258 mm | 950 | 12 | none |
| uniform sand | 14.167 mm | 0 | 2200 | none |

Uniform sand의 physical sink 2200 bilateral samples와 무낙상 baseline이 그대로 유지되었다. 이는 penetration persistence만으로 primary `SINK`를 정할 수 없다는 control evidence다.

## 지정 side mapping: 최초 2초 prefix

전체 10초 run의 deterministic prefix에서 지정 soft side가 항상 더 깊게 들어갔다. 장기 run에서는 path drift로 반대 발도 soft lane에 들어갈 수 있으므로 side mapping과 전체-run maxima를 구분한다.

| Pattern | Severity | Max penetration L/R | Max bilateral loaded asymmetry |
|---|---|---:|---:|
| asymmetric left | mild | 16.95 / 13.70 mm | 15.96 mm |
| asymmetric left | moderate | 21.81 / 14.13 mm | 21.15 mm |
| asymmetric left | severe | 35.40 / 19.65 mm | 31.77 mm |
| asymmetric right | mild | 14.33 / 16.46 mm | 15.67 mm |
| asymmetric right | moderate | 17.84 / 21.37 mm | 21.14 mm |
| asymmetric right | severe | 27.00 / 32.67 mm | 32.60 mm |

양쪽 pattern 모두 severity와 함께 초기 support-depth asymmetry가 약 16→21→32 mm로 증가했다.

## 10초 sanity matrix

`v/RMSE`는 body-forward mean velocity와 0.15 m/s command tracking RMSE다. `load Δ`는 좌우 loaded-contact sample count 차이다. `sink_physical` count는 persistence와 contact-episode reset의 영향을 받으므로 severity score로 해석하지 않는다.

| Scenario | Severity | Max pen. L/R | Max asym. | `sink_physical` L/R | Tilt | Pelvis z range | Peak ω | v/RMSE | load Δ | First fall/censor | Visual assessment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| concrete control | — | 2.98/3.58 mm | 2.56 mm | 0/0 | 1.71° | 15.00 mm | 0.82 rad/s | 0.143/0.037 | 112 | none | upright reference |
| uniform sand | — | 14.17/14.02 mm | 12.75 mm | 5172/5750 | 2.55° | 23.72 mm | 0.61 rad/s | 0.121/0.043 | 968 | none | bilateral compliance, stable gait |
| asymmetric left | mild | 18.65/23.75 mm | 23.49 mm | 5357/6256 | 2.58° | 31.45 mm | 0.85 rad/s | 0.097/0.064 | 1198 | none | close to uniform posture; slower |
| asymmetric left | moderate | 25.48/33.64 mm | 33.14 mm | 2282/2467 | 3.28° | 39.44 mm | 1.09 rad/s | 0.093/0.068 | 1207 | sample 4014, non-foot contact | visible squat/contact disturbance; transitional |
| asymmetric left | severe | 39.88/30.55 mm | 37.07 mm | 465/565 | 6.08° | 46.04 mm | 0.98 rad/s | 0.043/0.112 | 1000 | sample 825, non-foot contact | clear support loss and sustained slowing |
| asymmetric right | mild | 14.33/17.25 mm | 15.67 mm | 5266/5916 | 2.54° | 23.74 mm | 0.61 rad/s | 0.115/0.046 | 957 | none | visually close to uniform sand |
| asymmetric right | moderate | 30.19/24.49 mm | 29.42 mm | 5862/6186 | 2.53° | 33.53 mm | 0.80 rad/s | 0.090/0.069 | 664 | none | depth and speed effect; posture evidence weak |
| asymmetric right | severe | 69.46/36.87 mm | 68.87 mm | 546/732 | 17.91° | 157.27 mm | 2.83 rad/s | 0.010/0.172 | 135 | sample 1063, non-foot contact | obvious roll/crouch and near-stop |

모든 최종 run은 10,000 finite sensor samples와 drop 0을 유지했다. `first_fall_reasons=nonfoot_surface_contact`는 기존 fall/censor contract이며 base-height/orientation fall과 구분해야 한다. Right-severe는 8.27–8.45초에 max roll, penetration asymmetry와 minimum pelvis z가 연속해서 나타나 Viewer에서도 명확했다.

Loaded-contact imbalance는 severity에 따라 단조 증가하지 않아 final hazard gate candidate로 단독 사용하기 어렵다. 반면 초기 penetration asymmetry, pelvis z disturbance, forward-velocity degradation은 severity와 함께 더 일관된 분리를 보였다. Pelvis tilt는 right-moderate까지 약했고 severe에서만 명확해졌다.

## Scenario selection

### `BENIGN_SINK_PHENOMENON`

- Uniform sand: physical sink가 지속되지만 posture/gait가 유지되고 fall/censor가 없다.
- Left/right mild: support asymmetry는 증가하지만 tilt는 uniform과 유사하고 fall/censor가 없다.

### `HAZARD_SINK_CANDIDATE`

- `asymmetric_left/severe`: Slip 0, 6.08° tilt, 46 mm pelvis-z disturbance, mean forward velocity 0.043 m/s와 non-foot censor가 함께 발생한 clean candidate다.
- `asymmetric_right/severe`: Viewer에서 가장 명확하고 effect metric도 가장 크지만 후반에 established Slip 105 samples가 동반되었다. Scenario foundation에는 유용하나 향후 primary SINK study에서는 dual interval을 분리해야 한다.
- Moderate는 benign과 severe 사이의 transition evidence로 유지하되 primary hazard label로 선택하지 않는다.

### `UNUSABLE_SCENARIO`

Preliminary wide parameter sweep의 `solref=0.075/0.120/0.180` profiles는 mild에서도 조기 non-foot contact를 만들고 moderate/severe에서 약 0.1–0.36초 내 급격한 붕괴와 얕은 substrate saturation을 만들었다. 이는 locomotion degradation ladder보다 solver/geometry artifact에 가까워 최종 profiles에서 제외했다. Final scene은 top height를 바꾸지 않고 substrate depth와 profile 범위를 수정했으며 해당 saturation이나 non-finite explosion이 없다.

## 한계와 다음 단계

- 실제 deformable soil, soil displacement 또는 footprint를 simulation하지 않는다.
- 한 policy, speed와 initial gait condition의 deterministic sanity study라 일반화 근거가 아니다.
- Cause/effect continuous timing과 fall onset은 보존했지만 posture/gait degradation onset threshold는 만들지 않았다.
- 사람의 추가 Viewer review와 여러 speed/gait phase/seed 반복 뒤 effect metric 조합을 review한다.
- 그 전에는 primary SINK label, Pilot Dataset, Time-to-Separation 또는 ML 작업을 시작하지 않는다.

Final status: `SINK_HAZARD_CRITERIA_NOT_YET_FROZEN`.
