# Sink Hazard Transition and Criteria

## 결론

`SINK_HAZARD_TRANSITION_AND_CRITERIA`는 시작부터 비대칭 compliant lane 위에 서 있던 기존 scenario를 정상 보행 중 finite soft patch에 진입하는 사건으로 보강했다. 0.15 m/s의 8-run matrix에서 지정 foot의 patch contact(t0), 같은 contact episode의 physical sink(t1), posture degradation(t2), censor(t3)가 분리되었다.

- Left severe: t0 1,803 ms → t1 1,845 ms → t2 2,338 ms → t3 3,852 ms
- Right severe: t0 2,096 ms → t1 2,131 ms → t2 3,115 ms → t3 4,746 ms
- Uniform sand와 left/right mild 및 moderate transition은 t2가 없었고 8초 동안 censor 없이 걸었다.
- 모든 transition run에서 established Slip onset은 없었다. 이번 matrix의 dual phenomenon은 0건이다.
- Frozen effect criterion: patch-linked t1 뒤 pelvis tilt `> 0.04454633221030235 rad`(`2.5523168°`)가 20 consecutive ms 지속
- Final status: `SINK_HAZARD_CRITERIA_FROZEN`

이 결과는 synthetic MuJoCo contact-compliance scenario의 simulator-only ground truth다. 실제 deformable soil 물성 또는 이미 생성된 dataset/model을 뜻하지 않는다.

## 문제 정의와 기존 scenario의 한계

기존 full-lane sanity scene에서는 로봇이 simulation 시작부터 비대칭 compliant ground 위에 있었다. Severe case가 빠르게 자세를 잃어도 정상 보행 prefix와 원인/영향 사이 시간 여유를 명확히 분리하기 어려웠다. 이번 study의 질문은 다음과 같다.

> 정상 보행 뒤 physical sink가 시작된 hazardous episode에서, 향후 pelvis IMU6가 명확한 degradation보다 먼저 위험을 분리할 수 있는가?

이를 위해 원인 onset t1과 영향 onset t2를 별도로 보존한다. Hazardous episode의 future early-detection reference 후보는 t2가 아니라 t1이며 이번 milestone에서는 ML latency gate를 만들지 않는다.

## Finite patch geometry

새 XML을 추가하지 않고 canonical `scene_sink.xml`에 full-lane 슬롯과 transition 슬롯을 함께 유지한다. Pattern 선택은 `MjData` 생성 전에 한 topology만 활성화한다. 따라서 과거 `asymmetric_left/right` full-lane geometry와 profile은 그대로 재현된다.

- Walking direction: `+x`
- Pre-patch stable ground: concrete, `x=[-10.00, 0.35] m`
- Finite patch: `x=[0.35, 1.10] m`; selected left/right half만 기존 mild/moderate/severe profile
- Opposite patch half와 post-patch `x=[1.10, 10.00] m`: concrete
- 모든 surface top: 정확히 `z=0`
- 인접 box는 경계에서 맞닿고 volume overlap, hole, lowered surface, step은 없음
- Patch width along x: `0.75 m`
- Blue/orange는 side 확인용이며 물성이나 severity label이 아님

Patch 위치를 정하기 전 concrete 0.15 m/s trajectory를 측정했다. Pelvis x는 1.5/2.0/2.5/3.0초에 약 0.168/0.243/0.322/0.396 m였고 sole contact sphere의 전방 x는 2.0초 무렵 left 약 0.364 m, right 약 0.307 m였다. 이에 따라 start `x=0.35 m`를 선택했다. 실제 첫 지정 soft-side contact는 left 1.803초, right 2.096초로 목표 1.5–3.0초 안에 들며 각각 t0 전 1,000 ms baseline을 확보했다.

Severity의 friction/`solref`/`solimp`는 기존 sanity study 값을 변경하지 않았다.

## Timeline과 episode 정의

- `t0_patch_contact`: 해당 foot의 named sole geom과 soft patch geom의 첫 physical contact
- `t1_sink_physical`: 같은 raw contact episode에서 `loaded_penetration_change_m >= 5.5 mm`가 20 ms 지속되어 `sink_physical_active`가 처음 true인 sample
- `t2_degradation`: t1 뒤 frozen tilt criterion이 처음 active인 sample
- `t3_censor`: first fall 또는 non-foot surface contact sample

`BENIGN_SINK_EPISODE`는 t0와 t1이 있으나 관찰 가능한 pre-censor 구간에 t2가 없는 episode다. `HAZARDOUS_SINK_EPISODE`는 t0 → t1 뒤 t2가 발생한다. Established Slip onset이 t2보다 먼저라면 `DUAL_PHENOMENON`으로 별도 표시한다.

각 transition run은 global first t0 직전 최대 1,000 ms를 baseline으로 보존한다. Pelvis z, tilt, body-forward velocity와 pelvis angular speed에 대해 baseline mean과 post-event relative change를 계산한다. Exact pose/contact/terrain은 label/analysis 전용이며 runtime trace는 계속 `sequence`, `timestamp_us`, pelvis IMU6뿐이다.

## Experiment matrix

Config는 [`20260826_sink_transition_criteria.yaml`](../configs/experiment/20260826_sink_transition_criteria.yaml)이다. Primary matrix는 verified policy SHA-256 `2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`, fixed-stand initial condition, 0.15 m/s, 8초, 2 kHz physics와 1 kHz sensor를 공통 사용했다. 모든 run은 8,000 finite IMU samples, timestamp delta 1,000 µs와 drop 0이었다.

표의 시간은 simulation timestamp ms다. `z Δ`는 transition에서는 pre-event mean 대비 최대 drop, patch가 없는 controls에서는 full-run pelvis-z range다. `v mean/pre`는 full pre-censor mean과 t0 전 baseline mean이다.

| Scenario | Pattern | Severity | t0 ms | t1 ms | t2 ms | t3 ms | t1→t2 | t2→t3 | Max pen. asym. | Max tilt | z Δ | v mean/pre | Slip onset | Classification |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Concrete control | uniform | — | N/A | N/A | N/A | N/A | N/A | N/A | 2.56 mm | 1.71° | 15.00 mm range | 0.142/N/A m/s | N/A | BENIGN |
| Uniform sand | uniform | — | N/A | N/A | N/A | N/A | N/A | N/A | 12.73 mm | 2.55° | 23.72 mm range | 0.119/N/A m/s | N/A | BENIGN |
| Left mild | transition_left | mild | 1,803 | 1,851 | N/A | N/A | N/A | N/A | 16.66 mm | 1.78° | 15.56 mm | 0.125/0.150 m/s | N/A | BENIGN |
| Left moderate | transition_left | moderate | 1,803 | 1,848 | N/A | N/A | N/A | N/A | 21.36 mm | 2.28° | 20.36 mm | 0.113/0.150 m/s | N/A | BENIGN |
| Left severe | transition_left | severe | 1,803 | 1,845 | 2,338 | 3,852 | 493 ms | 1,514 ms | 35.30 mm | 4.30° | 29.48 mm | 0.143/0.150 m/s | N/A | HAZARD |
| Right mild | transition_right | mild | 2,096 | 2,133 | N/A | N/A | N/A | N/A | 18.13 mm | 1.81° | 14.13 mm | 0.120/0.158 m/s | N/A | BENIGN |
| Right moderate | transition_right | moderate | 2,096 | 2,132 | N/A | N/A | N/A | N/A | 24.15 mm | 2.18° | 19.13 mm | 0.112/0.158 m/s | N/A | BENIGN |
| Right severe | transition_right | severe | 2,096 | 2,131 | 3,115 | 4,746 | 984 ms | 1,631 ms | 39.46 mm | 3.89° | 26.33 mm | 0.119/0.158 m/s | N/A | HAZARD |

Uniform sand에는 bilateral `sink_physical_active`가 있었지만 finite patch t0가 없으므로 patch-linked t1은 N/A다. Physical penetration과 stable gait의 조합은 `BENIGN_SINK_PHENOMENON` control로 유지한다. Moderate는 penetration asymmetry와 z/velocity disturbance가 mild보다 크지만 frozen posture envelope 안에서 회복하고 censor가 없어 benign과 severe 사이의 transitional evidence로 설명된다.

## Control distribution과 degradation criterion

Threshold 선정에 concrete, uniform sand, left mild transition과 right mild transition을 benign controls로 사용했다. 아래 값은 8초 pre-censor exact diagnostics의 범위다.

| Control | Max pelvis tilt | Pelvis z range/drop | Peak angular speed | Mean forward velocity |
|---|---:|---:|---:|---:|
| Concrete | 1.710° | 15.00 mm range | 0.821 rad/s | 0.142 m/s |
| Uniform sand | 2.552° | 23.72 mm range | 0.615 rad/s | 0.119 m/s |
| Left mild transition | 1.780° | 15.56 mm pre-event drop | 0.993 rad/s | 0.125 m/s |
| Right mild transition | 1.808° | 14.13 mm pre-event drop | 1.144 rad/s | 0.120 m/s |

Pelvis z, forward-velocity drop와 angular speed는 일부 mild/moderate와 severe가 겹치거나 방향별 단조성이 약했다. Pelvis tilt는 benign upper envelope가 uniform sand의 `0.04454633221030235 rad`(`2.5523168°`)였고 moderate left/right의 max 2.284°/2.179°는 그 아래, severe left/right의 max 4.300°/3.895°는 명확히 위였다.

Frozen criterion은 다음 하나의 exact effect metric만 사용한다.

> Patch-linked t1 이후, pre-censor pelvis tilt가 `0.04454633221030235 rad`보다 큰 상태가 20 consecutive samples 지속하면 t2가 발생한다.

20 ms persistence를 적용하기 전 첫 threshold crossing은 left severe 2,319 ms, right severe 3,096 ms였고 첫 continuous crossing 길이는 각각 43 ms와 202 ms였다. Persistence가 충족된 t2는 2,338 ms와 3,115 ms다. Criterion은 terrain ID, Slip, penetration 크기, loaded-contact imbalance 또는 fall을 effect condition으로 사용하지 않는다.

유망 criterion에 허용된 severe 0.25 m/s robustness 두 건도 추가했다.

| Run | t0 | t1 | t2 | t3 | t1→t2 | t2→t3 | Max tilt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Left severe @ 0.25 m/s | 1,221 ms | 1,262 ms | 1,678 ms | 2,658 ms | 416 ms | 980 ms | 5.014° |
| Right severe @ 0.25 m/s | 1,508 ms | 1,540 ms | 2,495 ms | 5,774 ms | 955 ms | 3,279 ms | 4.707° |

두 robustness run도 Slip 없이 criterion을 통과하고 t2가 t3보다 먼저였다. 0.25 m/s left의 t0가 primary geometry 목표보다 빠른 것은 patch가 0.15 m/s trajectory에 맞춰졌기 때문이며 primary matrix의 timing 결과와 섞지 않는다.

## Viewer review와 dual phenomenon

Canonical visual-only Viewer로 left/right moderate와 severe를 실행했다. Headless와 Viewer는 별도 render state를 사용할 뿐 canonical physics/controller/diagnostics가 같았다.

- 모든 조건에서 gray concrete 위 정상 보행 prefix가 먼저 보였고 named soft-side contact의 t0와 일치했다.
- Left/right moderate는 patch에서 작은 roll/pitch와 crouch/velocity 변동이 생겼지만 upright gait를 유지하고 fall/non-foot censor 없이 회복했다.
- Left severe는 physical sink 뒤 pitch/roll과 support loss가 커졌고 t2 뒤 non-foot contact로 이어졌다.
- Right severe도 반대 방향 body response와 crouch가 명확해진 뒤 non-foot contact로 이어졌다.
- 네 Viewer run 모두 Slip onset은 없었다. Raw 8-run matrix에도 `DUAL_PHENOMENON`은 없다.

과거 full-lane `asymmetric_right/severe`의 후반 Slip 동반 결과는 history report에 그대로 남아 있으며 이번 clean transition 결과로 덮어쓰지 않는다.

## Criteria freeze review

Freeze 조건은 모두 충족했다.

1. Uniform sand는 t2가 없어 hazard로 분류되지 않았다.
2. Mild transition 양쪽 모두 t2가 없었다.
3. 같은 threshold가 left/right에 적용되었다.
4. Severe 양쪽과 허용된 0.25 m/s severe 양쪽을 검출했다.
5. Moderate는 t1은 있으나 envelope 안에서 회복하는 transitional benign case로 설명된다.
6. Viewer의 stable moderate/clear severe 관찰과 exact tilt가 일치했다.
7. Slip 여부를 criterion에 사용하지 않았다.
8. Fall/non-foot contact를 criterion에 사용하지 않았다.
9. 0.15 m/s severe에서 t2가 t3보다 1,514/1,631 ms 앞섰다.
10. 한 metric threshold와 짧은 persistence뿐이며 복잡한 state machine이 아니다.

따라서 simulator-only episode criterion 상태를 `SINK_HAZARD_CRITERIA_FROZEN`으로 변경한다. Hazardous episode의 detection reference 후보는 t1이고 t2는 사후 qualification effect다. `[t1,t2)`를 실제 class-2 training target으로 만드는 sample/window rule은 IMU-only observability 검증 전까지 만들지 않는다.

## Regression과 알려진 한계

- 기존 full-lane scene, severity profile, sanity config/report는 수정하지 않았다.
- 기존 full-lane 10초 8-run config를 다시 실행했으며 모든 run은 10,000 samples/drop 0이었다. 기존 report의 fall sample(left moderate 4,014, left severe 825, right severe 1,063), right-severe Slip 105 samples와 max tilt 1.71/2.55/2.58/3.28/6.08/2.54/2.53/17.91°가 재현되었다.
- Concrete/Ice/Uniform Sand 2초 regression은 각각 max penetration 3.579/14.258/14.167 mm, Slip 0/950/0 samples와 `sink_physical` 0/12/2,200 samples로 기존 baseline과 일치했다.
- Locked Slip oracle, 2 kHz physics, pelvis IMU6 1 kHz와 Viewer/headless parity를 regression test로 유지한다.
- 실제 토양 변형, footprint 또는 material displacement를 model하지 않는다.
- Criteria evidence는 한 verified policy, fixed initial condition과 deterministic runs에 한정된다.
- Benign controls는 0.15 m/s이고 0.25 m/s 추가 run은 severe 양쪽뿐이다. Pilot에서 speed/gait phase/seed와 broader normal coverage를 재검증해야 한다.
- Empirical upper envelope와 evaluation matrix가 같은 bounded study에서 나왔으므로 out-of-study false-positive audit가 필요하다.
- Dataset/Pilot/ML은 생성하거나 시작하지 않았다.

다음 순서는 별도 승인 뒤 Pilot raw dataset → raw IMU sanity/Time-to-Separation이다. 핵심 검증 질문은 t1 이후 t2 이전 구간에서 pelvis IMU6만으로 hazardous episode를 분리할 수 있는가이다.
