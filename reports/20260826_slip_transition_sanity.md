# Slip Hazard Transition Sanity

## 목적과 결론

`SLIP_HAZARD_TRANSITION_SANITY`는 정상 concrete 보행 뒤 finite low-friction patch에 진입하여 실제 established Slip이 발생하는 Pilot 후보 scenario를 검증했다. 기존 50 mm/3 ms physical oracle은 변경하지 않았다.

- 0.10/0.15/0.20/0.25 m/s transition 모두 `CLEAN_SLIP_EVENT`
- 모든 transition에서 left/right 양발 established Slip 확인
- 첫 접촉 foot은 0.20 m/s에서 right, 나머지 세 속도에서 left
- 모든 run에서 SINK hazard onset은 N/A; Slip 이전 Sink contamination 0건
- 0.10/0.15/0.20 m/s는 Slip evolution 뒤 base-height censor, 0.25 m/s는 8초 동안 censor 없이 recovery형 Slip을 제공
- Ice transition 중 `NO_SLIP_TRANSITION`과 `UNUSABLE_SLIP_SCENARIO`는 없음

이는 Pilot Dataset 자체가 아니라 event-condition sanity evidence다. Runtime input은 계속 pelvis IMU6 @ 1 kHz뿐이다.

## Finite patch geometry와 profile

새 scene이나 runner를 만들지 않고 Sink transition의 canonical `scene_sink.xml` topology를 재사용했다. `--slip-pattern transition`은 left/right finite patch를 동시에 활성화하여 full-width patch를 만든다.

- Walking direction: `+x`
- Concrete pre-patch: `x=[-10.00,0.35] m`
- Full-width Ice patch: `x=[0.35,1.10] m`, `y=[-10,10] m`
- Concrete post-patch: `x=[1.10,10.00] m`
- 모든 top surface: `z=0`
- 인접 box는 경계에서 맞닿고 active volume overlap, hole, step 또는 lowered surface가 없음
- Cyan patch color는 visual-only이며 physics/label에 사용하지 않음

Patch는 기존 frozen Ice engineering profile을 그대로 사용한다.

| Profile | Friction | `solref` | `solimp` |
|---|---|---|---|
| Concrete | `(1.00,0.005,0.0001)` | `(0.015,1.0)` | `(0.95,0.99,0.001,0.5,2.0)` |
| Ice patch | `(0.05,0.001,0.00001)` | `(0.015,1.0)` | `(0.95,0.99,0.001,0.5,2.0)` |

Friction, `solref`, `solimp` tuning은 수행하지 않았다. 0.15 m/s의 first patch contact는 1.803초로 기존 Sink range를 그대로 재사용할 수 있었다. 0.25 m/s는 1.221초로 1.5초보다 빠르지만 최소 1초 concrete prefix를 유지하므로 primary matrix에서 보존하고 별도 geometry tuning을 하지 않았다.

## Event contract

- `t0_patch_contact`: left/right 중 첫 named sole–Ice patch physical contact; per-foot t0도 보존
- `t1_established_slip`: frozen oracle의 첫 ANY-foot onset; per-foot onset도 보존
- `t2_degradation`: 기존 frozen pelvis-tilt diagnostic의 Slip 이후 첫 onset이며 SLIP qualification에는 사용하지 않음
- `t3_censor`: first fall 또는 non-foot surface contact

Established Slip은 valid loaded contact, touchdown transient 10 ms 제외, 같은 contact episode, pre-fall 조건에서 touchdown anchor drift `>=0.050 m`가 1 kHz에서 3 consecutive samples 지속되는 기존 정의다. Terrain 이름이나 patch contact만으로 SLIP을 만들지 않는다. Primary class는 affected foot을 맞히지 않는 ANY-SLIP이며 left/right ownership은 diagnostic이다.

## Experiment matrix와 결과

Config는 [`20260826_slip_transition_sanity.yaml`](../configs/experiment/20260826_slip_transition_sanity.yaml)이다. 모든 run은 verified policy SHA-256 `2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`, fixed-stand initial condition, 8초, 2 kHz physics와 pelvis IMU6 1 kHz를 사용했다. 모든 run은 8,000 finite samples, timestamp delta 1,000 µs와 drop 0이었다.

표의 시간은 simulation timestamp ms다. Drift와 Slip sample은 `[left/right]` 순서이며 t3 이후 metric은 제외했다.

| Speed | Patch x | First patch foot | t0 | Left Slip | Right Slip | ANY-SLIP t1 | t0→t1 | t3 | t1→t3 | Max anchor drift L/R | Slip samples L/R | SINK hazard | Classification |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.15 concrete control | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 3.09/6.98 mm | 0/0 | N/A | BENIGN_CONTROL |
| 0.10 m/s | `[0.35,1.10]` | Left | 2,406 | 4,559 | 4,031 | 4,031 | 1,625 ms | 5,062 | 1,031 ms | 88.76/99.74 mm | 200/22 | N/A | CLEAN_SLIP_EVENT |
| 0.15 m/s | `[0.35,1.10]` | Left | 1,803 | 3,088 | 3,334 | 3,088 | 1,285 ms | 5,133 | 2,045 ms | 136.30/137.19 mm | 153/391 | N/A | CLEAN_SLIP_EVENT |
| 0.20 m/s | `[0.35,1.10]` | Right | 1,505 | 2,879 | 2,634 | 2,634 | 1,129 ms | 5,087 | 2,453 ms | 151.82/169.16 mm | 356/251 | N/A | CLEAN_SLIP_EVENT |
| 0.25 m/s | `[0.35,1.10]` | Left | 1,221 | 1,915 | 2,488 | 1,915 | 694 ms | N/A | N/A | 157.16/199.47 mm | 940/815 | N/A | CLEAN_SLIP_EVENT |

Per-foot patch-contact t0는 0.10에서 L/R 2,406/2,695 ms, 0.15에서 1,803/2,096 ms, 0.20에서 1,807/1,505 ms, 0.25에서 1,221/1,507 ms였다. 첫 contact foot과 첫 Slip foot은 0.10 m/s에서 서로 달랐고, 나머지는 같았다. 따라서 ANY-SLIP aggregation은 initiating foot shortcut을 피하는 데 필요하다.

Optional t2는 0.10/0.15/0.20/0.25 m/s에서 각각 4,312/3,192/2,715/2,029 ms였으며 t1보다 281/104/81/114 ms 늦었다. 이는 posture effect가 Slip 뒤에 나타날 수 있음을 보여주는 diagnostic일 뿐 SLIP label gate가 아니다.

## Sink separation과 censor

Transition별 `sink_physical_active` sample은 L/R 기준 0/4, 0/5, 57/18, 38/53으로 일부 존재했지만 finite soft-compliance patch contact가 아니며 frozen SINK hazard onset은 모든 run에서 N/A였다. 따라서 SINK hazard가 t1보다 먼저인 contamination run은 없다. Sink physical precursor 수를 SLIP 또는 SINK class로 오해하지 않는다.

0.10/0.15/0.20 m/s의 t3 reason은 모두 `fallen_base_height`였다. Established Slip은 t3보다 각각 1,031/2,045/2,453 ms 먼저 발생했다. Oracle과 report metric은 `pre_fall_valid`를 사용하며 t3 이후 controller/limb motion을 valid Slip evidence로 사용하지 않았다. 0.25 m/s는 큰 bilateral drift 뒤에도 8초 동안 censor가 없어 recovery/non-fall Slip example을 제공한다.

## Viewer review

Canonical visual-only Viewer로 concrete control, 0.15 m/s transition, drift가 가장 큰 0.25 m/s transition을 확인했다.

- Concrete control은 patch 없이 upright normal gait를 유지했고 Slip이 없었다.
- 두 transition은 gray concrete 위 normal prefix 뒤 cyan full-width patch에 실제 sole이 진입했다.
- Patch contact 즉시 폭발하지 않았고 694–1,625 ms에 걸쳐 tangential sliding이 누적된 뒤 established Slip이 발생했다.
- 0.15 m/s에서는 양발 sliding과 자세 저하가 보인 뒤 base-height censor로 이어졌다.
- 0.25 m/s에서는 더 큰 bilateral sliding이 보였지만 duration 내 fall 없이 계속 움직였다.
- Patch 경계에서 geometry seam 때문에 튀거나 top-height step을 밟는 현상은 없었다.

Viewer는 render state만 복사하며 physics/controller/sensor/label을 바꾸지 않았다.

## Gait-phase sensitivity와 Pilot variation

첫 patch contact 시 해당 physical-contact episode age와 deterministic controller의 nominal 0–1 policy phase는 다음과 같았다. Phase는 0.6초 policy period와 20 ms control update를 기준으로 계산했으며 foot-contact state와 함께 gait-phase coverage를 해석하는 diagnostic이다.

| Speed | Left contact age | Right contact age | First foot | Policy phase at first t0 |
|---:|---:|---:|---|---:|
| 0.10 | 11 ms | 1 ms | Left | 0.000 |
| 0.15 | 0 ms | 6 ms | Left | 0.000 |
| 0.20 | 0 ms | 7 ms | Right | 0.500 |
| 0.25 | 0 ms | 8 ms | Left | 0.033 |

첫 foot은 속도에 따라 바뀌지만 모든 t0가 touchdown 또는 매우 이른 stance에 집중됐다. 따라서 Pilot Dataset에서는 patch start position/initial gait phase variation으로 contact age를 넓히고 left/right first-contact coverage를 균형화해야 한다. 대량 sweep이나 새 oracle은 필요 없으며 동일 full-width topology와 frozen profile을 유지한 소수의 재현 가능한 variation이면 된다.

## Regression, 한계와 다음 단계

- 2초 uniform baseline은 기존 값과 정확히 일치했다: Concrete max penetration 3.579 mm/Slip 0 samples, Ice 14.258 mm/Slip 950 samples, Sand 14.167 mm/Slip 0 samples와 Sink physical 2,200 samples. 모두 2,000 sensor samples/drop 0이었다.
- Frozen severe Sink transition도 기존 event index와 정확히 일치했다: left t0/t1/t2/t3 = 1802/1844/2337/3851, right = 2095/2130/3114/4745. 두 run 모두 SINK hazard이며 Slip 0 samples, 8,000 sensor samples/drop 0이었다.
- 11개 simulation test는 policy-backed end-to-end transition을 포함해 모두 통과했다. Viewer와 headless 경로는 동일 canonical physics를 사용하고 Viewer에는 복사된 render state만 전달한다.
- Frozen Slip 50 mm/3 ms와 frozen Sink threshold/status는 test parity로 고정했다.
- Runtime trace는 `sequence`, `timestamp_us`, pelvis IMU6만 포함하며 terrain/contact/drift/oracle은 exact diagnostic-only다.
- 한 policy, fixed initial condition과 deterministic speed matrix라 broader gait/seed 일반화 근거는 아니다.
- Full-width patch는 ANY-SLIP foundation에 적합하지만 unilateral friction event 연구는 이번 범위가 아니다.
- Ice는 실제 얼음 물성 측정값이 아니라 engineering contact approximation이다.
- `NO_SLIP_TRANSITION`이 없어 Pilot hard-normal에는 concrete와 별도의 near-threshold scenario 검토가 필요할 수 있으나 이번 milestone에서 friction을 tuning하지 않았다.

다음 단계는 별도 승인 뒤 Pilot Dataset이다. 이 작업에서는 dataset 생성, Time-to-Separation, PyTorch/model/ML 또는 E84 작업을 시작하지 않았다.
