# MuJoCo Simulation

이 문서는 Unitree G1 MuJoCo baseline을 headless 또는 interactive viewer로 실행하는 canonical guide다. 두 mode는 같은 controller, physics, sensor와 physical-label loop를 사용하며 viewer는 rendering과 wall-clock pacing만 담당한다.

## 준비

### 최초 1회 설치

사용할 Python environment를 활성화한 뒤 아래 block 전체를 붙여 넣는다.

```bash
cd /d/shin/Infineon_FastReflex
python -m pip install -e .
```

필수 dependency는 `numpy`, `mujoco`, `onnxruntime`, `PyYAML`이다. Viewer는 공식 `mujoco` Python package의 GUI를 사용하므로 별도 GUI framework가 필요하지 않지만, viewer mode에는 사용 가능한 X11/Wayland display가 필요하다. macOS의 passive viewer는 일반 `python` 대신 `mjpython scripts/fastreflex.py ...`로 실행해야 한다.

Walking policy ONNX는 third-party binary provenance 경계를 지키기 위해 repository에 포함하지 않는다. 검증된 Unitree G1 velocity policy 파일을 별도로 준비해야 한다. Local development의 권장 위치는 `artifacts/external/unitree_g1/g1_velocity_policy.onnx`이며, 기존 `*.onnx` ignore rule에 따라 Git에는 포함되지 않는다.

### 새 terminal에서 실행 전

권장 local 위치에 policy를 준비했다면 아래 block 전체를 붙여 넣는다. 이 absolute path는 local documentation example이며 source code나 committed YAML의 default가 아니다. 마지막 명령이 파일 정보를 출력하면 준비가 끝난 것이다.

```bash
cd /d/shin/Infineon_FastReflex
export FASTREFLEX_G1_POLICY=/d/shin/Infineon_FastReflex/artifacts/external/unitree_g1/g1_velocity_policy.onnx
ls -lh "$FASTREFLEX_G1_POLICY"
```

환경 변수를 사용하지 않으려면 `--policy`를 포함한 전체 명령을 실행한다.

```bash
cd /d/shin/Infineon_FastReflex
python scripts/fastreflex.py simulate \
  --terrain concrete \
  --speed 0.15 \
  --duration 2 \
  --headless \
  --policy artifacts/external/unitree_g1/g1_velocity_policy.onnx
```

Baseline은 policy SHA-256과 `[1,98]` input, `[1,29]` output을 실행 전에 검증한다.

## 기본 구조

```text
G1 model
  -> pretrained walking policy
  -> MuJoCo physics 2 kHz
  -> raw pelvis IMU6 sampling 1 kHz
  -> simulator-only physical diagnostics
```

Runtime trace는 `sequence`, `timestamp_us`, raw `pelvis_imu`만 갖는다. Foot contact, pose/velocity, penetration, `sink_physical`, pelvis posture/velocity와 fall state는 model input이 아닌 exact diagnostics다. Viewer는 canonical physics state를 별도 render buffer에 복사하므로 GUI control이나 mouse perturbation이 physics/controller/sensor/label에 전달되지 않는다.

## Headless 실행

Headless mode는 display가 없어도 동작하며 wall-clock sleep 없이 가능한 한 빠르게 실행한다. 아래 예시는 앞에서 `FASTREFLEX_G1_POLICY`를 설정한 terminal에서 실행한다.

Concrete:

```bash
python scripts/fastreflex.py simulate \
  --terrain concrete \
  --speed 0.15 \
  --duration 2 \
  --headless
```

Marble:

```bash
python scripts/fastreflex.py simulate \
  --terrain marble \
  --speed 0.15 \
  --duration 2 \
  --headless
```

Ice:

```bash
python scripts/fastreflex.py simulate \
  --terrain ice \
  --speed 0.15 \
  --duration 2 \
  --headless
```

Sand:

```bash
python scripts/fastreflex.py simulate \
  --terrain sand \
  --speed 0.15 \
  --duration 2 \
  --headless
```

아무 mode option도 지정하지 않으면 canonical config의 기본값인 headless를 사용한다. `--headless`와 `--viewer`를 동시에 지정하면 CLI error로 종료한다.

## Viewer 실행

Viewer는 공식 `mujoco.viewer.launch_passive`를 사용하고 pelvis tracking camera로 보행을 따라가며, simulation time이 wall clock과 대략 1x로 진행되도록 pacing한다. Window를 닫으면 현재까지의 trace summary를 출력하고 clean exit한다. Custom HUD나 live label overlay는 제공하지 않는다. 아래 예시도 앞에서 `FASTREFLEX_G1_POLICY`를 설정한 terminal에서 실행한다.

NORMAL-like motion 관찰 예시:

```bash
python scripts/fastreflex.py simulate \
  --terrain concrete \
  --speed 0.15 \
  --duration 10 \
  --viewer
```

Slip motion 관찰 예시:

```bash
python scripts/fastreflex.py simulate \
  --terrain ice \
  --speed 0.15 \
  --duration 10 \
  --viewer
```

Uniform Sand control 관찰 예시:

```bash
python scripts/fastreflex.py simulate \
  --terrain sand \
  --sink-pattern uniform \
  --speed 0.15 \
  --duration 10 \
  --viewer
```

동일 높이의 left lane만 더 compliant한 Sink scenario:

```bash
python scripts/fastreflex.py simulate \
  --terrain sand \
  --sink-pattern asymmetric_left \
  --sink-severity moderate \
  --speed 0.15 \
  --duration 10 \
  --viewer
```

Right lane severe hazard candidate:

```bash
python scripts/fastreflex.py simulate \
  --terrain sand \
  --sink-pattern asymmetric_right \
  --sink-severity severe \
  --speed 0.15 \
  --duration 10 \
  --viewer
```

환경 변수를 설정하지 않았다면 각 명령 끝에 `--policy artifacts/external/unitree_g1/g1_velocity_policy.onnx`를 추가한다. Viewer를 사용할 수 없는 환경에서는 display 확인 방법과 `--headless` 대안을 포함한 명확한 error를 출력한다.

## Terrain 설명

| Terrain | Engineering profile 의도 |
|---|---|
| `concrete` | hard, relatively high-friction reference |
| `marble` | hard contact, concrete보다 낮은 sliding friction |
| `ice` | hard contact, 매우 낮은 friction |
| `sand` | softer, damped, lower-impedance contact |

이 profile은 실제 재료 측정값이나 deformable-material model이 아니다. Relative behavior와 signal-separation을 보기 위한 engineering approximation이며 viewer를 위해 friction, `solref`, `solimp`를 바꾸지 않는다.

`uniform`은 기존 scene/profile을 그대로 사용한다. `asymmetric_left`와 `asymmetric_right`는 전용 scene의 같은 높이(`z=0`)인 left/right lane 중 지정된 lane에만 더 낮은 contact impedance를 적용한다. Patch는 hole이나 step이 아니며 지면 mesh가 변형되지 않는다. Viewer의 blue/orange는 lane side를 구분하는 visual-only 색이고 severity를 뜻하지 않는다. Asymmetric pattern은 `--terrain sand`에서만 허용된다.

## Hazard label 설명

Terrain 이름과 Hazard label은 독립적이다. `concrete = NORMAL`, `ice = SLIP`, `sand = SINK` 규칙은 없다.

- Established Slip: valid loaded contact에서 touchdown 뒤 10 ms를 제외하고, touchdown anchor 기준 tangential drift가 50 mm 이상인 상태가 3 ms 지속
- `sink_physical_active`: 같은 validity에서 first-loaded contact penetration보다 5.5 mm 이상 증가한 상태가 20 ms 지속

`sink_physical_active`는 물리적 침하 precursor diagnostic이며 primary `SINK` class가 아니다. Primary `SINK`는 physical sink와 meaningful locomotion/posture degradation이 함께 있는 hazard지만 최종 numeric gate는 아직 `SINK_HAZARD_CRITERIA_NOT_YET_FROZEN`이다. Uniform sand처럼 penetration이 있어도 안정적으로 걷는 상태는 그 자체만으로 `SINK`가 아니다.

## 출력 해석

Simulation 종료 후 JSON summary를 stdout에 출력하며 파일이나 dataset은 생성하지 않는다.

- `expected_samples` / `actual_samples`: 요청 duration의 예상 표본과 실제 수집 표본
- `dropped_samples`, `timestamp_delta_us`: 1 kHz 연속 sampling 확인
- `established_slip_samples`: 좌우 foot에서 frozen Slip oracle이 active였던 표본 수의 합
- `sink_physical_samples_per_foot`, `first_sink_physical_sample_per_foot`: 좌우 physical precursor의 active count와 onset
- `max_anchor_drift_m`: contact episode의 touchdown anchor 기준 최대 수평 이동
- `max_contact_penetration_m_per_foot`, `max_loaded_penetration_change_m_per_foot`: 좌우 침투와 first-loaded reference 대비 증가
- `max_bilateral_loaded_penetration_asymmetry_m`: 양발이 loaded인 구간의 최대 좌우 침투 차이
- `max_pelvis_tilt_deg`, `pelvis_z_range_m`, `peak_pelvis_angular_speed_rad_s`: posture disturbance candidate
- `mean_pelvis_forward_velocity_m_s`, `forward_velocity_rmse_m_s`: commanded forward speed 대비 gait effect candidate
- `loaded_contact_samples_per_foot`, `loaded_contact_imbalance_samples`: contact-duration disturbance candidate
- `first_fall_sample`, `first_fall_reasons`: 최초 fall censor 위치와 원인; 없으면 `null`과 빈 목록
- `viewer`, `terminated_by_viewer`: viewer mode 여부와 window가 duration 전에 닫혔는지 여부

Slip/physical-sink count를 양발 합계로 해석할 때는 하나의 timestamp에서 양쪽 foot이 모두 active하면 simulation의 `actual_samples`보다 커질 수 있다. Physical-sink sample count는 contact episode와 validity의 영향을 받으므로 severity score가 아니다. Viewer를 일찍 닫은 경우 `actual_samples`는 요청값보다 작지만, 수집한 구간의 timestamp가 연속이면 `dropped_samples`는 0이다.
