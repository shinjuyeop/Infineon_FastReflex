# MuJoCo Simulation

이 문서는 Unitree G1 MuJoCo baseline을 headless 또는 interactive viewer로 실행하는 canonical guide다. 두 mode는 같은 controller, physics, sensor와 physical-label loop를 사용하며 viewer는 rendering과 wall-clock pacing만 담당한다.

## 준비

Repository root에서 실행한다.

```bash
cd /d/shin/Infineon_FastReflex
```

Python 3.10 이상 환경에 project dependency를 설치한다.

```bash
python -m pip install -e .
```

필수 dependency는 `numpy`, `mujoco`, `onnxruntime`, `PyYAML`이다. Viewer는 공식 `mujoco` Python package의 GUI를 사용하므로 별도 GUI framework가 필요하지 않지만, viewer mode에는 사용 가능한 X11/Wayland display가 필요하다. macOS의 passive viewer는 일반 `python` 대신 `mjpython scripts/fastreflex.py ...`로 실행해야 한다.

Walking policy ONNX는 third-party binary provenance 경계를 지키기 위해 repository에 포함하지 않는다. 검증된 Unitree G1 velocity policy 파일을 준비하고 다음 두 방법 중 하나로 경로를 제공한다.

환경 변수:

```bash
export FASTREFLEX_G1_POLICY=/path/to/policy.onnx
```

또는 각 명령에 CLI option을 붙인다.

```bash
--policy /path/to/policy.onnx
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

Runtime trace는 `sequence`, `timestamp_us`, raw `pelvis_imu`만 갖는다. Foot contact, pose/velocity, penetration과 Slip/Sink oracle은 model input이 아닌 exact diagnostics다. Viewer는 canonical physics state를 별도 render buffer에 복사하므로 GUI control이나 mouse perturbation이 physics/controller/sensor/label에 전달되지 않는다.

## Headless 실행

Headless mode는 display가 없어도 동작하며 wall-clock sleep 없이 가능한 한 빠르게 실행한다.

```bash
python scripts/fastreflex.py simulate --terrain concrete --speed 0.15 --duration 2 --headless
python scripts/fastreflex.py simulate --terrain marble   --speed 0.15 --duration 2 --headless
python scripts/fastreflex.py simulate --terrain ice      --speed 0.15 --duration 2 --headless
python scripts/fastreflex.py simulate --terrain sand     --speed 0.15 --duration 2 --headless
```

아무 mode option도 지정하지 않으면 canonical config의 기본값인 headless를 사용한다. `--headless`와 `--viewer`를 동시에 지정하면 CLI error로 종료한다.

## Viewer 실행

Viewer는 공식 `mujoco.viewer.launch_passive`를 사용하고 pelvis tracking camera로 보행을 따라가며, simulation time이 wall clock과 대략 1x로 진행되도록 pacing한다. Window를 닫으면 현재까지의 trace summary를 출력하고 clean exit한다. Custom HUD나 live label overlay는 제공하지 않는다.

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

Sink/contact-compliance motion 관찰 예시:

```bash
python scripts/fastreflex.py simulate \
  --terrain sand \
  --speed 0.15 \
  --duration 10 \
  --viewer
```

환경 변수를 설정하지 않았다면 각 명령 끝에 `--policy /path/to/policy.onnx`를 추가한다. Viewer를 사용할 수 없는 환경에서는 display 확인 방법과 `--headless` 대안을 포함한 명확한 error를 출력한다.

## Terrain 설명

| Terrain | Engineering profile 의도 |
|---|---|
| `concrete` | hard, relatively high-friction reference |
| `marble` | hard contact, concrete보다 낮은 sliding friction |
| `ice` | hard contact, 매우 낮은 friction |
| `sand` | softer, damped, lower-impedance contact |

이 profile은 실제 재료 측정값이나 deformable-material model이 아니다. Relative behavior와 signal-separation을 보기 위한 engineering approximation이며 viewer를 위해 friction, `solref`, `solimp`를 바꾸지 않는다.

## Hazard label 설명

Terrain 이름과 Hazard label은 독립적이다. `concrete = NORMAL`, `ice = SLIP`, `sand = SINK` 규칙은 없다. Ice에서도 physical criterion 이전은 NORMAL candidate일 수 있고, sand에서도 criterion을 만족하지 않으면 SINK가 아니다.

- Established Slip: valid loaded contact에서 touchdown 뒤 10 ms를 제외하고, touchdown anchor 기준 tangential drift가 50 mm 이상인 상태가 3 ms 지속
- Established Sink: 같은 validity에서 first-loaded contact penetration보다 5.5 mm 이상 증가한 상태가 20 ms 지속

Sink penetration은 실제 모래 지면의 침하 깊이가 아니라 현재 MuJoCo contact-response 기반 diagnostic이다. Label은 terrain name이 아니라 [`hazards.py`](../src/fastreflex/simulation/hazards.py)의 bilateral physical metric으로 결정된다.

## 출력 해석

Simulation 종료 후 JSON summary를 stdout에 출력하며 파일이나 dataset은 생성하지 않는다.

- `expected_samples` / `actual_samples`: 요청 duration의 예상 표본과 실제 수집 표본
- `dropped_samples`, `timestamp_delta_us`: 1 kHz 연속 sampling 확인
- `established_slip_samples`, `established_sink_samples`: 좌우 foot에서 oracle이 active였던 표본 수의 합
- `max_anchor_drift_m`: contact episode의 touchdown anchor 기준 최대 수평 이동
- `max_contact_penetration_m`: named sole contact의 최대 MuJoCo penetration
- `max_loaded_penetration_change_m`: first-loaded reference 대비 최대 penetration 증가
- `first_fall_sample`, `first_fall_reasons`: 최초 fall censor 위치와 원인; 없으면 `null`과 빈 목록
- `viewer`, `terminated_by_viewer`: viewer mode 여부와 window가 duration 전에 닫혔는지 여부

Slip/Sink count는 bilateral 합계이므로 하나의 timestamp에서 양쪽 foot이 모두 active하면 simulation의 `actual_samples`보다 커질 수 있다. Viewer를 일찍 닫은 경우 `actual_samples`는 요청값보다 작지만, 수집한 구간의 timestamp가 연속이면 `dropped_samples`는 0이다.
