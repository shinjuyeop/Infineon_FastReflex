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
  -> raw pelvis IMU6 + bilateral FSR8 + Foot IMU12 sampling 1 kHz
  -> simulator-only physical diagnostics
```

Runtime trace는 `sequence`, `timestamp_us`, raw `pelvis_imu`, optional `foot_fsr`와 `foot_imu`를 갖는다. Foot IMU order는 left accel3/gyro3 뒤 right accel3/gyro3다. Foot contact, exact terrain geom identity, pose/velocity, penetration, `sink_physical`, pelvis posture/velocity와 fall state는 model input이 아닌 exact diagnostics다. Viewer는 canonical physics state를 별도 render buffer에 복사하므로 GUI control이나 mouse perturbation이 physics/controller/sensor/label에 전달되지 않는다.

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

Viewer는 공식 `mujoco.viewer.launch_passive`를 사용하고 pelvis tracking camera로 보행을 따라가며, simulation time이 wall clock과 대략 1x로 진행되도록 pacing한다. Window를 닫으면 현재까지의 trace summary를 출력하고 clean exit한다. 일반 `simulate --viewer`는 custom HUD를 표시하지 않으며, frozen decision HUD는 아래의 `visualize` workflow가 제공한다. 아래 예시도 앞에서 `FASTREFLEX_G1_POLICY`를 설정한 terminal에서 실행한다.

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

정상 concrete 보행 뒤 full-width finite Ice patch에 진입하는 Slip transition:

```bash
python scripts/fastreflex.py simulate \
  --terrain ice \
  --slip-pattern transition \
  --speed 0.15 \
  --duration 8 \
  --viewer
```

Marble prefix에서 같은 finite Ice patch를 보는 robustness replay:

```bash
python scripts/fastreflex.py simulate \
  --source-terrain marble \
  --terrain ice \
  --slip-pattern transition \
  --patch-start-x 0.36 \
  --patch-width 0.70 \
  --speed 0.25 \
  --duration 8 \
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

정상 concrete 보행 뒤 finite left soft patch에 진입하는 transition Sink:

```bash
python scripts/fastreflex.py simulate \
  --terrain sand \
  --sink-pattern transition_left \
  --sink-severity severe \
  --speed 0.15 \
  --duration 8 \
  --viewer
```

환경 변수를 설정하지 않았다면 각 명령 끝에 `--policy artifacts/external/unitree_g1/g1_velocity_policy.onnx`를 추가한다. Viewer를 사용할 수 없는 환경에서는 display 확인 방법과 `--headless` 대안을 포함한 명확한 error를 출력한다.

## Visualizing supported Hazard/Terrain decisions

### Purpose

`visualize`는 frozen unified dataset의 TRAIN 또는 VALIDATION run specification을 읽어 같은 MuJoCo 조건을 결정론적으로 다시 실행하고, 저장된 runtime trace와 parity가 통과한 경우에만 viewer를 연다. Viewer에는 frozen Hazard와 advisory Terrain inference, 그리고 별도로 표시된 simulator-only physical GT/diagnostic을 같은 simulation clock으로 겹쳐 보여 준다. 성능을 다시 평가하거나 dataset/model을 변경하는 workflow가 아니다.

Stored NPZ는 완전한 robot `qpos/qvel` trajectory를 갖고 있지 않으므로 NPZ 자체를 animation으로 가장하지 않는다. Viewer를 열기 전에 다음 항목을 exact equality, 즉 absolute tolerance `0.0`으로 비교한다.

- `timestamp_us`
- Pelvis IMU6
- FSR8
- first target contact/touchdown
- Slip event, Support event, I1 precursor event
- censor sample
- tangential drift, support spread, loaded-contact diagnostic trace

Parity 하나라도 실패하면 viewer는 열리지 않는다. Viewer replay가 끝난 뒤에도 저장 trace와 다시 비교해 rendering/pacing이 physics에 영향을 주지 않았음을 확인한다.

### Basic command

Canonical policy가 `artifacts/external/unitree_g1/g1_velocity_policy.onnx`에 있으면 다음 명령만 실행하면 된다.

```bash
python scripts/fastreflex.py visualize --run-id <RUN_ID>
```

다른 위치에 같은 verified policy가 있다면 `FASTREFLEX_G1_POLICY` 또는 `--policy`를 사용한다. 재생 속도는 `--speed 0.5`, `--speed 1.0`, `--speed 2.0` 중 하나이며 기본값은 near-real-time `1.0`이다. `--show-debug`는 threshold 상태, tensor provenance와 parity 설명을 HUD에 추가한다.

사용 가능한 TRAIN/VALIDATION ID와 canonical representative ID는 다음 명령으로 확인한다. 이 목록에는 HOLDOUT ID가 포함되지 않는다.

```bash
python scripts/fastreflex.py visualize --list-runs
```

### Representative validation cases

선정 규칙은 model confidence나 화면 모양을 보지 않고, 각 design group에서 manifest상 valid한 VALIDATION run ID를 사전순으로 정렬한 첫 항목을 택하는 것이다. 현재 frozen manifest에서 선택된 실제 ID는 다음과 같다.

ICE_SLIP_HAZARD — `uhr_ice_h_c20`:

```bash
python scripts/fastreflex.py visualize --run-id uhr_ice_h_c20
```

SAND_SUPPORT_HAZARD — `uhr_sand_h_c20`:

```bash
python scripts/fastreflex.py visualize --run-id uhr_sand_h_c20
```

SAND_BENIGN — `uhr_sand_b_c20`:

```bash
python scripts/fastreflex.py visualize --run-id uhr_sand_b_c20
```

HARD_GROUND_NORMAL — `uhr_hard_n_c20`:

```bash
python scripts/fastreflex.py visualize --run-id uhr_hard_n_c20
```

### Overlay interpretation

왼쪽 `MODEL OUTPUT` panel은 model/runtime output만 표시한다.

- 현재 Hazard probability, `p >= 0.99`, `REFLEX_REQUIRED`, 첫 reflex time
- held Terrain state, latest update time, touchdown foot
- Concrete/Marble/Ice/Sand probability
- Hazard 결정 뒤에만 계산되는 advisory cause refinement

오른쪽 `SIMULATOR GT / DIAGNOSTIC` panel은 `NEVER USED AS MODEL INPUT`이라고 명시하며 다음 simulator-only reference를 표시한다.

- physical label
- 현재 Slip active와 first event, tangential drift, 50 mm / 3 ms 기준
- 현재 I1 precursor active와 first event
- 현재 established Support active와 first event, support spread, 10 mm / 20 ms 기준
- censor time

Terrain은 Hazard tensor, threshold, persistence 또는 `REFLEX_REQUIRED` gate에 들어가지 않는다. GT와 diagnostic도 model input이 아니며 parity와 시각적 해석에만 사용한다.

### Expected sequence

- Ice Slip: normal walking → Ice contact → Slip progression 중 Hazard probability 상승 → `REFLEX_REQUIRED` → 이후 Terrain이 ICE로 갱신된다. Reflex가 Terrain을 기다리지 않는 것을 확인한다.
- Sand Support: normal walking → Sand contact → heterogeneous support deformation → I1 precursor → `REFLEX_REQUIRED` → established Support 순서로 진행할 수 있다.
- Sand benign: Terrain은 SAND로 갱신되지만 Hazard probability는 benign이고 `REFLEX_REQUIRED`는 false로 유지된다.
- Hard normal: Terrain은 CONCRETE 또는 MARBLE이고 Hazard probability와 `REFLEX_REQUIRED`가 benign 상태를 유지한다.

### Controls

MuJoCo passive viewer의 기본 mouse camera orbit/pan/zoom control을 그대로 사용한다. Window를 닫으면 replay가 종료된다. Replay 완료 전에 닫으면 최종 viewer/physics parity를 확인할 수 없으므로 command는 fail closed한다. 재생 속도는 command 시작 시 `--speed`로 선택한다.

### Limitations

- Visualization은 stored NPZ trajectory 재생이 아니라 frozen scenario의 deterministic re-simulation이다.
- Stored NPZ에는 전체 `qpos/qvel` replay가 없으며 parity를 통과하지 못한 근사 animation은 허용하지 않는다.
- Physical GT와 diagnostics는 visualization/evaluation reference일 뿐 model tensor가 아니다.
- 이 결과는 simulation evidence이며 real-robot validation이 아니다.
- HOLDOUT run ID는 명시적으로 거부하며 waveform을 열지 않는다.

## Terrain 설명

| Terrain | Engineering profile 의도 |
|---|---|
| `concrete` | hard, relatively high-friction reference |
| `marble` | hard contact, concrete보다 낮은 sliding friction |
| `ice` | hard contact, 매우 낮은 friction |
| `sand` | softer, damped, lower-impedance contact |

이 profile은 실제 재료 측정값이나 deformable-material model이 아니다. Relative behavior와 signal-separation을 보기 위한 engineering approximation이며 viewer를 위해 friction, `solref`, `solimp`를 바꾸지 않는다.

Terrain event GT는 named sole과 실제 ground geom의 exact identity다. Ice transition의 pre/post는 selected Concrete/Marble source, full-width patch는 Ice다. Sand transition은 affected static/deformable cells만 Sand이고 반대 lane과 pre/post는 source terrain이다. 이 geom mapping은 label/diagnostic 전용이며 runtime tensor에 노출하지 않는다.

`uniform`은 기존 scene/profile을 그대로 사용한다. Sink의 `asymmetric_left/right`와 `transition_left/right`는 `scene_sink.xml`에서 한쪽 compliance를 바꾼다. Slip의 `--slip-pattern transition`도 같은 finite topology를 재사용하되 기본 `x=[0.35,1.10] m`의 left/right patch를 모두 기존 Ice profile로 설정하고 그 전후는 concrete로 둔다. Pilot variation에는 `--patch-start-x`와 `--patch-width`를 사용하며 default는 각각 0.35 m와 0.75 m다. 모든 경계의 nominal top은 `z=0`이고 box가 맞닿을 뿐 겹치지 않는다. Hole, step, lowered surface 또는 deformable mesh는 없다. Cyan Ice patch와 Sink의 blue/orange는 visual-only이며 physics/label selection에 사용하지 않는다. Sink non-uniform pattern은 `--terrain sand`, Slip transition은 `--terrain ice`에서만 허용되고 두 pattern을 결합할 수 없다.

## Hazard label 설명

Terrain 이름과 Hazard label은 독립적이다. `concrete = NORMAL`, `ice = SLIP`, `sand = SINK` 규칙은 없다.

- Established Slip: valid loaded contact에서 touchdown 뒤 10 ms를 제외하고, touchdown anchor 기준 tangential drift가 50 mm 이상인 상태가 3 ms 지속
- `sink_physical_active`: 같은 validity에서 first-loaded contact penetration보다 5.5 mm 이상 증가한 상태가 20 ms 지속

`sink_physical_active`는 물리적 침하 precursor diagnostic이며 primary `SINK` class가 아니다. Frozen primary effect gate는 patch-linked physical sink 뒤 pelvis tilt가 benign-control upper envelope `0.04454633221030235 rad`를 초과한 상태가 20 ms 지속되는 것이다. Uniform sand처럼 penetration이 있어도 안정적으로 걷는 상태는 그 자체만으로 `SINK`가 아니다. Hazardous episode의 early-detection reference 후보는 effect가 명확해진 t2가 아니라 physical onset t1이며 실제 IMU label/window timing은 Pilot/Time-to-Separation 전까지 미확정이다.

Terrain Recognition에서는 이 Hazard label을 쓰지 않는다. Clean terrain touchdown 뒤 causal 20/30/50 ms sensor window를 구성하고, primary 50 ms 동안 same identity contact가 유지되며 other-terrain sample ratio가 20% 미만인 event만 eligible하다. Boundary-straddling event는 `AMBIGUOUS_BOUNDARY`로 제외한다.

## 출력 해석

Simulation 종료 후 JSON summary를 stdout에 출력하며 파일이나 dataset은 생성하지 않는다.

- `expected_samples` / `actual_samples`: 요청 duration의 예상 표본과 실제 수집 표본
- `dropped_samples`, `timestamp_delta_us`: 1 kHz 연속 sampling 확인
- `established_slip_samples`: 좌우 foot에서 frozen Slip oracle이 active였던 표본 수의 합
- `first_low_friction_patch_contact_sample_per_foot`: finite Ice patch의 per-foot t0
- `first_established_slip_after_patch_sample_per_foot`, `first_any_established_slip_after_patch_sample`: per-foot과 ANY-SLIP t1
- `slip_transition_qualification`: clean/no-slip/unusable transition classification
- `sink_physical_samples_per_foot`, `first_sink_physical_sample_per_foot`: 좌우 physical precursor의 active count와 onset
- `first_soft_patch_contact_sample_per_foot`, `first_sink_physical_after_patch_sample_per_foot`: transition t0와 같은 contact episode의 t1
- `first_sink_degradation_sample`, `first_sink_hazard_sample`: frozen tilt persistence와 patch-linked t2 onset
- `sink_episode_qualification`, `dual_phenomenon`: benign/hazardous/inconclusive episode와 Slip 선행 여부
- `max_anchor_drift_m`: contact episode의 touchdown anchor 기준 최대 수평 이동
- `max_contact_penetration_m_per_foot`, `max_loaded_penetration_change_m_per_foot`: 좌우 침투와 first-loaded reference 대비 증가
- `max_bilateral_loaded_penetration_asymmetry_m`: 양발이 loaded인 구간의 최대 좌우 침투 차이
- `max_pelvis_tilt_deg`, `pelvis_z_range_m`, `peak_pelvis_angular_speed_rad_s`: posture disturbance candidate
- `mean_pelvis_forward_velocity_m_s`, `forward_velocity_rmse_m_s`: commanded forward speed 대비 gait effect candidate
- `loaded_contact_samples_per_foot`, `loaded_contact_imbalance_samples`: contact-duration disturbance candidate
- `pre_event_*`, `max_*_from_pre_event`: t0 전 최대 1,000 ms baseline과 event-relative pelvis z/tilt/velocity/angular-speed 변화
- `first_fall_sample`, `first_fall_reasons`: 최초 fall censor 위치와 원인; 없으면 `null`과 빈 목록
- `viewer`, `terminated_by_viewer`: viewer mode 여부와 window가 duration 전에 닫혔는지 여부

Slip/physical-sink count를 양발 합계로 해석할 때는 하나의 timestamp에서 양쪽 foot이 모두 active하면 simulation의 `actual_samples`보다 커질 수 있다. Physical-sink sample count는 contact episode와 validity의 영향을 받으므로 severity score가 아니다. Viewer를 일찍 닫은 경우 `actual_samples`는 요청값보다 작지만, 수집한 구간의 timestamp가 연속이면 `dropped_samples`는 0이다.
