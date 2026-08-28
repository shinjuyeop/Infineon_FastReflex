# Architecture

현재 primary target은 Terrain Recognition + Walking Stability Detection + deterministic State Fusion이다. `TERRAIN_RECOGNITION_SUPPORTED` milestone에서 current calibrated transition simulator로 four-class Terrain dataset을 새로 만들고 FSR4/Foot IMU6/Fusion10, MLP/GRU, 20/30/50 ms와 LEFT_ONLY/BILATERAL_SHARED를 validation-only selection 뒤 one-shot holdout으로 평가했다. Current Terrain research candidate는 FSR4/MLP/50 ms/LEFT_ONLY다. Historical phase-aware exact MoS clock과 Stability detector는 계속 실패 상태이며 final sensor architecture는 freeze하지 않았다. 이전 `SINK_SENSOR_OBSERVABILITY_PROMISING`과 direct NORMAL/SLIP/SINK 연구는 historical evidence로 보존한다.

## MuJoCo Baseline

```text
Unitree G1 MJCF/meshes + user-supplied walking ONNX + terrain profile
                              |
                              v
                     canonical g1 simulation
                       /                 \
                      v                   v
 RuntimeTrace: pelvis IMU6 + FSR8   PhysicalDiagnostics: exact state
             + Foot IMU12
 candidate model input boundary      label/analysis only, runtime 금지
                      \                 /
                       v               v
                 one complete run NPZ
                  + manifest/metadata
```

- `simulation/g1.py`: 29-DOF model/actuator contract, fixed policy adapter, 0.5 ms physics step, 1 kHz raw pelvis IMU/observer sampling, exact stability capture와 in-memory smoke
- `simulation/sensors.py`: sole-terrain contact normal load의 observer-only virtual FSR8, bilateral ankle-roll Foot IMU12와 label-only exact terrain contact observer
- `simulation/terrain.py`: concrete, marble, ice, sand, historical same-height compliance와 passive vertical-DOF deformable-support profile
- `simulation/hazards.py`: bilateral contact/touchdown, established Slip, historical Sink diagnostics와 causal support-surface displacement spread 계산
- `simulation/stability.py`: privileged whole-body COM/XCoM/support MoS, stable envelope, pelvis-IMU rule, independent runtime state와 fusion truth table
- `dataset/collector.py`: experiment matrix 실행, historical annotation과 v3 d0/s1 episode label, one-run-per-NPZ 저장, manifest/metadata와 SHA/structure validation
- `dataset/terrain.py`: 144-run Terrain matrix, exact-touchdown clean/mixed indexing, per-foot sensor slicing, train-only normalization과 sealed holdout guard
- `training/terrain.py`: Terrain sensor/family/horizon/deployment validation selection과 one-shot holdout
- `configs/simulator/g1.yaml`: 하나의 canonical simulator config
- `configs/dataset/hazard.yaml`, `configs/dataset/terrain.yaml`: independent Hazard/Terrain schema와 label contract

Runtime trace는 `[sequence, timestamp_us, pelvis_imu]`와 candidate `foot_fsr`, `foot_imu`를 갖는다. FSR은 contact force scalar만 8채널로 관측하고 Foot IMU는 left/right ankle-roll-local accel3/gyro3를 관측한다. Contact 위치/geom, exact terrain identity, exact wrench, foot state, fall censor와 oracle은 별도 diagnostics object에만 존재한다. `simulate`는 파일을 저장하지 않고, `collect`만 runtime trace와 diagnostic을 분리된 NPZ key로 materialize한다. Generated `data/raw/`는 Git ignored다.

Optional viewer는 canonical physics state를 별도 MuJoCo render model/data에 복사해 약 60 Hz로 sync한다. GUI input은 render copy에만 머물고 physics loop에는 돌아오지 않으며, wall-clock pacing도 viewer mode에만 적용한다. 사용법은 [`simulation.md`](simulation.md)에 둔다.

Transition audit에서만 optional `SimulationStateTrace`가 robot qpos/qvel, controller observation/action/update timing, pelvis pose와 COM을 simulator-only로 capture한다. 이는 matched-prefix 검증 전용이며 runtime model input이나 dataset field가 아니다. A-side는 Concrete/Marble 중 하나이고 B-side Ice/Sand geom은 contact 전 robot dynamics에 영향을 주지 않아야 한다. Matched reference는 같은 patch scene에서 B start만 run 밖으로 옮겨 topology 차이를 제거한다.

Physics는 0.5 ms(2 kHz)이고 두 step마다 raw MuJoCo pelvis accelerometer/gyro를 읽어 1 ms(1 kHz), `float32`, `[accel_x/y/z, gyro_x/y/z]`로 제공한다. `imu` site는 pelvis 원점에 무회전으로 결합되어 local +x/+y/+z가 전방/좌측/위쪽이다. Timestamp는 run-local monotonic `int64` µs다.

### Terrain profiles

네 profile은 실제 재료 측정값이나 deformable-material model이 아니다. Relative signal-separation과 pipeline 검증을 위한 engineering approximation이며 terrain identity는 hazard label이 아니다.

| Profile | Sliding friction | Contact character |
|---|---:|---|
| concrete | 1.00 | hard, high-friction reference |
| marble | 0.45 | hard, lower friction |
| ice | 0.05 | hard, very low friction |
| sand | 0.70 | softer, damped contact impedance |

Ice가 자동으로 `SLIP`이거나 sand가 자동으로 `SINK`인 규칙은 없다. Established Slip은 frozen physical metric으로 계산한다. 기존 Pilot의 primary `SINK`는 penetration precursor 뒤 frozen pelvis-tilt degradation gate까지 발생한 episode로 불변이다. V3 Sink observability clock은 실제 support joint displacement 네 개의 spread가 10 mm 이상으로 20 ms 지속된 시점이며 outcome을 사용하지 않는다. 이 의미는 별도 `sink_observability_20260827`에만 materialize했고 기존 Pilot을 고치지 않았다.

### Asset and policy boundary

Walking에 필요한 G1 MJCF와 참조되는 36개 mesh만 BSD 3-Clause license/NOTICE와 함께 명시적으로 migration했다. Legacy sensor 선언을 복사하지 않고 current G1 ankle-roll body에 bilateral Foot IMU observer site를 새 contract로 추가했다. Matched hard/Ice/Sand run에서 sensor read 전후 physics/controller/diagnostic이 exact parity였다. Ankle force/torque sensor는 없고 rigid-body dynamics, inertial, joint, actuator와 sole contact geom은 유지했다. Policy binary는 vendoring하지 않으며 검증된 SHA-256과 ONNX input/output contract가 일치하는 user-supplied artifact만 실행한다.

## Current Rebuilt Terrain

```text
touchdown foot FSR4, 50 ms
  -> shared MLP Terrain Classifier
  -> Concrete / Marble / Ice / Sand
```

`terrain_transition_20260828`은 exact contact geom identity를 GT로 쓰되 runtime tensor에는 한 발의 FSR4/Foot IMU6/Fusion10만 제공한다. Primary validation에서 only FSR4가 macro F1/worst recall 0.90/0.85 gate를 모두 통과했고, MLP와 50 ms가 선택됐다. LEFT_ONLY one-shot holdout macro F1 0.9713/worst recall 0.95로 current research candidate가 됐다.

Terrain prediction은 clean touchdown 뒤 selected 50 ms observation에서 state를 갱신하고 다음 valid event까지 hold한다. LEFT_ONLY는 Pelvis IMU6 포함 10 channels지만 right-only Sand 18/144 runs에서 update가 없고 median delay 1114.5 ms다. BILATERAL_SHARED는 14 channels, 144/144 coverage, median 922 ms다. Recommendation은 `LEFT_FSR4_RECOMMENDED`이나 Stability와 actual hardware validation 전 final freeze는 아니다. Legacy Terrain v4는 source/data/model을 재사용하지 않은 historical comparison이다.

## Terrain + Stability + Fusion target

```text
                     runtime streams
                     /             \
                    v               v
 touchdown-centered Terrain    continuous Stability
  CONCRETE/MARBLE/ICE/SAND     STABLE/UNSTABLE
                    \               /
                     v             v
                    deterministic fusion
      NORMAL / SLIP_RISK / SINK_RISK / GENERIC_INSTABILITY
                         + RECOVERY_REQUIRED
```

Terrain과 Stability producer는 독립 update한다. Terrain은 valid update를 다음 touchdown까지 hold하고 Stability event가 terrain inference를 재시작시키지 않는다. `UNKNOWN + UNSTABLE`도 `GENERIC_INSTABILITY`, recovery true다.

Stability exact ground truth와 runtime detector는 분리한다.

```text
MuJoCo exact COM, COM velocity, loaded sole polygon
  -> XCoM and signed dynamic Margin of Stability
  -> stable-control phase lower envelope
  -> t_instability diagnostic

Pelvis IMU6 only
  -> deterministic causal rule
  -> optional small GRU only after exact-clock acceptance
```

First predeclared exact clock은 phase lower 0.5 percentile, additional 10 mm degradation, 20 ms persistence였다. Stable FP 45.45%와 fall coverage 66.67%로 실패했으므로 AI target으로 사용할 수 없다. IMU rule도 accepted holdout에서 stable FP 2/3, Recall@100 ms 0, median latency 353.5 ms였다. Threshold sweep이나 GRU training으로 실패를 덮지 않았다.

Historical direct-classification candidate families는 evidence 재현용으로 남는다.

- MLP baseline
- CNN1D
- GRU
- LSTM

현재 canonical implementation은 historical PoC와 future accepted stability baseline에 재사용할 작은 MLP/GRU만 포함한다. CNN1D/LSTM은 revised stability clock과 dataset acceptance 전에는 추가하지 않는다.

`training/sensor_ablation.py`는 historical 3-class Pilot path와 별도로 같은 canonical loader/model/trainer에서 v3 `NORMAL/SINK` binary study를 실행한다. Train-only normalization, three fixed seeds, validation run-balanced selection, sealed one-shot holdout와 selected ensemble의 1 ms replay를 하나의 config-driven flow로 유지한다. Generated raw dataset, checkpoints와 metrics JSON은 Git ignored다.

### Candidate sensor profiles

- `imu6`: historical pelvis IMU6 baseline
- `fsr8`: bilateral virtual FSR4×2 raw normal loads
- `fusion14`: IMU6 뒤 FSR8을 이어 붙인 raw 14 channels

Terrain-specific profiles는 touchdown한 한 발만 사용한다.

- `fsr4`: 해당 foot raw quadrant loads
- `foot_imu6`: 해당 foot ankle-roll-local accel3/gyro3
- `fusion10`: FSR4 뒤 Foot IMU6

Side ID와 pelvis IMU6는 Terrain tensor에 넣지 않는다. Same shared model을 양발에 적용하는 BILATERAL_SHARED도 model input은 4/6/10 channels로 유지한다.

Virtual FSR은 기존 sole collision contact에서만 읽으므로 geom, mass, friction, contact parameters, controller와 walking policy를 바꾸지 않는다. 40-run sensor Pilot은 기존 Pilot의 timestamp, IMU, event/censor와 모든 common diagnostic array가 bit-identical임을 강제했다. 이 결과는 idealized MuJoCo observability evidence이며 FSR hysteresis, drift, saturation, mounting과 실제 전기 변환을 검증하지 않는다. Sensor architecture는 unfrozen이다.

새 Sink-focused ablation에서는 IMU6와 Fusion14의 best validation run-balanced macro F1가 0.6986/0.6971로 near-tie였고, secondary class-recall rule로 IMU6/GRU를 선택했다. Holdout의 high balanced-soft FP 때문에 current profiles 중 어느 것도 final architecture로 채택하지 않는다.

`evaluation/fsr_distribution.py`는 raw FSR8에서 affected-foot load ratio, normalized CoP proxy, concentration과 bilateral distribution을 deterministic하게 계산하는 analysis-only responsibility다. Low-load foot ratio는 invalid로 제외하고 t0/t1 causal horizon을 run 단위로 비교한다. 이 representation은 runtime model input이나 frozen detector feature가 아니다.

`evaluation/fsr_temporal.py`는 같은 canonical distribution에서 net change와 raw 1 ms continuous-valid path를 계산하고 static horizon과 직접 비교한다. Invalid gap은 path jump로 연결하지 않으며 t2가 horizon에 포함된 run을 flag한다. Temporal representation도 analysis-only이며 feature adoption이나 detector freeze를 뜻하지 않는다.

## 현재 연구 순서

1. 완료된 direct NORMAL/SLIP/SINK Pilot, FSR와 deformable-support evidence를 historical baseline으로 보존한다.
2. Terrain과 Stability producer, deterministic fusion과 leakage boundary를 유지한다.
3. 완료된 calibrated transition operating points와 matched-prefix contract를 유지한다.
4. Supported rebuilt Terrain candidate와 LEFT_ONLY/BILATERAL tradeoff를 유지하되 final sensor architecture는 freeze하지 않는다.
5. Clean transition scenarios에서 stable-domain phase/contact representation으로 exact `t_instability`를 다시 사전 선언하고 stable FP/fall coverage/lead gate를 통과시킨다.
6. Gate 통과 뒤 같은 pelvis IMU6 deterministic rule과 small GRU를 동일 holdout에서 비교한다.
7. 이후 Terrain+Stability evidence를 함께 review해 final sensor architecture와 Float release 여부를 별도 승인한다.

첫 설계에서는 복잡한 handcrafted feature pipeline을 사용하지 않는다. Research 경계 뒤의 quantization, Vela, firmware, HIL은 E84 deployment repository가 담당한다.
