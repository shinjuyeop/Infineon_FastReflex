# Architecture

현재 repository는 `MUJOCO_BASELINE_READY` 상태다. 최소 Unitree G1 simulator와 Hazard Dataset Contract foundation만 구현했으며 dataset, model, training은 아직 구현하지 않았다.

## MuJoCo Baseline

```text
Unitree G1 MJCF/meshes + user-supplied walking ONNX + terrain profile
                              |
                              v
                     canonical g1 simulation
                       /                 \
                      v                   v
 RuntimeTrace: pelvis IMU6 only   PhysicalDiagnostics: exact state
 future model input boundary      label/analysis only, runtime 금지
```

- `simulation/g1.py`: 29-DOF model/actuator contract, fixed policy adapter, 0.5 ms physics step, 1 kHz raw pelvis IMU sampling과 in-memory smoke
- `simulation/terrain.py`: concrete, marble, ice, sand와 same-height asymmetric compliance profile
- `simulation/hazards.py`: bilateral contact/touchdown, foot cause metric, established Slip, `sink_physical` precursor와 pelvis effect diagnostic 계산
- `configs/simulator/g1.yaml`: 하나의 canonical simulator config

Runtime trace는 `[sequence, timestamp_us, pelvis_imu]`만 갖는다. Exact contact, force/load, foot state, fall censor와 oracle은 별도 diagnostics object에만 존재한다. Smoke command는 dataset이나 report artifact를 저장하지 않는다.

Optional viewer는 canonical physics state를 별도 MuJoCo render model/data에 복사해 약 60 Hz로 sync한다. GUI input은 render copy에만 머물고 physics loop에는 돌아오지 않으며, wall-clock pacing도 viewer mode에만 적용한다. 사용법은 [`simulation.md`](simulation.md)에 둔다.

Physics는 0.5 ms(2 kHz)이고 두 step마다 raw MuJoCo pelvis accelerometer/gyro를 읽어 1 ms(1 kHz), `float32`, `[accel_x/y/z, gyro_x/y/z]`로 제공한다. `imu` site는 pelvis 원점에 무회전으로 결합되어 local +x/+y/+z가 전방/좌측/위쪽이다. Timestamp는 run-local monotonic `int64` µs다.

### Terrain profiles

네 profile은 실제 재료 측정값이나 deformable-material model이 아니다. Relative signal-separation과 pipeline 검증을 위한 engineering approximation이며 terrain identity는 hazard label이 아니다.

| Profile | Sliding friction | Contact character |
|---|---:|---|
| concrete | 1.00 | hard, high-friction reference |
| marble | 0.45 | hard, lower friction |
| ice | 0.05 | hard, very low friction |
| sand | 0.70 | softer, damped contact impedance |

Ice가 자동으로 `SLIP`이거나 sand가 자동으로 `SINK`인 규칙은 없다. Established Slip은 frozen physical metric으로 계산하고, penetration persistence는 `sink_physical` precursor로만 사용한다. Primary `SINK`의 meaningful posture/gait degradation gate는 아직 frozen되지 않았다.

### Asset and policy boundary

Walking에 필요한 G1 MJCF와 참조되는 36개 mesh만 BSD 3-Clause license/NOTICE와 함께 명시적으로 migration했다. 과거 연구용 bilateral foot IMU와 ankle force/torque sensor는 제거했으며 rigid-body dynamics, inertial, joint, actuator와 sole contact geom은 유지했다. Policy binary는 vendoring하지 않으며 검증된 SHA-256과 ONNX input/output contract가 일치하는 user-supplied artifact만 실행한다.

## Existing Frozen Terrain

```text
Foot FSR4 + Foot IMU6
  -> Frozen Terrain Classifier
  -> Concrete / Marble / Ice / Sand
```

Terrain classifier는 legacy repository에서 검증된 별도 asset이다. 향후 provenance를 갖춘 frozen release로 명시적 검토 후 migration한다. 현재 baseline에는 classifier 코드나 model이 없다.

## New Hazard Model

```text
Waist/Pelvis IMU6 @ 1 kHz
  -> Raw causal sequence
  -> Minimal normalization
  -> PyTorch temporal model
  -> NORMAL / SLIP / SINK
```

Candidate model families:

- MLP baseline
- CNN1D
- GRU
- LSTM

## 초기 연구 순서

1. Dataset과 provenance를 먼저 설계한다.
2. Raw signal과 label 가능성을 분석한다.
3. Causal window와 latency를 연구한다.
4. 동일한 validation protocol로 candidate model을 비교한다.
5. 검증된 Float model과 계약 artifact만 export한다.

첫 설계에서는 복잡한 handcrafted feature pipeline을 사용하지 않는다. Research 경계 뒤의 quantization, Vela, firmware, HIL은 E84 deployment repository가 담당한다.
