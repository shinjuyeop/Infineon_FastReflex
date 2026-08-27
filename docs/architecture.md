# Architecture

현재 repository는 frozen deformable-support clock을 새 v3 dataset에 materialize하고 runtime sensor ablation을 마친 `SINK_SENSOR_OBSERVABILITY_PROMISING` 상태다. Selected IMU6/GRU는 holdout Recall@s1+100 ms 1.00과 median latency 78 ms를 보였지만 macro F1 0.731, SINK recall 0.786, benign sustained FP 9/13와 balanced FP 6/6으로 primary gates를 통과하지 못했다. 이 support-cell 구조는 soil model이 아닌 engineering proxy이며 actual sensor hardware, final sensor architecture, Full Hazard Dataset과 final detector는 아직 검증하지 않았다.

## MuJoCo Baseline

```text
Unitree G1 MJCF/meshes + user-supplied walking ONNX + terrain profile
                              |
                              v
                     canonical g1 simulation
                       /                 \
                      v                   v
 RuntimeTrace: IMU6 + candidate FSR8  PhysicalDiagnostics: exact state
 candidate model input boundary      label/analysis only, runtime 금지
                      \                 /
                       v               v
                 one complete run NPZ
                  + manifest/metadata
```

- `simulation/g1.py`: 29-DOF model/actuator contract, fixed policy adapter, 0.5 ms physics step, 1 kHz raw pelvis IMU/observer sampling과 in-memory smoke
- `simulation/sensors.py`: 기존 sole-terrain contact normal load를 foot-local quadrant에 합산하는 observer-only virtual FSR8
- `simulation/terrain.py`: concrete, marble, ice, sand, historical same-height compliance와 passive vertical-DOF deformable-support profile
- `simulation/hazards.py`: bilateral contact/touchdown, established Slip, historical Sink diagnostics와 causal support-surface displacement spread 계산
- `dataset/collector.py`: experiment matrix 실행, historical annotation과 v3 d0/s1 episode label, one-run-per-NPZ 저장, manifest/metadata와 SHA/structure validation
- `configs/simulator/g1.yaml`: 하나의 canonical simulator config
- `configs/dataset/hazard.yaml`: canonical raw schema와 frozen label contract

Runtime trace는 `[sequence, timestamp_us, pelvis_imu]`와 sensor-capable run의 candidate `foot_fsr`를 갖는다. FSR은 contact force scalar만 8채널로 관측하며 contact 위치/geom, exact wrench, foot state, fall censor와 oracle은 별도 diagnostics object에만 존재한다. `simulate`는 파일을 저장하지 않고, `collect`만 runtime trace와 diagnostic을 분리된 NPZ key로 materialize한다. Generated `data/raw/`는 Git ignored다.

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

Ice가 자동으로 `SLIP`이거나 sand가 자동으로 `SINK`인 규칙은 없다. Established Slip은 frozen physical metric으로 계산한다. 기존 Pilot의 primary `SINK`는 penetration precursor 뒤 frozen pelvis-tilt degradation gate까지 발생한 episode로 불변이다. V3 Sink observability clock은 실제 support joint displacement 네 개의 spread가 10 mm 이상으로 20 ms 지속된 시점이며 outcome을 사용하지 않는다. 이 의미는 별도 `sink_observability_20260827`에만 materialize했고 기존 Pilot을 고치지 않았다.

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
Candidate profile @ 1 kHz: IMU6 | FSR8 | Fusion14
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

현재 canonical implementation은 첫 PoC에 필요한 작은 MLP와 unidirectional GRU까지만 포함한다. CNN1D/LSTM은 Full Dataset 이후 동일 protocol의 full comparison 전에는 추가하지 않는다.

`training/sensor_ablation.py`는 historical 3-class Pilot path와 별도로 같은 canonical loader/model/trainer에서 v3 `NORMAL/SINK` binary study를 실행한다. Train-only normalization, three fixed seeds, validation run-balanced selection, sealed one-shot holdout와 selected ensemble의 1 ms replay를 하나의 config-driven flow로 유지한다. Generated raw dataset, checkpoints와 metrics JSON은 Git ignored다.

### Candidate sensor profiles

- `imu6`: historical pelvis IMU6 baseline
- `fsr8`: bilateral virtual FSR4×2 raw normal loads
- `fusion14`: IMU6 뒤 FSR8을 이어 붙인 raw 14 channels

Virtual FSR은 기존 sole collision contact에서만 읽으므로 geom, mass, friction, contact parameters, controller와 walking policy를 바꾸지 않는다. 40-run sensor Pilot은 기존 Pilot의 timestamp, IMU, event/censor와 모든 common diagnostic array가 bit-identical임을 강제했다. 이 결과는 idealized MuJoCo observability evidence이며 FSR hysteresis, drift, saturation, mounting과 실제 전기 변환을 검증하지 않는다. Sensor architecture는 unfrozen이다.

새 Sink-focused ablation에서는 IMU6와 Fusion14의 best validation run-balanced macro F1가 0.6986/0.6971로 near-tie였고, secondary class-recall rule로 IMU6/GRU를 선택했다. Holdout의 high balanced-soft FP 때문에 current profiles 중 어느 것도 final architecture로 채택하지 않는다.

`evaluation/fsr_distribution.py`는 raw FSR8에서 affected-foot load ratio, normalized CoP proxy, concentration과 bilateral distribution을 deterministic하게 계산하는 analysis-only responsibility다. Low-load foot ratio는 invalid로 제외하고 t0/t1 causal horizon을 run 단위로 비교한다. 이 representation은 runtime model input이나 frozen detector feature가 아니다.

`evaluation/fsr_temporal.py`는 같은 canonical distribution에서 net change와 raw 1 ms continuous-valid path를 계산하고 static horizon과 직접 비교한다. Invalid gap은 path jump로 연결하지 않으며 t2가 horizon에 포함된 run을 flag한다. Temporal representation도 analysis-only이며 feature adoption이나 detector freeze를 뜻하지 않는다.

## 초기 연구 순서

1. Dataset과 provenance를 먼저 설계한다.
2. Pilot raw sanity와 established-state MLP/GRU PoC로 기본 separability를 확인한다.
3. Onset-crossing causal window의 Time-to-Separation과 latency를 연구한다.
4. FSR Observability Pilot으로 IMU6/FSR8/Fusion14를 동일 rule에서 비교한다.
5. FSR load distribution의 t0/t1 physical observability와 terrain/phase shortcut을 분석한다.
6. Temporal redistribution이 static evidence를 앞당기는지 causal path로 검증한다.
7. Pilot evidence를 review해 sensor architecture를 결정한다.
8. 완료: Passive deformable-support proxy의 새 physical clock으로 matched balanced/uneven causal observability를 검증하고 PROMISING verdict를 기록했다.
9. 별도 승인과 설계 review 뒤에만 Full Hazard Dataset을 생성하고 동일 validation protocol로 full candidate family를 비교한다.
10. 검증된 Float model과 계약 artifact만 export한다.

첫 설계에서는 복잡한 handcrafted feature pipeline을 사용하지 않는다. Research 경계 뒤의 quantization, Vela, firmware, HIL은 E84 deployment repository가 담당한다.
