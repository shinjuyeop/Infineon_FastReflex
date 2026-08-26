# Minimal G1 MuJoCo Migration — 2026-08-26

## 목적과 범위

`MINIMAL_G1_MUJOCO_MIGRATION`은 Hazard Dataset Contract를 실행할 수 있는 최소 Unitree G1 walking foundation을 Research repository에 만든다. 시작 HEAD는 `ee428e0818d14dc030351eb264c6087f0d17baac`이고, legacy `/d/shin/Infineon`은 commit `4194af1e0d8db8d113609c11879713c29a583261`에서 read-only로 audit했다.

Migration한 항목은 다음과 같다.

- Unitree G1 29-DOF MJCF, scene, walking에 참조되는 36개 STL mesh
- 29 actuator order, default pose, PD gain, action scale, 20 ms fixed ONNX policy adapter
- pelvis `imu` site의 raw accelerometer/gyro reader
- concrete/marble/ice/sand contact profile
- bilateral sole contact/force, touchdown, foot world pose/velocity, episode, anchor drift, penetration 변화, pre-fall censor와 established Slip/Sink helper
- canonical simulator config, 기존 CLI의 `simulate`, 하나의 simulation test file

Legacy Slip/Sink detector, rejector/state machine, residual/Fusion20, Foot IMU/FSR/ankle F/T 연구 경로, terrain classifier, dataset/output, ML model/training, experiment runner, firmware/HIL/quantization은 migration하지 않았다.

## Asset와 policy provenance

G1 asset은 legacy의 `simulation/unitree_mujoco/unitree_robots/g1/`에서 명시적으로 가져왔다. Origin은 Unitree Robotics의 [`unitree_mujoco`](https://github.com/unitreerobotics/unitree_mujoco)이며, 동봉한 `LICENSE`는 BSD 3-Clause다. Migrated `g1.xml` SHA-256은 `aa23e81491ce19e40764b5db801c02a195ba8934d3695be9b80f939ed371a292`다.

Legacy model에서 연구용으로 추가됐던 bilateral foot IMU site/sensor와 ankle force/torque site/sensor를 제거했다. Named sole collision geom은 유지했고 rigid-body dynamics, inertial, joint와 actuator는 바꾸지 않았다. Scene 변경은 include 파일명, 단일 ground 이름 `terrain`, fixed timestep 0.0005 s뿐이다. Asset directory의 `NOTICE.md`가 이 경계를 기록한다.

Walking policy는 Unitree [`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab) revision `1425b15f73bd4095f0df53709d7c389c3eb9e790`의 G1 velocity deploy contract를 사용한다. 해당 upstream source는 [Apache License 2.0](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_mjlab/1425b15f73bd4095f0df53709d7c389c3eb9e790/LICENCE)이며, [deploy config](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_mjlab/1425b15f73bd4095f0df53709d7c389c3eb9e790/deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml)의 98-input/29-output observation-action 계약과 gain/scale을 보존했다.

검증한 local policy SHA-256은 `2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`이다. Legacy local binary에는 독립적인 artifact license/provenance가 완결되어 있지 않아 Git에 넣지 않았고, CLI argument/config/environment로 user-supplied path를 받아 hash와 tensor shape를 fail-closed 검증한다. Gait-mode smoke command speed는 legacy verified range인 0.1–0.5 m/s로 제한한다.

## Sensor와 diagnostic contract

- Physics: 0.5 ms, 2 kHz
- Sensor: 매 2 physics step, 1 ms, 1 kHz
- Runtime: `sequence int64`, `timestamp_us int64`, raw `pelvis_imu [N,6] float32`
- Channel: `[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]`
- Unit: accelerometer m/s², gyroscope rad/s
- Frame: pelvis-local +x forward, +y left, +z up

MJCF test는 `imu` site가 `pelvis` body 원점에 identity rotation으로 붙어 있고 neutral site frame과 body frame이 일치함을 확인한다. 좌우 hip의 ±y 배치, sole의 전후 +x 배치, injected accel/gyro channel order도 deterministic하게 확인했다. Filtering, smoothing, normalization과 handcrafted feature는 없다.

Physical diagnostics는 `SimulationResult.diagnostics`에 분리하고 `RuntimeTrace`에는 넣지 않는다. Slip은 5 N/2.5 N load hysteresis, touchdown 10 ms 제외, anchor drift 0.050 m, 3 ms persistence다. Sink는 같은 validity에서 first-loaded penetration 증가 0.0055 m, 20 ms persistence다. 두 oracle 모두 contact episode와 first-fall boundary를 넘지 않으며 terrain name을 입력으로 사용하지 않는다.

## Smoke 결과

2026-08-26에 같은 verified policy, command 0.15 m/s, 2.0 s headless 조건으로 실행했다. 모든 run은 2,000 expected/actual samples, 1,000 µs timestamp delta, drop 0, finite IMU6, first fall 없음이었다.

| Terrain | Touchdown L/R | Max anchor drift | Max contact penetration | Slip samples | Sink samples | Result |
|---|---:|---:|---:|---:|---:|---|
| concrete | 4 / 4 | 0.006981 m | 0.003579 m | 0 | 0 | PASS |
| ice | 43 / 50 | 0.156679 m | 0.014258 m | 950 | 12 | PASS |
| sand | 4 / 3 | 0.027265 m | 0.014167 m | 0 | 2200 | PASS |

Ice/Sand의 label count는 terrain lookup 결과가 아니라 bilateral actual metric 결과다. 이 smoke에서 event 유무는 acceptance condition이 아니며 profile parameter를 tuning하지 않았다. 한 sample에서 두 foot label이 각각 active할 수 있어 합계가 run sample 수보다 클 수 있다.

## 검증과 한계

- Standard-library test: 4/4 PASS, external ONNX를 제공한 end-to-end test 포함
- Python compile, YAML parse, model load, actuator/IMU/terrain/threshold parity PASS
- Smoke는 메모리에서만 실행했고 dataset/output artifact를 생성하지 않음
- Legacy와 E84 repository worktree 변경 0

Terrain profile은 실제 material measurement나 deformable continuum가 아닌 engineering approximation이다. Contact penetration은 MuJoCo contact response이며 실제 모래 변형 깊이를 의미하지 않는다. 실제 robot/target IMU mounting parity, long-run coverage, run provenance storage와 dataset schema materialization은 아직 검증하지 않았다.

다음 단계는 별도 승인 후 small pilot raw dataset generation이다. 이번 milestone은 dataset/model readiness를 주장하지 않는다.
