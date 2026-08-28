# Infineon FastReflex

## 목적

Unitree G1 simulation 기반 센서로 terrain과 walking stability를 독립적으로 추정하고 control-facing hazard state로 fuse하는 구조를 연구한다.

현재 primary architecture target은 다음 두 stream이다.

- Terrain: `CONCRETE / MARBLE / ICE / SAND / UNKNOWN`
- Stability: `STABLE / UNSTABLE`

Fusion은 `ICE + UNSTABLE → SLIP_RISK`, `SAND + UNSTABLE → SINK_RISK`, 그 외 `UNSTABLE → GENERIC_INSTABILITY`로 해석하고 terrain과 무관하게 recovery를 요청한다. Risk 이름은 terrain-conditioned advisory이며 causal Slip/Sink diagnosis가 아니다. Historical `NORMAL/SLIP/SINK` direct-classification 연구와 physical oracle evidence는 보존한다.

Stability의 첫 runtime input은 Waist/Pelvis IMU 6-axis다. Current rebuilt Terrain research candidate는 touchdown foot의 FSR4를 50 ms 관측하는 shared MLP이며, LEFT_ONLY 배치가 최소-channel candidate다. 최종 sensor architecture는 Stability 결과 전까지 고정하지 않는다.

- accelerometer: x/y/z
- gyroscope: x/y/z

## 기본 원칙

- 1 kHz sensor sequence를 기준으로 연구한다.
- Terrain producer와 Stability producer를 독립 update하고 latest valid terrain을 hold한다.
- Stability는 복잡한 수작업 전처리를 피하고 deterministic pelvis-IMU rule부터 검증한다.
- Exact COM/XCoM/support state는 ground truth와 analysis에만 사용하고 runtime detector에 넣지 않는다.
- AI는 exact stability clock이 acceptance를 통과한 뒤 작은 GRU부터 비교한다.
- dataset, training, validation, Float model export를 이 저장소에서 관리한다.
- 실험 차이는 새 runner가 아니라 config로 표현한다.
- quantization과 E84 deployment는 별도 저장소에서 수행한다.

## Pipeline

```text
MuJoCo runtime streams
  ├─ touchdown-centered Terrain Recognition -> held terrain_state
  └─ continuous Walking Stability Detection -> stability_state
                                              |
                                              v
              deterministic State Fusion -> hazard_state + recovery_required
```

## Repository boundary

이 저장소가 담당하는 범위:

- Unitree G1 MuJoCo simulation을 이용한 데이터 설계 및 수집
- terrain/stability dataset contract와 historical NORMAL/SLIP/SINK evidence 관리
- 모델 설계, 학습, run-disjoint / group-disjoint validation
- 검증된 Float model과 계약 artifact export

이 저장소가 담당하지 않는 범위:

- quantization, target conversion, Vela
- KIT_PSE84_AI / PSoC Edge E84 firmware integration
- HIL 및 target runtime validation

위 deployment 작업은 [`Infineon_FastReflex_E84`](https://github.com/shinjuyeop/Infineon_FastReflex_E84)에서 수행한다. 기존 `/d/shin/Infineon`의 코드와 asset은 자동으로 복사하지 않으며, 향후 명시적으로 검토된 migration만 허용한다.

## Current Status

`TERRAIN_RECOGNITION_SUPPORTED`

Calibrated Concrete/Marble→Ice/Sand에서 fresh `terrain_transition_20260828` dataset을 생성했다. 144 run, 1,152,000 raw samples와 3,139 clean 50 ms touchdown events가 four-class/side/source/stable-fall acceptance를 통과했고 drop, duplicate condition, split overlap과 pre-transition fall은 0이다. Exact terrain geom identity는 label-only이며 model tensor에는 해당 foot의 raw sensor만 들어간다.

50 ms validation sensor ablation의 mean macro F1/worst recall은 FSR4 0.9284/0.8571, Foot IMU6 0.9129/0.8095, Fusion10 0.9309/0.8333이었다. Qualification을 함께 통과한 최소 profile인 FSR4와 MLP/50 ms/LEFT_ONLY를 holdout 전에 선택했다. One-shot holdout은 macro F1 0.9713, worst recall 0.95, run-balanced macro F1 0.9563으로 통과했다. Recommendation은 `LEFT_FSR4_RECOMMENDED`이고 Pelvis IMU6 포함 10 physical channels다.

LEFT_ONLY는 126/144 transition에서 update가 가능했고 median/p95 delay는 1114.5/1238 ms였지만 right-only Sand 18 runs에는 clean left target touchdown이 없었다. BILATERAL_SHARED는 144/144, median/p95 922/1238 ms이고 Pelvis IMU 포함 14 channels다. 따라서 Terrain research candidate는 supported지만 `FINAL_SENSOR_ARCHITECTURE_FROZEN`은 아니다. Historical phase-aware exact MoS clock과 Stability detector는 계속 미지원 상태이며, legacy Terrain v4와 direct Slip/Sink 연구는 historical comparison으로만 보존한다.

Dataset, ablation, holdout과 hardware-latency audit의 전체 근거는 [`20260828_terrain_rebuild_sensor_ablation.md`](reports/20260828_terrain_rebuild_sensor_ablation.md)에 기록한다.

## 구조

```text
configs/              실험 차이를 표현할 configuration
src/fastreflex/       canonical Python package
scripts/fastreflex.py 단일 CLI entry point
docs/                 architecture와 연구 protocol
data/                 local dataset 경계
artifacts/            model 및 run artifact 경계
reports/              생성된 분석 보고서 경계
tests/                simulator와 이후 pipeline의 contract test
```

설계 개요는 [`docs/architecture.md`](docs/architecture.md), dataset 원칙은 [`docs/dataset.md`](docs/dataset.md), 검증 원칙은 [`docs/experiment_protocol.md`](docs/experiment_protocol.md)를 참고한다.

## CLI

Canonical `simulate` command는 display 없는 headless 실행과 visual-only MuJoCo viewer를 모두 지원한다. Policy path는 CLI 또는 `FASTREFLEX_G1_POLICY` 환경 변수로 제공한다. 다음 명령은 trace를 파일로 저장하지 않는 2초 headless smoke다.

```bash
python scripts/fastreflex.py simulate \
  --terrain concrete \
  --speed 0.15 \
  --duration 2.0 \
  --headless \
  --policy /path/to/policy.onnx
```

첫 raw Pilot Dataset은 기존 final directory를 덮어쓰지 않는 fail-closed `collect` command로 생성한다.

```bash
python scripts/fastreflex.py collect \
  --config configs/experiment/20260827_hazard_pilot_dataset.yaml
```

첫 bounded classification PoC는 fixed experiment config를 사용하는 canonical `train` command로 재현한다. 기존 local artifact가 있으면 덮어쓰지 않고 fail-closed한다.

```bash
python scripts/fastreflex.py train \
  --config configs/experiment/20260827_first_classification_poc.yaml
```

Frozen first-PoC classifier의 Time-to-Separation은 재학습 없이 canonical `evaluate` command로 replay한다. Config에 기록된 split, normalizer와 checkpoint SHA가 다르면 fail-closed한다.

```bash
python scripts/fastreflex.py evaluate \
  --config configs/experiment/20260827_time_to_separation.yaml
```

Terrain/Stability integrated sanity는 같은 canonical `evaluate` command에서 frozen 44-run matrix, exact-state gate, IMU rule, fusion과 terminal status replay를 실행한다. Existing artifact는 덮어쓰지 않는다.

```bash
python scripts/fastreflex.py evaluate \
  --config configs/experiment/20260827_terrain_stability_integrated_sanity.yaml
```

Transition scenario calibration은 prefix parity가 실패하면 calibration 전에 중단하고, PASS하면 frozen calibration, fresh Concrete validation과 Marble robustness를 순서대로 실행한다.

```bash
python scripts/fastreflex.py evaluate \
  --config configs/experiment/20260828_transition_scenario_calibration.yaml
```

Rebuilt Terrain dataset과 sensor ablation은 같은 canonical `collect`/`train` command를 사용한다. Raw NPZ, event index, checkpoints와 metrics는 Gitignored 경계에 생성된다.

```bash
python scripts/fastreflex.py collect \
  --config configs/experiment/20260828_terrain_rebuild_sensor_ablation.yaml \
  --policy /path/to/policy.onnx

python scripts/fastreflex.py train \
  --config configs/experiment/20260828_terrain_rebuild_sensor_ablation.yaml
```

Integrated calibration이 존재하면 canonical `simulate` 결과의 timestamp-synchronized status replay도 사용할 수 있다.

```bash
python scripts/fastreflex.py simulate \
  --terrain ice --slip-pattern transition --speed 0.15 --duration 8 \
  --policy /path/to/policy.onnx \
  --status-calibration artifacts/runs/20260827_terrain_stability_integrated_sanity/calibration.json
```

FSR observability ablation은 동일한 canonical `collect`와 `train` command에 sensor Pilot config를 제공한다. Existing dataset/artifact는 덮어쓰지 않는다.

```bash
python scripts/fastreflex.py collect \
  --config configs/experiment/20260827_fsr_observability_pilot.yaml

python scripts/fastreflex.py train \
  --config configs/experiment/20260827_fsr_observability_pilot.yaml
```

같은 sensor Pilot의 FSR load distribution은 simulation이나 재학습 없이 canonical `evaluate` command로 재현한다. Generated CSV/PNG는 Gitignored artifact에만 저장한다.

```bash
python scripts/fastreflex.py evaluate \
  --config configs/experiment/20260827_fsr_load_distribution_analysis.yaml

python scripts/fastreflex.py evaluate \
  --config configs/experiment/20260827_fsr_temporal_redistribution_analysis.yaml
```

`export`는 같은 entry point의 placeholder로 유지한다.

Viewer, 네 terrain 예제, policy 준비와 summary 해석은 [`docs/simulation.md`](docs/simulation.md)를 참고한다.

## Dependency

MuJoCo/runtime dependency에 더해 첫 raw-IMU PoC는 `torch`와 sanity plot용 `matplotlib`을 사용한다. Confusion matrix와 precision/recall/F1은 source에 작게 구현하며 `scikit-learn`이나 별도 ML framework는 추가하지 않는다.
