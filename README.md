# Infineon FastReflex

## 목적

Unitree G1 simulation 기반 센서로 terrain과 walking stability를 독립적으로 추정하고 control-facing hazard state로 fuse하는 구조를 연구한다.

현재 primary architecture target은 다음 두 stream이다.

- Terrain: `CONCRETE / MARBLE / ICE / SAND / UNKNOWN`
- Stability: `STABLE / UNSTABLE`

Fusion은 `ICE + UNSTABLE → SLIP_RISK`, `SAND + UNSTABLE → SINK_RISK`, 그 외 `UNSTABLE → GENERIC_INSTABILITY`로 해석하고 terrain과 무관하게 recovery를 요청한다. Risk 이름은 terrain-conditioned advisory이며 causal Slip/Sink diagnosis가 아니다. Historical `NORMAL/SLIP/SINK` direct-classification 연구와 physical oracle evidence는 보존한다.

Stability의 첫 runtime input은 Waist/Pelvis IMU 6-axis다. Legacy frozen Terrain reference는 foot FSR4 + foot/ankle IMU6를 요구하지만 clean migration은 pending이다. 최종 sensor architecture는 고정하지 않았다.

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

`INTEGRATED_SCENARIO_NEEDS_REVISION`

44-run `TERRAIN_STABILITY_INTEGRATED_SANITY`에서 hard controls는 6/6 stable, fall-intended Ice/Sand는 10/10과 12/12 fall이었지만 stable-intended는 Ice 1/8, Sand 4/8만 non-fall이었고 12개 run이 target transition 전에 fall했다. Phase-aware exact MoS clock도 stable FP 5/11, fall coverage 22/33으로 acceptance를 통과하지 못했다. Fail-closed로 Stability GRU는 실행하지 않았다. Fusion truth table과 independent state plumbing은 test를 통과했지만 integration readiness, terrain AI integration과 sensor architecture freeze를 뜻하지 않는다.

Legacy frozen Terrain v4는 artifact/checksum/input contract를 audit했으나 current repository에 left foot/ankle IMU6와 TFLite runtime parity가 없어 `TERRAIN_RUNTIME_MODEL_PENDING`이다. Integrated run의 terrain state는 명시적인 `ORACLE_PROXY`이며 AI latency로 주장하지 않는다. 이전 `SINK_SENSOR_OBSERVABILITY_PROMISING`과 모든 Slip/Sink historical report는 그대로 보존한다.

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
