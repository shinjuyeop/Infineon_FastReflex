# Infineon FastReflex

## 목적

Unitree G1 simulation 기반 센서 데이터를 이용하여 로봇 보행 중 위험 상태를 빠르게 분류하는 모델을 연구한다.

현재 신규 Hazard model의 primary task는 다음 3-class classification이다.

- `NORMAL`
- `SLIP`
- `SINK`

Runtime 입력은 Waist/Pelvis IMU 6-axis 신호다.

- accelerometer: x/y/z
- gyroscope: x/y/z

## 기본 원칙

- 1 kHz sensor sequence를 기준으로 연구한다.
- 복잡한 수작업 전처리를 최소화하고 Raw IMU와 최소 normalization에서 시작한다.
- PyTorch 기반으로 MLP baseline, 1D CNN, GRU, LSTM을 비교한다.
- dataset, training, validation, Float model export를 이 저장소에서 관리한다.
- 실험 차이는 새 runner가 아니라 config로 표현한다.
- quantization과 E84 deployment는 별도 저장소에서 수행한다.

## Pipeline

```text
MuJoCo
  -> Raw IMU Dataset
  -> Windowing
  -> PyTorch Model
  -> Training
  -> Validation
  -> Frozen Float Model
```

## Repository boundary

이 저장소가 담당하는 범위:

- Unitree G1 MuJoCo simulation을 이용한 데이터 설계 및 수집
- NORMAL / SLIP / SINK dataset 관리
- 모델 설계, 학습, run-disjoint / group-disjoint validation
- 검증된 Float model과 계약 artifact export

이 저장소가 담당하지 않는 범위:

- quantization, target conversion, Vela
- KIT_PSE84_AI / PSoC Edge E84 firmware integration
- HIL 및 target runtime validation

위 deployment 작업은 [`Infineon_FastReflex_E84`](https://github.com/shinjuyeop/Infineon_FastReflex_E84)에서 수행한다. 기존 `/d/shin/Infineon`의 코드와 asset은 자동으로 복사하지 않으며, 향후 명시적으로 검토된 migration만 허용한다.

## Current Status

`FIRST_CLASSIFICATION_POC_COMPLETE`

40-run `hazard_pilot_20260827` 중 33 valid run을 metadata-only 20/6/7 run-disjoint split로 고정하고, 50/100 ms raw pelvis IMU6 established-state MLP/GRU PoC를 완료했다. Validation으로 선택한 MLP 100 ms는 한 번 연 Pilot internal holdout에서 macro F1 0.8614를 보였다. 이는 Pilot-only established `NORMAL/SLIP/SINK` separability evidence이며 early detection, Time-to-Separation, Full Dataset, final model 또는 deployment readiness를 뜻하지 않는다. Walking policy ONNX, raw data와 generated training artifacts는 Git에 포함하지 않는다.

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

`evaluate`, `export`는 같은 entry point의 placeholder로 유지한다.

Viewer, 네 terrain 예제, policy 준비와 summary 해석은 [`docs/simulation.md`](docs/simulation.md)를 참고한다.

## Dependency

MuJoCo/runtime dependency에 더해 첫 raw-IMU PoC는 `torch`와 sanity plot용 `matplotlib`을 사용한다. Confusion matrix와 precision/recall/F1은 source에 작게 구현하며 `scikit-learn`이나 별도 ML framework는 추가하지 않는다.
