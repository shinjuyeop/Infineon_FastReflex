# Infineon FastReflex

## 목적

Unitree G1 simulation 기반 센서 데이터를 이용하여 로봇 보행 중 위험 상태를 빠르게 분류하는 모델을 연구한다.

현재 신규 Hazard model의 primary task는 다음 3-class classification이다.

- `NORMAL`
- `SLIP`
- `SINK`

입력 후보는 Waist/Pelvis IMU 6-axis 신호다.

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

`PROJECT_SCAFFOLD_ONLY`

현재는 repository 골격과 문서, canonical CLI placeholder만 존재한다. dataset, simulator integration, model, training pipeline은 아직 구현되지 않았다.

## 구조

```text
configs/              실험 차이를 표현할 configuration
src/fastreflex/       canonical Python package
scripts/fastreflex.py 단일 CLI entry point
docs/                 architecture와 연구 protocol
data/                 local dataset 경계
artifacts/            model 및 run artifact 경계
reports/              생성된 분석 보고서 경계
tests/                향후 test suite
```

설계 개요는 [`docs/architecture.md`](docs/architecture.md), dataset 원칙은 [`docs/dataset.md`](docs/dataset.md), 검증 원칙은 [`docs/experiment_protocol.md`](docs/experiment_protocol.md)를 참고한다.

## CLI placeholder

Python 3.10 이상에서 다음 도움말을 확인할 수 있다.

```bash
python scripts/fastreflex.py --help
```

향후 `collect`, `train`, `evaluate`, `export`가 같은 entry point에 구현된다. 현재 각 command는 미구현 상태를 알리고 정상 종료한다.

## 향후 dependency 후보

실제 구현이 시작될 때 필요한 것만 추가한다. 현재 후보는 `numpy`, `pandas`, `matplotlib`, `torch`, `scikit-learn`, `pyyaml`이다.
