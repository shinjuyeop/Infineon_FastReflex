# Infineon FastReflex

Unitree G1 MuJoCo 환경에서 낙상 위험을 조기에 감지하는 Hazard Reflex와,
위험 원인을 보조적으로 설명하는 Terrain advisory를 연구하는 repository다.

## Current Status

현재 결론은 **기존 제한 조건에서는 동작하지만, 새로운 물리 조건으로의 일반화는
아직 입증되지 않았다**는 것이다.

| 항목 | 현재 판정 | 의미 |
|---|---|---|
| Unified Hazard baseline | `SUPPORTED` | 기존 frozen HOLDOUT에서 Hazard 26/26, no-hazard 26/26, premature 0 |
| Terrain advisory | `SUPPORTED` | FSR4/MLP가 지면 종류를 보조 판정하며 Hazard 발생 자체를 gate하지 않음 |
| Model V2 generalization | `NOT_SUPPORTED` | one-shot Generalization HOLDOUT의 primary gate 실패 |
| Factor-conditioned intervention | `NOT_EFFECTIVE` | Sand 오경보는 줄었지만 Support 검출이 크게 저하됨 |
| 최신 boundary-resolution cycle | `INVALID` | 새 validation corpus가 물리 gate를 통과하지 못해 모델을 학습하지 않음 |
| Deployment | engineering reference only | E84 연계용 Float reference이며 release model이나 real-robot 검증 결과가 아님 |
| Deployment-aware QAT | `TRAIN_ACCEPTANCE_FAIL` | 3-seed QAT은 완료했지만 TRAIN-only robustness gate 실패로 candidate/handoff 없음 |

가장 최근 cycle에서는 독립적인 model-blind 120-run validation corpus를 한 번
생성했다. 그 결과 objective-eligible 102/120, strict Sand 43/60, invalid 14,
Slip/Dual contamination 4로 19개 물리 gate 중 8개를 실패했다. Protocol에 따라
candidate 학습, optimizer step, HNM, validation inference는 모두 0으로 유지했다.

따라서 다음 단계는 모델이나 threshold 변경이 아니라
`HAZARD_BOUNDARY_VALIDATION_PHYSICAL_REDESIGN`이다. 실패한 corpus를 수정하거나
재사용하지 않고, 새로운 물리 validation matrix를 model-blind하게 설계해야 한다.

이 scientific work와 별개로 frozen Float engineering reference를 대상으로 한
[deployment-aware QAT protocol](configs/experiment/20260904_deployment_aware_qat.yaml)을
한 번 실행했다. 세 seed 모두 학습했지만 TRAIN-only member max와 ensemble max,
material-improvement 및 threshold/persistence parity gate를 통과하지 못했다. 따라서
candidate freeze, development/golden evaluation, handoff export는 수행하지 않았다.
기존 generalization verdict와 scientific candidate는 변경되지 않았다.

Scientific boundary의 상세 근거는
[latest boundary-resolution report](reports/20260904_hazard_boundary_resolution.md),
deployment engineering 결과는
[deployment-aware QAT report](reports/20260904_deployment_aware_qat.md)에 있다.

## Reviewer Guide

처음 검토할 때는 아래 순서만 보면 현재 판단 근거를 빠르게 파악할 수 있다.

1. 이 README: 현재 결론, 범위, 다음 단계
2. [Latest result](reports/20260904_hazard_boundary_resolution.md): 최근 실패 원인과
   physical-gate stop
3. [Architecture](docs/architecture.md): runtime sensor, model, advisory 관계
4. [Dataset contract](docs/dataset.md): label, split, provenance, HOLDOUT 경계
5. [Experiment protocol](docs/experiment_protocol.md): 재학습·평가·중단 조건

검토가 필요한 핵심 질문은 다음 네 가지다.

- 기존 baseline 성능과 새 simulation scenario 일반화 실패를 구분한 해석이
  타당한가?
- Sand와 Support 경계용 physical gate 및 invalid 기준이 적절한가?
- 현재 `TRAINING_OBJECTIVE_SAMPLING_TENSION` 진단이 다음 모델 실험을 정당화하는가?
- 새 validation matrix를 만들기 전에 추가로 확인해야 할 물리 변수가 있는가?

전체 실험 이력은 [reports index](reports/README.md)에서 단계별로 찾을 수 있다.

## System Overview

```text
Hazard path
Pelvis IMU6 -> causal 80D -> GRU (20 ms)
            -> probability >= 0.99 for 5 ms
            -> NORMAL / HAZARD_REFLEX_REQUIRED

Terrain advisory
Touchdown-foot FSR4 -> clean 50 ms -> MLP
                    -> CONCRETE / MARBLE / ICE / SAND

Decision interpretation
REFLEX_REQUIRED + ICE          -> SLIP_RISK
REFLEX_REQUIRED + SAND         -> SUPPORT_RISK
REFLEX_REQUIRED + other/unknown -> GENERIC_DISTURBANCE
```

Terrain 결과는 Reflex를 발생시키거나 지연하지 않는다. Hazard 모델은 Pelvis IMU만
사용하며 Terrain, fall/recovery state, simulator-only physical oracle은 입력에
포함하지 않는다.

Unified physical label은 다음과 같다.

```text
ESTABLISHED_SLIP OR ESTABLISHED_SUPPORT
-> HAZARD_REFLEX_REQUIRED
```

Primary no-hazard는 established Slip, I1 precursor, established Support가 모두 없는
run이다. Slip/Support clock과 I1은 label 및 scoring reference이고 runtime feature가
아니다.

## Repository Map

| 위치 | 책임 |
|---|---|
| `scripts/fastreflex.py` | 단일 fail-closed CLI |
| `src/fastreflex/simulation/` | Unitree G1 physics, sensors, physical diagnostics |
| `src/fastreflex/dataset/` | dataset generation, labels, manifest, split integrity |
| `src/fastreflex/features.py` | Pelvis IMU6에서 causal 80D feature 생성 |
| `src/fastreflex/training/hazard.py` | Hazard TRAIN-only windowing, fitting, HNM 및 audit |
| `src/fastreflex/training/qat.py` | frozen Float reference의 deployment-only QAT derivative |
| `src/fastreflex/evaluation/` | frozen candidate replay, metrics, readiness, HOLDOUT guard |
| `src/fastreflex/models/` | shared MLP/GRU model definitions |
| `configs/dataset/` | canonical dataset contracts |
| `configs/model/` | model family 및 frozen candidate metadata |
| `configs/experiment/` | 날짜별 재현 가능한 실험 계약 |
| `reports/` | 실험별 결과와 provenance |
| `artifacts/releases/` | 명시적으로 검토된 frozen engineering handoff |

82개의 experiment config 중 일부만 current CLI에서 실행 가능하다. 나머지는
실패·중간 판단을 포함한 historical provenance이며 현재 source에서 임의로
fallback 실행하지 않는다. Git history와 각 report/config의 `source_commit`이
해당 시점의 구현 archive다.

## Canonical Commands

환경 설치 후 모든 명령은 repository root에서 실행한다.

G1 headless smoke simulation:

```bash
python scripts/fastreflex.py simulate \
  --terrain concrete --speed 0.15 --duration 2.0 --headless \
  --policy /path/to/g1_velocity_policy.onnx
```

Frozen Unified Hazard candidate 검증:

```bash
python scripts/fastreflex.py evaluate \
  --config configs/experiment/20260829_unified_hazard_reflex_system.yaml
```

Frozen Terrain candidate 검증:

```bash
python scripts/fastreflex.py evaluate \
  --config configs/experiment/20260828_terrain_rebuild_sensor_ablation.yaml
```

대표 TRAIN/VALIDATION run 시각화:

```bash
python scripts/fastreflex.py visualize --run-id uhr_ice_h_c20
```

HOLDOUT run은 visualization 대상이 아니다. Viewer 사용법과 overlay 해석은
[simulation guide](docs/simulation.md#visualizing-supported-hazardterrain-decisions)를
참고한다.

Reviewed Float engineering-reference bundle 검증·export:

```bash
python scripts/fastreflex.py export \
  --output /path/to/new/engineering_reference
```

Output directory는 존재하지 않아야 한다. Export는 기존 bundle을 덮어쓰지 않으며
frozen source checksum이 다르면 fail-closed한다.

## Scientific and Repository Boundaries

이 repository가 담당하는 범위:

- Unitree G1 MuJoCo simulation과 runtime/diagnostic separation
- Hazard/Terrain dataset contract와 provenance
- Float research model의 training/evaluation
- run-disjoint split, TRAIN-only preprocessing/HNM, sealed HOLDOUT protocol
- 검토된 engineering-reference artifact
- predeclared TRAIN-only deployment-aware QAT engineering derivative

이 repository가 담당하지 않는 범위:

- real-robot 성능 주장
- TFLite full-INT8 conversion, Vela, target conversion
- KIT_PSE84_AI / PSoC Edge E84 integration
- firmware, HIL, Recovery controller

Deployment 구현은 별도 `Infineon_FastReflex_E84` repository에서 수행한다.
기존 `/d/shin/Infineon` repository의 코드·데이터·artifact는 자동으로 복사하지
않는다.

Generated dataset과 일반 experiment output은 Git에 commit하지 않는다. 박사님이나
외부 검토자에게는 전체 local workspace를 압축하기보다 Git repository와 검토할
commit을 공유해야 한다.

## Verification

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall src scripts
python -m ruff check src scripts tests
```

Canonical current-state 문서는
[architecture](docs/architecture.md),
[dataset](docs/dataset.md),
[experiment protocol](docs/experiment_protocol.md)이다.
