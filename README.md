# Infineon FastReflex

Unitree G1 MuJoCo에서 검증된 physical Hazard Reflex와 Terrain advisory를 관리하는 research repository다.

## Current Status

`SUPPORTED`

```text
Hazard
Pelvis IMU6 -> causal 80D -> GRU(20 ms) -> threshold 0.99
            -> persistence 5 ms -> NORMAL / HAZARD_REFLEX_REQUIRED

Terrain
touchdown-foot FSR4 -> clean 50 ms -> MLP -> held Terrain state
                    -> advisory cause refinement only
```

Hazard verdict는 `UNIFIED_HAZARD_REFLEX_SUPPORTED_SINGLE_IMU`다. Final candidate는 11,010 parameters이며 fresh one-shot HOLDOUT에서 Hazard 26/26, Slip 13/13, Support 13/13, no-hazard 26/26, Sand benign specificity 100%, hard-ground specificity 100%, premature 0을 기록했다. Candidate freeze SHA-256은 `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`다.

Terrain verdict는 `TERRAIN_RECOGNITION_SUPPORTED`다. Current candidate는 FSR4/MLP/50 ms이고 `CONCRETE / MARBLE / ICE / SAND`를 출력한다. Terrain은 `REFLEX_REQUIRED`를 gate하거나 지연하지 않는다. Reflex 뒤의 cause만 다음처럼 보완한다.

```text
REFLEX_REQUIRED + ICE  -> SLIP_RISK
REFLEX_REQUIRED + SAND -> SUPPORT_RISK
REFLEX_REQUIRED + other/unknown -> GENERIC_DISTURBANCE
```

Final sensor architecture는 E84 resource와 hardware-realism 검증 전까지 freeze하지 않는다.

## Canonical source flow

```text
Hazard:
simulation/g1.py -> dataset/hazard.py -> features.py
                 -> training/hazard.py -> models/baselines.py
                 -> evaluation/hazard.py

Terrain:
simulation/{g1,sensors,terrain}.py -> dataset/terrain.py
                                  -> training/terrain.py
                                  -> evaluation/terrain.py -> advisory cause
```

주요 책임은 다음과 같다.

- `simulation/`: G1 physics, runtime sensors, physical Slip/Support/I1 관련 diagnostics
- `dataset/hazard.py`: unified run, physical labels, split, manifest integrity, HOLDOUT guard
- `features.py`: selected Pelvis IMU6 → causal 80D contract
- `training/hazard.py`: TRAIN-only normalization/windowing와 3-round HNM
- `evaluation/hazard.py`: continuous replay, 0.99/5 ms decision, metrics, frozen-candidate verification
- `dataset/terrain.py`: exact clean-touchdown indexing과 Terrain dataset contract
- `training/terrain.py`: Terrain training responsibility
- `evaluation/terrain.py`: FSR4/50 ms inference, held state, advisory cause refinement
- `models/baselines.py`: shared MLP/GRU definitions
- `scripts/fastreflex.py`: single fail-closed CLI

## Scientific boundaries

Unified physical label은 다음과 같다.

```text
ESTABLISHED_SLIP OR ESTABLISHED_SUPPORT
-> HAZARD_REFLEX_REQUIRED
```

Primary no-hazard는 established Slip, I1 precursor, established Support가 모두 없는 run이다. Physical Slip/Support clock과 I1은 label/scoring reference이며 runtime feature가 아니다. Fall/recovery와 Terrain도 Hazard tensor 또는 Hazard label에 들어가지 않는다.

Current preprocessing은 정확히 10 base × 8 causal transforms = 80D다. Feature order, float32 dtype, causal prefix behavior와 schema hash `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`를 test로 고정한다.

## Repository boundary

이 repository가 담당한다.

- Unitree G1 MuJoCo simulation과 runtime/diagnostic separation
- Hazard/Terrain dataset contract와 provenance
- Float model training/evaluation code와 protected research artifacts
- run-disjoint split, TRAIN-only preprocessing/HNM, sealed HOLDOUT protocol

이 repository가 담당하지 않는다.

- quantization, Vela, target conversion
- KIT_PSE84_AI / PSoC Edge E84 integration
- firmware, HIL, Recovery controller

Deployment 작업은 별도 `Infineon_FastReflex_E84` repository에서 명시적으로 시작한다. 기존 `/d/shin/Infineon`의 코드·데이터·artifact는 자동으로 복사하지 않는다.

## CLI

Canonical top-level commands는 `simulate`, `collect`, `train`, `evaluate`, `export`다.

G1 smoke simulation:

```bash
python scripts/fastreflex.py simulate \
  --terrain concrete --speed 0.15 --duration 2.0 --headless \
  --policy /path/to/g1_velocity_policy.onnx
```

Frozen Unified Hazard candidate의 read-only verification:

```bash
python scripts/fastreflex.py evaluate \
  --config configs/experiment/20260829_unified_hazard_reflex_system.yaml
```

Frozen Terrain candidate의 read-only verification:

```bash
python scripts/fastreflex.py evaluate \
  --config configs/experiment/20260828_terrain_rebuild_sensor_ablation.yaml
```

`collect`, `train`, `evaluate`는 explicit `--config`를 요구한다. Historical experiment config는 current source에서 다른 runner로 fallback하지 않고, report/config에 기록된 source commit을 사용하라는 메시지와 함께 fail-closed한다. Current frozen Hazard corpus는 이 consolidated CLI에서 재생성하지 않으며, current candidate를 같은 artifact identity에 암묵적으로 재학습하지 않는다. `export`는 reviewed Research-to-Deployment release가 생길 때까지 reserved다.

## Historical evidence

과거 MoS/Stability, direct classification, event-centric, Terrain-gated fusion, continuous Slip와 Support 진단 과정의 결론은 삭제하지 않았다. [`reports/`](reports/)와 Git history가 authoritative evidence/archive다. Current lineage의 핵심 보고서는 다음 두 개다.

- [`reports/20260829_unified_hazard_reflex_system.md`](reports/20260829_unified_hazard_reflex_system.md)
- [`reports/20260828_terrain_rebuild_sensor_ablation.md`](reports/20260828_terrain_rebuild_sensor_ablation.md)

Current architecture는 [`docs/architecture.md`](docs/architecture.md), dataset contract는 [`docs/dataset.md`](docs/dataset.md), 검증 규칙은 [`docs/experiment_protocol.md`](docs/experiment_protocol.md)에 있다.

## Verification

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall src scripts
```

기본 `pytest` command는 project environment의 pytest/plugin version이 일치할 때 사용한다.
