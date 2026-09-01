# Infineon FastReflex

Unitree G1 MuJoCo에서 검증된 physical Hazard Reflex와 Terrain advisory를 관리하는 research repository다.

## Current Status

`UNIFIED_BASELINE_SUPPORTED; ZERO_RETRAIN_GENERALIZATION_NOT_SUPPORTED; ICE_NEAR_HAZARD_TARGET_SEMANTICS_RESOLVED; MODEL_V2_DATASET_GENERATION_READY; MODEL_V2_DATA_ONLY_TRAINING_COMPLETE; MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED; MODEL_V2_INTERNAL_FAILURE_AUDIT_ACTIONABLE; MODEL_V2_EXTRACTION_REBALANCE_DESIGN_READY`

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

Current 256-run corpus의 scenario coverage audit verdict는 `SCENARIO_COVERAGE_DESIGN_READY`다. 기존 data/model/HOLDOUT을 변경하거나 재평가하지 않고 Ice benign, Terrain-first delayed Hazard, affected-side, speed와 gait-phase gap을 정량화했으며, 다음 generation을 위한 minimum informative scenario set을 [`reports/20260831_scenario_coverage_matrix_design.md`](reports/20260831_scenario_coverage_matrix_design.md)에 사전 설계했다.

후속 physical calibration verdict는 `GENERALIZATION_SCENARIO_CALIBRATION_BLOCKED`였다. 78개 model-blind pilot signatures에서 `DELAYED_SAND_SUPPORT_ONSET`, `RIGHT_SAND_SUPPORT`, `SPEED_STRATIFIED_HAZARD`는 READY였지만, `ICE_BENIGN_CONTROL`과 `DELAYED_ICE_SLIP` P0를 포함한 네 family는 BLOCKED였다. Pilot은 모두 future evaluation에서 제외하며 current HOLDOUT/training/model artifact는 건드리지 않았다. 상세 결과와 partial next-generation freeze는 [`reports/20260831_generalization_scenario_calibration.md`](reports/20260831_generalization_scenario_calibration.md)에 있다.

P0 Ice gap resolution verdict는 `ICE_GENERALIZATION_GAP_RESOLVED`다. 기존 `DELAYED_ICE_SLIP >=1000 ms`는 완화하지 않고 BLOCKED로 보존했으며, fresh episode-based `ONE_CONTACT_DELAYED_ICE_SLIP` 18/24와 fresh `ICE_BENIGN_CONTROL` 4/24를 model-blind하게 확보했다. 두 Ice family와 기존 READY 3개로 final five-family set을 freeze했으므로 scenario-calibration readiness는 `FULL_GENERALIZATION_DATASET_READY`다. 48개 fresh pilots도 future evaluation에서 제외하며 current HOLDOUT은 재오픈하지 않았다. 상세 결과는 [`reports/20260831_ice_generalization_gap_resolution.md`](reports/20260831_ice_generalization_gap_resolution.md)에 있다.

이 frozen five-family 설계로 fresh 72-run `generalization_hazard_reflex_20260831` corpus를 생성했다. Physical verdict는 `GENERALIZATION_DATASET_READY`지만, zero-retrain VALIDATION은 Hazard 13/26, Slip 7/12, Support 6/14, primary no-hazard 5/10, Ice-benign specificity 3/4, premature 7/26으로 predeclared gate를 실패했다. 따라서 model verdict는 `ZERO_RETRAIN_GENERALIZATION_NOT_SUPPORTED`이며 generalization HOLDOUT은 open count 0으로 sealed 상태다. Current Unified HOLDOUT도 재오픈하거나 새 inference하지 않았다. 상세 결과는 [`reports/20260831_generalization_dataset_zero_retrain.md`](reports/20260831_generalization_dataset_zero_retrain.md)에 있다.

후속 diagnostic-only audit verdict는 `GENERALIZATION_FAILURE_MODE_AUDIT_ACTIONABLE`이다. Delayed Ice와 Ice-benign false alert는 실제 42–45 mm near-slip episode와 current target boundary의 tension으로, delayed/Sand-benign alert는 deformable physics 이전의 benign transition transient로, right-only Support 0/4는 comparable Pelvis-IMU magnitude를 가진 side distribution/model failure로 국소화했다. Hazard TRAIN은 0.25 m/s와 left-only Support에 편중되어 있어 data/side/speed/hard-negative coverage correction이 정당화되지만 LSTM, longer history, threshold/persistence 변경은 정당화되지 않았다. Generalization HOLDOUT open count는 계속 0이다. 상세 결과는 [`reports/20260901_generalization_failure_mode_audit.md`](reports/20260901_generalization_failure_mode_audit.md)에 있다.

Ice near-hazard target-semantics verdict는 `ICE_NEAR_HAZARD_TARGET_SEMANTICS_RESOLVED`, recommendation은 `ICE_PHYSICAL_PRECURSOR_SUPPORTED`다. Frozen 50 mm/3 ms established-Slip oracle와 Model V1은 그대로 유지하고, loaded exact-Ice 30–50 mm 상태를 future Model V2용 별도 development precursor/acceptable-early region으로 권고한다. Fresh 48-run corpus의 discovery/one-shot confirmation이 progression을 재현했으며 velocity threshold는 overlap 때문에 추가하지 않았다. `ZERO_RETRAIN_GENERALIZATION_NOT_SUPPORTED`는 변경하지 않았고 Generalization HOLDOUT 36은 open count 0으로 sealed다. 상세 결과는 [`reports/20260901_ice_near_hazard_target_semantics.md`](reports/20260901_ice_near_hazard_target_semantics.md)에 있다.

첫 Model V2 dataset design verdict는 `MODEL_V2_DATASET_DESIGN_READY`다. Model V1을 restorable 상태로 보존하고 `RETAIN_AND_AUGMENT` 기준 fresh 412-run matrix(`V2_TRAIN` 310, `V2_VALIDATION` 102)를 simulation 전에 freeze했다. Primary Hazard와 Ice precursor annotation을 분리하고 delayed/multi-contact Ice, Ice benign/near-hazard, staged Sand benign, balanced right Support와 0.20/0.25/0.30 m/s coverage를 포함한다. 이 design milestone 자체에서는 raw V2 dataset이나 checkpoint를 만들지 않았으며 상세 설계는 [`reports/20260901_model_v2_dataset_design.md`](reports/20260901_model_v2_dataset_design.md)에 있다.

Frozen design을 그대로 실행해 `model_v2_hazard_reflex_20260901` augmentation corpus를 생성·freeze했다. 412개 primary run은 한 번씩 실행되었고 실제 결과는 valid 386, objectively invalid 26이다. `V2_TRAIN`의 actual established-Hazard/no-established-Hazard는 182/108이며 후자에는 I1-only 또는 censored-precursor 18개가 포함되어 confirmed no-hazard는 90이다. Slip/Support는 103/81, Hazard speed 0.20/0.25/0.30 m/s = 53/76/53이며 right-only Support 32와 staged-Sand usable hard negative 26을 포함한다. Dataset verdict는 `MODEL_V2_DATASET_GENERATION_READY`다. 상세 결과는 [`reports/20260901_model_v2_dataset_generation.md`](reports/20260901_model_v2_dataset_generation.md)에 있다.

첫 data-only V2는 Unified TRAIN + valid V2_TRAIN 442개만 사용해 새 normalizer와 같은 11,010-parameter GRU20 3-seed ensemble을 학습했다. V2_VALIDATION은 candidate freeze 뒤 한 번 평가했으며 Hazard 55/64, Slip 29/35, Support 27/30, confirmed no-hazard 26/26, premature 6/64였다. Staged/Speed Sand benign과 right-only Support는 각각 8/8, 12/12, 12/12로 개선됐지만 overall/Slip gate를 실패했으므로 verdict는 `MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED`다. Candidate는 external generalization candidate로 승격하지 않았고 상세 training 결과는 [`reports/20260901_model_v2_data_only_training.md`](reports/20260901_model_v2_data_only_training.md)에 있다.

Read-only internal failure audit verdict는 `MODEL_V2_INTERNAL_FAILURE_AUDIT_ACTIONABLE`이다. 여섯 Slip 실패는 모두 low-response miss가 아니라 frozen primary window보다 이른 high-confidence alert였고, 다섯 건은 future-Slip precursor 내부, 한 건은 30 mm precursor보다 5 ms 앞이었다. 실제 남은 실패는 `.25 m/s` delayed Marble Support 0/3으로, I1에서 `>=0.99`가 3 ms만 유지됐다. TRAIN timing과 event-local waveform은 exact match였지만 delayed Marble fit positive는 48개이고 source별 unique event-local waveform은 하나뿐이었다. 따라서 threshold/persistence, sensor, longer history, LSTM 변경보다 `MODEL_V2_EXTRACTION_REBALANCE_DESIGN`이 다음 최소 milestone이다. 재학습과 external Generalization V2 평가는 시작하지 않았고 Generalization HOLDOUT open count는 0이다. 상세 결과는 [`reports/20260901_model_v2_internal_failure_audit.md`](reports/20260901_model_v2_internal_failure_audit.md)에 있다.

Extraction-rebalance design verdict는 `MODEL_V2_EXTRACTION_REBALANCE_DESIGN_READY`다. Baseline V2 candidate와 442-run effective TRAIN을 보존한 채, 18개 delayed-Support TRAIN run 모두에 Concrete/Marble 대칭 규칙으로 I1·interval midpoint·Support의 5 ms causal neighborhood를 배정했다. Dry-run projected fit positive는 2,424→2,590(`+6.85%`)이며 Slip 1,680과 ordinary Support 640, fit negative 25,585는 그대로다. 모든 contradiction count는 0이다. 재학습과 V2_VALIDATION 재평가, external Generalization V2 inference는 아직 시작하지 않았고 Generalization HOLDOUT guard count는 0이다. Frozen design은 [`reports/20260901_model_v2_extraction_rebalance_design.md`](reports/20260901_model_v2_extraction_rebalance_design.md)에 있다.

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

Canonical top-level commands는 `simulate`, `collect`, `train`, `evaluate`, `visualize`, `export`다.

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

## Visualization

대표 TRAIN/VALIDATION case는 deterministic re-simulation parity를 통과한 뒤 MuJoCo viewer에서 frozen Hazard, advisory Terrain, simulator-only physical GT와 함께 재생할 수 있다.

```bash
python scripts/fastreflex.py visualize --run-id uhr_ice_h_c20
```

Representative run과 HUD 해석은 [`docs/simulation.md`](docs/simulation.md#visualizing-supported-hazardterrain-decisions)를 참고한다. HOLDOUT run은 visualization 대상에서 제외된다.

Viewer는 검증된 memory-only snapshot을 재생하며 종료 시 마지막 frame의 `ENDED / PAUSED` 상태로 열린 채 유지된다. `Space`, 방향키, `A/D`, `Home/End`, `R/H/I/T/G`로 재생·seek·event jump를 수행하고, `--pause-at`, `--pause-on-reflex`, `--single-step`, `--mode demo|analysis`를 지원한다.

## Historical evidence

과거 MoS/Stability, direct classification, event-centric, Terrain-gated fusion, continuous Slip와 Support 진단 과정의 결론은 삭제하지 않았다. [`reports/`](reports/)와 Git history가 authoritative evidence/archive다. Current lineage의 핵심 보고서는 다음과 같다.

- [`reports/20260829_unified_hazard_reflex_system.md`](reports/20260829_unified_hazard_reflex_system.md)
- [`reports/20260828_terrain_rebuild_sensor_ablation.md`](reports/20260828_terrain_rebuild_sensor_ablation.md)
- [`reports/20260831_scenario_coverage_matrix_design.md`](reports/20260831_scenario_coverage_matrix_design.md)
- [`reports/20260831_generalization_scenario_calibration.md`](reports/20260831_generalization_scenario_calibration.md)
- [`reports/20260831_ice_generalization_gap_resolution.md`](reports/20260831_ice_generalization_gap_resolution.md)
- [`reports/20260831_generalization_dataset_zero_retrain.md`](reports/20260831_generalization_dataset_zero_retrain.md)
- [`reports/20260901_generalization_failure_mode_audit.md`](reports/20260901_generalization_failure_mode_audit.md)
- [`reports/20260901_ice_near_hazard_target_semantics.md`](reports/20260901_ice_near_hazard_target_semantics.md)
- [`reports/20260901_model_v2_dataset_design.md`](reports/20260901_model_v2_dataset_design.md)
- [`reports/20260901_model_v2_dataset_generation.md`](reports/20260901_model_v2_dataset_generation.md)
- [`reports/20260901_model_v2_data_only_training.md`](reports/20260901_model_v2_data_only_training.md)
- [`reports/20260901_model_v2_internal_failure_audit.md`](reports/20260901_model_v2_internal_failure_audit.md)
- [`reports/20260901_model_v2_extraction_rebalance_design.md`](reports/20260901_model_v2_extraction_rebalance_design.md)

Current architecture는 [`docs/architecture.md`](docs/architecture.md), dataset contract는 [`docs/dataset.md`](docs/dataset.md), 검증 규칙은 [`docs/experiment_protocol.md`](docs/experiment_protocol.md)에 있다.

## Verification

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall src scripts
```

기본 `pytest` command는 project environment의 pytest/plugin version이 일치할 때 사용한다.
