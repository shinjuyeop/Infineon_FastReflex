# Experiment Protocol

## 현재 상태

`MINIMAL_G1_MUJOCO_MIGRATION`과 bounded `SINK_HAZARD_SCENARIO_SANITY`를 완료했으며 현재 상태는 `MUJOCO_BASELINE_READY`다. G1 보행, pelvis IMU6 1 kHz, physical diagnostic foundation과 spatially asymmetric compliant-contact scenario까지만 검증했다. Dataset, model, training과 evaluation pipeline은 아직 구현되지 않았다. 다음 단계에서도 [`dataset.md`](dataset.md)의 sensor, physical label, raw-run, split contract를 변경 review 없이 깨뜨리지 않는다.

## 고정 연구 원칙

- Source code보다 dataset contract를 먼저 확정한다.
- Raw run을 먼저 수집·시각화하고 signal separation 근거를 본 뒤 model을 만든다.
- 실험 차이는 `configs/`에 기록하고 하나의 canonical dataset/training pipeline을 재사용한다.
- Runtime input은 pelvis IMU6뿐이며 terrain/scenario/exact state는 label/diagnostic/metadata 전용이다.
- 기본 입력은 raw channels이고, 필요할 때 train split의 per-channel mean/std normalization만 허용한다.
- Dataset 생성 시 window를 고정하지 않는다.
- Legacy source와 과거 dataset을 bulk copy하지 않는다. 완료된 G1 migration처럼 필요한 범위와 provenance를 먼저 review한다.
- Random seed, code revision, dataset revision, config와 metric을 함께 기록한다.
- Quantization, target conversion, firmware와 HIL은 Research 결과가 freeze된 뒤 E84 deployment repository가 담당한다.

## 연구 단계

### Phase 1 — Dataset contract

- IMU6 schema, frame, units, timestamp와 missing-sample 정책 고정
- NORMAL/SLIP class 원칙, Sink cause/effect 분리와 dual/early interval 정책 정의
- authoritative raw-run storage, provenance, window 후보와 split 원칙 고정
- Deliverable: `docs/dataset.md`, `configs/dataset/hazard.yaml`

이 문서 milestone이 현재 Phase 1의 결과다.

### Phase 2 — Minimal G1 MuJoCo migration

- 완료: Pelvis IMU6 취득에 필요한 최소 G1 model/sensor/controller 경로만 명시적으로 review 후 migration
- 완료: IMU site/frame/axis/unit/channel과 2 kHz physics/1 kHz timestamp test
- 완료: bilateral physical contact/touchdown, locked Slip과 `sink_physical` threshold/persistence parity test
- 준수: Legacy training/dataset/model, terrain classifier와 deployment logic은 migration하지 않음

### Sink scenario sanity — classifier 이전 단계

- 완료: 기존 uniform sand를 benign control로 보존
- 완료: 동일 nominal height의 left/right compliance lane과 mild/moderate/severe config selection
- 완료: `sink_physical` cause와 pelvis posture/velocity/contact/fall effect diagnostic 분리
- 완료: bounded 8-run sanity matrix와 Viewer 확인
- 미확정: primary `SINK` effect metric 조합, numeric gate, training label timing

현재 상태는 `SINK_HAZARD_CRITERIA_NOT_YET_FROZEN`이다. `sink_physical_active`만으로 primary `SINK`를 만들지 않는다. Scenario와 결과는 [`20260826_sink_scenario_sanity.yaml`](../configs/experiment/20260826_sink_scenario_sanity.yaml) 및 [`20260826_sink_scenario_sanity.md`](../reports/20260826_sink_scenario_sanity.md)에 기록한다.

### Phase 3 — Small pilot raw dataset generation

- 소수 run으로 full-length 1 kHz raw trace 생성
- Manifest/provenance, missing sample, run boundary와 label diagnostics 검증
- 다양한 normal contact, established Slip과 review 후 frozen된 SINK hazard event가 실제로 포함되는지 확인
- 이 단계에서는 full dataset이나 model 성능을 주장하지 않음

### Phase 4 — Raw IMU visualization / Time-to-Separation

- IMU6와 exact continuous physical metrics를 동일 timeline에 표시
- 20/30/50/100 ms causal history의 관측 가능성 확인
- Stable established onset과 최초 physical motion onset을 구분
- Legacy에서 실패한 incipient Slip 정의를 재사용하거나 새 threshold를 임의로 만들지 않음
- Early interval, endpoint label과 latency anchor는 검증 결과를 근거로 별도 contract revision에서 결정

### Phase 5 — Full dataset generation

- Pilot에서 검증된 하나의 canonical generator와 schema 사용
- Speed, gait phase, initiating side, scenario family, seed, terrain realization을 class별로 점검
- NORMAL coverage를 우선 충분히 확보하고 hard-but-valid touchdown/turn/transition을 포함
- Split manifest를 run/group 단위로 freeze하고 TEST를 seal

### Phase 6 — PyTorch baseline comparison

동일 raw release, split, preprocessing, window/metric 정의로 다음 family를 비교한다.

- MLP baseline
- CNN1D
- GRU
- LSTM

Model별 handcrafted feature나 별도 dataset runner를 만들지 않는다. PyTorch 설치와 model 구현은 이 phase 전에는 하지 않는다.

### Phase 7 — Run/group-disjoint validation

- Validation으로 architecture, window와 hyperparameter 선택
- Source-run overlap 0을 자동 검증
- 가능한 범위에서 seed/variation/scenario realization/initial gait phase group overlap 0 검증
- Class별 metric, confusion matrix, false-positive rate, causal latency와 coverage strata 보고
- 일부 held-out scenario combination에서 generalization 확인
- TEST는 model 선택이나 threshold 재조정에 사용하지 않음

### Phase 8 — Frozen Float model export

- 선택이 완료된 Float model과 정확한 input/output contract freeze
- Dataset ID, split revision, train-only normalization, config, code revision과 metric provenance 포함
- Research repository에는 검토된 Float contract artifact만 export
- Quantization 이후 작업은 `Infineon_FastReflex_E84` repository로 넘김

다음 단계는 Sink effect gate와 timing에 대한 사람의 review다. 이를 freeze하기 전에는 Phase 3 pilot raw dataset generation을 시작하지 않으며, review 뒤에도 별도 승인 없이 자동으로 진행하지 않는다.

## Split과 leakage protocol

Window-level random split은 금지한다. 모든 derived window는 source `run_id`의 split을 상속한다. 권장 초기 비율 70/15/15는 dataset coverage가 허용할 때만 사용하며 실제 run 수와 split seed는 pilot 뒤 config로 freeze한다.

각 split release에서 최소 다음 audit을 남긴다.

- train/validation/test run ID intersection 0
- declared group intersection과 held-out scenario 목록
- class 및 speed/gait phase/side/scenario/seed/terrain realization coverage
- duplicate trace/hash audit
- normalization input이 TRAIN뿐인지 확인
- TEST materialization/access history

Legacy 연구의 중요한 교훈은 normal domain이 좁으면 touchdown, gait phase, speed, turn과 sensor variation이 hazard shortcut처럼 동작하여 unseen domain false positive를 만들 수 있다는 점이다. 따라서 class count만 맞추지 않고 NORMAL domain coverage와 run/group independence를 acceptance condition으로 본다. 과거 dataset은 새 dataset에 복사하지 않는다.

## Evaluation boundary

구체적인 metric acceptance threshold는 pilot과 Phase 4 결과 뒤 고정한다. 단, 향후 모든 family에 동일한 정의를 적용하고 다음을 함께 보고한다.

- overall accuracy와 macro metric
- NORMAL/SLIP/SINK별 precision, recall, F1와 support
- confusion matrix
- run/event 기준 false positive와 miss
- established onset 기준 causal detection latency
- window size, inference stride와 유효 window coverage
- speed/gait phase/side/scenario/seed/terrain group별 strata

Established onset은 최초 motion onset이 아니므로 early-detection 주장에는 별도로 검증된 latency anchor가 필요하다. 검증 전에는 20/30/50 ms early-detection 결론을 내리지 않는다.

## Repository boundary

Research repository 범위:

- dataset contract와 minimal simulator migration
- raw dataset/provenance, windowing, PyTorch training과 run/group-disjoint evaluation
- frozen Float model 및 계약 artifact

E84 deployment repository 범위:

- quantization과 target conversion
- Vela, firmware integration
- E84 runtime, HIL과 target validation

현재 milestone은 smoke simulation과 bounded Sink scenario sanity study만 실행했다. Dataset 생성, PyTorch 설치, model 구현/training/evaluation, quantization, E84 또는 HIL은 수행하지 않았다.
