# Experiment Protocol

## 현재 상태

`MINIMAL_G1_MUJOCO_MIGRATION`, Sink/Slip transition과 criteria freeze 뒤 40-run `hazard_pilot_20260827` raw artifact를 materialize했으며 현재 상태는 `PILOT_DATASET_READY`다. 이는 source/manifest/NPZ 구조와 physical outcome coverage가 준비됐다는 뜻이며 IMU signal separability, window, model, training 또는 evaluation이 준비됐다는 뜻은 아니다. 다음 단계에서도 [`dataset.md`](dataset.md)의 sensor, physical label, raw-run, split contract를 변경 review 없이 깨뜨리지 않는다.

## 고정 연구 원칙

- Source code보다 dataset contract를 먼저 확정한다.
- Raw run을 먼저 수집·시각화하고 signal separation 근거를 본 뒤 model을 만든다.
- 실험 차이는 `configs/`에 기록하고 하나의 canonical dataset/training pipeline을 재사용한다.
- Runtime input은 pelvis IMU6뿐이며 terrain/scenario/exact state는 label/diagnostic/metadata 전용이다.
- 기본 입력은 raw channels이고, 필요할 때 train split의 per-channel mean/std normalization만 허용한다.
- Dataset 생성 시 window를 고정하지 않는다.
- Legacy source와 과거 dataset을 bulk copy하지 않는다. 완료된 G1 migration처럼 필요한 범위와 provenance를 먼저 review한다.
- 실제 random source가 있을 때만 seed를 도입하고 code revision, dataset identity, config와 metric을 함께 기록한다.
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
- 완료: finite patch 진입 t0, physical sink t1, degradation t2와 censor t3 분리
- 완료: benign-control tilt envelope 기반 primary `SINK` effect gate와 20 ms persistence freeze
- 미확정: `[t1,t2)` training label/window timing과 IMU-only observability

현재 상태는 `SINK_HAZARD_CRITERIA_FROZEN`이다. `sink_physical_active`만으로 primary `SINK`를 만들지 않는다. Full-lane history는 [`20260826_sink_scenario_sanity.md`](../reports/20260826_sink_scenario_sanity.md), finite transition과 freeze 근거는 [`20260826_sink_transition_criteria.md`](../reports/20260826_sink_transition_criteria.md)에 기록한다.

### Slip transition sanity — Pilot 이전 단계

- 완료: concrete → full-width finite Ice patch → concrete topology
- 완료: t0 patch contact와 per-foot/ANY-foot established Slip t1 분리
- 완료: 0.10/0.15/0.20/0.25 m/s에서 clean bilateral Slip과 Viewer 확인
- 유지: frozen 50 mm/3 ms oracle, pre-fall validity와 terrain-label 금지

Slip transition 결과는 [`20260826_slip_transition_sanity.yaml`](../configs/experiment/20260826_slip_transition_sanity.yaml)과 [`20260826_slip_transition_sanity.md`](../reports/20260826_slip_transition_sanity.md)에 기록한다. 이 단계로 Pilot 이전 simulator condition 설계는 일단 종료한다.

### Phase 3 — Small pilot raw dataset generation

- 완료: 16 NORMAL-intended, 12 Slip-intended, 12 Sink-intended의 40개 complete 8초 run 생성
- 완료: 320,000개 raw pelvis IMU6 sample, 1 kHz timestamp, drop/invalid sensor 0 검증
- 완료: observed BENIGN/SLIP/SINK/DUAL/INVALID = 16/8/9/0/7과 left/right event coverage 확인
- 완료: one-run-per-NPZ, manifest/metadata, per-file/manifest SHA와 fail-closed atomic finalization 검증
- 유지: `[t1,t2)` SINK interval과 invalid/dual/censor sample은 training-ineligible
- 제한: full dataset, split, signal separability와 model 성능은 주장하지 않음

Config와 실제 결과는 [`20260827_hazard_pilot_dataset.yaml`](../configs/experiment/20260827_hazard_pilot_dataset.yaml), [`20260827_hazard_pilot_dataset.md`](../reports/20260827_hazard_pilot_dataset.md)에 기록한다.

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

연구 순서는 `MuJoCo baseline → Sink transition/criteria freeze → Slip transition sanity → Pilot Dataset → raw IMU sanity → Time-to-Separation → Full Dataset → PyTorch model comparison`이다. Pilot Dataset까지 완료했다. 다음 단계는 별도 승인 뒤 Phase 4 Raw IMU sanity이며 Slip t1 주변과 hazardous Sink의 `[t1,t2)`에서 pelvis IMU6 관측 가능성을 먼저 확인한다. Time-to-Separation이나 ML을 자동으로 시작하지 않는다.

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

현재 milestone은 smoke simulation, Sink/Slip finite transition condition과 bounded raw Pilot Dataset materialization까지만 실행했다. Raw signal 분석, Time-to-Separation, Full Dataset, PyTorch 설치, model 구현/training/evaluation, quantization, E84 또는 HIL은 수행하지 않았다.
