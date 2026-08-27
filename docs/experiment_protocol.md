# Experiment Protocol

## 현재 상태

`MINIMAL_G1_MUJOCO_MIGRATION`, Sink/Slip criteria freeze, 40-run Pilot과 first established-state PoC 뒤 frozen MLP 100 ms의 1 ms causal Time-to-Separation replay까지 완료했다. 현재 상태는 `TIME_TO_SEPARATION_ANALYZED`다. SLIP early-signal candidate는 있었지만 SINK 20~100 ms recall과 benign soft-ground false firing이 detector readiness를 지지하지 않으므로 final latency, Full Dataset, final model 또는 deployment readiness를 뜻하지 않는다. 다음 단계에서도 [`dataset.md`](dataset.md)의 sensor, physical label, raw-run, split contract를 변경 review 없이 깨뜨리지 않는다.

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

### First established-state classification PoC

- 완료: manifest metadata만으로 33 valid run을 20/6/7 run-disjoint split로 고정하고 INVALID 7 run 제외
- 완료: 50/100 ms causal same-class eligible window, 10 ms stride와 train-only per-run/class cap 적용
- 완료: train-only z-score와 MLP/GRU 4 candidates × 3 fixed seeds validation 비교
- 완료: validation으로 MLP 100 ms 선택 후 Pilot internal holdout을 한 번 평가해 macro F1 0.8614 확인
- 유지: model input은 pelvis IMU6뿐이며 exact diagnostic은 label/sanity/failure analysis에만 사용
- 제한: 모든 window가 이미 established class이므로 onset-crossing latency와 early detection은 검증하지 않음

Config와 실제 결과는 [`20260827_first_classification_poc.yaml`](../configs/experiment/20260827_first_classification_poc.yaml), [`20260827_first_classification_poc.md`](../reports/20260827_first_classification_poc.md)에 기록한다.

### Phase 4 — Time-to-Separation

- 완료: first-PoC split, train normalizer와 MLP 100 ms 3-seed checkpoint SHA를 고정하고 재학습 없이 replay
- 완료: 33 valid run full trace를 future sample 없이 1 ms stride로 argmax/probability/logit 기록
- 완료: 10 ms diagnostic persistence와 t1+0/20/30/50/100 ms event recall, benign/pre-t0 FP audit
- 결과: SLIP Recall@100 ms 0.8333 ± 0.0589, SINK 0.1111 ± 0.0000
- 결과: positive-margin SINK 7/7은 결국 t2 전에 검출되지만 median t1 latency는 seed별 296/425/435 ms
- 결과: BENIGN sustained hazard FP 8/16, 8/16, 7/16이며 Uniform Sand는 모든 seed에서 4/4
- 제한: joint promising 20~100 ms horizon 없음; persistence/threshold/early label을 freeze하지 않음

Config와 실제 결과는 [`20260827_time_to_separation.yaml`](../configs/experiment/20260827_time_to_separation.yaml), [`20260827_time_to_separation.md`](../reports/20260827_time_to_separation.md)에 기록한다.

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

Model별 handcrafted feature나 별도 dataset runner를 만들지 않는다. 첫 PoC의 canonical MLP/GRU를 재사용하고 Full Dataset이 준비된 뒤에만 CNN1D/LSTM을 더해 전체 family를 비교한다.

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

연구 순서는 `MuJoCo baseline → Sink transition/criteria freeze → Slip transition sanity → Pilot Dataset → First established-state PoC → Time-to-Separation → Full Dataset → full PyTorch model comparison`이다. Time-to-Separation까지 분석했다. 다음 단계는 별도 승인 뒤 Phase 5 Full Dataset design이며 continuous benign soft-ground coverage, onset-relative provenance와 fresh test reservation을 먼저 다룬다. Dataset 생성이나 full model comparison을 자동으로 시작하지 않는다.

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

Pilot과 Phase 4 결과만으로 metric acceptance threshold를 고정하지 않는다. Full Dataset에서 fresh test와 normal-domain coverage가 준비된 뒤 향후 모든 family에 동일한 정의를 적용하고 다음을 함께 보고한다.

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

현재 milestone은 smoke simulation, Sink/Slip finite transition, bounded raw Pilot, 첫 established-state PoC와 frozen-classifier Time-to-Separation replay까지 실행했다. Full Dataset, retraining, CNN/LSTM을 포함한 full model comparison, quantization, E84 또는 HIL은 수행하지 않았다.
