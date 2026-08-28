# Experiment Protocol

## 현재 상태

Historical Pilot/FSR analyses, frozen Slip/Sink criteria와 `SINK_SENSOR_OBSERVABILITY_PROMISING` evidence를 보존한 채 Terrain + Walking Stability + Fusion 구조를 검증한다. Current Terrain verdict는 `TERRAIN_RECOGNITION_SUPPORTED`다. Rebuilt FSR4/MLP/50 ms/LEFT_ONLY candidate의 one-shot holdout macro F1/worst recall은 0.9713/0.95다. 이전 integrated sanity의 exact MoS clock은 stable FP 5/11, fall coverage 22/33의 historical failure 상태이고 Stability AI는 수행하지 않았다. Final sensor architecture와 Stability detector readiness는 unfrozen이다.

## 고정 연구 원칙

- Source code보다 dataset contract를 먼저 확정한다.
- Raw run을 먼저 수집·시각화하고 signal separation 근거를 본 뒤 model을 만든다.
- 실험 차이는 `configs/`에 기록하고 하나의 canonical dataset/training pipeline을 재사용한다.
- Historical runtime baseline은 pelvis IMU6이며, 명시된 Pilot에서만 FSR8/Fusion14 candidate를 비교한다. Terrain/scenario/exact state는 label/diagnostic/metadata 전용이다.
- 기본 입력은 raw channels이고, 필요할 때 train split의 per-channel mean/std normalization만 허용한다.
- Dataset 생성 시 window를 고정하지 않는다.
- Legacy source와 과거 dataset을 bulk copy하지 않는다. 완료된 G1 migration처럼 필요한 범위와 provenance를 먼저 review한다.
- 실제 random source가 있을 때만 seed를 도입하고 code revision, dataset identity, config와 metric을 함께 기록한다.
- Quantization, target conversion, firmware와 HIL은 Research 결과가 freeze된 뒤 E84 deployment repository가 담당한다.

## Terrain/Stability integrated sanity protocol

이 architecture에서 Terrain과 Stability는 서로 독립 producer다. Terrain은 touchdown/patch-contact-centered valid state를 hold하고 Stability는 continuous update한다. Fusion은 Stable이면 NORMAL/recovery false, Unstable이면 terrain에 따라 Slip Risk/Sink Risk/Generic Instability와 recovery true를 만든다. UNKNOWN terrain도 recovery를 막지 않는다.

Ground truth와 runtime detector를 다음처럼 분리한다.

- Exact ground truth: whole-body COM/velocity, active sole polygon, XCoM, signed dynamic MoS, exact contact phase
- Runtime rule/AI: pelvis IMU6 only
- Forbidden runtime shortcut: terrain GT, COM/XCoM/MoS/contact, physical Slip/Sink clock, future fall/outcome와 scenario name

첫 frozen experiment는 hard stable six run의 phase별 MoS 0.5 percentile, additional 10 mm degradation과 20 ms persistence를 predeclared했다. Acceptance는 stable firing ≤5%, fall coverage ≥80%, Ice/Sand detection 존재, median lead ≥100 ms다. Scenario acceptance는 Ice/Sand stable coverage, fall-intended coverage, pre-transition fall 0과 finite sensor를 별도로 검사한다. Scenario 또는 exact clock이 실패하면 rule/AI 성능으로 primary conclusion을 내거나 threshold를 sweep하지 않는다.

실제 결과는 scenario와 exact-clock gate가 모두 FAIL이었다. Pelvis IMU rule은 secondary holdout에서 supported fall 4/4를 결국 감지했지만 stable FP 2/3, Recall@10/20/50/100 ms 0, median latency 353.5 ms였다. Optional GRU는 실행하지 않았다. Config와 report는 [`20260827_terrain_stability_integrated_sanity.yaml`](../configs/experiment/20260827_terrain_stability_integrated_sanity.yaml), [`20260827_terrain_stability_integrated_sanity.md`](../reports/20260827_terrain_stability_integrated_sanity.md)에 기록한다.

## Transition scenario calibration protocol

Scenario repair는 matched A-only reference와 transition의 first target contact 이전 robot/controller prefix를 먼저 비교한다. Same patch scene에서 reference patch만 run 밖으로 옮기고 qpos/qvel, IMU6, FSR8, controller observation/action/update timing, pelvis pose, COM과 contact를 검사한다. Target geometry boundary, top height, no hole/overlap과 pretarget dynamic-support contact도 독립 gate다. Prefix parity가 실패하면 calibration을 시작하지 않는다.

Calibration은 Ice speed/start/width와 Sand speed/start/width/side/pattern/frozen severity만 탐색한다. Ice friction, Sand travel/stiffness/damping, policy/controller, fall/Slip/Sink criterion은 바꾸지 않는다. Observed outcome은 intended role과 무관하게 분류하고, valid stable은 target contact 뒤 finite patch 완전 통과와 non-fall을 요구한다. `CALIBRATION_SELECTED` 뒤 calibration-unused Concrete conditions를 실행하며 결과를 보고 freeze를 수정하지 않는다. Concrete PASS 뒤에만 동일 B conditions에서 A를 Marble로 바꾼다.

실제 결과는 four prefix pair PASS, fresh Concrete Ice/Sand stable 4/fall 4씩, invalid/pre-transition fall 0, Marble Ice stable 3/fall 4와 Sand stable 4/fall 4였다. Verdict는 `TRANSITION_SCENARIOS_CALIBRATED`다. 이 milestone은 Stability oracle이나 detector acceptance가 아니다. Config와 report는 [`20260828_transition_scenario_calibration.yaml`](../configs/experiment/20260828_transition_scenario_calibration.yaml), [`20260828_transition_scenario_calibration.md`](../reports/20260828_transition_scenario_calibration.md)에 기록한다.

## Terrain rebuild and sensor ablation protocol

Terrain label은 exact foot-ground geom identity이고 model input은 touchdown foot의 FSR4/Foot IMU6/Fusion10뿐이다. Primary 50 ms clean event는 same identity continuous contact, mixed samples `<20%`, complete pre-fall causal window를 요구한다. Raw event index는 모두 보존하고 training/evaluation construction만 run/class당 두 event로 cap한다.

Split 88/28/28 runs와 seeds 17/29/43은 simulation 전에 고정했다. Holdout은 integrity/count만 먼저 확인한다. Validation에서 같은 MLP protocol로 three sensor profiles를 비교하고 macro F1≥0.90/worst recall≥0.85 qualified profile 중 best의 2 percentage points 안이면 fewer channels를 선택한다. 그 뒤 selected sensor에서 MLP/GRU, 20/30/50 ms, LEFT_ONLY/BILATERAL_SHARED를 순서대로 선택하고 selection JSON을 쓴 뒤 holdout guard를 한 번만 연다.

실제 선택은 FSR4→MLP→50 ms→LEFT_ONLY였다. Holdout macro F1 0.9713, worst recall 0.95로 `TERRAIN_RECOGNITION_SUPPORTED`다. LEFT_ONLY는 10 integrated channels지만 right-only Sand 18/144 runs에 update가 없고 median/p95 delay 1114.5/1238 ms다. BILATERAL_SHARED는 14 channels, 144/144 coverage와 922/1238 ms다. Recommendation은 `LEFT_FSR4_RECOMMENDED`이나 final freeze는 아니다. Config와 report는 [`20260828_terrain_rebuild_sensor_ablation.yaml`](../configs/experiment/20260828_terrain_rebuild_sensor_ablation.yaml), [`20260828_terrain_rebuild_sensor_ablation.md`](../reports/20260828_terrain_rebuild_sensor_ablation.md)에 기록한다.

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

### Deformable-support proxy sanity — future observability clock

- 보존: 기존 same-height penetration/support-loss 실패, outcome-based Sink contract와 Pilot annotation은 소급 변경하지 않음
- 완료: one-slide balanced plate와 four-cell uneven plate의 passive load-driven vertical motion, initial top parity와 unloaded drift 0 검증
- 완료: support joint displacement spread `>= 10 mm`, 20 ms persistence, same-foot patch episode/load/pre-censor gating 고정
- 결과: rigid/balanced benign `0/14`, primary moderate uneven `11/12`; left/right `6/6`, `5/6`; medial/lateral/localized `4/4`, `3/4`, `4/4`
- 결과: detected fall 1과 detected non-fall 10으로 physical onset과 outcome 분리
- 제한: soil model이나 real-world depth calibration이 아닌 deformable-support engineering proxy이며 runtime sensor observability는 미검증

Verdict는 `SINK_DEFORMABLE_SUPPORT_PROXY_SUPPORTED_FOR_OBSERVABILITY_STUDY`다. 새 clock을 사용하는 future dataset은 새 schema/provenance로 생성해야 하며 기존 Pilot을 relabel하지 않는다. Config와 결과는 [`20260827_sink_deformable_support_proxy_sanity.yaml`](../configs/experiment/20260827_sink_deformable_support_proxy_sanity.yaml), [`20260827_sink_deformable_support_proxy_sanity.md`](../reports/20260827_sink_deformable_support_proxy_sanity.md)에 기록한다.

### Sink sensor observability study

- 완료: v3 `sink_observability_20260827` 126 runs/1,008,000 samples를 pre-frozen 76/25/25 split로 생성
- 완료: observed SINK/BENIGN/INVALID `50/63/13`, left/right `28/22`, medial/lateral/localized `27/11/12` readiness 확인
- 완료: IMU6/FSR8/Fusion14 × MLP/GRU × 3 seeds를 train-only normalization과 같은 100 ms protocol로 비교
- 완료: validation near-tie/secondary recall rule로 IMU6/GRU 선택 후 holdout one-shot과 1 ms causal replay
- 결과: holdout macro F1 0.7307, NORMAL/SINK recall 0.9529/0.7862, Recall@s1/+20/+50/+100 `0.125/0.125/0.125/1.000`
- 결과: median/p95 latency `78.0/91.65 ms`, benign FP `9/13`, balanced deformable FP `6/6`
- 제한: simulation-only Sink-focused evidence이며 final detector/sensor architecture, real sensor와 Full Hazard validation 미완료

Verdict는 `SINK_SENSOR_OBSERVABILITY_PROMISING`이다. Acceptance gate를 결과 뒤 완화하거나 holdout을 재평가하지 않는다. Config와 결과는 [`20260827_sink_sensor_observability_study.yaml`](../configs/experiment/20260827_sink_sensor_observability_study.yaml), [`20260827_sink_sensor_observability_study.md`](../reports/20260827_sink_sensor_observability_study.md)에 기록한다.

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

### Phase 5 — FSR Observability Pilot

- 완료: 기존 40 conditions를 virtual FSR8과 별도 dataset으로 재수집하고 모든 common field bit parity 확인
- 완료: IMU6/FSR8/Fusion14를 동일 early-target, 20/6/7 split, 100 ms MLP, 3 seeds로 비교
- 완료: 1 ms causal replay, 10 ms sustained argmax와 20/30/50/100 ms horizon/benign FP audit
- 결과: Fusion14 SINK@100 ms `0.4074 ± 0.1048`, pooled positive-margin pre-t2 latency 101 ms
- 결과: Fusion14 total benign FP `5.67 ± 1.25/16`, Uniform Sand `1.67 ± 1.25/4`; mild/moderate Sink는 계속 `2/2`
- 제한: idealized MuJoCo normal load이며 actual FSR hardware, calibration과 mounting 미검증

Config와 결과는 [`20260827_fsr_observability_pilot.yaml`](../configs/experiment/20260827_fsr_observability_pilot.yaml), [`20260827_fsr_observability_pilot.md`](../reports/20260827_fsr_observability_pilot.md)에 기록한다.

### FSR Load Distribution Analysis

- 완료: 기존 sensor Pilot을 수정하지 않고 mild/moderate benign 4 vs severe hazardous 9 run을 t0/t1 기준으로 비교
- 완료: affected-foot canonicalization, low-load invalid handling, 10 ms causal horizon median과 pre-event delta 적용
- 결과: t0+20 CoP 신호는 50/100 ms에 반복되지 않았고, t1+50 medial-ratio delta는 AUROC 0.893이나 range overlap
- 결과: t1+100 affected total은 Pilot에서 완전 분리했지만 Uniform Sand와 겹치며, medial-ratio delta는 AUROC 0.929
- 결론: `FSR_LOAD_DISTRIBUTION_LATE_SEPARATION_ONLY`; handcrafted feature, threshold, trained detector와 sensor architecture는 미확정

Config와 결과는 [`20260827_fsr_load_distribution_analysis.yaml`](../configs/experiment/20260827_fsr_load_distribution_analysis.yaml), [`20260827_fsr_load_distribution_analysis.md`](../reports/20260827_fsr_load_distribution_analysis.md)에 기록한다.

### FSR Temporal Redistribution Analysis

- 완료: 기존 raw FSR8을 수정하지 않고 event initial/endpoint median과 raw 1 ms continuous-valid path 계산
- 완료: t0/t1 `0/20/30/50/75/100/150/200/300 ms`, pre-event instability와 t2-in-horizon flag 분석
- 결과: t1+20 front absolute change는 AUROC 1.000이나 +30 ms에 0.639로 하락
- 결과: t1+100 CoP path는 AUROC 1.000이나 Uniform Sand path가 severe보다 훨씬 큼
- 결론: `FSR_TEMPORAL_REDISTRIBUTION_NO_ADDED_VALUE`; temporal feature, classifier와 threshold는 미확정

Config와 결과는 [`20260827_fsr_temporal_redistribution_analysis.yaml`](../configs/experiment/20260827_fsr_temporal_redistribution_analysis.yaml), [`20260827_fsr_temporal_redistribution_analysis.md`](../reports/20260827_fsr_temporal_redistribution_analysis.md)에 기록한다.

### Phase 6 — Sensor architecture decision

- IMU6 historical evidence와 FSR/Fusion Pilot의 early recall, soft-ground FP, seed/side stability를 함께 review
- 실제 FSR의 hysteresis, drift, saturation, mounting, sample alignment를 반영한 후 채택 여부 결정
- Pilot 결과만으로 hardware channel count나 final runtime input을 freeze하지 않음

### Phase 7 — Full dataset generation

- Pilot에서 검증된 하나의 canonical generator와 schema 사용
- Speed, gait phase, initiating side, scenario family, seed, terrain realization을 class별로 점검
- NORMAL coverage를 우선 충분히 확보하고 hard-but-valid touchdown/turn/transition을 포함
- Split manifest를 run/group 단위로 freeze하고 TEST를 seal

### Phase 8 — PyTorch baseline comparison

동일 raw release, split, preprocessing, window/metric 정의로 다음 family를 비교한다.

- MLP baseline
- CNN1D
- GRU
- LSTM

Model별 handcrafted feature나 별도 dataset runner를 만들지 않는다. 첫 PoC의 canonical MLP/GRU를 재사용하고 Full Dataset이 준비된 뒤에만 CNN1D/LSTM을 더해 전체 family를 비교한다.

### Phase 9 — Run/group-disjoint validation

- Validation으로 architecture, window와 hyperparameter 선택
- Source-run overlap 0을 자동 검증
- 가능한 범위에서 seed/variation/scenario realization/initial gait phase group overlap 0 검증
- Class별 metric, confusion matrix, false-positive rate, causal latency와 coverage strata 보고
- 일부 held-out scenario combination에서 generalization 확인
- TEST는 model 선택이나 threshold 재조정에 사용하지 않음

### Phase 10 — Frozen Float model export

- 선택이 완료된 Float model과 정확한 input/output contract freeze
- Dataset ID, split revision, train-only normalization, config, code revision과 metric provenance 포함
- Research repository에는 검토된 Float contract artifact만 export
- Quantization 이후 작업은 `Infineon_FastReflex_E84` repository로 넘김

연구 순서는 `MuJoCo baseline → historical Sink/Slip criteria → Pilot/FSR/deformable-support evidence → Terrain+Stability architecture → transition calibration → rebuilt Terrain support → Stability ground-truth redesign → accepted pelvis IMU rule/optional GRU → joint sensor decision`이다. 다음 단계는 자동으로 시작하지 않으며 exact Stability clock을 먼저 사전 선언한다.

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

Current milestone은 rebuilt Terrain raw dataset, Foot IMU observer, sensor/model/horizon/deployment ablation과 one-shot holdout까지 실행했다. Stability ground-truth redesign, Stability AI, recovery controller, final sensor architecture freeze, full integrated dataset, quantization, E84 또는 HIL은 수행하지 않았다.
