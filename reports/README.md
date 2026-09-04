# Experiment Reports

이 directory는 bounded dataset·experiment별 결과와 provenance를 보존한다.
현재 상태와 설계 계약은 각각 repository
[README](../README.md)와 `docs/`의 canonical 문서가 기준이다.

## Recommended Review Path

전체 이력을 순서대로 읽을 필요는 없다. 현재 판단을 검토할 때는 아래 보고서부터
보는 것을 권장한다.

| 순서 | 보고서 | 확인할 내용 |
|---:|---|---|
| 1 | [Hazard boundary resolution](20260904_hazard_boundary_resolution.md) | 최신 물리 validation 실패, 중단 근거, 다음 단계 |
| 2 | [Factor-conditioned model training](20260904_sand_factor_conditioned_model_training.md) | Sand 개선과 Support 회귀 trade-off |
| 3 | [Generalization HOLDOUT](20260902_model_v2_generalization_holdout_one_shot_evaluation.md) | Model V2가 새 simulation scenario gate를 실패한 직접 근거 |
| 4 | [HOLDOUT failure interpretation](20260902_model_v2_holdout_failure_interpretation.md) | 실패를 data/physics 관점에서 해석한 근거 |
| 5 | [Unified Hazard baseline](20260829_unified_hazard_reflex_system.md) | 기존 제한 조건에서 지원되는 baseline |
| 6 | [Terrain sensor ablation](20260828_terrain_rebuild_sensor_ablation.md) | FSR4/MLP/50 ms advisory 선택 근거 |

Deployment handoff만 검토할 때는
[Float numerical contract](20260903_float_numerical_contract_resolution.md)와
[INT8 calibration handoff](20260903_int8_calibration_handoff.md)를 추가로 확인한다.
두 보고서는 engineering reference를 다루며 scientific generalization이나
real-robot 성능을 주장하지 않는다.

현재
[deployment-aware QAT](../configs/experiment/20260904_deployment_aware_qat.yaml)은
protocol과 implementation만 준비된 상태다. 아직 reviewed result report가 없으며
scientific lineage와 분리된 engineering work로 취급한다.

## How to Read the History

- 보고서는 당시 판단을 보존하는 immutable research evidence다.
- 후속 보고서가 이전 가설을 기각하더라도 과거 보고서를 수정하거나 삭제하지 않는다.
- 날짜가 최신이라는 이유만으로 결과가 더 강한 것은 아니다. 각 report의 split,
  authorization, gate, verdict를 함께 확인한다.
- Config가 현재 CLI에서 실행되지 않으면 report/config의 `source_commit`을
  사용한다. Current source가 historical runner를 추측해 재현하지 않는다.
- 전체 흐름은 baseline 확립, generalization 실패, Sand/Support 원인 탐색,
  factor-conditioned intervention 실패, boundary validation 물리 실패 순서다.

## Complete Chronological Index

### 2026-08-26 — Initial physical scenarios

- [Slip transition sanity](20260826_slip_transition_sanity.md)
- [Sink scenario sanity](20260826_sink_scenario_sanity.md)
- [Sink transition criteria](20260826_sink_transition_criteria.md)

### 2026-08-27 — Sensor observability and physical Hazard definition

- [First classification proof of concept](20260827_first_classification_poc.md)
- [FSR load-distribution analysis](20260827_fsr_load_distribution_analysis.md)
- [FSR observability pilot](20260827_fsr_observability_pilot.md)
- [FSR temporal redistribution analysis](20260827_fsr_temporal_redistribution_analysis.md)
- [Hazard pilot dataset](20260827_hazard_pilot_dataset.md)
- [Sink deformable-support proxy sanity](20260827_sink_deformable_support_proxy_sanity.md)
- [Sink physical Hazard redefinition](20260827_sink_physical_hazard_redefinition.md)
- [Sink sensor observability study](20260827_sink_sensor_observability_study.md)
- [Sink support-loss oracle sanity](20260827_sink_support_loss_oracle_sanity.md)
- [Terrain/stability integrated sanity](20260827_terrain_stability_integrated_sanity.md)
- [Time to separation](20260827_time_to_separation.md)

### 2026-08-28 — Detector and architecture selection

- [Dense fall-risk detector proof of concept](20260828_dense_fall_risk_detector_poc.md)
- [Event-centric Reflex trigger](20260828_event_centric_reflex_trigger.md)
- [Full-state stability ground-truth sanity](20260828_full_state_stability_ground_truth_sanity.md)
- [Temporal stability separability audit](20260828_temporal_stability_separability_audit.md)
- [Terrain-conditioned Reflex detector](20260828_terrain_conditioned_reflex_detector.md)
- [Terrain rebuild and sensor ablation](20260828_terrain_rebuild_sensor_ablation.md)
- [Transition-scenario calibration](20260828_transition_scenario_calibration.md)
- [Walking-stability ground-truth sanity](20260828_walking_stability_ground_truth_sanity.md)

### 2026-08-29 — Unified baseline

- [Causal Support/Terrain context fusion](20260829_causal_support_terrain_context_fusion.md)
- [Continuous Slip Reflex detector](20260829_continuous_slip_reflex_detector.md)
- [Per-foot Terrain-memory Support fusion](20260829_per_foot_terrain_memory_support_fusion.md)
- [Support early-mode resolution](20260829_support_early_mode_resolution.md)
- [Support failure-mode audit](20260829_support_failure_mode_audit.md)
- [Unified Hazard Reflex system](20260829_unified_hazard_reflex_system.md)

### 2026-08-31 — Generalization dataset preparation

- [Scenario-coverage matrix design](20260831_scenario_coverage_matrix_design.md)
- [Generalization scenario calibration](20260831_generalization_scenario_calibration.md)
- [Ice generalization-gap resolution](20260831_ice_generalization_gap_resolution.md)
- [Zero-retrain generalization evaluation](20260831_generalization_dataset_zero_retrain.md)

### 2026-09-01 — Model V2 dataset and training

- [Generalization failure-mode audit](20260901_generalization_failure_mode_audit.md)
- [Ice near-Hazard target semantics](20260901_ice_near_hazard_target_semantics.md)
- [Model V2 dataset design](20260901_model_v2_dataset_design.md)
- [Model V2 dataset generation](20260901_model_v2_dataset_generation.md)
- [Model V2 data-only training](20260901_model_v2_data_only_training.md)
- [Model V2 internal failure audit](20260901_model_v2_internal_failure_audit.md)
- [Model V2 extraction-rebalance design](20260901_model_v2_extraction_rebalance_design.md)
- [Model V2 extraction-rebalanced training](20260901_model_v2_extraction_rebalanced_training.md)
- [Model V2 rebalance regression audit](20260901_model_v2_rebalance_regression_audit.md)
- [Model V2 delayed-Support anchor-refinement design](20260901_model_v2_delayed_support_anchor_refinement_design.md)

### 2026-09-02 — Candidate freeze, HOLDOUT, and Sand study

- [Model V2 anchor-refined training](20260902_model_v2_anchor_refined_training.md)
- [Model V2 candidate-readiness review](20260902_model_v2_candidate_readiness_review.md)
- [Model V2 generalization development evaluation](20260902_model_v2_generalization_development_evaluation.md)
- [Model V2 final-candidate HOLDOUT-readiness review](20260902_model_v2_final_candidate_holdout_readiness_review.md)
- [Model V2 generalization HOLDOUT one-shot evaluation](20260902_model_v2_generalization_holdout_one_shot_evaluation.md)
- [Model V2 HOLDOUT failure interpretation](20260902_model_v2_holdout_failure_interpretation.md)
- [Sand-benign generalization-study design](20260902_sand_benign_generalization_study_design.md)
- [Sand-benign generalization-study generation](20260902_sand_benign_generalization_study_generation.md)
- [Sand-benign generalization-study redesign](20260902_sand_benign_generalization_study_redesign.md)
- [Sand-benign redesigned-study generation](20260902_sand_benign_generalization_study_redesigned_generation.md)
- [Sand-benign redesigned-domain failure review](20260902_sand_benign_redesigned_domain_failure_review.md)

### 2026-09-03 — Sand/Support hypothesis and physical recalibration

- [Mild-domain calibration redesign](20260903_sand_benign_mild_domain_calibration_redesign.md)
- [Mild-recalibrated study redesign](20260903_sand_benign_generalization_study_mild_recalibrated_redesign.md)
- [Mild-recalibrated study generation](20260903_sand_benign_generalization_study_mild_recalibrated_generation.md)
- [Mild-recalibrated discovery analysis](20260903_sand_benign_generalization_study_mild_recalibrated_discovery_analysis.md)
- [Mild-recalibrated confirmation analysis](20260903_sand_benign_generalization_study_mild_recalibrated_confirmation_analysis.md)
- [Sand generalization-hypothesis review](20260903_sand_generalization_hypothesis_review.md)
- [Factor-conditioned data intervention](20260903_sand_factor_conditioned_data_intervention.md)
- [Data-intervention failure audit](20260903_data_intervention_failure_audit.md)
- [Factor-conditioned recalibrated generation](20260903_sand_factor_conditioned_development_recalibrated_generation.md)
- [Delayed-Support physical review](20260903_sand_factor_conditioned_delayed_support_physical_review.md)
- [Support-recalibrated generation](20260903_sand_factor_conditioned_development_support_recalibrated_generation.md)
- [Ordinary-Support failure review](20260903_sand_factor_conditioned_ordinary_support_failure_review.md)
- [Float numerical contract](20260903_float_numerical_contract_resolution.md)
- [INT8 calibration handoff](20260903_int8_calibration_handoff.md)

### 2026-09-04 — Final data intervention and boundary stop

- [Controls-recalibrated generation](20260904_sand_factor_conditioned_development_controls_recalibrated_generation.md)
- [Factor-conditioned model training](20260904_sand_factor_conditioned_model_training.md)
- [Hazard boundary-resolution cycle](20260904_hazard_boundary_resolution.md)
