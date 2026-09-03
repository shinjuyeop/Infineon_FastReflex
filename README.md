# Infineon FastReflex

Unitree G1 MuJoCo에서 검증된 physical Hazard Reflex와 Terrain advisory를 관리하는 research repository다.

## Current Status

`UNIFIED_BASELINE_SUPPORTED; ZERO_RETRAIN_GENERALIZATION_NOT_SUPPORTED; ICE_NEAR_HAZARD_TARGET_SEMANTICS_RESOLVED; MODEL_V2_DATASET_GENERATION_READY; MODEL_V2_DATA_ONLY_TRAINING_COMPLETE; MODEL_V2_EXTRACTION_REBALANCED_TRAINING_COMPLETE; MODEL_V2_ANCHOR_REFINED_TRAINING_COMPLETE; V2_ANCHOR_REFINEMENT_EFFECTIVE; MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED; MODEL_V2_CANDIDATE_READINESS_REVIEW_COMPLETE; MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION_COMPLETE; GENERALIZATION_PRIMARY_GATES_FAIL; GENERALIZATION_DEVELOPMENT_SUPPORTED_WITH_ICE_TIMING_TENSION; MODEL_V2_FINAL_CANDIDATE_HOLDOUT_READINESS_REVIEW_COMPLETE; FINAL_GENERALIZATION_CANDIDATE_FROZEN; MODEL_V2_GENERALIZATION_HOLDOUT_ONE_SHOT_EVALUATION_COMPLETE; GENERALIZATION_HOLDOUT_PRIMARY_GATES_FAIL; MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED; MODEL_V2_HOLDOUT_FAILURE_INTERPRETATION_ACTIONABLE; SAND_BENIGN_GENERALIZATION_STUDY_DESIGN_READY; SAND_BENIGN_GENERALIZATION_STUDY_GENERATION_COMPLETE; SAND_BENIGN_GENERALIZATION_STUDY_GENERATION_PHYSICAL_YIELD_INSUFFICIENT; SAND_BENIGN_PHYSICAL_CALIBRATION_COMPLETE; SAND_BENIGN_GENERALIZATION_STUDY_REDESIGN_READY; SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION_COMPLETE; SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION_YIELD_INSUFFICIENT; SAND_BENIGN_REDESIGNED_DOMAIN_FAILURE_REVIEW_ACTIONABLE; SAND_BENIGN_MILD_DOMAIN_CALIBRATION_REDESIGN_READY; SAND_BENIGN_MILD_DOMAIN_MINIMAL_REDESIGN_READY; SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_GENERATION_READY; SAND_BENIGN_GENERALIZATION_STUDY_DISCOVERY_ANALYSIS_VALID; DOMAIN_DIVERSITY_GAP_SUPPORTED; SAND_BENIGN_GENERALIZATION_STUDY_CONFIRMATION_ANALYSIS_VALID; DOMAIN_DIVERSITY_GAP_NOT_CONFIRMED; SAND_GENERALIZATION_HYPOTHESIS_REVIEW_ACTIONABLE; FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS; SAND_FACTOR_CONDITIONED_DATA_INTERVENTION_GENERATION_COMPLETE; FACTOR_CONDITIONED_DATASET_GENERATION_GATES_FAILED; FACTOR_CONDITIONED_DATA_INTERVENTION_INVALID; DATA_INTERVENTION_FAILURE_AUDIT_COMPLETE; FACTOR_CONDITIONED_PHYSICAL_DOMAIN_REDESIGN_READY; SAND_FACTOR_CONDITIONED_DEVELOPMENT_RECALIBRATED_GENERATION_COMPLETE; SAND_FACTOR_CONDITIONED_DEVELOPMENT_RECALIBRATED_GENERATION_INSUFFICIENT; SAND_FACTOR_CONDITIONED_DEVELOPMENT_SUPPORT_RECALIBRATED_GENERATION_COMPLETE; SAND_FACTOR_CONDITIONED_DEVELOPMENT_SUPPORT_RECALIBRATED_GENERATION_INSUFFICIENT; SAND_FACTOR_CONDITIONED_ORDINARY_SUPPORT_FAILURE_REVIEW_COMPLETE; ORDINARY_SUPPORT_PHYSICAL_RECALIBRATION_READY; SAND_FACTOR_CONDITIONED_DEVELOPMENT_CONTROLS_RECALIBRATED_DESIGN_READY; DEPLOYMENT_ENGINEERING_CAN_PROCEED_IN_PARALLEL; DEPLOYMENT_ENGINEERING_REFERENCE_HANDOFF_EXPORTED; FLOAT_EXPORT_NUMERICAL_CONTRACT_RESOLVED; INT8_CALIBRATION_HANDOFF_EXPORTED; SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`

Latest milestone status: `SAND_FACTOR_CONDITIONED_ORDINARY_SUPPORT_FAILURE_REVIEW_COMPLETE; ORDINARY_SUPPORT_PHYSICAL_RECALIBRATION_READY; SAND_FACTOR_CONDITIONED_DEVELOPMENT_CONTROLS_RECALIBRATED_DESIGN_READY`.

Latest model-blind review found that all four observation-invalid ordinary-Support records established Support and then physically fell after only 563–990 ms; this was not a nine-second horizon or label defect. Saved evidence (48/48 stable controls) supports a source-speed correction: Concrete/.30 becomes right-only, unstable low-start joint corners are excluded, and Support semantics remain unchanged. A fresh 198-run design is frozen with Sand and delayed-Support logic preserved, but it has not been generated. The only recommended next milestone is `SAND_FACTOR_CONDITIONED_DEVELOPMENT_CONTROLS_RECALIBRATED_GENERATION`; details are in [`reports/20260903_sand_factor_conditioned_ordinary_support_failure_review.md`](reports/20260903_sand_factor_conditioned_ordinary_support_failure_review.md).

Exact frozen V2 engineering reference handoff는 [`artifacts/releases/model_v2_anchor_refined_gru20_20260902`](artifacts/releases/model_v2_anchor_refined_gru20_20260902)에 있다. 이 bundle은 세 checkpoint, normalizer, runtime contract, scientific verdict provenance와 layered golden vector를 고정한다. 이는 `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`이며 release model, real-robot support 또는 scientific support를 의미하지 않는다.

```bash
python scripts/fastreflex.py export
```

Export는 기존 bundle을 덮어쓰지 않으며 frozen source checksum이 하나라도 다르면 실패한다. Golden vector는 비보호 `V2_VALIDATION` 구간을 parity 용도로만 사용하고 Generalization HOLDOUT을 열지 않는다.

Deployment Float reference는 실제 1 kHz 호출과 같은 독립 batch-1 `[1,20,80]` 실행이다. 기존 batch-121 M1 golden은 byte-identical historical evidence로 보존하고, 새 batch-1 golden과 layer별 numerical contract를 bundle에 추가했다. Research batch-size sweep에서 faithful Float logits는 최대 `2.980232e-6` 변했지만 모든 threshold/persistence/decision은 exact였다. 상세 근거는 [`reports/20260903_float_numerical_contract_resolution.md`](reports/20260903_float_numerical_contract_resolution.md)에 있다.

M3 calibration handoff는 동일 bundle의 `calibration_manifest.json`과 `calibration_inputs/int8_representative.npz`다. Exact effective TRAIN 442개 run에서 모델 출력과 quantization 결과를 보지 않고 고른 2,597개 causal window이며 Generalization HOLDOUT과 평가 split은 열지 않았다. 상세 provenance는 [`reports/20260903_int8_calibration_handoff.md`](reports/20260903_int8_calibration_handoff.md)에 있다.

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

Frozen extraction 설계로 별도 3-seed GRU20 candidate를 학습했다. V1과 baseline data-only V2는 그대로 보존되며 delayed Support는 baseline 3/6→6/6, Marble은 0/3→3/3으로 개선됐다. 그러나 confirmed no-hazard specificity가 26/26→23/26, speed-Sand benign이 12/12→9/12로 회귀해 intervention verdict는 `V2_EXTRACTION_REBALANCE_NOT_EFFECTIVE`, internal verdict는 계속 `MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED`다. 새 candidate로 Generalization VALIDATION은 평가하지 않았고 Generalization HOLDOUT은 guard count 0으로 sealed다. 상세 결과는 [`reports/20260901_model_v2_extraction_rebalanced_training.md`](reports/20260901_model_v2_extraction_rebalanced_training.md)에 있다.

후속 read-only regression audit는 rebalanced candidate가 delayed Support를 고쳤지만 speed-Sand specificity를 회귀시켰고, 가장 가능성 높은 overlap을 dense `Support+[0..4]` anchor로 국소화했다. TRAIN-only anchor-refinement design은 I1 5개와 midpoint 5개를 보존하고 `L=I1+floor(3*(Support-I1)/4)` 한 개를 추가하는 11-endpoint rule을 freeze했으며 verdict는 `MODEL_V2_DELAYED_SUPPORT_ANCHOR_REFINEMENT_DESIGN_READY`다. 재학습은 수행하지 않았고 다음 milestone은 `MODEL_V2_ANCHOR_REFINED_TRAINING`이다. 새 candidate의 V2_VALIDATION 및 Generalization VALIDATION inference는 없었고 Generalization HOLDOUT guard count는 0이다. 상세 결과는 [`reports/20260901_model_v2_delayed_support_anchor_refinement_design.md`](reports/20260901_model_v2_delayed_support_anchor_refinement_design.md)에 있다.

Frozen late-interior anchor로 별도 3-seed GRU20 candidate를 학습했다. V1, baseline V2, rebalanced V2는 그대로 보존됐고 delayed Support는 6/6(Marble 3/3)을 유지하면서 speed-Sand benign과 confirmed specificity가 각각 12/12와 26/26으로 회복됐다. Read-only readiness review에서도 원래 Slip primary gate는 30/35로 실패하고 `MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED`가 그대로 유지됐다. 다만 남은 다섯 실패가 모두 frozen future-Slip precursor 내부의 sustained early response이고 genuine miss는 0이므로, anchor-refined V2를 변경 없이 `READY_FOR_EXTERNAL_DEVELOPMENT_EVALUATION`으로 승격했다. 이 readiness 결정은 당시 평가 자격일 뿐 external generalization이나 final freeze를 뜻하지 않았다. 상세 결과는 [`reports/20260902_model_v2_candidate_readiness_review.md`](reports/20260902_model_v2_candidate_readiness_review.md)에 있다.

Exact promoted V2를 frozen 36-run Generalization VALIDATION에 한 번 평가했다. Model V1의 Hazard 13/26, Slip 7/12, Support 6/14, primary specificity 5/10에서 V2는 각각 25/26, 11/12, 14/14, 10/10으로 개선됐고 Ice-benign도 3/4→4/4, premature도 7/26→1/26으로 개선됐다. 원래 Slip gate 95%는 91.67%로 실패하므로 primary verdict는 `GENERALIZATION_PRIMARY_GATES_FAIL`이다. 단 하나의 primary 실패는 이미 frozen된 future-Slip Ice precursor 안의 sustained early response이고 genuine detection failure는 0이어서 development verdict는 `GENERALIZATION_DEVELOPMENT_SUPPORTED_WITH_ICE_TIMING_TENSION`이다. Candidate retune/retraining은 없었고 Generalization HOLDOUT은 guard count 0으로 sealed이며 final Model V2/HOLDOUT support는 아직 없다. 다음 권고 milestone은 별도 `MODEL_V2_FINAL_CANDIDATE_FREEZE_AND_HOLDOUT_READINESS_REVIEW`다. 상세 결과는 [`reports/20260902_model_v2_generalization_development_evaluation.md`](reports/20260902_model_v2_generalization_development_evaluation.md)에 있다.

최종 readiness review는 같은 anchor-refined V2 `model_v2_anchor_refined_gru20_20260902`를 변경 없이 `FINAL_GENERALIZATION_CANDIDATE`로 freeze했고, 이어서 frozen 36-run Generalization HOLDOUT을 단 한 번 열었다. 같은 payload pass에서 V1/V2/Terrain을 평가한 결과 V1→V2는 Hazard 14/28→25/28, Slip 8/14→11/14, Support 6/14→14/14, premature 9/28→2/28로 개선됐지만 primary specificity는 5/8→5/8로 개선되지 않았다. V2는 overall Hazard, Slip, specificity gate를 실패했고, 두 supported Ice timing conflict 외에 0.30 m/s Ice Slip late detection 1건과 Sand benign false alert 3건이 있어 final verdict는 `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`다. Guard는 영구적으로 1이며 이 HOLDOUT은 다시 열거나 tuning/training evidence로 사용할 수 없다. 상세 결과는 [`reports/20260902_model_v2_generalization_holdout_one_shot_evaluation.md`](reports/20260902_model_v2_generalization_holdout_one_shot_evaluation.md)에 있다.

후속 failure interpretation은 HOLDOUT payload를 다시 읽지 않고 저장된 결과와 development-only evidence를 분리해 분석했다. Support branch는 14/14로 `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`지만 Slip은 두 known Ice timing conflict와 genuine +42 ms late detection 때문에, benign rejection은 Sand false alert 3건 때문에 각각 generalization 미지원이다. Sand는 exact fresh geometry와 speed/source interaction에서 실패했고 Generalization VALIDATION부터 0.99/5 ms 경계 바로 아래의 낮은 margin을 보였다. 따라서 historical final verdict는 유지하고 interpretation verdict는 `MODEL_V2_HOLDOUT_FAILURE_INTERPRETATION_ACTIONABLE`, 다음 단일 milestone은 fresh `SAND_BENIGN_GENERALIZATION_STUDY_DESIGN`이다. Consumed HOLDOUT은 향후 재학습·tuning·fresh evaluation에 영구 사용 금지다. 상세 결과는 [`reports/20260902_model_v2_holdout_failure_interpretation.md`](reports/20260902_model_v2_holdout_failure_interpretation.md)에 있다.

Fresh Sand-benign development study는 generation 전에 `SAND_BENIGN_GENERALIZATION_STUDY_DESIGN_READY`로 freeze했다. 176-run matrix는 Sand 144개를 Concrete/Marble, 0.20/0.25/0.30 m/s, geometry, indirect contact-phase, left/right topology, LOW/MEDIUM/NEAR_HAZARD intent에 대칭 배정하고 Support control 32개를 별도 포함한다. `STUDY_DISCOVERY`/`STUDY_CONFIRMATION`은 88/88로 고정했고, physical benign yield와 Pelvis/FSR observability 및 domain/model decision rule을 사전 정의했다. Model V2 final HOLDOUT은 계속 NOT_SUPPORTED이고, Support 성공은 보존하며 Slip strict timing은 secondary limitation이다. Architecture/sensor 변경과 dataset generation은 시작하지 않았고 consumed HOLDOUT은 tuning에 영구 사용 불가다. 상세 설계는 [`reports/20260902_sand_benign_generalization_study_design.md`](reports/20260902_sand_benign_generalization_study_design.md)에 있다.

동결된 fresh Sand study 176건을 backfill 없이 한 번 생성했지만 strict-benign yield는 Discovery 19/72, Confirmation 22/72이고 ordinary Support는 5/24만 valid했다. Valid Sand의 realized phase도 `DOUBLE_SUPPORT` 하나로 축소되어 generation verdict는 `SAND_BENIGN_GENERALIZATION_STUDY_GENERATION_PHYSICAL_YIELD_INSUFFICIENT`다. Model training/replay와 H1/H2/H3 판단은 수행하지 않았고, historical final V2 verdict는 계속 `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`, consumed HOLDOUT guard는 영구 1이다. `STUDY_CONFIRMATION`은 생성됐지만 model/80D/observability 분석 없이 `SEALED_FOR_STUDY_CONFIRMATION`으로 유지한다. 다음 최소 milestone은 `SAND_BENIGN_GENERALIZATION_STUDY_CALIBRATION_REVIEW`이며 상세 결과는 [`reports/20260902_sand_benign_generalization_study_generation.md`](reports/20260902_sand_benign_generalization_study_generation.md)에 있다.

후속 physical calibration은 model-blind pilot 3개, 총 96건으로 종료했다. First-contact sample의 DOUBLE collapse 대신 20 ms pre-contact exact phase에서 LEFT/RIGHT single-support를 확인했고, 9 s mild Sand는 11/12 strict, separate ordinary Support control은 11/12였다. Moderate는 최대 calibrated boundary-adjacent benign tier로 유지하지만 severe는 넓고 짧은 exposure를 합쳐 strict 0/36이어서 benign domain에서 제외했다. 이를 반영한 wholly fresh 176-run redesign은 88/88 Discovery/Confirmation, historical/split signature overlap 0, fail-closed physical gates로 `SAND_BENIGN_GENERALIZATION_STUDY_REDESIGN_READY`다. Model replay/training은 0이고 old HOLDOUT guard는 영구 1이며, 다음 milestone은 `SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION`이다. 상세 결과는 [`reports/20260902_sand_benign_generalization_study_redesign.md`](reports/20260902_sand_benign_generalization_study_redesign.md)에 있다.

Redesigned corpus 176건은 backfill/replacement 없이 모두 생성됐고 objective-valid 153, strict Sand 116, ordinary/delayed Support 24/8, actual Slip 5를 얻었다. Discovery yield와 모든 diversity/Support/integrity gate는 통과했지만 Confirmation mild `35/48`, Concrete/.25 strict `7/12`, total strict `52/72`의 세 frozen yield check가 실패해 verdict는 `SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION_YIELD_INSUFFICIENT`다. Model replay/training은 여전히 0이고 `REDESIGNED_CONFIRMATION`은 sealed, consumed HOLDOUT guard는 영구 1이다. Historical final V2는 `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`, Support branch는 `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`, whole-simulation status는 `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`로 유지한다. 다음 최소 milestone은 `SAND_BENIGN_REDESIGNED_DOMAIN_FAILURE_REVIEW`이며 상세 결과는 [`reports/20260902_sand_benign_generalization_study_redesigned_generation.md`](reports/20260902_sand_benign_generalization_study_redesigned_generation.md)에 있다.

Redesigned failure review는 insufficient-follow-up 20건이 9초 horizon에 도달한 stable late-entry가 아니라 모두 1.866–7.269초의 post-target `nonfoot_surface_contact` fall censor임을 확인했다. Confirmation target-entry p95는 Discovery와 같은 1,810 ms였지만 Sand width median은 36 mm 길었고, mild invalid width median은 .826 m여서 verdict는 `SAND_BENIGN_REDESIGNED_DOMAIN_FAILURE_REVIEW_ACTIONABLE`, feasibility는 `MILD_DOMAIN_RECALIBRATION_REQUIRED`다. 9초 observation, moderate, Support, phase, topology, label과 frozen gates는 유지하며 full study design/generation과 model replay는 시작하지 않았다. 다음 최소 milestone은 `SAND_BENIGN_MILD_DOMAIN_CALIBRATION_REDESIGN`이고 상세 결과는 [`reports/20260902_sand_benign_redesigned_domain_failure_review.md`](reports/20260902_sand_benign_redesigned_domain_failure_review.md)에 있다.

Fresh model-blind mild calibration은 사전 동결된 A/B/C pilot 72건으로 종료했고 final verification은 24/24 strict-benign, source-speed cell별 4/4였다. Residual failure는 width 단독이 아니라 joint start/width/exit와 topology interaction으로 국소화됐으며, Concrete/.25 right만 세 geometry에서 0/6이어서 common left + five-cell common right + Concrete/.25 left-only의 최소 rule을 동결했다. 다음 fresh 176-run design은 88/88 Discovery/Confirmation과 기존 moderate/Support/9초/label/gate를 유지하고 exact historical/split reuse 0이다. Model replay/training은 0, consumed HOLDOUT guard는 1이고 historical `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`와 `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`는 그대로다. 다음 milestone은 `SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_GENERATION`이며 상세 calibration과 frozen design은 [`reports/20260903_sand_benign_mild_domain_calibration_redesign.md`](reports/20260903_sand_benign_mild_domain_calibration_redesign.md), [`reports/20260903_sand_benign_generalization_study_mild_recalibrated_redesign.md`](reports/20260903_sand_benign_generalization_study_mild_recalibrated_redesign.md)에 있다.

Mild-recalibrated corpus 176건은 backfill/replacement/rerun 없이 한 번 생성됐고 objective-valid 169, strict Sand 134, Support 32, actual Slip 3, invalid 7을 얻었다. Broad mild는 양 split `48/48`, 모든 source-speed cell `8/8` strict였고 70개 frozen physical-generation gate가 모두 통과해 verdict는 `SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_GENERATION_READY`다. Model replay/training은 0이며 `MILD_RECALIBRATED_CONFIRMATION`은 model/80D/observability/hypothesis analysis 없이 sealed다. Historical final V2는 계속 `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`, Support branch는 `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`, whole-simulation status는 `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`이고 consumed HOLDOUT guard는 영구 1이다. 다음 milestone은 `SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_DISCOVERY_ANALYSIS`이며 상세 결과는 [`reports/20260903_sand_benign_generalization_study_mild_recalibrated_generation.md`](reports/20260903_sand_benign_generalization_study_mild_recalibrated_generation.md)에 있다.

Discovery 88건에 사전 동결된 run-balanced Pelvis/FSR/oracle 분석과 exact final V2 replay를 한 번 적용했다. Strict Sand는 67/69 specific이지만 24/69가 frozen adverse-margin 조건을 만족했고, 24건 모두 transition-left/right-single-precontact에 정량적으로 국소화됐다. Pelvis `[20,80]`은 reasonable-separation 4/4를 통과했고 realizable FSR increment는 1/4로 material하지 않았으며 Support는 16/16이었다. 따라서 Discovery hypothesis는 `DOMAIN_DIVERSITY_GAP_SUPPORTED`, validity는 `SAND_BENIGN_GENERALIZATION_STUDY_DISCOVERY_ANALYSIS_VALID`다. Training/tuning은 없고 Confirmation 88건은 계속 sealed이므로 아직 H1이 confirmed됐다고 주장하지 않는다. 다음 milestone은 `SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_CONFIRMATION_ANALYSIS`이며 상세 결과는 [`reports/20260903_sand_benign_generalization_study_mild_recalibrated_discovery_analysis.md`](reports/20260903_sand_benign_generalization_study_mild_recalibrated_discovery_analysis.md)에 있다.

동결된 H1과 Discovery scaler를 그대로 사용해 Confirmation 88건을 guard `0→1`로 한 번 열고 exact final V2를 한 번 replay했다. Strict Sand는 61/65 specific, adverse는 29/65였고 transition-left/right-single-precontact 방향, Support 16/16, non-material FSR 1/4는 재현됐다. 그러나 Pelvis window centroid separation이 `.208030 < .75`여서 reasonable-separation은 3/4에 그쳤고, frozen all-required H1은 실패했다. 따라서 validity는 `SAND_BENIGN_GENERALIZATION_STUDY_CONFIRMATION_ANALYSIS_VALID`, scientific verdict는 `DOMAIN_DIVERSITY_GAP_NOT_CONFIRMED`다. Training/tuning이나 H2/H3 대체 선택은 없었고 historical `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`, `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`, `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`는 그대로다. 다음 milestone은 `SAND_GENERALIZATION_HYPOTHESIS_REVIEW`이며 상세 결과는 [`reports/20260903_sand_benign_generalization_study_mild_recalibrated_confirmation_analysis.md`](reports/20260903_sand_benign_generalization_study_mild_recalibrated_confirmation_analysis.md)에 있다.

후속 read-only hypothesis review는 centroid distance가 오히려 73.00% 증가한 반면 window within-group RMS가 7.4073배 증가해 global separation이 하락했고, current-80D와 local 1NN/5NN은 안정적임을 확인했다. Frozen scaler와 preprocessing은 정확했고, topology/right-single direction은 약해졌지만 재현됐으며 source/speed FP identity는 이동했다. 따라서 broad Pelvis observability failure나 FSR fusion을 승격하지 않고 future-study hypothesis로 `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS`를 MODERATE confidence로 선택했다. Review verdict는 `SAND_GENERALIZATION_HYPOTHESIS_REVIEW_ACTIONABLE`, 다음 과학 milestone은 `SAND_FACTOR_CONDITIONED_TRAINING_DOMAIN_DESIGN`이다. 별도로 exact V2를 final 승인 없이 engineering reference로 쓰는 `MODEL_V2_DEPLOYMENT_ENGINEERING_REFERENCE_HANDOFF`는 병렬 진행 가능하지만 이 milestone에서는 deployment repo를 수정하지 않았다. 상세 결과는 [`reports/20260903_sand_generalization_hypothesis_review.md`](reports/20260903_sand_generalization_hypothesis_review.md)에 있다.

Factor-conditioned data intervention은 fresh `FACTOR_TRAIN` 108개와 `FACTOR_VALIDATION` 54개를 model-blind하게 한 번 생성했지만, objective-eligible yield가 81/162로 frozen minimum 140에 미달했다. Strict Sand는 TRAIN 30/72와 VALIDATION 12/36, ordinary/delayed Support는 26/36과 13/18이었으며 42건은 post-target observation 부족, 30건은 pretarget fall이었다. Integrity·phase/topology·physical-signature diversity gate는 통과했지만 총 21개 yield gate가 실패해 verdict는 `FACTOR_CONDITIONED_DATA_INTERVENTION_INVALID`다. Training, checkpoint, HNM, fresh model replay, V2_VALIDATION replay는 모두 0이며 historical final V2는 계속 `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`다. 다음 단일 milestone은 `DATA_INTERVENTION_FAILURE_AUDIT`이고 상세 결과는 [`reports/20260903_sand_factor_conditioned_data_intervention.md`](reports/20260903_sand_factor_conditioned_data_intervention.md)에 있다.

후속 physical-only audit는 실패를 width 단독이 아닌 family별 viable envelope 밖의 joint start/width/exit와 source-speed/topology-conditioned contact sequence 불안정으로 국소화했다. 30건은 target 이전 fall, 42건은 모두 실제 target 이후 fall censor였으므로 label 또는 observation horizon 변경은 정당화되지 않는다. 사전 동결된 model-blind pilot 2개는 허용된 64건 중 32건만 사용해 strict Sand 29/32, pretarget fall·Slip·dual Hazard 0을 확인했다. 이를 바탕으로 fresh 198-run `sand_factor_conditioned_development_recalibrated_20260903` 설계와 물리 gate를 동결했으며 전체 corpus는 아직 생성하지 않았다. Verdict는 `FACTOR_CONDITIONED_PHYSICAL_DOMAIN_REDESIGN_READY`, 다음 단일 milestone은 `SAND_FACTOR_CONDITIONED_DEVELOPMENT_RECALIBRATED_GENERATION`이다. 상세 audit은 [`reports/20260903_data_intervention_failure_audit.md`](reports/20260903_data_intervention_failure_audit.md)에 있다.

동결된 recalibrated matrix 198건은 replacement/backfill/rerun 없이 정확히 한 번 생성됐다. Recalibrated Sand는 mild 108/108, moderate 35/36 strict였고 모든 Sand·manifold·source-speed gate가 통과했지만, unchanged delayed Support가 TRAIN 9/12와 VALIDATION 3/6으로 최소 10/5에 미달했다. 전체 ledger는 53/55 PASS이므로 corpus는 failed physical-development evidence로 freeze되고 FACTOR_VALIDATION은 `SEALED_FAILED_PHYSICAL_EVIDENCE` 상태다. Training, HNM, normalizer fit, model inference는 모두 0이며 verdict는 `SAND_FACTOR_CONDITIONED_DEVELOPMENT_RECALIBRATED_GENERATION_INSUFFICIENT`다. 다음 최소 milestone은 `SAND_FACTOR_CONDITIONED_DELAYED_SUPPORT_PHYSICAL_REVIEW`이고 상세 결과는 [`reports/20260903_sand_factor_conditioned_development_recalibrated_generation.md`](reports/20260903_sand_factor_conditioned_development_recalibrated_generation.md)에 있다.

후속 delayed-Support physical review는 최신 18건의 saved evidence에서 실패를 `.20 m/s` 및 late `.30 m/s` geometry/contact-sequence가 real Slip·fall 영역에 접근한 `MULTIFACTOR_PHYSICAL_INSTABILITY`로 국소화했다. 사전 동결한 model-blind 24-run pilot은 all six source-speed cells에서 Support 23/24, dual/Slip 0을 얻어 unchanged Support semantics의 안정 strip을 확립했다. 이를 반영한 wholly fresh 198-run `sand_factor_conditioned_development_support_recalibrated_20260903` 설계를 freeze했으며 corpus generation과 training은 시작하지 않았다. Verdict는 `DELAYED_SUPPORT_PHYSICAL_RECALIBRATION_READY`, 다음 milestone은 `SAND_FACTOR_CONDITIONED_DEVELOPMENT_SUPPORT_RECALIBRATED_GENERATION`이다. 상세 결과는 [`reports/20260903_sand_factor_conditioned_delayed_support_physical_review.md`](reports/20260903_sand_factor_conditioned_delayed_support_physical_review.md)에 있다.

Support-recalibrated frozen matrix 198건은 replacement/backfill/rerun 없이 정확히 한 번 생성됐다. Delayed Support는 TRAIN 11/12와 VALIDATION 6/6으로 새 최소 gate를 모두 통과했고 Sand·factor·source-speed gate도 모두 통과했지만, ordinary Support TRAIN이 20/24로 최소 22에 미달했다. 전체 ledger는 57/58 PASS이므로 corpus는 failed physical evidence로 freeze했고 FACTOR_VALIDATION은 `SEALED_FAILED_PHYSICAL_EVIDENCE`다. Model/training/HNM/HOLDOUT access는 모두 0이며 다음 최소 milestone은 `SAND_FACTOR_CONDITIONED_ORDINARY_SUPPORT_FAILURE_REVIEW`다. 상세 결과는 [`reports/20260903_sand_factor_conditioned_development_support_recalibrated_generation.md`](reports/20260903_sand_factor_conditioned_development_support_recalibrated_generation.md)에 있다.

## Canonical source flow

```text
Hazard:
simulation/g1.py -> dataset/hazard.py -> features.py
                 -> training/hazard.py -> models/baselines.py
                 -> evaluation/{hazard,generalization,readiness,holdout}.py

Terrain:
simulation/{g1,sensors,terrain}.py -> dataset/terrain.py
                                  -> training/terrain.py
                                  -> evaluation/terrain.py -> advisory cause
```

주요 책임은 다음과 같다.

- `simulation/`: G1 physics, runtime sensors, physical Slip/Support/I1 관련 diagnostics
- `dataset/hazard.py`: unified run, physical labels, split, manifest integrity, HOLDOUT guard
- `dataset/sand_study.py`: frozen Sand-study generation, objective physical labels, diversity audit, Confirmation sealing
- `dataset/sand_calibration.py`: model-blind Sand calibration, censor-aware labels, redesigned matrix integrity
- `dataset/sand_mild_calibration.py`: mild physical ledger, recalibrated domain expansion, generation, audit and Confirmation sealing
- `dataset/sand_factor_conditioned.py`: fresh factor-conditioned split design, model-blind generation, eligibility gates, and dataset freeze
- `features.py`: selected Pelvis IMU6 → causal 80D contract
- `training/hazard.py`: TRAIN-only normalization/windowing와 3-round HNM
- `evaluation/hazard.py`: continuous replay, 0.99/5 ms decision, metrics, frozen-candidate verification
- `evaluation/sand.py`: sealed-split-aware Sand Discovery observability, factor localization, and exact frozen-V2 replay
- `evaluation/sand_factor_conditioned.py`: frozen fresh-development comparison metrics, available only after candidate freeze
- `evaluation/holdout.py`: Generalization HOLDOUT의 consumed one-shot result verification과 second-open refusal
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
- [`reports/20260901_model_v2_extraction_rebalanced_training.md`](reports/20260901_model_v2_extraction_rebalanced_training.md)
- [`reports/20260901_model_v2_rebalance_regression_audit.md`](reports/20260901_model_v2_rebalance_regression_audit.md)
- [`reports/20260902_model_v2_anchor_refined_training.md`](reports/20260902_model_v2_anchor_refined_training.md)
- [`reports/20260902_model_v2_candidate_readiness_review.md`](reports/20260902_model_v2_candidate_readiness_review.md)
- [`reports/20260902_model_v2_generalization_development_evaluation.md`](reports/20260902_model_v2_generalization_development_evaluation.md)
- [`reports/20260902_model_v2_final_candidate_holdout_readiness_review.md`](reports/20260902_model_v2_final_candidate_holdout_readiness_review.md)
- [`reports/20260902_model_v2_generalization_holdout_one_shot_evaluation.md`](reports/20260902_model_v2_generalization_holdout_one_shot_evaluation.md)
- [`reports/20260902_model_v2_holdout_failure_interpretation.md`](reports/20260902_model_v2_holdout_failure_interpretation.md)
- [`reports/20260903_sand_benign_generalization_study_mild_recalibrated_discovery_analysis.md`](reports/20260903_sand_benign_generalization_study_mild_recalibrated_discovery_analysis.md)
- [`reports/20260903_sand_benign_generalization_study_mild_recalibrated_confirmation_analysis.md`](reports/20260903_sand_benign_generalization_study_mild_recalibrated_confirmation_analysis.md)
- [`reports/20260903_sand_generalization_hypothesis_review.md`](reports/20260903_sand_generalization_hypothesis_review.md)
- [`reports/20260903_sand_factor_conditioned_data_intervention.md`](reports/20260903_sand_factor_conditioned_data_intervention.md)
- [`reports/20260903_data_intervention_failure_audit.md`](reports/20260903_data_intervention_failure_audit.md)
- [`reports/20260903_sand_factor_conditioned_development_recalibrated_generation.md`](reports/20260903_sand_factor_conditioned_development_recalibrated_generation.md)

Current architecture는 [`docs/architecture.md`](docs/architecture.md), dataset contract는 [`docs/dataset.md`](docs/dataset.md), 검증 규칙은 [`docs/experiment_protocol.md`](docs/experiment_protocol.md)에 있다.

## Verification

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall src scripts
```

기본 `pytest` command는 project environment의 pytest/plugin version이 일치할 때 사용한다.
