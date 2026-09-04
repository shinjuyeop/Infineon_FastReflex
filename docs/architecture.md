# Architecture

## 1. Current runtime architecture

Current supported system has two independent producers.

```text
                         ┌─ Pelvis IMU6 ─> causal 80D ─> GRU20
Unitree G1 simulation ───┤                              └─> REFLEX_REQUIRED
                         │
                         └─ touchdown FSR4 ─> 50 ms MLP ─> held Terrain
                                                           │
REFLEX_REQUIRED ────────────────────────────────────────────┴─> cause refinement
```

Hazard is the control-facing decision. Terrain is advisory-only. Missing, late, stale, wrong, or unknown Terrain cannot alter Hazard probability, GRU state, 5 ms persistence, or `REFLEX_REQUIRED`.

Current source flow is:

```text
simulation/g1.py
  -> dataset/hazard.py
  -> features.py
  -> training/hazard.py
  -> models/baselines.py
  -> evaluation/{hazard,generalization,readiness,sand}.py

frozen Float engineering reference
  -> training/qat.py
  -> deployment-only engineering derivative

simulation/{g1,sensors,terrain}.py
  -> dataset/terrain.py
  -> evaluation/terrain.py
  -> advisory cause

stored TRAIN/VALIDATION run specification
  -> visualization.py
  -> deterministic simulation/g1.py parity gate
  -> frozen evaluation/{hazard,terrain}.py replay
  -> isolated MuJoCo viewer overlay
```

## 2. Hazard preprocessing and model contract

Runtime input is Pelvis IMU6 at 1 kHz in this order:

```text
accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
```

`features.py` derives ten bases:

```text
raw IMU6
accel_norm, gyro_norm
horizontal_accel_norm, horizontal_gyro_norm
```

Each base uses eight representations in block-major order:

```text
base
delta_1ms
delta_5ms
delta_10ms
causal_mean_5ms
causal_mean_10ms
causal_variance_5ms
causal_variance_10ms
```

The result is float32 `[samples,80]`. Delta prefixes are zero. Rolling windows are trailing and include the current endpoint. No centered window or future sample is permitted. The selected schema hash is `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`.

The model is the shared `GRUBaseline` with input 80, hidden size 32, one unidirectional layer and two outputs. A causal input window is exactly `[20,80]`. The three-seed ensemble averages `HAZARD_REFLEX_REQUIRED` softmax probability. The current operating point is inclusive threshold 0.99 and five consecutive 1 ms samples. Parameter count is 11,010.

The supported runtime result remains the frozen Model V1 artifact. Separately, the exact anchor-refined Model V2 candidate `model_v2_anchor_refined_gru20_20260902` was frozen as `final_generalization_candidate` and evaluated in the single authorized Generalization HOLDOUT opening. It improved V1 substantially and corrected all 14 Support cases, but failed the frozen overall Hazard, Slip, and specificity gates. Its final verdict is `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`; it is not a simulation-generalization-supported release candidate and does not replace Model V1's supported result. The provisional architecture is retained only as the immutable object for failure interpretation, not as a real-robot, deployment, or final-sensor claim.

## 3. Terrain contract

Terrain uses one clean touchdown event at a time.

```text
exact terrain contact rising edge (label/scheduler only)
  -> touchdown foot FSR4 samples [t, t+50)
  -> frozen TRAIN normalization
  -> three-seed MLP mean probability
  -> CONCRETE / MARBLE / ICE / SAND
  -> held state from t+50 until the next valid update
```

Exact geom identity schedules and labels clean events but is not a model input. The selected deployment scheme is LEFT_ONLY research candidate; BILATERAL_SHARED remains documented evidence rather than a replacement candidate.

Cause refinement is deliberately small:

```text
REFLEX_REQUIRED + ICE  -> SLIP_RISK
REFLEX_REQUIRED + SAND -> SUPPORT_RISK
REFLEX_REQUIRED + other/unknown -> GENERIC_DISTURBANCE
```

The system never waits for Terrain.

## 4. Physical label boundary

Hazard labels use physical reference clocks only:

```text
ESTABLISHED_SLIP OR ESTABLISHED_SUPPORT
-> HAZARD_REFLEX_REQUIRED
```

Established Slip remains the frozen 50 mm/3 ms any-foot oracle. Established Support remains 10 mm support-surface spread with 20 ms causal persistence and contact-episode reset. I1 is the frozen causal positive loaded-foot spread-derivative reference used to define the acceptable Support alert region.

Primary no-hazard means:

```text
no established Slip
AND no I1 precursor
AND no established Support
```

These references are privileged scoring/label fields. They are excluded from runtime features. Fall/recovery and experimental design role never define the runtime Hazard label. Terrain identity/output never enters the Hazard tensor or label.

## 5. Dataset, training, and evaluation

`dataset/hazard.py` owns `HazardRun`, the source-balanced frozen split, physical signature, manifest/run SHA validation and one-shot `HoldoutGuard`. Runtime arrays and privileged diagnostics remain separately named even when stored in the same NPZ.

`training/hazard.py` owns positive windows, true negative regions, TRAIN-only normalizer fitting and HNM. The fixed HNM contract is three rounds after Round 0, 1 ms replay, K=12 per run, 30 ms minimum spacing, and no negative after I1 becomes active. V2 additionally masks future-Slip and censored Ice precursor endpoints while allowing explicitly observed benign releases outside every established-positive region. Supplied runs must declare `split: train`; otherwise training construction fails.

`evaluation/hazard.py` owns continuous replay, probability aggregation, threshold/persistence, physical run metrics and read-only verification of selected freezes. `evaluation/generalization.py` owns the development comparison and fail-closed Generalization split loader. `evaluation/readiness.py` owns metadata-only final-candidate, protected-data, and frozen-contract checks. `evaluation/holdout.py` owns the consumed one-shot summary verifier and permanent second-open refusal. Scientific HOLDOUT payloads are never reopened during routine verification.

`dataset/terrain.py` and `evaluation/terrain.py` own the current clean-event and
runtime-inference responsibilities. The Terrain candidate is frozen and the current
CLI supports read-only verification, not implicit retraining. Historical Terrain
training code is recoverable from its recorded source commit. No historical
Terrain-gated Hazard fusion module remains in the current dependency graph.

## 6. Simulation boundary

`simulation/g1.py` is the single G1 loop with 0.5 ms physics and 1 kHz sampling. Runtime signals are Pelvis IMU6 and optional FSR8/Foot IMU12 observers. Physical diagnostics are stored separately.

- `simulation/sensors.py`: FSR and Foot IMU observation
- `simulation/terrain.py`: Concrete/Marble/Ice/Sand and support mechanics
- `simulation/hazards.py`: contact/touchdown, Slip, support spread/loss and I1 inputs
- `simulation/stability.py`: shared gait-phase naming plus the exact frozen
  stability implementation referenced by existing dataset provenance; it is not a
  current runtime Hazard detector

The ordinary simulation viewer copies physics state into a render-only model. Viewer input cannot feed back into the canonical simulation.

`visualization.py` owns read-only run resolution, HOLDOUT rejection, exact stored/re-simulated parity, frozen inference alignment, snapshot playback and HUD formatting. `run_simulation(..., capture_render_trace=True)` optionally captures one memory-only full MuJoCo `mjSTATE_INTEGRATION` snapshot per 1 kHz sample; the default is `False`, the stored NPZ contract is unchanged, and Sand support degrees of freedom are included. The interactive viewer opens only after timestamp, IMU6, FSR8 and physical event-clock parity passes. It restores selected immutable snapshots with `mj_setState`, calls `mj_forward`, and renders with `viewer.sync`; it never calls `mj_step`. Playback speed, pause, backward/forward seek, event jumps and overlay text therefore cannot enter controller, physics, sensor, feature or model tensors. Closing the viewer is a clean user exit because scientific parity was completed headlessly before it opened.

## 7. Research-to-Deployment boundary

This repository ends at reviewed Float research artifacts and their contracts. The exact frozen V2 engineering reference is exported at `artifacts/releases/model_v2_anchor_refined_gru20_20260902`. The canonical `export` command verifies every source checksum, copies only the three selected Round-3 checkpoints and normalizer, and derives layered golden outputs from one non-protected `V2_VALIDATION` slice. It never trains, selects, tunes, opens Generalization HOLDOUT, or overwrites an existing release.

`training/qat.py` is a separately predeclared deployment-only engineering path
over that frozen Float reference. It may use only the authorized TRAIN-derived
sources in `20260904_deployment_aware_qat.yaml`; it does not reopen scientific
validation/HOLDOUT evidence or change the immutable generalization verdict. Research
emulation does not establish TFLite full-INT8, Vela, target-runtime, real-robot, or
safety support.

The bundle labels the candidate `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL` and preserves `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED` plus `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`. Its canonical deployment Float execution is one independent `[1,20,80] float32` window per invocation with a new zero hidden state. The original batch-121 M1 golden remains historical evidence; a separate batch-1 golden and `float_numerical_contract.json` define layer-specific continuous tolerances and require exact threshold, persistence, and final decisions. Research also owns a deterministic M3 calibration handoff: five runtime-uniform endpoints from every exact effective-TRAIN run plus valid physical precursor, Slip and Support anchors, deduplicated into 2,597 windows. The calibration manifest records every run/file checksum and prohibits model-output or quantization-result selection; clipping policy, quantization, Vela, E84 firmware integration, HIL, target latency and Recovery belong to the deployment repository. The handoff does not imply a supported research, real-robot, production, or safety release.

## 8. Historical research summary

MoS/Stability, direct NORMAL/SLIP/SINK classification, event-centric detectors, Terrain-gated branches, Support fusion variants and their diagnostic tests are not current runtime dependencies. Their scientific results remain in [`../reports/`](../reports/) and their original source is recoverable from Git history. Dated experiment configs remain as provenance; the current CLI rejects them as historical instead of silently selecting another implementation.

## 9. Sand physical-calibration boundary

`dataset/sand_calibration.py` owns model-blind pilot expansion, pre-simulation freeze, censor-aware physical summaries, failed-study Discovery tabulation, and redesigned-matrix integrity checks. It does not perform Hazard/Terrain inference or feature extraction. The older `dataset/sand_study.py` remains unchanged as the exact implementation provenance of the failed frozen study.

`dataset/sand_mild_calibration.py` owns the later mild-only physical ledger, deterministic expansion/validation, one-pass generation, model-blind physical audit, dataset freeze, and Confirmation sealing for the recalibrated matrix. Its geometry check treats start, width, exit, and topology jointly; it rejects the predeclared Concrete/.25 right topology and fails closed on historical or cross-split contamination. It loads no model, and its Confirmation loader rejects sealed records before NPZ access.

`dataset/sand_factor_conditioned.py` owns the separate fresh intervention matrix, TRAIN/VALIDATION identity freeze, historical and cross-split contamination rejection, actual-physics eligibility, model-blind generation, and failed-attempt dataset freeze. Its redesign expander accepts the legacy shared profile families and explicit source-speed profile maps, allowing a pre-frozen physical correction without a new runner or a change to historical configs. `evaluation/sand_factor_conditioned.py` defines the comparison surface that would have been available only after candidate freeze. The physical-generation gate failed before training, so that evaluator has not consumed a fresh payload and no candidate entered the runtime architecture.

`evaluation/sand.py` owns the corresponding Discovery-only run-balanced Pelvis/FSR/oracle analysis, metadata localization, and exact frozen-V2 replay. It uses one model-independent anchor per run, verifies every frozen dataset/model hash before analysis, rejects Confirmation before payload access, and refuses a second Discovery replay after the result artifact exists. FSR and privileged oracle vectors remain diagnostics and never enter the Hazard runtime model.

`evaluation/sand_confirmation.py` owns the separate one-shot Confirmation authorization and H1-replication wrapper. It reconstructs the omitted Discovery-pooled scaler values before access and requires their mean/std hashes to match the frozen Discovery results, then atomically records guard `0 -> 1`, deserializes each of the 88 Confirmation payloads once, reuses the exact metric body, and replays only the frozen final V2 once. Saved artifacts are the only post-consumption analysis surface; the wrapper rejects any second run. The valid result did not confirm H1 because window centroid separation failed its frozen threshold, so it does not authorize a data, model, architecture, or sensor intervention.

The subsequent saved-artifact-only hypothesis review found that the centroid-distance numerator increased while the window within-group RMS denominator increased 7.4073x. Current-80D and local-neighborhood geometry remained stable, and predeclared factor subsets exposed nonuniform trajectory-space spread. The review therefore selects `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS` only as a future independent-study hypothesis; runtime architecture, features, sensors, and weights remain unchanged. Deployment preprocessing, export, compatibility, profiling, and parity plumbing may use the exact frozen V2 as a non-final engineering reference in the deployment repository, but final weights, INT8/HIL sign-off, sensor architecture, and release claims remain blocked on research approval.

The attempted factor-conditioned corpus produced only 81/162 actual-role-eligible runs and failed 21 predeclared yield gates. The protocol stopped with `FACTOR_CONDITIONED_DATA_INTERVENTION_INVALID`: optimizer steps, normalizer fits, checkpoint writes, HNM, and model replay are all zero. Consequently the deployed/reference tensor contract, model graph, weights, operating point, and scientific status are unchanged.

The subsequent metadata-only failure audit and 32-run model-blind physical calibration affect only `dataset/sand_factor_conditioned.py` and future scenario parameters. They freeze a fresh 198-run factor-conditioned design while leaving features, model graph, weights, threshold/persistence, sensors, labels, and Support mechanics unchanged.

That exact design was then generated once. The canonical generator now supports the recalibrated config, deterministic complete gate audit, failed-result dataset freeze, and a split-aware loader that rejects `FACTOR_VALIDATION` before NPZ access. The physical ledger passed 53/55 gates: all Sand/manifold gates passed, while both delayed-Support split yields failed. The stop creates no candidate and changes no runtime feature, model, threshold, persistence, sensor, or deployment contract.

The corrected calibration path distinguishes touchdown-sample contact state from the primary 20 ms pre-contact gait phase. For Support controls, a fully established and sufficiently observed event is not retroactively invalidated by a later fall. These corrections affect only new calibration/redesign artifacts; historical manifests and verdicts are not rewritten.

The final controls-recalibrated physical generation reuses that canonical path with no new labeler or simulator branch. Its exact 198-run matrix completed once and passed all 61 frozen gates: ordinary Support was 24/24 TRAIN and 12/12 VALIDATION, delayed Support was 12/12 and 6/6, and strict Sand was 96/96 and 46/48. The dataset is frozen as physical DEVELOPMENT evidence. Before model training, its loader rejected `FACTOR_VALIDATION` before payload access and only the separately authorized training milestone could consume `FACTOR_TRAIN`; model architecture, operating point, and runtime sensor contract remained unchanged.

## 10. Factor-conditioned model result

`training/hazard.py` owns the completed factor-conditioned data-only cycle through the same canonical GRU trainer. It verifies the controls-recalibrated corpus, frozen V2 normalizer, architecture, feature, and reference-model hashes before fitting. The cycle trains one 11,010-parameter GRU family at three frozen seeds, applies exactly three TRAIN-only hard-negative-mining rounds, and freezes all candidate checkpoints before a separate authorization changes the fresh validation seal from closed to consumed. The evaluator keeps strict Sand, ordinary and delayed Support, the one actual Slip, and metadata-only invalid records distinct; it also replays the historical V2 development validation only as the predeclared regression check.

The factor-conditioned candidate improved fresh strict-Sand specificity from 44/46 to 46/46 and reduced high-margin adverse Sand responses from 17/46 to 2/46. It simultaneously reduced fresh Support detection from 16/18 to 8/18 and historical V2 Support detection from 30/30 to 17/30. The frozen verdict is therefore `FACTOR_CONDITIONED_DATA_INTERVENTION_NOT_EFFECTIVE`, and `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS` remains not established. The existing anchor-refined V2 candidate remains the sole deployment reference; the trained factor-conditioned candidate is development evidence only and does not change the runtime architecture, normalizer, operating point, sensor contract, or release status.

## 11. Hazard-boundary audit and validation stop

`training/hazard.py` also owns the saved-checkpoint, TRAIN-only boundary-failure audit. It reconstructs the exact prior endpoint populations and loss mass, replays the reference and failed Round 0–3 checkpoints on a frozen run-disjoint diagnostic partition, and measures fixed 80D, recurrent-state, logit, head, and temporal geometry. The loader refuses already consumed `FACTOR_VALIDATION` before any NPZ access. The audit does not simulate, optimize, change model weights, or reopen a validation split.

The audit freezes `TRAINING_OBJECTIVE_SAMPLING_TENSION` as the primary cause. Both GRUs retained perfect linear and nearest-neighbor separation on their final hidden states, while the fixed operating head lost Support and the prior inverse-frequency objective left Slip with 59.8% of positive-class loss mass. Longer history, LSTM, new features, and sensor fusion are therefore not part of the selected response. The only frozen candidate hypothesis is a within-class domain-balanced loss change on the same Pelvis-IMU6, causal 80D, GRU32, `.99`/5 ms contract.

`dataset/sand_factor_conditioned.py` owns the independent `hazard_boundary_resolution_validation_20260904` expander, historical/near/cross-role contamination checks, one-pass model-blind generator, physical audit, immutable dataset freeze, and validation seal. Its 120 runs completed once, but 8/19 physical gates failed. The corpus is `SEALED_FAILED_PHYSICAL_EVIDENCE`; there is no validation authorization, payload open, model inference, optimizer step, HNM round, or candidate freeze. The runtime and E84 engineering-reference interfaces remain unchanged. The next work boundary is model-blind physical redesign, not training or architecture expansion.
