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
  -> evaluation/hazard.py

simulation/{g1,sensors,terrain}.py
  -> dataset/terrain.py
  -> training/terrain.py
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

`training/hazard.py` owns positive windows, true negative regions, TRAIN-only normalizer fitting and HNM. The fixed HNM contract is three rounds after Round 0, 1 ms replay, K=12 per run, 30 ms minimum spacing, and no negative after I1 becomes active. Supplied runs must declare `split: train`; otherwise training construction fails.

`evaluation/hazard.py` owns continuous replay, probability aggregation, threshold/persistence, physical run metrics and read-only verification of the selected freeze. Scientific HOLDOUT is not reopened during routine verification. The freeze verifier checks candidate identity, schema, normalizer, all Hazard checkpoints and all protected Terrain hashes.

`dataset/terrain.py`, `training/terrain.py`, and `evaluation/terrain.py` own the corresponding clean-event, training and runtime inference responsibilities. No historical Terrain-gated Hazard fusion module remains in the current dependency graph.

## 6. Simulation boundary

`simulation/g1.py` is the single G1 loop with 0.5 ms physics and 1 kHz sampling. Runtime signals are Pelvis IMU6 and optional FSR8/Foot IMU12 observers. Physical diagnostics are stored separately.

- `simulation/sensors.py`: FSR and Foot IMU observation
- `simulation/terrain.py`: Concrete/Marble/Ice/Sand and support mechanics
- `simulation/hazards.py`: contact/touchdown, Slip, support spread/loss and I1 inputs
- `simulation/stability.py`: simulator diagnostics still computed by `g1.py` to preserve physics/viewer result parity; it is not a supported runtime Hazard detector

The ordinary simulation viewer copies physics state into a render-only model. Viewer input cannot feed back into the canonical simulation.

`visualization.py` owns read-only run resolution, HOLDOUT rejection, exact stored/re-simulated parity, frozen inference alignment, snapshot playback and HUD formatting. `run_simulation(..., capture_render_trace=True)` optionally captures one memory-only full MuJoCo `mjSTATE_INTEGRATION` snapshot per 1 kHz sample; the default is `False`, the stored NPZ contract is unchanged, and Sand support degrees of freedom are included. The interactive viewer opens only after timestamp, IMU6, FSR8 and physical event-clock parity passes. It restores selected immutable snapshots with `mj_setState`, calls `mj_forward`, and renders with `viewer.sync`; it never calls `mj_step`. Playback speed, pause, backward/forward seek, event jumps and overlay text therefore cannot enter controller, physics, sensor, feature or model tensors. Closing the viewer is a clean user exit because scientific parity was completed headlessly before it opened.

## 7. Research-to-Deployment boundary

This repository ends at reviewed Float research artifacts and their contracts. Quantization, Vela, E84 firmware integration, HIL, target latency and Recovery belong to the deployment repository. No deployment activity is implied by the supported research verdict.

## 8. Historical research summary

MoS/Stability, direct NORMAL/SLIP/SINK classification, event-centric detectors, Terrain-gated branches, Support fusion variants and their diagnostic tests are not current runtime dependencies. Their scientific results remain in [`../reports/`](../reports/) and their original source is recoverable from Git history. Dated experiment configs remain as provenance; the current CLI rejects them as historical instead of silently selecting another implementation.
