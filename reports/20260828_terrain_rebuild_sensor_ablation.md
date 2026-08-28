# Terrain Rebuild and Sensor Ablation

Date: 2026-08-28 (Asia/Seoul)  
Milestone: `TERRAIN_REBUILD_AND_SENSOR_ABLATION`  
Verdict: `TERRAIN_RECOGNITION_SUPPORTED`  
Sensor recommendation: `LEFT_FSR4_RECOMMENDED`  
Final sensor architecture frozen: **NO**

## 1. Why Terrain was rebuilt

Current simulation now has calibrated Concrete/Marble→Ice/Sand spatial transitions, clean pre-transition walking, and both stable and fall outcomes on Ice and Sand. Historical Terrain evidence did not use this exact simulator/runtime context. A fresh event-driven four-class dataset and validation were therefore required.

The classifier answers only “which terrain is this foot contacting now?” Classes are `CONCRETE`, `MARBLE`, `ICE`, and `SAND`. Fall, Slip, Sink, stability, source terrain, scenario pattern, run ID, event ID, boundary time, exact geom identity, and support displacement are not model inputs.

## 2. Legacy comparison boundary

No dataset, normalization, generated window, Keras/TFLite model, or training code was copied from `/d/shin/Infineon`. The legacy left-foot FSR4+IMU6 result remains historical comparison only. The rebuilt candidate is generated entirely from this repository at source commit `2875c1ff89743413fd34676532077bdc199915b3`.

## 3. Current terrain definitions

Terrain is independent of locomotion outcome:

- Ice stable and Ice fall events are both `ICE`.
- Sand stable-deformation and Sand fall/deformation events are both `SAND`.
- Concrete and Marble are distinct hard-terrain identities.
- Physical Slip/Sink diagnostics are metadata only.

All 72 Ice runs exhibited the frozen physical Slip diagnostic and none was tagged as Sand deformation. All 72 Sand runs exhibited at least 1 mm actual deformable-support motion and none exhibited the frozen Slip diagnostic.

## 4. Current transition scenario source

The frozen calibration mechanics were not changed. The 144 deterministic physical conditions comprise 36 runs for each source-target pair: Concrete→Ice, Concrete→Sand, Marble→Ice, and Marble→Sand. Each pair has 18 stable-domain and 18 fall-domain predeclared variations. Actual outcomes were 41 stable/31 fall Ice runs and 39 stable/33 fall Sand runs; intended labels were never used as Terrain targets.

The run split was frozen before simulation: TRAIN 88, VALIDATION 28, HOLDOUT 28. Physical-condition signature duplicates and split overlap were both zero. Pre-transition falls were zero.

## 5. Foot IMU implementation

Both observer sites are fixed to the ankle-roll body at local `pos=[0.035, 0.0, -0.020] m`, `quat=[1,0,0,0]`:

- left: `left_ankle_roll_link/left_foot_imu`
- right: `right_ankle_roll_link/right_foot_imu`

The site is centered longitudinally between the rear and front sole contact points and 15 mm above the nominal sole bottom. Its local neutral-pose convention is +x forward, +y robot-left, +z up. Raw output is 1 kHz `float32`, ordered left accel xyz, left gyro xyz, right accel xyz, right gyro xyz; units are m/s² and rad/s.

## 6. Sensor parity

Matched runs with Foot IMU reads disabled/enabled were evaluated for hard ground, Ice, and deformable Sand. In all three:

- qpos/qvel maximum absolute difference: `0.0 / 0.0`
- pelvis IMU, FSR, controller observation/action/update timing: exact
- exact contact, fall, Slip, and deformable support diagnostics: exact
- Foot IMU: finite, `[N,12]`, 1 kHz aligned

The MJCF declarations and reads are passive observers; they do not modify dynamics or control.

## 7. Dataset design

Dataset ID: `terrain_transition_20260828`  
Local path: `data/raw/terrain_transition_20260828/`  
Schema: `terrain_event_contract_v1`  
Runs/samples: 144 / 1,152,000  
Size: 108,596,314 bytes (about 103.6 MiB)

Each compressed NPZ stores the complete 8 s run once: pelvis IMU6, bilateral FSR8, bilateral Foot IMU12, and separately named diagnostic arrays. Windows are not duplicated in storage. The generated dataset, model checkpoints, and local metrics are Git ignored.

Manifest SHA-256: `2bca2070b24d4724b886ee83c20d493d63cb7cdd0a2ffc6596918a08c47e698a`  
Event-index SHA-256: `d5009f819eec035da1be81f1faa8dcebb1781a9385188f5ab82122cc6b733b24`

## 8. Touchdown event contract

Ground truth is the exact named sole-to-ground geom identity. Each foot/terrain identity rising edge is a candidate touchdown. A primary 50 ms event is clean only when the candidate identity remains present for the complete window, the window completes before first fall, sensors remain valid, and samples with any other terrain identity are strictly below 20%.

The 20% value was fixed before collection. A value equal to 20% is excluded. Boundary-straddling events are stored as `AMBIGUOUS_BOUNDARY`; they are not relabeled. The event CSV stores metadata and offsets only, not duplicate time-series.

## 9. Dataset coverage

There were 29,585 raw identity-onset candidates, 3,139 eligible 50 ms events, and 26,446 exclusions. Of the exclusions, 1,247 were explicit ambiguous-boundary events; most others were short contact chatter, post-fall, or incomplete/censored contacts.

| Coverage | Count |
|---|---:|
| CONCRETE | 991 |
| MARBLE | 971 |
| ICE | 803 |
| SAND | 374 |
| Left | 1,620 |
| Right | 1,519 |
| Ice stable/fall-run target events | 528 / 275 |
| Sand stable/fall-run target events | 210 / 164 |

Concrete-origin target events were Ice 343/Sand 178; Marble-origin target events were Ice 460/Sand 196. Dataset acceptance passed every predeclared gate: total/class/side minimums, stable/fall coverage, source diversity, finite Foot IMU, finite nonnegative FSR, zero drop, zero duplicate condition, zero split overlap, and zero pre-transition fall.

## 10. Split

The statistical split unit is the complete run/physical condition, never a window. The deterministic construction cap is at most two clean events per class per run. Primary bilateral TRAIN/VALIDATION/HOLDOUT event counts after the cap were 352/112/112, class-balanced at 88/28/28 per class. Holdout integrity and counts were checked before selection; its waveforms and performance remained sealed.

## 11. FSR4 result

At 50 ms with MLP and three fixed seeds, validation mean macro F1 was **0.9284** and mean worst-class recall was **0.8571**. The three-seed mean-logit ensemble macro F1 was 0.9282. FSR4 passed both sensor qualification gates.

## 12. Foot IMU6 result

Validation mean macro F1 was **0.9129** and mean worst-class recall was **0.8095**. The ensemble macro F1 was 0.9295. Despite useful signal, the profile failed the predeclared worst-class recall ≥0.85 qualification gate.

## 13. Fusion10 result

Validation mean macro F1 was **0.9309** and mean worst-class recall was **0.8333**. The ensemble macro F1 was 0.9461. It had the highest mean macro F1 but failed the worst-class recall gate.

## 14. Minimum-sensor selection

Only FSR4 simultaneously passed macro F1 ≥0.90 and worst recall ≥0.85. It was therefore selected without needing a post-result tolerance change. It also uses the fewest model channels: 4 versus 6 or 10.

## 15. MLP/GRU result

With selected FSR4 at 50 ms, MLP reached validation mean macro F1/worst recall **0.9284/0.8571**. GRU reached **0.6168/0.4286**. MLP was retained. No CNN/LSTM or capacity search was performed.

## 16. 20/30/50 ms horizon

| Horizon | Validation mean macro F1 | Mean worst recall | Gate |
|---:|---:|---:|---|
| 20 ms | 0.8789 | 0.6429 | FAIL |
| 30 ms | 0.8688 | 0.7024 | FAIL |
| 50 ms | 0.9284 | 0.8571 | PASS |

The shortest passing causal observation horizon is **50 ms**. Each horizon used its precomputed validity flag; no future sample was included.

## 17. One-shot holdout

The selected profile/family/horizon/deployment scheme was written to `selection_before_holdout.json` before the holdout guard opened. The guard opened once. No holdout-driven reselection occurred.

Selected LEFT_ONLY holdout results:

- accuracy: **0.9712**
- macro F1: **0.9713**
- worst-class recall: **0.9500**
- run-balanced macro F1: **0.9563**
- run-balanced accuracy: **0.9732**

Confusion matrix, rows/columns `[CONCRETE, MARBLE, ICE, SAND]`:

```text
[[27, 0, 1, 0],
 [ 1,27, 0, 0],
 [ 0, 0,28, 0],
 [ 0, 1, 0,19]]
```

## 18. Per-class performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| CONCRETE | 0.9643 | 0.9643 | 0.9643 | 28 |
| MARBLE | 0.9643 | 0.9643 | 0.9643 | 28 |
| ICE | 0.9655 | 1.0000 | 0.9825 | 28 |
| SAND | 1.0000 | 0.9500 | 0.9744 | 20 |

## 19. Ice stable/fall robustness

Observed-stable Ice recall was **1.0000** on 16 events; observed-fall-run Ice recall was **1.0000** on 12 events. This supports an ICE identity that does not depend on whether the later run falls.

## 20. Sand stable/fall robustness

Observed-stable Sand recall was **0.7500** on 4 left-foot events; observed-fall-run Sand recall was **1.0000** on 16 events. The 0.25 gap passed the predeclared catastrophic-gap gate, but stable-Sand left-only support is small and remains a limitation.

## 21. Concrete/Marble source robustness

For target Ice/Sand events, Concrete-origin accuracy was **1.0000** (24 events). Marble-origin accuracy was **0.9583** (24 events). Marble-origin Ice/Sand recalls were 1.00/0.90. Source history did not create a catastrophic failure.

## 22. Left/right robustness

The BILATERAL_SHARED validation ensemble was evaluated separately by foot:

- left: macro F1 0.9185; recalls Concrete/Marble/Ice/Sand = 1.00/0.75/0.9091/1.00
- right: macro F1 0.9277; recalls Concrete/Marble/Ice/Sand = 0.80/0.95/1.00/1.00

Neither side was catastrophic. The selected LEFT_ONLY model intentionally has no right-foot inference result; right events were not passed to it.

## 23. LEFT_ONLY result

LEFT_ONLY retraining used 328 train and 108 validation events. Three-seed validation mean macro F1/worst recall was **0.9612/0.8810**; its mean-logit ensemble macro F1/worst recall was **0.9821/0.9286**. It passed the validation gates and uses one FSR4 package.

## 24. BILATERAL_SHARED result

BILATERAL_SHARED used the same shared 4-channel model input on either foot, with no side ID. Its three-seed validation mean macro F1/worst recall was **0.9284/0.8571** and the ensemble was **0.9282/0.8571**. It also passed.

## 25. Update latency comparison

Latency is `terrain_valid - first_target_contact_any_foot`; model compute time is not included. The delay includes waiting through a boundary-straddling event until the next clean touchdown.

| Scheme | Evaluable/total | Median | p95 | Max | Unavailable |
|---|---:|---:|---:|---:|---:|
| LEFT_ONLY | 126/144 | 1114.5 ms | 1238 ms | 1238 ms | 18 |
| BILATERAL_SHARED | 144/144 | 922 ms | 1238 ms | 1238 ms | 0 |

The 18 LEFT_ONLY unavailable runs are right-lane-only Sand transitions in which the left foot never produces a clean Sand touchdown. This is a sensor-placement coverage tradeoff, not AI inference latency.

## 26. Physical channel-count implication

| Scheme | Sensor package | Channels incl. Pelvis IMU6 | Validation macro F1 mean | Worst recall mean | Median delay | p95 delay |
|---|---|---:|---:|---:|---:|---:|
| LEFT_ONLY | left FSR4 | 10 | 0.9612 | 0.8810 | 1114.5 ms | 1238 ms |
| BILATERAL_SHARED | FSR4×2, shared model | 14 | 0.9284 | 0.8571 | 922 ms | 1238 ms |

The current minimum-channel recommendation is `LEFT_FSR4_RECOMMENDED`. If every asymmetric right-foot terrain contact must update state, BILATERAL_SHARED remains the operational alternative. No post-result latency threshold was invented to force either choice.

## 27. Leakage audit

Model tensors contain only the touchdown foot's FSR4, Foot IMU6, or their ordered concatenation. Side ID and pelvis IMU are absent. Exact terrain/contact geom, source/target terrain, run/event IDs, observed fall, Slip/Sink, support displacement, contact point, scenario pattern, and boundary-relative time never enter the tensor. Normalization was fit on train events only. Runs are disjoint across all splits.

FSR is computed from MuJoCo contact normal force while the label is exact contact geom identity. It is not a numeric label field, but this remains simulation-derived observability evidence rather than proof of actual FSR hardware generalization.

## 28. Limitations

- Terrain profiles and Sand supports are engineering proxies, not measured real materials.
- Virtual FSR omits real sensor hysteresis, drift, saturation, mounting, calibration, and electrical conversion.
- Deterministic simulation can understate real domain variation.
- LEFT_ONLY misses 18/144 right-only target updates and has longer median update delay.
- Stable-Sand LEFT_ONLY holdout support is only four events.
- The candidate has not been quantized, exported as a frozen release, run on E84, or integrated with a supported Stability detector.

## 29. Verdict

The dataset acceptance gate and one-shot holdout gates passed. All four classes had meaningful recall, Ice stable/fall remained identical at 1.0 recall, Sand stable/fall passed the frozen catastrophic-gap rule, and Concrete/Marble source robustness was reasonable.

`TERRAIN_RECOGNITION_SUPPORTED`

The current research candidate is left-foot FSR4, MLP, 50 ms event-driven inference with held state. `FINAL_SENSOR_ARCHITECTURE_FROZEN` is explicitly not declared.

## 30. Recommended next step

Stop here for this milestone. The next separately authorized task should redesign and accept the Stability ground-truth clock before Stability AI or final sensor freeze. Recovery, full integrated dataset, quantization/E84, and deployment work were not started.
