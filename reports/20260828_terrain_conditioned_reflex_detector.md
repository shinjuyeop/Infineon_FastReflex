# Terrain-conditioned reflex detector development

## 1. Motivation

This milestone replaces the failed global `REFLEX_EVENT = Slip OR Support` detector with two independent continuous detectors gated by the protected Terrain producer. The question is no longer whether a short event-local window is separable; it is whether the complete runtime chain can operate continuously without unacceptable false alerts.

Starting provenance was recorded before work began:

- branch: `main`
- HEAD: `b6c91fe986abdffba2fbc1ec66915e15150fc93c`
- `origin/main`: `b6c91fe986abdffba2fbc1ec66915e15150fc93c`
- worktree: clean

## 2. Why global REFLEX_EVENT failed

The historical event-local classifiers had near-perfect AUROC/AUPRC, but full-run replay produced pervasive normal-gait alerts. IMU6 reached 100% Slip recall but only 25% Support recall with 0% no-event and hard-ground specificity; IMU6+FSR8 reached 100%/50% type recall with 25%/0% specificity. One union class and sparse negatives did not cover continuous normal operation.

The historical report and verdict `EVENT_CENTRIC_REFLEX_DETECTION_NOT_SUPPORTED` were not edited.

## 3. Terrain-conditioned architecture

The evaluated system was:

```text
Frozen Terrain == ICE  -> Slip detector -> REFLEX_REQUIRED
Frozen Terrain == SAND -> Support detector -> REFLEX_REQUIRED
Frozen Terrain == CONCRETE/MARBLE/UNKNOWN -> no terrain-specific branch
```

Detector probabilities were computed continuously. Only the current held Terrain model output could make an alert actionable. Ground-truth target terrain never bypassed this gate. Raw cross-terrain alerts were retained as diagnostics.

## 4. Frozen Terrain contract

The protected FSR4/MLP/50 ms ensemble (seeds 17/29/43) and train normalizer were loaded without retraining. The current `LEFT_ONLY` deployment contract was used: exact simulator contact scheduled clean left-foot touchdown events, the actual frozen FSR4 ensemble predicted a class after 50 ms, and that prediction was held until the next valid clean left touchdown.

Exact contact was scheduler-only privileged state. It selected when the already-defined clean-touchdown producer was allowed to run; the predicted class—not exact terrain identity—set branch state. Re-simulated TRAIN/VALIDATION FSR8, pelvis IMU6, loaded contact, event clocks and fall clocks were bit-exact with all 204 stored development runs.

## 5. Event-before-terrain audit

The timing gate failed globally and catastrophically for Slip:

| Branch | Event runs | Event before target Terrain output | Rate | Valid margin median | p95 | Missing target output |
|---|---:|---:|---:|---:|---:|---:|
| Slip / ICE | 96 | 96 | 100.0% | -147 ms | -147 ms | 24 |
| Support / SAND | 48 | 0 | 0.0% | 896 ms | 1,182 ms | 0 |
| Overall | 144 | 96 | 66.67% | -147 ms | 1,181.05 ms | 24 |

For the 72 Ice runs where a target `ICE` output eventually existed, event-minus-Terrain margins ranged from -394 to -147 ms. All Slip events therefore happened before the branch could become active. Sand margins ranged from 23 to 1,183 ms and supported a causal Support branch.

The predeclared feasibility target was event-before-terrain rate at most 5%. It failed overall and for Slip, and passed for Support.

## 6. Slip branch semantics

The physical oracle remained unchanged:

```text
touchdown-anchor tangential drift >= 0.050 m
for 3 consecutive 1 kHz samples
```

ANY left/right Slip was positive; bilateral correctness was diagnostic. The intended positive endpoints were `t_slip - 20 ... t_slip + 40 ms`, stride 5 ms, but only while the frozen state was `ICE`. Negative endpoints had to be active-ICE samples no later than `t_slip - 30 ms`.

## 7. Support branch semantics

The physical oracle remained unchanged:

```text
support spread = max(cell displacement) - min(cell displacement)
support spread >= 0.010 m for 20 ms
```

Balanced Sand deformation below 10 mm spread remained `NORMAL_SAND`, even when absolute displacement was about 20 mm. Support positives used `t_support - 20 ... t_support + 40 ms`; validation accepted detector latency from -30 to +50 ms.

## 8. Development corpus

The existing `reflex_event_20260828` corpus and split were reused.

| Cohort | TRAIN | VALIDATION | sealed HOLDOUT | Total |
|---|---:|---:|---:|---:|
| Ice Slip event | 72 | 24 | 24 | 120 |
| Sand Support event | 36 | 12 | 12 | 60 |
| Sand benign no-event | 36 | 12 | 12 | 60 |
| Hard controls | 8 | 4 | 4 | 16 |
| All runs | 152 | 52 | 52 | 256 |

The primary independent unit remained a simulation run. Fall/recovery outcome was diagnostic only and did not define detector labels.

## 9. Feature engineering

One canonical causal extractor produced deterministic schemas:

- pelvis or foot IMU: raw accel/gyro, accel and gyro norms, horizontal norms;
- FSR8: raw channels, per-foot total/front/rear/medial/lateral loads, front-rear and medial-lateral differences, normalized ratios, and bilateral totals/difference/ratio;
- temporal expansion: current base, 1/5/10 ms deltas, trailing 10 ms mean, trailing 10 ms variance.

This yielded 60 features for Pelvis IMU6, 180 for FSR8, 120 for Foot IMU12, and concatenated dimensions for multi-sensor candidates. All operations used current/past samples only. Terrain identity, patch geometry, event/fall clocks, outcome and Slip/Support oracle values were absent from model tensors.

## 10. Initial negative coverage

Initial negatives were drawn from the complete active branch interval and covered deterministic early/middle/late temporal bins with eight endpoints per bin. Event-run negatives ended at event-30 ms. Sand benign runs supplied full active-SAND normal coverage. Training used a deterministic 80/20 run-disjoint split inside TRAIN for fit versus epoch monitoring; VALIDATION was not used for epoch selection.

Under the actual Terrain gate:

- Slip TRAIN active positive endpoints: 0
- Slip TRAIN active negative endpoints: 0
- Support TRAIN active positive endpoints: 342
- Support TRAIN active negative candidates: 139,864

## 11. Hard-negative mining protocol

Every trainable candidate followed the frozen sequence:

```text
Round 0 train
-> full TRAIN replay
-> HNM1: top 8 per run, >=50 ms separation
-> Round 1 train
-> full TRAIN replay
-> HNM2: top 8 new endpoints per run
-> Round 2 final train
```

Positive regions were never eligible for negative mining. All initial positives were retained. Exactly two HNM rounds were allowed. The mining split was TRAIN only.

## 12. Round 0 training

All trainable Support candidates began with 1,443 fit windows: 243 event and 1,200 normal. Three seeds (`20260828/29/30`) used balanced cross-entropy, Adam at 1e-3, maximum 40 epochs and patience 6. Best epochs used TRAIN-internal monitor cross-entropy.

The 12 Phase A Slip combinations were enumerated but fail-closed before fitting because the actual gate supplied neither class. A model was not fabricated with ground-truth Ice gating.

## 13. HNM Round 1

For every Support candidate, HNM1 found hard negatives in 63 TRAIN runs and added up to 504 endpoints. Fit-window totals became 1,834–1,843 depending on duplicate/exclusion outcomes. The selected P3/GRU/20 ms candidate used 1,843 Round 1 windows.

## 14. HNM Round 2

HNM2 again found hard negatives in 63 TRAIN runs. Final Support fit-window totals were 2,221–2,243. The selected candidate used 2,243 windows: 243 positives and 2,000 negatives. No third mining round or result-driven feature/model expansion occurred.

## 15. Train full-replay diagnostic

At the diagnostic threshold 0.5, selected P3/GRU/20 ms Round 2 produced:

- Support event recall: 80.56%
- benign specificity: 50.0%
- premature event-run rate: 0%
- active-negative alert fraction: 0.7014%
- valid latency median/p95: -19/-1 ms

This was diagnostic only. The operational threshold was calibrated later on VALIDATION as predeclared.

## 16. Phase A Slip validation

All 12 combinations were represented in the result matrix:

| Sensor | Model/history candidates | Active TRAIN positive/negative | Validation recall | Result |
|---|---|---:|---:|---|
| S1 Pelvis IMU6 | MLP/GRU × 20/50 ms | 0 / 0 | 0% | gate-infeasible |
| S2 IMU6+FSR8 | MLP/GRU × 20/50 ms | 0 / 0 | 0% | gate-infeasible |
| S3 FSR8 | MLP/GRU × 20/50 ms | 0 / 0 | 0% | gate-infeasible |

The failure is upstream of sensor observability: the actual LEFT_ONLY Terrain producer never provided an active-ICE training interval before Slip.

## 17. Phase A Support validation

Each candidate used a 1 ms causal replay, 5 ms persistence and thresholds 0.10–0.98 in 0.02 steps.

| Candidate | Model | History | HNM fit windows R0/R1/R2 | Threshold | Recall | Benign spec. | Premature | Negative alert time | Latency med./p95 | PASS |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| P1 FSR8 | MLP | 20 | 1443/1843/2243 | — | 100% | 83.33% | 0% | 0.0130% | -7/-1 ms | no |
| P1 FSR8 | MLP | 50 | 1443/1838/2229 | 0.98 | 100% | 100% | 0% | 0% | -9.5/-1 ms | yes |
| P1 FSR8 | GRU | 20 | 1443/1843/2237 | 0.94 | 100% | 91.67% | 0% | 0.0283% | -9/-1 ms | yes |
| P1 FSR8 | GRU | 50 | 1443/1834/2234 | 0.98 | 100% | 100% | 0% | 0% | -9.5/-1 ms | yes |
| P2 IMU6+FSR8 | MLP | 20 | 1443/1843/2243 | 0.98 | 100% | 100% | 0% | 0% | -9.5/-1 ms | yes |
| P2 IMU6+FSR8 | MLP | 50 | 1443/1843/2234 | — | 100% | 75% | 0% | 0.0674% | -9/-1 ms | no |
| P2 IMU6+FSR8 | GRU | 20 | 1443/1843/2235 | 0.98 | 100% | 100% | 0% | 0% | -9/-1 ms | yes |
| P2 IMU6+FSR8 | GRU | 50 | 1443/1843/2221 | 0.98 | 100% | 100% | 0% | 0% | -9.5/-1 ms | yes |
| P3 IMU6 | MLP | 20 | 1443/1843/2243 | 0.96 | 100% | 100% | 0% | 0% | -3.5/0 ms | yes |
| P3 IMU6 | MLP | 50 | 1443/1843/2243 | — | 100% | 50% | 0% | 0.1153% | -8/1 ms | no |
| P3 IMU6 | GRU | 20 | 1443/1843/2243 | 0.94 | 100% | 100% | 0% | 0% | -8/-1 ms | yes |
| P3 IMU6 | GRU | 50 | 1443/1843/2221 | — | 100% | 50% | 0% | 0.3458% | -8.5/-1 ms | no |

## 18. Phase A sensor selection

The frozen priority selected P3, one-layer GRU, 20 ms, threshold 0.94:

- derived Pelvis IMU6 only, 60 model features;
- 9,090 parameters;
- 100% Support recall (12/12);
- 100% Sand benign specificity (12/12);
- 0% premature runs;
- 0% active-negative alert time;
- median/p95 latency -8/-1 ms;
- Concrete-origin 6/6 and Marble-origin 6/6;
- recovered-event 3/3 and fall-event 9/9.

P3/MLP/20 ms used the same six physical channels but had a later p95 of 0 ms, so the predeclared p95 priority selected the GRU.

## 19. Phase B activation decision

Phase B activated for `slip` only. Support had a valid Phase A selection and was not subjected to Foot IMU fishing. No q/dq, torque, current, FSR stability augmentation or new model family was introduced.

## 20. Foot IMU parity if used

Generated dataset: `reflex_event_foot_imu_20260828`.

- runs: 256, with the original split membership;
- size: 171,100,805 bytes in the recorded run payload summary;
- manifest SHA-256: `c206e05dec4f6bb63a667a5fe8f0c1fd0e9c4d45f4c4e8901a571c81c0dba234`;
- 204 TRAIN/VALIDATION pelvis IMU6, FSR8, loaded-contact, event and fall clocks: exact parity;
- all 256 event/contact/fall/outcome clocks: exact parity;
- matched observer-disabled/enabled hard, Ice and Sand cases: robot qpos/qvel, controller observation/action, policy update, pelvis pose, COM, contacts, fall, Slip and support diagnostics all exact;
- maximum qpos/qvel difference: 0.

Foot IMU12 was therefore observer-only. Generated NPZ and manifest remain Gitignored.

## 21. Phase B results if used

The fixed Phase B Slip candidates SF1 Foot IMU12, SF2 Foot IMU12+FSR8, and SF3 Foot IMU12+Pelvis IMU6 were each enumerated with MLP/GRU and 20/50 ms history: 12 combinations total.

All had active-ICE TRAIN positive/negative endpoint counts of 0/0, so Round 0, HNM1, Round 1, HNM2 and Round 2 were recorded as not performable under the actual frozen gate. Validation recall was 0% for every combination. Training on truth-gated Ice would have violated the system contract, so no checkpoint or threshold was selected.

## 22. Frozen final branch candidates

Validation-frozen Support candidate:

```text
P3 / Pelvis IMU6 derived / GRU / 20 ms / threshold 0.94 / persistence 5 ms
```

Slip has no candidate. Therefore this is a partial research selection, not a complete deployable architecture release. No final sensor architecture was frozen.

## 23. One-shot holdout

The holdout opening condition required both Slip and Support candidates. Slip had none, so:

- holdout performed: no;
- holdout guard open count: 0;
- reselection: no;
- HOLDOUT detector waveform/model access: none.

## 24. Slip holdout

Not performed because no Slip validation selection existed. Recall, premature rate, negative alert time and latency are not claimed.

## 25. Support holdout

Not performed because the integrated two-branch opening condition was not met. Validation support is strong, but one-shot holdout generalization is intentionally unclaimed.

## 26. Integrated system metrics

No integrated holdout metric was computed. On Support VALIDATION, system hard-ground specificity was 100% (4/4) and Sand no-event specificity was 100% (12/12). Raw P3 Support alerts occurred on 15/28 cross-terrain/hard diagnostic runs, demonstrating why the actual Terrain gate materially matters; none became a hard-ground system alert.

The system cannot provide an actionable Slip branch under current LEFT_ONLY Terrain timing, so an overall physical-event recall is not claimed.

## 27. Terrain timing margin

The 50 ms classifier observation is not the dominant Ice delay. The first clean left target touchdown yielding `ICE` arrived after every Slip event, and 24 Ice runs never yielded target `ICE` before censor. Sand target output preceded every Support event, usually by a large margin, although the smallest margin was 23 ms.

This is the central system result: event-wise Terrain accuracy and a protected Terrain verdict do not by themselves guarantee that a terrain-gated reflex branch is available before a faster contact event.

## 28. Recovered vs fall event

The selected Support candidate detected all validation subgroups:

- recovered/non-fall Support events: 3/3;
- fall Support events: 9/9;
- Concrete origin: 6/6;
- Marble origin: 6/6.

Outcome was not part of the label or model tensor.

## 29. Sensor-count tradeoff

For the supported partial path:

- Terrain: left-foot FSR4 = 4 physical channels;
- Slip: no selected detector;
- Support: Pelvis IMU6 = 6 physical channels;
- unique selected partial-path total: 10 physical channels.

Foot IMU12 is not counted because its fallback did not produce a Slip candidate. Model-derived dimensions are not counted as physical sensors.

## 30. Historical comparison

| Study | Primary result | Interpretation |
|---|---|---|
| Sparse fall prediction | insufficient reliable separation | too few independent windows |
| Dense fall risk | recoverable disturbances caused FP | fall outcome was not a robust reflex target |
| Global event detector | continuous normal gait caused FP | union class and sparse negatives failed |
| Terrain-conditioned + HNM | Support validation PASS; Slip gate-infeasible | branch-specific learning works where Terrain precedes event |

Historical MoS, full-state distance, sparse temporal, dense fall-risk and global event reports remain unchanged.

## 31. Limitations

- The current result is limited to the G1 MuJoCo policy, engineering terrain and frozen physical oracles.
- Support has validation evidence only; the integrated holdout stayed sealed.
- LEFT_ONLY was the protected minimum-channel Terrain deployment. BILATERAL_SHARED timing was not substituted after seeing the result.
- Exact terrain contact schedules clean touchdown events in simulation; real hardware needs an equivalent causal touchdown validity producer.
- Deterministic condition families produce correlated trajectories even though run IDs and physical signatures are independent.
- Raw cross-terrain Support firing remains common and makes correct Terrain state operationally important.
- Foot IMU parity establishes observability without physics mutation, not Slip usefulness under an inactive gate.

## 32. Verdict

`TERRAIN_CONDITIONED_REFLEX_PARTIALLY_SUPPORTED`

Supported branch: `SUPPORT_ONLY`.

The Support-specific continuous detector is strongly supported on VALIDATION with HNM and existing Pelvis IMU6. Slip is not a detector-model failure in this protocol; the frozen LEFT_ONLY Terrain branch activates after the physical Slip clock, making both Phase A and the bounded Foot IMU fallback structurally untrainable without truth leakage.

Regression result: 185 tests passed, one user-policy end-to-end smoke was skipped, and there were no failures. New focused terrain-gating/feature/HNM/evaluation tests contributed 13 passing cases. Protected Terrain hashes and the existing Fusion truth table passed unchanged.

## 33. Next recommendation

Do not add a larger Slip model, another sensor, severity classifier, recovery controller or final architecture freeze. The next bounded study should address the upstream Terrain-to-Slip timing contract—without retraining or rewriting this milestone—and predeclare whether a causal bilateral/shared terrain update or a non-terrain-gated local Slip path is architecturally allowed. Support holdout should remain sealed until a complete two-branch selection exists or a separately approved Support-only holdout protocol is declared in advance.

No Recovery, production enum change, E84 work, deployment export or final sensor freeze was started.
