# Support detector failure-mode audit

Milestone: `SUPPORT_FAILURE_MODE_AUDIT`  
Date: 2026-08-29 (Asia/Seoul)  
Verdict: `SUPPORT_FAILURE_MODE_IDENTIFIED`

## 1. Scope and isolation

This was a read-only failure analysis of the already frozen Sand Support branch. It did not train a model, change a feature, tune the 0.94 threshold or 5 ms persistence, change Terrain gating, open Foot IMU Phase B, generate a dataset, or modify the Slip branch.

The repository started at `HEAD = origin/main = 38c4a946cdec0d0c1286a5027a3b8077764eb6d1` on `main` with a clean worktree. The audit loaded only TRAIN and VALIDATION event waveforms and the matching `terrain_gate/development` cache. The module rejects any split other than `train` or `validation`, and rejects any Terrain gate cache not named `development`.

HOLDOUT waveform access count for this milestone was exactly 0. Neither historical metrics JSON containing consumed HOLDOUT detail nor HOLDOUT run IDs, score traces, features, or individual outcomes were read. The already published aggregate Support result, 9/12 with recovered 2/2 and fall 7/10, remains context only and was not used to form a diagnostic category or decision.

## 2. Frozen implementation reproduced

The canonical implementation remains `src/fastreflex/evaluation/terrain_conditioned_reflex.py`. The audit reused its causal feature extraction, frozen-Terrain state, 1 ms replay, threshold comparison, and persistence logic.

| Item | Frozen value |
| --- | --- |
| Candidate | P3 |
| Runtime sensor | Pelvis IMU6 |
| Derived input | 60 features |
| Model | one-layer GRU |
| History | 20 ms |
| Ensemble | seeds 20260828, 20260829, 20260830 |
| Parameter count | 9,090 per model |
| Threshold | 0.94 |
| Persistence | 5 consecutive 1 kHz samples |
| Valid Support latency | -30 to +50 ms |
| System gate | frozen Terrain state equals `SAND` |

The 60-D input is the ten Pelvis IMU base values—acceleration xyz, angular velocity xyz, acceleration norm, angular-velocity norm, horizontal acceleration norm, and horizontal angular-velocity norm—expanded as raw, causal 1/5/10 ms deltas, causal 10 ms mean, and causal 10 ms variance. Terrain identity, event/fall clock, outcome, patch geometry, FSR, and support oracle values are absent from the tensor.

The normalizer is the original TRAIN-only P3 normalizer fitted from 72 runs and 147,456 capped samples. Its feature schema SHA-256 is `4775bf9cdb1a6680c64c0c744caf69e34afb3628726594350133a59545835170`.

## 3. Baseline reproduction

| Split | Support events | Detected | Recall | Premature | Latency min / p10 / median / p95 / max (ms) |
| --- | ---: | ---: | ---: | ---: | --- |
| TRAIN | 36 | 29 | 80.56% | 0 | -17 / -17 / -15 / -1 / 36 |
| VALIDATION | 12 | 12 | 100.00% | 0 | -17 / -17 / -8 / -1 / -1 |
| Combined diagnostic | 48 | 41 | 85.42% | 0 | -17 / -17 / -15 / -1 / 36 |

The TRAIN 29/36 result reproduces the previously reported 80.56% full-run Support replay. VALIDATION reproduces the frozen selection evidence, including 12/12 recall, median -8 ms and p95 -1 ms. Validation was observed only; no selection or threshold calibration was repeated.

## 4. Negative baseline

The audit treated every run with no canonical Support event as a Support negative under the existing system contract. This includes Sand benign transitions, non-Sand transitions, and hard-ground controls. Alert time before `t_support - 30 ms` was also negative time on positive runs.

| Split | Negative runs | False reflex | Specificity | Active negative samples | Alert samples | Alert fraction | Sand benign | Hard controls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TRAIN | 116 | 0 | 100% | 147,017 | 0 | 0% | 36/36 specific | 8/8 specific |
| VALIDATION | 40 | 0 | 100% | 49,598 | 0 | 0% | 12/12 specific | 4/4 specific |

There was no premature Support event and no system false reflex. The result confirms that the frozen operating point is conservative, but the later miss analysis shows that lowering the score threshold is not the relevant explanation for the observed TRAIN misses.

## 5. Event diagnostic contract

`event_diagnostics.csv` has one row per TRAIN/VALIDATION Support event. Required clocks, score, threshold margin, raw and gated threshold crossing, raw and gated persistence, current Terrain state, first target-Terrain-valid time, touchdown, Support onset, outcome, source, pattern and selected IMU statistics are explicit.

`detector_max_score` means the ungated ensemble peak inside the canonical `[-30,+50] ms` response window. `active_gate_max_score` is the maximum among samples where the frozen Terrain state is actually `SAND`. This distinction is essential: the model can recognize the Support dynamics while the system output is suppressed by Terrain state.

For descriptive comparison only, detected events at or below the detected-event score-margin q25 were called low-margin successes. This is not a new acceptance criterion and was not used to tune the threshold.

| Diagnostic group | Events |
| --- | ---: |
| Clear success | 29 |
| Low-margin success | 12 |
| MISS | 7 |

The q25 score-margin cutoff was +0.034602. All 12 low-margin events still reached score 0.974602, remained above threshold for 13 consecutive milliseconds, had `SAND` at Support onset, and were detected. They are near-misses only in relative rank; they are not persistence-fragile cases.

## 6. Identified failure mode

All seven MISS events were in TRAIN. Every one had the same primary failure mode:

`GATING_SUPPRESSION`

Evidence:

- ungated event-window peak score was 0.9601–0.9938, always above threshold 0.94;
- raw score persistence was 16–63 ms, median 57 ms, always above the required 5 ms;
- gated threshold persistence was 0 ms for every miss;
- current Terrain state at Support onset was `MARBLE` for five misses and `ICE` for two, never `SAND`;
- the gate changed to the non-Sand state 7–29 ms before Support onset, median 12 ms;
- every detected event had `SAND` at Support onset and both raw and gated persistence satisfied.

Thus `SCORE_INSUFFICIENT`, `PERSISTENCE_FAILURE`, and sensor-level `SIGNAL_ABSENT` are not supported explanations for these development misses. The model response exists and is sustained; the current Terrain gating contract discards it. `TRANSITION_TIMING` is retained as a secondary descriptor because the first valid Sand clock occurs earlier, but the held state later changes before the Support event.

Representative plots show the frozen Support score, threshold, gated persistence, raw Pelvis acceleration/angular-velocity norms, the largest normalized causal feature for that event, the exact Support clock, and the held Terrain state. They do not plot or use HOLDOUT data.

## 7. Successful versus missed dynamics

| Metric | Detected events | MISS events |
| --- | --- | --- |
| Ungated peak score, min / median / max | 0.9746 / 0.9793 / 0.9937 | 0.9601 / 0.9910 / 0.9938 |
| Score margin, min / median / max | +0.0346 / +0.0393 / +0.0537 | +0.0201 / +0.0510 / +0.0538 |
| Raw persistence, min / median / max (ms) | 13 / 15 / 64 | 16 / 57 / 63 |
| Gated persistence, min / median / max (ms) | 13 / 15 / 64 | 0 / 0 / 0 |
| Gate at Support onset | SAND 41/41 | SAND 0/7 |

Misses do not have weaker median score or shorter raw persistence than successes. The direct input features also show substantial event-local dynamics. The largest absolute normalized feature was usually 1 ms pelvis `gyro_y` delta in misses, while successful groups more often peaked in 10 ms `gyro_y` variance or 10 ms horizontal-acceleration delta. This difference is descriptive only: the ensemble already converted every missed profile into a sustained positive score, so it is not evidence that feature engineering is the bottleneck.

Event peak score versus active-gate negative-run peak score had diagnostic AUROC 1.000. Event peaks were 0.9601–0.9938; negative-run active-gate peaks were 0.1956–0.9159. This is not a new model evaluation metric or threshold selection, but it shows that the current Pelvis IMU6 contains strong Support evidence in TRAIN/VALIDATION.

## 8. Recovered versus fall

| Split / outcome | Detected / events | Recall | Misses |
| --- | ---: | ---: | ---: |
| TRAIN recovered | 2/2 | 100% | 0 |
| TRAIN fall | 27/34 | 79.41% | 7 |
| VALIDATION recovered | 3/3 | 100% | 0 |
| VALIDATION fall | 9/9 | 100% | 0 |
| Combined recovered | 5/5 | 100% | 0 |
| Combined fall | 36/43 | 83.72% | 7 |

The development data independently reproduces the direction suggested by the published HOLDOUT aggregate: misses are concentrated in fall Support events. It does not show that fall dynamics produce a weaker model score. Median score margin was +0.0501 for fall and +0.0393 for recovered, and no event had raw persistence failure. Median peak timing was -6 ms for fall and -15 ms for recovered. Terrain-to-Support first-valid margin was longer for fall, not shorter: median 1,175 ms versus 612 ms.

Recovered has only five events, so no significance claim is made. The evidence supports a correlation between fall subgroup and gate-state timing in this corpus, not a theorem that fall events are intrinsically less observable.

## 9. Terrain-to-Support timing

The first target-valid clock by itself is insufficient to describe Support availability. Descriptive quartile bins gave:

| First Sand-valid to Support margin | Events | Recall |
| --- | ---: | ---: |
| Very short, <=464.75 ms | 12 | 100% |
| Middle | 12 | 83.33% |
| Long, >=1,175 ms | 24 | 79.17% |

This does not mean a long margin causes failure. It means the initially valid Sand state can be replaced by a later clean-touchdown prediction. All misses had a valid Sand output earlier, but the current held state was non-Sand at Support onset. Two misses retained Sand for roughly 70–75% of the last 100 ms and then switched shortly before the event; the other five had no Sand state in that interval. Current-state continuity, rather than the first-valid timestamp, explains the output suppression.

No Terrain model or deployment scheme was changed. This audit only identifies the interaction between the frozen Terrain producer and frozen Support consumer.

## 10. Sensor observability judgment

`SUPPORT_SIGNAL_PRESENT_BUT_DECISION_WEAK`

Within TRAIN/VALIDATION, Pelvis IMU6 already produces a well-separated and persistently high Support score for every miss. The evidence does not justify Foot IMU, FSR stability channels, q/dq, torque, current, or another physical sensor. The failure is in the current gating/decision handoff, not demonstrated lack of Support observability.

This judgment is bounded to current MuJoCo scenarios, the current G1 policy, and the consumed development split. It does not assert that Pelvis IMU6 is universally sufficient, and it does not upgrade the system verdict or freeze the sensor architecture.

## 11. Improvement candidates—not executed

| Priority | Candidate | Evidence addressed | Leakage risk | Trade-off | New sensor |
| ---: | --- | --- | --- | --- | --- |
| 1 | Evaluate a continuous Support score with Terrain applied at later fusion rather than as a hard pre-output gate | all 7 misses have valid raw score/persistence and 0 gated persistence | must not use exact terrain, Support oracle, event clock, or future Terrain; only causal runtime outputs | prior raw cross-terrain Support alerts mean specificity must be rebuilt with hard negatives and a new untouched final set | No |
| 2 | Predeclare and validate causal Terrain-state continuity/hysteresis for the Support handoff | gate switched away 7–29 ms before every miss despite earlier Sand validity | grace/hysteresis cannot be event-relative or chosen using future Support onset | may preserve stale Sand and increase Ice/hard false reflexes; requires fresh TRAIN/VALIDATION and a new final set | No |
| 3 | If raw-score failures appear after gating is corrected, add gate-churn/fall-balanced hard-positive auditing before model changes | fall has 7 development misses, but current raw score is already sufficient in all 7 | outcome can define cohorts, never runtime tensor or sample-time gate | may improve coverage of rare modes but cannot by itself override a suppressing gate; model work is lower priority | No initially |

Threshold lowering, persistence shortening, larger GRU, new features, and sensor augmentation are not supported as first actions by this audit. No candidate was implemented.

## 12. Integrity and artifacts

- protected Terrain paths: unchanged;
- frozen Support config, source, selection, TRAIN normalizer, and three checkpoints: unchanged;
- Slip freeze artifact identity: `df0a232ec242283ef8b25c59421cebde982a7a93febb655cc511fa2fa3de3229`, unchanged;
- dataset split: unchanged;
- HOLDOUT access count: 0;
- Support retraining: no;
- Slip/Terrain modification: no;
- output: `simulation/outputs/support_failure_mode_audit/summary.json`;
- event table: `simulation/outputs/support_failure_mode_audit/event_diagnostics.csv`;
- representative traces: `simulation/outputs/support_failure_mode_audit/traces/`.
- tests: 207 passed, 1 policy-dependent smoke skipped, 0 failed; this includes five new audit contract tests.

Generated summary, CSV, and plots are local experiment outputs and are not release artifacts.

## 13. Limitations

- There are only five recovered Support events in TRAIN/VALIDATION.
- All canonical Support positives are left-only `transition_left` / `lateral_deformable`; side and pattern subgroup robustness cannot be estimated.
- The audit can explain development misses, not the three individual HOLDOUT misses, because HOLDOUT was deliberately not reopened.
- The event-versus-negative peak AUROC is a diagnostic using frozen outputs, not a prospective acceptance result.
- Any future decision-logic development must use TRAIN/VALIDATION and then a newly created untouched final set; the existing HOLDOUT is consumed evidence.

## 14. Verdict

`SUPPORT_FAILURE_MODE_IDENTIFIED`

The frozen Support model observes the development Support events, including all seven system misses. The single identified root failure is causal Terrain-gate suppression immediately around Support onset. This result does not make the integrated architecture ready, does not authorize sensor augmentation, and does not change `CONTINUOUS_SLIP_REFLEX_PROMISING`.
