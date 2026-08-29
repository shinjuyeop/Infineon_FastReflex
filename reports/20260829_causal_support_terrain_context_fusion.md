# Causal Support Terrain context fusion

Milestone: `CAUSAL_SUPPORT_TERRAIN_CONTEXT_FUSION`  
Date: 2026-08-29 (Asia/Seoul)  
Primary verdict: `CAUSAL_SUPPORT_TERRAIN_FUSION_NOT_SUPPORTED`

## 1. Purpose

This experiment asked whether a causal post-detection Terrain context could remove the previously identified Support decision failure without changing the Support detector, Terrain recognizer, oracle, model inputs, sensors, or thresholds. The required architecture was detection first and interpretation second.

## 2. Prior gating failure

The preserved audit reproduced TRAIN Support recall 29/36 and VALIDATION 12/12. All seven TRAIN misses had raw Support peak 0.960–0.994 and raw persistence 16–63 ms, but the historical global Terrain gate suppressed output. The primary prior verdict was `GATING_SUPPRESSION`, not weak Support observability.

Prior audit commit: `e3e475db207775def9da1cf4f367c099e76aed6d` (`Identify support gating failure mode`). The canonical report is `reports/20260829_support_failure_mode_audit.md`.

## 3. Protected detector/model state

The frozen Support branch remained P3, Pelvis IMU6-derived 60-D input, one-layer GRU, 20 ms history, threshold 0.94 and five consecutive 1 kHz samples. No checkpoint, normalizer, feature, hidden-state handling, threshold, persistence, oracle, or timing window changed.

| Protected Support artifact | SHA-256 |
| --- | --- |
| TRAIN normalizer | `db04b0fcb134141e91acaac66111835cfdede25a516201788c5f7fd42ca43b20` |
| seed 20260828 | `26d2bbbffd65dd5cf06424366754749d485511c46791632f1fe2062ae046aa21` |
| seed 20260829 | `d3e023ee89240283eb5f6d8f76a9bd1365dbca97de64b1ec76d06a23d44cf586` |
| seed 20260830 | `0781b7ffc6887ad96e5ade7f482fe0d97b6e643322f6dd072767b4f15ab176dc` |

The protected Terrain FSR4/MLP/50 ms ensemble and Slip freeze artifact `df0a232ec242283ef8b25c59421cebde982a7a93febb655cc511fa2fa3de3229` were also unchanged.

## 4. Detection versus context separation

The new evaluation computes:

```text
frozen GRU probability
    -> score >= 0.94 for 5 ms, independent of Terrain
    -> RAW_SUPPORT_ALERT
    -> causal Terrain context authorization
    -> SUPPORT_RISK
```

Raw persistence is calculated once with an always-active gate. F0 and F1 consume the same immutable raw alert trace. TRAIN and VALIDATION raw event clocks, scores, threshold crossings, and persistence outputs were bit-identical across both policies. Terrain never resets the raw counter or changes a GRU input/state.

## 5. Frozen Terrain interface audit

The runtime `TerrainGateTrace` exposes one global held `state`, prediction update samples, predicted class IDs, class probabilities, first-target-valid sample, and clean-event count. It does not expose prediction foot identity or per-foot terrain memory.

The protected recognizer internally schedules LEFT_ONLY clean touchdowns, but its frozen consumer interface discards the foot field. Inferring a general left/right memory from simulator terrain truth or adding supporting-foot state would change the interface. Therefore:

`PER_FOOT_TERRAIN_CONTEXT_UNAVAILABLE_WITH_FROZEN_INTERFACE`

F2 was not implemented or evaluated.

## 6. F0 historical exact gate

F0 authorizes `SUPPORT_RISK` only while the current global frozen Terrain output is `SAND`. Unlike the historical implementation, raw Support persistence remains continuous for comparison; only post-detection authorization uses the current global state.

F0 reproduced the exact seven TRAIN context suppressions and 29/36 recall. It therefore cannot be selected even though its VALIDATION metrics alone pass.

## 7. F1 recent-SAND causal grace

F1 uses the predeclared 50 ms grace without a sweep. Every 1 kHz sample for which the held frozen output remains `SAND` refreshes a last-Sand clock. Current SAND is authorized; after a non-Sand update, the context remains true through exactly 50 ms and is false at 51 ms. Simulation reset and expiry clear memory; future output is never read.

F1 rescued two recent SAND→ICE suppressions but not five events whose last Sand sample was roughly 0.59 s earlier. It also authorized earlier raw Support-like alerts during stale-Sand grace, creating unacceptable premature system events.

## 8. F2 support-foot memory

F2 required causal prediction foot identity, independently updated left/right memory, and a canonical runtime supporting-foot state. None is present in the frozen `TerrainGateTrace` interface. LEFT_ONLY configuration alone cannot provide the missing right-foot memory or generic foot-associated output contract.

F2 status: unavailable. No exact contact or simulator terrain truth was injected into fusion to manufacture it.

## 9. Development corpus

The experiment reused only the existing TRAIN and VALIDATION event corpus and cached development Terrain outputs:

- TRAIN Support events: 36;
- VALIDATION Support events: 12;
- combined recovered/fall diagnostics: 5/43;
- TRAIN/VALIDATION Support negatives: 116/40;
- Sand benign negatives: 36/12;
- hard-ground controls: 8/4.

No simulation or dataset was regenerated, and no model training or hard-negative mining ran.

## 10. Raw Support replay parity

Raw Support valid detection was 36/36 TRAIN and 12/12 VALIDATION before context authorization. A run-local SHA-256 over all development endpoints/probabilities is retained in the ignored summary for provenance; checkpoint and normalizer hashes, rather than floating-point replay bytes, are the authoritative frozen identities.

Raw premature onsets also existed in 18/36 TRAIN and 6/12 VALIDATION event runs. This was not itself a system false reflex: F0 suppressed those early events. F1 converted them into system prematures because its recent-Sand memory remained active.

## 11. Terrain churn around Support

Across 48 Support events in `t_support ±100 ms`:

| Pattern | Event count |
| --- | ---: |
| SAND→MARBLE | 0 |
| SAND→ICE | 2 |
| SAND→CONCRETE | 0 |
| UNKNOWN transition | 0 |
| stable SAND for complete ±100 ms | 10 |
| other, mainly non-SAND→SAND or no within-window transition | 36 |

Terrain state at Support onset was SAND 41, MARBLE 5, ICE 2. The last-Sand-to-Support gap distribution was min/p10/median/p95/max 0/0/0/593.25/595 ms. Thus 50 ms grace can cover the two recent ICE churn cases but cannot cover the five long-gap MARBLE cases without violating the frozen bounded policy.

## 12. TRAIN historical miss rescue

| Policy | TRAIN recall | Historical seven rescued | Remaining | Context suppression |
| --- | ---: | ---: | ---: | ---: |
| F0 | 29/36 = 80.56% | 0/7 | 7 | 7/36 = 19.44% |
| F1, 50 ms | 31/36 = 86.11% | 2/7 | 5 | 5/36 = 13.89% |
| F2 | not implementable | — | — | — |

F1 reduced suppression by only 28.6%, not dramatically, and introduced 18/36 TRAIN premature system events. All 18 were classified as stale-Sand-context authorization.

## 13. Validation F0

| Metric | F0 result | Gate |
| --- | ---: | --- |
| Support recall | 12/12 = 100% | pass |
| Sand benign specificity | 12/12 = 100% | pass |
| Premature event-run rate | 0/12 = 0% | pass |
| Context suppression | 0/12 = 0% | pass |
| Fusion latency median / p95 | -8.5 / -1 ms | pass |
| Hard-ground specificity | 4/4 = 100% | pass |

F0 was nevertheless ineligible because it reproduced all seven known TRAIN suppression failures. The selection rule explicitly prohibited selecting it in that case.

## 14. Validation F1

| Metric | F1 result | Gate |
| --- | ---: | --- |
| Support recall | 12/12 = 100% | pass |
| Sand benign specificity | 12/12 = 100% | pass |
| Premature event-run rate | 6/12 = 50% | **fail**, required <=5% |
| Context suppression | 0/12 = 0% | pass |
| Fusion latency median / p95 | -8.5 / -1 ms | pass |
| Hard-ground specificity | 4/4 = 100% | pass |

All six premature events were stale-Sand-context authorizations while the current Terrain output was ICE. Their event-relative times were min/p10/median/p95/max -567/-566.5/-565.5/-565/-565 ms. F1 therefore trades a small suppression reduction for a large causal false-authorization problem.

## 15. Validation F2

F2 was not evaluated. The missing foot-associated frozen output is an interface limitation, not a negative F2 performance result.

## 16. Candidate selection

No policy was selected.

- F0: all numerical VALIDATION gates pass, but predeclared historical-suppression exclusion applies.
- F1: fails the predeclared premature gate by 45 percentage points.
- F2: unavailable with the frozen interface.

No grace duration, threshold, persistence, feature, or alternative policy was added after observing these results.

## 17. Freeze provenance

The complete experiment config—including F0, F1 50 ms, F2 availability rule, selection gates and holdout gates—was frozen at `configs/experiment/20260829_causal_support_terrain_context_fusion.yaml` before replay.

Because no policy passed selection, no selected-policy freeze artifact was created. This is fail-closed behavior, not an incomplete run.

Repository regression result: 218 tests passed, one policy-dependent simulation smoke was skipped, and zero tests failed. This includes 11 new raw-invariance, F0/F1, F2-interface, event, selection, guard, hash and regression tests.

## 18. One-shot holdout

HOLDOUT was not opened. Guard count remained 0 because the opening condition—at least one policy passing all VALIDATION selection rules—was not satisfied. No HOLDOUT waveform, per-run score, event ID, Support outcome, or Terrain context trace was read during this milestone.

## 19. Support recall

F1 preserved VALIDATION recall but did not supply acceptable system behavior due premature outputs. Development recall alone cannot justify a post-fusion policy. No HOLDOUT recall is claimed.

## 20. Benign Sand specificity

Both policies retained 100% Sand-benign specificity on TRAIN and VALIDATION. The F1 failure is not a benign Sand false-positive run; it is premature authorization within eventual Support-event runs.

## 21. Context suppression

F0 suppression was TRAIN 7/36 and VALIDATION 0/12. F1 was TRAIN 5/36 and VALIDATION 0/12. F1's 50 ms bound correctly rescued only events whose last Sand evidence was recent enough; it could not cover the five approximately 0.59 s gaps.

## 22. Premature alerts

The raw continuous detector produced early Support-like alerts in half of development Support-event runs. F0 current-state context blocked them. F1's grace authorized all 18 TRAIN and six VALIDATION early alerts while the current state was ICE.

This is the central fusion failure: post-detection context must preserve true event evidence without turning earlier raw dynamics into system reflexes. A simple recent-Sand latch does not meet both requirements.

## 23. Raw versus fusion latency

VALIDATION raw latency was median/p95 -8.5/-1 ms for the common frozen trace. Valid F0/F1 fusion detections had the same distribution because context was already active at those valid onsets. The distinction remains explicit even though the accepted-event timings match.

F1's invalid premature system events occurred about 565 ms before Support and are excluded from valid fusion latency.

## 24. Hard-ground behavior

Both F0 and F1 produced zero false reflexes on eight TRAIN and four VALIDATION hard-ground controls. Actual frozen Terrain outputs, not exact terrain truth, drove context.

## 25. False-positive mechanism

There were no negative-run system false positives, so stale context, Terrain misclassification, raw Support false alert, context-latch permissiveness and unknown negative mechanisms all had count zero.

There were six VALIDATION and 18 TRAIN premature event-run false authorizations. Every one was `stale_sand_context`: raw alert was present, recent-Sand grace was active, but the current global Terrain output was ICE.

## 26. Causal memory cost

F0 requires no extra state. F1 requires one last-Sand sample clock and a fixed 50 ms expiry comparison. It is causal and bounded, but small state cost does not compensate for failed decision behavior. F2 would require two foot memories and supporting-foot provenance unavailable today.

## 27. Asymmetric reflex architecture implication

Continuous Slip remains untouched and supported by its prior branch evidence. Continuous raw Support detection is technically valid and should remain separated from interpretation in future research, but this milestone did not identify an acceptable causal Sand authorization policy.

Therefore `TERRAIN_GATED_SUPPORT_PERSISTENCE` is not newly endorsed, but it is also not retired in favor of F1. The conceptual `SUPPORT_DETECTOR_CONTINUOUS + TERRAIN_CONTEXT_POST_FUSION` direction remains unvalidated.

## 28. Limitations

- Frozen Terrain exposes no per-foot provenance, preventing the physically preferred F2 test.
- The predeclared 50 ms F1 policy intentionally cannot bridge five roughly 0.59 s Sand gaps.
- Raw premature dynamics show that context cannot merely be widened without safety cost.
- HOLDOUT was correctly preserved, so no fusion generalization claim is available.
- Results are bounded to current MuJoCo scenarios, G1 policy, Terrain producer, Support oracle, and frozen detector.

## 29. Verdict

`CAUSAL_SUPPORT_TERRAIN_FUSION_NOT_SUPPORTED`

F1 does not satisfy the VALIDATION premature gate, F0 is disqualified by the known suppression it reproduces, and F2 is unavailable. No Support/Terrain/Slip model or sensor was modified, and no HOLDOUT was opened.

## 30. Next recommendation

Do not tune the 50 ms grace, Support threshold, persistence, or model. If work continues in a separately authorized milestone, first define whether the protected Terrain producer may expose causal touchdown-foot provenance and a runtime supporting-foot contract, then build a new validation and untouched final-set protocol around that interface. Sensor augmentation is not justified by this result, and no Recovery or final architecture freeze should begin.
