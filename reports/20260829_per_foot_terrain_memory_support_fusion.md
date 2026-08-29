# Per-foot Terrain Memory Support Fusion

## 1. Motivation

`PER_FOOT_TERRAIN_MEMORY_SUPPORT_FUSION` tested whether bilateral touchdown-foot provenance and independently held Terrain memory can remove the known global-state overwrite in the frozen Support branch. The experiment started on `main` with `HEAD = origin/main = c91eed6d8085b7f4c7216ebee5e5585fd99a66c3` and a clean worktree.

The primary verdict is:

`PER_FOOT_TERRAIN_SUPPORT_FUSION_NOT_SUPPORTED`

Per-foot memory removed context suppression, but it also authorized the frozen raw Support detector approximately 565 ms too early in half of the VALIDATION Support runs. Neither PF1 nor PF2 passed the fixed premature gate, no policy was selected, and Support HOLDOUT remained sealed with access count 0.

## 2. Why global Terrain state failed

Historical F0 used one held Terrain class. A later touchdown overwrote the earlier foot's SAND context even when the earlier foot could still carry load. Frozen Support raw evidence existed in all seven historical TRAIN misses, but the global state at Support onset was MARBLE in five and ICE in two, producing seven context suppressions.

The bounded 50 ms F1 grace rescued only two of seven. Extending that timer was prohibited because the five remaining gaps were 590–595 ms and F1 already produced premature authorization.

## 3. Existing foot provenance evidence

The existing clean-touchdown pipeline already records `foot` on every event row, slices only that touchdown foot's FSR4, and knows the event foot before the 50 ms window completes. This scheduler provenance is causal. Exact simulator terrain identity remains scheduler/label-only and is not copied to runtime output.

The canonical output now preserves:

```text
TerrainPrediction(
    class_id,
    probabilities,
    prediction_timestamp,
    touchdown_foot,
)
```

`touchdown_foot` is provenance, not an MLP tensor feature.

## 4. LEFT_ONLY vs BILATERAL_SHARED

| Scheme | Physical set with Pelvis IMU6 | Historical update coverage | Historical delay median/p95 | Per-foot memory capability |
|---|---:|---:|---:|---|
| LEFT_ONLY | FSR4 + IMU6 = 10 | 126/144 | 1114.5/1238 ms | No generic right-foot update |
| BILATERAL_SHARED | FSR8 + IMU6 = 14 | 144/144 | 922/1238 ms | Yes |

The four additional FSR channels solve bilateral provenance coverage, but this experiment shows that provenance coverage alone does not provide safe Support authorization.

## 5. Bilateral artifact audit/reconstruction

The prior report and metrics contained the exact BILATERAL_SHARED validation result, but only the selected LEFT_ONLY checkpoints had been retained. Fair deterministic reconstruction was possible because the 144-run Terrain dataset, events, run-disjoint TRAIN/VALIDATION split, FSR4 schema, MLP architecture, 50 ms window, seeds 17/29/43, optimizer, batch size, epoch limit and patience all remained available with frozen hashes.

Status:

`BILATERAL_SHARED_RECONSTRUCTED_FROM_FROZEN_PROTOCOL`

No Support result and no Terrain HOLDOUT waveform was used. Reconstruction reproduced every frozen validation signature field at tolerance `1e-12`, including all seed epochs/confusion matrices, normalization, ensemble confusion matrix and left/right breakdown.

| Metric | Historical | Reconstructed |
|---|---:|---:|
| TRAIN events | 352 | 352 |
| VALIDATION events | 112 | 112 |
| Three-seed macro F1 mean | 0.928414 | 0.928414 |
| Worst-class recall mean | 0.857143 | 0.857143 |
| Ensemble macro F1 | 0.928230 | 0.928230 |
| Left macro F1 | 0.918506 | 0.918506 |
| Right macro F1 | 0.927701 | 0.927701 |

Generated, Gitignored hashes are seed 17 `5a4b5824f7cc658ce002d83f90b8f1bb4c8f6a6b11ea72214d8d1f803137c8ca`, seed 29 `b0cbc4562c948425b029a9cdaed8b52753cc6b389f4e6567f1ead1d3c3b91e58`, seed 43 `c40ca1b2e778da2693b719da2409952c55fb4eefe9b24b37d8fd20713827db07`, and normalizer `badbd542ab47cffc1afceab531cf44c077d827e9bdb79cd11caf5031375f28bc`. Terrain reconstruction holdout access was 0.

## 6. Runtime TerrainPrediction schema

At each valid clean LEFT or RIGHT touchdown, the shared model receives exactly one `50 × 4` FSR window. It emits four probabilities and a class ID; the scheduler appends the causal prediction timestamp and touchdown foot. No side ID, scenario ID, target terrain, exact geom, fall state, Support state or future sample enters the tensor or output.

## 7. Per-foot memory semantics

Memory initializes to `UNKNOWN/UNKNOWN`. A valid prediction updates only the same touchdown foot and cannot overwrite the opposite foot. Each value is held until the next valid prediction for that same foot. There is no timer expiry. An absent, ambiguous, incomplete, invalid or censored inference produces no update and therefore preserves the previous valid value. Simulation reset clears both memories.

## 8. Supporting-foot rule

The final runtime policy does not use stored `loaded_contact` or exact simulator contact. Current FSR totals determine load:

```text
left_loaded  = sum(left_FSR4)  > 1e-6 N
right_loaded = sum(right_FSR4) > 1e-6 N
```

The epsilon is the existing canonical FSR feature epsilon and is consistent with the verified virtual-FSR airborne exact-zero contract. It has no persistence or hysteresis and was frozen before validation without outcome fitting.

## 9. PF1 any-loaded-Sand

PF1 authorizes when any currently FSR-loaded foot has SAND in the same foot's memory. An airborne SAND-memory foot cannot authorize PF1.

## 10. PF2 dominant-foot-Sand

PF2 selects `argmax(left_total_FSR, right_total_FSR)`, with deterministic LEFT tie-break, and authorizes only when at least one foot is loaded and the dominant foot's memory is SAND. PF2 was the sole bounded alternative; no additional policy was introduced after validation.

## 11. Raw Support invariance

Frozen Support remained Pelvis IMU6-derived 60D, GRU, 20 ms, threshold 0.94 and five consecutive 1 kHz samples. The development raw replay SHA was `c3b43d852b31f628365126f5e264409dccc02da8b0684c582a7fa17fb538dd98`, exactly matching the prior audit. PF1 and PF2 raw timestamps were bit-identical. Terrain memory never reset probability, GRU state or raw persistence.

## 12. Historical seven misses

| Policy | TRAIN recall | Historical misses rescued | Remaining | Context suppression |
|---|---:|---:|---:|---:|
| F0 global SAND | 29/36 (80.56%) | 0/7 | 7 | 7 |
| F1 50 ms grace | 31/36 (86.11%) | 2/7 | 5 | 5 |
| PF1 any loaded SAND | 36/36 (100%) | 7/7 | 0 | 0 |
| PF2 dominant SAND | 30/36 (83.33%) | 2/7 | 5 | 6 |

PF1 materially fixed suppression. PF2 rejected five events because the dominant foot did not carry SAND memory during the valid timing region.

## 13. Long-gap five MARBLE misses

| Run | Gap | Memory L/R at `t_support` | Loaded L/R | Dominant | Supporting SAND at `t_support` | PF1/PF2 rescue |
|---|---:|---|---|---|---|---|
| evt_c_sand_f09 | 595 ms | CONCRETE/SAND | yes/no | LEFT | none | yes/no |
| evt_c_sand_f11 | 590 ms | MARBLE/SAND | yes/yes | LEFT | RIGHT | yes/no |
| evt_m_sand_f03 | 595 ms | CONCRETE/SAND | yes/no | LEFT | none | yes/no |
| evt_m_sand_f05 | 595 ms | CONCRETE/SAND | yes/yes | LEFT | RIGHT | yes/no |
| evt_m_sand_f07 | 590 ms | MARBLE/SAND | yes/no | LEFT | none | yes/no |

Only two of five have a loaded SAND-memory foot exactly at `t_support`. PF1 still obtains a valid `[-30,+50] ms` detection in all five because the SAND-memory right foot becomes loaded within the valid window. Thus per-foot memory explains the decision rescue, but not a supporting-SAND condition exactly at onset for three cases. This is an important limitation rather than evidence to relax the load rule.

## 14. TRAIN diagnostic

| Policy | Recall | Suppression | Premature | Sand benign specificity | Hard specificity | Fusion latency median/p95 |
|---|---:|---:|---:|---:|---:|---:|
| PF1 | 36/36 | 0/36 | 18/36 (50%) | 36/36 | 8/8 | -9/-1 ms |
| PF2 | 30/36 | 6/36 | 18/36 (50%) | 36/36 | 8/8 | -15/-1 ms |

Both policies authorized 31 of 116 other non-Support run groups, all Ice-target runs with a held SAND memory following a Terrain error. Overall negative specificity was 73.28%, even though the predeclared Sand-benign and hard-control subsets remained perfect.

## 15. VALIDATION PF1

PF1 achieved Support recall 12/12, context suppression 0/12, Sand benign specificity 12/12, hard-ground specificity 4/4 and fusion latency median/p95 -8.5/-1 ms. It failed because 6/12 Support runs had an additional system alert 565–567 ms before `t_support`, for premature rate 50% versus the frozen 5% maximum.

## 16. VALIDATION PF2

PF2 had the same 12/12 recall, zero suppression, perfect specified negative groups, -8.5/-1 ms latency, and the same 6/12 premature failures. Dominant-foot authorization reduced alert duration from 2850 to 2546 samples but did not eliminate any premature event-run. It therefore failed the same gate.

## 17. False-authorization taxonomy

The six premature Support-run alerts per policy were raw Support false alerts authorized while the relevant per-foot SAND context was physically plausible; their lead was roughly 565 ms, far outside `[-30,+50] ms`. No unloaded SAND-memory foot bypassed either policy.

Separately, each policy generated nine VALIDATION alerts among 40 other non-Support groups, all Ice-target runs. The recorded taxonomy is stale per-foot Terrain memory; root-cause interpretation is an erroneous SAND Terrain prediction that remained held by the deliberately non-expiring same-foot memory. This is precisely the causal-memory cost revealed by the experiment. There were no Sand-benign or hard-ground false reflexes.

## 18. Selection

| Gate | PF1 | PF2 | Requirement |
|---|---:|---:|---:|
| Support recall | 100% PASS | 100% PASS | ≥95% |
| Sand benign specificity | 100% PASS | 100% PASS | ≥95% |
| Context suppression | 0% PASS | 0% PASS | ≤5% |
| Hard specificity | 100% PASS | 100% PASS | ≥95% |
| Median/p95 latency | -8.5/-1 ms PASS | -8.5/-1 ms PASS | ≤20/50 ms |
| Premature | 50% FAIL | 50% FAIL | ≤5% |

No candidate passed every gate. `selected_policy = null`.

## 19. Freeze provenance

The experiment config, bilateral reconstruction protocol, prediction schema, memory transitions, FSR epsilon, PF1/PF2 definitions, Support hashes and selection rule were fixed before Support validation. The bilateral artifact received its own reconstruction provenance after historical Terrain validation parity. Because no PF policy passed, no policy selection artifact was created and no holdout policy was frozen.

## 20. One-shot HOLDOUT

HOLDOUT was not opened:

```text
performed = false
guard_open_count = 0
reason = no_per_foot_policy_passed_validation
```

No holdout run IDs, waveforms, Support outcomes, Terrain predictions or per-foot memories were accessed.

## 21. Support recall

PF1's validation 100% recall confirms that bilateral memory can prevent the global gate from deleting valid raw evidence. PF2 also reached 100% on VALIDATION but remained only 83.33% on TRAIN. Recall alone is insufficient because both policies created unsafe early authorization.

## 22. Sand benign specificity

Both policies preserved 12/12 VALIDATION Sand benign runs and 36/36 TRAIN Sand benign runs. Physical Sand deformation without the Support oracle did not become a system reflex in this designated hard-negative group.

## 23. Context suppression

PF1 reduced suppression from historical F0's 7/36 to 0/36 TRAIN and remained 0/12 VALIDATION. PF2 left 6/36 TRAIN suppressions but none in VALIDATION. PF1 therefore validates the global-overwrite hypothesis as one genuine failure mechanism.

## 24. Premature

Suppression removal exposed a second independent failure: the continuous raw detector also produces a strong alert roughly 565 ms before the physical Support event. Global F0 happened to suppress it; correct Sand memory authorizes it. This is not a reason to reintroduce detector-internal Terrain reset, add a timer, or tune thresholds in this milestone.

## 25. Hard-ground behavior

Concrete/Marble hard controls remained 4/4 specific in VALIDATION and 8/8 in TRAIN. However, nine Ice-target non-Support runs generated alerts after SAND misclassification memory, so perfect hard-control specificity must not be generalized to all non-Support operation.

## 26. Fusion latency

For valid detections, both policies had VALIDATION median/p95 latency -8.5/-1 ms relative to the fixed Support clock. Per-foot lookup adds no artificial 50 ms delay when memory already exists. The Terrain classifier itself still requires 50 ms after each clean touchdown.

## 27. Per-foot memory age

Memory age was retained as a diagnostic and never used as a threshold. The long-gap examples show why a timer would be both tempting and incorrect: some useful SAND provenance is about 590 ms old, while other long-held errors authorize false reflexes. The experiment precluded memory-age tuning and confirms that a bare held class lacks validity/confidence semantics for safe causal authorization.

## 28. Sensor-channel tradeoff

BILATERAL_SHARED requires FSR8 plus Pelvis IMU6, 14 unique physical channels, versus LEFT_ONLY's 10. It provides complete bilateral prediction provenance and solved PF1 suppression, but the four extra channels did not make the overall fusion policy acceptable. No Foot IMU, q/dq, torque or current was added.

## 29. Slip architecture compatibility

The continuous Slip detector and its frozen evidence were not modified. The evaluated representation remains compatible with an asymmetric design—continuous Slip immediate reflex plus continuous Support post-fusion authorization—but this Support authorization contract is not supported and must not be promoted.

## 30. Limitations

- Virtual FSR and engineering terrains are noiseless simulation proxies.
- Exact contact is still used only to schedule the established clean-touchdown producer; hardware touchdown scheduling remains unvalidated.
- Per-foot held class has no bounded confidence decay or explicit invalidation event.
- The Support raw detector's early high-score mode remains unexplained by this fusion-only study.
- No policy passed validation, so holdout generalization is unknown.
- Historical Support HOLDOUT evidence was not reused.

## 31. Verdict

`PER_FOOT_TERRAIN_SUPPORT_FUSION_NOT_SUPPORTED`

Per-foot provenance is implementable and PF1 proves that `GLOBAL_STATE_OVERWRITE` caused the seven historical suppressions. Nevertheless, both predeclared policies failed the validation premature gate catastrophically at 50%. The Support/Terrain models, thresholds, persistence, sensors and memory rules were not tuned to manufacture a pass.

## 32. Next recommendation

Stop at this result. Preserve continuous raw Support detection and the new provenance capability as research infrastructure, but do not adopt PF1/PF2 or declare `BILATERAL_FSR8_PLUS_PELVIS_IMU6_SYSTEM_CANDIDATE`. A separately authorized study must first explain whether the early raw Support mode represents a distinct physical event, an oracle-clock mismatch, or a detector false mode; it must not begin with grace/memory-duration or threshold sweeps.

Slip retraining, Recovery, E84 work and final sensor freeze were not started.
