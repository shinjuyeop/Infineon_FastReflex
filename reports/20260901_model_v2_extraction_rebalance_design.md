# Model V2 Extraction Rebalance Design

## 1. Purpose

This milestone freezes the smallest justified change after `MODEL_V2_INTERNAL_FAILURE_AUDIT_ACTIONABLE`: rebalance delayed-Support positive endpoint extraction and optimizer exposure, without changing the physical corpus, target semantics, normalizer, negatives, HNM, model, hyperparameters, or runtime decision.

The verdict is:

```text
MODEL_V2_EXTRACTION_REBALANCE_DESIGN_READY
```

No training, checkpoint write, model inference, HNM replay, normalizer fit, or simulation occurred.

## 2. Starting state

The repository started clean at `ff058ae30e7097097a4d76f5b50fdc182c12be09`, equal to `origin/main`, with commit subject `Audit Model V2 internal failures`.

| Protected item | SHA-256 / state |
|---|---|
| Model V2 candidate freeze | `edb16b7d96fb38e680e36dcca5ecd7b1c1682ba410b94d21f40c2a38d4ed8725` |
| Model V2 normalizer | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` |
| Model V2 checkpoint, seed 20260828 | `dd6c8581161963265d4323b8316f01367e359357673f0596faaa2a27051771c8` |
| Model V2 checkpoint, seed 20260829 | `8e6709da112845840aae0094dd997fad4ee7f9d8256a2ee0fc5e9a0df3b724a0` |
| Model V2 checkpoint, seed 20260830 | `811f486c1bd47f91a854fdbd004b8408a5f00bfaa83a22ce91608de1d3b54c42` |
| Model V2 dataset freeze | `fd81b647051b63e7b76ab72c6dedd616cf06d752fcbc1db31ba02f8aebe68744` |
| Model V2 manifest | `7a036d3485bb19a3570a3dd4a41cc990375028f7e153e85525bd4bf19cff8b25` |
| Generalization VALIDATION V2 inference | false |
| Generalization HOLDOUT | 36 runs; guard count 0; unopened |
| Unified HOLDOUT | not reopened; no new inference |

Model V1 remains restorable under freeze SHA `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`.

## 3. Evidence boundary

The deterministic dry-run opened only the authorized effective TRAIN composition: Unified TRAIN 152 plus valid V2_TRAIN 290, exactly 442 runs under identity SHA `0466ea84871d178856ffb10d8b1c0cec730286b35b589c73ff1b5fb1065aa5ab`. It resolved physical metadata, positive and negative endpoints, partitions, and masks.

It did not open V2_VALIDATION waveforms, evaluate a model, access Generalization VALIDATION, open Generalization HOLDOUT, or generate data. Previously observed V2_VALIDATION results motivate the category-level hypothesis but no validation run ID, waveform value, or probability selected an endpoint.

## 4. Baseline V2 preservation

`model_v2_data_only_gru20_20260901` remains frozen and restorable at its original path. The design neither overwrites nor mutates its normalizer, checkpoints, candidate freeze, training record, HNM provenance, or evaluation record.

The new future candidate has the separate predeclared identity `model_v2_extraction_rebalanced_gru20_20260901`. The baseline and future candidates therefore remain directly comparable.

## 5. Failure-audit rationale

The audit found that all six Slip failures were strong early responses, five inside future-Slip precursor regions and one 5 ms before the frozen precursor. It did not establish missing signal, a broad `.20 m/s` extraction deficit, a right-side observability failure, or an HNM imbalance.

The remaining localized weakness was delayed Support. V2_VALIDATION was Concrete `3/3` and Marble `0/3`; the Marble trace reached about `0.99785` at I1 but stayed at or above `0.99` for only 3 consecutive milliseconds. Effective TRAIN had 18 delayed-Support runs, yet only 13 contributed fit positives and the extractor supplied isolated endpoints rather than persistence-length neighborhoods. That evidence supports a positive-extraction experiment and no broader intervention.

## 6. Why no new data

The physical interval is already present in both sources: nine valid V2_TRAIN delayed-Support runs from Concrete and nine from Marble, all with usable I1, established Support, and `I1→Support = 56 ms`. The failed Marble event-local waveform also had an exact fitted-TRAIN match in the prior audit. More simulation would confound physical coverage with endpoint exposure before the latter is tested.

## 7. Why no architecture change

The same 11,010-parameter GRU solved ordinary Support `24/24`, right-only Support `12/12`, and Concrete delayed Support `3/3`. All audited failures reached high probability. Longer history, LSTM, a larger GRU, new sensors, threshold change, and persistence change remain unjustified. The next experiment keeps Pelvis IMU6, causal 80D, GRU20 hidden32, one unidirectional layer, `0.99`, and 5 ms.

## 8. Current extraction exposure

The current extractor resolves 3,061 initial positives across fit and monitor. The run-level partition places 2,424 in optimizer fit and 637 in the internal monitor.

| Fit-positive role | Current |
|---|---:|
| Slip | 1,680 |
| Ordinary Support | 640 |
| Delayed Support | 104 |
| All Support | 744 |
| Total | 2,424 |

Delayed Support has 18 eligible TRAIN runs, but only Concrete `7/9` and Marble `6/9` currently reach fit, each with eight fit endpoints. Thus the current fit count is Concrete 56 plus Marble 48 = 104.

## 9. Rebalance design principles

The frozen intervention follows five rules:

1. Preserve exact baseline Slip and ordinary-Support endpoint identities and partitions.
2. Apply one symmetric delayed-Support rule to every eligible source/run.
3. Give every eligible delayed-Support run a 5 ms I1-onset neighborhood, an interval neighborhood, and a Support-local neighborhood.
4. Preserve every baseline negative endpoint, Ice mask, censor boundary, and HNM rule.
5. Keep the change small enough for clean attribution: fit positives increase only 166, or 6.85%.

The minimum grouping is `Support × ordinary/delayed × source terrain`. No Slip-side group or family sampler is added because the dry-run preserves every Slip stratum exactly.

## 10. Delayed-Support extraction

Eligibility is `valid V2_TRAIN`, family `DELAYED_SAND_SUPPORT_ONSET`, usable I1, usable established Support, and all selected endpoints before censor/fall. Invalid or anchorless runs would be rejected with a recorded reason; the frozen corpus has 18 eligible and zero ineligible runs.

For each eligible run, define `M = floor((I1 + Support) / 2)` and take this exact chronological, deduplicated union:

```text
I1 onset:      I1 + [0, 1, 2, 3, 4] ms
Interior:       M + [-2, -1, 0, 1, 2] ms
Support local:  S + [0, 1, 2, 3, 4] ms
```

The delayed-Support cap is exactly 15 endpoints per run. If physical interval shortening ever causes overlap, endpoints are deduplicated and chronologically truncated to 15 after causal/censor filtering; such a run must still pass the 5 ms I1-neighborhood guarantee or training must fail closed.

All 15 delayed-Support endpoints are assigned to optimizer fit. This positive-role-only override is what ensures all 18 eligible runs contribute to the optimization objective while preserving every negative endpoint and its existing fit/monitor assignment.

## 11. Source balance

Concrete and Marble use identical offsets, cap, eligibility, causal filtering, and fit assignment. There is no source-specific quota and no use of source validation performance in selection.

| Source | Eligible runs | Current fit-represented | Current fit positives | Proposed fit-represented | Proposed fit positives |
|---|---:|---:|---:|---:|---:|
| Concrete | 9 | 7 | 56 | 9 | 135 |
| Marble | 9 | 6 | 48 | 9 | 135 |

Both sources receive exactly 15 fit endpoints per eligible run. Projected counts differ by zero because all current intervals are usable and have the same 56 ms length.

## 12. Causal positive neighborhoods

The `I1+[0..4]` endpoints expose exactly five consecutive causal training decisions, matching but not tuning the already-frozen 5 ms runtime persistence. The midpoint neighborhood represents evolving causal sensor history approximately halfway through the 56 ms I1→Support interval. The `Support+[0..4]` neighborhood represents the established boundary locally.

I1 and Support are privileged offline label anchors only. A sample ending at `t` still contains exactly the 20 ms sensor prefix `[t-19, t]`; future IMU, centered context, time-to-Support, and Support distance are absent. The causality dry-run found zero invalid endpoint histories.

## 13. Pre-I1 negative preservation

No proposed delayed-Support positive precedes I1. All current pre-I1 staged/benign Sand negatives retain the same run, endpoint, role, and fit/monitor identity. The design changes neither static-entry semantics nor `training_negative_candidates` behavior.

The dry-run found zero pre-I1 delayed-Support positives and zero positive/negative endpoint collisions.

## 14. Ice precursor-mask preservation

Ice supervision is unchanged:

- loaded exact-Ice 30–50 mm future-Slip precursor is masked from ordinary negative mining outside canonical Slip positives;
- fully observed benign precursor release remains eligible negative;
- censored precursor remains masked;
- I1-active states are never ordinary negatives.

The exact per-run mask identity SHA is `32bae2d81b05709771545b375dd6cffb95d2dfc17c2808a7adf3a7f70174c35a`. It covers 41,479 future-Slip precursor samples, 1,734 censored-precursor samples, and 68,388 I1-positive samples. None is relabeled to suppress early Ice alerts.

## 15. Slip extraction preservation

The proposal retains exact baseline Slip endpoint identities and fit/monitor assignments, including the baseline union-cap result for the one run that contains both Support and Slip. There is no `.20 m/s` oversampling, right-side quota, mirroring, or validation-driven protection.

Event side is computed from physical Slip per-foot samples, not a combined run-level side. Immediate/delayed uses the V2 physical progression classification; historical Unified Ice Slip is the canonical immediate baseline.

| Slip cell | TRAIN runs | Fit-represented | Current fit windows | Proposed fit windows | Change |
|---|---:|---:|---:|---:|---:|
| 0.20 m/s | 37 | 29 | 435 | 435 | 0% |
| 0.25 m/s | 73 | 58 | 870 | 870 | 0% |
| 0.30 m/s | 31 | 25 | 375 | 375 | 0% |
| Left-only | 13 | 9 | 135 | 135 | 0% |
| Right-only | 8 | 5 | 75 | 75 | 0% |
| Bilateral | 120 | 98 | 1,470 | 1,470 | 0% |
| Immediate | 51 | 39 | 585 | 585 | 0% |
| Delayed | 90 | 73 | 1,095 | 1,095 | 0% |

The proposed Slip-window distribution is therefore exactly unchanged.

## 16. Rare-positive protection

The only protection is generic and metadata-driven: every eligible delayed-Support TRAIN run contributes the same 15 optimizer endpoints. It is not a per-run exception; eligibility and endpoints depend only on split validity, scenario family, source-independent physical I1/Support anchors, and causal boundaries.

No broader `scenario × source × speed × side` quota is introduced. The dry-run proves the current Slip cap does not erase any additional cell under the proposed policy, so optional Slip-side protection remains off.

## 17. Hard-negative preservation

All 32,209 initial negative identities remain exact, including 25,585 fit and 6,624 monitor endpoints. Their identity SHA is `392c1fda06953135bfed9a97c7cabb3915d4c50fdf9ff11b2ac8e6550448936c`.

| Fit-negative role | Current | Proposed | Change |
|---|---:|---:|---:|
| Hard normal | 3,206 | 3,206 | 0% |
| Ice benign | 1,629 | 1,629 | 0% |
| Benign near-threshold Ice | 193 | 193 | 0% |
| Staged Sand benign | 1,672 | 1,672 | 0% |
| Speed Sand benign | 2,257 | 2,257 | 0% |
| Other confirmed benign | 16,628 | 16,628 | 0% |

No critical negative family disappears or changes count.

## 18. HNM preservation

Future HNM remains exactly three effective-TRAIN-only rounds, 1 ms replay, at most 12/run/round, at least 30 ms spacing, exact-endpoint duplicate exclusion, and the existing forbidden masks. No HNM replay occurred here. Future training must record the same HNM policy and reject any forbidden-mask violation.

## 19. Normalizer reuse

The effective physical TRAIN corpus is unchanged, so the future experiment reuses the frozen V2 per-channel z-score normalizer with SHA `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a`. A new fit would introduce an unnecessary second variable. Normalizer fits in this milestone: zero.

## 20. Before/after dry-run counts

### Positive budget

| Partition / role | Current | Proposed | Difference |
|---|---:|---:|---:|
| Fit Slip | 1,680 | 1,680 | 0 |
| Fit ordinary Support | 640 | 640 | 0 |
| Fit delayed Support | 104 | 270 | +166 |
| Fit all Support | 744 | 910 | +166 |
| Fit total | 2,424 | 2,590 | +166 (+6.848185%) |
| Monitor Slip | 431 | 431 | 0 |
| Monitor ordinary Support | 167 | 167 | 0 |
| Monitor delayed Support | 39 | 0 | -39 |
| Monitor total | 637 | 598 | -39 |
| All extracted positives | 3,061 | 3,188 | +127 |

### Support comparison

`Runs` is the physical TRAIN-run count. `Fit runs` counts runs contributing optimizer windows. Per-run `min/median/max` includes zero for TRAIN runs assigned no fit endpoints.

| Cell | Current runs / fit runs | Current positives | Current min/median/max | Proposed runs / fit runs | Proposed positives | Proposed min/median/max |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary / Concrete | 50 / 39 | 312 | 0 / 8 / 8 | 50 / 39 | 312 | 0 / 8 / 8 |
| Ordinary / Marble | 51 / 41 | 328 | 0 / 8 / 8 | 51 / 41 | 328 | 0 / 8 / 8 |
| Delayed / Concrete | 9 / 7 | 56 | 0 / 8 / 8 | 9 / 9 | 135 | 15 / 15 / 15 |
| Delayed / Marble | 9 / 6 | 48 | 0 / 8 / 8 | 9 / 9 | 135 | 15 / 15 / 15 |
| Left-only Support | 87 / 67 | 536 | 0 / 8 / 8 | 87 / 72 | 702 | 0 / 8 / 15 |
| Right-only Support | 32 / 26 | 208 | 0 / 8 / 8 | 32 / 26 | 208 | 0 / 8 / 8 |

There are no bilateral Support events when side is taken from physical Support per-foot samples. The apparent combined bilateral run in run-level metadata contains left Support plus a later right Slip; the table correctly keeps the event roles separate.

### Frozen dry-run identities

| Identity | SHA-256 |
|---|---|
| Current positive endpoints | `c6a7bcd2dc8e8cc1ce89460af202488269c8c89363b0473e784ee9266ce0a4a9` |
| Proposed positive endpoints | `498f5d1f4419e3bfa72fc2f9649326db26f00e7a9523d9b3ecc8032436a3e0bb` |
| Negative endpoints | `392c1fda06953135bfed9a97c7cabb3915d4c50fdf9ff11b2ac8e6550448936c` |
| Masks | `32bae2d81b05709771545b375dd6cffb95d2dfc17c2808a7adf3a7f70174c35a` |
| Extraction policy | `3c7ce82ed905d932ec8f17d69d7e5edb5d79ee7602ba95ffe2a53d2407142cd2` |
| `EXTRACTION_REBALANCE_DESIGN_SHA` | `58a3949c29c8aa313cedc345dc8fa5eb0222cb85d94d31dac488adff69aed29b` |

The ignored summary is `artifacts/runs/20260901_model_v2_extraction_rebalance_design/extraction_design_summary.json`, file SHA `d4cf15c2fa2915858ae2bf900bc4df8a51406c507476169801a3b965d9f6e052`. Two full dry-run executions produced byte-identical JSON and CSV outputs. The committed config SHA is `280abbfd9da9cce948259492497090c887d238e691e0c4d8b2a0a2d52921c040`.

## 21. Contradictory-supervision audit

| Required violation | Count |
|---|---:|
| Future-Slip precursor ordinary-negative | 0 |
| Censored-precursor negative | 0 |
| I1-active or positive endpoint negative | 0 |
| Post-censor or post-fall | 0 |
| Pre-I1 delayed-Support positive | 0 |
| Future feature leakage | 0 |
| Eligible delayed run missing anchor | 0 |
| Eligible delayed run with short neighborhood | 0 |
| I1 persistence neighborhood shorter than 5 ms | 0 |

The design is not blocked by contradictory supervision.

## 22. Future batch provenance

The future trainer should keep the existing deterministic shuffled DataLoader and inverse-frequency weighted cross entropy. It must record per-epoch or aggregate positive batch exposure by Slip/Support, ordinary/delayed Support, Concrete/Marble delayed Support, speed, and physical event side. No adaptive family sampler is justified.

The positive-role fit override deliberately leaves all negative endpoint partitions unchanged. Its cost is that five runs formerly assigned to the internal monitor now contribute delayed-Support positives to fit while retaining their pre-I1 negatives in monitor; delayed-Support positive monitor count becomes zero. Thus the internal monitor is an epoch-selection instrument, not independent delayed-Support evidence. It still has 598 other positives and 6,624 negatives. Independent falsification remains the untouched, one-shot V2_VALIDATION split. Future implementation must report this provenance exactly rather than presenting the internal monitor as run-disjoint evidence for delayed Support.

## 23. Future evaluation contract

After exact implementation, future training must use the same 442 runs, normalizer, GRU, seeds `20260828/20260829/20260830`, Adam `0.001`, zero weight decay, inverse-frequency weighted cross entropy, batch 128, 40 maximum epochs, patience 6, three HNM rounds, threshold `0.99`, and persistence 5 ms.

Only after the separate candidate is frozen may V2_VALIDATION be evaluated once against unchanged gates: overall Hazard recall `>=0.90`, Slip `>=0.95`, Support `>=0.85`, confirmed no-hazard specificity `>=0.95`, Ice-benign specificity `>=0.95`, premature rate `<=0.10`, Slip p95 latency `<=+40 ms`, Support p95 established latency `<=+50 ms`, and the existing staged-Sand, speed-Sand, and right-Support gates. The Ice precursor-aware view remains secondary and separate.

The direct falsifiable delayed-Support comparison remains Concrete `3/3`, Marble `0/3` for the baseline V2 candidate. This design makes no performance claim.

## 24. External evidence protection

Generalization VALIDATION has received no V2 inference. Generalization HOLDOUT remains 36 runs with guard count zero, no waveform opening, and no inference. Unified HOLDOUT was not reopened. The extraction-rebalanced candidate must first pass its frozen internal decision; external evidence remains outside this milestone and the next internal-training milestone.

## 25. Limitations

- The 18 delayed-Support runs provide only one unique event-local waveform per source, so endpoint rebalance tests temporal exposure, not physical diversity.
- Source symmetry equalizes window counts but cannot create new source dynamics.
- The delayed-positive fit override reduces subtype coverage in the internal monitor and creates cross-role run overlap for five runs; the untouched external V2_VALIDATION split is therefore essential.
- This intervention is not designed to remove physically supported early Ice alerts. If the future Slip gate still fails only because of frozen-primary versus precursor semantics, that requires a separate readiness/metric-semantics decision, not negative relabeling.
- The design has not yet been implemented in canonical training code. The future training milestone must implement it and prove endpoint hashes match before any optimizer step.

## 26. Verdict

```text
MODEL_V2_EXTRACTION_REBALANCE_DESIGN_READY
```

The exact policy is frozen, all 18 eligible delayed-Support TRAIN runs receive symmetric 15-window fit exposure, three persistence-length causal neighborhoods are explicit, ordinary Support and Slip are unchanged, negative/mask identities are unchanged, the positive budget change is practical, contradiction counts are zero, and no external evidence was consumed.

Counters are all zero: optimizer steps, checkpoint writes, normalizer fits, HNM rounds, model inferences, threshold/persistence/architecture/seed searches, and simulation runs.

## 27. Recommended next milestone

```text
MODEL_V2_EXTRACTION_REBALANCED_TRAINING
```

That milestone should implement and hash-check this design, reuse the V2 normalizer, train the separate candidate with the unchanged recipe, freeze it, and evaluate V2_VALIDATION once. It must still not evaluate Generalization VALIDATION or open Generalization HOLDOUT. Training has not started here.
