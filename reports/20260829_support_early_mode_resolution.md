# Support Early Mode Resolution

## 1. Purpose

`SUPPORT_EARLY_MODE_RESOLUTION` investigated the frozen Support detector's approximately `t_support - 565 ms` raw alert before deciding whether it was a negative gait alias or a physical precursor. The experiment started on `main` with `HEAD = origin/main = f8e128e20fcd7ca70c6aa2553ccf2c56af823a42` and a clean worktree. Only TRAIN and VALIDATION waveforms were opened.

The cause verdict is `SUPPORT_EARLY_MODE_PHYSICAL_PRECURSOR`. The final runtime verdict is `CONTINUOUS_SUPPORT_REFLEX_NOT_SUPPORTED` because the frozen continuous detector failed the predeclared Ice non-Support specificity gate. Support HOLDOUT remained sealed.

## 2. Current Support evidence

The protected detector remained Pelvis IMU6-derived 60D, GRU, 20 ms history, threshold 0.94 and five consecutive 1 kHz samples. The established privileged clock remained `support_surface_spread >= 10 mm` for 20 ms. No Support/Terrain/Slip checkpoint, normalizer, threshold, persistence, sensor, physical oracle or model architecture changed.

The frozen raw detector retained established-event recall 36/36 TRAIN and 12/12 VALIDATION, with established latency median/p95 -9/-1 ms and -8.5/-1 ms respectively.

## 3. The -565 ms phenomenon

There were 24 development early episodes: 18/36 TRAIN Support runs and 6/12 VALIDATION Support runs. Their established-clock lead was 565–573 ms, median 566 ms (p10 565, p95 573). Each row is retained in the Gitignored audit CSV; HOLDOUT contributed no row.

## 4. Gait-period hypothesis

All 24 episodes were within the frozen ±50 ms diagnostic tolerance of one same-foot stride. The preceding left/right step period was 315–316 ms, median 315.5 ms. The measured same-foot stride was 581–582 ms, median 581.5 ms, and the absolute lead/stride mismatch was 8–16 ms, median 16 ms. The previous same-foot period before the early cycle was 588 ms. Thus the alert is strongly gait-periodic, but periodicity alone does not prove that its physical state is benign.

## 5. Premature vs true-event waveform similarity

The frozen normalized `20 × 60` premature/established pairs had median normalized L2 1.102 and cosine similarity 0.124; median per-channel temporal correlation was -0.094. Thirty-two phase/foot/touchdown-age matched TRAIN normal windows were used per alert. Their per-alert median L2 was 1.852 and cosine was -0.162. The early window is closer to the established window than to matched normal by the aggregate distance/cosine diagnostics, although channel-wise correlation is weak. This supports a repeated locomotion pattern with different amplitude/evolution rather than waveform identity.

## 6. Physical diagnostics at premature alert

At every early alert the affected LEFT support was loaded and the RIGHT was unloaded. Median values were:

| Diagnostic | Median | Range |
|---|---:|---:|
| Support spread | 6.545 mm | 6.508–6.553 mm |
| Max support displacement | 8.157 mm | 8.123–8.166 mm |
| Spread delta 1 ms | 0.0164 mm | 0.0100–0.0181 mm |
| Spread delta 20 ms | 0.872 mm | 0.730–0.900 mm |
| Spread delta 50 ms | 4.497 mm | 4.409–4.732 mm |
| Left FSR total | 387.24 N | 385.93–389.29 N |
| Right FSR total | 0 N | 0–0 N |

The loaded-foot physical-envelope exit completed 51–53 ms before the raw alert. Per-cell displacement was not stored in the existing corpus and was not reconstructed from simulator truth; spread and maximum displacement were the available exact diagnostics.

## 7. Matched benign references

References came only from TRAIN runs without an established Support event: Sand benign, Ice non-Support, hard-ground controls, and their valid pre-censor intervals. Matching required equal contact phase, the same loaded support foot and touchdown age within 50 ms. Scenario name, future outcome, fall time and Terrain identity were absent from the frozen model tensor.

## 8. TRAIN physical envelope

The diagnostic envelope used phase-aware TRAIN-only q99.5 bounds with 20 ms persistence. Sample counts were 10,184 no-support, 239,477 left-support, 251,790 right-support and 104,169 double-support. Validation was excluded from fit and fall outcome was not read. No-Support spread and positive spread-delta q99.5 were 0 in this engineered corpus; the 6.5 mm loaded-foot early state was therefore unambiguously outside the benign physical envelope.

## 9. Early-mode classification

| Class | Count |
|---|---:|
| `GAIT_ALIAS_FALSE_MODE` | 0 |
| `PHYSICAL_PRECURSOR_MODE` | 24 |
| `MIXED_OR_UNRESOLVED` | 0 |

All 24 alerts are stride-aligned, but a gait alias requires the physical state to remain inside the matched benign envelope. That condition failed in every run. The gait cycle explains when the detector responds; it does not make the state a valid hard negative.

## 10. Cause verdict

`SUPPORT_EARLY_MODE_PHYSICAL_PRECURSOR`

The approximately -565 ms mode occurs on a previous loaded-foot cycle with sustained heterogeneous support deformation. It is not justified as a normal-gait negative.

## 11. Branch decision

Branch B was executed. The task did not teach the frozen detector that these physically abnormal windows were normal and did not enter HNM. Exactly I1/I2/I3 were evaluated as simulator-only incipient reference candidates.

## 12. HNM protocol if used

HNM was not used because the cause verdict selected Branch B. The canonical evaluator nevertheless implements and tests the frozen Branch A/C contract—TRAIN-only endpoints, alias priority, K=16, 30 ms separation and established positive-region exclusion—without executing or fitting it to this result.

## 13. Round 0

Round 0 is the unchanged frozen ensemble. Relative to the established clock it had TRAIN recall 100%, premature 50%, Sand benign specificity 100%, Ice non-Support specificity 37.5%, hard specificity 50%, and negative-time alert fraction 0.93%.

## 14. HNM1

Not performed; Branch B forbids relabeling the identified physical precursor as a negative.

## 15. HNM2

Not performed.

## 16. HNM3

Not performed.

## 17. TRAIN continuous replay

Using the selected incipient clock as the early-warning boundary removes semantic premature events (0/36), but it does not solve continuous specificity. Frozen raw Support alerts occurred in 45/72 TRAIN Ice non-Support runs and 4/8 hard controls. Established Support recall remained 36/36, Sand benign false alerts were 0/36, and negative-time alert fraction was 0.931%.

## 18. Incipient reference if used

All three fixed candidates passed the physical-reference validation gates:

| Candidate | TRAIN-only q99.5 | VALIDATION coverage | Sand/Ice/hard FP | Median lead to established |
|---|---:|---:|---:|---:|
| I1 spread derivative | 0 | 12/12 | 0/0/0% | 1517.5 ms |
| I2 spread delta 20 ms | 0 | 12/12 | 0/0/0% | 1517.5 ms |
| I3 normalized spread + derivative | 0 | 12/12 | 0/0/0% | 1517.5 ms |

I1 is the diagnostic selection by simplicity. The zero threshold is not tuned against Support or fall runs; it follows from exactly zero heterogeneous spread in the balanced/no-Support TRAIN cohort. `t_incipient_support` is preserved alongside the unchanged established clock, but this simulation-specific degeneracy is a major limitation.

## 19. VALIDATION

With I1 defining the precursor-active interval, the frozen detector achieved Support recall 12/12, incipient-premature 0/12, Sand benign specificity 12/12, hard specificity 4/4, negative-time alert fraction 0.951%, and established latency median/p95 -8.5/-1 ms. It produced false alerts in 15/24 Ice non-Support runs, giving specificity 37.5% versus the required 90%.

## 20. Threshold selection

No detector threshold search was run. Branch B first evaluates the frozen detector, and it failed a primary negative-group gate at threshold 0.94. The HNM3 validation grid 0.50–0.99 is applicable only after Branch A/C retraining, which was not authorized by the physical-precursor cause verdict.

## 21. Continuous specificity

| Split | Sand benign | Ice non-Support | Hard ground | Negative alert fraction |
|---|---:|---:|---:|---:|
| TRAIN | 36/36 (100%) | 27/72 (37.5%) | 4/8 (50%) | 0.931% |
| VALIDATION | 12/12 (100%) | 9/24 (37.5%) | 4/4 (100%) | 0.951% |

The validation failure is not a Sand-benign deformation problem. It is broad cross-triggering on Ice trajectories when Terrain no longer hides the detector.

## 22. Ice non-Support analysis

Fifteen of 24 VALIDATION Ice runs generated at least one sustained raw Support alert. These runs contain physical Slip but no Support oracle. Because Terrain is intentionally absent from the detector, this is a genuine continuous detector specificity failure and cannot be repaired by reinstating a Terrain gate in this milestone.

## 23. Sand benign analysis

Balanced Sand deformation produced no incipient reference event and no frozen raw Support alert in all 36 TRAIN and 12 VALIDATION benign runs. The precursor reference therefore distinguishes heterogeneous spread from balanced deformation in the current simulator, but the evidence does not establish real-world universality.

## 24. Established Support timing

The established clock remains authoritative historical diagnostics: spread ≥10 mm for 20 ms. Frozen raw valid timing stayed within `[-30,+50] ms`; VALIDATION median/p95 was -8.5/-1 ms and TRAIN median/p95 was -9/-1 ms. No established threshold or persistence was modified.

## 25. Incipient timing if applicable

I1 completed on all 12 VALIDATION Support runs before the established event, with lead min/p10/median/p95/max 1235/1235/1517.5/1799/1799 ms. The frozen detector's first alert occurred a median 1226 ms after I1 onset. Therefore the detector does not estimate the incipient onset promptly; it merely alerts while the precursor state is already active. This prevents promotion to `INCIPIENT_SUPPORT_REFLEX_SUPPORTED`.

## 26. Selection/freeze

I1 is selected only as the simplest privileged diagnostic reference. No runtime Support candidate passed all validation gates, so no checkpoint/normalizer/threshold selection artifact was frozen and no policy may be inferred from validation afterward.

## 27. One-shot HOLDOUT

HOLDOUT was not opened:

```text
performed = false
guard_open_count = 0
reason = continuous_support_candidate_failed_validation
```

No holdout run ID, waveform, Support outcome or per-run score was accessed.

## 28. Integrated implication

Integrated Continuous Slip + Support + frozen Terrain replay was not run because Support failed validation. Terrain must remain advisory and must not reset Support probability/persistence, but the present raw Support branch is not specific enough to trigger `REFLEX_REQUIRED` independently.

## 29. Sensor implication

Support runtime input remains Pelvis IMU6. Privileged spread is a simulator-only reference and was never added to the runtime tensor. No Foot IMU, FSR, q/dq, torque or current augmentation was performed or selected. This bounded failure is evidence for a later, separately authorized observability/formulation decision, not an automatic sensor change.

## 30. Historical comparison

| Study | Key finding |
|---|---|
| Frozen Terrain-gated Support | Raw signal present, seven TRAIN decisions suppressed |
| Global/per-foot Terrain fusion | Suppression reduced, -565 ms mode exposed |
| Early-mode resolution | -565 ms mode is physical, but continuous IMU branch cross-triggers on 62.5% of VALIDATION Ice runs |

The historical gating and per-foot reports remain unchanged.

## 31. Limitations

- Balanced no-Support runs have exactly zero heterogeneous spread, so all three q99.5 incipient candidates collapse to a zero threshold.
- Per-cell displacement was unavailable in the stored corpus.
- The incipient reference begins at early Sand contact and substantially precedes the detector response.
- Exact `loaded_contact` and support deformation are privileged audit/reference signals, not deployable inputs.
- The development cause verdict has no one-shot holdout confirmation because the runtime validation gate failed.
- Current conclusions apply only to the frozen G1 policy, engineering terrain and simulator oracle.

## 32. Verdict

Cause:

`SUPPORT_EARLY_MODE_PHYSICAL_PRECURSOR`

Final Support verdict:

`CONTINUOUS_SUPPORT_REFLEX_NOT_SUPPORTED`

The early mode must not be mined as a normal gait negative, but reinterpreting it does not make the frozen continuous detector safe across Ice non-Support trajectories.

## 33. Next recommendation

Stop at this milestone. A future explicitly authorized study should decide whether the Support runtime target should predict heterogeneous support deformation or whether minimal additional observability is needed to separate Support from Ice Slip. Do not automatically start sensor augmentation, Recovery, E84, final sensor freeze or a larger model.
