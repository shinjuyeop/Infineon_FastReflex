# Sand Generalization Hypothesis Review

## 1. Purpose

This milestone reconciles the frozen mild-recalibrated Sand Discovery and Confirmation evidence without reopening a payload or rerunning a model or metric. It asks why global window centroid separation changed from `.890733` to `.208030`, what that disagreement means relative to stable local-neighborhood evidence, which one future-study hypothesis is best supported, and whether deployment engineering may proceed independently of final model approval.

The review verdict is `SAND_GENERALIZATION_HYPOTHESIS_REVIEW_ACTIONABLE`. The selected future-study hypothesis is `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS` at MODERATE confidence. This is not a rewrite of Discovery H1 or a new claim about consumed Confirmation.

## 2. Starting state

- Starting `HEAD` and `origin/main`: `ce8070930988872566c90121fc5e88d8c23fc4a8`
- Starting tracked worktree: clean
- Previous milestone: `SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_CONFIRMATION_ANALYSIS`
- Review config SHA-256: `8029e910eced683c3b38223314868ce894584fae0901d79d3767320db9047ac3`
- Discovery evidence input SHA-256: `255078ce8e8ee975adeae725b2f77ec2adf750561ca1e83ee6b5738ed35f9c55`
- Confirmation evidence input SHA-256: `33d80c80d6517bff9fea1ab2fd57782d1ac686d45fd2bbf6bac595c23409e59b`

All frozen input files, semantic interpretation hashes, model identities, and the historical HOLDOUT guard matched before review.

## 3. Evidence boundary

The review uses only already-saved Discovery and Confirmation analysis JSON, run-level result summaries, predeclared factor summaries, immutable reports/configs, and previously authorized development summaries. It performs no Discovery or Confirmation inference, payload deserialization, raw feature reconstruction, metric rerun, scaler fit, factor search, simulation, or training.

Where the frozen artifacts do not contain a required value, this review records `NOT_AVAILABLE_FROM_FROZEN_ANALYSIS_ARTIFACT`. In particular, per-class centroid RMS and per-run distance to the class centroid were not persisted and were not reconstructed.

The historical Generalization HOLDOUT remains guard `1`, scientific opens `1`, with zero payload reads or inference in this milestone. Mild-recalibrated Confirmation remains consumed and was not reopened.

## 4. Historical frozen verdicts

The following remain unchanged:

- Discovery: `DOMAIN_DIVERSITY_GAP_SUPPORTED`
- Confirmation: `DOMAIN_DIVERSITY_GAP_NOT_CONFIRMED`
- Confirmation validity: `SAND_BENIGN_GENERALIZATION_STUDY_CONFIRMATION_ANALYSIS_VALID`
- Final Model V2: `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- Whole simulation: `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- Support branch: `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`

This review does not promote H2 or H3 from the consumed Confirmation evidence.

## 5. Discovery vs Confirmation geometry

| Evidence | Discovery | Confirmation | Interpretation |
|---|---:|---:|---|
| Window centroid separation | .890733 | .208030 | global normalized metric did not replicate |
| Window centroid distance | 28.650376 | 49.564479 | centroids moved farther apart, +73.00% |
| Window within-group RMS | 32.164950 | 238.256187 | denominator increased 7.4073x |
| Balanced 1NN | .992754 | 1.000000 | local class neighborhoods remain distinct |
| Balanced 5NN | .992754 | 1.000000 | local class neighborhoods remain distinct |
| Local mixing | .007246 | .003077 | no broad cross-class mixing |
| Radius inclusion | .500000 | .500000 | unchanged global radius diagnostic |
| Strict Sand specificity | 67/69 (97.10%) | 61/65 (93.85%) | same qualitative issue, worse magnitude |
| Sand adverse rate | 24/69 (34.78%) | 29/65 (44.62%) | systematic margin issue replicated and worsened |
| Support recall | 16/16 | 16/16 | stable control |
| FSR material increment | 1/4 | 1/4 | non-material in both splits; 3/4 required |
| Topology/phase Cramér's V | .368453 / .392232 | .216645 / .210398 | direction stable, effect smaller |

The contradiction is specific: a global squared-distance-sensitive denominator changed drastically while local neighbor separation improved slightly.

## 6. Centroid-separation failure decomposition

| Quantity | Discovery | Confirmation | Ratio/change |
|---|---:|---:|---:|
| Centroid distance | 28.650376 | 49.564479 | 1.7300x / +73.00% |
| Within-Sand centroid RMS | `NOT_AVAILABLE_FROM_FROZEN_ANALYSIS_ARTIFACT` | same | not recomputed |
| Within-Sand pairwise median | 53.095501 | 52.002261 | .9794x / -2.06% |
| Within-Sand pairwise p95 | 95.715046 | 112.772383 | 1.1782x / +17.82% |
| Within-Support centroid RMS | `NOT_AVAILABLE_FROM_FROZEN_ANALYSIS_ARTIFACT` | same | not recomputed |
| Within-Support pairwise median | 26.513243 | 26.513243 | 1.0000x |
| Within-Support pairwise p95 | 35.180637 | 35.180637 | 1.0000x |
| Global within-group RMS | 32.164950 | 238.256187 | 7.4073x / +640.73% |
| Normalized centroid separation | .890733 | .208030 | .2335x / -76.65% |

The failure was not caused by class centroid means becoming similar: absolute centroid distance increased. The saved Support pairwise quantiles are identical, while Sand's median is stable and p95 rises moderately. Global RMS rises far more than those robust quantiles, identifying squared-tail leverage in window-space Sand geometry as the persisted mathematical driver. Exact allocation of global RMS to Sand versus Support is unavailable, but the unchanged saved Support distribution strongly argues against a Support-spread explanation.

Single-sample current-80D geometry is a control against a general preprocessing failure: centroid separation changes only `.689332 → .664068`, centroid distance `4.950781 → 4.693735`, and within-group RMS `7.181996 → 7.068158`. The large shift is specific to the flattened 20 ms trajectory representation.

## 7. Local-neighborhood evidence

Confirmation window 1NN and 5NN are both `1.0`, local mixing is `.003077`, Sand-in-Support 95% radius is `0`, and Support-in-Sand is `1.0`; the balanced radius summary remains `.5`. These metrics answer different geometric questions from centroid separation. They show locally coherent class neighborhoods inside an asymmetric and globally heterogeneous Sand envelope.

This evidence blocks promotion of a broad `PELVIS_OBSERVABILITY_TENSION`: broad observability failure would require repeated cross-class neighborhoods or high mixing, neither of which appears. It also does not prove that Pelvis is universally sufficient; local separability in these frozen anchors is narrower than runtime detection sufficiency.

## 8. Factor-conditioned representation heterogeneity

Predeclared factor-localized window metrics reveal that the Confirmation geometry shift is not uniform.

| Factor level | Discovery separation / RMS | Confirmation separation / RMS | Review |
|---|---:|---:|---|
| Concrete | .977 / 33.512 | 1.076 / 28.144 | stable |
| Marble | .839 / 33.359 | .263 / 330.528 | strong window-spread shift |
| 0.20 m/s | .842 / 34.847 | .309 / 392.653 | strong window-spread shift |
| 0.25 m/s | 1.392 / 32.474 | 1.445 / 25.906 | stable |
| 0.30 m/s | .894 / 34.562 | 1.015 / 28.340 | stable |
| Transition-left | .895 / 32.385 | .221 / 265.479 | strong window-spread shift |
| Transition-right | 1.270 / 34.159 | 1.825 / 21.106 | stable |
| Right-single precontact | .905 / 32.480 | .881 / 32.558 | stable |
| Left-single precontact | 1.270 / 34.159 | 1.825 / 21.106 | stable |
| Mild | 1.007 / 35.651 | .219 / 275.234 | strong window-spread shift |
| Moderate | 1.295 / 32.491 | 1.252 / 39.125 | stable |
| High exposure | .903 / 34.500 | .315 / 392.773 | strong window-spread shift |
| Low exposure | .935 / 34.342 | 1.169 / 25.280 | stable |
| Mid exposure | 1.274 / 33.362 | 1.791 / 20.364 | stable |

Confirmation has one double-support row, while right-/left-single subsets remain individually reasonably separated. The low-separation subsets—Marble, `.20`, left, mild, and high exposure—share membership with that lone predeclared double-support run, `sbmrc_c_bb_m_020_01`. The saved subset pattern and PC1 concentration are consistent with large leverage from a rare Sand trajectory/submanifold, but its exact per-run centroid distance is `NOT_AVAILABLE_FROM_FROZEN_ANALYSIS_ARTIFACT`; the review does not reconstruct or assert an exact contribution.

Window PCA reinforces nonuniform geometry: PC1 explained variance changes from `.369077` to `.983477`, while PC2 changes from `.230684` to `.006017`. This supports `CONFIRMATION_REPRESENTATION_DISTRIBUTION_SHIFT` and `FACTOR_CONDITIONED_REPRESENTATION_HETEROGENEITY`, not a scaler bug.

## 9. Frozen localization replication

| Factor | Discovery adverse direction | Confirmation direction | Stable? | Interpretation |
|---|---|---|---|---|
| Topology | left 24/55 > right 0/14 | left 26/52 > right 3/13 | YES | same direction; V `.3685 → .2166` |
| Precontact phase | right-single 24/52 > left-single 0/14 | right-single 25/51 > left-single 3/13 | YES | same coupled direction; V `.3922 → .2104` |
| Source | C/M adverse 12/12; FP 1/1 | adverse 11/18; FP 0/4 | NO | both represented, but magnitude/FP identity shifts |
| Speed | adverse `.20/.25/.30 = 8/9/7` | `11/12/6` | NO | all speeds contribute, but rank and FP cells shift |
| Mild/moderate | FP `2/0`, adverse `19/5` | FP `4/0`, adverse `22/7` | YES, qualified | FP remains mild-only; adverse is not severity-exclusive |

Topology and phase remain physically coupled and are interpreted as one stable physical direction. Their effect sizes decrease substantially but remain above the frozen association threshold. This supports a factor-conditioned model-boundary study target even though it cannot relabel H1 as confirmed.

## 10. Model behavior replication

Frozen V2's qualitative behavior replicates: adverse Sand margins remain systematic, the topology/phase direction persists, and Support remains 16/16. Magnitude worsens from two to four false Reflexes and from 34.78% to 44.62% adverse. The result is a systematic generalization-boundary problem, not random isolated score noise.

The evidence does not yet isolate model capacity from training-domain coverage. Local Pelvis neighborhoods are highly separable, but current V2 still acts on benign transients; that combination is compatible with a decision boundary trained on insufficient factor-conditioned trajectories. A capacity intervention has not been controlled against a data-domain intervention.

## 11. Mild/moderate interpretation

All six fresh-study false Reflexes are physically clean mild Sand: Discovery 2/48 and Confirmation 4/48. Strict moderate Sand is 21/21 and 17/17 specific, respectively. Adverse margins occur in both severities, so severity alone is not the decisive factor; actual false actions instead align with particular contact/topology dynamics inside the mild domain.

The mild label should not be treated as physically easier for the model. Conversely, the result does not justify relabeling mild runs or changing the physical target.

## 12. Source/speed interpretation

Discovery false Reflex cells were Concrete/.30 and Marble/.25. Confirmation false Reflex cells were Marble/.20 and Marble/.25, two each; other Confirmation cells had zero FP. Adverse margins are broader than the FP cells in both splits.

Marble-only and speed-only hypotheses are therefore not stable enough to select. Source and speed modulate margin magnitude, while the more stable cross-split invariant is the coupled topology/precontact-phase direction. Future coverage should remain symmetric over source and speed rather than target only the latest FP cells.

## 13. FSR/contact interpretation

Realizable FSR/contact material improvement is exactly 1/4 in both Discovery and Confirmation, below the frozen 3/4 requirement. Combined-minus-Pelvis improvement is limited to the epsilon-sensitive distance-ratio check; centroid, 5NN, and mixing improvements fail.

`HAZARD_FSR_FUSION_NOT_JUSTIFIED_BY_CURRENT_EVIDENCE` remains the supported conclusion. It also argues against retroactively promoting H2 from the failed centroid criterion: the available realizable contact representation does not provide the material complementary separation that H2 required. No sensor selection is reopened.

## 14. Privileged oracle interpretation

Discovery/Confirmation privileged-oracle centroid separation is `1.242028/1.241039`, 1NN/5NN is `1/1` in both, and local mixing is `.0125/.0125`. The physical semantics are distinguishable when exact loaded contact, Support spread, and simulator state are available.

Those fields are privileged and do not decide Pelvis versus model versus data coverage. Oracle strength neither justifies FSR fusion nor supplies a production sensor claim.

## 15. Scaler and normalization review

Using the Discovery-pooled scaler for Confirmation was the scientifically correct frozen operation. Mean/std hashes matched exactly, Confirmation did not refit the scaler, the same feature implementation was used, and current-80D geometry remained stable. There is no saved evidence of numerical or preprocessing inconsistency.

The large window RMS is therefore classified as `CONFIRMATION_REPRESENTATION_DISTRIBUTION_SHIFT`: legitimate fresh trajectories occupy a more extreme direction under the fixed Discovery coordinates. This is evidence about distribution geometry, not a normalization bug.

## 16. Metric robustness review

| Metric | Discovery | Confirmation | What it measures | Robust across splits? |
|---|---:|---:|---|---|
| Centroid separation | .890733 | .208030 | global centroid distance / squared-tail-sensitive RMS | NO |
| Balanced 1NN | .992754 | 1.000000 | nearest local class agreement | YES |
| Balanced 5NN | .992754 | 1.000000 | local five-neighbor agreement | YES |
| Local mixing | .007246 | .003077 | opposite-class share in local neighborhoods | YES |
| Radius inclusion | .500000 | .500000 | asymmetric global centroid-radius overlap | YES at summary level |
| Cramér's V localization | `.368/.392` | `.217/.210` | factor/adverse association | direction yes; magnitude no |
| V2 specificity | .971014 | .938462 | frozen operating-point behavior | qualitative issue yes; magnitude no |
| FSR increment | 1/4 | 1/4 | complementary realizable-contact benefit | YES |

No metric is discarded or ranked by convenience. Centroid separation measures global compactness, while NN and local mixing measure neighborhood topology; they can legitimately disagree for multimodal or sparse-tail classes. The appropriate methodological classification is `CENTROID_METRIC_SENSITIVE_TO_WITHIN_CLASS_HETEROGENEITY`.

The historical H1 all-required rule remains unchanged and failed validly. A future independent protocol may predeclare complementary robust global and local geometry diagnostics, but cannot rescore this Confirmation.

## 17. Representation-distribution shift

The supported synthesis is:

1. The same scaler and implementation rule out a silent preprocessing change.
2. Stable current-80D geometry localizes the shift to short trajectory shape, not the anchor sample alone.
3. Increased centroid distance rules out converging class means.
4. Identical saved Support quantiles argue against Support as the driver.
5. Stable Sand median but higher p95, 7.4x global RMS, and PC1 dominance indicate tail-sensitive Sand window heterogeneity.
6. Predeclared subgroup summaries localize that geometry shift to a restricted intersection while other subgroups remain well separated.

This supports factor-conditioned representation heterogeneity, but not universal Pelvis insufficiency or proof that one specific run alone caused the shift.

## 18. Candidate next hypotheses

| Candidate next hypothesis | Evidence for | Evidence against | Confidence | Needs new evidence? |
|---|---|---|---|---|
| `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS` | stable topology/phase direction; systematic V2 margins; local separability; unequal subgroup RMS; non-material FSR | original global criterion failed; source/speed FP cells move; data remedy untested | **MODERATE** | YES |
| `MODEL_BOUNDARY_GENERALIZATION_HYPOTHESIS` | V2 FP worsened despite strong local geometry; Support stable | factor-conditioned coverage remains explanatory; no data-vs-capacity control | LOW–MODERATE | YES |
| `PELVIS_REPRESENTATION_HETEROGENEITY_HYPOTHESIS` | window RMS/PCA instability; global/local metric disagreement | current 80D stable; single-support subgroups strong; no feature comparator | MODERATE | YES |
| `SAND_GENERALIZATION_HYPOTHESIS_UNRESOLVED` | per-run centroid distances unavailable; causality not fully separated | evidence is sufficient to define a narrow factor-conditioned future test | not selected | YES |

## 19. Selected next hypothesis

`FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS`

Confidence is MODERATE. The future question is whether a newly generated training corpus with symmetric coverage of the predeclared topology/precontact-phase trajectory manifolds improves Sand margins while preserving Support, under a fresh evaluation protocol that predeclares robust global and local representation checks.

This is a hypothesis for new independent evidence only. It does not say the original `DOMAIN_DIVERSITY_GAP_SUPPORTED` result was confirmed, does not copy consumed Confirmation runs, and does not authorize training in this milestone.

## 20. Architecture and sensor implications

| Intervention question | Justified now? | Reason |
|---|---|---|
| Longer history | NO | no controlled evidence that missing temporal extent caused the failures |
| LSTM | NO | no recurrence-family comparison or causal capacity diagnosis |
| Larger GRU | NO | capacity not separated from factor-conditioned domain coverage |
| Feature redesign | NOT AS CURRENT INTERVENTION | representation study remains a secondary future comparator, not the smallest next step |
| FSR Hazard fusion | NO | material increment 1/4 in both splits |
| Foot IMU | NO | no frozen evidence evaluates a material increment |
| More fresh domain coverage | **YES, design only** | stable factor direction and training-domain interpretation justify a controlled design |
| Model-capacity study | NOT YET PRIMARY | retain as future alternative if controlled data-domain evidence fails |

Final sensor architecture remains unfrozen. Terrain remains advisory and cannot gate Hazard.

## 21. Deployment parallelization

`DEPLOYMENT_ENGINEERING_CAN_PROCEED_IN_PARALLEL = YES`

The exact frozen V2 may be used as `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`, not as the final supported Hazard model:

- Candidate: `model_v2_anchor_refined_gru20_20260902`
- Candidate record SHA-256: `52644d3efb0e756002bd43a7f94aea0a16a65eb2ff014c72548f2b3b138b7bc2`
- Candidate freeze SHA-256: `95dab53275aa77ded36b479c3633ab879a1331cb2c96bd0eb014de2be99bd85f`
- Architecture SHA-256: `ae4753699399aa7e4a93639935e085156ef28ed100a77058d7308eb1549aa897`
- Feature schema SHA-256: `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`
- Normalizer SHA-256: `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a`
- Checkpoints: `7094a2dc…cb9`, `3ad298ee…c39`, `fe96dfeb…bbd`
- Runtime reference: Pelvis IMU6 → causal 80D → `[20,80]` → GRU32, three-seed mean, `.99`, 5 ms

Safe to begin now, in the deployment repository only:

- exact 80D preprocessing and parity implementation;
- ring/history buffer and threshold/persistence state machine;
- model export path and operator/U55 compatibility audit;
- Float reference inference and output-parity harness;
- quantization pipeline scaffolding and calibration-data interface;
- runtime ensemble-strategy review;
- timing and memory profiling;
- HIL transport/plumbing and parity-harness scaffolding.

Must wait:

- final production weight lock;
- final representative calibration corpus and INT8 accuracy sign-off;
- final HIL performance/safety sign-off;
- final sensor architecture;
- final release, production-readiness, or simulation-generalization claim.

No E84/deployment repository was modified in this milestone, and research weights remain unchanged.

## 22. Review freeze and hashes

| Required review identity | SHA-256 |
|---|---|
| `HYPOTHESIS_REVIEW_CONFIG_SHA` | `8029e910eced683c3b38223314868ce894584fae0901d79d3767320db9047ac3` |
| `DISCOVERY_EVIDENCE_INPUT_SHA` | `255078ce8e8ee975adeae725b2f77ec2adf750561ca1e83ee6b5738ed35f9c55` |
| `CONFIRMATION_EVIDENCE_INPUT_SHA` | `33d80c80d6517bff9fea1ab2fd57782d1ac686d45fd2bbf6bac595c23409e59b` |
| `REPRESENTATION_GEOMETRY_REVIEW_SHA` | `f8529ea07f887934fb562a890d2646496c69ef6e497f711ac90fdbdae645cef6` |
| `LOCALIZATION_REPLICATION_REVIEW_SHA` | `6fc3a23f44be5996661562de3265ee1b13198eab9ee983bd5f17a33f71a6b482` |
| `METRIC_ROBUSTNESS_REVIEW_SHA` | `a2201874a2acfc1ba8e72ab93a68608d7b2db574e4919ce197c6e61d7c2bb7d8` |
| `NEXT_HYPOTHESIS_DECISION_SHA` | `b1036e1818ed1f14a50ba0209ecb05263d99d489d9b5f1dc529224001bd6ca44` |
| `SAND_GENERALIZATION_HYPOTHESIS_REVIEW_SHA` | `de573be7497703a19d3921ca3085834707fd5656393989ab5fa94f5c2408102d` |
| Review interpretation file SHA-256 | `3925f4f75ef1dddf3b709e9434833f99839cf2d99f4a9cd779d3bb76046e2bd1` |

The final semantic hash binds the config, both evidence-input hashes, component hashes, frozen statuses, selected future hypothesis, deployment boundary, and zero-execution counters.

## 23. No-new-execution counters

```text
new simulations = 0
V1 inference = 0
new V2 Discovery inference = 0
new V2 Confirmation inference = 0
Terrain inference = 0
optimizer steps = 0
checkpoint writes = 0
normalizer fits = 0
HNM rounds = 0
threshold searches = 0
persistence searches = 0
architecture searches = 0
old HOLDOUT reads = 0
payload-derived metric reruns = 0
```

## 24. Limitations

- Exact Sand/Support class-specific centroid RMS and per-run centroid distances were not persisted.
- The lone Confirmation double-support run's leverage is supported indirectly by frozen subset geometry, not by a newly computed leave-one-out metric.
- Deterministic MuJoCo evidence is not real-soil, real-robot, safety, or deployment validation.
- Topology and precontact phase remain coupled.
- The review identifies the smallest testable hypothesis but does not prove that new data will outperform a model or feature intervention.
- Consumed Confirmation cannot be reused to validate the selected future hypothesis.

## 25. Verification

- Review plus frozen Discovery/Confirmation targeted tests: `15 passed`
- Repository-wide safe suite: `153 passed, 1 skipped`
- `compileall src scripts tests`: PASS
- Ruff configured checks, including `E9,F63,F7,F82`: PASS
- `git diff --check`: PASS
- Discovery/Confirmation evidence composite hashes: PASS
- Review component and semantic hash chain: PASS
- Historical HOLDOUT guard and frozen model identities: PASS

The review-specific verifier reads JSON/YAML and hashes only. No scientific model or metric execution is part of the review result.

## 26. Review validity and verdict

`SAND_GENERALIZATION_HYPOTHESIS_REVIEW_ACTIONABLE`

All evidence hashes matched, the central mathematical discrepancy is explainable from persisted values, one bounded future hypothesis is selectable without rewriting prior results, and the next study can be designed with genuinely new evidence. No metric was changed or rerun.

## 27. Recommended next scientific milestone

`SAND_FACTOR_CONDITIONED_TRAINING_DOMAIN_DESIGN`

This next milestone should freeze a new, symmetric factor-conditioned training-domain design and a future independent evaluation contract. It should not generate or train yet, reuse Confirmation examples, reopen historical HOLDOUT, or change architecture/sensors merely because H1 failed.

## 28. Parallel deployment milestone

`MODEL_V2_DEPLOYMENT_ENGINEERING_REFERENCE_HANDOFF`

This may proceed separately using the exact frozen V2 identity solely as an engineering reference. Do not start it automatically, modify research weights, or claim final model approval.
