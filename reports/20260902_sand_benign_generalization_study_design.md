# Sand Benign Generalization Study Design

## 1. Purpose

This design freezes a fresh, independent development study before any new simulation or model replay. Its primary question is whether broad, predeclared Sand-benign diversity remains distinguishable from true Support using the current Pelvis IMU6 → causal 80D representation. The study is designed to separate a domain-diversity limitation from Pelvis observability and GRU20 decision-boundary limitations; it is not a training, sensor-selection, architecture-search, or final-evaluation milestone.

The authoritative contract is `configs/experiment/20260902_sand_benign_generalization_study_design.yaml`, file SHA-256 `e45dcbe8130e5887c65ec7e9e3ef8c03744f8b589c81bc3ffe12536a0b145f70`.

## 2. Starting state

The starting `HEAD` and `origin/main` were both `67258c016adb67b7c40a2c2274b255810ae5e342` (`Interpret Model V2 HOLDOUT failures`), on `main`, with a clean tracked worktree. The previous verdict was `MODEL_V2_HOLDOUT_FAILURE_INTERPRETATION_ACTIONABLE`, and its single recommended next milestone was this design.

Model V1, baseline V2, extraction-rebalanced V2, final anchor-refined V2, and Terrain V1 were verified read-only. The protected Unified, Model V2, Generalization, and Ice-semantics manifests remained exact at 256, 412, 72, and 48 runs respectively. No dataset, model, normalizer, checkpoint, runtime code, or training code was modified.

## 3. Historical HOLDOUT result

The immutable final verdict remains `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`, and the system status remains `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`. Final V2 obtained Hazard 25/28, Slip 11/14, Support 14/14, primary specificity 5/8, and premature 2/28. The Support branch is `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`; Slip and benign rejection remain `SLIP_SIMULATION_GENERALIZATION_NOT_SUPPORTED` and `BENIGN_REJECTION_GENERALIZATION_NOT_SUPPORTED`.

The six Sand-benign HOLDOUT cases were physically no-hazard, but final V2 was specific on only 3/6. Saved summaries identify false alerts in Concrete/.20, Concrete/.30, and Marble/.30. This establishes the existence of a Sand generalization failure; it does not provide reusable examples for optimization.

## 4. Consumed-HOLDOUT boundary

The 36-run Generalization HOLDOUT guard remains permanently `1`, with one scientific opening. This milestone made zero HOLDOUT payload reads, model inferences, feature reconstructions, and visualizations. Only committed result summaries, permitted manifest metadata/signatures, and historical conclusions were used. Unpersisted information is `NOT_AVAILABLE_FROM_CONSUMED_HOLDOUT`.

Raw runs, reconstructed signals, exact failed-run waveforms, training, HNM, threshold/persistence tuning, model selection, and fresh evaluation use are forbidden. There is no guard reset and no second scientific opening.

## 5. Current failure interpretation

The historical coverage verdict remains `DATA_COVERAGE_HYPOTHESIS_NOT_SUPPORTED`; the precise interpretation is `DATA_COVERAGE_ALONE_INSUFFICIENT`. Coverage and corrected extraction materially improved V2, but source/speed labels did not ensure rich within-cell geometry, contact timing, gait phase, contact sequence, or load evolution.

Every source–speed cell had TRAIN, V2_VALIDATION, and Generalization VALIDATION representation. Exact fresh geometry was absent, while all six consumed HOLDOUT Sand runs shared the same new geometry and only three failed. The supported interpretation is therefore an interaction between scenario shift and source/speed/contact dynamics, not geometry alone. Development-only analysis found `LOW_MARGIN_BENIGN_GENERALIZATION` with high confidence and `RESIDUAL_SAND_SUPPORT_FEATURE_OVERLAP` with moderate confidence.

## 6. Scientific hypotheses

- `H1_DOMAIN_DIVERSITY_LIMITATION`: nominal source/speed coverage omitted enough combinations of geometry, transition/contact timing, gait phase, contact sequence, load evolution, and natural realization, so the learned benign manifold was narrow.
- `H2_PELVIS_REPRESENTATION_OBSERVABILITY_LIMITATION`: even after broad physical coverage, benign Sand and Support remain strongly mixed in Pelvis features, while realizable FSR/contact-load information materially improves separation.
- `H3_MODEL_DECISION_BOUNDARY_CAPACITY_LIMITATION`: the Pelvis representation is meaningfully separable under model-independent metrics, yet the frozen GRU20 fails systematically without a sparse metadata explanation.

No hypothesis is assumed. The primary contrast is H1 versus evidence that would justify a later H2 or H3 study.

## 7. Why Sand is the next priority

Support transferred cleanly: V2_VALIDATION 30/30, Generalization VALIDATION 14/14, Generalization HOLDOUT 14/14, delayed Support 4/4, and right-only Support 4/4. The one genuine +42 ms Slip failure is retained as `STRICT_SLIP_TIMING_LIMITATION`, but Sand caused three false actions among only eight primary no-hazard HOLDOUT runs and the largest validation-to-HOLDOUT deterioration.

The smallest identifiable next question is therefore Sand-benign diversity. Support remains a frozen comparison concept, and Slip, Ice semantics, Terrain, architecture, and sensors are outside this intervention.

## 8. Existing domain coverage

Previous Speed-Sand benign coverage was balanced at the coarse source/speed level: 36 V2_TRAIN runs, 12 V2_VALIDATION runs, six Generalization VALIDATION runs, and six historical HOLDOUT runs. Within each source–speed cell, however, V2_TRAIN used six fixed geometry templates, V2_VALIDATION used two, and Generalization VALIDATION used one. V2_TRAIN ended at start `.334`, V2_VALIDATION at `.348`, and Generalization VALIDATION used `.358/.725`; contact phase was mostly an indirect consequence rather than a balanced factor.

Frozen final V2 replay on authorized development data showed the following worsening margins:

| Development group | N | Median max p | p95 max p | Reflex |
|---|---:|---:|---:|---:|
| V2_TRAIN Speed-Sand benign | 36 | .7792 | .9355 | 0 |
| V2_VALIDATION Speed-Sand benign | 12 | .9537 | .9903 | 0 |
| Generalization VALIDATION Speed-Sand benign | 6 | .9868 | .9931 | 0 |

Binary development specificity therefore concealed a progressively thin boundary. Existing-domain HNM was extensive, so duplicating the same physical domain is not the selected intervention.

## 9. Study parameter domain

Numeric bounds come from canonical simulator constraints and prior development/calibration domains, never from a consumed HOLDOUT waveform. The `.362/.735` pair is not a planned point.

| Factor | Historical range/levels | Proposed levels/range | Sampling method | Reason |
|---|---|---|---|---|
| Source | Concrete, Marble | Concrete, Marble | full balance for every template and speed | estimate source interactions without targeting a failed source |
| Speed | .20/.25/.30 m/s | .20/.25/.30 m/s | full balance for every template and source | preserve the frozen endpoint grid; no failed-speed oversampling |
| Patch entry/start | phase audit .285–.375 m; benign .304–.358 m | .280–.395 m; planned .281–.392 | EARLY/MID/LATE, four template slots each per split | broaden entry timing inside previously feasible bounds and canonical geometry |
| Patch width | prior Sand .672–.816 m; benign .718–.746 m | .660–.820 m; planned .664–.814 | NARROW/MEDIUM/WIDE, four slots each per split | vary exposure duration across the union of validated Sand ranges |
| Topology / leading side | transition-left/right | transition-left/right | six each per source–speed–split | prevent a one-side contact manifold |
| Entry/contact phase | indirect only | exact loaded-state categories after generation | four balanced phase-offset slots, then model-blind physical stratification | direct policy-phase control does not exist; actual phase cannot be invented |
| Gait realization | deterministic simulator; no stochastic seed | two assignment cohorts `2026090201/2026090202` | fixed template permutation only; every run has distinct physical parameters | avoid pretending duplicate RNG reruns are independent |
| Benign severity | mild balanced deformation | mild/moderate/severe intent, actual LOW/MEDIUM/NEAR_HAZARD | four templates per intent level in every source–speed–split | span the canonical 20/40/65 mm compliant travel ladder |
| Observation | 8 s | 8 s, at least 1,000 ms after last target contact/exit | fixed | resolve no-Hazard outcome rather than censor it |

The simulator has no stochastic gait seed or direct initial gait-phase knob. Consequently “independent realization” means a distinct predeclared geometry/topology/speed configuration. Assignment seeds only freeze the balanced permutation and are never passed to the simulator as fictional randomness.

## 10. Source/speed matrix

Every Sand template expands identically across all six cells. Each cell receives 12 Discovery and 12 Confirmation cases, rather than one or two realizations.

| Speed | Concrete | Marble | Total |
|---:|---:|---:|---:|
| 0.20 | 24 | 24 | 48 |
| 0.25 | 24 | 24 | 48 |
| 0.30 | 24 | 24 | 48 |
| **Total** | **72** | **72** | **144** |

Concrete/.20, Concrete/.30, and Marble/.30 receive no extra cases. The old failed-cell pattern is not a sampling weight.

## 11. Geometry variation

The Sand template set is a balanced fractional factorial rather than the full Cartesian product. Within each split, 12 template rows contain four EARLY, four MID, and four LATE starts, plus four NARROW, four MEDIUM, and four WIDE widths. Starts and widths are paired differently between Discovery and Confirmation. Exact coordinates are fully enumerated in the config; no adaptive sampling or “sample until successful” rule exists.

The planned patch start range is `.281–.392 m`, the planned width range is `.664–.814 m`, and every end remains far inside the canonical `(-10,10) m` geometry bound. Target reach and post-target observation remain objective viability checks.

## 12. Gait/contact-phase variation

Direct gait-phase control is unavailable in the current deterministic simulator. Four `PHASE_A`–`PHASE_D` assignment slots therefore rotate geometry, width, topology, severity, and realization cohort; they are design strata, not claims about realized physical phase.

Before any V2 replay, actual entry state is derived from exact loaded contact as `LEFT_SINGLE_SUPPORT`, `RIGHT_SINGLE_SUPPORT`, `DOUBLE_SUPPORT`, or `NO_SUPPORT`, together with leading foot, first target contact, target contact duration, sequence count, and load-transition timing. A split must realize at least three phase categories globally, at least two per source–speed cell, and both leading feet per cell. Difficult phases remain; they are never excluded based on model behavior.

## 13. Natural realization diversity

Canonical simulation is deterministic, so a second run with identical physics and a different nominal seed would be a duplicate, not an independent realization. The two assignment cohorts instead select disjoint explicit start/width/topology/phase-slot combinations. Each source–speed–split cell has 12 distinct scenario signatures, and a post-generation physical-signature audit checks that at least 75% remain distinct in realized contact/load/Pelvis space.

This limitation is explicit: stochastic policy-state variation is not identified by this study. If deterministic physical variation cannot produce the required contact-phase and severity coverage, the result becomes inconclusive and a separate model-blind calibration/simulator design is required.

## 14. Physical benign definition

Design intent never defines the label. A strict confirmed-benign run must reach target Sand, have no established Slip, no I1 activation, no established Support, no fall/censor ambiguity over the target observation, at least 1,000 ms of post-target resolution, and finite complete required diagnostics. Only these runs enter frozen primary specificity.

An I1-only run with no later established Slip or Support is retained as a physically benign near-Support continuum case but is excluded from primary specificity, preserving the existing canonical no-hazard contract. Actual Slip and Support outcomes are retained and reclassified in their original split. A model false positive is never an invalid-run reason.

## 15. Benign severity strata

Severity is assigned from actual physical response after generation, not from `mild/moderate/severe` intent. The primary measure is peak absolute target balanced-plate vertical displacement during the fully observed target interval:

| Actual stratum | Frozen displacement interval | Additional required condition |
|---|---:|---|
| LOW | `[0, .030) m` | strict confirmed benign |
| MEDIUM | `[.030, .0525) m` | strict confirmed benign |
| NEAR_HAZARD | `[.0525, .070] m` | strict confirmed benign |

The boundaries are midpoints between the existing 20, 40, and 65 mm canonical deformable-support travel profiles, with 5 mm numeric headroom above severe. Peak support spread, load redistribution, transition displacement, and peak absolute FSR load derivative are also reported, but cannot override the strict physical label. Established Support remains 10 mm spread for 20 consecutive ms.

Each source–speed–split cell plans four low, four medium, and four near-hazard intents and must realize at least two strict-benign cases in each actual stratum. Failure of this yield gate is evidence that the planned benign envelope is not viable, not permission to move a boundary.

## 16. Support controls

Support controls preserve the frozen I1 and established-Support semantics and do not alter anchor extraction. Ordinary Support uses the existing moderate `lateral_deformable` topology: one left and one right template in every source–speed–split cell, for 24 total and exact left/right balance. Delayed Support uses the existing calibrated left-only `staged_lateral_deformable` topology at .25 m/s: two geometries per source per split, for eight total.

The delayed right-side and endpoint-speed combinations are not fabricated because canonical support does not exist for them. Source balance is exact: 16 controls from Concrete and 16 from Marble. These controls are for physical and representation comparison, not retraining.

## 17. Planned corpus

| Group | Source | Speed | Geometry/phase diversity | Planned runs | Role |
|---|---|---|---|---:|---|
| Broad Sand benign | C/M | .20/.25/.30 | 16 distinct low/medium templates per source–speed across both splits; all start/width/side/phase slots | 96 | strict-benign intent |
| Near-hazard Sand benign | C/M | .20/.25/.30 | 8 severe templates per source–speed across both splits; all start/width/side/phase strata | 48 | near-hazard strict-benign intent |
| Ordinary Support control | C/M | .20/.25/.30 | left/right and split-distinct geometry | 24 | Support comparison |
| Delayed Support control | C/M | .25 | two split-distinct left staged geometries | 8 | delayed-Support comparison |
| **Total** |  |  |  | **176** | development study |

The 144 Sand runs balance source 72/72, speed 48/48/48, severity intent 48/48/48, and topology 72/72. Each split contains 72 Sand and 16 Support controls, for 88/88. This is large enough that no source–speed conclusion rests on one run, while avoiding the 2×3×3×3×4×2 full Cartesian explosion.

## 18. Split strategy

`STUDY_DISCOVERY` and `STUDY_CONFIRMATION` are run-disjoint development roles, not TRAIN and not final HOLDOUT. Membership and all exact templates are frozen before generation. No row moves, replacement, or outcome-driven backfill are allowed.

Discovery contains 88 planned runs and may be used for viability, physical diversity, model-independent Pelvis/FSR analysis, and one exact frozen-V2 replay. Confirmation contains 88 and remains sealed until generation/integrity is complete, Discovery viability passes, implementations and hashes are reverified, and exactly one deterministic Discovery interpretation plus its supporting metric object is hash-frozen. Confirmation can then receive one unchanged diagnostic pass. It may not tune a model or replace the frozen hypothesis after opening.

## 19. Duplicate/diversity policy

The exact nine-field signature is `(source, target, speed, start, width, slip_pattern, sink_pattern, severity, support_pattern)`. All 176 planned rows are unique. Metadata-only comparison against the exact protected Unified 256, Model V2 412, Generalization 72—including consumed-HOLDOUT metadata only—and Ice-semantics 48 finds zero exact overlap.

For split near-duplicates, rows must share source, target, speed, and all mechanics and differ by less than 2 mm in start and 4 mm in width. Planned cross-split near-duplicate count is zero. After generation, a frozen scaled physical vector covers target-contact timing/duration, leading foot, entry phase, displacement, spread, load redistribution/derivative, and Pelvis RMS; same-domain pairs at scaled Euclidean distance at most `.10` are reported as near-duplicates. Model probability and classification success never enter this audit, and result-driven deletion is forbidden.

## 20. Invalid-run policy

Objective invalid reasons are nonfinite/unstable simulation, pre-target fall, target never reached, insufficient post-target observation, corrupt/malformed storage, and required sensor/diagnostic drop. Every attempted row remains in the manifest. There is no reserve grid, deterministic replacement, adaptive backfill, or failed-cell-specific regeneration; invalid rows reduce the valid denominator.

Each split must retain at least 54/72 strict-benign Sand cases, at least 9/12 per source–speed cell, at least 2/4 per actual severity and 4/6 per topology within every cell, at least 9/12 ordinary Support controls, and at least 3/4 delayed controls. Failure stops causal interpretation as `SAND_GENERALIZATION_STUDY_INCONCLUSIVE`; it does not relax labels or trigger selective replacement.

## 21. Pelvis separability analysis

The population is strict confirmed-benign Sand versus actual valid Support controls. Benign anchoring is model-independent: choose the earliest sample maximizing trailing-20-ms RMS Pelvis-acceleration deviation from the precontact baseline within target contact through 500 ms after last contact. Support primary anchoring is frozen I1; established Support is a secondary view.

The predeclared representations are final-V2-normalized current 80D and flattened trailing `[20,80]` (1,600D). Metrics are within-group-RMS-normalized centroid separation, leave-one-run-out balanced 1NN/5NN agreement, nearest-opposite/nearest-same distance ratio, five-neighbor opposite-class mixing, bidirectional 95% centroid-radius inclusion, deterministic PCA diagnostics, and within/between distance quantiles. PCA uses full SVD and a fixed sign rule.

“Reasonable Pelvis separation” requires all of: centroid separation at least `.75`, balanced 5NN agreement at least `.80`, local mixing at most `.30`, and median opposite/same neighbor-distance ratio at least `1.25`. “Strong mixing” requires at least three of: centroid separation at most `.60`, balanced 5NN at most `.70`, local mixing at least `.40`, distance ratio at most `1.10`, and bidirectional 95% radius inclusion at least `.75`.

## 22. FSR/contact diagnostic analysis

Three evidence layers stay separate:

1. runtime Pelvis: frozen causal 80D and `[20,80]`;
2. realizable contact/load: trailing FSR8 mean/max/std/delta, load imbalance, and FSR-derived contact;
3. privileged oracle: exact support spread, transition displacement, and exact loaded contact.

Pelvis+FSR uses Discovery-only pooled normalization frozen for Confirmation and never includes exact support spread/contact. A material realizable-FSR increment requires at least three of: centroid separation `+0.25`, balanced 5NN `+0.10`, local mixing `-0.15`, and opposite/same distance ratio `+0.20`, with no metric degrading by more than `.05`. Privileged spread is reported independently and cannot justify a practical sensor claim. There is no fusion-model, probe, Foot-IMU implementation, or sensor selection.

## 23. Frozen V2 diagnostic replay

The immutable diagnostic reference is `model_v2_anchor_refined_gru20_20260902`: Pelvis IMU6, causal 80D, `[20,80]`, GRU hidden 32, one unidirectional layer, exact three-seed mean, inclusive `.99`, and 5 ms persistence. It may be replayed once on Discovery after physical diversity is frozen, and once on Confirmation only after one interpretation hypothesis is frozen.

There is no training, normalizer fit, HNM, seed search, checkpoint selection, threshold search, persistence search, or architecture change. The fresh corpus is development evidence and cannot become a final test by naming convention.

## 24. Margin analysis

Future reports must include run-level strict-benign specificity, run maximum probability, maximum consecutive `p >= .99`, and mutually explicit margin bins `[.90,.95)`, `[.95,.99)`, `>=.99` with streak below 5 ms, and Reflex. Distributions are stratified by source, speed, start, width, topology, realized phase, and actual physical severity.

An adverse-margin run is frozen as either Reflex or max `p >= .95`. This tests `LOW_MARGIN_BENIGN_GENERALIZATION` without using old HOLDOUT probabilities as optimization targets.

## 25. Discovery decision rules

Before interpretation, integrity, yield, diversity, and duplicate prerequisites must pass. A systematic adverse pattern requires at least 20% adverse-margin runs, representation in both sources and at least two speeds, and at least two adverse cases in every implicated level. Metadata localization requires one predeclared factor with both an adverse-fraction range of at least `.25` and Cramér's V of at least `.20`, with at least eight valid cases per compared level.

The deterministic Discovery hierarchy is:

- domain diversity when systematic adverse margins are metadata-localized, Pelvis separation is reasonable, and realizable FSR has no material increment;
- Pelvis observability tension when the systematic pattern is nonlocalized, Pelvis is strongly mixed, and realizable FSR materially improves it;
- model representation/capacity tension when the pattern is nonlocalized, Pelvis separation is reasonable, and FSR has no material increment;
- inconclusive when prerequisites fail, no rule matches, or multiple rules match.

Exactly one resulting label and the full supporting metric object are hashed before Confirmation.

## 26. Confirmation protocol

Confirmation reuses the exact metric code, normalizers, thresholds, factor definitions, and decision rule frozen before its opening. The selected Discovery label is supported only if its entire rule matches again and, for domain localization, the same factor direction replicates. If it does not, the final study label is `SAND_GENERALIZATION_STUDY_INCONCLUSIVE`; another branch cannot be substituted after viewing Confirmation.

Confirmation is one-shot development replication. It is not available for tuning, training, model selection, or a later claim of independent final generalization.

## 27. Study verdict hierarchy

| Observation | Interpretation | Next milestone |
|---|---|---|
| Systematic factor-localized failures/low margins + reasonable Pelvis separation + limited FSR increment | `DOMAIN_DIVERSITY_GAP_SUPPORTED` | `SAND_DOMAIN_DATASET_DESIGN` |
| Broad nonlocalized Pelvis mixing + material realizable FSR separation | `PELVIS_OBSERVABILITY_TENSION_SUPPORTED` | `HAZARD_SENSOR_OBSERVABILITY_STUDY_DESIGN` |
| Broad coverage + reasonable Pelvis separation + systematic nonlocalized GRU20 failure + no material FSR increment | `MODEL_REPRESENTATION_OR_CAPACITY_TENSION_SUPPORTED` | `MODEL_V3_HYPOTHESIS_DESIGN` |
| Viability failure, contradiction, no unique match, or failed replication | `SAND_GENERALIZATION_STUDY_INCONCLUSIVE` | smallest targeted model-blind follow-up design |

These are scientific study conclusions, not model-support verdicts. The consumed final result cannot be rewritten by any branch.

## 28. Architecture implications

Pelvis IMU6 → causal 80D → GRU20 hidden 32 remains the frozen diagnostic object. `ARCHITECTURE_EVIDENCE_STILL_FAVORS_DATA_DOMAIN_STUDY` and `PELVIS_ONLY_HAZARD_STILL_PLAUSIBLE` remain current, not newly proven. No Model V3, LSTM, longer history, larger GRU, representation rewrite, threshold change, or persistence change is authorized.

Only a replicated `MODEL_REPRESENTATION_OR_CAPACITY_TENSION_SUPPORTED` result would justify a later controlled hypothesis design. It would not select an architecture by itself.

## 29. Sensor implications

The provisional 10-channel arrangement—Hazard Pelvis IMU6 and advisory Terrain left FSR4—remains plausible, while `FINAL_SENSOR_ARCHITECTURE_FROZEN = NO`. FSR8/contact analysis is descriptive and development-only. It never enters the Hazard runtime tensor, and exact support spread remains privileged ground truth.

Only a replicated `PELVIS_OBSERVABILITY_TENSION_SUPPORTED` result with material realizable-FSR increment would justify `HAZARD_SENSOR_OBSERVABILITY_STUDY_DESIGN`. No FSR fusion or Foot IMU work starts here.

## 30. Old HOLDOUT prohibition

The consumed Generalization HOLDOUT established the existence of Sand-benign generalization failure. It does **not** supply direct examples for future optimization. This study broadens the physical Sand domain symmetrically rather than reproducing three failed configurations.

No exact old `.362/.735` coordinates, run configuration, seed, contact timing, waveform, or probability target is copied. The failed source–speed cells are not oversampled. Metadata-only signature checks are permitted and show zero planned overlap; raw payload reuse, training, HNM, tuning, model selection, fresh evaluation, and guard reset remain forbidden.

## 31. Limitations

- This remains synthetic MuJoCo engineering evidence, not measured soil, real-robot, safety, production, or deployment evidence.
- The simulator is deterministic and lacks a direct stochastic gait seed or initial-phase control; geometry-induced phase diversity is not iid replication.
- Moderate/severe balanced-deformable benign yield is untested as a full matrix. The strict viability gate may stop the study and correctly force separate calibration.
- Support controls are modest, and delayed Support is calibrated only at .25 m/s on the left topology.
- Model-independent distances can diagnose overlap but cannot prove Bayes separability or production robustness.
- This design does not address the strict Slip timing limitation or create a future final HOLDOUT.

## 32. Verdict

`SAND_BENIGN_GENERALIZATION_STUDY_DESIGN_READY`

The question is identifiable; the 176-run matrix is balanced, broad, and non-HOLDOUT-targeted; physical labels and severity are objective; Discovery/Confirmation are frozen; Pelvis, realizable FSR, privileged-oracle, margin, and decision rules are predeclared; and no simulation, training, or consumed-HOLDOUT access occurred.

Frozen design hashes are:

| Contract | SHA-256 |
|---|---|
| `STUDY_PARAMETER_DOMAIN_SHA` | `c447edb8e54721beb4d5997cce00ba0884f277153eacd30e78e96e05bd899d29` |
| `STUDY_SCENARIO_MATRIX_SHA` | `31925f5719317f42eff46bca0c2ae0c8f7a8d7d7247a0304792a27970b066e38` |
| `STUDY_SPLIT_PLAN_SHA` | `7084b0f430f81cb9e676f08ae7d1995388e105f6a36b1d4019dfd8a0076eea56` |
| `STUDY_PHYSICAL_LABEL_CONTRACT_SHA` | `557b813ec9440b615a70c6fb16fda00a84b84100b938702bd754438f4304702a` |
| `STUDY_DIVERSITY_METRICS_SHA` | `dc5aeddfe5e0f6ec4e24fc582aa6d9cc2ea394f2ec1cbd8f038f474166521cee` |
| `STUDY_OBSERVABILITY_METRICS_SHA` | `5d946ac9cb908df903e24cb916182fb89cc4fc5633575264be51f1e532c10c04` |
| `STUDY_DECISION_RULE_SHA` | `5937bc4c728ca04c02fbf106383e7679358cb62d40ef6111c3333c784d770751` |
| `SAND_BENIGN_GENERALIZATION_STUDY_DESIGN_SHA` | `539f18b1e4c27abca826b7a2eac0d5c663e13035e63b1acb5fbab45136470a7f` |

All counters are zero: optimizer steps, checkpoint writes, normalizer fits, HNM rounds, threshold searches, persistence searches, architecture searches, seed searches, new simulation runs, old-HOLDOUT payload reads, inference, feature reconstruction, and visualization.

## 33. Recommended next milestone

`SAND_BENIGN_GENERALIZATION_STUDY_GENERATION`

Do not start generation automatically. The next milestone must reverify every design hash, protected artifact, protected dataset, guard state, exact historical signature exclusion, and planned matrix count before its first simulation.
