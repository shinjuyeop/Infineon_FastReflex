# Dataset Contract

## 1. Current unified Hazard corpus

The current dataset identity is `unified_hazard_reflex_20260829`. Its 256 predeclared runs contain four equally sized physical groups.

| Group | Meaning | Runs |
|---|---|---:|
| `ICE_SLIP_HAZARD` | established physical Slip | 64 |
| `SAND_SUPPORT_HAZARD` | established physical Support | 64 |
| `SAND_BENIGN` | soft-ground primary no-hazard | 64 |
| `HARD_GROUND_NORMAL` | Concrete/Marble primary no-hazard | 64 |

Every group has Concrete and Marble sources. Split membership is assigned from source/index before simulation and cannot change with the observed outcome.

| Split | Per group | Total |
|---|---:|---:|
| TRAIN | 38 | 152 |
| VALIDATION | 13 | 52 |
| HOLDOUT | 13 | 52 |

Physical signatures are unique, split overlap is zero, and prior research-manifest signatures are excluded. Invalid simulation outcomes never move between splits.

## 2. Fresh zero-retrain generalization corpus

The separate dataset identity `generalization_hazard_reflex_20260831` contains 72 fresh runs across five predeclared scenario families. It has 36 `GENERALIZATION_VALIDATION` and 36 consumed `GENERALIZATION_HOLDOUT` runs, with no TRAIN split. Its signatures have zero overlap with the Unified 256, calibration 78, Ice-resolution 48, historical exclusions, or the opposite generalization split.

This corpus was generated model-blind and frozen before any current-candidate replay. It is not part of model training, normalization, hard-negative mining, checkpoint selection, threshold selection, or persistence selection. Physical readiness is `GENERALIZATION_DATASET_READY`. Generalization VALIDATION is consumed development evidence. Generalization HOLDOUT was then opened exactly once for the frozen final candidate: guard `0 -> 1`, 36/36 payloads, one deserialization per run, and V1/V2/Terrain from the same pass. Its final verdict is `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`. The guard remains permanently 1, payloads cannot be reopened, and saved summaries cannot be used as tuning or training evidence.

Generated NPZ, manifest, freeze and evaluation artifacts remain Gitignored. The committed contract and result are `configs/experiment/20260831_generalization_dataset_zero_retrain.yaml` and `reports/20260831_generalization_dataset_zero_retrain.md`.

## 3. Ice near-hazard semantics corpus

`ice_near_hazard_semantics_20260901` is a 48-run development-only corpus used to resolve the physical meaning of loaded exact-Ice drift in `[30,50) mm`. It is neither Unified TRAIN nor Model V2 TRAIN and is excluded from final performance evidence. Its result supports a separately annotated `ICE_PRECURSOR_CANDIDATE` while preserving the established 50 mm/3 ms Slip oracle.

## 4. Model V2 augmentation corpus

`model_v2_hazard_reflex_20260901` is the frozen fresh augmentation corpus for the first data-only Model V2 experiment. The predeclared 412-run matrix executed once: 310 `V2_TRAIN` and 102 `V2_VALIDATION` designs produced 386 valid and 26 objectively invalid runs. Actual outcomes are retained independently of intent; no split move, replacement, or reserve activation occurred. Dataset generation and its freeze preceded all window extraction, normalization, HNM, and training.

The raw corpus is Gitignored under `data/raw/model_v2_hazard_reflex_20260901`. Its manifest SHA-256 is `7a036d3485bb19a3570a3dd4a41cc990375028f7e153e85525bd4bf19cff8b25`, NPZ aggregate SHA-256 is `5a8dfd54d1c08413dc6fb18d957269a5fbfea8cf526e4f29c78510886186e11c`, and dataset-freeze SHA-256 is `fd81b647051b63e7b76ab72c6dedd616cf06d752fcbc1db31ba02f8aebe68744`.

The frozen future strategy is `RETAIN_AND_AUGMENT`:

```text
effective Model V2 TRAIN
= unified_hazard_reflex_20260829 TRAIN
+ valid model_v2_hazard_reflex_20260901 V2_TRAIN
```

This yielded 442 training runs from actual valid outcomes. `V2_VALIDATION`, Unified VALIDATION/HOLDOUT, Generalization VALIDATION/HOLDOUT, calibration pilots, Ice-resolution pilots, and the Ice-semantics corpus were excluded from normalization, optimization, and HNM.

The first data-only V2 training milestone fit one new 80D normalizer and the unchanged three-seed GRU20 architecture, then froze the candidate before evaluating the 96 valid `V2_VALIDATION` runs. It did not pass the frozen overall/Slip internal gates and was not promoted to an external generalization candidate. The raw dataset and its freeze hashes remain unchanged; the trained research artifacts remain separately Gitignored under `artifacts/runs/20260901_model_v2_data_only_training`.

## 5. Runtime and diagnostic fields

The Hazard model may use only:

```text
timestamp_us: int64 [N]
pelvis_imu6: float32 [N,6]
```

The corpus also stores FSR and physical clocks for Terrain scheduling, label construction and scoring. They are separately named and are not concatenated into the current Hazard input.

```text
foot_fsr8
tangential_anchor_drift_m
tangential_velocity_mps
support_surface_spread_m
support_surface_max_displacement_m
loaded_contact
first Slip/Support/I1 clocks
Ice precursor candidate mask and same/next/later/benign/censored outcome code
fall and Terrain prediction provenance
```

`HazardRun.features["PELVIS_IMU6"]` is the authoritative raw runtime tensor. `PELVIS_IMU6_FSR8` is retained only as an aligned storage/helper representation and is not accepted by `extract_hazard_features`.

## 6. Physical labels

Primary runtime label semantics are exact:

```text
established Slip OR established Support
-> HAZARD_REFLEX_REQUIRED
```

Physical labels used for dataset audit are:

- `SLIP_HAZARD`
- `SUPPORT_HAZARD`
- `SLIP_AND_SUPPORT_HAZARD`
- `NO_HAZARD`
- `SUPPORT_PRECURSOR_ONLY`

`SUPPORT_PRECURSOR_ONLY` is excluded from primary no-hazard specificity. A primary `NO_HAZARD` run has no established Slip, no I1 precursor and no established Support.

Fall/recovery, intended role and Terrain do not define the label. I1 and the established clocks are prohibited from runtime features.

## 7. Integrity and HOLDOUT

Dataset identity and generated artifact identity remain separate. A manifest records `dataset_id`, creation time, source commit, schema, policy/simulator provenance and per-run SHA-256. Each run is one NPZ with a manifest row containing its file hash, split and diagnostic summary.

Load behavior is fail-closed:

- manifest SHA mismatch fails;
- run SHA mismatch fails;
- malformed shape/dtype/nonfinite values fail;
- `HOLDOUT` waveform loading without an explicit guard fails;
- the guard can be opened only once.

Routine candidate verification reads frozen metadata and artifact hashes only. It does not open HOLDOUT waveforms.

The Generalization one-shot evaluator claimed guard `0 -> 1` atomically before the first payload read and opened all 36 runs in one operation. The result verifier now permits only saved-summary reads plus safe IDs, counts, split membership, file existence, stored hashes, file sizes, and guard metadata. It never deserializes HOLDOUT NPZ payloads and refuses a second scientific open. The consumed HOLDOUT may never be reset, reopened, relabeled, filtered, or used for candidate adaptation.

## 8. Hazard preprocessing boundary

Raw Pelvis IMU6 is converted by `src/fastreflex/features.py`, not by an experiment module. The exact output is float32 `[N,80]` with ten bases and eight causal representations. Training and replay take `[20,80]` slices ending at the declared endpoint.

No future sample, Terrain value, physical clock, fall/recovery field or time-to-event field may appear in the schema. The frozen schema hash is `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`.

## 9. Terrain dataset

The current Terrain identity is `terrain_transition_20260828`: 144 run-disjoint simulations and exact per-foot terrain-contact provenance. Clean events are terrain-identity contact rising edges with complete causal observation, persistent same-class contact, pre-fall censoring and mixed-contact ratio `<20%`.

The supported model uses only the touchdown foot's raw FSR4 window:

```text
touchdown sample t
-> foot_fsr8[t:t+50, selected foot channels]
-> float32 [50,4]
```

Exact terrain identity is a label/scheduler reference, not a model feature. Normalization is fit on TRAIN events only. Current classes are `CONCRETE`, `MARBLE`, `ICE`, and `SAND`.

## 10. Generated artifact boundary

Generated raw datasets and arbitrary training outputs live under Gitignored `data/raw/` and `artifacts/runs/`. They are not source files and are not rewritten by consolidation. Only an explicitly reviewed Research-to-Deployment release may later be placed under `artifacts/releases/` with complete provenance.

## 11. Historical datasets

Pilot NORMAL/SLIP/SINK, deformable-support, dense fall-risk, event-centric and observer datasets remain scientific provenance, not current schemas. Their configs and reports are preserved. Reproduction of a historical config uses the source commit recorded in that config/report; current source does not retain historical runners merely to keep every dated config executable.

## 12. Sand-benign calibration and redesigned study

The failed `sand_benign_generalization_study_20260902` corpus remains immutable calibration evidence. Its 176 runs and failed `STUDY_CONFIRMATION` are not reused in a new split. Detailed follow-up reads are restricted to its 88-run `STUDY_DISCOVERY`; the failed Confirmation remains sealed from model, 80D, observability, and hypothesis analysis.

Three model-blind calibration datasets contain 24, 36, and 36 runs. They are Gitignored DEVELOPMENT/CALIBRATION artifacts only and are excluded from training and future evidence. The 96-run ceiling is exhausted. The calibration contract measures exact loaded-contact phase 20 ms before first pre-censor target contact, preserves the 1,000 ms observation requirement, and treats a Support event as usable only after I1 ordering, expected side, no Slip, and 1,000 ms of post-event observation. A later fall does not erase an already fully observed event.

The generated `sand_benign_generalization_redesigned_study_20260902` corpus contains 176 immutable records split 88/88 into `REDESIGNED_DISCOVERY` and sealed `REDESIGNED_CONFIRMATION`. It produced objective-valid 153/176 and substantially corrected the first study, but failed three localized Confirmation yield gates: mild 35/48, Concrete/.25 strict 7/12, and strict Sand 52/72. Its raw records remain calibration evidence and are never moved into another split.

The subsequent model-blind mild calibration contains three Gitignored batches of 36, 12, and 24 runs. All 72 planned records were retained with no adaptive replacement; final verification was strict-benign 24/24 and 4/4 in every source-speed cell. The calibrated domain is a joint start/width/exit/topology envelope, not a width threshold: common transition-left applies to all six cells, common transition-right applies to five cells, and Concrete/.25 is left-only because replicated right profiles were strict 0/6. These pilots are calibration-only and excluded from training and future study evidence.

The generated dataset `sand_benign_generalization_mild_recalibrated_study_20260903` contains 176 wholly fresh records split 88/88 into `MILD_RECALIBRATED_DISCOVERY` and initially sealed `MILD_RECALIBRATED_CONFIRMATION`. It retains 96 mild broad-benign, 48 moderate boundary-adjacent benign, 24 ordinary Support, 8 delayed Support, nine-second observation, censor-aware labels, and the existing gates. Objective-valid yield is 169/176: strict Sand 134, Support 32, actual moderate Slip 3, and invalid 7. All 96 mild records are strict-benign, every mild source-speed cell is 8/8 in both splits, and all 70 frozen generation gates pass. Historical exact reuse, run-ID reuse, cross-split exact overlap, and forbidden planned parameter-near overlap are zero. The manifest SHA-256 is `f19ec527cb9faac0d8f3a385a1a63e8a951ced7f275c7cfc3dd459cc42f375d1`, NPZ aggregate SHA-256 is `5f63a5e4def8d09159407109f2b51635c5819931551e604c138ba1f02693f3c4`, and semantic dataset-freeze SHA-256 is `706d939c03bf31df0fb39d1043e99dbbb05922664e207425c8c96ab7c93ee675`. Generation performed no model inference or training; the seal was preserved until the later separately frozen Confirmation milestone.

The subsequent Discovery-only analysis used all 88 Discovery records in one exact frozen-V2 replay while restricting primary class metrics to 69 strict Sand and 16 valid Support controls. Strict Sand specificity was 67/69, Support recall was 16/16, and 24/69 Sand runs met the frozen adverse-margin definition. Run-balanced Pelvis-window separation passed all four reasonable-separation criteria; realizable FSR/contact improvement passed only one of four material-increment checks. Adverse margins localized to the coupled transition-left/right-single-precontact region, so the frozen Discovery hypothesis is `DOMAIN_DIVERSITY_GAP_SUPPORTED`. This does not alter labels or training membership. At that checkpoint the 88 Confirmation payloads remained sealed and unanalyzed pending a separate replication milestone.

That separate Confirmation milestone subsequently claimed its study-specific guard exactly once and deserialized all 88 Confirmation payloads once. Primary metrics used 65 strict Sand and 16 Support; three actual Slips remained descriptive and four invalids remained provenance-only. Exact V2 replay was 61/65 specific on strict Sand, 29/65 adverse, and 16/16 on Support. The coupled localization direction and non-material FSR result replicated, but Pelvis-window reasonable separation passed only 3/4 because centroid separation was `.208030`, below `.75`; the valid scientific verdict is `DOMAIN_DIVERSITY_GAP_NOT_CONFIRMED`. Confirmation is now consumed development evidence and is excluded by default from training, HNM, tuning, model selection, and future fresh-final evidence.

The following hypothesis review uses only the saved analysis artifacts and does not alter dataset membership or eligibility. It selects `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS` for a future independent study because topology/phase direction and model-margin behavior persist while window-space spread is factor-conditioned and local class neighborhoods remain strong. Any next training-domain design must create new scenarios with symmetric predeclared coverage; it may not copy or adaptively mine consumed Confirmation, use the historical HOLDOUT, or treat either as fresh final evidence.

The fresh dataset `sand_factor_conditioned_development_20260903` retains one predeclared 162-run attempt: 108 `FACTOR_TRAIN` and 54 `FACTOR_VALIDATION`, with mild/moderate Sand and ordinary/delayed Support represented in all six source-speed cells. Exact historical reuse, forbidden historical near reuse, run-ID reuse, and cross-split exact/near reuse are all zero. The generated records comprise 42 strict-benign Sand, 39 valid-intent Support, 5 Slip, 2 dual Hazard, and 74 invalid outcomes. Only 81/162 satisfy the actual-physics eligibility contract, below the frozen minimum 140; all source-speed Sand minima and both roles' mild and Support minima fail. The corpus is therefore frozen as failed physical-development evidence and is prohibited from training or model evaluation in this cycle. Its semantic dataset-freeze SHA-256 is `4906682f9366bad572baeb529db81ca1d5b1b2878f1cd2e4782999e2588cd549`.

The metadata-only failure audit preserves every one of those 162 records and classifies them as 42 strict Sand, 39 Support, 5 Slip, 2 dual Hazard, 30 pretarget falls, 42 target-following fall censors, and 2 other invalid controls. Every mild run lay outside the prior topology-specific viable envelope. All 42 insufficient-follow-up records contain an actual fall, so neither label semantics nor the 1,000 ms post-target observation requirement changes. The actionable cause is family-specific joint geometry and contact-sequence instability, conditioned by source, speed, and the coupled topology/precontact-phase manifold; no generator or physical-accounting bug was found.

Two pre-frozen model-blind calibration datasets, `sand_factor_conditioned_physical_domain_calibration_20260903` and `sand_factor_conditioned_concrete_025_calibration_20260903`, contain 24 and 8 runs. They use 32/64 allowed pilot simulations and produce 29/32 strict Sand with no pretarget fall, Slip, or dual Hazard. Both datasets are Gitignored calibration-only evidence and are permanently excluded from training, validation, and future corpus membership.

The recalibrated dataset `sand_factor_conditioned_development_recalibrated_20260903` contains the exact 198 fresh planned records: `FACTOR_TRAIN` 132 and `FACTOR_VALIDATION` 66. One-pass model-blind generation completed all 198 without a pilot, replacement, backfill, or rerun. The corpus contains 143 strict Sand, 34 ordinary Support, 12 delayed Support, 1 Slip, 5 dual Hazard, and 3 other invalid records; objective-valid yield is 189/198. Mild Sand is 108/108 strict and reduced moderate Sand is 35/36 strict. All Sand, source-speed, coupled-manifold, contamination, uniqueness, and overlap gates pass.

The two delayed-Support gates fail at TRAIN 9/12 against minimum 10 and VALIDATION 3/6 against minimum 5, so the conjunctive generation verdict is `SAND_FACTOR_CONDITIONED_DEVELOPMENT_RECALIBRATED_GENERATION_INSUFFICIENT`. The failed corpus is immutable physical-development evidence and cannot enter training or evaluation. `FACTOR_VALIDATION` is `SEALED_FAILED_PHYSICAL_EVIDENCE`; its model inference, training use, and HNM counters are zero, and its loader rejects split identity before payload access. Manifest SHA-256 is `776c2a22c8963d5ddcdad49d2f36c109a21af4dfa1c0b5d2a924112b1fb6b19c`, NPZ aggregate SHA-256 is `c64c934ce51fab1669b1c0badb5b645d2eb15a7ac10f72d43b7dc17295f6b6fa`, and semantic dataset-freeze SHA-256 is `d7a7b06095ce80e0bfdc5766e9a8265178e8ef184e0ec3251d35eab555588f84`.

The next one-pass support-recalibrated corpus also contains exactly 198 frozen records and remains immutable failed physical evidence. Sand passed its physical gates (Mild 107/108, Moderate 33/36), and delayed Support passed at TRAIN 11/12 and VALIDATION 6/6. The sole failed gate was TRAIN ordinary Support at 20/24 against 22; VALIDATION passed at 11/12. All four observation-invalid ordinary controls established Support and then physically fell with only 563–990 ms available, so neither the nine-second horizon nor the 1,000 ms observation contract changes. One additional ordinary control was a genuine Dual Hazard.

The saved-evidence ordinary review freezes a new, ungenerated `sand_factor_conditioned_development_controls_recalibrated_20260903` design: 132 TRAIN plus 66 VALIDATION, with Mild 108, Moderate 36, ordinary Support 36, and delayed Support 18. Ordinary profiles are now explicit per source-speed cell; Concrete/.30 is right-only, Marble/.30 left uses only the higher-start stable strip, and the late/long Concrete/.20 left dual corner is excluded. The physical Support label and mechanics are unchanged. Sand mechanics, manifold structure, and Concrete/.25 exception are preserved; delayed Support remains LEFT_ONLY in the exact `.324–.332/.825–.833/1.153–1.165` envelope. All 198 planned IDs and signatures are unique, and historical/cross-split exact, forbidden-near, and run-ID overlaps are zero. Generation, training, and model inference have not started.
