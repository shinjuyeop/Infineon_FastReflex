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

The frozen future dataset identity is `sand_benign_generalization_mild_recalibrated_study_20260903`: 176 wholly fresh signatures split 88/88 into `MILD_RECALIBRATED_DISCOVERY` and sealed `MILD_RECALIBRATED_CONFIRMATION`. It retains 96 mild broad-benign, 48 moderate boundary-adjacent benign, 24 ordinary Support, 8 delayed Support, nine-second observation, censor-aware labels, and the existing gates. Historical exact reuse, run-ID reuse, and cross-split exact/forbidden-near overlap are zero. Full generation has not started.
