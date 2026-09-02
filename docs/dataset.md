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

The separate dataset identity `generalization_hazard_reflex_20260831` contains 72 fresh runs across five predeclared scenario families. It has 36 `GENERALIZATION_VALIDATION` and 36 sealed `GENERALIZATION_HOLDOUT` runs, with no TRAIN split. Its signatures have zero overlap with the Unified 256, calibration 78, Ice-resolution 48, historical exclusions, or the opposite generalization split.

This corpus was generated model-blind and frozen before any current-candidate replay. It is not part of model training, normalization, hard-negative mining, checkpoint selection, threshold selection, or persistence selection. Physical readiness is `GENERALIZATION_DATASET_READY`. Generalization VALIDATION is now consumed development evidence: exact Model V1 and the exact promoted Model V2 were compared there, and the Model V2 primary Slip gate remained failed at 11/12 despite strong overall transfer. The exact V2 candidate is frozen for final evaluation, while Generalization HOLDOUT remains unopened at guard count 0.

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

The Generalization final-candidate readiness verifier additionally permits safe HOLDOUT IDs, counts, split membership, file existence, stored hashes, file sizes, and guard metadata. It never deserializes HOLDOUT NPZ payloads. A future authorized evaluation must claim guard `0 -> 1` atomically, open all 36 runs in one operation, and cannot claim a second scientific open.

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
