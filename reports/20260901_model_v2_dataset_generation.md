# Model V2 Dataset Generation

## 1. Purpose

This milestone executed the frozen Model V2 dataset design, retained actual MuJoCo outcomes independently of intent, audited the resulting physical coverage, and froze the run-level corpus before any training work. The result is:

```text
MODEL_V2_DATASET_GENERATION_READY
MODEL_V2_DATA_ONLY_TRAINING_READY
```

No model inference influenced generation or acceptance. No Model V2 normalizer, training windows, HNM pool, checkpoint, or model exists at the end of this milestone.

## 2. Starting state

Generation started from clean `main` at `HEAD = origin/main = 1d821c9e94ce594c9fe100006a581977dc0f40c2` (`Design Model V2 dataset`). The frozen source was `configs/experiment/20260901_model_v2_dataset_design.yaml`, SHA-256 `27076a9e85921d369587025dae828fca2603f9e6145cd5f679241e5486bf9232`.

The execution record is `configs/experiment/20260901_model_v2_dataset_generation.yaml`, SHA-256 `9ff494b97727cdee7c3f4be917176c1bdd2d3ed4e8afea980356fa4a60a80970`. It references the design rather than duplicating its scenarios.

## 3. Model V1 preservation

Model V1 remains fully restorable and was not used on the new corpus.

| Protected item | Frozen SHA-256 | Result |
|---|---|---|
| Hazard freeze | `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2` | exact |
| Hazard feature schema | `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb` | exact |
| Hazard normalizer | `610a91c24a594a3cd1701b07549f4f3c4b7f4cb04eb354f9d41378214a6800a9` | exact |
| Hazard checkpoints | `e6bada49…`, `b04877dc…`, `b6c782bd…` | 3/3 exact |
| Terrain normalizer | `2c551181fbd23080c0e3008be8484bd0ee4139e594041ba6e46ab5b36ffb05de` | exact |
| Terrain checkpoints | `21b0d122…`, `de6a55d3…`, `465803f4…` | 3/3 exact |

Hazard remains Pelvis IMU6 → causal 80D → `[20,80]` → one-layer unidirectional GRU hidden 32, three seeds, threshold `0.99`, persistence `5 ms`, 11,010 parameters. Terrain V1 remains FSR4/MLP/50 ms and advisory-only.

The protected canonical files `dataset/hazard.py`, `simulation/g1.py`, `simulation/hazards.py`, and `simulation/terrain.py` retain their exact pre-generation hashes. The new generation responsibility is isolated in `dataset/generation.py`; the supported CLI now routes the explicit generation config through the existing `collect` command.

## 4. Frozen design verification

The design was re-expanded before simulation and failed closed against every frozen identity.

| Audit | Result |
|---|---:|
| Planned runs | 412 |
| `V2_TRAIN` | 310 |
| `V2_VALIDATION` | 102 |
| Unique run IDs | 412/412 |
| Unique physical signatures | 412/412 |
| Internal duplicate signatures | 0 |
| Exact TRAIN/VALIDATION overlap | 0 |
| Cross-split near duplicates under frozen tolerances | 0 |

Resolved matrix SHA-256 is `6d109808ac20c52bc913901dd61e2eaf1541c1b8d0163e81497b63964c239bd8`. Planned physical-signature SHA-256 is `2eded4ad80c5060f57ed44d37a6fcad709fb7cca36b4b116b5b2687c2c29f297`. Split hashes are:

- `V2_TRAIN`: `8749884bec50325d1d9bb82ec5f76ce9fea0fd07594cf1f4bd921ef4a82c277e`
- `V2_VALIDATION`: `826fa70578e153f8b83c888aa16ea7ca87996ac209d55df2dcb286a174fead0c`

## 5. Signature exclusion

All 12 canonical historical manifests were hash-pinned before simulation. Exact V2 overlap was zero for every reference, including:

| Reference | V2 overlap |
|---|---:|
| Unified 256 | 0 |
| Generalization 72 | 0 |
| Scenario calibration 78 | 0 |
| Ice-resolution 48 | 0 |
| Ice-semantics 48 | 0 |
| Fall-risk dense | 0 |
| Hazard pilot | 0 |
| Hazard sensor pilot | 0 |
| Reflex event | 0 |
| Reflex-event Foot IMU | 0 |
| Sink observability | 0 |
| Terrain transition | 0 |

The historical-exclusion provenance SHA-256 is `552a4e0c12b11df8c7e5cb00d906733aa7c96f97b94f903f7aeb5495fe99f58b`.

## 6. Generation protocol

The canonical `collect` command ran all primary specifications in frozen family/split/source/cell order. Each specification followed:

```text
frozen YAML scenario
-> unchanged MuJoCo / G1 policy
-> Pelvis IMU6 + FSR8
-> privileged physical oracles
-> run and episode annotations
-> deterministic one-run NPZ
```

The policy SHA remained `2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`. Physics remained 0.5 ms and sensing 1 kHz. No reserve grid existed; reserve activation, replacement, adaptive backfill, split moves, and result-driven deletion were all zero.

## 7. Dataset identity and storage

- Dataset ID: `model_v2_hazard_reflex_20260901`
- Raw path: `data/raw/model_v2_hazard_reflex_20260901`
- Designed/executed primary runs: 412/412
- Valid/objectively invalid: 386/26
- NPZ count: 412
- Raw NPZ bytes: 156,507,098 bytes (149.257 MiB)
- Wall-clock generation: 2,451.472 s (40 min 51.472 s)
- Reserve runs: 0

The 26 invalid rows remain in the manifest and have stored traces. Every invalid reason is `pretarget_fall_when_valid_encounter_required`; no valid unexpected physical outcome was called invalid.

## 8. Run integrity

An independent post-generation pass reopened the new dataset and verified 412/412 for each of:

- NPZ file SHA-256;
- runtime shape (`timestamp_us [8000]`, `pelvis_imu6 [8000,6]`, `foot_fsr8 [8000,8]`);
- contiguous timestamps `1000..8000000 us` at exactly 1 ms;
- finite IMU and FSR;
- frozen run ID, split, scenario, and physical signature;
- frozen policy SHA;
- deterministic file and manifest metadata.

No malformed trace or sensor drop occurred. Invalidity was strictly the pre-target physical encounter boundary above.

## 9. Designed vs actual physical outcomes

Total actual family outcomes are:

| Family | Designed | Valid | Established Hazard | No established Hazard | Intent match | Mismatch | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hard normal | 24 | 24 | 0 | 24 | 24 | 0 | 0 |
| Immediate Ice Slip | 48 | 48 | 48 | 0 | 48 | 0 | 0 |
| Delayed Ice Slip | 48 | 39 | 37 | 2 | 27 | 12 | 9 |
| Ice benign | 32 | 32 | 12 | 20 | 20 | 12 | 0 |
| Ice precursor | 44 | 42 | 38 | 4 | 34 | 8 | 2 |
| Left Support | 48 | 48 | 42 | 6 | 42 | 6 | 0 |
| Right Support | 48 | 46 | 44 | 2 | 44 | 2 | 2 |
| Delayed Support | 24 | 24 | 24 | 0 | 24 | 0 | 0 |
| Staged Sand benign | 48 | 35 | 1 | 34 | 34 | 1 | 13 |
| Speed Sand benign | 48 | 48 | 0 | 48 | 48 | 0 | 0 |

Intent mismatch is diagnostic. Actual established Slip/Support always controls the physical label.

## 10. V2_TRAIN coverage

| Family | Designed | Valid | Established Hazard | No established Hazard | Match | Mismatch | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hard normal | 18 | 18 | 0 | 18 | 18 | 0 | 0 |
| Immediate Ice Slip | 36 | 36 | 36 | 0 | 36 | 0 | 0 |
| Delayed Ice Slip | 36 | 29 | 27 | 2 | 19 | 10 | 7 |
| Ice benign | 24 | 24 | 8 | 16 | 16 | 8 | 0 |
| Ice precursor | 34 | 32 | 30 | 2 | 28 | 4 | 2 |
| Left Support | 36 | 36 | 30 | 6 | 30 | 6 | 0 |
| Right Support | 36 | 34 | 32 | 2 | 32 | 2 | 2 |
| Delayed Support | 18 | 18 | 18 | 0 | 18 | 0 | 0 |
| Staged Sand benign | 36 | 27 | 1 | 26 | 26 | 1 | 9 |
| Speed Sand benign | 36 | 36 | 0 | 36 | 36 | 0 | 0 |

Actual `V2_TRAIN` totals are 290 valid, 182 established Hazard, 108 no-established-Hazard, 103 Slip, and 81 Support. Of the 108, 8 are I1-only and 10 contain a censored Ice precursor; the confirmed no-hazard count is therefore 90 and ambiguous/censored is 18. Two TRAIN runs contain both Slip and Support, so subtype counts are not required to sum to the Hazard count.

## 11. V2_VALIDATION coverage

| Family | Designed | Valid | Established Hazard | No established Hazard | Match | Mismatch | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hard normal | 6 | 6 | 0 | 6 | 6 | 0 | 0 |
| Immediate Ice Slip | 12 | 12 | 12 | 0 | 12 | 0 | 0 |
| Delayed Ice Slip | 12 | 10 | 10 | 0 | 8 | 2 | 2 |
| Ice benign | 8 | 8 | 4 | 4 | 4 | 4 | 0 |
| Ice precursor | 10 | 10 | 8 | 2 | 6 | 4 | 0 |
| Left Support | 12 | 12 | 12 | 0 | 12 | 0 | 0 |
| Right Support | 12 | 12 | 12 | 0 | 12 | 0 | 0 |
| Delayed Support | 6 | 6 | 6 | 0 | 6 | 0 | 0 |
| Staged Sand benign | 12 | 8 | 0 | 8 | 8 | 0 | 4 |
| Speed Sand benign | 12 | 12 | 0 | 12 | 12 | 0 | 0 |

Actual `V2_VALIDATION` totals are 96 valid, 64 established Hazard, 32 no-established-Hazard, 35 Slip, and 30 Support. Six no-established-Hazard runs have a censored precursor, leaving 26 confirmed no-hazard; no validation run is I1-only. One validation run contains both established subtypes.

## 12. Hazard / no-hazard balance

Fresh valid totals are established-Hazard/no-established-Hazard `246/140`, Slip `138`, and Support `111`; three runs carry both established subtypes. The no-established-Hazard block contains 8 I1-only and 16 censored-precursor runs, leaving 116 confirmed no-hazard. This is not forced back to the planned 252/160 design intent. Invalid runs are excluded from usable balance but retained in provenance.

## 13. Slip coverage

`V2_TRAIN` has 103 actual Slip runs: 87 bilateral, 8 left-only, and 8 right-only. `V2_VALIDATION` has 35 actual Slip runs: 29 bilateral, 3 left-only, and 3 right-only. Natural Ice side was never used to alter split or acceptance.

## 14. Support coverage

`V2_TRAIN` has 81 Support runs: 49 left-only, 32 right-only, and 0 bilateral. `V2_VALIDATION` has 30: 18 left-only, 12 right-only, and 0 bilateral.

Within TRAIN, ordinary left Support is 30 left-only/6 none; ordinary right Support is 32 right-only/2 none; delayed Support is 18 left-only/0 none. One staged-Sand mismatch adds the remaining left Support. The unilateral Support left:right ratio is `49:32 = 1.53125` overall, or `48:32 = 1.5` within planned Support-positive families.

## 15. Speed coverage

Actual nominal-speed coverage is:

| Split / speed | Hazard | Slip | Support | No established Hazard |
|---|---:|---:|---:|---:|
| TRAIN 0.20 | 53 | 37 | 17 | 30 |
| TRAIN 0.25 | 76 | 35 | 42 | 30 |
| TRAIN 0.30 | 53 | 31 | 22 | 48 |
| VALIDATION 0.20 | 22 | 14 | 8 | 6 |
| VALIDATION 0.25 | 24 | 11 | 14 | 10 |
| VALIDATION 0.30 | 18 | 10 | 8 | 16 |

Original Unified TRAIN had Hazard speed `0/76/0`. Fresh V2 therefore materially fixes both endpoint Hazard gaps without model inference.

## 16. Side coverage

Actual TRAIN side coverage is:

| Subtype | Left-only | Right-only | Bilateral |
|---|---:|---:|---:|
| Slip | 8 | 8 | 87 |
| Support | 49 | 32 | 0 |

The new right-only Support count is scientifically meaningful against the original zero-right-only TRAIN boundary.

## 17. Source-terrain coverage

Designed source count was exactly 206/206. Objectively invalid runs remained source-balanced, leaving valid Concrete/Marble `193/193`: TRAIN `145/145`, validation `48/48`.

Major-family valid source counts are balanced or differ by at most one: immediate Ice 24/24, delayed Ice 20/19, Ice benign 16/16, Ice precursor 21/21, left Support 24/24, right Support 23/23, delayed Support 12/12, staged Sand 17/18, speed Sand benign 24/24, and hard normal 12/12.

## 18. Ice benign

TRAIN has 24/24 valid intended Ice-benign controls: 16 physical no-hazard and 8 accidental Slip. Candidate regions occur in 17 runs; 13 include a censored candidate. No fully observed benign-release precursor episode occurs inside this specific family. Peak run drift distribution is min/p25/median/p75/max `0.02355/0.02854/0.04502/0.05743/0.12422 m`.

Validation has 8/8 valid controls: 4 no-hazard and 4 accidental Slip. All 8 reach the candidate region; 4 contain censored candidates. Physics was not altered to improve the 20/32 total no-hazard yield.

## 19. Ice precursor outcomes

Across all valid Ice families:

| Split | Valid Ice runs | Runs reaching 30–50 mm | Episodes | Same | Next | Later | Benign release | Censored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 121 | 111 | 1,376 | 747 | 87 | 415 | 28 | 99 |
| VALIDATION | 40 | 40 | 461 | 257 | 25 | 142 | 7 | 30 |

Run-balanced outcome-presence counts are TRAIN same/next/later/benign/censored `101/61/88/17/52` and validation `34/20/28/4/17`. These categories can overlap within a run because different physical-contact episodes may have different outcomes.

The dedicated precursor family provides the intended contrast. TRAIN has 32 valid runs, 31 with a candidate, 30 actual Slip, and 2 actual no-hazard; its episode outcomes are 218 same, 25 next, 137 later, 9 benign, and 20 censored. Validation has 10 valid/candidate runs, 8 actual Slip, and 2 no-hazard; episode outcomes are 47/5/20/1/16. The contrast therefore does not collapse even though Ice-benign itself has no fully observed benign-release episode.

## 20. Delayed Ice

Delayed-Ice actual episode classes are:

| Split | Valid | Exactly one benign contact before Slip | Multi-contact delayed Slip | Immediate | No Slip | Invalid | Precursor runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 29 | 1 | 26 | 0 | 2 | 7 | 27 |
| VALIDATION | 10 | 2 | 8 | 0 | 0 | 2 | 10 |

TRAIN Slip side is bilateral 23, left-only 3, right-only 1, and none 2; validation is bilateral 9 and left-only 1. The frozen `MULTI_CONTACT` cells all produced multi-contact delayed Slip when valid. Several `EXACTLY_ONE` intents physically became multi-contact; those are retained mismatches, not relabeled or regenerated.

## 21. Staged Sand hard negatives

TRAIN contains 27 valid staged-Sand controls: 26 no-I1/no-Support/no-Slip usable hard negatives and one actual Slip+Support mismatch. Nine additional TRAIN designs are invalid pre-target falls. Validation contains 8 valid benign controls and 4 invalids.

TRAIN source/speed usability is:

| Source / speed | Valid | Usable benign | Hazard | Invalid |
|---|---:|---:|---:|---:|
| Concrete 0.20 | 1 | 0 | 1 | 5 |
| Concrete 0.25 | 6 | 6 | 0 | 0 |
| Concrete 0.30 | 6 | 6 | 0 | 0 |
| Marble 0.20 | 2 | 2 | 0 | 4 |
| Marble 0.25 | 6 | 6 | 0 | 0 |
| Marble 0.30 | 6 | 6 | 0 | 0 |

This directly supplies the previously absent delayed-Sand transition negatives, but 0.20 m/s staged coverage is a documented limitation.

## 22. Right-side Support

TRAIN right-Support designs yield 32 right-only, 2 no-Support, 0 left, and 0 bilateral among 34 valid runs; 2 designs are invalid. Concrete and Marble each contribute 16 right-only plus one none. By speed, right-only counts are `8/12/12` for 0.20/0.25/0.30; the two none are at 0.20. Validation contributes 12/12 right-only.

This materially fixes the original zero-right-only Hazard TRAIN gap and is sufficient to test the side-distribution hypothesis without extra runs.

## 23. Contradictory-supervision audit

Future TRAIN extraction can be internally consistent because the raw corpus stores per-sample/per-foot precursor masks and per-episode future outcomes.

| Region | TRAIN run count | TRAIN episode count | Future handling |
|---|---:|---:|---|
| Future-Slip precursor | 101 | 1,249 | mask from ordinary negatives and HNM |
| Fully observed benign precursor | 17 | 28 | valid negative |
| Censored precursor | 52 | 99 | mask; not confirmed negative |
| I1-positive | 89 | n/a | exclude positive region from negatives |
| Established positive | 182 | n/a | exclude positive region from negatives |

No final training-window pool was built in this milestone.

## 24. Effective training-pool projection

Using only actual valid outcomes:

```text
Unified TRAIN 152
+ valid fresh V2_TRAIN 290
= effective future TRAIN 442
```

| Dimension | Unified TRAIN | Fresh valid V2_TRAIN | Effective |
|---|---:|---:|---:|
| Total | 152 | 290 | 442 |
| Hazard | 76 | 182 | 258 |
| Confirmed no hazard | 76 | 90 | 166 |
| I1-only or censored precursor | 0 | 18 | 18 |
| Slip | 38 | 103 | 141 |
| Support | 38 | 81 | 119 |
| Concrete | 76 | 145 | 221 |
| Marble | 76 | 145 | 221 |

Effective Hazard speeds are 0.20/0.25/0.30 = `53/152/53`. Effective all-run nominal strata are 0.20/0.25/0.30/off-grid-hard = `83/220/101/38`. Effective actual side is left-only/right-only/bilateral/none = `98/39/121/184`.

`V2_VALIDATION`, Generalization VALIDATION/HOLDOUT, Unified VALIDATION/HOLDOUT, calibration pilots, Ice-resolution pilots, and Ice-semantics are excluded.

## 25. Dataset freeze

- Manifest SHA-256: `7a036d3485bb19a3570a3dd4a41cc990375028f7e153e85525bd4bf19cff8b25`
- Physical-signature SHA-256: `2eded4ad80c5060f57ed44d37a6fcad709fb7cca36b4b116b5b2687c2c29f297`
- NPZ aggregate SHA-256: `5a8dfd54d1c08413dc6fb18d957269a5fbfea8cf526e4f29c78510886186e11c`
- TRAIN split SHA-256: `8749884bec50325d1d9bb82ec5f76ce9fea0fd07594cf1f4bd921ef4a82c277e`
- VALIDATION split SHA-256: `826fa70578e153f8b83c888aa16ea7ca87996ac209d55df2dcb286a174fead0c`
- Dataset-freeze SHA-256: `fd81b647051b63e7b76ab72c6dedd616cf06d752fcbc1db31ba02f8aebe68744`

The manifest, coverage audit, NPZ files, and freeze record remain Gitignored. Any future data correction requires a new explicit dataset identity/milestone; this frozen corpus is not altered before training.

## 26. HOLDOUT preservation

```text
Current Unified HOLDOUT waveform reopened: NO
Current Unified HOLDOUT new inference: NO
Generalization HOLDOUT waveform opened: NO
Generalization HOLDOUT inference: NO
Generalization HOLDOUT guard count: 0
```

Generalization VALIDATION was not replayed, trained on, normalized, mined, or used to alter scenarios.

## 27. Limitations

- 26/412 designs are objectively invalid pre-target falls: staged Sand 13, delayed Ice 9, Ice precursor 2, and right Support 2. Twenty-five occur at nominal 0.20 m/s.
- Staged-Sand valid hard-negative coverage is strong overall but weak at 0.20 m/s, especially Concrete.
- Ice benign produces 12 accidental Slip outcomes and no fully observed benign-release precursor episode within that family. The dedicated precursor family still provides benign/future/censored contrast.
- Only 3 delayed-Ice runs realize exactly one benign contact before Slip; most valid delayed runs are multi-contact.
- Left ordinary Support has six no-Support mismatches; right ordinary Support has two.
- These results remain limited to the current deterministic G1 policy, simulator, material models, oracle definitions, and frozen scenario domain.

None of these limitations prevents the data-only hypothesis from being tested. No corrective backfill is authorized in this milestone.

## 28. Verdict

```text
MODEL_V2_DATASET_GENERATION_READY
MODEL_V2_DATA_ONLY_TRAINING_READY
```

Every frozen primary design executed once; exact/near-duplicate and historical exclusions remain zero; runtime integrity passes; actual outcomes are retained; both splits contain all major families; endpoint Hazard speed, right-only Support, staged-Sand benign, Ice benign, delayed/multi-contact Ice, and precursor contrast are physically represented; and future negative extraction can mask contradictory precursor/I1/established-positive regions.

Training/search counters remain:

```text
optimizer steps = 0
checkpoint writes = 0
normalizer fits = 0
HNM rounds = 0
threshold searches = 0
persistence searches = 0
architecture searches = 0
```

Verification completed with `76 passed, 1 skipped`, compileall PASS, critical Ruff `E9,F63,F7,F82` PASS, 412/412 new run-integrity checks PASS, Unified 256/256, Generalization 72/72, and Ice-semantics 48/48 hash checks PASS. Protected Ice Slip, Sand Support, Sand benign, and Hard normal representatives retained exact timestamps, Pelvis IMU, FSR, drift/spread, contact/touchdown/censor, Slip, Support, and I1 parity.

## 29. Recommended next milestone

```text
MODEL_V2_DATA_ONLY_TRAINING
```

That future milestone may apply `RETAIN_AND_AUGMENT`, fit a new normalizer on Unified TRAIN + valid V2_TRAIN only, construct precursor-aware windows, optionally retain frozen TRAIN-only HNM exclusions, and train the unchanged three-seed GRU20 architecture. It must evaluate first on `V2_VALIDATION` and must not open Generalization HOLDOUT. It is not started here.
