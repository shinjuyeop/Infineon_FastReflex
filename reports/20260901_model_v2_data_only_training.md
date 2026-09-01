# Model V2 Data-Only Training

## 1. Purpose

This milestone tested the frozen data-only hypothesis: retain the Model V1 Pelvis IMU6, causal 80D, GRU20 architecture and `0.99 / 5 ms` decision while replacing only the TRAIN normalizer/weights and expanding TRAIN coverage. Training completed with valid provenance, but the frozen candidate failed the predeclared internal overall-Hazard and Slip-recall gates.

```text
MODEL_V2_DATA_ONLY_TRAINING_COMPLETE
MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED
```

No retraining, threshold search, persistence search, architecture search, seed search, Generalization VALIDATION inference, or HOLDOUT access followed the result.

## 2. Starting state

Work started from clean `main` at `HEAD = origin/main = 44818915e873fdb7df137bd20d36cfcb0d927f4f` (`Generate Model V2 dataset`). The dataset-generation verdicts were `MODEL_V2_DATASET_GENERATION_READY` and `MODEL_V2_DATA_ONLY_TRAINING_READY`.

The training protocol was committed to `configs/experiment/20260901_model_v2_data_only_training.yaml` before the first normalizer fit or optimizer step. Its SHA-256 is `3370319500fe5df2a4dcf64b410c656e3290c11bfee465f1ccfb29dd5bbf7a22`.

## 3. Model V1 preservation

`MODEL_V1_RESTORABLE = YES`. V1 was replayed read-only on V2_VALIDATION only after the V2 candidate freeze; no V1 artifact was overwritten.

| Protected item | SHA-256 / result |
|---|---|
| Hazard freeze | `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2` exact |
| Feature schema | `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb` exact |
| Hazard normalizer | `610a91c24a594a3cd1701b07549f4f3c4b7f4cb04eb354f9d41378214a6800a9` exact |
| Hazard checkpoints | `e6bada49…`, `b04877dc…`, `b6c782bd…`, 3/3 exact |
| Terrain normalizer | `2c551181fbd23080c0e3008be8484bd0ee4139e594041ba6e46ab5b36ffb05de` exact |
| Terrain checkpoints | `21b0d122…`, `de6a55d3…`, `465803f4…`, 3/3 exact |

The historical Unified HOLDOUT result and `UNIFIED_HAZARD_REFLEX_SUPPORTED_SINGLE_IMU` verdict remain unchanged. The historical Generalization VALIDATION V1 result and `ZERO_RETRAIN_GENERALIZATION_NOT_SUPPORTED` verdict also remain unchanged.

## 4. Dataset freeze verification

The V2 corpus remained byte-exact before and after training.

| Item | Frozen value | Result |
|---|---|---|
| Dataset ID | `model_v2_hazard_reflex_20260901` | exact |
| Designed / executed | 412 / 412 | exact |
| Valid / invalid | 386 / 26 | exact |
| Valid TRAIN / validation | 290 / 96 | exact |
| Dataset-freeze SHA | `fd81b647051b63e7b76ab72c6dedd616cf06d752fcbc1db31ba02f8aebe68744` | exact |
| Manifest SHA | `7a036d3485bb19a3570a3dd4a41cc990375028f7e153e85525bd4bf19cff8b25` | exact |
| NPZ aggregate SHA | `5a8dfd54d1c08413dc6fb18d957269a5fbfea8cf526e4f29c78510886186e11c` | exact |
| TRAIN split SHA | `8749884bec50325d1d9bb82ec5f76ce9fea0fd07594cf1f4bd921ef4a82c277e` | exact |
| VALIDATION split SHA | `826fa70578e153f8b83c888aa16ea7ca87996ac209d55df2dcb286a174fead0c` | exact |
| NPZ verification | 412/412 | PASS |
| Historical/split/near-duplicate overlap | 0/0/0 | PASS |

Unified 256/256, Generalization 72/72, and Ice-semantics 48/48 NPZ hashes also passed. File hashing did not decode or inspect sealed HOLDOUT waveforms.

## 5. Effective TRAIN composition

The frozen `RETAIN_AND_AUGMENT` pool was exactly Unified TRAIN 152 + valid V2_TRAIN 290 = 442 runs. Effective identity SHA-256 is `0466ea84871d178856ffb10d8b1c0cec730286b35b589c73ff1b5fb1065aa5ab`.

| Dimension | Actual count |
|---|---:|
| Total | 442 |
| Established Hazard | 258 |
| Confirmed no hazard | 166 |
| I1-only/censored ambiguous | 18 |
| Slip | 141 |
| Support | 119 |
| Dual Slip+Support | 2 |
| Concrete / Marble | 221 / 221 |
| Hazard speed .20 / .25 / .30 | 53 / 152 / 53 |
| Actual side left / right / bilateral / none | 98 / 39 / 121 / 184 |

V2_VALIDATION, Unified VALIDATION/HOLDOUT, Generalization VALIDATION/HOLDOUT, calibration/Ice-resolution pilots, and Ice-semantics were absent from normalization, optimization, and HNM.

## 6. Frozen training protocol

- Runtime input: Pelvis IMU6 only.
- Features: unchanged causal 80D schema.
- Model: one-layer unidirectional GRU, hidden 32, two outputs, 11,010 parameters.
- History: `[20,80]` at 1 kHz.
- Seeds: `20260828`, `20260829`, `20260830`; all were retained.
- Optimizer: Adam, learning rate `0.001`, weight decay `0`.
- Loss: inverse-frequency weighted cross-entropy.
- Batch size: 128; maximum 40 epochs; patience 6.
- Initialization/shuffle: deterministic PyTorch defaults after each fixed seed.
- Runtime decision: mean three-seed probability `>=0.99` for 5 consecutive samples.
- HNM: exactly three effective-TRAIN-only rounds, 1 ms replay, maximum 12/run, minimum 30 ms within-round spacing.

No gradient clipping or synthetic side mirroring was introduced. No V2_VALIDATION result changed this protocol.

## 7. Target/window semantics

Slip positives retained `[-30,+40] ms` at 5 ms stride. Support positives retained five I1-to-established points plus established-relative offsets `[-20,0,+20,+40] ms`, with a union cap of 20/run.

Negative precedence was frozen as:

1. canonical established-Hazard positive;
2. Support I1-positive;
3. future-Slip Ice precursor mask;
4. censored/ambiguous mask;
5. confirmed benign negative.

The V1 whole-run pre-event negative boundary was retained. One correctness extension, discovered during the pre-optimizer dry audit, allows explicitly annotated fully observed `BENIGN_RELEASE` endpoints after a resolved earlier event only when the endpoint is outside active Slip, every canonical positive range, I1, future-Slip/censored precursor, fall, and censor. The corrected dry audit was frozen before any optimizer step.

## 8. Ice precursor masking

| Mask | TRAIN runs | TRAIN samples | Ordinary-negative violations |
|---|---:|---:|---:|
| Future-Slip precursor | 101 | 41,479 | 0 |
| Censored precursor | 52 | 1,734 | 0 |
| I1-positive | 89 | 68,388 | 0 |
| Post-fall/censor | n/a | n/a | 0 |

Canonical Slip/Support positive endpoints take precedence over masks. Fully observed benign-release endpoints remain negative evidence.

## 9. Hard-negative extraction

Pretraining extraction audit SHA-256 is `b5b0f21e091763e4353f8e6a669c3e24c6e4a7dcfb8179194d2d39b9b697b0ba`; materialized extraction audit SHA-256 is `61ce088b6a60b61aa6e252e8fe5ff8f1985f84f0d28858e86f9bf8b845f64f73`.

| Window role | Windows | Runs | Notes |
|---|---:|---:|---|
| Slip positive | 2,111 | 141 | actual Slip |
| Support positive | 950 | 119 | actual Support, I1 semantics |
| Hard normal negative | 4,083 | 56 | Unified + V2 hard controls |
| Ice benign negative | 1,862 | 24 | meaningful target-contact anchors |
| Benign near-threshold Ice negative | 240 | 17 | fully observed release only |
| Staged Sand benign negative | 2,055 | 27 | target transition/contact included |
| Speed Sand benign negative | 2,893 | 36 | all three nominal speeds |
| Other confirmed benign negative | 21,076 | 299 | run/gait capped |
| **Initial total** | **35,270** | **442** | 3,061 positive + 32,209 negative |

## 10. Sampling balance

Positive windows by nominal speed `.20/.25/.30` were `688/1,732/641`; `.25` was 56.6%, not a return to the original all-Hazard-at-.25 boundary. Negative windows were `.20/.25/.30 = 6,698/15,620/7,658`, plus 2,233 off-grid hard-control windows. Positive source counts were Concrete/Marble `1,564/1,497`; negative source counts were `15,852/16,357`.

The deterministic policy caps positives per run and negatives per gait/contact role. It does not weight families using V2_VALIDATION results.

## 11. New V2 normalizer

- Fit source: all 442 effective-TRAIN runs only.
- Method: unchanged per-channel z-score over causal 80D features.
- Per-run cap: 1,000 samples.
- Shape: mean `[80]`, std `[80]`.
- Fitted samples: 442,000.
- Fit-run identity SHA: `2610d3958ccba0b7880a3f5137402cf7f099351281d49a16e4c931f35538114d`.
- Normalizer SHA: `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a`.
- Logical fits: 1.

The V1 normalizer remained exact and separate.

## 12. Model architecture parity

Architecture SHA-256 is `ae4753699399aa7e4a93639935e085156ef28ed100a77058d7308eb1549aa897`. The instantiated model remained parameter-identical to V1: Pelvis IMU6 → causal 80D → `[20,80]` → GRU hidden 32, one layer, unidirectional, two outputs, 11,010 parameters.

## 13. Seed training

The HNM count shown for rounds 0–2 is the shared ensemble-mined pool added after that round. No seed was selected or dropped.

| Seed | Round | Positives | Negatives | HNM selected | Best/completed epochs | Best monitor CE | Checkpoint SHA |
|---:|---:|---:|---:|---:|---:|---:|---|
| 20260828 | 0 | 2,424 | 25,585 | 5,304 | 4/10 | 0.152215 | `8b92158e…` |
| 20260829 | 0 | 2,424 | 25,585 | 5,304 | 9/15 | 0.148357 | `a41f54ca…` |
| 20260830 | 0 | 2,424 | 25,585 | 5,304 | 13/19 | 0.131224 | `2a07ff61…` |
| 20260828 | 1 | 2,424 | 29,632 | 5,304 | 12/18 | 0.160234 | `d7a72294…` |
| 20260829 | 1 | 2,424 | 29,632 | 5,304 | 18/24 | 0.149869 | `caec78c2…` |
| 20260830 | 1 | 2,424 | 29,632 | 5,304 | 17/23 | 0.163742 | `9f984b87…` |
| 20260828 | 2 | 2,424 | 33,731 | 5,304 | 7/13 | 0.182891 | `dbf965c2…` |
| 20260829 | 2 | 2,424 | 33,731 | 5,304 | 4/10 | 0.219321 | `5894e5e5…` |
| 20260830 | 2 | 2,424 | 33,731 | 5,304 | 14/20 | 0.190544 | `394fac3a…` |
| 20260828 | 3 | 2,424 | 37,854 | 0 | 11/17 | 0.168871 | `dd6c8581…` |
| 20260829 | 3 | 2,424 | 37,854 | 0 | 7/13 | 0.172977 | `8e6709da…` |
| 20260830 | 3 | 2,424 | 37,854 | 0 | 13/19 | 0.204910 | `811f486c…` |

Total optimizer steps were 53,555. Twelve round/seed checkpoints were written; only the three Round-3 checkpoints are ensemble members.

## 14. HNM round 1

## 15. HNM round 2

## 16. HNM round 3

All three rounds had the same frozen capacity and coverage audit:

| Round | Eligible runs | Selected windows | Contributing runs | Duplicate endpoints | Spacing violations | Forbidden-mask violations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 442 | 5,304 | 442 | 0 | 0 | 0 |
| 2 | 442 | 5,304 | 442 | 0 | 0 | 0 |
| 3 | 442 | 5,304 | 442 | 0 | 0 | 0 |

Each round selected exactly 12 eligible endpoints per run. The family distribution was stable because counts followed the fixed run cap; for example each round selected Immediate Ice 432, Delayed Ice 348, Ice benign 288, staged Sand 324, speed Sand 432, left Support 432, and right Support 408. HNM provenance SHA-256 is `594b7a77091f1317e8b54fa272deee6776e7dee6399406251876310ffab90a0b`.

## 17. Candidate freeze

Before any V2_VALIDATION waveform load, the candidate was frozen as `model_v2_data_only_gru20_20260901`.

- Candidate-freeze SHA: `edb16b7d96fb38e680e36dcca5ecd7b1c1682ba410b94d21f40c2a38d4ed8725`.
- Normalizer SHA: `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a`.
- Seed `20260828`: `dd6c8581161963265d4323b8316f01367e359357673f0596faaa2a27051771c8`.
- Seed `20260829`: `8e6709da112845840aae0094dd997fad4ee7f9d8256a2ee0fc5e9a0df3b724a0`.
- Seed `20260830`: `811f486c1bd47f91a854fdbd004b8408a5f00bfaa83a22ce91608de1d3b54c42`.
- Extraction-policy SHA: `46aa8e72782e0749605cc0100376adbe3e97b97159b332079c95fbd0e1acda74`.
- Feature schema: unchanged.
- Threshold/persistence: `0.99 / 5 ms`.

Because internal validation failed, no `MODEL_V2_FREEZE_SHA` or external-generalization candidate was created. All training artifacts remain preserved for provenance.

## 18. V2_VALIDATION primary results

The candidate was evaluated once on the 96 valid V2_VALIDATION runs after freeze. Result artifact SHA-256 is `3b239225a7fb4ae7986b5910b803413e7cf5a22d3a56cca2c636a74d66441ccc`.

| Metric | V2 | Gate | Result |
|---|---:|---:|---|
| Overall Hazard recall | 55/64 = 85.94% | >=90% | FAIL |
| Slip recall | 29/35 = 82.86% | >=95% | FAIL |
| Support recall | 27/30 = 90.00% | >=85% | PASS |
| Confirmed no-hazard specificity | 26/26 = 100% | >=95% | PASS |
| Ice-benign no-established-Hazard specificity | 4/4 = 100% | >=95% | PASS |
| Premature rate | 6/64 = 9.38% | <=10% | PASS |
| Slip latency median / p95 | -18 / +9.5 ms | p95 <=+40 ms | PASS |
| Support established latency median / p95 | -21 / -1.1 ms | p95 <=+50 ms | PASS |
| Staged Sand specificity | 8/8 = 100% | >=95% | PASS |
| Speed Sand specificity | 12/12 = 100% | >=95% | PASS |
| Right-only Support recall | 12/12 = 100% | >=85% | PASS |

False Reflex count among confirmed no-hazard runs was 0. Nine Hazard runs failed the original primary timing contract: six Ice runs were premature and three delayed-Support runs were missed/out of range.

## 19. Slip

Slip improved from V1 `22/35` to V2 `29/35`, but six valid Ice trajectories fired before the original `Slip-30 ms` boundary. Valid detections had median/p95 `-18/+9.5 ms`, so latency was acceptable once detection entered the original window. The unresolved issue is primary boundary timing/coverage, not late response among counted detections.

## 20. Support

Support improved from V1 `12/30` to V2 `27/30`. Ordinary left and right Support were strong, while delayed Support remained 3/6. Three Concrete delayed-Support runs detected at I1+43 ms / established-13 ms. Marble had one no-alert run and two alerts about 599 ms after established Support, outside the frozen valid interval. No delayed-Support run was pre-I1 premature.

## 21. Right-only Support

Right-only Support improved from V1 `0/12` to V2 `12/12`. This directly validates the data-only side-coverage intervention for ordinary Support; no mirroring, Terrain gate, sensor addition, or architecture change was needed.

## 22. Staged Sand benign

All 8 eligible validation controls were true negatives: specificity `8/8`, false Reflex `0`, and no first Reflex relative to static entry. The systematic V1-style pre-I1 staged-entry false alert did not recur.

## 23. Ice benign

The four actual no-established-Hazard Ice controls had specificity `4/4`, no threshold excursion, peak drifts `33.46, 34.80, 48.11, 49.43 mm`, and candidate occurrence `4/4`. All four candidate outcomes were censored and none was a fully observed benign release, so confirmed-Ice-benign specificity has denominator zero and the 4/4 primary family result must not be overstated. The four accidental-Slip controls had primary recall 2/4; the other two were premature.

## 24. Delayed Ice

Delayed Ice recall was `8/10`; both failures were premature multi-contact cases. Exactly-one behavior was `2/2` detected with no premature result. Multi-contact behavior was `6/8` under the primary window, with the physical outcomes retained unchanged.

## 25. Speed robustness

| Speed | Slip recall | Support recall | Confirmed no-hazard specificity |
|---:|---:|---:|---:|
| 0.20 | 9/14 = 64.29% | 8/8 = 100% | 6/6 = 100% |
| 0.25 | 11/11 = 100% | 11/14 = 78.57% | 10/10 = 100% |
| 0.30 | 9/10 = 90.00% | 8/8 = 100% | 10/10 = 100% |

Endpoint-speed data improved V1 substantially, but 0.20 m/s Slip remained the weakest stratum and `.25` delayed Support retained a timing gap.

Actual-side results were:

| Subtype / side | V2 recall |
|---|---:|
| Slip left-only | 3/3 |
| Slip right-only | 0/3 |
| Slip bilateral | 26/29 |
| Support left-only | 15/18 |
| Support right-only | 12/12 |
| Support bilateral | n/a (0 runs) |

## 26. Precursor-aware secondary result

The secondary view did not rewrite any primary score.

| Episode outcome | Episodes | Alert in candidate | Alert before established Slip |
|---|---:|---:|---:|
| Same episode | 257 | 53 | 54 |
| Next episode | 25 | 0 | 2 |
| Later | 142 | 19 | 47 |
| **Future Slip total** | **424** | **72 (16.98%)** | **103 (24.29%)** |
| Benign release | 7 | 0 false alerts | n/a |
| Censored | 30 | 1 | reported separately |

Candidate-region alert timing relative to the 30 mm crossing had median/p95 `+3/+21.8 ms`. Signed Reflex-to-earliest-established-Slip timing had median/p95 `+21/+191.05 ms`; negative minima represent cross-foot candidate alerts after an earlier any-foot established event and do not rescue primary scoring. There were 352 future-Slip precursor episodes without an alert inside the candidate region.

## 27. V1 vs V2 on V2_VALIDATION

V1 replay occurred read-only only after the V2 candidate freeze. Its result SHA-256 is `3745d92dabb22cb1c5d06f48878fb8d216705285853025f6de15ad5bbbcbd170`.

| Metric | V1 | V2 | Delta |
|---|---:|---:|---:|
| Overall Hazard recall | 53.12% | 85.94% | +32.81 pp |
| Slip recall | 62.86% | 82.86% | +20.00 pp |
| Support recall | 40.00% | 90.00% | +50.00 pp |
| Confirmed no-hazard specificity | 88.46% | 100% | +11.54 pp |
| Premature rate | 20.31% | 9.38% | -10.94 pp |
| Right-only Support recall | 0% | 100% | +100 pp |
| Staged Sand specificity | 100% | 100% | 0 pp |
| Ice-benign specificity | 100% | 100% | 0 pp |

The data intervention produced large, coherent improvements, but the predeclared gates—not relative improvement—control the verdict.

## 28. Limitations

- Six Slip-bearing Ice runs still fired earlier than the original primary window; precursor-aware interpretation cannot hide those primary failures.
- 0.20 m/s Slip was only 9/14, and right-only Slip was 0/3 despite limited right-Slip TRAIN coverage.
- Delayed Support generalized asymmetrically by source: Concrete 3/3 versus Marble 0/3.
- V2_VALIDATION had no confirmed benign Ice precursor-release run; its four no-established-Hazard Ice controls were precursor-censored.
- Candidate-region future-Slip alert coverage was only 72/424 correlated episodes; episode counts are diagnostic, not independent-run accuracy.
- Results remain simulator/policy/domain specific.

The pattern points first to a target/extraction/coverage failure audit—especially early Ice timing, 0.20/right Slip, and delayed Marble Support. Because ordinary right Support and hard negatives improved strongly with unchanged capacity, architecture change is not yet justified.

## 29. Verdict

```text
MODEL_V2_DATA_ONLY_TRAINING_COMPLETE
MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED
ARCHITECTURE_CHANGE_NOT_YET_JUSTIFIED
```

Training integrity is valid: optimizer steps 53,555; checkpoint writes 12; logical normalizer fits 1; HNM rounds 3; threshold/persistence/architecture/seed searches 0. V2_VALIDATION optimizer leakage, Generalization VALIDATION training leakage, and HOLDOUT leakage were all zero.

Verification completed with `78 passed, 1 skipped`, compileall PASS, critical Ruff `E9,F63,F7,F82` PASS, changed-file default Ruff PASS, dataset hashes PASS, V1 protected-candidate verification PASS, and Generalization HOLDOUT guard 0.

## 30. Recommended next milestone

```text
MODEL_V2_INTERNAL_FAILURE_AUDIT
```

That milestone should inspect the frozen candidate’s six premature Ice cases, three delayed-Support failures, 0.20/right-only Slip pattern, and per-seed/ensemble timing without retraining or changing the current evidence. `MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION` is not authorized because internal validation was not supported.

External evidence status remains:

```text
Generalization VALIDATION V2 inference: NO
Current Unified HOLDOUT waveform reopened: NO
Current Unified HOLDOUT new inference: NO
Generalization HOLDOUT waveform opened: NO
Generalization HOLDOUT inference: NO
Generalization HOLDOUT guard count: 0
```
