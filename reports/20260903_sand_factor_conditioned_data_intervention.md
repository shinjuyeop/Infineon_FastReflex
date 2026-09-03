# Factor-Conditioned Sand Data Intervention

## 1. Purpose

This milestone executed one fresh, data-only intervention attempt for the frozen `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS`. It was designed to add factor-conditioned Sand benign and Support-control data while preserving the exact Pelvis-IMU6 GRU20 interface. The attempt stopped at the predeclared physical-generation gate, before training or model evaluation.

## 2. Starting state

The repository began this scientific cycle at `d7ee4aa37e4d5319d63b993c5fcb2bf846d916ab` with `HEAD == origin/main` and a clean tracked worktree. The historical final Model V2 verdict remains `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`; whole-simulation evidence remains `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`; the independently supported branch remains `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`.

## 3. Historical evidence boundary

The consumed Generalization HOLDOUT guard remained `1`, scientific open count remained `1`, and this milestone performed zero HOLDOUT payload reads, inference, feature reconstruction, or visualization. Old Sand Discovery/Confirmation and pilot payloads were not opened or trained on. Only saved reports, analysis artifacts, and manifest metadata were used to design and audit scenario uniqueness. Unified TRAIN, V2_TRAIN, and the planned one-time post-freeze V2_VALIDATION regression were the only authorized model-data roles; the physical gate prevented all of those training/evaluation operations from starting.

## 4. Scientific hypothesis

The selected hypothesis was exactly `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS`: fresh, symmetric coverage over topology, precontact phase, source, speed, and Sand severity could reduce the prior factor-conditioned benign weakness without changing sensors, features, model architecture, extraction policy, threshold, or persistence.

## 5. Fresh-domain design

The frozen design used six Concrete/Marble × 0.20/0.25/0.30 m/s cells and four groups: mild Sand, moderate Sand, ordinary Support, and delayed Support. It planned 162 main simulations—108 `FACTOR_TRAIN` and 54 `FACTOR_VALIDATION`—with no pilot, adaptive replacement, or backfill. Concrete/0.25 mild/moderate Sand used the previously established left-only physical exception. All scenario identities and split assignments were frozen before run 1.

## 6. Anti-contamination audit

| Check | Result | Compliance |
|---|---:|---|
| Historical exact scenario overlap | 0 | PASS |
| Historical forbidden near overlap | 0 | PASS |
| Historical run-ID reuse | 0 | PASS |
| Old Sand study scenario reuse | 0 | PASS |
| Old Discovery payload/training reuse | 0 | PASS |
| Old Confirmation payload/training reuse | 0 | PASS |
| Calibration-pilot payload/training reuse | 0 | PASS |
| FACTOR_TRAIN ↔ FACTOR_VALIDATION exact overlap | 0 | PASS |
| FACTOR_TRAIN ↔ FACTOR_VALIDATION forbidden near overlap | 0 | PASS |
| V2_VALIDATION training use | 0 | PASS |
| Generalization VALIDATION training use | 0 | PASS |
| Historical HOLDOUT payload access/inference | 0 / 0 | PASS |

The audit covered 13 historical manifests and used metadata only.

## 7. Physical calibration if any

No optional calibration pilot was run (`0/64`). The main matrix relied on the previously saved mild-domain physical calibration, but introduced wholly fresh start/width pairs to eliminate historical and cross-split near reuse. The fresh wide-patch timing did not transfer with sufficient physical yield; no after-the-fact geometry replacement was allowed.

## 8. Fresh corpus

Dataset ID: `sand_factor_conditioned_development_20260903`. All 162 planned 9-second, 1 kHz simulations completed once in 1,053.670 seconds. `N` below is actual eligible / planned; the parenthesized counts are eligible at 0.20/0.25/0.30 m/s.

| Role | Group | Source | Speed | Topology/eligible phase coverage | N |
|---|---|---|---|---|---:|
| FACTOR_TRAIN | mild Sand | Concrete | .20/.25/.30 | left/right; left/right single | 7/24 (1/0/6) |
| FACTOR_TRAIN | mild Sand | Marble | .20/.25/.30 | left/right; left/right single | 8/24 (1/4/3) |
| FACTOR_TRAIN | moderate Sand | Concrete | .20/.25/.30 | left/right; left/right single | 7/12 (2/3/2) |
| FACTOR_TRAIN | moderate Sand | Marble | .20/.25/.30 | left/right; left/right single | 8/12 (3/2/3) |
| FACTOR_TRAIN | ordinary Support | Concrete | .20/.25/.30 | left/right; left/right single | 9/12 (3/2/4) |
| FACTOR_TRAIN | ordinary Support | Marble | .20/.25/.30 | left/right; left/right single | 9/12 (2/3/4) |
| FACTOR_TRAIN | delayed Support | Concrete | .20/.25/.30 | left; right single | 5/6 (1/2/2) |
| FACTOR_TRAIN | delayed Support | Marble | .20/.25/.30 | left; right single | 4/6 (1/1/2) |
| FACTOR_VALIDATION | mild Sand | Concrete | .20/.25/.30 | left/right; right single | 2/12 (0/0/2) |
| FACTOR_VALIDATION | mild Sand | Marble | .20/.25/.30 | left/right; right single | 4/12 (0/2/2) |
| FACTOR_VALIDATION | moderate Sand | Concrete | .20/.25/.30 | left/right; right single | 3/6 (0/2/1) |
| FACTOR_VALIDATION | moderate Sand | Marble | .20/.25/.30 | left/right; left/right single | 3/6 (1/1/1) |
| FACTOR_VALIDATION | ordinary Support | Concrete | .20/.25/.30 | left/right; left/right single | 4/6 (1/1/2) |
| FACTOR_VALIDATION | ordinary Support | Marble | .20/.25/.30 | left/right; left/right single | 4/6 (1/1/2) |
| FACTOR_VALIDATION | delayed Support | Concrete | .20/.25/.30 | left; right single | 2/3 (0/1/1) |
| FACTOR_VALIDATION | delayed Support | Marble | .20/.25/.30 | left; right single | 2/3 (0/1/1) |

Concrete/0.25 Sand was left-only by design; other Sand cells planned both topologies. Actual outcomes were 42 strict benign, 39 Support, 5 Slip, 2 dual Hazard, and 74 invalid. The invalid reasons were 42 insufficient post-target observation, 30 pretarget fall, and 2 physical outcome mismatch. Hazard contamination and invalid records remain frozen provenance only.

## 9. Physical-generation gates

Overall result: **FAIL** — `FACTOR_CONDITIONED_DATASET_GENERATION_GATES_FAILED`.

| Gate | Actual | Required | Result |
|---|---:|---:|---|
| Complete planned execution | 162 | 162 | PASS |
| Objective eligible total | 81 | ≥140 | FAIL |
| TRAIN strict Sand | 30 | ≥60 | FAIL |
| TRAIN mild / moderate | 15 / 15 | ≥44 / ≥12 | FAIL / PASS |
| TRAIN ordinary / delayed Support | 18 / 9 | ≥20 / ≥10 | FAIL / FAIL |
| VALIDATION strict Sand | 12 | ≥30 | FAIL |
| VALIDATION mild / moderate | 6 / 6 | ≥22 / ≥6 | FAIL / PASS |
| VALIDATION ordinary / delayed Support | 8 / 4 | ≥10 / ≥5 | FAIL / FAIL |
| Six TRAIN strict-Sand cells | 3,3,8,4,6,6 | each ≥9 | FAIL |
| Six VALIDATION strict-Sand cells | 0,2,3,1,3,3 | each ≥4 | FAIL |
| Principal phases and topologies | both in both roles | both | PASS |
| Unique physical-signature fraction | 0.987654 | ≥0.80 | PASS |
| Integrity/model-output/backfill/replacement checks | all zero | zero | PASS |

Twenty-one yield gates failed. The dominant model-blind failure was timing/stability: 42 traces lacked the required 1,000 ms post-target observation and 30 fell before target contact. Therefore training was prohibited.

## 10. Dataset freeze

The failed attempt was retained as immutable, Gitignored physical evidence rather than deleted or relabeled. Verification passed all 12 hash-chain checks across manifest, NPZ aggregate, config provenance, scenario/split identities, implementation identities, physical audit, and semantic freeze. TRAIN eligible count is 57 and VALIDATION eligible count is 24, but neither role is authorized for this cycle because the corpus-level gate failed.

## 11. Training policy

The frozen policy authorized exactly Unified TRAIN + V2_TRAIN + eligible fresh FACTOR_TRAIN, one GRU family, seeds 20260828/20260829/20260830, and no validation early stopping. The enforcement dry-run was invoked only to verify the stop barrier and exited before loading training data with `physical-generation gates failed; training is prohibited`. No extraction audit or optimizer operation was performed.

| Seed | Round | Positives | Negatives | New Sand exposure | Support exposure | HNM added | Epoch | Steps | Checkpoint SHA |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 20260828 | 0–3 | NOT RUN | NOT RUN | 0 | 0 | 0 | 0 | 0 | none |
| 20260829 | 0–3 | NOT RUN | NOT RUN | 0 | 0 | 0 | 0 | 0 | none |
| 20260830 | 0–3 | NOT RUN | NOT RUN | 0 | 0 | 0 | 0 | 0 | none |

## 12. Normalizer preservation

The frozen V2 normalizer identity was verified as `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a`. It was not refit and was not applied to the failed fresh corpus. Normalizer fits: 0.

## 13. Training exposure

There was no training exposure: zero windows, batches, epochs, optimizer steps, or validation exposure. Old Sand Confirmation training use and fresh VALIDATION training use are both 0.

## 14. HNM

The planned canonical policy was exactly three TRAIN-only HNM rounds with stride 1 ms, K=12/run, and 30 ms spacing. Because the pre-training physical gate failed, actual HNM rounds and mined endpoints are 0; there is no HNM provenance artifact or HNM hash for a nonexistent candidate.

## 15. Candidate freeze

No candidate was trained or frozen. Candidate ID, checkpoint SHA, and candidate-freeze SHA are therefore `N/A`. The unchanged intended interface was Pelvis IMU6 → causal 80D → `[20,80]` → one-layer unidirectional GRU hidden 32 (11,010 parameters), threshold 0.99, persistence 5 ms. The historical reference V2 and its three checkpoints remain unchanged.

## 16. Fresh factor validation

`FACTOR_VALIDATION` model access was never authorized or opened. Reference V2 inference: 0. New-candidate inference: 0.

| Metric | Reference V2 | New Candidate | Delta | Intervention target |
|---|---|---|---|---|
| Strict Sand specificity | NOT EVALUATED | no candidate | N/A | material improvement |
| Mild Sand specificity | NOT EVALUATED | no candidate | N/A | improvement |
| Moderate Sand specificity | NOT EVALUATED | no candidate | N/A | improvement |
| False Reflex count | NOT EVALUATED | no candidate | N/A | reduction ≥2 |
| Adverse-margin rate | NOT EVALUATED | no candidate | N/A | reduction ≥0.10 |
| Median / p95 max p | NOT EVALUATED | no candidate | N/A | decrease |
| Support / ordinary / delayed / right recall | NOT EVALUATED | no candidate | N/A | preserve frozen gates |

## 17. Reference V2 vs new candidate

No comparison exists because generating model evidence after a failed physical gate would violate the frozen protocol. This absence is a required fail-closed outcome, not missing analysis.

## 18. Sand specificity

Unknown for this intervention. No probability, Reflex, specificity, or false-positive statistic was computed on the fresh corpus.

## 19. Margin behavior

Unknown. The 0.95 adverse-margin and 0.99/5 ms runtime rules remained frozen, but neither reference nor candidate probabilities were computed.

## 20. Factor localization

| Factor | Reference adverse | New adverse | Reference FP | New FP | Improvement? |
|---|---|---|---|---|---|
| transition-left | NOT OPENED | no candidate | NOT OPENED | no candidate | N/A |
| transition-right | NOT OPENED | no candidate | NOT OPENED | no candidate | N/A |
| right-single precontact | NOT OPENED | no candidate | NOT OPENED | no candidate | N/A |
| left-single precontact | NOT OPENED | no candidate | NOT OPENED | no candidate | N/A |
| Concrete | NOT OPENED | no candidate | NOT OPENED | no candidate | N/A |
| Marble | NOT OPENED | no candidate | NOT OPENED | no candidate | N/A |
| 0.20 m/s | NOT OPENED | no candidate | NOT OPENED | no candidate | N/A |
| 0.25 m/s | NOT OPENED | no candidate | NOT OPENED | no candidate | N/A |
| 0.30 m/s | NOT OPENED | no candidate | NOT OPENED | no candidate | N/A |
| mild | NOT OPENED | no candidate | NOT OPENED | no candidate | N/A |
| moderate | NOT OPENED | no candidate | NOT OPENED | no candidate | N/A |

Only physical localization is valid: phase/topology diversity passed, while objective yield failed every source-speed Sand minimum. This does not localize a model boundary.

## 21. Support preservation

Fresh physical Support-control viability failed its gates: ordinary Support was 26/36 and delayed Support was 13/18. No model recall was measured. The historical, independently established status `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED` is preserved and not rewritten by this invalid development attempt.

## 22. Historical V2_VALIDATION regression check

The planned one-time candidate replay was not performed because no candidate existed.

| Metric | Reference V2 | New candidate | Delta | Acceptable? |
|---|---|---|---|---|
| Hazard | saved historical result | no candidate | N/A | NOT TESTED |
| Slip | saved historical result | no candidate | N/A | NOT TESTED |
| Support | saved historical result | no candidate | N/A | NOT TESTED |
| Specificity | saved historical result | no candidate | N/A | NOT TESTED |
| Premature | saved historical result | no candidate | N/A | NOT TESTED |
| Delayed Support | saved historical result | no candidate | N/A | NOT TESTED |
| Right Support | saved historical result | no candidate | N/A | NOT TESTED |
| Speed-Sand benign | saved historical result | no candidate | N/A | NOT TESTED |

V2_VALIDATION payload access in this cycle was 0; the table was not used for tuning.

## 23. Slip preservation

No model Slip metric was evaluated. Five fresh designed-Sand runs realized Slip and two realized dual Hazard; all seven were retained as provenance and excluded from targeted membership. The historical model and its Slip limitations remain unchanged.

## 24. Intervention verdict

`FACTOR_CONDITIONED_DATA_INTERVENTION_INVALID`

This is a physical-domain execution verdict. It does not establish that the factor-conditioned data hypothesis is ineffective; the hypothesis was not tested by training or validation because the prerequisite corpus was invalid.

## 25. Scientific limitations

The choice to skip an optional model-blind pilot left the wholly fresh wide-patch timing insufficiently calibrated. The resulting corpus spans both principal phase/topology manifolds and is highly diverse, but only 50% (81/162) met the intended actual-physics roles. No conclusions about model specificity, margins, support recall, representation, or generalization can be drawn from this attempt.

## 26. Required new final evidence

Historical final Model V2 remains `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`. New-candidate final generalization is `NOT YET ESTABLISHED` because no candidate exists. Any future successful modified candidate requires a NEW independent external-validation dataset, readiness decision, and independent final HOLDOUT. The consumed historical HOLDOUT can never serve that role.

## 27. Recommended next milestone

Exactly one next scientific milestone is recommended: `DATA_INTERVENTION_FAILURE_AUDIT`.

That audit must remain read-only, use this fresh corpus's saved physical provenance plus existing representation evidence, and determine a separately reviewed physical-domain correction. It must not train another candidate, open either validation payload for selection, or begin final evidence generation automatically.

## Provenance hashes

| Artifact/contract | SHA-256 |
|---|---|
| Intervention/training config | `540034673d1703adce000182b73e2dc4c4bf8856e534e7489ea66bef6522246e` |
| Fresh parameter domain | `7ba26156fcf5384d056abaa346bb7bc2dcce3e84ece950c535b1fda70ca2483d` |
| Scenario-matrix contract | `b75dc50a3090615191b6d2846c19556f58d7c68ff1f3f1746434d69c500957b6` |
| Expanded scenario matrix | `10c6bd9f7ebd212b35b0e5bd335f6558da7b1178e70a9ed12d208d7054969d26` |
| Split-plan contract | `c28081592bbdf11c957156bfbd5e16e618694eb245042a12212b60f38a663f5d` |
| Physical-label contract | `ac25dbd7c701e47a559ff9e5f909029e15b1edb37188121cb6ad8ad4cc302d17` |
| Generation-gates contract | `4d01d404eab9921d612411ffd07b1a403b7fba757fbb3cce106a461ce49dad63` |
| TRAIN split | `38eac9969a46c6986ca7f99aa20252d456bb43ac61b391db77b5fe401cf1785e` |
| VALIDATION split | `b7ac7aa40ebff6b3e8de1038426b37c0b4580310cbe1e3d30861b8ba97bc49bd` |
| Scenario signatures | `7a70ef516fd7c21f8ce8a5322f86495b0cdba49b043f8d845160a76dac86c412` |
| Manifest | `d4ec98b5c2bec6c9009b5da958195935c46dd57489c8f0f15e08258ae84d6998` |
| NPZ aggregate | `ca595130cbc2037a16c792f0d4c1e5fc255498acd3246ad84f8d3d9cc1a62f9b` |
| Physical signatures | `369306ed7de79b0110a72d7da48fdf0f20ff1af4385723cc10973b0e897897b0` |
| Physical audit | `b334f615323a2d770f5b8c35b251368f10f1f00f481868933971cb5f00d67dee` |
| Dataset freeze, semantic | `4906682f9366bad572baeb529db81ca1d5b1b2878f1cd2e4782999e2588cd549` |
| Dataset freeze, file | `36f5c93c5dba3793101d1401ab88864fe8833d9904f7593a3a6593c7a978327f` |
| Frozen V2 normalizer | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` |
| HNM provenance | N/A—HNM not run |
| Checkpoints | N/A—0 writes |
| Candidate freeze | N/A—no candidate |
| Factor-validation result | N/A—split not opened |
| Historical regression result | N/A—V2_VALIDATION not opened |
| Physical failure interpretation | `bb53a193f5120e684b9552903aca2a7fe08abea03cf031e56638be6a0e4aba88` |
| Final intervention decision | `2a3f390f9b2160b3ebdb91d80142d4a2032ece91bb3a93dfa74b4efcd1b50cc0` |
| Cycle result | `280cd792272ce8d5972c0f98f7cf8b0c91d2648d87193aac819a7456ced80d0c` |

## Counters

| Counter | Actual |
|---|---:|
| New simulations | 162 |
| Calibration pilot simulations | 0 |
| Main corpus simulations | 162 |
| Optimizer steps | 0 |
| Checkpoint writes | 0 |
| Normalizer fits | 0 |
| HNM rounds | 0 |
| Threshold searches | 0 |
| Persistence searches | 0 |
| Architecture searches | 0 |
| Seed searches | 0 |
| Sensor-fusion experiments | 0 |
| Old HOLDOUT reads / inference | 0 / 0 |
| Old Sand Confirmation training use | 0 |
| Fresh VALIDATION training use | 0 |

Deployment engineering may continue in parallel against the already exported, explicitly non-final reference interface. This milestone did not modify `/d/shin/Infineon_FastReflex_E84`, did not change the research runtime interface, and did not authorize production, quantization, firmware, Vela, or HIL work here.
