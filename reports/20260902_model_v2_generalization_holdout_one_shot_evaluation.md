# Model V2 Generalization HOLDOUT One-Shot Evaluation

## 1. Purpose

This milestone performed the first and only scientific opening of the frozen 36-run `GENERALIZATION_HOLDOUT`. Frozen Model V1, the exact final Model V2 candidate, and frozen Terrain V1 advisory inference were evaluated from each payload in the same deterministic pass. No training, tuning, model selection, relabeling, exclusion, visualization, or second opening occurred.

The primary verdict is `GENERALIZATION_HOLDOUT_PRIMARY_GATES_FAIL`. The final verdict is `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`.

## 2. Starting state

The starting `HEAD` and `origin/main` were both `6b8f610aa36316f7b9a66864ffe0837ad16d0d83` (`Freeze Model V2 final candidate readiness`), and the tracked worktree was clean. The prior readiness status was `HOLDOUT_READY`; the candidate role was `FINAL_GENERALIZATION_CANDIDATE_FROZEN`.

## 3. Pre-open integrity

Metadata-only preflight did not deserialize a HOLDOUT payload. It verified the exact candidate and readiness records, frozen V1 and Terrain artifacts, all contract/source hashes, and protected corpora:

| Corpus | Verified files | Mismatch |
|---|---:|---:|
| Unified | 256/256 | 0 |
| Model V2 | 412/412 | 0 |
| Generalization | 72/72 | 0 |
| Ice semantics | 48/48 | 0 |

All 72 Generalization physical signatures were unique, with zero `GENERALIZATION_VALIDATION`/`GENERALIZATION_HOLDOUT` overlap. The Generalization HOLDOUT count was 36, the stored manifest and payload hashes matched, and the guard was absent (`0`). The frozen execution config SHA-256 was `ec53c761f426aaeba5528916c60a6c3f69550007987cdf5f3754304cd4bbef0a`; evaluator SHA-256 was `5450360221322035a1cc44f3051fc59731578e7c3392316ac2ac6e22a04b0a5d`.

Frozen V1 resolved exactly to freeze SHA-256 `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`, normalizer `610a91c24a594a3cd1701b07549f4f3c4b7f4cb04eb354f9d41378214a6800a9`, and checkpoint hashes `e6bada49beb2b68c2c33bb9a59a1b3a8dd0f556f174cafce12f43efd2d22d588`, `b04877dc08290a34077ae2deb753085d840640ee8da2bd920d010eb2ce8c2506`, and `b6c782bdfb3789ae7af785ec2b02260a8ae54179d7c531f87e27e0e35301a753`. Frozen Terrain V1 resolved to normalizer `2c551181fbd23080c0e3008be8484bd0ee4139e594041ba6e46ab5b36ffb05de` and checkpoint hashes `21b0d122b4200a96390b700f741d6b35a4e72226e61d204a420d23e086e1f628`, `de6a55d35531dfa96d73e86bb8b5596ead5e41809aba539dd83e32b473cd0d66`, and `465803f40fff371b9de2ca0ecaf7d9d41717d2be6b5cd33fe031d6d9ba237b31`; it remained read-only and advisory-only.

Before opening, 44 targeted safe tests passed. The complete verified-safe suite passed with 106 tests and one skip; compileall, critical Ruff `E9,F63,F7,F82`, full Ruff on changed code, and `git diff --check` also passed.

## 4. Frozen candidate

| Item | Frozen value |
|---|---|
| Candidate | `model_v2_anchor_refined_gru20_20260902` |
| Final candidate SHA-256 | `52644d3efb0e756002bd43a7f94aea0a16a65eb2ff014c72548f2b3b138b7bc2` |
| Development freeze SHA-256 | `95dab53275aa77ded36b479c3633ab879a1331cb2c96bd0eb014de2be99bd85f` |
| Promotion SHA-256 | `1e4931e35e873cd721b412c6a45f66340f7ee9eebc1900d9c4aa3dc9ab3d092f` |
| Internal evaluation SHA-256 | `cad3902137b622c3a3d15ecb3d6c3bb31ee9751f3605fb8d43daa6ac81695c07` |
| Generalization VALIDATION SHA-256 | `cc559c592f64ed8afd27b039d57ace35d8bbb3efad1cac64e3d10788e0ffa556` |
| Readiness review SHA-256 | `0167d72942ee402b7bdcb83f5bcd3e69c62f4db8044c1bf62bfd8607487eb7c6` |
| Normalizer SHA-256 | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` |
| Checkpoint SHA-256 | `7094a2dca40e8d3c84554619652d69c17920c8e82765460ff8621c13ef494cb9` |
| Checkpoint SHA-256 | `3ad298eea4c35eca896afd31f860fd6b44ce35d7d9978e2546bb40b693e62c39` |
| Checkpoint SHA-256 | `fe96dfeb8461871044de0f8672190680ce46164a1c28efbe6738d22f9d439bbd` |
| Architecture SHA-256 | `ae4753699399aa7e4a93639935e085156ef28ed100a77058d7308eb1549aa897` |
| Feature SHA-256 | `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb` |
| Runtime | threshold 0.99; persistence 5 ms; stride 1 ms |

The architecture remained Pelvis IMU6 to causal 80D `[20,80]`, one-layer unidirectional GRU hidden 32, linear 32-to-2 head, three-seed mean, 11,010 parameters. Candidate mutation after opening was false.

## 5. Frozen contracts

The primary contract SHA-256 was `feabfc4519e8ec28e59710810b6e587b7a8be1a128ecf57a028d32710c1b246e`. It retained established Slip at 50 mm/3 ms, established Support at 10 mm/20 ms, Slip window `[-30,+40]` ms, Support window I1 through Support +50 ms, and the first-response rule.

The Ice secondary contract SHA-256 was `085d6f73156a5618767284faa2ccdcd29d3645694f56155431159d533b77130a`. It retained loaded exact-Ice drift `[0.030,0.050)` m, 1,000 ms follow-up, and the five frozen outcome classes. It remained descriptive and did not rewrite a primary result. Verdict hierarchy SHA-256 was `e86fb11f457734c41cd7b9c66a827a22b587f7a1f95aa91130931f7586c8cba5`.

## 6. Guard transition

The durable guard transitioned `0 -> 1` at `2026-09-02T15:54:22.042348+09:00`, before the first payload deserialization. It records source commit `6b8f610aa36316f7b9a66864ffe0837ad16d0d83`, the frozen execution/evaluator/candidate/contract hashes, one claimed scientific opening, and permanent refusal of a second scientific opening. Guard record SHA-256 is `0913da8a583cab7834590d669ba9c8d470485b8129fe81f538aba6c461613154`.

## 7. One-shot execution

All 36 runs were processed. Each payload was deserialized exactly once, then the same in-memory arrays were used for V1 Hazard, final V2 Hazard, Terrain V1, primary physical scoring, and Ice-secondary scoring. Thus total payload deserializations were 36 and deserializations per run were one. `first_payload_read_after_guard_transition` is true; same-pass inference is true; partial/adaptive access and visualization are false.

All scientific mutation/search counters were zero: optimizer steps, checkpoint writes, normalizer fits, HNM rounds, threshold searches, persistence searches, architecture searches, seed searches, and new simulation runs. Scientific HOLDOUT opens equal one.

## 8. Dataset composition

Actual physical outcome, rather than scenario intent, determined every denominator.

| Family | N | Actual physical outcome |
|---|---:|---|
| `ONE_CONTACT_DELAYED_ICE_SLIP` | 6 | 6 Slip |
| `ICE_BENIGN_CONTROL` | 4 | 2 Slip, 2 no-hazard |
| `DELAYED_SAND_SUPPORT_ONSET` | 4 | 4 Support |
| `RIGHT_SAND_SUPPORT` | 4 | 4 Support |
| `SPEED_STRATIFIED_HAZARD / ICE_SLIP` | 6 | 6 Slip |
| `SPEED_STRATIFIED_HAZARD / SAND_SUPPORT` | 6 | 6 Support |
| `SPEED_STRATIFIED_HAZARD / SAND_BENIGN` | 6 | 6 no-hazard |
| **Total** | **36** | **28 Hazard: 14 Slip + 14 Support; 8 no-hazard** |

Concrete and Marble each contributed 18 runs. The frozen 0.20/0.25/0.30 m/s speed grid contributed two runs per speed and subtype for speed-stratified Ice Slip, Sand Support, and Sand benign.

## 9. V1 overall result

| Metric | Frozen V1 HOLDOUT result |
|---|---:|
| Overall Hazard recall | 14/28 = 50.00% |
| Slip recall | 8/14 = 57.14% |
| Support recall | 6/14 = 42.86% |
| Primary no-hazard specificity | 5/8 = 62.50% |
| Physical Ice-benign specificity | 2/2 = 100% |
| Premature Hazard-run rate | 9/28 = 32.14% |
| Slip latency median / p95 | -21 / -17 ms |
| I1-to-Reflex median / p95 | 1,217.5 / 1,231 ms |
| Reflex-to-Support lead median / p95 | 20 / 559 ms |
| Established-Support-relative median / p95 | -20 / -17.25 ms |

## 10. Final V2 overall result

| Metric | Final V2 HOLDOUT result |
|---|---:|
| Overall Hazard recall | 25/28 = 89.29% |
| Slip recall | 11/14 = 78.57% |
| Support recall | 14/14 = 100% |
| Primary no-hazard specificity | 5/8 = 62.50% |
| Physical Ice-benign specificity | 2/2 = 100% |
| Premature Hazard-run rate | 2/28 = 7.14% |
| Slip latency median / p95 | -13 / +7.5 ms |
| I1-to-Reflex median / p95 | 624 / 1,773 ms |
| Reflex-to-Support lead median / p95 | 23 / 52 ms |
| Established-Support-relative median / p95 | -23 / -17 ms |

## 11. Primary gates

| Metric | Result | Frozen gate | Status |
|---|---:|---:|---|
| Overall Hazard recall | 25/28 = 89.29% | >=90% | **FAIL** |
| Slip recall | 11/14 = 78.57% | >=95% | **FAIL** |
| Support recall | 14/14 = 100% | >=85% | PASS |
| Primary specificity | 5/8 = 62.50% | >=95% | **FAIL** |
| Ice-benign specificity | 2/2 = 100% | >=95% | PASS |
| Premature rate | 2/28 = 7.14% | <=10% | PASS |
| Slip p95 latency | +7.5 ms | <=+40 ms | PASS |
| Support p95 established latency | -17 ms | <=+50 ms | PASS |

There is no composite or averaged gate. Secondary semantics do not rescue the three failed primary gates.

## 12. V1 vs V2

| Metric | V1 | Final V2 | Delta | Gate | V2 status |
|---|---:|---:|---:|---:|---|
| Hazard | 14/28 (50.00%) | 25/28 (89.29%) | +39.29 pp | >=90% | **FAIL** |
| Slip | 8/14 (57.14%) | 11/14 (78.57%) | +21.43 pp | >=95% | **FAIL** |
| Support | 6/14 (42.86%) | 14/14 (100%) | +57.14 pp | >=85% | PASS |
| Primary specificity | 5/8 (62.50%) | 5/8 (62.50%) | 0.00 pp | >=95% | **FAIL** |
| Ice-benign specificity | 2/2 (100%) | 2/2 (100%) | 0.00 pp | >=95% | PASS |
| Premature | 9/28 (32.14%) | 2/28 (7.14%) | -25.00 pp | <=10% | PASS |
| Slip p95 | -17 ms | +7.5 ms | +24.5 ms | <=+40 ms | PASS |
| Support p95 | -17.25 ms | -17 ms | +0.25 ms | <=+50 ms | PASS |

Family-level primary evidence was:

| Family | N | V1 | V2 | V2 primary failures | Interpretation |
|---|---:|---|---|---:|---|
| `ONE_CONTACT_DELAYED_ICE_SLIP` | 6 | 3/6 recall; 3 premature | 5/6 recall; 1 premature | 1 | Supported Ice timing conflict; no genuine miss |
| `ICE_BENIGN_CONTROL` | 4 | Slip 1/2; no-hazard 2/2 | Slip 1/2; no-hazard 2/2 | 1 | Two controls became actual Slip; one supported timing conflict |
| `DELAYED_SAND_SUPPORT_ONSET` | 4 | 0/4; 4 premature | 4/4; 0 premature | 0 | Delayed-Support correction transferred |
| `RIGHT_SAND_SUPPORT` | 4 | 0/4 | 4/4 | 0 | Right-only Support correction transferred |
| `SPEED_STRATIFIED_HAZARD / ICE_SLIP` | 6 | 4/6; 1 premature | 5/6; 0 premature | 1 | One 0.30 m/s late detection |
| `SPEED_STRATIFIED_HAZARD / SAND_SUPPORT` | 6 | 6/6 | 6/6 | 0 | Stable across the frozen speed grid |
| `SPEED_STRATIFIED_HAZARD / SAND_BENIGN` | 6 | 3/6 specificity | 3/6 specificity | 3 | Material Sand false-alert weakness remains |

## 13. Delayed Ice

All six `ONE_CONTACT_DELAYED_ICE_SLIP` rows are reported below. Times are 1 kHz sample indices.

| Run | Source | Speed | Side | Target | Benign contacts | Precursor | V1 Reflex | V2 Reflex | Slip | V1 | V2 | V2 secondary |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `ghr_ocd_h_c020` | Concrete | 0.20 | bilateral | 1504 | 16 | 2466 | 2470 | 2478 | 2632 | premature | premature | supported future-Slip precursor |
| `ghr_ocd_h_c025` | Concrete | 0.25 | bilateral | 1220 | 1 | 1889 | 1886 | 1891 | 1910 | correct | correct | supported future-Slip precursor |
| `ghr_ocd_h_c030` | Concrete | 0.30 | bilateral | 1227 | 18 | 1894 | 1897 | 2057 | 2057 | premature | correct | outside precursor response |
| `ghr_ocd_h_m020` | Marble | 0.20 | bilateral | 1507 | 2 | 2465 | 2470 | 2474 | 2487 | correct | correct | supported future-Slip precursor |
| `ghr_ocd_h_m025` | Marble | 0.25 | bilateral | 1220 | 1 | 1887 | 1883 | 1887 | 1908 | correct | correct | supported future-Slip precursor |
| `ghr_ocd_h_m030` | Marble | 0.30 | bilateral | 1227 | 18 | 1889 | 1888 | 2051 | 2054 | premature | correct | supported future-Slip precursor |

V1 was 3/6 and V2 was 5/6. V2 had one premature `SUPPORTED_ICE_PRECURSOR_TIMING_CONFLICT` and no genuine detection failure in this family. By the saved benign-contact count, the exactly-one subgroup was 2/2 for V2 and the multi-contact subgroup was 3/4.

## 14. Ice benign

Only actual physical no-hazard controls enter Ice-benign specificity. The other two design controls became actual Slip and enter the Slip denominator.

| Run | Source | Max target drift | Precursor onset | V1 Reflex | V2 Reflex | Primary result | Secondary alert |
|---|---|---:|---:|---:|---:|---|---|
| `ghr_ibc_h_c030` | Concrete | 0.111417 m | 1915 | none | none | true negative | no alert |
| `ghr_ibc_h_m030` | Marble | 0.137078 m | 1899 | none | none | true negative | no alert |

The established 50 mm/3 ms physical oracle, not instantaneous maximum drift, remained authoritative. Both no-hazard controls contained a precursor occurrence and neither model fired, so V1 and V2 physical Ice-benign specificity were both 2/2. Across the complete saved secondary evidence there were two benign-release episodes and nine censored episodes; both models produced zero benign-release false alerts and zero in-candidate censored alerts. Per-run episode outcome lists were not serialized, so no post-open waveform reread was performed to subdivide those aggregates.

## 15. Delayed Sand Support

| Run | Source | Target | I1 | Support | V1 Reflex | V2 Reflex | V1 pre-I1? | V2 pre-I1? | I1-to-V2 | V2 lead |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| `ghr_dss_h_c300` | Concrete | 1220 | 3011 | 3067 | 2461 | 3015 | yes | no | 4 ms | 52 ms |
| `ghr_dss_h_c350` | Concrete | 1220 | 3011 | 3067 | 2461 | 3015 | yes | no | 4 ms | 52 ms |
| `ghr_dss_h_m300` | Marble | 1220 | 3012 | 3068 | 2459 | 3016 | yes | no | 4 ms | 52 ms |
| `ghr_dss_h_m350` | Marble | 1220 | 3012 | 3068 | 2459 | 3016 | yes | no | 4 ms | 52 ms |

V1 was 0/4 with four premature responses. Final V2 was 4/4, Concrete 2/2 and Marble 2/2, with no pre-I1 false response. The delayed-Support solution therefore generalized to this fresh HOLDOUT family.

## 16. Right Sand Support

| Run | Source | Actual side | I1 | Support | V1 | V2 Reflex | V2 lead | Terrain first valid | Order |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| `ghr_rss_h_c300` | Concrete | right-only | 1526 | 2173 | miss | 2150 | 23 ms | 3008 | Reflex before Terrain |
| `ghr_rss_h_c350` | Concrete | right-only | 1526 | 2173 | miss | 2150 | 23 ms | 3010 | Reflex before Terrain |
| `ghr_rss_h_m300` | Marble | right-only | 1528 | 2176 | miss | 2152 | 24 ms | 2992 | Reflex before Terrain |
| `ghr_rss_h_m350` | Marble | right-only | 1528 | 2176 | miss | 2152 | 24 ms | 2991 | Reflex before Terrain |

All four runs were actually right-only. V1 was 0/4 and V2 was 4/4. Terrain was available in 4/4, but only after the Hazard response; it remained advisory and did not alter Hazard scoring.

## 17. Speed robustness

These are descriptive cells, not post-hoc gates.

| Speed | Subtype | N | V1 | Final V2 |
|---:|---|---:|---:|---:|
| 0.20 | Ice Slip recall | 2 | 2/2 | 2/2 |
| 0.20 | Sand Support recall | 2 | 2/2 | 2/2 |
| 0.20 | Sand benign specificity | 2 | 1/2 | 1/2 |
| 0.25 | Ice Slip recall | 2 | 2/2 | 2/2 |
| 0.25 | Sand Support recall | 2 | 2/2 | 2/2 |
| 0.25 | Sand benign specificity | 2 | 1/2 | 2/2 |
| 0.30 | Ice Slip recall | 2 | 0/2 | 1/2 |
| 0.30 | Sand Support recall | 2 | 2/2 | 2/2 |
| 0.30 | Sand benign specificity | 2 | 1/2 | 0/2 |

The 0.30 m/s cells expose both the genuine late Slip response and two Sand-benign false alerts.

## 18. Side robustness

| Physical event | Actual side | N | V1 | Final V2 | Interpretation |
|---|---|---:|---:|---:|---|
| Slip | left-only | 0 | n/a | n/a | no evidence |
| Slip | right-only | 1 | 0/1 | 0/1 | supported Ice timing conflict; insufficient robustness denominator |
| Slip | bilateral | 13 | 8/13 | 11/13 | one timing conflict and one late response remain |
| Support | left-only | 10 | 6/10 | 10/10 | V2 correction transferred |
| Support | right-only | 4 | 0/4 | 4/4 | V2 correction transferred |
| Support | bilateral | 0 | n/a | n/a | no evidence |

`RIGHT_ONLY_SLIP_HOLDOUT_DENOMINATOR = 1`; right-only Slip robustness is not established.

## 19. Source robustness

| Source | Metric | N | V1 | Final V2 |
|---|---|---:|---:|---:|
| Concrete | Hazard recall | 14 | 6/14 | 11/14 |
| Concrete | Slip recall | 7 | 3/7 | 4/7 |
| Concrete | Support recall | 7 | 3/7 | 7/7 |
| Concrete | Specificity | 4 | 3/4 | 2/4 |
| Marble | Hazard recall | 14 | 8/14 | 14/14 |
| Marble | Slip recall | 7 | 5/7 | 7/7 |
| Marble | Support recall | 7 | 3/7 | 7/7 |
| Marble | Specificity | 4 | 2/4 | 3/4 |

No statistical-significance claim is made. Concrete contains all three V2 strict Hazard failures and two of the three no-hazard false alerts; Marble contains the remaining false alert.

## 20. Primary failures

Every final V2 primary failure is listed exactly once.

| Run | Family | Physical event | V2 Reflex | Primary class | Frozen secondary class | Genuine failure? |
|---|---|---|---:|---|---|---|
| `ghr_ibc_h_c020` | `ICE_BENIGN_CONTROL` | right-only Slip at 2628 | 2478 | `SUPPORTED_ICE_PRECURSOR_TIMING_CONFLICT` | `SUPPORTED_FUTURE_SLIP_PRECURSOR` | no |
| `ghr_ocd_h_c020` | `ONE_CONTACT_DELAYED_ICE_SLIP` | bilateral Slip at 2632 | 2478 | `SUPPORTED_ICE_PRECURSOR_TIMING_CONFLICT` | `SUPPORTED_FUTURE_SLIP_PRECURSOR` | no |
| `ghr_ssh_is_h_c030` | `SPEED_STRATIFIED_HAZARD / ICE_SLIP` | bilateral Slip at 2043 | 2085 | `LATE_DETECTION` | `OUTSIDE_PRECURSOR_RESPONSE` | **yes: +42 ms, 2 ms beyond window** |
| `ghr_ssh_sb_h_c020` | `SPEED_STRATIFIED_HAZARD / SAND_BENIGN` | no hazard | 5154 | `BENIGN_FALSE_ALERT` | `BENIGN_FALSE_REFLEX` | no |
| `ghr_ssh_sb_h_c030` | `SPEED_STRATIFIED_HAZARD / SAND_BENIGN` | no hazard | 4033 | `BENIGN_FALSE_ALERT` | `BENIGN_FALSE_REFLEX` | no |
| `ghr_ssh_sb_h_m030` | `SPEED_STRATIFIED_HAZARD / SAND_BENIGN` | no hazard | 4298 | `BENIGN_FALSE_ALERT` | `BENIGN_FALSE_REFLEX` | no |

The two early Ice responses remain primary failures even though the secondary contract supports their physical precursor interpretation. The 0.30 m/s Concrete response is a genuine late detection, not an Ice-timing rescue case.

## 21. Ice precursor secondary

The alert cells show `inside candidate / before established Slip`.

| Outcome | Episodes | V1 alerts | Final V2 alerts |
|---|---:|---:|---:|
| Same-episode Slip | 119 | 11 / 11 | 15 / 14 |
| Next-episode Slip | 14 | 0 / 0 | 0 / 1 |
| Later Slip | 69 | 9 / 12 | 3 / 12 |
| Benign release | 2 | 0 / 0 | 0 / 0 |
| Censored | 9 | 0 / 0 | 0 / 0 |
| **Future-Slip total** | **202** | **20 / 23** | **18 / 27** |

V1 precursor-to-Reflex median/p95 was 0.5/5.55 ms and Reflex-to-Slip median/p95 was 51/162.15 ms. Final V2 precursor-to-Reflex median/p95 was 9/98 ms and Reflex-to-Slip median/p95 was 19/150.6 ms. Benign-release false alerts and in-candidate censored alerts were zero for both models. `primary_scores_rewritten` is false; the official primary metrics above are unchanged.

## 22. Physical-response interpretation

There were six primary failures: two supported Ice precursor timing conflicts, one genuine late detection, three benign false alerts, zero pre-I1 Support false alerts, zero `CENSORED_OR_AMBIGUOUS`, and zero `OTHER`. Strict primary Hazard recall remains 25/28. The separately counted physically supported Hazard-response view is 27/28 and is **diagnostic only**.

The saved top-level field `genuine_detection_miss_count=0` counts only rows whose literal primary category is `GENUINE_DETECTION_MISS`. The run-level evidence separately and correctly records `ghr_ssh_is_h_c030` as category `LATE_DETECTION` with `genuine_detection_failure=true`. Scientific interpretation therefore reports zero literal hard-miss categories and one genuine late-detection failure; this distinction does not alter a score, gate, or verdict.

## 23. Terrain advisory

Terrain V1 evaluated 349/349 clean target events and classified 271 correctly: 77.65% accuracy where evaluable. Target Terrain became available in 35/36 runs and was unavailable in one no-hazard run. Target-contact-to-first-valid timing was 691 ms median and 1,488.4 ms p95.

Among the 28 Hazard runs, Terrain preceded V2 Reflex in 16 and V2 Reflex preceded Terrain in 12; none was unavailable for this order comparison. The selected deployment scheme remained left-only FSR4 `[50,4]`, 50 ms observation, frozen three-seed MLP. `advisory_only=true` and `hazard_gate=false`; Terrain neither delayed nor changed Hazard.

## 24. Evidence progression

The evidence sets and denominators remain separate.

| Evidence set | Model | Hazard | Slip | Support | Specificity | Premature |
|---|---|---:|---:|---:|---:|---:|
| V2_VALIDATION | final V2 | 59/64 (92.19%) | 30/35 (85.71%) | 30/30 (100%) | 26/26 (100%) | 5/64 (7.81%) |
| Generalization VALIDATION | V1 | 13/26 (50.00%) | 7/12 (58.33%) | 6/14 (42.86%) | 5/10 (50.00%) | 7/26 (26.92%) |
| Generalization VALIDATION | final V2 | 25/26 (96.15%) | 11/12 (91.67%) | 14/14 (100%) | 10/10 (100%) | 1/26 (3.85%) |
| Generalization HOLDOUT | V1 | 14/28 (50.00%) | 8/14 (57.14%) | 6/14 (42.86%) | 5/8 (62.50%) | 9/28 (32.14%) |
| Generalization HOLDOUT | final V2 | 25/28 (89.29%) | 11/14 (78.57%) | 14/14 (100%) | 5/8 (62.50%) | 2/28 (7.14%) |

## 25. Data-coverage hypothesis

`DATA_COVERAGE_HYPOTHESIS_NOT_SUPPORTED`

With the same GRU20 architecture, the expanded/corrected data and extraction substantially improved V1 Hazard, Slip, Support, and premature behavior and completely corrected delayed/side Support on HOLDOUT. It did not provide the predeclared final generalization support: overall Hazard narrowly missed, Slip materially missed, specificity did not improve over V1, three Sand-benign false alerts remained, and one 0.30 m/s Slip response was genuinely late. Coverage/extraction alone is therefore not supported as sufficient by the complete evidence.

## 26. Architecture implication

The evidence verdict is `ARCHITECTURE_EVIDENCE_REQUIRES_REVIEW`. GRU20 remains the exact frozen object to diagnose, but the HOLDOUT does not justify treating it as a simulation-generalization-supported release candidate. This single consumed evaluation does not isolate longer history, LSTM, a larger GRU, feature redesign, or sensor expansion as the remedy. None of those experiments was started. The next milestone must interpret the saved failures before any architecture choice; any changed candidate requires new independent final evidence.

## 27. Sensor implication

`10_CHANNEL_ARCHITECTURE_STILL_PLAUSIBLE`

The provisional Pelvis IMU6 Hazard plus left FSR4 Terrain architecture remains plausible: V2 corrected all Support cases, including 4/4 right-only Support, without Terrain gating. The late Slip and Sand false-alert failures prevent a stronger claim and do not independently prove that more sensors are required. `FINAL_SENSOR_ARCHITECTURE_FROZEN = NO` because resource, deployment, hardware-realism, and domain-gap validation remain outstanding.

## 28. Primary gate verdict

`GENERALIZATION_HOLDOUT_PRIMARY_GATES_FAIL`

## 29. Final HOLDOUT verdict

`MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`

The timing-tension qualification is unavailable because overall Hazard, Slip, and general specificity fail; the Slip failures include a genuine late detection; and the Sand-benign false-alert family is material. This is final fresh MuJoCo evidence for the exact frozen candidate, not real-world, safety, or production evidence.

## 30. Simulation research status

`SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`

## 31. Guard consumed status

Guard before was 0, its durable transition was `0 -> 1`, guard after is 1, and scientific opens equal one. No second scientific open was attempted. The evaluator refuses any future opening because the guard/artifact identity already exists. The guard must never be reset, and this HOLDOUT may never become training, tuning, threshold, persistence, extraction, or model-selection evidence.

One-shot result SHA-256 values are:

| Artifact | SHA-256 |
|---|---|
| Execution config | `ec53c761f426aaeba5528916c60a6c3f69550007987cdf5f3754304cd4bbef0a` |
| Evaluation result | `2948449fb818335ec2e03ac0b90c34280714bac53570e05f3e67ae1c9bd839da` |
| Run-level result | `18a7a40205f59dd230ef5cbb2a838027a2bfb5764f460d0c72abf1063539152d` |
| Primary metrics | `a0e4b9436c559df6f6966debdfa17b1e22ca494299db9b9ec742650f354e8615` |
| Secondary metrics | `190e7117dc83c221a34e0002314093cbb2237d6f78603e93bcd89507f9bc3628` |
| Terrain diagnostics | `2fd5e867c7b254dc09011b435c5c1a0202e2dc48373ef98c3a98bce77aaa3730` |
| Guard record | `0913da8a583cab7834590d669ba9c8d470485b8129fe81f538aba6c461613154` |

## 32. Limitations

This is a 36-run simulator HOLDOUT with small family/source/speed cells. Slip has no left-only run and only one actual right-only run; Support has no bilateral run. The two design-labeled Ice benign controls that became actual Slip demonstrate why scenario intent cannot substitute for physical labels. Terrain is advisory and its 77.65% event accuracy is not a Hazard gate. No statistical-significance, real-robot, domain-gap, safety, production-readiness, or final-sensor claim follows.

Post-open analysis is limited permanently to these saved summaries. The raw HOLDOUT may not be reread to enrich a table, reproduce a plot, test a hypothesis, or tune a candidate.

## 33. Recommended next milestone

`MODEL_V2_HOLDOUT_FAILURE_INTERPRETATION`

That future milestone may analyze the immutable saved summaries, but the consumed HOLDOUT may never become tuning or training evidence. It must not reopen payloads, retrain, generate simulations, change metrics, or begin architecture/sensor/deployment work. Any modified candidate would require a new, independently designed final evidence set.
