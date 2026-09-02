# Model V2 Generalization Development Evaluation

## 1. Purpose

This milestone evaluates the exact promoted Model V2 candidate on the previously frozen 36-run `GENERALIZATION_VALIDATION` split. It asks whether the data and extraction interventions developed on V2_TRAIN/V2_VALIDATION transfer to the independently designed MuJoCo scenario families that exposed Model V1's limitations.

The primary comparison is frozen Model V1 versus the promoted anchor-refined Model V2. Generalization VALIDATION is development evidence, not a final fresh test. No model selection, tuning, retraining, simulation, or HOLDOUT access is part of this milestone.

## 2. Starting state

The work started from clean `main` at `153eee9b30820d14224b68a83f2a283eee6d5e72` (`Review Model V2 candidate readiness`). `HEAD` equaled `origin/main`, and the tracked worktree was clean.

Before any promoted-V2 inference, the evaluation contract was frozen in [`configs/experiment/20260902_model_v2_generalization_development_evaluation.yaml`](../configs/experiment/20260902_model_v2_generalization_development_evaluation.yaml), SHA-256 `76aa8d6072bdd5cd521e980f62a5ea6b0ec611cec03a6faca883252fe2503b84`.

## 3. Promoted candidate

The promotion record resolved exactly; no substitute or intermediate V2 candidate was evaluated.

| Item | Frozen value |
|---|---|
| Candidate | `model_v2_anchor_refined_gru20_20260902` |
| Development promotion SHA-256 | `1e4931e35e873cd721b412c6a45f66340f7ee9eebc1900d9c4aa3dc9ab3d092f` |
| Candidate freeze SHA-256 | `95dab53275aa77ded36b479c3633ab879a1331cb2c96bd0eb014de2be99bd85f` |
| Internal evaluation freeze SHA-256 | `cad3902137b622c3a3d15ecb3d6c3bb31ee9751f3605fb8d43daa6ac81695c07` |
| Normalizer SHA-256 | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` |
| Checkpoint, seed 20260828 | `7094a2dca40e8d3c84554619652d69c17920c8e82765460ff8621c13ef494cb9` |
| Checkpoint, seed 20260829 | `3ad298eea4c35eca896afd31f860fd6b44ce35d7d9978e2546bb40b693e62c39` |
| Checkpoint, seed 20260830 | `fe96dfeb8461871044de0f8672190680ce46164a1c28efbe6738d22f9d439bbd` |
| Architecture SHA-256 | `ae4753699399aa7e4a93639935e085156ef28ed100a77058d7308eb1549aa897` |
| Feature schema SHA-256 | `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb` |
| Architecture | Pelvis IMU6, causal 80D, `[20,80]`, GRU hidden 32, one unidirectional layer, linear `32->2`, 11,010 parameters |
| Decision | three-seed mean, threshold 0.99, persistence 5 ms, replay stride 1 ms |

Model V1, baseline data-only V2, extraction-rebalanced V2, promoted anchor-refined V2, and Terrain V1 all passed their read-only artifact verifiers and remain exact/restorable. No checkpoint, normalizer, promotion record, or candidate freeze was changed.

## 4. Evidence boundary

Only Generalization VALIDATION waveforms were opened for this comparison. The promoted V2 ensemble was run once over all 36 authorized runs. Retained per-member probabilities were collected during that same fixed three-model traversal for failure diagnostics; no alternative checkpoint combination was run.

Generalization HOLDOUT operations were limited to existence, IDs, counts, stored hashes, and guard metadata. Its NPZ payloads were not deserialized. The historical Unified HOLDOUT was not reopened or reinferred.

```text
Promoted V2 Generalization VALIDATION inference: YES, 36/36 runs, one repetition
Intermediate V2 candidate inference: NO
Generalization HOLDOUT waveform opened: NO
Generalization HOLDOUT model inference: NO
Generalization HOLDOUT Terrain inference: NO
Generalization HOLDOUT visualization: NO
Generalization HOLDOUT guard count: 0
Current Unified HOLDOUT waveform reopened: NO
Current Unified HOLDOUT new inference: NO
```

## 5. Generalization dataset

The frozen dataset is `generalization_hazard_reflex_20260831`. The manifest SHA-256 is `72f5dd300bb5be78b22b9a8c1ad9788cdf45e7cda9b4f46d79eb290140146f53`; all 72 stored files match their manifest hashes. The split-membership hash is `929c963dff786d74728e1aad4be2457fe1dc6b4ef667fdddb1e5c01825822787`.

| Split | Runs | Canonical run-ID SHA-256 | Use here |
|---|---:|---|---|
| `GENERALIZATION_VALIDATION` | 36 | `2b568dc4ef452307cbb99b027162bf9da5a3d2977d70a36fc32b2de1b901e1a1` | V1 parity and one promoted-V2 evaluation |
| `GENERALIZATION_HOLDOUT` | 36 | `6c911c33bc7ea1eb89a58f129d44848989ba4f6aea070f9c15084bdcc2b00c1f` | sealed; metadata/hash checks only |

Validation composition was recomputed as 26 actual Hazard runs—12 Slip and 14 Support—and 10 actual primary no-hazard runs. Actual physical outcomes, rather than design intent, determine every denominator.

The broader integrity audit also reproduced Unified 256/256, Model V2 412/412, and Ice-semantics 48/48 file hashes. No dataset, label, split, or run was modified.

## 6. Frozen evaluation protocol

The original primary contract is unchanged:

- Hazard is established Slip OR established Support.
- Slip is ANY-foot touchdown-anchor tangential drift `>=0.050 m` for 3 ms.
- A Slip response is valid from `Slip-30 ms` through `Slip+40 ms`.
- Support is spread `>=0.010 m` for 20 ms; its valid response region begins at frozen I1 and ends at established Support `+50 ms`.
- A first response before the relevant lower bound is premature and cannot be rescued by a later valid response.
- Any response on a primary no-hazard run is a false positive.
- Terrain is advisory and never gates Hazard.

The separate frozen Ice precursor is loaded exact-Ice drift in the half-open interval `[0.030,0.050) m` before established Slip, with 1,000 ms follow-up and outcomes `SAME_EPISODE_SLIP`, `NEXT_EPISODE_SLIP`, `LATER_SLIP`, `BENIGN_RELEASE`, and `CENSORED`. It remains secondary evidence and does not rewrite the primary score.

## 7. Historical V1 baseline

| Metric | Historical V1 |
|---|---:|
| Overall Hazard recall | 13/26 (50.00%) |
| Slip recall | 7/12 (58.33%) |
| Support recall | 6/14 (42.86%) |
| Primary no-hazard specificity | 5/10 (50.00%) |
| Ice-benign specificity | 3/4 (75.00%) |
| Premature Hazard runs | 7/26 (26.92%) |
| Slip p95 latency | +5.3 ms |
| Support p95 established latency | -17.25 ms |

## 8. V1 replay parity

Direct replay of the exact V1 normalizer and three checkpoints on the 36 validation runs reproduced the committed historical primary result object exactly. This parity barrier completed before V2 interpretation. Family parity also reproduced delayed Ice 3/6 with three premature responses, Ice benign 3/4, delayed Sand Support 0/4 with four premature responses, right Sand Support 0/4, speed Slip 4/6, speed Support 6/6, and speed-Sand benign specificity 2/6.

```text
V1_REPLAY_PARITY: EXACT
```

## 9. Promoted V2 overall result

Promoted V2 achieved 25/26 primary-valid Hazard responses and 10/10 primary true negatives. Its only primary Hazard failure is one early delayed-Ice response.

| Metric | Promoted V2 |
|---|---:|
| Overall Hazard recall | 25/26 (96.15%) |
| Slip recall | 11/12 (91.67%) |
| Support recall | 14/14 (100%) |
| Primary no-hazard specificity | 10/10 (100%) |
| Ice-benign specificity | 4/4 (100%) |
| Premature Hazard runs | 1/26 (3.85%) |
| Slip latency, median / p95 | -13 / +11 ms |
| I1 to Support Reflex, median / p95 | +624 / +1,773 ms |
| Reflex to established Support, median / p95 | 23 / 52 ms lead |
| Support established latency, median / p95 | -23 / -17 ms |

## 10. Primary gates

| Metric | Result | Frozen gate | Status |
|---|---:|---:|---|
| Overall Hazard recall | 25/26 (96.15%) | >=90% | PASS |
| Slip recall | 11/12 (91.67%) | >=95% | **FAIL** |
| Support recall | 14/14 (100%) | >=85% | PASS |
| Primary no-hazard specificity | 10/10 (100%) | >=95% | PASS |
| Ice-benign specificity | 4/4 (100%) | >=95% | PASS |
| Premature rate | 1/26 (3.85%) | <=10% | PASS |
| Slip p95 valid latency | +11 ms | <=+40 ms | PASS |
| Support p95 established latency | -17 ms | <=+50 ms | PASS |

Slip recall remains below 95%; the primary verdict therefore fails even though every other gate passes.

## 11. Slip generalization

V2 improves Slip from 7/12 to 11/12. All six speed-stratified Slip cases pass, including both 0.20 and both 0.30 m/s endpoint cases. Five of six delayed-Ice cases pass the original window. The remaining run has a sustained, all-three-seed early response in a frozen future-Slip precursor state; it is still an official primary failure.

Slip p95 changes from +5.3 to +11 ms. Both values are inside the +40 ms gate, so the unresolved issue is the first response's early timing on one run, not late detection among valid successes.

## 12. Ice precursor secondary interpretation

There are 189 frozen precursor episodes over the validation split, including 171 future-Slip episodes across 12 runs. “Alert” below means a persisted Reflex within the precursor candidate; the separate before-Slip counts preserve the established-Slip relation.

| Category | Episodes / runs | V1 alert in candidate | V2 alert in candidate | V1 / V2 alert before Slip |
|---|---:|---:|---:|---:|
| Same-episode Slip | 102 / 12 | 8 | 12 | 8 / 10 |
| Next-episode Slip | 13 / 8 | 0 | 0 | 0 / 1 |
| Later Slip | 56 / 11 | 6 | 2 | 11 / 6 |
| **All future Slip** | **171 / 12** | **14** | **14** | **19 / 17** |
| Benign release | 5 / 4 | 0 | 0 | n/a |
| Censored | 13 / 8 | 1 | 0 | n/a |

V2 precursor-to-Reflex timing for in-candidate alerts is median 6.5 ms and p95 98 ms; Reflex-to-established-Slip is median 20 ms and p95 140.35 ms. V1 values are 0 / 7.7 ms and 51 / 163.05 ms, respectively. These are descriptive secondary distributions, not primary gate values. The candidate misses most precursor states and is not being redefined as a comprehensive precursor detector.

```text
ICE_PRECURSOR_SECONDARY_DEFINITION: UNCHANGED
PRIMARY_SCORES_REWRITTEN: NO
```

## 13. Ice benign

All four physically no-hazard Ice controls are true negatives. V2 removes V1's single false Reflex. Two runs contain frozen precursor candidates, but their follow-up is censored rather than a fully observed benign release; V2 does not alert on either. The other two have no qualifying loaded exact-Ice precursor episode.

| Run | Contact | Max target drift | Precursor outcome | V2 Reflex | Max p | Max >=.99 streak | Primary |
|---|---:|---:|---|---:|---:|---:|---|
| `ghr_ibc_v_c020` | 1504 | 116.81 mm | 2 censored episodes | none | .983396 | 0 ms | TN |
| `ghr_ibc_v_c030` | 1227 | 105.31 mm | no candidate | none | .540740 | 0 ms | TN |
| `ghr_ibc_v_m020` | 1507 | 86.89 mm | 1 censored episode | none | .927586 | 0 ms | TN |
| `ghr_ibc_v_m030` | 1227 | 83.20 mm | no candidate | none | .509163 | 0 ms | TN |

On `ghr_ibc_v_c020`, one member reaches 0.993533, but the fixed ensemble remains below threshold and never establishes a 5 ms Reflex. This supports specificity without threshold or seed selection.

## 14. Delayed Ice

This family improves from 3/6 to 5/6 under the original primary contract. The set contains two exactly-one-contact and four multi-contact delayed-Slip outcomes. V2 has one premature result, one physically supported precursor-early response, and zero genuine misses.

| Run | Contact pattern / benign contacts before Slip | Contact | Precursor | V2 Reflex | Slip | V1 | V2 primary / secondary | Source / speed / side |
|---|---:|---:|---:|---:|---:|---|---|---|
| `ghr_ocd_v_c020` | multi / 16 | 1504 | 2466 | 2478 | 2632 | premature | premature / supported future-Slip precursor | concrete / .20 / bilateral |
| `ghr_ocd_v_c025` | exactly one / 1 | 1220 | 1889 | 1891 | 1910 | correct | correct / supported precursor | concrete / .25 / bilateral |
| `ghr_ocd_v_c030` | multi / 18 | 1227 | 1894 | 2057 | 2057 | premature | correct / at established Slip | concrete / .30 / bilateral |
| `ghr_ocd_v_m020` | multi / 2 | 1507 | 2465 | 2474 | 2487 | correct | correct / supported precursor | marble / .20 / bilateral |
| `ghr_ocd_v_m025` | exactly one / 1 | 1220 | 1887 | 1887 | 1908 | correct | correct / supported precursor | marble / .25 / bilateral |
| `ghr_ocd_v_m030` | multi / 18 | 1227 | 1889 | 2051 | 2054 | premature | correct / supported precursor | marble / .30 / bilateral |

The unresolved `ghr_ocd_v_c020` response is 154 ms before established Slip, 124 ms before the primary window opens, and 12 ms after precursor entry. It is a primary `PREMATURE` response and a secondary physically supported future-Slip response.

## 15. Delayed Sand Support

V2 eliminates the staged-entry error and detects all four actual delayed Support events. Each run has a 1,791–1,792 ms static interval from first target contact to I1, no pre-I1 V2 Reflex, a Reflex 4 ms after I1, and a 52 ms lead to established Support. V1's first Reflex was 550–553 ms before I1 on all four.

| Run | Source | Target contact | I1 | Support | V2 Reflex | Pre-I1? | I1 to Reflex | Reflex to Support | Primary |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| `ghr_dss_v_c300` | concrete | 1220 | 3011 | 3067 | 3015 | no | 4 ms | 52 ms | correct |
| `ghr_dss_v_c350` | concrete | 1220 | 3011 | 3067 | 3015 | no | 4 ms | 52 ms | correct |
| `ghr_dss_v_m300` | marble | 1220 | 3012 | 3068 | 3016 | no | 4 ms | 52 ms | correct |
| `ghr_dss_v_m350` | marble | 1220 | 3012 | 3068 | 3016 | no | 4 ms | 52 ms | correct |

This is direct external support for the staged-Sand negatives and refined delayed-Support anchors.

## 16. Right Sand Support

All four actual right-only Support runs improve from V1 misses to valid V2 responses. The V2 first Reflex leads established Support by 23–24 ms; all three member maxima exceed 0.99 on every run.

| Run | Source | I1 | Support | First >=.99 | First Reflex | Max p | Primary |
|---|---|---:|---:|---:|---:|---:|---|
| `ghr_rss_v_c300` | concrete | 1526 | 2173 | 2146 | 2150 | .998960 | correct |
| `ghr_rss_v_c350` | concrete | 1526 | 2173 | 2146 | 2150 | .998957 | correct |
| `ghr_rss_v_m300` | marble | 1528 | 2176 | 2148 | 2152 | .999044 | correct |
| `ghr_rss_v_m350` | marble | 1528 | 2176 | 2148 | 2152 | .999065 | correct |

Terrain target state eventually becomes available on all four, but only 839–860 ms after the Hazard Reflex. This left-FSR4 Terrain limitation is separate from the successful Pelvis-IMU6 Hazard result.

## 17. Speed robustness

Every V2 speed-stratified cell is 2/2. Each cell has low `N=2`, so this is deterministic scenario coverage evidence rather than a statistical claim.

| Speed | Subtype | N | V1 | V2 |
|---:|---|---:|---:|---:|
| 0.20 | Slip recall | 2 | 1/2 (50%) | 2/2 (100%) |
| 0.25 | Slip recall | 2 | 2/2 (100%) | 2/2 (100%) |
| 0.30 | Slip recall | 2 | 1/2 (50%) | 2/2 (100%) |
| 0.20 | Support recall | 2 | 2/2 (100%) | 2/2 (100%) |
| 0.25 | Support recall | 2 | 2/2 (100%) | 2/2 (100%) |
| 0.30 | Support recall | 2 | 2/2 (100%) | 2/2 (100%) |
| 0.20 | Sand-benign specificity | 2 | 0/2 (0%) | 2/2 (100%) |
| 0.25 | Sand-benign specificity | 2 | 1/2 (50%) | 2/2 (100%) |
| 0.30 | Sand-benign specificity | 2 | 1/2 (50%) | 2/2 (100%) |

## 18. Side robustness

Actual physical sides, not design intent, produce the following denominators. No unilateral Slip run exists in this validation split.

| Physical event / side | N | V1 | V2 |
|---|---:|---:|---:|
| Slip / left-only | 0 | n/a | n/a |
| Slip / right-only | 0 | n/a | n/a |
| Slip / bilateral | 12 | 7/12 (58.33%) | 11/12 (91.67%) |
| Support / left-only | 10 | 6/10 (60%) | 10/10 (100%) |
| Support / right-only | 4 | 0/4 (0%) | 4/4 (100%) |
| Support / bilateral | 0 | n/a | n/a |

The right-only Support result transfers cleanly; right-only Slip remains untested here.

## 19. Source robustness

Both source domains improve sharply. Each top-level Hazard denominator is 13 and each specificity denominator is 5; low sample counts remain explicit.

| Source | Metric | N | V1 | V2 |
|---|---|---:|---:|---:|
| Concrete | Hazard recall | 13 | 6/13 (46.15%) | 12/13 (92.31%) |
| Concrete | Slip recall | 6 | 3/6 (50%) | 5/6 (83.33%) |
| Concrete | Support recall | 7 | 3/7 (42.86%) | 7/7 (100%) |
| Concrete | Specificity | 5 | 3/5 (60%) | 5/5 (100%) |
| Marble | Hazard recall | 13 | 7/13 (53.85%) | 13/13 (100%) |
| Marble | Slip recall | 6 | 4/6 (66.67%) | 6/6 (100%) |
| Marble | Support recall | 7 | 3/7 (42.86%) | 7/7 (100%) |
| Marble | Specificity | 5 | 2/5 (40%) | 5/5 (100%) |

Family-specific source results are also balanced: delayed Sand Support and right Sand Support are 2/2 for each source, Ice benign is 2/2 for each source, and delayed Ice is concrete 2/3 versus marble 3/3. In the speed family, V2 is 3/3 per source for Slip, Support, and Sand-benign specificity; V1 was respectively 2/3, 3/3, and 2/3 for Concrete and 2/3, 3/3, and 0/3 for Marble.

## 20. Terrain advisory timing

The existing frozen Terrain V1 traces were reused only as advisory timing metadata; no new Terrain inference was performed. Target Terrain was eventually available on 34/36 runs. Among 26 Hazard runs, Terrain preceded V2 Reflex on 13 and Reflex preceded Terrain on 13. Terrain does not enter any Hazard denominator or gate.

The right-Sand cases are the clearest separation: all four Hazard decisions are correct even though Reflex precedes a valid target Terrain state by 839–860 ms.

## 21. Primary failures

There is exactly one promoted-V2 primary Hazard failure; none is omitted.

| Run | Family | Physical event | V2 Reflex | Primary classification | Precursor/I1 relation | Failure class | Genuine miss? |
|---|---|---|---:|---|---|---|---|
| `ghr_ocd_v_c020` | one-contact delayed Ice | Slip at 2632 | 2478 | `PREMATURE` | future-Slip precursor began 2466 | `ICE_PRECURSOR_TIMING_CONFLICT` | no |

Its maximum ensemble probability is 0.997492, its longest `>=0.99` streak is 19 ms, and all seed maxima are high: 0.998904 / 0.999487 / 0.998557. This is not threshold, persistence, or seed instability.

## 22. Genuine detection failures

| Development physical-response class | Runs |
|---|---:|
| Primary Hazard successes | 25 |
| Primary fail with physically supported early response | 1 |
| Genuine detector failure | **0** |
| Support pre-I1 false response | 0 |
| Invalid benign early response | 0 |
| Primary no-hazard false Reflex | 0 |

The diagnostic physically supported Hazard-response view is therefore 26/26, but it is not a new formal gate and does not replace 25/26 primary Hazard recall or 11/12 primary Slip recall.

## 23. V1 versus V2 comparison

| Metric | V1 | Promoted V2 | Delta | Gate | V2 gate result |
|---|---:|---:|---:|---:|---|
| Overall Hazard recall | 13/26 (50.00%) | 25/26 (96.15%) | +46.15 pp | >=90% | PASS |
| Slip recall | 7/12 (58.33%) | 11/12 (91.67%) | +33.33 pp | >=95% | **FAIL** |
| Support recall | 6/14 (42.86%) | 14/14 (100%) | +57.14 pp | >=85% | PASS |
| Primary no-hazard specificity | 5/10 (50.00%) | 10/10 (100%) | +50.00 pp | >=95% | PASS |
| Ice-benign specificity | 3/4 (75.00%) | 4/4 (100%) | +25.00 pp | >=95% | PASS |
| Premature rate | 7/26 (26.92%) | 1/26 (3.85%) | -23.08 pp | <=10% | PASS |
| Slip p95 latency | +5.3 ms | +11 ms | +5.7 ms | <=+40 ms | PASS |
| Support p95 established latency | -17.25 ms | -17 ms | +0.25 ms | <=+50 ms | PASS |

Family comparison:

| Family | N | V1 result | V2 result | Delta | Main interpretation |
|---|---:|---:|---:|---:|---|
| `ONE_CONTACT_DELAYED_ICE_SLIP` | 6 | 3/6 | 5/6 | +33.33 pp | one remaining precursor timing conflict |
| `ICE_BENIGN_CONTROL` | 4 | 3/4 specificity | 4/4 | +25.00 pp | false Reflex removed |
| `DELAYED_SAND_SUPPORT_ONSET` | 4 | 0/4 | 4/4 | +100.00 pp | pre-I1 response eliminated |
| `RIGHT_SAND_SUPPORT` | 4 | 0/4 | 4/4 | +100.00 pp | right Support transfer |
| Speed-stratified Slip | 6 | 4/6 | 6/6 | +33.33 pp | both endpoint-speed gaps close |
| Speed-stratified Support | 6 | 6/6 | 6/6 | 0 pp | preserved |
| Speed-stratified Sand benign | 6 | 2/6 specificity | 6/6 | +66.67 pp | transition false alerts removed |

## 24. Original failure-resolution matrix

| V1 failure | Why it existed | V2 intervention | V1 external | V2 external | Resolution |
|---|---|---|---:|---:|---|
| Delayed Ice | missing delayed/multi-contact Ice coverage | retain and augment delayed Ice | 3/6, 3 premature | 5/6, 1 premature | **not fully primary-resolved**; remaining response is frozen precursor-supported |
| Ice benign | missing near-hazard benign coverage | Ice benign and precursor-aware coverage | 3/4 specificity | 4/4 | resolved |
| Delayed Sand pre-I1 | benign staged-entry transient | staged-Sand negatives plus refined Support anchors | 0/4, 4 premature | 4/4, 0 premature | resolved |
| Right Sand Support | right-side positive coverage absent | balanced right-Support augmentation | 0/4 | 4/4 | resolved |
| 0.20 Slip | endpoint speed coverage absent | source-balanced speed expansion | 1/2 | 2/2 | resolved |
| 0.30 Slip | endpoint speed coverage absent | source-balanced speed expansion | 1/2 | 2/2 | resolved |
| Speed Sand benign | narrow transition hard-negative coverage | speed-stratified Sand-benign negatives | 2/6 specificity | 6/6 | resolved |

Six of seven audited mechanisms fully resolve under the original primary contract. The seventh improves materially and leaves one timing-contract conflict rather than a missing response.

## 25. Data-coverage hypothesis

```text
DATA_COVERAGE_HYPOTHESIS_SUPPORTED
```

With the same GRU20 architecture, runtime features, threshold, and persistence, Model V2 fixes delayed/right Support, both speed endpoints, staged-Sand specificity, and Ice-benign specificity on independently designed development families. The remaining primary failure has an existing physical precursor explanation and is not a genuine detector miss. This pattern supports the hypothesis that Model V1's dominant generalization limitations came from data, side, speed, temporal-anchor, and hard-negative coverage rather than insufficient network capacity.

## 26. Architecture implication

The 20 ms, hidden-32 unidirectional GRU is sufficient for all 14 external Support runs, all 10 external no-hazard controls, and a physically supported response on every Hazard run. No result points to an absent long-memory response or insufficient capacity.

```text
GRU20_ARCHITECTURE_STILL_JUSTIFIED
LONGER_HISTORY_NOT_JUSTIFIED
LSTM_NOT_JUSTIFIED
LARGER_HIDDEN_SIZE_NOT_JUSTIFIED
FEATURE_REDESIGN_NOT_JUSTIFIED
SENSOR_EXPANSION_NOT_JUSTIFIED
```

## 27. Sensor implication

Pelvis IMU6 Hazard transfers to all right-only Support cases without Terrain gating. Combined with left FSR4 advisory Terrain, the provisional ten-channel architecture remains plausible.

```text
10_CHANNEL_ARCHITECTURE_STILL_PLAUSIBLE
FINAL_SENSOR_ARCHITECTURE_FROZEN: NO
```

Hardware realism, resource use, and domain-gap evidence remain future work.

## 28. Development verdict

```text
GENERALIZATION_DEVELOPMENT_SUPPORTED_WITH_ICE_TIMING_TENSION
```

Broad Hazard, Support, specificity, speed, side, and source generalization is strong. The sole remaining primary Slip failure is a sustained, all-seed early response inside a frozen future-Slip precursor, with no genuine detection failure and no benign false-positive problem. This verdict explicitly does not imply that the original Slip gate passed.

## 29. Primary-gate verdict

```text
GENERALIZATION_PRIMARY_GATES_FAIL
```

The reason is exact and singular: Slip recall is 11/12 (91.67%), below the frozen 95% gate. No gate, timing window, denominator, or score was changed after observing the result.

## 30. HOLDOUT preservation

```text
Current Unified HOLDOUT waveform reopened: NO
Current Unified HOLDOUT new inference: NO

Generalization HOLDOUT waveform opened: NO
Generalization HOLDOUT model inference: NO
Generalization HOLDOUT Terrain inference: NO
Generalization HOLDOUT visualization: NO
Generalization HOLDOUT guard count: 0
```

Generalization HOLDOUT remains the fresh 36-run simulation set for a later, separately approved one-shot evaluation.

## 31. Limitations

This is external development evidence from predeclared MuJoCo scenarios, not real-world validation, real-robot robustness, a final safety guarantee, or universal terrain generalization. Several family/source/speed cells have only two or three runs. The precursor secondary view is sparse and must not be treated as a replacement detector metric. Generalization VALIDATION has now been observed by Model V2 and must not be reused for post-result tuning within this lineage.

The frozen Model V2 candidate substantially generalized to the predeclared external MuJoCo development scenario families without retraining on Generalization VALIDATION. Final fresh simulation evidence still requires a separately authorized Generalization HOLDOUT evaluation; hardware and domain-gap evidence remain later work.

## 32. Recommended next milestone

The candidate remains the exact development-promoted candidate. It is not designated `FINAL_GENERALIZATION_CANDIDATE` here. With zero material genuine failure, the recommended next milestone is:

```text
MODEL_V2_FINAL_CANDIDATE_FREEZE_AND_HOLDOUT_READINESS_REVIEW
```

That milestone must review the external comparison, known Ice timing limitation, failed primary Slip gate, integrity, frozen evaluation contract, and sealed HOLDOUT state. It is not started in this work.

## 33. Deterministic artifacts

Generated results are under `artifacts/runs/20260902_model_v2_generalization_development_evaluation/` and contain no HOLDOUT waveform data.

| Artifact | SHA-256 |
|---|---|
| V1 metrics | `d43b7da20306570bd14bf09e4b553c56190d6819be357712f37ee65aea059c59` |
| V2 metrics (`GENERALIZATION_V2_EVALUATION_SHA`) | `291e40aaa96daddd1267f65ed188f205cde5d79d8782379f9bdcb9a95bc5260f` |
| Run-level result | `561d7394dcd3645cbaf64a6ec9089d98106738a37145653fd259c48e33f983a4` |
| Family result | `8127ad696fb293356b9be3a7c03ac838e438b2bffbaffb162dc5a74b900b75b1` |
| Speed result | `5959dc74e252fe753d965e3ba0a0ef88275cd48fc3219f80a7c6dd98905da57c` |
| Side result | `16c31adc207cefb66492b13f2712e89bffca14bb323b1119c3c014b21ff2993a` |
| Source result | `0245a6a0af829d5d4b3dd1e66bc41389018858584e6922222d63d826169cafb5` |
| Ice precursor secondary | `05d1f3e11418ae4b96d66cbce7bdd3e9c8b09ac412a35ede31698120a9dd59b0` |
| Failure-resolution matrix | `e99ebf7704764011411abe99834e18ccfdacf0144919c05e0f4d9d7a777b2276` |
| Evaluation freeze | `cc559c592f64ed8afd27b039d57ace35d8bbb3efad1cac64e3d10788e0ffa556` |

The evaluation implementation recorded source SHA-256 `36cfa3c32a49fc590872f16d68a43708b2de0020feceb07ab867b0845bba439a`. A verifier rechecks the frozen config, promotion, dataset metadata, output hashes, one-pass declaration, and zero HOLDOUT guard count without reopening a waveform.

## 34. Counters and verification

```text
optimizer steps = 0
checkpoint writes = 0
normalizer fits = 0
HNM rounds = 0
threshold searches = 0
persistence searches = 0
architecture searches = 0
seed searches = 0
new simulation runs = 0
```

Verification completed with 88 tests passed and one optional test skipped, full `compileall` over `src`, `scripts`, and `tests`, critical Ruff `E9/F63/F7/F82`, and `git diff --check`. Targeted tests cover validation-only split resolution, HOLDOUT fail-closed behavior, exact promotion resolution, prohibition of alternate candidate selection, frozen primary/secondary semantics, and deterministic run-level artifact hashing.
