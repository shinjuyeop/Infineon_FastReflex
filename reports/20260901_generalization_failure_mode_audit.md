# Generalization Failure-Mode Audit

## 1. Purpose

This milestone diagnoses why the frozen Unified Hazard candidate failed the fresh `generalization_hazard_reflex_20260831` validation split. It does not change the model, the 80D feature schema, the 20 ms history, the `0.99 / 5 ms` decision, the dataset, or the physical target semantics. The four primary questions are:

1. whether premature delayed-Ice alerts are physical precursors or benign low-friction transitions;
2. whether delayed-Sand alerts roughly 550 ms before I1 are physical precursors or staged-entry false alerts;
3. whether the 0/4 right-only Support result is a Pelvis-IMU observability limit or a side/model distribution failure; and
4. whether the `0.20 / 0.30 m/s` degradation is a simple speed-domain dependency.

The audit verdict is an actionability verdict, not a model-performance PASS.

## 2. Evidence boundary

The experiment was frozen before waveform inspection in `configs/experiment/20260901_generalization_failure_mode_audit.yaml` at starting commit `5964622f14b9e0aae43b2c9e66721b2cf8df6a85`. Its SHA-256 is `15dbd4002aed38fc9db849b17ec44dc4b270b9b2edad0e003954d8f4b2662cb1`.

| Evidence | Access in this audit | Role |
|---|---:|---|
| Generalization VALIDATION | 36/36 waveforms | development diagnosis and exact replay |
| Generalization HOLDOUT | 0/36 waveforms | sealed; existence, membership, and hashes only |
| Unified TRAIN | 152/152 waveforms | frozen normalization and coverage/distribution reference |
| Unified historical HOLDOUT | 0 waveforms | not reopened and no new inference |
| Calibration pilots | prior committed summaries only | qualitative context, never a denominator |

All 72 Generalization NPZ hashes match their frozen manifest, whose file SHA-256 remains `72f5dd300bb5be78b22b9a8c1ad9788cdf45e7cda9b4f46d79eb290140146f53`. All 256 Unified NPZ hashes match their manifest, whose file SHA-256 remains `d023384aaaac22076a2be7c2fa242acac91a548d82102640c39868990cb0e9d6`. Calibration data were not modified and no replacement run was generated.

The Generalization HOLDOUT guard count remains **0**. Hash verification of a sealed file is not waveform inspection; no sealed NPZ was passed to `numpy.load`, a feature extractor, a model, or a viewer.

## 3. Protected model and label semantics

The protected Hazard candidate remains:

```text
Pelvis IMU6 @ 1 kHz
-> frozen causal 80D schema fe5b6c1c...
-> [20, 80]
-> one-layer unidirectional GRU, hidden 32, 11,010 parameters
-> mean of seeds 20260828 / 20260829 / 20260830
-> probability >= 0.99 for 5 consecutive ms
-> HAZARD_REFLEX_REQUIRED
```

Candidate freeze SHA-256 is `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`; normalizer SHA-256 is `610a91c24a594a3cd1701b07549f4f3c4b7f4cb04eb354f9d41378214a6800a9`. All three checkpoint hashes and all protected Terrain hashes passed the canonical verification routine.

Physical labels remain frozen:

- Slip: touchdown-anchor tangential drift `>= 0.050 m` for 3 ms in a valid loaded contact episode;
- I1: positive loaded-foot support-spread derivative for 20 ms, simulator-privileged and not deployable;
- Support: support-surface spread `>= 0.010 m` for 20 ms;
- Hazard: established Slip or established Support;
- no-hazard: neither established Slip, I1, nor established Support.

Terrain remains FSR4/MLP/50 ms advisory only and is not a Hazard gate. The previous Generalization advisory result was 153/171 target events (89.47%) with 2/36 target-unavailable runs. Right-Sand has no clean target event from the left-only Terrain producer; any later advisory Sand state does not explain the Pelvis-IMU-only Hazard miss.

## 4. Reproduction of generalization failure

The current canonical evaluator reproduced the committed validation result exactly before interpretation.

| Metric | Reproduced | Previous | Exact parity |
|---|---:|---:|---:|
| overall Hazard recall | 13/26 = 50.00% | 13/26 = 50.00% | yes |
| Slip recall | 7/12 = 58.33% | 7/12 = 58.33% | yes |
| Support recall | 6/14 = 42.86% | 6/14 = 42.86% | yes |
| primary no-hazard specificity | 5/10 = 50.00% | 5/10 = 50.00% | yes |
| Ice-benign specificity | 3/4 = 75.00% | 3/4 = 75.00% | yes |
| premature Hazard | 7/26 = 26.92% | 7/26 = 26.92% | yes |
| Slip p95 latency | +5.3 ms | +5.3 ms | yes |
| Support p95 established latency | -17.25 ms | -17.25 ms | yes |

Family reproduction also matched: delayed Ice was 3/6 correct plus 3 premature, delayed Sand was 0/4 because all four Reflexes were premature, right-only Sand Support was 0/4 with no onset, and speed-stratified results were Slip 4/6, Support 6/6, and benign specificity 2/6.

## 5. Common timeline methodology

Every validation run was aligned without changing scoring. The run artifact records first target contact, every exact target touchdown, first clean left Terrain target output from the prior frozen evaluation, first `p >= 0.99` crossing, first sustained Reflex, I1, Slip, Support, fall/censor, affected side, source, speed, subtype, and the exclusive 10 ms-lookback gait phase:

```text
TOUCHDOWN_LOADING > CONTACT_RELEASE > DOUBLE_SUPPORT
> LEFT_SINGLE_SUPPORT > RIGHT_SINGLE_SUPPORT > NO_SUPPORT
```

The complete deployed ensemble trace and all three individual seed traces are preserved in the Gitignored `validation_probability_traces.npz`. The full 36-run table is `validation_run_audit.csv`. Threshold excursions at `0.90`, `0.95`, and `0.99`, probability at each physical clock, seed maxima, and maximum consecutive time above threshold are in `audit_diagnostics.json`.

For staged Sand, the first strictly positive stored maximum support displacement is used only as a post-hoc proxy for first deformable-support engagement. The dataset does not retain per-cell support velocity; therefore downward velocity is described by the 1 kHz first difference of the stored maximum displacement. Neither proxy changes I1 or Support.

## 6. Frozen-model probability behavior

| Family | Deployed output | Probability evidence | Seed evidence |
|---|---|---|---|
| delayed Ice | 3 correct, 3 premature | all maxima `0.999988–0.999993`; first Reflex 17–166 ms before Slip | all 6 unanimous at Reflex |
| Ice benign | 1 false positive, 3 true negatives | FP max `0.999956`; TN maxima `0.5536–0.9674` | FP unanimous; one TN had a single-seed crossing |
| delayed Sand | 4 premature | maxima `0.999191–0.999909`; sustained 10–65 ms | all 4 unanimous |
| right Support | 4 misses | ensemble maxima `0.965206–0.973568`; never crossed 0.99 | individual seed maxima varied, but no ensemble crossing |
| speed family | 10 correct, 2 misses, 4 FP, 2 TN | subtype- and source-specific rather than monotonic in speed | 14/18 unanimous reference responses; two Slip misses showed disagreement |

The model is not generally under-confident. Most false/premature decisions are high-confidence, unanimous responses. The right-only Support family is the distinct low-response case.

## 7. Feature distribution shift

Canonical 80D features were computed with the unchanged extractor and transformed with the original frozen Hazard TRAIN normalizer; no fit occurred. Descriptive raw-feature comparisons use deterministic 5 ms sampling from sample 19 to fall/censor. Event-local comparisons use `-1000 / +500 ms` only for post-hoc description.

Against all Unified TRAIN samples, all Generalization VALIDATION has median absolute mean shift `0.015 sigma`, maximum `0.305 sigma`, mean outside-TRAIN-p01–p99 fraction 2.85%, normalized L2/dimension median `0.435`, and p95 `1.820`. Thus the validation corpus as a whole is not a uniformly extreme feature outlier.

| Slice vs all Unified TRAIN | Median abs shift | Max abs shift | Outside TRAIN p01–p99 | Median scaled quantile distance |
|---|---:|---:|---:|---:|
| delayed Ice | 0.040 | 1.066 | 5.80% | 1.413 |
| Ice benign | 0.020 | 0.218 | 2.17% | 0.125 |
| delayed Sand | 0.019 | 0.074 | 1.37% | 0.148 |
| right Support | 0.025 | 0.093 | 1.49% | 0.139 |
| speed 0.20 | 0.021 | 0.336 | 3.28% | 0.419 |
| speed 0.25 | 0.015 | 0.520 | 2.64% | 0.201 |
| speed 0.30 | 0.011 | 0.254 | 2.82% | 0.351 |

Delayed-Ice full-run shift is concentrated in raw/casual gyro norms and gyro variance, but it does not separate the failure: the three correct delayed-Ice runs are farther from Unified Ice TRAIN than the three premature runs (maximum shift `0.804` versus `0.090 sigma` when each is compared specifically with Unified Ice TRAIN). Prematurity is therefore not explained by simple global OOD magnitude.

Delayed Sand is close to Unified Support TRAIN (maximum mean shift `0.044 sigma`). Right Support is also close in full-run magnitude (maximum `0.131 sigma`; 2.04% outside its Support-TRAIN p01–p99 range), but its I1-local window has a structured signed shift: accel-x 5 ms variance `+0.424 sigma`, accel-y causal mean about `-0.275 sigma`, and gyro-z causal mean about `+0.208 sigma`. The matched-speed left Support I1 window is essentially identical to the Unified Support reference under the same calculation (maximum `0.000061 sigma`). This is a side-pattern result, not a gross-amplitude absence.

The Gitignored `feature_distribution_shift.csv` contains all 80 features for every reported group, including median, IQR, p05, p95, standardized mean shift, p01–p99 exceedance fraction, scaled quantile distance, and frozen-normalized summaries.

## 8. Delayed Ice failure

All six validation runs are reported below. Drift and velocity are for the left contact episode carrying the first Reflex; episode maximum is the maximum drift before that episode ends.

| Run | Source / speed | Result | Reflex→Slip | Drift at Reflex | Velocity at Reflex | Episode max drift | Reflex→episode end |
|---|---|---|---:|---:|---:|---:|---:|
| `ghr_ocd_v_c020` | concrete / 0.20 | premature | 162 ms | 34.532 mm | 0.901 m/s | 44.924 mm | 11 ms |
| `ghr_ocd_v_c025` | concrete / 0.25 | correct | 24 ms | 28.208 mm | 0.775 m/s | 57.374 mm | 29 ms |
| `ghr_ocd_v_c030` | concrete / 0.30 | premature | 160 ms | 33.306 mm | 0.823 m/s | 41.947 mm | 10 ms |
| `ghr_ocd_v_m020` | marble / 0.20 | correct | 17 ms | 34.983 mm | 0.905 m/s | 102.220 mm | 50 ms |
| `ghr_ocd_v_m025` | marble / 0.25 | correct | 25 ms | 27.455 mm | 0.731 m/s | 134.766 mm | 75 ms |
| `ghr_ocd_v_m030` | marble / 0.30 | premature | 166 ms | 29.464 mm | 0.787 m/s | 41.558 mm | 14 ms |

Every premature Reflex occurs during a qualifying benign target episode, 10–14 ms before that episode ends and 11–15 ms before a subsequent exact target touchdown/re-contact. The episode reaches 41.56–44.92 mm but never the frozen 50 mm Slip threshold. The model output has fallen to `0.093–0.251` by the later established Slip, confirming that it responded to the earlier episode rather than maintaining a long anticipation of the scored event.

The three correct alerts and three premature alerts are physically similar at first response: correct drift is 27.45–34.98 mm with 0.731–0.905 m/s velocity; premature drift is 29.46–34.53 mm with 0.787–0.901 m/s velocity. The separating fact is future episode outcome, unavailable to a causal 20 ms detector at that instant.

Classification: **`EARLY_PHYSICAL_PRECURSOR_PRESENT + NEAR_HAZARD_FALSE_ALERT + TARGET_SEMANTICS_TENSION`**, HIGH confidence for the near-hazard pattern and MODERATE confidence for the semantics implication. This is not a pure feature-domain failure. It must not automatically be mined as a negative before a separate target-semantics decision.

## 9. Ice-benign false-positive

| Run | Result | Ensemble max | Reflex | Maximum target drift before Reflex / full relevant episode | Seed behavior |
|---|---|---:|---:|---:|---|
| `ghr_ibc_v_c020` | false positive | 0.999956 | 2470 | 34.844 / 43.569 mm | unanimous |
| `ghr_ibc_v_c030` | true negative | 0.553611 | none | 24.651 mm | all low |
| `ghr_ibc_v_m020` | true negative | 0.967391 | none | 36.841 mm | one seed exceeded 0.99 elsewhere |
| `ghr_ibc_v_m030` | true negative | 0.622182 | none | 23.268 mm | all low |

`ghr_ibc_v_c020` exactly reproduces first `p >= 0.99` at sample 2466, Reflex at 2470, maximum probability 0.999956, and 13 consecutive ms above threshold. At Reflex, left-foot drift is 34.844 mm and velocity is 0.944 m/s. Its benign contact episode peaks at 43.569 mm, only 6.431 mm below the Slip threshold, without established Slip. It is the same timestamp and nearly the same state as the `ghr_ocd_v_c020` premature response.

Classification: **`NEAR_HAZARD_FALSE_ALERT + TARGET_SEMANTICS_TENSION`**, HIGH confidence for the first component. The 1/4 family FP is not evidence that all benign Ice is indistinguishable: the other three ensemble maxima remain below 0.99. It is evidence that the current binary target has an unresolved near-slip boundary.

## 10. Delayed Sand pre-I1 alerts

| Run | First 0.99 crossing | First Reflex | Later target touchdown | First displacement | I1 | Support | Reflex→I1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ghr_dss_v_c300` | 1878 | 2461 | 2396 | 2992 | 3011 | 3067 | 550 ms |
| `ghr_dss_v_c350` | 1878 | 2461 | 2396 | 2992 | 3011 | 3067 | 550 ms |
| `ghr_dss_v_m300` | 1877 | 2459 | 2398 | 2993 | 3012 | 3068 | 553 ms |
| `ghr_dss_v_m350` | 1877 | 2459 | 2398 | 2993 | 3012 | 3068 | 553 ms |

The early first crossing is too brief to produce Reflex. The scored first Reflex instead occurs 61–65 ms after a later left target touchdown in a complete qualifying benign static-Sand episode. At Reflex, stored support spread is 0, maximum support displacement is 0, and the displacement-derived downward-velocity proxy is 0 for both feet. First positive displacement/spread begins 531–534 ms later; I1 follows 19 ms after that, and established Support follows another 56 ms later.

This rules out the first Reflex as an early deformable-support precursor in the stored physics. It aligns to a static-entry Sand contact transient. The comparison is reinforced by the four speed Sand-benign false positives, whose Reflexes also occur in left single support 62–87 ms after target touchdown with only 2.38–4.36 mm drift. Delayed Sand is close to Unified Support TRAIN in global feature distribution, so a general feature OOD explanation is weak.

Classification: **`BENIGN_TRANSITION_FALSE_ALERT`** (specifically a staged static-entry touchdown transient), HIGH confidence. `NEW_PHYSICAL_PRECURSOR_CANDIDATE` is not supported by these four runs. Current I1 and Support semantics remain unchanged.

## 11. Right-only Support miss

| Run | Ensemble max | Individual seed maxima | p at Support | Time above 0.95 | Result |
|---|---:|---|---:|---:|---|
| `ghr_rss_v_c300` | 0.971155 | 0.9845 / 0.9892 / 0.9816 | 0.002609 | 3 ms | miss |
| `ghr_rss_v_c350` | 0.973568 | 0.9947 / 0.9904 / 0.9834 | 0.002636 | 3 ms | miss |
| `ghr_rss_v_m300` | 0.973511 | 0.9851 / 0.9955 / 0.9887 | 0.001230 | 3 ms | miss |
| `ghr_rss_v_m350` | 0.965206 | 0.9942 / 0.9974 / 0.9857 | 0.001216 | 3 ms | miss |

All four runs have I1 at 1526–1528 and Support at 2173–2176, but the model remains essentially inactive at those clocks. The ensemble later reaches 0.90 for only 4–5 ms and 0.95 for 3 ms near sample 2454; it never reaches 0.99. This is `NO_STRONG_MODEL_RESPONSE`, not `STRONG_BUT_TOO_BRIEF_RESPONSE` at the deployed threshold.

Pelvis signal is nevertheless present. Against the matched-speed left Support group, right/left event-local acceleration-norm RMS is 10.085/10.410 (96.9%). Per-axis peak-to-peak right/left ratios range from 0.60 to 0.97. Lateral/sign-sensitive components mirror: accel-y has sign-flipped correlation 0.805, gyro-x 0.658, and gyro-z 0.732, while gyro-y has direct correlation 0.744. Magnitude is somewhat attenuated on several axes but not absent.

The matched-speed left Support is 2/2 and essentially coincident with Unified Support TRAIN in the event-local feature comparison. Unified TRAIN contains 38 left Support hazards and **0 right Support hazards**. Full-run right Support remains close to Support TRAIN, while its signed I1-local pattern shifts coherently.

Classification: **`SIDE_DISTRIBUTION_SHIFT + MODEL_GENERALIZATION_FAILURE`**, HIGH confidence. A fundamental `SIDE_OBSERVABILITY_LIMIT` is not supported by the comparable Pelvis-IMU magnitude, although a no-probe, four-run audit cannot prove perfect observability.

## 12. Speed robustness failure

| Speed | Slip | Support | Sand-benign specificity | Main probability/timing behavior |
|---:|---:|---:|---:|---|
| 0.20 | 1/2 | 2/2 | 0/2 | marble Slip peaks 0.999619 but Reflex is +573 ms late; both benign sources false-alert |
| 0.25 | 2/2 | 2/2 | 1/2 | all hazards correct; marble benign false-alert |
| 0.30 | 1/2 | 2/2 | 1/2 | concrete Slip Reflex is +46 ms, 6 ms outside its valid window; marble benign false-alert |

Original Unified Hazard TRAIN has all 76 Hazard runs at exactly 0.25 m/s. Only 38 hard-ground no-hazard runs span 0.1573–0.2779 m/s. Thus hazard-speed endpoints are genuinely absent from training.

However, the observed result is not a simple monotonic speed OOD failure:

- Support remains 6/6 across all three speeds;
- both missed Slip runs eventually sustain `p >= 0.99`, but at the wrong time;
- the missed source flips from marble at 0.20 to concrete at 0.30;
- the 0.25 Ice subtype has a larger maximum feature shift than the failing endpoint subtypes; and
- global median absolute shifts are small and decrease from 0.021 at 0.20 to 0.011 at 0.30.

The stronger interpretation is subtype-specific timing generalization on Slip plus transition false-positive sensitivity on benign Sand, with speed acting as a correlated uncovered factor. Classification for endpoint Slip is **`SPEED_DOMAIN_SHIFT + MODEL_GENERALIZATION_FAILURE`**, MODERATE confidence. Classification for speed Sand-benign is **`BENIGN_TRANSITION_FALSE_ALERT`**, HIGH confidence, with source/speed modulation.

## 13. Source-terrain dependence

Across all 36 runs, concrete has 6 correct, 2 false-positive, 3 miss, 4 premature, and 3 true-negative outcomes; marble has 7 correct, 3 false-positive, 3 miss, 3 premature, and 2 true-negative outcomes. There is no repository-wide concrete-versus-marble winner.

Meaningful local source structure exists:

- the Ice-benign false positive is concrete/0.20, while marble/0.20 peaks at 0.967 but stays negative;
- speed Sand-benign is false-positive on marble at all three speeds, versus only concrete/0.20;
- delayed-Ice premature outcomes are concrete/0.20, both sources/0.30, but neither source/0.25; and
- delayed Sand and right Support reproduce across both sources.

The two speed Slip misses occur on opposite sources at opposite endpoints, so source alone cannot explain them. Source should remain a stratification factor in future data, not a standalone fix.

## 14. Gait/contact-phase dependence

At first Reflex, all 7 premature decisions and all 5 false positives are `LEFT_SINGLE_SUPPORT`. Correct decisions are 10 left-single, 2 right-single, and 1 touchdown-loading. The two speed Slip misses with a Reflex occur at `DOUBLE_SUPPORT` and `TOUCHDOWN_LOADING`; the four right Support misses have no Reflex phase.

This clustering is real but confounded by scenario construction: the failing delayed-Ice, delayed-Sand, and Sand-benign events are left-foot transition episodes, and correct left Support is also left-single. Original TRAIN Hazard phases contain 42 left-single, 14 right-single, 19 touchdown-loading, and only 1 double-support event, with 0 contact-release events. Therefore left-single is covered and not itself causal; double-support/contact-release coverage remains sparse or absent and may contribute to endpoint Slip timing.

## 15. Persistence and seed disagreement

No miss is explained solely by a `>=0.99` excursion shorter than 5 ms.

- Right Support never reaches ensemble 0.99, so its mechanism is low/side-shifted response.
- Concrete/0.30 Slip sustains above threshold for 13 ms but starts 46 ms after Slip, just outside the +40 ms window.
- Marble/0.20 Slip sustains for 7 ms but starts 573 ms after Slip.
- Delayed-Ice premature, Ice-benign FP, delayed-Sand premature, and all four speed Sand-benign FPs are sustained decisions, not single spikes.

The major premature/false-positive families are generally unanimous across seeds. Seed disagreement is most visible in the two missed speed Slip runs and in right Support individual maxima, but changing ensemble rules would not solve the physical timing or benign-transition errors. Lowering the threshold or persistence would also convert the right Support 0.95/3 ms excursion only by expanding already severe false-positive exposure. Threshold and persistence changes are not justified.

## 16. Original TRAIN coverage gaps

Unified TRAIN contains 152 runs: 38 Ice Slip, 38 Sand Support, 38 Sand benign, and 38 hard-ground normal; sources are balanced 76/76. Scenario factors are not balanced.

| Failure mechanism | Meaningful original TRAIN examples | Coverage assessment |
|---|---:|---|
| benign Ice / near-slip without established Slip | 0 | absent |
| complete benign Ice episode followed by later Slip | 0 | absent |
| staged static-Sand entry then deformable Support | 0 | absent |
| right-only Support hazard | 0 | absent |
| Hazard at 0.20 m/s | 0 | absent |
| Hazard at 0.25 m/s | 76 | covered |
| Hazard at 0.30 m/s | 0 | absent |
| Sand benign | 38 | present, but only 0.25 m/s balanced-deformable transitions |
| hard-ground speed diversity | 38 | present only as no-hazard controls |

Affected-side counts are 33 bilateral Slip, 43 left-only hazards (5 Slip plus 38 Support), 0 right-only hazards, and 76 no-hazard runs. The support miss is therefore better attributed to training coverage/model invariance than to demonstrated sensor insufficiency. Sand benign was present, but the new speed and staged/static-entry transient variants expose a hard-negative coverage gap rather than total class absence.

## 17. Failure attribution matrix

Confidence is descriptive consistency across available development runs, not a p-value.

| Failure family | Main observed pattern | Physical evidence | Model evidence | Likely classification | Confidence |
|---|---|---|---|---|---|
| delayed Ice premature | terminal part of a benign low-friction episode | 29–35 mm drift, 0.79–0.90 m/s, episode peaks 42–45 mm; no Slip | unanimous 0.99999; same response range as correct episodes | `EARLY_PHYSICAL_PRECURSOR_PRESENT + NEAR_HAZARD_FALSE_ALERT + TARGET_SEMANTICS_TENSION` | HIGH / MODERATE semantics |
| Ice-benign FP | one near-slip benign episode | 34.8 mm at alert, 43.6 mm episode max, no established Slip | unanimous 0.999956 for 13 ms | `NEAR_HAZARD_FALSE_ALERT + TARGET_SEMANTICS_TENSION` | HIGH |
| delayed Sand pre-I1 | static-entry touchdown response | spread/displacement/velocity proxy all zero at Reflex; deformation starts 531–534 ms later | unanimous sustained 0.99919–0.99991 | `BENIGN_TRANSITION_FALSE_ALERT` | HIGH |
| right Sand miss | side-mirrored signal not learned | comparable norm; signed y/gyro mirroring; right Support absent from TRAIN | ensemble max 0.965–0.974, p at Support 0.001–0.003 | `SIDE_DISTRIBUTION_SHIFT + MODEL_GENERALIZATION_FAILURE` | HIGH |
| 0.20 Slip degradation | marble-only late response | established Slip present | max 0.999619 but Reflex +573 ms | `SPEED_DOMAIN_SHIFT + MODEL_GENERALIZATION_FAILURE` | MODERATE |
| 0.30 Slip degradation | concrete-only slightly late response | established Slip present | Reflex +46 ms, 6 ms outside window | `SPEED_DOMAIN_SHIFT + MODEL_GENERALIZATION_FAILURE` | MODERATE |
| speed Sand-benign FP | repeated target-touchdown transient | very low drift, no Support/I1 | 4/6 sustained unanimous FP; marble 3/3 | `BENIGN_TRANSITION_FALSE_ALERT` | HIGH |

One mechanism does not explain everything. Delayed Sand and speed Sand-benign share terrain-transition/contact-transient sensitivity. Delayed Ice and `ghr_ibc_v_c020` share a different, physically near-threshold low-friction precursor. Right-only Support is independent of speed Support, which remains 6/6.

Family-level action summary:

| Family | Failure | Main evidence | Likely cause | Confidence | Next action |
|---|---|---|---|---|---|
| delayed Ice / Ice benign | early versus current target boundary | correct and premature states overlap causally | missing near-hazard semantics and training coverage | HIGH | dedicated semantics study before negative mining |
| delayed Sand / Sand benign | benign transition FP | Reflex precedes all deformable physics | hard-negative coverage | HIGH | add frozen-label transition negatives in future data design |
| right Support | 0/4 | comparable signal magnitude, mirrored signed axes, zero TRAIN positives | side distribution/model invariance | HIGH | balanced right-side positive augmentation |
| speed Slip | 2/6 endpoint failures | late sustained responses; Hazard endpoints absent from TRAIN | timing-domain coverage | MODERATE | source-balanced speed augmentation |
| speed Support | none | 6/6 across speeds | current response robust in this small slice | MODERATE | retain as regression stratum |

## 18. Sensor-architecture implications

Verdict: **`10_CHANNEL_ARCHITECTURE_STILL_PLAUSIBLE`**.

The right-only Hazard failure raises a legitimate side question, but the current evidence shows comparable Pelvis-IMU magnitude and coherent sign-mirrored structure rather than signal absence. With 0 right Support positives in TRAIN, balanced-side coverage is the smaller, better-supported intervention. A dedicated sensor observability study is not yet required; it becomes justified if a side-balanced Model-v2 development set still produces right-only misses.

This does not freeze the final sensor architecture. The separate left-FSR4 Terrain limitation remains known: it cannot supply clean right-foot target events, but it is advisory and cannot cause the Pelvis-IMU Hazard miss.

## 19. Model-v2 decision

No Model v2 is built in this milestone.

| Question | Decision | Evidence |
|---|---|---|
| More diverse training data justified? | **Yes — `DATA_COVERAGE_EXPANSION_JUSTIFIED`** | all four new failure factors are absent or narrow in TRAIN |
| Hard-negative expansion justified? | **Yes — `HARD_NEGATIVE_EXPANSION_JUSTIFIED`** | delayed-Sand and 4/6 speed Sand-benign transition FPs |
| Right-side augmentation justified? | **Yes — `BALANCED_SIDE_AUGMENTATION_JUSTIFIED`** | right signal present, right Support positives 0 in TRAIN |
| Speed diversity justified? | **Yes — `SPEED_DIVERSITY_AUGMENTATION_JUSTIFIED`** | Hazard endpoints 0 in TRAIN; endpoint Slip timing failures |
| Target-semantics study required? | **Yes — `PHYSICAL_TARGET_SEMANTICS_STUDY_REQUIRED`** | delayed-Ice/IBC alerts correspond to repeatable 42–45 mm near-slip episodes |
| Sensor-observability study required now? | **No, conditional later** | magnitude is observable; resolve side coverage first |
| Longer than 20 ms history justified? | **No** | failures are coverage/timing/semantics; no memory-specific evidence |
| Replace GRU with LSTM? | **No — `ARCHITECTURE_CHANGE_NOT_YET_JUSTIFIED`** | no architecture-family root cause was isolated |
| Threshold or persistence change justified? | **No** | right misses are below threshold; lowering it worsens strong FPs; other misses are mistimed |

Near-slip Ice episodes must not be blindly added as hard negatives. The target-semantics question should be resolved first; static Sand transition episodes can remain negatives under the current frozen physical definition.

## 20. Recommended next milestone

The smallest justified next milestone is **`ICE_NEAR_HAZARD_TARGET_SEMANTICS_STUDY`**.

It should remain diagnostic/design-only and answer whether a causal 29–45 mm, high-velocity, non-established-slip episode is a desired Reflex precursor, a hard negative, or a separately represented near-hazard state. It must not rewrite current labels or inspect either HOLDOUT. Once resolved, the following data-design milestone can combine static Sand hard negatives, balanced right Support positives, and source-balanced 0.20/0.25/0.30 hazard coverage without ambiguous Ice targets.

## 21. Limitations

- Generalization VALIDATION is development evidence: families have only 2–6 runs per relevant stratum and no p-value-like confidence is claimed.
- No trainable probe was fitted. The right-side observability conclusion is based on amplitude, correlation, signed feature shift, and known TRAIN coverage.
- Exact same-source/same-speed pairs with opposite delayed-Ice outcomes do not exist. The strongest neutral pairs are cross-source within speed, `ghr_ibc_v_c020` versus `ghr_ocd_v_c020`, matched-source/speed right versus left Support, and each endpoint Slip versus its same-source 0.25 run.
- The stored Generalization NPZ does not include per-cell support velocity or an explicit static-entry/deformable-cell region bit. First positive stored displacement and its derivative are post-hoc proxies only.
- Full-run feature distributions use a declared 5 ms sampling stride; model inference remains full 1 kHz.
- The current interactive CLI does not natively index this fresh Generalization corpus. A neutral failure selection was frozen, six static explanatory figures were generated from VALIDATION only, no GUI was launched, and no HOLDOUT was visualized.
- Calibration pilots are context only and are excluded from every denominator.

Generated diagnostic artifacts are Gitignored under `artifacts/runs/20260901_generalization_failure_mode_audit/`. The canonical audit artifact declares SHA-256 `be1095ab978ca39ef6a10df9a52ef18e3e06a05acfd492ae622e081ee17c92e8` and preserves the run table, feature table, all seed traces, integrity record, neutral visualization selection, and six figures.

## 22. Verdict

**`GENERALIZATION_FAILURE_MODE_AUDIT_ACTIONABLE`**

The failure modes are sufficiently localized to justify specific follow-up work: target-semantics resolution for near-slip Ice, hard-negative coverage for benign Sand transitions, balanced right-side Support positives, and hazard-speed diversity. The evidence does not justify LSTM, longer history, threshold/persistence tuning, sensor expansion, or opening the sealed Generalization HOLDOUT.

Any future Model v2 design, training, and selection may use Unified TRAIN, Generalization VALIDATION, and newly created TRAIN data. The existing Generalization HOLDOUT must remain unopened until architecture, feature schema, threshold, persistence, dataset, and training decisions are all frozen; it may then be opened once for the final evaluation.

Regression verification completed with `71 passed, 1 skipped`; `python -m compileall src scripts tests` passed; and Ruff `E9,F63,F7,F82` passed. The bare `pytest` executable is not installed in this environment, so the repository-documented fail-closed invocation `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` was used.

Training/search counts for this audit are:

```text
optimizer steps = 0
checkpoint writes = 0
normalizer fits = 0
HNM rounds = 0
threshold searches = 0
persistence searches = 0
architecture searches = 0
```
