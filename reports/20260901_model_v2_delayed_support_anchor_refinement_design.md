# Model V2 Delayed-Support Anchor Refinement Design

## 1. Purpose

This design-only milestone freezes one smaller delayed-Support positive-anchor policy before any retraining. The selected policy preserves the successful I1 and midpoint neighborhoods, replaces dense `Support+[0..4]` exposure with one deterministic late interior anchor, and changes no data, target semantics, normalizer, negative, mask, model, HNM, or runtime setting.

Verdict: `MODEL_V2_DELAYED_SUPPORT_ANCHOR_REFINEMENT_DESIGN_READY`.

No training, HNM, checkpoint write, normalizer fit, model inference, validation search, or simulation occurred.

## 2. Starting state

- Starting `HEAD`: `19de8786b547f00aa5ddde3bb0b3ef5de5b74388`
- Starting `origin/main`: `19de8786b547f00aa5ddde3bb0b3ef5de5b74388`
- Starting parity: exact
- Starting tracked worktree: clean
- Starting commit subject: `Audit Model V2 rebalance regression`
- Previous verdict: `MODEL_V2_REBALANCE_REGRESSION_AUDIT_ACTIONABLE`

The candidate list was written to the design config before the TRAIN comparison. The resulting pre-analysis config SHA-256 is `97964d2e9137f0010a2601e0488ad2d94ce5d399142e9cd4b25b9a320aa537bd`; the frozen candidate-list SHA-256 is `cd6af9180613101f61fe271f2055d1c8a69daf7acf1d73ef429c8e0f47c555cb`.

## 3. Evidence boundary

Selection used only the 442-run effective TRAIN composition: Unified TRAIN 152 plus valid V2_TRAIN 290, under run-identity SHA `0466ea84871d178856ffb10d8b1c0cec730286b35b589c73ff1b5fb1065aa5ab`. The dry-run resolved frozen endpoint identities, normalized Pelvis features, and diagnostic physical state for TRAIN only.

The prior committed V2_VALIDATION result explains why Support-local anchors were studied, but no V2_VALIDATION waveform, offset, probability, or new candidate inference was used to choose this rule. Generalization VALIDATION and both protected holdouts remained sealed.

| Boundary | State |
|---|---|
| TRAIN inspected | yes, extraction/features only |
| V2_VALIDATION used to choose anchors | no |
| New V2_VALIDATION candidate inference | no |
| Generalization VALIDATION V2 inference | no |
| Unified HOLDOUT reopened | no |
| Generalization HOLDOUT opened / inference | no / no |
| Generalization HOLDOUT guard count | 0 |

## 4. Candidate preservation

Canonical fail-closed verification passed before analysis.

| Protected object | Verified SHA-256 / state |
|---|---|
| V1 candidate freeze | `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`, exact/restorable |
| V1 normalizer | `610a91c24a594a3cd1701b07549f4f3c4b7f4cb04eb354f9d41378214a6800a9`, exact |
| V1 checkpoints | `e6bada49…d588`, `b04877dc…506`, `b6c782bd…753`, 3/3 exact |
| Baseline V2 candidate freeze | `edb16b7d96fb38e680e36dcca5ecd7b1c1682ba410b94d21f40c2a38d4ed8725`, exact/restorable |
| Baseline V2 normalizer | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a`, exact |
| Baseline V2 checkpoints | `dd6c8581…71c8`, `8e6709da…24a0`, `811f486c…4c42`, 3/3 exact |
| Rebalanced V2 candidate freeze | `ca1ba7abae1746528cdd098b903ce8f967937a625773bb8029fc486645fc4533`, exact/restorable |
| Rebalanced V2 evaluation freeze | `3d415c0d4c635497f6d882218a059c9228f02fdda9936124439d709480d94dbc`, exact |
| Rebalanced V2 checkpoints | `f52a4c86…89883`, `c21e8fe…d648e`, `5d88c027…5f223`, 3/3 exact |

Every manifest row was rehashed against its NPZ: Unified 256, Model V2 412, Generalization 72, and Ice-semantics 48 produced zero mismatch. Their manifest SHA-256 values remain `d023384a…e9d6`, `7a036d34…8b25`, `72f5dd30…6f53`, and `6a472d4b…15759`. No artifact or dataset was modified.

## 5. Regression-audit rationale

The rebalanced candidate fixed delayed Support from `3/6` to `6/6`, including Marble `0/3` to `3/3`, but speed-Sand specificity moved from `12/12` to `9/12`. The prior audit rejected negative-cell absence and seed instability, found loss/monitor coupling without strong primary-cause evidence, and localized the closest newly added positive group to `Support+[0..4]`. This study therefore isolates positive-anchor specificity.

## 6. Fixed extraction components

The following remain exact:

- Slip positives: 1,680 (`0.20=435`, `0.25=870`, `0.30=375`; left 135, right 75, bilateral 1,470; immediate 585, delayed 1,095).
- Ordinary Support positives: 640.
- Delayed-Support eligible runs: 18/18, Concrete 9 and Marble 9.
- I1 neighborhood: `I1+[0,1,2,3,4]`.
- Midpoint: `M = I1 + floor((Support-I1)/2)`, then `M+[-2,-1,0,1,2]`.
- Support semantics: spread `>=0.010 m` for 20 ms; I1 and Slip/Ice semantics unchanged.
- Causal model tensor at endpoint `t`: `[t-19,t]`; timestamps are offline annotation only.
- Every pre-I1 delayed-Sand endpoint remains negative.

## 7. Current Support-local problem

The current rule contributes 90 dense late endpoints: 18 runs times `Support+[0..4]`. On TRAIN, this group has the smallest nearest-Speed-Sand Euclidean distance among the three delayed groups: current-80D median `2.2673`, compared with `2.3946` for I1 and `3.0950` for midpoint. Its window median is also smallest at `14.0159` versus `19.9882` and `19.1101`.

The group is physically established and left loaded, but privileged spread is not a Hazard runtime input. Retaining five consecutive Support-local endpoints is not required by the 5 ms persistence contract because the unchanged I1 neighborhood already provides five consecutive positive endpoints.

## 8. Predeclared anchor candidates

Exactly five policies were frozen before comparison; the set was not expanded afterward.

| Policy | Exact late component | Endpoints/run |
|---|---|---:|
| `CURRENT_RULE` | `Support+[0,1,2,3,4]` | 15 |
| `DROP_SUPPORT_LOCAL` | none | 10 |
| `SINGLE_SUPPORT_ENDPOINT` | `Support+[0]` | 11 |
| `SPARSE_SUPPORT_LOCAL` | `Support+[0,4]` | 12 |
| `LATE_PRE_SUPPORT_INTERIOR` | `L=I1+floor(3*(Support-I1)/4)`, `L+[0]` | 11 |

Every policy includes the same five I1 and five midpoint endpoints. There was no unrestricted millisecond or alpha search.

## 9. TRAIN reference groups

Positive references were I1 onset (90), midpoint (90), current Support-local (90), ordinary Support (640), and each predeclared late group. Negative references remained exact: speed-Sand 2,257, staged-Sand 1,672, and an identifiable other Sand-confirmed subset of 2,087 within the 16,628 other confirmed benign negatives.

Candidate summaries were stratified by Concrete/Marble. The nearest-negative-cell audit additionally covers all `.20/.25/.30 m/s`, transition-left/right, and contact-alignment cells.

## 10. Feature-distance methodology

The frozen V2 normalizer `e0d796e8…e92a` transformed the canonical 80D Pelvis endpoint features. For each positive query, the analysis computed nearest-reference Euclidean distance and maximum cosine similarity for both current 80D and flattened causal `20x80` windows. It reports p10/p25/median/p75/p90; no classifier, embedding, ROC, or learned score was used.

Euclidean distance and cosine do not agree uniformly. Cosine discards magnitude, while standardized feature magnitude is physically informative here. The decision therefore treats normalized Euclidean, source consistency, physical state, temporal coverage, and target economy jointly; cosine disagreement is retained as a limitation rather than hidden in an aggregate score.

## 11. Physical-state methodology

At each candidate endpoint, diagnostic-only quantities were support spread, one-step spread derivative, support displacement, loaded contacts, Pelvis accelerometer/gyroscope norm, and gait phase. Spread and contact are never introduced into the runtime tensor.

## 12. Current I1 group

All 90 I1 endpoints are bilaterally loaded. Median spread/displacement is `0.002820 m`, spread derivative `0.0002055 m/ms`, accelerometer norm `11.550`, and gyroscope norm `.3410`. Speed-Sand nearest distance is current `2.3946`, window `19.9882`; staged-Sand is `3.1168`, window `21.0469`. I1 has partial overlap, but it is not the leading ambiguous group and supplies the persistence-length onset neighborhood.

## 13. Current midpoint group

The 90 midpoint endpoints split `45/45` between bilateral and left-loaded state. Median spread/displacement is `0.008419 m`, derivative `0.0002082 m/ms`, accelerometer norm `11.050`, and gyroscope norm `.2897`. It has the strongest Speed-Sand current separation (`3.0950`) and good window separation (`19.1101`). It remains the cleanest temporal-progression group.

## 14. Current Support-local group

All 90 Support-local endpoints are left loaded. Median spread/displacement is `0.012749 m`, derivative `0.0000909 m/ms`, accelerometer norm `11.232`, and gyroscope norm `.2529`. Speed-Sand current/window medians are `2.2673/14.0159`; staged-Sand medians are `2.4847/13.2579`. This is the strongest benign overlap among the three current delayed groups.

## 15. Candidate policy comparisons

Distances below are nearest-negative Euclidean distributions for all delayed-Support endpoints in each complete policy. Each distribution is `p10/p25/median/p75/p90`.

| Policy | Endpoints/run | Delayed positives | Coverage | Speed-Sand current | Speed-Sand window median | Staged current/window median | C/M Speed current median | Temporal coverage | Decision |
|---|---:|---:|---|---|---:|---|---|---|---|
| `CURRENT_RULE` | 15 | 270 | 18/18 | `2.149/2.273/2.488/2.808/3.252` | 19.080 | 2.724 / 16.636 | 2.538 / 2.587 | I1→M→Support dense | reference |
| `DROP_SUPPORT_LOCAL` | 10 | 180 | 18/18 | `2.262/2.414/2.621/3.019/3.283` | 19.650 | 2.941 / 19.474 | 2.549 / 3.044 | I1→M only | good separation, loses late component |
| `SINGLE_SUPPORT_ENDPOINT` | 11 | 198 | 18/18 | `2.273/2.356/2.581/2.943/3.281` | 19.579 | 2.885 / 19.437 | 2.553 / 2.808 | I1→M→Support | Support instant remains ambiguous |
| `SPARSE_SUPPORT_LOCAL` | 12 | 216 | 18/18 | `2.153/2.292/2.558/2.912/3.281` | 19.468 | 2.862 / 18.672 | 2.549 / 2.737 | I1→M→Support | `Support+4` restores low-distance tail |
| `LATE_PRE_SUPPORT_INTERIOR` | 11 | 198 | 18/18 | `2.273/2.434/2.694/2.943/3.281` | 19.579 | 2.941 / 19.437 | 2.553 / 2.975 | I1→M→late interior | **selected** |

The complete-policy Speed-Sand current median improves `2.488→2.694`, and the staged current/window medians improve `2.724/16.636→2.941/19.437`. The selected rule is smaller by 72 delayed positives. `DROP_SUPPORT_LOCAL` has a slightly larger Speed-Sand window median (`19.650`, +`.071`) but forfeits a third late component; that small difference is insufficient to discard progression.

Median current cosine similarity is `.8500` for current and `.8709` for the selected complete policy; window cosine is `.7541→.7878`. These higher similarities do not corroborate the Euclidean improvement. The physical amplitude and Euclidean evidence, not cosine alone, support the decision.

Each dry-run identity was frozen independently:

| Policy | Positive endpoint SHA-256 | Extraction policy SHA-256 |
|---|---|---|
| `CURRENT_RULE` | `058619615ec9364861db6dd5c724927a4d0550f0507142612dfe0f9066eded2d` | `f15a6ca1cb8a4fb906a63cff1729a5c5ea57aa7094967bcf194c01b8ee118850` |
| `DROP_SUPPORT_LOCAL` | `88084ae2b00aef6ad415675a1446c0f0d6a7ca4524612a6eb73a4f3eca7c3ad0` | `b4ced771438eb88f7de9bd0a905cb0182a70d3f54eb6be4a5d3a083e49f18e02` |
| `SINGLE_SUPPORT_ENDPOINT` | `dcb777dd610cc50a846b6b31c3e7b8efc4bcafbbc3a5184982c0b48c8e9af92a` | `dbec6c8b303de599e5f4cc271e6c639fd2667a74f1e54007339b3902e07bd646` |
| `SPARSE_SUPPORT_LOCAL` | `032b0bb3b7abfa6ca0be699e7f7e8aafb70a31b90db442d25202fcb3306fe0a3` | `0700555e90e199d438147283b4d469465a97c95f6e64db2314d84d6720c2162f` |
| `LATE_PRE_SUPPORT_INTERIOR` | `248719864bc1974ac54a21de63f04a6d5e6f55ef3e3c37092cf0ec757872d09e` | `52004bc2ddc307316a7a888855a1bd8014e50b96aa45a178a10965e890f4b199` |

## 16. Source consistency

The selected complete policy improves current-distance medians in both sources: Concrete `2.538→2.553` and Marble `2.587→2.975`. Window medians improve Concrete `20.045→20.387` and Marble `19.140→19.710`. Staged-Sand current medians improve Concrete `2.743→3.165` and Marble `2.854→3.274`.

The isolated late interior anchor is balanced 9/9 and has Speed-Sand current medians `2.8598` Concrete and `2.9755` Marble. The gain is larger on Marble but does not degrade Concrete. One identical equation is used for both sources.

## 17. Temporal coverage

All 18 TRAIN intervals are exactly 56 ms. The selected late anchor therefore occurs at I1+42 ms, 14 ms before the privileged Support timestamp. The policy covers I1 onset, midpoint development, and a late evolving state. Its 11-endpoint cap is exact and every endpoint has only causal `[t-19,t]` history.

## 18. Negative-cell overlap

For each source/speed/topology cell, the table gives the Speed-Sand TRAIN negative nearest to the selected policy. Contact offset is relative to the nearest contact episode; `inside` is explicit. The selected late group is the nearest positive type in six of the twelve cells, while I1/midpoint remain nearest in the closer transition-left cells. Refinement therefore removes the dense Support-local attractor without merely relabeling it as `Support+0`.

| Source | Speed | Topology | Negative run:endpoint | Contact alignment | Nearest positive type | Distance |
|---|---:|---|---|---|---|---:|
| Concrete | .20 | transition-left | `m2v2_sbb_t_c_0200_g03:6617` | inside right, +323 ms | I1 | 2.153 |
| Concrete | .30 | transition-left | `m2v2_sbb_t_c_0300_g02:3762` | inside left, +185 ms | I1 | 2.511 |
| Concrete | .25 | transition-left | `m2v2_sbb_t_c_0250_g03:1274` | inside left, +54 ms | midpoint | 2.546 |
| Marble | .25 | transition-left | `m2v2_sbb_t_m_0250_g03:4814` | inside left, +25 ms | I1 | 2.564 |
| Marble | .20 | transition-left | `m2v2_sbb_t_m_0200_g03:3592` | inside left, +14 ms | I1 | 2.691 |
| Marble | .30 | transition-right | `m2v2_sbb_t_m_0300_g06:6695` | outside right, +2203 ms | late interior | 3.118 |
| Concrete | .30 | transition-right | `m2v2_sbb_t_c_0300_g04:118` | outside right, -1380 ms | late interior | 3.124 |
| Marble | .30 | transition-left | `m2v2_sbb_t_m_0300_g01:3770` | inside left, +193 ms | I1 | 3.136 |
| Marble | .25 | transition-right | `m2v2_sbb_t_m_0250_g04:4896` | outside right, -196 ms | late interior | 3.198 |
| Concrete | .25 | transition-right | `m2v2_sbb_t_c_0250_g04:114` | outside right, -1384 ms | late interior | 3.276 |
| Marble | .20 | transition-right | `m2v2_sbb_t_m_0200_g05:4235` | inside right, +0 ms | late interior | 3.284 |
| Concrete | .20 | transition-right | `m2v2_sbb_t_c_0200_g04:698` | outside right, -1422 ms | late interior | 3.305 |

The auxiliary fraction whose nearest benign point is closer than its nearest ordinary-Support point is `.833` current and `.909` selected. It does not support selection and is not used as an aggregate score; ordinary Support itself is feature-close and unchanged.

## 19. Positive-budget projections

| Fit-positive role | Baseline V2 | Rebalanced V2 | Selected refinement |
|---|---:|---:|---:|
| Slip | 1,680 | 1,680 | 1,680 |
| Ordinary Support | 640 | 640 | 640 |
| Delayed Support | 104 | 270 | 198 |
| All Support | 744 | 910 | 838 |
| Total positives | 2,424 | 2,590 | 2,518 |
| Fit negatives | 25,585 | 25,585 | 25,585 |

The selected policy lies between baseline and rebalanced exposure. Relative to the dense policy it removes 72 positives (`-2.78%` of all rebalanced positives) while preserving all 18 delayed runs.

## 20. Loss-weight projections

Canonical binary weights are `w_c=N/(2*n_c)` and will be recomputed from the refined extraction. No loss formulation or weight tuning is introduced.

| Policy | Positives | Negatives | Positive weight | Negative weight | Ratio | Ratio delta vs baseline | Ratio delta vs rebalanced |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline V2 | 2,424 | 25,585 | 5.777434 | .547372 | 10.554868 | 0% | +6.849% |
| `CURRENT_RULE` | 2,590 | 25,585 | 5.439189 | .550616 | 9.878378 | -6.409% | 0% |
| `DROP_SUPPORT_LOCAL` | 2,500 | 25,585 | 5.617000 | .548857 | 10.234000 | -3.040% | +3.600% |
| `SINGLE_SUPPORT_ENDPOINT` | 2,518 | 25,585 | 5.580421 | .549209 | 10.160842 | -3.733% | +2.859% |
| `SPARSE_SUPPORT_LOCAL` | 2,536 | 25,585 | 5.544361 | .549560 | 10.088722 | -4.416% | +2.129% |
| `LATE_PRE_SUPPORT_INTERIOR` | 2,518 | 25,585 | 5.580421 | .549209 | 10.160842 | -3.733% | +2.859% |

Selected historical-style projection:

| Round | Positives | Projected negatives | Positive weight | Negative weight | Ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 2,518 | 25,585 | 5.580421 | .549209 | 10.160842 |
| 1 | 2,518 | 29,632 | 6.384035 | .542488 | 11.768070 |
| 2 | 2,518 | 33,716 | 7.194996 | .537341 | 13.389992 |
| 3 | 2,518 | 37,828 | 8.011517 | .533282 | 15.023034 |

Round 1–3 negative counts are provenance projections, not assumed future HNM identities. Future policy is `RECOMPUTE_CANONICALLY_FROM_REFINED_EXTRACTION` because the ratio movement is modest and freezing weights would create a second intervention.

## 21. Monitor freeze design

The next replay will use a predeclared candidate-invariant TRAIN monitor. Starting from baseline V2 monitor positives, it removes only exact endpoints colliding with the union of all five frozen candidate fit identities, preserves every other positive and every baseline monitor negative, and freezes the resulting set before training.

| Monitor role | Baseline V2 | Rebalanced V2 | Proposed frozen monitor |
|---|---:|---:|---:|
| Slip | 431 | 431 | 431 |
| Ordinary Support | 167 | 167 | 167 |
| Delayed Support Concrete | 16 | 0 | 8 |
| Delayed Support Marble | 23 | 0 | 11 |
| Benign negatives | 6,624 | 6,624 | 6,624 |
| Total positives | 637 | 598 | 617 |
| Total negatives | 6,624 | 6,624 | 6,624 |

The earlier audit's 52 delayed-family monitor positives included 13 Slip-role endpoints; this role-correct table reports Support-role endpoints only. All 18 delayed-Support runs must contribute fit anchors, so run-disjoint delayed-Support monitoring is impossible under the current corpus. Exact endpoint disjointness is achieved. Twenty candidate-union collisions are removed, and the monitor remains constant across all future HNM rounds. This is experimental-control hygiene, not a performance intervention.

Monitor positive SHA: `e4cd285091e55c92c773512b44958273d7773a708bad876806cec6a8401f9c88`.

Monitor endpoint SHA: `39d30234f674446f305b1b51d446977ba301e6db1e0591ac14dbe7172cbb1bf5`.

## 22. Contradictory-supervision audit

All candidates have zero pre-I1 positive, future-feature, post-censor/fall, and positive-negative collision violations. Baseline masks and negatives remain unchanged.

| Check | Violations |
|---|---:|
| Future-Slip precursor ordinary negative | 0 |
| Censored precursor negative | 0 |
| Pre-I1 positive | 0 |
| Positive-negative collision | 0 |
| Post-censor/fall positive | 0 |
| Future feature leakage | 0 |

The unchanged negative categories are hard normal 3,206; Ice benign 1,629; benign near-threshold Ice 193; staged-Sand 1,672; speed-Sand 2,257; and other confirmed benign 16,628, total 25,585.

## 23. Selected anchor policy

Outcome: `LATE_INTERIOR_ANCHOR_READY`.

For every eligible delayed-Support TRAIN run, take the chronological union:

```text
I1 group:       I1 + [0, 1, 2, 3, 4]
M:              I1 + floor((Support - I1) / 2)
midpoint group: M + [-2, -1, 0, 1, 2]
L:              I1 + floor(3 * (Support - I1) / 4)
late group:     L + [0]
cap:            11 endpoints/run
```

The exact same rule applies to Concrete and Marble. It represents 18/18 runs, with 99 endpoints from nine Concrete runs and 99 from nine Marble runs, total delayed Support 198. Including unchanged Slip and ordinary Support yields 2,518 fit positives.

Why this rule: the isolated late anchor improves Speed-Sand current/window median distance from current Support-local `2.267/14.016` to `2.807/15.422`, and staged-Sand from `2.485/13.258` to `2.973/16.502`. It is already left loaded with median spread/displacement `10.849 mm`, derivative `.1478 mm/ms`, accelerometer norm `11.470`, and gyro norm `.2322`: an evolving late state, not the privileged established instant. It preserves a third temporal component with the same 11-endpoint budget as single Support, while providing stronger current separation than `Support+0` (`2.807` versus `2.420`).

Direct answers:

- A. `Support+[0..4]` contains a distinct late state beyond I1/midpoint, but its five dense samples do not provide enough unique safe Pelvis information to justify their budget.
- B. No. The useful late information is not worth retaining the full dense benign overlap.
- C. Yes. One late interior endpoint preserves progression with substantially better normalized Euclidean separation.
- D. On balance, yes for this frozen representation: current/window Euclidean distance and physical evolution improve, both sources agree, although cosine similarity remains mixed.
- E. No. The overlap is not yet fundamental enough to block anchor refinement or require a sensor/target observability study.

## 24. One-variable future replay contract

The next experiment changes only delayed-Support FIT anchors from the dense rule to the exact selected rule. It keeps 442 effective TRAIN runs, all datasets, V2 normalizer, Slip and ordinary-Support identities, all negative identities, masks, GRU20 hidden32 architecture, seeds, optimizer, loss formula, batch size, three-round HNM policy, threshold `.99`, persistence 5 ms, and target semantics fixed. The predeclared TRAIN monitor is also fixed across rounds to remove accidental checkpoint-selection coupling.

No refined candidate exists yet. It must be trained separately and frozen before a one-shot V2_VALIDATION evaluation.

## 25. Architecture/sensor implication

`ARCHITECTURE_CHANGE_NOT_YET_JUSTIFIED`. Longer history, LSTM, larger GRU, and sensor expansion are not recommended. Pelvis IMU6 Hazard plus left FSR4 Terrain remains plausible; final sensor architecture remains unfrozen. A Support target-observability study is not needed now and becomes relevant only if the frozen anchor replay fails.

## 26. External evidence protection

- New V2_VALIDATION candidate inference: **NO**
- Generalization VALIDATION V2 inference: **NO**
- Current Unified HOLDOUT waveform reopened: **NO**
- Generalization HOLDOUT waveform opened: **NO**
- Generalization HOLDOUT inference: **NO**
- Generalization HOLDOUT guard count: **0**

All counters remained zero: optimizer steps, checkpoint writes, normalizer fits, HNM rounds, threshold searches, persistence searches, architecture searches, seed searches, and new simulation runs.

## 27. Limitations

- The late group has only 18 source-balanced anchors and repeated source-local waveform structure; metrics are descriptive, not proof of learned generalization.
- Euclidean separation improves while cosine similarity is mixed. This motivates a frozen one-variable replay, not a claim of perfect separability.
- Privileged spread and contact explain physics but cannot be runtime inputs.
- The auxiliary nearest-benign-versus-ordinary-Support fraction does not improve.
- Monitor endpoints are exact-endpoint-disjoint but cannot be delayed-run-disjoint because all 18 delayed runs are intentionally fitted.
- No candidate model behavior was inspected; the design does not claim the future candidate will pass V2_VALIDATION.

## 28. Verdict

`MODEL_V2_DELAYED_SUPPORT_ANCHOR_REFINEMENT_DESIGN_READY`

One exact TRAIN-derived policy is frozen; 18/18 run coverage and source balance are preserved; temporal progression remains; normalized Euclidean overlap improves; negatives, masks, Slip, and ordinary Support are unchanged; monitor composition is frozen; and no validation/holdout search occurred.

Frozen identities:

| Artifact | SHA-256 |
|---|---|
| Pre-analysis config | `97964d2e9137f0010a2601e0488ad2d94ce5d399142e9cd4b25b9a320aa537bd` |
| Final design config | `a181461ecb1914ed7ecabdb9491eb591fc4712168f5c0517b2039d311a76ad51` |
| Candidate list | `cd6af9180613101f61fe271f2055d1c8a69daf7acf1d73ef429c8e0f47c555cb` |
| Selected candidate policy | `730e920eea16ac80d86e180ecd7b1281a88c2ac1b129c394a39dd0baa0c54de4` |
| Extraction policy | `52004bc2ddc307316a7a888855a1bd8014e50b96aa45a178a10965e890f4b199` |
| Positive endpoints | `248719864bc1974ac54a21de63f04a6d5e6f55ef3e3c37092cf0ec757872d09e` |
| Negative endpoints | `392c1fda06953135bfed9a97c7cabb3915d4c50fdf9ff11b2ac8e6550448936c` |
| Masks | `32bae2d81b05709771545b375dd6cffb95d2dfc17c2808a7adf3a7f70174c35a` |
| Monitor endpoints | `39d30234f674446f305b1b51d446977ba301e6db1e0591ac14dbe7172cbb1bf5` |
| Anchor-refinement design | `a0525ffce941e05b5e2a51ec6e9d9489ac3abf10f05e775aedf28087ffa7cfdc` |

Repeated TRAIN-only dry-run reproduced byte-identical positive identities. The anchor-refinement design hash is the canonical SHA-256 of the payload recorded by the deterministic dry-run: milestone/source commit, pre-analysis and candidate-list identities, exact selected rule and endpoint hashes, fit/monitor counts, loss policy, seals, and zero counters.

The positive-endpoint SHA covers the partition- and role-aware union of selected fit positives plus the frozen TRAIN-monitor positives. The negative-endpoint SHA likewise covers exact fit and monitor negative identities; the mask SHA covers the unchanged precursor/censor masks.

Final verification passed: `80 passed, 1 skipped`; `compileall` passed for `src`, `scripts`, and `tests`; critical Ruff `E9/F63/F7/F82` passed; `git diff --check` passed. Two independent dry-run executions produced analysis SHA `3d2c215aae7d15f51a154471b234b19e004db4c046e6b8e504ae1685f2929bc6`. V1, baseline V2, and rebalanced V2 verifiers returned `passed: true`; rehashing all 788 protected dataset rows found zero file-hash or size mismatch.

## 29. Recommended next milestone

Exactly one: `MODEL_V2_ANCHOR_REFINED_TRAINING`.

It was not started here.
