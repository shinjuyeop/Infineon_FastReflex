# Model V2 Rebalance Regression Audit

## 1. Purpose

This read-only audit localizes the tradeoff produced by the frozen delayed-Support extraction rebalance. The baseline and rebalanced Model V2 candidates were replayed only on authorized `V2_VALIDATION`; their frozen training, monitor, and HNM provenance was then compared without training, generation, threshold search, or artifact mutation.

The audit is actionable. The three new false reflexes are a narrow transition-left speed-Sand cluster whose later single-left loading states resemble the added delayed-Support `SUPPORT_LOCAL` endpoints more than the other new anchor types. Their exact negative cells were already present in fit and in every reconstructed HNM lineage. Loss weights and the monitor objective also changed, but neither has evidence strong enough to displace local positive-feature overlap as the leading mechanism.

Verdict: `MODEL_V2_REBALANCE_REGRESSION_AUDIT_ACTIONABLE`.

## 2. Starting state

- Starting `HEAD`: `3caea3de0d2c6f4d047ec45c04e8b14f722f5e02`
- Starting `origin/main`: `3caea3de0d2c6f4d047ec45c04e8b14f722f5e02`
- Starting parity: exact
- Starting tracked worktree: clean
- Previous milestone: `MODEL_V2_EXTRACTION_REBALANCED_TRAINING`
- Previous intervention verdict: `V2_EXTRACTION_REBALANCE_NOT_EFFECTIVE`
- Audit config SHA-256: `2527eff85abda1ae51a3c730a484d5c3b35fde33dd51fbbba38c4d5ac6eaed55`

## 3. Evidence boundary

Authorized evidence inspected:

- 442-run effective TRAIN and frozen endpoint identities
- the 96 valid `V2_VALIDATION` runs
- baseline and rebalanced training records, checkpoints, monitor partitions, and HNM provenance
- already-authorized diagnostic contact, displacement, spread, FSR, and event metadata

The read-only HNM comparison deterministically reconstructed selections from saved checkpoints and the frozen policy. This was analysis, not an HNM training round. No new Model V2 inference was run on Generalization VALIDATION. Generalization HOLDOUT and Unified HOLDOUT waveforms were not opened.

## 4. Candidate preservation

The canonical fail-closed verifiers passed for V1, baseline V2, and rebalanced V2.

| Protected object | Verified SHA-256 | Status |
|---|---|---|
| V1 candidate freeze | `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2` | exact/restorable |
| V1 normalizer | `610a91c24a594a3cd1701b07549f4f3c4b7f4cb04eb354f9d41378214a6800a9` | exact |
| V1 final checkpoints | `e6bada49…d588`, `b04877dc…506`, `b6c782bd…753` | 3/3 exact |
| Baseline V2 candidate freeze | `edb16b7d96fb38e680e36dcca5ecd7b1c1682ba410b94d21f40c2a38d4ed8725` | exact/restorable |
| Baseline V2 normalizer | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` | exact |
| Baseline V2 final checkpoints | `dd6c8581…71c8`, `8e6709da…24a0`, `811f486c…4c42` | 3/3 exact |
| Rebalanced V2 candidate freeze | `ca1ba7abae1746528cdd098b903ce8f967937a625773bb8029fc486645fc4533` | exact/restorable |
| Rebalanced evaluation freeze | `3d415c0d4c635497f6d882218a059c9228f02fdda9936124439d709480d94dbc` | exact |
| Rebalanced V2 final checkpoints | `f52a4c86…89883`, `c21e8fe…d648e`, `5d88c027…5f223` | 3/3 exact |
| Architecture / feature schema | `ae475369…a897` / `fe5b6c1c…8adb` | exact |

All manifest rows were rehashed against their NPZ files. Unified 256, Model V2 412, Generalization 72, and Ice-semantics 48 had zero mismatch. No dataset was modified. The Model V2 manifest and NPZ aggregate remained `7a036d34…8b25` and `5a8dfd54…e11c`.

## 5. Result reproduction

Fresh frozen-candidate replay reproduced the recorded aggregate results exactly.

| Metric | Baseline V2 | Rebalanced V2 |
|---|---:|---:|
| Hazard | 55/64 (85.94%) | 58/64 (90.63%) |
| Slip | 29/35 (82.86%) | 29/35 (82.86%) |
| Support | 27/30 (90.00%) | 30/30 (100%) |
| Confirmed no-hazard specificity | 26/26 (100%) | 23/26 (88.46%) |
| Premature | 6/64 (9.38%) | 6/64 (9.38%) |
| Delayed Support | 3/6 | 6/6 |
| Marble delayed Support | 0/3 | 3/3 |
| Speed-Sand benign | 12/12 | 9/12 |

Exact parity: **YES**. Interpretation proceeded only after this check passed.

## 6. Three regressed speed-Sand runs

All three runs are frozen-primary `NO_HAZARD`: I1, established Support, established Slip, and fall are absent; censor is sample 8000. Support spread is zero at each false reflex, so none satisfies the frozen Support semantics.

| Run | Source / speed / topology | Initial Sand contact / touchdown | Baseline `>=.99` / Reflex / max p | Rebalanced `>=.99` / Reflex / max p | Max consecutive `>=.99` B→R | Physical state at false Reflex |
|---|---|---:|---|---|---:|---|
| `m2v2_sbb_v_c_0200_g07` | Concrete / .20 / transition-left | 1808 / 1808 | — / — / .982175 | 5427 / 5431 / .998088 | 0→8 ms | `LEFT_LOADED`; spread 0; displacement 19.977 mm |
| `m2v2_sbb_v_m_0200_g07` | Marble / .20 / transition-left | 1810 / 1810 | — / — / .920206 | 6094 / 6098 / .999357 | 0→10 ms | `LEFT_LOADED`; spread 0; displacement 20.002 mm |
| `m2v2_sbb_v_m_0250_g07` | Marble / .25 / transition-left | 1220 / 1220 | — / — / .981436 | 4888 / 4892 / .999861 | 0→14 ms | `LEFT_LOADED`; spread 0; displacement 20.003 mm |

The three cells are `.20 Concrete left`, `.20 Marble left`, and `.25 Marble left`. All are `g07` transition-left cases; all matching transition-right `g08` controls remain true negatives, as do both `.30` cases. Thus the result clusters primarily by left-transition geometry, secondarily at lower speed and Marble. With only three failures, source and speed effects remain low-N observations rather than general claims.

All 12 speed-Sand controls had frozen designed topology `LEFT_AND_RIGHT_TOPOLOGY`; the topology column below is the designed Sand transition geometry.

| Run | Speed | Source | Topology | Baseline max p | Rebalanced max p | Baseline result | Rebalanced result | Reflex alignment | Seed pattern B→R |
|---|---:|---|---|---:|---:|---|---|---|---|
| **`m2v2_sbb_v_c_0200_g07`** | .20 | Concrete | left | .982175 | .998088 | TN | **FP** | later left load +80 ms | `ONE_HIGH→ALL_HIGH` |
| `m2v2_sbb_v_c_0200_g08` | .20 | Concrete | right | .992923 | .939166 | TN | TN | — | `TWO_HIGH→ALL_LOW` |
| `m2v2_sbb_v_c_0250_g07` | .25 | Concrete | left | .893497 | .948811 | TN | TN | — | `ALL_LOW→ALL_LOW` |
| `m2v2_sbb_v_c_0250_g08` | .25 | Concrete | right | .912747 | .896401 | TN | TN | — | `ALL_LOW→ALL_LOW` |
| `m2v2_sbb_v_c_0300_g07` | .30 | Concrete | left | .944595 | .877888 | TN | TN | — | `ALL_LOW→TWO_HIGH` |
| `m2v2_sbb_v_c_0300_g08` | .30 | Concrete | right | .710356 | .785044 | TN | TN | — | `ALL_LOW→ONE_HIGH` |
| **`m2v2_sbb_v_m_0200_g07`** | .20 | Marble | left | .920206 | .999357 | TN | **FP** | later left load +113 ms | `ALL_LOW→ALL_HIGH` |
| `m2v2_sbb_v_m_0200_g08` | .20 | Marble | right | .992874 | .965306 | TN | TN | — | `ONE_HIGH→ONE_HIGH` |
| **`m2v2_sbb_v_m_0250_g07`** | .25 | Marble | left | .981436 | .999861 | TN | **FP** | later left load +103 ms | `ALL_HIGH→ALL_HIGH` |
| `m2v2_sbb_v_m_0250_g08` | .25 | Marble | right | .754952 | .688862 | TN | TN | — | `ALL_LOW→ALL_LOW` |
| `m2v2_sbb_v_m_0300_g07` | .30 | Marble | left | .981878 | .994934 | TN | TN | no 5 ms persistence | `ONE_HIGH→TWO_HIGH` |
| `m2v2_sbb_v_m_0300_g08` | .30 | Marble | right | .888124 | .881284 | TN | TN | — | `ALL_LOW→ALL_LOW` |

## 7. Timing/alignment analysis

| Run | First `abs(Δp)>=.10` after initial contact | Later left-contact episode | Reflex from episode start | Episode duration | Pelvis IMU norm | FSR diagnostic norm |
|---|---:|---:|---:|---:|---:|---:|
| `m2v2_sbb_v_c_0200_g07` | 1827 (+19 ms) | 5351 | +80 ms | 354 ms | 11.114 | 202.70 |
| `m2v2_sbb_v_m_0200_g07` | 1829 (+19 ms) | 5985 | +113 ms | 360 ms | 10.528 | 210.73 |
| `m2v2_sbb_v_m_0250_g07` | 1241 (+21 ms) | 4789 | +103 ms | 338 ms | 10.507 | 208.54 |

The false reflexes do not align to initial Sand contact or touchdown. They occur well inside later continuous left-contact episodes, not at an episode boundary, static entry, or contact chatter. At the decision endpoint the gait state is single-left loaded, physical spread and spread derivative are zero, and only benign displacement is present. The mechanism is therefore a later speed-Sand loading transient. It is related to the earlier V1 staged-Sand failure at the level of a benign loading/transition response, but it is not the same temporal event: staged controls remain solved and these alerts arise during later repeated gait contact.

## 8. Baseline vs rebalanced probability behavior

The complete ensemble and three-seed traces were replayed and aligned to contact episodes. The compact trace overlay below reports the decision-relevant margins.

| Run | p at rebalanced Reflex B→R | Δp | Δlogit | Duration `>=.90` B→R | `>=.95` B→R | `>=.99` B→R | Response class |
|---|---:|---:|---:|---:|---:|---:|---|
| `m2v2_sbb_v_c_0200_g07` | .876610→.997624 | +.121015 | +4.079 | 13→28 ms | 8→15 ms | 0→8 ms | `MARGIN_SHIFT_ACROSS_THRESHOLD` |
| `m2v2_sbb_v_m_0200_g07` | .920206→.999357 | +.079151 | +4.904 | 5→20 ms | 0→12 ms | 0→10 ms | `NEW_STRONG_SUPPORT_RESPONSE` |
| `m2v2_sbb_v_m_0250_g07` | .962919→.999806 | +.036887 | +5.290 | 11→20 ms | 8→16 ms | 0→14 ms | `MARGIN_SHIFT_ACROSS_THRESHOLD` |

Two runs were already close in ensemble maximum (`.9822` and `.9814`) and crossed through persistence after rebalance. Marble `.20` moved from a clearly lower `.9202` maximum to a strong sustained response. This is not a single barely over-threshold incident.

## 9. Seed behavior

| Run | Baseline seed maxima | Baseline pattern | Rebalanced seed maxima | Rebalanced pattern | Classification |
|---|---|---|---|---|---|
| `m2v2_sbb_v_c_0200_g07` | .999702, .992460, .990277 | `ONE_HIGH` | .998892, .999454, .999903 | `ALL_HIGH` | `SYSTEMATIC_ENSEMBLE_REGRESSION` |
| `m2v2_sbb_v_m_0200_g07` | .995469, .996125, .996068 | `ALL_LOW` | .999681, .998948, .999960 | `ALL_HIGH` | `SYSTEMATIC_ENSEMBLE_REGRESSION` |
| `m2v2_sbb_v_m_0250_g07` | .999559, .999456, .998247 | `ALL_HIGH` but temporally incoherent | .999879, .999844, .999976 | `ALL_HIGH` aligned | `SYSTEMATIC_ENSEMBLE_REGRESSION` |

`HIGH` denotes a seed with a sustained 5 ms reflex, not merely a high isolated maximum. All three rebalanced ensembles are supported by all seeds. The Marble `.25` baseline seeds were individually high at different times but did not produce an ensemble reflex; rebalance aligned the high response. Random single-seed instability is not a plausible primary explanation.

## 10. Delayed-Support gain mechanism

| Run | Source | Baseline p@I1 | Rebalanced p@I1 | Baseline streak | Rebalanced streak | Baseline Reflex | Rebalanced Reflex | Result B→R |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `m2v2_dss_v_c_0250_s10` | Concrete | .994350 | .997968 | 7 ms | 11 ms | I1+43 | I1+4 | valid→valid |
| `m2v2_dss_v_c_0250_s11` | Concrete | .994350 | .997968 | 7 ms | 11 ms | I1+43 | I1+4 | valid→valid |
| `m2v2_dss_v_c_0250_s12` | Concrete | .994350 | .997968 | 7 ms | 11 ms | I1+43 | I1+4 | valid→valid |
| `m2v2_dss_v_m_0250_s10` | Marble | .997845 | .994613 | 3 ms | 10 ms | — | I1+4 | miss→valid |
| `m2v2_dss_v_m_0250_s11` | Marble | .997845 | .994613 | 3 ms | 10 ms | I1+655 (late) | I1+4 | miss→valid |
| `m2v2_dss_v_m_0250_s12` | Marble | .997845 | .994613 | 3 ms | 10 ms | I1+655 (late) | I1+4 | miss→valid |

Concrete I1/Support are 3011/3067; Marble are 3012/3068. Rebalanced Reflex is 52 ms before established Support in all six. Within I1 through Support+50, Concrete duration `>=.99` changes 9→28 ms and Marble changes 8→25 ms. Marble p at I1 actually falls by `.00323`, while the sustained post-I1 response grows from 3 to 10 ms. The solved behavior is therefore a broader Support-development persistence increase, not simply an `I1_ONSET_MARGIN_INCREASE` at the single I1 endpoint. Hidden-state histories were not persisted, so no hidden-state causal claim is made.

## 11. Positive-anchor feature overlap

Distances use the frozen V2 normalizer and identical metrics for all groups. Each query is the rebalanced false-Reflex endpoint; values are medians of per-query nearest-reference statistics. `Current` is normalized 80D endpoint Euclidean distance. `Window` is flattened normalized `20×80` Euclidean distance; cosine is reported to prevent scale alone from controlling the interpretation.

The rebalanced delayed-Support fit pool contains 270 endpoints: 90 I1-onset, 90 midpoint, and 90 Support-local. Relative to baseline delayed-Support fit identities, 39 are retained, 231 are new, and 65 old temporal identities are removed. Each anchor group contributes 77 new and 13 retained endpoints. This was a replacement plus net `+166`, not a pure append.

| Positive anchor type | Count | Regressed Sand current / cosine | Regressed Sand window / cosine | Preserved Sand current | TRAIN speed-Sand current | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| I1-onset | 90 | 4.817 / .318 | 47.567 / .178 | 5.956 | 6.254 | some endpoint overlap, not closest |
| Midpoint | 90 | 5.859 / .172 | 52.815 / .095 | 6.525 | 6.246 | weakest regressed similarity |
| Support-local | 90 | **3.885 / .540** | **42.196 / .327** | 6.177 | 5.905 | closest added anchor; regressed-specific overlap |
| Ordinary Support | 640 | 3.511 / .786 | 30.612 / .826 | 2.727 | 3.893 | closest overall, but unchanged between candidates |

For the three regressed queries, current-distance p10/median/p95 is `3.654/4.817/6.356` for I1, `4.900/5.859/6.883` for midpoint, `2.691/3.885/5.179` for Support-local, and `2.231/3.511/3.662` for ordinary Support. Window-distance p10/median/p95 is respectively `46.256/47.567/61.209`, `49.638/52.815/63.668`, `41.747/42.196/60.746`, and `27.697/30.612/41.599`.

Baseline delayed-Support and newly added delayed-Support median current distances are `3.994` and `3.885`; their window distances are `43.987` and `42.196`. The change makes the regression states modestly more surrounded by delayed-Support positives. Ordinary Support remains closer but cannot by itself explain the between-candidate change because its 640 endpoints are frozen.

Concrete/Marble source separation does not localize the effect further. Support-local current distances are `3.934/.531 cosine` for Concrete and `3.885/.540` for Marble; I1 is `4.817/.318` and `5.057/.285`; midpoint is `5.859/.172` and `6.603/.154`.

## 12. I1-onset analysis

Across the 90 TRAIN I1+0..4 endpoints, every state is bilaterally loaded. Median spread/displacement is 2.820 mm, median spread derivative is 0.2055 mm/ms, Pelvis-IMU norm is 11.555, and FSR diagnostic norm is 199.32. The false-reflex states are single-left loaded with zero spread, approximately 20 mm displacement, and Pelvis-IMU norms 10.51–11.11.

Simulator diagnostics distinguish the physical states, but runtime sees Pelvis IMU6 only. I1 endpoints are closer to the regressed controls than to the nine preserved controls, so partial runtime overlap exists. However, I1 is not the nearest added anchor, and Marble p@I1 does not increase. The specific hypothesis that dense I1-onset exposure alone caused the regression is `WEAK`, not supported strongly enough to select I1 removal.

## 13. Midpoint analysis

The 90 midpoint endpoints split evenly between bilaterally loaded and left-loaded states. Median spread/displacement is 8.419 mm, spread derivative 0.2082 mm/ms, Pelvis-IMU norm 11.054, and FSR norm 236.35. Midpoint has the largest regressed current and history-window distance of the three new groups. It is the cleanest candidate to preserve under a later anchor refinement.

## 14. Support-local analysis

All 90 Support+0..4 endpoints are left loaded. Median spread/displacement is 12.749 mm, median spread derivative is 0.0909 mm/ms, Pelvis-IMU norm is 11.235, and FSR norm is 221.72. The physical Support spread is absent from runtime input. In normalized Pelvis feature space this anchor group is markedly closer to regressed than preserved speed-Sand states (`3.885` versus `6.177` current-distance median; `.540` versus `.076` nearest cosine).

TRAIN-only negative comparisons give current-distance medians of `6.254/6.246/5.905` from speed-Sand negatives to I1/midpoint/Support-local and `6.391/6.377/6.114` from staged-Sand negatives. Support-local is the most benign-like added group under the TRAIN-only endpoint metric. History-window distance does not create a perfectly separable picture, so this is descriptive localization, not proof of a linearly separable boundary.

Anchor interpretation: `SUPPORT_LOCAL_OVERLAP`.

## 15. TRAIN negative-cell coverage

| Validation cell | Matching TRAIN runs / fit runs | Fit-negative endpoints | Baseline / rebalanced cumulative HNM | Nearest fit current / window | Nearest HNM current B / R | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Concrete .20 transition-left | 3 / 2 | 157 | 108 / 108 | 3.797 / 43.938 | 3.637 / 3.852 | present |
| Marble .20 transition-left | 3 / 2 | 160 | 108 / 108 | 2.461 / 63.439 | 2.204 / 2.480 | present |
| Marble .25 transition-left | 3 / 3 | 246 | 108 / 108 | 4.028 / 42.510 | 3.606 / 3.354 | present |

Each cell has three matching TRAIN runs with the same source, speed, and transition-left topology. The fit and both HNM trajectories contain nearby negative states. The correct classification is `NEGATIVE_CELL_PRESENT_BUT_POSITIVE_OVERLAP`, not `NEGATIVE_CELL_UNDEREXPOSURE`.

## 16. Benign-family probability shift

| Family | N | Baseline median max p | Rebalanced median max p | Delta | New FP |
|---|---:|---:|---:|---:|---:|
| Hard normal | 6 | .518976 | .652243 | +.133267 | 0 |
| Ice benign | 4 | .422139 | .614860 | +.192721 | 0 |
| Staged Sand benign | 8 | .278953 | .261305 | -.017647 | 0 |
| Speed Sand benign | 12 | .932401 | .943988 | +.011588 | 3 |
| Other confirmed no-hazard | 2 | .422139 | .614860 | +.192721 | 0 |

There is a broader candidate/calibration movement: hard and Ice benign maxima rise. Those families remain far below `.99`, while staged Sand moves down. Only speed-Sand begins close enough to the boundary for three narrow left-transition cases to acquire persistence. The performance regression is locally realized, though it sits on top of a non-uniform global trajectory shift.

All nine preserved speed-Sand controls remain non-reflex. Rebalanced maxima are `.9392, .9488, .8964, .8779, .7850, .9653, .6889, .9949, .8813`; only Marble `.30` transition-left has any `>=.99` duration (5 total samples, maximum consecutive 3). Thus the vulnerable cases combine left-transition geometry with a sustained coherent response, not merely high isolated probability.

## 17. Loss-weight coupling audit

The implementation recomputes inverse-frequency binary weights from the materialized fit pool in every round: `w_c = N / (2 n_c)`. HNM-added negatives therefore change weights round by round. The weights are not frozen to the baseline extraction.

| Candidate / round | Positive count | Negative count | Positive weight | Negative weight | Ratio |
|---|---:|---:|---:|---:|---:|
| Baseline / 0 | 2,424 | 25,585 | 5.777434 | .547372 | 10.554868 |
| Baseline / 1 | 2,424 | 29,632 | 6.612211 | .540902 | 12.224422 |
| Baseline / 2 | 2,424 | 33,731 | 7.457715 | .535931 | 13.915429 |
| Baseline / 3 | 2,424 | 37,854 | 8.308168 | .532018 | 15.616337 |
| Rebalanced / 0 | 2,590 | 25,585 | 5.439189 | .550616 | 9.878378 |
| Rebalanced / 1 | 2,590 | 29,632 | 6.220463 | .543703 | 11.440927 |
| Rebalanced / 2 | 2,590 | 33,716 | 7.008880 | .538409 | 13.017761 |
| Rebalanced / 3 | 2,590 | 37,828 | 7.802703 | .534234 | 14.605405 |

Relative to baseline, rebalanced positive weight changes `-5.85%, -5.92%, -6.02%, -6.08%`; negative weight changes `+0.59%, +0.52%, +0.46%, +0.42%`; the positive:negative ratio changes `-6.41%, -6.41%, -6.45%, -6.47%`. This is `LOSS_WEIGHT_COUPLED_INTERVENTION`: the prior experiment was not mathematically endpoint-identity-only. Its direction downweights positives relative to negatives, which argues against a simple positive-prior explanation for increased Support sensitivity. It remains a plausible trajectory coupling, not the leading cause.

## 18. Monitor/checkpoint coupling audit

| Candidate | Monitor positives | Monitor negatives | Delayed Support | Concrete | Marble | Other |
|---|---:|---:|---:|---:|---:|---:|
| Baseline V2 | 637 | 6,624 | 52 (16 C / 36 M) | 326 | 311 | 585 |
| Rebalanced V2 | 598 | 6,624 | 13 (0 C / 13 M) | 310 | 288 | 585 |

Positive endpoint intersection/union is `598/637`, Jaccard `.938776`; 39 delayed-Support endpoints are removed and none is added. Negative identities are exact. The checkpoint-selection objective therefore changed: `MONITOR_COMPOSITION_COUPLED_INTERVENTION`.

| Candidate / round | Best epochs by seed | Completed epochs by seed | Best monitor CE by seed | Final TRAIN loss by seed |
|---|---|---|---|---|
| Baseline / 0 | 4, 9, 13 | 10, 15, 19 | .1522, .1484, .1312 | .0722, .0512, .0417 |
| Rebalanced / 0 | 3, 8, 12 | 9, 14, 18 | .1741, .1713, .1468 | .0765, .0532, .0449 |
| Baseline / 1 | 12, 18, 17 | 18, 24, 23 | .1602, .1499, .1637 | .0511, .0368, .0401 |
| Rebalanced / 1 | 11, 14, 15 | 17, 20, 21 | .1648, .1800, .1715 | .0533, .0423, .0413 |
| Baseline / 2 | 7, 4, 14 | 13, 10, 20 | .1829, .2193, .1905 | .0935, .1070, .0663 |
| Rebalanced / 2 | 6, 8, 10 | 12, 14, 16 | .2018, .2031, .1940 | .0922, .0761, .0741 |
| Baseline / 3 | 11, 7, 13 | 17, 13, 19 | .1689, .1730, .2049 | .0735, .0849, .0855 |
| Rebalanced / 3 | 8, 12, 11 | 14, 18, 17 | .2075, .2116, .1999 | .0883, .0592, .0675 |

Rebalanced stops one epoch earlier for every seed in round 0 and generally earlier in round 1, but rounds 2–3 are mixed. Epoch-by-epoch histories were not persisted, and V2_VALIDATION inference on historical checkpoints is forbidden, so this audit cannot locate when specificity deteriorated or show that monitor removal selected the false-positive checkpoint. Monitor coupling is real; causal confidence is low.

## 19. HNM trajectory audit

| Round | Baseline selected | Rebalanced selected | Endpoint intersection / Jaccard | Same-run Jaccard | Speed-Sand B / R | Speed-Sand intersection / Jaccard |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,304 | 5,304 | 2,147 / .2538 | 1.0 | 432 / 432 | 137 / .1884 |
| 2 | 5,304 | 5,304 | 1,253 / .1339 | 1.0 | 432 / 432 | 91 / .1177 |
| 3 | 5,304 | 5,304 | 606 / .0606 | 1.0 | 432 / 432 | 35 / .0422 |

Every round has identical source (`2,652 Concrete / 2,652 Marble`), speed, and family counts. Speed-Sand is 432, staged-Sand 324, and every one of 442 TRAIN runs contributes in both candidates. The full speed count is also exact in each lineage: `.20=1,044`, `.25=2,664`, `.30=1,212`, with the remaining frozen speeds unchanged. Forbidden, duplicate, and spacing violations are zero.

Speed-Sand HNM timing relative to the nearest contact start is mixed rather than depleted: baseline/rebalanced medians are `29/47`, `2.5/28`, and `72/48 ms` in rounds 1–3. Endpoints inside a contact episode are `178/171`, `123/173`, and `261/213` of 432. Exact-cell coverage remains 108 cumulative endpoints for both candidates. HNM identity divergence is a substantial downstream trajectory effect, but it did not miss the vulnerable family or cells and does not independently explain the regression.

## 20. Local vs global regression interpretation

The leading interpretation is a local decision-boundary tradeoff caused by denser delayed-Support exposure around Support development, particularly the `SUPPORT_LOCAL` group. It acts on a subset of already high-margin single-left speed-Sand loading transients. The global loss/monitor/HNM trajectory changed and unrelated benign maxima sometimes rose, so the experiment was not a perfectly pure endpoint-identity intervention. Nevertheless:

- the class-weight change moves against a simple positive-prior explanation;
- staged-Sand does not shift upward;
- all exact negative cells are present and repeatedly mined;
- the regressions are systematic across seeds and geometrically clustered;
- Support-local similarity distinguishes regressed from preserved speed-Sand controls.

The specific shared-decision-boundary hypothesis naming **I1-onset** is `WEAK`. A broader hypothesis naming **Support-development / Support-local Pelvis response** is `SUPPORTED`.

## 21. Support target observability implication

At I1 the physical state is consistently bilateral with small but nonzero spread, whereas all false reflexes are single-left with zero spread. Pelvis-IMU distance still shows partial overlap, but I1 is not the strongest collision. Classification: `SUPPORT_I1_RUNTIME_PARTIALLY_OVERLAPPING`.

At Support-local, simulator spread cleanly separates true Support (median 12.75 mm) from the benign false-reflex states (0 mm), but spread is privileged and absent from the Pelvis-IMU6 runtime tensor. The current binary target therefore has a moderate broader observability tension: it rewards a Support-local state whose runtime projection overlaps benign left loading. This does not yet prove an unavoidable binary-representation failure or justify extra sensors. It does justify refining the ambiguous anchor group before another replay.

## 22. Failure attribution matrix

| Hypothesis | Evidence for | Evidence against | Confidence |
|---|---|---|---|
| I1-onset benign feature overlap | closer to regressions than preserved controls; systemic seed shift | Support-local is closer; Marble p@I1 falls; gain is sustained post-I1 response | LOW |
| Support-local positive overlap | closest added anchor; left-loaded state; strong regressed/preserved separation; privileged spread absent at runtime | ordinary Support is still closer and unchanged; descriptive distance is not causal | MODERATE |
| Global class-prior/loss-weight shift | weights change and unrelated benign maxima rise | positive:negative ratio falls ~6.4%; staged Sand moves down; new FPs are narrow | LOW |
| Monitor/checkpoint-selection coupling | 39 positives removed; objective and early epochs change | later stopping direction is mixed; full history and per-epoch evaluation unavailable | LOW |
| HNM trajectory shift | exact endpoint Jaccard falls .254→.061 | same runs/counts/cells; 432 speed-Sand every round; no violations | LOW |
| Exact speed-Sand negative underexposure | none | 3 matching runs/cell, 157–246 fit negatives, 108 HNM endpoints in both | HIGH (rejected) |
| Intrinsic Support target observability tension | Support-local privileged spread is absent at runtime and Pelvis projection overlaps | I1 remains partly distinguishable; only three validation failures | MODERATE |
| Random seed instability | baseline seeds differ | every rebalanced failure is all-seed high and ensemble-coherent | HIGH (rejected) |

## 23. Smallest justified next intervention

Choose `MODEL_V2_DELAYED_SUPPORT_ANCHOR_REFINEMENT_DESIGN`.

The single variable to design next is the ambiguous `SUPPORT_LOCAL` positive-anchor rule. Preserve all 18 delayed-Support runs, Concrete/Marble balance, effective TRAIN, I1-onset and midpoint groups, the current inverse-frequency formula, monitor-partition rule, HNM policy, normalizer, architecture, seeds, and runtime decision. Use TRAIN-only physical/runtime separability and causal semantics to define the refinement before any replay. Do not search exact offsets on `V2_VALIDATION` and do not simply revert the full delayed-Support rebalance.

The loss and monitor consequences must be recomputed and declared in that design, but their policies must not be independently altered in the same replay. This audit does not select a multi-variable intervention because their causal evidence is weaker than the anchor localization.

## 24. Architecture/sensor implication

Longer history, LSTM, a larger GRU, threshold/persistence change, and sensor expansion are not justified. The current Pelvis IMU6 Hazard plus left FSR4 Terrain architecture remains plausible; final sensor architecture remains not frozen. A target-observability study becomes the next branch only if TRAIN-only anchor refinement cannot separate Support development from benign loading without losing the delayed-Support gain.

## 25. External evidence preservation

- Generalization VALIDATION V2 inference: **NO**
- Current Unified HOLDOUT waveform reopened: **NO**
- Generalization HOLDOUT waveform opened: **NO**
- Generalization HOLDOUT inference: **NO**
- Generalization HOLDOUT guard count: **0**

Verification completed after the audit: `80 passed, 1 skipped`; `compileall` passed for `src`, `scripts`, and `tests`; critical Ruff `E9/F63/F7/F82` passed; `git diff --check` passed. The V1, baseline V2, and rebalanced V2 canonical verifiers all returned `passed: true`. Rehashing all 788 rows across the four protected datasets found zero file-hash or size mismatch.

## 26. Limitations

- The regression set is only three correlated transition-left runs; source/speed claims are low-N.
- Delayed-Support source replicates have limited event-local waveform diversity.
- Nearest-neighbor distances are descriptive in frozen input space, not a diagnostic classifier or proof of causal separability.
- Ordinary Support is closer than every new anchor but is unchanged, so causal localization relies on the marginal intervention and control comparisons.
- Full epoch histories and hidden states were not persisted. Validation-driven historical checkpoint selection was intentionally not recreated.
- Diagnostic spread and FSR clarify physics but are forbidden Hazard runtime inputs.

## 27. Verdict

`MODEL_V2_REBALANCE_REGRESSION_AUDIT_ACTIONABLE`

The regression is sufficiently localized to the added delayed-Support Support-local exposure and its overlap with benign left-loading Pelvis states. Loss, monitor, and HNM are documented coupled variables, but current evidence does not support selecting them as the first isolated remedy.

All audit counters remained zero:

- optimizer steps = 0
- checkpoint writes = 0
- normalizer fits = 0
- HNM rounds = 0
- threshold searches = 0
- persistence searches = 0
- architecture searches = 0
- seed searches = 0
- new simulation runs = 0

## 28. Recommended next milestone

Exactly one: `MODEL_V2_DELAYED_SUPPORT_ANCHOR_REFINEMENT_DESIGN`.

It was not started in this milestone.
