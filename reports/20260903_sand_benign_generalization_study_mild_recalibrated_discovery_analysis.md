# Mild-Recalibrated Sand Generalization Discovery Analysis

## 1. Purpose

This milestone analyzes only the fresh 88-run `MILD_RECALIBRATED_DISCOVERY` split. It applies the predeclared model-independent Pelvis, realizable FSR/contact, privileged-oracle, factor-localization, and frozen-V2 rules to select one Discovery hypothesis. It performs no intervention and does not open Confirmation.

The scientific result is `DOMAIN_DIVERSITY_GAP_SUPPORTED`. This is Discovery hypothesis-selection evidence, not Confirmation replication and not a model-support verdict.

## 2. Starting state

- Starting `HEAD` and `origin/main`: `52c0ab71c404fe44daf435ecb99bf502d56ccbc0`
- Starting tracked worktree: clean
- Previous milestone: `SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_GENERATION`
- Previous verdict: `SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_GENERATION_READY`
- Generation gates: 70/70 pass

The analysis contract was frozen before the successful V2 replay. Its file SHA-256 is `e7b4f2f53ac102ca822ad0d00c0b23650d7ed792ded75fcf5d7f371805f2556f`; the pinned canonical implementation SHA-256 is `9ded34b78647e64cd9825070fb80a1397e09d0c73f32df7b901500977ac4014e`.

## 3. Historical scientific boundary

The consumed Generalization HOLDOUT remains permanently closed: guard `1`, scientific opens `1`. This milestone made zero old-HOLDOUT payload reads, feature reconstructions, model inferences, and visualizations. Only previously committed aggregate development-history values and the historical binary specificity `3/6` were referenced.

The historical statuses remain unchanged:

- `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`

## 4. Fresh study integrity

All required identities matched before Discovery feature extraction and model replay.

| Object | Verified SHA-256 |
|---|---|
| Pre-simulation freeze | `d617d1ff423e3f0d416995a3228cbfe0d077ecbad0834356c10bb1d1eb1c5ee1` |
| Manifest | `f19ec527cb9faac0d8f3a385a1a63e8a951ced7f275c7cfc3dd459cc42f375d1` |
| Discovery split | `1c211c38ee2bd7f9e9a44e0f81ec6a0dd110a8e14809c12777c043d33928f93f` |
| Confirmation split | `3bfbb050db3ebadcc363d1c6e51013dc349dbac4e0594c183a55245ea38c2e80` |
| Scenario signatures | `be6d6f0d6bb312617784bad31b55cedfd686bb32aaebd37235efe7a24345fad1` |
| Physical signatures | `dc47c671d5bd7452902fbc7ba1a9e5491f2b1c231af994a881583c304783f51f` |
| NPZ aggregate | `5f63a5e4def8d09159407109f2b51635c5819931551e604c138ba1f02693f3c4` |
| Physical outcomes | `f71b80920b047ad27271ef08c0136a49686992c921ccbebd498f68263b2dfbd6` |
| Gate results | `5034afca0071473888fe5895093cb065868f65df909324b969f1ccd75c1f7a8c` |
| Physical audit | `ec88a6a9b99dc91cd36d4e1bdba88f21a60ca1f1967630e4975aa7eb3e8ffb90` |
| Semantic dataset freeze | `706d939c03bf31df0fb39d1043e99dbbb05922664e207425c8c96ab7c93ee675` |

No dataset, normalizer, checkpoint, or model artifact was rewritten.

## 5. Discovery population

| Physical population | Runs | Primary separability/replay role |
|---|---:|---|
| Strict mild Sand benign | 48 | Sand benign |
| Strict moderate Sand benign | 21 | Sand benign |
| Ordinary established Support | 12 | Support control |
| Delayed established Support | 4 | Support control |
| Invalid moderate intent | 3 | provenance and replay ledger only |
| **Total** | **88** | 85 primary-analysis eligible |

The model-independent analysis uses one anchor, one endpoint, and one vector per eligible run: 85 endpoints and 85 windows. Invalids remain in the immutable split and the 88-run replay ledger, but never enter Sand/Support class metrics.

## 6. Confirmation seal

The Confirmation seal matched `2795fa2cc02a049dbe0de2331820506845d333980e17fd0c6e33e5ce471082c2`. Its state remains `SEALED_FOR_MILD_RECALIBRATED_CONFIRMATION`.

| Confirmation operation | Count/status |
|---|---|
| Generated records | 88 |
| Payload deserializations | 0 |
| V2/V1/Terrain inference | 0/0/0 |
| Normalized 80D analysis | 0 |
| FSR/contact observability analysis | 0 |
| Visualization | 0 |
| Hypothesis testing | 0 |

## 7. Frozen V2

The only replayed candidate was `model_v2_anchor_refined_gru20_20260902`: Pelvis IMU6 → causal 80D → `[20,80]` → one-layer unidirectional GRU hidden 32 → 2 logits, with the exact seeds `20260828/20260829/20260830` averaged. Threshold was inclusive `.99`, persistence was 5 ms, and replay stride was 1 ms.

The final candidate record, candidate freeze, normalizer, checkpoints, and feature schema matched `52644d3e…b7bc2`, `95dab532…d85f`, `e0d796e8…e92a`, `7094a2dc…cb9` / `3ad298ee…c39` / `fe96dfeb…bbd`, and `fe5b6c1c…f8adb`, respectively. There was no candidate, seed, threshold, or persistence selection.

## 8. Pelvis preprocessing

The canonical causal feature extractor generated the exact frozen 80D schema. For benign Sand, the anchor is the earliest maximum trailing-20-ms RMS acceleration deviation from the mean acceleration over up to 200 ms immediately before first target contact; candidates span first target contact through 500 ms after the final pre-censor target contact. Support uses the manifest-frozen I1 first sample. Model probabilities never select an anchor.

The final-V2 normalizer is applied first. Distance representations then use Discovery-pooled per-dimension population mean/std with epsilon `1e-8`, exactly as frozen for later reuse. There is no class-directed dimension selection.

## 9. Pelvis current-80D separability

The current-80D centroid separation is `.689332`, while balanced 1NN/5NN agreement is `.992754/1.000000`, local mixing is `.060598`, the class-balanced median nearest-opposite/same ratio is `1.42903163e8`, and bidirectional radius inclusion is `.760870`. The very large ratio is caused by exact zero-distance same-class anchor vectors and the pre-frozen epsilon denominator; it is retained without clipping and interpreted alongside the other metrics.

Current-80D PCA PC1/PC2 explains `.444066/.314642`; its class projection 95% interval Jaccard overlaps are `.127620/.108926`. PCA is descriptive only.

## 10. Pelvis window separability

The predeclared decision representation, flattened `[20,80]`, meets all four reasonable-Pelvis criteria.

| Metric | Current 80D | Window `[20,80]` | Frozen decision reference |
|---|---:|---:|---|
| Centroid separation | .689332 | **.890733** | reasonable `>=.75`; strong-mixing `<=.60` |
| Balanced 1NN agreement | .992754 | .992754 | descriptive |
| Balanced 5NN agreement | 1.000000 | **.992754** | reasonable `>=.80`; strong-mixing `<=.70` |
| Nearest opposite/same ratio | 1.4290e8 | **1.9185e9** | reasonable `>=1.25`; strong-mixing `<=1.10` |
| Local opposite-class mixing | .060598 | **.007246** | reasonable `<=.30`; strong-mixing `>=.40` |
| Bidirectional 95% radius inclusion | .760870 | .500000 | strong-mixing `>=.75`; descriptive otherwise |
| Within Sand distance, p05/median/p95 | .559/14.481/21.575 | 5.856/53.096/95.715 | descriptive |
| Within Support distance, p05/median/p95 | .791/4.931/7.060 | 9.155/26.513/35.181 | descriptive |
| Between distance, p05/median/p95 | 2.975/6.525/18.754 | 37.941/44.637/89.019 | descriptive |

Window PCA PC1/PC2 explains `.369077/.230684`; projection-overlap Jaccards are `.069282/.074711`. Zero-distance duplicate caveats affect the ratio but not the passing centroid, 5NN, and mixing criteria.

## 11. NN/local mixing analysis

All nearest-neighbor queries exclude the query run; because there is exactly one vector per run, no same-run window can leak into neighbor evidence. Window balanced 1NN and 5NN are both `.992754`; balanced local mixing is `.007246`. The single imperfect balanced-neighbor contribution does not resemble broad Sand/Support mixing. The window 95%-radius inclusion is `.500000`, below the `.75` strong-mixing trigger.

## 12. Factor-localized Pelvis analysis

Predeclared Sand levels were compared against all 16 Support controls without choosing subgroups from model results.

| Factor level | Sand N | Window centroid separation | Balanced 5NN | Local mixing |
|---|---:|---:|---:|---:|
| Concrete | 33 | .977 | 1.000 | .021 |
| Marble | 36 | .839 | .986 | .017 |
| 0.20 m/s | 24 | .842 | .979 | .054 |
| 0.25 m/s | 21 | 1.392 | 1.000 | .010 |
| 0.30 m/s | 24 | .894 | .979 | .021 |
| transition-left | 55 | .895 | 1.000 | .000 |
| transition-right | 14 | 1.270 | .929 | .064 |
| right-single precontact | 52 | .905 | 1.000 | .000 |
| left-single precontact | 14 | 1.270 | .929 | .064 |
| mild/LOW | 48 | 1.007 | 1.000 | .000 |
| moderate/MEDIUM | 21 | 1.295 | .976 | .024 |

The three double-support rows are descriptive only (`N=3`). Entry-time was one frozen level for all 69 strict runs and cannot localize. Exposure LOW/MID/HIGH separation was `.935/1.274/.903` with mixing `.012/.015/.058`. Pelvis class mixing is low across the populated domain, including the factor region in which V2 margins localize.

## 13. Realizable FSR/contact diagnostics

The realizable 39D vector contains only trailing FSR8 mean/max/std/delta, bilateral load-imbalance mean/std/current, and per-foot FSR-sum-derived contact fraction/current. Exact loaded contact and Support spread are excluded. The combined vector is descriptive concatenation with the Pelvis window; no classifier, probe, or fusion model was trained.

| Metric | Pelvis runtime | FSR/contact realizable | Pelvis + contact descriptive | Privileged oracle |
|---|---:|---:|---:|---:|
| Centroid separation | .890733 | 1.103777 | .896029 | 1.242028 |
| Balanced 1NN | .992754 | 1.000000 | .992754 | 1.000000 |
| Balanced 5NN | .992754 | 1.000000 | .992754 | 1.000000 |
| Opposite/same ratio | 1.9185e9 | 3.2914e8 | 1.9505e9 | 2.0338e8 |
| Local mixing | .007246 | .000000 | .007246 | .012500 |
| 95% radius inclusion | .500000 | .500000 | .500000 | .209692 |

Combined-minus-Pelvis deltas are centroid `+.005296`, 5NN `+.000000`, mixing `+.000000`, and ratio `+3.1961e7`. Only the ratio satisfies its improvement threshold; the result is **1/4**, below the required 3/4, with no directional degradation beyond `.05`. Therefore `realizable_fsr_material_increment = false`. The ratio delta inherits the exact-duplicate/epsilon caveat and does not alter the outcome.

## 14. Privileged simulator-oracle diagnostics

The separate 16D oracle uses Support spread, transition displacement, and exact loaded contact. Its centroid separation is `1.242028`, balanced 1NN/5NN are `1.0/1.0`, local mixing is `.012500`, and radius inclusion is `.209692`. Between-distance p05/median/p95 is `4.559/7.464/9.131`, compared with Sand within `0.000493/3.768/8.480` and Support within `.0656/6.829/9.621`.

This confirms that privileged semantics are descriptively clearer on several metrics. It is not realizable runtime evidence and is not used to claim that FSR solves the problem.

## 15. Frozen V2 Discovery replay

The successful operation replayed the exact frozen three-seed V2 once across all 88 Discovery records. The primary ledger then uses 69 strict Sand and 16 actual valid Support; three invalids remain excluded from class scores. No run was regenerated, filtered, retrained, or re-scored under an alternate operating point.

## 16. Sand specificity

Strict Sand specificity is `67/69 = 97.10%`, with two false Reflexes. Both occur in physically clean broad-mild runs; moderate is `21/21` specific.

| Group | N | TN | FP | Specificity | Median max p | p95 max p | Reflex | Near threshold/adverse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All strict Sand | 69 | 67 | 2 | 97.10% | .920693 | .994464 | 2 | 24 |
| Mild | 48 | 46 | 2 | 95.83% | .909796 | .994485 | 2 | 19 |
| Moderate | 21 | 21 | 0 | 100.00% | .930102 | .992235 | 0 | 5 |
| Concrete | 33 | 32 | 1 | 96.97% | .917938 | .992000 | 1 | 12 |
| Marble | 36 | 35 | 1 | 97.22% | .921904 | .994846 | 1 | 12 |
| 0.20 m/s | 24 | 24 | 0 | 100.00% | .926453 | .994177 | 0 | 8 |
| 0.25 m/s | 21 | 20 | 1 | 95.24% | .921005 | .995560 | 1 | 9 |
| 0.30 m/s | 24 | 23 | 1 | 95.83% | .897755 | .991047 | 1 | 7 |
| transition-left | 55 | 53 | 2 | 96.36% | .930102 | .994890 | 2 | 24 |
| transition-right | 14 | 14 | 0 | 100.00% | .834467 | .947149 | 0 | 0 |
| right-single precontact | 52 | 50 | 2 | 96.15% | .933446 | .995021 | 2 | 24 |
| left-single precontact | 14 | 14 | 0 | 100.00% | .834467 | .947149 | 0 | 0 |
| double-support precontact | 3 | 3 | 0 | 100.00% | .930102 | .939419 | 0 | 0 |

## 17. Probability-margin analysis

Strict Sand maximum probability has median `.920693`, p75 `.980802`, p90 `.993474`, p95 `.994464`, and maximum `.998306`. The mutually exclusive frozen bins are:

| Bin | Runs |
|---|---:|
| `<.90` | 27 |
| `[.90,.95)` | 18 |
| `[.95,.99)` | 12 |
| `>=.99`, streak `<5 ms` | 10 |
| Reflex | 2 |

Thus the binary specificity is high, but `24/69 = 34.78%` satisfy the frozen adverse-margin definition. This is a continuing low-margin result and cannot justify threshold or persistence tuning.

## 18. Mild vs moderate

The physically recalibrated mild set is the harder model-margin population: 46/48 specific, 19/48 adverse, median/p95 `.909796/.994485`, with seven subpersistent `>=.99` excursions and two Reflexes. Moderate is 21/21 specific, 5/21 adverse, median/p95 `.930102/.992235`, with three subpersistent `>=.99` excursions and no Reflex.

This rejects the simple expectation that only boundary-moderate Sand deteriorates. The residual low-margin behavior and both false actions remain present in the now physically clean broad-mild domain.

## 19. Source/speed

No source or speed alone passes the frozen localization rule. All six cells nevertheless contribute their full available strict outcomes without subgroup selection.

| Source | Speed | N | TN | FP | Specificity | Median max p | p95 max p | Adverse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Concrete | .20 | 12 | 12 | 0 | 100.00% | .908451 | .960849 | 2 |
| Concrete | .25 | 9 | 9 | 0 | 100.00% | .960590 | .993701 | 5 |
| Concrete | .30 | 12 | 11 | 1 | 91.67% | .934212 | .991981 | 5 |
| Marble | .20 | 12 | 12 | 0 | 100.00% | .968570 | .994403 | 6 |
| Marble | .25 | 12 | 11 | 1 | 91.67% | .916286 | .996764 | 4 |
| Marble | .30 | 12 | 12 | 0 | 100.00% | .809575 | .985446 | 2 |

Adverse cases occur in both sources (`12/12`) and all three speeds (`8/9/7`), satisfying the systematic-pattern coverage condition.

## 20. Topology/phase

All 24 adverse cases are transition-left and right-single precontact. Transition-left has adverse fraction `24/55=.4364`; transition-right has `0/14`. Eligible right-single phase has `24/52=.4615`; left-single has `0/14`. Their fraction ranges and Cramér's V values are `.4364/.3685` for topology and `.4615/.3922` for phase, both above `.25/.20`.

Topology and phase are physically coupled in this deterministic design, so these are not treated as two independent causal discoveries. They jointly identify a predeclared contact-transition region and establish the required metadata localization.

## 21. Geometry/exposure

All strict Sand starts fall in the frozen MID stratum. Width has 68 WIDE and one NARROW row, so neither is eligible for a multi-level localization test. All first-contact times fall in the pre-frozen descriptive EARLY entry-time band.

Exposure LOW/MID/HIGH has `7/25`, `8/20`, and `9/24` adverse cases, respectively, while false Reflexes are `1/1/0`. Exposure does not supply the decisive frozen localization. Geometry and exposure remain descriptive context; no post-result threshold was created.

## 22. Support controls

All 16 Support controls were detected under the frozen I1-through-established-Support+50-ms semantics, with zero pre-I1 Reflexes.

| Support group | N | Correct | Recall | Premature/pre-I1 | I1→Reflex median (range), ms | Reflex→Support median (range), ms |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 12 | 12 | 100% | 0 | 627 (621–1773) | 19.5 (17–31) |
| Delayed | 4 | 4 | 100% | 0 | 4 (4–4) | 52 (52–52) |
| Concrete | 8 | 8 | 100% | 0 | 623.5 (4–1226) | 22 (18–52) |
| Marble | 8 | 8 | 100% | 0 | 626.5 (4–1773) | 21.5 (17–52) |
| Left-only | 9 | 9 | 100% | 0 | 627 (4–1773) | 21 (17–52) |
| Right-only | 7 | 7 | 100% | 0 | 624 (621–1228) | 23 (18–31) |

The delayed-anchor behavior remains exact at I1+4 ms, and the solved Support branch is preserved.

## 23. Development-history margin comparison

The denominators remain separate. No old HOLDOUT waveform or score was reconstructed.

| Development group | N | Specificity | Median max p | p90 | p95 | `[.95,.99)` | `>=.99`, `<5 ms` | Reflex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2_TRAIN Speed Sand | 36 | 36/36 | .779207 | .898661 | .935462 | 1 | 0 | 0 |
| V2_VALIDATION Speed Sand | 12 | 12/12 | .953668 | .987605 | .990276 | 5 | 1 | 0 |
| Generalization VALIDATION Speed Sand | 6 | 6/6 | .986838 | .992835 | .993071 | 3 | 2 | 0 |
| Fresh recalibrated Discovery | 69 | 67/69 | .920693 | .993474 | .994464 | 12 | 10 | 2 |
| Consumed Generalization HOLDOUT | 6 | historical binary 3/6 | not reopened | — | — | — | — | — |

Fresh Discovery is not a comfortable low-probability return to TRAIN behavior. It broadens the denominator, keeps median below V2_VALIDATION, but reproduces a thin upper margin and converts two cases into false Reflexes.

## 24. Failure/low-margin localization

Every frozen adverse-margin run is listed below. `Geometry/exposure` is `realization; start/width m; entry band; exposure band/ms`.

| Run | Mild/moderate | Source | Speed | Topology | Phase | Geometry/exposure | Max p | Streak | Classification |
|---|---|---|---:|---|---|---|---:|---:|---|
| `sbmrc_d_bb_c_020_01` | mild | C | .20 | left | right-single | d_l1; .326/.797; early; high/4491 | .967608 | 0 | `[.95,.99)` |
| `sbmrc_d_bb_c_020_05` | mild | C | .20 | left | right-single | d_l5; .338/.808; early; high/4618 | .955319 | 0 | `[.95,.99)` |
| `sbmrc_d_bb_c_025_02` | mild | C | .25 | left | right-single | d_l2; .329/.803; early; mid/3048 | .995560 | 1 | `>=.99,<5` |
| `sbmrc_d_bb_c_025_03` | mild | C | .25 | left | right-single | d_l3; .332/.805; early; mid/3051 | .989265 | 0 | `[.95,.99)` |
| `sbmrc_d_bb_c_025_06` | mild | C | .25 | left | right-single | d_l6; .341/.795; early; mid/3048 | .990912 | 1 | `>=.99,<5` |
| `sbmrc_d_bb_c_025_07` | mild | C | .25 | left | right-single | d_l7; .327/.801; early; mid/3108 | .962492 | 0 | `[.95,.99)` |
| `sbmrc_d_bb_c_030_01` | mild | C | .30 | left | right-single | d_l1; .326/.797; early; low/2735 | .993631 | 7 | **Reflex** |
| `sbmrc_d_bb_c_030_02` | mild | C | .30 | left | right-single | d_l2; .329/.803; early; low/2720 | .988750 | 0 | `[.95,.99)` |
| `sbmrc_d_bb_c_030_03` | mild | C | .30 | left | right-single | d_l3; .332/.805; early; low/2723 | .989829 | 0 | `[.95,.99)` |
| `sbmrc_d_bb_c_030_04` | mild | C | .30 | left | right-single | d_l4; .335/.807; early; low/2727 | .973055 | 0 | `[.95,.99)` |
| `sbmrc_d_bb_c_030_06` | mild | C | .30 | left | right-single | d_l6; .341/.795; early; low/2723 | .990631 | 1 | `>=.99,<5` |
| `sbmrc_d_bb_m_020_02` | mild | M | .20 | left | right-single | d_l2; .329/.803; early; high/3984 | .994628 | 3 | `>=.99,<5` |
| `sbmrc_d_bb_m_020_03` | mild | M | .20 | left | right-single | d_l3; .332/.805; early; high/3983 | .993939 | 3 | `>=.99,<5` |
| `sbmrc_d_bb_m_020_04` | mild | M | .20 | left | right-single | d_l4; .335/.807; early; high/4165 | .993435 | 2 | `>=.99,<5` |
| `sbmrc_d_bb_m_020_05` | mild | M | .20 | left | right-single | d_l5; .338/.808; early; high/4228 | .989451 | 0 | `[.95,.99)` |
| `sbmrc_d_bb_m_020_06` | mild | M | .20 | left | right-single | d_l6; .341/.795; early; high/3982 | .994219 | 3 | `>=.99,<5` |
| `sbmrc_d_bb_m_025_04` | mild | M | .25 | left | right-single | d_l4; .335/.807; early; mid/3339 | .954014 | 0 | `[.95,.99)` |
| `sbmrc_d_bb_m_025_05` | mild | M | .25 | left | right-single | d_l5; .338/.808; early; mid/3552 | .983224 | 0 | `[.95,.99)` |
| `sbmrc_d_bb_m_025_06` | mild | M | .25 | left | right-single | d_l6; .341/.795; early; mid/3361 | .995502 | 8 | **Reflex** |
| `sbmrc_d_nh_c_025_02` | moderate | C | .25 | left | right-single | d02; .310/.675; early; low/2213 | .960590 | 0 | `[.95,.99)` |
| `sbmrc_d_nh_m_020_03` | moderate | M | .20 | left | right-single | d03; .338/.791; early; high/4209 | .992235 | 1 | `>=.99,<5` |
| `sbmrc_d_nh_m_025_02` | moderate | M | .25 | left | right-single | d02; .332/.783; early; high/3618 | .998306 | 3 | `>=.99,<5` |
| `sbmrc_d_nh_m_030_01` | moderate | M | .30 | left | right-single | d01; .326/.775; early; low/2594 | .980802 | 0 | `[.95,.99)` |
| `sbmrc_d_nh_m_030_04` | moderate | M | .30 | left | right-single | d04; .344/.799; early; mid/2851 | .991121 | 2 | `>=.99,<5` |

The frozen quantitative status is `STRONGLY_METADATA_LOCALIZED`. The direction is transition-left/right-single-precontact adverse versus transition-right/left-single healthy. Source, speed, severity, and exposure do not independently meet the localization thresholds.

## 25. H1 evidence

The adverse pattern is systematic (`24/69`, both sources, all three speeds) and metadata-localized by two coupled predeclared physical factors. The decision Pelvis representation meets 4/4 reasonable-separation gates, its mixing is not broad, realizable FSR is non-material at 1/4 improvements, and Support remains 16/16. All H1 requirements pass.

## 26. H2 evidence

H2 requires nonlocalized systematic behavior, at least three strong-Pelvis-mixing checks, and material realizable FSR increment. The observed result is localized, passes zero of five strong-mixing checks, and has only one of four FSR improvements. Privileged-oracle clarity cannot substitute for realizable FSR evidence. H2 does not match.

## 27. H3 evidence

Reasonable Pelvis separation and non-material FSR are present, but H3 also requires systematic behavior without metadata localization. The topology/phase localization is decisive under the frozen rule, so H3 does not match.

## 28. Decision hierarchy

| Hypothesis | Required frozen evidence | Observed evidence | Meets rule? |
|---|---|---|---|
| H1 Domain diversity | systematic + localized + reasonable Pelvis + no material FSR | yes + yes + 4/4 + FSR 1/4 | **YES** |
| H2 Pelvis observability | systematic + nonlocalized + strong mixing + material FSR | yes + no + 0/5 + no | NO |
| H3 Model representation/capacity | systematic + nonlocalized + reasonable Pelvis + no material FSR | yes + no + yes + yes | NO |

Exactly one predeclared rule matches. No accuracy-only inference was used.

## 29. Frozen Discovery hypothesis

`DOMAIN_DIVERSITY_GAP_SUPPORTED`

The deterministic interpretation hash is:

`SAND_BENIGN_DISCOVERY_INTERPRETATION_SHA = 7c045cd98bb221a0f41911a9662b430548393be90ef192e3194bd619cc3f2ae5`

| Frozen artifact | SHA-256 |
|---|---|
| Pelvis analysis | `ee82865cd8171cc9d337f94f09d5d0242ce1aef92fe9a4064bb384a52e9be6ad` |
| FSR/contact analysis | `1fa3f98ef2319b052795f0e6ab30188d7b3ea9fa7cb7256ba67727080d034374` |
| Privileged-oracle analysis | `02a2bcab898533bc93cf3c00a3d80158682554acaa443e903ec7251434f2dd53` |
| V2 Discovery replay | `c6595d174c73f05d569e6d8d64650493481ecb1346b0713ab580b97446e2164e` |
| Factor localization | `86fbe59e89ed2e4ae7053bebbfd9bd262d88864c8e6ebb2b88f18a551551b0b2` |
| Hypothesis decision | `f650d829e471dd9bb047e7235ef65e822d041acb1040caf429b9006bc22bbb48` |
| Interpretation file | `8461e1ad6b31bb7715b7d0e91c81ea3c6d3547a7c3c4dc0c1d96437e339a66a5` |

## 30. Architecture implication

`ARCHITECTURE_STILL_FAVORS_DATA_DOMAIN`

The current evidence does not justify Model V3, LSTM, longer history, a larger GRU, or any model intervention. The two false actions and broad low margins remain important, but model-independent Pelvis separation and metadata localization place the next evidentiary burden on exact Confirmation replication.

## 31. Sensor implication

`PELVIS_ONLY_HAZARD_STILL_PLAUSIBLE`

Realizable FSR/contact did not materially improve the predeclared combined representation. No FSR Hazard fusion, Foot IMU work, probe, or sensor selection occurred. `FINAL_SENSOR_ARCHITECTURE_FROZEN = NO` remains unchanged.

## 32. Confirmation protocol

The next operation, if authorized separately, must reuse this exact config, implementation, preprocessing, Discovery-pooled scalers, thresholds, localization factors and direction, and the frozen `DOMAIN_DIVERSITY_GAP_SUPPORTED` label. It may open Confirmation once, run the same metrics and frozen V2 once, and test whether the entire H1 rule and topology/phase direction replicate. Hypothesis substitution after opening is prohibited.

Confirmation was not started here.

## 33. Limitations

- This is deterministic MuJoCo evidence, not measured soil, real-robot, safety, or deployment evidence.
- One model-independent anchor per run does not exhaust the trace.
- Pooled distance metrics do not prove Bayes separability.
- Exact duplicate anchor vectors produce epsilon-bounded, very large nearest-distance ratios; the other three reasonable-Pelvis criteria independently pass.
- Topology and precontact phase are coupled, so the two localization passes are one physical region rather than independent causal effects.
- Start and width strata lack eligible multi-level support after physical recalibration.
- Discovery selects H1 but cannot confirm it; Confirmation replication is required.

## 34. Analysis validity

`SAND_BENIGN_GENERALIZATION_STUDY_DISCOVERY_ANALYSIS_VALID`

Dataset and model integrity passed; the protocol was frozen before the successful replay; the exact V2 ran once on Discovery without tuning; one deterministic interpretation was frozen; Confirmation remained sealed; old HOLDOUT access remained zero. A pre-model-analysis attempt stopped on a missing moderate-only exposure field before model loading or replay; the all-Sand fallback was then explicitly frozen and pretested. The successful V2 replay count is exactly one split / 88 runs.

Scientific counters:

```text
optimizer steps = 0
checkpoint writes = 0
normalizer fits = 0
HNM rounds = 0
threshold searches = 0
persistence searches = 0
architecture searches = 0
new simulation runs = 0
V1 inference = 0
Terrain inference = 0
V2 Discovery replay = 1 split / 88 runs
V2 Confirmation inference = 0
old HOLDOUT payload reads = 0
old HOLDOUT feature reconstruction = 0
old HOLDOUT inference = 0
old HOLDOUT visualization = 0
Confirmation feature analysis = 0
```

Pre-replay targeted safety/regression tests passed `14/14`; post-replay targeted hash/seal tests also passed `14/14`. Final repository-wide verification passed `143 passed, 1 skipped`; `compileall src scripts tests`, Ruff `E9,F63,F7,F82`, `git diff --check`, the artifact-hash audit, dataset/model integrity, old-HOLDOUT guard, and Confirmation seal checks all passed.

## 35. Recommended next milestone

`SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_CONFIRMATION_ANALYSIS`

Do not start it automatically. It must test this exact frozen H1 interpretation and direction once, without hypothesis replacement, training, HNM, threshold/persistence tuning, architecture change, or sensor fusion.
