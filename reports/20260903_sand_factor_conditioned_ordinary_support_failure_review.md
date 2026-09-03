# Sand Factor-Conditioned Ordinary-Support Failure Review

## Starting state

The review started from scientific base `65f7cf32ca4b099b0380512c5b886874f5354bb1`, with `HEAD == origin/main` and a clean tracked worktree. The frozen `sand_factor_conditioned_development_support_recalibrated_20260903` corpus completed 198/198 runs and passed 57/58 physical generation gates. Its sole failed gate was `yield/FACTOR_TRAIN/ordinary_support`: 20/24 against the frozen minimum 22. No failed run was repaired, rerun, backfilled, supplemented, or promoted.

## Scientific boundary

This was a model-blind physical review. V1, V2, Terrain, Hazard probability, causal 80D reconstruction, training, optimizer, HNM, tuning, and sensor-fusion work were all zero. The permanently consumed Generalization HOLDOUT guard remains at one scientific open; no historical HOLDOUT payload, feature, inference, or visualization was accessed. The E84 repository was not touched.

The findings do not change `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`, `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`, or `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`. `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS` remains `NOT_YET_MODEL_TESTED`.

## Ordinary-Support failure recap

The latest ordinary controls produced 31 correct Support, four observation-invalid records, and one genuine Dual Hazard from 36. TRAIN was 20/24 and failed its already frozen gate; VALIDATION was 11/12 and passed. The Sand redesign did not fail, and delayed-Support recalibration did not fail. The only remaining physical-generation bottleneck was TRAIN ordinary Support.

Every row used the unchanged `lateral_deformable` ordinary-Support mechanics: entry medial/reference travel `0.004 m`, stiffness `50,000 N/m`, damping `1,000 Ns/m`; exit lateral-moderate travel `0.040 m`, stiffness `7,000 N/m`, damping `374 Ns/m`. `TD<I1` is the canonical count of complete target touchdowns before I1. Times are milliseconds. `Fall/censor` is the actual fall sample when present, otherwise the censor sample; `Post/req` is censor minus established-Support time against the unchanged 1,000 ms requirement.

## 36-run physical ledger

| Run | Split | Source | Speed | Side (intended→actual) | Topology | Phase | Entry | TD<I1 | I1 | Support | Slip | Start/width/exit | Fall/censor | Post/req | Obs valid | Final outcome |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| `sfcsr_t_osp_c_020_01` | TRAIN | concrete | 0.20 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1808 | 1 | 1827 | 2475 | — | 0.329/0.713/1.042 | 5547 | 3072/1000 | yes | SUPPORT |
| `sfcsr_t_osp_c_020_02` | TRAIN | concrete | 0.20 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1808 | 1 | 1827 | 2475 | — | 0.337/0.723/1.060 | 9000 | 6525/1000 | yes | SUPPORT |
| `sfcsr_t_osp_c_020_03` | TRAIN | concrete | 0.20 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1504 | 0 | 1523 | 2780 | — | 0.331/0.715/1.046 | 6702 | 3922/1000 | yes | SUPPORT |
| `sfcsr_t_osp_c_020_04` | TRAIN | concrete | 0.20 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1504 | 0 | 1523 | 2780 | — | 0.339/0.725/1.064 | 9000 | 6220/1000 | yes | SUPPORT |
| `sfcsr_t_osp_c_025_01` | TRAIN | concrete | 0.25 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 1 | 1239 | 2475 | — | 0.329/0.713/1.042 | 9000 | 6525/1000 | yes | SUPPORT |
| `sfcsr_t_osp_c_025_02` | TRAIN | concrete | 0.25 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 1 | 1239 | 2475 | — | 0.337/0.723/1.060 | 9000 | 6525/1000 | yes | SUPPORT |
| `sfcsr_t_osp_c_025_03` | TRAIN | concrete | 0.25 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1507 | 0 | 1526 | 2173 | — | 0.331/0.715/1.046 | 9000 | 6827/1000 | yes | SUPPORT |
| `sfcsr_t_osp_c_025_04` | TRAIN | concrete | 0.25 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1507 | 0 | 1526 | 2173 | — | 0.339/0.725/1.064 | 9000 | 6827/1000 | yes | SUPPORT |
| `sfcsr_t_osp_c_030_01` | TRAIN | concrete | 0.30 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 0 | 1246 | 3036 | — | 0.329/0.713/1.042 | 3599 | 563/1000 | no | INVALID |
| `sfcsr_t_osp_c_030_02` | TRAIN | concrete | 0.30 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 0 | 1246 | 3036 | — | 0.337/0.723/1.060 | 3602 | 566/1000 | no | INVALID |
| `sfcsr_t_osp_c_030_03` | TRAIN | concrete | 0.30 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1508 | 0 | 1527 | 2731 | — | 0.331/0.715/1.046 | 3581 | 850/1000 | no | INVALID |
| `sfcsr_t_osp_c_030_04` | TRAIN | concrete | 0.30 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1508 | 0 | 1527 | 2168 | — | 0.339/0.725/1.064 | 9000 | 6832/1000 | yes | SUPPORT |
| `sfcsr_t_osp_m_020_01` | TRAIN | marble | 0.20 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1810 | 1 | 1829 | 2476 | — | 0.329/0.713/1.042 | 5667 | 3191/1000 | yes | SUPPORT |
| `sfcsr_t_osp_m_020_02` | TRAIN | marble | 0.20 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1810 | 1 | 1829 | 2476 | — | 0.337/0.723/1.060 | 5842 | 3366/1000 | yes | SUPPORT |
| `sfcsr_t_osp_m_020_03` | TRAIN | marble | 0.20 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1507 | 0 | 1526 | 2774 | — | 0.331/0.715/1.046 | 7267 | 4493/1000 | yes | SUPPORT |
| `sfcsr_t_osp_m_020_04` | TRAIN | marble | 0.20 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1507 | 0 | 1526 | 2783 | — | 0.339/0.725/1.064 | 9000 | 6217/1000 | yes | SUPPORT |
| `sfcsr_t_osp_m_025_01` | TRAIN | marble | 0.25 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 1 | 1239 | 2474 | — | 0.329/0.713/1.042 | 5172 | 2698/1000 | yes | SUPPORT |
| `sfcsr_t_osp_m_025_02` | TRAIN | marble | 0.25 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 1 | 1239 | 2474 | — | 0.337/0.723/1.060 | 5172 | 2698/1000 | yes | SUPPORT |
| `sfcsr_t_osp_m_025_03` | TRAIN | marble | 0.25 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1509 | 0 | 1528 | 2176 | — | 0.331/0.715/1.046 | 9000 | 6824/1000 | yes | SUPPORT |
| `sfcsr_t_osp_m_025_04` | TRAIN | marble | 0.25 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1509 | 0 | 1528 | 2176 | — | 0.339/0.725/1.064 | 9000 | 6824/1000 | yes | SUPPORT |
| `sfcsr_t_osp_m_030_01` | TRAIN | marble | 0.30 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 0 | 1246 | 3036 | — | 0.329/0.713/1.042 | 4026 | 990/1000 | no | INVALID |
| `sfcsr_t_osp_m_030_02` | TRAIN | marble | 0.30 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 0 | 1246 | 3036 | — | 0.337/0.723/1.060 | 9000 | 5964/1000 | yes | SUPPORT |
| `sfcsr_t_osp_m_030_03` | TRAIN | marble | 0.30 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1511 | 0 | 1530 | 2170 | — | 0.331/0.715/1.046 | 4901 | 2731/1000 | yes | SUPPORT |
| `sfcsr_t_osp_m_030_04` | TRAIN | marble | 0.30 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1511 | 0 | 1530 | 2170 | — | 0.339/0.725/1.064 | 9000 | 6830/1000 | yes | SUPPORT |
| `sfcsr_v_osp_c_020_01` | VALIDATION | concrete | 0.20 | LEFT→BILATERAL | transition_left | RIGHT_SINGLE_SUPPORT | 1808 | 1 | 1827 | 2475 | 6874 | 0.349/0.745/1.094 | 9000 | 6525/1000 | yes | DUAL_HAZARD |
| `sfcsr_v_osp_c_020_02` | VALIDATION | concrete | 0.20 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1504 | 0 | 1523 | 2780 | — | 0.351/0.737/1.088 | 9000 | 6220/1000 | yes | SUPPORT |
| `sfcsr_v_osp_c_025_01` | VALIDATION | concrete | 0.25 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 1 | 1239 | 2475 | — | 0.349/0.745/1.094 | 5562 | 3087/1000 | yes | SUPPORT |
| `sfcsr_v_osp_c_025_02` | VALIDATION | concrete | 0.25 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1507 | 0 | 1526 | 2173 | — | 0.351/0.737/1.088 | 9000 | 6827/1000 | yes | SUPPORT |
| `sfcsr_v_osp_c_030_01` | VALIDATION | concrete | 0.30 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 0 | 1246 | 3036 | — | 0.349/0.745/1.094 | 9000 | 5964/1000 | yes | SUPPORT |
| `sfcsr_v_osp_c_030_02` | VALIDATION | concrete | 0.30 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1508 | 0 | 1527 | 2168 | — | 0.351/0.737/1.088 | 9000 | 6832/1000 | yes | SUPPORT |
| `sfcsr_v_osp_m_020_01` | VALIDATION | marble | 0.20 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1810 | 1 | 1829 | 2476 | — | 0.349/0.745/1.094 | 9000 | 6524/1000 | yes | SUPPORT |
| `sfcsr_v_osp_m_020_02` | VALIDATION | marble | 0.20 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1507 | 0 | 1526 | 2783 | — | 0.351/0.737/1.088 | 9000 | 6217/1000 | yes | SUPPORT |
| `sfcsr_v_osp_m_025_01` | VALIDATION | marble | 0.25 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1220 | 1 | 1239 | 2474 | — | 0.349/0.745/1.094 | 5246 | 2772/1000 | yes | SUPPORT |
| `sfcsr_v_osp_m_025_02` | VALIDATION | marble | 0.25 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1509 | 0 | 1528 | 2176 | — | 0.351/0.737/1.088 | 9000 | 6824/1000 | yes | SUPPORT |
| `sfcsr_v_osp_m_030_01` | VALIDATION | marble | 0.30 | LEFT→LEFT_ONLY | transition_left | RIGHT_SINGLE_SUPPORT | 1227 | 0 | 1246 | 3036 | — | 0.349/0.745/1.094 | 9000 | 5964/1000 | yes | SUPPORT |
| `sfcsr_v_osp_m_030_02` | VALIDATION | marble | 0.30 | RIGHT→RIGHT_ONLY | transition_right | LEFT_SINGLE_SUPPORT | 1511 | 0 | 1530 | 2170 | — | 0.351/0.737/1.088 | 9000 | 6830/1000 | yes | SUPPORT |

The timing columns establish the contact sequence directly: target entry preceded I1, I1 preceded established Support, and the four invalid records then physically fell. The dual record established Support first and Slip later. No row lacked I1 or established Support.

## Five-miss decomposition

| Run | Failure class | Source/speed | Support time | Censor/Slip time | Post-Support duration | Root cause |
|---|---|---|---:|---:|---:|---|
| `sfcsr_t_osp_c_030_01` | POST_SUPPORT_PHYSICAL_FALL | concrete/.30 | 3036 | fall 3599 | 563 | .30 left/topology geometry produced post-Support loss of stability |
| `sfcsr_t_osp_c_030_02` | POST_SUPPORT_PHYSICAL_FALL | concrete/.30 | 3036 | fall 3602 | 566 | repeated Concrete/.30 transition-left instability |
| `sfcsr_t_osp_c_030_03` | POST_SUPPORT_PHYSICAL_FALL | concrete/.30 | 2731 | fall 3581 | 850 | low-start/low-exit right geometry delayed Support and then fell |
| `sfcsr_t_osp_m_030_01` | POST_SUPPORT_PHYSICAL_FALL | marble/.30 | 3036 | fall 4026 | 990 | low-start left geometry was ten milliseconds short because of a real fall |
| `sfcsr_v_osp_c_020_01` | DUAL_SUPPORT_PLUS_SLIP | concrete/.20 | 2475 | Slip 6874 | 6525 | late/long left geometry caused genuine bilateral Support plus later Slip |

The four canonical invalid labels remain `insufficient_post_support_observation`; the failure-class column identifies the physical mechanism without rewriting those labels. There were no `SLIP_BEFORE_SUPPORT`, `SUPPORT_NOT_ESTABLISHED`, `PRETARGET_FAILURE`, or unexplained `OTHER` misses.

## Observation-failure interpretation

| Run | Support | Physical fall/censor | Available | Required | Nominal end | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `sfcsr_t_osp_c_030_01` | 3036 | 3599 | 563 | 1000 | 9000 | post-Support physical fall |
| `sfcsr_t_osp_c_030_02` | 3036 | 3602 | 566 | 1000 | 9000 | post-Support physical fall |
| `sfcsr_t_osp_c_030_03` | 2731 | 3581 | 850 | 1000 | 9000 | post-Support physical fall |
| `sfcsr_t_osp_m_030_01` | 3036 | 4026 | 990 | 1000 | 9000 | post-Support physical fall, marginal by 10 ms |

Support was established 5,964–6,269 ms before the nominal 9-second end in these records. It was therefore neither a nominal horizon shortage nor generally late Support. The actual censor was pulled forward by a physical fall. Extending the horizon would not recover any of the four and is not authorized.

## Source-speed localization

| Source | Speed | Planned | Correct Support | Dual | Invalid | Yield | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| concrete | .20 | 6 | 5 | 1 | 0 | 83.3% | isolated late Dual |
| concrete | .25 | 6 | 6 | 0 | 0 | 100% | stable |
| concrete | .30 | 6 | 3 | 0 | 3 | 50.0% | localized physical failure |
| marble | .20 | 6 | 6 | 0 | 0 | 100% | stable |
| marble | .25 | 6 | 6 | 0 | 0 | 100% | stable |
| marble | .30 | 6 | 5 | 0 | 1 | 83.3% | low-start left corner excluded |

The broad description `HIGH_SPEED_ORDINARY_SUPPORT_LOCALIZED` is true for the four fall-invalid records but incomplete. Concrete/.30 contributed three of them; both transition-left geometries failed, while the low right geometry failed and the higher right geometry succeeded. Marble/.30 separated the same low left point from a higher-start successful left point. This is source-speed/contact-sequence and side/topology-conditioned geometry, terminating in a post-Support fall.

## Cross-study comparison

| Evidence | N | Correct Support | Dual | Invalid | Yield |
|---|---:|---:|---:|---:|---:|
| Previous stable ordinary-Support evidence (two independent studies) | 48 | 48 | 0 | 0 | 100% |
| Latest FACTOR_TRAIN | 24 | 20 | 0 | 4 | 83.3% |
| Latest FACTOR_VALIDATION | 12 | 11 | 1 | 0 | 91.7% |

The prior stable evidence is the redesigned study plus the mild-recalibrated study, 24/24 each. Across their 48 controls, start was `.336–.349`, width `.710–.746`, exit `1.047–1.094`, entry `1220–1810`, I1 `1239–1829`, Support `2168–3036`, and valid post-Support duration `2698–6832`. Both studies deliberately used Concrete/.30 right-only and obtained 8/8 correct Support in that cell; Marble/.30 retained both sides and obtained 8/8.

| Physical contrast | Previously reliable | Latest failure | Finding |
|---|---|---|---|
| Concrete/.30 left | excluded; right-only 8/8 | transition-left 0/2 in latest corpus and 0/2 in preceding factor corpus | repeatable topology/cell exclusion |
| Concrete/.30 right | start at least .338 in stable family; Support 2168 | .331/.715/1.046 failed at Support 2731; .339/.725/1.064 succeeded at 2168 | low right geometry is unstable; stable higher-start right remains |
| Marble/.30 left | .336–.349 stable family | .329/.713/1.042 fell; .337/.723/1.060 succeeded | exclude low-start corner, retain evidenced left strip |
| Concrete/.20 left | stable joint combinations inside prior family | .349/.745/1.094 became bilateral then slipped at 6874 | exclude this late/long corner; no global low-speed redesign |
| Timing/semantics | entry→I1→Support with ≥1000 ms afterward | same ordering; four physical falls shortened observation | no label, threshold, or horizon defect |

## Geometry comparison

| Outcome group | Start | Width | Exit | Entry | I1 | Support time | Post-event duration |
|---|---|---|---|---|---|---|---|
| correct Support (latest, N=31) | .329–.351 | .713–.745 | 1.042–1.094 | 1220–1810 | 1239–1829 | 2168–3036 | 2698–6832 |
| observation-invalid (latest, N=4) | .329–.337 | .713–.723 | 1.042–1.060 | 1227–1508 | 1246–1527 | 2731–3036 | 563–990 |
| Dual/Slip (latest, N=1) | .349 | .745 | 1.094 | 1808 | 1827 | 2475 | 6525 |
| other invalid (latest, N=0) | — | — | — | — | — | — | — |
| previous stable correct Support (N=48) | .336–.349 | .710–.746 | 1.047–1.094 | 1220–1810 | 1239–1829 | 2168–3036 | 2698–6832 |

Ranges alone do not define the correction: source, speed, side, topology, and the joint start/width/exit combination are coupled. In particular, Concrete/.30 left remains excluded even where individual coordinates overlap a global range.

## Root physical cause

The deterministic classification is `MULTIFACTOR_ORDINARY_SUPPORT_INSTABILITY`, with `SOURCE_SPEED_CONTACT_SEQUENCE_LOCALIZED` as the design mechanism and `POST_SUPPORT_PHYSICAL_FALL_LOCALIZED` as the terminal mechanism. Confidence is high for the post-Support-fall interpretation and high for the Concrete/.30 transition-left exclusion because it repeated 0/4 across the two latest factor-conditioned corpora while two prior independent stable studies gave Concrete/.30 right-only 8/8.

No evidence supports a generator bug, labeling bug, I1 bug, established-Support bug, duration-threshold bug, or shared delayed-Support defect. The smallest correction is a source-speed-conditioned choice of already evidenced ordinary geometry and sides.

## Pilot decision

`NO_NEW_PILOT_REQUIRED`

The saved record provides 48/48 independently stable ordinary controls, repeated Concrete/.30 left failure, a successful Concrete/.30 right comparator, and same-cell Marble/.30 left separation. A pilot would repeat an already identified contrast without changing the correction. New pilot simulations and batches are both zero.

## Stable ordinary-Support envelope

This is a simulator/controller-specific envelope, not a universal real-world threshold.

- Applicability: Concrete and Marble at `.20`, `.25`, and `.30 m/s`.
- Default sides: LEFT and RIGHT. Exception: Concrete/.30 is RIGHT_ONLY. Marble/.30 LEFT is retained only in the stable higher-start strip.
- Contact expectation: LEFT uses `transition_left` with `RIGHT_SINGLE_SUPPORT`; RIGHT uses `transition_right` with `LEFT_SINGLE_SUPPORT`.
- Global evidenced geometry: start `.336–.351 m`, width `.710–.746 m`, exit `1.047–1.097 m`.
- Concrete/.20 LEFT: constrain to start `.336–.345`, width `.710–.739`, exit `1.047–1.084`; exclude the `.349/.745/1.094` dual corner.
- Marble/.30 LEFT: constrain to start `.336–.341`, width `.710–.733`, exit `1.047–1.074`; exclude start below `.336`.
- Timing: entry `1220–1810 ms`, I1 `1239–1829 ms`, established Support `2168–3036 ms`.
- Mechanics: unchanged `0.004 m / 50,000 N/m / 1,000 Ns/m` entry reference and `0.040 m / 7,000 N/m / 374 Ns/m` lateral exit.
- Observation: 9,000 ms nominal horizon and at least 1,000 ms after established Support.
- Expected yield: saved stable evidence 100%; future frozen gates retain margin at TRAIN ≥22/24 and VALIDATION ≥11/12 rather than equaling the optimistic point estimate.

Known exclusions are Concrete/.30 transition-left, Concrete/.30 low-start/low-exit right geometry, Marble/.30 low-start left geometry, and the Concrete/.20 late/long left dual corner.

## Support semantics

I1 definition, established-Support threshold, Support duration, Slip threshold and duration, side semantics, Support label, post-Support observation requirement, and ordinary-versus-delayed meaning are unchanged. Physical outcome remains authoritative over design intent.

## Sand physical status

Sand remains physically successful and was not recalibrated: mild 107/108 strict, moderate 33/36 strict, all six source-speed gates passed, adverse TRAIN/VALIDATION 42/42 and 21/21, comparison 29/30 and 15/15, and the Concrete/.25 exception was 18/18. The future design preserves these mechanics, topology rules, counts, and factor manifolds.

Accumulated anti-contamination exclusions leave no legal fresh right-mild point inside the literal prior coordinate box, and fewer than the required 18 Concrete/.25 mild points. The future matrix therefore uses only a ≤2 mm boundary realization for fresh IDs/signatures while retaining the successful joint Sand family; this is a freshness constraint, not a reopened Sand hypothesis. The already frozen physical generation gates remain the arbiter before any future training.

## Delayed-Support physical status

Delayed Support remains physically successful: TRAIN 11/12 met ≥11, VALIDATION 6/6 met ≥5, and all Slip/Dual contamination gates passed. Its exact LEFT_ONLY, `transition_left`, expected `RIGHT_SINGLE_SUPPORT`, start `.324–.332`, width `.825–.833`, exit `1.153–1.165` domain is preserved. No shared defect was found and no delayed-Support recalibration occurred.

## Future complete corpus design

The frozen but ungenerated dataset ID is `sand_factor_conditioned_development_controls_recalibrated_20260903`.

| Component | Latest failed design | Future design | Change | Reason |
|---|---:|---:|---|---|
| Mild Sand | 108 | 108 | same physics/manifolds; fresh boundary-safe realizations | preserve successful Sand and satisfy accumulated anti-overlap |
| Moderate Sand | 36 | 36 | same reduced transition-left diagnostic family; fresh coordinates | preserve successful reduced moderate design |
| ordinary Support | 36 | 36 | source-speed profiles; Concrete/.30 right-only; unstable joint corners excluded | apply smallest evidenced physical correction without shrinking denominator |
| delayed Support | 18 | 18 | exact same envelope; fresh coordinates | preserve successful recalibration |
| TRAIN | 132 | 132 | no count or role change | preserve statistical structure |
| VALIDATION | 66 | 66 | no count or role change | preserve independent frozen evaluation role |
| source-speed | six cells | six cells | same coverage; ordinary profiles can differ by cell | encode demonstrated source-speed interaction |
| factor manifolds | TRAIN 42 adverse/30 comparison; VALIDATION 21/15 | same | no causal-factor change | preserve successful factor-conditioned hypothesis |

The canonical generator now accepts explicit source-speed profile sets while preserving the legacy shared-profile form for frozen historical designs. The new matrix expands deterministically to 198 unique run IDs and 198 unique scenario signatures: TRAIN 132, VALIDATION 66, Mild 108, Moderate 36, ordinary Support 36, and delayed Support 18. It was not generated in this milestone.

## Future generation gates

Before any future run, the design freezes:

- complete execution 198 and objective-valid at least 180;
- TRAIN/VALIDATION strict Sand at least 91/44, Mild 68/34, Moderate 22/10, ordinary Support 22/11, and delayed Support 11/5;
- strict Sand per source-speed cell at least 14/7;
- Mild adverse manifold at least 39/19 and comparison manifold at least 28/14;
- designed Sand Slip+Dual at most 6;
- ordinary Support Slip+Dual at most 2 overall and 1 per split;
- delayed Support Slip+Dual at most 2 overall and 1 per split;
- pretarget fall at most 3 and target-following fall/censor at most 8;
- valid physical-signature uniqueness at least .80;
- all historical, failed-corpus, pilot, cross-split, exact, forbidden-near, and run-ID overlaps zero;
- model output, replacement, backfill, rerun, and adaptive change counts zero.

Any failure stops before training. The old TRAIN ordinary-Support 22/24 gate was not relaxed or retroactively reinterpreted.

## Anti-contamination

The future matrix protects every historical manifest already named by the prior design plus the failed support-recalibrated 198-run manifest. Validation reports historical exact overlap 0, forbidden-near overlap 0, run-ID reuse 0, cross-split exact overlap 0, and cross-split forbidden-near overlap 0. Failed corpora and calibration pilots remain physical evidence only and contribute no training rows.

## Model boundary

V1 inference, V2 inference, Terrain inference, Hazard probability, 80D analysis, training, optimizer steps, checkpoint writes, normalizer fits, HNM, threshold search, persistence search, architecture search, and sensor fusion are all zero. Model outputs were not consulted. Historical HOLDOUT access is zero.

## Review verdict

`ORDINARY_SUPPORT_PHYSICAL_RECALIBRATION_READY`

## Recommended next milestone

`SAND_FACTOR_CONDITIONED_DEVELOPMENT_CONTROLS_RECALIBRATED_GENERATION`

Do not start it in this milestone. That task should generate exactly one fully pre-frozen corpus, run zero model inference, apply every physical gate, freeze the dataset only if all gates pass, and stop before training on any failure.

## Deployment parallelization

No change was made to `/d/shin/Infineon_FastReflex_E84`. Independent deployment engineering may continue with `model_v2_anchor_refined_gru20_20260902` only as `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`.

## Counters

| Counter | Value |
|---|---:|
| saved ordinary-Support records reviewed | 36 |
| new pilot simulations | 0 |
| pilot batches | 0 |
| failed-corpus backfill / rerun | 0 / 0 |
| V1 / V2 / Terrain inference | 0 / 0 / 0 |
| Hazard probability / 80D reconstruction | 0 / 0 |
| training / optimizer / checkpoint writes | 0 / 0 / 0 |
| HNM / normalizer fits | 0 / 0 |
| threshold / persistence / architecture search | 0 / 0 / 0 |
| sensor fusion | 0 |
| historical HOLDOUT access | 0 |

## Tests

Targeted tests cover the failed 198-run freeze, deterministic 36-run and five-miss ledgers, unchanged Support semantics, zero backfill/rerun/model output, no-pilot physical-only decision, successful Sand and delayed-Support preservation, future 198-run anti-overlap, HOLDOUT protection, executable ordinary-contamination gates, and review/readiness hashes. The targeted old/new contract suite passed 11/11. Full pytest passed 195 with one pre-existing skip; `compileall`, selective Ruff (`E9,F63,F7,F82`), configured full Ruff, and `git diff --check` all passed.

## Hashes

| Artifact | SHA-256 |
|---|---|
| prior report | `9ff9159abfa4718300ea08edd8a9c585d62a2f9425d3be26481b484b2d4200cf` |
| prior execution config | `c4bde7adc1d4a917fe78da9b2f470c1f6155af63a06f314972609c64ef5aade4` |
| prior design | `b1f09effbb90313ef8c883db8e0ef376c5f4eb8e83ec66d8ed5fe52ca95ef775` |
| reviewed manifest | `bda6961f79525df237d440086057aea71a03afc5134a49c50f5e4d4a2193be67` |
| reviewed physical audit | `f4861dc18456da28f76caf38257de03abef90e184879378bcbe844301490f9a3` |
| reviewed dataset freeze | `fb575566574ef87bdc6ca8c161cb770c6d16e530b18fda3df0b65d213ad59922` |
| future design | `b18be44668f1d0e2c07b6a127c7fe626d42636a002ad023e48721af7c2443fb5` |
| review config | `2e01a24771de138442e30afce3967f63dbbbc45e06e3b8057d8fb87f96c81ee5` |
| canonical factor-conditioned implementation | `8dc85c501149c9deaf8e6b48cae19ef334144c64679e48c03a1eb2729550b50c` |
| ordinary 36-run ledger | `dc5b9c2149ff5ad04686d09c6d99bce9c6067cf124eb0063e6cbd68c6aeaa696` |
| five-miss ledger | `f5dce7aeb55818a21e0b439c659652b4f765392df7532bd6ee896a71b67c95a3` |
| readiness contract | `3518bb4b8cdeec8b47b59f9ed2bd8ccaadc02243c27fe17f54f04648a2c88deb` |
