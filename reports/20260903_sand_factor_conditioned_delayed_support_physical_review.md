# Sand Factor-Conditioned Delayed-Support Physical Review

Date: 2026-09-03 (Asia/Seoul)

Verdict: `DELAYED_SUPPORT_PHYSICAL_RECALIBRATION_READY`

Recommended next milestone: `SAND_FACTOR_CONDITIONED_DEVELOPMENT_SUPPORT_RECALIBRATED_GENERATION`

## Starting state

The scientific starting commit was `481f07ebba8905c07fe95f83075b53e106edb909`, equal to `origin/main`, with a clean tracked worktree. The latest 198-run dataset and its physical audit were hash-exact. It remains frozen failed physical evidence and was not repaired, backfilled, rerun, or used for training.

## Scientific boundary

This was a saved-evidence-first physical review followed by one pre-frozen, model-blind calibration batch. No Hazard, Terrain, V1, or V2 output was computed. No 80D feature was reconstructed. Support/Slip/I1 semantics, model threshold, persistence, architecture, sensors, and the 9 s simulation horizon were unchanged. The permanently consumed Generalization HOLDOUT was not opened.

## Latest delayed-Support failure recap

The latest corpus planned 18 delayed-Support controls and produced 12 correct Support, five genuine `DUAL_HAZARD`, and one invalid control. TRAIN was 9/12 against a frozen minimum of 10; VALIDATION was 3/6 against a minimum of 5. Those were the only two failed gates in the 53/55 generation ledger. The Sand domain itself passed all physical source-speed and manifold gates.

All 18 controls used the same mechanics: `staged_lateral_deformable`, `transition_left`, moderate exit compliance, and designed `LEFT_ONLY` Support. The entry half used the medial reference profile (4 mm travel, 50,000 N/m stiffness, 1,000 Ns/m damping); the exit half used the lateral moderate profile (40 mm travel, 7,000 N/m stiffness, 374 Ns/m damping). The canonical 10 mm/20-sample Support-spread rule and 50 mm/3-sample Slip-drift rule were not changed.

## 18-run physical ledger

Times are 1 kHz sample indices and therefore milliseconds from simulation start. “Observation valid” evaluates the existing post-Support contract; it does not relabel a genuine dual Hazard.

| Run | Split | Source | Speed | Side | Phase | Support established | Slip | Dual | Observation valid | Final outcome |
|---|---|---|---:|---|---|---:|---:|---|---|---|
| sfcr_t_dsp_c_020_01 | TRAIN | Concrete | 0.20 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3661 | 6928 | yes | yes | DUAL_HAZARD |
| sfcr_t_dsp_c_020_02 | TRAIN | Concrete | 0.20 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3661 | 7126 | yes | yes | DUAL_HAZARD |
| sfcr_t_dsp_c_025_01 | TRAIN | Concrete | 0.25 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3067 | — | no | yes | SUPPORT |
| sfcr_t_dsp_c_025_02 | TRAIN | Concrete | 0.25 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3067 | — | no | yes | SUPPORT |
| sfcr_t_dsp_c_030_01 | TRAIN | Concrete | 0.30 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3050 | — | no | yes | SUPPORT |
| sfcr_t_dsp_c_030_02 | TRAIN | Concrete | 0.30 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3053 | — | no | yes | SUPPORT |
| sfcr_t_dsp_m_020_01 | TRAIN | Marble | 0.20 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3662 | 6923 | yes | yes | DUAL_HAZARD |
| sfcr_t_dsp_m_020_02 | TRAIN | Marble | 0.20 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3663 | — | no | yes | SUPPORT |
| sfcr_t_dsp_m_025_01 | TRAIN | Marble | 0.25 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3068 | — | no | yes | SUPPORT |
| sfcr_t_dsp_m_025_02 | TRAIN | Marble | 0.25 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3068 | — | no | yes | SUPPORT |
| sfcr_t_dsp_m_030_01 | TRAIN | Marble | 0.30 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3051 | — | no | yes | SUPPORT |
| sfcr_t_dsp_m_030_02 | TRAIN | Marble | 0.30 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3058 | — | no | yes | SUPPORT |
| sfcr_v_dsp_c_020_01 | VALIDATION | Concrete | 0.20 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 4262 | — | no | no | INVALID |
| sfcr_v_dsp_c_025_01 | VALIDATION | Concrete | 0.25 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3067 | — | no | yes | SUPPORT |
| sfcr_v_dsp_c_030_01 | VALIDATION | Concrete | 0.30 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3106 | 3924 | yes | yes | DUAL_HAZARD |
| sfcr_v_dsp_m_020_01 | VALIDATION | Marble | 0.20 | LEFT_ONLY | DOUBLE_SUPPORT | 4259 | 2113 | yes | yes | DUAL_HAZARD |
| sfcr_v_dsp_m_025_01 | VALIDATION | Marble | 0.25 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3068 | — | no | yes | SUPPORT |
| sfcr_v_dsp_m_030_01 | VALIDATION | Marble | 0.30 | LEFT_ONLY | RIGHT_SINGLE_SUPPORT | 3128 | — | no | yes | SUPPORT |

The detailed physical ledger below reports clean target touchdowns before I1; delayed first-target time; I1; established Support and Slip; start/width/exit; peak Support spread; peak drift; normalized peak load derivative; fall/censor; and post-Support observation. Normalized load redistribution was 1.0 in every run. Peak spread is the physically audited load/spread magnitude, not a model feature.

| Run | Clean TD | Target | I1 | Support | Slip | Start/width/exit m | Peak spread m | Peak drift m | Norm. peak load derivative | Fall/censor | Post-Support ms |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|
| sfcr_t_dsp_c_020_01 | 4 | 1808 | 3607 | 3661 | 6928 | .326/.785/1.111 | .040096 | .056391 | 5061.3 | —/9000 | 5339 |
| sfcr_t_dsp_c_020_02 | 4 | 1808 | 3607 | 3661 | 7126 | .338/.801/1.139 | .040146 | .055422 | 5767.1 | —/9000 | 5339 |
| sfcr_t_dsp_c_025_01 | 4 | 1220 | 3011 | 3067 | — | .326/.785/1.111 | .040156 | .015524 | 6302.2 | —/9000 | 5933 |
| sfcr_t_dsp_c_025_02 | 4 | 1220 | 3011 | 3067 | — | .338/.801/1.139 | .040118 | .015524 | 4070.6 | 5238/5238 | 2171 |
| sfcr_t_dsp_c_030_01 | 2 | 1227 | 2421 | 3050 | — | .326/.785/1.111 | .040118 | .015533 | 5179.5 | —/9000 | 5950 |
| sfcr_t_dsp_c_030_02 | 2 | 1227 | 2421 | 3053 | — | .338/.801/1.139 | .040117 | .016121 | 3871.2 | —/9000 | 5947 |
| sfcr_t_dsp_m_020_01 | 4 | 1810 | 3609 | 3662 | 6923 | .326/.785/1.111 | .040130 | .052087 | 6118.1 | —/9000 | 5338 |
| sfcr_t_dsp_m_020_02 | 4 | 1810 | 3609 | 3663 | — | .338/.801/1.139 | .040121 | .037695 | 3661.0 | —/9000 | 5337 |
| sfcr_t_dsp_m_025_01 | 4 | 1220 | 3012 | 3068 | — | .326/.785/1.111 | .040155 | .015468 | 6580.5 | 5795/5795 | 2727 |
| sfcr_t_dsp_m_025_02 | 4 | 1220 | 3012 | 3068 | — | .338/.801/1.139 | .040148 | .015468 | 3274.1 | 5238/5238 | 2170 |
| sfcr_t_dsp_m_030_01 | 2 | 1227 | 2422 | 3051 | — | .326/.785/1.111 | .040141 | .016582 | 5281.4 | —/9000 | 5949 |
| sfcr_t_dsp_m_030_02 | 2 | 1227 | 2422 | 3058 | — | .338/.801/1.139 | .040165 | .015349 | 3449.6 | —/9000 | 5942 |
| sfcr_v_dsp_c_020_01 | 1 | 2454 | 2479 | 4262 | — | .354/.805/1.159 | .040218 | .037606 | 2624.0 | 5118/5118 | 856 |
| sfcr_v_dsp_c_025_01 | 4 | 1220 | 3011 | 3067 | — | .354/.805/1.159 | .040340 | .045719 | 8340.0 | —/9000 | 5933 |
| sfcr_v_dsp_c_030_01 | 2 | 1227 | 2421 | 3106 | 3924 | .354/.805/1.159 | .040092 | .053516 | 8040.8 | —/9000 | 5894 |
| sfcr_v_dsp_m_020_01 | 3 | 1780 | 2983 | 4259 | 2113 | .354/.805/1.159 | .040094 | .069888 | 6179.4 | 6961/6961 | 2702 |
| sfcr_v_dsp_m_025_01 | 4 | 1220 | 3012 | 3068 | — | .354/.805/1.159 | .040109 | .018531 | 5035.9 | —/9000 | 5932 |
| sfcr_v_dsp_m_030_01 | 2 | 1227 | 2422 | 3128 | — | .354/.805/1.159 | .040286 | .045358 | 15520.8 | —/9000 | 5872 |

Later falls after more than 1,000 ms of valid Support observation do not invalidate Support. Accordingly, the four correct controls with falls at 5238–5795 remain correct under the unchanged contract.

## Six-miss decomposition

The frozen robustness rule calls a Slip margin at least 5 mm above the canonical 50 mm threshold robust and a positive margin below 5 mm borderline. This diagnostic never changes the label: every threshold-satisfying event remains Slip/Dual.

| Run | Failure class | Support time | Slip time | Gap | Physical cause | Factor localization |
|---|---|---:|---:|---:|---|---|
| sfcr_t_dsp_c_020_01 | DUAL_SUPPORT_PLUS_SLIP | 3661 | 6928 | +3267 | Support first, then robust left Slip (+6.391 mm) | Concrete/.20, short geometry, late multi-contact Slip |
| sfcr_t_dsp_c_020_02 | DUAL_SUPPORT_PLUS_SLIP | 3661 | 7126 | +3465 | Support first, then robust right Slip (+5.422 mm) | Concrete/.20, short geometry, cross-side late Slip |
| sfcr_t_dsp_m_020_01 | DUAL_SUPPORT_PLUS_SLIP | 3662 | 6923 | +3261 | Support first, then borderline left Slip (+2.087 mm) | Marble/.20, short geometry, late multi-contact Slip |
| sfcr_v_dsp_c_020_01 | OBSERVATION_INVALID | 4262 | — | — | Support followed by physical fall/censor after only 856 ms | Concrete/.20, late geometry/contact instability |
| sfcr_v_dsp_c_030_01 | DUAL_SUPPORT_PLUS_SLIP | 3106 | 3924 | +818 | Support first, then borderline left Slip (+3.516 mm) | Concrete/.30, late geometry |
| sfcr_v_dsp_m_020_01 | DUAL_SUPPORT_PLUS_SLIP | 4259 | 2113 | −2146 | Robust left Slip first (+19.888 mm), then Support; DOUBLE_SUPPORT precontact; later fall | Marble/.20, late geometry/contact sequence |

There were no `OTHER` misses.

## Dual-Hazard interpretation

Four duals established the intended left Support first and later accumulated real Slip; one established real Slip 2.146 s before Support. Support side was correct in all five. Four Slip events were left-only; one was right-only after left Support, proving that the contamination is not merely a mistaken Support-side label.

The correct controls do not reproduce established Slip. Their peak drift stayed below 50 mm, although the late Concrete/.25 (45.719 mm) and Marble/.30 (45.358 mm) controls show that the old late geometry approached the boundary. The five dual labels therefore remain genuine. Calibrating geometry away from this physical Slip regime is a physical scenario-design correction made without model information; it is not a relaxation of model evaluation.

## Observation-invalid interpretation

`sfcr_v_dsp_c_020_01` is category B: a later physical fall/censor, not a nominal horizon shortage. Support established at 4262 ms and the physical fall censored the trace at 5118 ms, leaving 856 ms instead of the required 1,000 ms. Without that fall, the 9 s horizon had 4,738 ms remaining. The duration and observation contract therefore stay unchanged.

## Cross-study delayed-Support comparison

Only canonical physical metadata was used. The previous stable row is the eight delayed controls from the mild-recalibrated study; it contains four Concrete and four Marble runs at 0.25 m/s. No historical model output was read.

| Evidence | N | Correct Support | Dual | Invalid | Yield |
|---|---:|---:|---:|---:|---:|
| previous stable delayed-Support evidence | 8 | 8 | 0 | 0 | 100.0% |
| latest TRAIN | 12 | 9 | 3 | 0 | 75.0% |
| latest VALIDATION | 6 | 3 | 2 | 1 | 50.0% |
| pilot batch 1 | 24 | 23 | 0 | 1 | 95.8% |
| pilot batch 2 | 0 | 0 | 0 | 0 | not required |

The prior eight and all six latest 0.25 m/s controls were correct despite different starts, widths, sources, and later falls. Mechanics and semantics were identical. This isolates the new failure from a general Support implementation failure and from the successful recalibrated Sand geometry.

## Factor localization

The classification rule was frozen before pilot execution: `STABLE` requires at least 75% correct Support with at most one Slip/Dual per source-speed cell; `MARGINAL` covers at least 50% or one borderline Slip; `UNSTABLE` covers below 50% or repeated Slip/Dual. Rows that lack a physically meaningful comparator are marked non-localizing rather than used to invent symmetry.

| Factor | Saved/pilot evidence | Class | Interpretation |
|---|---|---|---|
| Concrete/.20, latest | 0/3 Support; 2 dual; 1 invalid | UNSTABLE | Both short profiles slipped late; late profile fell |
| Marble/.20, latest | 1/3 Support; 2 dual | UNSTABLE | Source changes severity but not the .20 instability |
| Concrete/.25, latest | 3/3 Support | STABLE | Matches prior stable .25 evidence |
| Marble/.25, latest | 3/3 Support | STABLE | Matches prior stable .25 evidence |
| Concrete/.30, latest | 2/3 Support; 1 borderline dual | MARGINAL | Failure only at late .354/.805 geometry |
| Marble/.30, latest | 3/3 Support | STABLE | Late point remained below Slip threshold |
| LEFT_ONLY Support side | intended and actual Support side matched in all 18 | NON_LOCALIZING | No right delayed semantic exists to compare; side mismatch is not the failure |
| RIGHT_SINGLE_SUPPORT precontact, latest | 12/17 correct; four dual; one invalid | MARGINAL | Necessary common phase, but speed/geometry separate success from failure |
| DOUBLE_SUPPORT precontact, latest | 0/1; Slip first, later fall | UNSTABLE | Contact-sequence exception at late Marble/.20 |
| Short .20 geometry, latest | 1/4 Support; three late duals | UNSTABLE | Too close to a late real-Slip regime |
| Late .354/.805 geometry, latest | .20 invalid/dual; .30 one borderline dual | UNSTABLE | Source-speed/contact sequence interaction |
| Intermediate pilot strip, all cells | 23/24 Support; zero Slip/Dual | STABLE | Only excluded earliest Concrete/.20 point fell before I1/Support |
| Pilot source-speed cells | C/.20 3/4; every other cell 4/4 | STABLE | Meets frozen per-cell rule without dropping a source-speed cell |

This is `MULTIFACTOR_PHYSICAL_INSTABILITY`: speed-conditioned start/width/exit placement changes target entry and subsequent contact sequence, while source terrain modulates whether the trajectory crosses real Slip. Geometry, speed, source, and contact sequence interact; no single Support parameter or phase explains every miss.

## Root physical cause

The unchanged staged Support mechanics reliably establish Support, but the old short and late control placements put some source-speed trajectories too close to a real Slip/fall regime. At 0.20 m/s, short controls commonly established Support and slipped 3.26–3.47 s later; late controls produced a phase/contact disruption, Slip-first dual, or early censor. At 0.30 m/s, only the late Concrete point crossed the threshold. The intermediate pilot exit strip removed all Slip/Dual contamination across all six cells while retaining the same Support physics.

There is no evidence of a generator bug: event clocks, sides, thresholds, censor handling, and deterministic reruns of the saved audit are coherent. There is also no evidence of coupling to the newly recalibrated benign-Sand profiles; the delayed controls use their own staged support pattern and failed while all Sand gates passed.

## Pilot decision

`DELAYED_SUPPORT_CALIBRATION_REQUIRED`

Saved evidence established the broad mechanism but did not define a robust common domain for 0.20 and 0.30 m/s. A fresh physical pilot was therefore necessary.

## Pilot results

The complete 24-run matrix was frozen before run 1. It used fresh IDs and signatures, covered Concrete/Marble × 0.20/0.25/0.30 with four profiles per cell, and had historical exact/forbidden-near overlap 0. All outcomes were retained. There was no within-batch adaptation, replacement, backfill, rerun, or model inference.

| Batch | Question | N | Correct Support | Dual | Slip-only | Invalid | Yield | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Can speed-conditioned fresh geometry retain left delayed Support with ≥21/24 correct, ≥3/4 per cell, ≤2 Slip/Dual, and ≤2 invalid? | 24 | 23 | 0 | 0 | 1 | 95.8% | PASS |
| 2 | — | 0 | 0 | 0 | 0 | 0 | — | NOT_REQUIRED |

The sole invalid point was Concrete/.20 at start .318 m, width .817 m, exit 1.135 m. Target contact began at 1802 ms, a physical fall occurred at 2235 ms, and neither I1 nor Support established. It is excluded rather than relabeled. All other pilot runs had RIGHT_SINGLE_SUPPORT precontact, correct LEFT_ONLY Support, two to four clean target touchdowns before I1, at least 5.337 s nominal post-Support horizon (or over 2.153 s before a later physical fall), and no Slip.

## Stable delayed-Support envelope

This is a simulator/frozen-policy design envelope, not a universal real-world Support threshold.

- Applicability: all six Concrete/Marble × 0.20/0.25/0.30 cells.
- Side/topology: LEFT_ONLY, `transition_left`, `staged_lateral_deformable`; no unsupported right-side symmetry claim.
- Fresh future start: .324–.332 m.
- Fresh future width: .825–.833 m.
- Fresh future exit: 1.153–1.165 m.
- Expected phase: RIGHT_SINGLE_SUPPORT 20 ms before first valid target contact.
- Observed entry: 1220–1810 ms; I1: 2421–3609 ms; Support: 3050–3663 ms.
- Mechanics: unchanged 4 mm/50,000 N/m/1,000 Ns/m entry reference and 40 mm/7,000 N/m/374 Ns/m lateral exit compliance.
- Horizon: 9,000 ms; required post-Support observation: 1,000 ms.
- Expected physical yield: pilot 95.8%; future frozen gate 16/18 = 88.9%.
- Exclusions: short .20 late-Slip placements, late .20 phase/contact disruption, late Concrete/.30 near-Slip placement, and the pilot fall corner at .318/.817/1.135.

## Support semantics

Unchanged. I1 is still 20 consecutive positive spread-derivative samples within the loaded contact episode; established Support still uses the 10 mm spread threshold for 20 samples; delayed Support still requires two clean target touchdowns before I1, I1 no later than Support, no Slip, LEFT_ONLY, and at least 1,000 ms of post-Support observation. Slip remains 50 mm tangential drift for three samples. No physical outcome was reinterpreted.

## Sand physical status

The successful recalibrated Sand domain is preserved conceptually: mild 108/108, moderate 35/36, no pretarget fall, no post-target fall/censor, and all source-speed/manifold gates passed. No new Sand calibration was run. The future corpus uses fresh Sand signatures within the same proven envelopes because actual runs from the failed 198-run corpus cannot be reused.

## Future complete corpus design

The future corpus is fully designed but not generated.

| Component | Previous failed design | New design | Reason |
|---|---|---|---|
| Sand mild | 108; successful envelope | 108 fresh signatures in same envelope | Preserve 108/108 success without reusing runs |
| Sand moderate | 36; 35/36 strict | 36 fresh, reduced comparator | Preserve successful reduced tier |
| ordinary Support | 36 | 36 fresh left/right controls | Preserve established ordinary Support coverage |
| delayed Support | 18; 12 correct, 5 dual, 1 invalid | 18 fresh controls in calibrated strip | Improve physics, not denominator |
| onset timing | old Support 3050–4262 ms | expected 3050–3663 ms | Exclude late unstable onset/contact cases |
| geometry | .326/.785, .338/.801, .354/.805 | start .324–.332, width .825–.833, exit 1.153–1.165 | Stay inside the interpolated all-cell stable strip and away from Slip/fall corners |
| support-side coverage | delayed LEFT_ONLY | delayed LEFT_ONLY; ordinary left/right retained | Preserve canonical meaningful semantics |
| source-speed coverage | all six cells | all six cells | No difficult cell removed |
| expected delayed yield | 66.7% observed | 95.8% pilot; ≥88.9% gate | Physical margin over frozen gate |

Dataset identity and counts:

- Dataset ID: `sand_factor_conditioned_development_support_recalibrated_20260903`
- FACTOR_TRAIN: 132
- FACTOR_VALIDATION: 66
- Mild: 108
- Moderate: 36
- Ordinary Support: 36
- Delayed Support: 18
- Total: 198

## Future generation gates

Before any future run, the design freezes: completion 198; overall objective-valid ≥180; TRAIN/VALIDATION strict Sand ≥91/44; mild ≥68/34; moderate ≥22/10; strict Sand per source-speed ≥14/7; adverse manifold ≥39/19; comparison manifold ≥28/14; ordinary Support ≥22/11; delayed Support ≥11/5; delayed Slip+Dual ≤2 overall and ≤1 per split; pretarget fall ≤3; post-target fall/censor ≤8; physical uniqueness ≥.80; every exact/near/run-ID overlap count 0; and model output, replacement, backfill, and rerun counts 0. Failure stops before training.

## Anti-contamination

The future matrix expands deterministically to 198 unique IDs and 198 unique scenario signatures. Against all protected historical manifests, including the failed 198-run corpus and the new pilot, exact overlap, forbidden-near overlap, and run-ID reuse are each 0. Cross-split exact and forbidden-near overlap are 0. Pilot runs and failed-corpus runs remain prohibited from future training.

## Model boundary

V1 inference, V2 inference, Terrain inference, Hazard probability calculation, 80D analysis, optimizer steps, checkpoint writes, normalizer fits, HNM, threshold/persistence search, architecture search, and sensor-fusion experiments were all zero.

## Historical scientific status

Unchanged:

- `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`
- `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS: NOT_YET_MODEL-TESTED`

The physical redesign does not constitute a model-level test of the factor-conditioned hypothesis.

## Review verdict

`DELAYED_SUPPORT_PHYSICAL_RECALIBRATION_READY`

The failure mechanism is understood, the unchanged Support semantics are reproducibly realized in a stable all-cell physical envelope, and a wholly fresh 198-run future design is frozen without contamination.

## Recommended next milestone

`SAND_FACTOR_CONDITIONED_DEVELOPMENT_SUPPORT_RECALIBRATED_GENERATION`

It was not started. That milestone must generate the complete corpus exactly once, enforce all physical gates, freeze it only on pass, and stop before training otherwise.

## Deployment parallelization

Independent E84 deployment engineering remains authorized. This milestone did not read or modify `/d/shin/Infineon_FastReflex_E84`; the research V2 remains only `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL` there.

## Counters

| Counter | Value |
|---|---:|
| saved delayed-Support records reviewed | 18 |
| new pilot simulations | 24 |
| pilot batches | 1 |
| future complete-corpus simulations | 0 |
| V1 inference | 0 |
| V2 inference | 0 |
| Terrain inference | 0 |
| Hazard probability | 0 |
| optimizer steps / checkpoint writes | 0 / 0 |
| normalizer fits / HNM | 0 / 0 |
| threshold / persistence / architecture searches | 0 / 0 / 0 |
| sensor fusion | 0 |
| old HOLDOUT reads / inference / visualization | 0 / 0 / 0 |
| failed-corpus backfill / rerun | 0 / 0 |

## Frozen hashes

- Review config: `9c495f9c0fe024e5889b1beaf11ae7d76c853ba85e21191a6d731763a5ff5238`
- Pilot config: `5288e9fd011b6db03624f814b2164058ecc31126e6a9cc723db2e0dac134250d`
- Pilot manifest: `6c69a4aaecdb5e09b095976109a7a88b50802a45de7aaf02e29ef04257468850`
- Pilot dataset-freeze file: `05e84b1f256fb48b7fa05016ddaaab50c9979a3aa2b154c668f9bf57c1c92920`
- Future design config: `b1f09effbb90313ef8c883db8e0ef376c5f4eb8e83ec66d8ed5fe52ca95ef775`
- Future scenario matrix: `6ed2c0c23ae036fd0bc8f523b3f254429cb1c835507d9859dadcbf82af6bd8b2`
- Future scenario signatures: `0944705e3cb18ff78f4edf68573fbf56477ae9fc7cf7576a2894145549feb4be`
- Saved-evidence review: `b92f2fa07dc98e9a3bf417cc637e1af715f275c0c590fc556e5b77b843a7e2ae`
- Pilot readiness: `501ade6bdd6dc66a35714786f7457209d0a27abd5cef4aa60d7ff095a6a36221`
- Future-design readiness: `55844373171e31787d2bd0e60e8e409dbc9e358d3da01d2e6fc7bd2a626f140f`
