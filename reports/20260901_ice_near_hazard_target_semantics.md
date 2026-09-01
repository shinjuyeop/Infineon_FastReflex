# Ice Near-Hazard Target Semantics Study

## 1. Purpose

This milestone resolves how sub-threshold Ice states should be represented in a future Model V2 dataset without changing the established Slip oracle or Model V1. The result is a development-only physical precursor category, not a retroactive change to Hazard scoring:

```text
ICE_PRECURSOR_CANDIDATE
= exact Ice contact AND loaded contact
  AND touchdown-anchor tangential drift in [30, 50) mm
  AND established Slip not yet active
```

The study recommendation is `ICE_PHYSICAL_PRECURSOR_SUPPORTED`. The primary verdict is `ICE_NEAR_HAZARD_TARGET_SEMANTICS_RESOLVED`.

## 2. Model V1 preservation

Starting `HEAD` and `origin/main` were both `0a076dbaf218b55af1079bec4747e280be161efe` (`Audit generalization failure modes`), branch `main`, with a clean tracked worktree. The protected baseline remained restorable before and after the study.

| Protected item | Frozen identity | Result |
|---|---|---|
| Hazard V1 freeze | `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2` | exact |
| Hazard feature schema | `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb` | exact |
| Hazard normalizer | `610a91c24a594a3cd1701b07549f4f3c4b7f4cb04eb354f9d41378214a6800a9` | exact |
| Hazard checkpoints | `e6bada49…`, `b04877dc…`, `b6c782bd…` | exact |
| Terrain normalizer | `2c551181fbd23080c0e3008be8484bd0ee4139e594041ba6e46ab5b36ffb05de` | exact |
| Terrain checkpoints | `21b0d122…`, `de6a55d3…`, `465803f4…` | exact |

The runtime contract is still Pelvis IMU6 → causal 80D → `[20,80]` → one-layer GRU hidden 32, three-seed mean, threshold `0.99`, persistence `5 ms`, 11,010 parameters. Optimizer steps, checkpoint writes, normalizer fits, HNM rounds, threshold/persistence searches, and architecture searches were all zero. Hazard/Terrain V1 files modified: zero.

## 3. Evidence boundary

Only development evidence was opened. The episode pool keeps source provenance and is not a formal generalization-performance pool.

| Evidence | Waveforms opened | Role |
|---|---:|---|
| Generalization VALIDATION Ice | 16 | authorized existing episode census and later V1 overlay |
| Ice-resolution pilots | 48 | authorized existing episode census and later V1 overlay |
| Scenario-calibration Ice pilots | 0 of 42 | manifest inspected; canonical episode arrays absent, so no boundaries were imputed |
| Unified TRAIN Ice | 0 of 38 | manifest inspected; exact contact/episode arrays absent, so no boundaries were imputed |
| Fresh semantics DISCOVERY | 32 | physical discovery |
| Fresh semantics CONFIRMATION | 16 | one-shot physical confirmation after interpretation freeze |
| Generalization HOLDOUT | 0 of 36 | sealed; open count 0 |
| Unified HOLDOUT | 0 | not reopened; no new inference |

The combined semantics-only pool is explicitly named `DEVELOPMENT_PHYSICAL_EPISODE_POOL`. Generalization HOLDOUT waveform access, inference, and visualization remained zero.

## 4. Frozen established-Slip semantics

Established Slip is unchanged: touchdown-anchor tangential foot drift `>=50 mm` sustained for `3 ms`, aggregated across either foot. Its anchor, contact-episode reset behavior, threshold, persistence, and any-foot aggregation were not modified.

`ICE_PRECURSOR_CANDIDATE` is a secondary development interpretation below that boundary. It neither declares established Slip nor replaces the historical binary label. The 100/250/500/1000 ms horizons are descriptive progression windows only.

## 5. Previous near-hazard observations

The exact failure-mode observations were reproduced before any new simulation. All 36 authorized Generalization VALIDATION replays matched the prior artifact; delayed Ice remained 3/6 correct and 3/6 premature.

| Delayed Ice run | Result | Drift / velocity at Reflex | Episode max | Reflex → release |
|---|---|---:|---:|---:|
| `ghr_ocd_v_c020` | premature | 34.532 mm / 0.901 m/s | 44.924 mm | 11 ms |
| `ghr_ocd_v_c025` | correct | 28.208 mm / 0.775 m/s | 57.374 mm | 29 ms |
| `ghr_ocd_v_c030` | premature | 33.306 mm / 0.823 m/s | 41.947 mm | 10 ms |
| `ghr_ocd_v_m020` | correct | 34.983 mm / 0.905 m/s | 102.220 mm | 50 ms |
| `ghr_ocd_v_m025` | correct | 27.455 mm / 0.731 m/s | 134.766 mm | 75 ms |
| `ghr_ocd_v_m030` | premature | 29.464 mm / 0.787 m/s | 41.558 mm | 14 ms |

Correct and premature states overlap at the causal observation: correct drift/velocity `27.45–34.98 mm / 0.731–0.905 m/s`; premature `29.46–34.53 mm / 0.787–0.901 m/s`. The premature episodes terminate 10–14 ms after Reflex and peak at 41.56–44.92 mm.

`ghr_ibc_v_c020` also reproduced exactly: first `p>=0.99` sample 2466, Reflex sample 2470, ensemble maximum `0.999956`, 13 consecutive threshold milliseconds, unanimous seeds, 34.844 mm drift and 0.944 m/s velocity at Reflex, and 43.569 mm episode maximum.

## 6. Episode-level methodology

The primary unit is a per-foot maximal contiguous physical-contact episode carrying exact Ice contact. A state landmark additionally requires loaded contact and finite drift/velocity. Each row records run/source/speed, foot, target touchdown when present, episode boundaries and duration, maximum drift, velocity, drift derivative, 20/30/40/50 mm landmarks, phase, same/next/later Slip, all fixed horizons, benign release, censor, and fall.

An episode contributes to every drift band it reaches. The landmark is the first loaded exact-Ice sample entering that band before same-foot established Slip. Outcomes are prioritized as `SAME_EPISODE_SLIP`, `NEXT_EPISODE_SLIP`, `LATER_SLIP` within 1 s after release, `BENIGN_RELEASE` with a fully observed 1 s no-Slip follow-up, then `CENSORED`. Censored is never benign. “Next” means the first chronological target-Ice episode starting at or after current release across either foot.

The Gitignored full table `development_physical_episode_pool.csv` contains 8,179 episode rows. Adjacent episodes within one run can be correlated, especially during contact chatter, so episode counts are descriptive and run-balanced summaries are reported alongside them.

## 7. Existing development episode census

| Source | Authorized Ice runs | Eligible episodes | Reach 30 mm | Same / next / later | Benign / censored |
|---|---:|---:|---:|---:|---:|
| Generalization VALIDATION | 16 | 1,634 | 189 | 102 / 13 / 57 | 4 / 13 |
| Ice-resolution pilots | 48 | 2,585 | 300 | 188 / 14 / 65 | 5 / 28 |
| Scenario-calibration pilots | 42 | 0 | — | — | — |
| Unified TRAIN | 38 | 0 | — | — | — |
| Existing usable total | 64 | 4,219 | 489 | 290 / 27 / 122 | 9 / 41 |

At 40–50 mm, existing evidence contained 371 exposed episodes: 290 same, 9 next, 49 later, 2 benign, and 21 censored. The nine fully observed benign releases at 30–40 mm were sparse relative to 448 future-Slip episodes and did not cover source/speed conditions evenly. That pre-simulation census justified the fresh targeted corpus.

## 8. Fresh semantics corpus

`ice_near_hazard_semantics_20260901` contains 48 model-blind MuJoCo runs: 32 `SEMANTICS_DISCOVERY` and 16 `SEMANTICS_CONFIRMATION`, no TRAIN split. The grid was declared in config SHA-256 `29ce97d2dcdb8f783518b956d36522370f9059c4914a64a4d5f7cdbf1f4446e0` before simulation.

```text
source:      concrete, marble
speed:       0.20, 0.25, 0.30 m/s
patch start: 0.326, 0.334 m
patch width: 0.232, 0.248, 0.730, 0.760 m
mechanics:   current full-width Ice transition only
```

All 48 predeclared signatures ran once. There was no adaptive replacement or backfill. Every run produced 8,000 samples; fresh NPZ total size is 15,865,753 bytes. Generation loaded no model output and recorded no confirmation outcome. Manifest file SHA-256 is `6a472d4b26e724355c6f2d88d0668c4f91c925037c63ad8e1838f83a53e15759`.

## 9. Signature and split integrity

Grid/split freeze canonical SHA-256 is `63145b29dbe8a57f27d1dea1720066385b1edbcf2d28af9a8cff3b62ef03d6a6`. Fresh internal duplicates, DISCOVERY/CONFIRMATION overlap, and signature overlap with Unified 256, Generalization 72, calibration 78, Ice-resolution 48, `fall_risk_dense_20260828`, and `reflex_event_20260828` were all zero.

The split formula was frozen before simulation and independent of outcome/model output. Both sources, all three speeds, both starts, and all four widths occur on both sides of the split. Confirmation remained unopened through discovery and interpretation/rule freeze, then opened exactly once. Generalization HOLDOUT open count remained zero.

## 10. Drift-band progression

Fresh DISCOVERY progression:

| Drift band | Episodes | Same | Next | Slip <=100 ms | <=250 ms | <=500 ms | <=1000 ms | Benign | Censored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–20 mm | 2,455 | 135 | 107 | 888 | 1,547 | 1,850 | 1,984 | 58 | 420 |
| 20–30 mm | 352 | 135 | 21 | 174 | 250 | 289 | 305 | 8 | 38 |
| 30–40 mm | 239 | 135 | 15 | 159 | 191 | 209 | 215 | 6 | 17 |
| 40–50 mm | 181 | 135 | 5 | 139 | 156 | 164 | 168 | 3 | 9 |
| >=50 mm territory | 139 | 135 | 0 | 131 | 136 | 136 | 136 | 0 | 3 |

Fresh CONFIRMATION progression:

| Drift band | Episodes | Same | Next | Slip <=100 ms | <=250 ms | <=500 ms | <=1000 ms | Benign | Censored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–20 mm | 1,505 | 92 | 76 | 553 | 1,013 | 1,201 | 1,276 | 29 | 202 |
| 20–30 mm | 242 | 92 | 15 | 123 | 181 | 207 | 218 | 6 | 18 |
| 30–40 mm | 169 | 92 | 12 | 108 | 137 | 149 | 157 | 5 | 7 |
| 40–50 mm | 122 | 92 | 5 | 94 | 108 | 113 | 117 | 4 | 1 |
| >=50 mm territory | 95 | 92 | 0 | 85 | 92 | 94 | 94 | 1 | 0 |

The monotonic pattern replicated. DISCOVERY run-balanced Slip-within-500 ms rose from `62.9%` at 20–30 to `74.2%` at 30–40 and `81.6%` at 40–50. CONFIRMATION was `85.5%`, `83.2%`, and `91.8%`, respectively. These are simulator-development descriptions, not calibrated real-world probabilities.

## 11. Velocity-conditioned progression

Values are median `[IQR]`; range, in m/s. Drift derivative uses the same format.

| Split / band | Outcome | N | Tangential velocity | Drift derivative |
|---|---|---:|---:|---:|
| Discovery 30–40 | future Slip | 216 | 0.949 `[0.721–1.280]`; 0.170–3.052 | 0.909 `[0.695–1.257]`; 0.155–3.177 |
| Discovery 30–40 | benign release | 6 | 0.293 `[0.142–0.441]`; 0.076–0.441 | 0.305 `[0.153–0.434]`; 0.099–0.434 |
| Discovery 40–50 | future Slip | 169 | 0.968 `[0.711–1.276]`; 0.133–3.191 | 1.027 `[0.666–1.285]`; 0.148–3.311 |
| Discovery 40–50 | benign release | 3 | 0.108 `[0.108–0.108]`; 0.108–0.108 | 0.080 `[0.080–0.080]`; 0.080–0.080 |
| Confirmation 30–40 | future Slip | 157 | 0.936 `[0.654–1.286]`; 0.138–3.052 | 0.911 `[0.648–1.256]`; 0.104–3.177 |
| Confirmation 30–40 | benign release | 5 | 0.594 `[0.587–0.853]`; 0.169–2.041 | 0.587 `[0.575–0.785]`; 0.194–1.920 |
| Confirmation 40–50 | future Slip | 117 | 0.968 `[0.698–1.249]`; 0.090–3.191 | 1.005 `[0.666–1.285]`; 0.067–3.311 |
| Confirmation 40–50 | benign release | 4 | 0.700 `[0.537–0.765]`; 0.217–0.791 | 0.684 `[0.545–0.733]`; 0.261–0.750 |

Future-Slip medians are higher, but confirmation ranges overlap materially in both bands. No tangential-velocity or derivative threshold was optimized or added to the frozen rule.

## 12. Contact-phase dependence

Confirmation at 30–40 mm:

| Post-hoc phase | Episodes | Same | Next | Later | Benign | Censored | Slip <=500 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Loading | 17 | 4 | 1 | 9 | 1 | 2 | 11 |
| Stance | 101 | 84 | 1 | 11 | 4 | 1 | 93 |
| Terminal <=20 ms | 51 | 4 | 10 | 33 | 0 | 4 | 45 |

At 40–50 mm, stance was 78 same-Slip episodes out of 80, while terminal states were 11 same, 5 next, 17 later, 3 benign, and 1 censored out of 37. Phase therefore changes *when* Slip appears: mid-stance states usually progress in the same episode, while terminal states more often release and predict the next/later low-friction episode. It does not justify using future release position as a runtime input.

## 13. Same-contact vs next-contact Slip

| Split / near-hazard band | Same-contact | Next-contact | Later <=1 s | Benign no Slip <=1 s | Censored |
|---|---:|---:|---:|---:|---:|
| Discovery 30–40 mm | 135 | 15 | 66 | 6 | 17 |
| Discovery 40–50 mm | 135 | 5 | 29 | 3 | 9 |
| Confirmation 30–40 mm | 92 | 12 | 53 | 5 | 7 |
| Confirmation 40–50 mm | 92 | 5 | 20 | 4 | 1 |

The terminal pattern is not merely an immediate incipient same-contact Slip signal. It also identifies a broader unstable low-friction walking condition whose current contact can release before next/later established Slip. This directly explains why delayed-family alerts could be physically meaningful while still being premature under the original event window.

## 14. Benign escape behavior

Fresh 30–40 mm benign escape was 6/239 in DISCOVERY and 5/169 in CONFIRMATION. At 40–50 mm it was 3/181 and 4/122. The fully observed benign examples prevent relabeling every sub-threshold sample as established Hazard, but they are a small minority relative to future Slip.

At the run level, 42/48 fresh runs contained a 30–50 mm candidate. From the earliest candidate, 37 established Slip within 1 s and 5 were censored before the full horizon; no run had a fully observed 1 s benign outcome from its *earliest* candidate. Episode-level benign releases can occur inside runs that become unstable later, which is precisely why outcome provenance and run-balanced reporting are required.

## 15. Source-terrain dependence

Confirmation 30–40 mm progression:

| Source | Episodes / runs | Same | Next | Later | Benign | Censored |
|---|---:|---:|---:|---:|---:|---:|
| Concrete | 86 / 6 | 44 | 6 | 26 | 5 | 5 |
| Marble | 83 / 7 | 48 | 6 | 27 | 0 | 2 |

Both sources reproduce high progression. Benign evidence is concentrated in Concrete—19/20 pooled 30–40 mm benign releases—so the study does not claim source-invariant benign calibration. The Marble denominator supports precursor progression, not precise benign escape probability.

## 16. Speed dependence

Confirmation 30–40 mm progression:

| Speed | Episodes / runs | Same | Next | Later | Benign | Censored |
|---|---:|---:|---:|---:|---:|---:|
| 0.20 m/s | 69 / 5 | 37 | 6 | 20 | 5 | 1 |
| 0.25 m/s | 29 / 5 | 15 | 3 | 7 | 0 | 4 |
| 0.30 m/s | 71 / 3 | 40 | 3 | 26 | 0 | 2 |

All three stable calibrated speeds reproduce the physical progression. Run N is small, particularly 0.30 m/s, so this is not a speed-specific probability calibration and does not replace the already-justified need for balanced Model V2 speed data.

## 17. Physical precursor interpretation

DISCOVERY selected Option B, `ICE_SLIP_PRECURSOR_SUPPORTED`, before confirmation. The physical rule SHA-256 is `d4225e278f173a660182c2993060c7ca0bf4a7b52253cfb4032732871bad8ef4`.

The rule uses only exact Ice identity, loaded contact, and the predeclared 30–50 mm drift interval. It has no V1 probability, no optimized velocity threshold, no future phase variable, and no status as established Slip. The interpretation freeze canonical SHA-256 is `8074583d9d2e983e9e586e39b53f50b27b728c814626a4841ca30a60a3615227`.

Severity context supports the distinction. Of 42 fresh runs with a candidate, 37 slipped within 1 s: peak drift median 103.6 mm, Slip-active union duration median 220 ms, and peak Pelvis gyro norm median 2.216 rad/s. Actual Slip side was bilateral in 26, left in 9, right in 2; five runs were censored with no established Slip. These diagnostics do not redefine Slip.

## 18. Confirmation result

Confirmation was opened once after the interpretation and rule were frozen. No rule element changed afterward.

- 30–40 mm: future Slip `157/169 = 92.9%`, benign `5/169 = 3.0%`, censored `7/169 = 4.1%`.
- 40–50 mm: future Slip `117/122 = 95.9%`, benign `4/122 = 3.3%`, censored `1/122 = 0.8%`.
- Run-balanced Slip within 500 ms: `83.2%` and `91.8%`.
- Both source terrains and all three speeds retained the direction of progression.

Confirmation therefore supports the frozen precursor interpretation. Confirmation canonical SHA-256 is `991235411af5f29c651e5615f40a050be9f67f43c608098461be75e763b4606b`; open count is exactly 1.

## 19. Post-freeze Model V1 overlay

Only after confirmation did read-only Model V1 replay run on 112 authorized Ice trajectories: 48 fresh, 48 Ice-resolution pilots, and 16 Generalization VALIDATION runs.

| Overlay diagnostic | Count |
|---|---:|
| Runs with an early V1 alert | 76 / 112 |
| Early alert onsets | 76 |
| Onsets inside frozen 30–50 mm precursor region | 27 |
| Matches to fully observed benign-release episodes | 0 |
| Matches to censored near-hazard episodes | 3 |
| Future-Slip candidate episodes without a matching V1 onset | 788 |

The missed-episode count is inflated by correlated contact chatter and is not a performance denominator. More importantly, only 27/76 alert onsets align with the frozen candidate, so V1 is not reinterpreted as a calibrated precursor detector.

For `ghr_ibc_v_c020`, Reflex occurred 4 ms after the episode entered 30 mm and inside the frozen candidate region. Because no Slip occurred and the full future horizon was not observable before fall/censor, the new physical classification is `AMBIGUOUS_CENSORED_NEAR_HAZARD_STATE`. Its official historical score remains `NO_HAZARD` false positive.

## 20. Historical-score preservation

`ZERO_RETRAIN_GENERALIZATION_NOT_SUPPORTED` is unchanged. No prior report, row, denominator, event window, or score was rewritten. The original validation contract had no accepted Ice precursor region; explaining an alert physically does not make the old frozen score disappear.

Generalization HOLDOUT 36 remains sealed for a future Model V2. Any future precursor-aware secondary metric must be explicitly separated from the original frozen HOLDOUT metric; the sealed set must not be silently relabeled.

## 21. Target semantics decision

| Candidate interpretation | Evidence for | Evidence against | Decision |
|---|---|---|---|
| All 30–50 mm states are established positive | 90–96% future progression | nonzero benign escape; 50 mm/3 ms oracle is frozen | Reject |
| All `<50 mm` states are hard negative | preserves original binary label | would mine a strongly predictive physical region as negative | Reject |
| Drift + velocity precursor | future medians higher | confirmation distributions overlap; no defensible threshold | Do not threshold |
| Contact-phase-aware precursor | phase separates same vs next/later timing | terminal state uses post-hoc release proximity; not causal input | Diagnostic modifier only |
| Separate precursor class / acceptable-early region | replicated discovery/confirmation progression with low benign overlap | simulator-only, correlated episodes, sparse benign/source coverage | Select |
| Keep established binary semantics unchanged | preserves historical oracle and scoring | does not alone express acceptable early warning | Select together with secondary precursor |

Exactly one recommendation is made:

```text
ICE_PHYSICAL_PRECURSOR_SUPPORTED
```

## 22. Model V2 data implication

Do not relabel the established Slip oracle. In future `MODEL_V2_DATASET_DESIGN`:

- keep the binary established-Hazard target at frozen 50 mm / 3 ms;
- tag loaded exact-Ice `[30,50) mm` states as `ICE_PRECURSOR_CANDIDATE`, not ordinary hard negatives;
- use the tag as a separate precursor head/target or an explicitly accepted-early region with its own metric;
- retain `BENIGN_RELEASE` examples as hard negatives for the *precursor* target, with source/speed provenance;
- exclude or down-weight `CENSORED` examples for outcome-supervised precursor loss rather than calling them benign;
- preserve same/next/later outcome and contact-phase metadata so training does not collapse the broader unstable condition into immediate same-contact Slip.

This study did not create a Model V2 training dataset or train a model. The already-justified staged Sand benign negatives, right-only Support positives, speed diversity, delayed/multi-contact Ice, and Ice benign data remain future design inputs.

## 23. Architecture/sensor implication

The architecture conclusion is unchanged:

```text
ARCHITECTURE_CHANGE_NOT_YET_JUSTIFIED
```

Target ambiguity does not justify LSTM, longer history, threshold/persistence retuning, or a new model family. The provisional sensor candidate remains Pelvis IMU6 + left FSR4 = 10 physical channels. It is neither expanded nor frozen by this study.

## 24. Limitations

- Evidence is limited to the current deterministic G1 policy, MuJoCo mechanics, Ice material, and calibrated geometry/speeds.
- Episodes within a run are correlated; contact chatter creates many short episodes. Run-balanced summaries reduce but do not eliminate this dependence.
- All 42 fresh runs whose earliest contact reached the candidate either slipped within 1 s (37) or censored (5); fully stable run-level candidate negatives remain absent.
- Fully observed 30–50 mm benign episodes are few and concentrated in Concrete. Marble benign escape probability is unresolved.
- Velocity shows overlap and has no frozen production threshold.
- The candidate uses privileged exact terrain/contact/drift quantities and is a target-design reference, not a deployable runtime rule.
- A single `>=50 mm` sample does not necessarily satisfy 3 ms persistence; the established oracle remains the sustained clock.
- No hardware realism, recovery controller, E84 resource, quantization, or HIL claim is made.

## 25. Verdict

```text
ICE_NEAR_HAZARD_TARGET_SEMANTICS_RESOLVED
ICE_PHYSICAL_PRECURSOR_SUPPORTED
```

The evidence is sufficient to make a defensible Model V2 target recommendation while preserving the established Slip oracle. Config SHA-256 is `29ce97d2dcdb8f783518b956d36522370f9059c4914a64a4d5f7cdbf1f4446e0`; study-summary canonical SHA-256 is `2b57c980638bf19ba8f76d1f8136bc6b3c8cc511f79b233f7d94b2d05da9a0f0`; integrity canonical SHA-256 is `6537e37b70a4e15ce84824051446269c71cc951a7f659db0668765520cbd8cd0`.

Verification completed with `71 passed, 1 skipped`, `python -m compileall src scripts tests` PASS, critical Ruff `E9,F63,F7,F82` PASS, fail-closed confirmation-reopen PASS, and four representative simulation regressions PASS (`uhr_ice_h_c20`, `uhr_sand_h_c20`, `uhr_sand_b_c20`, `uhr_hard_n_c20`). The standalone `pytest` executable was unavailable in this environment, so the full suite used the equivalent `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` invocation.

## 26. Recommended next milestone

```text
MODEL_V2_DATASET_DESIGN
```

That milestone should integrate the precursor-aware Ice target, Ice benign/delayed/multi-contact coverage, staged Sand benign hard negatives, right-only Support positives, and 0.20/0.30 m/s diversity. It is not started here.
