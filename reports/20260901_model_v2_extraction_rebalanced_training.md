# Model V2 Extraction-Rebalanced Training

## 1. Purpose

This milestone executed the frozen extraction-only intervention from `MODEL_V2_EXTRACTION_REBALANCE_DESIGN_READY`. It trained exactly one separate candidate, `model_v2_extraction_rebalanced_gru20_20260901`, to test whether balanced causal coverage around delayed-Support I1, interval midpoint, and established Support resolves the remaining Marble delayed-Support failure without damaging solved behavior.

The training completed, the candidate was frozen before one-shot `V2_VALIDATION`, and the target behavior improved. However, confirmed no-hazard and speed-Sand specificity regressed. The controlling intervention verdict is therefore `V2_EXTRACTION_REBALANCE_NOT_EFFECTIVE`.

## 2. Starting state

- Starting `HEAD`: `ef7b2ebc797d9317f461b8f3a4cc991cea71524f`
- Starting `origin/main`: `ef7b2ebc797d9317f461b8f3a4cc991cea71524f`
- Starting parity: exact
- Starting tracked worktree: clean
- Baseline V2 verdict: `MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED`
- Extraction design verdict: `MODEL_V2_EXTRACTION_REBALANCE_DESIGN_READY`
- Generalization HOLDOUT guard count: `0`

## 3. V1 and baseline V2 preservation

The historical V1 and baseline data-only V2 remain independently restorable. The new checkpoint path does not overlap either protected family.

| Protected object | SHA-256 after training | Status |
|---|---|---|
| V1 candidate freeze | `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2` | exact |
| V1 normalizer | `610a91c24a594a3cd1701b07549f4f3c4b7f4cb04eb354f9d41378214a6800a9` | exact |
| V1 seed 20260828 | `e6bada49beb2b68c2c33bb9a59a1b3a8dd0f556f174cafce12f43efd2d22d588` | exact |
| V1 seed 20260829 | `b04877dc08290a34077ae2deb753085d840640ee8da2bd920d010eb2ce8c2506` | exact |
| V1 seed 20260830 | `b6c782bdfb3789ae7af785ec2b02260a8ae54179d7c531f87e27e0e35301a753` | exact |
| Baseline V2 candidate freeze | `edb16b7d96fb38e680e36dcca5ecd7b1c1682ba410b94d21f40c2a38d4ed8725` | exact |
| Baseline V2 normalizer | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` | exact |
| Baseline V2 seed 20260828 | `dd6c8581161963265d4323b8316f01367e359357673f0596faaa2a27051771c8` | exact |
| Baseline V2 seed 20260829 | `8e6709da112845840aae0094dd997fad4ee7f9d8256a2ee0fc5e9a0df3b724a0` | exact |
| Baseline V2 seed 20260830 | `811f486c1bd47f91a854fdbd004b8408a5f00bfaa83a22ce91608de1d3b54c42` | exact |

Fresh replay of both protected ensembles on the authorized 96-run validation split reproduced their frozen result JSON exactly.

## 4. Dataset/extraction freeze verification

All manifest rows were rehashed against their NPZ files after training; mismatch count was zero for every protected corpus.

| Corpus | Runs | Manifest SHA-256 | NPZ aggregate SHA-256 | Mismatches |
|---|---:|---|---|---:|
| Unified | 256 | `d023384aaaac22076a2be7c2fa242acac91a548d82102640c39868990cb0e9d6` | `555a44bc00ff46dd51716fc5cd151932c0fa1a5360b1ad3ddf3cc5250ea6f4aa` | 0 |
| Model V2 | 412 | `7a036d3485bb19a3570a3dd4a41cc990375028f7e153e85525bd4bf19cff8b25` | `5a8dfd54d1c08413dc6fb18d957269a5fbfea8cf526e4f29c78510886186e11c` | 0 |
| Generalization | 72 | `72f5dd300bb5be78b22b9a8c1ad9788cdf45e7cda9b4f46d79eb290140146f53` | `0964e521b4625466e72a42aa07dd4b57285adad744400fdd962d8c3efac792ad` | 0 |
| Ice semantics | 48 | `6a472d4b26e724355c6f2d88d0668c4f91c925037c63ad8e1838f83a53e15759` | `be6761cfa744014d291481bf3c3ef293b9dfdae026f2bee60809e43aa813e2f5` | 0 |

The Model V2 dataset-freeze SHA is `fd81b647051b63e7b76ab72c6dedd616cf06d752fcbc1db31ba02f8aebe68744`. No dataset was generated or modified.

## 5. One-variable intervention

The only intentional variable was delayed-Support positive endpoint extraction. Every eligible `DELAYED_SAND_SUPPORT_ONSET` TRAIN run received up to 15 causal endpoints: five at I1 (`0..4 ms`), five around `floor((I1 + Support) / 2)` (`-2..2 ms`), and five at established Support (`0..4 ms`). All 18 eligible runs were assigned to fit, symmetrically across Concrete and Marble.

Held fixed: the 442-run effective TRAIN corpus, Slip endpoints, ordinary-Support endpoints, initial negatives, forbidden masks, HNM policy, normalizer, 80D causal features, GRU20 architecture, seeds, optimizer, loss, batch size, epoch/patience limits, threshold `0.99`, and persistence `5 ms`.

- Execution config SHA: `98c3e8327fd20dfd13b57a20b91f53ddd0fce4a4aed62266cfbad0459926fcd6`
- Design config SHA: `280abbfd9da9cce948259492497090c887d238e691e0c4d8b2a0a2d52921c040`
- Extraction design SHA: `58a3949c29c8aa313cedc345dc8fa5eb0222cb85d94d31dac488adff69aed29b`
- Extraction policy SHA: `3c7ce82ed905d932ec8f17d69d7e5edb5d79ee7602ba95ffe2a53d2407142cd2`

## 6. Pre-training extraction parity

The CLI dry run completed before optimizer step 1 and froze pretraining-audit SHA `f8e725b6aacc2709d2f9d699e4ed894bc60e6e7626a8f05e9db04c11bdd77951`.

| Contract | Actual | Expected | Status |
|---|---|---|---|
| Positive endpoint SHA | `498f5d1f4419e3bfa72fc2f9649326db26f00e7a9523d9b3ecc8032436a3e0bb` | same | exact |
| Negative endpoint SHA | `392c1fda06953135bfed9a97c7cabb3915d4c50fdf9ff11b2ac8e6550448936c` | same | exact |
| Mask SHA | `32bae2d81b05709771545b375dd6cffb95d2dfc17c2808a7adf3a7f70174c35a` | same | exact |
| Fit positives | 2,590 | 2,590 | exact |
| Slip positives | 1,680 | 1,680 | exact |
| Ordinary-Support positives | 640 | 640 | exact |
| Delayed-Support positives | 270 | 270 | exact |
| Fit negatives | 25,585 | 25,585 | exact |
| Monitor positives / negatives | 598 / 6,624 | 598 / 6,624 | exact |

All eight contradiction counters were zero: future-Slip precursor ordinary negative, censored precursor negative, I1/positive negative, post-censor/fall, pre-I1 delayed-Support positive, future-feature leakage, short delayed neighborhood, and persistence neighborhood shorter than 5 ms. Masked samples remained future-Slip `41,479`, censored precursor `1,734`, and I1-positive `68,388`.

## 7. Delayed-Support exposure

All 18 eligible delayed-Support TRAIN runs were represented in fit. Concrete contributed 9 runs and 135 endpoints; Marble contributed 9 runs and 135 endpoints. Monitor received no delayed-Support endpoints from this intervention, while the other 598 baseline monitor positives remained unchanged.

The materialized round-0 tensor independently matched the audit: fit shape `(28175, 20, 80)` with class counts `25,585 / 2,590`, and monitor shape `(7222, 20, 80)` with counts `6,624 / 598`.

## 8. Negative/Slip preservation

The fit Slip count remained `1,680`, ordinary Support remained `640`, and the negative endpoint identity SHA remained `392c1fda…8936c`. The proposed positive identity SHA matched the predeclared design exactly. No positive endpoint intersected an ordinary negative, no selected negative entered a precursor/I1/censor mask, and no causal window crossed its endpoint boundary.

## 9. Normalizer reuse

The baseline V2 normalizer was reused directly from `artifacts/runs/20260901_model_v2_data_only_training/normalization/gru_history20.json`.

- SHA-256 before and after: `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a`
- Logical normalizer fits: `0`
- New normalizer artifact writes: `0`
- Status: unchanged

## 10. Training protocol

The candidate used the fixed 11,010-parameter, one-layer, hidden-32, unidirectional GRU with input `[20,80]` and binary output. Seeds were `20260828`, `20260829`, and `20260830`; optimization was Adam at `0.001`, inverse-frequency weighted cross entropy, batch size `128`, maximum `40` epochs, patience `6`, deterministic shuffled DataLoader, and no gradient clipping. Selection used internal TRAIN-monitor validation loss. No seed was selected or discarded.

Training counters: optimizer steps `51,089`; checkpoint writes `12`; HNM rounds `3`; normalizer fits `0`; threshold, persistence, architecture, and seed searches `0`; new simulation runs `0`.

## 11. Seed training

| Seed | Round | Fit positive | Fit negative | Delayed Support exposure | HNM selected after round | Best/completed epoch | Optimizer steps | Checkpoint SHA |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20260828 | 0 | 2,590 | 25,585 | 270 | 5,304 | 3/9 | 1,989 | `04fec4340c67e13a3cd66529172753fc6a17aaaf0e5db464a2d6f614c6a79d52` |
| 20260829 | 0 | 2,590 | 25,585 | 270 | 5,304 | 8/14 | 3,094 | `0af867f21cecd8181ca52f39b8ad705ed614a94fca39418477ff68bc9a9ef414` |
| 20260830 | 0 | 2,590 | 25,585 | 270 | 5,304 | 12/18 | 3,978 | `ab9df5a4d10ad327483480c0ea6982d024069d8ae2292b708bfc355ad51db65c` |
| 20260828 | 1 | 2,590 | 29,632 | 270 | 5,304 | 11/17 | 4,284 | `0994278c084bee7bf0416cdd1c64df528bbbf0f0e8986026e88e6ec9ea8a585f` |
| 20260829 | 1 | 2,590 | 29,632 | 270 | 5,304 | 14/20 | 5,040 | `6c5f22f36640cc671f9c7b38186a3973b13f4de4deabd07a438eb051ee4478d9` |
| 20260830 | 1 | 2,590 | 29,632 | 270 | 5,304 | 15/21 | 5,292 | `f666d101af7321f3ffb6de1b3b4ddf816d0f005ad1cb7d889d6fa58fda1924c3` |
| 20260828 | 2 | 2,590 | 33,716 | 270 | 5,304 | 6/12 | 3,408 | `daee6ba43a7cfa7f65c45c27457a9db803bf0223823be8e1f855032b6569a5c5` |
| 20260829 | 2 | 2,590 | 33,716 | 270 | 5,304 | 8/14 | 3,976 | `41644bd03170d31ec8c27672359133377f8590641369a5b424107d8f59d75755` |
| 20260830 | 2 | 2,590 | 33,716 | 270 | 5,304 | 10/16 | 4,544 | `171de3c708c2411682d8ea73418455c27866cbeab000c53cd52417e111464e84` |
| 20260828 | 3 | 2,590 | 37,828 | 270 | — | 8/14 | 4,424 | `f52a4c86eb29f4a13e263ad1d5da83e55a722bcdfb03784337973471efe89883` |
| 20260829 | 3 | 2,590 | 37,828 | 270 | — | 12/18 | 5,688 | `c21e8fe4daa7a0877c1a403147f4e685ab250ae97e2b00a0188904651c5d648e` |
| 20260830 | 3 | 2,590 | 37,828 | 270 | — | 11/17 | 5,372 | `5d88c027c3a70afbfcaf6c6eb6f961346180649bb075f4cf1bef30958a85f223` |

## 12. HNM round 1

All 442 TRAIN runs were eligible and contributed. The round selected 5,304 windows: Concrete `2,652`, Marble `2,652`. Major family counts were hard-ground normal `456+216`, baseline immediate Ice Slip `432`, Ice Slip `456`, Ice benign `288`, Ice near-hazard `384`, delayed Ice `348`, delayed Support `216`, left/right Support matrices `432/408`, Sand benign `456`, Sand Support `456`, speed-Sand benign `432`, and staged-Sand benign `324`.

Speed counts were `.16:48, .17:24, .18:48, .19:24, .20:1,044, .21:24, .22:48, .23:24, .24:48, .25:2,664, .26:48, .27:24, .28:24, .30:1,212`. Forbidden, duplicate, and spacing violations were all `0`.

## 13. HNM round 2

The second round again scored 442 eligible TRAIN runs, selected 5,304 windows, and had 442 contributing runs. Concrete/Marble was `2,652/2,652`; family and speed distributions had the same counts as round 1. Selected probability median changed from `0.81145` to `0.37367`, as expected from changed model weights. Forbidden, duplicate, and spacing violations were all `0`.

## 14. HNM round 3

The third round again scored 442 eligible TRAIN runs, selected 5,304 windows, and had 442 contributing runs. Concrete/Marble was `2,652/2,652`; family and speed distributions again matched the counts listed for round 1. Selected probability median was `0.30778`. Forbidden, duplicate, and spacing violations were all `0`.

The baseline HNM policy remained unchanged: 1 ms replay, top 12 per run, 30 ms minimum spacing, accumulated exact-endpoint exclusion, TRAIN only. Selected endpoint identities were allowed to vary with the trained weights. HNM provenance SHA is `d75a1f6f21965e9c77600a167c8b08b3ec3975b5479939b0785b8e8a76e2009d`.

## 15. Exposure provenance

Each fit endpoint was presented once per completed epoch because the deterministic shuffled DataLoader used `drop_last=False`. Actual batch exposure aggregates over all seeds and four rounds were:

| Cell | Available endpoints | Actual batch exposures |
|---|---:|---:|
| Ordinary Support / Concrete | 312 | 59,280 |
| Ordinary Support / Marble | 328 | 62,320 |
| Delayed Support / Concrete | 135 | 25,650 |
| Delayed Support / Marble | 135 | 25,650 |
| 0.20 Slip | 435 | 82,650 |
| 0.25 Slip | 870 | 165,300 |
| 0.30 Slip | 375 | 71,250 |
| Left Slip | 135 | 25,650 |
| Right Slip | 75 | 14,250 |
| Bilateral Slip | 1,470 | 279,300 |

Total positive endpoint presentations were `492,100`. Exposure provenance SHA is `9c183313dc9fa6ee8322bdd48d053e79a2a3254a0cb202f2af1ab704f02e927d`.

## 16. Candidate freeze

The candidate was frozen before V2 waveform loading and is retained for provenance even though internal gates failed.

- Candidate ID: `model_v2_extraction_rebalanced_gru20_20260901`
- Candidate freeze SHA: `ca1ba7abae1746528cdd098b903ce8f967937a625773bb8029fc486645fc4533`
- Candidate evaluation freeze SHA: `3d415c0d4c635497f6d882218a059c9228f02fdda9936124439d709480d94dbc`
- Execution config SHA: `98c3e8327fd20dfd13b57a20b91f53ddd0fce4a4aed62266cfbad0459926fcd6`
- Extraction policy SHA: `3c7ce82ed905d932ec8f17d69d7e5edb5d79ee7602ba95ffe2a53d2407142cd2`
- V2 dataset freeze SHA: `fd81b647051b63e7b76ab72c6dedd616cf06d752fcbc1db31ba02f8aebe68744`
- Effective TRAIN hash: `0466ea84871d178856ffb10d8b1c0cec730286b35b589c73ff1b5fb1065aa5ab`
- Reused normalizer SHA: `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a`
- Final checkpoint SHAs: `f52a4c86…89883`, `c21e8fe4…d648e`, `5d88c027…5f223`
- Architecture SHA: `ae4753699399aa7e4a93639935e085156ef28ed100a77058d7308eb1549aa897`
- Feature schema SHA: `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`
- Runtime decision: threshold `0.99`, persistence `5 ms`
- V2 validation result SHA: `5e9d3e05d359671039a1ce873ea8c0b8bc521ab5d1c32aba8c9ab5819559b777`

No promoted `model_v2_freeze.json` was created because the internal primary gates did not all pass.

## 17. V2_VALIDATION primary result

The new candidate was evaluated once on the 96 valid `V2_VALIDATION` runs after candidate freeze.

| Metric | Result | Gate | Pass? |
|---|---:|---:|:---:|
| Overall Hazard recall | 58/64 = 90.625% | >=90% | yes |
| Slip recall | 29/35 = 82.857% | >=95% | no |
| Support recall | 30/30 = 100% | >=85% | yes |
| Confirmed no-hazard specificity | 23/26 = 88.462% | >=95% | no |
| Premature Hazard rate | 6/64 = 9.375% | <=10% | yes |
| Slip p95 established latency | 1.25 ms | <=40 ms | yes |
| Support p95 established latency | -18.35 ms | <=50 ms | yes |
| Ice-benign specificity | 4/4 = 100% | >=95% | yes |
| Staged-Sand benign specificity | 8/8 = 100% | >=95% | yes |
| Speed-Sand benign specificity | 9/12 = 75% | >=95% | no |
| Right-only Support recall | 12/12 = 100% | >=85% | yes |

Eight of eleven gates passed. Slip recall, confirmed specificity, and speed-Sand specificity failed.

## 18. Delayed Support

| Metric | Baseline V2 | Rebalanced V2 |
|---|---:|---:|
| Delayed Support overall | 3/6 (50%) | 6/6 (100%) |
| Concrete delayed Support | 3/3 (100%) | 3/3 (100%) |
| Marble delayed Support | 0/3 (0%) | 3/3 (100%) |
| Maximum probability at I1 | 0.997845 | 0.997968 |
| Maximum consecutive `>=0.99` | 8 ms | 14 ms |
| I1 to Reflex | Concrete 43 ms; Marble miss/655 ms | 4 ms on all six |
| Reflex to established Support | Concrete 13 ms lead; Marble miss/out-of-window | 52 ms lead on all six |
| Miss/out-of-window count | 3 | 0 |

The target intervention was effective in isolation: overall gain was `+3/6`, Marble gain was `+3/3`, and every rebalanced run satisfied the 5 ms persistence rule beginning at I1+4 ms.

| Run | Source | I1 | Support | First `>=0.99` | First Reflex | Max p | Max consecutive | Primary result | I1→Reflex / Reflex→Support |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| `m2v2_dss_v_c_0250_s10` | Concrete | 3011 | 3067 | 3011 | 3015 | 0.999833 | 14 ms | correct | 4 / 52 ms |
| `m2v2_dss_v_c_0250_s11` | Concrete | 3011 | 3067 | 3011 | 3015 | 0.999607 | 11 ms | correct | 4 / 52 ms |
| `m2v2_dss_v_c_0250_s12` | Concrete | 3011 | 3067 | 3011 | 3015 | 0.999607 | 11 ms | correct | 4 / 52 ms |
| `m2v2_dss_v_m_0250_s10` | Marble | 3012 | 3068 | 3012 | 3016 | 0.999698 | 10 ms | correct | 4 / 52 ms |
| `m2v2_dss_v_m_0250_s11` | Marble | 3012 | 3068 | 3012 | 3016 | 0.999698 | 10 ms | correct | 4 / 52 ms |
| `m2v2_dss_v_m_0250_s12` | Marble | 3012 | 3068 | 3012 | 3016 | 0.999842 | 14 ms | correct | 4 / 52 ms |

## 19. Marble delayed Support

All three Marble runs had I1 at sample `3012`, Support at `3068`, probability at I1 `0.994613`, and first reflex at `3016`. Each therefore produced I1-to-reflex `4 ms` and Reflex-to-Support lead `52 ms`. Maximum consecutive threshold excursions were `10, 10, 14 ms`; maximum probabilities were `0.999698, 0.999698, 0.999842`. Baseline V2 produced one miss and two reflexes 599 ms after Support, so all three were out of the valid primary window.

## 20. Ordinary/right Support preservation

Ordinary Support remained `24/24`: left-only `12/12`, right-only `12/12`. The frozen ordinary-Support subgroup contains the 24 non-delayed left/right matrix runs. Baseline right-only Support remained preserved at `12/12`. Total Support rose from baseline `27/30` to `30/30` because the three delayed Marble misses were resolved.

## 21. Specificity preservation

Specificity was not preserved. Hard-ground normal remained `6/6`, staged-Sand benign remained `8/8`, and Ice-benign remained `4/4`. Speed-Sand benign regressed from baseline `12/12` to `9/12`, causing confirmed specificity to fall from `26/26` to `23/26`. By speed, confirmed specificity was `.20:4/6`, `.25:9/10`, `.30:10/10`. These three false reflexes select the regression branch of the verdict hierarchy.

## 22. Slip preservation

Overall Slip was numerically preserved at baseline `29/35` but still failed the 95% gate.

| Stratum | Rebalanced V2 |
|---|---:|
| 0.20 m/s | 9/14 (64.286%) |
| 0.25 m/s | 11/11 (100%) |
| 0.30 m/s | 9/10 (90%) |
| Left-only | 3/3 (100%) |
| Right-only | 0/3 (0%) |
| Bilateral | 26/29 (89.655%) |
| Immediate timing | 4/4 (100%) |
| Delayed timing | 25/31 (80.645%) |

Of the six premature Ice cases, five alerts were inside a supported future-Slip precursor and one was before or outside precursor. None were inside benign-release or censored precursor. Primary scoring remained unchanged.

## 23. Ice precursor secondary view

The frozen secondary view covered 461 episodes: 424 future-Slip, 7 benign release, and 30 censored. Rebalanced V2 alerted inside 66 future-Slip candidate episodes and before Slip in 96; it missed 358 precursor states. It produced `0/7` benign-release false alerts and `1/30` censored candidate alert. Precursor-to-reflex median/p95 was `2/19.2 ms`; Reflex-to-established-Slip median/p95 was `21/159.25 ms`.

Compared with baseline V2, candidate-episode future-Slip alerts changed `72→66` and before-Slip alerts `103→96`; benign and censored counts stayed `0` and `1`. This view did not rewrite any primary metric.

## 24. V1 vs baseline V2 vs rebalanced V2

| Metric | V1 | Baseline V2 | Rebalanced V2 |
|---|---:|---:|---:|
| Overall Hazard | 53.125% | 85.938% | 90.625% |
| Slip | 62.857% | 82.857% | 82.857% |
| Support | 40% | 90% | 100% |
| Confirmed specificity | 88.462% | 100% | 88.462% |
| Premature rate | 20.313% | 9.375% | 9.375% |
| Right-only Support | 0% | 100% | 100% |
| Delayed Support | 0% | 50% | 100% |
| Marble delayed Support | 0% | 0% | 100% |
| Staged-Sand benign | 100% | 100% | 100% |
| Speed-Sand benign | 75% | 100% | 75% |

## 25. Intervention verdict

`V2_EXTRACTION_REBALANCE_NOT_EFFECTIVE`

The predeclared target-improved condition is true: delayed Support improved overall and on Marble. The solved-behavior-retained condition is false because confirmed and speed-Sand specificity regressed. Under the frozen hierarchy, a target gain accompanied by solved-behavior regression is not an effective intervention.

## 26. Full internal verdict

`MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED`

The candidate passed overall Hazard, Support, premature-rate, both latency, Ice-benign, staged-Sand, and right-Support gates. It failed Slip recall, confirmed specificity, and speed-Sand specificity. It is preserved as an auditable internal candidate but is not promoted for external generalization evaluation.

## 27. Limitations

- Evidence is internal `V2_VALIDATION` only; no external V2 inference was run.
- The delayed-Support source cells each still contain one unique event-local waveform repeated across seeds, so 3/3 per source does not establish broad waveform diversity.
- The extraction intervention improves the target behavior but does not explain why speed-Sand benign specificity regressed.
- No threshold/persistence tuning, architecture search, alternate seed selection, sensor expansion, or HNM redesign was authorized or performed.
- Generalization HOLDOUT remains sealed and cannot support any claim in this report.

Completion verification passed: `80 passed, 1 skipped`; `compileall` covered `src`, `scripts`, and `tests`; critical Ruff `E9/F63/F7/F82` and the full changed-file Ruff check passed; `git diff --check` passed; protected dataset/model integrity and HOLDOUT guard checks passed.

## 28. Recommended next milestone

`MODEL_V2_REBALANCE_REGRESSION_AUDIT`

This is the only recommended next milestone. It should be diagnostic-only and localize the three speed-Sand benign false reflexes relative to the extraction and HNM exposure. It is not started here. In particular, this milestone did not start another retraining, dataset generation, external Generalization evaluation, HOLDOUT access, threshold/persistence tuning, architecture change, Terrain retraining, deployment, quantization, HIL, Recovery, or GUI work.

Final training verdict: `MODEL_V2_EXTRACTION_REBALANCED_TRAINING_COMPLETE`.
