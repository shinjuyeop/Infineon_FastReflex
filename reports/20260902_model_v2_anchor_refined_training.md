# Model V2 Anchor-Refined Training

## 1. Purpose

This milestone executed the frozen `LATE_PRE_SUPPORT_INTERIOR` intervention from `MODEL_V2_DELAYED_SUPPORT_ANCHOR_REFINEMENT_DESIGN_READY`. It trained exactly one separate candidate, `model_v2_anchor_refined_gru20_20260902`, to test whether removing the dense established-Support-local positive group could retain delayed-Support recovery while restoring speed-Sand specificity.

The experiment achieved both intervention goals without regressing solved Support or benign-control behavior. Its intervention verdict is `V2_ANCHOR_REFINEMENT_EFFECTIVE`. The frozen primary Slip gate remains failed, so the independent internal verdict is `MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED` and the recommended next milestone is `MODEL_V2_CANDIDATE_READINESS_REVIEW`.

## 2. Starting state

- Starting `HEAD`: `9bef402e900523e4b5477bf47cd91c0adddf9b2a`
- Starting `origin/main`: `9bef402e900523e4b5477bf47cd91c0adddf9b2a`
- Starting parity: exact
- Starting tracked worktree: clean
- Prior verdict: `MODEL_V2_DELAYED_SUPPORT_ANCHOR_REFINEMENT_DESIGN_READY`
- Generalization HOLDOUT guard count: `0`

## 3. Historical candidate preservation

V1, baseline data-only V2, extraction-rebalanced V2, and Terrain V1 remained exact and independently restorable. The new checkpoint directory does not overlap any protected path.

| Protected object | SHA-256 after training | Status |
|---|---|---|
| V1 candidate freeze | `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2` | exact |
| V1 normalizer | `610a91c24a594a3cd1701b07549f4f3c4b7f4cb04eb354f9d41378214a6800a9` | exact |
| V1 seeds 20260828/29/30 | `e6bada49…d588`, `b04877dc…506`, `b6c782bd…753` | 3/3 exact |
| Baseline V2 candidate freeze | `edb16b7d96fb38e680e36dcca5ecd7b1c1682ba410b94d21f40c2a38d4ed8725` | exact |
| Baseline V2 normalizer | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` | exact |
| Baseline V2 seeds 20260828/29/30 | `dd6c8581…71c8`, `8e6709da…24a0`, `811f486c…4c42` | 3/3 exact |
| Rebalanced V2 candidate freeze | `ca1ba7abae1746528cdd098b903ce8f967937a625773bb8029fc486645fc4533` | exact |
| Rebalanced V2 evaluation freeze | `3d415c0d4c635497f6d882218a059c9228f02fdda9936124439d709480d94dbc` | exact |
| Rebalanced V2 seeds 20260828/29/30 | `f52a4c86…9883`, `c21e8fe4…648e`, `5d88c027…f223` | 3/3 exact |
| Terrain V1 normalizer | `2c551181fbd23080c0e3008be8484bd0ee4139e594041ba6e46ab5b36ffb05de` | exact |
| Terrain V1 seeds 17/29/43 | `21b0d122…f628`, `de6a55d3…0d66`, `465803f4…7b31` | 3/3 exact |

Fresh read-only replay of V1, baseline V2, and rebalanced V2 on the authorized 96-run validation split reproduced their frozen result objects exactly. No historical checkpoint was written.

## 4. Dataset integrity

Every manifest row was rehashed against its NPZ file. There were zero mismatches and no dataset generation or mutation.

| Corpus | Runs | Manifest SHA-256 | NPZ aggregate SHA-256 | Mismatches |
|---|---:|---|---|---:|
| Unified | 256 | `d023384aaaac22076a2be7c2fa242acac91a548d82102640c39868990cb0e9d6` | `555a44bc00ff46dd51716fc5cd151932c0fa1a5360b1ad3ddf3cc5250ea6f4aa` | 0 |
| Model V2 | 412 | `7a036d3485bb19a3570a3dd4a41cc990375028f7e153e85525bd4bf19cff8b25` | `5a8dfd54d1c08413dc6fb18d957269a5fbfea8cf526e4f29c78510886186e11c` | 0 |
| Generalization | 72 | `72f5dd300bb5be78b22b9a8c1ad9788cdf45e7cda9b4f46d79eb290140146f53` | `0964e521b4625466e72a42aa07dd4b57285adad744400fdd962d8c3efac792ad` | 0 |
| Ice semantics | 48 | `6a472d4b26e724355c6f2d88d0668c4f91c925037c63ad8e1838f83a53e15759` | `be6761cfa744014d291481bf3c3ef293b9dfdae026f2bee60809e43aa813e2f5` | 0 |

Effective TRAIN remained Unified TRAIN 152 plus valid V2_TRAIN 290, exactly 442 runs under identity SHA `0466ea84871d178856ffb10d8b1c0cec730286b35b589c73ff1b5fb1065aa5ab`. V2_VALIDATION remained 96 valid runs. The V2 dataset freeze SHA is `fd81b647051b63e7b76ab72c6dedd616cf06d752fcbc1db31ba02f8aebe68744`.

## 5. Frozen anchor policy

The selected delayed-Support rule was frozen before training:

- I1 neighborhood: `I1 + [0,1,2,3,4] ms`
- interval midpoint: `I1 + floor((Support-I1)/2) + [-2,-1,0,1,2] ms`
- late interior: `L = I1 + floor(3*(Support-I1)/4)`
- established-Support-local endpoints: none
- cap: 11 endpoints per eligible run

Slip positives, ordinary-Support positives, initial negatives, masks, feature schema, architecture, seeds, optimizer, HNM, threshold, and persistence were unchanged.

- Execution config SHA: `2da935d96c80452d69108ac14aa8a4df8edae297672cf25fd1945eb4deb64dbe`
- Anchor design config SHA: `a181461ecb1914ed7ecabdb9491eb591fc4712168f5c0517b2039d311a76ad51`
- Anchor-refinement design SHA: `a0525ffce941e05b5e2a51ec6e9d9489ac3abf10f05e775aedf28087ffa7cfdc`
- Candidate-list SHA: `cd6af9180613101f61fe271f2055d1c8a69daf7acf1d73ef429c8e0f47c555cb`
- Extraction-policy SHA: `52004bc2ddc307316a7a888855a1bd8014e50b96aa45a178a10965e890f4b199`

The execution config was frozen before optimizer step 1 and was not edited afterward.

## 6. Frozen monitor

The TRAIN-only candidate-invariant monitor SHA is `39d30234f674446f305b1b51d446977ba301e6db1e0591ac14dbe7172cbb1bf5`; positive identity SHA is `e4cd285091e55c92c773512b44958273d7773a708bad876806cec6a8401f9c88`.

| Monitor role | Endpoints |
|---|---:|
| Slip | 431 |
| Ordinary Support | 167 |
| Delayed Support, Concrete | 8 |
| Delayed Support, Marble | 11 |
| Total positive | 617 |
| Negative | 6,624 |

The monitor remained exactly `6,624 / 617` negative/positive in every seed and every round. Its endpoint intersection with fit was zero. It was never searched or adapted, and V2_VALIDATION was never used as a monitor.

## 7. Pre-training extraction reproduction

The mandatory zero-optimizer CLI dry run froze pretraining-audit SHA `5e592779af79953034da404135ecb0b98f3d2bce6cb1deb1f32b91ccdde1dbab`.

| Contract | Frozen result | Status |
|---|---|---|
| Positive endpoint SHA | `248719864bc1974ac54a21de63f04a6d5e6f55ef3e3c37092cf0ec757872d09e` | exact |
| Negative endpoint SHA | `392c1fda06953135bfed9a97c7cabb3915d4c50fdf9ff11b2ac8e6550448936c` | exact |
| Mask SHA | `32bae2d81b05709771545b375dd6cffb95d2dfc17c2808a7adf3a7f70174c35a` | exact |
| All positives / negatives | 3,135 / 32,209 | exact |
| Fit positives / negatives | 2,518 / 25,585 | exact |
| Fit Slip / ordinary Support / delayed Support | 1,680 / 640 / 198 | exact |
| Delayed eligible/represented runs | 18 / 18 | exact |
| Delayed endpoints, Concrete/Marble | 99 / 99 | exact |
| Fit/monitor endpoint overlap | 0 | exact |

## 8. Contradiction audit

All contradiction counters were zero: pre-I1 delayed-Support positive, positive/negative collision, future-Slip precursor negative, censored precursor negative, I1-positive negative, post-censor/fall endpoint, future-feature leakage, short delayed neighborhood, and persistence neighborhood shorter than 5 ms. All windows were causal and stopped at their endpoint.

Masked sample counts stayed fixed at future-Slip `41,479`, censored precursor `1,734`, and I1-positive `68,388`.

## 9. Normalizer reuse

The baseline V2 normalizer at `artifacts/runs/20260901_model_v2_data_only_training/normalization/gru_history20.json` was reused directly.

- Normalizer SHA before/after: `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a`
- Logical normalizer fits: `0`
- New normalizer writes: `0`

## 10. Training protocol

The candidate used Pelvis IMU6, causal 80D features, GRU history 20 ms, hidden size 32, one unidirectional layer, 11,010 parameters, and two output classes. Seeds were `20260828`, `20260829`, and `20260830`. Optimization was Adam at `0.001`, inverse-frequency weighted cross entropy, batch size 128, maximum 40 epochs, patience 6, deterministic shuffled DataLoader, no gradient clipping, and fixed TRAIN-monitor validation loss selection.

Runtime remained the mean probability of all three predeclared seeds, threshold `0.99`, and five consecutive 1 kHz samples. No seed or checkpoint was selected after evaluation.

## 11. Round 0

Round 0 materialized 25,585 negative and 2,518 positive fit windows. All 18 delayed-Support runs contributed 198 endpoints. The three seeds completed 12, 16, and 17 epochs; their frozen best epochs were 6, 10, and 11.

## 12. HNM 1

The first TRAIN-only mining pass selected 5,304 endpoints from 442/442 contributing runs, with Concrete/Marble `2,652/2,652`. Endpoint SHA was `4a0b0bb3610f1d7ce7eac035cc7a6c3f347b489e0ba2299186f1c04e4158d3fa`. Forbidden-mask, duplicate, and spacing violations were all zero.

## 13. Round 1

Round 1 used 29,640 negative and 2,518 positive fit windows. The fixed monitor remained `6,624/617`. Seeds completed 10, 10, and 13 epochs, with best epochs 4, 4, and 7.

## 14. HNM 2

The second pass selected 5,304 endpoints from 442/442 runs, again Concrete/Marble `2,652/2,652`. Endpoint SHA was `1ca59fc13213d0a8ee6987ad933d083848531d7ec829f84b1102dbd574699085`. All forbidden, duplicate, and spacing counters were zero.

## 15. Round 2

Round 2 used 33,692 negative and 2,518 positive fit windows. Seeds completed 11, 13, and 10 epochs, with best epochs 5, 7, and 4.

## 16. HNM 3

The final pass selected 5,304 endpoints from 442/442 runs, Concrete/Marble `2,652/2,652`. Endpoint SHA was `e1bcca00c28a553696198a36d48f7149d736f8884230addcf6e71f8f36e77d5f`. All forbidden, duplicate, and spacing counters were zero.

| HNM round | Selected | Effective added | Runs | Concrete | Marble | Speed Sand | Staged Sand | Forbidden | Duplicate | Spacing |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,304 | 4,055 | 442 | 2,652 | 2,652 | 432 | 324 | 0 | 0 | 0 |
| 2 | 5,304 | 4,052 | 442 | 2,652 | 2,652 | 432 | 324 | 0 | 0 | 0 |
| 3 | 5,304 | 4,109 | 442 | 2,652 | 2,652 | 432 | 324 | 0 | 0 | 0 |

Each pass also selected baseline immediate Ice 432, delayed Support 216, hard normal 456+216, Ice benign 288, Ice near-hazard 384, Ice Slip 456, delayed Ice 348, left/right Support 432/408, Sand benign 456, and Sand Support 456. HNM remained 1 ms replay, top 12 per run, 30 ms minimum spacing, accumulated exact-endpoint exclusion, effective TRAIN only. HNM provenance SHA is `cd69a67c6ec21ac44f5980998b11751bc117fe5aeaa8bcc98ac72f17a74c1db7`.

## 17. Round 3

Round 3 used 37,801 negative and 2,518 positive fit windows. Seeds completed 10, 13, and 13 epochs, with best epochs 4, 7, and 7. No HNM or model update followed.

## 18. Optimizer exposure

### Seed/round record

| Seed | Round | Fit + | Fit - | Delayed Support | Monitor -/+ | HNM added | Best/completed | Steps | Checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20260828 | 0 | 2,518 | 25,585 | 198 | 6,624/617 | 4,055 | 6/12 | 2,640 | `8620fca7c7f167b69daf50203762e30f88a53acf28faad169b2e7786827fdc30` |
| 20260829 | 0 | 2,518 | 25,585 | 198 | 6,624/617 | 4,055 | 10/16 | 3,520 | `898233a8304414ac3e1842d8cfef4d08764ddb87a5760161495e283e5b3a90be` |
| 20260830 | 0 | 2,518 | 25,585 | 198 | 6,624/617 | 4,055 | 11/17 | 3,740 | `9075679eabacfa904119acddf3c65bdcae8969aff6540593cda1c1dcbc6a70de` |
| 20260828 | 1 | 2,518 | 29,640 | 198 | 6,624/617 | 4,052 | 4/10 | 2,520 | `0571bfd51a73cba3ac1892e49eef804a032b06398ff7afa8da3a73299c89ab36` |
| 20260829 | 1 | 2,518 | 29,640 | 198 | 6,624/617 | 4,052 | 4/10 | 2,520 | `1332fce235c0436876b46b46edd9574978a606453108250c83f1dda020b14838` |
| 20260830 | 1 | 2,518 | 29,640 | 198 | 6,624/617 | 4,052 | 7/13 | 3,276 | `4f16b513e6024d0eb740e7fe4b3de82d885fc2d6da70ec5041130d166c06a771` |
| 20260828 | 2 | 2,518 | 33,692 | 198 | 6,624/617 | 4,109 | 5/11 | 3,113 | `fecd886c24349fc472bb9f33adf7c77e70210554cf5e600d855cefea82c18cdf` |
| 20260829 | 2 | 2,518 | 33,692 | 198 | 6,624/617 | 4,109 | 7/13 | 3,679 | `afe3c5218919b6f1004be9ba0ce0a23165598f9838a42a6841db31c955dced1a` |
| 20260830 | 2 | 2,518 | 33,692 | 198 | 6,624/617 | 4,109 | 4/10 | 2,830 | `cc4bf0f1c1ae412cc13d106f00857155470244f671a1ef8ed61572177083961a` |
| 20260828 | 3 | 2,518 | 37,801 | 198 | 6,624/617 | — | 4/10 | 3,150 | `7094a2dca40e8d3c84554619652d69c17920c8e82765460ff8621c13ef494cb9` |
| 20260829 | 3 | 2,518 | 37,801 | 198 | 6,624/617 | — | 7/13 | 4,095 | `3ad298eea4c35eca896afd31f860fd6b44ce35d7d9978e2546bb40b693e62c39` |
| 20260830 | 3 | 2,518 | 37,801 | 198 | 6,624/617 | — | 7/13 | 4,095 | `fe96dfeb8461871044de0f8672190680ce46164a1c28efbe6738d22f9d439bbd` |

### Actual class weights

| Round | Positive | Negative | Positive weight | Negative weight | N/P ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 2,518 | 25,585 | 5.580421 | 0.549209 | 10.160842 |
| 1 | 2,518 | 29,640 | 6.385624 | 0.542476 | 11.771247 |
| 2 | 2,518 | 33,692 | 7.190230 | 0.537368 | 13.380461 |
| 3 | 2,518 | 37,801 | 8.006156 | 0.533306 | 15.012311 |

Weights were derived mechanically from each common round's class counts; there was no manual weight change.

### Positive endpoint presentations

Because the deterministic DataLoader used `drop_last=False`, each fit endpoint was presented once per completed epoch.

| Cell | Available | Actual presentations |
|---|---:|---:|
| Delayed Support, Concrete | 99 | 14,652 |
| Delayed Support, Marble | 99 | 14,652 |
| Ordinary Support, Concrete | 312 | 46,176 |
| Ordinary Support, Marble | 328 | 48,544 |
| Slip 0.20 / 0.25 / 0.30 | 435 / 870 / 375 | 64,380 / 128,760 / 55,500 |
| Slip left / right / bilateral | 135 / 75 / 1,470 | 19,980 / 11,100 / 217,560 |
| Support left / right | 630 / 208 | 93,240 / 30,784 |

Total positive presentations were 372,664. Baseline/rebalanced/anchor-refined delayed-Support available endpoints were 104/270/198; anchor-refined retained all 18 runs with balanced 99/99 source coverage while removing 72 dense Support-local endpoints relative to rebalanced.

Initial negatives received `25,585 × 148 = 3,786,580` presentations across the 148 completed seed-epochs. Selected HNM endpoints were deduplicated against initial negatives, prior HNM, and positives before materialization; the effective additions were 4,055, 4,052, and 4,109. Total negative presentations were 4,635,809. Actual major family presentations, including initial and effective HNM windows, were hard normal 584,768, speed-Sand 403,390, staged-Sand 300,925, Ice benign 292,256, Ice near-hazard 359,580, Ice Slip 364,238, delayed Ice 326,385, delayed Support 180,769, left Support 348,670, and right Support 369,691. This is optimizer exposure, not a sampling weight. Exposure provenance SHA is `93ef7661ec0206de111e2ec883e47f33ea7114989274a052f2c1bacfa56ad7bb`.

## 19. Candidate freeze

The candidate was frozen before V2_VALIDATION waveform loading. No retraining followed.

- Candidate ID: `model_v2_anchor_refined_gru20_20260902`
- Candidate freeze SHA: `95dab53275aa77ded36b479c3633ab879a1331cb2c96bd0eb014de2be99bd85f`
- Candidate evaluation freeze SHA: `cad3902137b622c3a3d15ecb3d6c3bb31ee9751f3605fb8d43daa6ac81695c07`
- Final seed checkpoint SHAs: `7094a2dc…4cb9`, `3ad298ee…2c39`, `fe96dfeb…9bbd`
- Architecture SHA: `ae4753699399aa7e4a93639935e085156ef28ed100a77058d7308eb1549aa897`
- Feature schema SHA: `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`
- V2_VALIDATION result SHA: `6e53072c8ec1ccc5de63437ccd2edc08a6728e5506cecb06ca7e3a1b2263b6fb`

No promoted `model_v2_freeze.json` was created because the frozen internal gates did not all pass.

## 20. V2_VALIDATION primary result

The frozen candidate was evaluated once on the 96 valid V2_VALIDATION runs.

| Metric | Result | Gate | Pass |
|---|---:|---:|---|
| Overall Hazard | 59/64 = 92.19% | ≥90% | yes |
| Slip | 30/35 = 85.71% | ≥95% | **no** |
| Support | 30/30 = 100% | ≥85% | yes |
| Confirmed no-hazard specificity | 26/26 = 100% | ≥95% | yes |
| System premature | 5/64 = 7.81% | ≤10% | yes |
| Slip p95 valid latency | +27.2 ms | ≤+40 ms | yes |
| Support p95 established latency | -17 ms | ≤+50 ms | yes |
| Ice-benign primary specificity | 4/4 = 100% | ≥95% | yes |
| Staged-Sand specificity | 8/8 = 100% | ≥95% | yes |
| Speed-Sand specificity | 12/12 = 100% | ≥95% | yes |
| Right-only Support | 12/12 = 100% | ≥85% | yes |

Only `slip_hazard_recall` failed.

## 21. Delayed Support

The anchor-refined candidate retained rebalanced delayed-Support performance: 6/6 overall, Concrete 3/3, and Marble 3/3. All six reflexes began 4 ms after I1 and 52 ms before established Support.

The following table reports all four immutable models. `—` means no reflex onset. Probabilities are ensemble means at I1, the selected late-interior `L`, and Support.

| Model | Run | Src | I1/L/S | p(I1)/p(L)/p(S) | First crossing/reflex | Max ≥.99 | Result | I1→R / R→S ms |
|---|---|---|---|---|---|---:|---|---|
| V1 | `m2v2_dss_v_c_0250_s10` | C | 3011/3053/3067 | .000033/.644102/.770097 | 1878/2461 | 16 | premature | -550/606 |
| V1 | `m2v2_dss_v_c_0250_s11` | C | 3011/3053/3067 | .000033/.644102/.770097 | 1878/2461 | 11 | premature | -550/606 |
| V1 | `m2v2_dss_v_c_0250_s12` | C | 3011/3053/3067 | .000033/.644102/.770097 | 1878/2461 | 30 | premature | -550/606 |
| V1 | `m2v2_dss_v_m_0250_s10` | M | 3012/3054/3068 | .000041/.681352/.922829 | 1877/2459 | 10 | premature | -553/609 |
| V1 | `m2v2_dss_v_m_0250_s11` | M | 3012/3054/3068 | .000041/.681352/.922829 | 1877/2459 | 16 | premature | -553/609 |
| V1 | `m2v2_dss_v_m_0250_s12` | M | 3012/3054/3068 | .000041/.681352/.922829 | 1877/2459 | 19 | premature | -553/609 |
| Baseline | `m2v2_dss_v_c_0250_s10` | C | 3011/3053/3067 | .994350/.997000/.982430 | 3011/3054 | 7 | correct | 43/13 |
| Baseline | `m2v2_dss_v_c_0250_s11` | C | 3011/3053/3067 | .994350/.997000/.982430 | 3011/3054 | 7 | correct | 43/13 |
| Baseline | `m2v2_dss_v_c_0250_s12` | C | 3011/3053/3067 | .994350/.997000/.982430 | 3011/3054 | 7 | correct | 43/13 |
| Baseline | `m2v2_dss_v_m_0250_s10` | M | 3012/3054/3068 | .997845/.990205/.985145 | 3012/— | 3 | miss | —/— |
| Baseline | `m2v2_dss_v_m_0250_s11` | M | 3012/3054/3068 | .997845/.990205/.985145 | 3012/3667 | 8 | out of window | 655/-599 |
| Baseline | `m2v2_dss_v_m_0250_s12` | M | 3012/3054/3068 | .997845/.990205/.985145 | 3012/3667 | 8 | out of window | 655/-599 |
| Rebalanced | `m2v2_dss_v_c_0250_s10` | C | 3011/3053/3067 | .997968/.631937/.997614 | 3011/3015 | 14 | correct | 4/52 |
| Rebalanced | `m2v2_dss_v_c_0250_s11` | C | 3011/3053/3067 | .997968/.631937/.997614 | 3011/3015 | 11 | correct | 4/52 |
| Rebalanced | `m2v2_dss_v_c_0250_s12` | C | 3011/3053/3067 | .997968/.631937/.997614 | 3011/3015 | 11 | correct | 4/52 |
| Rebalanced | `m2v2_dss_v_m_0250_s10` | M | 3012/3054/3068 | .994613/.656111/.999222 | 3012/3016 | 10 | correct | 4/52 |
| Rebalanced | `m2v2_dss_v_m_0250_s11` | M | 3012/3054/3068 | .994613/.656111/.999222 | 3012/3016 | 10 | correct | 4/52 |
| Rebalanced | `m2v2_dss_v_m_0250_s12` | M | 3012/3054/3068 | .994613/.656111/.999222 | 3012/3016 | 14 | correct | 4/52 |
| Anchor | `m2v2_dss_v_c_0250_s10` | C | 3011/3053/3067 | .997022/.995104/.207440 | 3011/3015 | 8 | correct | 4/52 |
| Anchor | `m2v2_dss_v_c_0250_s11` | C | 3011/3053/3067 | .997022/.995104/.207440 | 3011/3015 | 8 | correct | 4/52 |
| Anchor | `m2v2_dss_v_c_0250_s12` | C | 3011/3053/3067 | .997022/.995104/.207440 | 3011/3015 | 8 | correct | 4/52 |
| Anchor | `m2v2_dss_v_m_0250_s10` | M | 3012/3054/3068 | .997653/.995990/.257422 | 3012/3016 | 8 | correct | 4/52 |
| Anchor | `m2v2_dss_v_m_0250_s11` | M | 3012/3054/3068 | .997653/.995990/.257422 | 3012/3016 | 8 | correct | 4/52 |
| Anchor | `m2v2_dss_v_m_0250_s12` | M | 3012/3054/3068 | .997653/.995990/.257422 | 3012/3016 | 8 | correct | 4/52 |

## 22. Marble delayed Support

Marble improved from baseline 0/3 to rebalanced 3/3 and stayed 3/3 after refinement. The refined model's probability remained above threshold through the I1 persistence neighborhood and at the late-interior anchor, then fell to about `.257` at established Support. This directly supports the intervention mechanism: the dense Support-local group was not required to retain the useful early Support decision.

## 23. Speed-Sand benign

Speed-Sand specificity recovered from rebalanced 9/12 to anchor-refined 12/12, matching baseline. All rows remained physically benign. A crossing with no five-sample reflex is not a false reflex.

| Run | Src/speed | Baseline max/R | Rebalanced max/R | Anchor max/cross/R | Anchor seed maxima (28/29/30) | Physical benign |
|---|---|---|---|---|---|---|
| `m2v2_sbb_v_c_0200_g07` | C/.20 | .982175/— | .998088/5431 | .959506/—/— | .997402/.997907/.997500 | yes |
| `m2v2_sbb_v_c_0200_g08` | C/.20 | .992923/— | .939166/— | .942828/—/— | .996438/.994556/.997102 | yes |
| `m2v2_sbb_v_c_0250_g07` | C/.25 | .893497/— | .948811/— | .947831/—/— | .970388/.951210/.976998 | yes |
| `m2v2_sbb_v_c_0250_g08` | C/.25 | .912747/— | .896401/— | .779207/—/— | .974662/.965630/.975360 | yes |
| `m2v2_sbb_v_c_0300_g07` | C/.30 | .944595/— | .877888/— | .977196/—/— | .951197/.986261/.997709 | yes |
| `m2v2_sbb_v_c_0300_g08` | C/.30 | .710356/— | .785044/— | .704605/—/— | .956358/.887705/.987984 | yes |
| `m2v2_sbb_v_m_0200_g07` | M/.20 | .920206/— | .999357/6098 | .993306/6097/— | .991777/.998320/.998940 | yes |
| `m2v2_sbb_v_m_0200_g08` | M/.20 | .992874/— | .965306/— | .967222/—/— | .998108/.997659/.991776 | yes |
| `m2v2_sbb_v_m_0250_g07` | M/.25 | .981436/— | .999861/4892 | .985881/—/— | .997879/.998277/.999641 | yes |
| `m2v2_sbb_v_m_0250_g08` | M/.25 | .754952/— | .688862/— | .735169/—/— | .993218/.953854/.991103 | yes |
| `m2v2_sbb_v_m_0300_g07` | M/.30 | .981878/— | .994934/— | .987796/—/— | .986530/.998143/.999644 | yes |
| `m2v2_sbb_v_m_0300_g08` | M/.30 | .888124/— | .881284/— | .834263/—/— | .988026/.977860/.971443 | yes |

The three prior regression rows all recovered: `m2v2_sbb_v_c_0200_g07`, `m2v2_sbb_v_m_0200_g07`, and `m2v2_sbb_v_m_0250_g07`. By source, anchor-refined was Concrete 6/6 and Marble 6/6; by speed it was 4/4 at `.20`, `.25`, and `.30` m/s.

## 24. Specificity preservation

| Behavior | Anchor-refined result |
|---|---:|
| Confirmed no-hazard | 26/26 |
| Hard normal | 6/6 |
| Ice benign primary | 4/4 no-established-hazard runs |
| Staged Sand benign | 8/8 |
| Speed Sand benign | 12/12 |
| Ordinary Support | 24/24 |
| Ordinary left/right Support | 12/12, 12/12 |
| All right-only Support | 12/12 |

No new solved-behavior regression was observed.

## 25. Slip preservation

Slip extraction and primary semantics were unchanged.

| Slice | Result |
|---|---:|
| Overall | 30/35 = 85.71% |
| 0.20 m/s | 9/14 = 64.29% |
| 0.25 m/s | 11/11 = 100% |
| 0.30 m/s | 10/10 = 100% |
| Left-only | 3/3 = 100% |
| Right-only | 0/3 = 0% |
| Bilateral | 27/29 = 93.10% |
| Immediate | 4/4 = 100% |
| Delayed | 26/31 = 83.87% |

All five premature Ice alerts occurred inside a future-Slip precursor; there were zero before/outside precursor, benign-release, or censored-precursor premature alerts. The anchor experiment did not alter the frozen primary timing boundary.

## 26. Ice precursor secondary result

The separate precursor-aware result covered 461 episodes: 424 future-Slip, 7 benign-release, and 30 censored. It recorded 44 future-Slip alerts inside the 30–50 mm candidate region and 58 alerts before established Slip. Benign-release alerts were 0 and censored candidate-region alerts were 0.

Precursor-to-reflex timing had median 2 ms and p95 98 ms. Reflex-to-established-Slip timing had median 19 ms and p95 152.4 ms. `primary_scores_rewritten` remained false; the secondary view did not change the 30/35 primary Slip score.

## 27. Four-model comparison

| Metric | V1 | Baseline V2 | Rebalanced V2 | Anchor-refined V2 |
|---|---:|---:|---:|---:|
| Hazard | 53.13% | 85.94% | 90.63% | **92.19%** |
| Slip | 62.86% | 82.86% | 82.86% | **85.71%** |
| Support | 40.00% | 90.00% | 100% | **100%** |
| Confirmed specificity | 88.46% | 100% | 88.46% | **100%** |
| Premature | 20.31% | 9.38% | 9.38% | **7.81%** |
| Delayed Support | 0/6 | 3/6 | 6/6 | **6/6** |
| Marble delayed Support | 0/3 | 0/3 | 3/3 | **3/3** |
| Right-only Support | 0/12 | 12/12 | 12/12 | **12/12** |
| Staged Sand benign | 8/8 | 8/8 | 8/8 | **8/8** |
| Speed Sand benign | 9/12 | 12/12 | 9/12 | **12/12** |
| 0.20 Slip | 35.71% | 64.29% | 64.29% | **64.29%** |
| Right-only Slip | 0/3 | 0/3 | 0/3 | **0/3** |

All four models remained immutable during comparison.

## 28. Anchor-refinement intervention verdict

`V2_ANCHOR_REFINEMENT_EFFECTIVE`

Delayed Support remained 6/6 including Marble 3/3, speed-Sand recovered from 9/12 to 12/12, confirmed specificity recovered from 23/26 to 26/26, and right Support/staged Sand remained 12/12 and 8/8. The predeclared support-gain, specificity-recovery, and solved-behavior conditions all passed.

## 29. Internal model verdict

`MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED`

The formal verdict uses all frozen primary gates. Slip was 30/35 (85.71%) against the 95% minimum; every other gate passed. Intervention success does not promote the candidate or rewrite the primary metric.

## 30. Architecture/sensor implications

This result supplies no evidence for longer history, LSTM, a larger GRU, Terrain gating, or sensor expansion. The targeted data-only timing refinement resolved its intended tradeoff with the existing Pelvis IMU6 → causal 80D → GRU20 hidden-32 architecture.

The provisional system remains Hazard Pelvis IMU6 plus Terrain left FSR4, ten channels total. Final sensor freeze remains **NO** pending hardware realism and deployment evidence.

## 31. External evidence preservation

- Generalization VALIDATION new-candidate inference: **NO**
- Current Unified HOLDOUT waveform reopened: **NO**
- Current Unified HOLDOUT new inference: **NO**
- Generalization HOLDOUT waveform opened: **NO**
- Generalization HOLDOUT inference: **NO**
- Generalization HOLDOUT guard count: `0`

No external validation, HOLDOUT visualization, or model selection was performed.

## 32. Limitations

The internal split contains only six delayed-Support runs and twelve speed-Sand controls; exact repeated local waveforms constrain diversity. Slip remains below the original primary gate, especially at `.20 m/s` and right-only geometry, while the precursor-aware secondary evidence shows that the five current failures are early physical precursor responses rather than low-response misses. This milestone does not decide whether that primary/secondary timing tension is acceptable for external development evaluation.

Training/evaluation artifacts are Gitignored research evidence, not a deployment release. Quantization, firmware, E84 integration, HIL, and Recovery remain out of scope.

## 33. Recommended next milestone

`MODEL_V2_CANDIDATE_READINESS_REVIEW`

The intervention is effective, Support and specificity are solved internally, and the only remaining gate is Slip under the established Ice precursor timing conflict. The next milestone should be a no-retraining, no-HOLDOUT scientific readiness review that preserves both the original primary metric and precursor-aware secondary metric. It is not started here.

Training execution verdict: `MODEL_V2_ANCHOR_REFINED_TRAINING_COMPLETE`.

Counters: optimizer steps `39,178`; checkpoint writes `12`; normalizer fits `0`; HNM rounds `3`; threshold searches `0`; persistence searches `0`; architecture searches `0`; seed searches `0`; monitor searches `0`; new simulation runs `0`.

Completion verification passed: `82 passed, 1 skipped`; `compileall` covered `src`, `scripts`, and `tests`; critical Ruff `E9/F63/F7/F82` and full changed-file Ruff passed; `git diff --check` passed. Read-only CLI verification passed for V1, Terrain V1, baseline V2, rebalanced V2, and anchor-refined V2. Post-training rehash checked Unified `256/256`, Model V2 `412/412`, Generalization `72/72`, and Ice semantics `48/48` with zero mismatch.
