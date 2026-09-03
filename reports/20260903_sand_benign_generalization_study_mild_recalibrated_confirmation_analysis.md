# Mild-Recalibrated Sand Generalization Confirmation Analysis

## 1. Purpose

This milestone consumes only the fresh 88-run `MILD_RECALIBRATED_CONFIRMATION` development split to test the single frozen Discovery hypothesis, `DOMAIN_DIVERSITY_GAP_SUPPORTED`. It repeats the predeclared model-independent Pelvis, realizable FSR/contact, privileged-oracle, factor-localization, and exact frozen-V2 analyses. It performs no new hypothesis selection, training, tuning, architecture change, or sensor intervention.

The analysis is valid, but one required H1 criterion does not replicate. The scientific Confirmation verdict is therefore `DOMAIN_DIVERSITY_GAP_NOT_CONFIRMED`.

## 2. Starting state

- Starting `HEAD` and `origin/main`: `cda763d1db1400ebf1fe601f19f5e04fe56243bd`
- Starting tracked worktree: clean
- Previous milestone: `SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_DISCOVERY_ANALYSIS`
- Discovery validity: `SAND_BENIGN_GENERALIZATION_STUDY_DISCOVERY_ANALYSIS_VALID`
- Frozen Discovery hypothesis: `DOMAIN_DIVERSITY_GAP_SUPPORTED`
- Physical generation: 176/176 generated, 70/70 gates pass

The Confirmation config and implementation were frozen before the first Confirmation payload access. Their SHA-256 values are `29883f802c0247a21337015037962d1f49c6077244798c86980fc85f9bdd4625` and `76aba665b4fffb5edd6ae51db6be36e818edded903874a7ccf7b704ae5ffce56`.

## 3. Historical evidence boundary

The historical Generalization HOLDOUT remains permanently consumed: guard `1`, scientific opens `1`. This milestone made zero old-HOLDOUT payload reads, feature reconstructions, model inferences, and visualizations. Its guard SHA-256 remains `0913da8a583cab7834590d669ba9c8d470485b8129fe81f538aba6c461613154`.

Before the Confirmation guard opened, there were zero Confirmation payload deserializations, feature analyses, model replays, and hypothesis tests. Preflight used only manifests, immutable hashes, file existence/size, model loading, and Discovery payloads needed to recover the already-frozen pooled scaler values. No Confirmation evidence informed the protocol freeze.

## 4. Fresh study identity

| Object | Verified identity |
|---|---|
| Dataset | `sand_benign_generalization_mild_recalibrated_study_20260903` |
| Total / Discovery / Confirmation | 176 / 88 / 88 |
| Manifest SHA-256 | `f19ec527cb9faac0d8f3a385a1a63e8a951ced7f275c7cfc3dd459cc42f375d1` |
| Discovery split SHA-256 | `1c211c38ee2bd7f9e9a44e0f81ec6a0dd110a8e14809c12777c043d33928f93f` |
| Confirmation split SHA-256 | `3bfbb050db3ebadcc363d1c6e51013dc349dbac4e0594c183a55245ea38c2e80` |
| Confirmation seal SHA-256 | `2795fa2cc02a049dbe0de2331820506845d333980e17fd0c6e33e5ce471082c2` |
| Semantic dataset-freeze SHA-256 | `706d939c03bf31df0fb39d1043e99dbbb05922664e207425c8c96ab7c93ee675` |

All identities matched before authorization. The dataset, split membership, physical outcomes, candidate, normalizer, and checkpoints were not modified.

## 5. Frozen Discovery hypothesis

The only hypothesis tested was `DOMAIN_DIVERSITY_GAP_SUPPORTED`, frozen under `SAND_BENIGN_DISCOVERY_INTERPRETATION_SHA = 7c045cd98bb221a0f41911a9662b430548393be90ef192e3194bd619cc3f2ae5`.

The required direction was transition-left more adverse than transition-right and, in the same physically coupled region, right-single precontact more adverse than left-single precontact. The metric implementation remained `src/fastreflex/evaluation/sand.py` at SHA-256 `9ded34b78647e64cd9825070fb80a1397e09d0c73f32df7b901500977ac4014e`. No metric, factor, direction, or H1/H2/H3 rule changed.

Discovery-pooled scaler values had not been stored, so they were deterministically reconstructed from Discovery with the exact frozen feature/anchor implementation. All five reconstructed mean/std hashes exactly matched the frozen Discovery artifacts before Confirmation opened. They were then reused without Confirmation refit; the resulting scaler artifact SHA-256 is `62935047a67b0ab44265299b5e4bbfb3e012c80347ae96a557ab70a07d65ae38`.

## 6. Confirmation authorization

The study guard changed exactly once from `0` to `1` at `2026-09-03T13:21:27.010635+09:00`, after config, dataset, Discovery interpretation, metric implementation, candidate, normalizer, checkpoint, and feature-schema verification. The authorization bound the starting commit, Confirmation split SHA, config SHA, Discovery interpretation SHA, candidate record SHA, and metric implementation SHA. Guard file SHA-256 is `5e2995cc2a1829c2c78cc0b49bab1d3192823aead0c3b7d126f9793427984238`.

Exactly 88 payloads were deserialized, once each, and the exact V2 was replayed once across the full split. The guard now rejects any second scientific run.

## 7. Confirmation population

| Physical population | Runs | Analysis role |
|---|---:|---|
| Strict mild Sand benign | 48 | Sand benign |
| Strict moderate Sand benign | 17 | Sand benign |
| Ordinary established Support | 12 | Support control |
| Delayed established Support | 4 | Support control |
| Actual moderate Slip | 3 | descriptive provenance only |
| Invalid | 4 | provenance only |
| **Total** | **88** | 81 primary separability eligible |

The primary benign-vs-Support denominator is 65 strict Sand plus 16 valid Support. Actual Slip was not relabeled as benign and invalid runs did not enter class metrics.

## 8. Frozen V2

The only replayed candidate was `model_v2_anchor_refined_gru20_20260902`: Pelvis IMU6, frozen causal 80D schema SHA-256 `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`, `[20,80]` history, one-layer unidirectional GRU hidden 32, and the mean of seeds `20260828/20260829/20260830`.

Candidate record, candidate freeze, normalizer, and checkpoint SHA-256 values remained `52644d3e…b7bc2`, `95dab532…d85f`, `e0d796e8…e92a`, and `7094a2dc…cb9` / `3ad298ee…c39` / `fe96dfeb…bbd`. The inclusive threshold was `.99`, persistence was 5 ms, and replay stride was 1 ms. There was no model, seed, normalizer, threshold, or persistence selection.

## 9. Pelvis Confirmation metrics

Both representations use one model-independent anchor and one vector per eligible run, the final-V2 normalizer, and the fixed Discovery-pooled distance scalers.

| Frozen metric | Current 80D | Window `[20,80]` |
|---|---:|---:|
| Centroid separation | .664068 | .208030 |
| Centroid distance | 4.693735 | 49.564479 |
| Within-group RMS | 7.068158 | 238.256187 |
| Balanced 1NN agreement | .992308 | 1.000000 |
| Balanced 5NN agreement | .992308 | 1.000000 |
| Opposite/same ratio | 1.4540e8 | 1.9185e9 |
| Ratio, Sand class median | 2.9081e8 | 3.8370e9 |
| Ratio, Support class median | 4.028235 | 5.113534 |
| Local opposite-class mixing | .062404 | .003077 |
| Bidirectional 95% radius inclusion | .776923 | .500000 |
| Sand in Support radius | .553846 | .000000 |
| Support in Sand radius | 1.000000 | 1.000000 |

| Representation / distance set | p05 | p25 | Median | p75 | p95 |
|---|---:|---:|---:|---:|---:|
| Current within Sand | .326791 | 4.003918 | 14.480732 | 17.341734 | 21.015723 |
| Current within Support | .791010 | 3.792818 | 4.930766 | 6.281907 | 7.059748 |
| Current between | 2.972907 | 4.993159 | 6.377396 | 15.257272 | 18.123408 |
| Window within Sand | 5.726670 | 32.345673 | 52.002261 | 78.547190 | 112.772383 |
| Window within Support | 9.154690 | 22.547347 | 26.513243 | 29.761376 | 35.180637 |
| Window between | 37.912403 | 42.086893 | 44.304576 | 48.105419 | 92.842110 |

Deterministic full-SVD PCA for current 80D explains `.463803/.293936` on PC1/PC2, with 95% projection-interval Jaccard overlaps `.139436/.107318`. Window PCA explains `.983477/.006017`, with overlaps `.530548/.061793`. The fixed current/window scaler mean hashes are `f81c632a…a845` and `4420b2d4…f502`; std hashes are `d5268e31…0aa` and `5381cbcc…3604`.

The window decision representation passes balanced 5NN, ratio, and mixing, but centroid separation is `.208030 < .75`; reasonable separation is therefore 3/4 and fails the frozen all-four rule. Only one of five strong-mixing checks triggers, so strong broad Pelvis mixing remains unsupported.

## 10. FSR/contact Confirmation metrics

The realizable 39D FSR/contact vector and descriptive Pelvis+FSR concatenation use the same frozen construction as Discovery. Exact loaded contact and Support spread are excluded; no classifier, probe, or fusion model was trained.

| Frozen metric | Pelvis window | Realizable FSR/contact | Pelvis + FSR/contact |
|---|---:|---:|---:|
| Centroid separation | .208030 | 1.100042 | .209191 |
| Balanced 1NN | 1.000000 | 1.000000 | 1.000000 |
| Balanced 5NN | 1.000000 | 1.000000 | 1.000000 |
| Opposite/same ratio | 1.9185e9 | 3.3022e8 | 1.9505e9 |
| Local mixing | .003077 | .000000 | .003077 |
| Radius inclusion | .500000 | .500000 | .500000 |

| Distance set | Realizable FSR p05 / median / p95 | Combined p05 / median / p95 |
|---|---:|---:|
| Within Sand | .280747 / 6.056224 / 14.214625 | 5.864617 / 52.885327 / 112.968255 |
| Within Support | .372426 / 3.572364 / 5.189081 | 9.187777 / 26.679274 / 35.445482 |
| Between | 6.593458 / 8.646141 / 10.901425 | 38.527839 / 45.207717 / 93.165832 |

FSR PC1/PC2 explained variance is `.674607/.138203`, with interval overlaps `.118490/.000000`. Combined values are `.983061/.006080` and `.530531/.060015`.

Combined-minus-Pelvis deltas are centroid `+.001161`, 5NN `+.000000`, mixing `+.000000`, and ratio `+3.1961e7`. Only ratio passes its material-improvement threshold. Confirmation is 1/4, below the frozen requirement of at least 3/4, with no directional metric degradation beyond `.05`; `realizable_fsr_material_increment = false`. Discovery was also 1/4, so this non-material outcome replicates. No fusion training occurred.

## 11. Privileged oracle

The separate 16D privileged oracle has centroid separation `1.241039`, balanced 1NN/5NN `1.0/1.0`, opposite/same ratio `2.2445e8`, local mixing `.012500`, and radius inclusion `.186058`. Within-Sand, within-Support, and between p05/median/p95 distances are `.000265/3.914703/8.529778`, `.065618/6.828612/9.620862`, and `4.698646/7.482771/9.120716`.

Oracle PCA explains `.389298/.235419`; its projection-interval overlaps are `.304510/.000000`. This is privileged simulator evidence only, not a runtime sensor or fusion claim.

## 12. V2 Confirmation replay

The exact candidate replayed one split / 88 runs once. The saved ledger contains 65 strict Sand, 16 valid Support, three actual Slips, and four invalid/nonprimary records. There was no second model run or alternate operating point.

Strict Sand result: TN `61`, FP `4`, specificity `61/65 = 93.85%`, and frozen adverse-margin count `29/65 = 44.62%`. Maximum probability has minimum `.643335`, median `.917938`, p75 `.984373`, p90 `.994148`, p95 `.994994`, and maximum `.996258`.

| Frozen margin bin | Runs |
|---|---:|
| `<.90` | 32 |
| `[.90,.95)` | 4 |
| `[.95,.99)` | 17 |
| `>=.99`, streak `<5 ms` | 8 |
| Reflex | 4 |

The three actual moderate Slip runs remain descriptive and outside benign/H1 metrics: max probabilities `.897079`, `.990119` (1 ms `>=.99`), and `.945049`; none produced a 5 ms Reflex.

## 13. Sand specificity

| Group | N | TN | FP | Specificity | Adverse | Median max p | p75 | p90 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All strict Sand | 65 | 61 | 4 | 93.85% | 29 | .917938 | .984373 | .994148 | .994994 | .996258 |
| Mild | 48 | 44 | 4 | 91.67% | 22 | .898964 | .987827 | .994635 | .995412 | .996258 |
| Moderate | 17 | 17 | 0 | 100.00% | 7 | .917938 | .965901 | .981943 | .992235 | .992235 |

The binary and margin results are worse than Discovery's 67/69 specificity and 24/69 adverse rate, but H1 Confirmation is governed by the full frozen rule rather than exact FP equality.

## 14. Mild result

Mild specificity is `44/48 = 91.67%`, with four false Reflexes and `22/48 = 45.83%` adverse runs. Median/p95 max probability is `.898964/.995412`. Margin bins are 25 `<.90`, one `[.90,.95)`, 12 `[.95,.99)`, six subpersistent `>=.99`, and four Reflexes.

Mild localization retains the same aggregate direction: transition-left is 21/38 adverse with four Reflexes versus transition-right 1/10 with zero; right-single precontact is 20/37 adverse with three Reflexes versus left-single 1/10 with zero. The one mild double-support case is adverse and Reflex but is below the frozen per-level support minimum.

## 15. Moderate result

Strict moderate specificity is `17/17 = 100%`, with zero false Reflexes and `7/17 = 41.18%` adverse runs. Median/p95 max probability is `.917938/.992235`. Margin bins are seven `<.90`, three `[.90,.95)`, five `[.95,.99)`, two subpersistent `>=.99`, and zero Reflexes.

The three actual moderate Slips are reported separately in Section 12 and never contaminate moderate benign specificity.

## 16. Support controls

All 16 Support controls are correct, for 100% recall with zero pre-I1 Reflexes.

| Support group | N | Correct | Recall | Pre-I1 | I1→Reflex median (range), ms | Reflex→Support median (range), ms |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 12 | 12 | 100% | 0 | 627 (621–1773) | 19.5 (17–31) |
| Delayed | 4 | 4 | 100% | 0 | 4 (4–4) | 52 (52–52) |
| Concrete | 8 | 8 | 100% | 0 | 623.5 (4–1226) | 22 (18–52) |
| Marble | 8 | 8 | 100% | 0 | 626.5 (4–1773) | 21 (17–52) |
| Left-only | 9 | 9 | 100% | 0 | 627 (4–1773) | 21 (17–52) |
| Right-only | 7 | 7 | 100% | 0 | 624 (621–1228) | 22 (18–31) |

The all-Support I1→Reflex median/range is `625/4–1773 ms`; Reflex→Support is `21.5/17–52 ms`. Support remains a successful control and does not rescue the failed Pelvis criterion.

## 17. Source/speed

| Source | Speed | N | TN | FP | Specificity | Adverse | Median max p | p95 max p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Concrete | .20 | 11 | 11 | 0 | 100.00% | 1 | .898964 | .946510 |
| Concrete | .25 | 10 | 10 | 0 | 100.00% | 4 | .866477 | .988402 |
| Concrete | .30 | 11 | 11 | 0 | 100.00% | 6 | .950812 | .991226 |
| Marble | .20 | 12 | 10 | 2 | 83.33% | 10 | .992942 | .996245 |
| Marble | .25 | 12 | 10 | 2 | 83.33% | 8 | .958810 | .995308 |
| Marble | .30 | 9 | 9 | 0 | 100.00% | 0 | .705801 | .863829 |

Both sources have at least two adverse runs (`Concrete 11`, `Marble 18`), and all three speeds have at least two (`.20: 11`, `.25: 12`, `.30: 6`). Overall speed fraction range is `.245455`, just below the frozen `.25` requirement despite Cramér's V `.203915`, so speed is not a replacement localization. Source/speed interactions are descriptive only.

## 18. Frozen localization replication

| Frozen factor direction | Discovery | Confirmation | Confirmation FP / median / p95 | Cramér's V | Replicated? |
|---|---:|---:|---:|---:|---|
| Transition-left adverse | 24/55 (43.64%) | 26/52 (50.00%) | 4 / .947586 / .995308 | .216645 topology | YES |
| Transition-right adverse | 0/14 (0.00%) | 3/13 (23.08%) | 0 / .834148 / .965426 | .216645 topology | YES |
| Right-single precontact adverse | 24/52 (46.15%) | 25/51 (49.02%) | 3 / .944361 / .994875 | .210398 phase | YES |
| Left-single precontact adverse | 0/14 (0.00%) | 3/13 (23.08%) | 0 / .834148 / .965426 | .210398 phase | YES |

Confirmation fraction ranges are `.269231` for topology and `.259427` for phase, above `.25`; both Cramér's V values exceed `.20`, and each compared level has at least eight runs. The exact frozen direction therefore replicates. The single double-support row is excluded from eligible phase-level comparison.

Topology and phase remain physically coupled and represent one transition region, not two independent causal effects. Other predeclared association checks do not pass: severity range/V is `.046569/.041171`; start is one MID level; width has only one eligible WIDE level. Entry time is uniformly EARLY. Exposure LOW/MID/HIGH has adverse `7/23`, `9/19`, and `13/23`; it is descriptive, not a substitute H1.

Factor-localized Pelvis windows further show that low centroid separation is concentrated in mild (`.218801`), Marble (`.262683`), high-exposure (`.315097`), and transition-left (`.220718`) subsets, while their nearest-neighbor mixing remains low. These were predeclared summaries, not new decision metrics.

Every frozen adverse-margin run, including all four false Reflexes, is listed below. Geometry is `realization; start/width m`; exposure is `band/ms`.

| Run | Severity | Source | Speed | Topology | Phase | Geometry | Exposure | Max p | `>=.99` streak | Reflex |
|---|---|---|---:|---|---|---|---|---:|---:|---|
| `sbmrc_c_bb_c_025_01` | mild | C | .25 | left | right-single | c_l1; .325/.805 | mid/3044 | .982545 | 0 | no |
| `sbmrc_c_bb_c_025_02` | mild | C | .25 | left | right-single | c_l2; .328/.807 | mid/3047 | .987347 | 0 | no |
| `sbmrc_c_bb_c_025_05` | mild | C | .25 | left | right-single | c_l5; .337/.797 | mid/3046 | .984373 | 0 | no |
| `sbmrc_c_bb_c_025_07` | mild | C | .25 | left | right-single | c_l7; .336/.801 | mid/3051 | .989265 | 0 | no |
| `sbmrc_c_bb_c_030_01` | mild | C | .30 | left | right-single | c_l1; .325/.805 | low/2734 | .971169 | 0 | no |
| `sbmrc_c_bb_c_030_02` | mild | C | .30 | left | right-single | c_l2; .328/.807 | low/2723 | .986343 | 0 | no |
| `sbmrc_c_bb_c_030_03` | mild | C | .30 | left | right-single | c_l3; .331/.809 | low/2724 | .950812 | 0 | no |
| `sbmrc_c_bb_c_030_04` | mild | C | .30 | left | right-single | c_l4; .334/.795 | low/2734 | .992549 | 4 | no |
| `sbmrc_c_bb_c_030_05` | mild | C | .30 | left | right-single | c_l5; .337/.797 | low/2721 | .989903 | 0 | no |
| `sbmrc_c_bb_c_030_06` | mild | C | .30 | left | right-single | c_l6; .340/.799 | low/2724 | .954569 | 0 | no |
| `sbmrc_c_bb_m_020_01` | mild | M | .20 | left | double | c_l1; .325/.805 | high/4145 | .996258 | 5 | **yes** |
| `sbmrc_c_bb_m_020_02` | mild | M | .20 | left | right-single | c_l2; .328/.807 | high/3984 | .994617 | 3 | no |
| `sbmrc_c_bb_m_020_03` | mild | M | .20 | left | right-single | c_l3; .331/.809 | high/4168 | .994480 | 2 | no |
| `sbmrc_c_bb_m_020_04` | mild | M | .20 | left | right-single | c_l4; .334/.795 | high/3987 | .996234 | 5 | **yes** |
| `sbmrc_c_bb_m_020_05` | mild | M | .20 | left | right-single | c_l5; .337/.797 | high/3984 | .994678 | 3 | no |
| `sbmrc_c_bb_m_020_06` | mild | M | .20 | left | right-single | c_l6; .340/.799 | high/4077 | .993649 | 3 | no |
| `sbmrc_c_bb_m_025_02` | mild | M | .25 | left | right-single | c_l2; .328/.807 | mid/3351 | .961842 | 0 | no |
| `sbmrc_c_bb_m_025_03` | mild | M | .25 | left | right-single | c_l3; .331/.809 | mid/3337 | .959275 | 0 | no |
| `sbmrc_c_bb_m_025_04` | mild | M | .25 | left | right-single | c_l4; .334/.795 | mid/3351 | .995073 | 9 | **yes** |
| `sbmrc_c_bb_m_025_05` | mild | M | .25 | left | right-single | c_l5; .337/.797 | mid/3360 | .995594 | 6 | **yes** |
| `sbmrc_c_bb_m_025_06` | mild | M | .25 | left | right-single | c_l6; .340/.799 | mid/3349 | .990018 | 1 | no |
| `sbmrc_c_bb_m_025_08` | mild | M | .25 | right | left-single | c_r2; .328/.791 | low/2744 | .958345 | 0 | no |
| `sbmrc_c_nh_c_020_03` | moderate | C | .20 | left | right-single | c03; .341/.821 | high/4287 | .975081 | 0 | no |
| `sbmrc_c_nh_m_020_01` | moderate | M | .20 | left | right-single | c01; .329/.805 | high/4213 | .992235 | 1 | no |
| `sbmrc_c_nh_m_020_02` | moderate | M | .20 | right | left-single | c02; .325/.815 | high/4819 | .968278 | 0 | no |
| `sbmrc_c_nh_m_020_03` | moderate | M | .20 | left | right-single | c03; .341/.821 | high/4336 | .992235 | 1 | no |
| `sbmrc_c_nh_m_020_04` | moderate | M | .20 | right | left-single | c04; .337/.831 | high/5088 | .963525 | 0 | no |
| `sbmrc_c_nh_m_025_03` | moderate | M | .25 | left | right-single | c03; .341/.821 | high/3892 | .965901 | 0 | no |
| `sbmrc_c_nh_m_025_04` | moderate | M | .25 | left | right-single | c04; .347/.829 | high/3907 | .954040 | 0 | no |

## 19. Discovery vs Confirmation

| Metric | Discovery | Confirmation | Replicates frozen direction/rule? |
|---|---:|---:|---|
| Strict Sand specificity | 67/69 (97.10%) | 61/65 (93.85%) | descriptive decrease |
| Mild specificity | 46/48 (95.83%) | 44/48 (91.67%) | descriptive decrease |
| Moderate specificity | 21/21 (100%) | 17/17 (100%) | yes, descriptive |
| Sand median max p | .920693 | .917938 | similar, descriptive |
| Sand p95 max p | .994464 | .994994 | similar, descriptive |
| Sand adverse rate | 24/69 (34.78%) | 29/65 (44.62%) | YES, systematic |
| Transition-left adverse | 24/55 (43.64%) | 26/52 (50.00%) | YES |
| Transition-right adverse | 0/14 (0%) | 3/13 (23.08%) | YES, still lower |
| Right-single adverse | 24/52 (46.15%) | 25/51 (49.02%) | YES |
| Left-single adverse | 0/14 (0%) | 3/13 (23.08%) | YES, still lower |
| Pelvis-window centroid separation | .890733 | .208030 | **NO**, criterion `>=.75` |
| Pelvis-window balanced 1NN | .992754 | 1.000000 | yes, descriptive |
| Pelvis-window balanced 5NN | .992754 | 1.000000 | YES, `>=.80` |
| Pelvis-window local mixing | .007246 | .003077 | YES, `<=.30` |
| FSR material improvements | 1/4 | 1/4 | YES, remains `<3/4` |
| Support recall / pre-I1 | 16/16 / 0 | 16/16 / 0 | YES |

The exact Pelvis replication table makes the sole H1 failure explicit.

| Pelvis-window metric | Discovery | Confirmation | Frozen criterion | Pass? |
|---|---:|---:|---:|---|
| Centroid separation | .890733 | .208030 | `>=.75` | **FAIL** |
| Balanced 1NN | .992754 | 1.000000 | descriptive | — |
| Balanced 5NN | .992754 | 1.000000 | `>=.80` | PASS |
| Opposite/same ratio | 1.9185e9 | 1.9185e9 | `>=1.25` | PASS |
| Local mixing | .007246 | .003077 | `<=.30` | PASS |
| Radius inclusion | .500000 | .500000 | strong mixing `>=.75` | not triggered |
| Reasonable-separation checks | 4/4 | 3/4 | all 4 required | **FAIL** |
| Strong-mixing checks | 0/5 | 1/5 | at least 3 required | remains unsupported |

| Combined-minus-Pelvis metric | Discovery | Confirmation | Frozen material criterion |
|---|---:|---:|---:|
| Centroid delta | +.005296 | +.001161 | `>=+.25` |
| Balanced 5NN delta | +.000000 | +.000000 | `>=+.10` |
| Local-mixing delta | +.000000 | +.000000 | `<=-.15` |
| Ratio delta | +3.1961e7 | +3.1961e7 | `>=+.20` |
| Material count | 1/4 | 1/4 | at least 3/4 |

## 20. Development-history progression

Denominators remain separate, and the historical HOLDOUT contributes binary committed context only.

| Evidence | N | Specificity | Median max p | p95 | Adverse | FP |
|---|---:|---:|---:|---:|---:|---:|
| V2_TRAIN Speed Sand | 36 | 36/36 | .779207 | .935462 | 1 | 0 |
| V2_VALIDATION Speed Sand | 12 | 12/12 | .953668 | .990276 | 6 | 0 |
| Generalization VALIDATION Speed Sand | 6 | 6/6 | .986838 | .993071 | 5 | 0 |
| Fresh recalibrated Discovery | 69 | 67/69 | .920693 | .994464 | 24 | 2 |
| Fresh recalibrated Confirmation | 65 | 61/65 | .917938 | .994994 | 29 | 4 |
| Consumed Generalization HOLDOUT Sand | 6 | historical binary 3/6 | not reopened | — | — | 3 |

Confirmation preserves a thin high-probability margin and adds two false actions relative to Discovery. It does not restore TRAIN-like confidence. That development-margin pattern alone cannot override the failed model-independent centroid criterion.

## 21. H1 replication test

| Frozen H1 requirement | Result | Status |
|---|---|---|
| Frozen Discovery hypothesis is H1 | unchanged | PASS |
| Confirmation population and integrity | 88 total; 65 Sand + 16 Support primary | PASS |
| Systematic adverse pattern | 29/65; both sources; all speeds | PASS |
| Exact topology/phase direction | both directions and association thresholds replicate | PASS |
| Reasonable Pelvis separation | 3/4; centroid `.208030 < .75` | **FAIL** |
| Strong Pelvis mixing unsupported | 1/5, below 3/5 | PASS |
| Realizable FSR material increment unsupported | 1/4, below 3/4 | PASS |
| Support controls strong | 16/16; pre-I1 0 | PASS |
| No hypothesis substitution | no H2/H3 promotion | PASS |

All H1 requirements were conjunctive. Eight pass and one fails, so `all_h1_checks_passed = false`.

## 22. Contradictory evidence

Confirmation strongly reproduces the model-margin pattern, the coupled localization direction, excellent run-disjoint nearest-neighbor agreement, very low local mixing, non-material FSR increment, and perfect Support control. However, under the fixed Discovery scaler, the window within-group RMS rises from `32.164950` to `238.256187` and normalized centroid separation falls from `.890733` to `.208030`. This violates the frozen reasonable-separation rule.

The nearest-neighbor and centroid results are different predeclared properties, not grounds to discard the inconvenient metric. Population and integrity support are sufficient, and the all-required contract resolves the evidence deterministically. The result is therefore valid non-confirmation, not an inconclusive analysis. Confirmation is not repurposed to select H2 or H3.

## 23. Confirmation scientific verdict

`DOMAIN_DIVERSITY_GAP_NOT_CONFIRMED`

The independent split did not reproduce the entire frozen H1 rule. This does not prove a Pelvis observability limitation, a GRU capacity limitation, or an alternate causal hypothesis; those alternatives were not tested for selection on Confirmation.

The immutable semantic freeze is:

`SAND_BENIGN_CONFIRMATION_INTERPRETATION_SHA = 7dc0e2696548446065a35f2466cc3efd9afc35748a60030cd154fef1c95b88b6`

| Frozen object | SHA-256 |
|---|---|
| Confirmation config | `29883f802c0247a21337015037962d1f49c6077244798c86980fc85f9bdd4625` |
| Confirmation implementation | `76aba665b4fffb5edd6ae51db6be36e818edded903874a7ccf7b704ae5ffce56` |
| Discovery scalers | `62935047a67b0ab44265299b5e4bbfb3e012c80347ae96a557ab70a07d65ae38` |
| Confirmation guard | `5e2995cc2a1829c2c78cc0b49bab1d3192823aead0c3b7d126f9793427984238` |
| Pelvis analysis | `d17f6666cb69c151454134e97a8f5b00a9e5aefccbb7e3ff72d430182a950238` |
| FSR/contact analysis | `4df38a71bf5d42f1b23f9a257f459b137e8e0a911c8330ab75af3a9a62b5561c` |
| Privileged oracle | `9c30c6a4636ed5352c592a6ee236c35feed7af51679dbe732ed621f4d18036d0` |
| V2 Confirmation replay | `7485d8dbab6d2ef011513593b0de39469c5d396292587e37386ac5be84afeec1` |
| Factor localization | `0c7f57612900dfde1565e62de3702a2fb3bfff129ffef8875d515f0a3c0ff02b` |
| Discovery/Confirmation replication | `d40dbcb258e3f5a6e13603153a299eba1128e67c6247a38e9e5ec0d8d1ed973e` |
| Confirmation decision | `247463fc8af007548676bae1e94256d325130ff078b888b88d1e1ddcd563713c` |
| Confirmation interpretation file | `94f03d0dd9bce0d64f3c7a8d85cafc29a3e927762863459a25fb6a9575ca3354` |

## 24. Architecture/sensor implication

No architecture or sensor intervention is authorized. Because H1 did not confirm, `ARCHITECTURE_STILL_FAVORS_DATA_DOMAIN` and `PELVIS_ONLY_HAZARD_STILL_PLAUSIBLE` cannot be promoted as confirmed conclusions from this study. The current result also does not establish their opposites.

Realizable FSR remains non-material at 1/4, so `HAZARD_FSR_FUSION_NOT_JUSTIFIED_BY_CURRENT_EVIDENCE` remains the cautious sensor statement. `FINAL_SENSOR_ARCHITECTURE_FROZEN = NO`. No Model V3, LSTM, history, GRU-size, FSR-fusion, or sensor change occurred.

## 25. Confirmation future-use status

`MILD_RECALIBRATED_CONFIRMATION` is `CONSUMED_FOR_FROZEN_H1_REPLICATION`. Its hypothesis-replication role is complete. Default future use is no training, HNM, threshold tuning, persistence tuning, model selection, or direct scenario copying. It can never become fresh final evidence. This study remains development evidence and is not a replacement HOLDOUT.

## 26. Historical HOLDOUT protection

Historical scientific status is unchanged:

- `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`

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
new V2 Discovery replay = 0
V2 Confirmation replay = 1 split / 88 runs
old HOLDOUT payload reads = 0
old HOLDOUT feature reconstruction = 0
old HOLDOUT inference = 0
old HOLDOUT visualization = 0
Confirmation protocol opens = 1
Confirmation payload deserializations = 88
training = 0
```

## 27. Limitations

- This is deterministic MuJoCo evidence, not measured soil, real-robot, safety, or deployment evidence.
- One model-independent anchor per run does not exhaust each trace.
- Pooled distance metrics do not prove Bayes separability.
- Topology and precontact phase are physically coupled.
- Exact same-class vectors make the ratio epsilon-bounded and very large; the failed centroid criterion is independent of that ratio.
- The fixed Discovery scaler is scientifically required for replication but exposes substantial Confirmation window dispersion.
- The study Confirmation is consumed development replication, not fresh final evidence.
- Non-confirmation of H1 does not constitute Confirmation-based support for H2 or H3.

Pre-open targeted verification passed `19/19`, and the repository-wide suite passed `148 passed, 1 skipped`. Post-open targeted hash/guard/regression verification passed `19/19`. Final verification also includes `compileall src scripts tests`, Ruff `E9,F63,F7,F82`, `git diff --check`, deterministic artifact hashes, protected model/dataset hashes, and old-HOLDOUT guard integrity.

## 28. Analysis-validity verdict

`SAND_BENIGN_GENERALIZATION_STUDY_CONFIRMATION_ANALYSIS_VALID`

The protocol and code were frozen before access; exact identities and recovered Discovery scaler hashes matched; the guard changed once; every Confirmation payload was deserialized once; exact V2 replayed once; no tuning, training, metric change, or hypothesis substitution occurred; and the historical HOLDOUT remained untouched. Scientific non-confirmation is not analysis invalidity.

## 29. Recommended next milestone

`SAND_GENERALIZATION_HYPOTHESIS_REVIEW`

The smallest next step is an evidence-preserving review that reconciles Discovery H1 with valid Confirmation non-replication without treating Confirmation as a free hypothesis-search set. Do not reopen or rerun Confirmation, retrospectively promote H2/H3, train, generate new Sand training data, perform HNM, change architecture/sensors, tune threshold/persistence, create a new final HOLDOUT, or start deployment work in this milestone.
