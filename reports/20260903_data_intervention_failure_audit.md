# Data-Intervention Failure Audit and Physical-Domain Redesign

## 1. Purpose and scientific boundary

This milestone explains why the fresh `sand_factor_conditioned_development_20260903` corpus failed its physical generation gates and freezes the smallest viable replacement design. The failed corpus and the two calibration pilots were analyzed only through actual-physics metadata. No Hazard model, model probability, normalized feature, or model-guided parameter choice entered the audit.

The selected hypothesis remains exactly `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS`. It is `NOT YET MODEL-TESTED`: this audit studies simulator/controller physical feasibility, not Hazard-model behavior.

## 2. Starting state

The expected scientific base was `eb2dac1b9452307a6b83ca601471db07f1c7ab47`. At the first repository check, a separately authorized deployment task had advanced local `HEAD` to `9b1f008c0a665dd02f2243f92203e66406fcaeb8`, while `origin/main` was still `eb2dac1...`, and had uncommitted release/docs/test changes. Those changes were preserved. That parallel task subsequently stabilized the shared repository at `cb1b83371c3ff8be26b44e85fb709f6449c7e1f7` before the physical pilots, then added `ff9873d` and `314ded6` while this audit was being finalized. All parallel commits were preserved, and the scientific input dataset remained byte exact throughout.

Input integrity:

| Input | SHA-256 | Result |
|---|---|---|
| Failed manifest | `d4ec98b5c2bec6c9009b5da958195935c46dd57489c8f0f15e08258ae84d6998` | PASS |
| Failed physical audit | `b334f615323a2d770f5b8c35b251368f10f1f00f481868933971cb5f00d67dee` | PASS |
| Failed dataset freeze file | `36f5c93c5dba3793101d1401ab88864fe8833d9904f7593a3a6593c7a978327f` | PASS |
| Failed semantic freeze | `4906682f9366bad572baeb529db81ca1d5b1b2878f1cd2e4782999e2588cd549` | PASS |
| Intervention config | `540034673d1703adce000182b73e2dc4c4bf8856e534e7489ea66bef6522246e` | PASS |
| Intervention report | `b919131ea4d3110070e8273e36dbbe4f5f5a6e509c1f4544d047934ae0bd36ef` | PASS |

The consumed Generalization HOLDOUT guard remained `guard_after=1`, `scientific_open_count=1`. HOLDOUT payload reads, inference, feature reconstruction, and visualization were all zero.

## 3. Failed intervention recap

All 162 planned runs completed once. The physical audit classifies every record with the frozen precedence `PRETARGET_FALL`, `TARGET_FOLLOWING_FALL_CENSOR`, `OTHER_INVALID`, then valid physical outcomes.

| Class | Total | FACTOR_TRAIN | FACTOR_VALIDATION |
|---|---:|---:|---:|
| STRICT_SAND_BENIGN | 42 | 30 | 12 |
| SUPPORT | 39 | 27 | 12 |
| SLIP | 5 | 4 | 1 |
| DUAL_HAZARD | 2 | 2 | 0 |
| PRETARGET_FALL | 30 | 15 | 15 |
| TARGET_FOLLOWING_FALL_CENSOR | 42 | 28 | 14 |
| OTHER_INVALID | 2 | 2 | 0 |
| Total | 162 | 108 | 54 |

Objective eligibility remained 81/162, comprising 42 strict Sand and 39 Support. Nothing was relabeled, removed, replaced, or backfilled.

## 4. Severity and control decomposition

| Split | Group | Planned | Strict | Support | Slip | Dual | Pretarget | Post-target | Other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FACTOR_TRAIN | mild Sand | 48 | 15 | 0 | 0 | 0 | 7 | 26 | 0 |
| FACTOR_TRAIN | moderate Sand | 24 | 15 | 0 | 4 | 0 | 3 | 2 | 0 |
| FACTOR_TRAIN | ordinary Support | 24 | 0 | 18 | 0 | 1 | 3 | 0 | 2 |
| FACTOR_TRAIN | delayed Support | 12 | 0 | 9 | 0 | 1 | 2 | 0 | 0 |
| FACTOR_VALIDATION | mild Sand | 24 | 6 | 0 | 0 | 0 | 6 | 12 | 0 |
| FACTOR_VALIDATION | moderate Sand | 12 | 6 | 0 | 1 | 0 | 3 | 2 | 0 |
| FACTOR_VALIDATION | ordinary Support | 12 | 0 | 8 | 0 | 0 | 4 | 0 | 0 |
| FACTOR_VALIDATION | delayed Support | 6 | 0 | 4 | 0 | 0 | 2 | 0 | 0 |

Mild geometry accounts for 51/74 invalid records. Moderate contributes 10/74 invalid records and five genuine Slip outcomes. The controls contribute 13/74 invalid records plus two genuine dual-Hazard outcomes. Moderate was not the dominant invalidity source, but its 5/36 Slip rate shows that it remains a useful boundary comparator rather than a generic benign source.

## 5. Source-speed failure map

The predeclared cell rule is `STABLE >=75%`, `MARGINAL >=50% and <75%`, and `UNSTABLE <50%`, using strict Sand plus qualified Support as the numerator.

| Split | Source/speed | Planned | Strict Sand | Support | Slip/Dual | Pretarget fall | Post-target censor | Other | Eligible rate | Region |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FACTOR_TRAIN | Concrete/.20 | 18 | 3 | 4 | 1 | 6 | 4 | 0 | 38.9% | UNSTABLE |
| FACTOR_TRAIN | Concrete/.25 | 18 | 3 | 4 | 1 | 0 | 8 | 2 | 38.9% | UNSTABLE |
| FACTOR_TRAIN | Concrete/.30 | 18 | 8 | 6 | 1 | 0 | 3 | 0 | 77.8% | STABLE |
| FACTOR_TRAIN | Marble/.20 | 18 | 4 | 3 | 1 | 6 | 4 | 0 | 38.9% | UNSTABLE |
| FACTOR_TRAIN | Marble/.25 | 18 | 6 | 4 | 1 | 3 | 4 | 0 | 55.6% | MARGINAL |
| FACTOR_TRAIN | Marble/.30 | 18 | 6 | 6 | 1 | 0 | 5 | 0 | 66.7% | MARGINAL |
| FACTOR_VALIDATION | Concrete/.20 | 9 | 0 | 1 | 1 | 5 | 2 | 0 | 11.1% | UNSTABLE |
| FACTOR_VALIDATION | Concrete/.25 | 9 | 2 | 2 | 0 | 1 | 4 | 0 | 44.4% | UNSTABLE |
| FACTOR_VALIDATION | Concrete/.30 | 9 | 3 | 3 | 0 | 0 | 3 | 0 | 66.7% | MARGINAL |
| FACTOR_VALIDATION | Marble/.20 | 9 | 1 | 1 | 0 | 5 | 2 | 0 | 22.2% | UNSTABLE |
| FACTOR_VALIDATION | Marble/.25 | 9 | 3 | 2 | 0 | 4 | 0 | 0 | 55.6% | MARGINAL |
| FACTOR_VALIDATION | Marble/.30 | 9 | 3 | 3 | 0 | 0 | 3 | 0 | 66.7% | MARGINAL |

The collapse is not one material-only effect. Both `.20` source cells are unstable, Concrete/.25 is unstable through a different immediate-censor mode, and `.30` is substantially better. This is source-speed-conditioned controller/contact-sequence modulation of a joint geometry error.

## 6. Topology/phase failure map

Topology and phase are physically coupled here: transition-left realizes right-single precontact and transition-right realizes left-single precontact. `NO_SUPPORT` means the run fell before target contact, not an independent phase assignment.

| Split | Topology / measured phase | Severity | N | Strict | Support | Slip/Dual | Pretarget | Post-target | Other | Eligible rate | Region |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TRAIN | left / NO_SUPPORT | mild | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 0.0% | UNSTABLE |
| TRAIN | left / NO_SUPPORT | moderate | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0.0% | UNSTABLE |
| TRAIN | left / NO_SUPPORT | Support | 4 | 0 | 0 | 0 | 4 | 0 | 0 | 0.0% | UNSTABLE |
| TRAIN | left / right-single | mild | 22 | 11 | 0 | 0 | 0 | 11 | 0 | 50.0% | MARGINAL |
| TRAIN | left / right-single | moderate | 12 | 11 | 0 | 1 | 0 | 0 | 0 | 91.7% | STABLE |
| TRAIN | left / right-single | Support | 20 | 0 | 18 | 2 | 0 | 0 | 0 | 90.0% | STABLE |
| TRAIN | right / NO_SUPPORT | mild | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0.0% | UNSTABLE |
| TRAIN | right / NO_SUPPORT | moderate | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0.0% | UNSTABLE |
| TRAIN | right / NO_SUPPORT | Support | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0.0% | UNSTABLE |
| TRAIN | right / left-single | mild | 19 | 4 | 0 | 0 | 0 | 15 | 0 | 21.1% | UNSTABLE |
| TRAIN | right / left-single | moderate | 9 | 4 | 0 | 3 | 0 | 2 | 0 | 44.4% | UNSTABLE |
| TRAIN | right / left-single | Support | 11 | 0 | 9 | 0 | 0 | 0 | 2 | 81.8% | STABLE |
| VALIDATION | left / NO_SUPPORT | mild | 4 | 0 | 0 | 0 | 4 | 0 | 0 | 0.0% | UNSTABLE |
| VALIDATION | left / NO_SUPPORT | moderate | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0.0% | UNSTABLE |
| VALIDATION | left / NO_SUPPORT | Support | 4 | 0 | 0 | 0 | 4 | 0 | 0 | 0.0% | UNSTABLE |
| VALIDATION | left / right-single | mild | 10 | 6 | 0 | 0 | 0 | 4 | 0 | 60.0% | MARGINAL |
| VALIDATION | left / right-single | moderate | 5 | 4 | 0 | 0 | 0 | 1 | 0 | 80.0% | STABLE |
| VALIDATION | left / right-single | Support | 8 | 0 | 8 | 0 | 0 | 0 | 0 | 100.0% | STABLE |
| VALIDATION | right / NO_SUPPORT | mild | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0.0% | UNSTABLE |
| VALIDATION | right / NO_SUPPORT | moderate | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0.0% | UNSTABLE |
| VALIDATION | right / NO_SUPPORT | Support | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0.0% | UNSTABLE |
| VALIDATION | right / left-single | mild | 8 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0% | UNSTABLE |
| VALIDATION | right / left-single | moderate | 4 | 2 | 0 | 1 | 0 | 1 | 0 | 50.0% | MARGINAL |
| VALIDATION | right / left-single | Support | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 100.0% | STABLE |

The adverse manifold was not the yield problem by itself. With contact realized, left/right-single moderate and Support cells were mostly stable. The strongest systematic failure was mild transition-right/left-single under the over-long patch; nevertheless the redesigned corpus retains that comparison manifold rather than removing it.

## 7. Geometry and exposure analysis

Values are `minimum / median / maximum`. They are simulator/policy design findings, not universal thresholds.

| Outcome | Start m | Width m | Exit m | Target exposure ms | Entry ms | Topology | Phase |
|---|---|---|---|---|---|---|---|
| Strict Sand | .318/.328/.339 | .842/.850/.886 | 1.160/1.185/1.223 | 1691/3616/5186 | 1220/1227/2114 | left 32, right 10 | right-single 32, left-single 10 |
| Pretarget fall | .320/.330/.338 | .850/.866/.882 | 1.180/1.195/1.218 | 0/0/0 | N/A | left 22, right 8 | NO_SUPPORT 30 |
| Post-target fall/censor | .318/.328/.340 | .842/.858/.902 | 1.160/1.187/1.242 | 338/692/3569 | 1220/1503/2112 | left 16, right 26 | right-single 16, left-single 26 |
| Slip | .319/.327/.331 | .842/.854/.870 | 1.165/1.185/1.195 | 1846/3726/5333 | 1220/1504/1511 | left 1, right 4 | right-single 1, left-single 4 |
| Dual Hazard | .327/.328/.328 | .842/.842/.842 | 1.169/1.169/1.170 | 4349/4469/4589 | 1220/1515/1810 | left 2 | right-single 2 |
| Support | .319/.329/.336 | .842/.850/.870 | 1.165/1.185/1.196 | 1817/3178/6579 | 1220/1227/2114 | left 26, right 13 | right-single 26, left-single 13 |
| Other invalid | .319/.325/.331 | .846/.850/.854 | 1.165/1.175/1.185 | 347/352/357 | 1507/1508/1508 | right 2 | left-single 2 |

Width alone does not separate outcomes: strict Sand spans `.842–.886`, overlapping every failed class. The actionable variable is the family-specific joint `(start, width, exit)` selection and the resulting discrete contact sequence/exposure. The failed design applied `.842–.902` widths and exits through `1.242` to families whose prior viable coordinates were substantially shorter.

## 8. Pretarget falls

All 30 pretarget cases fell before any actual target contact: target time is `NOT_REACHED`, target exposure is zero, and a target-relative delta does not exist. They cannot be rescued by labeling or observation changes and are excluded from the future domain.

| Run | Split | Fall ms | Source/speed | Topology | Severity | Start/width/exit | Realization |
|---|---|---:|---|---|---|---|---|
| sfci_t_sml_c_020_02 | TRAIN | 1613 | Concrete/.20 | left | mild | .330/.850/1.180 | tl02 |
| sfci_t_sml_c_020_03 | TRAIN | 1613 | Concrete/.20 | left | mild | .334/.858/1.192 | tl03 |
| sfci_t_sml_c_020_04 | TRAIN | 1613 | Concrete/.20 | left | mild | .338/.866/1.204 | tl04 |
| sfci_t_smd_c_020_02 | TRAIN | 1613 | Concrete/.20 | left | moderate | .335/.850/1.185 | tl02 |
| sfci_t_osp_c_020_02 | TRAIN | 1613 | Concrete/.20 | left | Support | .335/.850/1.185 | l02 |
| sfci_t_dsp_c_020_02 | TRAIN | 1613 | Concrete/.20 | left | Support | .336/.850/1.186 | d02 |
| sfci_t_sml_m_020_02 | TRAIN | 1622 | Marble/.20 | left | mild | .330/.850/1.180 | tl02 |
| sfci_t_sml_m_020_03 | TRAIN | 1622 | Marble/.20 | left | mild | .334/.858/1.192 | tl03 |
| sfci_t_sml_m_020_04 | TRAIN | 1622 | Marble/.20 | left | mild | .338/.866/1.204 | tl04 |
| sfci_t_smd_m_020_02 | TRAIN | 1622 | Marble/.20 | left | moderate | .335/.850/1.185 | tl02 |
| sfci_t_osp_m_020_02 | TRAIN | 1622 | Marble/.20 | left | Support | .335/.850/1.185 | l02 |
| sfci_t_dsp_m_020_02 | TRAIN | 1622 | Marble/.20 | left | Support | .336/.850/1.186 | d02 |
| sfci_t_sml_m_025_08 | TRAIN | 1316 | Marble/.25 | right | mild | .330/.866/1.196 | tr04 |
| sfci_t_smd_m_025_04 | TRAIN | 1316 | Marble/.25 | right | moderate | .331/.854/1.185 | tr02 |
| sfci_t_osp_m_025_04 | TRAIN | 1316 | Marble/.25 | right | Support | .331/.854/1.185 | r02 |
| sfci_v_sml_c_020_01 | VALIDATION | 1613 | Concrete/.20 | left | mild | .328/.874/1.202 | tl01 |
| sfci_v_sml_c_020_02 | VALIDATION | 1613 | Concrete/.20 | left | mild | .336/.882/1.218 | tl02 |
| sfci_v_smd_c_020_01 | VALIDATION | 1613 | Concrete/.20 | left | moderate | .329/.866/1.195 | tl01 |
| sfci_v_osp_c_020_01 | VALIDATION | 1613 | Concrete/.20 | left | Support | .329/.866/1.195 | l01 |
| sfci_v_dsp_c_020_01 | VALIDATION | 1613 | Concrete/.20 | left | Support | .330/.866/1.196 | d01 |
| sfci_v_osp_c_025_02 | VALIDATION | 1318 | Concrete/.25 | right | Support | .325/.870/1.195 | r01 |
| sfci_v_sml_m_020_01 | VALIDATION | 1622 | Marble/.20 | left | mild | .328/.874/1.202 | tl01 |
| sfci_v_sml_m_020_02 | VALIDATION | 1622 | Marble/.20 | left | mild | .336/.882/1.218 | tl02 |
| sfci_v_smd_m_020_01 | VALIDATION | 1622 | Marble/.20 | left | moderate | .329/.866/1.195 | tl01 |
| sfci_v_osp_m_020_01 | VALIDATION | 1622 | Marble/.20 | left | Support | .329/.866/1.195 | l01 |
| sfci_v_dsp_m_020_01 | VALIDATION | 1622 | Marble/.20 | left | Support | .330/.866/1.196 | d01 |
| sfci_v_sml_m_025_03 | VALIDATION | 1316 | Marble/.25 | right | mild | .320/.874/1.194 | tr01 |
| sfci_v_sml_m_025_04 | VALIDATION | 1316 | Marble/.25 | right | mild | .328/.882/1.210 | tr02 |
| sfci_v_smd_m_025_02 | VALIDATION | 1316 | Marble/.25 | right | moderate | .325/.870/1.195 | tr01 |
| sfci_v_osp_m_025_02 | VALIDATION | 1316 | Marble/.25 | right | Support | .325/.870/1.195 | r01 |

Repeated fall times across mild, moderate, and Support mechanics in the same source-speed/geometric region show that these are approach/contact-sequence failures, not target-label errors.

## 9. Post-target fall/censor analysis

The predeclared taxonomy yields 24 `IMMEDIATE_TARGET_INDUCED_INSTABILITY` cases and 18 `SHORT_LIVED_BENIGN_THEN_LATER_FALL` cases. There are zero no-fall strict-last-contact cases and zero true horizon cases. Thus all 42 are actual fall censors; extending the observation horizon would not recover any of them.

`Post` is valid observation after the last target contact; the frozen requirement is 1000 ms.

| Run | Split | Target/fall ms | From target | Post | Source/speed | Topology/phase | Severity | Start/width/exit | Exposure | Class |
|---|---|---|---:|---:|---|---|---|---|---:|---|
| sfci_t_sml_c_020_05 | TRAIN | 1499/7264 | 5765 | 56 | Concrete/.20 | right/left-single | mild | .318/.842/1.160 | 3518 | LATER |
| sfci_t_sml_c_020_06 | TRAIN | 1503/2448 | 945 | 0 | Concrete/.20 | right/left-single | mild | .322/.850/1.172 | 698 | IMMEDIATE |
| sfci_t_sml_c_020_07 | TRAIN | 1504/2446 | 942 | 0 | Concrete/.20 | right/left-single | mild | .326/.858/1.184 | 693 | IMMEDIATE |
| sfci_t_sml_c_020_08 | TRAIN | 1504/2443 | 939 | 0 | Concrete/.20 | right/left-single | mild | .330/.866/1.196 | 690 | IMMEDIATE |
| sfci_t_sml_c_025_01 | TRAIN | 1220/2153 | 933 | 0 | Concrete/.25 | left/right-single | mild | .326/.842/1.168 | 647 | IMMEDIATE |
| sfci_t_sml_c_025_02 | TRAIN | 1220/2153 | 933 | 0 | Concrete/.25 | left/right-single | mild | .330/.842/1.172 | 647 | IMMEDIATE |
| sfci_t_sml_c_025_03 | TRAIN | 1220/2153 | 933 | 0 | Concrete/.25 | left/right-single | mild | .334/.842/1.176 | 647 | IMMEDIATE |
| sfci_t_sml_c_025_04 | TRAIN | 1220/2153 | 933 | 0 | Concrete/.25 | left/right-single | mild | .338/.842/1.180 | 647 | IMMEDIATE |
| sfci_t_sml_c_025_05 | TRAIN | 1220/2153 | 933 | 0 | Concrete/.25 | left/right-single | mild | .326/.850/1.176 | 647 | IMMEDIATE |
| sfci_t_sml_c_025_06 | TRAIN | 1220/2153 | 933 | 0 | Concrete/.25 | left/right-single | mild | .330/.850/1.180 | 647 | IMMEDIATE |
| sfci_t_sml_c_025_07 | TRAIN | 1220/2153 | 933 | 0 | Concrete/.25 | left/right-single | mild | .334/.850/1.184 | 647 | IMMEDIATE |
| sfci_t_sml_c_025_08 | TRAIN | 1220/2153 | 933 | 0 | Concrete/.25 | left/right-single | mild | .338/.850/1.188 | 647 | IMMEDIATE |
| sfci_t_sml_c_030_03 | TRAIN | 1227/4542 | 3315 | 0 | Concrete/.30 | left/right-single | mild | .334/.858/1.192 | 2476 | LATER |
| sfci_t_sml_c_030_08 | TRAIN | 1508/4326 | 2818 | 0 | Concrete/.30 | right/left-single | mild | .330/.866/1.196 | 1771 | LATER |
| sfci_t_smd_c_030_04 | TRAIN | 1508/4328 | 2820 | 0 | Concrete/.30 | right/left-single | moderate | .331/.854/1.185 | 2081 | LATER |
| sfci_t_sml_m_020_05 | TRAIN | 2112/5492 | 3380 | 0 | Marble/.20 | right/left-single | mild | .318/.842/1.160 | 2227 | LATER |
| sfci_t_sml_m_020_06 | TRAIN | 1506/7274 | 5768 | 72 | Marble/.20 | right/left-single | mild | .322/.850/1.172 | 3439 | LATER |
| sfci_t_sml_m_020_07 | TRAIN | 1507/7280 | 5773 | 52 | Marble/.20 | right/left-single | mild | .326/.858/1.184 | 3545 | LATER |
| sfci_t_sml_m_020_08 | TRAIN | 1507/2437 | 930 | 0 | Marble/.20 | right/left-single | mild | .330/.866/1.196 | 684 | IMMEDIATE |
| sfci_t_sml_m_025_05 | TRAIN | 1509/1866 | 357 | 0 | Marble/.25 | right/left-single | mild | .318/.842/1.160 | 356 | IMMEDIATE |
| sfci_t_sml_m_025_06 | TRAIN | 1509/1866 | 357 | 0 | Marble/.25 | right/left-single | mild | .322/.850/1.172 | 356 | IMMEDIATE |
| sfci_t_sml_m_025_07 | TRAIN | 1511/1850 | 339 | 0 | Marble/.25 | right/left-single | mild | .326/.858/1.184 | 338 | IMMEDIATE |
| sfci_t_smd_m_025_03 | TRAIN | 1509/1865 | 356 | 0 | Marble/.25 | right/left-single | moderate | .319/.846/1.165 | 355 | IMMEDIATE |
| sfci_t_sml_m_030_02 | TRAIN | 1227/4539 | 3312 | 0 | Marble/.30 | left/right-single | mild | .330/.850/1.180 | 2459 | LATER |
| sfci_t_sml_m_030_03 | TRAIN | 1227/4537 | 3310 | 0 | Marble/.30 | left/right-single | mild | .334/.858/1.192 | 2457 | LATER |
| sfci_t_sml_m_030_06 | TRAIN | 1504/4280 | 2776 | 0 | Marble/.30 | right/left-single | mild | .322/.850/1.172 | 1790 | LATER |
| sfci_t_sml_m_030_07 | TRAIN | 1509/4276 | 2767 | 0 | Marble/.30 | right/left-single | mild | .326/.858/1.184 | 1766 | LATER |
| sfci_t_sml_m_030_08 | TRAIN | 1511/4275 | 2764 | 0 | Marble/.30 | right/left-single | mild | .330/.866/1.196 | 1767 | LATER |
| sfci_v_sml_c_020_03 | VALIDATION | 1503/2448 | 945 | 0 | Concrete/.20 | right/left-single | mild | .320/.874/1.194 | 698 | IMMEDIATE |
| sfci_v_sml_c_020_04 | VALIDATION | 1504/2443 | 939 | 0 | Concrete/.20 | right/left-single | mild | .328/.882/1.210 | 690 | IMMEDIATE |
| sfci_v_sml_c_025_01 | VALIDATION | 1220/2153 | 933 | 0 | Concrete/.25 | left/right-single | mild | .328/.878/1.206 | 647 | IMMEDIATE |
| sfci_v_sml_c_025_02 | VALIDATION | 1220/2153 | 933 | 0 | Concrete/.25 | left/right-single | mild | .332/.886/1.218 | 647 | IMMEDIATE |
| sfci_v_sml_c_025_03 | VALIDATION | 1220/2153 | 933 | 0 | Concrete/.25 | left/right-single | mild | .336/.894/1.230 | 648 | IMMEDIATE |
| sfci_v_sml_c_025_04 | VALIDATION | 1220/2153 | 933 | 3 | Concrete/.25 | left/right-single | mild | .340/.902/1.242 | 647 | IMMEDIATE |
| sfci_v_sml_c_030_03 | VALIDATION | 1501/1854 | 353 | 0 | Concrete/.30 | right/left-single | mild | .320/.874/1.194 | 351 | IMMEDIATE |
| sfci_v_sml_c_030_04 | VALIDATION | 1508/4276 | 2768 | 0 | Concrete/.30 | right/left-single | mild | .328/.882/1.210 | 1769 | LATER |
| sfci_v_smd_c_030_02 | VALIDATION | 1508/4346 | 2838 | 0 | Concrete/.30 | right/left-single | moderate | .325/.870/1.195 | 2108 | LATER |
| sfci_v_sml_m_020_03 | VALIDATION | 1491/7281 | 5790 | 53 | Marble/.20 | right/left-single | mild | .320/.874/1.194 | 3569 | LATER |
| sfci_v_sml_m_020_04 | VALIDATION | 1507/2437 | 930 | 0 | Marble/.20 | right/left-single | mild | .328/.882/1.210 | 684 | IMMEDIATE |
| sfci_v_sml_m_030_03 | VALIDATION | 1502/4281 | 2779 | 0 | Marble/.30 | right/left-single | mild | .320/.874/1.194 | 1795 | LATER |
| sfci_v_sml_m_030_04 | VALIDATION | 1511/4275 | 2764 | 0 | Marble/.30 | right/left-single | mild | .328/.882/1.210 | 1767 | LATER |
| sfci_v_smd_m_030_01 | VALIDATION | 1227/4013 | 2786 | 0 | Marble/.30 | left/right-single | moderate | .329/.866/1.195 | 2071 | LATER |

## 10. Slip and dual-Hazard contamination

| Run | Designed group | Source/speed | Topology/phase | Start/width/exit | Actual outcome | Peak displacement/spread m |
|---|---|---|---|---|---|---|
| sfci_t_smd_c_020_04 | moderate | Concrete/.20 | right/left-single | .331/.854/1.185 | Slip | .040155/.000000 |
| sfci_t_smd_c_025_01 | moderate | Concrete/.25 | left/right-single | .327/.842/1.169 | Slip | .040167/.000000 |
| sfci_t_smd_c_030_03 | moderate | Concrete/.30 | right/left-single | .319/.846/1.165 | Slip | .040174/.000000 |
| sfci_t_osp_m_020_01 | ordinary Support | Marble/.20 | left/right-single | .327/.842/1.169 | Dual | .040157/.040157 |
| sfci_t_dsp_m_025_01 | delayed Support | Marble/.25 | left/right-single | .328/.842/1.170 | Dual | .040232/.040232 |
| sfci_t_smd_m_030_04 | moderate | Marble/.30 | right/left-single | .331/.854/1.185 | Slip | .040182/.000000 |
| sfci_v_smd_c_020_02 | moderate | Concrete/.20 | right/left-single | .325/.870/1.195 | Slip | .040155/.000000 |

These seven events are genuine boundary outcomes and remain excluded from benign/control membership. Their concentration in moderate or Support mechanics, with zero mild Slip/Dual, does not indicate that the mild hypothesis manifold is intrinsically hazardous. It does justify retaining a contamination ceiling and reducing moderate's proportion.

## 11. Support review

Planned Support was 54: ordinary 36 and delayed 18. Qualified actual Support was 39: ordinary 26 and delayed 13. Ordinary invalid controls were seven pretarget falls and two physical mismatches; delayed invalid controls were four pretarget falls. The remaining two controls realized dual Hazard.

The two physical mismatches were Concrete/.25 transition-right ordinary controls at `.319/.846` and `.331/.854`; both reached the target but produced neither I1 nor established Support. These are actual non-Support outcomes, not label bugs.

The prior mild-recalibrated domain produced ordinary Support 24/24 and delayed Support 8/8. Therefore Support semantics and mechanics are unchanged in the redesign; only fresh coordinates are returned to the canonical, shorter control ranges.

## 12. Comparison with the viable mild domain

Every one of the 72 failed-design mild scenarios was outside the topology-specific proven envelope: 21 strict, 13 pretarget falls, and 38 post-target fall censors. In contrast, the prior mild-recalibrated study produced 96/96 strict mild Sand.

| Dimension | Prior viable mild-recalibrated | Failed factor-conditioned | New recalibrated factor-conditioned |
|---|---|---|---|
| Left start | .325–.342 | .326–.340 | .325–.356; C/.25 .336–.345 selected |
| Left width | .795–.809 | .842–.902 | .777–.809; C/.25 .765–.788 selected |
| Left exit | 1.123–1.147 | 1.168–1.242 | 1.121–1.139; C/.25 1.103–1.129 selected |
| Right start | .324–.330 | .318–.330 | .328–.340 selected |
| Right width | .791–.794 | .842–.882 | .779–.789 selected |
| Right exit | 1.115–1.123 | 1.160–1.210 | 1.113–1.124 selected |
| Topology | both except C/.25 left-only | both except C/.25 left-only | both except C/.25 left-only |
| Coupled phase | left/right-single; right/left-single | same when target reached | same, no independent claim |
| Severity | mild plus separate moderate | mild plus moderate | mild primary; moderate reduced proportion |
| Source/speed | all six | all six | all six in both splits |
| Physical validity | mild 96/96 strict | mild 21/72 strict | pilot-supported; future gates required |
| Scientific factor coverage | development study | factor coverage but invalid yield | explicit balanced adverse/comparison coverage |

The failed redesign expanded patch width by at least 33 mm beyond the prior left maximum and by at least 48 mm beyond the prior right maximum. Exits were correspondingly later. Factor conditioning did not require that physical expansion; it accidentally reintroduced the unstable wide-patch joint domain.

## 13. Root-cause interpretation

The smallest defensible mechanism is:

`JOINT_GEOMETRY_CONTACT_SEQUENCE_INSTABILITY_OUTSIDE_FAMILY_SPECIFIC_VIABLE_ENVELOPES`

This is a narrow multi-factor physical manifold, not a width-only rule:

1. Mild geometry was outside the previously viable topology-specific joint envelope in 72/72 cases.
2. All 72 censor invalids were actual falls: 30 before target, 24 within 1000 ms after first target, and 18 after a longer but ultimately unstable target sequence.
3. Failure timing is source-speed conditioned: `.20` pretarget falls recur at 1613 ms on Concrete and 1622 ms on Marble, while Concrete/.25 produces a separate 933 ms post-entry collapse.
4. The comparison topology is especially sensitive under the failed wide/late domain, but deleting it would remove the scientific factor being tested.
5. Moderate and Support families were also moved away from their prior family-specific anchors, explaining their secondary losses.
6. The physical labeler behaved consistently; exact frozen hashes, fall/target relations, outcome precedence, and two explicit Support intent mismatches show no generator or label-accounting bug.

## 14. Model-blind physical calibration

Calibration was required because the prior exact/near-exclusion shadows left insufficient fresh coordinate capacity inside the narrow proven right envelope. Two pre-frozen batches used 32/64 allowed simulations.

| Batch | Scientific question | N | Strict Sand | Support | Slip/Dual | Invalid | Conclusion |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | Does an exit-preserving later-start/narrower-width strip work across all cells and both manifolds? | 24 | 22 | 0 | 0 | 2 | Overall and both manifolds viable; C/.25 cell gate 2/4 failed |
| 2 | Can C/.25 left recover ≥75% when width/exit are reduced? | 8 | 7 | 0 | 0 | 1 | PASS; low-exit 4/4, high-exit 3/4 |
| Aggregate | — | 32 | 29 | 0 | 0 | 3 | 90.625% strict; no pretarget fall or Hazard |

Batch 1 source-speed strict counts were Concrete `.20/.25/.30 = 4/2/4` and Marble `4/4/4`. Its adverse manifold was 12/14 strict and comparison was 10/10. Batch 2 contributed 7/8 additional C/.25 adverse runs. All three pilot invalids were immediate post-target fall censors; no pilot produced a pretarget fall, Slip, dual Hazard, model output, replacement, or backfill.

The C/.25 failure corner is localized: batch-1 `.356/.791/1.147` and `.352/.787/1.139`, plus batch-2 `.354/.789/1.143`, failed at about 647 ms exposure. Batch-2 points through start `.352` and exit `1.137` were 7/7 strict. The future selected points stay below that corner.

Pilot exact signatures, historical exact/near overlap, and within-batch near overlap are zero. Pilot payloads are calibration-only and prohibited from future training or validation.

## 15. Stable physical envelope

The following are simulator/policy-specific bounded design envelopes. They are not universal Sand thresholds and remain subject to future generation gates.

| Family | Topology/cell | Frozen joint envelope or selected strip |
|---|---|---|
| Mild | left, five non-C/.25 cells | start `.325–.356`, width `.777–.809`, exit `1.121–1.147` |
| Mild | right, five non-C/.25 cells | start `.324–.344`, width `.779–.794`, exit `1.113–1.125` |
| Mild | C/.25 left-only | start `.336–.352`, width `.765–.789`, exit `1.101–1.137`; exclude late/high-exit corner |
| Moderate | five standard cells | fresh left diagnostic points `.326/.779–.799` inside prior strict region |
| Moderate | C/.25 | fresh left diagnostic points `.329/.851–.871` between fresh strict outcomes |
| Ordinary Support | left/right | unchanged mechanics, fresh points inside prior successful short-patch control region |
| Delayed Support | left-only | unchanged mechanics, fresh points inside prior successful staged range |

## 16. Factor-conditioned hypothesis preservation

The redesign preserves the scientific comparison. In each split, every nonexception source-speed cell contains both transition-left/right-single-precontact adverse runs and transition-right/left-single-precontact comparison runs. Concrete/.25 remains left-only because prior calibration found its right topology physically infeasible. Topology and phase are coupled by the simulator trajectory; this is explicitly one manifold, not two independently manipulated factors.

Mild planned factor counts are:

| Split | Non-C/.25 cell, each | C/.25 | Aggregate adverse | Aggregate comparison |
|---|---|---|---:|---:|
| FACTOR_TRAIN | 6 adverse + 6 comparison | 12 adverse | 42 | 30 |
| FACTOR_VALIDATION | 3 adverse + 3 comparison | 6 adverse | 21 | 15 |

## 17. Redesigned corpus

The frozen future dataset is `sand_factor_conditioned_development_recalibrated_20260903`. It is not generated in this milestone.

| Split | Mild | Moderate | Ordinary Support | Delayed Support | Total |
|---|---:|---:|---:|---:|---:|
| FACTOR_TRAIN | 72 | 24 | 24 | 12 | 132 |
| FACTOR_VALIDATION | 36 | 12 | 12 | 6 | 66 |
| Total | 108 | 36 | 36 | 18 | 198 |

Every source-speed cell independently receives TRAIN `12 mild + 4 moderate + 4 ordinary + 2 delayed = 22` and VALIDATION `6 mild + 2 moderate + 2 ordinary + 1 delayed = 11`. VALIDATION uses disjoint geometry and is not a near-copy of TRAIN.

Moderate decision: `KEEP_MODERATE_REDUCED`. Absolute moderate count remains 36, but its corpus share decreases from 22.22% to 18.18%. This retains a boundary-severity control without spending the primary mild factor-coverage budget on the less reliable moderate domain. Moderate remains adverse-direction diagnostic-only; the complete adverse/comparison test is carried by mild runs.

Expected yield uses the 29/32 mild pilot, prior 38/48 moderate physical yield, and prior successful canonical Support controls. The expected objective-valid total is comfortably above the frozen minimum 165; the design does not depend on optimistic yield equaling the gate.

## 18. Frozen generation gates

| Gate | Frozen requirement |
|---|---|
| Complete execution | exactly 198/198 |
| Overall objective valid | ≥165 |
| Pretarget fall | ≤4 |
| Target-following fall censor | ≤14 |
| Designed-Sand Slip + dual | ≤8 total; TRAIN ≤5; VALIDATION ≤3 |
| TRAIN strict Sand | ≥76 |
| TRAIN mild / moderate | ≥62 / ≥14 |
| TRAIN ordinary / delayed Support | ≥22 / ≥10 |
| TRAIN strict Sand per source-speed | ≥12 in each of six cells |
| TRAIN mild adverse / comparison | ≥36 / ≥25 |
| VALIDATION strict Sand | ≥36 |
| VALIDATION mild / moderate | ≥31 / ≥7 |
| VALIDATION ordinary / delayed Support | ≥11 / ≥5 |
| VALIDATION strict Sand per source-speed | ≥6 in each of six cells |
| VALIDATION mild adverse / comparison | ≥18 / ≥12 |
| Topology/phase | both manifolds in every nonexception cell and both splits; C/.25 left-only |
| Physical signature uniqueness | ≥0.80 |
| Historical exact / forbidden-near / run-ID overlap | 0 / 0 / 0 |
| Cross-split exact / forbidden-near overlap | 0 / 0 |
| Model output during generation | 0 |
| Adaptive backfill / replacement / rerun | 0 / 0 / 0 |
| Failure action | stop before training |

The thresholds preserve useful margin while remaining consistent with observed physical yields. They are not weakened to guarantee a pass.

## 19. Anti-contamination design

The exact 198-run matrix was checked against 16 protected manifests, including historical HOLDOUT/VALIDATION-derived corpora, old Sand studies, all earlier calibration pilots, the failed factor-conditioned corpus, and both new pilots.

| Check | Result |
|---|---:|
| Unique future run IDs | 198/198 |
| Unique future scenario signatures | 198/198 |
| Historical exact overlap | 0 |
| Historical forbidden-near overlap | 0 |
| Historical run-ID reuse | 0 |
| TRAIN/VALIDATION exact overlap | 0 |
| TRAIN/VALIDATION forbidden-near overlap | 0 |
| Pilot scenario reuse | 0 |

The canonical forbidden-near rule remains `|Δstart| < .002 m AND |Δwidth| < .004 m` within the same physical signature domain. No replacement or result-driven selection is allowed.

## 20. Model, historical, and deployment boundaries

All research-model counters are zero: V1/V2/Terrain inference, optimizer steps, checkpoint writes, normalizer fits, HNM rounds, threshold/persistence/architecture searches, and sensor-fusion experiments. The failed corpus received no Hazard inference. The full redesigned corpus was not generated.

Historical scientific status remains unchanged:

- `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`
- `FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS: NOT YET MODEL-TESTED`

Deployment engineering remains unaffected and may continue in parallel with the frozen V2 labeled only `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`. `/d/shin/Infineon_FastReflex_E84` was not modified.

## 21. Verdict and next milestone

`FACTOR_CONDITIONED_PHYSICAL_DOMAIN_REDESIGN_READY`

The failure mechanism is actionable, both factor manifolds remain physically feasible, a 90.625%-strict fresh calibration envelope exists across all cells with the preserved C/.25 exception, generation gates have margin, and no model or protected evidence contamination occurred.

The single recommended next milestone is:

`SAND_FACTOR_CONDITIONED_DEVELOPMENT_RECALIBRATED_GENERATION`

That milestone may generate the exact frozen 198-run corpus and apply physical gates only. It must use zero model inference and stop before training if any physical gate fails. It was not started here.

## 22. Provenance hashes

| Artifact | SHA-256 |
|---|---|
| Audit config | `9354701bb3c2d5b2710ca8c9ec2708f6bdf956027210ba715b2ca2dfe2380381` |
| Physical failure audit artifact | `ac34646fdda94cdf41a8df8132f2e158515a866fdf4568b10362b5ed9e6b20e9` |
| Pilot 1 config | `2439ac729e24636f9c48a38c01f22c2eaacd00efe25702546031870f9b45a922` |
| Pilot 1 matrix | `0244db23f6d5873c785496e66f5baa85dd82853af62a5fe44be9b31e0f60f844` |
| Pilot 1 manifest | `0017716c3a779e96b4ce2d0df569ce47fbfe892ab84c3daeb5025d4c857812bd` |
| Pilot 2 config | `c3609137cadd0f29909580b897bd304f9a8d5e70de439aa3cc5cd4643e1ca76d` |
| Pilot 2 matrix | `9d9264507b880ca3e3b7ee3027bdef9aa5852df7a5517369c26ab1bc19151aa5` |
| Pilot 2 manifest | `78377d3a3be41714e2852d7d043afd5ed04538da0a43ae4178ed92d96f033753` |
| Redesign config | `dcb1417eb1771b7e02652ab4979024fa145dbe156ff333416e485cd29679e449` |
| Redesign readiness artifact | `49f6897b8282648b94a4989ece90bc1c72b775e7c25ccd2b1c003c065c2e6ea5` |
| Redesign expanded matrix | `6ffad518466d3082a742787199c038732dea885c7ef508b2585a1dc267e39fc3` |
| Redesign scenario signatures | `0085a9568c3b30870739792a4cf552699e2dcf4ef45f4f00c3dd4780945e86bf` |
| Redesign TRAIN split | `59fa9edc13c2fbba90c2d54d0b07c2c04e6e716fed88e6646b2bc6d82baf18d6` |
| Redesign VALIDATION split | `35fa774081c3153e198a3d81f1ab7e931a7be75b626aac681afff5a6a16d1495` |
