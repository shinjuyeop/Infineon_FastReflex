# Redesigned Sand Domain Failure Review

## 1. Purpose

이 review는 `sand_benign_generalization_redesigned_study_20260902`가 실패한 세 Confirmation physical-yield gate의 원인을 model-blind physical evidence로만 국소화하고, 다음 fresh study 전에 필요한 최소 변경을 결정한다. 결론은 관측 시간을 늘리는 것이 아니라 broad-mild exposure geometry의 좁은 재보정이 필요하다는 것이다.

## 2. Starting state

- Starting HEAD / `origin/main`: `ca95a12fc36ab6ce5655c95267e11d10fff78ac9`, parity 확인
- Starting tracked worktree: clean
- Previous verdict: `SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION_YIELD_INSUFFICIENT`
- Historical final Model V2 verdict: `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- Whole-simulation status: `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- Support branch: `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`
- Review config: `configs/experiment/20260902_sand_benign_redesigned_domain_failure_review.yaml`
- Review config SHA-256: `ddae06af4167e75b02ad5060a644fc516f99a68d9f9234f27b9d229b258515da`

## 3. Historical evidence boundary

Consumed Generalization HOLDOUT payload는 읽지 않았다. Guard는 `guard_after=1`, `scientific_open_count=1`, second scientific open forbidden 상태 그대로다. 이번 분석은 현재 redesigned study의 physical trace, manifest, physical audit와 기존 development calibration 기록에만 한정했다.

Model V1/V2/Terrain inference, probability, normalized 80D, observability, GRU state, training, HNM, normalizer, threshold/persistence 및 architecture search는 모두 0이다. Confirmation에서는 요청된 physical failure review만 수행했고 model science는 열지 않았다.

## 4. Current generation result

| Item | Frozen result |
|---|---|
| Dataset ID | `sand_benign_generalization_redesigned_study_20260902` |
| Dataset-freeze semantic SHA-256 | `87956c511684a78780d8bc7c1ac50552779de55b85a739e0221d1fa449f9416a` |
| Dataset-freeze file SHA-256 | `c1a51f89d35ff32c880b8572062e8a36ab677c5d929884e68b89d5928a938ddb` |
| Manifest SHA-256 | `90970438abb9eced3742d387697cf3f3ff4bd8b905b3554ea01f6766a38e501b` |
| Physical-audit SHA-256 | `28c5dec8b6c4ed4bbb5056b50f712a11d784fcf63987b8e1cca24017d3cfa20f` |
| Confirmation-seal SHA-256 | `33a2a503ccb548b712c43c485175dd94e0787a7882a9ecb02ba8b7c3b2a76895` |
| Objective valid / invalid | 153 / 23 |
| Strict Sand / Support / Slip | 116 / 32 / 5 |
| Generation checks | 67 PASS / 3 FAIL |

첫 failed study의 valid 55, invalid 121에 비해 크게 개선됐고 Discovery physical gates, Support, phase, entry span, signature diversity와 contamination은 모두 통과했다. 이 결과를 broad design failure로 해석하지 않는다.

## 5. Three failed gates

| Frozen gate | Required | Actual | Gap |
|---|---:|---:|---:|
| Confirmation broad mild | >=40/48 | 35/48 | -5 |
| Confirmation Concrete/.25 strict Sand | >=8/12 | 7/12 | -1 |
| Confirmation strict Sand total | >=54/72 | 52/72 | -2 |

Gate는 변경하거나 사후 완화하지 않았다. 현재 corpus의 verdict도 바꾸지 않았다.

## 6. Invalid-run ledger

모든 stored trace는 9,000 samples지만 physical observation censor는 first fall에서 끝난다. `Outcome before invalidation`의 `NO_ESTABLISHED_EVENT`는 target contact 뒤 Slip, I1, Support가 확립되지 않았다는 뜻이며 strict-benign relabel이 아니다.

| Run | Split | Group | Source/speed | Topology | Start/width (m) | Target ms | Censor ms | Follow-up ms | Deficit ms | Fall | Phase/lead/load | Outcome before invalidation | Reason |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
| `sbgr_d_bb_c_020_08` | D | mild | concrete/.20 | right | 0.337/0.798 | 1504 | 2446 | 0 | 1000 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_d_nh_c_025_01` | D | moderate | concrete/.25 | left | 0.305/0.666 | 1220 | 3415 | 0 | 1000 | post-target | RIGHT_SINGLE_SUPPORT/LEFT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_d_nh_c_025_03` | D | moderate | concrete/.25 | left | 0.317/0.682 | 1220 | 3970 | 0 | 1000 | post-target | RIGHT_SINGLE_SUPPORT/LEFT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_d_nh_c_025_04` | D | moderate | concrete/.25 | left | 0.323/0.690 | 1220 | 4182 | 0 | 1000 | post-target | RIGHT_SINGLE_SUPPORT/LEFT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_d_bb_m_020_08` | D | mild | marble/.20 | right | 0.337/0.798 | 1507 | 7257 | 55 | 945 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_d_bb_m_025_08` | D | mild | marble/.25 | right | 0.337/0.798 | 1509 | 4917 | 0 | 1000 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_bb_c_020_04` | C | mild | concrete/.20 | right | 0.325/0.818 | 1504 | 2446 | 0 | 1000 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_bb_c_020_06` | C | mild | concrete/.20 | right | 0.332/0.826 | 1504 | 2446 | 0 | 1000 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_bb_c_020_07` | C | mild | concrete/.20 | left | 0.345/0.828 | — | 1613 | — | — | pre-target | NO_SUPPORT/NONE/NONE | UNRESOLVED | pre-target fall |
| `sbgr_c_nh_c_020_04` | C | moderate | concrete/.20 | left | 0.346/0.830 | — | 1613 | — | — | pre-target | NO_SUPPORT/NONE/NONE | UNRESOLVED | pre-target fall |
| `sbgr_c_bb_c_025_05` | C | mild | concrete/.25 | left | 0.339/0.820 | 1220 | 2153 | 0 | 1000 | post-target | RIGHT_SINGLE_SUPPORT/LEFT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_bb_c_025_07` | C | mild | concrete/.25 | left | 0.345/0.828 | 1220 | 2153 | 0 | 1000 | post-target | RIGHT_SINGLE_SUPPORT/LEFT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_bb_c_025_08` | C | mild | concrete/.25 | left | 0.340/0.826 | 1220 | 2153 | 0 | 1000 | post-target | RIGHT_SINGLE_SUPPORT/LEFT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_nh_c_025_01` | C | moderate | concrete/.25 | left | 0.306/0.698 | 1220 | 4169 | 0 | 1000 | post-target | RIGHT_SINGLE_SUPPORT/LEFT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_nh_c_025_04` | C | moderate | concrete/.25 | left | 0.324/0.722 | 1220 | 4315 | 0 | 1000 | post-target | RIGHT_SINGLE_SUPPORT/LEFT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_nh_c_030_04` | C | moderate | concrete/.30 | right | 0.336/0.832 | 1508 | 4298 | 0 | 1000 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_bb_m_020_06` | C | mild | marble/.20 | right | 0.332/0.826 | 1507 | 7263 | 61 | 939 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_bb_m_020_07` | C | mild | marble/.20 | left | 0.345/0.828 | — | 1622 | — | — | pre-target | NO_SUPPORT/NONE/NONE | UNRESOLVED | pre-target fall |
| `sbgr_c_bb_m_020_08` | C | mild | marble/.20 | right | 0.338/0.834 | 1507 | 7269 | 47 | 953 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_bb_m_025_04` | C | mild | marble/.25 | right | 0.325/0.818 | 1509 | 5472 | 50 | 950 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_bb_m_025_06` | C | mild | marble/.25 | right | 0.332/0.826 | 1509 | 1866 | 0 | 1000 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_bb_m_025_08` | C | mild | marble/.25 | right | 0.338/0.834 | 1511 | 1899 | 0 | 1000 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |
| `sbgr_c_bb_m_030_08` | C | mild | marble/.30 | right | 0.338/0.834 | 1511 | 4284 | 0 | 1000 | post-target | LEFT_SINGLE_SUPPORT/RIGHT/BILATERAL | NO_ESTABLISHED_EVENT | insufficient post-target observation |

Required aggregate breakdown은 다음과 같다.

| Split | Group | Source | Speed | N invalid | Pre-target | Insufficient follow-up | Other |
|---|---|---|---:|---:|---:|---:|---:|
| Discovery | mild | concrete | 0.20 | 1 | 0 | 1 | 0 |
| Discovery | moderate | concrete | 0.25 | 3 | 0 | 3 | 0 |
| Discovery | mild | marble | 0.20 | 1 | 0 | 1 | 0 |
| Discovery | mild | marble | 0.25 | 1 | 0 | 1 | 0 |
| Confirmation | mild | concrete | 0.20 | 3 | 1 | 2 | 0 |
| Confirmation | moderate | concrete | 0.20 | 1 | 1 | 0 | 0 |
| Confirmation | mild | concrete | 0.25 | 3 | 0 | 3 | 0 |
| Confirmation | moderate | concrete | 0.25 | 2 | 0 | 2 | 0 |
| Confirmation | moderate | concrete | 0.30 | 1 | 0 | 1 | 0 |
| Confirmation | mild | marble | 0.20 | 3 | 1 | 2 | 0 |
| Confirmation | mild | marble | 0.25 | 3 | 0 | 3 | 0 |
| Confirmation | mild | marble | 0.30 | 1 | 0 | 1 | 0 |

## 7. Insufficient-follow-up analysis

| Run | Split | Group | Source | Speed | Target time (ms) | Available follow-up (ms) | Extra required (ms) |
|---|---|---|---|---:|---:|---:|---:|
| `sbgr_d_bb_c_020_08` | D | mild | concrete | 0.20 | 1504 | 0 | 1000 |
| `sbgr_d_nh_c_025_01` | D | moderate | concrete | 0.25 | 1220 | 0 | 1000 |
| `sbgr_d_nh_c_025_03` | D | moderate | concrete | 0.25 | 1220 | 0 | 1000 |
| `sbgr_d_nh_c_025_04` | D | moderate | concrete | 0.25 | 1220 | 0 | 1000 |
| `sbgr_d_bb_m_020_08` | D | mild | marble | 0.20 | 1507 | 55 | 945 |
| `sbgr_d_bb_m_025_08` | D | mild | marble | 0.25 | 1509 | 0 | 1000 |
| `sbgr_c_bb_c_020_04` | C | mild | concrete | 0.20 | 1504 | 0 | 1000 |
| `sbgr_c_bb_c_020_06` | C | mild | concrete | 0.20 | 1504 | 0 | 1000 |
| `sbgr_c_bb_c_025_05` | C | mild | concrete | 0.25 | 1220 | 0 | 1000 |
| `sbgr_c_bb_c_025_07` | C | mild | concrete | 0.25 | 1220 | 0 | 1000 |
| `sbgr_c_bb_c_025_08` | C | mild | concrete | 0.25 | 1220 | 0 | 1000 |
| `sbgr_c_nh_c_025_01` | C | moderate | concrete | 0.25 | 1220 | 0 | 1000 |
| `sbgr_c_nh_c_025_04` | C | moderate | concrete | 0.25 | 1220 | 0 | 1000 |
| `sbgr_c_nh_c_030_04` | C | moderate | concrete | 0.30 | 1508 | 0 | 1000 |
| `sbgr_c_bb_m_020_06` | C | mild | marble | 0.20 | 1507 | 61 | 939 |
| `sbgr_c_bb_m_020_08` | C | mild | marble | 0.20 | 1507 | 47 | 953 |
| `sbgr_c_bb_m_025_04` | C | mild | marble | 0.25 | 1509 | 50 | 950 |
| `sbgr_c_bb_m_025_06` | C | mild | marble | 0.25 | 1509 | 0 | 1000 |
| `sbgr_c_bb_m_025_08` | C | mild | marble | 0.25 | 1511 | 0 | 1000 |
| `sbgr_c_bb_m_030_08` | C | mild | marble | 0.30 | 1511 | 0 | 1000 |

Breakdown은 Discovery/Confirmation `6/14`, mild/moderate `14/6`, Concrete/Marble `12/8`, speed .20/.25/.30=`6/12/2`, left/right topology=`8/12`, phase LEFT/RIGHT single=`12/8`이다. Variant는 `c01:1, c04:4, c05:1, c06:3, c07:1, c08:4, d01:1, d03:1, d04:1, d08:3`이다.

핵심은 20/20이 9초 horizon에서 끝난 run이 아니라 `nonfoot_surface_contact` fall로 1,866–7,269 ms에 조기 censor됐다는 점이다. 20/20 모두 target 뒤 fall이고, Slip/I1/Support established event는 0/20이다. 18건은 fall 직전까지 target contact가 이어져 available follow-up이 0 ms다. 따라서 이들은 successful late target acquisition이 아니다.

## 8. Required-extra-time distribution

20건의 `1000 - available_post_target_ms`는 다음과 같다.

| N | Min | Median | p75 | p90 | p95 | Max |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 939 ms | 1000 ms | 1000 ms | 1000 ms | 1000 ms | 1000 ms |

이는 current label의 산술적 deficit일 뿐, run horizon을 늘리면 fall censor가 사라진다는 근거가 아니다.

## 9. Mild failure analysis

Discovery mild는 45/48 strict였고 invalid 3건은 모두 `d08` width 0.798 m였다. Discovery `d01–d07`은 각 6/6 strict였다. Confirmation mild는 35/48 strict, pre-target fall 2, post-target fall-censor 11이었다.

| Split | Width (m) | Planned | Strict | Pre-target fall | Insufficient/fall-censor |
|---|---:|---:|---:|---:|---:|
| Discovery | .766–.792 | 43 | 43 | 0 | 0 |
| Discovery | .798 | 5 | 2 | 0 | 3 |
| Confirmation | .802–.812 | 19 | 19 | 0 | 0 |
| Confirmation | .818 | 6 | 4 | 0 | 2 |
| Confirmation | .820 | 6 | 5 | 0 | 1 |
| Confirmation | .826 | 6 | 2 | 0 | 4 |
| Confirmation | .828 | 6 | 3 | 2 | 1 |
| Confirmation | .834 | 5 | 2 | 0 | 3 |

Confirmation mild strict width median은 .812 m, invalid median은 .826 m다. 다만 .798 실패와 .802–.812 전부 성공이 공존하므로 width 하나만의 단조 threshold가 아니라 start/topology/source/speed와 exposure의 상호작용이다. Mild mechanics의 peak displacement는 12 insufficient case에서 약 .020 m였지만 모두 fall-censored됐다. `mild` intent가 physical stability를 보장하지 않았다.

따라서 mild 실패는 timing-only가 아니며 broad-mild physical domain에 좁은 geometry calibration이 더 필요하다. Moderate group은 두 split의 frozen aggregate와 cell gate를 모두 통과했으므로 좁히지 않는다.

## 10. Concrete/.25 analysis

| Run | Domain | Topology | Start/width (m) | Phase | Target ms | Outcome | Invalid reason | Deficit ms |
|---|---|---|---|---|---:|---|---|---:|
| `sbgr_c_bb_c_025_01` | mild | left | 0.326/0.804 | RIGHT_SINGLE_SUPPORT | 1220 | STRICT_BENIGN | — | — |
| `sbgr_c_bb_c_025_02` | mild | left | 0.321/0.802 | RIGHT_SINGLE_SUPPORT | 1220 | STRICT_BENIGN | — | — |
| `sbgr_c_bb_c_025_03` | mild | left | 0.332/0.812 | RIGHT_SINGLE_SUPPORT | 1220 | STRICT_BENIGN | — | — |
| `sbgr_c_bb_c_025_04` | mild | left | 0.327/0.810 | RIGHT_SINGLE_SUPPORT | 1220 | STRICT_BENIGN | — | — |
| `sbgr_c_bb_c_025_05` | mild | left | 0.339/0.820 | RIGHT_SINGLE_SUPPORT | 1220 | INVALID | insufficient post-target observation | 1000 |
| `sbgr_c_bb_c_025_06` | mild | left | 0.334/0.818 | RIGHT_SINGLE_SUPPORT | 1220 | STRICT_BENIGN | — | — |
| `sbgr_c_bb_c_025_07` | mild | left | 0.345/0.828 | RIGHT_SINGLE_SUPPORT | 1220 | INVALID | insufficient post-target observation | 1000 |
| `sbgr_c_bb_c_025_08` | mild | left | 0.340/0.826 | RIGHT_SINGLE_SUPPORT | 1220 | INVALID | insufficient post-target observation | 1000 |
| `sbgr_c_nh_c_025_01` | moderate | left | 0.306/0.698 | RIGHT_SINGLE_SUPPORT | 1220 | INVALID | insufficient post-target observation | 1000 |
| `sbgr_c_nh_c_025_02` | moderate | left | 0.312/0.706 | RIGHT_SINGLE_SUPPORT | 1220 | STRICT_BENIGN | — | — |
| `sbgr_c_nh_c_025_03` | moderate | left | 0.318/0.714 | RIGHT_SINGLE_SUPPORT | 1220 | STRICT_BENIGN | — | — |
| `sbgr_c_nh_c_025_04` | moderate | left | 0.324/0.722 | RIGHT_SINGLE_SUPPORT | 1220 | INVALID | insufficient post-target observation | 1000 |

12건 모두 target contact가 1,220 ms로 동일하므로 late-entry 설명은 성립하지 않는다. Invalid 5건 모두 2,153–4,315 ms의 post-target fall censor이며 horizon deficit이 아니다. Cell gate 7/12는 한 건 차이지만, 그 한 건을 살리기 위한 Concrete/.25-only tuning 근거는 없다. Mild width/exposure와 cell-conditioned physical interaction을 전체 축에 대칭적으로 재보정해야 한다.

## 11. Discovery vs Confirmation shift

| Metric | Discovery | Confirmation | Difference C-D |
|---|---:|---:|---:|
| Valid Sand | 66 | 55 | -11 |
| Strict Sand | 64 | 52 | -12 |
| Mild strict | 45 | 35 | -10 |
| Moderate strict | 19 | 17 | -2 |
| Slip | 2 | 3 | +1 |
| Invalid | 6 | 17 | +11 |
| Pre-target fall | 0 | 3 | +3 |
| Insufficient/fall-censor | 6 | 14 | +8 |
| Strict target-entry median | 1267 ms | 1363.5 ms | +96.5 ms |
| Strict target-entry p95 | 1810 ms | 1810 ms | 0 ms |
| Entry span | 892 ms | 894 ms | +2 ms |
| Sand patch-start median | .331 m | .332 m | +.001 m |
| Sand patch-width median | .782 m | .818 m | +.036 m |
| Strict precontact phase | LEFT 20 / RIGHT 41 / DOUBLE 3 | LEFT 15 / RIGHT 37 | principal categories preserved |

Target-entry p95와 span은 사실상 같지만 Confirmation width median은 36 mm 길다. Classification은 `CONFIRMATION_SHIFT_PARAMETER_INTERACTION`이다. Timing shift만으로 설명하지 않는다.

## 12. Speed/source timing effects

Insufficient 20건은 .20/.25/.30에 `6/12/2`로 집중되지만 .25 Concrete failures의 target contact는 1,220 ms로 매우 이르다. Strict population의 speed별 entry median은 .20=1,808 ms, .25=1,220 ms, .30=1,227 ms이고 전체 최대도 2,114 ms다. 9초 horizon에는 모두 6초 이상 nominal headroom이 있다.

Source 단독 원인도 아니다. Insufficient가 Concrete/Marble `12/8`이고 broad mild failures는 두 source에 나타난다. Source/speed는 유지해야 할 축이며, interaction은 geometry exposure와 함께 calibration해야 한다.

## 13. Physical stability at run end

| Class | Count | Evidence |
|---|---:|---|
| `CLEAN_LATE_TARGET_TIMING_LIMIT` | 0 | horizon-censored stable case 없음 |
| `LATE_TARGET_WITH_PHYSICAL_INSTABILITY` | 20 | target 뒤 `nonfoot_surface_contact` fall censor |
| `TARGET_NOT_RELIABLY_RESOLVED` | 3 | target 전 fall |
| `INCONCLUSIVE` | 0 | censor cause와 ordering이 모두 명시됨 |

20건 모두 현재 physical observation 끝에서 이미 낙상했다. 더 긴 run이 안정 관측을 제공할 것이라는 증거는 없고, 같은 trajectory에서는 fall censor만 그대로 유지된다.

## 14. Observation-duration hypothesis

아래 `Nominally covered`는 fall censor가 duration extension만큼 뒤로 이동한다는 비물리적 optimistic arithmetic이다. 실제 trajectory를 존중하면 physically covered count는 모든 duration에서 0이다.

| Candidate future duration | Nominally timing-covered cases | Remaining timing deficits | Comment |
|---|---:|---:|---|
| 9.5 s | 0 | 20 | +500 ms는 939–1000 ms deficit보다 짧음; 실제 20건 모두 조기 fall |
| 10.0 s | 20 | 0 | 산술상 +1000 ms; 실제 fall censor 때문에 0건만 defensible |
| 10.5 s | 20 | 0 | 추가 horizon이 이미 발생한 fall을 제거하지 않음 |
| 11.0 s | 20 | 0 | 동일 |

현재 labels는 변경하지 않는다. `OBSERVATION_WINDOW_TOO_SHORT` confidence는 LOW이며 observation duration은 9.0 s로 보존한다.

## 15. Other domain hypotheses

| Hypothesis | Evidence for | Evidence against | Confidence | Design implication |
|---|---|---|---|---|
| Observation window | Manifest reason이 insufficient follow-up이고 산술 deficit 939–1000 ms | 20/20이 1.866–7.269 s에 fall; horizon censor 0 | LOW | 9 s 유지 |
| Slow-speed late entry | .20/.25에 invalid 18/20 집중 | .25 failures target 1.220/1.509 s; strict p95 두 split 동일 | LOW | speed 축 유지, duration conditioning 금지 |
| Source-speed interaction | Concrete 12, Marble 8; .25 12건 | 단일 source 또는 speed로 한정되지 않음 | MODERATE | mild calibration을 모든 cell에 대칭 수행 |
| Mild physical instability | Mild invalid 16; 14 follow-up cases 모두 post-target fall; wider C variants 악화 | .802–.812 m는 19/19 strict여서 mild mechanics 전체가 실패한 것은 아님 | HIGH | broad-mild exposure boundary만 재보정 |
| Confirmation realization timing shift | C invalid 17 vs D 6; entry median +96.5 ms | p95 동일, widths +36 mm, failures는 fall | LOW | timing shift로 다루지 않음; parameter-interaction classification HIGH |
| Gate/domain mismatch | C misses are small: total -2, cell -1 | gate는 intended robust benign population을 합리적으로 요구; D가 모두 통과 | LOW | gate 완화 없음 |

Primary root cause는 `RESIDUAL_MILD_DOMAIN_INSTABILITY` HIGH와 `CONFIRMATION_SHIFT_PARAMETER_INTERACTION` HIGH다. `SOURCE_SPEED_ENTRY_INTERACTION`은 MODERATE, `PHYSICAL_YIELD_GATE_TOO_AGGRESSIVE`는 LOW다.

## 16. Pilot decision

`PILOT_NOT_REQUIRED`

이번 milestone에서 pilot simulation은 0이다. 이유는 duration 가설을 판정하는 데 추가 run이 필요하지 않기 때문이다. Current 176-run study와 기존 model-blind 96-run calibration은 (1) 20건 전부 조기 fall, (2) 두 split의 거의 같은 entry p95, (3) width/exposure에 따른 mild viability 이동을 이미 직접 보여준다. Duration pilot은 잘못된 가설을 더 시험하게 된다.

반면 exact robust mild geometry envelope를 full-study design으로 동결하기에는 .798 m 실패와 .802–.812 m 성공이 공존해 interaction boundary가 남는다. 이는 이 milestone에서 임의의 176-run matrix를 동결할 근거가 아니라 다음의 좁은 mild-domain calibration question이다.

## 17. Pilot results

새 pilot은 수행하지 않았다. Pilot config SHA, batch, reused pilot은 없다. `pilot_simulation_runs=0`, `full_new_study_generation_runs=0`이다.

## 18. Root-cause verdict

`SAND_BENIGN_REDESIGNED_DOMAIN_FAILURE_REVIEW_ACTIONABLE`

세 gate failure는 9초 observation horizon 부족이 아니다. Broad mild의 split-specific geometry exposure가 Confirmation에서 길어졌고, source/speed/topology interaction과 함께 physical falls를 늘렸다. Concrete/.25 역시 target가 1.220 s에 도달한 뒤 fall하므로 같은 결론이다.

## 19. Components preserved

| Component | Current result | Change next design? | Reason |
|---|---|---|---|
| Source | Both represented; no source-only failure | No | Concrete/Marble 축 유지 |
| Speed | All represented; timing-only 아님 | No | .20/.25/.30 축 유지 |
| Mild domain | Confirmation 35/48, fall-censor 11 + prefall 2 | **Calibrate geometry only** | mild mechanics가 아니라 exposure boundary 문제 |
| Moderate domain | D 19/24, C 17/24; all frozen gates PASS | No | 이미 성공한 boundary group |
| Support controls | Ordinary 24/24, delayed 8/8 | No | solved component |
| Phase measurement | global/cell gates PASS | No | -20 ms exact phase 유지 |
| Topology semantics | meaningful leading-side relation, gates PASS | No | semantics 유지; calibration balance에 포함 |
| Realization mechanism | unique planned geometry, overlap 0 | No semantic change | next study만 fresh variants 사용 |
| Geometry | C width median +36 mm; wide mild variants unstable | **Yes, mild only** | robust exposure boundary 재보정 |
| Observation duration | no horizon-censored invalid | No, 9.0 s 유지 | extension cannot remove early falls |
| Physical label contract | censor-aware behavior worked as designed | No | fall을 strict로 숨기지 않음 |
| Generation gates | 67/70 checks pass; failure is physical | No | post-hoc relaxation 금지 |

## 20. Minimal redesign

이 결과는 broad Sand domain redesign을 요구하지 않는다. 최소 physically justified change는 broad-mild geometry/exposure realization만 재보정하는 것이다. Source, speed, mild mechanics, moderate, Support, phase, topology, label, gates와 9초 observation은 보존한다.

그러나 full 176-run minimal study는 이번 milestone에서 freeze하지 않는다. 다음 calibration은 fresh, model-blind, symmetric mild-only scenarios로 `.798–.820 m` 부근의 non-monotonic transition을 source × speed × topology에 걸쳐 분리해야 한다. 결과가 robust envelope를 정하면 그 뒤에만 fresh Discovery/Confirmation matrix를 freeze한다.

따라서 이번 feasibility는 `MILD_DOMAIN_RECALIBRATION_REQUIRED`이며 `MINIMAL_REDESIGN_READY`가 아니다. New minimal-study design hashes는 발행하지 않았다. 후보 YAML이나 `TO_BE_FROZEN` hash는 repository에 남기지 않았다.

## 21. Fresh-study constraints

향후 study는 calibration 후에만 다음을 만족해야 한다.

- new dataset ID, new run IDs, new scenario signatures
- fresh Discovery와 fresh Confirmation
- failed first study, calibration 96, current redesigned 176, future calibration pilot의 exact reuse 0
- historical exact overlap 0
- cross-split exact/forbidden near overlap 0
- no adaptive replacement, backfill 또는 split move
- model-blind physical generation gate를 먼저 통과
- current mild/moderate/source/speed/Support balance와 unchanged gates 보존

현재 study 또는 pilot run을 다음 split에 재사용하지 않는다.

## 22. Architecture/model boundary

Architecture, feature, sensor, threshold, persistence, training 및 model artifact는 변경하지 않았다. V1 inference=0, V2 inference=0, Terrain inference=0, optimizer steps=0, checkpoint writes=0, normalizer fits=0, HNM rounds=0이다.

## 23. Historical HOLDOUT protection

- Old HOLDOUT raw payload reads: 0
- Old HOLDOUT inference/feature reconstruction/visualization: 0
- Guard: 1
- Scientific opens: 1
- Current redesigned study reused in future split: NO
- New pilot reused: NO; pilot 자체가 0건
- Current Confirmation model analysis: 0

## 24. Verdict

`SAND_BENIGN_REDESIGNED_DOMAIN_FAILURE_REVIEW_ACTIONABLE`

`MILD_DOMAIN_RECALIBRATION_REQUIRED`

Current dataset는 frozen contract 아래 계속 `PHYSICAL_GENERATION_FAILED`다. Model Discovery로 승격하지 않고, V2를 replay하지 않으며, Confirmation model science를 열지 않는다.

## 25. Recommended next milestone

`SAND_BENIGN_MILD_DOMAIN_CALIBRATION_REDESIGN`

목적은 broad-mild exposure boundary만 fresh physical calibration으로 정하는 것이다. Full new study generation, V2 replay, model analysis, Confirmation model analysis는 자동 시작하지 않는다.

## Tests

Regression coverage는 review config provenance, three failed gates, 23-invalid decomposition, 20/20 post-target fall censor, unchanged dataset failure/seal, zero model counters를 고정한다. Full repository verification은 129 tests PASS, user-supplied policy ONNX가 필요한 optional smoke 1건 SKIP이다. `compileall`, Ruff와 `git diff --check`도 통과했다.

## Git

Starting commit은 `ca95a12fc36ab6ce5655c95267e11d10fff78ac9`다. 이 milestone의 intended commit message는 `Review redesigned Sand domain failure`이며 final pushed commit/parity/clean status는 handoff에서 보고한다.
