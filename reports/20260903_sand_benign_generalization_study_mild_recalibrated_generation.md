# Mild-Recalibrated Sand Benign Generalization Study Generation

## 1. Purpose

동결된 mild-recalibrated design을 바꾸지 않고 fresh 176-run Sand-benign corpus를 한 번 생성해 execution integrity와 model-blind physical yield/diversity만 판정했다. Model V1/V2/Terrain inference, causal 80D/observability 분석, training, HNM, normalizer fit, threshold/persistence 및 architecture search는 수행하지 않았다.

## 2. Starting state

- Starting HEAD / `origin/main`: `c9a8bd7fe56b8228c9b7829c5ee4bc6981b10762`, parity
- Starting tracked worktree: clean
- Previous verdict: `SAND_BENIGN_MILD_DOMAIN_MINIMAL_REDESIGN_READY`
- Historical final status: `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`

## 3. Historical scientific boundary

Consumed Generalization HOLDOUT guard는 `guard_after=1`, `scientific_open_count=1`이고 SHA-256은 `0913da8a583cab7834590d669ba9c8d470485b8129fe81f538aba6c461613154`로 유지했다. Old HOLDOUT payload read, inference, feature reconstruction, visualization은 모두 0이다. 두 이전 Sand study와 모든 calibration pilot은 manifest/signature와 frozen aggregate provenance 비교에만 사용했고 raw record를 새 corpus에 복사하지 않았다. Model artifact와 historical dataset freeze를 포함해 execution config에 선언한 protected artifact hash는 모두 generation 전후 일치했다.

## 4. Frozen mild redesign

Design config file SHA-256은 `1301d64391b423eb50b2ac4188058e1ac0cd988dce477323f87891b946aeaeb5`, complete redesign SHA-256은 `09c2e1a22d47ba115dc2ef3db0251a7dd836096ffe2b9e370fbe9d1677416356`다.

| Component | SHA-256 |
|---|---|
| Parameter domain | `16b79593193fa5925cbbc2fa73dfef3764e655b4911f4b0b5dae7f2c3add0b74` |
| Scenario matrix | `24a4833c869a25df10cff387d9e58d02ca8079eca04ce1f611c0cf54347976ca` |
| Split plan | `656f807d1a8bc11df6529826cb25a99d7243a32d129704f61f02f9402b102104` |
| Physical-label contract | `030e85c6561b3fb9b757a2beb82e15f583264cbfe203b016b0070f82888bb39c` |
| Generation gates | `914ac15e564f8218bfc386e3cfbbdeac292a4f66a9c2df5611a6ff7dcf92ec34` |
| Diversity metrics | `f9fbfa697d33aae328848ee3ef3bfe420e63a3e1a8da54c35b975ffa47942e15` |
| Confirmation protocol | `f423ef4996173b9562c4f382849ea1b2ce9c212c10fbd5f324309ee7b6610cad` |

Generation execution config SHA-256 `151c89523a27fb92dd46cbc3dc3f60193c1916fb16503bce903e44fe6006add3`와 canonical mild generator SHA-256 `92fbc8c79997a93b88e740afcec43214aedc248aca1fafaae6425b87b6987fd0`을 run 1 전에 동결했다. Generation 시작 뒤 config, matrix, implementation 또는 parameter mutation은 0이다.

## 5. Calibration evidence

이전 redesigned study의 broad mild는 strict `80/96`, invalid `16/96`이었고 실패는 width 하나가 아니라 start/width/exit/topology의 joint interaction이었다. Fresh model-blind mild pilot A/B/C는 `36/12/24`, 총 72건을 사전 동결해 모두 보존했다. Final pilot C는 strict `24/24`, 여섯 source-speed cell 각각 `4/4`였다. Concrete/.25 transition-right는 세 geometry에서 `0/6`인 반면 같은 profile family는 나머지 다섯 cell에서 `15/15`여서 이 cell만 left-only로 동결했다. Pilot record는 새 study에 재사용하지 않았다.

## 6. Dataset plan

Dataset ID는 `sand_benign_generalization_mild_recalibrated_study_20260903`다. 각 split은 broad mild 48, boundary moderate 24, ordinary Support 12, delayed Support 4로 총 88건이다. 전체는 176건이고 Severe는 0이다. Planned run ID와 scenario signature는 각각 176/176 unique다.

| Group | Discovery | Confirmation | Total |
|---|---:|---:|---:|
| Broad mild | 48 | 48 | 96 |
| Boundary moderate | 24 | 24 | 48 |
| Ordinary Support | 12 | 12 | 24 |
| Delayed Support | 4 | 4 | 8 |
| Total | 88 | 88 | 176 |

## 7. Generation execution

Planned/attempted/completed는 `176/176/176`, Discovery/Confirmation은 `88/88`이다. Adaptive backfill, replacement, rerun, split move 및 outcome-driven regeneration은 모두 0이다. 실행 시간은 1,164.029 s였다. 176 NPZ의 합은 83,345,954 bytes이고 최종 directory는 184 files, 85,338,217 bytes다.

## 8. Objective validity

| Split | Valid | Invalid | Strict Sand | Support | Slip | Dual/other |
|---|---:|---:|---:|---:|---:|---:|
| Discovery | 85 | 3 | 69 | 16 | 0 | 0 |
| Confirmation | 84 | 4 | 65 | 16 | 3 | 0 |
| Total | 169 | 7 | 134 | 32 | 3 | 0 |

Overall objective-valid gate `169 >= 132`는 PASS다. Outcomes는 mutually exclusive다. Sand에서 Support 또는 dual hazard는 없고, Support 32건은 전부 별도 control이다.

## 9. Mild physical outcomes

Broad mild는 Discovery `48/48`, Confirmation `48/48`, aggregate `96/96` strict-benign이다. 모든 source-speed cell은 split마다 `8/8`이며 Slip, Support, invalid, fall은 모두 0이다.

| Split | Concrete/.20 | Concrete/.25 | Concrete/.30 | Marble/.20 | Marble/.25 | Marble/.30 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Discovery | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 48/48 |
| Confirmation | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 48/48 |

각 split의 mild topology는 transition-left 38, transition-right 10이다. 다섯 cell은 left 6/right 2이고 Concrete/.25는 frozen exception대로 left 8/right 0이다. Broad-mild yield gate와 모든 cell gate는 PASS다.

## 10. Moderate physical outcomes

| Split | Source/speed | Planned | Valid | Strict moderate | Support | Slip | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| Discovery | Concrete/.20 | 4 | 4 | 4 | 0 | 0 | 0 |
| Discovery | Concrete/.25 | 4 | 1 | 1 | 0 | 0 | 3 |
| Discovery | Concrete/.30 | 4 | 4 | 4 | 0 | 0 | 0 |
| Discovery | Marble/.20 | 4 | 4 | 4 | 0 | 0 | 0 |
| Discovery | Marble/.25 | 4 | 4 | 4 | 0 | 0 | 0 |
| Discovery | Marble/.30 | 4 | 4 | 4 | 0 | 0 | 0 |
| Confirmation | Concrete/.20 | 4 | 3 | 3 | 0 | 0 | 1 |
| Confirmation | Concrete/.25 | 4 | 2 | 2 | 0 | 0 | 2 |
| Confirmation | Concrete/.30 | 4 | 3 | 3 | 0 | 0 | 1 |
| Confirmation | Marble/.20 | 4 | 4 | 4 | 0 | 0 | 0 |
| Confirmation | Marble/.25 | 4 | 4 | 4 | 0 | 0 | 0 |
| Confirmation | Marble/.30 | 4 | 4 | 1 | 0 | 3 | 0 |

Aggregate strict moderate는 Discovery `21/24`, Confirmation `17/24`로 양쪽 `>=12` PASS이며 12개 cell 모두 `>=1/4` PASS다. Actual contamination은 Support 0, Slip 0/3이다. 세 Slip은 valid physical outcome으로 보존했고 strict benign으로 relabel하거나 교체하지 않았다.

## 11. Source/speed matrix

| Split | Source | Speed | Planned Sand | Valid | Strict benign | Support | Slip | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Discovery | Concrete | .20 | 12 | 12 | 12 | 0 | 0 | 0 |
| Discovery | Concrete | .25 | 12 | 9 | 9 | 0 | 0 | 3 |
| Discovery | Concrete | .30 | 12 | 12 | 12 | 0 | 0 | 0 |
| Discovery | Marble | .20 | 12 | 12 | 12 | 0 | 0 | 0 |
| Discovery | Marble | .25 | 12 | 12 | 12 | 0 | 0 | 0 |
| Discovery | Marble | .30 | 12 | 12 | 12 | 0 | 0 | 0 |
| Confirmation | Concrete | .20 | 12 | 11 | 11 | 0 | 0 | 1 |
| Confirmation | Concrete | .25 | 12 | 10 | 10 | 0 | 0 | 2 |
| Confirmation | Concrete | .30 | 12 | 11 | 11 | 0 | 0 | 1 |
| Confirmation | Marble | .20 | 12 | 12 | 12 | 0 | 0 | 0 |
| Confirmation | Marble | .25 | 12 | 12 | 12 | 0 | 0 | 0 |
| Confirmation | Marble | .30 | 12 | 12 | 9 | 0 | 3 | 0 |

Strict Sand total은 Discovery `69/72`, Confirmation `65/72`이고 모든 cell은 `>=8/12`다. Total/cell strict gates는 모두 PASS다.

## 12. Phase diversity

Eligible population은 strict Sand 134건이고 primary phase는 first censor-valid loaded target contact의 정확히 20 ms 전 상태다.

| Split | LEFT single | RIGHT single | DOUBLE | Both-phase cells | Usable cells | Leading feet | Status |
|---|---:|---:|---:|---:|---:|---|---|
| Discovery | 14 | 52 | 3 | 5/6 | 6/6 | LEFT 55, RIGHT 14 | PASS |
| Confirmation | 13 | 51 | 1 | 5/6 | 6/6 | LEFT 52, RIGHT 13 | PASS |

두 split 모두 principal phase는 LEFT/RIGHT 두 개이고, Concrete/.25는 frozen left-only topology 때문에 RIGHT single만 갖는 predeclared exception이다. 나머지 다섯 cell은 양 principal phase를 갖는다. Phase, usable-cell, both-cell 및 both-leading-foot gate는 모두 PASS다.

## 13. Topology/contact realization

| Split | Topology | Planned Sand | Strict | Actual leading foot | Pre-contact phase |
|---|---|---:|---:|---|---|
| Discovery | transition-left | 58 | 55 | LEFT 55 | RIGHT 52, DOUBLE 3 |
| Discovery | transition-right | 14 | 14 | RIGHT 14 | LEFT 14 |
| Confirmation | transition-left | 58 | 52 | LEFT 52 | RIGHT 51, DOUBLE 1 |
| Confirmation | transition-right | 14 | 13 | RIGHT 13 | LEFT 13 |

Strict Sand의 first-contact loaded side는 Discovery BILATERAL 67/LEFT 2, Confirmation BILATERAL 64/LEFT 1이다. Topology는 actual leading foot과 반대편 single-support approach phase를 일관되게 만들지만 topology 자체를 phase truth로 사용하지 않는다. Support controls에서는 각 split ordinary actual side가 LEFT_ONLY 5/RIGHT_ONLY 7, delayed가 LEFT_ONLY 4다. LEFT_ONLY Support는 RIGHT single/LEFT lead, RIGHT_ONLY Support는 LEFT single/RIGHT lead와 일치했다.

## 14. Entry-time diversity

Strict Sand first target contact는 overall `1,220–1,810 ms`, span 590 ms, median 1,227 ms다. Discovery와 Confirmation도 각각 `1,220–1,810 ms`, span 590 ms다.

| Axis | Range (ms) | Span (ms) |
|---|---:|---:|
| Concrete | 1,220–1,808 | 588 |
| Marble | 1,220–1,810 | 590 |
| .20 m/s | 1,245–1,810 | 565 |
| .25 m/s | 1,220–1,509 | 289 |
| .30 m/s | 1,227–1,511 | 284 |
| transition-left | 1,220–1,810 | 590 |
| transition-right | 1,503–1,511 | 8 |

Frozen gate는 split-global span `>=250 ms`이며 양쪽 모두 PASS다. Topology별 span은 descriptive이고 별도 frozen threshold는 없다.

## 15. Physical exposure diagnostics

Broad-mild strict records의 cumulative loaded-Sand exposure와 contact episode는 다음과 같다.

| Split | Exposure range / median (ms) | Episode range / median | Fall relation |
|---|---|---|---|
| Discovery | 2,016–4,618 / 3,076.5 | 9–17 / 14 | NO_FALL 48/48 |
| Confirmation | 2,021–4,471 / 3,064 | 9–16 / 14 | NO_FALL 48/48 |
| Aggregate | 2,016–4,618 / 3,064 | 9–17 / 14 | NO_FALL 96/96 |

Exposure는 outcome censor에 영향을 받으므로 label input이나 threshold가 아니다. 위 값은 generated mild domain이 한 가지 contact duration으로 붕괴하지 않았음을 보여 주는 diversity diagnostic일 뿐이다.

## 16. Support controls

| Split | Ordinary planned/qualified | Per source-speed | Delayed planned/qualified | Status |
|---|---:|---:|---:|---|
| Discovery | 12/12 | 2/2 each | 4/4 | PASS |
| Confirmation | 12/12 | 2/2 each | 4/4 | PASS |

Ordinary minimum `>=10/12`, 각 cell `>=1/2`, delayed minimum `>=3/4`를 모두 통과했다. Support 32건에서 Slip/dual/invalid는 0이며 Sand-benign population으로 섞이지 않았다.

## 17. Invalidity decomposition

Invalid 7건은 pre-target fall 1, insufficient post-target observation 6, 기타 0이다. Insufficient 여섯 건은 모두 target contact 뒤 fall censor이고 target-to-fall은 2,195–3,095 ms다. Stable horizon censor나 label determination 불능의 다른 원인은 없다.

| Split | Factor | Invalid | Localization |
|---|---|---:|---|
| Discovery | Concrete/.25 moderate left | 3 | post-target fall at 3,415/3,970/4,182 ms |
| Confirmation | Concrete/.20 moderate left | 1 | pre-target fall at 1,613 ms |
| Confirmation | Concrete/.25 moderate left | 2 | post-target fall at 4,169/4,315 ms |
| Confirmation | Concrete/.30 moderate right | 1 | post-target fall at 4,298 ms |

모든 invalid는 moderate Concrete에 국소화됐고 mild/Marble/Support에는 없다. Records는 원 split에 보존했으며 backfill하지 않았다.

## 18. Physical signatures

- Scenario signatures: 176/176 unique, exact duplicate 0
- Valid physical signatures: 167/169 unique, ratio 98.82%; frozen `>=80%` gate PASS
- Exact physical duplicate pairs: 2
- Repository-scaled distance `<=0.10` non-gating near pairs: 64, 그중 cross-split 37
- Planned Discovery/Confirmation exact scenario overlap: 0
- Frozen cross-split parameter-near overlap: 0

두 exact physical pairs는 ordinary-Support Concrete/.30의 corresponding Discovery/Confirmation records다. Scenario signature는 서로 다르며 physical uniqueness gate를 충분히 통과한다. Near-pair count는 post-generation waveform/signature diagnostic이고 frozen planned-parameter isolation gate를 대체하지 않는다.

## 19. Historical contamination

Unified, Model V2, Generalization, Ice semantics, 두 이전 Sand study, 기존 Sand calibration 96건, mild pilots 72건과 비교해 historical exact signature overlap과 run-ID reuse는 모두 0이다. Old study/pilot record reuse, split movement, raw payload copy도 0이다. Cross-split threshold를 historical reference에 그대로 적용한 non-gating near diagnostic은 157건이지만 frozen historical criterion은 exact-only이며 exact overlap은 0이다. 이 diagnostic을 generation 뒤 새 failure gate로 만들지 않았다.

## 20. Generation gates

전체 frozen ledger는 70/70 PASS이고 failure는 0이다. `physical_audit.json`에 row-level actual/requirement/status가 보존된다.

| Ledger group | Checks | Actual summary | Result |
|---|---:|---|---|
| Execution | 4 | attempted/completed 176; backfill/replacement 0 | 4/4 PASS |
| Overall objective-valid | 1 | 169, requirement >=132 | 1/1 PASS |
| Discovery yield totals | 5 | strict 69; mild 48; moderate 21; ordinary 12; delayed 4 | 5/5 PASS |
| Discovery cell yield | 18 | strict 9–12; moderate 1–4; ordinary 2 | 18/18 PASS |
| Confirmation yield totals | 5 | strict 65; mild 48; moderate 17; ordinary 12; delayed 4 | 5/5 PASS |
| Confirmation cell yield | 18 | strict 9–12; moderate 1–4; ordinary 2 | 18/18 PASS |
| Discovery diversity | 5 | two phases; both 5/6; usable 6/6; both feet; span 590 | 5/5 PASS |
| Confirmation diversity | 5 | two phases; both 5/6; usable 6/6; both feet; span 590 | 5/5 PASS |
| Physical uniqueness | 1 | 167/169 = 0.9882 | 1/1 PASS |
| Integrity | 8 | Severe/model fields/reuse/overlap 0; IDs/signatures complete | 8/8 PASS |

No failed gate exists. Integrity, yield, and diversity aggregate booleans are all true.

## 21. Discovery readiness

Discovery generation gates pass: strict Sand `69/72`, broad mild `48/48`, moderate `21/24`, ordinary/delayed Support `12/12` and `4/4`, all phase/diversity/integrity checks PASS다. Model inference는 0이다. 따라서 `MILD_RECALIBRATED_DISCOVERY`는 다음 별도 model/observability analysis milestone을 시작할 물리적 자격이 있지만 그 분석은 여기서 시작하지 않았다.

## 22. Confirmation sealing

Confirmation 88건은 함께 생성됐지만 objective physical audit와 integrity/gate 계산만 수행했다. V1/V2/Terrain inference, normalized 80D, observability, visualization, hypothesis selection은 모두 0이다. 상태는 `SEALED_FOR_MILD_RECALIBRATED_CONFIRMATION`이며 loader는 NPZ를 열기 전에 Confirmation 요청을 거부한다. 이후에도 Discovery analysis가 하나의 exact interpretation과 metric hash를 freeze하기 전까지 model science에 사용할 수 없다.

## 23. Dataset freeze

| Frozen object | SHA-256 |
|---|---|
| Generation config | `151c89523a27fb92dd46cbc3dc3f60193c1916fb16503bce903e44fe6006add3` |
| Pre-simulation freeze file | `d617d1ff423e3f0d416995a3228cbfe0d077ecbad0834356c10bb1d1eb1c5ee1` |
| `MILD_RECALIBRATED_STUDY_MANIFEST_SHA` | `f19ec527cb9faac0d8f3a385a1a63e8a951ced7f275c7cfc3dd459cc42f375d1` |
| `MILD_RECALIBRATED_DISCOVERY_SPLIT_SHA` | `1c211c38ee2bd7f9e9a44e0f81ec6a0dd110a8e14809c12777c043d33928f93f` |
| `MILD_RECALIBRATED_CONFIRMATION_SPLIT_SHA` | `3bfbb050db3ebadcc363d1c6e51013dc349dbac4e0594c183a55245ea38c2e80` |
| `MILD_RECALIBRATED_SCENARIO_SIGNATURE_SHA` | `be6d6f0d6bb312617784bad31b55cedfd686bb32aaebd37235efe7a24345fad1` |
| `MILD_RECALIBRATED_PHYSICAL_SIGNATURE_SHA` | `dc47c671d5bd7452902fbc7ba1a9e5491f2b1c231af994a881583c304783f51f` |
| `MILD_RECALIBRATED_NPZ_AGGREGATE_SHA` | `5f63a5e4def8d09159407109f2b51635c5819931551e604c138ba1f02693f3c4` |
| `MILD_RECALIBRATED_PHYSICAL_OUTCOME_SHA` | `f71b80920b047ad27271ef08c0136a49686992c921ccbebd498f68263b2dfbd6` |
| `MILD_RECALIBRATED_GENERATION_GATE_RESULT_SHA` | `5034afca0071473888fe5895093cb065868f65df909324b969f1ccd75c1f7a8c` |
| `MILD_RECALIBRATED_PHYSICAL_AUDIT_SHA` | `ec88a6a9b99dc91cd36d4e1bdba88f21a60ca1f1967630e4975aa7eb3e8ffb90` |
| Confirmation seal file | `2795fa2cc02a049dbe0de2331820506845d333980e17fd0c6e33e5ce471082c2` |
| `MILD_RECALIBRATED_DATASET_FREEZE_SHA` | `706d939c03bf31df0fb39d1043e99dbbb05922664e207425c8c96ab7c93ee675` |
| Dataset-freeze file | `dafcf8f42b2cc701358d6b55f40727ec234ce975ae992e7c3811b8640c5805b6` |
| Generation summary file | `b17e8d31676d8ce008965b97c183a02574bedd9324fbb2ea0497605e4bce033e` |

All 176 NPZ, manifest, audit, Confirmation seal, aggregate 및 semantic/file dataset-freeze checks가 Confirmation waveform을 deserialize하지 않는 verifier에서 PASS했다.

## 24. Historical scientific status

새 corpus는 model-performance evidence가 아니므로 다음 status는 변경하지 않는다.

- `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`

첫 Sand study는 valid `55/176`, invalid 121, strict Sand 41, Support 11이고 valid Sand phase가 DOUBLE_SUPPORT 하나로 붕괴했다. 이전 redesigned study는 valid `153/176`, invalid 23, strict Sand 116, Support 32이며 Discovery/Confirmation 모두 5/6 both-phase cells를 만들었지만 세 Confirmation yield gate를 실패했다. 새 mild-recalibrated study는 valid `169/176`, invalid 7, strict Sand 134, Support 32이고 양 split 모두 5/6 both-phase cells와 6/6 usable cells를 유지하며 70/70 gate를 통과했다. 이 비교는 physical dataset viability의 변화이지 Sand model generalization 주장이 아니다.

## 25. Limitations

- Evidence는 deterministic simulator와 frozen policy에 한정되며 hardware generalization을 주장하지 않는다.
- Moderate Concrete에 invalid 7건, Confirmation Marble/.30 moderate에 actual Slip 3건이 남지만 frozen boundary gates는 통과한다.
- Exact physical duplicate 2와 non-gating near pair 64가 있으며 unique ratio는 98.82%다.
- Transition-right entry-time span은 8 ms이지만 frozen gate는 split-global로 정의돼 있다.
- Historical-near 157건은 design proximity diagnostic이며 frozen exact-only contamination gate에는 포함되지 않는다.
- Pelvis observability와 Model V2 benign rejection은 아직 전혀 측정하지 않았다.

## 26. Verdict

`SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_GENERATION_READY`

Frozen physical-generation gate 70개가 모두 통과했으며 execution/integrity, historical isolation, model-blind boundary, Confirmation seal도 유지됐다. 이는 Discovery model science를 시작할 자격이지 Sand model generalization support가 아니다.

## 27. Recommended next milestone

`SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_DISCOVERY_ANALYSIS`

다음 milestone은 새 Discovery만 열어 causal 80D/FSR diagnostics와 exact frozen V2 replay를 수행하고 H1/H2/H3 중 하나를 metric hash와 함께 동결해야 한다. Confirmation은 계속 sealed다. 해당 분석은 자동으로 시작하지 않았다.
