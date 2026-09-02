# Sand-Benign Physical Calibration and Generalization Study Redesign

## 1. Starting state

- 시작 HEAD와 `origin/main`: `b7510761c86a8b55c644d1e1e6aedd4c187c236a`, parity 확인
- 시작 tracked worktree: clean
- 이전 milestone: `SAND_BENIGN_GENERALIZATION_STUDY_GENERATION`
- 이전 verdict: `SAND_BENIGN_GENERALIZATION_STUDY_GENERATION_PHYSICAL_YIELD_INSUFFICIENT`
- 실패 dataset: `sand_benign_generalization_study_20260902`, 176/176 runs, dataset-freeze file SHA-256 `6ae5b6ac668f320eac811ebb430eaaecbf24ce64ee541e73db35c0ac862c28f0`
- 이전 design/config SHA: `539f18b1e4c27abca826b7a2eac0d5c663e13035e63b1acb5fbab45136470a7f` / `d638e2c5f4da0365ac56537bfa4a26f592fc5f500a827f588a562c6d9e4f9268`

Historical final Model V2 verdict `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`, whole-simulation status `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`, Support branch `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`는 모두 그대로다. 이 milestone은 새 model evidence가 아니다.

## 2. Evidence boundaries

상세 raw 분석은 실패 study의 88-run `STUDY_DISCOVERY`에만 수행했다. 88행 run-level physical table은 Gitignored `artifacts/runs/20260902_sand_benign_generalization_study_redesign/failed_discovery_physical_calibration.json`에 저장했고 SHA-256은 `78219ab4021f3018ea82888e800f26b025d94b0884295e3039ef9001edd16bcc`다. Failed `STUDY_CONFIRMATION`에는 기존 aggregate objective summary 외 model/80D/observability/hypothesis 분석을 하지 않았다.

Consumed Generalization HOLDOUT guard는 `guard_after=1`, `scientific_open_count=1`, `second_scientific_open_forbidden=true`다. Payload read, inference, feature reconstruction, visualization은 모두 0이다. Historical collision audit는 manifest/signature metadata만 사용했다.

Model V1/V2/Terrain inference, probability, causal 80D, normalization, HNM, training, threshold/persistence 및 architecture search는 수행하지 않았다.

## 3. Failed-generation diagnosis

Discovery 88건 전체는 no-fall 29, pre-target fall 12, post-target fall 47이었다. 원 label은 strict benign 19, Support 5, Slip 2, invalid 62였다. Pre-target 또는 fall-censor ambiguity가 단순히 survivor selection 문제가 아니라 scenario domain 자체의 문제였다.

| Root cause | Evidence | Confidence | Effect on old design |
|---|---|---|---|
| Late target acquisition on unstable hard-ground prefix | Pre-target fall 12/88; 모두 .20/.25, .30은 0; 동일 geometry의 source별 fall clock이 거의 반복 | High | EARLY/MID/LATE label을 모든 speed에 공유한 설계가 실제 contact timing을 보장하지 못함 |
| Excess Sand exposure and severe compliance | Post-target fall 47/88; pilot severe 0/36 strict | High | broad width/severity matrix가 physical validity보다 nominal balance를 우선함 |
| Phase metric at touchdown | Failed valid Sand 21/21 contact-sample DOUBLE, 그러나 contact 20 ms 전은 RIGHT 13 / LEFT 8 | High | A/B/C/D collapse가 실제 single-support 다양성을 숨김 |
| A/B/C/D is metadata, not control | `SimulationConfig`와 controller에 phase slot 입력이 없고 global phase는 동일하게 0에서 시작 | High | assigned phase balance를 realized phase control로 해석할 수 없음 |
| Realization cohort is metadata only | cohort ID가 simulator seed, initial state, pre-roll 또는 policy phase를 변경하지 않음 | High | 두 cohort는 독립 physical realization mechanism이 아님 |
| Blanket fall invalidation for Support | established Support 뒤 >1 s가 관측된 run도 old summary에서 invalid | High | ordinary Support Discovery 2/12가 corrected contract에서는 7/12로 증가 |
| Shared template across all cells | Pilot에서 Concrete/.25 right-entry와 severe tier가 반복 실패 | High | source/speed-conditioned physical domain이 필요 |

## 4. Pre-target fall analysis

| Source / speed | Discovery runs | Pre-target falls | Observation |
|---|---:|---:|---|
| Concrete / .20 | 14 | 4 | late/step-missed entries fail before target |
| Concrete / .25 | 16 | 2 | late geometry only |
| Concrete / .30 | 14 | 0 | target is reached before prefix instability |
| Marble / .20 | 14 | 4 | Concrete과 같은 구조 |
| Marble / .25 | 16 | 2 | Concrete과 같은 구조 |
| Marble / .30 | 14 | 0 | pre-target fall 없음 |

반복되는 fall clock은 .20에서 약 1.62 s, .25에서 약 1.32 s였고 source 간 차이는 수 ms 수준이었다. Target contact가 censor 전에 없으므로 target Sand geom의 unintended early contact가 원인은 아니다. Canonical geometry는 pre-target 구간을 source terrain으로 유지한다. 결론은 late patch와 walking-policy prefix survival의 조합이며 simulator physics bug는 아니다.

## 5. Post-target fall/censor analysis

Discovery 47건은 target 후 낙상했다. 다수는 target 위 또는 exit 직후였고 moderate/severe에서 증가했다. Failed study의 width별 valid/strict는 NARROW `8/7`, MEDIUM `1/1`, WIDE `12/11`이어서 width 이름 자체는 severity를 설명하지 못했다. `patch_start + patch_width`로 결정되는 exit 위치와 gait entry timing의 결합이 더 중요했다.

Pilot 1은 calibrated exit `1.080–1.140 m`에서 15/24 strict를 얻었다. Invalid 9 중 네 low-speed right-entry run은 낙상 없이 8 s 종료에서 follow-up이 776–778 ms여서, label의 1,000 ms를 낮추지 않고 subsequent pilot과 redesign duration을 9 s로 연장했다. Pilot 2의 mild는 11/12 strict가 되어 이 선택을 검증했다.

## 6. Phase-control analysis

Failed valid Sand의 exact phase는 다음과 같다.

| Measurement | LEFT single | RIGHT single | DOUBLE | Interpretation |
|---|---:|---:|---:|---|
| First target-contact sample | 0 | 0 | 21 | touchdown 때문에 구조적으로 bilateral load가 생김 |
| First target contact -20 ms | 8 | 13 | 0 | 실제 approach gait는 두 single-support 상태를 포함 |

Pilot 1의 24개 planned entries도 20 ms pre-contact phase가 LEFT/RIGHT `12/12`였다. 따라서 기존 phase diversity는 물리적으로 전부 사라진 것이 아니라 측정 시점과 A/B/C/D control claim이 잘못됐다. 새 primary phase는 `first censor-valid target contact -20 ms` exact loaded-contact phase다. Contact-sample phase는 descriptive로 남긴다.

Direct A/B/C/D와 cohort ID는 제거한다. Explicit pre-roll/controller phase offset은 구현·검증하지 않았고 필요하다는 근거도 없다. Topology와 geometry를 넓게 predeclare한 뒤 model-blind post-generation phase를 측정·gate한다. Concrete/.25에서는 tested right-entry가 반복 불안정하므로 6개 cell 중 5개에 두 phase, 모든 cell에 최소 한 usable phase를 요구한다.

## 7. Geometry analysis

Pilot 1에서 geometry별 strict yield는 `4/6`, `3/6`, `5/6`, `3/6`이었다. 가장 안정적인 left anchor는 start/width `.340/.780`, nearby final anchor `.336/.786`이고, right anchor는 `.322/.778` 주변이다. Pilot 2에서는 calibrated left mild가 6/6, right mild가 5/6이었다.

새 matrix는 broad mild의 start `0.318–0.345 m`, width `0.766–0.834 m`; boundary-adjacent moderate의 start `0.305–0.346 m`, width `0.666–0.832 m`를 사용한다. Concrete/.25 moderate는 failed Discovery의 실제 strict-benign `.314/.678` left anchor 주변으로 condition한다. 이는 exact historical signature 재사용이 아니라 independently offset된 fresh local domain이다.

## 8. Severity-control analysis

Severity mechanics의 realized displacement는 의도대로 monotonic했다. 문제는 label mapping이 아니라 high compliance의 physical feasibility다.

| Pilot 2 severity | Runs | Strict | Slip | Invalid | Peak displacement |
|---|---:|---:|---:|---:|---|
| mild | 12 | 11 | 0 | 1 | 약 .020 m |
| moderate | 12 | 7 | 2 | 3 | 약 .040 m |
| severe | 12 | 0 | 3 | 9 | 약 .065 m |

Pilot 3에서 severe exposure를 width `.526–.582 m`까지 줄여도 0/24 strict였다. 따라서 severe total은 0/36 strict이며 새 benign domain에서 제외한다. 이를 moderate로 relabel하지 않는다.

새 near-hazard group의 정확한 의미는 `boundary-adjacent maximal calibrated benign mechanics`: strict physical labels를 만족하고 moderate mechanics에서 actual displacement `[.030,.0525) m`인 경우다. 다음 discrete severe tier가 일관되게 hazard/invalid로 전이되므로 어려운 benign boundary를 제공한다. 이는 과거의 `[.0525,.070] m` actual severity와 같다는 claim이 아니며 config에 claim limit를 명시했다.

## 9. Speed/source interactions

Pilot 2 strict yield는 Concrete .20/.25/.30=`3/1/4`, Marble=`4/3/3`이었다. Source보다 speed/topology interaction이 컸고, 특히 Concrete/.25 right entry가 mild/moderate에서 모두 조기 fall했다. Core axes Concrete/Marble × .20/.25/.30은 유지하되 physical equivalence를 위해 anchor topology/geometry를 cell-condition한다. 어떤 conditioning에도 model result는 사용하지 않았다.

## 10. Topology analysis

Transition-left는 주로 left leading foot과 RIGHT pre-contact support, transition-right는 right leading foot과 LEFT pre-contact support를 만들었다. 즉 topology는 label만 바꾸는 것이 아니라 meaningful approach/contact side를 바꾼다. 다만 target touchdown sample의 loaded side는 bilateral이므로 topology를 phase 자체로 간주하지 않는다.

Concrete/.25 broad/near-hazard와 일부 moderate cells는 안정성을 위해 left-only anchor를 쓴다. 전체 split에서는 양 leading foot과 양 pre-contact phase를 반드시 유지하고, cell-level gate로 과도한 collapse를 차단한다.

## 11. Realization mechanism

기존 `2026090201/2026090202` cohort는 ID 외 simulator state를 바꾸지 않았다. 이를 control factor에서 제거한다. 새 realization은 start/width의 작은 predeclared offset, topology, source/speed의 명시적 조합이고, 생성 후 first contact, duration, phase, leading foot, displacement, load transition, raw Pelvis RMS로 physical signature를 측정한다. 임의 noise나 result-driven seed search는 없다.

## 12. Ordinary and delayed Support controls

Old summary는 모든 fall을 먼저 invalid 처리해 Support가 충분히 확립·관측된 경우도 제거했다. Censor-aware corrected contract로 failed Discovery ordinary는 `2/12 -> 7/12`, delayed는 `3/4 -> 4/4`다. 이는 frozen historical 결과를 수정한 것이 아니라 새 설계를 위한 re-audit다.

Pilot 3은 historical known-good geometry 주변에서 ordinary Support 11/12를 재현했다. Right는 6/6, left는 5/6이며 유일한 실패는 Concrete/.30에서 Support 후 564 ms만에 fall한 경우였다. 새 ordinary controls는 benign geometry와 분리하고, 이 cell은 right anchor만 사용한다. Delayed controls는 failed Discovery corrected 4/4, original aggregate 6/8, Model V2 metadata에서 corrected 22/24의 충분한 근거가 있어 canonical staged-left mechanics를 유지한다.

## 13. Implementation issues

Simulator physics, terrain pre-contact leakage, walking policy에는 명확한 구현 bug가 없어서 변경하지 않았다. 발견한 문제는 다음 두 가지다.

1. `sand_study.py`가 first-contact sample의 loaded phase를 primary로 써 touchdown을 DOUBLE로 축약했다.
2. 같은 old summary가 group을 구분하지 않고 any fall을 invalid 처리해 이미 확립되고 1 s 이상 관측된 Support까지 버렸다.

Old study source/config/dataset은 frozen provenance이므로 in-place 변경하지 않았다. 새 canonical `sand_calibration.py`에 censor-aware target timing, 20 ms pre-contact phase, Support post-event follow-up contract, pilot freeze/integrity, failed Discovery table, redesigned matrix expansion/validation을 구현했다. Regression test는 touchdown-vs-precontact phase와 later-fall Support 보존을 고정한다. 첫 pilot 실행 시 잘못 적은 policy path가 preflight `FileNotFoundError`로 simulation 전에 실패했고 output도 생성되지 않았다. Path를 canonical ONNX로 고친 authoritative config SHA만 freeze한 뒤 24건을 실행했다.

## 14. Pilot decision and results

Decision은 `PILOT_REQUIRED`였다. Existing evidence만으로 9 s follow-up, severe feasibility, corrected Support behavior를 확정할 수 없었기 때문이다.

| Batch | Scientific question | Runs | Valid | Strict benign | Support | Slip | Invalid | Main conclusion |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | calibrated exit geometry가 두 pre-contact phase와 usable mild yield를 만드는가 | 24 | 15 | 15 | 0 | 0 | 9 | phase 12/12 balanced; 8 s가 low-speed follow-up에 짧음 |
| 2 | 9 s에서 mild/moderate/severe mapping과 cell interaction은 무엇인가 | 36 | 23 | 18 | 0 | 5 | 13 | mild 11/12, moderate 7/12, severe 0/12 |
| 3 | short exposure가 severe를 구조하고 known-good ordinary Support를 재현하는가 | 36 | 11 | 0 | 11 | 0 | 25 | severe 0/24; ordinary Support 11/12 |
| **Total** |  | **96** | **49** | **33** | **11** | **5** | **47** | hard ceiling 충족, model inference 0 |

각 batch는 첫 simulation 전에 exact parameters/config/signature hash를 저장했고 within-batch adaptation, replacement, backfill은 0이다.

| Batch | Config SHA-256 | Manifest SHA-256 | Dataset freeze identity |
|---|---|---|---|
| 1 | `1573238a9bb71e4b1f97d2f36ac20af9b626757b0de1926ad99f91b17fd875cf` | `a30f96943593c2322dde1898060919eab3ec9929167e358743335f1a809c4d77` | `0bf874a76b44e348d0c74d9c4b9caf0cf7c263f27058d0c9d434b8d9624e4214` |
| 2 | `db60250f86562253ab744d56f1ecd8ef739955aea63259506b7f29ca0408fcf7` | `ab481ddaa38f6fb84c9d3a7d0f0ac2a148463b4aa3eb341b4f8c3a29e9a37415` | `5473ebd8628844a4285d9245d03274211dfdfd1734dae9c85edb7b40e7cc108a` |
| 3 | `81e88615fd4f3c60d28a30ae5cc4191108675ebc5484b92077d3ae031223e31c` | `28274d1bad4f3bc42dd55b6cd4772feed6b6dde9ce48c536d775a640f12fba4a` | `6ae914b16cfb038bd3ac0ada0fe264394c6c360d2c786dcffc12269d14fd2a8a` |

## 15. Viable-domain conclusion

One unconditioned LOW/MEDIUM/severe matrix는 부적합하다. 그러나 9 s, mild broad domain, cell-conditioned moderate boundary domain, measured two-phase contract, separate Support geometry 조합은 full study를 fail-closed하게 시도할 충분한 물리 근거가 있다. Severe는 benign coverage에 포함하지 않고 별도 infeasible tier로 기록한다.

## 16. Factor decisions

| Factor | Failed-study behavior | Calibration evidence | New action | New domain/control | Confidence |
|---|---|---|---|---|---|
| source | Concrete/Marble 모두 low yield | mild는 양 source에서 유사, 일부 moderate 차이 | KEEP | both sources, symmetric counts | High |
| speed | .20 pre-target, .25 topology interaction | .30 pre-fall 0; Concrete/.25 right unstable | KEEP + CONDITION | .20/.25/.30 with physical anchors | High |
| patch start | nominal strata가 entry를 보장하지 않음 | `.318–.345` viable mild band | NARROW | group/cell anchors + offsets | High |
| patch width | MEDIUM label이 특히 붕괴 | exit position과 exposure가 더 설명적 | REDESIGN | mild `.766–.834`, moderate `.666–.832` | Medium |
| phase control | A/B/C/D가 simulator에 미적용 | -20 ms phase LEFT/RIGHT 확인 | MEASURE_ONLY | post-generation exact precontact phase | High |
| topology | leading side는 바꾸지만 entry load는 bilateral | topology가 opposite single-support approach를 생성 | KEEP + CONDITION | global both; unstable cells narrowed | High |
| realization mechanism | cohort ID만 다름 | no physical input path | REDESIGN | deterministic geometry variants | High |
| severity control | mapping은 monotonic, severe는 불안정 | mild 11/12, moderate 7/12, severe 0/36 | NARROW | mild + boundary-adjacent moderate; severe excluded | High |
| observation duration | low-speed exit 후 776–778 ms | 9 s mild 11/12 | REDESIGN | 9 s, follow-up remains 1,000 ms | High |
| ordinary Support control | reported 5/24 total | corrected Discovery 7/12; pilot 11/12 | RESTORE + SEPARATE | known-good lateral anchors | High |
| delayed Support control | 6/8 viable | corrected Discovery 4/4, historical 22/24 | KEEP | staged-left, .25, separate domain | High |

## 17. Redesigned study

New dataset concept/ID is `sand_benign_generalization_redesigned_study_20260902`. Full generation은 이 milestone에서 0이다.

| Group | Source | Speed | Domain diversity | Discovery N | Confirmation N | Total |
|---|---|---|---|---:|---:|---:|
| Broad Sand benign | Concrete + Marble | .20/.25/.30 | mild, cell anchors, 8 variants/cell | 48 | 48 | 96 |
| Near-hazard Sand benign | Concrete + Marble | .20/.25/.30 | boundary-adjacent moderate, 4 variants/cell | 24 | 24 | 48 |
| Ordinary Support | Concrete + Marble | .20/.25/.30 | separate lateral left/right or verified right-only cell anchors | 12 | 12 | 24 |
| Delayed Support | Concrete + Marble | .25 | two staged-left realizations/source/split | 4 | 4 | 8 |
| **Total** |  |  |  | **88** | **88** | **176** |

각 split은 Sand source-speed cell마다 broad 8 + boundary 4 = 12건을 독립적으로 갖는다. Confirmation은 failed study 또는 pilot run을 재사용하지 않으며 Discovery와 다른 exact/near signatures다.

## 18. Generation gates

| Gate | Requirement | Rationale |
|---|---|---|
| Attempted/completed | 176/176 manifest records | missing run 은 숨기지 않음 |
| Adaptive behavior | backfill 0, replacement 0, split move 0 | outcome-driven selection 방지 |
| Overall objective valid | >=132/176 | old 55/176 반복 차단 |
| Strict Sand per split | >=54/72 | original 75% target 유지 |
| Strict Sand per cell/split | >=8/12 | handful-only cell 방지 |
| Broad mild per split | >=40/48 | pilot 11/12에 근거한 robust base |
| Boundary moderate per split | >=12/24 | difficult benign을 실제 population으로 유지 |
| Boundary moderate per cell/split | >=1/4 | 모든 source-speed에 boundary evidence 필요 |
| Severe benign | required 0 | 0/36 strict를 relabel하지 않음 |
| Global precontact phase | LEFT/RIGHT single 두 category/split | DOUBLE-only collapse 금지 |
| Cell phase | 5/6 cells에서 두 category, 6/6에서 >=1 usable | Concrete/.25 feasibility를 명시적으로 제한 |
| Leading foot | split global both | topology collapse 방지 |
| Entry-time span | global >=250 ms/split | temporal realization 확보 |
| Ordinary Support | >=10/12/split and >=1/cell | pilot 11/12와 cell usability |
| Delayed Support | >=3/4/split | historical/failed-study viability |
| Physical uniqueness | valid signature fraction >=.80 | nominal-only variation 방지 |
| Scenario integrity | unique 100%, historical/exact/near split overlap 0 | contamination 방지 |
| Gate failure | model/Confirmation analysis 전에 중단 | physical failure를 model science로 오염시키지 않음 |

## 19. Discovery and Confirmation protocol

두 split은 한 frozen matrix 아래 함께 생성하되 `REDESIGNED_CONFIRMATION`은 즉시 sealed한다. Discovery physical/diversity gates가 모두 통과하고 exactly one H1/H2/H3 interpretation과 metric hash가 freeze될 때까지 Confirmation raw/model/80D/observability/visualization을 열지 않는다. Open 후 label 교체, threshold 변경, model tuning은 금지한다.

Future frozen V2 diagnostic은 exact `model_v2_anchor_refined_gru20_20260902`, threshold .99/5 ms를 최대 Discovery 1회, Confirmation 1회만 사용한다. 이 milestone에서는 0회다. H1 domain diversity, H2 Pelvis observability, H3 representation/capacity hierarchy와 unique-match requirement를 새 config에 고정했다.

## 20. Historical contamination audit

- New matrix: 176/176 unique run IDs, 176/176 unique scenario signatures
- Historical exact overlap across Unified 256, Model V2 412, Generalization 72, Ice semantics 48, failed Sand 176, pilots 24/36/36: 0
- Redesigned Discovery/Confirmation exact overlap: 0
- Redesigned Discovery/Confirmation parameter-near overlap: 0
- Consumed HOLDOUT exact start `.362` 사용: 0
- Consumed HOLDOUT exact width `.735` 사용: 0
- Failed-study Discovery/Confirmation split reuse: 0
- Pilot split reuse: 0
- Old HOLDOUT payload access: 0; guard remains 1
- Failed-study Confirmation model analysis: 0

Expanded scenario signature SHA-256은 `56dd39cb6d05b4c1908f2babf8e5b40309db97f92e79af99e91256bb9f1fb1cc`; split SHA는 Discovery `2b4e04f5de74501cadc72f9ce0a0a216a3af6e935f853dd22fe785a36278eade`, Confirmation `729a75ed4c6ae86c740b9d27c43c9ae1a5ae66fee8e48adef5e536ff95c3a494`다.

## 21. New design hashes

Design config file SHA-256은 `40dfdcda42ebabe324ae243904b8fb8154f0701f803f0cabe051baae23d83a9c`다.

| Object | SHA-256 |
|---|---|
| `REDESIGNED_PARAMETER_DOMAIN_SHA` | `02429ef22abe2670b7f812ac4a0586b29512eb74a99bb48e6e26e5997f019316` |
| `REDESIGNED_SCENARIO_MATRIX_SHA` | `51968c46a67de3a337b9858f93be8e25998fa8a15394760f16964fbb25a0a940` |
| `REDESIGNED_SPLIT_PLAN_SHA` | `80531f137065cdddb13633595c14de8c06ad1169165331d40dafec830b760d5d` |
| `REDESIGNED_PHYSICAL_LABEL_CONTRACT_SHA` | `030e85c6561b3fb9b757a2beb82e15f583264cbfe203b016b0070f82888bb39c` |
| `REDESIGNED_GENERATION_GATE_SHA` | `914ac15e564f8218bfc386e3cfbbdeac292a4f66a9c2df5611a6ff7dcf92ec34` |
| `REDESIGNED_DIVERSITY_METRIC_SHA` | `f9fbfa697d33aae328848ee3ef3bfe420e63a3e1a8da54c35b975ffa47942e15` |
| `REDESIGNED_CONFIRMATION_PROTOCOL_SHA` | `daca735533c229911f6179e7265333f75a8260ef3b5f6a20c7edf31f9fbb1781` |
| `SAND_BENIGN_GENERALIZATION_STUDY_REDESIGN_SHA` | `d853645939002f2460c99fe865ee4cc39c25334a5cb51dc5f489ec36202c05d2` |

## 22. Architecture and sensor boundary

Historical Model V2 candidate와 architecture는 untouched다. Calibration 성공/실패는 sensor 또는 model conclusion이 아니다. `PELVIS_ONLY_HAZARD_STILL_PLAUSIBLE`, `ARCHITECTURE_EVIDENCE_STILL_FAVORS_DATA_DOMAIN_STUDY`는 provisional statement로 유지하며 final sensor architecture는 unfrozen이다. FSR은 physical label/control diagnostic에만 사용했고 Hazard fusion은 0이다.

## 23. Counters

| Counter | Actual |
|---|---:|
| optimizer steps | 0 |
| checkpoint writes | 0 |
| normalizer fits | 0 |
| HNM rounds | 0 |
| threshold searches | 0 |
| persistence searches | 0 |
| architecture searches | 0 |
| model inference runs | 0 |
| old HOLDOUT payload reads | 0 |
| old HOLDOUT inference | 0 |
| failed-study Confirmation model analysis | 0 |
| new calibration simulation runs | 96 |
| full redesigned-study generation runs | 0 |

## 24. Limitations

- Deterministic current Unitree G1 walking policy 하나의 simulation domain 결과다.
- Explicit pre-roll 또는 controller phase offset은 구현하지 않았다.
- Severe .065 m tier는 near-hazard benign으로 제공할 수 없다. 새 boundary-adjacent semantics는 moderate .040 m까지만 주장한다.
- Concrete/.25는 tested right-entry Sand가 불안정하여 cell-level two-phase exception이 있다.
- Final 176-run outcome은 아직 생성되지 않았다. Frozen gates 중 하나라도 실패하면 model analysis 전에 중단한다.

## 25. Tests and integrity

Targeted regression은 censor-aware phase, later-fall Support, 176-run expansion, hashes, exact/near overlap, no-model collector path를 검증한다. Full pytest는 `121 passed, 1 skipped`, compileall과 Ruff critical `E9/F63/F7/F82`, `git diff --check`는 PASS다. Historical implementation/model/data artifact 28개, failed-study NPZ 176개, pilot NPZ 96개와 config hashes를 재검증했고 consumed HOLDOUT guard는 1이다.

## 26. Final verdict

`SAND_BENIGN_GENERALIZATION_STUDY_REDESIGN_READY`

Physical root cause와 구현/measurement defects는 충분히 분리됐고, severe를 억지로 보존하지 않는 새 domain, objective gates, fresh matrix, split sealing 및 hashes가 모두 freeze됐다.

## 27. Recommended next milestone

`SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION`

이 milestone은 시작하지 않았다.
