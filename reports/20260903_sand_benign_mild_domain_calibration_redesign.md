# Sand Benign Mild Domain Calibration and Redesign

## 1. Purpose

이 milestone은 redesigned Sand study의 residual broad-mild physical instability를 model-blind하게 국소화하고, 다음 fresh 176-run study에 사용할 최소 mild geometry/exposure domain을 동결한다. 결론은 width 단독 제한이 아니라 topology-conditioned joint start/width/exit domain과 Concrete/.25의 단일 left-only 예외다.

## 2. Starting state

- Starting HEAD / `origin/main`: `3e6a6306b5f52d333b67571921f0f867e5513b03`, parity
- Starting tracked worktree: clean
- Previous verdict: `SAND_BENIGN_REDESIGNED_DOMAIN_FAILURE_REVIEW_ACTIONABLE`
- Previous feasibility: `MILD_DOMAIN_RECALIBRATION_REQUIRED`
- Calibration contract SHA-256: `0ac6a6017b38fc412c6311a8ac89c513bd0381e15317de5e7c6dc623bb11c7bd`
- Historical final Model V2 verdict: `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- Whole-simulation status: `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- Support branch: `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`

Contract는 detailed trace analysis 및 새 simulation 전에 `configs/experiment/20260903_sand_benign_mild_domain_calibration_redesign.yaml`에 동결했고 이후 수정하지 않았다.

## 3. Historical scientific boundary

사용한 증거는 current redesigned study, 기존 96-run Sand calibration, 새 72-run physical calibration의 manifest와 physical traces뿐이다. Model V1/V2/Terrain inference, Hazard probability, threshold/reflex score, normalized 80D, observability, GRU state, margin, training/HNM, tuning 및 sensor experiment는 모두 0이다.

Consumed Generalization HOLDOUT은 raw payload read, inference, feature reconstruction, visualization이 모두 0이고 guard/scientific-open count는 `1/1` 그대로다. Current redesigned Confirmation에는 physical calibration analysis만 수행했고 model science는 0이다.

## 4. Current redesigned-study failure

Current study는 objective-valid 153/176, strict Sand 116, Support 32, Slip 5로 첫 study보다 크게 개선됐다. Discovery physical gates와 moderate, Support, phase, entry, diversity, contamination은 모두 통과했다. 남은 실패는 Confirmation mild 35/48, Concrete/.25 strict 7/12, strict total 52/72의 세 gate다.

Mild 96건은 strict 80, invalid 16이다. Invalid는 pre-target fall 2와 post-target fall-censor 14이며 stable horizon censor는 0이다. 따라서 observation duration은 9초로 유지한다.

## 5. Mild-only physical ledger

Model-blind ledger는 current mild 96건과 pilot 72건, 총 168행이다. 각 행은 source/speed/topology/start/width/exit, target/fall timing, loaded Sand exposure, contact episodes, phase/lead/load side, displacement/spread/load metrics와 objective outcome을 포함한다. Model field는 없다.

- Gitignored ledger: `artifacts/runs/20260903_sand_benign_mild_domain_calibration_redesign/physical_ledger.json`
- Rows: current 96 + new calibration 72 = 168
- SHA-256: `8c8465a54a765273d64b2cb134739414e1cc73715724bceffe1218660af2372e`

| Split | Source | Speed | Planned | Strict | Invalid | Pre-target | Post-target fall | Width range | Start range |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| D | concrete | .20 | 8 | 7 | 1 | 0 | 1 | .768–.798 | .318–.344 |
| D | concrete | .25 | 8 | 8 | 0 | 0 | 0 | .766–.792 | .320–.344 |
| D | concrete | .30 | 8 | 8 | 0 | 0 | 0 | .768–.798 | .318–.344 |
| D | marble | .20 | 8 | 7 | 1 | 0 | 1 | .768–.798 | .318–.344 |
| D | marble | .25 | 8 | 7 | 1 | 0 | 1 | .768–.798 | .318–.344 |
| D | marble | .30 | 8 | 8 | 0 | 0 | 0 | .768–.798 | .318–.344 |
| C | concrete | .20 | 8 | 5 | 3 | 1 | 2 | .804–.834 | .319–.345 |
| C | concrete | .25 | 8 | 5 | 3 | 0 | 3 | .802–.828 | .321–.345 |
| C | concrete | .30 | 8 | 8 | 0 | 0 | 0 | .804–.834 | .319–.345 |
| C | marble | .20 | 8 | 5 | 3 | 1 | 2 | .804–.834 | .319–.345 |
| C | marble | .25 | 8 | 5 | 3 | 0 | 3 | .804–.834 | .319–.345 |
| C | marble | .30 | 8 | 7 | 1 | 0 | 1 | .804–.834 | .319–.345 |

Current strict runs의 loaded-Sand exposure는 2,014–5,061 ms, median 3,042.5 ms이고 episode는 7–19, median 14다. Target-reaching invalid의 exposure는 334–3,461 ms, median 692 ms이고 episode는 1–17, median 4다. 짧은 invalid exposure는 원인이 아니라 fall censor의 결과다.

## 6. Width/start interactions

Current width `.802–.812`는 19/19 strict였지만 `.798`은 2/5, `.826–.834`는 7/17 strict였다. Pilot A에서도 left `.327/.805`와 `.335/.811`은 12/12 strict지만 right `.321/.807`, `.327/.817`, `.335/.825`는 각각 4/6, 4/6, 2/6이었다.

동일 command cell에서 1–6 mm start 이동 또는 patch exit 변화가 Sand contact episode selection을 바꾼다. 따라서 raw width를 독립 threshold로 쓰지 않고 topology별 joint `(start, width, exit)` envelope로 제한한다.

| Factor interaction | Stable evidence | Unstable evidence | Interpretation | Confidence |
|---|---|---|---|---|
| start × width | C verification joint points 24/24 | `.798` 2/5지만 `.802–.812` 19/19 | 두 값과 exit를 함께 보아야 함 | HIGH |
| start × topology | A left starts .327/.335는 12/12 | same neighborhood의 right starts .321/.327은 8/12 | start의 의미는 topology가 선택한 contact sequence에 의존 | HIGH |
| width × speed | .30 pilot 12/12, 넓은 A outer도 strict | .20/.25에 pilot invalid 11/11 집중 | higher speed는 tested tail을 견디지만 축은 유지 | HIGH |
| width × source | B/C right band는 eligible source 모두 strict | A right-inner는 C/.20 strict, M/.20 fall | source-only rule이 아니라 contact-sequence interaction | MODERATE |
| width × topology | A left inner/middle 12/12 | A right profiles 10/18 | topology별 envelope 필요 | HIGH |
| width × phase | verified RIGHT/LEFT phase 모두 존재 | invalid 10/11은 right-topology의 LEFT phase | phase는 topology와 confounded, direct control 아님 | MODERATE |
| start × speed | left verification across starts 14/14 | C/.25 right는 tested starts 전부 fall | one cell-topology exception을 지지 | HIGH |
| source × speed | B/C eligible cells 30/30 right strict | A M/.20과 M/.25 일부 geometry fall | six unrelated source domains 불필요 | MODERATE |
| source × topology | every source는 C에서 viable left를 보임 | right instability는 두 source에 존재하지만 C/.25만 반복적 | source별 topology set보다 단일 cell exception이 최소 | MODERATE |
| speed × topology | .30 left/right pilot 12/12 | C/.25 right 0/5 in new pilots | Concrete/.25 left-only가 simple physical rule | HIGH |
| speed × phase | Pilot C both phases 10/14 and 24/24 strict | Concrete/.25 LEFT precontact phase/right topology 반복 fall | phase 자체보다 realized topology/cell interaction | MODERATE |
| source × phase | both source에서 global LEFT/RIGHT phase와 strict evidence 존재 | A의 right/LEFT-phase fall은 두 source에 걸침 | source-conditioned phase control 근거 없음 | MODERATE |

## 7. Source/speed interactions

새 pilot 72건은 각 source-speed cell을 12건씩 포함했다.

| Source | Speed | Pilot strict | Pilot total | Main observation |
|---|---:|---:|---:|---|
| concrete | .20 | 10 | 12 | right middle/outer가 fall, verified inner band는 4/4 |
| concrete | .25 | 6 | 12 | right 0/5, left verification 4/4 |
| concrete | .30 | 12 | 12 | tested left/right region 전체 strict |
| marble | .20 | 10 | 12 | right outcome가 geometry-dependent, verified band 4/4 |
| marble | .25 | 11 | 12 | right outer만 fall, verified band 4/4 |
| marble | .30 | 12 | 12 | tested left/right region 전체 strict |

Source 단독 conditioning 근거는 없다. Speed/topology interaction은 있으며 Concrete/.25 right만 세 geometry에서 반복적으로 실패했다. 나머지 다섯 cell에서는 동일 right verification band가 strict 20/20이었다.

## 8. Topology/phase interactions

Pilot A는 left/right 18/18, phase RIGHT/LEFT single 18/18을 직접 비교했다. Left는 17/18 strict, right는 10/18 strict였다. Pilot B right 12건은 Concrete/.25 두 건만 실패했다. Pilot C는 five-cell left/right와 Concrete/.25 left-only rule로 phase LEFT/RIGHT `10/14`, strict 24/24를 얻었다.

Topology는 phase label의 대체물이 아니다. 이 simulator realization에서 left topology는 주로 RIGHT precontact support, right topology는 LEFT precontact support를 만든다. Exact phase는 계속 contact -20 ms에서 측정하고 A/B/C/D nominal phase control은 도입하지 않는다.

## 9. Exposure-duration analysis

Current strict/invalid loaded exposure median은 3,042.5/692 ms이고 pilot strict/invalid는 3,044/365 ms다. Pilot C strict 24건은 loaded exposure 2,021–4,466 ms, episode 8–17을 안정적으로 포함했다. 즉 long physical exposure 자체가 hazard라는 단조 결론은 맞지 않는다.

Cumulative exposure는 fall로 censor되므로 future design input threshold로 직접 사용할 수 없다. 가장 단순하고 재현 가능한 control은 topology와 joint start/width/exit geometry이며, post-generation exposure는 diversity/diagnostic metric으로만 사용한다.

## 10. Entry vs residence vs exit instability

Current target-reaching mild invalid 14건 중 10건은 target contact가 지속되는 동안 fall했고 4건은 마지막 target contact 뒤 47–61 ms에 fall했다. Pilot invalid 11건 중 9건은 target contact 중, 2건은 마지막 contact 뒤 62/210 ms에 fall했다.

Pilot invalid target→fall은 360–5,907 ms, median 406 ms이고 episode는 1–16이다. 많은 실패가 첫 1–4 episode에서 발생하지만 일부는 repeated residence 뒤 발생한다. Pilot C의 8–17 stable episodes는 residence length만으로 failure를 설명할 수 없음을 보여준다. 결론은 entry가 선택하는 contact sequence와 on-Sand residence/후속 transition의 interaction이며, exit-only 또는 horizon-only 문제가 아니다.

## 11. Common-domain hypothesis

하나의 unconditioned left/right rectangle은 기각한다. 특히 Concrete/.25 right가 pilot A/B와 이전 calibration의 세 independent geometry에서 strict 0/6인 반면 같은 geometry family의 다른 다섯 cell은 15/15 strict였다.

다만 geometry 숫자를 source/speed별 여섯 세트로 분할할 필요도 없다. 공통 left envelope는 모든 cell에, 공통 right envelope는 physically viable한 다섯 cell에 동일 적용할 수 있다.

## 12. Conditional-domain hypothesis

선택은 `SOURCE_SPEED_CONDITIONED_MILD_DOMAIN`이지만 conditioning은 하나뿐이다.

```text
all six cells:
  common transition-left envelope

all cells except Concrete/.25:
  common transition-right envelope

Concrete/.25:
  transition-left only
```

이는 missed gate 한 건을 구제하기 위한 최적화가 아니다. Concrete/.25 right는 target가 약 1.507 s에 도달한 뒤 여러 fresh geometry에서 반복 fall했고, same right band가 다른 cell에서 재현 가능했다.

## 13. Pilot decision

`PILOT_REQUIRED`

Current non-monotonic `.798–.834` evidence만으로는 robust envelope와 topology exception을 동결할 수 없었다. Contract의 72-run/3-batch ceiling 아래 localization, interaction replication, independent domain verification을 순차 수행했다.

## 14. Pilot batches

| Batch | Question | Runs | Config SHA-256 | Manifest SHA-256 |
|---|---|---:|---|---|
| A | six shared left/right geometry가 stable/transition/unstable region을 분리하는가 | 36 | `04a42b67dc964b2d49e4f341735dd807e1bbf2191356cd3cc4c4203710e27b66` | `1f5f7fe0e83a55a7e4be8240c3ff4ab2282e00194d6151c3ac909ac1a697c14f` |
| B | shorter right band가 five cells에서 재현되고 Concrete/.25 예외가 반복되는가 | 12 | `b2e296739708fd00e502e108afaf986ea333b09bfb65bdad9ed951f74a7bd183` | `b57603925da065f01657d65bacaf93d9bebd0859e319c32f9973ce7c22c6335a` |
| C | proposed conditional domain이 overall >=85%와 cell >=3/4를 만족하는가 | 24 | `b51e74871235be578ef62686f7ad440450a392693a7365b37502da33011be1bd` | `90bb0ce2cd9fb2e52f57bc45ad247a64e1217d477e4b098d016fe3393d1dcb50` |

각 config는 run 1 전에 freeze됐고 unique/fresh signature와 historical exact overlap 0을 preflight했다. Within-batch adaptation, replacement, backfill은 0이다.

## 15. Pilot physical results

| Batch | Question | Runs | Strict benign | Slip | Support | Invalid | Phase coverage | Main conclusion |
|---|---|---:|---:|---:|---:|---:|---|---|
| A | boundary localization | 36 | 27 | 0 | 0 | 9 | LEFT/RIGHT 18/18 | left 17/18; right outer unstable; C/.25 right 0/3 |
| B | interaction replication | 12 | 10 | 0 | 0 | 2 | LEFT 12 | failures exactly C/.25 right 2/2 |
| C | final domain verification | 24 | 24 | 0 | 0 | 0 | LEFT/RIGHT 10/14 | 100%, each cell 4/4 |
| **Total** |  | **72** | **61** | **0** | **0** | **11** | LEFT/RIGHT 40/32 | ceiling reached by planned verification |

Invalid 11건은 전부 target 이후 physical fall censor이며 horizon censor는 0이다. All outcomes were retained.

## 16. Stable/transition/unstable mild region

| Region | Physical evidence | Interpretation |
|---|---|---|
| Stable left | A inner/middle 12/12; C common-left 12/12; C/.25 additional left 2/2 | start `.325–.342`, width `.795–.811`, exit `1.124–1.146`의 tested joint region |
| Stable right | B/C eligible five cells 20/20 | start `.324–.330`, width `.791–.794`, exit `1.116–1.122` 근방 |
| Transition | A left outer 5/6; right inner/middle 4/6 each; historical `.798` 2/5 | cell/contact-sequence interaction이 outcome을 바꾸는 경계 |
| Unstable | A right outer 2/6; historical `.826–.834` 7/17; C/.25 right 0/6 | next broad-mild domain에서 제외 |

범위는 이 simulator/policy에서 검증된 design envelope이며 보편적 물리 threshold로 주장하지 않는다.

## 17. Final mild domain

Future exact matrix는 다음 bounded rule을 사용한다.

- Common left, all six cells: start `.325–.342`, width `.795–.809`, exit `1.123–1.147`
- Common right, all except Concrete/.25: start `.324–.330`, width `.791–.794`, exit `1.115–1.123`
- Concrete/.25: left-only, 8 fresh variants per split
- Other five cells: left 6 + right 2 fresh variants per split
- Mechanics: mild, expected displacement approximately .020 m, unchanged

Pilot C의 actual future-rule verification은 strict 24/24, each cell 4/4이고 loaded exposure, episodes, entry time 및 both phases를 유지했다.

| Source | Speed | Tested physical region | Stable region | Transition region | Unstable region | Future domain |
|---|---:|---|---|---|---|---|
| concrete | .20 | start .321–.341, width .791–.825 | all tested left; B/C right band | A right middle/outer | repeated right fall at starts .327/.335 | common left + common right |
| concrete | .25 | start .321–.342, width .791–.825 | left inner/middle and C four-left 4/4 | left outer | right 0/5 new, 0/6 with prior replication | common left only |
| concrete | .30 | start .321–.341, width .791–.825 | pilot 12/12 | none observed inside pilot | none observed inside pilot | common left + common right |
| marble | .20 | start .321–.341, width .791–.825 | all tested left; B/C right band | A right geometries | A right inner/outer falls | common left + common right |
| marble | .25 | start .321–.341, width .791–.825 | all tested left; A inner/middle and B/C right | A right outer boundary | A right outer fall | common left + common right |
| marble | .30 | start .321–.341, width .791–.825 | pilot 12/12 | none observed inside pilot | none observed inside pilot | common left + common right |

`Stable/transition/unstable`은 tested discrete points의 classification이며 untested 연속 조합에 대한 보편적 보장은 아니다. Future exact matrix는 stable column보다 더 좁은 frozen profiles만 사용한다.

## 18. Preserved study components

| Component | Status | Future action |
|---|---|---|
| Concrete/Marble | solved axis | KEEP equal counts |
| .20/.25/.30 | solved axis | KEEP equal counts |
| Broad mild | recalibrated | CHANGE geometry profiles only |
| Moderate | D 19/24, C 17/24, gates pass | KEEP unchanged domain |
| Ordinary Support | 12/12 each split | KEEP |
| Delayed Support | 4/4 each split | KEEP |
| Phase measurement | passed | KEEP exact contact -20 ms |
| Topology | physically meaningful | KEEP; one C/.25 left-only exception |
| Realization | unique geometry variants | KEEP strategy; use fresh values |
| Observation duration | horizon hypothesis rejected | KEEP 9 s |
| Physical labels | censor-aware and effective | KEEP exact contract |
| Generation gates | scientifically meaningful | KEEP thresholds unchanged |
| Confirmation protocol | protected | KEEP sealed protocol |

## 19. Future scenario matrix

Dataset concept는 `sand_benign_generalization_mild_recalibrated_study_20260903`이다.

| Group | Discovery | Confirmation | Total | Change |
|---|---:|---:|---:|---|
| Broad mild Sand | 48 | 48 | 96 | recalibrated profiles |
| Boundary moderate Sand | 24 | 24 | 48 | domain unchanged, fresh signatures |
| Ordinary Support | 12 | 12 | 24 | domain unchanged, fresh signatures |
| Delayed Support | 4 | 4 | 8 | domain unchanged, fresh signatures |
| **Total** | **88** | **88** | **176** | full generation not run |

Expanded matrix는 run ID/signature 176/176 unique, historical/current/pilot exact overlap 0, run-ID reuse 0, split exact/forbidden-near overlap 0이다. Expanded scenario signature SHA-256은 `be6d6f0d6bb312617784bad31b55cedfd686bb32aaebd37235efe7a24345fad1`이다.

## 20. Future physical generation gates

Threshold는 이전 redesign과 동일하다: completed 176, objective-valid >=132, strict Sand >=54/72 per split, cell >=8/12, mild >=40/48, moderate >=12/24 및 cell >=1/4, ordinary Support >=10/12 및 cell >=1, delayed >=3/4, global both phases, >=5/6 both-phase cells, every cell usable, both leading feet, entry span >=250 ms, physical uniqueness >=80%, contamination/exact/near overlap 0, backfill/replacement 0.

Gate SHA-256도 기존과 같은 `914ac15e564f8218bfc386e3cfbbdeac292a4f66a9c2df5611a6ff7dcf92ec34`다. 실패 편의를 위한 완화는 없다.

## 21. Discovery/Confirmation protocol

Future `MILD_RECALIBRATED_CONFIRMATION`은 generation 직후 sealed다. 모든 physical gates 통과 뒤 Discovery physical/diversity audit를 먼저 완료하고 exactly one interpretation과 metric hash를 동결한 뒤에만 Confirmation을 열 수 있다. Model tuning, hypothesis replacement, Confirmation-driven selection은 금지한다.

## 22. Historical contamination audit

- Historical manifests checked: Unified 256, Model V2 412, Generalization 72, Ice 48, first Sand 176, prior pilots 96, current redesign 176, new pilots 72
- Future planned exact overlap: 0
- Historical run ID reuse: 0
- Discovery/Confirmation exact overlap: 0
- Discovery/Confirmation forbidden near overlap: 0
- Current study reuse: 0
- Pilot reuse: 0
- Old HOLDOUT raw reads: 0; guard remains 1

Historical-near matches are non-gating diagnostics; only exact historical reuse is forbidden by the frozen rule. The future split-to-split near rule remains gating and is 0.

## 23. Architecture/model boundary

V1/V2/Terrain inference, optimizer steps, checkpoint writes, normalizer fits, HNM rounds, threshold/persistence searches, architecture searches 및 FSR fusion은 모두 0이다. No model field entered the 168-row ledger. Full 176-run generation은 0이다.

## 24. Limitations

이 calibration은 deterministic Unitree G1 simulator, frozen walking policy와 mild mechanics에 한정된다. Joint geometry bounds는 independently tested points의 bounded design envelope이지 연속 공간 전체의 universal guarantee가 아니다. C/.25 right exception은 replicated physical evidence에 근거하지만 future generation은 여전히 fail-closed gates를 통과해야 한다. Calibration yield는 Sand model generalization evidence가 아니다.

## 25. Verdict

`SAND_BENIGN_MILD_DOMAIN_CALIBRATION_REDESIGN_READY`

New minimal design status는 `SAND_BENIGN_MILD_DOMAIN_MINIMAL_REDESIGN_READY`다.

Design config file SHA-256은 `1301d64391b423eb50b2ac4188058e1ac0cd988dce477323f87891b946aeaeb5`다.

| Hash object | SHA-256 |
|---|---|
| `MILD_RECALIBRATED_PARAMETER_DOMAIN_SHA` | `16b79593193fa5925cbbc2fa73dfef3764e655b4911f4b0b5dae7f2c3add0b74` |
| `MILD_RECALIBRATED_SCENARIO_MATRIX_SHA` | `24a4833c869a25df10cff387d9e58d02ca8079eca04ce1f611c0cf54347976ca` |
| `MILD_RECALIBRATED_SPLIT_PLAN_SHA` | `656f807d1a8bc11df6529826cb25a99d7243a32d129704f61f02f9402b102104` |
| `MILD_RECALIBRATED_PHYSICAL_LABEL_CONTRACT_SHA` | `030e85c6561b3fb9b757a2beb82e15f583264cbfe203b016b0070f82888bb39c` |
| `MILD_RECALIBRATED_GENERATION_GATE_SHA` | `914ac15e564f8218bfc386e3cfbbdeac292a4f66a9c2df5611a6ff7dcf92ec34` |
| `MILD_RECALIBRATED_DIVERSITY_METRIC_SHA` | `f9fbfa697d33aae328848ee3ef3bfe420e63a3e1a8da54c35b975ffa47942e15` |
| `MILD_RECALIBRATED_CONFIRMATION_PROTOCOL_SHA` | `f423ef4996173b9562c4f382849ea1b2ce9c212c10fbd5f324309ee7b6610cad` |
| `SAND_BENIGN_MILD_RECALIBRATED_STUDY_REDESIGN_SHA` | `09c2e1a22d47ba115dc2ef3db0251a7dd836096ffe2b9e370fbe9d1677416356` |

## 26. Recommended next milestone

`SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_GENERATION`

자동 시작하지 않는다. 이 milestone은 calibration 72건과 future design freeze에서 종료하며 full 176 generation, V2 replay, Confirmation model analysis, 80D/FSR observability, training과 Model V3는 수행하지 않는다.
