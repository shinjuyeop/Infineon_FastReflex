# Sand Benign Generalization Study Mild-Recalibrated Redesign

## 1. Decision

`SAND_BENIGN_MILD_DOMAIN_MINIMAL_REDESIGN_READY`

이 문서는 model-blind mild calibration 뒤 동결한 다음 Sand development study의 exact design contract를 요약한다. Full dataset generation, Model V1/V2/Terrain inference, 80D/FSR observability, training 및 tuning은 이 milestone에서 수행하지 않았다.

## 2. Calibration basis

Current redesigned study의 mild 96건은 strict 80, invalid 16이었다. Width 단독으로는 `.798`에서 2/5, `.802–.812`에서 19/19, `.826–.834`에서 7/17로 non-monotonic했고, target-reaching invalid 14건은 모두 physical fall censor였다.

Fresh model-blind pilot 72건은 A/B/C `36/12/24`로 사전 동결했다. A/B는 boundary와 source-speed/topology interaction을 국소화했고, final verification C는 strict 24/24, 각 source-speed cell 4/4, Slip/Support/invalid 0을 기록했다. Concrete/.25 right는 세 geometry에서 0/6이지만 같은 geometry family는 다른 다섯 cell에서 15/15였으므로 단일 predeclared topology exception이 정당화된다.

## 3. Minimal change

변경하는 것은 broad-mild joint start/width/exit/topology domain뿐이다.

| Component | Frozen action |
|---|---|
| Concrete/Marble and .20/.25/.30 | keep balanced |
| Mild mechanics | keep approximately .020 m displacement |
| Moderate Sand | keep boundary-adjacent domain |
| Ordinary/delayed Support | keep domains and counts |
| Phase | keep exact loaded-contact phase at target contact -20 ms |
| Observation | keep 9 seconds |
| Physical labels | keep censor-aware contract |
| Generation/diversity gates | keep thresholds |
| Confirmation | keep sealed protocol |

## 4. Recalibrated mild domain

Geometry는 width 단독 threshold가 아니라 joint `(start, width, exit, topology)` profile이다.

| Applicability | Topology | Start (m) | Width (m) | Exit (m) | Per-cell/per-split profiles |
|---|---|---:|---:|---:|---:|
| All six source-speed cells | transition-left | .325–.342 | .795–.809 | 1.123–1.147 | 6, except Concrete/.25 uses 8 |
| All except Concrete/.25 | transition-right | .324–.330 | .791–.794 | 1.115–1.123 | 2 |
| Concrete/.25 | right excluded | — | — | — | 0 |

이 범위는 frozen simulator/policy에서 검증된 bounded design envelope이며 universal physical threshold가 아니다. Cumulative loaded-Sand exposure는 outcome censor의 영향을 받으므로 input threshold가 아니라 post-generation diversity/diagnostic metric으로만 사용한다.

## 5. Exact future matrix

- Dataset ID: `sand_benign_generalization_mild_recalibrated_study_20260903`
- Duration: 9 seconds
- Discovery: `MILD_RECALIBRATED_DISCOVERY`, 88
- Confirmation: `MILD_RECALIBRATED_CONFIRMATION`, 88
- Broad mild: 48 per split, 96 total
- Boundary moderate: 24 per split, 48 total
- Ordinary Support: 12 per split, 24 total
- Delayed Support: 4 per split, 8 total
- Total: 176

각 non-Concrete/.25 mild cell은 split마다 left 6/right 2이고, Concrete/.25는 left 8이다. Moderate와 Support는 semantics/domain을 바꾸지 않고 fresh exact geometry variants만 사용한다.

## 6. Freshness and split isolation

Expanded matrix는 run ID와 scenario signature 모두 176/176 unique다. Unified, Model V2, Generalization, Ice, 두 이전 Sand study, 96-run prior Sand calibration, 새 mild pilots 72와 비교한 historical exact overlap은 0이고 run-ID reuse도 0이다. Discovery–Confirmation exact overlap과 frozen parameter-near overlap은 각각 0이다.

Historical-near match는 design proximity를 나타내는 non-gating diagnostic이며 exact reuse가 아니다. Consumed Generalization HOLDOUT payload read는 0이고 permanent guard는 1이다.

| Object | SHA-256 |
|---|---|
| Expanded matrix | `54843d27ea8155d51823ec713e90f029415e2c7dda6de2f427a928f9ac3a33e6` |
| Expanded signatures | `be6d6f0d6bb312617784bad31b55cedfd686bb32aaebd37235efe7a24345fad1` |
| Discovery IDs | `1c211c38ee2bd7f9e9a44e0f81ec6a0dd110a8e14809c12777c043d33928f93f` |
| Confirmation IDs | `3bfbb050db3ebadcc363d1c6e51013dc349dbac4e0594c183a55245ea38c2e80` |

## 7. Physical generation gates

Complete records 176, objective-valid at least 132, strict Sand at least 54/72 per split and 8/12 per cell, mild at least 40/48, moderate at least 12/24 and 1/4 per cell, ordinary Support at least 10/12 and one per cell, delayed Support at least 3/4를 요구한다.

각 split은 both pre-contact phases, five of six both-phase cells, every-cell usable phase, both leading feet, entry span at least 250 ms와 unique physical signatures at least 80%를 충족해야 한다. Historical exact, cross-split exact/near, adaptive backfill 및 replacement는 모두 0이어야 한다. Gate 실패 시 model/Confirmation analysis 전에 중단한다.

## 8. Sealed Confirmation protocol

Full generation 직후 Confirmation은 `SEALED_FOR_MILD_RECALIBRATED_CONFIRMATION`이다. 모든 physical gates와 Discovery physical/diversity audit가 먼저 통과해야 하며, 그 뒤 정확히 한 interpretation label과 supporting metric hash를 동결해야 한다. 그때만 frozen V2를 Confirmation에 한 번 평가할 수 있다. Confirmation tuning, hypothesis replacement, training은 금지한다.

## 9. Frozen hashes

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

## 10. Boundary and next milestone

Historical Model V2 verdict는 `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`, Support branch는 `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`, whole-simulation status는 `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED` 그대로다. 이 design readiness는 Sand model generalization을 주장하지 않는다.

다음 단일 milestone은 `SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_GENERATION`이다. 자동 시작하지 않는다.
