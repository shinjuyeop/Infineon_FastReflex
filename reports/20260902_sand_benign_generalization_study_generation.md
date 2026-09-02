# Sand Benign Generalization Study Generation

## 1. Purpose

동결된 176-run Sand-benign generalization study를 정확히 한 번 생성하고, 모델과 무관한 실행·물리 label·yield·다양성·split sealing만 검증했다. Model V1/V2/Terrain replay, 확률, causal 80D, observability, 학습 및 H1/H2/H3 판단은 수행하지 않았다.

## 2. Starting state

- 시작 HEAD와 `origin/main`: `701c85267560c77c52da2dc7d26721f66040510d`, parity 확인
- 시작 tracked worktree: clean
- 시작 commit message: `Design Sand benign generalization study`
- historical final verdict: `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- previous design verdict: `SAND_BENIGN_GENERALIZATION_STUDY_DESIGN_READY`

## 3. Frozen design

Design file SHA-256은 `e45dcbe8130e5887c65ec7e9e3ef8c03744f8b589c81bc3ffe12536a0b145f70`, complete canonical design SHA는 `539f18b1e4c27abca826b7a2eac0d5c663e13035e63b1acb5fbab45136470a7f`다. Simulation 전 재계산한 component hash는 모두 일치했다.

| Object | SHA-256 |
|---|---|
| Parameter domain | `c447edb8e54721beb4d5997cce00ba0884f277153eacd30e78e96e05bd899d29` |
| Scenario matrix | `31925f5719317f42eff46bca0c2ae0c8f7a8d7d7247a0304792a27970b066e38` |
| Split plan | `7084b0f430f81cb9e676f08ae7d1995388e105f6a36b1d4019dfd8a0076eea56` |
| Physical-label contract | `557b813ec9440b615a70c6fb16fda00a84b84100b938702bd754438f4304702a` |
| Diversity metrics | `dc5aeddfe5e0f6ec4e24fc582aa6d9cc2ea394f2ec1cbd8f038f474166521cee` |
| Observability metrics | `5d946ac9cb908df903e24cb916182fb89cc4fc5633575264be51f1e532c10c04` |
| Decision rule | `5937bc4c728ca04c02fbf106383e7679358cb62d40ef6111c3333c784d770751` |

Generation config SHA-256 `d638e2c5f4da0365ac56537bfa4a26f592fc5f500a827f588a562c6d9e4f9268`을 첫 simulation 전에 기록했다. 실행 시작 후 design, matrix, label, gate, source를 변경하지 않았다.

## 4. Evidence boundary

이 milestone은 model-blind physical/diversity generation이다. 저장된 runtime arrays는 timestamp, raw Pelvis IMU6, FSR8, exact/loaded contact와 frozen Slip/Support/I1 diagnostics뿐이며 Hazard model output field는 없다. Confirmation에는 objective metadata와 frozen physical-signature audit만 사용했다.

## 5. Historical HOLDOUT protection

Consumed Generalization HOLDOUT guard는 `guard_after=1`, `scientific_open_count=1`, `second_scientific_open_forbidden=true`로 유지됐다. Old HOLDOUT payload read/inference/feature reconstruction/visualization은 모두 0이다. Collision 검사는 기존 manifest의 signature와 metadata만 읽었으며 NPZ waveform은 열지 않았다.

Unified/V2/Generalization/Ice-semantics manifest는 각각 256/412/72/48과 원 SHA를 유지했다. V1, baseline V2, extraction-rebalanced V2, final anchor-refined V2, Terrain V1 관련 18개 config/freeze/normalizer/checkpoint와 네 historical manifest, 총 22개 보호 artifact가 모두 generation 전후 일치했다.

## 6. Generation protocol

Canonical deterministic G1 simulator, frozen walking policy SHA `2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`, 0.5 ms physics step, 1 kHz observation, 8 s duration을 사용했다. 순서는 `split → group → template → source → speed`였고 각 planned run을 한 번 실행했다. Outcome에 따른 재실행, 교체, cell backfill 또는 parameter 변경은 없었다.

Strict benign은 target contact, no Slip/I1/Support, complete diagnostics, relevant fall/censor ambiguity 없음, 마지막 target observation 뒤 1,000 ms 이상을 모두 요구했다. I1-only는 보존하되 strict population에서 제외했다. Actual balanced displacement 구간은 LOW `[0,.030)`, MEDIUM `[.030,.0525)`, NEAR_HAZARD `[.0525,.070]` m로 고정했다.

## 7. Planned corpus

Sand 144건은 각 split 72건이며 source × speed 여섯 cell에 split당 12건씩 배치됐다. 전체 intent severity는 LOW/MEDIUM/NEAR_HAZARD 각 48, Sand topology는 transition-left/right 각 72다. Support control은 ordinary 24, delayed 8이다.

## 8. Execution result

| Group | Planned | Executed | Valid | Invalid |
|---|---:|---:|---:|---:|
| Sand broad | 96 | 96 | 34 | 62 |
| Sand near-hazard | 48 | 48 | 10 | 38 |
| Ordinary Support | 24 | 24 | 5 | 19 |
| Delayed Support | 8 | 8 | 6 | 2 |
| Discovery | 88 | 88 | 26 | 62 |
| Confirmation | 88 | 88 | 29 | 59 |
| Total | 176 | 176 | 55 | 121 |

총 generation 시간은 1,023.791 s였다. Dataset directory에는 176 NPZ와 seven manifest/audit/seal/freeze/summary files, 총 183 files 및 74,942,697 bytes가 있다. NPZ payload 합계는 72,970,898 bytes다.

Invalid 121건의 원인은 post-target fall/censor ambiguity 85, pre-target fall 32, insufficient post-target observation 4다. 모든 invalid record는 삭제하지 않고 원 split에 보존했다.

## 9. Dataset integrity

Planned/executed run ID와 scenario signature는 각각 176/176 unique다. Exact config duplicate, Discovery/Confirmation exact overlap, historical signature overlap은 모두 0이다. 176 NPZ의 개별 SHA, manifest SHA, dataset-freeze SHA를 다시 계산해 모두 일치함을 확인했다. Adaptive backfill과 replacement는 0이다.

## 10. Physical labels

| Split | Strict benign | I1-only | Support | Slip | Dual hazard | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| Discovery | 19 | 0 | 5 | 2 | 0 | 62 |
| Confirmation | 22 | 0 | 6 | 1 | 0 | 59 |
| Total | 41 | 0 | 11 | 3 | 0 | 121 |

이는 mutually exclusive objective outcome이다. Invalid 중에도 censor 전에 established Support 12건, Slip 11건, I1 16건이 관측됐지만 ambiguity 때문에 valid outcome numerator에는 포함하지 않았다.

## 11. Strict-benign yield

총 strict-benign gate는 Discovery `19/72 < 54/72`, Confirmation `22/72 < 54/72`로 모두 FAIL이다. 아래의 `L/M/N`은 actual severity별 `>=2`, `TL/TR`은 topology별 `>=4`, Strict는 `>=9/12` gate다.

| Split | Source/speed | Strict | L/M/N | TL/TR | Cell status |
|---|---|---:|---|---|---|
| Discovery | Concrete/.20 | 3 | 2/0/1 | 2/1 | FAIL |
| Discovery | Concrete/.25 | 2 | 1/1/0 | 2/0 | FAIL |
| Discovery | Concrete/.30 | 4 | 2/2/0 | 3/1 | FAIL |
| Discovery | Marble/.20 | 3 | 1/0/2 | 1/2 | FAIL |
| Discovery | Marble/.25 | 3 | 2/1/0 | 2/1 | FAIL |
| Discovery | Marble/.30 | 4 | 2/1/1 | 1/3 | FAIL |
| Confirmation | Concrete/.20 | 1 | 1/0/0 | 0/1 | FAIL |
| Confirmation | Concrete/.25 | 2 | 1/1/0 | 0/2 | FAIL |
| Confirmation | Concrete/.30 | 6 | 3/2/1 | 1/5 | FAIL |
| Confirmation | Marble/.20 | 1 | 1/0/0 | 0/1 | FAIL |
| Confirmation | Marble/.25 | 4 | 3/1/0 | 1/3 | FAIL |
| Confirmation | Marble/.30 | 8 | 3/2/3 | 3/5 | FAIL |

| Support gate | Required | Actual | Status |
|---|---:|---:|---|
| Discovery ordinary Support | >=9 | 2 | FAIL |
| Discovery delayed Support | >=3 | 3 | PASS |
| Confirmation ordinary Support | >=9 | 3 | FAIL |
| Confirmation delayed Support | >=3 | 3 | PASS |

Frozen yield checks 전체는 16 PASS / 62 FAIL이다. Outcome을 본 뒤 yield를 높이는 재생성은 하지 않았다.

## 12. Source-speed matrix

| Split | Source | Speed | Planned | Valid | Strict benign | I1-only | Support | Slip | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Discovery | Concrete | .20 | 12 | 3 | 3 | 0 | 0 | 0 | 9 |
| Discovery | Concrete | .25 | 12 | 2 | 2 | 0 | 0 | 0 | 10 |
| Discovery | Concrete | .30 | 12 | 5 | 4 | 0 | 0 | 1 | 7 |
| Discovery | Marble | .20 | 12 | 3 | 3 | 0 | 0 | 0 | 9 |
| Discovery | Marble | .25 | 12 | 3 | 3 | 0 | 0 | 0 | 9 |
| Discovery | Marble | .30 | 12 | 5 | 4 | 0 | 0 | 1 | 7 |
| Confirmation | Concrete | .20 | 12 | 1 | 1 | 0 | 0 | 0 | 11 |
| Confirmation | Concrete | .25 | 12 | 2 | 2 | 0 | 0 | 0 | 10 |
| Confirmation | Concrete | .30 | 12 | 7 | 6 | 0 | 0 | 1 | 5 |
| Confirmation | Marble | .20 | 12 | 1 | 1 | 0 | 0 | 0 | 11 |
| Confirmation | Marble | .25 | 12 | 4 | 4 | 0 | 0 | 0 | 8 |
| Confirmation | Marble | .30 | 12 | 8 | 8 | 0 | 0 | 0 | 4 |

Planned balance 12/cell/split은 정확하지만 physical valid/strict yield는 특히 .20/.25에서 크게 축소됐다.

## 13. Severity realization

| Split | Intent | Planned | Realized LOW | MEDIUM | NEAR_HAZARD | Hazard/other | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| Discovery | LOW | 24 | 10 | 0 | 0 | 0 | 14 |
| Discovery | MEDIUM | 24 | 0 | 5 | 0 | 1 | 18 |
| Discovery | NEAR_HAZARD | 24 | 0 | 0 | 4 | 1 | 19 |
| Confirmation | LOW | 24 | 12 | 0 | 0 | 0 | 12 |
| Confirmation | MEDIUM | 24 | 0 | 6 | 0 | 0 | 18 |
| Confirmation | NEAR_HAZARD | 24 | 0 | 0 | 4 | 1 | 19 |

Valid strict-benign에서는 intent와 realized stratum이 일치했지만 전체 planned 144건 중 strict-realized severity를 얻은 것은 LOW 22, MEDIUM 11, NEAR_HAZARD 8뿐이다. 세 valid Sand run은 Slip으로 actual outcome이 intent를 override했다.

## 14. Geometry realization

각 split에서 start EARLY/MID/LATE와 width NARROW/MEDIUM/WIDE는 모두 planned/executed 24/24였다.

| Split | Dimension/level | Valid | Strict | Entry timing range ms (span) |
|---|---|---:|---:|---|
| Discovery | Start EARLY/MID/LATE | 6/8/7 | 6/8/5 | 911–1842 (931) / 1220–1808 (588) / 1227–1810 (583) |
| Discovery | Width NARROW/MEDIUM/WIDE | 8/1/12 | 7/1/11 | 1220–1842 (622) / 1210–1210 (0) / 911–1808 (897) |
| Confirmation | Start EARLY/MID/LATE | 6/8/9 | 6/8/8 | 911–1509 (598) / 1220–1511 (291) / 1227–1511 (284) |
| Confirmation | Width NARROW/MEDIUM/WIDE | 3/10/10 | 2/10/10 | 1220–1227 (7) / 1227–1511 (284) / 911–1511 (600) |

Nominal geometry는 전체적으로 target-entry timing을 움직였지만 low yield로 일부 level, 특히 Discovery MEDIUM width가 1 valid run으로 붕괴했다. 따라서 geometry diversity를 study-ready라고 볼 수 없다.

## 15. Phase realization

Discovery Sand의 phase-slot 계획은 A/B/C/D 각 18이었다. 전체 72건의 assigned-to-realized crosstab은 A=`DOUBLE_SUPPORT 18`, B=`DOUBLE_SUPPORT 12 / NO_SUPPORT 6`, C=`DOUBLE_SUPPORT 11 / LEFT_SINGLE_SUPPORT 1 / NO_SUPPORT 6`, D=`DOUBLE_SUPPORT 18`이다. 그러나 physically valid Sand 21건은 모두 `DOUBLE_SUPPORT`였고 frozen global minimum three realized phases를 실패했다. `PHASE_DIVERSITY_COLLAPSE`가 관측됐다.

## 16. Topology/contact realization

| Planned topology (Discovery) | Planned | Actual leading foot | Loaded side at entry | Target-contact side |
|---|---:|---|---|---|
| transition-left | 36 | LEFT 32, NONE 4 | BILATERAL 31, LEFT 1, NONE 4 | BILATERAL 24, LEFT 8, NONE 4 |
| transition-right | 36 | RIGHT 28, NONE 8 | BILATERAL 28, NONE 8 | BILATERAL 15, RIGHT 13, NONE 8 |

Design topology는 leading side를 구분했지만 loaded state는 거의 bilateral이었고 valid source-speed cell마다 양 leading foot을 요구한 gate 다수가 실패했다. Design topology를 actual contact state로 대체하지 않았다.

## 17. Realization cohort diversity

Discovery valid Sand만 비교했다.

| Metric | Cohort 2026090201 (n=13) | Cohort 2026090202 (n=8) | Interpretation |
|---|---|---|---|
| Contact timing ms | 911 / 1227 / 1842 | 1220 / 1504 / 1511 | min/median/max differ |
| Contact duration ms | 555 / 2682 / 4147 | 1691 / 2976.5 / 4628 | distributions differ |
| Contact sequence count | 2 / 14 / 59 | 7 / 14 / 26 | spread differs |
| Balanced displacement m | .00657 / .02020 / .06523 | .02017 / .04014 / .06515 | all profiles represented unevenly |
| Normalized load transition | 1418 / 6672 / 7181 | 5946 / 6956 / 10306 | raw physical summaries differ |
| Pelvis accel RMS | 10.503 / 10.919 / 11.358 | 10.722 / 11.019 / 11.751 | raw summary spread differs |
| Pelvis gyro RMS | .343 / .491 / .736 | .334 / .560 / .685 | raw summary spread differs |
| Realized phase | DOUBLE_SUPPORT only | DOUBLE_SUPPORT only | phase diversity collapsed |
| Scaled physical distance | cross-cohort nearest median 4.9748 | cross-cohort nearest median 4.9748 | no exact collapse, no significance claim |

두 cohort는 ID만 다른 exact duplicate가 아니며 physical distances는 충분히 크다. 다만 valid phase가 한 category이고 cohort sample yield가 낮으므로 classification은 `REALIZATION_DIVERSITY_WEAK`이다.

## 18. Physical signature diversity

| Dimension | Planned coverage | Realized coverage | Status |
|---|---|---|---|
| Source | Concrete, Marble | 2/2 | PASS |
| Speed | .20, .25, .30 | 3/3 | PASS |
| Geometry entry | EARLY, MID, LATE | 3/3 executed; low valid yield | INSUFFICIENT |
| Geometry width | NARROW, MEDIUM, WIDE | 3/3 executed; low valid yield | INSUFFICIENT |
| Phase assignment | A/B/C/D | 4/4 | PASS |
| Realized contact phase | >=3 global, >=2/cell | 1 valid global | FAIL |
| Topology | left/right | 2/2 | PASS |
| Actual leading/loaded side | both feet per cell | many cells single/empty | FAIL |
| Actual severity | three strata per cell | three global, incomplete per cell | FAIL |
| Realization cohort | two distinct cohorts | two, but weak | INSUFFICIENT |

Valid physical signatures are 55/55 unique, so unique fraction 100% and exact duplicates 0이다. Frozen distance criterion은 near-duplicate 4건을 찾았고 모두 cross-split이었다. 그중 delayed Concrete/.25 pair는 distance `0.00000735`로 사실상 같은 physical realization이었다. Physical diversity checks는 21 PASS / 31 FAIL이므로 별도 diversity readiness도 미달이다.

Discovery strict-benign physical ranges는 balanced displacement `.00657–.06523 m`, contact duration `555–4628 ms`, contact sequence `2–59`, load-transition summary `1418–7761`, raw Pelvis accel RMS `10.503–11.751`, gyro RMS `.334–.723`이다. Strict-benign support spread는 definition상 모두 0이었고 normalized load redistribution summary는 모두 1.0으로 포화되어 이 metric의 변별력이 없었다.

## 19. Historical collision audit

| Signature check | Result |
|---|---:|
| Planned exact signatures | 176 |
| Executed unique scenario signatures | 176 |
| Exact scenario duplicates | 0 |
| Historical exact overlaps | 0 |
| Discovery/Confirmation exact overlap | 0 |
| Physical near duplicates | 4 |
| Cross-split physical near duplicates | 4 |

Historical comparison은 Unified, Model V2, Generalization, Ice semantics manifest signature만 사용했다. Consumed HOLDOUT waveform comparison은 0이다.

## 20. Support controls

| Type | Source | Planned | Actual valid Support | Other valid outcome | Invalid |
|---|---|---:|---:|---:|---:|
| Ordinary | Concrete | 12 | 2 | 0 | 10 |
| Ordinary | Marble | 12 | 3 | 0 | 9 |
| Delayed | Concrete | 4 | 3 | 0 | 1 |
| Delayed | Marble | 4 | 3 | 0 | 1 |

전체 control은 source Concrete/Marble 16/16, speed .20/.25/.30 = 8/16/8로 계획대로다. Established Support diagnostic은 23/32, I1은 27/32, valid Support는 11/32였다. Side diagnostic은 LEFT_ONLY 17, RIGHT_ONLY 6, NONE 9이고 fall/censor invalid는 20건이다. Ordinary control은 양 split의 >=9 gate를 실패했다. Delayed control은 split당 3/4가 valid했고 그 여섯 건 모두 I1 이전 clean target touchdown >=2와 left-only Support relation을 만족해 split gate를 통과했다.

## 21. Discovery readiness

Discovery는 생성·hash integrity 측면에서는 완전하지만 physical viability와 diversity가 부족하다. Strict benign은 19/72, ordinary Support는 2/12이며 valid Sand phase는 one category다. Frozen V2 replay는 수행하지 않았으므로 model/representation study로 진행할 수 없다.

## 22. Confirmation sealing

- generated: YES
- payload available on disk: YES
- objective physical integrity summary: YES
- V2 inference: NO
- normalized 80D analysis: NO
- observability analysis: NO
- visualization: NO
- scientific interpretation: NO
- status: `SEALED_FOR_STUDY_CONFIRMATION`

Study loader는 Confirmation payload 요청을 NPZ access 전에 거부하며 targeted test로 확인했다.

## 23. Dataset freeze

| Frozen object | SHA-256 |
|---|---|
| `STUDY_MANIFEST_SHA` | `70a619e35dba72478b298fb5082c68c07bc577e7536ba48b2ce9a52665f9834f` |
| `STUDY_DISCOVERY_SPLIT_SHA` | `42be5f3e29028c7e87eb0f7762e5a51d4aafc541fa0e37f531587f51da36e780` |
| `STUDY_CONFIRMATION_SPLIT_SHA` | `2a786fa88b1094908f427e320457907e846fd82410fc1aa650759058ae983893` |
| `STUDY_SCENARIO_SIGNATURE_SHA` | `06f98028e7cb4c019e128082a849748bcee24bcf4b3a6d82aed4de5cac25bf94` |
| `STUDY_PHYSICAL_SIGNATURE_SHA` | `4d289033ef6f74ec91bdef6e8037790f9e8d8831dc9b571df8abe55b5e3d6b56` |
| `STUDY_NPZ_AGGREGATE_SHA` | `86fc8bb9beac82f28c2e40069225eb8869ba4b4e8cff5fd713f2de711cefd27a` |
| `STUDY_PHYSICAL_OUTCOME_SHA` | `1565a522c6838f0694fa14823e8ebf65ac8f790f52d0bcc2942e5cf9dbd23d5b` |
| `STUDY_DIVERSITY_AUDIT_SHA` | `8a05e103fba3e0e0728e4541b15553aabc7dab0afc68a25c16e3d3e8bcfd28ec` |
| Confirmation seal | `845abbdd8c556076fcc2c42bd097e999d0fc0cdb26fbe84e20ad6f39111eef4a` |
| `SAND_BENIGN_GENERALIZATION_STUDY_DATASET_FREEZE_SHA` | `6ae5b6ac668f320eac811ebb430eaaecbf24ce64ee541e73db35c0ac862c28f0` |

Dataset ID는 `sand_benign_generalization_study_20260902`, gitignored path는 `data/raw/sand_benign_generalization_study_20260902`다.

## 24. Limitations

- Deterministic cohorts는 exact collapse하지 않았지만 valid phase가 모두 DOUBLE_SUPPORT라 `REALIZATION_DIVERSITY_WEAK`이다.
- Assigned four phase slots는 valid physical phase diversity로 번역되지 않았다.
- Intent severity는 valid strict cases에서 제대로 구현됐지만 invalid가 많아 각 source-speed cell의 three-strata gate가 무너졌다.
- Strict-benign yield가 두 split 모두 최소치보다 크게 낮다.
- Ordinary Support control은 5/24만 valid Support로 남아 신뢰할 만한 contrast를 만들지 못했다.
- 높은 fall/censor와 pre-target fall 비율은 현 geometry/speed matrix의 simulator viability limitation이다.
- 네 cross-split physical near duplicate가 있고, raw load-redistribution summary가 포화됐다.

No-backfill statement: physical outcome 때문에 재생성한 run은 없고, low-yield cell을 backfill하지 않았으며, simulation 시작 뒤 parameter를 바꾸지 않았다. Model result는 generation에 관여하지 않았고 consumed HOLDOUT example을 복제하지 않았다.

Counters: optimizer steps 0, checkpoint writes 0, normalizer fits 0, HNM rounds 0, threshold searches 0, persistence searches 0, architecture searches 0, seed searches 0, new simulation runs 176, model inference runs 0, old HOLDOUT payload reads/inference 0, Confirmation V2 inference 0.

## 25. Generation verdict

`SAND_BENIGN_GENERALIZATION_STUDY_GENERATION_PHYSICAL_YIELD_INSUFFICIENT`

Integrity는 PASS지만 strict-benign total/cell/severity/topology와 ordinary Support gates가 물질적으로 실패했다. Diversity도 부족하지만 frozen hierarchy에서 physical-yield failure가 우선한다. H1/H2/H3 결론은 내리지 않는다.

## 26. Recommended next milestone

`SAND_BENIGN_GENERALIZATION_STUDY_CALIBRATION_REVIEW`

생성된 결과를 adaptive rescue하거나 Discovery model replay를 시작하지 않는다. 별도 model-blind calibration review에서 pre-target/relevant fall과 contact-phase collapse의 최소 원인을 검토한 뒤 새로운 design 여부를 결정해야 한다. Confirmation은 계속 sealed다.
