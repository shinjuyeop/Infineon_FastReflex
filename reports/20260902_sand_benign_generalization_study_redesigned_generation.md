# Redesigned Sand Benign Generalization Study Generation

## 1. Purpose

동결된 redesign을 변경하지 않고 fresh 176-run corpus를 한 번 생성해 execution integrity와 model-blind physical viability/diversity만 판정했다. Model V1/V2/Terrain inference, 확률·80D·observability·margin 분석, training/HNM/normalizer/tuning은 수행하지 않았다.

## 2. Starting state

- Starting HEAD / `origin/main`: `b901192530fe4c53f8d16b9575e50d9f51652d41`, parity
- Starting tracked worktree: clean
- Previous verdict: `SAND_BENIGN_GENERALIZATION_STUDY_REDESIGN_READY`
- Historical final status: `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`

## 3. Historical evidence boundary

Consumed Generalization HOLDOUT guard는 `guard_after=1`, `scientific_open_count=1`로 유지했다. Payload read, inference, feature reconstruction, visualization은 모두 0이다. Failed first Sand study와 pilot 96건은 manifest/signature metadata와 frozen physical aggregate 이외에 재사용하지 않았다. Failed-study run ID reuse와 pilot run ID reuse는 각각 0이다.

## 4. Frozen redesign

Redesign config file SHA-256은 `40dfdcda42ebabe324ae243904b8fb8154f0701f803f0cabe051baae23d83a9c`, complete redesign SHA는 `d853645939002f2460c99fe865ee4cc39c25334a5cb51dc5f489ec36202c05d2`다. Component hash는 다음과 같이 generation 전후 동일했다.

| Component | SHA-256 |
|---|---|
| Parameter domain | `02429ef22abe2670b7f812ac4a0586b29512eb74a99bb48e6e26e5997f019316` |
| Scenario matrix | `51968c46a67de3a337b9858f93be8e25998fa8a15394760f16964fbb25a0a940` |
| Split plan | `80531f137065cdddb13633595c14de8c06ad1169165331d40dafec830b760d5d` |
| Physical-label contract | `030e85c6561b3fb9b757a2beb82e15f583264cbfe203b016b0070f82888bb39c` |
| Generation gates | `914ac15e564f8218bfc386e3cfbbdeac292a4f66a9c2df5611a6ff7dcf92ec34` |
| Diversity metrics | `f9fbfa697d33aae328848ee3ef3bfe420e63a3e1a8da54c35b975ffa47942e15` |
| Confirmation protocol | `daca735533c229911f6179e7265333f75a8260ef3b5f6a20c7edf31f9fbb1781` |

Generation execution config SHA-256 `5bbf800a3fc428c39a07839299f6a1c86696d9c230505609bfbda6567a8ceef9`와 canonical generator SHA-256 `f52dab56e8abda499f4b0cb462f9d3d4554522287762947ee2c877658ff9f756`을 run 1 전에 고정했다. Generation 시작 뒤 mutation은 0이다.

## 5. Calibration changes incorporated

- Primary phase: first censor-valid target contact의 정확히 20 ms 전 loaded-contact phase
- Support: established event 뒤 1,000 ms가 관측되면 later fall로 소급 무효화하지 않음
- Realization: metadata cohort가 아니라 frozen start/width/topology variants
- Ordinary Support: 별도 calibrated lateral-deformable domain
- Observation: 9 s, 1 kHz
- Severity: mild와 boundary-adjacent moderate만 사용; `SEVERE_DOMAIN_EXCLUDED_BY_PHYSICAL_CALIBRATION`
- Actual Slip/Support: intent mismatch여도 실제 outcome으로 보존하고 교체하지 않음

## 6. Dataset plan

Dataset ID는 `sand_benign_generalization_redesigned_study_20260902`다. 각 split은 broad mild 48, boundary moderate 24, ordinary Support 12, delayed Support 4로 총 88건이다. 전체는 176건이며 Severe는 0이다. Planned scenario signature 176개와 run ID 176개는 모두 unique하다.

## 7. Generation execution

Planned/attempted/completed는 `176/176/176`, Discovery/Confirmation은 `88/88`이다. Adaptive backfill, replacement, rerun, split move, parameter mutation은 모두 0이다. Generation은 1,167.775 s가 걸렸고 176 NPZ의 합은 83,341,626 bytes다. Dataset directory는 184 files, 85,261,563 bytes다.

## 8. Objective validity

| Split | Valid | Invalid | Strict Sand | Support | Slip | Dual |
|---|---:|---:|---:|---:|---:|---:|
| Discovery | 82 | 6 | 64 | 16 | 2 | 0 |
| Confirmation | 71 | 17 | 52 | 16 | 3 | 0 |
| Total | 153 | 23 | 116 | 32 | 5 | 0 |

Overall objective-valid gate `153 >= 132`는 PASS다. Support 32는 ordinary 24와 delayed 8이고, Sand에서는 Support가 없었다.

## 9. Sand physical outcomes

Broad mild 96건은 strict 80, invalid 16이다. Boundary moderate 48건은 strict 36, actual Slip 5, invalid 7이다. Slip 5건은 valid physical outcomes로 유지했고 benign으로 relabel하거나 재생성하지 않았다. Boundary moderate는 calibrated approximately .040 m domain일 뿐 historical severe 또는 `.0525–.070 m` near-hazard와 같다고 주장하지 않는다.

## 10. Mild benign yield

| Split | Source/speed | Planned | Valid | Strict mild | Support | Slip | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| Discovery | Concrete/.20 | 8 | 7 | 7 | 0 | 0 | 1 |
| Discovery | Concrete/.25 | 8 | 8 | 8 | 0 | 0 | 0 |
| Discovery | Concrete/.30 | 8 | 8 | 8 | 0 | 0 | 0 |
| Discovery | Marble/.20 | 8 | 7 | 7 | 0 | 0 | 1 |
| Discovery | Marble/.25 | 8 | 7 | 7 | 0 | 0 | 1 |
| Discovery | Marble/.30 | 8 | 8 | 8 | 0 | 0 | 0 |
| Confirmation | Concrete/.20 | 8 | 5 | 5 | 0 | 0 | 3 |
| Confirmation | Concrete/.25 | 8 | 5 | 5 | 0 | 0 | 3 |
| Confirmation | Concrete/.30 | 8 | 8 | 8 | 0 | 0 | 0 |
| Confirmation | Marble/.20 | 8 | 5 | 5 | 0 | 0 | 3 |
| Confirmation | Marble/.25 | 8 | 5 | 5 | 0 | 0 | 3 |
| Confirmation | Marble/.30 | 8 | 7 | 7 | 0 | 0 | 1 |

Discovery `45/48 >= 40`은 PASS, Confirmation `35/48 < 40`은 FAIL이다.

## 11. Moderate boundary yield

| Split | Source/speed | Planned | Valid | Strict moderate | Support | Slip | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|
| Discovery | Concrete/.20 | 4 | 4 | 4 | 0 | 0 | 0 |
| Discovery | Concrete/.25 | 4 | 1 | 1 | 0 | 0 | 3 |
| Discovery | Concrete/.30 | 4 | 4 | 4 | 0 | 0 | 0 |
| Discovery | Marble/.20 | 4 | 4 | 3 | 0 | 1 | 0 |
| Discovery | Marble/.25 | 4 | 4 | 4 | 0 | 0 | 0 |
| Discovery | Marble/.30 | 4 | 4 | 3 | 0 | 1 | 0 |
| Confirmation | Concrete/.20 | 4 | 3 | 3 | 0 | 0 | 1 |
| Confirmation | Concrete/.25 | 4 | 2 | 2 | 0 | 0 | 2 |
| Confirmation | Concrete/.30 | 4 | 3 | 3 | 0 | 0 | 1 |
| Confirmation | Marble/.20 | 4 | 4 | 4 | 0 | 0 | 0 |
| Confirmation | Marble/.25 | 4 | 4 | 4 | 0 | 0 | 0 |
| Confirmation | Marble/.30 | 4 | 4 | 1 | 0 | 3 | 0 |

Aggregate는 Discovery `19/24`, Confirmation `17/24`로 양쪽 `>=12` PASS이고 모든 cell이 `>=1/4` PASS다. Actual contamination은 Slip 2/3, Support 0/0이다.

## 12. Source/speed balance

| Split | Source | Speed | Planned Sand | Valid | Strict benign | Support | Slip | Invalid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Discovery | Concrete | .20 | 12 | 11 | 11 | 0 | 0 | 1 |
| Discovery | Concrete | .25 | 12 | 9 | 9 | 0 | 0 | 3 |
| Discovery | Concrete | .30 | 12 | 12 | 12 | 0 | 0 | 0 |
| Discovery | Marble | .20 | 12 | 11 | 10 | 0 | 1 | 1 |
| Discovery | Marble | .25 | 12 | 11 | 11 | 0 | 0 | 1 |
| Discovery | Marble | .30 | 12 | 12 | 11 | 0 | 1 | 0 |
| Confirmation | Concrete | .20 | 12 | 8 | 8 | 0 | 0 | 4 |
| Confirmation | Concrete | .25 | 12 | 7 | 7 | 0 | 0 | 5 |
| Confirmation | Concrete | .30 | 12 | 11 | 11 | 0 | 0 | 1 |
| Confirmation | Marble | .20 | 12 | 9 | 9 | 0 | 0 | 3 |
| Confirmation | Marble | .25 | 12 | 9 | 9 | 0 | 0 | 3 |
| Confirmation | Marble | .30 | 12 | 11 | 8 | 0 | 3 | 1 |

Discovery strict total `64/72`는 PASS다. Confirmation `52/72 < 54`와 Concrete/.25 `7/12 < 8`이 FAIL이며 나머지 11개 cell은 PASS다.

## 13. Physical phase diversity

Eligible population은 strict Sand다. Primary phase는 -20 ms exact loaded-contact이고 contact-sample phase는 descriptive only다.

| Split | LEFT single | RIGHT single | Other | Cells with both | Usable cells | Status |
|---|---:|---:|---:|---:|---:|---|
| Discovery | 20 | 41 | DOUBLE 3 | 5/6 | 6/6 | PASS |
| Confirmation | 15 | 37 | 0 | 5/6 | 6/6 | PASS |

두 split 모두 Concrete/.25에서 frozen predeclared single-phase exception을 사용했고 나머지 5개 cell은 양 principal phase를 포함했다. 양 leading foot과 global two-phase gate도 모두 PASS다. Strict Sand의 contact sample은 DOUBLE 114, LEFT single 2로, contact-sample phase가 primary가 될 수 없다는 calibration 결론을 재확인한다.

## 14. Topology/contact realization

| Split | Topology | Planned | Strict | Leading foot | -20 ms phase |
|---|---|---:|---:|---|---|
| Discovery | transition_left | 48 | 44 | LEFT 44 | RIGHT 41, DOUBLE 3 |
| Discovery | transition_right | 24 | 20 | RIGHT 20 | LEFT 20 |
| Confirmation | transition_left | 48 | 37 | LEFT 37 | RIGHT 37 |
| Confirmation | transition_right | 24 | 15 | RIGHT 15 | LEFT 15 |

Topology는 actual leading/approach relation을 바꿨지만 그 자체를 phase truth로 사용하지 않았다. First-contact loaded side는 strict Sand에서 BILATERAL 114, LEFT 2였다.

## 15. Entry-time diversity

Strict Sand first-contact span은 overall `1,220–2,114 ms`(894 ms), Discovery 892 ms, Confirmation 894 ms다. Source별 span은 Concrete 588 ms, Marble 894 ms; speed별은 .20 877 ms, .25 289 ms, .30 284 ms; topology별은 left 590 ms, right 615 ms다. Frozen split gate `>=250 ms`는 모두 PASS다.

## 16. Support-control integrity

| Split | Ordinary planned/qualified | Minimum | Per-cell | Delayed planned/qualified | Minimum |
|---|---:|---:|---:|---:|---:|
| Discovery | 12/12 | >=10 PASS | 2/2 each, PASS | 4/4 | >=3 PASS |
| Confirmation | 12/12 | >=10 PASS | 2/2 each, PASS | 4/4 | >=3 PASS |

Post-target fall 29건 중 9건은 Support가 확립된 뒤 1,000 ms 이상 관측되어 valid Support로 보존됐다(ordinary 7, delayed 2). 이는 later fall이 established Support를 소급 삭제하지 않는 corrected contract다.

## 17. Invalidity decomposition

Invalid 23건은 pre-target fall 3, insufficient post-target observation 20, 기타 0이다. Target 이후 fall은 29건이었으나 위 9개 fully observed Support는 valid이므로 invalid post-target/censor count는 20이다.

Failed first study의 invalid `121/176`에서 `23/176`으로 98건 감소했다. Pre-target fall은 32→3, old fall/censor ambiguity 85는 new insufficient-follow-up 20으로 감소했다. Strict Sand는 41→116, Support는 11→32로 증가했다. 이는 dataset 간 label contract 차이를 포함하는 descriptive comparison이며 model 성능 증거가 아니다.

## 18. Physical signatures

- Unique scenario signatures: 176/176; exact duplicates 0
- Valid physical signatures: 150/153 unique, ratio 98.04%; gate >=80% PASS
- Exact physical duplicates: 3
- Repository scaled-distance <=0.10 non-gating diagnostic: near pairs 18, cross-split 12
- Discovery/Confirmation exact scenario overlap: 0
- Discovery/Confirmation forbidden parameter-near overlap: 0
- Historical exact overlap: 0

Frozen historical contamination gate는 exact overlap만 정의한다. Cross-split 전용 near threshold를 historical data에 참고로 재사용하면 Model V2 metadata와 7개 near pairs가 나오며, 이는 숨기지 않고 `physical_audit.json`에 non-gating diagnostic으로 기록했다. 동결 설계에 없던 historical-near gate를 generation 후 추가하지 않았다.

## 19. Historical contamination audit

Unified, Model V2, Generalization, Ice semantics, failed first Sand study, pilot 1/2/3의 exact scenario overlaps와 run-ID reuse는 각 reference에서 모두 0이다. Failed-study split reuse 0, pilot run reuse 0이다. Consumed HOLDOUT은 allowed manifest/signature metadata만 사용했고 raw payload read는 0이다.

## 20. Generation gates

전체 70개 frozen check 중 67 PASS, 3 FAIL이다.

| Gate | Requirement | Actual | Status |
|---|---|---|---|
| `execution/attempted` | 176 | 176 | PASS |
| `execution/completed` | 176 | 176 | PASS |
| `execution/adaptive_backfill` | 0 | 0 | PASS |
| `execution/replacement` | 0 | 0 | PASS |
| `yield/overall_objective_valid` | >=132 | 153 | PASS |
| `yield/REDESIGNED_DISCOVERY/strict_sand` | >=54 | 64 | PASS |
| `yield/REDESIGNED_CONFIRMATION/strict_sand` | >=54 | 52 | **FAIL** |
| `yield/REDESIGNED_DISCOVERY/broad_mild` | >=40 | 45 | PASS |
| `yield/REDESIGNED_CONFIRMATION/broad_mild` | >=40 | 35 | **FAIL** |
| `yield/REDESIGNED_DISCOVERY/boundary_moderate` | >=12 | 19 | PASS |
| `yield/REDESIGNED_CONFIRMATION/boundary_moderate` | >=12 | 17 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.20/strict_sand` | >=8 | 11 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.25/strict_sand` | >=8 | 9 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.30/strict_sand` | >=8 | 12 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.20/strict_sand` | >=8 | 10 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.25/strict_sand` | >=8 | 11 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.30/strict_sand` | >=8 | 11 | PASS |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.20/strict_sand` | >=8 | 8 | PASS |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.25/strict_sand` | >=8 | 7 | **FAIL** |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.30/strict_sand` | >=8 | 11 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.20/strict_sand` | >=8 | 9 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.25/strict_sand` | >=8 | 9 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.30/strict_sand` | >=8 | 8 | PASS |
| 12 boundary-moderate cell checks | >=1 each | Discovery 1–4; Confirmation 1–4 | PASS |
| 12 ordinary-Support cell checks | >=1 each | 2 each | PASS |
| `yield/REDESIGNED_DISCOVERY/ordinary_support` | >=10 | 12 | PASS |
| `yield/REDESIGNED_CONFIRMATION/ordinary_support` | >=10 | 12 | PASS |
| `yield/REDESIGNED_DISCOVERY/delayed_support` | >=3 | 4 | PASS |
| `yield/REDESIGNED_CONFIRMATION/delayed_support` | >=3 | 4 | PASS |
| Two split principal-phase checks | >=2 | 2 each | PASS |
| Two split cells-with-both-phase checks | >=5 | 5 each | PASS |
| Two split every-cell-usable-phase checks | 6/6 | 6/6 each | PASS |
| Two split both-leading-feet checks | LEFT and RIGHT | both each | PASS |
| Two split entry-span checks | >=250 ms | 892 / 894 ms | PASS |
| `diversity/unique_physical_signature_fraction` | >=0.80 | 0.9804 | PASS |
| `integrity/severe_excluded` | 0 | 0 | PASS |
| `integrity/unique_run_ids` | 176 | 176 | PASS |
| `integrity/unique_scenario_signature_fraction` | 1.0 | 1.0 | PASS |
| `integrity/historical_exact_overlap` | 0 | 0 | PASS |
| `integrity/historical_run_id_reuse` | 0 | 0 | PASS |
| `integrity/cross_split_exact_overlap` | 0 | 0 | PASS |
| `integrity/cross_split_parameter_near_overlap` | 0 | 0 | PASS |
| `integrity/model_outputs` | 0 | 0 | PASS |

The authoritative row-by-row 70-check table is persisted in Gitignored `physical_audit.json`; the compact table above combines only sets of identical PASS checks and does not omit any check class.

### Complete frozen check ledger

| Check | Actual | Status |
|---|---|---|
| `execution/attempted` | 176 | PASS |
| `execution/completed` | 176 | PASS |
| `execution/adaptive_backfill` | 0 | PASS |
| `execution/replacement` | 0 | PASS |
| `yield/overall_objective_valid` | 153 | PASS |
| `yield/REDESIGNED_DISCOVERY/strict_sand` | 64 | PASS |
| `yield/REDESIGNED_DISCOVERY/broad_mild` | 45 | PASS |
| `yield/REDESIGNED_DISCOVERY/boundary_moderate` | 19 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.20/strict_sand` | 11 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.20/boundary_moderate` | 4 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.20/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.25/strict_sand` | 9 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.25/boundary_moderate` | 1 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.25/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.30/strict_sand` | 12 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.30/boundary_moderate` | 4 | PASS |
| `yield/REDESIGNED_DISCOVERY/concrete/0.30/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.20/strict_sand` | 10 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.20/boundary_moderate` | 3 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.20/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.25/strict_sand` | 11 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.25/boundary_moderate` | 4 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.25/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.30/strict_sand` | 11 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.30/boundary_moderate` | 3 | PASS |
| `yield/REDESIGNED_DISCOVERY/marble/0.30/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_DISCOVERY/ordinary_support` | 12 | PASS |
| `yield/REDESIGNED_DISCOVERY/delayed_support` | 4 | PASS |
| `diversity/REDESIGNED_DISCOVERY/principal_phases` | 2 | PASS |
| `diversity/REDESIGNED_DISCOVERY/cells_with_both_phases` | 5 | PASS |
| `diversity/REDESIGNED_DISCOVERY/every_cell_usable_phase` | 6 | PASS |
| `diversity/REDESIGNED_DISCOVERY/both_leading_feet` | LEFT 44, RIGHT 20 | PASS |
| `diversity/REDESIGNED_DISCOVERY/entry_time_span_ms` | 892 | PASS |
| `yield/REDESIGNED_CONFIRMATION/strict_sand` | 52 | **FAIL** |
| `yield/REDESIGNED_CONFIRMATION/broad_mild` | 35 | **FAIL** |
| `yield/REDESIGNED_CONFIRMATION/boundary_moderate` | 17 | PASS |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.20/strict_sand` | 8 | PASS |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.20/boundary_moderate` | 3 | PASS |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.20/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.25/strict_sand` | 7 | **FAIL** |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.25/boundary_moderate` | 2 | PASS |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.25/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.30/strict_sand` | 11 | PASS |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.30/boundary_moderate` | 3 | PASS |
| `yield/REDESIGNED_CONFIRMATION/concrete/0.30/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.20/strict_sand` | 9 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.20/boundary_moderate` | 4 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.20/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.25/strict_sand` | 9 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.25/boundary_moderate` | 4 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.25/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.30/strict_sand` | 8 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.30/boundary_moderate` | 1 | PASS |
| `yield/REDESIGNED_CONFIRMATION/marble/0.30/ordinary_support` | 2 | PASS |
| `yield/REDESIGNED_CONFIRMATION/ordinary_support` | 12 | PASS |
| `yield/REDESIGNED_CONFIRMATION/delayed_support` | 4 | PASS |
| `diversity/REDESIGNED_CONFIRMATION/principal_phases` | 2 | PASS |
| `diversity/REDESIGNED_CONFIRMATION/cells_with_both_phases` | 5 | PASS |
| `diversity/REDESIGNED_CONFIRMATION/every_cell_usable_phase` | 6 | PASS |
| `diversity/REDESIGNED_CONFIRMATION/both_leading_feet` | LEFT 37, RIGHT 15 | PASS |
| `diversity/REDESIGNED_CONFIRMATION/entry_time_span_ms` | 894 | PASS |
| `diversity/unique_physical_signature_fraction` | 0.9804 | PASS |
| `integrity/severe_excluded` | 0 | PASS |
| `integrity/unique_run_ids` | 176 | PASS |
| `integrity/unique_scenario_signature_fraction` | 1.0 | PASS |
| `integrity/historical_exact_overlap` | 0 | PASS |
| `integrity/historical_run_id_reuse` | 0 | PASS |
| `integrity/cross_split_exact_overlap` | 0 | PASS |
| `integrity/cross_split_parameter_near_overlap` | 0 | PASS |
| `integrity/model_outputs` | 0 | PASS |

## 21. Discovery readiness

Discovery physical generation alone is viable: strict `64/72`, mild `45/48`, moderate `19/24`, ordinary/delayed Support `12/12` and `4/4`, phase/diversity gates all pass. 그러나 full frozen generation은 Confirmation yield gate 실패 때문에 READY가 아니다. Model inference는 0이며 Discovery V2 analysis를 시작하지 않는다.

## 22. Confirmation sealing

Confirmation은 88건 생성되고 objective integrity/physical labels/gates만 확인했다. V1/V2/Terrain inference, 80D, observability, visualization, hypothesis selection은 모두 0이다. 상태는 `SEALED_FOR_REDESIGNED_CONFIRMATION`이고 loader는 NPZ access 전에 Confirmation 요청을 거부한다.

## 23. Dataset freeze

| Frozen object | SHA-256 |
|---|---|
| `REDESIGNED_STUDY_MANIFEST_SHA` | `90970438abb9eced3742d387697cf3f3ff4bd8b905b3554ea01f6766a38e501b` |
| `REDESIGNED_STUDY_DISCOVERY_SPLIT_SHA` | `2b4e04f5de74501cadc72f9ce0a0a216a3af6e935f853dd22fe785a36278eade` |
| `REDESIGNED_STUDY_CONFIRMATION_SPLIT_SHA` | `729a75ed4c6ae86c740b9d27c43c9ae1a5ae66fee8e48adef5e536ff95c3a494` |
| `REDESIGNED_STUDY_SCENARIO_SIGNATURE_SHA` | `56dd39cb6d05b4c1908f2babf8e5b40309db97f92e79af99e91256bb9f1fb1cc` |
| `REDESIGNED_STUDY_PHYSICAL_SIGNATURE_SHA` | `b210ee27ccfe5fbaa43323b6ed9ece9e62dcbd65905771d884e9f9fb64a0e7e3` |
| `REDESIGNED_STUDY_NPZ_AGGREGATE_SHA` | `82a5085272b72286430cf87d7f764d0dc232e9ac73890c3b086a7cb2cdb4388a` |
| `REDESIGNED_STUDY_PHYSICAL_OUTCOME_SHA` | `e8f434a12bafe904f9ff70c77d533b99227c5f134da48375ceee1956b3d23649` |
| `REDESIGNED_STUDY_GENERATION_GATE_RESULT_SHA` | `508952d804f1bf5c9114ac449bacbdc84376b0625b5c25fca51fdf7a13266ea6` |
| `REDESIGNED_STUDY_PHYSICAL_AUDIT_SHA` | `28c5dec8b6c4ed4bbb5056b50f712a11d784fcf63987b8e1cca24017d3cfa20f` |
| `REDESIGNED_STUDY_DATASET_FREEZE_SHA` | `87956c511684a78780d8bc7c1ac50552779de55b85a739e0221d1fa449f9416a` |
| Dataset-freeze file SHA-256 | `c1a51f89d35ff32c880b8572062e8a36ab677c5d929884e68b89d5928a938ddb` |
| Confirmation seal | `33a2a503ccb548b712c43c485175dd94e0787a7882a9ecb02ba8b7c3b2a76895` |

All 176 NPZ, manifest, audit, seal, aggregate, and semantic dataset-freeze hash checks pass without deserializing Confirmation payloads.

## 24. Historical status preservation

다음 scientific status는 generation 성공/실패와 무관하게 변경하지 않는다.

- `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`

새 corpus는 model-performance evidence가 아니다.

## 25. Limitations

- Confirmation broad mild가 `35/48`로 frozen minimum보다 5 부족하다.
- Confirmation Concrete/.25 strict Sand가 `7/12`로 1 부족하고 total strict가 `52/72`로 2 부족하다.
- Physical exact duplicates 3과 non-gating near pairs 18이 있지만 unique ratio gate는 통과한다.
- Historical-near criterion은 redesign에 freeze되지 않아 참고 진단으로만 보고했다.
- Deterministic simulator 결과만 다루며 hardware generalization을 주장하지 않는다.

## 26. Generation verdict

`SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION_YIELD_INSUFFICIENT`

Integrity와 diversity, overall valid, Discovery, boundary moderate, Support control은 통과했으나 frozen hierarchy에서 위 세 yield check 실패가 우선한다. H1/H2/H3 model-science conclusion은 내리지 않는다.

## 27. Recommended next milestone

`SAND_BENIGN_REDESIGNED_DOMAIN_FAILURE_REVIEW`

Confirmation을 model 분석에 사용하거나 대규모 pilot loop를 시작하지 않는다. 가장 작은 후속 범위는 Confirmation mild의 insufficient-follow-up과 Concrete/.25 yield를 현재 frozen physical metadata에서 검토하는 것이다.

## Counters

- Optimizer steps / checkpoint writes / normalizer fits / HNM: 0 / 0 / 0 / 0
- Threshold / persistence / architecture searches: 0 / 0 / 0
- New full-study simulation runs: 176
- V1 / V2 / Terrain inference: 0 / 0 / 0
- Old HOLDOUT payload reads / inference: 0 / 0
- Confirmation 80D / observability / hypothesis selection: 0 / 0 / 0
