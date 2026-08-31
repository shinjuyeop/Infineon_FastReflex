# Scenario Coverage Matrix Design

Date: 2026-08-31 (Asia/Seoul)
Milestone: `SCENARIO_COVERAGE_MATRIX_DESIGN`
Verdict: `SCENARIO_COVERAGE_DESIGN_READY`

## 1. Purpose

이번 milestone은 run 수를 먼저 늘리지 않고 current Unified Hazard corpus의 물리적 scenario coverage를 run 단위로 감사하고, 다음 generation이 채워야 할 causal/generalization gap을 사전에 고정한다. 모델 성능을 다시 평가하거나 simulation을 생성하는 단계가 아니다. 원칙은 `coverage first, count second`다.

작업 시작 시 `HEAD = origin/main = 0082f60e17d545896a1b55cac9c458557604fb86`, branch `main`, tracked worktree clean이었다. 집계 기준은 결과를 본 뒤 바꾸지 않도록 [`../configs/experiment/20260831_scenario_coverage_matrix_design.yaml`](../configs/experiment/20260831_scenario_coverage_matrix_design.yaml)에 먼저 기록했다.

## 2. Protected baseline

Dataset은 `unified_hazard_reflex_20260829`, manifest SHA-256은 `d023384aaaac22076a2be7c2fa242acac91a548d82102640c39868990cb0e9d6`다.

Hazard contract는 변경하지 않았다.

```text
Pelvis IMU6 -> causal 80D -> [20,80] -> GRU hidden32
            -> three-seed mean -> threshold 0.99
            -> persistence 5 ms -> REFLEX_REQUIRED
```

Hazard freeze SHA-256은 `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`다. Feature schema SHA-256은 `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`다.

Terrain contract도 변경하지 않았다.

```text
touchdown FSR4 -> [50,4] -> three-seed MLP
               -> CONCRETE / MARBLE / ICE / SAND
               -> advisory held state
```

Terrain은 Hazard를 gate하거나 지연하지 않는다. LEFT_ONLY producer의 normalizer와 세 checkpoint hash도 모두 protected verification을 통과했다.

## 3. Current dataset design

Frozen specification과 manifest는 256/256 run에서 ID, split, source/target, speed, mechanics 및 9-field physical signature가 정확히 일치했다. Split은 simulation 전에 고정됐고 outcome-driven 이동은 없었다.

| Designed group | TRAIN | VALIDATION | HOLDOUT | Total | Manifest physical label |
|---|---:|---:|---:|---:|---|
| `ICE_SLIP_HAZARD` | 38 | 13 | 13 | 64 | Slip 64 |
| `SAND_SUPPORT_HAZARD` | 38 | 13 | 13 | 64 | Support 64 |
| `SAND_BENIGN` | 38 | 13 | 13 | 64 | No Hazard 64 |
| `HARD_GROUND_NORMAL` | 38 | 13 | 13 | 64 | No Hazard 64 |
| **Total** | **152** | **52** | **52** | **256** | Slip 64 / Support 64 / No Hazard 128 |

각 group은 Concrete source 32, Marble source 32다. Manifest의 invalid run은 0이다.

## 4. Coverage methodology

Primary unit은 window가 아니라 independent run이다. 모든 `run_id`와 physical signature는 unique지만, 이 사실을 256개의 독립 scenario family로 해석하지 않는다. Coverage는 두 층으로 분리한다.

1. Scenario-family coverage: hazard type, source/target, temporal order, side, mechanics처럼 의미가 다른 물리 family.
2. Within-family variation: 같은 family 안의 patch start/width 또는 hard-control speed sweep.

Run-count 판정은 사전 고정한 다음 규칙을 쓴다.

| Category | Independent runs |
|---|---:|
| `MISSING` | 0 |
| `SPARSE` | 1–4 |
| `LIMITED` | 5–11 |
| `COVERED` | ≥12 |

이 category는 performance나 물리적 품질이 아니라 표본 존재량만 뜻한다. 한 값에 64개 run이 몰린 speed axis처럼 구조적 다양성이 없는 경우에는 run count와 별도로 명시한다.

전체 256-run count, physical side, continuous severity, physical/Terrain clocks는 frozen specification과 manifest metadata만 사용했다. Gait phase와 loaded-contact episode는 TRAIN/VALIDATION 204개의 `loaded_contact`만 read-only로 확인했다. Pelvis IMU6, FSR8, Hazard replay, Terrain replay와 label 재계산은 수행하지 않았다.

**HOLDOUT waveform reopened: NO**

HOLDOUT에는 specification, manifest metadata, 기존 committed report fact, file existence/size/SHA와 split membership만 사용했다. HOLDOUT waveform load, physical trace recomputation, Hazard/Terrain replay 및 새 scientific evaluation은 모두 0이다.

Temporal order의 established-Hazard 기준은 `hazard sample - first valid target Terrain sample`이다. `[-50,+50] ms`는 `NEAR_SIMULTANEOUS`, 그보다 작으면 Hazard-first, 크면 Terrain-first다. No-hazard와 Terrain-unavailable은 별도 class다.

Loaded-contact episode는 각 foot의 hysteretic `false -> true`부터 `true -> false`까지다. Target first contact 이후 시작하고 established Hazard 전에 끝난 episode를 두 foot에 걸쳐 센다. 이 값은 gait cycle이나 exact target-terrain touchdown과 동일하다고 주장하지 않는다.

Gait phase는 established Hazard sample에서 10 ms lookback을 사용하고 `TOUCHDOWN_LOADING > CONTACT_RELEASE > DOUBLE_SUPPORT > LEFT_SINGLE_SUPPORT > RIGHT_SINGLE_SUPPORT > NO_SUPPORT` priority로 배타적으로 분류했다.

## 5. Current scenario family taxonomy

| Family | Purpose | Source → target | Hazard/order | Physical side | Speed | Within-family variation | Known limitation |
|---|---|---|---|---|---|---|---|
| F1 Hard normal | Hard-ground no-reflex control | Concrete→Concrete; Marble→Marble | No Hazard | None | 0.1573–0.3650 | 32-speed sweep/source | no transition, fixed geometry |
| F2 Ice Slip | Low-friction transition Hazard | Concrete/Marble→Ice | 48 Hazard-first, 16 Terrain-unavailable | 56 bilateral, 8 left-only | 0.25 only | start 0.33–0.36; small width sweep | no Ice benign, no Terrain-first Slip, no right-only Slip |
| F3 Sand Support | Heterogeneous support Hazard | Concrete/Marble→Sand | established: 48 Terrain-first, 16 near; precursor: 64 Hazard-first | left-only 64 | 0.25 only | start 0.30/0.35; width sweep | I1 always contact+19 ms; one side/mechanics/severity |
| F4 Sand benign | Soft/deformable no-reflex control | Concrete/Marble→Sand | No Hazard | None; designed lane left/right 32/32 | 0.25 only | start, side and small width sweep | not speed-stratified; not matched to delayed-onset Support |

Top-level physical families are therefore four, not 256. Including source and discrete mechanics gives ten subfamilies; the remaining signature uniqueness is largely continuous within-family variation.

## 6. Coverage by hazard and terrain

### Hazard/benign coverage

| Axis | Category | Current runs | Judgment |
|---|---|---:|---|
| Hazard | Slip | 64 | `COVERED` |
| Hazard | Support | 64 | `COVERED` |
| Hazard | No Hazard | 128 | `COVERED` |
| Benign | Hard benign | 64 | `COVERED` |
| Benign | Sand benign | 64 | `COVERED` |
| Benign | Ice benign | 0 | `MISSING` |

### Target-terrain × physical-label matrix

| Target | Slip | Support | No Hazard | Total |
|---|---:|---:|---:|---:|
| Concrete | 0 | 0 | 32 | 32 |
| Marble | 0 | 0 | 32 | 32 |
| Ice | 64 | 0 | 0 | 64 |
| Sand | 0 | 64 | 64 | 128 |

Ice와 Sand는 모두 target class로 충분한 run 수가 있지만, class-conditional benign balance는 비대칭이다. Sand는 Hazard/benign 64/64이고 Ice는 Slip/benign 64/0이다.

### Source-target combinations

| Combination | Hazard | No Hazard | Total | Judgment |
|---|---:|---:|---:|---|
| Concrete→Ice | Slip 32 | 0 | 32 | Hazard `COVERED`; benign `MISSING` |
| Marble→Ice | Slip 32 | 0 | 32 | Hazard `COVERED`; benign `MISSING` |
| Concrete→Sand | Support 32 | 32 | 64 | both `COVERED` |
| Marble→Sand | Support 32 | 32 | 64 | both `COVERED` |
| Concrete hard | 0 | 32 | 32 | benign `COVERED` |
| Marble hard | 0 | 32 | 32 | benign `COVERED` |

Source balance 자체는 Concrete 128 / Marble 128로 대칭이다.

## 7. Temporal coverage

### Terrain output versus established Hazard

| Temporal class | All Hazard runs | Slip | Support | Judgment |
|---|---:|---:|---:|---|
| `HAZARD_BEFORE_TARGET_TERRAIN_VALID` | 48 | 48 | 0 | `COVERED` |
| `TARGET_TERRAIN_BEFORE_HAZARD` | 48 | 0 | 48 | `COVERED` overall; Slip `MISSING` |
| `NEAR_SIMULTANEOUS` | 16 | 0 | 16 | `COVERED` |
| `TERRAIN_UNAVAILABLE` | 16 | 16 | 0 | `COVERED` as an observed limitation |

Available Ice runs 48개에서 `Slip - Terrain valid`은 median -197.5 ms, range -432 to -141 ms였다. 즉 valid ICE가 먼저 나온 Slip은 0이다. 나머지 Ice 16개는 first valid target Terrain output이 없었다.

Support established clock 기준으로는 `Support - Terrain valid` median 618 ms, range 23–1182 ms다. 따라서 established event만 보면 Terrain-first coverage가 존재한다. 그러나 I1 precursor는 64/64에서 target contact +19 ms에 시작했고, Terrain valid보다 624 또는 1212 ms 먼저였다. I1-to-established delay는 1235–1806 ms다. 결론은 다음처럼 분리해야 한다.

- Terrain-first established Support: 48, `COVERED`.
- Terrain-first Support precursor/onset: 0, `MISSING`.

### Target contact to established Hazard

| Frozen delay bin | Slip | Support | Total | Judgment |
|---|---:|---:|---:|---|
| `<0 ms` | 0 | 0 | 0 | `MISSING` |
| `0–<100 ms` | 0 | 0 | 0 | `MISSING` |
| `100–<300 ms` | 0 | 0 | 0 | `MISSING` |
| `300–<600 ms` | 0 | 0 | 0 | `MISSING` |
| `600–<1000 ms` | 64 | 0 | 64 | `COVERED` |
| `≥1000 ms` | 0 | 64 | 64 | `COVERED` |

Slip delay는 min/median/max 688/692/969 ms이고 6개 exact outcome timing만 존재한다. Support established delay는 1254/1261/1825 ms이고 12개 exact timing만 존재한다. 따라서 “established Hazard가 접촉 직후 즉시 발생한다”는 가설은 반박되지만, 두 narrow timing modes에 집중된다는 가설은 지지된다. Support physical precursor가 19 ms 한 값에 고정된 점은 별도의 immediate-onset 편향이다.

### Loaded-contact delay

TRAIN/VALIDATION Hazard 102개만 계산했다.

| Completed post-target loaded-contact episodes | Slip | Support | Total | Judgment |
|---|---:|---:|---:|---|
| 0 (`IMMEDIATE`) | 0 | 0 | 0 | `MISSING` |
| 1 | 0 | 0 | 0 | `MISSING` |
| ≥2 | 51 | 51 | 102 | `COVERED` |

Raw loaded-contact hysteresis는 contact chatter를 gait step으로 과대계수할 수 있으므로 이 결과는 “둘 이상의 완료 episode가 있었다”는 범주 증거로만 사용한다. HOLDOUT 26개 Hazard phase/episode 값은 계산하지 않았다.

## 8. Foot and side coverage

| Physical affected side | Slip | Support | No Hazard | Total | Judgment |
|---|---:|---:|---:|---:|---|
| `LEFT_ONLY` | 8 | 64 | 0 | 72 | `COVERED` |
| `RIGHT_ONLY` | 0 | 0 | 0 | 0 | `MISSING` |
| `BILATERAL` | 56 | 0 | 0 | 56 | `COVERED` |
| `NONE` | 0 | 0 | 128 | 128 | `COVERED` |

Designed side와 actual side는 분리했다. Ice patch는 64개 모두 bilateral design이지만 physical Slip은 bilateral 56, left-only 8이었다. Sand Support는 `transition_left + lateral_deformable`로 설계됐고 actual Support도 left-only 64였다. Sand benign은 designed left/right lane 32/32지만 physical affected side는 모두 `NONE`이다. 실제 right-only Hazard는 Slip/Support 모두 0이다.

이 gap은 LEFT_ONLY Terrain producer의 update limitation과 다르다. Hazard side gap은 physical event mechanics의 gap이고 Terrain gap은 sensor-placement/update availability 문제다. Current corpus에서는 Ice Hazard 16/64가 target Terrain unavailable이지만 Sand benign과 Sand Support는 모두 target update metadata가 있다. 그러나 designed `transition_right` Sand-benign 32개에서 나중에 SAND-valued output이 있었다는 사실은 right foot 관측 증거가 아니다. Producer tensor는 여전히 left FSR4뿐이고, 기존 Terrain report는 right-lane-only target contact 18/144에서 update unavailable을 이미 기록했다. Physical right-only Hazard gap과 LEFT_ONLY sensor-placement gap을 하나의 failure로 합치지 않는다.

## 9. Gait-phase coverage

TRAIN/VALIDATION Hazard 102개에 대한 privileged audit 결과다.

| Exclusive phase | Slip | Support | Total | Judgment |
|---|---:|---:|---:|---|
| `TOUCHDOWN_LOADING` | 0 | 26 | 26 | `COVERED` |
| `CONTACT_RELEASE` | 0 | 0 | 0 | `MISSING` |
| `DOUBLE_SUPPORT` | 0 | 0 | 0 | `MISSING` |
| `LEFT_SINGLE_SUPPORT` | 33 | 25 | 58 | `COVERED` |
| `RIGHT_SINGLE_SUPPORT` | 18 | 0 | 18 | `COVERED` |
| `NO_SUPPORT` | 0 | 0 | 0 | `MISSING`; not automatically desirable physics |

Slip은 left/right single support 두 mode, Support는 touchdown/left single support 두 mode를 가진다. 따라서 gait phase가 오직 하나뿐이라는 가설은 반박되지만 전체 phase axis는 불완전하다. 특히 Support의 right-side/right-single coverage는 없다. HOLDOUT phase는 policy상 `NOT_COMPUTED`다.

## 10. Speed coverage

| Group | Unique raw values | Min | Median | Max |
|---|---:|---:|---:|---:|
| Ice Slip | 1 (`0.25`) | 0.25 | 0.25 | 0.25 |
| Sand Support | 1 (`0.25`) | 0.25 | 0.25 | 0.25 |
| Sand benign | 1 (`0.25`) | 0.25 | 0.25 | 0.25 |
| Hard normal | 32 | 0.1573 | 0.26115 | 0.3650 |
| All runs | 33 | 0.1573 | 0.25 | 0.3650 |

Hard-normal raw unique values are:

```text
0.1573, 0.1640, 0.1707, 0.1774, 0.1841, 0.1908, 0.1975, 0.2042,
0.2109, 0.2176, 0.2243, 0.2310, 0.2377, 0.2444, 0.2511, 0.2578,
0.2645, 0.2712, 0.2779, 0.2846, 0.2913, 0.2980, 0.3047, 0.3114,
0.3181, 0.3248, 0.3315, 0.3382, 0.3449, 0.3516, 0.3583, 0.3650
```

전체 min/max가 넓어 보여도 Hazard와 transition-benign family는 사실상 한 command speed에 고정돼 있다. 따라서 speed diversity 가설은 Hazard/Sand transition에 대해 확인된다.

## 11. Geometry and mechanics coverage

| Group/source | Patch start values (m) | Width range (m) | Unique widths / step |
|---|---|---|---|
| Ice Slip / Concrete | 0.33, 0.34, 0.35, 0.36 | 0.70155–0.74805 | 32 / 0.00150 |
| Ice Slip / Marble | 0.33, 0.34, 0.35, 0.36 | 0.70187–0.74837 | 32 / 0.00150 |
| Sand Support / Concrete | 0.30, 0.35 | 0.65023–0.74023 | 16 / 0.00600 |
| Sand Support / Marble | 0.30, 0.35 | 0.65041–0.74041 | 16 / 0.00600 |
| Sand benign / Concrete | 0.30, 0.35 | 0.72031–0.74776 | 16 / 0.00183 |
| Sand benign / Marble | 0.30, 0.35 | 0.72041–0.74741 | 16 / 0.00180 |
| Hard normal / both | 0.35 | 0.75 | 1 / fixed |

Discrete mechanics는 다섯 조합뿐이다.

| Designed group | Slip pattern | Sink pattern | Severity | Support pattern | Runs |
|---|---|---|---|---|---:|
| Hard normal | uniform | uniform | moderate | balanced_soft | 64 |
| Ice Slip | transition | uniform | moderate | balanced_soft | 64 |
| Sand Support | uniform | transition_left | moderate | lateral_deformable | 64 |
| Sand benign-left | uniform | transition_left | mild | balanced_deformable | 32 |
| Sand benign-right | uniform | transition_right | mild | balanced_deformable | 32 |

Support Hazard의 side/mechanics가 `transition_left + moderate + lateral_deformable` 한 조합에 집중됐다는 가설은 확인된다. Ice Slip도 bilateral transition 한 mechanics뿐이다.

## 12. Benign coverage

Hard benign 64와 Sand benign 64는 모두 manifest physical `NO_HAZARD`다. Sand benign은 mild balanced-deformable motion을 포함한다. Maximum deformation은 0.020139–0.020206 m지만 peak Support spread는 64/64에서 0이고, peak drift는 0.01072–0.02316 m로 Slip threshold 0.05 m 아래다.

TRAIN/VALIDATION Sand benign 51개는 target contact 이후 censor까지 모두 다수의 completed loaded-contact episode를 유지했다. 이 raw count는 chatter 때문에 step 수로 해석하지 않지만, current Sand benign이 단순 순간 접촉 control은 아니라는 점은 분명하다. 따라서 `DELAYED_BENIGN_SAND`를 독립 standalone family로 추가하면 중복이다. 다만 delayed Support와 동일한 topology/timing을 가진 matched control은 그 Hazard family 내부의 필수 counterpart로 남긴다.

Ice benign은 0이다. Current data만으로는 `ICE` advisory state 자체가 Hazard trigger가 아님을 terrain-class-specific hard negative로 검증할 수 없다.

## 13. Hazard severity and pseudo-diversity audit

Severity는 continuous distribution만 보고하며 weak/medium/strong class를 post-hoc으로 만들지 않는다.

| Observable | N | Min | Median | P95 | Max |
|---|---:|---:|---:|---:|---:|
| Slip peak drift (m) | 64 | 0.05737 | 0.14823 | 0.20546 | 0.21597 |
| Slip peak tangential velocity (m/s) | 64 | 2.0841 | 2.7144 | 6.0445 | 6.0445 |
| Support peak spread (m) | 64 | 0.040086 | 0.040104 | 0.040172 | 0.040172 |
| Support max deformation (m) | 64 | 0.040086 | 0.040104 | 0.040172 | 0.040172 |

Fall/censor metadata는 Slip 36/64, Support 57/64에 존재했다. 이는 severity diagnostic일 뿐 Hazard label이 아니다.

Slip outcome severity에는 실제 연속 범위가 있지만 speed/material/mechanics는 고정돼 있다. Support spread는 약 0.086 mm 범위에 몰려 사실상 한 severity operating point다. Exact physical signature 256개와 duplicate 0은 integrity 증거이지 256개 scenario-family 증거가 아니다. 예를 들어 Ice width의 1.5 mm increment 32개는 same-family robustness sweep이고, Sand benign의 약 1.8 mm increment도 새 physics family가 아니다.

## 14. Coverage gaps

| Gap | Current evidence | Severity | Why it matters | Proposed action |
|---|---|---|---|---|
| Ice benign | 0/64 Ice runs are benign | `P0 / MISSING` | ICE identity 자체와 reflex-worthy Slip을 분리할 hard negative가 없음 | Frozen Ice profile과 predeclared geometry/speed cells로 `ICE_BENIGN_CONTROL`; 실패 시 별도 physics calibration으로 격리 |
| Terrain-first Slip | Available Ice 48/48 are Hazard-first; 16 unavailable | `P0 / MISSING` | Terrain-known 상태 이후 delayed Slip generalization을 검증하지 못함 | `DELAYED_ICE_SLIP`, valid ICE 후 clean contact interval을 design intent로 고정 |
| Terrain-first Support onset | Established clock은 48 Terrain-first이나 I1은 64/64 contact+19 ms, Terrain보다 먼저 | `P0 / MISSING` at precursor onset | 현재 model의 long benign-Sand-to-later-support causal transition을 검증하지 못함 | benign Sand segment 후 delayed heterogeneous support; new topology면 calibration 별도 수행 |
| Right-only Hazard | physical right-only Slip 0, Support 0 | `P1 / MISSING` | side generalization과 pelvis-IMU symmetry를 검증하지 못함 | right-dominant Slip timing과 transition_right Support를 predeclare |
| Speed diversity | Hazard와 Sand benign 모두 0.25 한 값 | `P1 / structural gap` | frozen Hazard가 speed-correlated gait 변화를 일반화하는지 불명 | stable hard-control envelope 안의 0.20/0.25/0.30 representative strata |
| Gait-phase diversity | T/V에서 3/6 phase observed; Support right-single 0 | `P1 / partial` | onset timing이 일부 phase에 집중 | patch start/timing block을 phase target별 predeclare; actual phase는 oracle audit |
| Support mechanics/severity | left lateral moderate 한 Hazard mechanics; spread narrow | `P2 / structural gap` | topology/severity robustness가 불명 | P0/P1 뒤 별도 calibrated topology study; post-hoc severity labels 금지 |
| Standalone delayed Sand benign | Sand benign 64, T/V 51 all sustained no-Hazard post-contact | `DROP / duplicate` | 새 family로 세면 pseudo-diversity | delayed Support의 matched control로만 포함 |

사전 hypothesis 중 “immediate established Hazard bias”와 “single gait phase only”는 그대로 확인되지 않았다. 반면 Ice-benign, Terrain-first Slip, right-only Hazard, Hazard-speed, Support side/mechanics gap은 확인됐다. Support는 established clock과 precursor onset을 구분해야만 올바른 결론이 나온다.

## 15. Recommended scenario families

Priority 의미는 config에 고정했다: P0는 확인된 high-value gap을 닫기 위해 다음 dataset에 필수, P1은 calibrated capacity가 허용하면 포함, P2는 후속 robustness, DROP은 중복/저가치/부당한 tuning이 필요한 항목이다. `REQUIRES_NEW_PHYSICS_CALIBRATION` 표시는 다음 dataset 자동 포함이 아니며 별도 freeze/pass가 선행돼야 한다.

### Scenario family: `ICE_BENIGN_CONTROL`

- Purpose: ICE Terrain이 곧 Hazard가 아님을 검증하는 hard negative.
- Current gap: Ice benign 0.
- Source terrain: Concrete / Marble.
- Target terrain: Ice.
- Hazard intent: no established Slip, no Support, no I1.
- Temporal intent: valid ICE output 후에도 benign contact 유지.
- Affected side intent: `NONE`.
- Speed domain: 0.25 core; speed-stratified family와 결합할 때만 0.20/0.30.
- Physics/mechanics: frozen Ice material; predeclared patch start/width and gait timing only.
- Negative/control counterpart: matched `DELAYED_ICE_SLIP` cells.
- Expected value: terrain-class-specific no-reflex specificity.
- Priority: `P0`.

No-Slip은 design intent일 뿐 label이 아니다. Existing frozen variables 안에서 predeclared candidate cells가 모두 Slip을 만들면 결과에 맞춰 friction을 조정하지 않는다. 그 경우 `REQUIRES_NEW_PHYSICS_CALIBRATION`으로 분리하고 dataset generation을 중단한다.

### Scenario family: `DELAYED_ICE_SLIP`

- Purpose: valid ICE가 먼저 존재한 뒤 발생하는 Slip에 대한 zero-retrain generalization.
- Current gap: Terrain-first Slip 0; Terrain unavailable 16.
- Source terrain: Concrete / Marble.
- Target terrain: Ice.
- Hazard intent: one or more benign target-contact intervals 뒤 established Slip.
- Temporal intent: `TARGET_TERRAIN_BEFORE_HAZARD`, margin >50 ms.
- Affected side intent: bilateral reference; actual oracle decides.
- Speed domain: 0.25 core.
- Physics/mechanics: frozen Ice profile; patch start/width/speed/contact timing의 predeclared cells.
- Negative/control counterpart: `ICE_BENIGN_CONTROL` matched by source/speed/phase block.
- Expected value: current Hazard/Terrain asynchronous causal ordering stress test.
- Priority: `P0`.

### Scenario family: `DELAYED_SAND_SUPPORT_ONSET`

- Purpose: benign Sand가 observable해진 후 heterogeneous Support precursor와 established Support가 나오는 causal transition 검증.
- Current gap: I1 is contact+19 ms and precedes Terrain in 64/64 current Support runs.
- Source terrain: Concrete / Marble.
- Target terrain: benign/even Sand followed by heterogeneous Sand support.
- Hazard intent: later Support precursor/established event; label은 future oracle가 결정.
- Temporal intent: valid SAND, then at least one predeclared clean-contact interval, then I1/Support.
- Affected side intent: left reference first; right counterpart는 별도 family.
- Speed domain: 0.25.
- Physics/mechanics: initial balanced-deformable Sand segment followed by lateral-deformable segment.
- Negative/control counterpart: same source/speed/timing with balanced deformation maintained.
- Expected value: Terrain-first Support onset과 premature-reflex robustness.
- Priority: `P0`, but `REQUIRES_NEW_PHYSICS_CALIBRATION` because sequential support topology is not a current frozen scenario family.

### Scenario family: `RIGHT_DOMINANT_ICE_SLIP`

- Purpose: actual right-only Hazard and pelvis-IMU side generalization.
- Current gap: right-only Slip 0.
- Source terrain: Concrete / Marble.
- Target terrain: Ice.
- Hazard intent: right foot establishes Slip first; actual side is never preassigned.
- Temporal intent: current Hazard-first reference initially; delayed variant only after P0 feasibility.
- Affected side intent: `RIGHT_ONLY`.
- Speed domain: 0.25.
- Physics/mechanics: bilateral frozen Ice patch with patch-start/gait-phase timing chosen before outcomes; no unilateral friction retuning.
- Negative/control counterpart: matched left-dominant/bilateral cells and Ice benign control.
- Expected value: side symmetry without adding runtime sensors.
- Priority: `P1`.

### Scenario family: `RIGHT_SAND_SUPPORT`

- Purpose: right-side Support mechanics와 Hazard generalization.
- Current gap: Support is left-only 64/64.
- Source terrain: Concrete / Marble.
- Target terrain: Sand.
- Hazard intent: established Support from `transition_right` mechanics.
- Temporal intent: current established-delay reference; precursor timing recorded separately.
- Affected side intent: `RIGHT_ONLY`.
- Speed domain: 0.25.
- Physics/mechanics: existing transition_right and lateral-deformable primitives; no material retune.
- Negative/control counterpart: transition_right balanced-deformable benign control.
- Expected value: Hazard side symmetry; Terrain LEFT_ONLY availability는 별도 metric으로 보고.
- Priority: `P1`.

### Scenario family: `SPEED_STRATIFIED_HAZARD`

- Purpose: one-speed Hazard bias 제거.
- Current gap: all Slip/Support/Sand-benign transitions are 0.25 m/s.
- Source terrain: Concrete / Marble.
- Target terrain: Ice / Sand.
- Hazard intent: matched Slip, Support and benign controls; actual outcome oracle-driven.
- Temporal intent: current reference timing을 유지하되 각 speed에서 별도 보고.
- Affected side intent: existing reference side to isolate speed.
- Speed domain: representative 0.20 / 0.25 / 0.30 m/s; full dense sweep 금지.
- Physics/mechanics: same frozen materials and topology; hard-normal 0.1573–0.3650 envelope를 feasibility boundary로 사용.
- Negative/control counterpart: each speed에 matched no-Hazard control.
- Expected value: gait-frequency/IMU amplitude robustness.
- Priority: `P1`.

### Scenario family: `PHASE_SHIFTED_HAZARD`

- Purpose: missing or underrepresented contact phase의 onset robustness.
- Current gap: Contact release/double support 0; Support right-single 0 in authorized audit.
- Source terrain: Concrete / Marble.
- Target terrain: Ice / Sand.
- Hazard intent: same physics family with predeclared phase-targeted patch timing.
- Temporal intent: do not optimize after observed event; phase is audited after simulation.
- Affected side intent: balanced across intended left/right where existing mechanics permit.
- Speed domain: 0.25 to isolate phase.
- Physics/mechanics: patch start/timing values from current calibrated geometry domain.
- Negative/control counterpart: same phase-targeted geometry with benign target mechanics.
- Expected value: causal gait-state robustness.
- Priority: `P1`.

Additional material/severity ladders are `P2` and must use a separately frozen calibration. Arbitrary friction, stiffness, damping or travel changes to manufacture a desired result are prohibited.

## 16. Minimum informative next scenario set

다음 generation 설계는 일곱 family로 제한한다. 이는 run 수가 아니라 family topology다. Actual run count는 family acceptance, blocked design and statistical precision을 근거로 다음 milestone에서 정한다.

| Scenario ID | Source→target | Hazard intent | Temporal relation | Side | Speed strata | Patch/mechanics variation | Expected physical condition | Required balance |
|---|---|---|---|---|---|---|---|---|
| `ICE_BENIGN_CONTROL` | C/M→Ice | none | Terrain valid, stays benign | none | 0.25 | frozen Ice, predeclared short/bounded cells | valid ICE without established Hazard | 1:1 matched to delayed Slip cells |
| `DELAYED_ICE_SLIP` | C/M→Ice | Slip | Terrain valid >50 ms before Slip | bilateral reference | 0.25 | frozen start/width/contact timing | benign target interval then Slip | 1:1 Ice benign counterpart |
| `DELAYED_SAND_SUPPORT_ONSET` | C/M→Sand→heterogeneous Sand | Support | Terrain valid before I1 and Support | left | 0.25 | sequential balanced→lateral topology | benign Sand interval then support loss | 1:1 topology-matched benign control |
| `RIGHT_DOMINANT_ICE_SLIP` | C/M→Ice | Slip | current-order reference | right intent | 0.25 | phase/start timing only | right establishes first if physics does so | left/bilateral and benign matches |
| `RIGHT_SAND_SUPPORT` | C/M→Sand | Support | precursor and established reported separately | right | 0.25 | transition_right lateral deformation | right Support if oracle establishes it | right balanced-deformable benign 1:1 |
| `SPEED_STRATIFIED_HAZARD` | C/M→Ice/Sand | Slip/Support/none | same family timing | reference side | 0.20/0.25/0.30 | no material change | representative speed robustness | hazard/no-hazard per speed |
| `PHASE_SHIFTED_HAZARD` | C/M→Ice/Sand | Slip/Support/none | predeclared phase target | left/right intent | 0.25 | current-domain patch timing | phase diversity if naturally realized | phase-matched benign controls |

`DELAYED_SAND_SUPPORT_ONSET`은 별도 physics calibration을 통과하기 전 generation-eligible이 아니다. `ICE_BENIGN_CONTROL`과 `DELAYED_ICE_SLIP`도 existing frozen variables로 의도한 family가 형성되지 않으면 outcome에 맞춰 mechanics를 바꾸지 않고 calibration blocker를 기록한다.

## 17. Future split protocol

다음 dataset은 반드시 다음 순서를 지킨다.

```text
scenario specification
-> parameter-block and split freeze
-> simulation
-> frozen physical-oracle outcome
```

Split 설계 원칙은 다음과 같다.

- Source, target, intended side, speed stratum, temporal family, phase target와 mechanics topology를 포함한 scenario specification을 먼저 고정한다.
- Patch start/width처럼 가까운 parameter-near-duplicate는 block으로 묶어 한 split에만 둔다. 한 sweep의 인접 float를 TRAIN/VALIDATION/HOLDOUT에 흩어 놓지 않는다.
- Matched Hazard/control pair도 같은 split block에 둔다. Outcome이 다르다는 이유로 분리하거나 이동하지 않는다.
- VALIDATION은 known family 안의 disjoint parameter block으로 interpolation generalization을 본다.
- HOLDOUT은 predeclared entire subfamily 조합을 남겨 extrapolation을 본다. 예: unseen `source × side × speed × temporal/phase` cell. 어떤 cell인지는 simulation 전에 기록한다.
- TRAIN/VALIDATION/HOLDOUT은 actual Slip/Support/No-Hazard 비율을 본 뒤 재균형하지 않는다. Invalid outcome도 membership을 바꾸지 않고 invalid로 남긴다.
- 첫 zero-retrain test에서는 TRAIN을 학습에 쓰지 않는다. VALIDATION generalization을 먼저 보고 protocol/failure taxonomy를 freeze한 뒤에만 already-frozen current model로 HOLDOUT을 한 번 연다.
- HOLDOUT waveform access guard, open count, artifact identity와 no-reselection rule을 current protocol처럼 유지한다.

Interpolation HOLDOUT과 extrapolation HOLDOUT을 한 이름으로 섞지 않는다. 다음 config는 각 holdout block의 목적을 명시해야 한다.

## 18. Frozen-model generalization protocol

새 scenario가 생성되면 첫 행동은 retraining이 아니다.

```text
new predeclared scenario dataset
-> current frozen Hazard GRU
-> current frozen Terrain MLP
-> ZERO-RETRAIN generalization test
```

Hazard는 overall/Slip/Support recall, no-hazard specificity, premature rate, Slip/Support timing을 보고하고 side, speed, established temporal order, Support precursor order와 gait phase로 breakdown한다. Terrain은 target classification, first valid target update timing, unavailable rate와 LEFT_ONLY limitation을 보고한다. System은 Hazard versus physical event와 Hazard versus Terrain update를 함께 보고 Terrain-first/Hazard-first distribution을 만든다.

Current frozen model이 predeclared gate를 통과하면 우선 결론은 `NO RETRAIN`이다. 실패하면 다음 순서를 지킨다.

```text
failure
-> physical/scenario diagnosis
-> Pelvis IMU6 / FSR4 observability check
-> repeated systematic failure confirmation
-> only then retraining decision
```

한 failure나 desired outcome mismatch를 보고 threshold, history, feature, material 또는 split을 바꾸지 않는다.

## 19. Limitations

- Coverage는 current G1 policy와 MuJoCo engineering terrain에 한정된다. Ice/Sand는 measured material model이 아니다.
- 전체 256-run temporal/side/severity 수치는 authorized manifest metadata이며 waveform을 새로 재계산한 값이 아니다.
- Gait phase와 loaded-contact episode는 TRAIN/VALIDATION만 계산했으므로 HOLDOUT 일반화를 주장하지 않는다.
- Loaded-contact episode는 force hysteresis edge이며 exact step/gait cycle이 아니다. Future generation은 exact clean touchdown identity 또는 predeclared debounce가 필요하면 결과를 보기 전에 별도 정의해야 한다.
- Current LEFT_ONLY Terrain update availability와 physical right-only Hazard coverage는 서로 다른 축이다.
- Continuous outcome variation은 causal design diversity가 아니다. 특히 Support severity는 매우 좁다.
- 이번 milestone은 run count/power analysis, new simulation, model evaluation 또는 model verdict를 만들지 않았다.

## 20. Protected-change audit and verdict

| Protected item | Result |
|---|---|
| Hazard 80D / `[20,80]` / GRU hidden32 / ensemble 3 | unchanged / verified |
| Hazard threshold 0.99 / persistence 5 ms | unchanged / verified |
| Hazard checkpoints, normalizer, feature schema, freeze SHA | hash PASS |
| Terrain FSR4 / `[50,4]` / MLP / ensemble 3 / four classes | unchanged / verified |
| Terrain normalizer/checkpoints / advisory-only role | hash PASS |
| Existing NPZ modified | 0; 256/256 byte-level size/SHA PASS |
| Existing manifest modified | 0; declared SHA PASS |
| New simulation generated | 0 |
| Model retraining / threshold retuning / artifact refreeze | 0 / 0 / 0 |
| HOLDOUT waveform opened | 0 |
| HOLDOUT new scientific evaluation | 0 |

Current corpus는 four-family baseline 안에서는 balanced source, Slip/Support/no-hazard 및 Sand benign coverage가 충분하다. 그러나 Ice benign, Terrain-first delayed Slip, Terrain-first Support precursor onset, right-only Hazard와 hazard-speed coverage가 구조적으로 비어 있다. 이 gap을 채우는 seven-family minimum design, calibration boundary, matched controls, future split 및 zero-retrain protocol이 사전에 정의됐다.

`SCENARIO_COVERAGE_DESIGN_READY`

이 verdict는 current model 성능 verdict가 아니며 새 data 생성 승인도 아니다.
