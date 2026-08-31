# Generalization Scenario Calibration

Date: 2026-08-31 (Asia/Seoul)
Milestone: `GENERALIZATION_SCENARIO_CALIBRATION`
Verdict: `GENERALIZATION_SCENARIO_CALIBRATION_BLOCKED`

## 1. Purpose

이번 milestone은 `SCENARIO_COVERAGE_DESIGN_READY`에서 고정한 일곱 scenario family를 실제 MuJoCo physics에서 소규모 deterministic pilot으로 calibration한 작업이다. 목적은 full dataset을 만드는 것이 아니라 current mechanics에서 family semantics가 물리적으로 성립하는지 확인하고, model output과 독립적으로 다음 generation 후보를 동결하는 것이다.

작업 시작 기준은 branch `main`, `HEAD = origin/main = 49ae692ae202de41464753e7558df423127509ca`, tracked worktree clean이었다. 이 commit에는 coverage design config/report가 포함돼 있었다. Pilot config는 simulation 전에 [`../configs/experiment/20260831_generalization_scenario_calibration.yaml`](../configs/experiment/20260831_generalization_scenario_calibration.yaml)에 기록했고 SHA-256 `42088569621bf91cce22a12501698f23ae27b529f17fd356bbf8777f2ccab9c6`으로 동결했다.

Canonical pilot matrix는 78 signatures다. 최초 78-run 뒤 staged topology의 collision-mask implementation defect를 물리 contact audit로 발견해, grid와 criteria를 바꾸지 않고 해당 8 signatures만 corrected source로 다시 실행했다. 따라서 canonical calibration 결과는 78 pilots이고 실제 실행 trajectory는 86개다. 두 attempt의 동일한 8 signatures는 모두 future evaluation에서 제외된다.

## 2. Coverage gaps being targeted

이전 audit의 gap은 재정의하지 않았다.

- Ice benign: 0.
- Terrain-first Slip: 0; available Ice 48/48은 Hazard-first였다.
- Established Support는 Terrain-first case가 있었지만 I1은 Support 64/64에서 contact 약 +19 ms였다.
- Physical right-only Hazard: 0.
- Hazard와 Sand transition speed는 0.25 m/s 한 값에 집중됐다.
- TRAIN/VALIDATION Hazard phase는 6개 중 3개만 관찰됐다.

이번 calibration은 이 gap을 일곱 exact family 안에서만 다뤘고 새 family나 post-hoc severity class를 추가하지 않았다.

## 3. Protected baseline

Hazard contract는 다음과 같이 유지했다.

```text
Pelvis IMU6 @ 1 kHz -> causal 80D -> [20,80]
-> one-layer GRU hidden32 -> three-seed mean
-> threshold 0.99 -> persistence 5 ms -> REFLEX_REQUIRED
```

- Freeze SHA-256: `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`.
- Feature schema SHA-256: `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`.
- Model/normalizer/checkpoint, history, threshold와 persistence는 read-only였다.

Terrain contract도 `touchdown FSR4 -> [50,4] -> three-seed MLP -> held advisory state`로 유지했다. Normalizer SHA-256은 `2c551181fbd23080c0e3008be8484bd0ee4139e594041ba6e46ab5b36ffb05de`이며 세 checkpoint hash verification을 통과했다. Terrain은 Hazard를 gate하지 않는다.

Ice friction, Sand material/joint profile, G1 policy, controller, 0.5 ms physics timestep, 1 kHz sensor rate와 current Slip/Support/I1 정의는 변경하지 않았다.

## 4. Model-blind calibration protocol

Selection evidence는 actual terrain contact, exact 50 ms target touchdown, tangential drift, established Slip, support spread, I1, established Support, actual affected side, gait phase, fall/censor와 geometry뿐이었다. Hazard probability, `REFLEX_REQUIRED`, Terrain prediction/probability와 model success/failure는 selection input이 아니었다.

Physical rules는 실행 전에 고정했다.

- 공통: finite 8,000 samples, target contact, normal prefix ≥500 ms, no pre-transition fall.
- Slip: touchdown-anchor tangential drift ≥0.050 m for 3 ms.
- Support: support spread ≥0.010 m for 20 ms.
- I1: loaded-foot positive spread derivative for 20 ms.
- Clean target touchdown: exact target identity for 50 ms and mixed-contact ratio <0.20.
- Phase: physical event에서 10 ms lookback, coverage audit의 priority를 그대로 사용.
- Tie break: physical semantics와 diversity 뒤 lexicographic pilot ID.

원래 grid freeze artifact는 `PILOT_GRID_FROZEN_BEFORE_SIMULATION`이며 78/78 unique signatures와 current dataset overlap 0을 기록했다. 최초 physical selection은 `dc3b380123c9508eea78049eb064c748a3523fa1d3ce14901d07656d2160ee5b`였다.

그 뒤 READY였던 `RIGHT_SAND_SUPPORT`와 `SPEED_STRATIFIED_HAZARD` 대표에 initial post-freeze smoke를 수행했지만, 별도의 topology contact audit에서 staged static/tile 경계가 sample 0부터 `nonfoot_surface_contact`를 만드는 collision-mask 누락을 발견했다. 이 defect는 model output이 아니라 MuJoCo contact pair와 physical fall oracle로 발견했다. Candidate, material/joint parameter와 criteria를 바꾸지 않고 correction scope 8개를 실행 전에 artifact `7e47db02a0cc573b58babe9346aea716ef0bb90be0d108a3113aad1f3b918218`로 고정했다. Corrected family에는 이때까지 model replay가 없었다.

Final physical selection freeze는 `da275e1d606c4e4592eecd7c41a1ba5ff7e6773f1e3f28cfe050c3f8e39e8ff9`다. 최초 selection/smoke/visualization JSON과 잘못된 8 NPZ는 `before_implementation_correction` provenance로 보존했다. Final smoke는 이 final freeze 뒤 다시 수행했다. 이 correction 과정에서도 model 결과로 parameter, viability 또는 selected ID를 바꾼 횟수는 0이다.

## 5. Seven scenario families

| Scenario family | Priority | Scientific purpose | Source → target | Hazard / temporal intent | Side | Speed / phase intent | Mechanics | Calibration |
|---|---|---|---|---|---|---|---|---|
| `ICE_BENIGN_CONTROL` | P0 | Ice identity hard negative | Concrete/Marble → Ice | clean Ice opportunity, no Slip/I1/Support | none | 0.25 / natural | frozen bilateral Ice transition | required |
| `DELAYED_ICE_SLIP` | P0 | benign Ice interval 뒤 later Slip | Concrete/Marble → Ice | contact, ≥1 clean touchdown, Slip ≥1,000 ms later | oracle; bilateral reference | 0.25 / natural | frozen bilateral Ice transition | required |
| `DELAYED_SAND_SUPPORT_ONSET` | P0 | benign Sand 뒤 delayed I1/Support | Concrete/Marble → Sand | contact, I1 ≥300 ms later, then Support | left-only | 0.25 / natural | staged static Sand entry → exit lateral deformation | required; new physics topology |
| `RIGHT_DOMINANT_ICE_SLIP` | P1 | actual right-only Slip | Concrete/Marble → Ice | current-order Slip | right-only | 0.25 / natural | bilateral Ice geometry, predeclared timing | required |
| `RIGHT_SAND_SUPPORT` | P1 | actual right-only Support | Concrete/Marble → Sand | I1 and established Support reported separately | right-only | 0.25 / right-side natural | existing transition-right lateral deformation | required |
| `SPEED_STRATIFIED_HAZARD` | P1 | one-speed bias audit | Concrete/Marble → Ice/Sand | matched Slip/Support/Sand-benign | reference side | 0.20/0.25/0.30 | existing frozen mechanics | required |
| `PHASE_SHIFTED_HAZARD` | P1 | missing phase feasibility | Concrete/Marble → Ice/Sand | Slip/Support at natural contact-release or double-support | oracle | 0.25 / phase audit | existing patch-start timing only | required |

Family와 within-family parameter sweep는 분리했다. Patch start나 width 차이는 새 family가 아니다.

## 6. Pilot specification

모든 run은 8 s, 2 kHz physics, 1 kHz runtime sensors, verified G1 policy를 사용했다. Grid는 결과를 보기 전에 다음처럼 고정했다.

| Family | Expansion | Pilot count |
|---|---|---:|
| `ICE_BENIGN_CONTROL` | source 2 × start {0.30,0.35} × width {0.10,0.20,0.35} m | 12 |
| `DELAYED_ICE_SLIP` | source 2 × start {0.28,0.33} × width {0.75,1.00} m | 8 |
| `DELAYED_SAND_SUPPORT_ONSET` | source 2 × start {0.30,0.35} × width {0.80,1.20} m | 8 |
| `RIGHT_DOMINANT_ICE_SLIP` | source 2 × start {0.27,0.35} × width {0.70,0.80} m | 8 |
| `RIGHT_SAND_SUPPORT` | source 2 × start {0.30,0.35} × width {0.67,0.73} m | 8 |
| `SPEED_STRATIFIED_HAZARD` | source 2 × speed 3 × {Ice Slip, Sand Support, Sand benign} | 18 |
| `PHASE_SHIFTED_HAZARD` | source 2 × start {0.285,0.315,0.345,0.375} × {Ice Slip, Sand Support} | 16 |
| **Total** | seven families | **78** |

Round 2 parameter expansion은 수행하지 않았다. Implementation correction은 같은 8 signatures를 같은 criteria로 재실행한 source defect correction이며 parameter calibration round가 아니다.

## 7. Physical viability results

`Valid`는 `NONFINITE`, `NO_TARGET_CONTACT`, `PRETRANSITION_FALL`을 제외한 수다. `Mismatch`에는 timing/side/physical mismatch와 pre-transition fall을 모두 표시한다.

| Family | Priority | Pilot runs | Valid | Physically viable | Primary issue | Next generation |
|---|---:|---:|---:|---:|---|---|
| `ICE_BENIGN_CONTROL` | P0 | 12 | 12 | 0 | 6 Slip; 나머지 6은 clean 50 ms Ice touchdown 0 | `BLOCKED` |
| `DELAYED_ICE_SLIP` | P0 | 8 | 4 | 0 | 4 timing mismatch, 4 pre-transition fall | `BLOCKED` |
| `DELAYED_SAND_SUPPORT_ONSET` | P0 | 8 | 8 | 4 | width 0.80 m 4 viable; width 1.20 m 4 no I1/Support | `INCLUDE` |
| `RIGHT_DOMINANT_ICE_SLIP` | P1 | 8 | 8 | 0 | 8/8 actual bilateral, not right-only | `BLOCKED` |
| `RIGHT_SAND_SUPPORT` | P1 | 8 | 8 | 8 | 8/8 actual right-only Support | `INCLUDE` |
| `SPEED_STRATIFIED_HAZARD` | P1 | 18 | 18 | 18 | all 3 speeds × roles × sources viable | `INCLUDE` |
| `PHASE_SHIFTED_HAZARD` | P1 | 16 | 16 | 0 | contact-release/double-support 0; 15 phase mismatch, 1 outcome mismatch | `BLOCKED` |

Canonical outcome은 3 READY families와 4 BLOCKED families다. P0 두 개가 BLOCKED이므로 부분 성공을 전체 READY로 바꾸지 않는다.

## 8. Ice benign

`ICE_BENIGN_CONTROL`은 얻지 못했다.

- 12개 모두 target first contact는 sample 1220이었다.
- 6개는 established Slip이 없었지만 clean target touchdown이 0이었다.
- 나머지 6개는 established Slip을 만들었다: bilateral 3, left-only 3.
- Support와 I1은 전부 없었지만 `clean Ice opportunity + no Hazard`의 결합 조건을 만족한 run은 0이었다.
- Ice friction은 바꾸지 않았다.

결론은 `ICE_BENIGN_NOT_OBTAINED_WITH_CURRENT_CALIBRATED_DOMAIN`이다. 이 family는 새 physics calibration 전까지 generation에 포함하지 않는다.

## 9. Delayed Slip

`DELAYED_ICE_SLIP`도 얻지 못했다.

- 4개는 target contact 전에 fall censor가 발생했다.
- 나머지 4개는 bilateral Slip과 Slip 전 clean target touchdown 1개를 만들었다.
- 그러나 contact→Slip은 688–690 ms로 사전 고정한 ≥1,000 ms를 만족하지 못했다.
- I1과 Support는 없었다.

따라서 clean Ice contact가 먼저 있다는 사실만으로 delayed family를 주장하지 않는다. `DELAYED_ICE_SLIP`은 timing-order blocker다.

## 10. Delayed Support

`DELAYED_SAND_SUPPORT_ONSET`은 corrected staged topology에서 4/8이 성립했다. 새 option은 material parameter를 추가하지 않고 기존 static Sand, reference deformable profile과 moderate lateral profile을 공간적으로 순차 배치한다.

| Selected pilot | Source | Start / width (m) | First contact | Clean touchdowns ending before I1 | I1 | Established Support | I1→Support | Fall/censor |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `gsc_dss_001` | Concrete | 0.30 / 0.80 | 1220 | 2 | 3011 | 3067 | 56 ms | 8000 censor |
| `gsc_dss_003` | Concrete | 0.35 / 0.80 | 1220 | 2 | 3011 | 3067 | 56 ms | 8000 censor |
| `gsc_dss_005` | Marble | 0.30 / 0.80 | 1220 | 2 | 3012 | 3068 | 56 ms | 8000 censor |
| `gsc_dss_007` | Marble | 0.35 / 0.80 | 1220 | 2 | 3012 | 3068 | 56 ms | 8000 censor |

Contact→I1은 1,791–1,792 ms, contact→Support는 1,847–1,848 ms였다. 즉 exact Sand contact 뒤 I1이 false인 긴 물리 interval과 최소 두 clean target touchdown이 먼저 존재했다. Support는 actual left-only였고 pre-transition fall은 없었다. Width 1.20 m 네 후보는 Slip은 없었으나 I1/Support가 발생하지 않아 `PHYSICAL_OUTCOME_MISMATCH`였다.

이 결과는 physical calibration만으로 선택했다. Frozen Terrain output이 I1보다 앞섰는지는 section 17의 별도 diagnostic이다.

## 11. Side diversity

- Left-only: selected delayed-Sand 4개와 speed-stratified Support 6개.
- Right-only: `RIGHT_SAND_SUPPORT` 8/8; 이 중 diversity rule로 4개를 선택했다.
- Bilateral: speed-stratified Ice Slip 6/6.
- Right-only Slip: `RIGHT_DOMINANT_ICE_SLIP` 8개가 모두 bilateral이어서 0.

따라서 selected set은 left Support, right Support와 bilateral Slip을 포함하지만 right-only Slip은 여전히 없다. Designed side가 아니라 actual established event side로 판정했다.

## 12. Speed robustness

`SPEED_STRATIFIED_HAZARD`는 0.20/0.25/0.30 m/s 각각에 Concrete/Marble source와 Slip/Support/Sand-benign role이 하나씩 있어 18/18 viable이었다.

| Speed | Slip | Support | Benign | Actual phase summary |
|---:|---:|---:|---:|---|
| 0.20 | 2 bilateral | 2 left-only | 2 no-Hazard | Slip/Support left-single |
| 0.25 | 2 bilateral | 2 left-only | 2 no-Hazard | Slip/Support left-single |
| 0.30 | 2 bilateral | 2 left-only | 2 no-Hazard | Slip right-single; Support touchdown-loading |

Finite runtime, target contact와 stable prefix 조건은 모든 speed cell에서 통과했다. Dense speed sweep이나 policy domain 확장은 없었다.

## 13. Gait-phase coverage

`PHASE_SHIFTED_HAZARD`의 accepted target은 `CONTACT_RELEASE`와 `DOUBLE_SUPPORT`였다. Actual event phase는 left-single 10, touchdown-loading 4, right-single 1, no physical Support event 1이었다. Accepted target phase는 0이므로 family는 BLOCKED다.

Policy state 초기화, future-event steering 또는 result-driven patch retuning은 하지 않았다. Selected READY families에서 자연스럽게 얻은 phase는 left-single, right-single와 touchdown-loading이며, 별도 phase family의 intended missing phase를 대체한다고 주장하지 않는다.

## 14. Selected scenario specifications

다음 generation이 참조할 수 있는 family-level freeze는 아래 세 family다. `Allowed domain`은 family mechanism의 bounded generation domain이며, exact 78 pilot signatures는 반드시 제외한다. 새 run/split ID와 exact fresh parameter grid는 다음 milestone에서 outcome을 보기 전에 별도로 고정해야 한다.

| Family | Physics mechanism | Allowed domain | Required physical relation | Forbidden outcome | Known limitation |
|---|---|---|---|---|---|
| `DELAYED_SAND_SUPPORT_ONSET` | `transition_left + staged_lateral_deformable`, static Sand entry 뒤 exit lateral moderate deformation | source C/M; speed 0.25; start 0.30–0.35 m; calibrated width seed 0.80 m; left | contact → ≥2 clean target touchdowns before I1 → left-only Support | Slip, pre-transition fall, I1 <300 ms, non-left Support | new explicit opt-in topology; width 1.20 m failed; matched benign control still required |
| `RIGHT_SAND_SUPPORT` | existing `transition_right + lateral_deformable`, moderate | source C/M; speed 0.25; start 0.30–0.35 m; width 0.67–0.73 m; right | I1 and actual right-only established Support | Slip, left/bilateral Support, pre-transition fall | LEFT_ONLY Terrain availability is separate; matched right benign control required |
| `SPEED_STRATIFIED_HAZARD` | existing Ice transition, left Sand Support, balanced Sand benign | source C/M; speeds 0.20/0.25/0.30; frozen material/topology; bounded existing geometry | each speed/source retains Slip, Support and Sand-benign roles by actual oracle | nonfinite, no contact, pre-transition fall; role mismatch invalid | no Ice benign; exact pilot geometry cannot be reused |

Fresh generation values must be new physical signatures inside these domains; viability is still decided by the physical oracle without moving split membership. `ICE_BENIGN_CONTROL`, `DELAYED_ICE_SLIP`, `RIGHT_DOMINANT_ICE_SLIP`과 `PHASE_SHIFTED_HAZARD`는 승인 목록에 없다.

## 15. Physical timing table

아래 표는 selected Hazard-oriented pilots의 physical calibration clocks다. Sample과 ms는 1 kHz에서 수치가 같다. `Clean TD`는 exact target 50 ms window가 physical event 전에 끝난 수다. `Fall/censor`의 8000은 fall 없이 duration censor에 도달했다는 뜻이다.

| Pilot | Family/intent | Src / speed | Contact | Clean TD | I1 | Slip | Support | Fall/censor | Phase / side |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `gsc_dss_001` | delayed Sand / Support | C / .25 | 1220 | 3 | 3011 | — | 3067 | 8000 | left-single / left |
| `gsc_dss_003` | delayed Sand / Support | C / .25 | 1220 | 3 | 3011 | — | 3067 | 8000 | left-single / left |
| `gsc_dss_005` | delayed Sand / Support | M / .25 | 1220 | 3 | 3012 | — | 3068 | 8000 | left-single / left |
| `gsc_dss_007` | delayed Sand / Support | M / .25 | 1220 | 3 | 3012 | — | 3068 | 8000 | left-single / left |
| `gsc_rss_001` | right Sand / Support | C / .25 | 1507 | 1 | 1526 | — | 2173 | 3895 | right-single / right |
| `gsc_rss_002` | right Sand / Support | C / .25 | 1507 | 1 | 1526 | — | 2173 | 8000 | right-single / right |
| `gsc_rss_003` | right Sand / Support | C / .25 | 1507 | 1 | 1526 | — | 2173 | 8000 | right-single / right |
| `gsc_rss_008` | right Sand / Support | M / .25 | 1509 | 1 | 1528 | — | 2176 | 8000 | right-single / right |
| `gsc_ssh_001` | speed / Slip | C / .20 | 1503 | 1 | — | 2492 | — | 6202 | left-single / bilateral |
| `gsc_ssh_002` | speed / Slip | C / .25 | 1220 | 1 | — | 1910 | — | 5164 | left-single / bilateral |
| `gsc_ssh_003` | speed / Slip | C / .30 | 1227 | 0 | — | 1605 | — | 3153 | right-single / bilateral |
| `gsc_ssh_004` | speed / Slip | M / .20 | 1491 | 2 | — | 2323 | — | 4552 | left-single / bilateral |
| `gsc_ssh_005` | speed / Slip | M / .25 | 1220 | 1 | — | 1908 | — | 4356 | left-single / bilateral |
| `gsc_ssh_006` | speed / Slip | M / .30 | 1227 | 1 | — | 1597 | — | 3696 | right-single / bilateral |
| `gsc_ssh_007` | speed / Support | C / .20 | 1808 | 1 | 1827 | — | 2475 | 5769 | left-single / left |
| `gsc_ssh_008` | speed / Support | C / .25 | 1220 | 2 | 1239 | — | 2475 | 4232 | left-single / left |
| `gsc_ssh_009` | speed / Support | C / .30 | 1227 | 3 | 1246 | — | 3036 | 3599 | touchdown / left |
| `gsc_ssh_010` | speed / Support | M / .20 | 1810 | 1 | 1829 | — | 2476 | 8000 | left-single / left |
| `gsc_ssh_011` | speed / Support | M / .25 | 1220 | 2 | 1239 | — | 2474 | 5172 | left-single / left |
| `gsc_ssh_012` | speed / Support | M / .30 | 1227 | 3 | 1246 | — | 3036 | 3948 | touchdown / left |

Sand-benign speed pilots 6개도 selected family specification에 포함되지만 physical event가 없으므로 이 Hazard-oriented timing 표에서는 제외했다.

## 16. Pilot signature exclusion

Physical signature는 `source_terrain, target_terrain, speed_mps, patch_start_x_m, patch_width_m, slip_pattern, sink_pattern, sink_severity, support_pattern` 아홉 필드다.

- Canonical pilot signatures: 78.
- Unique signatures: 78.
- Current 256-run dataset signature overlap: 0.
- Future evaluation denominator에서 제외할 signatures: 78/78.
- Implementation defect 전후 staged trajectory도 같은 8 signatures이므로 둘 다 자동 제외된다.

다음 generation은 `calibration signatures ∩ final evaluation signatures = 0`을 simulation 전에 검증해야 한다. Pilot waveform이나 model smoke 결과를 generalization metric denominator로 재사용하지 않는다.

## 17. Frozen-model post-freeze smoke

Final physical selection 뒤 READY family별 lexicographic first selected pilot 세 개에 protected models를 read-only로 실행했다. 이는 performance evaluation이 아니고 selection을 바꾸지 않았다.

| Pilot | Physical reference | First Reflex | First valid target Terrain | Terrain-first annotation | Diagnostic only |
|---|---|---:|---:|---|---|
| `gsc_dss_001` | I1 3011; Support 3067 | 2461 | SAND 1851 | `TERRAIN_FIRST_CONFIRMED` | Reflex -550 ms vs I1, -606 ms vs Support |
| `gsc_rss_001` | I1 1526; Support 2173 | unavailable | SAND 3008 | `TERRAIN_FIRST_NOT_OBSERVED` | frozen Hazard miss on smoke |
| `gsc_ssh_001` | Slip 2492 | 2472 | ICE 2445 | `TERRAIN_FIRST_CONFIRMED` | Reflex -20 ms vs Slip |

Hazard/Terrain artifact verification은 모두 PASS였다. Terrain은 advisory-only로 유지됐다. `gsc_rss_001` miss를 보고 scenario를 폐기하거나 대체 ID를 고르지 않았다. Smoke artifact SHA-256은 `db28446a9c93810a64c44a589c9f44358ec91b23be43d48da8b6c296d6cb0392`다.

## 18. Visualization sanity

동일한 세 representative에 대해 8,000개의 immutable MuJoCo integration snapshots를 캡처했고 각 state size는 457이었다. Physical event sample에서 640×480 offscreen frame을 사람이 확인했다.

- `gsc_dss_001`: static Sand entry와 뒤쪽 staged deformable strip, upright G1 pose가 보였다.
- `gsc_rss_001`: right-side heterogeneous Sand strip과 event pose가 보였다.
- `gsc_ssh_001`: full-width Ice patch와 event pose가 보였다.

세 rerun은 timestamp, IMU6, FSR8, exact terrain contact, loaded contact, drift와 support spread가 pilot NPZ와 exact parity였다. Visualization artifact SHA-256은 `8d22454040787a28b1cc66d1a2bb7c3695d537ae50ce3b057e89c3dcc922af12`다. 화면이나 model trace는 selection input이 아니었다.

## 19. Limitations and next dataset generation contract

Limitations은 다음과 같다.

- Ice benign과 ≥1,000 ms delayed Ice Slip이라는 P0 gap이 남았다.
- Right-only Slip과 intended missing gait phases가 성립하지 않았다.
- Staged Support는 새 explicit opt-in topology이며 width 0.80 m seed에서만 4/4 성립했다. 별도의 fresh matched benign control이 필요하다.
- Engineering Ice/Sand profile은 measured real material model이 아니다.
- Model smoke는 세 calibration representatives뿐이며 recall/specificity denominator가 아니다.
- 최초 selection 뒤 initial smoke가 실행된 후 staged collision-mask defect를 발견했다. Correction은 model 결과와 무관하고 corrected family에는 이전 model output이 없었지만, config의 single-pass `selection_may_change_after_model_smoke=false` 순서를 완벽히 만족했다고 주장하지 않는다. 이 procedural deviation도 overall BLOCKED verdict의 보수적 근거이며, provenance를 단일-pass처럼 숨기지 않고 before-correction artifacts와 correction freeze를 보존했다.

새 option은 explicit `staged_lateral_deformable`에서만 활성화된다. Existing validation representatives `uhr_ice_h_c20`, `uhr_sand_h_c20`, `uhr_sand_b_c20`, `uhr_hard_n_c20`은 timestamp, IMU6, FSR8, physical event clocks와 diagnostic arrays 12항목 모두 tolerance 0 exact parity를 통과했다. 기존 source/default behavior는 바뀌지 않았다.

예정된 다음 milestone은 `GENERALIZATION_DATASET_GENERATION_AND_ZERO_RETRAIN_EVALUATION`이지만 이번 작업에서 자동 시작하지 않는다. 전체 intended generalization dataset은 두 P0 Ice blocker를 해결하고 별도 calibration freeze를 남기기 전에는 시작하면 안 된다. 이후 순서는 다음과 같다.

```text
approved scenario families
-> fresh physical signatures and matched controls
-> parameter block + split freeze BEFORE simulation
-> full generation
-> actual physical labels, with invalids not moved
-> current frozen models
-> VALIDATION protocol/failure taxonomy freeze
-> one-shot sealed new HOLDOUT
-> zero-retrain evaluation
```

Current `unified_hazard_reflex_20260829` HOLDOUT 52는 열지 않았다. Existing dataset/manifest 수정 0, optimizer steps 0, checkpoint writes 0, normalizer refits 0, threshold searches 0이다.

```text
Current HOLDOUT waveform reopened: NO
Current HOLDOUT new inference: NO
```

Verification은 full `pytest` 71 passed / 1 skipped, `python -m compileall src scripts tests` PASS, Ruff `E9,F63,F7,F82` PASS였다. Protected Hazard/Terrain verification도 PASS했고, existing manifest SHA-256 `d023384aaaac22076a2be7c2fa242acac91a548d82102640c39868990cb0e9d6`와 256/256 NPZ hashes가 일치했다. New training artifact는 0이다.

## 20. Verdict

READY families:

- `DELAYED_SAND_SUPPORT_ONSET` (P0)
- `RIGHT_SAND_SUPPORT` (P1)
- `SPEED_STRATIFIED_HAZARD` (P1)

BLOCKED families:

- `ICE_BENIGN_CONTROL` (P0): current frozen Ice domain에서 clean benign Ice를 얻지 못함.
- `DELAYED_ICE_SLIP` (P0): viable Slip의 contact delay가 688–690 ms로 ≥1,000 ms 기준 미달.
- `RIGHT_DOMINANT_ICE_SLIP` (P1): 8/8 bilateral Slip.
- `PHASE_SHIFTED_HAZARD` (P1): contact-release/double-support event 0.

중요 P0 두 family가 현재 mechanics에서 generation-eligible하지 않으므로 충분한 seven-family set을 freeze하지 못했다.

`GENERALIZATION_SCENARIO_CALIBRATION_BLOCKED`

이 verdict는 current supported Hazard/Terrain model verdict를 취소하지 않으며, model miss를 근거로 만든 verdict도 아니다.
