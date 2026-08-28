# Terrain Transition Scenario Calibration

Milestone: `TRANSITION_SCENARIO_CALIBRATION`

## 1. Previous integrated-sanity failure

Previous HEAD `1145105ea44d4ecde38df20dcfe6d92dc0bf1ba6`의 44-run integrated sanity는 Ice stable-intended `1/8`, Sand stable-intended `4/8`, pre-transition fall `12`였고 verdict는 `INTEGRATED_SCENARIO_NEEDS_REVISION`이었다. Exact MoS와 pelvis-IMU rule 결과는 historical evidence로 보존했다. 이번 작업은 Stability oracle, IMU threshold, GRU, Terrain model, recovery controller 또는 dataset을 수정하지 않았다.

작업 시작 시 실제 HEAD와 `origin/main`은 모두 위 SHA였고 worktree는 clean이었다.

## 2. Why scenario calibration precedes stability work

Target contact 전에 future Ice/Sand가 robot state나 controller action을 바꾸거나 robot이 먼저 넘어지면 이후의 instability clock과 detector latency는 의미가 없다. 따라서 순서를 다음처럼 고정했다.

```text
matched hard-prefix parity
→ bounded calibration
→ CALIBRATION_SELECTED freeze
→ calibration-unused Concrete validation
→ same frozen B conditions on Marble
```

Observed outcome은 intended name과 무관하게 `VALID_STABLE`, `VALID_FALL`, `INVALID_PRETRANSITION`, `INVALID_NO_TARGET_CONTACT`, `INVALID_OTHER`로 분류했다. Stable은 target contact, 최소 500 ms post-contact observation, non-fall과 finite patch 완전 통과를 모두 요구한다.

## 3. Prefix parity methodology

Transition과 matched reference는 같은 G1 initial state, policy SHA, speed, timing, duration, dt, A terrain, controller와 patch scene을 사용했다. Reference는 target patch start만 `8.0 m`로 옮겨 run 동안 A-only ground가 되게 했다. Sand support joint qpos/qvel은 robot comparison에서 제외했다.

첫 target contact 1 ms, 즉 2 physics steps 전까지 robot qpos/qvel, pelvis IMU6, FSR8, controller observation/action, policy update timing, pelvis pose, whole-body COM과 foot-contact boolean을 비교했다. 이는 predeclared 최소 safety margin 1 physics step보다 보수적이다. MuJoCo broadphase ordering 때문에 robot double state의 bitwise identity는 불가능했으며, 결과를 보기 전에 qpos/qvel/pelvis/COM에 `1e-12` absolute tolerance를 선언했다. IMU, FSR, observation, action과 discrete signals는 exact equality를 요구했다.

| Pair | Compare end / target contact (ms) | max qpos | max qvel | max IMU | max FSR | max action | Contact mismatch | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Concrete→Ice | 1220 / 1221 | 2.325e-15 | 1.781e-13 | 0 | 0 | 0 | 0 | `TRANSITION_PREFIX_PARITY_PASS` |
| Concrete→Sand | 1220 / 1221 | 2.325e-15 | 1.781e-13 | 0 | 0 | 0 | 0 | `TRANSITION_PREFIX_PARITY_PASS` |
| Marble→Ice | 1220 / 1221 | 6.849e-15 | 3.588e-13 | 0 | 0 | 0 | 0 | `TRANSITION_PREFIX_PARITY_PASS` |
| Marble→Sand | 1220 / 1221 | 6.849e-15 | 3.588e-13 | 0 | 0 | 0 | 0 | `TRANSITION_PREFIX_PARITY_PASS` |

모든 pair에서 controller observation/action과 policy-update mismatch도 0, pretarget dynamic-support contact와 pretarget fall도 0이었다.

## 4. Geometry/contact audit

- Target geoms의 minimum x는 patch boundary보다 앞서지 않았다.
- Pre ground의 end, target start, target end와 post ground의 start가 `1e-12 m` 이내에서 맞았다.
- Static ground와 initial deformable tile top은 모두 `z=0`이었다.
- Sand의 enabled cells는 B patch 안에만 존재하고 tile/world 및 tile/tile collision filter를 유지했다.
- `t_boundary_expected`는 canonical sole leading edge가 boundary에 도달한 diagnostic clock이고, primary transition clock은 named target geom의 actual contact다.
- 첫 target contact 이전 target contact, dynamic support contact, hidden overlap, initial state mismatch는 없었다.

Geometry/contact verdict는 네 transition 모두 PASS다.

## 5. Concrete→Ice calibration

Frozen Ice friction `(0.05, 0.001, 0.00001)`을 바꾸지 않고 speed `0.15/0.20/0.25 m/s`, patch start `0.25–0.37 m`, width `0.20–0.80 m`의 deterministic 23-run region을 탐색했다.

Observed result는 stable `3`, fall `16`, invalid `4`였다. Invalid 중 pre-transition fall은 3개였고 한 non-fall run은 finite patch를 완전히 통과하지 못해 `INVALID_OTHER`였다. 짧은 Ice patch도 자동 stable이 아니었고 intended-stable 다수가 실제 fall이었다. Slip diagnostic은 selected stable에서도 발생했으므로 Slip occurrence와 walking-stability outcome이 분리됨을 확인했다.

Selected Ice conditions:

| Outcome | Speed | Start / width (m) |
|---|---:|---|
| Stable | 0.25 | 0.33/0.75, 0.36/0.70, 0.36/0.75 |
| Fall | 0.25 | 0.33/0.70, 0.34/0.75 |
| Fall | 0.15 | 0.32/0.65, 0.35/0.75 |

## 6. Concrete→Sand calibration

Passive deformable-support mechanics는 reference/mild/moderate/severe travel `4/20/40/65 mm`, stiffness `50,000/12,000/7,000/4,500 N/m`, damping `1,000/490/374/300 N·s/m` 그대로였다. Speed, patch geometry, side, balanced/lateral pattern과 frozen severity만 선택했다.

14-run region에서 observed stable `6`, fall `7`, invalid `1`이었다. Mild balanced stable run도 약 20 mm 실제 support deformation이 있었고, moderate lateral fall은 약 40 mm deformation과 frozen Sink diagnostic을 만들었다.

Selected Sand conditions:

| Outcome | Mechanics | Speed | Start / width (m) |
|---|---|---:|---|
| Stable | mild balanced left/right | 0.25 | left 0.30/0.72, 0.30/0.74; right 0.35/0.72, 0.35/0.74 |
| Fall | moderate lateral left | 0.25 | 0.30/0.65, 0.30/0.75, 0.35/0.65, 0.35/0.75 |

## 7. Frozen operating points

Config의 `frozen_operating_points.status`는 `CALIBRATION_SELECTED`다. Selection hash는 calibration 전, calibration 후, fresh validation 후 모두 다음 값으로 동일했다.

```text
352d98039a07593d9cb688f90c409836f1e1da1a944eb083550e2ddc6efb21d1
```

Fresh validation 16 conditions는 calibration conditions와 exact signature가 disjoint하며 freeze domain 밖으로 나가면 실행 전에 fail한다. Final validation 결과를 보고 speed, patch, Ice friction, Sand mechanics 또는 intended role을 바꾸지 않았다.

## 8. Fresh Concrete validation

Calibration에서 직접 사용하지 않은 width combinations로 16 runs를 실행했다.

| Target | Observed stable | Observed fall | Invalid | Pre-transition fall |
|---|---:|---:|---:|---:|
| Ice | 4 | 4 | 0 | 0 |
| Sand | 4 | 4 | 0 | 0 |

모든 run은 finite였고 actual target contact와 최소 500 ms normal prefix가 존재했다. 모든 observed stable run은 finite B patch를 완전히 통과했다.

## 9. Marble→Ice robustness

Concrete calibration에서 freeze한 Ice conditions를 그대로 두고 A profile만 Marble로 바꿨다. 7 runs의 observed outcome은 stable `3`, fall `4`, invalid `0`, pre-transition fall `0`이었다.

Concrete에서 stable이던 `0.25 m/s, 0.33/0.75 m` condition은 Marble에서 fall로 바뀌었고, Concrete에서 fall이던 `0.15 m/s, 0.35/0.75 m` condition은 Marble에서 stable로 바뀌었다. 이는 robustness diagnostic으로만 기록했고 B parameters를 재조정하지 않았다.

## 10. Marble→Sand robustness

Frozen Sand conditions에서 A만 Marble로 바꾼 8 runs는 stable `4`, fall `4`, invalid `0`, pre-transition fall `0`이었다. Stable group은 실제 mild deformation 뒤 patch를 통과했고, fall group은 moderate uneven deformation과 later fall을 재현했다.

## 11. Stable/fall outcome and timing table

시간은 ms다. `Sink/deform`은 frozen Sink onset이 있으면 그 시각, balanced stable이면 첫 실제 support deformation 시각이다. Intended label은 outcome을 강제하지 않는다. Per-run `t_boundary_expected`, last target contact, maximum deformation과 finite audit는 generated `results.json`에도 보존했다.

| Run | A→B | v | start/width | intended | contact/touchdown | Slip | Sink/deform | fall | outcome |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| `cv_ice_stable_01` | Concrete→Ice | 0.25 | .360/.710 | stable | 1221/1221 | 2190 | — | — | `VALID_STABLE` |
| `cv_ice_stable_02` | Concrete→Ice | 0.25 | .360/.720 | stable | 1221/1221 | 2190 | — | — | `VALID_STABLE` |
| `cv_ice_stable_03` | Concrete→Ice | 0.25 | .360/.730 | stable | 1221/1221 | 2190 | — | — | `VALID_STABLE` |
| `cv_ice_stable_04` | Concrete→Ice | 0.25 | .360/.740 | stable | 1221/1221 | 2190 | — | — | `VALID_STABLE` |
| `cv_ice_fall_01` | Concrete→Ice | 0.15 | .320/.670 | fall | 1803/1803 | 2934 | — | 3492 | `VALID_FALL` |
| `cv_ice_fall_02` | Concrete→Ice | 0.15 | .320/.690 | fall | 1803/1803 | 2934 | — | 3492 | `VALID_FALL` |
| `cv_ice_fall_03` | Concrete→Ice | 0.15 | .350/.710 | fall | 1803/1803 | 3088 | — | 5140 | `VALID_FALL` |
| `cv_ice_fall_04` | Concrete→Ice | 0.15 | .350/.730 | fall | 1803/1803 | 3088 | — | 5134 | `VALID_FALL` |
| `cv_sand_stable_01` | Concrete→Sand | 0.25 | .300/.725 | stable | 1221/1221 | — | 1221 | — | `VALID_STABLE` |
| `cv_sand_stable_02` | Concrete→Sand | 0.25 | .300/.735 | stable | 1221/1221 | — | 1221 | — | `VALID_STABLE` |
| `cv_sand_stable_03` | Concrete→Sand | 0.25 | .350/.725 | stable | 1508/2089 | — | 1508 | — | `VALID_STABLE` |
| `cv_sand_stable_04` | Concrete→Sand | 0.25 | .350/.735 | stable | 1508/2089 | — | 1508 | — | `VALID_STABLE` |
| `cv_sand_fall_01` | Concrete→Sand | 0.25 | .300/.670 | fall | 1221/1221 | — | 3046 | 4557 | `VALID_FALL` |
| `cv_sand_fall_02` | Concrete→Sand | 0.25 | .300/.690 | fall | 1221/1221 | — | 3039 | 4223 | `VALID_FALL` |
| `cv_sand_fall_03` | Concrete→Sand | 0.25 | .350/.670 | fall | 1221/1221 | — | 2476 | 4222 | `VALID_FALL` |
| `cv_sand_fall_04` | Concrete→Sand | 0.25 | .350/.690 | fall | 1221/1221 | — | 2476 | 4251 | `VALID_FALL` |
| `mr_ice_s_01` | Marble→Ice | 0.25 | .330/.750 | stable | 1221/1221 | 1909 | — | 2064 | `VALID_FALL` |
| `mr_ice_s_02` | Marble→Ice | 0.25 | .360/.700 | stable | 1221/1221 | 2187 | — | — | `VALID_STABLE` |
| `mr_ice_s_03` | Marble→Ice | 0.25 | .360/.750 | stable | 1221/1221 | 2187 | — | — | `VALID_STABLE` |
| `mr_ice_f_01` | Marble→Ice | 0.25 | .330/.700 | fall | 1221/1221 | 1909 | — | 2064 | `VALID_FALL` |
| `mr_ice_f_02` | Marble→Ice | 0.25 | .340/.750 | fall | 1221/1221 | 1909 | — | 3180 | `VALID_FALL` |
| `mr_ice_f_03` | Marble→Ice | 0.15 | .320/.650 | fall | 1803/1803 | 2820 | — | 3476 | `VALID_FALL` |
| `mr_ice_f_04` | Marble→Ice | 0.15 | .350/.750 | fall | 1803/1803 | 3085 | — | — | `VALID_STABLE` |
| `mr_sand_s_01` | Marble→Sand | 0.25 | .300/.720 | stable | 1221/1221 | — | 1221 | — | `VALID_STABLE` |
| `mr_sand_s_02` | Marble→Sand | 0.25 | .300/.740 | stable | 1221/1221 | — | 1221 | — | `VALID_STABLE` |
| `mr_sand_s_03` | Marble→Sand | 0.25 | .350/.720 | stable | 1510/2090 | — | 1510 | — | `VALID_STABLE` |
| `mr_sand_s_04` | Marble→Sand | 0.25 | .350/.740 | stable | 1510/2090 | — | 1510 | — | `VALID_STABLE` |
| `mr_sand_f_01` | Marble→Sand | 0.25 | .300/.650 | fall | 1221/1221 | — | 3046 | 4229 | `VALID_FALL` |
| `mr_sand_f_02` | Marble→Sand | 0.25 | .300/.750 | fall | 1221/1221 | — | 2475 | 5173 | `VALID_FALL` |
| `mr_sand_f_03` | Marble→Sand | 0.25 | .350/.650 | fall | 1221/1221 | — | 2476 | 4585 | `VALID_FALL` |
| `mr_sand_f_04` | Marble→Sand | 0.25 | .350/.750 | fall | 1221/1221 | — | 2475 | 5217 | `VALID_FALL` |

## 12. Pre-transition fall audit

Exploration/calibration에는 `i_s020_p025_w065`, `i_s020_p027_w075`, `i_s025_p036_w080` 세 invalid pre-transition points가 있었다. 이들은 operating-point selection과 acceptance에서 제외했다. Freeze 이후 fresh Concrete `0/16`, Marble `0/15`로 pre-transition fall은 0이었다. Accepted validation/robustness run은 모두 actual target contact가 fall보다 앞섰다.

## 13. Physical Slip/Sink diagnostics

- Concrete→Ice stable: Slip `4/4`; fall `4/4`. Stable의 Slip onset은 2190 ms였지만 fall은 없었다.
- Marble→Ice stable/fall 전체에서도 observed outcome과 무관하게 target-linked Slip이 발생했다.
- Concrete→Sand stable: first deformation `4/4`, maximum `20.139–20.167 mm`, frozen uneven Sink onset 없음.
- Concrete→Sand fall: first deformation `4/4`, frozen Sink onset `4/4`, maximum `40.086–40.175 mm`.
- Marble→Sand stable/fall도 각각 약 20 mm/40 mm response를 재현했다.

Slip/Sink는 secondary physics diagnostic이며 walking Stability label이나 latency anchor로 사용하지 않았다.

## 14. Viewer sanity

Official MuJoCo passive viewer의 state-only render mirror로 다음 six representative runs를 8초 끝까지 replay했다.

1. `cv_ice_stable_01`
2. `cv_ice_fall_01`
3. `cv_sand_stable_01`
4. `cv_sand_fall_01`
5. `mr_ice_s_01`
6. `mr_sand_s_01`

모두 `terminated_by_viewer=false`였고 headless outcome과 같았다. A terrain의 정상 prefix, 실제 spatial boundary, B contact, full-width Ice band, Sand tile coverage/deformation, later fall/non-fall을 확인했다. Premature collision, geometry teleport, hidden step/hole은 관찰되지 않았다. Viewer는 canonical physics의 별도 render copy만 받았고 physics mutation은 없었다.

## 15. Limitations

- Frozen policy와 fixed initialization의 deterministic MuJoCo engineering scenario다. Real Ice/Sand material calibration이 아니다.
- Ice stable region은 patch start/width에 민감하다. 이번 validation domain 밖으로 일반화하지 않는다.
- Marble에서 일부 intended role의 outcome이 바뀌었다. Robustness minimum은 통과했지만 broad terrain/gait distribution evidence가 아니다.
- Sand는 passive vertical-support proxy이고 continuum soil model이 아니다.
- Terrain state는 계속 `ORACLE_PROXY`; `TERRAIN_RUNTIME_MODEL_PENDING`이다.
- Stability oracle, pelvis-IMU rule과 GRU를 평가하지 않았다.

## 16. Verdict

Primary verdict: `TRANSITION_SCENARIOS_CALIBRATED`.

Prefix parity, geometry/contact, fresh Concrete 4-way outcome coverage, zero validation pre-transition fall, actual target contact, freeze immutability와 Marble stable/fall robustness가 사전 acceptance를 모두 통과했다. Fusion truth table regression도 PASS했고 logic은 변경하지 않았다.

Generated artifact는 Gitignored `artifacts/runs/20260828_transition_scenario_calibration/results.json`이며 viewer confirmation 반영 후 SHA-256은 `61610e42576b6b1bf96158c8e234ae4768d4a212850d92046f81c46e7c994488`다.

## 17. Next recommendation

다음 승인 가능한 milestone은 clean transition scenarios에서 exact walking-instability ground truth를 새로 사전 선언하고 검증하는 것이다. 이번 작업은 그 다음 단계를 자동 시작하지 않는다. 특히 Stability ground-truth redesign, IMU rule retuning, Stability AI, Terrain model migration, recovery controller, Full Dataset과 E84 작업은 수행하지 않았다.
