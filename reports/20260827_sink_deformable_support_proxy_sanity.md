# Sink Deformable-Support Proxy Sanity

Milestone: `SINK_DEFORMABLE_SUPPORT_PROXY_SANITY`

## 1. Previous oracle failures

이 실험은 과거 결과를 수정하거나 재해석하지 않는다. Same-height static compliance에서 확인한 결과는 penetration-spread가 benign `0/14` false firing, primary uneven `2/12` detection이었고 support-loss가 benign `12/14` false firing, primary uneven `2/12` detection이었다. Penetration-spread는 크게 내려간 quadrant가 unload되면 valid loaded set에서 빠져 가장 중요한 순간의 spread가 사라지는 구조적 한계가 있었다. 기존 outcome-based Sink contract, Pilot raw annotation, Frozen Slip `50 mm + 3 ms`도 그대로 보존한다.

## 2. Why support displacement proxy

이번 proxy는 contact penetration 대신 support body의 실제 vertical joint displacement를 읽는다. Cell이 contact를 잃어도 joint position은 남으므로 unload된 cell을 metric에서 제거하지 않는다. 따라서 평평하게 함께 내려가는 support와 공간적으로 비균일하게 내려가는 support를 직접 구분할 수 있다.

## 3. Engineering-proxy limitation

이 구현은 `passive_deformable_support_engineering_proxy`다. Granular soil, 실제 모래 입자 거동, measured soil mechanics 또는 real-world Sink depth calibration을 주장하지 않는다. 목표는 MuJoCo 안에서 load-driven support collapse와 그 runtime observability를 제한된 조건으로 연구하는 것이다.

## 4. Passive mechanical implementation

Canonical `scene_sink.xml`을 확장했다. Balanced support는 side마다 하나의 vertical slide body가 네 contact geom을 함께 운반한다. Uneven support는 side마다 네 독립 body/cell과 네 vertical slide joint를 사용한다. 모든 joint는 z축 translation 한 자유도만 가지며 actuator가 없다. Nominal joint position과 spring reference는 0이고 surface top은 이웃 ground와 같은 `z=0`이다. Body gravity compensation은 1.0이라 무부하 중력 sag가 없고, 발/contact load가 spring/damper를 수동으로 압축한다. Time trigger, label trigger, qpos rewrite, teleport는 없다.

Tile mass는 모두 `5 kg`이고 hard-contact response와 sand friction을 사용한다. Robot의 29 actuator joint address를 명시적으로 읽도록 controller contract도 고정하여 support DOF가 robot observation/action에 섞이지 않게 했다.

## 5. Mechanical stabilization

Robot matrix를 보기 전 `100/250/400 N`, 각 1초 load와 1초 unload bench에서 다음 값을 확인한 뒤 parameter를 freeze했다. Robot 결과는 parameter 선택에 사용하지 않았고 이후 tuning하지 않았다.

| Profile | Travel (mm) | Stiffness (N/m) | Damping (N·s/m) | Steady displacement @ 100/250/400 N (mm) | Peak @ 400 N (mm) | Residual after unload (mm) |
|---|---:|---:|---:|---|---:|---:|
| reference | 4 | 50,000 | 1,000 | 2.000 / 4.000 / 4.002 | 4.169 | < 0.000001 |
| mild | 20 | 12,000 | 490 | 8.333 / 20.000 / 20.001 | 20.285 | < 0.000001 |
| moderate | 40 | 7,000 | 374 | 14.286 / 35.714 / 40.001 | 40.354 | < 0.000001 |
| severe | 65 | 4,500 | 300 | 22.222 / 55.556 / 65.001 | 65.304 | < 0.000001 |

모든 profile의 initial/unloaded displacement는 정확히 0이었고 load response는 severity에 따라 monotonic했다. Peak limit overshoot 최대값은 0.354 mm였으며 unload 후 residual 최대값은 약 `2.45e-11 mm`였다. Oscillatory divergence는 없었다.

## 6. Balanced vs uneven topology

Cell 순서는 `[entry_medial, entry_lateral, exit_medial, exit_lateral]`이다.

- `balanced_deformable`: 네 geom이 한 joint displacement를 공유하여 plate가 level을 유지한다.
- `medial_deformable`: 두 medial cell에 requested severity, 나머지에 4 mm reference response를 적용한다.
- `lateral_deformable`: 두 lateral cell에 requested severity, 나머지에 reference response를 적용한다.
- `localized_deformable`: entry-medial 한 cell에 requested severity, 나머지에 reference response를 적용한다.

Dynamic tile끼리와 surrounding static patch 사이의 artificial edge collision은 분리하고 robot geoms와의 physical contact는 허용했다. Oracle patch episode는 그중 named sole contact만 사용한다. Left/right와 medial/lateral mapping은 model contract test와 Viewer에서 확인했다.

## 7. Exact surface-displacement oracle

각 cell의 positive-downward displacement를 `d=[d0,d1,d2,d3]`라 할 때 primary metric은 다음과 같다.

```text
support_surface_spread_m = max(d) - min(d)
```

Rigid region은 0이다. 분석에는 per-cell displacement/velocity/contact, mean/max displacement, maximum downward velocity, residual displacement와 recovery time도 남긴다. Old penetration-spread, support-loss, old t2와 fall은 secondary comparison/outcome일 뿐 새 oracle input이 아니다.

## 8. Threshold / persistence

Predeclared criterion은 `support_surface_spread_m >= 0.010 m`가 1 kHz에서 20 consecutive samples 지속되는 것이다. Threshold와 persistence는 robot 결과를 보기 전에 config에 고정했으며 sweep하지 않았다. 이 10 mm는 universal real-world threshold가 아니라 이번 bounded sanity의 engineering criterion이다.

## 9. Causality

`d0`는 해당 foot이 deformable patch와 first physical contact한 시각이고 `s1`은 sustained spread의 first active sample이다. Evaluation은 같은 foot의 current physical-contact episode에서 patch contact를 이미 보았고, foot이 force-loaded이며, pre-censor일 때만 활성화된다. 개별 cell unload는 displacement vector를 mask하지 않는다. Foot episode 종료/변경 또는 censor에서 persistence를 reset한다.

Oracle은 현재/과거 support joint state와 contact episode만 사용한다. Future t2/fall/recovery, pelvis tilt, IMU, FSR, robot joint state와 terrain class rule을 사용하지 않는다. Synthetic prefix/suffix regression에서 future suffix를 바꾸어도 이미 결정된 s1이 변하지 않았고, contact-episode reset과 20 ms persistence도 통과했다.

## 10. Sanity matrix

미리 선언한 32개 deterministic 8초 run만 실행했다.

| Group | Runs | Factors |
|---|---:|---|
| rigid benign | 6 | Concrete/Marble/Uniform Sand × 0.15/0.25 m/s |
| balanced deformable benign | 8 | mild/moderate × left/right × 0.15/0.25 m/s |
| primary uneven moderate | 12 | medial/lateral/localized × left/right × 0.15/0.25 m/s |
| outcome-diversity subset | 6 | predeclared mild localized/medial and severe lateral |

각 run은 8,000 sensor samples, drop 0이었다. 전체 dynamic run에서 first patch contact 전 unloaded drift 최대값은 0 m였다.

## 11. Benign results

Rigid `0/6`, balanced deformable `0/8`, 합계 `0/14`에서 physical Sink firing이 없었다. Rigid spread는 0이고 balanced는 네 geom이 같은 joint를 공유해 최대 spread가 정확히 0이었다. Balanced plate의 최대 downward displacement는 mild 약 20.19 mm, moderate 약 40.17 mm였으나 level displacement이므로 Sink가 아니다. Balanced moderate 2개 run의 later fall도 label을 바꾸지 않았다.

## 12. Primary uneven results

Primary moderate는 `11/12` detected였다. 단위는 시간 ms, spread/displacement mm, velocity m/s다. `old t2 offset`과 `fall offset`은 s1 기준이며 `—`는 관측되지 않음을 뜻한다. Negative old-t2 offset은 historical outcome diagnostic이 새 s1보다 먼저 발생했음을 그대로 보존한 값이다.

| Run | d0 | s1 | d0→s1 | s1→old t2 | s1→fall | Max spread | Max displacement | Spread @ s1 | Max down velocity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| medial left 0.15 | 1803 | 3695 | 1892 | -223 | — | 27.394 | 27.394 | 15.144 | 0.310 |
| medial left 0.25 | 1221 | 1875 | 654 | 242 | — | 32.383 | 32.383 | 16.553 | 0.354 |
| medial right 0.15 | 2096 | 2783 | 687 | 1367 | — | 40.133 | 40.133 | 12.119 | 0.338 |
| medial right 0.25 | 1508 | 2223 | 715 | 753 | — | 40.131 | 40.131 | 10.346 | 0.446 |
| lateral left 0.15 | 1803 | 3066 | 1263 | 802 | — | 40.150 | 40.150 | 12.361 | 0.460 |
| lateral left 0.25 | 1221 | 2476 | 1255 | 824 | 3091 | 40.172 | 40.172 | 10.874 | 0.450 |
| lateral right 0.15 | 2096 | — | — | — | — | 6.408 | 6.408 | — | 0.159 |
| lateral right 0.25 | 1508 | 2174 | 666 | 833 | — | 40.153 | 40.153 | 10.509 | 0.474 |
| localized left 0.15 | 1803 | 3695 | 1892 | -223 | — | 27.394 | 27.394 | 15.144 | 0.310 |
| localized left 0.25 | 1221 | 1875 | 654 | 242 | — | 31.358 | 31.358 | 16.553 | 0.354 |
| localized right 0.15 | 2096 | 2783 | 687 | 1367 | — | 40.133 | 40.133 | 12.119 | 0.338 |
| localized right 0.25 | 1508 | 2223 | 715 | — | — | 31.351 | 31.351 | 10.346 | 0.307 |

유일한 miss인 lateral right 0.15 m/s는 최대 spread 6.408 mm로 predeclared 10 mm criterion에 미달했다. 결과 뒤 tuning하지 않았다.

Positive run의 s1 cell displacement는 다음과 같다. 순서는 section 6의 canonical cell order이고 단위는 mm다.

| Run | Cell displacement @ s1 |
|---|---|
| medial left 0.15 | `[15.144, 3.388, 0, 0]` |
| medial left 0.25 | `[16.553, 2.422, 0, 0]` |
| medial right 0.15 | `[12.119, 0, 0, 0]` |
| medial right 0.25 | `[10.346, 4.001, 0, 0]` |
| lateral left 0.15 | `[4.001, 12.361, 0, 0]` |
| lateral left 0.25 | `[4.001, 10.874, 0, 0]` |
| lateral right 0.25 | `[4.001, 10.509, 0, 0]` |
| localized left 0.15 | `[15.144, 3.388, 0, 0]` |
| localized left 0.25 | `[16.553, 2.422, 0, 0]` |
| localized right 0.15 | `[12.119, 0, 0, 0]` |
| localized right 0.25 | `[10.346, 4.001, 0, 0]` |

## 13. Side/pattern/speed coverage

| Stratum | Detection |
|---|---:|
| left | 6/6 |
| right | 5/6 |
| medial | 4/4 |
| lateral | 3/4 |
| localized | 4/4 |
| 0.15 m/s | 5/6 |
| 0.25 m/s | 6/6 |

Predeclared minimum인 total 9/12, side별 4/6, pattern별 2/4와 both-speed coverage를 모두 통과했다.

## 14. Timing

Detected primary 11개에서 d0→s1은 `654–1892 ms`, median `715 ms`였다. 이 값은 runtime detector latency가 아니라 load-driven mechanics가 10 mm/20 ms physical ground-truth criterion에 도달한 시각이다. 20 ms detector 가능성을 증명하지 않으며 그대로 후속 observability study의 alignment clock으로 사용한다.

## 15. Fall/non-fall outcome

Primary detected 11개 중 later fall은 `1`, 8초 non-fall은 `10`, measured mechanical recovery가 관측된 run은 `7`이었다. Fall positive는 lateral-left 0.25 m/s로 s1 뒤 3091 ms에 넘어졌다. Predeclared outcome subset은 4/6 detected였고 네 positive 모두 non-fall/recovery였다.

| Outcome-subset run | d0 (ms) | s1 (ms) | d0→s1 (ms) | Max spread (mm) | Recovery after last patch contact (ms) | Fall |
|---|---:|---:|---:|---:|---:|---|
| mild localized left 0.15 | 1803 | — | — | 8.093 | — | no |
| mild localized right 0.15 | 2096 | 2956 | 860 | 20.168 | 14 | no |
| mild medial left 0.25 | 1221 | 1879 | 658 | 20.037 | 24 | no |
| mild medial right 0.25 | 1508 | 2790 | 1282 | 20.154 | 73 | no |
| severe lateral left 0.15 | 1803 | 3060 | 1257 | 65.575 | 193 | no |
| severe lateral right 0.15 | 2096 | — | — | 7.039 | 1 | no |

따라서 detected fall과 detected non-fall/recovery가 모두 존재한다. Fall/recovery는 s1 판정에 사용하지 않았다.

## 16. Sensor descriptive audit

Physical gate 통과 후 representative uneven 세 개와 time-aligned matched balanced control을 s1-relative `-20/0/+20/+50/+100 ms`에서 raw FSR8, pelvis IMU6, Fusion14로만 비교했다. 다음은 s1 시각의 raw 값이다. Fusion14는 canonical channel contract대로 표의 IMU6 뒤 FSR8을 이어 붙인 값이다.

| Representative / cohort | FSR8 @ s1 (N) | IMU6 @ s1 |
|---|---|---|
| medial left 0.15 / uneven | `[168.12, 0, 0, 179.97, 0, 9.74, 0, 0]` | `[-0.377, -0.996, 9.987, -0.127, -0.120, 0.108]` |
| medial left 0.15 / balanced | `[187.20, 0, 191.27, 0, 0, 0, 0, 0]` | `[-1.329, -1.975, 11.216, 0.403, -0.208, 0.119]` |
| lateral left 0.25 / uneven | `[66.17, 44.40, 0, 289.18, 0, 0, 0, 0]` | `[0.377, -0.825, 11.438, 0.158, -0.175, 0.214]` |
| lateral left 0.25 / balanced | `[187.29, 120.06, 151.60, 97.89, 0, 0, 0, 0]` | `[-3.167, -4.375, 16.474, 1.252, -0.415, 0.201]` |
| localized right 0.15 / uneven | `[0, 0, 0, 0, 66.60, 37.90, 123.55, 107.68]` | `[-0.580, 1.514, 9.048, 0.162, -0.182, 0.030]` |
| localized right 0.15 / balanced | `[0, 48.03, 0, 0, 48.23, 57.82, 105.31, 115.36]` | `[-0.878, 1.975, 10.755, 0.011, -0.209, 0.072]` |

Raw trajectories는 s1 주변에서 변하지만 matched balanced에도 gait-phase 변화가 크다. 이는 candidate signal이 runtime에 존재한다는 작은 descriptive evidence일 뿐 clean separation, feature value 또는 classifier performance가 아니다. Classifier training과 handcrafted feature mining은 하지 않았다. Support joint label과 FSR force는 같은 dynamics에 결합돼 있으나 exact FSR threshold를 label에 쓰지 않아 이전 oracle보다 circularity가 줄었다. 실제 hardware validation은 여전히 필요하다.

## 17. Previous oracle comparison

| Oracle/evidence | Benign firing | Primary uneven detection | Interpretation |
|---|---:|---:|---|
| Historical penetration spread, old same-height physics | 0/14 | 2/12 | low coverage; 과거 결과 유지 |
| Historical support loss, old same-height physics | 12/14 | 2/12 | high benign false firing; 과거 결과 유지 |
| New displacement spread, deformable-support physics | 0/14 | 11/12 | physical gate PASS |

새 12-run physics에 historical diagnostics를 replay하면 penetration-spread는 0/12, support-loss는 8/12였다. 이 replay는 secondary comparison일 뿐 과거 experiment나 frozen Pilot label을 덮어쓰지 않는다.

## 18. Viewer review

Viewer physics-copy 경로로 rigid reference, balanced moderate, left medial moderate, right lateral moderate와 localized right moderate를 확인했다. Rigid는 displacement/spread 0, balanced는 plate displacement가 생겨도 spread 0, medial/localized는 실제 non-level downward motion과 s1을 보였다. Right-lateral representative miss도 4.16 mm 이하의 작은 non-level response를 그대로 보였다. Initial top parity와 sag 0, z-only joint, passive contact 이후 motion, no teleport/initial hole/collision explosion, side/pattern mapping을 scene/runtime contract와 함께 확인했다. Viewer는 parameter tuning에 사용하지 않았다.

## 19. Limitations

- Deterministic fixed policy/phase의 32-run bounded sanity이며 distribution coverage가 아니다.
- Slide-cell support는 continuum soil이나 granular coupling을 표현하지 않는다.
- Threshold와 mechanical parameters는 engineering values이며 real-world calibration이 없다.
- Right-lateral 0.15 m/s miss는 phase/load sensitivity가 남음을 보여 준다.
- Sensor audit는 세 representative pair의 raw descriptive view뿐이며 detector latency나 separability를 검증하지 않았다.
- 기존 Pilot dataset은 새 proxy state를 저장하지 않았고 소급 relabel할 수 없다.

## 20. Verdict

Physical oracle gate: PASS (`0/14` benign, `11/12` primary, left `6/6`, right `5/6`, pattern `4/4`, `3/4`, `4/4`, both speeds, causality/mechanics PASS).

Outcome diversity gate: PASS (detected fall `1`, detected non-fall `10`, detected recovery evidence present).

`SINK_DEFORMABLE_SUPPORT_PROXY_SUPPORTED_FOR_OBSERVABILITY_STUDY`

## 21. Next recommendation

새 surface-displacement s1을 future dataset/schema provenance에 명시한 뒤, matched balanced와 uneven support의 causal raw IMU6/FSR8/Fusion14 observability를 별도 승인된 bounded study로 검증한다. 기존 Pilot을 relabel하지 않고, Full Dataset 생성·classifier training·E84 작업은 자동으로 시작하지 않는다.
