# Sink Support-Loss Oracle Sanity

## 1. Previous spread failure

직전 `SINK_PHYSICAL_HAZARD_REDEFINITION`의 loaded-quadrant penetration spread는 benign 0/14였지만 intended uneven 2/12만 검출했다. Severe local collapse 때 unload된 quadrant가 valid set에서 빠져 metric 자체가 unavailable 또는 작아지는 구조가 원인이었다. 이번 sanity는 terrain을 바꾸지 않고, 같은 exact quadrant에서 이전에 확보한 support가 사라지는 현상을 두 번째 causal candidate로 검증했다.

결론부터 말하면 candidate는 causal하지만 benign control 12/14에서 정상 gait redistribution을 Sink로 오검출했고 intended uneven은 2/12만 검출했다. 기존 frozen SINK contract, [`docs/dataset.md`](../docs/dataset.md), README는 변경하지 않는다.

## 2. Support-loss physical hypothesis

Candidate는 한 foot contact episode의 안정적인 sole support map을 먼저 고정한 뒤, 그 map 중 현재 남아 있는 quadrant 수를 센다. Primary quantity는 exact sole-ground contact 존재와 fixed 2.5 N local validity cutoff만 사용한다. Terrain/scenario name, virtual FSR, IMU, future posture degradation, recovery, fall과 old t2는 oracle input이 아니다.

같은 `balanced_soft`, `medial_soft`, `lateral_soft`, `localized_soft` topology/profile과 canonical `scene_sink.xml`을 재사용했다. 모든 enabled surface top은 `z=0`이고 step, hole, seam 또는 새 scene은 추가하지 않았다.

## 3. Baseline support-map definition

Foot-local quadrant order는 virtual FSR과 같은 `front_left`, `front_right`, `rear_left`, `rear_right`다. `contact_present AND quadrant_normal_load >= 2.5 N`이면 supported다. 기존 foot load hysteresis 5 N on/2.5 N off와 touchdown transient 10 ms를 유지한다.

Touchdown age가 10 ms 이상이고 foot loaded/pre-fall state가 연속으로 유효한 첫 20개 1 kHz sample을 baseline window로 사용한다. Quadrant가 이 window의 50% 이상, 즉 최소 10 sample에서 supported이면 baseline mask에 포함한다. Presence aggregation은 force median보다 support-region 존재 가설을 직접 표현하며, 결과를 보기 전에 config에 고정했다. Baseline supported quadrant가 2개 미만이면 해당 loaded-contact session은 평가하지 않는다.

## 4. Validity and toe-off handling

Oracle은 같은 physical contact episode, loaded foot, pre-fall sample에서만 평가한다. Current total foot load가 baseline median total load의 30% 이상이어야 한다. Foot가 loaded contact를 벗어나거나 contact episode가 바뀌거나 fall/censor가 시작되면 baseline과 persistence를 reset하며, 같은 physical episode에서 다시 loaded가 되더라도 새 20 ms baseline을 만든다.

이 fixed gate는 정상 toe-off를 충분히 제거하지 못했다. 예를 들어 balanced-right moderate false onset에서 current/baseline total load는 `82.02/150.84 N`으로 gate를 여유 있게 통과했다. 정상 heel-to-toe support redistribution도 20 ms 이상 baseline quadrant 절반을 소실할 수 있었다.

## 5. Support-loss metric

```text
current_retained_count
  = count(baseline_support_mask AND current_quadrant_supported)

support_retention_ratio
  = current_retained_count / baseline_supported_count

support_loss_ratio
  = 1 - support_retention_ratio
```

Secondary `weighted_support_loss`는 baseline supported quadrant의 median loads로 정규화한 retained load weight를 1에서 뺀 값이며 diagnostic-only다. Exact baseline mask/count, retained count, binary loss, weighted loss, validity, active/onset arrays를 simulator-only diagnostics에 추가했다. 기존 `support_penetration_spread_m`은 삭제하거나 변경하지 않았다.

## 6. Fixed threshold and persistence

Predeclared primary candidate는 `support_loss_ratio >= 0.5`와 20 consecutive 1 kHz samples다. Threshold, 20 ms persistence, 2.5 N quadrant cutoff, 30% total-load gate를 결과 뒤에 sweep하거나 조정하지 않았다. Frozen Slip도 touchdown-anchor tangential drift 50 mm와 persistence 3 ms 그대로다.

## 7. Benign results

Benign firing은 `12/14`였다. Concrete 0.15와 marble 0.15만 no-fire였고, 나머지 concrete/marble 0.25, uniform sand 두 speed와 balanced mild/moderate 8개가 모두 적어도 한 foot에서 firing했다. 여러 0.25 m/s control은 patch t0 이전 initial gait의 right foot에서 619–620 ms에 firing했다. 따라서 provisional benign maximum 0을 즉시 위반하며 protocol상 `SUPPORT_LOSS_ORACLE_NOT_CLEAN`이다.

모든 26 run은 8,000 IMU6/FSR8 samples, 1 kHz timestamp, drop 0이었다. 표의 시간은 1-based simulation time ms다. Transition run의 max loss/weighted loss/spread는 scenario affected foot 범위이고, no-side control은 양발 범위다. Benign u1은 false firing을 숨기지 않도록 첫 any-foot onset이며 괄호로 scenario side와 다른 u1 side를 표시한다. Uneven u1은 affected foot의 해당 contact episode가 이미 soft patch와 접촉한 경우만 인정한다.

| Scenario | Role | Speed | Side/pattern | t0 | b0 | Base q | u1 | t0→b0 | t0→u1 | b0→u1 | Max loss | Max weighted | Old spread (mm) | Old t2 | Fall | Classification |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| concrete_s015 | benign | 0.15 | — | — | 67 | 4 | — | — | — | — | 0.750 | 0.933 | 2.578 | — | — | BENIGN_SUPPORT |
| concrete_s025 | benign | 0.25 | — | — | 267 | 4 | 620 | — | — | 353 | 0.750 | 0.960 | 2.744 | — | — | UNEVEN_SUPPORT_SINK |
| marble_s015 | benign | 0.15 | — | — | 67 | 4 | — | — | — | — | 0.750 | 0.935 | 2.506 | — | — | BENIGN_SUPPORT |
| marble_s025 | benign | 0.25 | — | — | 269 | 4 | 619 | — | — | 350 | 1.000 | 1.000 | 2.776 | — | — | UNEVEN_SUPPORT_SINK |
| uniform_sand_s015 | benign | 0.15 | — | — | 168 | 2 | 620 | — | — | 452 | 1.000 | 1.000 | 10.530 | — | — | UNEVEN_SUPPORT_SINK |
| uniform_sand_s025 | benign | 0.25 | — | — | 1,495 | 4 | 1,820 | — | — | 325 | 0.750 | 0.876 | 9.922 | — | — | UNEVEN_SUPPORT_SINK |
| balanced_left_mild_s015 | benign | 0.15 | left/balanced (u1 right) | 1,803 | 2,116 | 4 | 2,420 | 313 | 617 | 304 | 0.750 | 0.946 | 11.010 | — | — | UNEVEN_SUPPORT_SINK |
| balanced_left_mild_s025 | benign | 0.25 | left/balanced (u1 right) | 1,221 | 267 | 4 | 620 | -954 | -601 | 353 | 1.000 | 1.000 | 16.851 | 6,085 | — | UNEVEN_SUPPORT_SINK |
| balanced_right_mild_s015 | benign | 0.15 | right/balanced (u1 left) | 2,096 | 3,025 | 4 | 3,300 | 929 | 1,204 | 275 | 1.000 | 1.000 | 14.093 | — | — | UNEVEN_SUPPORT_SINK |
| balanced_right_mild_s025 | benign | 0.25 | right/balanced | 1,508 | 267 | 4 | 620 | -1,241 | -888 | 353 | 1.000 | 1.000 | 15.573 | — | — | UNEVEN_SUPPORT_SINK |
| balanced_left_moderate_s015 | benign | 0.15 | left/balanced (u1 right) | 1,803 | 2,113 | 4 | 2,420 | 310 | 617 | 307 | 0.750 | 0.948 | 13.619 | — | — | UNEVEN_SUPPORT_SINK |
| balanced_left_moderate_s025 | benign | 0.25 | left/balanced (u1 right) | 1,221 | 267 | 4 | 620 | -954 | -601 | 353 | 1.000 | 1.000 | 22.655 | — | — | UNEVEN_SUPPORT_SINK |
| balanced_right_moderate_s015 | benign | 0.15 | right/balanced | 2,096 | 5,665 | 3 | 6,060 | 3,569 | 3,964 | 395 | 1.000 | 1.000 | 18.231 | — | — | UNEVEN_SUPPORT_SINK |
| balanced_right_moderate_s025 | benign | 0.25 | right/balanced | 1,508 | 267 | 4 | 620 | -1,241 | -888 | 353 | 1.000 | 1.000 | 19.261 | 5,246 | — | UNEVEN_SUPPORT_SINK |
| medial_left_s015 | uneven | 0.15 | left/medial | 1,803 | 3,006 | 4 | 3,340 | 1,203 | 1,537 | 334 | 1.000 | 1.000 | 23.570 | 2,920 | 6,976 | UNEVEN_SUPPORT_SINK |
| medial_left_s025 | uneven | 0.25 | left/medial | 1,221 | 1,250 | 4 | — | 29 | — | — | 0.750 | 0.813 | 18.595 | 1,657 | 2,551 | BENIGN_SUPPORT |
| medial_right_s015 | uneven | 0.15 | right/medial | 2,096 | 2,120 | 4 | — | 24 | — | — | 0.750 | 0.786 | 24.145 | — | 2,897 | BENIGN_SUPPORT |
| medial_right_s025 | uneven | 0.25 | right/medial | 1,508 | 1,528 | 3 | — | 20 | — | — | 0.667 | 0.814 | 19.146 | — | 2,329 | BENIGN_SUPPORT |
| lateral_left_s015 | uneven | 0.15 | left/lateral | 1,803 | 1,832 | 4 | — | 29 | — | — | 0.750 | 0.953 | 21.044 | 2,883 | 3,144 | BENIGN_SUPPORT |
| lateral_left_s025 | uneven | 0.25 | left/lateral | 1,221 | 1,250 | 4 | — | 29 | — | — | 0.750 | 0.830 | 14.185 | 1,708 | 2,012 | BENIGN_SUPPORT |
| lateral_right_s015 | uneven | 0.15 | right/lateral | 2,096 | 2,120 | 4 | — | 24 | — | — | 0.750 | 0.851 | 21.736 | — | 2,846 | BENIGN_SUPPORT |
| lateral_right_s025 | uneven | 0.25 | right/lateral | 1,508 | 1,528 | 3 | — | 20 | — | — | 0.333 | 0.770 | 19.807 | — | 2,249 | BENIGN_SUPPORT |
| localized_left_s015 | uneven | 0.15 | left/localized | 1,803 | 3,006 | 4 | 3,340 | 1,203 | 1,537 | 334 | 1.000 | 1.000 | 23.570 | 2,920 | 6,976 | UNEVEN_SUPPORT_SINK |
| localized_left_s025 | uneven | 0.25 | left/localized | 1,221 | 1,250 | 4 | — | 29 | — | — | 0.750 | 0.813 | 18.595 | 1,657 | 2,551 | BENIGN_SUPPORT |
| localized_right_s015 | uneven | 0.15 | right/localized | 2,096 | 2,120 | 4 | — | 24 | — | — | 0.750 | 0.786 | 24.145 | — | 2,897 | BENIGN_SUPPORT |
| localized_right_s025 | uneven | 0.25 | right/localized | 1,508 | 1,528 | 3 | — | 20 | — | — | 0.667 | 0.814 | 19.146 | — | 2,329 | BENIGN_SUPPORT |

## 8. Uneven results

Affected-foot, patch-linked intended uneven detection은 `2/12`였다. 두 event는 `medial_left_s015`와 dynamically identical entry prefix를 공유하는 `localized_left_s015`다. 둘 다 baseline 4 quadrants에서 current retained 1 quadrant까지 감소해 max binary/weighted loss 1.0을 기록했다.

Timing은 두 trace 모두 `t0→b0 1,203 ms`, `b0→u1 334 ms`, `t0→u1 1,537 ms`여서 각 range와 median도 같은 값이다. 이 긴 delay와 동일 prefix 때문에 independent physical coverage로 해석할 수 없다.

## 9. Side, pattern, and speed coverage

- Left: `2/6`; right: `0/6`.
- Medial: `1/4`; lateral: `0/4`; localized: `1/4`.
- 0.15 m/s: `2/6`; 0.25 m/s: `0/6`.
- Baseline count <2인 uneven run은 없었으므로 coverage failure를 baseline absence로 설명할 수 없다.

Provisional requirements인 uneven >=9/12, each side >=4/6, each family >=2/4와 both-speed coverage를 모두 실패했다.

## 10. Outcome independence

두 detected event 모두 later fall했고 detected+no-fall은 0이다. u1은 3,340 ms, old t2는 2,920 ms, fall은 6,976 ms였다. Old t2가 이 trace에서 u1보다 먼저 발생했지만 oracle code는 pelvis state나 old t2를 읽지 않으며, future suffix를 변경해도 그 이전 baseline/loss/onset이 동일한 causal test를 통과했다. 전체 uneven 12개는 모두 나중에 fall했지만 fall이 missed event를 retroactively positive로 만들지 않았다.

## 11. Spread versus support loss

Old spread는 이전과 같이 `2/12`를 검출했고 support loss도 `2/12`였다. Case 관계는 다음과 같다.

- Spread missed, support loss detected: 2 (`medial_left_s015`, `localized_left_s015`).
- Both detected: 0.
- Spread detected, support loss missed: 2 (`medial_right_s015`, `localized_right_s015`).
- Neither detected: 8.

따라서 support-region dropout을 값으로 표현하기는 했지만 coverage를 개선하지 않았고, 동시에 catastrophic benign false-positive 문제를 만들었다. Weighted loss도 benign에서 최대 1.0까지 올라 primary failure를 구제하지 못하며 post-hoc oracle로 채택하지 않는다.

## 12. Missed-event reasons

Ten affected-foot misses를 사전 category로 분류했다.

| Reason | Count | Interpretation |
|---|---:|---|
| insufficient persistence | 6 | Valid >=50% loss의 longest streak가 1–6 ms로 20 ms 미만 |
| total-load validity gated out | 3 | >=50% loss는 있었지만 모든 해당 sample이 baseline total load 30% 미만 |
| no >=50% support loss | 1 | `lateral_right_s025` max binary loss 0.333 |
| baseline had <2 quadrants | 0 | 모든 run에 evaluable baseline 존재 |
| immediate censor | 0 | 별도 primary cause로 분류된 run 없음 |
| normal unloading only/unlinked episode | 0 | Miss의 primary category로 사용하지 않음 |
| topology unchanged despite deep penetration | 0 | 별도 primary category로 사용하지 않음 |

Miss count와 별개로 benign false firing의 지배적 원인은 meaningful total load가 남은 정상 heel-to-toe unloading이다. Fixed 30% gate를 결과 뒤에 올리거나 persistence를 늘리는 것은 금지된 tuning이므로 시행하지 않았다.

## 13. FSR circularity limitation

Acceptance를 실패했으므로 u1-aligned 0/+20/+50/+100 ms FSR8/Fusion14 descriptive audit은 수행하지 않았다. Primary는 raw FSR threshold가 아니라 exact contact presence와 fixed load validity에서 만든 binary support topology이지만, exact quadrant normal load와 virtual FSR은 모두 같은 MuJoCo contact-force physics를 공유한다. 따라서 simulation observability에서 label/input circularity가 남으며 real-world independent ground truth validation 없이는 해결되지 않는다. ML training은 수행하지 않았다.

## 14. Viewer sanity

Official MuJoCo Viewer로 balanced-right moderate 3.0 s, medial-left 4.1 s, lateral-left 3.1 s, localized-left 4.1 s playback을 실행했다. 네 run 모두 `viewer=true`, `terminated_by_viewer=false`, finite IMU/FSR와 drop 0으로 완료했다. Viewer는 label tuning에 쓰지 않았다. Exact geometry tests가 모든 uneven cell의 common top `z=0`, complete tiling, no step/hole 조건을 재검증한다.

## 15. Verdict

`SINK_SUPPORT_LOSS_ORACLE_NOT_SUPPORTED`

Candidate는 baseline state, reset, total-load gate와 persistence가 모두 causal이고 future-independent지만 benign 12/14 firing과 uneven 2/12 때문에 acceptance에서 멀다. `SUPPORT_LOSS_ORACLE_NOT_CLEAN` 상태이며 primary SINK semantics를 freeze하거나 canonical dataset contract를 바꾸지 않는다.

## 16. Next recommendation

이번 binary retained-quadrant count는 normal stance-to-toe redistribution과 hazardous local support loss를 구분하지 못한다. 다음 redesign은 새 승인을 받은 뒤에만 수행하며, gait-phase-aware expected support expiration 또는 force-independent terrain/contact occupancy를 사전에 정의하고 같은 controls에서 먼저 검증해야 한다. 이번 결과로 threshold, persistence, 30% gate를 조정하지 않는다. Full Dataset generation, ML training, E84, quantization과 real-terrain claim은 시작하지 않는다.
