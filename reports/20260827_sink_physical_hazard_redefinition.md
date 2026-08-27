# Sink Physical Hazard Redefinition

## 1. 왜 기존 outcome-based SINK가 어려웠는지

기존 frozen primary SINK는 patch-linked physical penetration t1 뒤에 미래 pelvis tilt degradation t2가 발생해야 hazardous episode로 확정한다. 따라서 현재 support response가 같아도 gait phase, controller recovery와 이후 fall 여부에 따라 BENIGN/SINK가 달라질 수 있다. 이번 study는 기존 `SINK_HAZARD_CRITERIA_FROZEN`을 삭제하거나 덮어쓰지 않고, 현재·과거 exact contact state만 쓰는 `UNEVEN_SUPPORT_SINK` 후보를 별도로 검증했다.

결론부터 말하면 simple penetration-spread candidate는 causal하지만 direction/side/speed coverage를 확보하지 못했다. 기존 contract와 README status는 변경하지 않는다.

## 2. 새 causal physical definition

후보 metric은 각 foot의 loaded quadrant에서 계산한 다음 값이다.

```text
support_penetration_spread_m
  = max(valid quadrant penetration)
  - min(valid quadrant penetration)
```

Foot total load는 기존 5 N on/2.5 N off hysteresis를 재사용한다. Touchdown 뒤 첫 10 ms, pre-fall이 아닌 sample, quadrant normal load가 2.5 N 미만인 region은 invalid다. 최소 두 quadrant가 valid할 때만 spread를 정의한다. Candidate u1은 spread가 threshold 이상인 상태가 같은 contact episode에서 20 consecutive samples 지속된 첫 sample이다.

Oracle은 terrain name, support-pattern name, IMU, FSR, future t2, recovery 또는 fall을 사용하지 않는다. Future sample을 바꾸어도 이미 결정된 u1 이전 결과가 바뀌지 않는 causal persistence test를 통과했다.

## 3. MuJoCo approximation limitation

이번 terrain은 deformable soil이나 입자 이동을 model하지 않는다. 정확한 의미는 같은 nominal top height에서 frozen moderate/severe contact compliance가 국소적으로 다른 engineering approximation이다. 결과를 실제 모래 침하 깊이 또는 universal real-world Sink threshold로 해석하지 않는다.

## 4. Uneven terrain construction

새 scene file을 만들지 않고 canonical `scene_sink.xml`의 finite patch에 disabled cell slot을 추가했다. Uneven pattern에서는 affected side를 entry/exit × medial/lateral 네 cell로 tile하고, 나머지 세 transition surface와 함께 한 topology만 활성화한다.

- `balanced_soft`: 기존 whole-side mild/moderate/severe profile과 exact parity
- `medial_soft`: medial 두 cell severe, lateral 두 cell moderate
- `lateral_soft`: lateral 두 cell severe, medial 두 cell moderate
- `localized_soft`: entry-medial 한 cell severe, 나머지 세 cell moderate
- 모든 enabled surface top은 정확히 `z=0`
- x/y cell boundary는 맞닿으며 gap, overlap volume, hole 또는 geometric step이 없다.
- Controller, mass, policy, friction과 frozen moderate/severe endpoint는 변경하지 않았다.

Viewer launch smoke는 balanced-right moderate, medial-right severe와 반대 방향 lateral-left severe에서 완료됐고 `terminated_by_viewer=false`, finite IMU/FSR, drop 0, established Slip 0을 확인했다. Exact geometry tests가 공통 top과 complete tiling을 검증한다. 자연적인 non-fall uneven case는 이번 matrix에서 없었다.

## 5. Exact penetration-spread diagnostic

각 named sole-ground MuJoCo contact의 world contact position을 ankle-roll body-local frame으로 변환하고 frozen FSR4 quadrant order로 mapping한다. 각 quadrant에 여러 contact가 있으면 maximum physical penetration을 사용한다. Maximum은 localized collapse를 평균으로 희석하지 않고 runtime FSR force를 ground truth aggregation에 사용하지 않으므로 선택했다. No/low-load quadrant는 0 m가 아니라 invalid(`NaN` in exact trace)로 유지한다.

Simulator-only diagnostics는 quadrant contact, normal load, maximum penetration, loaded mask/count, penetration spread, maximum penetration, load-weighted penetration standard deviation과 load concentration을 제공한다. 이 값들은 runtime model input이 아니다. Existing virtual FSR observer는 dynamics를 바꾸지 않았고 channel/order/force-sum parity test를 유지한다.

## 6. Benign envelope

사전 선언한 14 controls의 모든 pre-fall valid loaded-support sample에서 spread upper envelope를 구했다. 모든 run은 8,000 IMU6/FSR8 samples, 1 kHz timestamp, drop 0이었고 fall은 없었다.

| Control domain | Runs | Max spread range |
|---|---:|---:|
| Concrete | 2 | 2.578–2.744 mm |
| Marble | 2 | 2.506–2.776 mm |
| Uniform Sand | 2 | 9.922–10.530 mm |
| Balanced mild | 4 | 12.939–17.508 mm |
| Balanced moderate | 4 | 16.104–22.655 mm |

Upper envelope는 `balanced_left_moderate_s025`의 `0.022655311971902847 m`였다. 기존 future-outcome diagnostic은 balanced mild-left 0.25 m/s와 balanced moderate-right 0.25 m/s에서도 나중에 t2가 발생했지만 두 run 모두 8초 non-fall이었다. 이는 support state와 downstream controller outcome을 분리해야 한다는 문제 정의를 다시 보여준다.

## 7. Candidate threshold

사전 선언한 rule을 한 번만 적용했다.

```text
0.022655311971902847 m benign envelope
+ 0.001000000000000000 m fixed engineering margin
= 0.023655311971902848 m candidate threshold
```

Uneven 결과를 보고 threshold를 낮추거나 margin을 조정하지 않았다. 이 threshold는 freeze된 contract가 아니라 실패한 sanity candidate다.

## 8. Persistence

Persistence는 20 ms다. 같은 foot contact episode 안에서 valid spread가 threshold 이상인 20번째 consecutive 1 kHz sample에 u1이 causal하게 active된다. Load loss, invalid quadrant count, contact episode 변경 또는 threshold 미충족에서 count를 reset한다. Touchdown transient만으로 firing한 control은 없다.

## 9. Sanity matrix

Config는 [`20260827_sink_physical_hazard_redefinition.yaml`](../configs/experiment/20260827_sink_physical_hazard_redefinition.yaml)이다. Verified policy, fixed stand, 8초, 2 kHz physics/1 kHz sensors, speeds 0.15/0.25 m/s를 사용했다. 표의 event time은 1-based simulation timestamp ms다.

| Scenario | Role | Speed | Side/pattern | t0 | u1 | t0→u1 | Max spread | Max pen. | Old t2 | Fall | u1→t2 | u1→fall | Classification |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| concrete_s015 | control | 0.15 | — | — | — | — | 2.578 | 3.579 | — | — | — | — | BENIGN_SUPPORT |
| concrete_s025 | control | 0.25 | — | — | — | — | 2.744 | 3.812 | — | — | — | — | BENIGN_SUPPORT |
| marble_s015 | control | 0.15 | — | — | — | — | 2.506 | 3.610 | — | — | — | — | BENIGN_SUPPORT |
| marble_s025 | control | 0.25 | — | — | — | — | 2.776 | 3.516 | — | — | — | — | BENIGN_SUPPORT |
| uniform_sand_s015 | control | 0.15 | — | — | — | — | 10.530 | 14.167 | — | — | — | — | BENIGN_SUPPORT |
| uniform_sand_s025 | control | 0.25 | — | — | — | — | 9.922 | 14.770 | — | — | — | — | BENIGN_SUPPORT |
| balanced_left_mild_s015 | control | 0.15 | left/balanced | 1,803 | — | — | 12.939 | 17.077 | — | — | — | — | BENIGN_SUPPORT |
| balanced_left_mild_s025 | control | 0.25 | left/balanced | 1,221 | — | — | 17.286 | 17.533 | 6,085 | — | — | — | BENIGN_SUPPORT |
| balanced_right_mild_s015 | control | 0.15 | right/balanced | 2,096 | — | — | 17.508 | 18.921 | — | — | — | — | BENIGN_SUPPORT |
| balanced_right_mild_s025 | control | 0.25 | right/balanced | 1,508 | — | — | 15.573 | 16.523 | — | — | — | — | BENIGN_SUPPORT |
| balanced_left_moderate_s015 | control | 0.15 | left/balanced | 1,803 | — | — | 16.104 | 22.161 | — | — | — | — | BENIGN_SUPPORT |
| balanced_left_moderate_s025 | control | 0.25 | left/balanced | 1,221 | — | — | 22.655 | 23.829 | — | — | — | — | BENIGN_SUPPORT |
| balanced_right_moderate_s015 | control | 0.15 | right/balanced | 2,096 | — | — | 20.179 | 24.221 | — | — | — | — | BENIGN_SUPPORT |
| balanced_right_moderate_s025 | control | 0.25 | right/balanced | 1,508 | — | — | 19.261 | 21.632 | 5,246 | — | — | — | BENIGN_SUPPORT |
| medial_left_s015 | uneven | 0.15 | left/medial | 1,803 | — | — | 23.570 | 29.399 | 2,920 | 6,976 | — | — | BENIGN_SUPPORT |
| medial_left_s025 | uneven | 0.25 | left/medial | 1,221 | — | — | 18.595 | 25.724 | 1,657 | 2,551 | — | — | BENIGN_SUPPORT |
| medial_right_s015 | uneven | 0.15 | right/medial | 2,096 | 2,416 | 320 | 24.145 | 37.882 | — | 2,897 | — | 481 | UNEVEN_SUPPORT_SINK |
| medial_right_s025 | uneven | 0.25 | right/medial | 1,508 | — | — | 19.146 | 36.677 | — | 2,329 | — | — | BENIGN_SUPPORT |
| lateral_left_s015 | uneven | 0.15 | left/lateral | 1,803 | — | — | 21.044 | 123.101 | 2,883 | 3,144 | — | — | BENIGN_SUPPORT |
| lateral_left_s025 | uneven | 0.25 | left/lateral | 1,221 | — | — | 14.185 | 133.981 | 1,708 | 2,012 | — | — | BENIGN_SUPPORT |
| lateral_right_s015 | uneven | 0.15 | right/lateral | 2,096 | — | — | 21.736 | 61.140 | — | 2,846 | — | — | BENIGN_SUPPORT |
| lateral_right_s025 | uneven | 0.25 | right/lateral | 1,508 | — | — | 19.807 | 37.101 | — | 2,249 | — | — | BENIGN_SUPPORT |
| localized_left_s015 | uneven | 0.15 | left/localized | 1,803 | — | — | 23.570 | 29.399 | 2,920 | 6,976 | — | — | BENIGN_SUPPORT |
| localized_left_s025 | uneven | 0.25 | left/localized | 1,221 | — | — | 18.595 | 40.613 | 1,657 | 2,551 | — | — | BENIGN_SUPPORT |
| localized_right_s015 | uneven | 0.15 | right/localized | 2,096 | 2,416 | 320 | 24.145 | 29.866 | — | 2,897 | — | 481 | UNEVEN_SUPPORT_SINK |
| localized_right_s025 | uneven | 0.25 | right/localized | 1,508 | — | — | 19.146 | 35.164 | — | 2,329 | — | — | BENIGN_SUPPORT |

Penetration values are millimetres. No run was corrupt or censored before t0, so `INVALID=0`.

## 10. Left/right/direction results

- Benign controls: 14/14 no firing.
- Intended uneven: 2/12 detected.
- Left: 0/6; right: 2/6.
- 0.15 m/s: 2/6; 0.25 m/s: 0/6.
- Medial: 1/4; lateral: 0/4; localized: 1/4.
- Both detected traces are the same right-entry physical prefix: localized and full medial placement are dynamically identical until the robot leaves the entry cell.

The uneven maximum spread range, 14.185–24.145 mm, overlaps the balanced envelope up to 22.655 mm. Ten intended uneven cases remain below the fixed threshold. Lateral cases can show very large maximum penetration while only one quadrant remains meaningfully loaded; the minimum-two-loaded-quadrant validity then makes spread unavailable precisely as support is lost. Lowering the threshold would violate the predeclared selection rule and threaten balanced-control false positives.

## 11. Fall/non-fall outcome separation

All 12 intended uneven runs eventually had non-foot fall/censor, but only two satisfied u1. Among detected events, fall=2 and non-fall/recovery=0; both u1→fall intervals were 481 ms. Old t2 occurred in six intended uneven runs, including several that the new spread rule missed, and did not occur in the two detected right cases before fall.

This demonstrates that the implementation is causally independent of future outcome: neither later t2 nor fall retroactively creates u1. It does not demonstrate the desired physical coverage or at least one naturally non-fall Sink event.

## 12. FSR descriptive observability

Only the two duplicate-prefix detected events permit u1 alignment, so no population-level separation claim is possible. Their affected right-foot raw FSR4 at u1 was `[30.9302, 16.1872, 9.8924, 13.6833] N`; at +20/+50 ms load had concentrated in one channel (`55.6041`/`59.3881 N`) and at +100 ms the foot was airborne. A matched balanced-right moderate trace at the same t0+320 ms elapsed point already had single-channel concentration at +20/+50 ms and was also airborne at +100 ms. Fusion14 raw-vector L2 changes likewise overlapped: uneven 85.77/151.13/174.95 versus balanced 85.76/142.89/176.52 at +20/+50/+100 ms.

FSR/Fusion14 therefore show contact redistribution but not descriptive separation in this tiny detected subset. They were not used to choose the oracle or threshold, and no classifier was trained.

## 13. Failure cases

1. Balanced moderate at 0.25 m/s raises the benign envelope to 22.655 mm.
2. Ten of twelve intended uneven conditions do not clear 23.655 mm for 20 ms.
3. Detection is absent on the left, at 0.25 m/s and for every lateral pattern.
4. Localized and medial right 0.15 m/s are not independent physical responses before u1.
5. Severe lateral collapse can unload a quadrant; loaded-only spread then loses the region rather than representing its support loss.
6. All intended uneven runs fall, so recovery-independent positive evidence is absent.
7. Adjusting threshold, margin, load cutoff or persistence after these results would be prohibited tuning.

## 14. Comparison to old t1/t2 semantics

The old t1/t2 arrays, thresholds, dataset artifacts and reports remain unchanged. The new diagnostic correctly separates causal computation from future effect: balanced controls can have later old t2 without u1, and intended uneven runs can fall without u1. However, because the candidate misses most designed uneven support, the old outcome-based SINK is not semantically demoted in canonical `docs/dataset.md` during this milestone.

Frozen Slip remains 50 mm touchdown-anchor tangential drift plus 3 ms persistence. All Viewer representatives had established Slip 0, and the complete test suite retains Slip threshold/persistence regression and FSR observer parity.

## 15. Verdict

`SINK_PHYSICAL_HAZARD_REDEFINITION_NEEDS_REVISION`

The exact metric and persistence are simple and causal, controls do not fire, and the same-height terrain construction is valid. Acceptance nevertheless fails on intended uneven detection, left/right and direction coverage, multiple-speed coverage and non-fall evidence. The candidate threshold is recorded for audit but is not frozen as the primary SINK contract.

## 16. Next step

Do not generate the Full Dataset or train a model. A separately approved bounded redesign should first address support-region dropout without using FSR as ground truth—for example, retain touchdown-established quadrant support references so a formerly loaded region becoming unavailable is represented causally rather than removed from spread. That redesign needs a new predeclared benign envelope and the same no-tuning acceptance gate. Full Dataset, ML, E84, quantization and real-terrain claims remain out of scope.
