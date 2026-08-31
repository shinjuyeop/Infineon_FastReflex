# Ice Generalization Gap Resolution

## 1. Purpose

`ICE_GENERALIZATION_GAP_RESOLUTION`은 previous `GENERALIZATION_SCENARIO_CALIBRATION_BLOCKED` 전체를 다시 여는 작업이 아니라, P0 Ice gap 두 개만 current frozen mechanics에서 좁게 해소한 milestone이다. 시작 상태는 `main`, HEAD와 `origin/main` 모두 `618fb3d20b5656e19b9173a21aeca1b41742c9d7`였고 tracked worktree는 clean이었다.

이번 작업은 다음 질문만 다뤘다.

- 기존 `DELAYED_ICE_SLIP >=1000 ms`를 완화하지 않고, episode relation을 primary unit으로 쓰는 fresh `ONE_CONTACT_DELAYED_ICE_SLIP`이 존재하는가?
- Current frozen Ice에서 clean 50 ms opportunity를 가지며 Slip/I1/Support가 없는 `ICE_BENIGN_CONTROL`이 fresh signatures로 obtainable한가?
- 이 결론으로 다음 generation의 scenario set과 readiness를 freeze할 수 있는가?

Full generalization dataset generation은 시작하지 않았다.

## 2. Previous blocked findings

Previous calibration config/report SHA-256은 각각 `42088569621bf91cce22a12501698f23ae27b529f17fd356bbf8777f2ccab9c6`, `f78780c580a806bab2ab233072427f9f2db899296b3dcfb298f9ef0828ddb508`다. Pilot grid freeze는 `8d71a9a9aa4d3019e11facb9af2a8843a6a55f050b983e736ec64bdf98e50369`, staged collision-mask implementation correction은 `7e47db02a0cc573b58babe9346aea716ef0bb90be0d108a3113aad1f3b918218`, corrected physical selection freeze는 `da275e1d606c4e4592eecd7c41a1ba5ff7e6773f1e3f28cfe050c3f8e39e8ff9`였다.

- `ICE_BENIGN_CONTROL`: 12 pilots / 0 viable. 6개는 no Slip이지만 clean 50 ms Ice touchdown이 없었고 6개는 established Slip이 있었다. Verdict는 `ICE_BENIGN_NOT_OBTAINED_WITH_CURRENT_CALIBRATED_DOMAIN`이었다.
- `DELAYED_ICE_SLIP`: 8 pilots / frozen `>=1000 ms`에서 0 viable. Valid 4개는 한 번의 clean target touchdown 뒤 688–690 ms에 Slip이 있었지만 threshold를 만족하지 못했고, 나머지 4개는 target contact 전 fall이었다.
- READY는 `DELAYED_SAND_SUPPORT_ONSET`, `RIGHT_SAND_SUPPORT`, `SPEED_STRATIFIED_HAZARD`였다.
- `RIGHT_DOMINANT_ICE_SLIP`, `PHASE_SHIFTED_HAZARD`는 P1 BLOCKED로 유지하며 이번 primary target으로 사용하지 않았다.

## 3. Why >=1000 ms criterion was not relaxed

기존 `DELAYED_ICE_SLIP`의 `contact_to_slip_ms_min: 1000`은 그대로이고 verdict도 `BLOCKED`다. 이전 688–690 ms pilot을 새 `>=600/650/680 ms` threshold로 post-hoc PASS 처리하지 않았다. 새 family는 milliseconds cut-off가 아니라 “완전히 끝난 benign target-Ice physical contact episode가 이후의 causally distinct Ice Slip episode보다 먼저 존재하는가”를 묻는다. 따라서 `ONE_CONTACT_DELAYED_ICE_SLIP`은 이전 family의 criterion relaxation이 아니라 별도의 fresh hypothesis다.

## 4. New ONE_CONTACT_DELAYED_ICE_SLIP hypothesis

Frozen physical hypothesis는 다음 순서다.

```text
Concrete/Marble
→ exact Ice contact and Ice identity touchdown
→ at least one complete physical contact episode with no established Slip
→ that episode ends
→ a later, distinct physical contact episode attributable to Ice
→ established Slip
```

Primary unit은 per-foot physical contact episode다. `contact→Slip`, benign touchdown→Slip, benign episode end→Slip 시간은 모두 diagnostic이며 acceptance minimum은 없다. Exactly one과 two-or-more benign episodes를 각각 `ONE_CONTACT_DELAYED_SLIP_VIABLE`, `MULTI_CONTACT_DELAYED_SLIP_VIABLE`로 구분했으며 둘 다 “at least one” hypothesis를 만족한다.

## 5. Predeclared physical criteria

Criteria와 outcome priority는 simulation 전 config SHA-256 `95ce4b8ce48c47eba785a9a18cac0e6f9bff853050e1b8ad2f42ffaf4008ccd2`에 freeze했다.

- Physical episode는 `PhysicalDiagnostics.contact_episode_id`의 per-foot maximal contiguous non-negative ID interval이다.
- Complete episode는 contact release로 끝나며 exclusive end가 fall/duration censor보다 strictly earlier인 episode다.
- Target membership은 episode 안의 exact per-foot Ice contact, target touchdown은 `terrain_identity_touchdown`의 exact Ice false→true edge다.
- Qualifying benign episode는 complete, target Ice present, at least one exact Ice touchdown, same-foot established Slip false throughout, pre-censor, finite drift diagnostic을 모두 만족한다.
- Later distinct Slip은 다른 `(foot, contact_episode_id)` episode에 있고, 그 episode start가 benign episode end 이후이며, 같은 Slip episode에서 onset까지 exact Ice contact가 확인되어야 한다.
- Ice benign은 exact Ice contact, current helper와 동일한 clean 50 ms window, no Slip, no I1, no Support, no pre-target fall, finite runtime을 요구한다.
- `NONFINITE`, `NO_TARGET_CONTACT`, `PRETARGET_FALL`, `PHYSICAL_ORACLE_AMBIGUITY`, delayed viable categories, `SAME_CONTACT_SLIP`, `IMMEDIATE_SLIP`, `ICE_BENIGN_VIABLE`, `NO_SLIP_NO_CLEAN_ICE_CONTACT`, `OTHER_PHYSICAL_MISMATCH` 순으로 model-independent 분류했다.

Slip threshold, persistence, contact helper, Terrain clean-event helper를 바꾸지 않았다. Partial touchdown을 단순히 “아직 50 mm 미만”이라는 이유로 benign으로 부르지 않았다.

## 6. Fresh signature policy

Physical signature는 `(source, target, speed, start, width, slip_pattern, sink_pattern, severity, support_pattern)` 9-field tuple다.

| Audit | Reference signatures | Fresh signatures | Overlap |
|---|---:|---:|---:|
| Previous calibration pilots | 78 | 48 | 0 |
| Current unified dataset | 256 | 48 | 0 |

48/48 signatures는 unique이며 모두 future evaluation에서 제외한다. Current unified manifest metadata만 읽었고 HOLDOUT NPZ waveform은 열지 않았다. Grid-freeze artifact SHA-256은 `e8a975b24171b0a41d48aec750292e5032a34d86f9bae57270e2818a692c4cdb`다.

## 7. Pilot grid

Grid는 결과 전에 predeclared했고 Round 2 또는 post-result expansion은 없었다. Ice friction `[0.05, 0.001, 0.00001]`, controller, ONNX policy, 0.5 ms physics, 1 kHz sensor rate와 full-width `transition` topology를 유지했다.

| Family | Source | Speeds (m/s) | Starts (m) | Widths (m) | Runs |
|---|---|---|---|---|---:|
| `ONE_CONTACT_DELAYED_ICE_SLIP` | Concrete, Marble | 0.20, 0.25, 0.30 | 0.29, 0.34 | 0.71, 0.77 | 24 |
| `ICE_BENIGN_CONTROL` | Concrete, Marble | 0.20, 0.25, 0.30 | 0.305, 0.335 | 0.24, 0.30 | 24 |

Pilot outcome table:

| Candidate | Source | Speed | Geometry start × width (m) | Physical outcome | Benign episodes before Slip | Slip side |
|---|---:|---:|---:|---|---:|---|
| `igr_ibc_001` | Concrete | 0.20 | 0.305 × 0.24 | `NO_TARGET_CONTACT` | 0 | NONE |
| `igr_ibc_002` | Concrete | 0.20 | 0.305 × 0.30 | `NO_TARGET_CONTACT` | 0 | NONE |
| `igr_ibc_003` | Concrete | 0.20 | 0.335 × 0.24 | `ICE_BENIGN_VIABLE` | 0 | NONE |
| `igr_ibc_004` | Concrete | 0.20 | 0.335 × 0.30 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 16 | RIGHT |
| `igr_ibc_005` | Concrete | 0.25 | 0.305 × 0.24 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ibc_006` | Concrete | 0.25 | 0.305 × 0.30 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ibc_007` | Concrete | 0.25 | 0.335 × 0.24 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | LEFT |
| `igr_ibc_008` | Concrete | 0.25 | 0.335 × 0.30 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | LEFT |
| `igr_ibc_009` | Concrete | 0.30 | 0.305 × 0.24 | `SAME_CONTACT_SLIP` | 0 | BILATERAL |
| `igr_ibc_010` | Concrete | 0.30 | 0.305 × 0.30 | `SAME_CONTACT_SLIP` | 0 | BILATERAL |
| `igr_ibc_011` | Concrete | 0.30 | 0.335 × 0.24 | `ICE_BENIGN_VIABLE` | 0 | NONE |
| `igr_ibc_012` | Concrete | 0.30 | 0.335 × 0.30 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 13 | RIGHT |
| `igr_ibc_013` | Marble | 0.20 | 0.305 × 0.24 | `NO_TARGET_CONTACT` | 0 | NONE |
| `igr_ibc_014` | Marble | 0.20 | 0.305 × 0.30 | `NO_TARGET_CONTACT` | 0 | NONE |
| `igr_ibc_015` | Marble | 0.20 | 0.335 × 0.24 | `ICE_BENIGN_VIABLE` | 0 | NONE |
| `igr_ibc_016` | Marble | 0.20 | 0.335 × 0.30 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 2 | BILATERAL |
| `igr_ibc_017` | Marble | 0.25 | 0.305 × 0.24 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ibc_018` | Marble | 0.25 | 0.305 × 0.30 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ibc_019` | Marble | 0.25 | 0.335 × 0.24 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | LEFT |
| `igr_ibc_020` | Marble | 0.25 | 0.335 × 0.30 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | LEFT |
| `igr_ibc_021` | Marble | 0.30 | 0.305 × 0.24 | `SAME_CONTACT_SLIP` | 0 | BILATERAL |
| `igr_ibc_022` | Marble | 0.30 | 0.305 × 0.30 | `SAME_CONTACT_SLIP` | 0 | BILATERAL |
| `igr_ibc_023` | Marble | 0.30 | 0.335 × 0.24 | `ICE_BENIGN_VIABLE` | 0 | NONE |
| `igr_ibc_024` | Marble | 0.30 | 0.335 × 0.30 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 10 | RIGHT |
| `igr_ocd_001` | Concrete | 0.20 | 0.290 × 0.71 | `NO_TARGET_CONTACT` | 0 | NONE |
| `igr_ocd_002` | Concrete | 0.20 | 0.290 × 0.77 | `NO_TARGET_CONTACT` | 0 | NONE |
| `igr_ocd_003` | Concrete | 0.20 | 0.340 × 0.71 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 16 | BILATERAL |
| `igr_ocd_004` | Concrete | 0.20 | 0.340 × 0.77 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 16 | BILATERAL |
| `igr_ocd_005` | Concrete | 0.25 | 0.290 × 0.71 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ocd_006` | Concrete | 0.25 | 0.290 × 0.77 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ocd_007` | Concrete | 0.25 | 0.340 × 0.71 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | LEFT |
| `igr_ocd_008` | Concrete | 0.25 | 0.340 × 0.77 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | LEFT |
| `igr_ocd_009` | Concrete | 0.30 | 0.290 × 0.71 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ocd_010` | Concrete | 0.30 | 0.290 × 0.77 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ocd_011` | Concrete | 0.30 | 0.340 × 0.71 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 18 | BILATERAL |
| `igr_ocd_012` | Concrete | 0.30 | 0.340 × 0.77 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 18 | BILATERAL |
| `igr_ocd_013` | Marble | 0.20 | 0.290 × 0.71 | `NO_TARGET_CONTACT` | 0 | NONE |
| `igr_ocd_014` | Marble | 0.20 | 0.290 × 0.77 | `NO_TARGET_CONTACT` | 0 | NONE |
| `igr_ocd_015` | Marble | 0.20 | 0.340 × 0.71 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 2 | BILATERAL |
| `igr_ocd_016` | Marble | 0.20 | 0.340 × 0.77 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 2 | BILATERAL |
| `igr_ocd_017` | Marble | 0.25 | 0.290 × 0.71 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ocd_018` | Marble | 0.25 | 0.290 × 0.77 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ocd_019` | Marble | 0.25 | 0.340 × 0.71 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ocd_020` | Marble | 0.25 | 0.340 × 0.77 | `ONE_CONTACT_DELAYED_SLIP_VIABLE` | 1 | BILATERAL |
| `igr_ocd_021` | Marble | 0.30 | 0.290 × 0.71 | `NO_TARGET_CONTACT` | 0 | NONE |
| `igr_ocd_022` | Marble | 0.30 | 0.290 × 0.77 | `NO_TARGET_CONTACT` | 0 | NONE |
| `igr_ocd_023` | Marble | 0.30 | 0.340 × 0.71 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 18 | BILATERAL |
| `igr_ocd_024` | Marble | 0.30 | 0.340 × 0.77 | `MULTI_CONTACT_DELAYED_SLIP_VIABLE` | 18 | BILATERAL |

## 8. One-contact delayed Slip results

`ONE_CONTACT_DELAYED_ICE_SLIP` own grid 24개 중 18개가 viable했다: exactly-one 10, multi-contact 8, no target contact 6이다. Viable set은 Concrete/Marble 양 source와 0.20/0.25/0.30 m/s 세 speed stratum을 모두 포함한다. Readiness requirement 4 runs, both sources, at least two speeds를 넘었으므로 `READY`이며 scenario verdict는 `ONE_CONTACT_DELAYED_ICE_SLIP_SUPPORTED`다.

Viable contact→Slip은 160–1,128 ms, benign episode end→Slip은 140–789 ms였다. 이 범위는 diagnostic이지 새 acceptance threshold가 아니다. Viable physical Slip은 BILATERAL 16, LEFT 2였고 right-only를 만들기 위한 reselection은 하지 않았다.

Selected delayed-Slip timing:

| Run | First Ice contact | Benign touchdown | Benign episode end | Next target touchdown | Slip | Contact→Slip | Benign end→Slip | Benign episodes | Slip side |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `igr_ocd_003` | 1504 | 1504 | 1843 | 1902 | 2632 | 1128 ms | 789 ms | 16 | BILATERAL |
| `igr_ocd_018` | 1220 | 1220 | 1533 | 1711 | 1908 | 688 ms | 375 ms | 1 | BILATERAL |
| `igr_ocd_007` | 1220 | 1220 | 1529 | 1772 | 1910 | 690 ms | 381 ms | 1 | LEFT |
| `igr_ocd_009` | 1298 | 1298 | 1318 | 1423 | 1458 | 160 ms | 140 ms | 1 | BILATERAL |
| `igr_ocd_004` | 1504 | 1504 | 1843 | 1902 | 2632 | 1128 ms | 789 ms | 16 | BILATERAL |
| `igr_ocd_005` | 1220 | 1220 | 1529 | 1772 | 1910 | 690 ms | 381 ms | 1 | BILATERAL |

## 9. Contact episode analysis

각 pilot manifest에는 target first contact/touchdown, every per-foot episode boundary, Slip onset per foot, first qualifying episode, pre-Slip benign episode count, actual side와 fall/censor가 있다. Selected first benign episode evidence는 다음과 같다.

| Run | Episode | Complete duration | Exact Ice duration | Max drift in benign episode | Slip phase |
|---|---|---:|---:|---:|---|
| `igr_ocd_003` | right:3 | 346 ms | 339 ms | 0.007508 m | RIGHT_SINGLE_SUPPORT |
| `igr_ocd_018` | left:2 | 313 ms | 312 ms | 0.004381 m | LEFT_SINGLE_SUPPORT |
| `igr_ocd_007` | left:2 | 309 ms | 307 ms | 0.004401 m | LEFT_SINGLE_SUPPORT |
| `igr_ocd_009` | right:8 | 20 ms | 20 ms | 0.028248 m | RIGHT_SINGLE_SUPPORT |
| `igr_ocd_004` | right:3 | 346 ms | 339 ms | 0.007508 m | RIGHT_SINGLE_SUPPORT |
| `igr_ocd_005` | left:2 | 309 ms | 307 ms | 0.004401 m | LEFT_SINGLE_SUPPORT |

모든 row에서 benign episode 동안 frozen established-Slip state는 false였고 episode가 censor 전에 contact release로 끝났다. First Slip episode는 별도 episode였으며 onset 이전까지 exact Ice membership이 확인됐다. Oracle ambiguity는 0/48이었다.

## 10. Ice benign fresh audit

`ICE_BENIGN_CONTROL` own grid 24개 결과는 benign viable 4, one-contact delayed Slip 8, multi-contact delayed Slip 4, same-contact Slip 4, no target contact 4였다. Benign four는 양 source, speed 0.20/0.30 m/s에 존재했고 모두 start 0.335 m / width 0.24 m였다. Readiness requirement 2 runs와 both sources를 만족해 `READY`, scenario verdict는 `ICE_BENIGN_SUPPORTED`다.

Ice benign result. Contact duration은 censor 전 any-foot exact Ice union duration이고 max drift는 exact Ice contact sample에서 계산했다.

| Run | Source / speed | Ice contact duration | Clean Ice opportunity | Affected clean foot | Max drift | Slip/I1/Support | Fall | Result |
|---|---|---:|---:|---|---:|---|---:|---|
| `igr_ibc_003` | Concrete / 0.20 | 1325 ms | 2 | LEFT, RIGHT | 0.046764 m | none / none / none | 2866 | `ICE_BENIGN_VIABLE` |
| `igr_ibc_023` | Marble / 0.30 | 689 ms | 1 | LEFT | 0.043569 m | none / none / none | 1919 | `ICE_BENIGN_VIABLE` |
| `igr_ibc_011` | Concrete / 0.30 | 687 ms | 2 | LEFT | 0.029781 m | none / none / none | 1918 | `ICE_BENIGN_VIABLE` |
| `igr_ibc_015` | Marble / 0.20 | 1133 ms | 2 | LEFT, RIGHT | 0.048749 m | none / none / none | 2666 | `ICE_BENIGN_VIABLE` |

Fall은 target contact와 clean window 뒤에 발생했으며 pre-target fall은 아니다. Benign label은 fall/recovery가 아니라 pre-censor Slip/I1/Support와 clean Ice evidence로 정했다.

## 11. Current Ice-domain limitation

Fresh audit은 “Ice benign이 없다”는 previous conclusion을 갱신하지만 current Ice 전체가 benign하다는 의미는 아니다. Benign은 tested grid에서 start 0.335 m / width 0.24 m의 narrow geometry에만 나타났고, 20/24 candidates는 clean benign control이 아니었다. 모든 benign run도 이후 fall로 censor됐다. 따라서 frozen domain statement는 “bounded encounter에서 clean 50 ms no-Slip/I1/Support Ice opportunity를 재현 가능”이며 indefinitely stable Ice walking 또는 broad-geometry benignity claim은 금지한다.

## 12. Optional low-friction benign control concept

Current Ice benign이 supported됐으므로 `LOW_FRICTION_HARD_BENIGN_CONTROL`은 이번 milestone에서 추천하거나 구현하지 않는다. 향후 Hazard robustness stress control로 별도 material이 필요하면 `REQUIRES_NEW_PHYSICS_CALIBRATION`로 다뤄야 하며, 그 profile은 Ice가 아니고 current four-class Terrain MLP의 ICE ground truth evidence도 아니다.

## 13. Model-blind selection freeze

Physical simulation과 classification이 전부 끝난 뒤 `MODEL_BLIND_PHYSICAL_SELECTION_COMPLETE`를 선언했다. Physical selection artifact SHA-256은 `5d36b9b63a77bb2ae40f54ea3ed34024a25e50c269742c4cda7689b75a73cc48`, pilot manifest artifact SHA-256은 `a8bda0dab5b6b54266d8c6ba22750b23042e221d8e4be6aa9b8e038ac55eec03`다. Predeclared config SHA가 allowed mechanics, parameter domain, required episode relation, forbidden outcomes와 signature exclusions의 family-specification identity다.

Selection input은 exact contact, contact episode, exact touchdown, drift, Slip, I1/Support, side, fall/censor, geometry와 speed뿐이었다. Hazard probability/reflex와 Terrain output은 freeze 전에 load하지 않았다. Selected IDs는 delayed 6개와 benign 4개이며 selection 뒤 변경은 0이다.

## 14. Post-freeze Hazard/Terrain smoke

Freeze 이후에만 lexicographically first selected representative `igr_ocd_003`, `igr_ibc_003`를 deterministic re-simulation하고 current protected model을 read-only replay했다. Timestamp, Pelvis IMU6, FSR8, exact terrain/physical contact, loaded contact, episode ID, drift/velocity, Slip clocks, support spread와 Support clock 모두 exact parity였다. Smoke artifact SHA-256은 `1de99c4ca80920ab21d07aed165323ac788700f60a5062d3c51c30998c3119da`다.

| Pilot | Physical result | First Reflex | First valid ICE | Peak Hazard p | Diagnostic |
|---|---|---:|---:|---:|---|
| `igr_ocd_003` | multi-contact delayed Slip at 2632 | 2470 | 2443 | 0.999988 | Terrain and Reflex both precede Slip |
| `igr_ibc_003` | clean Ice, no Slip/I1/Support | 2470 | 2443 | 0.999981 | Hazard alert is a post-freeze false-positive diagnostic |

Benign representative의 Hazard alert는 scenario selection을 바꾸지 않았고 performance denominator로 사용하지 않았다. Terrain은 advisory only이며 Hazard를 gate하지 않았다. Hazard freeze `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`, feature schema `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`, Terrain normalizer `2c551181fbd23080c0e3008be8484bd0ee4139e594041ba6e46ab5b36ffb05de`와 three-seed checkpoints는 모두 protected verification을 통과했다.

Offscreen snapshot에서 `igr_ocd_003`의 benign Ice touchdown과 later distinct Slip pose, `igr_ibc_003`의 clean Ice contact를 확인했다. Ice patch, robot/surface geometry와 temporal distinction이 coherent했다. Visualization audit SHA-256은 `6e401675b77c3f7274af4585f56a23c5577d6f66064d2e1d0a92d4625673b156`이며 visualization은 selection input이 아니다.

## 15. Timing order

Selected delayed family smoke representative의 order는 다음과 같다.

```text
First Ice contact / benign touchdown 1504
→ benign episode end 1843
→ first valid Terrain ICE 2443
→ first Reflex 2470
→ established Slip 2632
```

따라서 annotation은 `TERRAIN_BEFORE_REFLEX`와 `TERRAIN_BEFORE_ESTABLISHED_SLIP`이다. Terrain→Reflex는 27 ms, Terrain→Slip은 189 ms, Reflex→Slip은 162 ms였다. 이는 post-freeze model timing diagnostic일 뿐 physical family definition이 아니다. Previous `gsc_ssh_001`은 fresh evidence나 PASS 근거로 재사용하지 않았다.

## 16. Existing READY families

Previous selection freeze의 다음 families는 재calibrate하거나 변경하지 않았다.

- `DELAYED_SAND_SUPPORT_ONSET`: P0 READY
- `RIGHT_SAND_SUPPORT`: P1 READY
- `SPEED_STRATIFIED_HAZARD`: P1 READY

Previous `DELAYED_ICE_SLIP >=1000 ms`는 BLOCKED, `RIGHT_DOMINANT_ICE_SLIP`과 `PHASE_SHIFTED_HAZARD`는 P1 BLOCKED로 남는다.

## 17. Final next-generation scenario set

| Family | Priority | Status | Include next dataset? | Reason |
|---|---|---|---|---|
| `ONE_CONTACT_DELAYED_ICE_SLIP` | P0 | READY | YES | 18/24 viable, both sources, 3 speed strata; distinct benign→Slip episode relation |
| `ICE_BENIGN_CONTROL` | P0 | READY | YES | 4/24 clean no-Slip/I1/Support runs, both sources |
| `DELAYED_SAND_SUPPORT_ONSET` | P0 | READY (preserved) | YES | Previous model-blind freeze; not recalibrated |
| `RIGHT_SAND_SUPPORT` | P1 | READY (preserved) | YES | Previous model-blind freeze; not recalibrated |
| `SPEED_STRATIFIED_HAZARD` | P1 | READY (preserved) | YES | Previous model-blind freeze; not recalibrated |

Fresh pilot signatures 48개와 previous pilot signatures 78개는 next dataset evaluation에서 제외한다. Next generation은 이 family contracts에서 new signatures를 만들어야 한다.

## 18. Dataset-generation readiness

Recommendation은 `FULL_GENERALIZATION_DATASET_READY`다. 두 fresh Ice P0 family가 모두 READY이고 기존 READY 3개가 유지되어 five-family final set을 방어 가능하게 freeze했다. 이것은 scenario-calibration readiness이지 model performance verdict가 아니다. 이 milestone에서는 generation을 자동 시작하지 않았다.

## 19. Limitations

- Benign Ice availability는 narrow tested geometry이고 모든 viable benign run은 clean evidence 뒤 later fall로 censor됐다.
- New delayed family는 at least one complete episode relation이며 “long delay”의 milliseconds guarantee가 아니다. Diagnostic contact→Slip은 160–1,128 ms다.
- Natural selected Slip phase는 left/right single support였고 right-only Slip과 missing `CONTACT_RELEASE`/`DOUBLE_SUPPORT` phase gap을 해결하지 않았다.
- Post-freeze model smoke는 fresh representatives 두 개의 compatibility/timing audit이며 performance estimate가 아니다. 특히 benign representative의 Hazard alert는 future generalization evaluation에서 재검증해야 한다.
- Simulator/model source, Ice material, controller, policy, threshold, normalizer와 checkpoint는 변경하지 않았다.
- Existing four validation representatives (`uhr_ice_h_c20`, `uhr_sand_h_c20`, `uhr_sand_b_c20`, `uhr_hard_n_c20`)는 exact parity PASS했다. Regression artifact SHA-256은 `9ca72c694d5a0bad080ac448f04af25e910e490446d06871153b056d29e94277`다.
- Verification은 full pytest `71 passed, 1 skipped`, `python -m compileall src scripts tests` PASS, Ruff `E9,F63,F7,F82` PASS였다. Existing unified manifest SHA-256과 256/256 NPZ byte hashes도 모두 일치했다.
- Current HOLDOUT waveform reopened: **NO**. Current HOLDOUT new inference: **NO**.
- Optimizer steps 0, checkpoint writes 0, normalizer fits 0, threshold searches 0, architecture searches 0이다.

## 20. Verdict

Scenario verdicts:

```text
ONE_CONTACT_DELAYED_ICE_SLIP_SUPPORTED
ICE_BENIGN_SUPPORTED
```

Overall milestone verdict:

```text
ICE_GENERALIZATION_GAP_RESOLVED
FULL_GENERALIZATION_DATASET_READY
```

모든 physical decision은 model-blind freeze 전에 끝났고 model replay로 selection을 바꾸지 않았다. Current HOLDOUT은 sealed 상태이고 training/new checkpoint는 없으며 full generalization dataset generation은 시작하지 않았다.
