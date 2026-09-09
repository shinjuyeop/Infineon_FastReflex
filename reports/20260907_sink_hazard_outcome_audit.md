# Sink 라벨과 로봇 위험·회복의 사례 대조

2026-09-07, Asia/Seoul. 기준 source commit `62ae1c42fe4bcfdd57006d114d2539ed19c2f4ac`.

## 결론

요청한 세 경우의 대표 사례를 확인했다. **현재 Support/Sink oracle과 로봇 몸체의 실제 악화 정도는 일대일로 대응하지 않는다.** 동일한 Support 양성 안에 몸체 변화가 작은 사례, 지속적으로 악화된 사례, 크게 흔들렸다가 회복하는 사례가 섞인다. 반대로 Support/Slip이 전혀 확정되지 않았는데 몸체가 붕괴하는 사례도 있다.

따라서 목적이 `대응이 필요한 로봇 불안정`이라면 물리 이벤트 라벨을 그대로 최종 위험 정답으로 사용하는 가정을 재검토할 근거가 있다. 다만 이번 결과만으로 양성 사례를 정상으로 재라벨하거나 새 위험 threshold를 정할 수는 없다. 현재 보행 controller가 위험을 성공적으로 억제했을 가능성은 남는다.

또한 **mechanical recovery는 로봇 회복이 아니며, nonfoot contact에 의한 fall flag는 반드시 몸체 낙상을 뜻하지 않는다.** 기존 설명에서 이 두 구분을 충분히 하지 않은 부분을 이번 사례 분석으로 바로잡는다.

## 방법과 재현성

- 기존 proxy sanity 32개와 boundary 120개는 저장된 metadata로 전체 집계했다.
- 사전에 선택한 10개를 canonical `run_simulation`으로 재현했다: 역사적 proxy 6개 + hard-ground 대조군 2개 + 현재 boundary 2개. 선택은 저장된 label/fall/old_t2/mechanical recovery를 이용한 목적 표집이며 유병률이나 전체 성능 추정이 아니다.
- boundary의 접촉 censor 1개는 접촉 geom/body를 읽기 위해 한 번 추가 재현했다. 총 시뮬레이션 11회, optimizer/model inference 0회. 원본 데이터·라벨·simulator 코드는 변경하지 않았다.
- 역사적 8개는 Support, 과거 자세 진단, fall clock이 저장값과 모두 일치했다. boundary 2개는 원본 IMU·timestamp·Support onset·Slip onset 배열이 모두 정확 일치하고 fall clock도 일치했다.
- 독립 진단은 pelvis tilt, pelvis 높이, 전진 속도, loaded contact, raw MoS를 기록했다. 사건 이전 1초, 사건 이후 pre-censor, 마지막 1초와 500ms bin을 요약했다. 새 이진 위험 기준이나 정상 envelope를 fit하지 않았다.
- 시간은 아래 표에서 **원본 simulation timestamp(초)**다. 샘플 index 0의 timestamp가 1ms이므로 index/1000과 1ms 차이가 있다.
- 과거 `old_t2`는 tilt >0.04454633221030235rad가 20샘플 지속되는 자세 진단이다. 별도 physical-sink 조건이 결합된 `sink_hazard_onset`과 구분하여 재계산했다. 이 오래된 자세 threshold도 일반 위험 정답으로 채택하지 않았다.
- MoS는 정상 보행에서도 음수가 될 수 있고 기존 안정성 oracle 검증이 실패했으므로 단독 판정에 사용하지 않았다.

설정: [audit config](../configs/experiment/20260907_sink_hazard_outcome_audit.yaml). 원시 진단/500ms 수치: [results.json](../artifacts/runs/20260907_sink_hazard_outcome_audit/results.json). 파일들은 Gitignored generated artifacts이며 원본 frozen corpus에 포함되지 않는다.

## 1. 위험 라벨이지만 몸체 변화가 작은 사례

`moderate_localized_right_s025`: 오른쪽 localized support, .25m/s.

| 항목 | 결과 |
|---|---:|
| 패치 접촉 | 1.508s |
| Support 확정 | 2.223s |
| Slip / old_t2 / fall | 모두 없음 |
| Support 이후 tilt 최대 / p95 | 1.654° / 1.429° |
| Support 이후 평균 pelvis 높이 | 0.779m |
| Support 이후 평균 전진 속도 | 0.233m/s (명령 .25) |
| 마지막 1초 tilt p95 / 평균 높이 / 평균 속도 | 0.721° / 0.781m / 0.241m/s |
| 같은 속도 hard-ground 대조군 마지막 1초 | 0.706° / 0.781m / 0.235m/s |

Support 이전 1초의 tilt p95는 1.754°, 평균 높이는 .780m, 속도는 .231m/s였다. 지면은 Support oracle을 충족하지만 몸체 자세·평균 진행은 큰 지속 악화를 보이지 않고 마지막에는 대조군 수준이다. 단, 순간 최소 높이 .763m 등 작은 반응은 있으므로 `아무 영향 없음`이라고 쓰지 않는다. 각 발의 지지 여유와 개입 필요성까지 완전히 정상이라고 입증한 것은 아니다.

**판정:** 요청 1의 유력한 관측 사례. 현재 라벨이 사건의 심각도를 구분하지 못한다는 근거이며, 이 사례의 위험 라벨이 무조건 잘못됐다는 확정은 아니다.

## 2. Support/Slip이 없는데 실제 몸체가 붕괴하는 사례

`balanced_moderate_right_s025`: 오른쪽 balanced support, .25m/s.

| 항목 | 결과 |
|---|---:|
| 패치 접촉 | 1.508s |
| Support / Slip | 둘 다 없음 |
| old_t2 자세 악화 | 2.370s |
| Fall | 5.723s, `fallen_base_height` |
| 낙상 직전까지 tilt 최대 | 27.880° |
| 마지막 pre-fall 1초 평균 높이 / 평균 속도 | .716m / .431m/s |
| 접촉 전 1초 평균 높이 / 평균 속도 | .780m / .231m/s |
| 8초 전체 최소 pelvis 높이 | .0875m |

균일 plate에서는 셀이 같은 joint를 공유하여 spread=0이므로 Support가 없다. 그러나 몸체는 실제로 낮아지고 넘어졌다. 단순 nonfoot-contact flag만으로 낙상을 추정한 사례가 아니다.

주의: 이것은 역사적 proxy 연구에서 `BENIGN_SOFT`로 기록된 사례이며, 현재 Unified 학습 데이터의 정상 라벨 오염이라고 소급 주장하지 않는다. **현재 physical oracle을 일반 위험 의미로 확장했을 때 생기는 반례**다.

추가로 `balanced_moderate_left_s025`도 Support 없이 5.145s에 넘어지지만, 4.902s에 Slip이 발생했다. 따라서 이 왼쪽 사례를 `통합 Hazard가 전혀 없는 낙상`으로 세면 안 된다. 오른쪽 사례와 구분했다.

**판정:** 요청 2 확인. 균일 침하라는 이유로 Support가 없다는 것과 로봇이 위험하지 않다는 것은 다르다.

## 3. 자세·보행이 악화된 뒤 회복한 사례

현재 사용자가 열었던 `hbv_dsp_c_020_01`: delayed Support, .20m/s, 9초.

| 항목 | 결과 |
|---|---:|
| 패치 접촉 / Support 확정 | 1.809s / 3.662s |
| 과거 자세 악화 진단 | 5.047s |
| Support 이후 tilt 최대 | 13.321° |
| Support 이후 최소 pelvis 높이 | .731m |
| 접촉 전 평균 높이 / 평균 속도 | .782m / .198m/s |
| 마지막 1초 tilt p95 / 평균 높이 / 평균 속도 | .959° / .781m / .192m/s |
| Slip / fall | 없음 |

**판정:** 요청 3 확인. 몸체 자세가 실제로 악화됐다가 기록 마지막 1초에는 이전 높이·진행 속도와 낮은 tilt 수준으로 돌아왔다. 이 경우 `넘어지지 않았다`는 이유로 정상으로 바꾸면 실제 악화와 회복을 놓친다. 사용자가 현재 보고 있는 이 run 자체는 의미 있는 위험 징후를 가진 양성 사례로 볼 근거가 있다. 회복은 관측된 9초 안에서의 결론이며 장기 안정성을 보장하지 않는다.

더 작은 회복 사례인 `mild_medial_left_s025`도 Support 1.879s, 자세 진단 2.868s, 최대 tilt 3.115°를 거친 뒤 마지막 1초 .691°/.781m/.237m/s로 돌아왔다. 전자는 명확한 몸체 변화, 후자는 작은 transient 반응으로 심각도가 다르다.

## 4. 초기 양성 낙상 사례의 재해석과 비낙상 미회복 사례

`moderate_lateral_left_s025`는 Support 2.476s, 자세 진단 3.300s, fall flag 5.567s다. Support 이후 pre-flag 최대 tilt 8.781°, 최소 높이 .737m, 평균 속도 .207m/s로 악화된다. 그런데 이 초기 보고서의 대표 `detected fall`도 이유는 `nonfoot_surface_contact`다. 8초 전체 최소 높이 .737m, minimum pelvis-up .98828로, 이번 재현에서 몸체 낙상을 확인한 사례는 아니다. 해당 run의 접촉 body는 추가 계측하지 않았으므로 아래 boundary의 발목 접촉과 동일 원인이라고 추정하지 않는다. 과거 보고서의 저장된 fall flag 수는 유지하되, 실제 낙상 ground truth라는 해석은 제한해야 한다.

`severe_lateral_left_s015`는 과거 보고서에 mechanical recovery 193ms, non-fall로 기록되어 있다. 하지만 재현한 몸체 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| Support | 3.060s |
| Fall | 8초 내 없음 |
| Support 이후 최대 tilt / 최소 높이 | 15.733° / .625m |
| 마지막 1초 tilt p95 / 평균 높이 | 14.642° / .731m |
| 마지막 1초 평균 속도 | 1.027m/s, 명령 .15m/s |
| 같은 명령 hard-ground 마지막 1초 평균 속도 | .145m/s |

**지면 셀이 복귀했어도 로봇은 기록 마지막 구간에서 정상 보행으로 회복했다고 볼 수 없다.** 마지막 순간 tilt만 보거나 낙상이 없다는 이유로 recovery로 분류하면 잘못된다.

## 5. 최신 corpus의 fall flag에도 별도 의미 검증이 필요하다

120개 manifest 전체의 조합은 다음과 같다. 아래 fall은 아직 **원본 simulator flag**이지 전부 확인된 몸체 낙상 건수가 아니다.

| Run 라벨 | Fall flag | Boundary eligible | 수 |
|---|---|---|---:|
| NO_HAZARD | 있음 | false | 14 |
| NO_HAZARD | 없음 | true | 43 |
| HAZARD | 없음 | true | 43 |
| HAZARD | 있음 | true | 16 |
| HAZARD | 없음 | false | 4 |

14개 `NO_HAZARD + fall`은 모두 invalid로 제외되어 있다. 이를 현재 유효 정상 학습 데이터 14개의 오라벨이라고 주장하면 안 된다. 기존 suitability 검토에서 이들을 낙상으로 요약한 표현은 정확히는 fall flag로 읽어야 한다.

`hbv_sml_c_025_03`를 추가 진단한 결과:

- Support/Slip 없음, 2.157s에 `nonfoot_surface_contact` 발생.
- 해당 순간 접촉은 `terrain_transition_right`와 **right_ankle_pitch_link의 unnamed geom** 사이였다. 접촉 거리 -.000255m, pelvis 높이 .7605m다.
- 몸체 전체 9초 최소 높이는 .7557m이고 minimum pelvis-up은 .99589다. 심한 몸체 붕괴의 증거와는 다르다.
- 발목 link가 닿은 것을 fall로 취급하는 기존 정책이 적용된 것이다. 접촉이 무해하다는 결론은 아니지만, 이 flag를 실제 낙상의 확정 정답으로 곧바로 사용하면 안 된다.

이 1개 진단을 다른 13개에도 적용해 모두 false fall이라고 일반화하지 않는다. Source geometry, contact-pair, pretarget failure 여부별로 추가 분류가 필요하다. 이번에는 fall predicate나 ground/foot geom 목록을 수정하지 않았다.

근거: [contact audit](../artifacts/runs/20260907_sink_hazard_outcome_audit/contact_censor_audit.json), [fall predicate](../src/fastreflex/simulation/g1.py).

## 6. 라벨 재정의와 시뮬레이션 재설계 중 무엇이 먼저인가

**현재 Sink mechanics를 전부 폐기할 근거는 없다.** 동일 simulator에서 작은 반응, 몸체 붕괴, 회복, 미회복 사례를 실제로 만들 수 있다. 문제는 이 서로 다른 결과를 `지면 spread` 하나로 위험/정상에 대응시킨 의미와, 일부 contact censor를 fall로 묶는 정의다.

우선순위는 다음과 같다.

1. **발목/비발 접촉과 실제 낙상을 분리해 감사한다.** 위험 접촉이라는 원시 주석을 보존하고, 몸체 붕괴와 구분한다. 기존 freeze는 변경하지 않는다.
2. **물리 사건과 로봇 반응을 두 축으로 기록한다.** 물리 축은 Slip/Support/없음, 반응 축은 작은 변화/지속 악화/악화 후 회복/몸체 낙상/관측 부족 등으로 진단한다. 이 진단 분류를 바로 새 이진 정답으로 확정하지 않는다.
3. **사용자 목표에 맞는 intervention-needed 의미를 정한다.** 회복 사례를 양성에 포함할지, 경미한 변형은 어떤 근거로 제외할지 별도 정상 controls와 충분한 관측 길이를 기준으로 정한다. 미래 outcome은 오프라인 검증 주석으로만 쓰고 runtime 입력에 넣지 않는다.
4. **같은 mechanics에서 경미/심각/회복 사례를 확보하는 새 matrix를 설계한다.** balanced만 정상, lateral만 위험인 구조적 연관을 완화한다. 초기 보행과 목표 진입이 안정적인 새 물리 validation이 먼저다.
5. 그 이후 고정된 새 target으로 raw IMU6 모델을 평가한다. 과거 MoS/전체-state 단일 oracle도 검증에 실패했으므로 tilt 한 값이나 fall 하나로 정답을 치환하지 않는다.

현재 데이터가 전부 잘못된 것도, 모델 문제만인 것도 아니다. **위험 정답의 의미를 먼저 점검해야 한다는 사용자의 의심은 이번 사례 대조로 뒷받침된다.** 모델의 정상/Sink 혼동 중 얼마가 이 문제 때문인지는 새 target 감사와 독립 평가 전에는 정량화할 수 없다.

## 그림과 원자료

같은 행의 y축을 통일했다. 회색 선은 패치 접촉, 주황은 Support 확정, 빨강은 fall flag, 초록은 명령 속도다. 낙상 사례는 첫 fall까지만 표시했다.

![Sink label and robot outcome comparison](../artifacts/runs/20260907_sink_hazard_outcome_audit/sink_outcome_comparison.png)

원본 근거: [proxy sanity](20260827_sink_deformable_support_proxy_sanity.md), [walking-stability oracle failure](20260828_walking_stability_ground_truth_sanity.md), [full-state oracle result](20260828_full_state_stability_ground_truth_sanity.md), [boundary manifest](../data/raw/hazard_boundary_resolution_validation_20260904/manifest.json).
