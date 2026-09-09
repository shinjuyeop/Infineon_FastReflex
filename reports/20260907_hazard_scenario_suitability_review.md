# Hazard 시뮬레이션 시나리오 적합성 검토

검토일: 2026-09-07 (Asia/Seoul). 검토 기준 commit: `62ae1c42fe4bcfdd57006d114d2539ed19c2f4ac`.

## 판정

현재 구성은 **고정된 G1 보행 정책에서 Slip/Support 물리 이벤트를 관측하는 제한된 연구용 benchmark로는 적합**하다. 하지만 **일반적인 위험 인지, 새로운 물리 조건에 대한 일반화, 실제 로봇에서의 위험 예방을 입증하는 데이터 구성으로는 부족**하다. 기존 성공 결과를 부정하는 결론이 아니라, 그 결과가 적용되는 범위를 제한하는 결론이다.

최근 이관한 `hazard_boundary_resolution_validation_20260904_imu6`는 전체 Hazard 학습 corpus가 아니다. 원본은 모래 경계 검증용 120-run corpus이며, 물리 gate 실패와 validation-only 자격을 그대로 보존했다. 이 ZIP만으로 정상/Slip/Support를 포괄하는 새 이진분류기의 학습·독립 성능 평가를 구성하는 것은 적합하지 않다.

사용자가 계획한 raw IMU `[B,T,6]` 모델은 기존 causal-80D GRU와 다른 실험이다. 기존 모델의 성공률과 20ms 선택 결과를 이 모델의 성능 보장으로 전용할 수 없다.

## 검토 범위와 방법

- 현재 simulator의 G1 제어, 지면 mechanics, 센서 정의, 물리 oracle, 윈도우 선택 코드를 검토했다.
- Unified, Model V2, Generalization, Sand Discovery/Confirmation, factor-conditioned intervention, 최신 boundary-resolution의 설정과 저장된 결과를 비교했다. 초기 Sink proxy와 fall-risk 실험의 라벨 차이도 확인했다.
- Unified 256개, V2 412개, boundary 120개의 manifest metadata를 직접 집계했다. 실험별 표본 수는 서로 다른 역할의 corpus를 설명하며 단순 합산하여 학습 표본 수로 제시하지 않는다.
- 이번 검토에서는 NPZ 파형을 읽지 않았으며 시뮬레이션, 학습, 모델 추론, HOLDOUT/Confirmation 재평가를 수행하지 않았다. 저장된 역사적 결과와 코드에서 확인한 사실, 새 검토자의 추론·제안을 구분한다.
- 실시간 프로세스 상태가 아니라 저장소에 정의된 시뮬레이션 구성을 검토했다. 실로봇 적합성에 대한 정량 성능은 이 검토로 산출할 수 없다.

## 1. 전체 시나리오 계보와 사용 범위

| 단계 / corpus | 구성 | 역할과 한계 |
|---|---|---|
| 초기 Slip/Sink sanity, observability | 마찰 전이, 침하, support-loss, 수직 변형 proxy 비교 | 물리·라벨 정의 탐색. 과거 Sink와 현재 Support 라벨을 동일한 클래스라고 섞으면 안 됨 |
| Dense fall-risk 연구 | 낙상까지 남은 시간으로 양성 구간 구성 | 현재 Slip/Support target과 다른 문제. 높은 window AUROC에도 모든 12개 후보의 연속 replay gate 실패 |
| Unified 256 | Ice Slip 64, Sand Support 64, Sand benign 64, hard normal 64 | TRAIN 152 / VALIDATION 52 / HOLDOUT 52. 제한된 baseline 구축 |
| Generalization 72 | 지연 Ice, Ice benign, 지연 Support, 오른발 Support, 속도별 위험·정상 | TRAIN 없음. VALIDATION 36 / 소비된 HOLDOUT 36. 일반화 실패를 드러낸 별도 평가 |
| V2 augmentation 412 | 아래 10개 family | 386 valid, 26 invalid. valid V2_TRAIN 290 + Unified TRAIN 152 = 기존 유효 학습 442 runs |
| Sand Discovery/Confirmation | 모래 정상·Support와 topology/phase 조건 조사 | 가설 탐색 및 별도 재현 검증. 소비된 Confirmation은 새 학습/최종 평가 자료가 아님 |
| Controls-recalibrated factor corpus 198 | 정상 모래와 일반/지연 Support 확대 | 196 objective-valid. FACTOR_TRAIN 132 추가 학습 후 정상 개선과 Support 회귀 발생 |
| Boundary validation 120 | 정상 의도 모래 60, 일반 Support 42, 지연 Support 18 | 102 objective-eligible, 8/19 물리 gate 실패. 학습 미실행 |

V2 manifest에 실제 기록된 설계 family 수는 다음과 같다. 아래 숫자는 valid-only가 아닌 실행된 설계 건수이다.

| V2 family | Runs |
|---|---:|
| HARD_GROUND_NORMAL_SPEED_MATRIX | 24 |
| BASELINE_IMMEDIATE_ICE_SLIP_SPEED_MATRIX | 48 |
| ONE_CONTACT_DELAYED_ICE_SLIP | 48 |
| ICE_BENIGN_CONTROL | 32 |
| ICE_NEAR_HAZARD_PRECURSOR | 44 |
| LEFT_SAND_SUPPORT_SPEED_MATRIX | 48 |
| RIGHT_SAND_SUPPORT_SPEED_MATRIX | 48 |
| DELAYED_SAND_SUPPORT_ONSET | 24 |
| STAGED_SAND_BENIGN_CONTROL | 48 |
| SPEED_STRATIFIED_SAND_BENIGN | 48 |

초기 baseline의 Ice 정상, 오른발 Support, 지연 위험, 속도 다양성 부족은 후속 corpus에서 일부 보완되었다. 따라서 전체 연구를 최신 모래 120개 또는 초기 4개 family만으로 설명하는 것도 부정확하다. 다만 추가 family의 존재가 물리 실현율이나 모델 일반화 성공을 의미하지는 않는다.

근거: [dataset contract](../docs/dataset.md), [Unified 설정](../configs/experiment/20260829_unified_hazard_reflex_system.yaml), [Generalization 설정](../configs/experiment/20260831_generalization_dataset_zero_retrain.yaml), [V2 생성 보고서](20260901_model_v2_dataset_generation.md), [factor 학습 보고서](20260904_sand_factor_conditioned_model_training.md), [초기 fall-risk 결과](20260828_dense_fall_risk_detector_poc.md).

## 2. 우선순위별 발견 사항

### P1 — 위험의 의미가 일반 위험보다 좁다

**확인 사실:** 현재 정답은 established Slip OR established Support다. Slip은 50mm/3ms, Support는 지지면 변위 차이 10mm/20ms이며, 낙상·회복 여부는 라벨의 정의가 아니다. 초기 proxy 보고서에는 balanced moderate가 나중에 넘어져도 Support 라벨을 바꾸지 않은 사례가 명시되어 있다.

**영향:** 이 모델을 일반 낙상 예측기, 전도 위험 전반의 인지기, 충돌/걸림 감지기로 부르면 목표와 정답이 어긋난다. 현재 정의에서는 균일 침하로 균형을 잃거나 장애물에 걸려도 Slip/Support oracle을 충족하지 않을 수 있다. 이는 현재 target을 기준으로 한 구현 버그가 아니라 적용 범위 문제다.

**제안:** 첫 신규 실험의 목표를 `현재 정의된 Slip/Support 이벤트 인지`로 명시한다. 낙상 예방이 목표라면 별도의 target과 정상/회복/실패 정의가 먼저 필요하다. 과거 라벨을 덮어쓰지 않는다.

근거: [물리 라벨](../docs/dataset.md#6-physical-labels), [proxy 보고서 sections 8–11](20260827_sink_deformable_support_proxy_sanity.md), [학습 양성 선택](../src/fastreflex/training/hazard.py).

### P1 — 정상·위험이 다른 지면 구조와 강하게 결합되어 있다

**확인 사실:** `balanced_deformable`의 네 geom은 한 수직 joint를 공유한다. 따라서 해당 셀들의 변위 차이 `max(d)-min(d)`는 정확히 0이다. 반면 `lateral_deformable` 등은 서로 다른 cell joint와 reference/severity 값을 사용한다. 초기 sanity에서 정상 balanced의 spread=0이 확인되었으며, 현재 boundary의 정상 의도 60개도 balanced mechanics다.

**영향에 대한 추론:** 모델이 일반적인 위험 단서보다 공유 plate 대 독립 tile의 동역학 특징을 학습할 가능성이 있다. 이것이 실제 주된 shortcut이라는 새 실험 증거는 없지만, 정상 클래스가 특정 oracle에 구조적으로 반응할 수 없다는 것은 확인된 설계 특성이다. 학습 split 내 정확도가 높아도 같은 재질·같은 mechanics에서 위험 경계를 구분하는 능력이 충분히 검증되지 않는다.

**제안:** 같은 mechanics에서도 실제 측정 결과가 정상/전조/위험으로 갈리는 연속 강도 조건을 확보한다. 균일 침하 정상과 경미한 비균일 변형 정상, 회복 가능한 작은 slip, 확정된 위험을 함께 갖추고 실제 oracle과 관측 자격으로 구분한다. 의도대로 나오지 않은 결과를 정상으로 강제하지 않는다.

주석: oracle이 contact를 잃은 개별 셀을 spread에서 제거하지 않는 것은 기존 penetration-based oracle의 실패를 피하기 위한 명시적 설계다. 이를 단순 마스킹 누락 버그로 판정하지 않았다.

근거: [shared joint mapping](../src/fastreflex/simulation/terrain.py), [spread oracle](../src/fastreflex/simulation/hazards.py), [proxy 보고서 sections 2, 6, 9, 11](20260827_sink_deformable_support_proxy_sanity.md).

### P1 — 최신 corpus는 정상 경계의 물리 실현율을 통과하지 못했다

**이번 manifest 재집계:**

| 설계 그룹 | 정상 | Support | Slip | Dual | Invalid | 총수 |
|---|---:|---:|---:|---:|---:|---:|
| mild Sand | 36 | 0 | 0 | 0 | 12 | 48 |
| moderate Sand | 7 | 0 | 3 | 0 | 2 | 12 |
| ordinary Support | 0 | 42 | 0 | 0 | 0 | 42 |
| delayed Support | 0 | 17 | 0 | 1 | 0 | 18 |
| 합계 | 43 | 59 | 3 | 1 | 14 | 120 |

정상 의도 60개 중 43개(71.7%)만 strict benign이다. Invalid 14개는 pretarget fall 6, insufficient post-target observation 8이다. `valid=true`는 106개이나 boundary objective-eligible은 102개다. 실제 Slip/Dual 4개는 물리 현상이 없다는 뜻이 아니라 이 연구의 요구 구성에 맞지 않는 사례다. binary `NO_HAZARD` 57개 중 invalid 14개를 정상으로 자동 사용하면 안 된다.

| Source / speed | Strict benign | Support | Slip/Dual | Invalid | Objective-eligible |
|---|---:|---:|---:|---:|---:|
| Concrete / .20 | 6 | 10 | 0 | 4 | 16/20 |
| Concrete / .25 | 5 | 10 | 0 | 5 | 15/20 |
| Concrete / .30 | 10 | 10 | 0 | 0 | 20/20 |
| Marble / .20 | 6 | 10 | 0 | 4 | 16/20 |
| Marble / .25 | 8 | 9 | 2 | 1 | 17/20 |
| Marble / .30 | 8 | 10 | 2 | 0 | 18/20 |

**영향:** 실패가 source/speed와 정상 의도에 집중되어 있다. 살아남은 run만 뽑아 무작위 분할하면 정상의 쉬운 일부 조건에 편향될 수 있다. 전체 120개를 학습 가능한 균형 60/60 데이터라고 주장할 수 없다.

**제안:** 기존 frozen corpus는 보존한다. 새로운 model-blind 물리 pilot에서 접근 보행, 목표 패치 접촉, 충분한 관측 길이를 먼저 검증하고 독립 validation matrix를 생성한다. 실패 후 같은 dataset ID 아래 run을 교체하는 방식은 사용하지 않는다.

근거: [boundary 결과](20260904_hazard_boundary_resolution.md), [원본 manifest](../data/raw/hazard_boundary_resolution_validation_20260904/manifest.json), [generation summary](../data/raw/hazard_boundary_resolution_validation_20260904/generation_summary.json).

### P1 — 실제 저장된 일반화 결과가 전체 적합성을 지지하지 않는다

| 저장된 평가 | 결과 | 해석 |
|---|---|---|
| Unified frozen HOLDOUT | Hazard 26/26, no-hazard 26/26, premature 0 | 제한된 기존 조건의 성공 |
| V2 Generalization HOLDOUT | Hazard 25/28, Slip 11/14, Support 14/14, no-hazard 5/8 | 전체 generalization primary gates 실패 |
| 같은 HOLDOUT의 speed-Sand | specificity 3/6 | 정상 모래 오경보가 남음 |
| Factor intervention fresh validation | Sand specificity 44/46 → 46/46; Support 16/18 → 8/18 | 정상 억제 개선과 위험 미검출 증가가 동시에 발생 |

Slip의 primary 실패 3개를 모두 무반응이라고 해석해서는 안 된다. 저장된 해석은 2개를 Ice precursor timing conflict, 1개를 +42ms late detection으로 구분한다. 조기 경보 허용 의미와 고정된 정답 시간 경계의 정합성 문제도 있다. 기존 점수는 그대로 실패로 유지한다.

또한 후속 TRAIN audit는 `TRAINING_OBJECTIVE_SAMPLING_TENSION`을 지지했다. 같은 failed model의 hidden representation에 분리 정보가 남았으며, 양성 손실에서 Slip/ordinary/delayed Support 비중은 59.8/28.5/11.7%였다. 따라서 모든 문제를 시나리오 부족이나 모델 용량 부족 하나로 단정할 수 없다.

**제안:** 데이터 재설계와 함께 run 수·유효 이벤트 수·학습 endpoint 수·클래스 내 물리 domain별 손실 비중을 각각 관리한다. 새 독립 validation 전에 구조/윈도우/threshold 탐색을 늘려 기존 실패를 덮지 않는다.

근거: [Unified 결과](20260829_unified_hazard_reflex_system.md), [Generalization 결과](20260902_model_v2_generalization_holdout_one_shot_evaluation.md), [실패 해석](20260902_model_v2_holdout_failure_interpretation.md), [factor 결과](20260904_sand_factor_conditioned_model_training.md), [boundary TRAIN audit](20260904_hazard_boundary_resolution.md).

### P2 — Run 수보다 독립된 물리 조건의 다양성이 제한적이다

**확인 사실:** 초기 Unified 위험/모래 run은 주로 .25m/s이며, 같은 mechanics 안에서 patch 시작점·폭을 바꾼다. V2는 .20/.25/.30m/s와 지연/좌우를 추가했지만 고정된 G1 모델과 동일 policy를 사용한다. 제어 코드의 command는 `(forward_speed, 0, 0)`, 초기 gait phase는 0이며 고정 관절 자세에서 시작한다. 기존 연구에서 topology와 접촉 전 지지 phase의 결합도 보고되었다.

**영향에 대한 추론:** 서로 다른 파일명·정확히 다른 geometry signature가 동역학적으로 충분히 다른 사례임을 보장하지 않는다. 실제 저장된 실패 해석에도 폭이 다른 TRAIN/HOLDOUT 사례가 표시 정밀도에서 같은 이벤트 시간과 +42ms latency를 보인 예가 있다. 정확 중복 방지는 잘 되어 있지만 그것만으로 독립 domain 일반화가 증명되지는 않는다.

**제안:** 같은 원형 시나리오의 근접 변형은 같은 split에 묶고, 별도로 speed/phase/side/mechanics 조건을 통째로 남긴 평가를 설계한다. 모델 seed 반복과 물리 realization 반복을 구분한다. 초기 상태·phase는 물리적으로 안정된 다양한 진입을 만들도록 pilot에서 설계하며 임의 phase 값만 바꾸지 않는다.

현재 corpus에는 회전·횡보·가감속, 경사·단차·장애물 걸림, 외력·하중 변화가 충분히 평가되었다는 근거가 없다. 이들은 일반 위험을 목표로 확장할 때 필요한 후보이며, 현재 Slip/Support 연구에 한꺼번에 추가해야 한다는 뜻은 아니다.

근거: [초기 coverage audit](20260831_scenario_coverage_matrix_design.md), [G1 controller](../src/fastreflex/simulation/g1.py), [factor 물리 실패 분석](20260903_data_intervention_failure_audit.md), [가까운 development analogue](20260902_model_v2_holdout_failure_interpretation.md).

### P2 — 실제 IMU/동역학과의 차이를 검증하는 실험이 부족하다

**확인 사실:** terrain class는 고정된 friction/solref/solimp를 사용하는 engineering profile이며 코드 자체도 measured material model이 아니라고 명시한다. pelvis IMU 수집은 `imu_acc`, `imu_gyro`를 그대로 float32로 기록한다. 해당 두 MJCF sensor에는 noise/delay 설정이 없고, 수집 경로에서 별도 센서 전달특성 처리를 확인하지 못했다. 다른 `secondary_imu`의 noise/cutoff 속성을 현재 pelvis 입력의 특성으로 혼동하면 안 된다.

**영향:** 1kHz 기록이라는 사실만으로 실제 1kHz IMU의 대역폭·지연·축 정렬·bias·클리핑 특성이 일치하지 않는다. 20ms 짧은 창의 모델은 이 차이에 대한 별도 검증이 필요하다. 아직 실패가 이 원인이라는 측정 결과는 없다.

**제안:** 실제 사용할 IMU와 제어계의 측정 근거로 센서 특성, 질량/하중, 마찰·강성·감쇠 변동 범위를 정한다. sensor-only corruption은 observation robustness 분석으로, 물리 변화는 simulator의 새 run으로 구분한다. 무조건 큰 Gaussian noise를 넣는 것으로 대체하지 않는다.

외부 근거는 개선 방향의 근거이며 이 G1 classifier의 성능 증거가 아니다. [Tan et al., Sim-to-Real: Learning Agile Locomotion for Quadruped Robots](https://arxiv.org/abs/1804.10332)는 system identification, actuator model, latency 및 환경 변화를 다룬다. 구체적인 IMU 정의는 [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)의 sensor 문서와 구현 버전을 함께 확인해야 한다.

로컬 근거: [terrain profiles](../src/fastreflex/simulation/terrain.py), [IMU 정의](../src/fastreflex/simulation/assets/unitree_g1/g1.xml), [센서 읽기](../src/fastreflex/simulation/g1.py).

### P2 — Window 분류와 지속 감시·개입 효과를 구분해야 한다

**확인 사실:** 기존 학습은 물리 이벤트 주변 양성과 이벤트 전 음성을 선택하고, 평가에서는 1ms 간격 연속 replay와 .99/5ms persistence를 적용한다. 저장된 초기 fall-risk 실험은 높은 window AUROC에도 연속 경보 gate를 실패했다. 현재 simulator의 보행 제어 경로에는 Hazard 출력에 따른 회복 개입이 연결되어 있지 않다.

**영향:** 임의 20/40ms 윈도우 accuracy는 정상 보행 중 오경보, 첫 이벤트 이후 재발, 회복 후 정상 복귀, 위험 예방 효과를 입증하지 않는다. 9초짜리 정상 run 몇십 개로 장시간 오경보율을 충분히 설명하기도 어렵다.

**제안:** 이벤트별 recall, 첫 경보 latency/lead, 정상 노출 시간당 false alerts, 조기 경보, 회복 후 reset과 재발을 별도 평가한다. 같은 run의 첫 이벤트와 후속 이벤트를 독립 run으로 세지 않는다. 위험 예방까지 주장할 때만 후속 과제로 동일 물리 조건에서 detector intervention on/off를 비교한다. 이번 검토에서 제어 개입을 구현하지 않았다.

근거: [현재 평가 계약](../docs/experiment_protocol.md), [윈도우 선택](../src/fastreflex/training/hazard.py), [fall-risk 연구](20260828_dense_fall_risk_detector_poc.md), [G1 제어](../src/fastreflex/simulation/g1.py).

## 3. 유지할 장점

1. 센서 입력과 simulator-only 물리 oracle을 분리한다. Terrain/I1/fall/time-to-event가 모델 입력으로 들어가지 않는 현재 경계는 유지할 가치가 있다.
2. Split을 simulation 전에 정하고 run 단위로 상속한다. TRAIN-only normalization/HNM, freeze/checksum, 소비된 HOLDOUT 보호가 갖추어져 있다.
3. 의도와 다른 실제 결과를 보존하고 invalid를 별도 기록한다. 최신 실패 corpus를 사후 교체하지 않고 학습 전에 중단한 것은 올바른 동작이다.
4. Normal Sand, Ice benign, 지연 위험과 오른발 Support를 후속 연구에서 추가했다. 초기 terrain=hazard 단순 구분을 완화하려는 설계가 존재한다.
5. 입력과 라벨의 causal regression, proxy의 수동 하중 bench, 연속 replay 평가를 이미 수행했다. 이 기반을 버리고 단순 CSV 랜덤 분할로 돌아갈 이유는 없다.

## 4. 다음 단계의 우선순위

아래는 실행하지 않은 제안이다. 고정된 과거 corpus·성능·threshold를 수정하는 지시가 아니다.

| 순서 | 작업 | 다음 단계로 넘어갈 조건 |
|---|---|---|
| 1 | Target을 Slip/Support reflex로 명시하고 precursor/확정/회복/제외 구간 정의 | run 라벨과 window 라벨을 혼동하지 않는 문서화된 정책 |
| 2 | 새 model-blind boundary physical pilot | 정상/위험 양쪽에서 접근 보행·접촉·관측 길이·실제 outcome이 사전 gate 충족 |
| 3 | 같은 mechanics의 정상/전조/위험 조건 및 좌우·phase·속도 교차 확보 | class와 mechanics/side/phase 결합을 표로 확인하고 불가능한 cell은 명시 |
| 4 | Training 데이터와 새로운 독립 validation matrix 분리 | exact/near-duplicate·원형 시나리오 그룹·물리 조건별 split 검증 |
| 5 | Raw IMU6 baseline 학습 설계 | 고정 label policy, TRAIN 정규화, domain별 endpoint/loss 비중, `[B,T,6]` 정의 |
| 6 | 전체 run replay 평가 | event recall/latency와 정상 시간당 오경보의 동시 통과; 하위 domain 결과 제시 |
| 7 | 센서/물리 robustness 및 독립된 새 최종 평가 | 실제 측정 기반 변동 범위와 사전 합의한 gate; 기존 HOLDOUT 재사용 없음 |

새 물리 validation을 먼저 만드는 순서는 현재 canonical next milestone `HAZARD_BOUNDARY_VALIDATION_PHYSICAL_REDESIGN`과 일치한다. 이번 검토로 임의의 sample 수, noise 크기, 새 threshold 또는 학습 승인을 만들어내지 않는다.

## 5. 사용자 이관 데이터에 대한 적용

- 현재 ZIP은 원본 IMU의 보존과 `[T,6]` 슬라이싱을 확인하는 이관·개발 자료로 사용할 수 있다.
- `training_eligible=false`, validation-only, failed physical evidence를 제거하여 학습 corpus로 바꾸면 안 된다. 전체 연구의 승인된 TRAIN 데이터와도 구분한다.
- 원본 `actual_hazard_label`을 전체 9초에 반복해 정답으로 만들지 않는다. 타임스탬프, censor, 반복 onset, 원본 자격을 사용해 후속 label policy를 구현한다.
- 기존 80D 모델에는 raw 값과 과거 차분·평균·분산이 들어간다. 새 raw6 모델의 적합성은 별도 validation으로 판단해야 한다.

**최종 검토 의견:** 물리 proxy 기반의 제한적 위험 이벤트 연구를 계속할 기반은 충분하다. 그러나 현 상태를 일반 위험 인지에 이미 적합한 완성 데이터셋이라고 보기는 어렵다. 가장 먼저 해결할 것은 최신 정상/위험 boundary 시나리오의 물리 유효성, target 의미, 같은 mechanics 내 경계 사례이며, 그다음에 새 raw6 모델을 평가하는 것이 타당하다.
