# 보행 교란 30 ms 재라벨링·학습 파일럿

**판정: `PILOT_30MS_NOT_ACHIEVED`.** 기존 센서 원본, 기존 Hazard/Terrain 모델,
초기 연구 단계에서는 E84 펌웨어를 유지했다. 후속 사용자 요청으로 새 MLP를
비출시 CPU 시험 배포했다. 아래 초기 미배포 기록은 당시 상태이며, 최신 결과는
마지막 「새 MLP E84 시험 배포」 절을 따른다.

## 범위와 정답의 한계

이 파일럿은 보행 안정성 저하의 일반적인 정답을 검증한 실험이 아니다.
정상 TRAIN hard-ground run의 1초 이후 최대 몸통 기울기·각속도 크기를 기준으로,
둘 중 하나를 20 ms 연속 초과한 구간에 새 `PERTURBED` 라벨을 부여했다.
onset은 그 20 ms 구간 시작점으로 소급한다. 범위 안으로 돌아온 첫 샘플부터
NORMAL로 취급한다. 미확정 마지막 초과 구간, 시작 1초, 기존 censor 이후는 제외한다.
이처럼 짧은 회복 구간도 별도 사건을 만드는 정의의 적절성은 아직 검증되지 않았다.

라벨 생성은 full-state tilt/각속도를 사용하지만 모델 입력은 골반 IMU에서
계산한 기존 causal 80D뿐이다. 정규화는 TRAIN에서만 새로 fit했고, 모델은
20 ms 과거창 GRU32와 MLP를 각각 seed 17로 학습했다. 기준과 데이터셋은
모델 출력 없이 먼저 고정했고, 원본 NPZ를 덮어쓰지 않았다.

## 분리와 실행

각 실험은 TRAIN 16 / DEVELOPMENT 8 / PILOT_TEST 8 run이다.
기존 Unified TRAIN/VALIDATION 중 Ice hazard, Sand support, Sand benign,
hard normal과 Concrete/Marble 출발 조건을 사용했다. 역할 간 run ID는 분리했다.
Historical HOLDOUT이나 sealed boundary dataset은 읽지 않았다. 그러나 동일한
scenario의 인접 매개변수를 공유하는 역사적 개발 자료이므로 fresh 일반화 검증은 아니다.

1. `20260908_walking_stability_latency_pilot.yaml`: 고정 10 ms 간격으로 학습창 추출.
2. `20260908_walking_stability_phase_coverage.yaml`: 같은 TRAIN/DEVELOPMENT에서
   시간축 전체에 걸친 seeded uniform 추출로 수정. PILOT_TEST는 c20/m20에서
   c21/m21로 교체했으며 첫 실험 평가 run을 새 모델의 최종 평가에 재사용하지 않았다.

각 실험은 DEVELOPMENT로 epoch, family, 출력 threshold를 선택하고 모델·정규화·
데이터 체크섬을 동결한 뒤 PILOT_TEST 모델 추론을 실행했다. 기본 출력 지속 조건은
5 ms다. 첫 실험 뒤의 persistence sweep은 DEVELOPMENT에서만 수행했고,
오경보와 30 ms 검출률을 함께 통과한 조합은 없었다.

## 판정 규칙

onset부터 30 ms 이내에 **새 경보가 시작**되어야 검출로 센다. 교란 전부터
켜진 경보를 빠른 검출로 세지 않는다. 미검출은 null이다. 음성 라벨 샘플에서
발생한 경보 onset은 오경보로 기록하고, 경보가 회복 후 남는 시간도 별도 기록한다.
사전 gate는 사건의 90% 이상을 30 ms 내 검출, 음성시간 기준 오경보 <=1회/분,
교란 없는 run에서 경보 0회다. 30 ms는 sample-time 차이이며 host 배치 추론이나
새 모델의 E84 통신/연산 시간 측정이 아니다.

## 결과

| 실험 | 선택용 30 ms 검출 | 선택용 오경보 | 평가용 30 ms 검출 | 평가용 오경보 | 교란 없는 평가 run의 경보 |
|---|---:|---:|---:|---:|---:|
| 고정 간격 | 37/38 | 412 | 29/34 | 456 | 174 |
| 시간 위상 보완 | 36/38 | 19 | 29/33 | 18 | 0 |

두 실험의 최종 평가 run이 다르므로 최종 점수 차이를 통제된 개선량으로 해석하지
않는다. 같은 DEVELOPMENT에서는 오경보가 412→19로 감소했다.
첫 모델의 정상 DEVELOPMENT c15에서 threshold .9 초과 비율은 샘플 위치
modulo 10에 따라 0~18.4%였지만, 학습에 사용된 `sample modulo 10 = 9` 위치에서는 0%였다.
이는 고정 간격 추출이 센서/제어 주기의 다른 위상을 놓치는 문제의 근거였다.

후속 모델의 평가 결과는 29/33=87.9%, 음성 구간 경보 onset 18회였다.
교란 없는 두 hard-ground run에서는 경보가 없었다. 따라서 빠른 검출만 강조하거나
정상 run 경보 0을 전체 오경보 0으로 표현하면 안 된다. 지연/미검출·오경보 gate를
함께 충족하지 못했으므로 30 ms 목표를 달성하지 못했다.

## 산출물과 재실행

원본과 별개로 gitignored `artifacts/runs/walking_stability_latency_pilot_20260908/`와
`artifacts/runs/walking_stability_phase_coverage_20260908/`에 다음을 보존한다:
원본 run/file 체크섬을 참조하는 dataset manifest, 파생 NPZ·라벨,
새 normalizer, `models/gru.pt`, `models/mlp.pt`, 선택용 결과,
candidate freeze, 최종 결과와 평가 확률. 선택 family는 두 실험 모두 MLP,
threshold .9, persistence 5 ms다. 새 모델은 연구 파일럿이며 기존 배포 모델을 대체하지 않는다.

```bash
python scripts/fastreflex.py evaluate \
  --config configs/experiment/20260908_walking_stability_phase_coverage.yaml
```

이 명령은 기존 산출물 체크섬을 검증하고 저장된 결과만 출력한다. 목표 gate 실패를
나타내는 exit 2가 예상된다. 학습은 같은 CLI의 `train`이며 기존 output이 있으면
덮어쓰지 않고 거부한다. source commit과 working-tree dirty 상태, 실행 코드 체크섬을
manifest에 기록했다.

## 다음 연구 경계

즉시 보드 교체나 평가 데이터에 맞춘 threshold 변경은 하지 않는다. 짧은 범위
복귀를 별도 안정/불안정으로 자르는 정의의 적절성과, IMU의 긴 과거 문맥이 필요한
사건을 TRAIN/DEVELOPMENT에서 먼저 조사해야 한다. 정의 수정은 결과를 좋게 보이게
하기 위한 변경이 아니라 별도 물리 검증을 거쳐야 한다.

Terrain은 이 재학습 대상이 아니다. 현재 보드 Terrain은 유효 왼발 착지 후 50 ms
FSR 관측창과 약 2 ms CPU 계산을 사용한다. 첫 지면 접촉 이후 대기·전송까지
항상 100 ms 안이라는 보장은 없다.

## 라벨 기준 빠른 점검

기존 파생 물리 trace의 TRAIN 16 / DEVELOPMENT 8 run만 사용해 회복 조건의
민감도를 점검했다. 모델 확률과 PILOT_TEST trace는 읽지 않았다. 원본 라벨·모델·
threshold를 변경하지 않았으며 아래 결과는 성능 개선이나 새 기준 채택이 아니다.
입력 NPZ의 SHA-256을 기존 manifest와 대조했다.

시작 조건은 기존 기준 초과 20 ms와 최초 초과 onset 그대로 유지했다. 회복만
(a) 기존 상한 이내 20 ms 연속, (b) 기존 상한의 80% 이내 20 ms 연속으로
비교했다. 회복 종료는 20번째 샘플 다음 경계로 두며 소급하지 않는다. 80%는
민감도 확인용 임의 후보로, 검증되거나 선택된 물리 기준이 아니다.

| 조건 | TRAIN 사건 수 | DEVELOPMENT 사건 수 | DEVELOPMENT 비정상 라벨 시간 |
|---|---:|---:|---:|
| 기존 즉시 회복 | 62 | 38 | 12,366 ms |
| 기존 상한 이내 20 ms 회복 | 60 | 35 | 13,056 ms |
| 상한 80% 이내 20 ms 회복 | 53 | 27 | 13,847 ms |

기존 사건 사이 정상 구간은 TRAIN 50개 중 2개, DEVELOPMENT 32개 중 3개가
20 ms 미만이었다. 따라서 짧은 회복으로 사건이 쪼개지는 문제는 실제 존재하지만,
지속 조건 하나로 대부분의 사건이나 오경보가 해결된다는 근거는 없다.
회복을 엄격하게 만들면 비정상으로 세는 시간이 늘어난다. 이 때문에 같은 경보를
새 라벨로 채점한 오경보 감소를 모델 개선으로 해석하면 안 된다.
끝까지 회복이 확인되지 않은 구간은 censor에서 잘랐다. 이는 회복 관측이 아니다.

정상 기준을 만든 TRAIN hard-ground 속도는 0.1707/0.2176 m/s이고 교란 run은
0.25 m/s다. DEVELOPMENT hard-ground 0.2511 m/s 두 run은 사건 0개였으므로,
속도 차이만으로 현재 오류를 설명할 수는 없다. 그래도 속도·동작 범위가 넓은
정상 기준으로 검증된 것은 아니다.

기존 물리 사건 manifest와 새 첫 몸체 교란 onset도 대조했다.
DEVELOPMENT Ice 두 run은 기존 SLIP 사건보다 77/80 ms 뒤, Sand hazard 두 run은
기존 SUPPORT 사건보다 45/48 ms 앞에서 몸체 교란이 시작됐다. TRAIN/DEVELOPMENT
전체에서는 Sand의 시간차가 -55~+537 ms였다. 기존 발 미끄러짐/지지 변형 사건과
몸체 교란은 같은 현상을 같은 시점에 정의하지 않는다. 어느 한쪽을 안정성의
정답으로 간주하거나 이 차이를 모델 지연으로 해석하면 안 된다.

DEVELOPMENT Sand benign 두 run도 새 교란 사건이 각각 7/14개였고 기존 물리
사건은 NONE이었다. 기존 benign/VALID_STABLE은 몸체 흔들림이 없다는 보장이
아니므로 새 라벨이 틀렸다는 증거는 아니지만, 현재 정의가 회복 가능한 모래 위
보행 변동까지 포함하는지 살펴볼 우선 대상이다.

결론: 임시 몸체 교란 기준이라는 해석을 유지한다. 회복 지속 조건을 보완하는
방향은 타당하지만 수치는 아직 확정하지 않는다. 다음 물리 검증은 위 Sand benign
구간과 Ice/Sand hazard 구간의 발 접촉·미끄러짐·침하 시계열 및 자세 회복을 함께
검토해야 한다. 이번 점검은 저장된 몸체 trace와 기존 사건 요약 비교이며 새 MuJoCo
재실행이나 발 접촉별 시계열 검증, 모델 재학습, E84 30 ms 측정을 수행하지 않았다.

재현 및 상세 결과(생성 산출물, gitignored):

```bash
python artifacts/runs/walking_stability_label_audit_20260908/audit.py
```

같은 디렉터리의 `results.json`에 run별 사건 수·원래 사건 간 간격·후보 구간·
기존 물리 사건과 시간차, 원본 manifest 체크섬을 기록했다.

## 정상 허용 범위 확대의 빠른 민감도 점검

동일 TRAIN 16 / DEVELOPMENT 8 run에서 몸체 기준의 크기와 지속 조건을 비교했다.
후보는 기존 크기 ×1.25/×1.5(지속 20 ms), 기존 크기에서 지속 40 ms,
그리고 ×1.25에서 40 ms 또는 ×2에서 5 ms의 두 조건 중 하나를 만족하는 방식이다.
모델 출력으로 후보를 선택하지 않았다. 회복은 기존 즉시 회복으로 고정해 시작
조건의 효과만 비교했다. 수치는 물리적으로 검증된 권고값이 아닌 탐색 후보다.

아래는 DEVELOPMENT 각 그룹 2 run의 평가 가능 시간 중 **비정상 라벨 시간 비율**이다.
이는 모델의 경보 시간 비율이나 오경보율이 아니다.

| 후보 | Hard 정상 | Sand benign | Ice hazard | Sand support hazard | Sand hazard 첫 onset의 기존 대비 이동 |
|---|---:|---:|---:|---:|---|
| 기존 | 0% | 31.06% | 44.37% | 28.08% | 0/0 ms |
| 크기 ×1.25, 20 ms | 0% | 22.95% | 41.71% | 22.38% | +272/+300 ms |
| 크기 ×1.5, 20 ms | 0% | 15.99% | 38.08% | 19.01% | +281/+529 ms |
| 기존 크기, 40 ms | 0% | 30.31% | 44.37% | 27.67% | 0/0 ms |
| 작은 교란 40 ms / 큰 교란 5 ms | 0% | 22.69% | 41.71% | 21.87% | +272/+475 ms |

정상 허용 범위를 넓히면 Sand benign의 교란 라벨 시간은 줄지만, Sand hazard의
초기 몸체 교란도 제거된다. 첫 onset의 이동은 모델 지연이 아니라 정답 정의의
이동이다. 이 기준으로 다시 재면 30 ms 점수가 좋아 보여도 기존 물리 시작점보다
빠르게 대응했다는 뜻이 아니다. 큰 교란의 빠른 경로를 추가한 이번 후보도 이
문제를 해소하지 못했다. 40 ms 조건은 오프라인 onset을 소급하기 때문에 표상
onset이 같을 수 있지만, 이를 온라인 규칙으로 사용하면 확인에 40샘플이 필요해
최초 초과부터 30 ms 목표와 충돌한다.

TRAIN/DEVELOPMENT에서 기존 outcome VALID_FALL인 8 run은 모든 후보에 교란
사건이 남았다. 이는 run 전체 교란 소실이 없다는 뜻일 뿐, 중요한 초기 위험을
놓치지 않았다는 검증이 아니다. 넘어짐 유무를 새 라벨의 정답으로 사용하지 않았다.
Sand benign 역시 모든 순간의 안정성이 검증된 정답이 아니므로 그 그룹의 교란을
모두 제거하는 방향으로 threshold를 맞추지 않는다.

기존 동결 결과에서 현재 모델의 NORMAL 라벨 시간 중 경보 점유율은 DEVELOPMENT
177/36,063 ms = 0.49%, PILOT_TEST 222/30,565 ms = 0.73%였다. 이번에 새 후보로
모델을 다시 채점하지 않고 저장된 기존 집계만 확인했다. 이 비율은 종료 후 남는
경보도 포함하며, 기존 임시 라벨 기준이라는 한계가 있다. 적어도 현재 평가에서
항상 위험으로 출력하는 모델은 아니다.

결론: 작은 보행 변동을 허용할 필요성은 있지만, 현재 두 몸체 지표의 상한을
일괄 확대하거나 지속 시간만 늘리는 방식은 아직 채택하지 않는다. 특히 Sand에서
초기 교란 제거가 커서 물리 trace의 회복/지지 상태 검증이 우선이다. 모델·데이터셋·
E84 배포는 유지하며 재학습하지 않았다.

상세 결과는 기존 audit 디렉터리의 `onset_sensitivity.json`에 저장했다. 동일
`audit.py`로 재현한다. 기존 라벨 사건 수와 비정상 시간이 24 run 모두 정확히
재현되는 것을 확인했다. 이번 분석은 저장 trace 기반이며 새로운 발 접촉/침하
시계열 검증이나 MuJoCo 재실행은 포함하지 않는다.

## MuJoCo 발 지지·침하 시계열 재검증

DEVELOPMENT 8 run을 MuJoCo로 다시 실행하고 원본 runtime 센서와 parity를 모두
통과했다. 몸체 라벨별 발 접촉, 접촉 중 이동, 지지 손실, 지면 변형 시계열을
보존했다. 재현 스크립트는 audit 디렉터리의 `physical_replay.py`, manifest는
`physical_replay.json`, run별 trace는 `*_physical.npz`다. 원본 및 출력 체크섬을 기록했다.

| DEVELOPMENT 사례 | 몸체 교란 사건 | 기존 지지 손실 active와 겹친 사건 |
|---|---:|---:|
| Sand benign Concrete 출발 | 7 | 4 |
| Sand benign Marble 출발 | 14 | 2 |
| Sand hazard Concrete 출발 | 3 | 3 |
| Sand hazard Marble 출발 | 5 | 2 |

Sand benign의 첫 몸체 교란(1698 sample)은 두 run 모두 지지 손실 active가 없고,
그 구간 최대 지면 변형도 약 0.01 mm였다. 반면 Sand hazard 첫 교란(2993/2990)의
구간 최대 지면 변형은 약 27.95/26.82 mm였다. 이는 초기 benign 몸체 흔들림과
hazard에서 진행된 침하를 구분할 물리 근거다. 단, 구간 최대값은 사후 요약으로
시작 순간에 알 수 있는 값이 아니며 온라인 라벨 입력으로 그대로 쓰면 안 된다.

지지 손실 active 자체는 별도의 기존 진단이다. 정상 hard-ground 두 run에서도
각각 22/1 sample에 나타났고, Sand hazard Marble의 첫 몸체 교란에는 나타나지
않았다. 따라서 이 active만으로 새 위험 라벨을 대체하거나 필수 AND 조건으로
걸면 정상 오검 또는 초기 교란 누락을 만들 수 있다. Sand benign의 모든 흔들림이
안전하다는 판정도 이번 자료로 확정하지 않는다.

결론: 기존 몸체 상한만으로 안정성 저하를 정의하기 부족하다는 물리 근거를
확인했다. 침하가 진행되는 접촉 구간과 자세 회복을 함께 보는 별도 기준의 검증이
필요하며, 단일 지지 손실 플래그로 대체하는 안은 채택하지 않았다. 새 정답과
30 ms 충족 모델이 확정된 상태가 아니므로 재학습·보드 모델 교체는 수행하지 않았다.

## 실제 E84 시간 기록 연결 확인

기존 배포 Hazard 모델을 유지하고 Terrain과 함께 `uhr_sand_h_c15` 8,000샘플을
실제 E84로 전송했다. 원본 센서/Terrain parity 통과, 실제 재생 속도는 약 0.247배였다.
E84 저장소의 canonical simulation bridge에 callback 진입/전송 시작/응답 수신
monotonic 시각과 사건별 새 경보 연결 함수를 추가했다. 관련 HIL 테스트 10개 통과.

새 파일럿의 임시 라벨을 사후 연결하면 첫 onset 2993에서 E84 새 경보 3019까지
**26 ms의 시뮬레이션 시간**이었다. 같은 사건의 실제 보드 판정 시각은 onset
callback 기준 약 **100.48~103.22 ms 범위**다. 호스트와 보드 시계를 동기화한
정확한 확정 시각이 아니라 요청 시작/응답 수신으로 제한한 범위다.

나머지 두 교란 구간(3256~3719, 3816~4222)에는 새 경보가 없어 미검출(null)이다.
기존 배포 모델의 동작을 새 임시 라벨에 연결한 진단일 뿐, 새 모델 성능이나
전체 30 ms 성공이 아니다. live 화면용 기존 matched-control 진단 onset은 2991로
파일럿 pooled-normal onset 2993과 다르므로 해당 자동 로그의 28 ms와 혼동하지 않는다.

산출물은 E84 저장소 `artifacts/runs/live_simulation/stability_timing_audit.json`과
`stability_timing_events.json`이다. 새 모델의 배포/양자화 및 새 물리 기준 확정은
아직 수행하지 않았다. 현재는 측정 경로 검증과 라벨 타당성 점검을 완료한 상태다.

## 새 MLP E84 시험 배포

사용자 요청으로 현재 동결된 파일럿 MLP를 E84에 비출시 시험 배포했다. 새 물리
라벨을 확정하거나 재학습한 것이 아니다. 모델 SHA-256은
`933cfa5d40b5940d16476532b3dc3efc114bcbfa552a645afaf0134b990423bd` 그대로다.
배포 소스·가중치 변환·펌웨어/HIL은 E84 저장소에서 관리한다. 원본 Hazard 모델,
정규화 및 배포 직전 combined firmware를 보존했다.

실행은 양자화 없는 CM55 Float32 CPU 단일 MLP다. IMU6→causal80D→파일럿의
TRAIN 정규화→20×80 창→64/32/2 MLP→class1 확률 .9 이상 5샘플을 사용한다.
Terrain은 기존 E84 모델을 유지하며 안정성 출력의 입력이나 gate로 쓰지 않는다.
첫 1초는 연구 라벨 제외 구간과 맞춰 WARMUP으로 데이터를 모으되 경보를 내지
않는다. 이를 보완하기 전 정상 run의 첫 1초에 경보 105샘플이 있었으며 그 로그는
E84 `stability_handoff/startup_audit`에 남겼다. 이 구간의 정상 판별이 검증된 것은 아니다.

PC에서 C Float kernel과 연구 PyTorch를 비교했고, 실제 보드 raw 경로에서도 확률과
경보 시계열을 비교한다. 보드 프로필 확인에 실패하면 전송을 중단한다. 첫 빌드의
Makefile define 위치 문제도 이 확인에서 차단했으며, 수정 후 재빌드/플래시했다.
후속 CLI에서는 `--model stability`가 필요하며, HUD는 STABILITY PILOT의
NORMAL/UNSTABLE을 표시한다. 기존 위험 표시의 정상 지면 복귀 latch는 유지한다.

CPU 모델 계산은 약 4.4 ms로 기존 1 ms 실행 예산을 초과한다. 해당 deadline
counter를 로그에 유지하되 느린 동기식 시험에서는 전송 무결성 실패와 구분한다.
따라서 모델의 sample-time 검출과 실제 경과 시간을 구분해야 하며, 이 배포는
1 kHz 또는 실제 30 ms 달성을 주장하지 않는다. 기존 전체 연구 gate 실패 판정도 유지한다.

```bash
cd /d/shin/Infineon_FastReflex_E84
python hil/host/replay.py --mode simulation --model stability \
  --research-root /d/shin/Infineon_FastReflex \
  --run-id uhr_sand_h_c15 --port /dev/ttyACM0 --speed 0.5 \
  --output artifacts/runs/stability_handoff/sand_demo.json
```

출처와 최종 보드 결과는 E84 `artifacts/runs/stability_handoff/`의 manifest,
`*_board.json`, build/flash 로그에 기록한다. 확률·판정 parity 실패는 결과를 저장하고
CLI exit 2로 반환한다. 성공은 배포 parity의 성공이며 물리 정답/30 ms 목표 승인과 다르다.

최종 진행 상태는 **USB 연결 문제로 미완료**다. 앞선 새 모델 Sand 실측에서는
세 교란 5/7/3 ms, 오경보 1건, 확률 최대오차 약 5.6e-6와 판정 일치를 확인했다.
그러나 준비 구간 반영 후 정상 run 재검증 중 timeout이 발생했고, 이후 플래시
검증 불일치와 CMSIS-DAP USB I/O 오류가 발생했다. CPU에서 쓰지 않는 U55 초기화를
제외했으나 이를 원인 해결로 검증하지 못했다. KitProg3만 지정한 USB 소프트 리셋도
디버거 연결을 복구하지 못했다. 최종 펌웨어의 보드 탑재 상태는 미확인이며,
물리적 USB/전원 재연결 후 재플래시 검증과 정상/Sand 재실행이 남았다.
현재 결과를 최종 배포 완료나 정상 보행 전체 검증 완료로 표현하지 않는다.
