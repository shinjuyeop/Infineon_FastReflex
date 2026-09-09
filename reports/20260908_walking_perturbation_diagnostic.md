# 보행 교란 진단과 E84 통합 첫 확인

기존 Hazard 라벨·가중치·threshold는 유지했다. Terrain은 기존 세 MLP의 Float32
가중치와 정규화를 E84 CM55로 옮겼고, Hazard는 기존 U55 INT8 프로토타입을 사용했다.
Terrain은 simulator contact로 선정한 raw FSR4 50 ms window를 입력받는다.
실제 센서 기반 착지 검출을 보드에서 구현한 결과는 아니다.

## 교란 기준

계약: `configs/experiment/20260908_walking_perturbation_diagnostic.yaml`.
같은 보행 정책·속도 0.25 m/s·8초 길이의 균일 Concrete 대조 시뮬레이션으로,
처음 1초를 제외한 pelvis tilt와 각속도 크기의 각각 최대값을 구했다.
둘 중 하나를 20 sample 연속 초과하면 그 구간 첫 샘플을 onset으로 기록한다.
모델 출력은 기준 선정에 사용하지 않았다. 이는 임시 움직임 진단이며 위험 라벨이 아니다.

Tilt 상한: 0.0284534469 rad; 각속도 상한: 0.9372617006 rad/s.

## 결과

샘플 인덱스는 0부터 시작하고 timestamp=(index+1) ms다. 표의 시간차는
시뮬레이션 시간이며 통신 왕복시간을 포함하지 않는다. 음수는 교란 onset보다
먼저 경보가 발생했음을 뜻하며, 이것만으로 예측 성능이라고 해석하지 않는다.

| 사례 | 교란 onset | 교란 확인 | E84 첫 경보 | 첫 경보 − onset |
|---|---:|---:|---:|---:|
| Slip | 1994 | 2013 | 1894 | -100 ms |
| Sink | 2982 | 3001 | 2459 | -523 ms |

정상 대조 8000샘플에서는 교란 onset 없음, 경보 0샘플, fall 없음이었다. 이 대조는 교란 기준을 만든 동일한 calibration 조건이며 독립적인 일반화·specificity 검증은 아니다.

Slip·Sink 각각 8,000샘플의 원본 센서 재현은 exact parity를 통과했다.
Terrain은 원본 PC 모델과 분류/state/update가 일치하고, 확률 오차는 사전 설정한
absolute tolerance 1e-4 이내였다. E84 Terrain 실행 시간은 약 1.97–1.98 ms다.
각 run의 최초 Hazard 경보 샘플은 Terrain 추가 전과 동일했다.
이는 두 사례의 연결 확인이며 모든 Terrain 출력이 맞다는 뜻은 아니다.

## 해석과 다음 판단

이번 두 사례에서는 현재 Hazard가 임시 몸통 교란 진단보다 먼저 반응했다.
따라서 바로 재라벨링·재학습하기보다, 다양한 정상/교란 사례에서 진단 기준의
적절성·선행 경보·미검출을 별도로 확인해야 한다. 판정 결과에 맞춰 기준을
조절하거나, 0.20배속 정도의 wall-clock 실행을 실제 실시간 성능으로 해석하지 않는다.

## 근거 위치

E84 저장소의 gitignored `artifacts/runs/live_simulation/{slip,sink,normal_control}.json`
및 `artifacts/runs/terrain_handoff/manifest.json`에 원시 보드 결과·기준·원본 모델
체크섬·가중치 export provenance가 있다. 펌웨어 build/flash log도 handoff 폴더에 보존한다.
화면은 Terrain 추정·Hazard 현재/이력·두 판단 시간으로 축소했고, render-only
표시등은 빨강=현재 경보, 노랑=이전 경보, 청록=경보 없음이다.
