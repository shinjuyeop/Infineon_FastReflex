# Documentation

[프로젝트 홈](../README.md) · [실험 보고서](../reports/README.md) · [E84 배포](https://github.com/shinjuyeop/Infineon_FastReflex_E84)

## 목적별 시작점

| 알고 싶은 것 | 문서 |
|---|---|
| 무엇을 구현했고 어디까지 검증했는가 | [README · Current Status](../README.md#current-status) |
| G1을 실행하고 Ice/Sand 장면을 재생하는 방법 | [Simulation](simulation.md) |
| 센서 → feature → 모델 → 경보의 연결 | [Architecture](architecture.md) |
| 데이터 schema, 라벨, split, provenance | [Dataset contract](dataset.md) |
| 학습·평가·HOLDOUT 접근·중단 규칙 | [Experiment protocol](experiment_protocol.md) |
| 물리 라벨, 학습 구간, 경보 조건과 검출 인정 시간의 차이 | [Label and scoring boundaries](experiment_protocol.md#label-decision-and-scoring-boundaries) |
| 일반화 실패와 후속 실험의 근거 | [Reports index](../reports/README.md) |
| 테스트 실행 조건과 알려진 오류 | [Tests](../tests/README.md) |
| README 시뮬레이션 이미지 출처 | [Media provenance](assets/README.md) |

## 다섯 분 안에 이해하기

1. README의 시뮬레이션 화면, 시스템 흐름과 [설계 판단](../README.md#engineering-decisions)을 봅니다.
2. `Current Status`에서 baseline, 일반화 평가, E84 engineering 결과를 구분합니다.
3. Architecture의 Hazard/Terrain 계약을 읽습니다. IMU 기반 경보와 FSR 기반 지형 정보는 독립적입니다.
4. 관심 있는 결과만 보고서 링크를 따라 확인합니다. 전체 실험을 날짜순으로 읽을 필요는 없습니다.

## 문서 역할

`architecture.md`, `dataset.md`, `experiment_protocol.md`는 현재 설계의 기준이며 제자리에서 갱신합니다. `reports/`의 날짜별 문서는 당시 설정과 판단을 보존합니다. 과거 실험의 구현은 보고서·config의 `source_commit`에 대응하며 현재 CLI가 모든 과거 runner를 제공하지는 않습니다.

Git에는 소스, 설정, 검토된 frozen bundle과 선정한 문서용 이미지를 포함합니다. 생성 dataset, 일반 checkpoint, full video와 임시 출력은 로컬 artifact입니다. [데이터 경계](../data/README.md)

## 용어

| 용어 | 이 프로젝트에서의 의미 |
|---|---|
| Hazard / Reflex | 모델이 반환하는 위험 경보. 회복 제어기 실행을 뜻하지 않음 |
| Terrain advisory | 위험 경보와 함께 제공하는 보조 지형 정보. Hazard 판단을 바꾸지 않음 |
| Physical oracle / GT | 시뮬레이터 물리 상태로 계산한 정답·진단. Hazard 입력에서 제외 |
| HOLDOUT | 후보와 계약을 동결한 뒤 제한적으로 여는 독립 평가 데이터 |
| Engineering reference | 배포 경로 검증용 동결 모델. 일반화·실물 로봇 검증과 별도 |
