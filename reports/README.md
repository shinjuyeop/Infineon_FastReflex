# Reports

Dataset 분석, model 비교, validation 결과를 생성할 위치다. 보고서는 사용한 dataset revision, code revision, config, artifact를 추적할 수 있어야 한다.

현재 reviewed report는 날짜와 lowercase snake_case 이름으로 이 directory에 보존한다. Canonical current-state contract는 `docs/`에 유지하고, 이 directory에는 각 bounded dataset/experiment의 provenance와 결과만 기록한다.
