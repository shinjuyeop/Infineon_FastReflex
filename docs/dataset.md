# Dataset

## 상태

Dataset은 아직 생성되거나 확정되지 않았다. 이 문서는 향후 설계를 위한 경계만 정의한다.

## 계획

- 입력 후보: Waist/Pelvis accelerometer x/y/z 및 gyroscope x/y/z
- sampling 후보: 1 kHz
- label 후보: `NORMAL`, `SLIP`, `SINK`
- 처리 방향: Raw causal sequence, windowing, 최소 normalization

## Provenance 원칙

각 dataset release는 생성 조건, simulator/version, sensor schema, sampling rate, label 기준, run/group 식별자, split 기준을 추적할 수 있어야 한다. 동일 run 또는 group이 training과 validation에 섞이지 않도록 설계한다.

로컬 raw/processed/cache 데이터는 `data/` 경계를 사용하고 Git에 commit하지 않는다. 실제 schema와 저장 형식은 데이터 수집 구현 전에 별도 검토한다.
