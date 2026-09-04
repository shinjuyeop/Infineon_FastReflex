# Data

Dataset의 로컬 저장 경계다. 생성된 데이터는 source directory에 두지 않으며
`raw/`, `processed/`, `cache/`는 Git에서 제외된다.

현재 개발 workspace에는 재현과 검증에 필요한 raw dataset이 있을 수 있지만 Git
repository나 검토용 배포물에는 포함하지 않는다. Dataset ID, split, schema,
source commit, manifest checksum의 current contract는
[`docs/dataset.md`](../docs/dataset.md)를 기준으로 한다.

새 dataset을 만들 때는 다음 경계를 지킨다.

- `raw/`: simulator가 생성한 immutable run payload와 manifest
- `processed/`: 재생성 가능한 변환 결과
- `cache/`: 삭제 가능한 계산 cache
- source tree: dataset payload 저장 금지

외부에 repository를 공유할 때는 이 directory의 README만 포함하고 ignored dataset
payload는 별도로 전달하지 않는다. Dataset을 별도 배포해야 한다면 manifest,
schema, checksum, 생성 config와 source commit을 함께 제공한다.
