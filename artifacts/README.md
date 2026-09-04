# Artifacts

학습 run, checkpoint, 임시 export 등 생성 artifact의 저장 경계다. 일반 생성
결과는 Git에 commit하지 않는다.

Directory 역할은 다음과 같다.

- `runs/`: 재현 가능한 experiment output; Git에서 제외
- `tmp/`: 중간 계산물; Git에서 제외하며 필요 없으면 삭제 가능
- `external/`: local-only external dependency 또는 reference
- `releases/`: 명시적으로 검토되어 provenance가 고정된 handoff만 version control

현재 reviewed artifact는
[`model_v2_anchor_refined_gru20_20260902`](releases/model_v2_anchor_refined_gru20_20260902)다.
세 checkpoint, normalizer, runtime contract, metrics, checksum과 golden vector를
포함하는 **deployment engineering reference**다. Release model, real-robot support,
scientific generalization support를 의미하지 않는다.

일반 run이나 checkpoint를 외부 검토자에게 전달하지 말고, 필요한 결과는
`reports/`의 reviewed report와 frozen release manifest를 통해 공유한다.
