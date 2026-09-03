# INT8 calibration handoff

## Verdict

`INT8_CALIBRATION_HANDOFF_EXPORTED`

Research now exports a deterministic representative input artifact for deployment M3. It is calibration evidence only, not scientific evaluation, accuracy evidence, a model release, or permission to reinterpret the frozen unsupported generalization verdict.

## Source and selection

The source is the exact effective TRAIN used by `model_v2_anchor_refined_gru20_20260902`: 152 `unified_hazard_reflex_20260829/train` runs plus 290 valid `model_v2_hazard_reflex_20260901/V2_TRAIN` runs. Their effective identity SHA-256 is `0466ea84871d178856ffb10d8b1c0cec730286b35b589c73ff1b5fb1065aa5ab`; the frozen training config SHA-256 is `2da935d96c80452d69108ac14aa8a4df8edae297672cf25fd1945eb4deb64dbe`.

For each of 442 runs the exporter selects five evenly spaced runtime-valid endpoints and adds any valid physical precursor, Slip, and Support endpoint. Endpoint collisions are deduplicated. This produces 2,597 `[20,80] float32` causal, normalized windows: 2,210 runtime-uniform selections, 127 precursor tags, 141 Slip tags, and 119 Support tags. Selection uses neither model output nor any quantization result.

No Unified validation/HOLDOUT, V2 validation, Generalization validation/HOLDOUT, pilot, or evaluation payload is opened. The existing protected HOLDOUT path remains untouched. No training, model selection, threshold tuning, or preprocessing change occurs.

## Frozen artifact

- Artifact: `artifacts/releases/model_v2_anchor_refined_gru20_20260902/calibration_inputs/int8_representative.npz`
- SHA-256: `cd82304d34b2cdc60a0feb3de3e84ca7bc7f45e73223e2df389bd695cddbab5f`
- Size: 12,351,989 bytes
- Manifest: `calibration_manifest.json`, including all run IDs, source file hashes, endpoints, selection tags, normalizer/schema identity, and raw value distribution
- Release manifest SHA-256: `d5d4e7225a35d7547e373b0ac62dbaf552d45c1a3290f214882a032355589dc7`

The artifact deliberately retains the frozen normalized values, including TRAIN outliers. Any robust range or clipping choice is a Deployment-owned quantization policy and must be recorded separately rather than silently changing this Research-owned evidence.

## Reproduction

```bash
python scripts/fastreflex.py export --output /tmp/model_v2_anchor_refined_gru20_20260902
pytest -q tests/test_export.py
```

The canonical exporter refuses to overwrite an existing bundle, validates every frozen source checksum, writes deterministic NPZ containers, and verifies the final 18-file release contract.
