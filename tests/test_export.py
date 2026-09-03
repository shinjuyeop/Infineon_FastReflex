from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fastreflex.export import verify_release


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/releases/model_v2_anchor_refined_gru20_20260902"


def test_reviewed_deployment_reference_release_is_intact() -> None:
    result = verify_release(RELEASE)
    assert result == {
        "release_id": "model_v2_anchor_refined_gru20_20260902",
        "release_manifest_sha256": (
            "9cbd42c95e42e90ef05f4a2ba77306a18dc3cbfa4e814c6e8432e29281a2b642"
        ),
        "files_verified": 14,
        "status": "PASS",
    }


def test_golden_evidence_is_non_protected_and_exercises_decision() -> None:
    manifest = json.loads((RELEASE / "golden_manifest.json").read_text())
    assert manifest["scientific_evidence"] is False
    assert manifest["protected_holdout_access"] is False
    assert manifest["source"]["split"] == "V2_VALIDATION"

    with np.load(
        RELEASE / "golden_outputs/runtime_chain.npz", allow_pickle=False
    ) as chain:
        assert chain["base_features"].shape == (140, 10)
        assert chain["causal_features"].shape == (140, 80)
        assert chain["model_windows"].shape == (121, 20, 80)
        assert chain["member_logits"].shape == (3, 121, 2)
        assert chain["member_hazard_probability"].shape == (3, 121)
        assert chain["reflex_required"].any()
        assert (~chain["reflex_required"]).any()


def test_decision_probe_freezes_inclusive_threshold_and_persistence() -> None:
    with np.load(
        RELEASE / "golden_inputs/decision_probe.npz", allow_pickle=False
    ) as inputs, np.load(
        RELEASE / "golden_outputs/decision_probe.npz", allow_pickle=False
    ) as outputs:
        probability = inputs["ensemble_hazard_probability"]
        crossing = outputs["threshold_crossing"]
        reflex = outputs["reflex_required"]
        assert crossing[1]
        assert probability[1] == 0.99
        assert not crossing[5]
        assert reflex[10]
        assert not reflex[11]
        assert reflex[16]
