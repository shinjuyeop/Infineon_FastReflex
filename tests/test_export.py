from __future__ import annotations

import numpy as np

from fastreflex.export import verify_release
from tests.support import REPOSITORY_ROOT as ROOT, load_json

RELEASE = ROOT / "artifacts/releases/model_v2_anchor_refined_gru20_20260902"


def test_reviewed_deployment_reference_release_is_intact() -> None:
    result = verify_release(RELEASE)
    assert result == {
        "release_id": "model_v2_anchor_refined_gru20_20260902",
        "release_manifest_sha256": (
            "d5d4e7225a35d7547e373b0ac62dbaf552d45c1a3290f214882a032355589dc7"
        ),
        "files_verified": 18,
        "status": "PASS",
    }


def test_int8_calibration_handoff_is_train_only_and_reproducible() -> None:
    manifest = load_json(RELEASE / "calibration_manifest.json")
    assert manifest["purpose"] == "formal_int8_representative_calibration_only"
    assert manifest["protected_holdout_access"] is False
    assert manifest["scientific_evidence"] is False
    assert {row["split"] for row in manifest["source_splits"]} == {
        "train",
        "V2_TRAIN",
    }
    assert manifest["selection"]["run_count"] == 442
    assert manifest["selection"]["window_count"] == 2597
    assert manifest["selection"]["model_output_used"] is False
    assert manifest["selection"]["quantization_result_used"] is False
    assert manifest["artifact"]["sha256"] == (
        "cd82304d34b2cdc60a0feb3de3e84ca7bc7f45e73223e2df389bd695cddbab5f"
    )
    with np.load(
        RELEASE / "calibration_inputs/int8_representative.npz",
        allow_pickle=False,
    ) as payload:
        assert payload["model_windows"].shape == (2597, 20, 80)
        assert payload["model_windows"].dtype == np.float32


def test_golden_evidence_is_non_protected_and_exercises_decision() -> None:
    manifest = load_json(RELEASE / "golden_manifest.json")
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


def test_deployment_float_contract_is_batch_one_and_evidence_based() -> None:
    contract = load_json(RELEASE / "float_numerical_contract.json")
    assert contract["verdict"] == "FLOAT_EXPORT_NUMERICAL_CONTRACT_RESOLVED"
    assert contract["protected_holdout_access"] is False
    assert contract["canonical_execution"]["input_shape"] == [1, 20, 80]
    assert contract["canonical_execution"]["batch_size"] == 1
    assert contract["canonical_execution"]["one_causal_endpoint_per_invocation"]
    logit_tolerance = contract["continuous_parity"]["member_logits"]
    assert logit_tolerance["absolute"] == 4e-6
    assert logit_tolerance["relative"] == 0.0
    assert "batch-size changes" in logit_tolerance["rationale"]
    evidence = contract["variability_evidence"]
    assert evidence["batch_one_repeated_exact"] is True
    assert (
        max(
            value["member_logits_max_absolute"]
            for value in evidence["batch_size_sweep"].values()
        )
        == 2.9802322387695312e-6
    )
    sensitivity = contract["threshold_sensitivity"]
    assert sensitivity["minimum_absolute_margin"] == 0.0009300009409586307
    assert sensitivity["margin_to_permitted_error_ratio"] > 400.0

    with (
        np.load(
            RELEASE / "golden_outputs/runtime_chain.npz", allow_pickle=False
        ) as historical,
        np.load(
            RELEASE / "golden_outputs/deployment_runtime_chain.npz",
            allow_pickle=False,
        ) as deployment,
    ):
        assert deployment["member_logits"].shape == (3, 121, 2)
        assert not np.array_equal(
            historical["member_logits"], deployment["member_logits"]
        )
        np.testing.assert_allclose(
            historical["member_logits"],
            deployment["member_logits"],
            atol=4e-6,
            rtol=0.0,
        )
        np.testing.assert_array_equal(
            historical["reflex_required"], deployment["reflex_required"]
        )


def test_decision_probe_freezes_inclusive_threshold_and_persistence() -> None:
    with (
        np.load(
            RELEASE / "golden_inputs/decision_probe.npz", allow_pickle=False
        ) as inputs,
        np.load(
            RELEASE / "golden_outputs/decision_probe.npz", allow_pickle=False
        ) as outputs,
    ):
        probability = inputs["ensemble_hazard_probability"]
        crossing = outputs["threshold_crossing"]
        reflex = outputs["reflex_required"]
        assert crossing[1]
        assert probability[1] == 0.99
        assert not crossing[5]
        assert reflex[10]
        assert not reflex[11]
        assert reflex[16]
