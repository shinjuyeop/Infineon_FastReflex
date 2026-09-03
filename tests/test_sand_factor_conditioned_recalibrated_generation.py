"""Frozen recalibrated factor-conditioned physical-generation contracts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fastreflex.dataset.generation import _load_yaml, sha256_file
from fastreflex.dataset.sand_factor_conditioned import (
    _factor_conditioned_recalibrated_audit,
    load_factor_conditioned_train_payload,
    validate_factor_conditioned_redesign,
    verify_factor_conditioned_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_development_recalibrated_generation.yaml"
)
REDESIGN_CONFIG = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_physical_domain_redesign.yaml"
)
CONFIG_SHA256 = "bd2f75f6882ae9e5573d9854ed67628f9c9750d35b24419363fe1c88f345e889"
DATASET_FREEZE_FILE_SHA256 = (
    "7b8ade8aff7dbb9321e1ac7a4474a892b7f58b15ed03f3093b819f3bae551cce"
)
DATASET_FREEZE_SEMANTIC_SHA256 = (
    "d7a7b06095ce80e0bfdc5766e9a8265178e8ef184e0ec3251d35eab555588f84"
)


def _generation_state() -> tuple[dict, Path, dict, dict, dict]:
    config = _load_yaml(CONFIG)
    dataset = ROOT / config["generation"]["dataset_path"]
    design = _load_yaml(REDESIGN_CONFIG)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((dataset / "physical_audit.json").read_text(encoding="utf-8"))
    return config, dataset, design, manifest, audit


def test_frozen_redesign_and_exact_matrix_remain_intact() -> None:
    config, _, design, _, _ = _generation_state()
    generation = config["generation"]
    assert sha256_file(CONFIG) == CONFIG_SHA256
    assert sha256_file(REDESIGN_CONFIG) == generation["redesign_config_sha256"]
    assert (
        sha256_file(ROOT / generation["readiness_artifact_path"])
        == generation["readiness_artifact_sha256"]
    )

    matrix = validate_factor_conditioned_redesign(ROOT, design)
    assert matrix["run_count"] == 198
    assert matrix["split_counts"] == {"FACTOR_TRAIN": 132, "FACTOR_VALIDATION": 66}
    assert matrix["unique_run_ids"] == matrix["unique_scenario_signatures"] == 198
    assert (
        matrix["scenario_matrix_sha256"]
        == (generation["expected_design_freeze"]["scenario_matrix_sha256"])
    )
    assert (
        matrix["scenario_signature_sha256"]
        == (generation["expected_design_freeze"]["scenario_signature_sha256"])
    )
    assert matrix["historical_contamination"]["exact_total"] == 0
    assert matrix["historical_contamination"]["near_total"] == 0
    assert matrix["historical_contamination"]["run_id_reuse_total"] == 0
    assert matrix["cross_split_exact_overlap"] == 0
    assert matrix["cross_split_parameter_near_overlap"] == 0


def test_generation_is_exactly_once_model_blind_and_audit_is_deterministic() -> None:
    _, _, design, manifest, stored_audit = _generation_state()
    matrix = validate_factor_conditioned_redesign(ROOT, design)
    recomputed = _factor_conditioned_recalibrated_audit(manifest, design, matrix)
    assert recomputed == stored_audit
    assert manifest["attempted_run_count"] == manifest["run_count"] == 198
    assert manifest["replacement_run_count"] == 0
    assert manifest["adaptive_backfill_count"] == 0
    assert manifest["rerun_count"] == 0
    assert manifest["model_inference_runs"] == 0
    assert stored_audit["gate_count"] == 55
    assert stored_audit["gate_pass_count"] == 53
    assert stored_audit["gate_fail_count"] == 2
    assert stored_audit["generation_verdict"] == (
        "SAND_FACTOR_CONDITIONED_DEVELOPMENT_RECALIBRATED_GENERATION_INSUFFICIENT"
    )
    assert {
        name
        for name, result in stored_audit["generation_gates"].items()
        if not result["passed"]
    } == {
        "yield/FACTOR_TRAIN/delayed_support",
        "yield/FACTOR_VALIDATION/delayed_support",
    }
    assert stored_audit["pilot_exact_overlap"] == 0
    assert stored_audit["pilot_forbidden_near_overlap"] == 0


def test_failed_dataset_freeze_is_deterministic_and_complete() -> None:
    _, dataset, _, manifest, audit = _generation_state()
    verification = verify_factor_conditioned_dataset(dataset)
    assert verification["passed"] is True
    assert verification["run_count"] == 198
    assert verification["dataset_freeze_file_sha256"] == DATASET_FREEZE_FILE_SHA256
    assert (
        verification["dataset_freeze_semantic_sha256"] == DATASET_FREEZE_SEMANTIC_SHA256
    )
    assert all(verification["checks"].values())
    freeze = json.loads((dataset / "dataset_freeze.json").read_text(encoding="utf-8"))
    assert freeze["generation_verdict"] == audit["generation_verdict"]
    assert freeze["FACTOR_MANIFEST_SHA"] == sha256_file(dataset / "manifest.json")
    assert freeze["FACTOR_PHYSICAL_AUDIT_SHA"] == sha256_file(
        dataset / "physical_audit.json"
    )
    assert len(manifest["runs"]) == 198


def test_factor_validation_is_sealed_before_payload_access() -> None:
    _, dataset, _, manifest, _ = _generation_state()
    seal = json.loads((dataset / "validation_seal.json").read_text(encoding="utf-8"))
    assert seal["status"] == "SEALED_FAILED_PHYSICAL_EVIDENCE"
    assert seal["generated"] is True
    assert seal["model_inference"] is False
    assert seal["training_use"] is False
    assert seal["hnm"] is False
    validation_run = next(
        row for row in manifest["runs"] if row["split"] == "FACTOR_VALIDATION"
    )
    with patch(
        "fastreflex.dataset.sand_factor_conditioned.np.load",
        side_effect=AssertionError("validation payload opened"),
    ):
        with pytest.raises(RuntimeError, match="FACTOR_VALIDATION is SEALED"):
            load_factor_conditioned_train_payload(dataset, validation_run["run_id"])


def test_protected_artifacts_and_consumed_holdout_guard_are_unchanged() -> None:
    config, _, _, _, audit = _generation_state()
    generation = config["generation"]
    for record in generation["protected_artifacts"]:
        assert sha256_file(ROOT / record["path"]) == record["sha256"]
    guard_path = ROOT / generation["consumed_holdout_guard_path"]
    assert sha256_file(guard_path) == generation["consumed_holdout_guard_sha256"]
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    assert guard["guard_after"] == guard["scientific_open_count"] == 1
    assert audit["old_holdout_payload_reads"] == 0
    assert audit["factor_validation_model_inference"] == 0
    assert audit["factor_validation_training_use"] == 0
    assert audit["factor_validation_hnm"] == 0
