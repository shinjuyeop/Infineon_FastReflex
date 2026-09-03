"""Support-recalibrated factor-conditioned generation contracts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fastreflex.dataset.generation import _load_yaml, sha256_file
from fastreflex.dataset.sand_factor_conditioned import (
    _factor_conditioned_recalibrated_audit,
    expand_factor_conditioned_redesign,
    load_factor_conditioned_train_payload,
    validate_factor_conditioned_redesign,
    verify_factor_conditioned_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_development_support_recalibrated.yaml"
)
EXECUTION = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_development_support_recalibrated_generation.yaml"
)
DATASET = (
    ROOT / "data/raw/sand_factor_conditioned_development_support_recalibrated_20260903"
)
DESIGN_SHA256 = "b1f09effbb90313ef8c883db8e0ef376c5f4eb8e83ec66d8ed5fe52ca95ef775"
EXECUTION_SHA256 = "c4bde7adc1d4a917fe78da9b2f470c1f6155af63a06f314972609c64ef5aade4"
DATASET_FREEZE_FILE_SHA256 = (
    "fb575566574ef87bdc6ca8c161cb770c6d16e530b18fda3df0b65d213ad59922"
)
DATASET_FREEZE_SEMANTIC_SHA256 = (
    "d50602a59d196416825b09a3b49fed297ef8f2bf324eba298aadba99c988ce3b"
)


def test_frozen_design_has_exact_fresh_matrix_and_support_envelope() -> None:
    design = _load_yaml(DESIGN)
    assert sha256_file(DESIGN) == DESIGN_SHA256
    rows = expand_factor_conditioned_redesign(design)
    audit = validate_factor_conditioned_redesign(ROOT, design)
    assert audit["run_count"] == 198
    assert audit["split_counts"] == {"FACTOR_TRAIN": 132, "FACTOR_VALIDATION": 66}
    assert audit["group_counts"] == {
        "delayed_support_control": 18,
        "ordinary_support_control": 36,
        "sand_benign_mild": 108,
        "sand_benign_moderate": 36,
    }
    assert audit["unique_run_ids"] == audit["unique_scenario_signatures"] == 198
    assert audit["historical_contamination"]["exact_total"] == 0
    assert audit["historical_contamination"]["near_total"] == 0
    assert audit["historical_contamination"]["run_id_reuse_total"] == 0
    assert audit["cross_split_exact_overlap"] == 0
    assert audit["cross_split_parameter_near_overlap"] == 0

    delayed = [row for row in rows if row["group"] == "delayed_support_control"]
    assert len(delayed) == 18
    assert {row["designed_side_topology"] for row in delayed} == {"LEFT_ONLY"}
    assert {row["sink_pattern"] for row in delayed} == {"transition_left"}
    assert {row["support_pattern"] for row in delayed} == {"staged_lateral_deformable"}
    assert min(row["patch_start_x_m"] for row in delayed) == 0.324
    assert max(row["patch_start_x_m"] for row in delayed) == 0.332
    assert min(row["patch_width_m"] for row in delayed) == 0.825
    assert max(row["patch_width_m"] for row in delayed) == 0.833
    assert min(
        row["patch_start_x_m"] + row["patch_width_m"] for row in delayed
    ) == pytest.approx(1.153)
    assert max(
        row["patch_start_x_m"] + row["patch_width_m"] for row in delayed
    ) == pytest.approx(1.165)


def test_frozen_sand_domain_and_separate_reference_overlap_are_preserved() -> None:
    design = _load_yaml(DESIGN)
    rows = expand_factor_conditioned_redesign(design)
    audit = validate_factor_conditioned_redesign(ROOT, design)
    sand = [row for row in rows if row["group"].startswith("sand_benign")]
    ordinary = [row for row in rows if row["group"] == "ordinary_support_control"]
    assert len(sand) == 144
    assert len(ordinary) == 36
    assert all(row["support_pattern"] == "balanced_deformable" for row in sand)
    assert all(row["support_pattern"] == "lateral_deformable" for row in ordinary)
    assert all(
        row["sink_pattern"] == "transition_left"
        for row in sand
        if row["source_terrain"] == "concrete" and row["speed_mps"] == 0.25
    )

    contamination = audit["historical_contamination"]
    failed = next(
        path
        for path in contamination["exact_by_reference"]
        if "development_recalibrated_20260903" in path
    )
    pilot = next(
        path
        for path in contamination["exact_by_reference"]
        if "delayed_support_calibration_20260903" in path
    )
    for path in (failed, pilot):
        assert contamination["exact_by_reference"][path] == 0
        assert contamination["near_by_reference"][path] == 0
        assert contamination["run_id_reuse_by_reference"][path] == 0


def test_execution_freeze_and_protected_evidence_are_exact() -> None:
    execution = _load_yaml(EXECUTION)
    generation = execution["generation"]
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    assert sha256_file(EXECUTION) == EXECUTION_SHA256
    assert generation["source_commit"] == "5346eca3aeae1b09b6e23ffffe92c61a15c363c9"
    assert generation["planned_total_runs"] == 198
    assert generation["planned_factor_train_runs"] == 132
    assert generation["planned_factor_validation_runs"] == 66
    for artifact in generation["implementation_artifacts"]:
        assert manifest["implementation_sha256"][artifact["path"]] == artifact["sha256"]
    for artifact in generation["protected_artifacts"]:
        assert sha256_file(ROOT / artifact["path"]) == artifact["sha256"]
    guard = json.loads(
        (ROOT / generation["consumed_holdout_guard_path"]).read_text(encoding="utf-8")
    )
    assert guard["guard_after"] == guard["scientific_open_count"] == 1


def test_generated_dataset_gate_ledger_and_freeze_are_deterministic() -> None:
    design = _load_yaml(DESIGN)
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    stored_audit = json.loads(
        (DATASET / "physical_audit.json").read_text(encoding="utf-8")
    )
    recomputed = _factor_conditioned_recalibrated_audit(
        manifest, design, validate_factor_conditioned_redesign(ROOT, design)
    )
    assert recomputed == stored_audit
    assert manifest["attempted_run_count"] == manifest["run_count"] == 198
    assert manifest["adaptive_backfill_count"] == 0
    assert manifest["replacement_run_count"] == 0
    assert manifest["rerun_count"] == 0
    assert manifest["model_inference_runs"] == 0
    assert stored_audit["gate_count"] == 58
    assert stored_audit["gate_pass_count"] == 57
    assert stored_audit["gate_fail_count"] == 1
    assert stored_audit["generation_verdict"] == (
        "SAND_FACTOR_CONDITIONED_DEVELOPMENT_SUPPORT_RECALIBRATED_"
        "GENERATION_INSUFFICIENT"
    )
    assert {
        name
        for name, result in stored_audit["generation_gates"].items()
        if not result["passed"]
    } == {"yield/FACTOR_TRAIN/ordinary_support"}
    verification = verify_factor_conditioned_dataset(DATASET)
    assert verification["passed"] is True
    assert verification["run_count"] == 198
    assert verification["dataset_freeze_file_sha256"] == DATASET_FREEZE_FILE_SHA256
    assert (
        verification["dataset_freeze_semantic_sha256"] == DATASET_FREEZE_SEMANTIC_SHA256
    )
    assert all(verification["checks"].values())


def test_factor_validation_remains_sealed_before_payload_access() -> None:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    seal = json.loads((DATASET / "validation_seal.json").read_text(encoding="utf-8"))
    assert seal["status"] == "SEALED_FAILED_PHYSICAL_EVIDENCE"
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
            load_factor_conditioned_train_payload(DATASET, validation_run["run_id"])
