"""Controls-recalibrated factor-conditioned generation contracts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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
    / "configs/experiment/20260903_sand_factor_conditioned_development_controls_recalibrated.yaml"
)
EXECUTION = (
    ROOT
    / "configs/experiment/20260904_sand_factor_conditioned_development_controls_recalibrated_generation.yaml"
)
DATASET = (
    ROOT
    / "data/raw/sand_factor_conditioned_development_controls_recalibrated_20260903"
)
DESIGN_SHA256 = "b18be44668f1d0e2c07b6a127c7fe626d42636a002ad023e48721af7c2443fb5"
EXECUTION_SHA256 = (
    "a9380ddb3f5649183878af7ac6b865bf6e372cbb4954d367072a1ba1366f0377"
)
MANIFEST_SHA256 = "70f850f22507384a50c81bdd065c7d485f2e37cd7e181fda20431ed2fede2d50"
DATASET_FREEZE_FILE_SHA256 = (
    "c9fee8eed1e75c23ce44c9d9fa4a9d204b150ed69e20b467d0c188e22e8194df"
)
DATASET_FREEZE_SEMANTIC_SHA256 = (
    "e397c78d19386732eb54ba388c551a8fe213a6097ba7d0819ff20f7b9b0255f4"
)


def test_frozen_design_matrix_and_all_physical_envelopes_are_exact() -> None:
    design = _load_yaml(DESIGN)
    rows = expand_factor_conditioned_redesign(design)
    audit = validate_factor_conditioned_redesign(ROOT, design)
    assert sha256_file(DESIGN) == DESIGN_SHA256
    assert audit["run_count"] == 198
    assert audit["split_counts"] == {"FACTOR_TRAIN": 132, "FACTOR_VALIDATION": 66}
    assert audit["group_counts"] == {
        "delayed_support_control": 18,
        "ordinary_support_control": 36,
        "sand_benign_mild": 108,
        "sand_benign_moderate": 36,
    }

    sand = [row for row in rows if row["group"].startswith("sand_benign")]
    delayed = [row for row in rows if row["group"] == "delayed_support_control"]
    ordinary = [row for row in rows if row["group"] == "ordinary_support_control"]
    assert {row["support_pattern"] for row in sand} == {"balanced_deformable"}
    assert {row["designed_side_topology"] for row in delayed} == {"LEFT_ONLY"}
    assert {row["sink_pattern"] for row in delayed} == {"transition_left"}
    assert {row["support_pattern"] for row in delayed} == {
        "staged_lateral_deformable"
    }
    assert all(0.324 <= row["patch_start_x_m"] <= 0.332 for row in delayed)
    assert all(0.825 <= row["patch_width_m"] <= 0.833 for row in delayed)
    assert all(
        1.153 <= row["patch_start_x_m"] + row["patch_width_m"] <= 1.165
        for row in delayed
    )

    concrete_030 = [
        row
        for row in ordinary
        if row["source_terrain"] == "concrete" and row["speed_mps"] == 0.30
    ]
    assert {row["designed_side"] for row in concrete_030} == {"RIGHT"}
    assert {row["sink_pattern"] for row in concrete_030} == {"transition_right"}
    marble_030_left = [
        row
        for row in ordinary
        if row["source_terrain"] == "marble"
        and row["speed_mps"] == 0.30
        and row["designed_side"] == "LEFT"
    ]
    assert marble_030_left
    assert min(row["patch_start_x_m"] for row in marble_030_left) >= 0.336
    concrete_020_left = [
        row
        for row in ordinary
        if row["source_terrain"] == "concrete"
        and row["speed_mps"] == 0.20
        and row["designed_side"] == "LEFT"
    ]
    assert not any(
        row["patch_start_x_m"] >= 0.349 and row["patch_width_m"] >= 0.745
        for row in concrete_020_left
    )


def test_freshness_and_anti_contamination_are_zero() -> None:
    audit = validate_factor_conditioned_redesign(ROOT, _load_yaml(DESIGN))
    historical = audit["historical_contamination"]
    assert audit["unique_run_ids"] == audit["unique_scenario_signatures"] == 198
    assert historical["exact_total"] == 0
    assert historical["near_total"] == 0
    assert historical["run_id_reuse_total"] == 0
    assert audit["cross_split_exact_overlap"] == 0
    assert audit["cross_split_parameter_near_overlap"] == 0
    failed_names = (
        "sand_factor_conditioned_development_20260903",
        "sand_factor_conditioned_development_recalibrated_20260903",
        "sand_factor_conditioned_development_support_recalibrated_20260903",
    )
    for path, count in historical["exact_by_reference"].items():
        if any(name in path for name in failed_names) or "calibration" in path:
            assert count == 0
            assert historical["near_by_reference"][path] == 0
            assert historical["run_id_reuse_by_reference"][path] == 0


def test_execution_and_protected_artifacts_remain_exact() -> None:
    execution = _load_yaml(EXECUTION)
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    generation = execution["generation"]
    assert sha256_file(EXECUTION) == EXECUTION_SHA256
    assert generation["source_commit"] == (
        "48e21858f777bac52876848ddff2df9448f08413"
    )
    assert generation["readiness_contract_sha256"] == (
        "3518bb4b8cdeec8b47b59f9ed2bd8ccaadc02243c27fe17f54f04648a2c88deb"
    )
    for artifact in generation["implementation_artifacts"]:
        assert manifest["implementation_sha256"][artifact["path"]] == artifact["sha256"]
    for artifact in generation["protected_artifacts"]:
        assert sha256_file(ROOT / artifact["path"]) == artifact["sha256"]
    guard = json.loads(
        (ROOT / generation["consumed_holdout_guard_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert guard["guard_after"] == guard["scientific_open_count"] == 1


def test_generated_gate_ledger_and_dataset_freeze_are_deterministic() -> None:
    design = _load_yaml(DESIGN)
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    stored = json.loads((DATASET / "physical_audit.json").read_text(encoding="utf-8"))
    recomputed = _factor_conditioned_recalibrated_audit(
        manifest, design, validate_factor_conditioned_redesign(ROOT, design)
    )
    assert recomputed == stored
    assert sha256_file(DATASET / "manifest.json") == MANIFEST_SHA256
    assert manifest["attempted_run_count"] == manifest["run_count"] == 198
    assert manifest["adaptive_backfill_count"] == 0
    assert manifest["replacement_run_count"] == 0
    assert manifest["rerun_count"] == 0
    assert manifest["model_inference_runs"] == 0
    assert not manifest["model_output_fields"]
    assert not any(row["model_outputs_present"] for row in manifest["runs"])
    assert stored["gate_count"] == stored["gate_pass_count"] == 61
    assert stored["gate_fail_count"] == 0
    assert stored["generation_verdict"] == (
        "SAND_FACTOR_CONDITIONED_DEVELOPMENT_CONTROLS_RECALIBRATED_GENERATION_READY"
    )
    verification = verify_factor_conditioned_dataset(DATASET)
    assert verification["passed"] is True
    assert verification["dataset_freeze_file_sha256"] == DATASET_FREEZE_FILE_SHA256
    assert (
        verification["dataset_freeze_semantic_sha256"]
        == DATASET_FREEZE_SEMANTIC_SHA256
    )
    assert all(verification["checks"].values())


def test_factor_validation_is_generated_but_model_sealed() -> None:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    seal = json.loads((DATASET / "validation_seal.json").read_text(encoding="utf-8"))
    assert seal["status"] == "SEALED_FOR_FUTURE_FACTOR_VALIDATION"
    assert seal["model_inference"] is False
    assert seal["training_use"] is False
    assert seal["hnm"] is False
    assert seal["normalized_80d_analysis"] is False
    validation_run = next(
        row for row in manifest["runs"] if row["split"] == "FACTOR_VALIDATION"
    )
    with patch(
        "fastreflex.dataset.sand_factor_conditioned.np.load",
        side_effect=AssertionError("validation payload opened"),
    ):
        try:
            load_factor_conditioned_train_payload(DATASET, validation_run["run_id"])
        except RuntimeError as exc:
            assert "FACTOR_VALIDATION is SEALED" in str(exc)
        else:
            raise AssertionError("sealed FACTOR_VALIDATION payload access succeeded")
