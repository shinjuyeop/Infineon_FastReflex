"""Ordinary-Support physical-review and future-design contracts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from fastreflex.dataset.generation import _load_yaml, canonical_sha256, sha256_file
from fastreflex.dataset.sand_factor_conditioned import (
    _factor_conditioned_recalibrated_audit,
    expand_factor_conditioned_redesign,
    validate_factor_conditioned_redesign,
)


ROOT = Path(__file__).resolve().parents[1]
REVIEW = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_ordinary_support_failure_review.yaml"
)
FUTURE_DESIGN = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_development_controls_recalibrated.yaml"
)
PRIOR_DESIGN = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_development_support_recalibrated.yaml"
)
FAILED_DATASET = (
    ROOT / "data/raw/sand_factor_conditioned_development_support_recalibrated_20260903"
)
REVIEW_SHA256 = "2e01a24771de138442e30afce3967f63dbbbc45e06e3b8057d8fb87f96c81ee5"
FUTURE_DESIGN_SHA256 = (
    "b18be44668f1d0e2c07b6a127c7fe626d42636a002ad023e48721af7c2443fb5"
)
MANIFEST_SHA256 = "bda6961f79525df237d440086057aea71a03afc5134a49c50f5e4d4a2193be67"
AUDIT_SHA256 = "f4861dc18456da28f76caf38257de03abef90e184879378bcbe844301490f9a3"
FREEZE_SHA256 = "fb575566574ef87bdc6ca8c161cb770c6d16e530b18fda3df0b65d213ad59922"
LEDGER_SHA256 = "dc5b9c2149ff5ad04686d09c6d99bce9c6067cf124eb0063e6cbd68c6aeaa696"
MISS_SHA256 = "f5dce7aeb55818a21e0b439c659652b4f765392df7532bd6ee896a71b67c95a3"
READINESS_SHA256 = "3518bb4b8cdeec8b47b59f9ed2bd8ccaadc02243c27fe17f54f04648a2c88deb"


def _ordinary_projection(row: dict[str, Any]) -> dict[str, Any]:
    contact = row["target_contact_summary"]
    fall = row["fall_censor_summary"]
    support = row["support_event_summary"]["first_sample"]
    return {
        "run_id": row["run_id"],
        "split": row["split"],
        "source_terrain": row["source_terrain"],
        "speed_mps": row["speed_mps"],
        "designed_side": row["designed_side"],
        "actual_side": row["actual_side"],
        "sink_pattern": row["sink_pattern"],
        "precontact_phase": contact["precontact_phase"],
        "entry_ms": contact["first_sample"],
        "clean_touchdowns_before_i1": contact["clean_touchdowns_before_i1"],
        "i1_ms": row["i1_summary"]["first_sample"],
        "support_ms": support,
        "slip_ms": row["slip_event_summary"]["first_sample"],
        "dual_hazard": row["objective_physical_outcome"] == "DUAL_HAZARD",
        "patch_start_x_m": row["patch_start_x_m"],
        "patch_width_m": row["patch_width_m"],
        "patch_exit_x_m": round(row["patch_start_x_m"] + row["patch_width_m"], 6),
        "support_pattern": row["support_pattern"],
        "fall_ms": fall["first_fall_sample"],
        "censor_ms": fall["censor_sample"],
        "post_support_ms": (
            None if support is None else fall["censor_sample"] - support
        ),
        "required_post_support_ms": 1000,
        "valid": row["valid"],
        "outcome": row["objective_physical_outcome"],
        "invalid_reason": row["invalid_reason"],
    }


def test_failed_198_freeze_and_ordinary_ledger_are_exact() -> None:
    manifest_path = FAILED_DATASET / "manifest.json"
    audit_path = FAILED_DATASET / "physical_audit.json"
    freeze_path = FAILED_DATASET / "dataset_freeze.json"
    assert sha256_file(manifest_path) == MANIFEST_SHA256
    assert sha256_file(audit_path) == AUDIT_SHA256
    assert sha256_file(freeze_path) == FREEZE_SHA256

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert manifest["attempted_run_count"] == manifest["run_count"] == 198
    assert manifest["adaptive_backfill_count"] == 0
    assert manifest["replacement_run_count"] == 0
    assert manifest["rerun_count"] == 0
    assert manifest["model_inference_runs"] == 0
    assert not any(row["model_outputs_present"] for row in manifest["runs"])
    assert [
        name for name, gate in audit["generation_gates"].items() if not gate["passed"]
    ] == ["yield/FACTOR_TRAIN/ordinary_support"]

    ledger = [
        _ordinary_projection(row)
        for row in manifest["runs"]
        if row["group"] == "ordinary_support_control"
    ]
    misses = [row for row in ledger if row["outcome"] != "SUPPORT"]
    assert len(ledger) == 36
    assert Counter(row["outcome"] for row in ledger) == {
        "SUPPORT": 31,
        "INVALID": 4,
        "DUAL_HAZARD": 1,
    }
    assert canonical_sha256(ledger) == LEDGER_SHA256
    assert canonical_sha256(misses) == MISS_SHA256
    assert {row["post_support_ms"] for row in misses if not row["dual_hazard"]} == {
        563,
        566,
        850,
        990,
    }


def test_support_semantics_and_no_pilot_boundary_are_unchanged() -> None:
    review = _load_yaml(REVIEW)
    prior = _load_yaml(PRIOR_DESIGN)
    future = _load_yaml(FUTURE_DESIGN)
    assert future["physical_label_contract"] == prior["physical_label_contract"]
    assert set(review["semantics"].values()) == {"unchanged"}
    assert review["pilot"]["decision"] == "NO_NEW_PILOT_REQUIRED"
    assert review["pilot"]["actual_runs"] == review["pilot"]["actual_batches"] == 0
    assert review["pilot"]["physical_only_contract"] == ("satisfied_without_execution")
    assert all(value == 0 for value in review["counters"].values() if value != 36)


def test_future_design_preserves_sand_and_delayed_support_without_overlap() -> None:
    design = _load_yaml(FUTURE_DESIGN)
    rows = expand_factor_conditioned_redesign(design)
    audit = validate_factor_conditioned_redesign(ROOT, design)
    assert sha256_file(FUTURE_DESIGN) == FUTURE_DESIGN_SHA256
    assert audit["run_count"] == 198
    assert audit["split_counts"] == {"FACTOR_TRAIN": 132, "FACTOR_VALIDATION": 66}
    assert audit["group_counts"] == {
        "delayed_support_control": 18,
        "ordinary_support_control": 36,
        "sand_benign_mild": 108,
        "sand_benign_moderate": 36,
    }
    assert audit["historical_contamination"]["exact_total"] == 0
    assert audit["historical_contamination"]["near_total"] == 0
    assert audit["historical_contamination"]["run_id_reuse_total"] == 0
    assert audit["cross_split_exact_overlap"] == 0
    assert audit["cross_split_parameter_near_overlap"] == 0

    sand = [row for row in rows if row["group"].startswith("sand_benign")]
    assert len(sand) == 144
    assert {row["support_pattern"] for row in sand} == {"balanced_deformable"}
    for split in ("FACTOR_TRAIN", "FACTOR_VALIDATION"):
        concrete_025_mild = [
            row
            for row in sand
            if row["split"] == split
            and row["group"] == "sand_benign_mild"
            and row["source_terrain"] == "concrete"
            and row["speed_mps"] == 0.25
        ]
        assert {row["sink_pattern"] for row in concrete_025_mild} == {"transition_left"}

    delayed = [row for row in rows if row["group"] == "delayed_support_control"]
    assert len(delayed) == 18
    assert {row["designed_side_topology"] for row in delayed} == {"LEFT_ONLY"}
    assert {row["sink_pattern"] for row in delayed} == {"transition_left"}
    assert {row["support_pattern"] for row in delayed} == {"staged_lateral_deformable"}
    assert all(0.324 <= row["patch_start_x_m"] <= 0.332 for row in delayed)
    assert all(0.825 <= row["patch_width_m"] <= 0.833 for row in delayed)
    assert all(
        1.153 <= row["patch_start_x_m"] + row["patch_width_m"] <= 1.165
        for row in delayed
    )


def test_future_ordinary_profiles_encode_source_speed_correction_and_gates() -> None:
    design = _load_yaml(FUTURE_DESIGN)
    rows = expand_factor_conditioned_redesign(design)
    ordinary = [row for row in rows if row["group"] == "ordinary_support_control"]
    concrete_030 = [
        row
        for row in ordinary
        if row["source_terrain"] == "concrete" and row["speed_mps"] == 0.30
    ]
    assert len(ordinary) == 36
    assert len(concrete_030) == 6
    assert {row["designed_side"] for row in concrete_030} == {"RIGHT"}
    assert {row["sink_pattern"] for row in concrete_030} == {"transition_right"}
    for source in ("concrete", "marble"):
        for speed in (0.20, 0.25, 0.30):
            if source == "concrete" and speed == 0.30:
                continue
            cell = [
                row
                for row in ordinary
                if row["source_terrain"] == source and row["speed_mps"] == speed
            ]
            assert {row["designed_side"] for row in cell} == {"LEFT", "RIGHT"}

    gates = design["generation_gates"]
    assert gates["FACTOR_TRAIN"]["ordinary_support_min"] == 22
    assert gates["FACTOR_VALIDATION"]["ordinary_support_min"] == 11
    assert gates["contamination"]["ordinary_support_Slip_plus_dual_max"] == 2
    assert gates["integrity"]["model_outputs"] == 0
    assert gates["integrity"]["replacement"] == 0
    assert gates["integrity"]["adaptive_backfill"] == 0
    assert gates["integrity"]["rerun"] == 0


def test_future_ordinary_contamination_gates_are_executable() -> None:
    design = _load_yaml(FUTURE_DESIGN)
    manifest = json.loads(
        (FAILED_DATASET / "manifest.json").read_text(encoding="utf-8")
    )
    matrix_audit = validate_factor_conditioned_redesign(ROOT, design)
    audit = _factor_conditioned_recalibrated_audit(manifest, design, matrix_audit)
    checks = audit["generation_gates"]
    assert checks["contamination/ordinary_support_slip_plus_dual"]["actual"] == 1
    assert (
        checks["contamination/FACTOR_TRAIN/ordinary_support_slip_plus_dual"]["actual"]
        == 0
    )
    assert (
        checks["contamination/FACTOR_VALIDATION/ordinary_support_slip_plus_dual"][
            "actual"
        ]
        == 1
    )


def test_review_readiness_and_holdout_hashes_are_deterministic() -> None:
    review = _load_yaml(REVIEW)
    assert sha256_file(REVIEW) == REVIEW_SHA256
    assert review["future_design"]["sha256"] == sha256_file(FUTURE_DESIGN)
    assert review["review_freeze"]["ordinary_support_ledger_sha256"] == (LEDGER_SHA256)
    assert review["review_freeze"]["five_miss_ledger_sha256"] == MISS_SHA256
    keys = (
        "review_evidence",
        "pilot",
        "ordinary_support_envelope",
        "semantics",
        "preserved_domains",
        "future_design",
        "decision",
        "historical_scientific_status",
        "counters",
    )
    assert canonical_sha256({key: review[key] for key in keys}) == READINESS_SHA256
    assert review["review_freeze"]["readiness_contract_sha256"] == READINESS_SHA256
    assert review["decision"]["verdict"] == (
        "ORDINARY_SUPPORT_PHYSICAL_RECALIBRATION_READY"
    )
    assert review["decision"]["recommended_next_milestone"] == (
        "SAND_FACTOR_CONDITIONED_DEVELOPMENT_CONTROLS_RECALIBRATED_GENERATION"
    )

    guard_record = review["frozen_inputs"]["historical_holdout_guard"]
    guard_path = ROOT / guard_record["path"]
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    assert sha256_file(guard_path) == guard_record["sha256"]
    assert guard["guard_after"] == guard["scientific_open_count"] == 1
    assert guard_record["payload_reads"] == 0
