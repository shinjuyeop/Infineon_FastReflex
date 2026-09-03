"""Frozen delayed-Support physical-review and future-design contracts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fastreflex.dataset.generation import _load_yaml, canonical_sha256, sha256_file
from fastreflex.dataset.sand_calibration import (
    _historical_overlap_audit,
    validate_sand_calibration_config,
)
from fastreflex.dataset.sand_factor_conditioned import (
    validate_factor_conditioned_redesign,
)


ROOT = Path(__file__).resolve().parents[1]
FAILED_DATASET = (
    ROOT / "data/raw/sand_factor_conditioned_development_recalibrated_20260903"
)
PILOT_CONFIG = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_delayed_support_calibration.yaml"
)
PILOT_DATASET = (
    ROOT / "data/raw/sand_factor_conditioned_delayed_support_calibration_20260903"
)
REVIEW_CONFIG = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_delayed_support_physical_review.yaml"
)
FUTURE_DESIGN = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_development_support_recalibrated.yaml"
)


def _manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def test_failed_198_corpus_remains_exactly_once_model_blind_evidence() -> None:
    assert sha256_file(FAILED_DATASET / "manifest.json") == (
        "776c2a22c8963d5ddcdad49d2f36c109a21af4dfa1c0b5d2a924112b1fb6b19c"
    )
    assert sha256_file(FAILED_DATASET / "physical_audit.json") == (
        "64bcf9311946e96da2ed39714a0a1ffb3d6aa765548542cdd2edf3bf413229f8"
    )
    assert sha256_file(FAILED_DATASET / "dataset_freeze.json") == (
        "7b8ade8aff7dbb9321e1ac7a4474a892b7f58b15ed03f3093b819f3bae551cce"
    )
    manifest = _manifest(FAILED_DATASET)
    assert manifest["attempted_run_count"] == manifest["run_count"] == 198
    assert manifest["replacement_run_count"] == 0
    assert manifest["adaptive_backfill_count"] == 0
    assert manifest["rerun_count"] == 0
    assert manifest["model_inference_runs"] == 0
    assert not any(row.get("model_outputs_present") for row in manifest["runs"])


def test_latest_delayed_support_ledger_and_semantics_are_unchanged() -> None:
    rows = [
        row
        for row in _manifest(FAILED_DATASET)["runs"]
        if row["group"] == "delayed_support_control"
    ]
    assert len(rows) == 18
    assert Counter(row["objective_physical_outcome"] for row in rows) == {
        "SUPPORT": 12,
        "DUAL_HAZARD": 5,
        "INVALID": 1,
    }
    assert all(row["designed_side_topology"] == "LEFT_ONLY" for row in rows)
    assert all(row["support_event_summary"]["side"] == "LEFT_ONLY" for row in rows)
    correct = [row for row in rows if row["objective_physical_outcome"] == "SUPPORT"]
    assert all(row["target_contact_summary"]["clean_touchdowns_before_i1"] >= 2 for row in correct)
    assert all(
        row["i1_summary"]["first_sample"]
        <= row["support_event_summary"]["first_sample"]
        for row in correct
    )
    assert all(row["slip_event_summary"]["first_sample"] is None for row in correct)
    assert all(
        row["fall_censor_summary"]["censor_sample"]
        - row["support_event_summary"]["first_sample"]
        >= 1000
        for row in correct
    )


def test_observation_invalid_is_physical_fall_censor_not_horizon_shortage() -> None:
    row = next(
        row
        for row in _manifest(FAILED_DATASET)["runs"]
        if row["run_id"] == "sfcr_v_dsp_c_020_01"
    )
    assert row["invalid_reason"] == "insufficient_post_support_observation"
    assert row["support_event_summary"]["first_sample"] == 4262
    assert row["fall_censor_summary"]["first_fall_sample"] == 5118
    assert row["fall_censor_summary"]["censor_sample"] == 5118
    assert 5118 - 4262 == 856
    assert 9000 - 4262 == 4738


def test_pilot_was_frozen_fresh_model_blind_and_passed_predeclared_gates() -> None:
    document = _load_yaml(PILOT_CONFIG)
    audit = validate_sand_calibration_config(ROOT, document)
    assert sha256_file(PILOT_CONFIG) == (
        "5288e9fd011b6db03624f814b2164058ecc31126e6a9cc723db2e0dac134250d"
    )
    assert audit["run_count"] == audit["unique_run_ids"] == 24
    assert audit["unique_signatures"] == 24
    assert audit["historical_signature_overlap"] == 0
    assert audit["scenario_matrix_sha256"] == (
        "facb110351ba9e7750c2d63db0229cb1218a577c87461326e3f24dd9e8d8b3d4"
    )
    assert audit["scenario_signature_sha256"] == (
        "d8c75a44e50492dcb701740c08424bd8312e8fc7ed2990d560ba283d9ab1a23f"
    )
    near = _historical_overlap_audit(
        ROOT,
        document["calibration"]["historical_manifests"],
        document["calibration"]["scenarios"],
        document["calibration"]["near_signature_policy"],
    )
    assert near["exact_total"] == near["near_total"] == near["run_id_reuse_total"] == 0

    assert sha256_file(PILOT_DATASET / "manifest.json") == (
        "6c69a4aaecdb5e09b095976109a7a88b50802a45de7aaf02e29ef04257468850"
    )
    manifest = _manifest(PILOT_DATASET)
    assert manifest["run_count"] == 24
    assert manifest["model_blind"] is True
    assert manifest["model_inference_runs"] == 0
    assert manifest["replacement_run_count"] == 0
    assert manifest["adaptive_within_batch"] is False
    assert not any(row.get("model_outputs_present") for row in manifest["runs"])
    outcomes = Counter(row["objective_physical_outcome"] for row in manifest["runs"])
    assert outcomes == {"SUPPORT": 23, "INVALID": 1}
    for source in ("concrete", "marble"):
        for speed in (0.20, 0.25, 0.30):
            cell = [
                row
                for row in manifest["runs"]
                if row["source_terrain"] == source and row["speed_mps"] == speed
            ]
            assert len(cell) == 4
            assert sum(row["objective_physical_outcome"] == "SUPPORT" for row in cell) >= 3


def test_future_complete_design_is_fresh_balanced_and_not_generated() -> None:
    document = _load_yaml(FUTURE_DESIGN)
    assert sha256_file(FUTURE_DESIGN) == (
        "b1f09effbb90313ef8c883db8e0ef376c5f4eb8e83ec66d8ed5fe52ca95ef775"
    )
    audit = validate_factor_conditioned_redesign(ROOT, document)
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
    assert audit["scenario_matrix_sha256"] == (
        "6ed2c0c23ae036fd0bc8f523b3f254429cb1c835507d9859dadcbf82af6bd8b2"
    )
    assert audit["scenario_signature_sha256"] == (
        "0944705e3cb18ff78f4edf68573fbf56477ae9fc7cf7576a2894145549feb4be"
    )
    assert not (ROOT / document["dataset_plan"]["dataset_path"]).exists()


def test_review_readiness_hashes_and_boundaries_are_deterministic() -> None:
    document = _load_yaml(REVIEW_CONFIG)
    assert sha256_file(REVIEW_CONFIG) == (
        "9c495f9c0fe024e5889b1beaf11ae7d76c853ba85e21191a6d731763a5ff5238"
    )
    sections = {
        "saved_evidence_sha256": (
            "frozen_inputs",
            "saved_evidence_review",
            "six_miss_ledger",
            "observation_invalid_interpretation",
        ),
        "pilot_readiness_sha256": ("pilot", "stable_delayed_support_envelope"),
        "future_design_readiness_sha256": ("future_complete_corpus", "decision"),
    }
    for name, selected in sections.items():
        assert document["review_freeze"][name] == canonical_sha256(
            {key: document[key] for key in selected}
        )
    assert document["decision"]["verdict"] == (
        "DELAYED_SUPPORT_PHYSICAL_RECALIBRATION_READY"
    )
    assert document["decision"]["next_milestone"] == (
        "SAND_FACTOR_CONDITIONED_DEVELOPMENT_SUPPORT_RECALIBRATED_GENERATION"
    )
    nonzero_allowed = {
        "saved_delayed_support_records_reviewed": 18,
        "new_pilot_simulations": 24,
        "pilot_batches": 1,
    }
    for name, value in document["counters"].items():
        assert value == nonzero_allowed.get(name, 0)


def test_historical_holdout_guard_remains_exact_and_unopened() -> None:
    path = (
        ROOT
        / "artifacts/runs/20260902_model_v2_generalization_holdout_one_shot_evaluation/holdout_access_guard.json"
    )
    assert sha256_file(path) == (
        "0913da8a583cab7834590d669ba9c8d470485b8129fe81f538aba6c461613154"
    )
    guard = json.loads(path.read_text(encoding="utf-8"))
    assert guard["guard_after"] == guard["scientific_open_count"] == 1
