from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from fastreflex.dataset.generation import _load_yaml, sha256_file
from fastreflex.dataset.sand_factor_conditioned import (
    audit_factor_conditioned_physical_failure,
    expand_factor_conditioned_redesign,
    validate_factor_conditioned_redesign,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_CONFIG = ROOT / "configs/experiment/20260903_data_intervention_failure_audit.yaml"
PILOT_CONFIGS = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_physical_domain_calibration.yaml",
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_concrete_025_calibration.yaml",
)
REDESIGN_CONFIG = (
    ROOT
    / "configs/experiment/20260903_sand_factor_conditioned_physical_domain_redesign.yaml"
)


def test_failure_audit_is_complete_metadata_only_and_deterministic() -> None:
    result = audit_factor_conditioned_physical_failure(ROOT, AUDIT_CONFIG)
    assert result["run_count"] == 162
    assert result["metadata_only"] is True
    assert result["model_output_fields"] == []
    assert all(result["input_integrity"].values())
    assert result["outcome_counts"] == {
        "STRICT_SAND_BENIGN": 42,
        "SUPPORT": 39,
        "SLIP": 5,
        "DUAL_HAZARD": 2,
        "PRETARGET_FALL": 30,
        "TARGET_FOLLOWING_FALL_CENSOR": 42,
        "OTHER_INVALID": 2,
    }
    assert sum(result["outcome_counts"].values()) == 162
    assert len(result["pretarget_falls"]) == 30
    assert len(result["post_target_fall_censors"]) == 42
    assert result["post_target_taxonomy_counts"] == {
        "IMMEDIATE_TARGET_INDUCED_INSTABILITY": 24,
        "SHORT_LIVED_BENIGN_THEN_LATER_FALL": 18,
    }
    assert all(row["target_ms"] is None for row in result["pretarget_falls"])
    assert all(
        row["fall_or_censor_ms"] is not None
        for row in result["post_target_fall_censors"]
    )
    assert result == audit_factor_conditioned_physical_failure(ROOT, AUDIT_CONFIG)


def test_failed_mild_domain_was_entirely_outside_proven_envelope() -> None:
    result = audit_factor_conditioned_physical_failure(ROOT, AUDIT_CONFIG)
    relation = result["mild_relation_to_proven_envelope"]
    assert relation["STRICT_SAND_BENIGN"] == {"OUTSIDE": 21}
    assert relation["PRETARGET_FALL"] == {"OUTSIDE": 13}
    assert relation["TARGET_FOLLOWING_FALL_CENSOR"] == {"OUTSIDE": 38}
    assert (
        result["source_speed"]["FACTOR_VALIDATION/concrete/0.20"]["physical_region"]
        == "UNSTABLE"
    )
    assert (
        result["source_speed"]["FACTOR_TRAIN/concrete/0.30"]["physical_region"]
        == "STABLE"
    )


@pytest.mark.parametrize(
    ("config_path", "manifest_sha", "strict", "invalid"),
    (
        (
            PILOT_CONFIGS[0],
            "0017716c3a779e96b4ce2d0df569ce47fbfe892ab84c3daeb5025d4c857812bd",
            22,
            2,
        ),
        (
            PILOT_CONFIGS[1],
            "78377d3a3be41714e2852d7d043afd5ed04538da0a43ae4178ed92d96f033753",
            7,
            1,
        ),
    ),
)
def test_model_blind_pilot_freezes_are_exact(
    config_path: Path, manifest_sha: str, strict: int, invalid: int
) -> None:
    document = _load_yaml(config_path)
    dataset = ROOT / document["calibration"]["dataset_path"]
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert sha256_file(manifest_path) == manifest_sha
    assert manifest["run_count"] == document["calibration"]["planned_run_count"]
    assert manifest["model_blind"] is True
    assert manifest["model_inference_runs"] == 0
    assert manifest["replacement_run_count"] == 0
    assert manifest["adaptive_within_batch"] is False
    outcomes = Counter(row["objective_physical_outcome"] for row in manifest["runs"])
    assert outcomes["STRICT_BENIGN"] == strict
    assert outcomes["INVALID"] == invalid
    assert outcomes["SLIP"] == outcomes["DUAL_HAZARD"] == 0
    assert not any(row.get("model_outputs_present") for row in manifest["runs"])


def test_redesign_is_fresh_balanced_and_frozen() -> None:
    document = _load_yaml(REDESIGN_CONFIG)
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
        "6ffad518466d3082a742787199c038732dea885c7ef508b2585a1dc267e39fc3"
    )
    assert audit["scenario_signature_sha256"] == (
        "0085a9568c3b30870739792a4cf552699e2dcf4ef45f4f00c3dd4780945e86bf"
    )
    assert not (ROOT / document["dataset_plan"]["dataset_path"]).exists()


def test_redesign_preserves_factor_manifolds_without_claiming_independence() -> None:
    document = _load_yaml(REDESIGN_CONFIG)
    rows = expand_factor_conditioned_redesign(document)
    for split in ("FACTOR_TRAIN", "FACTOR_VALIDATION"):
        mild = [
            row
            for row in rows
            if row["split"] == split and row["group"] == "sand_benign_mild"
        ]
        adverse = [
            row for row in mild if row["factor_manifold_intent"] == "ADVERSE_DIRECTION"
        ]
        comparison = [
            row
            for row in mild
            if row["factor_manifold_intent"] == "COMPARISON_DIRECTION"
        ]
        expected = (42, 30) if split == "FACTOR_TRAIN" else (21, 15)
        assert (len(adverse), len(comparison)) == expected
        assert {row["source_terrain"] for row in adverse} == {"concrete", "marble"}
        assert {row["speed_mps"] for row in adverse} == {0.20, 0.25, 0.30}
        assert {row["source_terrain"] for row in comparison} == {
            "concrete",
            "marble",
        }
        assert {row["speed_mps"] for row in comparison} == {0.20, 0.25, 0.30}
        assert not any(
            row["source_terrain"] == "concrete"
            and row["speed_mps"] == 0.25
            and row["sink_pattern"] == "transition_right"
            for row in mild
        )
    assert (
        document["parameter_domain"]["phase"]["independent_phase_manipulation_claimed"]
        is False
    )
    assert document["parameter_domain"]["moderate"]["decision"] == (
        "KEEP_MODERATE_REDUCED"
    )


def test_all_research_model_and_holdout_counters_remain_zero() -> None:
    audit = _load_yaml(AUDIT_CONFIG)
    redesign = _load_yaml(REDESIGN_CONFIG)
    assert all(value == 0 for value in audit["boundaries"].values())
    assert all(value == 0 for value in redesign["boundaries"].values())
    guard = json.loads(
        (
            ROOT
            / "artifacts/runs/20260902_model_v2_generalization_holdout_one_shot_evaluation/holdout_access_guard.json"
        ).read_text(encoding="utf-8")
    )
    assert guard["guard_after"] == guard["scientific_open_count"] == 1
