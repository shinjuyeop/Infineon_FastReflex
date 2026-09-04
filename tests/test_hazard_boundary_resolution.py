import json
from pathlib import Path
from unittest.mock import patch

import yaml

from fastreflex.dataset.sand_factor_conditioned import (
    BOUNDARY_VALIDATION_DATASET_ID,
    BOUNDARY_VALIDATION_SPLIT,
    expand_boundary_validation_design,
    validate_boundary_validation_design,
    verify_boundary_validation_dataset,
)
from fastreflex.features import feature_schema_hash
from fastreflex.training.hazard import (
    _load_factor_conditioned_runs,
    verify_boundary_failure_audit_result,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_CONFIG = (
    ROOT / "configs/experiment/20260904_hazard_boundary_failure_audit.yaml"
)
DESIGN_CONFIG = (
    ROOT / "configs/experiment/20260904_hazard_boundary_validation_generation.yaml"
)
RESOLUTION_CONFIG = (
    ROOT / "configs/experiment/20260904_hazard_boundary_resolution.yaml"
)
PRIOR_CONFIG = (
    ROOT / "configs/experiment/20260904_sand_factor_conditioned_model_training.yaml"
)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_train_only_root_cause_is_frozen_and_run_disjoint() -> None:
    verification = verify_boundary_failure_audit_result(ROOT, AUDIT_CONFIG)
    assert verification["primary_root_cause"] == "TRAINING_OBJECTIVE_SAMPLING_TENSION"
    split = json.loads(
        (
            ROOT
            / "artifacts/runs/20260904_hazard_boundary_failure_audit/diagnostic_split.json"
        ).read_text(encoding="utf-8")
    )
    assert split["created_before_checkpoint_diagnostics"] is True
    assert split["run_overlap"] == 0
    assert split["diagnostic_run_count"] > 0
    assert split["factor_validation_payload_reads"] == 0
    assert split["historical_holdout_payload_reads"] == 0


def test_consumed_factor_validation_cannot_reopen_before_npz_load() -> None:
    document = _load_yaml(PRIOR_CONFIG)
    artifact = ROOT / document["artifacts"]["path"]
    with patch(
        "fastreflex.training.hazard.np.load",
        side_effect=AssertionError("consumed FACTOR_VALIDATION payload read"),
    ):
        try:
            _load_factor_conditioned_runs(
                ROOT,
                document,
                "FACTOR_VALIDATION",
                candidate_freeze_path=artifact / "candidate_freeze.json",
                validation_authorization_path=(
                    artifact / "factor_validation_authorization.json"
                ),
            )
        except RuntimeError as error:
            assert "consumed" in str(error)
        else:
            raise AssertionError("consumed FACTOR_VALIDATION reopened")


def test_boundary_validation_design_is_model_blind_balanced_and_fresh() -> None:
    document = _load_yaml(DESIGN_CONFIG)
    rows = expand_boundary_validation_design(document)
    audit = validate_boundary_validation_design(ROOT, document)
    assert audit["run_count"] == 120
    assert all(row["split"] == BOUNDARY_VALIDATION_SPLIT for row in rows)
    assert audit["group_counts"] == {
        "delayed_support_control": 18,
        "ordinary_support_control": 42,
        "sand_benign_mild": 48,
        "sand_benign_moderate": 12,
    }
    assert audit["historical_contamination"]["exact_total"] == 0
    assert audit["historical_contamination"]["near_total"] == 0
    assert audit["cross_role_exact_overlap"] == 0
    assert audit["model_output_fields"] == []
    assert document["generation"]["dataset_id"] == BOUNDARY_VALIDATION_DATASET_ID
    assert document["root_cause"]["frozen_before_candidate_training"] is True


def test_historical_holdout_guard_remains_consumed() -> None:
    document = _load_yaml(DESIGN_CONFIG)
    guard_path = ROOT / document["generation"]["consumed_holdout_guard_path"]
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    assert guard["guard_after"] == 1
    assert guard["scientific_open_count"] == 1


def test_failed_physical_generation_stays_sealed_without_model_open() -> None:
    dataset = ROOT / "data/raw/hazard_boundary_resolution_validation_20260904"
    verification = verify_boundary_validation_dataset(dataset)
    assert verification["passed"] is True
    seal = json.loads((dataset / "validation_seal.json").read_text(encoding="utf-8"))
    audit = json.loads((dataset / "physical_audit.json").read_text(encoding="utf-8"))
    assert seal["status"] == "SEALED_FAILED_PHYSICAL_EVIDENCE"
    assert seal["model_inference"] is False
    assert audit["all_gates_passed"] is False
    assert audit["gate_fail_count"] == 8


def test_intervention_and_candidate_budget_froze_without_training() -> None:
    document = _load_yaml(RESOLUTION_CONFIG)
    intervention = document["selected_intervention"]
    stop = document["execution_stop"]
    assert document["failure_audit"]["primary_root_cause"] == (
        "TRAINING_OBJECTIVE_SAMPLING_TENSION"
    )
    assert intervention["coherence"] == "one_loss_weighting_change_only"
    assert intervention["hyperparameter_sweep"] is False
    assert intervention["architecture_comparator"] == "none"
    assert intervention["training_authorized"] is False
    assert stop["candidate_families_trained"] == 0
    assert stop["candidate_freezes"] == 0
    assert stop["fresh_validation_open_count"] == 0
    assert stop["validation_authorization_created"] is False
    assert stop["no_post_validation_retraining"] is True


def test_runtime_and_sensor_contract_did_not_change() -> None:
    document = _load_yaml(RESOLUTION_CONFIG)
    intervention = document["selected_intervention"]
    assert intervention["architecture"]["input_shape"] == [20, 80]
    assert intervention["architecture"]["hidden_size"] == 32
    assert intervention["architecture"]["parameters"] == 11010
    assert intervention["sensors"] == ["PELVIS_IMU6"]
    assert intervention["features"]["schema_sha256"] == feature_schema_hash()
    assert intervention["threshold"] == 0.99
    assert intervention["persistence_ms"] == 5


def test_milestone_semantic_result_hash_is_deterministic() -> None:
    from fastreflex.dataset.hazard import canonical_sha256

    path = (
        ROOT
        / "artifacts/runs/20260904_hazard_boundary_resolution/milestone_result.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = result.pop("semantic_result_sha256")
    assert canonical_sha256(result) == expected
    assert result["primary_scientific_outcome"] == "BOUNDARY_RESOLUTION_INVALID"
    assert result["fresh_validation_open_count"] == 0
