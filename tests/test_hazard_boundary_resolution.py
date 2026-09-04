import json
from pathlib import Path
from unittest.mock import patch

import yaml

from fastreflex.dataset.sand_factor_conditioned import (
    BOUNDARY_VALIDATION_DATASET_ID,
    BOUNDARY_VALIDATION_SPLIT,
    expand_boundary_validation_design,
    validate_boundary_validation_design,
)
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
