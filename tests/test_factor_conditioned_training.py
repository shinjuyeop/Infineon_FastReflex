"""Current factor-conditioned training and protected-data contracts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fastreflex.dataset.generation import canonical_sha256, sha256_file
from fastreflex.dataset.sand_factor_conditioned import verify_factor_conditioned_dataset
from fastreflex.features import feature_schema_hash
from fastreflex.training.hazard import (
    _load_factor_conditioned_runs,
    model_v2_anchor_refined_policy,
    verify_factor_conditioned_intervention_result,
)
from tests.support import REPOSITORY_ROOT as ROOT, load_json, load_yaml

CONFIG = (
    ROOT / "configs/experiment/20260904_sand_factor_conditioned_model_training.yaml"
)
DATASET = (
    ROOT / "data/raw/sand_factor_conditioned_development_controls_recalibrated_20260903"
)


@pytest.fixture(scope="module")
def document() -> dict[str, object]:
    return load_yaml(CONFIG)


def test_dataset_freeze_and_exact_split_identities(
    document: dict[str, object],
) -> None:
    dataset = document["factor_dataset"]
    verification = verify_factor_conditioned_dataset(DATASET)
    freeze = load_json(DATASET / "dataset_freeze.json")
    assert verification["passed"]
    assert verification["run_count"] == 198
    assert sha256_file(DATASET / "manifest.json") == dataset["manifest_sha256"]
    assert (
        sha256_file(DATASET / "physical_audit.json") == dataset["physical_audit_sha256"]
    )
    assert (
        sha256_file(DATASET / "validation_seal.json")
        == dataset["validation_seal_sha256"]
    )
    assert (
        freeze["FACTOR_DATASET_FREEZE_SHA"] == dataset["dataset_freeze_semantic_sha256"]
    )
    assert freeze["FACTOR_TRAIN_SPLIT_SHA"] == dataset["factor_train_split_sha256"]
    assert (
        freeze["FACTOR_VALIDATION_SPLIT_SHA"]
        == dataset["factor_validation_split_sha256"]
    )


def test_frozen_architecture_features_normalizer_and_runtime(
    document: dict[str, object],
) -> None:
    assert (
        canonical_sha256(document["architecture"])
        == document["reference_model"]["architecture_sha256"]
    )
    assert document["architecture"]["input_shape"] == [20, 80]
    assert document["architecture"]["parameters"] == 11010
    assert feature_schema_hash() == document["features"]["schema_sha256"]
    assert (
        canonical_sha256(model_v2_anchor_refined_policy())
        == document["training_protocol"]["extraction_policy_sha256"]
    )
    assert (
        sha256_file(ROOT / document["normalizer"]["path"])
        == document["normalizer"]["sha256"]
    )
    assert document["training"]["seeds"] == [20260828, 20260829, 20260830]
    assert document["runtime_decision"] == {
        "ensemble": "mean_probability_all_three_predeclared_seeds",
        "threshold": 0.99,
        "persistence_ms": 5,
    }


def test_only_authorized_train_sources_and_train_only_hnm(
    document: dict[str, object],
) -> None:
    expected = ["Unified_TRAIN", "V2_TRAIN", "FACTOR_TRAIN"]
    assert document["training_source_contract"]["authorized"] == expected
    assert document["training_protocol"]["data_sources"] == expected
    forbidden = set(document["training_source_contract"]["forbidden"])
    assert {
        "V2_VALIDATION",
        "FACTOR_VALIDATION",
        "Generalization_VALIDATION",
        "Generalization_HOLDOUT",
        "old_Sand_Discovery",
        "old_Sand_Confirmation",
    } <= forbidden
    assert document["hnm"]["source"] == "effective_TRAIN_only"
    assert document["hnm"]["rounds"] == 3
    assert document["hnm"]["validation_access"] == "prohibited"
    assert document["training_protocol"]["monitor"].endswith("TRAIN_only_partition")
    assert document["training_protocol"]["validation_early_stop"] is False


def test_factor_validation_is_inaccessible_without_freeze_or_authorization(
    document: dict[str, object], tmp_path: Path
) -> None:
    with patch(
        "fastreflex.training.hazard.np.load",
        side_effect=AssertionError("validation payload opened"),
    ):
        with pytest.raises(RuntimeError, match="sealed until candidate freeze"):
            _load_factor_conditioned_runs(ROOT, document, "FACTOR_VALIDATION")
    freeze = tmp_path / "candidate_freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "candidate_frozen_before_factor_validation": True,
                "factor_validation_evaluated": False,
            }
        ),
        encoding="utf-8",
    )
    with patch(
        "fastreflex.training.hazard.np.load",
        side_effect=AssertionError("validation payload opened"),
    ):
        with pytest.raises(RuntimeError, match="one-shot authorization"):
            _load_factor_conditioned_runs(
                ROOT, document, "FACTOR_VALIDATION", candidate_freeze_path=freeze
            )


def test_freeze_precedes_one_shot_validation_and_blocks_retraining() -> None:
    source = Path(_load_factor_conditioned_runs.__code__.co_filename).read_text(
        encoding="utf-8"
    )
    assert (
        source.index("candidate_freeze_sha = sha256_file(candidate_freeze_path)")
        < (source.index("validation_authorization = {"))
        < source.index("validation_runs, _, validation_rows")
    )
    assert '"open_count_before": 0' in source
    assert '"open_count_after": 1' in source
    assert 'artifact_path / "candidate_freeze.json"' in source
    assert '"no_retraining_after_validation": True' in source


def test_consumed_holdout_and_generalization_validation_stay_protected(
    document: dict[str, object],
) -> None:
    guard = document["historical_evidence_boundary"]
    path = ROOT / guard["holdout_guard_path"]
    assert sha256_file(path) == guard["holdout_guard_sha256"]
    state = load_json(path)
    assert state["guard_after"] == 1
    assert state["scientific_open_count"] == 1
    assert (
        "Generalization_VALIDATION"
        in document["historical_evidence_boundary"]["forbidden"]
    )


def test_completed_result_hash_chain_is_deterministic() -> None:
    result_path = (
        ROOT
        / "artifacts/runs/20260904_sand_factor_conditioned_model_training/training_result.json"
    )
    if not result_path.is_file():
        pytest.skip("scientific execution has not occurred yet")
    first = verify_factor_conditioned_intervention_result(ROOT, CONFIG)
    second = verify_factor_conditioned_intervention_result(ROOT, CONFIG)
    assert first == second
    assert first["status"] == "FACTOR_CONDITIONED_DATA_INTERVENTION_VERIFIED"
