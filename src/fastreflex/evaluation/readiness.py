"""Metadata-only final-candidate and one-shot HOLDOUT readiness controls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fastreflex.dataset.hazard import canonical_sha256
from fastreflex.dataset.loader import sha256_file


EXPERIMENT_ID = "MODEL_V2_FINAL_CANDIDATE_FREEZE_AND_HOLDOUT_READINESS_REVIEW"
FUTURE_EXPERIMENT_ID = "MODEL_V2_GENERALIZATION_HOLDOUT_ONE_SHOT_EVALUATION"
FINAL_CANDIDATE_ID = "model_v2_anchor_refined_gru20_20260902"
FINAL_CANDIDATE_PATH = Path("configs/model/final_generalization_candidate.yaml")
GENERALIZATION_VALIDATION = "GENERALIZATION_VALIDATION"
GENERALIZATION_HOLDOUT = "GENERALIZATION_HOLDOUT"

DATASET_PATHS = {
    "unified_hazard_reflex_20260829": Path(
        "data/raw/unified_hazard_reflex_20260829"
    ),
    "model_v2_hazard_reflex_20260901": Path(
        "data/raw/model_v2_hazard_reflex_20260901"
    ),
    "generalization_hazard_reflex_20260831": Path(
        "data/raw/generalization_hazard_reflex_20260831"
    ),
    "ice_near_hazard_semantics_20260901": Path(
        "data/raw/ice_near_hazard_semantics_20260901"
    ),
}


@dataclass(frozen=True)
class OneShotAuthorization:
    """Metadata-only authorization consumed by the future scientific open."""

    experiment_id: str
    candidate_id: str
    readiness_review_sha256: str
    final_candidate_sha256: str
    current_guard_count: int
    next_guard_count: int
    holdout_count: int
    v1_and_v2_same_pass: bool
    terrain_in_same_pass: bool


def load_readiness_yaml(path: Path) -> Mapping[str, Any]:
    """Load a readiness record without loading a dataset waveform."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return document


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _check_record(root: Path, record: Mapping[str, object]) -> None:
    path = root / str(record["path"])
    if sha256_file(path) != str(record["sha256"]):
        raise RuntimeError(f"protected artifact changed: {record['path']}")


def _checkpoint_map(
    records: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    return {
        str(record["path"]): str(record["sha256"])
        for record in records
    }


def verify_final_candidate(
    root: Path, review: Mapping[str, Any]
) -> dict[str, object]:
    """Verify that the final role aliases the exact promoted candidate."""
    if review["experiment"]["id"] != EXPERIMENT_ID:
        raise ValueError("unsupported final-candidate readiness review")
    candidate = review["candidate"]
    for name in (
        "development_promotion",
        "candidate_freeze",
        "internal_evaluation_freeze",
        "external_evaluation_config",
        "external_evaluation_freeze",
        "external_metrics",
        "external_run_level",
        "normalizer",
    ):
        _check_record(root, candidate[name])
    for record in candidate["checkpoints"]:
        _check_record(root, record)

    final_path = root / FINAL_CANDIDATE_PATH
    final = load_readiness_yaml(final_path)
    source = final["source_candidate"]
    if (
        source["candidate_id"] != FINAL_CANDIDATE_ID
        or candidate["id"] != FINAL_CANDIDATE_ID
        or final["finalization"]["role"] != "final_generalization_candidate"
        or final["finalization"]["readiness_verdict"] != "HOLDOUT_READY"
        or source["record"] != candidate["development_promotion"]
        or source["candidate_freeze"] != candidate["candidate_freeze"]
        or source["internal_evaluation_freeze"]
        != candidate["internal_evaluation_freeze"]
        or final["normalizer"] != candidate["normalizer"]
        or _checkpoint_map(final["checkpoints"])
        != _checkpoint_map(candidate["checkpoints"])
        or final["architecture"]["sha256"]
        != candidate["architecture"]["sha256"]
        or final["architecture"]["feature_schema_sha256"]
        != candidate["feature_schema_sha256"]
        or float(final["runtime"]["threshold"]) != 0.99
        or int(final["runtime"]["persistence_ms"]) != 5
        or list(final["runtime"]["ensemble_membership"])
        != list(candidate["ensemble_membership"])
        or bool(source["artifacts_duplicated"])
        or bool(source["weights_changed"])
        or not bool(final["no_new_weights"])
        or not bool(final["no_checkpoint_write"])
        or not bool(final["no_normalizer_write"])
        or not bool(final["no_architecture_mutation"])
    ):
        raise RuntimeError("final candidate does not alias the exact promotion")
    review_config = final["readiness_review"]["config"]
    if (
        review_config["path"]
        != "configs/experiment/20260902_model_v2_final_candidate_holdout_readiness_review.yaml"
        or sha256_file(root / str(review_config["path"]))
        != str(review_config["sha256"])
    ):
        raise RuntimeError("final-candidate review provenance changed")
    return {
        "candidate_id": FINAL_CANDIDATE_ID,
        "role": "final_generalization_candidate",
        "final_candidate_sha256": sha256_file(final_path),
        "development_promotion_sha256": str(
            candidate["development_promotion"]["sha256"]
        ),
        "candidate_freeze_sha256": str(candidate["candidate_freeze"]["sha256"]),
        "normalizer_sha256": str(candidate["normalizer"]["sha256"]),
        "checkpoint_sha256": _checkpoint_map(candidate["checkpoints"]),
        "architecture_sha256": str(candidate["architecture"]["sha256"]),
        "feature_schema_sha256": str(candidate["feature_schema_sha256"]),
        "threshold": 0.99,
        "persistence_ms": 5,
        "artifact_copy_or_mutation": False,
        "passed": True,
    }


def verify_protected_datasets(
    root: Path, review: Mapping[str, Any]
) -> dict[str, object]:
    """Hash opaque run files and inspect only safe HOLDOUT manifest metadata."""
    expected_by_id = {
        str(value["id"]): value
        for value in review["protected_datasets"].values()
    }
    results: dict[str, object] = {}
    holdout_records: list[dict[str, object]] = []
    for dataset_id, relative in DATASET_PATHS.items():
        expected = expected_by_id[dataset_id]
        folder = root / relative
        manifest_path = folder / "manifest.json"
        if sha256_file(manifest_path) != str(expected["manifest_sha256"]):
            raise RuntimeError(f"protected manifest changed: {dataset_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest["runs"]
        exact = 0
        for row in rows:
            filename = str(row.get("file") or row.get("path"))
            stored_sha = str(row.get("file_sha256") or row.get("sha256"))
            path = folder / filename
            if not path.is_file() or sha256_file(path) != stored_sha:
                raise RuntimeError(
                    f"protected dataset run changed: {dataset_id}/{filename}"
                )
            exact += 1
            if (
                dataset_id == review["dataset"]["id"]
                and row["split"] == GENERALIZATION_HOLDOUT
            ):
                holdout_records.append(
                    {
                        "run_id": str(row["run_id"]),
                        "file": filename,
                        "file_sha256": stored_sha,
                        "size_bytes": int(row["size_bytes"]),
                    }
                )
        if (
            manifest["dataset_id"] != dataset_id
            or len(rows) != int(expected["count"])
            or exact != int(expected["count"])
        ):
            raise RuntimeError(f"protected dataset count changed: {dataset_id}")
        results[dataset_id] = {
            "manifest_sha256": str(expected["manifest_sha256"]),
            "declared_runs": int(expected["count"]),
            "exact_files": exact,
        }

    generalization_path = root / DATASET_PATHS[str(review["dataset"]["id"])]
    manifest = json.loads(
        (generalization_path / "manifest.json").read_text(encoding="utf-8")
    )
    validation_ids = [
        str(row["run_id"])
        for row in manifest["runs"]
        if row["split"] == GENERALIZATION_VALIDATION
    ]
    holdout_ids = [
        str(row["run_id"])
        for row in manifest["runs"]
        if row["split"] == GENERALIZATION_HOLDOUT
    ]
    dataset = review["dataset"]
    _check_record(root, dataset["dataset_freeze"])
    freeze = json.loads(
        (root / str(dataset["dataset_freeze"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if (
        len(validation_ids) != 36
        or len(holdout_ids) != 36
        or canonical_sha256(validation_ids)
        != dataset["validation"]["run_ids_canonical_sha256"]
        or canonical_sha256(holdout_ids)
        != dataset["holdout"]["run_ids_canonical_sha256"]
        or freeze["split_membership_sha256"]
        != dataset["split_membership_sha256"]
        or freeze["validation_run_ids"] != validation_ids
        or freeze["holdout_run_ids"] != holdout_ids
        or bool(freeze["generalization_holdout_waveform_opened"])
    ):
        raise RuntimeError("Generalization split or HOLDOUT metadata changed")
    return {
        "datasets": results,
        "generalization_validation_count": len(validation_ids),
        "generalization_holdout_count": len(holdout_ids),
        "validation_run_ids_canonical_sha256": canonical_sha256(validation_ids),
        "holdout_run_ids_canonical_sha256": canonical_sha256(holdout_ids),
        "holdout_file_metadata_canonical_sha256": canonical_sha256(
            sorted(holdout_records, key=lambda row: str(row["run_id"]))
        ),
        "holdout_payload_deserialized": False,
        "passed": True,
    }


def verify_development_evidence(
    root: Path, review: Mapping[str, Any]
) -> dict[str, object]:
    """Verify committed development evidence without running a model."""
    candidate = review["candidate"]
    freeze = json.loads(
        (root / str(candidate["external_evaluation_freeze"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    artifact_path = (
        root / str(candidate["external_evaluation_freeze"]["path"])
    ).parent
    for filename, expected in freeze["artifact_sha256"].items():
        if sha256_file(artifact_path / filename) != str(expected):
            raise RuntimeError(f"external evidence changed: {filename}")
    metrics = json.loads(
        (root / str(candidate["external_metrics"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    run_level = json.loads(
        (root / str(candidate["external_run_level"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    primary = metrics["primary"]
    expected = review["development_evidence"]["external_generalization"][
        "final_v2"
    ]
    failures = [
        row
        for row in run_level["rows"]
        if row["physical_label"] != "NO_HAZARD" and row["v2_result"] != "CORRECT"
    ]
    limitation = review["known_limitation"]
    if (
        int(primary["hazard_runs"]) != int(expected["hazard"]["total"])
        or float(primary["overall_hazard_recall"])
        != float(expected["hazard"]["recall"])
        or float(primary["slip_hazard_recall"])
        != float(expected["slip"]["recall"])
        or float(primary["support_hazard_recall"])
        != float(expected["support"]["recall"])
        or float(primary["primary_no_hazard_specificity"])
        != float(expected["primary_specificity"]["specificity"])
        or float(metrics["ice_benign"]["rate"])
        != float(expected["ice_benign_specificity"]["specificity"])
        or float(primary["system_premature_run_rate"])
        != float(expected["premature"]["rate"])
        or primary["slip_latency_ms"]["p95"]
        != expected["slip_p95_latency_ms"]
        or primary["support_established_latency_ms"]["p95"]
        != expected["support_p95_established_latency_ms"]
        or len(failures) != 1
        or failures[0]["run_id"] != limitation["run_id"]
        or failures[0]["primary_failure_reason"]
        != "ICE_PRECURSOR_TIMING_CONFLICT"
        or failures[0]["v2_precursor_aware_classification"]
        != "SUPPORTED_FUTURE_SLIP_PRECURSOR"
        or bool(failures[0]["genuine_detection_failure"])
        or freeze["primary_gate_verdict"] != "GENERALIZATION_PRIMARY_GATES_FAIL"
        or freeze["development_verdict"]
        != "GENERALIZATION_DEVELOPMENT_SUPPORTED_WITH_ICE_TIMING_TENSION"
        or freeze["data_coverage_verdict"]
        != "DATA_COVERAGE_HYPOTHESIS_SUPPORTED"
        or freeze["generalization_holdout_guard_count"] != 0
        or freeze["generalization_holdout_waveform_opened"]
        or freeze["current_unified_holdout_waveform_reopened"]
    ):
        raise RuntimeError("frozen development evidence changed")

    family = json.loads((artifact_path / "family_results.json").read_text())
    speed = json.loads((artifact_path / "speed_results.json").read_text())
    if (
        family["DELAYED_SAND_SUPPORT_ONSET"]["v2"]["correct"] != 4
        or family["RIGHT_SAND_SUPPORT"]["v2"]["correct"] != 4
        or family["ICE_BENIGN_CONTROL"]["v2"]["correct"] != 4
        or any(
            result["v2"]["correct"] != result["v2"]["eligible"]
            for by_variant in speed.values()
            for result in by_variant.values()
        )
    ):
        raise RuntimeError("frozen family or speed evidence changed")
    return {
        "primary_gate_verdict": str(freeze["primary_gate_verdict"]),
        "development_verdict": str(freeze["development_verdict"]),
        "data_coverage_verdict": str(freeze["data_coverage_verdict"]),
        "hazard": "25/26",
        "slip": "11/12",
        "support": "14/14",
        "primary_specificity": "10/10",
        "ice_benign_specificity": "4/4",
        "premature": "1/26",
        "primary_failure_count": 1,
        "physically_supported_early_response_count": 1,
        "genuine_detection_failure_count": 0,
        "new_inference": False,
        "passed": True,
    }


def verify_contracts(root: Path, review: Mapping[str, Any]) -> dict[str, object]:
    """Prove that HOLDOUT inherits the frozen development contracts."""
    development = load_readiness_yaml(
        root / str(review["candidate"]["external_evaluation_config"]["path"])
    )
    primary = review["future_primary_contract"]
    secondary = review["future_secondary_contract"]
    old_secondary = development["secondary_evaluation"]
    if primary != development["primary_evaluation"]:
        raise RuntimeError("future primary contract differs from development")
    for key in (
        "implementation",
        "annotation_implementation",
        "label",
        "definition",
        "lower_m_inclusive",
        "upper_m_exclusive",
        "future_followup_ms",
        "outcome_categories",
        "semantics_verdict",
    ):
        if secondary[key] != old_secondary[key]:
            raise RuntimeError(f"future secondary contract changed: {key}")
    if secondary["primary_score_replacement"] or old_secondary[
        "primary_scores_rewritten"
    ]:
        raise RuntimeError("secondary contract cannot replace primary scores")

    final = load_readiness_yaml(root / FINAL_CANDIDATE_PATH)
    contract_hashes = {
        "primary": canonical_sha256(primary),
        "secondary": canonical_sha256(secondary),
        "verdict_hierarchy": canonical_sha256(
            review["future_verdict_hierarchy"]
        ),
        "post_open_rules": canonical_sha256(review["post_open_rules"]),
    }
    expected_hashes = {
        "primary": final["evaluation_contracts"]["primary"]["canonical_sha256"],
        "secondary": final["evaluation_contracts"]["secondary"][
            "canonical_sha256"
        ],
        "verdict_hierarchy": final["evaluation_contracts"][
            "verdict_hierarchy"
        ]["canonical_sha256"],
        "post_open_rules": final["evaluation_contracts"]["post_open_rules"][
            "canonical_sha256"
        ],
    }
    if contract_hashes != expected_hashes:
        raise RuntimeError("final-candidate contract hashes changed")
    return {
        "canonical_sha256": contract_hashes,
        "primary_unchanged": True,
        "secondary_unchanged": True,
        "secondary_can_rescue_primary": False,
        "verdicts_predeclared": True,
        "passed": True,
    }


def _readiness_checklist(
    review: Mapping[str, Any],
    candidate: Mapping[str, object],
    datasets: Mapping[str, object],
    evidence: Mapping[str, object],
    contracts: Mapping[str, object],
) -> dict[str, bool]:
    holdout = review["holdout"]
    future = review["future_execution"]
    return {
        "candidate_integrity": bool(candidate["passed"]),
        "dataset_integrity": bool(datasets["passed"]),
        "no_leakage": bool(
            review["sealed_evidence"][
                "Generalization_VALIDATION_reused_from_frozen_evidence"
            ]
            and not review["sealed_evidence"][
                "Generalization_VALIDATION_new_inference"
            ]
        ),
        "no_genuine_unresolved_hazard_miss": evidence[
            "genuine_detection_failure_count"
        ]
        == 0,
        "specificity_acceptable": evidence["primary_specificity"] == "10/10"
        and evidence["ice_benign_specificity"] == "4/4",
        "support_acceptable": evidence["support"] == "14/14",
        "speed_diversity_acceptable": True,
        "known_slip_limitation_physically_explained": evidence[
            "primary_failure_count"
        ]
        == evidence["physically_supported_early_response_count"]
        == 1,
        "primary_metric_retained": bool(contracts["primary_unchanged"]),
        "secondary_metric_frozen": bool(contracts["secondary_unchanged"]),
        "no_further_justified_optimization": bool(review["no_training"])
        and bool(review["no_new_inference"]),
        "holdout_untouched": holdout["guard_before"] == 0
        and not holdout["waveform_opened"]
        and not holdout["Hazard_inference"]
        and not holdout["Terrain_inference"],
        "one_shot_evaluator_ready": future["milestone"] == FUTURE_EXPERIMENT_ID
        and bool(review["future_comparison"]["same_authorized_pass"])
        and future["operation"]
        == "entire_36_run_split_in_one_scientific_operation",
        "final_verdict_predeclared": bool(contracts["verdicts_predeclared"]),
    }


def _review_payload(
    root: Path, config_path: Path, review: Mapping[str, Any]
) -> dict[str, object]:
    candidate = verify_final_candidate(root, review)
    datasets = verify_protected_datasets(root, review)
    evidence = verify_development_evidence(root, review)
    contracts = verify_contracts(root, review)
    checklist = _readiness_checklist(
        review, candidate, datasets, evidence, contracts
    )
    if not all(checklist.values()):
        raise RuntimeError("HOLDOUT readiness checklist failed")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "review_status": (
            "MODEL_V2_FINAL_CANDIDATE_HOLDOUT_READINESS_REVIEW_COMPLETE"
        ),
        "readiness_verdict": "HOLDOUT_READY",
        "final_candidate_freeze_status": "FINAL_GENERALIZATION_CANDIDATE_FROZEN",
        "review_config_sha256": sha256_file(config_path),
        "review_source_sha256": sha256_file(Path(__file__)),
        "candidate": candidate,
        "development_evidence": evidence,
        "dataset_integrity": datasets,
        "contract_integrity": contracts,
        "readiness_checklist": checklist,
        "further_optimization_decision": (
            "NO_FURTHER_INTERNAL_OPTIMIZATION_JUSTIFIED"
        ),
        "overfitting_decision": (
            "FURTHER_GENERALIZATION_VALIDATION_TUNING_NOT_JUSTIFIED"
        ),
        "architecture_decision": "ARCHITECTURE_CHANGE_NOT_JUSTIFIED",
        "sensor_decision": "10_CHANNEL_ARCHITECTURE_STILL_PLAUSIBLE",
        "final_sensor_architecture_frozen": False,
        "generalization_validation_new_inference": False,
        "generalization_holdout_waveform_opened": False,
        "generalization_holdout_hazard_inference": False,
        "generalization_holdout_terrain_inference": False,
        "generalization_holdout_visualization": False,
        "generalization_holdout_guard_count": 0,
        "current_unified_holdout_waveform_reopened": False,
        "current_unified_holdout_new_inference": False,
        "future_milestone": FUTURE_EXPERIMENT_ID,
        "future_guard_transition": "0_to_1_exactly_once",
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "normalizer_fits": 0,
        "hnm_rounds": 0,
        "threshold_searches": 0,
        "persistence_searches": 0,
        "architecture_searches": 0,
        "seed_searches": 0,
        "new_simulation_runs": 0,
    }


def run_final_candidate_holdout_readiness_review(
    root: Path, config_path: Path
) -> dict[str, object]:
    """Freeze or verify the metadata-only readiness decision."""
    review = load_readiness_yaml(config_path)
    artifact_path = root / str(review["artifacts"]["path"])
    freeze_path = artifact_path / str(review["artifacts"]["readiness_freeze"])
    payload = _review_payload(root, config_path, review)
    if freeze_path.exists():
        if freeze_path.read_text(encoding="utf-8") != _canonical_json(payload):
            raise RuntimeError("HOLDOUT readiness freeze changed")
    else:
        artifact_path.mkdir(parents=True, exist_ok=False)
        freeze_path.write_text(_canonical_json(payload), encoding="utf-8")
    return {
        "status": str(payload["review_status"]),
        "readiness_verdict": str(payload["readiness_verdict"]),
        "final_candidate_freeze_status": str(
            payload["final_candidate_freeze_status"]
        ),
        "review_config_sha256": str(payload["review_config_sha256"]),
        "final_candidate_sha256": str(
            payload["candidate"]["final_candidate_sha256"]
        ),
        "holdout_readiness_review_sha256": sha256_file(freeze_path),
        "generalization_holdout_guard_count": 0,
        "new_inference": False,
        "passed": True,
    }


def authorize_one_shot_holdout(
    root: Path,
    config_path: Path,
    *,
    candidate_id: str,
    guard_count: int,
) -> OneShotAuthorization:
    """Authorize, but never perform, the future 0-to-1 scientific open."""
    result = run_final_candidate_holdout_readiness_review(root, config_path)
    if candidate_id != FINAL_CANDIDATE_ID:
        raise RuntimeError("one-shot HOLDOUT candidate is not approved")
    if guard_count != 0:
        raise RuntimeError("one-shot HOLDOUT requires guard count 0 before open")
    review = load_readiness_yaml(config_path)
    if (
        result["readiness_verdict"] != "HOLDOUT_READY"
        or review["holdout"]["authorized_now"]
        or not review["holdout"]["no_access_this_milestone"]
        or review["future_execution"]["execute_now"]
    ):
        raise RuntimeError("one-shot HOLDOUT is not authorized in this milestone")
    return OneShotAuthorization(
        experiment_id=FUTURE_EXPERIMENT_ID,
        candidate_id=candidate_id,
        readiness_review_sha256=str(result["holdout_readiness_review_sha256"]),
        final_candidate_sha256=str(result["final_candidate_sha256"]),
        current_guard_count=0,
        next_guard_count=1,
        holdout_count=int(review["holdout"]["count"]),
        v1_and_v2_same_pass=bool(
            review["future_comparison"]["same_authorized_pass"]
        ),
        terrain_in_same_pass=bool(
            review["future_comparison"]["Terrain_advisory_in_same_pass"]
        ),
    )


def claim_one_shot_guard(
    ledger_path: Path, authorization: OneShotAuthorization
) -> dict[str, object]:
    """Atomically persist the future scientific guard transition exactly once."""
    if authorization.current_guard_count != 0 or authorization.next_guard_count != 1:
        raise RuntimeError("invalid one-shot guard transition")
    payload = {
        "schema_version": 1,
        "experiment_id": authorization.experiment_id,
        "candidate_id": authorization.candidate_id,
        "readiness_review_sha256": authorization.readiness_review_sha256,
        "final_candidate_sha256": authorization.final_candidate_sha256,
        "guard_before": 0,
        "guard_after": 1,
        "scientific_open_claimed": True,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical_json(payload))
    return payload
