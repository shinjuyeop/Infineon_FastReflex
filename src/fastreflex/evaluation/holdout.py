"""One-shot evaluation for the frozen Generalization HOLDOUT."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from fastreflex.dataset.hazard import (
    LABEL_NO_HAZARD,
    HoldoutGuard,
    canonical_sha256,
)
from fastreflex.dataset.loader import sha256_file
from fastreflex.evaluation.generalization import (
    EXPERIMENT_ID as DEVELOPMENT_EXPERIMENT_ID,
    HOLDOUT_COUNT,
    HOLDOUT_SPLIT,
    THRESHOLD,
    _family_results,
    _model_result,
    _precursor_run_counts,
    _run_level_results,
    _side_results,
    _source_results,
    _speed_results,
    load_generalization_manifest,
    load_generalization_split,
)
from fastreflex.evaluation.hazard import (
    load_hazard_normalizer,
    replay_hazard_runs,
    replay_hazard_runs_with_members,
)
from fastreflex.evaluation.readiness import (
    EXPERIMENT_ID as READINESS_EXPERIMENT_ID,
    FINAL_CANDIDATE_ID,
    FINAL_CANDIDATE_PATH,
    load_readiness_yaml,
    verify_final_candidate,
    verify_protected_datasets,
)
from fastreflex.evaluation.terrain import (
    TerrainTrace,
    load_frozen_terrain_candidate,
    replay_terrain_arrays,
    verify_supported_terrain_candidate,
)


EXPERIMENT_ID = "MODEL_V2_GENERALIZATION_HOLDOUT_ONE_SHOT_EVALUATION"
EVALUATOR_SCHEMA_VERSION = 1
STARTING_COMMIT = "6b8f610aa36316f7b9a66864ffe0837ad16d0d83"
PRIMARY_CONTRACT_SHA256 = (
    "feabfc4519e8ec28e59710810b6e587b7a8be1a128ecf57a028d32710c1b246e"
)
SECONDARY_CONTRACT_SHA256 = (
    "085d6f73156a5618767284faa2ccdcd29d3645694f56155431159d533b77130a"
)
VERDICT_HIERARCHY_SHA256 = (
    "e86fb11f457734c41cd7b9c66a827a22b587f7a1f95aa91130931f7586c8cba5"
)


def load_holdout_yaml(path: Path) -> Mapping[str, Any]:
    """Load the pre-frozen one-shot execution contract."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return document


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_new_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())


def _check_file(root: Path, record: Mapping[str, object]) -> None:
    path = root / str(record["path"])
    if sha256_file(path) != str(record["sha256"]):
        raise RuntimeError(f"protected artifact changed: {record['path']}")


def _git_revision(root: Path, name: str) -> str:
    return subprocess.run(
        ("git", "rev-parse", name),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_signature_integrity(
    manifest: Mapping[str, Any],
) -> dict[str, object]:
    rows = manifest["runs"]
    signatures: dict[str, set[str]] = {
        "GENERALIZATION_VALIDATION": set(),
        HOLDOUT_SPLIT: set(),
    }
    all_hashes: list[str] = []
    for row in rows:
        signature_hash = str(row["physical_signature_sha256"])
        if canonical_sha256(row["physical_signature"]) != signature_hash:
            raise RuntimeError(f"physical signature changed: {row['run_id']}")
        split = str(row["split"])
        if signature_hash in signatures[split]:
            raise RuntimeError(f"duplicate physical signature: {row['run_id']}")
        signatures[split].add(signature_hash)
        all_hashes.append(signature_hash)
    overlap = signatures["GENERALIZATION_VALIDATION"] & signatures[HOLDOUT_SPLIT]
    if len(all_hashes) != 72 or len(set(all_hashes)) != 72 or overlap:
        raise RuntimeError("Generalization physical signature integrity failed")
    return {
        "total": len(all_hashes),
        "unique": len(set(all_hashes)),
        "validation_holdout_overlap": len(overlap),
        "ordered_signature_hashes_canonical_sha256": canonical_sha256(all_hashes),
        "passed": True,
    }


def _verify_static_contracts(
    root: Path,
    config_path: Path,
    *,
    require_guard_zero: bool,
) -> dict[str, object]:
    document = load_holdout_yaml(config_path)
    experiment = document["experiment"]
    if (
        experiment["id"] != EXPERIMENT_ID
        or experiment["source_commit"] != STARTING_COMMIT
        or (
            require_guard_zero
            and (
                _git_revision(root, "HEAD") != STARTING_COMMIT
                or _git_revision(root, "origin/main") != STARTING_COMMIT
            )
        )
    ):
        raise RuntimeError("one-shot source revision changed")
    source_path = root / str(document["execution"]["evaluator_path"])
    source_sha = sha256_file(source_path)
    if (
        int(document["execution"]["evaluator_schema_version"])
        != EVALUATOR_SCHEMA_VERSION
        or source_sha != str(document["execution"]["evaluator_sha256"])
    ):
        raise RuntimeError("one-shot evaluator identity changed")
    for record in document["execution"]["source_files"]:
        _check_file(root, record)

    final_record = document["final_candidate"]["record"]
    _check_file(root, final_record)
    if (
        final_record["path"] != str(FINAL_CANDIDATE_PATH)
        or final_record["sha256"]
        != "52644d3efb0e756002bd43a7f94aea0a16a65eb2ff014c72548f2b3b138b7bc2"
        or document["final_candidate"]["candidate_id"] != FINAL_CANDIDATE_ID
        or document["final_candidate"]["role"]
        != "final_generalization_candidate"
    ):
        raise RuntimeError("one-shot candidate role is not the exact final candidate")

    readiness_config = root / str(document["readiness"]["config"]["path"])
    _check_file(root, document["readiness"]["config"])
    _check_file(root, document["readiness"]["freeze"])
    readiness = load_readiness_yaml(readiness_config)
    if readiness["experiment"]["id"] != READINESS_EXPERIMENT_ID:
        raise RuntimeError("readiness review identity changed")
    candidate = verify_final_candidate(root, readiness)
    if (
        candidate["candidate_id"] != FINAL_CANDIDATE_ID
        or candidate["final_candidate_sha256"] != final_record["sha256"]
    ):
        raise RuntimeError("final candidate readiness resolution changed")

    primary = document["primary_contract"]
    secondary = document["secondary_contract"]
    hierarchy = document["final_verdict_hierarchy"]
    if (
        primary != readiness["future_primary_contract"]
        or secondary != readiness["future_secondary_contract"]
        or hierarchy != readiness["future_verdict_hierarchy"]
        or canonical_sha256(primary) != PRIMARY_CONTRACT_SHA256
        or canonical_sha256(secondary) != SECONDARY_CONTRACT_SHA256
        or canonical_sha256(hierarchy) != VERDICT_HIERARCHY_SHA256
    ):
        raise RuntimeError("one-shot frozen contract changed")

    development_path = root / str(document["development_config"]["path"])
    _check_file(root, document["development_config"])
    development = load_holdout_yaml(development_path)
    if development["experiment"]["id"] != DEVELOPMENT_EXPERIMENT_ID:
        raise RuntimeError("development contract identity changed")
    manifest = load_generalization_manifest(root, development)
    signatures = _verify_signature_integrity(manifest)
    protected = verify_protected_datasets(root, readiness)

    from fastreflex.dataset.hazard import load_yaml as load_hazard_yaml
    from fastreflex.evaluation.hazard import verify_supported_candidate

    v1_config = root / str(document["baseline_v1"]["config"]["path"])
    _check_file(root, document["baseline_v1"]["config"])
    v1 = verify_supported_candidate(root, load_hazard_yaml(v1_config))
    terrain = verify_supported_terrain_candidate(root)
    if (
        v1["freeze_sha256"] != document["baseline_v1"]["freeze_sha256"]
        or v1["threshold"] != THRESHOLD
        or v1["persistence_ms"] != 5
        or not v1["passed"]
        or not terrain["passed"]
        or not terrain["advisory_only"]
        or terrain["hazard_gate"]
    ):
        raise RuntimeError("frozen V1 or Terrain candidate changed")

    guard_path = root / str(document["guard"]["record_path"])
    artifact_path = root / str(document["artifacts"]["path"])
    if require_guard_zero:
        if int(document["guard"]["required_before"]) != 0 or guard_path.exists():
            raise RuntimeError("HOLDOUT_EVALUATION_ABORTED_GUARD_STATE")
        if artifact_path.exists():
            raise RuntimeError("one-shot artifact identity already exists")
    return {
        "execution_config_sha256": sha256_file(config_path),
        "evaluator_sha256": source_sha,
        "candidate": candidate,
        "v1": v1,
        "terrain": terrain,
        "datasets": protected,
        "signatures": signatures,
        "guard_before": 0 if require_guard_zero else 1,
        "holdout_payload_deserialized": False,
        "passed": True,
    }


def preflight_holdout_evaluation(
    root: Path, config_path: Path
) -> dict[str, object]:
    """Verify every pre-open condition without deserializing HOLDOUT."""
    return _verify_static_contracts(
        root,
        config_path,
        require_guard_zero=True,
    )


def _claim_guard(
    root: Path,
    config_path: Path,
    preflight: Mapping[str, object],
) -> tuple[Mapping[str, object], Path]:
    document = load_holdout_yaml(config_path)
    guard_path = root / str(document["guard"]["record_path"])
    artifact_path = root / str(document["artifacts"]["path"])
    if (
        preflight["execution_config_sha256"] != sha256_file(config_path)
        or preflight["evaluator_sha256"] != sha256_file(Path(__file__))
        or guard_path.exists()
        or artifact_path.exists()
    ):
        raise RuntimeError("HOLDOUT_EVALUATION_ABORTED_GUARD_STATE")
    artifact_path.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "transition_timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(
            timespec="microseconds"
        ),
        "source_commit": STARTING_COMMIT,
        "execution_config_sha256": str(preflight["execution_config_sha256"]),
        "evaluator_sha256": str(preflight["evaluator_sha256"]),
        "evaluator_schema_version": EVALUATOR_SCHEMA_VERSION,
        "candidate_id": FINAL_CANDIDATE_ID,
        "final_candidate_sha256": str(
            preflight["candidate"]["final_candidate_sha256"]
        ),
        "primary_contract_sha256": PRIMARY_CONTRACT_SHA256,
        "secondary_contract_sha256": SECONDARY_CONTRACT_SHA256,
        "guard_before": 0,
        "guard_after": 1,
        "scientific_open_claimed": True,
        "scientific_open_count": 1,
        "first_payload_deserialized": False,
        "second_scientific_open_forbidden": True,
    }
    _write_new_json(guard_path, payload)
    directory = os.open(artifact_path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return payload, guard_path


def _terrain_replays(
    root: Path,
    document: Mapping[str, Any],
    data: Any,
) -> dict[str, TerrainTrace]:
    model_path = root / str(document["terrain_v1"]["model_path"])
    models, mean, std = load_frozen_terrain_candidate(model_path)
    traces: dict[str, TerrainTrace] = {}
    for run_id, run in sorted(data.runs.items()):
        fsr = run.features["PELVIS_IMU6_FSR8"][:, 6:]
        traces[run_id] = replay_terrain_arrays(
            run.timestamp_us,
            data.exact_terrain_contacts[run_id],
            fsr,
            run,
            models,
            mean,
            std,
            deployment_scheme="left_only",
        )
    return traces


def _terrain_diagnostics(
    data: Any,
    terrain: Mapping[str, TerrainTrace],
    v2_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    clean_total = 0
    evaluable_total = 0
    correct_total = 0
    target_available = 0
    first_target_relative: list[int] = []
    orders = {
        "TERRAIN_BEFORE_REFLEX": 0,
        "REFLEX_BEFORE_TERRAIN": 0,
        "UNAVAILABLE": 0,
    }
    for run_id, trace in sorted(terrain.items()):
        run = data.runs[run_id]
        if trace.prediction_true_ids is None:
            raise RuntimeError(f"Terrain truth provenance is unavailable: {run_id}")
        if len(trace.prediction_ids) != len(trace.prediction_true_ids):
            raise RuntimeError(f"Terrain prediction provenance changed: {run_id}")
        evaluable = len(trace.prediction_ids)
        correct = int(
            np.sum(trace.prediction_ids == trace.prediction_true_ids)
        )
        clean_total += trace.clean_event_count
        evaluable_total += evaluable
        correct_total += correct
        first_target = trace.first_target_valid_sample
        if first_target is not None:
            target_available += 1
            first_target_relative.append(first_target - run.first_contact_sample)
        v2_row = v2_rows[run_id]
        reflex = v2_row["system_first_onset"]
        if v2_row["physical_label"] == LABEL_NO_HAZARD:
            order = None
        elif first_target is None or reflex is None:
            order = "UNAVAILABLE"
            orders[order] += 1
        elif first_target <= int(reflex):
            order = "TERRAIN_BEFORE_REFLEX"
            orders[order] += 1
        else:
            order = "REFLEX_BEFORE_TERRAIN"
            orders[order] += 1
        rows.append(
            {
                "run_id": run_id,
                "clean_event_count": trace.clean_event_count,
                "evaluable_event_count": evaluable,
                "correct_clean_events": correct,
                "terrain_accuracy": (
                    None
                    if evaluable == 0
                    else correct / evaluable
                ),
                "target_first_valid": first_target,
                "target_available": first_target is not None,
                "target_contact_to_first_valid_ms": (
                    None
                    if first_target is None
                    else first_target - run.first_contact_sample
                ),
                "v2_reflex": reflex,
                "v2_terrain_reflex_order": order,
            }
        )
    selected = np.asarray(first_target_relative, dtype=float)
    distribution = {
        "median": None if not len(selected) else float(np.median(selected)),
        "p95": (
            None if not len(selected) else float(np.percentile(selected, 95))
        ),
    }
    return {
        "deployment_scheme": "left_only",
        "observation_ms": 50,
        "advisory_only": True,
        "hazard_gate": False,
        "clean_event_count": clean_total,
        "evaluable_event_count": evaluable_total,
        "correct_clean_events": correct_total,
        "accuracy": (
            None if not evaluable_total else correct_total / evaluable_total
        ),
        "target_available_runs": target_available,
        "target_unavailable_runs": len(terrain) - target_available,
        "target_contact_to_first_valid_ms": distribution,
        "hazard_order_counts": orders,
        "rows": rows,
    }


def _failure_category(row: Mapping[str, object]) -> str | None:
    if row["physical_label"] == LABEL_NO_HAZARD:
        return (
            "BENIGN_FALSE_ALERT"
            if row["v2_result"] == "FALSE_POSITIVE"
            else None
        )
    if row["v2_result"] == "CORRECT":
        return None
    relation = row["v2_precursor_aware_classification"]
    if relation == "SUPPORTED_FUTURE_SLIP_PRECURSOR":
        return "SUPPORTED_ICE_PRECURSOR_TIMING_CONFLICT"
    if relation == "PRE_I1_SAND_FALSE_ALERT":
        return "PRE_I1_SUPPORT_FALSE_ALERT"
    if relation == "CENSORED_PRECURSOR_ALERT":
        return "CENSORED_OR_AMBIGUOUS"
    if row["v2_first_reflex"] is None:
        return "GENUINE_DETECTION_MISS"
    if row["v2_result"] == "OUT_OF_VALID_WINDOW":
        return "LATE_DETECTION"
    return "OTHER"


def _verdicts(
    v2_result: Mapping[str, Any],
    run_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    gates = v2_result["gates"]
    primary = (
        "GENERALIZATION_HOLDOUT_PRIMARY_GATES_PASS"
        if all(gates.values())
        else "GENERALIZATION_HOLDOUT_PRIMARY_GATES_FAIL"
    )
    failures = [row for row in run_rows if row["primary_failure_category"]]
    genuine = [
        row
        for row in failures
        if row["primary_failure_category"] == "GENUINE_DETECTION_MISS"
    ]
    non_slip_gates = {
        name: passed
        for name, passed in gates.items()
        if name != "slip_hazard_recall"
    }
    qualified = bool(
        not all(gates.values())
        and not gates["slip_hazard_recall"]
        and all(non_slip_gates.values())
        and failures
        and all(
            row["primary_failure_category"]
            == "SUPPORTED_ICE_PRECURSOR_TIMING_CONFLICT"
            for row in failures
        )
        and not genuine
    )
    if all(gates.values()):
        final = "MODEL_V2_GENERALIZATION_HOLDOUT_SUPPORTED"
    elif qualified:
        final = (
            "MODEL_V2_GENERALIZATION_HOLDOUT_SUPPORTED_WITH_ICE_TIMING_TENSION"
        )
    else:
        final = "MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED"
    supported = final in (
        "MODEL_V2_GENERALIZATION_HOLDOUT_SUPPORTED",
        "MODEL_V2_GENERALIZATION_HOLDOUT_SUPPORTED_WITH_ICE_TIMING_TENSION",
    )
    return {
        "primary_gate_verdict": primary,
        "final_holdout_verdict": final,
        "data_coverage_verdict": (
            "DATA_COVERAGE_HYPOTHESIS_SUPPORTED"
            if supported
            else "DATA_COVERAGE_HYPOTHESIS_NOT_SUPPORTED"
        ),
        "architecture_verdict": (
            "ARCHITECTURE_CHANGE_NOT_JUSTIFIED"
            if supported and not genuine
            else "ARCHITECTURE_EVIDENCE_REQUIRES_REVIEW"
        ),
        "sensor_verdict": "10_CHANNEL_ARCHITECTURE_STILL_PLAUSIBLE",
        "simulation_research_status": (
            "SIMULATION_GENERALIZATION_EVIDENCE_COMPLETE"
            if supported
            else "SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED"
        ),
    }


def _execute_once(
    root: Path,
    config_path: Path,
    preflight: Mapping[str, object],
    guard: Mapping[str, object],
    guard_path: Path,
) -> dict[str, object]:
    document = load_holdout_yaml(config_path)
    development = load_holdout_yaml(
        root / str(document["development_config"]["path"])
    )
    memory_guard = HoldoutGuard()
    memory_guard.open_once()
    data = load_generalization_split(
        root,
        development,
        HOLDOUT_SPLIT,
        holdout_guard=memory_guard,
    )
    if (
        memory_guard.open_count != 1
        or len(data.runs) != HOLDOUT_COUNT
        or data.payload_deserializations != HOLDOUT_COUNT
    ):
        raise RuntimeError("one-shot HOLDOUT load contract failed")

    v1_normalizer = load_hazard_normalizer(
        root / str(document["baseline_v1"]["normalizer"]["path"])
    )
    v1_checkpoints = tuple(
        root / str(record["path"])
        for record in document["baseline_v1"]["checkpoints"]
    )
    v1_replays = replay_hazard_runs(data.runs, v1_normalizer, v1_checkpoints)
    v1_result = _model_result(data, v1_replays, document["primary_contract"]["gates"])

    v2_normalizer = load_hazard_normalizer(
        root / str(document["final_candidate"]["normalizer"]["path"])
    )
    v2_checkpoints = tuple(
        root / str(record["path"])
        for record in document["final_candidate"]["checkpoints"]
    )
    v2_replays, v2_members = replay_hazard_runs_with_members(
        data.runs, v2_normalizer, v2_checkpoints
    )
    v2_result = _model_result(data, v2_replays, document["primary_contract"]["gates"])
    terrain = _terrain_replays(root, document, data)

    v1_rows = {
        str(row["run_id"]): row for row in v1_result["primary"]["rows"]
    }
    v2_rows = {
        str(row["run_id"]): row for row in v2_result["primary"]["rows"]
    }
    terrain_rows = {
        run_id: {
            "terrain_first_target_valid_sample": trace.first_target_valid_sample,
            "terrain_target_available": trace.first_target_valid_sample is not None,
        }
        for run_id, trace in terrain.items()
    }
    run_rows = _run_level_results(
        data,
        v1_result,
        v2_result,
        v1_replays,
        v2_replays,
        v2_members,
        terrain_rows,
        document["final_candidate"]["ensemble_membership"],
    )
    for row in run_rows:
        row["primary_failure_category"] = _failure_category(row)

    family = _family_results(data, v1_rows, v2_rows)
    speed = _speed_results(data, v1_rows, v2_rows)
    side = _side_results(data, v1_rows, v2_rows)
    source = _source_results(data, v1_rows, v2_rows)
    secondary = {
        "v1": {
            **v1_result["ice_precursor_secondary"],
            "runs": _precursor_run_counts(data, v1_replays),
        },
        "v2": {
            **v2_result["ice_precursor_secondary"],
            "runs": _precursor_run_counts(data, v2_replays),
        },
        "primary_scores_rewritten": False,
        "contract_sha256": SECONDARY_CONTRACT_SHA256,
    }
    terrain_result = _terrain_diagnostics(data, terrain, v2_rows)
    verdicts = _verdicts(v2_result, run_rows)
    primary = {
        "v1": v1_result,
        "v2": v2_result,
        "primary_contract_sha256": PRIMARY_CONTRACT_SHA256,
        **verdicts,
    }

    artifacts: dict[str, object] = {
        "run_level_results.json": {"rows": run_rows},
        "primary_metrics.json": primary,
        "secondary_metrics.json": secondary,
        "terrain_diagnostics.json": terrain_result,
        "family_results.json": family,
        "speed_results.json": speed,
        "side_results.json": side,
        "source_results.json": source,
    }
    artifact_path = root / str(document["artifacts"]["path"])
    artifact_hashes: dict[str, str] = {}
    for name, value in artifacts.items():
        path = artifact_path / name
        _write_new_json(path, value)
        artifact_hashes[name] = sha256_file(path)

    failures = [row for row in run_rows if row["primary_failure_category"]]
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_commit": STARTING_COMMIT,
        "execution_config_sha256": str(preflight["execution_config_sha256"]),
        "evaluator_sha256": str(preflight["evaluator_sha256"]),
        "guard_record_sha256": sha256_file(guard_path),
        "candidate_id": FINAL_CANDIDATE_ID,
        "final_candidate_sha256": str(
            preflight["candidate"]["final_candidate_sha256"]
        ),
        "dataset_id": document["dataset"]["id"],
        "split": HOLDOUT_SPLIT,
        "holdout_runs": len(data.runs),
        "runs_processed": len(run_rows),
        "payload_deserializations": data.payload_deserializations,
        "payload_deserializations_per_run": 1,
        "first_payload_read_after_guard_transition": True,
        "same_pass_models": list(document["same_pass_models"]),
        "v1_hazard_inference": True,
        "final_v2_hazard_inference": True,
        "terrain_v1_inference": True,
        "visualization": False,
        "partial_adaptive_access": False,
        "guard_before": guard["guard_before"],
        "guard_after": guard["guard_after"],
        "scientific_open_count": 1,
        "second_open_attempted": False,
        "primary_failure_count": len(failures),
        "supported_ice_timing_conflict_count": sum(
            row["primary_failure_category"]
            == "SUPPORTED_ICE_PRECURSOR_TIMING_CONFLICT"
            for row in failures
        ),
        "genuine_detection_miss_count": sum(
            row["primary_failure_category"] == "GENUINE_DETECTION_MISS"
            for row in failures
        ),
        "benign_false_alert_count": sum(
            row["primary_failure_category"] == "BENIGN_FALSE_ALERT"
            for row in failures
        ),
        "pre_i1_support_false_alert_count": sum(
            row["primary_failure_category"] == "PRE_I1_SUPPORT_FALSE_ALERT"
            for row in failures
        ),
        "physically_supported_hazard_response_count": sum(
            bool(row["valid_detection"])
            for row in v2_result["primary"]["rows"]
            if row["physical_label"] != LABEL_NO_HAZARD
        )
        + sum(
            row["primary_failure_category"]
            == "SUPPORTED_ICE_PRECURSOR_TIMING_CONFLICT"
            for row in failures
        ),
        "physically_supported_response_is_diagnostic_only": True,
        **verdicts,
        "final_sensor_architecture_frozen": False,
        "candidate_mutated": False,
        "dataset_mutated": False,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "normalizer_fits": 0,
        "hnm_rounds": 0,
        "threshold_searches": 0,
        "persistence_searches": 0,
        "architecture_searches": 0,
        "seed_searches": 0,
        "new_simulation_runs": 0,
        "artifact_sha256": artifact_hashes,
        "status": "MODEL_V2_GENERALIZATION_HOLDOUT_ONE_SHOT_EVALUATION_COMPLETE",
    }
    result_path = artifact_path / "evaluation_result.json"
    _write_new_json(result_path, result)
    result_sha = sha256_file(result_path)
    freeze = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "execution_config_sha256": str(preflight["execution_config_sha256"]),
        "evaluator_sha256": str(preflight["evaluator_sha256"]),
        "guard_record_sha256": sha256_file(guard_path),
        "evaluation_result_sha256": result_sha,
        "artifact_sha256": artifact_hashes,
        "guard_after": 1,
        "scientific_open_count": 1,
        "payload_deserializations": data.payload_deserializations,
        "candidate_mutated": False,
        "dataset_mutated": False,
        "second_open_attempted": False,
        "status": result["status"],
    }
    freeze_path = artifact_path / "evaluation_freeze.json"
    _write_new_json(freeze_path, freeze)
    return {
        "status": str(result["status"]),
        "primary_gate_verdict": str(verdicts["primary_gate_verdict"]),
        "final_holdout_verdict": str(verdicts["final_holdout_verdict"]),
        "simulation_research_status": str(
            verdicts["simulation_research_status"]
        ),
        "execution_config_sha256": str(preflight["execution_config_sha256"]),
        "holdout_one_shot_result_sha256": result_sha,
        "holdout_run_level_sha256": artifact_hashes["run_level_results.json"],
        "holdout_primary_metrics_sha256": artifact_hashes["primary_metrics.json"],
        "holdout_secondary_metrics_sha256": artifact_hashes[
            "secondary_metrics.json"
        ],
        "holdout_terrain_diagnostic_sha256": artifact_hashes[
            "terrain_diagnostics.json"
        ],
        "holdout_guard_record_sha256": sha256_file(guard_path),
        "guard_after": 1,
        "scientific_open_count": 1,
        "passed": True,
    }


def run_generalization_holdout_one_shot_evaluation(
    root: Path, config_path: Path
) -> dict[str, object]:
    """Claim the durable guard and consume HOLDOUT in one operation."""
    preflight = preflight_holdout_evaluation(root, config_path)
    guard, guard_path = _claim_guard(root, config_path, preflight)
    artifact_path = guard_path.parent
    try:
        return _execute_once(root, config_path, preflight, guard, guard_path)
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "final_holdout_verdict": (
                "MODEL_V2_GENERALIZATION_HOLDOUT_INCONCLUSIVE"
            ),
            "reason": "POST_OPEN_EVALUATION_FAILURE",
            "guard_after": 1,
            "scientific_open_count": 1,
            "error_type": type(error).__name__,
            "error": str(error),
            "rerun_forbidden": True,
        }
        failure_path = artifact_path / "evaluation_failure.json"
        if not failure_path.exists():
            _write_new_json(failure_path, failure)
        raise RuntimeError(
            "POST_OPEN_EVALUATION_FAILURE; HOLDOUT is consumed and rerun is forbidden"
        ) from error


def verify_generalization_holdout_evaluation(
    root: Path, config_path: Path
) -> dict[str, object]:
    """Verify saved one-shot summaries without reopening HOLDOUT payloads."""
    static = _verify_static_contracts(
        root,
        config_path,
        require_guard_zero=False,
    )
    document = load_holdout_yaml(config_path)
    artifact_path = root / str(document["artifacts"]["path"])
    guard_path = root / str(document["guard"]["record_path"])
    freeze_path = artifact_path / "evaluation_freeze.json"
    result_path = artifact_path / "evaluation_result.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        guard["guard_before"] != 0
        or guard["guard_after"] != 1
        or guard["scientific_open_count"] != 1
        or not guard["scientific_open_claimed"]
        or freeze["guard_record_sha256"] != sha256_file(guard_path)
        or freeze["evaluation_result_sha256"] != sha256_file(result_path)
        or freeze["execution_config_sha256"]
        != static["execution_config_sha256"]
        or freeze["evaluator_sha256"] != static["evaluator_sha256"]
        or any(
            sha256_file(artifact_path / name) != expected
            for name, expected in freeze["artifact_sha256"].items()
        )
        or result["guard_after"] != 1
        or result["scientific_open_count"] != 1
        or result["payload_deserializations"] != HOLDOUT_COUNT
        or result["payload_deserializations_per_run"] != 1
        or not result["first_payload_read_after_guard_transition"]
        or result["candidate_mutated"]
        or result["dataset_mutated"]
        or result["second_open_attempted"]
    ):
        raise RuntimeError("saved one-shot HOLDOUT result integrity failed")
    return {
        "status": str(result["status"]),
        "primary_gate_verdict": str(result["primary_gate_verdict"]),
        "final_holdout_verdict": str(result["final_holdout_verdict"]),
        "simulation_research_status": str(result["simulation_research_status"]),
        "execution_config_sha256": str(static["execution_config_sha256"]),
        "holdout_one_shot_result_sha256": sha256_file(result_path),
        "holdout_run_level_sha256": freeze["artifact_sha256"][
            "run_level_results.json"
        ],
        "holdout_primary_metrics_sha256": freeze["artifact_sha256"][
            "primary_metrics.json"
        ],
        "holdout_secondary_metrics_sha256": freeze["artifact_sha256"][
            "secondary_metrics.json"
        ],
        "holdout_terrain_diagnostic_sha256": freeze["artifact_sha256"][
            "terrain_diagnostics.json"
        ],
        "holdout_guard_record_sha256": sha256_file(guard_path),
        "guard_after": 1,
        "scientific_open_count": 1,
        "holdout_payload_deserialized": False,
        "passed": True,
    }
