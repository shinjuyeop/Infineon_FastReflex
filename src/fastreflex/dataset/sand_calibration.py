"""Model-blind physical calibration for Sand generalization studies."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fastreflex.dataset.generation import (
    _load_yaml,
    _signature,
    _write_deterministic_npz,
    _write_json,
    annotate_model_v2_result,
    canonical_sha256,
    sha256_file,
)
from fastreflex.simulation.g1 import SimulationConfig, run_simulation
from fastreflex.simulation.stability import PHASE_NAMES


CALIBRATION_CONFIG_IDS = {
    "SAND_BENIGN_PHASE_GEOMETRY_CALIBRATION",
    "SAND_BENIGN_SEVERITY_CALIBRATION",
    "SAND_BENIGN_DOMAIN_CONTROL_CALIBRATION",
}
REDESIGN_SPLITS = ("REDESIGNED_DISCOVERY", "REDESIGNED_CONFIRMATION")
REDESIGNED_GENERATION_ID = "SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION"
REDESIGNED_DATASET_ID = "sand_benign_generalization_redesigned_study_20260902"


def _expand_calibration_scenarios(
    calibration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if "scenarios" in calibration:
        return [dict(row) for row in calibration["scenarios"]]
    fixed = dict(calibration["fixed_mechanics"])
    prefix = str(calibration["run_id_prefix"])
    scenarios: list[dict[str, Any]] = []
    for template in calibration["templates"]:
        for source in calibration["sources"]:
            for speed in calibration["speeds_mps"]:
                source_code = "c" if source == "concrete" else "m"
                speed_code = f"{int(round(float(speed) * 100)):03d}"
                scenarios.append(
                    {
                        **fixed,
                        **dict(template),
                        "run_id": (
                            f"{prefix}_{template['id']}_{source_code}_{speed_code}"
                        ),
                        "source_terrain": source,
                        "speed_mps": float(speed),
                    }
                )
    return scenarios


def _manifest_runs(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict):
        return list(document.get("runs", document.get("records", [])))
    return list(document)


def _historical_signatures(
    root: Path, references: Sequence[Mapping[str, Any]]
) -> tuple[set[tuple[Any, ...]], list[dict[str, Any]]]:
    signatures: set[tuple[Any, ...]] = set()
    provenance: list[dict[str, Any]] = []
    for reference in references:
        path = root / str(reference["path"])
        actual_sha = sha256_file(path)
        expected_sha = str(reference["sha256"])
        if actual_sha != expected_sha:
            raise RuntimeError(f"historical manifest changed: {path}")
        rows = _manifest_runs(path)
        for row in rows:
            raw = row.get("scenario_signature", row.get("physical_signature"))
            if isinstance(raw, list):
                signatures.add(tuple(raw))
            elif all(
                key in row
                for key in (
                    "source_terrain",
                    "target_terrain",
                    "speed_mps",
                    "patch_start_x_m",
                    "patch_width_m",
                    "slip_pattern",
                    "sink_pattern",
                    "sink_severity",
                    "support_pattern",
                )
            ):
                signatures.add(_signature(row))
        provenance.append(
            {
                "path": str(reference["path"]),
                "sha256": actual_sha,
                "run_count": len(rows),
            }
        )
    return signatures, provenance


def _scenario_signatures_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[Any, ...]]:
    signatures: list[tuple[Any, ...]] = []
    required = (
        "source_terrain",
        "target_terrain",
        "speed_mps",
        "patch_start_x_m",
        "patch_width_m",
        "slip_pattern",
        "sink_pattern",
        "sink_severity",
        "support_pattern",
    )
    for row in rows:
        raw = row.get("scenario_signature", row.get("physical_signature"))
        if isinstance(raw, list) and len(raw) == len(required):
            signatures.append(tuple(raw))
        elif all(key in row for key in required):
            signatures.append(_signature(row))
    return signatures


def _scenario_signatures_are_near(
    left: Sequence[Any], right: Sequence[Any], policy: Mapping[str, Any]
) -> bool:
    domain_indexes = (0, 1, 2, 5, 6, 7, 8)
    return all(left[index] == right[index] for index in domain_indexes) and (
        abs(float(left[3]) - float(right[3]))
        < float(policy["patch_start_difference_m_exclusive"])
        and abs(float(left[4]) - float(right[4]))
        < float(policy["patch_width_difference_m_exclusive"])
    )


def _historical_overlap_audit(
    root: Path,
    references: Sequence[Mapping[str, Any]],
    specifications: Sequence[Mapping[str, Any]],
    near_policy: Mapping[str, Any],
) -> dict[str, Any]:
    planned = [_signature(row) for row in specifications]
    planned_ids = {str(row["run_id"]) for row in specifications}
    exact_by_reference: dict[str, int] = {}
    near_by_reference: dict[str, int] = {}
    run_id_reuse_by_reference: dict[str, int] = {}
    for reference in references:
        relative = str(reference["path"])
        path = root / relative
        if sha256_file(path) != str(reference["sha256"]):
            raise RuntimeError(f"historical manifest changed: {path}")
        historical_rows = _manifest_runs(path)
        historical = _scenario_signatures_from_rows(historical_rows)
        historical_set = set(historical)
        exact_by_reference[relative] = len(set(planned) & historical_set)
        near_by_reference[relative] = sum(
            _scenario_signatures_are_near(left, right, near_policy)
            for left in planned
            for right in historical
            if left != right
        )
        run_id_reuse_by_reference[relative] = len(
            planned_ids & {str(row.get("run_id")) for row in historical_rows}
        )
    return {
        "exact_by_reference": exact_by_reference,
        "near_by_reference": near_by_reference,
        "run_id_reuse_by_reference": run_id_reuse_by_reference,
        "exact_total": sum(exact_by_reference.values()),
        "near_total": sum(near_by_reference.values()),
        "run_id_reuse_total": sum(run_id_reuse_by_reference.values()),
    }


def _annotation_specification(specification: Mapping[str, Any]) -> dict[str, Any]:
    routed = dict(specification)
    group = str(specification["group"])
    if "sand_benign" in group:
        routed["scenario_family"] = "STAGED_SAND_BENIGN_CONTROL"
    elif group == "ordinary_support_control":
        routed["scenario_family"] = (
            "LEFT_SAND_SUPPORT_SPEED_MATRIX"
            if specification["designed_side"] == "LEFT"
            else "RIGHT_SAND_SUPPORT_SPEED_MATRIX"
        )
    elif group == "delayed_support_control":
        routed["scenario_family"] = "DELAYED_SAND_SUPPORT_ONSET"
    else:
        raise ValueError(f"unsupported Sand calibration group: {group}")
    return routed


def validate_sand_calibration_config(
    root: Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a frozen pilot without reading any model or HOLDOUT payload."""
    experiment_id = str(document["experiment"]["id"])
    if experiment_id not in CALIBRATION_CONFIG_IDS:
        raise ValueError(f"unsupported calibration id: {experiment_id}")
    calibration = document["calibration"]
    scenarios = _expand_calibration_scenarios(calibration)
    if not scenarios or len(scenarios) > 96:
        raise ValueError("a calibration config must contain 1..96 scenarios")
    if len({str(row["run_id"]) for row in scenarios}) != len(scenarios):
        raise ValueError("calibration run IDs must be unique")
    signatures = [_signature(row) for row in scenarios]
    if len(set(signatures)) != len(signatures):
        raise ValueError("calibration scenario signatures must be unique")
    historical, provenance = _historical_signatures(
        root, calibration["historical_manifests"]
    )
    overlap = sorted(set(signatures) & historical, key=str)
    if overlap:
        raise ValueError("calibration overlaps a historical signature")
    return {
        "run_count": len(scenarios),
        "unique_run_ids": len(scenarios),
        "unique_signatures": len(signatures),
        "historical_signature_overlap": 0,
        "scenario_matrix_sha256": canonical_sha256(scenarios),
        "scenario_signature_sha256": canonical_sha256(
            [list(value) for value in signatures]
        ),
        "historical_manifests": provenance,
    }


def _side(values: np.ndarray) -> str:
    active = np.asarray(values, dtype=bool).reshape(2)
    if bool(active[0]) and bool(active[1]):
        return "BILATERAL"
    if bool(active[0]):
        return "LEFT"
    if bool(active[1]):
        return "RIGHT"
    return "NONE"


def _calibration_result_summary(
    row: dict[str, Any],
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> None:
    """Apply the corrected, censor-aware physical calibration contract."""
    censor = int(arrays["censor_sample"])
    fall_value = int(arrays["first_fall_sample"])
    fall = None if fall_value < 0 else fall_value
    target = np.asarray(arrays["target_terrain_contact"], dtype=bool)
    target_any = np.any(target[:censor], axis=1)
    target_samples = np.flatnonzero(target_any)
    first = None if not target_samples.size else int(target_samples[0])
    last = None if not target_samples.size else int(target_samples[-1])
    followup = 0 if last is None else max(0, censor - (last + 1))
    lookback = int(contract["precontact_phase_lookback_ms"])
    if first is None:
        leading = "NONE"
        contact_phase = "NO_SUPPORT"
        precontact_phase = "NO_SUPPORT"
        loaded_side = "NONE"
    else:
        leading = _side(target[first])
        contact_phase = PHASE_NAMES[int(arrays["gait_phase"][first])]
        precontact_phase = PHASE_NAMES[
            int(arrays["gait_phase"][max(0, first - lookback)])
        ]
        loaded_side = _side(arrays["loaded_contact"][first])

    slip = row["slip_event_summary"]["first_sample"]
    i1 = row["i1_summary"]["first_sample"]
    support = row["support_event_summary"]["first_sample"]
    group = str(row["group"])
    invalid_reason: str | None = None
    outcome = "INVALID"
    severity: str | None = None

    if first is None:
        invalid_reason = "pretarget_fall" if fall is not None else "no_target_contact"
    elif slip is not None and support is not None:
        outcome = "DUAL_HAZARD"
    elif slip is not None:
        outcome = "SLIP"
    elif support is not None:
        event_followup = censor - int(support)
        if event_followup < int(contract["support_post_event_followup_ms"]):
            invalid_reason = "insufficient_post_support_observation"
        else:
            outcome = "SUPPORT"
    elif "sand_benign" in group:
        if followup < int(contract["benign_post_target_followup_ms"]):
            invalid_reason = "insufficient_post_target_observation"
        elif i1 is not None:
            invalid_reason = "physical_outcome_mismatch"
        else:
            outcome = "STRICT_BENIGN"
    else:
        invalid_reason = "physical_outcome_mismatch"

    sample_count = len(arrays.get("timestamp_us", arrays["gait_phase"]))
    required_shapes = (
        "pelvis_imu6",
        "foot_fsr8",
        "target_terrain_contact",
        "loaded_contact",
        "support_surface_max_displacement_m",
        "support_surface_spread_m",
        "gait_phase",
    )
    trace_shape_ok = all(
        np.asarray(arrays[key]).shape[0] == sample_count
        for key in required_shapes
        if key in arrays
    )
    expected_samples = contract.get("expected_samples")
    if expected_samples is not None:
        trace_shape_ok = trace_shape_ok and sample_count == int(expected_samples)
    finite_ok = all(
        bool(np.all(np.isfinite(arrays[key])))
        for key in ("pelvis_imu6", "foot_fsr8")
        if key in arrays
    )
    if not finite_ok or not trace_shape_ok:
        outcome = "INVALID"
        invalid_reason = "nonfinite_or_malformed"

    touchdown_limit = censor if i1 is None else int(i1)
    clean_touchdowns = int(
        np.count_nonzero(
            np.any(
                np.asarray(arrays["target_terrain_touchdown"][:touchdown_limit]),
                axis=1,
            )
        )
    )
    if group == "ordinary_support_control":
        expected_side = f"{row['designed_side']}_ONLY"
        intent_match = bool(
            outcome == "SUPPORT"
            and i1 is not None
            and int(i1) <= int(support)
            and row["support_event_summary"]["side"] == expected_side
        )
    elif group == "delayed_support_control":
        intent_match = bool(
            outcome == "SUPPORT"
            and i1 is not None
            and int(i1) <= int(support)
            and clean_touchdowns >= 2
            and row["support_event_summary"]["side"] == "LEFT_ONLY"
        )
    else:
        intent_match = outcome == "STRICT_BENIGN"

    interval = target_any
    displacement = np.asarray(
        arrays["support_surface_max_displacement_m"][:censor], dtype=np.float64
    )
    spread = np.asarray(arrays["support_surface_spread_m"][:censor], dtype=np.float64)
    if np.any(interval):
        peak_displacement = float(np.max(displacement[interval]))
        peak_spread = float(np.max(spread[:censor][target[:censor]]))
    else:
        peak_displacement = 0.0
        peak_spread = 0.0
    if "sand_benign" in group and outcome == "STRICT_BENIGN":
        if peak_displacement < 0.030:
            severity = "LOW"
        elif peak_displacement < 0.0525:
            severity = "MEDIUM"
        elif peak_displacement <= 0.070:
            severity = "NEAR_HAZARD"
        else:
            severity = "OUT_OF_DOMAIN"

    row["valid"] = invalid_reason is None
    row["invalid_reason"] = invalid_reason
    row["objective_physical_outcome"] = outcome
    row["actual_benign_severity"] = severity
    row["target_contact_summary"].update(
        {
            "first_sample": first,
            "last_sample_before_censor": last,
            "duration_ms_before_censor": int(np.count_nonzero(interval)),
            "post_target_observation_ms": followup,
            "leading_foot": leading,
            "loaded_side_at_contact": loaded_side,
            "contact_sample_phase": contact_phase,
            "precontact_phase": precontact_phase,
            "precontact_phase_lookback_ms": lookback,
            "clean_touchdowns_before_i1": clean_touchdowns,
        }
    )
    fsr = np.asarray(
        arrays.get("foot_fsr8", np.zeros((sample_count, 8))), dtype=np.float64
    )
    foot_load = np.column_stack(
        (np.sum(fsr[:, :4], axis=1), np.sum(fsr[:, 4:], axis=1))
    )
    total_load = np.sum(foot_load, axis=1)
    if np.any(interval):
        target_load = foot_load[:censor][interval]
        target_total = np.sum(target_load, axis=1)
        redistribution = float(
            np.max(
                np.abs(target_load[:, 0] - target_load[:, 1])
                / np.maximum(target_total, 1.0e-9)
            )
        )
    else:
        redistribution = 0.0
    double_loaded = np.all(np.asarray(arrays["loaded_contact"][:censor]), axis=1)
    reference = total_load[:censor][double_loaded & (total_load[:censor] > 0)]
    if not reference.size:
        reference = total_load[:censor][total_load[:censor] > 0]
    body_weight_proxy = 1.0 if not reference.size else float(np.median(reference))
    derivative = np.abs(np.diff(total_load[:censor], prepend=total_load[0])) * 1000.0
    peak_derivative = (
        0.0 if not np.any(interval) else float(np.max(derivative[interval]))
    )
    normalized_derivative = peak_derivative / max(body_weight_proxy, 1.0e-9)
    imu = np.asarray(
        arrays.get("pelvis_imu6", np.zeros((sample_count, 6))), dtype=np.float64
    )
    selected_imu = imu[:censor][interval]
    if selected_imu.size:
        accel_rms = float(
            np.sqrt(np.mean(np.square(np.linalg.norm(selected_imu[:, :3], axis=1))))
        )
        gyro_rms = float(
            np.sqrt(np.mean(np.square(np.linalg.norm(selected_imu[:, 3:], axis=1))))
        )
    else:
        accel_rms = gyro_rms = 0.0
    raw_scenario_signature = row.get("physical_signature")
    if isinstance(raw_scenario_signature, list):
        scenario_signature = list(raw_scenario_signature)
        row["scenario_signature"] = scenario_signature
        row["scenario_signature_sha256"] = canonical_sha256(scenario_signature)
    physical_signature = {
        "first_target_contact_ms": first,
        "target_contact_duration_ms": int(np.count_nonzero(interval)),
        "leading_foot": leading,
        "precontact_phase_20ms": precontact_phase,
        "peak_transition_displacement_m": peak_displacement,
        "peak_support_spread_m": peak_spread,
        "normalized_load_redistribution": redistribution,
        "normalized_peak_load_derivative": normalized_derivative,
        "pelvis_accel_rms": accel_rms,
        "pelvis_gyro_rms": gyro_rms,
    }
    row["physical_signature"] = physical_signature
    row["physical_signature_sha256"] = canonical_sha256(physical_signature)
    row["model_outputs_present"] = False
    row["intent_match"] = bool(row["valid"] and intent_match)
    row["intent_mismatch"] = bool(row["valid"] and not row["intent_match"])


def _summary(rows: Sequence[Mapping[str, Any]], elapsed_s: float) -> dict[str, Any]:
    return {
        "run_count": len(rows),
        "valid_count": sum(bool(row["valid"]) for row in rows),
        "invalid_count": sum(not bool(row["valid"]) for row in rows),
        "outcome_counts": dict(
            sorted(
                Counter(str(row["objective_physical_outcome"]) for row in rows).items()
            )
        ),
        "invalid_reason_counts": dict(
            sorted(
                Counter(
                    str(row["invalid_reason"])
                    for row in rows
                    if row["invalid_reason"] is not None
                ).items()
            )
        ),
        "precontact_phase_counts": dict(
            sorted(
                Counter(
                    str(row["target_contact_summary"]["precontact_phase"])
                    for row in rows
                    if row["target_contact_summary"]["first_sample"] is not None
                ).items()
            )
        ),
        "elapsed_s": elapsed_s,
        "model_inference_runs": 0,
    }


def build_failed_discovery_calibration_table(
    dataset_path: Path, output_path: Path
) -> dict[str, Any]:
    """Materialize the failed study's Discovery-only physical audit table."""
    manifest = json.loads((dataset_path / "manifest.json").read_text(encoding="utf-8"))
    table: list[dict[str, Any]] = []
    contract = {
        "precontact_phase_lookback_ms": 20,
        "benign_post_target_followup_ms": 1000,
        "support_post_event_followup_ms": 1000,
    }
    for historical in manifest["runs"]:
        if historical["split"] != "STUDY_DISCOVERY":
            continue
        path = dataset_path / str(historical["file"])
        if sha256_file(path) != str(historical["file_sha256"]):
            raise RuntimeError(f"failed-study run integrity failed: {path.name}")
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        corrected = dict(historical)
        corrected["target_contact_summary"] = dict(historical["target_contact_summary"])
        group = str(historical["group"])
        corrected["group"] = "sand_benign" if "sand_benign" in group else group
        if group == "ordinary_support_control":
            corrected["designed_side"] = historical["designed_side_topology"]
        _calibration_result_summary(corrected, arrays, contract)
        first = corrected["target_contact_summary"]["first_sample"]
        fall = historical["fall_censor_summary"]["first_fall_sample"]
        if first is None:
            fall_relation = "PRE_TARGET_FALL" if fall is not None else "NO_TARGET"
        elif fall is None:
            fall_relation = "NO_FALL"
        else:
            fall_relation = "POST_TARGET_FALL"
        physical = historical.get("physical_diversity_summary", {})
        table.append(
            {
                "run_id": historical["run_id"],
                "source_terrain": historical["source_terrain"],
                "speed_mps": historical["speed_mps"],
                "group": group,
                "patch_start_x_m": historical["patch_start_x_m"],
                "patch_width_m": historical["patch_width_m"],
                "start_stratum": historical.get("start_stratum"),
                "width_stratum": historical.get("width_stratum"),
                "topology": historical["sink_pattern"],
                "phase_assignment": historical.get("phase_assignment"),
                "realization_cohort": historical.get("realization_cohort"),
                "severity_intent": historical.get("severity_intent"),
                "historical_valid": historical["valid"],
                "historical_invalid_reason": historical["invalid_reason"],
                "fall_sample": fall,
                "fall_relation": fall_relation,
                "target_contact_sample": first,
                "target_reached_before_censor": first is not None,
                "contact_sample_phase": corrected["target_contact_summary"][
                    "contact_sample_phase"
                ],
                "precontact_phase_20ms": corrected["target_contact_summary"][
                    "precontact_phase"
                ],
                "leading_foot": corrected["target_contact_summary"]["leading_foot"],
                "loaded_side_at_contact": corrected["target_contact_summary"][
                    "loaded_side_at_contact"
                ],
                "contact_side": historical["target_contact_summary"].get(
                    "target_contact_side"
                ),
                "contact_duration_ms": corrected["target_contact_summary"][
                    "duration_ms_before_censor"
                ],
                "post_target_observation_ms": corrected["target_contact_summary"][
                    "post_target_observation_ms"
                ],
                "normalized_load_redistribution": physical.get(
                    "normalized_load_redistribution"
                ),
                "normalized_peak_load_derivative": physical.get(
                    "normalized_peak_load_derivative"
                ),
                "balanced_displacement_m": corrected["physical_signature"][
                    "peak_transition_displacement_m"
                ],
                "support_spread_m": corrected["physical_signature"][
                    "peak_support_spread_m"
                ],
                "historical_physical_outcome": historical["objective_physical_outcome"],
                "corrected_calibration_valid": corrected["valid"],
                "corrected_calibration_invalid_reason": corrected["invalid_reason"],
                "corrected_calibration_outcome": corrected[
                    "objective_physical_outcome"
                ],
            }
        )
    result = {
        "schema_version": 1,
        "evidence_scope": "FAILED_STUDY_DISCOVERY_ONLY",
        "model_fields": [],
        "run_count": len(table),
        "runs": table,
    }
    output_path.parent.mkdir(parents=True, exist_ok=False)
    _write_json(output_path, result)
    return {
        "run_count": len(table),
        "sha256": sha256_file(output_path),
        "path": str(output_path),
    }


def expand_sand_benign_redesign(
    document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand the frozen redesigned study into exact fresh run specifications."""
    matrix = document["scenario_matrix"]
    rows: list[dict[str, Any]] = []
    group_codes = {
        "broad_sand_benign": "bb",
        "near_hazard_sand_benign": "nh",
        "ordinary_support_control": "os",
    }
    for split in REDESIGN_SPLITS:
        split_code = "d" if split == "REDESIGNED_DISCOVERY" else "c"
        variants_by_group = matrix["split_variants"][split]
        for cell in matrix["source_speed_cells"]:
            source = str(cell["source_terrain"])
            source_code = "c" if source == "concrete" else "m"
            speed = float(cell["speed_mps"])
            speed_code = f"{int(round(speed * 100)):03d}"
            for group in (
                "broad_sand_benign",
                "near_hazard_sand_benign",
                "ordinary_support_control",
            ):
                anchors = list(cell["anchors"][group])
                variants = list(variants_by_group[group])
                mechanics = dict(matrix["fixed_mechanics"][group])
                for index, variant in enumerate(variants, start=1):
                    anchor = anchors[(index - 1) % len(anchors)]
                    row = {
                        **mechanics,
                        "run_id": (
                            f"sbgr_{split_code}_{group_codes[group]}_"
                            f"{source_code}_{speed_code}_{index:02d}"
                        ),
                        "scenario_family": group,
                        "group": group,
                        "split": split,
                        "source_terrain": source,
                        "speed_mps": speed,
                        "patch_start_x_m": round(
                            float(anchor["patch_start_x_m"])
                            + float(variant["start_delta_m"]),
                            3,
                        ),
                        "patch_width_m": round(
                            float(anchor["patch_width_m"])
                            + float(variant["width_delta_m"]),
                            3,
                        ),
                        "sink_pattern": str(anchor["sink_pattern"]),
                        "realization_id": str(variant["id"]),
                    }
                    if group == "ordinary_support_control":
                        row["designed_side"] = str(anchor["designed_side"])
                    rows.append(row)
        for index, template in enumerate(
            matrix["delayed_support_templates"][split], start=1
        ):
            source = str(template["source_terrain"])
            source_code = "c" if source == "concrete" else "m"
            rows.append(
                {
                    **dict(matrix["fixed_mechanics"]["delayed_support_control"]),
                    **dict(template),
                    "run_id": f"sbgr_{split_code}_ds_{source_code}_{index:02d}",
                    "scenario_family": "delayed_support_control",
                    "group": "delayed_support_control",
                    "split": split,
                    "speed_mps": 0.25,
                    "designed_side": "LEFT",
                    "realization_id": str(template["id"]),
                }
            )
    return rows


def redesigned_component_hashes(
    document: Mapping[str, Any],
) -> dict[str, str]:
    keys = {
        "REDESIGNED_PARAMETER_DOMAIN_SHA": "parameter_domain",
        "REDESIGNED_SCENARIO_MATRIX_SHA": "scenario_matrix",
        "REDESIGNED_SPLIT_PLAN_SHA": "split_plan",
        "REDESIGNED_PHYSICAL_LABEL_CONTRACT_SHA": "physical_label_contract",
        "REDESIGNED_GENERATION_GATE_SHA": "generation_gates",
        "REDESIGNED_DIVERSITY_METRIC_SHA": "diversity_metrics",
        "REDESIGNED_CONFIRMATION_PROTOCOL_SHA": "confirmation_protocol",
    }
    return {name: canonical_sha256(document[key]) for name, key in keys.items()}


def validate_sand_benign_redesign(
    root: Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed on matrix, split, overlap, and deterministic hash drift."""
    rows = expand_sand_benign_redesign(document)
    matrix = document["scenario_matrix"]
    expected_counts = dict(matrix["counts"])
    split_counts = Counter(str(row["split"]) for row in rows)
    group_counts = Counter(str(row["group"]) for row in rows)
    if len(rows) != int(expected_counts["total"]):
        raise ValueError("redesigned total count changed")
    if dict(split_counts) != {
        "REDESIGNED_DISCOVERY": int(expected_counts["REDESIGNED_DISCOVERY"]),
        "REDESIGNED_CONFIRMATION": int(expected_counts["REDESIGNED_CONFIRMATION"]),
    }:
        raise ValueError("redesigned split counts changed")
    for group in (
        "broad_sand_benign",
        "near_hazard_sand_benign",
        "ordinary_support_control",
        "delayed_support_control",
    ):
        if group_counts[group] != int(expected_counts[group]):
            raise ValueError(f"redesigned group count changed: {group}")
    ids = [str(row["run_id"]) for row in rows]
    signatures = [_signature(row) for row in rows]
    if len(set(ids)) != len(ids) or len(set(signatures)) != len(signatures):
        raise ValueError("redesigned matrix has duplicate IDs or signatures")
    historical, provenance = _historical_signatures(
        root, document["historical_signature_policy"]["manifests"]
    )
    historical_overlap = len(set(signatures) & historical)
    if historical_overlap:
        raise ValueError("redesigned matrix overlaps historical signatures")
    if any(
        float(row["patch_start_x_m"]) == 0.362 or float(row["patch_width_m"]) == 0.735
        for row in rows
    ):
        raise ValueError("consumed HOLDOUT coordinate entered redesign")
    near_policy = document["historical_signature_policy"]["cross_split_near_duplicate"]
    historical_audit = _historical_overlap_audit(
        root,
        document["historical_signature_policy"]["manifests"],
        rows,
        near_policy,
    )
    if historical_audit["exact_total"]:
        raise ValueError("redesigned matrix overlaps historical exact signatures")
    near_pairs: list[tuple[str, str]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if left["split"] == right["split"]:
                continue
            same_domain = all(
                left[key] == right[key]
                for key in (
                    "source_terrain",
                    "target_terrain",
                    "speed_mps",
                    "slip_pattern",
                    "sink_pattern",
                    "sink_severity",
                    "support_pattern",
                )
            )
            near_geometry = abs(
                float(left["patch_start_x_m"]) - float(right["patch_start_x_m"])
            ) < float(near_policy["patch_start_difference_m_exclusive"]) and abs(
                float(left["patch_width_m"]) - float(right["patch_width_m"])
            ) < float(near_policy["patch_width_difference_m_exclusive"])
            if same_domain and near_geometry:
                near_pairs.append((str(left["run_id"]), str(right["run_id"])))
    if near_pairs:
        raise ValueError(f"redesigned splits have near duplicates: {near_pairs[:3]}")
    computed = redesigned_component_hashes(document)
    expected = {
        key: value
        for key, value in document.get("design_hashes", {}).items()
        if key != "SAND_BENIGN_GENERALIZATION_STUDY_REDESIGN_SHA"
    }
    if expected and computed != expected:
        raise ValueError("redesigned component hashes changed")
    bundle = {
        "experiment_id": document["experiment"]["id"],
        "dataset_id": document["dataset_plan"]["dataset_id"],
        "counts": matrix["counts"],
        "component_hashes": computed,
    }
    redesign_sha = canonical_sha256(bundle)
    expected_redesign = document.get("design_hashes", {}).get(
        "SAND_BENIGN_GENERALIZATION_STUDY_REDESIGN_SHA"
    )
    if expected_redesign and redesign_sha != expected_redesign:
        raise ValueError("redesigned study hash changed")
    return {
        "run_count": len(rows),
        "split_counts": dict(split_counts),
        "group_counts": dict(group_counts),
        "unique_run_ids": len(set(ids)),
        "unique_signatures": len(set(signatures)),
        "historical_signature_overlap": historical_overlap,
        "cross_split_exact_overlap": len(
            {_signature(row) for row in rows if row["split"] == REDESIGN_SPLITS[0]}
            & {_signature(row) for row in rows if row["split"] == REDESIGN_SPLITS[1]}
        ),
        "cross_split_parameter_near_duplicates": len(near_pairs),
        "scenario_signature_sha256": canonical_sha256(
            [list(value) for value in signatures]
        ),
        "split_sha256": {
            split: canonical_sha256(
                [row["run_id"] for row in rows if row["split"] == split]
            )
            for split in REDESIGN_SPLITS
        },
        "historical_manifests": provenance,
        "historical_contamination": historical_audit,
        "component_hashes": computed,
        "redesign_sha256": redesign_sha,
    }


def _redesigned_outcome_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = Counter(str(row["objective_physical_outcome"]) for row in rows)
    return {
        name: int(counts.get(name, 0))
        for name in (
            "STRICT_BENIGN",
            "SUPPORT",
            "SLIP",
            "DUAL_HAZARD",
            "INVALID",
        )
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(finite.size),
        "minimum": None if not finite.size else float(np.min(finite)),
        "median": None if not finite.size else float(np.median(finite)),
        "maximum": None if not finite.size else float(np.max(finite)),
        "span": None if not finite.size else float(np.ptp(finite)),
    }


def _physical_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    scales = {
        "first_target_contact_ms": 300.0,
        "target_contact_duration_ms": 300.0,
        "peak_transition_displacement_m": 0.020,
        "peak_support_spread_m": 0.010,
        "normalized_load_redistribution": 0.25,
        "normalized_peak_load_derivative": 50.0,
        "pelvis_accel_rms": 9.80665,
        "pelvis_gyro_rms": 1.0,
    }
    return float(
        np.sqrt(
            sum(
                ((float(left[key]) - float(right[key])) / scale) ** 2
                for key, scale in scales.items()
            )
        )
    )


def _physical_near_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["valid"]]
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(valid):
        for right in valid[index + 1 :]:
            same_domain = all(
                left[key] == right[key]
                for key in (
                    "group",
                    "source_terrain",
                    "speed_mps",
                    "sink_pattern",
                )
            )
            same_category = all(
                left["physical_signature"][key] == right["physical_signature"][key]
                for key in ("leading_foot", "precontact_phase_20ms")
            )
            if not same_domain or not same_category:
                continue
            distance = _physical_distance(
                left["physical_signature"], right["physical_signature"]
            )
            if distance <= 0.10:
                pairs.append(
                    {
                        "left_run_id": left["run_id"],
                        "right_run_id": right["run_id"],
                        "cross_split": left["split"] != right["split"],
                        "distance": distance,
                    }
                )
    return pairs


def audit_sand_benign_redesigned_manifest(
    manifest: Mapping[str, Any],
    design: Mapping[str, Any],
    matrix_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only the frozen model-blind physical generation gates."""
    rows = list(manifest["runs"])
    sand = [row for row in rows if "sand_benign" in str(row["group"])]
    strict_sand = [
        row for row in sand if row["objective_physical_outcome"] == "STRICT_BENIGN"
    ]
    cells = [
        (source, speed)
        for source in ("concrete", "marble")
        for speed in (0.20, 0.25, 0.30)
    ]

    def selected_cell(
        selected: Sequence[Mapping[str, Any]], source: str, speed: float
    ) -> list[Mapping[str, Any]]:
        return [
            row
            for row in selected
            if row["source_terrain"] == source and float(row["speed_mps"]) == speed
        ]

    source_speed: dict[str, dict[str, Any]] = {}
    mild: dict[str, dict[str, Any]] = {}
    moderate: dict[str, dict[str, Any]] = {}
    phase: dict[str, Any] = {}
    topology: dict[str, Any] = {}
    entry_by_split: dict[str, Any] = {}
    support_controls: dict[str, Any] = {}
    gates: dict[str, dict[str, Any]] = {}

    def gate(name: str, requirement: str, actual: Any, passed: bool) -> None:
        gates[name] = {
            "requirement": requirement,
            "actual": actual,
            "passed": bool(passed),
        }

    generation_gates = design["generation_gates"]
    for split in REDESIGN_SPLITS:
        split_sand = [row for row in sand if row["split"] == split]
        split_strict = [row for row in strict_sand if row["split"] == split]
        source_speed[split] = {}
        mild[split] = {}
        moderate[split] = {}
        for source, speed in cells:
            key = f"{source}/{speed:.2f}"
            cell_rows = selected_cell(split_sand, source, speed)
            cell_strict = selected_cell(split_strict, source, speed)
            source_speed[split][key] = {
                "planned_sand": len(cell_rows),
                "valid": sum(bool(row["valid"]) for row in cell_rows),
                "strict_benign": len(cell_strict),
                **_redesigned_outcome_counts(cell_rows),
            }
            mild_rows = [
                row for row in cell_rows if row["group"] == "broad_sand_benign"
            ]
            mild[split][key] = {
                "planned": len(mild_rows),
                "valid": sum(bool(row["valid"]) for row in mild_rows),
                "strict_benign_mild": sum(
                    row["objective_physical_outcome"] == "STRICT_BENIGN"
                    and row["actual_benign_severity"] == "LOW"
                    for row in mild_rows
                ),
                **_redesigned_outcome_counts(mild_rows),
            }
            moderate_rows = [
                row for row in cell_rows if row["group"] == "near_hazard_sand_benign"
            ]
            moderate[split][key] = {
                "planned": len(moderate_rows),
                "valid": sum(bool(row["valid"]) for row in moderate_rows),
                "strict_boundary_moderate": sum(
                    row["objective_physical_outcome"] == "STRICT_BENIGN"
                    and row["actual_benign_severity"] == "MEDIUM"
                    for row in moderate_rows
                ),
                **_redesigned_outcome_counts(moderate_rows),
            }

        eligible = split_strict
        phase_counts = Counter(
            str(row["target_contact_summary"]["precontact_phase"]) for row in eligible
        )
        cell_phase: dict[str, Any] = {}
        cells_with_both = 0
        cells_with_usable = 0
        for source, speed in cells:
            key = f"{source}/{speed:.2f}"
            phases = Counter(
                str(row["target_contact_summary"]["precontact_phase"])
                for row in selected_cell(eligible, source, speed)
            )
            principal = {
                name
                for name in ("LEFT_SINGLE_SUPPORT", "RIGHT_SINGLE_SUPPORT")
                if phases[name] > 0
            }
            cells_with_both += len(principal) == 2
            cells_with_usable += bool(principal)
            cell_phase[key] = {
                "counts": dict(sorted(phases.items())),
                "principal_phases": sorted(principal),
            }
        leading = Counter(
            str(row["target_contact_summary"]["leading_foot"]) for row in eligible
        )
        phase[split] = {
            "eligible_population": "STRICT_SAND_BENIGN",
            "counts": dict(sorted(phase_counts.items())),
            "leading_foot": dict(sorted(leading.items())),
            "source_speed_cells": cell_phase,
            "cells_with_both_principal_phases": cells_with_both,
            "cells_with_usable_phase": cells_with_usable,
            "concrete_0.25_predeclared_exception_used": (
                len(cell_phase["concrete/0.25"]["principal_phases"]) == 1
            ),
        }
        topology[split] = {
            name: {
                "planned": sum(row["sink_pattern"] == name for row in split_sand),
                "strict_benign": sum(
                    row["sink_pattern"] == name for row in split_strict
                ),
                "leading_foot": dict(
                    sorted(
                        Counter(
                            str(row["target_contact_summary"]["leading_foot"])
                            for row in split_strict
                            if row["sink_pattern"] == name
                        ).items()
                    )
                ),
                "precontact_phase": dict(
                    sorted(
                        Counter(
                            str(row["target_contact_summary"]["precontact_phase"])
                            for row in split_strict
                            if row["sink_pattern"] == name
                        ).items()
                    )
                ),
            }
            for name in ("transition_left", "transition_right")
        }
        entry_values = [
            float(row["target_contact_summary"]["first_sample"]) for row in split_strict
        ]
        entry_by_split[split] = _numeric_summary(entry_values)

        ordinary_rows = [
            row
            for row in rows
            if row["split"] == split and row["group"] == "ordinary_support_control"
        ]
        delayed_rows = [
            row
            for row in rows
            if row["split"] == split and row["group"] == "delayed_support_control"
        ]
        ordinary_by_cell = {
            f"{source}/{speed:.2f}": {
                "planned": len(selected_cell(ordinary_rows, source, speed)),
                "actual_support": sum(
                    row["objective_physical_outcome"] == "SUPPORT"
                    for row in selected_cell(ordinary_rows, source, speed)
                ),
                "qualified_support": sum(
                    bool(row["intent_match"])
                    for row in selected_cell(ordinary_rows, source, speed)
                ),
            }
            for source, speed in cells
        }
        support_controls[split] = {
            "ordinary": {
                "planned": len(ordinary_rows),
                "actual_support": sum(
                    row["objective_physical_outcome"] == "SUPPORT"
                    for row in ordinary_rows
                ),
                "qualified_support": sum(
                    bool(row["intent_match"]) for row in ordinary_rows
                ),
                "by_source_speed": ordinary_by_cell,
                "outcomes": _redesigned_outcome_counts(ordinary_rows),
            },
            "delayed": {
                "planned": len(delayed_rows),
                "actual_support": sum(
                    row["objective_physical_outcome"] == "SUPPORT"
                    for row in delayed_rows
                ),
                "qualified_support": sum(
                    bool(row["intent_match"]) for row in delayed_rows
                ),
                "outcomes": _redesigned_outcome_counts(delayed_rows),
            },
        }

        strict_min = int(generation_gates["strict_sand"]["per_split_min"])
        gate(
            f"yield/{split}/strict_sand",
            f">={strict_min}",
            len(split_strict),
            len(split_strict) >= strict_min,
        )
        mild_total = sum(
            values["strict_benign_mild"] for values in mild[split].values()
        )
        mild_min = int(generation_gates["strict_sand"]["broad_mild_per_split_min"])
        gate(
            f"yield/{split}/broad_mild",
            f">={mild_min}",
            mild_total,
            mild_total >= mild_min,
        )
        moderate_total = sum(
            values["strict_boundary_moderate"] for values in moderate[split].values()
        )
        moderate_min = int(
            generation_gates["strict_sand"]["boundary_adjacent_moderate_per_split_min"]
        )
        gate(
            f"yield/{split}/boundary_moderate",
            f">={moderate_min}",
            moderate_total,
            moderate_total >= moderate_min,
        )
        for source, speed in cells:
            key = f"{source}/{speed:.2f}"
            strict_cell_min = int(
                generation_gates["strict_sand"]["per_source_speed_split_min"]
            )
            gate(
                f"yield/{split}/{key}/strict_sand",
                f">={strict_cell_min}",
                source_speed[split][key]["strict_benign"],
                source_speed[split][key]["strict_benign"] >= strict_cell_min,
            )
            moderate_cell_min = int(
                generation_gates["strict_sand"][
                    "boundary_adjacent_moderate_per_source_speed_split_min"
                ]
            )
            gate(
                f"yield/{split}/{key}/boundary_moderate",
                f">={moderate_cell_min}",
                moderate[split][key]["strict_boundary_moderate"],
                moderate[split][key]["strict_boundary_moderate"] >= moderate_cell_min,
            )
            ordinary_cell_min = int(
                generation_gates["controls"][
                    "ordinary_support_each_source_speed_cell_min"
                ]
            )
            gate(
                f"yield/{split}/{key}/ordinary_support",
                f">={ordinary_cell_min}",
                ordinary_by_cell[key]["qualified_support"],
                ordinary_by_cell[key]["qualified_support"] >= ordinary_cell_min,
            )
        ordinary_min = int(
            generation_gates["controls"]["ordinary_support_per_split_min"]
        )
        gate(
            f"yield/{split}/ordinary_support",
            f">={ordinary_min}",
            support_controls[split]["ordinary"]["qualified_support"],
            support_controls[split]["ordinary"]["qualified_support"] >= ordinary_min,
        )
        delayed_min = int(generation_gates["controls"]["delayed_support_per_split_min"])
        gate(
            f"yield/{split}/delayed_support",
            f">={delayed_min}",
            support_controls[split]["delayed"]["qualified_support"],
            support_controls[split]["delayed"]["qualified_support"] >= delayed_min,
        )
        principal_count = sum(
            phase_counts[name] > 0
            for name in ("LEFT_SINGLE_SUPPORT", "RIGHT_SINGLE_SUPPORT")
        )
        gate(
            f"diversity/{split}/principal_phases",
            ">=2",
            principal_count,
            principal_count >= 2,
        )
        cell_both_min = int(
            generation_gates["phase_contact"][
                "source_speed_cells_with_both_precontact_single_support_categories_per_split_min"
            ]
        )
        gate(
            f"diversity/{split}/cells_with_both_phases",
            f">={cell_both_min}",
            cells_with_both,
            cells_with_both >= cell_both_min,
        )
        gate(
            f"diversity/{split}/every_cell_usable_phase",
            "6/6",
            cells_with_usable,
            cells_with_usable == 6,
        )
        gate(
            f"diversity/{split}/both_leading_feet",
            "LEFT_AND_RIGHT",
            dict(sorted(leading.items())),
            leading["LEFT"] > 0 and leading["RIGHT"] > 0,
        )
        entry_span = entry_by_split[split]["span"]
        entry_min = int(
            generation_gates["phase_contact"]["contact_entry_time_global_span_ms_min"]
        )
        gate(
            f"diversity/{split}/entry_time_span_ms",
            f">={entry_min}",
            entry_span,
            entry_span is not None and float(entry_span) >= entry_min,
        )

    objective_valid = sum(bool(row["valid"]) for row in rows)
    gate(
        "execution/attempted",
        "176",
        manifest["attempted_run_count"],
        manifest["attempted_run_count"] == 176,
    )
    gate("execution/completed", "176", len(rows), len(rows) == 176)
    gate(
        "execution/adaptive_backfill",
        "0",
        manifest["adaptive_backfill_count"],
        manifest["adaptive_backfill_count"] == 0,
    )
    gate(
        "execution/replacement",
        "0",
        manifest["replacement_run_count"],
        manifest["replacement_run_count"] == 0,
    )
    overall_min = int(generation_gates["overall"]["objective_valid_min"])
    gate(
        "yield/overall_objective_valid",
        f">={overall_min}",
        objective_valid,
        objective_valid >= overall_min,
    )
    severe_count = sum(row["sink_severity"] == "severe" for row in rows)
    gate("integrity/severe_excluded", "0", severe_count, severe_count == 0)
    gate(
        "integrity/unique_run_ids",
        "176",
        len({row["run_id"] for row in rows}),
        len({row["run_id"] for row in rows}) == 176,
    )
    scenario_fraction = matrix_audit["unique_signatures"] / max(len(rows), 1)
    gate(
        "integrity/unique_scenario_signature_fraction",
        "1.0",
        scenario_fraction,
        scenario_fraction == 1.0,
    )
    contamination = matrix_audit["historical_contamination"]
    gate(
        "integrity/historical_exact_overlap",
        "0",
        contamination["exact_total"],
        contamination["exact_total"] == 0,
    )
    gate(
        "integrity/historical_run_id_reuse",
        "0",
        contamination["run_id_reuse_total"],
        contamination["run_id_reuse_total"] == 0,
    )
    gate(
        "integrity/cross_split_exact_overlap",
        "0",
        matrix_audit["cross_split_exact_overlap"],
        matrix_audit["cross_split_exact_overlap"] == 0,
    )
    gate(
        "integrity/cross_split_parameter_near_overlap",
        "0",
        matrix_audit["cross_split_parameter_near_duplicates"],
        matrix_audit["cross_split_parameter_near_duplicates"] == 0,
    )
    gate(
        "integrity/model_outputs",
        "0",
        sum(bool(row["model_outputs_present"]) for row in rows),
        all(not row["model_outputs_present"] for row in rows),
    )

    physical_hashes = [
        str(row["physical_signature_sha256"]) for row in rows if row["valid"]
    ]
    unique_physical = len(set(physical_hashes))
    physical_ratio = unique_physical / max(len(physical_hashes), 1)
    physical_min = float(
        design["diversity_metrics"]["unique_physical_signature_fraction_min"]
    )
    gate(
        "diversity/unique_physical_signature_fraction",
        f">={physical_min}",
        physical_ratio,
        physical_ratio >= physical_min,
    )
    near_pairs = _physical_near_pairs(rows)

    entry_axes: dict[str, Any] = {}
    eligible_all = strict_sand
    for dimension, levels in (
        ("source_terrain", ("concrete", "marble")),
        ("speed_mps", (0.20, 0.25, 0.30)),
        ("sink_pattern", ("transition_left", "transition_right")),
    ):
        entry_axes[dimension] = {
            str(level): _numeric_summary(
                [
                    float(row["target_contact_summary"]["first_sample"])
                    for row in eligible_all
                    if row[dimension] == level
                ]
            )
            for level in levels
        }
    entry_timing = {
        "population": "STRICT_SAND_BENIGN",
        "overall": _numeric_summary(
            [
                float(row["target_contact_summary"]["first_sample"])
                for row in eligible_all
            ]
        ),
        "by_split": entry_by_split,
        "by_axis": entry_axes,
    }

    invalid = [row for row in rows if not row["valid"]]
    pretarget = sum(row["invalid_reason"] == "pretarget_fall" for row in invalid)
    insufficient = sum(
        row["invalid_reason"]
        in (
            "insufficient_post_target_observation",
            "insufficient_post_support_observation",
        )
        for row in invalid
    )
    posttarget_fall = sum(
        row["target_contact_summary"]["first_sample"] is not None
        and row["fall_censor_summary"]["first_fall_sample"] is not None
        for row in invalid
    )
    invalidity = {
        "total": len(invalid),
        "pretarget_fall": pretarget,
        "posttarget_or_censor": posttarget_fall,
        "insufficient_followup": insufficient,
        "other": len(invalid) - pretarget - insufficient,
        "reason_counts": dict(
            sorted(Counter(str(row["invalid_reason"]) for row in invalid).items())
        ),
    }

    integrity_pass = all(
        item["passed"]
        for name, item in gates.items()
        if name.startswith("execution/") or name.startswith("integrity/")
    )
    yield_pass = all(
        item["passed"] for name, item in gates.items() if name.startswith("yield/")
    )
    diversity_pass = all(
        item["passed"] for name, item in gates.items() if name.startswith("diversity/")
    )
    if not integrity_pass:
        verdict = "SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION_INVALID"
    elif not yield_pass:
        verdict = (
            "SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION_YIELD_INSUFFICIENT"
        )
    elif not diversity_pass:
        verdict = "SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION_DIVERSITY_INSUFFICIENT"
    else:
        verdict = "SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION_READY"
    split_outcomes = {
        split: _redesigned_outcome_counts(
            [row for row in rows if row["split"] == split]
        )
        for split in REDESIGN_SPLITS
    }
    return {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "planned_runs": 176,
        "attempted_runs": manifest["attempted_run_count"],
        "completed_runs": len(rows),
        "objective_valid": objective_valid,
        "objective_invalid": len(rows) - objective_valid,
        "split_outcomes": split_outcomes,
        "overall_outcomes": _redesigned_outcome_counts(rows),
        "invalidity": invalidity,
        "sand_source_speed": source_speed,
        "mild": mild,
        "moderate_boundary": moderate,
        "phase_diversity": phase,
        "topology_contact": topology,
        "entry_timing": entry_timing,
        "support_controls": support_controls,
        "physical_signatures": {
            "valid_count": len(physical_hashes),
            "unique_count": unique_physical,
            "unique_fraction": physical_ratio,
            "exact_duplicates": len(physical_hashes) - unique_physical,
            "near_duplicate_definition": "repository_scaled_distance_le_0.10_non_gating",
            "near_duplicate_count": len(near_pairs),
            "cross_split_near_duplicate_count": sum(
                bool(pair["cross_split"]) for pair in near_pairs
            ),
            "near_pairs": near_pairs,
        },
        "historical_contamination": contamination,
        "historical_parameter_near_diagnostic": {
            "definition": "cross_split_threshold_reused_as_non_gating_reference_only",
            "count": contamination["near_total"],
            "by_reference": contamination["near_by_reference"],
            "frozen_historical_gate": "EXACT_ONLY",
        },
        "severe_domain_status": "SEVERE_DOMAIN_EXCLUDED_BY_PHYSICAL_CALIBRATION",
        "generation_gates": gates,
        "integrity_pass": integrity_pass,
        "yield_pass": yield_pass,
        "diversity_pass": diversity_pass,
        "generation_verdict": verdict,
        "model_inference_runs": 0,
        "confirmation_model_analysis": False,
    }


def collect_sand_calibration_batch(
    root: Path,
    config_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Execute one immutable, deterministic, model-blind calibration batch."""
    document = _load_yaml(config_path)
    calibration = document["calibration"]
    audit = validate_sand_calibration_config(root, document)
    policy_path = root / str(calibration["policy_path"])
    if sha256_file(policy_path) != str(calibration["policy_sha256"]):
        raise RuntimeError("walking policy changed before calibration")
    simulator_config = root / str(calibration["simulator_config_path"])
    if sha256_file(simulator_config) != str(calibration["simulator_config_sha256"]):
        raise RuntimeError("simulator config changed before calibration")
    guard = json.loads(
        (root / str(calibration["consumed_holdout_guard_path"])).read_text(
            encoding="utf-8"
        )
    )
    if guard.get("guard_after") != 1 or guard.get("scientific_open_count") != 1:
        raise RuntimeError("consumed Generalization HOLDOUT guard changed")

    output_path = root / str(calibration["dataset_path"])
    partial_path = output_path.with_name(f"{output_path.name}_partial")
    if output_path.exists() or partial_path.exists():
        raise RuntimeError(f"calibration output already exists: {output_path}")
    partial_path.mkdir(parents=True)
    config_sha = sha256_file(config_path)
    freeze = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_FIRST_SIMULATION",
        "experiment_id": document["experiment"]["id"],
        "dataset_id": calibration["dataset_id"],
        "batch_index": calibration["batch_index"],
        "deterministic_seed": calibration["deterministic_seed"],
        "run_count": audit["run_count"],
        "config_path": str(config_path.relative_to(root)),
        "config_sha256": config_sha,
        "scenario_matrix_sha256": audit["scenario_matrix_sha256"],
        "scenario_signature_sha256": audit["scenario_signature_sha256"],
        "source_commit": calibration["source_commit"],
    }
    _write_json(partial_path / "pre_simulation_freeze.json", freeze)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        scenarios = _expand_calibration_scenarios(calibration)
        for index, specification in enumerate(scenarios, start=1):
            result = run_simulation(
                SimulationConfig(
                    physics_timestep_s=float(calibration["physics_timestep_s"]),
                    sensor_rate_hz=int(calibration["sensor_rate_hz"]),
                    duration_s=float(calibration["duration_s"]),
                    command_speed_mps=float(specification["speed_mps"]),
                    policy_path=policy_path,
                    terrain=str(specification["target_terrain"]),
                    slip_pattern=str(specification["slip_pattern"]),
                    sink_pattern=str(specification["sink_pattern"]),
                    sink_severity=str(specification["sink_severity"]),
                    patch_start_x_m=float(specification["patch_start_x_m"]),
                    patch_width_m=float(specification["patch_width_m"]),
                    headless=True,
                    sink_support_pattern=str(specification["support_pattern"]),
                    source_terrain=str(specification["source_terrain"]),
                ),
                observe_fsr=True,
                observe_foot_imu=False,
            )
            row, arrays = annotate_model_v2_result(
                _annotation_specification(specification), result
            )
            row["scenario_family"] = specification["scenario_family"]
            if result.stability is None:
                raise RuntimeError("calibration requires exact gait phase")
            arrays["gait_phase"] = np.asarray(
                result.stability.gait_phase, dtype=np.int8
            )
            _calibration_result_summary(row, arrays, calibration["label_contract"])
            filename = f"{specification['run_id']}.npz"
            run_path = partial_path / filename
            _write_deterministic_npz(run_path, arrays)
            row["file"] = filename
            row["file_sha256"] = sha256_file(run_path)
            row["size_bytes"] = run_path.stat().st_size
            row["execution_status"] = "COMPLETED"
            rows.append(row)
            if progress is not None:
                progress(
                    f"generated {index}/{audit['run_count']}: {specification['run_id']}"
                )

        elapsed_s = time.monotonic() - started
        summary = _summary(rows, elapsed_s)
        manifest = {
            "schema_version": 1,
            "dataset_id": calibration["dataset_id"],
            "created_at": calibration["created_at"],
            "generation_source_commit": calibration["source_commit"],
            "config_path": str(config_path.relative_to(root)),
            "config_sha256": config_sha,
            "matrix_audit": audit,
            "policy_sha256": calibration["policy_sha256"],
            "simulator_config_sha256": calibration["simulator_config_sha256"],
            "model_blind": True,
            "model_inference_runs": 0,
            "adaptive_within_batch": False,
            "replacement_run_count": 0,
            "run_count": len(rows),
            "generation_order": [row["run_id"] for row in rows],
            "runs": rows,
        }
        _write_json(partial_path / "manifest.json", manifest)
        _write_json(partial_path / "summary.json", summary)
        manifest_sha = sha256_file(partial_path / "manifest.json")
        (partial_path / "manifest.sha256").write_text(
            f"{manifest_sha}  manifest.json\n", encoding="utf-8"
        )
        dataset_freeze = {
            "schema_version": 1,
            "dataset_id": calibration["dataset_id"],
            "manifest_sha256": manifest_sha,
            "summary_sha256": sha256_file(partial_path / "summary.json"),
            "pre_simulation_freeze_sha256": sha256_file(
                partial_path / "pre_simulation_freeze.json"
            ),
            "npz_sha256": {row["file"]: row["file_sha256"] for row in rows},
        }
        dataset_freeze["dataset_freeze_sha256"] = canonical_sha256(dataset_freeze)
        _write_json(partial_path / "dataset_freeze.json", dataset_freeze)
        (partial_path / "dataset_freeze.sha256").write_text(
            f"{sha256_file(partial_path / 'dataset_freeze.json')}  dataset_freeze.json\n",
            encoding="utf-8",
        )
        partial_path.rename(output_path)
        return output_path, summary
    except Exception:
        shutil.rmtree(partial_path, ignore_errors=True)
        raise


def load_sand_benign_redesigned_manifest(
    dataset_path: Path,
) -> Mapping[str, Any]:
    """Load redesigned metadata without deserializing either split's payload."""
    manifest_path = dataset_path / "manifest.json"
    expected = (dataset_path / "manifest.sha256").read_text(encoding="utf-8").split()[0]
    if sha256_file(manifest_path) != expected:
        raise ValueError("redesigned Sand manifest integrity failed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != REDESIGNED_DATASET_ID:
        raise ValueError("unexpected redesigned Sand dataset identity")
    return manifest


def load_sand_benign_redesigned_discovery_payload(
    dataset_path: Path, run_id: str
) -> dict[str, np.ndarray]:
    """Open Discovery only and reject Confirmation before NPZ access."""
    manifest = load_sand_benign_redesigned_manifest(dataset_path)
    row = next((item for item in manifest["runs"] if item["run_id"] == run_id), None)
    if row is None:
        raise KeyError(f"unknown redesigned Sand run: {run_id}")
    if row["split"] != "REDESIGNED_DISCOVERY":
        raise RuntimeError(
            "REDESIGNED_CONFIRMATION is SEALED_FOR_REDESIGNED_CONFIRMATION"
        )
    path = dataset_path / str(row["file"])
    if sha256_file(path) != str(row["file_sha256"]):
        raise ValueError(f"redesigned Sand run integrity failed: {run_id}")
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def verify_sand_benign_redesigned_dataset(dataset_path: Path) -> dict[str, Any]:
    """Recompute frozen file hashes without loading sealed waveform arrays."""
    manifest = load_sand_benign_redesigned_manifest(dataset_path)
    freeze_path = dataset_path / "dataset_freeze.json"
    expected_freeze_file = (
        (dataset_path / "dataset_freeze.sha256").read_text(encoding="utf-8").split()[0]
    )
    freeze_file_sha = sha256_file(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    semantic = dict(freeze)
    expected_semantic_sha = semantic.pop("REDESIGNED_STUDY_DATASET_FREEZE_SHA")
    npz_hashes = {
        str(row["file"]): sha256_file(dataset_path / str(row["file"]))
        for row in manifest["runs"]
    }
    expected_npz = {
        str(row["file"]): str(row["file_sha256"]) for row in manifest["runs"]
    }
    checks = {
        "dataset_freeze_file_sha": freeze_file_sha == expected_freeze_file,
        "dataset_freeze_semantic_sha": (
            canonical_sha256(semantic) == expected_semantic_sha
        ),
        "manifest_sha": (
            sha256_file(dataset_path / "manifest.json")
            == freeze["REDESIGNED_STUDY_MANIFEST_SHA"]
        ),
        "physical_audit_sha": (
            sha256_file(dataset_path / "physical_audit.json")
            == freeze["REDESIGNED_STUDY_PHYSICAL_AUDIT_SHA"]
        ),
        "confirmation_seal_sha": (
            sha256_file(dataset_path / "confirmation_seal.json")
            == freeze["confirmation_seal_sha256"]
        ),
        "npz_hashes": npz_hashes == expected_npz,
        "npz_aggregate_sha": (
            canonical_sha256(npz_hashes) == freeze["REDESIGNED_STUDY_NPZ_AGGREGATE_SHA"]
        ),
        "run_count": len(manifest["runs"]) == 176,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "run_count": len(manifest["runs"]),
        "dataset_freeze_file_sha256": freeze_file_sha,
        "dataset_freeze_semantic_sha256": expected_semantic_sha,
    }


def collect_sand_benign_redesigned_study(
    root: Path,
    execution_config_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Generate the exact frozen 176-run redesign with no model execution."""
    execution = _load_yaml(execution_config_path)
    if execution["experiment"]["id"] != REDESIGNED_GENERATION_ID:
        raise ValueError("unsupported redesigned Sand generation config")
    generation = execution["generation"]
    redesign_path = root / str(generation["redesign_config_path"])
    if sha256_file(redesign_path) != str(generation["redesign_config_sha256"]):
        raise RuntimeError("frozen Sand redesign file changed")
    design = _load_yaml(redesign_path)
    matrix_audit = validate_sand_benign_redesign(root, design)
    if matrix_audit["component_hashes"] != dict(generation["redesign_hashes"]):
        raise RuntimeError("frozen Sand redesign component hashes changed")
    if matrix_audit["redesign_sha256"] != str(generation["redesign_sha"]):
        raise RuntimeError("complete frozen Sand redesign hash changed")
    expected_matrix = {
        "scenario_signature_sha256": str(
            generation["expected_scenario_signature_sha256"]
        ),
        "split_sha256": dict(generation["expected_split_sha256"]),
    }
    if any(matrix_audit[key] != value for key, value in expected_matrix.items()):
        raise RuntimeError("expanded redesigned Sand matrix changed")
    specifications = expand_sand_benign_redesign(design)
    expected_counts = (
        int(generation["planned_total_runs"]),
        int(generation["planned_discovery_runs"]),
        int(generation["planned_confirmation_runs"]),
    )
    actual_counts = (
        len(specifications),
        matrix_audit["split_counts"]["REDESIGNED_DISCOVERY"],
        matrix_audit["split_counts"]["REDESIGNED_CONFIRMATION"],
    )
    if (
        str(generation["dataset_id"]) != REDESIGNED_DATASET_ID
        or expected_counts != actual_counts
    ):
        raise RuntimeError("redesigned Sand execution identity/counts changed")
    policy_path = root / str(generation["policy_path"])
    if sha256_file(policy_path) != str(generation["policy_sha256"]):
        raise RuntimeError("walking policy differs from redesigned freeze")
    simulator_path = root / str(generation["simulator_config_path"])
    if sha256_file(simulator_path) != str(generation["simulator_config_sha256"]):
        raise RuntimeError("simulator config differs from redesigned freeze")
    for category in ("implementation_artifacts", "protected_artifacts"):
        for artifact in generation[category]:
            path = root / str(artifact["path"])
            if sha256_file(path) != str(artifact["sha256"]):
                raise RuntimeError(f"frozen artifact changed: {artifact['path']}")
    guard_path = root / str(generation["consumed_holdout_guard_path"])
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    if guard.get("guard_after") != 1 or guard.get("scientific_open_count") != 1:
        raise RuntimeError("consumed Generalization HOLDOUT guard changed")
    required_guards = (
        "no_model_inference",
        "no_training",
        "no_hnm",
        "no_normalizer_fit",
        "no_threshold_search",
        "no_persistence_search",
        "no_architecture_search",
        "old_holdout_access_forbidden",
        "confirmation_model_analysis_forbidden",
        "no_adaptive_backfill",
    )
    if not all(bool(execution["protocol_guards"][key]) for key in required_guards):
        raise RuntimeError("redesigned execution protocol guard disabled")
    if (
        execution["protocol_guards"]["failed_study_reuse"]
        or execution["protocol_guards"]["pilot_reuse"]
    ):
        raise RuntimeError("historical run reuse enabled")

    output_path = root / str(generation["dataset_path"])
    partial_path = output_path.with_name(f"{output_path.name}_partial")
    if output_path.exists() or partial_path.exists():
        raise RuntimeError(f"redesigned Sand output already exists: {output_path}")
    partial_path.mkdir(parents=True)
    config_sha = sha256_file(execution_config_path)
    label_execution = dict(generation["label_execution"])
    pre_simulation_freeze = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_FIRST_SIMULATION",
        "dataset_id": REDESIGNED_DATASET_ID,
        "source_commit": str(generation["source_commit"]),
        "execution_config_path": str(execution_config_path.relative_to(root)),
        "execution_config_sha256": config_sha,
        "redesign_config_sha256": str(generation["redesign_config_sha256"]),
        "redesign_sha256": matrix_audit["redesign_sha256"],
        "redesign_hashes": matrix_audit["component_hashes"],
        "scenario_signature_sha256": matrix_audit["scenario_signature_sha256"],
        "split_sha256": matrix_audit["split_sha256"],
        "planned_run_count": len(specifications),
        "adaptive_backfill": False,
        "replacement_runs": 0,
        "model_inference": False,
    }
    _write_json(partial_path / "pre_simulation_freeze.json", pre_simulation_freeze)
    rows: list[dict[str, Any]] = []
    attempted = 0
    started = time.monotonic()
    try:
        for index, specification in enumerate(specifications, start=1):
            attempted += 1
            result = run_simulation(
                SimulationConfig(
                    physics_timestep_s=float(
                        design["dataset_plan"]["physics_timestep_s"]
                    ),
                    sensor_rate_hz=int(design["dataset_plan"]["sensor_rate_hz"]),
                    duration_s=float(design["dataset_plan"]["simulation_duration_s"]),
                    command_speed_mps=float(specification["speed_mps"]),
                    policy_path=policy_path,
                    terrain=str(specification["target_terrain"]),
                    slip_pattern=str(specification["slip_pattern"]),
                    sink_pattern=str(specification["sink_pattern"]),
                    sink_severity=str(specification["sink_severity"]),
                    patch_start_x_m=float(specification["patch_start_x_m"]),
                    patch_width_m=float(specification["patch_width_m"]),
                    headless=True,
                    sink_support_pattern=str(specification["support_pattern"]),
                    source_terrain=str(specification["source_terrain"]),
                ),
                observe_fsr=True,
                observe_foot_imu=False,
            )
            row, arrays = annotate_model_v2_result(
                _annotation_specification(specification), result
            )
            row["scenario_family"] = specification["scenario_family"]
            if result.stability is None:
                raise RuntimeError("redesigned study requires exact gait phase")
            arrays["gait_phase"] = np.asarray(
                result.stability.gait_phase, dtype=np.int8
            )
            _calibration_result_summary(row, arrays, label_execution)
            row["execution_status"] = "COMPLETED"
            filename = f"{specification['run_id']}.npz"
            run_path = partial_path / filename
            _write_deterministic_npz(run_path, arrays)
            row["file"] = filename
            row["file_sha256"] = sha256_file(run_path)
            row["size_bytes"] = run_path.stat().st_size
            rows.append(row)
            if progress is not None and (index == 1 or index % 5 == 0):
                progress(
                    f"generated {index}/{len(specifications)}: "
                    f"{specification['run_id']}"
                )

        manifest = {
            "schema_version": 1,
            "dataset_id": REDESIGNED_DATASET_ID,
            "created_at": str(generation["generation_start"]),
            "generation_source_commit": str(generation["source_commit"]),
            "redesign_config_path": str(generation["redesign_config_path"]),
            "redesign_config_sha256": str(generation["redesign_config_sha256"]),
            "redesign_hashes": matrix_audit["component_hashes"],
            "redesign_sha256": matrix_audit["redesign_sha256"],
            "execution_config_path": str(execution_config_path.relative_to(root)),
            "execution_config_sha256": config_sha,
            "scenario_signature_sha256": matrix_audit["scenario_signature_sha256"],
            "split_sha256": matrix_audit["split_sha256"],
            "matrix_audit": matrix_audit,
            "policy_sha256": str(generation["policy_sha256"]),
            "simulator_config_sha256": str(generation["simulator_config_sha256"]),
            "model_blind": True,
            "model_inference_runs": 0,
            "old_holdout_payload_reads": 0,
            "attempted_run_count": attempted,
            "run_count": len(rows),
            "valid_count": sum(bool(row["valid"]) for row in rows),
            "invalid_count": sum(not bool(row["valid"]) for row in rows),
            "split_counts": dict(Counter(str(row["split"]) for row in rows)),
            "adaptive_backfill_count": 0,
            "replacement_run_count": 0,
            "generation_order": [str(row["run_id"]) for row in rows],
            "model_output_fields": [],
            "runs": rows,
        }
        manifest_path = partial_path / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        (partial_path / "manifest.sha256").write_text(
            f"{manifest_sha}  manifest.json\n", encoding="utf-8"
        )
        audit = audit_sand_benign_redesigned_manifest(manifest, design, matrix_audit)
        audit_path = partial_path / "physical_audit.json"
        _write_json(audit_path, audit)
        seal = {
            "schema_version": 1,
            "dataset_id": REDESIGNED_DATASET_ID,
            "split": "REDESIGNED_CONFIRMATION",
            "status": "SEALED_FOR_REDESIGNED_CONFIRMATION",
            "generated": True,
            "objective_integrity_checked": True,
            "model_inference": False,
            "normalized_80d_analysis": False,
            "observability_analysis": False,
            "visualization": False,
            "hypothesis_selection": False,
            "allowed_this_milestone": [
                "file_and_hash_integrity",
                "planned_signature_audit",
                "objective_physical_labels_and_generation_gates",
            ],
        }
        seal_path = partial_path / "confirmation_seal.json"
        _write_json(seal_path, seal)
        npz_hashes = {str(row["file"]): str(row["file_sha256"]) for row in rows}
        physical_outcomes = [
            {
                "run_id": row["run_id"],
                "valid": row["valid"],
                "outcome": row["objective_physical_outcome"],
                "actual_benign_severity": row["actual_benign_severity"],
                "invalid_reason": row["invalid_reason"],
            }
            for row in rows
        ]
        physical_signatures = [
            {"run_id": row["run_id"], **row["physical_signature"]} for row in rows
        ]
        freeze = {
            "schema_version": 1,
            "dataset_id": REDESIGNED_DATASET_ID,
            "generation_source_commit": str(generation["source_commit"]),
            "redesign_config_sha256": str(generation["redesign_config_sha256"]),
            "generation_config_sha256": config_sha,
            "run_count": len(rows),
            "valid_count": manifest["valid_count"],
            "invalid_count": manifest["invalid_count"],
            "REDESIGNED_STUDY_MANIFEST_SHA": manifest_sha,
            "REDESIGNED_STUDY_DISCOVERY_SPLIT_SHA": matrix_audit["split_sha256"][
                "REDESIGNED_DISCOVERY"
            ],
            "REDESIGNED_STUDY_CONFIRMATION_SPLIT_SHA": matrix_audit["split_sha256"][
                "REDESIGNED_CONFIRMATION"
            ],
            "REDESIGNED_STUDY_SCENARIO_SIGNATURE_SHA": matrix_audit[
                "scenario_signature_sha256"
            ],
            "REDESIGNED_STUDY_PHYSICAL_SIGNATURE_SHA": canonical_sha256(
                physical_signatures
            ),
            "REDESIGNED_STUDY_NPZ_AGGREGATE_SHA": canonical_sha256(npz_hashes),
            "REDESIGNED_STUDY_PHYSICAL_OUTCOME_SHA": canonical_sha256(
                physical_outcomes
            ),
            "REDESIGNED_STUDY_GENERATION_GATE_RESULT_SHA": canonical_sha256(
                audit["generation_gates"]
            ),
            "REDESIGNED_STUDY_PHYSICAL_AUDIT_SHA": sha256_file(audit_path),
            "pre_simulation_freeze_sha256": sha256_file(
                partial_path / "pre_simulation_freeze.json"
            ),
            "confirmation_seal_sha256": sha256_file(seal_path),
            "confirmation_status": "SEALED_FOR_REDESIGNED_CONFIRMATION",
            "generation_verdict": audit["generation_verdict"],
            "model_inference_runs": 0,
            "old_holdout_payload_reads": 0,
            "adaptive_backfill_count": 0,
        }
        freeze["REDESIGNED_STUDY_DATASET_FREEZE_SHA"] = canonical_sha256(freeze)
        freeze_path = partial_path / "dataset_freeze.json"
        _write_json(freeze_path, freeze)
        dataset_freeze_file_sha = sha256_file(freeze_path)
        (partial_path / "dataset_freeze.sha256").write_text(
            f"{dataset_freeze_file_sha}  dataset_freeze.json\n",
            encoding="utf-8",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.rename(output_path)
    except Exception:
        shutil.rmtree(partial_path, ignore_errors=True)
        raise

    summary = {
        "dataset_id": REDESIGNED_DATASET_ID,
        "output_path": str(output_path),
        "planned_runs": len(specifications),
        "attempted_runs": attempted,
        "completed_runs": len(rows),
        "valid_runs": sum(bool(row["valid"]) for row in rows),
        "invalid_runs": sum(not bool(row["valid"]) for row in rows),
        "discovery_runs": matrix_audit["split_counts"]["REDESIGNED_DISCOVERY"],
        "confirmation_runs": matrix_audit["split_counts"]["REDESIGNED_CONFIRMATION"],
        "adaptive_backfill_count": 0,
        "replacement_run_count": 0,
        "npz_bytes": sum(int(row["size_bytes"]) for row in rows),
        "file_count": len(list(output_path.iterdir())) + 1,
        "generation_seconds": round(time.monotonic() - started, 3),
        "manifest_sha256": manifest_sha,
        "dataset_freeze_file_sha256": dataset_freeze_file_sha,
        "dataset_freeze_semantic_sha256": freeze["REDESIGNED_STUDY_DATASET_FREEZE_SHA"],
        "generation_verdict": audit["generation_verdict"],
        "confirmation_status": "SEALED_FOR_REDESIGNED_CONFIRMATION",
        "model_inference_runs": 0,
        "old_holdout_payload_reads": 0,
    }
    _write_json(output_path / "generation_summary.json", summary)
    return output_path, summary
