"""Frozen Sand-benign study expansion, generation, and physical auditing."""

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
    _reference_signatures,
    _signature,
    _write_deterministic_npz,
    _write_json,
    annotate_model_v2_result,
    canonical_sha256,
    sha256_file,
)
from fastreflex.simulation.g1 import SimulationConfig, SimulationResult, run_simulation
from fastreflex.simulation.stability import PHASE_NAMES


SAND_STUDY_DESIGN_ID = "SAND_BENIGN_GENERALIZATION_STUDY_DESIGN"
SAND_STUDY_GENERATION_ID = "SAND_BENIGN_GENERALIZATION_STUDY_GENERATION"
SAND_STUDY_DATASET_ID = "sand_benign_generalization_study_20260902"
SAND_STUDY_SPLITS = ("STUDY_DISCOVERY", "STUDY_CONFIRMATION")


def _sand_study_intent_match(row: Mapping[str, Any]) -> bool:
    family = str(row["scenario_family"])
    subtype = str(row["actual_subtype"])
    i1 = row["i1_summary"]["first_sample"] is not None
    if family in ("broad_sand_benign", "near_hazard_sand_benign"):
        return subtype == "NONE" and not i1
    if family == "ordinary_support_control":
        expected_side = f"{row['designed_side_topology']}_ONLY"
        return (
            subtype == "SUPPORT"
            and row["support_event_summary"]["side"] == expected_side
        )
    if family == "delayed_support_control":
        first_contact = row["target_contact_summary"]["first_sample"]
        first_i1 = row["i1_summary"]["first_sample"]
        first_support = row["support_event_summary"]["first_sample"]
        clean_touchdowns = row["target_contact_summary"][
            "clean_touchdowns_before_i1"
        ]
        return (
            first_contact is not None
            and first_i1 is not None
            and first_support is not None
            and first_contact < first_i1 <= first_support
            and clean_touchdowns >= 2
            and subtype == "SUPPORT"
            and row["support_event_summary"]["side"] == "LEFT_ONLY"
        )
    raise ValueError(f"unknown Sand study family: {family}")


def _sand_study_component_hashes(
    design: Mapping[str, Any],
) -> dict[str, str]:
    components = {
        "STUDY_PARAMETER_DOMAIN_SHA": "parameter_domain",
        "STUDY_SCENARIO_MATRIX_SHA": "scenario_matrix",
        "STUDY_SPLIT_PLAN_SHA": "split_plan",
        "STUDY_PHYSICAL_LABEL_CONTRACT_SHA": "physical_label_contract",
        "STUDY_DIVERSITY_METRICS_SHA": "diversity_metrics",
        "STUDY_OBSERVABILITY_METRICS_SHA": "observability_metrics",
        "STUDY_DECISION_RULE_SHA": "decision_rules",
    }
    hashes = {
        name: canonical_sha256(design[key]) for name, key in components.items()
    }
    hashes["SAND_BENIGN_GENERALIZATION_STUDY_DESIGN_SHA"] = canonical_sha256(
        {
            "experiment_id": design["experiment"]["id"],
            "dataset_id": design["dataset_plan"]["dataset_id"],
            "counts": design["scenario_matrix"]["counts"],
            "component_hashes": hashes,
        }
    )
    return hashes


def expand_sand_benign_study_design(
    design: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand the exact Sand-study templates without consulting outcomes."""
    if design["experiment"]["id"] != SAND_STUDY_DESIGN_ID:
        raise ValueError("unsupported Sand study design document")
    matrix = design["scenario_matrix"]
    group_codes = {
        "broad_sand_benign": "bb",
        "near_hazard_sand_benign": "nb",
        "ordinary_support_control": "os",
        "delayed_support_control": "ds",
    }
    split_codes = {"STUDY_DISCOVERY": "d", "STUDY_CONFIRMATION": "c"}
    source_codes = {"concrete": "c", "marble": "m"}
    specifications: list[dict[str, Any]] = []
    for split in SAND_STUDY_SPLITS:
        for group_name, group in matrix["groups"].items():
            if "sand_benign" in group_name:
                fixed = dict(matrix["fixed_sand_mechanics"])
            else:
                fixed = dict(group["fixed_mechanics"])
            for template in group["templates"][split]:
                for source in group["sources"]:
                    for speed_value in group["speeds_mps"]:
                        speed = float(speed_value)
                        mechanics = {**fixed, **dict(template)}
                        side = str(
                            template.get(
                                "designed_side",
                                str(mechanics["sink_pattern"])
                                .removeprefix("transition_")
                                .upper(),
                            )
                        )
                        speed_code = f"{round(speed * 100):03d}"
                        run_id = "_".join(
                            (
                                "sbgs",
                                split_codes[split],
                                group_codes[group_name],
                                source_codes[str(source)],
                                speed_code,
                                str(template["id"]),
                            )
                        )
                        specifications.append(
                            {
                                "run_id": run_id,
                                "split": split,
                                "scenario_family": group_name,
                                "group": group_name,
                                "cell_id": str(template["id"]),
                                "source_terrain": str(source),
                                "target_terrain": str(mechanics["target_terrain"]),
                                "speed_mps": speed,
                                "nominal_speed_mps": speed,
                                "designed_role": str(group["role"]),
                                "designed_event_type": (
                                    "SUPPORT"
                                    if "support_control" in group_name
                                    else "CONTROL"
                                ),
                                "designed_side_topology": side,
                                "patch_start_x_m": float(
                                    mechanics["patch_start_x_m"]
                                ),
                                "patch_width_m": float(
                                    mechanics["patch_width_m"]
                                ),
                                "slip_pattern": str(mechanics["slip_pattern"]),
                                "sink_pattern": str(mechanics["sink_pattern"]),
                                "sink_severity": str(mechanics["sink_severity"]),
                                "support_pattern": str(
                                    mechanics["support_pattern"]
                                ),
                                "start_stratum": template.get("start_stratum"),
                                "width_stratum": template.get("width_stratum"),
                                "phase_assignment": template.get("phase_slot"),
                                "realization_cohort": template.get(
                                    "realization_cohort"
                                ),
                                "severity_intent": template.get(
                                    "severity_intent"
                                ),
                                "support_control_subtype": (
                                    group_name
                                    if "support_control" in group_name
                                    else None
                                ),
                            }
                        )
    return specifications


def _sand_study_cross_split_near_duplicates(
    specifications: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> list[tuple[str, str]]:
    discovery = [
        row for row in specifications if row["split"] == "STUDY_DISCOVERY"
    ]
    confirmation = [
        row for row in specifications if row["split"] == "STUDY_CONFIRMATION"
    ]
    comparable = (
        "source_terrain",
        "target_terrain",
        "speed_mps",
        "slip_pattern",
        "sink_pattern",
        "sink_severity",
        "support_pattern",
    )
    start_threshold = float(policy["patch_start_difference_m_exclusive"])
    width_threshold = float(policy["patch_width_difference_m_exclusive"])
    return [
        (str(left["run_id"]), str(right["run_id"]))
        for left in discovery
        for right in confirmation
        if all(left[field] == right[field] for field in comparable)
        and abs(float(left["patch_start_x_m"]) - float(right["patch_start_x_m"]))
        < start_threshold
        and abs(float(left["patch_width_m"]) - float(right["patch_width_m"]))
        < width_threshold
    ]


def validate_sand_benign_study_design(
    root: Path,
    design: Mapping[str, Any],
    specifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed before the first Sand-study simulation."""
    computed_hashes = _sand_study_component_hashes(design)
    if computed_hashes != dict(design["design_hashes"]):
        raise RuntimeError("frozen Sand study design hashes changed")
    matrix = design["scenario_matrix"]
    expected = int(matrix["counts"]["total"])
    ids = [str(row["run_id"]) for row in specifications]
    signatures = [_signature(row) for row in specifications]
    if len(specifications) != expected or len(set(ids)) != expected:
        raise ValueError("Sand study must expand to 176 unique run IDs")
    if len(set(signatures)) != expected:
        raise ValueError("Sand study contains duplicate scenario signatures")
    split_counts = Counter(str(row["split"]) for row in specifications)
    if split_counts != Counter(
        {"STUDY_DISCOVERY": 88, "STUDY_CONFIRMATION": 88}
    ):
        raise ValueError("Sand study split counts changed")
    group_counts = Counter(str(row["group"]) for row in specifications)
    expected_groups = {
        key: int(matrix["counts"][key])
        for key in (
            "broad_sand_benign",
            "near_hazard_sand_benign",
            "ordinary_support_control",
            "delayed_support_control",
        )
    }
    if dict(group_counts) != expected_groups:
        raise ValueError("Sand study group counts changed")
    sand = [row for row in specifications if "sand_benign" in row["group"]]
    source_speed_counts = Counter(
        (str(row["source_terrain"]), float(row["speed_mps"])) for row in sand
    )
    if len(sand) != 144 or set(source_speed_counts.values()) != {24}:
        raise ValueError("Sand study source-speed balance changed")
    if any(
        float(row["patch_start_x_m"]) == 0.362
        or float(row["patch_width_m"]) == 0.735
        for row in specifications
    ):
        raise ValueError("consumed HOLDOUT coordinates entered the study")
    references, provenance = _reference_signatures(
        root, design["protected_objects"]["datasets"]
    )
    overlap_by_reference = {
        path: len(set(signatures) & values)
        for path, values in references.items()
    }
    if any(overlap_by_reference.values()):
        raise ValueError("Sand study overlaps historical scenario signatures")
    split_signatures = {
        split: {
            _signature(row) for row in specifications if row["split"] == split
        }
        for split in SAND_STUDY_SPLITS
    }
    split_overlap = len(
        split_signatures["STUDY_DISCOVERY"]
        & split_signatures["STUDY_CONFIRMATION"]
    )
    near = _sand_study_cross_split_near_duplicates(
        specifications, design["duplicate_and_diversity_policy"]["near_parameter_duplicate"]
    )
    if split_overlap or near:
        raise ValueError("Sand study split separation changed")
    return {
        "passed": True,
        "runs": len(specifications),
        "split_counts": dict(split_counts),
        "group_counts": dict(group_counts),
        "sand_source_speed_counts": {
            f"{source}/{speed:.2f}": count
            for (source, speed), count in sorted(source_speed_counts.items())
        },
        "unique_run_ids": len(set(ids)),
        "unique_scenario_signatures": len(set(signatures)),
        "exact_duplicate_signatures": len(signatures) - len(set(signatures)),
        "discovery_confirmation_exact_overlap": split_overlap,
        "cross_split_parameter_near_duplicates": len(near),
        "historical_overlap_by_reference": overlap_by_reference,
        "historical_references": provenance,
        "expanded_matrix_sha256": canonical_sha256(list(specifications)),
        "scenario_signature_sha256": canonical_sha256(
            [list(signature) for signature in signatures]
        ),
        "split_sha256": {
            split: canonical_sha256(
                [
                    row["run_id"]
                    for row in specifications
                    if row["split"] == split
                ]
            )
            for split in SAND_STUDY_SPLITS
        },
        "design_hashes": computed_hashes,
    }


def _bool_side(values: np.ndarray) -> str:
    active = np.asarray(values, dtype=bool).reshape(2)
    if bool(active[0]) and bool(active[1]):
        return "BILATERAL"
    if bool(active[0]):
        return "LEFT"
    if bool(active[1]):
        return "RIGHT"
    return "NONE"


def _sand_study_result_summary(
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    result: SimulationResult,
) -> None:
    """Attach model-independent study labels and diversity summaries."""
    target = np.asarray(arrays["target_terrain_contact"], dtype=bool)
    target_any = np.any(target, axis=1)
    target_samples = np.flatnonzero(target_any)
    censor = int(arrays["censor_sample"])
    last_target = None if not target_samples.size else int(target_samples[-1])
    followup = (
        0 if last_target is None else max(0, censor - (last_target + 1))
    )
    if row["valid"] and row["fall_censor_summary"]["first_fall_sample"] is not None:
        row["valid"] = False
        row["invalid_reason"] = "fall_or_censor_ambiguity"
    elif row["valid"] and followup < 1000:
        row["valid"] = False
        row["invalid_reason"] = "insufficient_post_target_observation"
    first = row["target_contact_summary"]["first_sample"]
    if first is None:
        leading = "NONE"
        loaded_side = "NONE"
        entry_phase = "NO_SUPPORT"
    else:
        leading = _bool_side(target[int(first)])
        loaded_side = _bool_side(arrays["loaded_contact"][int(first)])
        entry_phase = PHASE_NAMES[int(arrays["gait_phase"][int(first)])]
    target_side = _bool_side(np.any(target[:censor], axis=0))
    interval = target_any[:censor]
    spread = np.asarray(arrays["support_surface_spread_m"], dtype=np.float64)
    displacement = np.asarray(
        arrays["support_surface_max_displacement_m"], dtype=np.float64
    )
    if np.any(interval):
        peak_spread = float(np.max(spread[:censor][target[:censor]]))
        peak_displacement = float(
            np.max(displacement[:censor][target[:censor]])
        )
    else:
        peak_spread = 0.0
        peak_displacement = 0.0
    fsr = np.asarray(arrays["foot_fsr8"], dtype=np.float64)
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
    double_loaded = np.all(arrays["loaded_contact"][:censor], axis=1)
    reference_values = total_load[:censor][
        double_loaded & (total_load[:censor] > 0)
    ]
    if not reference_values.size:
        reference_values = total_load[:censor][total_load[:censor] > 0]
    body_weight_proxy_n = (
        1.0 if not reference_values.size else float(np.median(reference_values))
    )
    derivative_n_per_s = (
        np.abs(np.diff(total_load[:censor], prepend=total_load[0])) * 1000.0
    )
    peak_load_derivative = (
        0.0
        if not np.any(interval)
        else float(np.max(derivative_n_per_s[interval]))
    )
    normalized_load_derivative = peak_load_derivative / max(
        body_weight_proxy_n, 1.0e-9
    )
    imu = np.asarray(arrays["pelvis_imu6"], dtype=np.float64)
    selected_imu = imu[:censor][interval]
    if selected_imu.size:
        accel_norm = np.linalg.norm(selected_imu[:, :3], axis=1)
        gyro_norm = np.linalg.norm(selected_imu[:, 3:], axis=1)
        accel_rms = float(np.sqrt(np.mean(np.square(accel_norm))))
        gyro_rms = float(np.sqrt(np.mean(np.square(gyro_norm))))
        accel_range = [float(np.min(accel_norm)), float(np.max(accel_norm))]
        gyro_range = [float(np.min(gyro_norm)), float(np.max(gyro_norm))]
    else:
        accel_rms = gyro_rms = 0.0
        accel_range = gyro_range = [0.0, 0.0]
    subtype = str(row["actual_subtype"])
    i1 = row["i1_summary"]["first_sample"] is not None
    if not row["valid"]:
        outcome = "INVALID"
    elif subtype == "SLIP_AND_SUPPORT":
        outcome = "DUAL_HAZARD"
    elif subtype == "SLIP":
        outcome = "SLIP"
    elif subtype == "SUPPORT":
        outcome = "SUPPORT"
    elif i1:
        outcome = "I1_ONLY_BENIGN"
    else:
        outcome = "STRICT_BENIGN"
    severity = None
    if "sand_benign" in str(row["group"]) and outcome == "STRICT_BENIGN":
        if 0.0 <= peak_displacement < 0.030:
            severity = "LOW"
        elif peak_displacement < 0.0525:
            severity = "MEDIUM"
        elif peak_displacement <= 0.070:
            severity = "NEAR_HAZARD"
        else:
            severity = "OUTSIDE_FROZEN_BENIGN_SEVERITY_RANGE"
    physical_signature = {
        "first_target_contact_ms": first,
        "target_contact_duration_ms": int(np.count_nonzero(interval)),
        "leading_foot": leading,
        "entry_gait_phase": entry_phase,
        "peak_transition_displacement_m": peak_displacement,
        "peak_support_spread_m": peak_spread,
        "normalized_load_redistribution": redistribution,
        "normalized_peak_load_derivative": normalized_load_derivative,
        "pelvis_accel_rms": accel_rms,
        "pelvis_gyro_rms": gyro_rms,
    }
    scenario_signature = list(row.pop("physical_signature"))
    row.pop("physical_signature_sha256")
    row["scenario_signature"] = scenario_signature
    row["scenario_signature_sha256"] = canonical_sha256(scenario_signature)
    row["objective_physical_outcome"] = outcome
    row["actual_benign_severity"] = severity
    first_i1 = row["i1_summary"]["first_sample"]
    touchdown_limit = censor if first_i1 is None else int(first_i1)
    clean_touchdowns = int(
        np.count_nonzero(
            np.any(
                np.asarray(
                    arrays["target_terrain_touchdown"][:touchdown_limit],
                    dtype=bool,
                ),
                axis=1,
            )
        )
    )
    row["target_contact_summary"].update(
        {
            "last_sample": last_target,
            "duration_ms": int(np.count_nonzero(interval)),
            "post_target_observation_ms": followup,
            "leading_foot": leading,
            "target_contact_side": target_side,
            "loaded_side_at_entry": loaded_side,
            "entry_gait_phase": entry_phase,
            "clean_touchdowns_before_i1": clean_touchdowns,
            "phase_assignment_realized_match": "NOT_COMPARABLE_INDIRECT_SLOT",
        }
    )
    row["physical_diversity_summary"] = {
        "peak_balanced_displacement_m": peak_displacement,
        "peak_support_spread_m": peak_spread,
        "normalized_load_redistribution": redistribution,
        "peak_load_derivative_n_per_s": peak_load_derivative,
        "normalized_peak_load_derivative": normalized_load_derivative,
        "body_weight_proxy_n": body_weight_proxy_n,
        "contact_sequence_count": int(
            row["target_contact_summary"]["episode_count"]
        ),
        "pelvis_accel_norm_rms": accel_rms,
        "pelvis_accel_norm_range": accel_range,
        "pelvis_gyro_norm_rms": gyro_rms,
        "pelvis_gyro_norm_range": gyro_range,
    }
    row["physical_signature"] = physical_signature
    row["physical_signature_sha256"] = canonical_sha256(physical_signature)
    row["intent_match"] = bool(row["valid"] and _sand_study_intent_match(row))
    row["intent_mismatch"] = bool(row["valid"] and not row["intent_match"])
    row["model_outputs_present"] = False
    row["result_metadata_minimum_pelvis_height_m"] = float(
        result.metadata["minimum_pelvis_height_m"]
    )


def _study_annotation_specification(
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Route through frozen physical annotation without changing its source."""
    routed = dict(specification)
    group = str(specification["group"])
    if "sand_benign" in group:
        routed["scenario_family"] = "STAGED_SAND_BENIGN_CONTROL"
    elif group == "ordinary_support_control":
        routed["scenario_family"] = (
            "LEFT_SAND_SUPPORT_SPEED_MATRIX"
            if specification["designed_side_topology"] == "LEFT"
            else "RIGHT_SAND_SUPPORT_SPEED_MATRIX"
        )
    elif group == "delayed_support_control":
        routed["scenario_family"] = "DELAYED_SAND_SUPPORT_ONSET"
    else:
        raise ValueError(f"unsupported Sand study group: {group}")
    return routed


def _study_scaled_physical_distance(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
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
                (
                    (float(left[key]) - float(right[key])) / scale
                )
                ** 2
                for key, scale in scales.items()
            )
        )
    )


def _study_physical_near_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    valid = [row for row in rows if row["valid"]]
    result: list[dict[str, object]] = []
    for index, left in enumerate(valid):
        for right in valid[index + 1 :]:
            same_domain = all(
                left[key] == right[key]
                for key in (
                    "source_terrain",
                    "speed_mps",
                    "severity_intent",
                    "sink_pattern",
                )
            )
            left_physical = left["physical_signature"]
            right_physical = right["physical_signature"]
            same_category = all(
                left_physical[key] == right_physical[key]
                for key in ("leading_foot", "entry_gait_phase")
            )
            if not same_domain or not same_category:
                continue
            distance = _study_scaled_physical_distance(
                left_physical, right_physical
            )
            if distance <= 0.10:
                result.append(
                    {
                        "left_run_id": left["run_id"],
                        "right_run_id": right["run_id"],
                        "cross_split": left["split"] != right["split"],
                        "distance": distance,
                    }
                )
    return result


def _study_outcome_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["objective_physical_outcome"]) for row in rows)
    return {
        name: int(counts.get(name, 0))
        for name in (
            "STRICT_BENIGN",
            "I1_ONLY_BENIGN",
            "SUPPORT",
            "SLIP",
            "DUAL_HAZARD",
            "INVALID",
        )
    }


def _study_numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(finite.size),
        "minimum": None if not finite.size else float(np.min(finite)),
        "median": None if not finite.size else float(np.median(finite)),
        "maximum": None if not finite.size else float(np.max(finite)),
        "span": None if not finite.size else float(np.ptp(finite)),
    }


def _study_realization_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    fields = {
        "target_contact_timing_ms": "first_target_contact_ms",
        "contact_duration_ms": "target_contact_duration_ms",
        "balanced_displacement_m": "peak_transition_displacement_m",
        "support_spread_m": "peak_support_spread_m",
        "normalized_load_redistribution": "normalized_load_redistribution",
        "normalized_load_transition": "normalized_peak_load_derivative",
        "pelvis_accel_rms": "pelvis_accel_rms",
        "pelvis_gyro_rms": "pelvis_gyro_rms",
    }
    return {
        name: _study_numeric_summary(
            [float(row["physical_signature"][key]) for row in rows]
        )
        for name, key in fields.items()
    }


def audit_sand_benign_study_manifest(
    manifest: Mapping[str, Any],
    design: Mapping[str, Any],
    matrix_audit: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit only physical and diversity evidence; never run a model."""
    rows = list(manifest["runs"])
    sand = [row for row in rows if "sand_benign" in row["group"]]
    execution_table = {
        group: {
            "planned": int(design["scenario_matrix"]["counts"][group]),
            "executed": sum(row["group"] == group for row in rows),
            "valid": sum(row["group"] == group and row["valid"] for row in rows),
            "invalid": sum(
                row["group"] == group and not row["valid"] for row in rows
            ),
        }
        for group in (
            "broad_sand_benign",
            "near_hazard_sand_benign",
            "ordinary_support_control",
            "delayed_support_control",
        )
    }
    for split in SAND_STUDY_SPLITS:
        selected = [row for row in rows if row["split"] == split]
        execution_table[split] = {
            "planned": 88,
            "executed": len(selected),
            "valid": sum(bool(row["valid"]) for row in selected),
            "invalid": sum(not bool(row["valid"]) for row in selected),
        }
    execution_table["total"] = {
        "planned": 176,
        "executed": len(rows),
        "valid": sum(bool(row["valid"]) for row in rows),
        "invalid": sum(not bool(row["valid"]) for row in rows),
    }
    split_outcomes = {
        split: _study_outcome_counts(
            [row for row in rows if row["split"] == split]
        )
        for split in SAND_STUDY_SPLITS
    }
    source_speed: dict[str, dict[str, dict[str, int]]] = {}
    for split in SAND_STUDY_SPLITS:
        source_speed[split] = {}
        for source in ("concrete", "marble"):
            for speed in (0.20, 0.25, 0.30):
                selected = [
                    row
                    for row in sand
                    if row["split"] == split
                    and row["source_terrain"] == source
                    and float(row["speed_mps"]) == speed
                ]
                source_speed[split][f"{source}/{speed:.2f}"] = {
                    "planned": len(selected),
                    "valid": sum(bool(row["valid"]) for row in selected),
                    **_study_outcome_counts(selected),
                }
    severity: dict[str, dict[str, dict[str, int]]] = {}
    for split in SAND_STUDY_SPLITS:
        severity[split] = {}
        for intent in ("LOW", "MEDIUM", "NEAR_HAZARD"):
            selected = [
                row
                for row in sand
                if row["split"] == split and row["severity_intent"] == intent
            ]
            realized = Counter(
                str(row["actual_benign_severity"])
                for row in selected
                if row["actual_benign_severity"] is not None
            )
            severity[split][intent] = {
                "planned": len(selected),
                "LOW": int(realized.get("LOW", 0)),
                "MEDIUM": int(realized.get("MEDIUM", 0)),
                "NEAR_HAZARD": int(realized.get("NEAR_HAZARD", 0)),
                "outside_range": int(
                    realized.get("OUTSIDE_FROZEN_BENIGN_SEVERITY_RANGE", 0)
                ),
                "hazard_or_i1_only": sum(
                    row["objective_physical_outcome"]
                    in ("I1_ONLY_BENIGN", "SUPPORT", "SLIP", "DUAL_HAZARD")
                    for row in selected
                ),
                "invalid": sum(not row["valid"] for row in selected),
            }
    geometry: dict[str, dict[str, dict[str, int]]] = {}
    for split in SAND_STUDY_SPLITS:
        geometry[split] = {}
        for dimension in ("start_stratum", "width_stratum"):
            geometry[split][dimension] = {}
            levels = (
                ("EARLY", "MID", "LATE")
                if dimension == "start_stratum"
                else ("NARROW", "MEDIUM", "WIDE")
            )
            for level in levels:
                selected = [
                    row
                    for row in sand
                    if row["split"] == split and row[dimension] == level
                ]
                geometry[split][dimension][level] = {
                    "planned": len(selected),
                    "executed": len(selected),
                    "valid": sum(bool(row["valid"]) for row in selected),
                    "strict_benign": sum(
                        row["objective_physical_outcome"] == "STRICT_BENIGN"
                        for row in selected
                    ),
                    "physical_realization": _study_realization_summary(
                        [row for row in selected if row["valid"]]
                    ),
                }
    discovery_sand = [
        row for row in sand if row["split"] == "STUDY_DISCOVERY"
    ]
    phase_crosstab = {
        slot: dict(
            Counter(
                str(row["target_contact_summary"]["entry_gait_phase"])
                for row in discovery_sand
                if row["phase_assignment"] == slot
            )
        )
        for slot in ("PHASE_A", "PHASE_B", "PHASE_C", "PHASE_D")
    }
    topology_contact = {
        topology: {
            "planned": sum(row["sink_pattern"] == topology for row in discovery_sand),
            "leading_foot": dict(
                Counter(
                    str(row["target_contact_summary"]["leading_foot"])
                    for row in discovery_sand
                    if row["sink_pattern"] == topology
                )
            ),
            "loaded_side_at_entry": dict(
                Counter(
                    str(row["target_contact_summary"]["loaded_side_at_entry"])
                    for row in discovery_sand
                    if row["sink_pattern"] == topology
                )
            ),
            "target_contact_side": dict(
                Counter(
                    str(row["target_contact_summary"]["target_contact_side"])
                    for row in discovery_sand
                    if row["sink_pattern"] == topology
                )
            ),
        }
        for topology in ("transition_left", "transition_right")
    }
    physical_hashes = [
        str(row["physical_signature_sha256"]) for row in rows if row["valid"]
    ]
    physical_exact_duplicates = len(physical_hashes) - len(set(physical_hashes))
    near_pairs = _study_physical_near_pairs(rows)
    cross_split_near = sum(bool(row["cross_split"]) for row in near_pairs)
    entry_timing: dict[str, dict[str, float | int | None]] = {}
    for split in SAND_STUDY_SPLITS:
        for source in ("concrete", "marble"):
            for speed in (0.20, 0.25, 0.30):
                selected = [
                    row
                    for row in sand
                    if row["split"] == split
                    and row["source_terrain"] == source
                    and float(row["speed_mps"]) == speed
                    and row["valid"]
                ]
                values = [
                    int(row["physical_signature"]["first_target_contact_ms"])
                    for row in selected
                ]
                entry_timing[f"{split}/{source}/{speed:.2f}"] = {
                    "count": len(values),
                    "minimum_ms": min(values, default=None),
                    "maximum_ms": max(values, default=None),
                    "span_ms": 0 if not values else max(values) - min(values),
                }
    yield_requirements = design["invalid_run_policy"]
    yield_checks: dict[str, bool] = {}
    for split, key in (
        ("STUDY_DISCOVERY", "discovery_minimum_viable_yield"),
        ("STUDY_CONFIRMATION", "confirmation_minimum_viable_yield"),
    ):
        requirements = yield_requirements[key]
        strict = [
            row
            for row in sand
            if row["split"] == split
            and row["objective_physical_outcome"] == "STRICT_BENIGN"
        ]
        yield_checks[f"{split}/strict_benign_total"] = len(strict) >= int(
            requirements["strict_benign_total_min"]
        )
        for source in ("concrete", "marble"):
            for speed in (0.20, 0.25, 0.30):
                cell = [
                    row
                    for row in strict
                    if row["source_terrain"] == source
                    and float(row["speed_mps"]) == speed
                ]
                prefix = f"{split}/{source}/{speed:.2f}"
                yield_checks[f"{prefix}/strict_benign"] = len(cell) >= int(
                    requirements["strict_benign_per_source_speed_cell_min"]
                )
                for severity_name in ("LOW", "MEDIUM", "NEAR_HAZARD"):
                    yield_checks[f"{prefix}/severity/{severity_name}"] = sum(
                        row["actual_benign_severity"] == severity_name
                        for row in cell
                    ) >= int(
                        requirements[
                            "strict_benign_per_actual_severity_per_source_speed_cell_min"
                        ]
                    )
                for topology in ("transition_left", "transition_right"):
                    yield_checks[f"{prefix}/topology/{topology}"] = sum(
                        row["sink_pattern"] == topology for row in cell
                    ) >= int(
                        requirements[
                            "strict_benign_per_topology_per_source_speed_cell_min"
                        ]
                    )
        ordinary = [
            row
            for row in rows
            if row["split"] == split
            and row["group"] == "ordinary_support_control"
            and row["objective_physical_outcome"] == "SUPPORT"
        ]
        delayed = [
            row
            for row in rows
            if row["split"] == split
            and row["group"] == "delayed_support_control"
            and row["objective_physical_outcome"] == "SUPPORT"
            and row["intent_match"]
        ]
        yield_checks[f"{split}/ordinary_support"] = len(ordinary) >= int(
            requirements["ordinary_support_actual_min"]
        )
        yield_checks[f"{split}/delayed_support"] = len(delayed) >= int(
            requirements["delayed_support_actual_min"]
        )
    physical_policy = design["diversity_metrics"]["physical_coverage"]
    diversity_checks: dict[str, bool] = {
        "physical_signature_fraction": (
            len(set(physical_hashes)) / max(len(physical_hashes), 1)
            >= float(physical_policy["unique_physical_signature_fraction_min"])
        ),
        "global_discovery_entry_phases": len(
            {
                row["target_contact_summary"]["entry_gait_phase"]
                for row in discovery_sand
                if row["valid"]
            }
        )
        >= int(physical_policy["global_actual_entry_phase_categories_min"]),
        "no_cross_split_physical_near_duplicates": cross_split_near == 0,
    }
    for split in SAND_STUDY_SPLITS:
        for source in ("concrete", "marble"):
            for speed in (0.20, 0.25, 0.30):
                selected = [
                    row
                    for row in sand
                    if row["split"] == split
                    and row["source_terrain"] == source
                    and float(row["speed_mps"]) == speed
                    and row["valid"]
                ]
                prefix = f"{split}/{source}/{speed:.2f}"
                diversity_checks[f"{prefix}/entry_phases"] = len(
                    {
                        row["target_contact_summary"]["entry_gait_phase"]
                        for row in selected
                    }
                ) >= int(
                    physical_policy[
                        "actual_entry_phase_categories_per_source_speed_split_min"
                    ]
                )
                diversity_checks[f"{prefix}/leading_feet"] = {
                    row["target_contact_summary"]["leading_foot"]
                    for row in selected
                }.issuperset({"LEFT", "RIGHT"})
                timing = entry_timing[prefix]
                diversity_checks[f"{prefix}/entry_span"] = int(
                    timing["span_ms"]
                ) >= int(
                    physical_policy[
                        "first_target_contact_time_span_ms_per_source_speed_split_min"
                    ]
                )
                strict = [
                    row
                    for row in selected
                    if row["objective_physical_outcome"] == "STRICT_BENIGN"
                ]
                diversity_checks[f"{prefix}/actual_severity"] = len(
                    {
                        row["actual_benign_severity"]
                        for row in strict
                        if row["actual_benign_severity"]
                        in ("LOW", "MEDIUM", "NEAR_HAZARD")
                    }
                ) >= int(
                    physical_policy[
                        "actual_severity_strata_per_source_speed_split_required"
                    ]
                )
    cohort_rows = [row for row in discovery_sand if row["valid"]]
    cohort_summary: dict[str, dict[str, object]] = {}
    for cohort in (2026090201, 2026090202):
        selected = [row for row in cohort_rows if row["realization_cohort"] == cohort]
        cohort_summary[str(cohort)] = _study_realization_summary(selected)
        cohort_summary[str(cohort)]["contact_sequence_count"] = (
            _study_numeric_summary(
                [
                    float(row["physical_diversity_summary"]["contact_sequence_count"])
                    for row in selected
                ]
            )
        )
        cohort_summary[str(cohort)]["realized_entry_phases"] = sorted(
            {
                str(row["physical_signature"]["entry_gait_phase"])
                for row in selected
            }
        )
    cohort_one = [
        row for row in cohort_rows if row["realization_cohort"] == 2026090201
    ]
    cohort_two = [
        row for row in cohort_rows if row["realization_cohort"] == 2026090202
    ]
    nearest_cross_cohort = [
        min(
            _study_scaled_physical_distance(
                left["physical_signature"], right["physical_signature"]
            )
            for right in cohort_two
        )
        for left in cohort_one
    ]
    cohort_exact_duplicates = len(
        {
            row["physical_signature_sha256"] for row in cohort_rows
        }
    ) != len(cohort_rows)
    threshold = float(
        execution["generation"]["cohort_audit"][
            "confirmed_median_nearest_distance_min"
        ]
    )
    median_cross = (
        0.0
        if not nearest_cross_cohort
        else float(np.median(nearest_cross_cohort))
    )
    if cohort_exact_duplicates:
        cohort_classification = "REALIZATION_DIVERSITY_COLLAPSED"
    elif median_cross >= threshold and all(
        len(cohort_summary[str(cohort)]["realized_entry_phases"]) >= 2
        for cohort in (2026090201, 2026090202)
    ):
        cohort_classification = "REALIZATION_DIVERSITY_CONFIRMED"
    else:
        cohort_classification = "REALIZATION_DIVERSITY_WEAK"
    diversity_checks["realization_cohort_not_collapsed"] = (
        cohort_classification != "REALIZATION_DIVERSITY_COLLAPSED"
    )
    discovery_strict = [
        row
        for row in discovery_sand
        if row["objective_physical_outcome"] == "STRICT_BENIGN"
    ]
    physical_benign_distributions = _study_realization_summary(discovery_strict)
    physical_benign_distributions["contact_sequence_count"] = (
        _study_numeric_summary(
            [
                float(row["physical_diversity_summary"]["contact_sequence_count"])
                for row in discovery_strict
            ]
        )
    )
    entry_diversity: dict[str, dict[str, object]] = {}
    for dimension, levels in (
        ("start_stratum", ("EARLY", "MID", "LATE")),
        ("width_stratum", ("NARROW", "MEDIUM", "WIDE")),
        ("phase_assignment", ("PHASE_A", "PHASE_B", "PHASE_C", "PHASE_D")),
        ("source_terrain", ("concrete", "marble")),
        ("speed_mps", (0.20, 0.25, 0.30)),
    ):
        entry_diversity[dimension] = {}
        for level in levels:
            selected = [
                row
                for row in discovery_sand
                if row[dimension] == level and row["valid"]
            ]
            entry_diversity[dimension][str(level)] = {
                "planned": sum(
                    row[dimension] == level for row in discovery_sand
                ),
                "valid": len(selected),
                "timing_ms": _study_numeric_summary(
                    [
                        float(row["physical_signature"]["first_target_contact_ms"])
                        for row in selected
                    ]
                ),
            }
    parameter_cells = []
    for row in sand:
        parameter_cells.append(
            {
                "run_id": row["run_id"],
                "split": row["split"],
                "source": row["source_terrain"],
                "speed_mps": row["speed_mps"],
                "start_stratum": row["start_stratum"],
                "width_stratum": row["width_stratum"],
                "phase_slot": row["phase_assignment"],
                "topology": row["sink_pattern"],
                "severity_intent": row["severity_intent"],
                "realization_cohort": row["realization_cohort"],
                "planned": 1,
                "executed": 1,
                "valid": int(bool(row["valid"])),
                "strict_benign": int(
                    row["objective_physical_outcome"] == "STRICT_BENIGN"
                ),
            }
        )
    support_controls: dict[str, object] = {"by_type_source": {}}
    for group in ("ordinary_support_control", "delayed_support_control"):
        support_controls["by_type_source"][group] = {}
        for source in ("concrete", "marble"):
            selected = [
                row
                for row in rows
                if row["group"] == group and row["source_terrain"] == source
            ]
            support_controls["by_type_source"][group][source] = {
                "planned": len(selected),
                "actual_support": sum(
                    row["objective_physical_outcome"] == "SUPPORT"
                    for row in selected
                ),
                "other_physical_outcome": sum(
                    row["valid"]
                    and row["objective_physical_outcome"] != "SUPPORT"
                    for row in selected
                ),
                "invalid": sum(not row["valid"] for row in selected),
            }
    support_rows = [row for row in rows if "support_control" in row["group"]]
    support_controls["side"] = dict(
        Counter(str(row["support_event_summary"]["side"]) for row in support_rows)
    )
    support_controls["speed"] = dict(
        Counter(f"{float(row['speed_mps']):.2f}" for row in support_rows)
    )
    support_controls["i1_present"] = sum(
        row["i1_summary"]["first_sample"] is not None for row in support_rows
    )
    support_controls["established_support"] = sum(
        row["support_event_summary"]["first_sample"] is not None
        for row in support_rows
    )
    support_controls["invalid"] = sum(not row["valid"] for row in support_rows)
    support_controls["fall_or_censor"] = sum(
        row["fall_censor_summary"]["first_fall_sample"] is not None
        for row in support_rows
    )
    integrity_checks = {
        "all_176_executed": len(rows) == 176,
        "unique_scenario_signatures_176": matrix_audit[
            "unique_scenario_signatures"
        ]
        == 176,
        "scenario_exact_duplicates_zero": matrix_audit[
            "exact_duplicate_signatures"
        ]
        == 0,
        "historical_overlap_zero": not any(
            matrix_audit["historical_overlap_by_reference"].values()
        ),
        "split_exact_overlap_zero": matrix_audit[
            "discovery_confirmation_exact_overlap"
        ]
        == 0,
        "no_adaptive_backfill": manifest["adaptive_backfill_count"] == 0,
        "no_model_outputs": all(not row["model_outputs_present"] for row in rows),
    }
    yield_pass = all(yield_checks.values())
    diversity_pass = all(diversity_checks.values())
    integrity_pass = all(integrity_checks.values())
    if not integrity_pass:
        verdict = "SAND_BENIGN_GENERALIZATION_STUDY_GENERATION_INVALID"
    elif not yield_pass:
        verdict = (
            "SAND_BENIGN_GENERALIZATION_STUDY_GENERATION_PHYSICAL_YIELD_INSUFFICIENT"
        )
    elif not diversity_pass:
        verdict = (
            "SAND_BENIGN_GENERALIZATION_STUDY_GENERATION_DIVERSITY_INSUFFICIENT"
        )
    else:
        verdict = "SAND_BENIGN_GENERALIZATION_STUDY_GENERATION_READY"
    return {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "execution_table": execution_table,
        "split_outcomes": split_outcomes,
        "overall_outcomes": _study_outcome_counts(rows),
        "sand_source_speed": source_speed,
        "severity_realization": severity,
        "geometry_realization": geometry,
        "discovery_phase_crosstab": phase_crosstab,
        "discovery_topology_contact": topology_contact,
        "entry_timing": entry_timing,
        "entry_diversity": entry_diversity,
        "parameter_cells": parameter_cells,
        "discovery_strict_benign_physical_distributions": (
            physical_benign_distributions
        ),
        "support_controls": support_controls,
        "physical_signatures": {
            "valid_count": len(physical_hashes),
            "unique_count": len(set(physical_hashes)),
            "exact_duplicates": physical_exact_duplicates,
            "near_duplicate_count": len(near_pairs),
            "cross_split_near_duplicate_count": cross_split_near,
            "near_pairs": near_pairs,
        },
        "realization_cohorts": {
            "summary": cohort_summary,
            "median_nearest_cross_cohort_distance": median_cross,
            "classification": cohort_classification,
        },
        "yield_checks": yield_checks,
        "yield_pass": yield_pass,
        "diversity_checks": diversity_checks,
        "diversity_pass": diversity_pass,
        "integrity_checks": integrity_checks,
        "integrity_pass": integrity_pass,
        "generation_verdict": verdict,
        "model_inference_runs": 0,
        "confirmation_model_or_representation_analysis": False,
    }


def load_sand_benign_study_manifest(dataset_path: Path) -> Mapping[str, Any]:
    """Load study metadata without opening a run payload."""
    manifest_path = dataset_path / "manifest.json"
    expected = (dataset_path / "manifest.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if sha256_file(manifest_path) != expected:
        raise ValueError("Sand study manifest integrity failed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != SAND_STUDY_DATASET_ID:
        raise ValueError("unexpected Sand study dataset identity")
    return manifest


def load_sand_benign_discovery_payload(
    dataset_path: Path, run_id: str
) -> dict[str, np.ndarray]:
    """Open one Discovery payload while refusing sealed Confirmation."""
    manifest = load_sand_benign_study_manifest(dataset_path)
    row = next(
        (item for item in manifest["runs"] if item["run_id"] == run_id), None
    )
    if row is None:
        raise KeyError(f"unknown Sand study run: {run_id}")
    if row["split"] != "STUDY_DISCOVERY":
        raise RuntimeError("STUDY_CONFIRMATION is SEALED_FOR_STUDY_CONFIRMATION")
    path = dataset_path / str(row["file"])
    if sha256_file(path) != row["file_sha256"]:
        raise ValueError(f"Sand study run integrity failed: {run_id}")
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def collect_sand_benign_study_dataset(
    root: Path,
    execution_config_path: Path,
    policy_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Execute the 176-run frozen physical study without model inference."""
    execution = _load_yaml(execution_config_path)
    if execution["experiment"]["id"] != SAND_STUDY_GENERATION_ID:
        raise ValueError("unsupported Sand study generation config")
    generation = execution["generation"]
    design_path = root / str(generation["design_config_path"])
    if sha256_file(design_path) != str(generation["design_config_sha256"]):
        raise RuntimeError("frozen Sand study design file changed")
    design = _load_yaml(design_path)
    specifications = expand_sand_benign_study_design(design)
    matrix_audit = validate_sand_benign_study_design(
        root, design, specifications
    )
    expected_hashes = dict(generation["design_hashes"])
    if matrix_audit["design_hashes"] != expected_hashes:
        raise RuntimeError("Sand study component hashes changed")
    expected_matrix = {
        "expanded_matrix_sha256": generation["expected_expanded_matrix_sha256"],
        "scenario_signature_sha256": generation[
            "expected_scenario_signature_sha256"
        ],
        "split_sha256": dict(generation["expected_split_sha256"]),
    }
    if any(matrix_audit[key] != value for key, value in expected_matrix.items()):
        raise RuntimeError("expanded Sand study matrix changed")
    expected_counts = (
        int(generation["planned_total_runs"]),
        int(generation["planned_discovery_runs"]),
        int(generation["planned_confirmation_runs"]),
    )
    actual_counts = (
        len(specifications),
        matrix_audit["split_counts"]["STUDY_DISCOVERY"],
        matrix_audit["split_counts"]["STUDY_CONFIRMATION"],
    )
    if (
        str(generation["dataset_id"]) != SAND_STUDY_DATASET_ID
        or expected_counts != actual_counts
    ):
        raise RuntimeError("Sand study execution counts changed")
    if sha256_file(policy_path) != str(generation["policy_sha256"]):
        raise RuntimeError("walking policy differs from Sand study freeze")
    simulator_config = root / "configs/simulator/g1.yaml"
    if sha256_file(simulator_config) != str(
        generation["simulator_config_sha256"]
    ):
        raise RuntimeError("simulator config differs from Sand study freeze")
    for artifact in generation["protected_artifacts"]:
        if sha256_file(root / str(artifact["path"])) != str(artifact["sha256"]):
            raise RuntimeError(
                f"protected artifact changed: {artifact['path']}"
            )
    for artifact in generation["implementation_artifacts"]:
        if sha256_file(root / str(artifact["path"])) != str(artifact["sha256"]):
            raise RuntimeError(
                f"generation implementation changed: {artifact['path']}"
            )
    holdout_guard = json.loads(
        (root / str(generation["consumed_holdout_guard_path"])).read_text(
            encoding="utf-8"
        )
    )
    if (
        holdout_guard.get("guard_after") != 1
        or holdout_guard.get("scientific_open_count") != 1
    ):
        raise RuntimeError("consumed Generalization HOLDOUT guard changed")
    output_path = root / str(generation["dataset_path"])
    partial_path = output_path.with_name(f"{output_path.name}_partial")
    if output_path.exists() or partial_path.exists():
        raise RuntimeError(f"Sand study dataset output already exists: {output_path}")
    partial_path.mkdir(parents=True)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    try:
        for index, specification in enumerate(specifications, start=1):
            config = SimulationConfig(
                physics_timestep_s=float(design["dataset_plan"]["physics_timestep_s"]),
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
            )
            result = run_simulation(config, observe_fsr=True, observe_foot_imu=False)
            row, arrays = annotate_model_v2_result(
                _study_annotation_specification(specification), result
            )
            row["scenario_family"] = specification["scenario_family"]
            if result.stability is None:
                raise RuntimeError("Sand study requires exact gait phase diagnostics")
            arrays["gait_phase"] = np.asarray(
                result.stability.gait_phase, dtype=np.int8
            )
            _sand_study_result_summary(row, arrays, result)
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
            "dataset_id": SAND_STUDY_DATASET_ID,
            "created_at": str(generation["generation_start"]),
            "generation_source_commit": str(generation["source_commit"]),
            "design_config_path": str(generation["design_config_path"]),
            "design_config_sha256": str(generation["design_config_sha256"]),
            "design_hashes": matrix_audit["design_hashes"],
            "execution_config_path": str(execution_config_path.relative_to(root)),
            "execution_config_sha256": sha256_file(execution_config_path),
            "expanded_matrix_sha256": matrix_audit["expanded_matrix_sha256"],
            "scenario_signature_sha256": matrix_audit[
                "scenario_signature_sha256"
            ],
            "split_sha256": matrix_audit["split_sha256"],
            "policy_sha256": sha256_file(policy_path),
            "simulator_config_sha256": str(generation["simulator_config_sha256"]),
            "model_blind": True,
            "model_inference_runs": 0,
            "adaptive_backfill_count": 0,
            "replacement_run_count": 0,
            "run_count": len(rows),
            "valid_count": sum(bool(row["valid"]) for row in rows),
            "invalid_count": sum(not bool(row["valid"]) for row in rows),
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "generation_order": [row["run_id"] for row in rows],
            "runtime_model_input_fields": ["timestamp_us", "pelvis_imu6"],
            "model_output_fields": [],
            "runs": rows,
        }
        manifest_path = partial_path / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        (partial_path / "manifest.sha256").write_text(
            f"{manifest_sha}  manifest.json\n", encoding="utf-8"
        )
        audit = audit_sand_benign_study_manifest(
            manifest, design, matrix_audit, execution
        )
        audit["matrix_audit"] = matrix_audit
        audit_path = partial_path / "diversity_audit.json"
        _write_json(audit_path, audit)
        diversity_audit_sha = sha256_file(audit_path)
        seal = {
            "schema_version": 1,
            "dataset_id": SAND_STUDY_DATASET_ID,
            "split": "STUDY_CONFIRMATION",
            "status": "SEALED_FOR_STUDY_CONFIRMATION",
            "generated": True,
            "payload_available_on_disk": True,
            "model_inference": False,
            "normalized_feature_analysis": False,
            "observability_analysis": False,
            "visualization": False,
            "scientific_interpretation": False,
            "allowed_this_milestone": [
                "file_and_hash_integrity",
                "planned_parameter_and_signature_verification",
                "objective_physical_label_and_yield_summary",
            ],
        }
        seal_path = partial_path / "confirmation_seal.json"
        _write_json(seal_path, seal)
        npz_hashes = {
            str(row["file"]): str(row["file_sha256"]) for row in rows
        }
        physical_outcomes = [
            {
                "run_id": row["run_id"],
                "valid": row["valid"],
                "outcome": row["objective_physical_outcome"],
                "actual_benign_severity": row["actual_benign_severity"],
            }
            for row in rows
        ]
        physical_signatures = [
            {"run_id": row["run_id"], **row["physical_signature"]}
            for row in rows
        ]
        freeze = {
            "schema_version": 1,
            "dataset_id": SAND_STUDY_DATASET_ID,
            "generation_source_commit": str(generation["source_commit"]),
            "design_config_sha256": str(generation["design_config_sha256"]),
            "generation_config_sha256": sha256_file(execution_config_path),
            "run_count": len(rows),
            "valid_count": manifest["valid_count"],
            "invalid_count": manifest["invalid_count"],
            "STUDY_MANIFEST_SHA": manifest_sha,
            "STUDY_DISCOVERY_SPLIT_SHA": matrix_audit["split_sha256"][
                "STUDY_DISCOVERY"
            ],
            "STUDY_CONFIRMATION_SPLIT_SHA": matrix_audit["split_sha256"][
                "STUDY_CONFIRMATION"
            ],
            "STUDY_SCENARIO_SIGNATURE_SHA": matrix_audit[
                "scenario_signature_sha256"
            ],
            "STUDY_PHYSICAL_SIGNATURE_SHA": canonical_sha256(
                physical_signatures
            ),
            "STUDY_NPZ_AGGREGATE_SHA": canonical_sha256(npz_hashes),
            "STUDY_PHYSICAL_OUTCOME_SHA": canonical_sha256(physical_outcomes),
            "STUDY_DIVERSITY_AUDIT_SHA": diversity_audit_sha,
            "confirmation_seal_sha256": sha256_file(seal_path),
            "confirmation_status": "SEALED_FOR_STUDY_CONFIRMATION",
            "generation_verdict": audit["generation_verdict"],
            "model_inference_runs": 0,
            "old_holdout_payload_reads": 0,
            "adaptive_backfill_count": 0,
        }
        freeze_path = partial_path / "dataset_freeze.json"
        _write_json(freeze_path, freeze)
        dataset_freeze_sha = sha256_file(freeze_path)
        (partial_path / "dataset_freeze.sha256").write_text(
            f"{dataset_freeze_sha}  dataset_freeze.json\n", encoding="utf-8"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.rename(output_path)
    except Exception:
        if partial_path.exists():
            shutil.rmtree(partial_path)
        raise
    summary = {
        "dataset_id": SAND_STUDY_DATASET_ID,
        "output_path": str(output_path),
        "planned_runs": len(specifications),
        "attempted_runs": len(rows),
        "completed_runs": len(rows),
        "valid_runs": sum(bool(row["valid"]) for row in rows),
        "invalid_runs": sum(not bool(row["valid"]) for row in rows),
        "discovery_runs": matrix_audit["split_counts"]["STUDY_DISCOVERY"],
        "confirmation_runs": matrix_audit["split_counts"][
            "STUDY_CONFIRMATION"
        ],
        "adaptive_backfill_count": 0,
        "npz_bytes": sum(int(row["size_bytes"]) for row in rows),
        "generation_seconds": round(time.monotonic() - started, 3),
        "manifest_sha256": manifest_sha,
        "dataset_freeze_sha256": dataset_freeze_sha,
        "generation_verdict": audit["generation_verdict"],
        "confirmation_status": "SEALED_FOR_STUDY_CONFIRMATION",
        "model_inference_runs": 0,
        "old_holdout_payload_reads": 0,
    }
    _write_json(output_path / "generation_summary.json", summary)
    return output_path, summary
