"""Fresh factor-conditioned Sand data intervention generation."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from collections.abc import Callable, Mapping
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
from fastreflex.dataset.sand_calibration import (
    _annotation_specification,
    _calibration_result_summary,
    _historical_overlap_audit,
    _scenario_signatures_are_near,
)
from fastreflex.simulation.g1 import SimulationConfig, run_simulation


FACTOR_CONDITIONED_INTERVENTION_ID = "SAND_FACTOR_CONDITIONED_DATA_INTERVENTION"
FACTOR_CONDITIONED_DATASET_ID = "sand_factor_conditioned_development_20260903"
FACTOR_CONDITIONED_SPLITS = ("FACTOR_TRAIN", "FACTOR_VALIDATION")
FACTOR_CONDITIONED_REDESIGN_ID = "SAND_FACTOR_CONDITIONED_PHYSICAL_DOMAIN_REDESIGN"


def _factor_conditioned_eligible(row: Mapping[str, Any]) -> bool:
    """Apply the predeclared actual-physics eligibility contract."""
    if not bool(row["valid"]) or not bool(row["intent_match"]):
        return False
    group = str(row["group"])
    outcome = str(row["objective_physical_outcome"])
    severity = row["actual_benign_severity"]
    if group == "sand_benign_mild":
        return outcome == "STRICT_BENIGN" and severity == "LOW"
    if group == "sand_benign_moderate":
        return outcome == "STRICT_BENIGN" and severity == "MEDIUM"
    if group in {"ordinary_support_control", "delayed_support_control"}:
        return outcome == "SUPPORT"
    raise ValueError(f"unsupported factor-conditioned group: {group}")


def factor_conditioned_component_hashes(
    document: Mapping[str, Any],
) -> dict[str, str]:
    """Hash the scientific pieces that must precede fresh simulation."""
    sections = {
        "FACTOR_PARAMETER_DOMAIN_SHA": "parameter_domain",
        "FACTOR_SCENARIO_MATRIX_SHA": "scenario_matrix",
        "FACTOR_SPLIT_PLAN_SHA": "split_plan",
        "FACTOR_PHYSICAL_LABEL_CONTRACT_SHA": "physical_label_contract",
        "FACTOR_GENERATION_GATES_SHA": "generation_gates",
        "FACTOR_TRAINING_PROTOCOL_SHA": "training_protocol",
        "FACTOR_VALIDATION_PROTOCOL_SHA": "validation_protocol",
        "FACTOR_DECISION_RULES_SHA": "decision_rules",
    }
    return {
        name: canonical_sha256(document[section]) for name, section in sections.items()
    }


def expand_factor_conditioned_design(
    document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand the frozen fresh TRAIN/VALIDATION matrix without model evidence."""
    if document["experiment"]["id"] != FACTOR_CONDITIONED_INTERVENTION_ID:
        raise ValueError("unsupported factor-conditioned intervention config")
    matrix = document["scenario_matrix"]
    rows: list[dict[str, Any]] = []
    source_codes = {"concrete": "c", "marble": "m"}
    split_codes = {"FACTOR_TRAIN": "t", "FACTOR_VALIDATION": "v"}
    group_codes = {
        "sand_benign_mild": "sml",
        "sand_benign_moderate": "smd",
        "ordinary_support_control": "osp",
        "delayed_support_control": "dsp",
    }
    for split in FACTOR_CONDITIONED_SPLITS:
        for cell in matrix["source_speed_cells"]:
            source = str(cell["source_terrain"])
            speed = float(cell["speed_mps"])
            left_only = source == "concrete" and speed == 0.25
            profile_keys = {
                "sand_benign_mild": (
                    "sand_mild_left_only" if left_only else "sand_mild_standard"
                ),
                "sand_benign_moderate": (
                    "sand_moderate_left_only" if left_only else "sand_moderate_standard"
                ),
                "ordinary_support_control": "ordinary_support",
                "delayed_support_control": "delayed_support",
            }
            for group, profile_key in profile_keys.items():
                for index, profile in enumerate(
                    matrix["profiles"][split][profile_key], start=1
                ):
                    support_group = group.endswith("support_control")
                    delayed = group == "delayed_support_control"
                    severity = "mild" if group == "sand_benign_mild" else "moderate"
                    support_pattern = (
                        "staged_lateral_deformable"
                        if delayed
                        else "lateral_deformable"
                        if support_group
                        else "balanced_deformable"
                    )
                    designed_side = str(profile.get("designed_side", "LEFT"))
                    run_id = "_".join(
                        (
                            str(matrix["run_id_prefix"]),
                            split_codes[split],
                            group_codes[group],
                            source_codes[source],
                            f"{int(round(speed * 100)):03d}",
                            f"{index:02d}",
                        )
                    )
                    rows.append(
                        {
                            "run_id": run_id,
                            "split": split,
                            "group": group,
                            "scenario_family": group,
                            "factor_manifold_intent": str(
                                profile.get("factor_manifold", "CONTROL")
                            ),
                            "source_terrain": source,
                            "target_terrain": "sand",
                            "speed_mps": speed,
                            "nominal_speed_mps": speed,
                            "designed_role": ("HAZARD" if support_group else "NORMAL"),
                            "designed_event_type": (
                                "SUPPORT" if support_group else "NONE"
                            ),
                            "designed_side": designed_side,
                            "designed_side_topology": (
                                f"{designed_side}_ONLY" if support_group else "NONE"
                            ),
                            "patch_start_x_m": float(profile["patch_start_x_m"]),
                            "patch_width_m": float(profile["patch_width_m"]),
                            "slip_pattern": "uniform",
                            "sink_pattern": (
                                "transition_left"
                                if delayed
                                else str(profile["sink_pattern"])
                            ),
                            "sink_severity": severity,
                            "support_pattern": support_pattern,
                            "severity_intent": (
                                "LOW"
                                if group == "sand_benign_mild"
                                else "BOUNDARY_ADJACENT"
                                if group == "sand_benign_moderate"
                                else "SUPPORT_CONTROL"
                            ),
                            "realization_id": str(profile["id"]),
                        }
                    )
    return rows


def validate_factor_conditioned_design(
    root: Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the fresh matrix and historical exclusions before simulation."""
    rows = expand_factor_conditioned_design(document)
    matrix = document["scenario_matrix"]
    ids = [str(row["run_id"]) for row in rows]
    signatures = [_signature(row) for row in rows]
    if len(rows) > 320 or not rows:
        raise ValueError("factor-conditioned main corpus must contain 1..320 runs")
    if len(set(ids)) != len(ids) or len(set(signatures)) != len(signatures):
        raise ValueError("factor-conditioned matrix has duplicate IDs or signatures")
    split_counts = Counter(str(row["split"]) for row in rows)
    group_counts = Counter((str(row["split"]), str(row["group"])) for row in rows)
    expected = matrix["counts"]
    if len(rows) != int(expected["total"]):
        raise ValueError("factor-conditioned total count changed")
    if len(rows) != int(document["generation_gates"]["planned_run_count"]):
        raise ValueError("factor-conditioned generation budget changed")
    for split in FACTOR_CONDITIONED_SPLITS:
        if split_counts[split] != int(expected[split]["total"]):
            raise ValueError(f"factor-conditioned {split} count changed")
        for group in (
            "sand_benign_mild",
            "sand_benign_moderate",
            "ordinary_support_control",
            "delayed_support_control",
        ):
            if group_counts[(split, group)] != int(expected[split][group]):
                raise ValueError(f"factor-conditioned {split}/{group} count changed")
    policy = document["anti_contamination"]["near_signature_policy"]
    historical = _historical_overlap_audit(
        root,
        document["anti_contamination"]["historical_manifests"],
        rows,
        policy,
    )
    if (
        historical["exact_total"]
        or historical["near_total"]
        or historical["run_id_reuse_total"]
    ):
        raise ValueError("factor-conditioned matrix overlaps historical evidence")
    train = [row for row in rows if row["split"] == FACTOR_CONDITIONED_SPLITS[0]]
    validation = [row for row in rows if row["split"] == FACTOR_CONDITIONED_SPLITS[1]]
    exact_overlap = len(
        {_signature(row) for row in train} & {_signature(row) for row in validation}
    )
    near_pairs = [
        (str(left["run_id"]), str(right["run_id"]))
        for left in train
        for right in validation
        if _scenario_signatures_are_near(_signature(left), _signature(right), policy)
    ]
    if exact_overlap or near_pairs:
        raise ValueError("factor TRAIN and VALIDATION have exact or near overlap")
    component_hashes = factor_conditioned_component_hashes(document)
    frozen = document.get("design_freeze", {})
    expected_components = frozen.get("component_hashes")
    if expected_components and "TO_BE_FROZEN" not in expected_components.values():
        if component_hashes != dict(expected_components):
            raise ValueError("factor-conditioned component hash changed")
    matrix_sha = canonical_sha256(rows)
    signature_sha = canonical_sha256([list(value) for value in signatures])
    split_sha = {
        split: canonical_sha256(
            [row["run_id"] for row in rows if row["split"] == split]
        )
        for split in FACTOR_CONDITIONED_SPLITS
    }
    for field, actual in (
        ("scenario_matrix_sha256", matrix_sha),
        ("scenario_signature_sha256", signature_sha),
    ):
        expected_value = frozen.get(field)
        if expected_value not in (None, "TO_BE_FROZEN") and expected_value != actual:
            raise ValueError(f"factor-conditioned {field} changed")
    expected_split = frozen.get("split_sha256")
    if expected_split and "TO_BE_FROZEN" not in expected_split.values():
        if split_sha != dict(expected_split):
            raise ValueError("factor-conditioned split hash changed")
    implementation = {
        str(record["path"]): sha256_file(root / str(record["path"]))
        for record in frozen.get("implementation_artifacts", ())
    }
    expected_implementation = {
        str(record["path"]): str(record["sha256"])
        for record in frozen.get("implementation_artifacts", ())
    }
    if (
        expected_implementation
        and "TO_BE_FROZEN" not in expected_implementation.values()
    ):
        if implementation != expected_implementation:
            dataset_path = root / str(
                document.get("factor_dataset", {}).get("path", "")
            )
            manifest_path = dataset_path / "manifest.json"
            frozen_implementation: dict[str, str] = {}
            if manifest_path.is_file():
                frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                frozen_implementation = {
                    str(path): str(value)
                    for path, value in frozen_manifest.get(
                        "implementation_sha256", {}
                    ).items()
                }
            if frozen_implementation != expected_implementation:
                raise ValueError("factor-conditioned implementation hash changed")
            implementation = frozen_implementation
    return {
        "run_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "group_counts": {
            f"{split}/{group}": count
            for (split, group), count in sorted(group_counts.items())
        },
        "unique_run_ids": len(set(ids)),
        "unique_scenario_signatures": len(set(signatures)),
        "historical_contamination": historical,
        "cross_split_exact_overlap": exact_overlap,
        "cross_split_parameter_near_overlap": len(near_pairs),
        "scenario_matrix_sha256": matrix_sha,
        "scenario_signature_sha256": signature_sha,
        "split_sha256": split_sha,
        "component_hashes": component_hashes,
        "implementation_sha256": implementation,
    }


def _failure_outcome_class(row: Mapping[str, Any]) -> str:
    """Map a frozen physical record to the predeclared audit class."""
    invalid_reason = row.get("invalid_reason")
    if invalid_reason == "pretarget_fall":
        return "PRETARGET_FALL"
    if invalid_reason == "insufficient_post_target_observation":
        return "TARGET_FOLLOWING_FALL_CENSOR"
    if not bool(row["valid"]):
        return "OTHER_INVALID"
    return {
        "STRICT_BENIGN": "STRICT_SAND_BENIGN",
        "SUPPORT": "SUPPORT",
        "SLIP": "SLIP",
        "DUAL_HAZARD": "DUAL_HAZARD",
    }.get(str(row["objective_physical_outcome"]), "OTHER_INVALID")


def _physical_quantiles(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _failure_cell_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(_failure_outcome_class(row) for row in rows)
    eligible = outcomes["STRICT_SAND_BENIGN"] + outcomes["SUPPORT"]
    rate = eligible / max(1, len(rows))
    category = "STABLE" if rate >= 0.75 else "MARGINAL" if rate >= 0.50 else "UNSTABLE"
    return {
        "planned": len(rows),
        "strict_sand": outcomes["STRICT_SAND_BENIGN"],
        "support": outcomes["SUPPORT"],
        "slip_dual": outcomes["SLIP"] + outcomes["DUAL_HAZARD"],
        "pretarget_fall": outcomes["PRETARGET_FALL"],
        "post_target_censor": outcomes["TARGET_FOLLOWING_FALL_CENSOR"],
        "other_invalid": outcomes["OTHER_INVALID"],
        "eligible": eligible,
        "eligible_rate": rate,
        "physical_region": category,
    }


def audit_factor_conditioned_physical_failure(
    root: Path, audit_config_path: Path
) -> dict[str, Any]:
    """Decompose the failed corpus from frozen metadata without model access."""
    document = _load_yaml(audit_config_path)
    if document["experiment"]["id"] != "DATA_INTERVENTION_FAILURE_AUDIT":
        raise ValueError("unsupported factor-conditioned failure audit")
    failed = document["failed_dataset"]
    dataset_path = root / str(failed["path"])
    expected_files = {
        "manifest": (dataset_path / "manifest.json", failed["manifest_sha256"]),
        "physical_audit": (
            dataset_path / "physical_audit.json",
            failed["physical_audit_sha256"],
        ),
        "dataset_freeze": (
            dataset_path / "dataset_freeze.json",
            failed["dataset_freeze_file_sha256"],
        ),
        "intervention_config": (
            root / str(failed["intervention_config"]["path"]),
            failed["intervention_config"]["sha256"],
        ),
        "intervention_report": (
            root / str(failed["intervention_report"]["path"]),
            failed["intervention_report"]["sha256"],
        ),
    }
    integrity = {
        name: sha256_file(path) == str(expected)
        for name, (path, expected) in expected_files.items()
    }
    if not all(integrity.values()):
        raise RuntimeError("failed factor-conditioned evidence changed")
    manifest = json.loads((dataset_path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_id"] != failed["id"]:
        raise RuntimeError("failed factor-conditioned dataset identity changed")
    if manifest["model_inference_runs"] or any(
        row.get("model_outputs_present") for row in manifest["runs"]
    ):
        raise RuntimeError("model output entered physical failure audit")
    rows = list(manifest["runs"])
    overall = Counter(_failure_outcome_class(row) for row in rows)
    by_split = {
        split: _failure_cell_summary([row for row in rows if row["split"] == split])
        for split in FACTOR_CONDITIONED_SPLITS
    }
    source_speed: dict[str, Any] = {}
    for split in FACTOR_CONDITIONED_SPLITS:
        for source in ("concrete", "marble"):
            for speed in (0.20, 0.25, 0.30):
                selected = [
                    row
                    for row in rows
                    if row["split"] == split
                    and row["source_terrain"] == source
                    and float(row["speed_mps"]) == speed
                ]
                source_speed[f"{split}/{source}/{speed:.2f}"] = _failure_cell_summary(
                    selected
                )
    topology_phase_severity: dict[str, Any] = {}
    for split in FACTOR_CONDITIONED_SPLITS:
        levels = sorted(
            {
                (
                    str(row["sink_pattern"]),
                    str(row["target_contact_summary"]["precontact_phase"]),
                    str(row["severity_intent"]),
                )
                for row in rows
                if row["split"] == split
            }
        )
        for topology, phase, severity in levels:
            selected = [
                row
                for row in rows
                if row["split"] == split
                and row["sink_pattern"] == topology
                and row["target_contact_summary"]["precontact_phase"] == phase
                and row["severity_intent"] == severity
            ]
            topology_phase_severity[f"{split}/{topology}/{phase}/{severity}"] = (
                _failure_cell_summary(selected)
            )

    requested_ms = int(document["post_target_taxonomy"]["requested_observation_ms"])
    pretarget_rows: list[dict[str, Any]] = []
    post_target_rows: list[dict[str, Any]] = []
    post_categories: Counter[str] = Counter()
    for row in rows:
        outcome_class = _failure_outcome_class(row)
        first = row["target_contact_summary"]["first_sample"]
        fall = row["fall_censor_summary"]["first_fall_sample"]
        censor = int(row["fall_censor_summary"]["censor_sample"])
        common = {
            "run_id": row["run_id"],
            "split": row["split"],
            "source": row["source_terrain"],
            "speed_mps": row["speed_mps"],
            "topology": row["sink_pattern"],
            "phase": row["target_contact_summary"]["precontact_phase"],
            "severity": row["severity_intent"],
            "start_m": row["patch_start_x_m"],
            "width_m": row["patch_width_m"],
            "exit_m": round(
                float(row["patch_start_x_m"]) + float(row["patch_width_m"]), 6
            ),
            "target_exposure_ms": row["physical_signature"][
                "target_contact_duration_ms"
            ],
            "factor_manifold_intent": row["factor_manifold_intent"],
            "realization_id": row["realization_id"],
        }
        if outcome_class == "PRETARGET_FALL":
            pretarget_rows.append(
                {
                    **common,
                    "fall_ms": fall,
                    "target_ms": first,
                    "fall_minus_target_ms": (
                        None if first is None else int(fall) - int(first)
                    ),
                }
            )
        elif outcome_class == "TARGET_FOLLOWING_FALL_CENSOR":
            elapsed = None if first is None else censor - int(first)
            if fall is not None and elapsed is not None and elapsed < requested_ms:
                category = "IMMEDIATE_TARGET_INDUCED_INSTABILITY"
            elif fall is not None and elapsed is not None and elapsed >= requested_ms:
                category = "SHORT_LIVED_BENIGN_THEN_LATER_FALL"
            elif fall is None and elapsed is not None and elapsed >= requested_ms:
                category = "PHYSICALLY_VALID_TARGET_PERIOD_STRICT_LAST_CONTACT_RULE"
            elif fall is None and elapsed is not None and elapsed < requested_ms:
                category = "TRUE_INSUFFICIENT_HORIZON"
            else:
                category = "OTHER"
            post_categories[category] += 1
            post_target_rows.append(
                {
                    **common,
                    "target_ms": first,
                    "fall_or_censor_ms": fall if fall is not None else censor,
                    "fall_minus_first_target_ms": elapsed,
                    "valid_post_last_target_ms": row["target_contact_summary"][
                        "post_target_observation_ms"
                    ],
                    "requested_post_target_ms": requested_ms,
                    "taxonomy": category,
                }
            )

    geometry: dict[str, Any] = {}
    for outcome_name in (
        "STRICT_SAND_BENIGN",
        "PRETARGET_FALL",
        "TARGET_FOLLOWING_FALL_CENSOR",
        "SLIP",
        "DUAL_HAZARD",
        "SUPPORT",
        "OTHER_INVALID",
    ):
        selected = [row for row in rows if _failure_outcome_class(row) == outcome_name]
        geometry[outcome_name] = {
            "start_m": _physical_quantiles(
                [float(row["patch_start_x_m"]) for row in selected]
            ),
            "width_m": _physical_quantiles(
                [float(row["patch_width_m"]) for row in selected]
            ),
            "exit_m": _physical_quantiles(
                [
                    float(row["patch_start_x_m"]) + float(row["patch_width_m"])
                    for row in selected
                ]
            ),
            "target_exposure_ms": _physical_quantiles(
                [
                    float(row["physical_signature"]["target_contact_duration_ms"])
                    for row in selected
                ]
            ),
            "entry_ms": _physical_quantiles(
                [
                    float(row["target_contact_summary"]["first_sample"])
                    for row in selected
                    if row["target_contact_summary"]["first_sample"] is not None
                ]
            ),
            "topology": dict(
                sorted(Counter(str(row["sink_pattern"]) for row in selected).items())
            ),
            "phase": dict(
                sorted(
                    Counter(
                        str(row["target_contact_summary"]["precontact_phase"])
                        for row in selected
                    ).items()
                )
            ),
        }

    mild_envelopes = document["historical_physical_reference"]["proven_mild_envelope"]

    def mild_relation(row: Mapping[str, Any]) -> str:
        if row["group"] != "sand_benign_mild":
            return "NOT_MILD"
        envelope = mild_envelopes[str(row["sink_pattern"])]
        values = (
            float(row["patch_start_x_m"]),
            float(row["patch_width_m"]),
            float(row["patch_start_x_m"]) + float(row["patch_width_m"]),
        )
        bounds = (
            envelope["patch_start_x_m"],
            envelope["patch_width_m"],
            envelope["patch_exit_x_m"],
        )
        if all(
            float(bound[0]) <= value <= float(bound[1])
            for value, bound in zip(values, bounds)
        ):
            return "INSIDE"
        distance = max(
            max(float(bound[0]) - value, 0.0, value - float(bound[1]))
            for value, bound in zip(values, bounds)
        )
        return "NEAR_BOUNDARY" if distance <= 0.004 else "OUTSIDE"

    mild_domain = {
        outcome_name: dict(
            sorted(
                Counter(
                    mild_relation(row)
                    for row in rows
                    if row["group"] == "sand_benign_mild"
                    and _failure_outcome_class(row) == outcome_name
                ).items()
            )
        )
        for outcome_name in (
            "STRICT_SAND_BENIGN",
            "PRETARGET_FALL",
            "TARGET_FOLLOWING_FALL_CENSOR",
        )
    }
    return {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "metadata_only": True,
        "model_output_fields": [],
        "input_integrity": integrity,
        "run_count": len(rows),
        "outcome_counts": {
            name: int(overall.get(name, 0))
            for name in (
                "STRICT_SAND_BENIGN",
                "SUPPORT",
                "SLIP",
                "DUAL_HAZARD",
                "PRETARGET_FALL",
                "TARGET_FOLLOWING_FALL_CENSOR",
                "OTHER_INVALID",
            )
        },
        "by_split": by_split,
        "source_speed": source_speed,
        "topology_phase_severity": topology_phase_severity,
        "geometry": geometry,
        "pretarget_falls": pretarget_rows,
        "post_target_fall_censors": post_target_rows,
        "post_target_taxonomy_counts": dict(sorted(post_categories.items())),
        "mild_relation_to_proven_envelope": mild_domain,
        "counters": document["boundaries"],
    }


def expand_factor_conditioned_redesign(
    document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand the frozen recalibrated factor-conditioned corpus design."""
    if document["experiment"]["id"] != FACTOR_CONDITIONED_REDESIGN_ID:
        raise ValueError("unsupported factor-conditioned redesign config")
    matrix = document["scenario_matrix"]
    rows: list[dict[str, Any]] = []
    split_codes = {"FACTOR_TRAIN": "t", "FACTOR_VALIDATION": "v"}
    group_codes = {
        "sand_benign_mild": "sml",
        "sand_benign_moderate": "smd",
        "ordinary_support_control": "osp",
        "delayed_support_control": "dsp",
    }
    for split in FACTOR_CONDITIONED_SPLITS:
        profiles = matrix["profiles"][split]
        for cell in matrix["source_speed_cells"]:
            source = str(cell["source_terrain"])
            speed = float(cell["speed_mps"])
            source_code = "c" if source == "concrete" else "m"
            speed_code = f"{int(round(speed * 100)):03d}"
            concrete_025 = source == "concrete" and speed == 0.25
            selected_profiles = {
                "sand_benign_mild": profiles[
                    "mild_concrete_025_left" if concrete_025 else "mild_standard"
                ],
                "sand_benign_moderate": profiles[
                    "moderate_concrete_025_left"
                    if concrete_025
                    else "moderate_standard"
                ],
                "ordinary_support_control": profiles["ordinary_support"],
                "delayed_support_control": profiles["delayed_support"],
            }
            for group, group_profiles in selected_profiles.items():
                for index, profile in enumerate(group_profiles, start=1):
                    support = group.endswith("support_control")
                    delayed = group == "delayed_support_control"
                    moderate = group != "sand_benign_mild"
                    side = str(profile.get("designed_side", "LEFT"))
                    topology = str(profile.get("sink_pattern", "transition_left"))
                    rows.append(
                        {
                            "run_id": (
                                f"{matrix['run_id_prefix']}_{split_codes[split]}_"
                                f"{group_codes[group]}_{source_code}_{speed_code}_{index:02d}"
                            ),
                            "split": split,
                            "group": group,
                            "scenario_family": group,
                            "factor_manifold_intent": str(
                                profile.get(
                                    "factor_manifold",
                                    "CONTROL" if support else "ADVERSE_DIRECTION",
                                )
                            ),
                            "source_terrain": source,
                            "target_terrain": "sand",
                            "speed_mps": speed,
                            "nominal_speed_mps": speed,
                            "designed_role": "HAZARD" if support else "NORMAL",
                            "designed_event_type": "SUPPORT" if support else "NONE",
                            "designed_side": side,
                            "designed_side_topology": (
                                f"{side}_ONLY" if support else "NONE"
                            ),
                            "patch_start_x_m": float(profile["patch_start_x_m"]),
                            "patch_width_m": float(profile["patch_width_m"]),
                            "slip_pattern": "uniform",
                            "sink_pattern": topology,
                            "sink_severity": "moderate" if moderate else "mild",
                            "support_pattern": (
                                "staged_lateral_deformable"
                                if delayed
                                else "lateral_deformable"
                                if support
                                else "balanced_deformable"
                            ),
                            "severity_intent": (
                                "SUPPORT_CONTROL"
                                if support
                                else "BOUNDARY_ADJACENT"
                                if moderate
                                else "LOW"
                            ),
                            "realization_id": str(profile["id"]),
                        }
                    )
    return rows


def factor_conditioned_redesign_component_hashes(
    document: Mapping[str, Any],
) -> dict[str, str]:
    sections = {
        "RECALIBRATED_PARAMETER_DOMAIN_SHA": "parameter_domain",
        "RECALIBRATED_SCENARIO_MATRIX_SHA": "scenario_matrix",
        "RECALIBRATED_SPLIT_PLAN_SHA": "split_plan",
        "RECALIBRATED_PHYSICAL_LABEL_CONTRACT_SHA": "physical_label_contract",
        "RECALIBRATED_GENERATION_GATES_SHA": "generation_gates",
        "RECALIBRATED_ANTI_CONTAMINATION_SHA": "anti_contamination",
    }
    return {
        name: canonical_sha256(document[section]) for name, section in sections.items()
    }


def validate_factor_conditioned_redesign(
    root: Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify deterministic counts, factor coverage, and fresh signatures."""
    rows = expand_factor_conditioned_redesign(document)
    counts = document["scenario_matrix"]["counts"]
    if len(rows) != int(counts["total"]) or not 160 <= len(rows) <= 320:
        raise ValueError("recalibrated factor-conditioned total changed")
    ids = [str(row["run_id"]) for row in rows]
    signatures = [_signature(row) for row in rows]
    if len(set(ids)) != len(ids) or len(set(signatures)) != len(signatures):
        raise ValueError("recalibrated matrix has duplicate IDs or signatures")
    split_counts = Counter(str(row["split"]) for row in rows)
    group_counts = Counter(str(row["group"]) for row in rows)
    group_names = (
        "sand_benign_mild",
        "sand_benign_moderate",
        "ordinary_support_control",
        "delayed_support_control",
    )
    for split in FACTOR_CONDITIONED_SPLITS:
        if split_counts[split] != int(counts[split]["total"]):
            raise ValueError(f"recalibrated {split} count changed")
        for group in group_names:
            actual = sum(
                row["split"] == split and row["group"] == group for row in rows
            )
            if actual != int(counts[split][group]):
                raise ValueError(f"recalibrated {split}/{group} count changed")
    expected_groups = {group: int(counts["aggregate"][group]) for group in group_names}
    if dict(group_counts) != expected_groups:
        raise ValueError("recalibrated aggregate group counts changed")

    for split in FACTOR_CONDITIONED_SPLITS:
        for source in ("concrete", "marble"):
            for speed in (0.20, 0.25, 0.30):
                mild = [
                    row
                    for row in rows
                    if row["split"] == split
                    and row["source_terrain"] == source
                    and float(row["speed_mps"]) == speed
                    and row["group"] == "sand_benign_mild"
                ]
                expected_mild = int(counts[split]["mild_per_source_speed"])
                if len(mild) != expected_mild:
                    raise ValueError("recalibrated mild cell count changed")
                topologies = {str(row["sink_pattern"]) for row in mild}
                expected_topologies = (
                    {"transition_left"}
                    if source == "concrete" and speed == 0.25
                    else {"transition_left", "transition_right"}
                )
                if topologies != expected_topologies:
                    raise ValueError("recalibrated factor manifold coverage changed")

    policy = document["anti_contamination"]["near_signature_policy"]
    historical = _historical_overlap_audit(
        root,
        document["anti_contamination"]["historical_manifests"],
        rows,
        policy,
    )
    if (
        historical["exact_total"]
        or historical["near_total"]
        or historical["run_id_reuse_total"]
    ):
        raise ValueError("recalibrated matrix overlaps protected evidence")
    train = [row for row in rows if row["split"] == "FACTOR_TRAIN"]
    validation = [row for row in rows if row["split"] == "FACTOR_VALIDATION"]
    exact_overlap = len(
        {_signature(row) for row in train} & {_signature(row) for row in validation}
    )
    near_pairs = [
        (str(left["run_id"]), str(right["run_id"]))
        for left in train
        for right in validation
        if _scenario_signatures_are_near(_signature(left), _signature(right), policy)
    ]
    if exact_overlap or near_pairs:
        raise ValueError("recalibrated TRAIN and VALIDATION overlap")
    component_hashes = factor_conditioned_redesign_component_hashes(document)
    matrix_sha = canonical_sha256(rows)
    signature_sha = canonical_sha256([list(value) for value in signatures])
    split_sha = {
        split: canonical_sha256(
            [row["run_id"] for row in rows if row["split"] == split]
        )
        for split in FACTOR_CONDITIONED_SPLITS
    }
    frozen = document.get("design_freeze", {})
    expected_components = frozen.get("component_hashes", {})
    if expected_components and "TO_BE_FROZEN" not in expected_components.values():
        if dict(expected_components) != component_hashes:
            raise ValueError("recalibrated component hash changed")
    for name, actual in (
        ("scenario_matrix_sha256", matrix_sha),
        ("scenario_signature_sha256", signature_sha),
    ):
        expected = frozen.get(name)
        if expected not in (None, "TO_BE_FROZEN") and expected != actual:
            raise ValueError(f"recalibrated {name} changed")
    expected_split = frozen.get("split_sha256", {})
    if expected_split and "TO_BE_FROZEN" not in expected_split.values():
        if dict(expected_split) != split_sha:
            raise ValueError("recalibrated split hash changed")
    return {
        "run_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "unique_run_ids": len(set(ids)),
        "unique_scenario_signatures": len(set(signatures)),
        "historical_contamination": historical,
        "cross_split_exact_overlap": exact_overlap,
        "cross_split_parameter_near_overlap": len(near_pairs),
        "scenario_matrix_sha256": matrix_sha,
        "scenario_signature_sha256": signature_sha,
        "split_sha256": split_sha,
        "component_hashes": component_hashes,
    }


def _factor_conditioned_audit(
    manifest: Mapping[str, Any],
    document: Mapping[str, Any],
    matrix_audit: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(manifest["runs"])
    gates = document["generation_gates"]
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, actual: Any, requirement: str, passed: bool) -> None:
        checks[name] = {
            "actual": actual,
            "requirement": requirement,
            "passed": bool(passed),
        }

    add(
        "execution/completed",
        len(rows),
        str(matrix_audit["run_count"]),
        len(rows) == matrix_audit["run_count"],
    )
    add(
        "execution/adaptive_backfill",
        manifest["adaptive_backfill_count"],
        "0",
        manifest["adaptive_backfill_count"] == 0,
    )
    add(
        "execution/replacement",
        manifest["replacement_run_count"],
        "0",
        manifest["replacement_run_count"] == 0,
    )
    add(
        "integrity/historical_exact_overlap",
        matrix_audit["historical_contamination"]["exact_total"],
        "0",
        matrix_audit["historical_contamination"]["exact_total"] == 0,
    )
    add(
        "integrity/historical_near_overlap",
        matrix_audit["historical_contamination"]["near_total"],
        "0",
        matrix_audit["historical_contamination"]["near_total"] == 0,
    )
    add(
        "integrity/historical_run_id_reuse",
        matrix_audit["historical_contamination"]["run_id_reuse_total"],
        "0",
        matrix_audit["historical_contamination"]["run_id_reuse_total"] == 0,
    )
    add(
        "integrity/cross_split_exact_overlap",
        matrix_audit["cross_split_exact_overlap"],
        "0",
        matrix_audit["cross_split_exact_overlap"] == 0,
    )
    add(
        "integrity/cross_split_near_overlap",
        matrix_audit["cross_split_parameter_near_overlap"],
        "0",
        matrix_audit["cross_split_parameter_near_overlap"] == 0,
    )
    add(
        "integrity/model_outputs",
        manifest["model_inference_runs"],
        "0",
        manifest["model_inference_runs"] == 0,
    )

    valid = [row for row in rows if _factor_conditioned_eligible(row)]
    add(
        "yield/objective_valid",
        len(valid),
        f">={gates['overall_objective_valid_min']}",
        len(valid) >= int(gates["overall_objective_valid_min"]),
    )
    by_role: dict[str, Any] = {}
    for split in FACTOR_CONDITIONED_SPLITS:
        selected = [row for row in rows if row["split"] == split]
        strict = [
            row
            for row in selected
            if _factor_conditioned_eligible(row)
            and row["objective_physical_outcome"] == "STRICT_BENIGN"
        ]
        mild = [
            row
            for row in strict
            if row["group"] == "sand_benign_mild"
            and row["actual_benign_severity"] == "LOW"
        ]
        moderate = [
            row
            for row in strict
            if row["group"] == "sand_benign_moderate"
            and row["actual_benign_severity"] == "MEDIUM"
        ]
        ordinary = [
            row
            for row in selected
            if _factor_conditioned_eligible(row)
            and row["group"] == "ordinary_support_control"
        ]
        delayed = [
            row
            for row in selected
            if _factor_conditioned_eligible(row)
            and row["group"] == "delayed_support_control"
        ]
        split_gate = gates[split]
        for name, values in (
            ("strict_sand", strict),
            ("mild", mild),
            ("moderate", moderate),
            ("ordinary_support", ordinary),
            ("delayed_support", delayed),
        ):
            minimum = int(split_gate[f"{name}_min"])
            add(
                f"yield/{split}/{name}",
                len(values),
                f">={minimum}",
                len(values) >= minimum,
            )
        cell_counts: dict[str, int] = {}
        for source in ("concrete", "marble"):
            for speed in (0.20, 0.25, 0.30):
                cell = [
                    row
                    for row in strict
                    if row["source_terrain"] == source
                    and float(row["speed_mps"]) == speed
                ]
                key = f"{source}/{speed:.2f}"
                cell_counts[key] = len(cell)
                minimum = int(split_gate["strict_sand_per_source_speed_min"])
                add(
                    f"yield/{split}/{key}/strict_sand",
                    len(cell),
                    f">={minimum}",
                    len(cell) >= minimum,
                )
        phases = Counter(
            str(row["target_contact_summary"]["precontact_phase"]) for row in strict
        )
        topologies = Counter(str(row["sink_pattern"]) for row in strict)
        add(
            f"diversity/{split}/principal_phases",
            sum(
                phases[name] > 0
                for name in ("LEFT_SINGLE_SUPPORT", "RIGHT_SINGLE_SUPPORT")
            ),
            "2",
            all(
                phases[name] > 0
                for name in ("LEFT_SINGLE_SUPPORT", "RIGHT_SINGLE_SUPPORT")
            ),
        )
        add(
            f"diversity/{split}/topologies",
            sum(
                topologies[name] > 0 for name in ("transition_left", "transition_right")
            ),
            "2",
            all(
                topologies[name] > 0 for name in ("transition_left", "transition_right")
            ),
        )
        by_role[split] = {
            "planned": len(selected),
            "objective_valid": len([row for row in selected if row in valid]),
            "strict_sand": len(strict),
            "mild": len(mild),
            "moderate": len(moderate),
            "ordinary_support": len(ordinary),
            "delayed_support": len(delayed),
            "source_speed_strict": cell_counts,
            "precontact_phase": dict(sorted(phases.items())),
            "topology": dict(sorted(topologies.items())),
        }
    physical_signatures = [canonical_sha256(row["physical_signature"]) for row in valid]
    unique_fraction = len(set(physical_signatures)) / max(1, len(physical_signatures))
    minimum_unique = float(gates["unique_physical_signature_fraction_min"])
    add(
        "diversity/unique_physical_signature_fraction",
        unique_fraction,
        f">={minimum_unique}",
        unique_fraction >= minimum_unique,
    )
    outcome_counts = Counter(str(row["objective_physical_outcome"]) for row in rows)
    invalid_reasons = Counter(
        str(row["invalid_reason"]) for row in rows if row["invalid_reason"] is not None
    )
    verdict = (
        "FACTOR_CONDITIONED_DATASET_GENERATION_READY"
        if all(value["passed"] for value in checks.values())
        else "FACTOR_CONDITIONED_DATASET_GENERATION_GATES_FAILED"
    )
    return {
        "generation_verdict": verdict,
        "all_gates_passed": all(value["passed"] for value in checks.values()),
        "generation_gates": checks,
        "role_summary": by_role,
        "physical_outcomes": dict(sorted(outcome_counts.items())),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "training_eligible_count": sum(
            bool(row.get("training_eligible")) for row in rows
        ),
        "validation_eligible_count": sum(
            bool(row.get("validation_eligible")) for row in rows
        ),
    }


def verify_factor_conditioned_dataset(dataset_path: Path) -> dict[str, Any]:
    """Verify frozen corpus files without running a model."""
    manifest_path = dataset_path / "manifest.json"
    freeze_path = dataset_path / "dataset_freeze.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    semantic = dict(freeze)
    expected_semantic = semantic.pop("FACTOR_DATASET_FREEZE_SHA")
    npz_hashes = {
        str(row["file"]): sha256_file(dataset_path / str(row["file"]))
        for row in manifest["runs"]
    }
    checks = {
        "dataset_id": manifest["dataset_id"] == FACTOR_CONDITIONED_DATASET_ID,
        "manifest_sha": sha256_file(manifest_path) == freeze["FACTOR_MANIFEST_SHA"],
        "physical_audit_sha": sha256_file(dataset_path / "physical_audit.json")
        == freeze["FACTOR_PHYSICAL_AUDIT_SHA"],
        "config_provenance": manifest["intervention_config_sha256"]
        == freeze["intervention_config_sha256"],
        "scenario_matrix": manifest["scenario_matrix_sha256"]
        == freeze["FACTOR_SCENARIO_MATRIX_SHA"],
        "scenario_signatures": manifest["scenario_signature_sha256"]
        == freeze["FACTOR_SCENARIO_SIGNATURE_SHA"],
        "split_hashes": manifest["split_sha256"]
        == {
            "FACTOR_TRAIN": freeze["FACTOR_TRAIN_SPLIT_SHA"],
            "FACTOR_VALIDATION": freeze["FACTOR_VALIDATION_SPLIT_SHA"],
        },
        "implementation_hashes": canonical_sha256(manifest["implementation_sha256"])
        == freeze["FACTOR_IMPLEMENTATION_SHA"],
        "npz_hashes": npz_hashes
        == {str(row["file"]): str(row["file_sha256"]) for row in manifest["runs"]},
        "npz_aggregate_sha": canonical_sha256(npz_hashes)
        == freeze["FACTOR_NPZ_AGGREGATE_SHA"],
        "semantic_freeze_sha": canonical_sha256(semantic) == expected_semantic,
        "freeze_file_sha": sha256_file(freeze_path)
        == (dataset_path / "dataset_freeze.sha256")
        .read_text(encoding="utf-8")
        .split()[0],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "run_count": len(manifest["runs"]),
        "dataset_freeze_file_sha256": sha256_file(freeze_path),
        "dataset_freeze_semantic_sha256": expected_semantic,
    }


def collect_factor_conditioned_dataset(
    root: Path,
    config_path: Path,
    policy_override: Path | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Generate and freeze one fresh model-blind factor-conditioned corpus."""
    document = _load_yaml(config_path)
    audit_before = validate_factor_conditioned_design(root, document)
    generation = document["generation"]
    if str(generation["dataset_id"]) != FACTOR_CONDITIONED_DATASET_ID:
        raise RuntimeError("factor-conditioned dataset identity changed")
    policy_path = (
        root / str(generation["policy_path"])
        if policy_override is None
        else policy_override
    )
    if sha256_file(policy_path) != str(generation["policy_sha256"]):
        raise RuntimeError("walking policy differs from intervention freeze")
    simulator_path = root / str(generation["simulator_config_path"])
    if sha256_file(simulator_path) != str(generation["simulator_config_sha256"]):
        raise RuntimeError("simulator config differs from intervention freeze")
    guard_path = root / str(
        document["historical_evidence_boundary"]["holdout_guard_path"]
    )
    if sha256_file(guard_path) != str(
        document["historical_evidence_boundary"]["holdout_guard_sha256"]
    ):
        raise RuntimeError("historical HOLDOUT guard changed")
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    if guard.get("guard_after") != 1 or guard.get("scientific_open_count") != 1:
        raise RuntimeError("historical HOLDOUT guard state changed")
    output_path = root / str(generation["dataset_path"])
    partial_path = output_path.with_name(f"{output_path.name}_partial")
    if output_path.exists() or partial_path.exists():
        raise RuntimeError(f"factor-conditioned output already exists: {output_path}")
    partial_path.mkdir(parents=True)
    specifications = expand_factor_conditioned_design(document)
    config_sha = sha256_file(config_path)
    _write_json(
        partial_path / "pre_simulation_freeze.json",
        {
            "schema_version": 1,
            "status": "FROZEN_BEFORE_FIRST_SIMULATION",
            "dataset_id": FACTOR_CONDITIONED_DATASET_ID,
            "source_commit": document["experiment"]["source_commit"],
            "intervention_config_sha256": config_sha,
            "component_hashes": audit_before["component_hashes"],
            "implementation_sha256": audit_before["implementation_sha256"],
            "scenario_matrix_sha256": audit_before["scenario_matrix_sha256"],
            "scenario_signature_sha256": audit_before["scenario_signature_sha256"],
            "split_sha256": audit_before["split_sha256"],
            "planned_run_count": len(specifications),
            "model_inference": False,
            "adaptive_backfill": False,
            "replacement": False,
        },
    )
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for index, specification in enumerate(specifications, start=1):
            result = run_simulation(
                SimulationConfig(
                    physics_timestep_s=float(generation["physics_timestep_s"]),
                    sensor_rate_hz=int(generation["sensor_rate_hz"]),
                    duration_s=float(generation["simulation_duration_s"]),
                    command_speed_mps=float(specification["speed_mps"]),
                    policy_path=policy_path,
                    terrain="sand",
                    slip_pattern="uniform",
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
            routed = _annotation_specification(specification)
            row, arrays = annotate_model_v2_result(routed, result)
            row["scenario_family"] = routed["scenario_family"]
            row["group"] = specification["group"]
            row["factor_manifold_intent"] = specification["factor_manifold_intent"]
            row["realization_id"] = specification["realization_id"]
            if result.stability is None:
                raise RuntimeError("factor-conditioned corpus requires gait phase")
            arrays["gait_phase"] = np.asarray(
                result.stability.gait_phase, dtype=np.int8
            )
            _calibration_result_summary(row, arrays, generation["label_execution"])
            phase = str(row["target_contact_summary"]["precontact_phase"])
            topology = str(row["sink_pattern"])
            row["factor_manifold"] = (
                "ADVERSE_DIRECTION"
                if topology == "transition_left" and phase == "RIGHT_SINGLE_SUPPORT"
                else "COMPARISON_DIRECTION"
                if topology == "transition_right" and phase == "LEFT_SINGLE_SUPPORT"
                else "OTHER_PREDECLARED"
            )
            eligible = _factor_conditioned_eligible(row)
            row["training_eligible"] = eligible and row["split"] == "FACTOR_TRAIN"
            row["validation_eligible"] = (
                eligible and row["split"] == "FACTOR_VALIDATION"
            )
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
                    f"generated {index}/{len(specifications)}: {specification['run_id']}"
                )
        manifest = {
            "schema_version": 1,
            "dataset_id": FACTOR_CONDITIONED_DATASET_ID,
            "created_at": generation["created_at"],
            "generation_source_commit": document["experiment"]["source_commit"],
            "intervention_config_path": str(config_path.relative_to(root)),
            "intervention_config_sha256": config_sha,
            "component_hashes": audit_before["component_hashes"],
            "implementation_sha256": audit_before["implementation_sha256"],
            "scenario_matrix_sha256": audit_before["scenario_matrix_sha256"],
            "scenario_signature_sha256": audit_before["scenario_signature_sha256"],
            "split_sha256": audit_before["split_sha256"],
            "matrix_audit": audit_before,
            "policy_sha256": generation["policy_sha256"],
            "simulator_config_sha256": generation["simulator_config_sha256"],
            "model_blind": True,
            "model_inference_runs": 0,
            "old_holdout_payload_reads": 0,
            "attempted_run_count": len(specifications),
            "run_count": len(rows),
            "adaptive_backfill_count": 0,
            "replacement_run_count": 0,
            "rerun_count": 0,
            "generation_order": [row["run_id"] for row in rows],
            "model_output_fields": [],
            "runs": rows,
        }
        manifest_path = partial_path / "manifest.json"
        _write_json(manifest_path, manifest)
        (partial_path / "manifest.sha256").write_text(
            f"{sha256_file(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        physical_audit = _factor_conditioned_audit(manifest, document, audit_before)
        audit_path = partial_path / "physical_audit.json"
        _write_json(audit_path, physical_audit)
        npz_hashes = {str(row["file"]): str(row["file_sha256"]) for row in rows}
        physical_signatures = [
            {"run_id": row["run_id"], **row["physical_signature"]} for row in rows
        ]
        physical_outcomes = [
            {
                "run_id": row["run_id"],
                "valid": row["valid"],
                "outcome": row["objective_physical_outcome"],
                "severity": row["actual_benign_severity"],
                "invalid_reason": row["invalid_reason"],
            }
            for row in rows
        ]
        freeze = {
            "schema_version": 1,
            "dataset_id": FACTOR_CONDITIONED_DATASET_ID,
            "generation_source_commit": document["experiment"]["source_commit"],
            "intervention_config_sha256": config_sha,
            "run_count": len(rows),
            "training_eligible_count": physical_audit["training_eligible_count"],
            "validation_eligible_count": physical_audit["validation_eligible_count"],
            "FACTOR_MANIFEST_SHA": sha256_file(manifest_path),
            "FACTOR_SCENARIO_MATRIX_SHA": audit_before["scenario_matrix_sha256"],
            "FACTOR_TRAIN_SPLIT_SHA": audit_before["split_sha256"]["FACTOR_TRAIN"],
            "FACTOR_VALIDATION_SPLIT_SHA": audit_before["split_sha256"][
                "FACTOR_VALIDATION"
            ],
            "FACTOR_SCENARIO_SIGNATURE_SHA": audit_before["scenario_signature_sha256"],
            "FACTOR_PHYSICAL_SIGNATURE_SHA": canonical_sha256(physical_signatures),
            "FACTOR_IMPLEMENTATION_SHA": canonical_sha256(
                audit_before["implementation_sha256"]
            ),
            "FACTOR_NPZ_AGGREGATE_SHA": canonical_sha256(npz_hashes),
            "FACTOR_PHYSICAL_OUTCOME_SHA": canonical_sha256(physical_outcomes),
            "FACTOR_GENERATION_GATE_RESULT_SHA": canonical_sha256(
                physical_audit["generation_gates"]
            ),
            "FACTOR_PHYSICAL_AUDIT_SHA": sha256_file(audit_path),
            "pre_simulation_freeze_sha256": sha256_file(
                partial_path / "pre_simulation_freeze.json"
            ),
            "generation_verdict": physical_audit["generation_verdict"],
            "model_inference_runs": 0,
            "old_holdout_payload_reads": 0,
            "adaptive_backfill_count": 0,
            "replacement_run_count": 0,
            "rerun_count": 0,
        }
        freeze["FACTOR_DATASET_FREEZE_SHA"] = canonical_sha256(freeze)
        freeze_path = partial_path / "dataset_freeze.json"
        _write_json(freeze_path, freeze)
        (partial_path / "dataset_freeze.sha256").write_text(
            f"{sha256_file(freeze_path)}  dataset_freeze.json\n", encoding="utf-8"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.rename(output_path)
    except Exception:
        shutil.rmtree(partial_path, ignore_errors=True)
        raise
    summary = {
        "dataset_id": FACTOR_CONDITIONED_DATASET_ID,
        "planned_runs": len(specifications),
        "completed_runs": len(rows),
        "training_eligible_runs": physical_audit["training_eligible_count"],
        "validation_eligible_runs": physical_audit["validation_eligible_count"],
        "generation_seconds": round(time.monotonic() - started, 3),
        "manifest_sha256": sha256_file(output_path / "manifest.json"),
        "dataset_freeze_file_sha256": sha256_file(output_path / "dataset_freeze.json"),
        "dataset_freeze_semantic_sha256": freeze["FACTOR_DATASET_FREEZE_SHA"],
        "generation_verdict": physical_audit["generation_verdict"],
        "model_inference_runs": 0,
        "old_holdout_payload_reads": 0,
    }
    _write_json(output_path / "generation_summary.json", summary)
    return output_path, summary
