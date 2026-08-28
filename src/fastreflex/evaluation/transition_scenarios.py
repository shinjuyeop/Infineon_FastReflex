"""Matched-prefix audit and frozen terrain-transition scenario calibration."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import mujoco
import numpy as np
import yaml

from fastreflex.simulation.g1 import (
    PHYSICS_TIMESTEP_S,
    SENSOR_RATE_HZ,
    SimulationConfig,
    SimulationResult,
    load_g1_model,
    load_simulation_config,
    run_simulation,
)
from fastreflex.simulation.stability import (
    HazardState,
    ParallelRuntimeState,
    StabilityState,
    TerrainState,
)
from fastreflex.simulation.terrain import (
    DEFORMABLE_SUPPORT_PROFILES,
    TERRAIN_PROFILES,
    low_friction_patch_geom_ids,
    soft_sink_geom_ids,
)


VALID_STABLE = "VALID_STABLE"
VALID_FALL = "VALID_FALL"
INVALID_PRETRANSITION = "INVALID_PRETRANSITION"
INVALID_NO_TARGET_CONTACT = "INVALID_NO_TARGET_CONTACT"
INVALID_OTHER = "INVALID_OTHER"
VALID_OUTCOMES = frozenset((VALID_STABLE, VALID_FALL))
SIGNATURE_FIELDS = (
    "target_terrain",
    "speed_mps",
    "patch_start_x_m",
    "patch_width_m",
    "slip_pattern",
    "sink_pattern",
    "sink_severity",
    "support_pattern",
)


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_true(values: np.ndarray) -> int | None:
    found = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if not found.size else int(found[0])


def _last_true(values: np.ndarray) -> int | None:
    found = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if not found.size else int(found[-1])


def _time_ms(result: SimulationResult, sample: int | None) -> float | None:
    if sample is None or not 0 <= sample < len(result.runtime.timestamp_us):
        return None
    return float(result.runtime.timestamp_us[sample]) / 1000.0


def target_contact_mask(
    result: SimulationResult, target_terrain: str
) -> np.ndarray:
    """Return exact named target-geom contact, never terrain-name inference."""
    if target_terrain == "ice":
        return np.asarray(result.diagnostics.low_friction_patch_contact, dtype=bool)
    if target_terrain == "sand":
        return np.asarray(result.diagnostics.soft_patch_contact, dtype=bool)
    raise ValueError(f"unsupported target terrain: {target_terrain}")


def _boundary_expected_sample(
    result: SimulationResult, patch_start_x_m: float
) -> int | None:
    # The canonical leading sole sphere is +0.12 m from the ankle-roll frame
    # with 5 mm radius. This clock is diagnostic; named-geom contact remains
    # the authoritative physical transition clock.
    leading_edge_x = result.diagnostics.foot_world_xyz[:, :, 0] + 0.125
    return _first_true(np.any(leading_edge_x >= patch_start_x_m, axis=1))


def _crossed_patch(result: SimulationResult, patch_end_x_m: float) -> bool:
    trailing_edge_x = result.diagnostics.foot_world_xyz[:, :, 0] - 0.055
    return bool(np.any(np.all(trailing_edge_x >= patch_end_x_m, axis=1)))


def classify_scenario_outcome(
    result: SimulationResult,
    specification: Mapping[str, object],
) -> str:
    """Classify observed physics without consulting the intended role."""
    target = str(specification["target_terrain"])
    contact = _first_true(np.any(target_contact_mask(result, target), axis=1))
    fall_raw = result.metadata["first_fall_sample"]
    fall = None if fall_raw is None else int(fall_raw)
    finite = bool(
        np.all(np.isfinite(result.runtime.pelvis_imu))
        and (
            result.runtime.foot_fsr is None
            or np.all(np.isfinite(result.runtime.foot_fsr))
        )
        and np.all(np.isfinite(result.diagnostics.foot_world_xyz))
        and result.metadata["actual_samples"] == result.metadata["expected_samples"]
        and not result.metadata["terminated_by_viewer"]
    )
    if not finite:
        return INVALID_OTHER
    if contact is None:
        return INVALID_NO_TARGET_CONTACT
    if fall is not None and fall < contact:
        return INVALID_PRETRANSITION
    minimum_prefix = int(specification.get("minimum_normal_prefix_ms", 500))
    boundary = _boundary_expected_sample(
        result, float(specification["patch_start_x_m"])
    )
    if boundary is None or contact < boundary or contact < minimum_prefix:
        return INVALID_OTHER
    if fall is not None:
        return VALID_FALL
    minimum_post = int(specification.get("minimum_post_contact_ms", 500))
    if len(result.runtime.sequence) - contact < minimum_post:
        return INVALID_OTHER
    patch_end = float(specification["patch_start_x_m"]) + float(
        specification["patch_width_m"]
    )
    if not _crossed_patch(result, patch_end):
        return INVALID_OTHER
    return VALID_STABLE


def scenario_timing_row(
    result: SimulationResult,
    specification: Mapping[str, object],
) -> dict[str, object]:
    target = str(specification["target_terrain"])
    contact_mask = target_contact_mask(result, target)
    any_target = np.any(contact_mask, axis=1)
    first_contact = _first_true(any_target)
    last_contact = _last_true(any_target)
    first_touchdown = _first_true(
        np.any(contact_mask & result.diagnostics.touchdown, axis=1)
    )
    first_slip = _first_true(
        result.diagnostics.any_established_slip_after_patch_onset
    )
    first_sink = _first_true(
        np.any(result.diagnostics.deformable_sink_onset, axis=1)
    )
    first_deformation = _first_true(
        np.any(result.diagnostics.support_surface_displacement_m > 1.0e-6, axis=(1, 2))
    )
    fall_raw = result.metadata["first_fall_sample"]
    fall = None if fall_raw is None else int(fall_raw)
    boundary = _boundary_expected_sample(
        result, float(specification["patch_start_x_m"])
    )
    outcome = classify_scenario_outcome(result, specification)
    return {
        "run_id": str(specification["id"]),
        "source_terrain": str(specification["source_terrain"]),
        "target_terrain": target,
        "speed_mps": float(specification["speed_mps"]),
        "patch_start_x_m": float(specification["patch_start_x_m"]),
        "patch_width_m": float(specification["patch_width_m"]),
        "sink_pattern": str(specification["sink_pattern"]),
        "sink_severity": str(specification["sink_severity"]),
        "support_pattern": str(specification["support_pattern"]),
        "intended_role": str(specification["intended_role"]),
        "t_boundary_expected_ms": _time_ms(result, boundary),
        "first_target_contact_ms": _time_ms(result, first_contact),
        "last_target_contact_ms": _time_ms(result, last_contact),
        "first_target_touchdown_ms": _time_ms(result, first_touchdown),
        "physical_slip_onset_ms": _time_ms(result, first_slip),
        "physical_sink_onset_ms": _time_ms(result, first_sink),
        "first_support_deformation_ms": _time_ms(result, first_deformation),
        "max_support_deformation_m": float(
            np.max(result.diagnostics.support_surface_max_displacement_m)
        ),
        "crossed_target_patch": _crossed_patch(
            result,
            float(specification["patch_start_x_m"])
            + float(specification["patch_width_m"]),
        ),
        "observed_fall": fall is not None,
        "fall_time_ms": _time_ms(result, fall),
        "pretransition_fall": bool(
            fall is not None and (first_contact is None or fall < first_contact)
        ),
        "valid_scenario_class": outcome,
        "finite_simulation": bool(
            np.all(np.isfinite(result.runtime.pelvis_imu))
            and result.metadata["actual_samples"] == result.metadata["expected_samples"]
        ),
    }


def _simulation_config(
    base: SimulationConfig,
    specification: Mapping[str, object],
    policy_path: Path,
    duration_s: float,
) -> SimulationConfig:
    config = replace(
        base,
        duration_s=duration_s,
        command_speed_mps=float(specification["speed_mps"]),
        policy_path=policy_path,
        source_terrain=str(specification["source_terrain"]),
        terrain=str(specification["target_terrain"]),
        slip_pattern=str(specification["slip_pattern"]),
        sink_pattern=str(specification["sink_pattern"]),
        sink_severity=str(specification["sink_severity"]),
        sink_support_pattern=str(specification["support_pattern"]),
        patch_start_x_m=float(specification["patch_start_x_m"]),
        patch_width_m=float(specification["patch_width_m"]),
        headless=True,
    )
    config.validate()
    return config


def construct_matched_reference(
    specification: Mapping[str, object],
    reference_patch_start_x_m: float,
    reference_patch_width_m: float,
) -> dict[str, object]:
    """Keep the same patch scene/mechanics and move only B beyond the run."""
    reference = dict(specification)
    reference["id"] = f"{specification['id']}_reference"
    reference["patch_start_x_m"] = float(reference_patch_start_x_m)
    reference["patch_width_m"] = float(reference_patch_width_m)
    return reference


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right), initial=0.0))


def compare_prefix_pair(
    transition: SimulationResult,
    reference: SimulationResult,
    specification: Mapping[str, object],
    tolerances: Mapping[str, object],
    minimum_safety_margin_physics_steps: int = 1,
) -> dict[str, object]:
    """Compare all robot/runtime/controller channels before first B contact."""
    if transition.state_trace is None or reference.state_trace is None:
        raise ValueError("prefix comparison requires captured state traces")
    first_contact = _first_true(
        np.any(
            target_contact_mask(
                transition, str(specification["target_terrain"])
            ),
            axis=1,
        )
    )
    if first_contact is None or first_contact < 2:
        return {
            "pair_id": str(specification["id"]),
            "passed": False,
            "verdict": "TRANSITION_PREFIX_PARITY_FAIL",
            "reason": "missing_or_too_early_target_contact",
        }
    end = first_contact
    left = transition.state_trace
    right = reference.state_trace
    max_diffs = {
        "qpos": _max_abs(left.robot_qpos[:end], right.robot_qpos[:end]),
        "qvel": _max_abs(left.robot_qvel[:end], right.robot_qvel[:end]),
        "imu": _max_abs(
            transition.runtime.pelvis_imu[:end], reference.runtime.pelvis_imu[:end]
        ),
        "fsr": _max_abs(
            transition.runtime.foot_fsr[:end], reference.runtime.foot_fsr[:end]
        ),
        "controller_observation": _max_abs(
            left.controller_observation[:end], right.controller_observation[:end]
        ),
        "controller_action": _max_abs(
            left.controller_action[:end], right.controller_action[:end]
        ),
        "pelvis_pose": _max_abs(left.pelvis_pose[:end], right.pelvis_pose[:end]),
        "com": _max_abs(left.whole_body_com[:end], right.whole_body_com[:end]),
    }
    contact_mismatches = int(
        np.count_nonzero(
            transition.diagnostics.physical_contact[:end]
            != reference.diagnostics.physical_contact[:end]
        )
    )
    policy_mismatches = int(
        np.count_nonzero(left.policy_updated[:end] != right.policy_updated[:end])
    )
    target_before_end_in_reference = bool(
        np.any(
            target_contact_mask(reference, str(specification["target_terrain"]))[
                :end
            ]
        )
    )
    dynamic_support_before_target = bool(
        np.any(transition.diagnostics.support_surface_cell_contact[:end])
    )
    transition_fall = transition.metadata["first_fall_sample"]
    reference_fall = reference.metadata["first_fall_sample"]
    pretarget_fall = bool(
        (transition_fall is not None and int(transition_fall) < first_contact)
        or (reference_fall is not None and int(reference_fall) < first_contact)
    )
    limits = {
        "qpos": float(tolerances["qpos_abs"]),
        "qvel": float(tolerances["qvel_abs"]),
        "imu": float(tolerances["imu_abs"]),
        "fsr": float(tolerances["fsr_abs"]),
        "controller_observation": float(
            tolerances["controller_observation_abs"]
        ),
        "controller_action": float(tolerances["controller_action_abs"]),
        "pelvis_pose": float(tolerances["pelvis_pose_abs"]),
        "com": float(tolerances["com_abs"]),
    }
    safety_margin_us = int(transition.runtime.timestamp_us[first_contact]) - int(
        transition.runtime.timestamp_us[end - 1]
    )
    safety_margin_physics_steps = int(
        round(safety_margin_us / (PHYSICS_TIMESTEP_S * 1_000_000.0))
    )
    passed = bool(
        all(max_diffs[name] <= limit for name, limit in limits.items())
        and contact_mismatches <= int(tolerances["contact_mismatch_count"])
        and policy_mismatches
        <= int(tolerances["policy_update_mismatch_count"])
        and not target_before_end_in_reference
        and not dynamic_support_before_target
        and not pretarget_fall
        and safety_margin_physics_steps >= minimum_safety_margin_physics_steps
    )
    return {
        "pair_id": str(specification["id"]),
        "source_terrain": str(specification["source_terrain"]),
        "target_terrain": str(specification["target_terrain"]),
        "comparison_end_sample": end - 1,
        "comparison_end_time_ms": _time_ms(transition, end - 1),
        "first_target_contact_ms": _time_ms(transition, first_contact),
        "safety_margin_physics_steps": safety_margin_physics_steps,
        "max_qpos_abs_diff": max_diffs["qpos"],
        "max_qvel_abs_diff": max_diffs["qvel"],
        "max_imu_abs_diff": max_diffs["imu"],
        "max_fsr_abs_diff": max_diffs["fsr"],
        "max_controller_observation_abs_diff": max_diffs[
            "controller_observation"
        ],
        "max_controller_action_abs_diff": max_diffs["controller_action"],
        "max_pelvis_pose_abs_diff": max_diffs["pelvis_pose"],
        "max_com_abs_diff": max_diffs["com"],
        "contact_mismatch_count": contact_mismatches,
        "policy_update_mismatch_count": policy_mismatches,
        "reference_target_contact_before_comparison_end": target_before_end_in_reference,
        "pretarget_dynamic_support_contact": dynamic_support_before_target,
        "pretarget_fall": pretarget_fall,
        "passed": passed,
        "verdict": (
            "TRANSITION_PREFIX_PARITY_PASS"
            if passed
            else "TRANSITION_PREFIX_PARITY_FAIL"
        ),
    }


def geometry_contact_audit(
    specification: Mapping[str, object],
) -> dict[str, object]:
    model, ground_ids = load_g1_model(
        str(specification["target_terrain"]),
        str(specification["sink_pattern"]),
        str(specification["sink_severity"]),
        str(specification["slip_pattern"]),
        float(specification["patch_start_x_m"]),
        float(specification["patch_width_m"]),
        str(specification["support_pattern"]),
        str(specification["source_terrain"]),
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    target_ids = (
        low_friction_patch_geom_ids(model, str(specification["slip_pattern"]))
        if specification["target_terrain"] == "ice"
        else soft_sink_geom_ids(
            model,
            str(specification["sink_pattern"]),
            str(specification["support_pattern"]),
        )
    )
    start = float(specification["patch_start_x_m"])
    end = start + float(specification["patch_width_m"])
    extents = []
    for geom_id in sorted(target_ids):
        extents.append(
            {
                "name": model.geom(geom_id).name,
                "minimum_x_m": float(data.geom_xpos[geom_id, 0] - model.geom_size[geom_id, 0]),
                "maximum_x_m": float(data.geom_xpos[geom_id, 0] + model.geom_size[geom_id, 0]),
                "top_z_m": float(data.geom_xpos[geom_id, 2] + model.geom_size[geom_id, 2]),
            }
        )
    pre_id = model.geom("terrain_transition_pre").id
    post_id = model.geom("terrain_transition_post").id
    pre_end = float(data.geom_xpos[pre_id, 0] + model.geom_size[pre_id, 0])
    post_start = float(data.geom_xpos[post_id, 0] - model.geom_size[post_id, 0])
    target_bounded = bool(
        extents
        and all(
            item["minimum_x_m"] >= start - 1.0e-12
            and item["maximum_x_m"] <= end + 1.0e-12
            for item in extents
        )
    )
    same_height = bool(
        all(abs(item["top_z_m"]) <= 1.0e-12 for item in extents)
        and abs(float(data.geom_xpos[pre_id, 2] + model.geom_size[pre_id, 2]))
        <= 1.0e-12
        and abs(float(data.geom_xpos[post_id, 2] + model.geom_size[post_id, 2]))
        <= 1.0e-12
    )
    complete_ground = bool(
        np.isclose(pre_end, start, atol=1.0e-12, rtol=0.0)
        and np.isclose(post_start, end, atol=1.0e-12, rtol=0.0)
        and pre_id in ground_ids
        and post_id in ground_ids
    )
    passed = target_bounded and same_height and complete_ground
    return {
        "pair_id": str(specification["id"]),
        "target_geom_extents": extents,
        "pre_ground_end_x_m": pre_end,
        "post_ground_start_x_m": post_start,
        "target_does_not_extend_before_boundary": target_bounded,
        "same_height_no_step": same_height,
        "complete_ground_no_hole": complete_ground,
        "passed": passed,
    }


def _condition_signature(specification: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(specification[field] for field in SIGNATURE_FIELDS)


def _assert_unique_and_disjoint(document: Mapping[str, object]) -> None:
    calibration = [
        *document["calibration"]["ice"],
        *document["calibration"]["sand"],
    ]
    validation = list(document["fresh_validation"]["runs"])
    marble = list(document["marble_robustness"]["runs"])
    all_specs = [
        *document["prefix_parity"]["matched_pairs"],
        *calibration,
        *validation,
        *marble,
    ]
    ids = [str(item["id"]) for item in all_specs]
    if len(ids) != len(set(ids)):
        raise ValueError("all transition run IDs must be unique")
    calibration_signatures = {_condition_signature(item) for item in calibration}
    validation_signatures = {_condition_signature(item) for item in validation}
    if calibration_signatures & validation_signatures:
        raise ValueError("calibration and fresh validation conditions overlap")
    if len(validation_signatures) != len(validation):
        raise ValueError("fresh validation conditions must all be physically distinct")
    domains = document["frozen_operating_points"]["validation_domains"]
    for item in validation:
        group = str(item["frozen_group"])
        domain = domains[group]
        for field in (
            "speed_mps",
            "patch_start_x_m",
            "slip_pattern",
            "sink_pattern",
            "sink_severity",
            "support_pattern",
        ):
            allowed = domain[field]
            allowed_values = allowed if isinstance(allowed, list) else [allowed]
            if item[field] not in allowed_values:
                raise ValueError(
                    f"fresh validation run {item['id']} escapes frozen {group} {field}"
                )
        width_min, width_max = domain["patch_width_m"]
        if not float(width_min) <= float(item["patch_width_m"]) <= float(width_max):
            raise ValueError(
                f"fresh validation run {item['id']} escapes frozen {group} width"
            )
    calibration_by_id = {str(item["id"]): item for item in calibration}
    selected = document["frozen_operating_points"]["selected_calibration_run_ids"]
    selected_ids = {str(value) for values in selected.values() for value in values}
    if not selected_ids <= calibration_by_id.keys():
        raise ValueError("frozen operating point references unknown calibration run")
    for item in marble:
        source = calibration_by_id[str(item["frozen_source_run_id"])]
        if any(item[field] != source[field] for field in SIGNATURE_FIELDS):
            raise ValueError("Marble robustness changed a frozen B-side condition")
        if item["source_terrain"] != "marble":
            raise ValueError("Marble robustness must change only source terrain")


def _physics_freeze_audit(document: Mapping[str, object]) -> dict[str, object]:
    frozen = document["common"]["frozen_physics"]
    ice_match = bool(
        np.array_equal(
            np.asarray(frozen["ice_friction"], dtype=np.float64),
            np.asarray(TERRAIN_PROFILES["ice"].friction, dtype=np.float64),
        )
    )
    sand_match = True
    for name, expected in frozen["sand_deformable_support"].items():
        actual = DEFORMABLE_SUPPORT_PROFILES[str(name)]
        sand_match &= bool(
            np.isclose(expected["travel_m"], actual.travel_m)
            and np.isclose(expected["stiffness_n_per_m"], actual.stiffness_n_per_m)
            and np.isclose(expected["damping_n_s_per_m"], actual.damping_n_s_per_m)
        )
    return {
        "ice_friction_unchanged": ice_match,
        "sand_mechanics_unchanged": sand_match,
        "passed": ice_match and sand_match,
    }


def _fusion_regression() -> dict[str, object]:
    expected = {
        TerrainState.ICE: HazardState.SLIP_RISK,
        TerrainState.SAND: HazardState.SINK_RISK,
        TerrainState.CONCRETE: HazardState.GENERIC_INSTABILITY,
        TerrainState.MARBLE: HazardState.GENERIC_INSTABILITY,
        TerrainState.UNKNOWN: HazardState.GENERIC_INSTABILITY,
    }
    rows = []
    passed = True
    for terrain, hazard in expected.items():
        state = ParallelRuntimeState().update_terrain(
            terrain, 0, valid=terrain != TerrainState.UNKNOWN
        ).update_stability(StabilityState.UNSTABLE, 1)
        ok = state.hazard_state == hazard and state.recovery_required
        passed &= ok
        rows.append(
            {
                "terrain": terrain.name,
                "hazard": state.hazard_state.name,
                "recovery_required": state.recovery_required,
                "passed": ok,
            }
        )
    stable = ParallelRuntimeState().update_terrain(
        TerrainState.ICE, 0
    ).update_stability(StabilityState.STABLE, 1)
    stable_ok = (
        stable.hazard_state == HazardState.NORMAL and not stable.recovery_required
    )
    passed &= stable_ok
    rows.append(
        {
            "terrain": "ICE",
            "stability": "STABLE",
            "hazard": stable.hazard_state.name,
            "recovery_required": stable.recovery_required,
            "passed": stable_ok,
        }
    )
    return {"passed": bool(passed), "rows": rows, "logic_changed": False}


def _run_specs(
    base: SimulationConfig,
    specifications: Sequence[Mapping[str, object]],
    policy_path: Path,
    duration_s: float,
    progress: Callable[[str], None],
    label: str,
) -> tuple[list[dict[str, object]], dict[str, SimulationResult]]:
    rows = []
    simulations = {}
    for index, raw in enumerate(specifications, start=1):
        specification = dict(raw)
        specification["minimum_normal_prefix_ms"] = 500
        specification["minimum_post_contact_ms"] = 500
        run_id = str(specification["id"])
        result = run_simulation(
            _simulation_config(base, specification, policy_path, duration_s)
        )
        row = scenario_timing_row(result, specification)
        rows.append(row)
        simulations[run_id] = result
        progress(
            f"{label} {index}/{len(specifications)} {run_id}: "
            f"{row['valid_scenario_class']}"
        )
    return rows, simulations


def _outcome_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {
        "ice_stable": 0,
        "ice_fall": 0,
        "sand_stable": 0,
        "sand_fall": 0,
        "pretransition_fall": 0,
        "invalid": 0,
    }
    for row in rows:
        outcome = str(row["valid_scenario_class"])
        target = str(row["target_terrain"])
        if outcome == VALID_STABLE:
            counts[f"{target}_stable"] += 1
        elif outcome == VALID_FALL:
            counts[f"{target}_fall"] += 1
        else:
            counts["invalid"] += 1
        counts["pretransition_fall"] += int(bool(row["pretransition_fall"]))
    return counts


def _minimums_pass(
    counts: Mapping[str, int], minimums: Mapping[str, object]
) -> bool:
    return bool(
        all(counts[name] >= int(value) for name, value in minimums.items())
        and counts["pretransition_fall"] == 0
        and counts["invalid"] == 0
    )


def run_transition_scenario_calibration(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Run prefix gate, calibration, frozen validation, and robustness."""
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if document["experiment"]["id"] != "TRANSITION_SCENARIO_CALIBRATION":
        raise ValueError("unsupported transition calibration experiment")
    _assert_unique_and_disjoint(document)
    if PHYSICS_TIMESTEP_S != float(document["common"]["physics_timestep_s"]):
        raise ValueError("experiment physics timestep differs from canonical value")
    if SENSOR_RATE_HZ != int(document["common"]["sensor_rate_hz"]):
        raise ValueError("experiment sensor rate differs from canonical value")
    frozen_hash_before = _canonical_hash(document["frozen_operating_points"])
    artifact_path = (repository_root / document["artifacts"]["path"]).resolve()
    artifact_path.relative_to(repository_root)
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite experiment artifacts: {artifact_path}")
    artifact_path.mkdir(parents=True, exist_ok=True)
    base = load_simulation_config(
        (repository_root / document["source"]["simulator_config"]).resolve()
    )
    policy_path = (repository_root / document["source"]["policy_path"]).resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"verified G1 policy is unavailable: {policy_path}")
    duration = float(document["common"]["duration_s"])
    physics_audit = _physics_freeze_audit(document)
    fusion = _fusion_regression()

    prefix_rows = []
    geometry_rows = []
    pairs = document["prefix_parity"]["matched_pairs"]
    for index, specification in enumerate(pairs, start=1):
        geometry = geometry_contact_audit(specification)
        geometry_rows.append(geometry)
        transition = run_simulation(
            _simulation_config(base, specification, policy_path, duration),
            capture_state_trace=True,
        )
        reference_spec = construct_matched_reference(
            specification,
            float(document["common"]["reference_patch_start_x_m"]),
            float(document["common"]["reference_patch_width_m"]),
        )
        reference = run_simulation(
            _simulation_config(base, reference_spec, policy_path, duration),
            capture_state_trace=True,
        )
        row = compare_prefix_pair(
            transition,
            reference,
            specification,
            document["prefix_parity"]["tolerances"],
            int(document["prefix_parity"]["safety_margin_physics_steps"]),
        )
        prefix_rows.append(row)
        progress(
            f"prefix {index}/{len(pairs)} {specification['id']}: {row['verdict']}"
        )
    prefix_passed = bool(
        physics_audit["passed"]
        and all(row["passed"] for row in geometry_rows)
        and all(row["passed"] for row in prefix_rows)
    )
    metrics: dict[str, object] = {
        "experiment": document["experiment"],
        "terrain_runtime": document["source"]["terrain_runtime"],
        "physics_freeze_audit": physics_audit,
        "prefix_parity": {
            "passed": prefix_passed,
            "verdict": (
                "TRANSITION_PREFIX_PARITY_PASS"
                if prefix_passed
                else "TRANSITION_PREFIX_PARITY_FAIL"
            ),
            "pairs": prefix_rows,
            "geometry_contact_audit": geometry_rows,
        },
        "fusion_regression": fusion,
        "calibration": {"performed": False, "runs": []},
        "fresh_validation": {"performed": False, "runs": []},
        "marble_robustness": {"performed": False, "runs": []},
        "viewer": {
            "implementation": document["viewer"]["implementation"],
            "representative_run_ids": document["viewer"]["representative_run_ids"],
            "physics_mutation": False,
            "visual_confirmation": "PENDING_MANUAL_REPLAY",
        },
    }
    if not prefix_passed:
        metrics["verdict"] = document["acceptance"]["verdicts"]["fail"]
        _write_json(artifact_path / "results.json", metrics)
        return artifact_path, metrics

    calibration_specs = [
        *document["calibration"]["ice"],
        *document["calibration"]["sand"],
    ]
    calibration_rows, _ = _run_specs(
        base, calibration_specs, policy_path, duration, progress, "calibration"
    )
    calibration_by_id = {str(row["run_id"]): row for row in calibration_rows}
    selected = document["frozen_operating_points"]["selected_calibration_run_ids"]
    selection_passed = True
    for group, run_ids in selected.items():
        expected = VALID_STABLE if str(group).endswith("stable") else VALID_FALL
        selection_passed &= all(
            calibration_by_id[str(run_id)]["valid_scenario_class"] == expected
            for run_id in run_ids
        )
    calibration_counts = _outcome_counts(calibration_rows)
    calibration_passed = bool(
        selection_passed
        and calibration_counts["ice_stable"] >= 2
        and calibration_counts["ice_fall"] >= 2
        and calibration_counts["sand_stable"] >= 2
        and calibration_counts["sand_fall"] >= 2
    )
    frozen_hash_after_calibration = _canonical_hash(
        document["frozen_operating_points"]
    )
    freeze_immutable = frozen_hash_after_calibration == frozen_hash_before
    metrics["calibration"] = {
        "performed": True,
        "runs": calibration_rows,
        "counts": calibration_counts,
        "selected_calibration_run_ids": selected,
        "selection_passed": bool(selection_passed),
        "passed": calibration_passed,
    }
    metrics["frozen_operating_points"] = {
        "status": document["frozen_operating_points"]["status"],
        "sha256_before_calibration": frozen_hash_before,
        "sha256_after_calibration": frozen_hash_after_calibration,
        "immutable": freeze_immutable,
    }
    if not calibration_passed or not freeze_immutable:
        metrics["verdict"] = document["acceptance"]["verdicts"]["fail"]
        _write_json(artifact_path / "results.json", metrics)
        return artifact_path, metrics

    validation_rows, validation_simulations = _run_specs(
        base,
        document["fresh_validation"]["runs"],
        policy_path,
        duration,
        progress,
        "fresh validation",
    )
    validation_counts = _outcome_counts(validation_rows)
    concrete_passed = _minimums_pass(
        validation_counts, document["acceptance"]["fresh_concrete_minimum"]
    )
    frozen_hash_after_validation = _canonical_hash(
        document["frozen_operating_points"]
    )
    no_post_validation_retuning = frozen_hash_after_validation == frozen_hash_before
    metrics["fresh_validation"] = {
        "performed": True,
        "runs": validation_rows,
        "counts": validation_counts,
        "passed": concrete_passed,
        "calibration_validation_disjoint": True,
    }
    metrics["frozen_operating_points"].update(
        {
            "sha256_after_validation": frozen_hash_after_validation,
            "no_post_validation_retuning": no_post_validation_retuning,
        }
    )
    if not concrete_passed or not no_post_validation_retuning:
        metrics["verdict"] = document["acceptance"]["verdicts"]["fail"]
        _write_json(artifact_path / "results.json", metrics)
        return artifact_path, metrics

    marble_rows, marble_simulations = _run_specs(
        base,
        document["marble_robustness"]["runs"],
        policy_path,
        duration,
        progress,
        "Marble robustness",
    )
    marble_counts = _outcome_counts(marble_rows)
    marble_passed = _minimums_pass(
        marble_counts, document["acceptance"]["marble_minimum"]
    )
    metrics["marble_robustness"] = {
        "performed": True,
        "runs": marble_rows,
        "counts": marble_counts,
        "passed": marble_passed,
        "recalibration_performed": False,
    }
    # Keep representative simulations alive until the end so the generated
    # result records that each requested ID was actually executed.
    executed_ids = set(validation_simulations) | set(marble_simulations)
    metrics["viewer"]["representative_runs_executed"] = all(
        str(run_id) in executed_ids
        for run_id in document["viewer"]["representative_run_ids"]
    )
    metrics["verdict"] = (
        document["acceptance"]["verdicts"]["pass"]
        if marble_passed
        else document["acceptance"]["verdicts"]["partial"]
    )
    _write_json(artifact_path / "results.json", metrics)
    return artifact_path, metrics
