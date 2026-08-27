"""Bounded one-run-per-NPZ Hazard pilot dataset collector."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from fastreflex.simulation.g1 import (
    CONTROL_PERIOD_S,
    IMU_CHANNELS,
    PHYSICS_TIMESTEP_S,
    POLICY_PERIOD_S,
    SENSOR_RATE_HZ,
    TESTED_POLICY_SHA256,
    RuntimeTrace,
    SimulationConfig,
    SimulationResult,
    load_simulation_config,
    run_simulation,
    sha256_file,
)
from fastreflex.simulation.sensors import FSR_CHANNELS, FSR_UNIT


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUN_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
OUTCOMES = ("BENIGN", "SLIP", "SINK", "DUAL", "INVALID")
SIDES = ("left", "right")
MANIFEST_FIELDS = (
    "run_id",
    "file",
    "scenario_family",
    "intended_role",
    "terrain",
    "speed_mps",
    "patch_start_x",
    "patch_width_m",
    "sink_side",
    "sink_severity",
    "observed_outcome",
    "sample_count",
    "valid_sample_count",
    "drop_count",
    "first_patch_contact_ms",
    "any_slip_onset_ms",
    "sink_physical_onset_ms",
    "sink_degradation_onset_ms",
    "first_censor_ms",
    "censor_reason",
    "first_contact_foot",
    "first_contact_policy_phase",
    "slip_foot_summary",
    "has_slip",
    "has_sink_hazard",
    "has_dual_hazard",
    "policy_sha256",
    "run_file_sha256",
)
OBSERVABILITY_MANIFEST_FIELDS = (
    *MANIFEST_FIELDS[:11],
    "sink_support_pattern",
    "sink_pattern",
    "split",
    "condition_signature",
    *MANIFEST_FIELDS[11:18],
    "deformable_sink_onset_ms",
    "mechanical_recovery_ms",
    *MANIFEST_FIELDS[18:25],
    "has_deformable_sink",
    *MANIFEST_FIELDS[25:],
)
BASE_SERIES_SHAPES = {
    "sequence": (),
    "timestamp_us": (),
    "pelvis_imu": (6,),
    "sample_valid": (),
    "channel_valid": (6,),
    "hazard_class_id": (),
    "training_eligible": (),
    "physical_contact": (2,),
    "touchdown": (2,),
    "loaded_contact": (2,),
    "low_friction_patch_contact": (2,),
    "tangential_anchor_drift_m": (2,),
    "tangential_velocity_mps": (2,),
    "established_slip_active": (2,),
    "established_slip_onset": (2,),
    "any_slip_active": (),
    "any_slip_onset": (),
    "contact_penetration_m": (2,),
    "loaded_penetration_change_m": (2,),
    "sink_physical_active": (2,),
    "sink_physical_onset": (2,),
    "sink_physical_after_patch_onset": (2,),
    "soft_patch_contact": (2,),
    "sink_degradation_active": (),
    "sink_degradation_onset": (),
    "sink_hazard_active": (),
    "sink_hazard_onset": (),
    "pelvis_tilt_rad": (),
    "pelvis_world_z_m": (),
    "pelvis_angular_velocity_rad_s": (3,),
    "pelvis_linear_velocity_m_s": (3,),
    "pelvis_forward_velocity_m_s": (),
    "forward_velocity_error_m_s": (),
    "pre_fall_valid": (),
    "fall_active": (),
    "dual_hazard_active": (),
}
SENSOR_SERIES_SHAPES = {
    "foot_fsr": (8,),
    "fsr_valid": (8,),
}
DEFORMABLE_SERIES_SHAPES = {
    "contact_episode_id": (2,),
    "support_surface_displacement_m": (2, 4),
    "support_surface_vertical_velocity_m_s": (2, 4),
    "support_surface_cell_contact": (2, 4),
    "support_surface_spread_m": (2,),
    "deformable_patch_episode_active": (2,),
    "deformable_sink_active": (2,),
    "deformable_sink_onset": (2,),
}
# Backward-compatible name for the frozen v1 IMU-only schema.
SERIES_SHAPES = BASE_SERIES_SHAPES
SCALAR_SHAPES = {
    "first_patch_contact_sample_per_foot": (2,),
    "first_slip_onset_sample_per_foot": (2,),
    "first_any_slip_onset_sample": (),
    "first_sink_physical_onset_sample_per_foot": (2,),
    "first_sink_degradation_onset_sample": (),
    "first_censor_sample": (),
    "censor_reason": (),
    "hazardous_sink_episode": (),
}
DEFORMABLE_SCALAR_SHAPES = {
    "first_deformable_sink_onset_sample_per_foot": (2,),
    "first_deformable_sink_onset_sample": (),
}


@dataclass(frozen=True)
class RunSpec:
    """One deterministic physical condition, independent of its outcome."""

    run_id: str
    scenario_family: str
    intended_role: str
    terrain: str
    speed_mps: float
    patch_start_x_m: float | None
    sink_side: str | None
    sink_severity: str | None
    sink_support_pattern: str
    split: str | None


@dataclass(frozen=True)
class CollectionConfig:
    """Validated experiment-level collection configuration."""

    config_path: Path
    dataset_id: str
    schema_version: str
    simulator_config_path: Path
    dataset_config_path: Path
    baseline_dataset_path: Path | None
    baseline_manifest_sha256: str | None
    require_clean_worktree: bool
    output_root: Path
    duration_s: float
    patch_width_m: float
    simulator_deterministic: bool
    policy_sha256: str
    minimum_post_d0_evaluation_ms: int
    recovery_threshold_m: float
    recovery_persistence_ms: int
    split: dict[str, tuple[str, ...]] | None
    runs: tuple[RunSpec, ...]


def _resolve_repository_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a repository-relative path")
    path = (REPOSITORY_ROOT / value).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{field} must remain inside the repository") from exc
    return path


def load_collection_config(path: Path) -> CollectionConfig:
    """Load and validate the bounded pilot matrix without executing it."""
    config_path = path.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("collection config must be a YAML mapping")
    try:
        experiment = document["experiment"]
        source = document["source"]
        output = document["output"]
        common = document["common"]
        raw_runs = document["runs"]
        dataset_id = str(experiment["dataset_id"])
        schema_version = str(experiment["schema_version"])
        simulator_config = _resolve_repository_path(
            source["simulator_config"], "source.simulator_config"
        )
        dataset_config = _resolve_repository_path(
            source["dataset_config"], "source.dataset_config"
        )
        baseline_dataset_path = (
            None
            if source.get("baseline_dataset") is None
            else _resolve_repository_path(
                source["baseline_dataset"], "source.baseline_dataset"
            )
        )
        baseline_manifest_sha256 = (
            None
            if source.get("baseline_manifest_sha256") is None
            else str(source["baseline_manifest_sha256"])
        )
        output_root = _resolve_repository_path(output["root"], "output.root")
        duration_s = float(common["duration_s"])
        patch_width_m = float(common["patch_width_m"])
        deterministic = bool(common["simulator_deterministic"])
        policy_sha256 = str(common["policy_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("collection config is missing required fields") from exc
    if not RUN_ID_PATTERN.fullmatch(dataset_id):
        raise ValueError("dataset_id must be lowercase snake_case")
    if schema_version not in {
        "hazard_dataset_contract_v1",
        "hazard_dataset_contract_v2",
        "hazard_dataset_contract_v3",
    }:
        raise ValueError("unsupported Hazard dataset schema version")
    if schema_version == "hazard_dataset_contract_v2" and baseline_dataset_path is None:
        raise ValueError("sensor dataset requires source.baseline_dataset for parity")
    if schema_version == "hazard_dataset_contract_v2" and not re.fullmatch(
        r"[0-9a-f]{64}", baseline_manifest_sha256 or ""
    ):
        raise ValueError("sensor dataset requires a pinned baseline manifest SHA-256")
    observability_schema = schema_version == "hazard_dataset_contract_v3"
    if observability_schema and baseline_dataset_path is not None:
        raise ValueError("observability dataset must not claim v2 observer parity")
    if duration_s <= 0.0 or patch_width_m <= 0.0:
        raise ValueError("duration and patch width must be positive")
    if not deterministic:
        raise ValueError("this pilot config must declare deterministic simulation")
    if not bool(output.get("fail_if_exists", False)):
        raise ValueError("pilot output must fail if the final dataset already exists")
    if policy_sha256 != TESTED_POLICY_SHA256:
        raise ValueError("pilot config policy SHA-256 is not the verified artifact")
    if not simulator_config.is_file() or not dataset_config.is_file():
        raise ValueError("referenced canonical config is missing")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("runs must be a non-empty list")

    runs: list[RunSpec] = []
    for raw in raw_runs:
        if not isinstance(raw, dict):
            raise ValueError("each run must be a mapping")
        try:
            run = RunSpec(
                run_id=str(raw["run_id"]),
                scenario_family=str(raw["scenario_family"]),
                intended_role=str(raw["intended_role"]),
                terrain=str(raw["terrain"]),
                speed_mps=float(raw["speed_mps"]),
                patch_start_x_m=(
                    None
                    if raw.get("patch_start_x") is None
                    else float(raw["patch_start_x"])
                ),
                sink_side=(
                    None if raw.get("sink_side") is None else str(raw["sink_side"])
                ),
                sink_severity=(
                    None
                    if raw.get("sink_severity") is None
                    else str(raw["sink_severity"])
                ),
                sink_support_pattern=str(
                    raw.get("sink_support_pattern", "balanced_soft")
                ),
                split=None if raw.get("split") is None else str(raw["split"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("run entry is missing required fields") from exc
        _validate_run_spec(run)
        runs.append(run)
    identifiers = [run.run_id for run in runs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("run_id values must be unique")
    split: dict[str, tuple[str, ...]] | None = None
    minimum_post_d0_evaluation_ms = 0
    recovery_threshold_m = 0.001
    recovery_persistence_ms = 20
    if observability_schema:
        try:
            raw_split = document["split"]
            labeling = document["labeling"]
            minimum_post_d0_evaluation_ms = int(
                labeling["minimum_post_d0_evaluation_ms"]
            )
            recovery_threshold_m = float(labeling["recovery_threshold_m"])
            recovery_persistence_ms = int(labeling["recovery_persistence_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("v3 observability labeling/split is incomplete") from exc
        split = {
            name: tuple(str(value) for value in raw_split[name])
            for name in ("train", "validation", "holdout")
        }
        _validate_observability_matrix(
            tuple(runs), split, patch_width_m
        )
        _validate_observability_contract(document, tuple(runs), split)
        if (
            minimum_post_d0_evaluation_ms <= 0
            or recovery_threshold_m < 0.0
            or recovery_persistence_ms <= 0
        ):
            raise ValueError("v3 observability labeling criteria are invalid")
    return CollectionConfig(
        config_path=config_path,
        dataset_id=dataset_id,
        schema_version=schema_version,
        simulator_config_path=simulator_config,
        dataset_config_path=dataset_config,
        baseline_dataset_path=baseline_dataset_path,
        baseline_manifest_sha256=baseline_manifest_sha256,
        require_clean_worktree=bool(source.get("require_clean_worktree", True)),
        output_root=output_root,
        duration_s=duration_s,
        patch_width_m=patch_width_m,
        simulator_deterministic=deterministic,
        policy_sha256=policy_sha256,
        minimum_post_d0_evaluation_ms=minimum_post_d0_evaluation_ms,
        recovery_threshold_m=recovery_threshold_m,
        recovery_persistence_ms=recovery_persistence_ms,
        split=split,
        runs=tuple(runs),
    )


def _validate_run_spec(run: RunSpec) -> None:
    if not RUN_ID_PATTERN.fullmatch(run.run_id):
        raise ValueError(f"invalid lowercase snake_case run_id: {run.run_id!r}")
    if run.scenario_family not in {"normal", "slip", "sink"}:
        raise ValueError(f"invalid scenario_family for {run.run_id}")
    if run.intended_role not in {"NORMAL", "SLIP", "SINK"}:
        raise ValueError(f"invalid intended_role for {run.run_id}")
    if run.terrain not in {"concrete", "marble", "ice", "sand"}:
        raise ValueError(f"invalid terrain for {run.run_id}")
    if not 0.1 <= run.speed_mps <= 0.5:
        raise ValueError(f"speed is outside the verified controller range: {run.run_id}")
    if run.sink_side is not None and run.sink_side not in SIDES:
        raise ValueError(f"invalid sink side for {run.run_id}")
    if run.sink_severity is not None and run.sink_severity not in {
        "mild",
        "moderate",
        "severe",
    }:
        raise ValueError(f"invalid sink severity for {run.run_id}")
    if run.sink_support_pattern not in {
        "balanced_soft",
        "balanced_deformable",
        "medial_deformable",
        "lateral_deformable",
        "localized_deformable",
    }:
        raise ValueError(f"invalid Sink support pattern for {run.run_id}")
    if run.split is not None and run.split not in {
        "train",
        "validation",
        "holdout",
    }:
        raise ValueError(f"invalid split for {run.run_id}")
    if run.scenario_family == "slip":
        if run.terrain != "ice" or run.patch_start_x_m is None:
            raise ValueError(f"Slip transition is incomplete: {run.run_id}")
        if run.sink_side is not None or run.intended_role != "SLIP":
            raise ValueError(f"Slip transition fields conflict: {run.run_id}")
    if run.sink_side is not None:
        if (
            run.terrain != "sand"
            or run.patch_start_x_m is None
            or run.sink_severity is None
        ):
            raise ValueError(f"Sink transition is incomplete: {run.run_id}")
    if run.scenario_family == "sink" and (
        run.sink_side is None or run.intended_role != "SINK"
    ):
        raise ValueError(f"Sink intended run is incomplete: {run.run_id}")
    if run.patch_start_x_m is not None and not np.isfinite(run.patch_start_x_m):
        raise ValueError(f"patch start is not finite: {run.run_id}")


def _condition_signature(run: RunSpec, patch_width_m: float) -> str:
    topology = (
        "rigid" if run.patch_start_x_m is None else run.sink_support_pattern
    )
    pattern = (
        "none"
        if run.patch_start_x_m is None
        else run.sink_support_pattern.removesuffix("_deformable")
    )
    values = (
        run.terrain,
        topology,
        run.sink_severity or "none",
        run.sink_side or "none",
        pattern,
        f"{run.speed_mps:.3f}",
        "none" if run.patch_start_x_m is None else f"{run.patch_start_x_m:.3f}",
        "none" if run.patch_start_x_m is None else f"{patch_width_m:.3f}",
    )
    return "|".join(values)


def _validate_observability_matrix(
    runs: tuple[RunSpec, ...],
    split: dict[str, tuple[str, ...]],
    patch_width_m: float,
) -> None:
    identifiers = {run.run_id for run in runs}
    assigned = [run_id for name in ("train", "validation", "holdout") for run_id in split[name]]
    if len(assigned) != len(set(assigned)) or set(assigned) != identifiers:
        raise ValueError("v3 split must cover every run exactly once")
    declared = {run.run_id: run.split for run in runs}
    for split_name, run_ids in split.items():
        if any(declared[run_id] != split_name for run_id in run_ids):
            raise ValueError("run-level split declaration and split block disagree")
    signatures = [_condition_signature(run, patch_width_m) for run in runs]
    if len(signatures) != len(set(signatures)):
        raise ValueError("duplicate physical condition signature")
    speed_contract = {
        "train": {0.12, 0.18, 0.24},
        "validation": {0.15, 0.27},
        "holdout": {0.21, 0.30},
    }
    for run in runs:
        assert run.split is not None
        if run.speed_mps not in speed_contract[run.split]:
            raise ValueError(f"split-specific speed violation: {run.run_id}")
    for split_name, expected_speeds in speed_contract.items():
        observed_speeds = {
            run.speed_mps for run in runs if run.split == split_name
        }
        if observed_speeds != expected_speeds:
            raise ValueError(f"{split_name} speed coverage violates frozen matrix")


def _validate_observability_contract(
    document: dict[str, object],
    runs: tuple[RunSpec, ...],
    split: dict[str, tuple[str, ...]],
) -> None:
    """Freeze the predeclared v3 mechanics, label, and matrix before simulation."""
    expected_label = {
        "physical_metric": "support_surface_spread_m",
        "spread_threshold_m": 0.010,
        "persistence_ms": 20,
        "d0": "deformable_support_first_physical_contact",
        "s1": "first_sustained_deformable_sink_active_sample",
        "positive_class_id": 2,
        "positive_d0_to_s1": "excluded",
        "positive_s1_to_contact_episode_end": "sink",
        "post_episode": "normal_reset",
        "no_s1_valid_run": "benign",
        "future_outcome_dependency": False,
        "fsr_or_imu_dependency": False,
    }
    labeling = document.get("labeling")
    if not isinstance(labeling, dict) or any(
        labeling.get(name) != value for name, value in expected_label.items()
    ):
        raise ValueError("v3 physical Sink label contract was modified")
    mechanics = document.get("mechanics")
    expected_mechanics = {
        "travel_mm": {"reference": 4, "mild": 20, "moderate": 40, "severe": 65},
        "stiffness_n_per_m": {
            "reference": 50000,
            "mild": 12000,
            "moderate": 7000,
            "severe": 4500,
        },
        "damping_n_s_per_m": {
            "reference": 1000,
            "mild": 490,
            "moderate": 374,
            "severe": 300,
        },
        "tuning_after_results": "prohibited",
    }
    if not isinstance(mechanics, dict) or any(
        mechanics.get(name) != value for name, value in expected_mechanics.items()
    ):
        raise ValueError("v3 deformable mechanics declaration was modified")

    rigid_count = sum(run.patch_start_x_m is None for run in runs)
    balanced_count = sum(
        run.sink_support_pattern == "balanced_deformable" for run in runs
    )
    moderate_uneven_count = sum(
        run.sink_severity == "moderate"
        and run.sink_support_pattern
        in {"medial_deformable", "lateral_deformable", "localized_deformable"}
        for run in runs
    )
    boundary_count = sum(
        run.sink_severity in {"mild", "severe"}
        and run.sink_support_pattern
        in {"medial_deformable", "lateral_deformable", "localized_deformable"}
        for run in runs
    )
    if not (
        120 <= len(runs) <= 140
        and 15 <= rigid_count <= 20
        and 35 <= balanced_count <= 45
        and 35 <= moderate_uneven_count <= 45
        and 20 <= boundary_count <= 30
    ):
        raise ValueError("v3 condition matrix composition violates frozen ranges")
    expected_positions = {
        "train": {0.30, 0.38},
        "validation": {0.34, 0.42},
        "holdout": {0.32, 0.40},
    }
    for split_name, run_ids in split.items():
        positions = {
            run.patch_start_x_m
            for run in runs
            if run.run_id in run_ids and run.patch_start_x_m is not None
        }
        if positions != expected_positions[split_name]:
            raise ValueError(f"{split_name} patch positions violate frozen matrix")


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if indices.size == 0 else int(indices[0])


def _first_true_per_foot(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            -1 if (first := _first_true(values[:, side])) is None else first
            for side in range(2)
        ],
        dtype=np.int64,
    )


def _sample_annotations(
    result: SimulationResult,
    intended_role: str | None,
    *,
    deformable_observability: bool = False,
    minimum_post_d0_evaluation_ms: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    runtime = result.runtime
    diagnostics = result.diagnostics
    sample_count = len(runtime.sequence)
    channel_valid = np.isfinite(runtime.pelvis_imu)
    sample_valid = np.all(channel_valid, axis=1)
    base_eligible = sample_valid & diagnostics.pre_fall_valid
    hazard_class_id = np.full(sample_count, -1, dtype=np.int8)
    hazard_class_id[base_eligible] = 0

    if deformable_observability:
        first_patch = _first_true(
            np.any(
                diagnostics.soft_patch_contact_onset
                & diagnostics.pre_fall_valid[:, None],
                axis=1,
            )
        )
        first_sink = _first_true(np.any(diagnostics.deformable_sink_onset, axis=1))
        first_censor = result.metadata["first_fall_sample"]
        dynamic_patch_expected = result.metadata["sink_support_pattern"].endswith(
            "_deformable"
        )
        insufficient_evaluation = bool(
            dynamic_patch_expected
            and first_sink is None
            and first_censor is not None
            and (
                first_patch is None
                or int(first_censor) - first_patch
                < minimum_post_d0_evaluation_ms
            )
        )
        invalid_run = bool(
            (first_censor is not None and int(first_censor) < SENSOR_RATE_HZ)
            or (dynamic_patch_expected and first_patch is None)
            or insufficient_evaluation
        )
        if invalid_run:
            hazard_class_id[:] = -1
        elif first_sink is not None:
            for side in range(2):
                for onset in np.flatnonzero(
                    diagnostics.deformable_sink_onset[:, side]
                ):
                    episode = int(diagnostics.contact_episode_id[onset, side])
                    if episode < 0:
                        raise ValueError("deformable s1 has no foot contact episode")
                    same_episode = diagnostics.contact_episode_id[:, side] == episode
                    patch_episode = (
                        same_episode
                        & diagnostics.deformable_patch_episode_active[:, side]
                    )
                    episode_patch_samples = np.flatnonzero(patch_episode)
                    if not episode_patch_samples.size:
                        raise ValueError("deformable s1 has no causal d0")
                    d0 = int(episode_patch_samples[0])
                    if d0 > onset:
                        raise ValueError("deformable d0 occurs after s1")
                    hazard_class_id[d0:onset] = -1
                    sink_episode = same_episode & (np.arange(sample_count) >= onset)
                    hazard_class_id[sink_episode & base_eligible] = 2
        return (
            sample_valid,
            channel_valid,
            hazard_class_id,
            hazard_class_id >= 0,
            np.zeros(sample_count, dtype=bool),
        )

    first_slip = _first_true(diagnostics.any_established_slip_onset)
    first_sink_hazard = _first_true(diagnostics.sink_hazard_onset)
    first_patch = _first_true(
        np.any(
            (
                diagnostics.low_friction_patch_contact_onset
                | diagnostics.soft_patch_contact_onset
            )
            & diagnostics.pre_fall_valid[:, None],
            axis=1,
        )
    )
    dual_run = bool(
        first_slip is not None
        and first_sink_hazard is not None
        and first_slip < first_sink_hazard
    )
    dual_hazard_active = np.zeros(sample_count, dtype=bool)
    first_censor = result.metadata["first_fall_sample"]
    transition_censored_without_outcome = bool(
        result.metadata.get("patch_start_x_m") is not None
        and first_censor is not None
        and (
            (intended_role == "SLIP" and first_slip is None)
            or (intended_role == "SINK" and first_sink_hazard is None)
        )
    )
    invalid_run = bool(
        (first_censor is not None and first_censor < SENSOR_RATE_HZ)
        or transition_censored_without_outcome
    )
    if invalid_run:
        hazard_class_id[:] = -1
    elif dual_run:
        slip_seen = np.logical_or.accumulate(diagnostics.any_established_slip)
        sink_seen = np.logical_or.accumulate(diagnostics.sink_hazard_active)
        dual_hazard_active = slip_seen & sink_seen
        hazard_class_id[:] = -1
    else:
        if first_slip is not None:
            if first_patch is not None and first_patch < first_slip:
                hazard_class_id[first_patch:first_slip] = -1
            slip_valid = base_eligible[first_slip:]
            hazard_class_id[first_slip:][slip_valid] = 1
        if first_sink_hazard is not None:
            sink_t1 = _first_true(
                np.any(diagnostics.sink_physical_after_patch_onset, axis=1)
            )
            unresolved_start = first_patch if first_patch is not None else sink_t1
            if unresolved_start is not None and unresolved_start < first_sink_hazard:
                hazard_class_id[unresolved_start:first_sink_hazard] = -1
            sink_valid = base_eligible[first_sink_hazard:]
            hazard_class_id[first_sink_hazard:][sink_valid] = 2
    training_eligible = hazard_class_id >= 0
    return (
        sample_valid,
        channel_valid,
        hazard_class_id,
        training_eligible,
        dual_hazard_active,
    )


def build_run_arrays(
    result: SimulationResult,
    intended_role: str | None = None,
    *,
    include_foot_fsr: bool = False,
    include_deformable_support: bool = False,
    minimum_post_d0_evaluation_ms: int = 0,
) -> dict[str, np.ndarray]:
    """Convert one simulation result into the documented raw NPZ schema."""
    if include_foot_fsr and result.runtime.foot_fsr is None:
        raise ValueError("sensor dataset requires an observed foot_fsr trace")
    diagnostics = result.diagnostics
    (
        sample_valid,
        channel_valid,
        hazard_class_id,
        training_eligible,
        dual_hazard_active,
    ) = _sample_annotations(
        result,
        intended_role,
        deformable_observability=include_deformable_support,
        minimum_post_d0_evaluation_ms=minimum_post_d0_evaluation_ms,
    )
    patch_onset = (
        diagnostics.low_friction_patch_contact_onset
        | diagnostics.soft_patch_contact_onset
    ) & diagnostics.pre_fall_valid[:, None]
    first_sink_per_foot = _first_true_per_foot(
        diagnostics.sink_physical_after_patch_onset
        if np.any(diagnostics.soft_patch_contact)
        else diagnostics.sink_physical_onset
    )
    first_censor = result.metadata["first_fall_sample"]
    censor_reason = "|".join(result.metadata["first_fall_reasons"])
    arrays = {
        "sequence": result.runtime.sequence.astype(np.int64, copy=False),
        "timestamp_us": result.runtime.timestamp_us.astype(np.int64, copy=False),
        "pelvis_imu": result.runtime.pelvis_imu.astype(np.float32, copy=False),
        "sample_valid": sample_valid,
        "channel_valid": channel_valid,
        "hazard_class_id": hazard_class_id,
        "training_eligible": training_eligible,
        "physical_contact": diagnostics.physical_contact,
        "touchdown": diagnostics.touchdown,
        "loaded_contact": diagnostics.loaded_contact,
        "low_friction_patch_contact": diagnostics.low_friction_patch_contact,
        "tangential_anchor_drift_m": diagnostics.tangential_anchor_drift_m,
        "tangential_velocity_mps": diagnostics.tangential_velocity_mps,
        "established_slip_active": diagnostics.established_slip,
        "established_slip_onset": diagnostics.established_slip_onset,
        "any_slip_active": diagnostics.any_established_slip,
        "any_slip_onset": diagnostics.any_established_slip_onset,
        "contact_penetration_m": diagnostics.contact_penetration_m,
        "loaded_penetration_change_m": diagnostics.loaded_penetration_change_m,
        "sink_physical_active": diagnostics.sink_physical_active,
        "sink_physical_onset": diagnostics.sink_physical_onset,
        "sink_physical_after_patch_onset": (
            diagnostics.sink_physical_after_patch_onset
        ),
        "soft_patch_contact": diagnostics.soft_patch_contact,
        "sink_degradation_active": diagnostics.sink_degradation_active,
        "sink_degradation_onset": diagnostics.sink_degradation_onset,
        "sink_hazard_active": diagnostics.sink_hazard_active,
        "sink_hazard_onset": diagnostics.sink_hazard_onset,
        "pelvis_tilt_rad": diagnostics.pelvis_tilt_rad,
        "pelvis_world_z_m": diagnostics.pelvis_world_z_m,
        "pelvis_angular_velocity_rad_s": (
            diagnostics.pelvis_angular_velocity_rad_s
        ),
        "pelvis_linear_velocity_m_s": diagnostics.pelvis_linear_velocity_m_s,
        "pelvis_forward_velocity_m_s": (
            diagnostics.pelvis_forward_velocity_m_s
        ),
        "forward_velocity_error_m_s": diagnostics.forward_velocity_error_m_s,
        "pre_fall_valid": diagnostics.pre_fall_valid,
        "fall_active": diagnostics.fall_active,
        "dual_hazard_active": dual_hazard_active,
        "first_patch_contact_sample_per_foot": _first_true_per_foot(patch_onset),
        "first_slip_onset_sample_per_foot": _first_true_per_foot(
            diagnostics.established_slip_onset
        ),
        "first_any_slip_onset_sample": np.asarray(
            -1
            if (first := _first_true(diagnostics.any_established_slip_onset)) is None
            else first,
            dtype=np.int64,
        ),
        "first_sink_physical_onset_sample_per_foot": first_sink_per_foot,
        "first_sink_degradation_onset_sample": np.asarray(
            -1
            if (first := _first_true(diagnostics.sink_hazard_onset)) is None
            else first,
            dtype=np.int64,
        ),
        "first_censor_sample": np.asarray(
            -1 if first_censor is None else first_censor,
            dtype=np.int64,
        ),
        "censor_reason": np.asarray(censor_reason, dtype=np.str_),
        "hazardous_sink_episode": np.asarray(
            np.any(diagnostics.sink_hazard_onset), dtype=bool
        ),
    }
    if include_foot_fsr:
        assert result.runtime.foot_fsr is not None
        arrays["foot_fsr"] = result.runtime.foot_fsr.astype(np.float32, copy=False)
        arrays["fsr_valid"] = np.isfinite(result.runtime.foot_fsr)
    if include_deformable_support:
        first_deformable_per_foot = _first_true_per_foot(
            diagnostics.deformable_sink_onset
        )
        first_deformable = _first_true(
            np.any(diagnostics.deformable_sink_onset, axis=1)
        )
        arrays.update(
            {
                "contact_episode_id": diagnostics.contact_episode_id,
                "support_surface_displacement_m": (
                    diagnostics.support_surface_displacement_m
                ),
                "support_surface_vertical_velocity_m_s": (
                    diagnostics.support_surface_vertical_velocity_m_s
                ),
                "support_surface_cell_contact": (
                    diagnostics.support_surface_cell_contact
                ),
                "support_surface_spread_m": (
                    diagnostics.support_surface_spread_m
                ),
                "deformable_patch_episode_active": (
                    diagnostics.deformable_patch_episode_active
                ),
                "deformable_sink_active": diagnostics.deformable_sink_active,
                "deformable_sink_onset": diagnostics.deformable_sink_onset,
                "first_deformable_sink_onset_sample_per_foot": (
                    first_deformable_per_foot
                ),
                "first_deformable_sink_onset_sample": np.asarray(
                    -1 if first_deformable is None else first_deformable,
                    dtype=np.int64,
                ),
            }
        )
    return {name: np.asarray(value) for name, value in arrays.items()}


def validate_run_arrays(
    arrays: dict[str, np.ndarray],
    expected_samples: int,
) -> None:
    """Fail closed on runtime corruption or diagnostic/schema misalignment."""
    sensor_schema = "foot_fsr" in arrays or "fsr_valid" in arrays
    deformable_schema = "support_surface_displacement_m" in arrays
    series_shapes = dict(BASE_SERIES_SHAPES)
    if sensor_schema:
        series_shapes.update(SENSOR_SERIES_SHAPES)
    scalar_shapes = dict(SCALAR_SHAPES)
    if deformable_schema:
        series_shapes.update(DEFORMABLE_SERIES_SHAPES)
        scalar_shapes.update(DEFORMABLE_SCALAR_SHAPES)
    expected_keys = set(series_shapes) | set(scalar_shapes)
    if set(arrays) != expected_keys:
        missing = sorted(expected_keys - set(arrays))
        extra = sorted(set(arrays) - expected_keys)
        raise ValueError(f"NPZ schema mismatch; missing={missing}, extra={extra}")
    for name, trailing_shape in series_shapes.items():
        expected_shape = (expected_samples, *trailing_shape)
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"{name} has shape {arrays[name].shape}, expected {expected_shape}"
            )
    for name, expected_shape in scalar_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"{name} has shape {arrays[name].shape}, expected {expected_shape}"
            )
    if arrays["sequence"].dtype != np.int64:
        raise ValueError("sequence must be int64")
    if arrays["timestamp_us"].dtype != np.int64:
        raise ValueError("timestamp_us must be int64")
    if arrays["pelvis_imu"].dtype != np.float32:
        raise ValueError("pelvis_imu must be float32")
    if sensor_schema and arrays["foot_fsr"].dtype != np.float32:
        raise ValueError("foot_fsr must be float32")
    if deformable_schema and arrays["support_surface_displacement_m"].dtype != np.float32:
        raise ValueError("support surface displacement must be float32")
    boolean_fields = (
        "sample_valid",
        "channel_valid",
        "training_eligible",
        "pre_fall_valid",
        "dual_hazard_active",
    ) + (("fsr_valid",) if sensor_schema else ()) + (
        (
            "support_surface_cell_contact",
            "deformable_patch_episode_active",
            "deformable_sink_active",
            "deformable_sink_onset",
        )
        if deformable_schema
        else ()
    )
    for name in boolean_fields:
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"{name} must be bool")
    if arrays["hazard_class_id"].dtype != np.int8:
        raise ValueError("hazard_class_id must be int8")
    for name in (
        "first_patch_contact_sample_per_foot",
        "first_slip_onset_sample_per_foot",
        "first_any_slip_onset_sample",
        "first_sink_physical_onset_sample_per_foot",
        "first_sink_degradation_onset_sample",
        "first_censor_sample",
        *(
            (
                "first_deformable_sink_onset_sample_per_foot",
                "first_deformable_sink_onset_sample",
            )
            if deformable_schema
            else ()
        ),
    ):
        if arrays[name].dtype != np.int64:
            raise ValueError(f"{name} must be int64")
    if arrays["hazardous_sink_episode"].dtype != np.bool_:
        raise ValueError("hazardous_sink_episode must be bool")
    if arrays["censor_reason"].dtype.kind != "U":
        raise ValueError("censor_reason must be a pickle-free Unicode scalar")
    if not np.array_equal(arrays["sequence"], np.arange(expected_samples)):
        raise ValueError("sequence is not contiguous from zero")
    expected_timestamp = (arrays["sequence"] + 1) * 1000
    if not np.array_equal(arrays["timestamp_us"], expected_timestamp):
        raise ValueError("timestamp_us is not contiguous at 1 kHz")
    if not np.all(np.isfinite(arrays["pelvis_imu"])):
        raise ValueError("runtime pelvis_imu contains non-finite values")
    if sensor_schema:
        if not np.all(np.isfinite(arrays["foot_fsr"])):
            raise ValueError("runtime foot_fsr contains non-finite values")
        if np.any(arrays["foot_fsr"] < 0.0):
            raise ValueError("runtime foot_fsr contains negative force")
        if not np.all(arrays["fsr_valid"]):
            raise ValueError("runtime foot_fsr contains an invalid channel")
    if not np.all(arrays["channel_valid"]) or not np.all(arrays["sample_valid"]):
        raise ValueError("authoritative runtime input contains an invalid sample")
    if not np.array_equal(
        arrays["training_eligible"], arrays["hazard_class_id"] >= 0
    ):
        raise ValueError("training eligibility and conservative class disagree")
    if not set(np.unique(arrays["hazard_class_id"])).issubset({-1, 0, 1, 2}):
        raise ValueError("hazard_class_id contains an unsupported value")
    invalid_seen = np.logical_or.accumulate(~arrays["pre_fall_valid"])
    if np.any(arrays["pre_fall_valid"] & invalid_seen):
        raise ValueError("pre_fall_valid becomes true after censor")
    if deformable_schema:
        if np.any(arrays["support_surface_displacement_m"] < 0.0):
            raise ValueError("support displacement sign violates positive-down contract")
        if not np.all(np.isfinite(arrays["support_surface_displacement_m"])):
            raise ValueError("support displacement contains non-finite values")
        expected_spread = np.ptp(
            arrays["support_surface_displacement_m"], axis=2
        )
        if not np.allclose(
            arrays["support_surface_spread_m"],
            expected_spread,
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError("support spread is inconsistent with support cells")
        if np.any(arrays["hazard_class_id"] == 1):
            raise ValueError("Sink-focused dataset unexpectedly contains Slip labels")
        first_per_foot = _first_true_per_foot(
            arrays["deformable_sink_onset"]
        )
        if not np.array_equal(
            arrays["first_deformable_sink_onset_sample_per_foot"],
            first_per_foot,
        ):
            raise ValueError("per-foot deformable Sink onset scalar is inconsistent")
        first_any = _first_true(np.any(arrays["deformable_sink_onset"], axis=1))
        if int(arrays["first_deformable_sink_onset_sample"]) != (
            -1 if first_any is None else first_any
        ):
            raise ValueError("deformable Sink onset scalar is inconsistent")
        if np.any(arrays["deformable_sink_onset"] & ~arrays["deformable_sink_active"]):
            raise ValueError("deformable Sink onset must be active")
        _validate_deformable_labels(arrays)


def _validate_deformable_labels(arrays: dict[str, np.ndarray]) -> None:
    """Reconstruct v3 labels from d0/s1/contact episodes only."""
    actual = arrays["hazard_class_id"]
    if np.all(actual == -1):
        return
    eligible = arrays["sample_valid"] & arrays["pre_fall_valid"]
    expected = np.full(len(actual), -1, dtype=np.int8)
    expected[eligible] = 0
    samples = np.arange(len(actual))
    for side in range(2):
        for onset_value in np.flatnonzero(
            arrays["deformable_sink_onset"][:, side]
        ):
            onset = int(onset_value)
            episode = int(arrays["contact_episode_id"][onset, side])
            same_episode = arrays["contact_episode_id"][:, side] == episode
            patch_episode = (
                same_episode
                & arrays["deformable_patch_episode_active"][:, side]
            )
            patch_samples = np.flatnonzero(patch_episode)
            if episode < 0 or not patch_samples.size:
                raise ValueError("deformable Sink onset has no causal d0 episode")
            d0 = int(patch_samples[0])
            if d0 > onset:
                raise ValueError("deformable d0 occurs after s1")
            expected[d0:onset] = -1
            expected[same_episode & (samples >= onset) & eligible] = 2
    if not np.array_equal(actual, expected):
        raise ValueError("v3 labels disagree with the frozen d0/s1 policy")


def write_run_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Write one complete run and verify a pickle-free round trip."""
    expected_samples = int(arrays["sequence"].shape[0])
    validate_run_arrays(arrays, expected_samples)
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as stored:
        round_trip = {name: stored[name] for name in stored.files}
    validate_run_arrays(round_trip, expected_samples)
    return sha256_file(path)


def _sample_time_ms(result: SimulationResult, sample: int | None) -> str:
    if sample is None or sample < 0:
        return ""
    return f"{float(result.runtime.timestamp_us[sample]) / 1000.0:.3f}"


def _first_contact_foot(per_foot: np.ndarray) -> str:
    left, right = (int(value) for value in per_foot)
    if left < 0 and right < 0:
        return ""
    if left == right:
        return "simultaneous"
    if right < 0 or (left >= 0 and left < right):
        return "left"
    return "right"


def _policy_phase_at_sample(
    result: SimulationResult,
    sample: int | None,
) -> str:
    if sample is None or sample < 0:
        return ""
    timestamp_s = float(result.runtime.timestamp_us[sample]) / 1_000_000.0
    control_updates = int(np.floor(timestamp_s / CONTROL_PERIOD_S + 1e-12))
    phase = (control_updates * CONTROL_PERIOD_S / POLICY_PERIOD_S) % 1.0
    return f"{phase:.6f}"


def _slip_foot_summary(diagnostics: object) -> str:
    active = np.any(diagnostics.established_slip, axis=0)
    if np.all(active):
        return "bilateral"
    if active[0]:
        return "left"
    if active[1]:
        return "right"
    return "none"


def _mechanical_recovery_sample(
    diagnostics: object,
    side: str | None,
    threshold_m: float,
    persistence_samples: int,
) -> int | None:
    if side not in SIDES:
        return None
    side_index = SIDES.index(side)
    patch_contact = diagnostics.soft_patch_contact[:, side_index]
    contact_samples = np.flatnonzero(patch_contact)
    if not contact_samples.size:
        return None
    start = int(contact_samples[-1]) + 1
    displacement = np.max(
        diagnostics.support_surface_displacement_m[:, side_index], axis=1
    )
    run_length = 0
    for sample in range(start, len(displacement)):
        if not diagnostics.pre_fall_valid[sample]:
            break
        run_length = run_length + 1 if displacement[sample] <= threshold_m else 0
        if run_length >= persistence_samples:
            return sample
    return None


def _manifest_row(
    spec: RunSpec,
    config: CollectionConfig,
    result: SimulationResult,
    arrays: dict[str, np.ndarray],
    file_name: str,
    run_hash: str,
) -> dict[str, object]:
    diagnostics = result.diagnostics
    first_patch_per_foot = arrays["first_patch_contact_sample_per_foot"]
    patch_samples = [int(value) for value in first_patch_per_foot if value >= 0]
    first_patch_sample = min(patch_samples) if patch_samples else None
    first_slip = _first_true(diagnostics.any_established_slip_onset)
    if np.any(diagnostics.soft_patch_contact):
        first_sink = _first_true(
            np.any(diagnostics.sink_physical_after_patch_onset, axis=1)
        )
    else:
        first_sink = _first_true(np.any(diagnostics.sink_physical_onset, axis=1))
    first_sink_hazard = _first_true(diagnostics.sink_hazard_onset)
    first_deformable_sink = _first_true(
        np.any(diagnostics.deformable_sink_onset, axis=1)
    )
    recovery_sample = _mechanical_recovery_sample(
        diagnostics,
        spec.sink_side,
        config.recovery_threshold_m,
        config.recovery_persistence_ms,
    )
    first_censor = int(arrays["first_censor_sample"])
    has_slip = first_slip is not None
    has_sink = first_sink_hazard is not None
    has_dual = bool(has_slip and has_sink and first_slip < first_sink_hazard)
    observability_schema = config.schema_version == "hazard_dataset_contract_v3"
    if observability_schema:
        first_patch = first_patch_sample
        insufficient_evaluation = bool(
            spec.sink_support_pattern.endswith("_deformable")
            and first_deformable_sink is None
            and first_censor >= 0
            and (
                first_patch is None
                or first_censor - first_patch
                < config.minimum_post_d0_evaluation_ms
            )
        )
        invalid = bool(
            not np.all(arrays["sample_valid"])
            or (first_censor >= 0 and first_censor < SENSOR_RATE_HZ)
            or (
                spec.sink_support_pattern.endswith("_deformable")
                and first_patch is None
            )
            or insufficient_evaluation
        )
    else:
        transition_censored_without_outcome = bool(
            spec.patch_start_x_m is not None
            and first_censor >= 0
            and (
                (spec.intended_role == "SLIP" and not has_slip)
                or (spec.intended_role == "SINK" and not has_sink)
            )
        )
        invalid = bool(
            not np.all(arrays["sample_valid"])
            or (first_censor >= 0 and first_censor < SENSOR_RATE_HZ)
            or transition_censored_without_outcome
        )
    if observability_schema:
        observed = (
            "INVALID"
            if invalid
            else "SINK"
            if first_deformable_sink is not None
            else "BENIGN"
        )
    elif invalid:
        observed = "INVALID"
    elif has_dual:
        observed = "DUAL"
    elif has_sink:
        observed = "SINK"
    elif has_slip:
        observed = "SLIP"
    else:
        observed = "BENIGN"
    assert observed in OUTCOMES
    row = {
        "run_id": spec.run_id,
        "file": file_name,
        "scenario_family": spec.scenario_family,
        "intended_role": spec.intended_role,
        "terrain": spec.terrain,
        "speed_mps": f"{spec.speed_mps:.2f}",
        "patch_start_x": (
            "" if spec.patch_start_x_m is None else f"{spec.patch_start_x_m:.2f}"
        ),
        "patch_width_m": (
            "" if spec.patch_start_x_m is None else f"{config.patch_width_m:.2f}"
        ),
        "sink_side": "" if spec.sink_side is None else spec.sink_side,
        "sink_severity": (
            "" if spec.sink_severity is None else spec.sink_severity
        ),
        "observed_outcome": observed,
        "sample_count": len(result.runtime.sequence),
        "valid_sample_count": int(np.count_nonzero(arrays["sample_valid"])),
        "drop_count": int(result.metadata["dropped_samples"]),
        "first_patch_contact_ms": _sample_time_ms(result, first_patch_sample),
        "any_slip_onset_ms": _sample_time_ms(result, first_slip),
        "sink_physical_onset_ms": _sample_time_ms(result, first_sink),
        "sink_degradation_onset_ms": _sample_time_ms(result, first_sink_hazard),
        "first_censor_ms": _sample_time_ms(
            result, None if first_censor < 0 else first_censor
        ),
        "censor_reason": str(arrays["censor_reason"]),
        "first_contact_foot": _first_contact_foot(first_patch_per_foot),
        "first_contact_policy_phase": _policy_phase_at_sample(
            result, first_patch_sample
        ),
        "slip_foot_summary": _slip_foot_summary(diagnostics),
        "has_slip": has_slip,
        "has_sink_hazard": has_sink,
        "has_dual_hazard": has_dual,
        "policy_sha256": result.metadata["policy_sha256"],
        "run_file_sha256": run_hash,
    }
    if observability_schema:
        row.update(
            {
                "sink_support_pattern": spec.sink_support_pattern,
                "sink_pattern": (
                    "uniform"
                    if spec.sink_side is None
                    else f"transition_{spec.sink_side}"
                ),
                "split": spec.split or "",
                "condition_signature": _condition_signature(
                    spec, config.patch_width_m
                ),
                "deformable_sink_onset_ms": _sample_time_ms(
                    result, first_deformable_sink
                ),
                "mechanical_recovery_ms": _sample_time_ms(
                    result, recovery_sample
                ),
                "has_deformable_sink": first_deformable_sink is not None,
            }
        )
    return row


def _simulation_config_for_run(
    base: SimulationConfig,
    spec: RunSpec,
    collection: CollectionConfig,
    policy_path: Path,
) -> SimulationConfig:
    is_slip = spec.scenario_family == "slip"
    sink_pattern = (
        "uniform" if spec.sink_side is None else f"transition_{spec.sink_side}"
    )
    return replace(
        base,
        duration_s=collection.duration_s,
        command_speed_mps=spec.speed_mps,
        policy_path=policy_path,
        terrain=spec.terrain,
        slip_pattern="transition" if is_slip else "uniform",
        sink_pattern=sink_pattern,
        sink_severity=(
            base.sink_severity
            if spec.sink_severity is None
            else spec.sink_severity
        ),
        sink_support_pattern=spec.sink_support_pattern,
        patch_start_x_m=(
            base.patch_start_x_m
            if spec.patch_start_x_m is None
            else spec.patch_start_x_m
        ),
        patch_width_m=collection.patch_width_m,
        headless=True,
    )


def _git_source_commit(require_clean: bool) -> str:
    if require_clean:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status.strip():
            raise RuntimeError(
                "collection requires a clean tracked worktree for source provenance"
            )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_manifest(
    path: Path,
    rows: list[dict[str, object]],
    schema_version: str = "hazard_dataset_contract_v1",
) -> None:
    fields = (
        OBSERVABILITY_MANIFEST_FIELDS
        if schema_version == "hazard_dataset_contract_v3"
        else MANIFEST_FIELDS
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(
    path: Path,
    config: CollectionConfig,
    source_commit: str,
    manifest_sha256: str,
) -> None:
    sensor_schema = config.schema_version in {
        "hazard_dataset_contract_v2",
        "hazard_dataset_contract_v3",
    }
    observability_schema = config.schema_version == "hazard_dataset_contract_v3"
    metadata = {
        "dataset_id": config.dataset_id,
        "schema_version": config.schema_version,
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(
            timespec="seconds"
        ),
        "source_repository": "https://github.com/shinjuyeop/Infineon_FastReflex",
        "source_commit": source_commit,
        "generator_version": source_commit,
        "experiment_config": str(config.config_path.relative_to(REPOSITORY_ROOT)),
        "experiment_config_sha256": sha256_file(config.config_path),
        "dataset_config": str(
            config.dataset_config_path.relative_to(REPOSITORY_ROOT)
        ),
        "dataset_config_sha256": sha256_file(config.dataset_config_path),
        "sample_rate_hz": SENSOR_RATE_HZ,
        "physics_rate_hz": int(round(1.0 / PHYSICS_TIMESTEP_S)),
        "channel_order": list(IMU_CHANNELS),
        "channel_units": [
            "m/s^2",
            "m/s^2",
            "m/s^2",
            "rad/s",
            "rad/s",
            "rad/s",
        ],
        "coordinate_frame": "mujoco_imu_site_local_equal_to_pelvis_body_frame",
        "policy_sha256": config.policy_sha256,
        "simulator_deterministic": config.simulator_deterministic,
        "random_seed": None,
        "controller_initial_policy_phase": 0.0,
        "policy_period_s": POLICY_PERIOD_S,
        "run_count": len(config.runs),
        "storage_format": "one_complete_run_per_compressed_npz",
        "label_contract_reference": "docs/dataset.md",
        "runtime_input_fields": [
            "sequence",
            "timestamp_us",
            "pelvis_imu",
            *(["foot_fsr"] if sensor_schema else []),
        ],
        "model_input_fields": [
            "pelvis_imu",
            *(["foot_fsr"] if sensor_schema else []),
        ],
        "alignment_only_fields": ["sequence", "timestamp_us"],
        "diagnostic_fields_are_runtime_input": False,
        "manifest_sha256": manifest_sha256,
    }
    if sensor_schema:
        metadata["candidate_sensor_profiles"] = {
            "imu6": list(IMU_CHANNELS),
            "fsr8": list(FSR_CHANNELS),
            "fusion14": [*IMU_CHANNELS, *FSR_CHANNELS],
        }
        metadata["foot_fsr"] = {
            "channel_order": list(FSR_CHANNELS),
            "dtype": "float32",
            "unit": FSR_UNIT,
            "sample_rate_hz": SENSOR_RATE_HZ,
            "construction": "summed_actual_sole_terrain_contact_normal_force_by_foot_local_quadrant",
        }
        if config.baseline_dataset_path is not None:
            metadata["baseline_dataset"] = str(
                config.baseline_dataset_path.relative_to(REPOSITORY_ROOT)
            )
            metadata["observer_only_common_field_parity"] = "bit_identical"
    if observability_schema:
        assert config.split is not None
        metadata["split_membership_frozen_before_simulation"] = True
        metadata["split_run_ids"] = {
            name: list(run_ids) for name, run_ids in config.split.items()
        }
        metadata["physical_condition_signatures"] = {
            run.run_id: _condition_signature(run, config.patch_width_m)
            for run in config.runs
        }
        metadata["sink_observability_label"] = {
            "metric": "support_surface_spread_m",
            "threshold_m": 0.010,
            "persistence_ms": 20,
            "positive_class_id": 2,
            "d0_to_s1": "excluded_only_when_s1_occurs_in_same_episode",
            "post_episode": "normal_reset",
            "minimum_post_d0_evaluation_ms": (
                config.minimum_post_d0_evaluation_ms
            ),
        }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")


def validate_dataset(path: Path) -> dict[str, object]:
    """Validate one finalized or temporary dataset directory end to end."""
    metadata_path = path / "metadata.json"
    manifest_path = path / "manifest.csv"
    runs_path = path / "runs"
    if (
        not metadata_path.is_file()
        or not manifest_path.is_file()
        or not runs_path.is_dir()
    ):
        raise ValueError("dataset layout is incomplete")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_manifest_fields = (
            OBSERVABILITY_MANIFEST_FIELDS
            if metadata.get("schema_version") == "hazard_dataset_contract_v3"
            else MANIFEST_FIELDS
        )
        if tuple(reader.fieldnames or ()) != expected_manifest_fields:
            raise ValueError("manifest columns do not match the canonical schema")
        rows = list(reader)
    required_metadata = {
        "dataset_id",
        "schema_version",
        "created_at",
        "source_repository",
        "source_commit",
        "generator_version",
        "experiment_config",
        "experiment_config_sha256",
        "dataset_config",
        "dataset_config_sha256",
        "sample_rate_hz",
        "physics_rate_hz",
        "channel_order",
        "channel_units",
        "coordinate_frame",
        "policy_sha256",
        "simulator_deterministic",
        "controller_initial_policy_phase",
        "policy_period_s",
        "run_count",
        "storage_format",
        "label_contract_reference",
        "manifest_sha256",
    }
    if not required_metadata.issubset(metadata):
        raise ValueError("metadata is missing required dataset identity fields")
    if metadata["schema_version"] not in {
        "hazard_dataset_contract_v1",
        "hazard_dataset_contract_v2",
        "hazard_dataset_contract_v3",
    }:
        raise ValueError("metadata schema version is unsupported")
    if metadata["sample_rate_hz"] != 1000 or metadata["physics_rate_hz"] != 2000:
        raise ValueError("metadata sampling rates violate the simulator contract")
    if metadata["channel_order"] != list(IMU_CHANNELS):
        raise ValueError("metadata channel order violates the runtime contract")
    sensor_schema = metadata["schema_version"] in {
        "hazard_dataset_contract_v2",
        "hazard_dataset_contract_v3",
    }
    observability_schema = metadata["schema_version"] == "hazard_dataset_contract_v3"
    if sensor_schema:
        foot_fsr = metadata.get("foot_fsr", {})
        if foot_fsr.get("channel_order") != list(FSR_CHANNELS):
            raise ValueError("metadata FSR channel order violates the runtime contract")
        if foot_fsr.get("unit") != FSR_UNIT:
            raise ValueError("metadata FSR unit violates the runtime contract")
    if observability_schema:
        required_v3_metadata = {
            "split_membership_frozen_before_simulation",
            "split_run_ids",
            "physical_condition_signatures",
            "sink_observability_label",
            "model_input_fields",
            "alignment_only_fields",
        }
        if not required_v3_metadata.issubset(metadata):
            raise ValueError("v3 metadata is missing the observability contract")
        if metadata["model_input_fields"] != ["pelvis_imu", "foot_fsr"]:
            raise ValueError("v3 model inputs must contain runtime sensors only")
        label = metadata["sink_observability_label"]
        if not (
            label.get("metric") == "support_surface_spread_m"
            and label.get("threshold_m") == 0.010
            and label.get("persistence_ms") == 20
        ):
            raise ValueError("v3 metadata changed the frozen Sink oracle")
    if not re.fullmatch(r"[0-9a-f]{40}", metadata["source_commit"]):
        raise ValueError("metadata source_commit is not a full Git SHA")
    if metadata["generator_version"] != metadata["source_commit"]:
        raise ValueError("generator version must be pinned to source_commit")
    if metadata["run_count"] != len(rows):
        raise ValueError("metadata and manifest run counts differ")
    if metadata["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("manifest SHA-256 does not match metadata")
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("manifest contains duplicate run IDs")
    if observability_schema:
        split_ids = metadata["split_run_ids"]
        assigned = [
            run_id
            for name in ("train", "validation", "holdout")
            for run_id in split_ids.get(name, [])
        ]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(run_ids):
            raise ValueError("v3 metadata split is not disjoint and exhaustive")
        split_lookup = {
            run_id: name
            for name in ("train", "validation", "holdout")
            for run_id in split_ids[name]
        }
        signatures = metadata["physical_condition_signatures"]
        if set(signatures) != set(run_ids) or len(set(signatures.values())) != len(
            run_ids
        ):
            raise ValueError("v3 physical condition signatures are incomplete or duplicate")
        for row in rows:
            run_id = row["run_id"]
            if row["split"] != split_lookup[run_id]:
                raise ValueError(f"v3 manifest split mismatch: {run_id}")
            if row["condition_signature"] != signatures[run_id]:
                raise ValueError(f"v3 condition signature mismatch: {run_id}")
    declared_files = {row["file"] for row in rows}
    actual_files = {f"runs/{item.name}" for item in runs_path.glob("*.npz")}
    if declared_files != actual_files:
        raise ValueError("manifest has a missing or orphan NPZ file")
    total_samples = 0
    outcomes = {name: 0 for name in OUTCOMES}
    for row in rows:
        run_path = path / row["file"]
        if row["file"] != f"runs/{row['run_id']}.npz":
            raise ValueError(f"run file does not match run_id: {row['run_id']}")
        if sha256_file(run_path) != row["run_file_sha256"]:
            raise ValueError(f"run SHA-256 mismatch: {row['run_id']}")
        if row["policy_sha256"] != metadata["policy_sha256"]:
            raise ValueError(f"policy SHA-256 mismatch: {row['run_id']}")
        with np.load(run_path, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
        if sensor_schema != ("foot_fsr" in arrays):
            raise ValueError(f"run sensor schema mismatch: {row['run_id']}")
        if observability_schema != ("support_surface_displacement_m" in arrays):
            raise ValueError(f"run deformable schema mismatch: {row['run_id']}")
        expected_samples = int(row["sample_count"])
        validate_run_arrays(arrays, expected_samples)
        if observability_schema:
            s1 = int(arrays["first_deformable_sink_onset_sample"])
            if row["observed_outcome"] == "SINK" and s1 < 0:
                raise ValueError(f"SINK run has no observed s1: {row['run_id']}")
            if row["observed_outcome"] == "BENIGN" and s1 >= 0:
                raise ValueError(f"BENIGN run contains observed s1: {row['run_id']}")
        if int(row["valid_sample_count"]) != expected_samples:
            raise ValueError(f"valid sample count mismatch: {row['run_id']}")
        if int(row["drop_count"]) != 0:
            raise ValueError(f"drop count is nonzero: {row['run_id']}")
        total_samples += expected_samples
        outcome = row["observed_outcome"]
        if outcome not in outcomes:
            raise ValueError(f"unknown observed outcome: {outcome}")
        outcomes[outcome] += 1
    return {
        "dataset_id": metadata["dataset_id"],
        "run_count": len(rows),
        "total_sensor_samples": total_samples,
        "outcomes": outcomes,
        "manifest_sha256": metadata["manifest_sha256"],
        "source_commit": metadata["source_commit"],
    }


def _verify_git_ignore(path: Path) -> None:
    try:
        relative = path.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", str(relative)],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"dataset output is not Git ignored: {relative}")


def _validate_observer_parity(
    candidate_arrays: dict[str, np.ndarray], baseline_run_path: Path
) -> None:
    """Require every frozen v1 field to be bit-identical to its source run."""
    if not baseline_run_path.is_file():
        raise FileNotFoundError(f"baseline parity run is missing: {baseline_run_path}")
    with np.load(baseline_run_path, allow_pickle=False) as stored:
        baseline_arrays = {name: stored[name] for name in stored.files}
    expected_fields = set(BASE_SERIES_SHAPES) | set(SCALAR_SHAPES)
    if set(baseline_arrays) != expected_fields:
        raise ValueError(f"baseline parity schema mismatch: {baseline_run_path.name}")
    for name, baseline in baseline_arrays.items():
        candidate = candidate_arrays[name]
        if np.issubdtype(baseline.dtype, np.inexact):
            identical = np.array_equal(candidate, baseline, equal_nan=True)
        else:
            identical = np.array_equal(candidate, baseline)
        if not identical:
            raise RuntimeError(
                f"observer-only parity failed for {baseline_run_path.stem}:{name}"
            )


def collect_dataset(
    config_path: Path,
    policy_path: Path,
    *,
    output_root: Path | None = None,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Execute every run, validate it, then atomically publish the dataset."""
    config = load_collection_config(config_path)
    if output_root is not None:
        config = replace(config, output_root=output_root.resolve())
    policy_path = policy_path.resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"policy artifact not found: {policy_path}")
    if sha256_file(policy_path) != config.policy_sha256:
        raise ValueError("policy SHA-256 does not match the experiment config")
    if config.baseline_dataset_path is not None:
        baseline_manifest = config.baseline_dataset_path / "manifest.csv"
        if sha256_file(baseline_manifest) != config.baseline_manifest_sha256:
            raise ValueError("baseline dataset manifest SHA-256 mismatch")
    source_commit = _git_source_commit(config.require_clean_worktree)
    final_path = config.output_root / config.dataset_id
    temporary_path = config.output_root / f".{config.dataset_id}.tmp"
    if final_path.exists():
        raise FileExistsError(f"dataset already exists; refusing overwrite: {final_path}")
    if temporary_path.exists():
        raise FileExistsError(
            f"incomplete temporary dataset requires review: {temporary_path}"
        )
    _verify_git_ignore(final_path)
    config.output_root.mkdir(parents=True, exist_ok=True)
    temporary_path.mkdir()
    (temporary_path / "runs").mkdir()
    rows: list[dict[str, object]] = []
    try:
        base = load_simulation_config(config.simulator_config_path)
        for index, spec in enumerate(config.runs, start=1):
            progress(f"[{index:02d}/{len(config.runs):02d}] {spec.run_id}")
            simulation_config = _simulation_config_for_run(
                base, spec, config, policy_path
            )
            result = run_simulation(simulation_config)
            arrays = build_run_arrays(
                result,
                intended_role=spec.intended_role,
                include_foot_fsr=(
                    config.schema_version
                    in {
                        "hazard_dataset_contract_v2",
                        "hazard_dataset_contract_v3",
                    }
                ),
                include_deformable_support=(
                    config.schema_version == "hazard_dataset_contract_v3"
                ),
                minimum_post_d0_evaluation_ms=(
                    config.minimum_post_d0_evaluation_ms
                ),
            )
            if config.baseline_dataset_path is not None:
                _validate_observer_parity(
                    arrays,
                    config.baseline_dataset_path / "runs" / f"{spec.run_id}.npz",
                )
            validate_run_arrays(arrays, simulation_config.expected_samples)
            file_name = f"runs/{spec.run_id}.npz"
            run_hash = write_run_npz(temporary_path / file_name, arrays)
            rows.append(
                _manifest_row(
                    spec,
                    config,
                    result,
                    arrays,
                    file_name,
                    run_hash,
                )
            )
        manifest_path = temporary_path / "manifest.csv"
        _write_manifest(manifest_path, rows, config.schema_version)
        _write_metadata(
            temporary_path / "metadata.json",
            config,
            source_commit,
            sha256_file(manifest_path),
        )
        summary = validate_dataset(temporary_path)
        temporary_path.rename(final_path)
        final_summary = validate_dataset(final_path)
        if summary != final_summary:
            raise RuntimeError("dataset changed during atomic finalization")
        return final_path, final_summary
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise
