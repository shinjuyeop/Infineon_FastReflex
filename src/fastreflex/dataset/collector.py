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
    }:
        raise ValueError("unsupported Hazard dataset schema version")
    if schema_version == "hazard_dataset_contract_v2" and baseline_dataset_path is None:
        raise ValueError("sensor dataset requires source.baseline_dataset for parity")
    if schema_version == "hazard_dataset_contract_v2" and not re.fullmatch(
        r"[0-9a-f]{64}", baseline_manifest_sha256 or ""
    ):
        raise ValueError("sensor dataset requires a pinned baseline manifest SHA-256")
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
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("run entry is missing required fields") from exc
        _validate_run_spec(run)
        runs.append(run)
    identifiers = [run.run_id for run in runs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("run_id values must be unique")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    runtime = result.runtime
    diagnostics = result.diagnostics
    sample_count = len(runtime.sequence)
    channel_valid = np.isfinite(runtime.pelvis_imu)
    sample_valid = np.all(channel_valid, axis=1)
    base_eligible = sample_valid & diagnostics.pre_fall_valid
    hazard_class_id = np.full(sample_count, -1, dtype=np.int8)
    hazard_class_id[base_eligible] = 0

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
    ) = _sample_annotations(result, intended_role)
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
    return {name: np.asarray(value) for name, value in arrays.items()}


def validate_run_arrays(
    arrays: dict[str, np.ndarray],
    expected_samples: int,
) -> None:
    """Fail closed on runtime corruption or diagnostic/schema misalignment."""
    sensor_schema = "foot_fsr" in arrays or "fsr_valid" in arrays
    series_shapes = dict(BASE_SERIES_SHAPES)
    if sensor_schema:
        series_shapes.update(SENSOR_SERIES_SHAPES)
    expected_keys = set(series_shapes) | set(SCALAR_SHAPES)
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
    for name, expected_shape in SCALAR_SHAPES.items():
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
    boolean_fields = (
        "sample_valid",
        "channel_valid",
        "training_eligible",
        "pre_fall_valid",
        "dual_hazard_active",
    ) + (("fsr_valid",) if sensor_schema else ())
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
    first_censor = int(arrays["first_censor_sample"])
    has_slip = first_slip is not None
    has_sink = first_sink_hazard is not None
    has_dual = bool(has_slip and has_sink and first_slip < first_sink_hazard)
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
    if invalid:
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
    return {
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


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(
    path: Path,
    config: CollectionConfig,
    source_commit: str,
    manifest_sha256: str,
) -> None:
    sensor_schema = config.schema_version == "hazard_dataset_contract_v2"
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
        metadata["baseline_dataset"] = str(
            config.baseline_dataset_path.relative_to(REPOSITORY_ROOT)
        )
        metadata["observer_only_common_field_parity"] = "bit_identical"
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
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
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
    }:
        raise ValueError("metadata schema version is unsupported")
    if metadata["sample_rate_hz"] != 1000 or metadata["physics_rate_hz"] != 2000:
        raise ValueError("metadata sampling rates violate the simulator contract")
    if metadata["channel_order"] != list(IMU_CHANNELS):
        raise ValueError("metadata channel order violates the runtime contract")
    sensor_schema = metadata["schema_version"] == "hazard_dataset_contract_v2"
    if sensor_schema:
        foot_fsr = metadata.get("foot_fsr", {})
        if foot_fsr.get("channel_order") != list(FSR_CHANNELS):
            raise ValueError("metadata FSR channel order violates the runtime contract")
        if foot_fsr.get("unit") != FSR_UNIT:
            raise ValueError("metadata FSR unit violates the runtime contract")
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
        expected_samples = int(row["sample_count"])
        validate_run_arrays(arrays, expected_samples)
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
                    config.schema_version == "hazard_dataset_contract_v2"
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
        _write_manifest(manifest_path, rows)
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
