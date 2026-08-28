"""Terrain-transition run collection and touchdown-event dataset contract."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from fastreflex.simulation.g1 import (
    SENSOR_RATE_HZ,
    TESTED_POLICY_SHA256,
    SimulationConfig,
    SimulationResult,
    load_simulation_config,
    run_simulation,
    sha256_file,
)
from fastreflex.simulation.sensors import FOOT_IMU_CHANNELS, FSR_CHANNELS
from fastreflex.simulation.terrain import TERRAIN_CLASS_ORDER


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TERRAIN_CLASS_NAMES = tuple(name.upper() for name in TERRAIN_CLASS_ORDER)
SIDES = ("left", "right")
SENSOR_PROFILE_CHANNELS = {"fsr4": 4, "foot_imu6": 6, "fusion10": 10}
RUNTIME_INPUT_FIELDS = ("pelvis_imu", "foot_fsr", "foot_imu")
DIAGNOSTIC_FIELDS = (
    "exact_terrain_contact",
    "physical_contact",
    "touchdown",
    "pre_fall_valid",
    "fall_active",
    "established_slip_active",
    "deformable_sink_active",
    "support_surface_displacement_m",
)
RUN_MANIFEST_FIELDS = (
    "run_id",
    "file",
    "split",
    "condition_signature",
    "source_terrain",
    "target_terrain",
    "intended_outcome",
    "observed_outcome",
    "speed_mps",
    "patch_start_x_m",
    "patch_width_m",
    "slip_pattern",
    "sink_pattern",
    "sink_severity",
    "support_pattern",
    "sample_count",
    "drop_count",
    "first_target_contact_us",
    "first_fall_us",
    "pretransition_fall",
    "physical_slip_present",
    "physical_sink_present",
    "run_file_sha256",
)
EVENT_INDEX_FIELDS = (
    "event_id",
    "run_id",
    "foot",
    "terrain_gt",
    "terrain_class_id",
    "touchdown_us",
    "touchdown_sample",
    "window_20ms_valid",
    "window_30ms_valid",
    "window_50ms_valid",
    "mixed_contact_ratio",
    "source_terrain",
    "target_terrain",
    "is_target_terrain",
    "observed_fall",
    "physical_slip_present",
    "physical_sink_present",
    "split",
    "eligible",
    "exclusion_reason",
)


@dataclass(frozen=True)
class TerrainRunSpec:
    """One predeclared transition condition and run-level split."""

    run_id: str
    split: str
    source_terrain: str
    target_terrain: str
    intended_outcome: str
    speed_mps: float
    patch_start_x_m: float
    patch_width_m: float
    slip_pattern: str
    sink_pattern: str
    sink_severity: str
    support_pattern: str
    condition_signature: str


@dataclass(frozen=True)
class TerrainCollectionConfig:
    """Validated terrain dataset collection contract."""

    config_path: Path
    dataset_id: str
    dataset_schema: str
    simulator_config_path: Path
    dataset_config_path: Path
    frozen_transition_config_path: Path
    policy_sha256: str
    require_clean_worktree: bool
    output_path: Path
    artifact_path: Path
    duration_s: float
    horizons_ms: tuple[int, ...]
    primary_window_ms: int
    mixed_ratio_max_exclusive: float
    max_events_per_class_per_run: int
    runs: tuple[TerrainRunSpec, ...]


@dataclass(frozen=True)
class TerrainNormalizer:
    """Per-channel z-score statistics fit only on declared train events."""

    mean: np.ndarray
    std: np.ndarray
    fit_event_ids: tuple[str, ...]
    fit_run_ids: tuple[str, ...]
    sample_count: int
    epsilon: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "per_channel_zscore",
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fit_event_ids": list(self.fit_event_ids),
            "fit_run_ids": list(self.fit_run_ids),
            "sample_count": self.sample_count,
            "epsilon": self.epsilon,
        }


@dataclass(frozen=True)
class TerrainWindowSet:
    """One causal touchdown window per selected event."""

    inputs: np.ndarray
    targets: np.ndarray
    run_ids: np.ndarray
    event_ids: np.ndarray
    feet: np.ndarray
    touchdown_samples: np.ndarray

    @property
    def selected_by_class(self) -> tuple[int, int, int, int]:
        counts = np.bincount(self.targets, minlength=4)
        return tuple(int(value) for value in counts[:4])

    def __len__(self) -> int:
        return int(len(self.targets))


class HoldoutGuard:
    """Fail closed until validation selections have been frozen."""

    def __init__(self) -> None:
        self._open = False
        self.open_count = 0

    def open_once(self) -> None:
        if self._open or self.open_count:
            raise RuntimeError("holdout guard may be opened exactly once")
        self._open = True
        self.open_count = 1

    def require_open(self) -> None:
        if not self._open:
            raise RuntimeError("holdout waveform access is sealed during selection")


def _repository_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a repository-relative path")
    path = (REPOSITORY_ROOT / value).resolve()
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{field} must remain inside the repository") from exc
    return path


def _condition_signature(values: Mapping[str, object]) -> str:
    identity = {
        name: values[name]
        for name in (
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
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _expand_run_matrix(document: Mapping[str, object]) -> tuple[TerrainRunSpec, ...]:
    matrix = document["run_matrix"]
    split = document["split"]
    if not isinstance(matrix, dict) or not isinstance(split, dict):
        raise ValueError("run_matrix and split must be mappings")
    sources = tuple(str(value) for value in matrix["source_terrains"])
    if sources != ("concrete", "marble"):
        raise ValueError("terrain dataset source terrains must be concrete and marble")
    groups = matrix["groups"]
    if not isinstance(groups, dict) or set(groups) != {
        "ice_stable",
        "ice_fall",
        "sand_stable",
        "sand_fall",
    }:
        raise ValueError("terrain run groups differ from the frozen four groups")
    runs: list[TerrainRunSpec] = []
    for source in sources:
        for group_name, raw_group in groups.items():
            if not isinstance(raw_group, dict):
                raise ValueError("terrain run group must be a mapping")
            widths = tuple(float(value) for value in raw_group["patch_width_m"])
            starts = raw_group["patch_start_x_m"]
            starts = tuple(float(value) for value in starts)
            patterns = raw_group["sink_pattern"]
            patterns = (
                tuple(str(value) for value in patterns)
                if isinstance(patterns, list)
                else (str(patterns),)
            )
            if len(widths) != 18:
                raise ValueError(f"{group_name} must contain 18 frozen variations")
            role = str(raw_group["intended_outcome"])
            split_indices = {
                name: tuple(int(value) for value in split[f"{role}_variant_indices"][name])
                for name in ("train", "validation", "holdout")
            }
            assigned = [value for indices in split_indices.values() for value in indices]
            if sorted(assigned) != list(range(18)) or len(set(assigned)) != 18:
                raise ValueError(f"{role} variant split is not disjoint and exhaustive")
            split_by_index = {
                value: name for name, indices in split_indices.items() for value in indices
            }
            for index, width in enumerate(widths):
                values: dict[str, object] = {
                    "source_terrain": source,
                    "target_terrain": str(raw_group["target_terrain"]),
                    "speed_mps": float(raw_group["speed_mps"]),
                    "patch_start_x_m": starts[index % len(starts)],
                    "patch_width_m": width,
                    "slip_pattern": str(raw_group["slip_pattern"]),
                    "sink_pattern": patterns[index % len(patterns)],
                    "sink_severity": str(raw_group["sink_severity"]),
                    "support_pattern": str(raw_group["support_pattern"]),
                }
                runs.append(
                    TerrainRunSpec(
                        run_id=f"tr_{source}_{group_name}_{index:02d}",
                        split=split_by_index[index],
                        intended_outcome=role,
                        condition_signature=_condition_signature(values),
                        **values,
                    )
                )
    return tuple(runs)


def load_terrain_collection_config(path: Path) -> TerrainCollectionConfig:
    """Load the frozen terrain matrix without accessing generated data."""
    path = path.resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("terrain experiment config must be a mapping")
    if document.get("experiment", {}).get("id") != "TERRAIN_REBUILD_AND_SENSOR_ABLATION":
        raise ValueError("unsupported terrain experiment")
    dataset_config_path = _repository_path(
        document["source"]["dataset_config"], "source.dataset_config"
    )
    with dataset_config_path.open("r", encoding="utf-8") as stream:
        dataset_document = yaml.safe_load(stream)
    if dataset_document.get("schema_version") != "terrain_event_contract_v1":
        raise ValueError("unsupported Terrain dataset schema")
    runs = _expand_run_matrix(document)
    if len(runs) != int(document["common"]["independent_runs"]):
        raise ValueError("expanded run count differs from the frozen declaration")
    if len({run.run_id for run in runs}) != len(runs):
        raise ValueError("duplicate terrain run id")
    if len({run.condition_signature for run in runs}) != len(runs):
        raise ValueError("duplicate terrain physical condition")
    split_counts = {
        name: sum(run.split == name for run in runs)
        for name in ("train", "validation", "holdout")
    }
    if split_counts != {
        name: int(value) for name, value in document["split"]["counts"].items()
    }:
        raise ValueError("expanded split counts differ from the frozen declaration")
    horizons = tuple(int(value) for value in document["event_eligibility"]["horizons_ms"])
    if horizons != (20, 30, 50):
        raise ValueError("terrain observation horizons must remain 20/30/50 ms")
    mixed_ratio = float(
        document["event_eligibility"]["mixed_terrain_sample_ratio_exclusive_max"]
    )
    if mixed_ratio != 0.20:
        raise ValueError("mixed-terrain exclusion threshold must remain 20%")
    config = TerrainCollectionConfig(
        config_path=path,
        dataset_id=str(document["experiment"]["dataset_id"]),
        dataset_schema=str(dataset_document["schema_version"]),
        simulator_config_path=_repository_path(
            document["source"]["simulator_config"], "source.simulator_config"
        ),
        dataset_config_path=dataset_config_path,
        frozen_transition_config_path=_repository_path(
            document["source"]["frozen_transition_config"],
            "source.frozen_transition_config",
        ),
        policy_sha256=str(document["source"]["policy_sha256"]),
        require_clean_worktree=bool(document["source"]["require_clean_worktree"]),
        output_path=_repository_path(
            document["output"]["dataset_path"], "output.dataset_path"
        ),
        artifact_path=_repository_path(
            document["output"]["artifact_path"], "output.artifact_path"
        ),
        duration_s=float(document["common"]["duration_s"]),
        horizons_ms=horizons,
        primary_window_ms=int(document["event_eligibility"]["primary_window_ms"]),
        mixed_ratio_max_exclusive=mixed_ratio,
        max_events_per_class_per_run=int(
            document["common"]["max_clean_events_per_class_per_run"]
        ),
        runs=runs,
    )
    if config.policy_sha256 != TESTED_POLICY_SHA256:
        raise ValueError("terrain config does not pin the verified G1 policy")
    for required in (
        config.simulator_config_path,
        config.dataset_config_path,
        config.frozen_transition_config_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    return config


def simulation_config_for_terrain_run(
    base: SimulationConfig,
    run: TerrainRunSpec,
    collection: TerrainCollectionConfig,
    policy_path: Path,
) -> SimulationConfig:
    return replace(
        base,
        duration_s=collection.duration_s,
        command_speed_mps=run.speed_mps,
        policy_path=policy_path,
        terrain=run.target_terrain,
        source_terrain=run.source_terrain,
        slip_pattern=run.slip_pattern,
        sink_pattern=run.sink_pattern,
        sink_severity=run.sink_severity,
        sink_support_pattern=run.support_pattern,
        patch_start_x_m=run.patch_start_x_m,
        patch_width_m=run.patch_width_m,
        headless=True,
    )


def terrain_identity_touchdown(contact: np.ndarray) -> np.ndarray:
    """Return causal rising edges independently for every foot/terrain pair."""
    values = np.asarray(contact, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != (2, 4):
        raise ValueError("exact terrain contact must have shape [N,2,4]")
    previous = np.concatenate((np.zeros((1, 2, 4), dtype=bool), values[:-1]))
    return values & ~previous


def build_touchdown_event_rows(
    run_id: str,
    split: str,
    source_terrain: str,
    target_terrain: str,
    timestamp_us: np.ndarray,
    exact_terrain_contact: np.ndarray,
    first_fall_sample: int | None,
    physical_slip_present: bool,
    physical_sink_present: bool,
    horizons_ms: Sequence[int] = (20, 30, 50),
    mixed_ratio_max_exclusive: float = 0.20,
) -> list[dict[str, object]]:
    """Index clean and excluded terrain-identity touchdown candidates."""
    timestamps = np.asarray(timestamp_us, dtype=np.int64)
    contact = np.asarray(exact_terrain_contact, dtype=bool)
    if contact.shape != (len(timestamps), 2, 4):
        raise ValueError("terrain contact/timestamp alignment mismatch")
    if tuple(horizons_ms) != (20, 30, 50):
        raise ValueError("event builder requires the frozen 20/30/50 horizons")
    touchdown = terrain_identity_touchdown(contact)
    observed_fall = first_fall_sample is not None
    target_id = TERRAIN_CLASS_ORDER.index(target_terrain)
    rows: list[dict[str, object]] = []
    for sample, foot_id, class_id in np.argwhere(touchdown):
        sample, foot_id, class_id = int(sample), int(foot_id), int(class_id)
        horizon_valid: dict[int, bool] = {}
        primary_stop = sample + 50
        if primary_stop <= len(contact):
            other = contact[sample:primary_stop, foot_id].copy()
            other[:, class_id] = False
            mixed_ratio = float(np.mean(np.any(other, axis=1)))
        else:
            mixed_ratio = float("nan")
        for horizon in horizons_ms:
            stop = sample + int(horizon)
            complete = stop <= len(contact)
            before_fall = first_fall_sample is None or stop <= first_fall_sample
            persistent = complete and bool(np.all(contact[sample:stop, foot_id, class_id]))
            if complete:
                other = contact[sample:stop, foot_id].copy()
                other[:, class_id] = False
                horizon_mixed = float(np.mean(np.any(other, axis=1)))
            else:
                horizon_mixed = 1.0
            horizon_valid[int(horizon)] = bool(
                complete
                and before_fall
                and persistent
                and horizon_mixed < mixed_ratio_max_exclusive
            )
        if first_fall_sample is not None and sample >= first_fall_sample:
            reason = "POST_FALL"
        elif primary_stop > len(contact) or (
            first_fall_sample is not None and primary_stop > first_fall_sample
        ):
            reason = "CENSORED"
        elif not np.all(contact[sample:primary_stop, foot_id, class_id]):
            reason = "CONTACT_NOT_PERSISTENT"
        elif mixed_ratio >= mixed_ratio_max_exclusive:
            reason = "AMBIGUOUS_BOUNDARY"
        else:
            reason = ""
        terrain_name = TERRAIN_CLASS_NAMES[class_id]
        event_id = f"{run_id}_{SIDES[foot_id]}_{sample:05d}_{terrain_name.lower()}"
        rows.append(
            {
                "event_id": event_id,
                "run_id": run_id,
                "foot": SIDES[foot_id],
                "terrain_gt": terrain_name,
                "terrain_class_id": class_id,
                "touchdown_us": int(timestamps[sample]),
                "touchdown_sample": sample,
                "window_20ms_valid": horizon_valid[20],
                "window_30ms_valid": horizon_valid[30],
                "window_50ms_valid": horizon_valid[50],
                "mixed_contact_ratio": mixed_ratio,
                "source_terrain": source_terrain,
                "target_terrain": target_terrain,
                "is_target_terrain": class_id == target_id,
                "observed_fall": observed_fall,
                "physical_slip_present": physical_slip_present,
                "physical_sink_present": physical_sink_present,
                "split": split,
                "eligible": horizon_valid[50],
                "exclusion_reason": reason,
            }
        )
    return sorted(rows, key=lambda row: (row["touchdown_sample"], row["foot"], row["terrain_class_id"]))


def select_capped_events(
    rows: Sequence[Mapping[str, object]],
    split: str,
    cap_per_run_class: int,
    *,
    foot: str | None = None,
) -> list[dict[str, object]]:
    """Apply the frozen deterministic cap without changing the raw index."""
    if cap_per_run_class <= 0:
        raise ValueError("event cap must be positive")
    selected: list[dict[str, object]] = []
    groups: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for row in rows:
        if str(row["split"]) != split or not bool(row["window_50ms_valid"]):
            continue
        if foot is not None and str(row["foot"]) != foot:
            continue
        key = (str(row["run_id"]), int(row["terrain_class_id"]))
        groups.setdefault(key, []).append(row)
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda row: (int(row["touchdown_sample"]), str(row["event_id"])))
        if len(candidates) > cap_per_run_class:
            indices = np.linspace(0, len(candidates) - 1, cap_per_run_class, dtype=np.int64)
            candidates = [candidates[int(index)] for index in indices]
        selected.extend(dict(row) for row in candidates)
    return sorted(selected, key=lambda row: str(row["event_id"]))


def extract_terrain_sensor_profile(
    foot_fsr: np.ndarray,
    foot_imu: np.ndarray,
    foot: str,
    profile: str,
) -> np.ndarray:
    """Slice only the touchdown foot; side identity is never appended."""
    fsr = np.asarray(foot_fsr, dtype=np.float32)
    imu = np.asarray(foot_imu, dtype=np.float32)
    if fsr.ndim != 2 or fsr.shape[1] != 8:
        raise ValueError("foot_fsr must have shape [N,8]")
    if imu.shape != (len(fsr), 12):
        raise ValueError("foot_imu must have shape [N,12] aligned to foot_fsr")
    if foot not in SIDES:
        raise ValueError("foot must be left or right")
    side = SIDES.index(foot)
    fsr4 = fsr[:, side * 4 : (side + 1) * 4]
    imu6 = imu[:, side * 6 : (side + 1) * 6]
    if profile == "fsr4":
        return fsr4
    if profile == "foot_imu6":
        return imu6
    if profile == "fusion10":
        return np.concatenate((fsr4, imu6), axis=1).astype(np.float32, copy=False)
    raise ValueError(f"unsupported terrain sensor profile: {profile}")


def _load_event_run(dataset_path: Path, run_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = dataset_path / "runs" / f"{run_id}.npz"
    with np.load(path, allow_pickle=False) as stored:
        fsr = np.asarray(stored["foot_fsr"], dtype=np.float32)
        imu = np.asarray(stored["foot_imu"], dtype=np.float32)
    if not np.all(np.isfinite(fsr)) or np.any(fsr < 0.0):
        raise ValueError(f"invalid FSR values in {run_id}")
    if not np.all(np.isfinite(imu)):
        raise ValueError(f"invalid foot IMU values in {run_id}")
    return fsr, imu


def build_terrain_windows(
    dataset_path: Path,
    rows: Sequence[Mapping[str, object]],
    profile: str,
    horizon_ms: int,
    normalizer: TerrainNormalizer | None = None,
    holdout_guard: HoldoutGuard | None = None,
) -> TerrainWindowSet:
    """Build causal post-touchdown windows from runtime sensors only."""
    if horizon_ms not in {20, 30, 50}:
        raise ValueError("terrain horizon must be 20, 30, or 50 ms")
    if profile not in SENSOR_PROFILE_CHANNELS:
        raise ValueError("unsupported terrain sensor profile")
    if any(str(row["split"]) == "holdout" for row in rows):
        if holdout_guard is None:
            raise RuntimeError("holdout windows require an explicit guard")
        holdout_guard.require_open()
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    inputs: list[np.ndarray] = []
    targets: list[int] = []
    run_ids: list[str] = []
    event_ids: list[str] = []
    feet: list[str] = []
    touchdown_samples: list[int] = []
    valid_key = f"window_{horizon_ms}ms_valid"
    for row in sorted(rows, key=lambda item: str(item["event_id"])):
        if not bool(row[valid_key]):
            raise ValueError(f"selected event is invalid at {horizon_ms} ms")
        run_id = str(row["run_id"])
        if run_id not in cache:
            cache[run_id] = _load_event_run(dataset_path, run_id)
        fsr, imu = cache[run_id]
        sample = int(row["touchdown_sample"])
        values = extract_terrain_sensor_profile(fsr, imu, str(row["foot"]), profile)
        window = values[sample : sample + horizon_ms]
        if window.shape != (horizon_ms, SENSOR_PROFILE_CHANNELS[profile]):
            raise ValueError("event window is incomplete")
        inputs.append(window)
        targets.append(int(row["terrain_class_id"]))
        run_ids.append(run_id)
        event_ids.append(str(row["event_id"]))
        feet.append(str(row["foot"]))
        touchdown_samples.append(sample)
    if not inputs:
        raise ValueError("no terrain event windows were selected")
    values = np.stack(inputs).astype(np.float32)
    if normalizer is not None:
        values = normalizer.transform(values)
    return TerrainWindowSet(
        inputs=values,
        targets=np.asarray(targets, dtype=np.int64),
        run_ids=np.asarray(run_ids, dtype=object),
        event_ids=np.asarray(event_ids, dtype=object),
        feet=np.asarray(feet, dtype=object),
        touchdown_samples=np.asarray(touchdown_samples, dtype=np.int64),
    )


def fit_terrain_normalizer(
    windows: TerrainWindowSet,
    *,
    epsilon: float = 1.0e-8,
) -> TerrainNormalizer:
    """Fit moments only from an already selected train-event WindowSet."""
    if windows.inputs.ndim != 3 or not np.all(np.isfinite(windows.inputs)):
        raise ValueError("normalizer inputs must be finite [events,time,channels]")
    values = windows.inputs.astype(np.float64).reshape(-1, windows.inputs.shape[-1])
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    if np.any(std <= epsilon):
        raise ValueError("near-constant terrain sensor channel")
    return TerrainNormalizer(
        mean=mean.astype(np.float32),
        std=np.maximum(std, epsilon).astype(np.float32),
        fit_event_ids=tuple(str(value) for value in windows.event_ids),
        fit_run_ids=tuple(sorted({str(value) for value in windows.run_ids})),
        sample_count=int(len(values)),
        epsilon=epsilon,
    )


def read_event_index(path: Path) -> list[dict[str, object]]:
    """Load event metadata without touching waveform arrays."""
    boolean_fields = {
        "window_20ms_valid",
        "window_30ms_valid",
        "window_50ms_valid",
        "is_target_terrain",
        "observed_fall",
        "physical_slip_present",
        "physical_sink_present",
        "eligible",
    }
    integer_fields = {"terrain_class_id", "touchdown_us", "touchdown_sample"}
    float_fields = {"mixed_contact_ratio"}
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, object] = dict(raw)
            for name in boolean_fields:
                row[name] = raw[name].lower() == "true"
            for name in integer_fields:
                row[name] = int(raw[name])
            for name in float_fields:
                row[name] = float(raw[name])
            rows.append(row)
    return rows


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if not len(indices) else int(indices[0])


def _result_arrays(result: SimulationResult) -> dict[str, np.ndarray]:
    runtime = result.runtime
    diagnostics = result.diagnostics
    if runtime.foot_fsr is None or runtime.foot_imu is None:
        raise ValueError("terrain collection requires FSR and bilateral foot IMU")
    if result.exact_terrain_contact is None:
        raise ValueError("terrain collection requires exact terrain contact GT")
    return {
        "sequence": runtime.sequence.astype(np.int64, copy=False),
        "timestamp_us": runtime.timestamp_us.astype(np.int64, copy=False),
        "pelvis_imu": runtime.pelvis_imu.astype(np.float32, copy=False),
        "foot_fsr": runtime.foot_fsr.astype(np.float32, copy=False),
        "foot_imu": runtime.foot_imu.astype(np.float32, copy=False),
        "exact_terrain_contact": result.exact_terrain_contact.astype(bool, copy=False),
        "physical_contact": diagnostics.physical_contact.astype(bool, copy=False),
        "touchdown": diagnostics.touchdown.astype(bool, copy=False),
        "pre_fall_valid": diagnostics.pre_fall_valid.astype(bool, copy=False),
        "fall_active": diagnostics.fall_active.astype(bool, copy=False),
        "established_slip_active": diagnostics.established_slip.astype(bool, copy=False),
        "deformable_sink_active": diagnostics.deformable_sink_active.astype(bool, copy=False),
        "support_surface_displacement_m": diagnostics.support_surface_displacement_m.astype(np.float32),
    }


def _validate_result_arrays(arrays: Mapping[str, np.ndarray], samples: int) -> None:
    shapes = {
        "sequence": (samples,),
        "timestamp_us": (samples,),
        "pelvis_imu": (samples, 6),
        "foot_fsr": (samples, 8),
        "foot_imu": (samples, 12),
        "exact_terrain_contact": (samples, 2, 4),
        "physical_contact": (samples, 2),
        "touchdown": (samples, 2),
        "pre_fall_valid": (samples,),
        "fall_active": (samples,),
        "established_slip_active": (samples, 2),
        "deformable_sink_active": (samples, 2),
        "support_surface_displacement_m": (samples, 2, 4),
    }
    if set(arrays) != set(shapes):
        raise ValueError("terrain NPZ fields differ from the frozen schema")
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} shape {arrays[name].shape} != {shape}")
    if arrays["pelvis_imu"].dtype != np.float32:
        raise ValueError("pelvis_imu must be float32")
    if arrays["foot_fsr"].dtype != np.float32 or np.any(arrays["foot_fsr"] < 0.0):
        raise ValueError("foot_fsr must be nonnegative float32")
    if arrays["foot_imu"].dtype != np.float32:
        raise ValueError("foot_imu must be float32")
    if not all(
        np.all(np.isfinite(arrays[name]))
        for name in ("pelvis_imu", "foot_fsr", "foot_imu")
    ):
        raise ValueError("runtime sensors contain a non-finite value")
    expected_sequence = np.arange(samples, dtype=np.int64)
    expected_timestamp = (expected_sequence + 1) * 1000
    if not np.array_equal(arrays["sequence"], expected_sequence):
        raise ValueError("terrain run has a dropped or duplicate sequence")
    if not np.array_equal(arrays["timestamp_us"], expected_timestamp):
        raise ValueError("terrain run timestamps are not aligned at 1 kHz")


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_source_commit(require_clean: bool) -> tuple[str, bool]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    clean = not status.strip()
    if require_clean and not clean:
        raise RuntimeError("terrain collection requires a clean tracked worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, clean


def _compare_parity_pair(
    config: SimulationConfig,
    group: str,
) -> dict[str, object]:
    without_read = run_simulation(
        config, observe_fsr=True, observe_foot_imu=False, capture_state_trace=True
    )
    with_read = run_simulation(
        config, observe_fsr=True, observe_foot_imu=True, capture_state_trace=True
    )
    assert without_read.state_trace is not None and with_read.state_trace is not None
    state_fields = (
        "robot_qpos",
        "robot_qvel",
        "controller_observation",
        "controller_action",
        "policy_updated",
        "pelvis_pose",
        "whole_body_com",
    )
    state_equal = {
        name: bool(
            np.array_equal(
                getattr(without_read.state_trace, name),
                getattr(with_read.state_trace, name),
            )
        )
        for name in state_fields
    }
    diagnostic_equal = {
        "pelvis_imu": bool(
            np.array_equal(without_read.runtime.pelvis_imu, with_read.runtime.pelvis_imu)
        ),
        "foot_fsr": bool(
            np.array_equal(without_read.runtime.foot_fsr, with_read.runtime.foot_fsr)
        ),
        "exact_contact": bool(
            np.array_equal(
                without_read.exact_terrain_contact, with_read.exact_terrain_contact
            )
        ),
        "fall": bool(
            np.array_equal(
                without_read.diagnostics.fall_active, with_read.diagnostics.fall_active
            )
        ),
        "slip": bool(
            np.array_equal(
                without_read.diagnostics.established_slip,
                with_read.diagnostics.established_slip,
            )
        ),
        "deformable_support": bool(
            np.array_equal(
                without_read.diagnostics.support_surface_displacement_m,
                with_read.diagnostics.support_surface_displacement_m,
            )
        ),
    }
    foot_imu = with_read.runtime.foot_imu
    passed = all(state_equal.values()) and all(diagnostic_equal.values())
    return {
        "group": group,
        "passed": passed,
        "state_exact": state_equal,
        "diagnostic_exact": diagnostic_equal,
        "foot_imu_shape": None if foot_imu is None else list(foot_imu.shape),
        "foot_imu_finite": bool(foot_imu is not None and np.all(np.isfinite(foot_imu))),
        "qpos_max_abs_difference": float(
            np.max(
                np.abs(
                    without_read.state_trace.robot_qpos
                    - with_read.state_trace.robot_qpos
                )
            )
        ),
        "qvel_max_abs_difference": float(
            np.max(
                np.abs(
                    without_read.state_trace.robot_qvel
                    - with_read.state_trace.robot_qvel
                )
            )
        ),
    }


def validate_foot_imu_observer_parity(
    base: SimulationConfig,
    policy_path: Path,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Verify that reading the new passive sensors cannot mutate simulation."""
    cases = (
        (
            "hard",
            replace(
                base,
                duration_s=2.0,
                command_speed_mps=0.25,
                policy_path=policy_path,
                terrain="concrete",
                source_terrain="concrete",
                slip_pattern="uniform",
                sink_pattern="uniform",
                sink_support_pattern="balanced_soft",
                headless=True,
            ),
        ),
        (
            "ice",
            replace(
                base,
                duration_s=4.0,
                command_speed_mps=0.25,
                policy_path=policy_path,
                terrain="ice",
                source_terrain="concrete",
                slip_pattern="transition",
                sink_pattern="uniform",
                sink_support_pattern="balanced_soft",
                patch_start_x_m=0.36,
                patch_width_m=0.72,
                headless=True,
            ),
        ),
        (
            "sand",
            replace(
                base,
                duration_s=4.0,
                command_speed_mps=0.25,
                policy_path=policy_path,
                terrain="sand",
                source_terrain="concrete",
                slip_pattern="uniform",
                sink_pattern="transition_left",
                sink_severity="mild",
                sink_support_pattern="balanced_deformable",
                patch_start_x_m=0.30,
                patch_width_m=0.73,
                headless=True,
            ),
        ),
    )
    results = []
    for group, config in cases:
        progress(f"[parity] {group}")
        results.append(_compare_parity_pair(config, group))
    return {
        "comparison": "observer_read_disabled_vs_enabled",
        "sensor_declarations_are_passive_mujoco_observers": True,
        "cases": results,
        "passed": all(bool(result["passed"]) for result in results),
    }


def _coverage_summary(
    event_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    clean = [row for row in event_rows if bool(row["window_50ms_valid"])]
    class_counts = {
        name: sum(str(row["terrain_gt"]) == name for row in clean)
        for name in TERRAIN_CLASS_NAMES
    }
    side_counts = {side: sum(str(row["foot"]) == side for row in clean) for side in SIDES}
    outcome_counts = {
        terrain: {
            outcome: sum(
                str(row["terrain_gt"]) == terrain
                and bool(row["is_target_terrain"])
                and ("fall" if bool(row["observed_fall"]) else "stable") == outcome
                for row in clean
            )
            for outcome in ("stable", "fall")
        }
        for terrain in ("ICE", "SAND")
    }
    source_target_counts = {
        source: {
            terrain: sum(
                str(row["source_terrain"]) == source
                and str(row["terrain_gt"]) == terrain
                and bool(row["is_target_terrain"])
                for row in clean
            )
            for terrain in ("ICE", "SAND")
        }
        for source in ("concrete", "marble")
    }
    ambiguous = sum(
        str(row["exclusion_reason"]) == "AMBIGUOUS_BOUNDARY" for row in event_rows
    )
    duplicate_conditions = len(manifest_rows) - len(
        {str(row["condition_signature"]) for row in manifest_rows}
    )
    pretransition_falls = sum(
        str(row["pretransition_fall"]).lower() == "true" for row in manifest_rows
    )
    split_ids = {
        name: {str(row["run_id"]) for row in manifest_rows if row["split"] == name}
        for name in ("train", "validation", "holdout")
    }
    split_overlap = sum(
        len(split_ids[left] & split_ids[right])
        for left, right in (("train", "validation"), ("train", "holdout"), ("validation", "holdout"))
    )
    dropped = sum(int(row["drop_count"]) for row in manifest_rows)
    gates = {
        "clean_events_at_least_240": len(clean) >= 240,
        "each_class_at_least_60": min(class_counts.values(), default=0) >= 60,
        "left_events_at_least_80": side_counts["left"] >= 80,
        "right_events_at_least_80": side_counts["right"] >= 80,
        "ice_stable_and_fall": min(outcome_counts["ICE"].values(), default=0) > 0,
        "sand_stable_and_fall": min(outcome_counts["SAND"].values(), default=0) > 0,
        "source_diversity": all(
            source_target_counts[source][terrain] > 0
            for source in source_target_counts
            for terrain in source_target_counts[source]
        ),
        "duplicate_conditions_zero": duplicate_conditions == 0,
        "split_overlap_zero": split_overlap == 0,
        "drop_zero": dropped == 0,
        "pretransition_fall_zero": pretransition_falls == 0,
    }
    return {
        "clean_event_count": len(clean),
        "class_counts": class_counts,
        "side_counts": side_counts,
        "target_outcome_counts": outcome_counts,
        "source_target_counts": source_target_counts,
        "ambiguous_boundary_count": ambiguous,
        "excluded_event_count": len(event_rows) - len(clean),
        "duplicate_condition_count": duplicate_conditions,
        "split_overlap_count": split_overlap,
        "drop_count": dropped,
        "pretransition_fall_count": pretransition_falls,
        "gates": gates,
        "passed": all(gates.values()),
    }


def validate_terrain_dataset(path: Path) -> dict[str, object]:
    """Validate identity, checksums, runtime arrays, split, and event gates."""
    with (path / "metadata.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema_version") != "terrain_event_contract_v1":
        raise ValueError("unexpected terrain dataset schema")
    if metadata.get("model_input_fields") != ["foot_fsr", "foot_imu"]:
        raise ValueError("terrain dataset model input declaration leaks a forbidden field")
    with (path / "manifest.csv").open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RUN_MANIFEST_FIELDS:
            raise ValueError("terrain manifest columns changed")
        manifest_rows = list(reader)
    event_rows = read_event_index(path / "events.csv")
    if metadata["manifest_sha256"] != sha256_file(path / "manifest.csv"):
        raise ValueError("terrain manifest checksum mismatch")
    if metadata["event_index_sha256"] != sha256_file(path / "events.csv"):
        raise ValueError("terrain event index checksum mismatch")
    if len(manifest_rows) != int(metadata["run_count"]):
        raise ValueError("terrain run count mismatch")
    run_ids = {row["run_id"] for row in manifest_rows}
    if len(run_ids) != len(manifest_rows):
        raise ValueError("duplicate run id in terrain manifest")
    if len({row["event_id"] for row in event_rows}) != len(event_rows):
        raise ValueError("duplicate terrain event id")
    split_by_run = {row["run_id"]: row["split"] for row in manifest_rows}
    for row in event_rows:
        if row["run_id"] not in run_ids or row["split"] != split_by_run[row["run_id"]]:
            raise ValueError("event/run split mismatch")
    total_samples = 0
    for row in manifest_rows:
        run_path = path / row["file"]
        if sha256_file(run_path) != row["run_file_sha256"]:
            raise ValueError(f"terrain run checksum mismatch: {row['run_id']}")
        with np.load(run_path, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
        samples = int(row["sample_count"])
        _validate_result_arrays(arrays, samples)
        total_samples += samples
    coverage = _coverage_summary(event_rows, manifest_rows)
    return {
        "dataset_id": metadata["dataset_id"],
        "run_count": len(manifest_rows),
        "event_index_count": len(event_rows),
        "total_sensor_samples": total_samples,
        "manifest_sha256": metadata["manifest_sha256"],
        "event_index_sha256": metadata["event_index_sha256"],
        "source_commit": metadata["source_commit"],
        "coverage": coverage,
    }


def collect_terrain_dataset(
    config_path: Path,
    policy_path: Path,
    *,
    output_path: Path | None = None,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Collect all frozen runs and atomically publish the Terrain dataset."""
    config = load_terrain_collection_config(config_path)
    if output_path is not None:
        config = replace(config, output_path=output_path.resolve())
    policy_path = policy_path.resolve()
    if sha256_file(policy_path) != config.policy_sha256:
        raise ValueError("policy SHA-256 differs from the terrain experiment")
    source_commit, worktree_clean = _git_source_commit(config.require_clean_worktree)
    final_path = config.output_path
    temporary_path = final_path.parent / f".{final_path.name}.tmp"
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite Terrain dataset: {final_path}")
    if temporary_path.exists():
        raise FileExistsError(f"incomplete Terrain dataset requires review: {temporary_path}")
    relative = final_path.relative_to(REPOSITORY_ROOT)
    if subprocess.run(
        ["git", "check-ignore", "--quiet", str(relative)],
        cwd=REPOSITORY_ROOT,
        check=False,
    ).returncode != 0:
        raise RuntimeError(f"Terrain dataset path is not Git ignored: {relative}")
    base = load_simulation_config(config.simulator_config_path)
    parity = validate_foot_imu_observer_parity(base, policy_path, progress)
    if not parity["passed"]:
        raise RuntimeError("Foot IMU observer changed matched simulation physics")
    temporary_path.mkdir(parents=True)
    (temporary_path / "runs").mkdir()
    manifest_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    try:
        for index, run in enumerate(config.runs, start=1):
            progress(f"[{index:03d}/{len(config.runs):03d}] {run.run_id}")
            simulation_config = simulation_config_for_terrain_run(
                base, run, config, policy_path
            )
            result = run_simulation(simulation_config, capture_state_trace=False)
            arrays = _result_arrays(result)
            _validate_result_arrays(arrays, simulation_config.expected_samples)
            run_path = temporary_path / "runs" / f"{run.run_id}.npz"
            np.savez_compressed(run_path, **arrays)
            run_hash = sha256_file(run_path)
            target_id = TERRAIN_CLASS_ORDER.index(run.target_terrain)
            target_contact = arrays["exact_terrain_contact"][:, :, target_id]
            first_target = _first_true(np.any(target_contact, axis=1))
            first_fall = result.metadata["first_fall_sample"]
            slip_present = bool(np.any(result.diagnostics.established_slip))
            sink_present = bool(
                np.max(result.diagnostics.support_surface_displacement_m) >= 0.001
                or np.any(result.diagnostics.sink_physical_active)
            )
            observed_fall = first_fall is not None
            manifest_rows.append(
                {
                    "run_id": run.run_id,
                    "file": f"runs/{run.run_id}.npz",
                    "split": run.split,
                    "condition_signature": run.condition_signature,
                    "source_terrain": run.source_terrain,
                    "target_terrain": run.target_terrain,
                    "intended_outcome": run.intended_outcome,
                    "observed_outcome": "fall" if observed_fall else "stable",
                    "speed_mps": run.speed_mps,
                    "patch_start_x_m": run.patch_start_x_m,
                    "patch_width_m": run.patch_width_m,
                    "slip_pattern": run.slip_pattern,
                    "sink_pattern": run.sink_pattern,
                    "sink_severity": run.sink_severity,
                    "support_pattern": run.support_pattern,
                    "sample_count": len(arrays["sequence"]),
                    "drop_count": int(result.metadata["dropped_samples"]),
                    "first_target_contact_us": "" if first_target is None else int(arrays["timestamp_us"][first_target]),
                    "first_fall_us": "" if first_fall is None else int(arrays["timestamp_us"][int(first_fall)]),
                    "pretransition_fall": bool(
                        first_fall is not None
                        and (first_target is None or int(first_fall) < first_target)
                    ),
                    "physical_slip_present": slip_present,
                    "physical_sink_present": sink_present,
                    "run_file_sha256": run_hash,
                }
            )
            event_rows.extend(
                build_touchdown_event_rows(
                    run.run_id,
                    run.split,
                    run.source_terrain,
                    run.target_terrain,
                    arrays["timestamp_us"],
                    arrays["exact_terrain_contact"],
                    None if first_fall is None else int(first_fall),
                    slip_present,
                    sink_present,
                    config.horizons_ms,
                    config.mixed_ratio_max_exclusive,
                )
            )
        _write_csv(temporary_path / "manifest.csv", RUN_MANIFEST_FIELDS, manifest_rows)
        _write_csv(temporary_path / "events.csv", EVENT_INDEX_FIELDS, event_rows)
        metadata = {
            "dataset_id": config.dataset_id,
            "schema_version": config.dataset_schema,
            "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
            "source_repository": "https://github.com/shinjuyeop/Infineon_FastReflex",
            "source_commit": source_commit,
            "generator_worktree_clean": worktree_clean,
            "experiment_config": str(config.config_path.relative_to(REPOSITORY_ROOT)),
            "experiment_config_sha256": sha256_file(config.config_path),
            "dataset_config": str(config.dataset_config_path.relative_to(REPOSITORY_ROOT)),
            "dataset_config_sha256": sha256_file(config.dataset_config_path),
            "frozen_transition_config": str(config.frozen_transition_config_path.relative_to(REPOSITORY_ROOT)),
            "frozen_transition_config_sha256": sha256_file(config.frozen_transition_config_path),
            "policy_sha256": config.policy_sha256,
            "run_count": len(config.runs),
            "sample_rate_hz": SENSOR_RATE_HZ,
            "storage_format": "one_complete_run_per_compressed_npz_plus_metadata_only_event_index",
            "runtime_input_fields": list(RUNTIME_INPUT_FIELDS),
            "model_input_fields": ["foot_fsr", "foot_imu"],
            "diagnostic_fields": list(DIAGNOSTIC_FIELDS),
            "diagnostic_fields_are_model_input": False,
            "terrain_class_order": list(TERRAIN_CLASS_NAMES),
            "foot_fsr_channel_order": list(FSR_CHANNELS),
            "foot_imu_channel_order": list(FOOT_IMU_CHANNELS),
            "foot_imu_site": {
                "left": {"body": "left_ankle_roll_link", "pos_m": [0.035, 0.0, -0.02], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
                "right": {"body": "right_ankle_roll_link", "pos_m": [0.035, 0.0, -0.02], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
            },
            "event_contract": {
                "horizons_ms": list(config.horizons_ms),
                "primary_window_ms": config.primary_window_ms,
                "mixed_ratio_max_exclusive": config.mixed_ratio_max_exclusive,
                "complete_pre_fall_window_required": True,
            },
            "split_frozen_before_simulation": True,
            "split_run_ids": {
                name: [run.run_id for run in config.runs if run.split == name]
                for name in ("train", "validation", "holdout")
            },
            "foot_imu_observer_parity": parity,
            "manifest_sha256": sha256_file(temporary_path / "manifest.csv"),
            "event_index_sha256": sha256_file(temporary_path / "events.csv"),
        }
        with (temporary_path / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        summary = validate_terrain_dataset(temporary_path)
        temporary_path.rename(final_path)
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    summary["size_bytes"] = sum(
        item.stat().st_size for item in final_path.rglob("*") if item.is_file()
    )
    summary["foot_imu_observer_parity"] = parity
    return final_path, summary
