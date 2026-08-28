"""Event-centric physical-disturbance dataset and reflex detector evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import gc
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from fastreflex.dataset.loader import Normalizer, WindowSet, extract_sensor_profile
from fastreflex.evaluation.stability_dense import (
    ReplayTrace,
    sustained_confirmation_sample,
    threshold_grid,
)
from fastreflex.evaluation.stability_temporal import (
    _file_sha256,
    _protected_hashes,
    binary_auprc,
    binary_auroc,
    predict_fall_probability,
)
from fastreflex.evaluation.transition_scenarios import (
    SIGNATURE_FIELDS,
    VALID_FALL,
    VALID_OUTCOMES,
    VALID_STABLE,
    classify_scenario_outcome,
    fusion_regression,
    target_contact_mask,
    transition_simulation_config,
)
from fastreflex.models.baselines import parameter_count
from fastreflex.simulation.g1 import (
    PHYSICS_TIMESTEP_S,
    SENSOR_RATE_HZ,
    SimulationConfig,
    SimulationResult,
    load_simulation_config,
    run_simulation,
)
from fastreflex.training.trainer import load_checkpoint, save_checkpoint, train_model


PELVIS_IMU6 = "PELVIS_IMU6"
PELVIS_IMU6_FSR8 = "PELVIS_IMU6_FSR8"
EVENT_REPRESENTATIONS = (PELVIS_IMU6, PELVIS_IMU6_FSR8)
EVENT_TYPE_NONE = "NONE"
EVENT_TYPE_SLIP = "SLIP"
EVENT_TYPE_SUPPORT = "SUPPORT"
EVENT_TYPE_BOTH = "SLIP_AND_SUPPORT"
EVENT_CLASS_NAMES = ("NORMAL", "REFLEX_EVENT")
EVENT_SIGNATURE_FIELDS = ("source_terrain", *SIGNATURE_FIELDS)
RUNTIME_FEATURE_NAMES = {
    PELVIS_IMU6: (
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ),
    PELVIS_IMU6_FSR8: (
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "left_front_left",
        "left_front_right",
        "left_rear_left",
        "left_rear_right",
        "right_front_left",
        "right_front_right",
        "right_rear_left",
        "right_rear_right",
    ),
}


@dataclass(frozen=True)
class EventRun:
    """One bounded run with runtime tensors and privileged event diagnostics."""

    run_id: str
    split: str
    source_terrain: str
    target_terrain: str
    design_role: str
    first_contact_sample: int
    first_touchdown_sample: int
    censor_sample: int
    outcome_diagnostic: str
    fall_sample_diagnostic: int | None
    features: Mapping[str, np.ndarray]
    timestamp_us: np.ndarray
    slip_event_samples_per_foot: tuple[int | None, int | None]
    support_event_samples_per_foot: tuple[int | None, int | None]
    event_sample: int | None
    event_type: str
    hard_stable_control: bool
    drift_m: np.ndarray
    tangential_velocity_mps: np.ndarray
    support_spread_m: np.ndarray
    support_max_displacement_m: np.ndarray
    loaded_contact: np.ndarray
    sink_pattern: str
    support_pattern: str


@dataclass(frozen=True)
class EventBatch:
    """Causal training windows with label provenance kept outside tensors."""

    windows: WindowSet
    rows: tuple[dict[str, object], ...]


class EventHoldoutGuard:
    """Allow one holdout waveform opening only after validation selection."""

    def __init__(self) -> None:
        self._opened = False
        self._open_count = 0

    def open_once(self) -> None:
        if self._opened or self._open_count:
            raise RuntimeError("event holdout may be opened exactly once")
        self._opened = True
        self._open_count = 1

    def require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("event holdout waveform access is sealed")

    @property
    def open_count(self) -> int:
        return self._open_count


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


def _load_yaml(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_true(values: np.ndarray) -> int | None:
    selected = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if not len(selected) else int(selected[0])


def _first_true_per_foot(values: np.ndarray) -> tuple[int | None, int | None]:
    array = np.asarray(values, dtype=bool)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("per-foot event trace must have shape [samples,2]")
    return tuple(_first_true(array[:, side]) for side in range(2))  # type: ignore[return-value]


def persistent_threshold_events(
    values: np.ndarray,
    valid: np.ndarray,
    episode_ids: np.ndarray,
    threshold: float,
    persistence_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a causal per-foot threshold with contact-episode reset."""
    metric = np.asarray(values, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    episodes = np.asarray(episode_ids, dtype=np.int64)
    if (
        metric.ndim != 2
        or metric.shape[1] != 2
        or validity.shape != metric.shape
        or episodes.shape != metric.shape
        or threshold < 0.0
        or persistence_samples <= 0
    ):
        raise ValueError("invalid persistent physical-event arrays or criteria")
    active = np.zeros(metric.shape, dtype=bool)
    onset = np.zeros(metric.shape, dtype=bool)
    for side in range(2):
        previous_episode = -2
        count = 0
        previous_active = False
        for sample in range(len(metric)):
            episode = int(episodes[sample, side])
            if episode != previous_episode:
                count = 0
                previous_active = False
                previous_episode = episode
            passes = bool(
                episode >= 0
                and validity[sample, side]
                and np.isfinite(metric[sample, side])
                and metric[sample, side] >= threshold
            )
            count = count + 1 if passes else 0
            current = bool(passes and count >= persistence_samples)
            active[sample, side] = current
            onset[sample, side] = current and not previous_active
            previous_active = current
    return active, onset


def union_event_clock(
    slip_onset: np.ndarray, support_onset: np.ndarray
) -> tuple[int | None, str]:
    """Return the first causal confirmation sample and union event metadata."""
    slip = _first_true(np.any(np.asarray(slip_onset, dtype=bool), axis=1))
    support = _first_true(np.any(np.asarray(support_onset, dtype=bool), axis=1))
    if slip is None and support is None:
        return None, EVENT_TYPE_NONE
    if slip is not None and support is not None:
        return min(slip, support), EVENT_TYPE_BOTH
    if slip is not None:
        return slip, EVENT_TYPE_SLIP
    assert support is not None
    return support, EVENT_TYPE_SUPPORT


def classify_event_detection(
    event_sample: int | None,
    detection_sample: int | None,
    minimum_latency_ms: int = -20,
    maximum_latency_ms: int = 50,
) -> str:
    """Classify a run's first sustained alert against the frozen event window."""
    if event_sample is None:
        return "NO_EVENT_FP" if detection_sample is not None else "NO_EVENT_TN"
    if detection_sample is None:
        return "EVENT_MISSED"
    latency = int(detection_sample) - int(event_sample)
    if latency < minimum_latency_ms:
        return "EVENT_PREMATURE_FP"
    if latency <= maximum_latency_ms:
        return "EVENT_VALID_DETECTION"
    return "EVENT_LATE"


def physical_signature(specification: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(specification[field] for field in EVENT_SIGNATURE_FIELDS)


def _scheduled_condition(
    dense_document: Mapping[str, object],
    source: str,
    target: str,
    role: str,
    index: int,
) -> dict[str, object]:
    schedule = dense_document["dataset"]["deterministic_condition_schedule"][
        f"{target}_{role}"
    ]
    source_key = f"{source}_width"
    if target == "ice" and role == "stable":
        width = float(schedule[source_key]["first"]) + (index - 1) * float(
            schedule[source_key]["step"]
        )
        start = float(schedule["patch_start_x_m"])
        sink_pattern = str(schedule["sink_pattern"])
    else:
        anchor_index = (index - 1) % 2
        local_index = (index - 1) // 2
        anchor = schedule["alternating_anchors"][anchor_index]
        start = float(anchor["patch_start_x_m"])
        width = float(anchor[f"{source}_width_first"]) + local_index * float(
            anchor["width_step"]
        )
        sink_pattern = str(
            anchor["sink_pattern"]
            if "sink_pattern" in anchor
            else schedule["sink_pattern"]
        )
    return {
        "patch_start_x_m": round(start, 5),
        "patch_width_m": round(width, 5),
        "slip_pattern": str(schedule["slip_pattern"]),
        "sink_pattern": sink_pattern,
        "sink_severity": str(schedule["sink_severity"]),
        "support_pattern": str(schedule["support_pattern"]),
    }


def generate_event_specifications(
    document: Mapping[str, object], dense_document: Mapping[str, object]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Expand 192 development, 48 fresh holdout, and 16 hard controls."""
    primary: list[dict[str, object]] = []
    ranges = (("train", 1, 18), ("validation", 19, 24), ("holdout", 31, 36))
    for source in document["dataset"]["primary_matrix"]["sources"]:
        for target in document["dataset"]["primary_matrix"]["targets"]:
            for role in document["dataset"]["primary_matrix"][
                "design_roles_for_condition_coverage_only"
            ]:
                for split, first, last in ranges:
                    for index in range(first, last + 1):
                        primary.append(
                            {
                                "id": f"evt_{str(source)[0]}_{target}_{str(role)[0]}{index:02d}",
                                "split": split,
                                "design_role": str(role),
                                "intended_role": str(role),
                                "source_terrain": str(source),
                                "target_terrain": str(target),
                                "speed_mps": float(document["common"]["primary_speed_mps"]),
                                "hard_stable_control": False,
                                **_scheduled_condition(
                                    dense_document,
                                    str(source),
                                    str(target),
                                    str(role),
                                    index,
                                ),
                            }
                        )
    controls: list[dict[str, object]] = []
    speeds = [
        float(value) for value in document["dataset"]["hard_controls"]["speed_mps"]
    ]
    for source in document["dataset"]["hard_controls"]["sources"]:
        for index, speed in enumerate(speeds, start=1):
            split = "train" if index <= 4 else "validation" if index <= 6 else "holdout"
            controls.append(
                {
                    "id": f"evt_control_{str(source)[0]}_{index:02d}",
                    "split": split,
                    "design_role": "hard_control",
                    "intended_role": "stable",
                    "source_terrain": str(source),
                    "target_terrain": str(source),
                    "speed_mps": speed,
                    "patch_start_x_m": 0.35,
                    "patch_width_m": 0.75,
                    "slip_pattern": "uniform",
                    "sink_pattern": "uniform",
                    "sink_severity": "moderate",
                    "support_pattern": "balanced_soft",
                    "hard_stable_control": True,
                }
            )
    return primary, controls


def audit_historical_dense_dataset(dataset_path: Path) -> dict[str, object]:
    """Check whether the historical dense files can reproduce both oracles."""
    manifest_path = dataset_path / "manifest.json"
    if not manifest_path.is_file():
        return {
            "dataset_present": False,
            "sufficient": False,
            "reason": "historical_dense_dataset_missing",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(manifest.get("runs", ()))
    if not rows:
        return {
            "dataset_present": True,
            "sufficient": False,
            "reason": "historical_dense_manifest_has_no_runs",
        }
    sample_path = dataset_path / str(rows[0]["file"])
    with np.load(sample_path, allow_pickle=False) as payload:
        fields = tuple(sorted(payload.files))
    required = {
        "pelvis_imu6",
        "foot_fsr8",
        "tangential_anchor_drift_m",
        "support_surface_spread_m",
        "contact_episode_id",
    }
    missing = tuple(sorted(required - set(fields)))
    return {
        "dataset_present": True,
        "dataset_id": manifest.get("dataset_id"),
        "run_count": len(rows),
        "sample_file": sample_path.name,
        "stored_fields": fields,
        "missing_required_fields": missing,
        "pelvis_imu6_available": "pelvis_imu6" in fields,
        "bilateral_fsr8_available": "foot_fsr8" in fields,
        "slip_clock_reproducible": {
            "tangential_anchor_drift_m",
            "contact_episode_id",
        }.issubset(fields),
        "support_clock_reproducible": {
            "support_surface_spread_m",
            "contact_episode_id",
        }.issubset(fields),
        "exact_scenario_provenance": all(
            "physical_signature" in row for row in rows
        ),
        "sufficient": not missing,
        "reason": (
            "sufficient_for_event_relabel"
            if not missing
            else "new_dataset_required_without_modifying_historical_dense"
        ),
    }


def validate_event_design(
    document: Mapping[str, object],
    dense_document: Mapping[str, object],
    primary: Sequence[Mapping[str, object]],
    controls: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Fail before simulation if a declared physical/data contract drifted."""
    if document["experiment"]["id"] != "EVENT_CENTRIC_REFLEX_TRIGGER_DEVELOPMENT":
        raise ValueError("unsupported event-centric experiment")
    slip = document["physical_oracles"]["slip"]
    support = document["physical_oracles"]["support"]
    if float(slip["threshold_m"]) != 0.050 or int(slip["persistence_ms"]) != 3:
        raise ValueError("frozen 50 mm / 3 ms Slip criterion changed")
    if slip["primary_aggregation"] != "any_foot":
        raise ValueError("primary Slip event must aggregate either foot")
    if float(support["threshold_m"]) != 0.010 or int(
        support["persistence_ms"]
    ) != 20:
        raise ValueError("frozen 10 mm / 20 ms support criterion changed")
    if tuple(document["representations"]) != EVENT_REPRESENTATIONS:
        raise ValueError("exactly the two frozen runtime representations are required")
    if tuple(document["windows"]["histories_ms"]) != (20, 50):
        raise ValueError("event history candidates changed")
    if tuple(document["windows"]["model_families"]) != ("mlp", "gru"):
        raise ValueError("event model candidates changed")
    if int(document["windows"]["endpoint_stride_ms"]) != 5:
        raise ValueError("event training stride changed")
    if tuple(document["windows"]["positive_endpoint_interval_ms"]) != (-10, 50):
        raise ValueError("event positive interval changed")
    if int(document["threshold_calibration"]["detector_persistence_ms"]) != 5:
        raise ValueError("runtime detector persistence changed")
    if tuple(document["model"]["seeds"]) != (20260828, 20260829, 20260830):
        raise ValueError("event model seeds changed")
    if len(primary) != 240 or len(controls) != 16:
        raise ValueError("event cohort must contain 240 transitions and 16 controls")
    ids = [str(row["id"]) for row in [*primary, *controls]]
    if len(ids) != len(set(ids)):
        raise ValueError("event run ids are duplicated")
    signatures = [physical_signature(row) for row in primary]
    duplicates = len(signatures) - len(set(signatures))
    if duplicates:
        raise ValueError("event physical signatures are duplicated")
    split_ids = {
        split: {str(row["id"]) for row in primary if row["split"] == split}
        for split in ("train", "validation", "holdout")
    }
    split_counts = {key: len(value) for key, value in split_ids.items()}
    if split_counts != {"train": 144, "validation": 48, "holdout": 48}:
        raise ValueError("event primary split counts changed")
    if any(
        split_ids[left] & split_ids[right]
        for left, right in (
            ("train", "validation"),
            ("train", "holdout"),
            ("validation", "holdout"),
        )
    ):
        raise ValueError("event split is not run-disjoint")
    old_dense_signatures = {
        physical_signature(
            {
                "source_terrain": str(source),
                "target_terrain": str(target),
                "speed_mps": float(document["common"]["primary_speed_mps"]),
                **_scheduled_condition(
                    dense_document,
                    str(source),
                    str(target),
                    str(role),
                    index,
                ),
            }
        )
        for source in document["dataset"]["primary_matrix"]["sources"]
        for target in document["dataset"]["primary_matrix"]["targets"]
        for role in document["dataset"]["primary_matrix"][
            "design_roles_for_condition_coverage_only"
        ]
        for index in range(1, 31)
    }
    old_development = {
        physical_signature(row)
        for row in primary
        if row["split"] in ("train", "validation")
    }
    holdout = {
        physical_signature(row) for row in primary if row["split"] == "holdout"
    }
    if old_development & holdout:
        raise ValueError("fresh event holdout overlaps development")
    if old_dense_signatures & holdout:
        raise ValueError("fresh event holdout overlaps the historical dense corpus")
    forbidden = tuple(str(value) for value in document["common"]["forbidden_model_inputs"])
    for names in RUNTIME_FEATURE_NAMES.values():
        if any(token in name for token in forbidden for name in names):
            raise ValueError("event runtime tensor contains a forbidden field")
    return {
        "passed": True,
        "primary_runs": len(primary),
        "hard_controls": len(controls),
        "split_counts": split_counts,
        "hard_control_split_counts": {
            split: sum(row["split"] == split for row in controls)
            for split in ("train", "validation", "holdout")
        },
        "duplicate_signatures": duplicates,
        "split_overlap": 0,
        "holdout_development_signature_overlap": 0,
        "holdout_historical_dense_signature_overlap": 0,
        "fresh_holdout_runs": len(split_ids["holdout"]),
        "representation_dimensions": {
            name: len(RUNTIME_FEATURE_NAMES[name]) for name in EVENT_REPRESENTATIONS
        },
        "fall_or_terrain_fields_in_runtime_tensor": False,
    }


def _hard_control_outcome(result: SimulationResult) -> str:
    finite = bool(
        np.all(np.isfinite(result.runtime.pelvis_imu))
        and result.runtime.foot_fsr is not None
        and np.all(np.isfinite(result.runtime.foot_fsr))
        and result.metadata["actual_samples"] == result.metadata["expected_samples"]
        and not result.metadata["terminated_by_viewer"]
    )
    if not finite or result.metadata["first_fall_sample"] is not None:
        return "INVALID_CONTROL"
    return VALID_STABLE


def _target_contact_samples(
    result: SimulationResult, target: str
) -> tuple[int, int]:
    contact = target_contact_mask(result, target)
    first = _first_true(np.any(contact, axis=1))
    touchdown = _first_true(np.any(contact & result.diagnostics.touchdown, axis=1))
    if first is None:
        raise ValueError(f"valid event run has no exact {target} contact")
    return first, first if touchdown is None else touchdown


def _reduce_simulation(
    specification: Mapping[str, object],
    result: SimulationResult,
    outcome: str,
) -> EventRun:
    fsr = result.runtime.foot_fsr
    if fsr is None:
        raise ValueError("event dataset requires bilateral FSR8")
    control = bool(specification["hard_stable_control"])
    if control:
        contact, touchdown = 0, 0
    else:
        contact, touchdown = _target_contact_samples(
            result, str(specification["target_terrain"])
        )
    fall_raw = result.metadata["first_fall_sample"]
    fall = None if fall_raw is None else int(fall_raw)
    censor = len(result.runtime.sequence) if fall is None else fall
    slip_onset = np.asarray(
        result.diagnostics.established_slip_after_patch_onset, dtype=bool
    )
    support_onset = np.asarray(result.diagnostics.deformable_sink_onset, dtype=bool)
    if control:
        slip_onset = np.zeros_like(slip_onset)
        support_onset = np.zeros_like(support_onset)
    event_sample, event_type = union_event_clock(slip_onset, support_onset)
    if event_sample is not None and event_sample >= censor:
        raise ValueError("physical event must be causally confirmed before censor")
    imu = np.asarray(result.runtime.pelvis_imu, dtype=np.float32).copy()
    fsr_copy = np.asarray(fsr, dtype=np.float32).copy()
    fusion = extract_sensor_profile(imu, fsr_copy, "fusion14")
    if (
        imu.shape != (len(result.runtime.sequence), 6)
        or fsr_copy.shape != (len(result.runtime.sequence), 8)
        or not np.all(np.isfinite(imu))
        or not np.all(np.isfinite(fsr_copy))
    ):
        raise ValueError("event runtime tensors are nonfinite or malformed")
    return EventRun(
        run_id=str(specification["id"]),
        split=str(specification["split"]),
        source_terrain=str(specification["source_terrain"]),
        target_terrain=str(specification["target_terrain"]),
        design_role=str(specification["design_role"]),
        first_contact_sample=contact,
        first_touchdown_sample=touchdown,
        censor_sample=censor,
        outcome_diagnostic=outcome,
        fall_sample_diagnostic=fall,
        features={PELVIS_IMU6: imu, PELVIS_IMU6_FSR8: fusion},
        timestamp_us=np.asarray(result.runtime.timestamp_us, dtype=np.int64).copy(),
        slip_event_samples_per_foot=_first_true_per_foot(slip_onset),
        support_event_samples_per_foot=_first_true_per_foot(support_onset),
        event_sample=event_sample,
        event_type=event_type,
        hard_stable_control=control,
        drift_m=np.asarray(
            result.diagnostics.tangential_anchor_drift_m, dtype=np.float32
        ).copy(),
        tangential_velocity_mps=np.asarray(
            result.diagnostics.tangential_velocity_mps, dtype=np.float32
        ).copy(),
        support_spread_m=np.asarray(
            result.diagnostics.support_surface_spread_m, dtype=np.float32
        ).copy(),
        support_max_displacement_m=np.asarray(
            result.diagnostics.support_surface_max_displacement_m,
            dtype=np.float32,
        ).copy(),
        loaded_contact=np.asarray(
            result.diagnostics.loaded_contact, dtype=bool
        ).copy(),
        sink_pattern=str(specification["sink_pattern"]),
        support_pattern=str(specification["support_pattern"]),
    )


def simulate_event_cohort(
    base: SimulationConfig,
    specifications: Sequence[Mapping[str, object]],
    policy_path: Path,
    document: Mapping[str, object],
    progress: Callable[[str], None],
) -> tuple[dict[str, EventRun], list[dict[str, object]]]:
    """Simulate sequentially and retain runtime sensors plus frozen event clocks."""
    runs: dict[str, EventRun] = {}
    invalid: list[dict[str, object]] = []
    duration_s = float(document["common"]["duration_s"])
    gate = document["common"]["scenario_gate"]
    for index, raw in enumerate(specifications, start=1):
        specification = dict(raw)
        specification["minimum_normal_prefix_ms"] = int(gate["normal_prefix_ms_min"])
        specification["minimum_post_contact_ms"] = int(gate["post_contact_ms_min"])
        result = run_simulation(
            transition_simulation_config(
                base, specification, policy_path, duration_s
            ),
            observe_fsr=True,
            observe_foot_imu=False,
            capture_state_trace=False,
        )
        outcome = (
            _hard_control_outcome(result)
            if specification["hard_stable_control"]
            else classify_scenario_outcome(result, specification)
        )
        progress(
            f"REFLEX COHORT {index}/{len(specifications)} "
            f"{specification['id']}: {outcome}"
        )
        if outcome not in VALID_OUTCOMES:
            invalid.append(
                {
                    "run_id": str(specification["id"]),
                    "split": str(specification["split"]),
                    "outcome": outcome,
                }
            )
        else:
            reduced = _reduce_simulation(specification, result, outcome)
            runs[reduced.run_id] = reduced
        del result
        if index % 8 == 0:
            gc.collect()
    return runs, invalid


def _event_run_to_npz(path: Path, run: EventRun) -> None:
    np.savez_compressed(
        path,
        timestamp_us=run.timestamp_us,
        pelvis_imu6=run.features[PELVIS_IMU6],
        foot_fsr8=run.features[PELVIS_IMU6_FSR8][:, 6:],
        tangential_anchor_drift_m=run.drift_m,
        tangential_velocity_mps=run.tangential_velocity_mps,
        support_surface_spread_m=run.support_spread_m,
        support_surface_max_displacement_m=run.support_max_displacement_m,
        loaded_contact=run.loaded_contact,
        first_target_contact_sample=np.asarray(
            run.first_contact_sample, dtype=np.int64
        ),
        first_target_touchdown_sample=np.asarray(
            run.first_touchdown_sample, dtype=np.int64
        ),
        censor_sample=np.asarray(run.censor_sample, dtype=np.int64),
        first_slip_event_sample_per_foot=np.asarray(
            [-1 if value is None else value for value in run.slip_event_samples_per_foot],
            dtype=np.int64,
        ),
        first_support_event_sample_per_foot=np.asarray(
            [
                -1 if value is None else value
                for value in run.support_event_samples_per_foot
            ],
            dtype=np.int64,
        ),
        first_reflex_event_sample=np.asarray(
            -1 if run.event_sample is None else run.event_sample, dtype=np.int64
        ),
    )


def _finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return 0.0 if not len(finite) else float(np.max(finite))


def _event_manifest_row(
    path: Path, run: EventRun, specification: Mapping[str, object]
) -> dict[str, object]:
    event = run.event_sample
    at_event_drift = (
        None
        if event is None
        else _finite_max(run.drift_m[event : event + 1])
    )
    at_event_spread = (
        None
        if event is None
        else _finite_max(run.support_spread_m[event : event + 1])
    )
    slip_feet = [value is not None for value in run.slip_event_samples_per_foot]
    return {
        "run_id": run.run_id,
        "file": path.name,
        "file_sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "split": run.split,
        "source_terrain": run.source_terrain,
        "target_terrain": run.target_terrain,
        "speed_mps": float(specification["speed_mps"]),
        "design_role_diagnostic_only": run.design_role,
        "observed_outcome_diagnostic_only": run.outcome_diagnostic,
        "fall_sample_diagnostic_only": run.fall_sample_diagnostic,
        "hard_stable_control": run.hard_stable_control,
        "first_target_contact_sample": run.first_contact_sample,
        "first_target_touchdown_sample": run.first_touchdown_sample,
        "censor_sample": run.censor_sample,
        "event_sample": run.event_sample,
        "event_type": run.event_type,
        "slip_event_samples_per_foot": list(run.slip_event_samples_per_foot),
        "support_event_samples_per_foot": list(run.support_event_samples_per_foot),
        "slip_side": (
            "BILATERAL"
            if all(slip_feet)
            else "LEFT"
            if slip_feet[0]
            else "RIGHT"
            if slip_feet[1]
            else "NONE"
        ),
        "drift_at_union_event_m": at_event_drift,
        "support_spread_at_union_event_m": at_event_spread,
        "peak_drift_m": _finite_max(run.drift_m[: run.censor_sample]),
        "peak_tangential_velocity_mps": _finite_max(
            run.tangential_velocity_mps[: run.censor_sample]
        ),
        "peak_support_spread_m": _finite_max(
            run.support_spread_m[: run.censor_sample]
        ),
        "maximum_support_deformation_m": _finite_max(
            run.support_max_displacement_m[: run.censor_sample]
        ),
        "sink_pattern": run.sink_pattern,
        "support_pattern": run.support_pattern,
        "physical_signature": list(physical_signature(specification)),
    }


def _dataset_manifest(
    document: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    dataset_path: Path,
) -> tuple[dict[str, object], str]:
    manifest = {
        "schema_version": 1,
        "dataset_id": document["dataset"]["dataset_id"],
        "created_at": f"{document['experiment']['date']}T00:00:00+09:00",
        "source_commit": document["experiment"]["source_commit_at_start"],
        "policy_sha256": document["source"]["policy_sha256"],
        "simulator_config_sha256": document["source"]["simulator_config_sha256"],
        "scenario_calibration_reference": {
            "path": document["source"]["scenario_calibration_config"],
            "sha256": document["source"]["scenario_calibration_sha256"],
        },
        "dense_condition_schedule_reference": {
            "path": document["source"]["dense_design_config"],
            "sha256": document["source"]["dense_design_sha256"],
        },
        "physical_oracles": document["physical_oracles"],
        "model_input_fields": {
            PELVIS_IMU6: ["pelvis_imu6"],
            PELVIS_IMU6_FSR8: ["pelvis_imu6", "foot_fsr8"],
        },
        "fall_outcome_role": "diagnostic_only_never_label_or_tensor",
        "run_count": len(rows),
        "runs": list(rows),
    }
    path = dataset_path / "manifest.json"
    _write_json(path, manifest)
    sha = _file_sha256(path)
    (dataset_path / "manifest.sha256").write_text(
        f"{sha}  manifest.json\n", encoding="utf-8"
    )
    return manifest, sha


def _load_event_run(path: Path, row: Mapping[str, object]) -> EventRun:
    with np.load(path, allow_pickle=False) as payload:
        timestamp = np.asarray(payload["timestamp_us"], dtype=np.int64)
        imu = np.asarray(payload["pelvis_imu6"], dtype=np.float32)
        fsr = np.asarray(payload["foot_fsr8"], dtype=np.float32)
        drift = np.asarray(payload["tangential_anchor_drift_m"], dtype=np.float32)
        velocity = np.asarray(payload["tangential_velocity_mps"], dtype=np.float32)
        spread = np.asarray(payload["support_surface_spread_m"], dtype=np.float32)
        deformation = np.asarray(
            payload["support_surface_max_displacement_m"], dtype=np.float32
        )
        loaded = np.asarray(payload["loaded_contact"], dtype=bool)
        contact = int(payload["first_target_contact_sample"])
        touchdown = int(payload["first_target_touchdown_sample"])
        censor = int(payload["censor_sample"])
        slip_raw = np.asarray(
            payload["first_slip_event_sample_per_foot"], dtype=np.int64
        )
        support_raw = np.asarray(
            payload["first_support_event_sample_per_foot"], dtype=np.int64
        )
        event_raw = int(payload["first_reflex_event_sample"])
    samples = len(timestamp)
    if (
        imu.shape != (samples, 6)
        or fsr.shape != (samples, 8)
        or drift.shape != (samples, 2)
        or velocity.shape != (samples, 2)
        or spread.shape != (samples, 2)
        or deformation.shape != (samples, 2)
        or loaded.shape != (samples, 2)
        or not np.all(np.isfinite(imu))
        or not np.all(np.isfinite(fsr))
        or np.any(fsr < 0.0)
        or not (0 <= contact < censor <= samples)
    ):
        raise ValueError(f"event run {row['run_id']} contains invalid tensors")
    fusion = extract_sensor_profile(imu, fsr, "fusion14")
    event = None if event_raw < 0 else event_raw
    if event != row["event_sample"]:
        raise ValueError("event clock differs between manifest and run payload")
    return EventRun(
        run_id=str(row["run_id"]),
        split=str(row["split"]),
        source_terrain=str(row["source_terrain"]),
        target_terrain=str(row["target_terrain"]),
        design_role=str(row["design_role_diagnostic_only"]),
        first_contact_sample=contact,
        first_touchdown_sample=touchdown,
        censor_sample=censor,
        outcome_diagnostic=str(row["observed_outcome_diagnostic_only"]),
        fall_sample_diagnostic=(
            None
            if row["fall_sample_diagnostic_only"] is None
            else int(row["fall_sample_diagnostic_only"])
        ),
        features={PELVIS_IMU6: imu, PELVIS_IMU6_FSR8: fusion},
        timestamp_us=timestamp,
        slip_event_samples_per_foot=tuple(
            None if value < 0 else int(value) for value in slip_raw
        ),  # type: ignore[arg-type]
        support_event_samples_per_foot=tuple(
            None if value < 0 else int(value) for value in support_raw
        ),  # type: ignore[arg-type]
        event_sample=event,
        event_type=str(row["event_type"]),
        hard_stable_control=bool(row["hard_stable_control"]),
        drift_m=drift,
        tangential_velocity_mps=velocity,
        support_spread_m=spread,
        support_max_displacement_m=deformation,
        loaded_contact=loaded,
        sink_pattern=str(row["sink_pattern"]),
        support_pattern=str(row["support_pattern"]),
    )


def load_event_runs(
    dataset_path: Path,
    manifest: Mapping[str, object],
    splits: Sequence[str],
    *,
    holdout_guard: EventHoldoutGuard | None = None,
) -> dict[str, EventRun]:
    """Load selected waveforms while enforcing the one-shot holdout guard."""
    if "holdout" in splits:
        if holdout_guard is None:
            raise RuntimeError("holdout loading requires an explicit event guard")
        holdout_guard.require_open()
    runs = {}
    for row in manifest["runs"]:
        if str(row["split"]) not in splits:
            continue
        path = dataset_path / str(row["file"])
        if _file_sha256(path) != str(row["file_sha256"]):
            raise ValueError(f"event dataset run integrity failed: {path.name}")
        run = _load_event_run(path, row)
        runs[run.run_id] = run
    return runs


def _cohort_summary(
    runs: Mapping[str, EventRun], invalid: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    primary = [run for run in runs.values() if not run.hard_stable_control]
    controls = [run for run in runs.values() if run.hard_stable_control]
    slip = [
        run
        for run in primary
        if run.event_type in (EVENT_TYPE_SLIP, EVENT_TYPE_BOTH)
    ]
    support = [
        run
        for run in primary
        if run.event_type in (EVENT_TYPE_SUPPORT, EVENT_TYPE_BOTH)
    ]
    no_event = [run for run in primary if run.event_sample is None]
    event = [run for run in primary if run.event_sample is not None]
    slip_ids = {run.run_id for run in slip}
    support_ids = {run.run_id for run in support}
    return {
        "primary_runs": len(primary),
        "event_runs": len(event),
        "no_event_runs": len(no_event),
        "slip_event_runs": len(slip),
        "support_event_runs": len(support),
        "slip_and_support_runs": sum(run.event_type == EVENT_TYPE_BOTH for run in primary),
        "sand_benign_no_event_runs": sum(
            run.target_terrain == "sand" and run.event_sample is None
            for run in primary
        ),
        "hard_ground_no_event_runs": sum(run.event_sample is None for run in controls),
        "bilateral_slip_runs": sum(
            all(value is not None for value in run.slip_event_samples_per_foot)
            for run in slip
        ),
        "left_slip_runs": sum(run.slip_event_samples_per_foot[0] is not None for run in slip),
        "right_slip_runs": sum(run.slip_event_samples_per_foot[1] is not None for run in slip),
        "event_outcome": {
            "recovered": sum(run.outcome_diagnostic == VALID_STABLE for run in event),
            "fall": sum(run.outcome_diagnostic == VALID_FALL for run in event),
        },
        "by_split": {
            split: {
                "total": sum(run.split == split for run in primary),
                "event": sum(run.split == split and run.event_sample is not None for run in primary),
                "no_event": sum(run.split == split and run.event_sample is None for run in primary),
                "slip": sum(
                    run.split == split and run.run_id in slip_ids for run in primary
                ),
                "support": sum(
                    run.split == split and run.run_id in support_ids for run in primary
                ),
            }
            for split in ("train", "validation", "holdout")
        },
        "by_source": {
            source: {
                "event": sum(run.source_terrain == source and run.event_sample is not None for run in primary),
                "no_event": sum(run.source_terrain == source and run.event_sample is None for run in primary),
            }
            for source in ("concrete", "marble")
        },
        "invalid": list(invalid),
        "invalid_count": len(invalid),
    }


def readiness_results(
    cohort: Mapping[str, object],
    design: Mapping[str, object],
    gates: Mapping[str, object],
) -> dict[str, bool]:
    source = cohort["by_source"]
    by_split = cohort["by_split"]
    return {
        "slip_event_runs": int(cohort["slip_event_runs"]) >= int(gates["slip_event_runs_min"]),
        "support_event_runs": int(cohort["support_event_runs"]) >= int(gates["support_event_runs_min"]),
        "sand_benign_no_event_runs": int(cohort["sand_benign_no_event_runs"]) >= int(gates["sand_benign_no_event_runs_min"]),
        "hard_ground_no_event_runs": int(cohort["hard_ground_no_event_runs"]) >= int(gates["hard_ground_no_event_runs_min"]),
        "pre_event_negative_slip": all(int(by_split[split]["slip"]) > 0 for split in ("train", "validation")),
        "pre_event_negative_support": all(int(by_split[split]["support"]) > 0 for split in ("train", "validation")),
        "concrete_origin": int(source["concrete"]["event"]) > 0,
        "marble_origin": int(source["marble"]["event"]) > 0,
        "left_right_slip": int(cohort["left_slip_runs"]) > 0 and int(cohort["right_slip_runs"]) > 0,
        "duplicate_signature": int(design["duplicate_signatures"]) <= int(gates["duplicate_signature_max"]),
        "split_overlap": int(design["split_overlap"]) <= int(gates["split_overlap_max"]),
        "nonfinite_input": True,
        "invalid_runs": int(cohort["invalid_count"]) == 0,
    }


def fit_event_normalizer(
    runs: Mapping[str, EventRun],
    train_ids: Sequence[str],
    representation: str,
    per_run_sample_cap: int,
    standard_deviation_floor: float,
) -> Normalizer:
    """Fit channel scales from train-run post-contact, pre-censor samples only."""
    chunks = []
    fit_ids = []
    for run_id in train_ids:
        run = runs[str(run_id)]
        eligible = np.arange(
            run.first_contact_sample, run.censor_sample, dtype=np.int64
        )
        if len(eligible) > per_run_sample_cap:
            positions = np.linspace(
                0, len(eligible) - 1, per_run_sample_cap, dtype=np.int64
            )
            eligible = eligible[positions]
        if len(eligible):
            chunks.append(run.features[representation][eligible].astype(np.float64))
            fit_ids.append(run.run_id)
    if not chunks:
        raise ValueError("event train-only normalization has no eligible samples")
    values = np.concatenate(chunks)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < standard_deviation_floor] = 1.0
    return Normalizer(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        sample_count=len(values),
        fit_run_ids=tuple(fit_ids),
        epsilon=float(standard_deviation_floor),
    )


def _normalizer_transform(normalizer: Normalizer, values: np.ndarray) -> np.ndarray:
    transformed = normalizer.transform(values)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("normalized event tensor is nonfinite")
    return transformed.astype(np.float32, copy=False)


def _causal_indices(endpoint: int, history_ms: int) -> np.ndarray:
    first = int(endpoint) - int(history_ms) + 1
    if first < 0:
        raise ValueError("event endpoint has insufficient causal history")
    return np.arange(first, int(endpoint) + 1, dtype=np.int64)


def _evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or not len(values):
        return np.empty(0, dtype=np.int64)
    if len(values) <= count:
        return values.astype(np.int64, copy=True)
    positions = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return values[positions].astype(np.int64, copy=False)


def event_positive_endpoints(
    run: EventRun,
    history_ms: int,
    stride_ms: int = 5,
    minimum_offset_ms: int = -10,
    maximum_offset_ms: int = 50,
) -> np.ndarray:
    """Return bounded event-local endpoints, never an unbounded post-event tail."""
    if run.event_sample is None:
        return np.empty(0, dtype=np.int64)
    endpoints = np.arange(
        run.event_sample + minimum_offset_ms,
        run.event_sample + maximum_offset_ms + 1,
        stride_ms,
        dtype=np.int64,
    )
    first = run.first_contact_sample + history_ms - 1
    return endpoints[(endpoints >= first) & (endpoints < run.censor_sample)]


def event_early_negative_endpoints(
    run: EventRun,
    history_ms: int,
    count: int,
    stride_ms: int = 5,
    latest_offset_ms: int = -30,
) -> np.ndarray:
    """Sample deterministic event-run controls no later than event-30 ms."""
    if run.event_sample is None:
        return np.empty(0, dtype=np.int64)
    first = run.first_contact_sample + history_ms - 1
    stop = run.event_sample + latest_offset_ms + 1
    candidates = np.arange(first, stop, stride_ms, dtype=np.int64)
    result = _evenly_spaced(candidates, count)
    if len(result) and int(result[-1]) > run.event_sample + latest_offset_ms:
        raise ValueError("event early negative crossed the frozen boundary")
    return result


def build_event_windows(
    runs: Mapping[str, EventRun],
    run_ids: Sequence[str],
    representation: str,
    history_ms: int,
    normalizer: Normalizer,
    *,
    stride_ms: int = 5,
    positive_cap: int = 13,
    negative_cap: int = 13,
) -> EventBatch:
    """Build event-local positives and time-matched no-event negatives."""
    selected = [runs[str(run_id)] for run_id in run_ids if str(run_id) in runs]
    event_runs = sorted(
        (run for run in selected if run.event_sample is not None),
        key=lambda run: run.run_id,
    )
    no_event_runs = sorted(
        (run for run in selected if run.event_sample is None),
        key=lambda run: run.run_id,
    )
    inputs: list[np.ndarray] = []
    targets: list[int] = []
    source_ids: list[str] = []
    endpoints: list[int] = []
    rows: list[dict[str, object]] = []
    elapsed_by_target: dict[str, list[int]] = {}

    def append(run: EventRun, endpoint: int, label: int, kind: str) -> None:
        indices = _causal_indices(endpoint, history_ms)
        if indices[0] < run.first_contact_sample or indices[-1] >= run.censor_sample:
            raise ValueError("event window crossed contact or censor boundary")
        raw = run.features[representation][indices]
        inputs.append(_normalizer_transform(normalizer, raw))
        targets.append(label)
        source_ids.append(run.run_id)
        endpoints.append(endpoint)
        rows.append(
            {
                "run_id": run.run_id,
                "endpoint_sample": endpoint,
                "label": label,
                "kind": kind,
                "event_type": run.event_type,
                "elapsed_since_contact_ms": endpoint - run.first_contact_sample,
                "event_offset_ms": (
                    None if run.event_sample is None else endpoint - run.event_sample
                ),
            }
        )

    for run in event_runs:
        positive = event_positive_endpoints(run, history_ms, stride_ms)[:positive_cap]
        negative = event_early_negative_endpoints(
            run, history_ms, min(len(positive), negative_cap), stride_ms
        )
        for endpoint in positive:
            append(run, int(endpoint), 1, "event_positive")
            elapsed_by_target.setdefault(run.target_terrain, []).append(
                int(endpoint - run.first_contact_sample)
            )
        for endpoint in negative:
            append(run, int(endpoint), 0, "event_pre_negative")

    all_elapsed = [value for values in elapsed_by_target.values() for value in values]
    for run in no_event_runs:
        elapsed = elapsed_by_target.get(run.target_terrain, all_elapsed)
        candidates = np.asarray(sorted(set(elapsed)), dtype=np.int64)
        candidates = _evenly_spaced(candidates, negative_cap)
        matched = run.first_contact_sample + candidates
        matched = matched[
            (matched >= run.first_contact_sample + history_ms - 1)
            & (matched < run.censor_sample)
        ]
        if not len(matched):
            fallback = np.arange(
                run.first_contact_sample + history_ms - 1,
                run.censor_sample,
                stride_ms,
                dtype=np.int64,
            )
            matched = _evenly_spaced(fallback, negative_cap)
        for endpoint in matched[:negative_cap]:
            append(run, int(endpoint), 0, "no_event_time_matched_negative")
    if not inputs or len(set(targets)) != 2:
        raise ValueError("event windows must contain both binary classes")
    target_array = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(target_array, minlength=3)
    return EventBatch(
        windows=WindowSet(
            inputs=np.stack(inputs).astype(np.float32),
            targets=target_array,
            run_ids=np.asarray(source_ids, dtype=str),
            endpoint_samples=np.asarray(endpoints, dtype=np.int64),
            available_by_class=tuple(int(value) for value in counts[:3]),
        ),
        rows=tuple(rows),
    )


def _predict_replay(
    run: EventRun,
    representation: str,
    history_ms: int,
    normalizer: Normalizer,
    models: Sequence[torch.nn.Module],
    batch_size: int = 1024,
) -> ReplayTrace:
    start = run.first_contact_sample + history_ms - 1
    endpoints = np.arange(start, run.censor_sample, dtype=np.int64)
    if not len(endpoints):
        return ReplayTrace(endpoints=endpoints, probabilities=np.empty(0))
    offsets = np.arange(history_ms - 1, -1, -1, dtype=np.int64)
    outputs = []
    features = run.features[representation]
    for first in range(0, len(endpoints), batch_size):
        selected = endpoints[first : first + batch_size]
        indices = selected[:, None] - offsets[None, :]
        windows = _normalizer_transform(normalizer, features[indices])
        tensor = torch.from_numpy(windows)
        probabilities = []
        with torch.no_grad():
            for model in models:
                model.eval()
                probabilities.append(
                    torch.softmax(model(tensor), dim=1)[:, 1].cpu().numpy()
                )
        outputs.append(np.mean(np.stack(probabilities), axis=0))
    return ReplayTrace(
        endpoints=endpoints,
        probabilities=np.concatenate(outputs).astype(np.float64, copy=False),
    )


def _load_models(paths: Sequence[Path]) -> list[torch.nn.Module]:
    return [load_checkpoint(path)[0] for path in paths]


def _replay_many(
    runs: Mapping[str, EventRun],
    representation: str,
    history_ms: int,
    normalizer: Normalizer,
    checkpoint_paths: Sequence[Path],
) -> dict[str, ReplayTrace]:
    models = _load_models(checkpoint_paths)
    traces = {
        run_id: _predict_replay(
            run, representation, history_ms, normalizer, models
        )
        for run_id, run in runs.items()
    }
    del models
    return traces


def _latency_distribution(values: Sequence[int]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p10", "median", "p95", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def evaluate_event_runs(
    runs: Mapping[str, EventRun],
    traces: Mapping[str, ReplayTrace],
    threshold: float,
    persistence_ms: int,
) -> dict[str, object]:
    """Evaluate event recall, false alerts, and physical-clock latency by run."""
    event_rows = []
    no_event_rows = []
    hard_rows = []
    latencies = []
    for run_id, trace in traces.items():
        run = runs[run_id]
        detection = sustained_confirmation_sample(
            trace.endpoints, trace.probabilities, threshold, persistence_ms
        )
        classification = classify_event_detection(run.event_sample, detection)
        row = {
            "run_id": run_id,
            "source_terrain": run.source_terrain,
            "target_terrain": run.target_terrain,
            "event_type": run.event_type,
            "event_sample": run.event_sample,
            "detection_sample": detection,
            "classification": classification,
            "outcome_diagnostic_only": run.outcome_diagnostic,
            "slip_side": (
                "BILATERAL"
                if all(value is not None for value in run.slip_event_samples_per_foot)
                else "LEFT"
                if run.slip_event_samples_per_foot[0] is not None
                else "RIGHT"
                if run.slip_event_samples_per_foot[1] is not None
                else "NONE"
            ),
            "peak_drift_mm": 1000.0 * _finite_max(run.drift_m[: run.censor_sample]),
            "peak_support_spread_mm": 1000.0
            * _finite_max(run.support_spread_m[: run.censor_sample]),
            "maximum_support_deformation_mm": 1000.0
            * _finite_max(run.support_max_displacement_m[: run.censor_sample]),
        }
        if run.event_sample is not None and detection is not None:
            row["latency_ms"] = int(detection - run.event_sample)
        if classification == "EVENT_VALID_DETECTION":
            assert detection is not None and run.event_sample is not None
            latencies.append(int(detection - run.event_sample))
        if run.hard_stable_control:
            hard_rows.append(row)
        elif run.event_sample is None:
            if detection is not None:
                row["first_fp_elapsed_since_contact_ms"] = int(
                    detection - run.first_contact_sample
                )
                row["fp_duration_ms"] = int(
                    np.count_nonzero(trace.probabilities >= threshold)
                )
            no_event_rows.append(row)
        else:
            event_rows.append(row)

    def recall(types: Sequence[str] | None = None) -> float:
        selected = (
            event_rows
            if types is None
            else [row for row in event_rows if row["event_type"] in types]
        )
        if not selected:
            return 0.0
        return float(
            sum(row["classification"] == "EVENT_VALID_DETECTION" for row in selected)
            / len(selected)
        )

    event_valid = sum(
        row["classification"] == "EVENT_VALID_DETECTION" for row in event_rows
    )
    premature = sum(
        row["classification"] == "EVENT_PREMATURE_FP" for row in event_rows
    )
    no_event_fp = sum(row["classification"] == "NO_EVENT_FP" for row in no_event_rows)
    hard_fp = sum(row["classification"] == "NO_EVENT_FP" for row in hard_rows)
    event_count = len(event_rows)
    no_event_count = len(no_event_rows)
    hard_count = len(hard_rows)
    return {
        "threshold": float(threshold),
        "event_runs": event_count,
        "no_event_transition_runs": no_event_count,
        "hard_ground_runs": hard_count,
        "overall_event_recall": 0.0 if not event_count else event_valid / event_count,
        "slip_event_recall": recall((EVENT_TYPE_SLIP, EVENT_TYPE_BOTH)),
        "support_event_recall": recall((EVENT_TYPE_SUPPORT, EVENT_TYPE_BOTH)),
        "no_event_transition_specificity": 0.0
        if not no_event_count
        else 1.0 - no_event_fp / no_event_count,
        "no_event_transition_fp_rate": 0.0
        if not no_event_count
        else no_event_fp / no_event_count,
        "hard_ground_specificity": 0.0
        if not hard_count
        else 1.0 - hard_fp / hard_count,
        "hard_ground_fp_rate": 0.0 if not hard_count else hard_fp / hard_count,
        "premature_event_run_fp_rate": 0.0
        if not event_count
        else premature / event_count,
        "latency_ms": _latency_distribution(latencies),
        "event_run_rows": event_rows,
        "no_event_run_rows": no_event_rows,
        "hard_ground_rows": hard_rows,
        "event_outcome_recall": {
            outcome: (
                0.0
                if not (selected := [row for row in event_rows if row["outcome_diagnostic_only"] == outcome])
                else sum(row["classification"] == "EVENT_VALID_DETECTION" for row in selected)
                / len(selected)
            )
            for outcome in (VALID_STABLE, VALID_FALL)
        },
    }


def event_gate_results(
    metrics: Mapping[str, object], gates: Mapping[str, object]
) -> dict[str, bool]:
    latency = metrics["latency_ms"]
    median = latency["median"]
    p95 = latency["p95"]
    return {
        "overall_event_recall": float(metrics["overall_event_recall"])
        >= float(gates["overall_event_recall_min"]),
        "slip_event_recall": float(metrics["slip_event_recall"])
        >= float(gates["slip_event_recall_min"]),
        "support_event_recall": float(metrics["support_event_recall"])
        >= float(gates["support_event_recall_min"]),
        "no_event_transition_specificity": float(
            metrics["no_event_transition_specificity"]
        )
        >= float(gates["no_event_transition_specificity_min"]),
        "hard_ground_specificity": float(metrics["hard_ground_specificity"])
        >= float(gates["hard_ground_specificity_min"]),
        "premature_event_run_fp_rate": float(
            metrics["premature_event_run_fp_rate"]
        )
        <= float(gates["premature_event_run_fp_rate_max"]),
        "median_latency_ms": median is not None
        and float(median) <= float(gates["median_latency_ms_max"]),
        "p95_latency_ms": p95 is not None
        and float(p95) <= float(gates["p95_latency_ms_max"]),
    }


def select_event_threshold(
    evaluations: Sequence[Mapping[str, object]],
    feasibility: Mapping[str, object],
) -> dict[str, object]:
    """Select a validation-only operating point by the frozen lexicographic rule."""
    feasible = [
        row
        for row in evaluations
        if float(row["metrics"]["no_event_transition_fp_rate"])
        <= float(feasibility["no_event_transition_fp_rate_max"])
        and float(row["metrics"]["hard_ground_fp_rate"])
        <= float(feasibility["hard_ground_fp_rate_max"])
        and float(row["metrics"]["premature_event_run_fp_rate"])
        <= float(feasibility["premature_event_run_fp_rate_max"])
    ]

    def rank(row: Mapping[str, object]) -> tuple[float, float, float, float]:
        metrics = row["metrics"]
        p95 = metrics["latency_ms"]["p95"]
        return (
            float(metrics["overall_event_recall"]),
            min(
                float(metrics["slip_event_recall"]),
                float(metrics["support_event_recall"]),
            ),
            -9999.0 if p95 is None else -float(p95),
            float(row["threshold"]),
        )

    if feasible:
        selected = max(feasible, key=rank)
        return {
            "selected": selected,
            "diagnostic_best": selected,
            "feasible_threshold_count": len(feasible),
            "reason": "frozen_false_alarm_feasibility_then_event_recall_latency_threshold",
        }
    diagnostic = max(
        evaluations,
        key=lambda row: (
            sum(
                (
                    float(row["metrics"]["no_event_transition_fp_rate"])
                    <= float(feasibility["no_event_transition_fp_rate_max"]),
                    float(row["metrics"]["hard_ground_fp_rate"])
                    <= float(feasibility["hard_ground_fp_rate_max"]),
                    float(row["metrics"]["premature_event_run_fp_rate"])
                    <= float(feasibility["premature_event_run_fp_rate_max"]),
                )
            ),
            *rank(row),
        ),
    )
    return {
        "selected": None,
        "diagnostic_best": diagnostic,
        "feasible_threshold_count": 0,
        "reason": "no_threshold_met_frozen_false_alarm_constraints",
    }


def select_event_candidate(
    candidates: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Choose the strongest passing candidate, then simpler/shorter on ties."""
    passing = [row for row in candidates if bool(row["validation_passed"])]

    def rank(row: Mapping[str, object]) -> tuple[float, float, float, int, int, float]:
        metrics = row["operating_point"]
        p95 = metrics["latency_ms"]["p95"]
        return (
            float(metrics["overall_event_recall"]),
            min(
                float(metrics["slip_event_recall"]),
                float(metrics["support_event_recall"]),
            ),
            -9999.0 if p95 is None else -float(p95),
            1 if row["model_family"] == "mlp" else 0,
            -int(row["history_ms"]),
            float(metrics["threshold"]),
        )

    if passing:
        selected = max(passing, key=rank)
        return {
            "selected": {
                "model_family": str(selected["model_family"]),
                "history_ms": int(selected["history_ms"]),
                "threshold": float(selected["operating_point"]["threshold"]),
            },
            "reason": "passing_gate_then_recall_type_latency_simplicity_history_threshold",
        }
    diagnostic = max(
        candidates,
        key=lambda row: (
            sum(bool(value) for value in row["validation_gates"].values()),
            *rank(row),
        ),
    )
    return {
        "selected": None,
        "reason": "no_candidate_met_all_validation_gates",
        "diagnostic_best": {
            "model_family": str(diagnostic["model_family"]),
            "history_ms": int(diagnostic["history_ms"]),
            "threshold": float(diagnostic["operating_point"]["threshold"]),
        },
    }


def _checkpoint_paths(
    artifact_path: Path,
    representation: str,
    family: str,
    history_ms: int,
    seeds: Sequence[int],
) -> list[Path]:
    return [
        artifact_path
        / "checkpoints"
        / f"{representation.lower()}_{family}_history{history_ms}_seed{seed}.pt"
        for seed in seeds
    ]


def _window_diagnostics(
    batch: EventBatch, probabilities: np.ndarray
) -> dict[str, float | int]:
    return {
        "auroc": binary_auroc(batch.windows.targets, probabilities),
        "auprc": binary_auprc(batch.windows.targets, probabilities),
        "window_count": len(probabilities),
    }


def _candidate_by_choice(
    candidates: Sequence[Mapping[str, object]], choice: Mapping[str, object]
) -> Mapping[str, object]:
    found = [
        row
        for row in candidates
        if str(row["model_family"]) == str(choice["model_family"])
        and int(row["history_ms"]) == int(choice["history_ms"])
    ]
    if len(found) != 1:
        raise ValueError("selected event candidate is missing or duplicated")
    return found[0]


def _train_candidates(
    document: Mapping[str, object],
    runs: Mapping[str, EventRun],
    normalizers: Mapping[str, Normalizer],
    primary_ids: Mapping[str, Sequence[str]],
    control_ids: Mapping[str, Sequence[str]],
    artifact_path: Path,
    progress: Callable[[str], None],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
    config = document["model"]
    window_config = document["windows"]
    seeds = [int(value) for value in config["seeds"]]
    candidates_by_representation: dict[str, list[dict[str, object]]] = {}
    selections: dict[str, dict[str, object]] = {}
    for representation in EVENT_REPRESENTATIONS:
        candidates = []
        for family in window_config["model_families"]:
            for history_ms in window_config["histories_ms"]:
                train_batch = build_event_windows(
                    runs,
                    [*primary_ids["train"], *control_ids["train"]],
                    representation,
                    int(history_ms),
                    normalizers[representation],
                    stride_ms=int(window_config["endpoint_stride_ms"]),
                    positive_cap=int(window_config["per_run_positive_cap"]),
                    negative_cap=int(window_config["per_run_negative_cap"]),
                )
                validation_batch = build_event_windows(
                    runs,
                    [*primary_ids["validation"], *control_ids["validation"]],
                    representation,
                    int(history_ms),
                    normalizers[representation],
                    stride_ms=int(window_config["endpoint_stride_ms"]),
                    positive_cap=int(window_config["per_run_positive_cap"]),
                    negative_cap=int(window_config["per_run_negative_cap"]),
                )
                checkpoints = _checkpoint_paths(
                    artifact_path,
                    representation,
                    str(family),
                    int(history_ms),
                    seeds,
                )
                training_rows = []
                for seed, checkpoint in zip(seeds, checkpoints):
                    model, training = train_model(
                        str(family),
                        int(history_ms),
                        train_batch.windows,
                        validation_batch.windows,
                        seed,
                        batch_size=int(config["batch_size"]),
                        max_epochs=int(config["max_epochs"]),
                        patience=int(config["early_stopping_patience"]),
                        learning_rate=float(config["learning_rate"]),
                        class_names=EVENT_CLASS_NAMES,
                        selection_metric="validation_loss",
                    )
                    save_checkpoint(
                        checkpoint,
                        model,
                        str(family),
                        int(history_ms),
                        seed,
                        training,
                        input_channels=len(RUNTIME_FEATURE_NAMES[representation]),
                        class_names=EVENT_CLASS_NAMES,
                    )
                    training_rows.append(
                        {
                            "seed": seed,
                            "best_epoch": training.best_epoch,
                            "epochs_completed": training.epochs_completed,
                            "best_validation_cross_entropy": min(
                                row["validation_cross_entropy"]
                                for row in training.history
                            ),
                        }
                    )
                    del model
                models = _load_models(checkpoints)
                probability = np.mean(
                    np.stack(
                        [
                            predict_fall_probability(model, validation_batch.windows)
                            for model in models
                        ]
                    ),
                    axis=0,
                )
                parameter_total = parameter_count(models[0])
                del models
                replay_ids = [
                    *primary_ids["validation"],
                    *control_ids["validation"],
                ]
                replay_runs = {
                    run_id: runs[run_id] for run_id in replay_ids if run_id in runs
                }
                traces = _replay_many(
                    replay_runs,
                    representation,
                    int(history_ms),
                    normalizers[representation],
                    checkpoints,
                )
                evaluations = []
                for threshold in threshold_grid(
                    **document["threshold_calibration"]["grid"]
                ):
                    metrics = evaluate_event_runs(
                        replay_runs,
                        traces,
                        threshold,
                        int(document["threshold_calibration"]["detector_persistence_ms"]),
                    )
                    evaluations.append({"threshold": threshold, "metrics": metrics})
                calibration = select_event_threshold(
                    evaluations,
                    document["threshold_calibration"]["feasibility"],
                )
                operating = calibration["selected"] or calibration["diagnostic_best"]
                operating_metrics = operating["metrics"]
                gates = event_gate_results(
                    operating_metrics, document["validation_gates"]
                )
                candidate = {
                    "model_family": str(family),
                    "history_ms": int(history_ms),
                    "train_windows": len(train_batch.windows),
                    "validation_windows": len(validation_batch.windows),
                    "train_independent_runs": len(set(train_batch.windows.run_ids)),
                    "validation_independent_runs": len(
                        set(validation_batch.windows.run_ids)
                    ),
                    "parameter_count": parameter_total,
                    "epoch_selection": "validation_cross_entropy_minimum",
                    "training": training_rows,
                    "window_diagnostics": _window_diagnostics(
                        validation_batch, probability
                    ),
                    "threshold_calibration": {
                        "selected_threshold": None
                        if calibration["selected"] is None
                        else float(calibration["selected"]["threshold"]),
                        "feasible_threshold_count": calibration[
                            "feasible_threshold_count"
                        ],
                        "reason": calibration["reason"],
                    },
                    "operating_point": operating_metrics,
                    "validation_gates": gates,
                    "validation_passed": calibration["selected"] is not None
                    and all(gates.values()),
                    "checkpoint_paths": [
                        str(path.relative_to(artifact_path)) for path in checkpoints
                    ],
                }
                candidates.append(candidate)
                progress(
                    f"REFLEX {representation} {family} history={history_ms} "
                    f"threshold={operating['threshold']:.2f} "
                    f"recall={operating_metrics['overall_event_recall']:.3f} "
                    f"specificity={operating_metrics['no_event_transition_specificity']:.3f} "
                    f"passed={candidate['validation_passed']}"
                )
                del train_batch, validation_batch, traces
                gc.collect()
        candidates_by_representation[representation] = candidates
        selections[representation] = select_event_candidate(candidates)
    return candidates_by_representation, selections


def _evaluate_holdout(
    document: Mapping[str, object],
    runs: Mapping[str, EventRun],
    candidates: Mapping[str, Sequence[Mapping[str, object]]],
    selections: Mapping[str, Mapping[str, object]],
    normalizers: Mapping[str, Normalizer],
    primary_ids: Sequence[str],
    control_ids: Sequence[str],
    artifact_path: Path,
) -> dict[str, object]:
    results = {}
    for representation in EVENT_REPRESENTATIONS:
        selected = selections[representation]["selected"]
        if selected is None:
            results[representation] = {
                "performed": False,
                "reason": "no_validation_selection",
            }
            continue
        candidate = _candidate_by_choice(candidates[representation], selected)
        paths = [artifact_path / str(value) for value in candidate["checkpoint_paths"]]
        selected_runs = {
            run_id: runs[run_id]
            for run_id in [*primary_ids, *control_ids]
            if run_id in runs
        }
        traces = _replay_many(
            selected_runs,
            representation,
            int(selected["history_ms"]),
            normalizers[representation],
            paths,
        )
        metrics = evaluate_event_runs(
            selected_runs,
            traces,
            float(selected["threshold"]),
            int(document["threshold_calibration"]["detector_persistence_ms"]),
        )
        gates = event_gate_results(metrics, document["holdout"]["gates"])
        results[representation] = {
            "performed": True,
            **dict(selected),
            "metrics": metrics,
            "gates": gates,
            "passed": all(gates.values()),
        }
    return results


def _selection_recommendation(
    selections: Mapping[str, Mapping[str, object]],
    holdout: Mapping[str, Mapping[str, object]],
    near_tie: Mapping[str, object],
) -> dict[str, object]:
    imu_pass = bool(holdout.get(PELVIS_IMU6, {}).get("passed", False))
    fusion_pass = bool(holdout.get(PELVIS_IMU6_FSR8, {}).get("passed", False))
    near = False
    if imu_pass and fusion_pass:
        imu = holdout[PELVIS_IMU6]["metrics"]
        fusion = holdout[PELVIS_IMU6_FSR8]["metrics"]
        near = bool(
            abs(
                float(imu["overall_event_recall"])
                - float(fusion["overall_event_recall"])
            )
            <= float(near_tie["recall_difference_max"])
            and abs(float(imu["latency_ms"]["p95"]) - float(fusion["latency_ms"]["p95"]))
            <= float(near_tie["p95_latency_difference_ms_max"])
        )
    if imu_pass:
        return {
            "representation": PELVIS_IMU6,
            "status": "PELVIS_IMU6_RECOMMENDED_FOR_REFLEX_EVENT",
            "reason": (
                "both_pass_near_tie_minimal_sensor_priority"
                if near
                else "imu6_validation_and_holdout_pass"
            ),
            "near_tie": near,
        }
    if fusion_pass:
        return {
            "representation": PELVIS_IMU6_FSR8,
            "status": "PELVIS_IMU6_PLUS_FSR8_RECOMMENDED",
            "reason": "imu6_not_supported_but_fusion14_validation_and_holdout_pass",
            "near_tie": False,
        }
    return {
        "representation": None,
        "status": "NO_RUNTIME_SENSOR_RECOMMENDATION",
        "reason": "neither_frozen_runtime_representation_passed",
        "near_tie": False,
    }


def _severity_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def distribution(
        field: str,
        selected: Sequence[Mapping[str, object]],
        scale: float = 1.0,
    ) -> dict[str, float | None]:
        values = [
            scale * float(row[field])
            for row in selected
            if row.get(field) is not None and np.isfinite(float(row[field]))
        ]
        if not values:
            return {key: None for key in ("min", "median", "p95", "max")}
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(array)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }

    slip = [
        row for row in rows if row["event_type"] in (EVENT_TYPE_SLIP, EVENT_TYPE_BOTH)
    ]
    support = [
        row
        for row in rows
        if row["event_type"] in (EVENT_TYPE_SUPPORT, EVENT_TYPE_BOTH)
    ]
    return {
        "slip": {
            "peak_drift_mm": distribution("peak_drift_m", slip, 1000.0),
            "peak_tangential_velocity_mps": distribution(
                "peak_tangential_velocity_mps", slip
            ),
            "bilateral_runs": sum(row["slip_side"] == "BILATERAL" for row in slip),
            "left_runs": sum(row["slip_side"] in ("LEFT", "BILATERAL") for row in slip),
            "right_runs": sum(row["slip_side"] in ("RIGHT", "BILATERAL") for row in slip),
            "reported_distance_unit": "mm",
        },
        "support": {
            "peak_spread_mm": distribution(
                "peak_support_spread_m", support, 1000.0
            ),
            "maximum_deformation_mm": distribution(
                "maximum_support_deformation_m", support, 1000.0
            ),
            "patterns": {
                pattern: sum(row["support_pattern"] == pattern for row in support)
                for pattern in sorted({str(row["support_pattern"]) for row in support})
            },
            "reported_distance_unit": "mm",
        },
        "severity_classifier_trained": False,
    }


def _representative_rows(
    holdout: Mapping[str, Mapping[str, object]], recommendation: Mapping[str, object]
) -> dict[str, object]:
    representation = recommendation["representation"]
    if representation is None or not holdout.get(str(representation), {}).get("performed"):
        return {}
    metrics = holdout[str(representation)]["metrics"]
    event_rows = list(metrics["event_run_rows"])
    no_event_rows = list(metrics["no_event_run_rows"])

    def first(rows: Sequence[Mapping[str, object]], predicate: Callable[[Mapping[str, object]], bool]) -> Mapping[str, object] | None:
        return next((row for row in rows if predicate(row)), None)

    return {
        "ice_recovered_severe_slip": first(
            event_rows,
            lambda row: row["target_terrain"] == "ice"
            and row["outcome_diagnostic_only"] == VALID_STABLE,
        ),
        "ice_fall_severe_slip": first(
            event_rows,
            lambda row: row["target_terrain"] == "ice"
            and row["outcome_diagnostic_only"] == VALID_FALL,
        ),
        "ice_bilateral_slip": first(
            event_rows,
            lambda row: row["target_terrain"] == "ice"
            and row["slip_side"] == "BILATERAL",
        ),
        "sand_benign_deformation": first(
            no_event_rows, lambda row: row["target_terrain"] == "sand"
        ),
        "sand_support_event_recovered": first(
            event_rows,
            lambda row: row["target_terrain"] == "sand"
            and row["outcome_diagnostic_only"] == VALID_STABLE,
        ),
        "sand_support_event_fall": first(
            event_rows,
            lambda row: row["target_terrain"] == "sand"
            and row["outcome_diagnostic_only"] == VALID_FALL,
        ),
        "terrain_output_assumption": "clean_target_touchdown_plus_50_ms",
    }


def _validation_representative_replay(
    candidates: Mapping[str, Sequence[Mapping[str, object]]],
    runs: Mapping[str, EventRun],
    normalizers: Mapping[str, Normalizer],
    artifact_path: Path,
) -> dict[str, object]:
    """Build representative validation timelines even when no candidate passes."""
    available = [row for rows in candidates.values() for row in rows]
    if not available:
        return {}

    def rank(row: Mapping[str, object]) -> tuple[int, float, float, float]:
        metrics = row["operating_point"]
        return (
            sum(bool(value) for value in row["validation_gates"].values()),
            float(metrics["overall_event_recall"]),
            float(metrics["no_event_transition_specificity"]),
            float(metrics["support_event_recall"]),
        )

    candidate = max(available, key=rank)
    representation = next(
        name for name, rows in candidates.items() if candidate in rows
    )
    paths = [artifact_path / str(value) for value in candidate["checkpoint_paths"]]
    validation_runs = {
        run_id: run for run_id, run in runs.items() if run.split == "validation"
    }
    traces = _replay_many(
        validation_runs,
        representation,
        int(candidate["history_ms"]),
        normalizers[representation],
        paths,
    )
    metrics = candidate["operating_point"]
    all_rows = [
        *metrics["event_run_rows"],
        *metrics["no_event_run_rows"],
    ]
    by_id = {str(row["run_id"]): row for row in all_rows}

    def probability_at(trace: ReplayTrace, sample: int | None) -> float | None:
        if sample is None or not len(trace.endpoints):
            return None
        position = int(np.searchsorted(trace.endpoints, sample))
        if position >= len(trace.endpoints) or int(trace.endpoints[position]) != sample:
            return None
        return float(trace.probabilities[position])

    detailed = []
    for run_id, row in by_id.items():
        run = validation_runs[run_id]
        trace = traces[run_id]
        detection = row["detection_sample"]
        detailed.append(
            {
                **dict(row),
                "first_target_contact_sample": run.first_contact_sample,
                "clean_target_touchdown_sample": run.first_touchdown_sample,
                "terrain_output_sample": run.first_touchdown_sample + 50,
                "probability_at_event": probability_at(trace, run.event_sample),
                "probability_at_detection": probability_at(trace, detection),
                "peak_probability": float(np.max(trace.probabilities)),
                "diagnostic_threshold": float(metrics["threshold"]),
            }
        )

    def first(predicate: Callable[[Mapping[str, object]], bool]) -> Mapping[str, object] | None:
        return next((row for row in detailed if predicate(row)), None)

    return {
        "status": "diagnostic_best_validation_replay_not_a_selected_detector",
        "representation": representation,
        "model_family": candidate["model_family"],
        "history_ms": candidate["history_ms"],
        "threshold": metrics["threshold"],
        "ice_recovered_severe_slip": first(
            lambda row: row["target_terrain"] == "ice"
            and row["outcome_diagnostic_only"] == VALID_STABLE
        ),
        "ice_fall_severe_slip": first(
            lambda row: row["target_terrain"] == "ice"
            and row["outcome_diagnostic_only"] == VALID_FALL
        ),
        "ice_bilateral_slip": first(
            lambda row: row["target_terrain"] == "ice"
            and row["slip_side"] == "BILATERAL"
        ),
        "sand_benign_deformation": first(
            lambda row: row["target_terrain"] == "sand"
            and row["event_type"] == EVENT_TYPE_NONE
        ),
        "sand_support_event_recovered": first(
            lambda row: row["target_terrain"] == "sand"
            and row["event_type"] == EVENT_TYPE_SUPPORT
            and row["outcome_diagnostic_only"] == VALID_STABLE
        ),
        "sand_support_event_fall": first(
            lambda row: row["target_terrain"] == "sand"
            and row["event_type"] == EVENT_TYPE_SUPPORT
            and row["outcome_diagnostic_only"] == VALID_FALL
        ),
    }


def _verdict(
    candidates: Mapping[str, Sequence[Mapping[str, object]]],
    selections: Mapping[str, Mapping[str, object]],
    holdout: Mapping[str, Mapping[str, object]],
) -> tuple[str, str]:
    if bool(holdout.get(PELVIS_IMU6, {}).get("passed", False)):
        return (
            "EVENT_CENTRIC_REFLEX_DETECTION_SUPPORTED_IMU6",
            "EVENT_CENTRIC_REFLEX_ARCHITECTURE_SUPPORTED",
        )
    if bool(holdout.get(PELVIS_IMU6_FSR8, {}).get("passed", False)):
        return (
            "EVENT_CENTRIC_REFLEX_DETECTION_SUPPORTED_IMU6_FSR8",
            "EVENT_CENTRIC_REFLEX_ARCHITECTURE_SUPPORTED",
        )
    gate_counts = [
        sum(bool(value) for value in row["validation_gates"].values())
        for rows in candidates.values()
        for row in rows
    ]
    any_selection = any(
        selections[name]["selected"] is not None for name in EVENT_REPRESENTATIONS
    )
    holdout_minor = any(
        row.get("performed")
        and sum(not bool(value) for value in row.get("gates", {}).values()) <= 2
        for row in holdout.values()
    )
    if (gate_counts and max(gate_counts) >= 6) or (any_selection and holdout_minor):
        return "EVENT_CENTRIC_REFLEX_DETECTION_PROMISING", "NOT_SUPPORTED"
    return "EVENT_CENTRIC_REFLEX_DETECTION_NOT_SUPPORTED", "NOT_SUPPORTED"


def run_event_centric_reflex_trigger(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Generate, train, select, and one-shot validate the frozen event study."""
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    document = _load_yaml(config_path)
    dense_path = repository_root / str(document["source"]["dense_design_config"])
    dense_document = _load_yaml(dense_path)
    primary_specs, control_specs = generate_event_specifications(
        document, dense_document
    )
    design = validate_event_design(
        document, dense_document, primary_specs, control_specs
    )
    historical_audit = audit_historical_dense_dataset(
        repository_root / str(document["source"]["historical_dense_dataset"])
    )
    if historical_audit["sufficient"]:
        raise ValueError("historical dense audit unexpectedly permits in-place reuse")
    if PHYSICS_TIMESTEP_S != float(document["common"]["physics_timestep_s"]):
        raise ValueError("event physics timestep differs from canonical value")
    if SENSOR_RATE_HZ != int(document["common"]["sensor_rate_hz"]):
        raise ValueError("event sample rate differs from canonical value")
    source = document["source"]
    for key, sha_key in (
        ("simulator_config", "simulator_config_sha256"),
        ("scenario_calibration_config", "scenario_calibration_sha256"),
        ("dense_design_config", "dense_design_sha256"),
        ("policy_path", "policy_sha256"),
    ):
        path = repository_root / str(source[key])
        if not path.is_file() or _file_sha256(path) != str(source[sha_key]):
            raise ValueError(f"event provenance source changed: {key}")

    dataset_path = (repository_root / str(document["dataset"]["path"])).resolve()
    artifact_path = (repository_root / str(document["artifacts"]["path"])).resolve()
    dataset_path.relative_to(repository_root)
    artifact_path.relative_to(repository_root)
    if dataset_path.exists() and any(dataset_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite event dataset: {dataset_path}")
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite event artifacts: {artifact_path}")
    dataset_path.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)
    protected_paths = [
        str(value) for value in document["terrain_regression"]["protected_paths"]
    ]
    terrain_before = _protected_hashes(repository_root, protected_paths)
    base = load_simulation_config(repository_root / str(source["simulator_config"]))
    generated, invalid = simulate_event_cohort(
        base,
        [*primary_specs, *control_specs],
        repository_root / str(source["policy_path"]),
        document,
        progress,
    )
    specs_by_id = {str(row["id"]): row for row in [*primary_specs, *control_specs]}
    manifest_rows = []
    for run_id in sorted(generated):
        run = generated[run_id]
        path = dataset_path / f"{run_id}.npz"
        _event_run_to_npz(path, run)
        manifest_rows.append(
            _event_manifest_row(path, run, specs_by_id[run_id])
        )
    cohort = _cohort_summary(generated, invalid)
    manifest, manifest_sha = _dataset_manifest(
        document, manifest_rows, dataset_path
    )
    dataset_summary = {
        "dataset_id": manifest["dataset_id"],
        "path": str(dataset_path),
        "run_files": len(manifest_rows),
        "size_bytes": sum(int(row["size_bytes"]) for row in manifest_rows),
        "manifest_sha256": manifest_sha,
    }
    readiness = readiness_results(cohort, design, document["readiness_gates"])
    severity = _severity_summary(manifest_rows)
    del generated
    gc.collect()
    terrain_after_dataset = _protected_hashes(repository_root, protected_paths)
    if not all(readiness.values()):
        metrics = {
            "experiment": document["experiment"],
            "historical_dense_audit": historical_audit,
            "design": design,
            "dataset": dataset_summary,
            "cohort": cohort,
            "readiness": {"passed": False, "gates": readiness},
            "severity": severity,
            "training": {"performed": False},
            "holdout": {"performed": False, "guard_open_count": 0},
            "terrain_regression": {
                "passed": terrain_before == terrain_after_dataset,
                "before": terrain_before,
                "after": terrain_after_dataset,
            },
            "fusion_regression": fusion_regression(),
            "verdict": "EVENT_CENTRIC_REFLEX_DETECTION_NOT_SUPPORTED",
            "architecture_recommendation": "NOT_SUPPORTED",
        }
        _write_json(artifact_path / "metrics.json", metrics)
        return artifact_path, metrics

    primary_ids = {
        split: [str(row["id"]) for row in primary_specs if row["split"] == split]
        for split in ("train", "validation", "holdout")
    }
    control_ids = {
        split: [str(row["id"]) for row in control_specs if row["split"] == split]
        for split in ("train", "validation", "holdout")
    }
    development_runs = load_event_runs(
        dataset_path, manifest, ("train", "validation")
    )
    normalizer_train_ids = [*primary_ids["train"], *control_ids["train"]]
    normalizers = {
        representation: fit_event_normalizer(
            development_runs,
            [run_id for run_id in normalizer_train_ids if run_id in development_runs],
            representation,
            int(document["normalization"]["per_run_sample_cap"]),
            float(document["normalization"]["standard_deviation_floor"]),
        )
        for representation in EVENT_REPRESENTATIONS
    }
    _write_json(
        artifact_path / "train_only_normalization.json",
        {name: normalizer.to_dict() for name, normalizer in normalizers.items()},
    )
    candidates, selections = _train_candidates(
        document,
        development_runs,
        normalizers,
        primary_ids,
        control_ids,
        artifact_path,
        progress,
    )
    validation_representatives = _validation_representative_replay(
        candidates,
        development_runs,
        normalizers,
        artifact_path,
    )
    _write_json(
        artifact_path / "selection_before_holdout.json",
        {"selections": selections, "holdout_opened": False},
    )
    guard = EventHoldoutGuard()
    if any(selections[name]["selected"] is not None for name in EVENT_REPRESENTATIONS):
        guard.open_once()
        holdout_runs = load_event_runs(
            dataset_path, manifest, ("holdout",), holdout_guard=guard
        )
        holdout = _evaluate_holdout(
            document,
            holdout_runs,
            candidates,
            selections,
            normalizers,
            primary_ids["holdout"],
            control_ids["holdout"],
            artifact_path,
        )
    else:
        holdout = {
            name: {"performed": False, "reason": "no_validation_selection"}
            for name in EVENT_REPRESENTATIONS
        }
    recommendation = _selection_recommendation(
        selections, holdout, document["selection"]["near_tie"]
    )
    representatives = _representative_rows(holdout, recommendation)
    if not representatives:
        representatives = validation_representatives
    terrain_after = _protected_hashes(repository_root, protected_paths)
    verdict, architecture = _verdict(candidates, selections, holdout)
    metrics = {
        "experiment": document["experiment"],
        "historical_dense_audit": historical_audit,
        "design": design,
        "dataset": dataset_summary,
        "cohort": cohort,
        "readiness": {"passed": True, "gates": readiness},
        "severity": severity,
        "normalization": {
            name: normalizer.to_dict() for name, normalizer in normalizers.items()
        },
        "validation": {
            "performed": True,
            "candidates": candidates,
            "selections": selections,
            "holdout_sealed_during_selection": True,
        },
        "holdout": {
            "performed": any(row["performed"] for row in holdout.values()),
            "guard_open_count": guard.open_count,
            "reselection_performed": False,
            "representations": holdout,
        },
        "selection": recommendation,
        "representative_replay": representatives,
        "terrain_interaction": {
            "terrain_candidate_latency_ms": 50,
            "event_detector_waits_for_terrain": False,
            "unknown_event_action": "GENERIC_DISTURBANCE_AND_REFLEX_REQUIRED",
            "later_terrain_refinement": True,
        },
        "terrain_regression": {
            "passed": terrain_before == terrain_after,
            "before": terrain_before,
            "after": terrain_after,
            "retrained": False,
        },
        "fusion_regression": fusion_regression(),
        "causality": {
            "passed": True,
            "future_samples_in_tensor": False,
            "fall_or_time_to_fall_in_tensor": False,
            "terrain_identity_in_tensor": False,
            "replay_stride_ms": 1,
            "detector_persistence_ms": 5,
            "valid_latency_ms": [-20, 50],
        },
        "label_contract": {
            "class_names": list(EVENT_CLASS_NAMES),
            "event_union": "ANY_SLIP_EVENT_OR_SUPPORT_REFLEX_EVENT",
            "fall_or_recovery_dependency": False,
        },
        "runtime_boundary": {
            "candidate_representations": list(EVENT_REPRESENTATIONS),
            "foot_imu_added": False,
            "q_dq_added": False,
            "severity_classifier_trained": False,
            "production_enum_changed": False,
            "final_sensor_architecture_frozen": False,
        },
        "verdict": verdict,
        "architecture_recommendation": architecture,
    }
    _write_json(artifact_path / "metrics.json", metrics)
    progress(
        json.dumps(
            {
                "selection": recommendation,
                "holdout": holdout,
                "verdict": verdict,
                "architecture": architecture,
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return artifact_path, metrics
