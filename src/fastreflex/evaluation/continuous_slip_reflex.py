"""Terrain-independent continuous Slip reflex detector development.

The physical Slip oracle remains privileged and is used only to construct
labels and score replay.  Every detector input is a causal runtime-sensor
trajectory, and the detector is evaluated continuously without consulting the
Terrain recognizer.  Frozen Terrain output is used later for advisory cause
refinement and for the already-selected Sand Support branch.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from fastreflex.dataset.loader import Normalizer, WindowSet
from fastreflex.evaluation.reflex_event import (
    EVENT_CLASS_NAMES,
    EVENT_TYPE_BOTH,
    EVENT_TYPE_NONE,
    EVENT_TYPE_SLIP,
    EVENT_TYPE_SUPPORT,
    EventHoldoutGuard,
    EventRun,
    _load_yaml,
    _write_json,
    load_event_runs,
)
from fastreflex.evaluation.stability_temporal import _file_sha256, _protected_hashes
from fastreflex.evaluation.terrain_conditioned_reflex import (
    ICE,
    SAND,
    BranchReplay,
    TerrainGateTrace,
    _relevant_run_ids,
    _replay_many as replay_frozen_support_many,
    evaluate_branch_replays as evaluate_frozen_support_replays,
    fit_branch_normalizer as fit_frozen_support_normalizer,
    fsr_feature_base,
    holdout_gate_from_observer_dataset,
    imu_feature_base,
    sustained_alert_trace,
    terrain_timing_audit,
)
from fastreflex.evaluation.transition_scenarios import fusion_regression
from fastreflex.models.baselines import parameter_count
from fastreflex.training.trainer import load_checkpoint, save_checkpoint, train_model


PHASE_A_CANDIDATES = {
    "A1": ("pelvis_imu6",),
    "A2": ("fsr8",),
    "A3": ("pelvis_imu6", "fsr8"),
}
PHASE_B_CANDIDATES = {
    "B1": ("foot_imu12",),
    "B2": ("foot_imu12", "fsr8"),
    "B3": ("foot_imu12", "pelvis_imu6"),
    "B4": ("foot_imu12", "fsr8", "pelvis_imu6"),
}
PHYSICAL_CHANNELS = {"pelvis_imu6": 6, "fsr8": 8, "foot_imu12": 12}
TEMPORAL_NAMES = (
    "base",
    "delta_1ms",
    "delta_5ms",
    "delta_10ms",
    "causal_mean_5ms",
    "causal_mean_10ms",
    "causal_variance_5ms",
    "causal_variance_10ms",
)


@dataclass(frozen=True)
class SlipReplay:
    """Continuous 1 kHz probability trace over one valid walking run."""

    endpoints: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class SlipWindowBatch:
    windows: WindowSet
    rows: tuple[dict[str, object], ...]


@dataclass
class SlipCandidateState:
    record: dict[str, object]
    normalizer: Normalizer
    checkpoint_paths: tuple[Path, ...]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _distribution(values: Sequence[int | float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {name: None for name in ("min", "p10", "median", "p95", "max")}
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _causal_delta(values: np.ndarray, lag: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    result = np.zeros_like(array)
    if lag < len(array):
        result[lag:] = array[lag:] - array[:-lag]
    return result


def _causal_rolling(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean and variance; the endpoint is the latest sample used."""
    array = np.asarray(values, dtype=np.float64)
    prefix = np.vstack((np.zeros((1, array.shape[1])), np.cumsum(array, axis=0)))
    square = np.vstack(
        (np.zeros((1, array.shape[1])), np.cumsum(array * array, axis=0))
    )
    ends = np.arange(1, len(array) + 1)
    starts = np.maximum(0, ends - int(width))
    counts = (ends - starts)[:, None]
    mean = (prefix[ends] - prefix[starts]) / counts
    variance = (square[ends] - square[starts]) / counts - mean * mean
    return mean.astype(np.float32), np.maximum(variance, 0.0).astype(np.float32)


def temporal_expansion(
    base: np.ndarray, base_names: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply the predeclared eight causal transforms in deterministic order."""
    values = np.asarray(base, dtype=np.float32)
    mean5, variance5 = _causal_rolling(values, 5)
    mean10, variance10 = _causal_rolling(values, 10)
    blocks = (
        values,
        _causal_delta(values, 1),
        _causal_delta(values, 5),
        _causal_delta(values, 10),
        mean5,
        mean10,
        variance5,
        variance10,
    )
    names = tuple(
        f"{transform}_{name}" for transform in TEMPORAL_NAMES for name in base_names
    )
    result = np.concatenate(blocks, axis=1)
    if result.shape[1] != len(names) or not np.all(np.isfinite(result)):
        raise ValueError("continuous Slip derived features are malformed")
    return result, names


def foot_imu_feature_base(
    foot_imu12: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Per-foot IMU dynamics plus two bilateral norm differences."""
    values = np.asarray(foot_imu12, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError("Foot IMU input must have shape [samples,12]")
    left, left_names = imu_feature_base(values[:, :6])
    right, right_names = imu_feature_base(values[:, 6:])
    bilateral = np.column_stack((left[:, 6] - right[:, 6], left[:, 7] - right[:, 7]))
    names = (
        *(f"left_{name}" for name in left_names),
        *(f"right_{name}" for name in right_names),
        "bilateral_accel_norm_difference",
        "bilateral_gyro_norm_difference",
    )
    return np.concatenate((left, right, bilateral), axis=1), tuple(names)


def _foot_imu_for_run(dataset_path: Path | None, run_id: str) -> np.ndarray | None:
    if dataset_path is None:
        return None
    path = dataset_path / f"{run_id}.npz"
    with np.load(path, allow_pickle=False) as stored:
        values = np.asarray(stored["foot_imu12"], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 12 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid Foot IMU tensor: {run_id}")
    return values


def extract_continuous_slip_features(
    run: EventRun,
    components: Sequence[str],
    *,
    foot_imu12: np.ndarray | None = None,
    epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build one terrain-free, event-clock-free causal runtime tensor."""
    chunks: list[np.ndarray] = []
    names: list[str] = []
    imu6 = np.asarray(run.features["PELVIS_IMU6"], dtype=np.float32)
    fusion = np.asarray(run.features["PELVIS_IMU6_FSR8"], dtype=np.float32)
    fsr8 = fusion[:, 6:]
    for component in components:
        if component == "pelvis_imu6":
            base, base_names = imu_feature_base(imu6)
            expanded, schema = temporal_expansion(base, base_names)
            chunks.append(expanded)
            names.extend(f"pelvis_{name}" for name in schema)
        elif component == "fsr8":
            base, base_names = fsr_feature_base(fsr8, epsilon)
            expanded, schema = temporal_expansion(base, base_names)
            chunks.append(expanded)
            names.extend(f"fsr_{name}" for name in schema)
        elif component == "foot_imu12":
            if foot_imu12 is None or len(foot_imu12) != len(imu6):
                raise ValueError(
                    "Foot IMU component requested without aligned observer data"
                )
            base, base_names = foot_imu_feature_base(foot_imu12)
            expanded, schema = temporal_expansion(base, base_names)
            chunks.append(expanded)
            names.extend(f"foot_{name}" for name in schema)
        else:
            raise ValueError(f"unsupported Slip component: {component}")
    if not chunks:
        raise ValueError("continuous Slip representation is empty")
    result = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    forbidden = ("terrain", "fall", "recovery", "slip_clock", "support_clock")
    if any(any(token in name for token in forbidden) for name in names):
        raise ValueError("privileged metadata leaked into Slip feature schema")
    if not np.all(np.isfinite(result)):
        raise ValueError("continuous Slip tensor is nonfinite")
    return result, tuple(names)


def feature_schema_for_components(components: Sequence[str]) -> tuple[str, ...]:
    samples = 12
    run = _synthetic_schema_run(samples)
    foot = np.zeros((samples, 12), dtype=np.float32)
    _, names = extract_continuous_slip_features(run, components, foot_imu12=foot)
    return names


def _synthetic_schema_run(samples: int) -> EventRun:
    imu = np.zeros((samples, 6), dtype=np.float32)
    fsr = np.zeros((samples, 8), dtype=np.float32)
    zeros2 = np.zeros((samples, 2), dtype=np.float32)
    return EventRun(
        run_id="schema",
        split="train",
        source_terrain="concrete",
        target_terrain="ice",
        design_role="stable",
        first_contact_sample=0,
        first_touchdown_sample=0,
        censor_sample=samples,
        outcome_diagnostic="VALID_STABLE",
        fall_sample_diagnostic=None,
        features={
            "PELVIS_IMU6": imu,
            "PELVIS_IMU6_FSR8": np.concatenate((imu, fsr), axis=1),
        },
        timestamp_us=np.arange(samples, dtype=np.int64) * 1000,
        slip_event_samples_per_foot=(None, None),
        support_event_samples_per_foot=(None, None),
        event_sample=None,
        event_type=EVENT_TYPE_NONE,
        hard_stable_control=False,
        drift_m=zeros2,
        tangential_velocity_mps=zeros2,
        support_spread_m=zeros2,
        support_max_displacement_m=zeros2,
        loaded_contact=np.zeros((samples, 2), dtype=bool),
        sink_pattern="uniform",
        support_pattern="balanced_soft",
    )


def slip_event_sample(run: EventRun) -> int | None:
    values = [
        int(value) for value in run.slip_event_samples_per_foot if value is not None
    ]
    return None if not values else min(values)


def support_event_sample(run: EventRun) -> int | None:
    values = [
        int(value) for value in run.support_event_samples_per_foot if value is not None
    ]
    return None if not values else min(values)


def continuous_positive_endpoints(
    run: EventRun,
    history_ms: int,
    *,
    interval_ms: tuple[int, int] = (-30, 40),
    stride_ms: int = 5,
) -> np.ndarray:
    event = slip_event_sample(run)
    if event is None:
        return np.empty(0, dtype=np.int64)
    endpoints = np.arange(
        event + interval_ms[0], event + interval_ms[1] + 1, stride_ms, dtype=np.int64
    )
    valid = (endpoints >= history_ms - 1) & (endpoints < run.censor_sample)
    if run.fall_sample_diagnostic is not None:
        valid &= endpoints < int(run.fall_sample_diagnostic)
    return endpoints[valid]


def continuous_negative_candidates(run: EventRun, history_ms: int) -> np.ndarray:
    """All established no-hazard endpoints, never post-Slip or post-Support."""
    first = history_ms - 1
    last = run.censor_sample - 1
    slip = slip_event_sample(run)
    support = support_event_sample(run)
    if slip is not None:
        last = min(last, slip - 40)
    if support is not None:
        last = min(last, support - 30)
    if run.fall_sample_diagnostic is not None:
        last = min(last, int(run.fall_sample_diagnostic) - 1)
    if last < first:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64)


def _evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    selected = np.asarray(values, dtype=np.int64)
    if count <= 0 or not len(selected):
        return np.empty(0, dtype=np.int64)
    if len(selected) <= count:
        return selected
    return selected[np.linspace(0, len(selected) - 1, count, dtype=np.int64)]


def gait_sampling_categories(run: EventRun) -> dict[str, np.ndarray]:
    """Privileged contact phase is used only to diversify negative sampling."""
    loaded = np.asarray(run.loaded_contact, dtype=bool)
    previous = np.vstack((np.zeros((1, 2), dtype=bool), loaded[:-1]))
    touchdown = np.any(loaded & ~previous, axis=1)
    release = np.any(~loaded & previous, axis=1)
    categories = {
        "touchdown_loading": np.flatnonzero(touchdown),
        "contact_release": np.flatnonzero(release),
        "left_support": np.flatnonzero(loaded[:, 0] & ~loaded[:, 1]),
        "right_support": np.flatnonzero(~loaded[:, 0] & loaded[:, 1]),
        "double_support": np.flatnonzero(np.all(loaded, axis=1)),
        "no_support": np.flatnonzero(~np.any(loaded, axis=1)),
    }
    return categories


def initial_negative_endpoints(
    run: EventRun,
    history_ms: int,
    *,
    per_category: int = 12,
    contact_shock_count: int = 12,
) -> np.ndarray:
    eligible = continuous_negative_candidates(run, history_ms)
    if not len(eligible):
        return eligible
    allowed = set(int(value) for value in eligible)
    selected: list[np.ndarray] = []
    for values in gait_sampling_categories(run).values():
        candidates = np.asarray([value for value in values if int(value) in allowed])
        selected.append(_evenly_spaced(candidates, per_category))
    accel = np.asarray(run.features["PELVIS_IMU6"], dtype=np.float32)[:, :3]
    shock = np.linalg.norm(_causal_delta(accel, 1), axis=1)
    order = sorted(eligible, key=lambda value: (-float(shock[value]), int(value)))
    selected.append(np.asarray(sorted(order[:contact_shock_count]), dtype=np.int64))
    return np.unique(np.concatenate(selected)) if selected else np.empty(0, np.int64)


def mine_hard_negative_endpoints(
    candidates: np.ndarray,
    probabilities: np.ndarray,
    *,
    top_k: int = 12,
    minimum_separation_ms: int = 30,
    excluded: Sequence[int] = (),
) -> np.ndarray:
    endpoints = np.asarray(candidates, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if endpoints.shape != scores.shape:
        raise ValueError("HNM endpoints and probabilities differ")
    prior = {int(value) for value in excluded}
    order = sorted(range(len(endpoints)), key=lambda i: (-scores[i], int(endpoints[i])))
    selected: list[int] = []
    for index in order:
        endpoint = int(endpoints[index])
        if endpoint in prior:
            continue
        if all(abs(endpoint - other) >= minimum_separation_ms for other in selected):
            selected.append(endpoint)
            if len(selected) >= top_k:
                break
    return np.asarray(sorted(selected), dtype=np.int64)


def fit_continuous_normalizer(
    runs: Mapping[str, EventRun],
    run_ids: Sequence[str],
    components: Sequence[str],
    *,
    foot_dataset_path: Path | None,
    per_run_sample_cap: int,
    standard_deviation_floor: float,
) -> Normalizer:
    chunks: list[np.ndarray] = []
    fit_ids: list[str] = []
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        features, _ = extract_continuous_slip_features(
            run,
            components,
            foot_imu12=_foot_imu_for_run(foot_dataset_path, run_id),
        )
        eligible = np.arange(0, run.censor_sample, dtype=np.int64)
        if run.fall_sample_diagnostic is not None:
            eligible = eligible[eligible < int(run.fall_sample_diagnostic)]
        eligible = _evenly_spaced(eligible, per_run_sample_cap)
        if len(eligible):
            chunks.append(features[eligible].astype(np.float64))
            fit_ids.append(run_id)
        del features
    if not chunks:
        raise ValueError("continuous Slip normalizer has no TRAIN samples")
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


def _normalize(normalizer: Normalizer, values: np.ndarray) -> np.ndarray:
    result = normalizer.transform(values).astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError("normalized continuous Slip tensor is nonfinite")
    return result


def _window_set(
    inputs: Sequence[np.ndarray],
    targets: Sequence[int],
    run_ids: Sequence[str],
    endpoints: Sequence[int],
) -> WindowSet:
    if not inputs:
        raise ValueError("continuous Slip window set is empty")
    labels = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("continuous Slip windows must contain both classes")
    return WindowSet(
        inputs=np.stack(inputs).astype(np.float32),
        targets=labels,
        run_ids=np.asarray(run_ids, dtype=str),
        endpoint_samples=np.asarray(endpoints, dtype=np.int64),
        available_by_class=tuple(int(value) for value in counts[:2]),
    )


def build_continuous_windows(
    runs: Mapping[str, EventRun],
    run_ids: Sequence[str],
    components: Sequence[str],
    history_ms: int,
    normalizer: Normalizer,
    *,
    foot_dataset_path: Path | None,
    per_category: int,
    contact_shock_count: int,
    extra_negative_endpoints: Mapping[str, Sequence[int]] | None = None,
) -> SlipWindowBatch:
    inputs: list[np.ndarray] = []
    targets: list[int] = []
    source_ids: list[str] = []
    endpoints: list[int] = []
    rows: list[dict[str, object]] = []
    extras = extra_negative_endpoints or {}
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        features, _ = extract_continuous_slip_features(
            run,
            components,
            foot_imu12=_foot_imu_for_run(foot_dataset_path, run_id),
        )
        positive = continuous_positive_endpoints(run, history_ms)
        negative = initial_negative_endpoints(
            run,
            history_ms,
            per_category=per_category,
            contact_shock_count=contact_shock_count,
        )
        if run_id in extras:
            allowed = set(
                int(value) for value in continuous_negative_candidates(run, history_ms)
            )
            extra = np.asarray(
                [int(value) for value in extras[run_id] if int(value) in allowed],
                dtype=np.int64,
            )
            negative = np.unique(np.concatenate((negative, extra)))
        if set(int(value) for value in positive) & set(
            int(value) for value in negative
        ):
            raise RuntimeError("Slip-positive window was silently used as a negative")
        for label, selected, kind in (
            (1, positive, "slip_event_positive"),
            (0, negative, "continuous_no_hazard_negative"),
        ):
            for endpoint in selected:
                first = int(endpoint) - history_ms + 1
                if first < 0 or int(endpoint) >= run.censor_sample:
                    raise ValueError("continuous Slip window crossed a causal boundary")
                if run.fall_sample_diagnostic is not None and int(endpoint) >= int(
                    run.fall_sample_diagnostic
                ):
                    raise ValueError("post-fall sample entered continuous Slip input")
                inputs.append(
                    _normalize(normalizer, features[first : int(endpoint) + 1])
                )
                targets.append(label)
                source_ids.append(run_id)
                endpoints.append(int(endpoint))
                rows.append(
                    {
                        "run_id": run_id,
                        "endpoint_sample": int(endpoint),
                        "label": label,
                        "kind": kind,
                        "split": run.split,
                    }
                )
        del features
    return SlipWindowBatch(
        windows=_window_set(inputs, targets, source_ids, endpoints), rows=tuple(rows)
    )


def _predict_windows(
    models: Sequence[torch.nn.Module], windows: np.ndarray
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(windows, dtype=np.float32))
    with torch.no_grad():
        values = [
            torch.softmax(model(tensor), dim=1)[:, 1].cpu().numpy() for model in models
        ]
    return np.mean(np.stack(values), axis=0).astype(np.float64)


def replay_continuous_slip_run(
    run: EventRun,
    components: Sequence[str],
    history_ms: int,
    normalizer: Normalizer,
    models: Sequence[torch.nn.Module],
    *,
    foot_imu12: np.ndarray | None,
    batch_size: int = 512,
) -> SlipReplay:
    """Replay at 1 ms with no Terrain gate and past/current samples only."""
    features, _ = extract_continuous_slip_features(
        run, components, foot_imu12=foot_imu12
    )
    stop = run.censor_sample
    if run.fall_sample_diagnostic is not None:
        stop = min(stop, int(run.fall_sample_diagnostic))
    endpoints = np.arange(history_ms - 1, stop, dtype=np.int64)
    offsets = np.arange(history_ms - 1, -1, -1, dtype=np.int64)
    probability: list[np.ndarray] = []
    for first in range(0, len(endpoints), batch_size):
        selected = endpoints[first : first + batch_size]
        indices = selected[:, None] - offsets[None, :]
        probability.append(
            _predict_windows(models, _normalize(normalizer, features[indices]))
        )
    del features
    return SlipReplay(
        endpoints=endpoints,
        probabilities=np.concatenate(probability)
        if probability
        else np.empty(0, dtype=np.float64),
    )


def replay_many(
    runs: Mapping[str, EventRun],
    run_ids: Sequence[str],
    components: Sequence[str],
    history_ms: int,
    normalizer: Normalizer,
    checkpoint_paths: Sequence[Path],
    *,
    foot_dataset_path: Path | None,
) -> dict[str, SlipReplay]:
    models = [load_checkpoint(path)[0] for path in checkpoint_paths]
    traces = {
        run_id: replay_continuous_slip_run(
            runs[run_id],
            components,
            history_ms,
            normalizer,
            models,
            foot_imu12=_foot_imu_for_run(foot_dataset_path, run_id),
        )
        for run_id in sorted(str(value) for value in run_ids)
    }
    del models
    return traces


def continuous_sustained_alert(
    probabilities: np.ndarray,
    threshold: float,
    persistence_ms: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    return sustained_alert_trace(
        probabilities,
        np.ones(len(probabilities), dtype=bool),
        threshold,
        persistence_ms,
    )


def _negative_interval_mask(run: EventRun, endpoints: np.ndarray) -> np.ndarray:
    slip = slip_event_sample(run)
    support = support_event_sample(run)
    last = run.censor_sample - 1
    if slip is not None:
        last = min(last, slip - 31)
    if support is not None:
        last = min(last, support - 31)
    if run.fall_sample_diagnostic is not None:
        last = min(last, int(run.fall_sample_diagnostic) - 1)
    return np.asarray(endpoints, dtype=np.int64) <= last


def evaluate_continuous_replays(
    runs: Mapping[str, EventRun],
    replays: Mapping[str, SlipReplay],
    threshold: float,
    persistence_ms: int = 5,
) -> dict[str, object]:
    """Separate Slip classification, no-hazard reflexes, and Support cross-triggers."""
    event_rows: list[dict[str, object]] = []
    no_hazard_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    cross_trigger_rows: list[dict[str, object]] = []
    latencies: list[int] = []
    negative_samples = 0
    negative_alert_samples = 0
    for run_id, replay in sorted(replays.items()):
        run = runs[run_id]
        alert, onset = continuous_sustained_alert(
            replay.probabilities, threshold, persistence_ms
        )
        onset_samples = replay.endpoints[onset]
        slip = slip_event_sample(run)
        support = support_event_sample(run)
        negative_mask = _negative_interval_mask(run, replay.endpoints)
        negative_samples += int(np.count_nonzero(negative_mask))
        negative_alert_samples += int(np.count_nonzero(alert & negative_mask))
        premature_samples = onset_samples[negative_mask[onset]]
        interval_rows.append(
            {
                "run_id": run_id,
                "no_hazard_interval_alert": bool(np.any(alert & negative_mask)),
                "negative_alert_samples": int(np.count_nonzero(alert & negative_mask)),
                "negative_samples": int(np.count_nonzero(negative_mask)),
            }
        )
        if slip is not None:
            valid = onset_samples[
                (onset_samples >= slip - 30) & (onset_samples <= slip + 40)
            ]
            late = onset_samples[onset_samples > slip + 40]
            first_valid = None if not len(valid) else int(valid[0])
            side = (
                "BILATERAL"
                if all(value is not None for value in run.slip_event_samples_per_foot)
                else "LEFT_ONLY"
                if run.slip_event_samples_per_foot[0] is not None
                else "RIGHT_ONLY"
            )
            row = {
                "run_id": run_id,
                "event_sample": slip,
                "first_alert_sample": None
                if not len(onset_samples)
                else int(onset_samples[0]),
                "first_valid_detection_sample": first_valid,
                "valid_detection": first_valid is not None,
                "latency_ms": None if first_valid is None else first_valid - slip,
                "premature_alert": bool(len(premature_samples)),
                "premature_episode_count": int(len(premature_samples)),
                "late_detection": first_valid is None and bool(len(late)),
                "negative_alert_duration_ms": int(
                    np.count_nonzero(alert & negative_mask)
                ),
                "source_terrain": run.source_terrain,
                "slip_side": side,
                "outcome_diagnostic_only": run.outcome_diagnostic,
            }
            event_rows.append(row)
            if first_valid is not None:
                latencies.append(first_valid - slip)
        elif support is not None:
            cross = onset_samples[
                (onset_samples >= support - 30) & (onset_samples <= support + 50)
            ]
            cross_trigger_rows.append(
                {
                    "run_id": run_id,
                    "support_event_sample": support,
                    "slip_detector_cross_trigger": bool(len(cross)),
                    "first_cross_trigger_sample": None
                    if not len(cross)
                    else int(cross[0]),
                    "source_terrain": run.source_terrain,
                    "outcome_diagnostic_only": run.outcome_diagnostic,
                }
            )
        else:
            any_alert = bool(np.any(alert))
            no_hazard_rows.append(
                {
                    "run_id": run_id,
                    "system_false_reflex": any_alert,
                    "first_alert_sample": None
                    if not np.any(onset)
                    else int(replay.endpoints[np.flatnonzero(onset)[0]]),
                    "hard_ground": run.hard_stable_control,
                    "target_terrain": run.target_terrain,
                }
            )

    event_count = len(event_rows)
    valid_count = sum(bool(row["valid_detection"]) for row in event_rows)
    premature_count = sum(bool(row["premature_alert"]) for row in event_rows)
    no_hazard_fp = sum(bool(row["system_false_reflex"]) for row in no_hazard_rows)
    interval_fp = sum(bool(row["no_hazard_interval_alert"]) for row in interval_rows)

    def subgroup(field: str, value: str) -> dict[str, int | float]:
        rows = [row for row in event_rows if row[field] == value]
        detected = sum(bool(row["valid_detection"]) for row in rows)
        return {
            "runs": len(rows),
            "detected": detected,
            "recall": 0.0 if not rows else detected / len(rows),
        }

    return {
        "threshold": float(threshold),
        "slip_event_runs": event_count,
        "slip_recall": 0.0 if not event_count else valid_count / event_count,
        "premature_slip_run_rate": 0.0
        if not event_count
        else premature_count / event_count,
        "true_no_hazard_runs": len(no_hazard_rows),
        "true_no_hazard_run_specificity": 1.0
        if not no_hazard_rows
        else 1.0 - no_hazard_fp / len(no_hazard_rows),
        "system_no_hazard_interval_runs": len(interval_rows),
        "system_no_hazard_specificity": 1.0
        if not interval_rows
        else 1.0 - interval_fp / len(interval_rows),
        "negative_time_samples": negative_samples,
        "negative_time_alert_samples": negative_alert_samples,
        "negative_time_alert_fraction": 0.0
        if not negative_samples
        else negative_alert_samples / negative_samples,
        "latency_ms": _distribution(latencies),
        "subgroups": {
            "recovered": subgroup("outcome_diagnostic_only", "VALID_STABLE"),
            "fall": subgroup("outcome_diagnostic_only", "VALID_FALL"),
            "bilateral": subgroup("slip_side", "BILATERAL"),
            "left_only": subgroup("slip_side", "LEFT_ONLY"),
            "right_only": subgroup("slip_side", "RIGHT_ONLY"),
            "concrete_origin": subgroup("source_terrain", "concrete"),
            "marble_origin": subgroup("source_terrain", "marble"),
        },
        "event_rows": event_rows,
        "no_hazard_rows": no_hazard_rows,
        "interval_rows": interval_rows,
        "cross_trigger_rows": cross_trigger_rows,
        "hazard_cross_trigger_count": sum(
            bool(row["slip_detector_cross_trigger"]) for row in cross_trigger_rows
        ),
        "system_false_reflex_count": no_hazard_fp,
    }


def _train_monitor_partition(
    runs: Mapping[str, EventRun], run_ids: Sequence[str], fraction: float
) -> tuple[list[str], list[str]]:
    groups: dict[tuple[str, str, bool, bool], list[str]] = {}
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        key = (
            run.source_terrain,
            run.target_terrain,
            slip_event_sample(run) is not None,
            run.hard_stable_control,
        )
        groups.setdefault(key, []).append(run_id)
    monitor: list[str] = []
    for values in groups.values():
        count = max(1, int(round(len(values) * fraction)))
        indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
        monitor.extend(values[int(index)] for index in indices)
    monitor_set = set(monitor)
    fit = [run_id for run_id in run_ids if run_id not in monitor_set]
    return sorted(fit), sorted(monitor_set)


def _checkpoint_paths(
    artifact_path: Path,
    phase: str,
    candidate_id: str,
    family: str,
    history_ms: int,
    round_id: int,
    seeds: Sequence[int],
) -> tuple[Path, ...]:
    folder = artifact_path / "checkpoints" / phase.lower() / "slip"
    return tuple(
        folder
        / f"{candidate_id.lower()}_{family}_history{history_ms}_round{round_id}_seed{seed}.pt"
        for seed in seeds
    )


def _merge_endpoint_maps(
    *maps: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    keys = {key for mapping in maps for key in mapping}
    return {
        key: tuple(
            sorted({int(value) for mapping in maps for value in mapping.get(key, ())})
        )
        for key in keys
    }


def _compact_replay_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        key: metrics[key]
        for key in (
            "slip_event_runs",
            "slip_recall",
            "premature_slip_run_rate",
            "true_no_hazard_runs",
            "true_no_hazard_run_specificity",
            "system_no_hazard_specificity",
            "negative_time_samples",
            "negative_time_alert_samples",
            "negative_time_alert_fraction",
            "latency_ms",
            "hazard_cross_trigger_count",
            "system_false_reflex_count",
        )
    }


def _mine_training_round(
    runs: Mapping[str, EventRun],
    run_ids: Sequence[str],
    traces: Mapping[str, SlipReplay],
    prior: Mapping[str, Sequence[int]],
    *,
    top_k: int,
    minimum_separation_ms: int,
) -> tuple[dict[str, tuple[int, ...]], dict[str, object]]:
    selected: dict[str, tuple[int, ...]] = {}
    scores: list[float] = []
    for run_id in sorted(str(value) for value in run_ids):
        candidates = continuous_negative_candidates(runs[run_id], history_ms=1)
        trace = traces[run_id]
        common, _, trace_indices = np.intersect1d(
            candidates, trace.endpoints, return_indices=True
        )
        probability = trace.probabilities[trace_indices]
        mined = mine_hard_negative_endpoints(
            common,
            probability,
            top_k=top_k,
            minimum_separation_ms=minimum_separation_ms,
            excluded=prior.get(run_id, ()),
        )
        selected[run_id] = tuple(int(value) for value in mined)
        lookup = {
            int(endpoint): float(score) for endpoint, score in zip(common, probability)
        }
        scores.extend(lookup[int(endpoint)] for endpoint in mined)
    return selected, {
        "source_split": "train",
        "runs_scored": len(run_ids),
        "runs_with_hard_negatives": sum(bool(values) for values in selected.values()),
        "mined_windows": sum(len(values) for values in selected.values()),
        "top_k_per_run": top_k,
        "minimum_separation_ms": minimum_separation_ms,
        "selected_probability": _distribution(scores),
        "support_hazard_regions_excluded": True,
        "positive_slip_regions_excluded": True,
    }


def train_slip_candidate(
    document: Mapping[str, object],
    runs: Mapping[str, EventRun],
    phase: str,
    candidate_id: str,
    components: Sequence[str],
    family: str,
    history_ms: int,
    artifact_path: Path,
    foot_dataset_path: Path | None,
    progress: Callable[[str], None],
) -> SlipCandidateState:
    """Complete Round 0 and exactly three TRAIN-only HNM iterations."""
    train_config = document["training"]
    hnm_config = document["hard_negative_mining"]
    train_ids = sorted(runs)
    fit_ids, monitor_ids = _train_monitor_partition(
        runs,
        train_ids,
        float(train_config["internal_monitor_run_fraction"]),
    )
    schema = feature_schema_for_components(components)
    normalizer = fit_continuous_normalizer(
        runs,
        train_ids,
        components,
        foot_dataset_path=foot_dataset_path,
        per_run_sample_cap=int(train_config["normalizer_per_run_sample_cap"]),
        standard_deviation_floor=float(train_config["standard_deviation_floor"]),
    )
    normalizer_path = (
        artifact_path / "normalization" / phase.lower() / f"{candidate_id.lower()}.json"
    )
    _write_json(
        normalizer_path,
        {
            **normalizer.to_dict(),
            "components": list(components),
            "feature_schema": list(schema),
            "feature_schema_sha256": _canonical_sha256(schema),
            "train_only": True,
        },
    )
    initial_positive_count = sum(
        len(continuous_positive_endpoints(runs[run_id], history_ms))
        for run_id in train_ids
    )
    initial_negative_candidate_count = sum(
        len(continuous_negative_candidates(runs[run_id], history_ms))
        for run_id in train_ids
    )
    if not initial_positive_count or not initial_negative_candidate_count:
        raise RuntimeError("continuous Slip training corpus lacks binary support")
    seeds = [int(value) for value in train_config["seeds"]]
    hnm_maps: list[dict[str, tuple[int, ...]]] = []
    rounds: list[dict[str, object]] = []
    final_paths: tuple[Path, ...] = ()
    for round_id in range(4):
        extras = _merge_endpoint_maps(*hnm_maps) if hnm_maps else {}
        fit_batch = build_continuous_windows(
            runs,
            fit_ids,
            components,
            history_ms,
            normalizer,
            foot_dataset_path=foot_dataset_path,
            per_category=int(train_config["initial_negative_per_gait_category"]),
            contact_shock_count=int(
                train_config["initial_contact_shock_negatives_per_run"]
            ),
            extra_negative_endpoints=extras,
        )
        monitor_batch = build_continuous_windows(
            runs,
            monitor_ids,
            components,
            history_ms,
            normalizer,
            foot_dataset_path=foot_dataset_path,
            per_category=int(train_config["initial_negative_per_gait_category"]),
            contact_shock_count=int(
                train_config["initial_contact_shock_negatives_per_run"]
            ),
            extra_negative_endpoints=extras,
        )
        paths = _checkpoint_paths(
            artifact_path,
            phase,
            candidate_id,
            family,
            history_ms,
            round_id,
            seeds,
        )
        training_rows: list[dict[str, object]] = []
        for seed, path in zip(seeds, paths):
            model, training = train_model(
                family,
                history_ms,
                fit_batch.windows,
                monitor_batch.windows,
                seed,
                batch_size=int(train_config["batch_size"]),
                max_epochs=int(train_config["max_epochs"]),
                patience=int(train_config["patience"]),
                learning_rate=float(train_config["learning_rate"]),
                class_names=EVENT_CLASS_NAMES,
                selection_metric="validation_loss",
            )
            save_checkpoint(
                path,
                model,
                family,
                history_ms,
                seed,
                training,
                input_channels=len(schema),
                class_names=EVENT_CLASS_NAMES,
            )
            training_rows.append(
                {
                    "seed": seed,
                    "best_epoch": training.best_epoch,
                    "epochs_completed": training.epochs_completed,
                    "best_internal_monitor_cross_entropy": min(
                        row["validation_cross_entropy"] for row in training.history
                    ),
                }
            )
            del model
        final_paths = paths
        traces = replay_many(
            runs,
            train_ids,
            components,
            history_ms,
            normalizer,
            paths,
            foot_dataset_path=foot_dataset_path,
        )
        train_metrics = evaluate_continuous_replays(
            runs,
            traces,
            0.5,
            int(document["validation"]["persistence_ms"]),
        )
        row: dict[str, object] = {
            "round": round_id,
            "fit_windows": len(fit_batch.windows),
            "fit_positive_windows": int(
                np.count_nonzero(fit_batch.windows.targets == 1)
            ),
            "fit_negative_windows": int(
                np.count_nonzero(fit_batch.windows.targets == 0)
            ),
            "monitor_windows": len(monitor_batch.windows),
            "independent_train_runs": len(train_ids),
            "training": training_rows,
            "train_continuous_replay_threshold_0_5": _compact_replay_metrics(
                train_metrics
            ),
        }
        if round_id < 3:
            prior = _merge_endpoint_maps(*hnm_maps) if hnm_maps else {}
            mined, mining = _mine_training_round(
                runs,
                train_ids,
                traces,
                prior,
                top_k=int(hnm_config["top_k_per_run"]),
                minimum_separation_ms=int(hnm_config["minimum_separation_ms"]),
            )
            hnm_maps.append(mined)
            row["mining_for_next_round"] = mining
        rounds.append(row)
        progress(
            f"{phase} {candidate_id} {family} {history_ms}ms ROUND {round_id} "
            f"windows={len(fit_batch.windows)} "
            f"train_fp={1.0-float(train_metrics['true_no_hazard_run_specificity']):.3f}"
        )
        del traces, fit_batch, monitor_batch
        gc.collect()
    record = {
        "phase": phase,
        "candidate_id": candidate_id,
        "components": list(components),
        "feature_dimension": len(schema),
        "feature_schema_sha256": _canonical_sha256(schema),
        "model_family": family,
        "history_ms": history_ms,
        "physical_channels": sum(PHYSICAL_CHANNELS[value] for value in set(components)),
        "parameter_count": parameter_count(load_checkpoint(final_paths[0])[0]),
        "independent_train_runs": len(train_ids),
        "fit_runs": len(fit_ids),
        "internal_monitor_runs": len(monitor_ids),
        "initial_positive_endpoints": initial_positive_count,
        "initial_negative_candidates": initial_negative_candidate_count,
        "training_status": "ROUND_0_HNM_1_ROUND_1_HNM_2_ROUND_2_HNM_3_ROUND_3_COMPLETE",
        "rounds": rounds,
        "checkpoint_paths": [
            str(path.relative_to(artifact_path)) for path in final_paths
        ],
        "validation": None,
    }
    return SlipCandidateState(record, normalizer, final_paths)


def threshold_values(grid: Mapping[str, object]) -> tuple[float, ...]:
    start, stop, step = (float(grid[name]) for name in ("start", "stop", "step"))
    count = int(round((stop - start) / step))
    values = tuple(round(start + index * step, 10) for index in range(count + 1))
    if values != tuple(round(0.10 + 0.02 * index, 10) for index in range(45)):
        raise ValueError("Slip threshold grid must remain 0.10..0.98 step 0.02")
    return values


def validation_gate_results(
    metrics: Mapping[str, object], gates: Mapping[str, object]
) -> dict[str, bool]:
    latency = metrics["latency_ms"]
    return {
        "slip_recall": float(metrics["slip_recall"]) >= float(gates["slip_recall_min"]),
        "premature_slip_run_rate": float(metrics["premature_slip_run_rate"])
        <= float(gates["premature_slip_run_rate_max"]),
        "true_no_hazard_run_specificity": float(
            metrics["true_no_hazard_run_specificity"]
        )
        >= float(gates["true_no_hazard_run_specificity_min"]),
        "system_no_hazard_specificity": float(metrics["system_no_hazard_specificity"])
        >= float(gates["system_no_hazard_specificity_min"]),
        "negative_time_alert_fraction": float(metrics["negative_time_alert_fraction"])
        <= float(gates["negative_time_alert_fraction_max"]),
        "median_latency_ms": latency["median"] is not None
        and float(latency["median"]) <= float(gates["median_latency_ms_max"]),
        "p95_latency_ms": latency["p95"] is not None
        and float(latency["p95"]) <= float(gates["p95_latency_ms_max"]),
    }


def select_validation_threshold(
    evaluations: Sequence[Mapping[str, object]], gates: Mapping[str, object]
) -> dict[str, object]:
    rows = []
    for evaluation in evaluations:
        checks = validation_gate_results(evaluation["metrics"], gates)
        rows.append(
            {**dict(evaluation), "gates": checks, "passed": all(checks.values())}
        )

    def rank(row: Mapping[str, object]) -> tuple[float, ...]:
        metrics = row["metrics"]
        p95 = metrics["latency_ms"]["p95"]
        return (
            float(metrics["slip_recall"]),
            float(metrics["true_no_hazard_run_specificity"]),
            -float(metrics["premature_slip_run_rate"]),
            -9999.0 if p95 is None else -float(p95),
            float(row["threshold"]),
        )

    passing = [row for row in rows if bool(row["passed"])]
    if passing:
        selected = max(passing, key=rank)
        return {
            "selected": selected,
            "diagnostic_best": selected,
            "passing_threshold_count": len(passing),
        }
    diagnostic = max(
        rows,
        key=lambda row: (
            sum(bool(value) for value in row["gates"].values()),
            *rank(row),
        ),
    )
    return {
        "selected": None,
        "diagnostic_best": diagnostic,
        "passing_threshold_count": 0,
    }


def validate_slip_candidate(
    document: Mapping[str, object],
    state: SlipCandidateState,
    validation_runs: Mapping[str, EventRun],
    *,
    foot_dataset_path: Path | None,
) -> dict[str, object]:
    record = state.record
    traces = replay_many(
        validation_runs,
        sorted(validation_runs),
        tuple(str(value) for value in record["components"]),
        int(record["history_ms"]),
        state.normalizer,
        state.checkpoint_paths,
        foot_dataset_path=foot_dataset_path,
    )
    evaluations = [
        {
            "threshold": threshold,
            "metrics": evaluate_continuous_replays(
                validation_runs,
                traces,
                threshold,
                int(document["validation"]["persistence_ms"]),
            ),
        }
        for threshold in threshold_values(document["validation"]["threshold_grid"])
    ]
    selection = select_validation_threshold(
        evaluations, document["validation"]["gates"]
    )
    operating = selection["selected"] or selection["diagnostic_best"]
    result = {
        "passed": selection["selected"] is not None,
        "selected_threshold": None
        if selection["selected"] is None
        else float(selection["selected"]["threshold"]),
        "passing_threshold_count": int(selection["passing_threshold_count"]),
        "operating_threshold": float(operating["threshold"]),
        "metrics": operating["metrics"],
        "gates": operating["gates"],
    }
    record["validation"] = result
    del traces
    gc.collect()
    return result


def candidate_identity(state: SlipCandidateState) -> dict[str, object]:
    record = state.record
    validation = record["validation"]
    return {
        "phase": record["phase"],
        "candidate_id": record["candidate_id"],
        "components": list(record["components"]),
        "feature_dimension": record["feature_dimension"],
        "feature_schema_sha256": record["feature_schema_sha256"],
        "model_family": record["model_family"],
        "history_ms": record["history_ms"],
        "physical_channels": record["physical_channels"],
        "parameter_count": record["parameter_count"],
        "threshold": validation["selected_threshold"],
        "persistence_ms": 5,
        "checkpoint_paths": list(record["checkpoint_paths"]),
    }


def select_slip_candidate(states: Sequence[SlipCandidateState]) -> dict[str, object]:
    passing = [state for state in states if bool(state.record["validation"]["passed"])]
    if not passing:
        diagnostic = max(
            states,
            key=lambda state: (
                sum(
                    bool(value)
                    for value in state.record["validation"]["gates"].values()
                ),
                float(state.record["validation"]["metrics"]["slip_recall"]),
                float(
                    state.record["validation"]["metrics"][
                        "true_no_hazard_run_specificity"
                    ]
                ),
                -float(state.record["physical_channels"]),
            ),
        )
        return {
            "selected": None,
            "reason": "no_candidate_passed_all_continuous_slip_validation_gates",
            "diagnostic_best": candidate_identity(diagnostic),
        }

    def primary_rank(state: SlipCandidateState) -> tuple[float, ...]:
        record = state.record
        metrics = record["validation"]["metrics"]
        p95 = metrics["latency_ms"]["p95"]
        return (
            float(metrics["slip_recall"]),
            float(metrics["true_no_hazard_run_specificity"]),
            -float(metrics["premature_slip_run_rate"]),
            -9999.0 if p95 is None else -float(p95),
            -float(record["physical_channels"]),
            -float(record["history_ms"]),
            -float(record["parameter_count"]),
        )

    best = max(passing, key=primary_rank)
    best_metrics = best.record["validation"]["metrics"]
    best_p95 = float(best_metrics["latency_ms"]["p95"])
    near = [
        state
        for state in passing
        if abs(
            float(state.record["validation"]["metrics"]["slip_recall"])
            - float(best_metrics["slip_recall"])
        )
        <= 0.02
        and abs(
            float(
                state.record["validation"]["metrics"]["true_no_hazard_run_specificity"]
            )
            - float(best_metrics["true_no_hazard_run_specificity"])
        )
        <= 0.02
        and abs(
            float(state.record["validation"]["metrics"]["latency_ms"]["p95"]) - best_p95
        )
        <= 5.0
    ]
    selected = min(
        near,
        key=lambda state: (
            int(state.record["physical_channels"]),
            int(state.record["history_ms"]),
            int(state.record["parameter_count"]),
            -float(state.record["validation"]["metrics"]["slip_recall"]),
        ),
    )
    return {
        "selected": candidate_identity(selected),
        "reason": "all_gates_with_predeclared_near_tie_minimum_sensor_rule",
    }


def _state_by_identity(
    states: Sequence[SlipCandidateState], identity: Mapping[str, object]
) -> SlipCandidateState:
    for state in states:
        if (
            state.record["phase"] == identity["phase"]
            and state.record["candidate_id"] == identity["candidate_id"]
            and state.record["model_family"] == identity["model_family"]
            and state.record["history_ms"] == identity["history_ms"]
        ):
            return state
    raise KeyError("selected continuous Slip candidate state is absent")


def _candidate_grid(
    phase: str,
) -> tuple[tuple[str, tuple[str, ...], str, int], ...]:
    candidates = PHASE_A_CANDIDATES if phase == "PHASE_A" else PHASE_B_CANDIDATES
    return tuple(
        (candidate_id, components, family, history)
        for candidate_id, components in candidates.items()
        for family in ("mlp", "gru")
        for history in (20, 50)
    )


def run_phase_training(
    document: Mapping[str, object],
    runs: Mapping[str, EventRun],
    phase: str,
    artifact_path: Path,
    foot_dataset_path: Path | None,
    progress: Callable[[str], None],
) -> list[SlipCandidateState]:
    return [
        train_slip_candidate(
            document,
            runs,
            phase,
            candidate_id,
            components,
            family,
            history,
            artifact_path,
            foot_dataset_path,
            progress,
        )
        for candidate_id, components, family, history in _candidate_grid(phase)
    ]


def run_phase_validation(
    document: Mapping[str, object],
    states: Sequence[SlipCandidateState],
    validation_runs: Mapping[str, EventRun],
    foot_dataset_path: Path | None,
    progress: Callable[[str], None],
) -> dict[str, object]:
    for state in states:
        record = state.record
        result = validate_slip_candidate(
            document,
            state,
            validation_runs,
            foot_dataset_path=foot_dataset_path,
        )
        progress(
            f"{record['phase']} {record['candidate_id']} {record['model_family']} "
            f"{record['history_ms']}ms VALIDATION pass={result['passed']} "
            f"recall={result['metrics']['slip_recall']:.3f} "
            f"specificity={result['metrics']['true_no_hazard_run_specificity']:.3f}"
        )
    return select_slip_candidate(states)


def reflex_decision(
    *, slip_alert: bool, support_alert: bool, terrain_state: int
) -> tuple[bool, str]:
    """Asymmetric safety action with later advisory cause refinement."""
    if slip_alert:
        return True, "SLIP_RISK" if terrain_state == ICE else "GENERIC_SLIP_DISTURBANCE"
    if support_alert and terrain_state == SAND:
        return True, "SUPPORT_RISK"
    return False, "NORMAL"


def audit_event_dataset(
    dataset_path: Path, manifest: Mapping[str, object]
) -> dict[str, object]:
    rows = list(manifest["runs"])
    if int(manifest["run_count"]) != len(rows):
        raise ValueError("event manifest run count mismatch")
    integrity = []
    for row in rows:
        path = dataset_path / str(row["file"])
        integrity.append(
            path.is_file() and _file_sha256(path) == str(row["file_sha256"])
        )
    if not all(integrity):
        raise ValueError("existing event corpus integrity audit failed")
    signatures = [tuple(row["physical_signature"]) for row in rows]
    split_by_id = {str(row["run_id"]): str(row["split"]) for row in rows}
    if len(split_by_id) != len(rows) or len(set(signatures)) != len(signatures):
        raise ValueError("event corpus contains duplicate ID or physical signature")

    def count(split: str, event_type: str | None = None) -> int:
        return sum(
            row["split"] == split
            and (event_type is None or row["event_type"] == event_type)
            for row in rows
        )

    slip_rows = [
        row for row in rows if row["event_type"] in (EVENT_TYPE_SLIP, EVENT_TYPE_BOTH)
    ]
    no_hazard = [row for row in rows if row["event_type"] == EVENT_TYPE_NONE]
    return {
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": _file_sha256(dataset_path / "manifest.json"),
        "run_count": len(rows),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "split": {
            split: {
                "total": count(split),
                "slip": sum(row["split"] == split for row in slip_rows),
                "support": count(split, EVENT_TYPE_SUPPORT),
                "no_event": count(split, EVENT_TYPE_NONE),
            }
            for split in ("train", "validation", "holdout")
        },
        "slip": {
            "total": len(slip_rows),
            "bilateral": sum(row["slip_side"] == "BILATERAL" for row in slip_rows),
            "left_only": sum(row["slip_side"] == "LEFT" for row in slip_rows),
            "right_only": sum(row["slip_side"] == "RIGHT" for row in slip_rows),
            "recovered": sum(
                row["observed_outcome_diagnostic_only"] == "VALID_STABLE"
                for row in slip_rows
            ),
            "fall": sum(
                row["observed_outcome_diagnostic_only"] == "VALID_FALL"
                for row in slip_rows
            ),
        },
        "no_hazard_controls": {
            "total_no_event": len(no_hazard),
            "hard_concrete": sum(
                row["hard_stable_control"] and row["target_terrain"] == "concrete"
                for row in no_hazard
            ),
            "hard_marble": sum(
                row["hard_stable_control"] and row["target_terrain"] == "marble"
                for row in no_hazard
            ),
            "benign_sand": sum(
                row["target_terrain"] == "sand" and not row["hard_stable_control"]
                for row in no_hazard
            ),
        },
        "duplicate_signature_count": len(signatures) - len(set(signatures)),
        "split_overlap_count": len(rows) - len(split_by_id),
        "file_integrity_passed": all(integrity),
        "holdout_waveform_opened": False,
    }


def audit_foot_imu_dataset(
    event_manifest_path: Path,
    event_manifest: Mapping[str, object],
    foot_dataset_path: Path,
    prior_parity_path: Path,
) -> dict[str, object]:
    manifest_path = foot_dataset_path / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with prior_parity_path.open("r", encoding="utf-8") as stream:
        prior_parity = json.load(stream)
    event_split = {
        str(row["run_id"]): str(row["split"]) for row in event_manifest["runs"]
    }
    observer_split = {str(row["run_id"]): str(row["split"]) for row in manifest["runs"]}
    integrity = [
        (foot_dataset_path / str(row["file"])).is_file()
        and _file_sha256(foot_dataset_path / str(row["file"]))
        == str(row["file_sha256"])
        for row in manifest["runs"]
    ]
    checks = {
        "run_count": int(manifest["run_count"]) == int(event_manifest["run_count"]),
        "source_manifest": manifest["source_event_manifest_sha256"]
        == _file_sha256(event_manifest_path),
        "split_parity": event_split == observer_split,
        "file_integrity": all(integrity),
        "event_clock_parity": bool(
            manifest["observer_parity"]["all_runs_clock_and_outcome_exact"]
        ),
        "policy_action_contact_physics_parity": bool(prior_parity.get("passed")),
        "observer_only": bool(manifest["observer_parity"]["foot_imu_observer_only"]),
        "holdout_selection_access": not bool(
            manifest.get("holdout_waveform_accessed_during_generation", False)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("existing Foot IMU observer corpus parity failed")
    return {
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": _file_sha256(manifest_path),
        "run_count": int(manifest["run_count"]),
        "size_bytes": sum(int(row["size_bytes"]) for row in manifest["runs"]),
        "reused_existing_generated_dataset": True,
        "regenerated": False,
        "checks": checks,
        "passed": True,
    }


def verify_and_prepare_frozen_support(
    document: Mapping[str, object],
    train_runs: Mapping[str, EventRun],
    repository_root: Path,
) -> tuple[dict[str, object], Normalizer, tuple[Path, ...]]:
    selection_path = repository_root / str(document["source"]["previous_selection"])
    metrics_path = repository_root / str(document["source"]["previous_reflex_metrics"])
    previous_config_path = repository_root / str(
        document["source"]["previous_reflex_config"]
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    previous_document = _load_yaml(previous_config_path)
    frozen = document["frozen_support"]
    selected = selection["final_selection"]["support"]["selected"]
    expected = {
        "components": list(frozen["components"]),
        "model_family": frozen["model_family"],
        "history_ms": int(frozen["history_ms"]),
        "threshold": float(frozen["probability_threshold"]),
        "persistence_ms": int(frozen["persistence_ms"]),
        "feature_schema_sha256": frozen["feature_schema_sha256"],
    }
    checks = {key: selected[key] == value for key, value in expected.items()}
    checks["previous_holdout_sealed"] = selection["holdout_opened"] is False
    checks["previous_verdict"] = metrics["supported_branch"] == "SUPPORT_ONLY"
    checkpoint_paths = tuple(
        repository_root / str(value) for value in frozen["checkpoints"]
    )
    checks["checkpoint_count"] = len(checkpoint_paths) == 3
    checks["checkpoints_present"] = all(path.is_file() for path in checkpoint_paths)
    if not all(checks.values()):
        raise RuntimeError("frozen Support candidate provenance changed")
    support_ids = _relevant_run_ids(train_runs, "support", "train")
    normalizer = fit_frozen_support_normalizer(
        train_runs,
        support_ids,
        tuple(str(value) for value in frozen["components"]),
        foot_dataset_path=None,
        per_run_sample_cap=int(
            previous_document["training"]["normalizer_per_run_sample_cap"]
        ),
        standard_deviation_floor=float(
            previous_document["training"]["standard_deviation_floor"]
        ),
    )
    return (
        {
            "checks": checks,
            "passed": True,
            "retrained": False,
            "identity": expected,
            "selection_sha256": _file_sha256(selection_path),
            "metrics_sha256": _file_sha256(metrics_path),
            "config_sha256": _file_sha256(previous_config_path),
            "checkpoint_sha256": {
                str(path.relative_to(repository_root)): _file_sha256(path)
                for path in checkpoint_paths
            },
            "validation_metrics": metrics["phase_a"]["selection"]["support"],
        },
        normalizer,
        checkpoint_paths,
    )


def holdout_slip_gate_results(
    metrics: Mapping[str, object], gates: Mapping[str, object]
) -> dict[str, bool]:
    latency = metrics["latency_ms"]
    return {
        "recall": float(metrics["slip_recall"]) >= float(gates["recall_min"]),
        "no_hazard_run_specificity": float(metrics["true_no_hazard_run_specificity"])
        >= float(gates["no_hazard_run_specificity_min"]),
        "premature_rate": float(metrics["premature_slip_run_rate"])
        <= float(gates["premature_rate_max"]),
        "negative_time_alert_fraction": float(metrics["negative_time_alert_fraction"])
        <= float(gates["negative_time_alert_fraction_max"]),
        "median_latency_ms": latency["median"] is not None
        and float(latency["median"]) <= float(gates["median_latency_ms_max"]),
        "p95_latency_ms": latency["p95"] is not None
        and float(latency["p95"]) <= float(gates["p95_latency_ms_max"]),
    }


def holdout_support_gate_results(
    metrics: Mapping[str, object], gates: Mapping[str, object]
) -> dict[str, bool]:
    latency = metrics["latency_ms"]
    return {
        "recall": float(metrics["event_recall"]) >= float(gates["recall_min"]),
        "sand_benign_specificity": float(metrics["benign_specificity"])
        >= float(gates["sand_benign_specificity_min"]),
        "premature_rate": float(metrics["premature_run_rate"])
        <= float(gates["premature_rate_max"]),
        "median_latency_ms": latency["median"] is not None
        and float(latency["median"]) <= float(gates["median_latency_ms_max"]),
        "p95_latency_ms": latency["p95"] is not None
        and float(latency["p95"]) <= float(gates["p95_latency_ms_max"]),
    }


def _integrated_holdout_metrics(
    runs: Mapping[str, EventRun],
    slip_replays: Mapping[str, SlipReplay],
    slip_metrics: Mapping[str, object],
    slip_threshold: float,
    support_replays: Mapping[str, BranchReplay],
    support_metrics: Mapping[str, object],
    terrain_gates: Mapping[str, TerrainGateTrace],
    support_threshold: float,
    persistence_ms: int,
    gates: Mapping[str, object],
) -> dict[str, object]:
    slip_events = {row["run_id"]: row for row in slip_metrics["event_rows"]}
    support_events = {row["run_id"]: row for row in support_metrics["event_rows"]}
    support_cross = {row["run_id"]: row for row in slip_metrics["cross_trigger_rows"]}
    event_rows: list[dict[str, object]] = []
    no_hazard_rows: list[dict[str, object]] = []
    for run_id, run in sorted(runs.items()):
        slip_replay = slip_replays[run_id]
        slip_alert, _ = continuous_sustained_alert(
            slip_replay.probabilities, slip_threshold, persistence_ms
        )
        support_replay = support_replays[run_id]
        support_alert, _ = sustained_alert_trace(
            support_replay.probabilities,
            support_replay.terrain_state == SAND,
            support_threshold,
            persistence_ms,
        )
        slip = slip_event_sample(run)
        support = support_event_sample(run)
        if slip is not None or support is not None:
            slip_valid = bool(slip_events.get(run_id, {}).get("valid_detection", False))
            support_valid = bool(
                support_events.get(run_id, {}).get("valid_detection", False)
            )
            cross = bool(
                support_cross.get(run_id, {}).get("slip_detector_cross_trigger", False)
            )
            detected = slip_valid if slip is not None else support_valid or cross
            event_rows.append(
                {
                    "run_id": run_id,
                    "physical_event": "SLIP" if slip is not None else "SUPPORT",
                    "detected": detected,
                    "native_branch_detection": slip_valid or support_valid,
                    "wrong_provisional_cause_cross_trigger": cross,
                    "outcome_diagnostic_only": run.outcome_diagnostic,
                }
            )
        else:
            false_reflex = bool(np.any(slip_alert) or np.any(support_alert))
            no_hazard_rows.append(
                {
                    "run_id": run_id,
                    "system_false_reflex": false_reflex,
                    "hard_ground": run.hard_stable_control,
                }
            )
    event_recall = (
        0.0
        if not event_rows
        else sum(bool(row["detected"]) for row in event_rows) / len(event_rows)
    )
    no_hazard_specificity = (
        1.0
        if not no_hazard_rows
        else 1.0
        - sum(bool(row["system_false_reflex"]) for row in no_hazard_rows)
        / len(no_hazard_rows)
    )
    hard = [row for row in no_hazard_rows if row["hard_ground"]]
    hard_specificity = (
        1.0
        if not hard
        else 1.0 - sum(bool(row["system_false_reflex"]) for row in hard) / len(hard)
    )
    metrics = {
        "event_runs": len(event_rows),
        "event_recall": event_recall,
        "no_hazard_runs": len(no_hazard_rows),
        "no_hazard_specificity": no_hazard_specificity,
        "hard_ground_runs": len(hard),
        "hard_ground_specificity": hard_specificity,
        "hazard_cross_trigger_count": sum(
            bool(row["wrong_provisional_cause_cross_trigger"]) for row in event_rows
        ),
        "system_false_reflex_count": sum(
            bool(row["system_false_reflex"]) for row in no_hazard_rows
        ),
        "event_rows": event_rows,
        "no_hazard_rows": no_hazard_rows,
    }
    checks = {
        "event_recall": event_recall >= float(gates["event_recall_min"]),
        "no_hazard_specificity": no_hazard_specificity
        >= float(gates["no_hazard_specificity_min"]),
        "hard_ground_specificity": hard_specificity
        >= float(gates["hard_ground_specificity_min"]),
    }
    return {"metrics": metrics, "gates": checks, "passed": all(checks.values())}


def evaluate_one_shot_holdout(
    document: Mapping[str, object],
    selected_state: SlipCandidateState,
    holdout_runs: Mapping[str, EventRun],
    terrain_gates: Mapping[str, TerrainGateTrace],
    support_normalizer: Normalizer,
    support_checkpoints: Sequence[Path],
    foot_dataset_path: Path,
) -> dict[str, object]:
    identity = candidate_identity(selected_state)
    phase_b = identity["phase"] == "PHASE_B"
    slip_replays = replay_many(
        holdout_runs,
        sorted(holdout_runs),
        tuple(str(value) for value in identity["components"]),
        int(identity["history_ms"]),
        selected_state.normalizer,
        selected_state.checkpoint_paths,
        foot_dataset_path=foot_dataset_path if phase_b else None,
    )
    slip_metrics = evaluate_continuous_replays(
        holdout_runs,
        slip_replays,
        float(identity["threshold"]),
        int(document["validation"]["persistence_ms"]),
    )
    slip_checks = holdout_slip_gate_results(
        slip_metrics, document["holdout"]["slip_gates"]
    )
    support = document["frozen_support"]
    support_replays = replay_frozen_support_many(
        holdout_runs,
        terrain_gates,
        sorted(holdout_runs),
        tuple(str(value) for value in support["components"]),
        int(support["history_ms"]),
        support_normalizer,
        support_checkpoints,
        None,
    )
    support_metrics = evaluate_frozen_support_replays(
        holdout_runs,
        terrain_gates,
        support_replays,
        "support",
        float(support["probability_threshold"]),
        int(support["persistence_ms"]),
    )
    support_checks = holdout_support_gate_results(
        support_metrics, document["holdout"]["support_gates"]
    )
    integrated = _integrated_holdout_metrics(
        holdout_runs,
        slip_replays,
        slip_metrics,
        float(identity["threshold"]),
        support_replays,
        support_metrics,
        terrain_gates,
        float(support["probability_threshold"]),
        int(document["validation"]["persistence_ms"]),
        document["holdout"]["integrated_gates"],
    )
    timing = terrain_timing_audit(holdout_runs, terrain_gates)
    reaction_rows = []
    for row in slip_metrics["event_rows"]:
        run_id = str(row["run_id"])
        terrain_valid = terrain_gates[run_id].first_target_valid_sample
        detection = row["first_valid_detection_sample"]
        reaction_rows.append(
            {
                "run_id": run_id,
                "slip_sample": row["event_sample"],
                "detection_sample": detection,
                "terrain_valid_sample": terrain_valid,
                "detector_before_terrain": detection is not None
                and (terrain_valid is None or int(detection) < int(terrain_valid)),
                "detection_minus_slip_ms": row["latency_ms"],
            }
        )
    detected_rows = [
        row for row in reaction_rows if row["detection_sample"] is not None
    ]
    return {
        "performed": True,
        "reselection_performed": False,
        "slip": {
            "selection": identity,
            "metrics": slip_metrics,
            "gates": slip_checks,
            "passed": all(slip_checks.values()),
        },
        "support": {
            "selection": dict(support),
            "metrics": support_metrics,
            "gates": support_checks,
            "passed": all(support_checks.values()),
        },
        "terrain": {
            "candidate": "FSR4_MLP_50MS_LEFT_ONLY",
            "retrained": False,
            "timing": timing,
        },
        "integrated": integrated,
        "reaction_timing": {
            "rows": reaction_rows,
            "slip_detector_before_terrain_count": sum(
                bool(row["detector_before_terrain"]) for row in detected_rows
            ),
            "detected_slip_count": len(detected_rows),
            "slip_detector_before_terrain_rate": 0.0
            if not detected_rows
            else sum(bool(row["detector_before_terrain"]) for row in detected_rows)
            / len(detected_rows),
            "slip_latency_ms": slip_metrics["latency_ms"],
            "support_latency_ms": support_metrics["latency_ms"],
        },
        "passed": all(slip_checks.values())
        and all(support_checks.values())
        and bool(integrated["passed"]),
    }


def _freeze_slip_candidate(
    document: Mapping[str, object],
    state: SlipCandidateState,
    config_path: Path,
    artifact_path: Path,
    repository_root: Path,
) -> dict[str, object]:
    identity = candidate_identity(state)
    normalizer_path = (
        artifact_path
        / "normalization"
        / str(identity["phase"]).lower()
        / f"{str(identity['candidate_id']).lower()}.json"
    )
    provenance = {
        "selection": identity,
        "source_commit": document["experiment"]["source_commit_at_start"],
        "experiment_config": str(config_path.relative_to(repository_root)),
        "experiment_config_sha256": _file_sha256(config_path),
        "normalizer": str(normalizer_path.relative_to(repository_root)),
        "normalizer_sha256": _file_sha256(normalizer_path),
        "checkpoint_sha256": {
            str(path.relative_to(repository_root)): _file_sha256(path)
            for path in state.checkpoint_paths
        },
        "terrain_gate_required": False,
        "terrain_identity_in_tensor": False,
        "fall_or_recovery_in_label": False,
        "persistence_ms": 5,
        "timing_contract_ms": [-30, 40],
    }
    provenance["artifact_sha256"] = _canonical_sha256(provenance)
    _write_json(artifact_path / "slip_candidate_freeze.json", provenance)
    return provenance


def _sensor_recommendation(selection: Mapping[str, object] | None) -> dict[str, object]:
    terrain_channels = 4
    support_channels = 6
    if selection is None:
        return {
            "terrain_branch": {"sensors": ["left_fsr4"], "channels": terrain_channels},
            "support_branch": {
                "sensors": ["pelvis_imu6"],
                "channels": support_channels,
            },
            "slip_branch": None,
            "total_unique_physical_channels": terrain_channels + support_channels,
            "final_sensor_architecture_frozen": False,
        }
    components = set(str(value) for value in selection["components"])
    fsr_channels = 8 if "fsr8" in components else terrain_channels
    foot_channels = 12 if "foot_imu12" in components else 0
    return {
        "terrain_branch": {"sensors": ["left_fsr4"], "channels": terrain_channels},
        "support_branch": {"sensors": ["pelvis_imu6"], "channels": support_channels},
        "slip_branch": {
            "sensors": sorted(components),
            "channels": int(selection["physical_channels"]),
        },
        "total_unique_physical_channels": fsr_channels
        + support_channels
        + foot_channels,
        "deduplicated_shared_channels": True,
        "final_sensor_architecture_frozen": False,
    }


def run_continuous_slip_reflex_detector(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Execute the bounded Phase A -> conditional Phase B -> holdout workflow."""
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    document = _load_yaml(config_path)
    if document["experiment"]["id"] != "CONTINUOUS_SLIP_REFLEX_DETECTOR_DEVELOPMENT":
        raise ValueError("unsupported continuous Slip experiment")
    artifact_path = repository_root / str(document["artifacts"]["path"])
    artifact_path.mkdir(parents=True, exist_ok=True)
    event_dataset = repository_root / str(document["source"]["event_dataset"])
    event_manifest_path = repository_root / str(document["source"]["event_manifest"])
    event_manifest = json.loads(event_manifest_path.read_text(encoding="utf-8"))
    dataset_audit = audit_event_dataset(event_dataset, event_manifest)
    foot_dataset_path = repository_root / str(document["source"]["foot_imu_dataset"])
    foot_audit = audit_foot_imu_dataset(
        event_manifest_path,
        event_manifest,
        foot_dataset_path,
        repository_root / str(document["source"]["previous_foot_parity"]),
    )
    protected_terrain_paths = [
        str(value) for value in document["protected_terrain_paths"]
    ]
    terrain_before = _protected_hashes(repository_root, protected_terrain_paths)
    support_paths = [
        str(document["source"]["previous_reflex_config"]),
        str(document["source"]["previous_reflex_metrics"]),
        str(document["source"]["previous_selection"]),
        *(str(value) for value in document["frozen_support"]["checkpoints"]),
    ]
    support_before = _protected_hashes(repository_root, support_paths)

    # VALIDATION is deliberately not loaded before every Phase A Round 3 exists.
    train_runs = load_event_runs(event_dataset, event_manifest, ("train",))
    (
        support_verification,
        support_normalizer,
        support_checkpoints,
    ) = verify_and_prepare_frozen_support(document, train_runs, repository_root)
    progress("PHASE A continuous Slip training begins; VALIDATION remains unopened")
    phase_a_states = run_phase_training(
        document,
        train_runs,
        "PHASE_A",
        artifact_path,
        None,
        progress,
    )
    progress("All Phase A Round 3 checkpoints exist; opening VALIDATION")
    validation_runs = load_event_runs(event_dataset, event_manifest, ("validation",))
    phase_a_selection = run_phase_validation(
        document, phase_a_states, validation_runs, None, progress
    )

    phase_b_states: list[SlipCandidateState] = []
    phase_b_selection: dict[str, object] = {
        "selected": None,
        "activated": False,
        "reason": "phase_a_has_supported_candidate",
    }
    final_selection = phase_a_selection
    selected_states = phase_a_states
    if phase_a_selection["selected"] is None:
        progress(
            "All Phase A candidates failed; activating parity-proven Foot IMU Phase B"
        )
        phase_b_states = run_phase_training(
            document,
            train_runs,
            "PHASE_B",
            artifact_path,
            foot_dataset_path,
            progress,
        )
        progress("All Phase B Round 3 checkpoints exist; replaying VALIDATION")
        phase_b_selection = {
            **run_phase_validation(
                document,
                phase_b_states,
                validation_runs,
                foot_dataset_path,
                progress,
            ),
            "activated": True,
        }
        final_selection = phase_b_selection
        selected_states = phase_b_states

    selected_state: SlipCandidateState | None = None
    freeze: dict[str, object] | None = None
    if final_selection["selected"] is not None:
        selected_state = _state_by_identity(
            selected_states, final_selection["selected"]
        )
        freeze = _freeze_slip_candidate(
            document, selected_state, config_path, artifact_path, repository_root
        )
    _write_json(
        artifact_path / "selection_before_holdout.json",
        {
            "slip_selection": final_selection,
            "slip_candidate_frozen": selected_state is not None,
            "frozen_support_verified": support_verification["passed"],
            "holdout_opened": False,
        },
    )

    guard = EventHoldoutGuard()
    if selected_state is not None:
        guard.open_once()
        holdout_runs = load_event_runs(
            event_dataset, event_manifest, ("holdout",), holdout_guard=guard
        )
        previous_document = _load_yaml(
            repository_root / str(document["source"]["previous_reflex_config"])
        )
        terrain_gates = holdout_gate_from_observer_dataset(
            previous_document, holdout_runs, foot_dataset_path, repository_root
        )
        holdout = evaluate_one_shot_holdout(
            document,
            selected_state,
            holdout_runs,
            terrain_gates,
            support_normalizer,
            support_checkpoints,
            foot_dataset_path,
        )
        holdout["guard_open_count"] = guard.open_count
    else:
        holdout = {
            "performed": False,
            "guard_open_count": guard.open_count,
            "reason": "no_validation_supported_continuous_slip_candidate",
        }

    terrain_after = _protected_hashes(repository_root, protected_terrain_paths)
    support_after = _protected_hashes(repository_root, support_paths)
    if selected_state is None:
        verdict = "CONTINUOUS_SLIP_REFLEX_NOT_SUPPORTED"
    elif not bool(holdout["passed"]):
        verdict = "CONTINUOUS_SLIP_REFLEX_PROMISING"
    elif selected_state.record["phase"] == "PHASE_B":
        verdict = "ASYMMETRIC_REFLEX_ARCHITECTURE_SUPPORTED_WITH_FOOT_IMU"
    else:
        verdict = "ASYMMETRIC_REFLEX_ARCHITECTURE_SUPPORTED_EXISTING_SENSORS"
    selected_identity = (
        None if selected_state is None else candidate_identity(selected_state)
    )
    metrics = {
        "experiment": document["experiment"],
        "dataset_audit": dataset_audit,
        "foot_imu_dataset_audit": foot_audit,
        "phase_a": {
            "candidates": [state.record for state in phase_a_states],
            "selection": phase_a_selection,
        },
        "phase_b": {
            "activated": bool(phase_b_states),
            "candidates": [state.record for state in phase_b_states],
            "selection": phase_b_selection,
            "dataset_reused": bool(phase_b_states),
        },
        "slip_candidate_freeze": freeze,
        "frozen_support_verification": support_verification,
        "holdout": holdout,
        "sensor_recommendation": _sensor_recommendation(selected_identity),
        "terrain_regression": {
            "passed": terrain_before == terrain_after,
            "retrained": False,
            "before": terrain_before,
            "after": terrain_after,
        },
        "support_regression": {
            "passed": support_before == support_after,
            "retrained": False,
            "before": support_before,
            "after": support_after,
        },
        "fusion_regression": fusion_regression(),
        "runtime_boundary": {
            "slip_detector_continuous": True,
            "terrain_gate_required_for_slip": False,
            "unknown_terrain_blocks_slip": False,
            "fall_or_recovery_in_label": False,
            "q_dq_torque_added": False,
            "support_candidate_changed": False,
            "terrain_candidate_changed": False,
            "final_sensor_architecture_frozen": False,
        },
        "verdict": verdict,
    }
    _write_json(artifact_path / "metrics.json", metrics)
    progress(
        json.dumps(
            {
                "verdict": verdict,
                "selection": final_selection,
                "holdout": holdout["performed"],
            },
            indent=2,
        )
    )
    return artifact_path, metrics
