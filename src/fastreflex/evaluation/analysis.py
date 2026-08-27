"""Raw pelvis IMU sanity summaries and bounded event-aligned plots."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np

from fastreflex.dataset.loader import (
    CLASS_NAMES,
    ManifestRecord,
    VALID_OUTCOMES,
)


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


CHANNELS = ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z")
UNITS = ("m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s")


def _summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _load_for_sanity(record: ManifestRecord) -> dict[str, np.ndarray]:
    if record.observed_outcome not in VALID_OUTCOMES:
        raise ValueError(f"refusing excluded run in sanity analysis: {record.run_id}")
    with np.load(record.path, allow_pickle=False) as stored:
        return {
            "pelvis_imu": np.asarray(stored["pelvis_imu"], dtype=np.float32),
            "hazard_class_id": np.asarray(stored["hazard_class_id"], dtype=np.int8),
            "training_eligible": np.asarray(stored["training_eligible"], dtype=bool),
            "first_any_slip_onset_sample": np.asarray(
                stored["first_any_slip_onset_sample"], dtype=np.int64
            ),
            "first_sink_physical_onset_sample_per_foot": np.asarray(
                stored["first_sink_physical_onset_sample_per_foot"], dtype=np.int64
            ),
            "first_sink_degradation_onset_sample": np.asarray(
                stored["first_sink_degradation_onset_sample"], dtype=np.int64
            ),
        }


def _subsample(values: np.ndarray, limit: int = 20_000) -> np.ndarray:
    if len(values) <= limit:
        return values
    return values[np.linspace(0, len(values) - 1, limit, dtype=np.int64)]


def _plot_class_distributions(
    class_values: Mapping[int, np.ndarray], path: Path
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for channel, axis in enumerate(axes.flat):
        values = [
            _subsample(class_values[class_id][:, channel])
            for class_id in range(3)
        ]
        axis.boxplot(values, labels=CLASS_NAMES, showfliers=False)
        axis.set_title(CHANNELS[channel])
        axis.set_ylabel(UNITS[channel])
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Eligible pelvis IMU6 samples by established-state class")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _aligned_trace(
    imu: np.ndarray, anchor: int, radius: int = 500
) -> np.ndarray | None:
    if anchor < radius or anchor + radius >= len(imu):
        return None
    return imu[anchor - radius : anchor + radius + 1]


def _plot_aligned(
    traces: Mapping[str, Sequence[np.ndarray]],
    title: str,
    path: Path,
) -> None:
    time_ms = np.arange(-500, 501)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for channel, axis in enumerate(axes.flat):
        for color, (label, collection) in zip(colors, traces.items()):
            stacked = np.stack(collection)[:, :, channel]
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            axis.plot(time_ms, mean, color=color, label=label)
            axis.fill_between(
                time_ms, mean - std, mean + std, color=color, alpha=0.14
            )
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(CHANNELS[channel])
        axis.set_ylabel(UNITS[channel])
        axis.grid(alpha=0.25)
        if channel >= 3:
            axis.set_xlabel("aligned time (ms)")
    axes.flat[0].legend(loc="best")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_raw_sanity(
    records: Mapping[str, ManifestRecord],
    valid_run_ids: Sequence[str],
    plots_path: Path,
) -> dict[str, object]:
    """Inspect only valid-run raw signals and generate the three required plots."""
    plots_path.mkdir(parents=True, exist_ok=True)
    by_class: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    per_run: dict[str, dict[str, object]] = {}
    slip_traces: list[np.ndarray] = []
    slip_aligned_runs: list[str] = []
    sink_t1_traces: list[np.ndarray] = []
    sink_t2_traces: list[np.ndarray] = []
    sink_t1_runs: list[str] = []
    sink_t2_runs: list[str] = []
    normal_reference_trace: np.ndarray | None = None
    normal_reference_run_id: str | None = None
    total_nonfinite = 0
    for run_id in valid_run_ids:
        record = records[run_id]
        arrays = _load_for_sanity(record)
        imu = arrays["pelvis_imu"]
        labels = arrays["hazard_class_id"]
        eligible = arrays["training_eligible"]
        used = eligible & (labels >= 0) & (labels <= 2)
        total_nonfinite += int((~np.isfinite(imu[used])).sum())
        selected = imu[used]
        run_ranges = np.ptp(selected, axis=0)
        per_run[run_id] = {
            "observed_outcome": record.observed_outcome,
            "eligible_sample_count": int(used.sum()),
            "class_sample_counts": {
                CLASS_NAMES[class_id]: int(np.sum(eligible & (labels == class_id)))
                for class_id in range(3)
            },
            "channel_min": selected.min(axis=0).tolist(),
            "channel_max": selected.max(axis=0).tolist(),
            "channel_range": run_ranges.tolist(),
            "nonfinite_eligible_values": int((~np.isfinite(selected)).sum()),
        }
        for class_id in range(3):
            mask = eligible & (labels == class_id)
            if mask.any():
                by_class[class_id].append(imu[mask])
        if record.observed_outcome == "BENIGN" and normal_reference_trace is None:
            normal_reference_trace = _aligned_trace(imu, len(imu) // 2)
            normal_reference_run_id = run_id
        if record.observed_outcome == "SLIP":
            anchor = int(arrays["first_any_slip_onset_sample"])
            trace = _aligned_trace(imu, anchor)
            if trace is not None:
                slip_traces.append(trace)
                slip_aligned_runs.append(run_id)
        if record.observed_outcome == "SINK":
            physical = arrays["first_sink_physical_onset_sample_per_foot"]
            physical = physical[physical >= 0]
            if len(physical):
                trace = _aligned_trace(imu, int(physical.min()))
                if trace is not None:
                    sink_t1_traces.append(trace)
                    sink_t1_runs.append(run_id)
            anchor = int(arrays["first_sink_degradation_onset_sample"])
            trace = _aligned_trace(imu, anchor)
            if trace is not None:
                sink_t2_traces.append(trace)
                sink_t2_runs.append(run_id)
    if total_nonfinite:
        raise ValueError("raw sanity found non-finite eligible IMU values")
    if normal_reference_trace is None or normal_reference_run_id is None:
        raise ValueError("raw sanity could not select a benign reference trace")
    class_values = {
        class_id: np.concatenate(parts, axis=0)
        for class_id, parts in by_class.items()
    }
    if any(values.size == 0 for values in class_values.values()):
        raise ValueError("raw sanity found an empty established-state class")
    statistics = {
        CLASS_NAMES[class_id]: {
            CHANNELS[channel]: {
                "unit": UNITS[channel],
                **_summary(values[:, channel]),
            }
            for channel in range(6)
        }
        for class_id, values in class_values.items()
    }
    largest_run_share_by_class: dict[str, dict[str, float | int | str]] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        run_id, count = max(
            (
                (run_id, int(per_run[run_id]["class_sample_counts"][class_name]))
                for run_id in valid_run_ids
            ),
            key=lambda item: item[1],
        )
        total = len(class_values[class_id])
        largest_run_share_by_class[class_name] = {
            "run_id": run_id,
            "sample_count": count,
            "class_sample_fraction": float(count / total),
        }
    thresholds_by_outcome: dict[str, list[float]] = {}
    outliers: dict[str, list[str]] = {}
    for outcome in VALID_OUTCOMES:
        outcome_ids = [
            run_id
            for run_id in valid_run_ids
            if records[run_id].observed_outcome == outcome
        ]
        ranges = np.asarray(
            [per_run[run_id]["channel_range"] for run_id in outcome_ids],
            dtype=np.float64,
        )
        median_range = np.median(ranges, axis=0)
        mad_range = np.median(np.abs(ranges - median_range), axis=0)
        thresholds = median_range + 6.0 * np.maximum(mad_range, 1.0e-12)
        thresholds_by_outcome[outcome] = thresholds.tolist()
        for run_index, run_id in enumerate(outcome_ids):
            channels = [
                CHANNELS[index]
                for index in range(6)
                if ranges[run_index, index] > thresholds[index]
            ]
            if channels:
                outliers[run_id] = channels
    clipping_suspicions: list[dict[str, object]] = []
    for class_id, values in class_values.items():
        for channel in range(6):
            series = values[:, channel]
            extreme_fraction = max(
                float(np.mean(series == series.min())),
                float(np.mean(series == series.max())),
            )
            if extreme_fraction > 0.005:
                clipping_suspicions.append(
                    {
                        "class": CLASS_NAMES[class_id],
                        "channel": CHANNELS[channel],
                        "repeated_extreme_fraction": extreme_fraction,
                    }
                )
    _plot_class_distributions(
        class_values, plots_path / "class_channel_distribution.png"
    )
    _plot_aligned(
        {"NORMAL ref": [normal_reference_trace], "SLIP t1": slip_traces},
        "Slip runs aligned to established-slip onset t1 (mean ± std)",
        plots_path / "slip_event_aligned.png",
    )
    _plot_aligned(
        {
            "NORMAL ref": [normal_reference_trace],
            "physical t1": sink_t1_traces,
            "degradation t2": sink_t2_traces,
        },
        "Sink runs aligned separately to physical t1 and degradation t2",
        plots_path / "sink_event_aligned.png",
    )
    return {
        "scope": "all 33 valid runs; eligible class samples for distributions",
        "channel_order": list(CHANNELS),
        "channel_units": list(UNITS),
        "class_sample_counts": {
            CLASS_NAMES[class_id]: int(len(values))
            for class_id, values in class_values.items()
        },
        "per_class_channel_statistics": statistics,
        "largest_run_share_by_class": largest_run_share_by_class,
        "per_run": per_run,
        "range_outlier_method": "within observed outcome; median + 6 * MAD",
        "range_outlier_threshold_by_outcome": thresholds_by_outcome,
        "range_outlier_runs": outliers,
        "nonfinite_eligible_values": total_nonfinite,
        "clipping_suspicions": clipping_suspicions,
        "event_alignment": {
            "normal_reference_run_id": normal_reference_run_id,
            "slip_t1_run_ids": slip_aligned_runs,
            "sink_t1_run_ids": sink_t1_runs,
            "sink_t2_run_ids": sink_t2_runs,
            "radius_ms": 500,
        },
    }
