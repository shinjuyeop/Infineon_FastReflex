"""Read-only run-level analysis of causal virtual-FSR redistribution dynamics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib
import numpy as np
import yaml

from fastreflex.dataset.loader import sha256_file
from fastreflex.evaluation.fsr_distribution import (
    ALL_METRICS as STATIC_METRICS,
    DistributionTrace,
    SinkRun,
    _event_samples,
    _git_commit,
    _load_arrays,
    _load_sink_runs,
    _range_overlap,
    _ranking_rows as static_ranking_rows,
    _resolve_path,
    _separation_rows as static_separation_rows,
    _spearman,
    _write_csv,
    _write_json,
    causal_trailing_median,
    derive_distribution_trace,
    pre_event_baseline_median,
    run_level_auc,
)
from fastreflex.simulation.hazards import LOAD_OFF_N


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


EPSILON = 1.0e-12
QUADRANT_METRICS = (
    "front_left_share",
    "front_right_share",
    "rear_left_share",
    "rear_right_share",
)
TEMPORAL_METRICS = (
    "quadrant_l1_change",
    "quadrant_path_length",
    "quadrant_path_rate",
    "excess_quadrant_path_rate",
    "concentration_delta",
    "concentration_abs_change",
    "concentration_path",
    "excess_concentration_path_rate",
    "entropy_delta",
    "entropy_abs_change",
    "entropy_path",
    "excess_entropy_path_rate",
    "max_share_delta",
    "max_share_abs_change",
    "cop_displacement",
    "cop_path_length",
    "cop_path_rate",
    "cop_path_efficiency",
    "excess_cop_path_rate",
    "medial_delta",
    "medial_abs_change",
    "medial_path",
    "front_delta",
    "front_abs_change",
    "front_path",
    "bilateral_share_delta",
    "bilateral_share_abs_change",
    "bilateral_share_path",
    "bilateral_shift_delta",
    "bilateral_shift_abs_change",
    "bilateral_shift_path",
)
DIRECTION_INDEPENDENT_METRICS = {
    "quadrant_l1_change",
    "quadrant_path_length",
    "quadrant_path_rate",
    "excess_quadrant_path_rate",
    "concentration_abs_change",
    "concentration_path",
    "excess_concentration_path_rate",
    "entropy_abs_change",
    "entropy_path",
    "excess_entropy_path_rate",
    "max_share_abs_change",
    "cop_displacement",
    "cop_path_length",
    "cop_path_rate",
    "cop_path_efficiency",
    "excess_cop_path_rate",
    "medial_abs_change",
    "medial_path",
    "front_abs_change",
    "front_path",
    "bilateral_share_abs_change",
    "bilateral_share_path",
    "bilateral_shift_abs_change",
    "bilateral_shift_path",
}
PATH_FAMILY = {
    "quadrant_path_length": "quadrant",
    "quadrant_path_rate": "quadrant",
    "excess_quadrant_path_rate": "quadrant",
    "concentration_path": "concentration",
    "excess_concentration_path_rate": "concentration",
    "entropy_path": "entropy",
    "excess_entropy_path_rate": "entropy",
    "cop_path_length": "cop",
    "cop_path_rate": "cop",
    "cop_path_efficiency": "cop",
    "excess_cop_path_rate": "cop",
    "medial_path": "medial",
    "front_path": "front",
    "bilateral_share_path": "bilateral_share",
    "bilateral_shift_path": "bilateral_shift",
}


@dataclass(frozen=True)
class TemporalTrace:
    distribution: DistributionTrace
    shares: np.ndarray
    concentration: np.ndarray
    entropy: np.ndarray
    max_share: np.ndarray
    cop: np.ndarray
    medial: np.ndarray
    front: np.ndarray
    bilateral_share: np.ndarray
    bilateral_shift: np.ndarray


def normalized_entropy(shares: np.ndarray) -> np.ndarray:
    """Return normalized four-bin entropy while preserving invalid rows as NaN."""
    values = np.asarray(shares, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("shares must have shape [N,4]")
    output = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    if valid.any():
        selected = values[valid]
        terms = np.zeros_like(selected)
        positive = selected > 0.0
        terms[positive] = selected[positive] * np.log(selected[positive] + EPSILON)
        output[valid] = -np.sum(terms, axis=1) / np.log(4.0)
    return output


def derive_temporal_trace(
    foot_fsr: np.ndarray,
    affected_side: str,
    low_load_threshold_n: float = LOAD_OFF_N,
) -> TemporalTrace:
    """Derive declared temporal inputs from affected-foot canonicalized FSR8."""
    distribution = derive_distribution_trace(
        foot_fsr, affected_side, low_load_threshold_n
    )
    shares = np.column_stack(
        [distribution.metrics[name] for name in QUADRANT_METRICS]
    )
    both_loaded = distribution.affected_loaded & distribution.unaffected_loaded
    bilateral_share = np.where(
        both_loaded, distribution.metrics["affected_load_share"], np.nan
    )
    bilateral_shift = np.where(
        both_loaded, distribution.metrics["signed_bilateral_shift"], np.nan
    )
    return TemporalTrace(
        distribution=distribution,
        shares=shares,
        concentration=distribution.metrics["load_concentration"],
        entropy=normalized_entropy(shares),
        max_share=distribution.metrics["max_quadrant_share"],
        cop=np.column_stack(
            (
                distribution.metrics["cop_x_proxy"],
                distribution.metrics["cop_y_proxy"],
            )
        ),
        medial=distribution.metrics["medial_ratio"],
        front=distribution.metrics["front_ratio"],
        bilateral_share=bilateral_share,
        bilateral_shift=bilateral_shift,
    )


def continuous_path_length(
    values: np.ndarray,
    start_sample: int,
    stop_sample: int,
    norm: str = "absolute",
) -> tuple[float | None, int, int]:
    """Accumulate adjacent valid 1 ms steps without bridging invalid gaps.

    The interval is inclusive. A single valid sample has zero path only when
    the requested interval itself contains a single sample.
    """
    series = np.asarray(values, dtype=np.float64)
    if series.ndim not in {1, 2}:
        raise ValueError("path values must be one- or two-dimensional")
    if start_sample < 0 or stop_sample < start_sample or stop_sample >= len(series):
        return None, 0, 0
    segment = series[start_sample : stop_sample + 1]
    valid = np.isfinite(segment) if segment.ndim == 1 else np.isfinite(segment).all(axis=1)
    valid_count = int(np.count_nonzero(valid))
    if len(segment) == 1:
        return (0.0, 0, valid_count) if valid_count == 1 else (None, 0, 0)
    pair_valid = valid[:-1] & valid[1:]
    pair_count = int(np.count_nonzero(pair_valid))
    if pair_count == 0:
        return None, 0, valid_count
    differences = np.diff(segment, axis=0)[pair_valid]
    if differences.ndim == 1 or norm == "absolute":
        distances = np.abs(differences) if differences.ndim == 1 else np.sum(np.abs(differences), axis=1)
    elif norm == "l1":
        distances = np.sum(np.abs(differences), axis=1)
    elif norm == "euclidean":
        distances = np.sqrt(np.sum(np.square(differences), axis=1))
    else:
        raise ValueError("unsupported path norm")
    return float(np.sum(distances)), pair_count, valid_count


def _window_median(
    values: np.ndarray, start_sample: int, stop_sample_exclusive: int
) -> tuple[np.ndarray | float | None, int]:
    series = np.asarray(values, dtype=np.float64)
    if start_sample < 0 or stop_sample_exclusive > len(series) or start_sample >= stop_sample_exclusive:
        return None, 0
    selected = series[start_sample:stop_sample_exclusive]
    if selected.ndim == 1:
        valid = np.isfinite(selected)
        if not valid.any():
            return None, 0
        return float(np.median(selected[valid])), int(np.count_nonzero(valid))
    valid = np.isfinite(selected).all(axis=1)
    if not valid.any():
        return None, 0
    median = np.median(selected[valid], axis=0)
    if selected.shape[1] == 4:
        total = float(np.sum(median))
        if total <= EPSILON:
            return None, int(np.count_nonzero(valid))
        median = median / total
    return median, int(np.count_nonzero(valid))


def pre_event_path_rate(
    values: np.ndarray,
    event_sample: int,
    start_ms: int,
    stop_ms: int,
    minimum_valid_samples: int,
    norm: str,
) -> tuple[float | None, int, tuple[int, int]]:
    """Compute path rate only inside the declared pre-event interval."""
    start = event_sample + start_ms
    stop = event_sample + stop_ms
    if stop <= start:
        raise ValueError("baseline path interval must have positive duration")
    path, _, valid_count = continuous_path_length(values, start, stop, norm)
    if path is None or valid_count < minimum_valid_samples:
        return None, valid_count, (start, stop + 1)
    return path / float(stop - start), valid_count, (start, stop + 1)


def temporal_horizon_metrics(
    trace: TemporalTrace,
    event_sample: int,
    horizon_ms: int,
    initial_width_ms: int,
    endpoint_width_ms: int,
    baseline_start_ms: int,
    baseline_stop_ms: int,
    minimum_baseline_valid_samples: int,
) -> tuple[dict[str, float | None], dict[str, object]]:
    """Compute all declared metrics using samples no later than event + horizon."""
    if horizon_ms < 0 or initial_width_ms <= 0 or endpoint_width_ms <= 0:
        raise ValueError("invalid temporal horizon request")
    endpoint = event_sample + horizon_ms
    if event_sample < 0 or endpoint >= len(trace.shares):
        raise ValueError("temporal interval is outside the trace")
    initial_stop = min(event_sample + initial_width_ms, endpoint + 1)
    endpoint_start = max(event_sample, endpoint - endpoint_width_ms + 1)
    endpoint_stop = endpoint + 1

    p0, initial_count = _window_median(trace.shares, event_sample, initial_stop)
    p1, endpoint_count = _window_median(trace.shares, endpoint_start, endpoint_stop)

    def endpoints(values: np.ndarray) -> tuple[float | np.ndarray | None, float | np.ndarray | None]:
        start_value, _ = _window_median(values, event_sample, initial_stop)
        end_value, _ = _window_median(values, endpoint_start, endpoint_stop)
        return start_value, end_value

    def scalar_change(values: np.ndarray) -> tuple[float | None, float | None]:
        start_value, end_value = endpoints(values)
        if start_value is None or end_value is None:
            return None, None
        delta = float(end_value) - float(start_value)
        return delta, abs(delta)

    quadrant_path, quadrant_pairs, interval_valid = continuous_path_length(
        trace.shares, event_sample, endpoint, "l1"
    )
    concentration_path, concentration_pairs, _ = continuous_path_length(
        trace.concentration, event_sample, endpoint, "absolute"
    )
    entropy_path, entropy_pairs, _ = continuous_path_length(
        trace.entropy, event_sample, endpoint, "absolute"
    )
    cop_path, cop_pairs, _ = continuous_path_length(
        trace.cop, event_sample, endpoint, "euclidean"
    )
    medial_path, medial_pairs, _ = continuous_path_length(
        trace.medial, event_sample, endpoint, "absolute"
    )
    front_path, front_pairs, _ = continuous_path_length(
        trace.front, event_sample, endpoint, "absolute"
    )
    bilateral_share_path, bilateral_share_pairs, _ = continuous_path_length(
        trace.bilateral_share, event_sample, endpoint, "absolute"
    )
    bilateral_shift_path, bilateral_shift_pairs, _ = continuous_path_length(
        trace.bilateral_shift, event_sample, endpoint, "absolute"
    )

    concentration_delta, concentration_abs = scalar_change(trace.concentration)
    entropy_delta, entropy_abs = scalar_change(trace.entropy)
    max_delta, max_abs = scalar_change(trace.max_share)
    medial_delta, medial_abs = scalar_change(trace.medial)
    front_delta, front_abs = scalar_change(trace.front)
    bilateral_share_delta, bilateral_share_abs = scalar_change(trace.bilateral_share)
    bilateral_shift_delta, bilateral_shift_abs = scalar_change(trace.bilateral_shift)

    quadrant_l1 = None
    if p0 is not None and p1 is not None:
        quadrant_l1 = float(np.sum(np.abs(np.asarray(p1) - np.asarray(p0))))
    cop0, cop1 = endpoints(trace.cop)
    cop_displacement = None
    if cop0 is not None and cop1 is not None:
        cop_displacement = float(
            np.sqrt(np.sum(np.square(np.asarray(cop1) - np.asarray(cop0))))
        )

    def rate(path: float | None) -> float | None:
        return None if path is None or horizon_ms == 0 else path / float(horizon_ms)

    quadrant_rate = rate(quadrant_path)
    concentration_rate = rate(concentration_path)
    entropy_rate = rate(entropy_path)
    cop_rate = rate(cop_path)
    baseline_arguments = (
        event_sample,
        baseline_start_ms,
        baseline_stop_ms,
        minimum_baseline_valid_samples,
    )
    baseline_quadrant_rate, baseline_quadrant_count, baseline_range = pre_event_path_rate(
        trace.shares, *baseline_arguments, "l1"
    )
    baseline_concentration_rate, _, _ = pre_event_path_rate(
        trace.concentration, *baseline_arguments, "absolute"
    )
    baseline_entropy_rate, _, _ = pre_event_path_rate(
        trace.entropy, *baseline_arguments, "absolute"
    )
    baseline_cop_rate, _, _ = pre_event_path_rate(
        trace.cop, *baseline_arguments, "euclidean"
    )

    def excess(post: float | None, baseline: float | None) -> float | None:
        return None if post is None or baseline is None else post - baseline

    cop_efficiency = None
    if cop_path is not None and cop_displacement is not None:
        cop_efficiency = 0.0 if cop_path <= EPSILON and cop_displacement <= EPSILON else cop_displacement / (cop_path + EPSILON)
    metrics = {
        "quadrant_l1_change": quadrant_l1,
        "quadrant_path_length": quadrant_path,
        "quadrant_path_rate": quadrant_rate,
        "excess_quadrant_path_rate": excess(quadrant_rate, baseline_quadrant_rate),
        "concentration_delta": concentration_delta,
        "concentration_abs_change": concentration_abs,
        "concentration_path": concentration_path,
        "excess_concentration_path_rate": excess(concentration_rate, baseline_concentration_rate),
        "entropy_delta": entropy_delta,
        "entropy_abs_change": entropy_abs,
        "entropy_path": entropy_path,
        "excess_entropy_path_rate": excess(entropy_rate, baseline_entropy_rate),
        "max_share_delta": max_delta,
        "max_share_abs_change": max_abs,
        "cop_displacement": cop_displacement,
        "cop_path_length": cop_path,
        "cop_path_rate": cop_rate,
        "cop_path_efficiency": cop_efficiency,
        "excess_cop_path_rate": excess(cop_rate, baseline_cop_rate),
        "medial_delta": medial_delta,
        "medial_abs_change": medial_abs,
        "medial_path": medial_path,
        "front_delta": front_delta,
        "front_abs_change": front_abs,
        "front_path": front_path,
        "bilateral_share_delta": bilateral_share_delta,
        "bilateral_share_abs_change": bilateral_share_abs,
        "bilateral_share_path": bilateral_share_path,
        "bilateral_shift_delta": bilateral_shift_delta,
        "bilateral_shift_abs_change": bilateral_shift_abs,
        "bilateral_shift_path": bilateral_shift_path,
    }
    diagnostics = {
        "initial_start_sample": event_sample,
        "initial_stop_sample_exclusive": initial_stop,
        "initial_valid_distribution_samples": initial_count,
        "endpoint_start_sample": endpoint_start,
        "endpoint_stop_sample_exclusive": endpoint_stop,
        "endpoint_valid_distribution_samples": endpoint_count,
        "interval_valid_distribution_samples": interval_valid,
        "baseline_start_sample": baseline_range[0],
        "baseline_stop_sample_exclusive": baseline_range[1],
        "baseline_valid_distribution_samples": baseline_quadrant_count,
        "path_pairs": {
            "quadrant": quadrant_pairs,
            "concentration": concentration_pairs,
            "entropy": entropy_pairs,
            "cop": cop_pairs,
            "medial": medial_pairs,
            "front": front_pairs,
            "bilateral_share": bilateral_share_pairs,
            "bilateral_shift": bilateral_shift_pairs,
        },
    }
    return metrics, diagnostics


def _temporal_separation_rows(
    per_run_rows: Sequence[Mapping[str, object]], runs: Sequence[SinkRun]
) -> list[dict[str, object]]:
    run_lookup = {run.run_id: run for run in runs}
    rows = []
    keys = sorted(
        {
            (str(row["alignment"]), int(row["horizon_ms"]), str(row["metric"]))
            for row in per_run_rows
        }
    )
    for alignment, horizon, metric in keys:
        selected = [
            row
            for row in per_run_rows
            if row["alignment"] == alignment
            and int(row["horizon_ms"]) == horizon
            and row["metric"] == metric
            and row["value"] != ""
        ]
        benign_pairs = [
            (str(row["run_id"]), float(row["value"]))
            for row in selected
            if row["group"] == "BENIGN_SINK"
        ]
        hazard_pairs = [
            (str(row["run_id"]), float(row["value"]))
            for row in selected
            if row["group"] == "HAZARDOUS_SINK"
        ]
        if not benign_pairs or not hazard_pairs:
            continue
        benign = np.asarray([value for _, value in benign_pairs], dtype=np.float64)
        hazard = np.asarray([value for _, value in hazard_pairs], dtype=np.float64)
        raw_auc, oriented_auc, direction = run_level_auc(benign, hazard)
        side_values: dict[str, dict[str, object]] = {}
        for side in ("left", "right"):
            side_benign = np.asarray(
                [value for run_id, value in benign_pairs if run_lookup[run_id].affected_side == side]
            )
            side_hazard = np.asarray(
                [value for run_id, value in hazard_pairs if run_lookup[run_id].affected_side == side]
            )
            if not len(side_benign) or not len(side_hazard):
                side_values[side] = {"auc": None, "consistent": False, "hazard_median": None}
                continue
            side_raw, side_oriented, _ = run_level_auc(side_benign, side_hazard)
            primary_oriented = side_raw if direction == "higher_is_hazardous" else 1.0 - side_raw
            side_values[side] = {
                "auc": side_oriented,
                "primary_direction_auc": primary_oriented,
                "consistent": primary_oriented >= 0.5,
                "hazard_median": float(np.median(side_hazard)),
            }
        rows.append(
            {
                "alignment": alignment,
                "horizon_ms": horizon,
                "metric": metric,
                "direction_independent": metric in DIRECTION_INDEPENDENT_METRICS,
                "benign_run_count": len(benign),
                "hazardous_run_count": len(hazard),
                "benign_median": float(np.median(benign)),
                "benign_min": float(np.min(benign)),
                "benign_max": float(np.max(benign)),
                "hazardous_median": float(np.median(hazard)),
                "hazardous_min": float(np.min(hazard)),
                "hazardous_max": float(np.max(hazard)),
                "range_overlap": _range_overlap(benign, hazard),
                "raw_auc": raw_auc,
                "oriented_auc": oriented_auc,
                "direction": direction,
                "left_oriented_auc": side_values["left"].get("auc"),
                "right_oriented_auc": side_values["right"].get("auc"),
                "left_primary_direction_auc": side_values["left"].get("primary_direction_auc"),
                "right_primary_direction_auc": side_values["right"].get("primary_direction_auc"),
                "left_hazardous_median": side_values["left"].get("hazard_median"),
                "right_hazardous_median": side_values["right"].get("hazard_median"),
                "left_right_direction_consistent": bool(
                    side_values["left"]["consistent"] and side_values["right"]["consistent"]
                ),
            }
        )
    return rows


def _temporal_ranking_rows(
    separation_rows: Sequence[Mapping[str, object]], top_count: int = 5
) -> list[dict[str, object]]:
    rankings = []
    keys = sorted(
        {(str(row["alignment"]), int(row["horizon_ms"])) for row in separation_rows}
    )
    for alignment, horizon in keys:
        candidates = [
            row
            for row in separation_rows
            if row["alignment"] == alignment and int(row["horizon_ms"]) == horizon
            and int(row["benign_run_count"]) >= 4
            and int(row["hazardous_run_count"]) >= 7
        ]
        candidates.sort(
            key=lambda row: (
                -float(row["oriented_auc"]),
                bool(row["range_overlap"]),
                not bool(row["left_right_direction_consistent"]),
                -min(int(row["benign_run_count"]), int(row["hazardous_run_count"])),
                not bool(row["direction_independent"]),
                str(row["metric"]),
            )
        )
        for rank, row in enumerate(candidates[:top_count], start=1):
            rankings.append({"rank": rank, **row})
    return rankings


def _static_analysis_rows(
    runs: Sequence[SinkRun],
    traces: Mapping[str, TemporalTrace],
    events: Mapping[str, Mapping[str, int | None]],
    horizons: Sequence[int],
    trailing_width: int,
    baseline_start: int,
    baseline_stop: int,
    minimum_baseline: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    per_run = []
    for run in runs:
        trace = traces[run.run_id].distribution
        event = events[run.run_id]
        for alignment in ("t0", "t1"):
            anchor = int(event[alignment])
            for metric in STATIC_METRICS:
                baseline_value, _, _ = pre_event_baseline_median(
                    trace.metrics[metric],
                    anchor,
                    baseline_start,
                    baseline_stop,
                    minimum_baseline,
                )
                for horizon in horizons:
                    value, _, _ = causal_trailing_median(
                        trace.metrics[metric], anchor, horizon, trailing_width
                    )
                    per_run.append(
                        {
                            "run_id": run.run_id,
                            "group": "BENIGN_SINK" if run.outcome == "BENIGN" else "HAZARDOUS_SINK",
                            "alignment": alignment,
                            "horizon_ms": horizon,
                            "metric": metric,
                            "raw_value": "" if value is None else value,
                            "baseline_delta": ""
                            if value is None or baseline_value is None
                            else value - baseline_value,
                        }
                    )
    separation = static_separation_rows(per_run, runs, ("t0", "t1"), horizons)
    return per_run, separation, static_ranking_rows(separation)


def _verify_previous_static_parity(
    previous_path: Path,
    manifest_sha256: str,
    computed_rankings: Sequence[Mapping[str, object]],
    parity_horizons: Sequence[int],
) -> dict[str, object]:
    summary_path = previous_path / "summary.json"
    ranking_path = previous_path / "feature_ranking.csv"
    if not summary_path.is_file() or not ranking_path.is_file():
        raise FileNotFoundError("previous static analysis artifact is unavailable")
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if summary.get("experiment_id") != "FSR_LOAD_DISTRIBUTION_ANALYSIS":
        raise ValueError("previous static analysis experiment ID mismatch")
    if summary.get("manifest_sha256") != manifest_sha256:
        raise ValueError("previous static analysis manifest mismatch")
    with ranking_path.open("r", encoding="utf-8", newline="") as stream:
        previous = list(csv.DictReader(stream))
    for alignment in ("t0", "t1"):
        for horizon in parity_horizons:
            expected = next(
                row
                for row in previous
                if row["alignment"] == alignment
                and int(row["horizon_ms"]) == horizon
                and int(row["rank"]) == 1
            )
            actual = next(
                row
                for row in computed_rankings
                if row["alignment"] == alignment
                and int(row["horizon_ms"]) == horizon
                and int(row["rank"]) == 1
            )
            if (
                expected["metric"] != actual["metric"]
                or expected["representation"] != actual["representation"]
                or not np.isclose(
                    float(expected["oriented_auc"]),
                    float(actual["oriented_auc"]),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise ValueError(
                    f"static parity mismatch at {alignment}+{horizon} ms"
                )
    return {
        "artifact_path": str(previous_path),
        "source_commit": summary.get("source_commit"),
        "parity_horizons_ms": list(parity_horizons),
        "status": "exact_top_metric_representation_auc_match",
    }


def _static_vs_temporal_rows(
    static_rankings: Sequence[Mapping[str, object]],
    temporal_rankings: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    keys = sorted(
        {
            (str(row["alignment"]), int(row["horizon_ms"]))
            for row in temporal_rankings
            if int(row["rank"]) == 1
        }
    )
    for alignment, horizon in keys:
        static = next(
            row
            for row in static_rankings
            if row["alignment"] == alignment
            and int(row["horizon_ms"]) == horizon
            and int(row["rank"]) == 1
        )
        temporal = next(
            row
            for row in temporal_rankings
            if row["alignment"] == alignment
            and int(row["horizon_ms"]) == horizon
            and int(row["rank"]) == 1
        )
        output.append(
            {
                "alignment": alignment,
                "horizon_ms": horizon,
                "static_metric": static["metric"],
                "static_representation": static["representation"],
                "static_oriented_auc": static["oriented_auc"],
                "static_range_overlap": static["range_overlap"],
                "static_benign_min": static["benign_min"],
                "static_benign_max": static["benign_max"],
                "static_hazardous_min": static["hazardous_min"],
                "static_hazardous_max": static["hazardous_max"],
                "temporal_metric": temporal["metric"],
                "temporal_oriented_auc": temporal["oriented_auc"],
                "temporal_range_overlap": temporal["range_overlap"],
                "temporal_benign_min": temporal["benign_min"],
                "temporal_benign_max": temporal["benign_max"],
                "temporal_hazardous_min": temporal["hazardous_min"],
                "temporal_hazardous_max": temporal["hazardous_max"],
                "temporal_left_oriented_auc": temporal["left_oriented_auc"],
                "temporal_right_oriented_auc": temporal["right_oriented_auc"],
                "temporal_minus_static_auc": float(temporal["oriented_auc"])
                - float(static["oriented_auc"]),
            }
        )
    return output


def _load_uniform_sand_controls(
    manifest_rows: Sequence[Mapping[str, str]],
    dataset_path: Path,
    low_load_threshold_n: float,
) -> list[dict[str, object]]:
    controls = []
    for row in manifest_rows:
        if not row["run_id"].startswith("normal_sand_"):
            continue
        arrays = _load_arrays(dataset_path / row["file"])
        traces = {
            side: derive_temporal_trace(
                arrays["foot_fsr"], side, low_load_threshold_n
            )
            for side in ("left", "right")
        }
        t1 = {
            side: int(arrays["first_sink_physical_onset_sample_per_foot"][index])
            for index, side in enumerate(("left", "right"))
        }
        if min(t1.values()) < 0:
            raise ValueError("Uniform Sand is missing per-foot physical Sink onset")
        controls.append(
            {
                "run_id": row["run_id"],
                "speed_mps": float(row["speed_mps"]),
                "traces": traces,
                "t1": t1,
            }
        )
    if len(controls) != 4:
        raise ValueError("expected four Uniform Sand controls")
    return controls


def _uniform_sand_rows(
    controls: Sequence[Mapping[str, object]],
    per_run_rows: Sequence[Mapping[str, object]],
    separation_rows: Sequence[Mapping[str, object]],
    horizons: Sequence[int],
    initial_width: int,
    endpoint_width: int,
    baseline_start: int,
    baseline_stop: int,
    minimum_baseline: int,
) -> list[dict[str, object]]:
    output = []
    for horizon in horizons:
        for metric in TEMPORAL_METRICS:
            primary = next(
                (
                    row
                    for row in separation_rows
                    if row["alignment"] == "t1"
                    and int(row["horizon_ms"]) == horizon
                    and row["metric"] == metric
                ),
                None,
            )
            if primary is None:
                continue
            sand_values = []
            selected_feet = []
            for control in controls:
                traces = control["traces"]
                events = control["t1"]
                assert isinstance(traces, Mapping) and isinstance(events, Mapping)
                candidates = []
                for side in ("left", "right"):
                    values, _ = temporal_horizon_metrics(
                        traces[side],
                        int(events[side]),
                        horizon,
                        initial_width,
                        endpoint_width,
                        baseline_start,
                        baseline_stop,
                        minimum_baseline,
                    )
                    if values[metric] is not None:
                        candidates.append((side, float(values[metric])))
                if not candidates:
                    continue
                chosen = (
                    max(candidates, key=lambda item: item[1])
                    if primary["direction"] == "higher_is_hazardous"
                    else min(candidates, key=lambda item: item[1])
                )
                sand_values.append(chosen[1])
                selected_feet.append(f"{control['run_id']}:{chosen[0]}")
            hazardous = [
                float(row["value"])
                for row in per_run_rows
                if row["alignment"] == "t1"
                and int(row["horizon_ms"]) == horizon
                and row["metric"] == metric
                and row["group"] == "HAZARDOUS_SINK"
                and row["value"] != ""
            ]
            if not sand_values or not hazardous:
                continue
            sand = np.asarray(sand_values, dtype=np.float64)
            hazard = np.asarray(hazardous, dtype=np.float64)
            raw_auc, oriented_auc, direction = run_level_auc(sand, hazard)
            primary_direction_auc = (
                raw_auc
                if primary["direction"] == "higher_is_hazardous"
                else 1.0 - raw_auc
            )
            output.append(
                {
                    "horizon_ms": horizon,
                    "metric": metric,
                    "selection": "per_run_more_hazardous_worst_foot_at_uniform_physical_sink_onset",
                    "primary_direction": primary["direction"],
                    "sand_run_count": len(sand),
                    "hazardous_run_count": len(hazard),
                    "sand_median": float(np.median(sand)),
                    "sand_min": float(np.min(sand)),
                    "sand_max": float(np.max(sand)),
                    "hazardous_median": float(np.median(hazard)),
                    "hazardous_min": float(np.min(hazard)),
                    "hazardous_max": float(np.max(hazard)),
                    "range_overlap": _range_overlap(sand, hazard),
                    "raw_auc": raw_auc,
                    "oriented_auc": oriented_auc,
                    "direction": direction,
                    "primary_direction_auc": primary_direction_auc,
                    "selected_pseudo_feet": "|".join(selected_feet),
                }
            )
    return output


def _metadata_audit_rows(
    per_run_rows: Sequence[Mapping[str, object]],
    rankings: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    for ranking in rankings:
        selected = [
            row
            for row in per_run_rows
            if row["alignment"] == ranking["alignment"]
            and int(row["horizon_ms"]) == int(ranking["horizon_ms"])
            and row["metric"] == ranking["metric"]
            and row["value"] != ""
        ]

        def median(rows: Sequence[Mapping[str, object]]) -> float | str:
            return "" if not rows else float(np.median([float(row["value"]) for row in rows]))

        severity = {
            name: median([row for row in selected if row["severity"] == name])
            for name in ("mild", "moderate", "severe")
        }
        severe = [row for row in selected if row["severity"] == "severe"]
        ordered = [severity[name] for name in ("mild", "moderate", "severe")]
        monotonic = False
        if all(value != "" for value in ordered):
            numeric = [float(value) for value in ordered]
            monotonic = (
                numeric[0] <= numeric[1] <= numeric[2]
                if ranking["direction"] == "higher_is_hazardous"
                else numeric[0] >= numeric[1] >= numeric[2]
            )

        def correlation(rows: Sequence[Mapping[str, object]], field: str) -> float | str:
            result = _spearman(
                [float(row["value"]) for row in rows],
                [float(row[field]) for row in rows],
            )
            return "" if result is None else result

        output.append(
            {
                "alignment": ranking["alignment"],
                "horizon_ms": ranking["horizon_ms"],
                "rank": ranking["rank"],
                "metric": ranking["metric"],
                "direction": ranking["direction"],
                "mild_median": severity["mild"],
                "moderate_median": severity["moderate"],
                "severe_median": severity["severe"],
                "severity_median_monotonic": monotonic,
                "left_severe_median": median(
                    [row for row in severe if row["affected_side"] == "left"]
                ),
                "right_severe_median": median(
                    [row for row in severe if row["affected_side"] == "right"]
                ),
                "severe_speed_spearman": correlation(severe, "speed_mps"),
                "all_run_contact_phase_spearman": correlation(
                    selected, "t0_contact_phase"
                ),
                "all_run_contact_age_spearman": correlation(
                    selected, "contact_age_endpoint_ms"
                ),
            }
        )
    return output


def _reference_terrain_rows(
    manifest_rows: Sequence[Mapping[str, str]],
    dataset_path: Path,
    low_load_threshold_n: float,
    window_ms: int,
) -> list[dict[str, object]]:
    output = []
    definitions = {
        "quadrant_path_rate": ("shares", "l1"),
        "concentration_path_rate": ("concentration", "absolute"),
        "entropy_path_rate": ("entropy", "absolute"),
        "cop_path_rate": ("cop", "euclidean"),
    }
    for row in manifest_rows:
        if not row["run_id"].startswith(
            ("normal_concrete_", "normal_marble_", "normal_sand_")
        ):
            continue
        arrays = _load_arrays(dataset_path / row["file"])
        traces = {
            side: derive_temporal_trace(
                arrays["foot_fsr"], side, low_load_threshold_n
            )
            for side in ("left", "right")
        }
        for metric, (attribute, norm) in definitions.items():
            worst_foot_rates = []
            for start in range(0, len(arrays["foot_fsr"]) - window_ms, window_ms):
                rates = []
                for side in ("left", "right"):
                    values = getattr(traces[side], attribute)
                    path, _, _ = continuous_path_length(
                        values, start, start + window_ms, norm
                    )
                    if path is not None:
                        rates.append(path / float(window_ms))
                if rates:
                    worst_foot_rates.append(max(rates))
            if not worst_foot_rates:
                continue
            output.append(
                {
                    "run_id": row["run_id"],
                    "terrain": row["terrain"],
                    "speed_mps": float(row["speed_mps"]),
                    "window_ms": window_ms,
                    "metric": metric,
                    "run_window_count": len(worst_foot_rates),
                    "run_median": float(np.median(worst_foot_rates)),
                    "run_p95": float(np.percentile(worst_foot_rates, 95)),
                }
            )
    return output


def _selected_series(
    per_run_rows: Sequence[Mapping[str, object]],
    metric: str,
    alignment: str,
    run_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (
            row
            for row in per_run_rows
            if row["metric"] == metric
            and row["alignment"] == alignment
            and row["run_id"] == run_id
            and row["value"] != ""
        ),
        key=lambda row: int(row["horizon_ms"]),
    )
    return (
        np.asarray([int(row["horizon_ms"]) for row in selected]),
        np.asarray([float(row["value"]) for row in selected]),
    )


def _plot_metric_trajectories(
    path: Path,
    per_run_rows: Sequence[Mapping[str, object]],
    runs: Sequence[SinkRun],
    metrics: Sequence[str],
    title: str,
) -> None:
    figure, axes = plt.subplots(
        len(metrics), 2, figsize=(14, 3.5 * len(metrics)), squeeze=False, sharex=True
    )
    colors = {"mild": "tab:blue", "moderate": "tab:orange", "severe": "tab:red"}
    for row_index, metric in enumerate(metrics):
        for column_index, alignment in enumerate(("t0", "t1")):
            axis = axes[row_index, column_index]
            for run in runs:
                horizons, values = _selected_series(
                    per_run_rows, metric, alignment, run.run_id
                )
                axis.plot(
                    horizons,
                    values,
                    color=colors[run.severity],
                    alpha=0.35,
                    linewidth=1.0,
                )
            for severity, color in colors.items():
                severity_rows = [run for run in runs if run.severity == severity]
                trajectories = []
                reference_horizons = None
                for run in severity_rows:
                    horizons, values = _selected_series(
                        per_run_rows, metric, alignment, run.run_id
                    )
                    if len(values):
                        reference_horizons = horizons
                        trajectories.append(values)
                if trajectories and len({len(values) for values in trajectories}) == 1:
                    axis.plot(
                        reference_horizons,
                        np.median(np.stack(trajectories), axis=0),
                        color=color,
                        linewidth=2.3,
                        label=severity,
                    )
            axis.set_title(f"{alignment}: {metric}")
            axis.grid(alpha=0.25)
            axis.set_xlabel(f"time from {alignment} (ms)")
            axis.set_ylabel(metric)
    axes[0, 0].legend(ncol=3)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_heatmap(
    path: Path, separation_rows: Sequence[Mapping[str, object]]
) -> None:
    columns = [
        (alignment, horizon)
        for alignment in ("t0", "t1")
        for horizon in (20, 50, 100)
    ]
    matrix = np.full((len(TEMPORAL_METRICS), len(columns)), np.nan)
    for row_index, metric in enumerate(TEMPORAL_METRICS):
        for column_index, (alignment, horizon) in enumerate(columns):
            match = next(
                (
                    row
                    for row in separation_rows
                    if row["metric"] == metric
                    and row["alignment"] == alignment
                    and int(row["horizon_ms"]) == horizon
                ),
                None,
            )
            if match is not None:
                matrix[row_index, column_index] = float(match["oriented_auc"])
    figure, axis = plt.subplots(figsize=(10, 12))
    image = axis.imshow(matrix, aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
    axis.set_xticks(
        range(len(columns)), [f"{alignment}+{horizon}" for alignment, horizon in columns]
    )
    axis.set_yticks(range(len(TEMPORAL_METRICS)), TEMPORAL_METRICS)
    axis.set_title("Run-level temporal redistribution oriented AUROC")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    figure.subplots_adjust(left=0.36, right=0.91, bottom=0.08, top=0.95)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_severity_progression(
    path: Path,
    per_run_rows: Sequence[Mapping[str, object]],
    best_t1_100: Mapping[str, object],
) -> None:
    selected = [
        row
        for row in per_run_rows
        if row["alignment"] == "t1"
        and int(row["horizon_ms"]) == 100
        and row["metric"] == best_t1_100["metric"]
        and row["value"] != ""
    ]
    groups = ("mild", "moderate", "severe")
    values = [
        [float(row["value"]) for row in selected if row["severity"] == group]
        for group in groups
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(values, labels=groups, showfliers=False)
    for index, group_values in enumerate(values, start=1):
        axis.scatter(
            np.full(len(group_values), index), group_values, color="black", alpha=0.7
        )
    axis.set(
        title=f"Severity progression: {best_t1_100['metric']} at t1+100 ms",
        ylabel=str(best_t1_100["metric"]),
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_uniform_sand(
    path: Path,
    rankings: Sequence[Mapping[str, object]],
    uniform_rows: Sequence[Mapping[str, object]],
) -> None:
    top = [
        row
        for row in rankings
        if row["alignment"] == "t1"
        and int(row["horizon_ms"]) == 100
        and int(row["rank"]) <= 5
    ]
    matches = [
        (
            row,
            next(
                (
                    item
                    for item in uniform_rows
                    if int(item["horizon_ms"]) == 100
                    and item["metric"] == row["metric"]
                ),
                None,
            ),
        )
        for row in top
    ]
    labels = [str(row["metric"]) for row, _ in matches]
    primary_auc = [float(row["oriented_auc"]) for row, _ in matches]
    sand_auc = [
        np.nan if uniform is None else float(uniform["primary_direction_auc"])
        for _, uniform in matches
    ]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(x - 0.18, primary_auc, width=0.36, label="benign Sink vs severe")
    axis.bar(x + 0.18, sand_auc, width=0.36, label="Uniform Sand vs severe")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("AUROC in primary hazardous direction")
    axis.set_title("Uniform Sand hard-negative audit at t1+100 ms")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_side_consistency(
    path: Path, rankings: Sequence[Mapping[str, object]]
) -> None:
    selected = [
        row
        for row in rankings
        if row["alignment"] == "t1"
        and int(row["horizon_ms"]) in {20, 50, 100}
        and int(row["rank"]) == 1
    ]
    labels = [f"{row['horizon_ms']}:{row['metric']}" for row in selected]
    left = [float(row["left_primary_direction_auc"]) for row in selected]
    right = [float(row["right_primary_direction_auc"]) for row in selected]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(x - 0.18, left, width=0.36, label="left severe")
    axis.bar(x + 0.18, right, width=0.36, label="right severe")
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("AUROC in pooled primary direction")
    axis.set_title("Affected-side consistency of top temporal candidates")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_fsr_temporal_redistribution_analysis(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Execute the bounded causal analysis without simulation or training."""
    repository_root = repository_root.resolve()
    with config_path.resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["experiment"]["id"] != "FSR_TEMPORAL_REDISTRIBUTION_ANALYSIS":
        raise ValueError("unsupported temporal redistribution experiment")
    declared_metrics = tuple(
        metric
        for family in config["temporal_metrics"].values()
        for metric in family
    )
    if declared_metrics != TEMPORAL_METRICS:
        raise ValueError("config temporal metric declaration does not match source")
    dataset_path = _resolve_path(repository_root, config["dataset"]["path"], "dataset.path")
    artifact_path = _resolve_path(repository_root, config["artifacts"]["path"], "artifacts.path")
    previous_static_path = _resolve_path(
        repository_root,
        config["previous_static_analysis"]["artifact_path"],
        "previous_static_analysis.artifact_path",
    )
    manifest_path = dataset_path / "manifest.csv"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != config["dataset"]["manifest_sha256"]:
        raise ValueError("sensor dataset manifest SHA-256 mismatch")
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite analysis artifacts: {artifact_path}")

    horizons = [int(value) for value in config["horizons_ms"]]
    low_load_threshold = float(config["load_threshold_n"])
    if not np.isclose(low_load_threshold, LOAD_OFF_N, rtol=0.0, atol=0.0):
        raise ValueError("load threshold must match the frozen load-off threshold")
    initial_width = int(config["initial_median_width_ms"])
    endpoint_width = int(config["endpoint_median_width_ms"])
    baseline_start = int(config["baseline_ms"]["start"])
    baseline_stop = int(config["baseline_ms"]["stop"])
    minimum_baseline = int(config["minimum_baseline_valid_samples"])
    runs, manifest_rows = _load_sink_runs(dataset_path)
    if sum(run.outcome == "BENIGN" for run in runs) != 4 or sum(run.outcome == "SINK" for run in runs) != 9:
        raise ValueError("unexpected primary Sink run counts")

    traces: dict[str, TemporalTrace] = {}
    events: dict[str, dict[str, int | None]] = {}
    per_run_rows: list[dict[str, object]] = []
    for run in runs:
        arrays = _load_arrays(run.path)
        trace = derive_temporal_trace(
            arrays["foot_fsr"], run.affected_side, low_load_threshold
        )
        event = _event_samples(arrays, run.affected_side)
        traces[run.run_id] = trace
        events[run.run_id] = event
        for alignment in ("t0", "t1"):
            anchor = int(event[alignment])
            for horizon in horizons:
                values, diagnostics = temporal_horizon_metrics(
                    trace,
                    anchor,
                    horizon,
                    initial_width,
                    endpoint_width,
                    baseline_start,
                    baseline_stop,
                    minimum_baseline,
                )
                endpoint = anchor + horizon
                t2 = event["t2"]
                t2_in_interval = bool(
                    t2 is not None and anchor <= int(t2) <= endpoint
                )
                post_t2_samples = (
                    0
                    if t2 is None or int(t2) > endpoint
                    else endpoint - max(anchor, int(t2)) + 1
                )
                both_loaded = trace.distribution.affected_loaded & trace.distribution.unaffected_loaded
                both_loaded_count = int(np.count_nonzero(both_loaded[anchor : endpoint + 1]))
                for metric in TEMPORAL_METRICS:
                    family = PATH_FAMILY.get(metric, "")
                    per_run_rows.append(
                        {
                            "run_id": run.run_id,
                            "group": "BENIGN_SINK" if run.outcome == "BENIGN" else "HAZARDOUS_SINK",
                            "severity": run.severity,
                            "affected_side": run.affected_side,
                            "speed_mps": run.speed_mps,
                            "t0_contact_phase": run.contact_phase,
                            "t0_sample": event["t0"],
                            "t1_sample": event["t1"],
                            "t2_sample": "" if t2 is None else t2,
                            "alignment": alignment,
                            "horizon_ms": horizon,
                            "endpoint_sample": endpoint,
                            "contact_age_endpoint_ms": endpoint - int(event["t0"]),
                            "t2_in_interval": t2_in_interval,
                            "samples_at_or_after_t2": post_t2_samples,
                            "metric": metric,
                            "direction_independent": metric in DIRECTION_INDEPENDENT_METRICS,
                            "value": "" if values[metric] is None else values[metric],
                            "valid_path_pairs": ""
                            if not family
                            else diagnostics["path_pairs"][family],
                            "initial_start_sample": diagnostics["initial_start_sample"],
                            "initial_stop_sample_exclusive": diagnostics["initial_stop_sample_exclusive"],
                            "initial_valid_distribution_samples": diagnostics["initial_valid_distribution_samples"],
                            "endpoint_start_sample": diagnostics["endpoint_start_sample"],
                            "endpoint_stop_sample_exclusive": diagnostics["endpoint_stop_sample_exclusive"],
                            "endpoint_valid_distribution_samples": diagnostics["endpoint_valid_distribution_samples"],
                            "interval_valid_distribution_samples": diagnostics["interval_valid_distribution_samples"],
                            "both_feet_loaded_interval_samples": both_loaded_count,
                            "baseline_start_sample": diagnostics["baseline_start_sample"],
                            "baseline_stop_sample_exclusive": diagnostics["baseline_stop_sample_exclusive"],
                            "baseline_valid_distribution_samples": diagnostics["baseline_valid_distribution_samples"],
                        }
                    )

    separation = _temporal_separation_rows(per_run_rows, runs)
    rankings = _temporal_ranking_rows(separation)
    pre_t2_rows = [
        row for row in per_run_rows if not bool(row["t2_in_interval"])
    ]
    pre_t2_separation = _temporal_separation_rows(pre_t2_rows, runs)
    pre_t2_rankings = _temporal_ranking_rows(pre_t2_separation)
    _, static_separation, static_rankings = _static_analysis_rows(
        runs,
        traces,
        events,
        horizons,
        endpoint_width,
        baseline_start,
        baseline_stop,
        minimum_baseline,
    )
    static_parity = _verify_previous_static_parity(
        previous_static_path,
        manifest_sha,
        static_rankings,
        [int(value) for value in config["previous_static_analysis"]["parity_horizons_ms"]],
    )
    static_vs_temporal = _static_vs_temporal_rows(static_rankings, rankings)
    sand_controls = _load_uniform_sand_controls(
        manifest_rows, dataset_path, low_load_threshold
    )
    uniform = _uniform_sand_rows(
        sand_controls,
        per_run_rows,
        separation,
        horizons,
        initial_width,
        endpoint_width,
        baseline_start,
        baseline_stop,
        minimum_baseline,
    )
    metadata_audit = _metadata_audit_rows(per_run_rows, rankings)
    reference_terrain = _reference_terrain_rows(
        manifest_rows,
        dataset_path,
        low_load_threshold,
        int(config["reference_terrain"]["window_ms"]),
    )

    artifact_path.mkdir(parents=True, exist_ok=True)
    plots_path = artifact_path / "plots"
    plots_path.mkdir()
    best_t1_100 = next(
        row
        for row in rankings
        if row["alignment"] == "t1"
        and int(row["horizon_ms"]) == 100
        and int(row["rank"]) == 1
    )
    _plot_metric_trajectories(
        plots_path / "quadrant_redistribution_trajectory.png",
        per_run_rows,
        runs,
        ("quadrant_l1_change",),
        "Affected-foot net quadrant redistribution",
    )
    _plot_metric_trajectories(
        plots_path / "quadrant_path_length_accumulation.png",
        per_run_rows,
        runs,
        ("quadrant_path_length", "quadrant_path_rate"),
        "Affected-foot raw 1 ms redistribution path",
    )
    _plot_metric_trajectories(
        plots_path / "concentration_entropy_dynamics.png",
        per_run_rows,
        runs,
        ("concentration_abs_change", "entropy_abs_change", "concentration_path", "entropy_path"),
        "Direction-independent concentration and entropy dynamics",
    )
    _plot_metric_trajectories(
        plots_path / "cop_path_trajectory.png",
        per_run_rows,
        runs,
        ("cop_displacement", "cop_path_length", "cop_path_efficiency"),
        "Normalized FSR load-center proxy movement",
    )
    _plot_heatmap(plots_path / "temporal_separation_heatmap.png", separation)
    _plot_severity_progression(
        plots_path / "severity_progression.png", per_run_rows, best_t1_100
    )
    _plot_uniform_sand(
        plots_path / "uniform_sand_comparison.png", rankings, uniform
    )
    _plot_side_consistency(
        plots_path / "left_right_consistency.png", rankings
    )

    _write_csv(artifact_path / "per_run_temporal_metrics.csv", per_run_rows)
    _write_csv(artifact_path / "horizon_separation.csv", separation)
    _write_csv(
        artifact_path / "pre_t2_horizon_separation.csv", pre_t2_separation
    )
    _write_csv(
        artifact_path / "pre_t2_feature_ranking.csv", pre_t2_rankings
    )
    _write_csv(artifact_path / "static_horizon_separation.csv", static_separation)
    _write_csv(artifact_path / "static_vs_temporal.csv", static_vs_temporal)
    _write_csv(artifact_path / "feature_ranking.csv", rankings)
    _write_csv(artifact_path / "uniform_sand_audit.csv", uniform)
    _write_csv(artifact_path / "metadata_audit.csv", metadata_audit)
    _write_csv(artifact_path / "reference_terrain.csv", reference_terrain)
    primary_horizons = (20, 50, 100)
    top_by_horizon = {
        f"{alignment}_{horizon}ms": next(
            dict(row)
            for row in rankings
            if row["alignment"] == alignment
            and int(row["horizon_ms"]) == horizon
            and int(row["rank"]) == 1
        )
        for alignment in ("t0", "t1")
        for horizon in primary_horizons
    }
    static_top_by_horizon = {
        f"{alignment}_{horizon}ms": next(
            dict(row)
            for row in static_rankings
            if row["alignment"] == alignment
            and int(row["horizon_ms"]) == horizon
            and int(row["rank"]) == 1
        )
        for alignment in ("t0", "t1")
        for horizon in primary_horizons
    }
    pre_t2_top_by_horizon = {
        f"{alignment}_{horizon}ms": next(
            dict(row)
            for row in pre_t2_rankings
            if row["alignment"] == alignment
            and int(row["horizon_ms"]) == horizon
            and int(row["rank"]) == 1
        )
        for alignment in ("t0", "t1")
        for horizon in primary_horizons
    }
    summary = {
        "experiment_id": config["experiment"]["id"],
        "analysis_only": True,
        "simulation_executed": False,
        "training_executed": False,
        "dataset_id": config["dataset"]["id"],
        "manifest_sha256": manifest_sha,
        "source_commit": _git_commit(repository_root),
        "run_counts": {
            "mild_benign": 2,
            "moderate_benign": 2,
            "severe_hazardous": 9,
            "uniform_sand_secondary": 4,
        },
        "statistical_unit": "one_physical_run",
        "horizons_ms": horizons,
        "initial_median_width_ms": initial_width,
        "endpoint_median_width_ms": endpoint_width,
        "path_sampling_ms": 1,
        "low_load_threshold_n": low_load_threshold,
        "static_parity": static_parity,
        "top_by_horizon": top_by_horizon,
        "pre_t2_top_by_horizon": pre_t2_top_by_horizon,
        "static_top_by_horizon": static_top_by_horizon,
        "artifact_files": [
            "per_run_temporal_metrics.csv",
            "horizon_separation.csv",
            "pre_t2_horizon_separation.csv",
            "pre_t2_feature_ranking.csv",
            "static_horizon_separation.csv",
            "static_vs_temporal.csv",
            "feature_ranking.csv",
            "uniform_sand_audit.csv",
            "metadata_audit.csv",
            "reference_terrain.csv",
            "plots/",
        ],
        "interpretation_guardrails": [
            "temporal_metrics_are_analysis_only_not_runtime_features",
            "path_metrics_use_raw_1ms_steps_and_do_not_bridge_invalid_gaps",
            "cop_is_a_normalized_load_center_proxy_not_physical_continuous_cop",
            "t2_contamination_is_flagged_per_run_horizon",
            "no_classifier_threshold_or_sensor_architecture_is_frozen",
        ],
    }
    _write_json(artifact_path / "summary.json", summary)
    progress("FSR temporal redistribution analysis complete; no simulation or training executed")
    return artifact_path, summary
