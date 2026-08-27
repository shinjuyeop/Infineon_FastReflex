"""Read-only run-level analysis of virtual-FSR load distribution around Sink events."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Callable, Mapping, Sequence

import matplotlib
import numpy as np
import yaml

from fastreflex.dataset.loader import sha256_file
from fastreflex.simulation.hazards import LOAD_OFF_N


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


EPSILON = 1.0e-12
ABSOLUTE_METRICS = (
    "affected_total_n",
    "unaffected_total_n",
    "bilateral_total_n",
)
DISTRIBUTION_METRICS = (
    "front_ratio",
    "rear_ratio",
    "local_left_ratio",
    "local_right_ratio",
    "medial_ratio",
    "lateral_ratio",
    "front_left_share",
    "front_right_share",
    "rear_left_share",
    "rear_right_share",
    "cop_x_proxy",
    "cop_y_proxy",
    "cop_radius_proxy",
    "max_quadrant_share",
    "load_concentration",
    "affected_load_share",
    "bilateral_asymmetry",
    "signed_bilateral_shift",
)
ALL_METRICS = (*ABSOLUTE_METRICS, *DISTRIBUTION_METRICS)
REDUNDANT_COMPLEMENT_METRICS = {
    "rear_ratio",
    "local_left_ratio",
    "local_right_ratio",
    "lateral_ratio",
}
QUADRANT_SHARE_METRICS = (
    "front_left_share",
    "front_right_share",
    "rear_left_share",
    "rear_right_share",
)


@dataclass(frozen=True)
class SinkRun:
    run_id: str
    path: Path
    outcome: str
    severity: str
    affected_side: str
    speed_mps: float
    contact_phase: float


@dataclass(frozen=True)
class DistributionTrace:
    metrics: dict[str, np.ndarray]
    affected_distribution_valid: np.ndarray
    affected_loaded: np.ndarray
    unaffected_loaded: np.ndarray
    bilateral_valid: np.ndarray


def canonicalize_affected_foot(
    foot_fsr: np.ndarray, affected_side: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return affected then unaffected FSR4 without mirroring foot-local axes."""
    values = np.asarray(foot_fsr, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("foot_fsr must have shape [N,8]")
    if affected_side not in {"left", "right"}:
        raise ValueError("affected_side must be left or right")
    feet = values.reshape(-1, 2, 4)
    affected_index = 0 if affected_side == "left" else 1
    return feet[:, affected_index], feet[:, 1 - affected_index]


def derive_distribution_trace(
    foot_fsr: np.ndarray,
    affected_side: str,
    low_load_threshold_n: float = LOAD_OFF_N,
) -> DistributionTrace:
    """Derive deterministic descriptive quantities from raw FSR8.

    Foot-internal ratios are NaN below the frozen 2.5 N load-off threshold.
    Bilateral metrics remain valid during single support and explicitly expose
    each foot's loaded state so gait-phase dependence cannot be hidden.
    """
    affected, unaffected = canonicalize_affected_foot(foot_fsr, affected_side)
    if not np.isfinite(affected).all() or np.any(affected < 0.0):
        raise ValueError("FSR input must be finite and nonnegative")
    affected_total = affected.sum(axis=1)
    unaffected_total = unaffected.sum(axis=1)
    bilateral_total = affected_total + unaffected_total
    affected_valid = affected_total >= low_load_threshold_n
    affected_loaded = affected_total >= low_load_threshold_n
    unaffected_loaded = unaffected_total >= low_load_threshold_n
    bilateral_valid = bilateral_total >= low_load_threshold_n

    internal_denominator = np.where(affected_valid, affected_total, np.nan)
    shares = affected / internal_denominator[:, None]
    front = affected[:, 0] + affected[:, 1]
    rear = affected[:, 2] + affected[:, 3]
    local_left = affected[:, 0] + affected[:, 2]
    local_right = affected[:, 1] + affected[:, 3]
    medial = local_right if affected_side == "left" else local_left
    lateral = local_left if affected_side == "left" else local_right
    front_ratio = front / internal_denominator
    rear_ratio = rear / internal_denominator
    local_left_ratio = local_left / internal_denominator
    local_right_ratio = local_right / internal_denominator
    medial_ratio = medial / internal_denominator
    lateral_ratio = lateral / internal_denominator
    cop_x = (front - rear) / internal_denominator
    cop_y = (local_left - local_right) / internal_denominator
    cop_radius = np.sqrt(np.square(cop_x) + np.square(cop_y))
    bilateral_denominator = np.where(bilateral_valid, bilateral_total, np.nan)

    metrics = {
        "affected_total_n": affected_total,
        "unaffected_total_n": unaffected_total,
        "bilateral_total_n": bilateral_total,
        "front_ratio": front_ratio,
        "rear_ratio": rear_ratio,
        "local_left_ratio": local_left_ratio,
        "local_right_ratio": local_right_ratio,
        "medial_ratio": medial_ratio,
        "lateral_ratio": lateral_ratio,
        "front_left_share": shares[:, 0],
        "front_right_share": shares[:, 1],
        "rear_left_share": shares[:, 2],
        "rear_right_share": shares[:, 3],
        "cop_x_proxy": cop_x,
        "cop_y_proxy": cop_y,
        "cop_radius_proxy": cop_radius,
        "max_quadrant_share": np.max(shares, axis=1),
        "load_concentration": np.sum(np.square(shares), axis=1),
        "affected_load_share": affected_total / bilateral_denominator,
        "bilateral_asymmetry": (
            np.abs(affected_total - unaffected_total) / bilateral_denominator
        ),
        "signed_bilateral_shift": (
            (affected_total - unaffected_total) / bilateral_denominator
        ),
    }
    return DistributionTrace(
        metrics={name: np.asarray(values, dtype=np.float64) for name, values in metrics.items()},
        affected_distribution_valid=affected_valid,
        affected_loaded=affected_loaded,
        unaffected_loaded=unaffected_loaded,
        bilateral_valid=bilateral_valid,
    )


def causal_trailing_median(
    values: np.ndarray,
    event_sample: int,
    horizon_ms: int,
    width_ms: int,
) -> tuple[float | None, int, tuple[int, int]]:
    """Summarize only samples available by an aligned horizon endpoint."""
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or horizon_ms < 0 or width_ms <= 0:
        raise ValueError("invalid causal trailing summary request")
    endpoint = event_sample + horizon_ms
    start = event_sample + max(0, horizon_ms - width_ms + 1)
    stop = min(endpoint + 1, len(series))
    if start < 0 or start >= stop or endpoint >= len(series):
        return None, 0, (start, stop)
    selected = series[start:stop]
    valid = np.isfinite(selected)
    if not valid.any():
        return None, 0, (start, stop)
    return float(np.median(selected[valid])), int(valid.sum()), (start, stop)


def pre_event_baseline_median(
    values: np.ndarray,
    event_sample: int,
    start_ms: int = -100,
    stop_ms: int = -20,
    minimum_valid_samples: int = 20,
) -> tuple[float | None, int, tuple[int, int]]:
    """Use only the declared pre-event interval, inclusive at both ends."""
    series = np.asarray(values, dtype=np.float64)
    start = event_sample + start_ms
    stop = event_sample + stop_ms + 1
    if start < 0 or stop > len(series) or start >= stop:
        return None, 0, (start, stop)
    selected = series[start:stop]
    valid = np.isfinite(selected)
    if int(valid.sum()) < minimum_valid_samples:
        return None, int(valid.sum()), (start, stop)
    return float(np.median(selected[valid])), int(valid.sum()), (start, stop)


def run_level_auc(
    benign_values: Sequence[float], hazardous_values: Sequence[float]
) -> tuple[float, float, str]:
    """Return raw and orientation-corrected AUROC using pairwise run ranks."""
    benign = np.asarray(benign_values, dtype=np.float64)
    hazard = np.asarray(hazardous_values, dtype=np.float64)
    if not len(benign) or not len(hazard):
        raise ValueError("AUROC requires both run groups")
    comparisons = hazard[:, None] - benign[None, :]
    auc = float((np.count_nonzero(comparisons > 0.0) + 0.5 * np.count_nonzero(comparisons == 0.0)) / comparisons.size)
    if auc >= 0.5:
        return auc, auc, "higher_is_hazardous"
    return auc, 1.0 - auc, "lower_is_hazardous"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    fields.extend(name for row in rows for name in row if name not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a repository-relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes repository") from exc
    return path


def _load_sink_runs(dataset_path: Path) -> tuple[list[SinkRun], list[dict[str, str]]]:
    with (dataset_path / "manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    runs = []
    for row in rows:
        if not row["sink_side"] or row["sink_severity"] not in {"mild", "moderate", "severe"}:
            continue
        if row["observed_outcome"] not in {"BENIGN", "SINK"}:
            continue
        runs.append(
            SinkRun(
                run_id=row["run_id"],
                path=dataset_path / row["file"],
                outcome=row["observed_outcome"],
                severity=row["sink_severity"],
                affected_side=row["sink_side"],
                speed_mps=float(row["speed_mps"]),
                contact_phase=float(row["first_contact_policy_phase"]),
            )
        )
    return runs, rows


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def _event_samples(arrays: Mapping[str, np.ndarray], side: str) -> dict[str, int | None]:
    side_index = 0 if side == "left" else 1
    t0 = int(arrays["first_patch_contact_sample_per_foot"][side_index])
    t1 = int(arrays["first_sink_physical_onset_sample_per_foot"][side_index])
    t2_value = int(arrays["first_sink_degradation_onset_sample"])
    censor = int(arrays["first_censor_sample"])
    t3 = len(arrays["sequence"]) if censor < 0 else censor
    if t0 < 0 or t1 < 0 or not t0 <= t1 < t3:
        raise ValueError("invalid unilateral Sink t0/t1/t3 ordering")
    return {"t0": t0, "t1": t1, "t2": None if t2_value < 0 else t2_value, "t3": t3}


def event_aligned_series(
    values: np.ndarray, event: int, start_ms: int, stop_ms: int
) -> np.ndarray:
    output = np.full(stop_ms - start_ms + 1, np.nan, dtype=np.float64)
    source_start = max(0, event + start_ms)
    source_stop = min(len(values), event + stop_ms + 1)
    if source_start < source_stop:
        destination_start = source_start - (event + start_ms)
        output[destination_start : destination_start + source_stop - source_start] = values[source_start:source_stop]
    return output


def _range_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(max(float(a.min()), float(b.min())) <= min(float(a.max()), float(b.max())))


def _separation_rows(
    per_run_rows: Sequence[Mapping[str, object]],
    runs: Sequence[SinkRun],
    alignments: Sequence[str],
    horizons: Sequence[int],
) -> list[dict[str, object]]:
    run_lookup = {run.run_id: run for run in runs}
    rows = []
    for alignment in alignments:
        for horizon in horizons:
            for metric in ALL_METRICS:
                selected = [
                    row for row in per_run_rows
                    if row["alignment"] == alignment
                    and int(row["horizon_ms"]) == horizon
                    and row["metric"] == metric
                ]
                for representation, field in (("raw", "raw_value"), ("delta", "baseline_delta")):
                    benign_pairs = [
                        (row["run_id"], float(row[field]))
                        for row in selected
                        if row["group"] == "BENIGN_SINK" and row[field] != ""
                    ]
                    hazard_pairs = [
                        (row["run_id"], float(row[field]))
                        for row in selected
                        if row["group"] == "HAZARDOUS_SINK" and row[field] != ""
                    ]
                    if not benign_pairs or not hazard_pairs:
                        continue
                    benign = np.asarray([value for _, value in benign_pairs])
                    hazard = np.asarray([value for _, value in hazard_pairs])
                    raw_auc, oriented_auc, direction = run_level_auc(benign, hazard)
                    side_auc: dict[str, float | None] = {}
                    for side in ("left", "right"):
                        side_benign = [value for run_id, value in benign_pairs if run_lookup[run_id].affected_side == side]
                        side_hazard = [value for run_id, value in hazard_pairs if run_lookup[run_id].affected_side == side]
                        side_auc[side] = (
                            None if not side_benign or not side_hazard
                            else run_level_auc(side_benign, side_hazard)[1]
                        )
                    rows.append(
                        {
                            "alignment": alignment,
                            "horizon_ms": horizon,
                            "metric": metric,
                            "metric_kind": "absolute" if metric in ABSOLUTE_METRICS else "distribution",
                            "representation": representation,
                            "benign_run_count": len(benign),
                            "hazardous_run_count": len(hazard),
                            "benign_median": float(np.median(benign)),
                            "benign_min": float(benign.min()),
                            "benign_max": float(benign.max()),
                            "hazardous_median": float(np.median(hazard)),
                            "hazardous_min": float(hazard.min()),
                            "hazardous_max": float(hazard.max()),
                            "absolute_median_difference": float(abs(np.median(hazard) - np.median(benign))),
                            "range_overlap": _range_overlap(benign, hazard),
                            "raw_auc": raw_auc,
                            "oriented_auc": oriented_auc,
                            "direction": direction,
                            "left_oriented_auc": side_auc["left"],
                            "right_oriented_auc": side_auc["right"],
                        }
                    )
    return rows


def _ranking_rows(separation_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rankings = []
    keys = sorted({(row["alignment"], int(row["horizon_ms"])) for row in separation_rows})
    for alignment, horizon in keys:
        candidates = [
            row for row in separation_rows
            if row["alignment"] == alignment and int(row["horizon_ms"]) == horizon
            and row["metric"] not in REDUNDANT_COMPLEMENT_METRICS
        ]
        candidates.sort(
            key=lambda row: (
                -float(row["oriented_auc"]),
                bool(row["range_overlap"]),
                row["metric"] in ABSOLUTE_METRICS,
                row["metric"],
                row["representation"],
            )
        )
        distinct_metrics = []
        seen_metrics = set()
        for row in candidates:
            if row["metric"] in seen_metrics:
                continue
            distinct_metrics.append(row)
            seen_metrics.add(row["metric"])
            if len(distinct_metrics) == 5:
                break
        for rank, row in enumerate(distinct_metrics, start=1):
            rankings.append({"rank": rank, **row})
    return rankings


def _first_sustained_outside(
    values: np.ndarray,
    start: int,
    stop: int,
    lower: float,
    upper: float,
    direction: str,
    persistence: int,
) -> int | None:
    count = 0
    for sample in range(start, min(stop, len(values))):
        value = float(values[sample])
        outside = bool(
            np.isfinite(value)
            and (
                (direction == "higher_is_hazardous" and value > upper)
                or (direction == "lower_is_hazardous" and value < lower)
            )
        )
        count = count + 1 if outside else 0
        if count >= persistence:
            return sample
    return None


def _load_uniform_sand_controls(
    manifest_rows: Sequence[Mapping[str, str]], dataset_path: Path
) -> list[dict[str, object]]:
    controls = []
    for row in manifest_rows:
        if not row["run_id"].startswith("normal_sand_"):
            continue
        arrays = _load_arrays(dataset_path / row["file"])
        traces = {
            side: derive_distribution_trace(arrays["foot_fsr"], side)
            for side in ("left", "right")
        }
        t1 = {
            side: int(arrays["first_sink_physical_onset_sample_per_foot"][index])
            for index, side in enumerate(("left", "right"))
        }
        if min(t1.values()) < 0:
            raise ValueError("Uniform Sand reference is missing per-foot physical Sink onset")
        controls.append({"run_id": row["run_id"], "traces": traces, "t1": t1})
    if len(controls) != 4:
        raise ValueError("expected four Uniform Sand reference runs")
    return controls


def _sand_pseudo_series(
    control: Mapping[str, object], metric: str, direction: str, stop_ms: int = 300
) -> np.ndarray:
    traces = control["traces"]
    events = control["t1"]
    assert isinstance(traces, Mapping) and isinstance(events, Mapping)
    sides = [
        event_aligned_series(
            traces[side].metrics[metric], int(events[side]), 0, stop_ms
        )
        for side in ("left", "right")
    ]
    if direction == "higher_is_hazardous":
        return np.fmax(sides[0], sides[1])
    return np.fmin(sides[0], sides[1])


def _benign_envelope_rows(
    runs: Sequence[SinkRun],
    traces: Mapping[str, DistributionTrace],
    events: Mapping[str, Mapping[str, int | None]],
    separation_rows: Sequence[Mapping[str, object]],
    sand_controls: Sequence[Mapping[str, object]],
    persistence: int,
) -> list[dict[str, object]]:
    rows = []
    for metric in ALL_METRICS:
        reference = max(
            (
                row for row in separation_rows
                if row["alignment"] == "t1"
                and int(row["horizon_ms"]) == 100
                and row["metric"] == metric
                and row["representation"] == "raw"
            ),
            key=lambda row: float(row["oriented_auc"]),
        )
        benign_values = []
        for run in runs:
            if run.outcome != "BENIGN":
                continue
            event = int(events[run.run_id]["t1"])
            aligned = event_aligned_series(
                traces[run.run_id].metrics[metric], event, 0, 300
            )
            benign_values.extend(aligned[np.isfinite(aligned)].tolist())
        sand_values = []
        for control in sand_controls:
            pseudo = _sand_pseudo_series(
                control, metric, str(reference["direction"])
            )
            sand_values.extend(pseudo[np.isfinite(pseudo)].tolist())
        envelope_scopes = (
            ("mild_moderate_sink_only", benign_values),
            ("mild_moderate_sink_plus_uniform_sand", [*benign_values, *sand_values]),
        )
        for envelope_scope, envelope_values in envelope_scopes:
            lower = float(np.min(envelope_values))
            upper = float(np.max(envelope_values))
            for run in runs:
                if run.outcome != "SINK":
                    continue
                event = events[run.run_id]
                t1 = int(event["t1"])
                t2 = event["t2"]
                stop = int(event["t3"]) if t2 is None else int(t2)
                first = _first_sustained_outside(
                    traces[run.run_id].metrics[metric],
                    t1,
                    stop,
                    lower,
                    upper,
                    str(reference["direction"]),
                    persistence,
                )
                rows.append(
                    {
                        "envelope_scope": envelope_scope,
                        "metric": metric,
                        "direction": reference["direction"],
                        "benign_envelope_min": lower,
                        "benign_envelope_max": upper,
                        "run_id": run.run_id,
                        "affected_side": run.affected_side,
                        "speed_mps": run.speed_mps,
                        "t0_sample": event["t0"],
                        "t1_sample": t1,
                        "t2_sample": "" if t2 is None else t2,
                        "first_sustained_outside_sample": "" if first is None else first,
                        "latency_from_t0_ms": "" if first is None else first - int(event["t0"]),
                        "latency_from_t1_ms": "" if first is None else first - t1,
                        "margin_to_t2_ms": "" if first is None or t2 is None else int(t2) - first,
                        "positive_margin": bool(t2 is not None and int(t2) > t1),
                    }
                )
    return rows


def _uniform_sand_rows(
    sand_controls: Sequence[Mapping[str, object]],
    separation_rows: Sequence[Mapping[str, object]],
    per_run_rows: Sequence[Mapping[str, object]],
    horizons: Sequence[int],
) -> list[dict[str, object]]:
    """Challenge t1 candidates with the more hazardous-looking of two sand feet."""
    output = []
    for horizon in horizons:
        for metric in ALL_METRICS:
            primary = next(
                row for row in separation_rows
                if row["alignment"] == "t1"
                and int(row["horizon_ms"]) == horizon
                and row["metric"] == metric
                and row["representation"] == "raw"
            )
            values = []
            choices = []
            for control in sand_controls:
                candidates = []
                traces = control["traces"]
                events = control["t1"]
                assert isinstance(traces, Mapping) and isinstance(events, Mapping)
                for side in ("left", "right"):
                    value, _, _ = causal_trailing_median(
                        traces[side].metrics[metric], int(events[side]), horizon, 10
                    )
                    if value is not None:
                        candidates.append((side, value))
                if not candidates:
                    continue
                chosen = (
                    max(candidates, key=lambda item: item[1])
                    if primary["direction"] == "higher_is_hazardous"
                    else min(candidates, key=lambda item: item[1])
                )
                values.append(chosen[1])
                choices.append(f"{control['run_id']}:{chosen[0]}")
            hazardous = [
                float(item["raw_value"])
                for item in per_run_rows
                if item["alignment"] == "t1"
                and int(item["horizon_ms"]) == horizon
                and item["metric"] == metric
                and item["group"] == "HAZARDOUS_SINK"
                and item["raw_value"] != ""
            ]
            if values and hazardous:
                raw_auc, oriented_auc, direction = run_level_auc(values, hazardous)
                output.append(
                    {
                        "horizon_ms": horizon,
                        "metric": metric,
                        "selection": "per_run_more_hazardous_pseudo_foot_at_uniform_sink_physical_onset",
                        "sand_run_count": len(values),
                        "hazardous_run_count": len(hazardous),
                        "sand_median": float(np.median(values)),
                        "sand_min": float(np.min(values)),
                        "sand_max": float(np.max(values)),
                        "hazardous_median": float(np.median(hazardous)),
                        "hazardous_min": float(np.min(hazardous)),
                        "hazardous_max": float(np.max(hazardous)),
                        "range_overlap": _range_overlap(np.asarray(values), np.asarray(hazardous)),
                        "raw_auc": raw_auc,
                        "oriented_auc": oriented_auc,
                        "direction": direction,
                        "selected_pseudo_feet": "|".join(choices),
                    }
                )
    return output


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(values: Sequence[float], metadata: Sequence[float]) -> float | None:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(metadata, dtype=np.float64)
    if len(x) < 3 or np.ptp(x) <= EPSILON or np.ptp(y) <= EPSILON:
        return None
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def _metadata_audit_rows(
    per_run_rows: Sequence[Mapping[str, object]],
    rankings: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    top = [
        row
        for row in rankings
        if row["alignment"] == "t1" and int(row["horizon_ms"]) == 100
    ]
    for ranking in top:
        field = "raw_value" if ranking["representation"] == "raw" else "baseline_delta"
        selected = [
            row
            for row in per_run_rows
            if row["alignment"] == "t1"
            and int(row["horizon_ms"]) == 100
            and row["metric"] == ranking["metric"]
            and row[field] != ""
        ]

        def median(rows: Sequence[Mapping[str, object]]) -> float | str:
            return "" if not rows else float(np.median([float(row[field]) for row in rows]))

        severity_medians = {
            severity: median([row for row in selected if row["severity"] == severity])
            for severity in ("mild", "moderate", "severe")
        }
        severe = [row for row in selected if row["severity"] == "severe"]
        speed_medians = {
            speed: median([row for row in severe if float(row["speed_mps"]) == speed])
            for speed in sorted({float(row["speed_mps"]) for row in severe})
        }
        ordered = [severity_medians[name] for name in ("mild", "moderate", "severe")]
        monotonic = False
        if all(value != "" for value in ordered):
            numeric = [float(value) for value in ordered]
            monotonic = (
                numeric[0] <= numeric[1] <= numeric[2]
                if ranking["direction"] == "higher_is_hazardous"
                else numeric[0] >= numeric[1] >= numeric[2]
            )
        output.append(
            {
                "rank": ranking["rank"],
                "metric": ranking["metric"],
                "representation": ranking["representation"],
                "direction": ranking["direction"],
                "mild_median": severity_medians["mild"],
                "moderate_median": severity_medians["moderate"],
                "severe_median": severity_medians["severe"],
                "severity_median_monotonic": monotonic,
                "left_severe_median": median(
                    [row for row in severe if row["affected_side"] == "left"]
                ),
                "right_severe_median": median(
                    [row for row in severe if row["affected_side"] == "right"]
                ),
                "severe_speed_medians_json": json.dumps(speed_medians, sort_keys=True),
                "severe_speed_spearman": ""
                if (correlation := _spearman(
                    [float(row[field]) for row in severe],
                    [float(row["speed_mps"]) for row in severe],
                )) is None
                else correlation,
                "all_run_contact_phase_spearman": ""
                if (correlation := _spearman(
                    [float(row[field]) for row in selected],
                    [float(row["t0_contact_phase"]) for row in selected],
                )) is None
                else correlation,
            }
        )
    return output


def _reference_terrain_rows(
    manifest_rows: Sequence[Mapping[str, str]], dataset_path: Path
) -> list[dict[str, object]]:
    output = []
    for row in manifest_rows:
        if not row["run_id"].startswith(("normal_concrete_", "normal_marble_", "normal_sand_")):
            continue
        arrays = _load_arrays(dataset_path / row["file"])
        left = derive_distribution_trace(arrays["foot_fsr"], "left")
        right = derive_distribution_trace(arrays["foot_fsr"], "right")
        symmetric = {
            "bilateral_total_n": left.metrics["bilateral_total_n"],
            "max_foot_load_share": np.fmax(
                left.metrics["affected_load_share"], right.metrics["affected_load_share"]
            ),
            "bilateral_asymmetry": left.metrics["bilateral_asymmetry"],
            "maximum_foot_concentration": np.fmax(
                left.metrics["load_concentration"], right.metrics["load_concentration"]
            ),
            "maximum_abs_cop_x_proxy": np.fmax(
                np.abs(left.metrics["cop_x_proxy"]), np.abs(right.metrics["cop_x_proxy"])
            ),
            "maximum_abs_cop_y_proxy": np.fmax(
                np.abs(left.metrics["cop_y_proxy"]), np.abs(right.metrics["cop_y_proxy"])
            ),
            "maximum_cop_radius_proxy": np.fmax(
                left.metrics["cop_radius_proxy"], right.metrics["cop_radius_proxy"]
            ),
        }
        for metric, values in symmetric.items():
            finite = np.asarray(values)[np.isfinite(values)]
            if not len(finite):
                continue
            output.append(
                {
                    "run_id": row["run_id"],
                    "terrain": row["terrain"],
                    "speed_mps": float(row["speed_mps"]),
                    "metric": metric,
                    "run_median": float(np.median(finite)),
                    "run_p95": float(np.percentile(finite, 95)),
                    "valid_sample_count": len(finite),
                }
            )
    return output


def _plot_group_trajectory(
    path: Path,
    runs: Sequence[SinkRun],
    traces: Mapping[str, DistributionTrace],
    events: Mapping[str, Mapping[str, int | None]],
    alignment: str,
    metrics: Sequence[str],
    title: str,
) -> None:
    def column_median(parts: Sequence[np.ndarray]) -> np.ndarray:
        stacked = np.stack(parts)
        result = np.full(stacked.shape[1], np.nan, dtype=np.float64)
        for column in range(stacked.shape[1]):
            finite = stacked[:, column][np.isfinite(stacked[:, column])]
            if len(finite):
                result[column] = np.median(finite)
        return result

    time_ms = np.arange(-100, 301)
    figure, axes = plt.subplots(len(metrics), 1, figsize=(12, 3.3 * len(metrics)), sharex=True)
    axes_array = np.atleast_1d(axes)
    colors = {"mild": "tab:blue", "moderate": "tab:orange", "severe": "tab:red"}
    for metric, axis in zip(metrics, axes_array):
        by_severity: dict[str, list[np.ndarray]] = {name: [] for name in colors}
        for run in runs:
            event = int(events[run.run_id][alignment])
            aligned = event_aligned_series(
                traces[run.run_id].metrics[metric], event, -100, 300
            )
            by_severity[run.severity].append(aligned)
            axis.plot(time_ms, aligned, color=colors[run.severity], alpha=0.18, linewidth=0.8)
        for severity, parts in by_severity.items():
            axis.plot(
                time_ms,
                column_median(parts),
                color=colors[severity],
                linewidth=2.0,
                label=severity,
            )
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
        if alignment == "t1":
            for run in runs:
                t2 = events[run.run_id]["t2"]
                if run.outcome == "SINK" and t2 is not None:
                    relative = int(t2) - int(events[run.run_id]["t1"])
                    if -100 <= relative <= 300:
                        axis.axvline(relative, color="tab:red", alpha=0.12, linewidth=0.7)
        axis.set_ylabel(metric)
        axis.grid(alpha=0.2)
    axes_array[0].legend(ncol=3)
    axes_array[-1].set_xlabel(f"time from {alignment} (ms)")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_heatmap(path: Path, separation: Sequence[Mapping[str, object]], horizons: Sequence[int]) -> None:
    columns = [(alignment, horizon) for alignment in ("t0", "t1") for horizon in horizons if horizon in {20, 50, 100, 150, 200}]
    figure, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True)
    for axis, representation in zip(axes, ("raw", "delta")):
        matrix = np.full((len(ALL_METRICS), len(columns)), np.nan)
        for row_index, metric in enumerate(ALL_METRICS):
            for column_index, (alignment, horizon) in enumerate(columns):
                match = next((row for row in separation if row["alignment"] == alignment and int(row["horizon_ms"]) == horizon and row["metric"] == metric and row["representation"] == representation), None)
                if match is not None:
                    matrix[row_index, column_index] = float(match["oriented_auc"])
        image = axis.imshow(matrix, aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
        axis.set_title(f"{representation} run-level oriented AUROC")
        axis.set_xticks(range(len(columns)), [f"{a}+{h}" for a, h in columns], rotation=45, ha="right")
        axis.set_yticks(range(len(ALL_METRICS)), ALL_METRICS)
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    figure.suptitle("Benign mild/moderate vs hazardous severe separation")
    figure.subplots_adjust(left=0.22, right=0.92, bottom=0.14, top=0.90, wspace=0.08)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_progression(
    path: Path,
    per_run: Sequence[Mapping[str, object]],
    best_row: Mapping[str, object],
) -> None:
    selected = [
        row for row in per_run
        if row["alignment"] == best_row["alignment"]
        and int(row["horizon_ms"]) == int(best_row["horizon_ms"])
        and row["metric"] == best_row["metric"]
    ]
    field = "raw_value" if best_row["representation"] == "raw" else "baseline_delta"
    groups = ("mild", "moderate", "severe")
    values = [[float(row[field]) for row in selected if row["severity"] == group and row[field] != ""] for group in groups]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(values, labels=groups, showfliers=False)
    for index, group_values in enumerate(values, start=1):
        axis.scatter(np.full(len(group_values), index), group_values, color="black", s=22, alpha=0.7)
    axis.set(ylabel=f"{best_row['metric']} ({best_row['representation']})", title="Mild → moderate → severe run-level progression")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_uniform_sand(path: Path, uniform_rows: Sequence[Mapping[str, object]], top_metrics: Sequence[str]) -> None:
    selected = [row for row in uniform_rows if int(row["horizon_ms"]) == 100 and row["metric"] in top_metrics]
    figure, axis = plt.subplots(figsize=(11, 5))
    x = np.arange(len(selected))
    sand = [float(row["sand_median"]) for row in selected]
    hazard = [float(row["hazardous_median"]) for row in selected]
    axis.scatter(x - 0.08, sand, label="uniform sand worst-case median", marker="o")
    axis.scatter(x + 0.08, hazard, label="hazardous severe median", marker="x")
    axis.set_xticks(x, [row["metric"] for row in selected], rotation=30, ha="right")
    axis.set_title("Uniform Sand secondary challenge at physical-sink onset +100 ms")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def run_fsr_load_distribution_analysis(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Execute the bounded read-only descriptive analysis and write ignored artifacts."""
    repository_root = repository_root.resolve()
    with config_path.resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["experiment"]["id"] != "FSR_LOAD_DISTRIBUTION_ANALYSIS":
        raise ValueError("unsupported evaluation experiment")
    dataset_path = _resolve_path(repository_root, config["dataset"]["path"], "dataset.path")
    artifact_path = _resolve_path(repository_root, config["artifacts"]["path"], "artifacts.path")
    manifest_path = dataset_path / "manifest.csv"
    if sha256_file(manifest_path) != config["dataset"]["manifest_sha256"]:
        raise ValueError("sensor dataset manifest SHA-256 mismatch")
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite analysis artifacts: {artifact_path}")
    artifact_path.mkdir(parents=True, exist_ok=True)
    plots_path = artifact_path / "plots"
    plots_path.mkdir()

    horizons = [int(value) for value in config["horizons_ms"]]
    width = int(config["trailing_summary_ms"])
    baseline = config["baseline_ms"]
    minimum_baseline = int(config["minimum_baseline_valid_samples"])
    runs, manifest_rows = _load_sink_runs(dataset_path)
    if sum(run.outcome == "BENIGN" for run in runs) != 4 or sum(run.outcome == "SINK" for run in runs) != 9:
        raise ValueError("unexpected primary Sink run counts")
    traces: dict[str, DistributionTrace] = {}
    events: dict[str, dict[str, int | None]] = {}
    per_run_rows: list[dict[str, object]] = []
    for run in runs:
        arrays = _load_arrays(run.path)
        trace = derive_distribution_trace(arrays["foot_fsr"], run.affected_side)
        event = _event_samples(arrays, run.affected_side)
        traces[run.run_id] = trace
        events[run.run_id] = event
        for alignment in ("t0", "t1"):
            anchor = int(event[alignment])
            for metric in ALL_METRICS:
                baseline_value, baseline_count, baseline_range = pre_event_baseline_median(
                    trace.metrics[metric],
                    anchor,
                    int(baseline["start"]),
                    int(baseline["stop"]),
                    minimum_baseline,
                )
                for horizon in horizons:
                    value, valid_count, sample_range = causal_trailing_median(
                        trace.metrics[metric], anchor, horizon, width
                    )
                    loaded_start, loaded_stop = sample_range
                    affected_loaded_count = int(
                        np.count_nonzero(trace.affected_loaded[loaded_start:loaded_stop])
                    )
                    unaffected_loaded_count = int(
                        np.count_nonzero(trace.unaffected_loaded[loaded_start:loaded_stop])
                    )
                    bilateral_valid_count = int(
                        np.count_nonzero(trace.bilateral_valid[loaded_start:loaded_stop])
                    )
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
                            "t2_sample": "" if event["t2"] is None else event["t2"],
                            "alignment": alignment,
                            "horizon_ms": horizon,
                            "endpoint_sample": anchor + horizon,
                            "metric": metric,
                            "metric_kind": "absolute" if metric in ABSOLUTE_METRICS else "distribution",
                            "raw_value": "" if value is None else value,
                            "valid_trailing_samples": valid_count,
                            "affected_loaded_trailing_samples": affected_loaded_count,
                            "unaffected_loaded_trailing_samples": unaffected_loaded_count,
                            "bilateral_valid_trailing_samples": bilateral_valid_count,
                            "trailing_start_sample": sample_range[0],
                            "trailing_stop_sample_exclusive": sample_range[1],
                            "baseline_median": "" if baseline_value is None else baseline_value,
                            "baseline_valid_samples": baseline_count,
                            "baseline_start_sample": baseline_range[0],
                            "baseline_stop_sample_exclusive": baseline_range[1],
                            "baseline_delta": "" if value is None or baseline_value is None else value - baseline_value,
                        }
                    )
    separation = _separation_rows(per_run_rows, runs, ("t0", "t1"), horizons)
    rankings = _ranking_rows(separation)
    sand_controls = _load_uniform_sand_controls(manifest_rows, dataset_path)
    envelope = _benign_envelope_rows(
        runs,
        traces,
        events,
        separation,
        sand_controls,
        int(config["benign_envelope"]["persistence_ms"]),
    )
    uniform = _uniform_sand_rows(
        sand_controls, separation, per_run_rows, horizons
    )
    metadata_audit = _metadata_audit_rows(per_run_rows, rankings)
    reference_terrain = _reference_terrain_rows(manifest_rows, dataset_path)

    primary_horizons = (20, 50, 100, 150, 200)
    top_by_horizon = {
        f"{alignment}_{horizon}ms": next(
            dict(row) for row in rankings
            if row["alignment"] == alignment
            and int(row["horizon_ms"]) == horizon
            and int(row["rank"]) == 1
        )
        for alignment in ("t0", "t1")
        for horizon in primary_horizons
    }
    best_t1_100 = top_by_horizon["t1_100ms"]
    top_metric_names = []
    for row in rankings:
        if row["alignment"] == "t1" and int(row["horizon_ms"]) == 100 and row["metric"] not in top_metric_names:
            top_metric_names.append(str(row["metric"]))
    top_metric_names = top_metric_names[:5]

    _plot_group_trajectory(plots_path / "affected_quadrant_shares_t0.png", runs, traces, events, "t0", QUADRANT_SHARE_METRICS, "Affected-foot quadrant shares aligned to patch contact t0")
    _plot_group_trajectory(plots_path / "affected_quadrant_shares_t1.png", runs, traces, events, "t1", QUADRANT_SHARE_METRICS, "Affected-foot quadrant shares aligned to physical Sink t1")
    _plot_group_trajectory(plots_path / "bilateral_asymmetry.png", runs, traces, events, "t1", ("affected_load_share", "bilateral_asymmetry", "signed_bilateral_shift"), "Bilateral load distribution aligned to t1")
    _plot_group_trajectory(plots_path / "front_medial_ratios.png", runs, traces, events, "t1", ("front_ratio", "medial_ratio"), "Affected-foot front and anatomical medial ratios")
    _plot_group_trajectory(plots_path / "cop_proxy.png", runs, traces, events, "t1", ("cop_x_proxy", "cop_y_proxy", "cop_radius_proxy"), "Affected-foot normalized load-center proxy")
    _plot_heatmap(plots_path / "horizon_separation_heatmap.png", separation, primary_horizons)
    _plot_progression(plots_path / "severity_progression.png", per_run_rows, best_t1_100)
    _plot_uniform_sand(plots_path / "uniform_sand_comparison.png", uniform, top_metric_names)

    _write_csv(artifact_path / "per_run_metrics.csv", per_run_rows)
    _write_csv(artifact_path / "horizon_separation.csv", separation)
    _write_csv(artifact_path / "feature_ranking.csv", rankings)
    _write_csv(artifact_path / "benign_envelope.csv", envelope)
    _write_csv(artifact_path / "uniform_sand_separation.csv", uniform)
    _write_csv(artifact_path / "metadata_audit.csv", metadata_audit)
    _write_csv(artifact_path / "reference_terrain.csv", reference_terrain)
    summary = {
        "experiment_id": config["experiment"]["id"],
        "analysis_only": True,
        "simulation_executed": False,
        "training_executed": False,
        "dataset_id": config["dataset"]["id"],
        "manifest_sha256": sha256_file(manifest_path),
        "source_commit": _git_commit(repository_root),
        "run_counts": {
            "mild_benign": 2,
            "moderate_benign": 2,
            "severe_hazardous": 9,
            "uniform_sand_secondary": 4,
            "concrete_reference": 4,
            "marble_reference": 4,
        },
        "statistical_unit": "one_physical_run",
        "horizons_ms": horizons,
        "trailing_summary_ms": width,
        "baseline_ms": baseline,
        "low_load_threshold_n": LOAD_OFF_N,
        "top_by_horizon": top_by_horizon,
        "top_t1_100_metric_names": top_metric_names,
        "artifact_files": [
            "per_run_metrics.csv", "horizon_separation.csv", "feature_ranking.csv",
            "benign_envelope.csv", "uniform_sand_separation.csv", "metadata_audit.csv",
            "reference_terrain.csv", "plots/",
        ],
        "interpretation_guardrails": [
            "ratios_and_cop_are_analysis_only_not_runtime_features",
            "cop_is_a_normalized_quadrant_load_center_proxy_not_physical_continuous_cop",
            "perfect_pilot_separation_is_only_PILOT_SEPARATION_CANDIDATE",
            "no_threshold_or_sensor_architecture_is_frozen",
        ],
    }
    _write_json(artifact_path / "summary.json", summary)
    progress("FSR load-distribution analysis complete; no simulation or training executed")
    return artifact_path, summary
