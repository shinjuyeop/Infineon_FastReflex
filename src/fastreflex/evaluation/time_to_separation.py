"""Read-only causal replay of frozen classifiers around physical hazard events."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib
import numpy as np
import torch
import yaml

from fastreflex.dataset.collector import validate_dataset
from fastreflex.dataset.loader import (
    CLASS_NAMES,
    ManifestRecord,
    Normalizer,
    load_manifest,
    sha256_file,
    validate_split,
)
from fastreflex.training.trainer import load_checkpoint


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


VALID_OUTCOMES = ("BENIGN", "SLIP", "SINK")
HORIZON_CLASSES = {"SLIP": 1, "SINK": 2}


@dataclass(frozen=True)
class ReplayTrace:
    """Full-run causal inference indexed by source endpoint sample."""

    endpoint_samples: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray


@dataclass(frozen=True)
class EventSamples:
    """Physical event references and the last exclusive evidence sample."""

    t0: int | None
    t1: int | None
    t2: int | None
    t3: int
    t3_source: str


def causal_window_indices(
    sample_count: int, window_samples: int, stride_samples: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint-aligned indices whose last element is never in the future."""
    if sample_count < window_samples:
        raise ValueError("trace is shorter than the causal window")
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window and stride must be positive")
    endpoints = np.arange(
        window_samples - 1, sample_count, stride_samples, dtype=np.int64
    )
    offsets = np.arange(-window_samples + 1, 1, dtype=np.int64)
    return endpoints, endpoints[:, None] + offsets[None, :]


def replay_causal(
    model: torch.nn.Module,
    imu: np.ndarray,
    normalizer: Normalizer,
    window_samples: int,
    stride_samples: int = 1,
    batch_size: int = 512,
) -> ReplayTrace:
    """Replay one full raw trace using only history available at each endpoint."""
    imu = np.asarray(imu, dtype=np.float32)
    if imu.ndim != 2 or imu.shape[1] != 6 or not np.isfinite(imu).all():
        raise ValueError("replay input must be finite [N,6] pelvis IMU")
    endpoints, indices = causal_window_indices(
        len(imu), window_samples, stride_samples
    )
    logits_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(endpoints), batch_size):
            batch_indices = indices[start : start + batch_size]
            windows = normalizer.transform(imu[batch_indices])
            logits_parts.append(model(torch.from_numpy(windows)).cpu().numpy())
    logits = np.concatenate(logits_parts).astype(np.float32, copy=False)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = (exponent / exponent.sum(axis=1, keepdims=True)).astype(
        np.float32, copy=False
    )
    predictions = logits.argmax(axis=1).astype(np.int8, copy=False)
    return ReplayTrace(endpoints, logits, probabilities, predictions)


def first_matching_endpoint(
    predictions: np.ndarray,
    endpoints: np.ndarray,
    target_class: int,
    start_sample: int,
    stop_sample: int,
) -> int | None:
    mask = (
        (endpoints >= start_sample)
        & (endpoints < stop_sample)
        & (predictions == target_class)
    )
    matches = endpoints[mask]
    return None if not len(matches) else int(matches[0])


def first_sustained_endpoint(
    predictions: np.ndarray,
    endpoints: np.ndarray,
    target_class: int,
    start_sample: int,
    stop_sample: int,
    persistence_samples: int,
) -> int | None:
    """Return the online confirmation endpoint completing a consecutive run."""
    if persistence_samples <= 0:
        raise ValueError("persistence must be positive")
    run_length = 0
    previous_endpoint: int | None = None
    for prediction, endpoint_value in zip(predictions, endpoints):
        endpoint = int(endpoint_value)
        if endpoint < start_sample or endpoint >= stop_sample:
            run_length = 0
            previous_endpoint = None
            continue
        consecutive = previous_endpoint is not None and endpoint == previous_endpoint + 1
        if int(prediction) == target_class:
            run_length = run_length + 1 if consecutive else 1
            if run_length >= persistence_samples:
                return endpoint
        else:
            run_length = 0
        previous_endpoint = endpoint
    return None


def _first_sustained_hazard(
    predictions: np.ndarray,
    endpoints: np.ndarray,
    start_sample: int,
    stop_sample: int,
    persistence_samples: int,
) -> int | None:
    hazard = np.where(predictions != 0, 1, 0).astype(np.int8)
    return first_sustained_endpoint(
        hazard,
        endpoints,
        1,
        start_sample,
        stop_sample,
        persistence_samples,
    )


def audit_false_positives(
    trace: ReplayTrace,
    start_sample: int,
    stop_sample: int,
    persistence_samples: int,
) -> dict[str, object]:
    mask = (trace.endpoint_samples >= start_sample) & (
        trace.endpoint_samples < stop_sample
    )
    predictions = trace.predictions[mask]
    endpoints = trace.endpoint_samples[mask]
    slip_windows = int(np.sum(predictions == 1))
    sink_windows = int(np.sum(predictions == 2))
    sustained_slip = first_sustained_endpoint(
        predictions, endpoints, 1, start_sample, stop_sample, persistence_samples
    )
    sustained_sink = first_sustained_endpoint(
        predictions, endpoints, 2, start_sample, stop_sample, persistence_samples
    )
    sustained_any = _first_sustained_hazard(
        predictions, endpoints, start_sample, stop_sample, persistence_samples
    )
    return {
        "audited_endpoint_count": int(len(predictions)),
        "hazard_window_count": slip_windows + sink_windows,
        "slip_window_count": slip_windows,
        "sink_window_count": sink_windows,
        "any_hazard_window": bool(slip_windows + sink_windows),
        "sustained_any_hazard": sustained_any is not None,
        "sustained_slip": sustained_slip is not None,
        "sustained_sink": sustained_sink is not None,
        "first_sustained_any_sample": sustained_any,
        "first_sustained_slip_sample": sustained_slip,
        "first_sustained_sink_sample": sustained_sink,
    }


def horizon_detected(
    sustained_sample: int | None,
    t1_sample: int,
    t3_sample: int,
    horizon_ms: int,
) -> bool:
    return bool(
        sustained_sample is not None
        and sustained_sample < t3_sample
        and sustained_sample <= t1_sample + horizon_ms
    )


def pre_degradation_detected(
    sustained_sample: int | None,
    t0_sample: int,
    t1_sample: int,
    t2_sample: int,
    horizon_ms: int,
) -> bool | None:
    """Return None for zero-margin cases that cannot support this question."""
    if t2_sample <= t1_sample:
        return None
    return bool(
        sustained_sample is not None
        and sustained_sample >= t0_sample
        and sustained_sample < t2_sample
        and sustained_sample <= t1_sample + horizon_ms
    )


def _first_nonnegative(values: np.ndarray) -> int | None:
    valid = np.asarray(values, dtype=np.int64)
    valid = valid[valid >= 0]
    return None if not len(valid) else int(valid.min())


def extract_event_samples(
    arrays: Mapping[str, np.ndarray], observed_outcome: str
) -> EventSamples:
    sample_count = int(len(arrays["pelvis_imu"]))
    censor = int(arrays["first_censor_sample"])
    t3 = sample_count if censor < 0 else censor
    t3_source = "run_end" if censor < 0 else "censor"
    if observed_outcome == "BENIGN":
        return EventSamples(None, None, None, t3, t3_source)
    t0 = _first_nonnegative(arrays["first_patch_contact_sample_per_foot"])
    if observed_outcome == "SLIP":
        t1 = int(arrays["first_any_slip_onset_sample"])
        if t0 is None or t1 < 0 or not t0 <= t1 < t3:
            raise ValueError("invalid SLIP t0/t1/t3 ordering")
        return EventSamples(t0, t1, None, t3, t3_source)
    if observed_outcome == "SINK":
        t1 = _first_nonnegative(
            arrays["first_sink_physical_onset_sample_per_foot"]
        )
        t2 = int(arrays["first_sink_degradation_onset_sample"])
        if t0 is None or t1 is None or t2 < 0 or not t0 <= t1 <= t2 < t3:
            raise ValueError("invalid SINK t0/t1/t2/t3 ordering")
        return EventSamples(t0, t1, t2, t3, t3_source)
    raise ValueError(f"unsupported replay outcome: {observed_outcome}")


def _resolve_repository_path(
    repository_root: Path, value: str, field: str
) -> Path:
    path = (repository_root / value).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{field} must remain inside repository") from exc
    return path


def _verify_sha(path: Path, expected: str, field: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{field} SHA-256 mismatch")


def load_and_verify_replay_contract(
    config_path: Path, repository_root: Path
) -> tuple[dict[str, object], dict[str, Path]]:
    """Load config and verify every frozen first-PoC input before replay."""
    repository_root = repository_root.resolve()
    with config_path.resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["experiment"]["id"] != "TIME_TO_SEPARATION":
        raise ValueError("unsupported evaluation experiment")
    paths = {
        "dataset": _resolve_repository_path(
            repository_root, config["dataset"]["path"], "dataset.path"
        ),
        "split": _resolve_repository_path(
            repository_root, config["first_poc"]["split"]["path"], "split.path"
        ),
        "normalization": _resolve_repository_path(
            repository_root,
            config["first_poc"]["normalization"]["path"],
            "normalization.path",
        ),
        "selection_metrics": _resolve_repository_path(
            repository_root,
            config["first_poc"]["selection_metrics"]["path"],
            "selection_metrics.path",
        ),
        "artifact": _resolve_repository_path(
            repository_root, config["artifacts"]["path"], "artifacts.path"
        ),
    }
    _verify_sha(
        paths["split"], config["first_poc"]["split"]["sha256"], "split"
    )
    _verify_sha(
        paths["normalization"],
        config["first_poc"]["normalization"]["sha256"],
        "normalization",
    )
    _verify_sha(
        paths["selection_metrics"],
        config["first_poc"]["selection_metrics"]["sha256"],
        "selection metrics",
    )
    for checkpoint in config["primary_model"]["checkpoints"]:
        seed = int(checkpoint["seed"])
        path = _resolve_repository_path(
            repository_root, checkpoint["path"], f"checkpoint[{seed}].path"
        )
        _verify_sha(path, checkpoint["sha256"], f"checkpoint[{seed}]")
        paths[f"checkpoint_{seed}"] = path
    return config, paths


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_run_arrays(record: ManifestRecord) -> dict[str, np.ndarray]:
    required = (
        "pelvis_imu",
        "first_patch_contact_sample_per_foot",
        "first_any_slip_onset_sample",
        "first_sink_physical_onset_sample_per_foot",
        "first_sink_degradation_onset_sample",
        "first_censor_sample",
    )
    with np.load(record.path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]) for name in required}


def _normalizer_from_document(document: Mapping[str, object]) -> Normalizer:
    if document["method"] != "per_channel_zscore":
        raise ValueError("unsupported frozen normalization")
    return Normalizer(
        mean=np.asarray(document["mean"], dtype=np.float32),
        std=np.asarray(document["std"], dtype=np.float32),
        sample_count=int(document["sample_count"]),
        fit_run_ids=tuple(document["fit_run_ids"]),
        epsilon=float(document["epsilon"]),
    )


def _sample_to_time_ms(sample: int | None) -> int | None:
    return None if sample is None else sample + 1


def _probability_at(trace: ReplayTrace, sample: int | None) -> list[float] | None:
    if sample is None:
        return None
    position = np.searchsorted(trace.endpoint_samples, sample)
    if position >= len(trace.endpoint_samples) or trace.endpoint_samples[position] != sample:
        return None
    return trace.probabilities[position].tolist()


def _prediction_fractions(
    trace: ReplayTrace, start_sample: int, stop_sample: int
) -> list[float] | None:
    mask = (trace.endpoint_samples >= start_sample) & (
        trace.endpoint_samples < stop_sample
    )
    if not mask.any():
        return None
    counts = np.bincount(trace.predictions[mask], minlength=3).astype(np.float64)
    return (counts / counts.sum()).tolist()


def analyze_positive_run(
    run_id: str,
    split_name: str,
    outcome: str,
    seed: int,
    trace: ReplayTrace,
    events: EventSamples,
    persistence_samples: int,
) -> dict[str, object]:
    target = HORIZON_CLASSES[outcome]
    assert events.t0 is not None and events.t1 is not None
    first_correct = first_matching_endpoint(
        trace.predictions,
        trace.endpoint_samples,
        target,
        events.t0,
        events.t3,
    )
    first_sustained = first_sustained_endpoint(
        trace.predictions,
        trace.endpoint_samples,
        target,
        events.t0,
        events.t3,
        persistence_samples,
    )
    pre_event = audit_false_positives(
        trace, int(trace.endpoint_samples[0]), events.t0, persistence_samples
    )
    row: dict[str, object] = {
        "run_id": run_id,
        "split": split_name,
        "outcome": outcome,
        "seed": seed,
        "t0_sample": events.t0,
        "t0_ms": _sample_to_time_ms(events.t0),
        "t1_sample": events.t1,
        "t1_ms": _sample_to_time_ms(events.t1),
        "t2_sample": events.t2,
        "t2_ms": _sample_to_time_ms(events.t2),
        "t3_sample_exclusive": events.t3,
        "t3_ms_exclusive": _sample_to_time_ms(events.t3),
        "t3_source": events.t3_source,
        "first_correct_sample": first_correct,
        "first_correct_ms": _sample_to_time_ms(first_correct),
        "first_sustained_correct_sample": first_sustained,
        "first_sustained_correct_ms": _sample_to_time_ms(first_sustained),
        "first_sustained_start_sample": (
            None if first_sustained is None else first_sustained - persistence_samples + 1
        ),
        "latency_from_t0_ms": (
            None if first_sustained is None else first_sustained - events.t0
        ),
        "latency_from_t1_ms": (
            None if first_sustained is None else first_sustained - events.t1
        ),
        "margin_to_t2_ms": (
            None
            if first_sustained is None or events.t2 is None
            else events.t2 - first_sustained
        ),
        "margin_to_t3_ms": (
            None if first_sustained is None else events.t3 - first_sustained
        ),
        "zero_margin_sink": bool(
            outcome == "SINK" and events.t1 == events.t2
        ),
        "pre_t0_any_hazard_window": pre_event["any_hazard_window"],
        "pre_t0_sustained_any_hazard": pre_event["sustained_any_hazard"],
        "pre_t0_sustained_slip": pre_event["sustained_slip"],
        "pre_t0_sustained_sink": pre_event["sustained_sink"],
        "pre_t0_hazard_window_count": pre_event["hazard_window_count"],
        "probability_at_t0": _probability_at(trace, events.t0),
        "probability_at_t1": _probability_at(trace, events.t1),
        "probability_at_t2": _probability_at(trace, events.t2),
        "prediction_fraction_t0_t1": _prediction_fractions(
            trace, events.t0, events.t1
        ),
        "prediction_fraction_t1_t2": (
            None
            if events.t2 is None
            else _prediction_fractions(trace, events.t1, events.t2)
        ),
        "prediction_fraction_t1_t3": _prediction_fractions(
            trace, events.t1, events.t3
        ),
        "prediction_fraction_t2_t3": (
            None
            if events.t2 is None
            else _prediction_fractions(trace, events.t2, events.t3)
        ),
    }
    return row


def _scenario_group(run_id: str) -> str:
    if run_id.startswith("normal_concrete_"):
        return "concrete"
    if run_id.startswith("normal_marble_"):
        return "marble"
    if run_id.startswith("normal_sand_"):
        return "uniform_sand"
    if "_mild_" in run_id:
        return "benign_sink_mild"
    if "_moderate_" in run_id:
        return "benign_sink_moderate"
    raise ValueError(f"unknown BENIGN scenario group: {run_id}")


def _benign_row(
    run_id: str,
    split_name: str,
    seed: int,
    trace: ReplayTrace,
    events: EventSamples,
    persistence_samples: int,
) -> dict[str, object]:
    audit = audit_false_positives(
        trace,
        int(trace.endpoint_samples[0]),
        events.t3,
        persistence_samples,
    )
    return {
        "run_id": run_id,
        "split": split_name,
        "scenario_group": _scenario_group(run_id),
        "seed": seed,
        "t3_sample_exclusive": events.t3,
        "t3_source": events.t3_source,
        **audit,
    }


def _split_lookup(split_document: Mapping[str, object]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split_name, run_ids in split_document["run_ids"].items():
        for run_id in run_ids:
            if run_id in lookup:
                raise ValueError("first-PoC split is not run-disjoint")
            lookup[run_id] = split_name
    return lookup


def _save_trajectory(
    path: Path,
    run_ids: Sequence[str],
    traces: Mapping[str, ReplayTrace],
) -> None:
    offsets = [0]
    for run_id in run_ids:
        offsets.append(offsets[-1] + len(traces[run_id].endpoint_samples))
    np.savez_compressed(
        path,
        run_ids=np.asarray(run_ids),
        offsets=np.asarray(offsets, dtype=np.int64),
        endpoint_samples=np.concatenate(
            [traces[run_id].endpoint_samples for run_id in run_ids]
        ),
        predictions=np.concatenate(
            [traces[run_id].predictions for run_id in run_ids]
        ),
        probabilities=np.concatenate(
            [traces[run_id].probabilities for run_id in run_ids]
        ),
        logits=np.concatenate([traces[run_id].logits for run_id in run_ids]),
    )


def _horizon_rows(
    positive_rows: Sequence[Mapping[str, object]],
    horizons: Sequence[int],
    split_names: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seeds = sorted({int(row["seed"]) for row in positive_rows})
    for outcome in ("SLIP", "SINK"):
        for scope in ("all", *split_names):
            for seed in seeds:
                selected = [
                    row
                    for row in positive_rows
                    if row["outcome"] == outcome
                    and int(row["seed"]) == seed
                    and (scope == "all" or row["split"] == scope)
                ]
                if not selected:
                    continue
                for horizon in horizons:
                    detected = [
                        row
                        for row in selected
                        if horizon_detected(
                            row["first_sustained_correct_sample"],
                            int(row["t1_sample"]),
                            int(row["t3_sample_exclusive"]),
                            horizon,
                        )
                    ]
                    latencies = [
                        float(row["latency_from_t1_ms"]) for row in detected
                    ]
                    row_out: dict[str, object] = {
                        "outcome": outcome,
                        "scope": scope,
                        "seed": seed,
                        "horizon_ms": horizon,
                        "event_count": len(selected),
                        "detected_count": len(detected),
                        "event_recall": len(detected) / len(selected),
                        "median_sustained_latency_ms": (
                            None if not latencies else float(np.median(latencies))
                        ),
                        "pre_t0_fp_run_count": sum(
                            bool(row["pre_t0_sustained_any_hazard"])
                            for row in selected
                        ),
                    }
                    if outcome == "SLIP":
                        margins = [
                            float(row["margin_to_t3_ms"]) for row in detected
                        ]
                        row_out.update(
                            {
                                "pre_degradation_eligible_count": "",
                                "pre_degradation_detected_count": "",
                                "pre_degradation_recall": "",
                                "median_margin_to_t2_ms": "",
                                "median_margin_to_t3_ms": (
                                    None if not margins else float(np.median(margins))
                                ),
                            }
                        )
                    else:
                        eligible = [
                            row
                            for row in selected
                            if int(row["t2_sample"]) > int(row["t1_sample"])
                        ]
                        pre_detected = [
                            row
                            for row in eligible
                            if pre_degradation_detected(
                                row["first_sustained_correct_sample"],
                                int(row["t0_sample"]),
                                int(row["t1_sample"]),
                                int(row["t2_sample"]),
                                horizon,
                            )
                        ]
                        margins = [
                            float(row["margin_to_t2_ms"]) for row in pre_detected
                        ]
                        row_out.update(
                            {
                                "pre_degradation_eligible_count": len(eligible),
                                "pre_degradation_detected_count": len(pre_detected),
                                "pre_degradation_recall": (
                                    None
                                    if not eligible
                                    else len(pre_detected) / len(eligible)
                                ),
                                "median_margin_to_t2_ms": (
                                    None if not margins else float(np.median(margins))
                                ),
                                "median_margin_to_t3_ms": "",
                            }
                        )
                    rows.append(row_out)
    return rows


def _aggregate_horizons(
    horizon_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    result: dict[str, object] = {}
    outcomes = sorted({str(row["outcome"]) for row in horizon_rows})
    for outcome in outcomes:
        result[outcome] = {}
        scopes = sorted(
            {str(row["scope"]) for row in horizon_rows if row["outcome"] == outcome}
        )
        for scope in scopes:
            result[outcome][scope] = {}
            horizons = sorted(
                {
                    int(row["horizon_ms"])
                    for row in horizon_rows
                    if row["outcome"] == outcome and row["scope"] == scope
                }
            )
            for horizon in horizons:
                selected = [
                    row
                    for row in horizon_rows
                    if row["outcome"] == outcome
                    and row["scope"] == scope
                    and int(row["horizon_ms"]) == horizon
                ]
                recalls = np.asarray(
                    [float(row["event_recall"]) for row in selected]
                )
                worst_index = int(np.argmin(recalls))
                entry: dict[str, object] = {
                    "event_recall_mean": float(recalls.mean()),
                    "event_recall_std": float(recalls.std()),
                    "event_recall_by_seed": {
                        str(row["seed"]): float(row["event_recall"])
                        for row in selected
                    },
                    "worst_seed": int(selected[worst_index]["seed"]),
                    "median_sustained_latency_ms_by_seed": {
                        str(row["seed"]): row["median_sustained_latency_ms"]
                        for row in selected
                    },
                    "pre_t0_fp_runs_by_seed": {
                        str(row["seed"]): int(row["pre_t0_fp_run_count"])
                        for row in selected
                    },
                }
                if outcome == "SINK":
                    eligible_recalls = [
                        float(row["pre_degradation_recall"])
                        for row in selected
                        if row["pre_degradation_recall"] is not None
                    ]
                    entry.update(
                        {
                            "pre_degradation_recall_mean": (
                                None
                                if not eligible_recalls
                                else float(np.mean(eligible_recalls))
                            ),
                            "pre_degradation_recall_std": (
                                None
                                if not eligible_recalls
                                else float(np.std(eligible_recalls))
                            ),
                            "pre_degradation_recall_by_seed": {
                                str(row["seed"]): row["pre_degradation_recall"]
                                for row in selected
                            },
                            "median_margin_to_t2_ms_by_seed": {
                                str(row["seed"]): row["median_margin_to_t2_ms"]
                                for row in selected
                            },
                        }
                    )
                else:
                    entry["median_margin_to_t3_ms_by_seed"] = {
                        str(row["seed"]): row["median_margin_to_t3_ms"]
                        for row in selected
                    }
                result[outcome][scope][str(horizon)] = entry
    return result


def _aggregate_benign(
    rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    result: dict[str, object] = {}
    groups = ("all", *sorted({str(row["scenario_group"]) for row in rows}))
    seeds = sorted({int(row["seed"]) for row in rows})
    for group in groups:
        per_seed: dict[str, object] = {}
        for seed in seeds:
            selected = [
                row
                for row in rows
                if int(row["seed"]) == seed
                and (group == "all" or row["scenario_group"] == group)
            ]
            if not selected:
                continue
            per_seed[str(seed)] = {
                "run_count": len(selected),
                "any_hazard_run_fp_count": sum(
                    bool(row["any_hazard_window"]) for row in selected
                ),
                "sustained_any_hazard_run_fp_count": sum(
                    bool(row["sustained_any_hazard"]) for row in selected
                ),
                "sustained_slip_run_fp_count": sum(
                    bool(row["sustained_slip"]) for row in selected
                ),
                "sustained_sink_run_fp_count": sum(
                    bool(row["sustained_sink"]) for row in selected
                ),
                "hazard_window_count": sum(
                    int(row["hazard_window_count"]) for row in selected
                ),
            }
            values = per_seed[str(seed)]
            values["any_hazard_run_fp_rate"] = (
                values["any_hazard_run_fp_count"] / values["run_count"]
            )
            values["sustained_any_hazard_run_fp_rate"] = (
                values["sustained_any_hazard_run_fp_count"] / values["run_count"]
            )
            values["sustained_slip_run_fp_rate"] = (
                values["sustained_slip_run_fp_count"] / values["run_count"]
            )
            values["sustained_sink_run_fp_rate"] = (
                values["sustained_sink_run_fp_count"] / values["run_count"]
            )
        sustained = np.asarray(
            [
                value["sustained_any_hazard_run_fp_count"]
                for value in per_seed.values()
            ],
            dtype=np.float64,
        )
        worst_seed = max(
            per_seed,
            key=lambda seed: per_seed[seed]["sustained_any_hazard_run_fp_count"],
        )
        result[group] = {
            "by_seed": per_seed,
            "sustained_any_hazard_run_fp_mean": float(sustained.mean()),
            "sustained_any_hazard_run_fp_std": float(sustained.std()),
            "worst_seed": int(worst_seed),
        }
    return result


def _aggregate_eventual_detection(
    rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    result: dict[str, object] = {}
    seeds = sorted({int(row["seed"]) for row in rows})
    for outcome in ("SLIP", "SINK"):
        by_seed: dict[str, object] = {}
        for seed in seeds:
            selected = [
                row
                for row in rows
                if row["outcome"] == outcome and int(row["seed"]) == seed
            ]
            detected = [
                row
                for row in selected
                if row["first_sustained_correct_sample"] is not None
                and int(row["first_sustained_correct_sample"])
                < int(row["t3_sample_exclusive"])
            ]
            latencies = [float(row["latency_from_t1_ms"]) for row in detected]
            margins_to_t3 = [float(row["margin_to_t3_ms"]) for row in detected]
            censor_margins = [
                float(row["margin_to_t3_ms"])
                for row in detected
                if row["t3_source"] == "censor"
            ]
            entry: dict[str, object] = {
                "event_count": len(selected),
                "detected_before_t3_count": len(detected),
                "detected_before_t3_recall": len(detected) / len(selected),
                "early_t0_t1_sustained_count": sum(
                    int(row["first_sustained_correct_sample"])
                    < int(row["t1_sample"])
                    for row in detected
                ),
                "median_latency_from_t1_ms": float(np.median(latencies)),
                "median_margin_to_t3_ms": float(np.median(margins_to_t3)),
                "median_margin_to_censor_ms": (
                    None
                    if not censor_margins
                    else float(np.median(censor_margins))
                ),
            }
            if outcome == "SINK":
                eligible = [
                    row
                    for row in selected
                    if int(row["t2_sample"]) > int(row["t1_sample"])
                ]
                before_t2 = [
                    row
                    for row in eligible
                    if row["first_sustained_correct_sample"] is not None
                    and int(row["first_sustained_correct_sample"])
                    < int(row["t2_sample"])
                ]
                in_t1_t2 = [
                    row
                    for row in before_t2
                    if int(row["first_sustained_correct_sample"])
                    >= int(row["t1_sample"])
                ]
                entry.update(
                    {
                        "positive_margin_event_count": len(eligible),
                        "detected_before_t2_count": len(before_t2),
                        "detected_before_t2_recall": len(before_t2) / len(eligible),
                        "detected_in_t1_t2_count": len(in_t1_t2),
                        "median_positive_margin_latency_from_t1_ms": float(
                            np.median(
                                [
                                    float(row["latency_from_t1_ms"])
                                    for row in before_t2
                                ]
                            )
                        ),
                        "median_margin_before_t2_ms": float(
                            np.median(
                                [float(row["margin_to_t2_ms"]) for row in before_t2]
                            )
                        ),
                    }
                )
            by_seed[str(seed)] = entry
        result[outcome] = {"by_seed": by_seed}
    return result


def _aggregate_positive_pre_event(
    rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for outcome in ("SLIP", "SINK"):
        result[outcome] = {}
        seeds = sorted(
            {int(row["seed"]) for row in rows if row["outcome"] == outcome}
        )
        target_field = (
            "pre_t0_sustained_slip" if outcome == "SLIP" else "pre_t0_sustained_sink"
        )
        for seed in seeds:
            selected = [
                row
                for row in rows
                if row["outcome"] == outcome and int(row["seed"]) == seed
            ]
            result[outcome][str(seed)] = {
                "run_count": len(selected),
                "any_hazard_window_run_count": sum(
                    bool(row["pre_t0_any_hazard_window"]) for row in selected
                ),
                "sustained_any_hazard_run_count": sum(
                    bool(row["pre_t0_sustained_any_hazard"]) for row in selected
                ),
                "sustained_target_class_run_count": sum(
                    bool(row[target_field]) for row in selected
                ),
            }
    return result


def _plot_probability_trajectories(
    path: Path,
    run_ids: Sequence[str],
    traces_by_run: Mapping[str, Sequence[ReplayTrace]],
    events_by_run: Mapping[str, EventSamples],
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15, 12))
    colors = ("tab:green", "tab:orange", "tab:red")
    for axis, run_id in zip(axes.flat, run_ids):
        traces = traces_by_run[run_id]
        endpoints = traces[0].endpoint_samples
        probabilities = np.stack([trace.probabilities for trace in traces])
        mean = probabilities.mean(axis=0)
        low = probabilities.min(axis=0)
        high = probabilities.max(axis=0)
        events = events_by_run[run_id]
        anchor = 0 if events.t1 is None else events.t1
        x = endpoints - anchor
        for class_id, name in enumerate(CLASS_NAMES):
            axis.plot(x, mean[:, class_id], color=colors[class_id], label=name)
            axis.fill_between(
                x,
                low[:, class_id],
                high[:, class_id],
                color=colors[class_id],
                alpha=0.12,
            )
        for label, sample, style in (
            ("t0", events.t0, ":"),
            ("t1", events.t1, "--"),
            ("t2", events.t2, "-."),
            ("t3", events.t3 if events.t3_source == "censor" else None, ":"),
        ):
            if sample is not None:
                axis.axvline(sample - anchor, color="black", linestyle=style, alpha=0.6)
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(run_id)
        axis.set_xlabel("time from t1 (ms)" if events.t1 is not None else "sample (ms)")
        axis.set_ylabel("seed mean probability")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(loc="best", ncol=3)
    figure.suptitle("Frozen MLP 100 ms causal probability replay (mean and seed range)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_imu_probability(
    path: Path,
    run_id: str,
    imu: np.ndarray,
    traces: Sequence[ReplayTrace],
    events: EventSamples,
) -> None:
    if events.t1 is None:
        raise ValueError("IMU probability diagnostic requires t1")
    start = max(0, events.t1 - 100)
    stop = min(len(imu), events.t1 + 201)
    sample_x = np.arange(start, stop) - events.t1
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for channel in range(3):
        axes[0].plot(
            sample_x, imu[start:stop, channel], label=f"accel_{'xyz'[channel]}"
        )
    for channel in range(3, 6):
        axes[1].plot(
            sample_x,
            imu[start:stop, channel],
            label=f"gyro_{'xyz'[channel - 3]}",
        )
    endpoints = traces[0].endpoint_samples
    mask = (endpoints >= start) & (endpoints < stop)
    probabilities = np.stack([trace.probabilities[mask] for trace in traces]).mean(axis=0)
    for class_id, name in enumerate(CLASS_NAMES):
        axes[2].plot(endpoints[mask] - events.t1, probabilities[:, class_id], label=name)
    for axis in axes:
        axis.axvline(0, color="black", linestyle="--", label="t1")
        if events.t2 is not None and start <= events.t2 < stop:
            axis.axvline(events.t2 - events.t1, color="purple", linestyle=":", label="t2")
        axis.grid(alpha=0.2)
        axis.legend(loc="best", ncol=4)
    axes[0].set_ylabel("acceleration (m/s²)")
    axes[1].set_ylabel("angular rate (rad/s)")
    axes[2].set_ylabel("probability")
    axes[2].set_xlabel("time from t1 (ms)")
    figure.suptitle(f"Raw pelvis IMU6 and frozen-model confidence: {run_id}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_time_to_separation(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Replay three frozen selected-model seeds without any training or tuning."""
    repository_root = repository_root.resolve()
    config, paths = load_and_verify_replay_contract(config_path, repository_root)
    artifact_path = paths["artifact"]
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite replay artifacts: {artifact_path}")
    dataset_summary = validate_dataset(paths["dataset"])
    if dataset_summary["manifest_sha256"] != config["dataset"]["manifest_sha256"]:
        raise ValueError("Pilot manifest SHA-256 changed")
    split_document = _read_json(paths["split"])
    normalization_document = _read_json(paths["normalization"])
    selection_document = _read_json(paths["selection_metrics"])
    if selection_document["selection"]["candidate_id"] != "mlp_100ms":
        raise ValueError("first-PoC selected model changed")
    split = {
        name: tuple(split_document["run_ids"][name])
        for name in ("train", "validation", "holdout")
    }
    records = load_manifest(paths["dataset"])
    validate_split(records, split)
    split_lookup = _split_lookup(split_document)
    if tuple(normalization_document["fit_run_ids"]) != split["train"]:
        raise ValueError("first-PoC normalizer train membership changed")
    normalizer = _normalizer_from_document(normalization_document)
    window_samples = int(config["primary_model"]["window_samples"])
    stride_samples = int(config["replay"]["stride_ms"])
    persistence = int(config["replay"]["sustained_correct_ms"])
    horizons = [int(value) for value in config["replay"]["horizons_ms"]]
    extended_horizons = [
        int(value) for value in config["replay"]["descriptive_extended_horizons_ms"]
    ]
    batch_size = int(config["replay"]["batch_size"])
    run_ids = tuple(
        run_id
        for split_name in ("train", "validation", "holdout")
        for run_id in split[split_name]
    )
    events_by_run: dict[str, EventSamples] = {}
    imu_by_representative: dict[str, np.ndarray] = {}
    plot_run_ids = set(config["plots"]["probability_runs"])
    imu_plot_run_ids = set(config["plots"]["imu_probability_runs"])
    representative_traces: dict[str, list[ReplayTrace]] = {
        run_id: [] for run_id in plot_run_ids | imu_plot_run_ids
    }
    positive_rows: list[dict[str, object]] = []
    benign_rows: list[dict[str, object]] = []
    artifact_path.mkdir(parents=True, exist_ok=True)
    (artifact_path / "trajectories").mkdir()
    (artifact_path / "plots").mkdir()
    for checkpoint in config["primary_model"]["checkpoints"]:
        seed = int(checkpoint["seed"])
        model, metadata = load_checkpoint(paths[f"checkpoint_{seed}"])
        if (
            metadata["family"] != "mlp"
            or int(metadata["window_samples"]) != window_samples
            or int(metadata["seed"]) != seed
        ):
            raise ValueError(f"checkpoint metadata changed for seed {seed}")
        progress(f"replaying frozen MLP 100 ms seed {seed}")
        traces: dict[str, ReplayTrace] = {}
        for run_id in run_ids:
            record = records[run_id]
            arrays = _load_run_arrays(record)
            events = extract_event_samples(arrays, record.observed_outcome)
            events_by_run[run_id] = events
            imu = np.asarray(arrays["pelvis_imu"], dtype=np.float32)
            trace = replay_causal(
                model,
                imu,
                normalizer,
                window_samples,
                stride_samples=stride_samples,
                batch_size=batch_size,
            )
            traces[run_id] = trace
            if run_id in representative_traces:
                representative_traces[run_id].append(trace)
            if run_id in imu_plot_run_ids:
                imu_by_representative[run_id] = imu
            if record.observed_outcome == "BENIGN":
                benign_rows.append(
                    _benign_row(
                        run_id,
                        split_lookup[run_id],
                        seed,
                        trace,
                        events,
                        persistence,
                    )
                )
            else:
                positive_rows.append(
                    analyze_positive_run(
                        run_id,
                        split_lookup[run_id],
                        record.observed_outcome,
                        seed,
                        trace,
                        events,
                        persistence,
                    )
                )
        _save_trajectory(
            artifact_path / "trajectories" / f"mlp_100ms_seed_{seed}.npz",
            run_ids,
            traces,
        )

    all_horizons = horizons + extended_horizons
    horizon_rows = _horizon_rows(
        positive_rows, all_horizons, ("train", "validation", "holdout")
    )
    _write_csv(artifact_path / "per_run.csv", positive_rows)
    _write_csv(artifact_path / "horizon_recall.csv", horizon_rows)
    _write_csv(artifact_path / "benign_false_positive.csv", benign_rows)
    _plot_probability_trajectories(
        artifact_path / "plots" / "probability_trajectories.png",
        config["plots"]["probability_runs"],
        representative_traces,
        events_by_run,
    )
    for run_id in config["plots"]["imu_probability_runs"]:
        _plot_imu_probability(
            artifact_path / "plots" / f"{run_id}_imu_probability.png",
            run_id,
            imu_by_representative[run_id],
            representative_traces[run_id],
            events_by_run[run_id],
        )

    difficult_rows = [
        row
        for row in positive_rows
        if row["run_id"] == "sink_right_severe_s025_p035"
    ]
    metrics = {
        "experiment_id": config["experiment"]["id"],
        "dataset": dataset_summary,
        "first_poc_contract": {
            "split_sha256": config["first_poc"]["split"]["sha256"],
            "normalization_sha256": config["first_poc"]["normalization"]["sha256"],
            "selection_metrics_sha256": config["first_poc"]["selection_metrics"]["sha256"],
            "checkpoint_sha256_by_seed": {
                str(item["seed"]): item["sha256"]
                for item in config["primary_model"]["checkpoints"]
            },
        },
        "replay": {
            "model": "mlp_100ms",
            "seeds": [
                int(item["seed"])
                for item in config["primary_model"]["checkpoints"]
            ],
            "window_samples": window_samples,
            "stride_ms": stride_samples,
            "sustained_confirmation_ms": persistence,
            "prediction": "argmax",
            "valid_run_count": len(run_ids),
            "positive_event_counts": {"SLIP": 8, "SINK": 9},
            "benign_run_count": 16,
        },
        "zero_margin_sink_run_ids": sorted(
            {
                str(row["run_id"])
                for row in positive_rows
                if row["zero_margin_sink"]
            }
        ),
        "horizon_metrics": _aggregate_horizons(horizon_rows),
        "eventual_detection": _aggregate_eventual_detection(positive_rows),
        "positive_run_pre_event_false_positive": _aggregate_positive_pre_event(
            positive_rows
        ),
        "benign_false_positive": _aggregate_benign(benign_rows),
        "difficult_sink_run": difficult_rows,
    }
    _write_json(artifact_path / "metrics.json", metrics)
    load_and_verify_replay_contract(config_path, repository_root)
    final_dataset_summary = validate_dataset(paths["dataset"])
    if final_dataset_summary != dataset_summary:
        raise RuntimeError("Pilot dataset changed during replay")
    progress("read-only causal replay and artifact generation complete")
    return artifact_path, metrics
