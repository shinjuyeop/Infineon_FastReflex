"""TRAIN-only construction and HNM for the supported Unified Hazard GRU."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fastreflex.dataset.hazard import (
    HazardRun,
    canonical_sha256,
    physical_hazard_label,
    slip_event_sample,
    support_event_sample,
)
from fastreflex.dataset.loader import Normalizer, WindowSet, sha256_file
from fastreflex.evaluation.hazard import HISTORY_MS, replay_hazard_run
from fastreflex.features import extract_hazard_features, hazard_feature_schema
from fastreflex.models.baselines import parameter_count
from fastreflex.training.trainer import (
    load_checkpoint,
    save_checkpoint,
    train_model,
)


EVENT_CLASS_NAMES = ("NORMAL", "HAZARD_REFLEX_REQUIRED")
HNM_ROUNDS = 3
HNM_TOP_K_PER_RUN = 12
HNM_MINIMUM_SPACING_MS = 30
HNM_REPLAY_STRIDE_MS = 1


@dataclass(frozen=True)
class HazardCandidate:
    history_ms: int
    normalizer: Normalizer
    checkpoint_paths: tuple[Path, ...]
    record: Mapping[str, object]


def _evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    selected = np.asarray(values, dtype=np.int64)
    if count <= 0 or not len(selected):
        return np.empty(0, dtype=np.int64)
    if len(selected) <= count:
        return selected
    return selected[np.linspace(0, len(selected) - 1, count, dtype=np.int64)]


def unified_positive_endpoints(
    run: HazardRun,
    precursor: int | None,
    history_ms: int = HISTORY_MS,
    *,
    cap: int = 20,
) -> np.ndarray:
    """Frozen bounded union positives, using physical references only."""
    selected: set[int] = set()
    slip = slip_event_sample(run)
    support = support_event_sample(run)
    if slip is not None:
        selected.update(range(slip - 30, slip + 41, 5))
    if support is not None and precursor is not None:
        selected.update(int(value) for value in np.linspace(precursor, support, 5))
        selected.update(support + offset for offset in (-20, 0, 20, 40))
    elif support is not None:
        selected.update(support + offset for offset in (-20, 0, 20, 40))
    values = np.asarray(sorted(selected), dtype=np.int64)
    valid = (values >= history_ms - 1) & (values < run.censor_sample)
    if run.fall_sample_diagnostic is not None:
        valid &= values < int(run.fall_sample_diagnostic)
    return _evenly_spaced(values[valid], cap)


def unified_negative_candidates(
    run: HazardRun, precursor: int | None, history_ms: int = HISTORY_MS
) -> np.ndarray:
    """Return only no-hazard endpoints; I1-active samples are never negative."""
    last = run.censor_sample - 1
    slip = slip_event_sample(run)
    support = support_event_sample(run)
    if slip is not None:
        last = min(last, slip - 31)
    if precursor is not None:
        last = min(last, int(precursor) - 1)
    elif support is not None:
        last = min(last, support - 1)
    if run.fall_sample_diagnostic is not None:
        last = min(last, int(run.fall_sample_diagnostic) - 1)
    first = history_ms - 1
    return (
        np.arange(first, last + 1, dtype=np.int64)
        if last >= first
        else np.empty(0, dtype=np.int64)
    )


def gait_sampling_categories(run: HazardRun) -> dict[str, np.ndarray]:
    """Use contact phase only to diversify TRAIN negative sampling."""
    loaded = np.asarray(run.loaded_contact, dtype=bool)
    previous = np.vstack((np.zeros((1, 2), dtype=bool), loaded[:-1]))
    touchdown = np.any(loaded & ~previous, axis=1)
    release = np.any(~loaded & previous, axis=1)
    return {
        "touchdown_loading": np.flatnonzero(touchdown),
        "contact_release": np.flatnonzero(release),
        "left_support": np.flatnonzero(loaded[:, 0] & ~loaded[:, 1]),
        "right_support": np.flatnonzero(~loaded[:, 0] & loaded[:, 1]),
        "double_support": np.flatnonzero(np.all(loaded, axis=1)),
        "no_support": np.flatnonzero(~np.any(loaded, axis=1)),
    }


def initial_negative_endpoints(
    run: HazardRun,
    precursor: int | None,
    history_ms: int = HISTORY_MS,
    *,
    per_category: int = 12,
) -> np.ndarray:
    eligible = unified_negative_candidates(run, precursor, history_ms)
    if not len(eligible):
        return eligible
    allowed = set(int(value) for value in eligible)
    selected = [
        _evenly_spaced(
            np.asarray(
                [value for value in values if int(value) in allowed], dtype=np.int64
            ),
            per_category,
        )
        for values in gait_sampling_categories(run).values()
    ]
    return np.unique(np.concatenate(selected)) if selected else np.empty(0, np.int64)


def mine_hard_negative_endpoints(
    candidates: np.ndarray,
    probabilities: np.ndarray,
    *,
    top_k: int = HNM_TOP_K_PER_RUN,
    minimum_separation_ms: int = HNM_MINIMUM_SPACING_MS,
    excluded: Sequence[int] = (),
) -> np.ndarray:
    """Choose highest-scoring TRAIN negatives with frozen spacing and K."""
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


def fit_hazard_normalizer(
    runs: Mapping[str, HazardRun],
    run_ids: Sequence[str],
    *,
    per_run_sample_cap: int = 1000,
    standard_deviation_floor: float = 1.0e-6,
) -> Normalizer:
    """Fit feature normalization only from explicitly supplied TRAIN runs."""
    chunks: list[np.ndarray] = []
    fit_ids: list[str] = []
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        if run.split != "train":
            raise ValueError("Hazard normalizer may use TRAIN runs only")
        features = extract_hazard_features(run.features["PELVIS_IMU6"])
        eligible = np.arange(0, run.censor_sample, dtype=np.int64)
        if run.fall_sample_diagnostic is not None:
            eligible = eligible[eligible < int(run.fall_sample_diagnostic)]
        eligible = _evenly_spaced(eligible, per_run_sample_cap)
        if len(eligible):
            chunks.append(features[eligible].astype(np.float64))
            fit_ids.append(run_id)
    if not chunks:
        raise ValueError("Hazard normalizer has no TRAIN samples")
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


def build_hazard_windows(
    runs: Mapping[str, HazardRun],
    run_ids: Sequence[str],
    precursor_samples: Mapping[str, int | None],
    normalizer: Normalizer,
    *,
    per_category: int = 12,
    positive_cap: int = 20,
    extra_negative_endpoints: Mapping[str, Sequence[int]] | None = None,
) -> WindowSet:
    """Materialize causal [20,80] binary windows with provenance."""
    inputs: list[np.ndarray] = []
    targets: list[int] = []
    source_ids: list[str] = []
    endpoint_rows: list[int] = []
    extras = extra_negative_endpoints or {}
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        if run.split != "train":
            raise ValueError("Hazard training windows may use TRAIN runs only")
        features = extract_hazard_features(run.features["PELVIS_IMU6"])
        precursor = precursor_samples.get(run_id)
        positive = unified_positive_endpoints(run, precursor, cap=positive_cap)
        negative = initial_negative_endpoints(
            run, precursor, per_category=per_category
        )
        allowed = set(int(value) for value in unified_negative_candidates(run, precursor))
        if run_id in extras:
            extra = np.asarray(
                [int(value) for value in extras[run_id] if int(value) in allowed],
                dtype=np.int64,
            )
            negative = np.unique(np.concatenate((negative, extra)))
        if set(int(value) for value in positive) & set(int(value) for value in negative):
            raise RuntimeError("Hazard positive was used as a negative")
        for label, endpoints in ((1, positive), (0, negative)):
            for endpoint in endpoints:
                first = int(endpoint) - HISTORY_MS + 1
                if first < 0 or int(endpoint) >= run.censor_sample:
                    raise ValueError("Hazard window crossed a causal boundary")
                window = normalizer.transform(features[first : int(endpoint) + 1])
                if window.shape != (HISTORY_MS, 80):
                    raise RuntimeError("Hazard GRU input shape changed")
                inputs.append(window)
                targets.append(label)
                source_ids.append(run_id)
                endpoint_rows.append(int(endpoint))
    labels = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    if not inputs or np.any(counts == 0):
        raise ValueError("Hazard training windows require both classes")
    return WindowSet(
        inputs=np.stack(inputs).astype(np.float32),
        targets=labels,
        run_ids=np.asarray(source_ids, dtype=str),
        endpoint_samples=np.asarray(endpoint_rows, dtype=np.int64),
        available_by_class=(int(counts[0]), int(counts[1]), 0),
    )


def _train_monitor_partition(
    runs: Mapping[str, HazardRun],
    run_ids: Sequence[str],
    precursor_samples: Mapping[str, int | None],
) -> tuple[list[str], list[str]]:
    groups: dict[tuple[str, str, str], list[str]] = {}
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        if run.split != "train":
            raise ValueError("Hazard HNM partition may use TRAIN runs only")
        groups.setdefault(
            (
                run.source_terrain,
                run.target_terrain,
                physical_hazard_label(run, precursor_samples.get(run_id)),
            ),
            [],
        ).append(run_id)
    monitor: list[str] = []
    for values in groups.values():
        count = max(1, int(round(len(values) * 0.20)))
        monitor.extend(
            values[int(index)]
            for index in np.linspace(0, len(values) - 1, count, dtype=np.int64)
        )
    monitor_set = set(monitor)
    return (
        sorted(run_id for run_id in run_ids if run_id not in monitor_set),
        sorted(monitor_set),
    )


def _merge_endpoint_maps(
    *values: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    keys = {key for mapping in values for key in mapping}
    return {
        key: tuple(
            sorted({int(value) for mapping in values for value in mapping.get(key, ())})
        )
        for key in keys
    }


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {key: None for key in ("min", "p10", "median", "p95", "max")}
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _mine_training_round(
    runs: Mapping[str, HazardRun],
    precursor_samples: Mapping[str, int | None],
    normalizer: Normalizer,
    checkpoint_paths: Sequence[Path],
    prior: Mapping[str, Sequence[int]],
) -> tuple[dict[str, tuple[int, ...]], dict[str, object]]:
    models = [load_checkpoint(path)[0] for path in checkpoint_paths]
    result: dict[str, tuple[int, ...]] = {}
    selected_scores: list[float] = []
    for run_id, run in sorted(runs.items()):
        candidates = unified_negative_candidates(run, precursor_samples.get(run_id), 1)
        replay = replay_hazard_run(run, normalizer, models)
        common, _, replay_indices = np.intersect1d(
            candidates, replay.endpoints, return_indices=True
        )
        scores = replay.probabilities[replay_indices]
        selected = mine_hard_negative_endpoints(
            common, scores, excluded=prior.get(run_id, ())
        )
        result[run_id] = tuple(int(value) for value in selected)
        lookup = {int(endpoint): float(score) for endpoint, score in zip(common, scores)}
        selected_scores.extend(lookup[int(endpoint)] for endpoint in selected)
    return result, {
        "runs_scored": len(runs),
        "mined_windows": sum(len(value) for value in result.values()),
        "selected_probability": _distribution(selected_scores),
        "train_only": True,
        "replay_stride_ms": HNM_REPLAY_STRIDE_MS,
        "top_k_per_run": HNM_TOP_K_PER_RUN,
        "minimum_spacing_ms": HNM_MINIMUM_SPACING_MS,
        "precursor_region_never_negative": True,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def train_hazard_candidate(
    repository_root: Path,
    runs: Mapping[str, HazardRun],
    precursor_samples: Mapping[str, int | None],
    artifact_path: Path,
    training_config: Mapping[str, object],
    progress: Callable[[str], None] = print,
) -> HazardCandidate:
    """Train Round 0 plus exactly three TRAIN-only hard-negative rounds."""
    if any(run.split != "train" for run in runs.values()):
        raise ValueError("Hazard candidate training may receive TRAIN runs only")
    root = repository_root.resolve()
    fit_ids, monitor_ids = _train_monitor_partition(
        runs, sorted(runs), precursor_samples
    )
    normalizer = fit_hazard_normalizer(runs, sorted(runs))
    schema = hazard_feature_schema()
    normalizer_path = artifact_path / "normalization" / "gru_history20.json"
    _write_json(
        normalizer_path,
        {
            **normalizer.to_dict(),
            "components": ["pelvis_imu6"],
            "feature_schema": list(schema),
            "feature_schema_sha256": canonical_sha256(schema),
            "train_only": True,
        },
    )
    accumulated: dict[str, tuple[int, ...]] = {}
    round_records: list[dict[str, object]] = []
    final_paths: tuple[Path, ...] = ()
    for round_id in range(HNM_ROUNDS + 1):
        fit_windows = build_hazard_windows(
            runs,
            fit_ids,
            precursor_samples,
            normalizer,
            extra_negative_endpoints=accumulated,
        )
        monitor_windows = build_hazard_windows(
            runs,
            monitor_ids,
            precursor_samples,
            normalizer,
            extra_negative_endpoints=accumulated,
        )
        paths: list[Path] = []
        epochs: list[int] = []
        for seed in training_config["seeds"]:
            path = (
                artifact_path
                / "checkpoints"
                / f"unified_gru_history20_round{round_id}_seed{seed}.pt"
            )
            model, result = train_model(
                "gru",
                HISTORY_MS,
                fit_windows,
                monitor_windows,
                int(seed),
                batch_size=int(training_config["batch_size"]),
                max_epochs=int(training_config["max_epochs"]),
                patience=int(training_config["patience"]),
                learning_rate=float(training_config["learning_rate"]),
                class_names=EVENT_CLASS_NAMES,
                selection_metric="validation_loss",
            )
            save_checkpoint(
                path,
                model,
                "gru",
                HISTORY_MS,
                int(seed),
                result,
                input_channels=len(schema),
                class_names=EVENT_CLASS_NAMES,
            )
            paths.append(path)
            epochs.append(result.best_epoch)
        final_paths = tuple(paths)
        record: dict[str, object] = {
            "round": round_id,
            "fit_windows": len(fit_windows),
            "monitor_windows": len(monitor_windows),
            "fit_class_counts": list(fit_windows.selected_by_class),
            "monitor_class_counts": list(monitor_windows.selected_by_class),
            "best_epochs": epochs,
        }
        progress(f"Hazard history=20 round={round_id} trained")
        if round_id < HNM_ROUNDS:
            mined, mining_record = _mine_training_round(
                runs,
                precursor_samples,
                normalizer,
                final_paths,
                accumulated,
            )
            accumulated = _merge_endpoint_maps(accumulated, mined)
            record["hard_negative_mining"] = mining_record
        round_records.append(record)
    return HazardCandidate(
        history_ms=HISTORY_MS,
        normalizer=normalizer,
        checkpoint_paths=final_paths,
        record={
            "model_family": "gru",
            "history_ms": HISTORY_MS,
            "feature_dimension": len(schema),
            "feature_schema_sha256": canonical_sha256(schema),
            "normalizer_path": str(normalizer_path.relative_to(root)),
            "normalizer_sha256": sha256_file(normalizer_path),
            "checkpoint_sha256": {
                str(path.relative_to(root)): sha256_file(path) for path in final_paths
            },
            "parameters": parameter_count(load_checkpoint(final_paths[0])[0]),
            "rounds": round_records,
            "hnm_rounds": HNM_ROUNDS,
            "validation_access_before_hnm3": False,
        },
    )
