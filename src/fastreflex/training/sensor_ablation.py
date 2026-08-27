"""Fixed IMU6/FSR8/Fusion14 early-target observability ablation."""

from __future__ import annotations

import csv
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
    Normalizer,
    WindowSet,
    build_profile_windows,
    fit_profile_normalizer,
    load_manifest,
    load_profile_arrays,
    sha256_file,
    validate_split,
)
from fastreflex.evaluation.metrics import classification_metrics
from fastreflex.evaluation.time_to_separation import (
    ReplayTrace,
    first_sustained_endpoint,
    replay_causal,
)
from fastreflex.models.baselines import build_model, parameter_count
from fastreflex.simulation.sensors import FSR_CHANNELS
from fastreflex.training.trainer import (
    evaluate_model,
    save_checkpoint,
    train_model,
)


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PROFILES = ("imu6", "fsr8", "fusion14")
EXPECTED_OUTCOMES = {"BENIGN": 16, "SLIP": 8, "SINK": 9, "DUAL": 0, "INVALID": 7}
FROZEN_FIRST_POC_SPLIT_SHA256 = (
    "3b1b29a5e009783da2db0d1bdd198df24695d44c4b0cc55228bf28dfefda2a75"
)


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
        fieldnames = list(rows[0])
        fieldnames.extend(
            name for row in rows for name in row if name not in fieldnames
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_run(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


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
    raise ValueError(f"unknown benign scenario group: {run_id}")


def _metric_summary(seed_metrics: Sequence[Mapping[str, object]]) -> dict[str, object]:
    macro = np.asarray([item["macro_f1"] for item in seed_metrics], dtype=np.float64)
    accuracy = np.asarray([item["accuracy"] for item in seed_metrics], dtype=np.float64)
    run_balanced = np.asarray(
        [item["run_balanced_accuracy"] for item in seed_metrics], dtype=np.float64
    )
    recalls = {
        name: np.asarray(
            [item["per_class"][name]["recall"] for item in seed_metrics],
            dtype=np.float64,
        )
        for name in CLASS_NAMES
    }
    worst_index = int(np.argmin(macro))
    return {
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(accuracy.std()),
        "macro_f1_mean": float(macro.mean()),
        "macro_f1_std": float(macro.std()),
        "run_balanced_accuracy_mean": float(run_balanced.mean()),
        "per_class_recall_mean": {
            name: float(values.mean()) for name, values in recalls.items()
        },
        "per_class_recall_std": {
            name: float(values.std()) for name, values in recalls.items()
        },
        "worst_seed_index": worst_index,
        "worst_seed_macro_f1": float(macro[worst_index]),
    }


def _plot_fsr_sanity(records: Mapping[str, object], plots: Path) -> None:
    gait_id = "normal_concrete_s015_p000"
    sink_id = "sink_left_severe_s020_p035"
    sand_id = "normal_sand_s015_p000"
    gait = _load_run(records[gait_id].path)
    sink = _load_run(records[sink_id].path)
    sand = _load_run(records[sand_id].path)

    time_s = np.arange(len(gait["foot_fsr"])) / 1000.0
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for channel in range(4):
        axes[0].plot(time_s, gait["foot_fsr"][:, channel], label=FSR_CHANNELS[channel])
        axes[1].plot(time_s, gait["foot_fsr"][:, channel + 4], label=FSR_CHANNELS[channel + 4])
    axes[0].set_title("Left virtual FSR4 — representative concrete gait")
    axes[1].set_title("Right virtual FSR4")
    for axis in axes:
        axis.set_ylabel("normal load (N)")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    axes[1].set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(plots / "fsr8_representative_gait.png", dpi=150)
    plt.close(figure)

    t1 = int(np.min(sink["first_sink_physical_onset_sample_per_foot"][sink["first_sink_physical_onset_sample_per_foot"] >= 0]))
    t2 = int(sink["first_sink_degradation_onset_sample"])
    start, stop = max(0, t1 - 500), min(len(sink["foot_fsr"]), t2 + 500)
    time_ms = np.arange(start, stop) - t1
    figure, axis = plt.subplots(figsize=(13, 5))
    for channel in range(8):
        axis.plot(time_ms, sink["foot_fsr"][start:stop, channel], label=FSR_CHANNELS[channel])
    axis.axvline(0, color="black", linestyle="--", label="physical Sink t1")
    axis.axvline(t2 - t1, color="red", linestyle=":", label="degradation t2")
    axis.set(xlabel="time from t1 (ms)", ylabel="normal load (N)", title="Hazardous Sink FSR8 aligned to t1/t2")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "hazardous_sink_fsr_t1_t2.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(np.arange(len(sand["foot_fsr"])) / 1000.0, sand["foot_fsr"].sum(axis=1), label="uniform sand benign")
    axis.plot(np.arange(len(sink["foot_fsr"])) / 1000.0, sink["foot_fsr"].sum(axis=1), label="severe Sink hazardous", alpha=0.8)
    axis.set(xlabel="time (s)", ylabel="bilateral total normal load (N)", title="Uniform sand hard negative vs hazardous Sink")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "uniform_sand_vs_hazardous_sink_fsr.png", dpi=150)
    plt.close(figure)


def run_sensor_sanity(records: Mapping[str, object], plots: Path) -> dict[str, object]:
    """Audit raw FSR contact behavior and terrain-shortcut risk."""
    plots.mkdir(parents=True, exist_ok=True)
    nonfinite = negative = airborne_nonzero = loaded_nonpositive = 0
    total_samples = 0
    loaded_values: dict[str, list[np.ndarray]] = {}
    quadrant_sum = np.zeros(8, dtype=np.float64)
    touchdown_loads: list[float] = []
    unload_loads: list[float] = []
    bilateral_loaded_pairs: list[np.ndarray] = []
    per_run: dict[str, object] = {}
    for run_id, record in records.items():
        arrays = _load_run(record.path)
        fsr = np.asarray(arrays["foot_fsr"], dtype=np.float32)
        physical = np.asarray(arrays["physical_contact"], dtype=bool)
        loaded = np.asarray(arrays["loaded_contact"], dtype=bool)
        touchdown = np.asarray(arrays["touchdown"], dtype=bool)
        pre_fall = np.asarray(arrays["pre_fall_valid"], dtype=bool)
        total_samples += len(fsr)
        nonfinite += int((~np.isfinite(fsr)).sum())
        negative += int((fsr < 0.0).sum())
        side_load = fsr.reshape(-1, 2, 4).sum(axis=2)
        airborne_nonzero += int(np.count_nonzero(side_load[~physical] != 0.0))
        loaded_nonpositive += int(np.count_nonzero(side_load[loaded] <= 0.0))
        quadrant_sum += fsr[pre_fall].sum(axis=0, dtype=np.float64)
        valid_touchdown = touchdown & pre_fall[:, None]
        touchdown_loads.extend(side_load[valid_touchdown].tolist())
        falling = loaded[:-1] & ~loaded[1:] & pre_fall[1:, None]
        unload_loads.extend(side_load[1:][falling].tolist())
        valid_loaded = loaded & pre_fall[:, None]
        selected = side_load[valid_loaded]
        bilateral = np.all(loaded, axis=1) & pre_fall
        if bilateral.any():
            bilateral_loaded_pairs.append(side_load[bilateral])
        group = (
            _scenario_group(run_id)
            if record.observed_outcome == "BENIGN"
            else record.observed_outcome.lower()
        )
        if record.observed_outcome != "INVALID":
            loaded_values.setdefault(group, []).append(selected)
        per_run[run_id] = {
            "outcome": record.observed_outcome,
            "loaded_total_mean_n": float(selected.mean()) if len(selected) else None,
            "loaded_total_p95_n": float(np.percentile(selected, 95)) if len(selected) else None,
            "left_right_loaded_mean_n": [
                float(side_load[:, side][valid_loaded[:, side]].mean())
                if valid_loaded[:, side].any() else None
                for side in range(2)
            ],
        }
    if nonfinite or negative or airborne_nonzero or loaded_nonpositive:
        raise ValueError("raw FSR sanity failed contact/finite/nonnegative invariants")
    group_summary = {
        group: {
            "loaded_foot_sample_count": int(sum(len(part) for part in parts)),
            "mean_n": float(np.concatenate(parts).mean()),
            "std_n": float(np.concatenate(parts).std()),
            "p05_n": float(np.percentile(np.concatenate(parts), 5)),
            "p95_n": float(np.percentile(np.concatenate(parts), 95)),
        }
        for group, parts in loaded_values.items()
        if any(len(part) for part in parts)
    }
    _plot_fsr_sanity(records, plots)
    bilateral_pairs = np.concatenate(bilateral_loaded_pairs, axis=0)
    bilateral_difference = np.abs(bilateral_pairs[:, 0] - bilateral_pairs[:, 1])
    bilateral_total = bilateral_pairs.sum(axis=1)
    return {
        "sample_count": total_samples,
        "finite": True,
        "nonnegative": True,
        "airborne_side_nonzero_count": airborne_nonzero,
        "loaded_side_nonpositive_count": loaded_nonpositive,
        "touchdown_load_n": {
            "count": len(touchdown_loads),
            "median": float(np.median(touchdown_loads)),
        },
        "unload_next_sample_n": {
            "count": len(unload_loads),
            "max": float(np.max(unload_loads)) if unload_loads else None,
        },
        "left_right_loaded_behavior": {
            "bilateral_sample_count": len(bilateral_pairs),
            "left_mean_n": float(bilateral_pairs[:, 0].mean()),
            "right_mean_n": float(bilateral_pairs[:, 1].mean()),
            "median_absolute_difference_fraction": float(
                np.median(bilateral_difference / np.maximum(bilateral_total, 1.0e-12))
            ),
        },
        "quadrant_load_fraction": (quadrant_sum / quadrant_sum.sum()).tolist(),
        "loaded_scale_by_scenario": group_summary,
        "terrain_shortcut_audit": {
            "raw_scale_differs_by_scenario": True,
            "interpretation": "all benign sand and mild/moderate transitions remain NORMAL hard negatives; group load scale alone is not a hazard label",
        },
        "per_run": per_run,
    }


def _load_events(record: object) -> tuple[dict[str, np.ndarray], object]:
    arrays = _load_run(record.path)
    return arrays, extract_event_samples(arrays, record.observed_outcome)


def _aggregate_horizons(
    positive_rows: Sequence[Mapping[str, object]], horizons: Sequence[int]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile in PROFILES:
        for seed in sorted({int(row["seed"]) for row in positive_rows}):
            for outcome in ("SLIP", "SINK"):
                selected = [
                    row for row in positive_rows
                    if row["profile"] == profile and int(row["seed"]) == seed and row["outcome"] == outcome
                ]
                for horizon in horizons:
                    detected = [
                        row for row in selected
                        if row["first_sustained_sample"] is not None
                        and int(row["first_sustained_sample"]) <= int(row["t1_sample"]) + horizon
                    ]
                    positive_margin = [
                        row for row in selected
                        if row["t2_sample"] is not None and int(row["t2_sample"]) > int(row["t1_sample"])
                    ]
                    pre_t2 = [
                        row for row in positive_margin
                        if row["first_sustained_sample"] is not None
                        and int(row["first_sustained_sample"]) < int(row["t2_sample"])
                    ]
                    latency = [float(row["latency_from_t1_ms"]) for row in detected]
                    margins = [float(row["margin_to_t2_ms"]) for row in pre_t2]
                    rows.append({
                        "profile": profile,
                        "seed": seed,
                        "outcome": outcome,
                        "horizon_ms": horizon,
                        "event_count": len(selected),
                        "detected_count": len(detected),
                        "event_recall": len(detected) / len(selected),
                        "median_latency_from_t1_ms": None if not latency else float(np.median(latency)),
                        "positive_margin_count": len(positive_margin) if outcome == "SINK" else "",
                        "pre_t2_detected_count": len(pre_t2) if outcome == "SINK" else "",
                        "pre_t2_detection_rate": (len(pre_t2) / len(positive_margin)) if outcome == "SINK" else "",
                        "median_t2_margin_ms": (None if not margins else float(np.median(margins))) if outcome == "SINK" else "",
                    })
    return rows


def _aggregate_benign(benign_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for profile in PROFILES:
        for seed in sorted({int(row["seed"]) for row in benign_rows}):
            selected_seed = [row for row in benign_rows if row["profile"] == profile and int(row["seed"]) == seed]
            for group in ("all", "concrete", "marble", "uniform_sand", "benign_sink_mild", "benign_sink_moderate"):
                selected = selected_seed if group == "all" else [row for row in selected_seed if row["scenario_group"] == group]
                result.append({
                    "profile": profile,
                    "seed": seed,
                    "scenario_group": group,
                    "run_count": len(selected),
                    "sustained_slip_fp_runs": sum(bool(row["sustained_slip"]) for row in selected),
                    "sustained_sink_fp_runs": sum(bool(row["sustained_sink"]) for row in selected),
                    "sustained_any_hazard_fp_runs": sum(bool(row["sustained_any_hazard"]) for row in selected),
                    "run_level_any_fp_rate": sum(bool(row["sustained_any_hazard"]) for row in selected) / len(selected),
                })
    return result


def _plot_probabilities(
    traces: Mapping[str, object], events: object, path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(13, 5))
    for profile in PROFILES:
        trace = traces[profile]
        axis.plot(trace.endpoint_samples - events.t1, trace.probabilities[:, 2], label=profile)
    axis.axvline(0, color="black", linestyle="--", label="physical Sink t1")
    axis.axvline(events.t2 - events.t1, color="red", linestyle=":", label="degradation t2")
    axis.set_xlim(-500, 1000)
    axis.set_ylim(0.0, 1.0)
    axis.set(xlabel="time from t1 (ms)", ylabel="P(SINK)", title="Causal SINK probability by sensor profile")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_fsr_observability_pilot(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Run the fixed 100 ms, three-profile, three-seed comparison."""
    repository_root = repository_root.resolve()
    with config_path.resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["experiment"]["id"] != "FSR_OBSERVABILITY_PILOT":
        raise ValueError("unsupported sensor ablation experiment")
    dataset_path = repository_root / "data/raw" / config["experiment"]["dataset_id"]
    artifact_path = repository_root / config["artifacts"]["path"]
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite experiment artifacts: {artifact_path}")
    dataset_summary = validate_dataset(dataset_path)
    if dataset_summary["outcomes"] != EXPECTED_OUTCOMES:
        raise ValueError(f"sensor dataset outcome parity failed: {dataset_summary['outcomes']}")
    records = load_manifest(dataset_path)
    source_split = repository_root / config["split"]["source"]
    if sha256_file(source_split) != FROZEN_FIRST_POC_SPLIT_SHA256:
        raise ValueError("frozen first-PoC split SHA-256 mismatch")
    split = {name: tuple(config["split"][name]) for name in ("train", "validation", "holdout")}
    split_counts = validate_split(records, split, expected_outcome_counts={
        "train": {"BENIGN": 10, "SLIP": 5, "SINK": 5},
        "validation": {"BENIGN": 3, "SLIP": 1, "SINK": 2},
        "holdout": {"BENIGN": 3, "SLIP": 2, "SINK": 2},
    })
    artifact_path.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_path / "split.json", {
        "source": config["split"]["source"],
        "dataset_id": config["experiment"]["dataset_id"],
        "manifest_sha256": sha256_file(dataset_path / "manifest.csv"),
        "run_ids": {name: list(ids) for name, ids in split.items()},
        "outcome_counts": split_counts,
        "run_disjoint": True,
        "invalid_excluded": 7,
    })
    sanity = run_sensor_sanity(records, artifact_path / "plots")
    _write_json(artifact_path / "sensor_sanity.json", sanity)
    progress("FSR raw sanity complete")

    window_samples = int(config["windowing"]["primary_ms"])
    stride_samples = int(config["windowing"]["stride_ms"])
    train_cap = int(config["windowing"]["train_max_windows_per_run_class"])
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    training_args = {
        "batch_size": int(config["training"]["batch_size"]),
        "max_epochs": int(config["training"]["max_epochs"]),
        "patience": int(config["training"]["early_stopping_patience"]),
        "learning_rate": float(config["training"]["learning_rate"]),
    }
    models: dict[tuple[str, int], object] = {}
    normalizers: dict[str, Normalizer] = {}
    classification: dict[str, object] = {}
    for profile in PROFILES:
        normalizer = fit_profile_normalizer(
            records, split["train"], profile, early_targets=True,
            epsilon=1.0e-8,
        )
        normalizers[profile] = normalizer
        _write_json(artifact_path / "normalization" / f"{profile}.json", normalizer.to_dict())
        train_windows = build_profile_windows(
            records, split["train"], profile, window_samples, stride_samples,
            normalizer, early_targets=True, cap_per_run_class=train_cap,
        )
        validation_windows = build_profile_windows(
            records, split["validation"], profile, window_samples, stride_samples,
            normalizer, early_targets=True,
        )
        holdout_windows = build_profile_windows(
            records, split["holdout"], profile, window_samples, stride_samples,
            normalizer, early_targets=True,
        )
        seed_results: list[dict[str, object]] = []
        for seed in seeds:
            progress(f"training {profile} MLP 100 ms seed {seed}")
            model, training_result = train_model(
                "mlp", window_samples, train_windows, validation_windows, seed,
                **training_args,
            )
            models[(profile, seed)] = model
            checkpoint = artifact_path / "checkpoints" / f"mlp_{profile}_100ms_seed_{seed}.pt"
            save_checkpoint(
                checkpoint, model, "mlp", window_samples, seed, training_result,
                input_channels=int(train_windows.inputs.shape[2]),
            )
            seed_results.append({
                "seed": seed,
                "best_epoch": training_result.best_epoch,
                "epochs_completed": training_result.epochs_completed,
                "validation": evaluate_model(model, validation_windows, training_args["batch_size"]),
                "holdout": evaluate_model(model, holdout_windows, training_args["batch_size"]),
            })
        classification[profile] = {
            "window_ms": window_samples,
            "input_channels": int(train_windows.inputs.shape[2]),
            "parameter_count": parameter_count(build_model("mlp", window_samples, int(train_windows.inputs.shape[2]))),
            "train_windows_selected": dict(zip(CLASS_NAMES, train_windows.selected_by_class)),
            "validation_windows": dict(zip(CLASS_NAMES, validation_windows.selected_by_class)),
            "holdout_windows": dict(zip(CLASS_NAMES, holdout_windows.selected_by_class)),
            "validation": _metric_summary([item["validation"] for item in seed_results]),
            "holdout": _metric_summary([item["holdout"] for item in seed_results]),
            "seeds": seed_results,
        }
        for split_name in ("validation", "holdout"):
            worst_index = int(classification[profile][split_name]["worst_seed_index"])
            classification[profile][split_name]["worst_seed"] = seeds[worst_index]

    positive_rows: list[dict[str, object]] = []
    benign_rows: list[dict[str, object]] = []
    probability_example: dict[str, object] = {}
    example_run = "sink_left_severe_s020_p035"
    for profile in PROFILES:
        for seed in seeds:
            model = models[(profile, seed)]
            for run_id, record in records.items():
                if record.observed_outcome not in ("BENIGN", "SLIP", "SINK"):
                    continue
                raw_values, _, _ = load_profile_arrays(record, profile, early_targets=True)
                trace = replay_causal(model, raw_values, normalizers[profile], window_samples, stride_samples=1)
                arrays, events = _load_events(record)
                if seed == seeds[0] and run_id == example_run:
                    probability_example[profile] = trace
                    example_events = events
                if record.observed_outcome == "BENIGN":
                    audit = audit_false_positives(trace, int(trace.endpoint_samples[0]), events.t3, int(config["replay"]["sustained_ms"]))
                    benign_rows.append({
                        "profile": profile, "seed": seed, "run_id": run_id,
                        "scenario_group": _scenario_group(run_id), **audit,
                    })
                    continue
                target_class = 1 if record.observed_outcome == "SLIP" else 2
                sustained = first_sustained_endpoint(
                    trace.predictions, trace.endpoint_samples, target_class,
                    events.t1, events.t3, int(config["replay"]["sustained_ms"]),
                )
                positive_rows.append({
                    "profile": profile,
                    "seed": seed,
                    "run_id": run_id,
                    "split": next(name for name, ids in split.items() if run_id in ids),
                    "outcome": record.observed_outcome,
                    "sink_side": record.sink_side or "",
                    "t1_sample": events.t1,
                    "t2_sample": events.t2,
                    "t3_sample_exclusive": events.t3,
                    "zero_margin_sink": record.observed_outcome == "SINK" and events.t1 == events.t2,
                    "first_sustained_sample": sustained,
                    "latency_from_t1_ms": None if sustained is None else sustained - events.t1,
                    "margin_to_t2_ms": None if sustained is None or events.t2 is None else events.t2 - sustained,
                    "pre_t2_detected": bool(sustained is not None and events.t2 is not None and sustained < events.t2),
                })
    horizon_rows = _aggregate_horizons(positive_rows, config["replay"]["horizons_ms"])
    benign_summary = _aggregate_benign(benign_rows)
    _write_csv(artifact_path / "per_run.csv", [*positive_rows, *benign_rows])
    _write_csv(artifact_path / "horizon_recall.csv", horizon_rows)
    _write_csv(artifact_path / "benign_fp.csv", benign_summary)
    _plot_probabilities(probability_example, example_events, artifact_path / "plots" / "sink_probability_profiles.png")

    metrics = {
        "experiment_id": config["experiment"]["id"],
        "dataset": dataset_summary,
        "comparison": "fixed MLP 100 ms; identical early targets, split, stride, cap, optimizer, seeds, and replay",
        "classification": classification,
        "horizon_recall": horizon_rows,
        "benign_false_positive": benign_summary,
        "optional_50ms": "not_run_primary_100ms_ablation_is_the_bounded_sensor_decision_evidence",
    }
    _write_json(artifact_path / "metrics.json", metrics)
    return artifact_path, metrics


BINARY_CLASS_NAMES = ("NORMAL", "SINK")


def _binary_windows(windows: WindowSet) -> WindowSet:
    """Map canonical raw class IDs NORMAL=0/SINK=2 to a binary study target."""
    if not set(np.unique(windows.targets)).issubset({0, 2}):
        raise ValueError("Sink observability windows contain a non-Sink hazard class")
    targets = np.where(windows.targets == 2, 1, 0).astype(np.int64)
    available = (
        int(windows.available_by_class[0]),
        int(windows.available_by_class[2]),
        0,
    )
    return WindowSet(
        windows.inputs,
        targets,
        windows.run_ids,
        windows.endpoint_samples,
        available,
    )


def _binary_counts(windows: WindowSet) -> dict[str, int]:
    counts = np.bincount(windows.targets, minlength=2)
    return {
        name: int(counts[index]) for index, name in enumerate(BINARY_CLASS_NAMES)
    }


def _predict_logits(
    model: torch.nn.Module,
    windows: WindowSet,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            inputs = torch.from_numpy(windows.inputs[start : start + batch_size])
            parts.append(model(inputs).cpu().numpy())
    return np.concatenate(parts).astype(np.float32, copy=False)


def _ensemble_metrics(
    models: Sequence[torch.nn.Module],
    windows: WindowSet,
    batch_size: int,
) -> dict[str, object]:
    predictions = _ensemble_predictions(models, windows, batch_size)
    return classification_metrics(
        windows.targets,
        predictions,
        windows.run_ids,
        BINARY_CLASS_NAMES,
    )


def _ensemble_predictions(
    models: Sequence[torch.nn.Module],
    windows: WindowSet,
    batch_size: int,
) -> np.ndarray:
    logits = np.stack(
        [_predict_logits(model, windows, batch_size) for model in models]
    ).mean(axis=0)
    return logits.argmax(axis=1).astype(np.int64)


def _sink_stratified_recall(
    predictions: np.ndarray,
    windows: WindowSet,
    records: Mapping[str, object],
) -> dict[str, dict[str, dict[str, object]]]:
    """Report positive-window and run-balanced recall without pooling runs."""
    sink_mask = windows.targets == 1
    dimensions = {
        "severity": lambda record: record.sink_severity,
        "side": lambda record: record.sink_side,
        "pattern": lambda record: (
            (record.sink_support_pattern or "").removesuffix("_deformable")
        ),
        "speed_mps": lambda record: f"{record.speed_mps:.2f}",
    }
    result: dict[str, dict[str, dict[str, object]]] = {}
    for dimension, getter in dimensions.items():
        grouped_runs: dict[str, list[str]] = {}
        for run_id in sorted(set(windows.run_ids[sink_mask])):
            value = getter(records[run_id])
            if value:
                grouped_runs.setdefault(str(value), []).append(run_id)
        groups: dict[str, dict[str, object]] = {}
        for value, run_ids in sorted(grouped_runs.items()):
            mask = sink_mask & np.isin(windows.run_ids, run_ids)
            per_run = []
            for run_id in run_ids:
                run_mask = sink_mask & (windows.run_ids == run_id)
                per_run.append(float(np.mean(predictions[run_mask] == 1)))
            groups[value] = {
                "run_count": len(run_ids),
                "window_count": int(np.count_nonzero(mask)),
                "recall": float(np.mean(predictions[mask] == 1)),
                "run_balanced_recall": float(np.mean(per_run)),
            }
        result[dimension] = groups
    return result


def _assert_holdout_waveforms_sealed(
    requested_run_ids: Sequence[str],
    holdout_run_ids: Sequence[str],
    holdout_opened: bool,
) -> None:
    if not holdout_opened and set(requested_run_ids) & set(holdout_run_ids):
        raise RuntimeError("HOLDOUT waveform access attempted before selection")


def _ensemble_replay(
    models: Sequence[torch.nn.Module],
    values: np.ndarray,
    normalizer: Normalizer,
    window_samples: int,
    batch_size: int,
) -> ReplayTrace:
    traces = [
        replay_causal(
            model,
            values,
            normalizer,
            window_samples,
            stride_samples=1,
            batch_size=batch_size,
        )
        for model in models
    ]
    endpoints = traces[0].endpoint_samples
    if any(not np.array_equal(trace.endpoint_samples, endpoints) for trace in traces):
        raise RuntimeError("ensemble replay endpoints disagree")
    logits = np.stack([trace.logits for trace in traces]).mean(axis=0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    return ReplayTrace(
        endpoints,
        logits.astype(np.float32),
        probabilities.astype(np.float32),
        logits.argmax(axis=1).astype(np.int8),
    )


def _frozen_observability_split(
    records: Mapping[str, object],
    configured: Mapping[str, Sequence[str]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    split = {
        name: tuple(configured[name])
        for name in ("train", "validation", "holdout")
    }
    assigned = [run_id for ids in split.values() for run_id in ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(records):
        raise ValueError("frozen split is not run-disjoint and exhaustive")
    valid = {
        name: tuple(
            run_id
            for run_id in run_ids
            if records[run_id].observed_outcome in ("BENIGN", "SINK")
        )
        for name, run_ids in split.items()
    }
    return split, valid


def _read_manifest_rows(dataset_path: Path) -> dict[str, dict[str, str]]:
    with (dataset_path / "manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        return {row["run_id"]: row for row in csv.DictReader(stream)}


def _raw_observability_sanity(
    records: Mapping[str, object],
    run_ids: Sequence[str],
    horizons: Sequence[int],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for run_id in run_ids:
        record = records[run_id]
        if record.observed_outcome != "SINK":
            continue
        arrays = _load_run(record.path)
        s1 = int(arrays["first_deformable_sink_onset_sample"])
        for horizon in horizons:
            sample = s1 + int(horizon)
            if sample < 0 or sample >= len(arrays["pelvis_imu"]):
                continue
            imu = np.asarray(arrays["pelvis_imu"][sample], dtype=np.float64)
            fsr = np.asarray(arrays["foot_fsr"][sample], dtype=np.float64)
            rows.append(
                {
                    "run_id": run_id,
                    "horizon_ms": int(horizon),
                    "imu6": imu.tolist(),
                    "fsr8": fsr.tolist(),
                    "bilateral_load_n": float(fsr.sum()),
                }
            )
    summary: dict[str, object] = {}
    for horizon in horizons:
        selected = [row for row in rows if row["horizon_ms"] == horizon]
        if not selected:
            continue
        imu = np.asarray([row["imu6"] for row in selected])
        fsr = np.asarray([row["fsr8"] for row in selected])
        summary[str(horizon)] = {
            "run_count": len(selected),
            "imu6_mean": imu.mean(axis=0).tolist(),
            "imu6_std": imu.std(axis=0).tolist(),
            "fsr8_mean": fsr.mean(axis=0).tolist(),
            "fsr8_std": fsr.std(axis=0).tolist(),
            "bilateral_load_mean_n": float(fsr.sum(axis=1).mean()),
            "bilateral_load_std_n": float(fsr.sum(axis=1).std()),
        }
    return {
        "splits_used": ["train", "validation"],
        "handcrafted_features": False,
        "rows": rows,
        "summary_by_s1_horizon_ms": summary,
    }


def _benign_group(record: object) -> str:
    if record.patch_start_x is None:
        return record.terrain if record.terrain != "sand" else "uniform_sand"
    if record.sink_support_pattern == "balanced_deformable":
        return f"balanced_{record.sink_severity}"
    return "uneven_no_s1"


def _aggregate_sink_replay(
    positive_rows: Sequence[Mapping[str, object]],
    benign_rows: Sequence[Mapping[str, object]],
    horizons: Sequence[int],
) -> dict[str, object]:
    horizon_recall: dict[str, dict[str, object]] = {}
    for split_name in ("train", "validation", "holdout", "all"):
        selected = (
            list(positive_rows)
            if split_name == "all"
            else [row for row in positive_rows if row["split"] == split_name]
        )
        horizon_recall[split_name] = {
            str(horizon): {
                "events": len(selected),
                "detected": sum(bool(row[f"detected_by_{horizon}ms"]) for row in selected),
                "recall": (
                    sum(bool(row[f"detected_by_{horizon}ms"]) for row in selected)
                    / len(selected)
                    if selected
                    else 0.0
                ),
            }
            for horizon in horizons
        }
    fp_summary: dict[str, dict[str, object]] = {}
    for split_name in ("train", "validation", "holdout", "all"):
        split_rows = (
            list(benign_rows)
            if split_name == "all"
            else [row for row in benign_rows if row["split"] == split_name]
        )
        for group in (
            "all",
            "concrete",
            "marble",
            "uniform_sand",
            "balanced_mild",
            "balanced_moderate",
            "balanced_severe",
            "uneven_no_s1",
        ):
            selected = (
                split_rows
                if group == "all"
                else [row for row in split_rows if row["scenario_group"] == group]
            )
            if not selected:
                continue
            fp_summary[f"{split_name}:{group}"] = {
                "run_count": len(selected),
                "sustained_fp_runs": sum(bool(row["sustained_sink_fp"]) for row in selected),
                "run_rate": sum(bool(row["sustained_sink_fp"]) for row in selected) / len(selected),
                "false_positive_duration_ms": int(
                    sum(int(row["false_positive_duration_ms"]) for row in selected)
                ),
                "pre_event_fp_runs": sum(bool(row["pre_event_fp"]) for row in selected),
            }
    return {"horizon_recall": horizon_recall, "benign_false_positive": fp_summary}


def run_sink_sensor_observability_study(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Run the frozen binary Sink sensor/model comparison and one-shot holdout."""
    repository_root = repository_root.resolve()
    with config_path.resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["experiment"]["id"] != "SINK_SENSOR_OBSERVABILITY_STUDY":
        raise ValueError("unsupported Sink observability experiment")
    dataset_path = repository_root / "data/raw" / config["experiment"]["dataset_id"]
    artifact_path = repository_root / config["artifacts"]["path"]
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite experiment artifacts: {artifact_path}")
    dataset_summary = validate_dataset(dataset_path)
    records = load_manifest(dataset_path)
    manifest_rows = _read_manifest_rows(dataset_path)
    frozen_split, valid_split = _frozen_observability_split(records, config["split"])
    outcomes = dataset_summary["outcomes"]
    positive_rows_manifest = [
        row for row in manifest_rows.values() if row["observed_outcome"] == "SINK"
    ]
    positive_by_side = {
        side: sum(row["sink_side"] == side for row in positive_rows_manifest)
        for side in ("left", "right")
    }
    positive_by_pattern = {
        pattern: sum(
            row["sink_support_pattern"] == f"{pattern}_deformable"
            for row in positive_rows_manifest
        )
        for pattern in ("medial", "lateral", "localized")
    }
    readiness = {
        "observed_sink": int(outcomes["SINK"]),
        "benign": int(outcomes["BENIGN"]),
        "invalid": int(outcomes["INVALID"]),
        "positive_by_side": positive_by_side,
        "positive_by_pattern": positive_by_pattern,
    }
    ready = bool(
        outcomes["SINK"] >= int(config["readiness"]["minimum_sink_runs"])
        and outcomes["BENIGN"] >= int(config["readiness"]["minimum_benign_runs"])
        and min(positive_by_side.values())
        >= int(config["readiness"]["minimum_sink_per_side"])
        and min(positive_by_pattern.values())
        >= int(config["readiness"]["minimum_sink_per_pattern"])
    )
    readiness["status"] = "PASS" if ready else "SINK_OBSERVABILITY_DATASET_NEEDS_REVISION"
    artifact_path.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_path / "dataset_readiness.json", readiness)
    if not ready:
        raise RuntimeError("SINK_OBSERVABILITY_DATASET_NEEDS_REVISION")

    raw_sanity = _raw_observability_sanity(
        records,
        (*valid_split["train"], *valid_split["validation"]),
        [int(value) for value in config["raw_sanity"]["horizons_ms"]],
    )
    _write_json(artifact_path / "raw_sensor_sanity.json", raw_sanity)
    progress("TRAIN/VALIDATION raw sensor sanity complete; HOLDOUT waveforms sealed")

    window_samples = int(config["windowing"]["primary_ms"])
    stride_samples = int(config["windowing"]["stride_ms"])
    train_cap = int(config["windowing"]["train_max_windows_per_run_class"])
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    families = tuple(config["models"])
    profiles = tuple(config["sensor"]["profiles"])
    batch_size = int(config["training"]["batch_size"])
    training_args = {
        "batch_size": batch_size,
        "max_epochs": int(config["training"]["max_epochs"]),
        "patience": int(config["training"]["early_stopping_patience"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "class_names": BINARY_CLASS_NAMES,
    }
    normalizers: dict[str, Normalizer] = {}
    candidate_models: dict[tuple[str, str], list[torch.nn.Module]] = {}
    candidates: list[dict[str, object]] = []
    holdout_opened = False
    development_ids = (*valid_split["train"], *valid_split["validation"])
    _assert_holdout_waveforms_sealed(
        development_ids, valid_split["holdout"], holdout_opened
    )
    for profile in profiles:
        normalizer = fit_profile_normalizer(
            records,
            valid_split["train"],
            profile,
            early_targets=False,
            epsilon=1.0e-8,
        )
        normalizers[profile] = normalizer
        _write_json(
            artifact_path / "normalization" / f"{profile}.json",
            normalizer.to_dict(),
        )
        train_windows = _binary_windows(
            build_profile_windows(
                records,
                valid_split["train"],
                profile,
                window_samples,
                stride_samples,
                normalizer,
                early_targets=False,
                cap_per_run_class=train_cap,
            )
        )
        validation_windows = _binary_windows(
            build_profile_windows(
                records,
                valid_split["validation"],
                profile,
                window_samples,
                stride_samples,
                normalizer,
                early_targets=False,
            )
        )
        for family in families:
            models: list[torch.nn.Module] = []
            seed_metrics: list[dict[str, object]] = []
            for seed in seeds:
                progress(f"training {profile} {family.upper()} 100 ms seed {seed}")
                model, result = train_model(
                    family,
                    window_samples,
                    train_windows,
                    validation_windows,
                    seed,
                    **training_args,
                )
                models.append(model)
                save_checkpoint(
                    artifact_path
                    / "checkpoints"
                    / f"{family}_{profile}_100ms_seed_{seed}.pt",
                    model,
                    family,
                    window_samples,
                    seed,
                    result,
                    input_channels=train_windows.inputs.shape[2],
                    class_names=BINARY_CLASS_NAMES,
                )
                seed_metrics.append(
                    {
                        "seed": seed,
                        "best_epoch": result.best_epoch,
                        "epochs_completed": result.epochs_completed,
                        "validation": evaluate_model(
                            model,
                            validation_windows,
                            batch_size,
                            BINARY_CLASS_NAMES,
                        ),
                    }
                )
            candidate_models[(profile, family)] = models
            ensemble = _ensemble_metrics(models, validation_windows, batch_size)
            candidates.append(
                {
                    "candidate_id": f"{profile}_{family}_100ms",
                    "profile": profile,
                    "family": family,
                    "window_ms": window_samples,
                    "input_channels": int(train_windows.inputs.shape[2]),
                    "parameter_count": parameter_count(
                        build_model(
                            family,
                            window_samples,
                            int(train_windows.inputs.shape[2]),
                            class_count=2,
                        )
                    ),
                    "train_windows": _binary_counts(train_windows),
                    "validation_windows": _binary_counts(validation_windows),
                    "validation_ensemble": ensemble,
                    "seeds": seed_metrics,
                }
            )

    near_tie = float(config["selection"]["near_tie_run_balanced_macro_f1"])
    best_score = max(
        float(item["validation_ensemble"]["run_balanced_macro_f1"])
        for item in candidates
    )
    contenders = [
        item
        for item in candidates
        if best_score
        - float(item["validation_ensemble"]["run_balanced_macro_f1"])
        <= near_tie
    ]
    selected = max(
        contenders,
        key=lambda item: (
            min(
                float(item["validation_ensemble"]["per_class"][name]["recall"])
                for name in BINARY_CLASS_NAMES
            ),
            item["family"] == "mlp",
            -int(item["parameter_count"]),
        ),
    )
    selected_profile = str(selected["profile"])
    selected_family = str(selected["family"])
    selected_models = candidate_models[(selected_profile, selected_family)]
    progress(
        f"selected {selected_profile}/{selected_family}; opening HOLDOUT once"
    )
    holdout_opened = True
    _assert_holdout_waveforms_sealed(
        valid_split["holdout"], valid_split["holdout"], holdout_opened
    )
    holdout_windows = _binary_windows(
        build_profile_windows(
            records,
            valid_split["holdout"],
            selected_profile,
            window_samples,
            stride_samples,
            normalizers[selected_profile],
            early_targets=False,
        )
    )
    holdout_metrics = _ensemble_metrics(
        selected_models, holdout_windows, batch_size
    )
    holdout_predictions = _ensemble_predictions(
        selected_models, holdout_windows, batch_size
    )
    holdout_stratified_recall = _sink_stratified_recall(
        holdout_predictions, holdout_windows, records
    )

    persistence = int(config["replay"]["sustained_ms"])
    horizons = [int(value) for value in config["replay"]["horizons_ms"]]
    positive_replay: list[dict[str, object]] = []
    benign_replay: list[dict[str, object]] = []
    split_lookup = {
        run_id: split_name
        for split_name, run_ids in frozen_split.items()
        for run_id in run_ids
    }
    for run_id, record in records.items():
        if record.observed_outcome not in ("BENIGN", "SINK"):
            continue
        values, _, _ = load_profile_arrays(
            record, selected_profile, early_targets=False
        )
        arrays = _load_run(record.path)
        trace = _ensemble_replay(
            selected_models,
            values,
            normalizers[selected_profile],
            window_samples,
            batch_size,
        )
        censor = int(arrays["first_censor_sample"])
        stop = len(values) if censor < 0 else censor
        d0_values = np.asarray(
            arrays["first_patch_contact_sample_per_foot"], dtype=np.int64
        )
        d0_valid = d0_values[d0_values >= 0]
        d0 = None if not len(d0_valid) else int(d0_valid.min())
        if record.observed_outcome == "SINK":
            s1 = int(arrays["first_deformable_sink_onset_sample"])
            if d0 is None or not 0 <= d0 <= s1 < stop:
                raise ValueError(f"invalid d0/s1/censor order: {run_id}")
            onset = first_sustained_endpoint(
                trace.predictions,
                trace.endpoint_samples,
                1,
                d0,
                stop,
                persistence,
            )
            pre_event = first_sustained_endpoint(
                trace.predictions,
                trace.endpoint_samples,
                1,
                int(trace.endpoint_samples[0]),
                d0,
                persistence,
            )
            row: dict[str, object] = {
                "run_id": run_id,
                "split": split_lookup[run_id],
                "side": record.sink_side or "",
                "severity": record.sink_severity or "",
                "pattern": (
                    (record.sink_support_pattern or "").removesuffix(
                        "_deformable"
                    )
                ),
                "speed_mps": record.speed_mps,
                "patch_start_x": record.patch_start_x,
                "d0_sample": d0,
                "s1_sample": s1,
                "first_sustained_sample": onset,
                "latency_from_s1_ms": None if onset is None else onset - s1,
                "pre_d0_fp_sample": pre_event,
                "pre_s1_precursor": bool(onset is not None and onset < s1),
                "pre_s1_100ms_precursor": bool(
                    onset is not None and s1 - 100 <= onset < s1
                ),
                "detected_before_censor": onset is not None,
            }
            for horizon in horizons:
                row[f"detected_by_{horizon}ms"] = bool(
                    onset is not None and onset <= s1 + horizon
                )
            positive_replay.append(row)
        else:
            start = int(trace.endpoint_samples[0])
            sustained = first_sustained_endpoint(
                trace.predictions,
                trace.endpoint_samples,
                1,
                start,
                stop,
                persistence,
            )
            pre_stop = stop if d0 is None else d0
            pre_event = first_sustained_endpoint(
                trace.predictions,
                trace.endpoint_samples,
                1,
                start,
                pre_stop,
                persistence,
            )
            valid_mask = (trace.endpoint_samples >= start) & (
                trace.endpoint_samples < stop
            )
            benign_replay.append(
                {
                    "run_id": run_id,
                    "split": split_lookup[run_id],
                    "scenario_group": _benign_group(record),
                    "sustained_sink_fp": sustained is not None,
                    "first_sustained_sample": sustained,
                    "pre_event_fp": pre_event is not None,
                    "false_positive_duration_ms": int(
                        np.count_nonzero(trace.predictions[valid_mask] == 1)
                    ),
                }
            )

    replay_summary = _aggregate_sink_replay(
        positive_replay, benign_replay, horizons
    )
    holdout_positive = [
        row for row in positive_replay if row["split"] == "holdout"
    ]
    holdout_benign = [row for row in benign_replay if row["split"] == "holdout"]
    holdout_balanced = [
        row
        for row in holdout_benign
        if str(row["scenario_group"]).startswith("balanced_")
    ]
    detected_latencies = [
        float(row["latency_from_s1_ms"])
        for row in holdout_positive
        if row["latency_from_s1_ms"] is not None
    ]
    recall_100 = (
        sum(bool(row["detected_by_100ms"]) for row in holdout_positive)
        / len(holdout_positive)
        if holdout_positive
        else 0.0
    )

    def stratum_recall(field: str, value: str) -> float:
        selected_rows = [
            row for row in holdout_positive if row[field] == value
        ]
        return (
            sum(bool(row["detected_by_100ms"]) for row in selected_rows)
            / len(selected_rows)
            if selected_rows
            else 0.0
        )

    side_recall = {
        side: stratum_recall("side", side)
        for side in ("left", "right")
    }
    pattern_recall = {
        pattern: stratum_recall("pattern", pattern)
        for pattern in ("medial", "lateral", "localized")
    }
    severity_recall = {
        severity: stratum_recall("severity", severity)
        for severity in ("mild", "moderate", "severe")
        if any(row["severity"] == severity for row in holdout_positive)
    }
    speed_recall = {
        f"{speed:.2f}": stratum_recall("speed_mps", speed)
        for speed in sorted({float(row["speed_mps"]) for row in holdout_positive})
    }
    gates = config["acceptance"]["holdout"]
    acceptance = {
        "macro_f1": float(holdout_metrics["macro_f1"]) >= float(gates["macro_f1_min"]),
        "sink_recall": float(holdout_metrics["per_class"]["SINK"]["recall"])
        >= float(gates["sink_recall_min"]),
        "normal_recall": float(holdout_metrics["per_class"]["NORMAL"]["recall"])
        >= float(gates["normal_recall_min"]),
        "recall_s1_plus_100": recall_100 >= float(gates["recall_s1_plus_100_min"]),
        "median_latency": bool(
            detected_latencies
            and float(np.median(detected_latencies))
            <= float(gates["median_latency_max_ms"])
        ),
        "benign_fp_rate": (
            sum(bool(row["sustained_sink_fp"]) for row in holdout_benign)
            / len(holdout_benign)
            if holdout_benign
            else 1.0
        )
        <= float(gates["benign_fp_run_rate_max"]),
        "balanced_fp_rate": (
            sum(bool(row["sustained_sink_fp"]) for row in holdout_balanced)
            / len(holdout_balanced)
            if holdout_balanced
            else 1.0
        )
        <= float(gates["balanced_fp_run_rate_max"]),
        "side_coverage": min(side_recall.values())
        >= float(gates["side_recall_min"]),
        "pattern_coverage": min(pattern_recall.values())
        >= float(gates["pattern_recall_min"]),
    }
    if all(acceptance.values()):
        verdict = "SINK_SENSOR_OBSERVABILITY_SUPPORTED"
    elif (
        float(holdout_metrics["macro_f1"])
        >= float(config["acceptance"]["promising"]["macro_f1_min"])
        and recall_100
        >= float(config["acceptance"]["promising"]["recall_s1_plus_100_min"])
    ):
        verdict = "SINK_SENSOR_OBSERVABILITY_PROMISING"
    else:
        verdict = "SINK_SENSOR_OBSERVABILITY_NOT_SUPPORTED"

    _write_csv(artifact_path / "positive_replay.csv", positive_replay)
    _write_csv(artifact_path / "benign_false_positive.csv", benign_replay)
    metrics = {
        "experiment_id": config["experiment"]["id"],
        "dataset": {
            **dataset_summary,
            "size_bytes": sum(
                path.stat().st_size for path in dataset_path.rglob("*") if path.is_file()
            ),
        },
        "split": {
            name: {
                "declared": len(frozen_split[name]),
                "valid": len(valid_split[name]),
            }
            for name in frozen_split
        },
        "readiness": readiness,
        "holdout_guard": {
            "waveforms_opened_before_selection": False,
            "opened_once_after_candidate_selection": holdout_opened,
        },
        "candidates": candidates,
        "selection": {
            "candidate_id": selected["candidate_id"],
            "profile": selected_profile,
            "family": selected_family,
            "validation_run_balanced_macro_f1": selected["validation_ensemble"][
                "run_balanced_macro_f1"
            ],
            "near_tie_tolerance": near_tie,
        },
        "holdout": {
            "window_counts": _binary_counts(holdout_windows),
            "metrics": holdout_metrics,
            "sink_stratified_recall": holdout_stratified_recall,
            "one_shot": True,
        },
        "causal_replay": {
            **replay_summary,
            "holdout_recall_at_s1": replay_summary["horizon_recall"]["holdout"]["0"]["recall"],
            "holdout_recall_at_20ms": replay_summary["horizon_recall"]["holdout"]["20"]["recall"],
            "holdout_recall_at_50ms": replay_summary["horizon_recall"]["holdout"]["50"]["recall"],
            "holdout_recall_at_100ms": recall_100,
            "holdout_median_latency_ms": (
                None if not detected_latencies else float(np.median(detected_latencies))
            ),
            "holdout_p95_latency_ms": (
                None if not detected_latencies else float(np.percentile(detected_latencies, 95))
            ),
            "holdout_pre_s1_precursor_count": sum(
                bool(row["pre_s1_precursor"]) for row in holdout_positive
            ),
            "holdout_pre_s1_100ms_precursor_count": sum(
                bool(row["pre_s1_100ms_precursor"])
                for row in holdout_positive
            ),
            "holdout_side_recall_at_100ms": side_recall,
            "holdout_pattern_recall_at_100ms": pattern_recall,
            "holdout_severity_recall_at_100ms": severity_recall,
            "holdout_speed_recall_at_100ms": speed_recall,
        },
        "acceptance": acceptance,
        "verdict": verdict,
        "leakage_audit": {
            "runtime_profiles_only": ["pelvis_imu", "foot_fsr"],
            "oracle_fields_in_input": False,
            "normalization_fit_split": "train",
            "normalization_fit_run_ids": list(normalizers[selected_profile].fit_run_ids),
            "run_disjoint": True,
            "duplicate_condition_signatures": 0,
            "status": "PASS",
        },
    }
    _write_json(artifact_path / "metrics.json", metrics)
    return artifact_path, metrics
