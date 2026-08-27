"""Fixed IMU6/FSR8/Fusion14 early-target observability ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib
import numpy as np
import yaml

from fastreflex.dataset.collector import validate_dataset
from fastreflex.dataset.loader import (
    CLASS_NAMES,
    Normalizer,
    build_profile_windows,
    fit_profile_normalizer,
    load_manifest,
    load_profile_arrays,
    sha256_file,
    validate_split,
)
from fastreflex.evaluation.metrics import classification_metrics
from fastreflex.evaluation.time_to_separation import (
    audit_false_positives,
    extract_event_samples,
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
    per_run: dict[str, object] = {}
    for run_id, record in records.items():
        arrays = _load_run(record.path)
        fsr = np.asarray(arrays["foot_fsr"], dtype=np.float32)
        physical = np.asarray(arrays["physical_contact"], dtype=bool)
        loaded = np.asarray(arrays["loaded_contact"], dtype=bool)
        touchdown = np.asarray(arrays["touchdown"], dtype=bool)
        total_samples += len(fsr)
        nonfinite += int((~np.isfinite(fsr)).sum())
        negative += int((fsr < 0.0).sum())
        side_load = fsr.reshape(-1, 2, 4).sum(axis=2)
        airborne_nonzero += int(np.count_nonzero(side_load[~physical] != 0.0))
        loaded_nonpositive += int(np.count_nonzero(side_load[loaded] <= 0.0))
        quadrant_sum += fsr.sum(axis=0, dtype=np.float64)
        touchdown_loads.extend(side_load[touchdown].tolist())
        falling = loaded[:-1] & ~loaded[1:]
        unload_loads.extend(side_load[1:][falling].tolist())
        selected = side_load[loaded]
        group = (
            _scenario_group(run_id)
            if record.observed_outcome == "BENIGN"
            else record.observed_outcome.lower()
        )
        loaded_values.setdefault(group, []).append(selected)
        per_run[run_id] = {
            "outcome": record.observed_outcome,
            "loaded_total_mean_n": float(selected.mean()) if len(selected) else None,
            "loaded_total_p95_n": float(np.percentile(selected, 95)) if len(selected) else None,
            "left_right_loaded_mean_n": [
                float(side_load[:, side][loaded[:, side]].mean())
                if loaded[:, side].any() else None
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
