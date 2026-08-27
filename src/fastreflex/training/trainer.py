"""Deterministic training for the small raw-IMU classification baselines."""

from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from fastreflex.dataset.loader import (
    CLASS_NAMES,
    WindowSet,
    build_windows,
    fit_normalizer,
    load_manifest,
    sha256_file,
    validate_split,
)
from fastreflex.evaluation.analysis import run_raw_sanity
from fastreflex.evaluation.metrics import classification_metrics
from fastreflex.models.baselines import build_model, parameter_count


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    epochs_completed: int
    best_validation: dict[str, object]
    history: tuple[dict[str, float], ...]


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _loader(
    windows: WindowSet,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(windows.inputs),
        torch.from_numpy(windows.targets),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def predict_model(
    model: nn.Module,
    windows: WindowSet,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    data = _loader(windows, batch_size, shuffle=False, seed=0)
    with torch.no_grad():
        for inputs, _ in data:
            predictions.append(model(inputs).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64, copy=False)


def evaluate_model(
    model: nn.Module,
    windows: WindowSet,
    batch_size: int = 128,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, object]:
    predictions = predict_model(model, windows, batch_size)
    return classification_metrics(
        windows.targets, predictions, windows.run_ids, class_names
    )


def train_model(
    family: str,
    window_samples: int,
    train_windows: WindowSet,
    validation_windows: WindowSet,
    seed: int,
    batch_size: int = 128,
    max_epochs: int = 40,
    patience: int = 6,
    learning_rate: float = 1.0e-3,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> tuple[nn.Module, TrainingResult]:
    """Train one seed, selecting epochs by validation macro F1."""
    set_deterministic(seed)
    torch.set_num_threads(1)
    if train_windows.inputs.ndim != 3:
        raise ValueError("training inputs must have shape [windows,time,channels]")
    input_channels = int(train_windows.inputs.shape[2])
    if validation_windows.inputs.shape[2] != input_channels:
        raise ValueError("training and validation sensor channel counts differ")
    class_count = len(class_names)
    model = build_model(
        family, window_samples, input_channels, class_count=class_count
    )
    counts = np.bincount(
        train_windows.targets, minlength=class_count
    ).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("training windows must cover every class")
    weights = counts.sum() / (class_count * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = _loader(train_windows, batch_size, shuffle=True, seed=seed)

    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, object] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_total = 0.0
        sample_total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(targets)
            sample_total += len(targets)
        validation_metrics = evaluate_model(
            model, validation_windows, batch_size, class_names
        )
        score = float(validation_metrics["macro_f1"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss_total / sample_total,
                "validation_macro_f1": score,
            }
        )
        if score > best_score + 1.0e-12:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_validation = validation_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None or best_validation is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, TrainingResult(
        best_epoch=best_epoch,
        epochs_completed=len(history),
        best_validation=best_validation,
        history=tuple(history),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    family: str,
    window_samples: int,
    seed: int,
    result: TrainingResult,
    input_channels: int | None = None,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> None:
    if input_channels is None:
        first_weight = next(model.parameters())
        if family == "mlp":
            input_channels = int(first_weight.shape[1] // window_samples)
        else:
            input_channels = int(first_weight.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "fastreflex_raw_imu_baseline",
            "family": family,
            "window_samples": window_samples,
            "input_channels": input_channels,
            "class_names": list(class_names),
            "seed": seed,
            "best_epoch": result.best_epoch,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "fastreflex_raw_imu_baseline":
        raise ValueError("unsupported checkpoint format")
    model = build_model(
        checkpoint["family"],
        int(checkpoint["window_samples"]),
        int(checkpoint.get("input_channels", 6)),
        class_count=len(checkpoint.get("class_names", CLASS_NAMES)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = {
        key: checkpoint[key]
        for key in ("family", "window_samples", "seed", "best_epoch")
    }
    metadata["input_channels"] = int(checkpoint.get("input_channels", 6))
    metadata["class_names"] = list(checkpoint.get("class_names", CLASS_NAMES))
    return model, metadata


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _class_count_dict(values: tuple[int, int, int]) -> dict[str, int]:
    return {name: int(values[index]) for index, name in enumerate(CLASS_NAMES)}


def _aggregate_candidate(
    family: str,
    window_ms: int,
    parameters: int,
    seed_results: list[dict[str, object]],
    train_windows: WindowSet,
    validation_windows: WindowSet,
) -> dict[str, object]:
    macro_f1 = np.asarray(
        [result["validation"]["macro_f1"] for result in seed_results],
        dtype=np.float64,
    )
    recall_values = {
        name: np.asarray(
            [
                result["validation"]["per_class"][name]["recall"]
                for result in seed_results
            ],
            dtype=np.float64,
        )
        for name in CLASS_NAMES
    }
    mean_recall = {
        name: float(values.mean()) for name, values in recall_values.items()
    }
    return {
        "candidate_id": f"{family}_{window_ms}ms",
        "family": family,
        "window_ms": window_ms,
        "parameter_count": parameters,
        "train_windows_available": _class_count_dict(
            train_windows.available_by_class
        ),
        "train_windows_selected": _class_count_dict(
            train_windows.selected_by_class
        ),
        "validation_windows": _class_count_dict(
            validation_windows.selected_by_class
        ),
        "validation_macro_f1_mean": float(macro_f1.mean()),
        "validation_macro_f1_std": float(macro_f1.std()),
        "validation_per_class_recall_mean": mean_recall,
        "validation_per_class_recall_std": {
            name: float(values.std()) for name, values in recall_values.items()
        },
        "minimum_mean_per_class_recall": min(mean_recall.values()),
        "seeds": seed_results,
    }


def _select_candidate(
    candidates: list[dict[str, object]], near_tie: float
) -> tuple[dict[str, object], dict[str, object]]:
    best_mean = max(float(item["validation_macro_f1_mean"]) for item in candidates)
    contenders = [
        item
        for item in candidates
        if best_mean - float(item["validation_macro_f1_mean"]) <= near_tie
    ]
    selected = max(
        contenders,
        key=lambda item: (
            float(item["minimum_mean_per_class_recall"]),
            -int(item["window_ms"]),
            -int(item["parameter_count"]),
            item["family"] == "mlp",
        ),
    )
    reason = {
        "best_validation_macro_f1_mean": best_mean,
        "near_tie_tolerance": near_tie,
        "near_tie_candidate_ids": [item["candidate_id"] for item in contenders],
        "rule": (
            "within the configured macro-F1 near-tie band, maximize minimum mean "
            "class recall, then prefer shorter window, fewer parameters, and MLP"
        ),
    }
    return selected, reason


def run_first_classification_poc(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Run the bounded four-candidate experiment and one selected holdout evaluation."""
    config_path = config_path.resolve()
    repository_root = repository_root.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["experiment"]["id"] != "FIRST_CLASSIFICATION_POC":
        raise ValueError("unsupported training experiment")
    dataset_path = (repository_root / config["dataset"]["path"]).resolve()
    artifact_path = (repository_root / config["artifacts"]["path"]).resolve()
    for path, name in ((dataset_path, "dataset"), (artifact_path, "artifact")):
        try:
            path.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(f"{name} path must remain inside repository") from exc
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite experiment artifacts: {artifact_path}")
    expected_manifest_sha = config["dataset"]["manifest_sha256"]
    actual_manifest_sha = sha256_file(dataset_path / "manifest.csv")
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError("dataset manifest SHA-256 mismatch")

    records = load_manifest(dataset_path)
    split = {
        name: tuple(config["split"][name])
        for name in ("train", "validation", "holdout")
    }
    split_counts = validate_split(
        records,
        split,
        expected_outcome_counts={
            "train": {"BENIGN": 10, "SLIP": 5, "SINK": 5},
            "validation": {"BENIGN": 3, "SLIP": 1, "SINK": 2},
            "holdout": {"BENIGN": 3, "SLIP": 2, "SINK": 2},
        },
    )
    artifact_path.mkdir(parents=True, exist_ok=True)
    split_document = {
        "dataset_id": config["dataset"]["dataset_id"],
        "manifest_sha256": actual_manifest_sha,
        "frozen_before_signal_analysis": bool(
            config["split"]["frozen_before_signal_analysis"]
        ),
        "selection_basis": config["split"]["selection_basis"],
        "run_ids": {name: list(values) for name, values in split.items()},
        "run_counts": {name: len(values) for name, values in split.items()},
        "observed_outcome_counts": split_counts,
        "pairwise_intersection_counts": {
            "train_validation": len(set(split["train"]) & set(split["validation"])),
            "train_holdout": len(set(split["train"]) & set(split["holdout"])),
            "validation_holdout": len(
                set(split["validation"]) & set(split["holdout"])
            ),
        },
        "excluded_manifest_runs": sorted(
            run_id
            for run_id, record in records.items()
            if record.observed_outcome not in ("BENIGN", "SLIP", "SINK")
        ),
    }
    _write_json(artifact_path / "split.json", split_document)
    progress("split frozen and validated")

    normalizer = fit_normalizer(
        records,
        split["train"],
        epsilon=float(config["normalization"]["epsilon"]),
    )
    _write_json(artifact_path / "normalization.json", normalizer.to_dict())
    progress("train-only normalization fitted")

    valid_run_ids = tuple(
        run_id
        for split_name in ("train", "validation", "holdout")
        for run_id in split[split_name]
    )
    raw_sanity = run_raw_sanity(
        records, valid_run_ids, artifact_path / "plots"
    )
    _write_json(artifact_path / "raw_sanity.json", raw_sanity)
    progress("raw IMU sanity and plots complete")

    stride_samples = int(config["windowing"]["stride_ms"])
    train_cap = int(config["windowing"]["train_max_windows_per_run_class"])
    seeds = [int(value) for value in config["training"]["seeds"]]
    batch_size = int(config["training"]["batch_size"])
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stopping"]["patience"])
    learning_rate = float(config["training"]["learning_rate"])
    candidates: list[dict[str, object]] = []
    window_cache: dict[int, tuple[WindowSet, WindowSet]] = {}
    checkpoint_paths: dict[tuple[str, int, int], Path] = {}
    for window_ms in config["windowing"]["candidates_ms"]:
        window_samples = int(window_ms)
        train_windows = build_windows(
            records,
            split["train"],
            window_samples,
            stride_samples,
            normalizer,
            cap_per_run_class=train_cap,
        )
        validation_windows = build_windows(
            records,
            split["validation"],
            window_samples,
            stride_samples,
            normalizer,
            cap_per_run_class=None,
        )
        window_cache[window_samples] = (train_windows, validation_windows)
        for family in ("mlp", "gru"):
            parameters = parameter_count(build_model(family, window_samples))
            seed_results: list[dict[str, object]] = []
            for seed in seeds:
                progress(f"training {family} {window_ms} ms seed {seed}")
                model, result = train_model(
                    family,
                    window_samples,
                    train_windows,
                    validation_windows,
                    seed,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    learning_rate=learning_rate,
                )
                checkpoint_path = (
                    artifact_path
                    / "checkpoints"
                    / f"{family}_{window_ms}ms_seed_{seed}.pt"
                )
                save_checkpoint(
                    checkpoint_path,
                    model,
                    family,
                    window_samples,
                    seed,
                    result,
                )
                checkpoint_paths[(family, window_samples, seed)] = checkpoint_path
                seed_results.append(
                    {
                        "seed": seed,
                        "best_epoch": result.best_epoch,
                        "epochs_completed": result.epochs_completed,
                        "validation": result.best_validation,
                        "history": list(result.history),
                    }
                )
            candidates.append(
                _aggregate_candidate(
                    family,
                    window_samples,
                    parameters,
                    seed_results,
                    train_windows,
                    validation_windows,
                )
            )

    selected, selection_reason = _select_candidate(
        candidates, float(config["selection"]["near_tie_macro_f1"])
    )
    selected_family = str(selected["family"])
    selected_window = int(selected["window_ms"])
    holdout_seed = int(config["training"]["holdout_seed"])
    progress(
        f"selected {selected['candidate_id']}; opening holdout once with seed {holdout_seed}"
    )
    holdout_window_counts: dict[str, dict[str, int]] = {}
    selected_holdout: WindowSet | None = None
    for window_ms in config["windowing"]["candidates_ms"]:
        holdout_windows = build_windows(
            records,
            split["holdout"],
            int(window_ms),
            stride_samples,
            normalizer,
            cap_per_run_class=None,
        )
        holdout_window_counts[f"{window_ms}ms"] = _class_count_dict(
            holdout_windows.selected_by_class
        )
        if int(window_ms) == selected_window:
            selected_holdout = holdout_windows
    if selected_holdout is None:
        raise RuntimeError("selected holdout windows were not materialized")
    model, checkpoint_metadata = load_checkpoint(
        checkpoint_paths[(selected_family, selected_window, holdout_seed)]
    )
    holdout_metrics = evaluate_model(model, selected_holdout, batch_size)
    confusion = np.asarray(holdout_metrics["confusion_matrix"], dtype=np.int64)
    with (artifact_path / "confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *CLASS_NAMES])
        for class_id, name in enumerate(CLASS_NAMES):
            writer.writerow([name, *confusion[class_id].tolist()])

    window_counts = {
        f"{window_ms}ms": {
            "train_available": _class_count_dict(
                window_cache[int(window_ms)][0].available_by_class
            ),
            "train_selected_after_cap": _class_count_dict(
                window_cache[int(window_ms)][0].selected_by_class
            ),
            "validation": _class_count_dict(
                window_cache[int(window_ms)][1].selected_by_class
            ),
            "holdout": holdout_window_counts[f"{window_ms}ms"],
        }
        for window_ms in config["windowing"]["candidates_ms"]
    }
    metrics = {
        "experiment_id": config["experiment"]["id"],
        "dataset_id": config["dataset"]["dataset_id"],
        "manifest_sha256": actual_manifest_sha,
        "window_counts": window_counts,
        "candidates": candidates,
        "selection": {
            "candidate_id": selected["candidate_id"],
            "family": selected_family,
            "window_ms": selected_window,
            "parameter_count": selected["parameter_count"],
            "reason": selection_reason,
        },
        "holdout": {
            "access_count": 1,
            "evaluated_candidate_id": selected["candidate_id"],
            "seed": holdout_seed,
            "checkpoint": checkpoint_metadata,
            "metrics": holdout_metrics,
        },
    }
    _write_json(artifact_path / "metrics.json", metrics)
    progress("holdout evaluation and artifact write complete")
    return artifact_path, metrics
