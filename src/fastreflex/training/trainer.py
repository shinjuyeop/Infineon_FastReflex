"""Generic deterministic training and checkpoint persistence."""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from fastreflex.dataset.loader import CLASS_NAMES, WindowSet
from fastreflex.evaluation.metrics import classification_metrics
from fastreflex.models.baselines import build_model
from fastreflex.models.checkpoint import load_checkpoint as load_frozen_checkpoint
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def validation_cross_entropy(
    model: nn.Module,
    windows: WindowSet,
    criterion: nn.Module,
    batch_size: int = 128,
) -> float:
    """Return deterministic validation loss without selecting a threshold."""
    model.eval()
    loss_total = 0.0
    sample_total = 0
    data = _loader(windows, batch_size, shuffle=False, seed=0)
    with torch.no_grad():
        for inputs, targets in data:
            loss = criterion(model(inputs), targets)
            loss_total += float(loss.detach()) * len(targets)
            sample_total += len(targets)
    if not sample_total:
        raise ValueError("validation windows must not be empty")
    return loss_total / sample_total


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
    selection_metric: str = "macro_f1",
) -> tuple[nn.Module, TrainingResult]:
    """Train one seed with an explicitly selected validation criterion."""
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
    counts = np.bincount(train_windows.targets, minlength=class_count).astype(
        np.float64
    )
    if np.any(counts == 0):
        raise ValueError("training windows must cover every class")
    weights = counts.sum() / (class_count * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = _loader(train_windows, batch_size, shuffle=True, seed=seed)
    if selection_metric not in ("macro_f1", "validation_loss"):
        raise ValueError(f"unsupported epoch selection metric: {selection_metric}")

    best_score = float("inf") if selection_metric == "validation_loss" else -1.0
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
        validation_loss = validation_cross_entropy(
            model, validation_windows, criterion, batch_size
        )
        score = (
            validation_loss
            if selection_metric == "validation_loss"
            else float(validation_metrics["macro_f1"])
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss_total / sample_total,
                "validation_macro_f1": float(validation_metrics["macro_f1"]),
                "validation_cross_entropy": validation_loss,
            }
        )
        improved = (
            score < best_score - 1.0e-12
            if selection_metric == "validation_loss"
            else score > best_score + 1.0e-12
        )
        if improved:
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
        input_channels = (
            int(first_weight.shape[1] // window_samples)
            if family == "mlp"
            else int(first_weight.shape[1])
        )
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
    """Compatibility wrapper around the inference-only checkpoint loader."""
    return load_frozen_checkpoint(path)
