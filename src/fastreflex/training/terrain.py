"""Training responsibility for the supported FSR4/MLP/50 ms Terrain model."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fastreflex.dataset.loader import sha256_file
from fastreflex.dataset.terrain import (
    TERRAIN_CLASS_NAMES,
    TerrainNormalizer,
    TerrainWindowSet,
    build_terrain_windows,
    fit_terrain_normalizer,
)
from fastreflex.evaluation.metrics import classification_metrics
from fastreflex.models.baselines import parameter_count
from fastreflex.training.trainer import (
    TrainingResult,
    save_checkpoint,
    train_model,
)


TERRAIN_PROFILE = "fsr4"
TERRAIN_FAMILY = "mlp"
TERRAIN_OBSERVATION_MS = 50
TERRAIN_SEEDS = (17, 29, 43)


@dataclass(frozen=True)
class TerrainCandidate:
    normalizer: TerrainNormalizer
    checkpoint_paths: tuple[Path, ...]
    validation_metrics: Mapping[str, object]
    record: Mapping[str, object]


def normalized_terrain_windows(
    windows: TerrainWindowSet, normalizer: TerrainNormalizer
) -> TerrainWindowSet:
    return TerrainWindowSet(
        inputs=normalizer.transform(windows.inputs),
        targets=windows.targets,
        run_ids=windows.run_ids,
        event_ids=windows.event_ids,
        feet=windows.feet,
        touchdown_samples=windows.touchdown_samples,
    )


def predict_terrain_logits(
    model: torch.nn.Module, windows: TerrainWindowSet
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(windows.inputs)).cpu().numpy()
    if logits.shape != (len(windows), 4) or not np.all(np.isfinite(logits)):
        raise ValueError("Terrain model produced invalid logits")
    return logits


def terrain_ensemble_metrics(
    models: Sequence[torch.nn.Module], windows: TerrainWindowSet
) -> tuple[dict[str, object], np.ndarray]:
    logits = np.mean(
        [predict_terrain_logits(model, windows) for model in models], axis=0
    )
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    return (
        classification_metrics(
            windows.targets,
            predictions,
            windows.run_ids,
            TERRAIN_CLASS_NAMES,
        ),
        predictions,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def train_terrain_candidate(
    dataset_path: Path,
    train_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    artifact_path: Path,
    settings: Mapping[str, object],
    progress: Callable[[str], None] = print,
) -> TerrainCandidate:
    """Train only the already-selected FSR4/MLP/50 ms candidate."""
    if any(str(row["split"]) != "train" for row in train_rows):
        raise ValueError("Terrain normalizer/training rows must be TRAIN only")
    if any(str(row["split"]) != "validation" for row in validation_rows):
        raise ValueError("Terrain evaluation rows must be VALIDATION only")
    raw_train = build_terrain_windows(
        dataset_path, train_rows, TERRAIN_PROFILE, TERRAIN_OBSERVATION_MS
    )
    normalizer = fit_terrain_normalizer(raw_train)
    train_windows = normalized_terrain_windows(raw_train, normalizer)
    validation_windows = normalized_terrain_windows(
        build_terrain_windows(
            dataset_path,
            validation_rows,
            TERRAIN_PROFILE,
            TERRAIN_OBSERVATION_MS,
        ),
        normalizer,
    )
    if min(train_windows.selected_by_class) == 0:
        raise ValueError("Terrain TRAIN events do not cover all four classes")
    if min(validation_windows.selected_by_class) == 0:
        raise ValueError("Terrain VALIDATION events do not cover all four classes")

    models: list[torch.nn.Module] = []
    results: list[TrainingResult] = []
    checkpoints: list[Path] = []
    seed_records: list[dict[str, object]] = []
    for seed in TERRAIN_SEEDS:
        progress(f"Terrain FSR4/MLP/50ms seed={seed}")
        model, result = train_model(
            TERRAIN_FAMILY,
            TERRAIN_OBSERVATION_MS,
            train_windows,  # type: ignore[arg-type]
            validation_windows,  # type: ignore[arg-type]
            seed,
            batch_size=int(settings["batch_size"]),
            max_epochs=int(settings["max_epochs"]),
            patience=int(settings["patience"]),
            learning_rate=float(settings["learning_rate"]),
            class_names=TERRAIN_CLASS_NAMES,
        )
        path = artifact_path / f"seed_{seed}.pt"
        save_checkpoint(
            path,
            model,
            TERRAIN_FAMILY,
            TERRAIN_OBSERVATION_MS,
            seed,
            result,
            input_channels=4,
            class_names=TERRAIN_CLASS_NAMES,
        )
        models.append(model)
        results.append(result)
        checkpoints.append(path)
        seed_records.append(
            {
                "seed": seed,
                "best_epoch": result.best_epoch,
                "epochs_completed": result.epochs_completed,
                "validation": result.best_validation,
            }
        )
    metrics, _ = terrain_ensemble_metrics(models, validation_windows)
    normalizer_path = artifact_path / "normalization.json"
    _write_json(normalizer_path, {**normalizer.to_dict(), "train_only": True})
    return TerrainCandidate(
        normalizer=normalizer,
        checkpoint_paths=tuple(checkpoints),
        validation_metrics=metrics,
        record={
            "profile": TERRAIN_PROFILE,
            "family": TERRAIN_FAMILY,
            "observation_ms": TERRAIN_OBSERVATION_MS,
            "input_channels": 4,
            "classes": list(TERRAIN_CLASS_NAMES),
            "seeds": seed_records,
            "parameter_count": parameter_count(models[0]),
            "normalizer_sha256": sha256_file(normalizer_path),
            "checkpoint_sha256": {
                path.name: sha256_file(path) for path in checkpoints
            },
            "validation_metrics": metrics,
            "holdout_opened": False,
            "terrain_role": "advisory_only",
        },
    )
