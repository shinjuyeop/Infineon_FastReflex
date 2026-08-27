"""Dependency-light classification metrics with run-balanced summaries."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from fastreflex.dataset.loader import CLASS_NAMES


def confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, class_count: int = 3
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.shape != predictions.shape:
        raise ValueError("target and prediction shapes differ")
    if np.any((targets < 0) | (targets >= class_count)):
        raise ValueError("target is outside the class range")
    if np.any((predictions < 0) | (predictions >= class_count)):
        raise ValueError("prediction is outside the class range")
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (3, 3):
        raise ValueError("expected a 3x3 confusion matrix")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(3, dtype=np.float64),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(3, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(3, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    total = int(matrix.sum())
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "support": total,
    }


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    run_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    result = metrics_from_confusion(confusion_matrix(targets, predictions))
    if run_ids is None:
        return result
    sources = np.asarray(list(run_ids), dtype=str)
    if sources.shape != targets.shape:
        raise ValueError("run_ids shape differs from targets")
    per_run: dict[str, dict[str, object]] = {}
    run_accuracies: list[float] = []
    recalls_by_class: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for run_id in sorted(set(sources)):
        mask = sources == run_id
        run_matrix = confusion_matrix(targets[mask], predictions[mask])
        run_metrics = metrics_from_confusion(run_matrix)
        run_accuracies.append(float(run_metrics["accuracy"]))
        for class_id in range(3):
            class_support = int(run_matrix[class_id].sum())
            if class_support:
                recalls_by_class[class_id].append(
                    float(run_matrix[class_id, class_id] / class_support)
                )
        per_run[run_id] = {
            "accuracy": run_metrics["accuracy"],
            "support": run_metrics["support"],
            "confusion_matrix": run_metrics["confusion_matrix"],
        }
    balanced_class_recall = {
        CLASS_NAMES[class_id]: (
            float(np.mean(values)) if values else 0.0
        )
        for class_id, values in recalls_by_class.items()
    }
    failure_runs = sorted(
        (
            {
                "run_id": run_id,
                "accuracy": float(values["accuracy"]),
                "errors": int(
                    values["support"]
                    - np.trace(np.asarray(values["confusion_matrix"], dtype=np.int64))
                ),
            }
            for run_id, values in per_run.items()
            if float(values["accuracy"]) < 1.0
        ),
        key=lambda item: (item["accuracy"], item["run_id"]),
    )
    result.update(
        {
            "run_balanced_accuracy": float(np.mean(run_accuracies)),
            "run_balanced_per_class_recall": balanced_class_recall,
            "run_balanced_macro_recall": float(
                np.mean(list(balanced_class_recall.values()))
            ),
            "per_run": per_run,
            "failure_runs": failure_runs,
        }
    )
    return result
