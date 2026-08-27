"""Canonical evaluation metrics and analysis."""

from .metrics import classification_metrics, confusion_matrix, metrics_from_confusion

__all__ = (
    "classification_metrics",
    "confusion_matrix",
    "metrics_from_confusion",
    "run_time_to_separation",
)


def __getattr__(name: str):
    if name == "run_time_to_separation":
        from .time_to_separation import run_time_to_separation

        return run_time_to_separation
    raise AttributeError(name)
