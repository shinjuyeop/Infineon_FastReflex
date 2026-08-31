"""Small shared tensor, normalization, and integrity types."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_NAMES = ("NORMAL", "HAZARD_REFLEX_REQUIRED")


@dataclass(frozen=True)
class Normalizer:
    """Per-channel z-score parameters with fit provenance."""

    mean: np.ndarray
    std: np.ndarray
    sample_count: int
    fit_run_ids: tuple[str, ...]
    epsilon: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "per_channel_zscore",
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "sample_count": self.sample_count,
            "fit_run_ids": list(self.fit_run_ids),
            "epsilon": self.epsilon,
        }


@dataclass(frozen=True)
class WindowSet:
    """Materialized causal windows and their source provenance."""

    inputs: np.ndarray
    targets: np.ndarray
    run_ids: np.ndarray
    endpoint_samples: np.ndarray
    available_by_class: tuple[int, ...]

    @property
    def selected_by_class(self) -> tuple[int, ...]:
        counts = np.bincount(self.targets, minlength=len(self.available_by_class))
        return tuple(int(value) for value in counts[: len(self.available_by_class)])

    def __len__(self) -> int:
        return int(self.targets.shape[0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
