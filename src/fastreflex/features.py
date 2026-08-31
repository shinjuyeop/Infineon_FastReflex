"""Canonical causal feature preprocessing for the supported Hazard model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np


HAZARD_BASE_FEATURES = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_norm",
    "gyro_norm",
    "horizontal_accel_norm",
    "horizontal_gyro_norm",
)
HAZARD_TEMPORAL_TRANSFORMS = (
    "base",
    "delta_1ms",
    "delta_5ms",
    "delta_10ms",
    "causal_mean_5ms",
    "causal_mean_10ms",
    "causal_variance_5ms",
    "causal_variance_10ms",
)
HAZARD_FEATURE_DIMENSION = 80
HAZARD_FEATURE_SCHEMA_SHA256 = (
    "fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb"
)


def imu_feature_base(imu6: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the frozen ten-channel Pelvis IMU base in exact order."""
    values = np.asarray(imu6, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Pelvis IMU input must have shape [samples,6]")
    accel, gyro = values[:, :3], values[:, 3:]
    derived = np.column_stack(
        (
            np.linalg.norm(accel, axis=1),
            np.linalg.norm(gyro, axis=1),
            np.linalg.norm(accel[:, :2], axis=1),
            np.linalg.norm(gyro[:, :2], axis=1),
        )
    ).astype(np.float32)
    return np.concatenate((values, derived), axis=1), HAZARD_BASE_FEATURES


def causal_delta(values: np.ndarray, lag_ms: int) -> np.ndarray:
    """Current-minus-past delta with the unavailable causal prefix zeroed."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or lag_ms <= 0:
        raise ValueError("causal delta expects [samples,features] and a positive lag")
    result = np.zeros_like(array)
    if lag_ms < len(array):
        result[lag_ms:] = array[lag_ms:] - array[:-lag_ms]
    return result


def causal_rolling(
    values: np.ndarray, width_ms: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return trailing mean and variance without accessing a future sample."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or width_ms <= 0:
        raise ValueError("causal rolling expects [samples,features] and positive width")
    prefix = np.vstack((np.zeros((1, array.shape[1])), np.cumsum(array, axis=0)))
    square = np.vstack(
        (np.zeros((1, array.shape[1])), np.cumsum(array * array, axis=0))
    )
    ends = np.arange(1, len(array) + 1)
    starts = np.maximum(0, ends - int(width_ms))
    counts = (ends - starts)[:, None]
    mean = (prefix[ends] - prefix[starts]) / counts
    variance = (square[ends] - square[starts]) / counts - mean * mean
    return mean.astype(np.float32), np.maximum(variance, 0.0).astype(np.float32)


def extract_hazard_features(imu6: np.ndarray) -> np.ndarray:
    """Build the selected 80D causal representation, preserving exact math."""
    base, _ = imu_feature_base(imu6)
    mean5, variance5 = causal_rolling(base, 5)
    mean10, variance10 = causal_rolling(base, 10)
    result = np.concatenate(
        (
            base,
            causal_delta(base, 1),
            causal_delta(base, 5),
            causal_delta(base, 10),
            mean5,
            mean10,
            variance5,
            variance10,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if result.shape != (len(base), HAZARD_FEATURE_DIMENSION):
        raise ValueError("Hazard features must have shape [samples,80]")
    if not np.all(np.isfinite(result)):
        raise ValueError("Hazard feature tensor is nonfinite")
    return result


def hazard_feature_schema() -> tuple[str, ...]:
    """Return the exact selected feature names in tensor order."""
    return tuple(
        f"pelvis_{transform}_{name}"
        for transform in HAZARD_TEMPORAL_TRANSFORMS
        for name in HAZARD_BASE_FEATURES
    )


def feature_schema_hash(schema: Sequence[str] | None = None) -> str:
    """Hash a schema with the frozen canonical JSON representation."""
    selected = hazard_feature_schema() if schema is None else tuple(schema)
    return hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


if feature_schema_hash() != HAZARD_FEATURE_SCHEMA_SHA256:
    raise RuntimeError("canonical Hazard feature order differs from the frozen schema")
