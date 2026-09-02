"""Runtime inference for the supported advisory-only Terrain candidate."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fastreflex.dataset.hazard import (
    EVENT_TYPE_BOTH,
    EVENT_TYPE_SLIP,
    EVENT_TYPE_SUPPORT,
    HazardRun,
)
from fastreflex.dataset.loader import sha256_file
from fastreflex.dataset.terrain import TERRAIN_CLASS_NAMES, build_touchdown_event_rows
from fastreflex.models.checkpoint import load_checkpoint
from fastreflex.simulation.g1 import SimulationResult


UNKNOWN = 0
CONCRETE = 1
MARBLE = 2
ICE = 3
SAND = 4
TERRAIN_STATE_NAMES = ("UNKNOWN", "CONCRETE", "MARBLE", "ICE", "SAND")
TERRAIN_PREDICTION_TO_STATE = {
    "CONCRETE": CONCRETE,
    "MARBLE": MARBLE,
    "ICE": ICE,
    "SAND": SAND,
}
TERRAIN_NORMALIZER_SHA256 = (
    "2c551181fbd23080c0e3008be8484bd0ee4139e594041ba6e46ab5b36ffb05de"
)
TERRAIN_CHECKPOINT_SHA256 = {
    "seed_17.pt": "21b0d122b4200a96390b700f741d6b35a4e72226e61d204a420d23e086e1f628",
    "seed_29.pt": "de6a55d35531dfa96d73e86bb8b5596ead5e41809aba539dd83e32b473cd0d66",
    "seed_43.pt": "465803f40fff371b9de2ca0ecaf7d9d41717d2be6b5cd33fe031d6d9ba237b31",
}


@dataclass(frozen=True)
class TerrainPrediction:
    """One causal Terrain output with touchdown provenance."""

    class_id: int
    probabilities: np.ndarray
    prediction_timestamp: int
    touchdown_foot: str


@dataclass(frozen=True)
class TerrainTrace:
    """Held prediction state; exact terrain truth is not exposed."""

    state: np.ndarray
    update_samples: np.ndarray
    prediction_ids: np.ndarray
    prediction_probabilities: np.ndarray
    first_target_valid_sample: int | None
    clean_event_count: int
    prediction_feet: np.ndarray | None = None
    prediction_true_ids: np.ndarray | None = None


def terrain_predictions(trace: TerrainTrace) -> tuple[TerrainPrediction, ...]:
    """Expose aligned prediction values and their causal update timestamps."""
    if trace.prediction_feet is None:
        raise ValueError("Terrain prediction foot provenance is unavailable")
    if not (
        len(trace.update_samples)
        == len(trace.prediction_ids)
        == len(trace.prediction_probabilities)
        == len(trace.prediction_feet)
    ):
        raise ValueError("Terrain prediction provenance arrays must align")
    return tuple(
        TerrainPrediction(
            class_id=int(class_id),
            probabilities=np.asarray(probabilities, dtype=np.float32),
            prediction_timestamp=int(timestamp),
            touchdown_foot=str(foot).upper(),
        )
        for timestamp, class_id, probabilities, foot in zip(
            trace.update_samples,
            trace.prediction_ids,
            trace.prediction_probabilities,
            trace.prediction_feet,
        )
    )


def load_frozen_terrain_candidate(
    model_path: Path,
) -> tuple[list[torch.nn.Module], np.ndarray, np.ndarray]:
    """Load the protected FSR4/MLP/50 ms three-seed ensemble."""
    checkpoints = sorted(model_path.glob("seed_*.pt"))
    if len(checkpoints) != 3:
        raise ValueError("frozen Terrain ensemble must contain three checkpoints")
    models: list[torch.nn.Module] = []
    for checkpoint in checkpoints:
        model, metadata = load_checkpoint(checkpoint)
        if (
            metadata["family"] != "mlp"
            or int(metadata["window_samples"]) != 50
            or int(metadata["input_channels"]) != 4
            or tuple(metadata["class_names"]) != TERRAIN_CLASS_NAMES
        ):
            raise ValueError("frozen Terrain checkpoint contract changed")
        models.append(model)
    with (model_path / "normalization.json").open("r", encoding="utf-8") as stream:
        normalizer = json.load(stream)
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    if mean.shape != (4,) or std.shape != (4,) or np.any(std <= 0.0):
        raise ValueError("frozen Terrain normalizer contract changed")
    return models, mean, std


def verify_supported_terrain_candidate(repository_root: Path) -> dict[str, object]:
    """Verify the protected FSR4/MLP/50 ms advisory artifacts read-only."""
    model_path = (
        repository_root.resolve()
        / "artifacts/runs/20260828_terrain_rebuild_sensor_ablation/selected_models"
    )
    actual_normalizer = sha256_file(model_path / "normalization.json")
    actual_checkpoints = {
        name: sha256_file(model_path / name) for name in TERRAIN_CHECKPOINT_SHA256
    }
    if actual_normalizer != TERRAIN_NORMALIZER_SHA256:
        raise RuntimeError("protected Terrain normalizer changed")
    if actual_checkpoints != TERRAIN_CHECKPOINT_SHA256:
        raise RuntimeError("protected Terrain checkpoints changed")
    models, mean, std = load_frozen_terrain_candidate(model_path)
    return {
        "passed": True,
        "verdict": "TERRAIN_RECOGNITION_SUPPORTED",
        "input": "FSR4",
        "model_family": "mlp",
        "observation_ms": 50,
        "classes": list(TERRAIN_CLASS_NAMES),
        "normalizer_sha256": actual_normalizer,
        "checkpoint_sha256": actual_checkpoints,
        "ensemble_size": len(models),
        "normalizer_shape": list(mean.shape),
        "normalizer_finite": bool(
            np.all(np.isfinite(mean)) and np.all(np.isfinite(std))
        ),
        "advisory_only": True,
        "hazard_gate": False,
    }


def terrain_fsr4_window(
    foot_fsr8: np.ndarray, touchdown_sample: int, foot: str
) -> np.ndarray:
    """Return the exact 50 ms, one-foot FSR4 observation window."""
    values = np.asarray(foot_fsr8, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 8 or np.any(values < 0.0):
        raise ValueError("Terrain FSR input must be nonnegative [samples,8]")
    if foot not in ("left", "right"):
        raise ValueError("Terrain touchdown foot must be left or right")
    first = int(touchdown_sample)
    last = first + 50
    if first < 0 or last > len(values):
        raise ValueError("Terrain observation exceeds the available causal trace")
    side = 0 if foot == "left" else 1
    return values[first:last, side * 4 : (side + 1) * 4]


def predict_terrain_window(
    window: np.ndarray,
    models: Sequence[torch.nn.Module],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Run the frozen ensemble without changing its normalization or averaging."""
    values = np.asarray(window, dtype=np.float32)
    if values.shape != (50, 4):
        raise ValueError("Terrain model input must have shape [50,4]")
    normal = ((values - mean) / std).astype(np.float32)[None]
    tensor = torch.from_numpy(normal)
    with torch.no_grad():
        probabilities = np.mean(
            [
                torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                for model in models
            ],
            axis=0,
        ).astype(np.float32)
    return int(np.argmax(probabilities)), probabilities


def replay_terrain(
    result: SimulationResult,
    run: HazardRun,
    models: Sequence[torch.nn.Module],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    deployment_scheme: str = "left_only",
) -> TerrainTrace:
    """Classify clean touchdown windows and hold the latest advisory state."""
    if result.exact_terrain_contact is None or result.runtime.foot_fsr is None:
        raise ValueError("Terrain replay requires exact contact and FSR8")
    return replay_terrain_arrays(
        result.runtime.timestamp_us,
        result.exact_terrain_contact,
        result.runtime.foot_fsr,
        run,
        models,
        mean,
        std,
        deployment_scheme=deployment_scheme,
    )


def replay_terrain_arrays(
    timestamp_us: np.ndarray,
    exact_terrain_contact: np.ndarray,
    foot_fsr8: np.ndarray,
    run: HazardRun,
    models: Sequence[torch.nn.Module],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    deployment_scheme: str = "left_only",
) -> TerrainTrace:
    """Replay Terrain from one already-loaded scientific run payload."""
    rows = build_touchdown_event_rows(
        run.run_id,
        run.split,
        run.source_terrain,
        run.target_terrain,
        timestamp_us,
        exact_terrain_contact,
        run.fall_sample_diagnostic,
        run.event_type in (EVENT_TYPE_SLIP, EVENT_TYPE_BOTH),
        run.event_type in (EVENT_TYPE_SUPPORT, EVENT_TYPE_BOTH),
    )
    eligible = [row for row in rows if bool(row["window_50ms_valid"])]
    if deployment_scheme == "left_only":
        eligible = [row for row in eligible if row["foot"] == "left"]
    elif deployment_scheme != "bilateral_shared":
        raise ValueError("unsupported frozen Terrain deployment scheme")

    state = np.full(len(run.timestamp_us), UNKNOWN, dtype=np.int8)
    updates: list[int] = []
    prediction_ids: list[int] = []
    probability_rows: list[np.ndarray] = []
    feet: list[str] = []
    true_ids: list[int] = []
    current = UNKNOWN
    cursor = 0
    for row in sorted(eligible, key=lambda value: int(value["touchdown_sample"])):
        touchdown = int(row["touchdown_sample"])
        update = touchdown + 50
        if update >= len(state) or update >= run.censor_sample:
            continue
        window = terrain_fsr4_window(
            foot_fsr8, touchdown, str(row["foot"])
        )
        prediction, probabilities = predict_terrain_window(
            window, models, mean, std
        )
        state[cursor:update] = current
        current = TERRAIN_PREDICTION_TO_STATE[TERRAIN_CLASS_NAMES[prediction]]
        cursor = update
        updates.append(update)
        prediction_ids.append(prediction)
        probability_rows.append(probabilities)
        feet.append(str(row["foot"]).upper())
        true_ids.append(int(row["terrain_class_id"]))
    state[cursor:] = current
    target_state = TERRAIN_PREDICTION_TO_STATE[run.target_terrain.upper()]
    valid = np.flatnonzero(
        (state == target_state)
        & (np.arange(len(state), dtype=np.int64) >= run.first_contact_sample)
    )
    return TerrainTrace(
        state=state,
        update_samples=np.asarray(updates, dtype=np.int64),
        prediction_ids=np.asarray(prediction_ids, dtype=np.int8),
        prediction_probabilities=(
            np.stack(probability_rows).astype(np.float32)
            if probability_rows
            else np.empty((0, 4), dtype=np.float32)
        ),
        first_target_valid_sample=None if not len(valid) else int(valid[0]),
        clean_event_count=len(eligible),
        prediction_feet=np.asarray(feet, dtype="<U5"),
        prediction_true_ids=np.asarray(true_ids, dtype=np.int8),
    )


def refine_hazard_cause(reflex_required: bool, terrain_state: int) -> str:
    """Refine cause after the independent Hazard decision; never gate it."""
    if not reflex_required:
        return "NORMAL"
    if int(terrain_state) == ICE:
        return "SLIP_RISK"
    if int(terrain_state) == SAND:
        return "SUPPORT_RISK"
    return "GENERIC_DISTURBANCE"
