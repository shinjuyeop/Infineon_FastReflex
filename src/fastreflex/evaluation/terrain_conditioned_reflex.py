"""Terrain-gated continuous Slip and support-reflex detector development.

The frozen Terrain ensemble is the only source of branch state.  Exact
simulator terrain contact is used solely to schedule the same clean touchdown
events used by the frozen Terrain study; it is never copied into a detector
input or used to override a model prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from fastreflex.dataset.loader import Normalizer, WindowSet
from fastreflex.dataset.terrain import (
    TERRAIN_CLASS_NAMES,
    build_touchdown_event_rows,
    validate_foot_imu_observer_parity,
)
from fastreflex.evaluation.reflex_event import (
    EVENT_CLASS_NAMES,
    EVENT_TYPE_BOTH,
    EVENT_TYPE_SLIP,
    EVENT_TYPE_SUPPORT,
    EventHoldoutGuard,
    EventRun,
    _hard_control_outcome,
    _load_yaml,
    _reduce_simulation,
    _write_json,
    generate_event_specifications,
    load_event_runs,
)
from fastreflex.evaluation.stability_temporal import (
    _file_sha256,
    _protected_hashes,
)
from fastreflex.evaluation.transition_scenarios import (
    VALID_OUTCOMES,
    classify_scenario_outcome,
    fusion_regression,
    transition_simulation_config,
)
from fastreflex.models.baselines import build_model, parameter_count
from fastreflex.simulation.g1 import (
    SimulationResult,
    load_simulation_config,
    run_simulation,
)
from fastreflex.training.trainer import (
    load_checkpoint,
    save_checkpoint,
    train_model,
)


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
BRANCH_STATE = {"slip": ICE, "support": SAND}
BRANCH_TARGET = {"slip": "ice", "support": "sand"}
BRANCH_EVENT_TYPES = {
    "slip": (EVENT_TYPE_SLIP, EVENT_TYPE_BOTH),
    "support": (EVENT_TYPE_SUPPORT, EVENT_TYPE_BOTH),
}
PHASE_A_CANDIDATES = {
    "slip": {
        "S1": ("pelvis_imu6",),
        "S2": ("pelvis_imu6", "fsr8"),
        "S3": ("fsr8",),
    },
    "support": {
        "P1": ("fsr8",),
        "P2": ("pelvis_imu6", "fsr8"),
        "P3": ("pelvis_imu6",),
    },
}
PHASE_B_CANDIDATES = {
    "slip": {
        "SF1": ("foot_imu12",),
        "SF2": ("foot_imu12", "fsr8"),
        "SF3": ("foot_imu12", "pelvis_imu6"),
    },
    "support": {
        "PF1": ("fsr8", "foot_imu12"),
        "PF2": ("foot_imu12",),
        "PF3": ("fsr8", "foot_imu12", "pelvis_imu6"),
    },
}


@dataclass(frozen=True)
class TerrainPrediction:
    """One causal classifier output plus scheduler-owned touchdown provenance."""

    class_id: int
    probabilities: np.ndarray
    prediction_timestamp: int
    touchdown_foot: str


@dataclass(frozen=True)
class TerrainGateTrace:
    """Held frozen-Terrain state and exact clean-event provenance."""

    state: np.ndarray
    update_samples: np.ndarray
    prediction_ids: np.ndarray
    prediction_probabilities: np.ndarray
    first_target_valid_sample: int | None
    clean_event_count: int
    prediction_feet: np.ndarray | None = None


def terrain_predictions(trace: TerrainGateTrace) -> tuple[TerrainPrediction, ...]:
    """Expose the canonical prediction schema without terrain-truth metadata."""
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


@dataclass(frozen=True)
class BranchWindowBatch:
    windows: WindowSet
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class BranchReplay:
    endpoints: np.ndarray
    probabilities: np.ndarray
    terrain_state: np.ndarray


@dataclass
class CandidateState:
    """Generated artifacts plus in-memory TRAIN-only normalization."""

    record: dict[str, object]
    normalizer: Normalizer | None
    checkpoint_paths: tuple[Path, ...]


class BranchHoldoutGuard(EventHoldoutGuard):
    """The event holdout remains sealed until both branches are frozen."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _causal_delta(values: np.ndarray, lag: int) -> np.ndarray:
    """Current-minus-past delta with the unavailable causal prefix zeroed."""
    array = np.asarray(values, dtype=np.float32)
    result = np.zeros_like(array)
    if lag < len(array):
        result[lag:] = array[lag:] - array[:-lag]
    return result


def _causal_rolling(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean/variance; no centered or future sample is accessed."""
    array = np.asarray(values, dtype=np.float64)
    prefix = np.vstack((np.zeros((1, array.shape[1])), np.cumsum(array, axis=0)))
    square = np.vstack(
        (np.zeros((1, array.shape[1])), np.cumsum(array * array, axis=0))
    )
    ends = np.arange(1, len(array) + 1)
    starts = np.maximum(0, ends - int(width))
    count = (ends - starts)[:, None]
    mean = (prefix[ends] - prefix[starts]) / count
    variance = (square[ends] - square[starts]) / count - mean * mean
    return mean.astype(np.float32), np.maximum(variance, 0.0).astype(np.float32)


def _temporal_expansion(base: np.ndarray, rolling_ms: int = 10) -> np.ndarray:
    mean, variance = _causal_rolling(base, rolling_ms)
    return np.concatenate(
        (
            np.asarray(base, dtype=np.float32),
            _causal_delta(base, 1),
            _causal_delta(base, 5),
            _causal_delta(base, 10),
            mean,
            variance,
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def imu_feature_base(imu6: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the declared ten-channel IMU base in deterministic order."""
    values = np.asarray(imu6, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("IMU tensor must have shape [samples,6]")
    accel, gyro = values[:, :3], values[:, 3:]
    derived = np.column_stack(
        (
            np.linalg.norm(accel, axis=1),
            np.linalg.norm(gyro, axis=1),
            np.linalg.norm(accel[:, :2], axis=1),
            np.linalg.norm(gyro[:, :2], axis=1),
        )
    ).astype(np.float32)
    names = (
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
    return np.concatenate((values, derived), axis=1), names


def fsr_feature_base(
    fsr8: np.ndarray, epsilon: float = 1.0e-6
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return raw FSR8 plus frozen foot-local and bilateral load features."""
    values = np.asarray(fsr8, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 8 or np.any(values < 0.0):
        raise ValueError("FSR tensor must be nonnegative [samples,8]")
    chunks = [values]
    names = [f"fsr_{index}" for index in range(8)]
    totals = []
    for side, label in enumerate(("left", "right")):
        foot = values[:, side * 4 : (side + 1) * 4]
        total = foot.sum(axis=1)
        front = foot[:, :2].sum(axis=1)
        rear = foot[:, 2:].sum(axis=1)
        medial = foot[:, (1, 3)].sum(axis=1)
        lateral = foot[:, (0, 2)].sum(axis=1)
        local = np.column_stack(
            (
                total,
                front,
                rear,
                medial,
                lateral,
                front - rear,
                medial - lateral,
                front / (total + epsilon),
                medial / (total + epsilon),
            )
        ).astype(np.float32)
        chunks.append(local)
        names.extend(
            f"{label}_{name}"
            for name in (
                "total",
                "front",
                "rear",
                "medial",
                "lateral",
                "front_minus_rear",
                "medial_minus_lateral",
                "front_ratio",
                "medial_ratio",
            )
        )
        totals.append(total)
    left, right = totals
    bilateral = np.column_stack(
        (left, right, left - right, left / (left + right + epsilon))
    ).astype(np.float32)
    chunks.append(bilateral)
    names.extend(
        (
            "bilateral_left_total",
            "bilateral_right_total",
            "bilateral_difference",
            "bilateral_left_ratio",
        )
    )
    return np.concatenate(chunks, axis=1), tuple(names)


def expanded_feature_names(base_names: Sequence[str]) -> tuple[str, ...]:
    prefixes = (
        "raw",
        "delta_1ms",
        "delta_5ms",
        "delta_10ms",
        "mean_10ms",
        "variance_10ms",
    )
    return tuple(f"{prefix}_{name}" for prefix in prefixes for name in base_names)


def extract_branch_features(
    run: EventRun,
    components: Sequence[str],
    *,
    foot_imu12: np.ndarray | None = None,
    epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build one causal runtime tensor; label/gate fields never enter it."""
    arrays: list[np.ndarray] = []
    names: list[str] = []
    imu6 = np.asarray(run.features["PELVIS_IMU6"], dtype=np.float32)
    fusion = np.asarray(run.features["PELVIS_IMU6_FSR8"], dtype=np.float32)
    fsr8 = fusion[:, 6:]
    for component in components:
        if component == "pelvis_imu6":
            base, base_names = imu_feature_base(imu6)
            arrays.append(_temporal_expansion(base))
            names.extend(
                f"pelvis_{name}" for name in expanded_feature_names(base_names)
            )
        elif component == "fsr8":
            base, base_names = fsr_feature_base(fsr8, epsilon)
            arrays.append(_temporal_expansion(base))
            names.extend(expanded_feature_names(base_names))
        elif component == "foot_imu12":
            if foot_imu12 is None:
                raise ValueError("Foot IMU representation requires observer dataset")
            foot = np.asarray(foot_imu12, dtype=np.float32)
            if foot.shape != (len(imu6), 12):
                raise ValueError("Foot IMU must align as [samples,12]")
            for side, label in enumerate(("left", "right")):
                base, base_names = imu_feature_base(foot[:, side * 6 : (side + 1) * 6])
                arrays.append(_temporal_expansion(base))
                names.extend(
                    f"foot_{label}_{name}"
                    for name in expanded_feature_names(base_names)
                )
        else:
            raise ValueError(f"unsupported branch sensor component: {component}")
    if not arrays:
        raise ValueError("branch representation cannot be empty")
    result = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError("derived runtime feature tensor is nonfinite")
    return result, tuple(names)


def branch_is_active(trace: TerrainGateTrace, branch: str) -> np.ndarray:
    """Terrain truth is intentionally absent: only held classifier state gates."""
    if branch not in BRANCH_STATE:
        raise ValueError("branch must be slip or support")
    return np.asarray(trace.state == BRANCH_STATE[branch], dtype=bool)


def _terrain_models(
    model_path: Path,
) -> tuple[list[torch.nn.Module], np.ndarray, np.ndarray]:
    checkpoints = sorted(model_path.glob("seed_*.pt"))
    if len(checkpoints) != 3:
        raise ValueError("frozen Terrain ensemble must contain three checkpoints")
    models = []
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


def frozen_terrain_gate_from_result(
    result: SimulationResult,
    run: EventRun,
    models: Sequence[torch.nn.Module],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    deployment_scheme: str = "left_only",
) -> TerrainGateTrace:
    """Replay actual frozen predictions at exact clean touchdown events."""
    if result.exact_terrain_contact is None or result.runtime.foot_fsr is None:
        raise ValueError("Terrain replay requires exact contact and FSR8")
    fall = run.fall_sample_diagnostic
    rows = build_touchdown_event_rows(
        run.run_id,
        run.split,
        run.source_terrain,
        run.target_terrain,
        result.runtime.timestamp_us,
        result.exact_terrain_contact,
        fall,
        run.event_type in BRANCH_EVENT_TYPES["slip"],
        run.event_type in BRANCH_EVENT_TYPES["support"],
    )
    eligible = [row for row in rows if bool(row["window_50ms_valid"])]
    if deployment_scheme == "left_only":
        eligible = [row for row in eligible if row["foot"] == "left"]
    elif deployment_scheme != "bilateral_shared":
        raise ValueError("unsupported frozen Terrain deployment scheme")
    state = np.full(len(run.timestamp_us), UNKNOWN, dtype=np.int8)
    updates: list[int] = []
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []
    prediction_feet: list[str] = []
    fsr = np.asarray(result.runtime.foot_fsr, dtype=np.float32)
    current = UNKNOWN
    cursor = 0
    for row in sorted(eligible, key=lambda item: int(item["touchdown_sample"])):
        touchdown = int(row["touchdown_sample"])
        update = touchdown + 50
        if update >= len(state) or update >= run.censor_sample:
            continue
        side = 0 if row["foot"] == "left" else 1
        window = fsr[touchdown:update, side * 4 : (side + 1) * 4]
        normalized = ((window - mean) / std).astype(np.float32)[None]
        tensor = torch.from_numpy(normalized)
        with torch.no_grad():
            probability = np.mean(
                [
                    torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                    for model in models
                ],
                axis=0,
            )
        prediction = int(np.argmax(probability))
        next_state = TERRAIN_PREDICTION_TO_STATE[TERRAIN_CLASS_NAMES[prediction]]
        state[cursor:update] = current
        current = next_state
        cursor = update
        updates.append(update)
        predictions.append(prediction)
        probabilities.append(probability.astype(np.float32))
        prediction_feet.append(str(row["foot"]).upper())
    state[cursor:] = current
    target_state = TERRAIN_PREDICTION_TO_STATE[run.target_terrain.upper()]
    target_after_contact = np.flatnonzero(
        (state == target_state)
        & (np.arange(len(state), dtype=np.int64) >= run.first_contact_sample)
    )
    first_target = (
        None if not len(target_after_contact) else int(target_after_contact[0])
    )
    return TerrainGateTrace(
        state=state,
        update_samples=np.asarray(updates, dtype=np.int64),
        prediction_ids=np.asarray(predictions, dtype=np.int8),
        prediction_probabilities=(
            np.stack(probabilities).astype(np.float32)
            if probabilities
            else np.empty((0, 4), dtype=np.float32)
        ),
        first_target_valid_sample=first_target,
        clean_event_count=len(eligible),
        prediction_feet=np.asarray(prediction_feet, dtype="<U5"),
    )


def _save_gate(path: Path, trace: TerrainGateTrace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        state=trace.state,
        update_samples=trace.update_samples,
        prediction_ids=trace.prediction_ids,
        prediction_probabilities=trace.prediction_probabilities,
        first_target_valid_sample=np.asarray(
            -1
            if trace.first_target_valid_sample is None
            else trace.first_target_valid_sample,
            dtype=np.int64,
        ),
        clean_event_count=np.asarray(trace.clean_event_count, dtype=np.int64),
        prediction_feet=(
            np.empty(0, dtype="<U5")
            if trace.prediction_feet is None
            else np.asarray(trace.prediction_feet, dtype="<U5")
        ),
    )


def _load_gate(path: Path, samples: int) -> TerrainGateTrace:
    with np.load(path, allow_pickle=False) as stored:
        first = int(stored["first_target_valid_sample"])
        trace = TerrainGateTrace(
            state=np.asarray(stored["state"], dtype=np.int8),
            update_samples=np.asarray(stored["update_samples"], dtype=np.int64),
            prediction_ids=np.asarray(stored["prediction_ids"], dtype=np.int8),
            prediction_probabilities=np.asarray(
                stored["prediction_probabilities"], dtype=np.float32
            ),
            first_target_valid_sample=None if first < 0 else first,
            clean_event_count=int(stored["clean_event_count"]),
            prediction_feet=(
                np.asarray(stored["prediction_feet"], dtype="<U5")
                if "prediction_feet" in stored.files
                else None
            ),
        )
    if trace.state.shape != (samples,):
        raise ValueError("Terrain gate trace length differs from event run")
    return trace


def _result_matches_event_run(
    result: SimulationResult, run: EventRun
) -> dict[str, bool]:
    fsr = result.runtime.foot_fsr
    if fsr is None:
        raise ValueError("deterministic Terrain replay requires FSR8")
    imu_equal = np.array_equal(result.runtime.pelvis_imu, run.features["PELVIS_IMU6"])
    fsr_equal = np.array_equal(fsr, run.features["PELVIS_IMU6_FSR8"][:, 6:])
    loaded_equal = np.array_equal(result.diagnostics.loaded_contact, run.loaded_contact)
    reduced = _reduce_simulation(
        {
            "id": run.run_id,
            "split": run.split,
            "source_terrain": run.source_terrain,
            "target_terrain": run.target_terrain,
            "design_role": run.design_role,
            "hard_stable_control": run.hard_stable_control,
            "sink_pattern": run.sink_pattern,
            "support_pattern": run.support_pattern,
        },
        result,
        run.outcome_diagnostic,
    )
    return {
        "pelvis_imu6": bool(imu_equal),
        "fsr8": bool(fsr_equal),
        "loaded_contact": bool(loaded_equal),
        "event_clock": reduced.event_sample == run.event_sample,
        "slip_clock": reduced.slip_event_samples_per_foot
        == run.slip_event_samples_per_foot,
        "support_clock": reduced.support_event_samples_per_foot
        == run.support_event_samples_per_foot,
        "fall_clock": reduced.fall_sample_diagnostic == run.fall_sample_diagnostic,
    }


def generate_terrain_gate_cache(
    document: Mapping[str, object],
    event_document: Mapping[str, object],
    specifications: Sequence[Mapping[str, object]],
    runs: Mapping[str, EventRun],
    repository_root: Path,
    output_path: Path,
    progress: Callable[[str], None],
) -> tuple[dict[str, TerrainGateTrace], dict[str, object]]:
    """Deterministically restore clean touchdown clocks absent from the corpus."""
    models, mean, std = _terrain_models(
        repository_root / str(document["source"]["terrain_models"])
    )
    base = load_simulation_config(
        repository_root / str(document["source"]["simulator_config"])
    )
    policy = repository_root / str(document["source"]["policy_path"])
    specs = {str(row["id"]): row for row in specifications}
    traces: dict[str, TerrainGateTrace] = {}
    parity_rows = []
    output_path.mkdir(parents=True, exist_ok=True)
    for index, run_id in enumerate(sorted(runs), start=1):
        run = runs[run_id]
        gate_path = output_path / f"{run_id}.npz"
        if gate_path.is_file():
            traces[run_id] = _load_gate(gate_path, len(run.timestamp_us))
            continue
        specification = specs[run_id]
        result = run_simulation(
            transition_simulation_config(
                base,
                specification,
                policy,
                float(event_document["common"]["duration_s"]),
            ),
            observe_fsr=True,
            observe_foot_imu=False,
            capture_state_trace=False,
        )
        parity = _result_matches_event_run(result, run)
        if not all(parity.values()):
            raise RuntimeError(f"deterministic event replay parity failed: {run_id}")
        trace = frozen_terrain_gate_from_result(
            result,
            run,
            models,
            mean,
            std,
            deployment_scheme=str(document["terrain_branch"]["deployment_scheme"]),
        )
        _save_gate(gate_path, trace)
        traces[run_id] = trace
        parity_rows.append({"run_id": run_id, **parity})
        progress(f"TERRAIN GATE {index}/{len(runs)} {run_id}")
        del result
        if index % 8 == 0:
            gc.collect()
    del models
    return traces, {
        "run_count": len(traces),
        "newly_replayed_runs": len(parity_rows),
        "all_deterministic_parity": all(
            all(value for key, value in row.items() if key != "run_id")
            for row in parity_rows
        ),
        "scheduler_uses_exact_contact_only_for_clean_event": True,
        "branch_gate_uses_terrain_truth": False,
        "model_output_is_actual_frozen_ensemble": True,
    }


def terrain_timing_audit(
    runs: Mapping[str, EventRun], gates: Mapping[str, TerrainGateTrace]
) -> dict[str, object]:
    margins = []
    rows = []
    for run_id, run in sorted(runs.items()):
        if run.hard_stable_control or run.event_sample is None:
            continue
        valid = gates[run_id].first_target_valid_sample
        margin = None if valid is None else int(run.event_sample - valid)
        if margin is not None:
            margins.append(margin)
        rows.append(
            {
                "run_id": run_id,
                "target": run.target_terrain,
                "target_contact_sample": run.first_contact_sample,
                "terrain_valid_sample": valid,
                "event_sample": run.event_sample,
                "event_minus_terrain_ms": margin,
                "event_before_terrain": valid is None or run.event_sample < valid,
            }
        )
    values = np.asarray(margins, dtype=np.float64)
    before = sum(bool(row["event_before_terrain"]) for row in rows)
    distribution = {
        key: None if not len(values) else float(function(values))
        for key, function in (
            ("minimum_ms", np.min),
            ("p10_ms", lambda x: np.percentile(x, 10)),
            ("median_ms", np.median),
            ("p95_ms", lambda x: np.percentile(x, 95)),
            ("maximum_ms", np.max),
        )
    }
    by_branch = {}
    for branch, target in BRANCH_TARGET.items():
        selected = [row for row in rows if row["target"] == target]
        by_branch[branch] = {
            "runs": len(selected),
            "event_before_terrain": sum(
                row["event_before_terrain"] for row in selected
            ),
            "event_before_terrain_rate": 0.0
            if not selected
            else sum(row["event_before_terrain"] for row in selected) / len(selected),
        }
    return {
        "distribution": distribution,
        "event_runs": len(rows),
        "event_before_terrain_count": before,
        "event_before_terrain_rate": 0.0 if not rows else before / len(rows),
        "by_branch": by_branch,
        "rows": rows,
    }


def branch_event_sample(run: EventRun, branch: str) -> int | None:
    values = (
        run.slip_event_samples_per_foot
        if branch == "slip"
        else run.support_event_samples_per_foot
    )
    finite = [int(value) for value in values if value is not None]
    return None if not finite else min(finite)


def _evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    selected = np.asarray(values, dtype=np.int64)
    if count <= 0 or not len(selected):
        return np.empty(0, dtype=np.int64)
    if len(selected) <= count:
        return selected
    return selected[np.linspace(0, len(selected) - 1, count, dtype=np.int64)]


def branch_positive_endpoints(
    run: EventRun,
    gate: TerrainGateTrace,
    branch: str,
    history_ms: int,
    *,
    stride_ms: int = 5,
) -> np.ndarray:
    event = branch_event_sample(run, branch)
    if event is None:
        return np.empty(0, dtype=np.int64)
    endpoints = np.arange(event - 20, event + 41, stride_ms, dtype=np.int64)
    active = branch_is_active(gate, branch)
    valid = (
        (endpoints >= history_ms - 1)
        & (endpoints < run.censor_sample)
        & active[np.clip(endpoints, 0, len(active) - 1)]
    )
    return endpoints[valid]


def branch_negative_candidates(
    run: EventRun,
    gate: TerrainGateTrace,
    branch: str,
    history_ms: int,
) -> np.ndarray:
    event = branch_event_sample(run, branch)
    last = run.censor_sample - 1 if event is None else event - 30
    first = max(run.first_contact_sample, history_ms - 1)
    if last < first:
        return np.empty(0, dtype=np.int64)
    endpoints = np.arange(first, last + 1, dtype=np.int64)
    return endpoints[branch_is_active(gate, branch)[endpoints]]


def initial_negative_endpoints(
    candidates: np.ndarray, per_temporal_bin: int = 8
) -> np.ndarray:
    """Cover early/middle/late branch operation without gait metadata."""
    values = np.asarray(candidates, dtype=np.int64)
    if not len(values):
        return values
    bins = np.array_split(values, 3)
    return np.unique(
        np.concatenate([_evenly_spaced(part, per_temporal_bin) for part in bins])
    )


def mine_hard_negative_endpoints(
    candidates: np.ndarray,
    probabilities: np.ndarray,
    *,
    top_k: int = 8,
    minimum_separation_ms: int = 50,
    excluded: Sequence[int] = (),
) -> np.ndarray:
    """Greedy deterministic TRAIN-only top-score mining with separation."""
    endpoints = np.asarray(candidates, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if endpoints.shape != scores.shape:
        raise ValueError("HNM endpoint/probability shape mismatch")
    excluded_values = {int(value) for value in excluded}
    order = sorted(range(len(endpoints)), key=lambda i: (-scores[i], int(endpoints[i])))
    selected: list[int] = []
    for index in order:
        endpoint = int(endpoints[index])
        if endpoint in excluded_values:
            continue
        if all(abs(endpoint - prior) >= minimum_separation_ms for prior in selected):
            selected.append(endpoint)
            if len(selected) >= top_k:
                break
    return np.asarray(sorted(selected), dtype=np.int64)


def _foot_imu_for_run(foot_dataset_path: Path | None, run_id: str) -> np.ndarray | None:
    if foot_dataset_path is None:
        return None
    path = foot_dataset_path / f"{run_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as stored:
        return np.asarray(stored["foot_imu12"], dtype=np.float32)


def feature_schema_for_components(components: Sequence[str]) -> tuple[str, ...]:
    """Compute schema without allowing scenario metadata to enter the vector."""
    names: list[str] = []
    if "pelvis_imu6" in components:
        _, base = imu_feature_base(np.zeros((10, 6), dtype=np.float32))
        names.extend(f"pelvis_{name}" for name in expanded_feature_names(base))
    if "fsr8" in components:
        _, base = fsr_feature_base(np.zeros((10, 8), dtype=np.float32))
        names.extend(expanded_feature_names(base))
    if "foot_imu12" in components:
        _, base = imu_feature_base(np.zeros((10, 6), dtype=np.float32))
        for side in ("left", "right"):
            names.extend(f"foot_{side}_{name}" for name in expanded_feature_names(base))
    # Preserve the explicit candidate component order rather than the checks above.
    ordered: list[str] = []
    groups = {
        "pelvis_imu6": tuple(name for name in names if name.startswith("pelvis_")),
        "fsr8": tuple(
            name
            for name in names
            if not name.startswith("pelvis_") and not name.startswith("foot_")
        ),
        "foot_imu12": tuple(name for name in names if name.startswith("foot_")),
    }
    for component in components:
        ordered.extend(groups[component])
    return tuple(ordered)


def fit_branch_normalizer(
    runs: Mapping[str, EventRun],
    run_ids: Sequence[str],
    components: Sequence[str],
    *,
    foot_dataset_path: Path | None,
    per_run_sample_cap: int,
    standard_deviation_floor: float,
) -> Normalizer:
    chunks = []
    fit_ids = []
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        features, _ = extract_branch_features(
            run,
            components,
            foot_imu12=_foot_imu_for_run(foot_dataset_path, run_id),
        )
        eligible = np.arange(
            run.first_contact_sample, run.censor_sample, dtype=np.int64
        )
        eligible = _evenly_spaced(eligible, per_run_sample_cap)
        if len(eligible):
            chunks.append(features[eligible].astype(np.float64))
            fit_ids.append(run_id)
        del features
    if not chunks:
        raise ValueError("branch normalizer has no TRAIN samples")
    values = np.concatenate(chunks)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < standard_deviation_floor] = 1.0
    return Normalizer(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        sample_count=len(values),
        fit_run_ids=tuple(fit_ids),
        epsilon=float(standard_deviation_floor),
    )


def _normalize(normalizer: Normalizer, values: np.ndarray) -> np.ndarray:
    result = normalizer.transform(values).astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError("normalized branch tensor is nonfinite")
    return result


def _window_set(
    inputs: Sequence[np.ndarray],
    targets: Sequence[int],
    run_ids: Sequence[str],
    endpoints: Sequence[int],
) -> WindowSet:
    if not inputs:
        raise ValueError("branch window set is empty")
    target = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(target, minlength=2)
    if np.any(counts == 0):
        raise ValueError("branch window set must include NORMAL and EVENT")
    return WindowSet(
        inputs=np.stack(inputs).astype(np.float32),
        targets=target,
        run_ids=np.asarray(run_ids, dtype=str),
        endpoint_samples=np.asarray(endpoints, dtype=np.int64),
        available_by_class=tuple(int(value) for value in counts[:2]),
    )


def build_branch_windows(
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    run_ids: Sequence[str],
    branch: str,
    components: Sequence[str],
    history_ms: int,
    normalizer: Normalizer,
    *,
    foot_dataset_path: Path | None,
    per_temporal_bin: int = 8,
    extra_negative_endpoints: Mapping[str, Sequence[int]] | None = None,
) -> BranchWindowBatch:
    inputs: list[np.ndarray] = []
    targets: list[int] = []
    source_ids: list[str] = []
    endpoints: list[int] = []
    rows: list[dict[str, object]] = []
    extras = extra_negative_endpoints or {}
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        if run.target_terrain != BRANCH_TARGET[branch] or run.hard_stable_control:
            continue
        features, _ = extract_branch_features(
            run,
            components,
            foot_imu12=_foot_imu_for_run(foot_dataset_path, run_id),
        )
        positive = branch_positive_endpoints(run, gates[run_id], branch, history_ms)
        negative = initial_negative_endpoints(
            branch_negative_candidates(run, gates[run_id], branch, history_ms),
            per_temporal_bin,
        )
        if run_id in extras:
            negative = np.unique(
                np.concatenate((negative, np.asarray(extras[run_id], dtype=np.int64)))
            )
        for label, selected, kind in (
            (1, positive, "event_positive"),
            (0, negative, "active_branch_negative"),
        ):
            for endpoint in selected:
                first = int(endpoint) - history_ms + 1
                if first < 0 or int(endpoint) >= run.censor_sample:
                    raise ValueError("branch window crossed causal boundary")
                inputs.append(
                    _normalize(normalizer, features[first : int(endpoint) + 1])
                )
                targets.append(label)
                source_ids.append(run_id)
                endpoints.append(int(endpoint))
                rows.append(
                    {
                        "run_id": run_id,
                        "endpoint_sample": int(endpoint),
                        "label": label,
                        "kind": kind,
                        "branch": branch,
                    }
                )
        del features
    return BranchWindowBatch(
        windows=_window_set(inputs, targets, source_ids, endpoints),
        rows=tuple(rows),
    )


def _predict_windows(
    models: Sequence[torch.nn.Module], windows: np.ndarray
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(windows, dtype=np.float32))
    with torch.no_grad():
        probabilities = [
            torch.softmax(model(tensor), dim=1)[:, 1].cpu().numpy() for model in models
        ]
    return np.mean(np.stack(probabilities), axis=0).astype(np.float64)


def replay_branch_run(
    run: EventRun,
    gate: TerrainGateTrace,
    components: Sequence[str],
    history_ms: int,
    normalizer: Normalizer,
    models: Sequence[torch.nn.Module],
    *,
    foot_imu12: np.ndarray | None,
    batch_size: int = 512,
) -> BranchReplay:
    features, _ = extract_branch_features(run, components, foot_imu12=foot_imu12)
    start = max(run.first_contact_sample, history_ms - 1)
    endpoints = np.arange(start, run.censor_sample, dtype=np.int64)
    probability = []
    offsets = np.arange(history_ms - 1, -1, -1, dtype=np.int64)
    for first in range(0, len(endpoints), batch_size):
        selected = endpoints[first : first + batch_size]
        indices = selected[:, None] - offsets[None, :]
        windows = _normalize(normalizer, features[indices])
        probability.append(_predict_windows(models, windows))
    del features
    return BranchReplay(
        endpoints=endpoints,
        probabilities=(
            np.concatenate(probability)
            if probability
            else np.empty(0, dtype=np.float64)
        ),
        terrain_state=gate.state[endpoints],
    )


def sustained_alert_trace(
    probabilities: np.ndarray,
    active_branch: np.ndarray,
    threshold: float,
    persistence_ms: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal persistence resets whenever probability or terrain gate drops."""
    probability = np.asarray(probabilities, dtype=np.float64)
    gate = np.asarray(active_branch, dtype=bool)
    if probability.shape != gate.shape or persistence_ms <= 0:
        raise ValueError("invalid sustained branch alert inputs")
    alert = np.zeros(len(probability), dtype=bool)
    onset = np.zeros(len(probability), dtype=bool)
    count = 0
    previous = False
    for index in range(len(probability)):
        passes = bool(gate[index] and probability[index] >= threshold)
        count = count + 1 if passes else 0
        current = count >= persistence_ms
        alert[index] = current
        onset[index] = current and not previous
        previous = current
    return alert, onset


def _distribution(values: Sequence[int]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {name: None for name in ("min", "p10", "median", "p95", "max")}
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def evaluate_branch_replays(
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    replays: Mapping[str, BranchReplay],
    branch: str,
    threshold: float,
    persistence_ms: int = 5,
) -> dict[str, object]:
    """Run-level branch metrics keep premature and later valid alerts separate."""
    lower, upper = (-20, 50) if branch == "slip" else (-30, 50)
    event_rows = []
    benign_rows = []
    hard_rows = []
    cross_rows = []
    latencies: list[int] = []
    negative_alert_samples = 0
    active_negative_samples = 0
    for run_id, replay in sorted(replays.items()):
        run = runs[run_id]
        gate_active = replay.terrain_state == BRANCH_STATE[branch]
        alert, onset = sustained_alert_trace(
            replay.probabilities, gate_active, threshold, persistence_ms
        )
        onset_samples = replay.endpoints[onset]
        event = branch_event_sample(run, branch)
        relevant = (
            not run.hard_stable_control and run.target_terrain == BRANCH_TARGET[branch]
        )
        if event is not None and relevant:
            premature = onset_samples[onset_samples < event + lower]
            valid = onset_samples[
                (onset_samples >= event + lower) & (onset_samples <= event + upper)
            ]
            first_valid = None if not len(valid) else int(valid[0])
            negative_mask = replay.endpoints < event + lower
            row = {
                "run_id": run_id,
                "event_sample": event,
                "terrain_valid_sample": gates[run_id].first_target_valid_sample,
                "first_valid_detection_sample": first_valid,
                "any_premature_alert": bool(len(premature)),
                "premature_count": int(len(premature)),
                "valid_detection": first_valid is not None,
                "latency_ms": None if first_valid is None else first_valid - event,
                "source_terrain": run.source_terrain,
                "outcome_diagnostic_only": run.outcome_diagnostic,
            }
            event_rows.append(row)
            if first_valid is not None:
                latencies.append(first_valid - event)
        elif relevant:
            negative_mask = np.ones(len(replay.endpoints), dtype=bool)
            any_alert = bool(np.any(alert))
            benign_rows.append(
                {
                    "run_id": run_id,
                    "system_false_alert": any_alert,
                    "first_alert_sample": None
                    if not np.any(onset)
                    else int(replay.endpoints[np.flatnonzero(onset)[0]]),
                }
            )
        else:
            negative_mask = np.zeros(len(replay.endpoints), dtype=bool)
            raw_alert, raw_onset = sustained_alert_trace(
                replay.probabilities,
                np.ones(len(replay.endpoints), dtype=bool),
                threshold,
                persistence_ms,
            )
            cross_rows.append(
                {"run_id": run_id, "raw_cross_terrain_alert": bool(np.any(raw_alert))}
            )
            if run.hard_stable_control:
                hard_rows.append(
                    {
                        "run_id": run_id,
                        "system_false_alert": bool(np.any(alert)),
                        "wrong_soft_terrain_output": bool(np.any(gate_active)),
                        "raw_alert": bool(np.any(raw_alert)),
                        "raw_onset_count": int(np.count_nonzero(raw_onset)),
                    }
                )
        active_negative = negative_mask & gate_active
        active_negative_samples += int(np.count_nonzero(active_negative))
        negative_alert_samples += int(np.count_nonzero(alert & active_negative))
    valid_count = sum(row["valid_detection"] for row in event_rows)
    premature_count = sum(row["any_premature_alert"] for row in event_rows)
    benign_fp = sum(row["system_false_alert"] for row in benign_rows)
    hard_fp = sum(row["system_false_alert"] for row in hard_rows)
    return {
        "threshold": float(threshold),
        "event_runs": len(event_rows),
        "event_recall": 0.0 if not event_rows else valid_count / len(event_rows),
        "premature_run_rate": 0.0
        if not event_rows
        else premature_count / len(event_rows),
        "benign_runs": len(benign_rows),
        "benign_specificity": 1.0
        if not benign_rows
        else 1.0 - benign_fp / len(benign_rows),
        "hard_ground_runs": len(hard_rows),
        "hard_ground_specificity": 1.0
        if not hard_rows
        else 1.0 - hard_fp / len(hard_rows),
        "active_negative_samples": active_negative_samples,
        "active_negative_alert_samples": negative_alert_samples,
        "active_negative_alert_fraction": 0.0
        if not active_negative_samples
        else negative_alert_samples / active_negative_samples,
        "latency_ms": _distribution(latencies),
        "event_rows": event_rows,
        "benign_rows": benign_rows,
        "hard_rows": hard_rows,
        "raw_cross_terrain_rows": cross_rows,
    }


def branch_gate_results(
    metrics: Mapping[str, object], gates: Mapping[str, object], branch: str
) -> dict[str, bool]:
    latency = metrics["latency_ms"]
    result = {
        "event_recall": float(metrics["event_recall"])
        >= float(gates["event_recall_min"]),
        "premature_run_rate": float(metrics["premature_run_rate"])
        <= float(gates["premature_run_rate_max"]),
        "median_latency_ms": latency["median"] is not None
        and float(latency["median"]) <= float(gates["median_latency_ms_max"]),
        "p95_latency_ms": latency["p95"] is not None
        and float(latency["p95"]) <= float(gates["p95_latency_ms_max"]),
        "active_negative_alert_fraction": float(
            metrics["active_negative_alert_fraction"]
        )
        <= float(gates["active_negative_alert_fraction_max"]),
    }
    if branch == "support":
        result["benign_specificity"] = float(metrics["benign_specificity"]) >= float(
            gates["benign_specificity_min"]
        )
    return result


def _threshold_values(grid: Mapping[str, object]) -> tuple[float, ...]:
    start, stop, step = (float(grid[name]) for name in ("start", "stop", "step"))
    count = int(round((stop - start) / step))
    values = tuple(round(start + index * step, 10) for index in range(count + 1))
    if values[0] != 0.10 or values[-1] != 0.98 or step != 0.02:
        raise ValueError("branch threshold grid must remain 0.10..0.98 step 0.02")
    return values


def select_branch_threshold(
    evaluations: Sequence[Mapping[str, object]],
    gates: Mapping[str, object],
    branch: str,
) -> dict[str, object]:
    rows = []
    for row in evaluations:
        checks = branch_gate_results(row["metrics"], gates, branch)
        rows.append({**dict(row), "gates": checks, "passed": all(checks.values())})

    def rank(row: Mapping[str, object]) -> tuple[float, float, float, float, float]:
        metrics = row["metrics"]
        p95 = metrics["latency_ms"]["p95"]
        return (
            -float(metrics["premature_run_rate"]),
            -9999.0 if p95 is None else -float(p95),
            float(metrics["event_recall"]),
            float(metrics["benign_specificity"]),
            float(row["threshold"]),
        )

    passing = [row for row in rows if bool(row["passed"])]
    if passing:
        selected = max(passing, key=rank)
        return {
            "selected": selected,
            "diagnostic_best": selected,
            "passing_threshold_count": len(passing),
        }
    diagnostic = max(
        rows,
        key=lambda row: (
            sum(bool(value) for value in row["gates"].values()),
            *rank(row),
        ),
    )
    return {
        "selected": None,
        "diagnostic_best": diagnostic,
        "passing_threshold_count": 0,
    }


def _relevant_run_ids(
    runs: Mapping[str, EventRun], branch: str, split: str
) -> list[str]:
    return sorted(
        run_id
        for run_id, run in runs.items()
        if run.split == split
        and not run.hard_stable_control
        and run.target_terrain == BRANCH_TARGET[branch]
    )


def _train_monitor_partition(
    runs: Mapping[str, EventRun], run_ids: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Deterministic one-in-five TRAIN-run monitor, separately by event status."""
    groups: dict[tuple[str, bool], list[str]] = {}
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        key = (run.source_terrain, run.event_sample is not None)
        groups.setdefault(key, []).append(run_id)
    monitor: list[str] = []
    for values in groups.values():
        count = max(1, int(round(len(values) * 0.20)))
        indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
        monitor.extend(values[int(index)] for index in indices)
    monitor_set = set(monitor)
    fit = [run_id for run_id in run_ids if run_id not in monitor_set]
    return sorted(fit), sorted(monitor_set)


def _checkpoint_paths(
    artifact_path: Path,
    phase: str,
    branch: str,
    candidate_id: str,
    family: str,
    history_ms: int,
    round_id: int,
    seeds: Sequence[int],
) -> tuple[Path, ...]:
    folder = artifact_path / "checkpoints" / phase.lower() / branch
    return tuple(
        folder
        / f"{candidate_id.lower()}_{family}_history{history_ms}_round{round_id}_seed{seed}.pt"
        for seed in seeds
    )


def _load_models(paths: Sequence[Path]) -> list[torch.nn.Module]:
    return [load_checkpoint(path)[0] for path in paths]


def _replay_many(
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    run_ids: Sequence[str],
    components: Sequence[str],
    history_ms: int,
    normalizer: Normalizer,
    checkpoint_paths: Sequence[Path],
    foot_dataset_path: Path | None,
) -> dict[str, BranchReplay]:
    models = _load_models(checkpoint_paths)
    traces = {}
    for run_id in run_ids:
        if run_id not in runs:
            continue
        traces[run_id] = replay_branch_run(
            runs[run_id],
            gates[run_id],
            components,
            history_ms,
            normalizer,
            models,
            foot_imu12=_foot_imu_for_run(foot_dataset_path, run_id),
        )
    del models
    return traces


def _mine_training_round(
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    run_ids: Sequence[str],
    branch: str,
    traces: Mapping[str, BranchReplay],
    prior: Mapping[str, Sequence[int]],
    config: Mapping[str, object],
) -> tuple[dict[str, tuple[int, ...]], dict[str, object]]:
    selected: dict[str, tuple[int, ...]] = {}
    scores = []
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        candidates = branch_negative_candidates(
            run, gates[run_id], branch, history_ms=1
        )
        trace = traces[run_id]
        common, candidate_indices, trace_indices = np.intersect1d(
            candidates, trace.endpoints, return_indices=True
        )
        probability = trace.probabilities[trace_indices]
        mined = mine_hard_negative_endpoints(
            common,
            probability,
            top_k=int(config["top_k_per_run"]),
            minimum_separation_ms=int(config["minimum_separation_ms"]),
            excluded=prior.get(run_id, ()),
        )
        selected[run_id] = tuple(int(value) for value in mined)
        if len(mined):
            lookup = {
                int(endpoint): float(score)
                for endpoint, score in zip(common, probability)
            }
            scores.extend(lookup[int(endpoint)] for endpoint in mined)
    return selected, {
        "runs_with_hard_negatives": sum(bool(values) for values in selected.values()),
        "mined_windows": sum(len(values) for values in selected.values()),
        "probability": _distribution([int(round(value * 1000)) for value in scores]),
        "source_split": "train",
        "top_k": int(config["top_k_per_run"]),
        "minimum_separation_ms": int(config["minimum_separation_ms"]),
    }


def _merge_endpoint_maps(
    *maps: Mapping[str, Sequence[int]]
) -> dict[str, tuple[int, ...]]:
    keys = {key for mapping in maps for key in mapping}
    return {
        key: tuple(
            sorted({int(value) for mapping in maps for value in mapping.get(key, ())})
        )
        for key in keys
    }


def train_branch_candidate(
    document: Mapping[str, object],
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    branch: str,
    phase: str,
    candidate_id: str,
    components: Sequence[str],
    family: str,
    history_ms: int,
    artifact_path: Path,
    foot_dataset_path: Path | None,
    progress: Callable[[str], None],
) -> CandidateState:
    """Execute fixed Round0 -> HNM1 -> Round1 -> HNM2 -> Round2 on TRAIN."""
    train_config = document["training"]
    hnm_config = document["hard_negative_mining"]
    train_ids = _relevant_run_ids(runs, branch, "train")
    fit_ids, monitor_ids = _train_monitor_partition(runs, train_ids)
    schema = feature_schema_for_components(components)
    positive_count = sum(
        len(branch_positive_endpoints(runs[run_id], gates[run_id], branch, history_ms))
        for run_id in train_ids
    )
    negative_count = sum(
        len(branch_negative_candidates(runs[run_id], gates[run_id], branch, history_ms))
        for run_id in train_ids
    )
    if positive_count == 0 or negative_count == 0:
        record = {
            "phase": phase,
            "branch": branch,
            "candidate_id": candidate_id,
            "components": list(components),
            "feature_dimension": len(schema),
            "feature_schema_sha256": _canonical_sha256(schema),
            "model_family": family,
            "history_ms": history_ms,
            "physical_channels": physical_channel_count(components),
            "parameter_count": parameter_count(
                build_model(family, history_ms, len(schema), class_count=2)
            ),
            "train_runs": len(train_ids),
            "fit_runs": len(fit_ids),
            "internal_monitor_runs": len(monitor_ids),
            "active_positive_endpoints": positive_count,
            "active_negative_endpoints": negative_count,
            "training_status": "NOT_TRAINED_TERRAIN_GATE_INFEASIBLE",
            "rounds": [
                {
                    "round": round_id,
                    "performed": False,
                    "reason": "frozen_terrain_branch_has_no_binary_training_support",
                }
                for round_id in range(3)
            ],
            "train_full_replay_threshold_0_5": None,
            "checkpoint_paths": [],
            "validation": None,
        }
        progress(
            f"{phase} {branch.upper()} {candidate_id} {family} {history_ms}ms "
            f"fail-closed: positives={positive_count} negatives={negative_count}"
        )
        return CandidateState(record=record, normalizer=None, checkpoint_paths=())
    normalizer = fit_branch_normalizer(
        runs,
        train_ids,
        components,
        foot_dataset_path=foot_dataset_path,
        per_run_sample_cap=int(train_config["normalizer_per_run_sample_cap"]),
        standard_deviation_floor=float(train_config["standard_deviation_floor"]),
    )
    normalizer_path = (
        artifact_path
        / "normalization"
        / phase.lower()
        / branch
        / f"{candidate_id.lower()}.json"
    )
    _write_json(
        normalizer_path,
        {
            **normalizer.to_dict(),
            "components": list(components),
            "feature_schema": list(schema),
            "feature_schema_sha256": _canonical_sha256(schema),
        },
    )
    base_fit = build_branch_windows(
        runs,
        gates,
        fit_ids,
        branch,
        components,
        history_ms,
        normalizer,
        foot_dataset_path=foot_dataset_path,
        per_temporal_bin=int(train_config["initial_negative_per_temporal_bin"]),
    )
    monitor = build_branch_windows(
        runs,
        gates,
        monitor_ids,
        branch,
        components,
        history_ms,
        normalizer,
        foot_dataset_path=foot_dataset_path,
        per_temporal_bin=int(train_config["initial_negative_per_temporal_bin"]),
    )
    seeds = [int(value) for value in train_config["seeds"]]
    hnm_maps: list[dict[str, tuple[int, ...]]] = []
    rounds = []
    final_paths: tuple[Path, ...] = ()
    for round_id in range(3):
        extras = _merge_endpoint_maps(*hnm_maps) if hnm_maps else {}
        train_batch = (
            base_fit
            if not extras
            else build_branch_windows(
                runs,
                gates,
                fit_ids,
                branch,
                components,
                history_ms,
                normalizer,
                foot_dataset_path=foot_dataset_path,
                per_temporal_bin=int(train_config["initial_negative_per_temporal_bin"]),
                extra_negative_endpoints=extras,
            )
        )
        paths = _checkpoint_paths(
            artifact_path,
            phase,
            branch,
            candidate_id,
            family,
            history_ms,
            round_id,
            seeds,
        )
        training_rows = []
        for seed, path in zip(seeds, paths):
            model, training = train_model(
                family,
                history_ms,
                train_batch.windows,
                monitor.windows,
                seed,
                batch_size=int(train_config["batch_size"]),
                max_epochs=int(train_config["max_epochs"]),
                patience=int(train_config["patience"]),
                learning_rate=float(train_config["learning_rate"]),
                class_names=EVENT_CLASS_NAMES,
                selection_metric="validation_loss",
            )
            save_checkpoint(
                path,
                model,
                family,
                history_ms,
                seed,
                training,
                input_channels=len(schema),
                class_names=EVENT_CLASS_NAMES,
            )
            training_rows.append(
                {
                    "seed": seed,
                    "best_epoch": training.best_epoch,
                    "epochs_completed": training.epochs_completed,
                    "best_internal_monitor_cross_entropy": min(
                        row["validation_cross_entropy"] for row in training.history
                    ),
                }
            )
            del model
        round_row: dict[str, object] = {
            "round": round_id,
            "fit_windows": len(train_batch.windows),
            "normal_windows": int(np.count_nonzero(train_batch.windows.targets == 0)),
            "event_windows": int(np.count_nonzero(train_batch.windows.targets == 1)),
            "monitor_windows": len(monitor.windows),
            "training": training_rows,
        }
        final_paths = paths
        if round_id < 2:
            traces = _replay_many(
                runs,
                gates,
                train_ids,
                components,
                history_ms,
                normalizer,
                paths,
                foot_dataset_path,
            )
            prior = _merge_endpoint_maps(*hnm_maps) if hnm_maps else {}
            mined, mining = _mine_training_round(
                runs,
                gates,
                train_ids,
                branch,
                traces,
                prior,
                hnm_config,
            )
            hnm_maps.append(mined)
            round_row["mining_for_next_round"] = mining
            del traces
        rounds.append(round_row)
        progress(
            f"{phase} {branch.upper()} {candidate_id} {family} {history_ms}ms "
            f"ROUND {round_id} windows={len(train_batch.windows)}"
        )
        if train_batch is not base_fit:
            del train_batch
        gc.collect()
    train_traces = _replay_many(
        runs,
        gates,
        train_ids,
        components,
        history_ms,
        normalizer,
        final_paths,
        foot_dataset_path,
    )
    train_metrics = evaluate_branch_replays(
        {run_id: runs[run_id] for run_id in train_ids},
        {run_id: gates[run_id] for run_id in train_ids},
        train_traces,
        branch,
        0.5,
        int(document["validation"]["persistence_ms"]),
    )
    del train_traces, base_fit, monitor
    record = {
        "phase": phase,
        "branch": branch,
        "candidate_id": candidate_id,
        "components": list(components),
        "feature_dimension": len(schema),
        "feature_schema_sha256": _canonical_sha256(schema),
        "model_family": family,
        "history_ms": history_ms,
        "physical_channels": physical_channel_count(components),
        "parameter_count": parameter_count(load_checkpoint(final_paths[0])[0]),
        "train_runs": len(train_ids),
        "fit_runs": len(fit_ids),
        "internal_monitor_runs": len(monitor_ids),
        "active_positive_endpoints": positive_count,
        "active_negative_endpoints": negative_count,
        "training_status": "ROUND_0_HNM_1_ROUND_1_HNM_2_ROUND_2_COMPLETE",
        "rounds": rounds,
        "train_full_replay_threshold_0_5": train_metrics,
        "checkpoint_paths": [
            str(path.relative_to(artifact_path)) for path in final_paths
        ],
        "validation": None,
    }
    return CandidateState(
        record=record, normalizer=normalizer, checkpoint_paths=final_paths
    )


def physical_channel_count(components: Sequence[str]) -> int:
    return sum(
        {"pelvis_imu6": 6, "fsr8": 8, "foot_imu12": 12}[component]
        for component in set(components)
    )


def validate_branch_candidate(
    document: Mapping[str, object],
    state: CandidateState,
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    artifact_path: Path,
    foot_dataset_path: Path | None,
) -> dict[str, object]:
    record = state.record
    branch = str(record["branch"])
    if state.normalizer is None or not state.checkpoint_paths:
        metrics = {
            "threshold": None,
            "event_runs": sum(
                run.split == "validation"
                and run.target_terrain == BRANCH_TARGET[branch]
                and branch_event_sample(run, branch) is not None
                for run in runs.values()
            ),
            "event_recall": 0.0,
            "premature_run_rate": 0.0,
            "benign_runs": sum(
                run.split == "validation"
                and run.target_terrain == BRANCH_TARGET[branch]
                and branch_event_sample(run, branch) is None
                for run in runs.values()
            ),
            "benign_specificity": 1.0,
            "hard_ground_runs": sum(
                run.split == "validation" and run.hard_stable_control
                for run in runs.values()
            ),
            "hard_ground_specificity": 1.0,
            "active_negative_samples": 0,
            "active_negative_alert_samples": 0,
            "active_negative_alert_fraction": 0.0,
            "latency_ms": _distribution([]),
            "event_rows": [],
            "benign_rows": [],
            "hard_rows": [],
            "raw_cross_terrain_rows": [],
        }
        checks = branch_gate_results(
            metrics, document["validation"][f"{branch}_gates"], branch
        )
        result = {
            "passed": False,
            "selected_threshold": None,
            "passing_threshold_count": 0,
            "operating_threshold": None,
            "metrics": metrics,
            "gates": checks,
            "reason": "training_infeasible_under_actual_frozen_terrain_gate",
        }
        record["validation"] = result
        return result
    validation_ids = sorted(
        run_id for run_id, run in runs.items() if run.split == "validation"
    )
    traces = _replay_many(
        runs,
        gates,
        validation_ids,
        tuple(str(value) for value in record["components"]),
        int(record["history_ms"]),
        state.normalizer,
        state.checkpoint_paths,
        foot_dataset_path,
    )
    evaluations = []
    for threshold in _threshold_values(document["validation"]["threshold_grid"]):
        metrics = evaluate_branch_replays(
            {run_id: runs[run_id] for run_id in validation_ids},
            {run_id: gates[run_id] for run_id in validation_ids},
            traces,
            branch,
            threshold,
            int(document["validation"]["persistence_ms"]),
        )
        evaluations.append({"threshold": threshold, "metrics": metrics})
    selection = select_branch_threshold(
        evaluations,
        document["validation"][f"{branch}_gates"],
        branch,
    )
    operating = selection["selected"] or selection["diagnostic_best"]
    result = {
        "passed": selection["selected"] is not None,
        "selected_threshold": None
        if selection["selected"] is None
        else float(selection["selected"]["threshold"]),
        "passing_threshold_count": int(selection["passing_threshold_count"]),
        "operating_threshold": float(operating["threshold"]),
        "metrics": operating["metrics"],
        "gates": operating["gates"],
    }
    record["validation"] = result
    del traces
    gc.collect()
    return result


def select_branch_candidate(states: Sequence[CandidateState]) -> dict[str, object]:
    passing = [
        state
        for state in states
        if bool(state.record.get("validation", {}).get("passed"))
    ]

    def rank(state: CandidateState) -> tuple[float, float, int, int]:
        record = state.record
        metrics = record["validation"]["metrics"]
        p95 = metrics["latency_ms"]["p95"]
        return (
            -float(metrics["premature_run_rate"]),
            -9999.0 if p95 is None else -float(p95),
            -int(record["physical_channels"]),
            -int(record["parameter_count"]),
        )

    if not passing:
        diagnostics = sorted(
            states,
            key=lambda state: (
                sum(
                    bool(value)
                    for value in state.record["validation"]["gates"].values()
                ),
                *rank(state),
            ),
            reverse=True,
        )
        return {
            "selected": None,
            "reason": "no_candidate_passed_all_branch_validation_gates",
            "diagnostic_best": None
            if not diagnostics
            else candidate_identity(diagnostics[0]),
        }
    selected = max(passing, key=rank)
    return {
        "selected": candidate_identity(selected),
        "reason": "all_gates_then_premature_p95_channels_parameters",
    }


def candidate_identity(state: CandidateState) -> dict[str, object]:
    record = state.record
    validation = record["validation"]
    return {
        "phase": record["phase"],
        "branch": record["branch"],
        "candidate_id": record["candidate_id"],
        "components": list(record["components"]),
        "feature_dimension": record["feature_dimension"],
        "feature_schema_sha256": record["feature_schema_sha256"],
        "model_family": record["model_family"],
        "history_ms": record["history_ms"],
        "physical_channels": record["physical_channels"],
        "parameter_count": record["parameter_count"],
        "threshold": validation["selected_threshold"],
        "persistence_ms": 5,
        "checkpoint_paths": list(record["checkpoint_paths"]),
    }


def _state_by_identity(
    states: Sequence[CandidateState], identity: Mapping[str, object]
) -> CandidateState:
    selected = [
        state
        for state in states
        if state.record["candidate_id"] == identity["candidate_id"]
        and state.record["model_family"] == identity["model_family"]
        and state.record["history_ms"] == identity["history_ms"]
    ]
    if len(selected) != 1:
        raise ValueError("frozen branch selection is missing or ambiguous")
    return selected[0]


def _observer_manifest_sha(rows: Sequence[Mapping[str, object]]) -> str:
    return _canonical_sha256(list(rows))


def generate_foot_imu_dataset(
    document: Mapping[str, object],
    event_document: Mapping[str, object],
    event_manifest: Mapping[str, object],
    specifications: Sequence[Mapping[str, object]],
    development_runs: Mapping[str, EventRun],
    repository_root: Path,
    progress: Callable[[str], None],
) -> tuple[Path, dict[str, object]]:
    """Generate observer-only Foot IMU12 for the fixed 256-run split."""
    output = repository_root / str(document["phase_b"]["dataset_path"])
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        return output, {
            "dataset_id": manifest["dataset_id"],
            "run_count": int(manifest["run_count"]),
            "manifest_sha256": _file_sha256(manifest_path),
            "size_bytes": sum(
                (output / str(row["file"])).stat().st_size for row in manifest["runs"]
            ),
            "observer_parity": manifest["observer_parity"],
            "reused_existing_generated_dataset": True,
        }
    output.mkdir(parents=True, exist_ok=True)
    base = load_simulation_config(
        repository_root / str(document["source"]["simulator_config"])
    )
    policy = repository_root / str(document["source"]["policy_path"])
    original_rows = {str(row["run_id"]): row for row in event_manifest["runs"]}
    rows = []
    parity_checks = []
    for index, specification in enumerate(specifications, start=1):
        run_id = str(specification["id"])
        result = run_simulation(
            transition_simulation_config(
                base,
                specification,
                policy,
                float(event_document["common"]["duration_s"]),
            ),
            observe_fsr=True,
            observe_foot_imu=True,
            capture_state_trace=False,
        )
        outcome = (
            _hard_control_outcome(result)
            if bool(specification["hard_stable_control"])
            else classify_scenario_outcome(result, specification)
        )
        if outcome not in VALID_OUTCOMES:
            raise RuntimeError(f"Foot IMU observer run became invalid: {run_id}")
        reduced = _reduce_simulation(specification, result, outcome)
        original = original_rows[run_id]
        clock_parity = {
            "event": reduced.event_sample == original["event_sample"],
            "fall": reduced.fall_sample_diagnostic
            == original["fall_sample_diagnostic_only"],
            "outcome": reduced.outcome_diagnostic
            == original["observed_outcome_diagnostic_only"],
            "contact": reduced.first_contact_sample
            == int(original["first_target_contact_sample"]),
        }
        if run_id in development_runs:
            sensor_parity = _result_matches_event_run(result, development_runs[run_id])
        else:
            sensor_parity = {"sealed_holdout_waveform_not_opened": True}
        if not all(clock_parity.values()) or not all(sensor_parity.values()):
            raise RuntimeError(f"Foot IMU observer parity failed: {run_id}")
        foot_imu = result.runtime.foot_imu
        if foot_imu is None or foot_imu.shape != (len(reduced.timestamp_us), 12):
            raise ValueError("Foot IMU observer tensor is absent or malformed")
        path = output / f"{run_id}.npz"
        np.savez_compressed(
            path,
            timestamp_us=reduced.timestamp_us,
            pelvis_imu6=reduced.features["PELVIS_IMU6"],
            foot_fsr8=reduced.features["PELVIS_IMU6_FSR8"][:, 6:],
            foot_imu12=np.asarray(foot_imu, dtype=np.float32),
            exact_terrain_contact=np.asarray(result.exact_terrain_contact, dtype=bool),
            first_target_contact_sample=np.asarray(
                reduced.first_contact_sample, dtype=np.int64
            ),
            censor_sample=np.asarray(reduced.censor_sample, dtype=np.int64),
        )
        rows.append(
            {
                "run_id": run_id,
                "file": path.name,
                "file_sha256": _file_sha256(path),
                "split": specification["split"],
                "event_type": reduced.event_type,
                "event_sample": reduced.event_sample,
                "size_bytes": path.stat().st_size,
            }
        )
        parity_checks.append({"run_id": run_id, **clock_parity, **sensor_parity})
        progress(f"FOOT IMU DATASET {index}/{len(specifications)} {run_id}")
        del result, reduced
        if index % 8 == 0:
            gc.collect()
    manifest = {
        "schema_version": "reflex_event_foot_imu_v1",
        "dataset_id": str(document["phase_b"]["dataset_id"]),
        "source_event_dataset_id": event_manifest["dataset_id"],
        "source_event_manifest_sha256": _file_sha256(
            repository_root / str(document["source"]["event_dataset"]) / "manifest.json"
        ),
        "run_count": len(rows),
        "split_membership_unchanged": True,
        "holdout_detector_selection_access_during_generation": False,
        "observer_parity": {
            "all_runs_clock_and_outcome_exact": all(
                all(value for key, value in row.items() if key != "run_id")
                for row in parity_checks
            ),
            "development_runtime_sensor_exact": True,
            "controller_observation_action_state_parity": "covered_by_canonical_observer_parity_regression",
            "foot_imu_observer_only": True,
        },
        "runs": rows,
        "rows_sha256": _observer_manifest_sha(rows),
    }
    _write_json(manifest_path, manifest)
    return output, {
        "dataset_id": manifest["dataset_id"],
        "run_count": len(rows),
        "manifest_sha256": _file_sha256(manifest_path),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "observer_parity": manifest["observer_parity"],
        "reused_existing_generated_dataset": False,
    }


def holdout_gate_from_observer_dataset(
    document: Mapping[str, object],
    runs: Mapping[str, EventRun],
    foot_dataset_path: Path,
    repository_root: Path,
) -> dict[str, TerrainGateTrace]:
    """Open stored exact scheduler traces only after the holdout guard opens."""
    models, mean, std = _terrain_models(
        repository_root / str(document["source"]["terrain_models"])
    )
    traces = {}
    for run_id, run in runs.items():
        with np.load(foot_dataset_path / f"{run_id}.npz", allow_pickle=False) as stored:
            exact = np.asarray(stored["exact_terrain_contact"], dtype=bool)
            fsr = np.asarray(stored["foot_fsr8"], dtype=np.float32)
        # Only the small subset of attributes consumed by the frozen scheduler.
        class RuntimeProxy:
            timestamp_us = run.timestamp_us
            foot_fsr = fsr

        class ResultProxy:
            exact_terrain_contact = exact
            runtime = RuntimeProxy()

        traces[run_id] = frozen_terrain_gate_from_result(
            ResultProxy(),  # type: ignore[arg-type]
            run,
            models,
            mean,
            std,
            deployment_scheme=str(document["terrain_branch"]["deployment_scheme"]),
        )
    del models
    return traces


def evaluate_selected_holdout(
    document: Mapping[str, object],
    selected_states: Mapping[str, CandidateState],
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    artifact_path: Path,
    foot_dataset_path: Path | None,
) -> dict[str, object]:
    branch_results = {}
    metrics_by_branch = {}
    for branch in ("slip", "support"):
        state = selected_states[branch]
        record = state.record
        traces = _replay_many(
            runs,
            gates,
            sorted(runs),
            tuple(str(value) for value in record["components"]),
            int(record["history_ms"]),
            state.normalizer,  # type: ignore[arg-type]
            state.checkpoint_paths,
            foot_dataset_path,
        )
        threshold = float(record["validation"]["selected_threshold"])
        metrics = evaluate_branch_replays(
            runs,
            gates,
            traces,
            branch,
            threshold,
            int(document["validation"]["persistence_ms"]),
        )
        checks = branch_gate_results(
            metrics, document["holdout"][f"{branch}_gates"], branch
        )
        branch_results[branch] = {
            "selection": candidate_identity(state),
            "metrics": metrics,
            "gates": checks,
            "passed": all(checks.values()),
        }
        metrics_by_branch[branch] = metrics
        del traces
    slip_events = {
        row["run_id"]: row for row in metrics_by_branch["slip"]["event_rows"]
    }
    support_events = {
        row["run_id"]: row for row in metrics_by_branch["support"]["event_rows"]
    }
    event_ids = set(slip_events) | set(support_events)
    detected = sum(
        bool(slip_events.get(run_id, {}).get("valid_detection"))
        or bool(support_events.get(run_id, {}).get("valid_detection"))
        for run_id in event_ids
    )
    noevent_rows = metrics_by_branch["support"]["benign_rows"]
    hard_ids = {
        row["run_id"]
        for branch in ("slip", "support")
        for row in metrics_by_branch[branch]["hard_rows"]
    }
    hard_fp_ids = {
        row["run_id"]
        for branch in ("slip", "support")
        for row in metrics_by_branch[branch]["hard_rows"]
        if row["system_false_alert"]
    }
    integrated = {
        "event_runs": len(event_ids),
        "physical_event_recall": 0.0 if not event_ids else detected / len(event_ids),
        "no_event_transition_runs": len(noevent_rows),
        "no_event_transition_specificity": 1.0
        if not noevent_rows
        else 1.0
        - sum(row["system_false_alert"] for row in noevent_rows) / len(noevent_rows),
        "hard_ground_runs": len(hard_ids),
        "hard_ground_specificity": 1.0
        if not hard_ids
        else 1.0 - len(hard_fp_ids) / len(hard_ids),
    }
    gates_config = document["holdout"]["integrated_gates"]
    integrated_checks = {
        "physical_event_recall": integrated["physical_event_recall"]
        >= float(gates_config["physical_event_recall_min"]),
        "no_event_transition_specificity": integrated["no_event_transition_specificity"]
        >= float(gates_config["no_event_transition_specificity_min"]),
        "hard_ground_specificity": integrated["hard_ground_specificity"]
        >= float(gates_config["hard_ground_specificity_min"]),
    }
    return {
        "performed": True,
        "guard_open_count": 1,
        "reselection_performed": False,
        "branches": branch_results,
        "integrated": {
            "metrics": integrated,
            "gates": integrated_checks,
            "passed": all(integrated_checks.values()),
        },
        "passed": all(row["passed"] for row in branch_results.values())
        and all(integrated_checks.values()),
    }


def _candidate_grid(
    phase: str, branch: str
) -> tuple[tuple[str, tuple[str, ...], str, int], ...]:
    candidates = PHASE_A_CANDIDATES if phase == "PHASE_A" else PHASE_B_CANDIDATES
    return tuple(
        (candidate_id, components, family, history)
        for candidate_id, components in candidates[branch].items()
        for family in ("mlp", "gru")
        for history in (20, 50)
    )


def _run_phase_training(
    document: Mapping[str, object],
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    branches: Sequence[str],
    phase: str,
    artifact_path: Path,
    foot_dataset_path: Path | None,
    progress: Callable[[str], None],
) -> dict[str, list[CandidateState]]:
    states: dict[str, list[CandidateState]] = {branch: [] for branch in branches}
    for branch in branches:
        for candidate_id, components, family, history in _candidate_grid(phase, branch):
            state = train_branch_candidate(
                document,
                runs,
                gates,
                branch,
                phase,
                candidate_id,
                components,
                family,
                history,
                artifact_path,
                foot_dataset_path,
                progress,
            )
            states[branch].append(state)
    return states


def _run_phase_validation(
    document: Mapping[str, object],
    states: Mapping[str, Sequence[CandidateState]],
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    artifact_path: Path,
    foot_dataset_path: Path | None,
    progress: Callable[[str], None],
) -> dict[str, dict[str, object]]:
    selections = {}
    for branch, branch_states in states.items():
        for state in branch_states:
            result = validate_branch_candidate(
                document,
                state,
                runs,
                gates,
                artifact_path,
                foot_dataset_path,
            )
            record = state.record
            progress(
                f"{record['phase']} VALIDATION {branch.upper()} "
                f"{record['candidate_id']} {record['model_family']} "
                f"{record['history_ms']}ms recall={result['metrics']['event_recall']:.3f} "
                f"passed={result['passed']}"
            )
        selections[branch] = select_branch_candidate(branch_states)
    return selections


def _public_candidate_records(
    states: Mapping[str, Sequence[CandidateState]]
) -> dict[str, list[dict[str, object]]]:
    return {
        branch: [state.record for state in branch_states]
        for branch, branch_states in states.items()
    }


def _verdict(
    phase_a_selection: Mapping[str, Mapping[str, object]],
    final_selection: Mapping[str, Mapping[str, object]],
    holdout: Mapping[str, object],
) -> tuple[str, str]:
    both = all(
        final_selection[branch].get("selected") is not None
        for branch in ("slip", "support")
    )
    if both and bool(holdout.get("passed")):
        foot = any(
            "foot_imu12" in final_selection[branch]["selected"]["components"]
            for branch in ("slip", "support")
        )
        return (
            "TERRAIN_CONDITIONED_REFLEX_SUPPORTED_WITH_FOOT_IMU"
            if foot
            else "TERRAIN_CONDITIONED_REFLEX_SUPPORTED_EXISTING_SENSORS",
            "BOTH_BRANCHES_SUPPORTED",
        )
    slip = final_selection["slip"].get("selected") is not None
    support = final_selection["support"].get("selected") is not None
    if slip or support:
        return (
            "TERRAIN_CONDITIONED_REFLEX_PARTIALLY_SUPPORTED",
            "SLIP_ONLY" if slip else "SUPPORT_ONLY",
        )
    return "TERRAIN_CONDITIONED_REFLEX_NOT_SUPPORTED", "NO_BRANCH_SUPPORTED"


def run_terrain_conditioned_reflex_detector(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Run the complete bounded Phase A -> conditional Phase B workflow."""
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    document = _load_yaml(config_path)
    if (
        document["experiment"]["id"]
        != "TERRAIN_CONDITIONED_REFLEX_DETECTOR_DEVELOPMENT"
    ):
        raise ValueError("unsupported terrain-conditioned reflex experiment")
    event_config_path = repository_root / str(document["source"]["event_config"])
    event_document = _load_yaml(event_config_path)
    dense_document = _load_yaml(
        repository_root / str(event_document["source"]["dense_design_config"])
    )
    primary_specs, control_specs = generate_event_specifications(
        event_document, dense_document
    )
    specifications = [*primary_specs, *control_specs]
    event_dataset = repository_root / str(document["source"]["event_dataset"])
    with (event_dataset / "manifest.json").open("r", encoding="utf-8") as stream:
        event_manifest = json.load(stream)
    if int(event_manifest["run_count"]) != 256:
        raise ValueError("event corpus must remain 240 transitions plus 16 controls")
    artifact_path = repository_root / str(document["artifacts"]["path"])
    artifact_path.mkdir(parents=True, exist_ok=True)
    protected_paths = [str(value) for value in document["protected_terrain_paths"]]
    terrain_before = _protected_hashes(repository_root, protected_paths)

    dev_runs = load_event_runs(event_dataset, event_manifest, ("train", "validation"))
    dev_specs = [
        row for row in specifications if row["split"] in ("train", "validation")
    ]
    dev_gates, replay_provenance = generate_terrain_gate_cache(
        document,
        event_document,
        dev_specs,
        dev_runs,
        repository_root,
        artifact_path / "terrain_gate" / "development",
        progress,
    )
    timing = terrain_timing_audit(dev_runs, dev_gates)
    _write_json(artifact_path / "terrain_timing.json", timing)

    progress("PHASE A training begins; VALIDATION remains unopened by HNM")
    phase_a_states = _run_phase_training(
        document,
        dev_runs,
        dev_gates,
        ("slip", "support"),
        "PHASE_A",
        artifact_path,
        None,
        progress,
    )
    progress("PHASE A all Round0->2 training complete; opening VALIDATION")
    phase_a_selection = _run_phase_validation(
        document,
        phase_a_states,
        dev_runs,
        dev_gates,
        artifact_path,
        None,
        progress,
    )
    failed = [
        branch
        for branch in ("slip", "support")
        if phase_a_selection[branch]["selected"] is None
    ]

    foot_dataset_path: Path | None = None
    foot_dataset_summary: dict[str, object] = {
        "generated": False,
        "reason": "both_phase_a_branches_selected" if not failed else "pending",
    }
    phase_b_states: dict[str, list[CandidateState]] = {}
    phase_b_selection: dict[str, dict[str, object]] = {}
    if failed:
        progress(f"PHASE B activated only for failed branches: {failed}")
        observer_parity = validate_foot_imu_observer_parity(
            load_simulation_config(
                repository_root / str(document["source"]["simulator_config"])
            ),
            repository_root / str(document["source"]["policy_path"]),
            progress,
        )
        if not bool(observer_parity["passed"]):
            raise RuntimeError("Foot IMU observer parity failed before Phase B")
        _write_json(artifact_path / "foot_imu_observer_parity.json", observer_parity)
        foot_dataset_path, foot_dataset_summary = generate_foot_imu_dataset(
            document,
            event_document,
            event_manifest,
            specifications,
            dev_runs,
            repository_root,
            progress,
        )
        foot_dataset_summary = {
            "generated": True,
            **foot_dataset_summary,
            "matched_observer_parity": observer_parity,
        }
        phase_b_states = _run_phase_training(
            document,
            dev_runs,
            dev_gates,
            failed,
            "PHASE_B",
            artifact_path,
            foot_dataset_path,
            progress,
        )
        progress("PHASE B all Round0->2 training complete; opening VALIDATION")
        phase_b_selection = _run_phase_validation(
            document,
            phase_b_states,
            dev_runs,
            dev_gates,
            artifact_path,
            foot_dataset_path,
            progress,
        )

    final_selection = {}
    selected_states: dict[str, CandidateState] = {}
    for branch in ("slip", "support"):
        choice = phase_a_selection[branch]
        states = phase_a_states[branch]
        if choice["selected"] is None and branch in phase_b_selection:
            choice = phase_b_selection[branch]
            states = phase_b_states[branch]
        final_selection[branch] = choice
        if choice["selected"] is not None:
            selected_states[branch] = _state_by_identity(states, choice["selected"])
    _write_json(
        artifact_path / "selection_before_holdout.json",
        {"final_selection": final_selection, "holdout_opened": False},
    )

    guard = BranchHoldoutGuard()
    if len(selected_states) == 2:
        guard.open_once()
        holdout_runs = load_event_runs(
            event_dataset, event_manifest, ("holdout",), holdout_guard=guard
        )
        if foot_dataset_path is not None:
            holdout_gates = holdout_gate_from_observer_dataset(
                document, holdout_runs, foot_dataset_path, repository_root
            )
        else:
            holdout_specs = [row for row in specifications if row["split"] == "holdout"]
            holdout_gates, _ = generate_terrain_gate_cache(
                document,
                event_document,
                holdout_specs,
                holdout_runs,
                repository_root,
                artifact_path / "terrain_gate" / "holdout",
                progress,
            )
        holdout = evaluate_selected_holdout(
            document,
            selected_states,
            holdout_runs,
            holdout_gates,
            artifact_path,
            foot_dataset_path,
        )
        holdout["guard_open_count"] = guard.open_count
    else:
        holdout = {
            "performed": False,
            "guard_open_count": guard.open_count,
            "reason": "both_branches_not_selected",
        }

    terrain_after = _protected_hashes(repository_root, protected_paths)
    verdict, supported_branch = _verdict(phase_a_selection, final_selection, holdout)
    metrics = {
        "experiment": document["experiment"],
        "development_corpus": {
            "dataset_id": event_manifest["dataset_id"],
            "manifest_sha256": _file_sha256(event_dataset / "manifest.json"),
            "run_count": event_manifest["run_count"],
            "train_runs": sum(
                row["split"] == "train" for row in event_manifest["runs"]
            ),
            "validation_runs": sum(
                row["split"] == "validation" for row in event_manifest["runs"]
            ),
            "holdout_runs": sum(
                row["split"] == "holdout" for row in event_manifest["runs"]
            ),
        },
        "terrain_gate_replay": replay_provenance,
        "terrain_timing": timing,
        "phase_a": {
            "candidates": _public_candidate_records(phase_a_states),
            "selection": phase_a_selection,
        },
        "phase_b": {
            "activated_branches": failed,
            "foot_imu_dataset": foot_dataset_summary,
            "candidates": _public_candidate_records(phase_b_states),
            "selection": phase_b_selection,
        },
        "final_selection": final_selection,
        "holdout": holdout,
        "terrain_regression": {
            "passed": terrain_before == terrain_after,
            "retrained": False,
            "before": terrain_before,
            "after": terrain_after,
        },
        "fusion_regression": fusion_regression(),
        "runtime_boundary": {
            "terrain_truth_in_branch_gate": False,
            "fall_or_recovery_in_label": False,
            "q_dq_torque_added": False,
            "severity_classifier_trained": False,
            "recovery_controller_changed": False,
            "final_sensor_architecture_frozen": False,
        },
        "verdict": verdict,
        "supported_branch": supported_branch,
    }
    _write_json(artifact_path / "metrics.json", metrics)
    progress(json.dumps({"verdict": verdict, "selection": final_selection}, indent=2))
    return artifact_path, metrics
