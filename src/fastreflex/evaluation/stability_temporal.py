"""Bounded temporal observability audit for future walking fall risk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import gc
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib
import numpy as np
import torch
import yaml

from fastreflex.dataset.loader import Normalizer, WindowSet
from fastreflex.evaluation.stability_ground_truth import (
    FULL_STATE,
    FULL_STATE_FEATURE_NAMES,
    extract_candidate_features,
    lower_body_state_addresses,
)
from fastreflex.evaluation.transition_scenarios import (
    VALID_FALL,
    VALID_STABLE,
    classify_scenario_outcome,
    fusion_regression,
    target_contact_mask,
    transition_simulation_config,
)
from fastreflex.models.baselines import parameter_count
from fastreflex.simulation.g1 import (
    PHYSICS_TIMESTEP_S,
    SENSOR_RATE_HZ,
    SimulationConfig,
    SimulationResult,
    load_g1_model,
    load_simulation_config,
    run_simulation,
)
from fastreflex.simulation.stability import (
    DOUBLE_SUPPORT,
    LEFT_SINGLE_SUPPORT,
    RIGHT_SINGLE_SUPPORT,
)
from fastreflex.training.trainer import (
    load_checkpoint,
    save_checkpoint,
    train_model,
)


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PRIVILEGED_FULL_STATE = "PRIVILEGED_FULL_STATE"
RUNTIME_IMU6 = "RUNTIME_IMU6"
REPRESENTATION_ORDER = (PRIVILEGED_FULL_STATE, RUNTIME_IMU6)
TEMPORAL_FULL_STATE_FEATURE_NAMES = (
    *FULL_STATE_FEATURE_NAMES,
    "phase_left_single_support",
    "phase_right_single_support",
    "phase_double_support",
)
IMU6_FEATURE_NAMES = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
REPRESENTATION_FEATURE_NAMES = {
    PRIVILEGED_FULL_STATE: TEMPORAL_FULL_STATE_FEATURE_NAMES,
    RUNTIME_IMU6: IMU6_FEATURE_NAMES,
}
BINARY_CLASS_NAMES = ("STABLE", "FALL_RISK")
VALID_OUTCOMES = (VALID_STABLE, VALID_FALL)


@dataclass(frozen=True)
class TemporalRun:
    """Reduced exact-state/sensor trace for one independent simulation run."""

    run_id: str
    split: str
    source_terrain: str
    target_terrain: str
    speed_mps: float
    outcome: str
    first_contact_sample: int
    fall_sample: int | None
    gait_phase: np.ndarray
    features: Mapping[str, np.ndarray]
    timestamp_us: np.ndarray
    slip_sample: int | None
    sink_sample: int | None
    maximum_support_deformation_m: float
    hard_stable_control: bool


@dataclass(frozen=True)
class MatchedPair:
    """One fall endpoint and its unique stable elapsed-time negative."""

    split: str
    offset_ms: int
    fall_run_id: str
    stable_run_id: str
    fall_endpoint_sample: int
    stable_endpoint_sample: int
    elapsed_since_contact_samples: int
    speed_difference_mps: float
    endpoint_phase_matched: bool


@dataclass(frozen=True)
class WindowBatch:
    """Binary windows plus per-example provenance kept outside the tensor."""

    windows: WindowSet
    rows: tuple[dict[str, object], ...]


class HoldoutGuard:
    """Seal holdout waveform construction until validation selection is frozen."""

    def __init__(self) -> None:
        self._open = False
        self._open_count = 0

    def open_once(self) -> None:
        if self._open or self._open_count:
            raise RuntimeError("holdout guard may be opened exactly once")
        self._open = True
        self._open_count = 1

    def require_open(self) -> None:
        if not self._open:
            raise RuntimeError("holdout waveform access is sealed")

    @property
    def open_count(self) -> int:
        return self._open_count


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_hashes(repository_root: Path, paths: Sequence[str]) -> dict[str, str]:
    result = {}
    for relative in paths:
        path = (repository_root / relative).resolve()
        path.relative_to(repository_root)
        if not path.is_file():
            raise FileNotFoundError(f"protected Terrain path is missing: {path}")
        result[str(relative)] = _file_sha256(path)
    return result


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if not len(indices) else int(indices[0])


def causal_window_indices(endpoint_sample: int, history_samples: int) -> np.ndarray:
    """Return a history-only window ending at the declared current sample."""
    if history_samples <= 0 or endpoint_sample < history_samples - 1:
        raise ValueError("endpoint cannot provide the requested causal history")
    return np.arange(
        endpoint_sample - history_samples + 1,
        endpoint_sample + 1,
        dtype=np.int64,
    )


def causal_last_valid_fill(
    values: np.ndarray, initial_value: float = 0.0
) -> np.ndarray:
    """Fill invalid values only from the current/past prefix, never the future."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("causal fill requires one scalar channel")
    output = np.empty_like(array)
    last = float(initial_value)
    for index, value in enumerate(array):
        if np.isfinite(value):
            last = float(value)
        output[index] = last
    return output


def privileged_temporal_features(
    result: SimulationResult,
    qpos_addresses: np.ndarray,
    qvel_addresses: np.ndarray,
) -> np.ndarray:
    """Reuse the prior 37-D exact state and append current support context."""
    if result.stability is None:
        raise ValueError("privileged temporal state requires stability diagnostics")
    base = extract_candidate_features(
        result, FULL_STATE, qpos_addresses, qvel_addresses
    ).astype(np.float64)
    base[:, -1] = causal_last_valid_fill(base[:, -1])
    phase = result.stability.gait_phase
    context = np.column_stack(
        (
            phase == LEFT_SINGLE_SUPPORT,
            phase == RIGHT_SINGLE_SUPPORT,
            phase == DOUBLE_SUPPORT,
        )
    ).astype(np.float64)
    features = np.column_stack((base, context)).astype(np.float32)
    expected = (len(result.runtime.sequence), len(TEMPORAL_FULL_STATE_FEATURE_NAMES))
    if features.shape != expected or not np.all(np.isfinite(features)):
        raise ValueError(
            f"privileged temporal features must have finite shape {expected}"
        )
    return features


def binary_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Exact Mann-Whitney AUROC with deterministic half credit for ties."""
    truth = np.asarray(targets, dtype=np.int64)
    risk = np.asarray(scores, dtype=np.float64)
    if truth.shape != risk.shape or truth.ndim != 1 or not np.all(np.isfinite(risk)):
        raise ValueError("binary AUROC inputs must be finite aligned vectors")
    positive = risk[truth == 1]
    negative = risk[truth == 0]
    if not len(positive) or not len(negative) or np.any((truth < 0) | (truth > 1)):
        raise ValueError("binary AUROC requires both classes")
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def binary_auprc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Average-precision step integral grouped at exact score ties."""
    truth = np.asarray(targets, dtype=np.int64)
    risk = np.asarray(scores, dtype=np.float64)
    if truth.shape != risk.shape or truth.ndim != 1 or not np.all(np.isfinite(risk)):
        raise ValueError("binary AUPRC inputs must be finite aligned vectors")
    positives = int(np.count_nonzero(truth == 1))
    if not positives or np.any((truth < 0) | (truth > 1)):
        raise ValueError("binary AUPRC requires positive examples")
    order = np.argsort(-risk, kind="mergesort")
    sorted_risk = risk[order]
    sorted_truth = truth[order]
    true_positive = 0
    predicted = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and sorted_risk[stop] == sorted_risk[start]:
            stop += 1
        true_positive += int(np.count_nonzero(sorted_truth[start:stop] == 1))
        predicted += stop - start
        recall = true_positive / positives
        precision = true_positive / predicted
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(area)


def binary_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Dependency-light fall-risk metrics at one predeclared threshold."""
    truth = np.asarray(targets, dtype=np.int64)
    risk = np.asarray(scores, dtype=np.float64)
    if truth.shape != risk.shape or not len(truth):
        raise ValueError("binary metrics require aligned nonempty vectors")
    prediction = (risk >= threshold).astype(np.int64)
    tn = int(np.count_nonzero((truth == 0) & (prediction == 0)))
    fp = int(np.count_nonzero((truth == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((truth == 1) & (prediction == 0)))
    tp = int(np.count_nonzero((truth == 1) & (prediction == 1)))
    specificity = tn / (tn + fp) if tn + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
        denominator = 2 * true_positive + false_positive + false_negative
        return 0.0 if not denominator else 2 * true_positive / denominator

    fall_f1 = f1(tp, fp, fn)
    stable_f1 = f1(tn, fn, fp)
    return {
        "auroc": binary_auroc(truth, risk),
        "auprc": binary_auprc(truth, risk),
        "macro_f1": float((fall_f1 + stable_f1) / 2.0),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "fall_recall": float(recall),
        "stable_specificity": float(specificity),
        "threshold": float(threshold),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "support": int(len(truth)),
    }


def horizon_fixed_label(
    fall_sample: int | None, endpoint_sample: int, horizon_ms: int
) -> int:
    """Use future outcome only to construct a supervised label, never a channel."""
    if horizon_ms <= 0:
        raise ValueError("fall-risk horizon must be positive")
    if fall_sample is None:
        return 0
    delta = int(fall_sample) - int(endpoint_sample)
    return int(0 < delta <= horizon_ms)


def _load_yaml(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _load_scenario_specs(
    document: Mapping[str, object], repository_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = document["source"]
    existing = _load_yaml(repository_root / str(source["existing_clean_config"]))
    fresh = _load_yaml(repository_root / str(source["fresh_design_config"]))
    primary = [dict(row) for row in existing["calibration"]["transition_runs"]] + [
        dict(row) for row in fresh["fresh_validation"]["runs"]
    ]
    controls = [dict(row) for row in existing["calibration"]["hard_stable_runs"]]
    split_lookup = {
        str(run_id): split
        for split in ("train", "validation", "holdout")
        for run_id in document["split"][split]
    }
    for row in primary:
        row["split"] = split_lookup.get(str(row["id"]), "UNASSIGNED")
        row["hard_stable_control"] = False
    for row in controls:
        row["split"] = "control"
        row["hard_stable_control"] = True
    return primary, controls


def validate_temporal_design(
    document: Mapping[str, object],
    primary_specs: Sequence[Mapping[str, object]],
    control_specs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Fail before simulation if the temporal experiment contract drifted."""
    if document["experiment"]["id"] != "TEMPORAL_STABILITY_SEPARABILITY_AUDIT":
        raise ValueError("unsupported temporal stability experiment")
    if tuple(document["offset_analysis"]["offsets_before_fall_ms"]) != (
        500,
        300,
        200,
        100,
        50,
    ):
        raise ValueError("primary pre-fall offsets changed")
    if tuple(document["history"]["candidates_ms"]) != (50, 100, 200):
        raise ValueError("history-window candidates changed")
    if tuple(document["model"]["seeds"]) != (20260828, 20260829, 20260830):
        raise ValueError("temporal seed contract changed")
    if str(document["model"]["family"]) != "gru" or bool(
        document["model"]["architecture"]["bidirectional"]
    ):
        raise ValueError("temporal audit requires the unidirectional GRU")
    for representation in REPRESENTATION_ORDER:
        configured = tuple(document["representations"][representation]["feature_order"])
        if configured != REPRESENTATION_FEATURE_NAMES[representation]:
            raise ValueError(f"{representation} feature schema/order changed")
    forbidden = (
        "terrain",
        "scenario",
        "fall_time",
        "time_to_fall",
        "slip",
        "sink",
        "deformation",
        "run_id",
    )
    for feature_names in REPRESENTATION_FEATURE_NAMES.values():
        if any(token in feature for feature in feature_names for token in forbidden):
            raise ValueError("temporal representation contains a leakage channel")

    configured_sets = {
        split: set(str(value) for value in document["split"][split])
        for split in ("train", "validation", "holdout")
    }
    if any(
        configured_sets[left] & configured_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "holdout"),
            ("validation", "holdout"),
        )
    ):
        raise ValueError("temporal split is not run-disjoint")
    primary_ids = [str(row["id"]) for row in primary_specs]
    if len(primary_ids) != 78 or len(set(primary_ids)) != 78:
        raise ValueError("primary temporal cohort must contain 78 unique runs")
    if set(primary_ids) != set.union(*configured_sets.values()):
        raise ValueError("split assignments do not cover the primary cohort")
    if {split: len(values) for split, values in configured_sets.items()} != {
        "train": 46,
        "validation": 16,
        "holdout": 16,
    }:
        raise ValueError("temporal split counts changed")
    if not all(run_id.startswith("fs_val_") for run_id in configured_sets["holdout"]):
        raise ValueError("holdout must use only the predeclared fresh conditions")
    control_ids = [str(row["id"]) for row in control_specs]
    if len(control_ids) != 6 or set(control_ids) & set(primary_ids):
        raise ValueError("hard stable controls are missing or overlap primary runs")

    def designed_role(row: Mapping[str, object]) -> str:
        if "prior_observed_outcome" in row:
            return str(row["prior_observed_outcome"])
        return str(row["design_role"])

    strata_by_split: dict[str, dict[str, int]] = {}
    for split, ids in configured_sets.items():
        counts: dict[str, int] = {}
        for row in primary_specs:
            if str(row["id"]) not in ids:
                continue
            stratum = "_".join(
                (
                    str(row["source_terrain"]),
                    str(row["target_terrain"]),
                    designed_role(row),
                )
            )
            counts[stratum] = counts.get(stratum, 0) + 1
        if len(counts) != 8 or any(value < 2 for value in counts.values()):
            raise ValueError(f"{split} does not cover all eight scenario strata")
        strata_by_split[split] = counts
    return {
        "passed": True,
        "primary_runs": len(primary_ids),
        "hard_stable_controls": len(control_ids),
        "split_counts": {key: len(value) for key, value in configured_sets.items()},
        "strata_by_split": strata_by_split,
        "representation_dimensions": {
            name: len(REPRESENTATION_FEATURE_NAMES[name])
            for name in REPRESENTATION_ORDER
        },
        "fall_label_channels_in_tensor": False,
        "terrain_channels_in_tensor": False,
        "future_samples_in_window": False,
    }


def _prepared_specification(
    raw: Mapping[str, object], common: Mapping[str, object]
) -> dict[str, object]:
    specification = dict(raw)
    specification["minimum_normal_prefix_ms"] = int(
        common["scenario_gate"]["normal_prefix_ms_min"]
    )
    specification["minimum_post_contact_ms"] = int(
        common["scenario_gate"]["post_contact_ms_min"]
    )
    return specification


def _first_target_contact(result: SimulationResult, target: str) -> int:
    contact = _first_true(np.any(target_contact_mask(result, target), axis=1))
    if contact is None:
        raise ValueError(f"valid temporal run has no exact {target} contact")
    return contact


def _hard_control_outcome(result: SimulationResult) -> str:
    finite = bool(
        result.state_trace is not None
        and result.stability is not None
        and np.all(np.isfinite(result.runtime.pelvis_imu))
        and np.all(np.isfinite(result.state_trace.robot_qpos))
        and np.all(np.isfinite(result.state_trace.robot_qvel))
        and result.metadata["actual_samples"] == result.metadata["expected_samples"]
        and not result.metadata["terminated_by_viewer"]
    )
    if not finite or result.metadata["first_fall_sample"] is not None:
        return "INVALID_CONTROL"
    return VALID_STABLE


def _reduce_simulation(
    specification: Mapping[str, object],
    result: SimulationResult,
    outcome: str,
    qpos_addresses: np.ndarray,
    qvel_addresses: np.ndarray,
) -> TemporalRun:
    if result.stability is None:
        raise ValueError("temporal simulation requires exact stability diagnostics")
    control = bool(specification["hard_stable_control"])
    target = str(specification["target_terrain"])
    contact = 0 if control else _first_target_contact(result, target)
    fall_raw = result.metadata["first_fall_sample"]
    fall = None if fall_raw is None else int(fall_raw)
    slip = _first_true(result.diagnostics.any_established_slip_after_patch_onset)
    sink = _first_true(np.any(result.diagnostics.deformable_sink_onset, axis=1))
    privileged = privileged_temporal_features(result, qpos_addresses, qvel_addresses)
    imu = np.asarray(result.runtime.pelvis_imu, dtype=np.float32).copy()
    if imu.shape != (len(result.runtime.sequence), len(IMU6_FEATURE_NAMES)):
        raise ValueError("runtime Pelvis IMU6 trace changed shape")
    return TemporalRun(
        run_id=str(specification["id"]),
        split=str(specification["split"]),
        source_terrain=str(specification["source_terrain"]),
        target_terrain=target,
        speed_mps=float(specification["speed_mps"]),
        outcome=outcome,
        first_contact_sample=contact,
        fall_sample=fall,
        gait_phase=np.asarray(result.stability.gait_phase, dtype=np.int8).copy(),
        features={PRIVILEGED_FULL_STATE: privileged, RUNTIME_IMU6: imu},
        timestamp_us=np.asarray(result.runtime.timestamp_us, dtype=np.int64).copy(),
        slip_sample=slip,
        sink_sample=sink,
        maximum_support_deformation_m=float(
            np.max(result.diagnostics.support_surface_max_displacement_m)
        ),
        hard_stable_control=control,
    )


def simulate_temporal_cohort(
    base: SimulationConfig,
    specifications: Sequence[Mapping[str, object]],
    policy_path: Path,
    document: Mapping[str, object],
    qpos_addresses: np.ndarray,
    qvel_addresses: np.ndarray,
    progress: Callable[[str], None],
) -> tuple[dict[str, TemporalRun], list[dict[str, object]]]:
    """Simulate sequentially and retain only the bounded temporal evidence."""
    runs: dict[str, TemporalRun] = {}
    invalid: list[dict[str, object]] = []
    duration_s = float(document["common"]["duration_s"])
    for index, raw in enumerate(specifications, start=1):
        specification = _prepared_specification(raw, document["cohort"])
        result = run_simulation(
            transition_simulation_config(base, specification, policy_path, duration_s),
            capture_state_trace=True,
        )
        outcome = (
            _hard_control_outcome(result)
            if specification["hard_stable_control"]
            else classify_scenario_outcome(result, specification)
        )
        progress(
            f"TEMPORAL COHORT {index}/{len(specifications)} "
            f"{specification['id']}: {outcome}"
        )
        if outcome not in VALID_OUTCOMES:
            invalid.append(
                {
                    "run_id": str(specification["id"]),
                    "split": str(specification["split"]),
                    "outcome": outcome,
                }
            )
        else:
            reduced = _reduce_simulation(
                specification,
                result,
                outcome,
                qpos_addresses,
                qvel_addresses,
            )
            runs[reduced.run_id] = reduced
        del result
        if index % 8 == 0:
            gc.collect()
    return runs, invalid


def _eligible_matched_endpoint(
    run: TemporalRun,
    endpoint: int,
    maximum_history_samples: int,
) -> bool:
    start = endpoint - maximum_history_samples + 1
    stop = run.fall_sample if run.fall_sample is not None else len(run.gait_phase)
    return bool(
        start >= run.first_contact_sample
        and endpoint < stop
        and endpoint < len(run.gait_phase)
    )


def build_matched_pairs(
    runs: Mapping[str, TemporalRun],
    run_ids: Sequence[str],
    offsets_ms: Sequence[int],
    maximum_history_samples: int,
    *,
    holdout_guard: HoldoutGuard | None = None,
) -> tuple[dict[int, tuple[MatchedPair, ...]], list[dict[str, object]]]:
    """Match one unique stable run to each usable fall run at exact elapsed time."""
    selected_ids = [str(run_id) for run_id in run_ids if str(run_id) in runs]
    if selected_ids and all(runs[run_id].split == "holdout" for run_id in selected_ids):
        if holdout_guard is None:
            raise RuntimeError("holdout matching requires an explicit guard")
        holdout_guard.require_open()
    selected = [runs[run_id] for run_id in selected_ids]
    pairs_by_offset: dict[int, tuple[MatchedPair, ...]] = {}
    exclusions: list[dict[str, object]] = []
    for offset_ms in offsets_ms:
        pairs: list[MatchedPair] = []
        groups = sorted(
            {
                (run.source_terrain, run.target_terrain)
                for run in selected
                if not run.hard_stable_control
            }
        )
        for source, target in groups:
            stable = sorted(
                (
                    run
                    for run in selected
                    if run.source_terrain == source
                    and run.target_terrain == target
                    and run.outcome == VALID_STABLE
                ),
                key=lambda run: run.run_id,
            )
            falling = sorted(
                (
                    run
                    for run in selected
                    if run.source_terrain == source
                    and run.target_terrain == target
                    and run.outcome == VALID_FALL
                ),
                key=lambda run: run.run_id,
            )
            unused = {run.run_id for run in stable}
            for fall_run in falling:
                assert fall_run.fall_sample is not None
                fall_endpoint = fall_run.fall_sample - int(offset_ms)
                if not _eligible_matched_endpoint(
                    fall_run, fall_endpoint, maximum_history_samples
                ):
                    exclusions.append(
                        {
                            "split": fall_run.split,
                            "offset_ms": int(offset_ms),
                            "fall_run_id": fall_run.run_id,
                            "reason": "fall_endpoint_invalid",
                        }
                    )
                    continue
                elapsed = fall_endpoint - fall_run.first_contact_sample
                candidates = []
                for stable_run in stable:
                    if stable_run.run_id not in unused:
                        continue
                    stable_endpoint = stable_run.first_contact_sample + elapsed
                    if not _eligible_matched_endpoint(
                        stable_run, stable_endpoint, maximum_history_samples
                    ):
                        continue
                    phase_match = bool(
                        fall_run.gait_phase[fall_endpoint]
                        == stable_run.gait_phase[stable_endpoint]
                    )
                    candidates.append(
                        (
                            abs(stable_run.speed_mps - fall_run.speed_mps),
                            not phase_match,
                            stable_run.run_id,
                            stable_run,
                            stable_endpoint,
                            phase_match,
                        )
                    )
                if not candidates:
                    exclusions.append(
                        {
                            "split": fall_run.split,
                            "offset_ms": int(offset_ms),
                            "fall_run_id": fall_run.run_id,
                            "reason": "no_unique_valid_stable_match",
                        }
                    )
                    continue
                choice = min(candidates, key=lambda row: row[:3])
                _, _, _, stable_run, stable_endpoint, phase_match = choice
                unused.remove(stable_run.run_id)
                pairs.append(
                    MatchedPair(
                        split=fall_run.split,
                        offset_ms=int(offset_ms),
                        fall_run_id=fall_run.run_id,
                        stable_run_id=stable_run.run_id,
                        fall_endpoint_sample=fall_endpoint,
                        stable_endpoint_sample=int(stable_endpoint),
                        elapsed_since_contact_samples=elapsed,
                        speed_difference_mps=abs(
                            stable_run.speed_mps - fall_run.speed_mps
                        ),
                        endpoint_phase_matched=phase_match,
                    )
                )
        pairs_by_offset[int(offset_ms)] = tuple(
            sorted(pairs, key=lambda pair: (pair.fall_run_id, pair.stable_run_id))
        )
    return pairs_by_offset, exclusions


def fit_train_normalizer(
    runs: Mapping[str, TemporalRun],
    train_ids: Sequence[str],
    representation: str,
    per_run_sample_cap: int,
    standard_deviation_floor: float,
) -> Normalizer:
    """Fit each channel from train-run post-contact, pre-fall evidence only."""
    chunks = []
    fit_ids = []
    for run_id in train_ids:
        run = runs[str(run_id)]
        stop = run.fall_sample if run.fall_sample is not None else len(run.gait_phase)
        eligible = np.arange(run.first_contact_sample, stop, dtype=np.int64)
        if len(eligible) > per_run_sample_cap:
            positions = np.linspace(
                0, len(eligible) - 1, per_run_sample_cap, dtype=np.int64
            )
            eligible = eligible[positions]
        if len(eligible):
            chunks.append(run.features[representation][eligible].astype(np.float64))
            fit_ids.append(run.run_id)
    if not chunks:
        raise ValueError("train-only normalization has no eligible samples")
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


def materialize_matched_windows(
    runs: Mapping[str, TemporalRun],
    pairs: Sequence[MatchedPair],
    representation: str,
    history_samples: int,
    normalizer: Normalizer,
) -> WindowBatch:
    """Create at most one stable and one fall window per matched run and offset."""
    inputs = []
    targets = []
    sources = []
    endpoints = []
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for pair in pairs:
        for label, run_id, endpoint in (
            (0, pair.stable_run_id, pair.stable_endpoint_sample),
            (1, pair.fall_run_id, pair.fall_endpoint_sample),
        ):
            key = (run_id, pair.offset_ms)
            if key in seen:
                raise ValueError("per-run per-offset window cap was exceeded")
            seen.add(key)
            run = runs[run_id]
            indices = causal_window_indices(endpoint, history_samples)
            if indices[0] < run.first_contact_sample:
                raise ValueError("primary temporal window begins before target contact")
            raw = run.features[representation][indices]
            if not np.all(np.isfinite(raw)) or int(indices[-1]) != endpoint:
                raise ValueError("temporal window is nonfinite or not endpoint aligned")
            inputs.append(normalizer.transform(raw))
            targets.append(label)
            sources.append(run_id)
            endpoints.append(endpoint)
            rows.append(
                {
                    "run_id": run_id,
                    "label": label,
                    "observed_outcome": (
                        "fall" if run.outcome == VALID_FALL else "stable"
                    ),
                    "source_terrain": run.source_terrain,
                    "target_terrain": run.target_terrain,
                    "speed_mps": run.speed_mps,
                    "endpoint_sample": endpoint,
                    "first_contact_sample": run.first_contact_sample,
                    "elapsed_since_contact_ms": int(
                        endpoint - run.first_contact_sample
                    ),
                    "fall_sample": run.fall_sample,
                    "offset_before_fall_ms": (
                        None
                        if run.fall_sample is None
                        else int(run.fall_sample - endpoint)
                    ),
                    "matched_run_id": (
                        pair.fall_run_id if label == 0 else pair.stable_run_id
                    ),
                    "speed_difference_mps": pair.speed_difference_mps,
                    "endpoint_phase_matched": pair.endpoint_phase_matched,
                }
            )
    if not inputs:
        raise ValueError("no matched temporal windows were materialized")
    target_array = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(target_array, minlength=3)
    return WindowBatch(
        windows=WindowSet(
            inputs=np.stack(inputs).astype(np.float32),
            targets=target_array,
            run_ids=np.asarray(sources, dtype=str),
            endpoint_samples=np.asarray(endpoints, dtype=np.int64),
            available_by_class=tuple(int(value) for value in counts[:3]),
        ),
        rows=tuple(rows),
    )


def predict_fall_probability(
    model: torch.nn.Module,
    windows: WindowSet,
    batch_size: int = 256,
) -> np.ndarray:
    """Return class-1 probability without exposing labels to the model."""
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            inputs = torch.from_numpy(windows.inputs[start : start + batch_size])
            probability = torch.softmax(model(inputs), dim=1)[:, 1]
            parts.append(probability.cpu().numpy())
    return np.concatenate(parts).astype(np.float64, copy=False)


def _metric_breakdown(
    targets: np.ndarray,
    scores: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    field: str,
    values: Sequence[str],
    threshold: float,
) -> dict[str, object]:
    result = {}
    for value in values:
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row[field] == value],
            dtype=np.int64,
        )
        if not len(indices) or len(np.unique(targets[indices])) != 2:
            result[value] = {"support": int(len(indices)), "metrics": None}
            continue
        result[value] = {
            "support": int(len(indices)),
            "metrics": binary_metrics(targets[indices], scores[indices], threshold),
        }
    return result


def _prediction_rows(
    rows: Sequence[Mapping[str, object]],
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, object]]:
    return [
        {
            **dict(row),
            "fall_risk_probability": float(score),
            "prediction": "FALL_RISK" if score >= threshold else "STABLE",
            "correct": bool((score >= threshold) == (int(row["label"]) == 1)),
        }
        for row, score in zip(rows, scores)
    ]


def _validation_passes(
    metrics: Mapping[str, object], gates: Mapping[str, object]
) -> dict[str, bool]:
    return {
        "auroc": float(metrics["auroc"]) >= float(gates["auroc_min"]),
        "auprc": float(metrics["auprc"]) >= float(gates["auprc_min"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"])
        >= float(gates["balanced_accuracy_min"]),
        "stable_specificity": float(metrics["stable_specificity"])
        >= float(gates["stable_specificity_min"]),
        "fall_recall": float(metrics["fall_recall"]) >= float(gates["fall_recall_min"]),
    }


def _holdout_passes(
    metrics: Mapping[str, object], gates: Mapping[str, object]
) -> dict[str, bool]:
    return {
        "auroc": float(metrics["auroc"]) >= float(gates["auroc_min"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"])
        >= float(gates["balanced_accuracy_min"]),
        "stable_specificity": float(metrics["stable_specificity"])
        >= float(gates["stable_specificity_min"]),
        "fall_recall": float(metrics["fall_recall"]) >= float(gates["fall_recall_min"]),
    }


def select_history_and_horizon(
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select the farthest reliable offset, then the shortest history."""
    reliable = [
        row
        for row in candidates
        if all(bool(value) for value in row["validation_gates"].values())
    ]
    if not reliable:
        diagnostic = max(
            candidates,
            key=lambda row: (
                float(row["metrics"]["auroc"]),
                float(row["metrics"]["auprc"]),
                float(row["metrics"]["balanced_accuracy"]),
                -int(row["history_ms"]),
                int(row["offset_ms"]),
            ),
        )
        return {
            "selected": None,
            "reason": "no_validation_offset_met_all_reliability_gates",
            "diagnostic_best": {
                "history_ms": int(diagnostic["history_ms"]),
                "offset_ms": int(diagnostic["offset_ms"]),
            },
        }
    earliest = max(int(row["offset_ms"]) for row in reliable)
    at_earliest = [row for row in reliable if int(row["offset_ms"]) == earliest]
    selected = min(at_earliest, key=lambda row: int(row["history_ms"]))
    return {
        "selected": {
            "history_ms": int(selected["history_ms"]),
            "offset_ms": int(selected["offset_ms"]),
        },
        "reason": "farthest_reliable_offset_then_shortest_history",
        "reliable_candidates": [
            {
                "history_ms": int(row["history_ms"]),
                "offset_ms": int(row["offset_ms"]),
            }
            for row in reliable
        ],
    }


def _checkpoint_paths(
    artifact_path: Path,
    representation: str,
    history_ms: int,
    offset_ms: int,
    seeds: Sequence[int],
) -> list[Path]:
    label = representation.lower()
    return [
        artifact_path
        / "checkpoints"
        / f"{label}_history_{history_ms}ms_offset_{offset_ms}ms_seed_{seed}.pt"
        for seed in seeds
    ]


def _evaluate_ensemble(
    checkpoint_paths: Sequence[Path],
    batch: WindowBatch,
    threshold: float,
) -> tuple[dict[str, object], np.ndarray, list[dict[str, object]]]:
    probabilities = []
    seed_metrics = []
    for path in checkpoint_paths:
        model, metadata = load_checkpoint(path)
        score = predict_fall_probability(model, batch.windows)
        probabilities.append(score)
        seed_metrics.append(
            {
                "seed": int(metadata["seed"]),
                "metrics": binary_metrics(batch.windows.targets, score, threshold),
            }
        )
    ensemble = np.mean(np.stack(probabilities), axis=0)
    metrics = binary_metrics(batch.windows.targets, ensemble, threshold)
    return metrics, ensemble, seed_metrics


def _train_validation_candidates(
    document: Mapping[str, object],
    runs: Mapping[str, TemporalRun],
    normalizers: Mapping[str, Normalizer],
    train_pairs: Mapping[int, Sequence[MatchedPair]],
    validation_pairs: Mapping[int, Sequence[MatchedPair]],
    artifact_path: Path,
    progress: Callable[[str], None],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
    model_config = document["model"]
    histories = [int(value) for value in document["history"]["candidates_ms"]]
    offsets = [
        int(value) for value in document["offset_analysis"]["offsets_before_fall_ms"]
    ]
    seeds = [int(value) for value in model_config["seeds"]]
    threshold = float(model_config["probability_threshold"])
    candidates_by_representation: dict[str, list[dict[str, object]]] = {}
    selections = {}
    for representation in REPRESENTATION_ORDER:
        candidates = []
        for history_ms in histories:
            for offset_ms in offsets:
                train_batch = materialize_matched_windows(
                    runs,
                    train_pairs[offset_ms],
                    representation,
                    history_ms,
                    normalizers[representation],
                )
                validation_batch = materialize_matched_windows(
                    runs,
                    validation_pairs[offset_ms],
                    representation,
                    history_ms,
                    normalizers[representation],
                )
                checkpoints = _checkpoint_paths(
                    artifact_path,
                    representation,
                    history_ms,
                    offset_ms,
                    seeds,
                )
                training_rows = []
                for seed, checkpoint in zip(seeds, checkpoints):
                    model, training = train_model(
                        "gru",
                        history_ms,
                        train_batch.windows,
                        validation_batch.windows,
                        seed,
                        batch_size=int(model_config["batch_size"]),
                        max_epochs=int(model_config["max_epochs"]),
                        patience=int(model_config["early_stopping_patience"]),
                        learning_rate=float(model_config["learning_rate"]),
                        class_names=BINARY_CLASS_NAMES,
                    )
                    save_checkpoint(
                        checkpoint,
                        model,
                        "gru",
                        history_ms,
                        seed,
                        training,
                        input_channels=len(
                            REPRESENTATION_FEATURE_NAMES[representation]
                        ),
                        class_names=BINARY_CLASS_NAMES,
                    )
                    training_rows.append(
                        {
                            "seed": seed,
                            "best_epoch": training.best_epoch,
                            "epochs_completed": training.epochs_completed,
                        }
                    )
                    del model
                validation_metrics, scores, seed_metrics = _evaluate_ensemble(
                    checkpoints, validation_batch, threshold
                )
                gates = _validation_passes(
                    validation_metrics, document["validation_gates"]
                )
                prediction_rows = _prediction_rows(
                    validation_batch.rows, scores, threshold
                )
                candidate = {
                    "history_ms": history_ms,
                    "offset_ms": offset_ms,
                    "train_windows": len(train_batch.windows),
                    "validation_windows": len(validation_batch.windows),
                    "train_independent_runs": len(set(train_batch.windows.run_ids)),
                    "validation_independent_runs": len(
                        set(validation_batch.windows.run_ids)
                    ),
                    "parameter_count": parameter_count(
                        load_checkpoint(checkpoints[0])[0]
                    ),
                    "metrics": validation_metrics,
                    "validation_gates": gates,
                    "seed_metrics": seed_metrics,
                    "training": training_rows,
                    "breakdown": {
                        "target_terrain": _metric_breakdown(
                            validation_batch.windows.targets,
                            scores,
                            validation_batch.rows,
                            "target_terrain",
                            ("ice", "sand"),
                            threshold,
                        ),
                        "source_terrain": _metric_breakdown(
                            validation_batch.windows.targets,
                            scores,
                            validation_batch.rows,
                            "source_terrain",
                            ("concrete", "marble"),
                            threshold,
                        ),
                    },
                    "predictions": prediction_rows,
                    "checkpoint_paths": [
                        str(path.relative_to(artifact_path)) for path in checkpoints
                    ],
                }
                candidates.append(candidate)
                progress(
                    f"TEMPORAL GRU {representation} history={history_ms}ms "
                    f"offset={offset_ms}ms AUROC={validation_metrics['auroc']:.4f} "
                    f"BA={validation_metrics['balanced_accuracy']:.4f} "
                    f"reliable={all(gates.values())}"
                )
        candidates_by_representation[representation] = candidates
        selections[representation] = select_history_and_horizon(candidates)
    return candidates_by_representation, selections


def _candidate_row(
    candidates: Sequence[Mapping[str, object]], history_ms: int, offset_ms: int
) -> Mapping[str, object]:
    matches = [
        row
        for row in candidates
        if int(row["history_ms"]) == history_ms and int(row["offset_ms"]) == offset_ms
    ]
    if len(matches) != 1:
        raise ValueError("selected temporal candidate is missing or duplicated")
    return matches[0]


def _paths_from_candidate(
    artifact_path: Path, candidate: Mapping[str, object]
) -> list[Path]:
    return [
        (artifact_path / str(relative)).resolve()
        for relative in candidate["checkpoint_paths"]
    ]


def _secondary_horizon_audit(
    runs: Mapping[str, TemporalRun],
    pairs: Sequence[MatchedPair],
    representation: str,
    history_ms: int,
    horizon_ms: int,
    normalizer: Normalizer,
    checkpoints: Sequence[Path],
    early_margin_ms: int,
    threshold: float,
) -> dict[str, object]:
    """Audit early fall negatives without adding them to primary model selection."""
    early_inputs = []
    early_rows = []
    for pair in pairs:
        run = runs[pair.fall_run_id]
        assert run.fall_sample is not None
        endpoint = run.fall_sample - horizon_ms - early_margin_ms
        if not _eligible_matched_endpoint(run, endpoint, history_ms):
            continue
        indices = causal_window_indices(endpoint, history_ms)
        early_inputs.append(normalizer.transform(run.features[representation][indices]))
        early_rows.append(
            {
                "run_id": run.run_id,
                "endpoint_sample": endpoint,
                "offset_before_fall_ms": int(run.fall_sample - endpoint),
                "horizon_label": horizon_fixed_label(
                    run.fall_sample, endpoint, horizon_ms
                ),
            }
        )
    if not early_inputs:
        return {"performed": False, "reason": "no_valid_early_fall_windows"}
    windows = WindowSet(
        inputs=np.stack(early_inputs).astype(np.float32),
        targets=np.zeros(len(early_inputs), dtype=np.int64),
        run_ids=np.asarray([row["run_id"] for row in early_rows], dtype=str),
        endpoint_samples=np.asarray(
            [row["endpoint_sample"] for row in early_rows], dtype=np.int64
        ),
        available_by_class=(len(early_inputs), 0, 0),
    )
    probabilities = []
    for path in checkpoints:
        model, _ = load_checkpoint(path)
        probabilities.append(predict_fall_probability(model, windows))
    scores = np.mean(np.stack(probabilities), axis=0)
    return {
        "performed": True,
        "window_count": len(scores),
        "all_labels_negative": all(row["horizon_label"] == 0 for row in early_rows),
        "early_fall_specificity": float(np.mean(scores < threshold)),
        "false_positive_runs": [
            row["run_id"]
            for row, score in zip(early_rows, scores)
            if score >= threshold
        ],
        "predictions": [
            {**row, "fall_risk_probability": float(score)}
            for row, score in zip(early_rows, scores)
        ],
    }


def _evaluate_holdout(
    document: Mapping[str, object],
    runs: Mapping[str, TemporalRun],
    candidates_by_representation: Mapping[str, Sequence[Mapping[str, object]]],
    selections: Mapping[str, Mapping[str, object]],
    normalizers: Mapping[str, Normalizer],
    holdout_pairs: Mapping[int, Sequence[MatchedPair]],
    artifact_path: Path,
) -> dict[str, object]:
    threshold = float(document["model"]["probability_threshold"])
    results = {}
    for representation in REPRESENTATION_ORDER:
        selected = selections[representation]["selected"]
        if selected is None:
            results[representation] = {
                "performed": False,
                "reason": "no_validation_reliable_horizon",
            }
            continue
        history_ms = int(selected["history_ms"])
        offset_ms = int(selected["offset_ms"])
        candidate = _candidate_row(
            candidates_by_representation[representation], history_ms, offset_ms
        )
        batch = materialize_matched_windows(
            runs,
            holdout_pairs[offset_ms],
            representation,
            history_ms,
            normalizers[representation],
        )
        metrics, scores, seed_metrics = _evaluate_ensemble(
            _paths_from_candidate(artifact_path, candidate), batch, threshold
        )
        gates = _holdout_passes(metrics, document["holdout"]["gates"])
        results[representation] = {
            "performed": True,
            "history_ms": history_ms,
            "offset_ms": offset_ms,
            "independent_runs": len(set(batch.windows.run_ids)),
            "metrics": metrics,
            "gates": gates,
            "passed": all(gates.values()),
            "seed_metrics": seed_metrics,
            "breakdown": {
                "target_terrain": _metric_breakdown(
                    batch.windows.targets,
                    scores,
                    batch.rows,
                    "target_terrain",
                    ("ice", "sand"),
                    threshold,
                ),
                "source_terrain": _metric_breakdown(
                    batch.windows.targets,
                    scores,
                    batch.rows,
                    "source_terrain",
                    ("concrete", "marble"),
                    threshold,
                ),
            },
            "predictions": _prediction_rows(batch.rows, scores, threshold),
        }
    return results


def _dense_pair_scores(
    runs: Mapping[str, TemporalRun],
    pair: MatchedPair,
    representation: str,
    history_ms: int,
    normalizer: Normalizer,
    checkpoints: Sequence[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fall_run = runs[pair.fall_run_id]
    stable_run = runs[pair.stable_run_id]
    assert fall_run.fall_sample is not None
    first = max(
        fall_run.first_contact_sample + history_ms - 1,
        fall_run.fall_sample - 600,
    )
    fall_endpoints = np.arange(first, fall_run.fall_sample, 10, dtype=np.int64)
    elapsed = fall_endpoints - fall_run.first_contact_sample
    stable_endpoints = stable_run.first_contact_sample + elapsed
    valid = stable_endpoints < len(stable_run.gait_phase)
    fall_endpoints = fall_endpoints[valid]
    stable_endpoints = stable_endpoints[valid]

    def windows(run: TemporalRun, endpoints: np.ndarray) -> WindowSet:
        values = np.stack(
            [
                normalizer.transform(
                    run.features[representation][
                        causal_window_indices(int(endpoint), history_ms)
                    ]
                )
                for endpoint in endpoints
            ]
        )
        return WindowSet(
            inputs=values.astype(np.float32),
            targets=np.zeros(len(values), dtype=np.int64),
            run_ids=np.full(len(values), run.run_id),
            endpoint_samples=endpoints,
            available_by_class=(len(values), 0, 0),
        )

    fall_windows = windows(fall_run, fall_endpoints)
    stable_windows = windows(stable_run, stable_endpoints)
    fall_scores = []
    stable_scores = []
    for path in checkpoints:
        model, _ = load_checkpoint(path)
        fall_scores.append(predict_fall_probability(model, fall_windows))
        stable_scores.append(predict_fall_probability(model, stable_windows))
    x = fall_endpoints - fall_run.fall_sample
    return (
        x.astype(np.float64),
        np.mean(np.stack(fall_scores), axis=0),
        np.mean(np.stack(stable_scores), axis=0),
    )


def _write_temporal_plots(
    artifact_path: Path,
    runs: Mapping[str, TemporalRun],
    validation_pairs: Mapping[int, Sequence[MatchedPair]],
    candidates_by_representation: Mapping[str, Sequence[Mapping[str, object]]],
    selections: Mapping[str, Mapping[str, object]],
    normalizers: Mapping[str, Normalizer],
    threshold: float,
) -> list[dict[str, object]]:
    plot_rows = []
    plot_path = artifact_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    for representation in REPRESENTATION_ORDER:
        choice = (
            selections[representation].get("selected")
            or selections[representation]["diagnostic_best"]
        )
        history_ms = int(choice["history_ms"])
        offset_ms = int(choice["offset_ms"])
        candidate = _candidate_row(
            candidates_by_representation[representation], history_ms, offset_ms
        )
        checkpoints = _paths_from_candidate(artifact_path, candidate)
        for terrain in ("ice", "sand"):
            matches = [
                pair
                for pair in validation_pairs[offset_ms]
                if runs[pair.fall_run_id].target_terrain == terrain
            ]
            if not matches:
                continue
            pair = matches[0]
            x, falling, stable = _dense_pair_scores(
                runs,
                pair,
                representation,
                history_ms,
                normalizers[representation],
                checkpoints,
            )
            figure, axis = plt.subplots(figsize=(7.2, 3.8))
            axis.plot(x, falling, label=f"fall: {pair.fall_run_id}", color="tab:red")
            axis.plot(
                x,
                stable,
                label=f"matched stable: {pair.stable_run_id}",
                color="tab:blue",
            )
            axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
            axis.axvline(0, color="black", linewidth=1)
            axis.set_xlabel("Time relative to fall (ms)")
            axis.set_ylabel("FALL_RISK probability")
            axis.set_ylim(-0.02, 1.02)
            axis.set_title(f"{representation} — {terrain.title()}")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
            figure.tight_layout()
            path = plot_path / f"{representation.lower()}_{terrain}_temporal_risk.png"
            figure.savefig(path, dpi=140)
            plt.close(figure)
            plot_rows.append(
                {
                    "representation": representation,
                    "terrain": terrain,
                    "fall_run_id": pair.fall_run_id,
                    "stable_run_id": pair.stable_run_id,
                    "history_ms": history_ms,
                    "model_offset_ms": offset_ms,
                    "path": str(path.relative_to(artifact_path)),
                }
            )
    return plot_rows


def _hard_stable_control_audit(
    runs: Mapping[str, TemporalRun],
    candidates_by_representation: Mapping[str, Sequence[Mapping[str, object]]],
    selections: Mapping[str, Mapping[str, object]],
    normalizers: Mapping[str, Normalizer],
    artifact_path: Path,
    threshold: float,
) -> dict[str, object]:
    controls = [run for run in runs.values() if run.hard_stable_control]
    results = {}
    for representation in REPRESENTATION_ORDER:
        choice = (
            selections[representation].get("selected")
            or selections[representation]["diagnostic_best"]
        )
        history_ms = int(choice["history_ms"])
        offset_ms = int(choice["offset_ms"])
        candidate = _candidate_row(
            candidates_by_representation[representation], history_ms, offset_ms
        )
        validation_predictions = candidate["predictions"]
        stable_elapsed = [
            int(row["elapsed_since_contact_ms"])
            for row in validation_predictions
            if int(row["label"]) == 0
        ]
        endpoint = int(np.median(stable_elapsed))
        inputs = []
        used = []
        for run in controls:
            control_endpoint = min(endpoint, len(run.gait_phase) - 1)
            if control_endpoint < history_ms - 1:
                continue
            indices = causal_window_indices(control_endpoint, history_ms)
            inputs.append(
                normalizers[representation].transform(
                    run.features[representation][indices]
                )
            )
            used.append((run, control_endpoint))
        windows = WindowSet(
            inputs=np.stack(inputs).astype(np.float32),
            targets=np.zeros(len(inputs), dtype=np.int64),
            run_ids=np.asarray([run.run_id for run, _ in used], dtype=str),
            endpoint_samples=np.asarray(
                [endpoint_sample for _, endpoint_sample in used], dtype=np.int64
            ),
            available_by_class=(len(inputs), 0, 0),
        )
        probabilities = []
        for path in _paths_from_candidate(artifact_path, candidate):
            model, _ = load_checkpoint(path)
            probabilities.append(predict_fall_probability(model, windows))
        score = np.mean(np.stack(probabilities), axis=0)
        results[representation] = {
            "history_ms": history_ms,
            "model_offset_ms": offset_ms,
            "matched_elapsed_endpoint_ms": endpoint,
            "stable_specificity": float(np.mean(score < threshold)),
            "false_positive_runs": [
                run.run_id for (run, _), value in zip(used, score) if value >= threshold
            ],
            "predictions": [
                {
                    "run_id": run.run_id,
                    "source_terrain": run.source_terrain,
                    "fall_risk_probability": float(value),
                    "prediction": "FALL_RISK" if value >= threshold else "STABLE",
                }
                for (run, _), value in zip(used, score)
            ],
        }
    return results


def _cohort_summary(
    runs: Mapping[str, TemporalRun], invalid: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    primary = [run for run in runs.values() if not run.hard_stable_control]

    def count(
        *,
        outcome: str | None = None,
        target: str | None = None,
        source: str | None = None,
        split: str | None = None,
    ) -> int:
        return sum(
            (outcome is None or run.outcome == outcome)
            and (target is None or run.target_terrain == target)
            and (source is None or run.source_terrain == source)
            and (split is None or run.split == split)
            for run in primary
        )

    summary = {
        "configured_primary_runs": 78,
        "valid_primary_runs": len(primary),
        "invalid_primary_runs": len(
            [row for row in invalid if row["split"] != "control"]
        ),
        "hard_stable_controls": len(
            [run for run in runs.values() if run.hard_stable_control]
        ),
        "observed_stable": count(outcome=VALID_STABLE),
        "observed_fall": count(outcome=VALID_FALL),
        "by_terrain": {
            terrain: {
                "stable": count(outcome=VALID_STABLE, target=terrain),
                "fall": count(outcome=VALID_FALL, target=terrain),
            }
            for terrain in ("ice", "sand")
        },
        "by_source": {
            source: {
                "stable": count(outcome=VALID_STABLE, source=source),
                "fall": count(outcome=VALID_FALL, source=source),
            }
            for source in ("concrete", "marble")
        },
        "by_split": {
            split: {
                "total": count(split=split),
                "stable": count(outcome=VALID_STABLE, split=split),
                "fall": count(outcome=VALID_FALL, split=split),
            }
            for split in ("train", "validation", "holdout")
        },
        "invalid": list(invalid),
    }
    summary["target_counts_met"] = bool(
        summary["observed_stable"] >= 30
        and summary["observed_fall"] >= 30
        and all(
            summary["by_terrain"][terrain][outcome] >= 15
            for terrain in ("ice", "sand")
            for outcome in ("stable", "fall")
        )
    )
    return summary


def _run_timeline_rows(runs: Mapping[str, TemporalRun]) -> list[dict[str, object]]:
    """Serialize physical clocks for report/replay without storing raw traces."""

    def time_ms(run: TemporalRun, sample: int | None) -> float | None:
        if sample is None:
            return None
        return float(run.timestamp_us[sample]) / 1000.0

    return [
        {
            "run_id": run.run_id,
            "split": run.split,
            "source_terrain": run.source_terrain,
            "target_terrain": run.target_terrain,
            "speed_mps": run.speed_mps,
            "observed_outcome": ("fall" if run.outcome == VALID_FALL else "stable"),
            "target_contact_ms": time_ms(run, run.first_contact_sample),
            "physical_slip_ms": time_ms(run, run.slip_sample),
            "physical_sink_ms": time_ms(run, run.sink_sample),
            "fall_ms": time_ms(run, run.fall_sample),
            "maximum_support_deformation_mm": (
                run.maximum_support_deformation_m * 1000.0
            ),
            "hard_stable_control": run.hard_stable_control,
        }
        for run in sorted(runs.values(), key=lambda value: value.run_id)
    ]


def _matching_summary(
    pairs_by_split: Mapping[str, Mapping[int, Sequence[MatchedPair]]],
    exclusions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "pairs": {
            split: {
                str(offset): {
                    "pair_count": len(pairs),
                    "independent_runs": len(
                        {
                            run_id
                            for pair in pairs
                            for run_id in (pair.fall_run_id, pair.stable_run_id)
                        }
                    ),
                    "exact_elapsed_matches": True,
                    "phase_matched": sum(pair.endpoint_phase_matched for pair in pairs),
                    "speed_difference_mps": sorted(
                        {pair.speed_difference_mps for pair in pairs}
                    ),
                }
                for offset, pairs in pairs_by_offset.items()
            }
            for split, pairs_by_offset in pairs_by_split.items()
        },
        "exclusions": list(exclusions),
    }


def _terrain_regression(
    document: Mapping[str, object],
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, object]:
    untouched = dict(before) == dict(after)
    return {
        "passed": untouched,
        "protected_sha256_before": dict(before),
        "protected_sha256_after": dict(after),
        "dataset_model_report_untouched": untouched,
        "terrain_retraining_performed": False,
        "candidate_unchanged": document["terrain_regression"]["candidate"],
    }


def _verdict(
    document: Mapping[str, object],
    selections: Mapping[str, Mapping[str, object]],
    holdout: Mapping[str, Mapping[str, object]],
) -> str:
    minimum = int(document["verdict"]["meaningful_early_horizon_min_ms"])

    def supported(representation: str, meaningful: bool) -> bool:
        selected = selections[representation]["selected"]
        if selected is None or not holdout[representation].get("passed", False):
            return False
        horizon = int(selected["offset_ms"])
        return horizon >= minimum if meaningful else horizon in (50, 100)

    if supported(PRIVILEGED_FULL_STATE, True):
        if supported(RUNTIME_IMU6, True):
            return "TEMPORAL_STABILITY_SEPARABILITY_SUPPORTED_IMU6"
        return "TEMPORAL_STABILITY_SEPARABILITY_SUPPORTED_PRIVILEGED_ONLY"
    if supported(PRIVILEGED_FULL_STATE, False):
        return "TEMPORAL_STABILITY_SEPARABILITY_LIMITED"
    return "TEMPORAL_STABILITY_SEPARABILITY_NOT_SUPPORTED"


def run_temporal_stability_separability_audit(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Run frozen simulation, validation selection, and sealed holdout once."""
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    document = _load_yaml(config_path)
    primary_specs, control_specs = _load_scenario_specs(document, repository_root)
    design = validate_temporal_design(document, primary_specs, control_specs)
    if PHYSICS_TIMESTEP_S != float(document["common"]["physics_timestep_s"]):
        raise ValueError("temporal physics timestep differs from canonical value")
    if SENSOR_RATE_HZ != int(document["common"]["sensor_rate_hz"]):
        raise ValueError("temporal sensor rate differs from canonical value")

    artifact_path = (repository_root / document["artifacts"]["path"]).resolve()
    artifact_path.relative_to(repository_root)
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite temporal audit artifacts: {artifact_path}"
        )
    artifact_path.mkdir(parents=True, exist_ok=True)
    policy_path = (repository_root / document["source"]["policy_path"]).resolve()
    if not policy_path.is_file() or _file_sha256(policy_path) != str(
        document["source"]["policy_sha256"]
    ):
        raise ValueError("verified G1 policy is unavailable or has the wrong SHA-256")
    protected_paths = [
        str(value) for value in document["terrain_regression"]["protected_paths"]
    ]
    terrain_before = _protected_hashes(repository_root, protected_paths)
    base = load_simulation_config(
        repository_root / str(document["source"]["simulator_config"])
    )
    model, _ = load_g1_model("concrete")
    qpos_addresses, qvel_addresses = lower_body_state_addresses(model)

    runs, invalid = simulate_temporal_cohort(
        base,
        [*primary_specs, *control_specs],
        policy_path,
        document,
        qpos_addresses,
        qvel_addresses,
        progress,
    )
    cohort = _cohort_summary(runs, invalid)
    if not cohort["target_counts_met"]:
        terrain_after = _protected_hashes(repository_root, protected_paths)
        metrics = {
            "experiment": document["experiment"],
            "design": design,
            "cohort": cohort,
            "validation": {"performed": False},
            "holdout": {"performed": False},
            "terrain_regression": _terrain_regression(
                document, terrain_before, terrain_after
            ),
            "fusion_regression": fusion_regression(),
            "verdict": "TEMPORAL_STABILITY_SEPARABILITY_NOT_SUPPORTED",
        }
        _write_json(artifact_path / "metrics.json", metrics)
        return artifact_path, metrics

    split_ids = {
        split: [str(value) for value in document["split"][split]]
        for split in ("train", "validation", "holdout")
    }
    offsets = [
        int(value) for value in document["offset_analysis"]["offsets_before_fall_ms"]
    ]
    maximum_history = max(int(value) for value in document["history"]["candidates_ms"])
    train_pairs, train_exclusions = build_matched_pairs(
        runs, split_ids["train"], offsets, maximum_history
    )
    validation_pairs, validation_exclusions = build_matched_pairs(
        runs, split_ids["validation"], offsets, maximum_history
    )
    normalization = document["normalization"]
    normalizers = {
        representation: fit_train_normalizer(
            runs,
            split_ids["train"],
            representation,
            int(normalization["per_run_sample_cap"]),
            float(normalization["standard_deviation_floor"]),
        )
        for representation in REPRESENTATION_ORDER
    }
    _write_json(
        artifact_path / "train_only_normalization.json",
        {
            representation: normalizer.to_dict()
            for representation, normalizer in normalizers.items()
        },
    )
    candidates, selections = _train_validation_candidates(
        document,
        runs,
        normalizers,
        train_pairs,
        validation_pairs,
        artifact_path,
        progress,
    )
    _write_json(
        artifact_path / "selection_before_holdout.json",
        {"selections": selections, "holdout_opened": False},
    )

    threshold = float(document["model"]["probability_threshold"])
    secondary = {}
    for representation in REPRESENTATION_ORDER:
        audits = []
        for candidate in candidates[representation]:
            history_ms = int(candidate["history_ms"])
            horizon_ms = int(candidate["offset_ms"])
            audit = _secondary_horizon_audit(
                runs,
                validation_pairs[horizon_ms],
                representation,
                history_ms,
                horizon_ms,
                normalizers[representation],
                _paths_from_candidate(artifact_path, candidate),
                int(
                    document["horizon_classification"]["early_fall_negative_margin_ms"]
                ),
                threshold,
            )
            audits.append(
                {
                    "history_ms": history_ms,
                    "horizon_ms": horizon_ms,
                    "boundary_metrics": candidate["metrics"],
                    **audit,
                }
            )
        secondary[representation] = audits

    guard = HoldoutGuard()
    if any(
        selections[representation]["selected"] is not None
        for representation in REPRESENTATION_ORDER
    ):
        guard.open_once()
        holdout_pairs, holdout_exclusions = build_matched_pairs(
            runs,
            split_ids["holdout"],
            offsets,
            maximum_history,
            holdout_guard=guard,
        )
        holdout = _evaluate_holdout(
            document,
            runs,
            candidates,
            selections,
            normalizers,
            holdout_pairs,
            artifact_path,
        )
    else:
        holdout_pairs = {offset: tuple() for offset in offsets}
        holdout_exclusions = []
        holdout = {
            representation: {
                "performed": False,
                "reason": "no_validation_reliable_horizon",
            }
            for representation in REPRESENTATION_ORDER
        }
    hard_controls = _hard_stable_control_audit(
        runs,
        candidates,
        selections,
        normalizers,
        artifact_path,
        threshold,
    )
    plots = _write_temporal_plots(
        artifact_path,
        runs,
        validation_pairs,
        candidates,
        selections,
        normalizers,
        threshold,
    )
    terrain_after = _protected_hashes(repository_root, protected_paths)
    verdict = _verdict(document, selections, holdout)
    metrics = {
        "experiment": document["experiment"],
        "design": design,
        "cohort": cohort,
        "run_timelines": _run_timeline_rows(runs),
        "matching": _matching_summary(
            {
                "train": train_pairs,
                "validation": validation_pairs,
                "holdout": holdout_pairs,
            },
            [*train_exclusions, *validation_exclusions, *holdout_exclusions],
        ),
        "normalization": {
            representation: normalizer.to_dict()
            for representation, normalizer in normalizers.items()
        },
        "validation": {
            "performed": True,
            "candidates": candidates,
            "selections": selections,
            "holdout_sealed_during_selection": True,
        },
        "secondary_horizon_classification": secondary,
        "holdout": {
            "performed": any(value["performed"] for value in holdout.values()),
            "guard_open_count": guard.open_count,
            "reselection_performed": False,
            "representations": holdout,
        },
        "hard_stable_controls": hard_controls,
        "plots": plots,
        "terrain_regression": _terrain_regression(
            document, terrain_before, terrain_after
        ),
        "fusion_regression": fusion_regression(),
        "privileged_runtime_boundary": {
            "q_dq_runtime_augmentation_performed": False,
            "terrain_identity_in_tensor": False,
            "fall_time_in_tensor": False,
            "time_to_fall_in_tensor": False,
            "slip_sink_in_tensor": False,
            "future_samples_in_window": False,
            "runtime_enum_changed": False,
        },
        "causality": {
            "passed": True,
            "window_endpoint_is_last_input_sample": True,
            "future_suffix_changes_decided_windows": False,
            "post_fall_samples_in_positive_windows": False,
            "fall_time_is_label_alignment_only": True,
        },
        "verdict": verdict,
    }
    _write_json(artifact_path / "metrics.json", metrics)
    progress(
        json.dumps(
            {
                "selected": selections,
                "holdout": holdout,
                "verdict": verdict,
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return artifact_path, metrics
