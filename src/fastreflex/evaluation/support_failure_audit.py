"""Read-only failure-mode audit for the frozen Support detector.

This module deliberately accepts only TRAIN and VALIDATION waveforms.  It
reuses the frozen Support normalizer, checkpoints, decision threshold,
persistence, and cached frozen-Terrain development traces without training or
selection.  HOLDOUT is not a valid argument or data path in this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fastreflex.dataset.loader import Normalizer
from fastreflex.evaluation.reflex_event import (
    EventHoldoutGuard,
    EventRun,
    _load_yaml,
    _write_json,
    load_event_runs,
)
from fastreflex.evaluation.stability_temporal import (
    _file_sha256,
    binary_auroc,
)
from fastreflex.evaluation.transition_scenarios import VALID_FALL, VALID_STABLE
from fastreflex.evaluation.terrain_conditioned_reflex import (
    BRANCH_STATE,
    SAND,
    TERRAIN_STATE_NAMES,
    BranchReplay,
    TerrainGateTrace,
    _canonical_sha256,
    _load_gate,
    _replay_many,
    branch_event_sample,
    evaluate_branch_replays,
    extract_branch_features,
    feature_schema_for_components,
    sustained_alert_trace,
)


AUDIT_ID = "SUPPORT_FAILURE_MODE_AUDIT"
VERDICTS = (
    "SUPPORT_FAILURE_MODE_IDENTIFIED",
    "SUPPORT_MULTIPLE_FAILURE_MODES_IDENTIFIED",
    "SUPPORT_FAILURE_MODE_INCONCLUSIVE",
)
ALLOWED_SPLITS = ("train", "validation")
VALID_LATENCY_MS = (-30, 50)
EXPECTED_SLIP_FREEZE_SHA256 = (
    "df0a232ec242283ef8b25c59421cebde982a7a93febb655cc511fa2fa3de3229"
)


def validate_audit_splits(splits: Sequence[str]) -> tuple[str, ...]:
    """Fail closed if any split other than TRAIN/VALIDATION is requested."""
    result = tuple(str(value).lower() for value in splits)
    if not result or len(set(result)) != len(result):
        raise ValueError("audit splits must be unique and nonempty")
    forbidden = set(result) - set(ALLOWED_SPLITS)
    if forbidden:
        raise RuntimeError(
            "Support failure audit forbids non-development split access: "
            + ", ".join(sorted(forbidden))
        )
    return result


def load_frozen_normalizer(path: Path) -> tuple[Normalizer, Mapping[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    normalizer = Normalizer(
        mean=np.asarray(document["mean"], dtype=np.float32),
        std=np.asarray(document["std"], dtype=np.float32),
        sample_count=int(document["sample_count"]),
        fit_run_ids=tuple(str(value) for value in document["fit_run_ids"]),
        epsilon=float(document["epsilon"]),
    )
    if normalizer.mean.shape != (60,) or normalizer.std.shape != (60,):
        raise ValueError("frozen Support normalizer must be 60-dimensional")
    if np.any(normalizer.std <= 0.0):
        raise ValueError("frozen Support normalizer standard deviation is invalid")
    return normalizer, document


def load_development_gates(
    gate_path: Path, runs: Mapping[str, EventRun]
) -> dict[str, TerrainGateTrace]:
    """Load only cached development traces selected by loaded development IDs."""
    if gate_path.name != "development":
        raise RuntimeError("audit may only use the development Terrain gate cache")
    traces: dict[str, TerrainGateTrace] = {}
    for run_id, run in sorted(runs.items()):
        if run.split not in ALLOWED_SPLITS:
            raise RuntimeError("non-development run reached Terrain gate loading")
        path = gate_path / f"{run_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        traces[run_id] = _load_gate(path, len(run.timestamp_us))
    return traces


def _first(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if not len(indices) else int(indices[0])


def _maximum_consecutive(values: np.ndarray) -> int:
    selected = np.asarray(values, dtype=bool)
    maximum = count = 0
    for value in selected:
        count = count + 1 if bool(value) else 0
        maximum = max(maximum, count)
    return maximum


def _percentiles(values: Sequence[float | int | None]) -> dict[str, float | None]:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(array):
        return {
            key: None for key in ("minimum", "p10", "p25", "median", "p75", "p90", "p95", "maximum")
        }
    functions = {
        "minimum": np.min,
        "p10": lambda x: np.percentile(x, 10),
        "p25": lambda x: np.percentile(x, 25),
        "median": np.median,
        "p75": lambda x: np.percentile(x, 75),
        "p90": lambda x: np.percentile(x, 90),
        "p95": lambda x: np.percentile(x, 95),
        "maximum": np.max,
    }
    return {key: float(function(array)) for key, function in functions.items()}


def _sample_time_ms(run: EventRun, sample: int | None) -> float | None:
    if sample is None or sample < 0 or sample >= len(run.timestamp_us):
        return None
    return float(run.timestamp_us[sample]) / 1000.0


def _side_label(run: EventRun) -> str:
    left, right = run.support_event_samples_per_foot
    if left is not None and right is not None:
        return "bilateral"
    if left is not None:
        return "left_only"
    if right is not None:
        return "right_only"
    return "none"


def _normalizer_z(normalizer: Normalizer, features: np.ndarray) -> np.ndarray:
    return normalizer.transform(features).astype(np.float64, copy=False)


def event_diagnostic_row(
    run: EventRun,
    gate: TerrainGateTrace,
    replay: BranchReplay,
    normalizer: Normalizer,
    feature_schema: Sequence[str],
    *,
    threshold: float,
    persistence_ms: int,
) -> dict[str, object]:
    """Create one frozen-decision diagnostic row for a canonical Support event."""
    event = branch_event_sample(run, "support")
    if event is None or run.target_terrain != "sand" or run.hard_stable_control:
        raise ValueError("event diagnostic requires a primary Sand Support event")
    lower, upper = VALID_LATENCY_MS
    endpoints = np.asarray(replay.endpoints, dtype=np.int64)
    probability = np.asarray(replay.probabilities, dtype=np.float64)
    active = np.asarray(replay.terrain_state == SAND, dtype=bool)
    raw_above = probability >= threshold
    gated_above = raw_above & active
    alert, onset = sustained_alert_trace(
        probability, active, threshold, persistence_ms
    )
    _, raw_onset = sustained_alert_trace(
        probability, np.ones(len(probability), dtype=bool), threshold, persistence_ms
    )
    valid = (endpoints >= event + lower) & (endpoints <= event + upper)
    negative = endpoints < event + lower
    valid_onsets = onset & valid
    late_onsets = onset & (endpoints > event + upper)
    premature_onsets = onset & negative
    raw_valid_onsets = raw_onset & valid
    raw_late_onsets = raw_onset & (endpoints > event + upper)

    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        raise ValueError("Support event has no canonical response samples")
    valid_probability = probability[valid_indices]
    local_peak = int(valid_indices[int(np.argmax(valid_probability))])
    peak_sample = int(endpoints[local_peak])
    active_valid = active & valid
    gated_valid_scores = probability[active_valid]
    first_valid_index = _first(valid_onsets)
    first_raw_cross = _first(raw_above & valid)
    first_gated_cross = _first(gated_above & valid)
    first_late_index = _first(late_onsets)
    first_premature_index = _first(premature_onsets)
    first_raw_valid_index = _first(raw_valid_onsets)
    first_raw_late_index = _first(raw_late_onsets)

    features, names = extract_branch_features(run, ("pelvis_imu6",))
    if tuple(names) != tuple(feature_schema):
        raise RuntimeError("frozen Support feature order changed")
    feature_z = _normalizer_z(normalizer, features)
    feature_valid = feature_z[endpoints[valid]]
    raw_z = feature_valid[:, :10]
    row_energy = np.linalg.norm(raw_z, axis=1)
    flat_index = int(np.argmax(np.abs(feature_valid)))
    feature_row, feature_column = np.unravel_index(flat_index, feature_valid.shape)
    imu = np.asarray(run.features["PELVIS_IMU6"], dtype=np.float64)
    valid_imu = imu[endpoints[valid]]
    accel_norm = np.linalg.norm(valid_imu[:, :3], axis=1)
    gyro_norm = np.linalg.norm(valid_imu[:, 3:], axis=1)
    baseline = (endpoints >= event - 130) & (endpoints < event - 30)
    baseline_imu = imu[endpoints[baseline]] if np.any(baseline) else valid_imu[:1]
    baseline_center = np.median(baseline_imu, axis=0)
    imu_excursion = np.linalg.norm(valid_imu - baseline_center, axis=1)
    score_differences = np.diff(valid_probability)

    first_valid_sample = (
        None if first_valid_index is None else int(endpoints[first_valid_index])
    )
    first_raw_sample = (
        None if first_raw_cross is None else int(endpoints[first_raw_cross])
    )
    first_gated_sample = (
        None if first_gated_cross is None else int(endpoints[first_gated_cross])
    )
    first_late_sample = (
        None if first_late_index is None else int(endpoints[first_late_index])
    )
    first_premature_sample = (
        None
        if first_premature_index is None
        else int(endpoints[first_premature_index])
    )
    first_raw_valid_sample = (
        None
        if first_raw_valid_index is None
        else int(endpoints[first_raw_valid_index])
    )
    first_raw_late_sample = (
        None if first_raw_late_index is None else int(endpoints[first_raw_late_index])
    )
    terrain_valid = gate.first_target_valid_sample
    gating_at_event_indices = np.flatnonzero(endpoints == event)
    gating_at_event = (
        TERRAIN_STATE_NAMES[int(replay.terrain_state[gating_at_event_indices[0]])]
        if len(gating_at_event_indices)
        else "UNAVAILABLE"
    )
    event_max_score = float(probability[local_peak])
    active_max_score = (
        None if not len(gated_valid_scores) else float(np.max(gated_valid_scores))
    )
    pre_event = (endpoints >= event - 100) & (endpoints <= event)
    gate_changes = np.flatnonzero(
        np.r_[False, replay.terrain_state[1:] != replay.terrain_state[:-1]]
        & (endpoints <= event)
    )
    last_gate_change = (
        None if not len(gate_changes) else int(endpoints[gate_changes[-1]])
    )
    contiguous_sand = 0
    event_index = np.flatnonzero(endpoints <= event)
    for index in event_index[::-1]:
        if replay.terrain_state[index] != SAND:
            break
        contiguous_sand += 1
    result: dict[str, object] = {
        "run_id": run.run_id,
        "split": run.split,
        "event_id": f"{run.run_id}:support",
        "canonical_event_label": run.event_type,
        "outcome": run.outcome_diagnostic,
        "recovered_or_fall": (
            "recovered"
            if run.outcome_diagnostic == VALID_STABLE
            else "fall"
            if run.outcome_diagnostic == VALID_FALL
            else "unknown"
        ),
        "source_ground": run.source_terrain,
        "target_terrain": run.target_terrain,
        "design_role_diagnostic_only": run.design_role,
        "sink_pattern": run.sink_pattern,
        "support_pattern": run.support_pattern,
        "support_side": _side_label(run),
        "detected": first_valid_sample is not None,
        "missed": first_valid_sample is None,
        "target_event_sample": event,
        "target_event_time_ms": _sample_time_ms(run, event),
        "detector_first_valid_output_sample": first_valid_sample,
        "detector_first_valid_output_time_ms": _sample_time_ms(
            run, first_valid_sample
        ),
        "latency_ms": (
            None if first_valid_sample is None else first_valid_sample - event
        ),
        "detector_max_score": event_max_score,
        "active_gate_max_score": active_max_score,
        "threshold": float(threshold),
        "max_score_minus_threshold": event_max_score - threshold,
        "first_raw_threshold_crossing_sample": first_raw_sample,
        "first_gated_threshold_crossing_sample": first_gated_sample,
        "first_threshold_crossing_time_ms": _sample_time_ms(
            run, first_gated_sample
        ),
        "valid_window_threshold_above_duration_ms": int(
            np.count_nonzero(gated_above & valid)
        ),
        "run_threshold_above_duration_ms": int(np.count_nonzero(gated_above)),
        "maximum_consecutive_threshold_above_ms": _maximum_consecutive(
            gated_above & valid
        ),
        "run_maximum_consecutive_threshold_above_ms": _maximum_consecutive(
            gated_above
        ),
        "raw_maximum_consecutive_threshold_above_ms": _maximum_consecutive(
            raw_above & valid
        ),
        "raw_persistence_satisfied": _maximum_consecutive(raw_above & valid)
        >= persistence_ms,
        "raw_first_valid_output_sample": first_raw_valid_sample,
        "raw_first_late_output_sample": first_raw_late_sample,
        "persistence_requirement_ms": int(persistence_ms),
        "persistence_satisfied": _maximum_consecutive(gated_above & valid)
        >= persistence_ms,
        "any_premature_alert": first_premature_sample is not None,
        "first_premature_alert_sample": first_premature_sample,
        "first_late_output_sample": first_late_sample,
        "terrain_valid_sample": terrain_valid,
        "terrain_valid_time_ms": _sample_time_ms(run, terrain_valid),
        "support_target_sample": event,
        "terrain_to_support_margin_ms": (
            None if terrain_valid is None else event - terrain_valid
        ),
        "touchdown_sample": run.first_touchdown_sample,
        "touchdown_time_ms": _sample_time_ms(run, run.first_touchdown_sample),
        "touchdown_to_support_margin_ms": event - run.first_touchdown_sample,
        "gating_state_at_support": gating_at_event,
        "gating_state_at_peak": TERRAIN_STATE_NAMES[
            int(replay.terrain_state[local_peak])
        ],
        "sand_gate_fraction_last_100ms": (
            0.0 if not np.any(pre_event) else float(np.mean(active[pre_event]))
        ),
        "sand_gate_contiguous_before_support_ms": contiguous_sand,
        "last_gate_change_before_support_sample": last_gate_change,
        "last_gate_change_to_support_ms": (
            None if last_gate_change is None else event - last_gate_change
        ),
        "active_gate_samples_in_valid_window": int(np.count_nonzero(active_valid)),
        "score_peak_relative_to_event_ms": peak_sample - event,
        "score_max_rise_per_ms": (
            0.0 if not len(score_differences) else float(np.max(score_differences))
        ),
        "raw_feature_z_energy_p90": float(np.percentile(row_energy, 90)),
        "raw_feature_z_energy_max": float(np.max(row_energy)),
        "all_feature_abs_z_p90": float(np.percentile(np.abs(feature_valid), 90)),
        "all_feature_abs_z_max": float(np.max(np.abs(feature_valid))),
        "largest_abs_z_feature": str(feature_schema[feature_column]),
        "largest_abs_z_feature_value": float(feature_valid[feature_row, feature_column]),
        "pelvis_accel_norm_p90": float(np.percentile(accel_norm, 90)),
        "pelvis_accel_norm_max": float(np.max(accel_norm)),
        "pelvis_gyro_norm_p90": float(np.percentile(gyro_norm, 90)),
        "pelvis_gyro_norm_max": float(np.max(gyro_norm)),
        "pelvis_imu_excursion_p90": float(np.percentile(imu_excursion, 90)),
        "pelvis_imu_excursion_max": float(np.max(imu_excursion)),
        "diagnostic_group": "unassigned",
        "failure_modes": "",
        "primary_failure_mode": "",
    }
    del features, feature_z
    return result


def assign_diagnostic_groups(rows: list[dict[str, object]]) -> dict[str, object]:
    detected = [row for row in rows if bool(row["detected"])]
    margins = np.asarray(
        [float(row["max_score_minus_threshold"]) for row in detected], dtype=float
    )
    cutoff = None if not len(margins) else float(np.percentile(margins, 25))
    for row in rows:
        if bool(row["missed"]):
            row["diagnostic_group"] = "miss"
        elif cutoff is not None and float(row["max_score_minus_threshold"]) <= cutoff:
            row["diagnostic_group"] = "low_margin_success"
        else:
            row["diagnostic_group"] = "clear_success"
    return {
        "low_margin_success_definition": "detected event score margin <= detected q25",
        "low_margin_success_cutoff": cutoff,
        "group_counts": {
            group: sum(row["diagnostic_group"] == group for row in rows)
            for group in ("clear_success", "low_margin_success", "miss")
        },
    }


def assign_failure_modes(rows: list[dict[str, object]]) -> dict[str, object]:
    clear = [row for row in rows if row["diagnostic_group"] == "clear_success"]
    signal_reference = _percentiles(
        [float(row["raw_feature_z_energy_p90"]) for row in clear]
    )
    feature_reference = _percentiles(
        [float(row["all_feature_abs_z_p90"]) for row in clear]
    )
    timing_reference = _percentiles(
        [row["terrain_to_support_margin_ms"] for row in clear]
    )
    counts: dict[str, int] = {}
    primary_counts: dict[str, int] = {}
    for row in rows:
        if not bool(row["missed"]):
            continue
        modes: list[str] = []
        raw_valid = bool(row["raw_persistence_satisfied"])
        if raw_valid and not bool(row["persistence_satisfied"]):
            modes.append("GATING_SUPPRESSION")
        if float(row["detector_max_score"]) < float(row["threshold"]):
            modes.append("SCORE_INSUFFICIENT")
        elif 0 < int(row["raw_maximum_consecutive_threshold_above_ms"]) < int(
            row["persistence_requirement_ms"]
        ):
            modes.append("PERSISTENCE_FAILURE")
        if not raw_valid and row["raw_first_late_output_sample"] is not None:
            modes.append("LATE_RESPONSE")
        signal_p10 = signal_reference["p10"]
        if (
            "SCORE_INSUFFICIENT" in modes
            and signal_p10 is not None
            and float(row["raw_feature_z_energy_p90"]) < signal_p10
        ):
            modes.append("SIGNAL_ABSENT")
        feature_p10, feature_p90 = (
            feature_reference["p10"],
            feature_reference["p90"],
        )
        value = float(row["all_feature_abs_z_p90"])
        if (
            "SCORE_INSUFFICIENT" in modes
            and
            feature_p10 is not None
            and feature_p90 is not None
            and (value < feature_p10 or value > feature_p90)
        ):
            modes.append("FEATURE_DISTRIBUTION_SHIFT")
        margin = row["terrain_to_support_margin_ms"]
        timing_p10, timing_p90 = timing_reference["p10"], timing_reference["p90"]
        if (
            row["gating_state_at_support"] != "SAND"
            or margin is None
            or (
                timing_p10 is not None
                and timing_p90 is not None
                and (float(margin) < timing_p10 or float(margin) > timing_p90)
            )
        ):
            modes.append("TRANSITION_TIMING")
        if not modes:
            modes.append("OTHER")
        unique = list(dict.fromkeys(modes))
        row["failure_modes"] = ";".join(unique)
        primary = next(
            (
                value
                for value in (
                    "GATING_SUPPRESSION",
                    "PERSISTENCE_FAILURE",
                    "SCORE_INSUFFICIENT",
                    "LATE_RESPONSE",
                    "SIGNAL_ABSENT",
                    "FEATURE_DISTRIBUTION_SHIFT",
                    "TRANSITION_TIMING",
                    "OTHER",
                )
                if value in unique
            ),
            "OTHER",
        )
        row["primary_failure_mode"] = primary
        primary_counts[primary] = primary_counts.get(primary, 0) + 1
        for mode in unique:
            counts[mode] = counts.get(mode, 0) + 1
    return {
        "classification_is_diagnostic_only": True,
        "clear_success_signal_reference": signal_reference,
        "clear_success_feature_reference": feature_reference,
        "clear_success_timing_reference": timing_reference,
        "miss_failure_mode_counts": dict(sorted(counts.items())),
        "miss_primary_failure_mode_counts": dict(sorted(primary_counts.items())),
    }


def _recall_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    detected = sum(bool(row["detected"]) for row in rows)
    return {
        "events": len(rows),
        "detected": detected,
        "missed": len(rows) - detected,
        "recall": None if not rows else detected / len(rows),
        "premature_events": sum(bool(row["any_premature_alert"]) for row in rows),
        "latency_ms": _percentiles([row["latency_ms"] for row in rows]),
    }


def subgroup_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    dimensions = {
        "outcome": "recovered_or_fall",
        "source_ground": "source_ground",
        "event_type": "canonical_event_label",
        "support_side": "support_side",
        "design_role": "design_role_diagnostic_only",
        "sink_pattern": "sink_pattern",
        "support_pattern": "support_pattern",
    }
    result: dict[str, object] = {}
    for label, key in dimensions.items():
        values = sorted({str(row[key]) for row in rows})
        result[label] = {
            value: _recall_rows([row for row in rows if str(row[key]) == value])
            for value in values
        }
    return result


def group_distributions(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    fields = (
        "detector_max_score",
        "max_score_minus_threshold",
        "score_max_rise_per_ms",
        "maximum_consecutive_threshold_above_ms",
        "score_peak_relative_to_event_ms",
        "terrain_to_support_margin_ms",
        "touchdown_to_support_margin_ms",
        "raw_feature_z_energy_p90",
        "all_feature_abs_z_p90",
        "pelvis_accel_norm_p90",
        "pelvis_gyro_norm_p90",
        "pelvis_imu_excursion_p90",
        "sand_gate_fraction_last_100ms",
        "sand_gate_contiguous_before_support_ms",
        "last_gate_change_to_support_ms",
    )
    result: dict[str, object] = {}
    for group in ("clear_success", "low_margin_success", "miss"):
        selected = [row for row in rows if row["diagnostic_group"] == group]
        result[group] = {
            "events": len(selected),
            "gating_state_at_support": {
                state: sum(row["gating_state_at_support"] == state for row in selected)
                for state in TERRAIN_STATE_NAMES
                if any(row["gating_state_at_support"] == state for row in selected)
            },
            "raw_persistence_satisfied": sum(
                bool(row["raw_persistence_satisfied"]) for row in selected
            ),
            "gated_persistence_satisfied": sum(
                bool(row["persistence_satisfied"]) for row in selected
            ),
            "distributions": {
                field: _percentiles([row[field] for row in selected])
                for field in fields
            },
        }
    return result


def outcome_comparison(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    fields = (
        "max_score_minus_threshold",
        "score_peak_relative_to_event_ms",
        "maximum_consecutive_threshold_above_ms",
        "terrain_to_support_margin_ms",
        "touchdown_to_support_margin_ms",
        "raw_feature_z_energy_p90",
        "pelvis_imu_excursion_p90",
    )
    for outcome in ("recovered", "fall"):
        selected = [row for row in rows if row["recovered_or_fall"] == outcome]
        result[outcome] = {
            **_recall_rows(selected),
            "distributions": {
                field: _percentiles([row[field] for row in selected])
                for field in fields
            },
            "persistence_failures": sum(
                "PERSISTENCE_FAILURE" in str(row["failure_modes"])
                for row in selected
            ),
        }
    return result


def timing_comparison(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    margins = np.asarray(
        [float(row["terrain_to_support_margin_ms"]) for row in rows if row["terrain_to_support_margin_ms"] is not None],
        dtype=float,
    )
    if not len(margins):
        return {"available": False}
    q25, q75 = float(np.percentile(margins, 25)), float(np.percentile(margins, 75))
    groups = {
        "very_short": [
            row
            for row in rows
            if row["terrain_to_support_margin_ms"] is not None
            and float(row["terrain_to_support_margin_ms"]) <= q25
        ],
        "middle": [
            row
            for row in rows
            if row["terrain_to_support_margin_ms"] is not None
            and q25 < float(row["terrain_to_support_margin_ms"]) < q75
        ],
        "long": [
            row
            for row in rows
            if row["terrain_to_support_margin_ms"] is not None
            and float(row["terrain_to_support_margin_ms"]) >= q75
        ],
    }
    return {
        "available": True,
        "diagnostic_bin_definition": {"very_short_max_ms": q25, "long_min_ms": q75},
        "margin_distribution_ms": _percentiles(margins.tolist()),
        "groups": {name: _recall_rows(selected) for name, selected in groups.items()},
    }


def negative_metrics(
    runs: Mapping[str, EventRun],
    replays: Mapping[str, BranchReplay],
    *,
    threshold: float,
    persistence_ms: int,
) -> tuple[dict[str, object], list[float]]:
    rows = []
    active_negative_samples = alert_samples = 0
    negative_max_scores: list[float] = []
    for run_id, run in sorted(runs.items()):
        replay = replays[run_id]
        active = replay.terrain_state == BRANCH_STATE["support"]
        alert, onset = sustained_alert_trace(
            replay.probabilities, active, threshold, persistence_ms
        )
        event = branch_event_sample(run, "support")
        if event is None:
            negative_mask = np.ones(len(replay.endpoints), dtype=bool)
            active_mask = active & negative_mask
            if np.any(active_mask):
                negative_max_scores.append(float(np.max(replay.probabilities[active_mask])))
            rows.append(
                {
                    "run_id": run_id,
                    "split": run.split,
                    "source_ground": run.source_terrain,
                    "target_terrain": run.target_terrain,
                    "hard_stable_control": run.hard_stable_control,
                    "system_false_reflex": bool(np.any(alert)),
                    "first_false_reflex_sample": (
                        None
                        if not np.any(onset)
                        else int(replay.endpoints[np.flatnonzero(onset)[0]])
                    ),
                }
            )
        else:
            negative_mask = replay.endpoints < event + VALID_LATENCY_MS[0]
        active_negative = active & negative_mask
        active_negative_samples += int(np.count_nonzero(active_negative))
        alert_samples += int(np.count_nonzero(alert & active_negative))
    false_reflexes = sum(bool(row["system_false_reflex"]) for row in rows)
    benign = [row for row in rows if row["target_terrain"] == "sand" and not row["hard_stable_control"]]
    hard = [row for row in rows if bool(row["hard_stable_control"])]
    return (
        {
            "negative_runs": len(rows),
            "false_positive_count": false_reflexes,
            "specificity": 1.0 if not rows else 1.0 - false_reflexes / len(rows),
            "negative_time_alert_fraction": (
                0.0 if not active_negative_samples else alert_samples / active_negative_samples
            ),
            "active_negative_samples": active_negative_samples,
            "active_negative_alert_samples": alert_samples,
            "false_reflex_count": false_reflexes,
            "sand_benign": {
                "runs": len(benign),
                "false_reflexes": sum(bool(row["system_false_reflex"]) for row in benign),
                "specificity": 1.0 if not benign else 1.0 - sum(bool(row["system_false_reflex"]) for row in benign) / len(benign),
            },
            "hard_ground": {
                "runs": len(hard),
                "false_reflexes": sum(bool(row["system_false_reflex"]) for row in hard),
                "specificity": 1.0 if not hard else 1.0 - sum(bool(row["system_false_reflex"]) for row in hard) / len(hard),
            },
            "rows": rows,
        },
        negative_max_scores,
    )


def observability_assessment(
    event_rows: Sequence[Mapping[str, object]], negative_max_scores: Sequence[float]
) -> dict[str, object]:
    event_scores = [float(row["detector_max_score"]) for row in event_rows]
    labels = np.asarray([1] * len(event_scores) + [0] * len(negative_max_scores), dtype=np.int64)
    scores = np.asarray([*event_scores, *negative_max_scores], dtype=np.float64)
    auroc = None if len(set(labels.tolist())) < 2 else float(binary_auroc(labels, scores))
    misses = [row for row in event_rows if bool(row["missed"])]
    partial_modes = sum(
        "SIGNAL_ABSENT" in str(row["failure_modes"])
        or "FEATURE_DISTRIBUTION_SHIFT" in str(row["failure_modes"])
        for row in misses
    )
    if auroc is None:
        judgment = "SUPPORT_SIGNAL_PARTIALLY_OBSERVABLE"
        reason = "negative active-gate score reference was unavailable"
    elif auroc >= 0.90 and partial_modes == 0:
        judgment = "SUPPORT_SIGNAL_PRESENT_BUT_DECISION_WEAK"
        reason = "event peak scores separate from active-gate negative peaks; misses are decision/timing failures"
    elif auroc >= 0.75:
        judgment = "SUPPORT_SIGNAL_PARTIALLY_OBSERVABLE"
        reason = "event and active-gate negative peak scores retain only partial separation"
    else:
        judgment = "SUPPORT_CURRENT_SENSOR_OBSERVABILITY_LIMITED"
        reason = "event and active-gate negative peak scores overlap strongly"
    return {
        "judgment": judgment,
        "reason": reason,
        "event_vs_negative_run_peak_auroc": auroc,
        "event_peak_score": _percentiles(event_scores),
        "active_gate_negative_run_peak_score": _percentiles(negative_max_scores),
        "misses_with_signal_absent_or_distribution_shift": partial_modes,
    }


def write_event_diagnostics(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("event diagnostic table cannot be empty")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_trace(
    path: Path,
    run: EventRun,
    gate: TerrainGateTrace,
    replay: BranchReplay,
    row: Mapping[str, object],
    normalizer: Normalizer,
    feature_schema: Sequence[str],
    *,
    threshold: float,
    persistence_ms: int,
) -> None:
    import matplotlib.pyplot as plt

    event = int(row["target_event_sample"])
    endpoints = np.asarray(replay.endpoints, dtype=np.int64)
    selected = (endpoints >= event - 180) & (endpoints <= event + 180)
    x = endpoints[selected] - event
    probability = replay.probabilities[selected]
    active = replay.terrain_state[selected] == SAND
    alert, _ = sustained_alert_trace(probability, active, threshold, persistence_ms)
    imu = np.asarray(run.features["PELVIS_IMU6"], dtype=np.float64)[endpoints[selected]]
    accel = np.linalg.norm(imu[:, :3], axis=1)
    gyro = np.linalg.norm(imu[:, 3:], axis=1)

    features, names = extract_branch_features(run, ("pelvis_imu6",))
    if tuple(names) != tuple(feature_schema):
        raise RuntimeError("trace feature schema differs from frozen Support input")
    normalized = normalizer.transform(features)[endpoints[selected]]
    top_name = str(row["largest_abs_z_feature"])
    top_index = tuple(feature_schema).index(top_name)

    figure, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(x, probability, label="Support score", color="tab:blue")
    axes[0].axhline(threshold, label="threshold", color="tab:red", linestyle="--")
    axes[0].fill_between(x, 0, 1, where=active, color="tab:green", alpha=0.12, transform=axes[0].get_xaxis_transform(), label="SAND gate")
    axes[0].fill_between(x, 0, 1, where=alert, color="tab:orange", alpha=0.20, transform=axes[0].get_xaxis_transform(), label="persistent output")
    axes[0].set_ylabel("probability")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="lower right", fontsize=8, ncol=2)
    axes[1].plot(x, accel, label="pelvis accel norm")
    axes[1].plot(x, gyro, label="pelvis gyro norm")
    axes[1].set_ylabel("raw norm")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(x, normalized[:, top_index], color="tab:purple", label=top_name)
    axes[2].axhline(0.0, color="black", linewidth=0.7)
    axes[2].set_ylabel("derived z")
    axes[2].legend(loc="upper right", fontsize=7)
    axes[3].step(x, replay.terrain_state[selected], where="post", label="Terrain gate state")
    axes[3].axhline(SAND, color="tab:green", linestyle=":", label="SAND")
    axes[3].set_ylabel("state")
    axes[3].set_xlabel("time relative to Support onset (ms)")
    axes[3].legend(loc="upper right", fontsize=8)
    for axis in axes:
        axis.axvline(0, color="black", linestyle="--", linewidth=1, label="Support onset")
        terrain_valid = row["terrain_valid_sample"]
        terrain_offset = None if terrain_valid is None else int(terrain_valid) - event
        touchdown_offset = int(row["touchdown_sample"]) - event
        if terrain_offset is not None and -180 <= terrain_offset <= 180:
            axis.axvline(terrain_offset, color="tab:green", linestyle="-.", linewidth=1)
        if -180 <= touchdown_offset <= 180:
            axis.axvline(touchdown_offset, color="tab:purple", linestyle=":", linewidth=1)
        axis.set_xlim(-180, 180)
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{run.run_id} | {row['diagnostic_group']} | {row['failure_modes'] or 'detected'}\n"
        f"Terrain valid {terrain_offset} ms; touchdown {touchdown_offset} ms relative to Support"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_representative_traces(
    output_path: Path,
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    replays: Mapping[str, BranchReplay],
    rows: Sequence[Mapping[str, object]],
    normalizer: Normalizer,
    feature_schema: Sequence[str],
    *,
    threshold: float,
    persistence_ms: int,
) -> list[str]:
    misses = [row for row in rows if bool(row["missed"])]
    near = sorted(
        (row for row in rows if row["diagnostic_group"] == "low_margin_success"),
        key=lambda row: (float(row["max_score_minus_threshold"]), str(row["run_id"])),
    )[:6]
    selected = [*misses, *near]
    paths = []
    trace_path = output_path / "traces"
    for row in selected:
        run_id = str(row["run_id"])
        path = trace_path / f"{run_id}.png"
        _plot_trace(
            path,
            runs[run_id],
            gates[run_id],
            replays[run_id],
            row,
            normalizer,
            feature_schema,
            threshold=threshold,
            persistence_ms=persistence_ms,
        )
        paths.append(str(path))
    return paths


def _sha256_rows(rows: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _protected_contract(
    repository_root: Path,
    prior_config: Mapping[str, object],
    selection: Mapping[str, object],
    normalizer_path: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    selected = selection["final_selection"]["support"]["selected"]
    support_artifact = repository_root / "artifacts/runs/20260828_terrain_conditioned_reflex_detector"
    checkpoints = [support_artifact / str(value) for value in selected["checkpoint_paths"]]
    support_paths = [
        repository_root / "configs/experiment/20260828_terrain_conditioned_reflex_detector.yaml",
        repository_root / "src/fastreflex/evaluation/terrain_conditioned_reflex.py",
        support_artifact / "selection_before_holdout.json",
        normalizer_path,
        *checkpoints,
    ]
    terrain_paths = [repository_root / str(value) for value in prior_config["protected_terrain_paths"]]
    slip_freeze_path = repository_root / "artifacts/runs/20260829_continuous_slip_reflex_detector/slip_candidate_freeze.json"
    slip_freeze = json.loads(slip_freeze_path.read_text(encoding="utf-8"))
    recorded_sha = str(slip_freeze.pop("artifact_sha256"))
    if recorded_sha != EXPECTED_SLIP_FREEZE_SHA256 or _canonical_sha256(slip_freeze) != recorded_sha:
        raise RuntimeError("protected Slip freeze artifact identity changed")
    slip_paths = [
        repository_root / "configs/experiment/20260829_continuous_slip_reflex_detector.yaml",
        repository_root / "src/fastreflex/evaluation/continuous_slip_reflex.py",
        slip_freeze_path,
        repository_root / str(slip_freeze["normalizer"]),
        *(repository_root / str(value) for value in slip_freeze["checkpoint_sha256"]),
    ]
    paths = [*terrain_paths, *support_paths, *slip_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("protected path missing: " + ", ".join(missing))
    hashes = {str(path.relative_to(repository_root)): _file_sha256(path) for path in paths}
    return hashes, {
        "terrain_paths": len(terrain_paths),
        "support_paths": len(support_paths),
        "slip_paths": len(slip_paths),
        "slip_freeze_artifact_sha256": recorded_sha,
    }


def run_support_failure_mode_audit(
    repository_root: Path,
    output_path: Path,
) -> dict[str, object]:
    """Replay the frozen Support branch on TRAIN/VALIDATION and write diagnostics."""
    repository_root = repository_root.resolve()
    splits = validate_audit_splits(ALLOWED_SPLITS)
    holdout_guard = EventHoldoutGuard()
    previous_config_path = repository_root / "configs/experiment/20260828_terrain_conditioned_reflex_detector.yaml"
    previous_config = _load_yaml(previous_config_path)
    selection_path = repository_root / "artifacts/runs/20260828_terrain_conditioned_reflex_detector/selection_before_holdout.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["final_selection"]["support"]["selected"]
    normalizer_path = repository_root / "artifacts/runs/20260828_terrain_conditioned_reflex_detector/normalization/phase_a/support/p3.json"
    normalizer, normalizer_document = load_frozen_normalizer(normalizer_path)
    contract = {
        "candidate_id": "P3",
        "components": ["pelvis_imu6"],
        "model_family": "gru",
        "history_ms": 20,
        "threshold": 0.94,
        "persistence_ms": 5,
        "feature_dimension": 60,
        "feature_schema_sha256": "4775bf9cdb1a6680c64c0c744caf69e34afb3628726594350133a59545835170",
    }
    for key, value in contract.items():
        selected_key = "threshold" if key == "threshold" else key
        if selected.get(selected_key) != value:
            raise RuntimeError(f"frozen Support contract changed: {key}")
    schema = feature_schema_for_components(contract["components"])
    if _canonical_sha256(schema) != contract["feature_schema_sha256"]:
        raise RuntimeError("frozen Support feature schema changed")
    if normalizer_document["feature_schema_sha256"] != contract["feature_schema_sha256"]:
        raise RuntimeError("frozen Support normalizer schema changed")
    protected_before, protected_contract = _protected_contract(
        repository_root, previous_config, selection, normalizer_path
    )

    dataset_path = repository_root / str(previous_config["source"]["event_dataset"])
    manifest_path = dataset_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = load_event_runs(dataset_path, manifest, splits)
    if any(run.split not in splits for run in runs.values()):
        raise RuntimeError("loaded waveform escaped development split contract")
    if set(normalizer.fit_run_ids) - set(runs):
        raise RuntimeError("frozen Support normalizer references non-development runs")
    gate_path = repository_root / "artifacts/runs/20260828_terrain_conditioned_reflex_detector/terrain_gate/development"
    gates = load_development_gates(gate_path, runs)
    checkpoint_paths = tuple(
        repository_root
        / "artifacts/runs/20260828_terrain_conditioned_reflex_detector"
        / str(value)
        for value in selected["checkpoint_paths"]
    )
    replays = _replay_many(
        runs,
        gates,
        sorted(runs),
        contract["components"],
        int(contract["history_ms"]),
        normalizer,
        checkpoint_paths,
        None,
    )

    event_rows: list[dict[str, object]] = []
    for run_id, run in sorted(runs.items()):
        if (
            not run.hard_stable_control
            and run.target_terrain == "sand"
            and branch_event_sample(run, "support") is not None
        ):
            event_rows.append(
                event_diagnostic_row(
                    run,
                    gates[run_id],
                    replays[run_id],
                    normalizer,
                    schema,
                    threshold=float(contract["threshold"]),
                    persistence_ms=int(contract["persistence_ms"]),
                )
            )
    grouping = assign_diagnostic_groups(event_rows)
    failure_modes = assign_failure_modes(event_rows)

    split_summary: dict[str, object] = {}
    all_negative_scores: list[float] = []
    for split in splits:
        split_runs = {run_id: run for run_id, run in runs.items() if run.split == split}
        split_replays = {run_id: replays[run_id] for run_id in split_runs}
        canonical = evaluate_branch_replays(
            split_runs,
            {run_id: gates[run_id] for run_id in split_runs},
            split_replays,
            "support",
            float(contract["threshold"]),
            int(contract["persistence_ms"]),
        )
        negative, negative_scores = negative_metrics(
            split_runs,
            split_replays,
            threshold=float(contract["threshold"]),
            persistence_ms=int(contract["persistence_ms"]),
        )
        all_negative_scores.extend(negative_scores)
        selected_rows = [row for row in event_rows if row["split"] == split]
        split_summary[split] = {
            "event_metrics": _recall_rows(selected_rows),
            "canonical_replay_metrics": {
                key: value
                for key, value in canonical.items()
                if key not in ("event_rows", "benign_rows", "hard_rows", "raw_cross_terrain_rows")
            },
            "negative_metrics": {key: value for key, value in negative.items() if key != "rows"},
            "subgroups": subgroup_metrics(selected_rows),
        }

    diagnostic_table = output_path / "event_diagnostics.csv"
    write_event_diagnostics(diagnostic_table, event_rows)
    trace_paths = write_representative_traces(
        output_path,
        runs,
        gates,
        replays,
        event_rows,
        normalizer,
        schema,
        threshold=float(contract["threshold"]),
        persistence_ms=int(contract["persistence_ms"]),
    )
    protected_after, _ = _protected_contract(
        repository_root, previous_config, selection, normalizer_path
    )
    integrity = {
        "protected_hashes_unchanged": protected_before == protected_after,
        "protected_contract": protected_contract,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "dataset_manifest_sha256": _file_sha256(manifest_path),
        "dataset_splits_changed": False,
        "support_retrained": False,
        "slip_modified": False,
        "terrain_modified": False,
        "holdout_access_count": holdout_guard.open_count,
        "holdout_waveform_loaded": False,
        "historical_metrics_files_read": False,
        "audit_splits": list(splits),
    }
    if not integrity["protected_hashes_unchanged"] or holdout_guard.open_count != 0:
        raise RuntimeError("audit integrity contract failed")

    failure_count = len(failure_modes["miss_primary_failure_mode_counts"])
    miss_count = sum(bool(row["missed"]) for row in event_rows)
    if miss_count == 0:
        verdict = "SUPPORT_FAILURE_MODE_INCONCLUSIVE"
    elif failure_count == 1:
        verdict = "SUPPORT_FAILURE_MODE_IDENTIFIED"
    else:
        verdict = "SUPPORT_MULTIPLE_FAILURE_MODES_IDENTIFIED"
    if verdict not in VERDICTS:
        raise RuntimeError("invalid audit verdict")

    summary = {
        "audit_id": AUDIT_ID,
        "scope": "TRAIN_VALIDATION_READ_ONLY",
        "frozen_support_contract": contract,
        "baseline": split_summary,
        "event_diagnostics": {
            "events": len(event_rows),
            "misses": miss_count,
            "rows_sha256": _sha256_rows(event_rows),
            "table": str(diagnostic_table),
            "representative_trace_paths": trace_paths,
            "grouping": grouping,
            "failure_modes": failure_modes,
            "group_distributions": group_distributions(event_rows),
            "outcome_comparison": outcome_comparison(event_rows),
            "terrain_support_timing": timing_comparison(event_rows),
        },
        "sensor_observability": observability_assessment(
            event_rows, all_negative_scores
        ),
        "integrity": integrity,
        "verdict": verdict,
    }
    _write_json(output_path / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("simulation/outputs/support_failure_mode_audit"),
    )
    arguments = parser.parse_args()
    output = arguments.output_path
    if not output.is_absolute():
        output = arguments.repository_root / output
    result = run_support_failure_mode_audit(arguments.repository_root, output)
    print(json.dumps({"verdict": result["verdict"], "output": str(output)}))


if __name__ == "__main__":
    main()
