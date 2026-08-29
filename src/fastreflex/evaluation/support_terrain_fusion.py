"""Causal post-detection Terrain context fusion for the frozen Support branch.

The frozen Support score and five-sample persistence are evaluated continuously.
Terrain output is applied only after the raw alert exists; it cannot modify the
model, hidden state, score, normalization, or persistence counter.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fastreflex.evaluation.continuous_slip_reflex import (
    holdout_gate_from_observer_dataset,
)
from fastreflex.evaluation.reflex_event import (
    EventHoldoutGuard,
    EventRun,
    _load_yaml,
    _write_json,
    load_event_runs,
)
from fastreflex.evaluation.stability_temporal import _file_sha256
from fastreflex.evaluation.support_failure_audit import (
    _percentiles,
    _protected_contract,
    load_development_gates,
    load_frozen_normalizer,
)
from fastreflex.evaluation.terrain_conditioned_reflex import (
    SAND,
    TERRAIN_STATE_NAMES,
    UNKNOWN,
    BranchReplay,
    TerrainGateTrace,
    _canonical_sha256,
    _replay_many,
    branch_event_sample,
    sustained_alert_trace,
    terrain_predictions,
)
from fastreflex.evaluation.transition_scenarios import fusion_regression


EXPERIMENT_ID = "CAUSAL_SUPPORT_TERRAIN_CONTEXT_FUSION"
POLICY_F0 = "F0"
POLICY_F1 = "F1"
POLICY_F2 = "F2"
IMPLEMENTED_POLICIES = (POLICY_F0, POLICY_F1)
VALID_LATENCY_MS = (-30, 50)
PER_FOOT_UNAVAILABLE = "PER_FOOT_TERRAIN_CONTEXT_UNAVAILABLE_WITH_FROZEN_INTERFACE"
VERDICTS = (
    "CAUSAL_SUPPORT_TERRAIN_FUSION_SUPPORTED",
    "CAUSAL_SUPPORT_TERRAIN_FUSION_PROMISING",
    "CAUSAL_SUPPORT_TERRAIN_FUSION_NOT_SUPPORTED",
)
PER_FOOT_EXPERIMENT_ID = "PER_FOOT_TERRAIN_MEMORY_SUPPORT_FUSION"
POLICY_PF1 = "PF1"
POLICY_PF2 = "PF2"
PER_FOOT_POLICIES = (POLICY_PF1, POLICY_PF2)
FOOT_NAMES = ("LEFT", "RIGHT")


@dataclass(frozen=True)
class PerFootTerrainMemoryTrace:
    """Causal held Terrain state and last-update clock for each foot."""

    state: np.ndarray
    last_update_sample: np.ndarray


def per_foot_terrain_memory(
    trace: TerrainGateTrace, samples: int | None = None
) -> PerFootTerrainMemoryTrace:
    """Replay independent LEFT/RIGHT memories from prediction provenance only."""
    sample_count = len(trace.state) if samples is None else int(samples)
    if sample_count != len(trace.state):
        raise ValueError("per-foot memory must align with Terrain trace")
    records = terrain_predictions(trace)
    updates = np.asarray(
        [record.prediction_timestamp for record in records], dtype=np.int64
    )
    predictions = np.asarray([record.class_id for record in records], dtype=np.int64)
    feet_array = np.asarray(
        [record.touchdown_foot for record in records], dtype="<U5"
    )
    if np.any(updates < 0) or np.any(updates >= sample_count):
        raise ValueError("Terrain prediction timestamp is outside the run")
    if np.any(predictions < 0) or np.any(predictions >= 4):
        raise ValueError("Terrain prediction class ID is invalid")
    if any(value not in FOOT_NAMES for value in feet_array.tolist()):
        raise ValueError("Terrain prediction foot must be LEFT or RIGHT")

    state = np.full((sample_count, 2), UNKNOWN, dtype=np.int8)
    last = np.full((sample_count, 2), -1, dtype=np.int64)
    current = np.full(2, UNKNOWN, dtype=np.int8)
    current_last = np.full(2, -1, dtype=np.int64)
    cursor = 0
    order = sorted(range(len(updates)), key=lambda index: (int(updates[index]), index))
    position = 0
    while position < len(order):
        update = int(updates[order[position]])
        state[cursor:update] = current
        last[cursor:update] = current_last
        while position < len(order) and int(updates[order[position]]) == update:
            index = order[position]
            side = FOOT_NAMES.index(str(feet_array[index]))
            current[side] = int(predictions[index]) + 1
            current_last[side] = update
            position += 1
        cursor = update
    state[cursor:] = current
    last[cursor:] = current_last
    return PerFootTerrainMemoryTrace(state=state, last_update_sample=last)


def fsr_loaded_feet(
    fsr8: np.ndarray, *, epsilon_n: float = 1.0e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Derive current support/load state causally from bilateral virtual FSR."""
    values = np.asarray(fsr8, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 8
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or epsilon_n < 0.0
    ):
        raise ValueError("FSR8 must be finite nonnegative [samples,8]")
    totals = np.column_stack((values[:, :4].sum(axis=1), values[:, 4:].sum(axis=1)))
    return totals > float(epsilon_n), totals


def per_foot_context(
    policy: str,
    memory: np.ndarray,
    loaded: np.ndarray,
    totals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PF1/PF2 authorization and causal dominant-foot index."""
    states = np.asarray(memory, dtype=np.int8)
    supports = np.asarray(loaded, dtype=bool)
    forces = np.asarray(totals, dtype=np.float64)
    if not (states.shape == supports.shape == forces.shape) or states.shape[1:] != (2,):
        raise ValueError("per-foot memory/load tensors must align as [samples,2]")
    dominant = np.argmax(forces, axis=1).astype(np.int8)  # exact ties select LEFT
    any_load = np.any(supports, axis=1)
    if policy == POLICY_PF1:
        context = np.any(supports & (states == SAND), axis=1)
    elif policy == POLICY_PF2:
        rows = np.arange(len(states), dtype=np.int64)
        context = any_load & (states[rows, dominant] == SAND)
    else:
        raise ValueError("per-foot policy must be PF1 or PF2")
    return context.astype(bool), dominant


def _context_gate(trace: TerrainGateTrace, context: np.ndarray) -> TerrainGateTrace:
    state = np.where(np.asarray(context, dtype=bool), SAND, UNKNOWN).astype(np.int8)
    return TerrainGateTrace(
        state=state,
        update_samples=trace.update_samples,
        prediction_ids=trace.prediction_ids,
        prediction_probabilities=trace.prediction_probabilities,
        first_target_valid_sample=trace.first_target_valid_sample,
        clean_event_count=trace.clean_event_count,
        prediction_feet=trace.prediction_feet,
    )


def raw_support_alert(
    probabilities: np.ndarray,
    *,
    threshold: float = 0.94,
    persistence_ms: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous raw alert/onset without accepting Terrain input."""
    probability = np.asarray(probabilities, dtype=np.float64)
    return sustained_alert_trace(
        probability,
        np.ones(len(probability), dtype=bool),
        threshold,
        persistence_ms,
    )


def current_sand_context(terrain_state: np.ndarray) -> np.ndarray:
    """F0: authorize only the current frozen global SAND output."""
    return np.asarray(terrain_state, dtype=np.int8) == SAND


def recent_sand_context(
    terrain_state: np.ndarray, *, grace_ms: int = 50
) -> np.ndarray:
    """F1: causal current/recent SAND memory with exact bounded expiry.

    Frozen Terrain output is a held 1 kHz state.  Every sample for which the
    producer continues to output SAND refreshes the causal last-SAND clock.
    A later non-SAND output does not erase memory; the context expires after
    exactly ``grace_ms`` samples without SAND.
    """
    if grace_ms < 0:
        raise ValueError("recent-SAND grace must be nonnegative")
    states = np.asarray(terrain_state, dtype=np.int8)
    context = np.zeros(len(states), dtype=bool)
    last_sand: int | None = None
    for sample, state in enumerate(states):
        if state == SAND:
            last_sand = sample
        context[sample] = (
            last_sand is not None and sample - last_sand <= grace_ms
        )
    return context


def context_for_policy(
    policy: str, terrain_state: np.ndarray, *, grace_ms: int = 50
) -> np.ndarray:
    if policy == POLICY_F0:
        return current_sand_context(terrain_state)
    if policy == POLICY_F1:
        return recent_sand_context(terrain_state, grace_ms=grace_ms)
    raise ValueError(f"policy is not implementable: {policy}")


def support_risk_trace(
    raw_alert: np.ndarray, context: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Post-detection conjunction; returns state and causal rising edges."""
    raw = np.asarray(raw_alert, dtype=bool)
    authorized = np.asarray(context, dtype=bool)
    if raw.shape != authorized.shape:
        raise ValueError("raw alert and Terrain context must align")
    risk = raw & authorized
    onset = risk & ~np.r_[False, risk[:-1]]
    return risk, onset


def terrain_interface_audit() -> dict[str, object]:
    fields = tuple(TerrainGateTrace.__dataclass_fields__)
    foot_fields = tuple(
        value for value in fields if "foot" in value or "side" in value
    )
    prediction_foot_available = "prediction_feet" in fields
    return {
        "interface_class": "HELD_STATE_WITH_TOUCHDOWN_FOOT_PROVENANCE",
        "trace_fields": list(fields),
        "global_held_state": "state" in fields,
        "prediction_timestamps": "update_samples" in fields,
        "prediction_classes": "prediction_ids" in fields,
        "prediction_probabilities": "prediction_probabilities" in fields,
        "prediction_foot_identity": prediction_foot_available,
        "per_foot_memory": prediction_foot_available,
        "foot_related_fields": list(foot_fields),
        "F2_implementable": prediction_foot_available,
        "F2_result": None if prediction_foot_available else PER_FOOT_UNAVAILABLE,
        "exact_terrain_truth_used_by_fusion": False,
    }


def _first_sample(endpoints: np.ndarray, mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return None if not len(indices) else int(endpoints[indices[0]])


def _churn_diagnostic(
    gate: TerrainGateTrace, run: EventRun, event: int
) -> dict[str, object]:
    first = max(0, event - 100)
    last = min(len(gate.state) - 1, event + 100)
    window = np.asarray(gate.state[first : last + 1], dtype=np.int8)
    transitions: list[str] = []
    for before, after in zip(window[:-1], window[1:]):
        if before == after:
            continue
        transitions.append(
            f"{TERRAIN_STATE_NAMES[int(before)]}->{TERRAIN_STATE_NAMES[int(after)]}"
        )
    sand_before = np.flatnonzero(np.asarray(gate.state[: event + 1]) == SAND)
    last_sand = None if not len(sand_before) else int(sand_before[-1])
    first_sand = gate.first_target_valid_sample
    sand_dwell = (
        0
        if first_sand is None or first_sand > event
        else int(np.count_nonzero(gate.state[first_sand : event + 1] == SAND))
    )
    contiguous = 0
    if last_sand is not None:
        cursor = last_sand
        while cursor >= 0 and gate.state[cursor] == SAND:
            contiguous += 1
            cursor -= 1
    return {
        "first_valid_sand_sample": first_sand,
        "last_sand_sample_before_support": last_sand,
        "sand_dwell_samples_before_support": sand_dwell,
        "last_contiguous_sand_dwell_ms": contiguous,
        "last_sand_to_support_gap_ms": (
            None if last_sand is None else event - last_sand
        ),
        "terrain_state_at_support": TERRAIN_STATE_NAMES[int(gate.state[event])],
        "terrain_transitions_support_pm100ms": ";".join(transitions),
        "has_sand_to_marble": "SAND->MARBLE" in transitions,
        "has_sand_to_ice": "SAND->ICE" in transitions,
        "has_sand_to_concrete": "SAND->CONCRETE" in transitions,
        "has_unknown_transition": any("UNKNOWN" in value for value in transitions),
        "stable_sand_pm100ms": bool(len(window) and np.all(window == SAND)),
    }


def _false_positive_mechanism(
    run: EventRun,
    policy: str,
    replay: BranchReplay,
    context: np.ndarray,
    risk_onset: np.ndarray,
) -> str:
    first = np.flatnonzero(risk_onset)
    if not len(first):
        return "none"
    index = int(first[0])
    current_sand = replay.terrain_state[index] == SAND
    if policy == POLICY_F1 and bool(context[index]) and not current_sand:
        return "stale_sand_context"
    if run.target_terrain != "sand" and current_sand:
        return "terrain_misclassification"
    if run.target_terrain == "sand" and branch_event_sample(run, "support") is None:
        return "raw_support_false_alert"
    if policy == POLICY_F1 and bool(context[index]):
        return "context_latch_too_permissive"
    return "unknown"


def evaluate_policy(
    policy: str,
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    replays: Mapping[str, BranchReplay],
    *,
    threshold: float = 0.94,
    persistence_ms: int = 5,
    grace_ms: int = 50,
) -> dict[str, object]:
    """Evaluate one frozen post-detection context policy on one split."""
    if policy not in IMPLEMENTED_POLICIES:
        raise ValueError("only F0/F1 are implementable with the frozen interface")
    event_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    raw_latencies: list[int] = []
    fusion_latencies: list[int] = []
    for run_id, run in sorted(runs.items()):
        replay = replays[run_id]
        raw_alert, raw_onset = raw_support_alert(
            replay.probabilities,
            threshold=threshold,
            persistence_ms=persistence_ms,
        )
        full_context = context_for_policy(
            policy, gates[run_id].state, grace_ms=grace_ms
        )
        context = full_context[replay.endpoints]
        risk, risk_onset = support_risk_trace(raw_alert, context)
        event = branch_event_sample(run, "support")
        relevant = (
            not run.hard_stable_control
            and run.target_terrain == "sand"
            and event is not None
        )
        if relevant:
            assert event is not None
            lower, upper = VALID_LATENCY_MS
            valid = (replay.endpoints >= event + lower) & (
                replay.endpoints <= event + upper
            )
            negative = replay.endpoints < event + lower
            raw_valid_sample = _first_sample(replay.endpoints, raw_onset & valid)
            fusion_valid_sample = _first_sample(
                replay.endpoints, risk_onset & valid
            )
            raw_premature_sample = _first_sample(
                replay.endpoints, raw_onset & negative
            )
            fusion_premature_sample = _first_sample(
                replay.endpoints, risk_onset & negative
            )
            premature_terrain_state = None
            premature_context = None
            premature_mechanism = "none"
            if fusion_premature_sample is not None:
                premature_index = int(
                    np.flatnonzero(
                        replay.endpoints == fusion_premature_sample
                    )[0]
                )
                premature_terrain_state = TERRAIN_STATE_NAMES[
                    int(replay.terrain_state[premature_index])
                ]
                premature_context = bool(context[premature_index])
                if (
                    policy == POLICY_F1
                    and premature_context
                    and replay.terrain_state[premature_index] != SAND
                ):
                    premature_mechanism = "stale_sand_context"
                elif replay.terrain_state[premature_index] == SAND:
                    premature_mechanism = "raw_support_false_alert"
                elif policy == POLICY_F1 and premature_context:
                    premature_mechanism = "context_latch_too_permissive"
                else:
                    premature_mechanism = "unknown"
            raw_present_valid = bool(np.any(raw_alert & valid))
            authorized_valid = bool(np.any(raw_alert & context & valid))
            suppression = raw_present_valid and not authorized_valid
            if raw_valid_sample is not None:
                raw_latencies.append(raw_valid_sample - event)
            if fusion_valid_sample is not None:
                fusion_latencies.append(fusion_valid_sample - event)
            event_index = np.flatnonzero(replay.endpoints == event)
            raw_at_event = bool(raw_alert[event_index[0]]) if len(event_index) else False
            context_at_event = bool(context[event_index[0]]) if len(event_index) else False
            risk_at_event = bool(risk[event_index[0]]) if len(event_index) else False
            event_rows.append(
                {
                    "run_id": run_id,
                    "split": run.split,
                    "policy": policy,
                    "outcome": run.outcome_diagnostic,
                    "source_ground": run.source_terrain,
                    "support_event_sample": event,
                    "raw_first_valid_sample": raw_valid_sample,
                    "raw_latency_ms": (
                        None
                        if raw_valid_sample is None
                        else raw_valid_sample - event
                    ),
                    "raw_premature_sample": raw_premature_sample,
                    "raw_alert_at_support": raw_at_event,
                    "fusion_first_valid_sample": fusion_valid_sample,
                    "fusion_latency_ms": (
                        None
                        if fusion_valid_sample is None
                        else fusion_valid_sample - event
                    ),
                    "fusion_premature_sample": fusion_premature_sample,
                    "fusion_premature_terrain_state": premature_terrain_state,
                    "fusion_premature_context": premature_context,
                    "fusion_premature_mechanism": premature_mechanism,
                    "fusion_detected": fusion_valid_sample is not None,
                    "context_suppression": suppression,
                    "context_at_support": context_at_event,
                    "support_risk_at_support": risk_at_event,
                    "raw_probability_peak_valid": float(
                        np.max(replay.probabilities[valid])
                    ),
                    **_churn_diagnostic(gates[run_id], run, event),
                }
            )
        elif event is None:
            any_risk = bool(np.any(risk))
            negative_rows.append(
                {
                    "run_id": run_id,
                    "split": run.split,
                    "policy": policy,
                    "source_ground": run.source_terrain,
                    "target_terrain": run.target_terrain,
                    "hard_stable_control": run.hard_stable_control,
                    "system_false_reflex": any_risk,
                    "raw_alert_present": bool(np.any(raw_alert)),
                    "false_positive_mechanism": (
                        _false_positive_mechanism(
                            run, policy, replay, context, risk_onset
                        )
                        if any_risk
                        else "none"
                    ),
                }
            )

    event_count = len(event_rows)
    detected = sum(bool(row["fusion_detected"]) for row in event_rows)
    suppressed = sum(bool(row["context_suppression"]) for row in event_rows)
    premature = sum(
        row["fusion_premature_sample"] is not None for row in event_rows
    )
    premature_latencies = [
        int(row["fusion_premature_sample"]) - int(row["support_event_sample"])
        for row in event_rows
        if row["fusion_premature_sample"] is not None
    ]
    raw_premature = sum(
        row["raw_premature_sample"] is not None for row in event_rows
    )
    sand_benign = [
        row
        for row in negative_rows
        if row["target_terrain"] == "sand"
        and not bool(row["hard_stable_control"])
    ]
    hard = [row for row in negative_rows if bool(row["hard_stable_control"])]
    all_fp = sum(bool(row["system_false_reflex"]) for row in negative_rows)
    benign_fp = sum(bool(row["system_false_reflex"]) for row in sand_benign)
    hard_fp = sum(bool(row["system_false_reflex"]) for row in hard)
    mechanisms: dict[str, int] = {}
    for row in negative_rows:
        if bool(row["system_false_reflex"]):
            key = str(row["false_positive_mechanism"])
            mechanisms[key] = mechanisms.get(key, 0) + 1
    premature_mechanisms: dict[str, int] = {}
    for row in event_rows:
        if row["fusion_premature_sample"] is not None:
            key = str(row["fusion_premature_mechanism"])
            premature_mechanisms[key] = premature_mechanisms.get(key, 0) + 1
    return {
        "policy": policy,
        "event_runs": event_count,
        "support_recall": 0.0 if not event_count else detected / event_count,
        "detected_events": detected,
        "missed_events": event_count - detected,
        "raw_detected_events": sum(
            row["raw_first_valid_sample"] is not None for row in event_rows
        ),
        "raw_premature_event_run_rate": (
            0.0 if not event_count else raw_premature / event_count
        ),
        "premature_event_runs": premature,
        "premature_event_run_rate": (
            0.0 if not event_count else premature / event_count
        ),
        "premature_latency_ms": _percentiles(premature_latencies),
        "context_suppression_count": suppressed,
        "context_suppression_rate": (
            0.0 if not event_count else suppressed / event_count
        ),
        "raw_latency_ms": _percentiles(raw_latencies),
        "fusion_latency_ms": _percentiles(fusion_latencies),
        "sand_benign_runs": len(sand_benign),
        "sand_benign_false_reflexes": benign_fp,
        "sand_benign_specificity": (
            1.0 if not sand_benign else 1.0 - benign_fp / len(sand_benign)
        ),
        "hard_ground_runs": len(hard),
        "hard_ground_false_reflexes": hard_fp,
        "hard_ground_specificity": (
            1.0 if not hard else 1.0 - hard_fp / len(hard)
        ),
        "all_negative_runs": len(negative_rows),
        "all_negative_false_reflexes": all_fp,
        "all_negative_specificity": (
            1.0
            if not negative_rows
            else 1.0 - all_fp / len(negative_rows)
        ),
        "false_positive_mechanisms": dict(sorted(mechanisms.items())),
        "premature_mechanisms": dict(sorted(premature_mechanisms.items())),
        "event_rows": event_rows,
        "negative_rows": negative_rows,
    }


def _per_foot_run_state(
    run: EventRun,
    gate: TerrainGateTrace,
    policy: str,
    epsilon_n: float,
) -> tuple[PerFootTerrainMemoryTrace, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    memory = per_foot_terrain_memory(gate)
    fsr8 = np.asarray(run.features["PELVIS_IMU6_FSR8"][:, 6:], dtype=np.float32)
    loaded, totals = fsr_loaded_feet(fsr8, epsilon_n=epsilon_n)
    context, dominant = per_foot_context(
        policy, memory.state, loaded, totals
    )
    return memory, loaded, totals, context, dominant


def _memory_names(values: np.ndarray) -> list[str]:
    return [TERRAIN_STATE_NAMES[int(value)] for value in values]


def _false_authorization_mechanism(
    run: EventRun,
    replay: BranchReplay,
    endpoint_index: int,
    memory: PerFootTerrainMemoryTrace,
    loaded: np.ndarray,
    dominant: np.ndarray,
    policy: str,
) -> str:
    sample = int(replay.endpoints[endpoint_index])
    memories = memory.state[sample]
    active = loaded[sample]
    if policy == POLICY_PF1 and np.any((memories == SAND) & ~active) and not np.any(
        (memories == SAND) & active
    ):
        return "unloaded_sand_memory_foot_incorrectly_authorized"
    selected = (
        np.flatnonzero(active & (memories == SAND))
        if policy == POLICY_PF1
        else np.asarray([int(dominant[sample])], dtype=np.int64)
    )
    ages = [
        sample - int(memory.last_update_sample[sample, side])
        for side in selected
        if int(memory.last_update_sample[sample, side]) >= 0
    ]
    if run.target_terrain != "sand":
        if ages and max(ages) > 50:
            return "stale_per_foot_terrain_memory"
        return "terrain_misclassification"
    if run.event_sample is None:
        return "raw_support_false_alert"
    if ages and max(ages) > 50 and replay.terrain_state[endpoint_index] != SAND:
        return "stale_per_foot_terrain_memory"
    return "raw_support_false_alert"


def evaluate_per_foot_policy(
    policy: str,
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    replays: Mapping[str, BranchReplay],
    *,
    threshold: float = 0.94,
    persistence_ms: int = 5,
    epsilon_n: float = 1.0e-6,
) -> dict[str, object]:
    """Evaluate PF1/PF2 using only prediction provenance and current FSR8."""
    if policy not in PER_FOOT_POLICIES:
        raise ValueError("selection pool contains exactly PF1 and PF2")
    contexts: dict[str, np.ndarray] = {}
    states: dict[
        str,
        tuple[PerFootTerrainMemoryTrace, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    proxy_gates = {}
    for run_id, run in runs.items():
        memory, loaded, totals, context, dominant = _per_foot_run_state(
            run, gates[run_id], policy, epsilon_n
        )
        contexts[run_id] = context
        states[run_id] = (memory, loaded, totals, dominant)
        proxy_gates[run_id] = _context_gate(gates[run_id], context)

    metrics = evaluate_policy(
        POLICY_F0,
        runs,
        proxy_gates,
        replays,
        threshold=threshold,
        persistence_ms=persistence_ms,
    )
    metrics["policy"] = policy
    system_alert_samples = 0
    prediction_count = {"LEFT": 0, "RIGHT": 0}
    class_distribution = {
        foot: {name: 0 for name in TERRAIN_STATE_NAMES[1:]}
        for foot in FOOT_NAMES
    }
    no_memory_support_events = 0

    for run_id, gate in gates.items():
        if gate.prediction_feet is None:
            raise ValueError("per-foot fusion requires prediction foot provenance")
        for foot, prediction in zip(gate.prediction_feet, gate.prediction_ids):
            name = str(foot).upper()
            prediction_count[name] += 1
            class_distribution[name][TERRAIN_STATE_NAMES[int(prediction) + 1]] += 1
        replay = replays[run_id]
        raw, _ = raw_support_alert(
            replay.probabilities,
            threshold=threshold,
            persistence_ms=persistence_ms,
        )
        context_at_endpoints = contexts[run_id][replay.endpoints]
        risk, _ = support_risk_trace(raw, context_at_endpoints)
        system_alert_samples += int(np.count_nonzero(risk))

    event_rows = {str(row["run_id"]): row for row in metrics["event_rows"]}
    for run_id, row in event_rows.items():
        run = runs[run_id]
        memory, loaded, totals, dominant = states[run_id]
        event = int(row["support_event_sample"])
        memories = memory.state[event]
        last = memory.last_update_sample[event]
        active = loaded[event]
        row["policy"] = policy
        row["terrain_memory_at_support"] = _memory_names(memories)
        row["terrain_memory_age_ms_at_support"] = [
            None if int(value) < 0 else event - int(value) for value in last
        ]
        row["loaded_feet_at_support"] = active.astype(bool).tolist()
        row["foot_total_fsr_n_at_support"] = totals[event].astype(float).tolist()
        row["dominant_foot_at_support"] = FOOT_NAMES[int(dominant[event])]
        row["sand_memory_feet_at_support"] = [
            FOOT_NAMES[index]
            for index in np.flatnonzero(memories == SAND).tolist()
        ]
        row["supporting_sand_memory_feet_at_support"] = [
            FOOT_NAMES[index]
            for index in np.flatnonzero(active & (memories == SAND)).tolist()
        ]
        if np.any(active) and not np.any((memories != UNKNOWN) & active):
            no_memory_support_events += 1
        premature = row["fusion_premature_sample"]
        if premature is not None:
            index = int(np.flatnonzero(replays[run_id].endpoints == int(premature))[0])
            row["fusion_premature_mechanism"] = _false_authorization_mechanism(
                run,
                replays[run_id],
                index,
                memory,
                loaded,
                dominant,
                policy,
            )

    mechanisms: dict[str, int] = {}
    for row in metrics["negative_rows"]:
        row["policy"] = policy
        if not bool(row["system_false_reflex"]):
            row["false_positive_mechanism"] = "none"
            continue
        run_id = str(row["run_id"])
        replay = replays[run_id]
        memory, loaded, _, dominant = states[run_id]
        raw, _ = raw_support_alert(
            replay.probabilities,
            threshold=threshold,
            persistence_ms=persistence_ms,
        )
        risk, onset = support_risk_trace(raw, contexts[run_id][replay.endpoints])
        indices = np.flatnonzero(onset)
        index = int(indices[0]) if len(indices) else int(np.flatnonzero(risk)[0])
        mechanism = _false_authorization_mechanism(
            runs[run_id], replay, index, memory, loaded, dominant, policy
        )
        row["false_positive_mechanism"] = mechanism
        mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1

    premature_mechanisms: dict[str, int] = {}
    for row in metrics["event_rows"]:
        if row["fusion_premature_sample"] is not None:
            key = str(row["fusion_premature_mechanism"])
            premature_mechanisms[key] = premature_mechanisms.get(key, 0) + 1
    metrics["false_positive_mechanisms"] = dict(sorted(mechanisms.items()))
    metrics["premature_mechanisms"] = dict(sorted(premature_mechanisms.items()))
    metrics["system_alert_samples"] = system_alert_samples
    metrics["system_alert_duration_ms"] = system_alert_samples
    metrics["terrain_prediction_count_by_foot"] = prediction_count
    metrics["terrain_prediction_class_distribution_by_foot"] = class_distribution
    metrics["support_events_without_valid_loaded_foot_memory"] = no_memory_support_events
    return metrics


def policy_gate_results(
    metrics: Mapping[str, object], gates: Mapping[str, object]
) -> dict[str, bool]:
    latency = metrics["fusion_latency_ms"]
    return {
        "support_recall": float(metrics["support_recall"])
        >= float(gates["support_recall_min"]),
        "sand_benign_specificity": float(metrics["sand_benign_specificity"])
        >= float(gates["sand_benign_specificity_min"]),
        "premature_event_run_rate": float(metrics["premature_event_run_rate"])
        <= float(gates["premature_event_run_rate_max"]),
        "median_fusion_latency": latency["median"] is not None
        and float(latency["median"])
        <= float(gates["median_fusion_latency_ms_max"]),
        "p95_fusion_latency": latency["p95"] is not None
        and float(latency["p95"]) <= float(gates["p95_fusion_latency_ms_max"]),
        "context_suppression_rate": float(metrics["context_suppression_rate"])
        <= float(gates["context_suppression_rate_max"]),
        "hard_ground_specificity": float(metrics["hard_ground_specificity"])
        >= float(gates["hard_ground_specificity_min"]),
    }


def select_validation_policy(
    validation: Mapping[str, Mapping[str, object]],
    train: Mapping[str, Mapping[str, object]],
    validation_gates: Mapping[str, object],
) -> dict[str, object]:
    candidates = []
    for policy in IMPLEMENTED_POLICIES:
        metrics = validation[policy]
        gates = policy_gate_results(metrics, validation_gates)
        historical_suppression = (
            policy == POLICY_F0
            and int(train[policy]["context_suppression_count"]) > 0
        )
        passed = all(gates.values()) and not historical_suppression
        candidates.append(
            {
                "policy": policy,
                "gates": gates,
                "historical_suppression_disqualifies": historical_suppression,
                "passed": passed,
            }
        )
    passing = [row for row in candidates if bool(row["passed"])]
    if not passing:
        return {"selected": None, "candidates": candidates}

    simplicity = {POLICY_F0: 0, POLICY_F1: 1, POLICY_F2: 2}

    def rank(row: Mapping[str, object]) -> tuple[float, ...]:
        metrics = validation[str(row["policy"])]
        latency = metrics["fusion_latency_ms"]
        return (
            -float(metrics["context_suppression_rate"]),
            float(metrics["support_recall"]),
            float(metrics["sand_benign_specificity"]),
            -float(metrics["premature_event_run_rate"]),
            -float(latency["p95"]),
            -float(simplicity[str(row["policy"])]),
        )

    selected = max(passing, key=rank)
    return {
        "selected": str(selected["policy"]),
        "candidates": candidates,
        "selection_priority_applied": True,
    }


def select_per_foot_policy(
    validation: Mapping[str, Mapping[str, object]],
    validation_gates: Mapping[str, object],
) -> dict[str, object]:
    """Select only PF1/PF2 using the predeclared validation ordering."""
    candidates = []
    for policy in PER_FOOT_POLICIES:
        checks = policy_gate_results(validation[policy], validation_gates)
        candidates.append(
            {"policy": policy, "gates": checks, "passed": all(checks.values())}
        )
    passing = [row for row in candidates if bool(row["passed"])]
    if not passing:
        return {"selected": None, "candidates": candidates}
    if len(passing) == 1:
        return {"selected": passing[0]["policy"], "candidates": candidates}

    pf1 = validation[POLICY_PF1]
    pf2 = validation[POLICY_PF2]
    same_primary = (
        float(pf1["support_recall"]) == float(pf2["support_recall"])
        and float(pf1["context_suppression_rate"])
        == float(pf2["context_suppression_rate"])
        and float(pf1["premature_event_run_rate"])
        == float(pf2["premature_event_run_rate"])
        and float(pf1["sand_benign_specificity"])
        == float(pf2["sand_benign_specificity"])
        and float(pf1["hard_ground_specificity"])
        == float(pf2["hard_ground_specificity"])
        and float(pf1["fusion_latency_ms"]["p95"])
        == float(pf2["fusion_latency_ms"]["p95"])
    )
    if same_primary:
        selected = POLICY_PF1
        reason = "near_tie_prefer_pf1"
    else:
        def rank(policy: str) -> tuple[float, ...]:
            metrics = validation[policy]
            return (
                float(metrics["support_recall"]),
                -float(metrics["context_suppression_rate"]),
                -float(metrics["premature_event_run_rate"]),
                float(metrics["sand_benign_specificity"]),
                -float(metrics["fusion_latency_ms"]["p95"]),
                float(policy == POLICY_PF1),
            )

        selected = max(PER_FOOT_POLICIES, key=rank)
        reason = "predeclared_priority"
    return {
        "selected": selected,
        "candidates": candidates,
        "selection_reason": reason,
    }


def _without_rows(metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in ("event_rows", "negative_rows")
    }


def _write_diagnostics(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("fusion diagnostics cannot be empty")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def churn_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    state_at_support: dict[str, int] = {}
    for row in rows:
        state = str(row["terrain_state_at_support"])
        state_at_support[state] = state_at_support.get(state, 0) + 1
    classified = [
        bool(row["has_sand_to_marble"])
        or bool(row["has_sand_to_ice"])
        or bool(row["has_sand_to_concrete"])
        or bool(row["has_unknown_transition"])
        or bool(row["stable_sand_pm100ms"])
        for row in rows
    ]
    return {
        "events": len(rows),
        "sand_to_marble": sum(bool(row["has_sand_to_marble"]) for row in rows),
        "sand_to_ice": sum(bool(row["has_sand_to_ice"]) for row in rows),
        "sand_to_concrete": sum(bool(row["has_sand_to_concrete"]) for row in rows),
        "unknown_transition": sum(
            bool(row["has_unknown_transition"]) for row in rows
        ),
        "stable_sand": sum(bool(row["stable_sand_pm100ms"]) for row in rows),
        "other": sum(not value for value in classified),
        "terrain_state_at_support": dict(sorted(state_at_support.items())),
        "last_sand_to_support_gap_ms": _percentiles(
            [row["last_sand_to_support_gap_ms"] for row in rows]
        ),
        "sand_dwell_samples_before_support": _percentiles(
            [row["sand_dwell_samples_before_support"] for row in rows]
        ),
    }


def _verify_declared_hashes(
    repository_root: Path, document: Mapping[str, object]
) -> dict[str, str]:
    declared = []
    support = document["frozen_support"]
    declared.append((support["normalizer"]["path"], support["normalizer"]["sha256"]))
    declared.extend((row["path"], row["sha256"]) for row in support["checkpoints"])
    terrain = document["frozen_terrain"]
    declared.append((terrain["normalizer"]["path"], terrain["normalizer"]["sha256"]))
    declared.extend((row["path"], row["sha256"]) for row in terrain["checkpoints"])
    result = {}
    for relative, expected in declared:
        path = repository_root / str(relative)
        actual = _file_sha256(path)
        if actual != str(expected):
            raise RuntimeError(f"protected hash changed: {relative}")
        result[str(relative)] = actual
    return result


def _raw_replay_sha(replays: Mapping[str, BranchReplay]) -> str:
    digest = hashlib.sha256()
    for run_id, replay in sorted(replays.items()):
        digest.update(run_id.encode())
        digest.update(np.asarray(replay.endpoints, dtype=np.int64).tobytes())
        digest.update(np.asarray(replay.probabilities, dtype=np.float64).tobytes())
    return digest.hexdigest()


def raw_policy_parity(
    first: Mapping[str, object], second: Mapping[str, object]
) -> bool:
    keys = (
        "support_event_sample",
        "raw_first_valid_sample",
        "raw_latency_ms",
        "raw_premature_sample",
        "raw_alert_at_support",
        "raw_probability_peak_valid",
    )
    first_rows = {str(row["run_id"]): row for row in first["event_rows"]}
    second_rows = {str(row["run_id"]): row for row in second["event_rows"]}
    return first_rows.keys() == second_rows.keys() and all(
        all(first_rows[run_id][key] == second_rows[run_id][key] for key in keys)
        for run_id in first_rows
    )


def _policy_freeze(
    document: Mapping[str, object],
    config_path: Path,
    selected: str,
    protected_hashes: Mapping[str, str],
    validation_metrics: Mapping[str, object],
) -> dict[str, object]:
    policy = document["policies"][selected]
    freeze = {
        "experiment": EXPERIMENT_ID,
        "source_commit": document["experiment"]["source_commit_at_start"],
        "config_path": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "selected_policy": selected,
        "policy_contract": policy,
        "frozen_support": document["frozen_support"],
        "frozen_terrain": document["frozen_terrain"],
        "protected_hashes": dict(protected_hashes),
        "validation_metrics": _without_rows(validation_metrics),
        "holdout_open_count_before_freeze": 0,
        "reselection_after_holdout": False,
    }
    freeze["artifact_sha256"] = _canonical_sha256(freeze)
    return freeze


def _load_frozen_support(
    repository_root: Path, document: Mapping[str, object]
) -> tuple[object, tuple[Path, ...]]:
    support = document["frozen_support"]
    normalizer, normalizer_document = load_frozen_normalizer(
        repository_root / str(support["normalizer"]["path"])
    )
    if normalizer_document["feature_schema_sha256"] != support["feature_schema_sha256"]:
        raise RuntimeError("frozen Support schema changed")
    checkpoints = tuple(
        repository_root / str(row["path"]) for row in support["checkpoints"]
    )
    return normalizer, checkpoints


def _replay_split(
    runs: Mapping[str, EventRun],
    gates: Mapping[str, TerrainGateTrace],
    document: Mapping[str, object],
    normalizer: object,
    checkpoints: Sequence[Path],
) -> dict[str, BranchReplay]:
    support = document["frozen_support"]
    return _replay_many(
        runs,
        gates,
        sorted(runs),
        tuple(str(value) for value in support["components"]),
        int(support["history_ms"]),
        normalizer,  # type: ignore[arg-type]
        checkpoints,
        None,
    )


def run_causal_support_terrain_context_fusion(
    config_path: Path,
    repository_root: Path,
) -> tuple[Path, dict[str, object]]:
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    document = _load_yaml(config_path)
    if document["experiment"]["id"] != EXPERIMENT_ID:
        raise ValueError("unsupported Support Terrain fusion experiment")
    output_path = repository_root / str(document["artifacts"]["path"])
    output_path.mkdir(parents=True, exist_ok=True)
    interface = terrain_interface_audit()
    declared_before = _verify_declared_hashes(repository_root, document)

    prior_config = _load_yaml(
        repository_root / str(document["source"]["prior_reflex_config"])
    )
    selection = json.loads(
        (repository_root / str(document["source"]["prior_selection"])).read_text(
            encoding="utf-8"
        )
    )
    normalizer_path = repository_root / str(
        document["frozen_support"]["normalizer"]["path"]
    )
    broad_before, broad_contract = _protected_contract(
        repository_root, prior_config, selection, normalizer_path
    )
    normalizer, checkpoints = _load_frozen_support(repository_root, document)
    dataset_path = repository_root / str(document["source"]["event_dataset"])
    manifest_path = repository_root / str(document["source"]["event_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    development_runs = load_event_runs(
        dataset_path, manifest, ("train", "validation")
    )
    development_gates = load_development_gates(
        repository_root
        / str(document["source"]["development_terrain_gate_cache"]),
        development_runs,
    )
    development_replays = _replay_split(
        development_runs,
        development_gates,
        document,
        normalizer,
        checkpoints,
    )
    raw_sha = _raw_replay_sha(development_replays)
    grace_ms = int(document["policies"][POLICY_F1]["grace_ms"])
    train: dict[str, dict[str, object]] = {}
    validation: dict[str, dict[str, object]] = {}
    diagnostics: list[dict[str, object]] = []
    for split, destination in (("train", train), ("validation", validation)):
        runs = {
            run_id: run
            for run_id, run in development_runs.items()
            if run.split == split
        }
        gates = {run_id: development_gates[run_id] for run_id in runs}
        replays = {run_id: development_replays[run_id] for run_id in runs}
        for policy in IMPLEMENTED_POLICIES:
            metrics = evaluate_policy(
                policy,
                runs,
                gates,
                replays,
                threshold=float(document["frozen_support"]["probability_threshold"]),
                persistence_ms=int(document["frozen_support"]["persistence_ms"]),
                grace_ms=grace_ms,
            )
            destination[policy] = metrics
            diagnostics.extend(metrics["event_rows"])

    selection_result = select_validation_policy(
        validation,
        train,
        document["selection"]["validation_gates"],
    )
    selected = selection_result["selected"]
    guard = EventHoldoutGuard()
    freeze = None
    holdout: dict[str, object] = {
        "performed": False,
        "guard_open_count": guard.open_count,
        "reason": "no_validation_policy_passed",
    }
    frozen_selection_sha = None
    if selected is not None:
        freeze = _policy_freeze(
            document,
            config_path,
            str(selected),
            declared_before,
            validation[str(selected)],
        )
        frozen_selection_sha = str(freeze["artifact_sha256"])
        _write_json(output_path / "policy_freeze_before_holdout.json", freeze)
        if guard.open_count != 0:
            raise RuntimeError("holdout was accessed before policy freeze")
        guard.open_once()
        holdout_runs = load_event_runs(
            dataset_path, manifest, ("holdout",), holdout_guard=guard
        )
        holdout_gates = holdout_gate_from_observer_dataset(
            prior_config,
            holdout_runs,
            repository_root / str(document["source"]["foot_observer_dataset"]),
            repository_root,
        )
        holdout_replays = _replay_split(
            holdout_runs,
            holdout_gates,
            document,
            normalizer,
            checkpoints,
        )
        selected_metrics = evaluate_policy(
            str(selected),
            holdout_runs,
            holdout_gates,
            holdout_replays,
            threshold=float(document["frozen_support"]["probability_threshold"]),
            persistence_ms=int(document["frozen_support"]["persistence_ms"]),
            grace_ms=grace_ms,
        )
        holdout_gates_result = policy_gate_results(
            selected_metrics, document["holdout"]["gates"]
        )
        diagnostics.extend(selected_metrics["event_rows"])
        holdout = {
            "performed": True,
            "policy": selected,
            "guard_open_count": guard.open_count,
            "metrics": selected_metrics,
            "gates": holdout_gates_result,
            "passed": all(holdout_gates_result.values()),
            "raw_replay_sha256": _raw_replay_sha(holdout_replays),
            "terrain_churn": churn_summary(selected_metrics["event_rows"]),
        }
        freeze_after = json.loads(
            (output_path / "policy_freeze_before_holdout.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            freeze_after["artifact_sha256"] != frozen_selection_sha
            or _canonical_sha256(
                {
                    key: value
                    for key, value in freeze_after.items()
                    if key != "artifact_sha256"
                }
            )
            != frozen_selection_sha
        ):
            raise RuntimeError("policy mutated after holdout")

    f0_misses = {
        row["run_id"]
        for row in train[POLICY_F0]["event_rows"]
        if not bool(row["fusion_detected"])
    }
    rescue = {}
    for policy in IMPLEMENTED_POLICIES:
        detected_ids = {
            row["run_id"]
            for row in train[policy]["event_rows"]
            if bool(row["fusion_detected"])
        }
        rescue[policy] = {
            "historical_misses": len(f0_misses),
            "rescued": len(f0_misses & detected_ids),
            "remaining": len(f0_misses - detected_ids),
            "train_recall": train[policy]["support_recall"],
            "context_suppression": train[policy]["context_suppression_count"],
        }
    rescue[POLICY_F2] = {
        "implementable": False,
        "result": PER_FOOT_UNAVAILABLE,
    }

    broad_after, _ = _protected_contract(
        repository_root, prior_config, selection, normalizer_path
    )
    declared_after = _verify_declared_hashes(repository_root, document)
    if broad_before != broad_after or declared_before != declared_after:
        raise RuntimeError("protected detector or Terrain artifact changed")
    if not fusion_regression()["passed"]:
        raise RuntimeError("fusion regression failed")

    if selected is None:
        verdict = "CAUSAL_SUPPORT_TERRAIN_FUSION_NOT_SUPPORTED"
    elif bool(holdout.get("passed", False)):
        verdict = "CAUSAL_SUPPORT_TERRAIN_FUSION_SUPPORTED"
    else:
        verdict = "CAUSAL_SUPPORT_TERRAIN_FUSION_PROMISING"
    if verdict not in VERDICTS:
        raise RuntimeError("invalid Support Terrain fusion verdict")

    _write_diagnostics(output_path / "event_diagnostics.csv", diagnostics)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "terrain_interface": interface,
        "frozen_support": document["frozen_support"],
        "frozen_terrain": document["frozen_terrain"],
        "raw_support_replay": {
            "development_sha256": raw_sha,
            "policy_independent": True,
            "terrain_can_reset_persistence": False,
            "train_F0_F1_bit_identical": raw_policy_parity(
                train[POLICY_F0], train[POLICY_F1]
            ),
            "validation_F0_F1_bit_identical": raw_policy_parity(
                validation[POLICY_F0], validation[POLICY_F1]
            ),
        },
        "development": {
            "train": {key: _without_rows(value) for key, value in train.items()},
            "validation": {
                key: {
                    **_without_rows(value),
                    "gates": policy_gate_results(
                        value, document["selection"]["validation_gates"]
                    ),
                }
                for key, value in validation.items()
            },
            "F2": {
                "implementable": False,
                "result": PER_FOOT_UNAVAILABLE,
            },
            "train_historical_miss_rescue": rescue,
            "terrain_churn": churn_summary(
                train[POLICY_F0]["event_rows"]
                + validation[POLICY_F0]["event_rows"]
            ),
        },
        "selection": selection_result,
        "freeze": freeze,
        "holdout": (
            {
                **holdout,
                "metrics": _without_rows(holdout["metrics"]),
            }
            if bool(holdout.get("performed"))
            else holdout
        ),
        "integrity": {
            "holdout_access_count_before_freeze": 0,
            "holdout_access_count_final": guard.open_count,
            "policy_mutated_after_holdout": False,
            "support_retrained": False,
            "support_threshold_changed": False,
            "support_persistence_changed": False,
            "terrain_retrained": False,
            "slip_modified": False,
            "protected_hashes_unchanged": True,
            "declared_hashes": declared_after,
            "broad_protected_contract": broad_contract,
            "event_manifest_sha256": _file_sha256(manifest_path),
            "fusion_regression": True,
            "simulator_viewer_physics_modified": False,
        },
        "verdict": verdict,
    }
    summary_path = output_path / "summary.json"
    _write_json(summary_path, summary)
    return summary_path, summary


def _reference_replays(
    replays: Mapping[str, BranchReplay],
    gates: Mapping[str, TerrainGateTrace],
) -> dict[str, BranchReplay]:
    return {
        run_id: BranchReplay(
            endpoints=replay.endpoints,
            probabilities=replay.probabilities,
            terrain_state=gates[run_id].state[replay.endpoints],
        )
        for run_id, replay in replays.items()
    }


def _metric_contract(metrics: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "event_runs",
        "support_recall",
        "detected_events",
        "missed_events",
        "premature_event_runs",
        "premature_event_run_rate",
        "context_suppression_count",
        "context_suppression_rate",
        "sand_benign_runs",
        "sand_benign_specificity",
        "hard_ground_runs",
        "hard_ground_specificity",
    )
    return {key: metrics[key] for key in keys}


def _per_foot_policy_freeze(
    document: Mapping[str, object],
    config_path: Path,
    selected: str,
    reconstruction: Mapping[str, object],
    protected_hashes: Mapping[str, str],
    validation_metrics: Mapping[str, object],
) -> dict[str, object]:
    freeze = {
        "experiment": PER_FOOT_EXPERIMENT_ID,
        "source_commit": document["experiment"]["source_commit_at_start"],
        "config_path": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "selected_policy": selected,
        "policy_contract": document["policies"][selected],
        "terrain_prediction_interface": document["terrain_prediction_interface"],
        "per_foot_memory": document["per_foot_memory"],
        "loaded_foot": document["loaded_foot"],
        "bilateral_terrain": {
            "contract": document["bilateral_terrain"],
            "reconstruction": reconstruction,
        },
        "frozen_support": document["frozen_support"],
        "protected_hashes": dict(protected_hashes),
        "validation_metrics": _without_rows(validation_metrics),
        "holdout_open_count_before_freeze": 0,
        "reselection_after_holdout": False,
    }
    freeze["artifact_sha256"] = _canonical_sha256(freeze)
    return freeze


def _memory_age_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ages = []
    missing = 0
    for row in rows:
        active = row["loaded_feet_at_support"]
        values = row["terrain_memory_age_ms_at_support"]
        selected = [value for value, loaded in zip(values, active) if loaded]
        if not selected or all(value is None for value in selected):
            missing += 1
        ages.extend(int(value) for value in selected if value is not None)
    return {
        "loaded_foot_memory_age_ms": _percentiles(ages),
        "events_without_valid_loaded_foot_memory": missing,
    }


def run_per_foot_terrain_memory_support_fusion(
    config_path: Path,
    repository_root: Path,
) -> tuple[Path, dict[str, object]]:
    """Reconstruct fixed bilateral Terrain output, select PF1/PF2, open holdout once."""
    from fastreflex.training.terrain import reconstruct_bilateral_shared_candidate

    root = repository_root.resolve()
    config_path = config_path.resolve()
    document = _load_yaml(config_path)
    if document["experiment"]["id"] != PER_FOOT_EXPERIMENT_ID:
        raise ValueError("unsupported per-foot Terrain fusion experiment")
    output = root / str(document["artifacts"]["path"])
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError("refusing to open the per-foot holdout more than once")
    output.mkdir(parents=True, exist_ok=True)

    source = document["source"]
    manifest_path = root / str(source["event_manifest"])
    if _file_sha256(manifest_path) != str(source["event_manifest_sha256"]):
        raise RuntimeError("reflex event manifest changed")
    slip_path = root / str(source["slip_freeze"])
    if _file_sha256(slip_path) != str(source["slip_freeze_sha256"]):
        raise RuntimeError("frozen Slip artifact changed")

    prior_global_document = _load_yaml(
        root / str(source["prior_global_fusion_config"])
    )
    protected_before = _verify_declared_hashes(root, prior_global_document)
    reconstruction_path, reconstruction = reconstruct_bilateral_shared_candidate(
        document["bilateral_terrain"], root
    )
    if not bool(reconstruction["historical_validation_parity"]):
        raise RuntimeError("bilateral Terrain validation parity failed")
    runtime_terrain_document = {
        "source": {"terrain_models": str(reconstruction_path.relative_to(root))},
        "terrain_branch": {"deployment_scheme": "bilateral_shared"},
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = root / str(source["event_dataset"])
    observer_path = root / str(source["foot_observer_dataset"])
    development_runs = load_event_runs(
        dataset_path, manifest, ("train", "validation")
    )
    bilateral_gates = holdout_gate_from_observer_dataset(
        runtime_terrain_document, development_runs, observer_path, root
    )
    if not all(
        gate.prediction_feet is not None for gate in bilateral_gates.values()
    ):
        raise RuntimeError("bilateral Terrain output lost touchdown-foot provenance")

    normalizer, checkpoints = _load_frozen_support(root, document)
    development_replays = _replay_split(
        development_runs,
        bilateral_gates,
        document,
        normalizer,
        checkpoints,
    )
    raw_sha = _raw_replay_sha(development_replays)
    prior_summary = json.loads(
        (root / str(source["prior_global_fusion_summary"])).read_text(
            encoding="utf-8"
        )
    )
    if raw_sha != prior_summary["raw_support_replay"]["development_sha256"]:
        raise RuntimeError("frozen raw Support replay changed")

    historical_gates = load_development_gates(
        root / str(source["historical_gate_cache"]), development_runs
    )
    historical_replays = _reference_replays(development_replays, historical_gates)
    reference: dict[str, dict[str, dict[str, object]]] = {
        "train": {},
        "validation": {},
    }
    per_foot: dict[str, dict[str, dict[str, object]]] = {
        "train": {},
        "validation": {},
    }
    diagnostics: list[dict[str, object]] = []
    negative_diagnostics: list[dict[str, object]] = []
    epsilon = float(document["loaded_foot"]["epsilon_n"])
    support = document["frozen_support"]
    for split in ("train", "validation"):
        runs = {
            run_id: run
            for run_id, run in development_runs.items()
            if run.split == split
        }
        old_gates = {run_id: historical_gates[run_id] for run_id in runs}
        new_gates = {run_id: bilateral_gates[run_id] for run_id in runs}
        old_replays = {run_id: historical_replays[run_id] for run_id in runs}
        new_replays = {run_id: development_replays[run_id] for run_id in runs}
        for policy in (POLICY_F0, POLICY_F1):
            result = evaluate_policy(
                policy,
                runs,
                old_gates,
                old_replays,
                threshold=float(support["probability_threshold"]),
                persistence_ms=int(support["persistence_ms"]),
                grace_ms=50,
            )
            reference[split][policy] = result
            expected = prior_summary["development"][split][policy]
            if _metric_contract(result) != _metric_contract(expected):
                raise RuntimeError(f"historical {split} {policy} reproduction failed")
        for policy in PER_FOOT_POLICIES:
            result = evaluate_per_foot_policy(
                policy,
                runs,
                new_gates,
                new_replays,
                threshold=float(support["probability_threshold"]),
                persistence_ms=int(support["persistence_ms"]),
                epsilon_n=epsilon,
            )
            per_foot[split][policy] = result
            diagnostics.extend(result["event_rows"])
            negative_diagnostics.extend(result["negative_rows"])

    selection = select_per_foot_policy(
        per_foot["validation"], document["selection"]["validation_gates"]
    )
    selected = selection["selected"]
    guard = EventHoldoutGuard()
    freeze = None
    holdout: dict[str, object] = {
        "performed": False,
        "guard_open_count": 0,
        "reason": "no_per_foot_policy_passed_validation",
    }
    if selected is not None:
        freeze = _per_foot_policy_freeze(
            document,
            config_path,
            str(selected),
            reconstruction,
            protected_before,
            per_foot["validation"][str(selected)],
        )
        freeze_path = output / "selection_before_holdout.json"
        _write_json(freeze_path, freeze)
        freeze_sha = str(freeze["artifact_sha256"])
        if guard.open_count != 0:
            raise RuntimeError("Support holdout was accessed before policy freeze")
        guard.open_once()
        holdout_runs = load_event_runs(
            dataset_path, manifest, ("holdout",), holdout_guard=guard
        )
        holdout_gates = holdout_gate_from_observer_dataset(
            runtime_terrain_document, holdout_runs, observer_path, root
        )
        holdout_replays = _replay_split(
            holdout_runs,
            holdout_gates,
            document,
            normalizer,
            checkpoints,
        )
        selected_metrics = evaluate_per_foot_policy(
            str(selected),
            holdout_runs,
            holdout_gates,
            holdout_replays,
            threshold=float(support["probability_threshold"]),
            persistence_ms=int(support["persistence_ms"]),
            epsilon_n=epsilon,
        )
        checks = policy_gate_results(selected_metrics, document["holdout"]["gates"])
        diagnostics.extend(selected_metrics["event_rows"])
        negative_diagnostics.extend(selected_metrics["negative_rows"])
        freeze_after = json.loads(freeze_path.read_text(encoding="utf-8"))
        if (
            freeze_after["artifact_sha256"] != freeze_sha
            or _canonical_sha256(
                {
                    key: value
                    for key, value in freeze_after.items()
                    if key != "artifact_sha256"
                }
            )
            != freeze_sha
        ):
            raise RuntimeError("per-foot policy mutated after holdout")
        holdout = {
            "performed": True,
            "guard_open_count": guard.open_count,
            "policy": selected,
            "metrics": selected_metrics,
            "gates": checks,
            "passed": all(checks.values()),
            "raw_replay_sha256": _raw_replay_sha(holdout_replays),
            "memory_age": _memory_age_summary(selected_metrics["event_rows"]),
        }

    historical_misses = {
        str(row["run_id"])
        for row in reference["train"][POLICY_F0]["event_rows"]
        if not bool(row["fusion_detected"])
    }
    miss_details = []
    for run_id in sorted(historical_misses):
        old_row = next(
            row
            for row in reference["train"][POLICY_F0]["event_rows"]
            if str(row["run_id"]) == run_id
        )
        rows = {
            policy: next(
                row
                for row in per_foot["train"][policy]["event_rows"]
                if str(row["run_id"]) == run_id
            )
            for policy in PER_FOOT_POLICIES
        }
        miss_details.append(
            {
                "run_id": run_id,
                "historical_terrain_state": old_row["terrain_state_at_support"],
                "historical_last_sand_gap_ms": old_row[
                    "last_sand_to_support_gap_ms"
                ],
                "terrain_memory_at_support": rows[POLICY_PF1][
                    "terrain_memory_at_support"
                ],
                "loaded_feet_at_support": rows[POLICY_PF1][
                    "loaded_feet_at_support"
                ],
                "dominant_foot_at_support": rows[POLICY_PF1][
                    "dominant_foot_at_support"
                ],
                "supporting_sand_memory_feet": rows[POLICY_PF1][
                    "supporting_sand_memory_feet_at_support"
                ],
                "PF1_rescued": bool(rows[POLICY_PF1]["fusion_detected"]),
                "PF2_rescued": bool(rows[POLICY_PF2]["fusion_detected"]),
            }
        )
    rescue = {
        policy: {
            "historical_misses": len(historical_misses),
            "rescued": sum(
                bool(row[f"{policy}_rescued"]) for row in miss_details
            ),
            "long_gap_marble_misses": sum(
                row["historical_terrain_state"] == "MARBLE"
                and row["historical_last_sand_gap_ms"] is not None
                and int(row["historical_last_sand_gap_ms"]) > 50
                for row in miss_details
            ),
            "long_gap_marble_rescued": sum(
                row["historical_terrain_state"] == "MARBLE"
                and row["historical_last_sand_gap_ms"] is not None
                and int(row["historical_last_sand_gap_ms"]) > 50
                and bool(row[f"{policy}_rescued"])
                for row in miss_details
            ),
        }
        for policy in PER_FOOT_POLICIES
    }

    protected_after = _verify_declared_hashes(root, prior_global_document)
    if protected_before != protected_after:
        raise RuntimeError("protected Support or selected Terrain artifact changed")
    if _file_sha256(slip_path) != str(source["slip_freeze_sha256"]):
        raise RuntimeError("frozen Slip artifact changed after evaluation")
    if not fusion_regression()["passed"]:
        raise RuntimeError("fusion regression failed")

    if selected is None:
        verdict = "PER_FOOT_TERRAIN_SUPPORT_FUSION_NOT_SUPPORTED"
    elif bool(holdout.get("passed")):
        verdict = "PER_FOOT_TERRAIN_SUPPORT_FUSION_SUPPORTED"
    else:
        verdict = "PER_FOOT_TERRAIN_SUPPORT_FUSION_PROMISING"

    _write_diagnostics(output / "event_diagnostics.csv", diagnostics)
    _write_diagnostics(output / "negative_diagnostics.csv", negative_diagnostics)
    summary = {
        "experiment_id": PER_FOOT_EXPERIMENT_ID,
        "start_state": document["experiment"],
        "bilateral_terrain": {
            "artifact_path": str(reconstruction_path.relative_to(root)),
            **reconstruction,
        },
        "terrain_prediction_interface": {
            **document["terrain_prediction_interface"],
            "prediction_foot_preserved_end_to_end": True,
            "model_tensor_contains_foot_id": False,
            "simulator_terrain_truth_in_runtime_output": False,
        },
        "per_foot_memory": document["per_foot_memory"],
        "loaded_foot": document["loaded_foot"],
        "raw_support_replay": {
            "development_sha256": raw_sha,
            "matches_prior_audit": True,
            "terrain_memory_can_reset_persistence": False,
            "PF1_PF2_bit_identical": raw_policy_parity(
                per_foot["train"][POLICY_PF1],
                per_foot["train"][POLICY_PF2],
            )
            and raw_policy_parity(
                per_foot["validation"][POLICY_PF1],
                per_foot["validation"][POLICY_PF2],
            ),
        },
        "historical_references": {
            split: {
                policy: _without_rows(metrics)
                for policy, metrics in values.items()
            }
            for split, values in reference.items()
        },
        "development": {
            split: {
                policy: {
                    **_without_rows(metrics),
                    **(
                        {
                            "gates": policy_gate_results(
                                metrics, document["selection"]["validation_gates"]
                            )
                        }
                        if split == "validation"
                        else {}
                    ),
                }
                for policy, metrics in values.items()
            }
            for split, values in per_foot.items()
        },
        "historical_miss_rescue": rescue,
        "historical_miss_details": miss_details,
        "selection": selection,
        "freeze": freeze,
        "holdout": (
            {**holdout, "metrics": _without_rows(holdout["metrics"])}
            if bool(holdout.get("performed"))
            else holdout
        ),
        "sensor_implication": {
            "terrain_and_context": "BILATERAL_FSR8",
            "support_detector": "PELVIS_IMU6",
            "unique_physical_channels": 14,
            "additional_sensor_type_required": False,
            "recommendation": (
                "BILATERAL_FSR8_PLUS_PELVIS_IMU6_SYSTEM_CANDIDATE"
                if verdict == "PER_FOOT_TERRAIN_SUPPORT_FUSION_SUPPORTED"
                else "SYSTEM_SENSOR_ARCHITECTURE_UNRESOLVED"
            ),
            "final_sensor_architecture_frozen": False,
        },
        "integrity": {
            "holdout_access_count_before_freeze": 0,
            "holdout_access_count_final": guard.open_count,
            "policy_mutated_after_holdout": False,
            "support_retrained": False,
            "support_threshold_changed": False,
            "terrain_model_selected_from_support_task": False,
            "terrain_holdout_access_for_reconstruction": 0,
            "slip_modified": False,
            "protected_hashes_unchanged": True,
            "protected_hashes": protected_after,
            "fusion_regression": True,
            "simulator_viewer_physics_modified": False,
        },
        "verdict": verdict,
    }
    _write_json(summary_path, summary)
    return summary_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiment/20260829_causal_support_terrain_context_fusion.yaml"
        ),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    experiment = _load_yaml(arguments.config)["experiment"]["id"]
    if experiment == PER_FOOT_EXPERIMENT_ID:
        summary_path, summary = run_per_foot_terrain_memory_support_fusion(
            arguments.config, arguments.repository_root
        )
    else:
        summary_path, summary = run_causal_support_terrain_context_fusion(
            arguments.config, arguments.repository_root
        )
    print(
        json.dumps(
            {
                "verdict": summary["verdict"],
                "summary": str(summary_path),
            }
        )
    )


if __name__ == "__main__":
    main()
