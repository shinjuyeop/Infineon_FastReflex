"""Causal post-detection Terrain context fusion for the frozen Support branch.

The frozen Support score and five-sample persistence are evaluated continuously.
Terrain output is applied only after the raw alert exists; it cannot modify the
model, hidden state, score, normalization, or persistence counter.
"""

from __future__ import annotations

import argparse
import csv
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
    BranchReplay,
    TerrainGateTrace,
    _canonical_sha256,
    _replay_many,
    branch_event_sample,
    sustained_alert_trace,
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
    per_foot_available = bool(foot_fields) and "state_per_foot" in fields
    return {
        "interface_class": "GLOBAL_HELD_STATE_WITH_PREDICTION_HISTORY",
        "trace_fields": list(fields),
        "global_held_state": "state" in fields,
        "prediction_timestamps": "update_samples" in fields,
        "prediction_classes": "prediction_ids" in fields,
        "prediction_probabilities": "prediction_probabilities" in fields,
        "prediction_foot_identity": False,
        "per_foot_memory": False,
        "foot_related_fields": list(foot_fields),
        "F2_implementable": per_foot_available,
        "F2_result": None if per_foot_available else PER_FOOT_UNAVAILABLE,
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
    if bool(interface["F2_implementable"]):
        raise RuntimeError("F2 implementation requires an explicit reviewed contract")
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
