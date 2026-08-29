"""Resolve the frozen Support detector's early continuous-alert mode.

The evaluator first audits gait periodicity and simulator-only support physics
on TRAIN/VALIDATION.  Privileged spread signals are kept out of the frozen
Pelvis-IMU tensor.  If the early mode is a physical precursor, the three fixed
incipient reference candidates are evaluated before any detector retraining.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from fastreflex.dataset.loader import Normalizer
from fastreflex.evaluation.reflex_event import (
    EVENT_TYPE_BOTH,
    EVENT_TYPE_SUPPORT,
    EventHoldoutGuard,
    EventRun,
    _load_yaml,
    _write_json,
    load_event_runs,
)
from fastreflex.evaluation.stability_temporal import _file_sha256
from fastreflex.evaluation.support_failure_audit import (
    _percentiles,
    load_development_gates,
    load_frozen_normalizer,
)
from fastreflex.evaluation.support_terrain_fusion import raw_support_alert
from fastreflex.evaluation.terrain_conditioned_reflex import (
    BranchReplay,
    _replay_many,
    branch_event_sample,
    extract_branch_features,
    feature_schema_for_components,
)


EXPERIMENT_ID = "SUPPORT_EARLY_MODE_RESOLUTION"
CAUSE_VERDICTS = (
    "SUPPORT_EARLY_MODE_GAIT_ALIAS",
    "SUPPORT_EARLY_MODE_PHYSICAL_PRECURSOR",
    "SUPPORT_EARLY_MODE_MIXED",
    "SUPPORT_EARLY_MODE_UNRESOLVED",
)
FINAL_VERDICTS = (
    "CONTINUOUS_SUPPORT_REFLEX_SUPPORTED",
    "INCIPIENT_SUPPORT_REFLEX_SUPPORTED",
    "CONTINUOUS_SUPPORT_REFLEX_PROMISING",
    "CONTINUOUS_SUPPORT_REFLEX_NOT_SUPPORTED",
)
EARLY_CLASSES = (
    "GAIT_ALIAS_FALSE_MODE",
    "PHYSICAL_PRECURSOR_MODE",
    "MIXED_OR_UNRESOLVED",
)
PHASE_NAMES = ("NO_SUPPORT", "LEFT_SUPPORT", "RIGHT_SUPPORT", "DOUBLE_SUPPORT")
INCIPIENT_IDS = ("I1", "I2", "I3")


@dataclass(frozen=True)
class IncipientFit:
    """TRAIN-benign-only parameters for one causal physical reference."""

    candidate_id: str
    threshold: float
    mean: np.ndarray
    std: np.ndarray
    fit_run_ids: tuple[str, ...]
    fit_sample_count: int
    quantile: float


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def support_event_sample(run: EventRun) -> int | None:
    return branch_event_sample(run, "support")


def phase_ids(loaded_contact: np.ndarray) -> np.ndarray:
    loaded = np.asarray(loaded_contact, dtype=bool)
    if loaded.ndim != 2 or loaded.shape[1] != 2:
        raise ValueError("loaded contact must be [samples,2]")
    return (loaded[:, 0].astype(np.int8) + 2 * loaded[:, 1].astype(np.int8))


def touchdown_samples(
    loaded_contact: np.ndarray, *, minimum_same_foot_separation_ms: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic per-foot rising edges with chatter suppression."""
    loaded = np.asarray(loaded_contact, dtype=bool)
    previous = np.vstack((np.zeros((1, 2), dtype=bool), loaded[:-1]))
    rising = loaded & ~previous
    result: list[np.ndarray] = []
    for side in range(2):
        kept: list[int] = []
        for sample in np.flatnonzero(rising[:, side]):
            value = int(sample)
            if not kept or value - kept[-1] >= minimum_same_foot_separation_ms:
                kept.append(value)
        result.append(np.asarray(kept, dtype=np.int64))
    return result[0], result[1]


def nearest_prior(values: np.ndarray, sample: int) -> int | None:
    candidates = np.asarray(values, dtype=np.int64)
    prior = candidates[candidates <= int(sample)]
    return None if not len(prior) else int(prior[-1])


def next_after(values: np.ndarray, sample: int) -> int | None:
    candidates = np.asarray(values, dtype=np.int64)
    following = candidates[candidates > int(sample)]
    return None if not len(following) else int(following[0])


def affected_support_foot(run: EventRun) -> int | None:
    values = run.support_event_samples_per_foot
    finite = [(int(value), side) for side, value in enumerate(values) if value is not None]
    return None if not finite else min(finite)[1]


def waveform_similarity(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    """Deterministic flattened and channel-wise 20 ms similarity diagnostics."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not a.size:
        raise ValueError("waveform pairs must align as nonempty [time,channels]")
    difference = float(np.linalg.norm(a - b) / np.sqrt(a.size))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = 0.0 if denominator == 0.0 else float(np.sum(a * b) / denominator)
    correlations = []
    for channel in range(a.shape[1]):
        x, y = a[:, channel], b[:, channel]
        if np.std(x) > 0.0 and np.std(y) > 0.0:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
    return {
        "normalized_l2": difference,
        "cosine_similarity": cosine,
        "mean_per_channel_correlation": (
            None if not correlations else float(np.mean(correlations))
        ),
    }


def physical_features(run: EventRun) -> tuple[np.ndarray, tuple[str, ...]]:
    """Causal privileged diagnostics; this function is never a model input."""
    spread = np.asarray(run.support_spread_m, dtype=np.float64)
    displacement = np.asarray(run.support_max_displacement_m, dtype=np.float64)
    if spread.shape != displacement.shape or spread.shape[1:] != (2,):
        raise ValueError("Support diagnostic arrays must align as [samples,2]")
    aggregate = [np.max(spread, axis=1), np.max(displacement, axis=1)]
    names = ["support_surface_spread_m", "support_surface_max_displacement_m"]
    for lag in (1, 5, 10, 20, 50):
        delta = np.zeros_like(spread)
        if lag < len(spread):
            delta[lag:] = spread[lag:] - spread[:-lag]
        aggregate.append(np.max(np.maximum(delta, 0.0), axis=1))
        names.append(f"spread_delta_{lag}ms_m")
    fsr = np.asarray(run.features["PELVIS_IMU6_FSR8"], dtype=np.float64)[:, 6:]
    left, right = fsr[:, :4].sum(axis=1), fsr[:, 4:].sum(axis=1)
    aggregate.extend((left, right, np.abs(left - right) / (left + right + 1.0e-6)))
    names.extend(("fsr_left_total_n", "fsr_right_total_n", "fsr_bilateral_imbalance"))
    values = np.column_stack(aggregate)
    if not np.all(np.isfinite(values)):
        raise ValueError("Support privileged diagnostics are nonfinite")
    return values, tuple(names)


def fit_phase_physical_envelope(
    runs: Mapping[str, EventRun], *, quantile: float = 0.995
) -> dict[str, object]:
    """Fit q99.5 using TRAIN no-Support evidence only, never outcome fields."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("physical-envelope quantile must lie inside (0,1)")
    chunks: dict[int, list[np.ndarray]] = {phase: [] for phase in range(4)}
    fit_ids = []
    names: tuple[str, ...] | None = None
    for run_id, run in sorted(runs.items()):
        if run.split != "train" or support_event_sample(run) is not None:
            continue
        values, schema = physical_features(run)
        names = schema if names is None else names
        if names != schema:
            raise RuntimeError("physical feature schema changed")
        phases = phase_ids(run.loaded_contact)
        valid = np.arange(run.first_contact_sample, run.censor_sample)
        for phase in range(4):
            selected = valid[phases[valid] == phase]
            if len(selected):
                chunks[phase].append(values[selected])
        fit_ids.append(run_id)
    if names is None or not fit_ids:
        raise ValueError("TRAIN no-Support physical envelope is empty")
    bounds = {}
    counts = {}
    for phase, parts in chunks.items():
        if not parts:
            raise ValueError(f"physical phase {phase} has no benign samples")
        values = np.concatenate(parts)
        bounds[PHASE_NAMES[phase]] = np.quantile(
            values, quantile, axis=0, method="linear"
        ).tolist()
        counts[PHASE_NAMES[phase]] = int(len(values))
    return {
        "quantile": float(quantile),
        "feature_names": list(names),
        "phase_bounds": bounds,
        "phase_sample_counts": counts,
        "fit_run_ids": fit_ids,
        "fit_run_count": len(fit_ids),
        "train_only": True,
        "validation_excluded": True,
        "fall_outcome_used": False,
    }


def physical_envelope_exit(
    run: EventRun, envelope: Mapping[str, object], *, persistence_ms: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    values, names = physical_features(run)
    if tuple(envelope["feature_names"]) != names:
        raise ValueError("physical envelope schema mismatch")
    phases = phase_ids(run.loaded_contact)
    abnormal = np.zeros(len(values), dtype=bool)
    # Physical Support divergence uses spread/delta channels only.  FSR and
    # absolute displacement remain diagnostics because their benign scale does
    # not encode heterogeneous support spread.
    physical_columns = [0, 2, 3, 4, 5, 6]
    spread = np.asarray(run.support_spread_m, dtype=np.float64)
    side_deltas = []
    for lag in (1, 5, 10, 20, 50):
        delta = np.zeros_like(spread)
        if lag < len(spread):
            delta[lag:] = spread[lag:] - spread[:-lag]
        side_deltas.append(np.maximum(delta, 0.0))
    side_values = np.stack((spread, *side_deltas), axis=2)
    for phase in range(4):
        mask = phases == phase
        bound = np.asarray(envelope["phase_bounds"][PHASE_NAMES[phase]])
        physical_bound = bound[physical_columns]
        # A persistent terrain deformation is a current support-dynamics
        # precursor only while the affected foot is loaded.  Unloaded samples
        # reset persistence even though the deformed simulator surface itself
        # remains deformed for later footsteps.
        abnormal[mask] = np.any(
            run.loaded_contact[mask, :, None]
            & (side_values[mask] > physical_bound[None, None, :]),
            axis=(1, 2),
        )
    active = np.zeros(len(values), dtype=bool)
    onset = np.zeros(len(values), dtype=bool)
    count = 0
    previous = False
    for sample, value in enumerate(abnormal):
        count = count + 1 if bool(value) else 0
        current = count >= int(persistence_ms)
        active[sample] = current
        onset[sample] = current and not previous
        previous = current
    return active, onset


def classify_early_mode(
    *, gait_alias: bool, physical_precursor: bool
) -> str:
    if physical_precursor and not gait_alias:
        return "PHYSICAL_PRECURSOR_MODE"
    if gait_alias and not physical_precursor:
        return "GAIT_ALIAS_FALSE_MODE"
    if physical_precursor and gait_alias:
        # Physical evidence has priority: gait periodicity does not make an
        # abnormal support state a valid negative example.
        return "PHYSICAL_PRECURSOR_MODE"
    return "MIXED_OR_UNRESOLVED"


def incipient_score(run: EventRun, fit: IncipientFit) -> np.ndarray:
    spread = np.asarray(run.support_spread_m, dtype=np.float64)
    derivative = np.zeros_like(spread)
    derivative[1:] = spread[1:] - spread[:-1]
    delta20 = np.zeros_like(spread)
    delta20[20:] = spread[20:] - spread[:-20]
    if fit.candidate_id == "I1":
        return np.max(
            np.where(run.loaded_contact, np.maximum(derivative, 0.0), 0.0),
            axis=1,
        )
    if fit.candidate_id == "I2":
        return np.max(
            np.where(run.loaded_contact, np.maximum(delta20, 0.0), 0.0),
            axis=1,
        )
    if fit.candidate_id == "I3":
        base = np.column_stack(
            (
                np.max(np.where(run.loaded_contact, spread, 0.0), axis=1),
                np.max(
                    np.where(
                        run.loaded_contact, np.maximum(derivative, 0.0), 0.0
                    ),
                    axis=1,
                ),
            )
        )
        z = (base - fit.mean) / fit.std
        return np.linalg.norm(np.maximum(z, 0.0), axis=1)
    raise ValueError("unknown incipient candidate")


def fit_incipient_candidate(
    candidate_id: str,
    runs: Mapping[str, EventRun],
    *,
    quantile: float = 0.995,
) -> IncipientFit:
    """Fit exactly I1/I2/I3 from TRAIN no-Support trajectories."""
    if candidate_id not in INCIPIENT_IDS:
        raise ValueError("incipient candidate must be I1, I2, or I3")
    ids = []
    base_chunks = []
    for run_id, run in sorted(runs.items()):
        if run.split != "train" or support_event_sample(run) is not None:
            continue
        spread = np.asarray(run.support_spread_m, dtype=np.float64)
        derivative = np.zeros_like(spread)
        derivative[1:] = spread[1:] - spread[:-1]
        valid = slice(run.first_contact_sample, run.censor_sample)
        base_chunks.append(
            np.column_stack(
                (
                    np.max(np.where(run.loaded_contact, spread, 0.0), axis=1),
                    np.max(
                        np.where(
                            run.loaded_contact,
                            np.maximum(derivative, 0.0),
                            0.0,
                        ),
                        axis=1,
                    ),
                )
            )[valid]
        )
        ids.append(run_id)
    if not base_chunks:
        raise ValueError("incipient fit requires TRAIN no-Support runs")
    base = np.concatenate(base_chunks)
    mean = base.mean(axis=0)
    std = base.std(axis=0)
    std[std < 1.0e-6] = 1.0e-6
    provisional = IncipientFit(candidate_id, 0.0, mean, std, tuple(ids), len(base), quantile)
    scores = []
    for run_id in ids:
        run = runs[run_id]
        scores.append(incipient_score(run, provisional)[run.first_contact_sample : run.censor_sample])
    threshold = float(np.quantile(np.concatenate(scores), quantile, method="linear"))
    return IncipientFit(candidate_id, threshold, mean, std, tuple(ids), len(base), quantile)


def persistent_onset(
    score: np.ndarray,
    threshold: float,
    *,
    persistence_ms: int,
    first_sample: int,
    censor_sample: int,
) -> int | None:
    """First causal strict-envelope exceedance; current and past only."""
    count = 0
    for sample in range(int(first_sample), int(censor_sample)):
        count = count + 1 if float(score[sample]) > float(threshold) else 0
        if count >= int(persistence_ms):
            return sample
    return None


def evaluate_incipient_candidate(
    fit: IncipientFit,
    runs: Mapping[str, EventRun],
    *,
    split: str,
    persistence_ms: int = 20,
) -> dict[str, object]:
    rows = []
    negative = []
    leads = []
    onset_map: dict[str, int | None] = {}
    for run_id, run in sorted(runs.items()):
        if run.split != split:
            continue
        onset = persistent_onset(
            incipient_score(run, fit),
            fit.threshold,
            persistence_ms=persistence_ms,
            first_sample=run.first_contact_sample,
            censor_sample=run.censor_sample,
        )
        onset_map[run_id] = onset
        established = support_event_sample(run)
        if established is not None:
            before = onset is not None and onset < established
            if before:
                leads.append(established - int(onset))
            rows.append(
                {
                    "run_id": run_id,
                    "incipient_sample": onset,
                    "established_sample": established,
                    "before_established": before,
                    "lead_ms": None if not before else established - int(onset),
                }
            )
        else:
            negative.append(
                {
                    "run_id": run_id,
                    "false_event": onset is not None,
                    "target_terrain": run.target_terrain,
                    "hard_ground": run.hard_stable_control,
                }
            )
    def false_rate(selected: Sequence[Mapping[str, object]]) -> float:
        return 0.0 if not selected else sum(bool(row["false_event"]) for row in selected) / len(selected)
    sand = [row for row in negative if row["target_terrain"] == "sand" and not row["hard_ground"]]
    ice = [row for row in negative if row["target_terrain"] == "ice" and not row["hard_ground"]]
    hard = [row for row in negative if row["hard_ground"]]
    coverage = 0.0 if not rows else sum(bool(row["before_established"]) for row in rows) / len(rows)
    metrics = {
        "candidate_id": fit.candidate_id,
        "threshold": fit.threshold,
        "split": split,
        "event_runs": len(rows),
        "coverage_before_established": coverage,
        "sand_benign_false_event_rate": false_rate(sand),
        "ice_non_support_false_event_rate": false_rate(ice),
        "hard_ground_false_event_rate": false_rate(hard),
        "lead_ms": _percentiles(leads),
        "event_rows": rows,
        "negative_rows": negative,
        "onset_map": onset_map,
    }
    metrics["gates"] = {
        "sand_benign_false_event_rate": metrics["sand_benign_false_event_rate"] <= 0.05,
        "ice_non_support_false_event_rate": metrics["ice_non_support_false_event_rate"] <= 0.05,
        "hard_ground_false_event_rate": metrics["hard_ground_false_event_rate"] <= 0.05,
        "coverage": coverage >= 0.90,
        "before_established": coverage >= 0.90,
        "median_lead": metrics["lead_ms"]["median"] is not None and metrics["lead_ms"]["median"] >= 50.0,
    }
    metrics["passed"] = all(metrics["gates"].values())
    return metrics


def mine_support_hard_negatives(
    endpoints: np.ndarray,
    probabilities: np.ndarray,
    *,
    positive_region: tuple[int, int] | None,
    gait_alias_endpoints: Sequence[int] = (),
    top_k: int = 16,
    minimum_separation_ms: int = 30,
    excluded: Sequence[int] = (),
) -> np.ndarray:
    """Deterministic TRAIN-only HNM contract retained for Branch A/C tests."""
    values = np.asarray(endpoints, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if values.shape != scores.shape:
        raise ValueError("HNM endpoint/probability arrays differ")
    forbidden = set(int(value) for value in excluded)
    if positive_region is not None:
        lower, upper = positive_region
        forbidden.update(int(value) for value in values[(values >= lower) & (values <= upper)])
    score_by_endpoint = {int(endpoint): float(score) for endpoint, score in zip(values, scores)}
    ordered_aliases = [
        int(value) for value in gait_alias_endpoints
        if int(value) in score_by_endpoint and int(value) not in forbidden
    ]
    ordered_scores = [
        int(values[index])
        for index in sorted(range(len(values)), key=lambda i: (-scores[i], int(values[i])))
        if int(values[index]) not in forbidden
    ]
    selected: list[int] = []
    for endpoint in (*ordered_aliases, *ordered_scores):
        if endpoint in selected:
            continue
        if all(abs(endpoint - prior) >= minimum_separation_ms for prior in selected):
            selected.append(endpoint)
            if len(selected) >= top_k:
                break
    return np.asarray(sorted(selected), dtype=np.int64)


def support_threshold_values() -> tuple[float, ...]:
    return tuple(round(0.50 + 0.01 * index, 2) for index in range(50))


def audit_early_alert(
    run: EventRun,
    replay: BranchReplay,
    premature_sample: int,
    normalizer: Normalizer,
    envelope: Mapping[str, object],
    matched_runs: Mapping[str, EventRun],
    *,
    history_ms: int = 20,
    alias_tolerance_ms: int = 50,
    touchdown_separation_ms: int = 100,
) -> dict[str, object]:
    event = support_event_sample(run)
    side = affected_support_foot(run)
    if event is None or side is None:
        raise ValueError("early audit requires a Support event")
    left, right = touchdown_samples(
        run.loaded_contact,
        minimum_same_foot_separation_ms=touchdown_separation_ms,
    )
    by_foot = (left, right)
    all_touchdowns = np.unique(np.concatenate((left, right)))
    alert_td = nearest_prior(by_foot[side], premature_sample)
    event_td = nearest_prior(by_foot[side], event)
    previous_same = None
    if alert_td is not None:
        earlier = by_foot[side][by_foot[side] < alert_td]
        previous_same = None if not len(earlier) else int(earlier[-1])
    previous_step_touchdown = None
    if alert_td is not None:
        earlier_steps = all_touchdowns[all_touchdowns < alert_td]
        previous_step_touchdown = (
            None if not len(earlier_steps) else int(earlier_steps[-1])
        )
    stride = None if alert_td is None or event_td is None else event_td - alert_td
    lead = event - premature_sample
    mismatch = None if stride is None else abs(lead - stride)
    gait_alias = mismatch is not None and mismatch <= alias_tolerance_ms

    features, schema = extract_branch_features(run, ("pelvis_imu6",))
    normalized = normalizer.transform(features)
    offsets = np.arange(history_ms - 1, -1, -1, dtype=np.int64)
    early_window = normalized[premature_sample - offsets]
    event_window = normalized[event - offsets]
    pair = waveform_similarity(early_window, event_window)

    phases = phase_ids(run.loaded_contact)
    touchdown_age = None if alert_td is None else premature_sample - alert_td
    matches: list[tuple[int, str, int]] = []
    for run_id, candidate in sorted(matched_runs.items()):
        candidate_phases = phase_ids(candidate.loaded_contact)
        candidate_touchdowns = touchdown_samples(
            candidate.loaded_contact,
            minimum_same_foot_separation_ms=touchdown_separation_ms,
        )[side]
        for endpoint in range(max(candidate.first_contact_sample, history_ms - 1), candidate.censor_sample):
            if candidate_phases[endpoint] != phases[premature_sample] or not candidate.loaded_contact[endpoint, side]:
                continue
            prior = nearest_prior(candidate_touchdowns, endpoint)
            if prior is None or touchdown_age is None:
                continue
            difference = abs((endpoint - prior) - touchdown_age)
            if difference <= 50:
                matches.append((difference, run_id, endpoint))
    matches.sort(key=lambda value: (value[0], value[1], value[2]))
    matched_similarity = []
    for _, run_id, endpoint in matches[:32]:
        candidate_features, candidate_schema = extract_branch_features(
            matched_runs[run_id], ("pelvis_imu6",)
        )
        if candidate_schema != schema:
            raise RuntimeError("matched frozen Support schema changed")
        window = normalizer.transform(candidate_features)[endpoint - offsets]
        matched_similarity.append(waveform_similarity(early_window, window))

    physical, onset = physical_envelope_exit(run, envelope, persistence_ms=20)
    physical_at_alert = bool(physical[premature_sample])
    prior_physical_onsets = np.flatnonzero(onset & (np.arange(len(onset)) <= premature_sample))
    latest_physical_onset = None if not len(prior_physical_onsets) else int(prior_physical_onsets[-1])
    classification = classify_early_mode(
        gait_alias=gait_alias,
        physical_precursor=physical_at_alert,
    )
    values, physical_names = physical_features(run)
    return {
        "run_id": run.run_id,
        "split": run.split,
        "raw_alert_sample": premature_sample,
        "support_sample": event,
        "premature_lead_ms": lead,
        "affected_foot": "LEFT" if side == 0 else "RIGHT",
        "alert_phase": PHASE_NAMES[int(phases[premature_sample])],
        "support_phase": PHASE_NAMES[int(phases[event])],
        "loaded_left_at_alert": bool(run.loaded_contact[premature_sample, 0]),
        "loaded_right_at_alert": bool(run.loaded_contact[premature_sample, 1]),
        "recent_left_touchdown": nearest_prior(left, premature_sample),
        "recent_right_touchdown": nearest_prior(right, premature_sample),
        "previous_same_foot_touchdown": previous_same,
        "alert_same_foot_touchdown": alert_td,
        "event_same_foot_touchdown": event_td,
        "next_same_foot_touchdown": next_after(by_foot[side], premature_sample),
        "previous_step_touchdown": previous_step_touchdown,
        "step_period_ms": (
            None
            if previous_step_touchdown is None or alert_td is None
            else alert_td - previous_step_touchdown
        ),
        "previous_same_foot_period_ms": (
            None
            if previous_same is None or alert_td is None
            else alert_td - previous_same
        ),
        "stride_period_ms": stride,
        "lead_stride_absolute_mismatch_ms": mismatch,
        "gait_alias_timing_match": gait_alias,
        "waveform_pair": pair,
        "matched_normal_windows": len(matched_similarity),
        "matched_normal_similarity": {
            key: _percentiles([row[key] for row in matched_similarity])
            for key in ("normalized_l2", "cosine_similarity", "mean_per_channel_correlation")
        },
        "physical_diagnostics": {
            name: float(values[premature_sample, index])
            for index, name in enumerate(physical_names)
        },
        "persistent_physical_envelope_exit_at_alert": physical_at_alert,
        "latest_physical_divergence_onset": latest_physical_onset,
        "physical_onset_to_alert_ms": None if latest_physical_onset is None else premature_sample - latest_physical_onset,
        "early_mode_classification": classification,
    }


def evaluate_continuous_support(
    runs: Mapping[str, EventRun],
    replays: Mapping[str, BranchReplay],
    *,
    split: str,
    threshold: float,
    persistence_ms: int,
    incipient_onsets: Mapping[str, int | None] | None = None,
) -> dict[str, object]:
    event_rows = []
    negative_rows = []
    latencies = []
    negative_samples = negative_alert_samples = 0
    for run_id, run in sorted(runs.items()):
        if run.split != split:
            continue
        replay = replays[run_id]
        alert, onset = raw_support_alert(
            replay.probabilities,
            threshold=threshold,
            persistence_ms=persistence_ms,
        )
        onset_samples = replay.endpoints[onset]
        event = support_event_sample(run)
        if event is not None:
            valid = onset_samples[(onset_samples >= event - 30) & (onset_samples <= event + 50)]
            established_premature = onset_samples[onset_samples < event - 30]
            incipient = None if incipient_onsets is None else incipient_onsets.get(run_id)
            boundary = event - 30 if incipient is None else int(incipient)
            premature = onset_samples[onset_samples < boundary]
            first_valid = None if not len(valid) else int(valid[0])
            if first_valid is not None:
                latencies.append(first_valid - event)
            event_rows.append(
                {
                    "run_id": run_id,
                    "valid_detection": first_valid is not None,
                    "valid_sample": first_valid,
                    "established_latency_ms": None if first_valid is None else first_valid - event,
                    "established_premature": bool(len(established_premature)),
                    "first_established_premature_sample": None if not len(established_premature) else int(established_premature[0]),
                    "incipient_sample": incipient,
                    "incipient_premature": bool(len(premature)),
                    "first_alert_after_incipient_sample": (
                        None if incipient is None or not np.any(onset_samples >= incipient)
                        else int(onset_samples[onset_samples >= incipient][0])
                    ),
                }
            )
            negative_mask = replay.endpoints < boundary
        else:
            false_alert = bool(len(onset_samples))
            negative_rows.append(
                {
                    "run_id": run_id,
                    "false_alert": false_alert,
                    "target_terrain": run.target_terrain,
                    "hard_ground": run.hard_stable_control,
                    "first_alert_sample": None if not len(onset_samples) else int(onset_samples[0]),
                }
            )
            negative_mask = np.ones(len(replay.endpoints), dtype=bool)
        negative_samples += int(np.count_nonzero(negative_mask))
        negative_alert_samples += int(np.count_nonzero(alert & negative_mask))

    def specificity(selected: Sequence[Mapping[str, object]]) -> float:
        return 1.0 if not selected else 1.0 - sum(bool(row["false_alert"]) for row in selected) / len(selected)
    sand = [row for row in negative_rows if row["target_terrain"] == "sand" and not row["hard_ground"]]
    ice = [row for row in negative_rows if row["target_terrain"] == "ice" and not row["hard_ground"]]
    hard = [row for row in negative_rows if row["hard_ground"]]
    return {
        "split": split,
        "threshold": threshold,
        "support_event_runs": len(event_rows),
        "support_recall": 0.0 if not event_rows else sum(bool(row["valid_detection"]) for row in event_rows) / len(event_rows),
        "established_premature_event_run_rate": 0.0 if not event_rows else sum(bool(row["established_premature"]) for row in event_rows) / len(event_rows),
        "incipient_premature_event_run_rate": None if incipient_onsets is None else (0.0 if not event_rows else sum(bool(row["incipient_premature"]) for row in event_rows) / len(event_rows)),
        "sand_benign_specificity": specificity(sand),
        "ice_non_support_specificity": specificity(ice),
        "hard_ground_specificity": specificity(hard),
        "negative_time_alert_fraction": 0.0 if not negative_samples else negative_alert_samples / negative_samples,
        "latency_ms": _percentiles(latencies),
        "event_rows": event_rows,
        "negative_rows": negative_rows,
    }


def continuous_validation_gates(metrics: Mapping[str, object]) -> dict[str, bool]:
    latency = metrics["latency_ms"]
    premature = metrics["incipient_premature_event_run_rate"]
    if premature is None:
        premature = metrics["established_premature_event_run_rate"]
    return {
        "support_recall": float(metrics["support_recall"]) >= 0.95,
        "premature_event_run_rate": float(premature) <= 0.10,
        "sand_benign_specificity": float(metrics["sand_benign_specificity"]) >= 0.95,
        "ice_non_support_specificity": float(metrics["ice_non_support_specificity"]) >= 0.90,
        "hard_ground_specificity": float(metrics["hard_ground_specificity"]) >= 0.95,
        "negative_time_alert_fraction": float(metrics["negative_time_alert_fraction"]) <= 0.02,
        "median_latency_ms": latency["median"] is not None and float(latency["median"]) <= 20.0,
        "p95_latency_ms": latency["p95"] is not None and float(latency["p95"]) <= 50.0,
    }


def _verify_hashes(root: Path, document: Mapping[str, object]) -> dict[str, str]:
    declared = []
    support = document["frozen_support"]
    declared.append((support["normalizer"]["path"], support["normalizer"]["sha256"]))
    declared.extend((row["path"], row["sha256"]) for row in support["checkpoints"])
    terrain = document["protected"]["terrain"]
    declared.append((terrain["normalizer"]["path"], terrain["normalizer"]["sha256"]))
    declared.extend((row["path"], row["sha256"]) for row in terrain["checkpoints"])
    slip = document["protected"]["slip_freeze"]
    declared.append((slip["path"], slip["sha256"]))
    result = {}
    for relative, expected in declared:
        path = root / str(relative)
        actual = _file_sha256(path)
        if actual != str(expected):
            raise RuntimeError(f"protected hash changed: {relative}")
        result[str(relative)] = actual
    return result


def _distribution_field(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float | None]:
    return _percentiles([row[field] for row in rows])


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if not isinstance(row[key], (dict, list))})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def run_support_early_mode_resolution(
    config_path: Path, repository_root: Path
) -> tuple[Path, dict[str, object]]:
    root = repository_root.resolve()
    document = _load_yaml(config_path.resolve())
    if document["experiment"]["id"] != EXPERIMENT_ID:
        raise ValueError("unsupported Support early-mode experiment")
    output = root / str(document["artifacts"]["output_path"])
    output.mkdir(parents=True, exist_ok=True)
    protected_before = _verify_hashes(root, document)
    manifest_path = root / str(document["source"]["event_manifest"])
    if _file_sha256(manifest_path) != str(document["source"]["event_manifest_sha256"]):
        raise RuntimeError("event manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = load_event_runs(
        root / str(document["source"]["event_dataset"]),
        manifest,
        ("train", "validation"),
    )
    guard = EventHoldoutGuard()
    if guard.open_count != 0:
        raise RuntimeError("Support holdout was accessed before selection")
    gates = load_development_gates(
        root / str(document["source"]["development_terrain_gate_cache"]), runs
    )
    normalizer, normalizer_document = load_frozen_normalizer(
        root / str(document["frozen_support"]["normalizer"]["path"])
    )
    schema = feature_schema_for_components(("pelvis_imu6",))
    if normalizer_document["feature_schema_sha256"] != document["frozen_support"]["feature_schema_sha256"]:
        raise RuntimeError("frozen Support feature schema changed")
    checkpoints = tuple(
        root / str(row["path"]) for row in document["frozen_support"]["checkpoints"]
    )
    replays = _replay_many(
        runs,
        gates,
        sorted(runs),
        ("pelvis_imu6",),
        int(document["frozen_support"]["history_ms"]),
        normalizer,
        checkpoints,
        None,
    )
    threshold = float(document["frozen_support"]["probability_threshold"])
    persistence = int(document["frozen_support"]["persistence_ms"])
    envelope = fit_phase_physical_envelope(
        runs, quantile=float(document["phase_a_audit"]["physical_envelope"]["quantile"])
    )
    matched = {
        run_id: run for run_id, run in runs.items()
        if run.split == "train" and support_event_sample(run) is None
    }
    audits = []
    for run_id, run in sorted(runs.items()):
        event = support_event_sample(run)
        if event is None:
            continue
        replay = replays[run_id]
        _, onset = raw_support_alert(
            replay.probabilities, threshold=threshold, persistence_ms=persistence
        )
        premature = replay.endpoints[onset & (replay.endpoints < event - 30)]
        if len(premature):
            audits.append(
                audit_early_alert(
                    run,
                    replay,
                    int(premature[0]),
                    normalizer,
                    envelope,
                    matched,
                    history_ms=int(document["frozen_support"]["history_ms"]),
                    alias_tolerance_ms=int(document["phase_a_audit"]["gait"]["alias_period_tolerance_ms"]),
                    touchdown_separation_ms=int(document["phase_a_audit"]["gait"]["repeated_same_foot_rising_suppression_ms"]),
                )
            )
    counts = {label: sum(row["early_mode_classification"] == label for row in audits) for label in EARLY_CLASSES}
    if audits and counts["PHYSICAL_PRECURSOR_MODE"] / len(audits) >= 0.80:
        cause = "SUPPORT_EARLY_MODE_PHYSICAL_PRECURSOR"
    elif audits and counts["GAIT_ALIAS_FALSE_MODE"] / len(audits) >= 0.80:
        cause = "SUPPORT_EARLY_MODE_GAIT_ALIAS"
    elif counts["PHYSICAL_PRECURSOR_MODE"] and counts["GAIT_ALIAS_FALSE_MODE"]:
        cause = "SUPPORT_EARLY_MODE_MIXED"
    else:
        cause = "SUPPORT_EARLY_MODE_UNRESOLVED"

    incipient = {}
    selected_incipient = None
    incipient_onsets = None
    if cause in ("SUPPORT_EARLY_MODE_PHYSICAL_PRECURSOR", "SUPPORT_EARLY_MODE_MIXED"):
        for candidate_id in INCIPIENT_IDS:
            fit = fit_incipient_candidate(candidate_id, runs, quantile=0.995)
            train_metrics = evaluate_incipient_candidate(fit, runs, split="train", persistence_ms=20)
            validation_metrics = evaluate_incipient_candidate(fit, runs, split="validation", persistence_ms=20)
            incipient[candidate_id] = {
                "fit": {
                    "threshold": fit.threshold,
                    "mean": fit.mean.tolist(),
                    "std": fit.std.tolist(),
                    "fit_run_ids": list(fit.fit_run_ids),
                    "fit_sample_count": fit.fit_sample_count,
                    "quantile": fit.quantile,
                    "train_only": True,
                    "fall_outcome_used": False,
                },
                "train": train_metrics,
                "validation": validation_metrics,
            }
        passing = [candidate_id for candidate_id in INCIPIENT_IDS if incipient[candidate_id]["validation"]["passed"]]
        if passing:
            selected_incipient = passing[0]
            incipient_onsets = incipient[selected_incipient]["validation"]["onset_map"] | incipient[selected_incipient]["train"]["onset_map"]

    train_raw = evaluate_continuous_support(
        runs, replays, split="train", threshold=threshold,
        persistence_ms=persistence, incipient_onsets=incipient_onsets,
    )
    validation_raw = evaluate_continuous_support(
        runs, replays, split="validation", threshold=threshold,
        persistence_ms=persistence, incipient_onsets=incipient_onsets,
    )
    validation_checks = continuous_validation_gates(validation_raw)
    validation_passed = all(validation_checks.values())
    holdout = {
        "performed": False,
        "guard_open_count": guard.open_count,
        "reason": (
            "continuous_support_candidate_failed_validation"
            if not validation_passed
            else "runner_requires_explicit_frozen_candidate_before_holdout"
        ),
    }
    # This bounded execution cannot promote a reference when the frozen runtime
    # detector fails a primary continuous specificity gate.
    if validation_passed and selected_incipient is not None:
        verdict = "CONTINUOUS_SUPPORT_REFLEX_PROMISING"
    else:
        verdict = "CONTINUOUS_SUPPORT_REFLEX_NOT_SUPPORTED"

    protected_after = _verify_hashes(root, document)
    if protected_after != protected_before:
        raise RuntimeError("protected artifacts changed during Support audit")
    summary = {
        "experiment": EXPERIMENT_ID,
        "start_state": document["experiment"],
        "development_runs": {
            split: sum(run.split == split for run in runs.values())
            for split in ("train", "validation")
        },
        "holdout_access_before_selection": 0,
        "frozen_support": {
            "feature_dimension": len(schema),
            "history_ms": int(document["frozen_support"]["history_ms"]),
            "threshold": threshold,
            "persistence_ms": persistence,
            "physical_oracle": document["frozen_support"]["physical_oracle"],
        },
        "phase_a": {
            "premature_alerts": len(audits),
            "by_split": {
                split: sum(row["split"] == split for row in audits)
                for split in ("train", "validation")
            },
            "classification_counts": counts,
            "premature_lead_ms": _distribution_field(audits, "premature_lead_ms"),
            "step_period_ms": _distribution_field(audits, "step_period_ms"),
            "stride_period_ms": _distribution_field(audits, "stride_period_ms"),
            "lead_stride_absolute_mismatch_ms": _distribution_field(audits, "lead_stride_absolute_mismatch_ms"),
            "gait_alias_timing_matches": sum(bool(row["gait_alias_timing_match"]) for row in audits),
            "persistent_physical_precursors": sum(bool(row["persistent_physical_envelope_exit_at_alert"]) for row in audits),
            "spread_at_alert_m": _percentiles([row["physical_diagnostics"]["support_surface_spread_m"] for row in audits]),
            "waveform_pair": {
                key: _percentiles([row["waveform_pair"][key] for row in audits])
                for key in ("normalized_l2", "cosine_similarity", "mean_per_channel_correlation")
            },
            "per_cell_displacement_available": False,
            "rows": audits,
        },
        "physical_envelope": envelope,
        "cause_verdict": cause,
        "branch": (
            "BRANCH_B_INCIPIENT_REFERENCE"
            if cause == "SUPPORT_EARLY_MODE_PHYSICAL_PRECURSOR"
            else "BRANCH_C_MIXED"
            if cause == "SUPPORT_EARLY_MODE_MIXED"
            else "BRANCH_A_HNM"
            if cause == "SUPPORT_EARLY_MODE_GAIT_ALIAS"
            else "NO_SAFE_BRANCH"
        ),
        "hnm_performed": False,
        "incipient": {
            "performed": bool(incipient),
            "candidates": incipient,
            "selected": selected_incipient,
        },
        "continuous_raw": {
            "train": train_raw,
            "validation": validation_raw,
            "validation_gates": validation_checks,
            "validation_passed": validation_passed,
        },
        "selection": {
            "selected_runtime_candidate": None,
            "reason": "validation_ice_non_support_specificity_failed" if not validation_checks["ice_non_support_specificity"] else "validation_gate_failed",
        },
        "holdout": holdout,
        "integrated_replay": {"performed": False, "reason": "Support validation did not pass"},
        "sensor_implication": {
            "support_runtime_sensor": "Pelvis IMU6",
            "augmentation_performed": False,
            "augmentation_justified_in_this_milestone": False,
        },
        "protected_hashes": protected_after,
        "cause_verdict_valid": cause in CAUSE_VERDICTS,
        "verdict": verdict,
    }
    _write_json(output / "summary.json", summary)
    _write_rows(output / "early_alert_audit.csv", audits)
    return output, summary
