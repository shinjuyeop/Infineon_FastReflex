"""Bounded terrain/stability integration sanity experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from fastreflex.dataset.loader import WindowSet
from fastreflex.models.baselines import parameter_count
from fastreflex.simulation.g1 import (
    SimulationConfig,
    SimulationResult,
    load_simulation_config,
    run_simulation,
)
from fastreflex.simulation.stability import (
    HazardState,
    IMURuleCalibration,
    InstabilityTrace,
    ParallelRuntimeState,
    PhaseEnvelope,
    StabilityState,
    StableCalibrationRun,
    TerrainState,
    causal_persistence,
    detect_instability,
    fit_imu_rule,
    fit_phase_envelope,
    format_runtime_status,
    run_imu_rule,
)
from fastreflex.training.trainer import (
    evaluate_model,
    save_checkpoint,
    train_model,
)


STABILITY_CLASS_NAMES = ("STABLE", "UNSTABLE")


@dataclass(frozen=True)
class IntegratedRun:
    specification: Mapping[str, object]
    simulation: SimulationResult
    instability: InstabilityTrace
    rule_onset: np.ndarray
    rule_active: np.ndarray

    @property
    def run_id(self) -> str:
        return str(self.specification["id"])


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _first_true(values: np.ndarray) -> int | None:
    found = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if not found.size else int(found[0])


def _first_true_any(values: np.ndarray) -> int | None:
    array = np.asarray(values, dtype=bool)
    return _first_true(np.any(array, axis=1))


def _time_us(result: SimulationResult, sample: int | None) -> int | None:
    if sample is None or sample < 0 or sample >= len(result.runtime.timestamp_us):
        return None
    return int(result.runtime.timestamp_us[sample])


def _time_ms(result: SimulationResult, sample: int | None) -> float | None:
    value = _time_us(result, sample)
    return None if value is None else value / 1000.0


def _transition_sample(run: IntegratedRun | SimulationResult, terrain: str) -> int | None:
    result = run.simulation if isinstance(run, IntegratedRun) else run
    if terrain == "ice":
        return _first_true_any(result.diagnostics.low_friction_patch_contact_onset)
    if terrain == "sand":
        return _first_true_any(result.diagnostics.soft_patch_contact_onset)
    return None


def _terrain_state(name: str) -> TerrainState:
    try:
        return TerrainState[name.upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported terrain state: {name}") from exc


def _simulation_config(
    base: SimulationConfig,
    specification: Mapping[str, object],
    policy_path: Path,
    duration_s: float,
) -> SimulationConfig:
    return replace(
        base,
        duration_s=duration_s,
        command_speed_mps=float(specification["speed_mps"]),
        policy_path=policy_path,
        terrain=str(specification["terrain"]),
        slip_pattern=str(specification["slip_pattern"]),
        sink_pattern=str(specification["sink_pattern"]),
        sink_severity=str(specification["sink_severity"]),
        sink_support_pattern=str(specification["support_pattern"]),
        patch_start_x_m=float(specification["patch_start_x_m"]),
        patch_width_m=float(specification["patch_width_m"]),
        headless=True,
    )


def _scenario_gate(
    specifications: Sequence[Mapping[str, object]],
    simulations: Mapping[str, SimulationResult],
    acceptance: Mapping[str, object],
) -> dict[str, object]:
    stable_min = int(acceptance["observed_stable_min_per_terrain"])
    fall_fraction_min = float(
        acceptance["observed_fall_fraction_min_per_intended_fall_terrain"]
    )
    stable_counts: dict[str, dict[str, int]] = {}
    fall_counts: dict[str, dict[str, int | float]] = {}
    pretransition_falls: list[str] = []
    missing_transitions: list[str] = []
    nonfinite: list[str] = []
    for terrain in ("ice", "sand"):
        stable_specs = [
            item
            for item in specifications
            if item["terrain"] == terrain and item["intended_role"] == "stable"
        ]
        fall_specs = [
            item
            for item in specifications
            if item["terrain"] == terrain and item["intended_role"] == "fall"
        ]
        observed_stable = sum(
            simulations[str(item["id"])].metadata["first_fall_sample"] is None
            for item in stable_specs
        )
        observed_fall = sum(
            simulations[str(item["id"])].metadata["first_fall_sample"] is not None
            for item in fall_specs
        )
        stable_counts[terrain] = {
            "observed_stable": observed_stable,
            "intended_stable": len(stable_specs),
        }
        fall_counts[terrain] = {
            "observed_fall": observed_fall,
            "intended_fall": len(fall_specs),
            "fraction": observed_fall / len(fall_specs),
        }
    for item in specifications:
        run_id = str(item["id"])
        result = simulations[run_id]
        terrain = str(item["terrain"])
        transition = _transition_sample(result, terrain)
        fall = result.metadata["first_fall_sample"]
        if terrain in {"ice", "sand"} and transition is None:
            missing_transitions.append(run_id)
        if fall is not None and transition is not None and int(fall) < transition:
            pretransition_falls.append(run_id)
        if not np.all(np.isfinite(result.runtime.pelvis_imu)):
            nonfinite.append(run_id)
    passed = bool(
        all(values["observed_stable"] >= stable_min for values in stable_counts.values())
        and all(values["fraction"] >= fall_fraction_min for values in fall_counts.values())
        and not pretransition_falls
        and not missing_transitions
        and not nonfinite
    )
    return {
        "passed": passed,
        "stable_counts": stable_counts,
        "fall_counts": fall_counts,
        "pretransition_falls": pretransition_falls,
        "missing_transitions": missing_transitions,
        "nonfinite_runtime_runs": nonfinite,
    }


def _oracle_gate(
    runs: Mapping[str, IntegratedRun],
    acceptance: Mapping[str, object],
) -> dict[str, object]:
    stable_runs = [
        run
        for run in runs.values()
        if run.specification["intended_role"] == "stable"
        and run.simulation.metadata["first_fall_sample"] is None
    ]
    fall_runs = [
        run
        for run in runs.values()
        if run.simulation.metadata["first_fall_sample"] is not None
    ]
    stable_firing = [
        run.run_id for run in stable_runs if _first_true(run.instability.onset) is not None
    ]
    detected_fall: list[str] = []
    lead_ms: list[float] = []
    detection_by_terrain = {"ice": [0, 0], "sand": [0, 0]}
    for run in fall_runs:
        terrain = str(run.specification["terrain"])
        if terrain in detection_by_terrain:
            detection_by_terrain[terrain][1] += 1
        t_instability = _first_true(run.instability.onset)
        t_fall = int(run.simulation.metadata["first_fall_sample"])
        if t_instability is not None and t_instability < t_fall:
            detected_fall.append(run.run_id)
            lead_ms.append(float(t_fall - t_instability))
            if terrain in detection_by_terrain:
                detection_by_terrain[terrain][0] += 1
    stable_rate = len(stable_firing) / len(stable_runs) if stable_runs else 1.0
    fall_rate = len(detected_fall) / len(fall_runs) if fall_runs else 0.0
    median_lead = float(np.median(lead_ms)) if lead_ms else None
    each_terrain = all(
        total > 0 and detected > 0
        for detected, total in detection_by_terrain.values()
    )
    passed = bool(
        stable_rate <= float(acceptance["stable_run_firing_rate_max"])
        and fall_rate >= float(acceptance["fall_run_detection_rate_min"])
        and each_terrain
        and median_lead is not None
        and median_lead
        >= float(acceptance["median_instability_to_fall_lead_ms_min"])
    )
    return {
        "passed": passed,
        "stable_runs": len(stable_runs),
        "stable_firing_runs": stable_firing,
        "stable_run_firing_rate": stable_rate,
        "fall_runs": len(fall_runs),
        "detected_fall_runs": detected_fall,
        "fall_detection_rate": fall_rate,
        "detection_by_terrain": {
            terrain: {"detected": values[0], "fall_runs": values[1]}
            for terrain, values in detection_by_terrain.items()
        },
        "instability_to_fall_lead_ms": {
            "median": median_lead,
            "p95": float(np.percentile(lead_ms, 95)) if lead_ms else None,
            "minimum": min(lead_ms) if lead_ms else None,
        },
    }


def _labels_for_run(run: IntegratedRun, exclusion_samples: int) -> tuple[np.ndarray, np.ndarray]:
    samples = len(run.simulation.runtime.timestamp_us)
    labels = np.zeros(samples, dtype=np.int64)
    eligible = np.ones(samples, dtype=bool)
    fall = run.simulation.metadata["first_fall_sample"]
    if fall is not None:
        eligible[int(fall) :] = False
    onset = _first_true(run.instability.onset)
    if onset is not None:
        lower = max(0, onset - exclusion_samples)
        upper = min(samples, onset + exclusion_samples + 1)
        eligible[lower:upper] = False
        labels[upper:] = 1
    return labels, eligible


def _normalizer(
    runs: Mapping[str, IntegratedRun], train_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, int]:
    values: list[np.ndarray] = []
    for run_id in train_ids:
        run = runs[run_id]
        fall = run.simulation.metadata["first_fall_sample"]
        stop = len(run.simulation.runtime.pelvis_imu) if fall is None else int(fall)
        values.append(run.simulation.runtime.pelvis_imu[:stop])
    combined = np.concatenate(values).astype(np.float64)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std < 1.0e-8] = 1.0
    return mean, std, len(combined)


def _window_set(
    runs: Mapping[str, IntegratedRun],
    run_ids: Sequence[str],
    window_samples: int,
    stride_samples: int,
    exclusion_samples: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> WindowSet:
    inputs: list[np.ndarray] = []
    targets: list[int] = []
    sources: list[str] = []
    endpoints: list[int] = []
    for run_id in run_ids:
        run = runs[run_id]
        normalized = (
            (run.simulation.runtime.pelvis_imu.astype(np.float64) - mean) / std
        ).astype(np.float32)
        labels, eligible = _labels_for_run(run, exclusion_samples)
        for endpoint in range(window_samples - 1, len(normalized), stride_samples):
            start = endpoint - window_samples + 1
            if not np.all(eligible[start : endpoint + 1]):
                continue
            window_labels = labels[start : endpoint + 1]
            if not np.all(window_labels == window_labels[-1]):
                continue
            inputs.append(normalized[start : endpoint + 1])
            targets.append(int(window_labels[-1]))
            sources.append(run_id)
            endpoints.append(endpoint)
    if not inputs:
        raise ValueError("no causal AI windows were materialized")
    target_array = np.asarray(targets, dtype=np.int64)
    available = np.bincount(target_array, minlength=3)
    return WindowSet(
        inputs=np.stack(inputs),
        targets=target_array,
        run_ids=np.asarray(sources, dtype=str),
        endpoint_samples=np.asarray(endpoints, dtype=np.int64),
        available_by_class=tuple(int(value) for value in available[:3]),
    )


def _replay_model(
    model: torch.nn.Module,
    imu: np.ndarray,
    window_samples: int,
    mean: np.ndarray,
    std: np.ndarray,
    sustained_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = ((imu.astype(np.float64) - mean) / std).astype(np.float32)
    prediction = np.zeros(len(normalized), dtype=np.int64)
    model.eval()
    endpoints = list(range(window_samples - 1, len(normalized)))
    with torch.no_grad():
        for offset in range(0, len(endpoints), 256):
            batch_endpoints = endpoints[offset : offset + 256]
            batch = np.stack(
                [
                    normalized[endpoint - window_samples + 1 : endpoint + 1]
                    for endpoint in batch_endpoints
                ]
            )
            output = model(torch.from_numpy(batch)).argmax(dim=1).cpu().numpy()
            prediction[np.asarray(batch_endpoints)] = output
    active, onset = causal_persistence(prediction == 1, sustained_samples)
    return active, onset


def _detector_metrics(
    runs: Mapping[str, IntegratedRun],
    run_ids: Sequence[str],
    onset_by_run: Mapping[str, np.ndarray],
    active_by_run: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    stable_ids = [
        run_id
        for run_id in run_ids
        if runs[run_id].specification["intended_role"] == "stable"
        and runs[run_id].simulation.metadata["first_fall_sample"] is None
    ]
    positive_ids = [
        run_id
        for run_id in run_ids
        if runs[run_id].simulation.metadata["first_fall_sample"] is not None
        and _first_true(runs[run_id].instability.onset) is not None
        and _first_true(runs[run_id].instability.onset)
        < int(runs[run_id].simulation.metadata["first_fall_sample"])
    ]
    stable_fp = [run_id for run_id in stable_ids if _first_true(onset_by_run[run_id]) is not None]
    latencies: list[float] = []
    leads: list[float] = []
    detected: list[str] = []
    preinstability_fp: list[str] = []
    per_terrain = {"ice": [0, 0], "sand": [0, 0]}
    detections: dict[str, dict[str, float]] = {}
    for run_id in positive_ids:
        run = runs[run_id]
        t_instability = _first_true(run.instability.onset)
        assert t_instability is not None
        t_fall = int(run.simulation.metadata["first_fall_sample"])
        onsets = np.flatnonzero(onset_by_run[run_id])
        if np.any(onsets < t_instability):
            preinstability_fp.append(run_id)
        terrain = str(run.specification["terrain"])
        if terrain in per_terrain:
            per_terrain[terrain][1] += 1
        valid = onsets[(onsets >= t_instability) & (onsets < t_fall)]
        if valid.size:
            detection = int(valid[0])
            detected.append(run_id)
            latencies.append(float(detection - t_instability))
            leads.append(float(t_fall - detection))
            detections[run_id] = {
                "latency_ms": float(detection - t_instability),
                "lead_before_fall_ms": float(t_fall - detection),
            }
            if terrain in per_terrain:
                per_terrain[terrain][0] += 1
    return {
        "stable_runs": len(stable_ids),
        "stable_false_positive_runs": stable_fp,
        "stable_false_positive_run_rate": len(stable_fp) / len(stable_ids) if stable_ids else None,
        "stable_false_positive_duration_ms": {
            run_id: int(np.count_nonzero(active_by_run[run_id]))
            for run_id in stable_fp
        }
        if active_by_run is not None
        else None,
        "positive_runs": len(positive_ids),
        "detected_runs": detected,
        "detections": detections,
        "recall": len(detected) / len(positive_ids) if positive_ids else 0.0,
        "recall_at_0_ms": sum(value <= 0 for value in latencies) / len(positive_ids) if positive_ids else 0.0,
        "recall_at_10_ms": sum(value <= 10 for value in latencies) / len(positive_ids) if positive_ids else 0.0,
        "recall_at_20_ms": sum(value <= 20 for value in latencies) / len(positive_ids) if positive_ids else 0.0,
        "recall_at_50_ms": sum(value <= 50 for value in latencies) / len(positive_ids) if positive_ids else 0.0,
        "recall_at_100_ms": sum(value <= 100 for value in latencies) / len(positive_ids) if positive_ids else 0.0,
        "latency_ms": {
            "median": float(np.median(latencies)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
        },
        "lead_before_fall_ms": {
            "median": float(np.median(leads)) if leads else None,
            "p05": float(np.percentile(leads, 5)) if leads else None,
        },
        "preinstability_false_positive_runs": preinstability_fp,
        "per_terrain": {
            terrain: {"detected": values[0], "positive_runs": values[1]}
            for terrain, values in per_terrain.items()
        },
    }


def _train_ai(
    config: Mapping[str, object],
    runs: Mapping[str, IntegratedRun],
    artifact_path: Path,
    progress: Callable[[str], None],
) -> tuple[dict[str, object], Mapping[str, np.ndarray], np.ndarray, np.ndarray]:
    split_config = config["split"]
    validation_ids = tuple(str(value) for value in split_config["validation"])
    holdout_ids = tuple(str(value) for value in split_config["holdout"])
    excluded = set(validation_ids) | set(holdout_ids)
    train_ids = tuple(
        run_id
        for run_id, run in runs.items()
        if run.specification["terrain"] in {"ice", "sand"} and run_id not in excluded
    )
    if set(train_ids) & set(validation_ids) or set(train_ids) & set(holdout_ids) or set(validation_ids) & set(holdout_ids):
        raise ValueError("AI run split is not disjoint")
    mean, std, normalization_samples = _normalizer(runs, train_ids)
    _write_json(
        artifact_path / "ai_normalization.json",
        {
            "method": "train_only_per_channel_zscore",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "sample_count": normalization_samples,
            "fit_run_ids": list(train_ids),
        },
    )
    exclusion = int(config["ambiguous_boundary_exclusion_ms"])
    stride = int(config["stride_ms"])
    seeds = [int(value) for value in config["seeds"]]
    models: dict[tuple[int, int], torch.nn.Module] = {}
    candidates: list[dict[str, object]] = []
    for window in config["window_candidates_ms"]:
        window_samples = int(window)
        train_windows = _window_set(
            runs, train_ids, window_samples, stride, exclusion, mean, std
        )
        validation_windows = _window_set(
            runs, validation_ids, window_samples, stride, exclusion, mean, std
        )
        seed_results: list[dict[str, object]] = []
        for seed in seeds:
            model, training = train_model(
                "gru",
                window_samples,
                train_windows,
                validation_windows,
                seed,
                batch_size=int(config["batch_size"]),
                max_epochs=int(config["max_epochs"]),
                patience=int(config["early_stopping_patience"]),
                learning_rate=float(config["learning_rate"]),
                class_names=STABILITY_CLASS_NAMES,
            )
            models[(window_samples, seed)] = model
            checkpoint = artifact_path / "checkpoints" / f"gru_{window_samples}ms_seed_{seed}.pt"
            save_checkpoint(
                checkpoint,
                model,
                "gru",
                window_samples,
                seed,
                training,
                class_names=STABILITY_CLASS_NAMES,
            )
            validation = evaluate_model(
                model,
                validation_windows,
                class_names=STABILITY_CLASS_NAMES,
            )
            seed_results.append(
                {
                    "seed": seed,
                    "best_epoch": training.best_epoch,
                    "validation": validation,
                }
            )
        candidate = {
            "window_ms": window_samples,
            "parameter_count": parameter_count(models[(window_samples, seeds[0])]),
            "validation_macro_f1_mean": float(
                np.mean([item["validation"]["macro_f1"] for item in seed_results])
            ),
            "seeds": seed_results,
            "train_window_count": len(train_windows),
            "validation_window_count": len(validation_windows),
        }
        candidates.append(candidate)
        progress(
            f"AI GRU {window_samples} ms validation macro-F1 "
            f"{candidate['validation_macro_f1_mean']:.4f}"
        )
    selected = max(
        candidates,
        key=lambda item: (item["validation_macro_f1_mean"], -item["window_ms"]),
    )
    selected_window = int(selected["window_ms"])
    holdout_seed = int(config["holdout_seed"])
    selected_model = models[(selected_window, holdout_seed)]
    holdout_windows = _window_set(
        runs,
        holdout_ids,
        selected_window,
        stride,
        exclusion,
        mean,
        std,
    )
    holdout_classification = evaluate_model(
        selected_model,
        holdout_windows,
        class_names=STABILITY_CLASS_NAMES,
    )
    ai_onset: dict[str, np.ndarray] = {}
    ai_active: dict[str, np.ndarray] = {}
    sustained = int(config["sustained_prediction_ms"])
    for run_id in runs:
        active, onset = _replay_model(
            selected_model,
            runs[run_id].simulation.runtime.pelvis_imu,
            selected_window,
            mean,
            std,
            sustained,
        )
        ai_active[run_id] = active
        ai_onset[run_id] = onset
    replay_metrics = _detector_metrics(
        runs, holdout_ids, ai_onset, ai_active
    )
    metrics = {
        "performed": True,
        "split": {
            "train": list(train_ids),
            "validation": list(validation_ids),
            "holdout": list(holdout_ids),
        },
        "candidates": candidates,
        "selected": {
            "family": "gru",
            "window_ms": selected_window,
            "seed": holdout_seed,
            "parameter_count": parameter_count(selected_model),
        },
        "holdout_classification": holdout_classification,
        "holdout_replay": replay_metrics,
    }
    return metrics, ai_onset, mean, std


def _run_row(run: IntegratedRun) -> dict[str, object]:
    result = run.simulation
    terrain = str(run.specification["terrain"])
    transition = _transition_sample(run, terrain)
    t_instability = _first_true(run.instability.onset)
    t_rule = _first_true(run.rule_onset)
    fall = result.metadata["first_fall_sample"]
    slip = _first_true(result.diagnostics.any_established_slip_after_patch_onset)
    sink = _first_true_any(result.diagnostics.deformable_sink_onset)
    return {
        "run_id": run.run_id,
        "group": run.specification["group"],
        "terrain": terrain,
        "speed_mps": run.specification["speed_mps"],
        "intended_role": run.specification["intended_role"],
        "observed_fall": fall is not None,
        "transition_ms": _time_ms(result, transition),
        "t_instability_ms": _time_ms(result, t_instability),
        "t_fall_ms": _time_ms(result, None if fall is None else int(fall)),
        "instability_to_fall_lead_ms": (
            None
            if fall is None or t_instability is None
            else float(int(fall) - t_instability)
        ),
        "physical_slip_diagnostic_ms": _time_ms(result, slip),
        "deformable_sink_diagnostic_ms": _time_ms(result, sink),
        "ground_truth_supported": (
            t_instability is None if fall is None else t_instability is not None and t_instability < int(fall)
        ),
        "rule_detected": t_rule is not None,
        "t_rule_detect_ms": _time_ms(result, t_rule),
        "rule_latency_ms": (
            None if t_instability is None or t_rule is None else float(t_rule - t_instability)
        ),
        "first_fall_reasons": list(result.metadata["first_fall_reasons"]),
        "no_support_samples": int(np.count_nonzero(result.stability.gait_phase == 0)) if result.stability is not None else None,
    }


def _fusion_metrics(
    runs: Mapping[str, IntegratedRun],
    ai_onset: Mapping[str, np.ndarray] | None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for run in runs.values():
        terrain = str(run.specification["terrain"])
        transition = _transition_sample(run, terrain)
        target_valid = transition
        if target_valid is None:
            target_valid = _first_true(np.any(run.simulation.diagnostics.touchdown, axis=1))
        t_instability = _first_true(run.instability.onset)
        rule_onsets = np.flatnonzero(run.rule_onset)
        valid_rule = rule_onsets if t_instability is None else rule_onsets[rule_onsets >= t_instability]
        rule_detect = None if not valid_rule.size else int(valid_rule[0])
        fusion_ready = (
            None
            if target_valid is None or rule_detect is None
            else max(target_valid, rule_detect)
        )
        fall = run.simulation.metadata["first_fall_sample"]
        oracle_supported = (
            t_instability is None
            if fall is None
            else t_instability is not None and t_instability < int(fall)
        )
        fusion_measurable = bool(
            oracle_supported
            and t_instability is not None
            and fusion_ready is not None
            and (fall is None or fusion_ready < int(fall))
        )
        recovery_measurable = bool(
            oracle_supported
            and t_instability is not None
            and rule_detect is not None
            and (fall is None or rule_detect < int(fall))
        )
        rows.append(
            {
                "run_id": run.run_id,
                "terrain_state": terrain.upper(),
                "terrain_source": "ORACLE_PROXY",
                "terrain_valid_sample": target_valid,
                "terrain_proxy_latency_ms": 0.0 if target_valid is not None else None,
                "rule_detect_sample": rule_detect,
                "fusion_ready_sample": fusion_ready,
                "fusion_latency_ms": float(fusion_ready - t_instability) if fusion_measurable else None,
                "recovery_required_latency_ms": float(rule_detect - t_instability) if recovery_measurable else None,
                "terrain_already_valid_at_detection": target_valid is not None and rule_detect is not None and target_valid <= rule_detect,
            }
        )
    measurable = [row["fusion_latency_ms"] for row in rows if row["fusion_latency_ms"] is not None]
    recovery = [row["recovery_required_latency_ms"] for row in rows if row["recovery_required_latency_ms"] is not None]
    return {
        "truth_table": {
            "ICE+STABLE": "NORMAL",
            "ICE+UNSTABLE": "SLIP_RISK",
            "SAND+STABLE": "NORMAL",
            "SAND+UNSTABLE": "SINK_RISK",
            "CONCRETE/MARBLE+UNSTABLE": "GENERIC_INSTABILITY",
            "UNKNOWN+UNSTABLE": "GENERIC_INSTABILITY",
            "UNKNOWN+UNSTABLE_RECOVERY_REQUIRED": True,
        },
        "runs": rows,
        "fusion_latency_ms": {
            "median": float(np.median(measurable)) if measurable else None,
            "p95": float(np.percentile(measurable, 95)) if measurable else None,
        },
        "recovery_required_latency_ms": {
            "median": float(np.median(recovery)) if recovery else None,
            "p95": float(np.percentile(recovery, 95)) if recovery else None,
        },
        "terrain_already_valid_count": sum(bool(row["terrain_already_valid_at_detection"]) for row in rows),
        "terrain_not_yet_valid_count": sum(row["rule_detect_sample"] is not None and not bool(row["terrain_already_valid_at_detection"]) for row in rows),
    }


def _status_replay(
    run: IntegratedRun,
    ai_onset: np.ndarray | None,
) -> str:
    result = run.simulation
    terrain = str(run.specification["terrain"])
    transition = _transition_sample(run, terrain)
    first_touchdown = _first_true(np.any(result.diagnostics.touchdown, axis=1))
    t_instability = _first_true(run.instability.onset)
    t_rule = _first_true(run.rule_onset)
    t_ai = None if ai_onset is None else _first_true(ai_onset)
    fall = result.metadata["first_fall_sample"]
    candidates = [value for value in (transition, t_instability, t_rule, t_ai, fall) if value is not None]
    sample = int(candidates[0]) if candidates else len(result.runtime.timestamp_us) - 1
    if t_rule is not None:
        sample = t_rule
    elif t_instability is not None:
        sample = t_instability
    runtime_terrain = _terrain_state(terrain)
    if terrain in {"ice", "sand"} and (transition is None or sample < transition):
        runtime_terrain = TerrainState.CONCRETE
    terrain_valid_sample = transition if terrain in {"ice", "sand"} else first_touchdown
    terrain_valid = terrain_valid_sample is not None and sample >= terrain_valid_sample
    stability_state = StabilityState.UNSTABLE if run.rule_active[sample] else StabilityState.STABLE
    state = ParallelRuntimeState(
        terrain_state=runtime_terrain,
        terrain_valid=terrain_valid,
        terrain_updated_at_us=_time_us(result, terrain_valid_sample),
        stability_state=stability_state,
        stability_valid=True,
        stability_updated_at_us=int(result.runtime.timestamp_us[sample]),
    ).update_stability(stability_state, int(result.runtime.timestamp_us[sample]))
    true_terrain = runtime_terrain
    gt = StabilityState.UNSTABLE if run.instability.active[sample] else StabilityState.STABLE
    ai_state = None
    if ai_onset is not None:
        ai_active, _ = causal_persistence(np.cumsum(ai_onset) > 0, 1)
        ai_state = StabilityState.UNSTABLE if ai_active[sample] else StabilityState.STABLE
    return f"RUN {run.run_id}\n" + format_runtime_status(
        true_terrain=true_terrain,
        state=state,
        stability_gt=gt,
        stability_ai=ai_state,
        timestamp_us=int(result.runtime.timestamp_us[sample]),
        event_times_us={
            "transition": _time_us(result, transition),
            "t_instability": _time_us(result, t_instability),
            "t_rule_detect": _time_us(result, t_rule),
            "t_ai_detect": _time_us(result, t_ai),
            "t_fall": _time_us(result, None if fall is None else int(fall)),
        },
    )


def render_simulation_status(
    result: SimulationResult,
    calibration_path: Path,
    run_id: str = "simulate",
) -> str:
    """Render one canonical ``simulate`` result using frozen calibration."""
    if result.stability is None:
        raise ValueError("simulation result has no exact stability diagnostics")
    with calibration_path.open("r", encoding="utf-8") as stream:
        calibration = json.load(stream)
    phase = calibration["phase_envelope"]
    envelope = PhaseEnvelope(
        lower_bound_m={int(key): float(value) for key, value in phase["lower_bound_m"].items()},
        quantile=float(phase["quantile"]),
        calibration_run_ids=tuple(phase["calibration_run_ids"]),
    )
    instability = detect_instability(
        result.stability,
        envelope,
        float(phase["fixed_margin_m"]),
        int(phase["persistence_ms"]),
    )
    rule = calibration["imu_rule"]
    rule_trace = run_imu_rule(
        result.runtime.pelvis_imu,
        IMURuleCalibration(
            acceleration_norm_center_m_s2=float(
                rule["acceleration_norm_center_m_s2"]
            ),
            thresholds=np.asarray(rule["thresholds"], dtype=np.float64),
            quantile=float(rule["quantile"]),
            calibration_run_ids=tuple(rule["calibration_run_ids"]),
        ),
        int(rule["persistence_ms"]),
        int(rule["stable_reset_ms"]),
    )
    integrated = IntegratedRun(
        specification={
            "id": run_id,
            "terrain": result.metadata["terrain"],
            "intended_role": "viewer",
            "group": "viewer",
        },
        simulation=result,
        instability=instability,
        rule_onset=rule_trace.onset,
        rule_active=rule_trace.active,
    )
    return _status_replay(integrated, None)


def run_integrated_stability_sanity(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Run the frozen matrix, gates, optional GRU, fusion, and status replay."""
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if document["experiment"]["id"] != "TERRAIN_STABILITY_INTEGRATED_SANITY":
        raise ValueError("unsupported integrated stability experiment")
    specifications = document["runs"]
    run_ids = [str(item["id"]) for item in specifications]
    if not 40 <= len(run_ids) <= 60 or len(run_ids) != len(set(run_ids)):
        raise ValueError("integrated matrix must contain 40-60 unique runs")
    artifact_path = (repository_root / document["artifacts"]["path"]).resolve()
    artifact_path.relative_to(repository_root)
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite experiment artifacts: {artifact_path}")
    artifact_path.mkdir(parents=True, exist_ok=True)
    base = load_simulation_config(
        (repository_root / document["source"]["simulator_config"]).resolve()
    )
    policy_path = repository_root / "artifacts" / "external" / "unitree_g1" / "g1_velocity_policy.onnx"
    if not policy_path.is_file():
        raise FileNotFoundError(f"verified G1 policy is unavailable: {policy_path}")
    duration = float(document["common"]["duration_s"])
    simulations: dict[str, SimulationResult] = {}
    for index, specification in enumerate(specifications, start=1):
        run_id = str(specification["id"])
        simulations[run_id] = run_simulation(
            _simulation_config(base, specification, policy_path, duration)
        )
        progress(
            f"simulation {index}/{len(specifications)} {run_id}: "
            f"fall={simulations[run_id].metadata['first_fall_sample'] is not None}"
        )
    scenario_gate = _scenario_gate(
        specifications, simulations, document["acceptance"]["scenario"]
    )

    oracle_config = document["stability_oracle"]
    calibration_runs: list[StableCalibrationRun] = []
    for run_id in oracle_config["stable_calibration_run_ids"]:
        result = simulations[str(run_id)]
        if result.stability is None:
            raise RuntimeError("simulation did not capture exact stability state")
        calibration_runs.append(
            StableCalibrationRun(
                run_id=str(run_id),
                diagnostics=result.stability,
                intended_stable=True,
                observed_fall=result.metadata["first_fall_sample"] is not None,
            )
        )
    envelope = fit_phase_envelope(
        calibration_runs, float(oracle_config["phase_lower_quantile"])
    )
    rule_config = document["imu_rule"]
    rule_calibration = fit_imu_rule(
        {
            str(run_id): simulations[str(run_id)].runtime.pelvis_imu
            for run_id in rule_config["stable_calibration_run_ids"]
        },
        float(rule_config["threshold_quantile"]),
    )
    runs: dict[str, IntegratedRun] = {}
    for specification in specifications:
        run_id = str(specification["id"])
        result = simulations[run_id]
        assert result.stability is not None
        instability = detect_instability(
            result.stability,
            envelope,
            float(oracle_config["fixed_margin_m"]),
            int(oracle_config["persistence_ms"]),
        )
        rule = run_imu_rule(
            result.runtime.pelvis_imu,
            rule_calibration,
            int(rule_config["persistence_ms"]),
            int(rule_config["stable_reset_ms"]),
        )
        runs[run_id] = IntegratedRun(
            specification=specification,
            simulation=result,
            instability=instability,
            rule_onset=rule.onset,
            rule_active=rule.active,
        )
    oracle_gate = _oracle_gate(
        runs, document["acceptance"]["stability_ground_truth"]
    )
    holdout_ids = tuple(
        str(value) for value in document["ai_baseline"]["split"]["holdout"]
    )
    rule_metrics = _detector_metrics(
        runs,
        holdout_ids,
        {run_id: run.rule_onset for run_id, run in runs.items()},
        {run_id: run.rule_active for run_id, run in runs.items()},
    )
    _write_json(
        artifact_path / "calibration.json",
        {
            "phase_envelope": {
                "quantile": envelope.quantile,
                "lower_bound_m": {
                    str(phase): value for phase, value in envelope.lower_bound_m.items()
                },
                "calibration_run_ids": list(envelope.calibration_run_ids),
                "fixed_margin_m": oracle_config["fixed_margin_m"],
                "persistence_ms": oracle_config["persistence_ms"],
            },
            "imu_rule": {
                "quantile": rule_calibration.quantile,
                "acceleration_norm_center_m_s2": rule_calibration.acceleration_norm_center_m_s2,
                "thresholds": rule_calibration.thresholds.tolist(),
                "calibration_run_ids": list(rule_calibration.calibration_run_ids),
                "persistence_ms": rule_config["persistence_ms"],
                "stable_reset_ms": rule_config["stable_reset_ms"],
            },
        },
    )

    ai_metrics: dict[str, object] = {
        "performed": False,
        "reason": "scenario_or_stability_ground_truth_gate_failed",
    }
    ai_onsets: Mapping[str, np.ndarray] | None = None
    if scenario_gate["passed"] and oracle_gate["passed"]:
        ai_metrics, ai_onsets, _, _ = _train_ai(
            document["ai_baseline"], runs, artifact_path, progress
        )
    fusion = _fusion_metrics(runs, ai_onsets)
    if not scenario_gate["passed"]:
        verdict = "INTEGRATED_SCENARIO_NEEDS_REVISION"
    elif not oracle_gate["passed"]:
        verdict = "STABILITY_GROUND_TRUTH_NEEDS_REVISION"
    elif ai_metrics["performed"]:
        rule_recall = float(rule_metrics["recall"])
        ai_recall = float(ai_metrics["holdout_replay"]["recall"])
        rule_fp = float(rule_metrics["stable_false_positive_run_rate"] or 0.0)
        ai_fp = float(ai_metrics["holdout_replay"]["stable_false_positive_run_rate"] or 0.0)
        meaningful = ai_recall >= rule_recall + 0.10 and ai_fp <= rule_fp
        verdict = (
            "TERRAIN_STABILITY_INTEGRATION_READY_AI_PROMISING"
            if meaningful
            else "TERRAIN_STABILITY_INTEGRATION_READY"
        )
    else:
        verdict = "TERRAIN_STABILITY_INTEGRATION_READY"
    viewer_ids = [str(value) for value in document["viewer"]["representative_run_ids"]]
    status_blocks = [
        _status_replay(
            runs[run_id], None if ai_onsets is None else ai_onsets[run_id]
        )
        for run_id in viewer_ids
    ]
    status_text = "\n\n".join(status_blocks) + "\n"
    with (artifact_path / "viewer_status.txt").open("w", encoding="utf-8") as stream:
        stream.write(status_text)
    progress(status_text)
    metrics: dict[str, object] = {
        "experiment": document["experiment"],
        "terrain_runtime": document["terrain_runtime"],
        "scenario_gate": scenario_gate,
        "phase_envelope": {
            "quantile": envelope.quantile,
            "lower_bound_m": {
                str(phase): value for phase, value in envelope.lower_bound_m.items()
            },
            "calibration_run_ids": list(envelope.calibration_run_ids),
        },
        "oracle_gate": oracle_gate,
        "imu_rule_calibration": {
            "acceleration_norm_center_m_s2": rule_calibration.acceleration_norm_center_m_s2,
            "thresholds": rule_calibration.thresholds.tolist(),
            "quantile": rule_calibration.quantile,
            "calibration_run_ids": list(rule_calibration.calibration_run_ids),
        },
        "rule_holdout": rule_metrics,
        "ai": ai_metrics,
        "fusion": fusion,
        "runs": [_run_row(runs[run_id]) for run_id in run_ids],
        "viewer": {
            "implementation": document["viewer"]["implementation"],
            "representative_run_ids": viewer_ids,
            "physics_mutation": False,
        },
        "verdict": verdict,
    }
    _write_json(artifact_path / "results.json", metrics)
    return artifact_path, metrics
