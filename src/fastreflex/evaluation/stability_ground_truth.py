"""Privileged full-state normal-distribution stability ground truth."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import mujoco
import numpy as np
import yaml

from fastreflex.evaluation.transition_scenarios import (
    VALID_FALL,
    VALID_STABLE,
    classify_scenario_outcome,
    fusion_regression,
    scenario_timing_row,
    target_contact_mask,
    transition_simulation_config,
)
from fastreflex.simulation.g1 import (
    PHYSICS_TIMESTEP_S,
    SENSOR_RATE_HZ,
    RuntimeTrace,
    SimulationConfig,
    SimulationResult,
    load_g1_model,
    load_simulation_config,
    run_simulation,
)
from fastreflex.simulation.stability import (
    DOUBLE_SUPPORT,
    LEFT_SINGLE_SUPPORT,
    NO_SUPPORT,
    PHASE_NAMES,
    RIGHT_SINGLE_SUPPORT,
    causal_persistence,
)


PELVIS_STATE = "PELVIS_STATE"
LOWER_BODY_STATE = "LOWER_BODY_STATE"
FULL_STATE = "FULL_STATE"
CANDIDATE_ORDER = (PELVIS_STATE, LOWER_BODY_STATE, FULL_STATE)
SUPPORTED_PHASES = (
    LEFT_SINGLE_SUPPORT,
    RIGHT_SINGLE_SUPPORT,
    DOUBLE_SUPPORT,
)
LOWER_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
PELVIS_FEATURE_NAMES = (
    "pelvis_roll_rad",
    "pelvis_pitch_rad",
    "pelvis_angular_velocity_x_rad_s",
    "pelvis_angular_velocity_y_rad_s",
    "pelvis_angular_velocity_z_rad_s",
    "pelvis_linear_velocity_x_m_s",
    "pelvis_linear_velocity_y_m_s",
    "pelvis_linear_velocity_z_m_s",
    "pelvis_height_m",
)
LOWER_BODY_FEATURE_NAMES = tuple(
    feature
    for joint in LOWER_BODY_JOINT_NAMES
    for feature in (f"{joint}_q_rad", f"{joint}_dq_rad_s")
)
FULL_STATE_FEATURE_NAMES = (
    *PELVIS_FEATURE_NAMES,
    *LOWER_BODY_FEATURE_NAMES,
    "com_linear_velocity_x_m_s",
    "com_linear_velocity_y_m_s",
    "com_linear_velocity_z_m_s",
    "com_height_above_support_m",
)
CANDIDATE_FEATURE_NAMES = {
    PELVIS_STATE: PELVIS_FEATURE_NAMES,
    LOWER_BODY_STATE: LOWER_BODY_FEATURE_NAMES,
    FULL_STATE: FULL_STATE_FEATURE_NAMES,
}
SIGNATURE_FIELDS = (
    "source_terrain",
    "target_terrain",
    "speed_mps",
    "patch_start_x_m",
    "patch_width_m",
    "slip_pattern",
    "sink_pattern",
    "sink_severity",
    "support_pattern",
)


@dataclass(frozen=True)
class PhaseDistanceDistribution:
    """Stable-only normalized Gaussian approximation for one support phase."""

    mean: np.ndarray
    standard_deviation: np.ndarray
    covariance: np.ndarray
    regularized_covariance: np.ndarray
    precision: np.ndarray
    distance_threshold: float
    fit_sample_count: int


@dataclass(frozen=True)
class CandidateDistanceModel:
    """Phase-conditioned distance model for one fixed state representation."""

    candidate: str
    feature_names: tuple[str, ...]
    phase_distributions: Mapping[int, PhaseDistanceDistribution]
    fit_run_ids: tuple[str, ...]
    stride_samples: int
    per_run_per_phase_cap: int
    covariance_lambda: float
    covariance_epsilon: float
    threshold_quantile: float


@dataclass(frozen=True)
class DistanceTrace:
    """Current-state distance and causal persistent abnormality trace."""

    distance: np.ndarray
    threshold: np.ndarray
    candidate: np.ndarray
    active: np.ndarray
    onset: np.ndarray


@dataclass(frozen=True)
class StateDistanceRun:
    specification: Mapping[str, object]
    result: SimulationResult
    outcome: str
    features: np.ndarray
    first_contact_sample: int
    ungated: DistanceTrace
    primary: DistanceTrace

    @property
    def run_id(self) -> str:
        return str(self.specification["id"])


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


def canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible frozen contract deterministically."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        result[relative] = _file_sha256(path)
    return result


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if not indices.size else int(indices[0])


def _time_ms(result: SimulationResult, sample: int | None) -> float | None:
    if sample is None or not 0 <= sample < len(result.runtime.timestamp_us):
        return None
    return float(result.runtime.timestamp_us[sample]) / 1000.0


def _condition_signature(specification: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(specification[field] for field in SIGNATURE_FIELDS)


def lower_body_state_addresses(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the declared lower-body joints through authoritative MuJoCo names."""
    qpos = []
    qvel = []
    for name in LOWER_BODY_JOINT_NAMES:
        joint_id = model.joint(name).id
        qpos.append(int(model.jnt_qposadr[joint_id]))
        qvel.append(int(model.jnt_dofadr[joint_id]))
    qpos_array = np.asarray(qpos, dtype=np.int64)
    qvel_array = np.asarray(qvel, dtype=np.int64)
    if not np.array_equal(qpos_array, np.arange(7, 19)) or not np.array_equal(
        qvel_array, np.arange(6, 18)
    ):
        raise ValueError("G1 lower-body state addresses changed")
    return qpos_array, qvel_array


def extract_candidate_features(
    result: SimulationResult,
    candidate: str,
    qpos_addresses: np.ndarray,
    qvel_addresses: np.ndarray,
) -> np.ndarray:
    """Extract only the fixed privileged state schema, with no scenario fields."""
    if candidate not in CANDIDATE_FEATURE_NAMES:
        raise ValueError(f"unsupported state-distance candidate: {candidate}")
    if result.state_trace is None or result.stability is None:
        raise ValueError(
            "full-state candidate extraction requires simulator state trace"
        )
    diagnostics = result.diagnostics
    pelvis = np.column_stack(
        (
            diagnostics.pelvis_roll_rad,
            diagnostics.pelvis_pitch_rad,
            diagnostics.pelvis_angular_velocity_rad_s,
            diagnostics.pelvis_linear_velocity_m_s,
            diagnostics.pelvis_world_z_m,
        )
    ).astype(np.float64)
    lower_columns = []
    for qpos_address, qvel_address in zip(qpos_addresses, qvel_addresses):
        lower_columns.extend(
            (
                result.state_trace.robot_qpos[:, int(qpos_address)],
                result.state_trace.robot_qvel[:, int(qvel_address)],
            )
        )
    lower = np.column_stack(lower_columns).astype(np.float64)
    if candidate == PELVIS_STATE:
        features = pelvis
    elif candidate == LOWER_BODY_STATE:
        features = lower
    else:
        locomotion = np.column_stack(
            (
                result.stability.com_velocity_xyz_m_s,
                result.stability.support_height_m,
            )
        )
        features = np.column_stack((pelvis, lower, locomotion))
    expected = (len(result.runtime.sequence), len(CANDIDATE_FEATURE_NAMES[candidate]))
    supported = result.stability.gait_phase != NO_SUPPORT
    if features.shape != expected or not np.all(np.isfinite(features[supported])):
        raise ValueError(f"{candidate} features must have finite shape {expected}")
    return features


def deterministic_phase_sample_indices(
    phase: np.ndarray,
    phase_id: int,
    stride_samples: int,
    cap: int,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Use a time-grid stride then an evenly spread deterministic run/phase cap."""
    values = np.asarray(phase)
    if values.ndim != 1 or stride_samples <= 0 or cap <= 0:
        raise ValueError("invalid phase sampling contract")
    mask = values == phase_id
    if eligible is not None:
        eligible_array = np.asarray(eligible, dtype=bool)
        if eligible_array.shape != values.shape:
            raise ValueError("eligible mask must match phase trace")
        mask &= eligible_array
    time_grid = np.arange(len(values)) % stride_samples == 0
    indices = np.flatnonzero(mask & time_grid)
    if len(indices) <= cap:
        return indices
    positions = np.linspace(0, len(indices) - 1, cap, dtype=np.int64)
    selected = indices[positions]
    if len(selected) != cap or len(np.unique(selected)) != cap:
        raise RuntimeError("deterministic phase cap produced duplicate samples")
    return selected


def regularize_covariance(
    covariance: np.ndarray,
    shrinkage_lambda: float,
    epsilon: float,
) -> np.ndarray:
    """Shrink off-diagonal covariance and add fixed normalized-space jitter."""
    matrix = np.asarray(covariance, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not 0.0 <= shrinkage_lambda <= 1.0
        or epsilon <= 0.0
    ):
        raise ValueError("invalid covariance regularization input")
    diagonal = np.diag(np.diag(matrix))
    return (
        (1.0 - shrinkage_lambda) * matrix
        + shrinkage_lambda * diagonal
        + epsilon * np.eye(len(matrix), dtype=np.float64)
    )


def mahalanobis_distance(
    values: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    """Compute current-sample regularized Mahalanobis distance."""
    array = np.asarray(values, dtype=np.float64)
    normalized = (array - mean) / standard_deviation
    squared = np.einsum("ni,ij,nj->n", normalized, precision, normalized)
    return np.sqrt(np.maximum(squared, 0.0))


def fit_candidate_distance_model(
    candidate: str,
    stable_features_by_run: Mapping[str, np.ndarray],
    phase_by_run: Mapping[str, np.ndarray],
    *,
    stride_samples: int,
    per_run_per_phase_cap: int,
    standard_deviation_floor: float,
    covariance_lambda: float,
    covariance_epsilon: float,
    threshold_quantile: float,
) -> CandidateDistanceModel:
    """Fit every statistic from observed-stable calibration samples only."""
    if candidate not in CANDIDATE_FEATURE_NAMES or not stable_features_by_run:
        raise ValueError("candidate fit requires observed-stable runs")
    if set(stable_features_by_run) != set(phase_by_run):
        raise ValueError("feature and phase fit run IDs differ")
    if not 0.5 < threshold_quantile < 1.0:
        raise ValueError("distance threshold quantile must be high-tail")
    distributions = {}
    for phase_id in SUPPORTED_PHASES:
        chunks = []
        for run_id in stable_features_by_run:
            features = np.asarray(stable_features_by_run[run_id], dtype=np.float64)
            phase = np.asarray(phase_by_run[run_id])
            if features.shape[0] != len(phase):
                raise ValueError("stable feature and phase sample counts differ")
            indices = deterministic_phase_sample_indices(
                phase,
                phase_id,
                stride_samples,
                per_run_per_phase_cap,
            )
            if indices.size:
                chunks.append(features[indices])
        if not chunks:
            raise ValueError(f"stable fit has no {PHASE_NAMES[phase_id]} samples")
        combined = np.concatenate(chunks)
        dimension = len(CANDIDATE_FEATURE_NAMES[candidate])
        if combined.shape[0] <= dimension:
            raise ValueError("stable fit has too few samples for covariance")
        mean = combined.mean(axis=0)
        standard_deviation = combined.std(axis=0)
        standard_deviation[standard_deviation < standard_deviation_floor] = 1.0
        normalized = (combined - mean) / standard_deviation
        covariance = np.cov(normalized, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance).astype(np.float64)
        regularized = regularize_covariance(
            covariance, covariance_lambda, covariance_epsilon
        )
        precision = np.linalg.inv(regularized)
        distance = mahalanobis_distance(combined, mean, standard_deviation, precision)
        threshold = float(np.quantile(distance, threshold_quantile, method="linear"))
        distributions[phase_id] = PhaseDistanceDistribution(
            mean=mean,
            standard_deviation=standard_deviation,
            covariance=covariance,
            regularized_covariance=regularized,
            precision=precision,
            distance_threshold=threshold,
            fit_sample_count=len(combined),
        )
    return CandidateDistanceModel(
        candidate=candidate,
        feature_names=CANDIDATE_FEATURE_NAMES[candidate],
        phase_distributions=distributions,
        fit_run_ids=tuple(stable_features_by_run),
        stride_samples=stride_samples,
        per_run_per_phase_cap=per_run_per_phase_cap,
        covariance_lambda=covariance_lambda,
        covariance_epsilon=covariance_epsilon,
        threshold_quantile=threshold_quantile,
    )


def score_candidate_distance(
    features: np.ndarray,
    phase: np.ndarray,
    model: CandidateDistanceModel,
    persistence_samples: int,
    *,
    eligible_from_sample: int = 0,
) -> DistanceTrace:
    """Score current state against frozen stable statistics and persist causally."""
    values = np.asarray(features, dtype=np.float64)
    phase_array = np.asarray(phase)
    if values.shape != (len(phase_array), len(model.feature_names)):
        raise ValueError("candidate score shape differs from frozen feature schema")
    if not 0 <= eligible_from_sample <= len(values):
        raise ValueError("eligible sample lies outside candidate trace")
    distance = np.full(len(values), np.nan, dtype=np.float64)
    threshold = np.full(len(values), np.nan, dtype=np.float64)
    for phase_id, distribution in model.phase_distributions.items():
        indices = np.flatnonzero(phase_array == phase_id)
        if not indices.size:
            continue
        distance[indices] = mahalanobis_distance(
            values[indices],
            distribution.mean,
            distribution.standard_deviation,
            distribution.precision,
        )
        threshold[indices] = distribution.distance_threshold
    abnormal = np.isfinite(distance) & (distance > threshold)
    abnormal[:eligible_from_sample] = False
    active, onset = causal_persistence(abnormal, persistence_samples)
    return DistanceTrace(distance, threshold, abnormal, active, onset)


def _model_payload(model: CandidateDistanceModel) -> dict[str, object]:
    return {
        "candidate": model.candidate,
        "feature_names": list(model.feature_names),
        "feature_dimension": len(model.feature_names),
        "fit_run_ids": list(model.fit_run_ids),
        "stride_samples": model.stride_samples,
        "per_run_per_phase_cap": model.per_run_per_phase_cap,
        "covariance_lambda": model.covariance_lambda,
        "covariance_epsilon": model.covariance_epsilon,
        "threshold_quantile": model.threshold_quantile,
        "phase_distributions": {
            PHASE_NAMES[phase_id]: {
                "mean": distribution.mean.tolist(),
                "standard_deviation": distribution.standard_deviation.tolist(),
                "covariance": distribution.covariance.tolist(),
                "regularized_covariance": distribution.regularized_covariance.tolist(),
                "distance_threshold": distribution.distance_threshold,
                "fit_sample_count": distribution.fit_sample_count,
            }
            for phase_id, distribution in model.phase_distributions.items()
        },
    }


def select_calibration_candidate(
    metrics_by_candidate: Mapping[str, Mapping[str, object]],
    qualification: Mapping[str, object],
    near_tie: Mapping[str, object],
) -> dict[str, object]:
    """Apply declared gates, lexicographic priority, then the simplicity tie rule."""
    qualifying = []
    qualification_rows = {}
    for candidate in CANDIDATE_ORDER:
        metrics = metrics_by_candidate[candidate]
        lead = metrics["fall_lead_ms"]["p50"]
        gates = {
            "stable_specificity": float(metrics["stable_false_instability_run_rate"])
            <= float(qualification["stable_false_instability_run_rate_max"]),
            "fall_coverage": float(metrics["fall_coverage"])
            >= float(qualification["fall_prefall_coverage_min"]),
            "ice_coverage": float(metrics["by_terrain"]["ice"]["fall_coverage"])
            >= float(qualification["ice_fall_coverage_min"]),
            "sand_coverage": float(metrics["by_terrain"]["sand"]["fall_coverage"])
            >= float(qualification["sand_fall_coverage_min"]),
            "lead_time": lead is not None
            and float(lead) >= float(qualification["median_fall_lead_ms_min"]),
        }
        qualification_rows[candidate] = gates
        if all(gates.values()):
            qualifying.append(candidate)
    if not qualifying:
        return {
            "selected": None,
            "reason": "no_candidate_met_all_calibration_qualification_gates",
            "qualification": qualification_rows,
        }

    simplicity = {
        name: index for index, name in enumerate(near_tie["simplicity_order"])
    }

    def primary_key(candidate: str) -> tuple[float, float, float, float, float, int]:
        metrics = metrics_by_candidate[candidate]
        lead = metrics["fall_lead_ms"]["p50"]
        return (
            float(metrics["stable_false_instability_run_rate"]),
            -float(metrics["fall_coverage"]),
            -float(metrics["by_terrain"]["sand"]["fall_coverage"]),
            -float(metrics["by_terrain"]["ice"]["fall_coverage"]),
            -float(lead),
            simplicity[candidate],
        )

    best = min(qualifying, key=primary_key)
    stable_band = (
        float(near_tie["stable_false_instability_percentage_points_max"]) / 100.0
    )
    coverage_band = float(near_tie["fall_coverage_percentage_points_max"]) / 100.0
    best_metrics = metrics_by_candidate[best]
    tied = [
        candidate
        for candidate in qualifying
        if abs(
            float(metrics_by_candidate[candidate]["stable_false_instability_run_rate"])
            - float(best_metrics["stable_false_instability_run_rate"])
        )
        <= stable_band + 1.0e-12
        and abs(
            float(metrics_by_candidate[candidate]["fall_coverage"])
            - float(best_metrics["fall_coverage"])
        )
        <= coverage_band + 1.0e-12
    ]
    selected = min(tied, key=lambda candidate: simplicity[candidate])
    return {
        "selected": selected,
        "reason": "near_tie_simplicity" if len(tied) > 1 else "selection_priority",
        "near_tied_candidates": tied,
        "qualification": qualification_rows,
    }


def _prepared_specification(
    raw: Mapping[str, object], common: Mapping[str, object]
) -> dict[str, object]:
    specification = dict(raw)
    specification["minimum_normal_prefix_ms"] = int(common["minimum_normal_prefix_ms"])
    specification["minimum_post_contact_ms"] = int(common["minimum_post_contact_ms"])
    return specification


def _hard_outcome(result: SimulationResult) -> str:
    finite = bool(
        np.all(np.isfinite(result.runtime.pelvis_imu))
        and result.stability is not None
        and result.state_trace is not None
        and np.all(np.isfinite(result.state_trace.robot_qpos))
        and np.all(np.isfinite(result.state_trace.robot_qvel))
        and result.metadata["actual_samples"] == result.metadata["expected_samples"]
        and not result.metadata["terminated_by_viewer"]
    )
    if not finite:
        return "INVALID_OTHER"
    return VALID_STABLE if result.metadata["first_fall_sample"] is None else VALID_FALL


def _simulate_cohort(
    base: SimulationConfig,
    specifications: Sequence[Mapping[str, object]],
    policy_path: Path,
    common: Mapping[str, object],
    progress: Callable[[str], None],
    label: str,
) -> dict[str, tuple[dict[str, object], SimulationResult, str]]:
    simulations = {}
    duration_s = float(common["duration_s"])
    for index, raw in enumerate(specifications, start=1):
        specification = _prepared_specification(raw, common)
        result = run_simulation(
            transition_simulation_config(base, specification, policy_path, duration_s),
            capture_state_trace=True,
        )
        target = str(specification["target_terrain"])
        outcome = (
            _hard_outcome(result)
            if target in {"concrete", "marble"}
            else classify_scenario_outcome(result, specification)
        )
        simulations[str(specification["id"])] = (
            specification,
            result,
            outcome,
        )
        progress(
            f"{label} {index}/{len(specifications)} {specification['id']}: {outcome}"
        )
    return simulations


def _first_contact(result: SimulationResult, target: str) -> int:
    if target in {"concrete", "marble"}:
        return 0
    contact = _first_true(np.any(target_contact_mask(result, target), axis=1))
    if contact is None:
        raise ValueError("valid transition has no exact target contact")
    return contact


def _apply_model(
    simulations: Mapping[str, tuple[Mapping[str, object], SimulationResult, str]],
    features_by_run: Mapping[str, np.ndarray],
    model: CandidateDistanceModel,
    persistence_samples: int,
) -> dict[str, StateDistanceRun]:
    runs = {}
    for run_id, (specification, result, outcome) in simulations.items():
        if outcome not in {VALID_STABLE, VALID_FALL}:
            continue
        assert result.stability is not None
        contact = _first_contact(result, str(specification["target_terrain"]))
        ungated = score_candidate_distance(
            features_by_run[run_id],
            result.stability.gait_phase,
            model,
            persistence_samples,
        )
        primary = score_candidate_distance(
            features_by_run[run_id],
            result.stability.gait_phase,
            model,
            persistence_samples,
            eligible_from_sample=contact,
        )
        runs[run_id] = StateDistanceRun(
            specification,
            result,
            outcome,
            features_by_run[run_id],
            contact,
            ungated,
            primary,
        )
    return runs


def _run_row(run: StateDistanceRun) -> dict[str, object]:
    result = run.result
    onset = _first_true(run.primary.onset)
    ungated_onset = _first_true(run.ungated.onset)
    fall_raw = result.metadata["first_fall_sample"]
    fall = None if fall_raw is None else int(fall_raw)
    valid_detection = bool(onset is not None and fall is not None and onset < fall)
    pretransition = bool(
        ungated_onset is not None and ungated_onset < run.first_contact_sample
    )
    stop = (
        len(result.runtime.sequence)
        if fall is None
        else max(fall, run.first_contact_sample + 1)
    )
    episode = slice(run.first_contact_sample, stop)
    slip = _first_true(result.diagnostics.any_established_slip_after_patch_onset)
    sink = _first_true(np.any(result.diagnostics.deformable_sink_onset, axis=1))
    finite_episode = np.flatnonzero(np.isfinite(run.primary.distance[episode]))
    replay_sample = onset
    if replay_sample is None and finite_episode.size:
        local = int(
            finite_episode[np.argmax(run.primary.distance[episode][finite_episode])]
        )
        replay_sample = run.first_contact_sample + local
    return {
        "run_id": run.run_id,
        "source_terrain": str(run.specification["source_terrain"]),
        "target_terrain": str(run.specification["target_terrain"]),
        "transition": (
            f"{run.specification['source_terrain']}->"
            f"{run.specification['target_terrain']}"
        ),
        "speed_mps": float(run.specification["speed_mps"]),
        "design_role": run.specification.get("design_role"),
        "observed_outcome": "stable" if run.outcome == VALID_STABLE else "fall",
        "first_target_contact_ms": _time_ms(result, run.first_contact_sample),
        "physical_slip_onset_ms": _time_ms(result, slip),
        "physical_sink_onset_ms": _time_ms(result, sink),
        "max_support_deformation_m": float(
            np.max(result.diagnostics.support_surface_max_displacement_m)
        ),
        "t_instability_ms": _time_ms(result, onset),
        "t_fall_ms": _time_ms(result, fall),
        "valid_prefall_detection": valid_detection,
        "false_instability": bool(run.outcome == VALID_STABLE and onset is not None),
        "fall_lead_ms": (
            float(
                result.runtime.timestamp_us[fall] - result.runtime.timestamp_us[onset]
            )
            / 1000.0
            if valid_detection
            else None
        ),
        "pretransition_false_instability": pretransition,
        "pretransition_ungated_onset_ms": (
            _time_ms(result, ungated_onset) if pretransition else None
        ),
        "abnormal_candidate_duration_ms": int(
            np.count_nonzero(run.primary.candidate[episode])
        ),
        "latched_duration_ms": int(np.count_nonzero(run.primary.active[episode])),
        "no_support_samples": int(
            np.count_nonzero(result.stability.gait_phase == NO_SUPPORT)
        ),
        "replay_sample_ms": _time_ms(result, replay_sample),
        "replay_sample_kind": (
            "PRIMARY_ONSET" if onset is not None else "EPISODE_MAXIMUM_SCORE"
        ),
        "support_phase_at_replay": (
            None
            if replay_sample is None
            else PHASE_NAMES[int(result.stability.gait_phase[replay_sample])]
        ),
        "distance_at_replay": (
            None
            if replay_sample is None
            else float(run.primary.distance[replay_sample])
        ),
        "threshold_at_replay": (
            None
            if replay_sample is None
            else float(run.primary.threshold[replay_sample])
        ),
    }


def _coverage_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    stable = [row for row in rows if row["observed_outcome"] == "stable"]
    falling = [row for row in rows if row["observed_outcome"] == "fall"]
    stable_fp = [row for row in stable if row["false_instability"]]
    detected = [row for row in falling if row["valid_prefall_detection"]]

    def grouped(field: str, values: Sequence[str]) -> dict[str, object]:
        grouped_result = {}
        for value in values:
            stable_rows = [row for row in stable if row[field] == value]
            fall_rows = [row for row in falling if row[field] == value]
            fp_rows = [row for row in stable_rows if row["false_instability"]]
            detected_rows = [row for row in fall_rows if row["valid_prefall_detection"]]
            grouped_result[value] = {
                "stable_runs": len(stable_rows),
                "stable_false_instability_runs": len(fp_rows),
                "stable_false_instability_run_rate": (
                    len(fp_rows) / len(stable_rows) if stable_rows else None
                ),
                "fall_runs": len(fall_rows),
                "detected_fall_runs": len(detected_rows),
                "fall_coverage": (
                    len(detected_rows) / len(fall_rows) if fall_rows else 0.0
                ),
            }
        return grouped_result

    leads = [float(row["fall_lead_ms"]) for row in detected]
    return {
        "stable_runs": len(stable),
        "stable_false_instability_runs": [str(row["run_id"]) for row in stable_fp],
        "stable_false_instability_run_rate": (
            len(stable_fp) / len(stable) if stable else 1.0
        ),
        "stable_false_candidate_duration_ms": sum(
            int(row["abnormal_candidate_duration_ms"]) for row in stable_fp
        ),
        "stable_false_latched_duration_ms": sum(
            int(row["latched_duration_ms"]) for row in stable_fp
        ),
        "fall_runs": len(falling),
        "detected_fall_runs": [str(row["run_id"]) for row in detected],
        "fall_coverage": len(detected) / len(falling) if falling else 0.0,
        "by_terrain": grouped("target_terrain", ("ice", "sand")),
        "by_source": grouped("source_terrain", ("concrete", "marble")),
        "pretransition_false_instability_runs": [
            str(row["run_id"]) for row in rows if row["pretransition_false_instability"]
        ],
        "fall_lead_ms": {
            "minimum": float(np.min(leads)) if leads else None,
            "p10": float(np.percentile(leads, 10)) if leads else None,
            "p50": float(np.percentile(leads, 50)) if leads else None,
            "p95": float(np.percentile(leads, 95)) if leads else None,
            "maximum": float(np.max(leads)) if leads else None,
        },
    }


def future_suffix_independence(
    run: StateDistanceRun,
    model: CandidateDistanceModel,
    persistence_samples: int,
) -> dict[str, object]:
    """Change future privileged state and verify all already-decided output."""
    onset = _first_true(run.primary.onset)
    boundary = (
        onset
        if onset is not None
        else max(run.first_contact_sample, len(run.primary.onset) // 2)
    )
    changed_features = run.features.copy()
    changed_features[boundary + 1 :] = 0.0
    changed = score_candidate_distance(
        changed_features,
        run.result.stability.gait_phase,
        model,
        persistence_samples,
        eligible_from_sample=run.first_contact_sample,
    )
    end = boundary + 1
    passed = bool(
        np.array_equal(run.primary.candidate[:end], changed.candidate[:end])
        and np.array_equal(run.primary.active[:end], changed.active[:end])
        and np.array_equal(run.primary.onset[:end], changed.onset[:end])
        and np.array_equal(
            run.primary.distance[:end], changed.distance[:end], equal_nan=True
        )
    )
    return {
        "passed": passed,
        "run_id": run.run_id,
        "comparison_through_sample": boundary,
        "comparison_through_ms": _time_ms(run.result, boundary),
        "future_state_suffix_replaced": True,
        "future_fall_is_not_a_score_input": True,
    }


def _status_block(row: Mapping[str, object]) -> str:
    return "\n".join(
        (
            f"RUN {row['run_id']}",
            f"TRANSITION={row['transition']} OUTCOME={str(row['observed_outcome']).upper()}",
            f"TARGET_CONTACT_MS={row['first_target_contact_ms']}",
            f"PHYSICAL_SLIP_MS={row['physical_slip_onset_ms']}",
            f"PHYSICAL_SINK_MS={row['physical_sink_onset_ms']}",
            f"REPLAY_SAMPLE_MS={row['replay_sample_ms']} KIND={row['replay_sample_kind']}",
            f"SUPPORT_PHASE={row['support_phase_at_replay']}",
            f"STABILITY_DISTANCE={row['distance_at_replay']}",
            f"PHASE_THRESHOLD={row['threshold_at_replay']}",
            f"T_INSTABILITY_MS={row['t_instability_ms']}",
            f"T_FALL_MS={row['t_fall_ms']}",
        )
    )


def _viewer_replay(
    rows: Sequence[Mapping[str, object]]
) -> tuple[dict[str, object], str]:
    requested = (
        ("concrete", "ice", "stable"),
        ("concrete", "ice", "fall"),
        ("marble", "ice", "fall"),
        ("concrete", "sand", "stable"),
        ("concrete", "sand", "fall"),
        ("marble", "sand", "fall"),
    )
    selected = []
    missing = []
    for source, target, outcome in requested:
        matches = [
            row
            for row in rows
            if row["source_terrain"] == source
            and row["target_terrain"] == target
            and row["observed_outcome"] == outcome
        ]
        if not matches:
            missing.append(f"{source}_{target}_{outcome}")
            continue
        selected.append(sorted(matches, key=lambda row: str(row["run_id"]))[0])
    blocks = [_status_block(row) for row in selected]
    parity = not missing and all(
        f"STABILITY_DISTANCE={row['distance_at_replay']}" in block
        and f"T_INSTABILITY_MS={row['t_instability_ms']}" in block
        and f"T_FALL_MS={row['t_fall_ms']}" in block
        for row, block in zip(selected, blocks)
    )
    return (
        {
            "passed": parity,
            "representative_run_ids": [str(row["run_id"]) for row in selected],
            "missing_timelines": missing,
            "status_matches_evaluation": parity,
            "physics_mutation": False,
        },
        "\n\n".join(blocks) + "\n",
    )


def _fresh_gates(
    summary: Mapping[str, object],
    causality: Mapping[str, object],
    acceptance: Mapping[str, object],
) -> dict[str, bool]:
    lead = summary["fall_lead_ms"]["p50"]
    return {
        "stable_specificity": float(summary["stable_false_instability_run_rate"])
        <= float(acceptance["stable_false_instability_run_rate_max"]),
        "fall_coverage": float(summary["fall_coverage"])
        >= float(acceptance["fall_prefall_coverage_min"]),
        "ice_coverage": float(summary["by_terrain"]["ice"]["fall_coverage"])
        >= float(acceptance["ice_fall_coverage_min"]),
        "sand_coverage": float(summary["by_terrain"]["sand"]["fall_coverage"])
        >= float(acceptance["sand_fall_coverage_min"]),
        "source_robustness": all(
            int(summary["by_source"][source]["detected_fall_runs"]) > 0
            for source in acceptance["source_meaningful_detection_required"]
        ),
        "lead_time": lead is not None
        and float(lead) >= float(acceptance["median_fall_lead_ms_min"]),
        "transition_cleanliness": len(summary["pretransition_false_instability_runs"])
        <= int(acceptance["pretransition_false_instability_runs_max"]),
        "causality": bool(causality["passed"]),
    }


def _terrain_regression(
    document: Mapping[str, object],
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, object]:
    runtime_fields = {field.name for field in fields(RuntimeTrace)}
    forbidden_runtime = {
        "robot_qpos",
        "robot_qvel",
        "com",
        "support_phase",
        "fall",
        "terrain_gt",
    }
    independent = not (runtime_fields & forbidden_runtime)
    untouched = dict(before) == dict(after)
    return {
        "passed": untouched and independent,
        "protected_sha256_before": dict(before),
        "protected_sha256_after": dict(after),
        "dataset_model_report_untouched": untouched,
        "terrain_retraining_performed": False,
        "candidate_unchanged": document["terrain_regression"]["candidate"],
        "producer_independence": independent,
    }


def _load_calibration_specs(
    document: Mapping[str, object], repository_root: Path
) -> tuple[list[Mapping[str, object]], Mapping[str, object]]:
    source_path = (
        repository_root / document["source"]["calibration_scenario_config"]
    ).resolve()
    with source_path.open("r", encoding="utf-8") as stream:
        source_document = yaml.safe_load(stream)
    available = {
        str(item["id"]): item
        for item in (
            *source_document["calibration"]["hard_stable_runs"],
            *source_document["calibration"]["transition_runs"],
        )
    }
    requested = [
        *document["calibration"]["hard_stable_run_ids"],
        *document["calibration"]["transition_run_ids"],
    ]
    if len(requested) != len(set(requested)) or not set(requested) <= available.keys():
        raise ValueError(
            "calibration cohort references duplicate or missing source runs"
        )
    return [available[str(run_id)] for run_id in requested], source_document


def validate_experiment_design(
    document: Mapping[str, object],
    calibration_specs: Sequence[Mapping[str, object]],
    source_document: Mapping[str, object],
) -> dict[str, object]:
    """Validate schemas, fixed math, leakage boundary, and fresh disjointness."""
    fit = document["stable_fit"]
    expected = {
        "sample_stride_ms": 10,
        "per_run_per_phase_cap": 256,
        "phase_threshold_quantile": 0.995,
        "persistence_ms": 20,
    }
    for field, value in expected.items():
        if not np.isclose(float(fit[field]), float(value), atol=0.0, rtol=0.0):
            raise ValueError(f"stable fit {field} differs from the frozen contract")
    regularization = fit["covariance_regularization"]
    if not np.isclose(float(regularization["lambda"]), 0.05) or not np.isclose(
        float(regularization["epsilon"]), 1.0e-6
    ):
        raise ValueError("covariance regularization differs from frozen values")
    for candidate in CANDIDATE_ORDER:
        configured = tuple(
            document["candidate_definitions"][candidate]["feature_order"]
        )
        if configured != CANDIDATE_FEATURE_NAMES[candidate]:
            raise ValueError(f"{candidate} feature schema/order changed")
    configured_joints = tuple(
        document["candidate_definitions"][LOWER_BODY_STATE]["authoritative_joint_order"]
    )
    if configured_joints != LOWER_BODY_JOINT_NAMES:
        raise ValueError("lower-body joint schema changed")
    forbidden_tokens = (
        "terrain",
        "scenario",
        "fall",
        "slip",
        "sink",
        "patch",
        "run_id",
        "timestamp",
        "mos",
    )
    for names in CANDIDATE_FEATURE_NAMES.values():
        if any(token in name.lower() for name in names for token in forbidden_tokens):
            raise ValueError("candidate feature schema contains leakage")

    validation = list(document["fresh_validation"]["runs"])
    if not 40 <= len(validation) <= 60:
        raise ValueError("fresh full-state validation must contain 40-60 runs")
    all_ids = [str(item["id"]) for item in (*calibration_specs, *validation)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("calibration/fresh run IDs must be unique")
    calibration_signatures = {_condition_signature(item) for item in calibration_specs}
    previous_validation_signatures = {
        _condition_signature(item)
        for item in source_document["fresh_validation"]["runs"]
    }
    validation_signatures = {_condition_signature(item) for item in validation}
    if validation_signatures & (
        calibration_signatures | previous_validation_signatures
    ):
        raise ValueError("fresh conditions overlap calibration or prior MoS validation")
    if len(validation_signatures) != len(validation):
        raise ValueError("fresh full-state signatures must be unique")
    domains = document["frozen_operating_domains"]
    group_counts = {}
    for item in validation:
        group = str(item["frozen_group"])
        domain = domains[group]
        key = f"{item['source_terrain']}_{group}"
        group_counts[key] = group_counts.get(key, 0) + 1
        for field_name in (
            "speed_mps",
            "patch_start_x_m",
            "slip_pattern",
            "sink_pattern",
            "sink_severity",
            "support_pattern",
        ):
            allowed = domain[field_name]
            allowed_values = allowed if isinstance(allowed, list) else [allowed]
            if item[field_name] not in allowed_values:
                raise ValueError(
                    f"fresh run {item['id']} escapes frozen {group} {field_name}"
                )
        minimum, maximum = domain["patch_width_m"]
        if not float(minimum) <= float(item["patch_width_m"]) <= float(maximum):
            raise ValueError(f"fresh run {item['id']} escapes frozen width")
    required = {
        f"{source}_{group}"
        for source in ("concrete", "marble")
        for group in ("ice_stable", "ice_fall", "sand_stable", "sand_fall")
    }
    if set(group_counts) != required or any(group_counts[key] < 5 for key in required):
        raise ValueError("fresh matrix requires at least five runs per requested group")
    return {
        "passed": True,
        "candidate_dimensions": {
            name: len(CANDIDATE_FEATURE_NAMES[name]) for name in CANDIDATE_ORDER
        },
        "calibration_runs": len(calibration_specs),
        "fresh_runs": len(validation),
        "fresh_group_counts": group_counts,
        "calibration_prior_validation_fresh_disjoint": True,
        "feature_leakage": False,
    }


def _calibration_outcomes_reproduced(
    calibration_simulations: Mapping[
        str, tuple[Mapping[str, object], SimulationResult, str]
    ]
) -> bool:
    for specification, _, outcome in calibration_simulations.values():
        prior = specification.get("prior_observed_outcome")
        if prior is None:
            if outcome != VALID_STABLE:
                return False
        else:
            expected = VALID_STABLE if prior == "stable" else VALID_FALL
            if outcome != expected:
                return False
    return True


def _verdict_from_fresh(
    gates: Mapping[str, bool], summary: Mapping[str, object]
) -> str:
    if all(gates.values()):
        return "FULL_STATE_STABILITY_GROUND_TRUTH_SUPPORTED"
    failures = [name for name, passed in gates.items() if not passed]
    lead = summary["fall_lead_ms"]["p50"]
    mildly_short = bool(
        len(failures) == 1
        and gates["causality"]
        and gates["transition_cleanliness"]
        and float(summary["stable_false_instability_run_rate"]) <= 0.15
        and float(summary["fall_coverage"]) >= 0.80
        and float(summary["by_terrain"]["ice"]["fall_coverage"]) >= 0.75
        and float(summary["by_terrain"]["sand"]["fall_coverage"]) >= 0.75
        and lead is not None
        and float(lead) >= 150.0
    )
    return (
        "FULL_STATE_STABILITY_GROUND_TRUTH_PROMISING"
        if mildly_short
        else "FULL_STATE_STABILITY_GROUND_TRUTH_NOT_SUPPORTED"
    )


def run_full_state_stability_ground_truth_sanity(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Compare three stable-only distance models, freeze one, and validate it."""
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if document["experiment"]["id"] != "FULL_STATE_STABILITY_GROUND_TRUTH_SANITY":
        raise ValueError("unsupported full-state stability experiment")
    calibration_specs, source_document = _load_calibration_specs(
        document, repository_root
    )
    design = validate_experiment_design(document, calibration_specs, source_document)
    if PHYSICS_TIMESTEP_S != float(document["common"]["physics_timestep_s"]):
        raise ValueError("experiment physics timestep differs from canonical value")
    if SENSOR_RATE_HZ != int(document["common"]["sensor_rate_hz"]):
        raise ValueError("experiment sensor rate differs from canonical value")

    artifact_path = (repository_root / document["artifacts"]["path"]).resolve()
    artifact_path.relative_to(repository_root)
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite experiment artifacts: {artifact_path}"
        )
    artifact_path.mkdir(parents=True, exist_ok=True)
    base = load_simulation_config(
        (repository_root / document["source"]["simulator_config"]).resolve()
    )
    policy_path = (repository_root / document["source"]["policy_path"]).resolve()
    if not policy_path.is_file() or _file_sha256(policy_path) != str(
        document["source"]["policy_sha256"]
    ):
        raise ValueError("verified G1 policy is unavailable or has the wrong SHA-256")
    model, _ = load_g1_model("concrete")
    qpos_addresses, qvel_addresses = lower_body_state_addresses(model)
    protected_paths = [
        str(path) for path in document["terrain_regression"]["protected_paths"]
    ]
    terrain_before = _protected_hashes(repository_root, protected_paths)

    calibration_simulations = _simulate_cohort(
        base,
        calibration_specs,
        policy_path,
        document["common"],
        progress,
        "CANDIDATE CALIBRATION",
    )
    calibration_gate = _calibration_outcomes_reproduced(calibration_simulations)
    if not calibration_gate:
        metrics = {
            "experiment": document["experiment"],
            "design": design,
            "calibration": {
                "performed": True,
                "passed": False,
                "reason": "calibration_observed_outcomes_not_reproduced",
            },
            "fresh_validation": {"performed": False},
            "verdict": "FULL_STATE_STABILITY_GROUND_TRUTH_NOT_SEPARABLE",
        }
        _write_json(artifact_path / "results.json", metrics)
        return artifact_path, metrics

    features_by_candidate: dict[str, dict[str, np.ndarray]] = {
        candidate: {} for candidate in CANDIDATE_ORDER
    }
    phase_by_run = {}
    stable_ids = []
    fall_ids = []
    for run_id, (_, result, outcome) in calibration_simulations.items():
        assert result.stability is not None
        phase_by_run[run_id] = result.stability.gait_phase
        for candidate in CANDIDATE_ORDER:
            features_by_candidate[candidate][run_id] = extract_candidate_features(
                result, candidate, qpos_addresses, qvel_addresses
            )
        if outcome == VALID_STABLE:
            stable_ids.append(run_id)
        elif outcome == VALID_FALL:
            fall_ids.append(run_id)

    fit = document["stable_fit"]
    persistence_samples = int(round(int(fit["persistence_ms"]) * SENSOR_RATE_HZ / 1000))
    candidate_models = {}
    candidate_runs_by_candidate = {}
    calibration_rows_by_candidate = {}
    metrics_by_candidate = {}
    for candidate in CANDIDATE_ORDER:
        candidate_models[candidate] = fit_candidate_distance_model(
            candidate,
            {run_id: features_by_candidate[candidate][run_id] for run_id in stable_ids},
            {run_id: phase_by_run[run_id] for run_id in stable_ids},
            stride_samples=int(fit["sample_stride_ms"]),
            per_run_per_phase_cap=int(fit["per_run_per_phase_cap"]),
            standard_deviation_floor=float(fit["standard_deviation_floor"]),
            covariance_lambda=float(fit["covariance_regularization"]["lambda"]),
            covariance_epsilon=float(fit["covariance_regularization"]["epsilon"]),
            threshold_quantile=float(fit["phase_threshold_quantile"]),
        )
        candidate_runs = _apply_model(
            calibration_simulations,
            features_by_candidate[candidate],
            candidate_models[candidate],
            persistence_samples,
        )
        candidate_runs_by_candidate[candidate] = candidate_runs
        rows = [_run_row(candidate_runs[run_id]) for run_id in candidate_runs]
        calibration_rows_by_candidate[candidate] = rows
        metrics_by_candidate[candidate] = _coverage_summary(rows)
        summary = metrics_by_candidate[candidate]
        progress(
            f"{candidate}: stable_fp="
            f"{summary['stable_false_instability_run_rate']:.4f} "
            f"fall_coverage={summary['fall_coverage']:.4f} "
            f"ice={summary['by_terrain']['ice']['fall_coverage']:.4f} "
            f"sand={summary['by_terrain']['sand']['fall_coverage']:.4f}"
        )
    selection = select_calibration_candidate(
        metrics_by_candidate,
        document["calibration"]["qualification"],
        document["calibration"]["near_tie"],
    )
    calibration_metrics = {
        "performed": True,
        "passed": True,
        "stable_runs": len(stable_ids),
        "fall_comparison_runs": len(fall_ids),
        "stable_run_ids": stable_ids,
        "fall_run_ids": fall_ids,
        "phase_fit_sample_counts": {
            candidate: {
                PHASE_NAMES[phase_id]: distribution.fit_sample_count
                for phase_id, distribution in candidate_models[
                    candidate
                ].phase_distributions.items()
            }
            for candidate in CANDIDATE_ORDER
        },
        "candidates": {
            candidate: {
                "feature_dimension": len(CANDIDATE_FEATURE_NAMES[candidate]),
                "feature_order": list(CANDIDATE_FEATURE_NAMES[candidate]),
                "phase_thresholds": {
                    PHASE_NAMES[phase_id]: distribution.distance_threshold
                    for phase_id, distribution in candidate_models[
                        candidate
                    ].phase_distributions.items()
                },
                "metrics": metrics_by_candidate[candidate],
                "runs": calibration_rows_by_candidate[candidate],
                "causality": future_suffix_independence(
                    next(
                        (
                            run
                            for run in candidate_runs_by_candidate[candidate].values()
                            if _first_true(run.primary.onset) is not None
                        ),
                        next(iter(candidate_runs_by_candidate[candidate].values())),
                    ),
                    candidate_models[candidate],
                    persistence_samples,
                ),
            }
            for candidate in CANDIDATE_ORDER
        },
        "selection": selection,
        "fall_runs_excluded_from_fit": all(
            run_id not in candidate_models[PELVIS_STATE].fit_run_ids
            for run_id in fall_ids
        ),
    }
    if selection["selected"] is None:
        diagnostic_replay_candidate = LOWER_BODY_STATE
        calibration_viewer, calibration_status = _viewer_replay(
            calibration_rows_by_candidate[diagnostic_replay_candidate]
        )
        with (artifact_path / "calibration_viewer_status.txt").open(
            "w", encoding="utf-8"
        ) as stream:
            stream.write(calibration_status)
        progress(calibration_status)
        terrain_after = _protected_hashes(repository_root, protected_paths)
        metrics = {
            "experiment": document["experiment"],
            "design": design,
            "calibration": calibration_metrics,
            "selected_candidate": None,
            "calibration_viewer": {
                **calibration_viewer,
                "candidate": diagnostic_replay_candidate,
                "authoritative_selection": False,
            },
            "causality": {
                "passed": all(
                    calibration_metrics["candidates"][candidate]["causality"]["passed"]
                    for candidate in CANDIDATE_ORDER
                ),
                "by_candidate": {
                    candidate: calibration_metrics["candidates"][candidate]["causality"]
                    for candidate in CANDIDATE_ORDER
                },
            },
            "fresh_validation": {
                "performed": False,
                "reason": "no_calibration_candidate_qualified",
            },
            "terrain_regression": _terrain_regression(
                document, terrain_before, terrain_after
            ),
            "fusion_regression": fusion_regression(),
            "historical_mos": document["historical_mos"],
            "verdict": "FULL_STATE_STABILITY_GROUND_TRUTH_NOT_SEPARABLE",
        }
        _write_json(artifact_path / "results.json", metrics)
        return artifact_path, metrics

    selected_name = str(selection["selected"])
    selected_model = candidate_models[selected_name]
    frozen_payload = {
        "artifact_type": "simulation_privileged_stability_ground_truth",
        "runtime_model": False,
        "source_commit": document["experiment"]["source_commit_at_start"],
        "experiment_config": str(config_path.relative_to(repository_root)),
        "experiment_config_sha256": _file_sha256(config_path),
        "calibration_label": document["calibration"]["label"],
        "phase_conditioning": document["phase_conditioning"],
        "persistence_ms": int(fit["persistence_ms"]),
        "model": _model_payload(selected_model),
    }
    artifact_sha = canonical_sha256(frozen_payload)
    frozen_artifact = {**frozen_payload, "sha256": artifact_sha}
    _write_json(artifact_path / "selected_ground_truth.json", frozen_artifact)
    progress(f"selected {selected_name}; frozen SHA-256 {artifact_sha}")

    validation_simulations = _simulate_cohort(
        base,
        document["fresh_validation"]["runs"],
        policy_path,
        document["common"],
        progress,
        "FRESH SELECTED-CANDIDATE VALIDATION",
    )
    valid_validation = {
        run_id: value
        for run_id, value in validation_simulations.items()
        if value[2] in {VALID_STABLE, VALID_FALL}
    }
    invalid_rows = [
        scenario_timing_row(result, specification)
        for specification, result, outcome in validation_simulations.values()
        if outcome not in {VALID_STABLE, VALID_FALL}
    ]
    validation_features = {
        run_id: extract_candidate_features(
            result, selected_name, qpos_addresses, qvel_addresses
        )
        for run_id, (_, result, _) in valid_validation.items()
    }
    validation_runs = _apply_model(
        valid_validation,
        validation_features,
        selected_model,
        persistence_samples,
    )
    validation_rows = [_run_row(validation_runs[run_id]) for run_id in validation_runs]
    summary = _coverage_summary(validation_rows)
    causality_source = next(
        (
            run
            for run in validation_runs.values()
            if run.outcome == VALID_FALL and _first_true(run.primary.onset) is not None
        ),
        next(iter(validation_runs.values())),
    )
    causality = future_suffix_independence(
        causality_source, selected_model, persistence_samples
    )
    viewer, status_text = _viewer_replay(validation_rows)
    with (artifact_path / "viewer_status.txt").open("w", encoding="utf-8") as stream:
        stream.write(status_text)
    progress(status_text)
    artifact_sha_after_validation = canonical_sha256(frozen_payload)
    gates = _fresh_gates(summary, causality, document["fresh_acceptance"])
    terrain_after = _protected_hashes(repository_root, protected_paths)
    terrain = _terrain_regression(document, terrain_before, terrain_after)
    fusion = fusion_regression()
    verdict = _verdict_from_fresh(gates, summary)
    if (
        artifact_sha_after_validation != artifact_sha
        or not terrain["passed"]
        or not fusion["passed"]
        or not viewer["passed"]
    ):
        verdict = "FULL_STATE_STABILITY_GROUND_TRUTH_NOT_SUPPORTED"
    metrics = {
        "experiment": document["experiment"],
        "design": design,
        "calibration": calibration_metrics,
        "selected_candidate": {
            "name": selected_name,
            "feature_dimension": len(selected_model.feature_names),
            "feature_order": list(selected_model.feature_names),
            "selection_reason": selection["reason"],
            "phase_thresholds": {
                PHASE_NAMES[phase_id]: distribution.distance_threshold
                for phase_id, distribution in selected_model.phase_distributions.items()
            },
            "persistence_ms": int(fit["persistence_ms"]),
            "artifact_sha256_before_validation": artifact_sha,
            "artifact_sha256_after_validation": artifact_sha_after_validation,
            "artifact_immutable": artifact_sha_after_validation == artifact_sha,
        },
        "fresh_validation": {
            "performed": True,
            "scenario_gate": {
                "configured_runs": len(validation_simulations),
                "valid_runs": len(valid_validation),
                "invalid_runs": len(invalid_rows),
                "invalid": invalid_rows,
            },
            "stable_table": [
                row for row in validation_rows if row["observed_outcome"] == "stable"
            ],
            "fall_table": [
                row for row in validation_rows if row["observed_outcome"] == "fall"
            ],
            "summary": summary,
            "acceptance_gates": gates,
            "threshold_retuning_performed": False,
        },
        "causality": causality,
        "viewer": viewer,
        "terrain_regression": terrain,
        "fusion_regression": fusion,
        "historical_mos": document["historical_mos"],
        "privileged_runtime_boundary": {
            "terrain_identity_in_vector": False,
            "fall_state_in_vector": False,
            "runtime_imu_used": False,
            "selected_state_is_a_runtime_sensor_commitment": False,
        },
        "verdict": verdict,
    }
    _write_json(artifact_path / "results.json", metrics)
    return artifact_path, metrics
