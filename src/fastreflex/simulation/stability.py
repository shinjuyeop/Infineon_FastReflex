"""Exact-state walking stability, causal IMU baseline, and state fusion.

Privileged MuJoCo quantities are kept in :class:`StabilityDiagnostics`.  The
runtime detector functions accept pelvis IMU6 only; this separation is a
deliberate leakage boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Mapping, Sequence

import mujoco
import numpy as np

from .hazards import FOOT_CONTACT_GEOM_NAMES, SIDES


GRAVITY_M_S2 = 9.81
NO_SUPPORT = 0
LEFT_SINGLE_SUPPORT = 1
RIGHT_SINGLE_SUPPORT = 2
DOUBLE_SUPPORT = 3
PHASE_NAMES = {
    NO_SUPPORT: "NO_SUPPORT",
    LEFT_SINGLE_SUPPORT: "LEFT_SINGLE_SUPPORT",
    RIGHT_SINGLE_SUPPORT: "RIGHT_SINGLE_SUPPORT",
    DOUBLE_SUPPORT: "DOUBLE_SUPPORT",
}


class TerrainState(IntEnum):
    UNKNOWN = 0
    CONCRETE = 1
    MARBLE = 2
    ICE = 3
    SAND = 4


class StabilityState(IntEnum):
    STABLE = 0
    UNSTABLE = 1


class HazardState(IntEnum):
    NORMAL = 0
    SLIP_RISK = 1
    SINK_RISK = 2
    GENERIC_INSTABILITY = 3


@dataclass(frozen=True)
class ExactStabilitySample:
    """One simulator-only whole-body and support-geometry sample."""

    com_xyz_m: np.ndarray
    com_velocity_xyz_m_s: np.ndarray
    foot_support_points_xyz_m: np.ndarray


@dataclass(frozen=True)
class StabilityDiagnostics:
    """Privileged exact-state stability trace; never a runtime model input."""

    com_xyz_m: np.ndarray
    com_velocity_xyz_m_s: np.ndarray
    support_height_m: np.ndarray
    foot_support_points_xyz_m: np.ndarray
    gait_phase: np.ndarray
    xcom_xy_m: np.ndarray
    raw_margin_of_stability_m: np.ndarray


@dataclass(frozen=True)
class StableCalibrationRun:
    """An observed-stable trace eligible for normal-envelope fitting."""

    run_id: str
    diagnostics: StabilityDiagnostics
    observed_stable: bool
    observed_fall: bool
    source_terrain: str = ""
    target_terrain: str = ""


@dataclass(frozen=True)
class PhaseEnvelope:
    """Normal lower MoS bounds fitted only from stable calibration controls."""

    lower_bound_m: Mapping[int, float]
    quantile: float
    calibration_run_ids: tuple[str, ...]


@dataclass(frozen=True)
class InstabilityTrace:
    residual_m: np.ndarray
    candidate: np.ndarray
    active: np.ndarray
    onset: np.ndarray


@dataclass(frozen=True)
class IMURuleCalibration:
    """Stable-control-only thresholds for three causal IMU features."""

    acceleration_norm_center_m_s2: float
    thresholds: np.ndarray
    quantile: float
    calibration_run_ids: tuple[str, ...]


@dataclass(frozen=True)
class IMURuleTrace:
    features: np.ndarray
    candidate: np.ndarray
    active: np.ndarray
    onset: np.ndarray


@dataclass(frozen=True)
class ParallelRuntimeState:
    """Independent terrain/stability producer state plus deterministic fusion."""

    terrain_state: TerrainState = TerrainState.UNKNOWN
    terrain_valid: bool = False
    terrain_updated_at_us: int | None = None
    stability_state: StabilityState = StabilityState.STABLE
    stability_valid: bool = False
    stability_updated_at_us: int | None = None
    hazard_state: HazardState = HazardState.NORMAL
    recovery_required: bool = False

    def update_terrain(
        self,
        state: TerrainState,
        timestamp_us: int,
        *,
        valid: bool = True,
    ) -> "ParallelRuntimeState":
        updated = replace(
            self,
            terrain_state=state,
            terrain_valid=valid,
            terrain_updated_at_us=timestamp_us,
        )
        return fuse_runtime_state(updated)

    def update_stability(
        self,
        state: StabilityState,
        timestamp_us: int,
        *,
        valid: bool = True,
    ) -> "ParallelRuntimeState":
        updated = replace(
            self,
            stability_state=state,
            stability_valid=valid,
            stability_updated_at_us=timestamp_us,
        )
        return fuse_runtime_state(updated)


def read_exact_stability_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> ExactStabilitySample:
    """Read whole-robot COM/velocity and both four-point sole geometries."""
    pelvis_id = model.body("pelvis").id
    mujoco.mj_subtreeVel(model, data)
    points = np.empty((2, 4, 3), dtype=np.float64)
    for side_index, side in enumerate(SIDES):
        for point_index, name in enumerate(FOOT_CONTACT_GEOM_NAMES[side]):
            points[side_index, point_index] = data.geom_xpos[model.geom(name).id]
    sample = ExactStabilitySample(
        com_xyz_m=data.subtree_com[pelvis_id].copy(),
        com_velocity_xyz_m_s=data.subtree_linvel[pelvis_id].copy(),
        foot_support_points_xyz_m=points,
    )
    if not (
        np.all(np.isfinite(sample.com_xyz_m))
        and np.all(np.isfinite(sample.com_velocity_xyz_m_s))
        and np.all(np.isfinite(sample.foot_support_points_xyz_m))
    ):
        raise ValueError("exact stability sample must be finite")
    return sample


def convex_hull(points_xy: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise 2-D convex hull without duplicate points."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("support points must have finite shape [N,2]")
    unique = sorted(set(map(tuple, points.tolist())))
    if len(unique) < 3:
        raise ValueError("support polygon requires at least three unique points")

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        raise ValueError("support polygon is degenerate")
    return hull


def support_polygon(
    foot_support_points_xyz_m: np.ndarray,
    loaded_contact: np.ndarray,
) -> np.ndarray | None:
    """Construct the active left/right/double support polygon."""
    points = np.asarray(foot_support_points_xyz_m, dtype=np.float64)
    loaded = np.asarray(loaded_contact, dtype=bool)
    if points.shape != (2, 4, 3) or loaded.shape != (2,):
        raise ValueError("support geometry must be [2,4,3] with two contact flags")
    if not np.any(loaded):
        return None
    return convex_hull(points[loaded, :, :2].reshape(-1, 2))


def signed_support_margin(point_xy: np.ndarray, polygon_xy: np.ndarray) -> float:
    """Signed Euclidean distance to a convex polygon (positive inside)."""
    point = np.asarray(point_xy, dtype=np.float64)
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    if point.shape != (2,) or polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("invalid point or support polygon shape")
    minimum_distance = np.inf
    inside = True
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        length_squared = float(edge @ edge)
        if length_squared <= 0.0:
            raise ValueError("support polygon contains a zero-length edge")
        projection = np.clip(float((point - start) @ edge) / length_squared, 0.0, 1.0)
        distance = float(np.linalg.norm(point - (start + projection * edge)))
        minimum_distance = min(minimum_distance, distance)
        if edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0]) < -1.0e-12:
            inside = False
    return minimum_distance if inside else -minimum_distance


def assign_gait_phase(loaded_contact: np.ndarray) -> np.ndarray:
    """Map exact left/right loaded state to the bounded phase contract."""
    loaded = np.asarray(loaded_contact, dtype=bool)
    if loaded.ndim != 2 or loaded.shape[1] != 2:
        raise ValueError("loaded contact must have shape [samples,2]")
    phase = np.full(len(loaded), NO_SUPPORT, dtype=np.int8)
    phase[loaded[:, 0] & ~loaded[:, 1]] = LEFT_SINGLE_SUPPORT
    phase[~loaded[:, 0] & loaded[:, 1]] = RIGHT_SINGLE_SUPPORT
    phase[loaded[:, 0] & loaded[:, 1]] = DOUBLE_SUPPORT
    return phase


def derive_stability_diagnostics(
    com_xyz_m: np.ndarray,
    com_velocity_xyz_m_s: np.ndarray,
    foot_support_points_xyz_m: np.ndarray,
    loaded_contact: np.ndarray,
) -> StabilityDiagnostics:
    """Derive causal XCoM and signed MoS from captured exact state."""
    com = np.asarray(com_xyz_m, dtype=np.float64)
    velocity = np.asarray(com_velocity_xyz_m_s, dtype=np.float64)
    points = np.asarray(foot_support_points_xyz_m, dtype=np.float64)
    loaded = np.asarray(loaded_contact, dtype=bool)
    samples = len(com)
    if (
        com.shape != (samples, 3)
        or velocity.shape != (samples, 3)
        or points.shape != (samples, 2, 4, 3)
        or loaded.shape != (samples, 2)
    ):
        raise ValueError("exact stability arrays have inconsistent shapes")
    if not (np.all(np.isfinite(com)) and np.all(np.isfinite(velocity)) and np.all(np.isfinite(points))):
        raise ValueError("exact stability arrays must be finite")

    phase = assign_gait_phase(loaded)
    height = np.full(samples, np.nan, dtype=np.float64)
    xcom = np.full((samples, 2), np.nan, dtype=np.float64)
    margin = np.full(samples, np.nan, dtype=np.float64)
    for sample in range(samples):
        polygon = support_polygon(points[sample], loaded[sample])
        if polygon is None:
            continue
        support_z = float(np.mean(points[sample, loaded[sample], :, 2]))
        height[sample] = float(com[sample, 2] - support_z)
        if height[sample] <= 0.0:
            continue
        omega0 = np.sqrt(GRAVITY_M_S2 / height[sample])
        xcom[sample] = com[sample, :2] + velocity[sample, :2] / omega0
        margin[sample] = signed_support_margin(xcom[sample], polygon)
    return StabilityDiagnostics(
        com_xyz_m=com,
        com_velocity_xyz_m_s=velocity,
        support_height_m=height,
        foot_support_points_xyz_m=points,
        gait_phase=phase,
        xcom_xy_m=xcom,
        raw_margin_of_stability_m=margin,
    )


def fit_phase_envelope(
    runs: Sequence[StableCalibrationRun],
    quantile: float,
) -> PhaseEnvelope:
    """Fit phase bounds, rejecting fall or non-stable observed outcomes."""
    if not 0.0 < quantile < 0.5:
        raise ValueError("phase envelope quantile must be in (0,0.5)")
    if not runs:
        raise ValueError("at least one stable calibration run is required")
    by_phase: dict[int, list[np.ndarray]] = {
        LEFT_SINGLE_SUPPORT: [],
        RIGHT_SINGLE_SUPPORT: [],
        DOUBLE_SUPPORT: [],
    }
    for run in runs:
        if not run.observed_stable or run.observed_fall:
            raise ValueError(
                "phase envelope accepts observed-stable non-fall calibration runs only"
            )
        values = run.diagnostics.raw_margin_of_stability_m
        for phase in by_phase:
            selected = values[
                (run.diagnostics.gait_phase == phase) & np.isfinite(values)
            ]
            if selected.size:
                by_phase[phase].append(selected)
    bounds: dict[int, float] = {}
    for phase, chunks in by_phase.items():
        if not chunks:
            raise ValueError(f"stable calibration has no {PHASE_NAMES[phase]} samples")
        bounds[phase] = float(
            np.quantile(np.concatenate(chunks), quantile, method="linear")
        )
    return PhaseEnvelope(
        lower_bound_m=bounds,
        quantile=quantile,
        calibration_run_ids=tuple(run.run_id for run in runs),
    )


def causal_persistence(values: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Latch after a current/past-only consecutive true run."""
    candidate = np.asarray(values, dtype=bool)
    if candidate.ndim != 1 or samples <= 0:
        raise ValueError("persistence requires a 1-D candidate and positive samples")
    active = np.zeros_like(candidate)
    onset = np.zeros_like(candidate)
    count = 0
    latched = False
    for index, value in enumerate(candidate):
        count = count + 1 if value else 0
        if not latched and count >= samples:
            latched = True
            onset[index] = True
        active[index] = latched
    return active, onset


def detect_instability(
    diagnostics: StabilityDiagnostics,
    envelope: PhaseEnvelope,
    fixed_margin_m: float,
    persistence_samples: int,
    *,
    eligible_from_sample: int = 0,
) -> InstabilityTrace:
    """Apply the predeclared phase-normalized exact-state instability oracle."""
    if fixed_margin_m < 0.0:
        raise ValueError("fixed stability margin must be nonnegative")
    if not 0 <= eligible_from_sample <= len(
        diagnostics.raw_margin_of_stability_m
    ):
        raise ValueError("eligible sample must lie within the stability trace")
    residual = np.full_like(diagnostics.raw_margin_of_stability_m, np.nan)
    for phase, bound in envelope.lower_bound_m.items():
        mask = (diagnostics.gait_phase == phase) & np.isfinite(
            diagnostics.raw_margin_of_stability_m
        )
        residual[mask] = diagnostics.raw_margin_of_stability_m[mask] - bound
    candidate = np.isfinite(residual) & (residual < -fixed_margin_m)
    candidate[:eligible_from_sample] = False
    active, onset = causal_persistence(candidate, persistence_samples)
    return InstabilityTrace(residual, candidate, active, onset)


def imu_rule_features(
    pelvis_imu: np.ndarray,
    acceleration_norm_center_m_s2: float,
) -> np.ndarray:
    """Compute the three declared current-sample IMU rule features."""
    imu = np.asarray(pelvis_imu, dtype=np.float64)
    if imu.ndim != 2 or imu.shape[1] != 6 or not np.all(np.isfinite(imu)):
        raise ValueError("pelvis IMU must have finite shape [samples,6]")
    gyro_roll_pitch = np.linalg.norm(imu[:, 3:5], axis=1)
    horizontal_acceleration = np.linalg.norm(imu[:, :2], axis=1)
    total_acceleration_deviation = np.abs(
        np.linalg.norm(imu[:, :3], axis=1) - acceleration_norm_center_m_s2
    )
    return np.column_stack(
        (gyro_roll_pitch, horizontal_acceleration, total_acceleration_deviation)
    )


def fit_imu_rule(
    stable_imu_by_run: Mapping[str, np.ndarray],
    quantile: float,
) -> IMURuleCalibration:
    """Fit all rule constants from predeclared stable TRAIN controls only."""
    if not stable_imu_by_run or not 0.5 < quantile < 1.0:
        raise ValueError("IMU rule requires stable runs and a high quantile")
    combined = np.concatenate(
        [np.asarray(values, dtype=np.float64) for values in stable_imu_by_run.values()]
    )
    center = float(np.median(np.linalg.norm(combined[:, :3], axis=1)))
    features = imu_rule_features(combined, center)
    return IMURuleCalibration(
        acceleration_norm_center_m_s2=center,
        thresholds=np.quantile(features, quantile, axis=0),
        quantile=quantile,
        calibration_run_ids=tuple(stable_imu_by_run),
    )


def run_imu_rule(
    pelvis_imu: np.ndarray,
    calibration: IMURuleCalibration,
    persistence_samples: int,
    reset_samples: int,
) -> IMURuleTrace:
    """Causally update STABLE/UNSTABLE with persistence and stable reset."""
    if persistence_samples <= 0 or reset_samples <= 0:
        raise ValueError("IMU rule persistence/reset must be positive")
    features = imu_rule_features(
        pelvis_imu, calibration.acceleration_norm_center_m_s2
    )
    candidate = np.any(features > calibration.thresholds[None, :], axis=1)
    active = np.zeros(len(candidate), dtype=bool)
    onset = np.zeros(len(candidate), dtype=bool)
    abnormal_count = 0
    normal_count = 0
    unstable = False
    for index, abnormal in enumerate(candidate):
        if abnormal:
            abnormal_count += 1
            normal_count = 0
        else:
            abnormal_count = 0
            normal_count += 1
        if not unstable and abnormal_count >= persistence_samples:
            unstable = True
            onset[index] = True
        elif unstable and normal_count >= reset_samples:
            unstable = False
        active[index] = unstable
    return IMURuleTrace(features, candidate, active, onset)


def fuse_runtime_state(state: ParallelRuntimeState) -> ParallelRuntimeState:
    """Apply the fixed terrain-conditioned control-facing fusion truth table."""
    if state.stability_state != StabilityState.UNSTABLE:
        return replace(
            state,
            hazard_state=HazardState.NORMAL,
            recovery_required=False,
        )
    if state.terrain_valid and state.terrain_state == TerrainState.ICE:
        hazard = HazardState.SLIP_RISK
    elif state.terrain_valid and state.terrain_state == TerrainState.SAND:
        hazard = HazardState.SINK_RISK
    else:
        hazard = HazardState.GENERIC_INSTABILITY
    return replace(state, hazard_state=hazard, recovery_required=True)


def format_runtime_status(
    *,
    true_terrain: TerrainState,
    state: ParallelRuntimeState,
    stability_gt: StabilityState,
    stability_ai: StabilityState | None,
    timestamp_us: int,
    event_times_us: Mapping[str, int | None],
) -> str:
    """Render a terminal status block without mutating simulation state."""
    lines = [
        f"TIME US             {timestamp_us}",
        f"TRUE TERRAIN        {true_terrain.name}",
        f"RUNTIME TERRAIN     {state.terrain_state.name if state.terrain_valid else 'UNKNOWN'} (ORACLE_PROXY)",
        f"STABILITY GT        {stability_gt.name}",
        f"STABILITY RULE      {state.stability_state.name}",
        f"STABILITY AI        {stability_ai.name if stability_ai is not None else 'NOT_RUN'}",
        f"HAZARD STATE        {state.hazard_state.name}",
        f"RECOVERY_REQUIRED   {str(state.recovery_required).upper()}",
    ]
    for name in ("transition", "t_instability", "t_rule_detect", "t_ai_detect", "t_fall"):
        value = event_times_us.get(name)
        lines.append(f"{name:<20}{'N/A' if value is None else value}")
    return "\n".join(lines)
