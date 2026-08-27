"""Simulator-only physical diagnostics for Hazard Dataset Contract labels."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


SIDES = ("left", "right")
FOOT_BODY_NAMES = tuple(f"{side}_ankle_roll_link" for side in SIDES)
FOOT_CONTACT_GEOM_NAMES = {
    side: tuple(f"{side}_foot_contact_{index}" for index in range(1, 5))
    for side in SIDES
}
LOAD_ON_N = 5.0
LOAD_OFF_N = 2.5
TOUCHDOWN_TRANSIENT_SAMPLES = 10
SUPPORT_BASELINE_SAMPLES = 20
SUPPORT_BASELINE_PRESENCE_RATIO = 0.5
SUPPORT_BASELINE_MIN_QUADRANTS = 2
SUPPORT_LOSS_THRESHOLD_RATIO = 0.5
SUPPORT_LOSS_PERSISTENCE_SAMPLES = 20
SUPPORT_TOTAL_LOAD_MIN_RATIO = 0.30
SURFACE_SPREAD_THRESHOLD_M = 0.010
SURFACE_SPREAD_PERSISTENCE_SAMPLES = 20
SLIP_THRESHOLD_M = 0.050
SLIP_PERSISTENCE_SAMPLES = 3
SINK_PHYSICAL_THRESHOLD_M = 0.0055
SINK_PHYSICAL_PERSISTENCE_SAMPLES = 20
PRE_EVENT_BASELINE_SAMPLES = 1000
SINK_HAZARD_TILT_THRESHOLD_RAD = 0.04454633221030235
SINK_HAZARD_TILT_PERSISTENCE_SAMPLES = 20


@dataclass(frozen=True)
class ExactFootSample:
    """One label-only MuJoCo foot sample for both feet."""

    physical_contact: np.ndarray
    normal_force_n: np.ndarray
    world_xyz: np.ndarray
    world_velocity_xyz: np.ndarray
    contact_penetration_m: np.ndarray
    quadrant_contact: np.ndarray
    quadrant_normal_force_n: np.ndarray
    quadrant_penetration_m: np.ndarray
    soft_patch_contact: np.ndarray
    low_friction_patch_contact: np.ndarray


@dataclass(frozen=True)
class PhysicalDiagnostics:
    """Full-run exact state separated from runtime model inputs."""

    physical_contact: np.ndarray
    soft_patch_contact: np.ndarray
    soft_patch_contact_onset: np.ndarray
    low_friction_patch_contact: np.ndarray
    low_friction_patch_contact_onset: np.ndarray
    touchdown: np.ndarray
    loaded_contact: np.ndarray
    contact_episode_id: np.ndarray
    foot_world_xyz: np.ndarray
    foot_world_velocity_xyz: np.ndarray
    tangential_anchor_drift_m: np.ndarray
    tangential_velocity_mps: np.ndarray
    contact_penetration_m: np.ndarray
    loaded_reference_penetration_m: np.ndarray
    loaded_penetration_change_m: np.ndarray
    bilateral_loaded_penetration_asymmetry_m: np.ndarray
    quadrant_contact: np.ndarray
    quadrant_normal_force_n: np.ndarray
    quadrant_penetration_m: np.ndarray
    quadrant_loaded: np.ndarray
    loaded_quadrant_count: np.ndarray
    quadrant_supported: np.ndarray
    support_baseline_established: np.ndarray
    support_baseline_onset: np.ndarray
    support_baseline_mask: np.ndarray
    baseline_supported_quadrant_count: np.ndarray
    baseline_median_quadrant_load_n: np.ndarray
    baseline_median_total_load_n: np.ndarray
    support_retained_quadrant_count: np.ndarray
    support_retention_ratio: np.ndarray
    support_loss_ratio: np.ndarray
    weighted_support_loss: np.ndarray
    support_loss_valid: np.ndarray
    support_loss_active: np.ndarray
    support_loss_onset: np.ndarray
    support_surface_displacement_m: np.ndarray
    support_surface_vertical_velocity_m_s: np.ndarray
    support_surface_cell_contact: np.ndarray
    support_surface_spread_m: np.ndarray
    support_surface_max_displacement_m: np.ndarray
    support_surface_mean_displacement_m: np.ndarray
    support_surface_max_downward_velocity_m_s: np.ndarray
    deformable_patch_episode_active: np.ndarray
    deformable_sink_active: np.ndarray
    deformable_sink_onset: np.ndarray
    support_penetration_spread_m: np.ndarray
    support_penetration_max_m: np.ndarray
    support_penetration_load_weighted_std_m: np.ndarray
    support_load_concentration: np.ndarray
    pre_fall_valid: np.ndarray
    established_slip: np.ndarray
    established_slip_onset: np.ndarray
    established_slip_after_patch_onset: np.ndarray
    any_established_slip: np.ndarray
    any_established_slip_onset: np.ndarray
    any_established_slip_after_patch_onset: np.ndarray
    sink_physical_active: np.ndarray
    sink_physical_onset: np.ndarray
    sink_physical_episode_id: np.ndarray
    sink_physical_after_patch_onset: np.ndarray
    sink_degradation_active: np.ndarray
    sink_degradation_onset: np.ndarray
    sink_hazard_active: np.ndarray
    sink_hazard_onset: np.ndarray
    pelvis_world_z_m: np.ndarray
    pelvis_orientation_wxyz: np.ndarray
    pelvis_roll_rad: np.ndarray
    pelvis_pitch_rad: np.ndarray
    pelvis_tilt_rad: np.ndarray
    pelvis_angular_velocity_rad_s: np.ndarray
    pelvis_linear_velocity_m_s: np.ndarray
    pelvis_forward_velocity_m_s: np.ndarray
    forward_velocity_error_m_s: np.ndarray
    pre_event_baseline_valid: np.ndarray
    pelvis_z_drop_from_pre_event_m: np.ndarray
    pelvis_tilt_change_from_pre_event_rad: np.ndarray
    forward_velocity_drop_from_pre_event_m_s: np.ndarray
    pelvis_angular_speed_change_from_pre_event_rad_s: np.ndarray
    fall_active: np.ndarray


def _foot_ids(
    model: mujoco.MjModel,
) -> tuple[tuple[int, int], tuple[frozenset[int], frozenset[int]]]:
    body_ids = (
        model.body(FOOT_BODY_NAMES[0]).id,
        model.body(FOOT_BODY_NAMES[1]).id,
    )
    geom_ids = (
        frozenset(
            model.geom(name).id for name in FOOT_CONTACT_GEOM_NAMES[SIDES[0]]
        ),
        frozenset(
            model.geom(name).id for name in FOOT_CONTACT_GEOM_NAMES[SIDES[1]]
        ),
    )
    return body_ids, geom_ids


def foot_quadrant_index(local_x_m: float, local_y_m: float) -> int:
    """Map foot-local +x front and +y left to four sole regions."""
    front_offset = 0 if local_x_m >= 0.0 else 2
    lateral_offset = 0 if local_y_m >= 0.0 else 1
    return front_offset + lateral_offset


def read_exact_foot_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ground_geom_ids: frozenset[int],
    soft_patch_geom_ids: frozenset[int] = frozenset(),
    low_friction_patch_geom_ids: frozenset[int] = frozenset(),
) -> ExactFootSample:
    """Read label-only contact, kinematics, force, and penetration."""
    body_ids, geom_ids = _foot_ids(model)
    contact = np.zeros(2, dtype=bool)
    soft_patch_contact = np.zeros(2, dtype=bool)
    low_friction_patch_contact = np.zeros(2, dtype=bool)
    normal_force = np.zeros(2, dtype=np.float64)
    penetration = np.zeros(2, dtype=np.float64)
    quadrant_contact = np.zeros((2, 4), dtype=bool)
    quadrant_force = np.zeros((2, 4), dtype=np.float64)
    quadrant_penetration = np.full((2, 4), np.nan, dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        item = data.contact[contact_id]
        geom1, geom2 = int(item.geom1), int(item.geom2)
        if geom1 not in ground_geom_ids and geom2 not in ground_geom_ids:
            continue
        foot_geom = geom2 if geom1 in ground_geom_ids else geom1
        for side_index, ids in enumerate(geom_ids):
            if foot_geom not in ids:
                continue
            contact[side_index] = True
            if geom1 in soft_patch_geom_ids or geom2 in soft_patch_geom_ids:
                soft_patch_contact[side_index] = True
            if (
                geom1 in low_friction_patch_geom_ids
                or geom2 in low_friction_patch_geom_ids
            ):
                low_friction_patch_contact[side_index] = True
            penetration[side_index] = max(
                penetration[side_index], max(0.0, -float(item.dist))
            )
            body_id = body_ids[side_index]
            world_delta = np.asarray(item.pos) - data.xpos[body_id]
            local_position = data.xmat[body_id].reshape(3, 3).T @ world_delta
            quadrant = foot_quadrant_index(
                float(local_position[0]), float(local_position[1])
            )
            physical_penetration = max(0.0, -float(item.dist))
            quadrant_contact[side_index, quadrant] = True
            previous_penetration = quadrant_penetration[side_index, quadrant]
            quadrant_penetration[side_index, quadrant] = (
                physical_penetration
                if not np.isfinite(previous_penetration)
                else max(previous_penetration, physical_penetration)
            )
            wrench.fill(0.0)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            contact_force = max(0.0, float(wrench[0]))
            normal_force[side_index] += contact_force
            quadrant_force[side_index, quadrant] += contact_force

    velocity = np.zeros(6, dtype=np.float64)
    world_velocity = []
    for body_id in body_ids:
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        world_velocity.append(velocity[3:].copy())
    return ExactFootSample(
        physical_contact=contact,
        normal_force_n=normal_force,
        world_xyz=np.stack(tuple(data.xpos[body_id].copy() for body_id in body_ids)),
        world_velocity_xyz=np.asarray(world_velocity, dtype=np.float64),
        contact_penetration_m=penetration,
        quadrant_contact=quadrant_contact,
        quadrant_normal_force_n=quadrant_force,
        quadrant_penetration_m=quadrant_penetration,
        soft_patch_contact=soft_patch_contact,
        low_friction_patch_contact=low_friction_patch_contact,
    )


def loaded_contact_from_force(normal_force_n: np.ndarray) -> np.ndarray:
    """Apply the label-only 5 N/2.5 N load hysteresis per foot."""
    force = np.asarray(normal_force_n, dtype=np.float64)
    if force.ndim != 2 or force.shape[1] != 2 or np.any(force < 0.0):
        raise ValueError("normal force must have shape (samples, 2) and be nonnegative")
    loaded = np.zeros(force.shape, dtype=bool)
    state = np.zeros(2, dtype=bool)
    for sample, value in enumerate(force):
        state = np.where(state, value >= LOAD_OFF_N, value >= LOAD_ON_N)
        loaded[sample] = state
    return loaded


def persistent_oracle(
    observable: np.ndarray,
    valid: np.ndarray,
    contact_episode_id: np.ndarray,
    threshold: float,
    persistence_samples: int,
) -> np.ndarray:
    """Apply persistence without crossing invalid or contact-episode edges."""
    values = np.asarray(observable, dtype=np.float64)
    allowed = np.asarray(valid, dtype=bool)
    episodes = np.asarray(contact_episode_id, dtype=np.int64)
    if not (values.shape == allowed.shape == episodes.shape) or values.ndim != 1:
        raise ValueError("oracle arrays must be aligned one-dimensional arrays")
    if threshold < 0.0 or persistence_samples <= 0:
        raise ValueError("oracle threshold and persistence must be positive")
    active = np.zeros(len(values), dtype=bool)
    count = 0
    previous_episode = -1
    for index, (value, is_valid, episode) in enumerate(
        zip(values, allowed, episodes)
    ):
        passes = bool(
            is_valid
            and episode >= 0
            and np.isfinite(value)
            and value >= threshold
        )
        if not passes:
            count = 0
        elif int(episode) != previous_episode:
            count = 1
        else:
            count += 1
        previous_episode = int(episode)
        active[index] = passes and count >= persistence_samples
    return active


def support_penetration_diagnostics(
    quadrant_contact: np.ndarray,
    quadrant_normal_force_n: np.ndarray,
    quadrant_penetration_m: np.ndarray,
    loaded_contact: np.ndarray,
    contact_episode_id: np.ndarray,
    pre_fall_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive direction-independent loaded-support penetration statistics."""
    contact = np.asarray(quadrant_contact, dtype=bool)
    load = np.asarray(quadrant_normal_force_n, dtype=np.float64)
    penetration = np.asarray(quadrant_penetration_m, dtype=np.float64)
    loaded = np.asarray(loaded_contact, dtype=bool)
    episodes = np.asarray(contact_episode_id, dtype=np.int64)
    pre_fall = np.asarray(pre_fall_valid, dtype=bool)
    if (
        contact.ndim != 3
        or contact.shape[1:] != (2, 4)
        or load.shape != contact.shape
        or penetration.shape != contact.shape
        or loaded.shape != contact.shape[:2]
        or episodes.shape != contact.shape[:2]
        or pre_fall.shape != (contact.shape[0],)
        or np.any(load < 0.0)
    ):
        raise ValueError("quadrant support arrays have inconsistent shapes")
    sample_count = contact.shape[0]
    transient = np.zeros((sample_count, 2), dtype=bool)
    for side in range(2):
        for start in np.flatnonzero(
            np.diff(np.r_[False, episodes[:, side] >= 0].astype(np.int8)) == 1
        ):
            episode = episodes[start, side]
            end = start
            while end < sample_count and episodes[end, side] == episode:
                end += 1
            transient[
                start : min(start + TOUCHDOWN_TRANSIENT_SAMPLES, end), side
            ] = True
    foot_valid = loaded & ~transient & pre_fall[:, None]
    quadrant_loaded = (
        contact
        & np.isfinite(penetration)
        & (load >= LOAD_OFF_N)
        & foot_valid[:, :, None]
    )
    count = np.count_nonzero(quadrant_loaded, axis=2).astype(np.int8)
    spread = np.full((sample_count, 2), np.nan, dtype=np.float64)
    maximum = np.full((sample_count, 2), np.nan, dtype=np.float64)
    weighted_std = np.full((sample_count, 2), np.nan, dtype=np.float64)
    concentration = np.full((sample_count, 2), np.nan, dtype=np.float64)
    for sample in range(sample_count):
        for side in range(2):
            valid = quadrant_loaded[sample, side]
            if np.count_nonzero(valid) < 2:
                continue
            values = penetration[sample, side, valid]
            weights = load[sample, side, valid]
            spread[sample, side] = float(np.max(values) - np.min(values))
            maximum[sample, side] = float(np.max(values))
            mean = float(np.average(values, weights=weights))
            weighted_std[sample, side] = float(
                np.sqrt(np.average(np.square(values - mean), weights=weights))
            )
            concentration[sample, side] = float(np.max(weights) / np.sum(weights))
    return {
        "quadrant_loaded": quadrant_loaded,
        "loaded_quadrant_count": count,
        "support_penetration_spread_m": spread,
        "support_penetration_max_m": maximum,
        "support_penetration_load_weighted_std_m": weighted_std,
        "support_load_concentration": concentration,
    }


def support_loss_diagnostics(
    quadrant_contact: np.ndarray,
    quadrant_normal_force_n: np.ndarray,
    loaded_contact: np.ndarray,
    contact_episode_id: np.ndarray,
    pre_fall_valid: np.ndarray,
    *,
    quadrant_load_cutoff_n: float = LOAD_OFF_N,
    touchdown_transient_samples: int = TOUCHDOWN_TRANSIENT_SAMPLES,
    baseline_samples: int = SUPPORT_BASELINE_SAMPLES,
    baseline_presence_ratio: float = SUPPORT_BASELINE_PRESENCE_RATIO,
    minimum_baseline_quadrants: int = SUPPORT_BASELINE_MIN_QUADRANTS,
    current_total_load_min_ratio: float = SUPPORT_TOTAL_LOAD_MIN_RATIO,
    loss_threshold_ratio: float = SUPPORT_LOSS_THRESHOLD_RATIO,
    persistence_samples: int = SUPPORT_LOSS_PERSISTENCE_SAMPLES,
) -> dict[str, np.ndarray]:
    """Derive a causal per-contact support map and retained-support oracle.

    A baseline is frozen from the first 20 consecutive eligible 1 kHz samples
    after the touchdown transient. State is discarded whenever loaded contact,
    the physical contact episode, or pre-fall validity is lost.
    """
    contact = np.asarray(quadrant_contact, dtype=bool)
    load = np.asarray(quadrant_normal_force_n, dtype=np.float64)
    loaded = np.asarray(loaded_contact, dtype=bool)
    episodes = np.asarray(contact_episode_id, dtype=np.int64)
    pre_fall = np.asarray(pre_fall_valid, dtype=bool)
    if (
        contact.ndim != 3
        or contact.shape[1:] != (2, 4)
        or load.shape != contact.shape
        or loaded.shape != contact.shape[:2]
        or episodes.shape != contact.shape[:2]
        or pre_fall.shape != (contact.shape[0],)
        or np.any(load < 0.0)
    ):
        raise ValueError("support-loss arrays have inconsistent shapes")
    if (
        quadrant_load_cutoff_n < 0.0
        or touchdown_transient_samples < 0
        or baseline_samples <= 0
        or not 0.0 < baseline_presence_ratio <= 1.0
        or not 1 <= minimum_baseline_quadrants <= 4
        or not 0.0 <= current_total_load_min_ratio <= 1.0
        or not 0.0 <= loss_threshold_ratio <= 1.0
        or persistence_samples <= 0
    ):
        raise ValueError("support-loss criteria are outside their valid ranges")

    sample_count = contact.shape[0]
    supported = contact & (load >= quadrant_load_cutoff_n)
    baseline_established = np.zeros((sample_count, 2), dtype=bool)
    baseline_onset = np.zeros((sample_count, 2), dtype=bool)
    baseline_mask = np.zeros((sample_count, 2, 4), dtype=bool)
    baseline_count = np.zeros((sample_count, 2), dtype=np.int8)
    baseline_quadrant_load = np.full(
        (sample_count, 2, 4), np.nan, dtype=np.float64
    )
    baseline_total_load = np.full((sample_count, 2), np.nan, dtype=np.float64)
    retained_count = np.zeros((sample_count, 2), dtype=np.int8)
    retention = np.full((sample_count, 2), np.nan, dtype=np.float64)
    loss_ratio = np.full((sample_count, 2), np.nan, dtype=np.float64)
    weighted_loss = np.full((sample_count, 2), np.nan, dtype=np.float64)
    loss_valid = np.zeros((sample_count, 2), dtype=bool)
    loss_active = np.zeros((sample_count, 2), dtype=bool)
    loss_onset = np.zeros((sample_count, 2), dtype=bool)
    required_presence = int(np.ceil(baseline_samples * baseline_presence_ratio))

    for side in range(2):
        baseline_support: np.ndarray | None = None
        baseline_load: np.ndarray | None = None
        baseline_total = np.nan
        baseline_window_support: list[np.ndarray] = []
        baseline_window_load: list[np.ndarray] = []
        active_episode = -1
        episode_start = -1
        persistence_count = 0
        previous_active = False

        def reset_state() -> None:
            nonlocal baseline_support, baseline_load, baseline_total
            nonlocal baseline_window_support, baseline_window_load
            nonlocal persistence_count, previous_active
            baseline_support = None
            baseline_load = None
            baseline_total = np.nan
            baseline_window_support = []
            baseline_window_load = []
            persistence_count = 0
            previous_active = False

        for sample in range(sample_count):
            episode = int(episodes[sample, side])
            if episode != active_episode:
                reset_state()
                active_episode = episode
                episode_start = sample if episode >= 0 else -1
            if (
                episode < 0
                or not loaded[sample, side]
                or not pre_fall[sample]
            ):
                reset_state()
                continue

            touchdown_age = sample - episode_start
            if baseline_support is None:
                if touchdown_age < touchdown_transient_samples:
                    continue
                baseline_window_support.append(supported[sample, side].copy())
                baseline_window_load.append(load[sample, side].copy())
                if len(baseline_window_support) < baseline_samples:
                    continue
                support_window = np.asarray(baseline_window_support)
                load_window = np.asarray(baseline_window_load)
                baseline_support = (
                    np.count_nonzero(support_window, axis=0) >= required_presence
                )
                baseline_load = np.median(load_window, axis=0)
                baseline_total = float(np.median(np.sum(load_window, axis=1)))
                baseline_onset[sample, side] = True

            baseline_established[sample, side] = True
            baseline_mask[sample, side] = baseline_support
            count = int(np.count_nonzero(baseline_support))
            baseline_count[sample, side] = count
            baseline_quadrant_load[sample, side] = baseline_load
            baseline_total_load[sample, side] = baseline_total
            if count < minimum_baseline_quadrants:
                persistence_count = 0
                previous_active = False
                continue

            current_retained = baseline_support & supported[sample, side]
            retained = int(np.count_nonzero(current_retained))
            retained_count[sample, side] = retained
            retention[sample, side] = retained / count
            loss_ratio[sample, side] = 1.0 - retention[sample, side]
            weight_denominator = float(np.sum(baseline_load[baseline_support]))
            if weight_denominator > 0.0:
                weighted_retention = float(
                    np.sum(baseline_load[current_retained]) / weight_denominator
                )
                weighted_loss[sample, side] = 1.0 - weighted_retention

            current_total = float(np.sum(load[sample, side]))
            valid = bool(
                np.isfinite(baseline_total)
                and baseline_total > 0.0
                and current_total
                >= current_total_load_min_ratio * baseline_total
            )
            loss_valid[sample, side] = valid
            passes = bool(valid and loss_ratio[sample, side] >= loss_threshold_ratio)
            persistence_count = persistence_count + 1 if passes else 0
            current_active = passes and persistence_count >= persistence_samples
            loss_active[sample, side] = current_active
            loss_onset[sample, side] = current_active and not previous_active
            previous_active = current_active

    return {
        "quadrant_supported": supported,
        "support_baseline_established": baseline_established,
        "support_baseline_onset": baseline_onset,
        "support_baseline_mask": baseline_mask,
        "baseline_supported_quadrant_count": baseline_count,
        "baseline_median_quadrant_load_n": baseline_quadrant_load,
        "baseline_median_total_load_n": baseline_total_load,
        "support_retained_quadrant_count": retained_count,
        "support_retention_ratio": retention,
        "support_loss_ratio": loss_ratio,
        "weighted_support_loss": weighted_loss,
        "support_loss_valid": loss_valid,
        "support_loss_active": loss_active,
        "support_loss_onset": loss_onset,
    }


def surface_displacement_diagnostics(
    support_surface_displacement_m: np.ndarray,
    support_surface_vertical_velocity_m_s: np.ndarray,
    support_surface_cell_contact: np.ndarray,
    patch_contact: np.ndarray,
    loaded_contact: np.ndarray,
    contact_episode_id: np.ndarray,
    pre_fall_valid: np.ndarray,
    *,
    threshold_m: float = SURFACE_SPREAD_THRESHOLD_M,
    persistence_samples: int = SURFACE_SPREAD_PERSISTENCE_SAMPLES,
) -> dict[str, np.ndarray]:
    """Derive a causal Sink clock from passive support-body joint state."""
    displacement = np.asarray(support_surface_displacement_m, dtype=np.float64)
    velocity = np.asarray(
        support_surface_vertical_velocity_m_s, dtype=np.float64
    )
    cell_contact = np.asarray(support_surface_cell_contact, dtype=bool)
    patch = np.asarray(patch_contact, dtype=bool)
    loaded = np.asarray(loaded_contact, dtype=bool)
    episodes = np.asarray(contact_episode_id, dtype=np.int64)
    pre_fall = np.asarray(pre_fall_valid, dtype=bool)
    if (
        displacement.ndim != 3
        or displacement.shape[1:] != (2, 4)
        or velocity.shape != displacement.shape
        or cell_contact.shape != displacement.shape
        or patch.shape != displacement.shape[:2]
        or loaded.shape != displacement.shape[:2]
        or episodes.shape != displacement.shape[:2]
        or pre_fall.shape != (displacement.shape[0],)
        or not np.all(np.isfinite(displacement))
        or not np.all(np.isfinite(velocity))
        or np.any(displacement < 0.0)
        or threshold_m < 0.0
        or persistence_samples <= 0
    ):
        raise ValueError("surface-displacement arrays or criteria are invalid")

    spread = np.max(displacement, axis=2) - np.min(displacement, axis=2)
    maximum = np.max(displacement, axis=2)
    mean = np.mean(displacement, axis=2)
    max_downward_velocity = np.maximum(0.0, np.max(velocity, axis=2))
    patch_episode_active = np.zeros(patch.shape, dtype=bool)
    sink_active = np.zeros(patch.shape, dtype=bool)
    sink_onset = np.zeros(patch.shape, dtype=bool)
    for side in range(2):
        active_episode = -1
        patch_seen = False
        persistence_count = 0
        previous_active = False
        for sample in range(displacement.shape[0]):
            episode = int(episodes[sample, side])
            if episode != active_episode:
                active_episode = episode
                patch_seen = False
                persistence_count = 0
                previous_active = False
            if episode < 0 or not pre_fall[sample]:
                patch_seen = False
                persistence_count = 0
                previous_active = False
                continue
            if patch[sample, side]:
                patch_seen = True
            patch_episode_active[sample, side] = patch_seen
            valid = bool(patch_seen and loaded[sample, side])
            passes = bool(valid and spread[sample, side] >= threshold_m)
            persistence_count = persistence_count + 1 if passes else 0
            current_active = passes and persistence_count >= persistence_samples
            sink_active[sample, side] = current_active
            sink_onset[sample, side] = current_active and not previous_active
            previous_active = current_active

    return {
        "support_surface_spread_m": spread,
        "support_surface_max_displacement_m": maximum,
        "support_surface_mean_displacement_m": mean,
        "support_surface_max_downward_velocity_m_s": max_downward_velocity,
        "deformable_patch_episode_active": patch_episode_active,
        "deformable_sink_active": sink_active,
        "deformable_sink_onset": sink_onset,
    }


def uneven_support_oracle(
    support_penetration_spread_m: np.ndarray,
    support_valid: np.ndarray,
    contact_episode_id: np.ndarray,
    threshold_m: float,
    persistence_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a causal per-foot spread threshold with contact-local persistence."""
    spread = np.asarray(support_penetration_spread_m, dtype=np.float64)
    valid = np.asarray(support_valid, dtype=bool)
    episodes = np.asarray(contact_episode_id, dtype=np.int64)
    if spread.ndim != 2 or spread.shape[1] != 2:
        raise ValueError("support spread must have shape (samples, 2)")
    if valid.shape != spread.shape or episodes.shape != spread.shape:
        raise ValueError("uneven-support oracle arrays must be aligned")
    active = np.column_stack(
        tuple(
            persistent_oracle(
                spread[:, side],
                valid[:, side],
                episodes[:, side],
                threshold_m,
                persistence_samples,
            )
            for side in range(2)
        )
    )
    onset = active & ~np.vstack((np.zeros((1, 2), dtype=bool), active[:-1]))
    return active, onset


def _derive_one_foot(
    physical_contact: np.ndarray,
    loaded_contact: np.ndarray,
    foot_xyz: np.ndarray,
    foot_velocity_xyz: np.ndarray,
    penetration: np.ndarray,
    pre_fall_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    sample_count = len(physical_contact)
    episode_id = np.full(sample_count, -1, dtype=np.int32)
    touchdown = np.zeros(sample_count, dtype=bool)
    drift = np.full(sample_count, np.nan, dtype=np.float64)
    tangential_velocity = np.full(sample_count, np.nan, dtype=np.float64)
    reference = np.full(sample_count, np.nan, dtype=np.float64)
    penetration_change = np.full(sample_count, np.nan, dtype=np.float64)
    transient = np.zeros(sample_count, dtype=bool)

    edges = np.diff(np.r_[False, physical_contact, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for current_episode, (start, end) in enumerate(zip(starts, ends)):
        episode_id[start:end] = current_episode
        touchdown[start] = True
        anchor_xy = foot_xyz[start, :2]
        drift[start:end] = np.linalg.norm(foot_xyz[start:end, :2] - anchor_xy, axis=1)
        tangential_velocity[start:end] = np.linalg.norm(
            foot_velocity_xyz[start:end, :2], axis=1
        )
        transient[start : min(start + TOUCHDOWN_TRANSIENT_SAMPLES, end)] = True
        loaded_indices = np.flatnonzero(loaded_contact[start:end])
        if loaded_indices.size:
            first_loaded = start + int(loaded_indices[0])
            selected = np.arange(first_loaded, end)
            selected = selected[loaded_contact[selected]]
            reference[selected] = penetration[first_loaded]
            penetration_change[selected] = penetration[selected] - penetration[first_loaded]

    slip_valid = loaded_contact & ~transient & pre_fall_valid
    sink_physical_valid = slip_valid & np.isfinite(penetration_change)
    sink_physical_active = persistent_oracle(
        penetration_change,
        sink_physical_valid,
        episode_id,
        SINK_PHYSICAL_THRESHOLD_M,
        SINK_PHYSICAL_PERSISTENCE_SAMPLES,
    )
    sink_physical_onset = sink_physical_active & ~np.r_[
        False, sink_physical_active[:-1]
    ]
    sink_physical_episode_id = np.full(sample_count, -1, dtype=np.int32)
    sink_edges = np.diff(
        np.r_[False, sink_physical_active, False].astype(np.int8)
    )
    for sink_episode, (start, end) in enumerate(
        zip(np.flatnonzero(sink_edges == 1), np.flatnonzero(sink_edges == -1))
    ):
        sink_physical_episode_id[start:end] = sink_episode
    return {
        "touchdown": touchdown,
        "episode_id": episode_id,
        "drift": drift,
        "tangential_velocity": tangential_velocity,
        "reference": reference,
        "penetration_change": penetration_change,
        "slip": persistent_oracle(
            drift,
            slip_valid,
            episode_id,
            SLIP_THRESHOLD_M,
            SLIP_PERSISTENCE_SAMPLES,
        ),
        "sink_physical_active": sink_physical_active,
        "sink_physical_onset": sink_physical_onset,
        "sink_physical_episode_id": sink_physical_episode_id,
    }


def derive_physical_diagnostics(
    physical_contact: np.ndarray,
    normal_force_n: np.ndarray,
    foot_world_xyz: np.ndarray,
    foot_world_velocity_xyz: np.ndarray,
    contact_penetration_m: np.ndarray,
    pre_fall_valid: np.ndarray,
    pelvis_world_z_m: np.ndarray,
    pelvis_orientation_wxyz: np.ndarray,
    pelvis_angular_velocity_rad_s: np.ndarray,
    pelvis_linear_velocity_m_s: np.ndarray,
    command_speed_mps: float,
    fall_active: np.ndarray,
    soft_patch_contact: np.ndarray | None = None,
    low_friction_patch_contact: np.ndarray | None = None,
    quadrant_contact: np.ndarray | None = None,
    quadrant_normal_force_n: np.ndarray | None = None,
    quadrant_penetration_m: np.ndarray | None = None,
    support_surface_displacement_m: np.ndarray | None = None,
    support_surface_vertical_velocity_m_s: np.ndarray | None = None,
    support_surface_cell_contact: np.ndarray | None = None,
) -> PhysicalDiagnostics:
    """Derive simulator-only cause/effect diagnostics, never terrain labels."""
    contact = np.asarray(physical_contact, dtype=bool)
    force = np.asarray(normal_force_n, dtype=np.float64)
    xyz = np.asarray(foot_world_xyz, dtype=np.float64)
    velocity = np.asarray(foot_world_velocity_xyz, dtype=np.float64)
    penetration = np.asarray(contact_penetration_m, dtype=np.float64)
    pre_fall = np.asarray(pre_fall_valid, dtype=bool)
    pelvis_z = np.asarray(pelvis_world_z_m, dtype=np.float64)
    orientation = np.asarray(pelvis_orientation_wxyz, dtype=np.float64)
    angular_velocity = np.asarray(pelvis_angular_velocity_rad_s, dtype=np.float64)
    linear_velocity = np.asarray(pelvis_linear_velocity_m_s, dtype=np.float64)
    fallen = np.asarray(fall_active, dtype=bool)
    sample_count = len(contact)
    patch_contact = (
        np.zeros((sample_count, 2), dtype=bool)
        if soft_patch_contact is None
        else np.asarray(soft_patch_contact, dtype=bool)
    )
    friction_patch_contact = (
        np.zeros((sample_count, 2), dtype=bool)
        if low_friction_patch_contact is None
        else np.asarray(low_friction_patch_contact, dtype=bool)
    )
    quadrant_contact_array = (
        np.zeros((sample_count, 2, 4), dtype=bool)
        if quadrant_contact is None
        else np.asarray(quadrant_contact, dtype=bool)
    )
    quadrant_force_array = (
        np.zeros((sample_count, 2, 4), dtype=np.float64)
        if quadrant_normal_force_n is None
        else np.asarray(quadrant_normal_force_n, dtype=np.float64)
    )
    quadrant_penetration_array = (
        np.full((sample_count, 2, 4), np.nan, dtype=np.float64)
        if quadrant_penetration_m is None
        else np.asarray(quadrant_penetration_m, dtype=np.float64)
    )
    surface_displacement_array = (
        np.zeros((sample_count, 2, 4), dtype=np.float64)
        if support_surface_displacement_m is None
        else np.asarray(support_surface_displacement_m, dtype=np.float64)
    )
    surface_velocity_array = (
        np.zeros((sample_count, 2, 4), dtype=np.float64)
        if support_surface_vertical_velocity_m_s is None
        else np.asarray(support_surface_vertical_velocity_m_s, dtype=np.float64)
    )
    surface_cell_contact_array = (
        np.zeros((sample_count, 2, 4), dtype=bool)
        if support_surface_cell_contact is None
        else np.asarray(support_surface_cell_contact, dtype=bool)
    )
    if (
        contact.shape != (sample_count, 2)
        or patch_contact.shape != (sample_count, 2)
        or friction_patch_contact.shape != (sample_count, 2)
        or force.shape != (sample_count, 2)
        or xyz.shape != (sample_count, 2, 3)
        or velocity.shape != (sample_count, 2, 3)
        or penetration.shape != (sample_count, 2)
        or pre_fall.shape != (sample_count,)
        or pelvis_z.shape != (sample_count,)
        or orientation.shape != (sample_count, 4)
        or angular_velocity.shape != (sample_count, 3)
        or linear_velocity.shape != (sample_count, 3)
        or fallen.shape != (sample_count,)
        or quadrant_contact_array.shape != (sample_count, 2, 4)
        or quadrant_force_array.shape != (sample_count, 2, 4)
        or quadrant_penetration_array.shape != (sample_count, 2, 4)
        or surface_displacement_array.shape != (sample_count, 2, 4)
        or surface_velocity_array.shape != (sample_count, 2, 4)
        or surface_cell_contact_array.shape != (sample_count, 2, 4)
    ):
        raise ValueError("physical diagnostic arrays have inconsistent shapes")
    if not (
        np.all(np.isfinite(pelvis_z))
        and np.all(np.isfinite(orientation))
        and np.all(np.isfinite(angular_velocity))
        and np.all(np.isfinite(linear_velocity))
        and np.isfinite(command_speed_mps)
    ):
        raise ValueError("pelvis effect diagnostics must be finite")
    loaded = loaded_contact_from_force(np.where(contact, force, 0.0)) & contact
    derived = tuple(
        _derive_one_foot(
            contact[:, side],
            loaded[:, side],
            xyz[:, side],
            velocity[:, side],
            penetration[:, side],
            pre_fall,
        )
        for side in range(2)
    )

    def stack(name: str) -> np.ndarray:
        return np.column_stack(tuple(values[name] for values in derived))

    both_loaded = np.all(loaded, axis=1)
    penetration_asymmetry = np.full(sample_count, np.nan, dtype=np.float64)
    penetration_asymmetry[both_loaded] = np.abs(
        penetration[both_loaded, 0] - penetration[both_loaded, 1]
    )

    qw, qx, qy, qz = orientation.T
    body_forward_world = np.column_stack(
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy + qw * qz),
            2.0 * (qx * qz - qw * qy),
        )
    )
    body_up_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    pelvis_roll = np.arctan2(
        2.0 * (qy * qz + qw * qx),
        body_up_world_z,
    )
    pelvis_pitch = np.arctan2(
        -2.0 * (qx * qz - qw * qy),
        np.sqrt(
            np.square(2.0 * (qy * qz + qw * qx))
            + np.square(body_up_world_z)
        ),
    )
    pelvis_tilt = np.arccos(np.clip(body_up_world_z, -1.0, 1.0))
    forward_velocity = np.einsum(
        "ij,ij->i", linear_velocity, body_forward_world
    )
    angular_speed = np.linalg.norm(angular_velocity, axis=1)

    patch_onset = patch_contact & ~np.vstack(
        (np.zeros((1, 2), dtype=bool), patch_contact[:-1])
    )
    friction_patch_onset = friction_patch_contact & ~np.vstack(
        (np.zeros((1, 2), dtype=bool), friction_patch_contact[:-1])
    )
    contact_episode_id = stack("episode_id").astype(np.int32)
    support = support_penetration_diagnostics(
        quadrant_contact_array,
        quadrant_force_array,
        quadrant_penetration_array,
        loaded,
        contact_episode_id,
        pre_fall,
    )
    support_loss = support_loss_diagnostics(
        quadrant_contact_array,
        quadrant_force_array,
        loaded,
        contact_episode_id,
        pre_fall,
    )
    surface_displacement = surface_displacement_diagnostics(
        surface_displacement_array,
        surface_velocity_array,
        surface_cell_contact_array,
        patch_contact,
        loaded,
        contact_episode_id,
        pre_fall,
    )
    established_slip = stack("slip")
    established_slip_onset = established_slip & ~np.vstack(
        (np.zeros((1, 2), dtype=bool), established_slip[:-1])
    )
    sink_physical_onset = stack("sink_physical_onset")

    def link_onset_to_patch(
        event_onset: np.ndarray,
        event_patch_contact: np.ndarray,
    ) -> np.ndarray:
        linked = np.zeros((sample_count, 2), dtype=bool)
        for side in range(2):
            patch_seen_episodes: set[int] = set()
            for sample in range(sample_count):
                episode = int(contact_episode_id[sample, side])
                if event_patch_contact[sample, side] and episode >= 0:
                    patch_seen_episodes.add(episode)
                linked[sample, side] = bool(
                    event_onset[sample, side]
                    and episode in patch_seen_episodes
                )
        return linked

    sink_after_patch_onset = link_onset_to_patch(
        sink_physical_onset,
        patch_contact,
    )
    slip_after_patch_onset = link_onset_to_patch(
        established_slip_onset,
        friction_patch_contact,
    )
    any_established_slip = np.any(established_slip, axis=1)
    any_established_slip_onset = any_established_slip & ~np.r_[
        False, any_established_slip[:-1]
    ]
    any_slip_after_patch_onset = np.any(slip_after_patch_onset, axis=1)

    baseline_valid = np.zeros(sample_count, dtype=bool)
    z_drop = np.full(sample_count, np.nan, dtype=np.float64)
    tilt_change = np.full(sample_count, np.nan, dtype=np.float64)
    forward_drop = np.full(sample_count, np.nan, dtype=np.float64)
    angular_speed_change = np.full(sample_count, np.nan, dtype=np.float64)
    event_patch_onset = patch_onset | friction_patch_onset
    event_samples = np.flatnonzero(np.any(event_patch_onset, axis=1))
    if event_samples.size:
        t0 = int(event_samples[0])
        baseline_start = max(0, t0 - PRE_EVENT_BASELINE_SAMPLES)
        baseline_valid[baseline_start:t0] = pre_fall[baseline_start:t0]
        baseline_indices = np.flatnonzero(baseline_valid)
        if baseline_indices.size:
            z_reference = float(np.mean(pelvis_z[baseline_indices]))
            tilt_reference = float(np.mean(pelvis_tilt[baseline_indices]))
            forward_reference = float(np.mean(forward_velocity[baseline_indices]))
            angular_speed_reference = float(np.mean(angular_speed[baseline_indices]))
            z_drop[t0:] = z_reference - pelvis_z[t0:]
            tilt_change[t0:] = pelvis_tilt[t0:] - tilt_reference
            forward_drop[t0:] = forward_reference - forward_velocity[t0:]
            angular_speed_change[t0:] = (
                angular_speed[t0:] - angular_speed_reference
            )

    degradation_active = np.zeros(sample_count, dtype=bool)
    consecutive = 0
    for sample in range(sample_count):
        if (
            pre_fall[sample]
            and pelvis_tilt[sample] > SINK_HAZARD_TILT_THRESHOLD_RAD
        ):
            consecutive += 1
        else:
            consecutive = 0
        degradation_active[sample] = (
            consecutive >= SINK_HAZARD_TILT_PERSISTENCE_SAMPLES
        )
    degradation_onset = degradation_active & ~np.r_[
        False, degradation_active[:-1]
    ]
    physical_sink_seen = np.logical_or.accumulate(
        np.any(sink_after_patch_onset, axis=1)
    )
    sink_hazard_active = degradation_active & physical_sink_seen
    sink_hazard_onset = sink_hazard_active & ~np.r_[
        False, sink_hazard_active[:-1]
    ]

    return PhysicalDiagnostics(
        physical_contact=contact,
        soft_patch_contact=patch_contact,
        soft_patch_contact_onset=patch_onset,
        low_friction_patch_contact=friction_patch_contact,
        low_friction_patch_contact_onset=friction_patch_onset,
        touchdown=stack("touchdown"),
        loaded_contact=loaded,
        contact_episode_id=contact_episode_id,
        foot_world_xyz=xyz.astype(np.float32),
        foot_world_velocity_xyz=velocity.astype(np.float32),
        tangential_anchor_drift_m=stack("drift").astype(np.float32),
        tangential_velocity_mps=stack("tangential_velocity").astype(np.float32),
        contact_penetration_m=penetration.astype(np.float32),
        loaded_reference_penetration_m=stack("reference").astype(np.float32),
        loaded_penetration_change_m=stack("penetration_change").astype(np.float32),
        bilateral_loaded_penetration_asymmetry_m=(
            penetration_asymmetry.astype(np.float32)
        ),
        quadrant_contact=quadrant_contact_array,
        quadrant_normal_force_n=quadrant_force_array.astype(np.float32),
        quadrant_penetration_m=quadrant_penetration_array.astype(np.float32),
        quadrant_loaded=support["quadrant_loaded"],
        loaded_quadrant_count=support["loaded_quadrant_count"],
        quadrant_supported=support_loss["quadrant_supported"],
        support_baseline_established=(
            support_loss["support_baseline_established"]
        ),
        support_baseline_onset=support_loss["support_baseline_onset"],
        support_baseline_mask=support_loss["support_baseline_mask"],
        baseline_supported_quadrant_count=(
            support_loss["baseline_supported_quadrant_count"]
        ),
        baseline_median_quadrant_load_n=(
            support_loss["baseline_median_quadrant_load_n"].astype(np.float32)
        ),
        baseline_median_total_load_n=(
            support_loss["baseline_median_total_load_n"].astype(np.float32)
        ),
        support_retained_quadrant_count=(
            support_loss["support_retained_quadrant_count"]
        ),
        support_retention_ratio=(
            support_loss["support_retention_ratio"].astype(np.float32)
        ),
        support_loss_ratio=support_loss["support_loss_ratio"].astype(np.float32),
        weighted_support_loss=(
            support_loss["weighted_support_loss"].astype(np.float32)
        ),
        support_loss_valid=support_loss["support_loss_valid"],
        support_loss_active=support_loss["support_loss_active"],
        support_loss_onset=support_loss["support_loss_onset"],
        support_surface_displacement_m=(
            surface_displacement_array.astype(np.float32)
        ),
        support_surface_vertical_velocity_m_s=(
            surface_velocity_array.astype(np.float32)
        ),
        support_surface_cell_contact=surface_cell_contact_array,
        support_surface_spread_m=(
            surface_displacement["support_surface_spread_m"].astype(np.float32)
        ),
        support_surface_max_displacement_m=(
            surface_displacement[
                "support_surface_max_displacement_m"
            ].astype(np.float32)
        ),
        support_surface_mean_displacement_m=(
            surface_displacement[
                "support_surface_mean_displacement_m"
            ].astype(np.float32)
        ),
        support_surface_max_downward_velocity_m_s=(
            surface_displacement[
                "support_surface_max_downward_velocity_m_s"
            ].astype(np.float32)
        ),
        deformable_patch_episode_active=(
            surface_displacement["deformable_patch_episode_active"]
        ),
        deformable_sink_active=surface_displacement["deformable_sink_active"],
        deformable_sink_onset=surface_displacement["deformable_sink_onset"],
        support_penetration_spread_m=(
            support["support_penetration_spread_m"].astype(np.float32)
        ),
        support_penetration_max_m=(
            support["support_penetration_max_m"].astype(np.float32)
        ),
        support_penetration_load_weighted_std_m=(
            support["support_penetration_load_weighted_std_m"].astype(np.float32)
        ),
        support_load_concentration=(
            support["support_load_concentration"].astype(np.float32)
        ),
        pre_fall_valid=pre_fall,
        established_slip=established_slip,
        established_slip_onset=established_slip_onset,
        established_slip_after_patch_onset=slip_after_patch_onset,
        any_established_slip=any_established_slip,
        any_established_slip_onset=any_established_slip_onset,
        any_established_slip_after_patch_onset=any_slip_after_patch_onset,
        sink_physical_active=stack("sink_physical_active"),
        sink_physical_onset=sink_physical_onset,
        sink_physical_episode_id=stack("sink_physical_episode_id").astype(np.int32),
        sink_physical_after_patch_onset=sink_after_patch_onset,
        sink_degradation_active=degradation_active,
        sink_degradation_onset=degradation_onset,
        sink_hazard_active=sink_hazard_active,
        sink_hazard_onset=sink_hazard_onset,
        pelvis_world_z_m=pelvis_z.astype(np.float32),
        pelvis_orientation_wxyz=orientation.astype(np.float32),
        pelvis_roll_rad=pelvis_roll.astype(np.float32),
        pelvis_pitch_rad=pelvis_pitch.astype(np.float32),
        pelvis_tilt_rad=pelvis_tilt.astype(np.float32),
        pelvis_angular_velocity_rad_s=angular_velocity.astype(np.float32),
        pelvis_linear_velocity_m_s=linear_velocity.astype(np.float32),
        pelvis_forward_velocity_m_s=forward_velocity.astype(np.float32),
        forward_velocity_error_m_s=(
            float(command_speed_mps) - forward_velocity
        ).astype(np.float32),
        pre_event_baseline_valid=baseline_valid,
        pelvis_z_drop_from_pre_event_m=z_drop.astype(np.float32),
        pelvis_tilt_change_from_pre_event_rad=tilt_change.astype(np.float32),
        forward_velocity_drop_from_pre_event_m_s=forward_drop.astype(np.float32),
        pelvis_angular_speed_change_from_pre_event_rad_s=(
            angular_speed_change.astype(np.float32)
        ),
        fall_active=fallen,
    )
