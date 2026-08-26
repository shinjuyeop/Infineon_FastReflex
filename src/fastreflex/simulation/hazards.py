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
SLIP_THRESHOLD_M = 0.050
SLIP_PERSISTENCE_SAMPLES = 3
SINK_THRESHOLD_M = 0.0055
SINK_PERSISTENCE_SAMPLES = 20


@dataclass(frozen=True)
class ExactFootSample:
    """One label-only MuJoCo foot sample for both feet."""

    physical_contact: np.ndarray
    normal_force_n: np.ndarray
    world_xyz: np.ndarray
    world_velocity_xyz: np.ndarray
    contact_penetration_m: np.ndarray


@dataclass(frozen=True)
class PhysicalDiagnostics:
    """Full-run exact state separated from runtime model inputs."""

    physical_contact: np.ndarray
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
    pre_fall_valid: np.ndarray
    established_slip: np.ndarray
    established_sink: np.ndarray


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


def read_exact_foot_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    terrain_geom_id: int,
) -> ExactFootSample:
    """Read label-only contact, kinematics, force, and penetration."""
    body_ids, geom_ids = _foot_ids(model)
    contact = np.zeros(2, dtype=bool)
    normal_force = np.zeros(2, dtype=np.float64)
    penetration = np.zeros(2, dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        item = data.contact[contact_id]
        geom1, geom2 = int(item.geom1), int(item.geom2)
        if terrain_geom_id not in (geom1, geom2):
            continue
        foot_geom = geom2 if geom1 == terrain_geom_id else geom1
        for side_index, ids in enumerate(geom_ids):
            if foot_geom not in ids:
                continue
            contact[side_index] = True
            penetration[side_index] = max(
                penetration[side_index], max(0.0, -float(item.dist))
            )
            wrench.fill(0.0)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            normal_force[side_index] += max(0.0, float(wrench[0]))

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
    sink_valid = slip_valid & np.isfinite(penetration_change)
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
        "sink": persistent_oracle(
            penetration_change,
            sink_valid,
            episode_id,
            SINK_THRESHOLD_M,
            SINK_PERSISTENCE_SAMPLES,
        ),
    }


def derive_physical_diagnostics(
    physical_contact: np.ndarray,
    normal_force_n: np.ndarray,
    foot_world_xyz: np.ndarray,
    foot_world_velocity_xyz: np.ndarray,
    contact_penetration_m: np.ndarray,
    pre_fall_valid: np.ndarray,
) -> PhysicalDiagnostics:
    """Derive established labels from physical metrics, never terrain identity."""
    contact = np.asarray(physical_contact, dtype=bool)
    force = np.asarray(normal_force_n, dtype=np.float64)
    xyz = np.asarray(foot_world_xyz, dtype=np.float64)
    velocity = np.asarray(foot_world_velocity_xyz, dtype=np.float64)
    penetration = np.asarray(contact_penetration_m, dtype=np.float64)
    pre_fall = np.asarray(pre_fall_valid, dtype=bool)
    sample_count = len(contact)
    if (
        contact.shape != (sample_count, 2)
        or force.shape != (sample_count, 2)
        or xyz.shape != (sample_count, 2, 3)
        or velocity.shape != (sample_count, 2, 3)
        or penetration.shape != (sample_count, 2)
        or pre_fall.shape != (sample_count,)
    ):
        raise ValueError("physical diagnostic arrays have inconsistent shapes")
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

    return PhysicalDiagnostics(
        physical_contact=contact,
        touchdown=stack("touchdown"),
        loaded_contact=loaded,
        contact_episode_id=stack("episode_id").astype(np.int32),
        foot_world_xyz=xyz.astype(np.float32),
        foot_world_velocity_xyz=velocity.astype(np.float32),
        tangential_anchor_drift_m=stack("drift").astype(np.float32),
        tangential_velocity_mps=stack("tangential_velocity").astype(np.float32),
        contact_penetration_m=penetration.astype(np.float32),
        loaded_reference_penetration_m=stack("reference").astype(np.float32),
        loaded_penetration_change_m=stack("penetration_change").astype(np.float32),
        pre_fall_valid=pre_fall,
        established_slip=stack("slip"),
        established_sink=stack("sink"),
    )
