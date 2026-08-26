"""Minimal MuJoCo terrain profiles for the G1 simulation baseline."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class TerrainProfile:
    """Engineering contact approximation, not a measured material model."""

    name: str
    friction: tuple[float, float, float]
    solref: tuple[float, float]
    solimp: tuple[float, float, float, float, float]
    description: str
    priority: int = 1
    condim: int = 3


TERRAIN_PROFILES = {
    "concrete": TerrainProfile(
        name="concrete",
        friction=(1.00, 0.005, 0.0001),
        solref=(0.015, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
        description="hard, relatively high-friction engineering reference",
    ),
    "marble": TerrainProfile(
        name="marble",
        friction=(0.45, 0.003, 0.0001),
        solref=(0.015, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
        description="hard contact with lower sliding friction than concrete",
    ),
    "ice": TerrainProfile(
        name="ice",
        friction=(0.05, 0.001, 0.00001),
        solref=(0.015, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
        description="hard contact with very low friction",
    ),
    "sand": TerrainProfile(
        name="sand",
        friction=(0.70, 0.010, 0.0010),
        solref=(0.050, 1.5),
        solimp=(0.70, 0.90, 0.010, 0.5, 2.0),
        description="softer, damped, lower-impedance engineering contact",
    ),
}


def get_terrain_profile(name: str) -> TerrainProfile:
    """Return one of the four contract terrain profiles."""
    try:
        return TERRAIN_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown terrain {name!r}; choose from {tuple(TERRAIN_PROFILES)}"
        ) from exc


def apply_terrain_profile(
    model: mujoco.MjModel,
    profile: TerrainProfile,
    geom_name: str = "terrain",
) -> int:
    """Apply a validated profile to the canonical ground plane."""
    geom_id = model.geom(geom_name).id
    friction = np.asarray(profile.friction, dtype=np.float64)
    solref = np.asarray(profile.solref, dtype=np.float64)
    solimp = np.asarray(profile.solimp, dtype=np.float64)
    if friction.shape != (3,) or np.any(friction < 0.0):
        raise ValueError(f"invalid friction values for {profile.name}")
    if solref.shape != (2,) or np.any(solref <= 0.0):
        raise ValueError(f"invalid solref values for {profile.name}")
    if solimp.shape != (5,) or not np.all(np.isfinite(solimp)):
        raise ValueError(f"invalid solimp values for {profile.name}")
    d_zero, d_width, width, midpoint, power = solimp
    if not (
        0.0 < d_zero < 1.0
        and 0.0 < d_width < 1.0
        and width > 0.0
        and 0.0 < midpoint < 1.0
        and power >= 1.0
    ):
        raise ValueError(f"solimp values are out of range for {profile.name}")
    if solref[0] < 2.0 * float(model.opt.timestep):
        raise ValueError(f"{profile.name} solref must be at least 2*timestep")

    model.geom_friction[geom_id] = friction
    model.geom_solref[geom_id] = solref
    model.geom_solimp[geom_id] = solimp
    model.geom_priority[geom_id] = profile.priority
    model.geom_condim[geom_id] = profile.condim
    return int(geom_id)
