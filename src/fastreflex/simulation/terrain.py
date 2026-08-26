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

SINK_PATTERNS = (
    "uniform",
    "asymmetric_left",
    "asymmetric_right",
    "transition_left",
    "transition_right",
)
SLIP_PATTERNS = ("uniform", "transition")
SINK_SEVERITIES = ("mild", "moderate", "severe")
SINK_PATCH_GEOM_NAMES = ("terrain_left", "terrain_right")
TRANSITION_PATCH_GEOM_NAMES = (
    "terrain_transition_left",
    "terrain_transition_right",
)
TRANSITION_GROUND_GEOM_NAMES = (
    "terrain_transition_pre",
    *TRANSITION_PATCH_GEOM_NAMES,
    "terrain_transition_post",
)
TRANSITION_PATCH_START_X_M = 0.35
TRANSITION_PATCH_END_X_M = 1.10

# Synthetic severity ladder for one compliant lane. These are engineering
# contact-response parameters derived from the uniform sand control, not soil
# measurements. Friction is intentionally unchanged so the study isolates
# spatially asymmetric compliance.
SINK_SEVERITY_PROFILES = {
    "mild": TerrainProfile(
        name="sink_mild",
        friction=TERRAIN_PROFILES["sand"].friction,
        solref=(0.055, 1.5),
        solimp=(0.68, 0.89, 0.011, 0.5, 2.0),
        description="mildly softer than the uniform sand control",
    ),
    "moderate": TerrainProfile(
        name="sink_moderate",
        friction=TERRAIN_PROFILES["sand"].friction,
        solref=(0.060, 1.5),
        solimp=(0.65, 0.87, 0.013, 0.5, 2.0),
        description="moderately softer asymmetric support",
    ),
    "severe": TerrainProfile(
        name="sink_severe",
        friction=TERRAIN_PROFILES["sand"].friction,
        solref=(0.070, 1.5),
        solimp=(0.60, 0.84, 0.016, 0.5, 2.0),
        description="severely softer bounded asymmetric support",
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


def get_sink_severity_profile(name: str) -> TerrainProfile:
    """Return one bounded synthetic compliance severity profile."""
    try:
        return SINK_SEVERITY_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown sink severity {name!r}; choose from {SINK_SEVERITIES}"
        ) from exc


def validate_sink_scenario(terrain: str, pattern: str, severity: str) -> None:
    """Validate the config-only sink scenario selection."""
    if pattern not in SINK_PATTERNS:
        raise ValueError(f"unknown sink pattern {pattern!r}; choose from {SINK_PATTERNS}")
    get_sink_severity_profile(severity)
    if pattern != "uniform" and terrain != "sand":
        raise ValueError("non-uniform sink patterns require terrain='sand'")


def validate_slip_scenario(
    terrain: str,
    slip_pattern: str,
    sink_pattern: str,
) -> None:
    """Validate the finite low-friction transition selection."""
    if slip_pattern not in SLIP_PATTERNS:
        raise ValueError(
            f"unknown slip pattern {slip_pattern!r}; choose from {SLIP_PATTERNS}"
        )
    if slip_pattern == "transition" and sink_pattern != "uniform":
        raise ValueError("sink and slip transition patterns cannot be combined")
    if slip_pattern == "transition" and terrain != "ice":
        raise ValueError("the transition slip pattern requires terrain='ice'")


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


def apply_sink_patch_profiles(
    model: mujoco.MjModel,
    pattern: str,
    severity: str,
) -> frozenset[int]:
    """Configure the canonical sink scene for a full-lane or finite patch."""
    validate_sink_scenario("sand", pattern, severity)
    if pattern == "uniform":
        raise ValueError("the uniform control must use the canonical baseline scene")

    full_lane_colors = {
        "terrain_left": (0.25, 0.45, 0.75, 1.0),
        "terrain_right": (0.75, 0.45, 0.20, 1.0),
    }
    transition_colors = {
        "terrain_transition_pre": (0.35, 0.35, 0.35, 1.0),
        "terrain_transition_left": (0.25, 0.45, 0.75, 1.0),
        "terrain_transition_right": (0.75, 0.45, 0.20, 1.0),
        "terrain_transition_post": (0.35, 0.35, 0.35, 1.0),
    }
    is_transition = pattern.startswith("transition_")
    _select_patch_topology(
        model,
        is_transition=is_transition,
        full_lane_colors=full_lane_colors,
        transition_colors=transition_colors,
    )

    if not is_transition:
        base_profile = get_terrain_profile("sand")
        ground_ids = {
            side: apply_terrain_profile(model, base_profile, f"terrain_{side}")
            for side in ("left", "right")
        }
        soft_side = pattern.removeprefix("asymmetric_")
        apply_terrain_profile(
            model,
            get_sink_severity_profile(severity),
            f"terrain_{soft_side}",
        )
        return frozenset(ground_ids.values())

    stable_profile = get_terrain_profile("concrete")
    ground_ids = {
        name: apply_terrain_profile(model, stable_profile, name)
        for name in TRANSITION_GROUND_GEOM_NAMES
    }
    soft_side = pattern.removeprefix("transition_")
    apply_terrain_profile(
        model,
        get_sink_severity_profile(severity),
        f"terrain_transition_{soft_side}",
    )
    return frozenset(ground_ids.values())


def _select_patch_topology(
    model: mujoco.MjModel,
    *,
    is_transition: bool,
    full_lane_colors: dict[str, tuple[float, ...]],
    transition_colors: dict[str, tuple[float, ...]],
) -> None:
    """Select one already-compiled scene topology before MjData exists."""

    def set_enabled(name: str, enabled: bool, rgba: tuple[float, ...]) -> None:
        geom_id = model.geom(name).id
        model.geom_contype[geom_id] = 1 if enabled else 0
        model.geom_conaffinity[geom_id] = 1 if enabled else 0
        model.geom_rgba[geom_id] = (*rgba[:3], 1.0 if enabled else 0.0)

    for name, rgba in full_lane_colors.items():
        set_enabled(name, not is_transition, rgba)
    for name, rgba in transition_colors.items():
        set_enabled(name, is_transition, rgba)


def apply_slip_patch_profiles(model: mujoco.MjModel) -> frozenset[int]:
    """Apply concrete-to-full-width-Ice-to-concrete transition profiles."""
    ice_color = (0.35, 0.70, 0.95, 1.0)
    _select_patch_topology(
        model,
        is_transition=True,
        full_lane_colors={
            "terrain_left": (0.25, 0.45, 0.75, 1.0),
            "terrain_right": (0.75, 0.45, 0.20, 1.0),
        },
        transition_colors={
            "terrain_transition_pre": (0.35, 0.35, 0.35, 1.0),
            "terrain_transition_left": ice_color,
            "terrain_transition_right": ice_color,
            "terrain_transition_post": (0.35, 0.35, 0.35, 1.0),
        },
    )
    concrete = get_terrain_profile("concrete")
    ground_ids = {
        name: apply_terrain_profile(model, concrete, name)
        for name in TRANSITION_GROUND_GEOM_NAMES
    }
    ice = get_terrain_profile("ice")
    for name in TRANSITION_PATCH_GEOM_NAMES:
        apply_terrain_profile(model, ice, name)
    return frozenset(ground_ids.values())


def soft_sink_geom_ids(
    model: mujoco.MjModel,
    pattern: str,
) -> frozenset[int]:
    """Return the finite soft-patch ids used by transition event timing."""
    if pattern == "uniform" or pattern.startswith("asymmetric_"):
        return frozenset()
    if pattern.startswith("transition_"):
        name = f"terrain_transition_{pattern.removeprefix('transition_')}"
    else:
        raise ValueError(f"unknown sink pattern {pattern!r}; choose from {SINK_PATTERNS}")
    return frozenset((int(model.geom(name).id),))


def low_friction_patch_geom_ids(
    model: mujoco.MjModel,
    slip_pattern: str,
) -> frozenset[int]:
    """Return the full-width Ice patch ids used by Slip event timing."""
    if slip_pattern == "uniform":
        return frozenset()
    if slip_pattern != "transition":
        raise ValueError(
            f"unknown slip pattern {slip_pattern!r}; choose from {SLIP_PATTERNS}"
        )
    return frozenset(int(model.geom(name).id) for name in TRANSITION_PATCH_GEOM_NAMES)
