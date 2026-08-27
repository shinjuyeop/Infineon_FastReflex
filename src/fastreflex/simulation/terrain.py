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


@dataclass(frozen=True)
class DeformableSupportProfile:
    """Passive vertical-joint parameters for an engineering support proxy."""

    name: str
    travel_m: float
    stiffness_n_per_m: float
    damping_n_s_per_m: float


@dataclass(frozen=True)
class DeformableSupportLayout:
    """Compiled joint/geom addresses for the active deformable support."""

    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    geom_ids: np.ndarray


@dataclass(frozen=True)
class DeformableSupportSample:
    """One simulator-only sample of support mechanics."""

    displacement_m: np.ndarray
    vertical_velocity_m_s: np.ndarray
    cell_contact: np.ndarray


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
SINK_SUPPORT_PATTERNS = (
    "balanced_soft",
    "medial_soft",
    "lateral_soft",
    "localized_soft",
)
DEFORMABLE_SUPPORT_PATTERNS = (
    "balanced_deformable",
    "medial_deformable",
    "lateral_deformable",
    "localized_deformable",
)
ALL_SINK_SUPPORT_PATTERNS = (
    *SINK_SUPPORT_PATTERNS,
    *DEFORMABLE_SUPPORT_PATTERNS,
)
DEFORMABLE_CELL_ORDER = (
    "entry_medial",
    "entry_lateral",
    "exit_medial",
    "exit_lateral",
)
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
UNEVEN_CELL_GEOM_NAMES = tuple(
    f"terrain_uneven_{side}_{segment}_{region}"
    for side in ("left", "right")
    for segment in ("entry", "exit")
    for region in ("medial", "lateral")
)
DEFORMABLE_BALANCED_GEOM_NAMES = tuple(
    f"terrain_deformable_balanced_{side}_{cell}"
    for side in ("left", "right")
    for cell in DEFORMABLE_CELL_ORDER
)
DEFORMABLE_CELL_GEOM_NAMES = tuple(
    f"terrain_deformable_{side}_{cell}"
    for side in ("left", "right")
    for cell in DEFORMABLE_CELL_ORDER
)
DEFORMABLE_GEOM_NAMES = (
    *DEFORMABLE_BALANCED_GEOM_NAMES,
    *DEFORMABLE_CELL_GEOM_NAMES,
)
TRANSITION_PATCH_START_X_M = 0.35
TRANSITION_PATCH_END_X_M = 1.10
TRANSITION_PATCH_WIDTH_M = 0.75
TRANSITION_MIN_X_M = -10.0
TRANSITION_MAX_X_M = 10.0

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

# These passive-joint values are bounded simulator engineering parameters, not
# measured soil mechanics. The independent-cell reference is intentionally
# low-travel so selected mild/moderate/severe cells can become non-level.
DEFORMABLE_SUPPORT_PROFILES = {
    "reference": DeformableSupportProfile(
        name="reference",
        travel_m=0.004,
        stiffness_n_per_m=50_000.0,
        damping_n_s_per_m=1_000.0,
    ),
    "mild": DeformableSupportProfile(
        name="mild",
        travel_m=0.020,
        stiffness_n_per_m=12_000.0,
        damping_n_s_per_m=490.0,
    ),
    "moderate": DeformableSupportProfile(
        name="moderate",
        travel_m=0.040,
        stiffness_n_per_m=7_000.0,
        damping_n_s_per_m=374.0,
    ),
    "severe": DeformableSupportProfile(
        name="severe",
        travel_m=0.065,
        stiffness_n_per_m=4_500.0,
        damping_n_s_per_m=300.0,
    ),
}

DEFORMABLE_CONTACT_PROFILE = TerrainProfile(
    name="deformable_support_contact",
    friction=TERRAIN_PROFILES["sand"].friction,
    solref=TERRAIN_PROFILES["concrete"].solref,
    solimp=TERRAIN_PROFILES["concrete"].solimp,
    description="hard contact on a passive vertically moving support body",
)


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


def get_deformable_support_profile(name: str) -> DeformableSupportProfile:
    """Return one predeclared passive vertical-support profile."""
    try:
        return DEFORMABLE_SUPPORT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown deformable support profile {name!r}; choose from "
            f"{tuple(DEFORMABLE_SUPPORT_PROFILES)}"
        ) from exc


def is_deformable_support_pattern(pattern: str) -> bool:
    return pattern in DEFORMABLE_SUPPORT_PATTERNS


def validate_sink_scenario(
    terrain: str,
    pattern: str,
    severity: str,
    support_pattern: str = "balanced_soft",
) -> None:
    """Validate the config-only sink scenario selection."""
    if pattern not in SINK_PATTERNS:
        raise ValueError(
            f"unknown sink pattern {pattern!r}; choose from {SINK_PATTERNS}"
        )
    get_sink_severity_profile(severity)
    if support_pattern not in ALL_SINK_SUPPORT_PATTERNS:
        raise ValueError(
            f"unknown sink support pattern {support_pattern!r}; "
            f"choose from {ALL_SINK_SUPPORT_PATTERNS}"
        )
    if support_pattern != "balanced_soft" and not pattern.startswith("transition_"):
        raise ValueError("uneven sink support requires a finite transition pattern")
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
    patch_start_x_m: float = TRANSITION_PATCH_START_X_M,
    patch_width_m: float = TRANSITION_PATCH_WIDTH_M,
    support_pattern: str = "balanced_soft",
) -> frozenset[int]:
    """Configure the canonical sink scene for a full-lane or finite patch."""
    validate_sink_scenario("sand", pattern, severity, support_pattern)
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
    if is_transition:
        configure_transition_geometry(model, patch_start_x_m, patch_width_m)
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
    if is_deformable_support_pattern(support_pattern):
        # Robot geoms use bit 1. Static transition geoms use type bit 2 and
        # moving tiles type bit 4, both with affinity bit 1. This keeps both
        # surfaces collidable with the robot while preventing tile/world and
        # tile/tile edge contacts from constraining vertical motion.
        for geom_id in ground_ids.values():
            model.geom_contype[geom_id] = 2
        affected_name = f"terrain_transition_{soft_side}"
        affected_id = model.geom(affected_name).id
        model.geom_contype[affected_id] = 0
        model.geom_conaffinity[affected_id] = 0
        model.geom_rgba[affected_id, 3] = 0.0
        ground_ids.pop(affected_name)
        ground_ids.update(
            _configure_deformable_support(
                model,
                soft_side,
                support_pattern,
                severity,
            )
        )
        return frozenset(ground_ids.values())
    if support_pattern != "balanced_soft":
        affected_name = f"terrain_transition_{soft_side}"
        affected_id = model.geom(affected_name).id
        model.geom_contype[affected_id] = 0
        model.geom_conaffinity[affected_id] = 0
        model.geom_rgba[affected_id, 3] = 0.0
        ground_ids.pop(affected_name)
        selected_soft_cells = {
            "medial_soft": {("entry", "medial"), ("exit", "medial")},
            "lateral_soft": {("entry", "lateral"), ("exit", "lateral")},
            "localized_soft": {("entry", "medial")},
        }[support_pattern]
        for segment in ("entry", "exit"):
            for region in ("medial", "lateral"):
                name = f"terrain_uneven_{soft_side}_{segment}_{region}"
                geom_id = model.geom(name).id
                model.geom_contype[geom_id] = 1
                model.geom_conaffinity[geom_id] = 1
                model.geom_rgba[geom_id] = (
                    0.55,
                    0.20 if (segment, region) in selected_soft_cells else 0.55,
                    0.75,
                    1.0,
                )
                profile = (
                    get_sink_severity_profile(severity)
                    if (segment, region) in selected_soft_cells
                    else get_sink_severity_profile("moderate")
                )
                ground_ids[name] = apply_terrain_profile(model, profile, name)
        return frozenset(ground_ids.values())
    apply_terrain_profile(
        model,
        get_sink_severity_profile(severity),
        f"terrain_transition_{soft_side}",
    )
    return frozenset(ground_ids.values())


def _set_deformable_joint_profile(
    model: mujoco.MjModel,
    joint_name: str,
    profile: DeformableSupportProfile,
) -> None:
    joint_id = model.joint(joint_name).id
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    model.jnt_range[joint_id] = (-profile.travel_m, 0.0)
    model.jnt_stiffness[joint_id] = profile.stiffness_n_per_m
    model.jnt_solref[joint_id] = (0.002, 1.0)
    model.jnt_solimp[joint_id] = (0.99, 0.999, 0.0001, 0.5, 2.0)
    model.dof_damping[dof_address] = profile.damping_n_s_per_m
    model.qpos0[qpos_address] = 0.0
    model.qpos_spring[qpos_address] = 0.0


def _configure_deformable_support(
    model: mujoco.MjModel,
    side: str,
    support_pattern: str,
    severity: str,
) -> dict[str, int]:
    """Enable one passive coupled plate or four passive independent cells."""
    selected_profile = get_deformable_support_profile(severity)
    reference_profile = get_deformable_support_profile("reference")
    active: dict[str, int] = {}

    def enable_geom(name: str, selected: bool) -> None:
        geom_id = model.geom(name).id
        model.geom_contype[geom_id] = 4
        model.geom_conaffinity[geom_id] = 1
        body_id = int(model.geom_bodyid[geom_id])
        model.body_contype[body_id] = 4
        model.body_conaffinity[body_id] = 1
        model.geom_rgba[geom_id] = (
            (0.65, 0.18, 0.72, 1.0)
            if selected
            else (0.35, 0.62, 0.42, 1.0)
        )
        apply_terrain_profile(model, DEFORMABLE_CONTACT_PROFILE, name)
        active[name] = int(geom_id)

    if support_pattern == "balanced_deformable":
        joint_name = f"deformable_balanced_{side}_slide"
        _set_deformable_joint_profile(model, joint_name, selected_profile)
        for cell in DEFORMABLE_CELL_ORDER:
            enable_geom(f"terrain_deformable_balanced_{side}_{cell}", True)
        return active

    selected_cells = {
        "medial_deformable": {"entry_medial", "exit_medial"},
        "lateral_deformable": {"entry_lateral", "exit_lateral"},
        "localized_deformable": {"entry_medial"},
    }[support_pattern]
    for cell in DEFORMABLE_CELL_ORDER:
        profile = selected_profile if cell in selected_cells else reference_profile
        _set_deformable_joint_profile(
            model,
            f"deformable_{side}_{cell}_slide",
            profile,
        )
        enable_geom(f"terrain_deformable_{side}_{cell}", cell in selected_cells)
    return active


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
    for name in UNEVEN_CELL_GEOM_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) != -1:
            set_enabled(name, False, (0.45, 0.25, 0.75, 0.0))
    for name in DEFORMABLE_GEOM_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) != -1:
            set_enabled(name, False, (0.35, 0.45, 0.55, 0.0))


def validate_transition_geometry(
    patch_start_x_m: float,
    patch_width_m: float,
) -> None:
    """Validate a bounded same-height transition inside the canonical scene."""
    patch_end_x_m = patch_start_x_m + patch_width_m
    if not (
        np.isfinite(patch_start_x_m)
        and np.isfinite(patch_width_m)
        and patch_width_m > 0.0
        and TRANSITION_MIN_X_M < patch_start_x_m < patch_end_x_m
        and patch_end_x_m < TRANSITION_MAX_X_M
    ):
        raise ValueError(
            "transition patch must have positive finite width within (-10, 10) m"
        )


def configure_transition_geometry(
    model: mujoco.MjModel,
    patch_start_x_m: float,
    patch_width_m: float,
) -> None:
    """Move existing boxes without changing their shared nominal top height."""
    validate_transition_geometry(patch_start_x_m, patch_width_m)
    patch_end_x_m = patch_start_x_m + patch_width_m

    def set_x_extent(name: str, minimum_x_m: float, maximum_x_m: float) -> None:
        geom_id = model.geom(name).id
        model.geom_pos[geom_id, 0] = (minimum_x_m + maximum_x_m) / 2.0
        model.geom_size[geom_id, 0] = (maximum_x_m - minimum_x_m) / 2.0

    set_x_extent(
        "terrain_transition_pre",
        TRANSITION_MIN_X_M,
        patch_start_x_m,
    )
    for name in TRANSITION_PATCH_GEOM_NAMES:
        set_x_extent(name, patch_start_x_m, patch_end_x_m)
    patch_midpoint_x_m = (patch_start_x_m + patch_end_x_m) / 2.0
    for side in ("left", "right"):
        for region in ("medial", "lateral"):
            for prefix in (
                "terrain_uneven",
                "terrain_deformable",
                "terrain_deformable_balanced",
            ):
                set_x_extent(
                    f"{prefix}_{side}_entry_{region}",
                    patch_start_x_m,
                    patch_midpoint_x_m,
                )
                set_x_extent(
                    f"{prefix}_{side}_exit_{region}",
                    patch_midpoint_x_m,
                    patch_end_x_m,
                )
    set_x_extent(
        "terrain_transition_post",
        patch_end_x_m,
        TRANSITION_MAX_X_M,
    )


def apply_slip_patch_profiles(
    model: mujoco.MjModel,
    patch_start_x_m: float = TRANSITION_PATCH_START_X_M,
    patch_width_m: float = TRANSITION_PATCH_WIDTH_M,
) -> frozenset[int]:
    """Apply concrete-to-full-width-Ice-to-concrete transition profiles."""
    configure_transition_geometry(model, patch_start_x_m, patch_width_m)
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
    support_pattern: str = "balanced_soft",
) -> frozenset[int]:
    """Return the finite soft-patch ids used by transition event timing."""
    if pattern == "uniform" or pattern.startswith("asymmetric_"):
        return frozenset()
    if pattern.startswith("transition_"):
        side = pattern.removeprefix("transition_")
        if support_pattern == "balanced_soft":
            names = (f"terrain_transition_{side}",)
        elif support_pattern == "balanced_deformable":
            names = tuple(
                f"terrain_deformable_balanced_{side}_{cell}"
                for cell in DEFORMABLE_CELL_ORDER
            )
        elif support_pattern in DEFORMABLE_SUPPORT_PATTERNS:
            names = tuple(
                f"terrain_deformable_{side}_{cell}"
                for cell in DEFORMABLE_CELL_ORDER
            )
        else:
            names = tuple(
                f"terrain_uneven_{side}_{segment}_{region}"
                for segment in ("entry", "exit")
                for region in ("medial", "lateral")
            )
    else:
        raise ValueError(
            f"unknown sink pattern {pattern!r}; choose from {SINK_PATTERNS}"
        )
    return frozenset(int(model.geom(name).id) for name in names)


def deformable_support_layout(
    model: mujoco.MjModel,
    support_pattern: str,
) -> DeformableSupportLayout:
    """Resolve active support joint and geom addresses in canonical cell order."""
    qpos_addresses = np.full((2, 4), -1, dtype=np.int32)
    dof_addresses = np.full((2, 4), -1, dtype=np.int32)
    geom_ids = np.full((2, 4), -1, dtype=np.int32)
    if not is_deformable_support_pattern(support_pattern):
        return DeformableSupportLayout(qpos_addresses, dof_addresses, geom_ids)

    for side_index, side in enumerate(("left", "right")):
        balanced = support_pattern == "balanced_deformable"
        for cell_index, cell in enumerate(DEFORMABLE_CELL_ORDER):
            geom_name = (
                f"terrain_deformable_balanced_{side}_{cell}"
                if balanced
                else f"terrain_deformable_{side}_{cell}"
            )
            geom_id = model.geom(geom_name).id
            if int(model.geom_contype[geom_id]) == 0:
                continue
            joint_name = (
                f"deformable_balanced_{side}_slide"
                if balanced
                else f"deformable_{side}_{cell}_slide"
            )
            joint_id = model.joint(joint_name).id
            qpos_addresses[side_index, cell_index] = int(
                model.jnt_qposadr[joint_id]
            )
            dof_addresses[side_index, cell_index] = int(
                model.jnt_dofadr[joint_id]
            )
            geom_ids[side_index, cell_index] = int(geom_id)
    return DeformableSupportLayout(qpos_addresses, dof_addresses, geom_ids)


def read_deformable_support_sample(
    data: mujoco.MjData,
    layout: DeformableSupportLayout,
) -> DeformableSupportSample:
    """Read positive-down joint motion and exact per-cell contact presence."""
    displacement = np.zeros((2, 4), dtype=np.float64)
    velocity = np.zeros((2, 4), dtype=np.float64)
    cell_contact = np.zeros((2, 4), dtype=bool)
    active = layout.qpos_addresses >= 0
    displacement[active] = np.maximum(
        0.0,
        -data.qpos[layout.qpos_addresses[active]],
    )
    velocity[active] = -data.qvel[layout.dof_addresses[active]]
    geom_to_cell = {
        int(geom_id): (side, cell)
        for side in range(2)
        for cell, geom_id in enumerate(layout.geom_ids[side])
        if geom_id >= 0
    }
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            location = geom_to_cell.get(geom_id)
            if location is not None:
                cell_contact[location] = True
    return DeformableSupportSample(
        displacement_m=displacement,
        vertical_velocity_m_s=velocity,
        cell_contact=cell_contact,
    )


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
