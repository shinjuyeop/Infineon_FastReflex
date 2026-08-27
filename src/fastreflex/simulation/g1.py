"""Canonical Unitree G1 MuJoCo walking and pelvis-IMU baseline."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import time
from typing import Any, ContextManager

import mujoco
import numpy as np
import yaml

from .hazards import (
    FOOT_CONTACT_GEOM_NAMES,
    PhysicalDiagnostics,
    derive_physical_diagnostics,
    read_exact_foot_sample,
)
from .terrain import (
    apply_slip_patch_profiles,
    apply_sink_patch_profiles,
    apply_terrain_profile,
    get_terrain_profile,
    low_friction_patch_geom_ids,
    soft_sink_geom_ids,
    TRANSITION_PATCH_START_X_M,
    TRANSITION_PATCH_WIDTH_M,
    validate_slip_scenario,
    validate_sink_scenario,
    validate_transition_geometry,
)
from .sensors import FSR_CHANNELS, read_virtual_fsr


PHYSICS_TIMESTEP_S = 0.0005
SENSOR_RATE_HZ = 1000
IMU_CHANNELS = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
ASSET_DIR = Path(__file__).resolve().parent / "assets" / "unitree_g1"
SCENE_PATH = ASSET_DIR / "scene.xml"
SINK_SCENE_PATH = ASSET_DIR / "scene_sink.xml"
UPSTREAM_REVISION = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
TESTED_POLICY_SHA256 = (
    "2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28"
)

# These constants and the 98-element observation order match Unitree RL MjLab's
# G1 velocity-policy deployment configuration at UPSTREAM_REVISION.
ACTUATOR_NAMES = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
DEFAULT_ANGLES = np.asarray(
    (
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
        0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    ),
    dtype=np.float64,
)
KPS = np.asarray(
    (
        40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
        40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
        40.2, 28.5, 28.5,
        14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
        14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
    ),
    dtype=np.float64,
)
KDS = np.asarray(
    (
        2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
        2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
        2.6, 1.8, 1.8,
        0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
        0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
    ),
    dtype=np.float64,
)
ACTION_SCALE = np.asarray(
    (
        0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
        0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
        0.55, 0.44, 0.44,
        0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
        0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
    ),
    dtype=np.float64,
)
POLICY_PERIOD_S = 0.6
CONTROL_PERIOD_S = 0.02
VIEWER_SYNC_PERIOD_S = 1.0 / 60.0


@dataclass(frozen=True)
class SimulationConfig:
    """Minimal configuration for one in-memory smoke simulation."""

    physics_timestep_s: float
    sensor_rate_hz: int
    duration_s: float
    command_speed_mps: float
    policy_path: Path | None
    terrain: str
    slip_pattern: str
    sink_pattern: str
    sink_severity: str
    patch_start_x_m: float
    patch_width_m: float
    headless: bool
    sink_support_pattern: str = "balanced_soft"

    @property
    def physics_steps_per_sample(self) -> int:
        return int(round(1.0 / (self.physics_timestep_s * self.sensor_rate_hz)))

    @property
    def expected_samples(self) -> int:
        return int(round(self.duration_s * self.sensor_rate_hz))

    @property
    def total_physics_steps(self) -> int:
        return self.expected_samples * self.physics_steps_per_sample

    def validate(self) -> None:
        if self.physics_timestep_s != PHYSICS_TIMESTEP_S:
            raise ValueError("the baseline physics timestep must be exactly 0.0005 s")
        if self.sensor_rate_hz != SENSOR_RATE_HZ:
            raise ValueError("the Hazard Dataset Contract sensor rate must be 1000 Hz")
        if self.duration_s <= 0.0:
            raise ValueError("duration must be positive")
        if not np.isclose(
            self.expected_samples / self.sensor_rate_hz,
            self.duration_s,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("duration must contain an integer number of 1 kHz samples")
        steps = 1.0 / (self.physics_timestep_s * self.sensor_rate_hz)
        if not np.isclose(steps, self.physics_steps_per_sample, atol=1e-12, rtol=0.0):
            raise ValueError("physics rate must divide the sensor rate exactly")
        if not 0.1 <= self.command_speed_mps <= 0.5:
            raise ValueError("walking command speed must be in [0.1, 0.5] m/s")
        get_terrain_profile(self.terrain)
        validate_slip_scenario(self.terrain, self.slip_pattern, self.sink_pattern)
        validate_sink_scenario(
            self.terrain,
            self.sink_pattern,
            self.sink_severity,
            self.sink_support_pattern,
        )
        validate_transition_geometry(self.patch_start_x_m, self.patch_width_m)


@dataclass(frozen=True)
class RuntimeTrace:
    """The complete runtime-facing signal; no exact state is included."""

    sequence: np.ndarray
    timestamp_us: np.ndarray
    pelvis_imu: np.ndarray
    foot_fsr: np.ndarray | None = None


@dataclass(frozen=True)
class SimulationResult:
    """Runtime trace plus separately named simulator-only diagnostics."""

    runtime: RuntimeTrace
    diagnostics: PhysicalDiagnostics
    metadata: dict[str, object]


def load_simulation_config(path: Path) -> SimulationConfig:
    """Load the one canonical simulator YAML configuration."""
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("simulator config must be a YAML mapping")
    try:
        simulation = document["simulation"]
        controller = document["controller"]
        terrain = document["terrain"]
        slip = document.get("slip", {})
        sink = document.get("sink", {})
        transition_patch = document.get("transition_patch", {})
        output = document["output"]
        raw_policy = controller["policy_path"]
        config = SimulationConfig(
            physics_timestep_s=float(simulation["physics_timestep_s"]),
            sensor_rate_hz=int(simulation["sensor_rate_hz"]),
            duration_s=float(simulation["duration_s"]),
            command_speed_mps=float(controller["command_speed_mps"]),
            policy_path=None if raw_policy in (None, "") else Path(raw_policy),
            terrain=str(terrain["type"]),
            slip_pattern=str(slip.get("pattern", "uniform")),
            sink_pattern=str(sink.get("pattern", "uniform")),
            sink_severity=str(sink.get("severity", "moderate")),
            patch_start_x_m=float(
                transition_patch.get("start_x_m", TRANSITION_PATCH_START_X_M)
            ),
            patch_width_m=float(
                transition_patch.get("width_m", TRANSITION_PATCH_WIDTH_M)
            ),
            headless=bool(output["headless"]),
            sink_support_pattern=str(
                sink.get("support_pattern", "balanced_soft")
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid canonical G1 simulator config") from exc
    config.validate()
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sensor_slice(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = model.sensor(name).id
    address = int(model.sensor_adr[sensor_id])
    dimension = int(model.sensor_dim[sensor_id])
    return data.sensordata[address : address + dimension]


def validate_model_contract(model: mujoco.MjModel) -> None:
    """Fail closed if the migrated model no longer matches the baseline."""
    actuators = tuple(model.actuator(index).name for index in range(model.nu))
    if actuators != ACTUATOR_NAMES or model.nq != 36 or model.nv != 35:
        raise ValueError("G1 29-DOF actuator or state layout changed")
    if not np.isclose(float(model.opt.timestep), PHYSICS_TIMESTEP_S):
        raise ValueError("G1 scene physics timestep changed")

    pelvis_id = model.body("pelvis").id
    imu_id = model.site("imu").id
    if int(model.site_bodyid[imu_id]) != pelvis_id:
        raise ValueError("imu site is not attached to the pelvis")
    if not np.allclose(model.site_pos[imu_id], 0.0):
        raise ValueError("imu site is not at the pelvis frame origin")
    if not np.allclose(model.site_quat[imu_id], (1.0, 0.0, 0.0, 0.0)):
        raise ValueError("imu site is rotated relative to the pelvis frame")
    for name in ("imu_acc", "imu_gyro"):
        if int(model.sensor_dim[model.sensor(name).id]) != 3:
            raise ValueError(f"{name} must have exactly three channels")
    for removed_name in (
        "left_foot_imu_acc",
        "left_foot_imu_gyro",
        "right_foot_imu_acc",
        "right_foot_imu_gyro",
        "left_ankle_ft_force",
        "left_ankle_ft_torque",
        "right_ankle_ft_force",
        "right_ankle_ft_torque",
    ):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, removed_name) != -1:
            raise ValueError(f"research-only sensor remains enabled: {removed_name}")


def load_g1_model(
    terrain_name: str,
    sink_pattern: str = "uniform",
    sink_severity: str = "moderate",
    slip_pattern: str = "uniform",
    patch_start_x_m: float = TRANSITION_PATCH_START_X_M,
    patch_width_m: float = TRANSITION_PATCH_WIDTH_M,
    sink_support_pattern: str = "balanced_soft",
) -> tuple[mujoco.MjModel, frozenset[int]]:
    """Load the baseline or one validated finite/full-lane patch scene."""
    validate_slip_scenario(terrain_name, slip_pattern, sink_pattern)
    validate_sink_scenario(
        terrain_name,
        sink_pattern,
        sink_severity,
        sink_support_pattern,
    )
    validate_transition_geometry(patch_start_x_m, patch_width_m)
    use_patch_scene = sink_pattern != "uniform" or slip_pattern == "transition"
    scene_path = SINK_SCENE_PATH if use_patch_scene else SCENE_PATH
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    validate_model_contract(model)
    if slip_pattern == "transition":
        ground_ids = apply_slip_patch_profiles(
            model,
            patch_start_x_m,
            patch_width_m,
        )
    elif sink_pattern == "uniform":
        ground_ids = frozenset(
            (apply_terrain_profile(model, get_terrain_profile(terrain_name)),)
        )
    else:
        ground_ids = apply_sink_patch_profiles(
            model,
            sink_pattern,
            sink_severity,
            patch_start_x_m,
            patch_width_m,
            sink_support_pattern,
        )
    return model, ground_ids


def launch_passive_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> ContextManager[Any]:
    """Launch MuJoCo's optional GUI with a clear availability error."""
    try:
        viewer_module = importlib.import_module("mujoco.viewer")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "MuJoCo viewer is unavailable; install the official mujoco package "
            "with GUI support and use --headless on display-free systems"
        ) from exc
    viewer = None
    try:
        viewer = viewer_module.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        )
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("pelvis").id
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = -130.0
            viewer.cam.elevation = -20.0
        return viewer
    except Exception as exc:
        if viewer is not None:
            viewer.close()
        raise RuntimeError(
            "MuJoCo viewer failed to launch; verify DISPLAY/Wayland access "
            "(and use mjpython on macOS), or run with --headless"
        ) from exc


def _copy_integration_state(
    source_model: mujoco.MjModel,
    source_data: mujoco.MjData,
    viewer_model: mujoco.MjModel,
    viewer_data: mujoco.MjData,
) -> None:
    """Copy render state without exposing canonical physics to GUI inputs."""
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(source_model, state_spec), dtype=np.float64)
    mujoco.mj_getState(source_model, source_data, state, state_spec)
    mujoco.mj_setState(viewer_model, viewer_data, state, state_spec)


def _pace_viewer(simulation_time_s: float, wall_start_s: float) -> None:
    target_wall_time = wall_start_s + simulation_time_s
    remaining_s = target_wall_time - time.monotonic()
    if remaining_s > 0.0:
        time.sleep(remaining_s)


def read_pelvis_imu(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Read raw pelvis-local accel xyz then gyro xyz as float32."""
    sample = np.concatenate(
        (_sensor_slice(model, data, "imu_acc"), _sensor_slice(model, data, "imu_gyro"))
    ).astype(np.float32)
    if sample.shape != (6,) or not np.all(np.isfinite(sample)):
        raise ValueError("pelvis IMU sample must be six finite values")
    return sample


def gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Rotate the world gravity unit vector into the body frame (wxyz)."""
    qw, qx, qy, qz = quaternion
    return np.asarray(
        (
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ),
        dtype=np.float64,
    )


class UnitreeG1Controller:
    """Fixed adapter for Unitree RL MjLab's pretrained G1 velocity policy."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy_path: Path,
        forward_speed_mps: float,
    ) -> None:
        if not policy_path.is_file():
            raise FileNotFoundError(f"Unitree G1 policy not found: {policy_path}")
        policy_hash = sha256_file(policy_path)
        if policy_hash != TESTED_POLICY_SHA256:
            raise ValueError(
                "policy SHA-256 does not match the verified Unitree G1 artifact: "
                f"{policy_hash}"
            )
        if not 0.1 <= forward_speed_mps <= 0.5:
            raise ValueError("forward speed must be in [0.1, 0.5] m/s")
        actuators = tuple(model.actuator(index).name for index in range(model.nu))
        if actuators != ACTUATOR_NAMES:
            raise ValueError("G1 29-DOF actuator order changed")
        control_steps = CONTROL_PERIOD_S / float(model.opt.timestep)
        self.control_decimation = int(round(control_steps))
        if not np.isclose(control_steps, self.control_decimation, atol=1e-12, rtol=0.0):
            raise ValueError("physics timestep must divide the 20 ms control period")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("onnxruntime is required for G1 policy inference") from exc
        self.session = ort.InferenceSession(
            str(policy_path), providers=("CPUExecutionProvider",)
        )
        if self.session.get_inputs()[0].shape != [1, 98]:
            raise ValueError("expected Unitree G1 policy input [1, 98]")
        if self.session.get_outputs()[0].shape != [1, 29]:
            raise ValueError("expected Unitree G1 policy output [1, 29]")

        self.model = model
        self.data = data
        self.policy_sha256 = policy_hash
        self.command = np.asarray((forward_speed_mps, 0.0, 0.0), dtype=np.float64)
        self.action = np.zeros(29, dtype=np.float32)
        self.target_position = DEFAULT_ANGLES.copy()
        self.step_count = 0
        self.global_phase = 0.0

        # Match the fixed stand pose reached by the official deployment FSM,
        # avoiding that unrelated startup transient in short smoke runs.
        self.data.qpos[7:] = DEFAULT_ANGLES
        mujoco.mj_forward(self.model, self.data)

    def apply(self) -> None:
        torque = (
            (self.target_position - self.data.qpos[7:]) * KPS
            - self.data.qvel[6:] * KDS
        )
        self.data.ctrl[:] = np.clip(
            torque,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

    def update_after_step(self) -> None:
        self.step_count += 1
        if self.step_count % self.control_decimation:
            return
        self.global_phase = (
            self.global_phase + CONTROL_PERIOD_S / POLICY_PERIOD_S
        ) % 1.0
        observation = np.concatenate(
            (
                _sensor_slice(self.model, self.data, "imu_gyro"),
                gravity_orientation(self.data.qpos[3:7]),
                self.command,
                (
                    np.sin(2.0 * np.pi * self.global_phase),
                    np.cos(2.0 * np.pi * self.global_phase),
                ),
                self.data.qpos[7:] - DEFAULT_ANGLES,
                self.data.qvel[6:],
                self.action,
            )
        ).astype(np.float32)
        if observation.shape != (98,) or not np.all(np.isfinite(observation)):
            raise ValueError("invalid 98-element G1 policy observation")
        input_name = self.session.get_inputs()[0].name
        action = self.session.run(None, {input_name: observation[None, :]})[0].squeeze()
        if action.shape != (29,) or not np.all(np.isfinite(action)):
            raise ValueError("invalid 29-element G1 policy action")
        self.action = action.astype(np.float32)
        self.target_position = self.action * ACTION_SCALE + DEFAULT_ANGLES


def _fall_reasons(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ground_geom_ids: frozenset[int],
    foot_geom_ids: frozenset[int],
) -> tuple[str, ...]:
    reasons = []
    pelvis_id = model.body("pelvis").id
    if float(data.qpos[2]) < 0.55:
        reasons.append("fallen_base_height")
    if float(data.xmat[pelvis_id, 8]) < 0.55:
        reasons.append("fallen_orientation")
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in ground_geom_ids or geom2 in ground_geom_ids:
            other = geom2 if geom1 in ground_geom_ids else geom1
            if other not in foot_geom_ids:
                reasons.append("nonfoot_surface_contact")
                break
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        reasons.append("nan_or_inf")
    return tuple(reasons)


def run_simulation(
    config: SimulationConfig,
    *,
    observe_fsr: bool = True,
) -> SimulationResult:
    """Run one smoke trace entirely in memory; no dataset or output is written."""
    config.validate()
    if config.policy_path is None:
        raise ValueError("policy path is required via --policy or FASTREFLEX_G1_POLICY")
    model, ground_geom_ids = load_g1_model(
        config.terrain,
        config.sink_pattern,
        config.sink_severity,
        config.slip_pattern,
        config.patch_start_x_m,
        config.patch_width_m,
        config.sink_support_pattern,
    )
    data = mujoco.MjData(model)
    controller = UnitreeG1Controller(
        model,
        data,
        config.policy_path,
        config.command_speed_mps,
    )
    foot_geom_ids = frozenset(
        model.geom(name).id
        for names in FOOT_CONTACT_GEOM_NAMES.values()
        for name in names
    )
    soft_patch_geom_ids = soft_sink_geom_ids(
        model,
        config.sink_pattern,
        config.sink_support_pattern,
    )
    low_friction_geom_ids = low_friction_patch_geom_ids(
        model,
        config.slip_pattern,
    )

    timestamp_us: list[int] = []
    imu: list[np.ndarray] = []
    foot_fsr: list[np.ndarray] = []
    contact: list[np.ndarray] = []
    normal_force: list[np.ndarray] = []
    foot_xyz: list[np.ndarray] = []
    foot_velocity: list[np.ndarray] = []
    penetration: list[np.ndarray] = []
    quadrant_contact: list[np.ndarray] = []
    quadrant_normal_force: list[np.ndarray] = []
    quadrant_penetration: list[np.ndarray] = []
    soft_patch_contact: list[np.ndarray] = []
    low_friction_patch_contact: list[np.ndarray] = []
    pre_fall: list[bool] = []
    pelvis_world_z: list[float] = []
    pelvis_orientation: list[np.ndarray] = []
    pelvis_angular_velocity: list[np.ndarray] = []
    pelvis_linear_velocity: list[np.ndarray] = []
    fall_active: list[bool] = []
    first_fall_sample: int | None = None
    first_fall_reasons: tuple[str, ...] = ()
    pelvis_id = model.body("pelvis").id
    minimum_pelvis_height_m = float(data.qpos[2])
    minimum_pelvis_up = float(data.xmat[pelvis_id, 8])
    terminated_by_viewer = False

    viewer_model: mujoco.MjModel | None = None
    viewer_data: mujoco.MjData | None = None
    viewer_context: ContextManager[Any] = nullcontext(None)
    if not config.headless:
        viewer_model, _ = load_g1_model(
            config.terrain,
            config.sink_pattern,
            config.sink_severity,
            config.slip_pattern,
            config.patch_start_x_m,
            config.patch_width_m,
            config.sink_support_pattern,
        )
        viewer_data = mujoco.MjData(viewer_model)
        _copy_integration_state(model, data, viewer_model, viewer_data)
        viewer_context = launch_passive_viewer(viewer_model, viewer_data)

    with viewer_context as viewer:
        wall_start_s = time.monotonic()
        next_viewer_sync_s = 0.0
        if viewer is not None:
            viewer.sync(state_only=True)
            next_viewer_sync_s = VIEWER_SYNC_PERIOD_S

        for physics_step in range(config.total_physics_steps):
            if viewer is not None and not viewer.is_running():
                terminated_by_viewer = True
                break
            controller.apply()
            mujoco.mj_step(model, data)
            controller.update_after_step()

            if viewer is not None:
                if float(data.time) + 1e-12 >= next_viewer_sync_s:
                    assert viewer_model is not None and viewer_data is not None
                    _copy_integration_state(model, data, viewer_model, viewer_data)
                    viewer.sync(state_only=True)
                    next_viewer_sync_s += VIEWER_SYNC_PERIOD_S
                _pace_viewer(float(data.time), wall_start_s)

            if (physics_step + 1) % config.physics_steps_per_sample:
                continue

            minimum_pelvis_height_m = min(
                minimum_pelvis_height_m, float(data.qpos[2])
            )
            minimum_pelvis_up = min(
                minimum_pelvis_up, float(data.xmat[pelvis_id, 8])
            )
            reasons = _fall_reasons(model, data, ground_geom_ids, foot_geom_ids)
            if reasons and first_fall_sample is None:
                first_fall_sample = len(timestamp_us)
                first_fall_reasons = reasons
            exact = read_exact_foot_sample(
                model,
                data,
                ground_geom_ids,
                soft_patch_geom_ids,
                low_friction_geom_ids,
            )
            pelvis_velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                pelvis_id,
                pelvis_velocity,
                0,
            )
            timestamp_us.append(int(round(float(data.time) * 1_000_000.0)))
            imu.append(read_pelvis_imu(model, data))
            if observe_fsr:
                foot_fsr.append(read_virtual_fsr(model, data, ground_geom_ids))
            contact.append(exact.physical_contact)
            normal_force.append(exact.normal_force_n)
            foot_xyz.append(exact.world_xyz)
            foot_velocity.append(exact.world_velocity_xyz)
            penetration.append(exact.contact_penetration_m)
            quadrant_contact.append(exact.quadrant_contact)
            quadrant_normal_force.append(exact.quadrant_normal_force_n)
            quadrant_penetration.append(exact.quadrant_penetration_m)
            soft_patch_contact.append(exact.soft_patch_contact)
            low_friction_patch_contact.append(exact.low_friction_patch_contact)
            pre_fall.append(first_fall_sample is None)
            pelvis_world_z.append(float(data.qpos[2]))
            pelvis_orientation.append(data.qpos[3:7].copy())
            pelvis_angular_velocity.append(pelvis_velocity[:3].copy())
            pelvis_linear_velocity.append(pelvis_velocity[3:].copy())
            fall_active.append(first_fall_sample is not None)

    timestamps = np.asarray(timestamp_us, dtype=np.int64)
    sequence = np.arange(len(timestamps), dtype=np.int64)
    runtime = RuntimeTrace(
        sequence=sequence,
        timestamp_us=timestamps,
        pelvis_imu=np.asarray(imu, dtype=np.float32).reshape(-1, 6),
        foot_fsr=(
            np.asarray(foot_fsr, dtype=np.float32).reshape(-1, len(FSR_CHANNELS))
            if observe_fsr
            else None
        ),
    )
    if runtime.pelvis_imu.shape != (len(timestamps), 6):
        raise RuntimeError("unexpected pelvis IMU shape")
    if observe_fsr and runtime.foot_fsr is not None:
        if runtime.foot_fsr.shape != (len(timestamps), len(FSR_CHANNELS)):
            raise RuntimeError("unexpected virtual FSR shape")
    expected_timestamps = (sequence + 1) * (1_000_000 // config.sensor_rate_hz)
    if not np.array_equal(timestamps, expected_timestamps):
        raise RuntimeError("1 kHz timestamp sequence contains a drop or jitter")

    diagnostics = derive_physical_diagnostics(
        np.asarray(contact, dtype=bool).reshape(-1, 2),
        np.asarray(normal_force, dtype=np.float64).reshape(-1, 2),
        np.asarray(foot_xyz, dtype=np.float64).reshape(-1, 2, 3),
        np.asarray(foot_velocity, dtype=np.float64).reshape(-1, 2, 3),
        np.asarray(penetration, dtype=np.float64).reshape(-1, 2),
        np.asarray(pre_fall),
        np.asarray(pelvis_world_z, dtype=np.float64),
        np.asarray(pelvis_orientation, dtype=np.float64).reshape(-1, 4),
        np.asarray(pelvis_angular_velocity, dtype=np.float64).reshape(-1, 3),
        np.asarray(pelvis_linear_velocity, dtype=np.float64).reshape(-1, 3),
        config.command_speed_mps,
        np.asarray(fall_active, dtype=bool),
        soft_patch_contact=np.asarray(soft_patch_contact, dtype=bool).reshape(-1, 2),
        low_friction_patch_contact=np.asarray(
            low_friction_patch_contact,
            dtype=bool,
        ).reshape(-1, 2),
        quadrant_contact=np.asarray(quadrant_contact, dtype=bool).reshape(-1, 2, 4),
        quadrant_normal_force_n=np.asarray(
            quadrant_normal_force, dtype=np.float64
        ).reshape(-1, 2, 4),
        quadrant_penetration_m=np.asarray(
            quadrant_penetration, dtype=np.float64
        ).reshape(-1, 2, 4),
    )
    metadata: dict[str, object] = {
        "terrain": config.terrain,
        "slip_pattern": config.slip_pattern,
        "sink_pattern": config.sink_pattern,
        "sink_severity": config.sink_severity,
        "sink_support_pattern": config.sink_support_pattern,
        "patch_start_x_m": (
            config.patch_start_x_m
            if config.slip_pattern == "transition"
            or config.sink_pattern.startswith("transition_")
            else None
        ),
        "patch_width_m": (
            config.patch_width_m
            if config.slip_pattern == "transition"
            or config.sink_pattern.startswith("transition_")
            else None
        ),
        "physics_timestep_s": config.physics_timestep_s,
        "physics_rate_hz": int(round(1.0 / config.physics_timestep_s)),
        "sensor_rate_hz": config.sensor_rate_hz,
        "expected_samples": config.expected_samples,
        "actual_samples": len(timestamps),
        "dropped_samples": 0,
        "timestamp_delta_us": 1000,
        "viewer": not config.headless,
        "terminated_by_viewer": terminated_by_viewer,
        "command_speed_mps": config.command_speed_mps,
        "policy_sha256": controller.policy_sha256,
        "policy_upstream_revision": UPSTREAM_REVISION,
        "first_fall_sample": first_fall_sample,
        "first_fall_reasons": first_fall_reasons,
        "minimum_pelvis_height_m": minimum_pelvis_height_m,
        "minimum_pelvis_up": minimum_pelvis_up,
    }
    return SimulationResult(runtime=runtime, diagnostics=diagnostics, metadata=metadata)


def _finite_max(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return None if finite.size == 0 else float(np.max(finite))


def _finite_max_per_foot(values: np.ndarray) -> list[float | None]:
    return [_finite_max(np.asarray(values)[:, side]) for side in range(2)]


def _first_true_per_foot(values: np.ndarray) -> list[int | None]:
    first: list[int | None] = []
    for side in range(2):
        indices = np.flatnonzero(np.asarray(values)[:, side])
        first.append(None if indices.size == 0 else int(indices[0]))
    return first


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values))
    return None if indices.size == 0 else int(indices[0])


def _first_true_any_foot(values: np.ndarray) -> int | None:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("per-foot event arrays must have shape (samples, 2)")
    return _first_true(np.any(array, axis=1))


def _degrees_or_none(value_rad: float | None) -> float | None:
    return None if value_rad is None else float(np.degrees(value_rad))


def summarize_result(result: SimulationResult) -> dict[str, object]:
    """Create a compact JSON-safe smoke summary without saving trace data."""
    diagnostics = result.diagnostics
    summary = dict(result.metadata)
    pelvis_z = diagnostics.pelvis_world_z_m
    angular_speed = np.linalg.norm(
        diagnostics.pelvis_angular_velocity_rad_s,
        axis=1,
    )
    forward_error = diagnostics.forward_velocity_error_m_s
    max_abs_roll = _finite_max(np.abs(diagnostics.pelvis_roll_rad))
    max_abs_pitch = _finite_max(np.abs(diagnostics.pelvis_pitch_rad))
    max_tilt = _finite_max(diagnostics.pelvis_tilt_rad)
    first_degradation = _first_true(diagnostics.sink_degradation_onset)
    first_sink_hazard = _first_true(diagnostics.sink_hazard_onset)
    first_patch_sink = _first_true_any_foot(
        diagnostics.sink_physical_after_patch_onset
    )
    first_slip = _first_true_any_foot(diagnostics.established_slip)
    first_low_friction_patch = _first_true_any_foot(
        diagnostics.low_friction_patch_contact_onset
    )
    first_transition_slip = _first_true(
        diagnostics.any_established_slip_after_patch_onset
    )
    first_post_slip_degradation = None
    if first_transition_slip is not None:
        sample_indices = np.arange(len(diagnostics.sink_degradation_onset))
        later_degradation = np.flatnonzero(
            diagnostics.sink_degradation_onset
            & (sample_indices >= first_transition_slip)
        )
        if later_degradation.size:
            first_post_slip_degradation = int(later_degradation[0])
    dual_phenomenon = bool(
        first_slip is not None
        and first_sink_hazard is not None
        and first_slip < first_sink_hazard
    )
    if dual_phenomenon:
        sink_episode_qualification = "DUAL_PHENOMENON"
    elif first_sink_hazard is not None:
        sink_episode_qualification = "HAZARDOUS_SINK_EPISODE"
    elif first_patch_sink is not None and result.metadata["first_fall_sample"] is None:
        sink_episode_qualification = "BENIGN_SINK_EPISODE"
    elif first_patch_sink is not None:
        sink_episode_qualification = "INCONCLUSIVE_SINK_EPISODE"
    else:
        sink_episode_qualification = None
    slip_transition_qualification = None
    if result.metadata["slip_pattern"] == "transition":
        if first_low_friction_patch is None:
            slip_transition_qualification = "UNUSABLE_SLIP_SCENARIO"
        elif (
            first_sink_hazard is not None
            and (
                first_transition_slip is None
                or first_sink_hazard < first_transition_slip
            )
        ):
            slip_transition_qualification = "UNUSABLE_SLIP_SCENARIO"
        elif first_transition_slip is not None:
            slip_transition_qualification = "CLEAN_SLIP_EVENT"
        elif result.metadata["first_fall_sample"] is not None:
            slip_transition_qualification = "UNUSABLE_SLIP_SCENARIO"
        else:
            slip_transition_qualification = "NO_SLIP_TRANSITION"
    pelvis_z_range = (
        None
        if pelvis_z.size == 0
        else float(np.max(pelvis_z) - np.min(pelvis_z))
    )
    pelvis_z_drop = (
        None
        if pelvis_z.size == 0
        else float(max(0.0, float(pelvis_z[0] - np.min(pelvis_z))))
    )
    summary.update(
        {
            "imu_finite": bool(np.all(np.isfinite(result.runtime.pelvis_imu))),
            "imu_shape": list(result.runtime.pelvis_imu.shape),
            "foot_fsr_shape": (
                None
                if result.runtime.foot_fsr is None
                else list(result.runtime.foot_fsr.shape)
            ),
            "foot_fsr_finite_nonnegative": (
                None
                if result.runtime.foot_fsr is None
                else bool(
                    np.all(np.isfinite(result.runtime.foot_fsr))
                    and np.all(result.runtime.foot_fsr >= 0.0)
                )
            ),
            "contact_samples_per_foot": np.count_nonzero(
                diagnostics.physical_contact, axis=0
            ).tolist(),
            "touchdowns_per_foot": np.count_nonzero(
                diagnostics.touchdown, axis=0
            ).tolist(),
            "max_anchor_drift_m": _finite_max(
                diagnostics.tangential_anchor_drift_m
            ),
            "max_anchor_drift_m_per_foot": _finite_max_per_foot(
                diagnostics.tangential_anchor_drift_m
            ),
            "max_contact_penetration_m": _finite_max(
                diagnostics.contact_penetration_m
            ),
            "max_contact_penetration_m_per_foot": _finite_max_per_foot(
                diagnostics.contact_penetration_m
            ),
            "max_loaded_penetration_change_m": _finite_max(
                diagnostics.loaded_penetration_change_m
            ),
            "max_loaded_penetration_change_m_per_foot": _finite_max_per_foot(
                diagnostics.loaded_penetration_change_m
            ),
            "max_bilateral_loaded_penetration_asymmetry_m": _finite_max(
                diagnostics.bilateral_loaded_penetration_asymmetry_m
            ),
            "max_support_penetration_spread_m": _finite_max(
                diagnostics.support_penetration_spread_m
            ),
            "max_support_penetration_spread_m_per_foot": _finite_max_per_foot(
                diagnostics.support_penetration_spread_m
            ),
            "valid_support_spread_samples_per_foot": np.count_nonzero(
                np.isfinite(diagnostics.support_penetration_spread_m), axis=0
            ).tolist(),
            "support_baselines_per_foot": np.count_nonzero(
                diagnostics.support_baseline_onset, axis=0
            ).tolist(),
            "support_loss_samples": int(
                np.count_nonzero(diagnostics.support_loss_active)
            ),
            "support_loss_samples_per_foot": np.count_nonzero(
                diagnostics.support_loss_active, axis=0
            ).tolist(),
            "first_support_loss_sample_per_foot": _first_true_per_foot(
                diagnostics.support_loss_onset
            ),
            "max_support_loss_ratio": _finite_max(
                diagnostics.support_loss_ratio
            ),
            "max_support_loss_ratio_per_foot": _finite_max_per_foot(
                diagnostics.support_loss_ratio
            ),
            "max_weighted_support_loss": _finite_max(
                diagnostics.weighted_support_loss
            ),
            "established_slip_samples": int(
                np.count_nonzero(diagnostics.established_slip)
            ),
            "established_slip_samples_per_foot": np.count_nonzero(
                diagnostics.established_slip,
                axis=0,
            ).tolist(),
            "sink_physical_samples": int(
                np.count_nonzero(diagnostics.sink_physical_active)
            ),
            "sink_physical_samples_per_foot": np.count_nonzero(
                diagnostics.sink_physical_active, axis=0
            ).tolist(),
            "first_sink_physical_sample_per_foot": _first_true_per_foot(
                diagnostics.sink_physical_onset
            ),
            "first_soft_patch_contact_sample_per_foot": _first_true_per_foot(
                diagnostics.soft_patch_contact_onset
            ),
            "first_low_friction_patch_contact_sample_per_foot": (
                _first_true_per_foot(
                    diagnostics.low_friction_patch_contact_onset
                )
            ),
            "first_sink_physical_after_patch_sample_per_foot": _first_true_per_foot(
                diagnostics.sink_physical_after_patch_onset
            ),
            "first_sink_degradation_sample": first_degradation,
            "first_sink_hazard_sample": first_sink_hazard,
            "sink_episode_qualification": sink_episode_qualification,
            "dual_phenomenon": dual_phenomenon,
            "first_established_slip_sample_per_foot": _first_true_per_foot(
                diagnostics.established_slip_onset
            ),
            "first_established_slip_after_patch_sample_per_foot": (
                _first_true_per_foot(
                    diagnostics.established_slip_after_patch_onset
                )
            ),
            "first_any_established_slip_sample": _first_true(
                diagnostics.any_established_slip_onset
            ),
            "first_any_established_slip_after_patch_sample": first_transition_slip,
            "first_post_slip_degradation_sample": first_post_slip_degradation,
            "slip_transition_qualification": slip_transition_qualification,
            "max_abs_pelvis_roll_deg": _degrees_or_none(max_abs_roll),
            "max_abs_pelvis_pitch_deg": _degrees_or_none(max_abs_pitch),
            "max_pelvis_tilt_deg": _degrees_or_none(max_tilt),
            "pelvis_z_range_m": pelvis_z_range,
            "pelvis_z_drop_from_initial_m": pelvis_z_drop,
            "peak_pelvis_angular_speed_rad_s": _finite_max(angular_speed),
            "mean_pelvis_forward_velocity_m_s": (
                None
                if diagnostics.pelvis_forward_velocity_m_s.size == 0
                else float(np.mean(diagnostics.pelvis_forward_velocity_m_s))
            ),
            "forward_velocity_rmse_m_s": (
                None
                if forward_error.size == 0
                else float(np.sqrt(np.mean(np.square(forward_error))))
            ),
            "pre_event_baseline_sample_range": (
                None
                if not np.any(diagnostics.pre_event_baseline_valid)
                else [
                    int(np.flatnonzero(diagnostics.pre_event_baseline_valid)[0]),
                    int(np.flatnonzero(diagnostics.pre_event_baseline_valid)[-1] + 1),
                ]
            ),
            "pre_event_mean_pelvis_z_m": (
                None
                if not np.any(diagnostics.pre_event_baseline_valid)
                else float(
                    np.mean(
                        diagnostics.pelvis_world_z_m[
                            diagnostics.pre_event_baseline_valid
                        ]
                    )
                )
            ),
            "pre_event_mean_pelvis_tilt_deg": (
                None
                if not np.any(diagnostics.pre_event_baseline_valid)
                else float(
                    np.degrees(
                        np.mean(
                            diagnostics.pelvis_tilt_rad[
                                diagnostics.pre_event_baseline_valid
                            ]
                        )
                    )
                )
            ),
            "pre_event_mean_forward_velocity_m_s": (
                None
                if not np.any(diagnostics.pre_event_baseline_valid)
                else float(
                    np.mean(
                        diagnostics.pelvis_forward_velocity_m_s[
                            diagnostics.pre_event_baseline_valid
                        ]
                    )
                )
            ),
            "pre_event_mean_angular_speed_rad_s": (
                None
                if not np.any(diagnostics.pre_event_baseline_valid)
                else float(
                    np.mean(
                        np.linalg.norm(
                            diagnostics.pelvis_angular_velocity_rad_s[
                                diagnostics.pre_event_baseline_valid
                            ],
                            axis=1,
                        )
                    )
                )
            ),
            "max_pelvis_z_drop_from_pre_event_m": _finite_max(
                diagnostics.pelvis_z_drop_from_pre_event_m[
                    diagnostics.pre_fall_valid
                ]
            ),
            "max_tilt_change_from_pre_event_deg": _degrees_or_none(
                _finite_max(
                    diagnostics.pelvis_tilt_change_from_pre_event_rad[
                        diagnostics.pre_fall_valid
                    ]
                )
            ),
            "max_forward_velocity_drop_from_pre_event_m_s": _finite_max(
                diagnostics.forward_velocity_drop_from_pre_event_m_s[
                    diagnostics.pre_fall_valid
                ]
            ),
            "max_angular_speed_change_from_pre_event_rad_s": _finite_max(
                diagnostics.pelvis_angular_speed_change_from_pre_event_rad_s[
                    diagnostics.pre_fall_valid
                ]
            ),
            "loaded_contact_samples_per_foot": np.count_nonzero(
                diagnostics.loaded_contact, axis=0
            ).tolist(),
            "loaded_contact_imbalance_samples": int(
                abs(
                    np.count_nonzero(diagnostics.loaded_contact[:, 0])
                    - np.count_nonzero(diagnostics.loaded_contact[:, 1])
                )
            ),
        }
    )
    return summary
