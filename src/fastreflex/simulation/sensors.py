"""Observer-only candidate sensors derived from existing MuJoCo contacts."""

from __future__ import annotations

import mujoco
import numpy as np

from .hazards import (
    FOOT_BODY_NAMES,
    FOOT_CONTACT_GEOM_NAMES,
    SIDES,
    foot_quadrant_index,
)


FSR_CHANNELS = (
    "left_front_left",
    "left_front_right",
    "left_rear_left",
    "left_rear_right",
    "right_front_left",
    "right_front_right",
    "right_rear_left",
    "right_rear_right",
)
FSR_UNIT = "N"


def fsr_quadrant_index(local_x_m: float, local_y_m: float) -> int:
    """Map foot-local +x front and +y left to the frozen four-channel order."""
    return foot_quadrant_index(local_x_m, local_y_m)


def read_virtual_fsr(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ground_geom_ids: frozenset[int],
) -> np.ndarray:
    """Read bilateral FSR4 from existing sole-ground contacts, in Newtons.

    MuJoCo's ``mj_contactForce`` returns force:torque in the contact frame.
    The installed MuJoCo API defines contact-frame axis 0 as the normal, so
    ``wrench[0]`` is the nonnegative scalar normal load.  Contact positions are
    transformed from world coordinates into each ankle-roll (sole) body frame;
    no dynamics, controller, geom, terrain, or oracle state is modified.
    """
    foot_body_ids = tuple(model.body(name).id for name in FOOT_BODY_NAMES)
    foot_geom_ids = tuple(
        frozenset(model.geom(name).id for name in FOOT_CONTACT_GEOM_NAMES[side])
        for side in SIDES
    )
    values = np.zeros(8, dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 not in ground_geom_ids and geom2 not in ground_geom_ids:
            continue
        sole_geom_id = geom2 if geom1 in ground_geom_ids else geom1
        for side_index, side_geom_ids in enumerate(foot_geom_ids):
            if sole_geom_id not in side_geom_ids:
                continue
            body_id = foot_body_ids[side_index]
            world_delta = np.asarray(contact.pos) - data.xpos[body_id]
            local_position = data.xmat[body_id].reshape(3, 3).T @ world_delta
            wrench.fill(0.0)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            normal_force_n = max(0.0, float(wrench[0]))
            channel = 4 * side_index + fsr_quadrant_index(
                float(local_position[0]), float(local_position[1])
            )
            values[channel] += normal_force_n
            break
    sample = values.astype(np.float32)
    if sample.shape != (8,) or not np.all(np.isfinite(sample)) or np.any(sample < 0.0):
        raise ValueError("virtual FSR sample must be eight finite nonnegative values")
    return sample
