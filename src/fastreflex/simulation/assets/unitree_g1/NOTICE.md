# Unitree G1 asset provenance

The MJCF and the 36 mesh files in this directory were explicitly migrated from
`/d/shin/Infineon/simulation/unitree_mujoco/unitree_robots/g1/` at legacy commit
`4194af1e0d8db8d113609c11879713c29a583261`.

They originate from Unitree Robotics' `unitree_mujoco` distribution and remain
under the accompanying BSD 3-Clause `LICENSE`. The official upstream is
<https://github.com/unitreerobotics/unitree_mujoco>.

Local changes to `g1.xml` are limited to removing legacy research additions for
bilateral foot IMUs and ankle force/torque sensors. The named sole collision
geometries, rigid-body dynamics, inertials, joints, actuators, and pelvis IMU
remain unchanged. `scene.xml` names the single ground plane `terrain` and fixes
the baseline physics timestep to 0.5 ms.

The G1 walking policy is not included in this directory or repository. It is a
user-supplied artifact from `unitreerobotics/unitree_rl_mjlab`.
