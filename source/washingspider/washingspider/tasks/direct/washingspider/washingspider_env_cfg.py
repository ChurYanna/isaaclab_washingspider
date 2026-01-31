# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configs for the WashingSpider Direct RL task.

This file is the *env_cfg_entry_point* referenced by gym.register().  The registry
currently points to:  ...washingspider_env_cfg:WashingspiderEnvCfg

So we MUST provide a class named `WashingspiderEnvCfg` (see bottom of file).
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # NOTE: if your base rigid-body name is NOT `base_link`, change it here to match the USD.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


@configclass
class WashingspiderFlatEnvCfg(DirectRLEnvCfg):
    # --- env ---
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 64  # NOTE: will be overridden in Rough; must match env.py observation concatenation
    state_space = 0

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",   
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # --- events ---
    events: EventCfg = EventCfg()

    # --- robot (WashingSpider) ---
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(
            # TODO: make this path relative if you plan to publish/share the extension
            usd_path="/home/aaa/isaacsim/selfdemo/washingspider/source/washingspider/washingspider/assets/usd/WashingSpider.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,  # 先关自碰撞更容易跑通；之后想开再开
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),  # rough 上初始抬高一点
            # joint_pos: 先不写：默认用 USD 里的默认角度；站不稳再补“站姿表”
        ),
        actuators={
            # 12个驱动关节：joint_bl1..joint_fr3（兼容你可能没 joint_ 前缀的情况）
            "legs": ImplicitActuatorCfg(
                joint_names_expr=["joint_(bl|br|fl|fr)(1|2|3)"],
                stiffness=30.0,
                damping=1.5,
                effort_limit=50.0,
                velocity_limit=20.0,
            ),
            #  "feet_passive": ImplicitActuatorCfg(
            #     joint_names_expr=[r"^joint_(bl|br|fl|fr)4(x|y)$"],
            #     joint_target_type="none",       # 关键：不控制这 8 个
            # ),
        },
    )

    # contact sensor: track contacts/air-time for reward + termination
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # --- reward scales ---
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    # joint_accel_reward_scale = -2.5e-7
    joint_accel_reward_scale = 0.0
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -2.0
    flat_orientation_reward_scale = -5.0

    # --- body height ---
    base_height_target = 0.40      # 你自己调：0.25~0.40 先试
    base_height_sigma = 0.12      # 容忍范围（越小越严格）
    base_height_reward_scale = 50.0 # 先从 0.5~2.0 试

    base_contact_reward_scale = -50.0   # 先用 -20 ~ -100 之间试


@configclass
class WashingspiderRoughEnvCfg(WashingspiderFlatEnvCfg):
    # --- env ---
    # NOTE: This must match env.py: concatenation of (vels, gravity, commands, joint pos/vel, height scan, actions)
    observation_space = 251

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # height scanner for perceptive locomotion
    # NOTE: if your base prim is `/Robot/base` not `/Robot/base_link`, change prim_path accordingly.
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0


# -------------------------------------------------------------------------
# IMPORTANT: this name MUST exist because gym.register points here:
#   env_cfg_entry_point: "...washingspider_env_cfg:WashingspiderEnvCfg"
# -------------------------------------------------------------------------
@configclass
class WashingspiderEnvCfg(WashingspiderRoughEnvCfg):
    """Default env cfg exported to the registry (currently using Rough)."""
    pass
