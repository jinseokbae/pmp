# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from errno import EEXIST
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

DOF_OFFSETS = [
    0,  # 'right_wrist_PRO', 'right_wrist_UDEV', 'right_wrist_FLEX',
    3,  # 'right_thumb_ABD',
    4,  # 'right_thumb_MCP'
    5,  # 'right_thumb_PIP'
    6,  # 'right_thumb_DIP'
    7,  # 'right_index_ABD
    8,  # 'right_index_MCP'
    9,  # 'right_index_PIP',
    10,  # 'right_index_DIP',
    11,  # 'right_middle_MCP',
    12,  # 'right_middle_PIP',
    13,  # 'right_middle_DIP',
    14,  # 'right_ring_ABD',
    15,  # 'right_ring_MCP',
    16,  # 'right_ring_PIP',
    17,  # 'right_ring_DIP',
    18,  # 'right_pinky_ABD',
    19,  # 'right_pinky_MCP',
    20,  # 'right_pinky_PIP',
    21,  # 'right_pinky_DIP',
    22,  # END
    # 22, # 'left_wrist_PRO', 'left_wrist_UDEV', 'left_wrist_FLEX',
    # 25, # 'left_thumb_ABD',
    # 26, # 'left_thumb_MCP',
    # 27, # 'left_thumb_PIP',
    # 28, # 'left_thumb_DIP',
    # 29, # 'left_index_ABD',
    # 30, # 'left_index_MCP',
    # 31, # 'left_index_PIP',
    # 32, # 'left_index_DIP',
    # 33, # 'left_middle_MCP',
    # 34, # 'left_middle_PIP',
    # 35, # 'left_middle_DIP',
    # 35, # 'left_ring_ABD',
    # 37, # 'left_ring_MCP',
    # 38, # 'left_ring_PIP',
    # 39, # 'left_ring_DIP',
    # 40, # 'left_pinky_ABD',
    # 41, # 'left_pinky_MCP',
    # 42, # 'left_pinky_PIP',
    # 43, # 'left_pinky_DIP',
    # 44, # END
]

MODE = "right"
# MODE = "left"
CAPSULE_HALF_LEN = 0.3

SAVE_NUM = 1000000

# AMP + MPL
# NUM_OBS = 16 + 13 + 23 # [dof_pos, dof_vel, scenario-specific]
# NUM_OBS = 16 + 13 + 6 + 20 + 60
# NUM_OBS = 16 + 13 + 4 + 6 + 15 + 20 + 4 + 3 #--> Dec-24 (A), (B)
# NUM_OBS = 16 + 13 + 4 + 6 + 15 + 20 + 20 + 3 #--> Dec-24 (C)
NUM_OBS = 16 + 13 + 4 + 6 + 15 + 20 + 3  # --> Dec-24 (D)
NUM_ACTIONS = 13 - 3  # we sync all the mcp joints
KEY_BODY_NAMES = ["%s_palm" % MODE]


# 5) pullcart
MIN_TARGET_CART_SPEED = 0.5
TARGET_CART_DIST = 1.0

# SAVE MAX Horizon
SAVE_MAX_HORIZON = 30

# AXIAL_FORCES_RATIO (percentage on how frequently axial forces are applied)
AXIAL_FORCES_RATIO = 0.9

FINGER_BODY_NAMES = [
    "thumb0",
    "thumb1",
    "thumb2",
    "thumb3",
    "index0",
    "index1",
    "index2",
    "index3",
    "middle0",
    "middle1",
    "middle2",
    "middle3",
    "ring0",
    "ring1",
    "ring2",
    "ring3",
    "pinky0",
    "pinky1",
    "pinky2",
    "pinky3",
]


class HandGraspGym(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        # self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self._contact_bodies = [MODE + "_" + k for k in self.cfg["env"]["contactBodies"]]

        self.save_mode = False #self.cfg["env"]["testMode"]
        self.save_wo_wrist = False

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.debug_mode = self.cfg["env"]["test"]

        self._approx_pd_force = None

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt
        self.actions = torch.zeros(self.num_envs, 13).to(self.device)
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # sensor
        # sensors_per_env = 1 # attached in capsule
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._humanoid_root_states = self._root_states.view(self.num_envs, -1, 13)[:, 0]  # (jsbae)
        self._else_root_states = self._root_states.view(self.num_envs, -1, 13)[:, 1:]
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states = self._initial_root_states.view(self.num_envs, -1, 13)  # (jsbae)
        # self._initial_root_states[:, 0, 7:13] = 0 # (jsbae)
        self._initial_humanoid_root_states = self._initial_root_states[:, 0]
        self._initial_else_root_states = self._initial_root_states[:, 1:]
        self._initial_else_root_states[:, :, 7:13] = 0
        self._initial_root_states = self._initial_root_states.view(self._root_states.shape)  # (jsbae)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._humanoid_dof_state = self._dof_state[:, : self.humanoid_num_dof]  # (jsbae)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._humanoid_dof_pos = self._dof_pos[:, : self.humanoid_num_dof]  # (jsbae)
        self._else_dof_pos = self._dof_pos[:, self.humanoid_num_dof :]  # (jsbae)
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self._humanoid_dof_vel = self._dof_vel[:, : self.humanoid_num_dof]  # (jsbae)
        self._else_dof_vel = self._dof_vel[:, self.humanoid_num_dof :]  # (jsbae)

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)  # (jsbae)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, -1, 13)[..., 0:3]  # (jsbae)
        self._humanoid_rigid_body_pos = self._rigid_body_pos[:, : self.humanoid_num_bodies]  # (jsbae)
        self._else_rigid_body_pos = self._rigid_body_pos[:, self.humanoid_num_bodies :]  # (jsbae)
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, -1, 13)[..., 3:7]  # (jsbae)
        self._humanoid_rigid_body_rot = self._rigid_body_rot[:, : self.humanoid_num_bodies]  # (jsbae)
        self._else_rigid_body_rot = self._rigid_body_rot[:, self.humanoid_num_bodies :]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, -1, 13)[..., 7:10]  # (jsbae)
        self._humanoid_rigid_body_vel = self._rigid_body_vel[:, : self.humanoid_num_bodies]  # (jsbae)
        self._else_rigid_body_vel = self._rigid_body_vel[:, self.humanoid_num_bodies :]  # (jsbae)
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, -1, 13)[..., 10:13]  # (jsbae)
        self._humanoid_rigid_body_ang_vel = self._rigid_body_ang_vel[:, : self.humanoid_num_bodies]  # (jsbae)
        self._else_rigid_body_ang_vel = self._rigid_body_ang_vel[:, self.humanoid_num_bodies :]
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, -1, 3)  # (jsbae)
        self._humanoid_contact_forces = self._contact_forces[:, : self.humanoid_num_bodies]  # (jsbae)
        self._else_contact_forces = self._contact_forces[:, self.humanoid_num_bodies :]  # (jsbae)

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        # (jsbae) To set action vector
        self.cur_targets = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)

        if self.viewer != None:
            self._init_camera()

        self.respawn_handle_pos = torch.FloatTensor([7.0000e-01, 7.9678e-07, 9.3000e-01]).to(self.device)

        self.initial_handle_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.initial_handle_rot = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)

        self.initial_wrist_dof = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

        self.is_axial_forces = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
        self.handle_forces = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.handle_forces_mag = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)
        self.handle_torques = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.wrist_target_dof_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )  # 0, 2 --> random, 1 --> 0

        self.init_rfingers_rot = (
            torch.FloatTensor(self.cfg["misc"]["init_rfingers_rot"])[None].to(self.device).repeat(self.num_envs, 1, 1)
        )
        self.init_lfingers_rot = (
            torch.FloatTensor(self.cfg["misc"]["init_lfingers_rot"])[None].to(self.device).repeat(self.num_envs, 1, 1)
        )

        if self.save_mode:
            self.handle_force_limit = [50, 70]
            self.handle_torque_limit = [-10, 10]
        else:
            self.handle_force_limit = [50, 70]
            self.handle_torque_limit = [-20, 20]

        self.apply_delay = 0

        self.handle_end1_offset = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.handle_end2_offset = torch.zeros_like(self.handle_end1_offset)
        self.handle_end1_offset[:, 0] = CAPSULE_HALF_LEN
        self.handle_end2_offset[:, 0] = -CAPSULE_HALF_LEN

        self.hand_contact_bodies = [
            "palm",
            "thumb0",
            "thumb1",
            "thumb2",
            "thumb3",
            "index0",
            "index1",
            "index2",
            "index3",
            "middle0",
            "middle1",
            "middle2",
            "middle3",
            "ring0",
            "ring1",
            "ring2",
            "ring3",
            "pinky0",
            "pinky1",
            "pinky2",
            "pinky3",
        ]
        self.unit_x = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.unit_x[:, 0] = 1

        self.unit_y = torch.zeros_like(self.unit_x)
        self.unit_y[:, 1] = 1

        self.unit_z = torch.zeros_like(self.unit_x)
        self.unit_z[:, 2] = 1

        self.default_rfingers_facing_dir = torch.stack([-self.unit_z] * 4 + [self.unit_y] * 16, dim=1)
        self.default_lfingers_facing_dir = torch.stack([-self.unit_z] * 4 + [-self.unit_y] * 16, dim=1)

        # to measure average contact forces sensed by capsule
        self.mean_object_contact_forces = 0
        self.temp_object_contact_forces = torch.zeros(self.num_envs, self.max_episode_length).to(self.device)

        if self.save_mode:
            NUM_OBS_FOR_SAVING = NUM_OBS - 9 - 3 if self.save_wo_wrist else NUM_OBS - 3  # exclude wrist target dof pos
            NUM_ACTIONS_FOR_SAVING = 10 if self.save_wo_wrist else 13

            self.save_buffer = torch.zeros(SAVE_NUM, NUM_OBS_FOR_SAVING + NUM_ACTIONS_FOR_SAVING).to(self.device)
            self.temp_save_buffer = torch.zeros(
                self.num_envs, self.max_episode_length, NUM_OBS_FOR_SAVING + NUM_ACTIONS_FOR_SAVING
            ).to(self.device)

            self.save_count = 0
            self.temp_save_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        return

    def update_epoch(self, epoch_num):
        super().update_epoch(epoch_num)
        if self.epoch_num < 2500:
            self.handle_force_limit = [50, 70]
            self.handle_torque_limit = [-20, 20]
        elif self.epoch_num < 5000:
            self.handle_force_limit = [50, 70]
            self.handle_torque_limit = [-20, 20]
        elif self.epoch_num < 7500:
            self.handle_force_limit = [50, 80]
            self.handle_torque_limit = [-23.3, 23.3]
        elif self.epoch_num < 10000:
            self.handle_force_limit = [50, 80]
            self.handle_torque_limit = [-23.3, 23.3]
        elif self.epoch_num < 12500:
            self.handle_force_limit = [50, 90]
            self.handle_torque_limit = [-26.6, 26.6]
        elif self.epoch_num < 15000:
            self.handle_force_limit = [50, 90]
            self.handle_torque_limit = [-26.6, 26.6]
        else:
            self.handle_foce_limite = [50, 100]
            self.handle_torque_limit = [-30, 30]

        return

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        return

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self.humanoid_handles[i]

            for j in range(self.humanoid_num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(col[0], col[1], col[2])
                )

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "mjcf/%s_mpl.xml" % MODE

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = True
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # we re-arrange filter to mitigate contact calculation
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(humanoid_asset)
        for k in range(len(rigid_shape_prop)):
            if k in range(0, 2):  # lower arm / palm
                rigid_shape_prop[k].filter = 1  # .... 000001
            elif k in range(2, 6):  # thumb
                rigid_shape_prop[k].filter = 3  # .... 000011
            elif k in range(6, 10):  # index
                rigid_shape_prop[k].filter = 5  # .... 000101
            elif k in range(10, 14):  # middle
                rigid_shape_prop[k].filter = 9  # .... 001001
            elif k in range(14, 18):  # ring
                rigid_shape_prop[k].filter = 17  # .... 010001
            elif k in range(18, 22):  # pinky
                rigid_shape_prop[k].filter = 33  # .... 100001
            # contact offset
            if k in range(1, 22):
                rigid_shape_prop[k].contact_offset = 0.002
        self.gym.set_asset_rigid_shape_properties(humanoid_asset, rigid_shape_prop)

        self.rigid_body_names = self.gym.get_asset_rigid_body_names(humanoid_asset)
        self.dof_names = self.gym.get_asset_dof_names(humanoid_asset)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)
        self.motor_efforts[self.motor_efforts < 1] = 40  # (jsbae)maximum for hand joints

        self.humanoid_num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)  # (jsbae)
        self.humanoid_num_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)  # (jsbae)
        self.humanoid_num_dof = self.gym.get_asset_dof_count(humanoid_asset)  # (jsbae)
        self.humanoid_num_joints = self.gym.get_asset_joint_count(humanoid_asset)  # (jsbae) NOT USING
        self.humanoid_num_actuators = self.gym.get_asset_actuator_count(humanoid_asset)  # (jsbae)
        self.humanoid_num_tendons = self.gym.get_asset_tendon_count(humanoid_asset)  # (jsbae)

        # (jsbae) save acuated dof indices as this model is underactuated
        actuated_dof_names = [
            self.gym.get_asset_actuator_joint_name(humanoid_asset, i) for i in range(self.humanoid_num_actuators)
        ]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(humanoid_asset, name) for name in actuated_dof_names]
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)

        mcp_names = ["index_MCP", "middle_MCP", "ring_MCP", "pinky_MCP"]
        abd_names = ["index_ABD", "pinky_ABD"]
        thumb_names = ["thumb_ABD", "thumb_MCP", "thumb_PIP", "thumb_DIP"]
        mcp_names = [MODE + "_" + n for n in mcp_names]
        abd_names = [MODE + "_" + n for n in abd_names]
        thumb_names = [MODE + "_" + n for n in thumb_names]

        mcp_indices = [self.dof_names.index(n) for n in mcp_names]
        abd_indices = [self.dof_names.index(n) for n in abd_names]
        thumb_indices = [self.dof_names.index(n) for n in thumb_names]
        self.mcp_indices = to_torch(mcp_indices, dtype=torch.long, device=self.device)
        self.abd_indices = to_torch(abd_indices, dtype=torch.long, device=self.device)
        self.thumb_indices = to_torch(thumb_indices, dtype=torch.long, device=self.device)

        if MODE == "right":
            self.rfingers_body_indices = [self.rigid_body_names.index(MODE + "_" + name) for name in FINGER_BODY_NAMES]
            self.rfingers_body_indices = to_torch(self.rfingers_body_indices, dtype=torch.long, device=self.device)
        elif MODE == "left":
            self.lfingers_body_indices = [self.rigid_body_names.index(MODE + "_" + name) for name in FINGER_BODY_NAMES]
            self.lfingers_body_indices = to_torch(self.lfingers_body_indices, dtype=torch.long, device=self.device)

        wrist_idx = [
            actuated_dof_names.index("%s_wrist_FLEX" % MODE),
            actuated_dof_names.index("%s_wrist_PRO" % MODE),
            actuated_dof_names.index("%s_wrist_UDEV" % MODE),
        ]
        self.motor_efforts[wrist_idx] = 40

        # tendon set up
        if self.humanoid_num_tendons > 0:
            limit_stiffness = 30
            t_damping = 0.1
            relevant_tendons = [
                "T_%s_index32_cpl" % MODE,
                "T_%s_index21_cpl" % MODE,
                "T_%s_middle32_cpl" % MODE,
                "T_%s_middle21_cpl" % MODE,
                "T_%s_ring32_cpl" % MODE,
                "T_%s_ring21_cpl" % MODE,
                "T_%s_pinky32_cpl" % MODE,
                "T_%s_pinky21_cpl" % MODE,
            ]
            tendon_props = self.gym.get_asset_tendon_properties(humanoid_asset)

            for i in range(self.humanoid_num_tendons):
                for rt in relevant_tendons:
                    if self.gym.get_asset_tendon_name(humanoid_asset, i) == rt:
                        tendon_props[i].limit_stiffness = limit_stiffness
                        tendon_props[i].damping = t_damping
            self.gym.set_asset_tendon_properties(humanoid_asset, tendon_props)
            # tendon set up end

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(1.0, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0, 0, 0, 1.0)

        self.start_rotation = torch.tensor(
            [start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device
        )

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        self.num_dof = self.humanoid_num_dof  # (jsbae)
        aggregate_mode = 0

        capsule_asset_options = gymapi.AssetOptions()
        capsule_asset_options.density = 5000  # mass ~ 0.02 ** 2 * np.pi * 0.6 * 5000 ~ 3.76 kg
        capsule_asset_options.disable_gravity = True

        capsule_pose = gymapi.Transform(gymapi.Vec3(0, 0, 1.5), gymapi.Quat(0, 0, 0.7071068, 0.7071068))

        capsule_asset = self.gym.create_capsule(
            self.sim, 0.02, CAPSULE_HALF_LEN, capsule_asset_options
        )  # radius and half-length
        capsule_rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(capsule_asset)
        capsule_rigid_shape_prop[0].contact_offset = 0.002
        self.gym.set_asset_rigid_shape_properties(capsule_asset, capsule_rigid_shape_prop)
        capsule_rigid_body_names = self.gym.get_asset_rigid_body_names(capsule_asset)

        # sensor_props = gymapi.ForceSensorProperties()
        # sensor_props.enable_forward_dynamics_forces = True
        # sensor_props.enable_constraint_solver_forces = False
        # sensor_props.use_world_frame = True

        # capsule_idx = self.gym.find_asset_rigid_body_index(capsule_asset, "capsule")
        # sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))

        # self.gym.create_asset_force_sensor(capsule_asset, capsule_idx, sensor_pose, sensor_props)

        self.else_rigid_body_names = capsule_rigid_body_names

        if self.debug_mode:
            sphere_asset_options = gymapi.AssetOptions()
            sphere_pose = gymapi.Transform()
            sphere_pose.p = gymapi.Vec3(0, 0, 0.9)
            sphere_asset_options.disable_gravity = True
            sphere_asset = self.gym.create_sphere(self.sim, 0.01, sphere_asset_options)
            wrist_asset = self.gym.create_sphere(self.sim, 0.02, sphere_asset_options)
            target_asset = self.gym.create_sphere(self.sim, 0.03, sphere_asset_options)
            contact_asset = self.gym.create_sphere(self.sim, 0.007, sphere_asset_options)
            contact_asset = self.gym.create_capsule(
                self.sim, 0.001, 0.005, capsule_asset_options
            )  # radius and half-length
            self.num_bodies = self.humanoid_num_bodies + 1 + 1 + (1 + 4 * 5) + (4 * 5) + 2  # + 2
            self.num_shapes = self.humanoid_num_shapes + 1 + 1 + (1 + 4 * 5) + (4 * 5) + 2  # + 2
        else:
            self.num_bodies = self.humanoid_num_bodies + 1
            self.num_shapes = self.humanoid_num_shapes + 1

        aggregate_mode = 1
        max_agg_bodies = self.num_bodies
        max_agg_shapes = self.num_shapes

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            # contact_filter = 0 --> enable all self-collision
            contact_filter = -1

            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, contact_filter, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.humanoid_num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863)
                )

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            if self._pd_control:
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                dof_prop["effort"][self.actuated_dof_indices.detach().cpu().numpy()] = (
                    self.motor_efforts.detach().cpu().numpy()
                )  # (jsbae) make sure every dof has effort greater than 0
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

            capsule_handle = self.gym.create_actor(env_ptr, capsule_asset, capsule_pose, "handle", i, 0, 0)

            # debug
            # hfilter = [self.gym.get_actor_rigid_shape_properties(env_ptr, handle)[k].filter for k in range(22)]
            # cfilter = [self.gym.get_actor_rigid_shape_properties(env_ptr, capsule_handle)[k].filter for k in range(len(self.gym.get_actor_rigid_shape_properties(env_ptr, capsule_handle)))]

            self.gym.set_rigid_body_color(
                env_ptr, capsule_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.6863, 0.549)
            )

            if self.debug_mode:
                virt_capsule_handle = self.gym.create_actor(
                    env_ptr, capsule_asset, capsule_pose, "virt_handle", i + 1 * self.num_envs, 0, 0
                )
                self.gym.set_rigid_body_color(
                    env_ptr, virt_capsule_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6863, 0.6863, 0.6863)
                )

                wrist_handle = self.gym.create_actor(
                    env_ptr, wrist_asset, sphere_pose, "wrist", i + 2 * self.num_envs, 0, 0
                )
                self.gym.set_rigid_body_color(env_ptr, wrist_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0, 0))

                for k in range(20):
                    sphere_handle = self.gym.create_actor(
                        env_ptr, sphere_asset, sphere_pose, "hand%02d" % k, i + (k + 3) * self.num_envs, 0, 0
                    )
                    if k // 4 == 0:
                        self.gym.set_rigid_body_color(
                            env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.6863, 0.3)
                        )
                    if k // 4 == 1:
                        self.gym.set_rigid_body_color(
                            env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.6863)
                        )
                    if k // 4 == 2:
                        self.gym.set_rigid_body_color(
                            env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6863, 0.3, 0.6863)
                        )
                    if k // 4 == 3:
                        self.gym.set_rigid_body_color(
                            env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6863, 0.6863, 0.3)
                        )
                    if k // 4 == 4:
                        self.gym.set_rigid_body_color(
                            env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.9)
                        )

                # contact
                for k in range(20):
                    sphere_handle = self.gym.create_actor(
                        env_ptr, contact_asset, sphere_pose, "contact%02d" % k, i + (k + 25) * self.num_envs, 0, 0
                    )
                    if k // 4 == 0:
                        self.gym.set_rigid_body_color(
                            env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.9863, 0.3)
                        )
                    if k // 4 == 1:
                        self.gym.set_rigid_body_color(
                            env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.9863)
                        )
                    if k // 4 == 2:
                        self.gym.set_rigid_body_color(
                            env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.9863, 0.3, 0.9863)
                        )
                    if k // 4 == 3:
                        self.gym.set_rigid_body_color(
                            env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.9863, 0.9863, 0.3)
                        )
                    if k // 4 == 4:
                        self.gym.set_rigid_body_color(
                            env_ptr, sphere_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 1.0)
                        )

                # end_handle = self.gym.create_actor(env_ptr, target_asset, sphere_pose, "end1_next", i + 25 * self.num_envs, 0, 0)
                # self.gym.set_rigid_body_color(env_ptr, end_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.5, 0, 0))

                # end_handle = self.gym.create_actor(env_ptr, target_asset, sphere_pose, "end2_next", i + 26 * self.num_envs, 0, 0)
                # self.gym.set_rigid_body_color(env_ptr, end_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.5, 0, 0))

                end_handle = self.gym.create_actor(
                    env_ptr, target_asset, sphere_pose, "end1", i + 23 * self.num_envs, 0, 0
                )
                self.gym.set_rigid_body_color(env_ptr, end_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.0, 0))

                end_handle = self.gym.create_actor(
                    env_ptr, target_asset, sphere_pose, "end2", i + 24 * self.num_envs, 0, 0
                )
                self.gym.set_rigid_body_color(env_ptr, end_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0, 1.0, 0))

            if aggregate_mode >= 1:
                self.gym.end_aggregate(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.humanoid_num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.humanoid_dof_stiffness = to_torch(dof_prop["stiffness"], device=self.device)
        self.humanoid_dof_damping = to_torch(dof_prop["damping"], device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)

        if self._pd_control:
            self._build_pd_action_offset_scale()

        rigid_body_prop = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
        if MODE == "right":
            rfingers_rigid_body_com = torch.zeros(
                (self.rfingers_body_indices.shape[0], 3), dtype=torch.float, device=self.device
            )
            rpalm_body_index = self.rigid_body_names.index("%s_palm" % MODE)
            for k in range(self.rfingers_body_indices.shape[0]):
                rfingers_rigid_body_com[k, 0] = rigid_body_prop[k + rpalm_body_index + 1].com.x
                rfingers_rigid_body_com[k, 1] = rigid_body_prop[k + rpalm_body_index + 1].com.y
                rfingers_rigid_body_com[k, 2] = rigid_body_prop[k + rpalm_body_index + 1].com.z
            self.rfingers_rigid_body_com = rfingers_rigid_body_com[None].repeat(self.num_envs, 1, 1)
        else:
            lfingers_rigid_body_com = torch.zeros(
                (self.lfingers_body_indices.shape[0], 3), dtype=torch.float, device=self.device
            )
            lpalm_body_index = self.rigid_body_names.index("%s_palm" % MODE)
            for k in range(self.lfingers_body_indices.shape[0]):
                lfingers_rigid_body_com[k, 0] = rigid_body_prop[k + lpalm_body_index + 1].com.x
                lfingers_rigid_body_com[k, 1] = rigid_body_prop[k + lpalm_body_index + 1].com.y
                lfingers_rigid_body_com[k, 2] = rigid_body_prop[k + lpalm_body_index + 1].com.z
            self.lfingers_rigid_body_com = lfingers_rigid_body_com[None].repeat(self.num_envs, 1, 1)

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(DOF_OFFSETS) - 1

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = DOF_OFFSETS[j]
            dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

            if dof_size == 3:
                if dof_offset == 0:
                    # Right wrist, Left wrist
                    curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
                    curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
                    curr_mid = 0.5 * (curr_high + curr_low)

                    # extend the action range to be a bit beyond the joint limits so that the motors
                    # don't lose their strength as they approach the joint limits
                    curr_scale = 0.5 * (curr_high - curr_low)
                    curr_low = curr_mid - curr_scale
                    curr_high = curr_mid + curr_scale
                    lim_low[dof_offset : (dof_offset + dof_size)] = curr_low
                    lim_high[dof_offset : (dof_offset + dof_size)] = curr_high
                else:
                    # TODO (jsbae) why -pi and pi for spherical joint?
                    lim_low[dof_offset : (dof_offset + dof_size)] = -np.pi
                    lim_high[dof_offset : (dof_offset + dof_size)] = np.pi

            elif dof_size == 1:
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                if self.dof_names[dof_offset] in ["right_elbow", "left_elbow", "right_knee", "left_knee"]:
                    curr_scale = 0.7 * (curr_high - curr_low)
                else:
                    if dof_offset in [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 18, 19]:
                        curr_scale = 0.8 * (curr_high - curr_low)
                    else:
                        curr_scale = 0.7 * (curr_high - curr_low)  # we don't care about it
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reward(self, actions):
        rew = compute_humanoid_reward(self.obs_buf)
        self.rew_buf[:] = rew + self._compute_scenario_specific_reward(self.obs_buf, actions)
        return

    def _compute_scenario_specific_reward(self, obs, actions):
        rew = torch.zeros(self.num_envs).to(self.device)

        ###############################################################333
        # # (A) handle reward
        handle_idx = self.else_rigid_body_names.index("capsule")
        handle_speed = self._else_rigid_body_vel[:, handle_idx].pow(2).sum(dim=-1)
        handle_speed_reward = torch.exp(-handle_speed)
        # loc_dist = (self._else_rigid_body_pos[:, handle_idx] - self.initial_handle_pos).pow(2).sum(dim=-1)
        # handle_loc_reward = torch.exp(-loc_dist)
        handle_linear_reward = handle_speed_reward
        # handle_linear_reward = handle_speed_reward * handle_loc_reward

        handle_ang_speed = self._else_rigid_body_ang_vel[:, handle_idx].pow(2).sum(dim=-1)
        handle_ang_speed_reward = torch.exp(-0.1 * handle_ang_speed)
        # ang_loc_dist = (self._else_rigid_body_rot[:, handle_idx] - self.initial_handle_rot).pow(2).sum(dim=-1)
        # handle_ang_loc_reward = torch.exp(-ang_loc_dist)
        handle_ang_reward = handle_ang_speed_reward
        # handle_ang_reward = handle_ang_speed_reward * handle_ang_loc_reward

        handle_reward = 0.3 * handle_linear_reward + 0.7 * handle_ang_reward
        ###############################################################333

        ###############################################################333
        # # (B) finger reward
        handle_idx = self.else_rigid_body_names.index("capsule")

        palm_idx = self.rigid_body_names.index("%s_palm" % MODE)
        rthumb3_idx = self.rigid_body_names.index("%s_thumb3" % MODE)
        rindex3_idx = self.rigid_body_names.index("%s_index3" % MODE)
        rmiddle3_idx = self.rigid_body_names.index("%s_middle3" % MODE)
        rring3_idx = self.rigid_body_names.index("%s_ring3" % MODE)
        rpinky3_idx = self.rigid_body_names.index("%s_pinky3" % MODE)

        handle_pos = self._else_rigid_body_pos[:, handle_idx]
        handle_rot = self._else_rigid_body_rot[:, handle_idx]

        # finger pos & facing direction
        if MODE == "right":
            fingers_rot = self._humanoid_rigid_body_rot[:, self.rfingers_body_indices]
            fingers_rot_flat = fingers_rot.view(-1, 4)
            # newer version of fingers pos
            fingers_pos = self._humanoid_rigid_body_pos[:, self.rfingers_body_indices] + my_quat_rotate(
                fingers_rot_flat, self.rfingers_rigid_body_com.view(-1, 3)
            ).view(fingers_rot.shape[0], -1, 3)
            init_rfingers_rot_flat = self.init_rfingers_rot.view(-1, 4)
            fingers_rot_delta_flat = quat_mul(fingers_rot_flat, quat_conjugate(init_rfingers_rot_flat))
            # fingers_facing_dir_flat = my_quat_rotate(fingers_rot_delta_flat, self.unit_y.repeat(fingers_rot.shape[1], 1))
            fingers_facing_dir_flat = my_quat_rotate(
                fingers_rot_delta_flat, self.default_rfingers_facing_dir.view(-1, 3)
            )

            # fingers_contact
            fingers_contact = self._humanoid_contact_forces[:, self.rfingers_body_indices]
        elif MODE == "left":
            fingers_rot = self._humanoid_rigid_body_rot[:, self.lfingers_body_indices]
            fingers_rot_flat = fingers_rot.view(-1, 4)
            # newer version of fingers pos
            fingers_pos = self._humanoid_rigid_body_pos[:, self.lfingers_body_indices] + my_quat_rotate(
                fingers_rot_flat, self.lfingers_rigid_body_com.view(-1, 3)
            ).view(fingers_rot.shape[0], -1, 3)
            init_lfingers_rot_flat = self.init_lfingers_rot.view(-1, 4)
            fingers_rot_delta_flat = quat_mul(fingers_rot_flat, quat_conjugate(init_lfingers_rot_flat))
            # fingers_facing_dir_flat = my_quat_rotate(fingers_rot_delta_flat, -self.unit_y.repeat(fingers_rot.shape[1], 1))
            fingers_facing_dir_flat = my_quat_rotate(
                fingers_rot_delta_flat, self.default_lfingers_facing_dir.view(-1, 3)
            )

            # fingers_contact
            fingers_contact = self._humanoid_contact_forces[:, self.lfingers_body_indices]
        fingers_facing_dir = fingers_facing_dir_flat.view(*fingers_rot.shape[:-1], 3)

        handle_end1_pos = my_quat_rotate(handle_rot, self.handle_end1_offset) + handle_pos
        handle_end2_pos = my_quat_rotate(handle_rot, self.handle_end2_offset) + handle_pos

        end1_to_x = fingers_pos - handle_end1_pos[:, None]
        end1_to_end2 = (handle_end2_pos - handle_end1_pos)[:, None]
        end1_to_end2_norm = normalize(end1_to_end2)
        proj_end1_to_x = (end1_to_x * end1_to_end2_norm).sum(dim=-1, keepdim=True) * end1_to_end2_norm
        rel_portion = (proj_end1_to_x * end1_to_end2_norm).sum(dim=-1, keepdim=True) / end1_to_end2.pow(2).sum(
            dim=-1, keepdim=True
        ).sqrt()
        rel_portion_clipped = rel_portion.clip(0, 1)
        proj_end1_to_x_clipped = rel_portion_clipped * end1_to_end2
        fingers_to_target_disp = proj_end1_to_x_clipped - end1_to_x
        fingers_to_target_dist = fingers_to_target_disp.pow(2).sum(dim=-1).sqrt()

        finger_reward = torch.exp(
            -128
            * (torch.maximum(torch.zeros_like(fingers_to_target_dist), fingers_to_target_dist - 0.02))
            .pow(2)
            .max(dim=-1)
            .values
        )
        ###############################################################333

        ###############################################################333
        # # (C) is_valid_tip_contact / mcp_max_reward
        mcp_actions_abs = actions[:, [3, 4, 8, 9, 10, 12]].abs().mean(dim=1)  # thumb ABD (for fist) + all fingers mcp
        mcp_actions_dist = (1 - mcp_actions_abs).pow(2)
        mcp_max_reward = torch.exp(-3 * mcp_actions_dist)

        tip_check_idx = [k - (palm_idx + 1) for k in [rindex3_idx, rmiddle3_idx, rring3_idx, rpinky3_idx]]
        is_valid_tip_contact = torch.all(
            (fingers_contact[:, tip_check_idx] * fingers_facing_dir[:, tip_check_idx]).sum(dim=-1) < 0, dim=-1
        ).float()

        tip_check_idx = [k - (palm_idx + 1) for k in [rthumb3_idx, rindex3_idx, rmiddle3_idx, rring3_idx, rpinky3_idx]]
        fist_dist = (
            torch.maximum(
                torch.zeros_like(fingers_to_target_disp[:, tip_check_idx, 0]),
                0.9
                - (normalize(fingers_to_target_disp[:, tip_check_idx]) * fingers_facing_dir[:, tip_check_idx]).sum(
                    dim=-1
                ),
            )
            .max(dim=-1)
            .values
        )
        fist_reward = torch.exp(-3 * fist_dist.pow(2))
        ###############################################################333

        ##################################################################
        # # (D) wrist reward
        wrist_pos_reward = torch.exp(
            -3 * (self.wrist_target_dof_pos - self._humanoid_dof_pos[:, :3]).pow(2).mean(dim=-1)
        )
        wrist_vel_reward = torch.exp(-0.1 * self._humanoid_dof_vel[:, :3].pow(2).mean(dim=-1))
        wrist_reward = wrist_pos_reward * wrist_vel_reward
        ##################################################################

        ##################################################################
        # # (E) torque minimizing reward
        energy_idx = self.actuated_dof_indices[[0, 1, 2, 7, 11]]  # /with wrist and index, pinky ABD
        torque_energy_reward = torch.exp(-0.01 * self._approx_pd_force[:, energy_idx].pow(2).mean(dim=-1))
        ##################################################################

        # rew = 0.7 * handle_reward * finger_reward + 0.2 * stable_reward + 0.1 * torque_energy_reward
        rew = (
            0.95 * handle_reward * finger_reward * wrist_reward * is_valid_tip_contact * mcp_max_reward * fist_reward
            + 0.05 * torque_energy_reward
        )

        return rew

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            self._humanoid_contact_forces,
            self._contact_body_ids,
            self._humanoid_rigid_body_pos,
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_height,
        )
        self._compute_scenario_specific_reset()
        if self.save_mode:
            store_ids = torch.where(self.reset_buf)[0]
            for k in store_ids:
                k = k.item()
                epi_len = self.temp_save_count[k]
                if epi_len >= self.max_episode_length - 5:
                    self.save_count = min(self.save_buffer.shape[0], self.save_count + SAVE_MAX_HORIZON)
                    save_tensors = self.temp_save_buffer[k, :SAVE_MAX_HORIZON].clone()
                    self.save_buffer[self.save_count - SAVE_MAX_HORIZON : self.save_count] = save_tensors

                    # debug average contact forces
                    if self.save_count != self.save_buffer.shape[0]:
                        temp_object_contact_forces = self.temp_object_contact_forces[k, :SAVE_MAX_HORIZON].sum().item()
                        self.mean_object_contact_forces = (
                            self.mean_object_contact_forces * (self.save_count - SAVE_MAX_HORIZON)
                            + temp_object_contact_forces
                        )
                        self.mean_object_contact_forces /= self.save_count

                # clear temp
                self.temp_save_buffer[k] = 0
                self.temp_save_count[k] = 0
                if self.save_count == self.save_buffer.shape[0]:
                    break

            if self.save_count == self.save_buffer.shape[0]:
                save_array = self.save_buffer.cpu().numpy()
                order = np.arange(save_array.shape[0])
                np.random.shuffle(order)
                save_array = save_array[order]
                from datetime import date

                datestring = date.today().strftime("%b-%d-%Y")
                np.save("%s_handgeneral_pairs_%s.npy" % (MODE, datestring), save_array)
                print(self.mean_object_contact_forces)
                breakpoint()

        return

    def _compute_scenario_specific_reset(self):
        # enablie early termination
        handle_idx = self.else_rigid_body_names.index("capsule")
        palm_handle_dist = (
            (self._humanoid_root_states[:, :3] - self._else_rigid_body_pos[:, handle_idx]).pow(2).sum(dim=-1).sqrt()
        )
        reset_idx = palm_handle_dist > 0.7
        self.reset_buf[reset_idx] = 1
        self._terminate_buf[reset_idx] = 1

        if self.debug_mode:
            # humanoid idx
            palm_idx = self.rigid_body_names.index("%s_palm" % MODE)
            rpinky3_idx = self.rigid_body_names.index("%s_pinky3" % MODE)
            # cart idx
            handle_idx = self.else_rigid_body_names.index("capsule")

            # palm states
            palm_pos = self._humanoid_rigid_body_pos[:, palm_idx]
            palm_rot = self._humanoid_rigid_body_rot[:, palm_idx]
            palm_vel = self._humanoid_rigid_body_vel[:, palm_idx]
            palm_ang_vel = self._humanoid_rigid_body_ang_vel[:, palm_idx]

            # finger pos
            if MODE == "right":
                fingers_rot = self._humanoid_rigid_body_rot[:, self.rfingers_body_indices]
                fingers_rot_flat = fingers_rot.view(-1, 4)
                # newer version of fingers pos
                fingers_pos = self._humanoid_rigid_body_pos[:, self.rfingers_body_indices] + my_quat_rotate(
                    fingers_rot_flat, self.rfingers_rigid_body_com.view(-1, 3)
                ).view(fingers_rot.shape[0], -1, 3)

                init_rfingers_rot_flat = self.init_rfingers_rot.view(-1, 4)
                fingers_rot_delta_flat = quat_mul(fingers_rot_flat, quat_conjugate(init_rfingers_rot_flat))
                # fingers_facing_dir_flat = my_quat_rotate(fingers_rot_delta_flat, self.unit_y.repeat(fingers_pos.shape[1], 1))
                fingers_facing_dir_flat = my_quat_rotate(
                    fingers_rot_delta_flat, self.default_rfingers_facing_dir.view(-1, 3)
                )
            elif MODE == "left":
                fingers_rot = self._humanoid_rigid_body_rot[:, self.lfingers_body_indices]
                fingers_rot_flat = fingers_rot.view(-1, 4)
                # newer version of fingers pos
                fingers_pos = self._humanoid_rigid_body_pos[:, self.lfingers_body_indices] + my_quat_rotate(
                    fingers_rot_flat, self.lfingers_rigid_body_com.view(-1, 3)
                ).view(fingers_rot.shape[0], -1, 3)

                init_lfingers_rot_flat = self.init_lfingers_rot.view(-1, 4)
                fingers_rot_delta_flat = quat_mul(fingers_rot_flat, quat_conjugate(init_lfingers_rot_flat))
                # fingers_facing_dir_flat = my_quat_rotate(fingers_rot_delta_flat, -self.unit_y.repeat(fingers_pos.shape[1], 1))
                fingers_facing_dir_flat = my_quat_rotate(
                    fingers_rot_delta_flat, self.default_lfingers_facing_dir.view(-1, 3)
                )

            # object pos
            handle_pos = self._else_rigid_body_pos[:, handle_idx]
            handle_rot = self._else_rigid_body_rot[:, handle_idx]

            # calculate
            quat_inv = quat_conjugate(palm_rot)
            # unit_z = torch.zeros(self.num_envs, 3).to(self.device)
            # unit_z[:, 2] = 1
            # z_90 = quat_from_angle_axis(angle=torch.ones(self.num_envs).to(self.device) * np.pi / 2, axis=unit_z)
            # quat_inv = quat_mul(z_90, quat_inv)

            # relative fingers pos
            quat_inv_expand_flat = quat_inv[:, None].repeat(1, fingers_pos.shape[1], 1).view(-1, 4)
            rel_fingers_pos_flat = (fingers_pos - palm_pos[:, None]).view(-1, 3)
            rel_fingers_pos_flat = my_quat_rotate(quat_inv_expand_flat, rel_fingers_pos_flat)
            rel_fingers_pos = rel_fingers_pos_flat.view(fingers_pos.shape)

            # relative handle pos
            rel_handle_pos = my_quat_rotate(quat_inv, handle_pos - palm_pos)

            # relative handle end pos
            offset1 = torch.zeros_like(rel_handle_pos)
            offset2 = offset1.clone()
            offset1[:, 0] = CAPSULE_HALF_LEN
            offset2[:, 0] = -CAPSULE_HALF_LEN
            rel_handle_end1 = my_quat_rotate(quat_mul(quat_inv, handle_rot), offset1) + rel_handle_pos
            rel_handle_end2 = my_quat_rotate(quat_mul(quat_inv, handle_rot), offset2) + rel_handle_pos

            # finger joint distance from the nearest point of the bar axis
            end1_to_x = rel_fingers_pos - rel_handle_end1[:, None]
            end1_to_end2 = (rel_handle_end2 - rel_handle_end1)[:, None]
            end1_to_end2_norm = normalize(end1_to_end2)
            proj_end1_to_x = (end1_to_x * end1_to_end2_norm).sum(dim=-1, keepdim=True) * end1_to_end2_norm
            rel_portion = (proj_end1_to_x * end1_to_end2_norm).sum(dim=-1, keepdim=True) / end1_to_end2.pow(2).sum(
                dim=-1, keepdim=True
            ).sqrt()
            rel_portion_clipped = rel_portion.clip(0, 1)
            proj_end1_to_x_clipped = rel_portion_clipped * end1_to_end2
            fingers_to_target_disp = proj_end1_to_x_clipped - end1_to_x
            x_to_target_norm = normalize(fingers_to_target_disp)

            # finger_facing_dir

            fingers_facing_dir_flat = my_quat_rotate(quat_inv_expand_flat, fingers_facing_dir_flat)
            fingers_facing_dir = fingers_facing_dir_flat.view(*fingers_pos.shape[:-1], 3)
            if MODE == "left":
                fingers_facing_dir[..., 0] = -fingers_facing_dir[..., 0]

            # render
            # hand
            self._else_root_states[:, 2, :3] = self._initial_humanoid_root_states[:, :3]
            self._else_root_states[:, 2, 0] += 0.5
            self._else_root_states[:, 2, 2] += 0.2
            # fingers
            if MODE == "left":
                rel_fingers_pos[..., 0] = -rel_fingers_pos[..., 0]

            for k in range(3, 23):
                self._else_root_states[:, k, :3] = self._else_root_states[:, 2, :3] + rel_fingers_pos[:, k - 3]

                # # contact dir viz version
                # contact_forces = self._humanoid_contact_forces[:, k-3].clone()
                # contact_dir = my_quat_rotate(quat_inv, normalize(contact_forces))
                # self._else_root_states[:, k + 20, :3] = self._else_root_states[:, k, :3] + contact_dir * (0.01 + 0.005)
                # self._else_root_states[:, k + 20, 3:7] = quat_from_two_vectors(self.unit_x, contact_dir)

                # # x to target viz version
                # self._else_root_states[:, k + 20, :3] = self._else_root_states[:, k, :3] + x_to_target_norm[:, k-3] * (0.01 + 0.005)
                # self._else_root_states[:, k + 20, 3:7] = quat_from_two_vectors(self.unit_x, x_to_target_norm[:, k-3])

                # # joint facing dir
                self._else_root_states[:, k + 20, :3] = self._else_root_states[:, k, :3] + fingers_facing_dir[
                    :, k - 3
                ] * (0.01 + 0.005)
                self._else_root_states[:, k + 20, 3:7] = quat_from_two_vectors(
                    self.unit_x, fingers_facing_dir[:, k - 3]
                )

            rel_handle_rot = quat_mul(quat_inv, self._else_root_states[:, 0, 3:7])
            if MODE == "left":
                rel_handle_pos[:, 0] = -rel_handle_pos[:, 0]
                rel_handle_end1[:, 0] = -rel_handle_end1[:, 0]
                rel_handle_end2[:, 0] = -rel_handle_end2[:, 0]
                temp = rel_handle_end1.clone()
                rel_handle_end1 = rel_handle_end2.clone()
                rel_handle_end2 = temp
                rel_handle_rot[:, [0, 3]] = -rel_handle_rot[:, [0, 3]]
                # here, quaternion mirroring according to x-axis is
                # q_x, q_y, q_z, q_w -> -q_x, q_y, q_z, -q_w

            # handle
            self._else_root_states[:, 1, :3] = self._else_root_states[:, 2, :3] + rel_handle_pos
            self._else_root_states[:, 1, 3:7] = rel_handle_rot

            # handle end points
            self._else_root_states[:, -2, :3] = self._else_root_states[:, 2, :3] + rel_handle_end1
            self._else_root_states[:, -1, :3] = self._else_root_states[:, 2, :3] + rel_handle_end2

            # debug version 3 obs
            handle_pos = self._else_rigid_body_pos[:, handle_idx]
            handle_rot = self._else_rigid_body_rot[:, handle_idx]

            finger_pos = self._humanoid_rigid_body_pos[:, palm_idx + 1 : rpinky3_idx + 1]

            handle_end1_pos = my_quat_rotate(handle_rot, self.handle_end1_offset) + handle_pos
            handle_end2_pos = my_quat_rotate(handle_rot, self.handle_end2_offset) + handle_pos

            # self._else_root_states[:, -2, :3] = handle_end1_pos
            # self._else_root_states[:, -1, :3] = handle_end2_pos

            end1_to_x = finger_pos - handle_end1_pos[:, None]
            end1_to_end2 = (handle_end2_pos - handle_end1_pos)[:, None]
            end1_to_end2_norm = normalize(end1_to_end2)
            proj_end1_to_x = (end1_to_x * end1_to_end2_norm).sum(dim=-1, keepdim=True) * end1_to_end2_norm
            rel_portion = (proj_end1_to_x * end1_to_end2_norm).sum(dim=-1, keepdim=True) / end1_to_end2.pow(2).sum(
                dim=-1, keepdim=True
            ).sqrt()
            rel_portion_clipped = rel_portion.clip(0, 1)
            proj_end1_to_x_clipped = rel_portion_clipped * end1_to_end2
            # self._else_root_states[:, -1, :3] = proj_end1_to_x_clipped[:, 0] + handle_end1_pos

        if self.reset_buf.sum() == 0:  # execute only when _set_env_state is not called
            num_actors = int(self._root_states.shape[0] / self.num_envs)
            global_indices = torch.arange(self.num_envs * num_actors, dtype=torch.int32, device=self.device)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(global_indices),
                len(global_indices),
            )

        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)
        obs = self._compute_scenario_specific_obs(obs, env_ids)

        if env_ids is None:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_observations_for_saving(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids, is_save_obs=True)
        obs = self._compute_scenario_specific_obs(obs, env_ids, is_save_obs=True)

        return obs

    def _compute_scenario_specific_obs(self, obs, env_ids=None, is_save_obs=False):
        # humanoid idx
        palm_idx = self.rigid_body_names.index("%s_palm" % MODE)
        # cart idx
        handle_idx = self.else_rigid_body_names.index("capsule")

        # # version 1
        # if (env_ids is None):
        #     # handle features
        #     handle_pos = self._else_rigid_body_pos[:, handle_idx]
        #     handle_rot = self._else_rigid_body_rot[:, handle_idx]
        #     handle_vel = self._else_rigid_body_vel[:, handle_idx]
        #     handle_ang_vel = self._else_rigid_body_ang_vel[:, handle_idx]
        #     handle_contact_forces = self._else_contact_forces[:, handle_idx, :3]
        #     # ref_features
        #     upper_arm_pos = self._humanoid_root_states[:, :3] # corresponds to self._humanoid_rigid_body_pos[:, upper_arm_idx]
        #     upper_arm_rot = self._humanoid_root_states[:, 3:7] # corresponds to self._humanoid_rigid_body_rot[:, upper_arm_idx]
        #     # hand features
        #     rhand_pos = self._humanoid_rigid_body_pos[:, palm_idx + 1:rpinky3_idx + 1, :3].mean(dim=-2)

        # else:
        #     # handle features
        #     handle_pos = self._else_rigid_body_pos[env_ids, handle_idx]
        #     handle_rot = self._else_rigid_body_rot[env_ids, handle_idx]
        #     handle_vel = self._else_rigid_body_vel[env_ids, handle_idx]
        #     handle_ang_vel = self._else_rigid_body_ang_vel[env_ids, handle_idx]
        #     handle_contact_forces = self._else_contact_forces[env_ids, handle_idx, :3]
        #     # goal_features
        #     upper_arm_pos = self._humanoid_root_states[env_ids, :3]
        #     upper_arm_rot = self._humanoid_root_states[env_ids, 3:7]
        #     # hand features
        #     rhand_pos = self._humanoid_rigid_body_pos[env_ids, palm_idx + 1:rpinky3_idx + 1, :3].mean(dim=-2)

        # # make goal-relative space
        # handle_pos = handle_pos - upper_arm_pos
        # rhand_pos = rhand_pos - upper_arm_pos

        # # upper arm features
        # upper_arm_features = upper_arm_rot

        # # hand features
        # hand_features = torch.cat([rhand_pos], dim=-1) # 3

        # # handle_features
        # handle_features = torch.cat([handle_pos, handle_rot, handle_vel, handle_ang_vel, handle_contact_forces], dim=-1) # 13 + 3

        # obs = torch.cat([obs, upper_arm_features, handle_features, hand_features], dim=-1)

        # # version 2
        # if (env_ids is None):
        #     # palm states
        #     palm_pos = self._humanoid_rigid_body_pos[:, palm_idx]
        #     palm_rot = self._humanoid_rigid_body_rot[:, palm_idx]
        #     # palm_vel = self._humanoid_rigid_body_vel[:, palm_idx]
        #     # palm_ang_vel = self._humanoid_rigid_body_ang_vel[:, palm_idx]

        #     # finger pos
        #     fingers_pos = self._humanoid_rigid_body_pos[:, palm_idx + 1:rpinky3_idx + 1]

        #     # object pos
        #     handle_pos = self._else_rigid_body_pos[:, handle_idx]
        #     handle_rot = self._else_rigid_body_rot[:, handle_idx]

        #     # hand contacts
        #     hand_contacts = self._humanoid_contact_forces[:, palm_idx: rpinky3_idx + 1]

        # else:
        #     # palm states
        #     palm_pos = self._humanoid_rigid_body_pos[env_ids, palm_idx]
        #     palm_rot = self._humanoid_rigid_body_rot[env_ids, palm_idx]
        #     # palm_vel = self._humanoid_rigid_body_vel[env_ids, palm_idx]
        #     # palm_ang_vel = self._humanoid_rigid_body_ang_vel[env_ids, palm_idx]

        #     # finger pos
        #     fingers_pos = self._humanoid_rigid_body_pos[env_ids, palm_idx + 1:rpinky3_idx + 1]

        #     # object pos
        #     handle_pos = self._else_rigid_body_pos[env_ids, handle_idx]
        #     handle_rot = self._else_rigid_body_rot[env_ids, handle_idx]

        #     # hand contacts
        #     hand_contacts = self._humanoid_contact_forces[env_ids, palm_idx: rpinky3_idx + 1]

        # quat_inv = quat_conjugate(palm_rot)

        # # relative fingers pos
        # quat_inv_expand_flat = quat_inv[:, None].repeat(1, fingers_pos.shape[1], 1).view(-1, 4)
        # rel_fingers_pos_flat = (fingers_pos - palm_pos[:, None]).view(-1, 3)
        # rel_fingers_pos_flat = my_quat_rotate(quat_inv_expand_flat, rel_fingers_pos_flat)
        # rel_fingers_pos = rel_fingers_pos_flat.view(fingers_pos.shape)

        # # relative contact forces
        # quat_inv_expand_flat = quat_inv[:, None].repeat(1, hand_contacts.shape[1], 1).view(-1, 4)
        # hand_contacts_flat = hand_contacts.reshape(-1, 3)
        # rel_hand_contacts_flat = my_quat_rotate(quat_inv_expand_flat, normalize(hand_contacts_flat))
        # rel_hand_contacts = rel_hand_contacts_flat.view(hand_contacts.shape)

        # # relative handle position
        # rel_handle_pos = my_quat_rotate(quat_inv, handle_pos - palm_pos)

        # # relative handle end pos
        # offset1 = torch.zeros_like(rel_handle_pos)
        # offset2= offset1.clone()
        # offset1[:, 0] = CAPSULE_HALF_LEN
        # offset2[:, 0] = -CAPSULE_HALF_LEN
        # rel_handle_end1 = my_quat_rotate(quat_mul(quat_inv, handle_rot), offset1) + rel_handle_pos
        # rel_handle_end2 = my_quat_rotate(quat_mul(quat_inv, handle_rot), offset2) + rel_handle_pos
        # # print(rel_handle_end1[0], rel_handle_end2[0])
        # # flat obs
        # rel_fingers_pos_flat = rel_fingers_pos.view(rel_fingers_pos.shape[0], -1)
        # # hand_contacts_flat = hand_contacts.view(hand_contacts.shape[0], -1)
        # rel_hand_contacts_flat = rel_hand_contacts.view(rel_hand_contacts.shape[0], -1)

        # # new_obs = torch.cat([rel_fingers_pos_flat, rel_handle_end1, rel_handle_end2, hand_contacts_flat], dim=-1)
        # new_obs = torch.cat([rel_fingers_pos_flat, rel_handle_end1, rel_handle_end2, rel_hand_contacts_flat], dim=-1)

        # obs = torch.cat([obs, new_obs], dim=-1)

        # # version 3
        # if (env_ids is None):
        #     # palm states
        #     palm_pos = self._humanoid_rigid_body_pos[:, palm_idx]
        #     palm_rot = self._humanoid_rigid_body_rot[:, palm_idx]
        #     # palm_vel = self._humanoid_rigid_body_vel[:, palm_idx]
        #     # palm_ang_vel = self._humanoid_rigid_body_ang_vel[:, palm_idx]

        #     # finger pos
        #     fingers_pos = self._humanoid_rigid_body_pos[:, palm_idx + 1:rpinky3_idx + 1]

        #     # object pos
        #     handle_pos = self._else_rigid_body_pos[:, handle_idx]
        #     handle_rot = self._else_rigid_body_rot[:, handle_idx]

        #     # hand contacts
        #     hand_contacts = self._humanoid_contact_forces[:, palm_idx: rpinky3_idx + 1]

        # else:
        #     # palm states
        #     palm_pos = self._humanoid_rigid_body_pos[env_ids, palm_idx]
        #     palm_rot = self._humanoid_rigid_body_rot[env_ids, palm_idx]
        #     # palm_vel = self._humanoid_rigid_body_vel[env_ids, palm_idx]
        #     # palm_ang_vel = self._humanoid_rigid_body_ang_vel[env_ids, palm_idx]

        #     # finger pos
        #     fingers_pos = self._humanoid_rigid_body_pos[env_ids, palm_idx + 1:rpinky3_idx + 1]

        #     # object pos
        #     handle_pos = self._else_rigid_body_pos[env_ids, handle_idx]
        #     handle_rot = self._else_rigid_body_rot[env_ids, handle_idx]

        #     # hand contacts
        #     hand_contacts = self._humanoid_contact_forces[env_ids, palm_idx: rpinky3_idx + 1]

        # quat_inv = quat_conjugate(palm_rot)

        # # relative fingers pos
        # quat_inv_expand_flat = quat_inv[:, None].repeat(1, fingers_pos.shape[1], 1).view(-1, 4)
        # rel_fingers_pos_flat = (fingers_pos - palm_pos[:, None]).view(-1, 3)
        # rel_fingers_pos_flat = my_quat_rotate(quat_inv_expand_flat, rel_fingers_pos_flat)
        # rel_fingers_pos = rel_fingers_pos_flat.view(fingers_pos.shape)

        # # relative handle position
        # rel_handle_pos = my_quat_rotate(quat_inv, handle_pos - palm_pos)

        # # relative handle end pos
        # offset1 = torch.zeros_like(rel_handle_pos)
        # offset2= offset1.clone()
        # offset1[:, 0] = CAPSULE_HALF_LEN
        # offset2[:, 0] = -CAPSULE_HALF_LEN
        # rel_handle_end1 = my_quat_rotate(quat_mul(quat_inv, handle_rot), offset1) + rel_handle_pos
        # rel_handle_end2 = my_quat_rotate(quat_mul(quat_inv, handle_rot), offset2) + rel_handle_pos
        # # print(rel_handle_end1[0], rel_handle_end2[0])

        # # relative fingers pos
        # quat_inv_expand_flat = quat_inv[:, None].repeat(1, fingers_pos.shape[1], 1).view(-1, 4)
        # rel_fingers_pos_flat = (fingers_pos - palm_pos[:, None]).view(-1, 3)
        # rel_fingers_pos_flat = my_quat_rotate(quat_inv_expand_flat, rel_fingers_pos_flat)
        # rel_fingers_pos = rel_fingers_pos_flat.view(fingers_pos.shape)

        # end1_to_x = rel_fingers_pos - rel_handle_end1[:, None]
        # end1_to_end2_norm = normalize(rel_handle_end2- rel_handle_end1)[:, None]
        # proj_end1_to_x = (end1_to_x * end1_to_end2_norm).sum(dim=-1, keepdim=True) * end1_to_end2_norm
        # x_to_target_norm = normalize(proj_end1_to_x - end1_to_x)

        # # relative contact forces
        # quat_inv_expand_flat = quat_inv[:, None].repeat(1, hand_contacts.shape[1], 1).view(-1, 4)
        # hand_contacts_flat = hand_contacts.reshape(-1, 3)
        # rel_hand_contacts_flat = my_quat_rotate(quat_inv_expand_flat, normalize(hand_contacts_flat))
        # rel_hand_contacts = rel_hand_contacts_flat.view(hand_contacts.shape)

        # # flat obs
        # fingers_to_target_pos_flat = x_to_target_norm.view(rel_fingers_pos.shape[0], -1)
        # rel_hand_contacts_flat = rel_hand_contacts.view(rel_hand_contacts.shape[0], -1)

        # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, fingers_to_target_pos_flat, rel_hand_contacts_flat], dim=-1)

        # obs = torch.cat([obs, new_obs], dim=-1)

        # version 4
        if env_ids is None:
            # palm states
            palm_pos = self._humanoid_rigid_body_pos[:, palm_idx]
            palm_rot = self._humanoid_rigid_body_rot[:, palm_idx]

            # finger pos & hand contacts
            if MODE == "right":
                fingers_rot = self._humanoid_rigid_body_rot[:, self.rfingers_body_indices]
                fingers_rot_flat = fingers_rot.view(-1, 4)
                # newer version of fingers_pos
                fingers_pos = self._humanoid_rigid_body_pos[:, self.rfingers_body_indices] + my_quat_rotate(
                    fingers_rot_flat, self.rfingers_rigid_body_com.view(-1, 3)
                ).view(fingers_rot.shape[0], -1, 3)
                init_rfingers_rot_flat = self.init_rfingers_rot.view(-1, 4)
                fingers_rot_delta_flat = quat_mul(fingers_rot_flat, quat_conjugate(init_rfingers_rot_flat))
                # fingers_facing_dir_flat = my_quat_rotate(fingers_rot_delta_flat, self.unit_y.repeat(fingers_pos.shape[1], 1))
                fingers_facing_dir_flat = my_quat_rotate(
                    fingers_rot_delta_flat, self.default_rfingers_facing_dir.view(-1, 3)
                )

                # fingers_contact
                fingers_contact = self._humanoid_contact_forces[:, self.rfingers_body_indices]
            elif MODE == "left":
                fingers_rot = self._humanoid_rigid_body_rot[:, self.lfingers_body_indices]
                fingers_rot_flat = fingers_rot.view(-1, 4)
                # newer version of fingers_pos
                fingers_pos = self._humanoid_rigid_body_pos[:, self.lfingers_body_indices] + my_quat_rotate(
                    fingers_rot_flat, self.lfingers_rigid_body_com.view(-1, 3)
                ).view(fingers_rot.shape[0], -1, 3)
                init_lfingers_rot_flat = self.init_lfingers_rot.view(-1, 4)
                fingers_rot_delta_flat = quat_mul(fingers_rot_flat, quat_conjugate(init_lfingers_rot_flat))
                # fingers_facing_dir_flat = my_quat_rotate(fingers_rot_delta_flat, -self.unit_y.repeat(fingers_pos.shape[1], 1))
                fingers_facing_dir_flat = my_quat_rotate(
                    fingers_rot_delta_flat, self.default_lfingers_facing_dir.view(-1, 3)
                )

                # fingers_contact
                fingers_contact = self._humanoid_contact_forces[:, self.lfingers_body_indices]

            # object pos
            handle_pos = self._else_rigid_body_pos[:, handle_idx]
            handle_rot = self._else_rigid_body_rot[:, handle_idx]

        else:
            # palm states
            palm_pos = self._humanoid_rigid_body_pos[env_ids, palm_idx]
            palm_rot = self._humanoid_rigid_body_rot[env_ids, palm_idx]

            # finger pos
            if MODE == "right":
                fingers_rot = self._humanoid_rigid_body_rot[env_ids[:, None], self.rfingers_body_indices[None]]
                fingers_rot_flat = fingers_rot.view(-1, 4)
                # newer version of fingers_pos
                fingers_pos = self._humanoid_rigid_body_pos[
                    env_ids[:, None], self.rfingers_body_indices[None]
                ] + my_quat_rotate(fingers_rot_flat, self.rfingers_rigid_body_com[env_ids].view(-1, 3)).view(
                    fingers_rot.shape[0], -1, 3
                )
                init_rfingers_rot_flat = self.init_rfingers_rot[env_ids].view(-1, 4)
                fingers_rot_delta_flat = quat_mul(fingers_rot_flat, quat_conjugate(init_rfingers_rot_flat))
                # fingers_facing_dir_flat = my_quat_rotate(fingers_rot_delta_flat, self.unit_y[env_ids].repeat(fingers_rot.shape[1], 1))
                fingers_facing_dir_flat = my_quat_rotate(
                    fingers_rot_delta_flat, self.default_rfingers_facing_dir[env_ids].view(-1, 3)
                )

                # fingers_contact
                fingers_contact = self._humanoid_contact_forces[env_ids[:, None], self.rfingers_body_indices[None]]

            elif MODE == "left":
                fingers_rot = self._humanoid_rigid_body_rot[env_ids[:, None], self.lfingers_body_indices[None]]
                fingers_rot_flat = fingers_rot.view(-1, 4)
                # newer version of fingers_pos
                fingers_pos = self._humanoid_rigid_body_pos[
                    env_ids[:, None], self.lfingers_body_indices[None]
                ] + my_quat_rotate(fingers_rot_flat, self.lfingers_rigid_body_com[env_ids].view(-1, 3)).view(
                    fingers_rot.shape[0], -1, 3
                )
                init_lfingers_rot_flat = self.init_lfingers_rot[env_ids].view(-1, 4)
                fingers_rot_delta_flat = quat_mul(fingers_rot_flat, quat_conjugate(init_lfingers_rot_flat))
                # fingers_facing_dir_flat = my_quat_rotate(fingers_rot_delta_flat, -self.unit_y[env_ids].repeat(fingers_rot.shape[1], 1))
                fingers_facing_dir_flat = my_quat_rotate(
                    fingers_rot_delta_flat, self.default_lfingers_facing_dir[env_ids].view(-1, 3)
                )

                # fingers contact
                fingers_contact = self._humanoid_contact_forces[env_ids[:, None], self.lfingers_body_indices[None]]
            # object pos
            handle_pos = self._else_rigid_body_pos[env_ids, handle_idx]
            handle_rot = self._else_rigid_body_rot[env_ids, handle_idx]

        quat_inv = quat_conjugate(palm_rot)

        # handle position in hand-centric coordinate
        rel_handle_pos = my_quat_rotate(quat_inv, handle_pos - palm_pos)

        # handle end position in hand-centric coordinate
        offset1 = torch.zeros_like(rel_handle_pos)
        offset2 = offset1.clone()
        offset1[:, 0] = CAPSULE_HALF_LEN
        offset2[:, 0] = -CAPSULE_HALF_LEN
        rel_handle_end1 = my_quat_rotate(quat_mul(quat_inv, handle_rot), offset1) + rel_handle_pos
        rel_handle_end2 = my_quat_rotate(quat_mul(quat_inv, handle_rot), offset2) + rel_handle_pos

        # relative hand pos from center of handle pos in hand-centric coordinate
        quat_inv_expand_flat = quat_inv[:, None].repeat(1, fingers_pos.shape[1], 1).view(-1, 4)
        rel_fingers_pos_flat = (fingers_pos - palm_pos[:, None]).view(-1, 3)
        rel_fingers_pos_flat = my_quat_rotate(quat_inv_expand_flat, rel_fingers_pos_flat)
        rel_fingers_pos = rel_fingers_pos_flat.view(fingers_pos.shape)
        finger_tips_body_indices = [3, 7, 11, 15, 19]
        rel_finger_tips_pos = rel_fingers_pos[:, finger_tips_body_indices]
        rel_finger_tips_pos_flat = rel_finger_tips_pos.view(rel_finger_tips_pos.shape[0], -1)

        # fingers_to_handle = rel_fingers_pos - rel_handle_pos[:, None] # (N, Nfing, 3)
        # fingers_to_handle_flat = fingers_to_handle.view(fingers_to_handle.shape[0], -1)

        # # finger joint distance from the nearest point of the bar axis
        # end1_to_x = rel_fingers_pos - rel_handle_end1[:, None]
        # end1_to_end2 = (rel_handle_end2 - rel_handle_end1)[:, None]
        # end1_to_end2_norm = normalize(end1_to_end2)
        # proj_end1_to_x = (end1_to_x * end1_to_end2_norm).sum(dim=-1, keepdim=True) * end1_to_end2_norm
        # rel_portion = (proj_end1_to_x * end1_to_end2_norm).sum(dim=-1, keepdim=True) / end1_to_end2.pow(2).sum(dim=-1, keepdim=True).sqrt()
        # rel_portion_clipped = rel_portion.clip(0, 1)
        # proj_end1_to_x_clipped = rel_portion_clipped * end1_to_end2
        # fingers_to_target_disp = (proj_end1_to_x_clipped - end1_to_x)
        # fingers_to_target_dist = fingers_to_target_disp.pow(2).sum(dim=-1).sqrt() # (N, Nfing)

        # finger facing dir in wrist coordinate
        fingers_facing_dir_flat = my_quat_rotate(quat_inv_expand_flat, fingers_facing_dir_flat)
        fingers_facing_dir = fingers_facing_dir_flat.view(*fingers_pos.shape[:-1], 3)
        # fingers_to_target_signed_dist = (fingers_to_target_disp * fingers_facing_dir).sum(dim=-1)
        # fingers_to_target_cosine = (normalize(fingers_to_target_disp) * fingers_facing_dir).sum(dim=-1)

        # contact marker for tips
        rindex3_idx = self.rigid_body_names.index("%s_index3" % MODE)
        rmiddle3_idx = self.rigid_body_names.index("%s_middle3" % MODE)
        rring3_idx = self.rigid_body_names.index("%s_ring3" % MODE)
        rpinky3_idx = self.rigid_body_names.index("%s_pinky3" % MODE)
        tip_check_idx = [k - (palm_idx + 1) for k in [rindex3_idx, rmiddle3_idx, rring3_idx, rpinky3_idx]]
        rel_fingers_contact = my_quat_rotate(quat_inv_expand_flat, fingers_contact.view(-1, 3)).view(
            *fingers_contact.shape
        )
        tip_valid_contact_marker = (
            (rel_fingers_contact[:, tip_check_idx] * fingers_facing_dir[:, tip_check_idx]).sum(dim=-1) < 0
        ).float()

        # # tip direction marker relative to rod --> Dec-24 (A)
        # finger_tips_facing_dir_to_contact_cosine = (normalize(rel_fingers_contact[:, tip_check_idx]) * fingers_facing_dir[:, tip_check_idx]).sum(dim=-1)

        # # tip direction marker relative to rod --> Dec-24 (B)
        # tip_prev_check_idx = [k - 1 for k in tip_check_idx]
        # fingers_facing_dir_mid = normalize(fingers_facing_dir[:, tip_prev_check_idx] + fingers_facing_dir[:, tip_check_idx])
        # finger_tips_to_target_to_facing_dir_cosine = (normalize(fingers_to_target_disp[:, tip_check_idx]) * fingers_facing_dir_mid).sum(dim=-1)

        # tip direction marker relative to rod --> Dec-24 (C), (D)
        fingers_facing_dir_to_contact_cosine = (normalize(rel_fingers_contact) * fingers_facing_dir).sum(dim=-1)

        if MODE == "left":
            rel_handle_end1[:, 0] = -rel_handle_end1[:, 0]
            rel_handle_end2[:, 0] = -rel_handle_end2[:, 0]
            temp = rel_handle_end1.clone()
            rel_handle_end1 = rel_handle_end2.clone()
            rel_handle_end2 = temp
            # fingers_to_handle[..., 0] = - fingers_to_handle[..., 0]
            # fingers_to_handle_flat = fingers_to_handle.view(fingers_to_handle.shape[0], -1)
            rel_finger_tips_pos[..., 0] = -rel_finger_tips_pos[..., 0]
            rel_finger_tips_pos_flat = rel_finger_tips_pos.view(rel_finger_tips_pos.shape[0], -1)

        # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, fingers_to_target_dist, fingers_to_handle_flat], dim=-1) # (3, 3, 20, 20 * 3)
        # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, fingers_to_target_signed_dist, fingers_to_handle_flat], dim=-1) # (3, 3, 20, 20 * 3)

        if is_save_obs:
            # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, rel_finger_tips_pos_flat, fingers_to_target_signed_dist], dim=-1) # (3, 3, 15, 20) --> Dec-21
            # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, tip_valid_contact_marker, rel_finger_tips_pos_flat, fingers_to_target_dist, fingers_to_target_cosine], dim=-1) # (3, 3, 4, 15, 20, 20) --> Dec-22
            # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, tip_valid_contact_marker, rel_finger_tips_pos_flat, fingers_to_target_dist, finger_tips_facing_dir_to_contact_cosine], dim=-1) # (3, 3, 4, 15, 20, 4) --> Dec-24 - (A)
            # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, tip_valid_contact_marker, rel_finger_tips_pos_flat, fingers_to_target_dist, finger_tips_to_target_to_facing_dir_cosine], dim=-1) # (3, 3, 4, 15, 20, 4) --> Dec-24 - (B)
            # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, tip_valid_contact_marker, rel_finger_tips_pos_flat, fingers_to_target_dist, fingers_facing_dir_to_contact_cosine], dim=-1) # (3, 3, 4, 15, 20, 4) --> Dec-24 - (C)
            new_obs = torch.cat(
                [
                    rel_handle_end1,
                    rel_handle_end2,
                    tip_valid_contact_marker,
                    rel_finger_tips_pos_flat,
                    fingers_facing_dir_to_contact_cosine,
                ],
                dim=-1,
            )  # (3, 3, 4, 15, 20) --> Dec-24 - (D)
        else:
            if env_ids is None:
                wrist_target_dof_pos_flat = self.wrist_target_dof_pos.view(self.wrist_target_dof_pos.shape[0], -1)
            else:
                wrist_target_dof_pos_flat = self.wrist_target_dof_pos[env_ids].view(
                    self.wrist_target_dof_pos[env_ids].shape[0], -1
                )
            # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, rel_finger_tips_pos_flat, fingers_to_target_signed_dist, wrist_target_dof_pos_flat], dim=-1) # (3, 3, 15, 20) --> Dec-21
            # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, tip_valid_contact_marker, rel_finger_tips_pos_flat, fingers_to_target_dist, fingers_to_target_cosine, wrist_target_dof_pos_flat], dim=-1) # (3, 3, 5, 15, 20, 20, 3) --> Dec-22
            # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, tip_valid_contact_marker, rel_finger_tips_pos_flat, fingers_to_target_dist, finger_tips_facing_dir_to_contact_cosine, wrist_target_dof_pos_flat], dim=-1) # (3, 3, 5, 15, 20, 4, 3) --> Dec-24 - (A)
            # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, tip_valid_contact_marker, rel_finger_tips_pos_flat, fingers_to_target_dist, finger_tips_to_target_to_facing_dir_cosine, wrist_target_dof_pos_flat], dim=-1) # (3, 3, 5, 15, 20, 4, 3) --> Dec-24 - (B)
            # new_obs = torch.cat([rel_handle_end1, rel_handle_end2, tip_valid_contact_marker, rel_finger_tips_pos_flat, fingers_to_target_dist, fingers_facing_dir_to_contact_cosine, wrist_target_dof_pos_flat], dim=-1) # (3, 3, 5, 15, 20, 4, 3) --> Dec-24 - (C)
            new_obs = torch.cat(
                [
                    rel_handle_end1,
                    rel_handle_end2,
                    tip_valid_contact_marker,
                    rel_finger_tips_pos_flat,
                    fingers_facing_dir_to_contact_cosine,
                    wrist_target_dof_pos_flat,
                ],
                dim=-1,
            )  # (3, 3, 4, 15, 20, 3) --> Dec-24 - (D)
        obs = torch.cat([obs, new_obs], dim=-1)

        return obs

    def _compute_humanoid_obs(self, env_ids=None, is_save_obs=False):
        if env_ids is None:
            root_states = self._humanoid_root_states
            dof_pos = self._humanoid_dof_pos
            dof_vel = self._humanoid_dof_vel
            key_body_pos = self._humanoid_rigid_body_pos[:, self._key_body_ids, :]
        else:
            root_states = self._humanoid_root_states[env_ids]
            dof_pos = self._humanoid_dof_pos[env_ids]
            dof_vel = self._humanoid_dof_vel[env_ids]
            key_body_pos = self._humanoid_rigid_body_pos[env_ids][:, self._key_body_ids, :]

        if MODE == "left":
            dof_pos[:, 1] = -dof_pos[:, 1]
            dof_vel[:, 1] = -dof_vel[:, 1]

        if self.save_wo_wrist and is_save_obs:
            obs = compute_humanoid_observations_wo_wrist(
                root_states, dof_pos, dof_vel, key_body_pos, self._local_root_obs
            )
        else:
            obs = compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, self._local_root_obs)

        return obs

    def _reset_actors(self, env_ids):
        ## Humanoid Hand
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        # hand pos
        self._humanoid_root_states[env_ids, 0] += 0.41
        self._humanoid_root_states[env_ids, 2] = 0.98

        if MODE == "right":
            init_z_angle = torch.zeros(env_ids.shape[0]).to(self.device).uniform_(20, 40) * np.pi / 180
        else:
            init_z_angle = torch.zeros(env_ids.shape[0]).to(self.device).uniform_(-40, -20) * np.pi / 180
        init_z_rot = quat_from_angle_axis(
            angle=init_z_angle, axis=torch.FloatTensor([0.0, 0.0, 1.0]).repeat(env_ids.shape[0], 1).to(self.device)
        )

        y_angle = torch.zeros(env_ids.shape[0]).to(self.device).uniform_(-100, -80) * np.pi / 180
        y_rot = quat_from_angle_axis(
            angle=y_angle, axis=torch.FloatTensor([0.0, 1.0, 0.0]).repeat(env_ids.shape[0], 1).to(self.device)
        )

        z_angle = torch.zeros(env_ids.shape[0]).to(self.device).uniform_(-10, 10) * np.pi / 180
        z_rot = quat_from_angle_axis(
            angle=z_angle, axis=torch.FloatTensor([0.0, 0.0, 1.0]).repeat(env_ids.shape[0], 1).to(self.device)
        )
        self._humanoid_root_states[env_ids, 3:7] = quat_mul(
            z_rot, quat_mul(y_rot, quat_mul(init_z_rot, self._humanoid_root_states[env_ids, 3:7]))
        )

        palm_pos = my_quat_rotate(
            self._humanoid_root_states[env_ids, 3:7],
            torch.FloatTensor([0, 0, -0.24]).repeat(env_ids.shape[0], 1).to(self.device),
        )
        disp_by_rand = palm_pos - torch.FloatTensor([0.24, 0, 0]).to(self.device)[None]
        self._humanoid_root_states[env_ids, :3] -= disp_by_rand

        self._humanoid_dof_pos[env_ids] = self._pd_action_offset

        # # fingers
        # self._humanoid_dof_pos[env_ids[:, None], self.actuated_dof_indices[None]] = self._pd_action_offset[self.actuated_dof_indices] - self._pd_action_scale[self.actuated_dof_indices]
        # # for thumbs
        # self._humanoid_dof_pos[env_ids, 3] = self._pd_action_offset[3]
        # self._humanoid_dof_pos[env_ids, 4] = 0
        # self._humanoid_dof_pos[env_ids, 5] = 0
        # self._humanoid_dof_pos[env_ids, 6] = self._pd_action_offset[6]
        # # zero vel
        # self._humanoid_dof_vel[env_ids] = 0

        # random initialization
        # fingers
        mcp_offset = torch.zeros(env_ids.shape[0], 1).uniform_(-0.8, 0.8).to(self.device)
        mcp_scale = torch.zeros(env_ids.shape[0], 4).uniform_(-0.15, 0.15).to(self.device)
        mcp_actions = mcp_offset + mcp_scale
        abd_actions = torch.zeros(env_ids.shape[0], 2).uniform_(-1.0, 1.0).to(self.device)

        self._humanoid_dof_pos[env_ids[:, None], self.mcp_indices[None]] += (
            mcp_actions * self._pd_action_scale[self.mcp_indices]
        )
        self._humanoid_dof_pos[env_ids[:, None], self.abd_indices[None]] += (
            abd_actions * self._pd_action_scale[self.abd_indices]
        )

        # for thumbs
        thumb_abd_actions = torch.zeros(env_ids.shape[0], 1).uniform_(-0.5, 0.5).to(self.device)
        thumb_else_offset = torch.zeros(env_ids.shape[0], 1).uniform_(-0.8, 0.8).to(self.device)
        thumb_else_scale = torch.zeros(env_ids.shape[0], 3).uniform_(-0.15, 0.15).to(self.device)
        thumb_actions = torch.cat([thumb_abd_actions, thumb_else_offset + thumb_else_scale], dim=-1)
        self._humanoid_dof_pos[env_ids[:, None], self.thumb_indices[None]] += (
            thumb_actions * self._pd_action_scale[self.thumb_indices]
        )

        # random_vel
        self._humanoid_dof_vel[env_ids] = 0
        self._humanoid_dof_vel[env_ids[:, None], self.actuated_dof_indices[None]] = (
            torch.zeros_like(self._humanoid_dof_vel[env_ids[:, None], self.actuated_dof_indices[None]]).uniform_(
                -0.05, 0.05
            )
            * self._pd_action_scale[self.actuated_dof_indices]
        )

        # wrist
        self._humanoid_dof_pos[env_ids, 0] = 0
        if MODE == "right":
            self._humanoid_dof_pos[env_ids, 1] = 1.57 - init_z_angle  # 1.57
            # self.initial_wrist_dof[env_ids] = 1.57 - init_z_angle
        else:
            self._humanoid_dof_pos[env_ids, 1] = -1.57 - init_z_angle  # 1.57
            # self.initial_wrist_dof[env_ids] = -1.57 - init_z_angle
        self._humanoid_dof_pos[env_ids, 2] = 0

        self.cur_targets[env_ids, : self.humanoid_num_dof] = self._pd_action_offset
        # when deubugging initial hand pose
        # self.cur_targets[env_ids, :self.humanoid_num_dof] = self._humanoid_dof_pos

        ## Else states
        self._else_root_states[env_ids, 0] = self._initial_else_root_states[env_ids, 0]
        self._else_root_states[env_ids, 0, :3] = self.respawn_handle_pos[None]

        # handle pos randomize
        handle_y_range = (-0.2, 0.2)
        self._else_root_states[env_ids, 0, 1] = torch.zeros_like(self._else_root_states[env_ids, 0, 1]).uniform_(
            *handle_y_range
        )

        # additional hand rot randomize according to handle
        handle_y_offset = (-0.78, 0) if MODE == "right" else (0, 0.78)
        wrist_dof_offset = torch.zeros_like(init_z_angle).uniform_(*handle_y_offset)
        self._humanoid_dof_pos[env_ids, 1] += wrist_dof_offset
        rot_offset = quat_from_angle_axis(
            angle=-wrist_dof_offset, axis=torch.FloatTensor([1.0, 0.0, 0.0]).repeat(env_ids.shape[0], 1).to(self.device)
        )

        new_handle_rot = quat_mul(rot_offset, self._else_root_states[env_ids, 0, 3:7])
        handle_pos_offset = self._else_root_states[env_ids, 0, :3] - self._humanoid_root_states[env_ids, :3]
        new_handle_pos = self._humanoid_root_states[env_ids, :3] + my_quat_rotate(rot_offset, handle_pos_offset)
        self._else_root_states[env_ids, 0, :3] = new_handle_pos
        self._else_root_states[env_ids, 0, 3:7] = new_handle_rot

        self.initial_wrist_dof[env_ids] = self._humanoid_dof_pos[env_ids, 1]
        # self._else_root_states[env_ids, 0, 3:7] = quat_unit(torch.zeros_like(self._else_root_states[env_ids, 0, 3:7]).uniform_(-1, 1))

        self.initial_handle_pos[env_ids] = self._else_root_states[env_ids, 0, :3]
        self.initial_handle_rot[env_ids] = self._else_root_states[env_ids, 0, 3:7]

        handle_forces_mag = torch.zeros_like(self.handle_forces[env_ids, :1]).uniform_(*self.handle_force_limit)
        handle_forces_mag_dir = torch.bernoulli(torch.ones_like(self.handle_forces[env_ids, :1]) * 0.5) * 2 - 1
        handle_forces_mag = handle_forces_mag * handle_forces_mag_dir
        handle_torques_mag = torch.zeros_like(self.handle_torques[env_ids, :1]).uniform_(*self.handle_torque_limit)

        handle_forces_dir = normalize(torch.zeros_like(self.handle_forces[env_ids]).uniform_(-1, 1))
        handle_torques_dir = normalize(torch.zeros_like(self.handle_torques[env_ids]).uniform_(-1, 1))
        # handle_torques_dir = torch.zeros_like(self.handle_torques[env_ids])

        self.handle_forces_mag[env_ids] = handle_forces_mag
        self.handle_forces[env_ids] = handle_forces_mag * handle_forces_dir
        self.handle_torques[env_ids] = handle_torques_mag * handle_torques_dir
        self.is_axial_forces[env_ids] = torch.bernoulli(
            torch.ones_like(self.handle_forces_mag[env_ids, 0]) * AXIAL_FORCES_RATIO
        )

        # wrist_target_dof_pos
        self.wrist_target_dof_pos[env_ids, 0] = 0
        self.wrist_target_dof_pos[env_ids, 1] = (
            torch.zeros_like(self.wrist_target_dof_pos[env_ids, 1]).uniform_(-0.8, 0.8) * self._pd_action_scale[1]
            + self._pd_action_offset[1]
        )
        self.wrist_target_dof_pos[env_ids, 2] = (
            torch.zeros_like(self.wrist_target_dof_pos[env_ids, 2]).uniform_(-0.8, 0.8) * self._pd_action_scale[2]
            + self._pd_action_offset[2]
        )

        num_actors = int(self._root_states.shape[0] / self.num_envs)
        global_indices = torch.arange(self.num_envs * num_actors, dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(global_indices),
            len(global_indices),
        )

        if self.num_dof != self.humanoid_num_dof:
            dof_actor_ids = torch.tensor([0] + self.else_dof_actor_indices, dtype=torch.int64, device=self.device)
            multi_env_ids_int32 = global_indices.view(self.num_envs, -1)[
                env_ids[:, None], dof_actor_ids[None]
            ].flatten()  # 0 is humanoid index
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(multi_env_ids_int32),
                len(multi_env_ids_int32),
            )
        else:
            multi_env_ids_int32 = global_indices.view(self.num_envs, -1)[env_ids, :1].flatten()
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(multi_env_ids_int32),
                len(multi_env_ids_int32),
            )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions):
        self.actions[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]] = actions.to(self.device).clone()
        self.actions[:, [9, 10, 12]] = self.actions[:, 8, None].clone()
        # self.actions[:, [8, 9, 10, 12]] = (self.actions[:, [8, 9, 10, 12]] * 2).clip(-1, 1)
        # print(self.actions[0, [4, 8, 9, 10, 12]].abs().mean())
        # print(self._else_contact_forces[:, 0].pow(2).sum(dim=-1).sqrt().item())

        if (self.progress_buf[0].item() == 0) and self.save_mode:
            print(self.mean_object_contact_forces)
        if MODE == "left":
            self.actions[:, 1] = -self.actions[:, 1].clone()

        if self._pd_control:
            self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(self.cur_targets)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
            self._approx_pd_force = torch.abs(
                self.humanoid_dof_stiffness[None]
                * (self.cur_targets[..., : self.humanoid_num_dof] - self._humanoid_dof_pos)
                - self.humanoid_dof_damping[None] * self._humanoid_dof_vel
            )

        # # apply force
        forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        axial_force_ids = torch.where(self.is_axial_forces)[0]

        handle_idx = self.else_rigid_body_names.index("capsule")
        if not self.save_mode:
            forces[:, self.humanoid_num_bodies + handle_idx] = (
                self.handle_forces * (1.0 + self.progress_buf / 30)[:, None]
            )
            forces[axial_force_ids, self.humanoid_num_bodies + handle_idx] = (
                my_quat_rotate(self._else_root_states[axial_force_ids, 0, 3:7], self.unit_x[axial_force_ids])
                * self.handle_forces_mag[axial_force_ids]
                * (1.0 + self.progress_buf[axial_force_ids] / 30)[:, None]
            )
            torques[:, self.humanoid_num_bodies + handle_idx] = (
                self.handle_torques[:] * (1.0 + self.progress_buf / 30)[:, None]
            )
        else:
            forces[:, self.humanoid_num_bodies + handle_idx] = self.handle_forces[:]
            forces[axial_force_ids, self.humanoid_num_bodies + handle_idx] = (
                my_quat_rotate(self._else_root_states[axial_force_ids, 0, 3:7], self.unit_x[axial_force_ids])
                * self.handle_forces_mag[axial_force_ids]
            )
            torques[:, self.humanoid_num_bodies + handle_idx] = self.handle_torques[:]

        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE
        )

        if self.save_mode:
            save_cond = self.progress_buf >= 1
            save_ids = torch.where(save_cond)[0]
            if save_ids.shape[0] > 0:
                save_obs = self._compute_observations_for_saving(save_ids)
                if self.save_wo_wrist:
                    save_acts = self.actions[save_ids, 3:]
                else:
                    save_acts = self.actions[save_ids]
                save_tensors = torch.cat([save_acts, save_obs], dim=-1)

                self.temp_save_buffer[save_ids, self.temp_save_count[save_ids]] = save_tensors
                self.temp_save_count[save_ids] += 1

                # debugging average contact forces of capsule
                self.temp_object_contact_forces[save_ids, self.temp_save_count[save_ids]] = (
                    self._else_contact_forces[save_ids, 0].pow(2).sum(dim=-1).sqrt()
                )

        return

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, mode="rgb_array", viewing_target="whole_body"):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render(mode)
        return

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        self.key_body_name_to_idx = dict()
        for idx, body_name in enumerate(KEY_BODY_NAMES):
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert body_id != -1
            body_ids.append(body_id)
            self.key_body_name_to_idx[body_name] = idx

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            try:
                assert body_id != -1
            except:
                raise ValueError("%s does not exist" % (body_name))
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        scaled_action = (
            self._pd_action_offset[self.actuated_dof_indices]
            + self._pd_action_scale[self.actuated_dof_indices] * action
        )
        self.cur_targets[:, self.actuated_dof_indices] = scaled_action
        # self.cur_targets[:, self.actuated_dof_indices[3:]] = self._pd_action_offset[self.actuated_dof_indices[3:]] + 0.5 * self._pd_action_scale[self.actuated_dof_indices[3:]]
        return

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0)

        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    # AMP + MPL
    dof_obs_size = 16
    dof_offsets = [
        0,  # 'right_wrist_PRO', 'right_wrist_UDEV', 'right_wrist_FLEX',
        3,  # 'right_thumb_ABD',
        4,  # 'right_thumb_MCP'
        5,  # 'right_thumb_PIP'
        6,  # 'right_thumb_DIP'
        7,  # 'right_index_ABD
        8,  # 'right_index_MCP'
        9,  # 'right_index_PIP',
        10,  # 'right_index_DIP',
        11,  # 'right_middle_MCP',
        12,  # 'right_middle_PIP',
        13,  # 'right_middle_DIP',
        14,  # 'right_ring_ABD',
        15,  # 'right_ring_MCP',
        16,  # 'right_ring_PIP',
        17,  # 'right_ring_DIP',
        18,  # 'right_pinky_ABD',
        19,  # 'right_pinky_MCP',
        20,  # 'right_pinky_PIP',
        21,  # 'right_pinky_DIP',
        22,  # END
    ]
    NOT_ACTUATED_DOF_INDICES = [9, 10, 12, 13, 14, 16, 17, 20, 21]

    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset : (dof_offset + dof_size)]
        if dof_offset in NOT_ACTUATED_DOF_INDICES:
            continue
        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset : (dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs


@torch.jit.script
def dof_to_obs_wo_wrist(pose):
    # type: (Tensor) -> Tensor
    # AMP + MPL
    dof_obs_size = 10
    dof_offsets = [
        0,  # 'right_wrist_PRO', 'right_wrist_UDEV', 'right_wrist_FLEX',
        3,  # 'right_thumb_ABD',
        4,  # 'right_thumb_MCP'
        5,  # 'right_thumb_PIP'
        6,  # 'right_thumb_DIP'
        7,  # 'right_index_ABD
        8,  # 'right_index_MCP'
        9,  # 'right_index_PIP',
        10,  # 'right_index_DIP',
        11,  # 'right_middle_MCP',
        12,  # 'right_middle_PIP',
        13,  # 'right_middle_DIP',
        14,  # 'right_ring_ABD',
        15,  # 'right_ring_MCP',
        16,  # 'right_ring_PIP',
        17,  # 'right_ring_DIP',
        18,  # 'right_pinky_ABD',
        19,  # 'right_pinky_MCP',
        20,  # 'right_pinky_PIP',
        21,  # 'right_pinky_DIP',
        22,  # END
    ]
    NOT_ACTUATED_DOF_INDICES = [0, 9, 10, 12, 13, 14, 16, 17, 20, 21]

    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset : (dof_offset + dof_size)]
        if dof_offset in NOT_ACTUATED_DOF_INDICES:
            continue
        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset : (dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs


@torch.jit.script
def compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    dof_obs = dof_to_obs(dof_pos)

    ACTUATED_DOF_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 18, 19]
    dof_vel = dof_vel[:, ACTUATED_DOF_INDICES]

    obs = torch.cat((dof_obs, dof_vel), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_observations_wo_wrist(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    dof_obs = dof_to_obs_wo_wrist(dof_pos)

    ACTUATED_DOF_INDICES = [3, 4, 5, 6, 7, 8, 11, 15, 18, 19]
    dof_vel = dof_vel[:, ACTUATED_DOF_INDICES]

    obs = torch.cat((dof_obs, dof_vel), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.zeros_like(obs_buf[:, 0])
    return reward


@torch.jit.script
def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    contact_body_ids,
    rigid_body_pos,
    max_episode_length,
    enable_early_termination,
    termination_height,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        # fall_height = rigid_body_pos[:,:3, -1].mean(dim=-1) - rigid_body_pos[:, contact_body_ids[-2:], -1].min(dim=-1).values < termination_height
        has_fallen = torch.logical_and(fall_contact, fall_height)
        # flip_torso_pelvis = rigid_body_pos[:, 1, -1] - rigid_body_pos[:, 0, -1] < 0
        # flip_head_torso = rigid_body_pos[:, 2, -1] - rigid_body_pos[:, 1, -1] < 0
        # flip_head_pelvis = rigid_body_pos[:, 2, -1] - rigid_body_pos[:, 0, -1] < 0
        # has_flipped = torch.logical_or(flip_torso_pelvis, flip_head_torso)
        # has_flipped = torch.logical_or(has_flipped, flip_head_pelvis)

        # need_reset = torch.logical_or(has_fallen, has_flipped)
        # if has_fallen[0]:
        #     breakpoint()
        need_reset = has_fallen

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        need_reset *= progress_buf > 1
        terminated = torch.where(need_reset, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
