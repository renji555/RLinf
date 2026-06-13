# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dual-GELLO intervention wrapper for dual-arm Franka environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.gello.gello_expert import GelloExpert


class DualGelloIntervention(gym.ActionWrapper):
    """Override the policy action with two GELLO teleoperation devices.

    Args:
        env: The wrapped dual-arm environment.
        left_port: Serial port for the left GELLO device.
        right_port: Serial port for the right GELLO device.
        gripper_enabled: Whether the gripper channel is present.
    """

    def __init__(
        self,
        env: gym.Env,
        left_port: str,
        right_port: str,
        gripper_enabled: bool = True,
        smooth_intervene: bool = False,
        is_collect: bool = False,
    ):
        super().__init__(env)
        self.gripper_enabled = gripper_enabled
        self.left_expert = GelloExpert(port=left_port)
        self.right_expert = GelloExpert(port=right_port)
        self._smooth_intervene = smooth_intervene
        self.smooth_intervene_mode = False
        self._vr_actions = ["left_arm", "right_arm"]
        self.is_collect = is_collect
        self._current_states = {
            "left_arm": np.zeros(7, dtype=np.float32),
            "right_arm": np.zeros(7, dtype=np.float32),
        }
        self._last_expert_action = {
            "left_arm": np.zeros(7, dtype=np.float32),
            "right_arm": np.zeros(7, dtype=np.float32),
        }

    def _compose_current_action(self) -> np.ndarray:
        final_action = []
        for key in self._vr_actions:
            final_action.extend(
                np.asarray(self._current_states[key], dtype=np.float32).tolist()
            )
        return np.asarray(final_action, dtype=np.float32)
    
    def set_smooth_intervene_mode(self, mode: bool):
        self.smooth_intervene_mode = mode
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.smooth_intervene_mode = False
        self._current_states = {
            "left_arm": np.zeros(7, dtype=np.float32),
            "right_arm": np.zeros(7, dtype=np.float32),
        }
        self._last_expert_action = {
            "left_arm": np.zeros(7, dtype=np.float32),
            "right_arm": np.zeros(7, dtype=np.float32),
        }
        return obs, info

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self.left_expert.ready or not self.right_expert.ready:
            expert_valid = False
        else:
            tcp_pose = self.get_wrapper_attr("get_tcp_pose")()
            action_scale = self.get_wrapper_attr("get_action_scale")()  
            # tcp_pose is (14,) = [left_7, right_7]
            left_tcp = tcp_pose[:7]
            right_tcp = tcp_pose[7:]

            left_a = self._compute_delta(self.left_expert, left_tcp, action_scale)
            right_a = self._compute_delta(self.right_expert, right_tcp, action_scale)
            any_active = (
                np.linalg.norm(left_a[:6]) > 0.001 or np.linalg.norm(right_a[:6]) > 0.001
            )
            if self.gripper_enabled:
                any_active = (
                    any_active or np.abs(left_a[6]) > 0.5 or np.abs(right_a[6]) > 0.5
                )
            expert_valid = any_active
        should_hold = self._smooth_intervene and self.smooth_intervene_mode and not expert_valid
        # If expert is valid, there is no reason to do other actions but update current states by expert action
        # If smooth intervene mode is active, and the expert is not valid, we should hold last expert action
        # If not collect mode, we should update current states by policy action
        # If collect mode, we should keep current states
        if expert_valid:
            self._current_states["left_arm"] = left_a.copy()
            self._current_states["right_arm"] = right_a.copy()
            self._last_expert_action["left_arm"] = left_a.copy()
            self._last_expert_action["right_arm"] = right_a.copy()
        elif should_hold:
            self._current_states["left_arm"] = self._last_expert_action["left_arm"].copy()
            self._current_states["right_arm"] = self._last_expert_action["right_arm"].copy()
        else:
            if not self.is_collect:
                self._current_states["left_arm"] = action[:7].copy()
                self._current_states["right_arm"] = action[7:].copy()

        action = self._compose_current_action()
        return action, expert_valid

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict, float, bool, bool, dict]:
        """Run one env step, replacing the action when the operator is active."""
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        in_smooth = self._smooth_intervene and self.smooth_intervene_mode
        if replaced or in_smooth:
            info["intervene_action"] = new_action
            info["intervene_flag"] = np.ones(1)
        return obs, rew, done, truncated, info

    def _compute_delta(
        self,
        expert: GelloExpert,
        tcp_pose: np.ndarray,
        action_scale: np.ndarray,
    ) -> np.ndarray:
        """Compute delta action from GELLO target pose relative to current TCP."""
        target_pos, target_quat, target_gripper = expert.get_action()
        r_target = R.from_quat(target_quat.copy())
        tcp_pos = tcp_pose[:3]
        r_tcp = R.from_quat(tcp_pose[3:].copy())

        delta_pos = (target_pos - tcp_pos) / action_scale[0]
        delta_euler = (r_target * r_tcp.inv()).as_euler("xyz") / action_scale[1]

        expert_a = np.clip(np.concatenate([delta_pos, delta_euler]), -1.0, 1.0)

        if self.gripper_enabled:
            grip = target_gripper / action_scale[2]
            grip = np.clip(-(2 * grip - 1.0), -1.0, 1.0)
            expert_a = np.concatenate([expert_a, grip])

        return expert_a
