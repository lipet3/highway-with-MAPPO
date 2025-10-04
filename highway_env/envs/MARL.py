from __future__ import annotations
import numpy as np
from typing import Tuple, Dict
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class MARLEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {"type": "MultiAgentObservation", "observation_config": {"type": "Kinematics"}},
            "action": {"type": "MultiAgentAction", "action_config": {"type": "DiscreteMetaAction"}},
            "lanes_count": 2,
            "vehicles_count": 10,
            "controlled_vehicles": 3,
            "duration": 40,
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -100.0,   # 碰撞惩罚
            "speed_reward": 5.0,          # 速度奖励系数
            "initial_lane_id": None,
            "reward_speed_range": [0, 30],
            "normalize_reward": False,
            "offroad_terminal": True,
        })
        return config

    # ---------- 回合汇总打印 ----------
    def _print_episode_summary(self) -> None:
        """打印上一回合累计统计"""
        if not getattr(self, "_ep_initialized", False):
            return
        n = self._ep_return.shape[0]
        team_mean = float(self._ep_return.mean())
        per_total = ", ".join([f"{i}:{self._ep_return[i]:.2f}" for i in range(n)])
        per_col = ", ".join([f"{i}:{self._ep_components['collision'][i]:.2f}" for i in range(n)])
        per_spd = ", ".join([f"{i}:{self._ep_components['speed'][i]:.2f}" for i in range(n)])
        print(f"[EP SUM] steps={self._ep_step_idx} | per-agent return [{per_total}] | team_mean={team_mean:.2f}")
        print(f"[EP SUM] components per-agent | collision [{per_col}] | speed [{per_spd}]")

    def _reset(self) -> None:
        # 如果上一回合已结束，则先打印上一回合的汇总
        if getattr(self, "_episode_done", False) and getattr(self, "_ep_initialized", False):
            self._print_episode_summary()
            self._episode_done = False

        # 初始化本回合统计器
        n_agents = int(self.config["controlled_vehicles"])
        self._ep_step_idx = 0
        self._ep_return = np.zeros(n_agents, dtype=float)
        self._ep_components = {
            "collision": np.zeros(n_agents, dtype=float),
            "speed": np.zeros(n_agents, dtype=float),
        }
        self._ep_initialized = True

        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a straight 2-lane road."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create IDM/HDV vehicles without breaking defaults."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> Tuple[float, ...]:
        """
        极简奖励: 每个 agent 各自的奖励组成 + 打印（每 5 步打印一次）
        同时累计到“回合统计”，用于回合结束时打印汇总。
        """
        rewards: list[float] = []
        n_agents = len(self.controlled_vehicles)

        for idx, v in enumerate(self.controlled_vehicles):
            components = {}
            # 碰撞（当前步是否处于 crashed 状态就扣一次，保持你当前策略，不做“一次性扣分”）
            components["collision"] = self.config["collision_reward"] if v.crashed else 0.0
            # 速度奖励
            scaled_speed = v.speed / max(1.0, self.config["reward_speed_range"][1])
            components["speed"] = scaled_speed * self.config["speed_reward"]

            final_reward = float(sum(components.values()))
            rewards.append(final_reward)

            # —— 回合累计（用于回合汇总打印）——
            # 防御：若并发改变了 n_agents（极少见），截断对齐
            if idx < self._ep_return.shape[0]:
                self._ep_return[idx] += final_reward
                self._ep_components["collision"][idx] += components["collision"]
                self._ep_components["speed"][idx] += components["speed"]

            # —— 步级节流打印：每 5 步打印一次（按步计数更稳健）——
            if self._ep_step_idx % 5 == 0:
                formatted = {k: f"{val:.2f}" for k, val in components.items()}
                print(f"[MARLEnv Reward] Agent {idx}: {formatted} -> Final: {final_reward:.2f}")

        if self._ep_step_idx % 5 == 0 and n_agents > 0:
            mean_reward = float(np.mean(rewards))
            print(f"[MARLEnv Reward] Mean reward across {n_agents} agents: {mean_reward:.2f}")

        # 递增步计数（放在最后，保证本步使用的 index 一致）
        self._ep_step_idx += 1

        return tuple(rewards)

    def _rewards(self, action: Action) -> Dict[str, Tuple[float, ...]]:
        """
        返回奖励组成，逐 agent 输出（给 info 或调试用）
        """
        collisions = []
        speeds = []
        for v in self.controlled_vehicles:
            collisions.append(float(v.crashed))
            speeds.append(v.speed / max(1.0, self.config["reward_speed_range"][1]))
        return {
            "collision": tuple(collisions),
            "speed": tuple(speeds),
        }

    def _is_terminated(self) -> bool:
        """Episode ends if ego crashes or goes offroad."""
        done = (
            self.vehicle.crashed
            or self.config["offroad_terminal"] and not self.vehicle.on_road
        )
        if done:
            self._episode_done = True
        return done

    def _is_truncated(self) -> bool:
        """Episode truncated if duration reached."""
        done = self.time >= self.config["duration"]
        if done:
            self._episode_done = True
        return done


# # 文件名: highway_env/envs/MARL.py
#
# from __future__ import annotations
# import numpy as np
# from typing import Tuple, Dict
# from highway_env import utils
# from highway_env.envs.common.abstract import AbstractEnv
# from highway_env.envs.common.action import Action
# from highway_env.road.road import Road, RoadNetwork
# from highway_env.utils import near_split
# from highway_env.vehicle.controller import ControlledVehicle
# from highway_env.vehicle.kinematics import Vehicle
#
# Observation = np.ndarray
#
#
# class MARLEnv(AbstractEnv):
#
#
#     @classmethod
#     def default_config(cls) -> dict:
#         config = super().default_config()
#         config.update({
#             "observation": {"type": "MultiAgentObservation", "observation_config": {"type": "Kinematics"}},
#             "action": {"type": "MultiAgentAction", "action_config": {"type": "DiscreteMetaAction"}},
#             "lanes_count": 2,
#             "vehicles_count": 10,
#             "controlled_vehicles": 3,
#             "duration": 40,
#             "ego_spacing": 2,
#             "vehicles_density": 1,
#             "collision_reward": -100.0,   # 碰撞惩罚
#             "speed_reward": 5.0,          # 速度奖励系数
#             "initial_lane_id": None,
#             "reward_speed_range": [0, 30],
#             "normalize_reward": False,
#             "offroad_terminal": True,
#         })
#         return config
#
#     def _reset(self) -> None:
#         self._create_road()
#         self._create_vehicles()
#
#     def _create_road(self) -> None:
#         """Create a straight 2-lane road."""
#         self.road = Road(
#             network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
#             np_random=self.np_random,
#             record_history=self.config["show_trajectories"],
#         )
#
#     def _create_vehicles(self) -> None:
#         """Create IDM/HDV vehicles without breaking defaults."""
#         other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
#         other_per_controlled = near_split(
#             self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
#         )
#
#         self.controlled_vehicles = []
#         for others in other_per_controlled:
#             vehicle = Vehicle.create_random(
#                 self.road,
#                 speed=25,
#                 lane_id=self.config["initial_lane_id"],
#                 spacing=self.config["ego_spacing"],
#             )
#             vehicle = self.action_type.vehicle_class(
#                 self.road, vehicle.position, vehicle.heading, vehicle.speed
#             )
#             self.controlled_vehicles.append(vehicle)
#             self.road.vehicles.append(vehicle)
#
#             for _ in range(others):
#                 vehicle = other_vehicles_type.create_random(
#                     self.road, spacing=1 / self.config["vehicles_density"]
#                 )
#                 vehicle.randomize_behavior()
#                 self.road.vehicles.append(vehicle)
#
#     def _reward(self, action: Action) -> Tuple[float, ...]:
#         """
#         极简奖励: 每个 agent 各自的奖励组成 + 打印 (每 5 步打印一次)
#         """
#         rewards = []
#
#         for idx, v in enumerate(self.controlled_vehicles):
#             components = {}
#             components["collision"] = self.config["collision_reward"] if v.crashed else 0.0
#             scaled_speed = v.speed / max(1.0, self.config["reward_speed_range"][1])
#             components["speed"] = scaled_speed * self.config["speed_reward"]
#
#             final_reward = sum(components.values())
#             rewards.append(final_reward)
#
#             # 每 5 步才打印一次
#             if self.time % 5 == 0:
#                 formatted = {k: f"{val:.2f}" for k, val in components.items()}
#                 print(f"[MARLEnv Reward] Agent {idx}: {formatted} -> Final: {final_reward:.2f}")
#
#         if self.time % 5 == 0:
#             mean_reward = np.mean(rewards)
#             print(f"[MARLEnv Reward] Mean reward across {len(self.controlled_vehicles)} agents: {mean_reward:.2f}")
#
#         return tuple(rewards)
#
#
#     def _rewards(self, action: Action) -> Dict[str, Tuple[float, ...]]:
#         """
#         返回奖励组成，逐 agent 输出
#         """
#         collisions = []
#         speeds = []
#
#         for v in self.controlled_vehicles:
#             collisions.append(float(v.crashed))
#             speeds.append(v.speed / max(1.0, self.config["reward_speed_range"][1]))
#
#         return {
#             "collision": tuple(collisions),
#             "speed": tuple(speeds),
#         }
#
#
#
#     def _is_terminated(self) -> bool:
#         """Episode ends if ego crashes or goes offroad."""
#         return (
#             self.vehicle.crashed
#             or self.config["offroad_terminal"] and not self.vehicle.on_road
#         )
#
#     def _is_truncated(self) -> bool:
#         """Episode truncated if duration reached."""
#         return self.time >= self.config["duration"]




# # In MARL.py (添加绘制图像之前)
#
# from __future__ import annotations
# import numpy as np
# from typing import Tuple, Dict
# from highway_env.envs.common.abstract import AbstractEnv
# from highway_env.road.lane import StraightLane, LineType
# from highway_env.road.road import Road, RoadNetwork
# from highway_env.vehicle.kinematics import Vehicle
# import time
# Observation = np.ndarray
#
#
# class MARLEnv(AbstractEnv):
#     metadata = {"render_modes": ["human", "rgb_array"], "name": "MARLEnv-v0"}
#
#     @classmethod
#     def default_config(cls) -> dict:
#         # ... (这部分完全不变) ...
#         config = super().default_config()
#         config.update({
#             "observation": {"type": "MultiAgentObservation", "observation_config": {"type": "Kinematics"}},
#             "action": {"type": "MultiAgentAction", "action_config": {"type": "DiscreteMetaAction"}},
#             "lanes_count": 2,
#             "vehicles_count": 1,
#             "controlled_vehicles": 4,
#             "duration": 40,
#             "offroad_terminal": True,
#             "collision_reward": -800,
#             "high_speed_reward": 1,
#             "low_speed_penalty": -2,
#             "lane_change_reward": 200,
#             "timeout_penalty": -500,
#             "low_speed_threshold": 15,
#             "high_speed_threshold": 25,
#             "max_steps": 2000,  # --- NEW: 最大步数超时终止条件
#             "screen_width": 1200,  # 画面更宽
#             "screen_height": 300,  # 高度适中
#         })
#         return config
#
#     def _reset(self) -> None:
#         self._create_road()
#         self._create_vehicles()
#         self.agent_a_lane_changed = False
#         self.steps = 0   # --- NEW: 步数计数器simulation_frequency
#
#     # 在 MARLEnv 类中，添加这个 reset 方法
#     # 在 MARLEnv.py 中，添加这个 reset 方法
#     def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[Observation, Dict]:
#         super().reset(seed=seed)
#         self._reset()  # 调用您自己的 _reset 来重置路况和车辆
#
#         # 获取初始观测
#         obs = self.observation_type.observe()
#
#         # ✨ 关键: 调用我们修改后的 _info 方法来获取包含所有信息的初始 info
#         info = self._info(obs, action=None)
#
#         return obs, info
#
#     def _create_road(self) -> None:
#         # ... (这部分完全不变) ...
#         net = RoadNetwork()
#         lane_width = 4.0
#         road_length = 1000.0
#         lane_0_start, lane_0_end = np.array([0.0, 0.0]), np.array([road_length, 0.0])
#         lane_1_start, lane_1_end = np.array([0.0, lane_width]), np.array([road_length, lane_width])
#         net.add_lane("a", "b", StraightLane(lane_0_start, lane_0_end, width=lane_width,
#                                             line_types=(LineType.STRIPED, LineType.CONTINUOUS_LINE)))
#         net.add_lane("a", "b", StraightLane(lane_1_start, lane_1_end, width=lane_width,
#                                             line_types=(LineType.STRIPED, LineType.STRIPED)))
#         self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
#
#     def _create_vehicles(self) -> None:
#         # ... (这部分完全不变) ...
#         self.controlled_vehicles = []
#         vehicle_b = self.action_type.vehicle_class(self.road, self.road.network.get_lane(("a", "b", 0)).position(20, 0),
#                                                    speed=20)
#         self.controlled_vehicles.append(vehicle_b)
#         vehicle_c = self.action_type.vehicle_class(self.road, self.road.network.get_lane(("a", "b", 0)).position(80, 0),
#                                                    speed=20)
#         self.controlled_vehicles.append(vehicle_c)
#         pos_b_x, pos_c_x = vehicle_b.position[0], vehicle_c.position[0]
#         random_x = self.np_random.uniform(pos_b_x - 40, pos_c_x + 40)
#         vehicle_a = self.action_type.vehicle_class(self.road,
#                                                    self.road.network.get_lane(("a", "b", 1)).position(random_x, 0),
#                                                    speed=20)
#         self.controlled_vehicles.insert(0, vehicle_a)
#         hdv = Vehicle(self.road, self.road.network.get_lane(("a", "b", 1)).position(vehicle_a.position[0] + 30, 0),
#                       speed=15)
#         self.road.vehicles.clear()
#         self.road.vehicles.extend(self.controlled_vehicles)
#         self.road.vehicles.append(hdv)
# # -------------------------------------------------------------
#     # === 在 MARLEnv 类里加： ===
#     def _get_action_mask(self) -> dict[str, np.ndarray]:
#         """
#         为每个智能体生成动作掩码（1=允许, 0=禁止），长度等于离散动作维度。
#         动作名字从 self.action_type.actions 读取，避免顺序错配。
#         """
#         acts = list(getattr(self.action_type, "actions", []))  # 例如 ['LANE_LEFT','LANE_RIGHT','IDLE','FASTER','SLOWER']
#         n_act = len(acts)
#         idx = {name: i for i, name in enumerate(acts)}  # 动作名 -> 索引
#
#         masks: dict[str, np.ndarray] = {}
#         for i, v in enumerate(self.controlled_vehicles):
#             m = np.ones(n_act, dtype=np.int8)
#
#             # --- 你的规则：
#             # agent_0（慢车道起步）：永远不能右拐；如果已经在快车道(索引0)则也不能再左拐
#             if i == 0:
#                 if "LANE_RIGHT" in idx:
#                     m[idx["LANE_RIGHT"]] = 0
#                 # 已在快车道：禁左拐
#                 if getattr(v, "lane_index", (None, None, None))[2] == 0 and "LANE_LEFT" in idx:
#                     m[idx["LANE_LEFT"]] = 0
#
#             # agent_1 / agent_2（快车道）：禁止左右变道，只能纵向控速/保持
#             else:
#                 if "LANE_LEFT" in idx:
#                     m[idx["LANE_LEFT"]] = 0
#                 if "LANE_RIGHT" in idx:
#                     m[idx["LANE_RIGHT"]] = 0
#                 # 纵向动作如 FASTER / SLOWER / IDLE 都保留
#
#             masks[f"agent_{i}"] = m
#         return masks
#
#     # -------------------------------------------------------------
#     # -------------------------------------------------------------
#     def get_global_state(self) -> np.ndarray:
#         """
#         返回全局状态，用于 MAPPO 的 Critic。
#         可以是所有 agent 的观测拼接在一起。
#         """
#         obs = self.observation_type.observe()
#         # obs 是 tuple，每个 agent 一个 (ndarray)
#         return np.concatenate([o.flatten() for o in obs], axis=0)
#
#     # 在 MARLEnv.py 中，用这个版本替换掉您现有的 step 方法
#     def step(self, action: Tuple[int, ...]) -> Tuple[
#         Observation, Tuple[float, ...], Tuple[bool, ...], Tuple[bool, ...], Dict]:
#
#         # 动作屏蔽逻辑
#         mutable_action = list(action)
#         vehicle_a = self.controlled_vehicles[0]
#         if self.agent_a_lane_changed or vehicle_a.lane_index[2] == 0:
#             self.agent_a_lane_changed = True
#             if mutable_action[0] == 0: mutable_action[0] = 1
#         if mutable_action[0] == 2: mutable_action[0] = 1
#         for i in range(1, 3):
#             if mutable_action[i] in [0, 2]:
#                 mutable_action[i] = 1
#
#         # 执行仿真
#         self._simulate(tuple(mutable_action))
#         self.steps += 1
#         obs = self.observation_type.observe()
#         rewards_dict = self._rewards(tuple(mutable_action))
#         rewards = tuple(rewards_dict[f"agent_{i}"] for i in range(len(self.controlled_vehicles)))
#         terminateds = self._is_terminated()
#         truncateds = self._is_truncated()
#
#         # --- info 必须包含 action_mask 和 global_state ---
#         info = self._info(obs, tuple(mutable_action))
#         masks = self._get_action_mask()
#         for k in info.keys():
#             info[k]["action_mask"] = masks[k]
#
#         global_state = self.get_global_state()
#
#         # 每个 agent 的 info 都有 global_state
#         for k in info.keys():
#             info[k]["global_state"] = global_state
#
#         # 关键：额外在最外层 info 加一份全局的
#         info["global_state"] = global_state
#
#         return obs, rewards, terminateds, truncateds, info
#
#     #  --------------------------------------------------------
#     def _simulate(self, action: Tuple[int, ...] | None = None) -> None:
#         """推进物理仿真，并记录速度/加速度/时间，同时捕获渲染帧"""
#         sim_freq = self.config["simulation_frequency"]
#         pol_freq = self.config["policy_frequency"]
#         dt = 1 / sim_freq
#         steps_per_action = int(sim_freq / pol_freq)
#
#         # 初始化历史记录
#         if not hasattr(self, "time"):
#             self.time = 0.0
#         if not hasattr(self, "time_history"):
#             self.time_history = []
#             self.speed_history = {f"agent_{i}": [] for i in range(len(self.controlled_vehicles))}
#             self.accel_history = {f"agent_{i}": [] for i in range(len(self.controlled_vehicles))}
#             self.prev_speeds = [v.speed for v in self.controlled_vehicles]
#         if not hasattr(self, "_captured_frames"):
#             self._captured_frames = []
#
#         for _ in range(steps_per_action):
#             # ✅ 应用智能体动作到车辆
#             if action is not None:
#                 self.action_type.act(action)
#             # 物理推进
#             self.road.step(dt)
#
#             # 更新时间
#             self.time += dt
#             self.time_history.append(self.time)
#
#             # 记录速度和加速度
#             for i, v in enumerate(self.controlled_vehicles):
#                 agent = f"agent_{i}"
#                 self.speed_history[agent].append(v.speed)
#                 accel = (v.speed - self.prev_speeds[i]) / dt
#                 self.accel_history[agent].append(accel)
#                 self.prev_speeds[i] = v.speed
#
#             # 捕获渲染帧（每个物理步都渲染）
#             # if hasattr(self, "render") and callable(self.render):
#             #     frame = self.render()
#             #     if frame is not None:
#             #         self._captured_frames.append(frame)
#             # 捕获渲染帧（仅在评估/录制模式才执行）
#             if getattr(self, "_record_video", False):
#                 if hasattr(self, "render") and callable(self.render):
#                     # frame = self.render(mode="rgb_array") if "rgb_array" in self.metadata["render_modes"] else self.render()
#                     frame = self.render()  # 不要传 mode （在评估时调用这个
#                     if frame is not None:
#                         self._captured_frames.append(frame)
#
#
#     def _rewards(self, action: Tuple[int, ...]) -> dict[str, float]:
#         dt = 1.0 / float(self.config.get("simulation_frequency", 15))
#
#         v_target = 30.0
#         v_low = 25.0
#
#         W = {
#             "collision": -600.0,
#             "offroad": -200.0,
#             "speed": 5.0,
#             "low_speed": -5.0,
#             "lane_bonus": 50.0,
#             "success": 300.0,  # 成功奖励
#             "timeout": -200.0
#         }
#
#         rewards_dict: dict[str, float] = {}
#
#         # --- 个体奖励 ---
#         for i, v in enumerate(self.controlled_vehicles):
#             r = 0.0
#             if getattr(v, "crashed", False):
#                 r += W["collision"]
#             if not getattr(v, "on_road", True):
#                 r += W["offroad"]
#
#             speed = float(getattr(v, "speed", 0.0))
#             r += W["speed"] * (speed / v_target)
#
#             if speed < v_low:
#                 penalty_frac = 1.0 - (speed / max(1.0, v_low))
#                 r += W["low_speed"] * penalty_frac
#
#             rewards_dict[f"agent_{i}"] = dt * r
#
#         # --- agent_0 换道奖励 ---
#         v0 = self.controlled_vehicles[0]
#         if not getattr(self, "agent_a_lane_changed", False) and v0.lane_index[2] == 0:
#             rewards_dict["agent_0"] += W["lane_bonus"]
#             self.agent_a_lane_changed = True
#
#         # --- 成功奖励（最靠后的 agent 超过 HDV +50m） ---
#         hdv = next((v for v in self.road.vehicles if v not in self.controlled_vehicles), None)
#         if hdv is not None:
#             min_agent_x = min(v.position[0] for v in self.controlled_vehicles)
#             if min_agent_x > hdv.position[0] + 50.0:
#                 for i in range(len(self.controlled_vehicles)):
#                     rewards_dict[f"agent_{i}"] += W["success"] / len(self.controlled_vehicles)
#                 # 标记成功（用于终止条件）
#                 self.success_reached = True
#
#         # --- 超时惩罚 ---
#         if self.time >= self.config["duration"] or self.steps >= self.config.get("max_steps", 2000):
#             for i in range(len(self.controlled_vehicles)):
#                 rewards_dict[f"agent_{i}"] += W["timeout"] / len(self.controlled_vehicles)
#
#         return rewards_dict
#
#     def _is_terminated(self) -> Tuple[bool, ...]:
#         # 碰撞 / 出界即终止
#         done = any(v.crashed or not v.on_road for v in self.controlled_vehicles)
#
#         # 最靠后车超过 HDV + 50m 即终止
#         hdv = next((v for v in self.road.vehicles if v not in self.controlled_vehicles), None)
#         if hdv is not None:
#             min_agent_x = min(v.position[0] for v in self.controlled_vehicles)
#             if min_agent_x > hdv.position[0] + 50.0:
#                 done = True
#
#         return tuple(done for _ in self.controlled_vehicles)
#
#     def _is_truncated(self) -> Tuple[bool, ...]:
#         time_trunc = self.time >= self.config["duration"]
#         step_trunc = self.steps >= self.config["max_steps"]
#         return tuple(time_trunc or step_trunc for _ in self.controlled_vehicles)
#
#     # 在 MARLEnv 类中，替换掉旧的 _info 方法
#     # 在 MARLEnv.py 中，用这个版本替换掉 _info 方法
#     # 在 MARLEnv.py 中
#     def _info(self, obs: tuple, action: tuple | None) -> dict:
#         info = {f"agent_{i}": {} for i in range(len(self.controlled_vehicles))}
#         masks = self._get_action_mask()
#         global_state = self.get_global_state()
#
#         for i, agent_id_key in enumerate(info.keys()):
#             agent_name = f"agent_{i}"
#             info[agent_id_key]["action_mask"] = masks[agent_name]
#             info[agent_id_key]["global_state"] = global_state
#
#         return info