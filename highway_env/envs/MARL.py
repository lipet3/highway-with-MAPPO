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
            "vehicles_count": 5,
            "controlled_vehicles": 3,
            "duration": 40,
            "ego_spacing": 1,
            "vehicles_density": 1,
            "collision_reward": -100.0,   # 碰撞惩罚
            "speed_reward": 5.0,          # 速度奖励系数
            "initial_lane_id": None,
            "reward_speed_range": [0, 30],
            "normalize_reward": False,
            "offroad_terminal": True,
            "screen_width": 1600,  # 或 1920
            "screen_height": 900,  # 或 1080
            "scaling": 3.0,  # 默认常见是 5.0；改小=拉远（试 3.0 或 2.5）
            "centering_position": [0.5, 0.6],  # 视野中心（横向0~1, 纵向0~1）
            "lane_change_reward": 100.0,  # 每次换道奖励

        })
        return config

    # ---------- 回合汇总打印 ----------
    def _print_episode_summary(self) -> None:
        """打印上一回合累计统计 + 每10回合区间平均"""
        if not getattr(self, "_ep_initialized", False):
            return

        # 第一次 reset 前若未初始化，保护
        if not hasattr(self, "_ep_count"):
            self._ep_count = 0
        if not hasattr(self, "_last10_team_means"):
            self._last10_team_means = []

        # 当前要打印的是第几回合（从1开始计数）
        ep_no = self._ep_count + 1

        n = self._ep_return.shape[0]
        team_mean = float(self._ep_return.mean())
        per_total = ", ".join([f"{i}:{self._ep_return[i]:.2f}" for i in range(n)])
        per_col = ", ".join([f"{i}:{self._ep_components['collision'][i]:.2f}" for i in range(n)])
        per_spd = ", ".join([f"{i}:{self._ep_components['speed'][i]:.2f}" for i in range(n)])
        per_lc = ", ".join([f"{i}:{self._ep_components['lane_change'][i]:.2f}" for i in range(n)])

        print(f"[EP SUM] components per-agent | lane_change [{per_lc}]")
        print(f"[EP SUM] ep={ep_no} steps={self._ep_step_idx} | per-agent return [{per_total}] | team_mean={team_mean:.2f}")
        print(f"[EP SUM] components per-agent | collision [{per_col}] | speed [{per_spd}]")

        # 10回合区间统计
        self._last10_team_means.append(team_mean)
        if len(self._last10_team_means) == 10:
            start_ep = ep_no - 9
            end_ep = ep_no
            block_avg = float(np.mean(self._last10_team_means))
            print(f"[EP AVG] {start_ep}-{end_ep} team_mean={block_avg:.2f}")
            # 如果希望“滑窗”而不是分段清零，把这一行换成：self._last10_team_means.pop(0)
            self._last10_team_means.clear()

    def _reset(self) -> None:
        # ------------- 新增：回合计数与打印上一回合 -------------
        # 首次进入时初始化计数与缓存
        if not hasattr(self, "_ep_count"):
            self._ep_count = 0
        if not hasattr(self, "_last10_team_means"):
            self._last10_team_means = []

        # 如果上一回合结束过，则先打印上一回合汇总并“回合+1”
        if getattr(self, "_episode_done", False) and getattr(self, "_ep_initialized", False):
            self._print_episode_summary()
            self._episode_done = False
            self._ep_count += 1
        # ------------------------------------------------------

        # 初始化本回合统计器
        n_agents = int(self.config["controlled_vehicles"])
        self._ep_step_idx = 0
        self._ep_return = np.zeros(n_agents, dtype=float)
        self._ep_components = {



            "collision": np.zeros(n_agents, dtype=float),
            "speed": np.zeros(n_agents, dtype=float),
            "lane_change": np.zeros(n_agents, dtype=float),  # + 新增的换道奖励 用于测试



        }
        self._ep_initialized = True

        self._create_road()
        self._create_vehicles()

        self._prev_lane_index = [v.lane_index for v in self.controlled_vehicles]
        self._ep_crash_given = np.zeros(n_agents, dtype=bool)

        n_agents = int(self.config["controlled_vehicles"])
        self._ep_crash_given = np.zeros(n_agents, dtype=bool)  # 新增


        # ==== 新增：为曲线 & 视频做最小初始化（不改变任何决策/奖励逻辑）====
        self.time = 0.0
        self.time_history = []                               # 物理时间（逐物理步）
        self.speed_history = {f"agent_{i}": [] for i in range(len(self.controlled_vehicles))}
        self.accel_history = {f"agent_{i}": [] for i in range(len(self.controlled_vehicles))}
        self._prev_speeds = [v.speed for v in self.controlled_vehicles]
        self._captured_frames = []                           # 逐物理步抓到的帧缓存
        self._record_video = True                            # 给 video.py 一个开关（不影响训练）
        # =======================================================

    # ==== 新增：逐物理步推进并抓帧（不改变动作语义）====
    def _simulate(self, action: Tuple[int, ...] | None = None) -> None:
        """
        用 simulation_frequency 推进物理仿真；对一次 env.step(action)，内部推进多个“物理子步”。
        - 只在第一个子步把离散动作下发给 action_type（保持 highway-env 语义：动作在若干物理步内生效）
        - 每个子步更新 time/speed/accel 历史
        - 如果 _record_video=True，每个子步 render 一帧塞进 _captured_frames（video.py 会优先消耗它）
        """
        sim_freq = int(self.config.get("simulation_frequency", 15))
        pol_freq = int(self.config.get("policy_frequency", 1))
        dt = 1.0 / max(1, sim_freq)
        steps_per_action = max(1, sim_freq // max(1, pol_freq))

        for k in range(steps_per_action):
            # 仅在第一个物理子步把动作下发给 env（避免重复触发 meta-action 的一次性效果）
            if action is not None and k == 0:
                # 这行只通过环境自身的 action_type 下发，不做任何 mask/改写
                self.action_type.act(action)

            # 物理推进一个子步
            self.road.step(dt)

            # 记录时间 & 速度/加速度（逐物理步）
            self.time += dt
            self.time_history.append(self.time)
            for i, v in enumerate(self.controlled_vehicles):
                key = f"agent_{i}"
                self.speed_history[key].append(float(v.speed))
                a = (float(v.speed) - float(self._prev_speeds[i])) / dt
                self.accel_history[key].append(a)
                self._prev_speeds[i] = float(v.speed)

            # 逐物理步抓帧（渲染模式由创建环境时设为 rgb_array；不改变任何显示参数）
            if getattr(self, "_record_video", True):
                try:
                    frame = self.render()   # 新版 highway-env: 不要传 mode
                    if frame is not None:
                        self._captured_frames.append(frame)
                except TypeError:
                    # 极老版本 fallback（通常用不到）：self.render(mode="rgb_array")
                    try:
                        frame = self.render(mode="rgb_array")
                        if frame is not None:
                            self._captured_frames.append(frame)
                    except Exception:
                        pass

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
            #-------------------------碰撞惩罚-------------------
            if v.crashed and not self._ep_crash_given[idx]:
                components["collision"] = float(self.config["collision_reward"])
                self._ep_crash_given[idx] = True
            else:
                components["collision"] = 0.0
            # -------------------------碰撞惩罚-------------------

            # -------------------------速度奖励-------------------
            # 速度奖励
            scaled_speed = v.speed / max(1.0, self.config["reward_speed_range"][1])
            components["speed"] = scaled_speed * self.config["speed_reward"]
            # -------------------------速度奖励-------------------

            #--------换道奖励：检测 lane_index 是否变化 ------------ #
            changed = (v.lane_index != self._prev_lane_index[idx])
            if changed:
                components["lane_change"] = float(self.config["lane_change_reward"])
                self._prev_lane_index[idx] = v.lane_index  # 记住最新车道
            else:
                components["lane_change"] = 0.0
            # --------换道奖励：检测 lane_index 是否变化 ------------ #

            final_reward = float(sum(components.values()))
            rewards.append(final_reward)

            # —— 回合累计（用于回合汇总打印）——
            # 防御：若并发改变了 n_agents（极少见），截断对齐
            if idx < self._ep_return.shape[0]:
                self._ep_return[idx] += final_reward
                self._ep_components["collision"][idx] += components["collision"]
                self._ep_components["speed"][idx] += components["speed"]
                self._ep_components["lane_change"][idx] += components["lane_change"] #新增

            # —— 步级节流打印：每 5 步打印一次（按步计数更稳健）——
            if self._ep_step_idx % 5 == 0:
                formatted = {k: f"{val:.2f}" for k, val in components.items()}
                # print(f"[MARLEnv Reward] Agent {idx}: {formatted} -> Final: {final_reward:.2f}")

        if self._ep_step_idx % 5 == 0 and n_agents > 0:
            mean_reward = float(np.mean(rewards))
            # print(f"[MARLEnv Reward] Mean reward across {n_agents} agents: {mean_reward:.2f}")

        # 递增步计数（放在最后，保证本步使用的 index 一致）
        self._ep_step_idx += 1

        return tuple(rewards)

    def _rewards(self, action: Action) -> Dict[str, Tuple[float, ...]]:
        collisions, speeds, lane_changes = [], [], []
        for i, v in enumerate(self.controlled_vehicles):
            collisions.append(float(v.crashed))
            speeds.append(v.speed / max(1.0, self.config["reward_speed_range"][1]))
            lane_changes.append(1.0 if v.lane_index != self._prev_lane_index[i] else 0.0)
        return {
            "collision": tuple(collisions),
            "speed": tuple(speeds),
            "lane_change": tuple(lane_changes),
        }

    def _is_terminated(self) -> bool:
        any_crash = any(v.crashed for v in getattr(self, "controlled_vehicles", []))
        any_offroad = self.config["offroad_terminal"] and any(
            not v.on_road for v in getattr(self, "controlled_vehicles", [])
        )
        done = any_crash or any_offroad
        if done:
            self._episode_done = True
        return done

    def _is_truncated(self) -> bool:
        """Episode truncated if duration reached."""
        done = self.time >= self.config["duration"]
        if done:
            self._episode_done = True
        return done
