# highway_env/envs/MARL.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict
import functools
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.envs.common.observation import ObservationType
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


# ===== 角色 one-hot 包装器（安全：只包观测，不改动力学） =====
class RoleAugObsWrapper(ObservationType):
    def __init__(self, env, base_obs: ObservationType):
        super().__init__(env, config={})
        self.base = base_obs

    def space(self):
        sp = self.base.space()
        # 典型：Tuple[Box,...]，每个 agent 一个 Box
        if isinstance(sp, spaces.Tuple) and len(sp) > 0 and isinstance(sp[0], spaces.Box):
            A = len(sp)
            per = sp[0]
            flat_dim = int(np.prod(per.shape)) if per.shape is not None else 0
            new_box = spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(flat_dim + A,), dtype=np.float32)
            return spaces.Tuple(tuple(new_box for _ in range(A)))
        # 兜底：返回原空间（不会启用 concat）
        return sp

    def observe(self):
        base = self.base.observe()            # tuple，长度=A
        A = len(base)
        roles = np.eye(A, dtype=np.float32)
        out = []
        for i in range(A):
            x = np.asarray(base[i], dtype=np.float32).reshape(-1)
            out.append(np.concatenate([x, roles[i]], axis=-1))
        return tuple(out)


class MARLEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {"type": "MultiAgentObservation",
                            "observation_config": {"type": "Kinematics"}},
            "action": {"type": "MultiAgentAction",
                       "action_config": {"type": "DiscreteMetaAction"}},
            "lanes_count": 2,
            "vehicles_count": 5,
            "controlled_vehicles": 3,
            "duration": 40,
            "ego_spacing": 1,
            "vehicles_density": 1,
            "collision_reward": -100.0,
            "speed_reward": 5.0,
            "initial_lane_id": None,
            "reward_speed_range": [0, 30],
            "normalize_reward": False,
            "offroad_terminal": True,
            "screen_width": 1600,
            "screen_height": 900,
            "scaling": 3.0,
            "centering_position": [0.5, 0.6],
            "lane_change_reward": 100.0,

            # 角色 one-hot 注入方式： "off" | "info" | "concat"
            # 默认 "info"：只放到 info，不改观测维度，不影响既有训练
            "role_onehot_mode": "concat",
        })
        return config

    # —— 只在这里接管观测空间，启用 concat 时包一层，不碰其它逻辑 ——
    def define_spaces(self):
        super().define_spaces()
        mode = str(self.config.get("role_onehot_mode", "info")).lower()
        if mode == "concat":
            self.observation_type = RoleAugObsWrapper(self, self.observation_type)
            self.observation_space = self.observation_type.space()

    # ---------- 回合汇总 ----------
    def _print_episode_summary(self) -> None:
        if not getattr(self, "_ep_initialized", False):
            return
        if not hasattr(self, "_ep_count"):
            self._ep_count = 0
        if not hasattr(self, "_last10_team_means"):
            self._last10_team_means = []

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

        self._last10_team_means.append(team_mean)
        if len(self._last10_team_means) == 10:
            start_ep, end_ep = ep_no - 9, ep_no
            block_avg = float(np.mean(self._last10_team_means))
            print(f"[EP AVG] {start_ep}-{end_ep} team_mean={block_avg:.2f}")
            self._last10_team_means.clear()

    def _reset(self) -> None:
        if not hasattr(self, "_ep_count"):
            self._ep_count = 0
        if not hasattr(self, "_last10_team_means"):
            self._last10_team_means = []
        if getattr(self, "_ep_initialized", False) and getattr(self, "_episode_done", False):
            self._print_episode_summary()
            self._ep_count += 1
            self._episode_done = False

        n_cfg_agents = int(self.config["controlled_vehicles"])
        self._ep_step_idx = 0
        self._ep_return = np.zeros(n_cfg_agents, dtype=float)
        self._ep_components = {
            "collision": np.zeros(n_cfg_agents, dtype=float),
            "speed": np.zeros(n_cfg_agents, dtype=float),
            "lane_change": np.zeros(n_cfg_agents, dtype=float),
        }
        self._ep_crash_given = np.zeros(n_cfg_agents, dtype=bool)
        self._ep_initialized = True

        self._create_road()
        self._create_vehicles()

        self._prev_lane_index = [v.lane_index for v in self.controlled_vehicles]

        # 历史
        self.time = 0.0
        self.time_history = []
        n_env_agents = len(self.controlled_vehicles)
        self.speed_history = {f"agent_{i}": [] for i in range(n_env_agents)}
        self.accel_history = {f"agent_{i}": [] for i in range(n_env_agents)}
        self._prev_speeds = [float(v.speed) for v in self.controlled_vehicles]
        self._last_time_for_hist = 0.0

    def step(self, action):
        t0 = float(getattr(self, "time", 0.0))
        obs, reward, terminated, truncated, info = super().step(action)
        t1 = float(getattr(self, "time", t0))
        dt = max(1e-6, t1 - t0)

        self.time_history.append(t1)
        for i, v in enumerate(self.controlled_vehicles):
            s = float(getattr(v, "speed", 0.0))
            a = (s - self._prev_speeds[i]) / dt
            self.speed_history[f"agent_{i}"].append(s)
            self.accel_history[f"agent_{i}"].append(a)
            self._prev_speeds[i] = s
        self._last_time_for_hist = t1

        # —— 角色 one-hot 注入到 info（默认启用，安全不改观测）——
        mode = str(self.config.get("role_onehot_mode", "info")).lower()
        if mode in ("info", "concat"):
            A = len(self.controlled_vehicles)
            roles = np.eye(A, dtype=np.float32)
            if not isinstance(info, dict):
                info = {}
            for i in range(A):
                key = f"agent_{i}"
                if key not in info or not isinstance(info.get(key), dict):
                    info[key] = {}
                info[key]["role_onehot"] = roles[i]

        return obs, reward, terminated, truncated, info

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vc = getattr(self.action_type, "vehicle_class", ControlledVehicle)
        base_cls = vc.func if isinstance(vc, functools.partial) else vc

        self.controlled_vehicles = []
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        for others in other_per_controlled:
            try:
                ego = base_cls.create_random(
                    self.road,
                    speed=25,
                    lane_id=self.config.get("initial_lane_id", None),
                    spacing=self.config["ego_spacing"],
                )
            except Exception:
                tmp = Vehicle.create_random(
                    self.road,
                    speed=25,
                    lane_id=self.config.get("initial_lane_id", None),
                    spacing=self.config["ego_spacing"],
                )
                ego = vc(self.road, tmp.position, tmp.heading, tmp.speed)

            self.controlled_vehicles.append(ego)
            self.road.vehicles.append(ego)

            for _ in range(others):
                ov = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                ov.randomize_behavior()
                self.road.vehicles.append(ov)

    def _reward(self, action: Action) -> Tuple[float, ...]:
        rewards: list[float] = []
        n_agents = len(self.controlled_vehicles)

        for idx, v in enumerate(self.controlled_vehicles):
            components = {}
            if v.crashed and not self._ep_crash_given[idx]:
                components["collision"] = float(self.config["collision_reward"])
                self._ep_crash_given[idx] = True
            else:
                components["collision"] = 0.0

            scaled_speed = v.speed / max(1.0, self.config["reward_speed_range"][1])
            components["speed"] = scaled_speed * self.config["speed_reward"]

            changed = (v.lane_index != self._prev_lane_index[idx])
            if changed:
                components["lane_change"] = float(self.config["lane_change_reward"])
                self._prev_lane_index[idx] = v.lane_index
            else:
                components["lane_change"] = 0.0

            final_reward = float(sum(components.values()))
            rewards.append(final_reward)

            if idx < self._ep_return.shape[0]:
                self._ep_return[idx] += final_reward
                self._ep_components["collision"][idx] += components["collision"]
                self._ep_components["speed"][idx] += components["speed"]
                self._ep_components["lane_change"][idx] += components["lane_change"]

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
        done = self.time >= self.config["duration"]
        if done:
            self._episode_done = True
        return done

    def render(self, *args, **kwargs):
        frame = super().render(*args, **kwargs)
        if getattr(self, "_record_video", False) and frame is not None:
            self._captured_frames.append(frame)
        return frame
