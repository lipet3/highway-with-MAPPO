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
from highway_env.vehicle.behavior import IDMVehicle

Observation = np.ndarray



class HDVNoLaneChange(IDMVehicle):
    def __init__(self, *args, **kwargs):
        # 禁用变道
        super().__init__(*args, **kwargs)
        self.enable_lane_change = False

    def act(self, action=None):
        # act 内只用 IDM 做纵向跟驰，不做任何横向变道决策
        if self.crashed:
            return
        action = {}
        self.follow_road()
        # lateral: 不变道，不用 change_lane_policy
        # longitudinal: 正常 IDM 部分
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        action["steering"] = self.steering_control(self.target_lane_index)
        action["steering"] = np.clip(action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        action["acceleration"] = self.acceleration(
            ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
        )
        action["acceleration"] = np.clip(action["acceleration"], -self.ACC_MAX, self.ACC_MAX)
        from highway_env.vehicle.kinematics import Vehicle
        Vehicle.act(self, action)
# ===== 角色 one-hot 包装器（只改观测，不改动力学） =====
class RoleAugObsWrapper(ObservationType):
    def __init__(self, env, base_obs: ObservationType):
        super().__init__(env, config={})
        self.base = base_obs

    def space(self):
        sp = self.base.space()
        if isinstance(sp, spaces.Tuple) and len(sp) > 0 and isinstance(sp[0], spaces.Box):
            A = len(sp)
            per = sp[0]
            flat_dim = int(np.prod(per.shape)) if per.shape is not None else 0
            new_box = spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(flat_dim + A,), dtype=np.float32)
            return spaces.Tuple(tuple(new_box for _ in range(A)))
        return sp

    def observe(self):
        base = self.base.observe()
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
            "vehicles_count": 1,  # 仅 1 辆 HDV
            "controlled_vehicles": 3,
            "duration": 40,
            "ego_spacing": 1,
            "vehicles_density": 1,
            "collision_reward": -100.0,

            "initial_lane_id": None,
            "reward_speed_range": [0, 30],
            "normalize_reward": False,
            "offroad_terminal": True,
            "screen_width": 1600,
            "screen_height": 900,
            "scaling": 3.0,
            "centering_position": [0.5, 0.6],

            # 身份 one-hot
            "role_onehot_mode": "concat",

            # === 定制化场景参数 ===
            "custom_spawn": True,
            "scenario_case": 1,
            "fast_lane_id": 0,
            "slow_lane_id": 1,
            "agent1_x": 60.0,
            "agent2_x": 120.0,
            "hdv_offset": 35.0,
            "a0_mid_margin": 10.0,
            "a0_pad": 40.0,
            "a0_speed": 25.0,
            "a1_speed": 25.0,
            "a2_speed": 25.0,
            "hdv_speed": 23.0,

            "slow_lane_speed_range": [20.0, 22.0],
            "fast_lane_speed_range": [23.0, 30.0],

            # === 奖励权重与阈值 ===
            "w_speed": 3.0,
            "w_low": 1.0,
            "v_min": 25.0,
            "v_max": 30.0,
            "v_low": 20.0,
            "lane_bonus_first": 30.0,
            "lane_penalty_again": -50.0,
            "pass_bonus": 10.0,
            "team_success_bonus": 300.0,
            # === 协同安全奖励 (Time Gap) ===
            "w_safe_tg": 5,  # 协同安全奖励的全局缩放权重
            "tg_danger_threshold": 1.2,  # 危险时间间隙的阈值 (秒)

        })
        return config

    def define_spaces(self):
        super().define_spaces()
        mode = str(self.config.get("role_onehot_mode", "info")).lower()
        if mode == "concat":
            self.observation_type = RoleAugObsWrapper(self, self.observation_type)
            self.observation_space = self.observation_type.space()

    def _print_episode_summary(self) -> None:
        if not getattr(self, "_ep_initialized", False): return
        if not hasattr(self, "_ep_count"): self._ep_count = 0
        if not hasattr(self, "_last10_team_means"): self._last10_team_means = []

        ep_no = self._ep_count + 1
        n = self._ep_return.shape[0]
        team_mean = float(self._ep_return.mean())
        per_total = ", ".join([f"{i}:{v:.2f}" for i, v in enumerate(self._ep_return)])

        per_col = ", ".join([f"{i}:{v:.2f}" for i, v in enumerate(self._ep_components['collision'])])
        per_spd = ", ".join([f"{i}:{v:.2f}" for i, v in enumerate(self._ep_components['speed'])])
        per_low = ", ".join([f"{i}:{v:.2f}" for i, v in enumerate(self._ep_components['low_speed'])])
        per_safe = ", ".join([f"{i}:{v:.2f}" for i, v in enumerate(self._ep_components['safe_tg'])])
        per_lane = ", ".join([f"{i}:{v:.2f}" for i, v in enumerate(self._ep_components['lane_a0'])])
        per_pass = ", ".join([f"{i}:{v:.2f}" for i, v in enumerate(self._ep_components['pass'])])
        per_team = ", ".join([f"{i}:{v:.2f}" for i, v in enumerate(self._ep_components['team'])])

        print(
            f"[EP SUM] ep={ep_no} steps={self._ep_step_idx} | per-agent return [{per_total}] | team_mean={team_mean:.2f}")
        print(f"[EP CMP] collision  [{per_col}]")
        print(f"[EP CMP] speed      [{per_spd}]")
        print(f"[EP CMP] low_speed  [{per_low}]")
        print(f"[EP CMP] safe_tg    [{per_safe}]")
        print(f"[EP CMP] lane_a0    [{per_lane}]")
        print(f"[EP CMP] pass       [{per_pass}]")
        print(f"[EP CMP] team       [{per_team}]")

        if self._reward_hist_agents is None or len(self._reward_hist_agents) != n:
            self._reward_hist_agents = [[] for _ in range(n)]
        for i in range(n): self._reward_hist_agents[i].append(float(self._ep_return[i]))
        self._reward_hist_team.append(team_mean)

        self._last10_team_means.append(team_mean)
        if len(self._last10_team_means) == 10:
            start_ep, end_ep = ep_no - 9, ep_no
            block_avg = float(np.mean(self._last10_team_means))
            print(f"[EP AVG] {start_ep}-{end_ep} team_mean={block_avg:.2f}")
            self._last10_team_means.clear()

    # ===================================================================================
    # ==== 新增: 公共 reset 方法, 覆盖父类, 确保返回 (obs, info) ====
    # ===================================================================================
    def reset(self, *args, **kwargs) -> tuple[Observation, dict]:
        """
        重置环境并返回 (observation, info)，info 中包含动作掩码。
        """
        ret = super().reset(*args, **kwargs)
        if isinstance(ret, tuple) and len(ret) == 2:
            obs, info = ret
        else:
            obs, info = ret, {}

        # 注入动作掩码
        if "available_actions" not in info:
            info["available_actions"] = self._get_available_actions()
        return obs, info

    def _reset(self) -> None:
        if not hasattr(self, "_ep_count"): self._ep_count = 0
        if not hasattr(self, "_last10_team_means"): self._last10_team_means = []
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
            "low_speed": np.zeros(n_cfg_agents, dtype=float),
            "safe_tg": np.zeros(n_cfg_agents, dtype=float),
            "lane_a0": np.zeros(n_cfg_agents, dtype=float),
            "pass": np.zeros(n_cfg_agents, dtype=float),
            "team": np.zeros(n_cfg_agents, dtype=float),
        }
        self._ep_crash_given = np.zeros(n_cfg_agents, dtype=bool)
        self._ep_initialized = True

        self._create_road()
        self._create_vehicles()

        self._prev_lane_index = [v.lane_index for v in self.controlled_vehicles]
        self._a0_initial_lane_index = self.controlled_vehicles[0].lane_index
        self._a0_ever_changed = False

        self.time = 0.0
        self.time_history, self.speed_history, self.accel_history = [], {}, {}
        n_env_agents = len(self.controlled_vehicles)
        for i in range(n_env_agents):
            self.speed_history[f"agent_{i}"] = []
            self.accel_history[f"agent_{i}"] = []
        self._prev_speeds = [float(v.speed) for v in self.controlled_vehicles]
        self._last_time_for_hist = 0.0

        if not hasattr(self, "_reward_hist_team"): self._reward_hist_team = []
        if not hasattr(self, "_reward_hist_agents"): self._reward_hist_agents = None

        self._lane_change_count = np.zeros(n_cfg_agents, dtype=int)
        self._team_success_given = False
        self._started_behind_hdv = np.zeros(n_cfg_agents, dtype=bool)
        self._passed_hdv = np.zeros(n_cfg_agents, dtype=bool)
        if hasattr(self, "_hdv") and self._hdv is not None:
            s0_hdv = self._s_on_hdv_lane(self._hdv)
            for i, v in enumerate(self.controlled_vehicles):
                s_i = self._s_on_hdv_lane(v)
                is_behind = s_i <= s0_hdv
                self._started_behind_hdv[i] = is_behind
                if not is_behind: self._passed_hdv[i] = True

    def step(self, action):
        t0 = float(getattr(self, "time", 0.0))
        obs, reward, terminated, truncated, info = super().step(action)
        t1 = float(getattr(self, "time", t0))
        dt = max(1e-6, t1 - t0)

        self.time_history.append(t1)
        for i, v in enumerate(self.controlled_vehicles):
            s = float(v.speed)
            a = (s - self._prev_speeds[i]) / dt
            self.speed_history[f"agent_{i}"].append(s)
            self.accel_history[f"agent_{i}"].append(a)
            self._prev_speeds[i] = s
        self._last_time_for_hist = t1

        mode = str(self.config.get("role_onehot_mode", "info")).lower()
        if mode in ("info", "concat"):
            A = len(self.controlled_vehicles)
            roles = np.eye(A, dtype=np.float32)
            if not isinstance(info, dict): info = {}
            for i in range(A):
                key = f"agent_{i}"
                if key not in info or not isinstance(info.get(key), dict): info[key] = {}
                info[key]["role_onehot"] = roles[i]

        # =================================================================
        # ==== 修改: 注入 "available_actions" 到 info 字典中 ====
        # =================================================================
        if not isinstance(info, dict): info = {}
        if "available_actions" not in info:
            info["available_actions"] = self._get_available_actions()

        return obs, reward, terminated, truncated, info

    def _create_road(self) -> None:
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])
        try:
            slow_id = int(self.config["slow_lane_id"])
            cap = float(self.config.get("slow_lane_speed_cap", 25.0))

            graph = self.road.network.graph
            u = next(iter(graph))
            v = next(iter(graph[u]))
            lanes = graph[u][v]

            if 0 <= slow_id < len(lanes): lanes[slow_id].speed_limit = cap
        except Exception:
            pass

    def _create_vehicles(self) -> None:
        if not self.config.get("custom_spawn", True):
            return

        vc = getattr(self.action_type, "vehicle_class", ControlledVehicle)
        base_cls = vc.func if isinstance(vc, functools.partial) else vc
        # other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        fast_id, slow_id = int(self.config["fast_lane_id"]), int(self.config["slow_lane_id"])

        graph = self.road.network.graph
        u = next(iter(graph))
        v = next(iter(graph[u]))

        lane_fast, lane_slow = (u, v, fast_id), (u, v, slow_id)

        a1_x, a2_x = float(self.config["agent1_x"]), float(self.config["agent2_x"])
        a0_mid_margin, a0_pad = float(self.config["a0_mid_margin"]), float(self.config["a0_pad"])
        hdv_offset = float(self.config["hdv_offset"])

        rng = self.np_random
        slow_range, fast_range = self.config["slow_lane_speed_range"], self.config["fast_lane_speed_range"]
        a0_v, hdv_v = float(rng.uniform(*slow_range)), float(rng.uniform(*slow_range))
        a1_v, a2_v = float(rng.uniform(*fast_range)), float(rng.uniform(*fast_range))
        case = self.np_random.choice([1, 2, 3], p=[0.5, 0.25, 0.25])

        if case == 1:
            low, high = sorted((a1_x + a0_mid_margin, a2_x - a0_mid_margin))
        elif case == 2:
            low, high = a2_x + 10.0, a2_x + a0_pad
        else:
            low, high = sorted((a1_x - a0_pad, a1_x - 10.0))
        a0_x = float(rng.uniform(low, high))

        def _spawn(cls, lane_idx, s, speed, is_controlled=False):
            lane = self.road.network.get_lane(lane_idx)
            pos, heading = lane.position(s, 0.0), lane.heading_at(s)
            car = cls(self.road, pos, heading, speed)
            car.lane_index, car.lane = lane_idx, lane
            if is_controlled: self.controlled_vehicles.append(car)
            self.road.vehicles.append(car)
            return car

        self.controlled_vehicles = []
        _spawn(base_cls, lane_slow, a0_x, a0_v, True)
        _spawn(base_cls, lane_fast, a1_x, a1_v, True)
        _spawn(base_cls, lane_fast, a2_x, a2_v, True)

        hdv = _spawn(HDVNoLaneChange, lane_slow, a0_x + hdv_offset, hdv_v)
        if hasattr(hdv, 'randomize_behavior'): hdv.randomize_behavior()
        self._hdv = hdv

    def _s_on_hdv_lane(self, veh) -> float:
        hdv_lane = getattr(getattr(self, "_hdv", None), "lane", None) or self.road.network.get_lane(
            self._hdv.lane_index)
        return float(hdv_lane.local_coordinates(veh.position)[0])

    # =================================================================
    # ==== 新增: _get_available_actions 方法, 定义角色动作空间 ====
    # =================================================================
    def _get_available_actions(self) -> list[list[int]]:
        """为每个智能体生成动作掩码。"""
        avail_actions = []
        n_agents = self.config["controlled_vehicles"]
        for i in range(n_agents):
            if i == 0:
                # agent_0: 领头车，可以执行所有5个动作
                # [LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER]
                avail_actions.append([1, 1, 1, 1, 1])
            else:
                # agent_1, agent_2: 护航车，不能变道 (动作0和2被屏蔽)
                avail_actions.append([0, 1, 0, 1, 1])
        return avail_actions

    def available_actions(self) -> dict:
        """
        返回 dict 形式的动作掩码，键为 agent_i，值为 (n_actions,) 的 0/1 向量。
        供上层 PettingZooWrapper 优先转发使用。
        """
        mats = self._get_available_actions()
        out = {}
        for i, row in enumerate(mats):
            out[f"agent_{i}"] = np.asarray(row, dtype=np.float32)
        return out

    def _get_comfort_score(self, time_gap: float, is_same_lane: bool) -> float:
        if time_gap is None:
            return 0.0
        T1 = self.config["tg_danger_threshold"]
        score = 0.0
        if time_gap < T1:
            score = -min(1.0, (T1 - max(time_gap, 0.0)) / T1)
        if not is_same_lane:
            score = min(0.0, score)
        return score

    def _calculate_cooperative_safety_reward(self):
        ALPHA_SCALE = self.config["w_safe_tg"]
        EPS = 0.1
        agents = self.controlled_vehicles
        n = len(agents)
        safety_rewards = np.zeros(n, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = agents[i], agents[j]
                if vi.position[0] > vj.position[0]:
                    front, rear, f_idx, r_idx = vi, vj, i, j
                else:
                    front, rear, f_idx, r_idx = vj, vi, j, i

                if rear.lane_index == self._prev_lane_index[r_idx]:
                    continue

                rear_lane = self.road.network.get_lane(rear.lane_index)
                s_rear = float(rear_lane.local_coordinates(rear.position)[0])
                s_front = float(rear_lane.local_coordinates(front.position)[0])
                distance = s_front - s_rear
                if distance <= 0.0:
                    continue
                v_follow = max(rear.speed, EPS)
                time_gap = distance / v_follow
                is_same_lane = front.lane_index == rear.lane_index
                score = self._get_comfort_score(time_gap, is_same_lane)
                safety_rewards[r_idx] += score
                safety_rewards[f_idx] += score
        if hasattr(self, "_hdv") and self._hdv is not None and n > 0:
            a0 = agents[0]
            if not getattr(self, "_a0_ever_changed", False):
                if getattr(self, "_a0_initial_lane_index", None) is None:
                    self._a0_initial_lane_index = a0.lane_index
                if a0.lane_index == self._hdv.lane_index and a0.lane_index == self._a0_initial_lane_index:
                    lane = self.road.network.get_lane(a0.lane_index)
                    s_a0 = float(lane.local_coordinates(a0.position)[0])
                    s_h = float(lane.local_coordinates(self._hdv.position)[0])
                    if s_a0 < s_h:
                        distance, v_follow = s_h - s_a0, max(a0.speed, EPS)
                    else:
                        distance, v_follow = s_a0 - s_h, max(self._hdv.speed, EPS)
                    if distance > 0.0:
                        score = self._get_comfort_score(distance / v_follow, True)
                        safety_rewards[0] += score
        return safety_rewards * ALPHA_SCALE

    def _reward(self, action: Action) -> Tuple[float, ...]:
        n_agents = len(self.controlled_vehicles)
        rewards = [0.0] * n_agents
        safe_gains = self._calculate_cooperative_safety_reward()

        v_min, v_max, v_low = self.config["v_min"], self.config["v_max"], self.config["v_low"]
        w_speed, w_low = self.config["w_speed"], self.config["w_low"]
        lane_bonus_first, lane_penalty_again = self.config["lane_bonus_first"], self.config["lane_penalty_again"]
        pass_bonus, team_bonus_total = self.config["pass_bonus"], self.config["team_success_bonus"]

        hdv_s = self._s_on_hdv_lane(self._hdv) if hasattr(self, "_hdv") and self._hdv is not None else -1e9

        for idx, v in enumerate(self.controlled_vehicles):
            comp = {"collision": 0.0, "speed": 0.0, "low_speed": 0.0, "safe_tg": 0.0,
                    "lane_a0": 0.0, "pass": 0.0, "team": 0.0}

            if v.crashed and not self._ep_crash_given[idx]:
                comp["collision"] = self.config["collision_reward"]
                self._ep_crash_given[idx] = True

            vi = float(v.speed)
            den_spd = max(1e-6, (v_max - v_min))
            den_low = max(1e-6, (v_min - v_low))
            if vi >= v_min:
                comp["speed"] = w_speed * max(0.0, min(1.0, (vi - v_min) / den_spd))
            if vi < v_min:
                comp["low_speed"] = -w_low * max(0.0, min(1.0, (v_min - vi) / den_low))

            changed = (v.lane_index != self._prev_lane_index[idx])
            if changed:
                self._lane_change_count[idx] += 1
                self._prev_lane_index[idx] = v.lane_index
                if idx == 0:
                    comp["lane_a0"] = lane_bonus_first if self._lane_change_count[idx] == 1 else lane_penalty_again
                    self._a0_ever_changed = True

            si = self._s_on_hdv_lane(v)
            if (not self._passed_hdv[idx]) and (si > hdv_s):
                comp["pass"] = pass_bonus
                self._passed_hdv[idx] = True

            comp["safe_tg"] = float(safe_gains[idx])
            total = sum(comp.values())
            rewards[idx] += total
            self._ep_return[idx] += total
            for k in comp: self._ep_components[k][idx] += comp[k]

        if not self._team_success_given and bool(np.all(self._passed_hdv[:n_agents])):
            per = team_bonus_total / float(n_agents)
            for i in range(n_agents):
                rewards[i] += per
                self._ep_return[i] += per
                self._ep_components["team"][i] += per
            self._team_success_given = True

        self._ep_step_idx += 1
        return tuple(rewards)

    def _rewards(self, action: Action) -> Dict[str, Tuple[float, ...]]:
        collisions, speeds, lane_changes = [], [], []
        vmax = max(1.0, self.config["reward_speed_range"][1])
        for i, v in enumerate(self.controlled_vehicles):
            collisions.append(float(v.crashed))
            speeds.append(v.speed / vmax)
            lane_changes.append(1.0 if v.lane_index != self._prev_lane_index[i] else 0.0)
        return {"collision": tuple(collisions), "speed": tuple(speeds), "lane_change": tuple(lane_changes)}

    def save_reward_curves(self, out_dir="videos",
                           raw_name="reward_curves.png",
                           ema_name="reward_curves_ema.png",
                           csv_filename="reward_curves.csv",
                           ema_alpha: float | None = 0.2) -> tuple[str, str]:
        import os, csv
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)
        team = getattr(self, "_reward_hist_team", [])
        agents = getattr(self, "_reward_hist_agents", None)
        if not team or not agents:
            print("[WARN] 奖励历史为空，未绘制。")
            return "", ""

        E = len(team)
        xs = np.arange(1, E + 1)

        def _ema(x, alpha):
            if alpha is None or alpha <= 0 or alpha >= 1: return None
            y, m = [], 0.0
            for k, v in enumerate(x):
                m = v if k == 0 else (alpha * v + (1 - alpha) * m)
                y.append(m)
            return y

        plt.figure(figsize=(10, 5))
        for i, hist in enumerate(agents):
            if len(hist) > 0: plt.plot(xs, hist, label=f"agent_{i}")
        plt.plot(xs, team, label="team_mean", linewidth=2)
        plt.xlabel("Episode");
        plt.ylabel("Return");
        plt.title("Per-agent returns and team mean")
        plt.legend(ncol=2, fontsize=8);
        plt.grid(True, alpha=0.3)
        raw_path = os.path.join(out_dir, raw_name)
        plt.savefig(raw_path, dpi=150, bbox_inches="tight");
        plt.close()

        plt.figure(figsize=(10, 5))
        for i, hist in enumerate(agents):
            sm = _ema(hist, ema_alpha)
            if sm: plt.plot(xs, sm, linestyle="--", label=f"agent_{i} EMA")
        sm_team = _ema(team, ema_alpha)
        if sm_team: plt.plot(xs, sm_team, linestyle="-", linewidth=2, label="team_mean EMA")
        plt.xlabel("Episode");
        plt.ylabel("Return (EMA)");
        plt.title("EMA of per-agent returns and team mean")
        plt.legend(ncol=2, fontsize=8);
        plt.grid(True, alpha=0.3)
        ema_path = os.path.join(out_dir, ema_name)
        plt.savefig(ema_path, dpi=150, bbox_inches="tight");
        plt.close()

        if csv_filename:
            csv_path = os.path.join(out_dir, csv_filename)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                header = ["episode"] + [f"agent_{i}" for i in range(len(agents))] + ["team_mean"]
                w.writerow(header)
                for e in range(E):
                    row = [e + 1] + [agents[i][e] if e < len(agents[i]) else "" for i in range(len(agents))] + [team[e]]
                    w.writerow(row)

        print(f"[OK] 奖励曲线已保存：{raw_path} | {ema_path}")
        return raw_path, ema_path

    def _success(self) -> bool:
        if getattr(self, "_team_success_given", False): return True
        return bool(np.all(self._passed_hdv))

    def _is_terminated(self) -> bool:
        done = any(v.crashed for v in self.controlled_vehicles) or \
               (self.config["offroad_terminal"] and any(not v.on_road for v in self.controlled_vehicles)) or \
               self._success()
        if done: self._episode_done = True
        return done

    def _is_truncated(self) -> bool:
        done = self.time >= self.config["duration"]
        if done: self._episode_done = True
        return done

    def render(self, *args, **kwargs):
        frame = super().render(*args, **kwargs)
        if getattr(self, "_record_video", False) and frame is not None:
            self._captured_frames.append(frame)
        return frame
