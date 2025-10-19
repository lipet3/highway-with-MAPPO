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
            "vehicles_count": 1,                 # 仅 1 辆 HDV
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
            "scenario_case": 1,          # 1=A0介于 A1/A2；2=A0最前；3=A0最后
            "fast_lane_id": 0,           # 快车道
            "slow_lane_id": 1,           # 慢车道
            "agent1_x": 60.0,
            "agent2_x": 140.0,
            "hdv_offset": 35.0,
            "a0_mid_margin": 10.0,
            "a0_pad": 40.0,
            "a0_speed": 25.0,
            "a1_speed": 25.0,
            "a2_speed": 25.0,
            "hdv_speed": 23.0,

            # === 奖励权重与阈值（本次只实现 6 个分量）===
            "w_speed": 3.0,                 # 速度奖励权重
            "w_low": 1.0,                   # 低速惩罚权重
            "v_min": 25.0,                  # 速度奖励下阈
            "v_max": 30.0,                  # 速度奖励上阈
            "v_low": 20.0,                  # 低速惩罚下阈

            "lane_bonus_first": 50.0,       # A0 首次变道奖励
            "lane_penalty_again": -50.0,    # A0 后续变道惩罚

            "pass_bonus": 10.0,             # 单车首次超车奖励
            "team_success_bonus": 200.0,    # 团队成功总奖（人均分配）
        })
        return config

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

        per_col  = ", ".join([f"{i}:{self._ep_components['collision'][i]:.2f}"  for i in range(n)])
        per_spd  = ", ".join([f"{i}:{self._ep_components['speed'][i]:.2f}"      for i in range(n)])
        per_low  = ", ".join([f"{i}:{self._ep_components['low_speed'][i]:.2f}"  for i in range(n)])
        per_lane = ", ".join([f"{i}:{self._ep_components['lane_a0'][i]:.2f}"    for i in range(n)])
        per_pass = ", ".join([f"{i}:{self._ep_components['pass'][i]:.2f}"       for i in range(n)])
        per_team = ", ".join([f"{i}:{self._ep_components['team'][i]:.2f}"       for i in range(n)])

        print(f"[EP SUM] ep={ep_no} steps={self._ep_step_idx} | per-agent return [{per_total}] | team_mean={team_mean:.2f}")
        print(f"[EP CMP] collision  [{per_col}]")
        print(f"[EP CMP] speed      [{per_spd}]")
        print(f"[EP CMP] low_speed  [{per_low}]")
        print(f"[EP CMP] lane_a0    [{per_lane}]")
        print(f"[EP CMP] pass       [{per_pass}]")
        print(f"[EP CMP] team       [{per_team}]")

        # —— 跨回合曲线数据 ——
        if self._reward_hist_agents is None or len(self._reward_hist_agents) != n:
            self._reward_hist_agents = [[] for _ in range(n)]
        for i in range(n):
            self._reward_hist_agents[i].append(float(self._ep_return[i]))
        self._reward_hist_team.append(team_mean)

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
            "collision":  np.zeros(n_cfg_agents, dtype=float),
            "speed":      np.zeros(n_cfg_agents, dtype=float),
            "low_speed":  np.zeros(n_cfg_agents, dtype=float),
            "lane_a0":    np.zeros(n_cfg_agents, dtype=float),
            "pass":       np.zeros(n_cfg_agents, dtype=float),
            "team":       np.zeros(n_cfg_agents, dtype=float),
        }
        self._ep_crash_given = np.zeros(n_cfg_agents, dtype=bool)
        self._ep_initialized = True

        self._create_road()
        self._create_vehicles()  # 内部会设置 self._hdv

        self._prev_lane_index = [v.lane_index for v in self.controlled_vehicles]

        # 历史
        self.time = 0.0
        self.time_history = []
        n_env_agents = len(self.controlled_vehicles)
        self.speed_history = {f"agent_{i}": [] for i in range(n_env_agents)}
        self.accel_history = {f"agent_{i}": [] for i in range(n_env_agents)}
        self._prev_speeds = [float(v.speed) for v in self.controlled_vehicles]
        self._last_time_for_hist = 0.0

        # 跨回合曲线
        if not hasattr(self, "_reward_hist_team"):
            self._reward_hist_team = []
        if not hasattr(self, "_reward_hist_agents"):
            self._reward_hist_agents = None

        # 计数与标记
        self._lane_change_count = np.zeros(n_cfg_agents, dtype=int)

        self._team_success_given = False
        # 起步相对 HDV 的前后标记（用于“穿越事件”判定）
        self._started_behind_hdv = np.zeros(n_cfg_agents, dtype=bool)

        # --- START: 关键修复 ---
        self._passed_hdv = np.zeros(n_cfg_agents, dtype=bool)  # 先创建
        if hasattr(self, "_hdv") and self._hdv is not None:
            s0_hdv = self._s_on_hdv_lane(self._hdv)
            for i, v in enumerate(self.controlled_vehicles):
                s_i = self._s_on_hdv_lane(v)
                is_behind = (s_i <= s0_hdv)
                self._started_behind_hdv[i] = is_behind
                # 如果智能体启动时就不在后面，则直接视为已完成超车子任务
                if not is_behind:
                    self._passed_hdv[i] = True

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

        # 角色 one-hot -> info
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
        """
        定制生成：
          - agent_1 与 agent_2 在快车道 lane_0，s=60/140
          - agent_0 在慢车道 lane_1，位置按场景规则采样
          - HDV 在慢车道 lane_1，纵向为 agent_0 的 s + hdv_offset
        """
        if not self.config.get("custom_spawn", True):
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
            return

        # ===== 定制生成 =====
        vc = getattr(self.action_type, "vehicle_class", ControlledVehicle)
        base_cls = vc.func if isinstance(vc, functools.partial) else vc
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        fast_id = int(self.config.get("fast_lane_id", 0))
        slow_id = int(self.config.get("slow_lane_id", 1))

        # 读取直道网络的节点与车道列表
        graph = self.road.network.graph
        u = next(iter(graph))          # 起点键（常见为 0）
        v = next(iter(graph[u]))       # 终点键（常见为 1）
        lanes_uv = graph[u][v]         # 车道对象列表
        n_lanes = len(lanes_uv)

        assert 0 <= fast_id < n_lanes, f"fast_id={fast_id} 越界，现有 {n_lanes} 条车道"
        assert 0 <= slow_id < n_lanes, f"slow_id={slow_id} 越界，现有 {n_lanes} 条车道"

        lane_fast = (u, v, fast_id)
        lane_slow = (u, v, slow_id)

        a1_x = float(self.config.get("agent1_x", 60.0))
        a2_x = float(self.config.get("agent2_x", 140.0))
        a0_mid_margin = float(self.config.get("a0_mid_margin", 10.0))
        a0_pad = float(self.config.get("a0_pad", 40.0))
        hdv_offset = float(self.config.get("hdv_offset", 35.0))

        a0_v = float(self.config.get("a0_speed", 25.0))
        a1_v = float(self.config.get("a1_speed", 25.0))
        a2_v = float(self.config.get("a2_speed", 25.0))
        hdv_v = float(self.config.get("hdv_speed", 23.0))

        case = int(self.config.get("scenario_case", 1))
        rng = self.np_random

        # —— 计算 agent_0 的 s —— #
        if case == 1:
            low, high = a1_x + a0_mid_margin, a2_x - a0_mid_margin
            low, high = sorted((low, high))
        elif case == 2:
            low, high = a2_x + 10.0, a2_x + a0_pad
        elif case == 3:
            low, high = a1_x - a0_pad, a1_x - 10.0
            low, high = sorted((low, high))
        else:
            low, high = a1_x - a0_pad, a2_x + a0_pad
        a0_x = float(rng.uniform(low, high))

        # —— 生成工具 —— #
        def _spawn_ctrl(lane_idx, s, speed):
            lane = self.road.network.get_lane(lane_idx)
            pos = lane.position(s, 0.0)
            heading = lane.heading_at(s)
            car = base_cls(self.road, pos, heading, speed)
            car.lane_index = lane_idx
            car.lane = lane
            return car

        def _spawn_other(vehicle_cls, lane_idx, s, speed):
            lane = self.road.network.get_lane(lane_idx)
            pos = lane.position(s, 0.0)
            heading = lane.heading_at(s)
            try:
                car = vehicle_cls(self.road, pos, heading, speed)
            except Exception:
                car = Vehicle(self.road, pos, heading, speed)
            car.lane_index = lane_idx
            car.lane = lane
            return car

        self.controlled_vehicles = []

        # 索引固定：0→1→2
        a0 = _spawn_ctrl(lane_slow, a0_x, a0_v)  # agent_0
        self.controlled_vehicles.append(a0); self.road.vehicles.append(a0)

        a1 = _spawn_ctrl(lane_fast, a1_x, a1_v)  # agent_1
        self.controlled_vehicles.append(a1); self.road.vehicles.append(a1)

        a2 = _spawn_ctrl(lane_fast, a2_x, a2_v)  # agent_2
        self.controlled_vehicles.append(a2); self.road.vehicles.append(a2)

        # HDV：慢车道，位于 A0 前方固定偏移
        hdv_s = a0_x + hdv_offset
        hdv = _spawn_other(other_vehicles_type, lane_slow, hdv_s, hdv_v)
        try:
            hdv.randomize_behavior()
        except Exception:
            pass
        self.road.vehicles.append(hdv)
        self._hdv = hdv  # 保存引用

    # —— 统一在 HDV 车道坐标系计算纵向 s —— #
    def _s_on_hdv_lane(self, veh) -> float:
        """将任意车辆投影到 HDV 车道坐标系，返回统一的纵向 s。"""
        hdv_lane = self._hdv.lane or self.road.network.get_lane(self._hdv.lane_index)
        return float(hdv_lane.local_coordinates(veh.position)[0])

    # ===== 奖励：6 个分量 =====
    def _reward(self, action: Action) -> Tuple[float, ...]:
        def clamp01(x): return max(0.0, min(1.0, float(x)))

        n_agents = len(self.controlled_vehicles)
        rewards = [0.0] * n_agents

        # 速度参数
        v_min = float(self.config["v_min"])
        v_max = float(self.config["v_max"])
        v_low = float(self.config["v_low"])
        w_speed = float(self.config["w_speed"])
        w_low   = float(self.config["w_low"])

        # A0 变道参数
        lane_bonus_first = float(self.config["lane_bonus_first"])
        lane_penalty_again = float(self.config["lane_penalty_again"])

        # 超车与团队
        pass_bonus = float(self.config["pass_bonus"])
        team_bonus_total = float(self.config["team_success_bonus"])

        # HDV 统一参考 s
        hdv_s = self._s_on_hdv_lane(self._hdv) if hasattr(self, "_hdv") and self._hdv is not None else -1e9

        for idx, v in enumerate(self.controlled_vehicles):
            comp = {
                "collision": 0.0,
                "speed":     0.0,
                "low_speed": 0.0,
                "lane_a0":   0.0,
                "pass":      0.0,
                "team":      0.0,
            }

            # 1) 碰撞惩罚（只扣一次）
            if v.crashed and not self._ep_crash_given[idx]:
                comp["collision"] = float(self.config["collision_reward"])
                self._ep_crash_given[idx] = True

            # 2) 速度奖励：vi∈[25,30] → [0,1]
            vi = float(v.speed)
            if vi >= v_min:
                comp["speed"] = w_speed * clamp01((vi - v_min) / max(1e-6, (v_max - v_min)))

            # 3) 低速惩罚：vi∈[20,25] → [-1,0]
            if vi < v_min:
                comp["low_speed"] = - w_low * clamp01((v_min - vi) / max(1e-6, (v_min - v_low)))

            # 4) A0 单次变道奖/罚（仅 idx==0）
            changed = (v.lane_index != self._prev_lane_index[idx])
            if changed:
                self._lane_change_count[idx] += 1
                self._prev_lane_index[idx] = v.lane_index
                if idx == 0:
                    comp["lane_a0"] = lane_bonus_first if self._lane_change_count[idx] == 1 else lane_penalty_again

            # 5) 超车奖励：仅“从后到前”穿越一次（统一在 HDV 车道坐标判定）
            si = self._s_on_hdv_lane(v)
            if (self._started_behind_hdv[idx]  # 起步在后
                and (not self._passed_hdv[idx])  # 尚未给过奖
                and (si > hdv_s)):               # 现在超过
                comp["pass"] = pass_bonus
                self._passed_hdv[idx] = True

            # 汇总
            total = sum(comp.values())
            rewards[idx] += total
            self._ep_return[idx] += total
            for k in comp:
                self._ep_components[k][idx] += comp[k]

        # 6) 团队成功奖：三车都已超车，等分一次
        if (not self._team_success_given) and bool(np.all(self._passed_hdv[:n_agents])):
            per = team_bonus_total / float(n_agents)
            for i in range(n_agents):
                rewards[i] += per
                self._ep_return[i] += per
                self._ep_components["team"][i] += per
            self._team_success_given = True

        self._ep_step_idx += 1
        return tuple(rewards)

    # 旧接口（调试可用）
    def _rewards(self, action: Action) -> Dict[str, Tuple[float, ...]]:
        collisions, speeds, lane_changes = [], [], []
        for i, v in enumerate(self.controlled_vehicles):
            collisions.append(float(v.crashed))
            speeds.append(v.speed / max(1.0, self.config["reward_speed_range"][1]))
            lane_changes.append(1.0 if v.lane_index != self._prev_lane_index[i] else 0.0)
        return {"collision": tuple(collisions), "speed": tuple(speeds), "lane_change": tuple(lane_changes)}

    # —— 奖励曲线：原始与 EMA 分图 ——
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
            if alpha is None or alpha <= 0 or alpha >= 1:
                return None
            y, m = [], 0.0
            for k, v in enumerate(x):
                m = v if k == 0 else (alpha * v + (1 - alpha) * m)
                y.append(m)
            return y

        # 原始曲线
        plt.figure(figsize=(10, 5))
        for i, hist in enumerate(agents):
            if len(hist) == 0:
                continue
            plt.plot(xs, hist, label=f"agent_{i}")
        plt.plot(xs, team, label="team_mean", linewidth=2)
        plt.xlabel("Episode"); plt.ylabel("Return")
        plt.title("Per-agent returns and team mean")
        plt.legend(ncol=2, fontsize=8); plt.grid(True, alpha=0.3)
        raw_path = os.path.join(out_dir, raw_name)
        plt.savefig(raw_path, dpi=150, bbox_inches="tight"); plt.close()

        # EMA 曲线
        plt.figure(figsize=(10, 5))
        for i, hist in enumerate(agents):
            sm = _ema(hist, ema_alpha)
            if sm is not None:
                plt.plot(xs, sm, linestyle="--", label=f"agent_{i} EMA")
        sm_team = _ema(team, ema_alpha)
        if sm_team is not None:
            plt.plot(xs, sm_team, linestyle="-", linewidth=2, label="team_mean EMA")
        plt.xlabel("Episode"); plt.ylabel("Return (EMA)")
        plt.title("EMA of per-agent returns and team mean")
        plt.legend(ncol=2, fontsize=8); plt.grid(True, alpha=0.3)
        ema_path = os.path.join(out_dir, ema_name)
        plt.savefig(ema_path, dpi=150, bbox_inches="tight"); plt.close()

        # CSV
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
        # 确保团队成功奖只被判断一次，避免与 _reward 冲突
        if getattr(self, "_team_success_given", False):
            return True  # 如果已经成功，就保持成功状态
        if not hasattr(self, "_hdv") or self._hdv is None:
            return False
        all_have_passed = bool(np.all(self._passed_hdv))
        if all_have_passed:
            return True
        return False

    def _is_terminated(self) -> bool:
        # 失败条件
        any_crash = any(v.crashed for v in self.controlled_vehicles)
        any_offroad = self.config["offroad_terminal"] and any(not v.on_road for v in self.controlled_vehicles)

        # --- START: 添加成功终止逻辑 ---
        # 假设你已经有了一个 _success() 方法来判断最终胜利条件
        is_successful = self._success()
        # --- END: 添加成功终止逻辑 ---

        done = any_crash or any_offroad or is_successful
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
