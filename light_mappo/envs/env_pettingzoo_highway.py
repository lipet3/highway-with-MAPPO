# 文件名: light_mappo/envs/env_pettingzoo_highway.py

from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import numpy as np
import gymnasium as gym
import highway_env  # 确保自定义环境已注册 (如 MARLEnv)


class HighwayParallelEnv(ParallelEnv):
    """
    将底层 highway_env (如 MARLEnv) 包装成 PettingZoo Parallel API。
    不在这里配置参数，所有配置只在底层环境中修改。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "name": "MARL-v0"}

    def __init__(self, num_agents=3, render_mode=None,debug=False):  # 新增 debug 开关,用于检查典转 tuple → 再转回字典 的过程是否正确

        self.debug = debug  # 新增 debug 开关,用于检查典转 tuple → 再转回字典 的过程是否正确
        # 1. 初始化 PettingZoo 的基本属性
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        # 2. 创建底层环境（配置全部依赖底层 MARLEnv）
        self.env = gym.make("MARL-v0", render_mode=render_mode)

        # --- 如果外层被 TimeLimit 包裹，取出内部真实环境 ---
        if hasattr(self.env, "env"):
            base_env = self.env.env
        else:
            base_env = self.env

        # 3. 从底层环境推导观测和动作空间
        obs_shape = (int(np.prod(base_env.observation_space[0].shape)),)
        act_space = base_env.action_space[0]

        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {agent: act_space for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        obs_tuple, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]

        observations = {
            agent: obs_tuple[i].flatten().astype(np.float32)
            for i, agent in enumerate(self.agents)
        }
        infos = {agent: info.get(agent, {}) for agent in self.agents}
        return observations, infos


    def available_actions(self):
        """
        返回 dict[agent_id] -> (n_actions,) 的 0/1 掩码。
        最小实现：基于车道边界禁用 LANE_LEFT/RIGHT；其余动作默认可用。
        （你愿意的话，可再按速度上/下限禁用 FASTER/SLOWER）
        """
        masks = {}
        # 取某个 agent 的离散动作维度
        some_agent = self.possible_agents[0]
        n_actions = self.action_spaces[some_agent].n

        # 尝试取底层 HighwayEnv 的受控车与配置
        try:
            base_env = getattr(self, "env", None)
            lanes = int(getattr(base_env, "config", {}).get("lanes_count", 3))
            controlled = getattr(base_env, "controlled_vehicles", [])
        except Exception:
            base_env, lanes, controlled = None, 3, []

        for idx, agent in enumerate(self.possible_agents):
            m = np.ones((n_actions,), dtype=np.float32)
            try:
                veh = controlled[idx] if idx < len(controlled) else None
                lane = getattr(veh, "lane_index", (None, None, None))[2] if veh is not None else None
                if lane is not None:
                    # 约定：动作索引 0: LANE_LEFT, 2: LANE_RIGHT
                    if lane <= 0:
                        m[0] = 0.0
                    if lane >= lanes - 1:
                        m[2] = 0.0
                # （可选）基于车速上/下限屏蔽 3:FASTER / 4:SLOWER
                # vmax = base_env.config.get("reward_speed_range", [0, 30])[1] if base_env else 30
                # spd  = getattr(veh, "speed", 0.0) if veh is not None else 0.0
                # if spd >= vmax: m[3] = 0.0
                # if spd <= 0.0:  m[4] = 0.0
            except Exception:
                pass
            masks[agent] = m
        return masks


    def step(self, actions):
        action_tuple = tuple(actions[agent] for agent in self.agents)
        if self.debug:
            print(f"[PZ IN] actions to env: {action_tuple}")
        obs_tuple, rew_out, term_out, trunc_out, info = self.env.step(action_tuple)
        if self.debug:
            print(f"[PZ OUT] rewards raw: {rew_out}, terminated raw: {term_out}, truncated raw: {trunc_out}")

        # 观测：既兼容 tuple/list 也兼容单个 array
        if isinstance(obs_tuple, (list, tuple)):
            observations = {
                agent: np.asarray(obs_tuple[i]).flatten().astype(np.float32)
                for i, agent in enumerate(self.agents)
            }
        else:  # 标量/单个数组 => 对每个 agent 复制同一份（很少见，但以防万一）
            flat = np.asarray(obs_tuple).flatten().astype(np.float32)
            observations = {agent: flat.copy() for agent in self.agents}

        # 奖励：标量 => 广播；序列 => 逐 agent 取
        if isinstance(rew_out, (list, tuple, np.ndarray)):
            rewards = {agent: float(rew_out[i]) for i, agent in enumerate(self.agents)}
        else:
            rewards = {agent: float(rew_out) for agent in self.agents}

        # done/trunc：标量 => 广播；序列 => 逐 agent 取
        if isinstance(term_out, (list, tuple, np.ndarray)):
            terminations = {agent: bool(term_out[i]) for i, agent in enumerate(self.agents)}
        else:
            terminations = {agent: bool(term_out) for agent in self.agents}

        if isinstance(trunc_out, (list, tuple, np.ndarray)):
            truncations = {agent: bool(trunc_out[i]) for i, agent in enumerate(self.agents)}
        else:
            truncations = {agent: bool(trunc_out) for agent in self.agents}

        # infos：尽量拿底层的 per-agent info；没有就给空 dict
        infos = {agent: (info.get(agent, {}) if isinstance(info, dict) else {}) for agent in self.agents}

        # 任一 agent 结束 => 清空 agents（PettingZoo 规范）
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

