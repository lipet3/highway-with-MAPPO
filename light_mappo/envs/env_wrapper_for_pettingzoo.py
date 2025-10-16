import numpy as np
from envs.env_pettingzoo_highway import HighwayParallelEnv
from gym import spaces

class PettingZooWrapper:
    def __init__(self, num_agents, render_mode=None, debug=False):
        self.env = HighwayParallelEnv(num_agents=num_agents, render_mode=render_mode)
        self.num_agents = num_agents
        self.debug = bool(debug)
        self._seed = None

        # for调试打印计数
        self._dbg_steps = 0
        self._dbg_max_prints = 2

        self.observation_space = [self.env.observation_space(agent) for agent in self.env.possible_agents]
        self.action_space = [self.env.action_space(agent) for agent in self.env.possible_agents]

        single_obs_dim = self.observation_space[0].shape[0]  # D' = D + A (若已 concat)
        share_obs_dim = single_obs_dim * self.num_agents     # A * D'

        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        if self.debug:
            print(f"[DEBUG] PettingZooWrapper initialized:")
            print(f"  - num_agents: {self.num_agents}")
            print(f"  - single_obs_dim (D'): {single_obs_dim}")
            print(f"  - share_obs_dim: {share_obs_dim}")
            print(f"  - observation_space shape: {[s.shape for s in self.observation_space]}")
            print(f"  - action_space: {[s.n for s in self.action_space]}")

    # ===== 核心校验：尾部 A×A 是否为单位阵，行序是否稳定 =====
    def _check_tail(self, obs_mat: np.ndarray, when: str):
        """
        obs_mat: (A, D')，D' 包含 one-hot 尾部
        校验最后 A 列是否等于单位阵；打印前两次以便人工确认行序→身份。
        """
        A = obs_mat.shape[0]
        tail = obs_mat[:, -A:]
        ok = np.allclose(tail, np.eye(A, dtype=np.float32), atol=1e-6)

        if self.debug and self._dbg_steps < self._dbg_max_prints:
            print(f"[DBG][{when}] A={A}, D'={obs_mat.shape[1]}, agents={self.env.possible_agents}")
            for i in range(A):
                print(f"  row {i} tail -> {tail[i].tolist()}")
            self._dbg_steps += 1

        if self.debug and not ok:
            print("[DBG] tail observed:\n", tail)
            raise AssertionError("role one-hot tail mismatch: expected identity matrix at obs[:, -A:]")

    def reset(self):
        if self._seed is not None:
            obs_dict, _ = self.env.reset(seed=int(self._seed))
            self._seed = None
        else:
            obs_dict, _ = self.env.reset()

        # 以 possible_agents 的固定顺序堆叠，保证行序稳定
        obs = np.stack([np.asarray(obs_dict[a], dtype=np.float32) for a in self.env.possible_agents], axis=0)

        if self.debug:
            print(f"[Wrapper RESET] obs shape: {obs.shape}, dtype: {obs.dtype}")
        self._check_tail(obs, when="reset")

        return obs

    def step(self, actions):
        if self.debug:
            print(f"[Wrapper IN] actions shape: {actions.shape}, dtype: {actions.dtype}")

        # one-hot -> 索引
        if actions.ndim == 3:
            decoded_actions = np.argmax(actions, axis=2)
        elif actions.ndim == 2:
            decoded_actions = np.argmax(actions, axis=1)
        else:
            raise ValueError(f"Unexpected actions shape: {actions.shape}")

        action_dict = {agent: int(decoded_actions[i]) for i, agent in enumerate(self.env.possible_agents)}
        obs_dict, reward_dict, term_dict, trunc_dict, _ = self.env.step(action_dict)

        obs = np.stack([np.asarray(obs_dict[a], dtype=np.float32) for a in self.env.possible_agents], axis=0)
        rewards = np.expand_dims(np.array([reward_dict[a] for a in self.env.possible_agents], dtype=np.float32), axis=1)
        dones = np.logical_or(
            np.array([term_dict[a] for a in self.env.possible_agents], dtype=bool),
            np.array([trunc_dict[a] for a in self.env.possible_agents], dtype=bool),
        )
        infos = [{} for _ in range(self.num_agents)]

        if self.debug:
            print(f"[Wrapper OUT] obs shape: {obs.shape}, rewards shape: {rewards.shape}, dones shape: {dones.shape}")
        self._check_tail(obs, when="step")

        return obs, rewards, dones, infos

    def render(self, mode="human"):
        return self.env.render()

    def seed(self, seed: int):
        self._seed = int(seed)

    def close(self):
        self.env.close()
