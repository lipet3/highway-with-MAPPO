import numpy as np
from light_mappo.envs.env_pettingzoo_highway import HighwayParallelEnv
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

    def available_actions(self):
        """
        返回 (A, n_actions) 的 0/1 掩码矩阵（与 self.agents 顺序对齐）。
        优先转发底层 env.available_actions()；若不可用，则按“快车道禁左、慢车道禁右”
        的规则在包装器内合成。任何拿不到的信息时回退为全 1。
        """
        import numpy as np

        # ===== 1) 优先：转发底层的 available_actions() =====
        try:
            masks_dict = self.env.available_actions()  # {agent_id: (n_actions,)}
            order = getattr(self, "agents", getattr(self.env, "agents", None)) or list(masks_dict.keys())
            mats = [np.asarray(masks_dict[a], dtype=np.float32) for a in order]
            mat = np.stack(mats, axis=0).astype(np.float32)

            # 形状兜底（与包装器动作空间对齐）
            try:
                na = int(self.action_space[0].n)
                if mat.shape[1] < na:  # 右侧用 1 填充（全可用）
                    pad = np.ones((mat.shape[0], na - mat.shape[1]), dtype=np.float32)
                    mat = np.concatenate([mat, pad], axis=1)
                elif mat.shape[1] > na:
                    mat = mat[:, :na]
            except Exception:
                pass
            return mat
        except Exception:
            pass  # 转发失败则进入回退逻辑

        # ===== 2) 回退：本地基于车道规则合成掩码 =====
        core = getattr(self, "_core_env", None) or getattr(self, "env", None)
        # 拿不到必要信息就全 1 返回，确保对其它算法 0 影响
        try:
            actions = list(getattr(getattr(core, "action_type", None), "actions",
                                   ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]))
            idx = {name: i for i, name in enumerate(actions)}
            i_left = idx.get("LANE_LEFT", None)
            i_right = idx.get("LANE_RIGHT", None)

            A = len(getattr(core, "controlled_vehicles", [])) or int(getattr(self, "num_agents", 0))
            NA = len(actions)
            mask = np.ones((A, NA), dtype=np.float32)

            fast_id = int(getattr(core, "config", {}).get("fast_lane_id", 0))
            slow_id = int(getattr(core, "config", {}).get("slow_lane_id", 1))

            for i, v in enumerate(getattr(core, "controlled_vehicles", [])):
                lane = getattr(v, "lane_index", (0, 1, slow_id))
                lid = lane[2] if isinstance(lane, (list, tuple)) and len(lane) >= 3 else slow_id
                if lid == fast_id and i_left is not None:
                    mask[i, i_left] = 0.0  # 快车道禁左
                if lid == slow_id and i_right is not None:
                    mask[i, i_right] = 0.0  # 慢车道禁右

            # 与包装器动作空间对齐
            try:
                na = int(self.action_space[0].n)
                if NA < na:
                    pad = np.ones((A, na - NA), dtype=np.float32)
                    mask = np.concatenate([mask, pad], axis=1)
                elif NA > na:
                    mask = mask[:, :na]
            except Exception:
                pass

            # 若上面循环一个车都没取到，回退全 1
            if mask.size == 0 or A == 0:
                na = int(getattr(self.action_space[0], "n", NA))
                return np.ones((int(getattr(self, "num_agents", 1)), na), dtype=np.float32)

            return mask
        except Exception:
            # 最终兜底：全 1（不改变任何现有算法行为）
            try:
                A = int(getattr(self, "num_agents", 1))
                na = int(self.action_space[0].n)
            except Exception:
                A, na = 1, len(actions) if 'actions' in locals() else 5
            return np.ones((A, na), dtype=np.float32)

    def get_state(self, obs_mat):
        """
        QMIX 的全局 state。直接将 (A,D') 的 obs 拼成 (A*D',)。
        （Runner 里已自己拼了，这个方法可选）
        """
        return obs_mat.reshape(-1).astype(np.float32)



    def render(self, mode="human"):
        return self.env.render()

    def seed(self, seed: int):
        self._seed = int(seed)

    def close(self):
        self.env.close()
