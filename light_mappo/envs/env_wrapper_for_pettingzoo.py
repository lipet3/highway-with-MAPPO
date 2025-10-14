import numpy as np
from envs.env_pettingzoo_highway import HighwayParallelEnv
from gym import spaces

class PettingZooWrapper:
    def __init__(self, num_agents, render_mode=None, debug=False):
        self.env = HighwayParallelEnv(num_agents=num_agents, render_mode=render_mode)
        self.num_agents = num_agents
        self.debug = debug
        self._seed = None

        self.observation_space = [self.env.observation_space(agent) for agent in self.env.possible_agents]
        self.action_space = [self.env.action_space(agent) for agent in self.env.possible_agents]

        single_obs_dim = self.observation_space[0].shape[0]
        share_obs_dim = single_obs_dim * self.num_agents
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        if self.debug:
            print(f"[DEBUG] PettingZooWrapper initialized:")
            print(f"  - num_agents: {self.num_agents}")
            print(f"  - single_obs_dim: {single_obs_dim}")
            print(f"  - share_obs_dim: {share_obs_dim}")
            print(f"  - observation_space shape: {[s.shape for s in self.observation_space]}")
            print(f"  - action_space: {[s.n for s in self.action_space]}")

    def reset(self):
        if self._seed is not None:
            obs_dict, _ = self.env.reset(seed=int(self._seed))
            self._seed = None
        else:
            obs_dict, _ = self.env.reset()

        obs = np.stack([obs_dict[a] for a in self.env.possible_agents])
        if self.debug:
            print(f"[Wrapper RESET] obs shape: {obs.shape}, dtype: {obs.dtype}")
        return obs

    def step(self, actions):
        if self.debug:
            print(f"[Wrapper IN] actions shape: {actions.shape}, dtype: {actions.dtype}")

        if actions.ndim == 3:
            decoded_actions = np.argmax(actions, axis=2)
        elif actions.ndim == 2:
            decoded_actions = np.argmax(actions, axis=1)
        else:
            raise ValueError(f"Unexpected actions shape: {actions.shape}")

        action_dict = {agent: decoded_actions[i] for i, agent in enumerate(self.env.possible_agents)}
        obs_dict, reward_dict, term_dict, trunc_dict, _ = self.env.step(action_dict)

        obs = np.stack([obs_dict[a] for a in self.env.possible_agents])
        rewards = np.expand_dims(np.array([reward_dict[a] for a in self.env.possible_agents]), axis=1)
        dones = np.logical_or(
            np.array([term_dict[a] for a in self.env.possible_agents]),
            np.array([trunc_dict[a] for a in self.env.possible_agents]),
        )
        infos = [{} for _ in range(self.num_agents)]

        if self.debug:
            print(f"[Wrapper OUT] obs shape: {obs.shape}, rewards shape: {rewards.shape}, dones shape: {dones.shape}")

        return obs, rewards, dones, infos

    def render(self, mode="human"):
        # 实际像素返回由创建 env 时的 render_mode 决定；这里把帧往上抛
        return self.env.render()

    def seed(self, seed: int):
        self._seed = int(seed)

    def close(self):
        self.env.close()

