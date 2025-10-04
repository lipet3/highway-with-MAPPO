

# 文件名: light_mappo/envs/env_wrapper_for_pettingzoo.py (最终精确版)

import numpy as np
from envs.env_pettingzoo_highway import HighwayParallelEnv
from gym import spaces

class PettingZooWrapper:
    def __init__(self, num_agents, render_mode=None, debug=False):    # debug是用来看输入到 wrapper 的动作 shape 是否符合预期 输出到 Runner 的 obs、rewards、dones shape 是否符合
        self.env = HighwayParallelEnv(num_agents=num_agents, render_mode=render_mode)
        self.num_agents = num_agents
        self.debug = debug  # 增加 debug 开关

        # 接口 1: observation_space
        self.observation_space = [self.env.observation_space(agent) for agent in self.env.possible_agents]

        # 接口 2: action_space
        self.action_space = [self.env.action_space(agent) for agent in self.env.possible_agents]

        # 接口 3: share_observation_space
        single_obs_dim = self.observation_space[0].shape[0]
        share_obs_dim = single_obs_dim * self.num_agents
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

        # 调试信息
        print(f"[DEBUG] PettingZooWrapper initialized:")
        print(f"  - num_agents: {self.num_agents}")
        print(f"  - single_obs_dim: {single_obs_dim}")
        print(f"  - share_obs_dim: {share_obs_dim}")
        print(f"  - observation_space shape: {[space.shape for space in self.observation_space]}")
        print(f"  - action_space: {[space.n for space in self.action_space]}")

    def reset(self):
        obs_dict, _ = self.env.reset(seed=self._seed)
        obs_list = [obs_dict[agent] for agent in self.env.possible_agents]
        obs = np.stack(obs_list)

        if self.debug:
            print(f"[Wrapper RESET] obs shape: {obs.shape}, dtype: {obs.dtype}")

        return obs

    def step(self, actions):
        # --- Debug 输入 ---
        if self.debug:
            print(f"[Wrapper IN] actions shape: {actions.shape}, dtype: {actions.dtype}")

        # 解码 one-hot actions
        if actions.ndim == 3:  # (n_rollout_threads, num_agents, action_dim)
            decoded_actions = np.argmax(actions, axis=2)
        elif actions.ndim == 2:  # (num_agents, action_dim)
            decoded_actions = np.argmax(actions, axis=1)
        else:
            raise ValueError(f"Unexpected actions shape: {actions.shape}")

        action_dict = {
            agent: decoded_actions[i]
            for i, agent in enumerate(self.env.possible_agents)
        }

        obs_dict, reward_dict, term_dict, trunc_dict, _ = self.env.step(action_dict)

        # 输出转换
        obs = np.stack([obs_dict[agent] for agent in self.env.possible_agents])
        rewards = np.array([reward_dict[agent] for agent in self.env.possible_agents])
        rewards = np.expand_dims(rewards, axis=1)
        dones = np.logical_or(
            np.array([term_dict[agent] for agent in self.env.possible_agents]),
            np.array([trunc_dict[agent] for agent in self.env.possible_agents])
        )
        infos = [{} for _ in range(self.num_agents)]

        # --- Debug 输出 ---
        if self.debug:
            print(f"[Wrapper OUT] obs shape: {obs.shape}, dtype: {obs.dtype}")
            print(f"[Wrapper OUT] rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
            print(f"[Wrapper OUT] dones shape: {dones.shape}, dtype: {dones.dtype}")

        return obs, rewards, dones, infos


# class PettingZooWrapper:
#     def __init__(self, num_agents, render_mode=None):
#         self.env = HighwayParallelEnv(num_agents=num_agents, render_mode=render_mode)
#         # self.num_agents = self.env.num_agents
#         self.num_agents = num_agents
#         # 接口 1: observation_space
#         # DummyVecEnv 期望这是一个列表，每个元素是单个 agent 的空间
#         self.observation_space = [self.env.observation_space(agent) for agent in self.env.possible_agents]
#
#         # 接口 2: action_space
#         # DummyVecEnv 期望这是一个列表
#         self.action_space = [self.env.action_space(agent) for agent in self.env.possible_agents]
#
#         # 接口 3: share_observation_space (最关键的补充)
#         single_obs_dim = self.observation_space[0].shape[0]
#         share_obs_dim = single_obs_dim * self.num_agents
#         # DummyVecEnv 期望这是一个列表
#         self.share_observation_space = [
#             spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
#             for _ in range(self.num_agents)
#         ]
#
#         # 调试信息
#         print(f"[DEBUG] PettingZooWrapper initialized:")
#         print(f"  - num_agents: {self.num_agents}")
#         print(f"  - single_obs_dim: {single_obs_dim}")
#         print(f"  - share_obs_dim: {share_obs_dim}")
#         print(f"  - observation_space shape: {[space.shape for space in self.observation_space]}")
#         print(f"  - action_space: {[space.n for space in self.action_space]}")
#
#     def reset(self):
#         """
#         DummyVecEnv 会自动将输出包装成 (1, num_agents, obs_dim) 的数组。
#         我们只需要返回 (num_agents, obs_dim) 的数组即可。
#         """
#         # obs_dict, _ = self.env.reset()
#         obs_dict, _ = self.env.reset(seed=self._seed)
#         obs_list = [obs_dict[agent] for agent in self.env.possible_agents]
#         return np.stack(obs_list)
#
#     def step(self, actions):
#         """
#         DummyVecEnv 传入的 actions 是 (num_agents, action_dim) 的 one-hot 数组。
#         """
#         # 解码 one-hot actions - 根据实际维度调整
#         if actions.ndim == 3:  # (n_rollout_threads, num_agents, action_dim)
#             decoded_actions = np.argmax(actions, axis=2)
#         elif actions.ndim == 2:  # (num_agents, action_dim)
#             decoded_actions = np.argmax(actions, axis=1)
#         else:
#             raise ValueError(f"Unexpected actions shape: {actions.shape}")
#
#         action_dict = {
#             agent: decoded_actions[i]
#             for i, agent in enumerate(self.env.possible_agents)
#         }
#
#         obs_dict, reward_dict, term_dict, trunc_dict, _ = self.env.step(action_dict)
#
#         # 转换并返回，DummyVecEnv 会自动为我们 stack 和 expand_dims
#         obs = np.stack([obs_dict[agent] for agent in self.env.possible_agents])
#         rewards = np.array([reward_dict[agent] for agent in self.env.possible_agents])
#         # light_mappo 的 rewards 需要有 channel 维度, DummyVecEnv 会处理
#         rewards = np.expand_dims(rewards, axis=1)  # 增加一个维度 -> (4, 1)
#
#         dones = np.logical_or(
#             np.array([term_dict[agent] for agent in self.env.possible_agents]),
#             np.array([trunc_dict[agent] for agent in self.env.possible_agents])
#         )
#
#         infos = [{} for _ in range(self.num_agents)]
#
#         return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        # DummyVecEnv 在 init_env 时已经调用了 seed
        # PettingZoo 在 reset 时接受 seed，我们可以在这里存一下
        self._seed = seed
        # 并且，为了让 DummyVecEnv 满意，我们 reset 一次
        # self.env.reset(seed=self._seed)

    def render(self, mode="human"):
        return self.env.render()