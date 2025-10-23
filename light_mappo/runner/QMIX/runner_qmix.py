import numpy as np

class RunnerQMIX:
    def __init__(self, env, policy, buffer, cfg, logger=None):
        """
        env: 你的 PettingZooWrapper
        policy: QMIXPolicy
        buffer: ReplayBuffer
        """
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.cfg = cfg
        self.logger = logger

        self.A = cfg.num_agents
        self.obs_dim    = cfg.obs_shape
        self.state_dim  = cfg.state_shape
        self.n_actions  = cfg.n_actions
        self.max_steps  = cfg.episode_length
        self.update_per_ep = getattr(cfg, "update_per_ep", 4)
        self.batch_size = getattr(cfg, "batch_size", 32)

    def _state_from_obs(self, obs_mat):  # (A,D') -> (A*D',)
        return obs_mat.reshape(-1)

    def run_episode(self, greedy=False):
        obs_mat = self.env.reset()                                  # (A,D')
        state   = self._state_from_obs(obs_mat)                     # (S,)
        try:
            avail = self.env.available_actions()                    # (A,NA)
        except Exception:
            avail = np.ones((self.A, self.n_actions), dtype=np.float32)

        ep = self.policy.new_episode(self.max_steps, self.A, self.obs_dim, self.state_dim, self.n_actions)
        t = 0
        ep_ret = 0.0

        while t < self.max_steps:
            acts_idx = self.policy.select_actions(obs_mat, avail, greedy=greedy)  # (A,)
            acts_oh  = np.eye(self.n_actions, dtype=np.float32)[acts_idx]         # (A,NA)

            next_obs, rew, done, infos = self.env.step(acts_oh)      # obs:(A,D') rew:(A,) done:(A,)

            # 写入
            ep.o[t]   = obs_mat
            ep.s[t]   = state
            ep.a[t,:,0] = acts_idx
            ep.r[t,0] = float(np.mean(rew))                          # 也可换成 sum
            ep.avl[t] = avail
            ep.done[t,0] = float(np.any(done))

            ep_ret += float(np.mean(rew))
            t += 1

            obs_mat = next_obs
            state   = self._state_from_obs(obs_mat)
            try:
                avail = self.env.available_actions()
            except Exception:
                avail = np.ones((self.A, self.n_actions), dtype=np.float32)

            if np.any(done):
                break

        ep.o[t] = obs_mat
        ep.s[t] = state
        ep.avl[t] = avail
        ep.t = t

        return ep, ep_ret

    def train(self):
        total_steps = 0
        for it in range(self.cfg.train_iters):
            ep, ep_ret = self.run_episode(greedy=False)
            self.buffer.add(ep)
            if self.logger is not None:
                self.logger.logkv("episode_return", ep_ret)

            if len(self.buffer) >= self.batch_size:
                for _ in range(self.update_per_ep):
                    batch = self.buffer.sample(self.batch_size)
                    out = self.policy.learner.update(batch)
                    if self.logger is not None:
                        self.logger.logkv("qmix_loss", out["loss"])

            if self.logger is not None and hasattr(self.logger, "dumpkvs"):
                self.logger.dumpkvs()
