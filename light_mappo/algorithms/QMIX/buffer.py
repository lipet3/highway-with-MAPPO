import numpy as np
import random

class EpisodeBatch:
    def __init__(self, max_steps, n_agents, obs_dim, state_dim, n_actions):
        self.max_steps = max_steps
        self.n_agents  = n_agents
        self.obs_dim   = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.o   = np.zeros((max_steps+1, n_agents, obs_dim), np.float32)
        self.s   = np.zeros((max_steps+1, state_dim),         np.float32)
        self.a   = np.zeros((max_steps,   n_agents, 1),       np.int64)
        self.r   = np.zeros((max_steps,   1),                 np.float32)    # 需要时可换成 per-agent
        self.avl = np.ones ((max_steps+1, n_agents, n_actions), np.float32)
        self.done= np.zeros((max_steps,   1),                 np.float32)
        self.t   = 0  # 实际步数（<= max_steps）

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = []
        self.ptr  = 0

    def add(self, ep: EpisodeBatch):
        if len(self.data) < self.capacity: self.data.append(ep)
        else:
            self.data[self.ptr] = ep
            self.ptr = (self.ptr + 1) % self.capacity

    def __len__(self): return len(self.data)

    def sample(self, batch_size):
        idxs = [random.randrange(0, len(self.data)) for _ in range(batch_size)]
        return [self.data[i] for i in idxs]
