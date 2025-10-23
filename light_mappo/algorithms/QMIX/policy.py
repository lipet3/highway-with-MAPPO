import numpy as np
import torch
from .agent_net import AgentQNet
from .mixer import QMixMixer
from .buffer import EpisodeBatch
from .learner import QMixLearner

class QMIXPolicy:
    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A          = cfg.num_agents
        self.obs_dim    = cfg.obs_shape
        self.state_dim  = cfg.state_shape
        self.n_actions  = cfg.n_actions
        self.use_gru    = getattr(cfg, "rnn", True)
        self.hidden_dim = getattr(cfg, "hidden_dim", 128)
        self.embed_dim  = getattr(cfg, "embed_dim", 32)

        self.agent_q = AgentQNet(self.obs_dim, self.n_actions, self.hidden_dim, self.use_gru)
        self.mixer   = QMixMixer(self.A, self.state_dim, self.embed_dim)
        self.learner = QMixLearner(self.agent_q, self.mixer, cfg)

        self.eps_start = getattr(cfg, "eps_start", 1.0)
        self.eps_end   = getattr(cfg, "eps_end", 0.05)
        self.eps_decay = getattr(cfg, "eps_decay", 20000)
        self.step_cnt  = 0

    def epsilon(self):
        # 线性/指数都可；这里用指数衰减
        t = self.step_cnt
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-t / max(1, self.eps_decay))

    def q_values(self, obs_mat):
        """
        obs_mat: (A, obs_dim) 或 (1,A,obs_dim)
        返回: (A, n_actions) 的 Q
        """
        x = torch.as_tensor(obs_mat, dtype=torch.float32, device=self.device)
        if x.dim() == 2: x = x.unsqueeze(0)              # (1,A,D)
        q, _ = self.agent_q(x)                           # (1,A,NA)
        return q.squeeze(0).detach().cpu().numpy()

    def select_actions(self, obs_mat, avail_mat=None, greedy=False):
        """
        epsilon-greedy 选动作；支持可行动作掩码。
        返回：index 动作 (A,)
        """
        self.step_cnt += 1
        eps = 0.0 if greedy else self.epsilon()
        q = self.q_values(obs_mat)                       # (A,NA)
        if avail_mat is not None:
            q = np.where(avail_mat > 0.5, q, -1e9)
        acts = np.argmax(q, axis=-1)
        if not greedy:
            rand_mask = (np.random.rand(self.A) < eps)
            if avail_mat is None:
                rand_a = np.random.randint(0, q.shape[-1], size=(self.A,))
            else:
                rand_a = []
                for i in range(self.A):
                    legal = np.nonzero(avail_mat[i] > 0.5)[0]
                    if len(legal) == 0: legal = np.arange(q.shape[-1])
                    rand_a.append(np.random.choice(legal))
                rand_a = np.array(rand_a, dtype=np.int64)
            acts = np.where(rand_mask, rand_a, acts)
        return acts

    # 供 Runner 方便构造新 episode
    def new_episode(self, max_steps, A, obs_dim, state_dim, n_actions):
        return EpisodeBatch(max_steps, A, obs_dim, state_dim, n_actions)
