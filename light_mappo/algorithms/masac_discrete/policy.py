import numpy as np
import torch
from .learner import DMASACLearner

class DMASACPolicy:
    def __init__(self, cfg):
        self.A = cfg.num_agents
        self.obs_dim    = cfg.obs_shape
        self.state_dim  = cfg.state_shape
        self.n_actions  = cfg.n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learner = DMASACLearner(self.obs_dim, self.state_dim, self.n_actions, self.A, cfg)

    def select_actions(self, obs, avail=None, greedy=False):
        """
        obs:   (A, Do) numpy
        avail: (A, NA) numpy  —— 1 表示可用，0 表示禁用；允许 None
        return: a_idx (A,) numpy 的离散动作索引
        """
        A = obs.shape[0]
        device = self.learner.device

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

        if avail is None:
            avail_np = np.ones((A, self.learner.NA), dtype=np.float32)
        else:
            avail_np = np.asarray(avail, dtype=np.float32)

        avail_t = torch.as_tensor(avail_np, dtype=torch.float32, device=device)

        # 用 actor.dist 得到 mask 后的概率（内部已 masked_softmax），但我们仍然再“掩一次”兜底
        probs, _ = self.learner.actor.dist(obs_t, avail=avail_t)  # (A, NA)
        # —— 最后一轮掩码 + 归一化（防外部传入异常）——
        probs = probs * (avail_t > 0.5)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if greedy:
            a_idx_t = torch.argmax(probs, dim=-1)
        else:
            m = torch.distributions.Categorical(probs=probs)
            a_idx_t = m.sample()

        a_idx = a_idx_t.detach().cpu().numpy()  # (A,)

        # ===== 最后一闸：严格检查/修正非法动作 =====
        pick_ok = avail_np[np.arange(A), a_idx] > 0.5
        if not np.all(pick_ok):
            bad_ids = np.where(~pick_ok)[0].tolist()
            print(f"[ACTCHK] illegal picks at agents={bad_ids}, will fix to best valid.")
            probs_np = probs.detach().cpu().numpy()
            for i in bad_ids:
                valid = avail_np[i] > 0.5
                if valid.any():
                    # 在合法集合里选概率最大的动作
                    best = np.argmax((probs_np[i] + 1e-12) * valid.astype(probs_np[i].dtype))
                    a_idx[i] = int(best)
                else:
                    # 极端 fallback：如果这一行全 0，就选 0（基本不会发生）
                    a_idx[i] = 0

        return a_idx