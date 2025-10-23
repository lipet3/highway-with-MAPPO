import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

def hard_update(target, source):
    target.load_state_dict(source.state_dict())

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

class QMixLearner:
    def __init__(self, agent_q, mixer, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_q     = agent_q.to(self.device)
        self.target_agent_q = deepcopy(agent_q).to(self.device).eval()
        self.mixer       = mixer.to(self.device)
        self.target_mixer= deepcopy(mixer).to(self.device).eval()

        self.gamma       = getattr(cfg, "gamma", 0.99)
        self.lr          = getattr(cfg, "lr", 3e-4)
        self.double_q    = getattr(cfg, "double_q", True)
        self.tau         = getattr(cfg, "target_tau", 0.005)
        self.update_intv = getattr(cfg, "target_update_interval", 200)

        self.n_agents    = cfg.num_agents
        self.n_actions   = cfg.n_actions

        self.optim = torch.optim.Adam(list(self.agent_q.parameters()) + list(self.mixer.parameters()), lr=self.lr)
        self.train_step = 0
        hard_update(self.target_agent_q, self.agent_q)
        hard_update(self.target_mixer, self.mixer)

    def _tensor(self, x, dtype=None):
        t = torch.as_tensor(x, device=self.device)
        return t.to(dtype) if dtype is not None else t

    def update(self, batch_list):
        """
        batch_list: List[EpisodeBatch]
        统一 pad 到相同 T（取各自 ep.t）
        """
        B = len(batch_list)
        T = max(ep.t for ep in batch_list)
        A, NA = self.n_agents, self.n_actions

        # 准备张量 [B,T,...]
        o   = torch.zeros((B, T+1, A, batch_list[0].obs_dim),  device=self.device)
        s   = torch.zeros((B, T+1, batch_list[0].state_dim),   device=self.device)
        a   = torch.zeros((B, T,   A, 1),      dtype=torch.long, device=self.device)
        r   = torch.zeros((B, T,   1),         device=self.device)
        avl = torch.ones ((B, T+1, A, NA),     device=self.device)
        d   = torch.zeros((B, T,   1),         device=self.device)

        for b, ep in enumerate(batch_list):
            t = ep.t
            o[b,:t+1]   = self._tensor(ep.o[:t+1])
            s[b,:t+1]   = self._tensor(ep.s[:t+1])
            a[b,:t]     = self._tensor(ep.a[:t])
            r[b,:t]     = self._tensor(ep.r[:t])
            avl[b,:t+1] = self._tensor(ep.avl[:t+1])
            d[b,:t]     = self._tensor(ep.done[:t])

        # 计算 Q_i(o_t, a_t)
        # 先得到全部动作的Q，再 gather 被选动作
        q_all, _ = self.agent_q(o[:,:T])                     # (B,T,A,NA)
        a_idx    = a.squeeze(-1)                             # (B,T,A)
        agent_q  = torch.gather(q_all, -1, a_idx.unsqueeze(-1))  # (B,T,A,1)

        # 计算 mixer 得到 Q_tot
        q_tot = self.mixer(agent_q, s[:,:T])                 # (B,T,1)

        # 目标值：r + gamma * (1-d) * max_{a'} Q_tot'(t+1)
        with torch.no_grad():
            # 下个时刻各动作的 Q
            q_next_all, _ = self.agent_q(o[:,1:])            # (B,T,A,NA)
            q_next_all_t, _= self.target_agent_q(o[:,1:])    # (B,T,A,NA)

            # 掩码非法动作
            mask = (avl[:,1:] > 0.5).float()                 # (B,T,A,NA)
            q_next_all  = q_next_all  * mask + (1 - mask) * (-1e9)
            q_next_all_t= q_next_all_t* mask + (1 - mask) * (-1e9)

            if getattr(self, "double_q", True):
                a_next = q_next_all.argmax(-1, keepdim=True)        # (B,T,A,1)
                agent_q_tp1 = torch.gather(q_next_all_t, -1, a_next)  # 目标网络评估
            else:
                agent_q_tp1, _ = torch.max(q_next_all_t, dim=-1, keepdim=True)

            q_tot_tp1 = self.target_mixer(agent_q_tp1, s[:,1:])  # (B,T,1)
            y = r + (1 - d) * self.gamma * q_tot_tp1

        loss = F.mse_loss(q_tot, y)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.agent_q.parameters()) + list(self.mixer.parameters()),
                                 max_norm=getattr(self, "grad_clip", 10.0))
        self.optim.step()

        self.train_step += 1
        if self.train_step % self.update_intv == 0:
            soft_update(self.target_agent_q, self.agent_q, self.tau)
            soft_update(self.target_mixer,   self.mixer,   self.tau)

        return {"loss": float(loss.item())}
