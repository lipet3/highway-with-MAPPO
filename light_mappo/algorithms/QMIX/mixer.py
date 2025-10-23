import torch
import torch.nn as nn
import torch.nn.functional as F

class QMixMixer(nn.Module):
    """
    QMIX mixing network：超网络根据 s 生成非负权重，保证单调性。
    输入:
      agent_qs: (B,T,A,1) 或 (B,A,1)
      state:    (B,T,S)   或 (B,S)
    输出:
      q_tot:    (B,T,1)   或 (B,1)
    """
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents, self.embed_dim = n_agents, embed_dim
        self.w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim*n_agents), nn.ReLU(),
            nn.Linear(embed_dim*n_agents, n_agents*embed_dim)
        )
        self.b1 = nn.Linear(state_dim, embed_dim)
        self.w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.b2 = nn.Linear(state_dim, 1)

    def forward(self, agent_qs, state):
        squeeze_T = False
        if agent_qs.dim() == 3:  # (B,A,1)
            squeeze_T = True
            agent_qs = agent_qs.unsqueeze(1)  # (B,1,A,1)
            state = state.unsqueeze(1)        # (B,1,S)

        B, T, A, _ = agent_qs.shape
        s = state.reshape(B*T, -1)
        w1 = torch.abs(self.w1(s)).view(B*T, A, self.embed_dim)
        b1 = self.b1(s).view(B*T, 1, self.embed_dim)
        h  = torch.bmm(agent_qs.reshape(B*T,1,A), w1) + b1        # (BT,1,E)
        h  = F.elu(h)
        w2 = torch.abs(self.w2(s)).view(B*T, self.embed_dim, 1)
        b2 = self.b2(s).view(B*T, 1, 1)
        y  = torch.bmm(h, w2) + b2                                 # (BT,1,1)
        y  = y.view(B, T, 1)
        return y[:,0] if squeeze_T else y
