import torch
import torch.nn as nn
import torch.nn.functional as F

def mlp(in_dim, out_dim, hidden=256, depth=2, act=nn.ReLU):
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)

def masked_softmax(logits, mask, dim=-1, eps=1e-12):
    """
    logits: (..., A)
    mask:   (..., A) 0/1
    """
    mask = mask.float()
    masked = logits + (mask + 1e-8).log()  # -inf 到 0 的稳定替代：log(1)=0, log(0)=-inf
    # 上式等价于把非法动作置为 -inf 后再 softmax
    probs = F.softmax(masked, dim=dim)
    # 归一化防止 mask 全 0
    probs = probs * mask
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    return probs / denom

class ActorCategorical(nn.Module):
    """ π(a|o) —— 离散策略，输出 logits -> softmax（带掩码）"""
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden=hidden)

    def logits(self, obs):
        return self.net(obs)

    def dist(self, obs, avail=None):
        logits = self.logits(obs)  # (N, A)
        if avail is not None:
            probs = masked_softmax(logits, avail, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
        return probs, logits

class CriticDiscrete(nn.Module):
    """
    Q(s, a_i) —— 给定全局 state 和 agent 身份 one-hot，输出该 agent 对所有动作的 Q 向量
    输入: [state_dim + A]  输出: n_actions
    """
    def __init__(self, state_dim, n_actions, n_agents, hidden=256):
        super().__init__()
        self.n_actions = n_actions
        self.net = mlp(state_dim + n_agents, n_actions, hidden=hidden)

    def forward(self, state, agent_id_onehot):
        """
        state:           (N, S)
        agent_onehot:    (N, A_agents)
        return: Q_all:   (N, n_actions)
        """
        x = torch.cat([state, agent_id_onehot], dim=-1)
        return self.net(x)
