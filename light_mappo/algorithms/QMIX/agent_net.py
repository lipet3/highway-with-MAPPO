import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentQNet(nn.Module):
    """
    个体Q网络：支持 MLP 或 GRU。
    输入: obs_dim；输出: n_actions
    """
    def __init__(self, obs_dim, n_actions, hidden_dim=128, use_gru=True):
        super().__init__()
        self.use_gru = use_gru
        if use_gru:
            self.fc1 = nn.Linear(obs_dim, hidden_dim)
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.fc2 = nn.Linear(hidden_dim, n_actions)
        else:
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            )

    def forward(self, obs, h=None):
        """
        obs: (B,T,A,obs_dim) 或 (B,A,obs_dim) 或 (N,obs_dim)
        返回: q (同批量维度, n_actions), 以及 h_n（若用GRU）
        """
        if not self.use_gru:
            x = obs
            for _ in range(3):  # 将输入 reshape 成 (N, obs_dim)
                if x.dim() > 2: x = x.reshape(-1, x.shape[-1])
            return self.net(x), None

        # GRU 分支：把时间维展开为序列
        if obs.dim() == 2:      # (N, obs_dim)
            x = F.relu(self.fc1(obs)).unsqueeze(1)  # (N,1,H)
            y, h_n = self.gru(x, h)                 # (N,1,H)
            q = self.fc2(y.squeeze(1))              # (N,n_actions)
            return q, h_n

        if obs.dim() == 3:      # (B,A,obs_dim) → 视为 (B,1, A*obs_dim)? 不，逐agent算Q
            B, A, D = obs.shape
            x = F.relu(self.fc1(obs))               # (B,A,H)
            x = x.reshape(B*A, 1, -1)               # (B*A,1,H)
            y, h_n = self.gru(x, None)              # (B*A,1,H)
            q = self.fc2(y.squeeze(1)).reshape(B, A, -1)  # (B,A,n_actions)
            return q, h_n

        # (B,T,A,obs_dim)
        B, T, A, D = obs.shape
        x = F.relu(self.fc1(obs))                   # (B,T,A,H)
        x = x.reshape(B*T*A, 1, -1)
        y, h_n = self.gru(x, None)                  # (B*T*A,1,H)
        q = self.fc2(y.squeeze(1)).reshape(B, T, A, -1)
        return q, h_n
