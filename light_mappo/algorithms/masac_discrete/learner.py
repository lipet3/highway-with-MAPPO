import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .networks import ActorCategorical, CriticDiscrete, masked_softmax

class DMASACLearner:
    """
    离散 MASAC（共享 Actor + 共享双Q，掩码 + α 自动温度）
    Critic: Q_i(s, ·) 通过 (state, agent_id_onehot) -> n_actions
    """
    def __init__(self, obs_dim, state_dim, n_actions, n_agents, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A, self.NA = n_agents, n_actions
        hid = getattr(cfg, "hidden_dim", 256)

        self.actor  = ActorCategorical(obs_dim, n_actions, hidden=hid).to(self.device)
        self.q1     = CriticDiscrete(state_dim, n_actions, n_agents, hidden=hid).to(self.device)
        self.q2     = CriticDiscrete(state_dim, n_actions, n_agents, hidden=hid).to(self.device)
        self.tq1    = deepcopy(self.q1).to(self.device).eval()
        self.tq2    = deepcopy(self.q2).to(self.device).eval()

        lr = getattr(cfg, "lr", 3e-4)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optim     = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        # === ① α 的学习率更小 & 取值范围约束 ===
        alpha_lr = getattr(cfg, "alpha_lr", 1e-4)                 # 更小的 lr
        self.log_alpha   = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self._alpha_min  = float(getattr(cfg, "alpha_min", 1e-5)) # 下界
        self._alpha_max  = float(getattr(cfg, "alpha_max", 10.0)) # 上界
        self.target_entropy = 0.98 * float(torch.log(torch.tensor(n_actions, dtype=torch.float32)))  # log|A|

        self.gamma = getattr(cfg, "gamma", 0.99)
        self.tau   = getattr(cfg, "target_tau", 0.005)

        # 预构建 agent 身份 one-hot（A, A）
        self.agent_eye = torch.eye(n_agents, dtype=torch.float32, device=self.device)

    def _apply_mask_and_norm(self, probs, avail, eps=1e-12):
        """把不可用动作的概率归零并重新归一化；全0时退回均匀分布。"""
        if avail is None:
            return probs
        probs = probs * avail
        z = probs.sum(dim=-1, keepdim=True)
        bad = (z < eps)
        if bad.any():
            # 退回均匀分布到可行动作上
            # 注意：avail 是 0/1 掩码
            safe = avail.clone()
            cnt = safe.sum(dim=-1, keepdim=True).clamp_min(1.0)
            probs = torch.where(bad, safe / cnt, probs / z.clamp_min(eps))
        else:
            probs = probs / z
        return probs

    def _debug_check_mask(self, probs, avail, tag="[MASKCHK]", tol=1e-7):
        """检查不可行动作上的概率最大值是否近似为 0；打印前几条。"""
        if avail is None:
            return
        invalid_mass = (probs * (avail < 0.5).float()).max().item()
        if invalid_mass > tol:
            print(f"{tag} invalid_prob_max={invalid_mass:.2e}")

    @property
    def alpha(self):
        # 约束 α 到安全范围
        return self.log_alpha.exp().clamp(self._alpha_min, self._alpha_max)

    def _to(self, x, dtype=None):
        t = torch.as_tensor(x, device=self.device)
        return t.to(dtype) if dtype is not None else t

    def _q_all_actions(self, qnet, state, agent_ids):
        """
        state:  (B, S)
        agent_ids: (B,A) 每个样本中的 agent 索引 [0..A-1]
        返回 Q_all: (B,A,NA)
        """
        B, A = agent_ids.shape
        s_rep = state.unsqueeze(1).repeat(1, A, 1).reshape(B*A, -1)
        id_oh = self.agent_eye[agent_ids.reshape(-1)]  # (B*A, A)
        q_all = qnet(s_rep, id_oh).reshape(B, A, -1)
        return q_all

    def update(self, batch, update_actor=True):
        import torch.nn.functional as F
        import torch.nn as nn

        s, obs, a_idx, r, s2, obs2, d, avl, avl2 = batch
        B, A, Do = obs.shape  # batch, agents, obs_dim

        # ---- to tensor on device ----
        s = self._to(s, torch.float32)
        s2 = self._to(s2, torch.float32)
        obs = self._to(obs, torch.float32)
        obs2 = self._to(obs2, torch.float32)
        a_i = self._to(a_idx, torch.long)  # (B,A)
        r = self._to(r, torch.float32)  # (B,1)
        d = self._to(d, torch.float32)  # (B,1)
        avl = self._to(avl, torch.float32)  # (B,A,NA)
        avl2 = self._to(avl2, torch.float32)  # (B,A,NA)

        # agent ids: 0..A-1，形状 (B,A)
        agent_ids = torch.arange(A, device=self.device).unsqueeze(0).repeat(B, 1)

        # -------- Critic target (s') --------
        with torch.no_grad():
            # π'(·|o', mask)
            obs2_flat = obs2.reshape(B * A, Do)
            avl2_flat = avl2.reshape(B * A, self.NA)
            probs2, _ = self.actor.dist(obs2_flat, avail=avl2_flat)  # (B*A, NA)

            # ★ 二次掩码 + 归一化 + 自检
            probs2 = self._apply_mask_and_norm(probs2, avl2_flat)
            self._debug_check_mask(probs2, avl2_flat, tag="[MASKCHK s']")

            # Q'(s',·)
            q1p = self._q_all_actions(self.tq1, s2, agent_ids)  # (B,A,NA)
            q2p = self._q_all_actions(self.tq2, s2, agent_ids)  # (B,A,NA)
            qmin_p = torch.min(q1p, q2p).reshape(B * A, self.NA)

            # V'(s') = E_{π} [Qmin - α log π]
            logp2 = torch.log(probs2.clamp_min(1e-12))
            v_each_agent = (probs2 * (qmin_p - self.alpha * logp2)).sum(dim=-1).reshape(B, A, 1)
            V_tp1 = v_each_agent.mean(dim=1)  # (B,1)

            y = r + (1.0 - d) * self.gamma * V_tp1  # (B,1)

        # -------- Critic update (s, a) --------
        q1_all = self._q_all_actions(self.q1, s, agent_ids)  # (B,A,NA)
        q2_all = self._q_all_actions(self.q2, s, agent_ids)  # (B,A,NA)

        a_onehot = F.one_hot(a_i, num_classes=self.NA).float()  # (B,A,NA)
        q1_taken = (q1_all * a_onehot).sum(dim=-1)  # (B,A)
        q2_taken = (q2_all * a_onehot).sum(dim=-1)  # (B,A)

        y_expand = y.repeat(1, A)  # (B,A)
        q_loss = F.mse_loss(q1_taken, y_expand) + F.mse_loss(q2_taken, y_expand)

        self.q_optim.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.q_optim.step()

        info = {"q_loss": float(q_loss.item()), "alpha": float(self.alpha.item())}

        # -------- Actor + Temperature α --------
        if update_actor:
            obs_flat = obs.reshape(B * A, Do)
            avl_flat = avl.reshape(B * A, self.NA)
            probs, _ = self.actor.dist(obs_flat, avail=avl_flat)  # (B*A, NA)

            # ★ 二次掩码 + 归一化 + 自检
            probs = self._apply_mask_and_norm(probs, avl_flat)
            self._debug_check_mask(probs, avl_flat, tag="[MASKCHK s]")

            logp = torch.log(probs.clamp_min(1e-12))

            q1_now = self._q_all_actions(self.q1, s, agent_ids).reshape(B * A, self.NA)
            q2_now = self._q_all_actions(self.q2, s, agent_ids).reshape(B * A, self.NA)
            qmin = torch.min(q1_now, q2_now)

            # J_pi = E[ Σ_a π(a|o) ( α logπ(a|o) - Qmin(s,a) ) ]
            actor_loss = (probs * (self.alpha * logp - qmin)).sum(dim=-1).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optim.step()
            info["actor_loss"] = float(actor_loss.item())

            # 温度：J(α) = E[ -α (H - H_target) ]，  H = -Σ π logπ
            entropy = -(probs * logp).sum(dim=-1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        # -------- target soft update --------
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.tq1.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.tq2.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return info

