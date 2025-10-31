# 文件: light_mappo/runner/masac_discrete/runner_masac.py
import os
import numpy as np
import torch
import matplotlib

# 如需无窗保存可改为 "Agg"
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class RunnerDMASAC:
    def __init__(self, env, policy, buffer, cfg, logger=None):
        self.env, self.pi, self.buf, self.cfg, self.logger = env, policy, buffer, cfg, logger
        self.A = cfg.num_agents
        self.Do = cfg.obs_shape
        self.S = cfg.state_shape
        self.NA = cfg.n_actions
        self.max_len = cfg.episode_length
        self.batch_size = getattr(cfg, "batch_size", 256)
        self.update_per_step = getattr(cfg, "update_per_step", 1)

        # ==== EMA 平滑系数 ====
        self.ema_alpha = float(getattr(cfg, "ema_alpha", 0.2))

        # ==== 可视化缓存 ====
        self.plot_initialized = False
        self.ep_list = []
        self.team_mean_list = []
        self.agent_returns = []  # list of np.array(A,)

        # ==== 保存设置 ====
        self.base_save_dir = r"C:\Users\z8603\Desktop\study\highway with mappo\light_mappo\results\HighwayEnv\highway\masac"
        os.makedirs(self.base_save_dir, exist_ok=True)
        self.run_dir = self._prepare_run_dir(self.base_save_dir)
        print(f"✅ 本次训练输出目录: {self.run_dir}")

    # ====== NEW: 通用工具（掩码抽取/整理） ======
    def _coerce_masks(self, masks):
        """
        把各种可能的掩码结构整理成 (A, NA) 的 float32 数组；失败返回 None。
        """
        try:
            if isinstance(masks, dict) and any(k.startswith("agent_") for k in masks.keys()):
                rows = []
                for i in range(self.A):
                    key = f"agent_{i}"
                    if key in masks and isinstance(masks[key], dict) and "available_actions" in masks[key]:
                        rows.append(np.asarray(masks[key]["available_actions"], dtype=np.float32))
                if len(rows) == self.A:
                    arr = np.stack(rows, axis=0)
                    return self._fix_mask_shape(arr)
            if isinstance(masks, dict) and "available_actions" in masks:
                arr = np.asarray(masks["available_actions"], dtype=np.float32)
                return self._fix_mask_shape(arr)
            if isinstance(masks, (list, tuple)):
                arr = np.asarray(masks, dtype=np.float32)
                return self._fix_mask_shape(arr)
            arr = np.asarray(masks, dtype=np.float32)
            return self._fix_mask_shape(arr)
        except Exception:
            return None

    def _fix_mask_shape(self, arr):
        """把掩码修成 (A, NA)。一维则广播到 A；多于 NA 的列会截断；不足会右侧补 1。"""
        if arr.ndim == 1:
            arr = np.tile(arr[None, :], (self.A, 1))
        if arr.shape[0] != self.A:
            return None
        if arr.shape[1] < self.NA:
            pad = np.ones((self.A, self.NA - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > self.NA:
            arr = arr[:, :self.NA]
        arr = (arr > 0.5).astype(np.float32)
        return arr

    def _masks_from_info(self, info):
        """尽量从 info 里抽取掩码，抽不到返回 None。"""
        if info is None: return None
        if isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
            rows = []
            for i in range(min(self.A, len(info))):
                if "available_actions" in info[i]:
                    rows.append(info[i]["available_actions"])
            if len(rows) == self.A:
                return self._coerce_masks(rows)
        if isinstance(info, dict):
            return self._coerce_masks(info)
        return None

    def _env_available_or_default(self):
        """优先调用 env.available_actions()，否则回退全 1。"""
        try:
            avail = self.env.available_actions()
            if avail is None: return np.ones((self.A, self.NA), dtype=np.float32)
            arr = self._coerce_masks(avail)
            if arr is None: return np.ones((self.A, self.NA), dtype=np.float32)
            return arr
        except Exception:
            return np.ones((self.A, self.NA), dtype=np.float32)

    # ====== NEW END ======

    def _prepare_run_dir(self, base_dir):
        i = 1
        while True:
            d = os.path.join(base_dir, f"run{i}")
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
                return d
            i += 1

    def _state(self, obs):
        return obs.reshape(-1)

    def _ema_series(self, xs, alpha):
        if xs is None or len(xs) == 0 or alpha <= 0 or alpha >= 1: return None
        out, m = [], None
        for v in xs:
            v = float(v)
            m = v if m is None else (alpha * v + (1 - alpha) * m)
            out.append(m)
        return out

    def _ema_2d(self, arr2d, alpha):
        if arr2d is None or len(arr2d) == 0 or alpha <= 0 or alpha >= 1: return None
        arr2d = np.asarray(arr2d, dtype=np.float32)
        T, A = arr2d.shape
        out = np.zeros_like(arr2d)
        for j in range(A):
            m = None
            for t in range(T):
                v = float(arr2d[t, j])
                m = v if m is None else (alpha * v + (1 - alpha) * m)
                out[t, j] = m
        return out

    def _dig_core_env(self, obj):
        seen = set()
        stack = [getattr(obj, "env", obj)]
        core = None
        while stack:
            cur = stack.pop()
            if id(cur) in seen: continue
            seen.add(id(cur))
            if hasattr(cur, "save_reward_curves"):
                core = cur
                break
            if hasattr(cur, "controlled_vehicles") and hasattr(cur, "road"):
                core = cur
            for name in ("env", "unwrapped", "par_env", "aec_env", "raw_env", "_env", "wrapped_env", "environment"):
                if hasattr(cur, name):
                    nxt = getattr(cur, name)
                    if isinstance(nxt, (list, tuple)):
                        stack.extend(list(nxt))
                    else:
                        stack.append(nxt)
        return core

    def _save_reward_curves_if_possible(self, force: bool = False, ema_alpha: float = 0.2):
        core = self._dig_core_env(self.env)
        if core is None or not hasattr(core, "save_reward_curves"):
            if force: print("[WARN] 未找到 MARLEnv 或缺少 save_reward_curves，无法保存奖励曲线。")
            return
        try:
            raw_path, ema_path = core.save_reward_curves(out_dir=self.run_dir, ema_alpha=ema_alpha)
            if force or (raw_path or ema_path): print(f"[OK] 奖励曲线已更新：{self.run_dir}")
        except Exception as e:
            if force: print(f"[WARN] 保存奖励曲线失败：{e}")

    def render_train(self, ep, team_mean, per_agent_returns):
        if not self.plot_initialized:
            plt.ion()
            self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 6))
            self.ax_team, self.ax_agents = self.axs[0], self.axs[1]
            self.plot_initialized = True

        self.ep_list.append(ep)
        self.team_mean_list.append(team_mean)
        self.agent_returns.append(per_agent_returns)
        agents_np = np.array(self.agent_returns)
        T = len(self.ep_list)

        self.ax_team.clear()
        self.ax_team.plot(self.ep_list, self.team_mean_list, label="Team Mean (raw)")
        sm_team = self._ema_series(self.team_mean_list, self.ema_alpha)
        if sm_team is not None and len(sm_team) == T:
            self.ax_team.plot(self.ep_list, sm_team, linestyle="--", label=f"Team Mean EMA (α={self.ema_alpha})")
        self.ax_team.set_title("Team Mean Return");
        self.ax_team.set_xlabel("Episode");
        self.ax_team.set_ylabel("Return")
        self.ax_team.grid(True);
        self.ax_team.legend()

        self.ax_agents.clear()
        for i in range(self.A): self.ax_agents.plot(self.ep_list, agents_np[:, i], label=f"Agent {i} (raw)")
        self.ax_agents.set_title("Per-Agent Return");
        self.ax_agents.set_xlabel("Episode");
        self.ax_agents.set_ylabel("Return")
        self.ax_agents.grid(True);
        self.ax_agents.legend(ncol=2)

        plt.tight_layout()
        plt.pause(0.001)

    def run_one_episode(self, greedy=False, ep_idx=0):
        # --- MODIFIED: 兼容 Gymnasium reset 返回 (obs, info) 并获取掩码 ---
        ret = self.env.reset()
        if isinstance(ret, tuple) and len(ret) == 2:
            obs, info0 = ret
        else:
            obs, info0 = ret, None
        s = self._state(obs)
        avail = self._masks_from_info(info0)
        if avail is None: avail = self._env_available_or_default()
        # --- MODIFICATION END ---

        per_agent_returns = np.zeros(self.A, dtype=np.float32)
        for t in range(self.max_len):
            a_idx = self.pi.select_actions(obs, avail=avail, greedy=greedy)

            try:
                mask_ok = avail[np.arange(self.A), a_idx] > 0.5
                if not np.all(mask_ok):
                    bad = np.where(~mask_ok)[0].tolist()
                    print(f"[RUNCHK] get illegal actions from policy at agents={bad}")
            except Exception:
                pass

            a_onehot = np.eye(self.NA, dtype=np.float32)[a_idx]

            # --- MODIFIED: 兼容 Gym/Gymnasium 的 step 返回格式 ---
            step_ret = self.env.step(a_onehot)
            if isinstance(step_ret, tuple) and len(step_ret) == 5:
                obs2, rew, terminated, truncated, infos = step_ret
                done = np.array(terminated) | np.array(truncated)
            else:
                obs2, rew, done, infos = step_ret
            # --- MODIFICATION END ---

            s2 = self._state(obs2)

            # --- MODIFIED: 从 step 返回中获取掩码 ---
            avail2 = self._masks_from_info(infos)
            if avail2 is None: avail2 = self._env_available_or_default()
            # --- MODIFICATION END ---

            team_r = float(np.mean(rew))
            self.buf.add(s, obs, a_idx, team_r, s2, obs2, np.any(done), avail, avail2)
            per_agent_returns += np.squeeze(rew).astype(np.float32)

            if self.buf.size >= self.batch_size:
                for _ in range(self.update_per_step):
                    batch = self.buf.sample(self.batch_size)
                    out = self.pi.learner.update(batch, update_actor=True)
                    if self.logger is not None:
                        for k, v in out.items(): self.logger.logkv(k, v)

            obs, s, avail = obs2, s2, avail2
            if np.any(done): break

        team_mean = float(np.mean(per_agent_returns))
        # This print is now redundant if MARL.py prints, but kept for standalone debugging
        # print(f"[EP SUM] ep={ep_idx} | team_mean={team_mean:.2f} | per-agent={per_agent_returns.round(2).tolist()}")
        return team_mean, per_agent_returns

    def train(self):
        for it in range(self.cfg.train_iters):
            ep_idx = it + 1
            team_mean, agent_returns = self.run_one_episode(greedy=False, ep_idx=ep_idx)
            self.render_train(ep_idx, team_mean, agent_returns)
            if ep_idx % 100 == 0:
                self._save_reward_curves_if_possible(force=False, ema_alpha=self.ema_alpha)

        curve_path = os.path.join(self.run_dir, "masac_training_curve.png")
        plt.savefig(curve_path)
        print(f"✅ 训练曲线已保存: {curve_path}")

        actor_final = os.path.join(self.run_dir, "actor_final.pth")
        q1_final = os.path.join(self.run_dir, "critic_q1_final.pth")
        q2_final = os.path.join(self.run_dir, "critic_q2_final.pth")
        torch.save(self.pi.learner.actor.state_dict(), actor_final)
        torch.save(self.pi.learner.q1.state_dict(), q1_final)
        torch.save(self.pi.learner.q2.state_dict(), q2_final)
        print(f"✅ 最终模型已保存到: {actor_final} | {q1_final} | {q2_final}")

        self._save_reward_curves_if_possible(force=True, ema_alpha=self.ema_alpha)
