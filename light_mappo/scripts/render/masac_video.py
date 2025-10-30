#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, numpy as np, torch, csv
from pathlib import Path

# === 项目根路径（本文件位于 light_mappo/scripts/render/videos/） ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# === 硬编码：你的 run2 目录、智能体数量、评估回合数 ===
DEFAULT_RUN_DIR = r"C:\Users\z8603\Desktop\study\highway with mappo\light_mappo\results\HighwayEnv\highway\masac\run7"
NUM_AGENTS       = 3
RENDER_EPISODES  = 10

# === 训练时的网络尺寸（只用来构建 policy 形状；不会更新） ===
HIDDEN_DIM  = 256
LR          = 3e-4
GAMMA       = 0.99
TARGET_TAU  = 0.005

# === 与原 video.py 保持一致的导入 ===
from light_mappo.envs.env_wrapper_for_pettingzoo import PettingZooWrapper
from light_mappo.envs.env_wrappers import DummyVecEnv
from light_mappo.algorithms.masac_discrete.policy import DMASACPolicy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

TRAIN_ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def make_rgb_env(num_agents: int):
    """与 MAPPO 的 video.py 一致：DummyVecEnv 包装 PettingZooWrapper(rgb_array)"""
    def get_env_fn(_):
        def init_env():
            return PettingZooWrapper(num_agents=num_agents, render_mode="rgb_array", debug=False)
        return init_env
    return DummyVecEnv([get_env_fn(0)])


def dig_core_env(obj):
    """递归剥壳，拿到底层 highway_env 环境对象（访问 config / controlled_vehicles / render）。"""
    seen = set(); stack = [obj]
    while stack:
        cur = stack.pop()
        if id(cur) in seen: continue
        seen.add(id(cur))
        if hasattr(cur, "controlled_vehicles") and hasattr(cur, "road"):
            return cur
        for name in ("env","unwrapped","par_env","aec_env","raw_env","_env","wrapped_env","environment"):
            if hasattr(cur, name):
                nxt = getattr(cur, name)
                if isinstance(nxt, (list, tuple)): stack.extend(list(nxt))
                else: stack.append(nxt)
    return obj


def resolve_actor_ckpt(run_or_pth: str) -> str:
    """允许传 run 目录或 actor 的 .pth；优先 actor_final.pth，否则取最新 actor_epXXXX.pth。"""
    p = os.path.abspath(os.path.expanduser(run_or_pth))
    if os.path.isfile(p) and p.lower().endswith(".pth"):
        return p
    if os.path.isdir(p):
        final = os.path.join(p, "actor_final.pth")
        if os.path.isfile(final): return final
        import re, glob
        cands = glob.glob(os.path.join(p, "actor_ep*.pth"))
        if cands:
            def ep_num(x):
                m = re.search(r"actor_ep(\d+)\.pth$", os.path.basename(x))
                return int(m.group(1)) if m else -1
            cands.sort(key=ep_num)
            return cands[-1]
    raise FileNotFoundError(f"未在 {p} 找到 actor_final.pth 或 actor_epXXXX.pth")


@torch.no_grad()
def record_episodes_masac(policy, envs, episodes: int, num_agents: int, out_dir: str):
    """与原 MAPPO video.py 的流程保持一致：一物理步一帧，保存视频、速度/加速度曲线与 CSV"""
    os.makedirs(out_dir, exist_ok=True)

    A  = num_agents
    nA = envs.action_space[0].n

    for ep in range(1, episodes + 1):
        # reset
        try:
            obs_list = envs.reset(seed=1 + ep * 1000)
        except TypeError:
            obs_list = envs.reset()
        obs = np.concatenate(obs_list)  # (A, Do)

        # 找到底层 env
        core = dig_core_env(envs.envs[0])

        # 评估阶段：让 policy_frequency = simulation_frequency（与原脚本一致）
        sim_fps = int(getattr(core, "config", {}).get("simulation_frequency", 15))
        try:
            core.configure({"policy_frequency": sim_fps})
        except Exception:
            core.config["policy_frequency"] = sim_fps
        pol_freq  = int(core.config.get("policy_frequency", 1))
        dt_policy = 1.0 / float(pol_freq)
        fps_out   = sim_fps  # 一步一帧

        # 关闭内部记录缓存
        if hasattr(core, "_record_video"):
            core._record_video = False

        # 动作名映射（训练动作顺序 → 环境动作索引）
        env_actions = list(getattr(core.action_type, "actions", [])) or []
        mapping = None
        try:
            if env_actions:
                idx_env = {n:i for i,n in enumerate(env_actions)}
                if all(a in idx_env for a in TRAIN_ACTIONS):
                    mapping = np.array([idx_env[a] for a in TRAIN_ACTIONS], dtype=np.int64)
        except Exception:
            mapping = None

        # 曲线缓存
        n_agents_in_env = len(getattr(core, "controlled_vehicles", []))
        agents_keys = [f"agent_{i}" for i in range(n_agents_in_env)]
        times_buf, speed_buf = [], {k: [] for k in agents_keys}
        accel_buf = {k: [] for k in agents_keys}

        # 视频 writer（参数与原脚本一致）
        ep_path = os.path.join(out_dir, f"render_ep{ep}.mp4")
        writer = imageio.get_writer(
            ep_path, fps=fps_out, codec="libx264",
            format="ffmpeg", macro_block_size=1, quality=8, pixelformat="yuv420p"
        )

        steps, t, prev_speeds = 0, 0.0, None
        MAX_STEPS = 100000
        try:
            while steps < MAX_STEPS:
                # 可行动作掩码
                try:
                    masks_dict = envs.envs[0].available_actions()
                    order = [f"agent_{i}" for i in range(A)]
                    avail = np.stack([masks_dict[a] for a in order], axis=0).astype(np.float32)
                except Exception:
                    avail = np.ones((A, nA), dtype=np.float32)

                # === MASAC：与训练一致的选择接口（贪心） ===
                a_idx = policy.select_actions(obs, avail=avail, greedy=True)  # (A,)
                if mapping is not None:
                    a_idx = np.take(mapping, a_idx)
                a_onehot    = np.eye(nA, dtype=np.float32)[a_idx]   # (A, nA)
                actions_env = a_onehot[np.newaxis, :, :]            # (1, A, nA) for DummyVecEnv

                # step
                obs_list, rewards, dones, infos = envs.step(actions_env)
                obs = np.concatenate(obs_list)

                # 采样速度/加速度
                cvs = getattr(core, "controlled_vehicles", []) or []
                cur_speeds = [float(v.speed) for v in cvs]
                if prev_speeds is None:
                    for i, s in enumerate(cur_speeds):
                        k = agents_keys[i]; speed_buf[k].append(s); accel_buf[k].append(0.0)
                    times_buf.append(t)
                else:
                    for i, s in enumerate(cur_speeds):
                        k = agents_keys[i]
                        speed_buf[k].append(s)
                        accel_buf[k].append((s - prev_speeds[i]) / dt_policy)
                    times_buf.append(t)
                prev_speeds = cur_speeds
                t += dt_policy

                # 抓帧
                try:
                    frame = core.render()
                    if frame is not None:
                        writer.append_data(frame)
                except Exception:
                    pass

                steps += 1
                if (dones == True).all() or np.all(dones):
                    break
        finally:
            writer.close()

        print(f"[OK] 保存：{ep_path} | fps={fps_out} | steps={steps} | sample_pts={len(times_buf)}")

        # 速度/加速度曲线与 CSV（与原脚本一致）
        if len(times_buf) > 1 and any(len(v) > 1 for v in speed_buf.values()):
            # 速度
            plt.figure(figsize=(10, 4))
            for agent, speeds in speed_buf.items():
                L = min(len(times_buf), len(speeds))
                if L > 1:
                    plt.plot(times_buf[:L], speeds[:L], label=str(agent))
            plt.xlabel("Time (s)"); plt.ylabel("Speed (m/s)")
            plt.title(f"Episode {ep} - Speed"); plt.legend(ncol=3, fontsize=8)
            plt.grid(True, alpha=0.3)
            sp_png = os.path.join(out_dir, f"speeds_ep{ep}.png")
            plt.savefig(sp_png, dpi=150, bbox_inches="tight"); plt.close()

            # 加速度
            plt.figure(figsize=(10, 4))
            for agent, accs in accel_buf.items():
                L = min(len(times_buf), len(accs))
                if L > 1:
                    plt.plot(times_buf[:L], accs[:L], label=str(agent))
            plt.xlabel("Time (s)"); plt.ylabel("Acceleration (m/s^2)")
            plt.title(f"Episode {ep} - Acceleration"); plt.legend(ncol=3, fontsize=8)
            plt.grid(True, alpha=0.3)
            ac_png = os.path.join(out_dir, f"accels_ep{ep}.png")
            plt.savefig(ac_png, dpi=150, bbox_inches="tight"); plt.close()

            # CSV
            csv_path = os.path.join(out_dir, f"kinematics_ep{ep}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                header = ["time"] + [f"{a}_speed" for a in agents_keys] + [f"{a}_accel" for a in agents_keys]
                w.writerow(header)
                max_len = max(
                    len(times_buf),
                    *(len(speed_buf[a]) for a in agents_keys),
                    *(len(accel_buf[a]) for a in agents_keys),
                )
                for i in range(max_len):
                    row = [times_buf[i] if i < len(times_buf) else ""]
                    for a in agents_keys: row.append(speed_buf[a][i] if i < len(speed_buf[a]) else "")
                    for a in agents_keys: row.append(accel_buf[a][i] if i < len(accel_buf[a]) else "")
                    w.writerow(row)
            print(f"[OK] 曲线/CSV 已保存：{sp_png} | {ac_png} | {csv_path}")
        else:
            lens = [len(v) for v in speed_buf.values()] if speed_buf else []
            print(f"[WARN] 曲线数据不足：times={len(times_buf)} agents={n_agents_in_env} lens={lens}")


def main():
    torch.manual_seed(1); np.random.seed(1)

    # 1) 解析权重 & 输出目录（硬编码 run2）
    ckpt = resolve_actor_ckpt(DEFAULT_RUN_DIR)
    out_dir = os.path.join(DEFAULT_RUN_DIR, "videos")
    print(f"[INFO] 使用 MASAC actor 权重：{ckpt}")
    print(f"[INFO] 输出目录：{out_dir}")

    # 2) 环境（推导维度、抓帧）
    envs = make_rgb_env(NUM_AGENTS)
    A  = NUM_AGENTS
    Dp = envs.observation_space[0].shape[0]
    NA = envs.action_space[0].n

    # 3) 构建 MASAC Policy（仅前向）
    cfg = type("Cfg", (), dict(
        num_agents=A, obs_shape=Dp, state_shape=A*Dp, n_actions=NA,
        hidden_dim=HIDDEN_DIM, lr=LR, gamma=GAMMA, target_tau=TARGET_TAU
    ))()
    policy = DMASACPolicy(cfg)

    # 4) 加载 actor
    state = torch.load(ckpt, map_location="cpu")
    try:
        policy.learner.actor.load_state_dict(state, strict=True)
    except Exception:
        if isinstance(state, dict):
            for k in ["actor","state_dict","model_state_dict"]:
                if k in state:
                    state = state[k]; break
        policy.learner.actor.load_state_dict(state, strict=False)
    policy.learner.actor.eval()

    # 5) 录制
    record_episodes_masac(policy, envs, RENDER_EPISODES, A, out_dir)
    envs.close()


if __name__ == "__main__":
    main()
