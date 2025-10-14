#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
from pathlib import Path

# === 确保能 import 项目 ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import get_config
from envs.env_wrappers import DummyVecEnv
from runner.shared.env_runner import EnvRunner as Runner
from runner.shared.env_runner import _t2n  # 复用工具

# --- 无显示环境也能画图 ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 根据需要改这里 ===
DEFAULT_RUN_DIR = r"C:\Users\z8603\Desktop\study\highway with mappo\light_mappo\results\HighwayEnv\highway\mappo\check\run9"
OUT_DIR = "videos"  # 输出目录

# 训练时使用的离散动作顺序（你给定的“正确表”）
TRAIN_ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

def make_rgb_env(all_args):
    """创建 render_mode='rgb_array' 的环境，只开 1 个并行环境。"""
    from envs.env_wrapper_for_pettingzoo import PettingZooWrapper
    def get_env_fn(_):
        def init_env():
            # 不做任何 mask 相关设置
            return PettingZooWrapper(num_agents=all_args.num_agents, render_mode="rgb_array")
        return init_env
    return DummyVecEnv([get_env_fn(0)])

def parse_args(argv, parser):
    """补齐渲染用到但 config.py 未定义的参数；若已存在则跳过，避免冲突。"""
    try:
        parser.add_argument("--scenario_name", type=str, default="highway")
    except Exception:
        pass
    try:
        parser.add_argument("--num_agents", type=int, default=3)
    except Exception:
        pass
    return parser.parse_known_args(argv)[0]

def _normalize_model_dir(user_dir: str) -> str:
    """允许传 runN 或 models 目录；返回包含 actor.pt/critic.pt 的 models 目录。"""
    if not user_dir:
        raise ValueError("请用 --model_dir 指定 runN 或 models 目录。")
    path = os.path.abspath(os.path.expanduser(user_dir))
    if not os.path.isdir(path):
        raise ValueError(f"模型目录不存在：{path}")
    models = path if os.path.basename(path).lower() == "models" else os.path.join(path, "models")
    a, c = os.path.join(models, "actor.pt"), os.path.join(models, "critic.pt")
    if not (os.path.isfile(a) and os.path.isfile(c)):
        raise ValueError(f"未在 {models} 找到 actor.pt / critic.pt")
    return models

def _dig_core_env(obj):
    """尽量挖到自定义 MARLEnv 实例（含 controlled_vehicles 的那个）."""
    seen = set()
    cur = obj
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "controlled_vehicles"):
            return cur
        for name in ("env", "unwrapped", "_env", "wrapped_env", "par_env", "aec_env"):
            if hasattr(cur, name):
                cur = getattr(cur, name)
                break
        else:
            break
    return obj  # 找不到就返回原对象

def _get_sim_fps(envs, all_args):
    """优先从底层 config['simulation_frequency'] 获取 fps；否则 1/ifi；再不行 30。"""
    try:
        core = _dig_core_env(envs.envs[0])
        cfg = getattr(core, "config", None)
        if isinstance(cfg, dict) and "simulation_frequency" in cfg:
            return int(cfg["simulation_frequency"])
    except Exception:
        pass
    if hasattr(all_args, "ifi"):
        try:
            if float(all_args.ifi) > 0:
                return int(round(1.0 / float(all_args.ifi)))
        except Exception:
            pass
    return 30

def _get_policy_dt(core) -> float:
    """每个 policy step 覆盖的物理时间 Δt（不改环境：用 config 粗略推导）"""
    cfg = getattr(core, "config", {}) if hasattr(core, "config") else {}
    sim_fps = int(cfg.get("simulation_frequency", 15))
    pol_freq = int(cfg.get("policy_frequency", 1))
    sim_fps = max(1, sim_fps)
    pol_freq = max(1, pol_freq)
    steps_per_action = max(1, sim_fps // pol_freq)
    return steps_per_action / float(sim_fps)

def _sha1(p: str) -> str:
    import hashlib
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()

@torch.no_grad()
def record_episodes(runner, envs, episodes: int):
    """
    已加载策略 → 决策 → env.step；
    - 录制：优先消耗 env._captured_frames（逐物理步），否则每个 policy step 渲染一帧；
    - 曲线：每回合导出速度/加速度 png 和 CSV（零侵入：直接读 controlled_vehicles）；
    - 不使用/不依赖任何 mask；
    - 可选：将“训练动作顺序”索引映射为“环境动作顺序”索引（仅 Discrete）。
    """
    import imageio
    import csv

    os.makedirs(OUT_DIR, exist_ok=True)
    n_threads = runner.n_rollout_threads
    num_agents = runner.num_agents
    sim_fps = _get_sim_fps(envs, runner.all_args)

    # ====== 调试：打印权重与 SHA1 ======
    actor_p = os.path.join(runner.all_args.model_dir, "actor.pt")
    critic_p = os.path.join(runner.all_args.model_dir, "critic.pt")
    try:
        print("[DEBUG] actor:", actor_p, _sha1(actor_p))
        print("[DEBUG] critic:", critic_p, _sha1(critic_p))
    except Exception as e:
        print("[DEBUG] SHA1 读取失败:", e)

    for ep in range(1, episodes + 1):
        # reset（尝试每回合换个 seed）
        try:
            obs = envs.reset(seed=int(getattr(runner.all_args, "seed", 1)) + ep * 1000)
        except TypeError:
            obs = envs.reset()

        core = _dig_core_env(envs.envs[0])

        # ====== 环境动作表（用于可选 remap） ======
        env_actions = list(getattr(core.action_type, "actions", [])) or []
        print("[DEBUG] env action list:", env_actions)
        try:
            print("[DEBUG] cfg:", {k: core.config.get(k) for k in [
                "lanes_count","vehicles_count","controlled_vehicles",
                "simulation_frequency","policy_frequency","duration"
            ]})
        except Exception:
            pass

        # === 训练动作顺序 -> 环境动作顺序 的索引映射（仅当完全匹配时启用）===
        mapping = None
        try:
            if len(env_actions) > 0:
                idx_env = {name: i for i, name in enumerate(env_actions)}
                if all(a in idx_env for a in TRAIN_ACTIONS):
                    mapping = np.array([idx_env[a] for a in TRAIN_ACTIONS], dtype=np.int64)
        except Exception:
            mapping = None
        if mapping is not None:
            print("[DEBUG] action index remap (train->env):", dict(zip(range(len(mapping)), mapping.tolist())))
        else:
            print("[DEBUG] action index remap disabled (env/train 清单对不上或非 Discrete).")

        # 帧缓存：若环境内部有逐物理步帧，清空读指针
        use_core_frames = hasattr(core, "_captured_frames") and isinstance(core._captured_frames, list)
        if use_core_frames:
            core._captured_frames.clear()
        last_frame_idx = 0

        # ====== 本地“曲线数据”（零侵入：直接读 controlled_vehicles） ======
        n_agents_in_env = len(getattr(core, "controlled_vehicles", []))
        agents_keys = [f"agent_{i}" for i in range(n_agents_in_env)]
        times_buf: list[float] = []
        speed_buf: dict[str, list] = {k: [] for k in agents_keys}
        accel_buf: dict[str, list] = {k: [] for k in agents_keys}
        dt_policy = _get_policy_dt(core)
        t = 0.0
        prev_speeds = None  # 用于差分

        # RNN/mask（与训练一致，但不做任何 mask 修改）
        rnn_states = np.zeros((n_threads, num_agents, runner.recurrent_N, runner.hidden_size), dtype=np.float32)
        masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)

        # 打开 MP4 写入器
        ep_path = os.path.join(OUT_DIR, f"render_ep{ep}.mp4")
        writer = imageio.get_writer(
            ep_path, fps=int(sim_fps), codec="libx264", format="ffmpeg",
            macro_block_size=1, quality=8, pixelformat="yuv420p"
        )

        steps = 0
        MAX_STEPS = 40_000
        try:
            while steps < MAX_STEPS:
                runner.trainer.prep_rollout()
                action, rnn_states = runner.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), n_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), n_threads))

                # 与训练一致的动作适配 + （可选）索引重映射（仅 Discrete）
                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        actions_env = uc if i == 0 else np.concatenate((actions_env, uc), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    act_idx = np.squeeze(actions, axis=2) if actions.ndim == 3 else actions
                    if mapping is not None:
                        act_idx = np.take(mapping, act_idx)
                    nA = envs.action_space[0].n
                    actions_env = np.squeeze(np.eye(nA)[act_idx], 2) if act_idx.ndim == 3 else np.eye(nA)[act_idx]
                else:
                    actions_env = actions

                # 环境前进一步
                obs, rewards, dones, infos = envs.step(actions_env)

                # ====== 采样速度并差分得到加速度（每个 policy step） ======
                cvs = getattr(core, "controlled_vehicles", [])
                cur_speeds = [float(v.speed) for v in cvs]
                if prev_speeds is None:
                    for i, s in enumerate(cur_speeds):
                        k = agents_keys[i]
                        speed_buf[k].append(s)
                        accel_buf[k].append(0.0)
                    times_buf.append(t)
                else:
                    for i, s in enumerate(cur_speeds):
                        k = agents_keys[i]
                        speed_buf[k].append(s)
                        a = (s - prev_speeds[i]) / max(1e-9, dt_policy)
                        accel_buf[k].append(a)
                    times_buf.append(t)
                prev_speeds = cur_speeds
                t += dt_policy

                # 写帧
                if use_core_frames:
                    cur_len = len(core._captured_frames)
                    if cur_len > last_frame_idx:
                        for f in core._captured_frames[last_frame_idx:]:
                            writer.append_data(f)
                        last_frame_idx = cur_len
                else:
                    frame = None
                    try:
                        frame = envs.envs[0].render()
                    except Exception:
                        frame = None
                    if frame is None:
                        try:
                            arr = envs.render("rgb_array")
                            if isinstance(arr, np.ndarray) and arr.ndim == 4:
                                frame = arr[0]
                        except Exception:
                            frame = None
                    if frame is not None:
                        writer.append_data(frame)

                steps += 1

                # 提前结束：所有 agent 都 done
                if (dones == True).all() or np.all(dones):
                    break

                # 维护 RNN/mask
                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), runner.recurrent_N, runner.hidden_size), dtype=np.float32
                )
                masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        finally:
            writer.close()

        print(f"[OK] 保存：{ep_path} | fps={sim_fps} | steps={steps} | "
              f"{'core_frames='+str(last_frame_idx) if use_core_frames else 'per-step frames'}")

        # ===== 曲线/CSV：用本地 times_buf/speed_buf/accel_buf 生成 =====
        if len(times_buf) > 1 and any(len(v) > 0 for v in speed_buf.values()):
            # 速度曲线
            plt.figure(figsize=(10, 4))
            for agent, speeds in speed_buf.items():
                L = min(len(times_buf), len(speeds))
                if L > 0:
                    plt.plot(times_buf[:L], speeds[:L], label=str(agent))
            plt.xlabel("Time (s)")
            plt.ylabel("Speed (m/s)")
            plt.title(f"Episode {ep} - Speed")
            plt.legend(ncol=3, fontsize=8)
            plt.grid(True, alpha=0.3)
            sp_png = os.path.join(OUT_DIR, f"speeds_ep{ep}.png")
            plt.savefig(sp_png, dpi=150, bbox_inches="tight")
            plt.close()

            # 加速度曲线
            plt.figure(figsize=(10, 4))
            for agent, accs in accel_buf.items():
                L = min(len(times_buf), len(accs))
                if L > 0:
                    plt.plot(times_buf[:L], accs[:L], label=str(agent))
            plt.xlabel("Time (s)")
            plt.ylabel("Acceleration (m/s^2)")
            plt.title(f"Episode {ep} - Acceleration")
            plt.legend(ncol=3, fontsize=8)
            plt.grid(True, alpha=0.3)
            ac_png = os.path.join(OUT_DIR, f"accels_ep{ep}.png")
            plt.savefig(ac_png, dpi=150, bbox_inches="tight")
            plt.close()

            # CSV
            csv_path = os.path.join(OUT_DIR, f"kinematics_ep{ep}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                header = ["time"] + [f"{a}_speed" for a in agents_keys] + [f"{a}_accel" for a in agents_keys]
                w.writerow(header)
                max_len = max(len(times_buf), *(len(speed_buf[a]) for a in agents_keys), *(len(accel_buf[a]) for a in agents_keys))
                for i in range(max_len):
                    row = [times_buf[i] if i < len(times_buf) else ""]
                    for a in agents_keys:
                        row.append(speed_buf[a][i] if i < len(speed_buf[a]) else "")
                    for a in agents_keys:
                        row.append(accel_buf[a][i] if i < len(accel_buf[a]) else "")
                    w.writerow(row)

            print(f"[OK] 曲线/CSV 已保存：{sp_png} | {ac_png} | {csv_path}")
        else:
            print("[WARN] 曲线数据为空。")

def main(argv):
    parser = get_config()
    all_args = parse_args(argv, parser)

    # 渲染只开 1 线程
    all_args.use_render = True
    if not getattr(all_args, "n_render_rollout_threads", None):
        all_args.n_render_rollout_threads = 1
    all_args.n_rollout_threads = all_args.n_render_rollout_threads

    # 设备 & 随机
    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    # 模型目录：允许直接运行按钮（未传则用默认 runN）
    model_dir = all_args.model_dir or DEFAULT_RUN_DIR
    model_dir = _normalize_model_dir(model_dir)  # -> models 目录
    all_args.model_dir = model_dir
    print(f"[INFO] 使用模型目录：{model_dir}")
    run_dir = Path(model_dir).parent  # runN

    # 环境（rgb_array）
    envs = make_rgb_env(all_args)

    # Runner & 加载
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }
    runner = Runner(config)
    runner.restore()  # 从 all_args.model_dir 载入 actor.pt/critic.pt

    # 渲染多少个 episode：用 config.py 的 render_episodes（默认 5）
    episodes = int(getattr(all_args, "render_episodes", 5))
    record_episodes(runner, envs, episodes)

    envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])
