import os, sys, numpy as np, torch
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import get_config
from envs.env_wrappers import DummyVecEnv
from runner.shared.env_runner import EnvRunner as Runner
from runner.shared.env_runner import _t2n

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RUN_DIR = r"C:\Users\z8603\Desktop\study\highway with mappo\light_mappo\results\HighwayEnv\highway\mappo\check\run10"
OUT_DIR = "videos"
TRAIN_ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

def make_rgb_env(all_args):
    from envs.env_wrapper_for_pettingzoo import PettingZooWrapper
    def get_env_fn(_):
        def init_env():
            return PettingZooWrapper(num_agents=all_args.num_agents, render_mode="rgb_array")
        return init_env
    return DummyVecEnv([get_env_fn(0)])

def parse_args(argv, parser):
    try: parser.add_argument("--scenario_name", type=str, default="highway")
    except Exception: pass
    try: parser.add_argument("--num_agents", type=int, default=3)
    except Exception: pass
    return parser.parse_known_args(argv)[0]

def _normalize_model_dir(user_dir: str) -> str:
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

def dig_core_env(obj):
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

def _sha1(p: str) -> str:
    import hashlib
    with open(p, "rb") as f: return hashlib.sha1(f.read()).hexdigest()

@torch.no_grad()
def record_episodes(runner, envs, episodes: int):
    import imageio, csv
    os.makedirs(OUT_DIR, exist_ok=True)
    n_threads = runner.n_rollout_threads
    num_agents = runner.num_agents

    for ep in range(1, episodes + 1):
        # reset
        try:
            obs = envs.reset(seed=int(getattr(runner.all_args, "seed", 1)) + ep * 1000)
        except TypeError:
            obs = envs.reset()
        core = dig_core_env(envs.envs[0])

        # 评估期：让 policy_frequency = simulation_frequency（不改训练，仅本次渲染）
        sim_fps = int(getattr(core, "config", {}).get("simulation_frequency", 15))
        try:
            core.configure({"policy_frequency": sim_fps})
        except Exception:
            # 老版本没有 configure 时直接改字典
            core.config["policy_frequency"] = sim_fps
        pol_freq = int(core.config.get("policy_frequency", 1))  # 现在应等于 sim_fps
        dt_policy = 1.0 / float(pol_freq)
        fps_out = sim_fps  # 一步一帧，流畅

        # 关闭环境内部帧缓存（我们每步抓一次）
        if hasattr(core, "_record_video"): core._record_video = False

        # 动作表（可选 remap）
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

        # RNN/mask
        rnn_states = np.zeros((n_threads, num_agents, runner.recurrent_N, runner.hidden_size), dtype=np.float32)
        masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)

        # writer
        ep_path = os.path.join(OUT_DIR, f"render_ep{ep}.mp4")
        writer = imageio.get_writer(ep_path, fps=fps_out, codec="libx264",
                                    format="ffmpeg", macro_block_size=1,
                                    quality=8, pixelformat="yuv420p")

        steps, t, prev_speeds = 0, 0.0, None
        MAX_STEPS = 100000
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

                # 适配动作
                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        actions_env = uc if i == 0 else np.concatenate((actions_env, uc), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    act_idx = np.squeeze(actions, axis=2) if actions.ndim == 3 else actions
                    if mapping is not None: act_idx = np.take(mapping, act_idx)
                    nA = envs.action_space[0].n
                    actions_env = np.squeeze(np.eye(nA)[act_idx], 2) if act_idx.ndim == 3 else np.eye(nA)[act_idx]
                else:
                    actions_env = actions

                # 前进一步（现在一步=一个物理子步）
                obs, rewards, dones, infos = envs.step(actions_env)

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

                # 每步抓一帧
                try:
                    frame = core.render()
                    if frame is not None: writer.append_data(frame)
                except Exception:
                    pass

                steps += 1
                if (dones == True).all() or np.all(dones): break

                # 维护 RNN/mask
                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), runner.recurrent_N, runner.hidden_size), dtype=np.float32
                )
                masks = np.ones((n_threads, num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        finally:
            writer.close()

        print(f"[OK] 保存：{ep_path} | fps={fps_out} | steps={steps} | sample_pts={len(times_buf)}")

        # 曲线/CSV
        if len(times_buf) > 1 and any(len(v) > 1 for v in speed_buf.values()):
            plt.figure(figsize=(10, 4))
            for agent, speeds in speed_buf.items():
                L = min(len(times_buf), len(speeds))
                if L > 1: plt.plot(times_buf[:L], speeds[:L], label=str(agent))
            plt.xlabel("Time (s)"); plt.ylabel("Speed (m/s)")
            plt.title(f"Episode {ep} - Speed"); plt.legend(ncol=3, fontsize=8)
            plt.grid(True, alpha=0.3)
            sp_png = os.path.join(OUT_DIR, f"speeds_ep{ep}.png")
            plt.savefig(sp_png, dpi=150, bbox_inches="tight"); plt.close()

            plt.figure(figsize=(10, 4))
            for agent, accs in accel_buf.items():
                L = min(len(times_buf), len(accs))
                if L > 1: plt.plot(times_buf[:L], accs[:L], label=str(agent))
            plt.xlabel("Time (s)"); plt.ylabel("Acceleration (m/s^2)")
            plt.title(f"Episode {ep} - Acceleration"); plt.legend(ncol=3, fontsize=8)
            plt.grid(True, alpha=0.3)
            ac_png = os.path.join(OUT_DIR, f"accels_ep{ep}.png")
            plt.savefig(ac_png, dpi=150, bbox_inches="tight"); plt.close()

            csv_path = os.path.join(OUT_DIR, f"kinematics_ep{ep}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                header = ["time"] + [f"{a}_speed" for a in agents_keys] + [f"{a}_accel" for a in agents_keys]
                w.writerow(header)
                max_len = max(len(times_buf), *(len(speed_buf[a]) for a in agents_keys), *(len(accel_buf[a]) for a in agents_keys))
                for i in range(max_len):
                    row = [times_buf[i] if i < len(times_buf) else ""]
                    for a in agents_keys: row.append(speed_buf[a][i] if i < len(speed_buf[a]) else "")
                    for a in agents_keys: row.append(accel_buf[a][i] if i < len(accel_buf[a]) else "")
                    w.writerow(row)
            print(f"[OK] 曲线/CSV 已保存：{sp_png} | {ac_png} | {csv_path}")
        else:
            lens = [len(v) for v in speed_buf.values()] if speed_buf else []
            print(f"[WARN] 曲线数据不足：times={len(times_buf)} agents={n_agents_in_env} lens={lens}")

def main(argv):
    parser = get_config()
    all_args = parse_args(argv, parser)
    all_args.use_render = True
    if not getattr(all_args, "n_render_rollout_threads", None):
        all_args.n_render_rollout_threads = 1
    all_args.n_rollout_threads = all_args.n_render_rollout_threads

    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(all_args.seed); np.random.seed(all_args.seed)

    model_dir = all_args.model_dir or DEFAULT_RUN_DIR
    model_dir = _normalize_model_dir(model_dir); all_args.model_dir = model_dir
    print(f"[INFO] 使用模型目录：{model_dir}")
    run_dir = Path(model_dir).parent

    envs = make_rgb_env(all_args)
    config = {"all_args": all_args, "envs": envs, "eval_envs": None,
              "num_agents": all_args.num_agents, "device": device, "run_dir": run_dir}
    runner = Runner(config); runner.restore()

    episodes = int(getattr(all_args, "render_episodes", 5))
    record_episodes(runner, envs, episodes)
    envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])

