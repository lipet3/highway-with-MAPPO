#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 项目根目录加入 sys.path（本脚本位于 light_mappo/scripts/）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from light_mappo.envs.env_wrapper_for_pettingzoo import PettingZooWrapper
from light_mappo.algorithms.masac_discrete.policy import DMASACPolicy


# ===== 用户可配置区域 =====
# 硬编码模型 run 目录（可手动修改）
RUN_DIR = r"C:\Users\z8603\Desktop\study\highway with mappo\light_mappo\results\HighwayEnv\highway\masac\run34"
# 评估回合数（可手动修改）
NUM_EVAL_EPISODES = 300

# 每多少回合打印一次聚合指标
PRINT_EVERY = 100


def _resolve_actor_ckpt(run_or_pth: str) -> str:
    p = os.path.abspath(os.path.expanduser(run_or_pth))
    if os.path.isfile(p) and p.lower().endswith(".pth"):
        return p
    if os.path.isdir(p):
        final = os.path.join(p, "actor_final.pth")
        if os.path.isfile(final):
            return final
        # 回退：找最新的 actor_epXXXX.pth
        import re, glob
        cands = glob.glob(os.path.join(p, "actor_ep*.pth"))
        if cands:
            def ep_num(x):
                m = re.search(r"actor_ep(\d+)\.pth$", os.path.basename(x))
                return int(m.group(1)) if m else -1
            cands.sort(key=ep_num)
            return cands[-1]
    raise FileNotFoundError(f"未在 {p} 找到 actor_final.pth 或 actor_epXXXX.pth")


def _dig_core_env(obj):
    """从 PettingZooWrapper → 底层 highway_env 环境对象（访问车辆/速度/碰撞）。"""
    seen = set()
    stack = [obj]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if hasattr(cur, "controlled_vehicles") and hasattr(cur, "road"):
            return cur
        for name in ("env", "unwrapped", "par_env", "aec_env", "raw_env", "_env", "wrapped_env", "environment"):
            if hasattr(cur, name):
                nxt = getattr(cur, name)
                if isinstance(nxt, (list, tuple)):
                    stack.extend(list(nxt))
                else:
                    stack.append(nxt)
    return obj


@torch.no_grad()
def evaluate():
    # 1) 环境（不渲染，最快速评估）
    num_agents = 3  # 与训练一致；如需不同，请和环境保持一致
    env = PettingZooWrapper(num_agents=num_agents, render_mode=None, debug=False)

    # 2) 维度与策略
    A = num_agents
    obs_dim = env.observation_space[0].shape[0]
    n_actions = env.action_space[0].n

    cfg = type("Cfg", (), dict(
        num_agents=A,
        obs_shape=obs_dim,
        state_shape=A * obs_dim,
        n_actions=n_actions,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        target_tau=0.005,
    ))()
    policy = DMASACPolicy(cfg)

    # 3) 加载权重
    ckpt = _resolve_actor_ckpt(RUN_DIR)
    state = torch.load(ckpt, map_location="cpu")
    try:
        policy.learner.actor.load_state_dict(state, strict=True)
    except Exception:
        if isinstance(state, dict):
            for k in ["actor", "state_dict", "model_state_dict"]:
                if k in state:
                    state = state[k]
                    break
        policy.learner.actor.load_state_dict(state, strict=False)
    policy.learner.actor.eval()

    # 4) 评估循环
    records = []  # 每回合记录
    hundred_window = []  # 最近 PRINT_EVERY 回合窗口

    # 碰撞回合路径图保存目录（仅保留发生碰撞的回合）
    collision_dir = os.path.join(RUN_DIR, "collision")
    os.makedirs(collision_dir, exist_ok=True)

    for ep in range(1, NUM_EVAL_EPISODES + 1):
        # reset
        try:
            obs = env.reset()
        except TypeError:
            obs = env.reset(seed=None)

        core = _dig_core_env(env)
        # 初始化速度采样
        step_count = 0
        speed_sum = 0.0  # 按每步对所有 agent 的速度求平均后，再累加
        collided = False

        # 回合循环：记录每个智能体的 (x坐标, 车道编号, 速度) 轨迹
        x_trajectories = [[] for _ in range(A)]      # 每个智能体在每个时间步的 x 坐标
        lane_trajectories = [[] for _ in range(A)]   # 每个智能体在每个时间步的车道编号
        speed_trajectories = [[] for _ in range(A)]  # 每个智能体在每个时间步的速度
        
        # 辅助函数：记录当前步的 x 坐标、车道编号和速度
        def _record_trajectories():
            cvs = getattr(core, "controlled_vehicles", []) or []
            for i in range(A):
                if i < len(cvs):
                    # 记录x坐标
                    pos = getattr(cvs[i], "position", None)
                    if pos is not None and len(pos) >= 1:
                        x_trajectories[i].append(float(pos[0]))
                    else:
                        # 如果x坐标不存在，使用上一个值或默认值
                        if len(x_trajectories[i]) > 0:
                            x_trajectories[i].append(x_trajectories[i][-1])
                        else:
                            x_trajectories[i].append(0.0)
                    
                    # 记录车道编号
                    lane_idx = getattr(cvs[i], "lane_index", None)
                    if lane_idx is not None and isinstance(lane_idx, tuple) and len(lane_idx) >= 3:
                        lane_id = int(lane_idx[2]) if lane_idx[2] is not None else 0
                        lane_trajectories[i].append(lane_id)
                    else:
                        lane_trajectories[i].append(0)  # 默认车道 0

                    # 记录速度
                    speed_val = float(getattr(cvs[i], "speed", 0.0))
                    speed_trajectories[i].append(speed_val)
                else:
                    # 如果车辆不存在，使用上一个值（如果存在）或默认值
                    if len(x_trajectories[i]) > 0:
                        x_trajectories[i].append(x_trajectories[i][-1])
                    else:
                        x_trajectories[i].append(0.0)
                    
                    if len(lane_trajectories[i]) > 0:
                        lane_trajectories[i].append(lane_trajectories[i][-1])
                    else:
                        lane_trajectories[i].append(0)

                    if len(speed_trajectories[i]) > 0:
                        speed_trajectories[i].append(speed_trajectories[i][-1])
                    else:
                        speed_trajectories[i].append(0.0)
        
        # 记录初始状态（reset 后的状态）
        _record_trajectories()
        
        while True:
            # 掩码
            try:
                avail = env.available_actions()  # (A, NA)
                avail = np.asarray(avail, dtype=np.float32)
                if avail.shape != (A, n_actions):
                    # 兜底对齐
                    if avail.shape[1] < n_actions:
                        pad = np.ones((A, n_actions - avail.shape[1]), dtype=np.float32)
                        avail = np.concatenate([avail, pad], axis=1)
                    elif avail.shape[1] > n_actions:
                        avail = avail[:, :n_actions]
            except Exception:
                avail = np.ones((A, n_actions), dtype=np.float32)

            # 选择动作（贪心）
            a_idx = policy.select_actions(obs, avail=avail, greedy=True)
            a_onehot = np.eye(n_actions, dtype=np.float32)[a_idx]

            # 环境步进
            obs2, rewards, dones, infos = env.step(a_onehot)

            # 统计速度 / 碰撞
            cvs = getattr(core, "controlled_vehicles", []) or []
            if cvs:
                spds = [float(v.speed) for v in cvs]
                speed_sum += float(np.mean(spds))
            if any(getattr(v, "crashed", False) for v in cvs):
                collided = True

            # 记录x坐标和车道编号（每步都记录）
            _record_trajectories()

            step_count += 1
            obs = obs2
            if np.any(dones):
                break

        ep_avg_speed = (speed_sum / step_count) if step_count > 0 else 0.0

        # 若发生碰撞，绘制并保存路径图（横坐标：x坐标，纵坐标：车道编号 0/1/2）
        if collided:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
                for i in range(A):
                    if len(x_trajectories[i]) == 0 or len(lane_trajectories[i]) == 0:
                        continue
                    xs = x_trajectories[i]  # x轴：x坐标
                    lanes = lane_trajectories[i]  # y轴：车道编号
                    # agent_2（索引为2）用虚线，其他用实线
                    linestyle = '--' if i == 2 else '-'
                    ax.plot(xs, lanes, label=f"agent_{i}", color=colors[i % len(colors)], 
                           linewidth=1.5, linestyle=linestyle, marker='o', markersize=3)
                    # 标记最后一个点
                    if len(xs) > 0 and len(lanes) > 0:
                        ax.scatter(xs[-1], lanes[-1], color=colors[i % len(colors)], s=30, zorder=5)
                ax.set_xlabel("x", fontsize=12)
                ax.set_ylabel("Lane", fontsize=12)
                ax.set_title(f"Episode {ep} Lane Trajectories (collision)", fontsize=13)
                ax.set_yticks([0, 1, 2])  # 固定显示车道 0, 1, 2
                ax.set_ylim(-0.2, 2.2)  # 留出一点边距
                ax.grid(True, alpha=0.3)
                ax.legend()
                out_png = os.path.join(collision_dir, f"ep{ep:05d}.png")
                fig.tight_layout()
                fig.savefig(out_png, dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] Failed to save trajectory plot for ep {ep}: {e}")
                plt.close("all")
            
            # 保存碰撞回合的 CSV 文件（每个 step 的每辆车的 x 坐标和车道）
            try:
                # 确定最大步数（所有轨迹的最大长度）
                max_steps = max(len(x_trajectories[i]) for i in range(A)) if A > 0 else 0
                
                # 构建 CSV 数据
                csv_data = []
                for step in range(max_steps):
                    row = {"step": step}
                    for i in range(A):
                        # x 坐标
                        x_val = x_trajectories[i][step] if step < len(x_trajectories[i]) else (x_trajectories[i][-1] if len(x_trajectories[i]) > 0 else 0.0)
                        row[f"agent_{i}_x"] = float(x_val)
                        
                        # 车道编号（只记录车道 0 和 1）
                        lane_val = lane_trajectories[i][step] if step < len(lane_trajectories[i]) else (lane_trajectories[i][-1] if len(lane_trajectories[i]) > 0 else 0)
                        # 只记录车道 0 和 1，其他车道也记录但标注
                        row[f"agent_{i}_lane"] = int(lane_val)

                        # 速度
                        speed_val = speed_trajectories[i][step] if step < len(speed_trajectories[i]) else (speed_trajectories[i][-1] if len(speed_trajectories[i]) > 0 else 0.0)
                        row[f"agent_{i}_speed"] = float(speed_val)
                    csv_data.append(row)
                
                # 保存为 CSV
                csv_df = pd.DataFrame(csv_data)
                out_csv = os.path.join(collision_dir, f"ep{ep:05d}.csv")
                csv_df.to_csv(out_csv, index=False, encoding="utf-8")
                print(f"[OK] 碰撞回合 {ep} 的轨迹 CSV 已保存: {out_csv}")
            except Exception as e:
                print(f"[WARN] Failed to save trajectory CSV for ep {ep}: {e}")
                import traceback
                traceback.print_exc()
        ep_info = {
            "episode": ep,
            "collided": int(bool(collided)),
            "avg_speed": float(ep_avg_speed),
        }
        records.append(ep_info)
        hundred_window.append(ep_info)

        # 每 PRINT_EVERY 回合打印一次统计
        if ep % PRINT_EVERY == 0:
            window = hundred_window[-PRINT_EVERY:]
            col_rate = float(np.mean([x["collided"] for x in window]))
            avg_spd = float(np.mean([x["avg_speed"] for x in window]))
            print(f"[EVAL] ep={ep:5d} | last_{PRINT_EVERY}_episodes: collision_rate={col_rate:.3f}, avg_speed={avg_spd:.2f}")

    # 5) 写入 Excel
    out_dir = os.path.join(RUN_DIR, "eval_metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"metrics_E{NUM_EVAL_EPISODES}.xlsx")
    df = pd.DataFrame.from_records(records)
    df.to_excel(out_xlsx, index=False)
    print(f"[OK] 评估结果已保存: {out_xlsx}")


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    evaluate()


