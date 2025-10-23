import argparse
import numpy as np

from light_mappo.envs.env_wrapper_for_pettingzoo import PettingZooWrapper
from light_mappo.algorithms.QMIX.policy import QMIXPolicy
from light_mappo.algorithms.QMIX.buffer import ReplayBuffer
from light_mappo.runner.QMIX.runner_qmix import RunnerQMIX
import sys
import os

sys.path.append("C:/Users/z8603/Desktop/study/highway with mappo/light_mappo")
def make_env(num_agents):
    # 由 PettingZooWrapper 内部自己创建 HighwayParallelEnv
    return PettingZooWrapper(num_agents=num_agents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--episode_length", type=int, default=200)
    parser.add_argument("--train_iters", type=int, default=20000)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--update_per_ep", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--double_q", action="store_true", default=True)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--target_update_interval", type=int, default=200)
    parser.add_argument("--rnn", action="store_true", default=True)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=20000)
    args = parser.parse_args()

    env = make_env(args.num_agents)
    # 从 wrapper 推导维度
    A = args.num_agents
    Dp = env.observation_space[0].shape[0]
    NA = env.action_space[0].n

    args.num_agents  = A
    args.obs_shape   = Dp
    args.state_shape = A * Dp
    args.n_actions   = NA

    policy = QMIXPolicy(args)
    buffer = ReplayBuffer(args.buffer_size)
    runner = RunnerQMIX(env, policy, buffer, args, logger=None)

    runner.train()

if __name__ == "__main__":
    main()
