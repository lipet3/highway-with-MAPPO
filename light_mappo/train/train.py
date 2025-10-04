

# !/usr/bin/env python
# !/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# 直接添加light_mappo目录的路径
sys.path.append("C:/Users/z8603/Desktop/study/highway with mappo/light_mappo")

from config import get_config
from envs.env_wrappers import DummyVecEnv



"""Train script for MPEs."""




def make_train_env(all_args):
    # 1. 导入我们的终极适配器
    from envs.env_wrapper_for_pettingzoo import PettingZooWrapper

    # 2. 创建一个返回适配器实例的函数列表
    #    DummyVecEnv 会帮我们处理好这个列表
    def get_env_fn(rank):  # rank 是并行环境的索引
        def init_env():
            # 这里创建的是我们的终极适配器
            env = PettingZooWrapper(num_agents=all_args.num_agents)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    # 3. 仍然使用 DummyVecEnv，它会把我们的适配器包装起来
    #    这符合 light_mappo 的原始设计，侵入性最小
    #    n_rollout_threads 已经是 1 了
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])






# make_eval_env 也做完全相同的修改
def make_eval_env(all_args):
    from envs.env_wrapper_for_pettingzoo import PettingZooWrapper
    def get_env_fn(rank):
        def init_env():
            env = PettingZooWrapper(num_agents=all_args.num_agents)
            # env.seed(all_args.seed * 50000 + rank * 1000)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])



def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="highway", help="Which scenario to run on")
    parser.add_argument("--num_agents", type=int, default=3, help="number of agents in highway")
    all_args = parser.parse_known_args(args)[0]
    return all_args




def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert (
        all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
    ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
