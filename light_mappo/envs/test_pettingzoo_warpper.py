# 文件名: light_mappo/envs/test_pettingzoo_wrapper.py

from pettingzoo.test import parallel_api_test
from env_pettingzoo_highway import HighwayParallelEnv  # 从我们刚创建的文件中导入

if __name__ == "__main__":
    # 创建我们的包装器环境实例
    my_env = HighwayParallelEnv(num_agents=4)

    # 使用官方的 API 测试工具进行检查
    print("正在使用 PettingZoo 官方 API 测试工具进行检查...")
    parallel_api_test(my_env, num_cycles=1000)

    print("\n检查通过！您的 PettingZoo 包装器符合标准。")

    # 可选：手动运行几步看看
    print("\n--- 手动测试 ---")
    obs, info = my_env.reset()
    print("Reset 后的观测 (第一个 agent):", obs['agent_0'].shape)

    for _ in range(5):
        actions = {agent: my_env.action_space(agent).sample() for agent in my_env.agents}
        obs, rewards, terminations, truncations, infos = my_env.step(actions)
        print("Step 后的奖励 (第一个 agent):", rewards['agent_0'])