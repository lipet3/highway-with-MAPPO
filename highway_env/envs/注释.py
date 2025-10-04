import time  # 导入 time 模块，虽然在此代码中未直接使用，但通常用于调试

# 定义观测值的类型提示，明确 obs 是一个 NumPy 数组
Observation = np.ndarray


class MARLEnv(AbstractEnv):
    """
    一个为“三车协同汇流并组建车队”任务高度定制的多智能体强化学习环境。
    """

    # metadata 用于向外部库（如 Gymnasium, PettingZoo）声明环境支持的功能
    metadata = {"render_modes": ["human", "rgb_array"], "name": "MARLEnv-v0"}

    @classmethod
    def default_config(cls) -> dict:
        """
        定义环境的所有可配置参数的默认值。
        这使得环境的创建和修改非常灵活。
        """
        # 首先，继承父类的默认配置
        config = super().default_config()
        # 然后，用我们自定义的参数来更新或覆盖它
        config.update({
            # --- 核心结构配置 ---
            "observation": {"type": "MultiAgentObservation", "observation_config": {"type": "Kinematics"}},
            "action": {"type": "MultiAgentAction", "action_config": {"type": "DiscreteMetaAction"}},

            # --- 场景配置 ---
            "lanes_count": 2,  # 道路的车道数量
            "vehicles_count": 1,  # 由环境AI控制的背景车辆（HDV）数量
            "controlled_vehicles": 3,  # 由我们的 MARL 策略控制的智能体数量 (A, B, C)

            # --- 任务与终止条件配置 ---
            "duration": 40,  # 回合的最大模拟时长（秒），超时则截断
            "max_steps": 2000,  # 回合的最大决策步数，超时则截断
            "offroad_terminal": True,  # 车辆驶出路面则回合终止

            # --- 奖励函数工程 (Reward Engineering) ---
            "collision_reward": -800,  # 发生碰撞时的惩罚
            "high_speed_reward": 1,  # 保持高速行驶的奖励
            "low_speed_penalty": -2,  # 速度过慢的惩罚
            "lane_change_reward": 20000,  # 车辆A成功换道时的一次性巨额奖励
            "timeout_penalty": -500,  # 因超时而结束回合时的惩罚（当前未直接使用，但可扩展）

            # --- 物理参数阈值 ---
            "low_speed_threshold": 15,  # [m/s] 低于此速度则触发惩罚
            "high_speed_threshold": 25,  # [m/s] 高于此速度则触发奖励

            # --- 渲染配置 ---
            "screen_width": 1200,  # 渲染窗口的宽度（像素）
            "screen_height": 300,  # 渲染窗口的高度（像素）
        })
        return config

    def _reset(self) -> None:
        """
        在每个回合开始时被调用，用于重置环境到初始状态。
        """
        self._create_road()  # 重新创建道路
        self._create_vehicles()  # 重新按规则生成车辆

        # 重置用于跟踪任务状态的变量
        self.agent_a_lane_changed = False  # 标记车辆A是否已成功换道，确保巨额奖励只给一次
        self.steps = 0  # 将决策步计数器归零
        self.time = 0.0  # 将模拟时间计数器归零

        # --- NEW: 重置用于绘图” -
        self.time_history = []
        self.speed_history = {f"agent_{i}": [] for i in range(self.config["controlled_vehicles"])}
        self.accel_history = {f"agent_{i}": [] for i in range(self.config["controlled_vehicles"])}
        self.prev_speeds = [v.speed for v in self.controlled_vehicles] if hasattr(self, 'controlled_vehicles') else []

    def _create_road(self) -> None:
        """
        创建道路布局。此实现使用了 highway-env 的旧版本 API。
        """
        net = RoadNetwork()
        lane_width = 4.0
        road_length = 1000.0

        # 定义车道0（右侧）和车道1（左侧）的起点和终点
        lane_0_start, lane_0_end = np.array([0.0, 0.0]), np.array([road_length, 0.0])
        lane_1_start, lane_1_end = np.array([0.0, lane_width]), np.array([road_length, lane_width])

        # 将创建的 StraightLane 对象添加到路网中
        net.add_lane("a", "b", StraightLane(lane_0_start, lane_0_end, width=lane_width,
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS_LINE)))
        net.add_lane("a", "b", StraightLane(lane_1_start, lane_1_end, width=lane_width,
                                            line_types=(LineType.STRIPED, LineType.STRIPED)))

        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """
        根据任务设计，精确地创建并放置所有车辆。
        这是环境定制化最核心的部分之一。
        """
        self.controlled_vehicles = []

        # 1. 创建固定的车队目标：车辆B和车辆C
        vehicle_b = self.action_type.vehicle_class(self.road, self.road.network.get_lane(("a", "b", 0)).position(20, 0),
                                                   speed=20)
        self.controlled_vehicles.append(vehicle_b)

        vehicle_c = self.action_type.vehicle_class(self.road, self.road.network.get_lane(("a", "b", 0)).position(80, 0),
                                                   speed=20)
        self.controlled_vehicles.append(vehicle_c)

        # 2. 创建随机化的主角：车辆A
        pos_b_x, pos_c_x = vehicle_b.position[0], vehicle_c.position[0]
        # 在一个广阔的范围内随机化A的初始位置，以增强策略的泛化能力
        random_x = self.np_random.uniform(pos_b_x - 40, pos_c_x + 40)
        vehicle_a = self.action_type.vehicle_class(self.road,
                                                   self.road.network.get_lane(("a", "b", 1)).position(random_x, 0),
                                                   speed=20)
        # 将车辆A插入到列表的第一个位置，方便我们始终用索引0来访问它
        self.controlled_vehicles.insert(0, vehicle_a)

        # 3. 创建作为障碍物的HDV
        hdv = Vehicle(self.road, self.road.network.get_lane(("a", "b", 1)).position(vehicle_a.position[0] + 30, 0),
                      speed=15)

        # 4. 将所有车辆部署到道路上
        self.road.vehicles.clear()  # 确保每次重置时都是一个全新的开始
        self.road.vehicles.extend(self.controlled_vehicles)
        self.road.vehicles.append(hdv)

    def step(self, action: Tuple[int, ...]) -> Tuple[
        Observation, Tuple[float, ...], Tuple[bool, ...], Tuple[bool, ...], Dict]:
        """
        执行一个完整的决策步。这是环境的“心脏”。
        """
        # --- 动作屏蔽 (Action Masking) ---
        # 我们先复制一份动作列表，因为元组是不可修改的
        mutable_action = list(action)
        vehicle_a = self.controlled_vehicles[0]
        # 检查车辆A是否已经完成了换道任务
        if self.agent_a_lane_changed or vehicle_a.lane_index[2] == 0:
            self.agent_a_lane_changed = True
            # 如果是，则锁定它的车道，让它专注于保持队形
            if mutable_action[0] in [2, 3]: mutable_action[0] = 0  # 2=向右, 3=向左 -> 0=保持

        # 永久锁定车辆B和C的车道，确保车队目标稳定
        for i in range(1, 3):
            if mutable_action[i] in [2, 3]: mutable_action[i] = 0

        # --- 核心交互流程 ---
        # 1. 将（可能被修改过的）动作应用到环境中，并推进物理模拟
        self._simulate(tuple(mutable_action))

        # 2. 从环境中收集所有智能体的新观测
        obs = self.observation_type.observe()

        # 3. 根据新状态计算每个智能体的奖励
        rewards_dict = self._rewards(tuple(mutable_action))
        rewards = tuple(rewards_dict[f"agent_{i}"] for i in range(len(self.controlled_vehicles)))

        # 4. 增加决策步计数器
        self.steps += 1

        # 5. 检查回合是否因为任务内在逻辑而终止（成功或失败）
        terminateds = self._is_terminated()
        # 6. 检查回合是否因为外部限制而截断（超时）
        truncateds = self._is_truncated()

        # 7. 收集用于调试的辅助信息
        info = self._info(obs, tuple(mutable_action))

        return obs, rewards, terminateds, truncateds, info

    def _simulate(self, action: Tuple[int, ...] | None = None) -> None:
        """
        重写父类的 _simulate 方法，以在其中加入我们自定义的数据记录逻辑。
        这保证了我们的数据记录与物理模拟的每一步都严格同步。
        """
        # 获取配置参数
        sim_freq = self.config["simulation_frequency"]  # 物理模拟频率 (e.g., 15 Hz)
        pol_freq = self.config["policy_frequency"]  # 决策频率 (e.g., 1 Hz)
        dt = 1 / sim_freq  # 每个物理步的时间长度 (e.g., 1/15 s)引擎每秒钟会计算 15 次车辆的位置、速度、加速度等状态。每一次计算的时间间隔，就是物理时间步 dt = 1 / 15 ≈ 0.0667 秒。这是物理世界流逝的时间。

        # 每个决策步需要执行多少个物理步
        steps_per_action = int(sim_freq / pol_freq)

        # 循环执行多个物理步
        for _ in range(steps_per_action):
            # 1. 让 highway-env 的 road 对象执行一个物理步
            #    这一步会更新所有车辆的位置和速度
            super()._simulate(action)  # 调用父类的 _simulate 来应用动作和推进物理
            action = None  # 确保动作只在第一个物理步被应用

            # 2. 更新我们的累积时间，并将其记入历史
            #    我们不再手动 self.time += dt，因为 super()._simulate() 已经做了
            self.time_history.append(self.time)

            # 3. 记录每个智能体的速度和加速度
            for i, v in enumerate(self.controlled_vehicles):
                agent = f"agent_{i}"
                self.speed_history[agent].append(v.speed)
                # 通过后向差分法，用当前速度和上一个物理步的速度来计算瞬时加速度
                # 注意：这里我们不再需要 prev_speeds 列表，因为父类的 _simulate 已经更新了车辆状态
                # self.accel_history[agent].append(v.acceleration[0]) # 直接使用车辆内置的加速度更准确
                # 为了兼容旧版本，我们仍然手动计算
                # 假设 prev_speeds 在 _reset 中被正确初始化
                if len(self.speed_history[agent]) > 1:
                    accel = (self.speed_history[agent][-1] - self.speed_history[agent][-2]) / dt
                else:
                    accel = 0
                self.accel_history[agent].append(accel)

    def _rewards(self, action: Tuple[int, ...]) -> dict[str, float]:
        # 计算奖励函数，返回一个字典，键是 agent 名称，值是奖励
        # 参数 action: 多个智能体的动作元组，例如 (0, 1, 2)

        dt = 1.0 / float(self.config.get("simulation_frequency", 15))
        # 根据仿真频率计算单步时间间隔 dt，默认频率 15 Hz，即 dt = 1/15 秒
        # 所有奖励会乘上 dt，相当于按时间积分

        # === 参数设定 ===
        v_target = 30.0  # 目标速度（单位：m/s）
        v_low = 25.0  # 低速阈值，低于此速度会惩罚

        # 奖励权重字典 W
        W = {
            "collision": -200.0,  # 碰撞惩罚
            "offroad": -200.0,  # 离开道路边界时惩罚
            "speed": 10.0,  # 速度奖励系数（与 v_target 比例关系）
            "low_speed": -5.0,  # 低速惩罚系数
            "lane_bonus": 100.0,  # 首次成功换道奖励（一次性）
            "success": 200.0,  # 全部超越 HDV 的奖励
            "timeout": -200.0  # 超时惩罚
        }

        # === 初始化团队奖励 ===
        team_reward = 0.0
        # 最终团队奖励将在每个 agent 上共享

        # === 按车逐个计算基础奖励 ===
        for v in self.controlled_vehicles:
            r = 0.0  # 每辆车的单车奖励

            # 碰撞检查
            if getattr(v, "crashed", False):
                r += W["collision"]  # 如果发生碰撞，添加碰撞惩罚

            # 是否离开道路（部分环境中可能有 on_road 属性）
            if not getattr(v, "on_road", True):
                r += W["offroad"]  # 离开道路加惩罚

            # 速度奖励：按当前车速相对于目标速度 v_target 的比例计算
            speed = float(getattr(v, "speed", 0.0))
            r += W["speed"] * (speed / v_target)

            # 低速惩罚处理：速度低于 v_low，按比例惩罚
            if speed < v_low:
                penalty_frac = 1.0 - (speed / max(1.0, v_low))
                # penalty_frac 越大，说明速度越低
                r += W["low_speed"] * penalty_frac

            # 将单车奖励积分到团队奖励
            team_reward += dt * r

        # === Agent_0 首次换道奖励（一次性事件，存到 self 标志中避免重复触发） ===
        v0 = self.controlled_vehicles[0]
        if not getattr(self, "agent_a_lane_changed", False) and v0.lane_index[2] == 0:
            # 条件说明：
            # - 尚未触发过换道奖励
            # - 当前 lane_index 的第 3 个元素（通常是车道号）为 0，说明已经在目标车道
            team_reward += W["lane_bonus"]
            self.agent_a_lane_changed = True  # 标记为已触发

        # === 成功完成任务条件 ===
        # 条件是三辆控制车全部超越环境里的 HDV（人类驾驶车）
        hdv = next(
            (v for v in self.road.vehicles if v not in self.controlled_vehicles),
            None
        )
        if hdv is not None:
            # 检查每辆车的 x 位置是否都比 HDV 领先
            all_ahead = all(v.position[0] > hdv.position[0] for v in self.controlled_vehicles)
            if all_ahead:
                team_reward += W["success"]

        # === 超时惩罚 ===
        # 时间达到预设持续时间，或步数达到 max_steps 视为任务失败
        if self.time >= self.config["duration"] or self.steps >= self.config.get("max_steps", 2000):
            team_reward += W["timeout"]

        # === 奖励分发给每个 agent ===
        # 团队奖励在多智能体场景下对所有 agent 平分（这里是直接相同赋值）
        rewards_dict: dict[str, float] = {}
        for i in range(len(self.controlled_vehicles)):
            rewards_dict[f"agent_{i}"] = team_reward

        return rewards_dict

    def _is_terminated(self) -> Tuple[bool, ...]:
        """
        检查回合是否因为任务内在逻辑而结束（成功或失败）。
        在协作任务中，通常一个智能体的成败决定了整个团队的成败。
        """
        # 失败条件：任何车辆碰撞或驶出路面
        failure = any(v.crashed or not v.on_road for v in self.controlled_vehicles)

        # 成功条件：所有智能体都已超越 HDV
        success = False
        # 用一种非常健壮的方式找到 HDV
        hdv = next((v for v in self.road.vehicles if v not in self.controlled_vehicles), None)
        if hdv is not None:
            if all(v.position[0] > hdv.position[0] for v in self.controlled_vehicles):
                success = True

        done = failure or success
        # 将这个统一的结束信号广播给所有智能体
        return tuple(done for _ in self.controlled_vehicles)



# # In MARL.py (添加绘制图像之前)
#
# from __future__ import annotations
# import numpy as np
# from typing import Tuple, Dict
# from highway_env.envs.common.abstract import AbstractEnv
# from highway_env.road.lane import StraightLane, LineType
# from highway_env.road.road import Road, RoadNetwork
# from highway_env.vehicle.kinematics import Vehicle
# import time
# Observation = np.ndarray
#
#
# class MARLEnv(AbstractEnv):
#     metadata = {"render_modes": ["human", "rgb_array"], "name": "MARLEnv-v0"}
#
#     @classmethod
#     def default_config(cls) -> dict:
#         # ... (这部分完全不变) ...
#         config = super().default_config()
#         config.update({
#             "observation": {"type": "MultiAgentObservation", "observation_config": {"type": "Kinematics"}},
#             "action": {"type": "MultiAgentAction", "action_config": {"type": "DiscreteMetaAction"}},
#             "lanes_count": 2,
#             "vehicles_count": 1,
#             "controlled_vehicles": 3,
#             "duration": 40,
#             "offroad_terminal": True,
#             "collision_reward": -800,
#             "high_speed_reward": 1,
#             "low_speed_penalty": -2,
#             "lane_change_reward": 20000,
#             "timeout_penalty": -500,
#             "low_speed_threshold": 15,
#             "high_speed_threshold": 25,
#             "max_steps": 2000,  # --- NEW: 最大步数超时终止条件
#             "screen_width": 1200,  # 画面更宽
#             "screen_height": 300,  # 高度适中
#         })
#         return config
#
#     def _reset(self) -> None:
#         self._create_road()
#         self._create_vehicles()
#         self.agent_a_lane_changed = False
#         self.steps = 0   # --- NEW: 步数计数器simulation_frequency
#
#     def _create_road(self) -> None:
#         # ... (这部分完全不变) ...
#         net = RoadNetwork()
#         lane_width = 4.0
#         road_length = 1000.0
#         lane_0_start, lane_0_end = np.array([0.0, 0.0]), np.array([road_length, 0.0])
#         lane_1_start, lane_1_end = np.array([0.0, lane_width]), np.array([road_length, lane_width])
#         net.add_lane("a", "b", StraightLane(lane_0_start, lane_0_end, width=lane_width,
#                                             line_types=(LineType.STRIPED, LineType.CONTINUOUS_LINE)))
#         net.add_lane("a", "b", StraightLane(lane_1_start, lane_1_end, width=lane_width,
#                                             line_types=(LineType.STRIPED, LineType.STRIPED)))
#         self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
#
#     def _create_vehicles(self) -> None:
#         # ... (这部分完全不变) ...
#         self.controlled_vehicles = []
#         vehicle_b = self.action_type.vehicle_class(self.road, self.road.network.get_lane(("a", "b", 0)).position(20, 0),
#                                                    speed=20)
#         self.controlled_vehicles.append(vehicle_b)
#         vehicle_c = self.action_type.vehicle_class(self.road, self.road.network.get_lane(("a", "b", 0)).position(80, 0),
#                                                    speed=20)
#         self.controlled_vehicles.append(vehicle_c)
#         pos_b_x, pos_c_x = vehicle_b.position[0], vehicle_c.position[0]
#         random_x = self.np_random.uniform(pos_b_x - 40, pos_c_x + 40)
#         vehicle_a = self.action_type.vehicle_class(self.road,
#                                                    self.road.network.get_lane(("a", "b", 1)).position(random_x, 0),
#                                                    speed=20)
#         self.controlled_vehicles.insert(0, vehicle_a)
#         hdv = Vehicle(self.road, self.road.network.get_lane(("a", "b", 1)).position(vehicle_a.position[0] + 30, 0),
#                       speed=15)
#         self.road.vehicles.clear()
#         self.road.vehicles.extend(self.controlled_vehicles)
#         self.road.vehicles.append(hdv)
#
#     # --- MODIFIED: 这是最核心、最根本的修复 ---
#     def step(self, action: Tuple[int, ...]) -> Tuple[
#         Observation, Tuple[float, ...], Tuple[bool, ...], Tuple[bool, ...], Dict]:
#
#         # 1. 动作屏蔽
#         mutable_action = list(action)
#         vehicle_a = self.controlled_vehicles[0]
#         if self.agent_a_lane_changed or vehicle_a.lane_index[2] == 0:
#             self.agent_a_lane_changed = True
#             if mutable_action[0] in [2, 3]:
#                 mutable_action[0] = 0
#         for i in range(1, 3):
#             if mutable_action[i] in [2, 3]:
#                 mutable_action[i] = 0
#
#         # 2. 应用动作并推进物理模拟
#         self._simulate(tuple(mutable_action))
#
#         # 3. 收集多智能体观测
#         obs = self.observation_type.observe()
#
#         # 4. 奖励
#         rewards_dict = self._rewards(tuple(mutable_action))
#         rewards = tuple(rewards_dict[f"agent_{i}"] for i in range(len(self.controlled_vehicles)))
#
#         # 5. 步数
#         self.steps += 1
#
#         # 6. 终止与截断
#         terminateds = self._is_terminated()
#         truncateds = self._is_truncated()
#
#         # 7. info
#         info = self._info(obs, tuple(mutable_action))
#
#         return obs, rewards, terminateds, truncateds, info
#
#     # def step(self, action: Tuple[int, ...]) -> Tuple[
#     #     Observation, Tuple[float, ...], Tuple[bool, ...], Tuple[bool, ...], Dict]:
#     #     """
#     #     完整地、正确地实现多智能体的 step 逻辑，不再调用 super().step()。
#     #     """
#     #     # 1. 动作屏蔽
#     #     mutable_action = list(action)
#     #     vehicle_a = self.controlled_vehicles[0]
#     #     if self.agent_a_lane_changed or vehicle_a.lane_index[2] == 0:
#     #         self.agent_a_lane_changed = True
#     #         if mutable_action[0] in [2, 3]: mutable_action[0] = 0
#     #     for i in range(1, 3):
#     #         if mutable_action[i] in [2, 3]: mutable_action[i] = 0
#     #
#     #     # 2. 应用动作并推进物理模拟
#     #     self._simulate(tuple(mutable_action))
#     #
#     #     # 3. 收集多智能体观测
#     #     obs = self.observation_type.observe()
#     #
#     #     # 4. 计算多智能体奖励 (调用我们自己的 _rewards 方法)
#     #     rewards_dict = self._rewards(tuple(mutable_action))
#     #     # 转换成 PettingZoo 包装器期望的元组格式
#     #     rewards = tuple(rewards_dict[f"agent_{i}"] for i in range(len(self.controlled_vehicles)))
#     #
#     #     # 5. 更新计数器
#     #     self.steps += 1   # --- NEW: 累加步数
#     #
#     #
#     #     # 6. 检查终止和截断信号 (调用我们自己的方法)
#     #     terminateds = self._is_terminated()
#     #     truncateds = self._is_truncated()
#     #
#     #     # 7. 收集信息
#     #     info = self._info(obs, tuple(mutable_action))
#     #
#     #     # 8. 更新时间
#     #     self.time += 1 / self.config["simulation_frequency"]
#     #
#     #     # 9. 计算加速度
#     #     if not hasattr(self, "speed_history"):
#     #         self.speed_history = {f"agent_{i}": [] for i in range(len(self.controlled_vehicles))}
#     #         self.accel_history = {f"agent_{i}": [] for i in range(len(self.controlled_vehicles))}
#     #         self.prev_speeds = [v.speed for v in self.controlled_vehicles]
#     #
#     #     dt = 1 / self.config["simulation_frequency"]
#     #
#     #     for i, v in enumerate(self.controlled_vehicles):
#     #         agent = f"agent_{i}"
#     #         # 速度
#     #         self.speed_history[agent].append(v.speed)
#     #         # 加速度
#     #         accel = (v.speed - self.prev_speeds[i]) / dt
#     #         self.accel_history[agent].append(accel)
#     #         self.prev_speeds[i] = v.speed
#     #env_inst.time
#     #     return obs, rewards, terminateds, truncateds, info
#
# #  --------------------------------------------------------
#     def _simulate(self, action: Tuple[int, ...] | None = None) -> None:
#         """推进物理仿真，并记录速度/加速度/时间，同时捕获渲染帧"""
#         sim_freq = self.config["simulation_frequency"]
#         pol_freq = self.config["policy_frequency"]
#         dt = 1 / sim_freq
#         steps_per_action = int(sim_freq / pol_freq)
#
#         # 初始化历史记录
#         if not hasattr(self, "time"):
#             self.time = 0.0
#         if not hasattr(self, "time_history"):
#             self.time_history = []
#             self.speed_history = {f"agent_{i}": [] for i in range(len(self.controlled_vehicles))}
#             self.accel_history = {f"agent_{i}": [] for i in range(len(self.controlled_vehicles))}
#             self.prev_speeds = [v.speed for v in self.controlled_vehicles]
#         if not hasattr(self, "_captured_frames"):
#             self._captured_frames = []
#
#         for _ in range(steps_per_action):
#             # 物理推进
#             self.road.step(dt)
#
#             # 更新时间
#             self.time += dt
#             self.time_history.append(self.time)
#
#             # 记录速度和加速度
#             for i, v in enumerate(self.controlled_vehicles):
#                 agent = f"agent_{i}"
#                 self.speed_history[agent].append(v.speed)
#                 accel = (v.speed - self.prev_speeds[i]) / dt
#                 self.accel_history[agent].append(accel)
#                 self.prev_speeds[i] = v.speed
#
#             # 捕获渲染帧（每个物理步都渲染）
#             if hasattr(self, "render") and callable(self.render):
#                 frame = self.render()
#                 if frame is not None:
#                     self._captured_frames.append(frame)
#
#     #     def _simulate(self, action: Tuple[int, ...] | None = None) -> None:
# #         """推进物理仿真，并记录速度/加速度/时间，保证和视频帧严格对应"""
# #         sim_freq = self.config["simulation_frequency"]
# #         pol_freq = self.config["policy_frequency"]
# #         dt = 1 / sim_freq
# #         steps_per_action = int(sim_freq / pol_freq)
# #
# #         # 初始化历史记录
# #         if not hasattr(self, "time"):
# #             self.time = 0.0
# #         if not hasattr(self, "time_history"):
# #             self.time_history = []
# #             self.speed_history = {f"agent_{i}": [] for i in range(len(self.controlled_vehicles))}
# #             self.accel_history = {f"agent_{i}": [] for i in range(len(self.controlled_vehicles))}
# #             self.prev_speeds = [v.speed for v in self.controlled_vehicles]
# #
# #         for _ in range(steps_per_action):
# #             # 物理推进
# #             self.road.step(dt)
# #
# #             # 更新时间
# #             self.time += dt
# #             self.time_history.append(self.time)
# #
# #             # 记录速度和加速度
# #             for i, v in enumerate(self.controlled_vehicles):
# #                 agent = f"agent_{i}"
# #                 self.speed_history[agent].append(v.speed)
# #                 accel = (v.speed - self.prev_speeds[i]) / dt
# #                 self.accel_history[agent].append(accel)
# #                 self.prev_speeds[i] = v.speed
# # --------------------------------------------------------
#     def _rewards(self, action: Tuple[int, ...]) -> dict[str, float]:
#         # ... (这部分完全不变) ...
#         rewards_dict = {}
#         for i, vehicle in enumerate(self.controlled_vehicles):
#             agent_id = f"agent_{i}"
#             reward = 0
#             if vehicle.crashed: reward += self.config["collision_reward"]
#             if vehicle.speed > self.config["high_speed_threshold"]: reward += self.config["high_speed_reward"]
#             if vehicle.speed < self.config["low_speed_threshold"]: reward += self.config["low_speed_penalty"]
#             rewards_dict[agent_id] = reward
#         vehicle_a = self.controlled_vehicles[0]
#         if not self.agent_a_lane_changed and vehicle_a.lane_index[2] == 0:
#             rewards_dict["agent_0"] += self.config["lane_change_reward"]
#             self.agent_a_lane_changed = True
#         return rewards_dict
#
#     def _is_terminated(self) -> Tuple[bool, ...]:
#         # 1. 原始条件：任意车撞毁或离开道路
#         done = any(v.crashed or not v.on_road for v in self.controlled_vehicles)
#
#         # 2. 新增条件：三辆智能体车的横坐标都超过 HDV
#         # 找到 HDV（非受控车辆）
#         hdv = next((v for v in self.road.vehicles if v not in self.controlled_vehicles), None)
#         if hdv is not None:
#             all_ahead = all(v.position[0] > hdv.position[0] for v in self.controlled_vehicles)
#             if all_ahead:
#                 done = True
#
#         return tuple(done for _ in self.controlled_vehicles)
#
#     def _is_truncated(self) -> Tuple[bool, ...]:
#         time_trunc = self.time >= self.config["duration"]
#         step_trunc = self.steps >= self.config["max_steps"]
#         return tuple(time_trunc or step_trunc for _ in self.controlled_vehicles)
#
#     def _info(self, obs: tuple, action: tuple | None) -> dict:
#
#         if action is None:
#             return {f"agent_{i}": {} for i in range(len(self.controlled_vehicles))}
#         return {f"agent_{i}": {} for i in range(len(action))}
