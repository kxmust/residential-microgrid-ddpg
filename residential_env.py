import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MicrogridEnv(gym.Env):

    def __init__(self, rm_data, episode_length=24, is_train=False):
        super(MicrogridEnv, self).__init__()

        # 环境参数
        self.episode_length = episode_length
        self.data_length = rm_data.len_data

        # 微电网参数
        self.max_pv = 4.0
        self.max_load = 5.0
        self.max_ug_price = 5.2

        # 将电池的充放电功率设置为相同
        self.battery_charge_power = - 5.0
        self.battery_discharge_power = 5.0
        self.battery_capacity = 50.0    #(kWh)
        self.battery_eff = 1.0    # 电池转换效率
        self.initial_soc = 0.5

        # 奖励系数(优化目标)
        self.m1 = 0.60
        self.m2 = 0.40

        # 数据
        self.rm_data = rm_data

        # 定义状态空间: [负载, 光伏, 电价, SOC]
        # 状态空间取值的上限
        self.obs_high = np.array([self.max_load, self.max_pv,
                             self.max_ug_price, 1.0], dtype=np.float32)
        # 状态空间取值下限
        self.obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 定义强化学习的观察空间，系统会自动进行归一化处理
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        # 定义动作空间: 电池充放电功率
        self.action_space = spaces.Box(low=np.array([self.battery_charge_power], dtype=np.float32),
                                       high=np.array([self.battery_discharge_power], dtype=np.float32),
                                       dtype=np.float32)

        # 初始化状态
        self.len_t = 0      # 已经模拟的时长
        self.state = None
        self.done = False
        self.episode_start_idx = 0   # 开始训练时，随机抽取一个时刻作为初始时刻

        # 动作
        self.action_bound = self.action_space.high.item()
        self.is_train = is_train

        # 共有 720个数据，后两天用于测试
        self.test_start = 672

    # 初始化环境
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 随机选择起始点（增强数据多样性）
        if self.is_train:
            max_start = self.test_start - self.episode_length -1
            self.episode_start_idx = self.np_random.integers(0, max_start) \
                if max_start > 0 else 0
        else:
            self.episode_start_idx = self.test_start    # 测试时使用后两天的数据

        self.len_t = 0
        self.done = False

        load = self.rm_data.load(self.episode_start_idx)
        pv = self.rm_data.pv(self.episode_start_idx)
        ug_price = self.rm_data.ug_price(self.episode_start_idx)
        if self.is_train:
            soc = np.random.uniform(0.3, 0.7)
        else:
            soc = self.initial_soc

        self.state = np.array([load, pv, ug_price, soc], dtype=np.float32)
        return self.state, {}

    # 对输入数据归一化
    def _normalize(self, state):
        return (state - self.obs_low) / (self.obs_high - self.obs_low + 1e-8)

    def _clip_battery_action(self, battery_power, current_soc):
        """根据当前 SOC 裁剪电池动作，确保物理可行性"""
        # 计算 SOC 约束下的功率限制
        max_charge = (current_soc - 1.0) * self.battery_capacity / self.battery_eff
        max_discharge = current_soc * self.battery_capacity / self.battery_eff

        clipped_power = np.clip(
            battery_power,
            max(self.battery_charge_power, max_charge),  # 充电（负功率）
            min(self.battery_discharge_power, max_discharge)  # 放电（正功率）
        )
        return clipped_power

    # 环境步进，基于动作，给出下一时刻的状态
    def step(self, action):
        # 当前状态
        load, pv, ug_price, soc = self.state

        # 当前动作
        battery_power = self._clip_battery_action(action, soc)

        # 奖励
        reward = self.reward_function(self.state, action)

        # 计算下一时刻状态
        self.len_t += 1
        self.episode_start_idx += 1

        if self.is_train:
            self.done = self.len_t >= self.episode_length

        if not self.done:
            next_load = self.rm_data.load(self.episode_start_idx)
            next_pv = self.rm_data.pv(self.episode_start_idx)
            next_ug_price = self.rm_data.ug_price(self.episode_start_idx)
            next_soc = soc - battery_power * self.battery_eff / self.battery_capacity

            # SOC 安全检查（理论上不应该发生）
            if next_soc < 0. - 1e-7 or next_soc > 1. + 1e-7:
                reward -= 5.0  # 大惩罚
                next_soc = np.clip(next_soc, 0., 1.)
                self.done = True

            next_state = np.array([next_load, next_pv, next_ug_price, next_soc], dtype=np.float32)

            self.state = next_state

        return self.state, reward, self.done, False, {}

    # 奖励函数
    def reward_function(self, state, action):
        load, pv, ug_price, soc = state
        battery_power = self._clip_battery_action(action, soc)


        # 电网交互功率（正：买电；负：卖电）
        Pe = load - pv - battery_power * self.battery_eff
        if Pe >= 0:
            cost = Pe * ug_price
        else:
            cost = Pe * ug_price * 0.7  # 卖电收益较低

        profit_norm =  -cost/ (self.max_load * self.max_ug_price)

        # SOC平衡惩罚
        next_soc = soc - battery_power * self.battery_eff / self.battery_capacity
        soc_penalty = (next_soc - 0.5)**2

        reward = self.m1 * profit_norm - self.m2 * soc_penalty

        return float(reward)


