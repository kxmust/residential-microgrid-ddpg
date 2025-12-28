from DDPG import DDPG
from residential_env import MicrogridEnv
from get_data import rm_data
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
np.random.seed(0)
torch.manual_seed(0)

# 构建微电网环境
env = MicrogridEnv(rm_data, episode_length=48)

control_agent = DDPG(
    state_dim = 4,    # 输入四个状态，[负载, 光伏, 电价, SOC]
    action_dim = 1,   # 输出电池充放电功率
    hidden_dim = 64,
    action_bound = env.action_bound,
    sigma = 0.3,
    actor_lr = 3e-4,
    critic_lr = 4e-4,
    gamma = 0.99,
    tau = 0.8,
    is_train = 1,
    device = device,
)

def train_agent(env, agent, num_episodes, minimal_size, batch_size, is_plt = False):
    return_list = []
    env.is_train = True
    agent.is_train = True

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(env._normalize(state)).item()
                    next_state, reward, done, _, _ = env.step(action)
                    agent.replay_buffer.add(env._normalize(state), action/env.action_bound,
                                                    reward, env._normalize(next_state), done)
                    state = next_state
                    episode_return += reward
                    if agent.replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = agent.replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)

                agent.train_num += 1
                if (agent.train_num + 1) % 25 == 0:
                    agent.decay_noise()

                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                                      'return': '%.3f' % np.mean(return_list[-10:]),
                                      'sigma': '%.3f' % agent.sigma})


                pbar.update(1)
    torch.save(agent.actor.state_dict(), 'Model_Save/actor_weights.pth')
    if is_plt:
        # 平滑训练曲线
        def moving_average(a, window_size):
            cumulative_sum = np.cumsum(np.insert(a, 0, 0))
            middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
            r = np.arange(1, window_size - 1, 2)
            begin = np.cumsum(a[:window_size - 1])[::2] / r
            end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
            return np.concatenate((begin, middle, end))

        episodes_list = list(range(len(return_list)))
        mv_return = moving_average(return_list, 9)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DDPG')
        plt.savefig('plots/DDPG_train.png')
        plt.show()

    return return_list

def test_microgrid(env, agent, test_time, plot=True):
    state, _ = env.reset()
    done = False
    total_cost = 0.0

    # 记录各变量
    records = {
        't': [],
        'load': [],
        'pv': [],
        'ug_price': [],
        'soc': [],
        'battery_power': [],
        'grid_power': [],
        'reward': [],
        'cost': [],
    }

    while not done:
        # 策略输出动作
        action = agent.take_action(env._normalize(state)).item()
        next_state, reward, done, _, _ = env.step(action)

        # 解析状态
        load, pv, ug_price, soc = state
        power_battery = env._clip_battery_action(action, soc)

        Pe = load - pv - power_battery  # 电网交互功率

        if Pe >= 0:
            cost = Pe * ug_price
        else:
            cost = Pe * ug_price * 0.7

        total_cost += cost

        # 保存记录
        records['t'].append(env.len_t)
        records['load'].append(load)
        records['pv'].append(pv)
        records['ug_price'].append(ug_price)
        records['soc'].append(soc)
        records['battery_power'].append(power_battery)
        records['grid_power'].append(Pe)
        records['reward'].append(reward)
        records['cost'].append(cost)

        state = next_state

        if env.len_t >= test_time-1:
            done = True

    # 转成 numpy 数组
    for k in records:
        records[k] = np.array(records[k])

    print(f"\n✅ 测试完成，总电费成本为：{total_cost:.2f} 元, SOC均值为:{np.mean(records['soc']):.2f}")

    # 绘图部分
    if plot:
        t = records['t']
        plt.figure(figsize=(12, 8))

        # (1) 功率流：负载、光伏、电池功率、电网功率
        plt.subplot(3, 1, 1)
        plt.plot(t, records['load'], label='Load(kW)', c='c', linewidth=2)
        plt.plot(t, records['pv'], label='PV(kW)', c='g', linewidth=2)
        plt.plot(t, records['ug_price'], label='UG_Price(Cents/kW)', color='tab:orange', linewidth=2)
        plt.bar(t, records['battery_power'], label='Battery_Power(kW)', color='tab:blue', alpha=0.5)
        plt.ylabel("kW")
        plt.title("RM Scheduling Process")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)

        # (2) SOC变化
        plt.subplot(3, 1, 2)
        plt.plot(t, records['soc'], color='tab:green', linewidth=2, label='SOC')
        plt.ylabel("SOC")
        plt.title("The change of SOC")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)

        # (3) 电价与成本
        plt.subplot(3, 1, 3)
        plt.plot(t, records['cost'], label='Cost(Cents)', color='tab:red', linewidth=1.5, linestyle='--')
        plt.ylabel("Price")
        plt.title("The change of COST")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return total_cost, records

if __name__ == "__main__":
    is_train = 0
    is_test = 1
    if is_train:
        train_agent(env, control_agent, num_episodes=10000, minimal_size=2000, batch_size=64, is_plt = True)
    if is_test:
        env.is_train = False
        control_agent.is_train = False
        control_agent.actor.load_state_dict(torch.load('Model_Save/actor_weights.pth'))
        test_microgrid(env, control_agent, test_time=48, plot=True)