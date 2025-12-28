import torch
import torch.nn.functional as F
import collections
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.fc1.bias, 0.1)

        self.fc_out = torch.nn.Linear(hidden_dim, action_dim)
        torch.nn.init.kaiming_uniform_(self.fc_out.weight, nonlinearity='tanh')
        torch.nn.init.constant_(self.fc_out.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = F.tanh(self.fc_out(x))
        return action

class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.fc1.bias, 0.1)

        self.fc_out = torch.nn.Linear(hidden_dim, 1)
        torch.nn.init.uniform_(self.fc_out.weight, -0.003, 0.003)
        torch.nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        q = self.fc_out(x)
        return q

class DDPG:
    """ DDPG算法 """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma,
                 actor_lr, critic_lr, tau, gamma, device, is_train):
        # 构建 DDPG 网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 构建优化器
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=actor_lr, momentum=0.9)
        self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=critic_lr, momentum=0.9)

        # 其他参数
        self.action_bound = action_bound
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.is_train = is_train
        self.noise_decay = 0.99
        self.std = 0.1
        self.train_num = 0
        self.min_sigma = 0.01
        self.initial_sigma = sigma

        # 经验池
        self.replay_buffer = ReplayBuffer(capacity=50000)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().squeeze(0).numpy()

        # 如果是训练，则给动作添加噪声，增加探索
        if self.is_train:
            noise = np.random.randn() * self.sigma
            action += noise
            action = np.clip(action, -1, 1)

        return action * self.action_bound

    def decay_noise(self):
        # 噪声衰减
        self.sigma = max(self.min_sigma, self.sigma * self.noise_decay)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if (self.train_num + 1) % 10 == 0:
            self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
            self.soft_update(self.critic, self.target_critic)  # 软更新价值网络