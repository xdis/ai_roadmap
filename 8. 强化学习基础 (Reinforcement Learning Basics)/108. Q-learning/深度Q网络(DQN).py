import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from collections import deque
import random

# 定义深度Q网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# DQN智能体
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=64, update_target_every=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.training_step = 0
        
        # Q网络
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax(1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # 从回放缓冲区采样
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # 提取数据
        states = torch.FloatTensor(batch[0])
        actions = torch.LongTensor(batch[1]).unsqueeze(1)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
        next_states = torch.FloatTensor(batch[3])
        dones = torch.FloatTensor(batch[4]).unsqueeze(1)
        
        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions)
        
        # 计算下一状态的最大Q值
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # 计算损失并优化
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.training_step += 1
        if self.training_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

# 训练DQN
def train_dqn(env_name='CartPole-v1', episodes=500, render=False):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 更新网络
            loss = agent.update()
            
            state = next_state
            episode_reward += reward
        
        rewards.append(episode_reward)
        
        # 打印训练进度
        if (episode+1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, rewards

# 训练DQN
agent, rewards = train_dqn()

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title('DQN Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()