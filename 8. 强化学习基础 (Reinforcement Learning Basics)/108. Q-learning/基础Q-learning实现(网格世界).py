import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm

class GridWorldEnv:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.start = (0, 0)  # 左上角
        self.goal = (width-1, height-1)  # 右下角
        self.obstacles = [(1, 1), (2, 2), (3, 1)]  # 障碍物位置
        
        # 动作空间: 上(0), 右(1), 下(2), 左(3)
        self.action_space = 4
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
    def reset(self):
        self.current_pos = self.start
        return self.current_pos
    
    def step(self, action):
        # 计算下一个位置
        x, y = self.current_pos
        dx, dy = self.actions[action]
        next_x = max(0, min(self.width-1, x + dx))
        next_y = max(0, min(self.height-1, y + dy))
        next_pos = (next_x, next_y)
        
        # 检查是否碰到障碍物
        if next_pos in self.obstacles:
            next_pos = (x, y)  # 保持原位
        
        # 更新位置
        self.current_pos = next_pos
        
        # 计算奖励
        if next_pos == self.goal:
            reward = 1.0  # 到达目标
            done = True
        elif next_pos in self.obstacles:
            reward = -1.0  # 尝试进入障碍物
            done = False
        elif next_pos == (x, y):  # 撞墙
            reward = -0.5
            done = False
        else:
            reward = -0.04  # 移动惩罚，鼓励寻找最短路径
            done = False
        
        return next_pos, reward, done, {}

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.epsilon_min = 0.01  # 最小探索率
        
        # 初始化Q表
        self.q_table = {}
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # 探索
        else:
            # 找出最佳动作
            q_values = [self.get_q_value(state, a) for a in range(self.action_size)]
            return np.argmax(q_values)  # 利用
    
    def learn(self, state, action, reward, next_state, done):
        # 获取当前Q值
        current_q = self.get_q_value(state, action)
        
        # 计算TD目标
        if done:
            target_q = reward
        else:
            # 下一状态的最大Q值
            next_max_q = max([self.get_q_value(next_state, a) for a in range(self.action_size)])
            target_q = reward + self.gamma * next_max_q
        
        # 更新Q值
        self.q_table[(state, action)] = current_q + self.alpha * (target_q - current_q)
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_q_learning(env, agent, episodes=500):
    rewards_per_episode = []
    steps_per_episode = []
    
    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 学习
            agent.learn(state, action, reward, next_state, done)
            
            total_reward += reward
            step += 1
            state = next_state
        
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step)
        
    return rewards_per_episode, steps_per_episode

def visualize_q_values(env, agent):
    # 创建一个表示Q值的网格
    grid = np.zeros((env.height, env.width, 4))  # 4个动作
    
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            for action in range(env.action_space):
                grid[y, x, action] = agent.get_q_value(state, action)
    
    # 可视化每个状态的最佳动作
    policy_grid = np.zeros((env.height, env.width), dtype=str)
    action_symbols = ['↑', '→', '↓', '←']
    
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            if state == env.goal:
                policy_grid[y, x] = 'G'  # 目标
            elif state in env.obstacles:
                policy_grid[y, x] = 'X'  # 障碍物
            else:
                # 找出最佳动作
                best_action = np.argmax([agent.get_q_value(state, a) for a in range(env.action_space)])
                policy_grid[y, x] = action_symbols[best_action]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.max(grid, axis=2), annot=policy_grid, fmt='', cmap='YlGnBu')
    plt.title('Q Values and Optimal Policy')
    plt.show()

def plot_learning_curve(rewards, steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 绘制奖励曲线
    ax1.plot(rewards)
    ax1.set_title('Rewards per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # 绘制步数曲线
    ax2.plot(steps)
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    
    plt.tight_layout()
    plt.show()

# 创建环境和智能体
env = GridWorldEnv()
agent = QLearningAgent(state_size=(env.width, env.height), action_size=env.action_space)

# 训练
rewards, steps = train_q_learning(env, agent, episodes=500)

# 可视化结果
plot_learning_curve(rewards, steps)
visualize_q_values(env, agent)