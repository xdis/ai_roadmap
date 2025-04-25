import numpy as np
import matplotlib.pyplot as plt
import random

class GridWorld:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.start = (0, 0)  # 左上角
        self.goal = (width-1, height-1)  # 右下角
        
        # 障碍物位置
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        
    def get_state_space(self):
        return self.width * self.height
    
    def get_action_space(self):
        # 上、右、下、左
        return 4
    
    def get_next_state(self, state, action):
        x, y = state
        
        # 执行动作
        if action == 0:  # 上
            y = max(0, y - 1)
        elif action == 1:  # 右
            x = min(self.width - 1, x + 1)
        elif action == 2:  # 下
            y = min(self.height - 1, y + 1)
        elif action == 3:  # 左
            x = max(0, x - 1)
            
        # 检查是否碰到障碍物
        if (x, y) in self.obstacles:
            return state  # 保持原位
        
        return (x, y)
    
    def get_reward(self, state, action, next_state):
        if next_state == self.goal:
            return 1.0  # 到达目标
        elif next_state in self.obstacles:
            return -1.0  # 碰到障碍物
        elif next_state == state:  # 尝试移动到障碍物
            return -0.5  # 轻微惩罚
        else:
            return -0.04  # 每一步的小惩罚，鼓励找到最短路径
    
    def is_terminal(self, state):
        return state == self.goal

def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=1000):
    """
    Q-learning算法
    
    参数:
    - env: 环境
    - alpha: 学习率
    - gamma: 折扣因子
    - epsilon: ε-贪心策略中的探索概率
    - episodes: 训练回合数
    
    返回:
    - Q表: 状态-动作价值函数
    """
    # 初始化Q表
    state_space = env.get_state_space()
    action_space = env.get_action_space()
    Q = np.zeros((env.width, env.height, action_space))
    
    # 记录每个回合的总奖励
    rewards = []
    
    for episode in range(episodes):
        # 初始化状态
        state = env.start
        total_reward = 0
        done = False
        
        while not done:
            # ε-贪心策略选择动作
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, action_space - 1)  # 探索
            else:
                action = np.argmax(Q[state[0], state[1]])  # 利用
            
            # 执行动作，观察新状态和奖励
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(state, action, next_state)
            total_reward += reward
            
            # Q-learning更新
            best_next_action = np.argmax(Q[next_state[0], next_state[1]])
            td_target = reward + gamma * Q[next_state[0], next_state[1], best_next_action]
            td_error = td_target - Q[state[0], state[1], action]
            Q[state[0], state[1], action] += alpha * td_error
            
            # 转移到下一状态
            state = next_state
            done = env.is_terminal(state)
        
        rewards.append(total_reward)
        
        # 简单的进度打印
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {np.mean(rewards[-100:]):.2f}")
    
    return Q, rewards

# 运行Q-learning
env = GridWorld()
q_table, rewards = q_learning(env)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rewards) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Q-learning Learning Curve')
plt.grid(True)
plt.show()

# 可视化学到的策略
def visualize_policy(env, q_table):
    policy = np.zeros((env.height, env.width), dtype=str)
    arrows = ['↑', '→', '↓', '←']
    
    for i in range(env.width):
        for j in range(env.height):
            if (i, j) == env.goal:
                policy[j, i] = 'G'
            elif (i, j) in env.obstacles:
                policy[j, i] = 'X'
            else:
                best_action = np.argmax(q_table[i, j])
                policy[j, i] = arrows[best_action]
    
    return policy

policy = visualize_policy(env, q_table)
print("学到的策略:")
print(policy)