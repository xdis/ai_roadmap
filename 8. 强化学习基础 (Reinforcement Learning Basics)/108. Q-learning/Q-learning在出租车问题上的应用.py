import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

# 创建环境
env = gym.make('Taxi-v3')

# Q-learning参数
alpha = 0.1  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 初始探索率
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000

# 初始化Q表
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# 记录训练过程
rewards_per_episode = []
steps_per_episode = []

# 训练过程
for episode in tqdm(range(episodes), desc="Training"):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        # 选择动作 (ε-贪心)
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # 探索
        else:
            action = np.argmax(q_table[state])  # 利用
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # Q-learning更新
        old_q = q_table[state, action]
        next_max_q = np.max(q_table[next_state])
        
        if done:
            target_q = reward
        else:
            target_q = reward + gamma * next_max_q
        
        q_table[state, action] = old_q + alpha * (target_q - old_q)
        
        state = next_state
        total_reward += reward
        steps += 1
    
    # 记录本回合结果
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)
    
    # 衰减探索率
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# 绘制学习曲线
def moving_average(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(moving_average(rewards_per_episode, 10))
plt.title('Average Reward (Window=10)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(moving_average(steps_per_episode, 10))
plt.title('Average Steps (Window=10)')
plt.xlabel('Episode')
plt.ylabel('Steps')

plt.tight_layout()
plt.show()

# 测试学习到的策略
def test_policy(env, q_table, n_episodes=10):
    total_reward = 0
    total_steps = 0
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            # 选择当前状态下Q值最高的动作
            action = np.argmax(q_table[state])
            
            # 执行动作
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # 可选：显示环境
            # env.render()
        
        total_reward += episode_reward
        total_steps += episode_steps
    
    avg_reward = total_reward / n_episodes
    avg_steps = total_steps / n_episodes
    
    print(f"测试结果 - 平均奖励: {avg_reward:.2f}, 平均步数: {avg_steps:.2f}")

# 测试学习到的策略
test_policy(env, q_table)