import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Softmax输出动作概率
        return F.softmax(self.fc3(x), dim=-1)

# REINFORCE算法
class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def update(self, rewards, log_probs):
        # 计算discounted rewards
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # 标准化returns（减少方差）
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # 计算策略梯度损失
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # 更新策略网络
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

# 训练函数
def train_reinforce(env_name='CartPole-v1', episodes=1000, render=False):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCE(state_dim, action_dim)
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        rewards = []
        log_probs = []
        done = False
        
        # 单次回合交互
        while not done:
            if render:
                env.render()
            
            # 选择动作并与环境交互
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 存储奖励和log概率
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
        
        # 更新策略
        agent.update(rewards, log_probs)
        
        # 记录总奖励
        total_reward = sum(rewards)
        total_rewards.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")
    
    return agent, total_rewards

if __name__ == "__main__":
    agent, rewards = train_reinforce()
    
    # 绘制学习曲线
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE Learning Curve')
    plt.grid(True)
    plt.show()