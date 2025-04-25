# Actor-Critic 方法：从零掌握这一强化学习核心技术

## 1. 基础概念理解

### Actor-Critic 方法的本质

Actor-Critic (演员-评论家) 是一种结合了策略梯度和值函数近似的强化学习方法，融合了两种主要强化学习范式的优势：
- **策略学习**（策略梯度方法）：直接学习行为策略 
- **值函数学习**（如Q-learning）：学习状态或状态-动作对的价值

在Actor-Critic框架中：
- **Actor(演员)**：负责学习和改进策略函数 π(a|s)，决定在给定状态下采取什么行动
- **Critic(评论家)**：负责学习值函数 V(s) 或 Q(s,a)，评估Actor的决策质量

这种结构的关键优势在于：Critic的评估可以**立即**指导Actor的改进，无需等待整个情节完成。

### 与其他强化学习方法的对比

| 方法 | 基本原理 | 优势 | 劣势 |
|-----|---------|------|------|
| 纯策略梯度 (REINFORCE) | 使用蒙特卡洛回报估计策略梯度 | 直接优化策略；适合连续动作 | 高方差；样本效率低；需完整情节 |
| 值函数方法 (Q-learning) | 学习状态-动作值函数 | 样本效率高；可离线学习 | 难以应用于连续动作空间；函数近似可能不稳定 |
| Actor-Critic | 结合策略学习和值函数近似 | 降低方差；可在线学习；适用于连续动作 | 实现复杂；超参数敏感；可能收敛到局部最优 |

### Actor-Critic 的工作原理

Actor-Critic方法的工作流程：

1. **Actor**基于当前策略π选择动作
2. 执行动作，观察奖励和下一个状态
3. **Critic**评估当前状态的值或当前状态-动作对的值
4. 使用Critic的评估计算TD误差或优势值
5. **Actor**使用这一评估更新策略参数
6. **Critic**更新自己的价值估计
7. 重复上述过程

### 基于优势的Actor-Critic

Actor-Critic方法常使用**优势函数** A(s,a) 进行策略更新，而不是直接使用值函数，其中：

A(s,a) = Q(s,a) - V(s)

优势函数衡量特定动作相对于该状态下平均表现的"优势"，这能显著降低方差并提高学习稳定性。

## 2. 技术细节探索

### 数学表示和理论基础

#### 策略梯度定理

Actor-Critic方法基于策略梯度定理：

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]$$

其中：
- $J(\theta)$ 是目标函数（期望累积奖励）
- $\pi_\theta$ 是参数化策略
- $Q^\pi(s,a)$ 是动作值函数

#### Actor-Critic 更新规则

1. **Critic更新**：
   
   TD(0)目标：$y_t = r_t + \gamma V(s_{t+1})$
   值函数更新：$\phi \leftarrow \phi - \alpha_\phi \nabla_\phi(V_\phi(s_t) - y_t)^2$

2. **Actor更新**：
   
   使用优势函数：$\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t)$
   
   其中 $A(s_t, a_t)$ 可以用TD误差近似：$r_t + \gamma V(s_{t+1}) - V(s_t)$

### 不同形式的Actor-Critic

1. **Q值Actor-Critic**：Critic学习Q(s,a)，优势函数 A(s,a) = Q(s,a) - V(s)

2. **TD误差Actor-Critic**：使用TD误差 $\delta = r + \gamma V(s') - V(s)$ 作为优势估计

3. **优势Actor-Critic**：显式学习优势函数或通过值函数估计

4. **自然Actor-Critic**：使用自然策略梯度代替普通梯度更新策略

### 网络架构设计

典型的Actor-Critic网络架构有两种主要设计：

1. **分离式架构**：
   - Actor和Critic使用分开的神经网络
   - 优势：角色职责清晰，调整灵活
   - 劣势：参数效率较低，可能需要更多样本

2. **共享式架构**：
   - Actor和Critic共享一部分网络层（通常是特征提取层）
   - 优势：参数效率高，有利于特征共享
   - 劣势：可能导致训练干扰，需谨慎调参

对于连续动作空间，Actor通常输出高斯分布的参数（均值和方差）或直接输出确定性动作。

## 3. 实践与实现

### 基础Actor-Critic算法实现

下面是一个基础Actor-Critic算法的PyTorch实现，用于解决CartPole问题：

```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络（策略）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络（值函数）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def evaluate(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        _, state_value = self.forward(state)
        return state_value.item()

# Actor-Critic算法
class ActorCriticAgent:
    def __init__(self, input_dim, n_actions, gamma=0.99, lr=3e-4):
        self.gamma = gamma
        self.model = ActorCritic(input_dim, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropy_weight = 0.01  # 用于鼓励探索
        
    def select_action(self, state):
        action, log_prob = self.model.act(state)
        value = self.model.evaluate(state)
        
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        return action
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def update(self):
        # 计算回报
        R = 0
        returns = []
        
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        
        # 标准化回报以减少方差
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # 将存储的值转换为张量
        log_probs = torch.cat(self.log_probs)
        values = torch.tensor(self.values)
        
        # 计算优势 (这里用简单的方式：回报 - 值函数)
        advantages = returns - values
        
        # 计算Actor（策略）损失
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # 计算Critic（值函数）损失
        critic_loss = F.mse_loss(values, returns)
        
        # 总损失
        loss = actor_loss + 0.5 * critic_loss
        
        # 梯度优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 重置记忆
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        return loss.item()

# 训练函数
def train(env_name='CartPole-v1', num_episodes=500, gamma=0.99, lr=3e-4, render=False):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    agent = ActorCriticAgent(input_dim, n_actions, gamma, lr)
    rewards = []
    running_reward = 10
    
    for i_episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_reward(reward)
            state = next_state
            episode_reward += reward
            
        # 更新策略
        loss = agent.update()
        
        # 记录奖励
        rewards.append(episode_reward)
        running_reward = 0.05 * episode_reward + 0.95 * running_reward
        
        if i_episode % 10 == 0:
            print(f'Episode {i_episode}\tLast reward: {episode_reward:.2f}\tAverage reward: {running_reward:.2f}')
        
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is now {running_reward} and the last episode runs to {episode_reward}!")
            break
            
    return rewards, agent

# 运行训练
rewards, agent = train()

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title('Actor-Critic Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.savefig('actor_critic_learning_curve.png')
plt.show()
```

### 连续动作空间的Actor-Critic

对于连续动作空间，需要修改Actor输出层以生成动作分布的参数。下面是一个处理连续动作空间的Actor-Critic示例：

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.distributions import Normal

class ContinuousActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(ContinuousActorCritic, self).__init__()
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络 - 输出高斯分布的均值和标准差
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化方法很重要，特别是对于策略网络
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.shared(x)
        
        # Actor输出
        action_mean = self.mean(features)
        action_log_std = self.log_stimport torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.distributions import Normal

class ContinuousActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(ContinuousActorCritic, self).__init__()
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络 - 输出高斯分布的均值和标准差
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化方法很重要，特别是对于策略网络
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.shared(x)
        
        # Actor输出
        action_mean = self.mean(features)
        action_log_std = self.log_st