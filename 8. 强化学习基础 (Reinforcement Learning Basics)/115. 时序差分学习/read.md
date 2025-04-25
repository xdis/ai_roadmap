# 时序差分学习：从零掌握这一强化学习核心技术

## 1. 基础概念理解

### 什么是时序差分学习？

时序差分(Temporal Difference, TD)学习是强化学习中的核心算法家族，它结合了动态规划和蒙特卡洛方法的优点。TD学习的本质是**直接从经验中学习，无需完整模型，同时通过bootstrapping(自举)实现在线、增量学习**。

最本质特征是：**TD学习使用当前估计来更新之前的估计**。

### 为什么TD学习如此重要？

TD学习之所以成为强化学习的基石，有几个关键原因：

1. **实时学习**：不需要等待回合结束即可更新价值估计
2. **样本效率**：通过bootstrapping复用先前的学习结果
3. **无模型学习**：不需要环境的完整转移概率模型
4. **处理连续任务**：能在非终止性环境中有效学习
5. **减少方差**：比蒙特卡洛方法有更低的估计方差

### TD学习与其他方法的比较

| 特性 | 动态规划 | 蒙特卡洛 | TD学习 |
|------|---------|---------|-------|
| 需要环境模型 | 是 | 否 | 否 |
| 完整回合 | 否 | 是 | 否 |
| 更新基础 | 当前估计(bootstrapping) | 实际回报 | 样本回报+当前估计 |
| 方差 | 低 | 高 | 中 |
| 偏差 | 取决于初始值 | 低 | 中 |
| 收敛速度 | 快 | 慢 | 中等 |
| 适用情境 | 已知模型 | 回合式任务 | 大多数RL任务 |

### TD学习的核心思想：bootstrapping与TD误差

TD学习的关键在于**TD误差**，它表示实际观测到的奖励与预期价值之间的差异：

**TD误差(δ)** = 奖励(R) + 折扣因子(γ) × 下一状态的估计价值(V(s')) - 当前状态的估计价值(V(s))

即：**δ = R + γV(s') - V(s)**

更新规则：V(s) ← V(s) + α × δ

这种更新方式被称为TD(0)，是最简单的TD学习形式，表示只向前看一步。

## 2. 技术细节探索

### TD学习的数学基础

TD学习基于**贝尔曼方程**，该方程描述了状态价值函数的递归关系：

V<sub>π</sub>(s) = ∑<sub>a</sub> π(a|s) ∑<sub>s',r</sub> p(s',r|s,a)[r + γV<sub>π</sub>(s')]

TD学习的核心是用样本更新来近似这个等式。

#### TD更新的形式化表示

对于状态价值函数更新：
V(s<sub>t</sub>) ← V(s<sub>t</sub>) + α[r<sub>t+1</sub> + γV(s<sub>t+1</sub>) - V(s<sub>t</sub>)]

对于动作价值函数更新：
Q(s<sub>t</sub>,a<sub>t</sub>) ← Q(s<sub>t</sub>,a<sub>t</sub>) + α[r<sub>t+1</sub> + γQ(s<sub>t+1</sub>,a<sub>t+1</sub>) - Q(s<sub>t</sub>,a<sub>t</sub>)]

### 主要TD算法类型

#### 1. TD(0)：最基本的TD算法

TD(0)是最简单的TD学习形式，用于**评估**给定策略的价值函数：

```
初始化V(s)为任意值
对于每一回合：
    初始化状态s
    对于回合中的每一步：
        选择动作a (基于给定策略π)
        执行动作a，观察奖励r和新状态s'
        V(s) ← V(s) + α[r + γV(s') - V(s)]  // TD更新
        s ← s'
```

#### 2. SARSA：on-policy TD控制算法

SARSA是一种on-policy TD控制方法，用于**同时学习**价值函数和策略：

名称来源：使用当前状态(S)、当前动作(A)、奖励(R)、下一状态(S')和下一动作(A')进行更新。

```
初始化Q(s,a)为任意值
对于每一回合：
    初始化状态s
    选择动作a (基于s的Q值，如ε-贪婪)
    对于回合中的每一步：
        执行动作a，观察奖励r和新状态s'
        选择新动作a' (基于s'的Q值，如ε-贪婪)
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]  // SARSA更新
        s ← s'
        a ← a'
```

#### 3. Q-learning：off-policy TD控制算法

Q-learning是一种off-policy TD控制方法，可以直接学习最优价值函数，不受行为策略影响：

```
初始化Q(s,a)为任意值
对于每一回合：
    初始化状态s
    对于回合中的每一步：
        选择动作a (基于s的Q值，如ε-贪婪)
        执行动作a，观察奖励r和新状态s'
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]  // Q-learning更新
        s ← s'
```

Q-learning的核心区别在于使用下一状态的最大Q值进行更新，而不管实际选择的下一个动作。

### 扩展TD方法

#### 1. n步TD方法

TD(0)只向前看一步，但可以扩展到n步：

n步TD回报：G<sub>t:t+n</sub> = r<sub>t+1</sub> + γr<sub>t+2</sub> + ... + γ<sup>n-1</sup>r<sub>t+n</sub> + γ<sup>n</sup>V(s<sub>t+n</sub>)

n步TD更新：V(s<sub>t</sub>) ← V(s<sub>t</sub>) + α[G<sub>t:t+n</sub> - V(s<sub>t</sub>)]

这提供了蒙特卡洛方法(n=∞)和TD(0)(n=1)之间的平滑过渡。

#### 2. TD(λ)与资格迹

TD(λ)结合了不同n值的n步回报，按照递减权重进行加权：

λ回报：G<sub>t</sub><sup>λ</sup> = (1-λ)∑<sub>n=1</sub><sup>∞</sup> λ<sup>n-1</sup>G<sub>t:t+n</sub>

资格迹(Eligibility Traces)是实现TD(λ)的有效方法，它跟踪每个状态的"资格"来接收学习更新：

```
初始化V(s)为任意值，e(s)=0 (所有状态的资格迹)
对于每一回合：
    初始化状态s
    对于回合中的每一步：
        选择动作a (基于给定策略π)
        执行动作a，观察奖励r和新状态s'
        δ ← r + γV(s') - V(s)  // TD误差
        e(s) ← e(s) + 1  // 访问状态s，增加其资格
        对于所有状态s：
            V(s) ← V(s) + αδe(s)  // 按资格更新所有状态
            e(s) ← γλe(s)  // 衰减所有资格迹
        s ← s'
```

## 3. 实践与实现

### 基本TD算法实现

#### TD(0)实现

```python
import numpy as np
import gym

def td0(env, num_episodes=1000, alpha=0.1, gamma=0.99):
    # 初始化值函数
    V = np.zeros(env.observation_space.n)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 选择动作 (简单策略：随机选择)
            action = env.action_space.sample()
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # TD(0)更新
            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]
            V[state] = V[state] + alpha * td_error
            
            state = next_state
    
    return V
```

#### SARSA实现

```python
def sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    # 初始化动作值函数
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # ε-贪婪策略
    def epsilon_greedy_policy(state):
        if np.random.random() < epsilon:
            return env.action_space.sample()  # 探索：随机动作
        else:
            return np.argmax(Q[state])  # 利用：最佳动作
    
    for episode in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy_policy(state)
        done = False
        
        while not done:
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 选择下一个动作
            next_action = epsilon_greedy_policy(next_state)
            
            # SARSA更新
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            td_error = td_target - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * td_error
            
            state, action = next_state, next_action
    
    return Q
```

#### Q-learning实现

```python
def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    # 初始化动作值函数
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # ε-贪婪策略选择动作
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # 探索
            else:
                action = np.argmax(Q[state])  # 利用
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning更新
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)
            td_error = td_target - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * td_error
            
            state = next_state
    
    return Q
```

### 复杂环境的TD学习

对于状态空间大或连续的环境，需要函数近似：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gym

# 简单的值函数近似网络
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# 使用函数近似的TD(0)
def td0_function_approximation(env, value_net, optimizer, num_episodes=1000, 
                              gamma=0.99, lr=0.01):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 选择动作 (简单策略：随机选择)
            action = env.action_space.sample()
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            # 计算TD目标
            with torch.no_grad():
                next_value = value_net(next_state_tensor)
                td_target = reward + gamma * next_value * (not done)
            
            # 当前状态的值估计
            value = value_net(state_tensor)
            
            # TD误差作为损失
            loss = nn.MSELoss()(value, td_target)
            
            # 更新值网络
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
```

### Q-learning函数近似实现(Deep Q-Network的简化版)

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.model(x)

def dqn_simple(env, q_net, optimizer, num_episodes=1000, gamma=0.99, 
              epsilon=0.1, batch_size=32):
    # 创建经验回放缓冲区
    replay_buffer = []
    buffer_size = 10000
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # ε-贪婪策略选择动作
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # 探索
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = q_net(state_tensor).detach().numpy()
                action = np.argmax(q_values)  # 利用
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验到回放缓冲区
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)  # 保持缓冲区大小
            
            # 从回放缓冲区采样批次训练网络
            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # 转换为张量
                states_tensor = torch.FloatTensor(np.array(states))
                actions_tensor = torch.LongTensor(actions).unsqueeze(-1)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(-1)
                next_states_tensor = torch.FloatTensor(np.array(next_states))
                dones_tensor = torch.FloatTensor(dones).unsqueeze(-1)
                
                # 当前Q值
                current_q_values = q_net(states_tensor).gather(1, actions_tensor)
                
                # 计算目标Q值
                with torch.no_grad():
                    max_next_q = q_net(next_states_tensor).max(1)[0].unsqueeze(-1)
                    target_q_values = rewards_tensor + gamma * max_next_q * (1 - dones_tensor)
                
                # 更新网络
                loss = nn.MSELoss()(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
```

## 4. 高级应用与变体

### Expected SARSA

Expected SARSA是SARSA的变体，使用下一状态动作的期望值而不是单个样本：

更新规则：
Q(s<sub>t</sub>,a<sub>t</sub>) ← Q(s<sub>t</sub>,a<sub>t</sub>) + α[r<sub>t+1</sub> + γ∑<sub>a</sub>π(a|s<sub>t+1</sub>)Q(s<sub>t+1</sub>,a) - Q(s<sub>t</sub>,a<sub>t</sub>)]

```python
def expected_sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    # 初始化动作值函数
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # ε-贪婪策略选择动作
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 计算下一状态的期望Q值
            next_q_values = Q[next_state]
            greedy_action = np.argmax(next_q_values)
            
            # ε-贪婪策略下的期望值
            expected_q = 0
            for a in range(env.action_space.n):
                if a == greedy_action:
                    expected_q += (1 - epsilon + epsilon / env.action_space.n) * next_q_values[a]
                else:
                    expected_q += (epsilon / env.action_space.n) * next_q_values[a]
            
            # Expected SARSA更新
            td_target = reward + gamma * expected_q * (not done)
            td_error = td_target - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * td_error
            
            state = next_state
    
    return Q
```

### Double Q-learning

Double Q-learning通过使用两个Q函数来解决Q-learning中的过度估计问题：

```python
def double_q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    # 初始化两个独立的动作值函数
    Q1 = np.zeros((env.observation_space.n, env.action_space.n))
    Q2 = np.zeros((env.observation_space.n, env.action_space.n))
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # ε-贪婪策略基于两个Q表的平均
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # 使用两个Q值的和来选择动作
                action = np.argmax(Q1[state] + Q2[state])
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 随机更新Q1或Q2
            if np.random.random() < 0.5:
                # 使用Q1确定最优动作，但使用Q2获取值
                best_action = np.argmax(Q1[next_state])
                td_target = reward + gamma * Q2[next_state, best_action] * (not done)
                td_error = td_target - Q1[state, action]
                Q1[state, action] = Q1[state, action] + alpha * td_error
            else:
                # 使用Q2确定最优动作，但使用Q1获取值
                best_action = np.argmax(Q2[next_state])
                td_target = reward + gamma * Q1[next_state, best_action] * (not done)
                td_error = td_target - Q2[state, action]
                Q2[state, action] = Q2[state, action] + alpha * td_error
            
            state = next_state
    
    # 返回两个Q表的平均
    return (Q1 + Q2) / 2
```

### Distributional RL

分布式强化学习 (如C51算法) 不只学习动作值的期望，而是学习整个回报分布：

```python
class CategoricalDQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms=51, vmin=-10, vmax=10):
        super(CategoricalDQN, self).__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.delta = (vmax - vmin) / (num_atoms - 1)
        self.support = torch.linspace(vmin, vmax, num_atoms)
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * num_atoms)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        logits = self.network(x).view(batch_size, self.action_dim, self.num_atoms)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_qvalues(self, x):
        probs = self.forward(x)
        q_values = torch.sum(probs * self.support.expand_as(probs), dim=2)
        return q_values
```

### 优先级经验回放

优先级经验回放通过TD误差的大小来确定采样重要性：

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
    
    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer)), None, None
        
        priorities = self.priorities[:len(self.buffer)]
        
        # 计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
```

### TD3（Twin Delayed DDPG）

TD3是用于连续动作空间的算法，解决了过度估计问题：

核心特点：
1. 使用两个Q网络并取最小值来减少过度估计
2. 延迟更新策略网络减少方差
3. 在目标策略中添加噪声以平滑值估计

```python
def update_td3(policy_net, q_net1, q_net2, target_policy, target_q1, target_q2, 
               policy_opt, q_opt, replay_buffer, batch_size, gamma, tau, policy_delay):
    # 从经验回放中采样
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = batch
    
    # 将数据转换为张量
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards).unsqueeze(-1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(-1)
    
    # 计算目标Q值
    with torch.no_grad():
        # 在目标策略中加入噪声并裁剪
        noise = torch.randn_like(actions) * 0.2
        noise = torch.clamp(noise, -0.5, 0.5)
        
        next_actions = target_policy(next_states) + noise
        next_actions = torch.clamp(next_actions, -1, 1)
        
        # 使用两个目标网络并取最小值
        q1_next = target_q1(next_states, next_actions)
        q2_next = target_q2(next_states, next_actions)
        q_next = torch.min(q1_next, q2_next)
        
        q_target = rewards + gamma * (1 - dones) * q_next
    
    # 更新Q网络
    q1_current = q_net1(states, actions)
    q2_current = q_net2(states, actions)
    
    q1_loss = F.mse_loss(q1_current, q_target)
    q2_loss = F.mse_loss(q2_current, q_target)
    q_loss = q1_loss + q2_loss
    
    q_opt.zero_grad()
    q_loss.backward()
    q_opt.step()
    
    # 延迟更新策略网络
    if global_step % policy_delay == 0:
        # 通过最大化Q值更新策略
        policy_loss = -q_net1(states, policy_net(states)).mean()
        
        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()
        
        # 软更新目标网络
        for param, target_param in zip(policy_net.parameters(), target_policy.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(q_net1.parameters(), target_q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(q_net2.parameters(), target_q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

### 实际应用案例

#### 案例一：机器人导航任务

```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gym.envs.registration import register

# 注册一个简单的导航环境
register(
    id='RobotNavigation-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

env = gym.make('RobotNavigation-v0')

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# 训练机器人导航
def train_robot_navigation():
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    q_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    
    replay_buffer = []
    buffer_size = 10000
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    num_episodes = 500
    reward_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # ε-贪婪策略
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = q_net(state_tensor).detach().numpy()
                action = np.argmax(q_values)
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # 存储经验
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)
            
            # 从回放缓冲区采样批次训练网络
            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions).unsqueeze(-1)
                rewards = torch.FloatTensor(rewards).unsqueeze(-1)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones).unsqueeze(-1)
                
                # 当前Q值
                current_q = q_net(states).gather(1, actions)
                
                # 目标Q值
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0].unsqueeze(-1)
                    target_q = rewards + gamma * next_q * (1 - dones)
                
                # 计算损失并更新
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
            
        # 更新目标网络
        if episode % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())
        
        # 衰减探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        reward_history.append(total_reward)
        
        if episode % 20 == 0:
            avg_reward = np.mean(reward_history[-20:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
    
    return q_net, reward_history
```

#### 案例二：库存管理系统

```python
class InventoryEnv:
    def __init__(self, max_inventory=20, max_demand=10):
        self.max_inventory = max_inventory
        self.max_demand = max_demand
        self.state = 0  # 当前库存
        
    def reset(self):
        self.state = 5  # 初始库存
        return self.state
    
    def step(self, action):
        """
        动作: 订购的物品数量
        """
        # 确保不超过最大库存
        new_inventory = min(self.state + action, self.max_inventory)
        
        # 成本: 订购成本 + 持有成本
        ordering_cost = 2 * action
        holding_cost = 1 * new_inventory
        
        # 模拟客户需求
        demand = np.random.randint(0, self.max_demand+1)
        
        # 计算销售收入和缺货成本
        sales = min(new_inventory, demand)
        revenue = 5 * sales
        stockout_cost = 4 * max(0, demand - new_inventory)
        
        # 更新库存
        self.state = new_inventory - sales
        
        # 计算奖励(利润)
        reward = revenue - ordering_cost - holding_cost - stockout_cost
        
        # 非终止环境
        done = False
        
        return self.state, reward, done, {}

def train_inventory_management():
    env = InventoryEnv()
    action_dim = 11  # 可以订购0-10个物品
    state_dim = env.max_inventory + 1  # 库存状态：0-20
    
    # 使用Q表进行学习
    Q = np.zeros((state_dim, action_dim))
    
    # 训练参数
    alpha = 0.1  # 学习率
    gamma = 0.99  # 折扣因子
    epsilon = 0.1  # 探索率
    num_episodes = 10000
    
    avg_rewards = []
    window_size = 100
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # ε-贪婪策略
            if np.random.random() < epsilon:
                action = np.random.randint(action_dim)
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning更新
            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            
            state = next_state
            total_reward += reward
            
            # 设置回合长度限制
            if done or total_reward < -500:
                break
        
        # 记录奖励
        avg_rewards.append(total_reward)
        
        if episode % 500 == 0:
            print(f"Episode {episode}, Avg Reward: {np.mean(avg_rewards[-100:]):.2f}")
    
    # 分析最优策略
    optimal_policy = {}
    for state in range(state_dim):
        optimal_action = np.argmax(Q[state])
        optimal_policy[state] = optimal_action
        print(f"库存量 {state}，最优订购量: {optimal_action}")
    
    return Q, optimal_policy
```

### 总结与展望

#### 时序差分学习的关键优势

1. **在线学习**：TD方法允许增量学习，不需等待回合结束
2. **样本效率**：Bootstrap使得TD比蒙特卡洛更有效地利用经验
3. **无模型**：不需要环境的转移模型
4. **功能全面**：既可用于策略评估，也可用于策略改进
5. **灵活性**：易于与函数近似、深度学习等技术结合

#### 最佳实践

1. **超参数选择**：
   - 学习率(α)：通常从0.01-0.1开始尝试
   - 折扣因子(γ)：0.9-0.99，根据问题的时间范围调整
   - ε-贪婪中的ε：通常从0.1开始，并考虑随时间衰减

2. **算法选择指南**：
   - 对于on-policy学习，选择SARSA或Expected SARSA
   - 对于off-policy学习和直接学习最优策略，选择Q-learning
   - 当过度估计是问题时，使用Double Q-learning
   - 对于复杂环境，考虑深度学习扩展如DQN

3. **函数近似注意事项**：
   - TD与函数近似结合时，小学习率通常更稳定
   - 考虑使用目标网络和经验回放来稳定训练
   - 规范化输入状态以改善学习

#### 未来方向

1. **更好的探索策略**：结合TD学习与高级探索策略
2. **多步方法的自适应调整**：根据数据特性动态调整n和λ
3. **离线RL**：从固定数据集中有效学习策略
4. **表征学习**：结合TD与表征学习改善状态编码
5. **多任务和迁移学习**：利用TD学习加速跨任务知识迁移

时序差分学习作为强化学习的核心方法，已经发展出丰富的算法家族，从简单的TD(0)到复杂的深度Q-Network。掌握这些方法，可以处理从简单表格环境到复杂连续控制任务的各种强化学习问题。随着计算能力的提升和算法的不断创新，TD学习的应用领域将继续扩展，在游戏AI、机器人控制、自动驾驶、金融交易等众多领域发挥关键作用。

Similar code found with 2 license types