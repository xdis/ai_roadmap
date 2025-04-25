# 马尔可夫决策过程：从零开始的完整指南

## 1. 基础概念理解

### 什么是马尔可夫决策过程(MDP)?

马尔可夫决策过程(Markov Decision Process, MDP)是强化学习的数学基础，它为序列决策问题提供了形式化框架。MDP描述了一个智能体在环境中进行决策的过程，其中下一个状态仅依赖于当前状态和所采取的动作，而不依赖于历史状态和动作序列。

### MDP的五元组表示

MDP由五个核心元素组成，通常表示为元组 $(S, A, P, R, \gamma)$:

1. **状态空间(State Space, $S$)**: 描述环境可能处于的所有状态的集合
   - 例如：在网格世界中，状态可以是智能体的位置坐标
   - 状态可以是离散的或连续的

2. **动作空间(Action Space, $A$)**: 智能体可以执行的所有动作的集合
   - 例如：上、下、左、右移动
   - 动作也可以是离散的或连续的

3. **转移概率(Transition Probability, $P$)**: 定义状态转移的动态规律
   - $P(s'|s,a)$ 表示在状态$s$下执行动作$a$后转移到状态$s'$的概率
   - 确定性环境中，$P(s'|s,a) = 1$对某个特定的$s'$，其他为0

4. **奖励函数(Reward Function, $R$)**: 定义智能体获得的即时反馈
   - $R(s,a,s')$ 表示从状态$s$执行动作$a$转移到状态$s'$时获得的奖励
   - 也可简化为 $R(s,a)$ 或 $R(s)$，取决于具体问题设定

5. **折扣因子(Discount Factor, $\gamma$)**: 控制未来奖励相对于即时奖励的重要性
   - $\gamma \in [0,1]$
   - $\gamma = 0$：仅考虑即时奖励
   - $\gamma = 1$：所有时间步的奖励同等重要（只适用于有限时间步问题）

### 马尔可夫性质

MDP的核心特性是马尔可夫性质(Markov Property)，意味着当前状态包含预测未来所需的所有信息：

$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1},...,s_0, a_0) = P(s_{t+1}|s_t, a_t)$

这个性质大大简化了决策问题，使我们可以只考虑当前状态而不需要追踪整个历史。

### MDP的目标

在MDP中，智能体的目标是找到一个策略 $\pi$，使从初始状态开始的期望累积奖励最大化：

$\pi^* = \arg\max_\pi E_\pi[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)]$

其中 $\pi(a|s)$ 表示在状态 $s$ 选择动作 $a$ 的概率。

## 2. 技术细节探索

### 价值函数与最优性

#### 状态价值函数(State-Value Function)

状态价值函数 $V^\pi(s)$ 定义为遵循策略 $\pi$ 从状态 $s$ 开始的期望累积奖励：

$V^\pi(s) = E_\pi[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) | s_0 = s]$

#### 动作价值函数(Action-Value Function)

动作价值函数 $Q^\pi(s,a)$ 定义为在状态 $s$ 执行动作 $a$ 然后遵循策略 $\pi$ 的期望累积奖励：

$Q^\pi(s,a) = E_\pi[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) | s_0 = s, a_0 = a]$

#### 最优价值函数

最优状态价值函数 $V^*(s)$ 和最优动作价值函数 $Q^*(s,a)$ 定义为：

$V^*(s) = \max_\pi V^\pi(s)$
$Q^*(s,a) = \max_\pi Q^\pi(s,a)$

#### 最优策略

最优策略 $\pi^*$ 是使价值函数最大化的策略：

$\pi^*(s) = \arg\max_a Q^*(s,a)$

### 贝尔曼方程

贝尔曼方程是解决MDP问题的核心等式，它将当前状态的价值与后继状态的价值关联起来。

#### 贝尔曼期望方程

对于给定策略 $\pi$，贝尔曼期望方程为：

$V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} P(s', r|s, a)[r + \gamma V^\pi(s')]$

$Q^\pi(s,a) = \sum_{s',r} P(s', r|s, a)[r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$

#### 贝尔曼最优方程

最优价值函数满足贝尔曼最优方程：

$V^*(s) = \max_a \sum_{s',r} P(s', r|s, a)[r + \gamma V^*(s')]$

$Q^*(s,a) = \sum_{s',r} P(s', r|s, a)[r + \gamma \max_{a'} Q^*(s',a')]$

### 求解MDP的方法

#### 动态规划方法

1. **策略迭代(Policy Iteration)**
   - 交替执行策略评估和策略改进
   - **策略评估**：计算当前策略的价值函数
   - **策略改进**：基于当前价值函数更新策略

   ```python
   def policy_iteration(mdp, theta=1e-6, gamma=0.9):
       # 初始化任意策略和价值函数
       V = {s: 0 for s in mdp.states}
       policy = {s: random.choice(mdp.actions(s)) for s in mdp.states}
       
       while True:
           # 策略评估
           while True:
               delta = 0
               for s in mdp.states:
                   v = V[s]
                   # 计算状态s在当前策略下的价值
                   a = policy[s]
                   V[s] = sum(p * (r + gamma * V[s_prime]) 
                             for s_prime, p, r in mdp.transitions(s, a))
                   delta = max(delta, abs(v - V[s]))
               if delta < theta:
                   break
           
           # 策略改进
           policy_stable = True
           for s in mdp.states:
               old_action = policy[s]
               # 找到使价值最大的动作
               policy[s] = max(mdp.actions(s), 
                             key=lambda a: sum(p * (r + gamma * V[s_prime]) 
                                          for s_prime, p, r in mdp.transitions(s, a)))
               if old_action != policy[s]:
                   policy_stable = False
                   
           if policy_stable:
               return V, policy
   ```

2. **值迭代(Value Iteration)**
   - 直接迭代最优贝尔曼方程
   - 不需要显式的策略评估步骤

   ```python
   def value_iteration(mdp, theta=1e-6, gamma=0.9):
       # 初始化价值函数
       V = {s: 0 for s in mdp.states}
       
       while True:
           delta = 0
           for s in mdp.states:
               v = V[s]
               # 贝尔曼最优方程更新
               V[s] = max(sum(p * (r + gamma * V[s_prime]) 
                           for s_prime, p, r in mdp.transitions(s, a)) 
                           for a in mdp.actions(s))
               delta = max(delta, abs(v - V[s]))
           if delta < theta:
               break
               
       # 提取最优策略
       policy = {s: max(mdp.actions(s), 
                       key=lambda a: sum(p * (r + gamma * V[s_prime]) 
                                    for s_prime, p, r in mdp.transitions(s, a))) 
                 for s in mdp.states}
               
       return V, policy
   ```

#### 模型无关的方法

当环境模型未知时，我们可以使用模型无关方法：

1. **蒙特卡洛方法**：通过采样完整轨迹估计价值
2. **时序差分学习**：结合蒙特卡洛和动态规划的思想
   - Q-learning, SARSA 等

## 3. 实践与实现

### MDP实现示例：网格世界

下面是一个简单的网格世界MDP实现：

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GridWorldMDP:
    def __init__(self, width=5, height=5, start=(0,0), goal=(4,4)):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        
        # 定义状态空间
        self.states = [(x, y) for x in range(width) for y in range(height)]
        
        # 定义动作空间: 上(0)、右(1)、下(2)、左(3)
        self.action_space = [0, 1, 2, 3]
        self.action_names = ['↑', '→', '↓', '←']
        
        # 定义障碍物
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        
        # 移除障碍物状态
        self.states = [s for s in self.states if s not in self.obstacles]
        
        # 定义奖励函数
        self.rewards = {self.goal: 1.0}  # 目标状态奖励为1
        
        # 定义转移概率
        # 在此简单示例中，转移是确定性的
        
    def actions(self, state):
        """返回状态s可用的动作列表"""
        return self.action_space
    
    def transitions(self, state, action):
        """
        返回从状态s执行动作a后的转移列表
        每个转移为三元组(下一状态, 概率, 奖励)
        """
        # 确定下一个状态
        x, y = state
        next_state = state
        
        # 上
        if action == 0:
            next_state = (x, max(0, y-1))
        # 右
        elif action == 1:
            next_state = (min(self.width-1, x+1), y)
        # 下
        elif action == 2:
            next_state = (x, min(self.height-1, y+1))
        # 左
        elif action == 3:
            next_state = (max(0, x-1), y)
        
        # 如果下一状态是障碍物，则保持原位
        if next_state in self.obstacles:
            next_state = state
            
        # 获取奖励
        reward = self.rewards.get(next_state, -0.04)  # 默认步数惩罚为-0.04
        
        return [(next_state, 1.0, reward)]  # 确定性转移
    
    def is_terminal(self, state):
        """判断状态是否是终止状态"""
        return state == self.goal
    
    def visualize_values(self, V):
        """可视化状态价值函数"""
        grid = np.zeros((self.height, self.width)) - 2
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) in self.obstacles:
                    grid[y, x] = np.nan
                elif (x, y) in V:
                    grid[y, x] = V[(x, y)]
        
        plt.figure(figsize=(10, 8))
        mask = np.isnan(grid)
        sns.heatmap(grid, annot=True, fmt='.2f', cmap="YlGnBu", mask=mask)
        plt.title("State Value Function")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
    
    def visualize_policy(self, policy):
        """可视化策略"""
        grid = np.zeros((self.height, self.width), dtype=object)
        grid.fill('')
        
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) in self.obstacles:
                    grid[y, x] = 'X'
                elif (x, y) == self.goal:
                    grid[y, x] = 'G'
                elif (x, y) in policy:
                    action = policy[(x, y)]
                    grid[y, x] = self.action_names[action]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(np.zeros((self.height, self.width)), cmap='Pastel1')
        for i in range(self.height):
            for j in range(self.width):
                plt.text(j, i, grid[i, j], ha='center', va='center', fontsize=20)
        plt.grid(True)
        plt.title("Optimal Policy")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
```

### 值迭代解决网格世界问题

```python
def value_iteration_gridworld():
    # 创建网格世界MDP
    grid_world = GridWorldMDP(width=5, height=5)
    
    # 初始化值函数
    V = {s: 0 for s in grid_world.states}
    gamma = 0.9  # 折扣因子
    theta = 1e-6  # 收敛阈值
    max_iterations = 1000
    
    # 值迭代
    for i in range(max_iterations):
        delta = 0
        for s in grid_world.states:
            if grid_world.is_terminal(s):
                continue
                
            v = V[s]
            # 计算状态s下所有动作的值，并选择最大的
            action_values = []
            for a in grid_world.actions(s):
                value = 0
                for next_s, prob, reward in grid_world.transitions(s, a):
                    value += prob * (reward + gamma * V[next_s])
                action_values.append(value)
            
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        
        # 检查是否收敛
        if delta < theta:
            print(f"值迭代在第 {i+1} 次迭代后收敛")
            break
    
    # 提取最优策略
    policy = {}
    for s in grid_world.states:
        if grid_world.is_terminal(s):
            continue
            
        best_action = None
        best_value = float('-inf')
        for a in grid_world.actions(s):
            value = 0
            for next_s, prob, reward in grid_world.transitions(s, a):
                value += prob * (reward + gamma * V[next_s])
            
            if value > best_value:
                best_value = value
                best_action = a
        
        policy[s] = best_action
    
    # 可视化结果
    grid_world.visualize_values(V)
    grid_world.visualize_policy(policy)
    
    return V, policy

# 运行值迭代
V, policy = value_iteration_gridworld()
```

### 策略迭代解决网格世界问题

```python
def policy_iteration_gridworld():
    # 创建网格世界MDP
    grid_world = GridWorldMDP(width=5, height=5)
    
    # 初始化策略（随机策略）
    policy = {s: np.random.choice(grid_world.action_space) for s in grid_world.states}
    V = {s: 0 for s in grid_world.states}
    
    gamma = 0.9  # 折扣因子
    theta = 1e-6  # 收敛阈值
    max_iterations = 100
    
    # 策略迭代
    for i in range(max_iterations):
        # 策略评估
        for _ in range(100):  # 最多迭代100次
            delta = 0
            for s in grid_world.states:
                if grid_world.is_terminal(s):
                    continue
                    
                v = V[s]
                # 根据当前策略计算状态值
                a = policy[s]
                V[s] = 0
                for next_s, prob, reward in grid_world.transitions(s, a):
                    V[s] += prob * (reward + gamma * V[next_s])
                
                delta = max(delta, abs(v - V[s]))
            
            if delta < theta:
                break
        
        # 策略改进
        policy_stable = True
        for s in grid_world.states:
            if grid_world.is_terminal(s):
                continue
                
            old_action = policy[s]
            
            # 找到使值最大的动作
            best_action = None
            best_value = float('-inf')
            for a in grid_world.actions(s):
                value = 0
                for next_s, prob, reward in grid_world.transitions(s, a):
                    value += prob * (reward + gamma * V[next_s])
                
                if value > best_value:
                    best_value = value
                    best_action = a
            
            policy[s] = best_action
            
            if old_action != policy[s]:
                policy_stable = False
        
        # 如果策略稳定，则结束迭代
        if policy_stable:
            print(f"策略迭代在第 {i+1} 次迭代后收敛")
            break
    
    # 可视化结果
    grid_world.visualize_values(V)
    grid_world.visualize_policy(policy)
    
    return V, policy

# 运行策略迭代
V, policy = policy_iteration_gridworld()
```

## 4. 高级应用与变体

### 部分可观察马尔可夫决策过程(POMDP)

在现实世界中，智能体通常无法完全观察到环境状态，这就引入了部分可观察马尔可夫决策过程(Partially Observable MDP, POMDP)概念。

POMDP由八元组 $(S, A, P, R, \Omega, O, \gamma, b_0)$ 定义:
- $S, A, P, R, \gamma$ 与MDP相同
- $\Omega$: 观察空间
- $O$: 观察函数，$O(o|s',a)$ 表示在执行动作 $a$ 后到达状态 $s'$ 时观察到 $o$ 的概率
- $b_0$: 初始状态信念分布

在POMDP中，智能体维护一个信念状态(Belief State)，表示当前时刻对实际状态的概率分布。

求解POMDP比MDP复杂得多，常用方法包括:
- SARSOP (Successive Approximations of the Reachable Space under Optimal Policies)
- PBVI (Point-Based Value Iteration)
- POMCP (Partially Observable Monte-Carlo Planning)

### 平均奖励MDP

在无限时间步且无折扣的情况下，累积奖励可能是无限的。此时，我们可以考虑平均奖励(Average Reward)标准：

$\rho^\pi = \lim_{n\to\infty} \frac{1}{n}E[\sum_{t=0}^{n-1} R(s_t, a_t)]$

相应的贝尔曼方程变为：

$h^\pi(s) = E[R(s,a) - \rho^\pi + h^\pi(s') | s, \pi]$

其中 $h^\pi(s)$ 是偏差函数(Differential Value Function)。

### 多标准MDP

有时我们需要同时优化多个目标，这就是多标准MDP(Multi-Criteria MDP)。例如，自动驾驶中既要保证安全又要减少旅行时间。

在多标准MDP中，奖励是一个向量而不是标量：

$\vec{R}(s,a,s') = [R_1(s,a,s'), R_2(s,a,s'), ..., R_k(s,a,s')]$

解决方法包括:
- 将多个目标加权组合成单一目标
- 使用帕累托优化方法
- 约束优化方法，如约束马尔可夫决策过程(CMDP)

### 分层强化学习与选项框架

为了处理复杂的长期任务，可以使用分层强化学习(Hierarchical RL)方法，如选项框架(Options Framework)。

选项是一个三元组 $o = (\mathcal{I}, \pi, \beta)$:
- $\mathcal{I}$: 启动集，表示可以选择该选项的状态集合
- $\pi$: 选项内部策略
- $\beta$: 终止函数，$\beta(s)$ 表示在状态 $s$ 终止选项的概率

选项允许智能体在不同抽象层次上做决策，从而处理更复杂的任务。

### 转移学习与微调

在实际应用中，我们通常需要在不同但相关的MDP之间转移学习。例如，机器人在不同地形上学习导航。

方法包括:
- 价值函数转移
- 模型转移
- 策略转移与微调

### 连续状态与动作空间

实际问题中，状态和动作空间通常是连续的，这需要函数近似方法:
- 线性函数近似
- 核方法
- 神经网络

### 稀疏奖励与内在动机

在稀疏奖励情况下，可以使用内在动机(Intrinsic Motivation)来促进探索:
- 好奇心驱动探索
- 熵最大化
- 噪声网络蒸馏(RND)

## 实际应用案例

### 机器人路径规划

```python
class RobotNavigation:
    def __init__(self, grid_size=10, obstacles=None):
        self.grid_size = grid_size
        self.obstacles = obstacles or []
        self.start = (0, 0)
        self.goal = (grid_size-1, grid_size-1)
        
        # 创建MDP
        self.states = [(x, y) for x in range(grid_size) for y in range(grid_size) if (x, y) not in self.obstacles]
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上、右、下、左
    
    def solve_with_value_iteration(self, gamma=0.95, theta=1e-6):
        # 初始化值函数
        V = {s: 0 for s in self.states}
        
        # 值迭代
        while True:
            delta = 0
            for s in self.states:
                if s == self.goal:
                    continue
                
                v = V[s]
                
                # 计算最大Q值
                max_q = float('-inf')
                for a in self.actions:
                    next_s = (s[0] + a[0], s[1] + a[1])
                    
                    # 检查边界和障碍物
                    if (next_s[0] < 0 or next_s[0] >= self.grid_size or 
                        next_s[1] < 0 or next_s[1] >= self.grid_size or 
                        next_s in self.obstacles):
                        next_s = s  # 保持原位
                    
                    # 计算奖励
                    if next_s == self.goal:
                        r = 100  # 到达目标的高奖励
                    elif next_s == s:
                        r = -1   # 撞墙惩罚
                    else:
                        r = -0.1  # 步数惩罚
                    
                    q = r + gamma * V[next_s]
                    max_q = max(max_q, q)
                
                V[s] = max_q
                delta = max(delta, abs(v - V[s]))
            
            if delta < theta:
                break
        
        # 提取策略
        policy = {}
        for s in self.states:
            if s == self.goal:
                continue
                
            best_action = None
            best_value = float('-inf')
            
            for a in self.actions:
                next_s = (s[0] + a[0], s[1] + a[1])
                
                # 检查边界和障碍物
                if (next_s[0] < 0 or next_s[0] >= self.grid_size or 
                    next_s[1] < 0 or next_s[1] >= self.grid_size or 
                    next_s in self.obstacles):
                    next_s = s
                
                # 计算奖励
                if next_s == self.goal:
                    r = 100
                elif next_s == s:
                    r = -1
                else:
                    r = -0.1
                
                value = r + gamma * V[next_s]
                
                if value > best_value:
                    best_value = value
                    best_action = a
            
            policy[s] = best_action
        
        return V, policy
    
    def simulate_path(self, policy, max_steps=100):
        current = self.start
        path = [current]
        
        for _ in range(max_steps):
            if current == self.goal:
                break
                
            action = policy[current]
            next_s = (current[0] + action[0], current[1] + action[1])
            
            # 检查边界和障碍物
            if (next_s[0] < 0 or next_s[0] >= self.grid_size or 
                next_s[1] < 0 or next_s[1] >= self.grid_size or 
                next_s in self.obstacles):
                next_s = current
            
            current = next_s
            path.append(current)
        
        return path
    
    def visualize_path(self, path):
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # 标记障碍物
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = -1
            
        # 标记路径
        for i, (x, y) in enumerate(path):
            grid[y, x] = i + 1
        
        # 可视化
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap='viridis')
        
        # 标记起点和终点
        plt.scatter(self.start[0], self.start[1], c='green', s=200, marker='*', label='Start')
        plt.scatter(self.goal[0], self.goal[1], c='red', s=200, marker='*', label='Goal')
        
        # 标记路径
        xs, ys = zip(*path)
        plt.plot(xs, ys, 'w-', linewidth=2)
        
        plt.colorbar(label='Step')
        plt.legend()
        plt.title('Robot Navigation Path')
        plt.show()
```

### 资源分配优化

```python
class ResourceAllocationMDP:
    def __init__(self, n_resources=3, n_tasks=2, max_allocation=5):
        self.n_resources = n_resources  # 资源数量
        self.n_tasks = n_tasks          # 任务数量
        self.max_allocation = max_allocation  # 每个任务最大资源分配
        
        # 状态空间: (r1, r2, ..., rn) 表示每个任务分配的资源数量
        self.states = self._generate_states()
        
        # 动作空间: 资源重新分配方案
        self.actions = self._generate_actions()
    
    def _generate_states(self):
        # 生成所有可能的资源分配状态
        states = []
        
        def backtrack(task, remaining, allocation):
            if task == self.n_tasks:
                if remaining == 0:
                    states.append(tuple(allocation))
                return
                
            for i in range(min(remaining, self.max_allocation) + 1):
                allocation[task] = i
                backtrack(task + 1, remaining - i, allocation)
                allocation[task] = 0
        
        backtrack(0, self.n_resources, [0] * self.n_tasks)
        return states
    
    def _generate_actions(self):
        # 简化: 动作为资源重新分配
        return self.states
    
    def reward(self, state, action):
        # 简化的奖励函数，基于任务完成效率
        # 假设资源分配越均匀，效率越高
        old_std = np.std(state)
        new_std = np.std(action)
        
        # 转移成本(资源重新分配的成本)
        reallocation_cost = sum(abs(a - s) for a, s in zip(action, state)) * 0.1
        
        # 效率提升奖励
        efficiency_gain = 5 * (old_std - new_std)
        
        return efficiency_gain - reallocation_cost
    
    def is_valid_transition(self, state, action):
        # 检查是否为有效转移(总资源数量保持不变)
        return sum(state) == sum(action)
    
    def solve_with_value_iteration(self, gamma=0.95, theta=1e-6):
        # 初始化值函数
        V = {s: 0 for s in self.states}
        
        # 值迭代
        while True:
            delta = 0
            for s in self.states:
                v = V[s]
                
                # 计算最大Q值
                max_q = float('-inf')
                for a in self.actions:
                    if not self.is_valid_transition(s, a):
                        continue
                    
                    r = self.reward(s, a)
                    q = r + gamma * V[a]  # 动作直接决定下一状态
                    max_q = max(max_q, q)
                
                # 如果没有有效动作，保持原值
                if max_q != float('-inf'):
                    V[s] = max_q
                    delta = max(delta, abs(v - V[s]))
            
            if delta < theta:
                break
        
        # 提取策略
        policy = {}
        for s in self.states:
            best_action = None
            best_value = float('-inf')
            
            for a in self.actions:
                if not self.is_valid_transition(s, a):
                    continue
                    
                r = self.reward(s, a)
                value = r + gamma * V[a]
                
                if value > best_value:
                    best_value = value
                    best_action = a
            
            policy[s] = best_action
        
        return V, policy
```

### 库存管理系统

```python
class InventoryManagementMDP:
    def __init__(self, max_inventory=10, max_order=5, demand_probs=None):
        self.max_inventory = max_inventory  # 最大库存容量
        self.max_order = max_order          # 单次最大订货量
        
        # 需求概率分布(简化为离散分布)
        self.demand_probs = demand_probs or {
            0: 0.1,
            1: 0.2,
            2: 0.3,
            3: 0.2,
            4: 0.1,
            5: 0.1
        }
        
        # 状态空间: 当前库存水平(0到max_inventory)
        self.states = list(range(max_inventory + 1))
        
        # 动作空间: 订货量(0到max_order)
        self.actions = list(range(max_order + 1))
        
        # 成本参数
        self.holding_cost = 1.0    # 每单位库存持有成本
        self.ordering_cost = 3.0   # 固定订货成本
        self.stockout_cost = 5.0   # 每单位缺货成本
        self.unit_price = 2.0      # 每单位商品成本
    
    def transition_prob(self, state, action, next_state):
        """计算状态转移概率 P(next_state | state, action)"""
        # 当前库存 + 订货量
        total_inventory = state + action
        
        # 需求量 = 总库存 - 下一状态库存
        demand = total_inventory - next_state
        
        # 如果需求为负或超过最大值，概率为0
        if demand < 0 or demand > max(self.demand_probs.keys()):
            return 0.0
        
        # 返回相应需求的概率
        return self.demand_probs.get(demand, 0.0)
    
    def reward(self, state, action, next_state):
        """计算奖励(负成本)"""
        # 当前库存 + 订货量
        total_inventory = state + action
        
        # 需求量
        demand = total_inventory - next_state
        
        # 固定订货成本
        fixed_cost = self.ordering_cost if action > 0 else 0
        
        # 单位商品成本
        unit_cost = self.unit_price * action
        
        # 库存持有成本
        holding_cost = self.holding_cost * next_state
        
        # 缺货成本(如果需求超过总库存)
        stockout = max(0, demand - total_inventory)
        stockout_cost = self.stockout_cost * stockout
        
        # 总成本
        total_cost = fixed_cost + unit_cost + holding_cost + stockout_cost
        
        # 返回负成本作为奖励
        return -total_cost
    
    def get_valid_next_states(self, state, action):
        """给定当前状态和动作，返回所有可能的下一状态及其概率"""
        valid_next_states = {}
        total_inventory = state + action
        
        # 考虑所有可能的需求
        for demand, prob in self.demand_probs.items():
            next_inventory = max(0, total_inventory - demand)
            next_inventory = min(next_inventory, self.max_inventory)
            
            if next_inventory in valid_next_states:
                valid_next_states[next_inventory] += prob
            else:
                valid_next_states[next_inventory] = prob
        
        return valid_next_states
    
    def solve_with_value_iteration(self, gamma=0.95, theta=1e-6):
        """使用值迭代算法求解最优策略"""
        # 初始化值函数
        V = {s: 0 for s in self.states}
        
        # 值迭代
        iteration = 0
        while True:
            delta = 0
            for s in self.states:
                v = V[s]
                
                # 计算最大Q值
                max_q = float('-inf')
                for a in self.actions:
                    # 确保不超过最大库存
                    if s + a > self.max_inventory:
                        continue
                    
                    q_value = 0
                    next_states = self.get_valid_next_states(s, a)
                    
                    for next_s, prob in next_states.items():
                        reward = self.reward(s, a, next_s)
                        q_value += prob * (reward + gamma * V[next_s])
                    
                    max_q = max(max_q, q_value)
                
                # 更新值函数
                V[s] = max_q
                delta = max(delta, abs(v - V[s]))
            
            iteration += 1
            if delta < theta:
                print(f"值迭代在第 {iteration} 次迭代后收敛")
                break
        
        # 提取策略
        policy = {}
        for s in self.states:
            best_action = None
            best_value = float('-inf')
            
            for a in self.actions:
                # 确保不超过最大库存
                if s + a > self.max_inventory:
                    continue
                
                q_value = 0
                next_states = self.get_valid_next_states(s, a)
                
                for next_s, prob in next_states.items():
                    reward = self.reward(s, a, next_s)
                    q_value += prob * (reward + gamma * V[next_s])
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = a
            
            policy[s] = best_action
        
        return V, policy
```

## 总结

马尔可夫决策过程(MDP)是强化学习的基础数学框架，它为解决序列决策问题提供了形式化工具。MDP的五个核心要素(状态、动作、转移概率、奖励和折扣因子)使我们能够对各种决策问题进行建模并寻找最优策略。

通过掌握MDP的核心概念、贝尔曼方程和求解方法(如策略迭代和值迭代)，我们可以解决从简单的网格世界到复杂的实际应用问题。随着问题复杂度的增加，我们可以使用各种MDP变体，如部分可观察MDP、多标准MDP和分层MDP，以处理不同类型的挑战。

MDP为强化学习提供了严格的理论基础，理解它对于掌握更高级的强化学习算法和应用至关重要。通过实践和实现，我们可以将这些理论概念应用到各种现实世界的问题中，如机器人控制、资源分配、库存管理等。

Similar code found with 1 license type