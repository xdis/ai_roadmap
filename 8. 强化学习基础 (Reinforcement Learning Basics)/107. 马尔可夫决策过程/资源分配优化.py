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