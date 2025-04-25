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