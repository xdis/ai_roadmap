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