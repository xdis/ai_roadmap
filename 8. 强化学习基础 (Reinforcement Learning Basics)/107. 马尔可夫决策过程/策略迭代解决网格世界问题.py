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