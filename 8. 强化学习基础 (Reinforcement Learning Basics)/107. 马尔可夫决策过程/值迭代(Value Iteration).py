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