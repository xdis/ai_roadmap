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