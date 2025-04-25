def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # 保存最佳路径概率
    path = {}  # 保存最佳路径
    
    # 初始化
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
    
    # 递推
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        
        for y in states:
            # 选择最大概率的前导状态
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            
            V[t][y] = prob
            newpath[y] = path[state] + [y]
            
        path = newpath
    
    # 找出最优路径
    (prob, state) = max((V[len(obs) - 1][y], y) for y in states)
    return path[state]