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