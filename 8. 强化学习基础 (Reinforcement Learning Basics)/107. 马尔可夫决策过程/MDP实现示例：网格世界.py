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