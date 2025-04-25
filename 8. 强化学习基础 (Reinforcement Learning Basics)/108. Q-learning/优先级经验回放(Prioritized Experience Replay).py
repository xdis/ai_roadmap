class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # 决定优先级程度
        self.beta = beta    # 用于重要性采样权重
        self.beta_increment = beta_increment  # beta逐渐增长到1
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # 新经验给予最高优先级
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) < self.capacity:
            probs = self.priorities[:len(self.buffer)]
        else:
            probs = self.priorities
        
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        
        # 增加beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = list(zip(*samples))
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)