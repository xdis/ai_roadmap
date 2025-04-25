# 在DQNAgent类中的update方法中实现Double DQN
def update(self):
    if len(self.replay_buffer) < self.batch_size:
        return 0
    
    transitions = self.replay_buffer.sample(self.batch_size)
    batch = list(zip(*transitions))
    
    states = torch.FloatTensor(batch[0])
    actions = torch.LongTensor(batch[1]).unsqueeze(1)
    rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
    next_states = torch.FloatTensor(batch[3])
    dones = torch.FloatTensor(batch[4]).unsqueeze(1)
    
    current_q = self.q_network(states).gather(1, actions)
    
    # Double DQN: 使用online网络选择动作，target网络评估价值
    with torch.no_grad():
        # 使用online网络选择best_action
        best_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        # 使用target网络评估这些动作
        next_q_values = self.target_network(next_states).gather(1, best_actions)
        target_q = rewards + (1 - dones) * self.gamma * next_q_values
    
    loss = F.mse_loss(current_q, target_q)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    # 更新target网络
    self.training_step += 1
    if self.training_step % self.update_target_every == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
    
    return loss.item()