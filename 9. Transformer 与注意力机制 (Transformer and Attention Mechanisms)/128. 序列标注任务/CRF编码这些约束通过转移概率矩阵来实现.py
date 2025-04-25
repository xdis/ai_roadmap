# 转移矩阵示例 (简化版)
transitions = torch.full((num_tags, num_tags), -10000.0)
# 允许的转换
transitions[tag2idx['O'], tag2idx['O']] = 0.0
transitions[tag2idx['O'], tag2idx['B-PER']] = 0.0
transitions[tag2idx['B-PER'], tag2idx['I-PER']] = 0.0
# 不允许从O到I-X
transitions[tag2idx['O'], tag2idx['I-PER']] = -10000.0