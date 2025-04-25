# 简化的自注意力计算
def self_attention(query, key, value, mask=None):
    # 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    
    # 应用因果掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 应用softmax获取注意力权重
    weights = F.softmax(scores, dim=-1)
    
    # 计算加权和
    return torch.matmul(weights, value)