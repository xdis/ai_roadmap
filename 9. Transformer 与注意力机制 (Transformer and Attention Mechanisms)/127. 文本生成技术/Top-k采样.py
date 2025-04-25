def top_k_sampling(logits, k=50, temperature=1.0):
    """Top-k采样实现"""
    # 应用温度
    scaled_logits = logits / temperature
    
    # 获取top-k值和索引
    top_k_values, top_k_indices = torch.topk(scaled_logits, k)
    
    # 创建一个全零的概率分布
    probs = torch.zeros_like(scaled_logits)
    
    # 在top-k位置填入缩放后的概率
    probs.scatter_(0, top_k_indices, F.softmax(top_k_values, dim=-1))
    
    # 采样
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token