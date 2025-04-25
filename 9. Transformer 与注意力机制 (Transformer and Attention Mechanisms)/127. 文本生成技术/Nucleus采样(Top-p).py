def nucleus_sampling(logits, p=0.9, temperature=1.0):
    """Nucleus (Top-p) 采样实现"""
    # 应用温度
    scaled_logits = logits / temperature
    
    # 计算softmax
    probs = F.softmax(scaled_logits, dim=-1)
    
    # 按概率排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 创建掩码，仅保留累积概率≤p的标记
    sorted_indices_to_remove = cumulative_probs > p
    # 将掩码向右移动一位，确保至少保留一个标记
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # 将低于阈值的概率设为0
    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    probs = probs.masked_fill(indices_to_remove, 0.0)
    
    # 重新归一化概率
    probs = probs / probs.sum()
    
    # 采样
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token