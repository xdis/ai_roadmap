def create_attention_mask(seq_len):
    # 创建上三角掩码 (不允许关注未来位置)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # 转换为布尔值，True代表允许注意