def relative_attention(q, k, v, max_relative_position=16):
    # 标准注意力分数
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    
    # 获取序列长度
    batch_size, num_heads, seq_len, d_k = q.size()
    
    # 生成相对位置索引
    range_vec = torch.arange(seq_len)
    relative_position = range_vec.unsqueeze(1) - range_vec.unsqueeze(0)
    
    # 截断相对位置到最大范围
    relative_position = torch.clamp(relative_position, -max_relative_position, max_relative_position)
    
    # 调整到非负索引
    relative_position = relative_position + max_relative_position
    
    # 创建可学习的相对位置偏置矩阵
    relative_bias = nn.Parameter(torch.zeros(2 * max_relative_position + 1, num_heads))
    
    # 应用相对位置偏置
    relative_bias_score = relative_bias[relative_position.view(-1)].view(seq_len, seq_len, -1)
    relative_bias_score = relative_bias_score.permute(2, 0, 1).unsqueeze(0)
    
    # 添加到注意力分数
    matmul_qk = matmul_qk + relative_bias_score
    
    # 缩放
    matmul_qk = matmul_qk / math.sqrt(d_k)
    
    # 应用softmax
    attn_weights = F.softmax(matmul_qk, dim=-1)
    
    # 注意力加权和
    output = torch.matmul(attn_weights, v)
    
    return output