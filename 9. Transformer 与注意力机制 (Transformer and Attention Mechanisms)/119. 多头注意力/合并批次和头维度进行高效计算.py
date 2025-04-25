# 高效实现
def efficient_attention(q, k, v, mask=None):
    # q, k, v: [batch, heads, seq_len, d_k]
    batch_size, num_heads, seq_len, d_k = q.shape
    
    # 重塑为 [batch*heads, seq_len, d_k]
    q_flat = q.reshape(-1, seq_len, d_k)
    k_flat = k.reshape(-1, seq_len, d_k)
    v_flat = k.reshape(-1, seq_len, d_k)
    
    # 统一计算所有批次和头
    attn_flat = scaled_dot_product_attention(q_flat, k_flat, v_flat, mask)
    
    # 重新整形回 [batch, heads, seq_len, d_k]
    return attn_flat.reshape(batch_size, num_heads, seq_len, d_k)