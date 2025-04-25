import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 创建所有头的线性投影，合并为一个矩阵提高效率
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        Q, K, V: [batch_size, num_heads, seq_len, d_k]
        mask: [batch_size, 1, seq_len, seq_len] 或 None
        """
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, heads, seq_len, seq_len]
        attn_scores = attn_scores / math.sqrt(self.d_k)
        
        # 应用掩码(如果有)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 注意力加权和
        output = torch.matmul(attn_weights, V)  # [batch, heads, seq_len, d_k]
        return output
        
    def forward(self, q, k, v, mask=None):
        """
        q, k, v: [batch_size, seq_len, d_model]
        mask: [batch_size, 1, seq_len] 或 None
        返回: [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)
        
        # 1. 线性投影
        q = self.W_q(q)  # [batch, seq_len, d_model]
        k = self.W_k(k)  # [batch, seq_len, d_model]
        v = self.W_v(v)  # [batch, seq_len, d_model]
        
        # 2. 将投影结果重塑为多头形式
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 现在 q, k, v 形状: [batch, heads, seq_len, d_k]
        
        # 3. 应用注意力
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        
        # 4. "拼接"多头结果
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # 5. 最终线性投影
        output = self.W_o(attn_output)
        
        return output