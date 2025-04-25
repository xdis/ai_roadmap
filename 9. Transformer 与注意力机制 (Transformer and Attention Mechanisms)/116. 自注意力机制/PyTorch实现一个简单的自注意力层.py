import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads=1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # 确保嵌入维度能被注意力头数整除
        assert self.head_dim * heads == embed_size
        
        # 定义线性层生成Q, K, V
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, embed_size)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # 生成查询、键、值向量
        q = self.q_linear(x)  # (batch_size, seq_len, embed_size)
        k = self.k_linear(x)  # (batch_size, seq_len, embed_size)
        v = self.v_linear(x)  # (batch_size, seq_len, embed_size)
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 注意力计算: Q * K^T
        energy = torch.matmul(q, k.permute(0, 1, 3, 2))  # batch, heads, seq_len, seq_len
        
        # 缩放
        energy = energy / (self.head_dim ** 0.5)
        
        # 应用掩码(如果有)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # 注意力权重
        attention = F.softmax(energy, dim=-1)
        
        # 注意力加权和
        out = torch.matmul(attention, v)  # (batch, heads, seq_len, head_dim)
        
        # 重新整形
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.embed_size)
        
        # 最终线性投影
        out = self.fc_out(out)
        return out