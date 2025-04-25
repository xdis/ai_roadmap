class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # 第一个LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        # 多头自注意力
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        # 第二个LayerNorm
        self.norm2 = nn.LayerNorm(dim)
        # MLP块
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features, dim, dropout)
    
    def forward(self, x):
        # x: [B, N, D]
        # 自注意力需要序列在第一维
        x_norm = self.norm1(x)  # [B, N, D]
        # 转置后进行注意力计算
        x_t = x_norm.transpose(0, 1)  # [N, B, D]
        attn_output, _ = self.attn(x_t, x_t, x_t)
        attn_output = attn_output.transpose(0, 1)  # [B, N, D]
        
        # 残差连接1
        x = x + attn_output
        
        # MLP块处理
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        
        # 残差连接2
        x = x + mlp_output
        
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x