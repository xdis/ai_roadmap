import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引使用cos
        
        # 添加批次维度并注册为缓冲区(非参数)
        pe = pe.unsqueeze(0)  # [1, max_seq_length, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        # 只使用序列实际长度对应的位置编码
        x = x + self.pe[:, :x.size(1)]
        return x