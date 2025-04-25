import torch
import torch.nn as nn
import math

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # 掩码自注意力子层
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 编码器-解码器注意力子层
        attn_output, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 0).transpose(0, 1)
        return mask
        
    def encode(self, src, src_mask=None):
        # src: [batch_size, src_len]
        
        # 嵌入和位置编码
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)  # [batch_size, src_len, d_model]
        src = src.transpose(0, 1)  # [src_len, batch_size, d_model]
        src = self.positional_encoding(src)
        src = self.dropout(src)
        
        # 编码器层
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
            
        return src
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt: [batch_size, tgt_len]
        # memory: [src_len, batch_size, d_model]
        
        # 嵌入和位置编码
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)  # [batch_size, tgt_len, d_model]
        tgt = tgt.transpose(0, 1)  # [tgt_len, batch_size, d_model]
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)
        
        # 解码器层
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
            
        # 输出投影
        output = self.output_linear(tgt.transpose(0, 1))  # [batch_size, tgt_len, tgt_vocab_size]
        
        return output
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 创建掩码(如果未提供)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
        # 编码
        memory = self.encode(src, src_mask)
        
        # 解码
        output = self.decode(tgt, memory, tgt_mask, src_mask)
        
        return output