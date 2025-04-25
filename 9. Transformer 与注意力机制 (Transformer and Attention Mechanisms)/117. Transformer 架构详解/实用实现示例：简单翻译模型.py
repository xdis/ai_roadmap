# 完整Transformer实现(简化版)
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_seq_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # 嵌入层和位置编码
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 编码器和解码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def generate_square_subsequent_mask(self, sz):
        """生成解码器的后续掩码，防止关注未来位置"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        if src_mask is None:
            src_mask = torch.ones((src.shape[0], 1, src.shape[1])).to(src.device)
            
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
            tgt_mask = tgt_mask.unsqueeze(0).expand(tgt.shape[0], -1, -1)
        
        # 编码器部分
        src_emb = self.dropout(self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model)))
        enc_output = src_emb
        
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        # 解码器部分
        tgt_emb = self.dropout(self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        dec_output = tgt_emb
        
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        # 最终线性层和softmax
        output = self.fc_out(dec_output)
        return output