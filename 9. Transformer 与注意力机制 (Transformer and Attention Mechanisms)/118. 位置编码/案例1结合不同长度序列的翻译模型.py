class TranslationModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, max_length=5000):
        super(TranslationModel, self).__init__()
        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer组件
        self.transformer = nn.Transformer(d_model=d_model)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt):
        # 应用嵌入和位置编码
        src = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Transformer处理
        output = self.transformer(src, tgt)
        
        # 投影到词汇表
        return self.output_layer(output)