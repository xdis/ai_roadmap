class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length=5000, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
    def forward(self, x):
        """
        x: [batch_size, seq_len] - 输入标记的整数索引
        """
        # 创建词嵌入，并缩放
        token_embeddings = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # 添加位置编码
        embeddings = self.position_encoding(token_embeddings)
        
        # 应用dropout
        return self.dropout(embeddings)