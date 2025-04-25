class MultiTaskSequenceLabeling(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, ner_tag_size, pos_tag_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, bidirectional=True, batch_first=True)
        
        # 任务特定输出层
        self.ner_classifier = nn.Linear(hidden_dim, ner_tag_size)
        self.pos_classifier = nn.Linear(hidden_dim, pos_tag_size)
        
        # 任务特定CRF层
        self.ner_crf = CRF(ner_tag_size)
        self.pos_crf = CRF(pos_tag_size)
        
    def forward(self, x, ner_tags=None, pos_tags=None, mask=None):
        # 共享表示
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        
        # NER任务
        ner_emissions = self.ner_classifier(lstm_out)
        
        # POS任务
        pos_emissions = self.pos_classifier(lstm_out)
        
        # 如果是训练模式
        if ner_tags is not None and pos_tags is not None:
            # 计算损失
            ner_loss = -self.ner_crf(ner_emissions, ner_tags, mask=mask, reduction='mean')
            pos_loss = -self.pos_crf(pos_emissions, pos_tags, mask=mask, reduction='mean')
            
            # 总损失为各任务损失的加权和
            total_loss = ner_loss + pos_loss
            return total_loss
        else:
            # 预测模式
            ner_predictions = self.ner_crf.decode(ner_emissions, mask=mask)
            pos_predictions = self.pos_crf.decode(pos_emissions, mask=mask)
            return ner_predictions, pos_predictions