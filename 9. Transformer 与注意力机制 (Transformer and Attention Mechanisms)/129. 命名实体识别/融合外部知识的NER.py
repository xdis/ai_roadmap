class KnowledgeEnhancedNER(nn.Module):
    def __init__(self, bert_model_name, num_labels, entity_embedding_dim, entity_vocab_size):
        super(KnowledgeEnhancedNER, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 实体嵌入层
        self.entity_embeddings = nn.Embedding(entity_vocab_size, entity_embedding_dim)
        
        # 融合层
        self.fusion = nn.Linear(
            self.bert.config.hidden_size + entity_embedding_dim, 
            self.bert.config.hidden_size
        )
        
        # 分类层
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, entity_ids=None, labels=None):
        # BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch, seq_len, hidden]
        
        if entity_ids is not None:
            # 获取实体嵌入
            entity_embeds = self.entity_embeddings(entity_ids)  # [batch, seq_len, entity_dim]
            
            # 融合两种表示
            combined_output = torch.cat([sequence_output, entity_embeds], dim=-1)
            sequence_output = self.fusion(combined_output)
        
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits