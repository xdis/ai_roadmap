class BertForQuestionAnswering(nn.Module):
    def __init__(self, bert_model_name):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)  # start/end
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]  # [batch, seq_len, hidden_size]
        
        # 预测开始/结束位置
        logits = self.qa_outputs(sequence_output)  # [batch, seq_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch, seq_len]
        end_logits = end_logits.squeeze(-1)  # [batch, seq_len]
        
        return start_logits, end_logits