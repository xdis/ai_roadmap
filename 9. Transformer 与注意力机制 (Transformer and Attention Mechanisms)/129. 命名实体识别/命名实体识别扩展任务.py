# 基于MRC(机器阅读理解)的嵌套NER示例
class MRCForNestedNER(nn.Module):
    def __init__(self, bert_model_name, dropout_prob=0.1):
        super(MRCForNestedNER, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)  # 开始/结束位置
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        if start_positions is not None and end_positions is not None:
            # 计算损失
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            # 预测模式
            return start_logits, end_logits