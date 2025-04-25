# 跨度表示方法示例(简化版)
class SpanBasedNER(nn.Module):
    def __init__(self, encoder, num_entity_types):
        super().__init__()
        self.encoder = encoder  # BERT或BiLSTM编码器
        self.start_classifier = nn.Linear(encoder.hidden_size, 1)  # 起始位置分类器
        self.end_classifier = nn.Linear(encoder.hidden_size, 1)    # 结束位置分类器
        self.span_classifier = nn.Linear(encoder.hidden_size * 2, num_entity_types)  # 跨度分类器
        
    def forward(self, input_ids, attention_mask):
        # 获取上下文表示
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # 预测开始和结束位置
        start_logits = self.start_classifier(sequence_output).squeeze(-1)  # [batch, seq_len]
        end_logits = self.end_classifier(sequence_output).squeeze(-1)      # [batch, seq_len]
        
        # 为每个可能的跨度生成表示并分类
        batch_size, seq_len, hidden_size = sequence_output.shape
        span_logits = []
        
        for batch_idx in range(batch_size):
            batch_spans = []
            for start_idx in range(seq_len):
                for end_idx in range(start_idx, min(start_idx + 10, seq_len)):  # 限制跨度长度
                    # 获取跨度表示(简单连接起始和结束表示)
                    span_repr = torch.cat([
                        sequence_output[batch_idx, start_idx],
                        sequence_output[batch_idx, end_idx]
                    ])
                    # 对跨度进行分类
                    span_logit = self.span_classifier(span_repr)
                    batch_spans.append((start_idx, end_idx, span_logit))
            span_logits.append(batch_spans)
            
        return start_logits, end_logits, span_logits