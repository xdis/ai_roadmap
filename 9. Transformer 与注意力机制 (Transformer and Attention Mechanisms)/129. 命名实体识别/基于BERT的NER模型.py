from transformers import BertModel, BertTokenizer
import torch.nn as nn

class BertNER(nn.Module):
    def __init__(self, bert_model_name, num_labels, use_crf=True):
        super(BertNER, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.use_crf = use_crf
        
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_labels, batch_first=True)
            
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            if self.use_crf:
                loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
                return loss
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                # 只使用有效位置的logits计算损失
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                return loss
        else:
            if self.use_crf:
                return self.crf.decode(logits, mask=attention_mask.bool())
            else:
                return torch.argmax(logits, dim=2)