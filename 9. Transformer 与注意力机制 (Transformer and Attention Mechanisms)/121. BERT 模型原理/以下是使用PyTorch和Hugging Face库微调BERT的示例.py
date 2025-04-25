import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW  # 从 PyTorch 导入 AdamW 而不是 transformers

class BertForSequenceClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)  # 768 是BERT-base的隐藏层维度
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 用[CLS]标记的表示作为整个序列的表示
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 分类层
        logits = self.classifier(pooled_output)
        return logits

# 训练函数
def train_bert_classifier(model, train_dataloader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, token_type_ids)
            
            # 计算损失
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch: {epoch+1}, Loss: {total_loss/len(train_dataloader)}")