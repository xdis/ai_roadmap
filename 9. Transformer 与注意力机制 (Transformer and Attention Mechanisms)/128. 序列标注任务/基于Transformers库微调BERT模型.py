from transformers import BertTokenizer, BertForTokenClassification
from transformers import AdamW
import torch
from torch.utils.data import Dataset, DataLoader

# 数据集类
class BERTNERDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, tag2idx, max_len=128):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_tags = self.tags[idx]
        
        # 分词
        tokens = []
        labels = []
        for word, tag in zip(sentence, word_tags):
            # 处理WordPiece分词
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.tokenizer.unk_token]
            
            tokens.extend(word_tokens)
            
            # 为拆分的词标注标签(第一个子词保留原标签，其余用-100)
            labels.append(self.tag2idx[tag])
            labels.extend([-100] * (len(word_tokens) - 1))  # -100在PyTorch中会被忽略
        
        # 添加特殊标记
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        labels = [-100] + labels + [-100]  # CLS和SEP标记不参与损失计算
        
        # 转换为ID并截断/填充
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
            labels = labels[:self.max_len]
        
        # 创建attention mask
        attention_mask = [1] * len(token_ids)
        
        # 填充
        padding_length = self.max_len - len(token_ids)
        token_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        labels += [-100] * padding_length
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# 加载预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(tag2idx))

# 创建数据集和数据加载器
train_dataset = BERTNERDataset(train_sentences, train_tags, tokenizer, tag2idx)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        # 将数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# 保存模型
model.save_pretrained("./bert-chinese-ner")
tokenizer.save_pretrained("./bert-chinese-ner")