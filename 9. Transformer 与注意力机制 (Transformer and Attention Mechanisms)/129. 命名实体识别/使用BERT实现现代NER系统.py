from transformers import BertTokenizer, BertForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import classification_report

# 数据集类
class BertNERDataset(Dataset):
    def __init__(self, texts, tags, tokenizer, tag2id, max_len=128):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        word_tags = self.tags[idx]
        
        # BERT分词
        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )
        
        # 获取词到标记的映射
        word_ids = encoding.word_ids()
        
        # 对齐标签
        labels = []
        previous_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                # 特殊标记
                labels.append(-100)
            elif word_id != previous_word_id:
                # 第一个子标记
                labels.append(self.tag2id[word_tags[word_id]])
            else:
                # 非首个子标记
                # 如果是B-XXX，则变为I-XXX
                prev_tag = word_tags[word_id]
                if prev_tag.startswith("B-"):
                    labels.append(self.tag2id["I-" + prev_tag[2:]])
                else:
                    labels.append(self.tag2id[prev_tag])
            
            previous_word_id = word_id
        
        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels)
        }

# 准备数据
def prepare_data_for_bert():
    # 加载CoNLL格式数据
    train_sentences, train_tags = process_conll_data("train.txt") 
    dev_sentences, dev_tags = process_conll_data("dev.txt")
    
    # 构建标签映射
    unique_tags = set()
    for tags in train_tags + dev_tags:
        for tag in tags:
            unique_tags.add(tag)
    
    tag2id = {tag: id for id, tag in enumerate(sorted(list(unique_tags)))}
    id2tag = {id: tag for tag, id in tag2id.items()}
    
    return train_sentences, train_tags, dev_sentences, dev_tags, tag2id, id2tag

# 训练函数
def train_bert_ner(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=3):
    best_f1 = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # 将数据移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        # 打印训练损失
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
        # 评估
        f1 = evaluate_bert_ner(model, val_loader, device, id2tag)
        print(f"Validation F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            # 保存最佳模型
            torch.save(model.state_dict(), "bert_ner_best.pt")
    
    return model

# 评估函数
def evaluate_bert_ner(model, data_loader, device, id2tag):
    model.eval()
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 获取预测结果
            pred_ids = torch.argmax(logits, dim=2).cpu().numpy()
            
            # 处理每个序列
            for i, (preds, label, mask) in enumerate(zip(pred_ids, labels, attention_mask)):
                pred_list = []
                true_list = []
                
                for j, (p, l) in enumerate(zip(preds, label)):
                    if l.item() == -100:
                        continue  # 跳过特殊标记
                    
                    pred_tag = id2tag[p]
                    true_tag = id2tag[l.item()] if l.item() != -100 else "O"
                    
                    pred_list.append(pred_tag)
                    true_list.append(true_tag)
                
                predictions.append(pred_list)
                true_labels.append(true_list)
    
    # 计算指标
    report = classification_report(true_labels, predictions, digits=4)
    print(report)
    
    # 返回F1分数
    from seqeval.metrics import f1_score
    return f1_score(true_labels, predictions)

# 主函数
def main():
    # 准备数据
    train_sentences, train_tags, dev_sentences, dev_tags, tag2id, id2tag = prepare_data_for_bert()
    
    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForTokenClassification.from_pretrained(
        "bert-base-chinese", 
        num_labels=len(tag2id)
    )
    
    # 创建数据集
    train_dataset = BertNERDataset(train_sentences, train_tags, tokenizer, tag2id)
    dev_dataset = BertNERDataset(dev_sentences, dev_tags, tokenizer, tag2id)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
    
    # 准备训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * 3  # 假设3个epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )
    
    # 训练
    train_bert_ner(model, train_loader, dev_loader, optimizer, scheduler, device)
    
    # 加载最佳模型进行评估
    model.load_state_dict(torch.load("bert_ner_best.pt"))
    f1 = evaluate_bert_ner(model, dev_loader, device, id2tag)
    print(f"Best model F1: {f1:.4f}")

if __name__ == "__main__":
    main()