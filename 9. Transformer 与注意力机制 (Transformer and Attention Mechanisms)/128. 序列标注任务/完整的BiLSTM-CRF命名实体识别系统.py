import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 数据集类
class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.tags = tags
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag = self.tags[idx]
        
        # 转换为索引
        sentence_ix = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w in sentence]
        tag_ix = [self.tag_to_ix[t] for t in tag]
        
        return torch.tensor(sentence_ix), torch.tensor(tag_ix)

# 准备批处理数据
def collate_fn(batch):
    sentences, tags = zip(*batch)
    # 获取句子长度
    lengths = [len(s) for s in sentences]
    max_len = max(lengths)
    
    # 填充句子和标签
    padded_sentences = torch.zeros(len(sentences), max_len, dtype=torch.long)
    padded_tags = torch.zeros(len(tags), max_len, dtype=torch.long)
    mask = torch.zeros(len(sentences), max_len, dtype=torch.bool)
    
    for i, (sentence, tag) in enumerate(zip(sentences, tags)):
        padded_sentences[i, :len(sentence)] = sentence
        padded_tags[i, :len(tag)] = tag
        mask[i, :len(sentence)] = 1
    
    return padded_sentences, padded_tags, mask

# 训练函数
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for sentences, tags, mask in train_loader:
        sentences = sentences.to(device)
        tags = tags.to(device)
        mask = mask.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        loss = model(sentences, tags, mask)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# 评估函数
def evaluate(model, data_loader, tag_to_ix, device):
    model.eval()
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    
    true_tags_all = []
    pred_tags_all = []
    
    with torch.no_grad():
        for sentences, tags, mask in data_loader:
            sentences = sentences.to(device)
            tags = tags.to(device)
            mask = mask.to(device)
            
            # 预测
            pred_tags = model(sentences, mask=mask)
            
            # 转换预测标签
            batch_size, seq_len = sentences.size()
            for i in range(batch_size):
                length = mask[i].sum().item()
                true_tags = [ix_to_tag[tag.item()] for tag in tags[i][:length]]
                pred_tag_seq = [ix_to_tag[tag] for tag in pred_tags[i][:length]]
                
                true_tags_all.append(true_tags)
                pred_tags_all.append(pred_tag_seq)
    
    # 计算指标
    from seqeval.metrics import classification_report
    report = classification_report(true_tags_all, pred_tags_all)
    return report

# 主函数
def main():
    # 加载数据
    train_sentences, train_tags = load_data("train.txt")
    dev_sentences, dev_tags = load_data("dev.txt")
    
    # 构建词汇表
    word_to_ix, tag_to_ix = build_vocab(train_sentences, train_tags)
    word_to_ix["<PAD>"] = len(word_to_ix)
    word_to_ix["<UNK>"] = len(word_to_ix)
    
    # 创建数据加载器
    train_dataset = NERDataset(train_sentences, train_tags, word_to_ix, tag_to_ix)
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
    
    dev_dataset = NERDataset(dev_sentences, dev_tags, word_to_ix, tag_to_ix)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, embedding_dim=100, hidden_dim=200).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        
        # 每个epoch评估一次
        report = evaluate(model, dev_loader, tag_to_ix, device)
        print(report)
    
    # 保存模型
    torch.save(model.state_dict(), "bilstm_crf_ner.pt")