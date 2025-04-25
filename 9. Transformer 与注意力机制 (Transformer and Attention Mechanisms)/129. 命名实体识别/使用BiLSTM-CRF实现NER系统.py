import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# 为了简化，使用torchcrf包
from torchcrf import CRF

# 数据集类
class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_to_idx, tag_to_idx):
        self.sentences = sentences
        self.tags = tags
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag = self.tags[idx]
        
        # 转换为索引
        sentence_idx = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in sentence]
        tag_idx = [self.tag_to_idx[t] for t in tag]
        
        return torch.tensor(sentence_idx), torch.tensor(tag_idx)

# 定义模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_idx, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx)
        
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                           num_layers=1, bidirectional=True, batch_first=True)
                           
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)
        
    def forward(self, sentence, tags=None, mask=None):
        embeds = self.word_embeds(sentence)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:
            # 训练模式
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # 预测模式
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions

# 数据整理函数
def collate_fn(batch):
    sentences, tags = zip(*batch)
    
    # 填充
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
    
    # 计算mask
    mask = (sentences_padded != 0).float()
    
    return sentences_padded, tags_padded, mask

# 训练函数
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for sentences, tags, mask in train_loader:
        sentences = sentences.to(device)
        tags = tags.to(device) 
        mask = mask.to(device)
        
        # 前向传播
        loss = model(sentences, tags, mask)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# 评估函数
def evaluate(model, data_loader, tag_to_idx, idx_to_tag, device):
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for sentences, tags, mask in data_loader:
            sentences = sentences.to(device)
            mask = mask.to(device)
            
            # 预测
            predictions = model(sentences, mask=mask)
            
            # 处理每个样本
            for i, (pred, true, m) in enumerate(zip(predictions, tags, mask)):
                length = int(m.sum().item())
                
                # 转换为标签
                pred_tags = [idx_to_tag[p] for p in pred[:length]]
                true_tags = [idx_to_tag[t.item()] for t in true[:length]]
                
                y_pred.append(pred_tags)
                y_true.append(true_tags)
    
    # 计算指标
    from seqeval.metrics import classification_report
    report = classification_report(y_true, y_pred)
    print(report)
    
    # 计算F1分数
    from seqeval.metrics import f1_score
    return f1_score(y_true, y_pred)

# 主函数
def main():
    # 加载数据
    train_sentences, train_labels = process_conll_data("train.txt")
    dev_sentences, dev_labels = process_conll_data("dev.txt")
    
    # 构建词汇表和标签映射
    word_to_idx = build_vocab(train_sentences)
    tag_to_idx, idx_to_tag = build_tag_map(train_labels)
    
    # 创建数据集
    train_dataset = NERDataset(train_sentences, train_labels, word_to_idx, tag_to_idx)
    dev_dataset = NERDataset(dev_sentences, dev_labels, word_to_idx, tag_to_idx)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, embedding_dim=100, hidden_dim=200).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}, loss: {train_loss:.4f}")
        
        # 评估
        f1 = evaluate(model, dev_loader, tag_to_idx, idx_to_tag, device)
        print(f"Dev F1: {f1:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "bilstm_crf_ner.pt")