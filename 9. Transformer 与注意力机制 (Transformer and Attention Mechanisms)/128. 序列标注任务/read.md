# 序列标注任务：从零开始掌握

## 1. 基础概念理解

### 什么是序列标注？

**序列标注**（Sequence Labeling）是一类将标签分配给序列中每个元素的机器学习任务。与文本分类（为整个文本分配单一标签）不同，序列标注为输入序列中的每个标记（token）分配一个标签。

```
输入：  [我  爱  北  京  天  安  门]
标签：  [O  O  B-LOC I-LOC I-LOC I-LOC I-LOC]
```

### 常见序列标注任务

1. **命名实体识别（NER）**：识别文本中的实体（人名、地名、组织等）
   ```
   乔布斯创立了苹果公司
   [B-PER I-PER O O B-ORG I-ORG]
   ```

2. **词性标注（POS Tagging）**：标识每个词的词性
   ```
   我 喜欢 看 电影
   [PN V V NN]
   ```

3. **分块（Chunking）**：识别短语结构
   ```
   [他]NP [正在]ADVP [研究]VP [自然语言处理]NP
   ```

4. **语义角色标注（SRL）**：标识谓词和相关论元
   ```
   [小明]A0 [送给]V [小红]A2 [一本书]A1
   ```

### 标注方案（Tagging Schemes）

为了表示实体边界和类型，常用的标注方案包括：

1. **IO标注法**：
   - I-X：属于实体X的标记
   - O：不属于任何实体的标记

2. **BIO标注法**（最常用）：
   - B-X：实体X的起始标记
   - I-X：实体X的内部标记
   - O：不属于任何实体的标记

3. **BIOES标注法**：
   - B-X：实体X的起始标记
   - I-X：实体X的内部标记
   - E-X：实体X的结束标记
   - S-X：单个标记构成的实体X
   - O：不属于任何实体的标记

### 序列标注的核心挑战

1. **上下文依赖性**：标签取决于周围的标记
2. **标签间依赖性**：标签序列需要遵循一定的约束（如I-X标签通常不会直接跟在O后面）
3. **实体边界模糊**：确定实体的精确边界往往很困难
4. **数据稀疏性**：某些实体类型样本可能很少

### 评估指标

序列标注通常使用以下指标评估：

- **精确率（Precision）**：预测为正例的样本中真正例的比例
- **召回率（Recall）**：真正例中被正确预测的比例
- **F1分数**：精确率和召回率的调和平均
- **实体级评估**：只有当实体的边界和类型都正确时才被视为正确预测

## 2. 技术细节探索

### 传统序列标注方法

#### 隐马尔可夫模型（HMM）

HMM基于马尔可夫假设，认为当前状态只与前一个状态有关：

```
P(y₁, y₂, ..., yₙ | x₁, x₂, ..., xₙ) 
```

核心参数包括：
- 转移概率：P(yᵢ | yᵢ₋₁)
- 发射概率：P(xᵢ | yᵢ)

使用Viterbi算法进行解码，时间复杂度为O(n·k²)，其中n是序列长度，k是标签数量。

```python
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # 保存最佳路径概率
    path = {}  # 保存最佳路径
    
    # 初始化
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
    
    # 递推
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        
        for y in states:
            # 选择最大概率的前导状态
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            
            V[t][y] = prob
            newpath[y] = path[state] + [y]
            
        path = newpath
    
    # 找出最优路径
    (prob, state) = max((V[len(obs) - 1][y], y) for y in states)
    return path[state]
```

#### 条件随机场（CRF）

CRF是一种判别式模型，建模条件概率P(y|x)，避免了HMM的独立性假设：

```
P(y₁, y₂, ..., yₙ | x₁, x₂, ..., xₙ) ∝ exp(∑ᵢ ∑ⱼ λⱼfⱼ(yᵢ₋₁, yᵢ, x, i))
```

其中fⱼ是特征函数，λⱼ是对应的权重。

CRF优势：
- 可以使用丰富的特征
- 考虑了全局最优标注序列
- 解决了标注偏置问题

```python
# 使用sklearn-crfsuite的CRF实现示例
import sklearn_crfsuite

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
    }
    # 添加前后词特征...
    return features

# 提取特征
X_train = [[word2features(s, i) for i in range(len(s))] for s in train_data]
y_train = [[label for _, label in s] for s in train_data]

# 训练CRF模型
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)
```

### 深度学习方法

#### BiLSTM-CRF模型

结合了双向LSTM和CRF的优势：
- **BiLSTM**：捕获长距离上下文信息
- **CRF**：对标签序列建模，保证输出的一致性

架构：
1. 嵌入层：将文本标记转换为向量
2. BiLSTM层：提取上下文相关特征
3. CRF层：预测最优标签序列

```python
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                           num_layers=1, bidirectional=True)
        
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size)
        
    def _get_lstm_features(self, sentence):
        embeds = self.word_embeds(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def forward(self, sentence, tags=None, mask=None):
        emissions = self._get_lstm_features(sentence)
        
        if tags is not None:
            # 训练模式
            return -self.crf(emissions, tags, mask=mask, reduction='mean')
        else:
            # 预测模式
            return self.crf.decode(emissions, mask=mask)
```

#### 基于Transformer的序列标注

利用预训练语言模型（如BERT）进行序列标注：

1. **特征提取**：使用BERT等模型获取上下文敏感的词表示
2. **分类层**：在BERT输出上添加分类头（通常是线性层）
3. **微调**：针对具体序列标注任务进行微调

```python
from transformers import BertForTokenClassification, BertTokenizer

# 加载预训练模型
model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(tag2idx))
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 处理输入
text = "我爱北京天安门"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
inputs = torch.tensor([token_ids])

# 预测
outputs = model(inputs)
predictions = torch.argmax(outputs.logits, dim=2)
predicted_tags = [idx2tag[p.item()] for p in predictions[0]]
```

### 标注一致性实现

使用诸如CRF这样的算法来保证输出序列的一致性，使标签序列符合约束：
- 约束1：I-标签不能跟随O标签
- 约束2：I-X标签不能跟随B-Y或I-Y（当X≠Y时）

CRF编码这些约束通过转移概率矩阵来实现：

```python
# 转移矩阵示例 (简化版)
transitions = torch.full((num_tags, num_tags), -10000.0)
# 允许的转换
transitions[tag2idx['O'], tag2idx['O']] = 0.0
transitions[tag2idx['O'], tag2idx['B-PER']] = 0.0
transitions[tag2idx['B-PER'], tag2idx['I-PER']] = 0.0
# 不允许从O到I-X
transitions[tag2idx['O'], tag2idx['I-PER']] = -10000.0
```

## 3. 实践与实现

### 数据预处理

#### 数据格式

序列标注数据通常以列格式（CoNLL格式）存储：

```
我 O
爱 O
北 B-LOC
京 I-LOC
天 I-LOC
安 I-LOC
门 I-LOC

他 O
在 O
深 B-LOC
圳 I-LOC
```

#### 数据加载与处理

```python
def load_data(filename):
    """加载CoNLL格式数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sentences = []
    tags = []
    sentence = []
    tag = []
    
    for line in lines:
        line = line.strip()
        if line:
            word, label = line.split()
            sentence.append(word)
            tag.append(label)
        elif sentence:  # 空行表示句子边界
            sentences.append(sentence)
            tags.append(tag)
            sentence = []
            tag = []
    
    # 处理最后一个句子
    if sentence:
        sentences.append(sentence)
        tags.append(tag)
    
    return sentences, tags

# 构建词汇表和标签映射
def build_vocab(sentences, tags):
    word_to_ix = {}
    tag_to_ix = {}
    
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    
    for tag_seq in tags:
        for tag in tag_seq:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    
    return word_to_ix, tag_to_ix
```

### 使用BiLSTM-CRF实现NER

完整的BiLSTM-CRF命名实体识别系统：

```python
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
```

### 使用BERT实现序列标注

基于Transformers库微调BERT模型：

```python
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
```

### 实体级评估实现

使用seqeval库进行实体级评估：

```python
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# 真实标签和预测标签
true_tags = [
    ['O', 'O', 'B-PER', 'I-PER', 'O'],
    ['B-ORG', 'I-ORG', 'O', 'B-LOC', 'I-LOC']
]
pred_tags = [
    ['O', 'O', 'B-PER', 'I-PER', 'O'],
    ['B-ORG', 'I-ORG', 'O', 'B-PER', 'I-PER']  # LOC预测为PER
]

# 计算指标
p = precision_score(true_tags, pred_tags)
r = recall_score(true_tags, pred_tags)
f1 = f1_score(true_tags, pred_tags)

print(f"Precision: {p:.4f}")
print(f"Recall: {r:.4f}")
print(f"F1-score: {f1:.4f}")

# 详细报告
report = classification_report(true_tags, pred_tags)
print(report)
```

## 4. 高级应用与变体

### 多任务序列标注

同时进行多种序列标注任务（如NER和POS标注）：

```python
class MultiTaskSequenceLabeling(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, ner_tag_size, pos_tag_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, bidirectional=True, batch_first=True)
        
        # 任务特定输出层
        self.ner_classifier = nn.Linear(hidden_dim, ner_tag_size)
        self.pos_classifier = nn.Linear(hidden_dim, pos_tag_size)
        
        # 任务特定CRF层
        self.ner_crf = CRF(ner_tag_size)
        self.pos_crf = CRF(pos_tag_size)
        
    def forward(self, x, ner_tags=None, pos_tags=None, mask=None):
        # 共享表示
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        
        # NER任务
        ner_emissions = self.ner_classifier(lstm_out)
        
        # POS任务
        pos_emissions = self.pos_classifier(lstm_out)
        
        # 如果是训练模式
        if ner_tags is not None and pos_tags is not None:
            # 计算损失
            ner_loss = -self.ner_crf(ner_emissions, ner_tags, mask=mask, reduction='mean')
            pos_loss = -self.pos_crf(pos_emissions, pos_tags, mask=mask, reduction='mean')
            
            # 总损失为各任务损失的加权和
            total_loss = ner_loss + pos_loss
            return total_loss
        else:
            # 预测模式
            ner_predictions = self.ner_crf.decode(ner_emissions, mask=mask)
            pos_predictions = self.pos_crf.decode(pos_emissions, mask=mask)
            return ner_predictions, pos_predictions
```

### 嵌套实体识别

处理嵌套实体（一个实体包含另一个实体）的方法：

1. **多层序列标注**：为每个实体类型训练单独的模型
2. **跨度表示方法**：预测所有可能的实体跨度

```python
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
```

### 低资源场景下的序列标注

在标注数据有限时的技术：

1. **数据增强**：
   - 实体替换：替换文本中现有实体以创建新样本
   - 同义词替换：替换非实体词汇以增加多样性

```python
def augment_data(sentences, tags, entity_dict):
    """使用实体替换进行数据增强"""
    augmented_sentences = []
    augmented_tags = []
    
    for sentence, tag_seq in zip(sentences, tags):
        # 查找实体位置
        entities = []
        current_entity = {"start": -1, "end": -1, "type": ""}
        
        for i, tag in enumerate(tag_seq):
            if tag.startswith("B-"):
                if current_entity["start"] != -1:
                    entities.append(current_entity.copy())
                current_entity = {"start": i, "end": i, "type": tag[2:]}
            elif tag.startswith("I-") and current_entity["start"] != -1:
                if tag[2:] == current_entity["type"]:
                    current_entity["end"] = i
            elif current_entity["start"] != -1:
                entities.append(current_entity.copy())
                current_entity = {"start": -1, "end": -1, "type": ""}
        
        if current_entity["start"] != -1:
            entities.append(current_entity)
        
        # 对每个实体执行替换
        for entity in entities:
            if entity["type"] in entity_dict:
                replacements = entity_dict[entity["type"]]
                for replacement in replacements:
                    # 创建新句子和标签序列
                    new_sentence = sentence.copy()
                    new_tags = tag_seq.copy()
                    
                    # 替换实体
                    orig_length = entity["end"] - entity["start"] + 1
                    new_length = len(replacement)
                    
                    # 替换词语
                    new_sentence[entity["start"]:entity["start"]+1] = replacement
                    if orig_length > 1:
                        del new_sentence[entity["start"]+1:entity["start"]+orig_length]
                    
                    # 调整标签
                    new_tags[entity["start"]] = f"B-{entity['type']}"
                    for j in range(1, new_length):
                        if entity["start"]+j < len(new_tags):
                            new_tags[entity["start"]+j] = f"I-{entity['type']}"
                        else:
                            new_tags.append(f"I-{entity['type']}")
                    
                    if orig_length > 1:
                        del new_tags[entity["start"]+new_length:entity["start"]+orig_length]
                    
                    augmented_sentences.append(new_sentence)
                    augmented_tags.append(new_tags)
    
    return augmented_sentences, augmented_tags
```

2. **少样本学习**：
   - 原型网络(Prototypical Networks)
   - 模型无关元学习(Model-Agnostic Meta-Learning)

3. **迁移学习**：从相关任务或领域迁移知识

### 主动学习序列标注

减少标注成本的主动学习策略：

```python
def token_entropy_uncertainty(model, unlabeled_loader, device, n_samples=10):
    """基于标记熵的不确定度采样"""
    model.eval()
    uncertainties = []
    samples = []
    
    with torch.no_grad():
        for i, (inputs, idx) in enumerate(unlabeled_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 多次前向传播(使用dropout)
            model.train()  # 启用dropout
            logits_samples = []
            for _ in range(n_samples):
                outputs = model(**inputs)
                logits_samples.append(outputs.logits.cpu())
            
            # 计算每个标记的熵
            logits_samples = torch.stack(logits_samples)  # [n_samples, batch, seq_len, n_classes]
            mean_probs = torch.softmax(logits_samples, dim=-1).mean(dim=0)  # [batch, seq_len, n_classes]
            entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)  # [batch, seq_len]
            
            # 获取每个序列的平均熵
            masked_entropy = entropy * inputs['attention_mask'].cpu()
            seq_entropy = masked_entropy.sum(dim=1) / inputs['attention_mask'].sum(dim=1).cpu()
            
            # 保存结果
            uncertainties.extend(seq_entropy.tolist())
            samples.extend(idx.tolist())
    
    # 按不确定度排序
    sorted_indices = [idx for _, idx in sorted(zip(uncertainties, samples), reverse=True)]
    return sorted_indices
```

### 跨语言序列标注

利用多语言预训练模型进行跨语言序列标注：

```python
from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification

# 加载多语言模型
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base', num_labels=len(tag2idx))

# 在源语言(如英语)训练
# ...训练代码...

# 在目标语言(如中文)测试
chinese_text = "我去了北京和上海"
inputs = tokenizer(chinese_text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

# 解码预测结果
pred_tags = [idx2tag[p.item()] for p in predictions[0][1:-1]]  # 去除特殊标记
```

### 基于Prompt的序列标注

将序列标注转换为生成任务：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 加载模型
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# 将NER任务转化为生成任务
def create_prompt(text):
    return f"Find entities in text: {text}"

def parse_generated_entities(generated_text):
    # 解析生成的实体标注结果
    entities = []
    for line in generated_text.split('\n'):
        if ':' in line:
            try:
                entity_type, entity_text = line.split(':', 1)
                entities.append({
                    'type': entity_type.strip(),
                    'text': entity_text.strip()
                })
            except:
                continue
    return entities

# 使用示例
text = "苹果公司由史蒂夫·乔布斯于1976年创建。"
prompt = create_prompt(text)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=150)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 生成结果示例: "ORGANIZATION: 苹果公司\nPERSON: 史蒂夫·乔布斯\nDATE: 1976年"
entities = parse_generated_entities(generated_text)
print(entities)
```

### 基于图神经网络的序列标注

利用句法依存树等图结构信息：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphNeuralNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        
        # 图卷积层
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # 输出层
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)  # 结合LSTM和GCN特征
        
    def forward(self, x, edge_index, batch_idx):
        # LSTM特征提取
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        
        # 图卷积特征提取
        gcn_out = F.relu(self.gcn1(lstm_out, edge_index))
        gcn_out = self.gcn2(gcn_out, edge_index)
        
        # 特征融合
        combined_features = torch.cat([lstm_out, gcn_out], dim=-1)
        
        # 分类
        logits = self.classifier(combined_features)
        return logits
```

## 总结与最佳实践

### 选择合适的序列标注模型

1. **任务复杂度**：
   - 简单任务：传统CRF或BiLSTM-CRF
   - 复杂任务：基于Transformer的模型

2. **数据规模**：
   - 小型数据集：BiLSTM-CRF + 预训练词嵌入
   - 中等规模：微调BERT等预训练模型
   - 大规模：自定义Transformer架构

3. **计算资源**：
   - 有限资源：压缩版预训练模型(DistilBERT等)
   - 充足资源：全尺寸预训练模型

### 序列标注优化技巧

1. **数据预处理**：
   - 仔细处理子词分割与标签对齐
   - 考虑使用字符级表示用于中文等语言

2. **特征工程**：
   - 结合词法、句法特征
   - 使用外部知识库(词典等)

3. **训练技巧**：
   - 学习率预热和衰减
   - 梯度裁剪防止梯度爆炸
   - 标签平滑减轻过拟合

4. **集成和后处理**：
   - 模型集成提高稳定性
   - 规则后处理纠正常见错误

### 实用性建议

1. **处理边界情况**：
   - 实体跨句子边界
   - 不平衡标签分布
   - 长序列处理

2. **领域适应**：
   - 使用领域特定词表和嵌入
   - 领域适应训练技术

3. **部署优化**：
   - 模型压缩和量化
   - 批处理以提高吞吐量

4. **质量监控**：
   - 实体级别错误分析
   - 持续评估和更新

序列标注是NLP领域的基础技术，掌握其核心原理和实现方法对于构建高质量的文本分析系统至关重要。从本文介绍的基础概念到高级应用，相信您已经对序列标注有了全面而深入的理解，能够应对各种实际应用场景中的序列标注任务。

Similar code found with 3 license types