# spaCy 库完全指南

## 1. 基础概念理解

### 什么是spaCy
spaCy是一个现代化的自然语言处理(NLP)Python库，专为生产环境设计，注重效率和性能。它由Matthew Honnibal和Ines Montani于2015年创建，并由Explosion AI公司维护。spaCy提供了一套完整的NLP功能，适用于信息提取、自然语言理解、预处理文本以及深度学习模型的NLP功能。

### spaCy的核心特点
- **高效性能**：使用Cython开发的核心组件，处理速度快
- **准确性**：采用最先进的统计模型，提供高准确度的语言分析
- **生产就绪**：为实际应用场景优化，而非仅用于研究或教学
- **深度学习集成**：与PyTorch、TensorFlow等无缝协作
- **多语言支持**：提供多种语言的预训练模型

### spaCy与NLTK的比较
| 特性 | spaCy | NLTK |
|------|-------|------|
| 设计理念 | 生产环境应用 | 教学和研究 |
| 性能 | 更快(Cython加速) | 较慢(纯Python) |
| 易用性 | 面向对象API，简洁 | 功能性API，较分散 |
| 模型选择 | 少而精，预训练模型 | 多种算法实现 |
| 自定义性 | 强(可定制管道) | 强(更学术化) |
| 学习曲线 | 中等 | 较陡峭 |

### spaCy的架构
spaCy采用管道(pipeline)架构，文本通过一系列处理组件依次处理：

1. **Tokenizer**：将文本分割成词元(tokens)
2. **Tagger**：分配词性标签(POS)
3. **Parser**：确定句法依存关系
4. **NER**：识别命名实体
5. **Lemmatizer**：提取词元的基本形式
6. **Additional components**：可选自定义组件

这种设计允许灵活的管道配置和高效的处理流程。

### 安装和设置

```python
# 基本安装
pip install spacy

# 下载语言模型(英语)
# 小型模型
python -m spacy download en_core_web_sm

# 中型模型(更精确但更大)
python -m spacy download en_core_web_md

# 大型模型(最精确，包含词向量)
python -m spacy download en_core_web_lg

# 基本使用
import spacy

# 加载模型
nlp = spacy.load('en_core_web_sm')

# 处理文本
doc = nlp("spaCy is a modern NLP library with amazing features!")

# 查看结果
for token in doc:
    print(token.text, token.pos_, token.dep_)
```

## 2. 技术细节探索

### 核心数据结构

#### Doc对象
Doc对象是spaCy处理文本的核心结构，代表一个已处理的文档。

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("This is a sentence about Apple Inc.")

# Doc属性
print(f"文本: {doc.text}")
print(f"Token数量: {len(doc)}")
print(f"命名实体: {doc.ents}")
```

#### Token对象
Token代表文档中的单个标记(单词、标点等)，包含丰富的语言学属性。

```python
# Token属性
for token in doc:
    print(f"\nToken: {token.text}")
    print(f"  词形还原: {token.lemma_}")
    print(f"  词性标注: {token.pos_}")
    print(f"  精细词性: {token.tag_}")
    print(f"  是否停用词: {token.is_stop}")
    print(f"  是否标点: {token.is_punct}")
    print(f"  句法依存: {token.dep_}")
    print(f"  所在句子: {token.sent}")
```

#### Span对象
Span表示Doc中的一个连续序列，如短语或句子。

```python
# 创建Span
apple_span = doc[5:7]  # "Apple Inc."
print(f"Span文本: {apple_span.text}")
print(f"Span开始: {apple_span.start}")
print(f"Span结束: {apple_span.end}")
print(f"Span根节点: {apple_span.root.text}")

# 遍历句子(每个句子是一个Span)
for sent in doc.sents:
    print(f"句子: {sent.text}")
```

### 语言处理组件详解

#### 1. 分词器(Tokenizer)
spaCy的分词非常智能，能处理复杂的语言现象。

```python
doc = nlp("Let's explore spaCy's tokenization!")

print("标记:")
for token in doc:
    print(f"[{token.text}]", end=" ")

# 查看分词细节
print("\n\n标记属性:")
for token in doc:
    print(f"{token.text:{15}} | 是否字母: {token.is_alpha:{5}} | "
          f"是否标点: {token.is_punct:{5}} | 是否空格: {token.is_space:{5}}")
```

#### 2. 词性标注(POS Tagging)
spaCy使用统计模型给每个Token分配词性标签。

```python
doc = nlp("The quick brown fox jumps over the lazy dog.")

print("词性标注:")
for token in doc:
    print(f"{token.text:{10}} | {token.pos_:{6}} | {token.tag_:{6}} | {spacy.explain(token.tag_)}")
```

#### 3. 依存句法分析(Dependency Parsing)
确定词语间的语法关系，构建句法树。

```python
doc = nlp("The artificial intelligence researcher developed a new algorithm.")

print("依存句法关系:")
for token in doc:
    print(f"{token.text:{12}} → {token.head.text:{12}} | {token.dep_:{10}} | {spacy.explain(token.dep_)}")

# 可视化依存树
from spacy import displacy
displacy.render(doc, style="dep", jupyter=True)  # 在Jupyter中显示
# 或保存为HTML
html = displacy.render(doc, style="dep", page=True)
with open("dependency_tree.html", "w", encoding="utf-8") as f:
    f.write(html)
```

#### 4. 命名实体识别(NER)
识别文本中的命名实体，如人名、组织、地点等。

```python
doc = nlp("Apple Inc. is planning to open a new store in San Francisco next month for $10 million.")

print("命名实体:")
for ent in doc.ents:
    print(f"实体: {ent.text:{20}} | 标签: {ent.label_:{5}} | 解释: {spacy.explain(ent.label_)}")

# 可视化实体
displacy.render(doc, style="ent", jupyter=True)
```

#### 5. 词向量和语义相似性
spaCy的中型和大型模型包含预训练词向量，可用于计算语义相似性。

```python
# 注意：需要使用带词向量的模型，如en_core_web_md或en_core_web_lg
nlp = spacy.load('en_core_web_md')

# 计算词语相似性
tokens = nlp("dog cat banana")
for token1 in tokens:
    for token2 in tokens:
        print(f"{token1.text} <-> {token2.text}: {token1.similarity(token2):.3f}")

# 计算文档相似性
doc1 = nlp("I like machine learning and NLP")
doc2 = nlp("I enjoy deep learning and natural language processing")
similarity = doc1.similarity(doc2)
print(f"文档相似度: {similarity:.3f}")
```

#### 6. 规则匹配(Rule-based Matching)
使用灵活的模式匹配系统在文本中查找特定模式。

```python
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# 定义匹配模式 - 查找形容词+名词的组合
pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}]
matcher.add("ADJ_NOUN_PATTERN", [pattern])

doc = nlp("I bought a new laptop and an expensive smartphone.")
matches = matcher(doc)

print("规则匹配结果:")
for match_id, start, end in matches:
    span = doc[start:end]
    print(f"匹配: {span.text} | 规则: {nlp.vocab.strings[match_id]}")
```

### 多语言支持
spaCy支持多种语言，每种语言都有专门的预训练模型。

```python
# 加载中文模型
# 首先安装: python -m spacy download zh_core_web_sm
nlp_zh = spacy.load('zh_core_web_sm')
doc_zh = nlp_zh("北京是中国的首都。")

# 加载德语模型
# 首先安装: python -m spacy download de_core_news_sm
nlp_de = spacy.load('de_core_news_sm')
doc_de = nlp_de("Berlin ist die Hauptstadt von Deutschland.")

# 显示结果
print("中文处理:")
for token in doc_zh:
    print(f"{token.text:{5}} | {token.pos_:{5}} | {token.dep_}")

print("\n德语处理:")
for token in doc_de:
    print(f"{token.text:{10}} | {token.pos_:{5}} | {token.dep_}")
```

## 3. 实践与实现

### 构建完整NLP处理流水线

```python
import spacy
from spacy.language import Language

# 创建自定义组件
@Language.component("custom_component")
def custom_processing(doc):
    # 对每个Token添加自定义属性
    for token in doc:
        # 示例：添加是否是自定义关键词的属性
        token._.set("is_keyword", token.text.lower() in ["ai", "nlp", "python", "spacy"])
    return doc

# 注册Token扩展属性
from spacy.tokens import Token
Token.set_extension("is_keyword", default=False)

# 创建流水线
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("custom_component", after="parser")

# 显示管道组件
print("流水线组件:", nlp.pipe_names)

# 处理文本
text = "SpaCy is an advanced Python library for NLP and AI applications."
doc = nlp(text)

# 检查结果
print("\n处理结果:")
for token in doc:
    if token._.is_keyword:
        print(f"关键词: {token.text}")
```

### 自定义实体识别系统

```python
import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher

# 创建一个自定义的产品实体识别器
def create_product_recognizer():
    nlp = spacy.load("en_core_web_sm")
    
    # 注册自定义实体类型
    if "PRODUCT" not in nlp.pipe_labels["ner"]:
        nlp.get_pipe("ner").add_label("PRODUCT")
    
    # 创建短语匹配器
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    # 添加产品名称
    product_names = ["iphone", "macbook pro", "airpods", 
                     "galaxy s21", "surface pro", "playstation 5"]
    patterns = [nlp.make_doc(name) for name in product_names]
    matcher.add("PRODUCTS", patterns)
    
    # 自定义组件添加到管道
    @Language.component("product_recognizer")
    def product_recognition_component(doc):
        matches = matcher(doc)
        spans = []
        for match_id, start, end in matches:
            # 创建Span并标记为PRODUCT实体
            span = Span(doc, start, end, label="PRODUCT")
            spans.append(span)
        
        # 如果存在重叠，只保留最长的匹配
        if spans:
            doc.ents = list(doc.ents) + spacy.util.filter_spans(spans)
        return doc
    
    # 添加到管道
    nlp.add_pipe("product_recognizer", after="ner")
    return nlp

# 创建并测试产品识别器
product_nlp = create_product_recognizer()
test_text = "I just bought a new iPhone and I love it. My friend has a Galaxy S21."
doc = product_nlp(test_text)

print("识别的产品实体:")
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")
```

### 文本分类实现

```python
import spacy
from spacy.training import Example
import random

# 创建文本分类器
def train_text_classifier(train_data, labels, iterations=20):
    # 创建空模型
    nlp = spacy.blank("en")
    
    # 创建文本分类器
    textcat = nlp.add_pipe("textcat")
    
    # 添加标签
    for label in labels:
        textcat.add_label(label)
    
    # 准备训练数据
    train_examples = []
    for text, annotations in train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        train_examples.append(example)
    
    # 训练模型
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        print("开始训练...")
        for i in range(iterations):
            losses = {}
            random.shuffle(train_examples)
            for example in train_examples:
                nlp.update([example], sgd=optimizer, losses=losses)
            print(f"迭代 {i+1}: 损失 {losses['textcat']:.3f}")
    
    return nlp

# 简单数据集 - 新闻分类示例
train_data = [
    ("Google unveiled a new smartphone with advanced AI features", {"cats": {"TECH": 1.0, "SPORTS": 0.0, "POLITICS": 0.0}}),
    ("The tech giant released new software update for their operating system", {"cats": {"TECH": 1.0, "SPORTS": 0.0, "POLITICS": 0.0}}),
    ("The football team won the championship after an intense match", {"cats": {"TECH": 0.0, "SPORTS": 1.0, "POLITICS": 0.0}}),
    ("The athlete broke the world record in the recent Olympics", {"cats": {"TECH": 0.0, "SPORTS": 1.0, "POLITICS": 0.0}}),
    ("The president signed a new executive order on climate change", {"cats": {"TECH": 0.0, "SPORTS": 0.0, "POLITICS": 1.0}}),
    ("The election results surprised many political analysts", {"cats": {"TECH": 0.0, "SPORTS": 0.0, "POLITICS": 1.0}})
]

labels = ["TECH", "SPORTS", "POLITICS"]
classifier = train_text_classifier(train_data, labels)

# 测试分类器
test_texts = [
    "Apple announced their newest iPhone model yesterday",
    "The basketball tournament ended with an upset victory",
    "The senate debated the new tax policy last week"
]

print("\n分类结果:")
for text in test_texts:
    doc = classifier(text)
    print(f"\n文本: {text}")
    scores = doc.cats
    for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {score:.3f}")
```

### 信息提取系统

```python
import spacy
from spacy.matcher import Matcher

# 创建信息提取系统
def create_info_extractor():
    nlp = spacy.load("en_core_web_sm")
    
    # 创建匹配器
    matcher = Matcher(nlp.vocab)
    
    # 电子邮件模式
    email_pattern = [
        {"LIKE_EMAIL": True}
    ]
    
    # 电话号码模式(简化版)
    # 真实场景中可能需要更复杂的正则表达式
    phone_pattern = [
        {"SHAPE": "ddd-ddd-dddd"}
    ]
    
    # 日期模式(简化版)
    date_pattern = [
        {"IS_DIGIT": True}, {"TEXT": "/"}, {"IS_DIGIT": True}, {"TEXT": "/"}, {"IS_DIGIT": True}
    ]
    
    # 添加模式
    matcher.add("EMAIL", [email_pattern])
    matcher.add("PHONE", [phone_pattern])
    matcher.add("DATE", [date_pattern])
    
    # 信息提取函数
    def extract_info(text):
        doc = nlp(text)
        matches = matcher(doc)
        
        # 提取匹配项
        extracted_info = []
        for match_id, start, end in matches:
            match_type = nlp.vocab.strings[match_id]
            span = doc[start:end]
            extracted_info.append((match_type, span.text))
        
        # 提取标准命名实体
        for ent in doc.ents:
            extracted_info.append((ent.label_, ent.text))
        
        return extracted_info
    
    return extract_info

# 创建并测试提取器
extractor = create_info_extractor()
text = """
John Smith can be reached at john.smith@example.com or 555-123-4567. 
He works for Acme Corporation in New York and has a meeting on 12/25/2023.
His budget is $5000 for the project.
"""

info = extractor(text)
print("提取的信息:")
for info_type, value in info:
    print(f"{info_type}: {value}")
```

### 文本相似性分析系统

```python
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 创建文本相似度分析器
def create_similarity_analyzer():
    # 加载带有词向量的模型
    nlp = spacy.load("en_core_web_md")
    
    def compute_similarity(texts, reference=None):
        """计算文本之间的相似度"""
        # 处理所有文本
        docs = list(nlp.pipe(texts))
        
        # 如果指定了参考文本，计算每个文本与参考文本的相似度
        if reference:
            ref_doc = nlp(reference)
            similarities = [ref_doc.similarity(doc) for doc in docs]
            return pd.DataFrame({
                'text': texts,
                'similarity_to_reference': similarities
            }).sort_values('similarity_to_reference', ascending=False)
        
        # 否则计算所有文本对之间的相似度
        else:
            # 获取文档向量
            vectors = np.array([doc.vector for doc in docs])
            similarity_matrix = cosine_similarity(vectors)
            
            # 创建相似度矩阵DataFrame
            df = pd.DataFrame(similarity_matrix, index=texts, columns=texts)
            return df
    
    return compute_similarity

# 创建并测试相似度分析器
similarity_analyzer = create_similarity_analyzer()

documents = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers.",
    "Natural language processing is a branch of AI that helps computers understand human language.",
    "Python is a popular programming language used for AI and data science projects.",
    "Artificial intelligence aims to create systems capable of performing tasks that require human intelligence."
]

# 计算所有文档间的相似度
similarity_matrix = similarity_analyzer(documents)
print("文档相似度矩阵:")
print(similarity_matrix)

# 与参考文档比较
reference = "Artificial intelligence and machine learning are transforming how we analyze data."
ranked_similarity = similarity_analyzer(documents, reference)
print("\n与参考文档的相似度排名:")
print(ranked_similarity)
```

## 4. 高级应用与变体

### 与深度学习框架集成

```python
import spacy
import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from spacy.tokens import DocBin

# 使用spaCy进行预处理并输入到PyTorch模型
class SpacyTorchPipeline:
    def __init__(self, batch_size=32):
        # 加载spaCy模型
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # PyTorch相关设置
        self.vocab = None
        self.model = None
        
    def preprocess_with_spacy(self, texts):
        """使用spaCy处理文本并返回DocBin"""
        docs = DocBin()
        for doc in self.nlp.pipe(texts, batch_size=self.batch_size):
            # 过滤停用词，仅保留有意义的词
            filtered_doc = [token for token in doc if not token.is_stop and not token.is_punct]
            if filtered_doc:  # 确保文档不为空
                doc_obj = self.nlp.make_doc(" ".join([token.text for token in filtered_doc]))
                docs.add(doc_obj)
        return docs
    
    def texts_to_indices(self, texts):
        """将文本转换为索引序列"""
        docs = list(self.nlp.pipe(texts, batch_size=self.batch_size))
        
        # 转换为索引，并填充到相同长度
        indices = []
        for doc in docs:
            # 过滤并索引化
            filtered_tokens = [token.text.lower() for token in doc 
                              if not token.is_stop and not token.is_punct]
            indices.append([self.vocab[token] for token in filtered_tokens])
        
        # 填充序列
        max_len = max(len(seq) for seq in indices)
        padded_indices = [seq + [0] * (max_len - len(seq)) for seq in indices]
        
        return torch.tensor(padded_indices, device=self.device)
    
    def build_vocab(self, texts):
        """构建词汇表"""
        # 使用spaCy进行标记化
        docs = list(self.nlp.pipe(texts, batch_size=self.batch_size))
        
        # 生成词汇表迭代器
        def yield_tokens():
            for doc in docs:
                yield [token.text.lower() for token in doc 
                      if not token.is_stop and not token.is_punct]
        
        # 构建词汇表
        self.vocab = {w: i+1 for i, w in enumerate(set(w for tokens in yield_tokens() for w in tokens))}
        self.vocab['<pad>'] = 0
        
        return self.vocab
    
    def create_simple_model(self, vocab_size, embed_dim=100, hidden_dim=256, output_dim=1):
        """创建简单的LSTM模型"""
        class LSTMClassifier(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                embedded = self.embedding(x)
                output, (hidden, _) = self.lstm(embedded)
                # 使用最后一个隐藏状态
                return self.fc(hidden.squeeze(0))
        
        self.model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim)
        self.model.to(self.device)
        return self.model
    
    def train_model(self, train_data, val_data=None, epochs=5):
        """训练模型的简化版本"""
        # 这里只是示意，实际训练会更复杂
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for texts, labels in train_data:
                # 转换为索引
                x = self.texts_to_indices(texts)
                y = torch.tensor(labels, dtype=torch.float).to(self.device)
                
                # 前向传播
                outputs = self.model(x).squeeze()
                loss = criterion(outputs, y)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.4f}")
            
            # 如果有验证集，计算验证准确率
            if val_data:
                self.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for texts, labels in val_data:
                        x = self.texts_to_indices(texts)
                        y = torch.tensor(labels, dtype=torch.float).to(self.device)
                        
                        outputs = self.model(x).squeeze()
                        predicted = (outputs > 0).float()
                        
                        total += y.size(0)
                        correct += (predicted == y).sum().item()
                
                accuracy = correct / total
                print(f"Validation Accuracy: {accuracy:.4f}")

# 注意：这是一个简化的示例，实际应用中需要更完整的实现
pipeline = SpacyTorchPipeline()
print("创建了SpaCy-PyTorch集成管道")
```

### 自定义实体链接系统

```python
import spacy
from spacy.tokens import Span
from spacy.language import Language
import pandas as pd

# 创建简单的实体链接器
def create_entity_linker():
    # 加载模型
    nlp = spacy.load("en_core_web_sm")
    
    # 创建知识库(简化版)
    knowledge_base = {
        "Apple": {
            "Q312": {
                "name": "Apple Inc.",
                "description": "American technology company",
                "type": "ORGANIZATION"
            },
            "Q89": {
                "name": "Apple (fruit)",
                "description": "Fruit of the apple tree",
                "type": "PRODUCT"
            }
        },
        "Python": {
            "Q28865": {
                "name": "Python (programming language)",
                "description": "High-level programming language",
                "type": "LANGUAGE"
            },
            "Q201": {
                "name": "Python (snake)",
                "description": "Group of non-venomous snakes",
                "type": "ANIMAL"
            }
        }
    }
    
    # 注册自定义属性
    if not Span.has_extension("kb_id"):
        Span.set_extension("kb_id", default=None)
    if not Span.has_extension("kb_info"):
        Span.set_extension("kb_info", default=None)
    
    # 创建自定义组件
    @Language.component("entity_linker")
    def entity_linking_component(doc):
        # 遍历所有命名实体
        for ent in doc.ents:
            # 检查实体是否在知识库中
            if ent.text in knowledge_base:
                candidates = knowledge_base[ent.text]
                
                # 简单消歧：根据实体类型选择候选项
                # 实际系统中会有更复杂的消歧逻辑
                for kb_id, info in candidates.items():
                    if info["type"] == ent.label_:
                        ent._.kb_id = kb_id
                        ent._.kb_info = info
                        break
                
                # 如果没有匹配的类型，使用第一个候选项
                if ent._.kb_id is None and candidates:
                    first_id = list(candidates.keys())[0]
                    ent._.kb_id = first_id
                    ent._.kb_info = candidates[first_id]
        
        return doc
    
    # 添加到管道
    nlp.add_pipe("entity_linker", after="ner")
    return nlp

# 测试实体链接器
entity_linker = create_entity_linker()
texts = [
    "Apple is developing new technology.",
    "I ate an apple for breakfast.",
    "Python is my favorite programming language.",
    "The python slithered through the grass."
]

print("实体链接结果:")
for text in texts:
    doc = entity_linker(text)
    for ent in doc.ents:
        kb_info = ent._.kb_info
        if kb_info:
            print(f"文本: '{text}'")
            print(f"实体: '{ent.text}' | 类型: {ent.label_}")
            print(f"链接到: {kb_info['name']} ({ent._.kb_id})")
            print(f"描述: {kb_info['description']}\n")
```

### 多语言文本处理系统

```python
import spacy
import pandas as pd

# 创建多语言处理系统
class MultilingualNLPProcessor:
    def __init__(self):
        # 加载多语言模型
        try:
            self.models = {
                "en": spacy.load("en_core_web_sm"),
                "es": spacy.load("es_core_news_sm"),
                "de": spacy.load("de_core_news_sm"),
                "fr": spacy.load("fr_core_news_sm"),
                "zh": spacy.load("zh_core_web_sm")
            }
            self.supported_langs = list(self.models.keys())
        except OSError as e:
            print(f"错误: 无法加载所有语言模型。{e}")
            print("请确保已安装所需模型: python -m spacy download [model_name]")
            self.models = {}
            self.supported_langs = []
    
    def detect_language(self, text):
        """简单的语言检测(实际应用中可能需要更复杂的方法)"""
        # 这里使用一个简化的方法，基于特定字符的存在
        # 实际应用应该使用语言检测库如langdetect或fastText
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return "zh"  # 汉字范围
        elif any(c in "áéíóúüñ¿¡" for c in text):
            return "es"  # 西班牙语特殊字符
        elif any(c in "äöüß" for c in text):
            return "de"  # 德语特殊字符
        elif any(c in "éèêëàâçîïôùû" for c in text):
            return "fr"  # 法语特殊字符
        else:
            return "en"  # 默认为英语
    
    def process_text(self, text, lang=None):
        """处理指定语言的文本"""
        # 如果未指定语言，尝试检测
        if lang is None:
            lang = self.detect_language(text)
        
        # 检查是否支持该语言
        if lang not in self.models:
            print(f"警告: 不支持语言 '{lang}'。使用英语模型代替。")
            lang = "en"
        
        # 处理文本
        doc = self.models[lang](text)
        
        return {
            "language": lang,
            "text": text,
            "tokens": [token.text for token in doc],
            "pos_tags": [token.pos_ for token in doc],
            "entities": [(ent.text, ent.label_) for ent in doc.ents]
        }
    
    def batch_process(self, texts, langs=None):
        """批量处理多语言文本"""
        results = []
        
        # 如果未提供语言列表，为每个文本检测语言
        if langs is None:
            langs = [self.detect_language(text) for text in texts]
        elif len(langs) != len(texts):
            print("错误: 语言列表长度必须与文本列表长度相同")
            return results
        
        # 处理每个文本
        for text, lang in zip(texts, langs):
            results.append(self.process_text(text, lang))
        
        return pd.DataFrame(results)

# 测试多语言处理系统
multilingual_processor = MultilingualNLPProcessor()

if multilingual_processor.supported_langs:
    multilingual_texts = [
        "SpaCy is an advanced NLP library.",
        "El procesamiento del lenguaje natural es fascinante.",
        "Die Verarbeitung natürlicher Sprache ist ein spannendes Forschungsgebiet.",
        "Le traitement du langage naturel est un domaine important de l'IA.",
        "自然语言处理是人工智能的重要分支。"
    ]
    
    results = multilingual_processor.batch_process(multilingual_texts)
    print("多语言处理结果:")
    print(results[["language", "text"]])
    
    # 显示第一个文本的详细信息
    print("\n英语文本的详细分析:")
    first_result = results.iloc[0]
    print(f"文本: {first_result['text']}")
    print(f"语言: {first_result['language']}")
    
    # 显示标记和词性
    tokens = first_result['tokens']
    pos_tags = first_result['pos_tags']
    for token, pos in zip(tokens, pos_tags):
        print(f"{token}: {pos}")
    
    # 显示实体
    print("\n识别的实体:")
    for entity, label in first_result['entities']:
        print(f"{entity}: {label}")
else:
    print("未能加载任何语言模型。请检查模型安装。")
```

### 大规模文本处理优化

```python
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
import time
import multiprocessing
import os

# spaCy大规模文本处理优化
class OptimizedTextProcessor:
    def __init__(self, model="en_core_web_sm", batch_size=1000,
                 n_process=None, disable=None):
        # 设置进程数
        if n_process is None:
            n_process = max(1, multiprocessing.cpu_count() - 1)
        
        # 加载模型
        print(f"加载spaCy模型 '{model}'...")
        self.nlp = spacy.load(model, disable=disable)
        
        # 配置
        self.batch_size = batch_size
        self.n_process = n_process
    
    def _process_texts(self, texts):
        """内部处理方法"""
        return list(self.nlp.pipe(texts, batch_size=self.batch_size))
    
    def process_large_dataset(self, texts, output_dir=None):
        """处理大型文本数据集"""
        start_time = time.time()
        print(f"开始处理 {len(texts)} 文本, 批量大小={self.batch_size}, 进程数={self.n_process}")
        
        # 多进程处理
        docs = []
        if self.n_process > 1:
            partitions = [texts[i:i+self.batch_size] 
                          for i in range(0, len(texts), self.batch_size)]
            
            pool = multiprocessing.Pool(processes=self.n_process)
            
            # 使用进度条显示处理进度
            all_docs = []
            for batch_docs in tqdm(pool.imap(self._process_texts, partitions), 
                                   total=len(partitions),
                                   desc="处理批次"):
                all_docs.extend(batch_docs)
            
            pool.close()
            pool.join()
            docs = all_docs
        else:
            # 单进程处理
            for doc in tqdm(self.nlp.pipe(texts, batch_size=self.batch_size),
                           total=len(texts), desc="处理文本"):
                docs.append(doc)
        
        # 如果指定了输出目录，保存结果
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 保存为二进制格式
            doc_bin = DocBin(attrs=["LEMMA", "POS", "DEP", "ENT_IOB", "ENT_TYPE"])
            for doc in docs:
                doc_bin.add(doc)
            
            output_path = os.path.join(output_dir, "processed_docs.spacy")
            doc_bin.to_disk(output_path)
            print(f"保存处理结果到 {output_path}")
        
        total_time = time.time() - start_time
        print(f"处理完成，耗时 {total_time:.2f} 秒")
        print(f"每秒处理 {len(texts)/total_time:.2f} 文本")
        
        return docs
    
    def extract_entities(self, docs):
        """从处理后的文档中提取实体"""
        entities = []
        for doc in docs:
            doc_entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                           for ent in doc.ents]
            entities.append(doc_entities)
        return entities
    
    def extract_noun_chunks(self, docs):
        """提取名词短语"""
        noun_chunks = []
        for doc in docs:
            chunks = [(chunk.text, chunk.root.text, chunk.root.pos_,
                      chunk.root.dep_) for chunk in doc.noun_chunks]
            noun_chunks.append(chunks)
        return noun_chunks

# 示例使用
text_processor = OptimizedTextProcessor(disable=["parser"])
sample_texts = ["This is text " + str(i) for i in range(100)]  # 简单示例
print("处理示例文本...")
docs = text_processor.process_large_dataset(sample_texts)
print(f"提取 {len(docs)} 文档中的实体...")
entities = text_processor.extract_entities(docs)
print("完成")
```

### spaCy管道组合与定制

```python
import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler
import json

# 创建复杂的自定义spaCy管道
def create_custom_pipeline():
    # 基础模型
    nlp = spacy.load("en_core_web_sm")
    
    # 1. 添加自定义分词规则
    infixes = nlp.Defaults.infixes + [r'(?<=[0-9])\.(?=[0-9])']  # 处理小数点
    infix_regex = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_regex.finditer
    
    # 2. 自定义实体规则
    entity_ruler = EntityRuler(nlp)
    patterns = [
        {"label": "ORG", "pattern": "DeepMind"},
        {"label": "PRODUCT", "pattern": "GPT-4"},
        {"label": "PRODUCT", "pattern": "ChatGPT"},
        {"label": "TECH", "pattern": "transformer"},
        {"label": "TECH", "pattern": "deep learning"}
    ]
    entity_ruler.add_patterns(patterns)
    
    # 在NER之前添加实体规则
    if "entity_ruler" not in nlp.pipe_names:
        nlp.add_pipe("entity_ruler", before="ner")
    
    # 3. 自定义后处理组件
    @Language.component("text_categorizer")
    def text_categorizer(doc):
        # 简单的主题分类器
        tech_keywords = ["AI", "algorithm", "computer", "software", "programming", 
                         "technology", "data", "model", "neural", "network"]
        business_keywords = ["company", "market", "business", "finance", "economic", 
                            "invest", "stock", "profit", "revenue", "corporate"]
        science_keywords = ["research", "science", "physics", "chemistry", "biology", 
                           "experiment", "theory", "scientific", "study", "discovery"]
        
        # 计算关键词匹配
        text_lower = doc.text.lower()
        tech_count = sum(1 for word in tech_keywords if word.lower() in text_lower)
        business_count = sum(1 for word in business_keywords if word.lower() in text_lower)
        science_count = sum(1 for word in science_keywords if word.lower() in text_lower)
        
        # 确定分类
        counts = {"TECH": tech_count, "BUSINESS": business_count, "SCIENCE": science_count}
        doc._.category = max(counts.items(), key=lambda x: x[1])[0] if any(counts.values()) else "OTHER"
        doc._.category_scores = {category: count/sum(counts.values()) if sum(counts.values()) > 0 else 0 
                                for category, count in counts.items()}
        
        return doc
    
    # 4. 添加自定义扩展属性
    Doc.set_extension("category", default="")
    Doc.set_extension("category_scores", default={})
    
    # 5. 添加自定义组件到管道
    if "text_categorizer" not in nlp.pipe_names:
        nlp.add_pipe("text_categorizer", last=True)
    
    # 6. 文档汇总组件
    @Language.component("document_summarizer")
    def document_summarizer(doc):
        # 简单提取式摘要 - 选取重要句子
        # 这里只是一个简化版本
        sentences = list(doc.sents)
        if sentences:
            # 基于命名实体数量的简单评分
            scores = []
            for sent in sentences:
                # 计算句子中包含的命名实体数量
                ent_count = sum(1 for _ in doc.ents if sent.start <= _.start < sent.end)
                # 简单的长度归一化
                score = ent_count / len(sent)
                scores.append((sent, score))
            
            # 选取评分最高的句子作为摘要
            sorted_sents = sorted(scores, key=lambda x: x[1], reverse=True)
            top_sentences = [sent.text for sent, _ in sorted_sents[:2]]  # 取前两句
            
            doc._.summary = " ".join(top_sentences)
        else:
            doc._.summary = ""
        
        return doc
    
    # 注册摘要属性
    Doc.set_extension("summary", default="")
    
    # 添加摘要组件
    if "document_summarizer" not in nlp.pipe_names:
        nlp.add_pipe("document_summarizer", last=True)
    
    # 显示完整管道
    print("定制管道组件:", nlp.pipe_names)
    
    return nlp

# 测试自定义管道
custom_nlp = create_custom_pipeline()

test_text = """
DeepMind researchers have published a new paper on transformer models for AI. 
The technology uses deep learning algorithms to process large amounts of data.
GPT-4 is the latest language model developed by OpenAI, showing significant improvements over ChatGPT.
This research could revolutionize how companies approach natural language processing tasks.
"""

doc = custom_nlp(test_text)

# 显示处理结果
print("\n处理结果:")
print(f"文本分类: {doc._.category}")
print(f"分类置信度: {json.dumps(doc._.category_scores, indent=2)}")
print(f"文档摘要: {doc._.summary}")

print("\n识别的实体:")
for ent in doc.ents:
    print(f"  {ent.text} ({ent.label_})")

print("\n词法分析:")
for token in list(doc)[:10]:  # 只显示前10个标记
    print(f"  {token.text:{15}} | {token.pos_:{6}} | {token.lemma_:{10}} | {token.dep_}")
```

spaCy是一个强大的现代NLP库，通过本指南的学习，你已经了解了从基础到高级的全面内容。你可以使用spaCy构建各种NLP应用，包括信息提取、文本分类、实体链接等。库的高效性和灵活性使其成为生产环境中的理想选择，而其与深度学习框架的集成能力，让你能够构建更复杂、更强大的NLP系统。