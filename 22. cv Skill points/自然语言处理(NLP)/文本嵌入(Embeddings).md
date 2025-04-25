# 文本嵌入(Embeddings)

## 1. 什么是文本嵌入

文本嵌入(Text Embeddings)是将文本(单词、短语或文档)转换成固定长度的数值向量的过程。这些向量能够捕捉文本的语义信息，使计算机能够"理解"文本的含义。

通俗来说，文本嵌入就是将人类语言翻译成机器可以理解的数字语言。

### 1.1 为什么需要文本嵌入

1. **计算机只能处理数字**：自然语言无法直接被计算机理解和处理
2. **语义表示**：向量空间中相似含义的词或文本会彼此接近
3. **降维表示**：将高维度的文本数据转换为低维度的向量
4. **机器学习输入**：为机器学习模型提供结构化的输入

### 1.2 文本嵌入的核心特性

假设"王"、"男人"、"女人"和"后"有以下嵌入向量(简化示例):

```
王 = [0.2, 0.3, 0.5]
男人 = [0.4, 0.2, 0.3]
女人 = [0.4, 0.8, 0.3]
后 = [0.2, 0.9, 0.5]
```

一个好的嵌入模型会保持以下关系:
`王 - 男人 + 女人 ≈ 后`

这个例子说明了嵌入向量可以捕捉到"王对于男人相当于后对于女人"这样的语义关系。

## 2. 文本嵌入的类型和发展

### 2.1 One-Hot编码

最基本的文本表示方法，为每个词分配一个独特的位置。

```python
def one_hot_encoding(word, vocabulary):
    """为单词创建one-hot编码向量"""
    vector = [0] * len(vocabulary)
    if word in vocabulary:
        vector[vocabulary.index(word)] = 1
    return vector

# 示例
vocabulary = ["我", "喜欢", "自然", "语言", "处理"]
encoding1 = one_hot_encoding("喜欢", vocabulary)
encoding2 = one_hot_encoding("语言", vocabulary)

print(f"'喜欢'的one-hot编码: {encoding1}")
print(f"'语言'的one-hot编码: {encoding2}")
```

**缺点**:
- 向量维度等于词汇表大小，维度灾难
- 无法表示词与词之间的相似性
- 所有单词之间的距离都相等

### 2.2 词袋模型(Bag of Words)

统计文本中每个词出现的次数。

```python
from sklearn.feature_extraction.text import CountVectorizer

# 创建文本集合
corpus = [
    "我喜欢自然语言处理",
    "自然语言处理很有趣",
    "深度学习改变了自然语言处理"
]

# 创建词袋模型
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# 查看词汇表
print("词汇表:", vectorizer.get_feature_names_out())

# 查看文档向量表示
print("文档向量:")
print(X.toarray())
```

**缺点**:
- 忽略了词序和语法
- 无法捕捉词义
- 高维稀疏向量

### 2.3 TF-IDF

在词袋模型基础上，考虑词频(TF)和逆文档频率(IDF)，降低常见词的权重，提高罕见词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 创建TF-IDF向量化器
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

# 查看文档的TF-IDF表示
print("TF-IDF文档向量:")
print(X_tfidf.toarray())
```

**缺点**:
- 仍然是基于词频的表示，没有捕捉语义
- 高维稀疏向量

### 2.4 Word2Vec

Google开发的一种神经网络模型，学习词的分布式表示，能够捕捉词的语义关系。

```python
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 加载预训练的Word2Vec模型
model = api.load("word2vec-google-news-300")  # 这会下载模型，可能需要一些时间

# 查看词向量
print("'king'的词向量(前10个元素):", model['king'][:10])

# 查找最相似的词
similar_words = model.most_similar("king", topn=5)
print("与'king'最相似的词:", similar_words)

# 词向量运算示例
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print("king - man + woman =", result)  # 应该接近'queen'

# 计算词之间的相似度
similarity = model.similarity('king', 'queen')
print("'king'和'queen'的相似度:", similarity)
```

#### 2.4.1 Word2Vec的工作原理

Word2Vec主要有两种训练模型:

1. **CBOW(Continuous Bag of Words)**: 根据上下文词预测目标词
2. **Skip-gram**: 根据目标词预测上下文词

```python
from gensim.models import Word2Vec

# 准备训练数据 - 每个句子是一个词列表
sentences = [
    ["我", "喜欢", "自然", "语言", "处理"],
    ["自然", "语言", "处理", "很", "有趣"],
    ["深度", "学习", "改变", "了", "自然", "语言", "处理"]
]

# 训练CBOW模型
cbow_model = Word2Vec(sentences, vector_size=100, window=2, min_count=1, sg=0)  # sg=0表示CBOW模型

# 训练Skip-gram模型
skipgram_model = Word2Vec(sentences, vector_size=100, window=2, min_count=1, sg=1)  # sg=1表示Skip-gram模型

# 获取词向量
print("'自然'的CBOW向量(前5个元素):", cbow_model.wv['自然'][:5])
print("'自然'的Skip-gram向量(前5个元素):", skipgram_model.wv['自然'][:5])
```

### 2.5 GloVe (Global Vectors)

Stanford开发的一种基于全局词共现统计的词嵌入方法。

```python
# 使用预训练的GloVe向量需要先下载
# 这里展示如何加载和使用GloVe向量

import numpy as np

def load_glove_vectors(file_path):
    """加载GloVe词向量"""
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            vector = np.array([float(val) for val in values[1:]])
            word_vectors[word] = vector
    return word_vectors

# 使用示例(需要下载GloVe向量文件)
# vectors = load_glove_vectors('glove.6B.100d.txt')
# print("'king'的GloVe向量(前5个元素):", vectors['king'][:5])
```

### 2.6 FastText

Facebook开发的一种考虑子词信息的词嵌入方法，能更好地处理未登录词。

```python
from gensim.models import FastText

# 训练FastText模型
fasttext_model = FastText(sentences, vector_size=100, window=2, min_count=1)

# 获取词向量
print("'自然'的FastText向量(前5个元素):", fasttext_model.wv['自然'][:5])

# FastText可以处理未登录词
# 假设"人工智能"不在训练数据中
if "人工智能" not in fasttext_model.wv:
    print("'人工智能'不在词汇表中，但FastText仍可生成向量:")
    print(fasttext_model.wv["人工智能"][:5])
```

### 2.7 基于预训练语言模型的嵌入(Transformer-based Embeddings)

最新的文本嵌入方法基于BERT、RoBERTa等预训练语言模型，能够生成上下文相关的词表示。

```python
from transformers import BertTokenizer, BertModel
import torch

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 准备输入
text = "I love natural language processing."
inputs = tokenizer(text, return_tensors="pt")

# 获取BERT嵌入
with torch.no_grad():
    outputs = model(**inputs)

# 获取[CLS]标记的嵌入，通常用作整个句子的表示
sentence_embedding = outputs.last_hidden_state[:, 0, :]
print("句子嵌入的形状:", sentence_embedding.shape)
print("句子嵌入的前5个元素:", sentence_embedding[0, :5].numpy())

# 获取每个词的上下文嵌入
token_embeddings = outputs.last_hidden_state
print("词嵌入的形状:", token_embeddings.shape)
```

#### 2.7.1 中文BERT嵌入示例

```python
# 加载中文BERT模型
chinese_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
chinese_model = BertModel.from_pretrained('bert-base-chinese')

# 准备中文输入
chinese_text = "我喜欢自然语言处理。"
chinese_inputs = chinese_tokenizer(chinese_text, return_tensors="pt")

# 获取中文BERT嵌入
with torch.no_grad():
    chinese_outputs = chinese_model(**chinese_inputs)

# 获取句子嵌入
chinese_sentence_embedding = chinese_outputs.last_hidden_state[:, 0, :]
print("中文句子嵌入的前5个元素:", chinese_sentence_embedding[0, :5].numpy())
```

### 2.8 最新的文本嵌入模型

#### 2.8.1 OpenAI的文本嵌入模型

```python
# 使用OpenAI的API获取嵌入
import openai

# 设置API密钥
# openai.api_key = "your-api-key"

def get_embedding(text, model="text-embedding-ada-002"):
    """获取文本的OpenAI嵌入"""
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    embedding = response['data'][0]['embedding']
    return embedding

# 示例
# text = "自然语言处理是计算机科学的一个重要分支"
# embedding = get_embedding(text)
# print(f"OpenAI嵌入的维度: {len(embedding)}")
# print(f"嵌入的前5个元素: {embedding[:5]}")
```

#### 2.8.2 Sentence Transformers

专门为句子嵌入优化的模型库。

```python
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 准备句子
sentences = [
    "这是第一个句子",
    "这是完全不同的句子",
    "这个句子和第一个很相似"
]

# 计算嵌入
embeddings = model.encode(sentences)

# 查看嵌入的形状
print("嵌入的形状:", embeddings.shape)

# 计算句子之间的相似度
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)
print("相似度矩阵:")
print(similarity_matrix)
```

## 3. 文本嵌入的实际应用

### 3.1 文本相似度计算

```python
def find_most_similar(query, candidates, model):
    """找到与查询最相似的文本"""
    # 编码查询和候选文本
    query_embedding = model.encode(query)
    candidate_embeddings = model.encode(candidates)
    
    # 计算相似度
    similarities = cosine_similarity(
        [query_embedding], 
        candidate_embeddings
    )[0]
    
    # 获取最相似的文本
    best_match_idx = similarities.argmax()
    best_match = candidates[best_match_idx]
    best_similarity = similarities[best_match_idx]
    
    return best_match, best_similarity, similarities

# 示例
# query = "如何学习自然语言处理"
# candidates = [
#     "自然语言处理入门指南",
#     "深度学习在计算机视觉中的应用",
#     "如何开始学习NLP和深度学习",
#     "Python编程基础教程"
# ]

# best_match, similarity, all_similarities = find_most_similar(query, candidates, model)
# print(f"查询: '{query}'")
# print(f"最佳匹配: '{best_match}' (相似度: {similarity:.4f})")
# print("所有候选的相似度:")
# for i, (candidate, sim) in enumerate(zip(candidates, all_similarities)):
#     print(f"{i+1}. '{candidate}': {sim:.4f}")
```

### 3.2 文本聚类

```python
from sklearn.cluster import KMeans

def cluster_texts(texts, model, n_clusters=3):
    """对文本进行聚类"""
    # 获取文本嵌入
    embeddings = model.encode(texts)
    
    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # 组织结果
    result = {}
    for i, cluster in enumerate(clusters):
        if cluster not in result:
            result[cluster] = []
        result[cluster].append(texts[i])
    
    return result, clusters

# 示例
# texts = [
#     "自然语言处理是AI的分支",
#     "深度学习改变了NLP",
#     "计算机视觉处理图像数据",
#     "图像识别使用CNN",
#     "Transformer用于NLP任务",
#     "目标检测是CV的关键任务"
# ]

# clusters, labels = cluster_texts(texts, model, n_clusters=2)
# print("聚类结果:")
# for cluster_id, cluster_texts in clusters.items():
#     print(f"簇 #{cluster_id}:")
#     for text in cluster_texts:
#         print(f"  - {text}")
```

### 3.3 文本分类

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def classify_texts(texts, labels, model):
    """使用嵌入进行文本分类"""
    # 获取文本嵌入
    embeddings = model.encode(texts)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    
    # 训练分类器
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # 预测和评估
    predictions = classifier.predict(X_test)
    report = classification_report(y_test, predictions)
    
    return classifier, report

# 示例
# texts = [
#     "这部电影很精彩，我非常喜欢",
#     "演员的表演太糟糕了，浪费时间",
#     "剧情有趣，值得推荐",
#     "音效很差，画面也不清晰",
#     # ...更多文本
# ]

# labels = [1, 0, 1, 0]  # 1表示积极，0表示消极

# classifier, report = classify_texts(texts, labels, model)
# print("分类报告:")
# print(report)
```

### 3.4 信息检索

```python
def create_search_index(documents, model):
    """创建搜索索引"""
    # 获取文档嵌入
    document_embeddings = model.encode(documents)
    return document_embeddings

def search(query, documents, document_embeddings, model, top_k=3):
    """搜索相关文档"""
    # 获取查询嵌入
    query_embedding = model.encode(query)
    
    # 计算相似度
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    
    # 获取最相似的文档
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "document": documents[idx],
            "similarity": similarities[idx]
        })
    
    return results

# 示例
# documents = [
#     "自然语言处理(NLP)是人工智能的一个分支",
#     "深度学习在近年来取得了巨大突破",
#     "BERT是一种基于Transformer的预训练语言模型",
#     "Word2Vec是一种流行的词嵌入技术",
#     "GPT模型使用了自回归的预训练方法"
# ]

# # 创建索引
# document_embeddings = create_search_index(documents, model)

# # 搜索
# query = "预训练语言模型"
# results = search(query, documents, document_embeddings, model)

# print(f"查询: '{query}'")
# print("搜索结果:")
# for i, result in enumerate(results):
#     print(f"{i+1}. '{result['document']}' (相似度: {result['similarity']:.4f})")
```

## 4. 评估文本嵌入质量

### 4.1 内在评估(Intrinsic Evaluation)

测量词向量在词汇语义相似性任务上的表现。

```python
def evaluate_word_analogy(model, analogies):
    """评估词类比任务"""
    correct = 0
    total = len(analogies)
    
    for a, b, c, expected in analogies:
        try:
            # 计算: a - b + c ≈ expected
            result = model.most_similar(positive=[c, a], negative=[b], topn=1)
            predicted = result[0][0]
            
            if predicted == expected:
                correct += 1
        except KeyError:
            # 如果词不在词汇表中，跳过
            total -= 1
            continue
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

# 示例
# analogies = [
#     ('king', 'man', 'woman', 'queen'),
#     ('paris', 'france', 'rome', 'italy'),
#     ('good', 'better', 'bad', 'worse'),
#     # ...更多类比
# ]

# accuracy = evaluate_word_analogy(word2vec_model, analogies)
# print(f"词类比任务准确率: {accuracy:.4f}")
```

### 4.2 外在评估(Extrinsic Evaluation)

测量嵌入在下游任务(如分类、聚类)中的表现。

```python
from sklearn.metrics import accuracy_score

def evaluate_classification(texts, labels, model):
    """评估文本分类任务"""
    # 获取文本嵌入
    embeddings = model.encode(texts)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42
    )
    
    # 训练分类器
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # 预测和评估
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy

# 示例
# texts = [...]  # 文本列表
# labels = [...]  # 标签列表

# accuracy = evaluate_classification(texts, labels, model)
# print(f"分类任务准确率: {accuracy:.4f}")
```

## 5. 实用技巧与最佳实践

### 5.1 选择合适的嵌入模型

1. **任务导向**:
   - 对于单词级任务，Word2Vec、GloVe或FastText可能足够
   - 对于句子级任务，BERT或Sentence Transformers可能更好
   - 对于长文档，考虑使用专门的文档嵌入模型

2. **资源考虑**:
   - 如果计算资源有限，选择轻量级模型(Word2Vec, FastText)
   - 高性能场景可以使用预训练的Transformer模型

3. **语言特性**:
   - 对于形态丰富的语言(如德语)，FastText可能表现更好
   - 中文等语言可能需要特殊的分词处理

### 5.2 嵌入的预处理和后处理

```python
def preprocess_and_embed(texts, model):
    """文本预处理和嵌入"""
    # 清理文本(示例)
    processed_texts = []
    for text in texts:
        # 转小写
        text = text.lower()
        # 简单的标点符号处理
        for char in ".,!?;:":
            text = text.replace(char, " " + char + " ")
        processed_texts.append(text)
    
    # 获取嵌入
    embeddings = model.encode(processed_texts)
    
    # 标准化嵌入(L2范数)
    from sklearn.preprocessing import normalize
    normalized_embeddings = normalize(embeddings)
    
    return normalized_embeddings

# PCA降维可视化
def visualize_embeddings(embeddings, labels):
    """使用PCA可视化嵌入"""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    # 降维到2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(set(labels)):
        mask = [l == label for l in labels]
        plt.scatter(
            reduced_embeddings[mask, 0], 
            reduced_embeddings[mask, 1], 
            label=f"类别 {label}"
        )
    
    plt.legend()
    plt.title("文本嵌入的PCA可视化")
    plt.xlabel("第一主成分")
    plt.ylabel("第二主成分")
    plt.grid(True)
    plt.show()
```

## 6. 总结与未来发展

### 6.1 文本嵌入的发展趋势

1. **更大的预训练模型**:
   - GPT-4, PaLM, Claude等大型语言模型提供的文本表示更加强大
   - 模型尺寸和数据量不断增长

2. **多模态嵌入**:
   - 结合文本、图像、音频等多种模态的联合嵌入
   - CLIP, DALL-E等模型展示了跨模态嵌入的潜力

3. **更高效的嵌入模型**:
   - 轻量级但高性能的嵌入模型
   - 知识蒸馏和模型压缩技术

4. **领域特定嵌入**:
   - 针对特定领域(医学、法律、金融等)优化的嵌入模型
   - 更好的多语言和跨语言嵌入

### 6.2 选择正确的嵌入方法

文本嵌入方法的选择应该基于:

1. **任务复杂性**: 简单任务可能不需要最先进的嵌入
2. **可用资源**: 考虑计算和存储限制
3. **数据规模**: 数据量小时可能需要使用预训练模型
4. **实时性要求**: 对于实时应用，可能需要选择计算效率高的方法

下面是一个简单的决策流程图:

```
开始
 |
 v
需要处理单个词或短语? 
 |     |
 | 是  v
 |    考虑Word2Vec或FastText
 |     |
 v 否  v
需要上下文相关的表示?
 |     |
 | 是  v
 |    考虑BERT或其变体
 |     |
 v 否  v
需要句子级别的表示?
 |     |
 | 是  v
 |    考虑Sentence Transformers
 |     |
 v 否  v
需要文档级别的表示?
 |     |
 | 是  v
 |    考虑Doc2Vec或句子嵌入的聚合
 |     |
 v     v
考虑其他特殊需求(如多语言、领域特定等)
```

## 7. 实际案例：构建语义搜索引擎

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticSearchEngine:
    """简单的语义搜索引擎"""
    
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        """初始化搜索引擎"""
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, documents):
        """添加文档到索引"""
        self.documents.extend(documents)
        # 计算嵌入
        embeddings = self.model.encode(documents)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(self, query, top_k=3):
        """搜索文档"""
        # 计算查询嵌入
        query_embedding = self.model.encode([query])[0]
        
        # 计算相似度
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 获取最相似的文档
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "similarity": similarities[idx]
            })
        
        return results

# 使用示例
# search_engine = SemanticSearchEngine()

# # 添加文档
# documents = [
#     "自然语言处理是人工智能的一个重要分支",
#     "机器学习算法可以从数据中学习模式",
#     "深度学习使用神经网络进行特征学习",
#     "词嵌入是NLP中的基础技术",
#     "BERT是一种强大的预训练语言模型",
#     "神经网络由多层神经元组成",
#     "Transformer架构使用自注意力机制",
#     "GPT是一种生成式预训练模型"
# ]
# search_engine.add_documents(documents)

# # 搜索
# query = "预训练语言模型"
# results = search_engine.search(query, top_k=3)

# print(f"查询: '{query}'")
# print("搜索结果:")
# for i, result in enumerate(results):
#     print(f"{i+1}. '{result['document']}' (相似度: {result['similarity']:.4f})")
```

文本嵌入是现代NLP的核心技术，为各种应用提供了强大的文本表示能力。通过选择合适的嵌入方法并正确应用，我们可以有效地处理各种自然语言处理任务。从简单的词袋模型到复杂的预训练语言模型，文本嵌入技术不断发展，为计算机理解人类语言提供了越来越强大的工具。