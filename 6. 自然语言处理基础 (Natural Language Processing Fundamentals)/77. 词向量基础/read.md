# 词向量基础 (Word Embeddings)

## 1. 基础概念理解

### 什么是词向量
词向量是将单词映射到连续向量空间的技术，将文本中的词语表示为密集的实数向量，使得语义相似的词在向量空间中彼此接近。

### 为什么需要词向量
- **传统表示法的局限性**：One-hot编码将每个词表示为一个稀疏向量，无法捕捉词之间的语义关系，维度爆炸
- **分布式表示的优势**：词向量能够在低维空间中捕捉词的语义和句法特性
- **计算效率**：密集向量比稀疏向量计算效率更高

### 核心思想
- **分布式假设**：上下文相似的词，其含义也相似
- **语义映射**：在向量空间中，相似词的距离较近，不相关词的距离较远
- **语义运算**：支持向量运算，如"king" - "man" + "woman" ≈ "queen"

## 2. 技术细节探索

### 词向量生成方法

#### 基于计数的方法
- **共现矩阵**：统计词在特定上下文中出现的频率
- **SVD降维**：对高维共现矩阵进行降维
- **PMI (点互信息)**：衡量词对的相关性

#### 基于预测的方法
1. **Word2Vec**
   - **CBOW (Continuous Bag of Words)**：用上下文预测目标词
   - **Skip-gram**：用目标词预测上下文
   - **负采样技术**：提高训练效率
   - **分层Softmax**：优化大词汇量计算

2. **GloVe (Global Vectors)**
   - 结合计数和预测方法的优势
   - 基于全局词共现统计
   - 优化词向量使其点积近似于共现概率的对数

3. **FastText**
   - 扩展Word2Vec，将词表示为字符n-gram的集合
   - 能处理罕见词和词形变化

## 3. 实践与实现

### 使用预训练词向量
```python
# 使用Gensim加载预训练的Word2Vec模型
import gensim.downloader as api

# 加载预训练Word2Vec模型
word_vectors = api.load("word2vec-google-news-300")

# 查找相似词
similar_words = word_vectors.most_similar("computer", topn=5)
print(similar_words)

# 词向量运算
result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)  # 应输出类似于 [('queen', 0.7)]
```

### 训练自定义词向量
```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# 假设sentences是已分词的句子列表
sentences = [["这是", "一个", "例子"], ["这是", "另一个", "例子"]]

# 训练Word2Vec模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 保存模型
model.save("word2vec.model")

# 加载模型
loaded_model = Word2Vec.load("word2vec.model")

# 获取词向量
vector = loaded_model.wv["例子"]
```

### 评估词向量
- **语义相似度任务**：测试模型对于语义相似词的识别能力
- **词类比任务**：测试语义关系捕捉能力，如"man:woman::king:queen"
- **下游任务性能**：在具体NLP任务中的表现

## 4. 高级应用与变体

### 上下文词向量
- **ELMo**：基于双向LSTM生成的上下文相关词表示
- **BERT嵌入**：从预训练Transformer模型中提取的上下文敏感表示
- **GPT系列**：基于自回归语言模型的表示

### 多语言词向量
- **跨语言词向量对齐**：将不同语言的词向量空间对齐
- **多语言联合训练**：使用并行语料同时训练多语言词向量

### 应用场景
- **文本分类**：使用词向量作为特征进行分类
- **命名实体识别**：识别文本中的实体名称
- **机器翻译**：作为神经机器翻译的输入表示
- **文档聚类**：通过词向量聚类相似文档
- **问答系统**：计算问题与答案间的语义相似性

### 最新进展
- **句子和段落向量**：Doc2Vec、Sentence-BERT等
- **子词嵌入**：处理未登录词和形态丰富的语言
- **知识增强词向量**：结合知识图谱信息的词向量

通过掌握词向量技术，你将能够构建更强大的NLP应用，处理更复杂的语言理解任务。