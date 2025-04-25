# Word2Vec 完全指南

## 1. 基础概念理解

### 什么是Word2Vec
Word2Vec是一种用于生成词向量的经典神经网络模型，由Google团队于2013年提出，它通过浅层神经网络从大规模文本语料中学习词的分布式表示。

### Word2Vec的核心思想
- **上下文预测**：基于"一个词的含义由其上下文决定"的语言学假设
- **低维密集表示**：将词映射到通常为50-300维的连续向量空间
- **语义相似性捕捉**：相似含义的词在向量空间中距离较近

### Word2Vec的两种架构
1. **CBOW (Continuous Bag of Words)**
   - 用上下文词预测中心词
   - 给定窗口内的周围词，预测中心词的概率
   - 适合较小的数据集，能生成更平滑的词向量

2. **Skip-gram**
   - 用中心词预测上下文词
   - 给定中心词，预测其周围词的概率
   - 对罕见词表现更好，适合大型语料

## 2. 技术细节探索

### 模型架构详解

#### CBOW架构
- **输入层**：上下文词的one-hot向量
- **投影层**：将上下文词向量平均
- **输出层**：预测中心词的softmax概率分布
- **数学表达**：最大化 P(w_t | w_{t-n}, ..., w_{t-1}, w_{t+1}, ..., w_{t+n})

#### Skip-gram架构
- **输入层**：中心词的one-hot向量
- **投影层**：中心词的词向量
- **输出层**：预测各上下文词的softmax概率分布
- **数学表达**：最大化 P(w_{t-n}, ..., w_{t-1}, w_{t+1}, ..., w_{t+n} | w_t)

### 训练优化技术

#### 1. 负采样 (Negative Sampling)
- **原理**：不使用完整softmax，而只更新正样本和少量负样本
- **公式**：log(σ(v'_{w_O}·v_{w_I})) + Σ[E_i~P(w)][log(σ(-v'_{w_i}·v_{w_I}))]
- **优势**：大幅减少计算量，使模型能处理大规模词汇表

#### 2. 分层Softmax (Hierarchical Softmax)
- **原理**：用二叉树结构替代传统softmax，将预测转化为路径判断问题
- **实现**：词汇表构建为Huffman树，频繁词获得更短路径
- **计算复杂度**：从O(V)降低到O(log V)，V为词汇表大小

#### 3. 子采样 (Subsampling)
- **原理**：以一定概率丢弃高频词，减少训练样本中的不平衡
- **公式**：P(w_i) = 1 - √(t/f(w_i))，t为阈值，f(w_i)为词频

## 3. 实践与实现

### 使用Gensim训练Word2Vec模型

```python
import gensim
from gensim.models import Word2Vec
import multiprocessing

# 准备训练数据（已分词的句子列表）
sentences = [['我', '爱', '自然语言', '处理'], 
             ['Word2Vec', '是', '词', '嵌入', '技术']]

# 训练模型
model = Word2Vec(
    sentences,
    vector_size=100,    # 词向量维度
    window=5,           # 上下文窗口大小
    min_count=1,        # 词出现最小次数
    sg=1,               # 1为Skip-gram, 0为CBOW
    hs=0,               # 0使用负采样, 1使用分层softmax
    negative=5,         # 负采样样本数
    ns_exponent=0.75,   # 负采样分布指数
    alpha=0.025,        # 初始学习率
    min_alpha=0.0001,   # 最终学习率
    seed=42,            # 随机种子
    workers=multiprocessing.cpu_count() # 并行线程数
)

# 保存模型
model.save("word2vec.model")

# 加载模型
loaded_model = Word2Vec.load("word2vec.model")

# 获取词向量
vector = model.wv['自然语言']

# 计算相似度
similarity = model.wv.similarity('词', '嵌入')

# 查找最相似词
similar_words = model.wv.most_similar('处理', topn=5)

# 词向量运算示例
result = model.wv.most_similar(positive=['女人', '国王'], negative=['男人'])
```

### 使用预训练模型

```python
import gensim.downloader as api

# 查看可用的预训练模型
print(list(api.info()['models'].keys()))

# 加载预训练模型
word_vectors = api.load("word2vec-google-news-300")

# 使用预训练向量进行词汇相似性任务
result = word_vectors.most_similar('computer')
print(result)
```

### 词向量可视化

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_with_labels(low_dim_embs, labels, filename='word2vec.png'):
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                    textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)
    plt.show()

# 获取前n个词的词向量
word_vectors = [model.wv[word] for word in model.wv.index_to_key[:100]]
words = model.wv.index_to_key[:100]

# 降维可视化
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(word_vectors)

# 绘制词向量图
plot_with_labels(embeddings_2d, words)
```

## 4. 高级应用与变体

### Word2Vec的扩展

#### Doc2Vec (Paragraph Vectors)
- **原理**：对Word2Vec的扩展，学习句子、段落或文档的向量表示
- **模型变体**：
  - **PV-DM**：结合文档向量和上下文词向量预测目标词
  - **PV-DBOW**：仅使用文档向量预测文档中的随机词

#### FastText
- **核心改进**：将每个词表示为字符n-gram的集合
- **优势**：
  - 能处理未登录词（OOV问题）
  - 对形态丰富的语言效果更好
  - 能学习词的内部结构

#### Retrofitting & Counter-fitting
- **思路**：将词汇知识（如同义词、反义词关系）融入预训练词向量
- **实现**：通过后处理步骤优化现有词向量，使语义关系更准确

### 工业级应用

1. **搜索引擎优化**
   - 语义搜索：理解查询意图
   - 查询扩展：基于语义相似性扩展关键词

2. **推荐系统**
   - 内容表示：将文本内容映射为向量
   - 语义匹配：基于内容语义相似性推荐

3. **情感分析**
   - 特征提取：使用词向量作为文本特征
   - 情感词典扩展：找到语义相似的情感词

4. **机器翻译**
   - 跨语言词向量空间对齐
   - 作为神经机器翻译模型的输入表示

### 使用Word2Vec的实用技巧

1. **数据预处理策略**
   - 词规范化：小写转换、词形还原
   - 多词短语识别：识别"New_York"等复合词
   - 领域特定预处理：去除特定领域噪声

2. **超参数调优**
   - **向量维度**：100-300之间，取决于词汇表大小和数据量
   - **窗口大小**：小窗口(2-5)捕获句法关系，大窗口(5-15)捕获语义关系
   - **负样本数**：一般设为5-20，数据量大时取大值

3. **领域适应**
   - 在通用预训练向量基础上进行领域微调
   - 混合通用语料与专业语料训练

### 局限性与挑战

1. **多义词问题**：无法区分同形异义词（如"bank"可表示"银行"或"河岸"）
2. **上下文独立**：词向量与上下文无关，每个词只有一个表示
3. **语法关系捕捉有限**：主要捕捉词的语义相似性，对复杂语法关系表达有限
4. **需要大量数据**：在小数据集上容易过拟合

### 与现代技术的关系
尽管BERT、GPT等上下文敏感模型逐渐成为主流，Word2Vec因其简单高效仍有广泛应用，特别是在资源受限场景、作为复杂模型的初始化权重或基线模型。

通过掌握Word2Vec，你已经建立了理解现代NLP嵌入技术的坚实基础，并获得了解决实际问题的强大工具。

Similar code found with 1 license type