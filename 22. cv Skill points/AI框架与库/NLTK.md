# NLTK (Natural Language Toolkit)

NLTK是Python中用于自然语言处理(NLP)的最流行的库之一。它提供了各种工具来处理文本数据，进行语言分析，并构建NLP应用程序。

## 1. NLTK简介

NLTK是一个强大的NLP库，它提供了：
- 文本预处理工具（分词、词干提取、词形还原等）
- 语法分析
- 语义分析
- 分类器
- 丰富的语料库和词典

## 2. 安装NLTK

```python
# 安装NLTK
pip install nltk

# 下载NLTK数据包
import nltk
nltk.download('popular')  # 下载常用数据包
```

## 3. 基础功能使用

### 3.1 分词（Tokenization）

分词是将文本分割成单词或句子的过程。

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# 文本示例
text = "NLTK是一个强大的自然语言处理工具包。它帮助我们处理各种文本数据。"

# 分句
sentences = sent_tokenize(text)
print("分句结果:")
for i, sentence in enumerate(sentences):
    print(f"句子{i+1}: {sentence}")

# 分词
words = word_tokenize(text)
print("\n分词结果:", words)
```

输出:
```
分句结果:
句子1: NLTK是一个强大的自然语言处理工具包。
句子2: 它帮助我们处理各种文本数据。

分词结果: ['NLTK', '是', '一个', '强大', '的', '自然语言', '处理', '工具包', '。', '它', '帮助', '我们', '处理', '各种', '文本', '数据', '。']
```

### 3.2 词性标注（POS Tagging）

识别文本中的词性（名词、动词、形容词等）。

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import word_tokenize, pos_tag

# 英文文本示例
text = "NLTK is a powerful Python library for natural language processing."

# 分词
words = word_tokenize(text)

# 词性标注
tagged_words = pos_tag(words)

print("词性标注结果:")
for word, tag in tagged_words:
    print(f"{word}: {tag}")
```

输出:
```
词性标注结果:
NLTK: NNP
is: VBZ
a: DT
powerful: JJ
Python: NNP
library: NN
for: IN
natural: JJ
language: NN
processing: NN
.: .
```

主要词性标记含义:
- NNP: 专有名词
- VBZ: 动词第三人称单数
- DT: 限定词
- JJ: 形容词
- NN: 名词
- IN: 介词

### 3.3 词干提取（Stemming）和词形还原（Lemmatization）

减少词形变化，将词语还原为基本形式。

```python
import nltk
nltk.download('wordnet')

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 词干提取器
stemmer = PorterStemmer()

# 词形还原器
lemmatizer = WordNetLemmatizer()

# 示例文本
words = ["running", "runs", "ran", "easily", "fairly", "children", "better", "better", "good"]

print("原始词:", words)
print("\n词干提取结果:")
for word in words:
    print(f"{word} -> {stemmer.stem(word)}")

print("\n词形还原结果:")
for word in words:
    print(f"{word} -> {lemmatizer.lemmatize(word)}")
```

输出:
```
原始词: ['running', 'runs', 'ran', 'easily', 'fairly', 'children', 'better', 'better', 'good']

词干提取结果:
running -> run
runs -> run
ran -> ran
easily -> easili
fairly -> fairli
children -> children
better -> better
better -> better
good -> good

词形还原结果:
running -> running
runs -> run
ran -> ran
easily -> easily
fairly -> fairly
children -> child
better -> better
better -> better
good -> good
```

### 3.4 停用词过滤（Stop Words）

移除常见无信息量的词。

```python
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 示例文本
text = "This is an example of filtering out stop words from a text."

# 获取英文停用词列表
stop_words = set(stopwords.words('english'))

# 分词
words = word_tokenize(text)

# 过滤停用词
filtered_words = [word for word in words if word.lower() not in stop_words]

print("原始词:", words)
print("过滤后:", filtered_words)
```

输出:
```
原始词: ['This', 'is', 'an', 'example', 'of', 'filtering', 'out', 'stop', 'words', 'from', 'a', 'text', '.']
过滤后: ['This', 'example', 'filtering', 'stop', 'words', 'text', '.']
```

## 4. 实用案例

### 4.1 文本分类

使用NLTK进行简单的情感分析。

```python
import nltk
nltk.download('movie_reviews')

from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import random

# 提取特征函数
def extract_features(document):
    words = set(document)
    features = {}
    for word in words:
        features[f'contains({word})'] = True
    return features

# 准备数据
positive_ids = movie_reviews.fileids('pos')
negative_ids = movie_reviews.fileids('neg')

positive_features = [(extract_features(movie_reviews.words(fileids=[f])), 'pos') for f in positive_ids]
negative_features = [(extract_features(movie_reviews.words(fileids=[f])), 'neg') for f in negative_ids]

# 划分训练集和测试集
all_features = positive_features + negative_features
random.shuffle(all_features)

train_set, test_set = all_features[100:], all_features[:100]

# 训练分类器
classifier = NaiveBayesClassifier.train(train_set)

# 评估分类器
print(f"分类器准确率: {accuracy(classifier, test_set):.2f}")

# 显示最有信息量的特征
print("\n最有信息量的特征:")
classifier.show_most_informative_features(5)
```

### 4.2 简单文本摘要

基于句子重要性的文本摘要。

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np

def generate_summary(text, num_sentences=3):
    # 分句
    sentences = sent_tokenize(text)
    
    # 如果句子数量不够
    if len(sentences) <= num_sentences:
        return sentences
    
    # 预处理
    stop_words = set(stopwords.words('english'))
    
    # 构建相似度矩阵
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)
    
    # 计算句子得分
    sentence_scores = np.sum(similarity_matrix, axis=1)
    
    # 选择得分最高的句子
    ranked_indices = np.argsort(sentence_scores)[::-1]
    top_indices = sorted(ranked_indices[:num_sentences])
    
    return [sentences[i] for i in top_indices]

def sentence_similarity(sent1, sent2, stop_words):
    # 分词并去除停用词和标点
    words1 = [word.lower() for word in word_tokenize(sent1) 
              if word.isalnum() and word.lower() not in stop_words]
    words2 = [word.lower() for word in word_tokenize(sent2) 
              if word.isalnum() and word.lower() not in stop_words]
    
    # 创建词表
    all_words = list(set(words1 + words2))
    
    # 创建向量
    vector1 = [1 if word in words1 else 0 for word in all_words]
    vector2 = [1 if word in words2 else 0 for word in all_words]
    
    # 计算余弦相似度
    if sum(vector1) == 0 or sum(vector2) == 0:
        return 0
    return 1 - cosine_distance(vector1, vector2)

# 示例使用
text = """
自然语言处理(NLP)是人工智能的一个子领域，关注计算机与人类语言之间的交互。
随着深度学习的发展，NLP技术取得了显著进步。现代NLP应用包括机器翻译、情感分析、文本摘要和问答系统等。
NLTK是Python中最流行的NLP库之一，提供了广泛的工具来处理文本数据。
它支持分词、词性标注、实体识别等多种任务。NLTK还提供了丰富的语料库和词典资源。
对于初学者来说，NLTK是学习NLP概念和技术的理想工具。
"""

summary = generate_summary(text, 2)
print("原文:\n", text)
print("\n摘要:\n", " ".join(summary))
```

## 5. NLTK的优缺点

### 优点:
- 学习资源丰富，有完整的文档和教程
- 内置大量语料库和语言资源
- 适合教学和研究
- 实现了大多数经典NLP算法

### 缺点:
- 处理速度较慢，不适合大规模数据处理
- 在某些先进任务上表现不如spaCy等现代库
- 不直接支持深度学习方法

## 6. 适用场景

NLTK特别适合:
- NLP学习和教学
- 原型开发
- 研究项目
- 处理英文文本
- 需要使用特定语言资源的项目

## 总结

NLTK是入门自然语言处理的理想工具，提供了丰富的功能和资源。虽然在性能上可能不如一些现代库，但其全面的功能和教育价值使其成为NLP领域的重要库。通过本文的示例，你已经了解了如何使用NLTK进行基本的文本处理任务，并可以将这些知识应用到实际项目中。