# 文本预处理：从零掌握这一自然语言处理核心技术

## 1. 基础概念理解

### 什么是文本预处理？

**文本预处理**是将原始文本数据转换为机器学习算法可以处理的格式的过程。这是自然语言处理(NLP)管道中的第一步，也是最关键的步骤之一。原始文本通常包含噪音、不规则格式、拼写错误等问题，预处理的目的是清理和标准化这些文本数据。

### 为什么文本预处理如此重要？

1. **提高模型性能**：干净、标准化的数据可以显著提升模型准确性和效率
2. **减少噪声影响**：过滤无意义内容，使模型专注于有用信息
3. **降低计算成本**：减小数据维度，提高处理效率
4. **提高可解释性**：标准化数据使分析结果更容易理解

### 文本预处理的基本流程

![文本预处理流程](https://i.imgur.com/WfNjnrQ.png)

一个完整的文本预处理管道通常包括以下步骤：

1. **文本收集和导入**：从各种来源获取文本数据
2. **文本清洗**：删除或替换不需要的字符、标记和噪声
3. **文本规范化**：将不同表达形式的相同内容转换为统一格式
4. **分词**：将文本分割成单词或子词单元
5. **停用词移除**：过滤常见但信息量少的词（如"the"、"is"、"and"等）
6. **词干提取/词形还原**：将词转换为其基本形式
7. **特征提取/向量化**：将文本转换为数值表示
8. **特征选择/降维**：选择最相关的特征或减少特征维度

## 2. 技术细节探索

### 2.1 文本清洗

文本清洗是移除或替换文本中不需要的部分的过程。

#### 常见清洗操作

1. **删除HTML标签**：网络文本常包含HTML标记，需要去除

```python
import re

def remove_html_tags(text):
    """移除HTML标签"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

html_text = "<p>这是<b>一个</b>例子</p>"
print(remove_html_tags(html_text))  # 输出: 这是一个例子
```

2. **删除特殊字符和数字**：根据需要保留或移除

```python
def remove_special_chars(text, remove_digits=False):
    """移除特殊字符和可选移除数字"""
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

text = "Hello! This is a sample text with numbers (123) and symbols #@%."
print(remove_special_chars(text))  # 保留数字
print(remove_special_chars(text, remove_digits=True))  # 移除数字
```

3. **大小写转换**：通常转为小写以减少词汇量

```python
def convert_to_lowercase(text):
    """转换为小写"""
    return text.lower()

text = "Text Preprocessing IS Important"
print(convert_to_lowercase(text))  # 输出: text preprocessing is important
```

4. **删除多余空格**：标准化空白字符

```python
def remove_extra_spaces(text):
    """删除多余空格"""
    return re.sub(r'\s+', ' ', text).strip()

text = "Too    many    spaces    here."
print(remove_extra_spaces(text))  # 输出: Too many spaces here.
```

5. **删除URL和电子邮件地址**：

```python
def remove_urls_emails(text):
    """移除URL和电子邮件地址"""
    # URL模式
    url_pattern = r'https?://\S+|www\.\S+'
    # 电子邮件模式
    email_pattern = r'\S+@\S+'
    
    text = re.sub(url_pattern, '', text)
    text = re.sub(email_pattern, '', text)
    return text

text = "访问 https://example.com 或联系 user@example.com 获取帮助"
print(remove_urls_emails(text))  # 输出: 访问  或联系  获取帮助
```

### 2.2 文本规范化

文本规范化是将不同形式的相同单词转换为标准形式的过程。

#### 词干提取(Stemming)

词干提取通过删除词缀来获取单词的词干，这是一种基于规则的快速但不精确的方法。

```python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

def compare_stemmers(words):
    """比较不同的词干提取器"""
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer('english')
    
    print("原始词\t\tPorter\t\tLancaster\tSnowball")
    print("-" * 60)
    
    for word in words:
        print(f"{word}\t\t{porter.stem(word)}\t\t{lancaster.stem(word)}\t\t{snowball.stem(word)}")

# 测试不同词干提取器
words = ["running", "runs", "runner", "ran", "easily", "fairly"]
compare_stemmers(words)
```

运行结果示例：
```
原始词          Porter          Lancaster       Snowball
------------------------------------------------------------
running         run             run             run
runs            run             run             run
runner          runner          run             runner
ran             ran             ran             ran
easily          easili          easy            easili
fairly          fairli          fair            fair
```

#### 词形还原(Lemmatization)

词形还原将单词转换为其基本词典形式（词元），通常基于词典，精确度更高但速度较慢。

```python
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def lemmatize_words(words):
    """使用WordNet词形还原器"""
    lemmatizer = WordNetLemmatizer()
    
    print("原始词\t\t名词形式\t\t动词形式")
    print("-" * 50)
    
    for word in words:
        print(f"{word}\t\t{lemmatizer.lemmatize(word)}\t\t{lemmatizer.lemmatize(word, pos='v')}")

# 测试词形还原
words = ["running", "studies", "better", "worst", "caring", "forests"]
lemmatize_words(words)
```

运行结果示例：
```
原始词          名词形式         动词形式
--------------------------------------------------
running         running         run
studies         study           study
better          better          better
worst           worst           worst
caring          caring          care
forests         forest          forest
```

#### 拼写纠正

处理错别字和拼写变体：

```python
from spellchecker import SpellChecker

def correct_spelling(text):
    """简单拼写纠正示例"""
    spell = SpellChecker()
    corrected_text = []
    
    for word in text.split():
        # 保留标点
        punctuation = ''
        if not word[-1].isalnum():
            punctuation = word[-1]
            word = word[:-1]
        
        # 纠正拼写
        corrected_word = spell.correction(word)
        if corrected_word:
            corrected_text.append(corrected_word + punctuation)
        else:
            corrected_text.append(word + punctuation)
    
    return ' '.join(corrected_text)

misspelled_text = "Thiss is ann examplle of misspeled text."
print(correct_spelling(misspelled_text))  # 输出: "This is an example of misspelled text."
```

#### 文本规范化中的特殊情况处理

1. **缩略语扩展**：将缩略语展开为完整形式

```python
def expand_contractions(text):
    """扩展英文常见缩略语"""
    contractions_dict = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "cannot", "won't": "will not", "shouldn't": "should not",
        "I'm": "I am", "you're": "you are", "he's": "he is",
        "we're": "we are", "they're": "they are", "it's": "it is"
    }
    
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    return text

text = "I don't think it's going to work if you're not careful."
print(expand_contractions(text))
# 输出: "I do not think it is going to work if you are not careful."
```

2. **文本标准化**：统一表达方式

```python
import unicodedata

def normalize_unicode(text):
    """Unicode标准化，处理特殊字符和变音符号"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

text = "Café has a résumé with naïve coördination."
print(normalize_unicode(text))  # 输出: "Cafe has a resume with naive coordination."
```

### 2.3 分词(Tokenization)

分词是将文本分割成更小单元（如单词、句子或子词）的过程。这些单元称为"标记"(tokens)。

#### 单词分词

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, WhitespaceTokenizer

def compare_word_tokenizers(text):
    """比较不同单词分词器"""
    # NLTK默认分词器
    nltk_tokens = word_tokenize(text)
    
    # Treebank分词器
    treebank_tokenizer = TreebankWordTokenizer()
    treebank_tokens = treebank_tokenizer.tokenize(text)
    
    # 空格分词器
    whitespace_tokenizer = WhitespaceTokenizer()
    whitespace_tokens = whitespace_tokenizer.tokenize(text)
    
    print("NLTK分词器:", nltk_tokens)
    print("Treebank分词器:", treebank_tokens)
    print("空格分词器:", whitespace_tokens)

text = "Hello world! This is a test, with punctuation; and symbols."
compare_word_tokenizers(text)
```

#### 句子分词

```python
from nltk.tokenize import sent_tokenize

def sentence_tokenize(text):
    """句子级分词"""
    sentences = sent_tokenize(text)
    for i, sent in enumerate(sentences, 1):
        print(f"句子 {i}: {sent}")

long_text = """自然语言处理是计算机科学的一个分支。它关注计算机与人类语言的交互。
NLP技术被广泛应用于机器翻译、文本分类等任务。这是一个快速发展的领域！"""

sentence_tokenize(long_text)
```

#### 子词分词(Subword Tokenization)

子词分词将单词分成更小的单元，适用于处理未见过的词和复合词。

```python
# 使用SentencePiece(需要安装: pip install sentencepiece)
import sentencepiece as spm

def train_and_use_sentencepiece(text_file, model_prefix, vocab_size=1000):
    """训练并使用SentencePiece分词器"""
    # 训练模型
    spm.SentencePieceTrainer.train(
        f'--input={text_file} --model_prefix={model_prefix} --vocab_size={vocab_size}'
    )
    
    # 加载模型
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    
    # 读取原始文本进行测试
    with open(text_file, 'r', encoding='utf-8') as f:
        sample_text = f.read()[:100]  # 仅使用前100个字符作为示例
    
    # 编码和解码
    encoded = sp.encode_as_pieces(sample_text)
    decoded = sp.decode(encoded)
    
    print("原文:", sample_text)
    print("分词结果:", encoded)
    print("解码后:", decoded)
    
    return sp

# 注: 需要先将文本保存到文件中
# with open('sample.txt', 'w', encoding='utf-8') as f:
#     f.write("这是用于训练子词分词器的示例文本。它应该包含足够的内容来学习常见模式。")
#
# tokenizer = train_and_use_sentencepiece('sample.txt', 'subword_model')
```

### 2.4 停用词移除

停用词是出现频率高但提供很少语义价值的词（如"the"、"a"、"is"等）。

```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_stopwords(text, language='english', custom_stopwords=None):
    """移除停用词"""
    # 获取停用词列表
    stop_words = set(stopwords.words(language))
    
    # 添加自定义停用词
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    # 分词
    word_tokens = word_tokenize(text)
    
    # 过滤停用词
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    
    return ' '.join(filtered_text)

text = "This is an example sentence demonstrating stop word removal."
print("原始文本:", text)
print("移除停用词后:", remove_stopwords(text))

# 中文示例
nltk.download('stopwords')
from nltk.corpus import stopwords

def remove_chinese_stopwords(text, custom_stopwords=None):
    """移除中文停用词"""
    # 创建中文停用词列表(示例)
    chinese_stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '们', '到', '一', '上', '这', '为', '那'}
    
    # 添加自定义停用词
    if custom_stopwords:
        chinese_stopwords.update(custom_stopwords)
    
    # 使用jieba分词(需要安装: pip install jieba)
    import jieba
    word_tokens = list(jieba.cut(text))
    
    # 过滤停用词
    filtered_text = [word for word in word_tokens if word not in chinese_stopwords]
    
    return ''.join(filtered_text)

chinese_text = "我是一个中文句子，用来演示停用词的移除效果。"
print("原始中文:", chinese_text)
print("移除停用词后:", remove_chinese_stopwords(chinese_text))
```

### 2.5 文本向量化

向量化是将文本转换为数值向量的过程，使机器学习算法能够处理文本数据。

#### 词袋模型(Bag of Words)

```python
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words_example():
    """词袋模型示例"""
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?'
    ]
    
    # 创建词袋模型
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    # 获取特征名（词汇表）
    feature_names = vectorizer.get_feature_names_out()
    
    print("词汇表:")
    print(feature_names)
    
    print("\n文档-词矩阵:")
    print(X.toarray())
    
    # 查看第一个文档的词频
    print("\n第一个文档的词频:")
    for word, count in zip(feature_names, X.toarray()[0]):
        if count > 0:
            print(f"{word}: {count}")

bag_of_words_example()
```

#### TF-IDF (词频-逆文档频率)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_example():
    """TF-IDF示例"""
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?'
    ]
    
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    # 获取特征名
    feature_names = vectorizer.get_feature_names_out()
    
    print("词汇表:")
    print(feature_names)
    
    print("\nTF-IDF矩阵:")
    print(X.toarray())
    
    # 查看第一个文档的TF-IDF值
    print("\n第一个文档的TF-IDF值:")
    for word, tfidf in zip(feature_names, X.toarray()[0]):
        if tfidf > 0:
            print(f"{word}: {tfidf:.4f}")

tfidf_example()
```

#### Word2Vec词嵌入

```python
from gensim.models import Word2Vec

def word2vec_example():
    """Word2Vec示例"""
    # 准备训练语料库
    sentences = [
        ['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
        ['this', 'is', 'the', 'second', 'sentence'],
        ['yet', 'another', 'sentence'],
        ['one', 'more', 'sentence'],
        ['and', 'the', 'final', 'sentence']
    ]
    
    # 训练Word2Vec模型
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    # 保存和加载模型
    model.save("word2vec.model")
    new_model = Word2Vec.load("word2vec.model")
    
    # 获取词向量
    vector = new_model.wv['sentence']
    print("'sentence'的词向量(前10维):")
    print(vector[:10])
    
    # 查找最相似的词
    similar_words = new_model.wv.most_similar('sentence', topn=3)
    print("\n与'sentence'最相似的词:")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")

    # 删除临时模型文件
    import os
    os.remove("word2vec.model")
    os.remove("word2vec.model.vectors.npy")

word2vec_example()
```

## 3. 实践与实现

### 3.1 完整的文本预处理管道

下面是一个集成多种预处理技术的示例：

```python
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# 下载必要的NLTK资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    """完整的文本预处理管道"""
    
    def __init__(self, language='english', remove_stopwords=True, do_lemmatization=True):
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.do_lemmatization = do_lemmatization
        self.stopwords = set(stopwords.words(language)) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if do_lemmatization else None
    
    def preprocess(self, text):
        """应用完整的预处理管道"""
        if not text or not isinstance(text, str):
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 删除URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 删除电子邮件
        text = re.sub(r'\S+@\S+', '', text)
        
        # 删除标点符号
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 删除数字
        text = re.sub(r'\d+', '', text)
        
        # 删除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 分词
        tokens = word_tokenize(text)
        
        # 移除停用词
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stopwords]
        
        # 词形还原
        if self.do_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # 重新组合为文本
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_dataframe(self, df, text_column):
        """预处理DataFrame中的文本列"""
        df['processed_text'] = df[text_column].apply(self.preprocess)
        return df

# 使用示例
preprocessor = TextPreprocessor()

# 预处理单个文本
sample_text = "Hello! This is an example text with numbers (123) and a URL: https://example.com."
processed = preprocessor.preprocess(sample_text)
print("原文:", sample_text)
print("处理后:", processed)

# 预处理DataFrame(假设有一个包含文本的DataFrame)
data = {
    'id': [1, 2, 3],
    'text': [
        "First example with some punctuation!",
        "Second example with a URL: https://example.org",
        "Third example with numbers 42 and special chars @#$"
    ]
}
df = pd.DataFrame(data)
processed_df = preprocessor.preprocess_dataframe(df, 'text')
print("\nDataFrame处理结果:")
print(processed_df)
```

### 3.2 使用spaCy进行更高级的预处理

spaCy是一个更现代的NLP库，提供了高级的预处理功能：

```python
# 需要安装: pip install spacy
# 还需要下载语言模型: python -m spacy download en_core_web_sm
import spacy

def preprocess_with_spacy(text):
    """使用spaCy进行文本预处理"""
    # 加载spaCy模型
    nlp = spacy.load("en_core_web_sm")
    
    # 解析文本
    doc = nlp(text)
    
    # 基本预处理
    tokens = []
    for token in doc:
        # 过滤掉标点和停用词
        if not token.is_punct and not token.is_stop:
            # 使用词形还原
            tokens.append(token.lemma_)
    
    return " ".join(tokens)

def analyze_text_with_spacy(text):
    """使用spaCy分析文本"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    print("TokenText\tPOS\tLemma\tIsStop")
    print("-" * 50)
    
    for token in doc:
        print(f"{token.text}\t{token.pos_}\t{token.lemma_}\t{token.is_stop}")
    
    print("\n命名实体识别结果:")
    for ent in doc.ents:
        print(f"{ent.text}\t{ent.label_}")

text = "Apple Inc. was founded by Steve Jobs in California. He was a brilliant entrepreneur."
print("spaCy预处理结果:", preprocess_with_spacy(text))
print("\nspaCy详细分析:")
analyze_text_with_spacy(text)
```

### 3.3 中文文本预处理示例

```python
# 需要安装: pip install jieba
import jieba
import re

def preprocess_chinese_text(text):
    """中文文本预处理示例"""
    # 清理文本
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)  # 仅保留中文、英文和数字
    text = re.sub(r'\s+', ' ', text).strip()  # 规范化空格
    
    # 分词
    words = jieba.cut(text)
    
    # 简单的停用词列表
    stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '们', '到'}
    
    # 过滤停用词
    filtered_words = [word for word in words if word not in stopwords and word != ' ']
    
    return ' '.join(filtered_words)

chinese_text = "今天天气真不错，我在北京三里屯的咖啡厅喝咖啡。"
print("原始中文:", chinese_text)
print("处理后:", preprocess_chinese_text(chinese_text))
```

### 3.4 处理大规模文本数据

处理大型文本语料库需要考虑效率和内存使用：

```python
def process_large_file(input_file, output_file, batch_size=1000):
    """批量处理大型文本文件"""
    preprocessor = TextPreprocessor()
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        batch = []
        for i, line in enumerate(fin):
            batch.append(line.strip())
            
            # 当达到批处理大小时，进行处理
            if len(batch) >= batch_size:
                processed_batch = [preprocessor.preprocess(text) for text in batch]
                fout.write('\n'.join(processed_batch) + '\n')
                batch = []  # 重置批处理
                print(f"已处理 {i+1} 行")
        
        # 处理剩余的文本
        if batch:
            processed_batch = [preprocessor.preprocess(text) for text in batch]
            fout.write('\n'.join(processed_batch))
            print(f"完成处理，共 {i+1} 行")

# 使用示例
# process_large_file('large_corpus.txt', 'processed_corpus.txt', batch_size=5000)
```

## 4. 高级应用与变体

### 4.1 使用预训练词嵌入

```python
# 需要安装: pip install gensim
from gensim.models import KeyedVectors
import numpy as np

def use_pretrained_word2vec():
    """使用预训练的Word2Vec模型"""
    # 加载预训练模型
    # 注: 需要先下载模型文件，如Google的Word2Vec或GloVe
    # model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    
    # 由于文件太大，这里仅展示概念代码
    print("加载预训练的Word2Vec模型...")
    
    # 假设模型已加载，获取词向量
    # vector = model['computer']
    
    # 两个词的相似度
    # similarity = model.similarity('computer', 'laptop')
    
    # 相似词查找
    # similar_words = model.most_similar('computer', topn=5)
    
    # 平均多个词向量来表示一个句子
    def get_sentence_vector(sentence, model):
        """获取句子的向量表示"""
        words = sentence.lower().split()
        word_vectors = []
        
        for word in words:
            if word in model.key_to_index:
                word_vectors.append(model[word])
        
        if word_vectors:
            # 所有词向量的平均值
            return np.mean(word_vectors, axis=0)
        else:
            # 如果没有任何词在词汇表中，则返回零向量
            return np.zeros(model.vector_size)

# use_pretrained_word2vec()
```

### 4.2 文本增强(Text Augmentation)

文本增强通过创建样本的变体来增加训练数据量：

```python
import random
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

def synonym_replacement(words, n):
    """随机替换n个非停用词为其同义词"""
    new_words = words.copy()
    stopwords_english = set(stopwords.words('english'))
    # 不属于停用词的词汇索引
    non_stop_indices = [i for i, word in enumerate(words) if word.lower() not in stopwords_english]
    
    # 随机选择n个非停用词进行替换(如果可能)
    n = min(n, len(non_stop_indices))
    replace_indices = random.sample(non_stop_indices, n)
    
    for idx in replace_indices:
        word = words[idx]
        synonyms = []
        
        # 获取同义词
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word:
                    synonyms.append(synonym)
        
        if synonyms:
            new_words[idx] = random.choice(synonyms)
    
    return new_words

def random_deletion(words, p=0.1):
    """以概率p随机删除词语"""
    if len(words) == 1:
        return words
    
    new_words = []
    for word in words:
        # 以1-p的概率保留词语
        if random.random() > p:
            new_words.append(word)
    
    # 确保至少保留一个词
    if not new_words:
        new_words.append(random.choice(words))
    
    return new_words

def random_swap(words, n=1):
    """随机交换n对词语位置"""
    new_words = words.copy()
    length = len(new_words)
    
    for _ in range(n):
        if length > 1:  # 至少需要两个词才能交换
            idx1, idx2 = random.sample(range(length), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    
    return new_words

def text_augmentation_demo():
    """文本增强示例"""
    text = "The quick brown fox jumps over the lazy dog"
    words = text.split()
    
    print("原始文本:", text)
    
    # 同义词替换
    augmented1 = ' '.join(synonym_replacement(words, 2))
    print("同义词替换:", augmented1)
    
    # 随机删除
    augmented2 = ' '.join(random_deletion(words))
    print("随机删除:", augmented2)
    
    # 随机交换
    augmented3 = ' '.join(random_swap(words, 2))
    print("随机交换:", augmented3)

text_augmentation_demo()
```

### 4.3 处理特殊领域文本

```python
def preprocess_medical_text(text):
    """医疗文本预处理示例"""
    # 医学术语和缩写字典(示例)
    medical_terms = {
        "MI": "myocardial infarction",
        "HTN": "hypertension",
        "DM": "diabetes mellitus",
        "COPD": "chronic obstructive pulmonary disease"
    }
    
    # 转换为小写但保留缩写
    for term in medical_terms.keys():
        text = re.sub(rf'\b{term}\b', f"__{term}__", text)
    
    text = text.lower()
    
    # 恢复缩写并展开
    for term, expansion in medical_terms.items():
        text = text.replace(f"__{term.lower()}__", expansion)
    
    # 标准清洗步骤
    text = re.sub(r'[^\w\s]', ' ', text)  # 移除标点
    text = re.sub(r'\s+', ' ', text).strip()  # 规范化空格
    
    return text

def preprocess_social_media_text(text):
    """社交媒体文本预处理示例"""
    # 处理表情符号(简化示例)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # 表情
                               u"\U0001F300-\U0001F5FF"  # 符号和象形文字
                               u"\U0001F680-\U0001F6FF"  # 交通和地图
                               u"\U0001F700-\U0001F77F"  # 字母符号
                               u"\U0001F780-\U0001F7FF"  # 几何符号
                               u"\U0001F800-\U0001F8FF"  # 箭头
                               u"\U0001F900-\U0001F9FF"  # 补充符号
                               u"\U0001FA00-\U0001FA6F"  # 象形文字扩展
                               u"\U0001FA70-\U0001FAFF"  # 符号和象形文字扩展
                               u"\U00002702-\U000027B0"  # 装饰符号
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'EMOJI', text)
    
    # 处理主题标签
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 处理@提及
    text = re.sub(r'@(\w+)', r'USER', text)
    
    # 展开常见网络用语(示例)
    slang_dict = {
        "lol": "laugh out loud",
        "brb": "be right back",
        "idk": "i do not know",
        "tbh": "to be honest"
    }
    
    words = text.lower().split()
    for i, word in enumerate(words):
        if word in slang_dict:
            words[i] = slang_dict[word]
    
    text = ' '.join(words)
    
    # 常规清洁
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 示例
medical_text = "Pt with HTN presented with MI and DM. COPD exacerbation."
print("医疗文本原文:", medical_text)
print("处理后:", preprocess_medical_text(medical_text))

social_text = "OMG! This is sooo funny 😂 #awesome #nlp @user lol"
print("\n社交媒体原文:", social_text)
print("处理后:", preprocess_social_media_text(social_text))
```

### 4.4 使用Transformers进行现代文本处理

```python
# 需要安装: pip install transformers
from transformers import AutoTokenizer

def transformer_tokenization():
    """使用Transformer模型的分词器"""
    # 加载BERT分词器
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # 示例文本
    text = "Transformer models have revolutionized NLP."
    
    # 使用分词器
    tokens = tokenizer.tokenize(text)
    print("BERT标记化结果:", tokens)
    
    # 转换为ID
    input_ids = tokenizer.encode(text)
    print("Token IDs:", input_ids)
    
    # 解码回文本
    decoded = tokenizer.decode(input_ids)
    print("解码后:", decoded)
    
    # 处理长文本
    long_text = "This is a very long text " * 50 + "that exceeds the maximum context length."
    
    # 使用截断
    encoded = tokenizer(long_text, truncation=True, max_length=20)
    print("\n截断后的Token IDs:", encoded['input_ids'])
    print("解码后:", tokenizer.decode(encoded['input_ids']))

transformer_tokenization()
```

### 4.5 创建自定义预处理管道

```python
class CustomPreprocessingPipeline:
    """可配置的自定义预处理管道"""
    
    def __init__(self, steps=None):
        """
        初始化预处理管道
        steps: 预处理步骤列表，每个元素是(名称, 函数)元组
        """
        self.steps = steps or []
    
    def add_step(self, name, func):
        """添加预处理步骤"""
        self.steps.append((name, func))
        return self
    
    def remove_step(self, name):
        """移除预处理步骤"""
        self.steps = [step for step in self.steps if step[0] != name]
        return self
    
    def process(self, text):
        """应用完整预处理管道"""
        result = text
        for name, func in self.steps:
            result = func(result)
        return result
    
    def process_batch(self, texts):
        """批量处理文本"""
        return [self.process(text) for text in texts]

# 定义一些预处理函数
def lowercase(text):
    return text.lower()

def remove_special_chars(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

# 使用示例
pipeline = CustomPreprocessingPipeline()
pipeline.add_step('lowercase', lowercase)
pipeline.add_step('remove_special', remove_special_chars)
pipeline.add_step('clean_whitespace', remove_extra_whitespace)

text = "Hello, World!  This is an   example."
processed = pipeline.process(text)
print("原文:", text)
print("处理后:", processed)

# 添加或移除步骤
pipeline.add_step('custom', lambda x: x.replace('example', 'sample'))
processed = pipeline.process(text)
print("添加步骤后:", processed)

pipeline.remove_step('remove_special')
processed = pipeline.process(text)
print("移除步骤后:", processed)
```

## 总结

文本预处理是NLP管道的基础环节，对下游任务的性能有着决定性影响。本文从基础概念、技术细节、实践实现到高级应用全面介绍了文本预处理技术。

### 主要技术要点

1. **文本清洗**：移除噪声、标点、特殊字符等
2. **文本规范化**：词干提取、词形还原、大小写转换等
3. **分词**：句子级、单词级、子词级分词技术
4. **停用词移除**：过滤高频但低信息量的词语
5. **向量化**：将文本转换为数值表示(词袋模型、TF-IDF、词嵌入)

### 进阶技术

1. **领域适应**：处理医疗、法律、社交媒体等特定领域的文本
2. **文本增强**：通过同义词替换、随机删除等技术扩充训练数据
3. **多语言预处理**：针对不同语言的特定技术和挑战
4. **现代预处理技术**：使用Transformers等先进模型的分词和表示

### 最佳实践

1. **创建灵活的预处理管道**：根据任务需求选择合适的预处理步骤
2. **性能与质量平衡**：在大规模处理时考虑效率和准确性的平衡
3. **持续评估**：通过下游任务性能评估预处理效果
4. **关注数据特性**：根据文本类型和来源调整预处理策略

掌握文本预处理技术是成为NLP专家的必经之路，灵活运用这些技术可以显著提高各种自然语言处理任务的性能。随着领域的发展，预处理技术也在不断演进，保持学习新技术的习惯将使您在NLP领域保持竞争力。

Similar code found with 1 license type