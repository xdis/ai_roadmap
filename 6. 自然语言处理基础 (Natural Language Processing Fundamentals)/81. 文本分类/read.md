# 文本分类完全指南

## 1. 基础概念理解

### 什么是文本分类
文本分类是自然语言处理的核心任务，旨在将文本文档（如新闻文章、评论、邮件等）根据内容自动分配到预定义的类别。它是许多现代应用的基础，从垃圾邮件过滤到内容推荐，从情感分析到法律文书归类。

### 文本分类的类型
1. **二元分类**：将文本划分为两个类别（如垃圾邮件/非垃圾邮件）
2. **多类分类**：将文本分配到多个互斥类别中的一个（如新闻主题分类）
3. **多标签分类**：一篇文本可同时属于多个类别（如文章可同时标记为"技术"和"教育"）
4. **层次分类**：类别之间存在层次关系（如"计算机科学"→"机器学习"→"深度学习"）

### 文本分类应用场景
- **情感分析**：识别文本的情感倾向（正面、负面、中性）
- **垃圾内容过滤**：过滤垃圾邮件、评论、信息
- **主题分类**：将新闻文章归类到体育、政治、科技等主题
- **意图识别**：理解用户查询的意图（如搜索、购买、咨询）
- **有害内容检测**：识别并过滤仇恨言论、暴力内容等
- **客户反馈分类**：将产品反馈归类到不同问题类型

### 文本分类流程
1. **数据收集与标注**：获取带标签的文本数据
2. **预处理**：清理、标准化文本
3. **特征提取**：将文本转化为机器学习算法可处理的形式
4. **模型选择与训练**：选择合适的分类算法并训练
5. **评估**：使用各种指标评估模型性能
6. **部署与监控**：部署模型并持续监控其性能

### 评估指标
- **准确率(Accuracy)**：正确预测的比例
- **精确率(Precision)**：预测为某类中实际为该类的比例
- **召回率(Recall)**：实际为某类中被正确预测出的比例
- **F1分数**：精确率和召回率的调和平均
- **混淆矩阵**：详细展示各类别之间的预测分布
- **ROC曲线和AUC**：评估二分类器性能的曲线和面积指标

## 2. 技术细节探索

### 文本预处理技术
文本预处理对分类效果至关重要，常见步骤包括：

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 下载必要资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    
    # 移除特殊字符和数字
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # 分词
    tokens = word_tokenize(text)
    
    # 移除停用词
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 词干提取
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    # 词形还原(通常选择词干提取或词形还原其一)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # 返回处理后的文本(可根据需要选择返回词干化或词形还原后的结果)
    return ' '.join(lemmatized_tokens)
```

### 特征提取方法

#### 1. 词袋模型(Bag of Words)和TF-IDF
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 原始文本
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# 词袋模型
count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(corpus)
print("词袋特征名称:", count_vectorizer.get_feature_names_out())
print("词袋特征矩阵:\n", X_count.toarray())

# TF-IDF模型
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)
print("TF-IDF特征名称:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF特征矩阵:\n", X_tfidf.toarray())
```

#### 2. N-gram特征
```python
# 使用N-gram捕捉短语和上下文
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))  # 同时包含单个词和两个词的组合
X_ngram = ngram_vectorizer.fit_transform(corpus)
print("N-gram特征名称:", ngram_vectorizer.get_feature_names_out())
```

#### 3. 词嵌入特征
```python
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# 准备数据
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

# 训练Word2Vec模型
w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# 获取文档向量(简单平均法)
def document_vector(doc, model):
    words = word_tokenize(doc.lower())
    # 过滤不在模型词汇表中的词
    words = [word for word in words if word in model.wv]
    if len(words) == 0:
        return np.zeros(model.vector_size)
    # 计算所有词向量的平均
    return np.mean([model.wv[word] for word in words], axis=0)

# 为所有文档生成向量
doc_vectors = np.array([document_vector(doc, w2v_model) for doc in corpus])
print("词嵌入文档向量形状:", doc_vectors.shape)
```

#### 4. 预训练语言模型特征
```python
from transformers import AutoTokenizer, AutoModel
import torch

# 加载预训练模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 获取BERT特征
def get_bert_features(texts, tokenizer, model):
    # 准备批量输入
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # 获取模型输出
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # 使用[CLS]标记的最后隐藏状态作为文档表示
    return model_output.last_hidden_state[:, 0, :].numpy()

bert_features = get_bert_features(corpus, tokenizer, model)
print("BERT文档向量形状:", bert_features.shape)
```

### 分类算法

#### 1. 传统机器学习模型
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 假设我们有预处理好的特征X和标签y
X = X_tfidf
y = [0, 1, 0, 1]  # 示例标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 朴素贝叶斯
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
print("朴素贝叶斯模型性能:\n", classification_report(y_test, nb_preds))

# 逻辑回归
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
print("逻辑回归模型性能:\n", classification_report(y_test, lr_preds))

# 支持向量机
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
print("SVM模型性能:\n", classification_report(y_test, svm_preds))

# 随机森林
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("随机森林模型性能:\n", classification_report(y_test, rf_preds))
```

#### 2. 深度学习模型
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 准备数据
texts = corpus
labels = [0, 1, 0, 1]  # 示例标签

# 创建词汇表和序列
max_words = 1000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x_data = pad_sequences(sequences, maxlen=max_len)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.25, random_state=42)

# CNN模型
def build_cnn_model():
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # 二分类使用sigmoid
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# LSTM模型
def build_lstm_model():
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # 二分类使用sigmoid
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 训练CNN模型
cnn_model = build_cnn_model()
cnn_model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

# 训练LSTM模型
lstm_model = build_lstm_model()
lstm_model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
```

#### 3. 预训练语言模型微调
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
from datasets import Dataset

# 准备数据
texts = corpus
labels = [0, 1, 0, 1]  # 示例标签

# 创建Hugging Face Dataset
dataset_dict = {'text': texts, 'label': labels}
dataset = Dataset.from_dict(dataset_dict)
train_test_dataset = dataset.train_test_split(test_size=0.2)

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 数据预处理函数
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 对数据集应用预处理
tokenized_datasets = train_test_dataset.map(tokenize_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# 训练模型
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
```

### 处理数据不平衡

```python
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 假设y_train是不平衡的类标签
# 1. 类权重方法
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# 在训练时使用类权重
# 对于Scikit-learn:
model = LogisticRegression(class_weight=class_weight_dict)

# 对于Keras:
# model.fit(X_train, y_train, class_weight=class_weight_dict)

# 2. 过采样方法 - SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 3. 欠采样方法
under_sampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)
```

## 3. 实践与实现

### 构建完整文本分类系统的步骤

#### 步骤1：数据准备
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据(以新闻分类为例)
# 可以使用scikit-learn自带的数据集或从CSV文件加载
from sklearn.datasets import fetch_20newsgroups

# 加载20 Newsgroups数据集的一部分类别
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# 创建DataFrame
df = pd.DataFrame({
    'text': newsgroups.data,
    'category': newsgroups.target
})

# 查看数据分布
print(df['category'].value_counts())

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
```

#### 步骤2：数据预处理和特征提取
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 下载必要资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 文本预处理函数
def preprocess_text(text):
    # 处理空值
    if isinstance(text, float):
        return ""
    
    # 转换为小写
    text = text.lower()
    
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 移除特殊字符和数字
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 分词
    tokens = word_tokenize(text)
    
    # 移除停用词
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# 应用预处理
train_df['processed_text'] = train_df['text'].apply(preprocess_text)
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

# TF-IDF特征提取
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train = tfidf_vectorizer.fit_transform(train_df['processed_text'])
X_test = tfidf_vectorizer.transform(test_df['processed_text'])

y_train = train_df['category']
y_test = test_df['category']
```

#### 步骤3：模型选择、训练与评估
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 定义和训练多个模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVM': LinearSVC(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': MultinomialNB()
}

results = {}

for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=newsgroups.target_names)
    
    # 存储结果
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'report': report,
        'predictions': y_pred
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

# 找到表现最好的模型
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
print(f"\nBest performing model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

# 可视化混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# 绘制最佳模型的混淆矩阵
plot_confusion_matrix(y_test, results[best_model_name]['predictions'], newsgroups.target_names)
```

#### 步骤4：使用深度学习模型
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 数据准备
train_texts = train_df['processed_text'].tolist()
test_texts = test_df['processed_text'].tolist()

# 标签编码
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_df['category'])
y_test_encoded = label_encoder.transform(test_df['category'])

# 转换为独热编码(用于多类分类)
y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded, num_classes=len(newsgroups.target_names))
y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, num_classes=len(newsgroups.target_names))

# 文本序列化
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)

# 转换文本为序列
X_train_seq = tokenizer.texts_to_sequences(train_texts)
X_test_seq = tokenizer.texts_to_sequences(test_texts)

# 填充序列
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# 构建BiLSTM模型
def build_bilstm_model(vocab_size, embedding_dim=128, input_length=max_len, num_classes=len(newsgroups.target_names)):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 创建模型
vocab_size = min(max_words, len(tokenizer.word_index) + 1)
bilstm_model = build_bilstm_model(vocab_size)

# 模型摘要
bilstm_model.summary()

# 训练模型
history = bilstm_model.fit(
    X_train_pad, y_train_categorical,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 评估模型
loss, accuracy = bilstm_model.evaluate(X_test_pad, y_test_categorical)
print(f"测试准确率: {accuracy:.4f}")

# 可视化训练过程
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)
```

#### 步骤5：使用预训练语言模型(BERT)
```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 标签编码
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['category'])
test_labels = label_encoder.transform(test_df['category'])

# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 准备BERT输入
def prepare_bert_input(texts, max_length=128):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    # 转换为张量
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks

# 获取训练和测试数据的BERT输入
train_input_ids, train_attention_masks = prepare_bert_input(train_df['processed_text'].tolist())
test_input_ids, test_attention_masks = prepare_bert_input(test_df['processed_text'].tolist())

# 创建PyTorch数据集
train_labels_tensor = torch.tensor(train_labels)
test_labels_tensor = torch.tensor(test_labels)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels_tensor)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels_tensor)

# 创建数据加载器
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 加载预训练BERT模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(newsgroups.target_names)
)

# 配置训练参数
optimizer = AdamW(model.parameters(), lr=2e-5)

# 设置设备(GPU或CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 训练模型
epochs = 4

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        # 解包批次数据并移动到设备
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        # 清除之前的梯度
        model.zero_grad()
        
        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average training loss: {avg_train_loss:.4f}")

# 评估模型
model.eval()
predictions = []
true_labels = []

for batch in test_dataloader:
    input_ids, attention_mask, labels = [b.to(device) for b in batch]
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
    
    predictions.extend(batch_predictions)
    true_labels.extend(labels.cpu().numpy())

# 计算准确率和分类报告
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions, target_names=newsgroups.target_names)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Classification Report:\n{report}")
```

#### 步骤6：模型保存和部署
```python
import pickle
import joblib
from sklearn.pipeline import Pipeline

# 保存传统机器学习模型和预处理管道
best_model = results[best_model_name]['model']

# 创建完整管道
model_pipeline = Pipeline([
    ('vectorizer', tfidf_vectorizer),
    ('classifier', best_model)
])

# 保存管道
joblib.dump(model_pipeline, 'text_classifier_pipeline.joblib')

# 保存深度学习模型
bilstm_model.save('bilstm_text_classifier.h5')

# 保存BERT模型
torch.save(model.state_dict(), 'bert_text_classifier.pt')

# 加载和使用模型进行预测
def predict_with_ml_pipeline(text):
    # 加载管道
    loaded_pipeline = joblib.load('text_classifier_pipeline.joblib')
    
    # 预测
    predicted_class = loaded_pipeline.predict([text])[0]
    
    return newsgroups.target_names[predicted_class]

# 测试ML管道
test_text = "The new graphics card has impressive rendering capabilities"
print(f"ML模型预测: '{test_text}' 属于类别 '{predict_with_ml_pipeline(test_text)}'")

# 使用BERT模型预测
def predict_with_bert(text, model_path='bert_text_classifier.pt'):
    # 加载模型
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(newsgroups.target_names)
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # 准备输入
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
    
    return newsgroups.target_names[prediction]

# API服务示例(使用Flask)
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': '没有提供文本'})
    
    # 使用加载的模型预测
    prediction = predict_with_ml_pipeline(text)
    
    return jsonify({
        'text': text,
        'predicted_category': prediction
    })

# 启动API服务
if __name__ == '__main__':
    app.run(debug=True)
```

## 4. 高级应用与变体

### 多标签文本分类

多标签分类允许一篇文本同时属于多个类别：

```python
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain

# 假设我们有多标签数据
texts = [
    "Machine learning and artificial intelligence are related fields",
    "Politics and economy are major news topics",
    "Sports news: football, basketball and tennis",
    "Science fiction and fantasy movies"
]

# 多标签(每个文本有多个标签)
multi_labels = [
    ['tech', 'science'],
    ['politics', 'economy'],
    ['sports', 'news'],
    ['entertainment', 'fiction']
]

# 对标签进行二值化编码
mlb = MultiLabelBinarizer()
y_multi = mlb.fit_transform(multi_labels)

# 特征提取
vectorizer = TfidfVectorizer(max_features=1000)
X_multi = vectorizer.fit_transform(texts)

# 方法1: 二元相关性
classifier = BinaryRelevance(
    classifier=LogisticRegression(),
    require_dense=[False, True]
)

# 方法2: 分类器链
# chain_classifier = ClassifierChain(
#     classifier=LogisticRegression(),
#     require_dense=[False, True]
# )

# 方法3: 使用scikit-learn的MultiOutputClassifier
# multi_classifier = MultiOutputClassifier(LogisticRegression())

# 训练模型
classifier.fit(X_multi, y_multi)

# 预测
new_text = ["Artificial intelligence is transforming the economy"]
new_X = vectorizer.transform(new_text)
predicted = classifier.predict(new_X)

# 解码预测结果
predicted_labels = mlb.inverse_transform(predicted.toarray())
print(f"预测标签: {predicted_labels}")
```

### 层次文本分类

处理具有层次结构的标签：

```python
import networkx as nx
from sklearn.preprocessing import LabelEncoder

# 定义标签层次结构
hierarchy = {
    'science': ['physics', 'biology', 'chemistry'],
    'technology': ['software', 'hardware'],
    'arts': ['literature', 'music', 'painting']
}

# 创建层次图
G = nx.DiGraph()
for parent, children in hierarchy.items():
    G.add_node(parent)
    for child in children:
        G.add_edge(parent, child)

# 层次分类方法1: 层次学习(本地分类器方法)
def hierarchical_classification(text, vectorizer, top_level_model, child_models):
    # 向量化文本
    X = vectorizer.transform([text])
    
    # 预测顶层类别
    top_prediction = top_level_model.predict(X)[0]
    
    # 如果有子类别，继续预测
    if top_prediction in child_models:
        child_model = child_models[top_prediction]
        child_prediction = child_model.predict(X)[0]
        return f"{top_prediction}/{child_prediction}"
    else:
        return top_prediction

# 注: 实际实现需要为每个层级训练不同的分类器
```

### 零样本/少样本文本分类

使用预训练语言模型处理新类别或样本稀少的情况：

```python
from transformers import pipeline

# 零样本分类
zero_shot_classifier = pipeline("zero-shot-classification")

# 文本示例
text = "Breaking news: Massive earthquake hits coastal region causing widespread damage"

# 可能的类别
candidate_labels = ["politics", "sports", "disaster", "entertainment", "technology"]

# 进行零样本分类
result = zero_shot_classifier(text, candidate_labels)

print(f"Zero-shot classification results:")
for i, label in enumerate(result['labels']):
    print(f"{label}: {result['scores'][i]:.4f}")
```

### 主动学习

通过智能选择最有价值的样本进行标注来减少标注成本：

```python
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

# 初始标记数据
n_initial = 10
initial_idx = np.random.choice(range(len(X_train.toarray())), size=n_initial, replace=False)

X_initial = X_train[initial_idx]
y_initial = y_train.iloc[initial_idx]

# 未标记数据池
X_pool = np.delete(X_train.toarray(), initial_idx, axis=0)
y_pool = y_train.drop(initial_idx)

# 初始化主动学习器
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    X_training=X_initial, y_training=y_initial,
    query_strategy=uncertainty_sampling
)

# 主动学习循环
n_queries = 20
for _ in range(n_queries):
    # 选择最不确定的实例进行查询
    query_idx, query_instance = learner.query(X_pool, n_instances=1)
    
    # 获取真实标签(在实际应用中，这里是人工标注的部分)
    y_new = y_pool.iloc[query_idx]
    
    # 教导学习器
    learner.teach(query_instance, y_new)
    
    # 从池中移除已标记的实例
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = y_pool.drop(query_idx)

# 评估最终的分类器
accuracy = learner.score(X_test.toarray(), y_test)
print(f"主动学习后的分类器准确率: {accuracy:.4f}")
```

### 可解释文本分类

理解模型做出决策的原因：

```python
import lime
from lime.lime_text import LimeTextExplainer

# 创建文本解释器
explainer = LimeTextExplainer(class_names=newsgroups.target_names)

# 选择要解释的实例
idx = np.random.randint(0, len(test_df))
text_to_explain = test_df.iloc[idx]['processed_text']
true_class = test_df.iloc[idx]['category']

# 定义预测函数
def predictor(texts):
    # 使用之前训练的模型进行预测
    vectorized_texts = tfidf_vectorizer.transform(texts)
    return best_model.predict_proba(vectorized_texts)

# 生成解释
exp = explainer.explain_instance(
    text_to_explain, 
    predictor, 
    num_features=10
)

# 显示解释
print(f"解释文本 (真实类别: {newsgroups.target_names[true_class]}):")
print(text_to_explain[:300] + "...")
print("\n特征重要性:")
for word, weight in exp.as_list():
    print(f"{word}: {weight:.4f}")

# 可视化
exp.as_pyplot_figure()
plt.tight_layout()
plt.show()
```

### 跨语言文本分类

使用多语言模型在不同语言间转移知识：

```python
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# 加载多语言模型和分词器
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base", 
    num_labels=len(newsgroups.target_names)
)

# 在英语数据上微调
# ...训练代码类似于前面的BERT部分...

# 然后可以对其他语言进行预测
chinese_text = "人工智能是计算机科学的一个分支，它研究如何让计算机模仿人类的智能行为。"
spanish_text = "La inteligencia artificial es una rama de la informática que estudia cómo hacer que las computadoras imiten el comportamiento inteligente humano."

# 分词化多语言文本
encoded_dict = tokenizer.batch_encode_plus(
    [chinese_text, spanish_text],
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

input_ids = encoded_dict['input_ids'].to(device)
attention_mask = encoded_dict['attention_mask'].to(device)

# 预测
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
for i, text in enumerate([chinese_text, spanish_text]):
    print(f"文本: {text[:50]}...")
    print(f"预测类别: {newsgroups.target_names[predictions[i]]}")
```

### 多模态文本分类

结合文本和其他模态(如图像)进行分类：

```python
from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderProcessor
from PIL import Image
import requests

# 加载多模态模型和处理器
processor = VisionTextDualEncoderProcessor.from_pretrained("clip-base")
model = VisionTextDualEncoderModel.from_pretrained("clip-base")

# 示例文本和图像
text = "A sports car racing on a track"
image_url = "http://example.com/sportscar.jpg"

# 下载图像
image = Image.open(requests.get(image_url, stream=True).raw)

# 处理输入
inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

# 计算相似性得分
outputs = model(**inputs)
similarity_score = outputs.logits_per_image.item()

print(f"文本与图像的相似性得分: {similarity_score:.4f}")
```

### 自适应文本分类

随着新数据的不断产生，模型需要适应数据分布的变化：

```python
from sklearn.linear_model import SGDClassifier

# 在线学习分类器
classifier = SGDClassifier(loss='log', alpha=1e-5, max_iter=1000, tol=1e-3)

# 初始批次训练
classifier.partial_fit(X_train[:1000], y_train[:1000], classes=np.unique(y_train))

# 评估
initial_accuracy = classifier.score(X_test, y_test)
print(f"初始准确率: {initial_accuracy:.4f}")

# 模拟数据流和在线更新
for i in range(1, 10):
    # 假设接收到新批次数据
    start_idx = i * 1000
    end_idx = min((i + 1) * 1000, len(X_train))
    
    if start_idx >= len(X_train):
        break
    
    # 更新模型
    classifier.partial_fit(X_train[start_idx:end_idx], y_train[start_idx:end_idx])
    
    # 评估
    current_accuracy = classifier.score(X_test, y_test)
    print(f"批次 {i} 后的准确率: {current_accuracy:.4f}")
```

通过掌握这些技术和方法，你可以构建各种复杂的文本分类系统，从简单的情感分析到复杂的多标签、多语言分类任务。文本分类是自然语言处理的基础技术，也是许多更高级应用的关键组成部分。

Similar code found with 2 license types