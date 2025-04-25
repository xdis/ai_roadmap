"""
特征工程模块：实现文本特征提取和表示方法，包括词袋模型、TF-IDF、词嵌入等
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Any, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import gensim.downloader as gensim_downloader
from gensim.models import Word2Vec
import pickle
import os
import joblib
from .preprocessing import TextPreprocessor, preprocess_for_sentiment

# 如果可用，尝试导入PyTorch和TensorFlow
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class BagOfWordsVectorizer:
    """词袋模型向量化器"""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 1),
                 binary: bool = False, save_path: Optional[str] = None):
        """
        初始化词袋模型向量化器
        
        Args:
            max_features: 保留的最大特征数
            ngram_range: n-gram范围，如(1,2)表示包含unigrams和bigrams
            binary: 是否使用二元特征（出现为1，未出现为0）
            save_path: 模型保存路径
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.binary = binary
        self.save_path = save_path
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            binary=binary
        )
        self.feature_names = None
    
    def fit(self, texts: List[str]) -> 'BagOfWordsVectorizer':
        """
        拟合词袋模型
        
        Args:
            texts: 文本列表
            
        Returns:
            self
        """
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # 保存模型
        if self.save_path:
            self.save(self.save_path)
            
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        将文本转换为词袋表示
        
        Args:
            texts: 文本列表
            
        Returns:
            词袋表示
        """
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        拟合并转换文本
        
        Args:
            texts: 文本列表
            
        Returns:
            词袋表示
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称
        
        Returns:
            特征名称列表
        """
        return self.feature_names.tolist() if self.feature_names is not None else []
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
    
    @classmethod
    def load(cls, path: str) -> 'BagOfWordsVectorizer':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型
        """
        vectorizer = cls()
        vectorizer.vectorizer = joblib.load(path)
        vectorizer.feature_names = vectorizer.vectorizer.get_feature_names_out()
        return vectorizer


class TfidfFeaturizer:
    """TF-IDF特征提取器"""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 1),
                 use_idf: bool = True, save_path: Optional[str] = None):
        """
        初始化TF-IDF特征提取器
        
        Args:
            max_features: 保留的最大特征数
            ngram_range: n-gram范围
            use_idf: 是否使用IDF权重
            save_path: 模型保存路径
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.save_path = save_path
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            use_idf=use_idf
        )
        self.feature_names = None
    
    def fit(self, texts: List[str]) -> 'TfidfFeaturizer':
        """
        拟合TF-IDF模型
        
        Args:
            texts: 文本列表
            
        Returns:
            self
        """
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # 保存模型
        if self.save_path:
            self.save(self.save_path)
            
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        将文本转换为TF-IDF表示
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF表示
        """
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        拟合并转换文本
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF表示
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称
        
        Returns:
            特征名称列表
        """
        return self.feature_names.tolist() if self.feature_names is not None else []
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
    
    @classmethod
    def load(cls, path: str) -> 'TfidfFeaturizer':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型
        """
        featurizer = cls()
        featurizer.vectorizer = joblib.load(path)
        featurizer.feature_names = featurizer.vectorizer.get_feature_names_out()
        return featurizer


class Word2VecFeaturizer:
    """Word2Vec词嵌入特征提取器"""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1,
                 workers: int = 4, sg: int = 0, pretrained: bool = False,
                 pretrained_model: str = 'word2vec-google-news-300',
                 save_path: Optional[str] = None):
        """
        初始化Word2Vec特征提取器
        
        Args:
            vector_size: 词向量维度
            window: 上下文窗口大小
            min_count: 忽略出现次数小于此值的词
            workers: 训练线程数
            sg: 模型类型，0为CBOW，1为Skip-gram
            pretrained: 是否使用预训练模型
            pretrained_model: 预训练模型名称
            save_path: 模型保存路径
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.pretrained = pretrained
        self.pretrained_model = pretrained_model
        self.save_path = save_path
        self.model = None
        
        if pretrained:
            try:
                self.model = gensim_downloader.load(pretrained_model)
                print(f"Loaded pretrained model: {pretrained_model}")
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
                self.pretrained = False
                self.model = None
    
    def fit(self, tokenized_texts: List[List[str]]) -> 'Word2VecFeaturizer':
        """
        拟合Word2Vec模型
        
        Args:
            tokenized_texts: 已分词的文本列表
            
        Returns:
            self
        """
        if not self.pretrained:
            self.model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
                sg=self.sg
            )
        
        # 保存模型
        if self.save_path and not self.pretrained:
            self.save(self.save_path)
            
        return self
    
    def transform(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """
        将分词文本转换为文档向量表示（使用平均词向量）
        
        Args:
            tokenized_texts: 已分词的文本列表
            
        Returns:
            文档向量表示
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        result = []
        
        for tokens in tokenized_texts:
            # 计算单词向量的平均值
            word_vectors = []
            for token in tokens:
                try:
                    # 对于预训练模型和自训练模型获取词向量的方式不同
                    if self.pretrained:
                        if token in self.model:
                            word_vectors.append(self.model[token])
                    else:
                        if token in self.model.wv:
                            word_vectors.append(self.model.wv[token])
                except KeyError:
                    continue
            
            # 如果没有任何词向量，则使用零向量
            if not word_vectors:
                if self.pretrained:
                    vec_size = self.model.vector_size
                else:
                    vec_size = self.model.vector_size
                avg_vector = np.zeros(vec_size)
            else:
                avg_vector = np.mean(word_vectors, axis=0)
                
            result.append(avg_vector)
        
        return np.array(result)
    
    def fit_transform(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """
        拟合并转换文本
        
        Args:
            tokenized_texts: 已分词的文本列表
            
        Returns:
            文档向量表示
        """
        self.fit(tokenized_texts)
        return self.transform(tokenized_texts)
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is not None and not self.pretrained:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
    
    @classmethod
    def load(cls, path: str) -> 'Word2VecFeaturizer':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型
        """
        featurizer = cls(pretrained=False)
        featurizer.model = Word2Vec.load(path)
        return featurizer


class BERTFeaturizer:
    """BERT特征提取器"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 max_length: int = 128,
                 pooling_strategy: str = 'mean_pooling',
                 device: Optional[str] = None):
        """
        初始化BERT特征提取器
        
        Args:
            model_name: BERT模型名称
            max_length: 最大序列长度
            pooling_strategy: 输出池化策略，可选'cls'或'mean_pooling'
            device: 计算设备，可选'cpu'或'cuda'
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is not available. Install it with: pip install transformers")
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        
        # 如果没有指定设备，检查CUDA是否可用
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # 评估模式
        self.model.eval()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        将文本转换为BERT表示
        
        Args:
            texts: 文本列表
            
        Returns:
            BERT特征表示
        """
        # 准备张量
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # 不计算梯度
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**encoded_input)
            # 获取所有token的最后一层隐藏状态
            last_hidden_state = outputs.last_hidden_state
            
            # 根据池化策略提取特征
            if self.pooling_strategy == 'cls':
                # [CLS] token的表示
                features = last_hidden_state[:, 0, :]
            elif self.pooling_strategy == 'mean_pooling':
                # 对所有token进行平均池化
                attention_mask = encoded_input['attention_mask']
                # 扩展维度以便进行广播
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                # 池化: 对所有token进行加权平均，忽略padding
                sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, 1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                features = sum_embeddings / sum_mask
            else:
                raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        
        # 转换为numpy数组
        return features.cpu().numpy()


class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    """自定义特征提取器，结合多种特征工程方法"""
    
    def __init__(self, use_bow: bool = True, use_tfidf: bool = True, 
                 use_word_embeddings: bool = False, use_bert: bool = False,
                 preprocessor: Optional[TextPreprocessor] = None,
                 bow_params: Optional[Dict[str, Any]] = None,
                 tfidf_params: Optional[Dict[str, Any]] = None,
                 word2vec_params: Optional[Dict[str, Any]] = None,
                 bert_params: Optional[Dict[str, Any]] = None):
        """
        初始化自定义特征提取器
        
        Args:
            use_bow: 是否使用词袋模型
            use_tfidf: 是否使用TF-IDF
            use_word_embeddings: 是否使用词嵌入
            use_bert: 是否使用BERT
            preprocessor: 文本预处理器
            bow_params: 词袋模型参数
            tfidf_params: TF-IDF参数
            word2vec_params: Word2Vec参数
            bert_params: BERT参数
        """
        self.use_bow = use_bow
        self.use_tfidf = use_tfidf
        self.use_word_embeddings = use_word_embeddings
        self.use_bert = use_bert
        
        # 初始化预处理器
        self.preprocessor = preprocessor if preprocessor else TextPreprocessor()
        
        # 初始化特征提取器
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.word2vec_featurizer = None
        self.bert_featurizer = None
        
        # 设置默认参数
        self.bow_params = bow_params or {}
        self.tfidf_params = tfidf_params or {}
        self.word2vec_params = word2vec_params or {}
        self.bert_params = bert_params or {}
        
        # 根据使用的特征提取方法初始化相应的特征提取器
        if use_bow:
            self.bow_vectorizer = BagOfWordsVectorizer(**self.bow_params)
        
        if use_tfidf:
            self.tfidf_vectorizer = TfidfFeaturizer(**self.tfidf_params)
        
        if use_word_embeddings:
            self.word2vec_featurizer = Word2VecFeaturizer(**self.word2vec_params)
        
        if use_bert and TRANSFORMERS_AVAILABLE:
            self.bert_featurizer = BERTFeaturizer(**self.bert_params)
    
    def fit(self, X: List[str], y=None) -> 'CustomFeatureExtractor':
        """
        拟合特征提取器
        
        Args:
            X: 文本列表
            y: 标签（可选）
            
        Returns:
            self
        """
        # 预处理文本
        processed_texts = [
            self.preprocessor.get_text_from_tokens(self.preprocessor.process_text(text))
            for text in X
        ]
        
        # 获取分词结果，用于Word2Vec
        tokenized_texts = [self.preprocessor.process_text(text) for text in X]
        
        # 拟合特征提取器
        if self.use_bow and self.bow_vectorizer:
            self.bow_vectorizer.fit(processed_texts)
        
        if self.use_tfidf and self.tfidf_vectorizer:
            self.tfidf_vectorizer.fit(processed_texts)
        
        if self.use_word_embeddings and self.word2vec_featurizer:
            self.word2vec_featurizer.fit(tokenized_texts)
        
        # BERT不需要拟合
            
        return self
    
    def transform(self, X: List[str]) -> np.ndarray:
        """
        转换文本为特征表示
        
        Args:
            X: 文本列表
            
        Returns:
            特征表示
        """
        # 预处理文本
        processed_texts = [
            self.preprocessor.get_text_from_tokens(self.preprocessor.process_text(text))
            for text in X
        ]
        
        # 获取分词结果，用于Word2Vec
        tokenized_texts = [self.preprocessor.process_text(text) for text in X]
        
        features_list = []
        
        # 转换文本
        if self.use_bow and self.bow_vectorizer:
            bow_features = self.bow_vectorizer.transform(processed_texts)
            features_list.append(bow_features)
        
        if self.use_tfidf and self.tfidf_vectorizer:
            tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
            features_list.append(tfidf_features)
        
        if self.use_word_embeddings and self.word2vec_featurizer:
            w2v_features = self.word2vec_featurizer.transform(tokenized_texts)
            features_list.append(w2v_features)
        
        if self.use_bert and self.bert_featurizer:
            bert_features = self.bert_featurizer.transform(X)  # 使用原始文本
            features_list.append(bert_features)
        
        # 特征组合
        if not features_list:
            raise ValueError("No features extracted. Enable at least one feature extraction method.")
        
        # 如果只有一种特征，直接返回
        if len(features_list) == 1:
            return features_list[0]
        
        # 否则水平拼接所有特征
        return np.hstack(features_list)
    
    def fit_transform(self, X: List[str], y=None) -> np.ndarray:
        """
        拟合并转换文本
        
        Args:
            X: 文本列表
            y: 标签（可选）
            
        Returns:
            特征表示
        """
        self.fit(X, y)
        return self.transform(X)
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 创建要保存的对象，不包括不可序列化的BERT特征提取器
        save_obj = {
            'use_bow': self.use_bow,
            'use_tfidf': self.use_tfidf,
            'use_word_embeddings': self.use_word_embeddings,
            'use_bert': self.use_bert,
            'preprocessor': self.preprocessor,
            'bow_params': self.bow_params,
            'tfidf_params': self.tfidf_params,
            'word2vec_params': self.word2vec_params,
            'bert_params': self.bert_params
        }
        
        # 单独保存可序列化的特征提取器
        if self.use_bow and self.bow_vectorizer:
            bow_path = os.path.join(os.path.dirname(path), "bow_vectorizer.joblib")
            self.bow_vectorizer.save(bow_path)
            save_obj['bow_vectorizer_path'] = bow_path
        
        if self.use_tfidf and self.tfidf_vectorizer:
            tfidf_path = os.path.join(os.path.dirname(path), "tfidf_vectorizer.joblib")
            self.tfidf_vectorizer.save(tfidf_path)
            save_obj['tfidf_vectorizer_path'] = tfidf_path
        
        if self.use_word_embeddings and self.word2vec_featurizer:
            w2v_path = os.path.join(os.path.dirname(path), "word2vec_model.bin")
            self.word2vec_featurizer.save(w2v_path)
            save_obj['word2vec_featurizer_path'] = w2v_path
        
        # 保存配置
        with open(path, 'wb') as f:
            pickle.dump(save_obj, f)
    
    @classmethod
    def load(cls, path: str) -> 'CustomFeatureExtractor':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型
        """
        with open(path, 'rb') as f:
            save_obj = pickle.load(f)
        
        # 创建实例
        instance = cls(
            use_bow=save_obj['use_bow'],
            use_tfidf=save_obj['use_tfidf'],
            use_word_embeddings=save_obj['use_word_embeddings'],
            use_bert=save_obj['use_bert'],
            preprocessor=save_obj['preprocessor'],
            bow_params=save_obj['bow_params'],
            tfidf_params=save_obj['tfidf_params'],
            word2vec_params=save_obj['word2vec_params'],
            bert_params=save_obj['bert_params']
        )
        
        # 加载特征提取器
        if 'bow_vectorizer_path' in save_obj:
            instance.bow_vectorizer = BagOfWordsVectorizer.load(save_obj['bow_vectorizer_path'])
        
        if 'tfidf_vectorizer_path' in save_obj:
            instance.tfidf_vectorizer = TfidfFeaturizer.load(save_obj['tfidf_vectorizer_path'])
        
        if 'word2vec_featurizer_path' in save_obj:
            instance.word2vec_featurizer = Word2VecFeaturizer.load(save_obj['word2vec_featurizer_path'])
        
        # BERT特征提取器需要重新初始化
        if instance.use_bert and TRANSFORMERS_AVAILABLE:
            instance.bert_featurizer = BERTFeaturizer(**instance.bert_params)
        
        return instance


# 辅助函数
def get_important_features(vectorizer, model, top_n: int = 20) -> Dict[str, float]:
    """
    获取最重要的特征及其权重
    
    Args:
        vectorizer: 特征提取器（如TfidfVectorizer）
        model: 已训练的模型（如LogisticRegression）
        top_n: 返回的特征数量
        
    Returns:
        特征名称到重要性的映射
    """
    if hasattr(vectorizer, 'get_feature_names'):
        feature_names = vectorizer.get_feature_names()
    elif hasattr(vectorizer, 'get_feature_names_out'):
        feature_names = vectorizer.get_feature_names_out()
    else:
        feature_names = [f"feature_{i}" for i in range(model.coef_.shape[1])]
    
    # 对于二分类问题
    if len(model.coef_.shape) == 1:
        coef = model.coef_
    else:
        coef = model.coef_[0]  # 取第一个类别的系数
    
    # 获取最重要的特征索引
    top_indices = np.argsort(np.abs(coef))[-top_n:]
    
    # 创建特征名到重要性的映射
    feature_importance = {feature_names[i]: coef[i] for i in top_indices}
    
    return feature_importance


if __name__ == "__main__":
    # 示例用法
    texts = [
        "I love this movie, it's amazing!",
        "This movie is terrible, I hate it.",
        "The acting was good but the plot was confusing.",
        "Great special effects and awesome action scenes!"
    ]
    
    # 使用TF-IDF特征
    tfidf = TfidfFeaturizer(max_features=1000)
    tfidf_features = tfidf.fit_transform(texts)
    print(f"TF-IDF特征形状: {tfidf_features.shape}")
    
    # 预处理文本并使用Word2Vec
    preprocessor = TextPreprocessor()
    tokenized_texts = [preprocessor.process_text(text) for text in texts]
    
    w2v = Word2VecFeaturizer(vector_size=50, pretrained=False)
    w2v_features = w2v.fit_transform(tokenized_texts)
    print(f"Word2Vec特征形状: {w2v_features.shape}")
    
    # 使用自定义特征提取器
    extractor = CustomFeatureExtractor(
        use_bow=True,
        use_tfidf=True,
        use_word_embeddings=True,
        use_bert=False,  # BERT需要预先安装transformers库
        bow_params={'max_features': 500},
        tfidf_params={'max_features': 500},
        word2vec_params={'vector_size': 50}
    )
    
    features = extractor.fit_transform(texts)
    print(f"组合特征形状: {features.shape}")