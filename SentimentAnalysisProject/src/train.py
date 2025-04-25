"""
模型训练脚本：用于训练情感分析模型
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import joblib
import json
import logging
from datetime import datetime

# 获取当前文件所在目录的父目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将父目录添加到Python模块搜索路径中
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 导入自定义模块
from src.preprocessing import TextPreprocessor, preprocess_for_sentiment
from src.features import TfidfFeaturizer, Word2VecFeaturizer, CustomFeatureExtractor
from src.models.classical import SentimentClassifier, EnsembleClassifier, train_model_with_cv
from src.models.deep_learning import PyTorchSentimentModel, TensorFlowSentimentModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data(data_path: str, text_column: str = 'text', label_column: str = 'label',
             encoding: str = 'utf-8', sep: str = ',') -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    加载数据集
    
    Args:
        data_path: 数据文件路径
        text_column: 文本列名
        label_column: 标签列名
        encoding: 文件编码
        sep: 分隔符（对于CSV文件）
        
    Returns:
        数据框、文本列表和标签列表
    """
    # 检查文件扩展名
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(data_path, encoding=encoding, sep=sep)
    elif file_ext == '.tsv':
        df = pd.read_csv(data_path, encoding=encoding, sep='\t')
    elif file_ext == '.json':
        df = pd.read_json(data_path, encoding=encoding)
    elif file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # 检查必要的列是否存在
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in the data")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in the data")
    
    # 提取文本和标签
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    logger.info(f"Loaded {len(texts)} samples from {data_path}")
    return df, texts, labels


def preprocess_texts(texts: List[str], advanced: bool = False, 
                    save_preprocessor: bool = True, 
                    preprocessor_path: Optional[str] = None) -> Tuple[List[str], TextPreprocessor]:
    """
    预处理文本
    
    Args:
        texts: 原始文本列表
        advanced: 是否使用高级预处理
        save_preprocessor: 是否保存预处理器
        preprocessor_path: 预处理器保存路径
        
    Returns:
        预处理后的文本列表和预处理器
    """
    logger.info("开始文本预处理...")
    
    # 创建预处理器
    preprocessor = TextPreprocessor(remove_stopwords=True, stemming=False, lemmatization=True)
    
    # 预处理文本
    processed_texts = []
    for i, text in enumerate(texts):
        if advanced:
            processed_text = preprocess_for_sentiment(text, advanced=True)
        else:
            tokens = preprocessor.process_text(text)
            processed_text = preprocessor.get_text_from_tokens(tokens)
        
        processed_texts.append(processed_text)
        
        # 显示进度
        if (i + 1) % 1000 == 0 or i == len(texts) - 1:
            logger.info(f"预处理进度: {i+1}/{len(texts)}")
    
    # 保存预处理器
    if save_preprocessor and preprocessor_path:
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        with open(preprocessor_path, 'wb') as f:
            joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"预处理器已保存至 {preprocessor_path}")
    
    logger.info("文本预处理完成")
    return processed_texts, preprocessor


def extract_features(texts: List[str], method: str = 'tfidf', 
                   model_params: Optional[Dict[str, Any]] = None,
                   save_path: Optional[str] = None) -> Tuple[np.ndarray, Any]:
    """
    提取文本特征
    
    Args:
        texts: 预处理后的文本列表
        method: 特征提取方法
        model_params: 模型参数
        save_path: 特征提取器保存路径
        
    Returns:
        特征矩阵和特征提取器
    """
    logger.info(f"开始使用 {method} 提取特征...")
    
    # 默认参数
    if model_params is None:
        model_params = {}
    
    # 根据方法选择特征提取器
    if method == 'tfidf':
        # 默认TF-IDF参数
        default_params = {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'use_idf': True
        }
        # 更新参数
        params = {**default_params, **model_params}
        
        # 创建特征提取器
        feature_extractor = TfidfFeaturizer(**params)
        # 提取特征
        features = feature_extractor.fit_transform(texts)
        
    elif method == 'word2vec':
        # 需要分词的文本
        preprocessor = TextPreprocessor()
        tokenized_texts = [preprocessor.process_text(text) for text in texts]
        
        # 默认Word2Vec参数
        default_params = {
            'vector_size': 100,
            'window': 5,
            'min_count': 1,
            'workers': 4,
            'sg': 1,  # Skip-gram
            'pretrained': False
        }
        # 更新参数
        params = {**default_params, **model_params}
        
        # 创建特征提取器
        feature_extractor = Word2VecFeaturizer(**params)
        # 提取特征
        features = feature_extractor.fit_transform(tokenized_texts)
        
    elif method == 'custom':
        # 默认自定义特征提取器参数
        default_params = {
            'use_bow': True,
            'use_tfidf': True,
            'use_word_embeddings': True,
            'use_bert': False,
            'bow_params': {'max_features': 3000},
            'tfidf_params': {'max_features': 3000},
            'word2vec_params': {'vector_size': 100, 'pretrained': False}
        }
        # 更新参数
        params = {**default_params, **model_params}
        
        # 创建特征提取器
        feature_extractor = CustomFeatureExtractor(**params)
        # 提取特征
        features = feature_extractor.fit_transform(texts)
        
    else:
        raise ValueError(f"Unsupported feature extraction method: {method}")
    
    # 保存特征提取器
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if hasattr(feature_extractor, 'save'):
            feature_extractor.save(save_path)
        else:
            with open(save_path, 'wb') as f:
                joblib.dump(feature_extractor, save_path)
        logger.info(f"特征提取器已保存至 {save_path}")
    
    logger.info(f"特征提取完成，特征维度: {features.shape}")
    return features, feature_extractor


def train_classical_models(X_train: np.ndarray, y_train: np.ndarray, 
                         model_types: List[str] = None,
                         use_ensemble: bool = True,
                         class_weight: Optional[str] = 'balanced',
                         save_dir: Optional[str] = None) -> Any:
    """
    训练传统机器学习模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        model_types: 模型类型列表
        use_ensemble: 是否使用集成学习
        class_weight: 类别权重
        save_dir: 模型保存目录
        
    Returns:
        训练好的模型
    """
    if model_types is None:
        model_types = ['logistic_regression', 'naive_bayes', 'svm']
    
    logger.info(f"开始训练传统机器学习模型: {', '.join(model_types)}")
    
    # 训练模型
    model = train_model_with_cv(
        model_types=model_types,
        X_train=X_train,
        y_train=y_train,
        cv=5,
        class_weight=class_weight,
        use_ensemble=use_ensemble,
        save_dir=save_dir
    )
    
    logger.info("传统机器学习模型训练完成")
    return model


def train_deep_learning_model(X_train: np.ndarray, y_train: np.ndarray,
                           framework: str = 'pytorch',
                           model_type: str = 'mlp',
                           model_params: Optional[Dict[str, Any]] = None,
                           training_params: Optional[Dict[str, Any]] = None,
                           save_path: Optional[str] = None) -> Any:
    """
    训练深度学习模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        framework: 深度学习框架，'pytorch'或'tensorflow'
        model_type: 模型类型
        model_params: 模型参数
        training_params: 训练参数
        save_path: 模型保存路径
        
    Returns:
        训练好的模型
    """
    # 默认参数
    if model_params is None:
        model_params = {}
        
    if training_params is None:
        training_params = {
            'batch_size': 32,
            'epochs': 20,
            'validation_split': 0.1,
            'early_stopping': True,
            'learning_rate': 1e-3,
            'optimizer_type': 'adam'
        }
    
    logger.info(f"开始训练{framework} {model_type}模型...")
    
    # 选择框架
    if framework.lower() == 'pytorch':
        # 创建PyTorch模型
        model = PyTorchSentimentModel(
            model_type=model_type,
            model_params=model_params
        )
    elif framework.lower() == 'tensorflow':
        # 创建TensorFlow模型
        model = TensorFlowSentimentModel(
            model_type=model_type,
            model_params=model_params
        )
    else:
        raise ValueError(f"Unsupported deep learning framework: {framework}")
    
    # 训练模型
    history = model.fit(
        X=X_train,
        y=y_train,
        **training_params
    )
    
    # 保存模型
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        logger.info(f"模型已保存至 {save_path}")
    
    logger.info(f"{framework} {model_type}模型训练完成")
    return model


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        
    Returns:
        评估指标
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    logger.info("开始评估模型...")
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 记录结果
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    logger.info(f"评估指标: {metrics}")
    logger.info(f"混淆矩阵:\n{cm}")
    
    return metrics


def save_metrics(metrics: Dict[str, float], model_name: str, save_dir: Optional[str] = None) -> None:
    """
    保存评估指标
    
    Args:
        metrics: 评估指标
        model_name: 模型名称
        save_dir: 保存目录
    """
    if save_dir is None:
        save_dir = 'models'
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 添加时间戳
    metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 保存指标
    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"评估指标已保存至 {metrics_path}")


def main(args: argparse.Namespace) -> None:
    """
    主函数
    
    Args:
        args: 命令行参数
    """
    start_time = time.time()
    logger.info("===== 开始情感分析模型训练 =====")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置保存路径
    preprocessor_path = os.path.join(args.output_dir, "preprocessor.joblib")
    feature_extractor_path = os.path.join(args.output_dir, f"{args.feature_method}_extractor.joblib")
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 加载数据
    df, texts, labels = load_data(
        args.data_path,
        text_column=args.text_column,
        label_column=args.label_column
    )
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=labels
    )
    
    logger.info(f"数据集分割: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
    
    # 预处理文本
    X_train_processed, preprocessor = preprocess_texts(
        X_train,
        advanced=args.advanced_preprocessing,
        save_preprocessor=True,
        preprocessor_path=preprocessor_path
    )
    
    X_test_processed, _ = preprocess_texts(
        X_test,
        advanced=args.advanced_preprocessing,
        save_preprocessor=False
    )
    
    # 提取特征
    X_train_features, feature_extractor = extract_features(
        X_train_processed,
        method=args.feature_method,
        save_path=feature_extractor_path
    )
    
    X_test_features, _ = extract_features(
        X_test_processed,
        method=args.feature_method,
        save_path=None
    )
    
    # 训练模型
    if args.model_type == 'classical':
        # 训练传统机器学习模型
        model_types = args.classical_models.split(',')
        model = train_classical_models(
            X_train_features, y_train,
            model_types=model_types,
            use_ensemble=args.use_ensemble,
            class_weight='balanced',
            save_dir=model_dir
        )
        model_name = "ensemble" if args.use_ensemble else model_types[0]
        
    elif args.model_type == 'deep_learning':
        # 训练深度学习模型
        model = train_deep_learning_model(
            X_train_features, y_train,
            framework=args.dl_framework,
            model_type=args.dl_model,
            save_path=os.path.join(model_dir, f"{args.dl_framework}_{args.dl_model}_model")
        )
        model_name = f"{args.dl_framework}_{args.dl_model}"
        
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # 评估模型
    metrics = evaluate_model(model, X_test_features, y_test)
    
    # 保存评估指标
    save_metrics(metrics, model_name, model_dir)
    
    # 完成
    elapsed_time = time.time() - start_time
    logger.info(f"===== 情感分析模型训练完成! 用时: {elapsed_time:.2f}秒 =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to the input data file")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Name of the column containing text data")
    parser.add_argument("--label_column", type=str, default="label",
                       help="Name of the column containing labels")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Proportion of the dataset to include in the test split")
    
    # 预处理参数
    parser.add_argument("--advanced_preprocessing", action="store_true",
                       help="Use advanced preprocessing techniques")
    
    # 特征提取参数
    parser.add_argument("--feature_method", type=str, default="tfidf",
                       choices=["tfidf", "word2vec", "custom"],
                       help="Feature extraction method")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="classical",
                       choices=["classical", "deep_learning"],
                       help="Type of model to train")
    parser.add_argument("--classical_models", type=str, 
                       default="logistic_regression,naive_bayes,svm",
                       help="Comma-separated list of classical models to train")
    parser.add_argument("--use_ensemble", action="store_true",
                       help="Use ensemble learning for classical models")
    parser.add_argument("--dl_framework", type=str, default="pytorch",
                       choices=["pytorch", "tensorflow"],
                       help="Deep learning framework")
    parser.add_argument("--dl_model", type=str, default="mlp",
                       choices=["mlp", "textcnn", "rnn"],
                       help="Deep learning model type")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="./models",
                       help="Directory to save models and results")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    main(args)