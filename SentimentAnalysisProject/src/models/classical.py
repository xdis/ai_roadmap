"""
传统机器学习模型模块：实现基于传统机器学习的情感分析模型
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os
from datetime import datetime
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentClassifier:
    """情感分类器基类，提供统一的接口"""
    
    def __init__(self, model_type: str = 'logistic_regression', 
                 class_weight: Optional[str] = 'balanced',
                 model_params: Optional[Dict[str, Any]] = None):
        """
        初始化情感分类器
        
        Args:
            model_type: 模型类型，可选'logistic_regression', 'naive_bayes', 'svm', 'random_forest', 'gradient_boosting'
            class_weight: 类别权重，用于处理不平衡数据
            model_params: 模型参数
        """
        self.model_type = model_type
        self.class_weight = class_weight
        self.model_params = model_params or {}
        self.model = None
        self.classes_ = None
        self.trained = False
        
        # 根据模型类型初始化相应的分类器
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """根据模型类型初始化分类器"""
        if self.model_type == 'logistic_regression':
            params = {
                'max_iter': 1000,
                'C': 1.0,
                'solver': 'liblinear',
                'random_state': 42
            }
            params.update(self.model_params)
            if self.class_weight:
                params['class_weight'] = self.class_weight
            self.model = LogisticRegression(**params)
            
        elif self.model_type == 'naive_bayes':
            params = {
                'alpha': 1.0
            }
            params.update(self.model_params)
            self.model = MultinomialNB(**params)
            
        elif self.model_type == 'svm':
            params = {
                'C': 1.0,
                'random_state': 42
            }
            params.update(self.model_params)
            if self.class_weight:
                params['class_weight'] = self.class_weight
            self.model = LinearSVC(**params)
            
        elif self.model_type == 'random_forest':
            params = {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            }
            params.update(self.model_params)
            if self.class_weight:
                params['class_weight'] = self.class_weight
            self.model = RandomForestClassifier(**params)
            
        elif self.model_type == 'gradient_boosting':
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
            params.update(self.model_params)
            self.model = GradientBoostingClassifier(**params)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SentimentClassifier':
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
            
        Returns:
            self
        """
        logger.info(f"开始训练 {self.model_type} 模型...")
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.trained = True
        logger.info(f"{self.model_type} 模型训练完成")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的类别
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        # 有些模型没有predict_proba方法，比如LinearSVC
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # 将decision_function的输出转换为概率
            decisions = self.model.decision_function(X)
            if decisions.ndim == 1:
                # 二分类情况
                probs = 1 / (1 + np.exp(-decisions))
                return np.vstack([1 - probs, probs]).T
            else:
                # 多分类情况
                exp_decisions = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
                return exp_decisions / np.sum(exp_decisions, axis=1, keepdims=True)
        else:
            raise NotImplementedError(f"{self.model_type}模型不支持概率预测")
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if not self.trained:
            raise ValueError("模型尚未训练，无法保存")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        joblib.dump(self.model, path)
        
        # 保存模型元数据
        metadata = {
            'model_type': self.model_type,
            'class_weight': self.class_weight,
            'model_params': self.model_params,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存元数据
        metadata_path = path + '.meta.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"模型已保存至 {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SentimentClassifier':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型
        """
        # 加载元数据
        metadata_path = path + '.meta.json'
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"找不到模型元数据文件: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 创建分类器实例
        classifier = cls(
            model_type=metadata['model_type'],
            class_weight=metadata['class_weight'],
            model_params=metadata['model_params']
        )
        
        # 加载模型
        classifier.model = joblib.load(path)
        classifier.classes_ = np.array(metadata['classes']) if metadata['classes'] else None
        classifier.trained = True
        
        logger.info(f"已加载{metadata['model_type']}模型")
        return classifier


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """集成分类器，将多个基础分类器的结果组合起来"""
    
    def __init__(self, models: List[SentimentClassifier], 
                 weights: Optional[List[float]] = None,
                 voting: str = 'soft'):
        """
        初始化集成分类器
        
        Args:
            models: 基础分类器列表
            weights: 各分类器的权重
            voting: 投票方式，'hard'或'soft'
        """
        self.models = models
        self.weights = weights
        self.voting = voting
        self.classes_ = None
        self.trained = all(model.trained for model in models)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleClassifier':
        """
        训练模型（如果尚未训练）
        
        Args:
            X: 特征矩阵
            y: 标签
            
        Returns:
            self
        """
        # 如果所有模型都已经训练过，则跳过
        if not self.trained:
            for i, model in enumerate(self.models):
                if not model.trained:
                    logger.info(f"训练集成分类器中的基础分类器 {i+1}/{len(self.models)}...")
                    model.fit(X, y)
        
        # 确保所有模型的类别标签一致
        classes = [model.classes_ for model in self.models]
        if not all(np.array_equal(c, classes[0]) for c in classes):
            raise ValueError("所有基础分类器必须具有相同的类别标签")
        
        self.classes_ = classes[0]
        self.trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的类别
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        if self.voting == 'hard':
            # 硬投票：选择被最多模型预测的类别
            predictions = np.array([model.predict(X) for model in self.models])
            if self.weights is not None:
                # 带权重的投票
                weights = np.array(self.weights)[:, np.newaxis]
                weighted_predictions = np.zeros_like(predictions, dtype=float)
                for i, class_val in enumerate(self.classes_):
                    weighted_predictions += weights * (predictions == class_val)
                return self.classes_[np.argmax(weighted_predictions, axis=0)]
            else:
                # 不带权重的投票
                return np.apply_along_axis(
                    lambda x: np.bincount(x, minlength=len(self.classes_)).argmax(),
                    axis=0,
                    arr=predictions
                )
        else:  # soft voting
            # 软投票：根据概率加权平均
            probas = self._predict_proba(X)
            return self.classes_[np.argmax(probas, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        return self._predict_proba(X)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        内部方法：预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率
        """
        # 获取所有模型的概率预测
        probas = [model.predict_proba(X) for model in self.models]
        
        # 应用权重
        if self.weights is not None:
            weights = np.array(self.weights) / sum(self.weights)
            # 加权平均
            return np.sum([w * p for w, p in zip(weights, probas)], axis=0)
        else:
            # 简单平均
            return np.mean(probas, axis=0)
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if not self.trained:
            raise ValueError("模型尚未训练，无法保存")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存每个基础模型
        model_paths = []
        for i, model in enumerate(self.models):
            model_path = os.path.join(os.path.dirname(path), f"base_model_{i}.joblib")
            model.save(model_path)
            model_paths.append(model_path)
        
        # 保存元数据
        metadata = {
            'model_paths': model_paths,
            'weights': self.weights,
            'voting': self.voting,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存元数据
        with open(path, 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"集成模型已保存至 {path}")
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleClassifier':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型
        """
        # 加载元数据
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到模型元数据文件: {path}")
        
        with open(path, 'r') as f:
            metadata = json.load(f)
        
        # 加载基础模型
        models = []
        for model_path in metadata['model_paths']:
            model = SentimentClassifier.load(model_path)
            models.append(model)
        
        # 创建实例
        ensemble = cls(
            models=models,
            weights=metadata['weights'],
            voting=metadata['voting']
        )
        
        ensemble.classes_ = np.array(metadata['classes']) if metadata['classes'] else None
        ensemble.trained = True
        
        logger.info(f"已加载集成模型")
        return ensemble


def create_optimized_model(model_type: str, param_grid: Dict[str, List[Any]], 
                          X_train: np.ndarray, y_train: np.ndarray,
                          cv: int = 5, scoring: str = 'f1_weighted',
                          n_jobs: int = -1) -> SentimentClassifier:
    """
    使用网格搜索创建优化的模型
    
    Args:
        model_type: 模型类型
        param_grid: 参数网格
        X_train: 训练特征
        y_train: 训练标签
        cv: 交叉验证折数
        scoring: 评分标准
        n_jobs: 并行作业数
        
    Returns:
        优化后的模型
    """
    # 创建基础模型
    base_model = SentimentClassifier(model_type=model_type)
    
    # 设置网格搜索
    grid_search = GridSearchCV(
        base_model.model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )
    
    # 训练并优化模型
    logger.info(f"使用网格搜索优化{model_type}模型...")
    grid_search.fit(X_train, y_train)
    
    # 获取最佳参数
    best_params = grid_search.best_params_
    logger.info(f"最佳参数: {best_params}")
    
    # 创建使用最佳参数的模型
    best_model = SentimentClassifier(model_type=model_type, model_params=best_params)
    best_model.fit(X_train, y_train)
    
    return best_model


def train_model_with_cv(model_types: List[str], X_train: np.ndarray, y_train: np.ndarray,
                      cv: int = 5, class_weight: Optional[str] = 'balanced',
                      use_ensemble: bool = True, save_dir: Optional[str] = None) -> Union[SentimentClassifier, EnsembleClassifier]:
    """
    使用交叉验证训练多个模型，可选择使用集成方法
    
    Args:
        model_types: 模型类型列表
        X_train: 训练特征
        y_train: 训练标签
        cv: 交叉验证折数
        class_weight: 类别权重
        use_ensemble: 是否使用集成
        save_dir: 模型保存目录
        
    Returns:
        训练好的模型
    """
    models = []
    
    for model_type in model_types:
        logger.info(f"训练{model_type}模型...")
        model = SentimentClassifier(model_type=model_type, class_weight=class_weight)
        model.fit(X_train, y_train)
        models.append(model)
        
        # 保存模型
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{model_type}_model.joblib")
            model.save(model_path)
    
    # 如果只训练了一个模型，直接返回
    if len(models) == 1:
        return models[0]
    
    # 使用集成方法
    if use_ensemble:
        logger.info("创建集成模型...")
        ensemble = EnsembleClassifier(models=models, voting='soft')
        
        # 保存集成模型
        if save_dir:
            ensemble_path = os.path.join(save_dir, "ensemble_model.json")
            ensemble.save(ensemble_path)
        
        return ensemble
    else:
        # 返回所有模型中最好的一个（这里简单地返回第一个模型）
        return models[0]


if __name__ == "__main__":
    # 示例用法
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    
    # 加载数据（这里使用20newsgroups作为示例）
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    
    # 提取特征
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data.data).toarray()
    y = data.target
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model_types = ['logistic_regression', 'naive_bayes', 'svm']
    sentiment_model = train_model_with_cv(
        model_types=model_types,
        X_train=X_train,
        y_train=y_train,
        use_ensemble=True
    )
    
    # 评估模型
    from sklearn.metrics import classification_report
    y_pred = sentiment_model.predict(X_test)
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))