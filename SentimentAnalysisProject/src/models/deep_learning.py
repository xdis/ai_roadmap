"""
深度学习模型模块：实现基于深度学习的情感分析模型
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
import os
import json
import pickle
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Embedding, Dropout, GlobalMaxPooling1D, Input, Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau as KerasReduceLROnPlateau

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


#===============================#
#       PyTorch模型实现         #
#===============================#

class SentimentDataset(Dataset):
    """PyTorch情感数据集类"""
    
    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        初始化数据集
        
        Args:
            features: 特征矩阵
            labels: 标签向量（可选，用于训练集）
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            如果有标签，返回(特征, 标签)对；否则只返回特征
        """
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]


class MLP(nn.Module):
    """多层感知机模型"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                dropout_rate: float = 0.5):
        """
        初始化MLP模型
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            dropout_rate: Dropout概率
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # 构建层
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            模型输出
        """
        return self.layers(x)


class TextCNN(nn.Module):
    """文本CNN模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                filter_sizes: List[int], num_filters: int,
                output_dim: int, dropout_rate: float = 0.5,
                pretrained_embeddings: Optional[np.ndarray] = None):
        """
        初始化TextCNN模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            filter_sizes: 卷积核尺寸列表
            num_filters: 每种尺寸的卷积核数量
            output_dim: 输出维度
            dropout_rate: Dropout概率
            pretrained_embeddings: 预训练词嵌入（可选）
        """
        super(TextCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # 词嵌入层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=False  # 允许微调
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                     out_channels=num_filters, 
                     kernel_size=size)
            for size in filter_sizes
        ])
        
        # Dropout和全连接层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列，形状为[batch_size, seq_len]
            
        Returns:
            模型输出
        """
        # x: [batch_size, seq_len]
        
        # 获取词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 调整维度顺序以适应卷积层
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        # 卷积和池化
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # 每个conved: [batch_size, num_filters, seq_len - filter_size + 1]
        
        # 全局最大池化
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # 每个pooled: [batch_size, num_filters]
        
        # 拼接所有池化结果
        cat = torch.cat(pooled, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        
        # Dropout和分类
        dropped = self.dropout(cat)
        return self.fc(dropped)


class RNN(nn.Module):
    """循环神经网络模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                rnn_type: str = 'lstm', bidirectional: bool = True,
                num_layers: int = 1, dropout_rate: float = 0.5):
        """
        初始化RNN模型
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            rnn_type: RNN类型，'lstm'或'gru'
            bidirectional: 是否使用双向RNN
            num_layers: RNN层数
            dropout_rate: Dropout概率
        """
        super(RNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # RNN层
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=dropout_rate if num_layers > 1 else 0,
                             batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout_rate if num_layers > 1 else 0,
                            batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # 确定全连接层的输入维度
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Dropout和全连接层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为[batch_size, seq_len, input_dim]
            
        Returns:
            模型输出
        """
        # x: [batch_size, seq_len, input_dim]
        
        if self.rnn_type == 'lstm':
            # LSTM返回outputs和(hidden, cell)
            outputs, (hidden, _) = self.rnn(x)
        else:
            # GRU返回outputs和hidden
            outputs, hidden = self.rnn(x)
        
        # 对于双向RNN，连接正向和反向的最后一个隐藏状态
        if self.bidirectional:
            # hidden: [num_layers * 2, batch_size, hidden_dim]
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            # hidden: [batch_size, hidden_dim * 2]
        else:
            # 提取最后一层的隐藏状态
            hidden = hidden[-1, :, :]
            # hidden: [batch_size, hidden_dim]
        
        # Dropout和分类
        dropped = self.dropout(hidden)
        return self.fc(dropped)


class PyTorchSentimentModel:
    """PyTorch情感分析模型包装类"""
    
    def __init__(self, model_type: str = 'mlp', model_params: Optional[Dict[str, Any]] = None,
                device: Optional[str] = None):
        """
        初始化PyTorch情感分析模型
        
        Args:
            model_type: 模型类型，可选'mlp', 'textcnn', 'rnn'
            model_params: 模型参数
            device: 计算设备，可选'cpu'或'cuda'
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        
        # 如果没有指定设备，检查CUDA是否可用
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"使用设备: {self.device}")
        
        self.model = None
        self.trained = False
        self.classes_ = None
    
    def _initialize_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """
        根据模型类型初始化模型
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            
        Returns:
            初始化的模型
        """
        if self.model_type == 'mlp':
            # 默认参数
            params = {
                'hidden_dims': [256, 128],
                'dropout_rate': 0.5
            }
            params.update(self.model_params)
            
            model = MLP(
                input_dim=input_dim,
                hidden_dims=params['hidden_dims'],
                output_dim=output_dim,
                dropout_rate=params['dropout_rate']
            )
            
        elif self.model_type == 'textcnn':
            # 默认参数
            params = {
                'vocab_size': 10000,
                'embedding_dim': 300,
                'filter_sizes': [3, 4, 5],
                'num_filters': 100,
                'dropout_rate': 0.5,
                'pretrained_embeddings': None
            }
            params.update(self.model_params)
            
            model = TextCNN(
                vocab_size=params['vocab_size'],
                embedding_dim=params['embedding_dim'],
                filter_sizes=params['filter_sizes'],
                num_filters=params['num_filters'],
                output_dim=output_dim,
                dropout_rate=params['dropout_rate'],
                pretrained_embeddings=params['pretrained_embeddings']
            )
            
        elif self.model_type == 'rnn':
            # 默认参数
            params = {
                'hidden_dim': 128,
                'rnn_type': 'lstm',
                'bidirectional': True,
                'num_layers': 2,
                'dropout_rate': 0.5
            }
            params.update(self.model_params)
            
            model = RNN(
                input_dim=input_dim,
                hidden_dim=params['hidden_dim'],
                output_dim=output_dim,
                rnn_type=params['rnn_type'],
                bidirectional=params['bidirectional'],
                num_layers=params['num_layers'],
                dropout_rate=params['dropout_rate']
            )
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model.to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           batch_size: int = 32, epochs: int = 10, 
           validation_split: float = 0.1, early_stopping: bool = True,
           learning_rate: float = 1e-3, optimizer_type: str = 'adam',
           class_weights: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
            batch_size: 批大小
            epochs: 训练轮数
            validation_split: 验证集比例
            early_stopping: 是否使用早停
            learning_rate: 学习率
            optimizer_type: 优化器类型
            class_weights: 类别权重
            
        Returns:
            训练历史记录
        """
        # 构建类别映射
        unique_classes = np.unique(y)
        self.classes_ = unique_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        num_classes = len(unique_classes)
        
        # 将标签转换为模型可用的格式
        y_model = np.array([self.class_to_idx[cls] for cls in y])
        
        # 初始化模型
        input_dim = X.shape[1]
        self.model = self._initialize_model(input_dim, num_classes)
        
        # 分割训练集和验证集
        if validation_split > 0:
            val_size = int(len(X) * validation_split)
            train_size = len(X) - val_size
            
            # 随机分割
            indices = np.random.permutation(len(X))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            X_train, y_train = X[train_indices], y_model[train_indices]
            X_val, y_val = X[val_indices], y_model[val_indices]
            
            train_dataset = SentimentDataset(X_train, y_train)
            val_dataset = SentimentDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            # 使用所有数据进行训练
            train_dataset = SentimentDataset(X, y_model)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = None
        
        # 设置优化器
        if optimizer_type.lower() == 'adam':
            optimizer = Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == 'rmsprop':
            optimizer = RMSprop(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # 设置学习率调度器
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        # 设置损失函数（带有类别权重）
        if class_weights is not None:
            class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # 早停设置
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # 训练循环
        logger.info(f"开始训练 {self.model_type} 模型...")
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                # 将数据移动到设备
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 更新统计信息
                train_loss += loss.item() * batch_features.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # 计算训练指标
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # 验证
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        # 将数据移动到设备
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        # 前向传播
                        outputs = self.model(batch_features)
                        loss = criterion(outputs, batch_labels)
                        
                        # 更新统计信息
                        val_loss += loss.item() * batch_features.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()
                
                # 计算验证指标
                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # 更新学习率
                scheduler.step(val_loss)
                
                # 早停检查
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            break
                
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                          f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
        
        self.trained = True
        logger.info(f"{self.model_type} 模型训练完成")
        return history
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征矩阵
            batch_size: 批大小
            
        Returns:
            预测的类别
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练")
        
        # 评估模式
        self.model.eval()
        
        # 创建数据集和数据加载器
        dataset = SentimentDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # 预测
        predictions = []
        
        with torch.no_grad():
            for batch_features in dataloader:
                # 如果batch_features不是元组（没有标签）
                if not isinstance(batch_features, tuple):
                    batch_features = batch_features.to(self.device)
                    outputs = self.model(batch_features)
                    _, batch_preds = torch.max(outputs, 1)
                    predictions.extend(batch_preds.cpu().numpy())
        
        # 将索引转换回原始类别
        idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        return np.array([idx_to_class[idx] for idx in predictions])
    
    def predict_proba(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵
            batch_size: 批大小
            
        Returns:
            预测的概率
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练")
        
        # 评估模式
        self.model.eval()
        
        # 创建数据集和数据加载器
        dataset = SentimentDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # 预测
        probas = []
        
        with torch.no_grad():
            for batch_features in dataloader:
                # 如果batch_features不是元组（没有标签）
                if not isinstance(batch_features, tuple):
                    batch_features = batch_features.to(self.device)
                    outputs = self.model(batch_features)
                    batch_probas = F.softmax(outputs, dim=1)
                    probas.extend(batch_probas.cpu().numpy())
        
        return np.array(probas)
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型状态
        model_path = path + '.pt'
        torch.save(self.model.state_dict(), model_path)
        
        # 保存模型元数据
        metadata = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'class_to_idx': self.class_to_idx,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存元数据
        metadata_path = path + '.meta.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"模型已保存至 {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'PyTorchSentimentModel':
        """
        加载模型
        
        Args:
            path: 模型路径
            device: 计算设备
            
        Returns:
            加载的模型
        """
        # 加载元数据
        metadata_path = path + '.meta.json'
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"找不到模型元数据文件: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 创建模型实例
        model_instance = cls(
            model_type=metadata['model_type'],
            model_params=metadata['model_params'],
            device=device
        )
        
        # 设置类别信息
        model_instance.classes_ = np.array(metadata['classes']) if metadata['classes'] else None
        model_instance.class_to_idx = metadata['class_to_idx']
        
        # 加载模型权重
        model_path = path + '.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型权重文件: {model_path}")
        
        # 初始化模型架构
        input_dim = None
        if metadata['model_type'] == 'mlp':
            # 从模型参数中获取输入维度
            for param_name, param_value in metadata['model_params'].items():
                if param_name == 'input_dim':
                    input_dim = param_value
                    break
            
            # 如果未找到输入维度，使用默认值
            if input_dim is None:
                # 这里需要一个合理的默认值，或者可以在元数据中保存这个信息
                input_dim = 300  # 一个常见的词嵌入维度
        
        # 或者，可以强制要求用户在加载模型时提供输入维度
        if input_dim is None:
            raise ValueError("无法确定输入维度，请在元数据中指定或在加载时提供")
        
        # 初始化模型
        model_instance.model = model_instance._initialize_model(
            input_dim=input_dim,
            output_dim=len(model_instance.classes_)
        )
        
        # 加载模型权重
        model_instance.model.load_state_dict(torch.load(model_path, map_location=model_instance.device))
        model_instance.model.eval()  # 设置为评估模式
        model_instance.trained = True
        
        logger.info(f"已加载{metadata['model_type']}模型")
        return model_instance


#===============================#
#      TensorFlow模型实现       #
#===============================#

def create_mlp_model(input_dim: int, hidden_dims: List[int], output_dim: int, 
                    dropout_rate: float = 0.5) -> Model:
    """
    创建MLP模型
    
    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        dropout_rate: Dropout概率
        
    Returns:
        Keras模型
    """
    inputs = Input(shape=(input_dim,))
    x = inputs
    
    for hidden_dim in hidden_dims:
        x = Dense(hidden_dim, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    
    # 输出层
    outputs = Dense(output_dim, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)


def create_textcnn_model(max_sequence_length: int, vocab_size: int, embedding_dim: int, 
                        filter_sizes: List[int], num_filters: int, output_dim: int, 
                        dropout_rate: float = 0.5, 
                        embedding_matrix: Optional[np.ndarray] = None) -> Model:
    """
    创建TextCNN模型
    
    Args:
        max_sequence_length: 最大序列长度
        vocab_size: 词汇表大小
        embedding_dim: 词嵌入维度
        filter_sizes: 卷积核尺寸列表
        num_filters: 每种尺寸的卷积核数量
        output_dim: 输出维度
        dropout_rate: Dropout概率
        embedding_matrix: 预训练词嵌入矩阵
        
    Returns:
        Keras模型
    """
    # 输入层
    inputs = Input(shape=(max_sequence_length,), dtype='int32')
    
    # 嵌入层
    if embedding_matrix is not None:
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_sequence_length,
            trainable=False
        )
    else:
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_sequence_length
        )
    
    embedded = embedding_layer(inputs)
    
    # 应用不同尺寸的卷积核
    conv_blocks = []
    for filter_size in filter_sizes:
        conv = Conv1D(filters=num_filters,
                     kernel_size=filter_size,
                     activation='relu')(embedded)
        pool = GlobalMaxPooling1D()(conv)
        conv_blocks.append(pool)
    
    # 拼接不同卷积的结果
    concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    
    # Dropout和分类
    dropout = Dropout(dropout_rate)(concat)
    outputs = Dense(output_dim, activation='softmax')(dropout)
    
    return Model(inputs=inputs, outputs=outputs)


def create_rnn_model(input_dim: int, max_sequence_length: int, hidden_dim: int, 
                    output_dim: int, rnn_type: str = 'lstm', bidirectional: bool = True,
                    num_layers: int = 1, dropout_rate: float = 0.5) -> Model:
    """
    创建RNN模型
    
    Args:
        input_dim: 输入维度
        max_sequence_length: 最大序列长度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        rnn_type: RNN类型，'lstm'或'gru'
        bidirectional: 是否使用双向RNN
        num_layers: RNN层数
        dropout_rate: Dropout概率
        
    Returns:
        Keras模型
    """
    # 输入层
    inputs = Input(shape=(max_sequence_length, input_dim))
    
    # RNN层
    rnn_layer = LSTM if rnn_type.lower() == 'lstm' else GRU
    x = inputs
    
    for i in range(num_layers):
        return_sequences = i < num_layers - 1  # 最后一层不返回序列
        
        if bidirectional:
            x = Bidirectional(rnn_layer(hidden_dim, return_sequences=return_sequences, dropout=dropout_rate))(x)
        else:
            x = rnn_layer(hidden_dim, return_sequences=return_sequences, dropout=dropout_rate)(x)
    
    # Dropout和分类
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)


class TensorFlowSentimentModel:
    """TensorFlow情感分析模型包装类"""
    
    def __init__(self, model_type: str = 'mlp', model_params: Optional[Dict[str, Any]] = None):
        """
        初始化TensorFlow情感分析模型
        
        Args:
            model_type: 模型类型，可选'mlp', 'textcnn', 'rnn'
            model_params: 模型参数
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.trained = False
        self.classes_ = None
        
        # 检查GPU是否可用
        if tf.config.list_physical_devices('GPU'):
            logger.info("TensorFlow将使用GPU")
        else:
            logger.info("TensorFlow将使用CPU")
    
    def _create_model(self, input_dim: int, output_dim: int) -> Model:
        """
        根据模型类型创建模型
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            
        Returns:
            创建的模型
        """
        if self.model_type == 'mlp':
            # 默认参数
            params = {
                'hidden_dims': [256, 128],
                'dropout_rate': 0.5
            }
            params.update(self.model_params)
            
            return create_mlp_model(
                input_dim=input_dim,
                hidden_dims=params['hidden_dims'],
                output_dim=output_dim,
                dropout_rate=params['dropout_rate']
            )
            
        elif self.model_type == 'textcnn':
            # 默认参数
            params = {
                'max_sequence_length': 100,
                'vocab_size': 10000,
                'embedding_dim': 300,
                'filter_sizes': [3, 4, 5],
                'num_filters': 100,
                'dropout_rate': 0.5,
                'embedding_matrix': None
            }
            params.update(self.model_params)
            
            return create_textcnn_model(
                max_sequence_length=params['max_sequence_length'],
                vocab_size=params['vocab_size'],
                embedding_dim=params['embedding_dim'],
                filter_sizes=params['filter_sizes'],
                num_filters=params['num_filters'],
                output_dim=output_dim,
                dropout_rate=params['dropout_rate'],
                embedding_matrix=params['embedding_matrix']
            )
            
        elif self.model_type == 'rnn':
            # 默认参数
            params = {
                'max_sequence_length': 100,
                'hidden_dim': 128,
                'rnn_type': 'lstm',
                'bidirectional': True,
                'num_layers': 2,
                'dropout_rate': 0.5
            }
            params.update(self.model_params)
            
            return create_rnn_model(
                input_dim=input_dim,
                max_sequence_length=params['max_sequence_length'],
                hidden_dim=params['hidden_dim'],
                output_dim=output_dim,
                rnn_type=params['rnn_type'],
                bidirectional=params['bidirectional'],
                num_layers=params['num_layers'],
                dropout_rate=params['dropout_rate']
            )
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           batch_size: int = 32, epochs: int = 10, 
           validation_split: float = 0.1, early_stopping: bool = True,
           learning_rate: float = 1e-3, optimizer_type: str = 'adam',
           class_weights: Optional[Dict[int, float]] = None) -> Any:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
            batch_size: 批大小
            epochs: 训练轮数
            validation_split: 验证集比例
            early_stopping: 是否使用早停
            learning_rate: 学习率
            optimizer_type: 优化器类型
            class_weights: 类别权重
            
        Returns:
            训练历史记录
        """
        # 构建类别映射
        unique_classes = np.unique(y)
        self.classes_ = unique_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        num_classes = len(unique_classes)
        
        # 将标签转换为模型可用的格式（独热编码）
        y_model = np.array([self.class_to_idx[cls] for cls in y])
        y_one_hot = tf.keras.utils.to_categorical(y_model, num_classes=num_classes)
        
        # 初始化模型
        input_dim = X.shape[1]
        self.model = self._create_model(input_dim, num_classes)
        
        # 设置优化器
        if optimizer_type.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # 编译模型
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 设置回调
        callbacks = []
        
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ))
        
        callbacks.append(KerasReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ))
        
        # 训练模型
        logger.info(f"开始训练 {self.model_type} 模型...")
        history = self.model.fit(
            X, y_one_hot,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        self.trained = True
        logger.info(f"{self.model_type} 模型训练完成")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的类别
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练")
        
        # 预测概率
        probas = self.model.predict(X)
        
        # 获取最高概率的类别索引
        pred_indices = np.argmax(probas, axis=1)
        
        # 将索引转换回原始类别
        idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        return np.array([idx_to_class[idx] for idx in pred_indices])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练")
        
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if not self.trained or self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        model_path = path + '.h5'
        self.model.save(model_path)
        
        # 保存元数据
        metadata = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'class_to_idx': self.class_to_idx,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存元数据
        metadata_path = path + '.meta.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"模型已保存至 {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TensorFlowSentimentModel':
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
        
        # 创建模型实例
        model_instance = cls(
            model_type=metadata['model_type'],
            model_params=metadata['model_params']
        )
        
        # 设置类别信息
        model_instance.classes_ = np.array(metadata['classes']) if metadata['classes'] else None
        model_instance.class_to_idx = metadata['class_to_idx']
        
        # 加载模型
        model_path = path + '.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        model_instance.model = load_model(model_path)
        model_instance.trained = True
        
        logger.info(f"已加载{metadata['model_type']}模型")
        return model_instance


if __name__ == "__main__":
    # 示例用法
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # 生成一些示例数据
    texts = [
        "I love this movie, it's amazing!",
        "This movie is terrible, I hate it.",
        "The acting was good but the plot was confusing.",
        "Great special effects and awesome action scenes!",
        "Boring and predictable storyline.",
        "The characters were well developed and engaging.",
        "The soundtrack was amazing!",
        "Poor directing and terrible camera work.",
        "I would highly recommend this film to everyone!",
        "Don't waste your time on this movie."
    ]
    
    labels = ['positive', 'negative', 'neutral', 'positive', 'negative', 
              'positive', 'positive', 'negative', 'positive', 'negative']
    
    # 创建特征
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts).toarray()
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, random_state=42
    )
    
    # 使用PyTorch模型
    print("使用PyTorch模型:")
    pytorch_model = PyTorchSentimentModel(model_type='mlp')
    history = pytorch_model.fit(
        X_train, y_train,
        batch_size=2, epochs=10,
        validation_split=0.2
    )
    
    # 预测
    y_pred = pytorch_model.predict(X_test)
    print(f"预测结果: {y_pred}")
    
    # 使用TensorFlow模型
    print("\n使用TensorFlow模型:")
    tf_model = TensorFlowSentimentModel(model_type='mlp')
    history = tf_model.fit(
        X_train, y_train,
        batch_size=2, epochs=10,
        validation_split=0.2
    )
    
    # 预测
    y_pred = tf_model.predict(X_test)
    print(f"预测结果: {y_pred}")