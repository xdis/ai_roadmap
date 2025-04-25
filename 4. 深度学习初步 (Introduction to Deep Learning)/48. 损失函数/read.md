# 损失函数

## 1. 损失函数概述

损失函数(Loss Function)是深度学习中的核心组件，用于衡量模型预测值与真实值之间的差异。它为模型提供了一个优化目标，通过最小化损失函数，模型能够学习到更准确的参数。损失函数的选择直接影响模型的训练过程和最终性能。

### 1.1 损失函数的作用

- **提供优化目标**：为神经网络的训练提供明确的优化方向
- **量化模型表现**：数值化地评估模型预测的好坏
- **指导参数更新**：通过反向传播算法，损失函数的梯度指导权重的更新
- **反映任务目标**：不同的任务需要不同的损失函数来反映其特定目标

### 1.2 损失函数的数学表示

一般形式：
$$L(y, \hat{y}) = f(y, \hat{y})$$

其中：
- $L$ 是损失函数
- $y$ 是真实标签
- $\hat{y}$ 是模型预测值
- $f$ 是一个衡量差异的函数

## 2. 回归问题中的损失函数

回归问题旨在预测连续的数值，常用的损失函数包括：

### 2.1 均方误差(Mean Squared Error, MSE)

**数学表达式**:
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**特点**:
- 将误差平方，放大大误差的影响
- 计算简单，易于求导
- 凸函数，容易找到全局最小值

**优点**:
- 数学性质良好，导数连续
- 当数据符合高斯分布时，等价于最大似然估计

**缺点**:
- 对异常值敏感
- 单位与原始数据的平方有关，解释性不直观

**适用场景**:
- 回归问题，特别是误差服从高斯分布的情况
- 希望惩罚大误差的情况

**Python实现**:
```python
def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# PyTorch实现
import torch.nn as nn
criterion = nn.MSELoss()
loss = criterion(predictions, targets)

# TensorFlow/Keras实现
from tensorflow.keras.losses import MeanSquaredError
loss_fn = MeanSquaredError()
loss = loss_fn(y_true, y_pred)
```

### 2.2 平均绝对误差(Mean Absolute Error, MAE)

**数学表达式**:
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**特点**:
- 使用误差的绝对值，不会放大大误差的影响
- 在零点不可导，需要特殊处理

**优点**:
- 对异常值较MSE更鲁棒
- 单位与原始数据相同，解释性更强

**缺点**:
- 在0处不可微，优化算法可能会遇到困难
- 对小误差的惩罚较小，可能导致模型不够精确

**适用场景**:
- 回归问题，特别是存在异常值的情况
- 对绝对误差更关注的应用

**Python实现**:
```python
def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# PyTorch实现
import torch.nn as nn
criterion = nn.L1Loss()  # MAE也称为L1损失
loss = criterion(predictions, targets)

# TensorFlow/Keras实现
from tensorflow.keras.losses import MeanAbsoluteError
loss_fn = MeanAbsoluteError()
loss = loss_fn(y_true, y_pred)
```

### 2.3 平均绝对百分比误差(Mean Absolute Percentage Error, MAPE)

**数学表达式**:
$$MAPE = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$$

**特点**:
- 计算预测误差相对于真实值的百分比
- 结果以百分比表示，直观易懂

**优点**:
- 不受数据规模影响，可比较不同量级的误差
- 结果具有良好的解释性

**缺点**:
- 当真实值接近0时，会出现数值不稳定或无穷大的问题
- 对于低估和高估的惩罚不对称

**适用场景**:
- 对相对误差敏感的场景
- 具有较大数值范围的回归问题

**Python实现**:
```python
def mape_loss(y_true, y_pred):
    # 避免除以0
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# TensorFlow/Keras实现
from tensorflow.keras.losses import MeanAbsolutePercentageError
loss_fn = MeanAbsolutePercentageError()
loss = loss_fn(y_true, y_pred)
```

### 2.4 Huber损失(Huber Loss)

**数学表达式**:
$$
L_\delta(y, \hat{y}) = 
\begin{cases}
\frac{1}{2}(y - \hat{y})^2, & \text{if } |y - \hat{y}| \leq \delta \\
\delta(|y - \hat{y}| - \frac{1}{2}\delta), & \text{otherwise}
\end{cases}
$$

**特点**:
- 结合了MSE和MAE的优点
- 对于小误差使用MSE，对于大误差使用MAE

**优点**:
- 对异常值比MSE更鲁棒
- 比MAE更易于优化（处处可导）

**缺点**:
- 需要调整超参数δ
- 计算相对复杂

**适用场景**:
- 可能存在异常值的回归问题
- 希望平衡优化稳定性和对异常值的鲁棒性

**Python实现**:
```python
def huber_loss(y_true, y_pred, delta=1.0):
    residual = np.abs(y_true - y_pred)
    mask = residual <= delta
    squared_loss = 0.5 * np.square(residual)
    linear_loss = delta * (residual - 0.5 * delta)
    return np.mean(np.where(mask, squared_loss, linear_loss))

# PyTorch实现
import torch.nn as nn
criterion = nn.SmoothL1Loss()  # PyTorch中的Huber Loss实现
loss = criterion(predictions, targets)

# TensorFlow/Keras实现
from tensorflow.keras.losses import Huber
loss_fn = Huber(delta=1.0)
loss = loss_fn(y_true, y_pred)
```

### 2.5 Log-Cosh损失(Log-Cosh Loss)

**数学表达式**:
$$L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}\log(\cosh(y_i - \hat{y}_i))$$

**特点**:
- log(cosh(x))近似于x²/2（当x较小时）和|x|-log(2)（当x较大时）
- 结合了MSE和MAE的优点

**优点**:
- 处处二阶可导
- 对异常值比MSE更鲁棒
- 比Huber Loss不需要调整超参数

**缺点**:
- 计算成本相对较高

**适用场景**:
- 需要平滑梯度的回归问题
- 可能存在少量异常值的情况

**Python实现**:
```python
def logcosh_loss(y_true, y_pred):
    return np.mean(np.log(np.cosh(y_pred - y_true)))

# TensorFlow/Keras实现
from tensorflow.keras.losses import LogCosh
loss_fn = LogCosh()
loss = loss_fn(y_true, y_pred)
```

## 3. 分类问题中的损失函数

分类问题旨在将输入分配到不同的类别，常用的损失函数包括：

### 3.1 二元交叉熵(Binary Cross-Entropy)

**数学表达式**:
$$BCE = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**特点**:
- 用于二分类问题
- 与Sigmoid激活函数配合使用
- 基于信息论中的交叉熵概念

**优点**:
- 当预测概率偏离真实标签时，提供较大的梯度
- 理论上与最大似然估计相符
- 收敛通常比均方误差快

**缺点**:
- 对类别不平衡敏感
- 计算中需要避免数值问题（如log(0)）

**适用场景**:
- 二分类问题（输出为概率）
- 每个样本属于一个类别（互斥）

**Python实现**:
```python
def binary_crossentropy(y_true, y_pred):
    # 避免数值问题
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# PyTorch实现
import torch.nn as nn
criterion = nn.BCELoss()  # 需要先应用sigmoid激活
loss = criterion(predictions, targets)

# 或者直接使用BCEWithLogitsLoss（整合了sigmoid和BCE）
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets)

# TensorFlow/Keras实现
from tensorflow.keras.losses import BinaryCrossentropy
loss_fn = BinaryCrossentropy()
loss = loss_fn(y_true, y_pred)
```

### 3.2 分类交叉熵(Categorical Cross-Entropy)

**数学表达式**:
$$CCE = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{C}y_{ij}\log(\hat{y}_{ij})$$

其中C是类别数量，$y_{ij}$是第i个样本属于第j类的真实标签（通常是one-hot编码），$\hat{y}_{ij}$是预测概率。

**特点**:
- 用于多分类问题
- 与Softmax激活函数配合使用
- 计算每个类别预测概率与真实标签的交叉熵

**优点**:
- 对错误分类提供强梯度
- 直接优化概率输出

**缺点**:
- 计算one-hot编码可能浪费内存
- 对类别不平衡敏感

**适用场景**:
- 多分类问题
- 每个样本仅属于一个类别

**Python实现**:
```python
def categorical_crossentropy(y_true, y_pred):
    # 避免数值问题
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# PyTorch实现
import torch.nn as nn
criterion = nn.CrossEntropyLoss()  # 注意：直接接受logits，内部应用softmax
loss = criterion(logits, targets)  # targets是类别索引，非one-hot

# TensorFlow/Keras实现
from tensorflow.keras.losses import CategoricalCrossentropy
loss_fn = CategoricalCrossentropy()  # 假设输入已经过softmax
loss = loss_fn(y_true, y_pred)  # y_true是one-hot编码
```

### 3.3 稀疏分类交叉熵(Sparse Categorical Cross-Entropy)

**数学表达式**:
与分类交叉熵相同，但接受整数标签而非one-hot编码。

**特点**:
- 功能与分类交叉熵相同
- 使用整数类别标签而非one-hot编码

**优点**:
- 内存效率更高，特别是类别数量大时
- 在数值上等价于分类交叉熵

**缺点**:
- 在某些框架中实现可能不如标准交叉熵优化

**适用场景**:
- 类别数量大的多分类问题
- 内存有限的情况

**Python实现**:
```python
# PyTorch实现（等同于CrossEntropyLoss）
import torch.nn as nn
criterion = nn.CrossEntropyLoss()  # 接受类索引作为目标
loss = criterion(logits, class_indices)

# TensorFlow/Keras实现
from tensorflow.keras.losses import SparseCategoricalCrossentropy
loss_fn = SparseCategoricalCrossentropy()
loss = loss_fn(class_indices, predictions)  # class_indices是整数标签
```

### 3.4 Hinge损失(Hinge Loss)

**数学表达式**:
$$L(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})$$

其中，y∈{-1,1}是真实标签，$\hat{y}$是预测分数。

**特点**:
- 最初用于支持向量机
- 鼓励正确类别分数比错误类别至少高出一个边界值

**优点**:
- 对最大间隔分类有良好的泛化能力
- 对异常值相对鲁棒

**缺点**:
- 不直接输出概率
- 通常需要特殊的标签编码(-1和1)

**适用场景**:
- 二分类问题
- 支持向量机等最大间隔分类器

**Python实现**:
```python
def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

# PyTorch实现
def hinge_loss_torch(y_true, y_pred):
    return torch.mean(torch.clamp(1 - y_true * y_pred, min=0))

# TensorFlow/Keras实现
from tensorflow.keras.losses import Hinge
loss_fn = Hinge()
loss = loss_fn(y_true, y_pred)  # y_true应为{-1,1}
```

### 3.5 Focal损失(Focal Loss)

**数学表达式**:
$$FL(p_t) = -\alpha_t(1-p_t)^\gamma\log(p_t)$$

其中，$p_t$是正确类别的预测概率，$\alpha$是类别权重，$\gamma$是聚焦参数（通常为2）。

**特点**:
- 二元交叉熵的扩展
- 通过降低易分样本的权重，使损失函数聚焦于难分样本

**优点**:
- 解决了类别不平衡问题
- 自动调整简单样本和困难样本的权重

**缺点**:
- 需要调整额外的超参数
- 计算复杂度增加

**适用场景**:
- 类别严重不平衡的分类问题
- 目标检测等存在大量简单负样本的任务

**Python实现**:
```python
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    # 避免数值问题
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # 计算交叉熵CE
    ce = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    # 应用focal项
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal_term = np.power(1 - p_t, gamma)
    
    return np.mean(alpha_t * focal_term * ce)

# PyTorch实现
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        p_t = target * torch.sigmoid(input) + (1 - target) * (1 - torch.sigmoid(input))
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_term = torch.pow(1 - p_t, self.gamma)
        loss = alpha_t * focal_term * bce_loss
        return loss.mean()
```

## 4. 多任务学习和特殊任务的损失函数

### 4.1 多任务损失(Multi-task Loss)

**数学表达式**:
$$L_{total} = \sum_{i=1}^{k} w_i L_i$$

其中，$w_i$是第i个任务的权重，$L_i$是第i个任务的损失。

**特点**:
- 组合多个任务的损失
- 允许模型同时优化多个任务目标

**优点**:
- 支持多任务学习
- 可调整各任务的相对重要性

**缺点**:
- 需要平衡不同任务的贡献
- 可能存在任务冲突问题

**适用场景**:
- 多任务学习
- 同时包含回归和分类的问题

**Python实现**:
```python
def multi_task_loss(classification_loss, regression_loss, alpha=0.5):
    return alpha * classification_loss + (1 - alpha) * regression_loss

# PyTorch示例
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享层
        self.shared = nn.Sequential(...)
        # 任务特定层
        self.task1_head = nn.Sequential(...)
        self.task2_head = nn.Sequential(...)
        
    def forward(self, x):
        shared_features = self.shared(x)
        task1_pred = self.task1_head(shared_features)
        task2_pred = self.task2_head(shared_features)
        return task1_pred, task2_pred

# 损失计算
def compute_loss(task1_pred, task1_target, task2_pred, task2_target, w1=0.5, w2=0.5):
    task1_loss = nn.CrossEntropyLoss()(task1_pred, task1_target)
    task2_loss = nn.MSELoss()(task2_pred, task2_target)
    total_loss = w1 * task1_loss + w2 * task2_loss
    return total_loss
```

### 4.2 对比损失(Contrastive Loss)

**数学表达式**:
$$L(y, d) = (1-y)\frac{1}{2}d^2 + y\frac{1}{2}(\max(0, m-d))^2$$

其中，$y$是标签（0表示相似样本对，1表示不相似样本对），$d$是样本对的距离，$m$是边界参数。

**特点**:
- 用于学习样本对之间的相似性
- 推动相似样本更接近，不相似样本更远离

**优点**:
- 适用于相似性学习和度量学习
- 能够学习有效的特征表示

**缺点**:
- 需要样本对数据
- 超参数敏感

**适用场景**:
- 人脸识别
- 相似图像检索
- 孪生网络

**Python实现**:
```python
def contrastive_loss(y_true, distance, margin=1.0):
    similar_pair_loss = y_true * 0.5 * distance**2
    dissimilar_pair_loss = (1 - y_true) * 0.5 * np.maximum(0, margin - distance)**2
    return np.mean(similar_pair_loss + dissimilar_pair_loss)

# PyTorch实现
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, target):
        # 欧氏距离
        distance = F.pairwise_distance(output1, output2)
        # 损失计算
        loss = (1 - target) * torch.pow(distance, 2) + \
               target * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()
```

### 4.3 三元组损失(Triplet Loss)

**数学表达式**:
$$L(a, p, n) = \max(0, d(a, p) - d(a, n) + \text{margin})$$

其中，$a$是锚点样本，$p$是正样本（与锚点同类），$n$是负样本（与锚点不同类），$d$是距离函数。

**特点**:
- 使用三元组(锚点、正样本、负样本)训练
- 优化特征空间使同类样本接近，不同类样本远离

**优点**:
- 直接优化特征空间中的相对距离
- 比对比损失更有效

**缺点**:
- 需要特殊的三元组采样策略
- 训练可能不稳定

**适用场景**:
- 人脸识别
- 图像检索
- 度量学习

**Python实现**:
```python
def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = np.sum(np.square(anchor - positive))
    neg_dist = np.sum(np.square(anchor - negative))
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    return loss

# PyTorch实现
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        neg_dist = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()
```

### 4.4 KL散度损失(Kullback-Leibler Divergence)

**数学表达式**:
$$D_{KL}(P||Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

其中，$P$是真实分布，$Q$是预测分布。

**特点**:
- 衡量两个概率分布的差异
- 非对称度量

**优点**:
- 适用于概率分布的比较
- 在变分自编码器等生成模型中很有用

**缺点**:
- 非对称性可能导致训练不稳定
- 要求Q(i)>0（当P(i)>0时）

**适用场景**:
- 概率分布的比较
- 变分自编码器(VAE)
- 知识蒸馏

**Python实现**:
```python
def kl_divergence(p, q):
    # 避免数值问题
    epsilon = 1e-15
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))

# PyTorch实现
import torch.nn.functional as F
kl_loss = F.kl_div(pred_logits.log_softmax(dim=1), target_probs, reduction='batchmean')

# TensorFlow/Keras实现
from tensorflow.keras.losses import KLDivergence
loss_fn = KLDivergence()
loss = loss_fn(y_true, y_pred)
```

### 4.5 CTC损失(Connectionist Temporal Classification)

**特点**:
- 用于序列到序列的学习
- 允许输入和输出序列长度不同
- 不需要输入和输出的精确对齐

**优点**:
- 适用于语音识别、光学字符识别等序列任务
- 解决了序列对齐问题

**缺点**:
- 计算复杂度高
- 实现较为复杂

**适用场景**:
- 语音识别
- 手写文字识别
- 任何序列到序列的问题（尤其是未对齐的）

**Python实现**:
```python
# PyTorch实现
import torch.nn as nn
ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

# TensorFlow实现
import tensorflow as tf
loss = tf.nn.ctc_loss(
    labels=sparse_labels,
    logits=logits,
    label_length=label_lengths,
    logit_length=logit_lengths,
    blank_index=0
)
```

## 5. 损失函数的工程实践

### 5.1 损失函数的选择策略

1. **根据问题类型选择**:
   - 回归问题：MSE, MAE, Huber Loss
   - 二分类问题：Binary Cross-Entropy
   - 多分类问题：Categorical Cross-Entropy
   - 序列生成：CTC Loss, Sequence Loss
   - 距离学习：Contrastive Loss, Triplet Loss

2. **考虑特定需求**:
   - 对异常值敏感？→ 选择MAE或Huber Loss
   - 类别不平衡？→ 考虑Focal Loss或加权交叉熵
   - 多任务学习？→ 使用多任务损失函数

3. **实验比较**:
   - 在验证集上比较不同损失函数的性能
   - 考虑结合多个损失函数

### 5.2 自定义损失函数

**PyTorch示例**:
```python
class CustomLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomLoss, self).__init__()
        
    def forward(self, inputs, targets):
        # 自定义损失计算逻辑
        loss = ...
        return loss

# 使用方法
criterion = CustomLoss()
loss = criterion(model_output, target)
```

**TensorFlow/Keras示例**:
```python
# 函数式API
def custom_loss(y_true, y_pred):
    # 自定义损失计算逻辑
    loss = ...
    return loss

model.compile(optimizer='adam', loss=custom_loss)

# 类API
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_loss"):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred):
        # 自定义损失计算逻辑
        loss = ...
        return loss

model.compile(optimizer='adam', loss=CustomLoss())
```

### 5.3 损失函数的加权和组合

**加权交叉熵**:
```python
# PyTorch实现
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=2.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, input, target):
        loss = -self.pos_weight * target * torch.log(input) - \
               (1 - target) * torch.log(1 - input)
        return loss.mean()

# TensorFlow/Keras实现
from tensorflow.keras.losses import BinaryCrossentropy
loss_fn = BinaryCrossentropy(pos_weight=2.0)  # 正样本权重为2
```

**损失函数组合**:
```python
# 多个损失函数的加权组合
def combined_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    ce_loss = categorical_crossentropy(y_true, y_pred)
    dice_loss = 1 - dice_coefficient(y_true, y_pred)
    return alpha * ce_loss + beta * dice_loss
```

### 5.4 损失函数的可视化与分析

```python
import matplotlib.pyplot as plt
import numpy as np

# 定义一系列预测值
y_pred = np.linspace(0.01, 0.99, 100)

# 计算不同真实值下的损失
y_true_1 = np.ones_like(y_pred)  # 正类
y_true_0 = np.zeros_like(y_pred)  # 负类

# 计算二元交叉熵
bce_1 = -np.log(y_pred)  # 真实标签为1的BCE
bce_0 = -np.log(1 - y_pred)  # 真实标签为0的BCE

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(y_pred, bce_1, label='BCE (y_true=1)')
plt.plot(y_pred, bce_0, label='BCE (y_true=0)')
plt.xlabel('Predicted Probability')
plt.ylabel('Loss')
plt.title('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.show()
```

## 6. 前沿研究与发展趋势

### 6.1 基于深度学习的损失函数设计

- **学习型损失函数**：通过神经网络学习损失函数的参数
- **AutoML中损失函数的优化**：自动搜索最优损失函数

### 6.2 自监督学习中的对比损失

- **SimCLR**：使用对比损失进行自监督表示学习
- **MoCo**：使用动量对比学习表示

### 6.3 生成模型中的损失函数

- **GAN中的对抗损失**：判别器和生成器之间的博弈
- **变分自编码器中的重构损失和KL散度**

## 7. 总结与最佳实践

1. **理解任务目标**：选择与任务目标一致的损失函数
2. **考虑数据特性**：类别平衡、异常值、数据规模等
3. **多尝试对比**：不同损失函数可能对同一问题有不同的效果
4. **组合损失函数**：复杂任务可能需要组合多个损失函数
5. **关注梯度行为**：损失函数的梯度影响模型的学习过程
6. **定期监控**：训练过程中监控损失值的变化趋势
7. **结合正则化**：与适当的正则化方法结合使用

## 8. 代码实践：自定义损失函数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 示例：带焦点调整的交叉熵，同时处理类别不平衡
class FocalLossWithClassWeight(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None):
        super(FocalLossWithClassWeight, self).__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        
    def forward(self, input, target):
        # 计算交叉熵
        if self.class_weights is not None:
            # 创建与目标形状匹配的权重
            weights = target * self.class_weights[1] + (1 - target) * self.class_weights[0]
            ce_loss = F.binary_cross_entropy_with_logits(
                input, target, weight=weights, reduction='none'
            )
        else:
            ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        
        # 计算focal项
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        
        # 最终损失
        loss = focal_term * ce_loss
        return loss.mean()

# 示例使用
model = nn.Linear(10, 1)  # 简单二分类模型
criterion = FocalLossWithClassWeight(gamma=2.0, class_weights=torch.tensor([0.5, 2.0]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(100):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```