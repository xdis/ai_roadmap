# 权重初始化方法 (Weight Initialization Methods)

## 1. 基础概念理解

### 1.1 权重初始化的重要性

权重初始化是深度学习中一个看似简单但极其关键的步骤，它对模型训练有着决定性影响：

- **训练稳定性**：适当的初始化可以防止梯度消失和爆炸问题
- **收敛速度**：良好的初始化可以加速模型收敛
- **泛化能力**：初始化也会影响模型的最终性能和泛化能力
- **突破局部最优**：合适的初始化能够帮助模型避开不良的局部最优解

神经网络训练本质上是一个优化过程，而初始状态（即权重的初始值）对优化结果有着深远影响。在深度学习早期，很多研究者发现深层网络难以训练，而合适的权重初始化是解决这一问题的关键因素之一。

### 1.2 为什么不能全部初始化为零或随机值

初始化时存在几个明显的误区：

**全部初始化为零或相同值**：
- 对称性问题：如果所有权重相同，那么同一层的所有神经元会学习相同的特征
- 网络表达能力受限：不同神经元无法学习不同特征，相当于降低了网络容量
- 梯度更新相同：反向传播时各神经元的梯度更新也相同，导致无法打破对称性

**简单随机初始化（如标准正态分布）**：
- 方差不当：如果权重方差太大，容易导致梯度爆炸；太小则可能引起梯度消失
- 不考虑网络结构：不同深度和宽度的网络层需要不同的初始化方差
- 不考虑激活函数：不同的激活函数对初始权重的要求也不同

### 1.3 权重初始化的基本目标

有效的权重初始化方法应当满足以下目标：

1. **维持前向传播中激活值的分布**：防止激活值爆炸或消失
2. **维持反向传播中梯度的分布**：防止梯度爆炸或消失
3. **打破网络的对称性**：使不同神经元能学习不同特征
4. **考虑网络架构特性**：适应不同的网络深度、宽度和激活函数

这些目标使得权重初始化成为一个需要精心设计的技术问题，而不仅仅是简单的随机赋值过程。

### 1.4 初始化如何影响梯度流和信息流

从信息流角度看，神经网络包括两个方向的传播：

**前向传播**（输入→输出）：
- 如果权重过大：激活函数可能饱和（如sigmoid），导致输入信息丢失
- 如果权重过小：深层网络中信号可能变得太微弱而无法传递有效信息

**反向传播**（输出→输入）：
- 如果权重过大：梯度可能爆炸，导致训练不稳定
- 如果权重过小：梯度可能消失，导致深层网络参数无法有效更新

理想的权重初始化应保证信息（前向）和梯度（反向）都能在网络中顺畅流动，不会因网络的深度而显著衰减或放大。

## 2. 常见的权重初始化方法

### 2.1 传统随机初始化

最简单的初始化方法是从固定范围的均匀分布或正态分布中采样：

**均匀分布初始化**：
```python
# PyTorch实现
nn.init.uniform_(layer.weight, a=-0.05, b=0.05)
```

**正态分布初始化**：
```python
# PyTorch实现
nn.init.normal_(layer.weight, mean=0, std=0.05)
```

这些方法简单但效果有限，尤其对于较深的网络。它们没有考虑网络的深度和宽度，也没有针对不同的激活函数进行特殊设计。

### 2.2 Xavier/Glorot初始化

Xavier初始化（也称Glorot初始化）是首个专门为深度网络设计的权重初始化方法，由Xavier Glorot和Yoshua Bengio在2010年提出。它的核心思想是：保持每一层输入和输出的方差一致。

**理论依据**：

对于一个神经网络层，如果有$n_{in}$个输入，$n_{out}$个输出，Xavier初始化认为权重应该满足：

- 均值为0
- 方差为$\frac{2}{n_{in} + n_{out}}$

**均匀分布版本**：
```python
# 手动实现
limit = np.sqrt(6 / (fan_in + fan_out))
weights = np.random.uniform(-limit, limit, (fan_in, fan_out))

# PyTorch实现
nn.init.xavier_uniform_(layer.weight)
```

**正态分布版本**：
```python
# 手动实现
std = np.sqrt(2 / (fan_in + fan_out))
weights = np.random.normal(0, std, (fan_in, fan_out))

# PyTorch实现
nn.init.xavier_normal_(layer.weight)
```

Xavier初始化特别适合于线性激活或tanh、sigmoid等饱和激活函数，它的优点是考虑了网络结构（输入输出维度）。

### 2.3 He初始化（Kaiming初始化）

He初始化是由何恺明（Kaiming He）等人在2015年提出的，专门为ReLU激活函数设计，考虑了ReLU对分布的影响（将约一半的激活值置为0）。

**理论依据**：

对于使用ReLU激活的网络层，权重应该满足：

- 均值为0
- 方差为$\frac{2}{n_{in}}$（仅考虑输入维度）

**均匀分布版本**：
```python
# 手动实现
limit = np.sqrt(6 / fan_in)
weights = np.random.uniform(-limit, limit, (fan_in, fan_out))

# PyTorch实现
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
```

**正态分布版本**：
```python
# 手动实现
std = np.sqrt(2 / fan_in)
weights = np.random.normal(0, std, (fan_in, fan_out))

# PyTorch实现
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

He初始化特别适合于ReLU及其变体（如Leaky ReLU, PReLU等），在使用这些激活函数的深层网络中表现出色。

### 2.4 LeCun初始化

LeCun初始化由Yann LeCun提出，特别适合于sigmoid激活函数。其思想是保持每层输出的方差为1。

**理论依据**：

- 均值为0
- 方差为$\frac{1}{n_{in}}$

```python
# 手动实现
std = np.sqrt(1 / fan_in)
weights = np.random.normal(0, std, (fan_in, fan_out))

# PyTorch实现
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
```

### 2.5 正交初始化（Orthogonal Initialization）

正交初始化特别适用于RNN等循环网络，它保证权重矩阵是正交的，有助于保持梯度范数在反向传播过程中的稳定性。

```python
# 手动实现
random_matrix = np.random.randn(fan_out, fan_in)
u, _, vh = np.linalg.svd(random_matrix, full_matrices=False)
weights = u @ vh  # 正交矩阵

# PyTorch实现
nn.init.orthogonal_(layer.weight, gain=1.0)
```

正交初始化的一个关键优势是它可以防止长序列训练中的梯度消失/爆炸问题，对RNN、LSTM和GRU特别有效。

### 2.6 不同初始化方法的对比

| 初始化方法 | 适用激活函数 | 核心思想 | 优点 | 缺点 |
|------------|--------------|----------|------|------|
| 标准随机   | 无特定       | 简单随机分布 | 简单易实现 | 不考虑网络结构 |
| Xavier/Glorot | tanh, sigmoid | 保持输入输出方差一致 | 适合中等深度网络 | 不适合ReLU |
| He (Kaiming) | ReLU及变体 | 考虑ReLU的影响 | 适合深层ReLU网络 | 对其他激活函数效果不佳 |
| LeCun | sigmoid | 保持输出方差为1 | 适合sigmoid网络 | 较少使用 |
| 正交 | 适合RNN | 保持权重矩阵正交 | 防止循环网络中的梯度问题 | 计算开销较大 |

## 3. 数学原理与证明

### 3.1 前向和反向传播中的方差分析

要理解不同初始化方法的数学原理，需要分析权重对前向和反向传播中信号方差的影响。

**前向传播方差分析**：

考虑一个线性层：$y = Wx$，其中$W$是权重矩阵，$x$是输入向量。

如果$x$的各个元素是独立同分布的，均值为0，方差为$\sigma_x^2$，且$W$的元素均值为0，方差为$\sigma_w^2$，则$y$的方差：

$$\text{Var}(y_i) = n_{in} \cdot \sigma_w^2 \cdot \sigma_x^2$$

为了保持方差在各层之间一致（假设$\sigma_x^2 = 1$），我们需要：

$$\sigma_w^2 = \frac{1}{n_{in}}$$

**反向传播方差分析**：

梯度反向传播时，有：

$$\frac{\partial L}{\partial x} = W^T \frac{\partial L}{\partial y}$$

分析表明，为保持梯度方差稳定，应满足：

$$\sigma_w^2 = \frac{1}{n_{out}}$$

结合以上两个条件，Xavier初始化设定：

$$\sigma_w^2 = \frac{2}{n_{in} + n_{out}}$$

作为前向和反向传播需求的一个平衡。

### 3.2 Xavier/Glorot初始化的数学证明

Xavier初始化的核心假设是：
1. 激活函数大约是线性的（如tanh在原点附近）
2. 权重是零均值的
3. 需要在前向和反向传播中保持方差一致

推导过程（简化版）：

1. 对于线性变换$y = Wx + b$，其中$x$的各元素独立同分布，均值为0，方差为$\sigma_x^2$
2. 假设权重$W$的元素独立同分布，均值为0，方差为$\sigma_w^2$
3. 则$y$的方差为$\text{Var}(y_i) = n_{in} \cdot \sigma_w^2 \cdot \sigma_x^2$
4. 为使$\text{Var}(y_i) = \text{Var}(x_i) = \sigma_x^2$，需要$n_{in} \cdot \sigma_w^2 = 1$
5. 同理，反向传播时为保持梯度方差，需要$n_{out} \cdot \sigma_w^2 = 1$
6. 综合两个条件，取折中：$\sigma_w^2 = \frac{2}{n_{in} + n_{out}}$

这就是Xavier初始化的理论基础。

### 3.3 He初始化的数学证明

He初始化考虑了ReLU激活函数的特性，即约一半的神经元输出为0：

1. 对于$y = \text{ReLU}(Wx + b)$，ReLU将约一半的值置为0
2. 这意味着方差会减半：$\text{Var}(\text{ReLU}(z)) \approx \frac{1}{2}\text{Var}(z)$（对于零均值的$z$）
3. 为补偿这一点，初始方差应该加倍：$\sigma_w^2 = \frac{2}{n_{in}}$

这就是He初始化比Xavier初始化多出一个因子2的原因，专门为ReLU设计。

### 3.4 正交初始化的性质

正交矩阵有几个关键性质：
1. $W^TW = WW^T = I$（单位矩阵）
2. 它保持向量的长度：$\|Wx\|_2 = \|x\|_2$

这意味着当权重是正交矩阵时，信号在前向传播和梯度在反向传播过程中的范数不会改变，有效防止梯度消失/爆炸。

对于RNN，当权重矩阵是正交的时，状态转移不会导致信息丢失或爆炸，使长期依赖的学习更加稳定。

## 4. 实际实现与代码示例

### 4.1 PyTorch中的权重初始化

PyTorch提供了全面的权重初始化函数：

```python
import torch
import torch.nn as nn
import torch.nn.init as init

# 创建一个简单网络
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, init_method='xavier_uniform'):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 初始化权重
        self.initialize_weights(init_method)
    
    def initialize_weights(self, method):
        if method == 'xavier_uniform':
            init.xavier_uniform_(self.fc1.weight)
            init.xavier_uniform_(self.fc2.weight)
        elif method == 'xavier_normal':
            init.xavier_normal_(self.fc1.weight)
            init.xavier_normal_(self.fc2.weight)
        elif method == 'kaiming_uniform':
            init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
            init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        elif method == 'kaiming_normal':
            init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
            init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        elif method == 'orthogonal':
            init.orthogonal_(self.fc1.weight)
            init.orthogonal_(self.fc2.weight)
        else:
            raise ValueError(f"Unsupported initialization method: {method}")
        
        # 偏置通常初始化为零或小常数
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建不同初始化的模型
input_size, hidden_size, output_size = 784, 256, 10
model_xavier = SimpleNet(input_size, hidden_size, output_size, 'xavier_uniform')
model_kaiming = SimpleNet(input_size, hidden_size, output_size, 'kaiming_normal')
model_orthogonal = SimpleNet(input_size, hidden_size, output_size, 'orthogonal')
```

### 4.2 TensorFlow中的权重初始化

TensorFlow/Keras也提供了多种初始化器：

```python
import tensorflow as tf
from tensorflow.keras import layers, initializers

# 使用不同初始化器创建模型
def create_model(init_method='glorot_uniform'):
    # 选择初始化器
    if init_method == 'glorot_uniform':  # 等同于xavier_uniform
        initializer = initializers.GlorotUniform()
    elif init_method == 'glorot_normal':  # 等同于xavier_normal
        initializer = initializers.GlorotNormal()
    elif init_method == 'he_uniform':  # 等同于kaiming_uniform
        initializer = initializers.HeUniform()
    elif init_method == 'he_normal':  # 等同于kaiming_normal
        initializer = initializers.HeNormal()
    elif init_method == 'orthogonal':
        initializer = initializers.Orthogonal()
    else:
        raise ValueError(f"Unsupported initialization method: {init_method}")
    
    # 创建模型
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,),
                    kernel_initializer=initializer, bias_initializer='zeros'),
        layers.Dense(10, kernel_initializer=initializer, bias_initializer='zeros')
    ])
    
    return model

# 创建不同初始化的模型
model_glorot = create_model('glorot_uniform')
model_he = create_model('he_normal')
model_orthogonal = create_model('orthogonal')
```

### 4.3 自定义初始化方法实现

实现自定义初始化方法可以帮助理解初始化的核心原理：

```python
import numpy as np
import torch.nn as nn

# 自定义Xavier初始化
def custom_xavier_init(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        fan_in = layer.weight.size(1)  # 输入特征维度
        if isinstance(layer, nn.Conv2d):
            fan_in *= layer.weight.size(2) * layer.weight.size(3)  # 考虑卷积核尺寸
        fan_out = layer.weight.size(0)  # 输出特征维度
        if isinstance(layer, nn.Conv2d):
            fan_out *= layer.weight.size(2) * layer.weight.size(3)  # 考虑卷积核尺寸
        
        # 计算标准差
        std = np.sqrt(2.0 / (fan_in + fan_out))
        
        # 使用正态分布初始化
        nn.init.normal_(layer.weight, mean=0, std=std)
        
        # 初始化偏置为零
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

# 自定义He初始化
def custom_he_init(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        fan_in = layer.weight.size(1)  # 输入特征维度
        if isinstance(layer, nn.Conv2d):
            fan_in *= layer.weight.size(2) * layer.weight.size(3)
        
        # 计算标准差
        std = np.sqrt(2.0 / fan_in)
        
        # 使用正态分布初始化
        nn.init.normal_(layer.weight, mean=0, std=std)
        
        # 初始化偏置为零
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

# 应用到模型的所有层
def initialize_model(model, init_method='xavier'):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if init_method == 'xavier':
                custom_xavier_init(m)
            elif init_method == 'he':
                custom_he_init(m)
            elif init_method == 'orthogonal':
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
```

### 4.4 CNN中的权重初始化

卷积神经网络在初始化时需要考虑卷积核的维度：

```python
class ConvNet(nn.Module):
    def __init__(self, init_method='kaiming'):
        super(ConvNet, self).__init__()
        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
        # 初始化权重
        self.initialize_weights(init_method)
    
    def initialize_weights(self, method):
        # 根据方法选择初始化函数
        if method == 'kaiming':
            init_fn = lambda w: nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        elif method == 'xavier':
            init_fn = nn.init.xavier_normal_
        elif method == 'orthogonal':
            init_fn = nn.init.orthogonal_
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # 对每层应用初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_fn(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_fn(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 4.5 RNN/LSTM中的权重初始化

循环神经网络需要特别注意初始化，以防止长序列训练中的梯度问题：

```python
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 创建LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # 初始化
        self.initialize_weights()
    
    def initialize_weights(self):
        # 对LSTM使用正交初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # 输入到隐藏状态的权重
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # 隐藏状态到隐藏状态的权重（循环权重）
                nn.init.orthogonal_(param)  # 使用正交初始化
            elif 'bias' in name:
                nn.init.zeros_(param)  # 偏置初始化为零
        
        # 全连接层使用Xavier初始化
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out
```

## 5. 权重初始化的进阶技术

### 5.1 LSUV（Layer-Sequential Unit-Variance）初始化

LSUV是一种数据驱动的初始化方法，它首先使用正交初始化，然后通过实际数据调整每层的权重使输出方差为1：

```python
def lsuv_initialization(model, data_loader, device='cuda', needed_variance=1.0, epsilon=1e-8, max_attempts=10):
    """
    LSUV初始化实现
    """
    # 获取一批数据
    x_batch = next(iter(data_loader))[0].to(device)
    
    # 收集需要初始化的层
    layers_to_init = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight'):
            layers_to_init.append(module)
    
    # 先使用正交初始化
    for layer in layers_to_init:
        nn.init.orthogonal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    # 逐层调整方差
    model.to(device)
    model.eval()  # 设置为评估模式
    
    for layer_idx, layer in enumerate(layers_to_init):
        x_clone = x_batch.clone()
        
        # 创建前向钩子以获取当前层的输出
        outputs = []
        def hook_fn(module, input, output):
            outputs.append(output.detach())
        
        # 注册钩子
        hook = layer.register_forward_hook(hook_fn)
        
        # 前向传播以获取当前层输出
        with torch.no_grad():
            _ = model(x_clone)
        
        # 获取当前层输出
        curr_out = outputs[0]
        
        # 如果是卷积层，需要重新组织输出
        if isinstance(layer, nn.Conv2d):
            curr_out = curr_out.transpose(0, 1).contiguous().view(curr_out.size(1), -1).transpose(0, 1)
        
        # 计算当前方差
        curr_var = torch.var(curr_out)
        
        # 调整权重以达到目标方差
        attempts = 0
        while abs(curr_var - needed_variance) > epsilon and attempts < max_attempts:
            # 计算缩放因子
            scale = torch.sqrt(needed_variance / (curr_var + epsilon))
            
            # 缩放权重
            with torch.no_grad():
                layer.weight.data *= scale
            
            # 再次前向传播以获取新的输出
            outputs = []
            _ = model(x_clone)
            curr_out = outputs[0]
            
            # 重新计算方差
            if isinstance(layer, nn.Conv2d):
                curr_out = curr_out.transpose(0, 1).contiguous().view(curr_out.size(1), -1).transpose(0, 1)
            curr_var = torch.var(curr_out)
            
            attempts += 1
        
        # 移除钩子
        hook.remove()
        
        print(f"Layer {layer_idx}: final variance = {curr_var.item():.4f} after {attempts} attempts")
    
    return model
```

LSUV的优势在于它根据实际数据调整权重，比纯粹的理论方法更有针对性。

### 5.2 谱归一化（Spectral Normalization）

谱归一化通过对权重矩阵的最大奇异值进行归一化，控制网络的Lipschitz常数，提高训练稳定性：

```python
class SpectralNorm:
    def __init__(self, module, name='weight', power_iterations=1):
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        # 获取原始权重
        weight = getattr(self.module, self.name)
        
        # 参数维度
        height = weight.shape[0]
        width = weight.view(height, -1).shape[1]
        
        # 随机初始化u和v向量
        u = nn.Parameter(torch.randn(height, 1), requires_grad=False)
        v = nn.Parameter(torch.randn(width, 1), requires_grad=False)
        
        # 将u和v正规化
        u.data = F.normalize(u.data, dim=0, eps=1e-12)
        v.data = F.normalize(v.data, dim=0, eps=1e-12)
        
        # 注册缓冲区
        self.module.register_buffer('u', u)
        self.module.register_buffer('v', v)
        
        self.original_forward = module.forward
        module.forward = self.forward_with_spectral_norm
    
    def _power_method(self, weight):
        """幂法计算最大奇异值"""
        for _ in range(self.power_iterations):
            # v = W^T u / ||W^T u||
            v = F.normalize(torch.matmul(weight.t(), self.module.u), dim=0, eps=1e-12)
            # u = W v / ||W v||
            u = F.normalize(torch.matmul(weight, v), dim=0, eps=1e-12)
        
        # 更新u和v
        self.module.v.data = v.data
        self.module.u.data = u.data
        
        # 计算奇异值 σ = u^T W v
        sigma = torch.matmul(torch.matmul(u.t(), weight), v)
        return sigma.item()
    
    def forward_with_spectral_norm(self, *args, **kwargs):
        """前向传播时应用谱归一化"""
        weight = getattr(self.module, self.name)
        weight_mat = weight.view(weight.shape[0], -1)
        
        # 计算最大奇异值
        sigma = self._power_method(weight_mat)
        
        # 归一化权重
        weight_sn = weight / sigma
        
        # 替换原始权重
        setattr(self.module, self.name, weight_sn)
        
        # 调用原始前向传播
        output = self.original_forward(*args, **kwargs)
        
        # 恢复原始权重
        setattr(self.module, self.name, weight)
        
        return output

# 示例用法
def apply_spectral_norm(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            SpectralNorm(module)
    return model
```

谱归一化在GAN训练和增强网络鲁棒性方面表现出色。

### 5.3 DeltaOrthogonal初始化

DeltaOrthogonal初始化是为循环神经网络设计的高级初始化方法，它将循环权重矩阵初始化为单位矩阵加上正交矩阵的缩放版本：

```python
def delta_orthogonal_init(weight, gain=1.0, delta=0.0):
    """DeltaOrthogonal初始化"""
    rows, cols = weight.shape
    
    # 生成随机正交矩阵
    if rows < cols:
        # 如果行数小于列数，先生成方阵再裁剪
        orthogonal = torch.nn.init._generate_matrix_from_distribution(
            rows, rows, torch.randn, gain)
        orthogonal = orthogonal.triu(1)  # 上三角（不含对角线）
        orthogonal = orthogonal / (torch.norm(orthogonal, 2) + 1e-10)  # 归一化
        
        weight.data.copy_(orthogonal)
    else:
        # 如果行数大于等于列数
        orthogonal = torch.nn.init._generate_matrix_from_distribution(
            cols, cols, torch.randn, gain)
        orthogonal = orthogonal.triu(1)  # 上三角（不含对角线）
        orthogonal = orthogonal / (torch.norm(orthogonal, 2) + 1e-10)  # 归一化
        
        # 将矩阵填充到所需尺寸
        weight.data.zero_()
        weight.data[:cols, :cols].copy_(torch.eye(cols) * delta + orthogonal)
    
    return weight

# 对LSTM应用DeltaOrthogonal初始化
def apply_delta_orthogonal_to_lstm(lstm, gain=1.0, delta=0.001):
    """对LSTM应用DeltaOrthogonal初始化"""
    for name, param in lstm.named_parameters():
        if 'weight_hh' in name:  # 循环权重
            hidden_size = param.shape[0] // 4  # LSTM有4个门
            
            # 对每个门分别初始化
            for i in range(4):
                delta_orthogonal_init(
                    param.data[i*hidden_size:(i+1)*hidden_size, :],
                    gain=gain,
                    delta=delta if i == 2 else 0.0  # 仅对遗忘门使用delta
                )
```

DeltaOrthogonal初始化特别有助于解决循环神经网络中的长期依赖问题。

### 5.4 神经网络架构搜索中的初始化方法

神经架构搜索（NAS）中，初始化方法也是重要的超参数。下面实现了一个简单的搜索框架：

```python
def evaluate_initialization(model_class, init_method, train_loader, val_loader, 
                           epochs=5, device='cuda', lr=0.001):
    """评估特定初始化方法的性能"""
    model = model_class().to(device)
    
    # 应用初始化方法
    if init_method == 'xavier':
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif init_method == 'kaiming':
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif init_method == 'orthogonal':
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    else:
        raise ValueError(f"Unsupported method: {init_method}")
    
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 训练和评估
    best_val_acc = 0.0
    training_curve = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        
        training_curve.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return {
        'best_val_acc': best_val_acc,
        'training_curve': training_curve,
        'init_method': init_method
    }

# 搜索最佳初始化方法
def find_best_initialization(model_class, train_loader, val_loader):
    """搜索最佳初始化方法"""
    init_methods = ['xavier', 'kaiming', 'orthogonal']
    results = []
    
    for init_method in init_methods:
        print(f"Evaluating {init_method} initialization...")
        result = evaluate_initialization(model_class, init_method, train_loader, val_loader)
        results.append(result)
        print(f"Best validation accuracy: {result['best_val_acc']:.2f}%")
    
    # 找出最佳方法
    best_result = max(results, key=lambda x: x['best_val_acc'])
    print(f"\nBest initialization method: {best_result['init_method']} "
          f"with validation accuracy: {best_result['best_val_acc']:.2f}%")
    
    # 可视化比较
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for result in results:
        plt.plot([data['epoch'] for data in result['training_curve']],
                [data['train_loss'] for data in result['training_curve']],
                label=result['init_method'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for result in results:
        plt.plot([data['epoch'] for data in result['training_curve']],
                [data['val_acc'] for data in result['training_curve']],
                label=result['init_method'])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return best_result
```

这种搜索方法可以帮助为特定模型和任务找到最佳的初始化策略。

## 6. 初始化策略的实验与分析

### 6.1 不同初始化对模型收敛的影响

实验比较不同初始化方法的收敛速度和稳定性：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, 
                                         shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, 
                                        shuffle=False, num_workers=2)

# CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化方法
init_methods = {
    'xavier_uniform': lambda m: nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None,
    'xavier_normal': lambda m: nn.init.xavier_normal_(m.weight) if hasattr(m, 'weight') else None,
    'kaiming_uniform': lambda m: nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu') if hasattr(m, 'weight') else None,
    'kaiming_normal': lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu') if hasattr(m, 'weight') else None,
    'orthogonal': lambda m: nn.init.orthogonal_(m.weight) if hasattr(m, 'weight') else None,
    'uniform_small': lambda m: nn.init.uniform_(m.weight, -0.01, 0.01) if hasattr(m, 'weight') else None,
}

# 训练函数
def train_model(model, trainloader, criterion, optimizer, device, epochs=10):
    model.to(device)
    
    # 记录训练过程
    history = {
        'train_loss': [],
        'train_acc': [],
        'time_per_epoch': []
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_time = time.time() - start_time
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        # 记录结果
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['time_per_epoch'].append(epoch_time)
        
        print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}, '
              f'Acc: {train_acc:.2f}%, Time: {epoch_time:.2f}s')
    
    return history

# 评估函数
def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    return acc

# 主实验
def run_initialization_experiment(epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    for name, init_fn in init_methods.items():
        print(f"\nTraining with {name} initialization...")
        
        # 创建模型并应用初始化
        model = CNN()
        
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_fn(m)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 训练模型
        history = train_model(model, trainloader, criterion, optimizer, device, epochs)
        
        # 评估模型
        test_acc = evaluate_model(model, testloader, device)
        print(f"Test accuracy with {name} initialization: {test_acc:.2f}%")
        
        # 保存结果
        history['test_acc'] = test_acc
        results[name] = history
    
    # 可视化比较
    plt.figure(figsize=(18, 5))
    
    # 训练损失比较
    plt.subplot(1, 3, 1)
    for name, history in results.items():
        plt.plot(range(1, epochs+1), history['train_loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    
    # 训练准确率比较
    plt.subplot(1, 3, 2)
    for name, history in results.items():
        plt.plot(range(1, epochs+1), history['train_acc'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    
    # 每轮训练时间比较
    plt.subplot(1, 3, 3)
    avg_times = {name: sum(history['time_per_epoch'])/len(history['time_per_epoch']) 
                for name, history in results.items()}
    names = list(avg_times.keys())
    times = list(avg_times.values())
    plt.bar(names, times)
    plt.xlabel('Initialization Method')
    plt.ylabel('Average Time per Epoch (s)')
    plt.title('Training Efficiency Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('initialization_comparison.png')
    plt.show()
    
    # 输出最终测试准确率
    print("\nFinal test accuracies:")
    for name, history in results.items():
        print(f"{name}: {history['test_acc']:.2f}%")
    
    return results

# 运行实验
results = run_initialization_experiment(epochs=10)
```

### 6.2 超深网络中初始化的重要性

在超深网络（如ResNet-152或更深）中，初始化的重要性进一步放大：

```python
class DeepResNet(nn.Module):
    def __init__(self, num_blocks=50):  # 极深的网络
        super(DeepResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 创建多个残差块
        self.layers = self._make_layer(64, num_blocks)
        
        self.linear = nn.Linear(64, 10)
    
    def _make_layer(self, channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(self.in_channels, channels))
            self.in_channels = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 深度网络的初始化实验
def deep_network_initialization_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    init_methods = {
        'kaiming_normal': lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') if hasattr(m, 'weight') else None,
        'xavier_normal': lambda m: nn.init.xavier_normal_(m.weight) if hasattr(m, 'weight') else None,
        'orthogonal': lambda m: nn.init.orthogonal_(m.weight) if hasattr(m, 'weight') else None,
    }
    
    results = {}
    
    for name, init_fn in init_methods.items():
        print(f"\nTraining deep network with {name} initialization...")
        
        try:
            # 创建超深网络
            model = DeepResNet(num_blocks=50)  # 50个残差块
            
            # 应用初始化
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    init_fn(m)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            
            model.to(device)
            
            # 使用小批量进行前向和反向传播测试
            inputs = torch.randn(2, 3, 32, 32).to(device)
            targets = torch.LongTensor([0, 1]).to(device)
            
            # 设置优化器和损失函数
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 检查梯度
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append((name, grad_norm))
            
            # 分析结果
            results[name] = {
                'success': True,
                'initial_loss': loss.item(),
                'grad_norms': grad_norms
            }
            
            print(f"Successfully initialized! Initial loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"Initialization failed with error: {e}")
            results[name] = {
                'success': False,
                'error': str(e)
            }
    
    # 分析梯度范数
    plt.figure(figsize=(10, 6))
    
    for name, result in results.items():
        if result['success']:
            # 提取梯度范数
            layer_indices = range(len(result['grad_norms']))
            norms = [norm for _, norm in result['grad_norms']]
            
            plt.semilogy(layer_indices, norms, label=name)
    
    plt.xlabel('Layer Index')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title('Gradient Flow in Deep Network with Different Initializations')
    plt.legend()
    plt.grid(True)
    plt.savefig('deep_network_gradients.png')
    plt.show()
    
    return results

# 运行深度网络实验
deep_results = deep_network_initialization_experiment()
```

### 6.3 分析初始化对权重和激活分布的影响

观察不同初始化如何影响网络中的权重和激活值分布：

```python
def analyze_weight_activation_distributions(init_methods=None):
    if init_methods is None:
        init_methods = {
            'uniform_std0.01': lambda m: nn.init.uniform_(m.weight, -0.01, 0.01) if hasattr(m, 'weight') else None,
            'xavier_uniform': lambda m: nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None,
            'kaiming_normal': lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu') if hasattr(m, 'weight') else None,
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成一些随机输入数据
    inputs = torch.randn(100, 3, 32, 32).to(device)
    
    results = {}
    
    for name, init_fn in init_methods.items():
        print(f"Analyzing {name} initialization...")
        
        # 创建模型
        model = CNN()
        
        # 应用初始化
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_fn(m)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model.to(device)
        model.eval()
        
        # 收集权重
        weights = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
                weights[name] = module.weight.data.cpu().flatten().numpy()
        
        # 收集激活值
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        # 注册钩子来获取激活值
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # 前向传播
        with torch.no_grad():
            _ = model(inputs)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 保存结果
        results[name] = {
            'weights': weights,
            'activations': activations
        }
    
    # 可视化权重分布
    plt.figure(figsize=(15, 10))
    
    # 为每种初始化方法选择不同的颜色
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # 权重分布
    layer_names = list(results[list(results.keys())[0]]['weights'].keys())
    num_layers = min(4, len(layer_names))  # 最多显示4层
    
    for i, layer_name in enumerate(layer_names[:num_layers]):
        plt.subplot(2, num_layers, i+1)
        
        for j, (method_name, result) in enumerate(results.items()):
            weights = result['weights'][layer_name]
            plt.hist(weights, bins=50, alpha=0.5, label=method_name, color=colors[j%len(colors)])
            
            # 添加统计信息
            mean = np.mean(weights)
            std = np.std(weights)
            plt.axvline(mean, color=colors[j%len(colors)], linestyle='dashed', alpha=0.5)
            plt.text(0.95, 0.95-j*0.1, f'{method_name}: μ={mean:.4f}, σ={std:.4f}',
                    transform=plt.gca().transAxes, ha='right', va='top', fontsize=8)
        
        plt.title(f'Weight Distribution - {layer_name}')
        if i == 0:
            plt.legend()
    
    # 激活值分布
    activation_layers = [name for name in list(results[list(results.keys())[0]]['activations'].keys())
                        if 'relu' in name.lower()][:num_layers]
    
    for i, layer_name in enumerate(activation_layers):
        plt.subplot(2, num_layers, i+num_layers+1)
        
        for j, (method_name, result) in enumerate(results.items()):
            if layer_name in result['activations']:
                activations = result['activations'][layer_name].flatten()
                plt.hist(activations, bins=50, alpha=0.5, label=method_name, color=colors[j%len(colors)])
                
                # 添加统计信息
                mean = np.mean(activations)
                std = np.std(activations)
                plt.text(0.95, 0.95-j*0.1, f'{method_name}: μ={mean:.4f}, σ={std:.4f}',
                        transform=plt.gca().transAxes, ha='right', va='top', fontsize=8)
        
        plt.title(f'Activation Distribution - {layer_name}')
    
    plt.tight_layout()
    plt.savefig('def analyze_weight_activation_distributions(init_methods=None):
    if init_methods is None:
        init_methods = {
            'uniform_std0.01': lambda m: nn.init.uniform_(m.weight, -0.01, 0.01) if hasattr(m, 'weight') else None,
            'xavier_uniform': lambda m: nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None,
            'kaiming_normal': lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu') if hasattr(m, 'weight') else None,
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成一些随机输入数据
    inputs = torch.randn(100, 3, 32, 32).to(device)
    
    results = {}
    
    for name, init_fn in init_methods.items():
        print(f"Analyzing {name} initialization...")
        
        # 创建模型
        model = CNN()
        
        # 应用初始化
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_fn(m)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model.to(device)
        model.eval()
        
        # 收集权重
        weights = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
                weights[name] = module.weight.data.cpu().flatten().numpy()
        
        # 收集激活值
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        # 注册钩子来获取激活值
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # 前向传播
        with torch.no_grad():
            _ = model(inputs)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 保存结果
        results[name] = {
            'weights': weights,
            'activations': activations
        }
    
    # 可视化权重分布
    plt.figure(figsize=(15, 10))
    
    # 为每种初始化方法选择不同的颜色
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # 权重分布
    layer_names = list(results[list(results.keys())[0]]['weights'].keys())
    num_layers = min(4, len(layer_names))  # 最多显示4层
    
    for i, layer_name in enumerate(layer_names[:num_layers]):
        plt.subplot(2, num_layers, i+1)
        
        for j, (method_name, result) in enumerate(results.items()):
            weights = result['weights'][layer_name]
            plt.hist(weights, bins=50, alpha=0.5, label=method_name, color=colors[j%len(colors)])
            
            # 添加统计信息
            mean = np.mean(weights)
            std = np.std(weights)
            plt.axvline(mean, color=colors[j%len(colors)], linestyle='dashed', alpha=0.5)
            plt.text(0.95, 0.95-j*0.1, f'{method_name}: μ={mean:.4f}, σ={std:.4f}',
                    transform=plt.gca().transAxes, ha='right', va='top', fontsize=8)
        
        plt.title(f'Weight Distribution - {layer_name}')
        if i == 0:
            plt.legend()
    
    # 激活值分布
    activation_layers = [name for name in list(results[list(results.keys())[0]]['activations'].keys())
                        if 'relu' in name.lower()][:num_layers]
    
    for i, layer_name in enumerate(activation_layers):
        plt.subplot(2, num_layers, i+num_layers+1)
        
        for j, (method_name, result) in enumerate(results.items()):
            if layer_name in result['activations']:
                activations = result['activations'][layer_name].flatten()
                plt.hist(activations, bins=50, alpha=0.5, label=method_name, color=colors[j%len(colors)])
                
                # 添加统计信息
                mean = np.mean(activations)
                std = np.std(activations)
                plt.text(0.95, 0.95-j*0.1, f'{method_name}: μ={mean:.4f}, σ={std:.4f}',
                        transform=plt.gca().transAxes, ha='right', va='top', fontsize=8)
        
        plt.title(f'Activation Distribution - {layer_name}')
    
    plt.tight_layout()
    plt.savefig('
