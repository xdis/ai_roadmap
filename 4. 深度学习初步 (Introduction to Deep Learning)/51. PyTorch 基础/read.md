# PyTorch 基础

## 1. PyTorch 简介

PyTorch 是一个开源的深度学习框架，由 Facebook 的 AI Research 团队开发，它提供了强大的计算能力和灵活的编程接口，已成为学术研究和工业应用中最受欢迎的深度学习框架之一。

### 1.1 PyTorch 的主要特点

1. **动态计算图**：PyTorch 使用动态计算图，允许在运行时构建和修改网络结构，提供更直观的调试体验
2. **Python 优先**：与 Python 生态系统深度集成，编程风格自然，学习曲线平缓
3. **强大的 GPU 加速**：自动利用 NVIDIA GPU 进行高效计算
4. **丰富的工具和库生态系统**：提供从数据加载到模型部署的完整工具链
5. **命令式编程风格**：类似于 NumPy 的编程体验，但支持 GPU 加速和自动微分

### 1.2 PyTorch 与其他框架的比较

| 特性 | PyTorch | TensorFlow | JAX |
|------|---------|------------|-----|
| 计算图 | 动态 | 静态(1.x)/动态(2.x) | 变换的静态 |
| 易用性 | 高 | 中 | 中 |
| 调试 | 简单 | 复杂 | 中等 |
| 部署 | 较好 | 非常好 | 较好 |
| 社区支持 | 强大 | 强大 | 增长中 |
| 主要应用场景 | 研究、开发 | 研究、生产 | 研究 |

## 2. PyTorch 安装与环境设置

### 2.1 安装 PyTorch

PyTorch 可以通过多种方式安装，最常用的是通过 pip 或 conda：

**使用 pip 安装**：

```bash
# 安装 CPU 版本
pip install torch torchvision torchaudio

# 安装 CUDA 11.7 支持的 GPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

**使用 conda 安装**：

```bash
# 安装 CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 安装 GPU 版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 2.2 验证安装

安装完成后，可以通过以下代码验证安装是否成功：

```python
import torch

# 检查 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 检查 CUDA 是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
```

### 2.3 开发环境推荐

1. **Jupyter Notebook/Lab**：交互式开发和可视化
2. **Visual Studio Code + Python 扩展**：全功能 IDE 体验
3. **PyCharm Professional**：专业 Python IDE，内置深度学习支持
4. **Google Colab**：免费 GPU 资源，适合初学者和小项目

## 3. PyTorch 核心概念

### 3.1 张量 (Tensor)

张量是 PyTorch 的核心数据结构，类似于 NumPy 的多维数组，但可以在 GPU 上运行并自动计算梯度。

**创建张量**：

```python
import torch

# 从 Python 列表创建
x = torch.tensor([[1, 2], [3, 4]])

# 创建特定形状的张量
zeros = torch.zeros(2, 3)       # 全 0 张量
ones = torch.ones(2, 3)         # 全 1 张量
rand = torch.rand(2, 3)         # 均匀分布随机数张量
randn = torch.randn(2, 3)       # 标准正态分布随机数张量
arange = torch.arange(10)       # 序列张量
linspace = torch.linspace(0, 1, 5)  # 线性间隔张量

# 创建特定数据类型的张量
x_float = torch.tensor([1.0, 2.0], dtype=torch.float32)
x_long = torch.tensor([1, 2], dtype=torch.int64)
```

**张量属性和操作**：

```python
x = torch.randn(3, 4, 5)

# 张量属性
print(f"形状: {x.shape}")       # torch.Size([3, 4, 5])
print(f"维度: {x.dim()}")       # 3
print(f"数据类型: {x.dtype}")   # torch.float32
print(f"存储设备: {x.device}")  # cpu 或 cuda:0

# 索引和切片（类似 NumPy）
print(x[0])                     # 第一个元素
print(x[:, 0:2, :])             # 高级索引和切片

# 改变形状
y = x.view(3, 20)               # 改变形状为 3x20
y = x.reshape(3, 20)            # 类似 view，但某些情况下会复制数据
y = x.permute(2, 0, 1)          # 维度换位
y = x.transpose(0, 1)           # 交换两个维度
```

**张量数学运算**：

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# 基本运算
c = a + b                       # 加法
c = torch.add(a, b)             # 函数形式的加法
c = a - b                       # 减法
c = a * b                       # 逐元素乘法
c = a / b                       # 逐元素除法

# 矩阵运算
c = torch.mm(a, b.t())          # 矩阵乘法
c = a @ b.t()                   # 矩阵乘法（Python 3.5+ 语法）
c = torch.matmul(a, b.t())      # 广义矩阵乘法

# 统计操作
mean = torch.mean(a)            # 均值
sum = torch.sum(a)              # 求和
max_val, max_idx = torch.max(a, dim=1)  # 最大值及其索引
```

### 3.2 自动微分 (Autograd)

自动微分是 PyTorch 中计算梯度的核心功能，支持神经网络训练过程中的反向传播。

**基本用法**：

```python
# 创建需要梯度的张量
x = torch.randn(2, 2, requires_grad=True)
y = torch.randn(2, 2, requires_grad=True)

# 前向计算
z = x * 2 + y * y

# 计算梯度
z.backward(torch.ones_like(z))

# 查看梯度
print(x.grad)  # dz/dx = 2
print(y.grad)  # dz/dy = 2*y
```

**梯度累积与清零**：

```python
x = torch.randn(2, 2, requires_grad=True)

# 多次前向和反向传播会累积梯度
for _ in range(3):
    y = x * 2
    y.sum().backward()
    
print(x.grad)  # 梯度值为 6 (3次累积，每次为2)

# 清零梯度
x.grad.zero_()
```

**detach 和 no_grad**：

```python
# 分离张量，阻止梯度传播
x = torch.randn(2, 2, requires_grad=True)
y = x * 2
z = y.detach()  # z 不会追踪梯度

# 临时禁用梯度计算（推理阶段常用）
with torch.no_grad():
    y = x * 2  # 不会记录梯度
```

### 3.3 神经网络模块 (nn.Module)

PyTorch 提供了 `nn.Module` 类作为所有神经网络模块的基类，用于创建网络层和完整网络。

**创建一个简单的神经网络**：

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 定义前向传播
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化网络
net = SimpleNet(10, 50, 2)
print(net)  # 打印网络结构
```

**常用神经网络层**：

```python
# 全连接层
linear = nn.Linear(in_features=10, out_features=20)

# 卷积层
conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# 池化层
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

# 归一化层
batchnorm = nn.BatchNorm2d(num_features=16)
layernorm = nn.LayerNorm(normalized_shape=[16, 32, 32])

# 激活函数
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

# Dropout
dropout = nn.Dropout(p=0.5)
```

**顺序容器**：

```python
# 使用 Sequential 顺序构建网络
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),
    nn.Linear(20, 2)
)
```

### 3.4 优化器 (Optimizer)

优化器负责更新网络参数，PyTorch 提供了多种优化算法。

**常用优化器**：

```python
import torch.optim as optim

# 定义网络
model = SimpleNet(10, 50, 2)

# 随机梯度下降(SGD)优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# RMSprop 优化器
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

**优化步骤**：

```python
# 训练循环
for epoch in range(100):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    # 更新学习率（如果使用调度器）
    scheduler.step()
```

### 3.5 损失函数 (Loss Functions)

损失函数用于衡量模型输出与目标之间的差异。

**常用损失函数**：

```python
import torch.nn as nn

# 二分类交叉熵损失
bce_loss = nn.BCELoss()

# 分类交叉熵损失（带logits）
ce_loss = nn.CrossEntropyLoss()

# 均方误差损失（回归）
mse_loss = nn.MSELoss()

# L1损失（回归）
l1_loss = nn.L1Loss()

# 平滑L1损失（回归）
smooth_l1 = nn.SmoothL1Loss()
```

## 4. 数据处理与加载

### 4.1 Dataset 和 DataLoader

PyTorch 提供了 `Dataset` 和 `DataLoader` 类，帮助管理和批量加载数据。

**自定义数据集**：

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

# 创建示例数据
data = np.random.randn(100, 3, 32, 32)
targets = np.random.randint(0, 10, 100)

# 实例化数据集
dataset = CustomDataset(data, targets)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
```

**内置数据集**：

```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### 4.2 数据预处理和增强

**常用数据变换**：

```python
import torchvision.transforms as transforms

# 图像预处理变换
transform = transforms.Compose([
    # 几何变换
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    
    # 颜色变换
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    
    # 转换为张量
    transforms.ToTensor(),
    
    # 标准化
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

## 5. 模型训练与评估

### 5.1 基本训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 假设已经定义好模型、数据集和数据加载器
model = SimpleNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 将模型移至 GPU（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    
    for i, (inputs, targets) in enumerate(train_loader):
        # 将数据移至 GPU
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计损失
        running_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
```

### 5.2 模型评估

```python
model.eval()  # 设置为评估模式
correct = 0
total = 0

with torch.no_grad():  # 禁用梯度计算
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        # 统计准确率
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'测试准确率: {100 * correct / total:.2f}%')
```

### 5.3 保存和加载模型

```python
# 保存完整模型（包括架构和参数）
torch.save(model, 'model.pth')

# 仅保存模型参数（推荐方式）
torch.save(model.state_dict(), 'model_weights.pth')

# 加载完整模型
loaded_model = torch.load('model.pth')

# 加载模型参数（需要先定义模型架构）
model = SimpleNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # 设置为评估模式
```

## 6. 实践案例：MNIST 数字识别

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型、损失函数和优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
    
    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], 测试准确率: {100 * correct / total:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'mnist_cnn.pth')
```

## 7. PyTorch 进阶特性

### 7.1 分布式训练

```python
# 单机多卡训练
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 创建模型并移至 GPU
model = CNN().cuda()
# 包装为 DDP 模型
model = DDP(model, device_ids=[local_rank])

# 训练过程与普通训练类似，但使用 DistributedSampler
```

### 7.2 模型量化

```python
# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 静态量化 (需要校准数据)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# 使用校准数据运行模型
torch.quantization.convert(model, inplace=True)
```

### 7.3 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

# 创建梯度缩放器
scaler = GradScaler()

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 使用自动混合精度
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 缩放梯度并执行反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 7.4 模型剪枝和稀疏化

```python
import torch.nn.utils.prune as prune

# 剪枝单个层
prune.l1_unstructured(model.conv1, name='weight', amount=0.3)

# 全局剪枝
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)
```

## 8. 常见问题与调试技巧

### 8.1 内存管理

- **检查内存使用**：使用 `nvidia-smi` 监控 GPU 内存
- **减少内存使用**：
  - 使用较小的批量大小
  - 使用混合精度训练
  - 及时删除不需要的中间变量
  - 使用梯度累积处理大批量
  - 选择内存效率更高的模型架构

### 8.2 常见错误及解决方法

1. **CUDA out of memory**：
   - 减小批量大小
   - 检查并释放不必要的变量
   - 使用混合精度训练

2. **维度不匹配**：
   - 检查模型输入输出形状
   - 使用 `print(tensor.shape)` 调试
   - 理解广播规则

3. **梯度爆炸/消失**：
   - 使用梯度裁剪：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
   - 检查权重初始化
   - 使用批量归一化或层归一化

### 8.3 性能优化

1. **数据加载优化**：
   - 增加 `num_workers`
   - 使用 `pin_memory=True`
   - 预加载和缓存数据

2. **计算优化**：
   - 使用更高效的操作和算法
   - 避免不必要的CPU-GPU数据传输
   - 使用内置的融合操作（如 `torch.addmm`）

## 9. 资源与社区

### 9.1 官方资源

- [PyTorch 官方网站](https://pytorch.org/)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 示例](https://github.com/pytorch/examples)
- [PyTorch 教程](https://pytorch.org/tutorials/)

### 9.2 推荐学习资源

- [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch)
- [PyTorch 论坛](https://discuss.pytorch.org/)
- [Coursera 上的 PyTorch 课程](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
- [Fast.ai 深度学习课程](https://www.fast.ai/)

### 9.3 模型库与扩展

- [torchvision](https://pytorch.org/vision/stable/index.html)：用于计算机视觉
- [torchaudio](https://pytorch.org/audio/stable/index.html)：用于音频处理
- [torchtext](https://pytorch.org/text/stable/index.html)：用于自然语言处理
- [PyTorch Lightning](https://www.pytorchlightning.ai/)：高级 PyTorch 包装器
- [Hugging Face Transformers](https://huggingface.co/transformers/)：预训练模型库

## 10. 总结

- PyTorch 是一个灵活而强大的深度学习框架，提供了动态计算图和命令式编程风格
- 核心组件包括张量、自动微分、神经网络模块和优化器
- PyTorch 生态系统丰富，包括数据处理、模型部署和扩展工具
- 进阶特性包括分布式训练、量化、混合精度和模型剪枝
- 通过实践案例和调试技巧，可以高效地使用 PyTorch 开发深度学习应用

通过掌握 PyTorch 的基础知识和核心概念，您可以更轻松地实现和训练深度学习模型，进一步探索人工智能的各个领域。