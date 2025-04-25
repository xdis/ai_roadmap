# PyTorch 基础教程

PyTorch 是一个开源的深度学习框架，由 Facebook 的 AI 研究团队开发。它以其动态计算图和 Python 优先的设计理念而受到广泛欢迎。相比于其他框架，PyTorch 的语法更直观，更接近 Python 的编程风格，使得开发和调试模型变得更加简单。

## 1. PyTorch 核心概念

### 1.1 张量 (Tensor)

张量是 PyTorch 中的基本数据结构，类似于 NumPy 的数组，但可以在 GPU 上运行以加速计算。

```python
import torch

# 创建张量
x = torch.tensor([1, 2, 3, 4])
print(x)  # tensor([1, 2, 3, 4])

# 创建特定形状的张量
zeros = torch.zeros(2, 3)  # 2x3 的全 0 张量
print(zeros)
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

# 创建随机张量
random_tensor = torch.rand(2, 2)
print(random_tensor)
# 输出类似: tensor([[0.1234, 0.5678],
#                  [0.9012, 0.3456]])

# 张量的形状
print(zeros.shape)  # torch.Size([2, 3])

# 张量的数据类型
print(x.dtype)  # torch.int64

# 张量设备
print(x.device)  # cpu (或 cuda:0 如果在 GPU 上)
```

### 1.2 张量操作

```python
# 基本数学运算
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)  # tensor([5, 7, 9])
print(a * b)  # tensor([4, 10, 18]) (元素级乘法)
print(torch.matmul(a, b))  # tensor(32) (点积)

# 维度变换
c = torch.rand(2, 3)
print(c.shape)  # torch.Size([2, 3])
print(c.T.shape)  # torch.Size([3, 2]) (转置)

# 重塑张量
d = torch.arange(6)  # tensor([0, 1, 2, 3, 4, 5])
print(d.reshape(2, 3))
# tensor([[0, 1, 2],
#         [3, 4, 5]])

# 切片和索引
print(d[2:5])  # tensor([2, 3, 4])
```

## 2. 自动微分 (Autograd)

PyTorch 的核心功能之一是自动微分系统，它能自动计算梯度，使得构建和训练神经网络变得简单。

```python
x = torch.tensor(2.0, requires_grad=True)  # 标记需要计算梯度
y = x**2 + 2*x + 1

# 计算 y 关于 x 的梯度
y.backward()

# 访问 x 的梯度
print(x.grad)  # tensor(6.) (dy/dx = 2x + 2 = 2*2 + 2 = 6)
```

## 3. 构建神经网络

PyTorch 提供了 `nn` 模块来构建神经网络。

### 3.1 简单的全连接网络

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # 10 输入特征，20 隐藏单元
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)   # 20 隐藏单元，1 输出
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleNN()
print(model)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 假设的训练数据
x = torch.randn(5, 10)  # 5 个样本，每个有 10 个特征
y = torch.randn(5, 1)   # 5 个样本的目标值

# 训练循环
for epoch in range(100):
    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
```

## 4. 计算机视觉示例：图像分类

使用 PyTorch 进行图像分类的简单例子：

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=True, 
                                          transform=transform, 
                                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                          batch_size=64, 
                                          shuffle=True)

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型（简化版）
def train(epochs=3):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
```

## 5. PyTorch 的优势

1. **动态计算图**：PyTorch 使用动态计算图，这意味着图是在运行时定义的，而不是预先定义的。这使得调试更容易，代码更直观。

2. **Python 集成**：PyTorch 与 Python 生态系统紧密集成，可以使用标准的 Python 调试工具。

3. **社区支持**：拥有庞大的社区和丰富的文档。

4. **灵活性**：更容易编写自定义层、损失函数和训练循环。

## 6. 模型保存与加载

```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 设置为评估模式
```

## 7. GPU 加速

PyTorch 可以轻松地将计算从 CPU 移动到 GPU 以加速训练。

```python
# 检查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 将模型移到 GPU
model = CNN().to(device)

# 将张量移到 GPU
x = torch.randn(5, 10).to(device)
```

## 总结

PyTorch 是一个强大而灵活的深度学习框架，它的语法直观，接近 Python 的编程风格。通过本教程，你已经了解了 PyTorch 的基本概念，包括张量、自动微分、神经网络构建以及模型的训练和评估。随着你的深入学习，你可以探索更高级的主题，如迁移学习、自定义数据集、复杂架构和分布式训练。