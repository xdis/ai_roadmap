# CNN 架构(LeNet, AlexNet)

## 基础概念理解

### CNN架构的演进
- 卷积神经网络(CNN)架构随着时间不断发展，从简单到复杂
- 早期架构(如LeNet)奠定了基础结构，创新了关键组件
- 里程碑架构(如AlexNet)引发了深度学习革命，推动了CV领域发展
- 每代架构都引入新的设计思想，解决前代架构的限制和问题

### 架构设计的基本考量
- **深度与宽度**：层数和每层神经元数量的平衡
- **计算效率**：参数数量与表达能力的权衡
- **梯度流动**：确保训练过程中梯度能够有效传播
- **特征表示**：学习从低级到高级的层次化表示
- **泛化能力**：减少过拟合，提高模型在新数据上的表现

### 评估CNN架构的指标
- **准确率**：在标准基准数据集上的分类性能
- **参数数量**：模型复杂度和存储需求
- **计算量**：FLOPs(浮点运算次数)和推理速度
- **内存占用**：训练和推理时的内存需求
- **收敛速度**：达到目标性能所需的训练时间

## LeNet架构详解

### LeNet-5概述
- 由Yann LeCun于1998年提出，专为手写数字识别设计
- 是第一个成功应用于商业产品的卷积神经网络
- 用于美国邮政编码识别系统
- 奠定了现代CNN的基本结构：卷积层+池化层+全连接层

### LeNet-5架构细节
- **输入**：32×32灰度图像
- **网络深度**：7层(不计输入层)
- **卷积层**：使用5×5卷积核
- **下采样层**：使用2×2池化(原始实现使用非线性平均值)
- **激活函数**：tanh或sigmoid(现代版本通常使用ReLU)
- **全连接层**：实现最终分类
- **总参数量**：约60,000

### LeNet-5层级结构
1. **输入层**：32×32灰度图像
2. **C1卷积层**：6个5×5卷积核，输出6@28×28特征图
3. **S2池化层**：2×2平均池化，输出6@14×14特征图
4. **C3卷积层**：16个5×5卷积核，输出16@10×10特征图
5. **S4池化层**：2×2平均池化，输出16@5×5特征图
6. **C5卷积层**：120个5×5卷积核，输出120维向量
7. **F6全连接层**：84个神经元
8. **输出层**：10个神经元(对应10个数字)

### LeNet的创新点
- **局部感受野**：卷积操作捕获局部空间模式
- **权重共享**：减少参数量，提高泛化能力
- **下采样**：提高空间不变性，减少计算量
- **层次化表示**：从简单到复杂的特征学习
- **端到端训练**：通过反向传播优化整个网络

### LeNet的局限性
- 较浅的网络深度限制了表示能力
- 使用tanh/sigmoid激活函数易导致梯度消失
- 训练数据和计算资源有限
- 没有使用正则化技术或归一化层
- 当时硬件限制了网络规模的拓展

### LeNet的Python实现
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1: 卷积层，6个5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)
        # S2: 池化层
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 卷积层，16个5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        # S4: 池化层
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C5: 卷积层，120个5x5卷积核
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, padding=0)
        # F6: 全连接层
        self.fc1 = nn.Linear(120, 84)
        # 输出层
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # 确保输入尺寸为32x32
        if x.shape[2] != 32 or x.shape[3] != 32:
            x = F.interpolate(x, size=(32, 32))
            
        # 卷积层C1 + 激活
        x = F.tanh(self.conv1(x))  # 输出: 6@28x28
        
        # 池化层S2
        x = self.pool1(x)  # 输出: 6@14x14
        
        # 卷积层C3 + 激活
        x = F.tanh(self.conv2(x))  # 输出: 16@10x10
        
        # 池化层S4
        x = self.pool2(x)  # 输出: 16@5x5
        
        # 卷积层C5 + 激活
        x = F.tanh(self.conv3(x))  # 输出: 120@1x1
        
        # 展平
        x = x.view(-1, 120)
        
        # 全连接层F6 + 激活
        x = F.tanh(self.fc1(x))
        
        # 输出层
        x = self.fc2(x)
        
        return x

# 创建LeNet-5模型
model = LeNet5()
```

## AlexNet架构详解

### AlexNet概述
- 由Alex Krizhevsky、Ilya Sutskever和Geoffrey Hinton于2012年提出
- 在ImageNet大规模视觉识别挑战赛(ILSVRC)中取得突破性成功
- 大幅降低了错误率(从26.2%到15.3%)，标志着深度学习在计算机视觉领域的崛起
- 比LeNet更深更宽，能够处理更复杂的数据和任务

### AlexNet架构细节
- **输入**：224×224×3彩色图像
- **网络深度**：8层(5个卷积层+3个全连接层)
- **卷积核大小**：11×11, 5×5, 3×3
- **池化**：最大池化(Max Pooling)
- **激活函数**：ReLU(突破性应用，解决梯度消失问题)
- **正则化**：Dropout和数据增强
- **总参数量**：约6000万(比LeNet多约1000倍)

### AlexNet层级结构
1. **输入层**：224×224×3彩色图像
2. **卷积层1**：96个11×11卷积核，步长4，ReLU激活
3. **最大池化1**：3×3池化核，步长2
4. **卷积层2**：256个5×5卷积核，ReLU激活
5. **最大池化2**：3×3池化核，步长2
6. **卷积层3**：384个3×3卷积核，ReLU激活
7. **卷积层4**：384个3×3卷积核，ReLU激活
8. **卷积层5**：256个3×3卷积核，ReLU激活
9. **最大池化3**：3×3池化核，步长2
10. **全连接层1**：4096个神经元，ReLU激活，Dropout
11. **全连接层2**：4096个神经元，ReLU激活，Dropout
12. **输出层**：1000个神经元(对应ImageNet的1000个类别)

### AlexNet的创新点
- **ReLU激活函数**：加速收敛，解决深层网络中的梯度消失
- **多GPU训练**：原始版本在两块GTX 580 GPU上分布训练
- **局部响应归一化(LRN)**：增强特征图中的高激活值
- **Dropout**：防止过拟合，提高模型泛化能力
- **数据增强**：随机裁剪、水平翻转等，增加训练样本多样性
- **更大的模型规模**：更深、更宽的网络结构

### AlexNet的影响
- 重新点燃了深度学习研究的热情
- 验证了深度卷积网络在视觉任务上的强大能力
- 促进了GPU在深度学习中的广泛应用
- 引入了多种现在仍在使用的训练技术
- 为后续VGG、GoogLeNet等架构奠定了基础

### AlexNet与LeNet的对比
| 特性 | LeNet-5 | AlexNet |
|-----|---------|---------|
| 年份 | 1998 | 2012 |
| 层数 | 7层 | 8层 |
| 输入 | 32×32×1 | 224×224×3 |
| 参数数量 | ~60K | ~60M |
| 激活函数 | tanh/sigmoid | ReLU |
| 正则化 | 无 | Dropout, LRN |
| 池化方式 | 平均池化 | 最大池化 |
| 应用领域 | 手写数字识别 | 自然图像分类 |
| 训练硬件 | CPU | GPU |

### AlexNet的Python实现
```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            # 卷积层1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # LRN层 (现在很少使用)
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # 池化层1
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 卷积层2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # 池化层2
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 卷积层3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 卷积层4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 卷积层5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 池化层3
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            # 展平
            nn.Dropout(),
            # 全连接层1
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(),
            # 全连接层2
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            # 输出层
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平
        x = x.view(x.size(0), 256 * 6 * 6)
        # 分类
        x = self.classifier(x)
        return x

# 创建AlexNet模型
model = AlexNet()
```

## 实践与实现

### 在现代框架中使用预训练模型

#### PyTorch
```python
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# 加载预训练的AlexNet
alexnet = models.alexnet(pretrained=True)
alexnet.eval()  # 设置为评估模式

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载并预处理图像
img = Image.open('sample.jpg')
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# 使用模型预测
with torch.no_grad():
    output = alexnet(batch_t)

# 获取预测结果
_, predicted_idx = torch.max(output, 1)
```

#### TensorFlow/Keras
```python
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# 加载预训练的VGG16 (AlexNet在TF中没有直接提供)
model = VGG16(weights='imagenet', include_top=True)

# 加载并预处理图像
img_path = 'sample.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 预测
preds = model.predict(x)
```

### 训练自己的LeNet/AlexNet

#### PyTorch上训练LeNet
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 数据准备
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('训练完成')

# 测试
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'准确率: {100 * correct / total}%')
```

## 高级应用与变体

### LeNet变种
- **LeNet-4**: 更简化的LeNet版本，有4个参数层
- **Boosted LeNet-4**: 使用多个LeNet-4网络组合，提高性能
- **现代LeNet**: 使用ReLU激活函数、Dropout和批归一化的改进版本
- **LeNet-5近似版本**: 使用最大池化替代平均池化的常见实现

### AlexNet变种
- **CaffeNet**: Caffe框架中的AlexNet实现，层顺序略有不同
- **One-GPU AlexNet**: 单GPU版本的AlexNet，把特征图通道放在一起
- **OverFeat**: 基于AlexNet的改进版本，用于物体检测和定位
- **ZFNet**: ILSVRC 2013冠军，是AlexNet的调优版本，使用更小的卷积核和步长

### 使用迁移学习扩展应用范围
- **特征提取**：冻结预训练网络，只训练新添加的分类层
- **微调(Fine-tuning)**：调整预训练网络的后几层，适应新任务
- **逐层微调**：从输出层向输入层逐步解冻和微调

```python
# PyTorch 迁移学习示例
import torch
import torchvision.models as models
import torch.nn as nn

# 加载预训练的AlexNet
model = models.alexnet(pretrained=True)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False
    
# 替换最后的分类器，适应新任务
model.classifier[6] = nn.Linear(4096, num_new_classes)

# 只训练新添加的层
optimizer = torch.optim.SGD(model.classifier[6].parameters(), lr=0.001, momentum=0.9)
```

## 理解架构设计选择

### 网络设计中的关键权衡
- **模型复杂度 vs. 准确率**：更多参数通常能提高性能，但增加过拟合风险
- **计算效率 vs. 表示能力**：较小的层可能更快但表达能力有限
- **卷积核大小选择**：大卷积核感受野更大，小卷积核参数更少
- **层数选择**：更深的网络可学习更复杂特征，但更难训练

### 设计策略演进
- **LeNet时代**：手工设计网络结构，基于领域知识
- **AlexNet时代**：增加网络规模，使用GPU克服训练障碍
- **现代架构**：更科学的设计原则，如残差连接、分组卷积等
- **神经架构搜索(NAS)**：使用自动化方法寻找最佳架构

### 深入理解卷积核大小选择
- **LeNet的5×5卷积核**：适合捕获数字的主要特征
- **AlexNet的11×11, 5×5, 3×3卷积核**：
  - 11×11在第一层用于捕获大尺度结构
  - 后续层使用较小卷积核，增加深度而非宽度

## 实际应用场景

### LeNet应用
- **文档分析与OCR**：手写/打印文字识别
- **签名验证**：银行和安全系统中的签名认证
- **简单物体分类**：小规模图像分类任务
- **教学和入门**：深度学习初学者的典型第一个项目

### AlexNet应用
- **大规模图像分类**：ImageNet等大型数据集上的图像识别
- **迁移学习基础模型**：为其他计算机视觉任务提取特征
- **物体检测和定位的基础**：许多目标检测系统使用修改版AlexNet
- **医学图像分析**：X光、MRI等医学影像的疾病诊断

### 局限性与何时不使用
- **移动和嵌入式设备**：参数量大，计算需求高，不适合资源受限设备
- **实时应用**：在计算资源有限的环境下，推理速度可能不足
- **小数据集**：对于小数据集，这些模型容易过拟合
- **现代任务**：对于最先进的性能要求，应考虑更现代的架构

## 学习资源

### 论文
- **LeNet**: "Gradient-Based Learning Applied to Document Recognition" by Yann LeCun et al. (1998)
- **AlexNet**: "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al. (2012)

### 书籍
- 《深度学习》by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- 《动手学深度学习》by 李沐等

### 在线课程和教程
- Stanford CS231n: 卷积神经网络视觉识别
- Coursera深度学习专项课程(Andrew Ng)
- PyTorch和TensorFlow官方教程

### 代码实现
- LeNet和AlexNet的官方及社区实现
- 框架中的预训练模型
- 各种改进和优化版本的GitHub仓库

## 下一步学习

学习完LeNet和AlexNet后，可以进一步探索：

1. **更复杂的现代CNN架构**：VGG、GoogLeNet/Inception、ResNet等
2. **理解架构设计的原则与科学**：1×1卷积、残差连接等创新
3. **深入理解卷积神经网络的理论基础**：为什么CNN对视觉任务如此有效
4. **优化技术**：批归一化、学习率调度等提高训练效果的方法
5. **应用CNN到实际问题**：图像分类、物体检测、分割等任务的实践