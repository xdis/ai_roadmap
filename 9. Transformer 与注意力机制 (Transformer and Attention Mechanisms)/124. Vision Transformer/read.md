# Vision Transformer (ViT)：从零开始掌握

## 1. 基础概念理解

### 什么是Vision Transformer？

Vision Transformer (ViT) 是将原本为自然语言处理设计的Transformer架构应用于计算机视觉任务的创新模型。2020年由Google研究团队在论文《An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale》中提出。

### ViT的核心思想

ViT的核心思想非常直接：**将图像视为一系列的"词元(token)"序列**，类似于NLP中处理文本的方式。这种方法与传统的卷积神经网络(CNN)形成鲜明对比：

| 特性 | Vision Transformer | 卷积神经网络 |
|------|-------------------|------------|
| 基本单元 | 自注意力机制 | 卷积核 |
| 感受野 | 全局（可见所有图像块） | 局部（逐层扩展） |
| 位置信息 | 显式位置编码 | 隐式学习 |
| 参数共享 | 相对较少 | 大量参数共享 |
| 归纳偏置 | 较少 | 平移不变性等先验 |

### ViT的基本工作流程

Vision Transformer处理图像的步骤如下：

1. **图像分块**：将输入图像分割成固定大小的块(patches)
2. **线性映射**：将每个图像块映射到一个嵌入向量
3. **位置编码**：添加位置编码以保留空间信息
4. **添加分类标记**：添加特殊的[CLS]标记用于分类
5. **Transformer编码器**：通过多层Transformer编码器处理序列
6. **分类头**：基于[CLS]标记的表示进行预测

```
                     Transformer编码器
                     ↗  ↑  ↑  ↑  ↑  ↖
                   [CLS][P₁][P₂][P₃]...[Pₙ]
                     ↑   ↑   ↑   ↑     ↑
                     +   +   +   +     +  ← 位置编码
                     ↑   ↑   ↑   ↑     ↑
                   [CLS] ┌─┐ ┌─┐ ┌─┐   ┌─┐
                         │ │ │ │ │ │   │ │
                         └─┘ └─┘ └─┘   └─┘ ← 图像块
```

### ViT的优势与局限

**优势：**
- 全局感受野，可捕获远距离依赖关系
- 计算效率高（并行计算）
- 在大规模数据集上表现优异
- 结构简洁，易于扩展

**局限：**
- 需要大量数据训练
- 缺乏CNN固有的归纳偏置（如平移不变性）
- 计算开销大（尤其是对于高分辨率图像）

## 2. 技术细节探索

### 图像分块与嵌入

ViT的第一步是将输入图像分成固定大小的块。标准ViT将224×224像素的图像分成16×16的块，产生196个patches (14×14)。

从代码来看：
```python
class PatchEmbed(nn.Module):
    """将图像分割成块并嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # 将图像分成块并投影到嵌入维度
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        return x
```

**技术要点：**
- 使用卷积层(`kernel_size=patch_size, stride=patch_size`)巧妙地同时完成分块和线性投影
- 对于每个16×16×3的块，投影到embed_dim(通常是768)维度的嵌入向量
- `flatten(2)`将空间维度展平，`transpose(1, 2)`调整维度顺序为(batch, num_patches, embed_dim)

### 位置编码

由于Transformer缺乏CNN那样的空间感知能力，ViT需要显式添加位置信息：

```python
# 位置编码通常是一个可学习的嵌入
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
```

位置编码直接加到图像块嵌入上，使模型能区分不同位置的块。

### 多头自注意力机制

自注意力是Transformer的核心，允许模型关注图像的不同区域：

```python
class Attention(nn.Module):
    """多头自注意力"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
```

**注意力计算流程：**

1. 线性投影生成查询(Q)、键(K)、值(V)三个矩阵
2. 计算注意力权重：`attn = (q @ k.transpose(-2, -1)) * scale`
3. 应用softmax归一化：`attn = attn.softmax(dim=-1)`
4. 计算加权和：`x = (attn @ v).transpose(1, 2)`
5. 将多头结果连接并投影回原始维度

### ViT的Transformer编码器

Transformer编码器由多个相同的层组成，每层包含：

1. **多头自注意力**
2. **层归一化**
3. **MLP块**（通常是两层全连接网络）
4. **残差连接**

```python
class Block(nn.Module):
    """Transformer编码器块"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### 分类头及预训练

标准ViT使用特殊的[CLS]标记进行图像分类：

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
# ...
cls_token = self.cls_token.expand(B, -1, -1)
x = torch.cat((cls_token, x), dim=1)
# ...
x = self.norm(x)
return self.head(x[:, 0])  # 取[CLS]标记
```

ViT通常在大型数据集上预训练（如ImageNet-21k），然后在特定任务上微调。

## 3. 实践与实现

### 完整的ViT实现

下面是一个完整的PyTorch版ViT实现：

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """将图像分割成块并嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # 将图像分成块并投影到嵌入维度
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        return x

class Attention(nn.Module):
    """多头自注意力"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # 每个形状: B, num_heads, N, C//num_heads
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 计算加权和并重塑
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Transformer编码器块"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer模型"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        
        # 图像块嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 可学习的分类标记和位置嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer编码器
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=drop_rate, attn_drop=attn_drop_rate) 
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # B, num_patches, embed_dim
        
        # 添加分类标记
        cls_token = self.cls_token.expand(B, -1, -1)  # B, 1, embed_dim
        x = torch.cat((cls_token, x), dim=1)  # B, 1+num_patches, embed_dim
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 通过Transformer编码器
        x = self.blocks(x)
        x = self.norm(x)
        
        return x[:, 0]  # 返回[CLS]标记的表示
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
```

### 使用预训练ViT模型

使用torchvision或huggingface加载预训练模型：

```python
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

# 加载预训练模型
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.eval()

# 预处理
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图像
from PIL import Image
img = Image.open("example.jpg")
input_tensor = preprocess(img).unsqueeze(0)  # 添加batch维度

# 进行推理
with torch.no_grad():
    output = model(input_tensor)
    
# 获取预测类别
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)
```

### ViT训练技巧

ViT的训练有几个关键点：

1. **数据增强**：关键是应用RandAugment、Mixup和CutMix等增强技术
2. **正则化**：使用dropout、stochastic depth和weight decay
3. **学习率调度**：使用cosine decay和warm-up
4. **梯度裁剪**：避免训练中梯度爆炸

```python
# 数据增强示例
from torchvision import transforms
import random

class RandomCutmix:
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, batch):
        if random.random() > self.prob:
            return batch
            
        # Cutmix实现...
        return mixed_batch

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.05, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## 4. 高级应用与变体

### Swin Transformer

Swin Transformer通过引入层次结构和局部窗口注意力机制解决了ViT的局限性：

```
└── 特点
    ├── 移位窗口注意力机制
    ├── 层次化特征图
    └── 线性计算复杂度
```

Swin Transformer的主要创新是**移位窗口(Shifted Window)**机制，允许跨窗口信息交流。

### DeiT (Data-efficient Image Transformer)

DeiT通过改进训练方法，解决了ViT对大规模数据的依赖问题：

```
└── 关键创新
    ├── 教师-学生知识蒸馏
    ├── 额外的蒸馏token
    └── 强化的数据增强
```

### MobileViT

为移动设备设计的轻量级Vision Transformer：

```
└── 设计思路
    ├── CNN与Transformer结合
    ├── 局部与全局特征融合
    └── 减少计算开销
```

### MAE (Masked Autoencoders)

一种自监督预训练方法：

```python
# MAE主要思想
# 1. 随机遮蔽大部分图像块(如75%)
# 2. 仅对可见块应用编码器
# 3. 添加遮蔽标记并用解码器重建原图

class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder  # ViT编码器
        self.mask_ratio = mask_ratio
        
        # 简化的解码器
        self.decoder = nn.TransformerDecoder(...)
        
    def forward(self, x):
        # 1. 创建遮蔽
        patches = self.patchify(x)
        mask = self.random_masking(patches.shape[0], self.mask_ratio)
        
        # 2. 编码可见块
        visible_patches = patches[~mask]
        encoded = self.encoder(visible_patches)
        
        # 3. 解码并重建
        full_tokens = self.generate_full_tokens(encoded, mask)
        reconstructed = self.decoder(full_tokens)
        
        # 4. 计算重建损失
        loss = self.reconstruction_loss(reconstructed, patches, mask)
        return loss
```

### ViT在下游任务中的应用

Vision Transformer不仅限于图像分类，还被应用于：

1. **目标检测**：DETR、Deformable DETR
2. **图像分割**：SETR、Segmenter
3. **视频理解**：ViViT、TimeSformer
4. **多模态学习**：CLIP、DALL-E

### 最新研究趋势

1. **效率优化**：减少计算复杂度(如线性注意力)
2. **架构改进**：适应不同任务的特定结构
3. **自监督学习**：减少对标注数据的依赖
4. **多模态融合**：结合视觉与语言理解
5. **动态推理**：按需调整计算资源分配

## 实践案例：CIFAR-10图像分类

下面是一个在CIFAR-10上训练简化版ViT的完整示例：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义ViT模型(简化版，小图像)
class SimpleViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, 
                 embed_dim=192, depth=6, num_heads=3, num_classes=10):
        super().__init__()
        # 图像分块嵌入
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 分类标记和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer编码器
        self.blocks = nn.Sequential(*[Block(
            dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        # 图像分块并嵌入
        x = self.patch_embed(x)
        
        # 添加分类标记和位置编码
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        # Transformer编码
        x = self.blocks(x)
        x = self.norm(x)
        
        # 分类预测
        x = self.head(x[:, 0])
        return x

# 数据加载与预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128,
                         shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128,
                       shuffle=False, num_workers=2)

# 创建模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleViT().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc

# 测试函数
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc

# 训练循环
epochs = 100
best_acc = 0

for epoch in range(epochs):
    train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
    test_loss, test_acc = test(model, testloader, criterion, device)
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_vit_cifar10.pth')
```

## 总结

Vision Transformer代表了计算机视觉领域的范式转变，将Transformer从NLP领域成功迁移到视觉任务中。尽管ViT需要更多的数据和计算资源，但它的灵活性、可扩展性和强大的表示能力使其成为当前计算机视觉研究的热点。

从基础的图像分类到复杂的目标检测、图像分割和多模态学习，ViT及其变体正在改变计算机视觉的格局。随着效率优化和架构创新的持续进行，Vision Transformer将在未来继续发挥重要作用。

您现在已经从基础概念理解到技术细节、实践实现和高级应用，全面掌握了Vision Transformer这一核心技术。无论是在研究还是工程应用中，这些知识都将帮助您更好地理解和利用ViT模型。

Similar code found with 4 license types