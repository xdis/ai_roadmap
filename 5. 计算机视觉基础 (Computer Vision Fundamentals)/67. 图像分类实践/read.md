# 图像分类实践

## 基础概念理解

### 图像分类任务简介
- **定义**：将输入图像分配到预定义类别中的一个或多个类别的任务
- **输入**：图像（RGB或灰度）
- **输出**：类别标签或类别概率分布
- **应用**：物体识别、场景分类、医学诊断、工业质检等
- **挑战**：视角变化、光照条件、背景复杂度、类内变化、类间相似性

### 分类问题类型
- **二分类**：将图像分为两个类别（例如：猫 vs 狗）
- **多分类**：将图像分为多个互斥的类别（例如：CIFAR-10的10个类别）
- **多标签分类**：一张图像可以同时属于多个类别（例如：图像同时包含人和汽车）
- **层次分类**：类别之间存在层次关系（例如：动物→哺乳动物→猫科动物→家猫）
- **细粒度分类**：区分相似类别间的微小差异（例如：鸟类或车型识别）

### 常用术语与评估指标
- **精度(Accuracy)**：正确分类的样本比例
- **精确率(Precision)**：预测为正的样本中真正为正的比例
- **召回率(Recall)**：真正为正的样本中被正确预测的比例
- **F1分数**：精确率和召回率的调和平均
- **混淆矩阵**：展示预测类别与真实类别的对应关系
- **Top-k准确率**：预测的k个最可能类别中包含真实类别的比例
- **ROC曲线和AUC**：展示不同阈值下的真正例率与假正例率关系

### 主流图像分类数据集
- **MNIST**：手写数字，10类，28×28灰度图像，6万训练+1万测试
- **CIFAR-10/100**：小尺寸自然图像，10/100类，32×32彩色图像，5万训练+1万测试
- **ImageNet**：大规模视觉识别挑战数据集，1000类，超过120万训练图像
- **Caltech-101/256**：物体分类，101/256类，每类约30-800张图像
- **PASCAL VOC**：视觉对象分类，20类，训练+验证共11K图像
- **MS COCO**：大规模物体检测和分类，80类，33万图像，20万标注

## 图像分类实践流程

### 数据准备与预处理
- **数据收集**：公开数据集或自建数据集
- **数据清洗**：移除噪声数据、修正错误标签
- **数据增强**：
  - 几何变换：旋转、缩放、平移、翻转、裁剪
  - 色彩变换：亮度、对比度、饱和度调整，颜色抖动
  - 噪声添加：高斯噪声、椒盐噪声
  - 混合增强：CutMix、MixUp、Mosaic等
- **数据规范化**：
  - 像素值缩放（如0-255→0-1或-1-1）
  - 通道均值和标准差归一化
- **数据分割**：训练集、验证集、测试集划分

### 模型选择与构建
- **基础模型**：
  - 传统CNN：LeNet、AlexNet、VGG
  - 高效架构：ResNet、Inception/GoogLeNet、DenseNet
  - 轻量级模型：MobileNet、ShuffleNet、EfficientNet
  - 新兴架构：Vision Transformer (ViT)、ConvNeXt
- **模型定制**：
  - 调整输入尺寸和通道数
  - 修改网络层结构
  - 替换分类头
  - 添加正则化层
- **预训练策略**：
  - 从头训练
  - 迁移学习（固定特征提取器）
  - 微调（调整部分或全部预训练权重）

### 训练与优化
- **损失函数选择**：
  - 交叉熵损失（多分类标准选择）
  - 二元交叉熵（二分类或多标签）
  - 焦点损失（处理类别不平衡）
  - 标签平滑（提高泛化能力）
- **优化器选择**：
  - SGD（随机梯度下降）
  - Adam/AdamW
  - RMSProp
  - SGDM（带动量SGD）
- **学习率策略**：
  - 固定学习率
  - 学习率衰减（线性、阶梯式、余弦）
  - 循环学习率
  - 在温预热
- **训练技巧**：
  - 批归一化
  - 权重衰减（L2正则化）
  - Dropout
  - 提前停止（Early Stopping）
  - 梯度裁剪
  - 混合精度训练

### 评估与改进
- **验证集评估**：使用各种指标评估模型在验证集上的表现
- **测试集评估**：最终在测试集上评估模型性能
- **错误分析**：
  - 检查混淆矩阵
  - 分析错误分类的样本
  - 识别存在挑战的类别
- **模型解释**：
  - 可视化卷积核和特征图
  - 类激活映射（CAM/Grad-CAM）
  - LIME或SHAP解释
- **模型集成**：
  - 投票集成
  - 平均预测概率
  - 堆叠集成
  - TTA（测试时增强）

### 部署与应用
- **模型压缩**：
  - 量化（int8/int4）
  - 剪枝
  - 知识蒸馏
  - 低秩分解
- **模型转换**：
  - ONNX
  - TensorRT
  - CoreML
  - TensorFlow Lite
- **推理优化**：
  - 批处理
  - 计算图优化
  - 算子融合
- **部署平台**：
  - 云服务器
  - 边缘设备
  - 移动设备
  - Web浏览器（TensorFlow.js）

## 实践代码示例

### 使用PyTorch实现完整图像分类流程

#### 1. 数据准备与加载
```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# 定义数据增强和预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集（以CIFAR-10为例）
full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                           download=True, transform=train_transform)

# 分割训练集和验证集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 修改验证集的变换
val_dataset.dataset.transform = val_transform

# 测试集
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=val_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')
```

#### 2. 构建/加载预训练模型
```python
import torch.nn as nn
import torchvision.models as models

# 选择预训练模型（ResNet-18）
def create_model(num_classes=10, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    
    # 修改分类器头部以匹配目标类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

# 实例化模型
model = create_model(num_classes=len(classes))

# 移至GPU（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 打印模型结构（可选）
print(model)
```

#### 3. 定义训练函数
```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

def train_model(model, train_loader, val_loader, num_epochs=10):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 跟踪最佳模型
    best_acc = 0.0
    best_model_wts = model.state_dict().copy()
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                dataloader = train_loader
            else:
                model.eval()   # 设置模型为评估模式
                dataloader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            # 迭代数据
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 如果是训练阶段，则反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 如果是最佳验证准确率，保存模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
                
        print()
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

# 训练模型
trained_model = train_model(model, train_loader, val_loader, num_epochs=10)

# 保存模型
torch.save(trained_model.state_dict(), 'cifar10_resnet18.pth')
```

#### 4. 评估模型
```python
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, dataloader, class_names):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 打印分类报告
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    
    # 绘制混淆矩阵热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # 计算总体准确率
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    return accuracy, report, cm

# 评估最终模型
evaluate_model(trained_model, test_loader, classes)
```

#### 5. 模型解释与可视化
```python
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz
import cv2

def visualize_model_explanation(model, img_tensor, class_idx, class_names):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 使用GuidedGradCam解释预测
    guided_gc = GuidedGradCam(model, model.layer4[1].conv2)
    attributions = guided_gc.attribute(img_tensor, target=class_idx)
    
    # 将属性转换为热力图
    attr = attributions.squeeze().cpu().permute(1, 2, 0).detach().numpy()
    attr = np.sum(attr, axis=2)
    
    # 归一化热力图
    attr = (attr - attr.min()) / (attr.max() - attr.min())
    
    # 获取原始图像
    orig_img = img_tensor.squeeze().cpu().permute(1, 2, 0).detach().numpy()
    orig_img = (orig_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    orig_img = np.clip(orig_img, 0, 1)
    
    # 创建热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * attr), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # 叠加热力图
    superimposed = 0.6 * heatmap + 0.4 * orig_img
    
    # 绘制
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(orig_img)
    axs[0].set_title(f'Original - Class: {class_names[class_idx]}')
    axs[0].axis('off')
    
    axs[1].imshow(heatmap)
    axs[1].set_title('Attribution Heatmap')
    axs[1].axis('off')
    
    axs[2].imshow(superimposed)
    axs[2].set_title('Superimposed')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# 获取并可视化一个样本
dataiter = iter(test_loader)
images, labels = next(dataiter)
img = images[0]
label = labels[0]

# 可视化模型解释
visualize_model_explanation(trained_model, img, label, classes)
```

#### 6. 模型部署准备
```python
import torch.onnx

def export_to_onnx(model, sample_input, onnx_path):
    # 确保模型处于评估模式
    model.eval()
    
    # 导出模型为ONNX格式
    torch.onnx.export(
        model,                     # 要导出的模型
        sample_input,              # 模型输入样例
        onnx_path,                 # 导出文件路径
        export_params=True,        # 导出模型权重
        opset_version=11,          # ONNX版本
        do_constant_folding=True,  # 优化常量折叠
        input_names=['input'],     # 输入名称
        output_names=['output'],   # 输出名称
        dynamic_axes={             # 动态轴（批次大小可变）
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {onnx_path}")

# 创建样本输入
sample_input = torch.randn(1, 3, 224, 224).to(device)

# 导出模型
export_to_onnx(trained_model, sample_input, "cifar10_resnet18.onnx")
```

### 使用TensorFlow/Keras实现图像分类

#### 1. 数据准备与加载
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 加载CIFAR-10数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 规范化像素值
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 类别编码
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 划分训练集和验证集
val_size = 10000
x_val = x_train[-val_size:]
y_val = y_train[-val_size:]
x_train = x_train[:-val_size]
y_train = y_train[:-val_size]

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)

# 类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
```

#### 2. 构建模型
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

def create_model(input_shape=(32, 32, 3), num_classes=10):
    # 调整输入大小
    inputs = Input(shape=input_shape)
    # 使用ResNet50作为基础模型，不包括顶层
    # 注意：因为CIFAR-10图像较小，所以会调整网络结构
    resized_inputs = tf.keras.layers.experimental.preprocessing.Resizing(
        224, 224)(inputs)
    
    base_model = ResNet50(weights='imagenet', include_top=False, 
                         input_tensor=resized_inputs)
    
    # 冻结基础模型
    base_model.trainable = False
    
    # 添加全局平均池化和分类器
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # 构建完整模型
    model = Model(inputs=inputs, outputs=predictions)
    
    return model

# 创建模型
model = create_model(input_shape=(32, 32, 3), num_classes=10)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 打印模型摘要
model.summary()
```

#### 3. 训练模型
```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 设置回调
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    mode='min',
    verbose=1
)

# 训练模型
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# 微调模型 - 解冻一部分层
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):  # 这是基础ResNet模型
        for i, resnet_layer in enumerate(layer.layers):
            # 解冻最后15层
            if i > len(layer.layers) - 15:
                resnet_layer.trainable = True
            else:
                resnet_layer.trainable = False

# 使用较小的学习率重新编译
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 额外微调几个epoch
history_fine = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint, early_stopping, reduce_lr]
)
```

#### 4. 评估和可视化
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 绘制训练历史
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 绘制训练历史
plot_history(history)
if 'history_fine' in locals():
    plot_history(history_fine)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f'Test accuracy: {test_acc:.4f}')

# 生成预测
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred_classes)

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 保存模型
model.save('cifar10_resnet_model.h5')
```

#### 5. 可视化类激活映射(CAM)
```python
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 创建一个模型从输入图像到最后一个卷积层的激活
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # 在这个监控下传递样本图像
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # 提取梯度
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # 池化梯度
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 加权激活
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # 归一化
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    # 调整大小
    heatmap = np.uint8(255 * heatmap)
    
    # 使用jet颜色映射
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # 创建热力图图像
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # 叠加图像
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    return superimposed_img

# 选择一个样本图像
img_index = 12
img = x_test[img_index] * 255.0  # 恢复到0-255范围
img = img.astype(np.uint8)

# 准备输入
img_array = np.expand_dims(x_test[img_index], axis=0)

# 找到最后一个卷积层名称
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break
    elif isinstance(layer, tf.keras.Model):  # ResNet是一个嵌套模型
        for internal_layer in reversed(layer.layers):
            if isinstance(internal_layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = internal_layer.name
                break
        if last_conv_layer_name:
            break

# 生成类激活热力图
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# 显示原始图像
plt.figure(figsize=(16, 5))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title(f'Original - {class_names[y_true[img_index]]}')
plt.axis('off')

# 显示热力图
plt.subplot(1, 3, 2)
plt.imshow(heatmap)
plt.title('Grad-CAM Heatmap')
plt.axis('off')

# 显示叠加图像
superimposed = display_gradcam(img, heatmap)
plt.subplot(1, 3, 3)
plt.imshow(superimposed)
plt.title('Superimposed')
plt.axis('off')

plt.tight_layout()
plt.show()
```

## 高级技术与最佳实践

### 处理类别不平衡
- **重采样技术**：
  - 欠采样多数类：随机移除或聚类移除
  - 过采样少数类：复制、SMOTE、ADASYN
- **类权重调整**：根据类别频率设置权重
- **焦点损失(Focal Loss)**：降低容易样本的贡献，关注难样本
- **混合技术**：结合重采样和损失调整
- **数据增强集中**：对少数类应用更多数据增强

### 处理小数据集
- **迁移学习**：利用在大型数据集上预训练的模型
- **极致数据增强**：更激进的增强策略扩充训练集
- **正则化技术**：Dropout、L1/L2正则化、批归一化
- **半监督学习**：利用未标注数据辅助训练
- **数据合成**：使用GAN生成新样本
- **预训练-微调策略**：逐步解冻和微调网络层

### 模型集成方法
- **基本集成**：
  - 投票法：多个模型投票决定最终类别
  - 平均法：对各模型预测概率取平均
- **Bagging**：
  - 随机森林
  - 多重采样训练CNN
- **Boosting**：
  - AdaBoost
  - 梯度提升
- **Stacking**：使用元学习器组合多个基础模型
- **Snapshot Ensemble**：在单次训练中保存多个检查点
- **测试时增强(TTA)**：对测试图像应用多种变换并平均结果

### 最佳实践与技巧
- **初始化策略**：He初始化、Xavier初始化
- **混合精度训练**：使用FP16和FP32混合精度
- **经验法则**：
  - 从小模型开始，逐步扩大
  - 先使用简单的数据增强，再尝试复杂方法
  - 使用学习率预热和退火
  - 批量大小平衡计算效率和泛化性
- **团队协作**：
  - 版本控制（Git）管理代码
  - 实验跟踪（MLflow, Weights & Biases）
  - 模型打包和分享（ONNX, SavedModel）
- **调试提示**：
  - 先在小数据集上验证
  - 过拟合小批次数据检查模型容量
  - 记录和可视化中间结果
  - 分析模型失败案例

## 实际应用场景案例

### 零售商品识别
- **应用**：识别超市货架上的商品和品牌
- **挑战**：品类多、视角变化、光照不均、遮挡
- **数据集**：RPC (Retail Product Checkout)、RP2K、SKU-110K
- **方法**：
  - 细粒度分类，识别相似产品
  - 多标签分类处理一张图像多个产品
  - 模型蒸馏以适应移动设备

### 医学图像分类
- **应用**：X光、CT、MRI图像疾病诊断
- **挑战**：数据稀缺、类别不平衡、专业知识要求高
- **数据集**：ISIC (皮肤病变)、ChestX-ray14、RSNA肺炎检测
- **方法**：
  - 迁移学习处理小数据集
  - 特殊数据增强保持医学特征
  - 解释性技术提高医生信任
  - 多模态融合（图像+病历）

### 农业作物监测
- **应用**：识别农作物类型、疾病、害虫等
- **挑战**：环境变化、疾病相似性、实时性要求
- **数据集**：PlantVillage、Kaggle玉米叶片疾病
- **方法**：
  - 考虑作物生长周期的分类系统
  - 轻量级模型适应边缘设备
  - 定期重训练适应季节变化
  - 结合专家知识的分层分类

### 工业质检
- **应用**：检测生产线上产品缺陷
- **挑战**：缺陷样本少、实时要求高、环境受控
- **方法**：
  - 异常检测与分类结合
  - 定制数据增强模拟真实缺陷
  - 注意力机制关注潜在缺陷区域
  - 无监督/半监督学习应对样本稀缺

## 学习资源

### 书籍与教程
- 《Deep Learning for Computer Vision》by Adrian Rosebrock
- 《Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow》by Aurélien Géron
- 《Deep Learning》by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- 《Computer Vision: Algorithms and Applications》by Richard Szeliski
- 《动手学深度学习》by 李沐等

### 在线课程
- CS231n: 斯坦福计算机视觉卷积神经网络课程
- fast.ai: 实用深度学习课程
- Coursera: 深度学习专项课程(Andrew Ng)
- Udacity: 计算机视觉纳米学位

### 竞赛与实战
- Kaggle图像分类竞赛
- AI Challenger竞赛
- CVPR, ICCV, ECCV研讨会挑战

### 开源项目与框架
- PyTorch Image Models (timm)
- TensorFlow 模型花园
- MMClassification
- Keras Applications
- fastai

## 未来发展方向

### 新兴趋势
- **自监督学习**：无需标签的表示学习
- **少样本学习**：用最少的标签学习新类别
- **多模态学习**：结合图像、文本、音频等多种数据形式
- **神经架构搜索(NAS)**：自动发现最优网络结构
- **视觉Transformer模型**：ViT、Swin Transformer等
- **绿色AI**：节能高效的模型架构和训练方法

### 行业发展
- **边缘部署**：模型压缩与硬件协同优化
- **云-边协同**：混合部署策略
- **特定领域芯片**：NPU、视觉AI芯片
- **标准化与互操作性**：框架间互通、ONNX等标准

### 研究前沿
- **可解释AI**：更好理解分类决策过程
- **对抗鲁棒性**：抵抗对抗样本攻击
- **公平性与偏见缓解**：减少模型中的不公平偏见
- **隐私保护学习**：联邦学习、差分隐私
- **物理信息结合**：融入物理约束的视觉模型