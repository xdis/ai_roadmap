# 神经网络可视化：从零掌握这一深度学习核心技术

## 1. 基础概念理解

### 什么是神经网络可视化？

神经网络可视化是一套用于呈现、解释和分析深度学习模型内部工作机制的技术和方法。它让我们能够将神经网络这一"黑盒"转变为可理解、可分析的系统，帮助研究人员和工程师更好地理解模型决策过程。

### 为什么需要神经网络可视化？

1. **模型解释性**：理解模型是如何做出决策的，提高模型透明度和可信度
2. **诊断与调试**：找出模型存在的问题，如过拟合、梯度消失或爆炸
3. **模型优化**：基于可视化结果改进模型架构和超参数
4. **知识提取**：发现模型学到的有意义表示和特征
5. **教育价值**：帮助更直观地理解神经网络的工作原理

### 神经网络可视化的分类

神经网络可视化方法可以分为以下几类：

1. **网络架构可视化**：展示模型的结构和连接
2. **权重与激活可视化**：呈现模型参数和中间激活值
3. **特征可视化**：理解网络学习到的特征和表示
4. **注意力机制可视化**：展示模型关注的输入区域
5. **流程可视化**：展示信息在网络中的传播过程
6. **决策边界可视化**：直观呈现模型的决策规则

### 可视化的关键挑战

1. **高维数据降维**：神经网络处理高维数据，需要降维以便可视化
2. **可解释性与准确性的权衡**：简化可视化可能损失准确性
3. **计算开销**：某些可视化技术计算成本高
4. **适用性**：不同类型的网络(CNN、RNN、Transformer)需要不同的可视化方法

## 2. 技术细节探索

### 网络架构可视化

#### 计算图可视化

计算图展示了模型中的操作和数据流，帮助理解整体架构：

```python
# TensorFlow示例 - 使用TensorBoard可视化计算图
import tensorflow as tf

# 创建一个简单的模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 设置TensorBoard回调
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="./logs", histogram_freq=1, write_graph=True)

# 编译并训练模型
model.compile(optimizer='adam', loss='categorical_crossentropy')
# model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])

# 然后运行: tensorboard --logdir=./logs
```

#### 层次结构可视化

层次结构可视化展示网络的层级关系，有助于理解复杂网络：

```python
# 在PyTorch中使用torchviz可视化模型
import torch
from torchviz import make_dot

# 定义模型
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, 3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(32*14*14, 10)
)

# 创建一个示例输入
x = torch.randn(1, 3, 32, 32)
y = model(x)

# 生成计算图可视化
dot = make_dot(y, params=dict(model.named_parameters()))
# dot.render("model_architecture", format="png")
```

### 权重与参数可视化

#### 权重直方图与分布

观察权重的分布可以帮助发现潜在问题，如梯度消失：

```python
# PyTorch中的权重直方图可视化
import matplotlib.pyplot as plt
import numpy as np

# 获取模型所有权重
weights = []
for name, param in model.named_parameters():
    if 'weight' in name:
        weights.append(param.data.cpu().numpy().flatten())

# 将所有权重合并为一个数组
all_weights = np.concatenate(weights)

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(all_weights, bins=50)
plt.title('Distribution of Weights')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()
```

#### 卷积滤波器可视化

可视化卷积层的滤波器可以展示它们学习到的特征检测器：

```python
def visualize_filters(model, layer_idx, grid_size=(8, 8), figsize=(12, 12)):
    """可视化卷积层的滤波器"""
    # 获取特定层
    layer = model.layers[layer_idx]
    
    # 获取层的权重
    filters, biases = layer.get_weights()
    
    # 规范化滤波器权重以便可视化
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    # 设置图像大小和网格
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    
    # 绘制每个滤波器
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[3]:
            # 绘制单个滤波器
            ax.imshow(filters[:, :, 0, i], cmap='viridis')
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用示例
# visualize_filters(model, 0)  # 可视化第一个卷积层的滤波器
```

### 特征可视化

#### 激活图(Activation Maps)

激活图展示了神经网络每一层对输入的反应：

```python
def get_activation_maps(model, layer_names, images):
    """获取指定层的激活图"""
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    activations = activation_model.predict(images)
    
    return dict(zip(layer_names, activations))

def plot_activation_maps(activations, layer_name, image_idx=0, num_features=64):
    """绘制激活图"""
    activation = activations[layer_name]
    
    # 设置网格大小
    grid_size = int(np.ceil(np.sqrt(num_features)))
    
    # 创建图形
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    # 绘制激活图
    for i in range(min(num_features, activation.shape[-1])):
        ax = axes[i]
        feature = activation[image_idx, :, :, i]
        ax.imshow(feature, cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Feature {i+1}')
    
    # 隐藏空白子图
    for i in range(activation.shape[-1], len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用示例
# activations = get_activation_maps(model, ['conv2d_1', 'conv2d_2'], images)
# plot_activation_maps(activations, 'conv2d_1')
```

#### t-SNE/UMAP降维可视化

对神经网络中高维特征进行降维，观察数据点的分布：

```python
# 使用t-SNE可视化特征空间
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def extract_features(model, layer_name, data):
    """从特定层提取特征"""
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    features = feature_model.predict(data)
    return features

def visualize_tsne(features, labels, n_components=2, perplexity=30):
    """使用t-SNE进行降维并可视化"""
    # 应用t-SNE降维
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(features)
    
    # 创建数据框用于可视化
    tsne_df = pd.DataFrame(data=tsne_result, columns=['t-SNE 1', 't-SNE 2'])
    tsne_df['Label'] = labels
    
    # 绘制结果
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='t-SNE 1', y='t-SNE 2',
        hue='Label',
        palette=sns.color_palette("hls", len(np.unique(labels))),
        data=tsne_df,
        legend="full",
        alpha=0.3
    )
    plt.title('t-SNE visualization of features')
    plt.show()

# 使用示例
# features = extract_features(model, 'flatten', x_test)
# visualize_tsne(features, y_test)
```

### 注意力机制可视化

注意力可视化展示了模型关注的输入区域，尤其适用于Transformer等架构：

```python
# 可视化Transformer模型的注意力权重
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens, layer=0, head=0):
    """可视化注意力权重"""
    # 获取特定层和头的注意力权重
    attn = attention_weights[layer][head].numpy()
    
    # 创建热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn, 
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0
    )
    plt.title(f"Attention weights for layer {layer}, head {head}")
    plt.tight_layout()
    plt.show()

# 使用示例(需要BERT等Transformer模型)
# inputs = tokenizer(["This is an example sentence"], return_tensors="tf")
# outputs = model(inputs, output_attentions=True)
# attention_weights = outputs.attentions
# visualize_attention(attention_weights, tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
```

### 梯度可视化

#### 梯度流可视化

跟踪梯度的流动可以帮助发现训练问题：

```python
# 可视化梯度流
import numpy as np
import matplotlib.pyplot as plt

def plot_gradient_flow(named_parameters):
    """绘制模型中所有梯度的范数"""
    ave_grads = []
    max_grads= []
    layers = []
    
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    
    plt.figure(figsize=(10, 8))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], 
              ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.show()

# 使用示例
# model.zero_grad()
# loss = criterion(output, target)
# loss.backward()
# plot_gradient_flow(model.named_parameters())
```

#### 类激活映射(CAM)与Grad-CAM

Grad-CAM可以突出显示对模型预测贡献最大的图像区域：

```python
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """生成Grad-CAM热力图"""
    # 获取最后一层卷积层的输出和模型输出
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 获取类别预测的梯度
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 梯度与输出特征图的乘积
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 加权求和生成热力图
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

    # 标准化热力图
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    """显示原始图像和叠加的Grad-CAM热力图"""
    # 读取原始图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整热力图大小以匹配原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 叠加热力图和原始图像
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img / 255.0, 0, 1)
    
    # 显示结果
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img / 255.0)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(superimposed_img)
    ax[1].set_title('Grad-CAM Heatmap')
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# 使用示例
# image = preprocess_image(img_path)
# heatmap = make_gradcam_heatmap(image, model, 'conv5_block3_out')
# display_gradcam(img_path, heatmap)
```

## 3. 实践与实现

### PyTorch中的可视化实现

#### 使用TensorBoard与PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# 设置数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 创建模型
model = SimpleCNN()

# 设置TensorBoard
writer = SummaryWriter('runs/mnist_cnn')

# 添加计算图
images, _ = next(iter(trainloader))
grid = torchvision.utils.make_grid(images)
writer.add_image('mnist_images', grid)
writer.add_graph(model, images)

# 训练模型并记录指标
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):  # 仅做演示，通常会训练多个epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            # 记录损失
            writer.add_scalar('training loss',
                            running_loss / 100,
                            epoch * len(trainloader) + i)
            
            # 记录权重和梯度直方图
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            
            running_loss = 0.0

# 关闭SummaryWriter
writer.close()
```

#### 特征可视化与类激活映射

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 加载预训练模型
model = models.resnet50(pretrained=True)
model.eval()

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 提取特征和梯度的钩子函数
class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None
    
    def hook_fn(self, module, input, output):
        self.features = output.detach()
    
    def remove(self):
        self.hook.remove()

class SaveGradients:
    def __init__(self, module):
        self.hook = module.register_backward_hook(self.hook_fn)
        self.gradients = None
    
    def hook_fn(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
    
    def remove(self):
        self.hook.remove()

# 实现GradCAM
def grad_cam(model, image, target_class=None):
    # 获取最后一个卷积层
    final_layer = model.layer4[-1].conv3
    
    # 注册钩子
    final_activations = SaveFeatures(final_layer)
    final_gradients = SaveGradients(final_layer)
    
    # 前向传播
    x = preprocess(image).unsqueeze(0)  # 添加批次维度
    logits = model(x)
    
    # 如果没有指定目标类，则使用预测概率最高的类
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    
    # 反向传播以获取梯度
    model.zero_grad()
    logits[0, target_class].backward()
    
    # 计算加权特征图
    gradients = final_gradients.gradients.mean([0, 2, 3], keepdim=True)
    activations = final_activations.features
    weighted_activations = activations * gradients
    
    # 应用ReLU并归一化
    heatmap = torch.sum(weighted_activations, dim=1).squeeze()
    heatmap = torch.maximum(heatmap, torch.tensor(0.))
    heatmap = heatmap.detach().numpy()
    
    # 归一化到0-1范围
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max()
    
    # 移除钩子
    final_activations.remove()
    final_gradients.remove()
    
    return heatmap, target_class

def show_cam_on_image(img, heatmap, alpha=0.5):
    # 调整热力图大小以匹配原图
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # 应用颜色映射
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 将热力图和原图混合
    superimposed_img = np.float32(heatmap) * alpha + np.float32(img)
    superimposed_img = np.clip(superimposed_img / 255, 0, 1)
    
    return superimposed_img

# 使用示例
if __name__ == "__main__":
    # 加载图像
    img_path = "path/to/your/image.jpg"
    img = Image.open(img_path)
    
    # 获取GradCAM热力图
    heatmap, predicted_class = grad_cam(model, img)
    
    # 将热力图应用到原图上
    orig_img = np.array(img)
    superimposed = show_cam_on_image(orig_img, heatmap)
    
    # 显示结果
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(heatmap)
    ax[1].set_title('GradCAM Heatmap')
    ax[1].axis('off')
    
    ax[2].imshow(superimposed)
    ax[2].set_title('GradCAM Overlay')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()
```

### TensorFlow/Keras中的可视化实现

#### 中间激活可视化

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 加载预训练模型
model = keras.applications.VGG16(weights='imagenet', include_top=True)

# 创建一个模型，获取中间层输出
layer_outputs = [layer.output for layer in model.layers[1:8]]  # 前几个卷积层的输出
activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)

# 加载一张图片
img_path = 'path/to/your/image.jpg'
img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = keras.applications.vgg16.preprocess_input(img_array)

# 获取所有层的激活
activations = activation_model.predict(img_array)

# 可视化每个层的激活
layer_names = [layer.name for layer in model.layers[1:8]]
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    # 获取层中特征图数量
    n_features = layer_activation.shape[-1]
    
    # 特征图的大小
    size = layer_activation.shape[1]
    
    # 在网格中显示特征图
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            if channel_index < n_features:
                # 处理第channel_index个特征图
                channel_image = layer_activation[0, :, :, channel_index]
                
                # 处理特征可视化
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std() + 1e-5
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                
                # 将其放入网格中
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
    
    # 显示网格
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()
```

#### 嵌入层可视化

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist

# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 预处理数据
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# 创建一个简单的模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu', name='embedding'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译和训练模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

# 创建一个新模型，输出嵌入层
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('embedding').output)

# 获取测试数据的嵌入
embeddings = embedding_model.predict(x_test)

# 使用t-SNE将嵌入降维到2D空间
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# 可视化嵌入
plt.figure(figsize=(12, 10))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_test, 
                     cmap='tab10', alpha=0.5)
plt.colorbar(scatter)
plt.title('t-SNE Visualization of MNIST Embeddings')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.show()
```

## 4. 高级应用与变体

### 深度梦境(Deep Dream)

深度梦境是一种增强图像中网络检测到的模式的技术：

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

# 加载预训练的InceptionV3模型
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# 选择层
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]

# 创建特征提取模型
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

def preprocess_image(image_path):
    """预处理图像以符合InceptionV3的要求"""
    img = PIL.Image.open(image_path)
    img = img.resize((512, 512))  # 调整大小
    img = np.array(img)
    
    # 添加批次维度
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(img)

def deprocess(img):
    """将张量转换回图像"""
    img = img[0]  # 移除批次维度
    
    # 将像素值映射回0-255范围
    img = (img - np.min(img)) * 255 / (np.max(img) - np.min(img))
    return np.uint8(img)

def calc_loss(img, model):
    """计算特征图的损失（我们想要最大化这个损失）"""
    # 将图像传递给模型
    outputs = model(img)
    
    # 我们希望放大这些层所有的激活
    loss = tf.zeros(shape=())
    
    for activation in outputs:
        # 通过添加激活的均值来增加损失
        loss += tf.reduce_mean(activation)
        
    return loss

@tf.function
def deepdream_step(img, model, step_size):
    """单个deep dream步骤"""
    with tf.GradientTape() as tape:
        # 这个需要梯度
        tape.watch(img)
        loss = calc_loss(img, model)
    
    # 计算图像关于损失的梯度
    gradients = tape.gradient(loss, img)
    
    # 标准化梯度
    gradients = tf.math.reduce_std(gradients) / (tf.reduce_mean(tf.abs(gradients)) + 1e-8)
    
    # 更新图像以最大化损失
    img = img + gradients * step_size
    img = tf.clip_by_value(img, 0, 255)
    
    return img

def run_deep_dream_with_octaves(model, img, steps_per_octave=100, step_size=0.01, octaves=3, octave_scale=1.3):
    """使用多尺度方法运行深度梦境"""
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    
    base_shape = img.shape[1:3]
    
    img = tf.convert_to_tensor(img)
    
    # 存储原始图像
    orig_img = img
    
    for octave in range(octaves):
        # 调整图像大小
        new_size = tf.cast(tf.convert_to_tensor(base_shape) * (octave_scale ** octave), tf.int32)
        img = tf.image.resize(img, new_size)
        
        for step in range(steps_per_octave):
            # 应用深度梦境步骤
            img = deepdream_step(img, model, step_size)
        
        # 上采样图像以准备下一个尺度
        img = tf.image.resize(img, base_shape)
        
        # 加回更多的细节
        img = tf.keras.applications.inception_v3.preprocess_input(orig_img + img)
    
    return deprocess(img)

# 使用示例
img_path = 'path/to/your/image.jpg'
original_img = preprocess_image(img_path)
dream_img = run_deep_dream_with_octaves(dream_model, original_img, 
                                        steps_per_octave=50, step_size=0.01)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(deprocess(original_img))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(dream_img)
plt.title('Deep Dream')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 神经风格迁移(Neural Style Transfer)

神经风格迁移可以将一张图片的艺术风格应用到另一张图片上：

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import time

# 加载VGG19模型
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# 内容层和风格层
content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

def vgg_layers(layer_names):
    """ 创建一个提取指定层输出的模型 """
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# 加载和预处理图像
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# 计算内容损失
def content_loss(base_content, target_content):
    return tf.reduce_mean(tf.square(base_content - target_content))

# 计算风格损失
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def style_loss(base_style, target_style):
    gram_base = gram_matrix(base_style)
    gram_target = gram_matrix(target_style)
    return tf.reduce_mean(tf.square(gram_base - gram_target))

# 总变分损失，用于平滑图像
def total_variation_loss(image):
    x_deltas = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_deltas = image[:, 1:, :, :] - image[:, :-1, :, :]
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

# 计算总损失
def style_content_loss(outputs, content_targets, style_targets, 
                      content_weight=1e3, style_weight=1e-2, tv_weight=30):
    content_outputs = outputs[:len(content_layers)]
    style_outputs = outputs[len(content_layers):]

    content_loss_value = tf.add_n([content_loss(content_outputs[i], content_targets[i])
                              for i in range(len(content_layers))])
    content_loss_value *= content_weight / len(content_layers)

    style_loss_value = tf.add_n([style_loss(style_outputs[i], style_targets[i])
                             for i in range(len(style_layers))])
    style_loss_value *= style_weight / len(style_layers)

    tv_loss = total_variation_loss(outputs[-1]) * tv_weight

    total_loss = content_loss_value + style_loss_value + tv_loss
    
    return total_loss

# 风格迁移的主要函数
def run_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    # 加载图像
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    
    # 定义优化变量
    image = tf.Variable(content_image)
    
    # 创建特征提取器
    extractor = vgg_layers(content_layers + style_layers)
    
    # 预处理图像
    content_image = tf.keras.applications.vgg19.preprocess_input(content_image*255.0)
    style_image = tf.keras.applications.vgg19.preprocess_input(style_image*255.0)
    
    # 获取目标特征
    content_targets = extractor(content_image)[:len(content_layers)]
    style_targets = extractor(style_image)[len(content_layers):]
    
    # 优化器
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    
    # 显示进度图像的函数
    def tensor_to_image(tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)
    
    # 优化过程
    best_loss = float('inf')
    best_img = None
    
    # 迭代优化
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            # 预处理当前图像
            preprocessed_img = tf.keras.applications.vgg19.preprocess_input(image*255.0)
            # 提取特征
            outputs = extractor(preprocessed_img)
            outputs.append(image)  # 添加图像用于计算TV损失
            # 计算损失
            loss = style_content_loss(outputs, content_targets, style_targets,
                                    content_weight, style_weight)
            
        # 计算梯度
        grads = tape.gradient(loss, image)
        # 优化
        opt.apply_gradients([(grads, image)])
        # 裁剪图像保持像素值在[0,1]范围
        image.assign(tf.clip_by_value(image, 0.0, 1.0))
        
        # 保存最佳图像
        if loss < best_loss:
            best_loss = loss
            best_img = tensor_to_image(image)
        
        # 输出进度
        if i % 50 == 0:
            print(f'Iteration: {i}, Loss: {loss.numpy():.4f}')
    
    return best_img

# 使用示例
content_path = 'path/to/content/image.jpg'
style_path = 'path/to/style/image.jpg'

result = run_style_transfer(content_path, style_path, num_iterations=500)
result.save('stylized_image.jpg')

# 显示结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(load_img(content_path)[0])
plt.title('Content Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(load_img(style_path)[0])
plt.title('Style Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title('Stylized Image')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 注意力可视化与决策解释

```python
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# 加载模型
model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5"
model = tf.keras.Sequential([
    hub.KerasLayer(model_url)
])
model.build([None, 224, 224, 3])

def preprocess_image(image_path, target_size=(224, 224)):
    """预处理图像"""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    
    if img.shape[-1] == 4:  # 如果有alpha通道
        img = img[..., :3]
        
    return img

def get_integrated_gradients(img_array, model, class_idx, baseline=None, steps=50):
    """计算Integrated Gradients"""
    # 如果没有提供baseline，则使用零图像
    if baseline is None:
        baseline = np.zeros_like(img_array)
        
    # 创建从baseline到图像的路径
    alphas = np.linspace(0, 1, steps+1)
    
    # 在路径上生成插值图像
    interpolated_images = np.zeros((steps+1,) + img_array.shape)
    for i, alpha in enumerate(alphas):
        interpolated_images[i] = baseline + alpha * (img_array - baseline)
    
    # 将图像批量传递给模型，并计算梯度
    ig_grads = np.zeros_like(img_array)
    
    # 将图像转换为张量
    interpolated_tensors = tf.convert_to_tensor(interpolated_images, dtype=tf.float32)
    
    # 计算所有内插图像的梯度
    with tf.GradientTape() as tape:
        tape.watch(interpolated_tensors)
        preds = model(interpolated_tensors)
        outputs = preds[:, class_idx]
    
    grads = tape.gradient(outputs, interpolated_tensors)
    grads = grads.numpy()
    
    # 积分近似（梯度乘以输入差异的平均值）
    avg_grads = np.average(grads, axis=0)
    ig_grads = (img_array - baseline) * avg_grads
    
    return ig_grads

def visualize_attributions(img, attributions, percentile=99):
    """可视化归因"""
    # 对归因进行归一化
    attributions = np.sum(np.abs(attributions), axis=-1)
    attributions = attributions / np.max(attributions)
    
    # 创建热力图
    heatmap = np.uint8(255 * attributions)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 将热力图叠加到原始图像上
    img_with_heatmap = (heatmap * 0.5 + img * 255 * 0.5).astype(np.uint8)
    
    return heatmap, img_with_heatmap

# 使用示例
def explain_prediction(image_path):
    # 加载和预处理图像
    img = preprocess_image(image_path)
    img_array = np.expand_dims(img, axis=0)
    
    # 做出预测
    preds = model.predict(img_array)
    top_pred = np.argmax(preds[0])
    
    # 打印顶部预测
    decoded_preds = decode_predictions(preds, top=3)[0]
    print("Top predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i+1}: {label} ({score:.2f})")
    
    # 计算Integrated Gradients
    attributions = get_integrated_gradients(img, model, top_pred)
    
    # 可视化
    heatmap, img_with_heatmap = visualize_attributions(img, attributions[0])
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Original Image\nPredicted: {decoded_preds[0][1]}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Attribution Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_with_heatmap)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 对图像进行解释
# explain_prediction('path/to/image.jpg')
```

### 可解释AI工具套件

```python
# 使用SHAP库解释深度学习模型的预测
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def explain_with_shap(model, images, labels, class_names=None, num_samples=100):
    """使用SHAP解释模型预测"""
    # 创建背景数据集
    background = images[:100]  # 使用一小部分数据作为背景
    
    # 创建DeepExplainer
    explainer = shap.DeepExplainer(model, background)
    
    # 选择要解释的图像
    sample_size = min(10, len(images))
    images_to_explain = images[:sample_size]
    
    # 计算SHAP值
    shap_values = explainer.shap_values(images_to_explain)
    
    # 可视化SHAP值
    shap.image_plot(shap_values, -images_to_explain, show=False)
    plt.tight_layout()
    plt.show()
    
    # 为单个图像创建更详细的可视化
    for i in range(min(3, sample_size)):
        plt.figure(figsize=(15, 5))
        
        # 显示原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(images_to_explain[i])
        if labels is not None and i < len(labels):
            true_label = class_names[labels[i]] if class_names else str(labels[i])
            plt.title(f"True: {true_label}")
        plt.axis('off')
        
        # 预测类别
        preds = model.predict(np.expand_dims(images_to_explain[i], axis=0))
        pred_class = np.argmax(preds[0])
        pred_label = class_names[pred_class] if class_names else str(pred_class)
        
        # 显示该类的SHAP值
        plt.subplot(1, 3, 2)
        plt.imshow(shap_values[pred_class][i].sum(-1), cmap='hot')
        plt.title(f"SHAP values for '{pred_label}'")
        plt.axis('off')
        
        # 显示SHAP叠加
        plt.subplot(1, 3, 3)
        # 将SHAP值归一化并创建热力图
        shap_abs = np.abs(shap_values[pred_class][i].sum(-1))
        shap_norm = shap_abs / np.max(shap_abs)
        img = images_to_explain[i]
        heatmap = np.uint8(255 * shap_norm)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 叠加热力图
        img_with_heatmap = (heatmap * 0.7 + img * 255 * 0.3).astype(np.uint8)
        plt.imshow(img_with_heatmap)
        plt.title('SHAP Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# 使用示例
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# x_train = x_train.astype('float32') / 255.0
# explain_with_shap(model, x_test[:100], y_test[:100].flatten(), class_names)
```

### 交互式可视化系统

```python
# 使用Streamlit创建交互式神经网络可视化系统
# 保存为app.py并使用命令 streamlit run app.py 运行

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image
import time

# 设置页面配置
st.set_page_config(page_title="神经网络可视化工具", layout="wide")

# 标题
st.title("神经网络可视化交互式探索")

# 模型选择
model_option = st.selectbox("选择模型", ["VGG16", "MobileNetV2", "ResNet50"])

@st.cache_resource
def load_model(model_name):
    """加载预训练模型"""
    if model_name == "VGG16":
        model = VGG16(weights='imagenet')
    elif model_name == "MobileNetV2":
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
    else:  # ResNet50
        model = tf.keras.applications.ResNet50(weights='imagenet')
    return model

# 加载选定的模型
model = load_model(model_option)

# 侧边栏 - 上传图像
st.sidebar.title("上传图像")
uploaded_file = st.sidebar.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

# 侧边栏 - 可视化选项
st.sidebar.title("可视化选项")
vis_option = st.sidebar.selectbox(
    "选择可视化方法", 
    ["Grad-CAM", "Guided Backpropagation", "Integrated Gradients", "Deep Dream"]
)

# 侧边栏 - 参数设置
st.sidebar.title("参数设置")
if vis_option == "Grad-CAM":
    layer_name = st.sidebar.selectbox(
        "选择层", 
        [layer.name for layer in model.layers if 'conv' in layer.name]
    )
elif vis_option == "Deep Dream":
    octave_scale = st.sidebar.slider("Octave Scale", 1.0, 2.0, 1.3)
    num_octaves = st.sidebar.slider("Octaves", 1, 5, 3)
    steps_per_octave = st.sidebar.slider("Steps per Octave", 10, 50, 20)
    
# 实现Grad-CAM
def make_gradcam_heatmap(img_array, model, layer_name, pred_index=None):
    # 创建获取指定层输出的模型
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    # 计算梯度
    with tf.GradientTape() as tape:
        layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # 获取梯度
    grads = tape.gradient(class_channel, layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 加权并激活
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, layer_output[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

# 实现Deep Dream
def run_deep_dream(img_array, model, layer_name='mixed3', octave_scale=1.3, num_octaves=3, steps_per_octave=20):
    # 创建特征提取模型
    feature_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    def calc_loss(img, feature_extractor):
        # 计算特征图的均值作为损失
        features = feature_extractor(img)
        return tf.reduce_mean(features)
    
    @tf.function
    def deepdream_step(img, feature_extractor):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = calc_loss(img, feature_extractor)
            
        # 计算梯度
        grads = tape.gradient(loss, img)
        # 标准化梯度
        grads /= tf.math.reduce_std(grads) + 1e-8
        # 更新图像
        img = img + grads * 0.01
        # 裁剪以保持合理的值
        img = tf.clip_by_value(img, -1, 1)
        return img
    
    # 转换输入图像
    img = img_array.copy()
    img = tf.convert_to_tensor(img)
    
    for octave in range(num_octaves):
        # 调整图像大小以适应当前octave
        new_size = tf.cast(tf.shape(img)[1:3] * octave_scale**(octave), tf.int32)
        img = tf.image.resize(img, new_size)
        
        # 执行梯度上升
        for step in range(steps_per_octave):
            img = deepdream_step(img, feature_extractor)
            
    img = (img + 1) / 2  # 恢复为0-1范围
    return img.numpy()

# 处理上传的图像
if uploaded_file is not None:
    # 显示原始图像
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("原始图像")
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        st.image(pil_image, use_column_width=True)
        
        # 预处理图像
        img = pil_image.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # 做出预测
        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=3)[0]
        
        # 显示预测结果
        st.subheader("预测结果:")
        for i, (imagenet_id, label, score) in enumerate(decoded_preds):
            st.write(f"{i+1}: {label} ({score:.2f})")
    
    with col2:
        st.header(f"{vis_option}可视化")
        
        # 应用选定的可视化方法
        if vis_option == "Grad-CAM":
            with st.spinner('正在生成Grad-CAM...'):
                # 获取热力图
                heatmap = make_gradcam_heatmap(img_array, model, layer_name)
                
                # 调整热力图大小以匹配原图
                heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
                
                # 将热力图转换为RGB
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # 叠加热力图到原图
                img_array_vis = image.img_to_array(img)
                superimposed_img = heatmap * 0.4 + img_array_vis
                superimposed_img = np.clip(superimposed_img / 255.0, 0, 1)
                
                # 显示热力图和叠加图
                st.image(heatmap / 255.0, caption='热力图', use_column_width=True)
                st.image# 使用Streamlit创建交互式神经网络可视化系统
# 保存为app.py并使用命令 streamlit run app.py 运行

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image
import time

# 设置页面配置
st.set_page_config(page_title="神经网络可视化工具", layout="wide")

# 标题
st.title("神经网络可视化交互式探索")

# 模型选择
model_option = st.selectbox("选择模型", ["VGG16", "MobileNetV2", "ResNet50"])

@st.cache_resource
def load_model(model_name):
    """加载预训练模型"""
    if model_name == "VGG16":
        model = VGG16(weights='imagenet')
    elif model_name == "MobileNetV2":
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
    else:  # ResNet50
        model = tf.keras.applications.ResNet50(weights='imagenet')
    return model

# 加载选定的模型
model = load_model(model_option)

# 侧边栏 - 上传图像
st.sidebar.title("上传图像")
uploaded_file = st.sidebar.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

# 侧边栏 - 可视化选项
st.sidebar.title("可视化选项")
vis_option = st.sidebar.selectbox(
    "选择可视化方法", 
    ["Grad-CAM", "Guided Backpropagation", "Integrated Gradients", "Deep Dream"]
)

# 侧边栏 - 参数设置
st.sidebar.title("参数设置")
if vis_option == "Grad-CAM":
    layer_name = st.sidebar.selectbox(
        "选择层", 
        [layer.name for layer in model.layers if 'conv' in layer.name]
    )
elif vis_option == "Deep Dream":
    octave_scale = st.sidebar.slider("Octave Scale", 1.0, 2.0, 1.3)
    num_octaves = st.sidebar.slider("Octaves", 1, 5, 3)
    steps_per_octave = st.sidebar.slider("Steps per Octave", 10, 50, 20)
    
# 实现Grad-CAM
def make_gradcam_heatmap(img_array, model, layer_name, pred_index=None):
    # 创建获取指定层输出的模型
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    # 计算梯度
    with tf.GradientTape() as tape:
        layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # 获取梯度
    grads = tape.gradient(class_channel, layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 加权并激活
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, layer_output[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

# 实现Deep Dream
def run_deep_dream(img_array, model, layer_name='mixed3', octave_scale=1.3, num_octaves=3, steps_per_octave=20):
    # 创建特征提取模型
    feature_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    def calc_loss(img, feature_extractor):
        # 计算特征图的均值作为损失
        features = feature_extractor(img)
        return tf.reduce_mean(features)
    
    @tf.function
    def deepdream_step(img, feature_extractor):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = calc_loss(img, feature_extractor)
            
        # 计算梯度
        grads = tape.gradient(loss, img)
        # 标准化梯度
        grads /= tf.math.reduce_std(grads) + 1e-8
        # 更新图像
        img = img + grads * 0.01
        # 裁剪以保持合理的值
        img = tf.clip_by_value(img, -1, 1)
        return img
    
    # 转换输入图像
    img = img_array.copy()
    img = tf.convert_to_tensor(img)
    
    for octave in range(num_octaves):
        # 调整图像大小以适应当前octave
        new_size = tf.cast(tf.shape(img)[1:3] * octave_scale**(octave), tf.int32)
        img = tf.image.resize(img, new_size)
        
        # 执行梯度上升
        for step in range(steps_per_octave):
            img = deepdream_step(img, feature_extractor)
            
    img = (img + 1) / 2  # 恢复为0-1范围
    return img.numpy()

# 处理上传的图像
if uploaded_file is not None:
    # 显示原始图像
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("原始图像")
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        st.image(pil_image, use_column_width=True)
        
        # 预处理图像
        img = pil_image.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # 做出预测
        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=3)[0]
        
        # 显示预测结果
        st.subheader("预测结果:")
        for i, (imagenet_id, label, score) in enumerate(decoded_preds):
            st.write(f"{i+1}: {label} ({score:.2f})")
    
    with col2:
        st.header(f"{vis_option}可视化")
        
        # 应用选定的可视化方法
        if vis_option == "Grad-CAM":
            with st.spinner('正在生成Grad-CAM...'):
                # 获取热力图
                heatmap = make_gradcam_heatmap(img_array, model, layer_name)
                
                # 调整热力图大小以匹配原图
                heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
                
                # 将热力图转换为RGB
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # 叠加热力图到原图
                img_array_vis = image.img_to_array(img)
                superimposed_img = heatmap * 0.4 + img_array_vis
                superimposed_img = np.clip(superimposed_img / 255.0, 0, 1)
                
                # 显示热力图和叠加图
                st.image(heatmap / 255.0, caption='热力图', use_column_width=True)
                st.image

Similar code found with 3 license types