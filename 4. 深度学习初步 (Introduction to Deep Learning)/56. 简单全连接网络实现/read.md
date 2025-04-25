# 简单全连接网络实现

## 1. 全连接网络基础概念

全连接网络(Fully Connected Network，简称FCN)，又称多层感知机(Multi-Layer Perceptron，MLP)，是深度学习中最基本的神经网络结构。在全连接网络中，一层的每个神经元与下一层的所有神经元相连，形成了"全连接"的结构。

### 1.1 全连接网络的基本结构

全连接网络通常包含以下组成部分：

1. **输入层**：接收原始数据，每个节点表示一个输入特征
2. **隐藏层**：一个或多个中间层，执行特征转换和提取
3. **输出层**：产生最终预测结果
4. **连接权重**：层与层之间的连接强度
5. **偏置项**：增加模型的灵活性
6. **激活函数**：引入非线性变换，增强网络表达能力

![全连接网络结构图](https://example.com/fully_connected_network.png)

### 1.2 全连接层的数学表示

全连接层的前向传播可以表示为：

$$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

其中：
- $Z^{[l]}$ 是第 $l$ 层的加权输入
- $W^{[l]}$ 是第 $l$ 层的权重矩阵
- $A^{[l-1]}$ 是第 $l-1$ 层的激活输出
- $b^{[l]}$ 是第 $l$ 层的偏置向量
- $g^{[l]}$ 是第 $l$ 层的激活函数
- $A^{[l]}$ 是第 $l$ 层的激活输出

### 1.3 全连接网络的主要特点

1. **模型简单直观**：结构清晰，易于理解和实现
2. **参数量大**：每个连接都有一个权重参数，参数量随网络规模呈二次方增长
3. **计算复杂度高**：由于参数量大，计算和存储需求较高
4. **易于过拟合**：参数众多使模型容易记住训练数据的噪声
5. **特征提取能力有限**：没有考虑数据的结构信息（如图像的空间信息）
6. **通用性强**：可以应用于多种类型的数据和任务

## 2. 手动实现全连接网络

### 2.1 从零实现单层神经网络

以下是使用NumPy实现一个简单单层神经网络的例子：

```python
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, output_size):
        # 初始化权重和偏置
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.b = np.zeros((output_size, 1))
        
    def sigmoid(self, z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Sigmoid函数的导数"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        """前向传播"""
        # 计算加权和
        z = np.dot(self.W, X) + self.b
        # 应用激活函数
        a = self.sigmoid(z)
        return a, z
    
    def compute_cost(self, A, Y):
        """计算二元交叉熵损失"""
        m = Y.shape[1]  # 样本数量
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return cost
    
    def backward(self, X, Y, A):
        """反向传播计算梯度"""
        m = X.shape[1]  # 样本数量
        
        # 计算输出层误差
        dZ = A - Y
        
        # 计算权重和偏置的梯度
        dW = 1/m * np.dot(dZ, X.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        
        return dW, db
    
    def update_parameters(self, dW, db, learning_rate):
        """更新参数"""
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
    
    def train(self, X, Y, learning_rate=0.1, num_iterations=1000):
        """训练模型"""
        costs = []
        
        for i in range(num_iterations):
            # 前向传播
            A, _ = self.forward(X)
            
            # 计算代价
            cost = self.compute_cost(A, Y)
            
            # 反向传播
            dW, db = self.backward(X, Y, A)
            
            # 更新参数
            self.update_parameters(dW, db, learning_rate)
            
            # 记录代价
            if i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost}")
        
        return costs
    
    def predict(self, X, threshold=0.5):
        """预测"""
        A, _ = self.forward(X)
        return (A > threshold).astype(int)

# 使用示例
if __name__ == "__main__":
    # 生成简单的XOR数据
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # 2x4，两个特征，四个样本
    Y = np.array([[0, 1, 1, 0]])  # 1x4，一个输出，四个样本
    
    # 创建网络并训练
    # 注意：单层网络无法学习XOR问题，这里只是示例
    nn = SimpleNeuralNetwork(input_size=2, output_size=1)
    costs = nn.train(X, Y, learning_rate=0.1, num_iterations=1000)
    
    # 预测
    predictions = nn.predict(X)
    print("Predictions:", predictions)
    print("Accuracy:", np.mean(predictions == Y))
```

### 2.2 实现多层全连接网络

拓展上述实现为多层网络：

```python
import numpy as np

class DeepNeuralNetwork:
    def __init__(self, layer_dims):
        """
        初始化L层神经网络
        
        参数：
        layer_dims -- 包含每层单元数量的列表 [n_x, n_h1, n_h2, ..., n_y]
        """
        self.parameters = {}
        self.L = len(layer_dims) - 1  # 层数（输入层不计算在内）
        
        # 初始化参数
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    def sigmoid(self, Z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-Z))
    
    def relu(self, Z):
        """ReLU激活函数"""
        return np.maximum(0, Z)
    
    def sigmoid_backward(self, dA, Z):
        """Sigmoid函数的导数"""
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)
    
    def relu_backward(self, dA, Z):
        """ReLU函数的导数"""
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    
    def forward_propagation(self, X):
        """
        前向传播
        
        参数：
        X -- 输入数据，形状为 (输入大小, 样本数)
        
        返回：
        AL -- 最后一层的激活值
        caches -- 包含每层"线性激活缓存"的列表
        """
        caches = []
        A = X
        
        # 计算L-1层，使用ReLU
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            
            # 线性前向传播
            Z = np.dot(W, A_prev) + b
            
            # 激活函数
            A = self.relu(Z)
            
            # 保存缓存
            cache = (A_prev, W, b, Z)
            caches.append(cache)
        
        # 计算第L层，使用Sigmoid
        A_prev = A
        W = self.parameters['W' + str(self.L)]
        b = self.parameters['b' + str(self.L)]
        
        Z = np.dot(W, A_prev) + b
        A = self.sigmoid(Z)
        
        cache = (A_prev, W, b, Z)
        caches.append(cache)
        
        return A, caches
    
    def compute_cost(self, AL, Y):
        """
        计算成本函数
        
        参数：
        AL -- 预测概率向量，形状为 (1, 样本数)
        Y -- 真实标签向量，形状为 (1, 样本数)
        
        返回：
        cost -- 交叉熵成本
        """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        cost = np.squeeze(cost)
        
        return cost
    
    def backward_propagation(self, AL, Y, caches):
        """
        反向传播
        
        参数：
        AL -- 预测概率向量，形状为 (1, 样本数)
        Y -- 真实标签向量，形状为 (1, 样本数)
        caches -- 前向传播中保存的缓存列表
        
        返回：
        grads -- 包含梯度的字典
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # 确保Y与AL形状相同
        
        # 初始化反向传播
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        # 获取最后一层的缓存
        current_cache = caches[self.L-1]
        A_prev, W, b, Z = current_cache
        
        # 对最后一层使用Sigmoid反向传播
        dZ = self.sigmoid_backward(dAL, Z)
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        grads["dW" + str(self.L)] = dW
        grads["db" + str(self.L)] = db
        
        # 对1到L-1层使用ReLU反向传播
        for l in reversed(range(1, self.L)):
            current_cache = caches[l-1]
            A_prev, W, b, Z = current_cache
            
            dZ = self.relu_backward(dA_prev, Z)
            dW = 1/m * np.dot(dZ, A_prev.T)
            db = 1/m * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)
            
            grads["dW" + str(l)] = dW
            grads["db" + str(l)] = db
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        """
        使用梯度下降更新参数
        
        参数：
        grads -- 包含梯度的字典
        learning_rate -- 学习率
        """
        for l in range(1, self.L + 1):
            self.parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    
    def train(self, X, Y, learning_rate=0.01, num_iterations=3000, print_cost=True):
        """
        训练L层神经网络
        
        参数：
        X -- 训练数据，形状为 (n_x, 样本数)
        Y -- 真实标签，形状为 (1, 样本数)
        learning_rate -- 学习率
        num_iterations -- 迭代次数
        print_cost -- 是否打印成本
        
        返回：
        costs -- 成本列表
        """
        costs = []
        
        for i in range(num_iterations):
            # 前向传播
            AL, caches = self.forward_propagation(X)
            
            # 计算成本
            cost = self.compute_cost(AL, Y)
            
            # 反向传播
            grads = self.backward_propagation(AL, Y, caches)
            
            # 更新参数
            self.update_parameters(grads, learning_rate)
            
            # 打印成本
            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
                costs.append(cost)
        
        return costs
    
    def predict(self, X):
        """
        使用训练好的神经网络进行预测
        
        参数：
        X -- 输入数据，形状为 (n_x, 样本数)
        
        返回：
        predictions -- 预测向量，形状为 (1, 样本数)
        """
        AL, _ = self.forward_propagation(X)
        predictions = (AL > 0.5).astype(int)
        return predictions

# 使用示例
if __name__ == "__main__":
    # 生成简单的XOR数据
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # 2x4
    Y = np.array([[0, 1, 1, 0]])  # 1x4
    
    # 创建神经网络 [2, 4, 1] - 2个输入，4个隐藏单元，1个输出
    nn = DeepNeuralNetwork([2, 4, 1])
    costs = nn.train(X, Y, learning_rate=0.1, num_iterations=10000, print_cost=True)
    
    # 预测
    predictions = nn.predict(X)
    print("Predictions:", predictions)
    print("Accuracy:", np.mean(predictions == Y))
```

## 3. 使用PyTorch实现全连接网络

### 3.1 PyTorch中的全连接层：nn.Linear

PyTorch提供了`nn.Linear`模块来实现全连接层：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # 第一个全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 第二个全连接层
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 输出激活函数
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 前向传播
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# 使用PyTorch处理XOR问题
def train_xor_network():
    # 准备XOR数据
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    # 创建模型
    model = SimpleNN(input_size=2, hidden_size=4, output_size=1)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 训练模型
    num_epochs = 10000
    losses = []
    
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, Y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        if (epoch+1) % 100 == 0:
            losses.append(loss.item())
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 测试模型
    with torch.no_grad():
        test_outputs = model(X)
        predicted = (test_outputs >= 0.5).float()
        accuracy = (predicted == Y).sum().item() / Y.size(0)
        print(f'Accuracy: {accuracy:.4f}')
        
        print("Input -> Output")
        for i in range(X.shape[0]):
            print(f"{X[i].numpy()} -> {test_outputs[i].item():.4f} -> {predicted[i].item()}")
    
    # 可视化损失
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch (x100)')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.show()
    
    return model

if __name__ == "__main__":
    trained_model = train_xor_network()
```

### 3.2 实现一个多层全连接神经网络解决MNIST分类问题

下面是一个使用PyTorch实现的多层全连接网络来解决MNIST手写数字分类问题：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义超参数
input_size = 784  # 28x28
hidden_sizes = [500, 300, 100]
output_size = 10
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 定义多层全连接神经网络
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MultiLayerPerceptron, self).__init__()
        
        # 创建层列表
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 添加其他隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 添加输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # 创建顺序容器
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # 输入数据形状调整：[batch_size, 1, 28, 28] -> [batch_size, 784]
        x = x.view(-1, 28 * 28)
        return self.model(x)

# 初始化模型
model = MultiLayerPerceptron(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    output_size=output_size
).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
def train_model():
    train_losses = []
    train_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                     f'Loss: {loss.item():.4f}')
        
        # 每个epoch结束后计算训练损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Training Loss: {epoch_loss:.4f}, '
              f'Training Accuracy: {epoch_acc:.4f}')
        
        # 在测试集上评估
        test_acc = evaluate_model()
        print(f'Test Accuracy: {test_acc:.4f}')
    
    return train_losses, train_accs

# 评估模型
def evaluate_model():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy

# 可视化一些结果
def visualize_results(test_images, actual_labels, predicted_labels, num_samples=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(test_images[i].numpy().reshape(28, 28), cmap='gray')
        plt.title(f'Actual: {actual_labels[i]}\nPredicted: {predicted_labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 训练模型并获取损失和准确率
train_losses, train_accs = train_model()

# 可视化训练过程
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accs)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.tight_layout()
plt.show()

# 在测试集上获取一些预测结果进行可视化
model.eval()
test_images = []
actual_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # 收集一些样本用于可视化
        if len(test_images) < 5:
            test_images.extend(images.cpu()[:5 - len(test_images)])
            actual_labels.extend(labels.cpu().numpy()[:5 - len(actual_labels)])
            predicted_labels.extend(predicted.cpu().numpy()[:5 - len(predicted_labels)])
        else:
            break

# 可视化结果
visualize_results(test_images, actual_labels, predicted_labels)
```

## 4. 全连接网络的应用与局限性

### 4.1 全连接网络的典型应用场景

全连接网络非常适合以下应用场景：

1. **表格数据分类与回归**：金融数据预测、医疗诊断等
2. **特征提取后的分类任务**：作为其他网络（如CNN）的最后几层
3. **低维数据处理**：处理较小特征空间的数据
4. **简单模式识别**：基础形状识别、简单决策系统
5. **其他网络的组件**：作为更复杂网络架构的一部分

### 4.2 全连接网络的局限性

全连接网络存在以下局限性：

1. **参数量大**：随着输入维度增加，参数量呈二次方增长
2. **计算复杂度高**：需要大量矩阵乘法运算
3. **无法利用数据结构**：不能有效处理图像、序列等结构化数据
4. **易过拟合**：参数多导致容易记住训练数据的噪声
5. **缺乏特征共享**：每个神经元只关注其专有权重
6. **缺乏位置不变性**：无法识别位置变化的相同特征

### 4.3 从全连接网络到专用架构的演变

正是由于全连接网络的局限性，深度学习领域发展出了各种专用架构：

1. **卷积神经网络（CNN）**：引入了局部连接和参数共享，专为图像处理设计
2. **循环神经网络（RNN）**：引入了序列处理能力，适合时间序列和文本数据
3. **Transformer**：完全依赖注意力机制，解决长程依赖问题
4. **图神经网络（GNN）**：专为图结构数据设计，捕捉节点间关系

然而，尽管有这些专用架构，全连接网络仍是深度学习的基础组件，几乎所有复杂网络的最后几层都是全连接层，用于将提取的特征映射到最终输出。

## 5. 高级全连接网络技巧与变体

### 5.1 残差连接在全连接网络中的应用

残差连接可以帮助解决深层全连接网络的梯度消失问题：

```python
class ResidualMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualMLP, self).__init__()
        
        # 第一个全连接块
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        
        # 第二个全连接块（残差路径）
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        # 保存输入作为残差连接
        identity = x
        
        # 主路径
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        # 添加残差连接（如果输入和输出维度不同，需要先进行投影）
        if identity.shape[1] != out.shape[1]:
            identity = nn.Linear(identity.shape[1], out.shape[1]).to(identity.device)(identity)
        
        out += identity
        out = self.relu2(out)
        
        return out
```

### 5.2 自调整架构：Dynamic Layer Width

动态层宽技术允许网络在训练过程中调整隐藏层的宽度：

```python
class DynamicWidthMLP(nn.Module):
    def __init__(self, input_size, max_hidden_size, output_size, sparsity=0.5):
        super(DynamicWidthMLP, self).__init__()
        
        self.input_size = input_size
        self.max_hidden_size = max_hidden_size
        self.output_size = output_size
        self.sparsity = sparsity
        
        # 创建过大的隐藏层
        self.fc1 = nn.Linear(input_size, max_hidden_size)
        self.fc2 = nn.Linear(max_hidden_size, output_size)
        
        # 创建门控机制
        self.gates = nn.Parameter(torch.ones(max_hidden_size))
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
    
    def update_gates(self):
        # 应用软门控：根据权重大小选择性地激活神经元
        with torch.no_grad():
            # 根据第一层权重的L2范数对门控值进行排序
            weight_norms = torch.norm(self.fc1.weight, dim=1)
            _, indices = torch.sort(weight_norms, descending=True)
            
            # 保留前(1-sparsity)的神经元，其余置为0
            keep_num = int(self.max_hidden_size * (1 - self.sparsity))
            mask = torch.zeros_like(self.gates)
            mask[indices[:keep_num]] = 1.0
            
            # 更新门控值
            self.gates.data = mask
    
    def forward(self, x):
        # 前向传播时使用门控机制
        x = self.fc1(x)
        x = x * self.gates.unsqueeze(0)  # 应用门控
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

### 5.3 混合专家模型（Mixture of Experts）

混合专家模型组合多个全连接子网络，每个专门处理不同类型的输入：

```python
class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_experts=3):
        super(MixtureOfExperts, self).__init__()
        
        # 门控网络 - 决定使用哪个专家
        self.gate = nn.Sequential(
            nn.Linear(input_size, num_experts),
            nn.Softmax(dim=1)
        )
        
        # 创建多个专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # 确定专家权重
        expert_weights = self.gate(x)
        
        # 获取每个专家的输出
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_size]
        
        # 按专家权重加权组合
        combined_output = torch.bmm(
            expert_weights.unsqueeze(1),  # [batch_size, 1, num_experts]
            expert_outputs  # [batch_size, num_experts, output_size]
        ).squeeze(1)  # [batch_size, output_size]
        
        return combined_output
```

### 5.4 渐进式全连接网络

渐进式增长技术可以从小网络开始，逐渐增加网络容量：

```python
class ProgressiveGrowingMLP(nn.Module):
    def __init__(self, input_size, initial_hidden_size, output_size, growth_factor=2):
        super(ProgressiveGrowingMLP, self).__init__()
        
        self.input_size = input_size
        self.current_hidden_size = initial_hidden_size
        self.output_size = output_size
        self.growth_factor = growth_factor
        
        # 初始网络
        self.input_layer = nn.Linear(input_size, self.current_hidden_size)
        self.hidden_layers = nn.ModuleList([])
        self.output_layer = nn.Linear(self.current_hidden_size, output_size)
        
        self.activation = nn.ReLU()
    
    def grow_network(self):
        """增加网络容量"""
        new_hidden_size = int(self.current_hidden_size * self.growth_factor)
        
        # 创建新的输出层
        new_output_layer = nn.Linear(new_hidden_size, self.output_size)
        
        # 传递旧输出层的权重到新层
        with torch.no_grad():
            new_output_layer.weight[:, :self.current_hidden_size] = self.output_layer.weight
            new_output_layer.bias = nn.Parameter(self.output_layer.bias.clone())
        
        # 添加新的隐藏层
        new_hidden_layer = nn.Linear(self.current_hidden_size, new_hidden_size)
        self.hidden_layers.append(new_hidden_layer)
        
        # 更新输出层和当前隐藏层大小
        self.output_layer = new_output_layer
        self.current_hidden_size = new_hidden_size
        
        print(f"Network grown: Hidden size now {self.current_hidden_size}")
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        
        x = self.output_layer(x)
        return x
```

## 6. 总结与最佳实践

### 6.1 全连接网络设计的最佳实践

1. **层数与宽度选择**：
   - 从小型网络开始，根据验证性能逐步增加复杂度
   - 通常隐藏层宽度设置为输入与输出维度的几何平均值附近
   - 隐藏层宽度通常遵循锥形或漏斗形（逐层减小）

2. **激活函数选择**：
   - 隐藏层通常使用ReLU或其变体（LeakyReLU, GELU等）
   - 输出层根据任务选择：分类用Softmax/Sigmoid，回归用线性函数

3. **正则化技术**：
   - 使用权重衰减（L2正则化）控制过拟合
   - 合理设置Dropout率（通常0.2-0.5）
   - 考虑批量归一化提高训练稳定性

4. **权重初始化**：
   - 使用He初始化（ReLU激活）或Xavier初始化（Sigmoid/Tanh激活）
   - 避免对称权重破坏网络多样性

5. **学习率设置**：
   - 使用学习率调度器（如学习率退火）
   - 考虑自适应优化器（Adam, AdamW等）

### 6.2 解决全连接网络常见问题

1. **处理过拟合**：
   - 增加训练数据或使用数据增强
   - 添加Dropout或权重约束
   - 使用提前停止策略

2. **处理梯度消失/爆炸**：
   - 使用批量归一化层
   - 尝试残差连接
   - 监控并裁剪梯度

3. **优化大型全连接网络**：
   - 使用低精度计算（如FP16）
   - 考虑模型剪枝和压缩
   - 利用稀疏矩阵计算

### 6.3 全连接网络与其他架构的结合

1. **CNN+FC**：卷积层提取特征，全连接层进行分类
2. **RNN+FC**：循环层处理序列，全连接层进行转换
3. **Transformer+FC**：注意力机制处理关系，全连接层进行投影
4. **GNN+FC**：图卷积层处理图结构，全连接层聚合特征

### 6.4 未来发展趋势

1. **动态架构**：根据输入自动调整网络结构
2. **神经架构搜索（NAS）**：自动寻找最优全连接层配置
3. **知识蒸馏**：将大型全连接网络的知识转移到小型网络
4. **稀疏全连接**：降低参数密度但保持表示能力
5. **量子全连接网络**：利用量子计算加速全连接层计算

## 7. 参考文献与资源

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition.
3. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting.
4. PyTorch Documentation: https://pytorch.org/docs/stable/nn.html#linear-layers
5. TensorFlow Documentation: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense