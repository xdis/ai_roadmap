# JAX 基础介绍

JAX 是由 Google 开发的高性能数值计算库，它结合了 NumPy 的易用性和自动微分（AutoGrad）的强大功能，同时能够在 GPU 和 TPU 上高效运行。JAX 的设计理念是"NumPy on steroids"，让你可以轻松编写能够自动微分和加速的代码。

## JAX 的核心特性

1. **NumPy 风格的 API**：如果你熟悉 NumPy，上手 JAX 会很容易
2. **自动微分**：轻松计算任何函数的导数
3. **即时编译 (JIT)**：加速代码执行
4. **GPU/TPU 加速**：无需修改代码即可利用硬件加速
5. **函数式编程设计**：强调纯函数，更容易并行化和优化

## 基础示例

### 安装 JAX

```python
# CPU 版本
pip install jax

# GPU 版本 (需要 CUDA)
pip install jax[cuda]
```

### 基本用法 - NumPy 风格操作

```python
import jax
import jax.numpy as jnp

# 创建数组 (类似 NumPy)
x = jnp.array([1, 2, 3, 4])
y = jnp.ones((3, 3))

# 矩阵运算
z = jnp.dot(jnp.ones((2, 3)), jnp.ones((3, 2)))
print(z)  # 输出 3.0 的 2x2 矩阵

# JAX 数组和 NumPy 数组的主要区别是 JAX 数组是不可变的
# 这意味着操作不会改变原始数组，而是返回新数组
```

### 自动微分 - 求导

JAX 的自动微分功能使计算梯度变得简单：

```python
from jax import grad

# 定义一个函数
def f(x):
    return x ** 2  # 平方函数

# 计算导数 df/dx = 2x
df_dx = grad(f)

# 计算 x=3 处的导数
result = df_dx(3.0)
print(result)  # 输出: 6.0

# 高阶导数也很容易计算
d2f_dx2 = grad(grad(f))
print(d2f_dx2(3.0))  # 输出: 2.0 (平方函数的二阶导数是常数2)
```

### 使用 JIT 加速计算

JIT（Just-In-Time）编译可以显著加速你的代码：

```python
from jax import jit
import time

# 定义一个计算量大的函数
def slow_function(x):
    # 做一些重复计算
    for _ in range(1000):
        x = x + x * 0.1
    return x

# 创建 JIT 版本
fast_function = jit(slow_function)

# 准备输入数据
x = jnp.ones((1000, 1000))

# 比较性能
start = time.time()
result_slow = slow_function(x)
print(f"普通函数耗时: {time.time() - start:.4f}秒")

# 第一次调用 JIT 函数会编译，可能会慢一点
_ = fast_function(x)

# 测量编译后的性能
start = time.time()
result_fast = fast_function(x)
print(f"JIT加速后耗时: {time.time() - start:.4f}秒")
```

### 批量操作 - vmap

使用 `vmap` 可以高效地对批量数据进行操作：

```python
from jax import vmap

# 定义一个函数计算向量的平方和
def squared_norm(x):
    return jnp.sum(x ** 2)

# 创建一个批处理版本，对一批向量计算平方和
batch_squared_norm = vmap(squared_norm)

# 创建一批数据: 10个3维向量
batch_vectors = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# 一次计算所有向量的平方和
results = batch_squared_norm(batch_vectors)
print(results)  # 输出: [14, 77, 194, 365]
```

## 简单的神经网络示例

用 JAX 实现一个简单的神经网络：

```python
import jax
import jax.numpy as jnp
from jax import random, grad, jit

# 设置随机数种子
key = random.PRNGKey(42)

# 生成一些随机数据
n_samples = 100
x_data = random.normal(key, (n_samples, 1))
y_data = 3 * x_data + 2 + 0.2 * random.normal(key, (n_samples, 1))

# 定义模型参数
def init_params():
    w = jnp.array(0.0)
    b = jnp.array(0.0)
    return w, b

# 定义预测函数
def predict(params, x):
    w, b = params
    return w * x + b

# 定义损失函数 (均方误差)
def loss_fn(params, x, y):
    predictions = predict(params, x)
    return jnp.mean((predictions - y) ** 2)

# 使用 JAX 的 grad 自动计算梯度
grad_fn = jit(grad(loss_fn))  # 对参数的梯度，并使用 JIT 加速

# 简单的梯度下降训练
def train(learning_rate=0.1, n_epochs=100):
    params = init_params()
    
    for epoch in range(n_epochs):
        # 计算梯度
        grads = grad_fn(params, x_data, y_data)
        
        # 更新参数 (注意 JAX 中的不可变性)
        w, b = params
        w = w - learning_rate * grads[0]
        b = b - learning_rate * grads[1]
        params = (w, b)
        
        # 打印进度
        if epoch % 10 == 0:
            current_loss = loss_fn(params, x_data, y_data)
            print(f"Epoch {epoch}, Loss: {current_loss:.4f}, w: {w:.4f}, b: {b:.4f}")
    
    return params

# 训练模型
trained_params = train()
print(f"训练结果 - w: {trained_params[0]:.4f}, b: {trained_params[1]:.4f}")
print(f"预期结果 - w: 3.0000, b: 2.0000")
```

## JAX 与其他框架的对比

| 特性 | JAX | PyTorch | TensorFlow |
|------|-----|---------|------------|
| 编程模型 | 函数式 | 命令式/函数式 | 命令式/声明式 |
| 自动微分 | 是 | 是 | 是 |
| GPU/TPU 支持 | 是 | 是 | 是 |
| 易用性 | NumPy 风格，简洁 | 灵活，动态 | 完整生态系统 |
| 不可变性 | 是 | 否 | 否 |
| 主要用途 | 研究，特别是复杂数学计算 | 研究和生产 | 研究和生产 |
| 生态系统 | 发展中 | 丰富 | 丰富 |

## JAX 的适用场景

JAX 特别适合:
- 需要高性能数值计算的科学研究
- 复杂的梯度计算和优化问题
- 机器学习研究，特别是需要自定义算法
- 对速度和并行计算有高需求的场景

## 总结

JAX 是一个强大的数值计算库，它结合了 NumPy 的熟悉语法和现代硬件加速能力。其函数式编程方法和自动微分功能使它成为机器学习研究和科学计算的优秀工具。对于需要高性能计算和自定义算法的场景，JAX 是一个值得考虑的选择。