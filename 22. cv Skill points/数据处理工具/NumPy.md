# NumPy 基础教程

NumPy（Numerical Python）是Python中用于科学计算的核心库，它提供了高性能的多维数组对象和处理这些数组的工具。

## 1. NumPy 的核心：ndarray

NumPy 的核心是 `ndarray`（N-dimensional array，多维数组）对象。与Python原生列表相比，NumPy数组具有以下优势：

- 计算速度更快
- 内存使用更高效
- 提供大量的数学函数操作

### 创建数组

```python
import numpy as np

# 从列表创建数组
arr1 = np.array([1, 2, 3, 4, 5])
print("一维数组:", arr1)

# 创建二维数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("二维数组:\n", arr2)

# 创建特殊数组
zeros = np.zeros((3, 3))  # 3x3的全0数组
print("全0数组:\n", zeros)

ones = np.ones((2, 4))    # 2x4的全1数组
print("全1数组:\n", ones)

# 创建等差数列
arange = np.arange(0, 10, 2)  # 从0到10，步长为2
print("等差数列:", arange)

# 创建等间隔数列
linspace = np.linspace(0, 1, 5)  # 0到1之间均匀分布的5个点
print("等间隔数列:", linspace)

# 创建随机数组
random_arr = np.random.random((2, 2))
print("随机数组:\n", random_arr)
```

## 2. 数组基本操作

### 数组形状操作

```python
import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# 获取数组形状
print("数组形状:", arr.shape)  # (2, 4)

# 获取元素总数
print("元素总数:", arr.size)   # 8

# 获取数组维度
print("数组维度:", arr.ndim)   # 2

# 改变数组形状
reshaped = arr.reshape(4, 2)
print("重塑形状后:\n", reshaped)

# 展平数组
flattened = arr.flatten()
print("展平后:", flattened)
```

### 数组索引和切片

```python
import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 获取单个元素
print("第二行第三列的元素:", arr[1, 2])  # 7

# 获取一行
print("第二行:", arr[1])  # [5 6 7 8]

# 切片操作
print("前两行:\n", arr[:2])
print("所有行的第2和第3列:\n", arr[:, 1:3])

# 条件索引
print("大于5的元素:", arr[arr > 5])
```

## 3. 数组运算

### 算术运算

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 加法
print("a + b =", a + b)  # [5 7 9]

# 减法
print("a - b =", a - b)  # [-3 -3 -3]

# 乘法 (元素级)
print("a * b =", a * b)  # [4 10 18]

# 除法
print("a / b =", a / b)  # [0.25 0.4 0.5]

# 乘方
print("a ** 2 =", a ** 2)  # [1 4 9]

# 与标量运算
print("a + 10 =", a + 10)  # [11 12 13]
```

### 统计函数

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# 求和
print("求和:", arr.sum())  # 15

# 平均值
print("平均值:", arr.mean())  # 3.0

# 最大值
print("最大值:", arr.max())  # 5

# 最小值
print("最小值:", arr.min())  # 1

# 标准差
print("标准差:", arr.std())  # 约1.41

# 二维数组按轴计算
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print("按行求和:", arr2d.sum(axis=1))  # [6 15]
print("按列求和:", arr2d.sum(axis=0))  # [5 7 9]
```

## 4. 广播机制

NumPy的广播机制允许不同形状的数组进行运算:

```python
import numpy as np

# 创建一个3x3的二维数组
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 用一个一维数组与二维数组的每一行相加
row_vector = np.array([10, 20, 30])
print("广播加法:\n", arr + row_vector)

# 用一个列向量与二维数组的每一列相加
col_vector = np.array([[100], [200], [300]])
print("列广播加法:\n", arr + col_vector)
```

## 5. NumPy与计算机视觉的关系

在计算机视觉中，NumPy是处理图像数据的基础:

```python
import numpy as np
import matplotlib.pyplot as plt

# 创建一个简单的灰度图像 (10x10像素)
image = np.zeros((10, 10))
image[2:8, 2:8] = 1  # 在中间创建一个白色方块

# 显示图像
plt.imshow(image, cmap='gray')
plt.title('简单的方形图像')
plt.show()

# 图像旋转 (创建旋转矩阵)
def rotate_image(image, angle_degrees):
    angle = np.radians(angle_degrees)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    # 旋转矩阵
    rot_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # 获取原始图像中心点
    center = np.array(image.shape) / 2
    
    # 创建新图像
    new_image = np.zeros_like(image)
    
    # 对每个像素点应用旋转
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 计算相对于中心的位置
            pos = np.array([i, j]) - center
            # 应用旋转
            new_pos = np.dot(rot_matrix, pos) + center
            # 如果新位置在图像范围内，复制像素值
            new_i, new_j = int(new_pos[0]), int(new_pos[1])
            if 0 <= new_i < image.shape[0] and 0 <= new_j < image.shape[1]:
                new_image[i, j] = image[new_i, new_j]
    
    return new_image

# 旋转图像示例 (这是个简化版，实际应用中通常使用scipy或OpenCV)
# rotated_image = rotate_image(image, 45)
# plt.imshow(rotated_image, cmap='gray')
# plt.title('旋转后的图像')
# plt.show()
```

## 6. 性能比较

NumPy与Python原生列表的速度比较:

```python
import numpy as np
import time

# 创建数据
size = 1000000
py_list = list(range(size))
np_array = np.array(py_list)

# Python列表计算
start = time.time()
result_list = [x * 2 for x in py_list]
list_time = time.time() - start
print(f"Python列表耗时: {list_time:.6f}秒")

# NumPy数组计算
start = time.time()
result_array = np_array * 2
numpy_time = time.time() - start
print(f"NumPy数组耗时: {numpy_time:.6f}秒")
print(f"NumPy比Python列表快 {list_time/numpy_time:.1f} 倍")
```

## 总结

NumPy提供了:
1. 高效的多维数组对象
2. 快速的数组操作
3. 丰富的数学函数库
4. 在数据科学、机器学习和计算机视觉中的广泛应用

NumPy是Python科学计算的基石，掌握它是进入数据科学和人工智能领域的第一步。