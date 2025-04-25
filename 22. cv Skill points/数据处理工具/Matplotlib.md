# Matplotlib：Python数据可视化利器

Matplotlib是Python中最流行的数据可视化库，它提供了一个类似MATLAB的接口，能够创建各种高质量的图表。下面我将通过简单的例子，帮你理解Matplotlib的基本用法。

## 1. Matplotlib的基本结构

Matplotlib主要由两部分组成：
- **pyplot**：提供类似MATLAB的绘图函数接口
- **面向对象的API**：提供更细粒度的控制

```python
# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
```

## 2. 创建简单的折线图

```python
# 创建一些数据
x = np.linspace(0, 10, 100)  # 生成0到10之间的100个点
y = np.sin(x)                # 计算每个点的正弦值

# 创建一个图形
plt.figure(figsize=(8, 4))   # 设置图形大小，单位为英寸

# 绘制图形
plt.plot(x, y, 'b-', label='sin(x)')  # 'b-'表示蓝色实线

# 添加标题和标签
plt.title('正弦函数')
plt.xlabel('x值')
plt.ylabel('y值')
plt.legend()  # 显示图例

# 显示网格
plt.grid(True)

# 显示图形
plt.show()
```

这段代码会生成一个蓝色的正弦函数图，并包含标题、坐标轴标签和图例。

## 3. 多图绘制

```python
# 创建数据
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建一个包含两个子图的图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# 在第一个子图中绘制sin函数
ax1.plot(x, y1, 'r-')
ax1.set_title('正弦函数')
ax1.set_ylabel('sin(x)')
ax1.grid(True)

# 在第二个子图中绘制cos函数
ax2.plot(x, y2, 'g-')
ax2.set_title('余弦函数')
ax2.set_xlabel('x')
ax2.set_ylabel('cos(x)')
ax2.grid(True)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
```

这段代码创建了两个子图，分别显示正弦和余弦函数。

## 4. 常见图表类型

### 4.1 散点图

```python
# 创建随机数据
np.random.seed(42)  # 设定随机种子，确保结果可重现
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

# 创建散点图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)
plt.title('散点图示例')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.colorbar()  # 添加颜色条
plt.grid(True)
plt.show()
```

这段代码创建了一个散点图，点的颜色和大小都是随机的。

### 4.2 柱状图

```python
# 创建数据
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 55, 15]

# 创建柱状图
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color='skyblue')
plt.title('柱状图示例')
plt.xlabel('类别')
plt.ylabel('值')
plt.grid(True, axis='y')  # 只在y轴显示网格线
plt.show()
```

### 4.3 饼图

```python
# 创建数据
labels = ['苹果', '香蕉', '橙子', '草莓', '葡萄']
sizes = [15, 30, 25, 10, 20]
explode = (0, 0.1, 0, 0, 0)  # 突出显示第二个切片（香蕉）

# 创建饼图
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')  # 确保饼图是圆形的
plt.title('水果比例')
plt.show()
```

## 5. 自定义图表样式

Matplotlib提供了多种样式设置来美化图表：

```python
# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')  # 使用Seaborn的暗网格样式

# 创建数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制线条，自定义样式
plt.plot(x, y1, 'r-', linewidth=2, label='sin(x)')
plt.plot(x, y2, 'b--', linewidth=2, label='cos(x)')

# 添加标题和标签，自定义字体
plt.title('三角函数', fontsize=16)
plt.xlabel('x值', fontsize=12)
plt.ylabel('y值', fontsize=12)

# 添加图例
plt.legend(fontsize=12)

# 设置刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 在图中添加文本
plt.text(5, 0.5, '重要点', fontsize=12, 
         bbox=dict(facecolor='yellow', alpha=0.5))

# 显示图形
plt.show()
```

## 6. 保存图表

```python
# 创建一个简单的图表
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro-')
plt.title('平方关系')

# 保存图表到文件
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
# 也可以保存为其他格式，如pdf, svg, jpg等
# plt.savefig('my_plot.pdf', bbox_inches='tight')
```

## 7. 结合Pandas使用Matplotlib

Matplotlib经常与Pandas一起使用，可以直接从DataFrame绘制图表：

```python
import pandas as pd

# 创建一个简单的DataFrame
data = {
    '年份': [2010, 2011, 2012, 2013, 2014, 2015],
    '销量': [120, 150, 130, 180, 210, 240],
    '利润': [20, 25, 22, 30, 35, 40]
}
df = pd.DataFrame(data)

# 使用pandas内置的绘图功能（基于matplotlib）
ax = df.plot(x='年份', y=['销量', '利润'], figsize=(10, 6), 
             style=['o-', 's--'], secondary_y='利润')
ax.set_xlabel('年份')
ax.set_ylabel('销量')
ax.right_ax.set_ylabel('利润')
ax.set_title('销量和利润随时间的变化')
ax.grid(True)
plt.show()
```

## 8. 实际应用：数据分析可视化

```python
# 模拟一个真实的数据分析场景
np.random.seed(42)
# 假设这是某电商平台一周的销售数据
days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
sales = np.random.normal(1000, 200, 7)  # 平均1000，标准差200的正态分布
visitors = sales * 5 + np.random.normal(0, 200, 7)  # 访客数约为销售的5倍

# 创建一个包含两个子图的图形：柱状图和折线图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：销售额柱状图
ax1.bar(days, sales, color='skyblue')
ax1.set_title('每日销售额')
ax1.set_xlabel('星期')
ax1.set_ylabel('销售额')
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# 子图2：销售额与访客数的关系散点图
ax2.scatter(visitors, sales, alpha=0.7, s=100)
ax2.set_title('访客数与销售额的关系')
ax2.set_xlabel('访客数')
ax2.set_ylabel('销售额')
ax2.grid(True, linestyle='--', alpha=0.7)

# 添加趋势线（最小二乘法拟合）
z = np.polyfit(visitors, sales, 1)
p = np.poly1d(z)
x_line = np.linspace(min(visitors), max(visitors), 100)
ax2.plot(x_line, p(x_line), 'r--', linewidth=2)

# 在图中添加相关系数
from scipy.stats import pearsonr
corr, _ = pearsonr(visitors, sales)
ax2.text(min(visitors) + 50, max(sales) - 100, f'相关系数: {corr:.2f}', 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# 调整布局
plt.tight_layout()
plt.show()
```

## 总结

Matplotlib是一个功能强大且灵活的可视化库，通过本文的示例，你应该已经了解了：

1. 如何创建基本的图表（折线图、散点图、柱状图、饼图）
2. 如何自定义图表的外观（颜色、样式、标签等）
3. 如何创建子图和复合图表
4. 如何结合Pandas进行数据可视化
5. 如何应用到实际数据分析场景

掌握这些基础知识后，你可以探索更高级的功能，如3D图表、等高线图、热图等。Matplotlib的官方文档提供了详尽的资源和更多示例。