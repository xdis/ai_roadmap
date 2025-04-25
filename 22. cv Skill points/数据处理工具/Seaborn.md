# Seaborn: 统计数据可视化库

Seaborn 是基于 Matplotlib 的 Python 数据可视化库，专注于统计数据的可视化表示。它提供了高级接口来绘制有吸引力且信息丰富的统计图形，同时有合理的默认设置和美观的主题风格。

## 1. Seaborn 的主要特点

- **美观的默认风格和主题**: 默认生成视觉吸引力强的图表
- **基于 Pandas 数据结构**: 与 Pandas DataFrames 无缝集成
- **内置统计分析功能**: 可以直接在绘图函数中进行统计分析
- **多变量数据可视化**: 轻松展示多个变量之间的关系
- **分类数据的特殊处理**: 提供专门的绘图函数处理分类数据

## 2. 安装和导入

```python
# 安装 Seaborn
pip install seaborn

# 导入常用库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set_theme()  # 使用默认主题
```

## 3. 基础绘图示例

### 3.1 散点图 (Scatter Plot)

```python
# 加载内置数据集
tips = sns.load_dataset("tips")

# 创建散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x="total_bill", y="tip", data=tips, hue="time")
plt.title("消费金额与小费关系")
plt.show()
```

这段代码绘制了一个散点图，展示了账单总额与小费的关系，并用不同颜色区分了就餐时间（午餐/晚餐）。

### 3.2 箱线图 (Box Plot)

```python
plt.figure(figsize=(10, 6))
sns.boxplot(x="day", y="total_bill", data=tips)
plt.title("不同日期的消费金额分布")
plt.show()
```

箱线图显示了数据的分布情况，包括中位数、四分位数和异常值，适合比较不同类别的数值分布。

### 3.3 直方图和密度图 (Histogram & KDE)

```python
plt.figure(figsize=(10, 6))
sns.histplot(tips["total_bill"], kde=True, bins=20)
plt.title("账单金额分布")
plt.xlabel("账单金额")
plt.ylabel("频次")
plt.show()
```

直方图显示数值分布，kde=True 添加核密度估计曲线，更平滑地展示分布趋势。

## 4. 高级绘图功能

### 4.1 成对关系图 (Pairplot)

```python
# 加载内置数据集
iris = sns.load_dataset("iris")

# 创建成对关系图
sns.pairplot(iris, hue="species")
plt.suptitle("鸢尾花数据集各特征间的关系", y=1.02)
plt.show()
```

成对关系图展示数据集中所有数值变量之间的两两关系，对角线上是各变量的分布，非常适合数据探索。

### 4.2 热力图 (Heatmap)

```python
# 计算相关系数矩阵
corr = iris.drop("species", axis=1).corr()

# 创建热力图
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("特征相关性热力图")
plt.show()
```

热力图很适合展示相关性矩阵，使用颜色深浅表示变量间关系的强弱，annot=True 显示具体数值。

### 4.3 分类散点图 (Categorical Scatter)

```python
plt.figure(figsize=(10, 6))
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
plt.title("各天消费分布")
plt.show()
```

分类散点图展示了离散类别与连续变量的关系，jitter=True 添加随机抖动避免点重叠。

### 4.4 小提琴图 (Violin Plot)

```python
plt.figure(figsize=(10, 6))
sns.violinplot(x="day", y="total_bill", data=tips, hue="sex", split=True)
plt.title("不同性别在各天的消费分布")
plt.show()
```

小提琴图结合了箱线图与密度图的特点，显示数据分布形状，split=True 可以对比同一类别中不同群体的分布。

## 5. 结合 Pandas 进行数据分析

```python
# 加载数据
titanic = sns.load_dataset("titanic")

# 数据预处理
titanic_survived = titanic.groupby(["sex", "class"])["survived"].mean().reset_index()

# 创建热图
plt.figure(figsize=(10, 6))
titanic_pivot = titanic_survived.pivot(index="class", columns="sex", values="survived")
sns.heatmap(titanic_pivot, annot=True, cmap="YlGnBu", fmt=".2%")
plt.title("泰坦尼克号生存率 (按性别和舱位)")
plt.show()
```

这个例子展示了如何结合 Pandas 进行数据处理，然后用 Seaborn 可视化结果，分析泰坦尼克号乘客的生存率。

## 6. 图表美化与定制

```python
# 设置主题样式
sns.set_theme(style="whitegrid", palette="pastel")

# 创建图表
plt.figure(figsize=(12, 6))
sns.barplot(x="day", y="total_bill", data=tips, hue="sex", errorbar=None)

# 自定义图表
plt.title("每日平均消费金额", fontsize=16)
plt.xlabel("星期", fontsize=12)
plt.ylabel("平均消费额", fontsize=12)
plt.legend(title="性别")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

这段代码展示了如何设置主题样式、调整图表尺寸和自定义标题、标签等元素。

## 7. 实用技巧

### 7.1 FacetGrid (多面网格)

```python
# 创建多面网格图
g = sns.FacetGrid(tips, col="time", row="sex", height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
g.add_legend()
g.fig.suptitle("不同时间和性别的消费与小费关系", y=1.05)
plt.show()
```

FacetGrid 可以基于分类变量创建子图网格，非常适合对比分析不同条件下的数据特征。

### 7.2 JointGrid (联合图)

```python
# 创建联合图
g = sns.JointGrid(data=tips, x="total_bill", y="tip", height=7)
g.plot_joint(sns.scatterplot)
g.plot_marginals(sns.histplot, kde=True)
plt.suptitle("账单金额与小费的联合分布", y=1.02)
plt.show()
```

联合图在一张图中同时展示两个变量的关系和各自的分布，非常直观。

## 8. 小结

Seaborn 是数据可视化的强大工具，特别适合:
- 探索数据关系和分布
- 创建统计图表
- 制作出版质量的可视化图表
- 快速生成美观的数据可视化

掌握 Seaborn 可以大大提高数据分析和展示效率，尤其适合数据科学和机器学习项目中的数据探索阶段。