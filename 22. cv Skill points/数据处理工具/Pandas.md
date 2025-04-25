# Pandas 数据处理库

Pandas 是 Python 中用于数据分析和处理的核心库，它提供了强大的数据结构和函数，使数据处理变得简单高效。

## 核心数据结构

Pandas 主要有两种数据结构：

1. **Series**：一维数组，类似于带标签的 NumPy 数组
2. **DataFrame**：二维表格数据结构，类似于 Excel 电子表格

## 基础使用示例

### 导入 Pandas

```python
import pandas as pd
import numpy as np  # 常与 NumPy 一起使用
```

### 创建 Series

```python
# 从列表创建 Series
s = pd.Series([1, 3, 5, 7, 9])
print(s)
# 输出:
# 0    1
# 1    3
# 2    5
# 3    7
# 4    9
# dtype: int64

# 指定索引
s = pd.Series([1, 3, 5, 7, 9], index=['a', 'b', 'c', 'd', 'e'])
print(s)
# 输出:
# a    1
# b    3
# c    5
# d    7
# e    9
# dtype: int64

# 从字典创建 Series
d = {'a': 1, 'b': 3, 'c': 5}
s = pd.Series(d)
print(s)
# 输出:
# a    1
# b    3
# c    5
# dtype: int64
```

### 创建 DataFrame

```python
# 从字典创建 DataFrame
data = {
    '姓名': ['张三', '李四', '王五'],
    '年龄': [25, 30, 35],
    '城市': ['北京', '上海', '广州']
}
df = pd.DataFrame(data)
print(df)
# 输出:
#    姓名  年龄  城市
# 0  张三  25  北京
# 1  李四  30  上海
# 2  王五  35  广州

# 从 NumPy 数组创建 DataFrame
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(array, columns=['A', 'B', 'C'])
print(df)
# 输出:
#    A  B  C
# 0  1  2  3
# 1  4  5  6
# 2  7  8  9
```

## 数据导入与导出

Pandas 可以轻松读取和写入各种格式的数据文件：

```python
# 读取 CSV 文件
df = pd.read_csv('data.csv')

# 读取 Excel 文件
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 导出到 CSV
df.to_csv('output.csv', index=False)

# 导出到 Excel
df.to_excel('output.xlsx', index=False)
```

## 数据查看与选择

```python
# 查看前几行数据
print(df.head())  # 默认前 5 行

# 查看后几行数据
print(df.tail(3))  # 后 3 行

# 查看基本信息
print(df.info())

# 查看统计摘要
print(df.describe())

# 列选择
print(df['姓名'])  # 选择单列
print(df[['姓名', '年龄']])  # 选择多列

# 行选择 - 位置索引
print(df.iloc[0])  # 第一行
print(df.iloc[0:2])  # 前两行
print(df.iloc[[0, 2]])  # 第一行和第三行

# 行选择 - 标签索引（如果设置了索引）
df = df.set_index('姓名')
print(df.loc['张三'])

# 条件选择
print(df[df['年龄'] > 30])  # 年龄大于 30 的所有行
```

## 数据处理

### 处理缺失值

```python
# 检查缺失值
print(df.isnull().sum())

# 删除缺失值
df_clean = df.dropna()

# 填充缺失值
df_filled = df.fillna(0)  # 用 0 填充所有缺失值
df_filled = df.fillna(method='ffill')  # 用前一个值填充
```

### 数据操作

```python
# 添加新列
df['新列'] = df['年龄'] * 2

# 删除列
df = df.drop(columns=['新列'])

# 重命名列
df = df.rename(columns={'年龄': 'Age'})

# 排序
df_sorted = df.sort_values('年龄', ascending=False)  # 按年龄降序排列
```

### 数据聚合与分组

```python
# 分组统计
grouped = df.groupby('城市')
print(grouped['年龄'].mean())  # 各城市年龄平均值

# 聚合计算
result = df.groupby('城市').agg({
    '年龄': ['mean', 'min', 'max', 'count']
})
print(result)
```

### 数据合并

```python
# 定义两个 DataFrame
df1 = pd.DataFrame({
    '员工ID': [1, 2, 3, 4],
    '姓名': ['张三', '李四', '王五', '赵六']
})

df2 = pd.DataFrame({
    '员工ID': [1, 2, 3, 5],
    '部门': ['销售', '技术', '财务', '人事']
})

# 合并数据 (类似 SQL JOIN)
merged = pd.merge(df1, df2, on='员工ID', how='inner')
print(merged)
# 输出:
#    员工ID  姓名  部门
# 0     1  张三  销售
# 1     2  李四  技术
# 2     3  王五  财务

# 连接数据 (按行拼接)
concat_df = pd.concat([df1, df1], ignore_index=True)
print(concat_df)
```

## 数据分析实例

下面是一个简单的数据分析实例，展示如何使用 Pandas 进行基本的数据分析：

```python
# 假设我们有一个销售数据文件
# 读取数据
sales_df = pd.read_csv('sales.csv')

# 1. 数据检查
print(sales_df.head())
print(sales_df.info())
print(sales_df.describe())

# 2. 缺失值处理
print(sales_df.isnull().sum())
sales_df = sales_df.dropna()

# 3. 按产品类型分组，计算各类产品的销售额总和、平均值等
product_stats = sales_df.groupby('产品类型').agg({
    '销售额': ['sum', 'mean', 'count'],
    '利润': ['sum', 'mean']
})

# 4. 找出销售额最高的前 10 个产品
top_products = sales_df.nlargest(10, '销售额')

# 5. 计算每月销售趋势
sales_df['日期'] = pd.to_datetime(sales_df['日期'])
sales_df['月份'] = sales_df['日期'].dt.strftime('%Y-%m')
monthly_sales = sales_df.groupby('月份')['销售额'].sum().reset_index()

# 6. 数据可视化 (搭配 matplotlib 使用)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(monthly_sales['月份'], monthly_sales['销售额'])
plt.title('月度销售额')
plt.xlabel('月份')
plt.ylabel('销售额')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 性能提示

1. 尽量使用向量化操作，避免使用循环
2. 对于大型数据集，考虑使用 `chunksize` 参数分批读取
3. 使用适当的数据类型可以节省内存
4. 对于不需要的列可以提前丢弃，减少内存使用

## 小结

Pandas 是数据分析和处理的强大工具，掌握它可以让你：

- 轻松导入和导出各种格式的数据
- 高效地清洗和转换数据
- 执行复杂的数据分析和聚合
- 与其他数据科学库（如 Matplotlib、Scikit-learn）无缝集成

通过以上基础知识和示例，你已经可以开始使用 Pandas 进行基本的数据处理和分析工作了！