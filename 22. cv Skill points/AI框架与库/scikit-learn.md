# scikit-learn 基础

scikit-learn (简称 sklearn) 是 Python 中最流行的机器学习库之一，它提供了简单且高效的工具，用于数据分析和数据挖掘。它建立在 NumPy、SciPy 和 matplotlib 之上，是大多数机器学习任务的首选库。

## 1. 为什么选择 scikit-learn？

- **简单易用**：一致的接口设计使学习和使用变得简单
- **广泛的算法支持**：包含分类、回归、聚类、降维等几乎所有常见机器学习算法
- **完善的文档**：详细的文档和示例使学习曲线更平缓
- **与 Python 科学栈集成**：与 NumPy、Pandas 等库无缝协作
- **开源社区支持**：大型活跃社区持续改进和支持

## 2. 基本工作流程

scikit-learn 的工作流程通常包括以下步骤：

1. 数据准备（加载和清洗）
2. 特征工程（特征提取、选择和转换）
3. 模型训练
4. 模型评估
5. 预测

## 3. 核心组件和功能

### 3.1 数据预处理

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# 标准化：使数据均值为0，方差为1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 归一化：将数据缩放到[0,1]区间
min_max_scaler = MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)

# 独热编码：将分类特征转换为数值向量
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)
```

### 3.2 分类算法

#### 逻辑回归

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载示例数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 评估模型
accuracy = model.score(X_test, y_test)
print(f"准确率: {accuracy:.2f}")

# 进行预测
predictions = model.predict(X_test)
```

#### 决策树

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# 创建并训练决策树
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# 可视化决策树
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
```

### 3.3 回归算法

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

# 加载波士顿房价数据集
boston = load_boston()
X, y = boston.data, boston.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练线性回归模型
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 预测并评估
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差: {mse:.2f}")
print(f"R² 分数: {regressor.score(X_test, y_test):.2f}")
```

### 3.4 聚类算法

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 创建示例数据
X = np.random.rand(100, 2)

# 应用K均值聚类
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X')
plt.title('K-means 聚类结果')
plt.show()
```

### 3.5 模型评估与交叉验证

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC

# 使用交叉验证评估模型
model = SVC()
scores = cross_val_score(model, X, y, cv=5)  # 5折交叉验证
print(f"交叉验证分数: {scores}")
print(f"平均分数: {scores.mean():.2f}")

# 网格搜索优化超参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_:.2f}")

# 使用最佳模型
best_model = grid_search.best_estimator_
```

### 3.6 降维和特征选择

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# 主成分分析 (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化PCA结果
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title('PCA 降维到2维')
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.show()

# 基于统计测试的特征选择
selector = SelectKBest(f_classif, k=2)  # 选择最佳的2个特征
X_selected = selector.fit_transform(X, y)
```

## 4. 完整机器学习流程示例

下面是一个完整的机器学习流程，包含数据处理、特征工程、模型训练和评估：

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# 1. 加载数据 (以鸢尾花数据集为例)
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. 数据探索
print(X.head())
print(f"数据形状: {X.shape}")
print(X.describe())

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 创建处理和训练管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 步骤1: 标准化
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # 步骤2: 分类器
])

# 5. 训练模型
pipeline.fit(X_train, y_train)

# 6. 评估模型
y_pred = pipeline.predict(X_test)
print(f"模型准确率: {pipeline.score(X_test, y_test):.2f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 8. 保存模型 (可选)
import joblib
joblib.dump(pipeline, 'iris_model.pkl')

# 9. 加载模型并进行预测 (可选)
loaded_model = joblib.load('iris_model.pkl')
# 对新数据进行预测
new_data = X_test[:3]
predictions = loaded_model.predict(new_data)
print(f"新样本预测结果: {predictions}")
```

## 5. scikit-learn 的优势和局限性

### 优势：
- 简单易用的接口
- 广泛的算法实现
- 高质量的文档和示例
- 与数据科学生态系统无缝集成
- 强大的社区支持

### 局限性：
- 不适合深度学习任务 (应考虑 TensorFlow 或 PyTorch)
- 处理大规模数据集时性能可能受限
- 不特别针对计算机视觉或自然语言处理优化

## 6. 使用建议

1. **从简单开始**：先尝试简单模型，再逐步增加复杂度
2. **使用管道**：Pipeline 和 ColumnTransformer 简化工作流程
3. **交叉验证**：始终使用交叉验证评估模型
4. **超参数调优**：使用 GridSearchCV 或 RandomizedSearchCV
5. **模型保存**：使用 joblib 保存训练好的模型

scikit-learn 是机器学习入门和日常数据科学工作的绝佳工具，掌握它将为你的AI学习打下坚实基础。