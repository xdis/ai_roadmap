# Databricks 基础介绍

Databricks 是一个基于云的数据分析平台，整合了数据工程、数据科学和机器学习功能。它建立在 Apache Spark 的基础上，提供了一个协作环境，使数据团队能够高效地处理大规模数据。

## Databricks 核心特点

1. **统一数据分析平台**：集成数据处理、分析和机器学习
2. **基于 Apache Spark**：利用分布式计算能力处理大数据
3. **笔记本协作环境**：支持多种语言（Python、SQL、R、Scala）
4. **Delta Lake**：支持可靠的数据湖存储
5. **MLflow**：简化机器学习工作流程
6. **自动扩展**：根据工作负载动态调整计算资源

## 基本使用示例

### 1. 连接到 Databricks

```python
# 使用 Databricks Connect 从本地环境连接到 Databricks 集群
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .remote("databricks://YOUR_WORKSPACE_ID:PORT") \
    .getOrCreate()
```

### 2. 基本数据操作

```python
# 读取数据
df = spark.read.format("csv").option("header", "true").load("/databricks-datasets/samples/population-vs-price/data_geo.csv")

# 显示数据
display(df.limit(5))

# 基本数据处理
from pyspark.sql.functions import col

# 筛选数据
filtered_df = df.filter(col("2014_Population_estimate") > 10000)

# 选择列
selected_df = df.select("State", "2014_Population_estimate", "2015_median_sales_price")

# 创建新列
df_with_price_per_person = df.withColumn(
    "price_per_person", 
    col("2015_median_sales_price") / col("2014_Population_estimate")
)
```

### 3. SQL 查询

```python
# 创建临时视图
df.createOrReplaceTempView("population_data")

# 使用 SQL 查询
sql_df = spark.sql("""
    SELECT State, 
           `2014_Population_estimate` as population,
           `2015_median_sales_price` as price
    FROM population_data
    WHERE `2014_Population_estimate` > 10000
    ORDER BY price DESC
    LIMIT 10
""")

display(sql_df)
```

### 4. 数据可视化

```python
# Databricks 笔记本内置可视化功能
display(df.select("State", "2015_median_sales_price"))

# 使用 matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
pd_df = df.select("State", "2015_median_sales_price").toPandas()
pd_df.sort_values("2015_median_sales_price", ascending=False).head(10).plot.bar(x="State", y="2015_median_sales_price")
plt.title("各州房价前十")
plt.ylabel("房价 (美元)")
plt.tight_layout()
plt.show()
```

### 5. 使用 Delta Lake

```python
# 保存数据到 Delta 格式
df.write.format("delta").save("/mnt/delta/population_data")

# 读取 Delta 表
delta_df = spark.read.format("delta").load("/mnt/delta/population_data")

# 创建Delta表
spark.sql("CREATE TABLE IF NOT EXISTS population_delta USING DELTA LOCATION '/mnt/delta/population_data'")

# 更新数据
spark.sql("""
    UPDATE population_delta 
    SET `2015_median_sales_price` = `2015_median_sales_price` * 1.1
    WHERE State = 'CA'
""")
```

### 6. 机器学习示例 (使用 MLflow)

```python
import mlflow
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# 准备特征
feature_cols = ["2014_Population_estimate"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df.select("2014_Population_estimate", "2015_median_sales_price"))
data = data.withColumnRenamed("2015_median_sales_price", "label").select("features", "label")

# 分割训练集和测试集
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

# 启用 MLflow 跟踪
with mlflow.start_run(run_name="Simple Spark ML"):
    # 创建模型
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    
    # 训练模型
    model = lr.fit(train_data)
    
    # 记录参数
    mlflow.log_param("max_iter", 10)
    mlflow.log_param("reg_param", 0.3)
    
    # 预测
    predictions = model.transform(test_data)
    
    # 评估
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    
    # 记录指标
    mlflow.log_metric("rmse", rmse)
    
    # 输出结果
    print(f"模型训练完成! RMSE: {rmse}")
```

## Databricks 与常规数据处理工具的比较

| 特性 | Databricks | Python/Pandas | 传统数据仓库 |
|------|------------|---------------|-------------|
| 数据规模 | TB/PB级大数据 | GB级数据 | TB级数据 |
| 计算模式 | 分布式 | 单机 | 中心化服务器 |
| 协作能力 | 强，多人协作 | 弱，主要单人工作 | 有限 |
| 扩展性 | 高，自动扩展 | 低，受单机限制 | 中等，需手动扩展 |
| 适用场景 | 大数据分析、ML | 数据探索、原型 | 结构化查询、报表 |

## 使用要点

1. **利用分布式计算**：设计数据处理流程时应当考虑并行化
2. **避免收集大数据到驱动程序**：使用 `collect()` 要谨慎，可能导致内存溢出
3. **合理分区**：通过 `repartition()` 或 `coalesce()` 优化数据分布
4. **缓存中间结果**：使用 `cache()` 或 `persist()` 保存重复使用的数据
5. **利用 Delta Lake 事务能力**：确保数据一致性和可靠性

## 总结

Databricks 作为现代数据处理平台，结合了 Apache Spark 的分布式计算能力和云服务的便利性，特别适合大规模数据分析和机器学习应用。通过统一的笔记本环境，团队可以使用 Python、SQL、R 或 Scala 语言协作开发数据应用，而内置的 MLflow 和 Delta Lake 进一步增强了其在机器学习工作流和可靠数据湖方面的能力。