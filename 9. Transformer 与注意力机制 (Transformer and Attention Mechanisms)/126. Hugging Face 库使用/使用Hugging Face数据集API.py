from datasets import load_dataset, DatasetDict, Features, Value, ClassLabel

# 1. 加载内置数据集
imdb = load_dataset("imdb")
print(f"IMDB数据集: {imdb}")  # 查看数据结构

# 2. 从本地文件加载数据集
csv_dataset = load_dataset("csv", data_files={"train": "data/train.csv", "test": "data/test.csv"})

# 3. 数据处理
# 过滤数据
short_reviews = imdb["train"].filter(lambda x: len(x["text"]) < 1000)

# 数据映射
def add_length(example):
    example["length"] = len(example["text"])
    return example

dataset_with_length = imdb.map(add_length)

# 4. 打乱和分割数据
train_test = imdb["train"].train_test_split(test_size=0.1)
print(f"训练集大小: {len(train_test['train'])}, 测试集大小: {len(train_test['test'])}")

# 5. 保存和加载处理后的数据集
train_test.save_to_disk("./imdb_split")
reloaded_dataset = DatasetDict.load_from_disk("./imdb_split")

# 6. 创建自定义数据集
from datasets import Dataset
import pandas as pd

df = pd.DataFrame({
    "text": ["这是第一个样本", "这是第二个样本", "这是第三个样本"],
    "label": [0, 1, 0]
})

custom_dataset = Dataset.from_pandas(df)