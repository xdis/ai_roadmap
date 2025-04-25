from datasets import load_dataset

# 加载内置数据集
squad_dataset = load_dataset("squad")  # 加载问答数据集
print(squad_dataset.column_names)      # 查看数据列

# 数据映射处理：在整个数据集上应用函数
def preprocess_function(examples):
    return tokenizer(examples["question"], examples["context"], truncation=True)

tokenized_dataset = squad_dataset.map(preprocess_function, batched=True)

# 数据过滤、选择和格式转换
filtered = tokenized_dataset.filter(lambda x: len(x["question"]) > 10)
selected = tokenized_dataset.select([0, 10, 20, 30])  # 选择特定样本
pytorch_dataset = tokenized_dataset.with_format("torch")  # 转为PyTorch格式