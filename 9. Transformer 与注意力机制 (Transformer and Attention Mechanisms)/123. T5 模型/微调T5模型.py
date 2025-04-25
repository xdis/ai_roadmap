from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 加载模型和分词器
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# 准备数据集（以摘要任务为例）
dataset = load_dataset("cnn_dailymail", "3.0.0")

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    targets = examples["highlights"]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 处理数据集
processed_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = processed_dataset["train"].select(range(1000))  # 为简化示例只选取一部分

# 设置训练参数
training_args = {
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 4,
    "num_train_epochs": 3,
}

# 配置数据加载器
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=training_args["per_device_train_batch_size"],
    shuffle=True
)

# 设置优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=training_args["learning_rate"])

# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
for epoch in range(training_args["num_train_epochs"]):
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# 保存微调后的模型
model.save_pretrained("./t5-fine-tuned-summarization")
tokenizer.save_pretrained("./t5-fine-tuned-summarization")