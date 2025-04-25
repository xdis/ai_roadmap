from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. 加载数据集
dataset = load_dataset("glue", "mnli")

# 2. 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=3  # MNLI有3个类别
)

# 3. 数据预处理
def preprocess_function(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. 定义评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro")
    }

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",              # 输出目录
    learning_rate=2e-5,                  # 学习率
    per_device_train_batch_size=16,      # 训练批次大小
    per_device_eval_batch_size=16,       # 评估批次大小
    num_train_epochs=3,                  # 训练轮数
    weight_decay=0.01,                   # 权重衰减
    evaluation_strategy="epoch",         # 每epoch评估一次
    save_strategy="epoch",               # 每epoch保存一次
    load_best_model_at_end=True,         # 加载最佳模型
)

# 6. 创建Trainer实例
trainer = Trainer(
    model=model,                        # 模型
    args=training_args,                 # 训练参数
    train_dataset=tokenized_dataset["train"],  # 训练集
    eval_dataset=tokenized_dataset["validation_matched"],  # 验证集
    compute_metrics=compute_metrics,    # 评估指标
)

# 7. 开始微调
trainer.train()

# 8. 保存模型
model_path = "./bert-finetuned-mnli"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# 9. 模型评估
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 10. 使用微调模型进行推理
from transformers import pipeline
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

result = classifier(
    "The company reported profits this quarter, contradicting analysts' expectations of losses."
)
print(result)