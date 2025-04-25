import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 1. 加载预训练模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 配置LoRA
lora_config = LoraConfig(
    r=8,                           # 低秩矩阵的秩
    lora_alpha=32,                 # 缩放参数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 要应用LoRA的模块名称
    lora_dropout=0.05,             # Dropout概率
    bias="none",                   # 是否包含偏置项
    task_type=TaskType.CAUSAL_LM   # 任务类型
)

# 3. 创建PEFT模型
peft_model = get_peft_model(model, lora_config)

# 4. 显示可训练参数数量
trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in peft_model.parameters())
print(f"可训练参数: {trainable_params} ({trainable_params/total_params*100:.2f}%)")
print(f"总参数: {total_params}")

# 5. 准备数据集
dataset = load_dataset("your_dataset")  # 替换为您的数据集
# 数据预处理代码...

# 6. 设置训练参数
training_args = TrainingArguments(
    output_dir="./lora_model",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=1000,
    save_steps=100,
    logging_steps=10,
    fp16=True,  # 使用半精度加速
)

# 7. 创建Trainer并开始训练
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset["train"],
    # 其他Trainer参数...
)

# 8. 训练模型
trainer.train()

# 9. 保存微调后的LoRA权重(非常小!)
peft_model.save_pretrained("./lora_weights")