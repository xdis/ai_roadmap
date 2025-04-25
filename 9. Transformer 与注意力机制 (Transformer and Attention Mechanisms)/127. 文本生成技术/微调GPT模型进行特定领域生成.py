from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# 1. 加载模型和分词器
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 特殊处理：添加填充标记(GPT2默认没有)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# 2. 准备数据集
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128  # 上下文窗口大小
    )

train_dataset = load_dataset("path/to/your/train.txt", tokenizer)
eval_dataset = load_dataset("path/to/your/eval.txt", tokenizer)

# 3. 数据整理器(处理填充、掩码等)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # 不使用掩码语言建模，用因果语言建模
)

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
    logging_dir="./logs",
)

# 5. 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 6. 微调模型
trainer.train()

# 7. 保存微调后的模型
model_path = "./gpt2-finetuned"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# 8. 使用微调后的模型生成文本
fine_tuned_model = GPT2LMHeadModel.from_pretrained(model_path)
outputs = fine_tuned_model.generate(
    inputs.input_ids,
    max_length=100,
    do_sample=True,
    top_p=0.9,
    temperature=0.7
)
print("微调后生成:", tokenizer.decode(outputs[0], skip_special_tokens=True))