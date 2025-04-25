from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载基础模型
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# 2. 定义LoRA配置
lora_config = LoraConfig(
    r=16,                      # LoRA矩阵的秩
    lora_alpha=32,             # LoRA alpha参数
    target_modules=["q_proj", "v_proj"],  # 要应用LoRA的模块
    lora_dropout=0.05,         # LoRA dropout
    bias="none",               # 是否包括偏置参数
    task_type=TaskType.CAUSAL_LM  # 任务类型
)

# 3. 创建PEFT模型
peft_model = get_peft_model(model, lora_config)
print(f"可训练参数比例: {peft_model.print_trainable_parameters()}")

# 4. 微调PEFT模型
# (使用与常规微调类似的Trainer API)
...

# 5. 推理
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = peft_model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 6. 保存和加载PEFT模型
peft_model.save_pretrained("./peft_model")