from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 配置8位量化
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    device_map="auto",
    quantization_config=quantization_config
)

# 验证模型大小
model_size = sum(p.numel() for p in model.parameters()) * 1 / 8 / 1024 / 1024  # 转换为MB
print(f"量化后模型大小: {model_size:.2f} MB")