from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. 加载预训练模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 2. 准备输入提示
prompt = "人工智能正在改变世界，特别是在"
inputs = tokenizer(prompt, return_tensors="pt")

# 3. 贪婪生成(简单但多样性低)
greedy_output = model.generate(
    inputs.input_ids, 
    max_length=100,
    do_sample=False  # 贪婪解码
)
print("贪婪生成:", tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# 4. 使用Top-k采样(平衡质量和多样性)
topk_output = model.generate(
    inputs.input_ids,
    max_length=100,
    do_sample=True,  # 启用采样
    temperature=0.7, # 温度参数
    top_k=50,        # Top-k参数
)
print("Top-k采样:", tokenizer.decode(topk_output[0], skip_special_tokens=True))

# 5. 使用nucleus(Top-p)采样
nucleus_output = model.generate(
    inputs.input_ids,
    max_length=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,       # Top-p参数
)
print("Nucleus采样:", tokenizer.decode(nucleus_output[0], skip_special_tokens=True))

# 6. 结合多种策略
combined_output = model.generate(
    inputs.input_ids,
    max_length=100,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    no_repeat_ngram_size=2,  # 避免重复的n-gram
    repetition_penalty=1.2,  # 重复惩罚
    num_return_sequences=3   # 返回多个序列
)

# 打印多个生成结果
for i, output in enumerate(combined_output):
    print(f"生成 {i+1}:", tokenizer.decode(output, skip_special_tokens=True))