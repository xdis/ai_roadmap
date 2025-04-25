from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载GPT-2模型和分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 编码输入文本
input_text = "Once upon a time in a land far away,"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 生成文本
output_sequences = model.generate(
    input_ids,
    max_length=100,               # 最大长度
    num_return_sequences=3,       # 返回3个序列
    temperature=0.8,              # 温度参数(越高越随机)
    top_k=50,                    # Top-K采样
    top_p=0.95,                  # Top-P(核采样)
    repetition_penalty=1.2,      # 重复惩罚
    do_sample=True,              # 使用采样
    no_repeat_ngram_size=2       # 避免重复的n元组
)

# 解码并打印生成的文本
for i, seq in enumerate(output_sequences):
    generated_text = tokenizer.decode(seq, skip_special_tokens=True)
    print(f"Generated {i+1}: {generated_text}")