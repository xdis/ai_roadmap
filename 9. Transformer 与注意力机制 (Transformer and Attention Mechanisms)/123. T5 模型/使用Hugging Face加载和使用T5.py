from transformers import T5ForConditionalGeneration, T5Tokenizer

# 加载预训练模型和分词器
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 翻译任务
input_text = "translate English to German: The house is wonderful."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 生成翻译
outputs = model.generate(input_ids, max_length=40)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translation)  # 输出: Das Haus ist wunderbar.