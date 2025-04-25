from transformers import MarianMTModel, MarianTokenizer

# 加载德语到英语的翻译模型
model_name = "Helsinki-NLP/opus-mt-de-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 翻译文本
def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 德语到英语翻译
german_text = "Ich liebe es, mit Hugging Face zu arbeiten."
english_text = translate(german_text)
print(f"德语: {german_text}")
print(f"英语: {english_text}")

# 多语言模型(XLM-RoBERTa)
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

# 加载多语言分类模型
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 可以微调处理多种语言