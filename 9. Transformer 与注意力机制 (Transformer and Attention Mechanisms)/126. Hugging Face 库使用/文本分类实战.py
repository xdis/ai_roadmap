import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 加载分词器和预训练模型
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# 2. 准备输入文本
text = "I really enjoyed this movie, it was fantastic!"

# 3. 分词处理
inputs = tokenizer(text, return_tensors="pt")

# 4. 模型推理
with torch.no_grad():
    outputs = model(**inputs)
    
# 5. 处理预测结果
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
positive_prob = probabilities[0][1].item()
print(f"Positive sentiment probability: {positive_prob:.4f}")

# 获取预测标签
predicted_class = torch.argmax(probabilities, dim=-1).item()
print(f"Predicted class: {'positive' if predicted_class == 1 else 'negative'}")