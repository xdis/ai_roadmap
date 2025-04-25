from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification

# 加载多语言模型
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base', num_labels=len(tag2idx))

# 在源语言(如英语)训练
# ...训练代码...

# 在目标语言(如中文)测试
chinese_text = "我去了北京和上海"
inputs = tokenizer(chinese_text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

# 解码预测结果
pred_tags = [idx2tag[p.item()] for p in predictions[0][1:-1]]  # 去除特殊标记