from transformers import BertForTokenClassification, BertTokenizer

# 加载预训练模型
model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(tag2idx))
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 处理输入
text = "我爱北京天安门"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
inputs = torch.tensor([token_ids])

# 预测
outputs = model(inputs)
predictions = torch.argmax(outputs.logits, dim=2)
predicted_tags = [idx2tag[p.item()] for p in predictions[0]]