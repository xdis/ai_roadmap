# 使用多语言BERT进行跨语言迁移
# 在一种语言上训练，在另一种语言上测试
from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification

# 加载多语言模型
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(tag2id))

# 训练(例如使用英语数据)
# train_bert_ner(model, en_train_loader, en_dev_loader, optimizer, scheduler, device)

# 测试(例如使用中文数据)
# evaluate_bert_ner(model, zh_test_loader, device, id2tag)