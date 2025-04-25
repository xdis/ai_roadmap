from transformers import AutoModel, AutoTokenizer

# 自动选择正确的模型和分词器类
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 以下类似的Auto类用于不同任务
# AutoModelForSequenceClassification
# AutoModelForQuestionAnswering
# AutoModelForTokenClassification
# AutoModelForMaskedLM
# AutoModelForCausalLM