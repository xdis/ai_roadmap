# 分词器工作流程
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 输入文本
text = "Hello, how are you?"

# 完整处理流程
encoded = tokenizer(
    text,
    padding="max_length",  # 填充策略
    truncation=True,       # 截断策略
    max_length=10,         # 最大长度
    return_tensors="pt"    # 返回PyTorch张量
)

# encoded包含:
# - input_ids: 标记ID列表
# - attention_mask: 注意力掩码
# - token_type_ids: 标记类型ID(用于某些模型)