from transformers import BertConfig

# 创建自定义配置
config = BertConfig(
    vocab_size=30522,          # 词汇表大小
    hidden_size=768,           # 隐藏层维度
    num_hidden_layers=6,       # Transformer层数
    num_attention_heads=12,    # 注意力头数
    intermediate_size=3072,    # 前馈网络维度
)

# 使用自定义配置创建模型
from transformers import BertModel
model = BertModel(config)  # 从配置创建模型