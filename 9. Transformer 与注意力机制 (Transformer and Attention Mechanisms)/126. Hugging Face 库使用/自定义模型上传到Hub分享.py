from transformers import AutoModel, AutoTokenizer
from huggingface_hub import notebook_login

# 1. 登录Hugging Face Hub
notebook_login()

# 2. 加载并修改模型
model_checkpoint = "bert-base-uncased"
model = AutoModel.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 3. 自定义修改...
# (例如微调、架构调整等)

# 4. 将模型推送到Hub
model_name = "my-custom-bert"
model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)

# 现在可以通过"your-username/my-custom-bert"访问