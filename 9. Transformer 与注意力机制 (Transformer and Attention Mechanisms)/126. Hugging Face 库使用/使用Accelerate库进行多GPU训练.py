from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader

# 1. 初始化Accelerator
accelerator = Accelerator()

# 2. 准备模型和优化器
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 3. 准备数据集和数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=16)
eval_dataloader = DataLoader(eval_dataset, batch_size=16)

# 4. 使用accelerator准备所有组件
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# 5. 训练循环
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)  # 替代loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    # 评估
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)