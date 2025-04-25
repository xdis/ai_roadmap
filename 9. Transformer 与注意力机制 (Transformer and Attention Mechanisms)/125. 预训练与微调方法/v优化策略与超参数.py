# 常用微调超参数范围
learning_rates = [1e-5, 3e-5, 5e-5]  # 通常比预训练小
batch_sizes = [16, 32]  # 通常较小
epochs = [2, 3, 4]  # 通常不需要很多轮次

# 学习率预热和衰减
from transformers import get_linear_schedule_with_warmup

def create_optimizer_and_scheduler(model, train_steps, warmup_ratio=0.1):
    """创建优化器和学习率调度器"""
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
    warmup_steps = int(train_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=train_steps
    )
    
    return optimizer, scheduler