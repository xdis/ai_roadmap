import torch.quantization as quantization

def quantization_aware_fine_tuning(model, train_dataloader, eval_dataloader, num_epochs=3):
    """量化感知微调"""
    # 1. 为量化准备模型(替换特定操作为可量化版本)
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    model_prepared = quantization.prepare_qat(model)
    
    # 2. 量化感知训练
    optimizer = AdamW(model_prepared.parameters(), lr=5e-5)
    
    for epoch in range(num_epochs):
        # 训练循环
        model_prepared.train()
        for batch in train_dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            
            outputs = model_prepared(**inputs)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
        
        # 评估
        model_prepared.eval()
        eval_loss = 0
        for batch in eval_dataloader:
            with torch.no_grad():
                inputs, labels = batch
                outputs = model_prepared(**inputs)
                eval_loss += outputs.loss.item()
    
    # 3. 转换为量化模型
    model_quantized = quantization.convert(model_prepared)
    
    return model_quantized