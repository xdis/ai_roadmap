def domain_adaptation_ner(source_model, target_data, epochs=3):
    """源领域模型向目标领域的适应"""
    # 准备目标领域数据
    target_loader = DataLoader(target_data, batch_size=16, shuffle=True)
    
    # 设置不同层的学习率
    # 直觉：低层特征更通用，高层特征更特定于任务
    no_decay = ["bias", "LayerNorm.weight"]
    # 获取BERT所有层的名称
    bert_layers = [f"encoder.layer.{i}." for i in range(12)]
    
    optimizer_grouped_parameters = [
        # 低层BERT参数(小学习率)
        {
            "params": [p for n, p in source_model.named_parameters() 
                      if any(layer in n for layer in bert_layers[:4])
                      and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": 1e-5
        },
        {
            "params": [p for n, p in source_model.named_parameters() 
                      if any(layer in n for layer in bert_layers[:4])
                      and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": 1e-5
        },
        # 高层BERT参数(中等学习率)
        {
            "params": [p for n, p in source_model.named_parameters() 
                      if any(layer in n for layer in bert_layers[4:])
                      and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": 3e-5
        },
        {
            "params": [p for n, p in source_model.named_parameters() 
                      if any(layer in n for layer in bert_layers[4:])
                      and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": 3e-5
        },
        # 分类层参数(高学习率)
        {
            "params": [p for n, p in source_model.named_parameters() 
                      if "classifier" in n and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": 5e-5
        },
        {
            "params": [p for n, p in source_model.named_parameters() 
                      if "classifier" in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": 5e-5
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters)
    
    # 微调
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_model.to(device)
    
    for epoch in range(epochs):
        source_model.train()
        total_loss = 0
        
        for batch in target_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = source_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(target_loader):.4f}")
    
    return source_model