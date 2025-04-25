def full_fine_tuning(pretrained_model, train_dataloader, optimizer, num_epochs):
    """全参数微调"""
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            inputs, labels = batch
            outputs = pretrained_model(**inputs)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()