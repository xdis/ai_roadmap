def get_layer_wise_learning_rates(model, base_lr=5e-5, decay_factor=0.9):
    """为不同层设置不同学习率"""
    parameters = []
    num_layers = len(model.encoder.layer)
    
    # 输出层使用较高学习率
    parameters.append({
        'params': model.classifier.parameters(),
        'lr': base_lr
    })
    
    # 编码器层使用递减学习率
    for i in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay_factor ** (num_layers - i))
        parameters.append({
            'params': model.encoder.layer[i].parameters(),
            'lr': layer_lr
        })
    
    # 嵌入层使用最低学习率
    parameters.append({
        'params': model.embeddings.parameters(),
        'lr': base_lr * (decay_factor ** (num_layers + 1))
    })
    
    return parameters