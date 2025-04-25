def token_entropy_uncertainty(model, unlabeled_loader, device, n_samples=10):
    """基于标记熵的不确定度采样"""
    model.eval()
    uncertainties = []
    samples = []
    
    with torch.no_grad():
        for i, (inputs, idx) in enumerate(unlabeled_loader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 多次前向传播(使用dropout)
            model.train()  # 启用dropout
            logits_samples = []
            for _ in range(n_samples):
                outputs = model(**inputs)
                logits_samples.append(outputs.logits.cpu())
            
            # 计算每个标记的熵
            logits_samples = torch.stack(logits_samples)  # [n_samples, batch, seq_len, n_classes]
            mean_probs = torch.softmax(logits_samples, dim=-1).mean(dim=0)  # [batch, seq_len, n_classes]
            entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)  # [batch, seq_len]
            
            # 获取每个序列的平均熵
            masked_entropy = entropy * inputs['attention_mask'].cpu()
            seq_entropy = masked_entropy.sum(dim=1) / inputs['attention_mask'].sum(dim=1).cpu()
            
            # 保存结果
            uncertainties.extend(seq_entropy.tolist())
            samples.extend(idx.tolist())
    
    # 按不确定度排序
    sorted_indices = [idx for _, idx in sorted(zip(uncertainties, samples), reverse=True)]
    return sorted_indices