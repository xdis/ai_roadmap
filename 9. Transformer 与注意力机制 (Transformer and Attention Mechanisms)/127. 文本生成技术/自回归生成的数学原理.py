def autoregressive_generation(model, prompt_ids, max_length):
    """自回归生成的基本实现"""
    input_ids = prompt_ids.clone()
    
    # 逐标记生成
    for _ in range(max_length):
        # 获取模型对下一个标记的预测
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
        
        # 选择下一个标记(贪婪策略)
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # 将新标记添加到序列中
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        
        # 检查是否生成了结束标记
        if next_token_id.item() == model.config.eos_token_id:
            break
            
    return input_ids