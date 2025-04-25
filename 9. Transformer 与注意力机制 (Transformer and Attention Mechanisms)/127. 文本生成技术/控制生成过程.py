# 自定义生成函数，实现更精细的控制
def custom_generate(model, tokenizer, prompt, max_length=100):
    # 准备初始输入
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    
    # 逐标记生成
    for _ in range(max_length):
        # 获取模型输出
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :].squeeze()
        
        # 应用重复惩罚
        if input_ids.shape[1] > 1:
            # 增加已生成标记的惩罚
            for token_id in input_ids[0][-5:]:  # 考虑最后5个标记
                next_token_logits[token_id] /= 1.5  # 降低再次出现的概率
        
        # 应用关键词增强
        keywords = ["创新", "研究", "发展"]  # 示例关键词
        keyword_ids = []
        for keyword in keywords:
            keyword_id = tokenizer.encode(keyword, add_special_tokens=False)
            if len(keyword_id) == 1:  # 确保是单标记
                keyword_ids.append(keyword_id[0])
                
        # 提高关键词的生成概率
        for kid in keyword_ids:
            next_token_logits[kid] *= 1.2
            
        # 应用温度
        temperature = 0.7
        next_token_logits = next_token_logits / temperature
        
        # 应用Top-p采样
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除低于top_p的标记
        top_p = 0.9
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[indices_to_remove] = -float('Inf')
        
        # 采样下一个标记
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        
        # 添加到序列
        input_ids = torch.cat((input_ids, next_token), dim=1)
        
        # 检查是否生成了EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    # 返回生成的文本
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)