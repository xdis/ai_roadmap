BERT风格预训练核心代码示例,pydef causal_language_modeling_loss(logits, input_ids):
    """计算因果语言建模损失"""
    # 移位输入作为目标：预测下一个标记
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # 计算交叉熵损失
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1))
    return loss