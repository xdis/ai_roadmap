def greedy_decode(model, src, max_len, start_symbol, device):
    model.eval()
    
    src = src.to(device)
    src_mask = (src == PAD_IDX).transpose(0, 1).to(device)
    
    # 编码源序列
    memory = model.encode(src, src_mask)
    
    # 准备目标序列起始标记
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src).to(device)
    
    # 逐个生成标记
    for i in range(max_len-1):
        # 解码当前序列
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model.decode(ys, memory, tgt_mask, src_mask)
        
        # 获取下一个标记
        prob = out[:, -1]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        # 添加到目标序列
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word).to(device)], dim=1)
        
        # 检查是否生成结束标记
        if next_word == EOS_IDX:
            break
            
    return ys