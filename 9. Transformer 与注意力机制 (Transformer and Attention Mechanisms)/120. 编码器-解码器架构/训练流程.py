def train_transformer(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        src, tgt = batch.src.to(device), batch.tgt.to(device)
        tgt_input = tgt[:, :-1]  # 排除最后一个标记
        tgt_output = tgt[:, 1:]  # 排除第一个标记(通常是<BOS>)
        
        # 创建源和目标序列的掩码
        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt_input, src_padding_mask, tgt_mask)
        
        # 计算损失
        loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                         tgt_output.contiguous().view(-1))
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)