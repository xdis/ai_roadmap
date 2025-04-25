# 梯度累积示例(每4步更新一次)
accumulation_steps = 4
optimizer.zero_grad()
for i, batch in enumerate(data_loader):
    outputs = model(batch)
    loss = criterion(outputs, batch.target) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()