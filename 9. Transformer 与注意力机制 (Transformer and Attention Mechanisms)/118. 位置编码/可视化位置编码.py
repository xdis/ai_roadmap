def visualize_positional_encoding(d_model=64, max_length=100):
    # 创建位置编码
    pe = torch.zeros(max_length, d_model)
    position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # 可视化
    plt.figure(figsize=(15, 8))
    
    # 热图显示所有位置编码
    plt.subplot(1, 2, 1)
    plt.imshow(pe.numpy(), aspect='auto', cmap='viridis')
    plt.title('Position Encodings')
    plt.xlabel('Encoding Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    
    # 选择展示几个具体位置的编码
    plt.subplot(1, 2, 2)
    positions = [0, 10, 20, 30, 40]
    for pos in positions:
        plt.plot(pe[pos, :].numpy(), label=f'Position {pos}')
    plt.title('Position Encoding Values for Different Positions')
    plt.xlabel('Encoding Dimension')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# 调用可视化函数
visualize_positional_encoding()