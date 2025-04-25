def benchmark_attention_variants(sequence_lengths, batch_size=32, d_model=512, num_heads=8):
    """比较不同注意力变体的性能"""
    # 创建测试数据
    results = {
        "standard": [],
        "flash_attention": [],
        "linear_attention": [],
        "local_attention": []
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建不同的注意力实现
    standard_attn = MultiHeadAttention(d_model, num_heads).to(device)
    flash_attn = FlashMultiHeadAttention(d_model, num_heads).to(device)
    linear_attn = LinearMultiHeadAttention(d_model, num_heads).to(device)
    local_attn = LocalMultiHeadAttention(d_model, num_heads, window_size=128).to(device)
    
    for seq_len in sequence_lengths:
        x = torch.randn(batch_size, seq_len, d_model).to(device)
        
        # 测量标准注意力时间
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # 预热
        _ = standard_attn(x, x, x)
        torch.cuda.synchronize()
        
        # 计时
        start.record()
        _ = standard_attn(x, x, x)
        end.record()
        torch.cuda.synchronize()
        results["standard"].append(start.elapsed_time(end))
        
        # 测试其他变体...
        # [类似的测量代码]
    
    # 绘制性能比较图
    plt.figure(figsize=(10, 6))
    for name, times in results.items():
        plt.plot(sequence_lengths, times, label=name)
    
    plt.xlabel("Sequence Length")
    plt.ylabel("Time (ms)")
    plt.title("Attention Variants Performance Comparison")
    plt.legend()
    plt.grid()
    plt.show()