def visualize_multihead_attention(model, sentence, tokenizer):
    """可视化多头注意力模式"""
    tokens = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        # 获取第一层注意力
        attention = model(tokens.input_ids, output_attentions=True).attentions[0]
    
    # 获取注意力权重，形状: [batch, heads, seq_len, seq_len]
    attn_weights = attention.squeeze(0)  # 移除批次维度
    
    # 获取标记化的单词
    tokens_list = tokenizer.convert_ids_to_tokens(tokens.input_ids[0])
    
    num_heads = attn_weights.shape[0]
    fig, axes = plt.subplots(2, num_heads//2, figsize=(20, 10))
    axes = axes.flatten()
    
    # 为每个注意力头创建一个热图
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn_weights[h].numpy(), cmap="viridis")
        ax.set_title(f"Head {h+1}")
        
        # 设置标签
        ax.set_xticks(range(len(tokens_list)))
        ax.set_yticks(range(len(tokens_list)))
        ax.set_xticklabels(tokens_list, rotation=90)
        ax.set_yticklabels(tokens_list)
        
    plt.tight_layout()
    plt.colorbar(im, ax=axes)
    plt.show()