def analyze_attention_patterns(model, dataset, num_samples=100):
    """分析多头注意力的行为模式"""
    head_patterns = {
        "syntax_heads": [],
        "semantic_heads": [],
        "position_heads": []
    }
    
    for sample in dataset[:num_samples]:
        # 获取模型注意力权重
        attention_weights = get_model_attention(model, sample)
        
        # 遍历所有层和头
        for layer_idx, layer_attention in enumerate(attention_weights):
            for head_idx, head_attention in enumerate(layer_attention):
                # 分析此头的注意力模式
                if is_syntax_focused(head_attention, sample):
                    head_patterns["syntax_heads"].append((layer_idx, head_idx))
                elif is_semantic_focused(head_attention, sample):
                    head_patterns["semantic_heads"].append((layer_idx, head_idx))
                elif is_position_focused(head_attention):
                    head_patterns["position_heads"].append((layer_idx, head_idx))
    
    # 统计结果
    for pattern_type, heads in head_patterns.items():
        print(f"{pattern_type}: {len(set(heads))} unique heads")
        most_common = Counter(heads).most_common(3)
        print(f"  Most common: {most_common}")
    
    return head_patterns