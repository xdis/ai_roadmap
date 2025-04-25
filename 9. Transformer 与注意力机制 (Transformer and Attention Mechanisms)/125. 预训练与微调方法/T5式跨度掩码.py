def span_corruption(text, tokenizer, mean_span_length=3):
    """T5风格的跨度掩码"""
    tokens = tokenizer.tokenize(text)
    
    # 决定要掩盖的跨度
    spans = []
    current_span = []
    
    # 简化的跨度选择逻辑
    for i, token in enumerate(tokens):
        if random.random() < 0.15:  # 15%概率开始新跨度
            current_span.append(i)
            # 连续掩盖几个标记形成跨度
            span_length = np.random.poisson(mean_span_length)
            for j in range(1, span_length):
                if i + j < len(tokens):
                    current_span.append(i + j)
            
            if current_span:
                spans.append(current_span)
                current_span = []
    
    # 创建输入-输出对
    corrupted_text = []
    target_text = []
    
    for i, token in enumerate(tokens):
        if any(i in span for span in spans):
            span_idx = next(idx for idx, span in enumerate(spans) if i in span)
            if i == min(spans[span_idx]):
                sentinel = f"<extra_id_{span_idx}>"
                corrupted_text.append(sentinel)
            
            # 添加到目标
            if i == min(spans[span_idx]):
                target_text.append(f"<extra_id_{span_idx}>")
            target_text.append(token)
        else:
            corrupted_text.append(token)
    
    # 添加结束标记到目标
    if spans:
        target_text.append(f"<extra_id_{len(spans)}>")
    
    return tokenizer.convert_tokens_to_string(corrupted_text), tokenizer.convert_tokens_to_string(target_text)