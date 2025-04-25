def augment_data(sentences, tags, entity_dict):
    """使用实体替换进行数据增强"""
    augmented_sentences = []
    augmented_tags = []
    
    for sentence, tag_seq in zip(sentences, tags):
        # 查找实体位置
        entities = []
        current_entity = {"start": -1, "end": -1, "type": ""}
        
        for i, tag in enumerate(tag_seq):
            if tag.startswith("B-"):
                if current_entity["start"] != -1:
                    entities.append(current_entity.copy())
                current_entity = {"start": i, "end": i, "type": tag[2:]}
            elif tag.startswith("I-") and current_entity["start"] != -1:
                if tag[2:] == current_entity["type"]:
                    current_entity["end"] = i
            elif current_entity["start"] != -1:
                entities.append(current_entity.copy())
                current_entity = {"start": -1, "end": -1, "type": ""}
        
        if current_entity["start"] != -1:
            entities.append(current_entity)
        
        # 对每个实体执行替换
        for entity in entities:
            if entity["type"] in entity_dict:
                replacements = entity_dict[entity["type"]]
                for replacement in replacements:
                    # 创建新句子和标签序列
                    new_sentence = sentence.copy()
                    new_tags = tag_seq.copy()
                    
                    # 替换实体
                    orig_length = entity["end"] - entity["start"] + 1
                    new_length = len(replacement)
                    
                    # 替换词语
                    new_sentence[entity["start"]:entity["start"]+1] = replacement
                    if orig_length > 1:
                        del new_sentence[entity["start"]+1:entity["start"]+orig_length]
                    
                    # 调整标签
                    new_tags[entity["start"]] = f"B-{entity['type']}"
                    for j in range(1, new_length):
                        if entity["start"]+j < len(new_tags):
                            new_tags[entity["start"]+j] = f"I-{entity['type']}"
                        else:
                            new_tags.append(f"I-{entity['type']}")
                    
                    if orig_length > 1:
                        del new_tags[entity["start"]+new_length:entity["start"]+orig_length]
                    
                    augmented_sentences.append(new_sentence)
                    augmented_tags.append(new_tags)
    
    return augmented_sentences, augmented_tags