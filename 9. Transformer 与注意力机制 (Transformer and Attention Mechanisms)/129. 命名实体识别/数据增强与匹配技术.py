def entity_replacement_augmentation(sentences, tags, entity_dict):
    """使用词典替换创建数据增强样本"""
    augmented_sentences = []
    augmented_tags = []
    
    for sentence, tag_seq in zip(sentences, tags):
        # 找出句子中的所有实体
        entities = []
        current_entity = {"start": -1, "end": -1, "type": None, "text": ""}
        
        for i, tag in enumerate(tag_seq):
            if tag.startswith("B-"):
                if current_entity["start"] != -1:
                    entities.append(current_entity.copy())
                entity_type = tag[2:]
                current_entity = {"start": i, "end": i, "type": entity_type, "text": sentence[i]}
            elif tag.startswith("I-") and current_entity["start"] != -1:
                if tag[2:] == current_entity["type"]:
                    current_entity["end"] = i
                    current_entity["text"] += sentence[i]
            elif current_entity["start"] != -1:
                entities.append(current_entity.copy())
                current_entity = {"start": -1, "end": -1, "type": None, "text": ""}
        
        if current_entity["start"] != -1:
            entities.append(current_entity)
        
        # 替换实体创建新样本
        for entity in entities:
            if entity["type"] in entity_dict:
                replacement_candidates = entity_dict[entity["type"]]
                
                # 每个候选替换创建一个新样本
                for replacement in replacement_candidates:
                    if replacement == entity["text"]:
                        continue  # 跳过相同实体
                    
                    # 创建新句子和标签
                    new_sentence = sentence.copy()
                    new_tags = tag_seq.copy()
                    
                    # 替换实体文本
                    for j in range(entity["start"], entity["end"] + 1):
                        if j == entity["start"]:
                            new_sentence[j] = replacement  # 替换为整个新实体
                            # 其他位置留空，后面会删除
                        else:
                            new_sentence[j] = ""
                    
                    # 压缩空白项
                    new_sentence = [w for w in new_sentence if w]
                    
                    # 调整标签序列
                    new_tags = new_tags[:entity["start"]]
                    new_tags.append(f"B-{entity['type']}")
                    new_tags.extend(f"I-{entity['type']}" for _ in range(len(replacement) - 1))
                    new_tags.extend(tag_seq[entity["end"] + 1:])
                    
                    augmented_sentences.append(new_sentence)
                    augmented_tags.append(new_tags)
    
    return augmented_sentences, augmented_tags