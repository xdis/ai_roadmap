def distant_supervision_ner(texts, entity_dict):
    """使用远程监督自动标注NER数据"""
    labeled_sentences = []
    labeled_tags = []
    
    for text in texts:
        tags = ["O"] * len(text)
        
        # 使用词典进行自动标注
        for entity_type, entities in entity_dict.items():
            for entity in entities:
                # 查找所有出现位置
                start = 0
                while start < len(text):
                    pos = text.find(entity, start)
                    if pos == -1:
                        break
                    
                    # 标注实体
                    tags[pos] = f"B-{entity_type}"
                    for i in range(pos + 1, pos + len(entity)):
                        if i < len(text):
                            tags[i] = f"I-{entity_type}"
                    
                    start = pos + 1
        
        # 解决重叠问题(保留最后一次标注)
        labeled_sentences.append(list(text))
        labeled_tags.append(tags)
    
    return labeled_sentences, labeled_tags