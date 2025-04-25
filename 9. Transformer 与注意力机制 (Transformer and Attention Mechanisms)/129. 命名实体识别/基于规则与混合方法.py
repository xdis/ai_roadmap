def rule_based_ner(text, entity_dict):
    """基于词典和规则的NER"""
    entities = []
    
    # 1. 基于词典匹配
    for entity_type, terms in entity_dict.items():
        for term in terms:
            # 查找所有出现位置
            start = 0
            while start < len(text):
                pos = text.find(term, start)
                if pos == -1:
                    break
                
                entities.append({
                    "text": term,
                    "start": pos,
                    "end": pos + len(term),
                    "type": entity_type
                })
                
                start = pos + 1
    
    # 2. 基于规则匹配(示例:日期格式)
    import re
    date_patterns = [
        r'\d{4}年\d{1,2}月\d{1,2}日',
        r'\d{4}-\d{1,2}-\d{1,2}',
        r'\d{4}/\d{1,2}/\d{1,2}'
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "type": "DATE"
            })
    
    # 解决重叠问题(优先选择更长的实体)
    entities.sort(key=lambda x: (x["start"], -len(x["text"])))
    
    filtered_entities = []
    occupied = set()
    
    for entity in entities:
        # 检查是否与已选实体重叠
        overlap = False
        for pos in range(entity["start"], entity["end"]):
            if pos in occupied:
                overlap = True
                break
        
        if not overlap:
            filtered_entities.append(entity)
            # 标记已占用的位置
            for pos in range(entity["start"], entity["end"]):
                occupied.add(pos)
    
    return filtered_entities