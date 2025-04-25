from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

def analyze_ner_errors(true_labels, pred_labels, texts, id2tag=None):
    """分析NER错误类型"""
    errors = {
        "boundary_errors": [],  # 边界错误
        "type_errors": [],      # 类型错误
        "missing_entities": [], # 漏检
        "false_entities": []    # 误检
    }
    
    for i, (true, pred, text) in enumerate(zip(true_labels, pred_labels, texts)):
        # 提取真实实体
        true_entities = []
        entity = {"start": None, "end": None, "type": None}
        
        for j, tag in enumerate(true):
            if tag.startswith("B-"):
                if entity["start"] is not None:
                    true_entities.append(entity.copy())
                entity = {"start": j, "end": j, "type": tag[2:], "text": text[j]}
            elif tag.startswith("I-") and entity["start"] is not None and tag[2:] == entity["type"]:
                entity["end"] = j
                entity["text"] += text[j]
            elif entity["start"] is not None:
                true_entities.append(entity.copy())
                entity = {"start": None, "end": None, "type": None}
        
        if entity["start"] is not None:
            true_entities.append(entity.copy())
        
        # 提取预测实体
        pred_entities = []
        entity = {"start": None, "end": None, "type": None}
        
        for j, tag in enumerate(pred):
            if tag.startswith("B-"):
                if entity["start"] is not None:
                    pred_entities.append(entity.copy())
                entity = {"start": j, "end": j, "type": tag[2:], "text": text[j]}
            elif tag.startswith("I-") and entity["start"] is not None and tag[2:] == entity["type"]:
                entity["end"] = j
                entity["text"] += text[j]
            elif entity["start"] is not None:
                pred_entities.append(entity.copy())
                entity = {"start": None, "end": None, "type": None}
        
        if entity["start"] is not None:
            pred_entities.append(entity.copy())
        
        # 分析错误
        for t_entity in true_entities:
            found = False
            for p_entity in pred_entities:
                # 完全匹配
                if t_entity["start"] == p_entity["start"] and t_entity["end"] == p_entity["end"]:
                    if t_entity["type"] != p_entity["type"]:
                        # 类型错误
                        errors["type_errors"].append({
                            "text": t_entity["text"],
                            "true_type": t_entity["type"],
                            "pred_type": p_entity["type"],
                            "sentence": "".join(text)
                        })
                    found = True
                    break
                # 边界重叠但不完全相同
                elif (t_entity["start"] <= p_entity["end"] and t_entity["end"] >= p_entity["start"]):
                    # 边界错误
                    errors["boundary_errors"].append({
                        "true_entity": t_entity,
                        "pred_entity": p_entity,
                        "sentence": "".join(text)
                    })
                    found = True
                    break
            
            if not found:
                # 漏检
                errors["missing_entities"].append({
                    "entity": t_entity,
                    "sentence": "".join(text)
                })
        
        # 检查误检(预测有但实际没有的实体)
        for p_entity in pred_entities:
            found = False
            for t_entity in true_entities:
                if (t_entity["start"] <= p_entity["end"] and t_entity["end"] >= p_entity["start"]):
                    found = True
                    break
            
            if not found:
                # 误检
                errors["false_entities"].append({
                    "entity": p_entity,
                    "sentence": "".join(text)
                })
    
    # 打印错误统计
    print(f"边界错误: {len(errors['boundary_errors'])}")
    print(f"类型错误: {len(errors['type_errors'])}")
    print(f"漏检: {len(errors['missing_entities'])}")
    print(f"误检: {len(errors['false_entities'])}")
    
    return errors