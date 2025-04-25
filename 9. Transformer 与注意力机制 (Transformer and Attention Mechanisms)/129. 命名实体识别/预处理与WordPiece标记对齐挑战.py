def align_labels_with_tokens(labels, word_ids):
    """将原始标签与WordPiece标记对齐"""
    aligned_labels = []
    current_word = None
    
    for word_id in word_ids:
        # 特殊标记
        if word_id is None:
            aligned_labels.append(-100)  # 忽略特殊标记的损失
        # 开始新词
        elif word_id != current_word:
            current_word = word_id
            aligned_labels.append(labels[word_id])
        # 当前词的延续部分
        else:
            # 检查原始标签
            if labels[word_id].startswith("B-"):
                aligned_labels.append("I" + labels[word_id][1:])
            else:
                aligned_labels.append(labels[word_id])
    
    return aligned_labels

# 使用示例
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
text = "李明去北京大学"
labels = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG"]

# 分词
tokenized = tokenizer(text, return_offsets_mapping=True, return_word_ids=True)
aligned_labels = align_labels_with_tokens(labels, tokenized.word_ids())