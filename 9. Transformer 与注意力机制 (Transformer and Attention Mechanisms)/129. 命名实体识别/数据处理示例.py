def process_conll_data(filename):
    """处理CoNLL格式数据"""
    sentences = []
    labels = []
    
    sentence = []
    label = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # 假设格式是"字 标签"
                token, tag = line.split()
                sentence.append(token)
                label.append(tag)
            elif sentence:  # 句子结束
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
    
    # 处理最后一个句子
    if sentence:
        sentences.append(sentence)
        labels.append(label)
    
    return sentences, labels

def build_vocab(sentences, min_freq=1):
    """构建词汇表"""
    word_count = {}
    for sentence in sentences:
        for word in sentence:
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
    
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_count.items():
        if count >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx

def build_tag_map(labels):
    """构建标签映射"""
    tag_to_idx = {'<PAD>': 0, 'START': 1, 'STOP': 2}
    for sentence_labels in labels:
        for tag in sentence_labels:
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)
    
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    return tag_to_idx, idx_to_tag