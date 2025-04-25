def load_data(filename):
    """加载CoNLL格式数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sentences = []
    tags = []
    sentence = []
    tag = []
    
    for line in lines:
        line = line.strip()
        if line:
            word, label = line.split()
            sentence.append(word)
            tag.append(label)
        elif sentence:  # 空行表示句子边界
            sentences.append(sentence)
            tags.append(tag)
            sentence = []
            tag = []
    
    # 处理最后一个句子
    if sentence:
        sentences.append(sentence)
        tags.append(tag)
    
    return sentences, tags

# 构建词汇表和标签映射
def build_vocab(sentences, tags):
    word_to_ix = {}
    tag_to_ix = {}
    
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    
    for tag_seq in tags:
        for tag in tag_seq:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    
    return word_to_ix, tag_to_ix