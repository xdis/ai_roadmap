from rank_bm25 import BM25Okapi

# 创建语料库
corpus = [
    "北京是中国的首都",
    "上海是中国最大的城市",
    "深圳是中国的经济特区"
]
tokenized_corpus = [doc.split() for doc in corpus]

# 创建BM25模型
bm25 = BM25Okapi(tokenized_corpus)

# 检索相关文档
query = "中国首都"
tokenized_query = query.split()
doc_scores = bm25.get_scores(tokenized_query)

# 获取排序结果
sorted_indices = np.argsort(doc_scores)[::-1]
for idx in sorted_indices:
    print(f"文档: {corpus[idx]}, 分数: {doc_scores[idx]}")