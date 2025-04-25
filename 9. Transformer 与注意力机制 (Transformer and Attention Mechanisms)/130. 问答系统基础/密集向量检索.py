from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# 加载编码器
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 编码文档
corpus = ["北京是中国的首都", "上海是中国最大的城市", "深圳是中国的经济特区"]
corpus_embeddings = model.encode(corpus)

# 创建索引
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.asarray(corpus_embeddings))

# 检索相关文档
query = "中国首都在哪里?"
query_embedding = model.encode([query])
distances, indices = index.search(np.asarray(query_embedding), k=2)

# 输出检索结果
for i, idx in enumerate(indices[0]):
    print(f"第{i+1}相关文档: {corpus[idx]} (距离: {distances[0][i]})")