from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 设置检索系统
# 创建检索模型和向量数据库
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = retrieval_model.get_sentence_embedding_dimension()

# 创建向量索引
index = faiss.IndexFlatIP(dimension)

# 2. 准备知识库
documents = [
    "巴黎是法国的首都，也是最大的城市。",
    "艾菲尔铁塔高324米，建于1889年。",
    "莫奈是印象派代表画家，代表作有《日出·印象》。",
    # 更多文档...
]

# 3. 构建索引
doc_embeddings = retrieval_model.encode(documents)
index.add(np.array(doc_embeddings))

# 4. 检索增强生成
def retrieve_and_generate(query, model, tokenizer, top_k=3):
    # 检索相关文档
    query_embedding = retrieval_model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    
    # 获取相关文档
    retrieved_docs = [documents[idx] for idx in indices[0]]
    
    # 构建增强提示
    context = "\n".join(retrieved_docs)
    prompt = f"""已知信息:
{context}

基于上述信息，请回答: {query}
"""
    
    # 生成回答
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 使用示例
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

query = "艾菲尔铁塔有多高？"
answer = retrieve_and_generate(query, model, tokenizer)
print(f"问题: {query}\n回答: {answer}")