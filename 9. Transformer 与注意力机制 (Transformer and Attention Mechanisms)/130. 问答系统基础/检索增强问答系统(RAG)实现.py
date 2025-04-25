from transformers import AutoModelForQuestionAnswering, AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

class RetrievalAugmentedQA:
    def __init__(self, retriever_type="tfidf", reader_model="deepset/roberta-base-squad2"):
        # 初始化阅读器
        self.reader_tokenizer = AutoTokenizer.from_pretrained(reader_model)
        self.reader_model = AutoModelForQuestionAnswering.from_pretrained(reader_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reader_model.to(self.device)
        
        # 初始化检索器
        self.retriever_type = retriever_type
        if retriever_type == "tfidf":
            self.retriever = TfidfVectorizer(lowercase=True, stop_words="english")
        elif retriever_type == "dpr":
            self.question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            self.context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            self.question_encoder.to(self.device)
            self.context_encoder.to(self.device)
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")
            
        self.documents = []
        self.document_embeddings = None
        self.index = None
    
    def add_documents(self, documents):
        """添加文档到知识库"""
        self.documents = documents
        
        if self.retriever_type == "tfidf":
            # 构建TF-IDF矩阵
            self.document_embeddings = self.retriever.fit_transform(documents)
        elif self.retriever_type == "dpr":
            # 使用DPR编码文档
            with torch.no_grad():
                inputs = self.reader_tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeddings = self.context_encoder(**inputs).pooler_output
                self.document_embeddings = embeddings.cpu().numpy()
            
            # 创建FAISS索引
            dimension = self.document_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.document_embeddings)
    
    def retrieve_documents(self, query, top_k=3):
        """检索与查询相关的文档"""
        if self.retriever_type == "tfidf":
            # TF-IDF检索
            query_vec = self.retriever.transform([query]).toarray()[0]
            scores = np.dot(self.document_embeddings, query_vec)
            top_indices = np.argsort(scores.flatten())[-top_k:][::-1]
            
            return [(self.documents[i], scores.flatten()[i]) for i in top_indices]
        
        elif self.retriever_type == "dpr":
            # DPR检索
            with torch.no_grad():
                q_inputs = self.reader_tokenizer(query, return_tensors="pt")
                q_inputs = {k: v.to(self.device) for k, v in q_inputs.items()}
                q_embedding = self.question_encoder(**q_inputs).pooler_output.cpu().numpy()
                
                # 搜索最相似的文档
                scores, indices = self.index.search(q_embedding, top_k)
                
                return [(self.documents[int(i)], float(s)) for s, i in zip(scores[0], indices[0])]
    
    def answer_question(self, question, use_retrieved_docs=True, top_k=3):
        """回答问题"""
        if use_retrieved_docs:
            # 检索相关文档
            retrieved_docs = self.retrieve_documents(question, top_k=top_k)
            context = " ".join([doc for doc, _ in retrieved_docs])
        else:
            # 使用所有文档作为上下文
            context = " ".join(self.documents)
        
        # 使用阅读器提取答案
        inputs = self.reader_tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation="only_second",
            padding="max_length",
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.reader_model(input_ids=input_ids, attention_mask=attention_mask)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
        
        # 找到最佳答案位置
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        # 确保答案边界合理
        if end_idx < start_idx:
            # 寻找合理边界
            all_scores = start_scores.cpu().numpy()[:, None] + end_scores.cpu().numpy()[None, :]
            all_scores = np.triu(all_scores)  # 上三角矩阵
            max_idx = np.unravel_index(np.argmax(all_scores), all_scores.shape)
            start_idx, end_idx = max_idx
        
        # 获取答案文本
        answer_tokens = input_ids[0][start_idx:end_idx+1]
        answer = self.reader_tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # 计算置信分数
        score = (start_scores[0, start_idx] + end_scores[0, end_idx]).item()
        
        return {
            "answer": answer,
            "score": score,
            "retrieved_docs": retrieved_docs if use_retrieved_docs else None
        }

# 使用示例
if __name__ == "__main__":
    # 示例文档集
    documents = [
        "自然语言处理（NLP）是人工智能的一个子领域，专注于计算机与人类语言之间的交互。",
        "NLP的应用包括机器翻译、情感分析、文本摘要和问答系统等。",
        "基于Transformer架构的大型语言模型，如BERT、GPT和T5，已经在各种NLP任务上取得了显著的进展。",
        "问答系统旨在自动回答用户用自然语言提出的问题。",
        "现代问答系统通常结合了信息检索和机器阅读理解技术。",
        "2017年，Google发布了BERT模型，它在多项NLP基准测试上取得了突破性成果。"
    ]
    
    # 初始化系统
    qa_system = RetrievalAugmentedQA(retriever_type="tfidf")
    qa_system.add_documents(documents)
    
    # 测试问题
    questions = [
        "什么是自然语言处理？",
        "NLP有哪些应用？",
        "哪些模型在NLP任务上取得了进展？",
        "问答系统是什么？"
    ]
    
    for question in questions:
        result = qa_system.answer_question(question, top_k=2)
        print(f"问题: {question}")
        print(f"答案: {result['answer']}")
        print(f"分数: {result['score']:.4f}")
        print("检索的文档:")
        for doc, score in result["retrieved_docs"]:
            print(f" - {doc} (相关性: {score:.4f})")
        print("-" * 50)