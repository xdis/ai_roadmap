import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import pipeline

# 1. 使用pipeline实现简单问答
def simple_qa_pipeline():
    # 使用预训练问答模型创建pipeline
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2"
    )
    
    # 准备问题和上下文
    context = """
    自然语言处理（NLP）是人工智能的一个子领域，专注于计算机与人类语言之间的交互。
    它涉及开发能够理解、解释和生成人类语言的算法和模型。NLP的应用包括机器翻译、
    情感分析、文本摘要和问答系统等。近年来，基于Transformer架构的大型语言模型，
    如BERT、GPT和T5，已经在各种NLP任务上取得了显著的进展。
    """
    
    question = "NLP的应用包括哪些？"
    
    # 获取答案
    result = qa_pipeline(question=question, context=context)
    print(f"答案: {result['answer']}")
    print(f"分数: {result['score']:.4f}")
    print(f"开始位置: {result['start']}")
    print(f"结束位置: {result['end']}")
    
    return result

# 2. 从头实现问答系统
class QuestionAnsweringSystem:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def answer_question(self, question, context, max_length=384, max_answer_length=30):
        # 分词处理
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=max_length,
            truncation="only_second",
            padding="max_length",
        )
        
        # 将输入移至设备
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 模型预测
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
        
        # 获取最可能的答案位置
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        # 确保答案边界合理
        if end_idx < start_idx or end_idx - start_idx + 1 > max_answer_length:
            # 如果预测不合理，尝试找到更合理的边界
            all_scores = start_scores.cpu().numpy()[:, None] + end_scores.cpu().numpy()[None, :]
            all_scores = np.triu(all_scores)  # 上三角矩阵，确保end >= start
            all_scores[all_scores == 0] = -1e10  # 过滤无效位置
            
            # 添加长度限制
            for i in range(all_scores.shape[0]):
                for j in range(all_scores.shape[1]):
                    if j - i + 1 > max_answer_length:
                        all_scores[i, j] = -1e10
            
            max_idx = np.unravel_index(np.argmax(all_scores), all_scores.shape)
            start_idx, end_idx = max_idx
        
        # 获取答案文本
        answer_tokens = input_ids[0][start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # 计算置信分数
        score = (start_scores[0, start_idx] + end_scores[0, end_idx]).item()
        
        return {
            "answer": answer,
            "score": score,
            "start_idx": start_idx.item(),
            "end_idx": end_idx.item()
        }

# 使用示例
def run_custom_qa():
    qa_system = QuestionAnsweringSystem()
    
    context = """
    自然语言处理（NLP）是人工智能的一个子领域，专注于计算机与人类语言之间的交互。
    它涉及开发能够理解、解释和生成人类语言的算法和模型。NLP的应用包括机器翻译、
    情感分析、文本摘要和问答系统等。近年来，基于Transformer架构的大型语言模型，
    如BERT、GPT和T5，已经在各种NLP任务上取得了显著的进展。
    """
    
    questions = [
        "什么是自然语言处理？",
        "NLP的应用包括哪些？",
        "哪些模型在NLP任务上取得了显著进展？"
    ]
    
    for question in questions:
        result = qa_system.answer_question(question, context)
        print(f"问题: {question}")
        print(f"答案: {result['answer']}")
        print(f"分数: {result['score']:.4f}")
        print("-" * 50)
    
    return qa_system

# 主函数
if __name__ == "__main__":
    print("======= 使用Pipeline的问答系统 =======")
    simple_qa_pipeline()
    
    print("\n======= 自定义问答系统 =======")
    run_custom_qa()