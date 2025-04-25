class MultiHopQASystem:
    def __init__(self):
        # 初始化基本QA模型
        self.qa_model = QuestionAnsweringSystem()
        # 初始化文档检索器
        self.retriever = RetrievalAugmentedQA(retriever_type="tfidf")
    
    def decompose_question(self, question):
        """将复杂问题分解为多个子问题(简化示例，实际中可使用LLM)"""
        # 简单规则示例
        if "之前" in question or "之后" in question:
            parts = question.split("之前") if "之前" in question else question.split("之后")
            q1 = parts[0] + "？"
            q2 = "之前".join(parts) if "之前" in question else "之后".join(parts)
            return [q1, q2]
        # 更复杂的分解需要更高级的NLP技术
        return [question]  # 如果无法分解，则返回原问题
    
    def answer_multi_hop(self, question, documents):
        """多跳问答"""
        # 添加文档到检索器
        self.retriever.add_documents(documents)
        
        # 1. 问题分解
        sub_questions = self.decompose_question(question)
        
        # 2. 逐步回答子问题
        context_so_far = ""
        answers = []
        
        for i, sub_q in enumerate(sub_questions):
            print(f"子问题 {i+1}: {sub_q}")
            
            # 结合已有答案构建上下文
            if i > 0:
                # 检索相关文档并加入上下文
                retrieved_docs = self.retriever.retrieve_documents(sub_q, top_k=2)
                retrieval_context = " ".join([doc for doc, _ in retrieved_docs])
                full_context = context_so_far + " " + retrieval_context
            else:
                # 第一个问题直接检索
                retrieved_docs = self.retriever.retrieve_documents(sub_q, top_k=2)
                full_context = " ".join([doc for doc, _ in retrieved_docs])
            
            # 回答子问题
            answer_obj = self.qa_model.answer_question(sub_q, full_context)
            sub_answer = answer_obj["answer"]
            answers.append(sub_answer)
            
            # 更新累积上下文
            context_so_far += f" {sub_q} {sub_answer}."
            print(f"子答案: {sub_answer}")
        
        # 3. 合成最终答案(简化处理)
        if len(answers) == 1:
            return answers[0]
        else:
            # 使用最后一个答案作为最终答案
            return answers[-1]

# 使用示例
def test_multi_hop_qa():
    documents = [
        "阿尔伯特·爱因斯坦于1879年3月14日出生于德国乌尔姆。",
        "爱因斯坦在1905年发表了特殊相对论。",
        "玛丽·居里于1867年11月7日出生于华沙。",
        "居里夫人在1903年获得诺贝尔物理学奖，1911年获得诺贝尔化学奖。",
        "爱因斯坦于1921年获得了诺贝尔物理学奖。",
        "爱因斯坦在1955年4月18日去世，享年76岁。"
    ]
    
    multi_hop_system = MultiHopQASystem()
    
    questions = [
        "爱因斯坦出生于哪里？",  # 单跳问题
        "爱因斯坦获得诺贝尔奖之前发表了什么重要理论？"  # 多跳问题
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        answer = multi_hop_system.answer_multi_hop(question, documents)
        print(f"最终答案: {answer}")
        print("-" * 50)