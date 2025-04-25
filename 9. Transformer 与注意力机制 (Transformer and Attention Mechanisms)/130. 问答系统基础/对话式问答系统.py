class ConversationalQA:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.reader_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reader_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reader_model.to(self.device)
        
        # 对话历史
        self.conversation_history = []
        # 文档知识库
        self.knowledge_base = []
        
    def add_to_knowledge_base(self, documents):
        """添加文档到知识库"""
        self.knowledge_base.extend(documents)
    
    def process_history(self, current_question):
        """处理对话历史，解决指代消解问题"""
        # 简化的指代消解
        # 实际应用中可以使用更复杂的NLP技术
        
        if not self.conversation_history:
            return current_question
        
        # 简单规则处理代词
        if any(word in current_question.lower() for word in ["他", "她", "它", "它们", "这个", "那个"]):
            last_question, last_answer = self.conversation_history[-1]
            
            # 从上一个问题中提取可能的实体(简化处理)
            entities = []
            for entity in ["爱因斯坦", "居里夫人", "特殊相对论", "诺贝尔奖"]:
                if entity in last_question or entity in last_answer:
                    entities.append(entity)
            
            # 替换代词
            if entities:
                for pronoun in ["他", "她", "它", "它们", "这个", "那个"]:
                    if pronoun in current_question:
                        # 简单替换第一个代词
                        return current_question.replace(pronoun, entities[0], 1)
        
        return current_question
    
    def retrieve_relevant_context(self, question, top_k=2):
        """检索与问题相关的文档(简化版)"""
        if not self.knowledge_base:
            return ""
            
        # 简化的相关性评分(实际应用中使用更复杂的检索)
        scores = []
        for doc in self.knowledge_base:
            # 计算问题和文档的词重叠
            q_words = set(question.lower().split())
            d_words = set(doc.lower().split())
            overlap = len(q_words.intersection(d_words))
            scores.append(overlap)
        
        # 获取top-k相关文档
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        relevant_docs = [self.knowledge_base[i] for i in sorted_indices if scores[i] > 0]
        
        return " ".join(relevant_docs)
    
    def answer_question(self, question):
        """回答问题并维护对话历史"""
        # 处理对话历史(指代消解)
        processed_question = self.process_history(question)
        
        # 检索相关上下文
        context = self.retrieve_relevant_context(processed_question)
        
        if not context:
            answer = "对不起，我没有足够的信息回答这个问题。"
        else:
            # 使用阅读理解模型提取答案
            inputs = self.reader_tokenizer(
                processed_question,
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
            
            if end_idx < start_idx:
                answer = "对不起，我无法找到合适的答案。"
            else:
                # 获取答案文本
                answer_tokens = input_ids[0][start_idx:end_idx+1]
                answer = self.reader_tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # 更新对话历史
        self.conversation_history.append((question, answer))
        
        return {
            "original_question": question,
            "processed_question": processed_question,
            "answer": answer,
            "context_used": context if context else None
        }
    
    def get_conversation_history(self):
        """返回对话历史"""
        return self.conversation_history
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
        return "对话历史已清除"

# 使用示例
def test_conversational_qa():
    conv_qa = ConversationalQA()
    
    # 添加知识库
    documents = [
        "阿尔伯特·爱因斯坦于1879年3月14日出生于德国乌尔姆。",
        "爱因斯坦在1905年发表了特殊相对论。",
        "爱因斯坦于1921年获得了诺贝尔物理学奖。",
        "爱因斯坦在1955年4月18日去世，享年76岁。",
        "玛丽·居里于1867年出生于波兰华沙。",
        "居里夫人是首位获得两次诺贝尔奖的科学家。"
    ]
    conv_qa.add_to_knowledge_base(documents)
    
    # 模拟对话
    conversation = [
        "爱因斯坦出生于哪里？",
        "他是什么时候发表特殊相对论的？",
        "他获得了什么奖项？",
        "他什么时候去世的？",
        "居里夫人是谁？"
    ]
    
    for question in conversation:
        result = conv_qa.answer_question(question)
        print(f"用户: {question}")
        print(f"系统: {result['answer']}")
        print(f"(处理后的问题: {result['processed_question']})")
        print("-" * 50)
    
    # 显示对话历史
    print("对话历史:")
    for i, (q, a) in enumerate(conv_qa.get_conversation_history()):
        print(f"轮次 {i+1} - 问: {q} | 答: {a}")