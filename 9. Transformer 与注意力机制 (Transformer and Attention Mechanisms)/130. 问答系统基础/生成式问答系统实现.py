from transformers import T5ForConditionalGeneration, T5Tokenizer

class GenerativeQASystem:
    def __init__(self, model_name="t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def answer_question(self, question, context, max_length=64):
        """生成问题的答案"""
        # T5模型需要特定格式的输入
        input_text = f"question: {question} context: {context}"
        
        # 编码输入
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 生成答案
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        
        # 解码答案
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer

# 使用示例
def test_generative_qa():
    qa_system = GenerativeQASystem()
    
    context = """
    自然语言处理（NLP）是人工智能的一个子领域，专注于计算机与人类语言之间的交互。
    它涉及开发能够理解、解释和生成人类语言的算法和模型。NLP的应用包括机器翻译、
    情感分析、文本摘要和问答系统等。近年来，基于Transformer架构的大型语言模型，
    如BERT、GPT和T5，已经在各种NLP任务上取得了显著的进展。
    """
    
    questions = [
        "什么是自然语言处理？",
        "NLP的应用包括哪些？",
        "哪些模型在NLP任务上取得了进展？"
    ]
    
    for question in questions:
        answer = qa_system.answer_question(question, context)
        print(f"问题: {question}")
        print(f"答案: {answer}")
        print("-" * 50)
    
    return qa_system