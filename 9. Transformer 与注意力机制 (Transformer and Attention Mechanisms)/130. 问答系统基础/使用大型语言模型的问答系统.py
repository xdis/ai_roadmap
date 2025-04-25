from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMQuestionAnswerer:
    def __init__(self, model_name="gpt2-large"):  # 实际应用中可以使用更强大的模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 如果tokenizer没有设置pad_token，则设置为eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def answer_question(self, question, context=None, max_length=100):
        """使用LLM生成答案"""
        if context:
            prompt = f"上下文：{context}\n\n问题：{question}\n\n答案："
        else:
            prompt = f"问题：{question}\n\n答案："
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 生成答案
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=len(input_ids[0]) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 解码答案并删除输入部分
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_text[len(prompt):]
        
        return answer