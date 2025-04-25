# 条件文本生成示例

# 1. 情感控制生成
prompt_templates = {
    "positive": "以积极乐观的语气写一篇关于未来的短文：\n",
    "negative": "以消极悲观的语气写一篇关于未来的短文：\n",
    "neutral": "以客观中立的语气写一篇关于未来的短文：\n"
}

for tone, prompt in prompt_templates.items():
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    print(f"【{tone}】: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")

# 2. 多步指令提示
def generate_with_instruction(model, tokenizer, instruction, max_length=200):
    prompt = f"指令: {instruction}\n回答: "
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 使用示例
instructions = [
    "写一封商业邮件，请求延长项目截止日期",
    "解释量子计算的基本原理，让一个10岁的孩子能理解",
    "写一首关于春天的短诗"
]

for instruction in instructions:
    result = generate_with_instruction(model, tokenizer, instruction)
    print(f"指令: {instruction}\n回答: {result}\n{'='*50}")