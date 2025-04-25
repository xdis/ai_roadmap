# 链式思考提示示例

questions = [
    "如果一个公司第一年亏损100万，第二年盈利300万，第三年亏损50万，总体盈亏是多少？",
    "一个家庭打算装修新房，客厅40平方米，卧室30平方米，卫生间10平方米，如果装修费用是每平方米800元，总共需要多少钱？"
]

def chain_of_thought_prompt(question):
    return f"""请一步步思考下面的问题：
{question}

让我们逐步解决：
1."""

# 为每个问题生成链式思考回答
for question in questions:
    prompt = chain_of_thought_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"问题: {question}\n\n{answer}\n{'='*50}")