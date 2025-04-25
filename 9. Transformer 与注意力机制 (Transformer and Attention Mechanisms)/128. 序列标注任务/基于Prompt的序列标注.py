from transformers import T5ForConditionalGeneration, T5Tokenizer

# 加载模型
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# 将NER任务转化为生成任务
def create_prompt(text):
    return f"Find entities in text: {text}"

def parse_generated_entities(generated_text):
    # 解析生成的实体标注结果
    entities = []
    for line in generated_text.split('\n'):
        if ':' in line:
            try:
                entity_type, entity_text = line.split(':', 1)
                entities.append({
                    'type': entity_type.strip(),
                    'text': entity_text.strip()
                })
            except:
                continue
    return entities

# 使用示例
text = "苹果公司由史蒂夫·乔布斯于1976年创建。"
prompt = create_prompt(text)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=150)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 生成结果示例: "ORGANIZATION: 苹果公司\nPERSON: 史蒂夫·乔布斯\nDATE: 1976年"
entities = parse_generated_entities(generated_text)
print(entities)