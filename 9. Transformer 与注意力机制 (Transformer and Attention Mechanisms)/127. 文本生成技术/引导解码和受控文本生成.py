from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

# 简化版的属性引导生成
def attribute_guided_generation(model, tokenizer, prompt, attribute_words, strength=0.5):
    """
    使用属性词汇引导生成方向
    - attribute_words: 想要增强的关键词列表
    - strength: 引导强度
    """
    # 将属性词转换为ID
    attribute_ids = []
    for word in attribute_words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        attribute_ids.extend(ids)
    
    attribute_ids = torch.tensor(attribute_ids)
    
    # 输入编码
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    
    # 生成参数
    max_length = 50
    
    for _ in range(max_length):
        # 获取模型输出
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :].squeeze()
        
        # 原始概率分布
        orig_probs = F.softmax(next_token_logits, dim=-1)
        
        # 提升属性词汇的概率
        for attr_id in attribute_ids:
            next_token_logits[attr_id] += strength
        
        # 更新概率分布
        modified_probs = F.softmax(next_token_logits, dim=-1)
        
        # 采样下一个标记
        next_token = torch.multinomial(modified_probs, num_samples=1).unsqueeze(0)
        
        # 添加到序列
        input_ids = torch.cat((input_ids, next_token), dim=1)
        
        # 检查是否生成了EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# 使用示例
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "未来的技术发展将"
attributes = ["创新", "可持续", "人工智能"]

guided_text = attribute_guided_generation(model, tokenizer, prompt, attributes)
print(guided_text)