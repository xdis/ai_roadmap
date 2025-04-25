class PromptEncoder(nn.Module):
    """编码用于提示调优的软提示"""
    def __init__(self, prompt_length, hidden_size):
        super().__init__()
        self.prompt_length = prompt_length
        # 初始化软提示嵌入
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(prompt_length, hidden_size)
        )
        self._init_prompt_embeddings()
        
    def _init_prompt_embeddings(self):
        # 使用截断正态分布初始化
        nn.init.normal_(self.prompt_embeddings, std=0.02)
        
    def forward(self, input_embeddings):
        batch_size = input_embeddings.shape[0]
        # 重复扩展软提示以匹配批次大小
        prompts = self.prompt_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        # 将提示拼接到输入嵌入之前
        return torch.cat([prompts, input_embeddings], dim=1)

# 应用提示调优
def apply_prompt_tuning(model, tokenizer, prompt_length=20):
    """应用提示调优到预训练模型"""
    # 冻结原始模型参数
    for param in model.parameters():
        param.requires_grad = False
        
    # 创建提示编码器
    prompt_encoder = PromptEncoder(
        prompt_length=prompt_length,
        hidden_size=model.config.hidden_size
    )
    
    # 保存原始嵌入层forward方法
    original_embed_forward = model.get_input_embeddings().forward
    
    # 创建新的forward方法
    def new_embed_forward(input_ids=None, **kwargs):
        # 获取原始嵌入
        inputs_embeds = original_embed_forward(input_ids)
        # 添加提示嵌入
        return prompt_encoder(inputs_embeds)
    
    # 替换嵌入层forward方法
    model.get_input_embeddings().forward = new_embed_forward
    
    # 为提示标记调整attention_mask
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            # 扩展attention_mask以包括提示标记
            prompt_mask = torch.ones(
                (attention_mask.shape[0], prompt_length), 
                device=attention_mask.device
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
            kwargs["attention_mask"] = attention_mask
        return super(type(model), model).prepare_inputs_for_generation(input_ids, **kwargs)
    
    # 添加prepare_inputs_for_generation方法
    model.prepare_inputs_for_generation = types.MethodType(
        prepare_inputs_for_generation, model
    )
    
    return model, prompt_encoder