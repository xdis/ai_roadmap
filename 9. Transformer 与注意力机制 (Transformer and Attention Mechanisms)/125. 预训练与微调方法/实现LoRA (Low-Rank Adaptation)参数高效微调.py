import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA矩阵
        self.lora_A = nn.Parameter(torch.zeros((rank, in_dim)))
        self.lora_B = nn.Parameter(torch.zeros((out_dim, rank)))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # 返回LoRA增量部分
        return (self.lora_B @ self.lora_A @ x.T).T * self.scaling

def add_lora_to_model(model, lora_rank=8, lora_alpha=32, target_modules=["q_proj", "v_proj"]):
    """向模型添加LoRA层"""
    # 跟踪原始前向传播函数
    orig_forwards = {}
    
    for name, module in model.named_modules():
        # 检查模块名称是否包含目标字符串
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                in_dim = module.in_features
                out_dim = module.out_features
                
                # 创建LoRA层
                lora_layer = LoRALayer(in_dim, out_dim, lora_rank, lora_alpha)
                
                # 保存原始前向传播
                orig_forward = module.forward
                orig_forwards[name] = orig_forward
                
                # 创建新的前向传播函数
                def make_forward(orig_forward, lora_layer):
                    def lora_forward(x):
                        # 原始输出 + LoRA输出
                        return orig_forward(x) + lora_layer(x)
                    return lora_forward
                
                # 设置新的前向传播
                module.forward = make_forward(orig_forward, lora_layer)
                
                # 将LoRA层添加为模块属性
                module.lora_layer = lora_layer
                
    # 冻结原始模型参数
    for param in model.parameters():
        param.requires_grad = False
        
    # 解冻LoRA参数
    for name, module in model.named_modules():
        if hasattr(module, 'lora_layer'):
            for param in module.lora_layer.parameters():
                param.requires_grad = True
                
    return model, orig_forwards

# 使用示例
def fine_tune_with_lora():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model, _ = add_lora_to_model(model)
    
    # 现在只有LoRA参数可训练
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params}, 总参数: {all_params}, 比例: {trainable_params/all_params:.4%}")
    
    # 继续常规微调...