class Adapter(nn.Module):
    """Transformer层适配器"""
    def __init__(self, input_dim, adapter_dim):
        super().__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # 初始化为接近恒等映射
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
        
    def forward(self, hidden_states):
        residual = hidden_states
        x = self.down_project(hidden_states)
        x = self.activation(x)
        x = self.up_project(x)
        output = self.layer_norm(residual + x)
        return output

def add_adapters_to_model(model, adapter_dim=64):
    """向BERT模型添加适配器"""
    for layer in model.encoder.layer:
        # 保存原始前馈神经网络
        orig_output = layer.output
        hidden_dim = orig_output.dense.weight.size(0)
        
        # 创建并注入适配器
        adapter = Adapter(hidden_dim, adapter_dim)
        
        # 修改前向传播
        def custom_forward(self, hidden_states):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.activation(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            hidden_states = adapter(hidden_states)  # 添加适配器
            return hidden_states
            
        # 替换前向传播方法
        import types
        layer.output.forward = types.MethodType(custom_forward, layer.output)
    
    # 冻结原始模型参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻适配器参数
    for name, param in model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True