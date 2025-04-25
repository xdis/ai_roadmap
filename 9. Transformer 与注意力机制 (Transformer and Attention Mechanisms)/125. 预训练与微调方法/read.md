# 预训练与微调方法：从零开始掌握

## 1. 基础概念理解

### 什么是预训练与微调？

**预训练(Pretraining)**是指在大规模无标注或自监督数据上训练模型，学习通用的语言或视觉表示。

**微调(Fine-tuning)**是指在预训练模型的基础上，使用特定任务的数据进一步训练，使模型适应具体任务需求。

这种"预训练-微调"范式彻底改变了深度学习领域：

```
预训练阶段：大规模通用数据 → 通用知识表示
     ↓
微调阶段：特定任务数据 → 特定任务优化
```

### 核心原理与优势

| **传统方法** | **预训练-微调方法** |
|------------|-------------------|
| 从零训练每个任务模型 | 先学习通用知识，再专注特定任务 |
| 需要大量任务特定数据 | 可以使用少量任务数据 |
| 训练时间长，资源消耗大 | 高效利用计算资源，快速部署 |
| 性能受限于可用标注数据 | 可充分利用大量无标注数据 |

### 预训练目标类型

根据模型架构和任务类型，预训练目标主要分为：

1. **掩码语言建模(MLM)**：代表是BERT，随机掩盖输入的一部分标记，让模型预测被掩盖的内容
   ```
   输入: "The [MASK] brown fox [MASK] over the lazy dog"
   任务: 预测[MASK]处应为"quick"和"jumps"
   ```

2. **因果语言建模(CLM)**：代表是GPT系列，预测序列中下一个标记
   ```
   输入: "The quick brown fox"
   任务: 预测下一个词"jumps"
   ```

3. **序列到序列预训练**：代表是T5，结合各种任务形式进行预训练
   ```
   输入: "translate English to French: Hello world"
   输出: "Bonjour le monde"
   ```

4. **对比学习**：代表是CLIP，通过对比正负样本对学习表示
   ```
   任务: 最大化配对图像-文本的相似度，最小化非配对样本相似度
   ```

### 主要微调策略

1. **全参数微调**：调整预训练模型的所有参数
2. **特征提取**：冻结预训练层，仅训练任务特定层
3. **分层微调**：对不同层应用不同学习率
4. **参数高效微调**：仅调整少量参数(如LoRA、Adapters)

## 2. 技术细节探索

### 预训练架构详解

Transformer预训练模型主要分为三类架构：

```
┌─ 编码器(Encoder-only): BERT, RoBERTa
│   特点：双向上下文，适合理解任务
│
├─ 解码器(Decoder-only): GPT系列, LLaMA
│   特点：自回归生成，适合生成任务
│
└─ 编码器-解码器(Encoder-Decoder): T5, BART
    特点：序列转换，适合翻译、摘要等任务
```

### 预训练目标深入分析

#### 掩码语言模型(MLM)

BERT风格预训练核心代码示例：

```python
def create_masked_input(tokens, tokenizer, mask_prob=0.15):
    """创建MLM训练样本"""
    input_ids = tokens.clone()
    labels = tokens.clone()
    
    # 创建概率掩码
    probability_matrix = torch.full(labels.shape, mask_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    
    # 掩盖标记
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 设置非掩码位置为-100(忽略)
    
    # 80%时间用[MASK]替换
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # 10%时间随机替换为其他标记
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    
    # 10%时间保持不变
    
    return input_ids, labels
```

#### 因果语言模型(CLM)

GPT风格预训练原理：

```python
def causal_language_modeling_loss(logits, input_ids):
    """计算因果语言建模损失"""
    # 移位输入作为目标：预测下一个标记
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # 计算交叉熵损失
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1))
    return loss
```

#### T5式跨度掩码

```python
def span_corruption(text, tokenizer, mean_span_length=3):
    """T5风格的跨度掩码"""
    tokens = tokenizer.tokenize(text)
    
    # 决定要掩盖的跨度
    spans = []
    current_span = []
    
    # 简化的跨度选择逻辑
    for i, token in enumerate(tokens):
        if random.random() < 0.15:  # 15%概率开始新跨度
            current_span.append(i)
            # 连续掩盖几个标记形成跨度
            span_length = np.random.poisson(mean_span_length)
            for j in range(1, span_length):
                if i + j < len(tokens):
                    current_span.append(i + j)
            
            if current_span:
                spans.append(current_span)
                current_span = []
    
    # 创建输入-输出对
    corrupted_text = []
    target_text = []
    
    for i, token in enumerate(tokens):
        if any(i in span for span in spans):
            span_idx = next(idx for idx, span in enumerate(spans) if i in span)
            if i == min(spans[span_idx]):
                sentinel = f"<extra_id_{span_idx}>"
                corrupted_text.append(sentinel)
            
            # 添加到目标
            if i == min(spans[span_idx]):
                target_text.append(f"<extra_id_{span_idx}>")
            target_text.append(token)
        else:
            corrupted_text.append(token)
    
    # 添加结束标记到目标
    if spans:
        target_text.append(f"<extra_id_{len(spans)}>")
    
    return tokenizer.convert_tokens_to_string(corrupted_text), tokenizer.convert_tokens_to_string(target_text)
```

### 微调技术详解

#### 全参数微调

最直接的方法，但计算资源需求大：

```python
def full_fine_tuning(pretrained_model, train_dataloader, optimizer, num_epochs):
    """全参数微调"""
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            inputs, labels = batch
            outputs = pretrained_model(**inputs)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

#### 分层学习率微调

基于层深度分配不同学习率：

```python
def get_layer_wise_learning_rates(model, base_lr=5e-5, decay_factor=0.9):
    """为不同层设置不同学习率"""
    parameters = []
    num_layers = len(model.encoder.layer)
    
    # 输出层使用较高学习率
    parameters.append({
        'params': model.classifier.parameters(),
        'lr': base_lr
    })
    
    # 编码器层使用递减学习率
    for i in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay_factor ** (num_layers - i))
        parameters.append({
            'params': model.encoder.layer[i].parameters(),
            'lr': layer_lr
        })
    
    # 嵌入层使用最低学习率
    parameters.append({
        'params': model.embeddings.parameters(),
        'lr': base_lr * (decay_factor ** (num_layers + 1))
    })
    
    return parameters
```

#### Adapters适配器

插入小型可训练模块，保持大部分预训练参数不变：

```python
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
```

### 优化策略与超参数

微调中的关键超参数设置：

```python
# 常用微调超参数范围
learning_rates = [1e-5, 3e-5, 5e-5]  # 通常比预训练小
batch_sizes = [16, 32]  # 通常较小
epochs = [2, 3, 4]  # 通常不需要很多轮次

# 学习率预热和衰减
from transformers import get_linear_schedule_with_warmup

def create_optimizer_and_scheduler(model, train_steps, warmup_ratio=0.1):
    """创建优化器和学习率调度器"""
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
    warmup_steps = int(train_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=train_steps
    )
    
    return optimizer, scheduler
```

### 灾难性遗忘问题及解决方案

灾难性遗忘是指微调可能导致模型丢失预训练阶段获取的通用知识。解决方法包括：

1. **正则化技术**：通过惩罚项限制参数偏离预训练值
2. **混合任务学习**：微调时混合原始预训练任务
3. **选择性微调**：仅微调最相关的参数
4. **经验回放**：定期回顾预训练数据
5. **知识蒸馏**：保留预训练模型作为教师

## 3. 实践与实现

### 使用Hugging Face框架预训练小型Transformer

```python
from transformers import RobertaConfig, RobertaForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# 1. 准备数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 2. 训练分词器
tokenizer_train = ByteLevelBPETokenizer()
files = [f for f in dataset["train"]["text"] if f]
tokenizer_train.train(files=files, vocab_size=30522, min_frequency=2, special_tokens=[
    "<s>", "<pad>", "</s>", "<unk>", "<mask>"
])
tokenizer_train.save_model("tokenizer")

from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("tokenizer")

# 3. 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, 
                                remove_columns=["text"])

# 4. 创建小型Roberta配置
config = RobertaConfig(
    vocab_size=30522,
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=1024,
)
model = RobertaForMaskedLM(config)

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="my-small-roberta",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# 6. 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("my-small-roberta")
```

### 微调BERT模型进行文本分类

```python
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. 加载数据集
dataset = load_dataset("glue", "sst2")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2. 预处理数据
def preprocess_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 3. 加载预训练模型
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 4. 定义评估函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 5. 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 6. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# 7. 开始微调
trainer.train()
```

### 实现LoRA (Low-Rank Adaptation)参数高效微调

```python
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
```

## 4. 高级应用与变体

### 提示调优(Prompt Tuning)

添加少量可学习的提示标记：

```python
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
```

### 指令微调(Instruction Fine-tuning)

通过指令格式增强模型的理解和执行能力：

```python
# 指令微调数据格式
instruction_examples = [
    {
        "instruction": "将以下英语文本翻译成法语",
        "input": "Hello, how are you today?",
        "output": "Bonjour, comment allez-vous aujourd'hui?"
    },
    {
        "instruction": "总结以下文本的主要内容",
        "input": "生成式人工智能是一种能够创建各种内容的AI系统...[长文本]",
        "output": "生成式AI是可创建内容的系统，包括文本、图像等。"
    }
]

# 转换为训练格式
def format_instruction(example):
    if example["input"]:
        formatted_text = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: "
    else:
        formatted_text = f"Instruction: {example['instruction']}\nOutput: "
    
    return {
        "text": formatted_text,
        "target": example["output"]
    }

# 微调代码(基于Transformers)
def instruction_tuning(model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 设置特殊标记
    tokenizer.pad_token = tokenizer.eos_token
    
    # 数据处理
    def preprocess(examples):
        formatted = [format_instruction(ex) for ex in examples]
        inputs = [ex["text"] for ex in formatted]
        targets = [ex["target"] for ex in formatted]
        
        model_inputs = tokenizer(inputs, padding="max_length", max_length=512, truncation=True)
        labels = tokenizer(targets, padding="max_length", max_length=512, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    processed_dataset = dataset.map(preprocess, batched=True)
    
    # 训练配置与执行
    training_args = Seq2SeqTrainingArguments(
        output_dir="instruction-tuned-model",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        fp16=True,
    )
    
    # 开始训练
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )
    
    trainer.train()
    return model, tokenizer
```

### RLHF (Reinforcement Learning from Human Feedback)

基于人类反馈的强化学习微调，用于改进模型与人类偏好的一致性：

```python
# RLHF简化流程

# 1. 监督微调(SFT)阶段
def supervised_fine_tuning(base_model, sft_dataset):
    # 类似常规微调，训练模型生成人类偏好的回答
    # ...

# 2. 奖励模型训练阶段
class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # 加载预训练模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # 打分
        
def train_reward_model(model_name, comparison_dataset):
    """训练奖励模型，学习人类偏好"""
    reward_model = RewardModel(model_name)
    
    # 处理数据
    def process_comparisons(examples):
        # 每个示例包含提示和两个回答(A优于B)
        prompts = examples["prompt"]
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        
        chosen_inputs = tokenizer(prompts, chosen, return_tensors="pt", padding=True)
        rejected_inputs = tokenizer(prompts, rejected, return_tensors="pt", padding=True)
        
        return {
            "chosen_input_ids": chosen_inputs.input_ids,
            "chosen_attention_mask": chosen_inputs.attention_mask,
            "rejected_input_ids": rejected_inputs.input_ids,
            "rejected_attention_mask": rejected_inputs.attention_mask,
        }
    
    # 训练
    def compute_loss(batch, reward_model):
        chosen_rewards = reward_model(
            batch["chosen_input_ids"], batch["chosen_attention_mask"]
        )
        rejected_rewards = reward_model(
            batch["rejected_input_ids"], batch["rejected_attention_mask"]
        )
        
        # 优选回答应获得更高奖励
        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        return loss
    
    # 训练循环
    # ...
    
    return reward_model

# 3. 强化学习优化阶段(PPO算法)
def rlhf_training(sft_model, reward_model, prompts_dataset):
    """使用PPO算法优化SFT模型"""
    # PPO训练配置
    # ...
    
    # 训练循环
    for epoch in range(num_epochs):
        for prompt_batch in prompts_dataset:
            # 生成回答
            with torch.no_grad():
                responses = sft_model.generate(prompt_batch)
                
            # 计算奖励
            rewards = reward_model(prompt_batch, responses)
            
            # PPO更新
            # 计算优势估计
            # 计算策略损失
            # 添加KL惩罚(与原始SFT模型比较)
            # 优化模型
            
    return optimized_model
```

### 多任务微调(Multi-task Fine-tuning)

同时训练多个任务，增强模型通用性：

```python
def multitask_fine_tuning(model_name, datasets_dict):
    """多任务微调"""
    # 加载模型
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # 处理不同任务数据
    processed_datasets = {}
    for task_name, dataset in datasets_dict.items():
        processed_datasets[task_name] = process_task_dataset(
            dataset, task_name, tokenizer
        )
    
    # 混合数据集
    from datasets import concatenate_datasets
    # 可使用不同的采样策略(这里简单连接)
    combined_dataset = concatenate_datasets(
        [ds for ds in processed_datasets.values()]
    )
    
    # 定义训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir="multitask-model",
        learning_rate=3e-5,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
    )
    
    # 微调
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    return model, tokenizer

def process_task_dataset(dataset, task_name, tokenizer):
    """处理特定任务数据集"""
    task_prefixes = {
        "translation": "translate English to German: ",
        "summarization": "summarize: ",
        "question_answering": "answer the question: ",
        "classification": "classify: ",
    }
    
    def preprocess(examples):
        prefix = task_prefixes.get(task_name, "")
        
        # 根据任务类型格式化输入输出
        if task_name == "translation":
            inputs = [prefix + text for text in examples["english"]]
            targets = examples["german"]
        elif task_name == "summarization":
            inputs = [prefix + text for text in examples["article"]]
            targets = examples["summary"]
        elif task_name == "question_answering":
            inputs = [prefix + q + " context: " + c 
                     for q, c in zip(examples["question"], examples["context"])]
            targets = examples["answer"]
        else:
            inputs = [prefix + text for text in examples["text"]]
            targets = examples["label"]
            
        # 分词处理
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    return dataset.map(preprocess, batched=True)
```

### 连续预训练(Continual Pretraining)

在专业领域数据上继续预训练，适应特定领域：

```python
def continual_pretraining(base_model_name, domain_corpus, output_dir):
    """在领域数据上继续预训练"""
    # 加载基础模型和分词器
    model = RobertaForMaskedLM.from_pretrained(base_model_name)
    tokenizer = RobertaTokenizer.from_pretrained(base_model_name)
    
    # 数据处理
    def tokenize_function(examples):
        return tokenizer(examples["text"], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=512)
                        
    tokenized_datasets = domain_corpus.map(
        tokenize_function, 
        batched=True, 
        num_proc=4, 
        remove_columns=["text"]
    )
    
    # 创建数据整理器(MLM任务)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )
    
    # 训练参数
    # 使用较小的学习率，避免偏离太远
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        learning_rate=1e-5,  # 小学习率
        weight_decay=0.01,
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # 保存模型和分词器
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer
```

### 量化感知微调(QAT)

为低精度部署优化模型：

```python
import torch.quantization as quantization

def quantization_aware_fine_tuning(model, train_dataloader, eval_dataloader, num_epochs=3):
    """量化感知微调"""
    # 1. 为量化准备模型(替换特定操作为可量化版本)
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    model_prepared = quantization.prepare_qat(model)
    
    # 2. 量化感知训练
    optimizer = AdamW(model_prepared.parameters(), lr=5e-5)
    
    for epoch in range(num_epochs):
        # 训练循环
        model_prepared.train()
        for batch in train_dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            
            outputs = model_prepared(**inputs)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
        
        # 评估
        model_prepared.eval()
        eval_loss = 0
        for batch in eval_dataloader:
            with torch.no_grad():
                inputs, labels = batch
                outputs = model_prepared(**inputs)
                eval_loss += outputs.loss.item()
    
    # 3. 转换为量化模型
    model_quantized = quantization.convert(model_prepared)
    
    return model_quantized
```

## 总结与最佳实践

### 不同场景下的选择指南

| **场景** | **推荐方法** | **优势** |
|---------|------------|---------|
| 数据充足 | 全参数微调 | 最大适应性和性能 |
| 资源受限 | LoRA、Adapters | 参数高效，低内存需求 |
| 无需额外训练 | 提示工程(ICL) | 零代码，快速应用 |
| 特定领域任务 | 连续预训练+微调 | 最佳领域适应性 |
| 多样化应用 | 多任务微调 | 通用性强，减少过拟合 |
| 人类对齐 | RLHF | 更符合人类偏好 |

### 未来发展趋势

1. **更高效的微调方法**：减少计算资源需求
2. **自动化微调流程**：AutoML应用于微调阶段
3. **跨模态迁移学习**：在不同模态间进行知识转移
4. **更好的遗忘解决方案**：平衡新旧知识
5. **可解释性的预训练与微调**：理解参数变化

预训练与微调方法已成为现代深度学习的基础范式，掌握这些技术能够大幅提升模型效果，减少计算资源需求，并使AI系统更好地适应各种应用场景。随着研究不断深入，这一领域将持续发展，为AI技术的广泛应用奠定更坚实的基础。

Similar code found with 2 license types