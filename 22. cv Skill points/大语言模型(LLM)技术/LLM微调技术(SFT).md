## 8. LLM微调技术(SFT)详解

监督微调(Supervised Fine-Tuning, SFT)是使大型语言模型适应特定任务或领域的关键技术。以下是对SFT的详细解释：

### 8.1 什么是监督微调(SFT)

监督微调是在预训练完成后，使用带标注的数据对模型进行进一步训练的过程。它有几个重要目的：

- 使模型适应特定领域的知识和语言
- 教会模型遵循特定格式的指令
- 优化模型在特定任务上的表现
- 减轻预训练阶段可能存在的偏见

### 8.2 SFT的工作原理

SFT的核心思想很简单：通过高质量的人类标注数据，指导模型生成我们期望的输出：

1. **准备数据集**：创建"提示(prompt)-回答(completion)"对的数据集
2. **调整学习率**：通常使用比预训练阶段更小的学习率
3. **训练模型**：让模型通过监督学习方式适应这些样本
4. **评估性能**：在验证集上测试模型表现

### 8.3 SFT的代码实现

以下是一个简化的SFT实现示例，使用PyTorch和Hugging Face的Transformers库：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

# 1. 加载预训练模型和分词器
model_name = "facebook/opt-1.3b"  # 以OPT-1.3B为例
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 准备微调数据集
sft_data = [
    {"prompt": "请解释什么是机器学习？", "completion": "机器学习是人工智能的一个子领域，它专注于开发能够从数据中学习的算法和模型，无需显式编程即可执行任务。"},
    {"prompt": "如何计算两点之间的距离？", "completion": "在二维平面上，两点(x1,y1)和(x2,y2)之间的欧几里得距离可以用公式sqrt((x2-x1)^2 + (y2-y1)^2)计算。"},
    # 更多数据对...
]

# 3. 格式化数据集
def format_instruction(example):
    # 将提示和回答组合成一个文本
    text = f"问题：{example['prompt']}\n回答：{example['completion']}"
    return {"text": text}

# 转换为Hugging Face数据集格式
dataset = Dataset.from_list(sft_data)
formatted_dataset = dataset.map(format_instruction)

# 4. 数据预处理
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = formatted_dataset.map(preprocess_function, batched=True)

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./sft-model",
    learning_rate=2e-5,  # 微调时使用较小的学习率
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=True,  # 使用混合精度训练加速
)

# 6. 创建Trainer并开始微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 开始微调
trainer.train()

# 7. 保存微调后的模型
model.save_pretrained("./my-sft-model")
tokenizer.save_pretrained("./my-sft-model")
```

### 8.4 参数高效微调(PEFT)

传统SFT需要更新模型的所有参数，对于大型模型来说计算成本高。参数高效微调(PEFT)方法解决了这个问题：

#### 8.4.1 LoRA (Low-Rank Adaptation)

LoRA是一种流行的PEFT方法，它只训练低秩矩阵来调整预训练权重：

```python
from peft import get_peft_model, LoraConfig, TaskType

# 定义LoRA配置
lora_config = LoraConfig(
    r=8,                     # LoRA矩阵的秩
    lora_alpha=32,           # LoRA alpha参数
    target_modules=["q_proj", "v_proj"],  # 要应用LoRA的模块
    lora_dropout=0.05,       # LoRA dropout
    bias="none",
    task_type=TaskType.CAUSAL_LM  # 任务类型
)

# 创建PEFT模型
peft_model = get_peft_model(model, lora_config)

# LoRA微调只更新一小部分参数
print(f"可训练参数: {peft_model.num_parameters()['trainable']}")
print(f"总参数: {peft_model.num_parameters()['total']}")

# 训练过程与普通SFT相同，但更新的只是LoRA参数
```

### 8.5 SFT的最佳实践

1. **数据质量至关重要**：
   - 使用高质量、多样性的数据
   - 确保数据覆盖目标应用场景
   - 定期审查训练样本以排除有害内容

2. **超参数调整**：
   - 较小的学习率（通常1e-5到5e-5）
   - 适当的批次大小（根据GPU内存调整）
   - 使用学习率预热和合适的权重衰减

3. **评估与监控**：
   - 使用多种指标评估模型（不仅仅是损失值）
   - 监控过拟合现象
   - 进行人工质量评估

### 8.6 SFT的实际应用

SFT被广泛应用于多个领域：

- **垂直领域适应**：使通用LLM适应医疗、法律、金融等特定领域
- **指令调优**：使模型更好地理解和遵循人类指令
- **对话系统**：训练模型进行自然、有帮助且安全的对话
- **语言本地化**：针对特定语言进行优化
- **个性化助手**：根据特定风格或人格调整模型输出

## 总结

监督微调(SFT)是大语言模型开发流程中不可或缺的一环，它能显著提升模型在特定任务和领域的性能。结合参数高效微调技术(如LoRA)，SFT可以在有限计算资源下高效地适应各种应用场景，是LLM落地应用的关键技术。

了解了SFT的原理和实践方法后，你已经具备了开始尝试微调自己的语言模型的基础知识，可以将强大的通用模型转变为专门针对特定任务优化的工具。

## 总结

大语言模型基于Transformer架构，利用自注意力机制处理文本序列中的长距离依赖关系。通过在海量文本数据上预训练，再结合微调和RLHF，LLM能够理解和生成高质量的文本内容。

虽然现代LLM表现出惊人的能力，但理解它们的基本架构、训练过程和局限性对于有效利用这些模型至关重要。随着技术的发展，我们可以期待LLM在架构、训练方法和应用场景上的进一步创新。

你还有关于LLM原理与架构的具体问题吗？