# PEFT (Parameter-Efficient Fine-Tuning)

## 什么是PEFT？

PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）是一种针对大型预训练模型的优化技术，它允许我们使用较少的计算资源高效地微调大型模型。传统的微调方法需要更新模型的所有参数，而PEFT只更新一小部分参数或添加少量新参数，大大降低了计算和存储成本。

## 为什么PEFT很重要？

随着模型规模的增长（例如GPT-3有1750亿参数，GPT-4更大），全参数微调带来的挑战：

1. **内存消耗巨大**：需要存储完整模型和所有参数的梯度
2. **计算成本高**：更新所有参数需要大量计算资源
3. **灾难性遗忘**：容易丢失预训练阶段获得的通用知识

PEFT通过只更新少量参数（通常<1%）解决了这些问题。

## 主要的PEFT方法

### 1. LoRA (Low-Rank Adaptation)

LoRA通过在原始权重旁边添加低秩矩阵来实现高效微调。

#### 工作原理

对于原始权重矩阵W，LoRA添加一个低秩分解：ΔW = A×B，其中：
- A的尺寸为 d×r
- B的尺寸为 r×k
- r是秩，通常远小于d和k（比如r=8）

最终参数更新：W' = W + ΔW = W + A×B

#### 代码示例

```python
import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification

# 加载预训练模型
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
)

# 配置LoRA参数
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                           # 低秩分解的秩
    lora_alpha=32,                 # 缩放参数
    lora_dropout=0.1,              # Dropout概率
    target_modules=["query", "key", "value"]  # 要应用LoRA的模块
)

# 创建PEFT模型
peft_model = get_peft_model(model, peft_config)

# 查看可训练参数数量
trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in peft_model.parameters())
print(f"可训练参数：{trainable_params} ({trainable_params/all_params*100:.2f}%)")

# 训练过程类似普通模型训练
# ...

# 保存PEFT模型权重（只保存LoRA参数，不保存底层模型）
peft_model.save_pretrained("./my_lora_model")
```

### 2. Prompt Tuning

Prompt Tuning是在输入嵌入层添加一些可学习的"提示"向量，仅更新这些向量而保持模型其他部分不变。

#### 代码示例

```python
import torch
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from transformers import AutoModelForCausalLM

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 配置Prompt Tuning
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,  # 从文本初始化
    num_virtual_tokens=20,                    # 虚拟token数量
    prompt_tuning_init_text="分类任务：",      # 初始化提示文本
    tokenizer_name_or_path="gpt2"             # 使用的tokenizer
)

# 创建PEFT模型
peft_model = get_peft_model(model, peft_config)
```

### 3. P-Tuning

P-Tuning在输入序列中插入一些连续的可训练参数（称为"提示向量"），这些参数可以针对特定任务进行优化。

### 4. Prefix Tuning

Prefix Tuning在模型的每一层添加可训练的前缀向量，影响模型的内部状态。

#### 代码示例

```python
from peft import PrefixTuningConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM

# 加载预训练模型
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# 配置Prefix Tuning
peft_config = PrefixTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,           # 虚拟token数量
    prefix_projection=True,          # 是否使用投影层
    encoder_hidden_size=512,         # 投影层隐藏大小
)

# 创建PEFT模型
peft_model = get_peft_model(model, peft_config)
```

### 5. QLoRA

QLoRA结合了量化技术和LoRA，先将预训练模型量化为4位精度，然后应用LoRA微调。这进一步降低了内存需求。

## PEFT的实际应用

1. **领域适配**：将通用大型模型调整为特定领域（例如医疗、法律）
2. **任务微调**：针对特定任务（分类、摘要、问答等）优化模型
3. **低资源设备部署**：降低模型存储空间，实现在消费级硬件上部署

## PEFT与传统微调比较

| 特性 | 全参数微调 | PEFT |
|------|------------|------|
| 参数量 | 全部 | <1% |
| GPU内存 | 高 | 低 |
| 训练速度 | 慢 | 快 |
| 存储需求 | 大 | 小 |
| 迁移效率 | 低 | 高 |

## 结合使用多种PEFT方法

在实际应用中，可以结合多种PEFT方法获得更好效果：

```python
from peft import MultitaskPromptTuningConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# 为不同任务配置不同的提示
peft_config = MultitaskPromptTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    task_ids={
        "翻译": 0,
        "摘要": 1,
        "问答": 2
    }
)

peft_model = get_peft_model(model, peft_config)
```

## 实战例子：使用LoRA微调Stable Diffusion模型

```python
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

# 加载预训练的Stable Diffusion
model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")

# 配置LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

# 应用LoRA到文本编码器和UNet
text_encoder_peft = get_peft_model(model.text_encoder, lora_config)
unet_peft = get_peft_model(model.unet, lora_config)

# 替换原始模型中的组件
model.text_encoder = text_encoder_peft
model.unet = unet_peft

# 现在可以用很少的资源微调这个模型了
```

## 总结

PEFT技术是大型模型时代的关键工具，它让个人开发者和小型组织也能高效地调整和部署大型AI模型。通过只更新一小部分参数，PEFT显著降低了计算、内存和存储需求，同时保持了模型性能。

对于入门者，建议从LoRA开始尝试，它是目前最流行、文档最完善的PEFT方法，在各种场景下都表现良好。