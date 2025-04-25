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