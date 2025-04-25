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