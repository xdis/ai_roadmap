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