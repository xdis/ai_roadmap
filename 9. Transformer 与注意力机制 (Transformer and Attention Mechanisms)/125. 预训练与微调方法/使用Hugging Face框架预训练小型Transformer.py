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