import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from datasets import load_dataset

# 1. 加载预训练模型和分词器
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 2. 创建问答pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# 3. 基本问答示例
context = """
Hugging Face是一家总部位于纽约的AI创业公司，成立于2016年。
该公司开发了用于构建应用程序的机器学习库，最初是基于
PyTorch、TensorFlow和scikit-learn的自然语言处理技术。
现在，他们提供了transformers、tokenizers和datasets库，
这些库已成为NLP社区的重要工具。2021年，
公司筹集了4000万美元的资金，估值超过5亿美元。
"""

questions = [
    "Hugging Face是什么时候成立的？",
    "Hugging Face的主要产品是什么？",
    "Hugging Face总部在哪里？",
    "Hugging Face在2021年筹集了多少资金？"
]

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"问题: {question}")
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['score']:.4f}\n")

# 4. 微调问答模型
# 加载SQuAD数据集
squad_dataset = load_dataset("squad")

# 数据预处理
def preprocess_squad(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        
        sequence_ids = inputs.sequence_ids(i)
        
        # 找到上下文的起始和结束位置
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1
            
        # 如果答案不在上下文中，标记为不可能
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # 找到答案的起始和结束位置
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# 应用预处理
tokenized_squad = squad_dataset.map(
    preprocess_squad, 
    batched=True, 
    remove_columns=squad_dataset["train"].column_names
)

# 微调模型
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
)

# 开始训练
trainer.train()

# 保存微调后的模型
model.save_pretrained("./my-qa-model")
tokenizer.save_pretrained("./my-qa-model")