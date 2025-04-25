from transformers import pipeline

# 创建各种NLP任务的pipeline
sentiment_analyzer = pipeline("sentiment-analysis")
question_answerer = pipeline("question-answering")
summarizer = pipeline("summarization")
generator = pipeline("text-generation")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# 情感分析
result = sentiment_analyzer("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.999}]

# 问答
context = "Hugging Face was founded in 2016. It was originally focused on building conversational AI."
question = "When was Hugging Face founded?"
answer = question_answerer(question=question, context=context)
print(answer)  # {'answer': '2016', 'start': 22, 'end': 26, 'score': 0.98}

# 文本摘要
summary = summarizer("Transformers library is developed by Hugging Face...", max_length=50, min_length=10)
print(summary)

# 文本生成
text = generator("Once upon a time", max_length=30, num_return_sequences=2)
print(text)

# 翻译
translation = translator("Hello, how are you?")
print(translation)  # [{'translation_text': 'Bonjour, comment allez-vous?'}]