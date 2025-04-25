# 创建app.py文件
import gradio as gr
from transformers import pipeline

# 加载模型
generator = pipeline("text-generation", model="gpt2")

# 定义预测函数
def predict(prompt, max_length=100):
    outputs = generator(prompt, max_length=max_length, do_sample=True)
    return outputs[0]["generated_text"]

# 创建Gradio接口
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(placeholder="请输入提示文本...", lines=3),
        gr.Slider(minimum=10, maximum=500, value=100, label="最大长度")
    ],
    outputs="text",
    title="GPT-2 文本生成器",
    description="这个应用使用GPT-2模型生成文本。输入提示，模型将继续写作。",
)

# 启动应用
if __name__ == "__main__":
    demo.launch()

# 然后可以部署到Hugging Face Spaces