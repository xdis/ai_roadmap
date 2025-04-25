from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 初始化FastAPI应用
app = FastAPI()

# 加载模型
classifier = pipeline("sentiment-analysis")

# 定义请求模型
class TextRequest(BaseModel):
    text: str

# 创建API端点
@app.post("/analyze")
def analyze_sentiment(request: TextRequest):
    result = classifier(request.text)
    return {"result": result}

# 启动服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)