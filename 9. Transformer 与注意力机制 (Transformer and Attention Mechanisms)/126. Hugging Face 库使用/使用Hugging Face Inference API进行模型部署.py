import requests
import json

# 使用Inference API (需要Hugging Face API令牌)
API_TOKEN = "your_api_token_here"
API_URL = "https://api-inference.huggingface.co/models/gpt2"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# 示例请求
output = query({
    "inputs": "The quick brown fox jumps over the",
    "parameters": {
        "max_length": 50,
        "temperature": 0.7
    }
})

print(output)