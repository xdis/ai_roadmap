# LLM应用开发详解

大语言模型(LLM)应用开发是指利用预训练的大型语言模型构建实用的AI应用程序。这个领域结合了AI技术与软件工程，使开发者能够创建智能、交互式的应用。下面我将详细解释LLM应用开发的核心概念、常用技术和实践方法。

## 1. LLM应用开发基础

### 1.1 LLM应用的基本工作流程

LLM应用的基本工作流程包括：
1. 接收用户输入
2. 预处理输入数据
3. 构建适当的提示(Prompt)
4. 调用LLM获取回复
5. 处理LLM输出
6. 将处理后的信息返回给用户

```python
import openai

def basic_llm_app(user_input):
    """简单的LLM应用示例"""
    # 1. 接收用户输入 - 已通过参数传入
    
    # 2. 预处理输入(简化版)
    processed_input = user_input.strip()
    
    # 3. 构建提示
    prompt = f"用户: {processed_input}\n助手:"
    
    # 4. 调用LLM
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个乐于助人的AI助手。"},
            {"role": "user", "content": processed_input}
        ],
        max_tokens=500,
        temperature=0.7
    )
    
    # 5. 处理LLM输出
    assistant_response = response.choices[0].message.content
    
    # 6. 返回结果给用户
    return assistant_response

# 使用示例
result = basic_llm_app("如何学习Python编程?")
print(result)
```

### 1.2 常用的LLM接口

开发LLM应用时，可以使用多种接口：

```python
# OpenAI API调用示例
def call_openai_api(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# Hugging Face模型本地部署调用示例
def call_local_huggingface_model(prompt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Anthropic Claude API调用示例
def call_anthropic_api(prompt):
    import anthropic
    
    client = anthropic.Client(api_key="your_api_key")
    response = client.messages.create(
        model="claude-2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    
    return response.content[0].text
```

## 2. LLM应用架构设计

### 2.1 基本的Web应用架构

使用FastAPI构建LLM应用的后端：

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import uvicorn

# 定义请求和响应模型
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# 初始化FastAPI应用
app = FastAPI(title="LLM聊天应用")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置OpenAI API
openai.api_key = "your_api_key"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 调用LLM
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个乐于助人的AI助手。"},
                {"role": "user", "content": request.message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # 提取回复
        assistant_response = response.choices[0].message.content
        
        return ChatResponse(response=assistant_response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2.2 前端界面集成

使用HTML、JavaScript和Vue.js构建简单的前端：

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM聊天应用</title>
    <script src="https://cdn.jsdelivr.net/npm/vue@2.6.14/dist/vue.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e3f2fd;
            text-align: right;
        }
        .assistant-message {
            background-color: #f5f5f5;
        }
        .input-area {
            display: flex;
            margin-top: 20px;
        }
        input {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            margin-left: 10px;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:disabled {
            background-color: #cccccc;
        }
    </style>
</head>
<body>
    <div id="app" class="chat-container">
        <h1>LLM聊天应用</h1>
        
        <div class="chat-history">
            <div v-for="(message, index) in messages" :key="index" 
                 :class="['message', message.role === 'user' ? 'user-message' : 'assistant-message']">
                <strong>{{ message.role === 'user' ? '你' : 'AI助手' }}:</strong> {{ message.content }}
            </div>
        </div>
        
        <div class="input-area">
            <input v-model="userInput" @keyup.enter="sendMessage" placeholder="输入消息..." :disabled="loading">
            <button @click="sendMessage" :disabled="loading || !userInput.trim()">
                {{ loading ? '处理中...' : '发送' }}
            </button>
        </div>
    </div>

    <script>
        new Vue({
            el: '#app',
            data: {
                userInput: '',
                messages: [],
                loading: false
            },
            methods: {
                sendMessage() {
                    if (this.loading || !this.userInput.trim()) return;
                    
                    // 添加用户消息
                    const userMessage = this.userInput.trim();
                    this.messages.push({
                        role: 'user',
                        content: userMessage
                    });
                    this.userInput = '';
                    
                    // 显示加载状态
                    this.loading = true;
                    
                    // 调用后端API
                    axios.post('http://localhost:8000/chat', {
                        message: userMessage
                    })
                    .then(response => {
                        // 添加助手回复
                        this.messages.push({
                            role: 'assistant',
                            content: response.data.response
                        });
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('发送消息时出错');
                    })
                    .finally(() => {
                        this.loading = false;
                    });
                }
            }
        });
    </script>
</body>
</html>
```

## 3. 高级LLM应用功能

### 3.1 对话状态管理

管理对话历史记录以实现上下文感知对话：

```python
class ConversationManager:
    """管理对话上下文的工具类"""
    
    def __init__(self, max_history=10, system_prompt="你是一个乐于助人的AI助手。"):
        self.conversations = {}  # 使用字典存储不同用户的对话
        self.max_history = max_history
        self.system_prompt = system_prompt
    
    def add_message(self, conversation_id, role, content):
        """添加消息到指定对话"""
        # 如果是新对话，初始化
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        # 添加消息
        self.conversations[conversation_id].append({
            "role": role,
            "content": content
        })
        
        # 保持对话历史在限制范围内
        if len(self.conversations[conversation_id]) > self.max_history:
            # 保留系统消息，删除最早的用户-助手对话
            if self.conversations[conversation_id][0]["role"] == "system":
                # 保留系统消息，移除最早的用户和助手消息
                self.conversations[conversation_id] = [
                    self.conversations[conversation_id][0]
                ] + self.conversations[conversation_id][3:]
            else:
                # 没有系统消息，直接移除最早的两条
                self.conversations[conversation_id] = self.conversations[conversation_id][2:]
    
    def get_messages(self, conversation_id):
        """获取指定对话的所有消息"""
        if conversation_id not in self.conversations:
            # 新对话，初始化系统提示
            return [{"role": "system", "content": self.system_prompt}]
        
        return self.conversations[conversation_id]
    
    def clear_conversation(self, conversation_id):
        """清除指定对话历史"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

# 使用示例
conversation_manager = ConversationManager()

# 用户发送第一条消息
conversation_id = "user123"
user_message = "你好，请介绍一下Python。"
conversation_manager.add_message(conversation_id, "user", user_message)

# 获取完整对话历史（包括系统提示）
messages = conversation_manager.get_messages(conversation_id)

# 调用OpenAI API
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=500
)
assistant_response = response.choices[0].message.content

# 保存助手回复
conversation_manager.add_message(conversation_id, "assistant", assistant_response)
```

### 3.2 流式响应处理

使用流式响应提供更好的用户体验：

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import uvicorn
import asyncio
import json

app = FastAPI()

class StreamRequest(BaseModel):
    message: str
    conversation_id: str

@app.post("/chat/stream")
async def stream_chat(request: StreamRequest):
    async def generate():
        try:
            # 获取对话历史(这里简化了，实际中应使用ConversationManager)
            messages = [
                {"role": "system", "content": "你是一个乐于助人的AI助手。"},
                {"role": "user", "content": request.message}
            ]
            
            # 创建流式请求
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                stream=True  # 启用流式传输
            )
            
            # 流式返回结果
            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.get("content", "")
                    if content:
                        # 使用SSE格式返回数据
                        yield f"data: {json.dumps({'content': content})}\n\n"
            
            # 发送完成信号
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
        
        except Exception as e:
            # 发送错误信息
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# 前端JavaScript示例
"""
// 使用EventSource处理流式响应
const eventSource = new EventSource('/chat/stream?message=讲解Python编程&conversation_id=123');

let fullResponse = '';

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.error) {
        console.error('Error:', data.error);
        eventSource.close();
        return;
    }
    
    // 追加内容
    fullResponse += data.content;
    
    // 更新UI显示
    document.getElementById('response').textContent = fullResponse;
    
    // 检查是否完成
    if (data.done) {
        eventSource.close();
    }
};

eventSource.onerror = (error) => {
    console.error('EventSource error:', error);
    eventSource.close();
};
"""
```

### 3.3 工具集成(Function Calling)

让LLM调用外部工具提升应用能力：

```python
import openai
import json
import requests
from datetime import datetime

# 定义可用工具(函数)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "搜索商品信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "category": {
                        "type": "string",
                        "description": "商品类别，例如：电子产品、服装、食品",
                        "enum": ["电子产品", "服装", "食品", "家居", "全部"]
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最大返回结果数量"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 实现工具函数
def get_weather(city):
    """获取城市天气信息(模拟实现)"""
    # 实际应用中应调用真实的天气API
    weather_data = {
        "北京": {"temperature": "23°C", "condition": "晴朗", "humidity": "45%"},
        "上海": {"temperature": "26°C", "condition": "多云", "humidity": "60%"},
        "广州": {"temperature": "29°C", "condition": "阵雨", "humidity": "75%"}
    }
    
    if city in weather_data:
        return weather_data[city]
    else:
        return {"error": f"未找到{city}的天气信息"}

def search_products(query, category="全部", max_results=5):
    """搜索商品信息(模拟实现)"""
    # 实际应用中应查询数据库或调用电商API
    products = [
        {"name": "iPhone 15", "category": "电子产品", "price": 5999},
        {"name": "MacBook Pro", "category": "电子产品", "price": 12999},
        {"name": "Nike跑鞋", "category": "服装", "price": 899},
        {"name": "羽绒服", "category": "服装", "price": 1299},
        {"name": "巧克力", "category": "食品", "price": 58},
        {"name": "茶叶", "category": "食品", "price": 128},
        {"name": "沙发", "category": "家居", "price": 3499}
    ]
    
    # 过滤商品
    if category != "全部":
        filtered_products = [p for p in products if p["category"] == category]
    else:
        filtered_products = products
    
    # 关键词搜索
    results = [p for p in filtered_products if query.lower() in p["name"].lower()]
    
    # 限制结果数量
    return results[:max_results]

# 处理工具调用
def handle_tool_calls(tool_calls):
    """处理工具调用请求并返回结果"""
    results = []
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name == "get_weather":
            result = get_weather(function_args.get("city"))
        elif function_name == "search_products":
            result = search_products(
                function_args.get("query"),
                function_args.get("category", "全部"),
                function_args.get("max_results", 5)
            )
        else:
            result = {"error": f"未知函数: {function_name}"}
        
        results.append({
            "tool_call_id": tool_call.id,
            "function": {"name": function_name, "arguments": tool_call.function.arguments},
            "result": result
        })
    
    return results

# 示例：处理用户查询
def process_user_query(query):
    """处理用户查询，需要时调用工具"""
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以回答问题并提供帮助。需要时，你可以调用工具来获取信息。"},
        {"role": "user", "content": query}
    ]
    
    # 第一轮：让模型决定是否调用工具
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    messages.append(response_message)
    
    # 检查是否需要工具调用
    if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
        # 处理工具调用
        tool_results = handle_tool_calls(response_message.tool_calls)
        
        # 将结果添加到对话中
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": json.dumps(result["result"])
            })
        
        # 第二轮：让模型基于工具结果生成最终回复
        final_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        return final_response.choices[0].message.content
    else:
        # 不需要工具调用，直接返回回复
        return response_message.content

# 使用示例
print(process_user_query("北京今天天气怎么样？"))
print(process_user_query("有哪些电子产品推荐？"))
```

## 4. 增强LLM应用功能

### 4.1 使用RAG(检索增强生成)

RAG通过检索相关文档增强LLM回答的准确性：

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import openai
import os

class RAGSystem:
    """基于检索增强生成的问答系统"""
    
    def __init__(self, docs_dir, embed_model="openai", collection_name="my_documents"):
        self.docs_dir = docs_dir
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
    
    def load_documents(self):
        """加载文档"""
        loader = DirectoryLoader(self.docs_dir, glob="**/*.txt")
        documents = loader.load()
        print(f"加载了 {len(documents)} 个文档")
        return documents
    
    def process_documents(self, documents):
        """处理文档并创建向量存储"""
        # 文档分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"文档被分割成 {len(chunks)} 个块")
        
        # 创建向量存储
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory="./chroma_db"
        )
        self.vector_store.persist()
    
    def setup(self):
        """设置RAG系统"""
        documents = self.load_documents()
        self.process_documents(documents)
    
    def query(self, user_question, num_docs=3):
        """处理用户查询"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call setup() first.")
        
        # 检索相关文档
        retrieved_docs = self.vector_store.similarity_search(user_question, k=num_docs)
        
        # 提取文档内容
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 构建提示
        prompt = f"""
        回答以下问题，使用提供的上下文信息。如果上下文中没有相关信息，请明确说明信息不足，不要编造答案。

        上下文信息:
        {context}

        问题: {user_question}
        """
        
        # 调用LLM生成回答
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个知识丰富的助手，只使用提供的上下文信息回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        answer = response.choices[0].message.content
        
        return {
            "question": user_question,
            "answer": answer,
            "sources": [{"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")} for doc in retrieved_docs]
        }

# 使用示例
def rag_example():
    # 确保文档目录存在
    docs_dir = "./knowledge_base"
    os.makedirs(docs_dir, exist_ok=True)
    
    # 创建示例文档(实际应用中这些文档已经存在)
    with open(f"{docs_dir}/python_info.txt", "w") as f:
        f.write("Python是一种广泛使用的解释型、高级编程语言。Python的设计强调代码的可读性和简洁的语法，使开发者能够用更少的代码表达想法。Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。")
    
    with open(f"{docs_dir}/machine_learning.txt", "w") as f:
        f.write("机器学习是人工智能的一个子领域，专注于开发能够从数据中学习和改进的算法和统计模型，而无需显式编程。常见的机器学习算法包括线性回归、决策树、神经网络和支持向量机等。")
    
    # 创建并设置RAG系统
    rag_system = RAGSystem(docs_dir)
    rag_system.setup()
    
    # 查询示例
    result = rag_system.query("什么是Python编程语言？")
    print(f"问题: {result['question']}")
    print(f"回答: {result['answer']}")
    print("参考来源:")
    for source in result['sources']:
        print(f"- {source['content'][:100]}...")

# 运行示例
rag_example()
```

### 4.2 使用代理(Agent)模式

构建能自主解决复杂任务的AI代理：

```python
import openai
import json
import re
import time
from typing import List, Dict, Any

class Tool:
    """工具基类"""
    def __init__(self, name: str, description: str, parameters: Dict):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def execute(self, **kwargs):
        raise NotImplementedError("子类必须实现execute方法")

class WeatherTool(Tool):
    """天气查询工具"""
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="获取指定城市的当前天气信息",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州"
                    }
                },
                "required": ["city"]
            }
        )
    
    def execute(self, city):
        # 模拟实现，实际应调用天气API
        weather_data = {
            "北京": {"temperature": "23°C", "condition": "晴朗", "humidity": "45%"},
            "上海": {"temperature": "26°C", "condition": "多云", "humidity": "60%"},
            "广州": {"temperature": "29°C", "condition": "阵雨", "humidity": "75%"}
        }
        
        if city in weather_data:
            return weather_data[city]
        else:
            return {"error": f"未找到{city}的天气信息"}

class CalculatorTool(Tool):
    """计算器工具"""
    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算，支持加减乘除和基本数学函数",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如：2 + 2 * 3"
                    }
                },
                "required": ["expression"]
            }
        )
    
    def execute(self, expression):
        try:
            # 安全地计算表达式
            # 注意：实际应用中需要更安全的实现
            # 这里使用了非常简化的方法，有安全风险
            allowed_chars = set("0123456789+-*/() .")
            if not all(c in allowed_chars for c in expression):
                return {"error": "表达式包含不允许的字符"}
            
            result = eval(expression)
            return {"result": result}
        except Exception as e:
            return {"error": f"计算错误: {str(e)}"}

class WikipediaTool(Tool):
    """维基百科查询工具"""
    def __init__(self):
        super().__init__(
            name="search_wikipedia",
            description="搜索维基百科获取信息",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最大返回结果数量"
                    }
                },
                "required": ["query"]
            }
        )
    
    def execute(self, query, max_results=1):
        # 模拟实现，实际应调用维基百科API
        wiki_data = {
            "人工智能": "人工智能（AI）是计算机科学的一个分支，致力于创建能够模仿人类智能的系统。",
            "机器学习": "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习和改进的算法。",
            "Python": "Python是一种广泛使用的解释型、高级编程语言，以简洁易读的语法著称。"
        }
        
        # 简单的模糊匹配
        results = []
        for key, value in wiki_data.items():
            if query.lower() in key.lower():
                results.append({"title": key, "summary": value})
                if len(results) >= max_results:
                    break
        
        if results:
            return {"results": results}
        else:
            return {"error": f"未找到关于'{query}'的信息"}

class Agent:
    """LLM代理，能够规划和执行任务"""
    
    def __init__(self, tools: List[Tool], model="gpt-3.5-turbo"):
        self.tools = tools
        self.model = model
        self.messages = []
        self.tools_for_llm = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            } for tool in tools
        ]
        self.tool_map = {tool.name: tool for tool in tools}
    
    def add_message(self, role, content):
        """添加消息到对话历史"""
        self.messages.append({"role": role, "content": content})
    
    def format_tool_results(self, results):
        """格式化工具执行结果，用于展示"""
        formatted = []
        for result in results:
            tool_name = result.get("function", {}).get("name", "unknown")
            tool_result = result.get("result", {})
            formatted.append(f"工具: {tool_name}\n结果: {json.dumps(tool_result, ensure_ascii=False, indent=2)}")
        
        return "\n\n".join(formatted)
    
    def run(self, task, max_steps=5):
        """运行代理完成任务"""
        self.messages = [
            {"role": "system", "content": "你是一个智能代理，可以使用工具来解决问题。分析任务，确定需要使用的工具，执行工具并利用结果来完成任务。"},
            {"role": "user", "content": task}
        ]
        
        steps_taken = 0
        final_answer = None
        
        while steps_taken < max_steps:
            steps_taken += 1
            print(f"\n===== 步骤 {steps_taken} =====")
            
            # 调用LLM获取下一步行动
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools_for_llm,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            self.messages.append(response_message)
            
            # 检查是否直接给出了最终答案
            if not response_message.get("tool_calls"):
                final_answer = response_message.get("content")
                print(f"代理给出最终答案: {final_answer}")
                break
            
            # 处理工具调用
            print("代理决定使用工具:")
            tool_results = []
            
            for tool_call in response_message.get("tool_calls", []):
                function_name = tool_call.get("function", {}).get("name")
                function_args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                
                print(f"- 调用工具: {function_name}, 参数: {function_args}")
                
                if function_name in self.tool_map:
                    # 执行工具
                    tool = self.tool_map[function_name]
                    result = tool.execute(**function_args)
                    
                    tool_results.append({
                        "tool_call_id": tool_call.get("id"),
                        "function": {"name": function_name, "arguments": function_args},
                        "result": result
                    })
                else:
                    print(f"警告: 未知工具 {function_name}")
            
            # 将工具结果添加到对话中
            for result in tool_results:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": json.dumps(result["result"])
                })
            
            formatted_results = self.format_tool_results(tool_results)
            print(f"工具执行结果:\n{formatted_results}")
            
            # 如果这是最后一步，让代理给出最终答案
            if steps_taken == max_steps:
                final_response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.messages + [
                        {"role": "user", "content": "你已经使用了所有可用步骤。请基于收集到的信息给出最终答案。"}
                    ]
                )
                final_answer = final_response.choices[0].message.get("content")
                print(f"达到最大步骤数。代理给出最终答案: {final_answer}")
        
        return {
            "task": task,
            "steps_taken": steps_taken,
            "final_answer": final_answer,
            "conversation": self.messages
        }

# 使用示例
def agent_example():
    # 创建工具
    tools = [
        WeatherTool(),
        CalculatorTool(),
        WikipediaTool()
    ]
    
    # 创建代理
    agent = Agent(tools)
    
    # 运行代理完成任务
    result = agent.run("北京的天气怎么样？计算一下(23+17)*2，以及查询什么是机器学习。")
    
    print("\n===== 最终结果 =====")
    print(f"任务: {result['task']}")
    print(f"步骤数: {result['steps_taken']}")
    print(f"最终答案: {result['final_answer']}")

# 运行示例
agent_example()
```

## 5. 构建实际应用

### 5.1 文档问答应用

构建一个能回答与特定文档相关问题的应用：

```python
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import uuid
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai

app = FastAPI(title="文档问答应用")

# 设置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 文档存储路径
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 向量数据库存储
vectorstores = {}

# 配置OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# 文档加载器映射
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader
}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并处理"""
    try:
        # 生成唯一文档ID
        document_id = str(uuid.uuid4())
        
        # 获取文件扩展名
        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in LOADER_MAPPING:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")
        
        # 保存文件
        file_path = os.path.join(UPLOAD_DIR, f"{document_id}{ext}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 处理文档
        result = await process_document(document_id, file_path, ext)
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunks": result["chunks"],
            "message": "文档上传并处理成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_document(document_id, file_path, extension):
    """处理文档并创建向量存储"""
    try:
        # 选择合适的加载器
        loader_class = LOADER_MAPPING[extension.lower()]
        loader = loader_class(file_path)
        
        # 加载文档
        documents = loader.load()
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # 创建向量存储
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # 保存向量存储
        vectorstores[document_id] = vectorstore
        
        return {
            "chunks": len(chunks),
            "message": "文档处理成功"
        }
    except Exception as e:
        # 清理文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise Exception(f"处理文档时出错: {str(e)}")

@app.post("/ask")
async def ask_question(document_id: str = Form(...), question: str = Form(...)):
    """基于文档回答问题"""
    try:
        if document_id not in vectorstores:
            raise HTTPException(status_code=404, detail="文档未找到，请先上传文档")
        
        # 从向量存储中检索相关内容
        vectorstore = vectorstores[document_id]
        docs = vectorstore.similarity_search(question, k=3)
        
        # 提取文档内容
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 构建提示
        prompt = f"""
        使用以下上下文信息回答问题。如果上下文中没有足够信息，请明确说明你不知道，不要编造答案。

        上下文:
        {context}

        问题: {question}
        """
        
        # 调用LLM生成回答
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个文档问答助手，根据提供的上下文信息回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        answer = response.choices[0].message.content
        
        return {
            "question": question,
            "answer": answer,
            "context": [doc.page_content for doc in docs]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """删除文档"""
    if document_id in vectorstores:
        # 移除向量存储
        del vectorstores[document_id]
        
        # 尝试删除文件
        for ext in LOADER_MAPPING.keys():
            file_path = os.path.join(UPLOAD_DIR, f"{document_id}{ext}")
            if os.path.exists(file_path):
                os.remove(file_path)
                return {"message": f"文档 {document_id} 已删除"}
        
        return {"message": f"向量存储已删除，但未找到对应文件"}
    else:
        raise HTTPException(status_code=404, detail="文档未找到")

# 运行服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 5.2 AI助手机器人

构建能在多平台部署的AI助手机器人：

```python
import os
import json
import logging
import asyncio
import openai
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import requests
from typing import Dict, List, Optional, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ai-assistant")

# 创建FastAPI应用
app = FastAPI(title="AI助手机器人")

# 配置OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# 用户会话存储
user_sessions = {}

# 定义助手配置
ASSISTANT_CONFIG = {
    "name": "智能小助手",
    "description": "一个乐于帮助用户的智能助手，能够回答问题、提供信息和帮助完成任务。",
    "instructions": """
    你是一个有用、礼貌、友好的助手。回答用户的问题时：
    1. 保持回答简洁清晰
    2. 当不确定时，承认你的局限性
    3. 避免有害、不当或具有歧视性的回答
    4. 使用相关事实和信息回答
    5. 当用户问候时，热情回应
    6. 使用emoji表情使对话更友好 🙂
    """
}

# 消息记忆管理
class MessageMemory:
    """管理用户会话的消息历史"""
    
    def __init__(self, max_messages=20):
        self.max_messages = max_messages
        self.messages = [
            {"role": "system", "content": ASSISTANT_CONFIG["instructions"]}
        ]
    
    def add_message(self, role, content):
        """添加消息到历史"""
        self.messages.append({"role": role, "content": content})
        
        # 如果超过最大消息数，移除最早的用户和助手消息
        if len(self.messages) > self.max_messages + 1:  # +1是因为系统消息
            # 保留系统消息
            excess = len(self.messages) - self.max_messages - 1
            self.messages = [self.messages[0]] + self.messages[excess+1:]
    
    def get_messages(self):
        """获取所有消息"""
        return self.messages
    
    def clear(self):
        """清除所有历史，但保留系统消息"""
        self.messages = [self.messages[0]]

# 平台适配器
class PlatformAdapter:
    """不同平台的消息适配器基类"""
    
    def format_response(self, content, user_id):
        """将助手回复格式化为平台特定格式"""
        raise NotImplementedError
    
    def parse_message(self, payload):
        """从平台特定格式解析用户消息"""
        raise NotImplementedError

class WebAdapter(PlatformAdapter):
    """Web平台适配器"""
    
    def format_response(self, content, user_id):
        return {"message": content, "user_id": user_id}
    
    def parse_message(self, payload):
        return {
            "user_id": payload.get("user_id"),
            "message": payload.get("message"),
            "platform": "web"
        }

class SlackAdapter(PlatformAdapter):
    """Slack平台适配器"""
    
    def format_response(self, content, user_id):
        return {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": content
                    }
                }
            ]
        }
    
    def parse_message(self, payload):
        event = payload.get("event", {})
        return {
            "user_id": event.get("user"),
            "message": event.get("text"),
            "platform": "slack",
            "channel": event.get("channel")
        }

# 平台适配器映射
platform_adapters = {
    "web": WebAdapter(),
    "slack": SlackAdapter()
}

# AI处理逻辑
async def process_message(user_message, user_id, platform="web"):
    """处理用户消息并生成回复"""
    # 获取或创建用户会话
    if user_id not in user_sessions:
        user_sessions[user_id] = MessageMemory()
    
    memory = user_sessions[user_id]
    
    # 添加用户消息
    memory.add_message("user", user_message)
    
    try:
        # 调用OpenAI生成回复
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=memory.get_messages(),
            temperature=0.7,
            max_tokens=500
        )
        
        # 提取助手回复
        assistant_message = response.choices[0].message.content
        
        # 添加助手回复到历史
        memory.add_message("assistant", assistant_message)
        
        # 使用适配器格式化回复
        adapter = platform_adapters.get(platform, platform_adapters["web"])
        formatted_response = adapter.format_response(assistant_message, user_id)
        
        return formatted_response
    
    except Exception as e:
        logger.error(f"生成回复时出错: {str(e)}")
        error_message = "抱歉，我现在无法回答。请稍后再试。"
        memory.add_message("assistant", error_message)
        
        adapter = platform_adapters.get(platform, platform_adapters["web"])
        return adapter.format_response(error_message, user_id)

# API端点
@app.post("/chat")
async def chat_endpoint(request: Request, background_tasks: BackgroundTasks):
    """通用聊天端点，支持不同平台"""
    try:
        payload = await request.json()
        
        # 确定平台
        platform = payload.get("platform", "web")
        
        if platform not in platform_adapters:
            raise HTTPException(status_code=400, detail=f"不支持的平台: {platform}")
        
        # 解析消息
        adapter = platform_adapters[platform]
        parsed_message = adapter.parse_message(payload)
        
        user_id = parsed_message.get("user_id")
        message = parsed_message.get("message")
        
        if not user_id or not message:
            raise HTTPException(status_code=400, detail="缺少user_id或message")
        
        # 处理消息
        response = await process_message(message, user_id, platform)
        
        # 如果是Slack，需要异步发送回复
        if platform == "slack":
            background_tasks.add_task(
                send_slack_response,
                response,
                parsed_message.get("channel")
            )
            return {"status": "processing"}
        
        return response
    
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def send_slack_response(response, channel):
    """向Slack发送回复"""
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {slack_token}"
    }
    
    payload = {
        "channel": channel,
        "blocks": response["blocks"]
    }
    
    try:
        requests.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json=payload
        )
    except Exception as e:
        logger.error(f"发送Slack消息时出错: {str(e)}")

@app.delete("/sessions/{user_id}")
async def clear_session(user_id: str):
    """清除用户会话历史"""
    if user_id in user_sessions:
        user_sessions[user_id].clear()
        return {"message": f"用户 {user_id} 的会话已清除"}
    else:
        raise HTTPException(status_code=404, detail="用户会话未找到")

# 启动服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 6. LLM应用开发最佳实践

### 6.1 提示工程与系统设计

- **明确的系统提示**：为每个应用定义清晰的系统提示，设定模型行为边界
- **分层提示策略**：将复杂任务分解为多个步骤，每步使用专门设计的提示
- **模块化设计**：将应用分解为功能独立的模块，便于测试和维护
- **错误处理**：设计健壮的错误处理机制，防止模型失效时应用崩溃
- **用户反馈机制**：收集用户反馈改进提示和应用功能

### 6.2 性能优化与成本控制

- **缓存相似请求**：对频繁查询的问题缓存结果，减少API调用
- **批处理请求**：将多个请求合并处理，减少API调用次数
- **调整模型参数**：根据需求选择合适的模型大小和参数设置
- **本地部署轻量级模型**：对于简单任务，考虑使用本地部署的小型模型
- **监控使用量**：建立监控系统，跟踪API调用和成本

### 6.3 安全性与隐私保护

- **输入过滤**：过滤敏感或有害的用户输入
- **输出审查**：检查模型输出是否符合安全标准
- **数据最小化**：只收集和处理必要的用户数据
- **加密存储**：加密存储敏感信息和对话历史
- **访问控制**：实施细粒度的访问控制机制

## 总结

LLM应用开发是一个快速发展的领域，融合了AI技术、软件工程和用户体验设计。通过使用预训练的大语言模型，开发者可以创建智能交互式应用，解决各种复杂问题。

从简单的聊天机器人到复杂的代理系统，LLM应用的范围非常广泛。关键是理解LLM的能力和局限性，设计适当的提示和系统架构，并实施必要的安全措施和性能优化。

随着技术的不断进步，LLM应用将变得更加强大和易于开发。掌握本文介绍的技术和最佳实践，将使你能够构建高质量、实用的AI应用，为用户提供有价值的服务。

Similar code found with 2 license types