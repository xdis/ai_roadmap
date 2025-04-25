# AutoGen框架应用开发

## 什么是AutoGen

AutoGen是微软开源的一个框架，专为构建基于大型语言模型(LLM)的对话式AI代理而设计。它允许开发者创建多个能够相互协作的自主代理，每个代理都可以执行特定的任务、使用工具并与其他代理交流。

AutoGen的核心优势在于：

1. **多代理协作**: 创建多个具有不同角色的代理，协同工作解决复杂问题
2. **自主能力**: 代理可以独立做出决策，无需人类持续干预
3. **丰富的交互方式**: 支持代理之间的对话、代码执行、工具使用等多种交互方式
4. **灵活性**: 易于自定义和扩展，适应各种应用场景

## 安装AutoGen

首先需要安装AutoGen及其依赖：

```bash
pip install pyautogen
```

如果需要使用高级功能，可以选择以下安装方式：

```bash
pip install "pyautogen[teachable,lmm]"  # 包含可教学代理和多模态功能
```

## AutoGen基本概念

### 1. 代理(Agent)

代理是AutoGen的核心概念，它代表一个能够接收信息、处理信息、生成响应并采取行动的实体。AutoGen提供了多种预定义的代理类型：

- **ConversableAgent**: 基础的会话代理，能够参与对话
- **AssistantAgent**: 助手代理，通常由LLM驱动，提供智能回答
- **UserProxyAgent**: 用户代理，可以代表用户与其他代理交互，也可以执行代码
- **GroupChatManager**: 群聊管理器，协调多个代理之间的对话

### 2. 对话(Conversation)

代理之间通过对话进行交流，一个对话由一系列消息组成，每条消息都有发送者、接收者、内容和可能的附加信息。

### 3. 工具(Tool)和函数(Function)

代理可以使用工具和函数来执行特定任务，例如搜索信息、执行代码、访问外部API等。

## 基础示例：简单的助手-用户对话

下面是一个基础示例，展示如何创建一个简单的助手代理和用户代理之间的对话：

```python
import autogen

# 配置OpenAI API
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": "你的OpenAI API密钥"
    }
]

# 创建助手代理
assistant = autogen.AssistantAgent(
    name="助手",
    llm_config={"config_list": config_list}
)

# 创建用户代理
user_proxy = autogen.UserProxyAgent(
    name="用户",
    human_input_mode="TERMINATE",  # 设置为TERMINATE表示对话结束时才请求人类输入
    max_consecutive_auto_reply=10  # 最多自动回复10次
)

# 开始对话
user_proxy.initiate_chat(
    assistant,
    message="你能解释一下什么是递归函数吗？并给出一个Python示例。"
)
```

在这个例子中：
- 我们创建了一个助手代理，配置了大语言模型
- 创建了一个用户代理，设置为对话结束时才请求人类输入
- 用户代理发起了与助手的对话，询问关于递归函数的问题
- 助手会使用LLM生成回答，并可能包含Python代码示例

## 代码执行代理

AutoGen的一个强大功能是允许代理执行代码。下面的例子展示了如何创建一个能够执行Python代码的用户代理：

```python
import autogen

# 配置OpenAI API
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": "你的OpenAI API密钥"
    }
]

# 创建助手代理
assistant = autogen.AssistantAgent(
    name="程序员助手",
    llm_config={"config_list": config_list}
)

# 创建能够执行代码的用户代理
user_proxy = autogen.UserProxyAgent(
    name="用户",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",  # 代码执行的工作目录
        "use_docker": False,   # 不使用Docker执行代码
        "last_n_messages": 3,  # 仅考虑最后3条消息中的代码
    }
)

# 开始对话
user_proxy.initiate_chat(
    assistant,
    message="使用Python创建一个简单的计算器程序，能够进行加减乘除四则运算。"
)
```

在这个例子中：
- 我们配置了用户代理可以执行代码
- 指定了代码执行的工作目录和其他配置
- 当助手回复包含Python代码时，用户代理会自动执行代码并返回结果
- 助手可以看到代码执行结果，并基于结果提供进一步的建议或修改

## 多代理协作：专家组系统

AutoGen的一个核心优势是能够创建多个代理协同工作的系统。下面是一个专家组系统的例子：

```python
import autogen
from autogen import GroupChat, GroupChatManager

# 配置OpenAI API
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": "你的OpenAI API密钥"
    }
]

llm_config = {"config_list": config_list}

# 创建多个专家代理
mathematician = autogen.AssistantAgent(
    name="数学家",
    llm_config=llm_config,
    system_message="你是一名数学专家，专长于解决数学问题。"
)

programmer = autogen.AssistantAgent(
    name="程序员",
    llm_config=llm_config,
    system_message="你是一名经验丰富的Python程序员，擅长编写高效、清晰的代码。"
)

data_scientist = autogen.AssistantAgent(
    name="数据科学家",
    llm_config=llm_config,
    system_message="你是一名数据科学家，擅长数据分析、统计和机器学习算法。"
)

# 创建用户代理
user_proxy = autogen.UserProxyAgent(
    name="用户",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "groupchat",
        "use_docker": False,
    }
)

# 创建群聊
groupchat = GroupChat(
    agents=[user_proxy, mathematician, programmer, data_scientist],
    messages=[],
    max_round=20  # 最多20轮对话
)

# 创建群聊管理器
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# 开始群聊
user_proxy.initiate_chat(
    manager,
    message="我需要编写一个程序来分析股票价格数据，计算移动平均线并可视化结果。请提供完整的解决方案。"
)
```

在这个例子中：
- 我们创建了三个专家代理：数学家、程序员和数据科学家
- 创建了一个用户代理来代表用户与专家互动
- 设置了一个群聊环境，并使用群聊管理器来协调对话
- 用户提出了一个需要多学科知识的问题
- 各个专家代理会根据自己的专长协作解决问题

群聊管理器会智能地决定在每一轮对话中哪个代理最适合回应，确保对话的流畅性和问题的有效解决。

## 自定义代理能力

AutoGen允许你自定义代理的能力，例如添加特定的工具或函数：

```python
import autogen
import requests

# 配置OpenAI API
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": "你的OpenAI API密钥"
    }
]

# 定义一个获取天气的函数
def get_weather(location):
    """获取指定地点的天气信息"""
    # 这是一个示例函数，实际使用时需要替换为真实的API调用
    return f"{location}的天气晴朗，温度25°C"

# 定义一个网络搜索函数
def web_search(query):
    """进行网络搜索"""
    # 这是一个示例函数，实际使用时需要替换为真实的搜索API调用
    return f"搜索结果：关于'{query}'的信息..."

# 创建带有工具的助手代理
assistant = autogen.AssistantAgent(
    name="智能助手",
    llm_config={
        "config_list": config_list,
        "functions": [
            {
                "name": "get_weather",
                "description": "获取指定地点的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "地点名称，如'北京'、'上海'等"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "web_search",
                "description": "进行网络搜索查询",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询内容"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    }
)

# 创建用户代理，注册函数
user_proxy = autogen.UserProxyAgent(
    name="用户",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "functions"}
)

# 注册函数到用户代理
user_proxy.register_function(
    function_map={
        "get_weather": get_weather,
        "web_search": web_search
    }
)

# 开始对话
user_proxy.initiate_chat(
    assistant,
    message="我想知道北京的天气，并搜索一下近期的旅游景点信息。"
)
```

在这个例子中：
- 我们定义了两个自定义函数：获取天气和网络搜索
- 在助手代理的配置中定义了这些函数的接口
- 将实际函数实现注册到用户代理中
- 当助手决定调用这些函数时，用户代理会执行对应的函数并返回结果

## 可教学代理

AutoGen提供了可教学代理功能，允许代理从人类反馈中学习：

```python
import autogen
from autogen.agentchat.contrib.teachable_agent import TeachableAgent

# 配置OpenAI API
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": "你的OpenAI API密钥"
    }
]

# 创建可教学代理
teachable_assistant = TeachableAgent(
    name="可教学助手",
    llm_config={"config_list": config_list},
    teach_config={
        "reset_db": True,  # 重置知识库
        "path_to_db_dir": "./teachable_agent_db"  # 知识库路径
    }
)

# 创建用户代理
user_proxy = autogen.UserProxyAgent(
    name="用户",
    human_input_mode="ALWAYS",  # 始终请求人类输入
    code_execution_config={"work_dir": "teaching"}
)

# 开始对话
user_proxy.initiate_chat(
    teachable_assistant,
    message="你知道我公司的特殊假期政策吗？"
)

# 此时代理可能不知道，用户可以教导它
# 用户输入: "我们公司的特殊假期政策是每年额外提供5天带薪假期，可用于志愿服务。"

# 之后再次询问
user_proxy.initiate_chat(
    teachable_assistant,
    message="再次说明我们公司的特殊假期政策是什么？"
)
# 此时代理应该能够回答正确
```

在这个例子中：
- 我们创建了一个可教学代理，配置了知识库存储路径
- 用户可以通过对话教导代理特定的知识
- 代理会记住这些教导，并在后续的对话中使用

## 使用RAG(检索增强生成)增强代理能力

AutoGen可以与检索增强生成(RAG)技术结合，使代理能够访问外部知识：

```python
import autogen
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb

# 配置OpenAI API
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": "你的OpenAI API密钥"
    }
]

# 创建一个向量数据库客户端
chroma_client = chromadb.Client()
collection_name = "my_documentation"

# 创建或获取集合
try:
    collection = chroma_client.get_collection(collection_name)
except:
    collection = chroma_client.create_collection(collection_name)
    # 添加一些示例文档到集合
    collection.add(
        documents=[
            "AutoGen是微软开发的一个框架，用于构建基于LLM的对话式AI代理。",
            "AutoGen支持多代理协作，允许创建具有不同角色和专长的代理团队。",
            "AutoGen的代理可以执行代码、使用工具和通过对话互相协作。"
        ],
        ids=["doc1", "doc2", "doc3"],
        metadatas=[{"source": "docs"}, {"source": "docs"}, {"source": "docs"}]
    )

# 创建检索增强的用户代理
retrieve_user_proxy = RetrieveUserProxyAgent(
    name="RAG用户",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "qa",
        "docs_path": "./",  # 文档路径（此处不使用，因为我们用的是内存中的向量数据库）
        "chromadb_config": {
            "client": chroma_client,
            "collection_name": collection_name
        }
    }
)

# 创建助手代理
assistant = autogen.AssistantAgent(
    name="知识库助手",
    llm_config={"config_list": config_list}
)

# 开始对话
retrieve_user_proxy.initiate_chat(
    assistant,
    message="请告诉我关于AutoGen的信息？"
)
```

在这个例子中：
- 我们创建了一个ChromaDB向量数据库，并添加了一些关于AutoGen的文档
- 创建了一个检索增强的用户代理，配置了向量数据库连接
- 当用户提问时，代理会从向量数据库中检索相关信息，并将这些信息提供给助手
- 助手可以基于检索到的信息提供更准确的回答

## 实际应用：创建一个数据分析助手

下面是一个更复杂的示例，展示如何创建一个数据分析助手：

```python
import autogen
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 配置OpenAI API
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": "你的OpenAI API密钥"
    }
]

# 创建数据科学助手
data_scientist = autogen.AssistantAgent(
    name="数据分析师",
    llm_config={
        "config_list": config_list,
        "temperature": 0,  # 降低随机性，使回答更确定
    },
    system_message="""
    你是一名专业的数据分析师。你的职责是帮助用户分析数据、创建可视化图表、解释数据趋势和发现数据中的洞见。
    你应该生成清晰、高效的Python代码，优先使用pandas、matplotlib和seaborn等库。
    确保你的代码包含详细注释，并解释每个分析步骤的目的和结果。
    """
)

# 创建用户代理
user_proxy = autogen.UserProxyAgent(
    name="用户",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=5,
    code_execution_config={
        "work_dir": "data_analysis",
        "use_docker": False,
    }
)

# 创建示例数据
sample_data = pd.DataFrame({
    '日期': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    '销售量': [100 + i + 10 * (i % 7 == 5) + 15 * (i % 7 == 6) - 20 * (i % 30 == 0) + 5 * i // 10 for i in range(100)],
    '广告费用': [50 + 0.5 * i + 2 * (i % 7 == 0) + 5 * (i % 20 < 5) for i in range(100)],
    '客户满意度': [4.0 + 0.5 * (i % 5 == 0) - 0.5 * (i % 30 == 0) + 0.1 * (i // 10) for i in range(100)]
})

# 保存示例数据
sample_data.to_csv("data_analysis/sales_data.csv", index=False)

# 开始对话
user_proxy.initiate_chat(
    data_scientist,
    message="""
    我有一个销售数据集，包含日期、销售量、广告费用和客户满意度。
    数据文件保存在'sales_data.csv'中。请帮我分析以下问题：
    1. 销售量与广告费用之间是否有相关性？
    2. 客户满意度与销售量是如何关联的？
    3. 销售数据是否存在周期性趋势？
    4. 创建一个综合仪表板，展示关键指标和趋势。
    
    请提供详细的分析和可视化结果。
    """
)
```

在这个例子中：
- 我们创建了一个专门的数据分析师代理，并提供了详细的系统消息
- 创建了一个可执行代码的用户代理
- 生成了一个示例销售数据集
- 用户提出了一系列数据分析问题
- 数据分析师会生成代码来加载数据、分析趋势、创建可视化并解释结果

## 小结

AutoGen是一个功能强大的框架，可以帮助开发者创建协作式的AI代理系统。主要优势包括：

1. **灵活的代理架构**: 可以创建各种类型的代理，如助手、用户代理、群聊管理器等
2. **代码执行能力**: 代理可以生成和执行代码，实现复杂的任务
3. **多代理协作**: 支持多个代理之间的协作，组成专家团队解决问题
4. **工具和函数集成**: 可以为代理添加自定义工具和函数，扩展其能力
5. **可教学性**: 代理可以从人类反馈中学习，提高其性能
6. **与RAG集成**: 可以通过检索增强生成技术让代理访问外部知识

通过本文介绍的基本概念和代码示例，你可以开始使用AutoGen构建自己的代理系统。随着经验的积累，你可以探索更复杂的功能，如定制专用代理、构建复杂的多代理系统、集成外部API和服务等。

## 进阶学习资源

- [AutoGen官方文档](https://microsoft.github.io/autogen/)
- [AutoGen GitHub仓库](https://github.com/microsoft/autogen)
- [AutoGen示例集](https://github.com/microsoft/autogen/tree/main/notebook)