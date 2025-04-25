# 工具调用(Tool Calling)

## 什么是工具调用(Tool Calling)

工具调用(Tool Calling)是指让AI系统能够与外部工具和API交互，以执行超出其原始训练能力的任务的技术。通过工具调用，AI可以：

1. **扩展能力范围**：获取实时信息、执行计算、控制外部服务等
2. **提高回答准确性**：基于最新或专业数据源回答问题
3. **实现具体操作**：代替用户执行实际任务，如查询天气、发送邮件等

工具调用本质上是让AI系统知道"什么时候"和"如何"使用外部工具来完成特定任务。

## 工具调用的基本原理

工具调用的核心流程包括：

1. **工具定义**：明确定义工具的功能、所需参数和返回值
2. **识别需求**：AI识别出需要使用工具的情况
3. **参数提取**：从用户输入或上下文中提取工具所需的参数
4. **工具执行**：调用相应的工具并获取返回结果
5. **结果整合**：将工具执行结果整合到回答中

## 常见的工具调用框架

多种AI框架支持工具调用功能，常见的包括：

- **OpenAI Function Calling**：在GPT模型中集成的功能调用能力
- **LangChain Tools**：提供丰富的预定义工具和自定义工具接口
- **LlamaIndex Tool Abstraction**：为LLM提供工具使用能力
- **AutoGen Tool Use**：在多代理系统中支持工具调用

## 基础示例：使用OpenAI Function Calling

下面是一个使用OpenAI Function Calling的基础示例：

```python
import openai
import json
import os
from datetime import datetime

# 设置OpenAI API密钥
openai.api_key = "你的API密钥"

# 定义一个获取当前天气的函数
def get_weather(location, unit="celsius"):
    """
    获取指定位置的天气信息
    这是一个模拟函数，实际应用中应该调用真实的天气API
    """
    # 这里应该是实际的API调用
    # 为了示例，我们返回一个模拟的结果
    weather_data = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "晴朗",
        "humidity": 60,
        "wind_speed": 5,
        "forecast": ["晴朗", "多云", "小雨"]
    }
    return weather_data

# 定义工具说明
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定位置的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，如'北京'、'上海'等"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，'celsius'(摄氏度)或'fahrenheit'(华氏度)"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 处理用户查询
def process_query(query):
    # 发送请求到OpenAI
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    
    # 检查是否需要调用工具
    if hasattr(message, 'tool_calls') and message.tool_calls:
        # 存储所有工具调用的结果
        tool_results = []
        
        # 处理每个工具调用
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # 调用相应的函数
            if function_name == "get_weather":
                location = function_args.get("location")
                unit = function_args.get("unit", "celsius")
                function_response = get_weather(location, unit)
            else:
                function_response = f"未知函数: {function_name}"
            
            # 将结果添加到消息中
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response, ensure_ascii=False)
            })
        
        # 发送第二个请求，包含工具的结果
        second_response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "user", "content": query},
                message,
                *tool_results
            ]
        )
        
        # 返回最终回答
        return second_response.choices[0].message.content
    
    # 如果不需要调用工具，直接返回回答
    return message.content

# 示例使用
user_query = "北京现在的天气怎么样？"
answer = process_query(user_query)
print(f"问题: {user_query}")
print(f"回答: {answer}")
```

在这个例子中：
1. 我们定义了一个获取天气的函数和相应的工具描述
2. 用户询问北京天气时，AI模型识别出需要使用天气工具
3. 模型提取参数(location="北京")并调用工具
4. 我们将工具结果发回给模型，由它生成最终的人类可读回答

## 使用LangChain实现工具调用

LangChain提供了更简洁的工具调用实现方式：

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
import os

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "你的API密钥"

# 创建自定义工具
@tool
def calculate_mortgage(principal: float, interest_rate: float, years: int) -> str:
    """
    计算每月还款金额
    
    Args:
        principal: 贷款金额(元)
        interest_rate: 年利率(0.05表示5%)
        years: 贷款年限
    
    Returns:
        每月还款金额
    """
    # 计算月利率
    monthly_rate = interest_rate / 12
    # 计算总支付期数
    total_payments = years * 12
    # 计算每月还款金额
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**total_payments) / ((1 + monthly_rate)**total_payments - 1)
    return f"每月还款金额: {monthly_payment:.2f}元"

# 初始化语言模型
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")

# 加载工具
# 这里除了我们自定义的工具外，还使用了LangChain内置的计算器工具
tools = load_tools(["llm-math"], llm=llm)
tools.append(calculate_mortgage)

# 初始化代理
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True  # 显示详细过程
)

# 测试代理
question = "如果我贷款50万元，年利率4.1%，贷款期限30年，每月要还多少钱？然后计算一下总共需要还款多少钱？"
response = agent.run(question)
print(response)
```

在这个例子中：
1. 我们创建了一个自定义的房贷计算工具
2. 结合了LangChain内置的数学计算工具
3. 使用LangChain的代理系统自动处理工具选择、参数提取和结果整合
4. 代理可以综合使用多个工具回答复杂问题

## 实际应用：创建一个具有多工具能力的助手

下面是一个更复杂的例子，展示如何创建一个具有多种工具能力的助手：

```python
import os
import requests
import json
from datetime import datetime
from typing import List, Optional
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# 设置API密钥
os.environ["OPENAI_API_KEY"] = "你的API密钥"

# 天气查询工具
@tool
def get_weather(location: str) -> str:
    """获取指定城市的天气情况"""
    # 这里应该调用真实的天气API
    # 返回模拟数据
    return f"{location}今天天气晴朗，温度24°C，湿度60%，微风"

# 日期时间工具
@tool
def get_current_datetime() -> str:
    """获取当前的日期和时间"""
    now = datetime.now()
    return f"当前日期和时间是: {now.strftime('%Y-%m-%d %H:%M:%S')}"

# 货币转换工具
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    将一种货币转换为另一种货币
    
    Args:
        amount: 金额
        from_currency: 源货币代码，如USD、CNY等
        to_currency: 目标货币代码，如USD、CNY等
    """
    # 这里应该调用实际的货币转换API
    # 使用固定汇率作为示例
    rates = {
        "USD": {"CNY": 7.2, "EUR": 0.93, "JPY": 150.2},
        "CNY": {"USD": 0.14, "EUR": 0.13, "JPY": 20.8},
        "EUR": {"USD": 1.07, "CNY": 7.75, "JPY": 161.5},
        "JPY": {"USD": 0.0067, "CNY": 0.048, "EUR": 0.0062}
    }
    
    if from_currency in rates and to_currency in rates[from_currency]:
        rate = rates[from_currency][to_currency]
        converted_amount = amount * rate
        return f"{amount} {from_currency} = {converted_amount:.2f} {to_currency} (汇率: 1 {from_currency} = {rate} {to_currency})"
    else:
        return f"不支持从 {from_currency} 到 {to_currency} 的转换"

# 查询股票价格工具
@tool
def get_stock_price(symbol: str) -> str:
    """
    获取指定股票代码的最新价格
    
    Args:
        symbol: 股票代码，如AAPL（苹果）、MSFT（微软）等
    """
    # 这里应该调用实际的股票API
    # 使用模拟数据
    stock_data = {
        "AAPL": 173.45,
        "MSFT": 403.78,
        "GOOGL": 164.25,
        "AMZN": 178.75,
        "TSLA": 175.43,
        "BABA": 72.56,
        "PDD": 124.87
    }
    
    if symbol in stock_data:
        return f"{symbol}的最新价格是 ${stock_data[symbol]}"
    else:
        return f"未找到股票代码 {symbol} 的信息"

# 城市信息查询工具
@tool
def get_city_info(city: str) -> str:
    """
    获取城市的基本信息
    
    Args:
        city: 城市名称，如'北京'、'纽约'等
    """
    city_info = {
        "北京": "中国首都，人口约2170万，著名景点包括故宫、长城等",
        "上海": "中国最大的城市，国际金融中心，人口约2480万",
        "纽约": "美国最大城市，全球金融中心，人口约840万",
        "东京": "日本首都，全球最大都市圈，人口约1370万",
        "伦敦": "英国首都，全球重要金融中心，人口约900万"
    }
    
    if city in city_info:
        return city_info[city]
    else:
        return f"没有找到关于{city}的详细信息"

# 计算器工具
@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式的结果
    
    Args:
        expression: 数学表达式，如"(23+42)*2"
    """
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 单位转换工具
@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    进行单位转换
    
    Args:
        value: 数值
        from_unit: 源单位，如"kg"、"mile"等
        to_unit: 目标单位，如"g"、"km"等
    """
    # 定义转换关系
    conversions = {
        "length": {
            "m": 1,
            "km": 1000,
            "cm": 0.01,
            "mile": 1609.34,
            "foot": 0.3048
        },
        "weight": {
            "kg": 1,
            "g": 0.001,
            "pound": 0.453592,
            "ounce": 0.0283495
        },
        "volume": {
            "liter": 1,
            "ml": 0.001,
            "gallon": 3.78541,
            "cup": 0.236588
        }
    }
    
    # 查找单位类别
    unit_category = None
    from_factor = None
    to_factor = None
    
    for category, units in conversions.items():
        if from_unit in units and to_unit in units:
            unit_category = category
            from_factor = units[from_unit]
            to_factor = units[to_unit]
            break
    
    if unit_category:
        # 转换到基本单位，再转换到目标单位
        base_value = value * from_factor
        result = base_value / to_factor
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    else:
        return f"无法转换从 {from_unit} 到 {to_unit}，不支持或不在同一类别"

# 初始化记忆系统
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 初始化语言模型
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")

# 创建工具列表
tools = [
    get_weather,
    get_current_datetime,
    convert_currency,
    get_stock_price,
    get_city_info,
    calculate,
    convert_units
]

# 初始化代理
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True  # 显示详细过程
)

# 创建简单的交互循环
def chat_assistant():
    print("智能助手已启动！(输入'退出'结束对话)")
    
    while True:
        user_input = input("\n你: ")
        
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("助手: 再见！")
            break
        
        try:
            response = agent.run(user_input)
            print(f"助手: {response}")
        except Exception as e:
            print(f"助手: 抱歉，我处理您的请求时遇到了问题: {str(e)}")

# 启动聊天助手
if __name__ == "__main__":
    chat_assistant()
```

在这个完整示例中：
1. 我们定义了多个实用工具：天气查询、日期时间、货币转换、股票价格、城市信息、计算器、单位转换
2. 使用LangChain的代理系统自动选择并调用适当的工具
3. 添加了对话记忆，使助手能够记住之前的交互
4. 提供了一个简单的交互界面供用户测试

## 工具调用的最佳实践

在实现工具调用功能时，以下是一些最佳实践：

### 1. 工具设计原则

- **单一职责**: 每个工具应该专注于单一功能
- **明确参数**: 参数名称和描述应该清晰明确
- **详细描述**: 提供关于工具用途和使用场景的详细信息
- **错误处理**: 妥善处理工具执行过程中可能出现的错误

### 2. 工具使用策略

- **按需调用**: 仅在必要时调用工具，避免过度使用
- **参数验证**: 在调用工具前验证参数的有效性
- **结果整合**: 将工具返回的结果自然地整合到回答中
- **透明度**: 在适当情况下，让用户知道系统正在使用工具

### 3. 安全考虑

- **权限控制**: 限制工具的访问权限和操作范围
- **输入验证**: 防止恶意输入和注入攻击
- **资源限制**: 设置API调用频率和资源使用限制
- **敏感信息保护**: 避免在工具输出中暴露敏感信息

## 案例研究：为特定行业构建工具集

不同行业可能需要特定的工具集。以下是几个行业案例示例：

### 金融助手

金融领域的工具可能包括：
- 股票价格查询
- 投资组合分析
- 贷款计算器
- 货币转换
- 经济指标查询

### 医疗助手

医疗领域的工具可能包括：
- 症状查询
- 药物信息检索
- 医院预约查询
- 健康数据分析
- 医学术语解释

### 教育助手

教育领域的工具可能包括：
- 概念解释
- 问题求解
- 学习资源推荐
- 语言翻译
- 知识测验生成

## 小结

工具调用是扩展AI系统能力的强大方法，能够让AI与外部系统交互并执行更复杂的任务。通过工具调用，AI系统能够：

1. 获取实时数据和信息
2. 执行复杂计算和专业分析
3. 与外部服务和API集成
4. 执行具体的操作和任务

无论是使用OpenAI的Function Calling还是LangChain的工具框架，核心原则都是清晰定义工具功能，识别何时使用工具，以及如何将工具结果整合到回答中。

随着AI技术的发展，工具调用将成为构建更强大、更实用的AI系统的关键组成部分。