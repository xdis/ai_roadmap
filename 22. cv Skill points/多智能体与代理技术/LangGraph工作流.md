# LangGraph工作流应用开发

## 什么是LangGraph

LangGraph是一个基于LangChain构建的框架，专为创建复杂的、有状态的多智能体工作流而设计。它将图结构与大型语言模型(LLM)的能力相结合，使开发者能够创建更加灵活、可维护的AI工作流程。

LangGraph的核心价值在于：

1. **状态管理**: 提供强大的状态管理能力，跟踪工作流中的变量和条件
2. **组件化工作流**: 将复杂流程分解为可重用的节点和边
3. **灵活的流程控制**: 支持条件分支、循环和动态路径选择
4. **多智能体协作**: 便于构建多个代理协同工作的系统

## 安装LangGraph

首先需要安装LangGraph及其依赖：

```bash
pip install langgraph
# 如果需要使用LangChain功能
pip install langchain
```

## LangGraph基本概念

### 1. 图(Graph)

LangGraph的核心是图结构，包含节点(Nodes)和边(Edges)：
- **节点**: 执行特定任务的组件，如思考、决策、工具使用等
- **边**: 定义节点间的转换逻辑，可以是条件式的或固定的

### 2. 状态(State)

状态是工作流执行过程中的数据容器，包含：
- 消息历史
- 中间结果
- 环境变量
- 智能体状态

### 3. 通道(Channel)

用于节点间通信的机制，可以传递消息、事件或任务。

## 基础示例：简单的决策工作流

下面是一个基础示例，展示如何创建一个简单的决策工作流：

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "你的API密钥"

# 定义状态类型
class SimpleState(TypedDict):
    question: str
    thinking: str
    answer: str

# 创建语言模型
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 定义思考节点
def think(state: SimpleState) -> SimpleState:
    """思考问题的答案"""
    question = state["question"]
    
    # 创建思考提示
    prompt = PromptTemplate.from_template(
        "问题: {question}\n\n请仔细思考这个问题，分析所有相关因素:"
    )
    
    # 生成思考内容
    thinking = llm.invoke(prompt.format(question=question)).content
    
    # 更新状态
    return {"thinking": thinking}

# 定义回答节点
def answer(state: SimpleState) -> SimpleState:
    """基于思考给出答案"""
    question = state["question"]
    thinking = state["thinking"]
    
    # 创建回答提示
    prompt = PromptTemplate.from_template(
        "问题: {question}\n\n思考过程: {thinking}\n\n基于以上思考，给出简洁明了的最终答案:"
    )
    
    # 生成答案
    final_answer = llm.invoke(prompt.format(question=question, thinking=thinking)).content
    
    # 更新状态
    return {"answer": final_answer}

# 创建工作流图
workflow = StateGraph(SimpleState)

# 添加节点
workflow.add_node("think", think)
workflow.add_node("answer", answer)

# 定义边（流程）
workflow.set_entry_point("think")  # 设置入口节点
workflow.add_edge("think", "answer")  # 从思考到回答
workflow.add_edge("answer", END)  # 回答后结束

# 编译工作流
app = workflow.compile()

# 使用工作流
result = app.invoke({"question": "人工智能可能带来哪些社会影响？"})
print("思考过程:", result["thinking"])
print("\n最终答案:", result["answer"])
```

在这个例子中：
- 我们定义了一个包含问题、思考和答案的简单状态
- 创建了两个节点：一个用于思考，一个用于给出答案
- 设置了从思考到回答的工作流程
- 使用工作流解答了一个关于AI社会影响的问题

## 条件分支工作流

LangGraph支持基于条件的分支流程，下面是一个示例：

```python
from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import EnumOutputParser

# 定义状态类型
class MathProblemState(TypedDict):
    problem: str
    problem_type: str
    solution_path: str
    solution: str

# 创建语言模型
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 定义问题分类节点
def classify_problem(state: MathProblemState) -> MathProblemState:
    """识别数学问题的类型"""
    problem = state["problem"]
    
    # 创建分类提示
    prompt = PromptTemplate.from_template(
        "数学问题: {problem}\n\n这是哪种类型的数学问题？只回答以下选项之一: 代数, 几何, 统计, 微积分"
    )
    
    # 使用输出解析器确保获得有效的分类
    parser = EnumOutputParser(enum_values=["代数", "几何", "统计", "微积分"])
    
    # 生成问题类型
    problem_type = parser.parse(llm.invoke(prompt.format(problem=problem)).content)
    
    # 更新状态
    return {"problem_type": problem_type}

# 定义代数解法节点
def solve_algebra(state: MathProblemState) -> MathProblemState:
    """解决代数问题"""
    problem = state["problem"]
    
    # 创建解题提示
    prompt = PromptTemplate.from_template(
        "代数问题: {problem}\n\n请详细解答这个代数问题，给出解题步骤:"
    )
    
    # 生成解题路径
    solution_path = llm.invoke(prompt.format(problem=problem)).content
    
    # 更新状态
    return {"solution_path": solution_path}

# 定义几何解法节点
def solve_geometry(state: MathProblemState) -> MathProblemState:
    """解决几何问题"""
    problem = state["problem"]
    
    # 创建解题提示
    prompt = PromptTemplate.from_template(
        "几何问题: {problem}\n\n请详细解答这个几何问题，给出解题步骤:"
    )
    
    # 生成解题路径
    solution_path = llm.invoke(prompt.format(problem=problem)).content
    
    # 更新状态
    return {"solution_path": solution_path}

# 定义统计解法节点
def solve_statistics(state: MathProblemState) -> MathProblemState:
    """解决统计问题"""
    problem = state["problem"]
    
    # 创建解题提示
    prompt = PromptTemplate.from_template(
        "统计问题: {problem}\n\n请详细解答这个统计问题，给出解题步骤:"
    )
    
    # 生成解题路径
    solution_path = llm.invoke(prompt.format(problem=problem)).content
    
    # 更新状态
    return {"solution_path": solution_path}

# 定义微积分解法节点
def solve_calculus(state: MathProblemState) -> MathProblemState:
    """解决微积分问题"""
    problem = state["problem"]
    
    # 创建解题提示
    prompt = PromptTemplate.from_template(
        "微积分问题: {problem}\n\n请详细解答这个微积分问题，给出解题步骤:"
    )
    
    # 生成解题路径
    solution_path = llm.invoke(prompt.format(problem=problem)).content
    
    # 更新状态
    return {"solution_path": solution_path}

# 定义总结节点
def summarize_solution(state: MathProblemState) -> MathProblemState:
    """总结最终答案"""
    problem = state["problem"]
    solution_path = state["solution_path"]
    
    # 创建总结提示
    prompt = PromptTemplate.from_template(
        "问题: {problem}\n\n解题过程: {solution_path}\n\n请简洁清晰地总结最终答案:"
    )
    
    # 生成最终答案
    solution = llm.invoke(prompt.format(problem=problem, solution_path=solution_path)).content
    
    # 更新状态
    return {"solution": solution}

# 定义路由函数
def route_to_solver(state: MathProblemState) -> str:
    """根据问题类型选择合适的解题方法"""
    return state["problem_type"]

# 创建工作流图
workflow = StateGraph(MathProblemState)

# 添加节点
workflow.add_node("classify", classify_problem)
workflow.add_node("algebra", solve_algebra)
workflow.add_node("geometry", solve_geometry)
workflow.add_node("statistics", solve_statistics)
workflow.add_node("calculus", solve_calculus)
workflow.add_node("summarize", summarize_solution)

# 设置入口点
workflow.set_entry_point("classify")

# 添加条件分支
workflow.add_conditional_edges(
    "classify",
    route_to_solver,
    {
        "代数": "algebra",
        "几何": "geometry",
        "统计": "statistics",
        "微积分": "calculus"
    }
)

# 添加到总结的边
workflow.add_edge("algebra", "summarize")
workflow.add_edge("geometry", "summarize")
workflow.add_edge("statistics", "summarize")
workflow.add_edge("calculus", "summarize")
workflow.add_edge("summarize", END)

# 编译工作流
app = workflow.compile()

# 使用工作流
result = app.invoke({
    "problem": "在一个三角形中，已知两边长分别为3和4，夹角为60度，求第三边长度。"
})

print("问题类型:", result["problem_type"])
print("\n解题过程:")
print(result["solution_path"])
print("\n最终答案:")
print(result["solution"])
```

在这个例子中：
- 我们创建了一个数学问题解答工作流
- 首先分类数学问题的类型（代数、几何、统计或微积分）
- 根据问题类型选择不同的解题方法
- 最后总结出最终答案
- 使用条件分支来实现动态选择解题路径

## 循环工作流：反思与改进

LangGraph支持创建循环工作流，允许代理反复迭代直到达到特定条件：

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import EnumOutputParser

# 定义状态类型
class EssayState(TypedDict):
    topic: str
    current_draft: str
    feedback: str
    revision_count: int
    is_complete: bool

# 创建语言模型
llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

# 定义初始草稿节点
def write_initial_draft(state: EssayState) -> EssayState:
    """创建初始文章草稿"""
    topic = state["topic"]
    
    # 创建提示
    prompt = PromptTemplate.from_template(
        "主题: {topic}\n\n请写一篇关于这个主题的文章草稿，大约500字:"
    )
    
    # 生成初始草稿
    draft = llm.invoke(prompt.format(topic=topic)).content
    
    # 更新状态
    return {
        "current_draft": draft,
        "revision_count": 0,
        "is_complete": False
    }

# 定义评估节点
def evaluate_draft(state: EssayState) -> EssayState:
    """评估当前草稿并提供反馈"""
    draft = state["current_draft"]
    revision_count = state["revision_count"]
    
    # 创建评估提示
    prompt = PromptTemplate.from_template(
        "文章草稿:\n\n{draft}\n\n这是第{revision_count}稿。请评估这篇文章的质量，"
        "并提供具体的改进建议。考虑结构、内容、语言表达等方面:"
    )
    
    # 生成反馈
    feedback = llm.invoke(prompt.format(
        draft=draft,
        revision_count=revision_count
    )).content
    
    # 更新状态
    return {"feedback": feedback}

# 定义修改节点
def revise_draft(state: EssayState) -> EssayState:
    """根据反馈修改草稿"""
    draft = state["current_draft"]
    feedback = state["feedback"]
    revision_count = state["revision_count"]
    
    # 创建修改提示
    prompt = PromptTemplate.from_template(
        "原始草稿:\n\n{draft}\n\n反馈意见:\n\n{feedback}\n\n"
        "请根据以上反馈修改草稿，输出完整的修改版本:"
    )
    
    # 生成修改后的草稿
    revised_draft = llm.invoke(prompt.format(
        draft=draft,
        feedback=feedback
    )).content
    
    # 更新状态
    return {
        "current_draft": revised_draft,
        "revision_count": revision_count + 1
    }

# 定义完成检查节点
def check_completion(state: EssayState) -> EssayState:
    """检查是否完成修改"""
    draft = state["current_draft"]
    feedback = state["feedback"]
    revision_count = state["revision_count"]
    
    # 如果已经修改了3次或达到了质量标准，则完成
    if revision_count >= 3:
        return {"is_complete": True}
    
    # 创建评估提示
    prompt = PromptTemplate.from_template(
        "文章草稿:\n\n{draft}\n\n反馈意见:\n\n{feedback}\n\n"
        "基于以上草稿和反馈，这篇文章是否已经达到高质量标准？只回答 '是' 或 '否':"
    )
    
    # 判断是否完成
    completion_response = llm.invoke(prompt.format(
        draft=draft,
        feedback=feedback
    )).content.strip().lower()
    
    is_complete = "是" in completion_response
    
    # 更新状态
    return {"is_complete": is_complete}

# 定义路由函数
def should_continue(state: EssayState) -> str:
    """决定是继续修改还是完成"""
    if state["is_complete"]:
        return "complete"
    else:
        return "continue"

# 创建工作流图
workflow = StateGraph(EssayState)

# 添加节点
workflow.add_node("write_draft", write_initial_draft)
workflow.add_node("evaluate", evaluate_draft)
workflow.add_node("revise", revise_draft)
workflow.add_node("check", check_completion)

# 设置入口点
workflow.set_entry_point("write_draft")

# 添加边
workflow.add_edge("write_draft", "evaluate")
workflow.add_edge("evaluate", "revise")
workflow.add_edge("revise", "check")

# 添加条件边（循环）
workflow.add_conditional_edges(
    "check",
    should_continue,
    {
        "continue": "evaluate",  # 继续评估和修改
        "complete": END  # 完成
    }
)

# 编译工作流
app = workflow.compile()

# 使用工作流
result = app.invoke({
    "topic": "人工智能对未来教育的影响"
})

print(f"经过{result['revision_count']}次修改后的最终文章:")
print(result["current_draft"])
```

在这个例子中：
- 我们创建了一个文章写作和修改的工作流
- 工作流程包括：初始草稿、评估、修改和完成检查
- 使用循环结构，允许多次评估和修改
- 设置结束条件：修改次数达到3次或评估认为质量已足够高

## 多代理协作工作流

LangGraph非常适合构建多代理协作系统，下面是一个示例：

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI

# 定义状态类型
class TeamState(TypedDict):
    messages: List[Dict[str, Any]]
    task: str
    current_agent: str
    solution: str
    is_complete: bool

# 创建语言模型
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 定义代理生成响应函数
def generate_agent_response(agent_name: str, agent_role: str, task: str, messages: List[Dict[str, Any]]) -> str:
    """生成特定代理的响应"""
    # 构建代理的系统提示
    system_prompt = f"你是{agent_name}，{agent_role}。你的任务是与团队合作解决问题。"
    
    # 转换消息格式
    prompt_messages = [SystemMessage(content=system_prompt)]
    
    # 添加任务描述
    prompt_messages.append(HumanMessage(content=f"任务: {task}"))
    
    # 添加历史消息
    for msg in messages:
        if msg["role"] == "ai":
            prompt_messages.append(AIMessage(content=f"{msg['agent']}: {msg['content']}"))
        else:
            prompt_messages.append(HumanMessage(content=f"{msg['agent']}: {msg['content']}"))
    
    # 生成响应
    response = llm.invoke(prompt_messages).content
    
    return response

# 定义各个代理节点
def product_manager(state: TeamState) -> TeamState:
    """产品经理代理"""
    messages = state["messages"]
    task = state["task"]
    
    response = generate_agent_response(
        agent_name="产品经理",
        agent_role="负责定义需求、确保产品符合用户期望并协调团队",
        task=task,
        messages=messages
    )
    
    # 添加消息到历史
    messages.append({
        "role": "ai",
        "agent": "产品经理",
        "content": response
    })
    
    # 更新状态
    return {
        "messages": messages,
        "current_agent": "designer"
    }

def designer(state: TeamState) -> TeamState:
    """设计师代理"""
    messages = state["messages"]
    task = state["task"]
    
    response = generate_agent_response(
        agent_name="设计师",
        agent_role="负责产品的UI/UX设计，确保用户体验良好",
        task=task,
        messages=messages
    )
    
    # 添加消息到历史
    messages.append({
        "role": "ai",
        "agent": "设计师",
        "content": response
    })
    
    # 更新状态
    return {
        "messages": messages,
        "current_agent": "developer"
    }

def developer(state: TeamState) -> TeamState:
    """开发者代理"""
    messages = state["messages"]
    task = state["task"]
    
    response = generate_agent_response(
        agent_name="开发者",
        agent_role="负责实现产品功能，编写代码",
        task=task,
        messages=messages
    )
    
    # 添加消息到历史
    messages.append({
        "role": "ai",
        "agent": "开发者",
        "content": response
    })
    
    # 更新状态
    return {
        "messages": messages,
        "current_agent": "tester"
    }

def tester(state: TeamState) -> TeamState:
    """测试员代理"""
    messages = state["messages"]
    task = state["task"]
    
    response = generate_agent_response(
        agent_name="测试员",
        agent_role="负责测试产品功能，确保质量",
        task=task,
        messages=messages
    )
    
    # 添加消息到历史
    messages.append({
        "role": "ai",
        "agent": "测试员",
        "content": response
    })
    
    # 更新状态
    return {
        "messages": messages,
        "current_agent": "finalizer"
    }

def finalizer(state: TeamState) -> TeamState:
    """总结解决方案"""
    messages = state["messages"]
    task = state["task"]
    
    # 创建总结提示
    prompt = PromptTemplate.from_template(
        "任务: {task}\n\n团队讨论:\n{discussion}\n\n"
        "请总结团队的最终解决方案，包括产品需求、设计、实现和测试计划:"
    )
    
    # 转换消息为文本讨论
    discussion = "\n".join([f"{msg['agent']}: {msg['content']}" for msg in messages])
    
    # 生成解决方案总结
    solution = llm.invoke(prompt.format(task=task, discussion=discussion)).content
    
    # 更新状态
    return {
        "solution": solution,
        "is_complete": True
    }

# 定义路由函数
def get_next_agent(state: TeamState) -> str:
    """决定下一个发言的代理"""
    current = state["current_agent"]
    if current == "finalizer" or state.get("is_complete", False):
        return "complete"
    else:
        return current

# 创建工作流图
workflow = StateGraph(TeamState)

# 添加节点
workflow.add_node("product_manager", product_manager)
workflow.add_node("designer", designer)
workflow.add_node("developer", developer)
workflow.add_node("tester", tester)
workflow.add_node("finalizer", finalizer)

# 设置入口点
workflow.set_entry_point("product_manager")

# 添加条件边
workflow.add_conditional_edges(
    "product_manager",
    get_next_agent,
    {
        "designer": "designer",
        "complete": END
    }
)

workflow.add_conditional_edges(
    "designer",
    get_next_agent,
    {
        "developer": "developer",
        "complete": END
    }
)

workflow.add_conditional_edges(
    "developer",
    get_next_agent,
    {
        "tester": "tester",
        "complete": END
    }
)

workflow.add_conditional_edges(
    "tester",
    get_next_agent,
    {
        "finalizer": "finalizer",
        "complete": END
    }
)

workflow.add_conditional_edges(
    "finalizer",
    get_next_agent,
    {
        "complete": END
    }
)

# 编译工作流
app = workflow.compile()

# 使用工作流
result = app.invoke({
    "task": "设计并开发一个智能家居控制应用，能够通过手机控制家中的灯光、温度和安防系统。",
    "messages": [],
    "current_agent": "product_manager",
    "solution": "",
    "is_complete": False
})

print("团队讨论:")
for message in result["messages"]:
    print(f"{message['agent']}: {message['content']}\n")

print("\n最终解决方案:")
print(result["solution"])
```

在这个例子中：
- 我们创建了一个产品开发团队的多代理协作工作流
- 团队包括产品经理、设计师、开发者和测试员
- 各个代理按顺序贡献自己的专业知识
- 最后由总结器生成综合解决方案
- 代理之间通过消息历史进行沟通和协作

## 工具使用工作流

LangGraph可以与工具集成，创建能够使用外部工具的代理工作流：

```python
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage, FunctionMessage
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
import requests
import json

# 定义工具函数
@tool
def get_weather(location: str) -> str:
    """获取指定地点的天气信息"""
    # 这是一个示例函数，实际使用时需要替换为真实的API调用
    return f"{location}的天气晴朗，温度25°C，湿度60%"

@tool
def search_information(query: str) -> str:
    """搜索特定信息"""
    # 这是一个示例函数，实际使用时需要替换为真实的搜索API调用
    return f"关于'{query}'的搜索结果：这是一些相关信息..."

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {str(e)}"

# 定义状态类型
class AssistantState(TypedDict):
    messages: List[Dict[str, Any]]
    question: str
    thinking: Optional[str]
    tools_to_use: List[str]
    tool_results: Dict[str, str]
    answer: Optional[str]

# 创建语言模型
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 定义思考节点
def think(state: AssistantState) -> AssistantState:
    """思考如何回答问题，确定需要使用的工具"""
    question = state["question"]
    
    # 创建思考提示
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个智能助手，可以使用各种工具来回答问题。可用的工具有：天气查询、信息搜索、计算器。"),
        HumanMessage(content=f"问题: {question}\n\n思考这个问题需要哪些工具来帮助回答。列出你需要使用的工具名称（天气查询、信息搜索或计算器）。")
    ])
    
    # 生成思考过程
    thinking = llm.invoke(prompt).content
    
    # 确定需要使用的工具
    tools_to_use = []
    if "天气" in thinking.lower():
        tools_to_use.append("weather")
    if "搜索" in thinking.lower() or "信息" in thinking.lower():
        tools_to_use.append("search")
    if "计算" in thinking.lower():
        tools_to_use.append("calculator")
    
    # 更新状态
    return {
        "thinking": thinking,
        "tools_to_use": tools_to_use,
        "tool_results": {}
    }

# 定义使用天气工具节点
def use_weather_tool(state: AssistantState) -> AssistantState:
    """使用天气工具"""
    if "weather" not in state["tools_to_use"]:
        return {}  # 不需要使用该工具
    
    question = state["question"]
    
    # 创建提示
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="从问题中提取需要查询天气的地点。"),
        HumanMessage(content=f"问题: {question}\n\n请提取出需要查询天气的地点名称:")
    ])
    
    # 提取地点
    location = llm.invoke(prompt).content
    
    # 使用工具获取天气
    weather_result = get_weather(location)
    
    # 更新工具结果
    tool_results = state.get("tool_results", {})
    tool_results["weather"] = weather_result
    
    # 更新状态
    return {"tool_results": tool_results}

# 定义使用搜索工具节点
def use_search_tool(state: AssistantState) -> AssistantState:
    """使用搜索工具"""
    if "search" not in state["tools_to_use"]:
        return {}  # 不需要使用该工具
    
    question = state["question"]
    
    # 创建提示
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="从问题中提取需要搜索的关键信息。"),
        HumanMessage(content=f"问题: {question}\n\n请提取出需要搜索的关键词或短语:")
    ])
    
    # 提取搜索查询
    search_query = llm.invoke(prompt).content
    
    # 使用工具搜索信息
    search_result = search_information(search_query)
    
    # 更新工具结果
    tool_results = state.get("tool_results", {})
    tool_results["search"] = search_result
    
    # 更新状态
    return {"tool_results": tool_results}

# 定义使用计算工具节点
def use_calculator_tool(state: AssistantState) -> AssistantState:
    """使用计算工具"""
    if "calculator" not in state["tools_to_use"]:
        return {}  # 不需要使用该工具
    
    question = state["question"]
    
    # 创建提示
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="从问题中提取需要计算的数学表达式。"),
        HumanMessage(content=f"问题: {question}\n\n请提取出需要计算的数学表达式，使用Python语法表示:")
    ])
    
    # 提取表达式
    expression = llm.invoke(prompt).content
    
    # 使用工具计算
    calc_result = calculate(expression)
    
    # 更新工具结果
    tool_results = state.get("tool_results", {})
    tool_results["calculator"] = calc_result
    
    # 更新状态
    return {"tool_results": tool_results}

# 定义答案生成节点
def generate_answer(state: AssistantState) -> AssistantState:
    """生成最终答案"""
    question = state["question"]
    thinking = state["thinking"]
    tool_results = state["tool_results"]
    
    # 创建提示
    tools_info = "\n".join([f"{k}: {v}" for k, v in tool_results.items()])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个智能助手，使用工具结果来回答用户问题。"),
        HumanMessage(content=f"问题: {question}\n\n你的思考: {thinking}\n\n工具结果:\n{tools_info}\n\n基于以上信息，请提供完整、准确的回答:")
    ])
    
    # 生成答案
    answer = llm.invoke(prompt).content
    
    # 更新状态
    return {"answer": answer}

# 创建工作流图
workflow = StateGraph(AssistantState)

# 添加节点
workflow.add_node("think", think)
workflow.add_node("use_weather_tool", use_weather_tool)
workflow.add_node("use_search_tool", use_search_tool)
workflow.add_node("use_calculator_tool", use_calculator_tool)
workflow.add_node("generate_answer", generate_answer)

# 设置入口点
workflow.set_entry_point("think")

# 添加边
workflow.add_edge("think", "use_weather_tool")
workflow.add_edge("use_weather_tool", "use_search_tool")
workflow.add_edge("use_search_tool", "use_calculator_tool")
workflow.add_edge("use_calculator_tool", "generate_answer")
workflow.add_edge("generate_answer", END)

# 编译工作流
app = workflow.compile()

# 使用工作流
result = app.invoke({
    "question": "北京明天的天气怎么样？顺便帮我计算一下25乘以4的结果。",
    "messages": []
})

print("思考过程:", result["thinking"])
print("\n使用的工具:", result["tools_to_use"])
print("\n工具结果:", result["tool_results"])
print("\n最终答案:", result["answer"])
```

在这个例子中：
- 我们创建了一个具有工具使用能力的助手工作流
- 助手可以使用天气查询、信息搜索和计算等工具
- 工作流程包括：思考要使用的工具、使用各种工具、最后生成答案
- 这个模式可以扩展到更多复杂的工具使用场景

## 小结

LangGraph是一个强大的框架，它使开发者能够创建复杂的、有状态的多智能体工作流。它的核心优势在于：

1. **结构化工作流**: 使用图结构组织复杂的AI流程
2. **状态管理**: 高效跟踪工作流中的各种状态
3. **灵活路由**: 支持条件分支、循环和动态路径选择
4. **多代理协作**: 便于构建多个代理协同工作的系统
5. **工具使用**: 可以与外部工具和API集成

通过本文介绍的基本概念和代码示例，你可以开始使用LangGraph构建自己的智能工作流。随着经验的积累，你可以探索更复杂的功能，如嵌套子流程、并行执行、异步工作流等。

## 进阶学习资源

- [LangGraph官方文档](https://github.com/langchain-ai/langgraph)
- [LangChain文档](https://python.langchain.com/)
- [LangGraph示例](https://github.com/langchain-ai/langgraph/tree/main/examples)