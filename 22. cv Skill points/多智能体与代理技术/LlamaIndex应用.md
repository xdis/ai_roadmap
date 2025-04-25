# LlamaIndex应用开发

## 什么是LlamaIndex

LlamaIndex（原名GPT Index）是一个数据框架，它可以帮助你使用大型语言模型(LLM)构建高级应用，特别是那些需要基于外部数据回答问题的应用。LlamaIndex的核心功能是：

1. **数据连接与索引**: 帮助你连接、处理和索引私有或特定领域的数据
2. **基于检索的生成**: 根据索引数据生成高质量的回答
3. **结构化知识**: 将非结构化数据转换为结构化格式，便于查询和推理

## 为什么需要LlamaIndex

大型语言模型虽然功能强大，但它们有两个主要限制：

1. **知识截止日期**: 模型训练数据有截止日期，无法了解之后的信息
2. **无法访问私有数据**: 模型无法直接访问你的特定数据（如公司文档、个人笔记等）

LlamaIndex通过建立数据与LLM之间的桥梁解决这些问题，允许LLM基于最新的或专有的数据来回答问题。

## 安装LlamaIndex

首先需要安装LlamaIndex及其依赖：

```bash
pip install llama-index
pip install llama-index-llms-openai  # 如果使用OpenAI模型
pip install pypdf  # 如果需要处理PDF文件
```

## LlamaIndex基本工作流程

LlamaIndex的典型工作流程包括以下步骤：

1. **加载数据**: 从各种来源（文档、API、数据库等）获取数据
2. **索引数据**: 将数据解析并索引，为查询做准备
3. **查询数据**: 使用自然语言查询索引，获取相关信息和答案

## 基础示例：简单文档问答

下面是一个基础示例，展示如何使用LlamaIndex创建一个简单的文档问答系统：

```python
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "你的API密钥"

# 1. 加载文档
documents = SimpleDirectoryReader("./data").load_data()
print(f"加载了 {len(documents)} 个文档")

# 2. 创建索引
# VectorStoreIndex将文档切分成块，并创建向量嵌入
llm = OpenAI(model="gpt-3.5-turbo")
index = VectorStoreIndex.from_documents(documents, llm=llm)

# 3. 创建查询引擎
query_engine = index.as_query_engine()

# 4. 进行查询
response = query_engine.query("文档中提到了哪些重要概念?")
print(response)
```

这个例子中：
- 我们从一个目录加载所有文档
- 创建一个向量存储索引，将文档内容转换为向量表示
- 创建一个查询引擎
- 使用自然语言提问并获取回答

## 自定义文档处理

LlamaIndex支持多种类型的文档，并允许自定义文档处理：

```python
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI

# 创建自定义文档
text = """
人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
机器学习是AI的一个子领域，专注于使用数据和算法来模仿人类的学习方式，逐渐提高准确性。
深度学习是机器学习的一种特殊形式，使用神经网络处理数据，类似于人脑的工作方式。
"""

document = Document(text=text)

# 自定义节点解析器
parser = SimpleNodeParser.from_defaults(
    chunk_size=100,  # 每个块的最大字符数
    chunk_overlap=20  # 块之间的重叠字符数
)

# 将文档分割成节点
nodes = parser.get_nodes_from_documents([document])
print(f"文档被分割成 {len(nodes)} 个节点")

# 创建索引
llm = OpenAI(model="gpt-3.5-turbo")
index = VectorStoreIndex(nodes, llm=llm)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("什么是深度学习?")
print(response)
```

在这个例子中：
- 我们创建了一个包含人工智能相关文本的自定义文档
- 使用SimpleNodeParser将文档分割成更小的节点（块）
- 自定义了块的大小和重叠度
- 基于这些节点创建索引并进行查询

## 持久化索引

在实际应用中，为了避免重复索引大量数据，可以将索引保存到磁盘：

```python
from llama_index.core import StorageContext, load_index_from_storage

# 假设我们已经创建了索引
# 保存索引到磁盘
index.storage_context.persist("./storage")

# 之后加载索引
storage_context = StorageContext.from_defaults(persist_dir="./storage")
loaded_index = load_index_from_storage(storage_context)

# 使用加载的索引
query_engine = loaded_index.as_query_engine()
response = query_engine.query("机器学习与深度学习的区别是什么?")
print(response)
```

## 高级查询技术

LlamaIndex支持多种高级查询技术，如下面的示例：

```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine

# 假设我们已经有了索引
# 自定义检索器
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3  # 检索最相关的3个节点
)

# 自定义响应合成器
response_synthesizer = CompactAndRefine(
    llm=OpenAI(model="gpt-3.5-turbo"),
    verbose=True  # 显示详细过程
)

# 创建自定义查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

# 查询
response = query_engine.query("比较机器学习和深度学习")
print(response)
```

在这个例子中：
- 我们创建了一个自定义检索器，指定检索最相关的3个节点
- 使用了CompactAndRefine响应合成器，这是一种高效处理长文本的方法
- 将检索器和响应合成器组合成自定义查询引擎

## 知识图谱构建

LlamaIndex还支持构建知识图谱，这对于复杂的知识体系特别有用：

```python
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex

# 假设已有文档集合
# 创建知识图谱索引
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    llm=OpenAI(model="gpt-3.5-turbo")
)

# 可视化知识图谱(需要安装pyvis)
kg_index.visualize("knowledge_graph.html")

# 使用知识图谱进行查询
query_engine = kg_index.as_query_engine()
response = query_engine.query("AI和机器学习有什么关系?")
print(response)
```

这个例子中：
- 我们基于文档创建了一个知识图谱索引
- 生成了一个可视化的知识图谱
- 使用知识图谱进行问答

## 多模态索引

LlamaIndex也支持处理图像等多模态数据：

```python
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from PIL import Image
import requests
from io import BytesIO

# 加载图像
response = requests.get("https://example.com/image.jpg")
image = Image.open(BytesIO(response.content))

# 创建图像节点
image_node = ImageNode(image=image)

# 使用多模态模型处理图像
model = OpenAIMultiModal(model="gpt-4-vision-preview")

# 查询图像
response = model.complete(
    prompt="描述这张图片中的内容",
    image_documents=[image_node]
)
print(response)
```

在这个例子中：
- 我们加载了一张图像并创建了ImageNode
- 使用OpenAI的多模态模型处理图像
- 向模型提问关于图像内容的问题

## 实际应用：创建一个文档助手

下面是一个更完整的示例，展示如何创建一个文档问答助手：

```python
import os
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, 
    StorageContext, load_index_from_storage,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings.openai import OpenAIEmbedding

# 设置默认配置
os.environ["OPENAI_API_KEY"] = "你的API密钥"
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

def load_or_create_index(data_dir, storage_dir):
    """加载现有索引或创建新索引"""
    if os.path.exists(storage_dir):
        print("加载现有索引...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
    else:
        print("创建新索引...")
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(storage_dir)
    return index

def document_assistant(data_dir="./docs", storage_dir="./storage"):
    """文档问答助手"""
    index = load_or_create_index(data_dir, storage_dir)
    query_engine = index.as_query_engine()
    
    print("文档助手已准备就绪，请输入问题(输入'退出'结束)")
    while True:
        question = input("\n问题: ")
        if question.lower() == "退出":
            break
            
        response = query_engine.query(question)
        print(f"\n回答: {response}")

# 运行助手
if __name__ == "__main__":
    document_assistant()
```

这个文档助手：
- 会检查是否有现有索引，如果有则加载，没有则创建新索引
- 提供了一个交互式界面来提问和获取回答
- 在实际使用中可以大大节省索引时间

## 高级应用：可查询的知识库应用

下面是一个更高级的应用，它组合了多种LlamaIndex功能：

```python
import streamlit as st
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader,
    StorageContext, load_index_from_storage
)
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.llms.openai import OpenAI
import os

# 设置环境
os.environ["OPENAI_API_KEY"] = "你的API密钥"

def init_indexes():
    """初始化不同领域的索引"""
    indexes = {}
    
    # 加载技术文档索引
    if os.path.exists("./storage/tech"):
        storage_context = StorageContext.from_defaults(persist_dir="./storage/tech")
        indexes["tech"] = load_index_from_storage(storage_context)
    else:
        docs = SimpleDirectoryReader("./docs/tech").load_data()
        indexes["tech"] = VectorStoreIndex.from_documents(docs, llm=OpenAI(model="gpt-3.5-turbo"))
        indexes["tech"].storage_context.persist("./storage/tech")
    
    # 加载商业文档索引
    if os.path.exists("./storage/business"):
        storage_context = StorageContext.from_defaults(persist_dir="./storage/business")
        indexes["business"] = load_index_from_storage(storage_context)
    else:
        docs = SimpleDirectoryReader("./docs/business").load_data()
        indexes["business"] = VectorStoreIndex.from_documents(docs, llm=OpenAI(model="gpt-3.5-turbo"))
        indexes["business"].storage_context.persist("./storage/business")
        
    return indexes

def create_router_query_engine(indexes):
    """创建路由查询引擎"""
    query_engine_tools = []
    
    # 为每个领域创建查询工具
    for name, index in indexes.items():
        query_engine = index.as_query_engine()
        description = f"{name}领域的知识库"
        query_engine_tools.append(
            QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name=name,
                description=description
            )
        )
    
    # 创建路由查询引擎
    router_query_engine = RouterQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        llm=OpenAI(model="gpt-3.5-turbo"),
        select_multi=True  # 允许从多个工具获取答案
    )
    
    return router_query_engine

# Streamlit应用
def main():
    st.title("智能知识库助手")
    st.write("这个助手可以回答技术和商业领域的问题")
    
    # 初始化索引和查询引擎
    if "query_engine" not in st.session_state:
        with st.spinner("正在加载知识库..."):
            indexes = init_indexes()
            st.session_state.query_engine = create_router_query_engine(indexes)
    
    # 用户输入界面
    query = st.text_input("请输入您的问题:")
    
    if query:
        with st.spinner("思考中..."):
            response = st.session_state.query_engine.query(query)
            st.write("回答:")
            st.write(response.response)

if __name__ == "__main__":
    main()
```

这个高级应用：
- 使用Streamlit创建了一个Web界面
- 处理多个领域的知识库
- 使用路由查询引擎智能选择合适的知识源
- 持久化索引以提高性能

## 小结

LlamaIndex是一个强大的框架，它使开发者能够有效地将大型语言模型与自定义数据集成，创建智能应用。它的核心优势在于：

1. **简化数据接入**: 提供了简单的方式来处理各种数据源
2. **优化查询效率**: 使用向量索引、知识图谱等技术提高查询效率
3. **灵活的架构**: 支持自定义每个环节的处理流程
4. **多模态支持**: 能够处理文本、图像等多种类型的数据

通过本文介绍的基本概念和代码示例，你可以开始使用LlamaIndex构建自己的智能应用。随着经验的积累，你可以探索更多高级功能，如自定义检索策略、复杂的查询路由、多层知识结构等。

## 进阶学习资源

- [LlamaIndex官方文档](https://docs.llamaindex.ai/)
- [LlamaIndex GitHub仓库](https://github.com/jerryjliu/llama_index)
- [LlamaHub](https://llamahub.ai/) - 预构建的数据连接器和工具集合
- [LlamaIndex Discord社区](https://discord.gg/dGcwcsnxhU)