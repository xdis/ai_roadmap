# spaCy 自然语言处理库

spaCy 是一个现代化的自然语言处理（NLP）Python库，专注于提供高效快速、生产级别的文本处理工具。它被设计为能够快速处理大量文本，并提供丰富的语言分析功能。

## 核心优势

- **高效性能**：使用Cython编写的核心组件，处理速度快
- **开箱即用**：提供预训练模型，无需复杂配置
- **生产可用**：设计适合实际应用场景，稳定可靠
- **多语言支持**：支持60多种语言的模型
- **易于集成**：与深度学习框架（如PyTorch、TensorFlow）无缝配合

## 安装方法

```python
# 基础安装
pip install spacy

# 下载中文模型(小型)
python -m spacy download zh_core_web_sm

# 下载英文模型(小型)
python -m spacy download en_core_web_sm
```

## 基础功能演示

### 1. 文本处理流程

```python
import spacy

# 加载中文模型
nlp = spacy.load("zh_core_web_sm")

# 处理文本
text = "spaCy是一个强大的自然语言处理库，适合生产环境使用。"
doc = nlp(text)

# 基本文本分析
for token in doc:
    print(f"单词: {token.text}, 词性: {token.pos_}, 词汇类别: {token.tag_}, 依存关系: {token.dep_}")
```

### 2. 命名实体识别(NER)

```python
import spacy
from spacy import displacy

# 加载英文模型(用于演示实体识别，中文模型同样适用)
nlp = spacy.load("en_core_web_sm")

# 示例文本
text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

# 打印识别的实体
for ent in doc.ents:
    print(f"实体: {ent.text}, 类型: {ent.label_}")

# 在终端输出:
# 实体: Apple, 类型: ORG
# 实体: U.K., 类型: GPE
# 实体: $1 billion, 类型: MONEY

# 可视化实体（在Jupyter中使用）
# displacy.render(doc, style="ent")
```

### 3. 词向量与相似度计算

```python
import spacy

# 需要加载含有词向量的模型，如en_core_web_md
# 注意：小型模型(sm)不包含词向量
nlp = spacy.load("en_core_web_md")  # 中型英文模型

# 比较两个词的相似度
word1 = nlp("cat")
word2 = nlp("dog")
word3 = nlp("banana")

print(f"猫和狗的相似度: {word1.similarity(word2)}")        # 相似度较高
print(f"猫和香蕉的相似度: {word1.similarity(word3)}")      # 相似度较低

# 比较两个文档的相似度
doc1 = nlp("I like fast cars")
doc2 = nlp("I like quick automobiles")
print(f"文档相似度: {doc1.similarity(doc2)}")  # 相似度较高
```

### 4. 自定义管道组件

```python
import spacy
from spacy.language import Language

# 创建自定义组件
@Language.component("custom_component")
def custom_component(doc):
    # 为每个token添加自定义属性
    for token in doc:
        # 判断是否为数字
        token._.is_digit = token.text.isdigit()
    return doc

# 使用自定义组件
nlp = spacy.load("en_core_web_sm")

# 注册自定义属性
from spacy.tokens import Token
Token.set_extension("is_digit", default=False)

# 添加到处理管道
nlp.add_pipe("custom_component", last=True)

# 测试自定义组件
doc = nlp("这个句子包含数字123和单词。")
for token in doc:
    print(f"{token.text}: {token._.is_digit}")
```

### 5. 语法依存分析

```python
import spacy

nlp = spacy.load("zh_core_web_sm")
doc = nlp("我昨天在北京买了一本有趣的书")

# 打印依存关系
for token in doc:
    print(f"{token.text} --> {token.head.text} ({token.dep_})")

# 可视化依存树（在Jupyter中使用）
# from spacy import displacy
# displacy.render(doc, style="dep")
```

## 实际应用案例

### 文本分类示例

```python
import spacy
from spacy.training import Example

# 创建一个空的中文模型
nlp = spacy.blank("zh")

# 添加文本分类器
textcat = nlp.add_pipe("textcat")
textcat.add_label("正面")
textcat.add_label("负面")

# 训练数据
train_data = [
    ("这部电影太棒了，我非常喜欢", {"cats": {"正面": 1.0, "负面": 0.0}}),
    ("服务态度很差，不会再来了", {"cats": {"正面": 0.0, "负面": 1.0}}),
    ("价格合理，质量不错", {"cats": {"正面": 1.0, "负面": 0.0}}),
    ("体验太糟糕了，浪费时间", {"cats": {"正面": 0.0, "负面": 1.0}})
]

# 转换为Example对象
examples = []
for text, annotations in train_data:
    examples.append(Example.from_dict(nlp.make_doc(text), annotations))

# 简单训练几轮(实际应用需要更多数据和轮次)
nlp.initialize()
for i in range(20):
    losses = {}
    nlp.update(examples, losses=losses)
    print(f"Loss: {losses}")

# 测试模型
test_text = "这个产品质量很好，我很满意"
doc = nlp(test_text)
print(f"文本: {test_text}")
print(f"预测: 正面概率: {doc.cats['正面']:.2f}, 负面概率: {doc.cats['负面']:.2f}")
```

## spaCy与其他NLP库的比较

- **NLTK**: 学术导向，更适合学习NLP概念
- **spaCy**: 生产导向，速度快，API简洁
- **Stanford CoreNLP**: 功能全面但Java实现，集成较复杂
- **Gensim**: 专注于主题建模和向量空间建模
- **Hugging Face Transformers**: 专注于最新的深度学习模型

## 注意事项

1. 不同大小的模型在功能上有差异:
   - sm (小): 基本功能，无词向量
   - md (中): 包含词向量
   - lg (大): 更大的词向量，更高精度

2. 内存考量:
   - 加载模型会占用内存(几百MB到几GB不等)
   - 处理大量文本要考虑批量处理

3. 语言支持:
   - 不同语言模型的能力不同
   - 英文模型通常功能最全面

## 学习资源

- [spaCy官方文档](https://spacy.io/usage)
- [spaCy课程](https://course.spacy.io/)
- [spaCy Universe](https://spacy.io/universe) (项目实例)

spaCy是NLP工作的得力工具，适合从信息提取、文本分类到复杂的自然语言理解等多种应用场景。