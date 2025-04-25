# TextSentimentAnalyzer: Advanced Sentiment Analysis System

## 项目概述

TextSentimentAnalyzer 是一个全面的文本情感分析系统，能够准确判断文本内容的情感倾向（积极、消极或中性）。本项目结合传统机器学习和深度学习方法，提供了从数据预处理到模型部署的完整工作流程。

## 主要功能

- 文本预处理与特征提取
- 多种情感分析模型实现（传统ML和深度学习）
- 模型训练、评估与比较
- 用户友好的Web界面
- 实时情感分析预测

## 技术栈

- **编程语言**: Python 3.8+
- **自然语言处理**: NLTK, spaCy
- **机器学习**: Scikit-learn
- **深度学习**: PyTorch/TensorFlow
- **Web框架**: Flask
- **数据分析**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn

## 项目结构

```
sentiment_analysis/
├── data/                    # 数据集文件夹
├── models/                  # 保存训练好的模型
├── notebooks/               # Jupyter笔记本
│   ├── data_analysis.ipynb  # 数据分析
│   └── model_eval.ipynb     # 模型评估
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # 文本预处理
│   ├── features.py          # 特征工程
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical.py     # 传统机器学习模型
│   │   └── deep_learning.py # 深度学习模型
│   ├── train.py             # 训练脚本
│   ├── evaluate.py          # 评估脚本
│   └── predict.py           # 预测脚本
├── app/
│   ├── __init__.py
│   ├── app.py               # Web应用
│   ├── static/              # 静态文件
│   └── templates/           # HTML模板
├── requirements.txt         # 项目依赖
└── README.md                # 项目说明
```

## 快速开始

### 环境设置

1. 克隆仓库:
```bash
git clone https://github.com/yourusername/TextSentimentAnalyzer.git
cd TextSentimentAnalyzer
```

2. 创建并激活虚拟环境:
```bash
python -m venv venv
source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate
```

3. 安装依赖:
```bash
pip install -r requirements.txt
```

### 数据准备

1. 下载数据集（例如：IMDB影评、Twitter情感数据集等）
2. 将数据放入`data/`文件夹

### 模型训练

```bash
python src/train.py --model classical  # 训练传统机器学习模型
python src/train.py --model deep       # 训练深度学习模型
```

### 启动Web应用

```bash
python app/app.py
```
然后访问 http://localhost:5000 查看Web界面。

## 模型性能

| 模型类型 | 准确率 | F1分数 | 训练时间 |
|---------|-------|-------|---------|
| 朴素贝叶斯 | ~82% | ~0.81 | 快 |
| SVM | ~85% | ~0.84 | 中等 |
| LSTM | ~88% | ~0.87 | 慢 |
| BERT | ~92% | ~0.91 | 非常慢 |

## 未来改进

- 实现多语言支持
- 增加细粒度情感分析（超出简单的积极/消极/中性分类）
- 优化模型推理速度
- 加入用户反馈机制改进模型

## 贡献指南

欢迎提交问题和拉取请求。对于重大更改，请先开issue讨论您想要更改的内容。

## 许可证

[MIT](https://choosealicense.com/licenses/mit/)