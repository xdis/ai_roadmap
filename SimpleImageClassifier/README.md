# 简单图像分类器

一个使用卷积神经网络（CNN）识别不同类型花卉的基础图像分类项目。

## 项目概述

本项目展示了如何使用TensorFlow/Keras构建一个简单的图像分类器。该分类器经过训练，能够从图像中识别不同种类的花卉。

## 数据集

本项目使用包含5种不同花卉类别的Flowers数据集：
- 雏菊 (Daisy)
- 蒲公英 (Dandelion)
- 玫瑰 (Rose)
- 向日葵 (Sunflower)
- 郁金香 (Tulip)

**注意：** 数据集会在运行训练脚本时由TensorFlow自动下载。您无需手动下载或准备任何数据文件。

## 项目结构

```
SimpleImageClassifier/
├── data/                    # 数据集文件夹（保持为空，数据由TensorFlow缓存）
│   ├── train/               # 训练数据占位符
│   └── test/                # 测试数据占位符
├── models/                  # 保存的模型
├── notebooks/               # Jupyter笔记本
│   └── exploration.ipynb    # 数据探索分析
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载和预处理
│   ├── model.py             # 模型定义
│   ├── train.py             # 训练脚本
│   └── predict.py           # 预测脚本
├── requirements.txt         # 项目依赖
├── environment.yml          # Conda环境配置
└── README.md                # 项目描述
```

## 安装和设置

### 方法1：使用pip

1. 克隆此仓库
2. 安装所需依赖：
   ```
   pip install -r requirements.txt
   ```

### 方法2：使用Conda环境（推荐）

1. 克隆此仓库
2. 创建并激活conda环境：
   ```
   conda env create -f environment.yml
   conda activate image_classifier_env
   ```

## 模型训练和使用

### 首次设置

首次运行此项目时，您需要训练模型。训练脚本将：
1. 自动下载花卉数据集
2. 处理图像
3. 训练模型
4. 将训练好的模型保存到`models`目录

```bash
python src/train.py
```

训练选项：
```bash
# 使用基础CNN模型训练（更快但精度较低）
python src/train.py --model_type basic

# 使用迁移学习模型训练（默认，精度更高）
python src/train.py --model_type transfer

# 设置自定义训练轮数
python src/train.py --epochs 10

# 为迁移学习模型启用微调
python src/train.py --fine_tune --fine_tune_epochs 5
```

### 使用预训练模型

**一旦您已经训练了模型，就不需要再次训练**，除非您想尝试不同的模型配置或提高性能。

训练好的模型会被保存在`models`目录中，文件名为`flower_classifier_model.h5`，同时会生成一个包含类别标签的`class_names.txt`文件。

要使用您的预训练模型进行预测：

```bash
python src/predict.py --image 您的花卉图片路径.jpg
```

额外的预测选项：
```bash
# 显示更多预测类别
python src/predict.py --image 您的图片路径.jpg --top_k 5

# 使用特定的模型文件
python src/predict.py --image 您的图片路径.jpg --model models/您的自定义模型.h5
```

## 故障排除

### 常见问题

1. **模块未找到错误**：确保使用`pip install -r requirements.txt`安装所有依赖

2. **Python环境问题**：如果您有多个Python环境，确保使用正确的环境：
   ```bash
   # 使用您环境中的特定Python解释器
   /您的Python解释器路径 src/train.py
   ```

3. **空数据目录**：`data/train`和`data/test`目录是有意保持为空的。数据集会由TensorFlow自动下载并缓存在TensorFlow的数据集目录中。

## 系统要求

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow
- TensorFlow Datasets

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件。