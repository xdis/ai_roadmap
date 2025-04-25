# AWS服务基础指南：EC2、S3和SageMaker

AWS(Amazon Web Services)是目前市场上最流行的云计算平台之一，为开发者和企业提供各种云服务。本文将重点介绍三个核心AWS服务：EC2、S3和SageMaker，并附带简单的代码示例。

## EC2 (Elastic Compute Cloud)

EC2是AWS的虚拟服务器服务，允许用户在云中租用计算资源。

### 核心概念

- **实例(Instance)**: 一个虚拟服务器
- **AMI (Amazon Machine Image)**: 用于创建实例的预配置模板
- **安全组(Security Group)**: 控制进出实例的流量的虚拟防火墙
- **密钥对(Key Pair)**: 安全登录实例的认证方式

### 代码示例: 使用AWS SDK创建EC2实例

以下是使用Python的boto3库创建EC2实例的简单示例：

```python
import boto3

# 创建EC2客户端
ec2 = boto3.resource('ec2')

# 创建新的EC2实例
instances = ec2.create_instances(
    ImageId='ami-0c55b159cbfafe1f0',  # Amazon Linux 2 AMI ID (示例ID)
    MinCount=1,
    MaxCount=1,
    InstanceType='t2.micro',
    KeyName='my-key-pair',  # 你的密钥对名称
    SecurityGroupIds=[
        'sg-12345678',  # 你的安全组ID
    ]
)

# 打印实例ID
print(f"创建的实例ID: {instances[0].id}")
```

### 使用EC2部署计算机视觉模型

假设你有一个计算机视觉模型需要部署在服务器上：

1. 选择合适的实例类型（如GPU实例p2.xlarge用于深度学习）
2. 使用预置ML框架的AMI（如AWS Deep Learning AMI）
3. 安装必要的依赖
4. 上传你的模型并启动服务

```bash
# SSH登录到EC2实例
ssh -i "my-key-pair.pem" ec2-user@ec2-xx-xxx-xx-xxx.compute-1.amazonaws.com

# 安装依赖
pip install tensorflow opencv-python flask

# 启动模型服务
python model_server.py
```

## S3 (Simple Storage Service)

S3是AWS的对象存储服务，可用于存储和检索任意数量的数据。

### 核心概念

- **存储桶(Bucket)**: 存储对象的容器，具有全局唯一的名称
- **对象(Object)**: 存储在桶中的文件和元数据
- **前缀(Prefix)**: S3中的文件夹结构
- **访问控制**: 可通过策略和ACL控制访问权限

### 代码示例: 上传和下载文件

使用Python的boto3库进行文件操作：

```python
import boto3

# 创建S3客户端
s3 = boto3.client('s3')

# 上传文件
def upload_file(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    
    try:
        s3.upload_file(file_name, bucket, object_name)
        print(f"文件 {file_name} 上传成功!")
        return True
    except Exception as e:
        print(f"上传失败: {e}")
        return False

# 下载文件
def download_file(bucket, object_name, file_name):
    try:
        s3.download_file(bucket, object_name, file_name)
        print(f"文件 {object_name} 下载成功!")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

# 使用示例
upload_file('local_image.jpg', 'my-cv-models-bucket', 'images/test_image.jpg')
download_file('my-cv-models-bucket', 'models/resnet50.h5', 'local_model.h5')
```

### 用于计算机视觉的S3应用场景

- 存储训练和测试图像数据集
- 保存训练好的模型文件
- 存储推理结果和分析报告
- 作为静态网站托管可视化结果

## SageMaker

SageMaker是AWS的全托管机器学习服务，简化了AI开发、训练和部署流程。

### 核心概念

- **笔记本实例**: 用于开发和实验的Jupyter环境
- **训练作业**: 在托管环境中训练模型
- **模型**: 训练完成的模型成品
- **端点**: 部署模型用于推理的HTTPS端点

### 代码示例: 训练和部署模型

使用SageMaker训练和部署一个简单的图像分类模型：

```python
import sagemaker
from sagemaker.tensorflow import TensorFlow

# 初始化SageMaker会话
session = sagemaker.Session()
role = sagemaker.get_execution_role()

# 定义数据位置
training_data_uri = 's3://my-cv-bucket/training-data'
validation_data_uri = 's3://my-cv-bucket/validation-data'

# 创建TensorFlow估算器
tf_estimator = TensorFlow(
    entry_point='train.py',  # 训练脚本
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # GPU实例
    framework_version='2.4.1',
    py_version='py37',
    hyperparameters={
        'epochs': 10,
        'batch-size': 32,
        'learning-rate': 0.001
    }
)

# 开始训练
tf_estimator.fit({
    'train': training_data_uri,
    'validation': validation_data_uri
})

# 部署模型到端点
predictor = tf_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.xlarge'
)

# 使用模型进行预测
response = predictor.predict({
    'instances': [image_data]  # 预处理后的图像数据
})
```

### SageMaker与计算机视觉

SageMaker特别适合计算机视觉任务，因为它：

1. 提供GPU加速训练
2. 支持常见的深度学习框架(TensorFlow, PyTorch)
3. 内置图像分类、目标检测等算法
4. 提供SageMaker Ground Truth服务进行数据标注

## AWS服务连接使用的完整案例

以下是一个结合EC2、S3和SageMaker的典型计算机视觉工作流：

1. **数据准备**：将图像数据集上传到S3
2. **模型开发**：在SageMaker笔记本中进行探索性开发
3. **模型训练**：使用SageMaker训练任务训练模型
4. **模型部署**：选择SageMaker端点(实时推理)或批量转换(批处理)
5. **应用服务**：在EC2上运行web服务，调用SageMaker端点

### 示例：车牌识别系统

```python
# 完整流程代码示例
import boto3
import sagemaker
import json
import numpy as np
from PIL import Image
import io

# 1. 数据准备 - 上传图像到S3
s3 = boto3.client('s3')
s3.upload_file('car_images_dataset.zip', 'license-plate-project', 'datasets/car_images.zip')

# 2. 使用SageMaker训练模型
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# 创建训练作业
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='plate_detection.py',
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    framework_version='1.8.0',
    py_version='py3',
    hyperparameters={
        'epochs': 20,
        'batch-size': 64,
    }
)

estimator.fit({'training': 's3://license-plate-project/datasets/car_images.zip'})

# 3. 部署模型
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.large'
)

# 4. 使用模型进行推理
def process_image(image_path):
    # 预处理图像
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return image_array.tolist()

# 发送推理请求
response = predictor.predict({
    'image': process_image('test_car.jpg')
})

plate_number = json.loads(response)['prediction']
print(f"检测到的车牌号: {plate_number}")

# 5. 将结果存储回S3
with open('results.json', 'w') as f:
    json.dump({'plate_number': plate_number}, f)

s3.upload_file('results.json', 'license-plate-project', 'results/test_car_result.json')
```

## 成本考虑与最佳实践

- **按需实例 vs. 预留实例 vs. Spot实例**: 根据工作负载选择合适的EC2定价模型
- **存储分层**: 使用S3的不同存储类别(Standard, Infrequent Access, Glacier)优化成本
- **自动扩展**: 设置EC2和SageMaker自动扩展以适应负载变化
- **关闭闲置资源**: 不使用时停止EC2实例和SageMaker笔记本
- **使用免费套餐**: 充分利用AWS免费套餐(Free Tier)进行学习和小型项目

## 总结

AWS的EC2、S3和SageMaker服务为计算机视觉应用提供了强大的云基础设施：

- **EC2**: 提供计算能力，适合运行Web服务和定制应用
- **S3**: 提供可靠的存储，适合存储数据集、模型和结果
- **SageMaker**: 提供端到端的机器学习平台，简化模型开发和部署

这些服务可以单独使用，也可以组合使用以构建完整的计算机视觉应用流程。开始时可以利用AWS免费套餐进行学习和实验，熟悉后再根据项目需求扩展使用。