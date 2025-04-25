# CI/CD 管道设计基础

## 什么是 CI/CD？

**持续集成(CI, Continuous Integration)** 和 **持续交付/部署(CD, Continuous Delivery/Deployment)** 是现代软件开发的核心实践。

- **持续集成**：开发人员频繁地将代码合并到主分支，并自动运行测试，确保新代码不会破坏现有功能
- **持续交付**：自动将验证通过的代码发布到预生产环境，准备随时可以部署
- **持续部署**：自动将验证通过的代码直接部署到生产环境

## CI/CD 管道的基本组成

一个典型的 CI/CD 管道包括以下阶段：

1. **源代码管理** - 代码的提交和版本控制
2. **构建** - 编译代码、创建可执行文件或容器镜像
3. **测试** - 运行自动化测试（单元测试、集成测试等）
4. **部署** - 将应用发布到不同环境（开发、测试、生产）

## 简单的 CI/CD 配置示例

### GitHub Actions 配置示例 (Python项目)

```yaml
# .github/workflows/python-app.yml
name: Python 应用 CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: 设置 Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: 安装依赖
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest
    
    - name: 运行测试
      run: |
        pytest
        
    - name: 构建并推送 Docker 镜像
      if: github.event_name == 'push'
      run: |
        docker build -t myapp:latest .
        # 这里通常会推送到 Docker 仓库
```

### GitLab CI/CD 配置示例 (Java项目)

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

build-job:
  stage: build
  script:
    - echo "开始构建..."
    - ./gradlew build
  artifacts:
    paths:
      - build/libs/*.jar

test-job:
  stage: test
  script:
    - echo "开始测试..."
    - ./gradlew test

deploy-job:
  stage: deploy
  script:
    - echo "部署到测试环境..."
    - scp build/libs/*.jar user@test-server:/app/
  only:
    - main
```

## MLOps 中的 CI/CD 特点

机器学习项目的 CI/CD 管道有一些独特的考虑：

1. **数据版本控制** - 不仅跟踪代码变化，还需要跟踪数据变化
2. **模型训练** - 作为管道的一部分自动化训练模型
3. **模型评估** - 自动化验证模型性能
4. **模型注册** - 跟踪和管理模型版本
5. **模型部署** - 将模型部署为API或服务

### 简单的 MLOps 管道示例 (使用 GitHub Actions)

```yaml
# .github/workflows/mlops.yml
name: ML模型 CI/CD

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # 每周日重新训练

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: 设置 Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: 安装依赖
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: 获取并准备数据
      run: python scripts/prepare_data.py
      
    - name: 训练模型
      run: python scripts/train_model.py
      
    - name: 评估模型
      run: python scripts/evaluate_model.py
      
    - name: 注册模型
      if: success()
      run: python scripts/register_model.py
      
    - name: 部署模型
      if: success()
      run: |
        # 部署模型到API服务
        python scripts/deploy_model.py
```

## CI/CD 最佳实践

1. **自动化一切** - 减少手动步骤，提高一致性
2. **快速反馈** - 尽早发现并解决问题
3. **小批量提交** - 频繁提交小的代码变更
4. **环境一致性** - 确保开发、测试和生产环境尽可能相似
5. **监控和日志** - 实时了解系统状态和性能

## 简单的监控代码示例

```python
# monitoring.py
import time
import logging
from prometheus_client import start_http_server, Counter, Gauge

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义监控指标
http_requests_total = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint'])
model_prediction_duration = Gauge('model_prediction_duration', 'Model prediction duration in seconds')

def log_request(method, endpoint):
    """记录HTTP请求"""
    http_requests_total.labels(method=method, endpoint=endpoint).inc()
    logger.info(f"收到 {method} 请求: {endpoint}")

def time_prediction(func):
    """测量模型预测时间的装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        model_prediction_duration.set(duration)
        logger.info(f"模型预测耗时: {duration:.4f}秒")
        return result
    return wrapper

# 启动监控服务器
def start_monitoring(port=8000):
    start_http_server(port)
    logger.info(f"监控服务运行在端口 {port}")
```

## CI/CD 工具生态系统

- **源代码控制**: Git (GitHub, GitLab, Bitbucket)
- **CI/CD 平台**: Jenkins, GitHub Actions, GitLab CI/CD, CircleCI, Azure DevOps
- **容器化**: Docker, Kubernetes
- **配置管理**: Ansible, Puppet, Chef
- **监控**: Prometheus, Grafana, ELK Stack

通过采用 CI/CD 实践和工具，团队可以更快、更可靠地交付软件，同时减少人为错误和提高代码质量。