# GitHub Actions 入门指南

GitHub Actions 是 GitHub 提供的一项持续集成/持续部署(CI/CD)服务，它允许你自动化软件开发工作流程。GitHub Actions 可以帮助你自动执行测试、构建、部署等任务，而无需配置外部 CI/CD 系统。

## 核心概念

### 1. 工作流（Workflow）

工作流是一个自动化过程，由一个或多个作业组成，可以由各种事件触发。工作流以 YAML 文件的形式存储在仓库的 `.github/workflows` 目录中。

### 2. 事件（Event）

事件是触发工作流的特定活动，例如：
- 推送代码到仓库
- 创建拉取请求
- 定时执行
- 手动触发

### 3. 作业（Job）

作业是工作流中的一组步骤，它们在同一运行器上执行。一个工作流可以包含多个作业，这些作业可以并行或按顺序运行。

### 4. 步骤（Step）

步骤是可以执行命令或使用操作的单个任务。一个作业由多个步骤组成，这些步骤按顺序执行。

### 5. 操作（Action）

操作是一个可重用的代码单元，可以在不同的工作流中使用。GitHub 提供了许多预建的操作，你也可以创建自己的操作。

## 实际例子：Python 项目的 CI 工作流

以下是一个简单的 GitHub Actions 工作流配置，用于 Python 项目的持续集成：

```yaml
# 文件位置：.github/workflows/python-ci.yml
name: Python CI

# 触发条件：当推送到 main 分支或创建 pull request 时
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    # 指定运行环境
    runs-on: ubuntu-latest
    
    # 测试策略：在不同的 Python 版本上运行测试
    strategy:
      matrix:
        python-version: [3.7, 3.8, 3.9]
    
    steps:
    # 检出代码
    - uses: actions/checkout@v2
    
    # 设置 Python 环境
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    # 安装依赖
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest pytest-cov
    
    # 运行测试
    - name: Test with pytest
      run: |
        pytest --cov=./ --cov-report=xml
    
    # 上传测试覆盖率报告
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

## 代码解析

1. **name**: 工作流的名称，会显示在 GitHub 界面上。

2. **on**: 指定触发工作流的事件。在这个例子中，当推送到 main 分支或创建针对 main 分支的拉取请求时触发。

3. **jobs**: 定义要执行的作业。这里只有一个名为 `test` 的作业。

4. **runs-on**: 指定作业运行的环境，这里使用 Ubuntu 最新版本。

5. **strategy.matrix**: 使用矩阵策略，在多个 Python 版本上运行测试。

6. **steps**: 作业中的步骤，按顺序执行：
   - 检出代码
   - 设置 Python 环境
   - 安装依赖
   - 运行测试
   - 上传测试覆盖率报告

## 机器学习项目示例

下面是一个针对机器学习项目的 GitHub Actions 工作流示例：

```yaml
# 文件位置：.github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  # 每天凌晨 3 点运行
  schedule:
    - cron: '0 3 * * *'

jobs:
  train-and-evaluate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    # 运行数据预处理
    - name: Preprocess data
      run: python src/preprocess.py
    
    # 训练模型
    - name: Train model
      run: python src/train.py
    
    # 评估模型
    - name: Evaluate model
      run: python src/evaluate.py
    
    # 将模型指标保存为 artifact
    - name: Save metrics
      uses: actions/upload-artifact@v2
      with:
        name: model-metrics
        path: metrics/
    
    # 将模型文件保存为 artifact
    - name: Save model
      uses: actions/upload-artifact@v2
      with:
        name: trained-model
        path: models/model.pkl
```

## 模型部署工作流

以下是一个将机器学习模型部署到生产环境的示例：

```yaml
# 文件位置：.github/workflows/model-deploy.yml
name: Deploy Model

on:
  # 只有手动触发
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ github.event.inputs.environment }}
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install awscli
    
    # 下载之前保存的模型
    - name: Download model
      uses: actions/download-artifact@v2
      with:
        name: trained-model
        path: models/
    
    # 部署到 AWS SageMaker (示例)
    - name: Deploy to SageMaker
      run: python scripts/deploy_to_sagemaker.py
      env:
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        AWS_REGION: us-west-2
        MODEL_NAME: "my-ml-model"
        ENVIRONMENT: ${{ github.event.inputs.environment }}
```

## GitHub Actions 的优势

1. **集成在 GitHub 中**：无需配置外部 CI/CD 服务。

2. **广泛的生态系统**：GitHub Marketplace 提供了数千个预建的操作。

3. **灵活性**：支持各种操作系统、编程语言和工具。

4. **自动化**：可以自动化从代码检查到部署的整个工作流程。

5. **可扩展性**：可以使用自定义操作扩展功能。

## 实际应用场景

### 1. 代码质量检查

```yaml
# 文件位置：.github/workflows/code-quality.yml
name: Code Quality

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lint:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pylint black
    
    - name: Lint with flake8
      run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Check style with black
      run: black --check .
```

### 2. 模型性能监控

```yaml
# 文件位置：.github/workflows/model-monitoring.yml
name: Model Performance Monitoring

on:
  schedule:
    - cron: '0 0 * * *'  # 每天运行一次

jobs:
  monitor:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Fetch latest production data
      run: python scripts/fetch_data.py
    
    - name: Evaluate model on new data
      run: python scripts/evaluate_production.py
    
    - name: Send alert if performance degrades
      if: ${{ failure() }}
      uses: dawidd6/action-send-mail@v3
      with:
        server_address: smtp.gmail.com
        server_port: 465
        username: ${{ secrets.EMAIL_USERNAME }}
        password: ${{ secrets.EMAIL_PASSWORD }}
        subject: Model Performance Alert
        body: Model performance has degraded below threshold!
        to: team@example.com
        from: GitHub Actions
```

## 小技巧和最佳实践

1. **使用缓存**：缓存依赖项可以加速工作流程。

```yaml
- name: Cache pip dependencies
  uses: actions/cache@v2
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

2. **使用环境变量和密钥**：敏感信息应存储为仓库密钥。

```yaml
- name: Deploy with sensitive info
  env:
    API_KEY: ${{ secrets.API_KEY }}
  run: python deploy.py
```

3. **使用矩阵构建**：在多个环境中测试。

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: [3.7, 3.8, 3.9]
```

4. **工作流可视化**：GitHub 提供了工作流执行的可视化界面，帮助你理解和调试工作流。

## 总结

GitHub Actions 是一个强大的 CI/CD 工具，对于 DevOps 和 MLOps 实践非常有用。它可以帮助你自动化软件开发生命周期中的各种任务，从代码检查、测试到部署，让你能够专注于开发而不是重复性的手动操作。

通过上面的示例，你应该已经了解了如何为不同的场景配置 GitHub Actions 工作流。记住，工作流配置文件应放在仓库的 `.github/workflows` 目录中，并以 `.yml` 或 `.yaml` 扩展名保存。