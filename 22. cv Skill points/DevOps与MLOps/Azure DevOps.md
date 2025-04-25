# Azure DevOps 基础指南

Azure DevOps 是微软提供的一套开发工具和服务，它帮助团队进行规划、开发、测试和部署应用程序。本指南将简单介绍 Azure DevOps 的核心概念和基本使用方法。

## Azure DevOps 的主要组件

1. **Azure Boards** - 敏捷项目管理工具
2. **Azure Repos** - 源代码管理（Git/TFVC）
3. **Azure Pipelines** - CI/CD 持续集成/持续部署
4. **Azure Test Plans** - 测试管理工具
5. **Azure Artifacts** - 包管理服务

## Azure Pipelines 示例 - CI/CD 流程

Azure Pipelines 是 Azure DevOps 中最常用的功能之一，它让你可以自动构建、测试和部署代码。下面是一个简单的示例：

### 基本 YAML 管道配置

```yaml
# azure-pipelines.yml 文件示例
trigger:
- main  # 当 main 分支有推送时触发管道

pool:
  vmImage: 'ubuntu-latest'  # 使用最新的 Ubuntu 虚拟机

steps:
- script: echo Hello, world!
  displayName: '打印问候消息'
  
- script: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
  displayName: '安装依赖'

- script: |
    pytest tests/ --doctest-modules --junitxml=junit/test-results.xml
  displayName: '运行测试'

- task: PublishTestResults@2
  inputs:
    testResultsFormat: 'JUnit'
    testResultsFiles: '**/test-results.xml'
  condition: succeededOrFailed()
  displayName: '发布测试结果'
```

这个简单的管道:
1. 在 main 分支有更新时被触发
2. 在 Ubuntu 环境中运行
3. 打印问候消息
4. 安装 Python 依赖
5. 运行测试
6. 发布测试结果

### 使用变量和参数

```yaml
# 使用变量让管道更灵活
variables:
  pythonVersion: '3.9'
  projectName: 'my-ml-project'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '$(pythonVersion)'
    addToPath: true
  displayName: '配置 Python $(pythonVersion)'

- script: |
    echo "正在构建项目: $(projectName)"
    pip install -r requirements.txt
  displayName: '安装依赖'
```

## 在 Azure DevOps 中集成 MLOps

MLOps (Machine Learning Operations) 是 DevOps 在机器学习领域的延伸。以下是一个简单的 ML 模型训练和部署管道：

```yaml
# 机器学习模型训练和部署的 YAML 管道
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

stages:
- stage: Train
  displayName: '训练模型'
  jobs:
  - job: TrainJob
    steps:
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install azureml-sdk
      displayName: '安装依赖'
    
    - script: |
        python src/train.py --data-folder data/ --model-output models/
      displayName: '训练模型'
    
    - task: PublishPipelineArtifact@1
      inputs:
        targetPath: 'models/'
        artifact: 'trained-model'
        publishLocation: 'pipeline'
      displayName: '发布模型'

- stage: Deploy
  displayName: '部署模型'
  dependsOn: Train
  jobs:
  - job: DeployJob
    steps:
    - task: DownloadPipelineArtifact@2
      inputs:
        artifactName: 'trained-model'
        targetPath: '$(Pipeline.Workspace)/models'
      displayName: '下载模型'
    
    - script: |
        python src/deploy.py --model-path $(Pipeline.Workspace)/models/model.pkl
      displayName: '部署模型到 Azure ML'
```

## 使用 Python SDK 与 Azure DevOps 交互

你可以使用 Python 来以编程方式与 Azure DevOps 交互：

```python
# 示例：使用 Python SDK 连接到 Azure DevOps
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
import pprint

# 设置连接信息
personal_access_token = '你的个人访问令牌'  # 从 Azure DevOps 设置中获取
organization_url = 'https://dev.azure.com/你的组织名'

# 创建连接
credentials = BasicAuthentication('', personal_access_token)
connection = Connection(base_url=organization_url, creds=credentials)

# 获取客户端
build_client = connection.clients.get_build_client()
work_client = connection.clients.get_work_item_tracking_client()

# 获取项目
projects = work_client.get_projects()
print("组织中的项目:")
for project in projects:
    print(f" - {project.name}")

# 获取构建定义
build_definitions = build_client.get_definitions(project='你的项目名')
print("\n构建定义:")
for definition in build_definitions:
    print(f" - {definition.name} (ID: {definition.id})")

# 创建新工作项
from azure.devops.v6_0.work_item_tracking.models import JsonPatchOperation

new_task = [
    JsonPatchOperation(
        op="add",
        path="/fields/System.Title",
        value="新的任务标题"
    ),
    JsonPatchOperation(
        op="add",
        path="/fields/System.Description",
        value="任务描述内容"
    ),
    JsonPatchOperation(
        op="add",
        path="/fields/System.AssignedTo",
        value="username@example.com"
    )
]

# 创建一个类型为 Task 的工作项
work_item = work_client.create_work_item(
    document=new_task,
    project='你的项目名',
    type='Task'
)

print(f"\n已创建新任务: {work_item.id} - {work_item.fields['System.Title']}")
```

## 实用技巧

1. **服务连接**: 管道需要访问外部服务时，使用服务连接安全地存储凭据
   
2. **环境变量**: 使用环境变量或变量组存储敏感信息，而不是硬编码

3. **自托管代理**: 如果需要特殊环境配置，可以设置自己的构建代理

4. **审批门控**: 在关键部署前添加手动批准步骤

5. **模板复用**: 将常用管道步骤提取为模板以便复用

## 小结

Azure DevOps 提供了一个完整的工具集，帮助团队实现 DevOps 和 MLOps 实践。通过上述示例，你可以开始使用 Azure DevOps 构建自动化工作流，加速软件开发和机器学习模型部署过程。

从简单的 CI/CD 管道开始，随着经验积累，可以逐步添加更复杂的功能，如环境部署策略、测试自动化和安全扫描等。