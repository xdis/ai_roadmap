# Docker容器化基础教程

Docker是一种容器化技术，可以将应用程序及其依赖项打包到一个标准化的单元中，从而简化了开发、测试和部署流程。本文将介绍Docker的基本概念和在DevOps与MLOps中的应用。

## 一、Docker核心概念

### 1. 容器 vs 虚拟机

**容器**：轻量级、共享主机操作系统内核、启动迅速（秒级）、资源占用少。
**虚拟机**：完整的操作系统副本、隔离性强但资源占用大、启动较慢（分钟级）。

![容器vs虚拟机](https://www.docker.com/sites/default/files/d8/2018-11/docker-containerized-and-vm-transparent-bg.png)

### 2. Docker三大核心组件

- **镜像(Image)**: 只读模板，包含创建容器的指令
- **容器(Container)**: 镜像的运行实例，可以启动、停止、删除
- **仓库(Repository)**: 存储和分发镜像的地方，如Docker Hub

## 二、Docker基础命令

### 1. 安装与验证

```bash
# 验证安装
docker --version
docker info

# 运行Hello World测试
docker run hello-world
```

### 2. 镜像管理

```bash
# 列出本地镜像
docker images

# 拉取镜像
docker pull python:3.9

# 构建镜像
docker build -t myapp:1.0 .

# 删除镜像
docker rmi python:3.8
```

### 3. 容器管理

```bash
# 运行容器
docker run -d -p 8080:80 --name mywebapp nginx

# 列出运行中的容器
docker ps

# 列出所有容器（包括已停止的）
docker ps -a

# 停止容器
docker stop mywebapp

# 启动已停止的容器
docker start mywebapp

# 删除容器
docker rm mywebapp
```

## 三、Dockerfile详解

Dockerfile是创建Docker镜像的脚本，包含一系列指令。

### 简单Dockerfile示例：

```dockerfile
# 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "app.py"]
```

### 常用Dockerfile指令：

- **FROM**: 指定基础镜像
- **WORKDIR**: 设置工作目录
- **COPY/ADD**: 复制文件到容器
- **RUN**: 执行命令并创建新层
- **EXPOSE**: 声明容器监听的端口
- **ENV**: 设置环境变量
- **CMD/ENTRYPOINT**: 指定容器启动时执行的命令

## 四、Docker Compose多容器应用

Docker Compose用于定义和运行多容器Docker应用程序。

### docker-compose.yml示例：

```yaml
version: '3'

services:
  webapp:
    build: ./webapp
    ports:
      - "5000:5000"
    depends_on:
      - db
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/mydb
  
  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=mydb

volumes:
  postgres_data:
```

### 基本命令：

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 停止所有服务
docker-compose down
```

## 五、MLOps中的Docker应用

### 1. 机器学习模型部署示例

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ model/
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. 完整的FastAPI模型服务示例

```python
# app.py
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# 定义请求数据模型
class PredictionRequest(BaseModel):
    features: list[float]

# 加载模型
model = joblib.load('model/model.pkl')

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ML Model API"}

@app.post("/predict")
def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}
```

### 3. Docker部署与扩展

```bash
# 构建并运行模型API容器
docker build -t ml-model-api .
docker run -d -p 8000:8000 --name ml-service ml-model-api

# 扩展到Kubernetes（仅命令示例）
kubectl create deployment ml-model --image=ml-model-api
kubectl scale deployment ml-model --replicas=3
kubectl expose deployment ml-model --port=8000 --type=LoadBalancer
```

## 六、实用Tips与最佳实践

1. **镜像层优化**：减少层数、优化层顺序、使用.dockerignore
2. **多阶段构建**：减小最终镜像体积
3. **非root用户**：容器内使用非特权用户提高安全性
4. **健康检查**：使用HEALTHCHECK指令
5. **体积优化**：使用alpine等轻量级基础镜像
6. **CI/CD集成**：自动构建、测试、部署Docker镜像

### 多阶段构建示例：

```dockerfile
# 构建阶段
FROM python:3.9 AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# 运行阶段
FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY --from=builder /build/requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels

COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 七、练习与实战

1. 构建一个简单的Python应用容器
2. 使用Docker Compose搭建带数据库的Web应用
3. 尝试将现有机器学习模型容器化并部署

## 资源链接

- [Docker官方文档](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Docker Compose文档](https://docs.docker.com/compose/)