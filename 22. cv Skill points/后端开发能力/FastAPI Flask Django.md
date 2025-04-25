# Python后端框架对比：FastAPI、Flask与Django

本文将简单介绍三个流行的Python后端框架，并通过简单的代码示例帮助你理解它们的基本用法和区别。

## 1. Flask：轻量级微框架

Flask是一个轻量级的微框架，拥有简洁的API和灵活的设计，适合小型项目或API服务。

### 特点
- 轻量级，核心功能简单
- 高度可定制性
- 丰富的第三方插件生态系统
- 不内置ORM、表单验证等功能
- 适合入门者和小型项目

### 基本示例

创建一个简单的Flask API服务：

```python
from flask import Flask, jsonify, request

# 创建Flask应用实例
app = Flask(__name__)

# 模拟数据库
tasks = [
    {"id": 1, "title": "学习Flask", "completed": False},
    {"id": 2, "title": "构建API", "completed": False}
]

# 定义路由：获取所有任务
@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify({"tasks": tasks})

# 定义路由：根据ID获取任务
@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = next((task for task in tasks if task["id"] == task_id), None)
    if task is None:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({"task": task})

# 定义路由：创建新任务
@app.route('/tasks', methods=['POST'])
def create_task():
    if not request.json or 'title' not in request.json:
        return jsonify({"error": "Title field is required"}), 400
    
    task = {
        "id": tasks[-1]["id"] + 1 if tasks else 1,
        "title": request.json['title'],
        "completed": False
    }
    tasks.append(task)
    return jsonify({"task": task}), 201

if __name__ == '__main__':
    app.run(debug=True)
```

### 运行说明
1. 安装Flask: `pip install flask`
2. 运行应用: `python app.py`
3. 访问API: http://127.0.0.1:5000/tasks

## 2. FastAPI：现代高性能框架

FastAPI是一个现代、快速（高性能）的Web框架，用于构建API，基于Python 3.6+的标准类型提示。

### 特点
- 基于现代Python特性（类型提示）
- 极高的性能，接近Node.js和Go
- 自动生成交互式API文档
- 内置请求验证
- 异步支持
- 简单易学

### 基本示例

创建一个简单的FastAPI服务：

```python
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# 创建FastAPI应用实例
app = FastAPI(title="任务管理API")

# 定义数据模型
class Task(BaseModel):
    id: Optional[int] = None
    title: str
    completed: bool = False

# 模拟数据库
tasks = [
    Task(id=1, title="学习FastAPI", completed=False),
    Task(id=2, title="构建API", completed=False)
]

# 定义路由：获取所有任务
@app.get("/tasks", response_model=List[Task])
def get_tasks():
    return tasks

# 定义路由：根据ID获取任务
@app.get("/tasks/{task_id}", response_model=Task)
def get_task(task_id: int):
    task = next((task for task in tasks if task.id == task_id), None)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

# 定义路由：创建新任务
@app.post("/tasks", response_model=Task, status_code=201)
def create_task(task: Task = Body(...)):
    task.id = tasks[-1].id + 1 if tasks else 1
    tasks.append(task)
    return task

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
```

### 运行说明
1. 安装FastAPI和ASGI服务器: `pip install fastapi uvicorn`
2. 运行应用: `python app.py` 或 `uvicorn app:app --reload`
3. 访问API: http://127.0.0.1:8000/tasks
4. 访问自动生成的文档: http://127.0.0.1:8000/docs

## 3. Django：全功能Web框架

Django是一个高级的Python Web框架，提供了完整的Web开发解决方案。

### 特点
- 全功能框架，包含ORM、表单处理、管理后台等
- "电池内置"的设计理念
- 安全性高，内置多种安全措施
- 丰富的生态系统和插件
- 适合大型复杂项目
- 学习曲线较陡

### 基本示例

一个Django REST框架示例（先需创建Django项目）：

```python
# models.py
from django.db import models

class Task(models.Model):
    title = models.CharField(max_length=200)
    completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title

# serializers.py
from rest_framework import serializers
from .models import Task

class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = Task
        fields = ['id', 'title', 'completed', 'created_at']

# views.py
from rest_framework import viewsets
from .models import Task
from .serializers import TaskSerializer

class TaskViewSet(viewsets.ModelViewSet):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer

# urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TaskViewSet

router = DefaultRouter()
router.register(r'tasks', TaskViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

### 运行说明
1. 安装Django和DRF: `pip install django djangorestframework`
2. 创建项目: `django-admin startproject myproject`
3. 创建应用: `python manage.py startapp tasks`
4. 添加上述代码到相应文件
5. 配置settings.py中的INSTALLED_APPS添加'rest_framework'和'tasks'
6. 应用迁移: `python manage.py makemigrations` 和 `python manage.py migrate`
7. 运行服务器: `python manage.py runserver`
8. 访问API: http://127.0.0.1:8000/tasks/

## 框架对比

| 特性 | Flask | FastAPI | Django |
|------|-------|---------|--------|
| 类型 | 微框架 | API框架 | 全功能框架 |
| 学习曲线 | 简单 | 中等 | 陡峭 |
| 性能 | 良好 | 极佳 | 良好 |
| 适用场景 | 小型项目、API | 现代API开发 | 大型复杂应用 |
| 自动文档 | 需插件 | 内置 | 需插件 |
| 数据库ORM | 需插件 | 需插件 | 内置 |
| 异步支持 | 有限 | 完全支持 | 部分支持 |
| 社区大小 | 大 | 快速增长 | 非常大 |

## 如何选择？

- **Flask**：如果你需要一个轻量级框架，可以完全控制组件选择，或者正在学习Python Web开发。
- **FastAPI**：如果你需要构建高性能的现代API，重视开发速度和自动文档。
- **Django**：如果你正在构建一个功能丰富的Web应用，需要内置的ORM和管理界面。

## 入门建议

初学者建议先从Flask或FastAPI开始，理解Web框架的基本概念后再学习Django。FastAPI尤其适合现代API开发，结合了简单性和高性能。