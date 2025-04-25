# Kubernetes编排基础

Kubernetes (K8s) 是一个开源的容器编排平台，用于自动化部署、扩展和管理容器化应用程序。在DevOps和MLOps中，Kubernetes是实现可靠、可扩展的应用和机器学习模型部署的关键工具。

## 1. Kubernetes核心概念

### 1.1 基本架构

Kubernetes集群由以下组件组成：
- **Master节点**：控制平面，管理集群
  - API Server：所有操作的入口
  - Scheduler：调度Pod到合适的节点
  - Controller Manager：维护集群状态
  - etcd：存储集群数据
- **Worker节点**：运行应用的节点
  - Kubelet：管理节点上的Pod和容器
  - Kube-proxy：网络代理
  - Container Runtime：运行容器(如Docker)

### 1.2 核心资源对象

- **Pod**：最小部署单元，包含一个或多个容器
- **Deployment**：管理Pod的副本集
- **Service**：为Pod提供网络访问
- **ConfigMap/Secret**：配置管理
- **PersistentVolume**：存储管理

## 2. 基本操作示例

### 2.1 创建简单的应用部署

以下是一个部署简单Web应用的YAML文件：

```yaml
# simple-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  labels:
    app: web
spec:
  replicas: 3  # 运行3个副本
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: nginx
        image: nginx:latest
        ports:
        - containerPort: 80
```

使用kubectl应用此配置：

```bash
kubectl apply -f simple-deployment.yaml
```

### 2.2 创建Service暴露应用

```yaml
# web-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: web-service
spec:
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer  # 对外暴露服务
```

应用Service配置：

```bash
kubectl apply -f web-service.yaml
```

## 3. MLOps中的Kubernetes应用

### 3.1 部署机器学习模型服务

这是一个部署TensorFlow Serving模型的例子：

```yaml
# ml-model-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-model-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: tf-serving
  template:
    metadata:
      labels:
        app: tf-serving
    spec:
      containers:
      - name: tf-serving
        image: tensorflow/serving
        args:
        - "--model_name=my_model"
        - "--model_base_path=/models/my_model"
        ports:
        - containerPort: 8501
        volumeMounts:
        - name: model-volume
          mountPath: /models
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
```

对应的Service：

```yaml
# ml-model-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: tf-model-service
spec:
  selector:
    app: tf-serving
  ports:
  - port: 8501
    targetPort: 8501
  type: ClusterIP
```

### 3.2 持久卷配置(用于存储模型)

```yaml
# model-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

## 4. Kubernetes资源管理

### 4.1 资源限制

```yaml
# resource-limits.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: resource-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: resource-demo
  template:
    metadata:
      labels:
        app: resource-demo
    spec:
      containers:
      - name: resource-demo
        image: nginx
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"  # 0.25 CPU
          limits:
            memory: "128Mi"
            cpu: "500m"  # 0.5 CPU
```

### 4.2 水平自动扩展

```yaml
# horizontal-autoscaler.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tf-model-server
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 5. 常用kubectl命令

```bash
# 查看所有Pod
kubectl get pods

# 查看Pod详情
kubectl describe pod <pod-name>

# 查看日志
kubectl logs <pod-name>

# 进入容器
kubectl exec -it <pod-name> -- /bin/bash

# 查看所有部署
kubectl get deployments

# 扩展副本数
kubectl scale deployment <deployment-name> --replicas=5
```

## 6. 使用Helm简化部署

Helm是Kubernetes的包管理工具，可以简化应用部署。

### 6.1 安装Helm Chart示例

```bash
# 添加存储库
helm repo add stable https://charts.helm.sh/stable

# 安装应用
helm install my-release stable/mysql
```

### 6.2 创建自己的Helm Chart

```bash
# 创建Chart
helm create my-app

# 目录结构
my-app/
  Chart.yaml          # 包含Chart元数据
  values.yaml         # 默认配置值
  templates/          # 模板文件
    deployment.yaml
    service.yaml
```

## 7. 在MLOps流程中集成Kubernetes

MLOps典型工作流程：

1. **模型训练**：使用Kubernetes Job或CronJob调度训练任务
2. **模型评估**：验证模型性能
3. **模型部署**：将模型部署到生产环境
4. **监控和更新**：监控模型性能并更新

### 7.1 训练任务示例

```yaml
# training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training
spec:
  template:
    spec:
      containers:
      - name: training
        image: my-training-image:latest
        command: ["python", "train.py", "--epochs=100"]
        resources:
          limits:
            nvidia.com/gpu: 2  # 使用2个GPU
      restartPolicy: Never
  backoffLimit: 2  # 失败后重试次数
```

## 8. Kubernetes监控

常用监控工具：
- Prometheus：指标收集
- Grafana：可视化
- Kubernetes Dashboard：集群管理UI

## 总结

Kubernetes提供了强大的容器编排能力，特别适合MLOps场景下的模型训练、部署和管理。主要优势包括：

- **可扩展性**：根据负载自动扩展
- **自动恢复**：自动重启失败的容器
- **滚动更新**：无停机更新应用
- **资源隔离**：有效管理计算资源
- **声明式配置**：基于YAML的配置管理

通过掌握这些基础知识，你可以开始在DevOps和MLOps工作流程中应用Kubernetes，实现机器学习系统的高效部署和管理。