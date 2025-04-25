# RESTful API 设计指南

## 什么是 RESTful API？

REST（Representational State Transfer）是一种软件架构风格，用于设计网络应用程序。RESTful API 是遵循 REST 原则设计的应用程序接口，它使用 HTTP 请求来获取（GET）、创建（POST）、更新（PUT/PATCH）和删除（DELETE）数据。

## RESTful API 的基本原则

1. **资源导向**: 所有操作都是针对资源（资源通常是名词）
2. **HTTP 方法映射**: 使用 HTTP 方法表示操作类型
3. **无状态**: 每个请求包含所有必要的信息，服务器不保存客户端状态
4. **统一接口**: 使用统一的接口简化架构
5. **数据格式**: 通常使用 JSON 或 XML 格式传输数据

## HTTP 方法及其用途

| HTTP 方法 | 用途 | 示例 |
|----------|------|------|
| GET | 获取资源 | 获取用户列表或特定用户 |
| POST | 创建资源 | 添加新用户 |
| PUT | 完全更新资源 | 更新用户的所有信息 |
| PATCH | 部分更新资源 | 仅更新用户的部分信息 |
| DELETE | 删除资源 | 删除用户 |

## URL 设计规范

良好的 RESTful API URL 设计应该遵循以下规范：

1. **使用名词（复数形式）表示资源**:
   - `/users` - 用户资源集合
   - `/images` - 图像资源集合

2. **使用 ID 标识特定资源**:
   - `/users/123` - ID 为 123 的用户

3. **使用嵌套关系表示从属资源**:
   - `/users/123/orders` - 用户 123 的所有订单
   - `/users/123/orders/456` - 用户 123 的特定订单 456

4. **使用查询参数进行过滤、排序、分页**:
   - `/users?role=admin` - 获取所有管理员用户
   - `/images?size=large` - 获取所有大尺寸图片
   - `/users?page=2&limit=10` - 分页获取用户

## 状态码使用

| 状态码 | 含义 | 使用场景 |
|--------|------|---------|
| 200 OK | 请求成功 | GET、PUT、PATCH 成功 |
| 201 Created | 创建成功 | POST 请求创建资源成功 |
| 204 No Content | 成功但无返回内容 | DELETE 操作成功 |
| 400 Bad Request | 请求格式错误 | 参数验证失败 |
| 401 Unauthorized | 身份验证失败 | 未提供认证信息或认证失败 |
| 403 Forbidden | 权限不足 | 已认证但无权访问资源 |
| 404 Not Found | 资源不存在 | 请求的资源不存在 |
| 405 Method Not Allowed | 方法不允许 | 资源不支持请求的 HTTP 方法 |
| 500 Internal Server Error | 服务器内部错误 | 服务器发生未预期的错误 |

## 代码示例：使用 Python Flask 实现 RESTful API

### 基本框架设置

```python
from flask import Flask, request, jsonify
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

# 模拟数据库
users = [
    {"id": 1, "name": "张三", "email": "zhangsan@example.com"},
    {"id": 2, "name": "李四", "email": "lisi@example.com"}
]
```

### 实现用户资源接口

```python
class UserList(Resource):
    def get(self):
        """获取所有用户"""
        return jsonify(users)
    
    def post(self):
        """创建新用户"""
        new_user = request.get_json()
        new_user['id'] = len(users) + 1
        users.append(new_user)
        return new_user, 201

class User(Resource):
    def get(self, user_id):
        """获取特定用户"""
        user = next((u for u in users if u['id'] == user_id), None)
        if user:
            return jsonify(user)
        return {"error": "用户不存在"}, 404
    
    def put(self, user_id):
        """完全更新用户"""
        user = next((u for u in users if u['id'] == user_id), None)
        if not user:
            return {"error": "用户不存在"}, 404
            
        updated_user = request.get_json()
        updated_user['id'] = user_id
        
        # 更新用户
        for i, u in enumerate(users):
            if u['id'] == user_id:
                users[i] = updated_user
                break
                
        return updated_user
    
    def patch(self, user_id):
        """部分更新用户"""
        user = next((u for u in users if u['id'] == user_id), None)
        if not user:
            return {"error": "用户不存在"}, 404
            
        updates = request.get_json()
        
        # 只更新提供的字段
        for i, u in enumerate(users):
            if u['id'] == user_id:
                for key, value in updates.items():
                    if key != 'id':  # 不允许更新 ID
                        users[i][key] = value
                return users[i]
    
    def delete(self, user_id):
        """删除用户"""
        global users
        initial_count = len(users)
        users = [u for u in users if u['id'] != user_id]
        
        if len(users) < initial_count:
            return "", 204
        return {"error": "用户不存在"}, 404

# 注册路由
api.add_resource(UserList, '/users')
api.add_resource(User, '/users/<int:user_id>')

if __name__ == '__main__':
    app.run(debug=True)
```

## 调用示例

### 获取所有用户

```bash
curl -X GET http://localhost:5000/users
```

响应:
```json
[
  {"id": 1, "name": "张三", "email": "zhangsan@example.com"},
  {"id": 2, "name": "李四", "email": "lisi@example.com"}
]
```

### 获取特定用户

```bash
curl -X GET http://localhost:5000/users/1
```

响应:
```json
{"id": 1, "name": "张三", "email": "zhangsan@example.com"}
```

### 创建新用户

```bash
curl -X POST -H "Content-Type: application/json" -d '{"name": "王五", "email": "wangwu@example.com"}' http://localhost:5000/users
```

响应:
```json
{"id": 3, "name": "王五", "email": "wangwu@example.com"}
```

### 更新用户

```bash
curl -X PUT -H "Content-Type: application/json" -d '{"name": "张三(修改)", "email": "zhangsan_new@example.com"}' http://localhost:5000/users/1
```

响应:
```json
{"id": 1, "name": "张三(修改)", "email": "zhangsan_new@example.com"}
```

### 部分更新用户

```bash
curl -X PATCH -H "Content-Type: application/json" -d '{"email": "zhangsan_updated@example.com"}' http://localhost:5000/users/1
```

响应:
```json
{"id": 1, "name": "张三(修改)", "email": "zhangsan_updated@example.com"}
```

### 删除用户

```bash
curl -X DELETE http://localhost:5000/users/2
```

响应: (空，状态码 204)

## RESTful API 最佳实践

1. **版本控制**: 在 URL 或头信息中指明 API 版本
   ```
   /api/v1/users
   ```

2. **分页处理**: 对大量数据结果进行分页
   ```
   /api/users?page=2&limit=10
   ```

3. **错误处理**: 返回明确的错误信息
   ```json
   {
     "error": "无效的用户 ID",
     "code": "INVALID_USER_ID", 
     "status": 400
   }
   ```

4. **HATEOAS (超媒体即应用状态引擎)**: 在响应中包含相关资源的链接
   ```json
   {
     "id": 1,
     "name": "张三",
     "links": [
       {"rel": "self", "href": "/users/1"},
       {"rel": "orders", "href": "/users/1/orders"}
     ]
   }
   ```

5. **认证与授权**: 使用 OAuth 2.0、JWT 等进行身份验证和授权

## 总结

构建良好的 RESTful API 需要遵循以下关键点：

1. 以资源为中心设计 URL
2. 正确使用 HTTP 方法表达操作意图
3. 合理使用状态码传达结果状态
4. 使用 JSON 作为主要数据交换格式
5. 实现适当的错误处理和响应格式
6. 考虑安全性、缓存、性能等因素

通过遵循这些原则，可以设计出直观、一致、易于使用的 API，使前端和后端的交互更加高效。