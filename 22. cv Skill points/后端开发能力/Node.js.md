# Node.js 基础指南

Node.js 是一个基于 Chrome V8 引擎的 JavaScript 运行环境，允许在服务器端运行 JavaScript 代码。它是非阻塞、事件驱动的，特别适合构建高并发的网络应用程序。

## 1. Node.js 核心特点

- **非阻塞 I/O 模型**：能够处理大量并发连接
- **事件驱动**：基于事件循环机制
- **单线程**：使用单线程处理请求，但通过异步操作提高效率
- **跨平台**：可在 Windows、macOS、Linux 等多种平台运行

## 2. 安装 Node.js

访问 [Node.js 官网](https://nodejs.org/)下载安装包，或使用包管理器安装：

```bash
# Windows (使用 Chocolatey)
choco install nodejs

# macOS (使用 Homebrew)
brew install node

# Linux (使用 apt)
sudo apt update
sudo apt install nodejs npm
```

安装完成后，验证安装：

```bash
node -v
npm -v
```

## 3. Node.js 基础概念与示例

### 3.1 创建简单的 HTTP 服务器

```javascript
// 引入 HTTP 模块
const http = require('http');

// 创建 HTTP 服务器
const server = http.createServer((req, res) => {
  // 设置响应头
  res.writeHead(200, {'Content-Type': 'text/plain'});
  
  // 发送响应数据
  res.end('Hello World! 这是我的第一个 Node.js 服务器\n');
});

// 服务器监听在 3000 端口
server.listen(3000, '127.0.0.1', () => {
  console.log('服务器正在运行，访问 http://127.0.0.1:3000/');
});
```

**代码解析**：
- 首先引入 Node.js 的 HTTP 模块
- 使用 `createServer` 方法创建一个服务器，并传入回调函数处理请求和响应
- 回调函数接收两个参数：请求对象(req)和响应对象(res)
- 使用 `res.writeHead` 设置 HTTP 响应头
- 使用 `res.end` 发送响应数据并结束响应
- 最后在 3000 端口启动服务器，并输出启动成功的消息

### 3.2 文件系统操作

```javascript
// 引入文件系统模块
const fs = require('fs');

// 同步读取文件
try {
  const data = fs.readFileSync('example.txt', 'utf8');
  console.log('同步读取文件内容:', data);
} catch (err) {
  console.error('同步读取错误:', err);
}

// 异步读取文件
fs.readFile('example.txt', 'utf8', (err, data) => {
  if (err) {
    console.error('异步读取错误:', err);
    return;
  }
  console.log('异步读取文件内容:', data);
});

// 写入文件
const content = '这是要写入文件的内容';
fs.writeFile('output.txt', content, (err) => {
  if (err) {
    console.error('写入错误:', err);
    return;
  }
  console.log('文件写入成功!');
});
```

**代码解析**：
- 使用 `fs` 模块处理文件系统操作
- 展示了同步和异步两种读取文件的方式
- 同步方法使用 try/catch 捕获错误
- 异步方法使用回调函数处理结果
- 使用 `writeFile` 异步写入文件内容

### 3.3 使用 Express 框架创建 API

Express 是 Node.js 最流行的 Web 应用框架，简化了 API 和网站开发。

安装 Express:
```bash
npm install express
```

创建简单的 Express 应用:

```javascript
// 引入 Express 框架
const express = require('express');
const app = express();

// 解析 JSON 请求体
app.use(express.json());

// 模拟数据库
const users = [
  { id: 1, name: '张三', age: 28 },
  { id: 2, name: '李四', age: 32 },
  { id: 3, name: '王五', age: 25 }
];

// 获取所有用户
app.get('/api/users', (req, res) => {
  res.json(users);
});

// 获取特定用户
app.get('/api/users/:id', (req, res) => {
  const user = users.find(u => u.id === parseInt(req.params.id));
  if (!user) return res.status(404).json({ message: '用户不存在' });
  res.json(user);
});

// 创建新用户
app.post('/api/users', (req, res) => {
  const { name, age } = req.body;
  if (!name || !age) {
    return res.status(400).json({ message: '请提供姓名和年龄' });
  }
  
  const newUser = {
    id: users.length + 1,
    name,
    age
  };
  
  users.push(newUser);
  res.status(201).json(newUser);
});

// 启动服务器
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`API 服务器运行在 http://localhost:${PORT}`);
});
```

**代码解析**：
- 引入并初始化 Express 应用
- 使用 `express.json()` 中间件解析 JSON 请求体
- 实现了三个 API 端点:
  - GET `/api/users` - 获取所有用户
  - GET `/api/users/:id` - 获取特定 ID 的用户
  - POST `/api/users` - 创建新用户
- 使用适当的 HTTP 状态码和 JSON 响应
- 使用参数化路由 (`:id`) 捕获 URL 中的变量

## 4. Node.js 的异步编程模型

Node.js 的一个核心特性是异步编程。以下是不同的异步编程方式：

### 4.1 回调函数（传统方式）

```javascript
function fetchData(callback) {
  setTimeout(() => {
    const data = { name: '示例数据' };
    callback(null, data);
  }, 1000);
}

fetchData((err, data) => {
  if (err) {
    console.error('出错了:', err);
    return;
  }
  console.log('获取到数据:', data);
});
```

### 4.2 Promise（更现代的方式）

```javascript
function fetchData() {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      const data = { name: '示例数据' };
      // 假设成功获取数据
      resolve(data);
      // 如果出错，可以使用: reject(new Error('获取数据失败'));
    }, 1000);
  });
}

fetchData()
  .then(data => {
    console.log('获取到数据:', data);
  })
  .catch(err => {
    console.error('出错了:', err);
  });
```

### 4.3 Async/Await（最新最直观的方式）

```javascript
function fetchData() {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      const data = { name: '示例数据' };
      resolve(data);
    }, 1000);
  });
}

async function processData() {
  try {
    const data = await fetchData();
    console.log('获取到数据:', data);
    // 处理更多异步操作...
    const processedData = await anotherAsyncFunction(data);
    console.log('处理后的数据:', processedData);
  } catch (err) {
    console.error('出错了:', err);
  }
}

processData();
```

## 5. Node.js 包管理与模块系统

### 5.1 NPM 基本命令

```bash
# 初始化项目
npm init

# 安装包
npm install express

# 安装开发依赖
npm install --save-dev nodemon

# 全局安装
npm install -g pm2

# 运行脚本
npm run start
```

### 5.2 创建和使用自己的模块

文件: `calculator.js`
```javascript
// 定义模块功能
function add(a, b) {
  return a + b;
}

function subtract(a, b) {
  return a - b;
}

// 导出模块功能
module.exports = {
  add,
  subtract
};
```

文件: `app.js`
```javascript
// 导入模块
const calculator = require('./calculator');

// 使用模块功能
console.log('加法结果:', calculator.add(5, 3));  // 输出: 8
console.log('减法结果:', calculator.subtract(5, 3));  // 输出: 2
```

## 6. 实际应用场景

Node.js 适合构建以下类型的应用：

- **API 服务器**：为前端或移动应用提供数据服务
- **实时应用**：聊天应用、在线游戏、协作工具
- **流数据应用**：数据转换、处理大文件
- **微服务**：构建分布式系统的小型服务
- **命令行工具**：自动化脚本和工具

## 7. 调试与开发工具

- **Nodemon**：自动重启服务器工具
- **PM2**：生产环境进程管理器
- **VS Code 调试**：Node.js 集成调试
- **Chrome DevTools**：使用 `--inspect` 标志进行调试

## 8. 注意事项与最佳实践

- 避免在主事件循环中执行长时间运算，会阻塞所有请求
- 妥善处理错误和异常（特别是在异步代码中）
- 使用环境变量存储敏感信息和配置
- 在生产环境中使用进程管理器（如 PM2）
- 实施适当的日志记录和监控系统