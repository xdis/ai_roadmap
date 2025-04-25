# SecureDataFetchMPC

一个使用多方计算(MPC)协议安全地进行HTTP请求并将数据存储到Excel的项目。

## 项目概述

此项目演示了如何使用MPC(多方计算)协议安全地处理敏感信息(如Cookie)，进行HTTP请求获取用户数据，并将结果导出到Excel文件。

主要特点：
- 使用Shamir秘密共享算法实现MPC协议
- 安全处理HTTP请求中的敏感信息
- 自动将获取的用户数据导出到Excel文件
- 提供完整的错误处理和日志输出

## 安装

1. 克隆此仓库
2. 安装依赖:
   ```
   pip install -r requirements.txt
   ```
3. (可选) 创建`.env`文件，用于存储敏感信息:
   ```
   COOKIE_VALUE=your_actual_cookie_value
   ```

## 使用方法

运行主程序:

```
python main.py
```

这将:
1. 使用MPC协议分割敏感信息(Cookie)
2. 模拟多方协作重建敏感信息
3. 执行HTTP请求获取用户数据
4. 将获取的数据导出到Excel文件(`user_data.xlsx`)

## 技术细节

### MPC协议实现

项目使用Shamir秘密共享实现MPC协议，这是一种(t,n)阈值方案:
- 将秘密(如Cookie)分成n份
- 至少需要t份才能重建秘密
- 少于t份时无法获得任何关于秘密的信息

### 模块说明

- `config.py`: 配置参数和API设置
- `mpc_client.py`: MPC协议实现，基于Shamir秘密共享
- `data_handler.py`: 处理API请求和数据导出逻辑
- `main.py`: 程序入口

### 安全注意事项

- 项目默认使用模拟数据运行，确保代码可以完整执行
- 在生产环境中，各方应在物理隔离的机器上运行
- 实际使用时应实施更严格的安全措施

## 模拟与实际使用

代码默认使用模拟数据运行，以确保可以完整执行。如果要使用实际API:

1. 在`main.py`中将`use_mock`参数设置为`False`:
   ```python
   run_secure_data_fetch(use_mock=False)
   ```

2. 确保提供了正确的API URL、请求头和Cookie值(在`config.py`中)

## 数据格式

项目处理的用户数据格式示例:
```json
{
    "is_robot": "1",
    "system_enabled": false,
    "qiyu_system_url": "https://qiyukf.com/script/1a3aedbf845e814bcbb594b56f09789e.js",
    "user_id": "39ea6f5b-2ace-db67-e000-168897ece264",
    "user_name": "柏乐飞",
    "mobile": "13528788620",
    "project_name": "整体测试对接新平台企业合同用",
    "dept_name": "广州公司",
    "source": "泊寓后台",
    "tags": "广州公司：店长、店助、系统管理员;泊寓测试：线索商机超级权限、系统管理员、超级管理员;深圳公司：店长、店助、系统管理员;",
    "domain_account": "bailf"
}
```

导出的Excel文件将包含所有这些字段，并进行适当的列名本地化。