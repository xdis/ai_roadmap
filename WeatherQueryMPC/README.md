# WeatherQueryMPC

WeatherQueryMPC是一个使用多方计算(Multi-Party Computation, MPC)协议安全查询天气信息的工具。该工具通过OpenAI API获取指定城市的天气信息，同时确保API密钥和查询过程的安全性。

## 功能特点

- 使用MPC协议保护API密钥和查询数据的安全
- 支持查询任何城市的当前天气信息
- 提供详细的天气报告，包括温度、湿度和风况等
- 日志记录每次查询的结果
- 易于使用的命令行界面

## 安装要求

该项目需要以下Python包：

```
requests>=2.28.0
cryptography>=37.0.0
```

## 使用方法

1. 直接运行快捷脚本查询深圳天气：

```bash
python run_weather_query.py
```

2. 查询其他城市的天气，例如北京：

```bash
python run_weather_query.py 北京
```

3. 或者直接使用主程序，提供更多参数：

```bash
python weather_query_mpc.py --location 上海 --api-key YOUR_API_KEY --log-file custom_log.log
```

## 项目结构

```
WeatherQueryMPC/
│
├── weather_query_mpc.py   # 主程序
├── run_weather_query.py   # 运行脚本
├── keys/                  # 存储加密密钥的目录
│   └── secure_key.key     # 自动生成的加密密钥
├── weather_query.log      # 查询日志
└── README.md              # 项目说明文档
```

## MPC协议说明

本项目实现了一个简化版的MPC协议，用于保护API密钥和查询数据。在真实的MPC系统中，计算会分散到多个参与方，而不会有任何一方获得完整的密钥或数据。

当前实现使用了Fernet对称加密来模拟MPC的安全性，将API密钥和查询数据加密后再进行处理，从而演示MPC的基本概念。

## 注意事项

- API密钥已在`run_weather_query.py`中预设，但建议不要在实际生产环境中硬编码API密钥
- 该项目仅用于演示目的，尚未实现完整的MPC协议
- 查询结果依赖于OpenAI模型的能力和知识

## 许可证

MIT License