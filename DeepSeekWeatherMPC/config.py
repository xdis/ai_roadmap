"""
配置文件 - 存储项目的配置参数
"""
import os
from dotenv import load_dotenv

# 加载环境变量（如果存在.env文件）
load_dotenv()

# DeepSeek API设置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "6666"  # 使用用户指定的密钥

# MPC设置
MPC_PARTIES = 3  # 参与计算的方数
THRESHOLD = 2    # 恢复秘密所需的最小分享数

# 查询设置
WEATHER_QUERY = "深圳今天天气怎么样"