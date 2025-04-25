#!/usr/bin/env python3
"""
WeatherQueryMPC的运行脚本
此脚本提供了一种简单的方式来运行WeatherQueryMPC程序，
使用指定的OpenAI API密钥和位置。
"""

import os
import sys
import subprocess

# 项目中提供的OpenAI API密钥
API_KEY = "9999"

def main():
    """运行天气查询的主函数"""
    # 设置目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    keys_dir = os.path.join(script_dir, "keys")
    os.makedirs(keys_dir, exist_ok=True)
    
    # 获取位置（如果未提供则默认为深圳）
    location = "深圳"
    if len(sys.argv) > 1:
        location = sys.argv[1]
    
    # 使用API密钥运行主程序
    cmd = [
        sys.executable,  # 当前Python解释器
        os.path.join(script_dir, "weather_query_mpc.py"),
        "--location", location,
        "--api-key", API_KEY,
        "--key-path", os.path.join(keys_dir, "secure_key.key"),
        "--log-file", os.path.join(script_dir, "weather_query.log")
    ]
    
    print(f"正在为 {location} 运行WeatherQueryMPC...")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()