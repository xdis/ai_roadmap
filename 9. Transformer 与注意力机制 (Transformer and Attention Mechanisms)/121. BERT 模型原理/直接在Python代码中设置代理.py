import requests
import json

# 测试代理是否工作
print("正在测试Clash代理...")

# 设置代理配置
proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}

try:
    # 通过代理访问IP检测服务
    response = requests.get('http://httpbin.org/ip', proxies=proxies, timeout=5)
    ip_info = response.json()
    print(f"当前IP地址: {ip_info['origin']}")
    print("代理工作正常！✓")
    
    # 测试访问Hugging Face
    hf_response = requests.get('https://huggingface.co', proxies=proxies, timeout=5)
    print(f"Hugging Face访问状态码: {hf_response.status_code}")
    if hf_response.status_code == 200:
        print("成功访问Hugging Face！✓")
    else:
        print(f"访问Hugging Face返回状态码: {hf_response.status_code}")
        
except Exception as e:
    print(f"代理测试失败: {str(e)}")
    print("请检查Clash是否正确运行，以及端口设置是否为7890")