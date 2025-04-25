"""
WeatherQueryMPC - 使用MPC协议的安全天气查询系统
该程序使用多方计算(MPC)协议通过OpenAI API安全地查询天气信息。
"""

import os
import json
import time
import argparse
import requests
from cryptography.fernet import Fernet
from datetime import datetime
import socket
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random

# 配置
API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-3.5-turbo"

# 设置默认超时时间和重试策略
DEFAULT_TIMEOUT = 30  # 30秒超时
DEFAULT_RETRIES = 3   # 重试3次
DEFAULT_BACKOFF_FACTOR = 2  # 指数退避因子

# 天气数据模拟 - 当API调用失败时使用
MOCK_WEATHER_DATA = {
    "深圳": {
        "temperature": "26°C",
        "humidity": "80%",
        "wind": "东南风 3级",
        "condition": "多云",
        "forecast": "今天多云，有阵雨，气温25°C至32°C"
    },
    "北京": {
        "temperature": "22°C",
        "humidity": "45%",
        "wind": "西北风 2级",
        "condition": "晴朗",
        "forecast": "今天晴朗，气温18°C至25°C"
    },
    "上海": {
        "temperature": "24°C",
        "humidity": "65%",
        "wind": "东风 3级",
        "condition": "阴天",
        "forecast": "今天阴天，气温20°C至26°C"
    },
    "广州": {
        "temperature": "28°C",
        "humidity": "75%",
        "wind": "东南风 2级",
        "condition": "多云",
        "forecast": "今天多云，气温24°C至30°C"
    }
}

class SecureComputation:
    """
    使用简化的MPC方法处理安全计算方面。
    在真实的MPC系统中，计算会分散到多个参与方。
    """
    
    def __init__(self, key_path=None, timeout=DEFAULT_TIMEOUT, retries=DEFAULT_RETRIES, 
                 backoff_factor=DEFAULT_BACKOFF_FACTOR, mock_enabled=True):
        """初始化安全计算环境"""
        self.timeout = timeout
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.mock_enabled = mock_enabled
        
        if key_path and os.path.exists(key_path):
            with open(key_path, 'rb') as key_file:
                self.key = key_file.read()
        else:
            self.key = Fernet.generate_key()
            if key_path:
                os.makedirs(os.path.dirname(key_path), exist_ok=True)
                with open(key_path, 'wb') as key_file:
                    key_file.write(self.key)
        
        self.cipher = Fernet(self.key)
        
        # 设置Session和重试策略
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.retries,
            backoff_factor=self.backoff_factor,  # 增加退避因子，减少频繁请求
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    
    def encrypt(self, data):
        """加密数据"""
        if isinstance(data, str):
            return self.cipher.encrypt(data.encode()).decode()
        return self.cipher.encrypt(json.dumps(data).encode()).decode()
    
    def decrypt(self, encrypted_data):
        """解密数据"""
        try:
            decrypted = self.cipher.decrypt(encrypted_data.encode())
            try:
                return json.loads(decrypted)
            except json.JSONDecodeError:
                return decrypted.decode()
        except Exception as e:
            print(f"解密错误: {e}")
            return None
    
    def secure_api_call(self, api_key, prompt, location):
        """
        使用MPC原则进行安全API调用
        
        在真正的MPC环境中，API密钥会在各方之间分割，
        并且计算会在不让任何一方获知完整密钥的情况下完成。
        这是一个简化的演示。
        """
        # 加密API密钥和输入数据（代表安全共享）
        encrypted_key = self.encrypt(api_key)
        encrypted_prompt = self.encrypt(prompt)
        encrypted_location = self.encrypt(location)
        
        # 记录安全计算过程
        print(f"[MPC] 安全计算在 {datetime.now()} 启动")
        print(f"[MPC] 输入数据已安全分区")
        
        # 在真实MPC系统中，这些将由不同的参与方处理
        # 这里我们通过在安全环境中解密来模拟
        return self._perform_api_call(
            self.decrypt(encrypted_key),
            self.decrypt(encrypted_prompt),
            self.decrypt(encrypted_location)
        )
    
    def _perform_api_call(self, api_key, prompt, location):
        """使用组装好的数据实际执行API调用"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 为特定位置格式化天气查询
        formatted_prompt = prompt.format(location=location)
        
        data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "temperature": 0.7
        }
        
        try:
            print(f"[MPC] 正在为 {location} 天气发起安全API请求")
            print(f"[MPC] 使用超时设置: {self.timeout}秒，重试次数: {self.retries}次，退避因子: {self.backoff_factor}")
            
            # 检查网络连接
            self._check_network_connection()
            
            # 使用session和timeout进行请求
            response = self.session.post(
                API_URL, 
                headers=headers, 
                json=data, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "location": location,
                    "response": result["choices"][0]["message"]["content"],
                    "timestamp": datetime.now().isoformat(),
                    "model": MODEL,
                    "source": "api"
                }
            else:
                error_message = self._analyze_error(response)
                print(error_message)
                
                # 如果是429错误且启用了模拟功能，返回模拟数据
                if response.status_code == 429 and self.mock_enabled:
                    return self._get_mock_weather(location)
                
                return {
                    "success": False,
                    "error": error_message,
                    "timestamp": datetime.now().isoformat()
                }
        except requests.exceptions.Timeout as e:
            error_message = f"请求超时: 连接到OpenAI API服务器超时。请检查您的网络连接并增加超时时间。详细信息: {str(e)}"
            print(error_message)
            
            # 超时时如果启用了模拟功能，返回模拟数据
            if self.mock_enabled:
                return self._get_mock_weather(location)
                
            return {
                "success": False,
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        except requests.exceptions.ConnectionError as e:
            error_message = f"连接错误: 无法连接到OpenAI API服务器。请检查您的网络连接、代理设置或防火墙。详细信息: {str(e)}"
            print(error_message)
            
            # 连接错误时如果启用了模拟功能，返回模拟数据
            if self.mock_enabled:
                return self._get_mock_weather(location)
                
            return {
                "success": False,
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            error_message = f"请求失败: {str(e)}"
            print(error_message)
            
            # 其他错误时如果启用了模拟功能，返回模拟数据
            if self.mock_enabled:
                return self._get_mock_weather(location)
                
            return {
                "success": False,
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
    
    def _check_network_connection(self):
        """检查网络连接状态"""
        try:
            # 尝试连接到一个可靠的服务器来测试网络连接
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            print("[MPC] 网络连接正常")
        except OSError:
            print("[MPC] 警告: 网络连接可能不稳定，这可能导致API调用失败")
    
    def _analyze_error(self, response):
        """分析API响应错误"""
        status_code = response.status_code
        error_text = response.text
        
        error_messages = {
            400: "请求格式错误: API请求格式不正确",
            401: "认证失败: API密钥无效或已过期",
            403: "权限不足: 您的API密钥没有足够的权限执行此操作",
            404: "资源不存在: 请求的资源不存在",
            429: "请求频率限制: 您已超过API请求速率限制，请减少请求频率或使用备用方案",
            500: "服务器错误: OpenAI服务器内部错误",
            502: "网关错误: OpenAI API服务器可能暂时不可用",
            503: "服务不可用: OpenAI API服务暂时不可用",
            504: "网关超时: 服务器响应超时"
        }
        
        if status_code in error_messages:
            return f"API错误 {status_code}: {error_messages[status_code]}。详细信息: {error_text}"
        else:
            return f"API错误: {status_code} - {error_text}"
    
    def _get_mock_weather(self, location):
        """获取模拟的天气数据，当API调用失败时使用"""
        global MOCK_WEATHER_DATA
        
        print(f"[MPC] API调用失败，切换到模拟天气数据")
        
        # 检查请求的位置是否在预设数据中，如果不在则使用一个随机位置的数据
        if location not in MOCK_WEATHER_DATA:
            random_location = random.choice(list(MOCK_WEATHER_DATA.keys()))
            weather = MOCK_WEATHER_DATA[random_location]
            print(f"[MPC] 没有找到 {location} 的预设数据，使用 {random_location} 的数据模拟")
        else:
            weather = MOCK_WEATHER_DATA[location]
        
        # 生成模拟的天气响应
        response = f"{location}今天的天气情况如下：\n\n" \
                   f"温度：{weather['temperature']}\n" \
                   f"湿度：{weather['humidity']}\n" \
                   f"风况：{weather['wind']}\n" \
                   f"天气状况：{weather['condition']}\n" \
                   f"天气预报：{weather['forecast']}\n\n" \
                   f"(注意：由于API调用失败，这是模拟生成的天气数据，仅供参考)"
        
        return {
            "success": True,
            "location": location,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "model": "模拟数据",
            "source": "mock"
        }

def log_result(result, log_file="weather_query.log"):
    """将结果记录到文件"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"--- 天气查询: {datetime.now().isoformat()} ---\n")
        if result["success"]:
            f.write(f"位置: {result['location']}\n")
            f.write(f"天气: {result['response']}\n")
            if "source" in result:
                f.write(f"数据源: {result['source']}\n")
            f.write(f"模型: {result['model']}\n")
        else:
            f.write(f"错误: {result['error']}\n")
        f.write("\n")

def check_api_key_format(api_key):
    """检查API密钥格式是否有效"""
    # 检查基本格式
    if not api_key or not isinstance(api_key, str):
        return False, "API密钥为空或格式错误"
    
    # 检查是否为标准OpenAI API密钥格式
    if api_key.startswith("sk-") and len(api_key) > 20:
        return True, "API密钥格式正确"
    
    return False, "API密钥格式不符合标准格式(应以'sk-'开头)"

def explain_rate_limiting():
    """解释API速率限制的详细信息"""
    print("\n" + "="*50)
    print("OpenAI API速率限制说明")
    print("="*50)
    print("您遇到了429错误（请求速率限制），这可能是由以下原因导致的：")
    print("1. 项目密钥(sk-proj-*)通常有比较严格的限制")
    print("2. 每个OpenAI账户的API密钥都有使用配额限制：")
    print("   - 免费试用账户：每分钟3次请求")
    print("   - 付费账户：根据账户级别不同，有不同的限制")
    print("3. 可能的解决方法：")
    print("   - 减少请求频率，添加更长的间隔时间")
    print("   - 使用指数退避策略（已在代码中实现）")
    print("   - 升级到付费账户或增加账户额度")
    print("   - 使用多个API密钥轮换使用")
    print("   - 启用本地模拟数据功能（已在代码中实现）")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="使用MPC的安全天气查询")
    parser.add_argument("--location", default="深圳", help="要查询天气的位置")
    parser.add_argument("--api-key", help="OpenAI API密钥")
    parser.add_argument("--key-path", default="keys/secure_key.key", help="加密密钥的路径")
    parser.add_argument("--log-file", default="weather_query.log", help="日志文件路径")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="API请求超时时间(秒)")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="API请求重试次数")
    parser.add_argument("--backoff-factor", type=float, default=DEFAULT_BACKOFF_FACTOR, help="重试退避因子")
    parser.add_argument("--no-mock", action="store_true", help="禁用模拟数据功能")
    
    args = parser.parse_args()
    
    # 使用参数或环境变量中的API密钥
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误: 需要OpenAI API密钥。请通过--api-key或OPENAI_API_KEY环境变量提供。")
        return
    
    # 检查API密钥格式
    key_valid, key_message = check_api_key_format(api_key)
    print(f"API密钥检查: {key_message}")
    
    # 如果是项目密钥格式，提供额外的警告
    if api_key.startswith("sk-proj-"):
        print("注意: 您正在使用项目密钥(sk-proj-*)，这类密钥通常有严格的速率限制")
    
    # 初始化安全计算环境
    secure_comp = SecureComputation(
        os.path.join(os.path.dirname(__file__), args.key_path),
        timeout=args.timeout,
        retries=args.retries,
        backoff_factor=args.backoff_factor,
        mock_enabled=not args.no_mock
    )
    
    # 天气查询提示模板
    prompt = "请告诉我{location}今天的天气情况，包括温度、湿度、风况等信息。请用中文回答，简明扼要。"
    
    # 执行安全计算
    print(f"正在查询 {args.location} 的天气...")
    start_time = time.time()
    
    result = secure_comp.secure_api_call(api_key, prompt, args.location)
    end_time = time.time()
    
    # 输出结果
    print("\n" + "="*50)
    print(f"天气查询结果 ({end_time - start_time:.2f}秒)")
    print("="*50)
    
    if result["success"]:
        print(f"位置: {args.location}")
        print(f"天气信息:\n{result['response']}")
        
        # 显示数据源信息
        if "source" in result and result["source"] == "mock":
            print(f"\n[注意] 上述数据来自本地模拟，非实时API查询结果")
    else:
        print(f"错误: {result['error']}")
        
        # 如果是429错误，提供更详细的解释
        if "429" in result["error"]:
            explain_rate_limiting()
        else:
            print("\n网络连接问题排查建议:")
            print("1. 检查您的互联网连接是否稳定")
            print("2. 确认是否需要使用代理才能访问OpenAI API")
            print("3. 验证您的防火墙设置是否允许访问api.openai.com")
            print("4. 尝试增加超时时间: --timeout 60")
            print("5. 检查API密钥是否有效")
    
    # 记录结果
    log_file_path = os.path.join(os.path.dirname(__file__), args.log_file)
    log_result(result, log_file_path)
    print(f"\n结果已记录到: {log_file_path}")

if __name__ == "__main__":
    main()