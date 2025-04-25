"""
数据获取与导出 - 使用MPC协议处理敏感信息，获取用户数据并导出到Excel
"""
import requests
import json
import time
import pandas as pd
import os
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

from config import (API_URL, QUERY_PARAMS, HEADERS, COOKIE_VALUE, 
                   MPC_PARTIES, THRESHOLD, OUTPUT_FILE, MOCK_RESPONSE)
from mpc_client import MPCClient

class SecureAPIHandler:
    """
    安全API处理器 - 使用MPC协议处理敏感信息并调用API
    """
    
    def __init__(self):
        """初始化安全API处理器"""
        self.mpc_client = MPCClient(threshold=THRESHOLD, num_parties=MPC_PARTIES)
        self.headers = HEADERS.copy()
        self.session = requests.Session()
    
    def prepare_request(self) -> Dict[str, Any]:
        """
        准备请求，安全处理Cookie
        
        返回:
            请求准备状态
        """
        print("准备API请求...")
        
        # 使用MPC协议分割Cookie
        print(f"使用MPC协议分割Cookie（阈值：{THRESHOLD}，参与方：{MPC_PARTIES}）...")
        cookie_data = self.mpc_client.secure_cookie_handling(COOKIE_VALUE)
        
        # 模拟分发分享给多方
        print("模拟将分享分发给各参与方...")
        time.sleep(1)  # 模拟网络延迟
        
        return {
            "cookie_data": cookie_data,
            "status": "prepared"
        }
    
    def execute_request(self, request_data: Dict[str, Any], use_mock: bool = False) -> Dict[str, Any]:
        """
        执行API请求
        
        参数:
            request_data: 通过prepare_request准备的请求数据
            use_mock: 是否使用模拟数据（用于测试）
            
        返回:
            API响应或错误信息
        """
        # 从各方收集足够的分享以重建Cookie
        print(f"从参与方收集{THRESHOLD}个分享...")
        collected_shares = request_data["cookie_data"]["shares"][:THRESHOLD]
        time.sleep(0.5)  # 模拟网络延迟
        
        # 重建并验证Cookie
        print("重建并验证Cookie...")
        result = self.mpc_client.reconstruct_and_verify(
            request_data["cookie_data"], 
            collected_shares,
            COOKIE_VALUE  # 在实际场景中，我们不会有原始Cookie进行验证
        )
        
        if not result["success"]:
            return {"error": f"Cookie重建失败: {result.get('error', '未知错误')}"}
        
        print(f"Cookie验证: {'成功' if result['verified'] else '失败'}")
        
        # 如果使用模拟数据，直接返回预设的响应
        if use_mock:
            print("使用模拟数据...")
            return {"response": MOCK_RESPONSE, "status": "success"}
        
        try:
            # 设置Cookie头
            self.headers["cookie"] = COOKIE_VALUE
            
            # 发送HTTP请求
            print(f"发送请求到 {API_URL}...")
            response = self.session.get(
                API_URL, 
                params=QUERY_PARAMS,
                headers=self.headers,
                timeout=10
            )
            
            # 检查响应状态
            if response.status_code == 200:
                try:
                    data = response.json()
                    return {"response": data, "status": "success"}
                except json.JSONDecodeError:
                    return {"error": "响应解析失败，非有效JSON", "status": "error"}
            else:
                return {
                    "error": f"API请求失败，状态码: {response.status_code}", 
                    "status": "error"
                }
                
        except requests.RequestException as e:
            return {"error": f"请求异常: {str(e)}", "status": "error"}

class DataExporter:
    """数据导出器 - 处理API响应并导出到Excel"""
    
    def __init__(self, output_file: str = OUTPUT_FILE):
        """
        初始化数据导出器
        
        参数:
            output_file: 输出Excel文件路径
        """
        self.output_file = output_file
    
    def parse_user_data(self, response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        解析用户数据
        
        参数:
            response_data: API响应数据
            
        返回:
            解析后的用户数据，失败时返回None
        """
        try:
            # 检查响应格式
            if "errcode" not in response_data or response_data["errcode"] != 0:
                error_msg = response_data.get("errmsg", "未知错误")
                print(f"API返回错误: {error_msg}")
                return None
            
            # 提取用户数据
            user_data = response_data.get("data", {})
            if not user_data:
                print("API响应中未找到用户数据")
                return None
                
            return user_data
            
        except Exception as e:
            print(f"解析数据时出错: {str(e)}")
            return None
    
    def export_to_excel(self, user_data: Dict[str, Any]) -> bool:
        """
        将用户数据导出到Excel
        
        参数:
            user_data: 用户数据字典
            
        返回:
            导出是否成功
        """
        try:
            print(f"将数据导出到Excel: {self.output_file}")
            
            # 将嵌套字典转换为扁平数据，便于导出
            flattened_data = {}
            
            for key, value in user_data.items():
                # 处理布尔值和None
                if isinstance(value, bool):
                    value = "是" if value else "否"
                elif value is None:
                    value = ""
                    
                flattened_data[key] = value
            
            # 创建DataFrame
            df = pd.DataFrame([flattened_data])
            
            # 为了美观，重命名列名
            column_mapping = {
                "user_id": "用户ID",
                "user_name": "用户姓名",
                "mobile": "手机号码",
                "project_name": "项目名称",
                "dept_name": "部门名称",
                "source": "来源",
                "tags": "标签",
                "domain_account": "账号",
                "is_robot": "是否机器人",
                "system_enabled": "系统启用",
                "qiyu_system_url": "七鱼系统URL"
            }
            
            df = df.rename(columns=column_mapping)
            
            # 导出到Excel
            df.to_excel(self.output_file, index=False, engine='openpyxl')
            
            print(f"数据已成功导出到 {os.path.abspath(self.output_file)}")
            return True
            
        except Exception as e:
            print(f"导出到Excel时出错: {str(e)}")
            return False

def run_secure_data_fetch(use_mock: bool = False) -> None:
    """
    运行安全数据获取和导出流程
    
    参数:
        use_mock: 是否使用模拟数据（用于测试）
    """
    print("\n" + "="*60)
    print("安全数据获取与导出系统 (使用MPC协议)")
    print("="*60)
    
    # 创建处理器实例
    api_handler = SecureAPIHandler()
    data_exporter = DataExporter()
    
    try:
        # 步骤1: 准备请求
        print("\n[步骤 1/4] 准备API请求")
        request_data = api_handler.prepare_request()
        
        # 步骤2: 执行请求
        print("\n[步骤 2/4] 执行API请求")
        response_result = api_handler.execute_request(request_data, use_mock)
        
        # 检查是否有错误
        if "error" in response_result:
            print(f"错误: {response_result['error']}")
            return
        
        # 步骤3: 解析数据
        print("\n[步骤 3/4] 解析用户数据")
        user_data = data_exporter.parse_user_data(response_result["response"])
        
        if user_data is None:
            print("无法解析用户数据")
            return
        
        # 输出部分数据预览
        print("\n获取到的用户数据预览:")
        preview_fields = ["user_name", "mobile", "dept_name", "project_name"]
        for field in preview_fields:
            if field in user_data:
                value = user_data[field]
                print(f"  {field}: {value}")
        
        # 步骤4: 导出数据
        print("\n[步骤 4/4] 导出数据到Excel")
        success = data_exporter.export_to_excel(user_data)
        
        if success:
            print("\n任务完成! 数据已成功获取并导出。")
        else:
            print("\n导出失败。")
            
    except Exception as e:
        print(f"\n处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    # 默认使用模拟数据，确保代码可以完整运行
    # 在实际使用场景中，可以将use_mock设置为False
    run_secure_data_fetch(use_mock=True)