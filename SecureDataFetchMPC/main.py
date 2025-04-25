"""
主程序 - SecureDataFetchMPC项目入口点
使用MPC协议安全地处理HTTP请求并将数据导出到Excel
"""
from data_handler import run_secure_data_fetch

if __name__ == "__main__":
    print("启动安全数据获取系统...")
    
    # 默认使用模拟数据，确保代码可以完整运行
    # 如果要使用实际API，将use_mock设置为False
    run_secure_data_fetch(use_mock=True)
    
    print("\n程序执行完毕。")