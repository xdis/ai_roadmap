"""
ResumeFilterMPC 项目的主入口点。
"""
import asyncio
import logging
import sys
import os
from typing import Dict, Any

from mpc_client import MPCClient
from jd_extractor import JDExtractor
from resume_analyzer import ResumeAnalyzer
import utils
import config

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resume_filter.log'),
        logging.StreamHandler(sys.stdout)  # 添加控制台输出
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """应用程序的主入口点。"""
    logger.info("启动拉勾网简历筛选MPC系统")
    
    # 设置所需目录
    utils.setup_directories()
    
    # 初始化MPC客户端
    mpc_client = MPCClient(config.BROWSER_WIDTH, config.BROWSER_HEIGHT)
    logger.info(f"已初始化MPC客户端，视口大小为 {config.BROWSER_WIDTH}x{config.BROWSER_HEIGHT}")
    
    # 设置浏览器视口大小
    mpc_client.set_viewport_size()
    
    try:
        # 提取职位描述和要求
        jd_extractor = JDExtractor(mpc_client)
        job_details = await jd_extractor.extract_job_description()
        
        if not job_details:
            logger.error("无法提取职位详情，退出。")
            return
        
        logger.info(f"成功提取职位详情: {job_details.get('title', '未知')}")
        logger.info(f"找到 {len(job_details.get('skills', []))} 项所需技能, {len(job_details.get('experience', []))} 项经验要求")
        
        # 处理所有候选人
        resume_analyzer = ResumeAnalyzer(mpc_client, job_details)
        all_candidates = await resume_analyzer.process_all_candidates()
        
        if not all_candidates:
            logger.warning("没有处理任何候选人。")
            return
        
        # 生成报告
        report_file = utils.save_report(all_candidates, job_details)
        
        # 记录统计摘要
        suitable_candidates = [c for c in all_candidates if c.get('is_suitable', False)]
        logger.info(f"已处理 {len(all_candidates)} 名候选人，其中 {len(suitable_candidates)} 名合适")
        logger.info(f"报告已保存到 {report_file}")
        
        print("\n===== 简历筛选摘要 =====")
        print(f"职位: {job_details.get('title', '未知')}")
        print(f"总候选人数: {len(all_candidates)}")
        print(f"合适的候选人: {len(suitable_candidates)}")
        print(f"不合适的候选人: {len(all_candidates) - len(suitable_candidates)}")
        print(f"报告已保存到: {report_file}")
        print("===================================\n")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}", exc_info=True)
    finally:
        logger.info("简历筛选过程已完成")

if __name__ == "__main__":
    # 使用 asyncio.run() 替代旧的事件循环方法
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("用户中断了程序")
    finally:
        logger.info("应用程序已终止")