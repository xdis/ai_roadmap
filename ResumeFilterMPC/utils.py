"""
ResumeFilterMPC 项目的实用工具函数。
"""
import os
import json
import logging
import datetime
from typing import Dict, List, Any

# 配置日志记录
logger = logging.getLogger(__name__)

def setup_directories() -> None:
    """创建项目所需的目录。"""
    dirs = ["screenshots", "logs", "reports"]
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"已创建目录: {directory}")

def save_report(candidates: List[Dict[str, Any]], job_details: Dict[str, Any]) -> str:
    """
    保存候选人筛选结果的报告。
    
    参数:
        candidates: 已处理的候选人列表
        job_details: 职位要求详情
        
    返回:
        保存的报告文件路径
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = "reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    report_file = os.path.join(report_dir, f"resume_filtering_report_{timestamp}.json")
    
    report_data = {
        "timestamp": timestamp,
        "job_title": job_details.get("title", "未知职位"),
        "job_requirements": {
            "skills": job_details.get("skills", []),
            "experience": job_details.get("experience", []),
            "education": job_details.get("education", [])
        },
        "candidates": []
    }
    
    # 添加候选人数据（不包括大型HTML内容）
    for candidate in candidates:
        candidate_data = {
            "name": candidate.get("name", "未知"),
            "match_score": candidate.get("match_score", 0),
            "is_suitable": candidate.get("is_suitable", False),
            "current_position": candidate.get("current_position", ""),
            "education": candidate.get("education", ""),
            "experience_years": candidate.get("experience_years", 0),
            "extracted_skills": candidate.get("extracted_skills", []),
            "matching_reasons": candidate.get("matching_reasons", []),
            "missing_requirements": candidate.get("missing_requirements", []),
            "screenshot": candidate.get("screenshot", "")
        }
        report_data["candidates"].append(candidate_data)
    
    # 按匹配分数排序候选人
    report_data["candidates"].sort(key=lambda x: x["match_score"], reverse=True)
    
    # 添加统计摘要
    report_data["summary"] = {
        "total_candidates": len(candidates),
        "suitable_candidates": sum(1 for c in candidates if c.get("is_suitable", False)),
        "unsuitable_candidates": sum(1 for c in candidates if not c.get("is_suitable", False)),
        "average_match_score": sum(c.get("match_score", 0) for c in candidates) / len(candidates) if candidates else 0
    }
    
    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"筛选报告已保存到 {report_file}")
    
    # 为了更好的可读性生成Markdown摘要
    md_report_file = os.path.join(report_dir, f"resume_filtering_report_{timestamp}.md")
    generate_markdown_report(report_data, md_report_file)
    
    return report_file

def generate_markdown_report(report_data: Dict[str, Any], output_file: str) -> None:
    """
    从JSON报告数据生成Markdown报告。
    
    参数:
        report_data: 报告数据字典
        output_file: 保存Markdown报告的路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write(f"# 简历筛选报告\n\n")
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 写入职位详情
        f.write(f"## 职位: {report_data.get('job_title', '未知职位')}\n\n")
        
        # 写入要求
        f.write("### 职位要求\n\n")
        
        # 技能
        if report_data["job_requirements"]["skills"]:
            f.write("#### 所需技能\n")
            for skill in report_data["job_requirements"]["skills"]:
                f.write(f"- {skill}\n")
            f.write("\n")
        
        # 经验
        if report_data["job_requirements"]["experience"]:
            f.write("#### 经验要求\n")
            for exp in report_data["job_requirements"]["experience"]:
                f.write(f"- {exp}\n")
            f.write("\n")
        
        # 教育
        if report_data["job_requirements"]["education"]:
            f.write("#### 教育要求\n")
            for edu in report_data["job_requirements"]["education"]:
                f.write(f"- {edu}\n")
            f.write("\n")
        
        # 写入摘要
        f.write("## 总结\n\n")
        f.write(f"- 总候选人数: {report_data['summary']['total_candidates']}\n")
        f.write(f"- 合适的候选人: {report_data['summary']['suitable_candidates']}\n")
        f.write(f"- 不合适的候选人: {report_data['summary']['unsuitable_candidates']}\n")
        f.write(f"- 平均匹配分数: {report_data['summary']['average_match_score']:.2f}%\n\n")
        
        # 写入候选人详情
        f.write("## 候选人详情\n\n")
        
        # 按适合性分组
        suitable = [c for c in report_data["candidates"] if c["is_suitable"]]
        unsuitable = [c for c in report_data["candidates"] if not c["is_suitable"]]
        
        # 写入合适的候选人
        f.write("### 合适的候选人\n\n")
        if suitable:
            for i, candidate in enumerate(suitable, 1):
                f.write(f"#### {i}. {candidate['name']} (匹配分数: {candidate['match_score']:.2f}%)\n\n")
                f.write(f"- 当前职位: {candidate['current_position']}\n")
                f.write(f"- 教育背景: {candidate['education']}\n")
                f.write(f"- 工作经验: {candidate['experience_years']} 年\n")
                
                if candidate["extracted_skills"]:
                    f.write("- 技能:\n")
                    for skill in candidate["extracted_skills"]:
                        f.write(f"  - {skill}\n")
                
                if candidate["matching_reasons"]:
                    f.write("- 匹配原因:\n")
                    for reason in candidate["matching_reasons"]:
                        f.write(f"  - {reason}\n")
                
                if candidate["missing_requirements"]:
                    f.write("- 缺失要求:\n")
                    for missing in candidate["missing_requirements"]:
                        f.write(f"  - {missing}\n")
                
                if candidate["screenshot"]:
                    f.write(f"- [截图]({candidate['screenshot']})\n")
                
                f.write("\n")
        else:
            f.write("未找到合适的候选人。\n\n")
        
        # 写入不合适的候选人
        f.write("### 不合适的候选人\n\n")
        if unsuitable:
            for i, candidate in enumerate(unsuitable, 1):
                f.write(f"#### {i}. {candidate['name']} (匹配分数: {candidate['match_score']:.2f}%)\n\n")
                f.write(f"- 当前职位: {candidate['current_position']}\n")
                f.write(f"- 教育背景: {candidate['education']}\n")
                f.write(f"- 工作经验: {candidate['experience_years']} 年\n")
                
                if candidate["missing_requirements"]:
                    f.write("- 拒绝原因:\n")
                    for missing in candidate["missing_requirements"]:
                        f.write(f"  - {missing}\n")
                
                f.write("\n")
        else:
            f.write("未找到不合适的候选人。\n\n")
    
    logger.info(f"Markdown报告已生成于 {output_file}")