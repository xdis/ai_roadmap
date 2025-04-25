"""
从拉勾网职位发布中解析职位要求的模块。
"""
import re
import logging
from typing import List, Dict, Any

from mpc_client import MPCClient
import config

# 配置日志记录
logger = logging.getLogger(__name__)

class JDExtractor:
    """从拉勾网职位发布中提取和处理职位描述要求。"""
    
    def __init__(self, mpc_client: MPCClient):
        """
        初始化职位描述提取器。
        
        参数:
            mpc_client: 已初始化的MPC客户端，用于浏览器自动化
        """
        self.mpc_client = mpc_client
        self.job_title = ""
        self.job_requirements = []
        self.required_skills = []
        self.required_experience = []
        self.required_education = []
        self.job_details = {}
        
    async def extract_job_description(self) -> Dict[str, Any]:
        """
        从职位发布页面提取职位描述和要求。
        
        返回:
            包含职位要求和其他详情的字典
        """
        logger.info("从拉勾网提取职位描述")
        
        # 导航到职位页面
        success = self.mpc_client.navigate_to_url(config.LAGOU_JOB_URL)
        if not success:
            logger.error("无法导航到职位页面")
            return {}
            
        # 对职位页面截图以供验证
        self.mpc_client.take_screenshot("job_page.png")
        
        # 等待页面完全加载
        self.mpc_client.wait_for_selector(".position-title", timeout=10000)
        
        # 提取职位标题
        job_title_element = self.mpc_client.get_element_text(".position-title")
        if job_title_element:
            self.job_title = job_title_element
            logger.info(f"找到职位标题: {self.job_title}")
        
        # 点击职位查看详情
        self.mpc_client.click_element(".position-item")
        
        # 等待职位详情加载
        self.mpc_client.wait_for_selector(".job-description", timeout=10000)
        
        # 提取职位描述内容
        job_description = self.mpc_client.get_element_text(".job-description")
        if job_description:
            logger.info("成功提取职位描述文本")
            # 处理职位描述以提取要求
            self._process_job_description(job_description)
        else:
            logger.error("无法提取职位描述文本")
        
        # 汇总所有提取的信息
        self.job_details = {
            "title": self.job_title,
            "requirements": self.job_requirements,
            "skills": self.required_skills,
            "experience": self.required_experience,
            "education": self.required_education,
            "raw_description": job_description
        }
        
        logger.info(f"提取了 {len(self.required_skills)} 项技能要求, {len(self.required_experience)} 项经验要求")
        return self.job_details
    
    def _process_job_description(self, description: str) -> None:
        """
        处理职位描述文本以提取结构化要求。
        
        参数:
            description: 原始职位描述文本
        """
        if not description:
            return
            
        # 提取一般要求
        requirements_section = self._extract_section(description, ["任职要求", "岗位要求", "职位要求"])
        if requirements_section:
            # 按项目符号或数字分割
            requirements = re.split(r'\d+[.、）\)\.]|\n+[-•*]', requirements_section)
            # 清理并过滤空项
            self.job_requirements = [req.strip() for req in requirements if req.strip()]
        
        # 提取技能
        self.required_skills = self._extract_skills(description)
        
        # 提取经验要求
        self.required_experience = self._extract_experience(description)
        
        # 提取教育要求
        self.required_education = self._extract_education(description)
    
    def _extract_section(self, text: str, section_headers: List[str]) -> str:
        """
        从职位描述中提取特定部分。
        
        参数:
            text: 完整的职位描述文本
            section_headers: 要提取部分的可能标题
            
        返回:
            提取的部分文本或空字符串
        """
        for header in section_headers:
            pattern = rf"{header}[:：]?(.*?)(?=\n\n|\n[^\n]+[:：]|$)"
            matches = re.search(pattern, text, re.DOTALL)
            if matches:
                return matches.group(1).strip()
        return ""
    
    def _extract_skills(self, text: str) -> List[str]:
        """
        从职位描述中提取技术技能。
        
        参数:
            text: 职位描述文本
            
        返回:
            提取的技能列表
        """
        # 常见编程语言和技术
        common_skills = [
            "Python", "Java", "C\\+\\+", "JavaScript", "TypeScript", "Go", "Golang", "Rust",
            "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle",
            "React", "Vue", "Angular", "Node.js", "Django", "Flask", "Spring", "SpringBoot",
            "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Alibaba Cloud", "Tencent Cloud",
            "Machine Learning", "深度学习", "人工智能", "AI", "NLP", "计算机视觉", "Computer Vision",
            "数据分析", "Data Analysis", "大数据", "Big Data", "Hadoop", "Spark", "Flink",
            "DevOps", "CI/CD", "Git", "Linux", "Unix", "Shell"
        ]
        
        skills = []
        for skill in common_skills:
            if re.search(rf"\b{skill}\b", text, re.IGNORECASE):
                skills.append(skill.replace("\\", ""))
                
        # 查找特定技能的年限要求，如"3年以上Python经验"
        experience_patterns = [
            r"(\d+)[+]?\s*[年年]+以上\s*([A-Za-z0-9_+#]+)经验",
            r"(\d+)[+]?\s*年以上\s*([A-Za-z0-9_+#]+)开发经验",
            r"([A-Za-z0-9_+#]+)\s*(\d+)[+]?\s*[年年]+以上经验"
        ]
        
        for pattern in experience_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    # 两种顺序都可能："3年以上Python" 或 "Python 3年以上"
                    skill = match.group(2) if match.group(2) else match.group(1)
                    if skill and skill not in skills:
                        skills.append(skill)
        
        return skills
    
    def _extract_experience(self, text: str) -> List[str]:
        """
        从职位描述中提取经验要求。
        
        参数:
            text: 职位描述文本
            
        返回:
            经验要求列表
        """
        experience_requirements = []
        
        # 查找一般经验要求
        general_exp_patterns = [
            r"(\d+)[-~+]?(\d*)?\s*年以上[相关]?[工作]?经[验历]",
            r"工作经验[：:]\s*(\d+)[-~](\d+)年",
            r"([初中高资深]+级)\s*[开发工程师]+"
        ]
        
        for pattern in general_exp_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                experience_requirements.append(match.group(0))
        
        return experience_requirements
    
    def _extract_education(self, text: str) -> List[str]:
        """
        从职位描述中提取教育要求。
        
        参数:
            text: 职位描述文本
            
        返回:
            教育要求列表
        """
        education_requirements = []
        
        edu_patterns = [
            r"[学学历历要求]+[：:]\s*([本硕博专科]+科及以上)",
            r"([本硕博专科]+科及以上学[历历])",
            r"([本硕博专科]+科[以及/或]以上)",
            r"本科[及或/以及]+以上学[历历]"
        ]
        
        for pattern in edu_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                education_requirements.append(match.group(0))
        
        return education_requirements