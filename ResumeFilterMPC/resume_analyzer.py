"""
处理拉勾网候选人简历的模块。
"""
import re
import logging
from typing import List, Dict, Any, Tuple
import time

from mpc_client import MPCClient
import config

# 配置日志记录
logger = logging.getLogger(__name__)

class ResumeAnalyzer:
    """分析候选人简历并与职位要求进行比较。"""
    
    def __init__(self, mpc_client: MPCClient, job_details: Dict[str, Any]):
        """
        初始化简历分析器。
        
        参数:
            mpc_client: 已初始化的MPC客户端，用于浏览器自动化
            job_details: 包含职位要求的字典
        """
        self.mpc_client = mpc_client
        self.job_details = job_details
        self.current_page = 1
        self.total_pages = 1
        self.processed_candidates = []
        self.page_has_candidates = True
    
    async def process_all_candidates(self) -> List[Dict[str, Any]]:
        """
        处理列表中的所有候选人，浏览所有页面。
        
        返回:
            带有评分的已处理候选人数据列表
        """
        logger.info("开始处理所有候选人")
        
        # 导航到候选人页面
        success = self.mpc_client.navigate_to_url(config.LAGOU_CANDIDATES_URL)
        if not success:
            logger.error("无法导航到候选人页面")
            return []
        
        # 等待候选人列表加载
        time.sleep(config.PAGE_LOAD_WAIT)
        
        # 对候选人页面截图
        self.mpc_client.take_screenshot("candidates_page.png")
        
        # 检查是否有候选人
        if not self._page_has_candidates():
            logger.info("页面上未找到候选人")
            return []
        
        # 确定总页数
        self._determine_total_pages()
        
        # 处理每一页的候选人
        while self.current_page <= self.total_pages and self.page_has_candidates:
            logger.info(f"正在处理第 {self.current_page} 页（共 {self.total_pages} 页）的候选人")
            
            # 处理当前页面上的候选人
            candidates_on_page = await self._process_current_page()
            self.processed_candidates.extend(candidates_on_page)
            
            # 检查是否有更多页面，如果有，转到下一页
            if self.current_page < self.total_pages:
                success = self._go_to_next_page()
                if not success:
                    logger.warning(f"无法导航到第 {self.current_page + 1} 页")
                    break
                self.current_page += 1
            else:
                break
        
        # 按匹配分数排序候选人
        self.processed_candidates.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        logger.info(f"已完成处理 {self.current_page} 页中的 {len(self.processed_candidates)} 名候选人")
        return self.processed_candidates
    
    async def _process_current_page(self) -> List[Dict[str, Any]]:
        """
        处理当前页面上的所有候选人。
        
        返回:
            当前页面的已处理候选人数据列表
        """
        # 等待页面完全加载
        time.sleep(config.PAGE_LOAD_WAIT)
        
        # 获取页面上的所有候选人元素
        candidate_elements = self.mpc_client.get_elements_by_selector(config.CANDIDATE_LIST_SELECTOR)
        
        if not candidate_elements:
            logger.warning("当前页面未找到候选人元素")
            self.page_has_candidates = False
            return []
            
        candidates_on_page = []
        
        # 处理每个候选人元素
        for i, _ in enumerate(candidate_elements):
            # 需要再次获取元素，以防DOM在查看候选人后发生变化
            updated_elements = self.mpc_client.get_elements_by_selector(config.CANDIDATE_LIST_SELECTOR)
            if i >= len(updated_elements):
                break
                
            # 点击候选人查看详情
            self.mpc_client.click_element(config.CANDIDATE_LIST_SELECTOR, index=i)
            
            # 等待候选人详情加载
            time.sleep(config.PAGE_LOAD_WAIT)
            
            # 处理候选人详情
            candidate_data = await self._process_candidate_details()
            
            if candidate_data:
                candidates_on_page.append(candidate_data)
                logger.info(f"已处理候选人: {candidate_data.get('name', '未知')}, 匹配分数: {candidate_data.get('match_score', 0)}")
            
            # 返回候选人列表
            self.mpc_client.navigate_to_url(config.LAGOU_CANDIDATES_URL)
            
            # 等待候选人列表重新加载
            time.sleep(config.PAGE_LOAD_WAIT)
        
        return candidates_on_page
        
    async def _process_candidate_details(self) -> Dict[str, Any]:
        """
        处理单个候选人的简历详情。
        
        返回:
            包含候选人信息和匹配分数的字典
        """
        # 对简历截图
        screenshot_filename = f"candidate_{int(time.time())}.png"
        self.mpc_client.take_screenshot(screenshot_filename)
        
        # 提取候选人基本信息
        name = self.mpc_client.get_element_text(".candidate-name") or "未知"
        current_position = self.mpc_client.get_element_text(".current-position") or ""
        education = self.mpc_client.get_element_text(".education-info") or ""
        experience_years = self._extract_experience_years(
            self.mpc_client.get_element_text(".experience-years") or ""
        )
        
        # 从简历中提取技能和经验
        resume_content = self.mpc_client.get_page_html()
        extracted_skills = self._extract_skills_from_resume(resume_content)
        
        # 检查简历附件
        has_attachment = self._check_for_attachments()
        
        # 计算匹配分数
        match_score, matching_reasons, missing_requirements = self._calculate_match_score(
            extracted_skills, experience_years, education, resume_content
        )
        
        # 确定候选人是否合适
        is_suitable = match_score >= config.MINIMUM_MATCH_SCORE
        
        # 如果不合适，在系统中标记为不合适
        if not is_suitable:
            self._mark_as_unsuitable()
        
        # 汇总候选人数据
        candidate_data = {
            "name": name,
            "current_position": current_position,
            "education": education,
            "experience_years": experience_years,
            "extracted_skills": extracted_skills,
            "has_attachment": has_attachment,
            "match_score": match_score,
            "matching_reasons": matching_reasons,
            "missing_requirements": missing_requirements,
            "is_suitable": is_suitable,
            "screenshot": screenshot_filename
        }
        
        return candidate_data
    
    def _check_for_attachments(self) -> bool:
        """
        检查候选人是否附有简历文档。
        
        返回:
            如果找到附件，则为True，否则为False
        """
        # 执行JavaScript检查保存按钮
        js_script = """
        const buttons = document.querySelectorAll('div.options-btn');
        for (let i = 0; i < buttons.length; i++) {
            if (buttons[i].textContent.includes('保存')) {
                return true;
            }
        }
        return false;
        """
        
        result = self.mpc_client.execute_js(js_script)
        return bool(result)
    
    def _extract_skills_from_resume(self, resume_content: str) -> List[str]:
        """
        从简历中提取提到的技能。
        
        参数:
            resume_content: 简历的HTML内容
            
        返回:
            提取的技能列表
        """
        extracted_skills = []
        
        # 检查职位要求中的每项技能
        for skill in self.job_details.get("skills", []):
            if re.search(rf"\b{re.escape(skill)}\b", resume_content, re.IGNORECASE):
                extracted_skills.append(skill)
        
        # 检查其他常见技能
        common_skills = [
            "Python", "Java", "C\\+\\+", "JavaScript", "TypeScript", "Go", "Golang", "Rust",
            "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle",
            "React", "Vue", "Angular", "Node.js", "Django", "Flask", "Spring", "SpringBoot",
            "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Alibaba Cloud", "Tencent Cloud",
            "Machine Learning", "深度学习", "人工智能", "AI", "NLP", "计算机视觉", "Computer Vision",
            "数据分析", "Data Analysis", "大数据", "Big Data", "Hadoop", "Spark", "Flink",
            "DevOps", "CI/CD", "Git", "Linux", "Unix", "Shell"
        ]
        
        for skill in common_skills:
            cleaned_skill = skill.replace("\\", "")
            if cleaned_skill not in extracted_skills and re.search(rf"\b{skill}\b", resume_content, re.IGNORECASE):
                extracted_skills.append(cleaned_skill)
        
        return extracted_skills
    
    def _extract_experience_years(self, experience_text: str) -> int:
        """
        从文本中提取工作经验年限。
        
        参数:
            experience_text: 包含经验信息的文本
            
        返回:
            工作经验年数，如果未找到则为0
        """
        if not experience_text:
            return 0
            
        # 寻找如"3年经验"、"3 years"等模式
        experience_patterns = [
            r"(\d+)\s*年",
            r"(\d+)\s*years",
            r"(\d+)\s*年以上",
            r"工作经验\s*(\d+)"
        ]
        
        for pattern in experience_patterns:
            match = re.search(pattern, experience_text)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    pass
        
        return 0
    
    def _calculate_match_score(
        self, 
        skills: List[str], 
        experience_years: int, 
        education: str, 
        resume_content: str
    ) -> Tuple[float, List[str], List[str]]:
        """
        根据职位要求计算候选人的匹配分数。
        
        参数:
            skills: 候选人的技能列表
            experience_years: 候选人的工作经验年限
            education: 候选人的教育背景
            resume_content: 完整的简历内容
            
        返回:
            (匹配分数, 匹配原因, 缺失要求)的元组
        """
        score = 0
        max_score = 0
        matching_reasons = []
        missing_requirements = []
        
        # 评分技能（总分的50%）
        if self.job_details.get("skills"):
            max_score += 50
            required_skills = self.job_details["skills"]
            matched_skills = [skill for skill in required_skills if skill in skills]
            skill_score = (len(matched_skills) / len(required_skills)) * 50 if required_skills else 0
            score += skill_score
            
            if matched_skills:
                matching_reasons.append(f"匹配 {len(matched_skills)}/{len(required_skills)} 项所需技能: {', '.join(matched_skills)}")
            
            missing_skills = [skill for skill in required_skills if skill not in skills]
            if missing_skills:
                missing_requirements.append(f"缺少所需技能: {', '.join(missing_skills)}")
        
        # 评分经验（总分的30%）
        if self.job_details.get("experience"):
            max_score += 30
            required_exp = 0
            for exp_req in self.job_details["experience"]:
                # 尝试从如"3年以上经验"的要求中提取年限
                exp_match = re.search(r"(\d+)", exp_req)
                if exp_match:
                    required_exp = max(required_exp, int(exp_match.group(1)))
            
            if required_exp > 0:
                if experience_years >= required_exp:
                    score += 30
                    matching_reasons.append(f"经验（{experience_years}年）满足或超过所需的{required_exp}年")
                elif experience_years >= required_exp * 0.7:
                    # 对于几乎满足要求的情况给予部分分数
                    exp_score = (experience_years / required_exp) * 30
                    score += exp_score
                    matching_reasons.append(f"经验（{experience_years}年）部分满足所需的{required_exp}年")
                else:
                    missing_requirements.append(f"经验不足: {experience_years}年 vs 所需{required_exp}年")
        
        # 评分教育（总分的20%）
        if self.job_details.get("education"):
            max_score += 20
            
            # 将教育水平映射为数值
            education_levels = {
                "专科": 1,
                "本科": 2,
                "硕士": 3,
                "博士": 4
            }
            
            # 确定候选人的教育水平
            candidate_edu_level = 0
            for level, value in education_levels.items():
                if level in education:
                    candidate_edu_level = max(candidate_edu_level, value)
            
            # 确定所需的教育水平
            required_edu_level = 0
            for edu_req in self.job_details["education"]:
                for level, value in education_levels.items():
                    if level in edu_req:
                        required_edu_level = max(required_edu_level, value)
            
            if required_edu_level > 0:
                if candidate_edu_level >= required_edu_level:
                    score += 20
                    matching_reasons.append(f"教育背景满足或超过要求")
                else:
                    missing_requirements.append(f"教育背景不满足要求")
        
        # 如果有最高分数，则标准化分数
        if max_score > 0:
            final_score = (score / max_score) * 100
        else:
            final_score = 0
            
        return final_score, matching_reasons, missing_requirements
    
    def _mark_as_unsuitable(self) -> bool:
        """
        在拉勾系统中将当前候选人标记为不合适。
        
        返回:
            如果成功则为True，否则为False
        """
        try:
            # 寻找"不合适"按钮
            reject_button = self.mpc_client.get_elements_by_selector(config.CANDIDATE_REJECT_SELECTOR)
            
            if reject_button:
                # 点击拒绝按钮
                self.mpc_client.click_element(config.CANDIDATE_REJECT_SELECTOR)
                logger.info("已将候选人标记为不合适")
                
                # 等待确认对话框并确认（如果需要）
                time.sleep(1)
                confirm_button = self.mpc_client.get_elements_by_selector(".confirm-btn")
                if confirm_button:
                    self.mpc_client.click_element(".confirm-btn")
                    logger.info("已确认拒绝")
                
                return True
            else:
                logger.warning("找不到拒绝按钮")
                return False
                
        except Exception as e:
            logger.error(f"标记候选人为不合适时出错: {str(e)}")
            return False
    
    def _page_has_candidates(self) -> bool:
        """
        检查当前页面是否有候选人。
        
        返回:
            如果找到候选人则为True，否则为False
        """
        candidate_elements = self.mpc_client.get_elements_by_selector(config.CANDIDATE_LIST_SELECTOR)
        return len(candidate_elements) > 0
    
    def _determine_total_pages(self) -> None:
        """确定候选人列表中的总页数。"""
        # 寻找分页信息
        pagination_info = self.mpc_client.get_element_text(".pagination-info")
        if pagination_info:
            # 尝试从如"1/5"或"第1页，共5页"的文本中提取总页数
            match = re.search(r'(\d+)\s*/\s*(\d+)', pagination_info)
            if match:
                try:
                    self.current_page = int(match.group(1))
                    self.total_pages = int(match.group(2))
                    logger.info(f"找到分页信息: 当前页 {self.current_page}, 总页数 {self.total_pages}")
                except (ValueError, IndexError):
                    pass
        
        # 如果无法从文本中提取，检查下一页按钮是否启用
        if self.total_pages <= 1:
            next_page_btn = self.mpc_client.get_elements_by_selector(config.NEXT_PAGE_SELECTOR)
            if next_page_btn:
                # 如果下一页按钮存在，则至少有2页
                self.total_pages = 2
                logger.info("找到下一页按钮，设置总页数至少为2")
    
    def _go_to_next_page(self) -> bool:
        """
        导航到候选人的下一页。
        
        返回:
            如果成功则为True，否则为False
        """
        next_btn = self.mpc_client.get_elements_by_selector(config.NEXT_PAGE_SELECTOR)
        if not next_btn:
            logger.warning("未找到下一页按钮")
            return False
            
        # 检查下一页按钮是否被禁用
        is_disabled = self.mpc_client.execute_js(
            f"return document.querySelector('{config.NEXT_PAGE_SELECTOR}').getAttribute('aria-disabled') === 'true'"
        )
        
        if is_disabled:
            logger.info("下一页按钮已禁用，没有更多页面")
            return False
            
        # 点击下一页按钮
        success = self.mpc_client.click_element(config.NEXT_PAGE_SELECTOR)
        if not success:
            logger.warning("无法点击下一页按钮")
            return False
            
        # 等待页面加载
        time.sleep(config.PAGE_LOAD_WAIT)
        return True