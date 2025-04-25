"""
ResumeFilterMPC 项目的配置设置。
"""

# URL 配置
LAGOU_JOB_URL = "https://easy.lagou.com/position/multiChannel/myOnlinePositions.htm"
LAGOU_CANDIDATES_URL = "https://easy.lagou.com/can/new/index.htm"

# 浏览器设置
BROWSER_WIDTH = 1401
BROWSER_HEIGHT = 900

# CSS 选择器（可能需要更新以应对反爬虫措施）
CANDIDATE_LIST_SELECTOR = "div.style_name__Sfspg"  # 列表中的候选人姓名
NEXT_PAGE_SELECTOR = "li.lg-pagination-next"       # 下一页按钮
SAVE_RESUME_SELECTOR = "div.options-btn"           # 简历详情中的保存按钮
CANDIDATE_REJECT_SELECTOR = "div.reject-btn"       # "不合适"按钮

# 时间配置
PAGE_LOAD_WAIT = 3   # 等待页面加载的秒数
SCROLL_INTERVAL = 1  # 滚动间隔的秒数

# 分析参数
MINIMUM_MATCH_SCORE = 70  # 考虑候选人合适的最低匹配百分比