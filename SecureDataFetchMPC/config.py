"""
配置文件 - 存储项目的配置参数和HTTP请求信息
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API请求配置
API_URL = "https://test.inboyu.com/user/get-user-base-info"
QUERY_PARAMS = {
    "project_id": "39ef3060-6fef-1032-902b-ab988101ffe1"
}

# 请求头
HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-HK;q=0.6",
    "Connection": "keep-alive",
    "Referer": "https://test.inboyu.com/bill/contract?contract_id=3a196895-f72a-aabe-6271-e74f8d4f57fa",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": "\"Google Chrome\";v=\"135\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\""
}

# Cookie值 - 使用MPC协议进行保护
COOKIE_VALUE = os.getenv("COOKIE_VALUE", "gr_user_id=911a474d-6713-4a40-b1af-2b732b491abe; 91920d7ba6c457fc_gr_last_sent_cs1=39f1a0ac-6c39-3c1e-9257-d6a3ded549af; 8d6e63d0296f3c2a_gr_last_sent_cs1=39f1a0ac-6c39-3c1e-9257-d6a3ded549af; range=61d58a12-0867-11e6-b879-00163e003632; _function=; 86dbace612a70e11_gr_last_sent_cs1=IFSBYZFWP; a5d8d66a2d91a434_gr_last_sent_cs1=IFLPQRAJV; 86dbace612a70e11_gr_cs1=IFSBYZFWP; a5d8d66a2d91a434_gr_cs1=IFLPQRAJV; 989a4737ffd7f901_gr_last_sent_cs1=39ea6f5b-2ace-db67-e000-168897ece264; a00aa6528157fff1_gr_last_sent_cs1=39ea6f5b-2ace-db67-e000-168897ece264; 989a4737ffd7f901_gr_cs1=39ea6f5b-2ace-db67-e000-168897ece264; PHPSESSID=6mddt47msg39u2aj8goiiuctd3; 8d6e63d0296f3c2a_gr_cs1=39f1a0ac-6c39-3c1e-9257-d6a3ded549af; access_token_prod=eyJhbGciOiJSUzI1NiIsImtpZCI6ImU5YWM4NzQ2ZjQzMDRmNDM5OWYxYjc5OWVmNWMzZTAzIn0.eyJpYXQiOjE3NDUyMjE5NzYsIm5iZiI6MTc0NTIyMTk3NiwiZXhwIjoxNzQ1MzA4Mzc2LCJpc3MiOiJ4aWFvYm8iLCJhdWQiOiJ1c2VyX2NlbnRlciIsInV1aWQiOiIzYTE5Njk4Mi1kYjhhLWUwNGYtYTdkYS03YTdiNmQxYjc2NDMiLCJ0aW1lb3V0Ijo4NjQwMH0.El6uEixuGHbZoKrsKJpTGHwG83lOg2gAl3qkrmHmHCsXml_0J8RMwLit299w7wWXntGZDQcIRgBgiXCI_21nKapEitU18TThOqrI-HzRb2PaGKToG81CB-N2P88Hjd_6FeV3le2cfh0SV90JRFJItZ1bcvOuTjVW58ETcElEiYoqCFDPP6slVrg_Q-b6OjiniwI9Wmsx4MQaqRBX3_T9_EJIjRggteNSmnR5XXL8gjJjsLf-7mCTGSIgmPXTeUQQHXW3moSF1PBHkb5sS5w713OFoXszQz1AMWQUDA-haDIZDEJS4_HXtko7Sl6JxQPSJevGOWzD_UIXr10f5i3tEg; access_token_prod_expired=1745308378; 91920d7ba6c457fc_gr_cs1=39f1a0ac-6c39-3c1e-9257-d6a3ded549af; acw_tc=0ae5a7de17452850931264733e0063321f33ae1516ae611a3a35804b8609f1; a00aa6528157fff1_gr_session_id=3742402b-808e-42c6-884b-41da9a223876; a00aa6528157fff1_gr_last_sent_sid_with_cs1=3742402b-808e-42c6-884b-41da9a223876; access_token_test=eyJhbGciOiJSUzI1NiIsImtpZCI6ImU5YWM4NzQ2ZjQzMDRmNDM5OWYxYjc5OWVmNWMzZTAzIn0.eyJpYXQiOjE3NDUyODUxMTUsIm5iZiI6MTc0NTI4NTExNSwiZXhwIjoxNzQ1MzcxNTE1LCJpc3MiOiJ4aWFvYm8iLCJhdWQiOiJ1c2VyX2NlbnRlciIsInV1aWQiOiIzYTE5NmQ0Ni00N2IzLTU5N2QtZDJmZS0yODU5NGYwYjMzMTQiLCJ0aW1lb3V0Ijo4NjQwMH0.j8s9vTRvCcIiq8XD1Aa2h3g8YSMSPe_fr_NgDd8bm6HiqBa1VFoVdocdgeQiBfh9tLp3YRgFQ-Ofcmznh-I56zpXIOZ5lVyV6Nnna6920_RA9VHy0TRlguIdFCUPQZ0_rWednO6ehggwo1GdIQnoAZgoKpb2WkUzZX9ADLPZ_OCPMXNMJ8sFwtlFGdczNbWPMfetAzle3OEYGiTR1QO1HaTDjexhQvdyuhiDNlfDvZnvVwbVaYBHOFbtn1JWn-ovwMgUpyO9LAwhu0EwdImaLqqOMdgimUY4JEBlbbBKrfdLjt_tV6MK0YvQ8oOs64ax_WJVafZbshjSdqIcwNJFEw; access_token_test_expired=1745371515; _identity=ba91c3039b5fbc08b2ca8e191eefc1b9e5edfd874731e16886d49ee48edbb8e3a%3A2%3A%7Bi%3A0%3Bs%3A9%3A%22_identity%22%3Bi%3A1%3Bs%3A51%3A%22%5B%2239ea6f5b-2ace-db67-e000-168897ece264%22%2Cnull%2C14400%5D%22%3B%7D; _csrf=8fe736ee9920e846c235a4d8d3401dc0f51d7ff892b9198a265827d38453e90ba%3A2%3A%7Bi%3A0%3Bs%3A5%3A%22_csrf%22%3Bi%3A1%3Bs%3A32%3A%22PxkbeN7PN-hKZ7yyDvOi1FmIKge7SvtC%22%3B%7D; a00aa6528157fff1_gr_cs1=39ea6f5b-2ace-db67-e000-168897ece264; defaultDepartment_39ea6f5b-2ace-db67-e000-168897ece264=22891f9fd86eb8d1c28b5f4b973fa38bdc72bf67a9d49160d111e8724219bc80a%3A2%3A%7Bi%3A0%3Bs%3A54%3A%22defaultDepartment_39ea6f5b-2ace-db67-e000-168897ece264%22%3Bi%3A1%3Bs%3A36%3A%2261d58a12-0867-11e6-b879-00163e003632%22%3B%7D; project_id=39ef3060-6fef-1032-902b-ab988101ffe1; _identity=ba91c3039b5fbc08b2ca8e191eefc1b9e5edfd874731e16886d49ee48edbb8e3a%3A2%3A%7Bi%3A0%3Bs%3A9%3A%22_identity%22%3Bi%3A1%3Bs%3A51%3A%22%5B%2239ea6f5b-2ace-db67-e000-168897ece264%22%2Cnull%2C14400%5D%22%3B%7D; acw_tc=ac11000117452852696881695e008c7e339ff633699dcfb1d43bebb14f0582")

# MPC设置
MPC_PARTIES = 3  # 参与计算的方数量
THRESHOLD = 2    # 恢复秘密所需的最小分享数

# 输出设置
OUTPUT_FILE = "user_data.xlsx"  # Excel输出文件

# 模拟数据 - 用于本地测试
MOCK_RESPONSE = {
    "errcode": 0,
    "data": {
        "is_robot": "1",
        "system_enabled": False,
        "qiyu_system_url": "https://qiyukf.com/script/1a3aedbf845e814bcbb594b56f09789e.js",
        "user_id": "39ea6f5b-2ace-db67-e000-168897ece264",
        "user_name": "柏乐飞",
        "mobile": "13528788620",
        "project_name": "整体测试对接新平台企业合同用",
        "dept_name": "广州公司",
        "source": "泊寓后台",
        "tags": "广州公司：店长、店助、系统管理员;泊寓测试：线索商机超级权限、系统管理员、超级管理员;深圳公司：店长、店助、系统管理员;",
        "domain_account": "bailf"
    }
}