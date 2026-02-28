# -*- coding: utf-8 -*-
"""
Coze API 配置文件
"""
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Coze API 配置
COZE_API_TOKEN = "你的coze_api_token"
COZE_BOT_ID = "你的coze_bot_id"

# Coze API 基础URL (中国区)
COZE_CN_BASE_URL = "https://api.coze.cn"

# 用户ID (可以是任意字符串，用于标识用户)
DEFAULT_USER_ID = "user_12345" 