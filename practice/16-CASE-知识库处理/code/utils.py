"""
工具函数模块
提供各种通用工具函数
"""

import os
import json
import re
from pathlib import Path
from typing import Any, List, Optional
from datetime import datetime
from loguru import logger

import jieba


def preprocess_json_response(response: Optional[str]) -> str:
    """预处理AI响应，移除markdown代码块格式"""
    if not response:
        return ""
    
    # 移除markdown代码块格式
    if response.startswith('```json'):
        response = response[7:]
    elif response.startswith('```'):
        response = response[3:]
    
    if response.endswith('```'):
        response = response[:-3]
    
    return response.strip()


def preprocess_text(text: str, stop_words: Optional[List[str]] = None) -> List[str]:
    """文本预处理和分词"""
    if not text:
        return []
    
    # 移除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    
    # 使用jieba分词
    words = jieba.lcut(text)
    
    # 默认停用词
    if stop_words is None:
        stop_words = [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这'
        ]
    
    # 过滤停用词和短词
    words = [word for word in words if len(word) > 1 and word not in stop_words]
    
    return words


def save_json(data: Any, file_path: Path, indent: int = 2) -> None:
    """保存数据为JSON文件"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: Path) -> Any:
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_datetime_str() -> str:
    """获取当前日期时间字符串"""
    return datetime.now().strftime("%Y年%m月%d日")


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_api_key() -> str:
    """获取API密钥"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")
    return api_key
