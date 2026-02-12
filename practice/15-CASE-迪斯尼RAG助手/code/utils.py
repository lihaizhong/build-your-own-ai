"""
工具函数模块
提供各种通用工具函数
"""

import json
import hashlib
from pathlib import Path
from typing import Any, List, Optional
from datetime import datetime
import pickle


def get_file_hash(file_path: Path) -> str:
    """计算文件的MD5哈希值"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def save_json(data: Any, file_path: Path, indent: int = 2):
    """保存数据为JSON文件"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: Path) -> Any:
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, file_path: Path):
    """保存数据为pickle文件"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path: Path) -> Any:
    """加载pickle文件
    
    自动处理模块导入路径问题，支持跨环境加载索引文件
    """
    import sys
    
    # 添加 code 目录到模块搜索路径，解决 pickle 反序列化时的模块路径问题
    # file_path 通常是: project_root/user_data/indexes/xxx.pkl
    # 需要从 indexes 上两级到 project_root，然后找到 code 目录
    indexes_dir = file_path.parent  # user_data/indexes
    user_data_dir = indexes_dir.parent  # user_data
    project_root = user_data_dir.parent  # project_root
    code_dir = project_root / "code"
    
    if code_dir.exists() and str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    
    with open(file_path, "rb") as f:
        return pickle.load(f)


def table_to_markdown(table_data: List[List[str]], headers: Optional[List[str]] = None) -> str:
    """将表格数据转换为Markdown格式"""
    if not table_data:
        return ""
    
    if headers is None:
        headers = table_data[0]
        table_data = table_data[1:]
    
    # 创建表头
    md_lines = ["| " + " | ".join(headers) + " |"]
    
    # 创建分隔线
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # 创建数据行
    for row in table_data:
        md_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    
    return "\n".join(md_lines)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将文本分割成块
    
    Args:
        text: 输入文本
        chunk_size: 每块的大小
        overlap: 块之间的重叠大小
    
    Returns:
        文本块列表
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_text(text: str) -> str:
    """清理文本，去除多余空格和换行"""
    if not text:
        return ""
    # 替换多个空格为单个空格
    text = " ".join(text.split())
    # 替换多个换行为单个换行
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix