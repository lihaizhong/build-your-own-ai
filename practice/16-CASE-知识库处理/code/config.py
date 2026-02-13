"""
配置管理模块
管理项目的所有配置参数
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(verbose=True)


def get_project_root() -> Path:
    """获取项目根目录"""
    try:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        return project_root
    except NameError:
        return Path.cwd()


@dataclass
class Config:
    """配置类"""
    
    # 路径配置
    project_root: Path = field(default_factory=get_project_root)
    data_dir: Optional[Path] = None
    indexes_dir: Optional[Path] = None
    versions_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    
    # Embedding配置
    text_embedding_model: str = "text-embedding-v4"
    text_embedding_dim: int = 1024
    
    # LLM配置
    llm_model: str = "qwen-turbo-latest"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000
    
    # 检索配置
    top_k: int = 5
    score_threshold: float = 0.7
    
    # 健康度检查权重
    coverage_weight: float = 0.4
    freshness_weight: float = 0.3
    consistency_weight: float = 0.3
    
    # 分词停用词
    stop_words: Optional[List[str]] = None
    
    def __post_init__(self):
        """初始化路径"""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.indexes_dir is None:
            self.indexes_dir = self.project_root / "user_data" / "indexes"
        if self.versions_dir is None:
            self.versions_dir = self.project_root / "user_data" / "versions"
        if self.cache_dir is None:
            self.cache_dir = self.project_root / "user_data" / "cache"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"
        
        # 确保目录存在
        self._ensure_directories()
        
        # 默认停用词
        if self.stop_words is None:
            self.stop_words = [
                '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
                '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
                '你', '会', '着', '没有', '看', '好', '自己', '这'
            ]
    
    def _ensure_directories(self):
        """确保所有目录存在"""
        for dir_path in [self.data_dir, self.indexes_dir, self.versions_dir,
                        self.cache_dir, self.output_dir]:
            if dir_path is not None:
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result


# 全局配置实例
config = Config()


def load_env_config():
    """加载环境变量配置"""
    config.llm_model = os.environ.get("LLM_MODEL", config.llm_model)
    config.text_embedding_model = os.environ.get(
        "TEXT_EMBEDDING_MODEL", config.text_embedding_model
    )
