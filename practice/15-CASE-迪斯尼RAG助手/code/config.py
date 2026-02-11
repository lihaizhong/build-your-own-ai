"""
配置管理模块
管理项目的所有配置参数
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
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
    project_root: Path = get_project_root()
    data_dir: Path | None = None
    documents_dir: Path | None = None
    images_dir: Path | None = None
    indexes_dir: Path | None = None
    cache_dir: Path | None = None
    output_dir: Path | None = None
    
    # Embedding配置
    text_embedding_model: str = "text-embedding-v4"
    text_embedding_dim: int = 1024
    image_embedding_dim: int = 512
    clip_model_name: str = "ViT-B/32"
    
    # FAISS配置
    index_type: str = "IndexFlatL2"
    nlist: int = 100  # IVF索引的聚类中心数
    
    # 检索配置
    top_k: int = 5
    score_threshold: float = 0.7
    
    # LLM配置
    llm_model: str = "qwen-max"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # OCR配置
    tesseract_config: str = "--psm 6"
    ocr_language: str = "chi_sim+eng"
    
    # 检索关键词触发
    image_keywords: list | None = None
    
    def __post_init__(self):
        """初始化路径"""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.documents_dir is None:
            self.documents_dir = self.data_dir / "documents"
        if self.images_dir is None:
            self.images_dir = self.data_dir / "images"
        if self.indexes_dir is None:
            self.indexes_dir = self.project_root / "user_data" / "indexes"
        if self.cache_dir is None:
            self.cache_dir = self.project_root / "user_data" / "cache"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"
        
        # 确保目录存在
        self._ensure_directories()
        
        if self.image_keywords is None:
            self.image_keywords = ["海报", "图片", "图像", "照片", "截图", "展示"]
    
    def _ensure_directories(self):
        """确保所有目录存在"""
        for dir_path in [self.documents_dir, self.images_dir, 
                        self.indexes_dir, self.cache_dir, self.output_dir]:
            if dir_path is not None:
                dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
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
    
    # 从环境变量更新配置
    config.llm_model = os.environ.get("LLM_MODEL", config.llm_model)
    config.text_embedding_model = os.environ.get(
        "TEXT_EMBEDDING_MODEL", config.text_embedding_model
    )