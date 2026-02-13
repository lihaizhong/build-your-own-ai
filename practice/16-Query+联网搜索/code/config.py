"""
配置管理模块
管理项目的所有配置参数
"""

import os
from pathlib import Path
from typing import Dict, Any
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
    output_dir: Path | None = None

    # LLM配置
    llm_model: str = "qwen-max"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000

    # 搜索配置
    search_platforms: list = field(default_factory=lambda: [
        "Google", "百度", "必应", "知乎", "微博", "微信公众号",
        "GitHub", "Stack Overflow", "知乎专栏", "CSDN"
    ])

    # 时效性关键词
    time_sensitive_keywords: list = field(default_factory=lambda: [
        "今天", "昨天", "最近", "最新", "当前", "今年", "本月", "本周",
        "现在", "目前", "即时", "实时", "刚刚", "近期"
    ])

    # 天气相关关键词
    weather_keywords: list = field(default_factory=lambda: [
        "天气", "气温", "温度", "下雨", "晴天", "阴天", "台风",
        "暴雨", "雾霾", "空气质量", "湿度"
    ])

    # 新闻资讯关键词
    news_keywords: list = field(default_factory=lambda: [
        "新闻", "消息", "报道", "事件", "发生", "宣布", "发布",
        "热点", "头条", "动态", "资讯"
    ])

    # 价格行情关键词
    price_keywords: list = field(default_factory=lambda: [
        "价格", "股价", "汇率", "油价", "金价", "行情", "涨跌",
        "股市", "基金", "期货", "比特币"
    ])

    def __post_init__(self):
        """初始化路径"""
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"

        # 确保目录存在
        self._ensure_directories()

    def _ensure_directories(self):
        """确保所有目录存在"""
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
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
