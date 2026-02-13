"""
知识库处理模块
包含知识库问题生成、对话知识沉淀、健康度检查、版本管理等功能
"""

from .config import config
from .question_generator import KnowledgeBaseOptimizer
from .conversation_extractor import ConversationKnowledgeExtractor
from .health_checker import KnowledgeBaseHealthChecker
from .version_manager import KnowledgeBaseVersionManager

__all__ = [
    "config",
    "KnowledgeBaseOptimizer",
    "ConversationKnowledgeExtractor", 
    "KnowledgeBaseHealthChecker",
    "KnowledgeBaseVersionManager",
]
