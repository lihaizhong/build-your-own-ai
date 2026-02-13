"""
Query+联网搜索改写模块
支持自动判断是否需要联网搜索，并进行查询改写
"""

from .query_classifier import QueryClassifier, SearchNeedType
from .query_rewriter import WebQueryRewriter
from .search_strategy import SearchStrategyGenerator, SearchStrategy
from .pipeline import WebSearchPipeline

__all__ = [
    "QueryClassifier",
    "SearchNeedType",
    "WebQueryRewriter",
    "SearchStrategyGenerator",
    "SearchStrategy",
    "WebSearchPipeline",
]
