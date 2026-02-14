"""
整合管道模块
将分类、改写、策略生成整合为完整的处理流程
"""

from typing import Optional
from dataclasses import dataclass, field
from loguru import logger

try:
    from .query_classifier import QueryClassifier, SearchNeedType
    from .query_rewriter import WebQueryRewriter
    from .search_strategy import SearchStrategyGenerator
    from .config import config
except ImportError:
    from query_classifier import QueryClassifier, SearchNeedType
    from query_rewriter import WebQueryRewriter
    from search_strategy import SearchStrategyGenerator
    from config import config


@dataclass
class PipelineResult:
    """处理管道结果数据类"""
    
    # 原始查询
    original_query: str
    
    # 分类结果
    need_web_search: bool
    search_type: str
    classification_reason: str
    
    # 改写结果
    rewritten_query: str = ""
    keywords: list[str] = field(default_factory=list)
    
    # 搜索策略
    platforms: list[str] = field(default_factory=list)
    search_operators: list[str] = field(default_factory=list)
    priority: int = 0
    notes: str = ""
    
    # 搜索URL（可选）
    search_urls: dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "original_query": self.original_query,
            "need_web_search": self.need_web_search,
            "search_type": self.search_type,
            "classification_reason": self.classification_reason,
            "rewritten_query": self.rewritten_query,
            "keywords": self.keywords,
            "platforms": self.platforms,
            "search_operators": self.search_operators,
            "priority": self.priority,
            "notes": self.notes,
            "search_urls": self.search_urls
        }
    
    def format_summary(self) -> str:
        """格式化输出摘要"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"原始查询: {self.original_query}")
        lines.append("-" * 60)
        
        if self.need_web_search:
            lines.append(f"需要联网搜索: 是")
            lines.append(f"搜索类型: {self.search_type}")
            lines.append(f"判断理由: {self.classification_reason}")
            lines.append("")
            lines.append(f"改写后查询: {self.rewritten_query}")
            lines.append(f"关键词: {', '.join(self.keywords)}")
            lines.append("")
            lines.append(f"推荐平台: {', '.join(self.platforms)}")
            if self.search_operators:
                lines.append(f"搜索操作符: {', '.join(self.search_operators)}")
            lines.append(f"优先级: {self.priority}/5")
            if self.notes:
                lines.append(f"搜索建议: {self.notes}")
            if self.search_urls:
                lines.append("")
                lines.append("搜索链接:")
                for platform, url in self.search_urls.items():
                    lines.append(f"  - {platform}: {url}")
        else:
            lines.append(f"需要联网搜索: 否")
            lines.append(f"判断理由: {self.classification_reason}")
            lines.append("")
            lines.append("建议: 使用本地知识库或已有数据回答")
        
        lines.append("=" * 60)
        return "\n".join(lines) # type: ignore


class WebSearchPipeline:
    """
    联网搜索处理管道
    整合分类、改写、策略生成的完整流程
    """

    def __init__(self, model: Optional[str] = None):
        """
        初始化处理管道

        Args:
            model: 使用的LLM模型名称
        """
        self.model = model or config.llm_model

        # 初始化各组件
        self.classifier = QueryClassifier(model=self.model)
        self.rewriter = WebQueryRewriter(model=self.model)
        self.strategy_generator = SearchStrategyGenerator(model=self.model)

        logger.info(f"联网搜索处理管道初始化完成 (模型: {self.model})")

    def process(
        self,
        query: str,
        generate_urls: bool = True
    ) -> PipelineResult:
        """
        执行完整的处理流程

        Args:
            query: 用户查询
            generate_urls: 是否生成搜索URL

        Returns:
            处理结果对象
        """
        logger.info(f"开始处理查询: '{query}'")

        # Step 1: 分类
        logger.info("【Step 1】查询分类...")
        classification = self.classifier.classify(query)

        # 构建基础结果
        result = PipelineResult(
            original_query=query,
            need_web_search=classification["need_web_search"],
            search_type=classification["search_type"],
            classification_reason=classification["reason"]
        )

        # 如果不需要联网搜索，直接返回
        if not classification["need_web_search"]:
            logger.info("查询不需要联网搜索，处理完成")
            return result

        # Step 2: 改写
        logger.info("【Step 2】查询改写...")
        search_type = self.classifier.quick_classify(query)
        rewrite_result = self.rewriter.rewrite(query, search_type)

        result.rewritten_query = rewrite_result["rewritten_query"]
        result.keywords = rewrite_result["keywords"]

        # Step 3: 生成搜索策略
        logger.info("【Step 3】生成搜索策略...")
        strategy = self.strategy_generator.generate(
            query=result.rewritten_query,
            search_type=search_type,
            keywords=result.keywords
        )

        result.platforms = strategy.platforms
        result.search_operators = strategy.search_operators
        result.priority = strategy.priority
        result.notes = strategy.notes

        # Step 4: 生成搜索URL（可选）
        if generate_urls and result.platforms:
            logger.info("【Step 4】生成搜索URL...")
            for platform in result.platforms[:3]:  # 最多生成3个URL
                search_query = result.keywords[0] if result.keywords else result.rewritten_query
                url = self.strategy_generator.generate_search_url(
                    query=search_query,
                    platform=platform
                )
                if url:
                    result.search_urls[platform] = url

        logger.info("查询处理完成")
        return result

    def process_batch(
        self,
        queries: list[str],
        generate_urls: bool = True
    ) -> list[PipelineResult]:
        """
        批量处理查询

        Args:
            queries: 查询列表
            generate_urls: 是否生成搜索URL

        Returns:
            处理结果列表
        """
        results = []
        for query in queries:
            result = self.process(query, generate_urls)
            results.append(result)
        return results

    def quick_process(self, query: str) -> dict:
        """
        快速处理，只返回关键信息

        Args:
            query: 用户查询

        Returns:
            简化的结果字典
        """
        result = self.process(query, generate_urls=False)

        return {
            "need_web_search": result.need_web_search,
            "search_type": result.search_type,
            "rewritten_query": result.rewritten_query,
            "keywords": result.keywords[:3] if result.keywords else [],
            "platforms": result.platforms[:3] if result.platforms else []
        }

    def classify_only(self, query: str) -> dict:
        """
        仅执行分类步骤

        Args:
            query: 用户查询

        Returns:
            分类结果字典
        """
        classification = self.classifier.classify(query)

        return {
            "query": query,
            "need_web_search": classification["need_web_search"],
            "search_type": classification["search_type"],
            "reason": classification["reason"],
            "suggested_platforms": classification["suggested_platforms"]
        }

    def rewrite_only(
        self,
        query: str,
        search_type_str: str
    ) -> dict:
        """
        仅执行改写步骤

        Args:
            query: 用户查询
            search_type_str: 搜索类型字符串

        Returns:
            改写结果字典
        """
        # 转换搜索类型
        type_mapping = {
            "时效性": SearchNeedType.TIME_SENSITIVE,
            "天气": SearchNeedType.WEATHER,
            "新闻资讯": SearchNeedType.NEWS,
            "价格行情": SearchNeedType.PRICE,
            "实时数据": SearchNeedType.REALTIME_DATA,
            "不需要联网": SearchNeedType.NOT_NEEDED,
        }
        search_type = type_mapping.get(search_type_str, SearchNeedType.NOT_NEEDED)

        result = self.rewriter.rewrite(query, search_type)

        return {
            "original_query": query,
            "rewritten_query": result["rewritten_query"],
            "keywords": result["keywords"],
            "search_type": search_type.value
        }

    def strategy_only(
        self,
        query: str,
        search_type_str: str,
        keywords: Optional[list[str]] = None
    ) -> dict:
        """
        仅生成搜索策略

        Args:
            query: 改写后的查询
            search_type_str: 搜索类型字符串
            keywords: 关键词列表

        Returns:
            搜索策略字典
        """
        # 转换搜索类型
        type_mapping = {
            "时效性": SearchNeedType.TIME_SENSITIVE,
            "天气": SearchNeedType.WEATHER,
            "新闻资讯": SearchNeedType.NEWS,
            "价格行情": SearchNeedType.PRICE,
            "实时数据": SearchNeedType.REALTIME_DATA,
            "不需要联网": SearchNeedType.NOT_NEEDED,
        }
        search_type = type_mapping.get(search_type_str, SearchNeedType.NOT_NEEDED)

        strategy = self.strategy_generator.generate(query, search_type, keywords)

        return self.strategy_generator.to_dict(strategy)


def demo():
    """演示函数"""
    from loguru import logger
    import sys

    # 配置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    print("\n" + "=" * 60)
    print(" 联网搜索处理管道演示")
    print("=" * 60)

    # 创建管道
    pipeline = WebSearchPipeline()

    # 测试查询
    test_queries = [
        "今天北京的天气怎么样？",
        "最新的iPhone 16价格是多少？",
        "什么是RAG技术？",
        "最近有什么AI新闻？",
        "特斯拉今天的股价是多少？",
        "迪士尼乐园在哪里？"
    ]

    for query in test_queries:
        print(f"\n处理查询: {query}")
        print("-" * 60)

        result = pipeline.process(query)
        print(result.format_summary())

        print()

    print("\n演示结束！")


if __name__ == "__main__":
    demo()
