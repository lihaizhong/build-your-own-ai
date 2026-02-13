"""
联网搜索Query改写模块
对需要联网搜索的查询进行改写，优化搜索效果
"""

import os
from typing import Optional
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

try:
    from .config import config
    from .query_classifier import SearchNeedType
except ImportError:
    from config import config
    from query_classifier import SearchNeedType

# 加载根目录的 .env 文件
project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)


class WebQueryRewriter:
    """
    联网搜索Query改写器
    根据搜索场景类型对查询进行针对性改写，提升搜索效果
    """

    # 各类型的改写策略
    REWRITE_STRATEGIES = {
        SearchNeedType.TIME_SENSITIVE: "添加时间限定词，明确时间范围",
        SearchNeedType.WEATHER: "添加地点和时间限定，使用标准天气查询格式",
        SearchNeedType.NEWS: "添加'最新'、'新闻'等关键词，优化新闻搜索",
        SearchNeedType.PRICE: "添加'实时'、'当前'等关键词，明确查询时间点",
        SearchNeedType.REALTIME_DATA: "添加'实时'、'最新'等关键词",
    }

    def __init__(self, model: Optional[str] = None):
        """
        初始化查询改写器

        Args:
            model: 使用的LLM模型名称
        """
        self.model = model or config.llm_model
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")

        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY未设置，请检查环境变量")

        # 初始化OpenAI客户端（兼容DashScope）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        logger.info(f"联网搜索Query改写器初始化完成 (模型: {self.model})")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用LLM生成响应

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词

        Returns:
            LLM生成的响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return response.choices[0].message.content or ""  # type: ignore

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return ""

    def rewrite(
        self,
        query: str,
        search_type: SearchNeedType,
        context: Optional[str] = None
    ) -> dict:
        """
        根据搜索类型改写查询

        Args:
            query: 原始查询
            search_type: 搜索类型
            context: 可选的上下文信息

        Returns:
            改写结果字典:
            {
                "original_query": 原查询,
                "rewritten_query": 改写后的查询,
                "search_type": 搜索类型,
                "rewrite_strategy": 改写策略,
                "keywords": 提取的关键词列表
            }
        """
        logger.info(f"改写查询: '{query}' (类型: {search_type.value})")

        # 如果不需要联网搜索，直接返回原查询
        if search_type == SearchNeedType.NOT_NEEDED:
            return {
                "original_query": query,
                "rewritten_query": query,
                "search_type": search_type.value,
                "rewrite_strategy": "无需改写",
                "keywords": []
            }

        # 根据搜索类型选择改写策略
        strategy = self.REWRITE_STRATEGIES.get(search_type, "通用优化")

        # 构建系统提示词
        system_prompt = f"""你是一个专业的搜索查询优化专家，负责改写用户的查询以获得更好的搜索结果。

当前查询类型: {search_type.value}
改写策略: {strategy}

改写要求：
1. 保持用户原始意图不变
2. 添加适当的时间、地点等限定词
3. 使用搜索引擎友好的表达方式
4. 提取查询中的关键词用于搜索
5. 如果查询模糊，合理补充缺失信息

请根据查询类型进行针对性改写。"""

        # 获取当前时间信息
        current_time = datetime.now().strftime("%Y年%m月%d日")

        # 构建用户提示词
        user_prompt = f"""请改写以下查询以优化搜索效果：

原始查询: {query}
当前日期: {current_time}
上下文信息: {context if context else "无"}

请以JSON格式返回结果：
{{
    "rewritten_query": "改写后的查询",
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "explanation": "改写说明"
}}

注意：
1. 改写后的查询应该简洁明确，适合搜索引擎
2. keywords应该包含核心搜索词，不超过5个
3. 如果是时效性查询，添加具体日期或时间范围
4. 如果是天气查询，明确地点和时间
5. 如果是价格查询，明确具体商品或股票代码

请直接返回JSON，不要有其他内容。"""

        # 调用LLM
        result_text = self._call_llm(system_prompt, user_prompt)

        # 解析结果
        import json
        try:
            # 尝试提取JSON内容
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = json.loads(result_text)

            rewrite_result = {
                "original_query": query,
                "rewritten_query": result.get("rewritten_query", query),
                "search_type": search_type.value,
                "rewrite_strategy": strategy,
                "keywords": result.get("keywords", []),
                "explanation": result.get("explanation", "")
            }

            logger.info(f"改写结果: '{rewrite_result['rewritten_query']}'")
            return rewrite_result

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            # 返回默认结果
            return {
                "original_query": query,
                "rewritten_query": query,
                "search_type": search_type.value,
                "rewrite_strategy": strategy,
                "keywords": [],
                "explanation": "改写失败，使用原查询"
            }

    def rewrite_for_time_sensitive(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """
        时效性查询改写

        Args:
            query: 原始查询
            context: 上下文信息

        Returns:
            改写后的查询
        """
        result = self.rewrite(query, SearchNeedType.TIME_SENSITIVE, context)
        return result["rewritten_query"]

    def rewrite_for_weather(
        self,
        query: str,
        location: Optional[str] = None
    ) -> str:
        """
        天气查询改写

        Args:
            query: 原始查询
            location: 地点信息

        Returns:
            改写后的查询
        """
        context = f"地点: {location}" if location else None
        result = self.rewrite(query, SearchNeedType.WEATHER, context)
        return result["rewritten_query"]

    def rewrite_for_news(self, query: str) -> str:
        """
        新闻查询改写

        Args:
            query: 原始查询

        Returns:
            改写后的查询
        """
        result = self.rewrite(query, SearchNeedType.NEWS)
        return result["rewritten_query"]

    def rewrite_for_price(self, query: str) -> str:
        """
        价格查询改写

        Args:
            query: 原始查询

        Returns:
            改写后的查询
        """
        result = self.rewrite(query, SearchNeedType.PRICE)
        return result["rewritten_query"]

    def extract_keywords(self, query: str) -> list[str]:
        """
        从查询中提取关键词

        Args:
            query: 用户查询

        Returns:
            关键词列表
        """
        system_prompt = "你是一个关键词提取专家，请从用户查询中提取核心关键词用于搜索引擎。"

        user_prompt = f"""请从以下查询中提取核心关键词：

查询: {query}

要求：
1. 提取3-5个最重要的关键词
2. 关键词应该能够准确表达查询意图
3. 按重要性排序

请直接返回关键词列表，每行一个，不要编号和其他内容。"""

        result_text = self._call_llm(system_prompt, user_prompt)
        keywords = [k.strip() for k in result_text.split("\n") if k.strip()]
        return keywords[:5]  # 最多返回5个关键词

    def expand_query(self, query: str, search_type: SearchNeedType) -> list[str]:
        """
        查询扩展：生成多个相关查询变体

        Args:
            query: 原始查询
            search_type: 搜索类型

        Returns:
            查询变体列表
        """
        system_prompt = f"""你是一个查询扩展专家，负责生成多个相关的搜索查询变体。

查询类型: {search_type.value}

请生成3-5个查询变体，要求：
1. 保持原始查询的核心意图
2. 使用不同的表达方式
3. 添加不同的限定词或修饰语
4. 适合搜索引擎检索"""

        user_prompt = f"""请为以下查询生成多个变体：

原始查询: {query}

请直接返回查询变体，每行一个，不要编号和其他内容。"""

        result_text = self._call_llm(system_prompt, user_prompt)
        variants = [v.strip() for v in result_text.split("\n") if v.strip()]
        return variants[:5]


if __name__ == "__main__":
    # 测试代码
    from loguru import logger

    logger.add("logs/query_rewriter.log", rotation="1 day")

    rewriter = WebQueryRewriter()

    # 测试改写
    test_cases = [
        ("今天北京的天气怎么样？", SearchNeedType.WEATHER),
        ("最新的iPhone价格", SearchNeedType.TIME_SENSITIVE),
        ("最近有什么AI新闻", SearchNeedType.NEWS),
        ("特斯拉今天的股价", SearchNeedType.PRICE),
        ("什么是机器学习？", SearchNeedType.NOT_NEEDED),
    ]

    for query, search_type in test_cases:
        print(f"\n{'=' * 60}")
        print(f"原查询: {query}")
        print(f"搜索类型: {search_type.value}")
        result = rewriter.rewrite(query, search_type)
        print(f"改写后: {result['rewritten_query']}")
        print(f"关键词: {result['keywords']}")
        if result.get('explanation'):
            print(f"说明: {result['explanation']}")
