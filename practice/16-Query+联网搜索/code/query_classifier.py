"""
联网搜索需求判断模块
通过LLM自动判断查询是否需要联网搜索
"""

import os
from enum import Enum
from typing import Optional
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

try:
    from .config import config
except ImportError:
    from config import config

# 加载根目录的 .env 文件
project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)


class SearchNeedType(Enum):
    """搜索需求类型枚举"""

    TIME_SENSITIVE = "时效性"
    WEATHER = "天气"
    NEWS = "新闻资讯"
    PRICE = "价格行情"
    REALTIME_DATA = "实时数据"
    NOT_NEEDED = "不需要联网"


class QueryClassifier:
    """
    查询分类器
    通过LLM自动判断查询是否需要联网搜索，以及具体的搜索场景类型
    """

    # 搜索场景描述
    SEARCH_SCENARIOS = {
        SearchNeedType.TIME_SENSITIVE: "需要最新、即时或当前时间点的信息，如'今天'、'最近'、'最新'等",
        SearchNeedType.WEATHER: "需要天气、气温、空气质量等气象相关信息",
        SearchNeedType.NEWS: "需要新闻、热点事件、动态报道等资讯信息",
        SearchNeedType.PRICE: "需要股票、汇率、商品价格等实时行情信息",
        SearchNeedType.REALTIME_DATA: "需要实时变化的数据，如交通、比赛比分、直播状态等",
        SearchNeedType.NOT_NEEDED: "不需要联网搜索，可以使用已有知识库回答",
    }

    def __init__(self, model: Optional[str] = None):
        """
        初始化查询分类器

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

        logger.info(f"查询分类器初始化完成 (模型: {self.model})")

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
                temperature=0.1,  # 分类任务使用较低温度
                max_tokens=500
            )

            return response.choices[0].message.content or ""  # type: ignore

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return ""

    def classify(self, query: str) -> dict:
        """
        对查询进行分类，判断是否需要联网搜索

        Args:
            query: 用户查询

        Returns:
            包含分类结果的字典:
            {
                "query": 原查询,
                "need_web_search": 是否需要联网搜索,
                "search_type": 搜索类型,
                "reason": 判断理由,
                "suggested_platforms": 建议的搜索平台
            }
        """
        logger.info(f"分类查询: '{query}'")

        # 构建系统提示词
        system_prompt = """你是一个专业的查询分析助手，负责判断用户查询是否需要联网搜索获取最新信息。

你需要判断以下搜索场景：
1. 时效性 - 需要最新、即时或当前时间点的信息
2. 天气 - 需要天气、气温、空气质量等气象信息
3. 新闻资讯 - 需要新闻、热点事件、动态报道
4. 价格行情 - 需要股票、汇率、商品价格等实时行情
5. 实时数据 - 需要实时变化的数据（如交通、比赛比分等）
6. 不需要联网 - 可以使用已有知识库回答的问题

请根据查询内容，判断是否需要联网搜索，并说明理由。"""

        # 构建用户提示词
        user_prompt = f"""请分析以下查询是否需要联网搜索：

查询：{query}

请以JSON格式返回分析结果，格式如下：
{{
    "need_web_search": true或false,
    "search_type": "时效性/天气/新闻资讯/价格行情/实时数据/不需要联网",
    "reason": "判断理由",
    "suggested_platforms": ["建议的搜索平台1", "建议的搜索平台2"]
}}

注意：
1. 如果查询包含"今天"、"昨天"、"最近"、"最新"等时效性词汇，通常需要联网搜索
2. 如果询问天气、股价、新闻等实时信息，必须联网搜索
3. 如果是通用知识问题（如"什么是RAG"），通常不需要联网搜索
4. suggested_platforms只需在需要联网搜索时提供

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

            # 标准化结果
            classification_result = {
                "query": query,
                "need_web_search": result.get("need_web_search", False),
                "search_type": result.get("search_type", "不需要联网"),
                "reason": result.get("reason", ""),
                "suggested_platforms": result.get("suggested_platforms", [])
            }

            logger.info(f"分类结果: {classification_result['search_type']} - 需要联网: {classification_result['need_web_search']}")
            return classification_result

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            # 返回默认结果
            return {
                "query": query,
                "need_web_search": False,
                "search_type": "不需要联网",
                "reason": "无法解析分类结果",
                "suggested_platforms": []
            }

    def quick_classify(self, query: str) -> SearchNeedType:
        """
        快速分类，只返回搜索类型

        Args:
            query: 用户查询

        Returns:
            搜索需求类型枚举值
        """
        result = self.classify(query)
        type_str = result.get("search_type", "不需要联网")

        type_mapping = {
            "时效性": SearchNeedType.TIME_SENSITIVE,
            "天气": SearchNeedType.WEATHER,
            "新闻资讯": SearchNeedType.NEWS,
            "价格行情": SearchNeedType.PRICE,
            "实时数据": SearchNeedType.REALTIME_DATA,
            "不需要联网": SearchNeedType.NOT_NEEDED,
        }

        return type_mapping.get(type_str, SearchNeedType.NOT_NEEDED)

    def batch_classify(self, queries: list[str]) -> list[dict]:
        """
        批量分类查询

        Args:
            queries: 查询列表

        Returns:
            分类结果列表
        """
        results = []
        for query in queries:
            result = self.classify(query)
            results.append(result)
        return results

    def is_time_sensitive(self, query: str) -> bool:
        """
        判断查询是否具有时效性

        Args:
            query: 用户查询

        Returns:
            是否需要联网搜索时效性信息
        """
        result = self.classify(query)
        return result["search_type"] == "时效性" or result["need_web_search"]

    def get_search_scenario_description(self, search_type: SearchNeedType) -> str:
        """
        获取搜索场景描述

        Args:
            search_type: 搜索类型

        Returns:
            场景描述
        """
        return self.SEARCH_SCENARIOS.get(search_type, "未知场景")


if __name__ == "__main__":
    # 测试代码
    from loguru import logger

    logger.add("logs/query_classifier.log", rotation="1 day")

    classifier = QueryClassifier()

    # 测试查询
    test_queries = [
        "今天北京的天气怎么样？",
        "最新的iPhone价格是多少？",
        "什么是RAG技术？",
        "最近有什么新闻？",
        "特斯拉今天的股价是多少？",
        "迪士尼乐园在哪里？"
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"查询: {query}")
        result = classifier.classify(query)
        print(f"需要联网: {result['need_web_search']}")
        print(f"搜索类型: {result['search_type']}")
        print(f"判断理由: {result['reason']}")
        if result['suggested_platforms']:
            print(f"建议平台: {', '.join(result['suggested_platforms'])}")
