"""
Query改写核心模块
支持5种Query类型改写：上下文依赖型、对比型、模糊指代型、多意图型、反问型
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from dashscope import Generation
from dotenv import load_dotenv
from loguru import logger

# 加载根目录的 .env 文件
project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)


class QueryType(Enum):
    """Query类型枚举"""

    CONTEXT_DEPENDENT = "上下文依赖型"
    COMPARATIVE = "对比型"
    VAGUE_REFERENCE = "模糊指代型"
    MULTI_INTENT = "多意图型"
    RHETORICAL = "反问型"
    NORMAL = "普通型"


class QueryRewriter:
    """
    Query改写器
    通过LLM自动识别Query类型并进行改写，提升RAG检索效果
    """

    def __init__(self, model: str = "qwen-turbo-latest"):
        """
        初始化Query改写器

        Args:
            model: 使用的LLM模型名称
        """
        self.model = model
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY未设置，请检查环境变量")

    def _call_llm(self, prompt: str) -> str:
        """
        调用LLM生成响应

        Args:
            prompt: 输入提示词

        Returns:
            LLM生成的响应文本
        """
        try:
            response = Generation.call(
                model=self.model,
                prompt=prompt,
                api_key=self.api_key,
            )
            if response.status_code == 200:
                return response.output.text.strip()
            else:
                logger.error(f"LLM调用失败: {response.code} - {response.message}")
                return ""
        except Exception as e:
            logger.error(f"LLM调用异常: {e}")
            return ""

    def identify_query_type(self, query: str, conversation_history: str = "") -> QueryType:
        """
        识别Query类型

        Args:
            query: 用户查询
            conversation_history: 对话历史

        Returns:
            Query类型枚举值
        """
        prompt = f"""请分析以下用户查询的类型。

对话历史：
{conversation_history if conversation_history else "无"}

当前查询：{query}

Query类型定义：
1. 上下文依赖型：包含"还有"、"其他"、"另外"等需要上下文理解的词汇
2. 对比型：包含"哪个"、"比较"、"更"、"区别"等比较词汇
3. 模糊指代型：包含"它"、"他们"、"这个"、"那个"、"都"等指代词
4. 多意图型：包含多个独立问题，通常用"和"、"同时"、"另外"连接
5. 反问型：包含"不会"、"难道"、"不是吗"等反问语气
6. 普通型：不属于以上任何类型的普通查询

请直接返回类型名称（如：上下文依赖型），不要有其他内容。"""

        result = self._call_llm(prompt)
        
        type_mapping = {
            "上下文依赖型": QueryType.CONTEXT_DEPENDENT,
            "对比型": QueryType.COMPARATIVE,
            "模糊指代型": QueryType.VAGUE_REFERENCE,
            "多意图型": QueryType.MULTI_INTENT,
            "反问型": QueryType.RHETORICAL,
            "普通型": QueryType.NORMAL,
        }
        
        for key, value in type_mapping.items():
            if key in result:
                return value
        
        return QueryType.NORMAL

    def rewrite_context_dependent_query(
        self, query: str, conversation_history: str
    ) -> str:
        """
        上下文依赖型Query改写

        Args:
            query: 用户查询
            conversation_history: 对话历史

        Returns:
            改写后的完整查询
        """
        prompt = f"""你是一个查询改写专家。请根据对话历史，将用户的上下文依赖型查询改写为独立完整的查询。

对话历史：
{conversation_history}

当前查询：{query}

改写要求：
1. 补充查询中缺失的上下文信息
2. 使查询独立完整，不需要依赖对话历史
3. 保持用户原始意图
4. 直接输出改写后的查询，不要有任何解释

改写后的查询："""

        return self._call_llm(prompt)

    def rewrite_comparative_query(
        self, query: str, context_info: str = ""
    ) -> str:
        """
        对比型Query改写

        Args:
            query: 用户查询
            context_info: 相关上下文信息

        Returns:
            改写后的查询
        """
        prompt = f"""你是一个查询改写专家。请将用户的对比型查询改写为更明确的检索查询。

上下文信息：
{context_info if context_info else "无"}

当前查询：{query}

改写要求：
1. 明确对比的对象和维度
2. 拆分为多个具体的检索点
3. 保持对比意图
4. 直接输出改写后的查询

改写后的查询："""

        return self._call_llm(prompt)

    def rewrite_vague_reference_query(
        self, query: str, conversation_history: str
    ) -> str:
        """
        模糊指代型Query改写

        Args:
            query: 用户查询
            conversation_history: 对话历史

        Returns:
            改写后的明确查询
        """
        prompt = f"""你是一个查询改写专家。请根据对话历史，将用户查询中的模糊指代替换为具体对象。

对话历史：
{conversation_history}

当前查询：{query}

改写要求：
1. 识别查询中的代词（它、他们、这个、那个等）
2. 根据对话历史确定代词指代的具体对象
3. 将代词替换为具体对象名称
4. 直接输出改写后的查询

改写后的查询："""

        return self._call_llm(prompt)

    def rewrite_multi_intent_query(self, query: str) -> list[str]:
        """
        多意图型Query改写

        Args:
            query: 用户查询

        Returns:
            拆分后的多个独立查询列表
        """
        prompt = f"""你是一个查询改写专家。请将用户的多意图查询拆分为多个独立的子查询。

当前查询：{query}

拆分要求：
1. 识别查询中的多个独立问题
2. 每个子查询保持完整独立
3. 保持原始意图不变
4. 每行输出一个子查询，不要编号和其他内容

拆分后的子查询："""

        result = self._call_llm(prompt)
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        return queries if queries else [query]

    def rewrite_rhetorical_query(self, query: str) -> str:
        """
        反问型Query改写

        Args:
            query: 用户查询

        Returns:
            改写后的正面查询
        """
        prompt = f"""你是一个查询改写专家。请将用户的反问型查询改写为正面直接的查询。

当前查询：{query}

改写要求：
1. 去除反问语气
2. 转换为正面直接的提问方式
3. 保持原始意图
4. 直接输出改写后的查询

改写后的查询："""

        return self._call_llm(prompt)

    def auto_rewrite_query(
        self,
        query: str,
        conversation_history: str = "",
        context_info: str = "",
    ) -> dict:
        """
        自动识别Query类型并进行改写

        Args:
            query: 用户查询
            conversation_history: 对话历史
            context_info: 相关上下文信息

        Returns:
            包含原查询、类型、改写结果的字典
        """
        result = {
            "original_query": query,
            "query_type": "",
            "rewritten_query": "",
            "sub_queries": [],
        }

        # 识别Query类型
        query_type = self.identify_query_type(query, conversation_history)
        result["query_type"] = query_type.value
        logger.info(f"识别Query类型: {query_type.value}")

        # 根据类型进行改写
        if query_type == QueryType.NORMAL:
            result["rewritten_query"] = query
            logger.info("普通型查询，无需改写")

        elif query_type == QueryType.CONTEXT_DEPENDENT:
            rewritten = self.rewrite_context_dependent_query(
                query, conversation_history
            )
            result["rewritten_query"] = rewritten
            logger.info(f"上下文依赖型改写: {query} -> {rewritten}")

        elif query_type == QueryType.COMPARATIVE:
            rewritten = self.rewrite_comparative_query(query, context_info)
            result["rewritten_query"] = rewritten
            logger.info(f"对比型改写: {query} -> {rewritten}")

        elif query_type == QueryType.VAGUE_REFERENCE:
            rewritten = self.rewrite_vague_reference_query(query, conversation_history)
            result["rewritten_query"] = rewritten
            logger.info(f"模糊指代型改写: {query} -> {rewritten}")

        elif query_type == QueryType.MULTI_INTENT:
            sub_queries = self.rewrite_multi_intent_query(query)
            result["sub_queries"] = sub_queries
            result["rewritten_query"] = " | ".join(sub_queries)
            logger.info(f"多意图型拆分: {query} -> {sub_queries}")

        elif query_type == QueryType.RHETORICAL:
            rewritten = self.rewrite_rhetorical_query(query)
            result["rewritten_query"] = rewritten
            logger.info(f"反问型改写: {query} -> {rewritten}")

        return result
