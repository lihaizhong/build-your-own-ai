"""
搜索策略生成模块
根据查询类型生成详细的搜索策略，包括平台选择和关键词建议
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

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


@dataclass
class SearchStrategy:
    """搜索策略数据类"""
    
    query: str  # 改写后的查询
    search_type: str  # 搜索类型
    platforms: list[str]  # 推荐平台列表
    keywords: list[str]  # 搜索关键词
    search_operators: list[str]  # 搜索操作符
    priority: int  # 优先级 (1-5)
    notes: str  # 备注


class SearchStrategyGenerator:
    """
    搜索策略生成器
    根据查询类型和改写结果，生成详细的搜索策略
    """

    # 平台特性配置
    PLATFORM_FEATURES = {
        "Google": {
            "适合类型": ["时效性", "新闻资讯", "价格行情", "实时数据"],
            "优势": "全球最大搜索引擎，覆盖面广",
            "搜索技巧": ["使用site:限定网站", "使用引号精确匹配", "使用-排除关键词"]
        },
        "百度": {
            "适合类型": ["时效性", "新闻资讯", "天气"],
            "优势": "中文搜索优化，本地化程度高",
            "搜索技巧": ["使用site:限定网站", "使用intitle:标题搜索"]
        },
        "必应": {
            "适合类型": ["时效性", "新闻资讯"],
            "优势": "与微软生态集成，学术搜索强",
            "搜索技巧": ["使用site:限定网站", "使用filetype:文件类型"]
        },
        "知乎": {
            "适合类型": ["时效性", "新闻资讯"],
            "优势": "专业知识问答，深度讨论",
            "搜索技巧": ["关注高赞回答", "查看专栏文章"]
        },
        "微博": {
            "适合类型": ["新闻资讯", "实时数据"],
            "优势": "实时热点，社会舆论",
            "搜索技巧": ["查看热搜榜", "关注大V动态"]
        },
        "微信公众号": {
            "适合类型": ["新闻资讯", "价格行情"],
            "优势": "深度文章，专业解读",
            "搜索技巧": ["使用搜狗微信搜索", "关注行业公众号"]
        },
        "GitHub": {
            "适合类型": ["时效性"],
            "优势": "开源项目，技术文档",
            "搜索技巧": ["使用stars:>筛选", "使用language:语言筛选"]
        },
        "Stack Overflow": {
            "适合类型": ["时效性"],
            "优势": "技术问答，代码示例",
            "搜索技巧": ["使用tags标签筛选", "查看高赞回答"]
        },
        "雪球": {
            "适合类型": ["价格行情"],
            "优势": "股票讨论，投资分析",
            "搜索技巧": ["查看股票代码", "关注投资组合"]
        },
        "东方财富": {
            "适合类型": ["价格行情", "新闻资讯"],
            "优势": "财经数据，行情分析",
            "搜索技巧": ["查看个股页面", "关注研报"]
        },
        "中国天气网": {
            "适合类型": ["天气"],
            "优势": "官方天气数据，准确可靠",
            "搜索技巧": ["输入城市名称", "查看7天预报"]
        },
        "墨迹天气": {
            "适合类型": ["天气"],
            "优势": "用户体验好，数据直观",
            "搜索技巧": ["查看小时预报", "查看空气质量"]
        }
    }

    # 搜索操作符模板
    SEARCH_OPERATORS = {
        "精确匹配": '"{keyword}"',
        "标题搜索": 'intitle:{keyword}',
        "网站限定": 'site:{domain} {keyword}',
        "文件类型": 'filetype:{ext} {keyword}',
        "排除关键词": '{keyword} -{exclude}',
        "时间限定": '{keyword} after:{date}'
    }

    def __init__(self, model: Optional[str] = None):
        """
        初始化搜索策略生成器

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

        logger.info(f"搜索策略生成器初始化完成 (模型: {self.model})")

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
                max_tokens=1500
            )

            return response.choices[0].message.content or ""  # type: ignore

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return ""

    def generate(
        self,
        query: str,
        search_type: SearchNeedType,
        keywords: Optional[list[str]] = None
    ) -> SearchStrategy:
        """
        生成搜索策略

        Args:
            query: 改写后的查询
            search_type: 搜索类型
            keywords: 搜索关键词列表

        Returns:
            搜索策略对象
        """
        logger.info(f"生成搜索策略: '{query}' (类型: {search_type.value})")

        # 如果不需要联网搜索，返回空策略
        if search_type == SearchNeedType.NOT_NEEDED:
            return SearchStrategy(
                query=query,
                search_type=search_type.value,
                platforms=[],
                keywords=keywords or [],
                search_operators=[],
                priority=0,
                notes="无需联网搜索，使用本地知识库"
            )

        # 构建系统提示词
        system_prompt = f"""你是一个专业的搜索策略专家，负责为用户查询制定最优的搜索策略。

搜索类型: {search_type.value}

可用平台及其特点：
{self._format_platform_features()}

搜索操作符模板：
{self._format_search_operators()}

请根据查询类型和关键词，推荐最合适的搜索平台和搜索策略。"""

        # 构建用户提示词
        user_prompt = f"""请为以下查询生成搜索策略：

查询: {query}
搜索类型: {search_type.value}
关键词: {keywords if keywords else '待提取'}

请以JSON格式返回结果：
{{
    "platforms": ["平台1", "平台2", "平台3"],
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "search_operators": ["搜索操作符1", "搜索操作符2"],
    "priority": 1-5的优先级数字,
    "notes": "搜索建议和注意事项"
}}

要求：
1. platforms: 推荐2-4个最适合的平台，按优先级排序
2. keywords: 提取或生成3-5个核心搜索关键词
3. search_operators: 生成1-3个有用的搜索操作符
4. priority: 根据查询紧急程度和重要性给出1-5的优先级
5. notes: 提供实用的搜索建议

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

            strategy = SearchStrategy(
                query=query,
                search_type=search_type.value,
                platforms=result.get("platforms", []),
                keywords=result.get("keywords", keywords or []),
                search_operators=result.get("search_operators", []),
                priority=result.get("priority", 3),
                notes=result.get("notes", "")
            )

            logger.info(f"搜索策略生成完成: 平台={strategy.platforms}, 优先级={strategy.priority}")
            return strategy

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            # 返回默认策略
            return self._get_default_strategy(query, search_type, keywords)

    def _get_default_strategy(
        self,
        query: str,
        search_type: SearchNeedType,
        keywords: Optional[list[str]] = None
    ) -> SearchStrategy:
        """
        获取默认搜索策略

        Args:
            query: 查询
            search_type: 搜索类型
            keywords: 关键词

        Returns:
            默认搜索策略
        """
        # 根据搜索类型推荐平台
        default_platforms = {
            SearchNeedType.TIME_SENSITIVE: ["Google", "百度", "必应"],
            SearchNeedType.WEATHER: ["中国天气网", "墨迹天气", "百度"],
            SearchNeedType.NEWS: ["微博", "知乎", "微信公众号"],
            SearchNeedType.PRICE: ["东方财富", "雪球", "百度"],
            SearchNeedType.REALTIME_DATA: ["微博", "百度", "Google"],
            SearchNeedType.NOT_NEEDED: []
        }

        return SearchStrategy(
            query=query,
            search_type=search_type.value,
            platforms=default_platforms.get(search_type, ["Google", "百度"]),
            keywords=keywords or [],
            search_operators=[f'"{query}"'],
            priority=3,
            notes="使用默认搜索策略"
        )

    def _format_platform_features(self) -> str:
        """格式化平台特性信息"""
        lines = []
        for platform, features in self.PLATFORM_FEATURES.items():
            lines.append(f"- {platform}: {features['优势']}")
        return "\n".join(lines)

    def _format_search_operators(self) -> str:
        """格式化搜索操作符模板"""
        lines = []
        for name, template in self.SEARCH_OPERATORS.items():
            lines.append(f"- {name}: {template}")
        return "\n".join(lines)

    def get_platform_recommendation(self, search_type: SearchNeedType) -> list[str]:
        """
        根据搜索类型获取平台推荐

        Args:
            search_type: 搜索类型

        Returns:
            推荐平台列表
        """
        recommendations = []
        for platform, features in self.PLATFORM_FEATURES.items():
            if search_type.value in features["适合类型"]:
                recommendations.append(platform)
        return recommendations[:3]  # 返回前3个推荐

    def generate_search_url(
        self,
        query: str,
        platform: str,
        operator: Optional[str] = None
    ) -> str:
        """
        生成搜索URL

        Args:
            query: 查询
            platform: 平台名称
            operator: 搜索操作符

        Returns:
            搜索URL
        """
        # 平台URL模板
        url_templates = {
            "Google": "https://www.google.com/search?q={query}",
            "百度": "https://www.baidu.com/s?wd={query}",
            "必应": "https://www.bing.com/search?q={query}",
            "知乎": "https://www.zhihu.com/search?q={query}",
            "微博": "https://s.weibo.com/weibo/{query}",
            "GitHub": "https://github.com/search?q={query}",
            "Stack Overflow": "https://stackoverflow.com/search?q={query}",
            "雪球": "https://xueqiu.com/k?q={query}",
            "东方财富": "https://so.eastmoney.com/news/s?keyword={query}",
            "中国天气网": "http://www.weather.com.cn/weather/{query}.shtml",
        }

        # 合并查询和操作符
        search_query = f"{operator} {query}" if operator else query

        template = url_templates.get(platform)
        if template:
            return template.format(query=search_query)
        return ""

    def to_dict(self, strategy: SearchStrategy) -> dict:
        """
        将搜索策略转换为字典

        Args:
            strategy: 搜索策略对象

        Returns:
            字典表示
        """
        return {
            "query": strategy.query,
            "search_type": strategy.search_type,
            "platforms": strategy.platforms,
            "keywords": strategy.keywords,
            "search_operators": strategy.search_operators,
            "priority": strategy.priority,
            "notes": strategy.notes
        }


if __name__ == "__main__":
    # 测试代码
    from loguru import logger

    logger.add("logs/search_strategy.log", rotation="1 day")

    generator = SearchStrategyGenerator()

    # 测试生成策略
    test_cases = [
        ("今天北京天气 2026年2月", SearchNeedType.WEATHER),
        ("最新AI新闻", SearchNeedType.NEWS),
        ("特斯拉股价 实时", SearchNeedType.PRICE),
        ("iPhone 16 Pro 最新价格", SearchNeedType.TIME_SENSITIVE),
    ]

    for query, search_type in test_cases:
        print(f"\n{'=' * 60}")
        print(f"查询: {query}")
        print(f"类型: {search_type.value}")

        strategy = generator.generate(query, search_type)

        print(f"\n推荐平台: {', '.join(strategy.platforms)}")
        print(f"关键词: {', '.join(strategy.keywords)}")
        print(f"搜索操作符: {', '.join(strategy.search_operators)}")
        print(f"优先级: {strategy.priority}")
        print(f"备注: {strategy.notes}")

        # 生成搜索URL
        if strategy.platforms:
            url = generator.generate_search_url(
                strategy.keywords[0] if strategy.keywords else query,
                strategy.platforms[0]
            )
            print(f"搜索URL: {url}")
