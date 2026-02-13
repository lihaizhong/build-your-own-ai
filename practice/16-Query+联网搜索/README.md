# Query+联网搜索改写系统

基于LLM的智能查询改写系统，自动判断是否需要联网搜索，并进行查询优化和搜索策略生成。

> 本项目使用根目录的 `pyproject.toml` 和 `.env`，无需单独配置环境。

## 功能特性

### 🔍 智能分类
- 自动判断查询是否需要联网搜索
- 支持6种搜索场景识别：时效性、天气、新闻资讯、价格行情、实时数据、不需要联网
- 提供详细的判断理由

### ✏️ 查询改写
- 根据搜索类型进行针对性改写
- 添加时间、地点等限定词
- 提取核心搜索关键词
- 支持查询扩展（生成多个查询变体）

### 🎯 搜索策略生成
- 推荐最适合的搜索平台
- 生成搜索操作符建议
- 生成可直接使用的搜索URL
- 提供搜索优先级评估

## 支持的搜索场景

| 场景类型 | 特征 | 示例 |
|---------|------|------|
| 时效性 | 需要最新、即时信息 | "今天的热搜"、"最新的iPhone价格" |
| 天气 | 需要天气、气象信息 | "北京今天天气"、"明天会下雨吗" |
| 新闻资讯 | 需要新闻、热点事件 | "最近的AI新闻"、"今天头条" |
| 价格行情 | 需要实时价格数据 | "特斯拉股价"、"今日油价" |
| 实时数据 | 需要实时变化数据 | "比赛比分"、"交通状况" |
| 不需要联网 | 通用知识问题 | "什么是RAG"、"迪士尼在哪里" |

## 项目结构

```
16-Query+联网搜索/
├── README.md                   # 项目说明
└── code/
    ├── __init__.py            # 模块初始化
    ├── config.py              # 配置管理
    ├── query_classifier.py    # 查询分类器
    ├── query_rewriter.py      # 查询改写器
    ├── search_strategy.py     # 搜索策略生成器
    ├── pipeline.py            # 整合管道
    └── main.py                # 主程序入口
```

## 快速开始

### 1. 环境准备

确保根目录的 `.env` 文件包含：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

### 2. 运行演示

```bash
# 在根目录下运行
cd practice/16-Query+联网搜索/code
python main.py
```

## 核心 API

### WebSearchPipeline（推荐）

完整的处理管道，整合分类、改写、策略生成：

```python
from pipeline import WebSearchPipeline

# 初始化
pipeline = WebSearchPipeline()

# 完整处理
result = pipeline.process("今天北京的天气怎么样？")

# 输出结果
print(result.format_summary())

# 或者获取字典格式
print(result.to_dict())
```

返回结果包含：
- `original_query`: 原始查询
- `need_web_search`: 是否需要联网搜索
- `search_type`: 搜索类型
- `rewritten_query`: 改写后的查询
- `keywords`: 搜索关键词
- `platforms`: 推荐平台
- `search_operators`: 搜索操作符
- `search_urls`: 搜索URL

### 单独使用各组件

#### 1. 查询分类器

```python
from query_classifier import QueryClassifier

classifier = QueryClassifier()

# 完整分类
result = classifier.classify("今天北京的天气？")
# {"need_web_search": True, "search_type": "天气", "reason": "..."}

# 快速分类
search_type = classifier.quick_classify("今天的新闻")
# 返回 SearchNeedType 枚举
```

#### 2. 查询改写器

```python
from query_rewriter import WebQueryRewriter
from query_classifier import SearchNeedType

rewriter = WebQueryRewriter()

# 通用改写
result = rewriter.rewrite(
    query="今天北京的天气",
    search_type=SearchNeedType.WEATHER
)

# 特定类型改写
rewritten = rewriter.rewrite_for_weather("天气怎么样", location="北京")
rewritten = rewriter.rewrite_for_news("最新的AI动态")
rewritten = rewriter.rewrite_for_price("特斯拉股价")

# 提取关键词
keywords = rewriter.extract_keywords("iPhone 16 Pro Max 最新价格")

# 查询扩展
variants = rewriter.expand_query("今天的天气", SearchNeedType.WEATHER)
```

#### 3. 搜索策略生成器

```python
from search_strategy import SearchStrategyGenerator
from query_classifier import SearchNeedType

generator = SearchStrategyGenerator()

# 生成策略
strategy = generator.generate(
    query="北京天气 2026年2月13日",
    search_type=SearchNeedType.WEATHER,
    keywords=["北京", "天气", "今天"]
)

# 获取平台推荐
platforms = generator.get_platform_recommendation(SearchNeedType.NEWS)

# 生成搜索URL
url = generator.generate_search_url(
    query="北京天气",
    platform="百度"
)
```

### 批量处理

```python
pipeline = WebSearchPipeline()

queries = [
    "今天天气",
    "最新新闻", 
    "什么是机器学习"
]

results = pipeline.process_batch(queries)
for result in results:
    print(result.format_summary())
```

### 快速处理

```python
pipeline = WebSearchPipeline()

# 只返回关键信息
result = pipeline.quick_process("今天的新闻")
# {
#     "need_web_search": True,
#     "search_type": "新闻资讯",
#     "rewritten_query": "...",
#     "keywords": [...],
#     "platforms": [...]
# }
```

## 与RAG系统集成

```python
from pipeline import WebSearchPipeline

def enhanced_rag_query(user_query: str):
    """增强的RAG查询流程"""
    
    # 初始化管道
    pipeline = WebSearchPipeline()
    
    # 判断是否需要联网搜索
    result = pipeline.process(user_query)
    
    if result.need_web_search:
        # 需要联网搜索
        print(f"检测到需要联网搜索: {result.search_type}")
        print(f"改写后查询: {result.rewritten_query}")
        print(f"推荐平台: {result.platforms}")
        
        # 执行联网搜索（需要实现）
        search_results = web_search(
            query=result.rewritten_query,
            platforms=result.platforms
        )
        
        # 结合搜索结果生成答案
        answer = generate_answer_with_context(
            query=user_query,
            context=search_results
        )
    else:
        # 使用本地知识库
        answer = rag_query(user_query)
    
    return answer
```

## 支持的搜索平台

| 平台 | 适用场景 | 特点 |
|------|---------|------|
| Google | 全类型 | 全球最大搜索引擎 |
| 百度 | 时效性、新闻、天气 | 中文搜索优化 |
| 必应 | 时效性、新闻 | 学术搜索强 |
| 知乎 | 新闻、时效性 | 深度讨论 |
| 微博 | 新闻、实时数据 | 实时热点 |
| 微信公众号 | 新闻、行情 | 深度文章 |
| GitHub | 技术查询 | 开源项目 |
| Stack Overflow | 技术查询 | 技术问答 |
| 雪球 | 价格行情 | 股票讨论 |
| 东方财富 | 价格行情 | 财经数据 |
| 中国天气网 | 天气 | 官方数据 |

## 处理流程

```
用户查询
    ↓
┌─────────────────┐
│   查询分类器    │  → 判断是否需要联网搜索
│ QueryClassifier │  → 识别搜索类型
└────────┬────────┘
         │
    需要联网？
         │
    ┌────┴────┐
    │         │
   是        否
    │         │
    ↓         └→ 使用本地知识库
┌─────────────────┐
│   查询改写器    │  → 改写查询
│ QueryRewriter   │  → 提取关键词
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 搜索策略生成器  │  → 推荐平台
│StrategyGenerator│  → 生成操作符
└────────┬────────┘  → 生成搜索URL
         │
         ↓
    执行联网搜索
```

## 依赖

- Python >= 3.11
- openai >= 1.0.0（兼容DashScope API）
- python-dotenv >= 1.0.0
- loguru >= 0.7.0

## 配置说明

主要配置在 `code/config.py` 中：

```python
@dataclass
class Config:
    # LLM配置
    llm_model: str = "qwen-max"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # 搜索平台列表
    search_platforms: list = [...]
    
    # 各类型关键词配置
    time_sensitive_keywords: list = [...]
    weather_keywords: list = [...]
    news_keywords: list = [...]
    price_keywords: list = [...]
```

## 常见问题

### Q1: 如何添加新的搜索场景？

在 `query_classifier.py` 中添加新的 `SearchNeedType` 枚举值，并更新相关提示词。

### Q2: 如何自定义搜索平台？

在 `search_strategy.py` 的 `PLATFORM_FEATURES` 中添加新的平台配置。

### Q3: 如何调整分类准确性？

修改 `query_classifier.py` 中的提示词，增加更多判断规则和示例。

## 许可证

本项目遵循项目根目录下的LICENSE文件中定义的许可证。

## 更新日志

### v1.0.0 (2026-02-13)
- ✨ 完整的查询处理管道
- 🔍 支持6种搜索场景识别
- ✏️ 智能查询改写和关键词提取
- 🎯 搜索策略生成和URL构建
- 📚 与RAG系统集成示例
