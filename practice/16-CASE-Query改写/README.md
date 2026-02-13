# Query改写系统

RAG检索增强的Query改写系统，通过LLM自动识别查询类型并进行改写，提升检索效果。

> 本项目使用根目录的 `pyproject.toml` 和 `.env`，无需单独配置环境。

## 支持的Query类型

| 类型 | 特征 | 示例 |
|------|------|------|
| 上下文依赖型 | 包含"还有"、"其他"、"另外"等需要上下文理解的词汇 | "还有其他景点吗？" |
| 对比型 | 包含"哪个"、"比较"、"更"、"区别"等比较词汇 | "Python和Java哪个更好？" |
| 模糊指代型 | 包含"它"、"他们"、"这个"、"那个"等指代词 | "它的主要优势是什么？" |
| 多意图型 | 包含多个独立问题 | "什么是RAG？如何搭建？有什么优缺点？" |
| 反问型 | 包含"不会"、"难道"、"不是吗"等反问语气 | "难道深度学习不需要大量数据吗？" |

## 项目结构

```
16-CASE-Query改写/
├── .gitignore
├── README.md
└── code/
    ├── __init__.py
    ├── query_rewriter.py   # Query改写核心实现
    └── main.py             # 演示程序
```

## 快速开始

### 1. 环境准备

```bash
# 在根目录配置环境变量
# 编辑 build-your-own-ai/.env 文件，确保有 DASHSCOPE_API_KEY
```

### 2. 运行演示

```bash
# 在根目录下运行
cd practice/16-CASE-Query改写/code
python main.py
```

## 核心API

### QueryRewriter 类

```python
from query_rewriter import QueryRewriter

# 初始化
rewriter = QueryRewriter(model="qwen-turbo-latest")

# 自动改写（推荐）
result = rewriter.auto_rewrite_query(
    query="还有其他景点吗？",
    conversation_history="用户：巴黎是法国的首都。\n助手：是的。",
    context_info=""
)

# 返回结果
# {
#     "original_query": "还有其他景点吗？",
#     "query_type": "上下文依赖型",
#     "rewritten_query": "巴黎除了埃菲尔铁塔和卢浮宫，还有哪些著名景点？",
#     "sub_queries": []
# }
```

### 单独使用各类型改写方法

```python
# 上下文依赖型
rewritten = rewriter.rewrite_context_dependent_query(query, conversation_history)

# 对比型
rewritten = rewriter.rewrite_comparative_query(query, context_info)

# 模糊指代型
rewritten = rewriter.rewrite_vague_reference_query(query, conversation_history)

# 多意图型（返回列表）
sub_queries = rewriter.rewrite_multi_intent_query(query)

# 反问型
rewritten = rewriter.rewrite_rhetorical_query(query)
```

## 与RAG系统集成

```python
from query_rewriter import QueryRewriter

def enhanced_retrieval(query: str, conversation_history: str = ""):
    """增强的RAG检索流程"""
    rewriter = QueryRewriter()
    
    # 1. Query改写
    result = rewriter.auto_rewrite_query(query, conversation_history)
    
    # 2. 根据改写结果进行检索
    if result['sub_queries']:
        # 多意图型：对每个子查询分别检索
        all_docs = []
        for sub_query in result['sub_queries']:
            docs = retrieve(sub_query)
            all_docs.extend(docs)
        return deduplicate(all_docs)
    else:
        # 其他类型：使用改写后的查询检索
        return retrieve(result['rewritten_query'])
```

## 依赖

- Python >= 3.11
- dashscope >= 1.24.2
- python-dotenv >= 1.0.0
- loguru >= 0.7.0