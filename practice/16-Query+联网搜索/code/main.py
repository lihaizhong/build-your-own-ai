"""
Query+联网搜索改写系统 - 主程序入口
演示完整的处理流程
"""

import sys
import json
from pathlib import Path
from loguru import logger

# 添加路径支持直接运行
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .pipeline import WebSearchPipeline, PipelineResult
    from .config import config
except ImportError:
    from pipeline import WebSearchPipeline, PipelineResult
    from config import config


def setup_logger():
    """配置日志系统"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )


def print_logo():
    """打印Logo"""
    logo = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║        Query + 联网搜索改写系统 v1.0.0                   ║
║                                                          ║
║     自动判断搜索需求 · 智能改写查询 · 生成搜索策略       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(logo)


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
    else:
        print("-" * 60)


def demo_basic():
    """基础功能演示"""
    print_separator("基础功能演示")

    pipeline = WebSearchPipeline()

    # 测试不同类型的查询
    test_queries = [
        ("今天北京的天气怎么样？", "天气查询"),
        ("最新的iPhone 16 Pro价格是多少？", "时效性查询"),
        ("什么是RAG技术？", "通用知识查询"),
        ("最近有什么AI领域的重大新闻？", "新闻资讯查询"),
        ("特斯拉今天的股价是多少？", "价格行情查询"),
        ("迪士尼乐园在哪里？", "静态信息查询"),
    ]

    for query, desc in test_queries:
        print(f"\n【{desc}】")
        print(f"查询: {query}")

        result = pipeline.quick_process(query)

        if result["need_web_search"]:
            print(f"✓ 需要联网搜索")
            print(f"  类型: {result['search_type']}")
            print(f"  改写: {result['rewritten_query']}")
            print(f"  关键词: {', '.join(result['keywords'])}")
            print(f"  平台: {', '.join(result['platforms'])}")
        else:
            print(f"✗ 不需要联网搜索")
            print(f"  类型: {result['search_type']}")


def demo_full_pipeline():
    """完整流程演示"""
    print_separator("完整处理流程演示")

    pipeline = WebSearchPipeline()

    # 测试查询
    query = "今天上海的天气怎么样，适合户外活动吗？"

    print(f"原始查询: {query}")
    print("\n正在处理...")

    # 执行完整流程
    result = pipeline.process(query)

    # 显示详细结果
    print(result.format_summary())


def demo_batch_processing():
    """批量处理演示"""
    print_separator("批量处理演示")

    pipeline = WebSearchPipeline()

    queries = [
        "今天天气",
        "最新新闻",
        "股价行情",
        "什么是机器学习",
        "最近的科技动态"
    ]

    print(f"批量处理 {len(queries)} 个查询...\n")

    results = pipeline.process_batch(queries)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.original_query}")
        print(f"   需要: {'联网搜索' if result.need_web_search else '本地知识库'}")
        print(f"   类型: {result.search_type}")
        if result.rewritten_query:
            print(f"   改写: {result.rewritten_query}")
        print()


def demo_search_scenarios():
    """搜索场景演示"""
    print_separator("搜索场景演示")

    pipeline = WebSearchPipeline()

    scenarios = {
        "时效性场景": [
            "今天的热搜榜是什么？",
            "最近上映的电影有哪些？",
            "今年的诺贝尔奖得主是谁？"
        ],
        "天气场景": [
            "北京今天的天气？",
            "明天上海会下雨吗？",
            "深圳的空气质量怎么样？"
        ],
        "新闻资讯场景": [
            "最新的AI新闻",
            "最近有什么重大事件？",
            "今天的头条新闻是什么？"
        ],
        "价格行情场景": [
            "苹果公司的股价",
            "今天的油价是多少？",
            "比特币现在的价格"
        ]
    }

    for scenario_name, queries in scenarios.items():
        print(f"\n【{scenario_name}】")

        for query in queries:
            result = pipeline.classify_only(query)

            icon = "🌐" if result["need_web_search"] else "📚"
            print(f"  {icon} {query}")
            print(f"     → {result['search_type']}")


def interactive_mode():
    """交互模式"""
    print_separator("交互模式")
    print("输入查询进行测试，输入 'quit' 或 'exit' 退出")
    print("输入 'json' 查看JSON格式输出\n")

    pipeline = WebSearchPipeline()

    while True:
        try:
            query = input("请输入查询: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("退出交互模式。")
                break

            if query.lower() == 'json':
                print("当前为JSON输出模式，输入查询查看JSON结果")
                json_mode = True
                continue

            # 处理查询
            result = pipeline.process(query)

            # 输出结果
            if query.lower() == 'json':
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(result.format_summary())

        except KeyboardInterrupt:
            print("\n\n程序被中断，正在退出...")
            break
        except EOFError:
            print("\n\n检测到EOF，正在退出...")
            break
        except Exception as e:
            logger.error(f"处理失败: {e}")
            print(f"\n抱歉，处理查询时出现错误: {e}")


def demo_integration_with_rag():
    """演示与RAG系统集成"""
    print_separator("与RAG系统集成示例")

    print("""
以下是如何将本系统集成到RAG流程中的示例代码：

```python
from pipeline import WebSearchPipeline

# 初始化管道
pipeline = WebSearchPipeline()

def enhanced_rag_query(user_query: str):
    '''增强的RAG查询流程'''
    
    # Step 1: 判断是否需要联网搜索
    result = pipeline.process(user_query)
    
    if result.need_web_search:
        # 需要联网搜索
        print(f"检测到需要联网搜索: {result.search_type}")
        print(f"改写后查询: {result.rewritten_query}")
        print(f"推荐平台: {result.platforms}")
        
        # 执行联网搜索
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
        print("使用本地知识库回答")
        answer = rag_query(user_query)
    
    return answer
```
""")


def main():
    """主函数"""
    # 配置日志
    setup_logger()

    # 打印Logo
    print_logo()

    # 运行各演示
    demo_basic()
    demo_full_pipeline()
    demo_search_scenarios()
    demo_batch_processing()
    demo_integration_with_rag()

    # 交互模式
    print_separator()
    print("演示完成！")
    print()
    start_interactive = input("是否进入交互模式？(y/n): ").strip().lower()
    if start_interactive == 'y':
        interactive_mode()

    print("\n感谢使用 Query+联网搜索改写系统！")


if __name__ == "__main__":
    main()
