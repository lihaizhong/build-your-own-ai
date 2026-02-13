"""
Query改写演示程序
演示5种Query类型的改写效果
"""

from query_rewriter import QueryRewriter, QueryType


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
    else:
        print("-" * 60)


def demo_context_dependent():
    """演示上下文依赖型Query改写"""
    print_separator("上下文依赖型 Query 改写")

    rewriter = QueryRewriter()

    # 示例1
    conversation_history = "用户：巴黎是法国的首都，有很多著名景点。\n助手：是的，巴黎有埃菲尔铁塔、卢浮宫等著名景点。"
    query = "还有其他景点吗？"

    print(f"对话历史：\n{conversation_history}")
    print(f"\n原查询：{query}")

    result = rewriter.auto_rewrite_query(query, conversation_history)
    print(f"查询类型：{result['query_type']}")
    print(f"改写后：{result['rewritten_query']}")


def demo_comparative():
    """演示对比型Query改写"""
    print_separator("对比型 Query 改写")

    rewriter = QueryRewriter()

    query = "Python和Java哪个更适合做机器学习？"
    context_info = "Python和Java都是流行的编程语言，Python在数据科学领域应用广泛，Java在企业级应用中占主导地位。"

    print(f"原查询：{query}")
    print(f"上下文信息：{context_info}")

    result = rewriter.auto_rewrite_query(query, context_info=context_info)
    print(f"查询类型：{result['query_type']}")
    print(f"改写后：{result['rewritten_query']}")


def demo_vague_reference():
    """演示模糊指代型Query改写"""
    print_separator("模糊指代型 Query 改写")

    rewriter = QueryRewriter()

    conversation_history = "用户：我最近在研究Transformer模型，它比RNN效果好很多。\n助手：Transformer确实在NLP领域表现出色。"
    query = "它的主要优势是什么？"

    print(f"对话历史：\n{conversation_history}")
    print(f"\n原查询：{query}")

    result = rewriter.auto_rewrite_query(query, conversation_history)
    print(f"查询类型：{result['query_type']}")
    print(f"改写后：{result['rewritten_query']}")


def demo_multi_intent():
    """演示多意图型Query改写"""
    print_separator("多意图型 Query 改写")

    rewriter = QueryRewriter()

    query = "RAG系统如何搭建，需要哪些组件，有什么优缺点？"

    print(f"原查询：{query}")

    result = rewriter.auto_rewrite_query(query)
    print(f"查询类型：{result['query_type']}")
    print(f"拆分后的子查询：")
    for i, sub_query in enumerate(result['sub_queries'], 1):
        print(f"  {i}. {sub_query}")


def demo_rhetorical():
    """演示反问型Query改写"""
    print_separator("反问型 Query 改写")

    rewriter = QueryRewriter()

    query = "难道深度学习不需要大量数据吗？"

    print(f"原查询：{query}")

    result = rewriter.auto_rewrite_query(query)
    print(f"查询类型：{result['query_type']}")
    print(f"改写后：{result['rewritten_query']}")


def demo_normal():
    """演示普通型Query（无需改写）"""
    print_separator("普通型 Query（无需改写）")

    rewriter = QueryRewriter()

    query = "什么是向量数据库？"

    print(f"原查询：{query}")

    result = rewriter.auto_rewrite_query(query)
    print(f"查询类型：{result['query_type']}")
    print(f"改写后：{result['rewritten_query']}")
    print("(普通型查询无需改写，直接使用原查询)")


def interactive_demo():
    """交互式演示"""
    print_separator("交互式 Query 改写演示")

    rewriter = QueryRewriter()

    print("请输入您的查询（输入 'quit' 退出）：")

    while True:
        print()
        query = input("查询: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("退出演示。")
            break

        if not query:
            continue

        # 可选：输入对话历史
        print("是否需要提供对话历史？(y/n，默认n): ", end="")
        need_history = input().strip().lower() == 'y'

        conversation_history = ""
        if need_history:
            print("请输入对话历史（多行输入，以空行结束）：")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            conversation_history = "\n".join(lines)

        # 执行改写
        result = rewriter.auto_rewrite_query(query, conversation_history)

        print()
        print(f"查询类型: {result['query_type']}")
        if result['sub_queries']:
            print("拆分后的子查询:")
            for i, sub_query in enumerate(result['sub_queries'], 1):
                print(f"  {i}. {sub_query}")
        else:
            print(f"改写后: {result['rewritten_query']}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print(" Query改写系统演示")
    print(" 支持5种Query类型：上下文依赖型、对比型、模糊指代型、多意图型、反问型")
    print("=" * 60)

    # 运行各类型演示
    demo_context_dependent()
    demo_comparative()
    demo_vague_reference()
    demo_multi_intent()
    demo_rhetorical()
    demo_normal()

    # 交互式演示
    print_separator()
    print("以上是各类型Query改写的示例演示")
    print("接下来可以进入交互式演示模式")
    print()

    start_interactive = input("是否开始交互式演示？(y/n): ").strip().lower()
    if start_interactive == 'y':
        interactive_demo()

    print("\n演示结束！")


if __name__ == "__main__":
    main()
