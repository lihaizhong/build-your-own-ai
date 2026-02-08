
"""
完整的RAG问答系统
基于PDF文档的智能问答系统，支持页码显示和DeepSeek集成
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores.faiss import FAISS
# 使用自定义的RAG实现，不依赖可能已过时的RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 导入自定义模块
from .pdf_processor import process_pdf_with_page_numbers
from .deepseek_integration import get_dashscope_llm, get_embeddings, get_integration_info


# 加载环境变量
load_dotenv(verbose=True)

# PDF文件路径
PDF_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "AI产品经理面试题65道.pdf")


def create_complete_rag_system(pdf_path: str = PDF_FILE_PATH):
    """
    创建完整的RAG系统
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        完整的RAG系统实例
    """
    print("=" * 60)
    print("创建完整的RAG问答系统")
    print("=" * 60)
    
    # 1. 加载PDF文档并处理页码
    print("\n1. 加载PDF文档并处理页码...")
    try:
        documents = process_pdf_with_page_numbers(pdf_path)
        print(f"成功加载 {len(documents)} 个文档块")
        
        # 显示文档统计信息
        from .pdf_processor import PDFProcessor
        processor = PDFProcessor(pdf_path)
        stats = processor.get_document_stats()
        print("文档统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"加载PDF文档失败: {str(e)}")
        raise
    
    # 2. 文本分割
    print("\n2. 分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_documents = text_splitter.split_documents(documents)
    print(f"分割后得到 {len(split_documents)} 个文档块")
    
    # 3. 创建向量数据库
    print("\n3. 创建向量数据库...")
    try:
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(split_documents, embeddings)
        print("向量数据库创建成功")
    except Exception as e:
        print(f"创建向量数据库失败: {str(e)}")
        raise
    
    # 4. 初始化DashScope LLM
    print("\n4. 初始化DashScope LLM...")
    try:
        llm = get_dashscope_llm()
        print("DashScope LLM初始化成功")
    except Exception as e:
        print(f"初始化DashScope LLM失败: {str(e)}")
        raise
    
    # 5. 创建检索器
    print("\n5. 创建检索器...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # 创建RAG系统字典
    rag_system = {
        'llm': llm,
        'retriever': retriever,
        'vector_store': vector_store
    }
    
    print("RAG系统创建完成！")
    return rag_system


def format_source_documents(source_docs):
    """
    格式化源文档信息，包含页码
    
    Args:
        source_docs: 源文档列表
        
    Returns:
        格式化的文档信息列表
    """
    formatted_sources = []
    for doc in source_docs:
        page_num = doc.metadata.get('page_number', '未知')
        source = doc.metadata.get('source', '未知')
        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        
        formatted_sources.append({
            'page': page_num,
            'source': source,
            'content': content
        })
    
    return formatted_sources


def query_rag_system(rag_system, question):
    """
    查询RAG系统
    
    Args:
        rag_system: RAG系统字典
        question: 用户问题
        
    Returns:
        包含回答和来源文档的字典
    """
    # 检索相关文档
    print(f"正在检索与'{question}'相关的文档...")
    source_documents = rag_system['retriever'].get_relevant_documents(question)
    
    if not source_documents:
        return {
            'result': '未找到相关文档，请尝试其他问题。',
            'source_documents': []
        }
    
    # 组合检索到的文档内容
    context = "\n\n".join([doc.page_content for doc in source_documents])
    
    # 构建提示词
    prompt = f"""基于以下文档内容回答用户问题。如果文档中没有相关信息，请如实说明。

文档内容:
{context}

用户问题: {question}

请用中文回答，并在回答后列出相关的文档页码信息。"""
    
    # 使用LLM生成回答
    print("正在生成回答...")
    response = rag_system['llm'].invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    
    return {
        'result': answer,
        'source_documents': source_documents
    }


def display_answer_with_sources(result, question):
    """
    显示问答结果和来源信息
    
    Args:
        result: RAG系统返回的结果
        question: 用户问题
    """
    print("\n" + "=" * 60)
    print(f"问题: {question}")
    print("=" * 60)
    print(f"回答: {result['result']}")
    
    # 显示来源文档
    if result.get('source_documents'):
        print("\n来源文档:")
        sources = format_source_documents(result['source_documents'])
        for i, source in enumerate(sources, 1):
            print(f"\n{i}. 页码: {source['page']}, 来源: {source['source']}")
            print(f"   内容: {source['content']}")
    else:
        print("\n未找到相关文档")


def interactive_rag_system():
    """
    交互式RAG问答系统
    """
    try:
        # 创建RAG系统
        rag_system = create_complete_rag_system()
        
        print("\n" + "=" * 60)
        print("RAG问答系统已就绪！")
        print("=" * 60)
        print("提示: 输入 'quit' 或 'exit' 退出系统")
        print("-" * 60)
        
        while True:
            try:
                # 获取用户输入
                question = input("\n请输入您的问题: ").strip()
                
                # 检查退出命令
                if question.lower() in ['quit', 'exit', 'q']:
                    print("感谢使用RAG问答系统，再见！")
                    break
                
                # 空输入处理
                if not question:
                    print("请输入有效的问题")
                    continue
                
                # 执行查询
                result = query_rag_system(rag_system, question)
                
                # 显示结果
                display_answer_with_sources(result, question)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断，正在退出...")
                break
            except Exception as e:
                print(f"查询时出错: {str(e)}")
                continue
                
    except Exception as e:
        print(f"启动RAG系统失败: {str(e)}")
        print("\n请确保:")
        print("1. 已正确配置.env文件中的API密钥")
        print("2. PDF文件存在且可访问")
        print("3. 网络连接正常")


def test_rag_system():
    """
    测试RAG系统功能
    """
    print("=" * 60)
    print("RAG系统功能测试")
    print("=" * 60)
    
    try:
        # 创建RAG系统
        rag_system = create_complete_rag_system()
        
        # 测试问题
        test_questions = [
            "产品经理需要具备哪些核心能力？",
            "如何进行用户需求分析？",
            "产品设计的基本原则是什么？"
        ]
        
        print("\n开始测试问答功能...")
        for i, question in enumerate(test_questions, 1):
            print(f"\n测试 {i}: {question}")
            try:
                result = query_rag_system(rag_system, question)
                display_answer_with_sources(result, question)
            except Exception as e:
                print(f"测试失败: {str(e)}")
        
        print("\n" + "=" * 60)
        print("RAG系统测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试失败: {str(e)}")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("AI产品经理面试题RAG问答系统")
    print("=" * 60)
    
    # 显示集成信息
    integration_info = get_integration_info()
    print("\nAPI集成配置:")
    for key, value in integration_info.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    # 检查PDF文件是否存在
    if not os.path.exists(PDF_FILE_PATH):
        print(f"\n错误: PDF文件不存在: {PDF_FILE_PATH}")
        print("请确保AI产品经理面试题65道.pdf文件存在于data目录中")
        return
    
    # 询问用户选择运行模式
    print("\n请选择运行模式:")
    print("1. 交互式问答模式")
    print("2. 自动测试模式")
    print("3. 退出")
    
    try:
        choice = input("\n请输入选择 (1/2/3): ").strip()
        
        if choice == "1":
            interactive_rag_system()
        elif choice == "2":
            test_rag_system()
        elif choice == "3":
            print("退出程序")
        else:
            print("无效选择，使用默认交互模式")
            interactive_rag_system()
            
    except KeyboardInterrupt:
        print("\n\n程序被用户中断，退出")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")


if __name__ == "__main__":
    main()
