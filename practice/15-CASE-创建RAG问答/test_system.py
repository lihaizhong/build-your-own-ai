"""
简单的RAG系统测试脚本
"""

import sys
import os

# 添加code目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from pdf_processor import PDFProcessor # type: ignore
from deepseek_integration import get_integration_info, get_embeddings, get_dashscope_llm # type: ignore


def test_pdf_processing():
    """测试PDF处理"""
    print("=" * 60)
    print("测试PDF处理功能")
    print("=" * 60)
    
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "data/AI产品经理面试题65道.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"错误: PDF文件不存在: {pdf_path}")
        return False
    
    try:
        processor = PDFProcessor(pdf_path)
        stats = processor.get_document_stats()
        print("PDF文档统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        documents = processor.load_and_process()
        print(f"\n成功处理 {len(documents)} 个文档块")
        
        # 显示前3个文档的页码信息
        print("\n前3个文档示例:")
        for i, doc in enumerate(documents[:3], 1):
            page_num = doc.metadata.get('page_number', '未知')
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"{i}. 页码: {page_num}, 内容预览: {content_preview}")
        
        return True
        
    except Exception as e:
        print(f"PDF处理测试失败: {str(e)}")
        return False


def test_api_integration():
    """测试API集成"""
    print("\n" + "=" * 60)
    print("测试API集成功能")
    print("=" * 60)
    
    try:
        info = get_integration_info()
        print("API配置信息:")
        for key, value in info.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")
        
        # 测试嵌入模型
        print("\n测试DashScope嵌入模型...")
        embeddings = get_embeddings()
        test_text = "这是一个测试文本"
        test_embedding = embeddings.embed_query(test_text)
        print(f"嵌入模型测试成功，生成的嵌入向量维度: {len(test_embedding)}")
        
        # 测试LLM（简单的调用）
        print("\n测试DashScope LLM模型...")
        llm = get_dashscope_llm()
        test_prompt = "你好，请简要介绍一下自己"
        test_response = llm.invoke(test_prompt)
        response_text = test_response.content if hasattr(test_response, 'content') else str(test_response)
        print(f"LLM模型测试成功，响应: {response_text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"API集成测试失败: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("RAG系统功能测试")
    print("=" * 60)
    
    results = {
        'PDF处理': test_pdf_processing(),
        'API集成': test_api_integration()
    }
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！RAG系统已准备就绪。")
        print("\n您可以运行以下命令启动完整的RAG系统:")
        print("  uv run python code/rag_example.py")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复问题。")
    
    print("=" * 60)


if __name__ == "__main__":
    main()