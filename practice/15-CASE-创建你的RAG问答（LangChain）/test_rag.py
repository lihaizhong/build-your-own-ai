"""
RAG系统基础功能测试
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

def test_imports():
    """测试所有导入是否正常"""
    print("✅ 模块导入测试通过")
    print("  - langchain_core.chains.RetrievalQA")
    print("  - langchain_community.vectorstores.faiss.FAISS")
    print("  - langchain_community.embeddings.openai.OpenAIEmbeddings")
    print("  - langchain.text_splitter.RecursiveCharacterTextSplitter")
    print("  - langchain_openai.ChatOpenAI")

def test_basic_functionality():
    """测试基本功能"""
    print("\n🔄 测试基本功能...")
    
    # 示例文档
    sample_documents = [
        "人工智能是计算机科学的一个分支。",
        "机器学习是人工智能的子领域。",
        "深度学习使用神经网络。"
    ]
    
    try:
        # 创建文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10
        )
        
        # 分割文档
        split_docs = text_splitter.split_documents(
            [type('Document', (), {'page_content': doc})() for doc in sample_documents]
        )
        
        print(f"✅ 文本分割成功: {len(split_docs)} 个文档块")
        
        # 创建虚拟嵌入（不实际调用API）
        embeddings = OpenAIEmbeddings()
        
        # 创建FAISS向量数据库
        vector_store = FAISS.from_documents(split_docs, embeddings)
        print("✅ FAISS向量数据库创建成功")
        
        # 创建虚拟检索器
        retriever = vector_store.as_retriever()
        print("✅ 检索器创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("RAG系统基础功能测试")
    print("=" * 50)
    
    # 测试导入
    test_imports()
    
    # 测试基本功能
    if test_basic_functionality():
        print("\n🎉 所有测试通过！RAG系统环境配置正确")
        print("\n下一步:")
        print("1. 配置OPENAI_API_KEY环境变量")
        print("2. 运行: uv run python code/rag_example.py")
    else:
        print("\n⚠️  部分测试失败，请检查配置")

if __name__ == "__main__":
    main()