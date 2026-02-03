
"""
RAG问答系统示例
演示如何使用LangChain构建检索增强生成(RAG)系统
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.documents import Document

# 加载环境变量
load_dotenv()

# 配置OpenAI API密钥
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

# 初始化模型和嵌入
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY) # type: ignore
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # type: ignore

# 示例文档
sample_documents = [
    "人工智能是计算机科学的一个分支，致力于创造智能机器。",
    "机器学习是人工智能的子领域，通过让计算机从数据中自动学习。",
    "深度学习使用神经网络模拟人脑的学习过程。",
    "自然语言处理使计算机能够理解和生成人类语言。",
    "RAG（检索增强生成）结合了信息检索和文本生成技术。"
]

def create_rag_system():
    """创建RAG系统"""
    print("正在创建RAG系统...")
    
    # 文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 分割文档
    documents = [Document(page_content=doc) for doc in sample_documents]
    split_documents = text_splitter.split_documents(documents)
    
    # 创建向量数据库
    vector_store = FAISS.from_documents(split_documents, embeddings)
    
    # 创建检索问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    
    print("RAG系统创建完成！")
    return qa_chain

def test_rag_system(qa_chain):
    """测试RAG系统"""
    print("\n=== RAG系统测试 ===")
    
    test_questions = [
        "什么是人工智能？",
        "机器学习和深度学习有什么区别？",
        "RAG技术有什么优势？"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        result = qa_chain({"query": question})
        print(f"回答: {result['result']}")
        
        # 显示来源文档
        if result.get('source_documents'):
            print("来源文档:")
            for doc in result['source_documents'][:2]:  # 显示前2个来源
                print(f"  - {doc.page_content[:100]}...")

def main():
    """主函数"""
    print("=" * 50)
    print("RAG系统演示")
    print("=" * 50)
    
    try:
        # 创建RAG系统
        qa_chain = create_rag_system()
        
        # 测试系统
        test_rag_system(qa_chain)
        
        print("\n" + "=" * 50)
        print("RAG系统演示完成！")
        print("如需使用真实文档，请修改代码中的sample_documents部分")
        print("=" * 50)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        print("\n请确保:")
        print("1. 已设置OPENAI_API_KEY环境变量")
        print("2. 已安装所有依赖包")
        print("3. 网络连接正常")

if __name__ == "__main__":
    main()
