"""
向量数据库与元数据管理示例
演示如何使用 FAISS 和 ChromaDB 进行向量搜索
"""

import os
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb


def get_project_path(*paths: str) -> str:
    """
    获取项目路径的统一方法
    
    Args:
        *paths: 路径组件
        
    Returns:
        完整的项目路径
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        return os.path.join(project_dir, *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)


class VectorDatabaseDemo:
    """向量数据库演示类"""
    
    def __init__(self) -> None:
        """初始化演示类"""
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.chroma_client = chromadb.PersistentClient(
            path=get_project_path('model', 'chroma_db')
        )
        
    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        生成文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量数组
        """
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def create_collection(self, collection_name: str = "documents") -> None:
        """
        创建 ChromaDB 集合
        
        Args:
            collection_name: 集合名称
        """
        # 删除已存在的集合
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        
        # 创建新集合
        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"集合 '{collection_name}' 创建成功")
        return collection
    
    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str]
    ) -> None:
        """
        添加文档到集合
        
        Args:
            collection_name: 集合名称
            documents: 文档列表
            metadatas: 元数据列表
            ids: 文档ID列表
        """
        collection = self.chroma_client.get_collection(collection_name)
        
        # 生成嵌入向量
        embeddings = self.generate_embeddings(documents)
        
        # 添加文档
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"成功添加 {len(documents)} 个文档到集合 '{collection_name}'")
    
    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 3
    ) -> dict:
        """
        搜索相似文档
        
        Args:
            collection_name: 集合名称
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            搜索结果
        """
        collection = self.chroma_client.get_collection(collection_name)
        
        # 生成查询嵌入
        query_embedding = self.model.encode([query])
        
        # 搜索
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        return results


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("向量数据库与元数据管理演示")
    print("=" * 60)
    
    # 创建演示实例
    demo = VectorDatabaseDemo()
    
    # 示例文档
    documents = [
        "三国演义是中国古典四大名著之一，描写了东汉末年到西晋初年的历史。",
        "曹操是三国时期的重要人物，被称为'治世之能臣，乱世之奸雄'。",
        "刘备是三国时期蜀汉的开国皇帝，以仁德著称。",
        "诸葛亮是三国时期著名的政治家、军事家，被誉为'卧龙'。",
        "孙权是三国时期东吴的建立者，继承父兄基业，据守江东。",
    ]
    
    # 元数据
    metadatas = [
        {"category": "文学", "year": "明代"},
        {"category": "人物", "period": "三国"},
        {"category": "人物", "period": "三国"},
        {"category": "人物", "period": "三国"},
        {"category": "人物", "period": "三国"},
    ]
    
    # 文档ID
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # 创建集合
    collection = demo.create_collection("sanguo_documents")
    
    # 添加文档
    demo.add_documents(
        collection_name="sanguo_documents",
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    # 搜索示例
    print("\n" + "=" * 60)
    print("搜索示例")
    print("=" * 60)
    
    queries = [
        "谁是三国时期的著名人物？",
        "诸葛亮有什么特点？",
        "三国演义讲的是什么？"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        results = demo.search("sanguo_documents", query, n_results=2)
        
        for i, (doc, metadata, distance) in enumerate(
            zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ),
            1
        ):
            print(f"  {i}. {doc}")
            print(f"     元数据: {metadata}")
            print(f"     相似度: {1 - distance:.4f}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()