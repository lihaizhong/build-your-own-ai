"""
检索层模块
Step3: 混合检索和关键词触发
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import numpy as np

from .config import config
from .embedding import VectorStore, TextEmbeddingModel, ImageEmbeddingModel


@dataclass
class RetrievalResult:
    """检索结果数据类"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]
    result_type: str = "text"  # "text" or "image"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
            "result_type": self.result_type
        }


class TextRetriever:
    """文本检索器"""
    
    def __init__(self, vector_store: VectorStore):
        """
        初始化文本检索器
        
        Args:
            vector_store: 向量存储对象
        """
        self.vector_store = vector_store
        self.text_index = vector_store.text_index
        self.text_embedding_model = vector_store.text_embedding_model
        
        if self.text_embedding_model is None:
            self.text_embedding_model = TextEmbeddingModel()
            vector_store.text_embedding_model = self.text_embedding_model
        
        logger.info("文本检索器初始化完成")
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        检索相关文本
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 分数阈值
        
        Returns:
            检索结果列表
        """
        if self.text_index is None or self.text_index.size == 0:
            logger.warning("文本索引为空，无法检索")
            return []
        
        top_k = top_k or config.top_k
        score_threshold = score_threshold or config.score_threshold
        
        logger.info(f"文本检索: '{query}' (top_k={top_k})")
        
        # 生成查询向量
        query_vector = self.text_embedding_model.embed_text(query) # type: ignore
        query_vector_np = np.array(query_vector, dtype=np.float32)
        
        # 搜索相似向量
        results = self.text_index.search(query_vector_np, k=top_k)
        
        # 构建检索结果
        retrieval_results = []
        for idx, distance in results:
            chunk = self.text_index.documents[idx]
            
            # 计算相似度分数（距离越小，相似度越高）
            # L2距离转相似度: 1 / (1 + distance)
            similarity = 1.0 / (1.0 + distance)
            
            if similarity >= score_threshold:
                result = RetrievalResult(
                    content=chunk.text,
                    source=chunk.source,
                    score=similarity,
                    metadata=chunk.metadata,
                    result_type="text"
                )
                retrieval_results.append(result)
        
        logger.info(f"文本检索完成，返回 {len(retrieval_results)} 个结果")
        
        return retrieval_results


class ImageRetriever:
    """图像检索器"""
    
    def __init__(self, vector_store: VectorStore):
        """
        初始化图像检索器
        
        Args:
            vector_store: 向量存储对象
        """
        self.vector_store = vector_store
        self.image_index = vector_store.image_index
        self.image_embedding_model = vector_store.image_embedding_model
        
        if self.image_embedding_model is None:
            self.image_embedding_model = ImageEmbeddingModel()
            vector_store.image_embedding_model = self.image_embedding_model
        
        logger.info("图像检索器初始化完成")
    
    def retrieve_by_text(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        通过文本查询检索图像
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 分数阈值
        
        Returns:
            检索结果列表
        """
        if self.image_index is None or self.image_index.size == 0:
            logger.warning("图像索引为空，无法检索")
            return []
        
        top_k = top_k or config.top_k
        score_threshold = score_threshold or config.score_threshold
        
        logger.info(f"图像文本检索: '{query}' (top_k={top_k})")
        
        # 使用CLIP文本编码器生成查询向量
        query_vector = self.image_embedding_model.embed_text_for_image_search(query) # type: ignore
        query_vector_np = np.array(query_vector, dtype=np.float32)
        
        # 搜索相似向量
        results = self.image_index.search(query_vector_np, k=top_k)
        
        # 构建检索结果
        retrieval_results = []
        for idx, distance in results:
            image_data = self.image_index.documents[idx]
            
            # 计算相似度分数
            similarity = 1.0 / (1.0 + distance)
            
            if similarity >= score_threshold:
                # 使用OCR文本作为内容描述
                content = image_data.ocr_text or f"图像: {image_data.image_path.name}"
                
                result = RetrievalResult(
                    content=content,
                    source=str(image_data.image_path),
                    score=similarity,
                    metadata=image_data.metadata,
                    result_type="image"
                )
                retrieval_results.append(result)
        
        logger.info(f"图像检索完成，返回 {len(retrieval_results)} 个结果")
        
        return retrieval_results
    
    def retrieve_by_image(
        self, 
        image_path: str, 
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        通过图像查询检索相似图像
        
        Args:
            image_path: 查询图像路径
            top_k: 返回结果数量
            score_threshold: 分数阈值
        
        Returns:
            检索结果列表
        """
        if self.image_index is None or self.image_index.size == 0:
            logger.warning("图像索引为空，无法检索")
            return []
        
        top_k = top_k or config.top_k
        score_threshold = score_threshold or config.score_threshold
        
        logger.info(f"图像相似检索: {image_path} (top_k={top_k})")
        
        # 生成查询图像向量
        from pathlib import Path
        query_vector = self.image_embedding_model.embed_image_path(Path(image_path)) # type: ignore
        query_vector_np = np.array(query_vector, dtype=np.float32)
        
        # 搜索相似向量
        results = self.image_index.search(query_vector_np, k=top_k)
        
        # 构建检索结果
        retrieval_results = []
        for idx, distance in results:
            image_data = self.image_index.documents[idx]
            
            # 计算相似度分数
            similarity = 1.0 / (1.0 + distance)
            
            if similarity >= score_threshold:
                content = image_data.ocr_text or f"图像: {image_data.image_path.name}"
                
                result = RetrievalResult(
                    content=content,
                    source=str(image_data.image_path),
                    score=similarity,
                    metadata=image_data.metadata,
                    result_type="image"
                )
                retrieval_results.append(result)
        
        logger.info(f"图像相似检索完成，返回 {len(retrieval_results)} 个结果")
        
        return retrieval_results


class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, vector_store: VectorStore):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储对象
        """
        self.vector_store = vector_store
        self.text_retriever = TextRetriever(vector_store)
        self.image_retriever = ImageRetriever(vector_store)
        
        # 检测关键词
        self.image_keywords = config.image_keywords
        
        logger.info("混合检索器初始化完成")
    
    def _should_trigger_image_search(self, query: str) -> bool:
        """
        检测是否应该触发图像检索
        
        Args:
            query: 查询文本
        
        Returns:
            是否触发图像检索
        """
        query_lower = query.lower()
        for keyword in self.image_keywords: # type: ignore
            if keyword.lower() in query_lower:
                return True
        return False
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        force_image_search: bool = False
    ) -> Dict[str, List[RetrievalResult]]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 分数阈值
            force_image_search: 强制图像检索
        
        Returns:
            {"text": 文本结果列表, "image": 图像结果列表}
        """
        results = {
            "text": [],
            "image": []
        }
        
        # 文本检索（始终执行）
        text_results = self.text_retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )
        results["text"] = text_results
        
        # 图像检索（关键词触发或强制）
        if force_image_search or self._should_trigger_image_search(query):
            logger.info(f"触发图像检索: '{query}'")
            image_results = self.image_retriever.retrieve_by_text(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold
            )
            results["image"] = image_results
        
        logger.info(
            f"混合检索完成: 文本结果 {len(results['text'])} 个, "
            f"图像结果 {len(results['image'])} 个"
        )
        
        return results
    
    def retrieve_unified(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        统一检索，返回混合结果
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 分数阈值
        
        Returns:
            混合检索结果列表
        """
        results = self.retrieve(query, top_k, score_threshold)
        
        # 合并文本和图像结果
        unified_results = results["text"] + results["image"]
        
        # 按分数排序
        unified_results.sort(key=lambda x: x.score, reverse=True)
        
        # 限制返回数量
        top_k = top_k or config.top_k
        unified_results = unified_results[:top_k]
        
        return unified_results
    
    def format_results(
        self, 
        results: Dict[str, List[RetrievalResult]]
    ) -> str:
        """
        格式化检索结果
        
        Args:
            results: 检索结果字典
        
        Returns:
            格式化的结果字符串
        """
        formatted = []
        
        if results["text"]:
            formatted.append("【文本检索结果】")
            for i, result in enumerate(results["text"], 1):
                formatted.append(
                    f"{i}. {result.content[:100]}...\n"
                    f"   来源: {result.source}\n"
                    f"   相似度: {result.score:.3f}"
                )
        
        if results["image"]:
            formatted.append("\n【图像检索结果】")
            for i, result in enumerate(results["image"], 1):
                formatted.append(
                    f"{i}. {result.content[:100]}...\n"
                    f"   来源: {result.source}\n"
                    f"   相似度: {result.score:.3f}"
                )
        
        return "\n".join(formatted)


if __name__ == "__main__":
    # 测试代码
    from loguru import logger
    from .config import load_env_config
    
    logger.add("logs/retrieval.log", rotation="1 day")
    
    # 加载环境变量
    load_env_config()
    
    # 加载索引
    vector_store = VectorStore()
    vector_store.load_indexes()
    
    # 创建混合检索器
    retriever = HybridRetriever(vector_store)
    
    # 测试查询
    queries = [
        "迪士尼有哪些经典动画电影？",
        "展示一下迪士尼的海报",
        "迪士尼乐园的门票价格是多少？"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print(f"{'='*60}")
        
        results = retriever.retrieve(query)
        print(retriever.format_results(results))
