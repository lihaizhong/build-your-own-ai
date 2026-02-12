"""
向量化层模块
Step2: 文本Embedding、图像Embedding和FAISS索引系统
"""

import os
from pathlib import Path
from typing import List, Any, Optional, Tuple
from loguru import logger
import numpy as np
import faiss
import dashscope
from dashscope import TextEmbedding
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

try:
    from .config import config
    from .data_processor import TextChunk, ImageData
    from .utils import save_pickle, load_pickle
except ImportError:
    from config import config
    from data_processor import TextChunk, ImageData
    from utils import save_pickle, load_pickle


class TextEmbeddingModel:
    """文本Embedding模型"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化文本Embedding模型
        
        Args:
            api_key: DashScope API密钥
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if self.api_key:
            dashscope.api_key = self.api_key
        
        self.model_name = config.text_embedding_model
        self.embedding_dim = config.text_embedding_dim
        logger.info(f"文本Embedding模型初始化: {self.model_name} (维度: {self.embedding_dim})")
    
    def embed_text(self, text: str) -> List[float]:
        """
        对单个文本进行向量化
        
        Args:
            text: 输入文本
        
        Returns:
            文本向量
        """
        if not text or not text.strip():
            return [0.0] * self.embedding_dim
        
        try:
            resp = TextEmbedding.call(
                model=self.model_name,
                input=text,
                text_type="document"
            )
            
            if resp.status_code == 200:
                embedding = resp.output['embeddings'][0]['embedding']
                return embedding
            else:
                logger.error(f"文本Embedding失败: {resp.message}")
                return [0.0] * self.embedding_dim
                
        except Exception as e:
            logger.error(f"文本Embedding异常: {e}")
            return [0.0] * self.embedding_dim
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量对文本进行向量化
        
        Args:
            texts: 输入文本列表
        
        Returns:
            文本向量列表
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_text_chunks(self, chunks: List[TextChunk]) -> Tuple[np.ndarray, List[TextChunk]]:
        """
        对文本块进行向量化
        
        Args:
            chunks: 文本块列表
        
        Returns:
            (向量数组, 文本块列表)
        """
        logger.info(f"开始对 {len(chunks)} 个文本块进行向量化...")
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        # 过滤掉无效的embedding
        valid_indices = []
        valid_embeddings = []
        valid_chunks = []
        
        for i, embedding in enumerate(embeddings):
            if any(v != 0.0 for v in embedding):
                valid_indices.append(i)
                valid_embeddings.append(embedding)
                valid_chunks.append(chunks[i])
        
        logger.info(f"成功向量化 {len(valid_embeddings)}/{len(chunks)} 个文本块")
        
        return np.array(valid_embeddings, dtype=np.float32), valid_chunks


class ImageEmbeddingModel:
    """图像Embedding模型（CLIP）"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化图像Embedding模型
        
        Args:
            model_name: CLIP模型名称
        """
        self.model_name = model_name or config.clip_model_name
        self.embedding_dim = config.image_embedding_dim
        
        logger.info(f"加载CLIP模型: {self.model_name}")
        
        # 加载CLIP模型和处理器
        self.model = CLIPModel.from_pretrained(
            f"openai/{self.model_name}"
        )
        self.processor = CLIPProcessor.from_pretrained(
            f"openai/{self.model_name}"
        )
        
        # 设置为评估模式
        self.model.eval()
        
        # 限制线程数，避免multiprocessing资源泄漏
        torch.set_num_threads(1)
        
        # 检查是否有GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) # type: ignore
        
        logger.info(f"图像Embedding模型加载完成 (设备: {self.device}, 维度: {self.embedding_dim})")
    
    def embed_image(self, image: Image.Image) -> List[float]:
        """
        对单个图像进行向量化
        
        Args:
            image: PIL图像对象
        
        Returns:
            图像向量
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt") # type: ignore
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # 处理不同版本transformers的返回值
                # 新版本返回 BaseModelOutputWithPooling，需要访问 pooler_output
                # 旧版本直接返回 tensor
                if hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output # type: ignore
                else:
                    image_features = outputs
            
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # type: ignore
            
            return image_features.cpu().numpy()[0].tolist()
            
        except Exception as e:
            logger.error(f"图像Embedding异常: {e}")
            return [0.0] * self.embedding_dim
    
    def embed_image_path(self, image_path: Path) -> List[float]:
        """
        对图像文件进行向量化
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            图像向量
        """
        try:
            image = Image.open(image_path)
            return self.embed_image(image)
        except Exception as e:
            logger.error(f"图像Embedding失败 {image_path.name}: {e}")
            return [0.0] * self.embedding_dim
    
    def embed_images(self, images: List[Image.Image]) -> List[List[float]]:
        """
        批量对图像进行向量化
        
        Args:
            images: PIL图像对象列表
        
        Returns:
            图像向量列表
        """
        embeddings = []
        for image in images:
            embedding = self.embed_image(image)
            embeddings.append(embedding)
        return embeddings
    
    def embed_image_data_list(self, image_data_list: List[ImageData]) -> Tuple[np.ndarray, List[ImageData]]:
        """
        对图像数据列表进行向量化
        
        Args:
            image_data_list: 图像数据列表
        
        Returns:
            (向量数组, 图像数据列表)
        """
        logger.info(f"开始对 {len(image_data_list)} 个图像进行向量化...")
        
        valid_embeddings = []
        valid_image_data = []
        
        for image_data in image_data_list:
            embedding = self.embed_image_path(image_data.image_path)
            
            if any(v != 0.0 for v in embedding):
                valid_embeddings.append(embedding)
                valid_image_data.append(image_data)
        
        logger.info(f"成功向量化 {len(valid_embeddings)}/{len(image_data_list)} 个图像")
        
        return np.array(valid_embeddings, dtype=np.float32), valid_image_data
    
    def embed_text_for_image_search(self, text: str) -> List[float]:
        """
        对文本进行向量化（用于图像搜索）
        
        Args:
            text: 输入文本
        
        Returns:
            文本向量
        """
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True) # type: ignore
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                # 处理不同版本transformers的返回值
                # 新版本返回 BaseModelOutputWithPooling，需要访问 pooler_output
                # 旧版本直接返回 tensor
                if hasattr(outputs, 'pooler_output'):
                    text_features = outputs.pooler_output # type: ignore
                else:
                    text_features = outputs
            
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # type: ignore
            
            return text_features.cpu().numpy()[0].tolist()
            
        except Exception as e:
            logger.error(f"文本Embedding异常: {e}")
            return [0.0] * self.embedding_dim


class FAISSIndex:
    """FAISS索引管理器"""
    
    def __init__(self, embedding_dim: int, index_type: str = "IndexFlatL2"):
        """
        初始化FAISS索引
        
        Args:
            embedding_dim: 向量维度
            index_type: 索引类型
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # 创建索引
        if index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, config.nlist)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        self.documents = []  # 存储原始文档数据
        logger.info(f"创建FAISS索引: {index_type} (维度: {embedding_dim})")
    
    def add_vectors(self, vectors: np.ndarray, documents: List[Any]):
        """
        添加向量到索引
        
        Args:
            vectors: 向量数组
            documents: 对应的文档数据
        """
        if self.index_type == "IndexIVFFlat":
            self.index.train(vectors) # type: ignore
        
        self.index.add(vectors) # type: ignore
        self.documents.extend(documents)
        logger.info(f"添加 {len(documents)} 个向量到索引 (总数: {self.index.ntotal})")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        搜索最相似的向量
        
        Args:
            query_vector: 查询向量，形状为 (embedding_dim,)
            k: 返回的结果数量，默认为5
        
        Returns:
            [(文档索引, 相似度距离)] 列表，按相似度排序（距离越小越相似）
        """
        if self.index.ntotal == 0:
            return []
        
        query_vector = np.array([query_vector], dtype=np.float32)
        # distances: 相似度距离数组，形状为 (1, k)
        # indices: 对应的文档索引数组，形状为 (1, k)
        distances: np.ndarray
        indices: np.ndarray
        distances, indices = self.index.search(query_vector, k) # type: ignore
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS返回-1表示没有足够的结果
                results.append((int(idx), float(dist)))
        
        return results
    
    def save(self, index_path: Path, documents_path: Path):
        """
        保存索引和文档数据
        
        Args:
            index_path: 索引文件路径
            documents_path: 文档数据文件路径
        """
        index_path.parent.mkdir(parents=True, exist_ok=True)
        documents_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        faiss.write_index(self.index, str(index_path))
        
        # 保存文档数据
        save_pickle(self.documents, documents_path)
        
        logger.info(f"索引已保存: {index_path} (向量数: {self.index.ntotal})")
    
    def load(self, index_path: Path, documents_path: Path):
        """
        加载索引和文档数据
        
        Args:
            index_path: 索引文件路径
            documents_path: 文档数据文件路径
        """
        # 加载索引
        self.index = faiss.read_index(str(index_path))
        
        # 加载文档数据
        self.documents = load_pickle(documents_path)
        
        logger.info(f"索引已加载: {index_path} (向量数: {self.index.ntotal})")
    
    @property
    def size(self) -> int:
        """获取索引中的向量数量"""
        return self.index.ntotal


class VectorStore:
    """向量存储管理器"""
    
    def __init__(self):
        """初始化向量存储"""
        self.text_index: Optional[FAISSIndex] = None
        self.image_index: Optional[FAISSIndex] = None
        self.text_embedding_model: Optional[TextEmbeddingModel] = None
        self.image_embedding_model: Optional[ImageEmbeddingModel] = None
        
        logger.info("向量存储管理器初始化完成")
    
    def build_text_index(self, chunks: List[TextChunk]) -> FAISSIndex:
        """
        构建文本索引
        
        Args:
            chunks: 文本块列表
        
        Returns:
            文本索引对象
        """
        if self.text_embedding_model is None:
            self.text_embedding_model = TextEmbeddingModel()
        
        # 生成文本向量
        embeddings, valid_chunks = self.text_embedding_model.embed_text_chunks(chunks)
        
        # 创建FAISS索引
        self.text_index = FAISSIndex(
            embedding_dim=config.text_embedding_dim,
            index_type=config.index_type
        )
        
        # 添加向量到索引
        self.text_index.add_vectors(embeddings, valid_chunks)
        
        logger.info(f"文本索引构建完成 (向量数: {len(valid_chunks)})")
        
        return self.text_index
    
    def build_image_index(self, image_data_list: List[ImageData]) -> FAISSIndex:
        """
        构建图像索引
        
        Args:
            image_data_list: 图像数据列表
        
        Returns:
            图像索引对象
        """
        if self.image_embedding_model is None:
            self.image_embedding_model = ImageEmbeddingModel()
        
        # 生成图像向量
        embeddings, valid_image_data = self.image_embedding_model.embed_image_data_list(
            image_data_list
        )
        
        # 创建FAISS索引
        self.image_index = FAISSIndex(
            embedding_dim=config.image_embedding_dim,
            index_type=config.index_type
        )
        
        # 添加向量到索引
        self.image_index.add_vectors(embeddings, valid_image_data)
        
        logger.info(f"图像索引构建完成 (向量数: {len(valid_image_data)})")
        
        return self.image_index
    
    def save_indexes(self):
        """保存索引"""
        if self.text_index:
            text_index_path = config.indexes_dir / "text_index.faiss" # type: ignore
            text_docs_path = config.indexes_dir / "text_documents.pkl" # type: ignore
            self.text_index.save(text_index_path, text_docs_path)
        
        if self.image_index:
            image_index_path = config.indexes_dir / "image_index.faiss" # type: ignore
            image_docs_path = config.indexes_dir / "image_documents.pkl" # type: ignore
            self.image_index.save(image_index_path, image_docs_path)
    
    def load_indexes(self):
        """加载索引"""
        text_index_path = config.indexes_dir / "text_index.faiss" # type: ignore
        text_docs_path = config.indexes_dir / "text_documents.pkl" # type: ignore
        
        if text_index_path.exists() and text_docs_path.exists():
            self.text_index = FAISSIndex(config.text_embedding_dim)
            self.text_index.load(text_index_path, text_docs_path)
            logger.info("文本索引加载完成")
        
        image_index_path = config.indexes_dir / "image_index.faiss" # type: ignore
        image_docs_path = config.indexes_dir / "image_documents.pkl" # type: ignore
        
        if image_index_path.exists() and image_docs_path.exists():
            self.image_index = FAISSIndex(config.image_embedding_dim)
            self.image_index.load(image_index_path, image_docs_path)
            logger.info("图像索引加载完成")


if __name__ == "__main__":
    # 测试代码
    from loguru import logger
    from .data_processor import DocumentProcessor, ImageProcessor
    
    logger.add("logs/embedding.log", rotation="1 day")
    
    # 加载环境变量
    from .config import load_env_config
    load_env_config()
    
    # 初始化向量存储
    vector_store = VectorStore()
    
    # 处理文档
    doc_processor = DocumentProcessor()
    chunks = doc_processor.process_directory()
    
    if chunks:
        vector_store.build_text_index(chunks)
    
    # 处理图像
    img_processor = ImageProcessor()
    images = img_processor.process_directory()
    
    if images:
        vector_store.build_image_index(images)
    
    # 保存索引
    vector_store.save_indexes()