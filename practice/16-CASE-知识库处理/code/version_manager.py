"""
知识库版本管理与性能比较模块
支持知识库版本创建、差异比较、性能评估和回归测试
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np
import faiss
from openai import OpenAI
from loguru import logger

try:
    from .config import config
    from .utils import save_json, load_json, get_api_key
except ImportError:
    from config import config
    from utils import save_json, load_json, get_api_key

# Embedding配置
TEXT_EMBEDDING_MODEL = "text-embedding-v4"
TEXT_EMBEDDING_DIM = 1024


class KnowledgeBaseVersionManager:
    """知识库版本管理器 - 版本创建、比较和性能评估"""
    
    def __init__(self, model: Optional[str] = None):
        self.model = model or config.llm_model
        self.versions: Dict[str, Dict[str, Any]] = {}
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=get_api_key(),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """获取文本的 Embedding"""
        response = self.client.embeddings.create(
            model=TEXT_EMBEDDING_MODEL,
            input=text,
            dimensions=TEXT_EMBEDDING_DIM
        )
        return list(response.data[0].embedding)
    
    def create_version(
        self, 
        knowledge_base: List[Dict[str, Any]], 
        version_name: str, 
        description: str = ""
    ) -> Dict[str, Any]:
        """创建知识库版本"""
        # 构建向量索引
        metadata_store, text_index = self._build_vector_index(knowledge_base)
        
        version_info = {
            "version_name": version_name,
            "description": description,
            "created_date": datetime.now().isoformat(),
            "knowledge_base": knowledge_base,
            "metadata_store": metadata_store,
            "text_index": text_index,
            "statistics": self._calculate_version_statistics(knowledge_base)
        }
        
        self.versions[version_name] = version_info
        logger.info(f"版本 '{version_name}' 创建成功")
        
        return version_info
    
    def _build_vector_index(
        self, 
        knowledge_base: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], faiss.IndexIDMap]:
        """构建向量索引"""
        metadata_store = []
        text_vectors = []
        
        for i, chunk in enumerate(knowledge_base):
            content = chunk.get('content', '')
            if not content.strip():
                continue
            
            metadata = {
                "id": i,
                "content": content,
                "chunk_id": chunk.get('id', f'chunk_{i}')
            }
            
            # 获取文本embedding
            vector = self._get_text_embedding(content)
            text_vectors.append(vector)
            metadata_store.append(metadata)
        
        # 创建FAISS索引
        text_index = faiss.IndexFlatL2(TEXT_EMBEDDING_DIM)
        text_index_map = faiss.IndexIDMap(text_index)
        
        if text_vectors:
            text_ids = [m["id"] for m in metadata_store]
            text_index_map.add_with_ids(
                np.array(text_vectors).astype('float32'), 
                np.array(text_ids)
            ) # type: ignore
        
        return metadata_store, text_index_map
    
    def _calculate_version_statistics(
        self, 
        knowledge_base: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算版本统计信息"""
        total_chunks = len(knowledge_base)
        total_content_length = sum(
            len(chunk.get('content', '')) for chunk in knowledge_base
        )
        
        return {
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "average_chunk_length": total_content_length / total_chunks if total_chunks > 0 else 0
        }
    
    def compare_versions(
        self, 
        version1_name: str, 
        version2_name: str
    ) -> Dict[str, Any]:
        """比较两个版本的差异"""
        if version1_name not in self.versions or version2_name not in self.versions:
            return {"error": "版本不存在"}
        
        v1 = self.versions[version1_name]
        v2 = self.versions[version2_name]
        
        kb1 = v1['knowledge_base']
        kb2 = v2['knowledge_base']
        
        comparison = {
            "version1": version1_name,
            "version2": version2_name,
            "comparison_date": datetime.now().isoformat(),
            "changes": self._detect_changes(kb1, kb2),
            "statistics_comparison": self._compare_statistics(
                v1['statistics'], v2['statistics']
            )
        }
        
        return comparison
    
    def _detect_changes(
        self, 
        kb1: List[Dict[str, Any]], 
        kb2: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """检测知识库变化"""
        changes: Dict[str, List[Dict[str, Any]]] = {
            "added_chunks": [],
            "removed_chunks": [],
            "modified_chunks": [],
            "unchanged_chunks": []
        }
        
        # 创建ID映射
        kb1_dict = {chunk.get('id'): chunk for chunk in kb1}
        kb2_dict = {chunk.get('id'): chunk for chunk in kb2}
        
        kb1_ids: Set[str] = set(kb1_dict.keys()) # type: ignore
        kb2_ids: Set[str] = set(kb2_dict.keys()) # type: ignore
        
        added_ids = kb2_ids - kb1_ids
        removed_ids = kb1_ids - kb2_ids
        common_ids = kb1_ids & kb2_ids
        
        # 记录新增的知识切片
        for chunk_id in added_ids:
            chunk = kb2_dict.get(chunk_id)
            if chunk:
                changes["added_chunks"].append({
                    "id": chunk_id,
                    "content": chunk.get('content', '')
                })
        
        # 记录删除的知识切片
        for chunk_id in removed_ids:
            chunk = kb1_dict.get(chunk_id)
            if chunk:
                changes["removed_chunks"].append({
                    "id": chunk_id,
                    "content": chunk.get('content', '')
                })
        
        # 检测修改的知识切片
        for chunk_id in common_ids:
            chunk1 = kb1_dict.get(chunk_id)
            chunk2 = kb2_dict.get(chunk_id)
            
            if chunk1 and chunk2:
                if chunk1.get('content') != chunk2.get('content'):
                    changes["modified_chunks"].append({
                        "id": chunk_id,
                        "old_content": chunk1.get('content', ''),
                        "new_content": chunk2.get('content', '')
                    })
                else:
                    changes["unchanged_chunks"].append({"id": chunk_id})
        
        return changes
    
    def _compare_statistics(
        self, 
        stats1: Dict[str, Any], 
        stats2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """比较统计信息"""
        comparison = {}
        
        for key in stats1.keys():
            if key in stats2:
                if isinstance(stats1[key], (int, float)):
                    val1 = stats1[key]
                    val2 = stats2[key]
                    comparison[key] = {
                        "version1": val1,
                        "version2": val2,
                        "difference": val2 - val1,
                        "percentage_change": (
                            (val2 - val1) / val1 * 100 if val1 != 0 else 0
                        )
                    }
        
        return comparison
    
    def evaluate_version_performance(
        self, 
        version_name: str, 
        test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """评估版本性能"""
        if version_name not in self.versions:
            return {"error": "版本不存在"}
        
        performance_metrics = {
            "version_name": version_name,
            "evaluation_date": datetime.now().isoformat(),
            "query_results": [],
            "overall_metrics": {}
        }
        
        total_queries = len(test_queries)
        correct_answers = 0
        response_times = []
        
        for query_info in test_queries:
            query = query_info['query']
            expected_answer = query_info.get('expected_answer', '')
            
            # 使用embedding检索
            start_time = datetime.now()
            retrieved_chunks = self._retrieve_relevant_chunks(query, version_name)
            end_time = datetime.now()
            
            response_time = (end_time - start_time).total_seconds()
            response_times.append(response_time)
            
            # 评估检索质量
            is_correct = self._evaluate_retrieval_quality(
                query, retrieved_chunks, expected_answer
            )
            if is_correct:
                correct_answers += 1
            
            performance_metrics["query_results"].append({
                "query": query,
                "retrieved_chunks": len(retrieved_chunks),
                "response_time": response_time,
                "is_correct": is_correct
            })
        
        # 计算整体指标
        accuracy = correct_answers / total_queries if total_queries > 0 else 0
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )
        
        performance_metrics["overall_metrics"] = {
            "accuracy": accuracy,
            "avg_response_time": avg_response_time,
            "total_queries": total_queries,
            "correct_answers": correct_answers
        }
        
        return performance_metrics
    
    def _retrieve_relevant_chunks(
        self, 
        query: str, 
        version_name: str, 
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """使用embedding和faiss检索相关知识切片"""
        if version_name not in self.versions:
            return []
        
        version_info = self.versions[version_name]
        metadata_store = version_info['metadata_store']
        text_index = version_info['text_index']
        
        # 获取查询的embedding
        query_vector = np.array([self._get_text_embedding(query)]).astype('float32')
        
        # 使用faiss进行检索
        distances, indices = text_index.search(query_vector, k)
        
        relevant_chunks = []
        for i, doc_id in enumerate(indices[0]):
            if doc_id != -1:
                match = next(
                    (item for item in metadata_store if item["id"] == doc_id), 
                    None
                )
                if match:
                    chunk = {
                        "id": match["chunk_id"],
                        "content": match["content"],
                        "similarity_score": 1.0 / (1.0 + distances[0][i])
                    }
                    relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def _evaluate_retrieval_quality(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        expected_answer: str
    ) -> bool:
        """评估检索质量"""
        if not retrieved_chunks:
            return False
        
        for chunk in retrieved_chunks:
            content = chunk.get('content', '').lower()
            if expected_answer.lower() in content:
                return True
        
        return False
    
    def compare_version_performance(
        self, 
        version1_name: str, 
        version2_name: str, 
        test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """比较两个版本的性能"""
        perf1 = self.evaluate_version_performance(version1_name, test_queries)
        perf2 = self.evaluate_version_performance(version2_name, test_queries)
        
        if "error" in perf1 or "error" in perf2:
            return {"error": "版本评估失败"}
        
        metrics1 = perf1["overall_metrics"]
        metrics2 = perf2["overall_metrics"]
        
        comparison = {
            "version1": version1_name,
            "version2": version2_name,
            "comparison_date": datetime.now().isoformat(),
            "performance_comparison": {
                "accuracy": {
                    "version1": metrics1["accuracy"],
                    "version2": metrics2["accuracy"],
                    "improvement": metrics2["accuracy"] - metrics1["accuracy"]
                },
                "response_time": {
                    "version1": metrics1["avg_response_time"],
                    "version2": metrics2["avg_response_time"],
                    "improvement": metrics1["avg_response_time"] - metrics2["avg_response_time"]
                }
            },
            "recommendation": self._generate_performance_recommendation(
                metrics1, metrics2
            )
        }
        
        return comparison
    
    def _generate_performance_recommendation(
        self, 
        perf1: Dict[str, Any], 
        perf2: Dict[str, Any]
    ) -> str:
        """生成性能建议"""
        acc1 = perf1["accuracy"]
        acc2 = perf2["accuracy"]
        time1 = perf1["avg_response_time"]
        time2 = perf2["avg_response_time"]
        
        if acc2 > acc1 and time2 <= time1:
            return f"推荐使用版本2，准确率提升{(acc2-acc1)*100:.1f}%，响应时间{'提升' if time2 < time1 else '相当'}"
        elif acc2 > acc1 and time2 > time1:
            return "版本2准确率更高但响应时间较长，需要权衡"
        elif acc2 < acc1 and time2 < time1:
            return "版本2响应更快但准确率较低，需要权衡"
        else:
            return "推荐使用版本1，性能更优"
    
    def generate_regression_test(
        self, 
        version_name: str, 
        test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成回归测试"""
        if version_name not in self.versions:
            return {"error": "版本不存在"}
        
        regression_results = {
            "version_name": version_name,
            "test_date": datetime.now().isoformat(),
            "test_results": [],
            "pass_rate": 0
        }
        
        passed_tests = 0
        total_tests = len(test_queries)
        
        for query_info in test_queries:
            query = query_info['query']
            expected_answer = query_info.get('expected_answer', '')
            
            retrieved_chunks = self._retrieve_relevant_chunks(query, version_name)
            is_passed = self._evaluate_retrieval_quality(
                query, retrieved_chunks, expected_answer
            )
            
            if is_passed:
                passed_tests += 1
            
            regression_results["test_results"].append({
                "query": query,
                "expected": expected_answer,
                "retrieved": len(retrieved_chunks),
                "passed": is_passed
            })
        
        regression_results["pass_rate"] = (
            passed_tests / total_tests if total_tests > 0 else 0
        )
        
        return regression_results
    
    def save_version(self, version_name: str, file_path: Optional[str] = None) -> None:
        """保存版本到文件（不含FAISS索引）"""
        if version_name not in self.versions:
            raise ValueError(f"版本 '{version_name}' 不存在")
        
        if file_path is None:
            file_path = str(config.versions_dir / f"{version_name}.json") # type: ignore
        
        version_data = self.versions[version_name].copy()
        # 移除FAISS索引（无法序列化）
        version_data.pop('text_index', None)
        
        save_json(version_data, file_path) # type: ignore
        logger.info(f"版本 '{version_name}' 已保存到 {file_path}")
    
    def load_version(self, file_path: str, version_name: Optional[str] = None) -> Dict[str, Any]:
        """从文件加载版本"""
        version_data = load_json(file_path) # type: ignore
        
        if version_name is None:
            version_name = version_data.get('version_name', 'loaded_version')
        
        # 重新构建索引
        knowledge_base = version_data['knowledge_base']
        metadata_store, text_index = self._build_vector_index(knowledge_base)
        
        version_data['metadata_store'] = metadata_store
        version_data['text_index'] = text_index
        
        self.versions[version_name] = version_data # type: ignore
        logger.info(f"版本 '{version_name}' 已从 {file_path} 加载")
        
        return version_data
