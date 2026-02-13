"""
知识库问题生成与检索优化模块
使用BM25算法实现知识库的智能问题生成和检索优化
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from loguru import logger

from .config import config
from .utils import preprocess_text, preprocess_json_response, get_api_key


class KnowledgeBaseOptimizer:
    """知识库优化器 - 基于BM25的问题生成与检索优化"""
    
    def __init__(self, model: Optional[str] = None):
        self.model = model or config.llm_model
        self.knowledge_base: List[Dict[str, Any]] = []
        self.content_bm25: Optional[BM25Okapi] = None
        self.question_bm25: Optional[BM25Okapi] = None
        self.content_documents: List[List[str]] = []
        self.question_documents: List[List[str]] = []
        self.content_metadata: List[Dict[str, Any]] = []
        self.question_metadata: List[Dict[str, Any]] = []
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=get_api_key(),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def get_completion(self, prompt: str, model: Optional[str] = None) -> str:
        """调用LLM生成文本"""
        model = model or self.model
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            temperature=config.llm_temperature,
        )
        return response.choices[0].message.content or ""
    
    def generate_questions_for_chunk(
        self, 
        knowledge_chunk: str, 
        num_questions: int = 5
    ) -> List[Dict[str, Any]]:
        """为单个知识切片生成多样化问题"""
        instruction = """
你是一个专业的问答系统专家。给定的知识内容能回答哪些多样化的问题，这些问题可以：
1. 使用不同的问法（直接问、间接问、对比问等）
2. 避免重复和相似的问题
3. 确保问题不超出知识内容范围

请返回JSON格式：
{
    "questions": [
        {
            "question": "问题内容",
            "question_type": "问题类型（直接问/间接问/对比问/条件问等）",
            "difficulty": "难度等级（简单/中等/困难）"
        }
    ]
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 知识内容 ###
{knowledge_chunk}

### 生成问题数量 ###
{num_questions}

### 生成结果 ###
"""
        
        response = self.get_completion(prompt)
        response = preprocess_json_response(response)
        
        try:
            result = json.loads(response)
            return result.get('questions', [])
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return [{"question": f"关于{knowledge_chunk[:50]}...的问题", 
                    "question_type": "直接问", 
                    "difficulty": "中等"}]
    
    def generate_diverse_questions(
        self, 
        knowledge_chunk: str, 
        num_questions: int = 8
    ) -> List[Dict[str, Any]]:
        """生成更多样化的问题（更丰富）"""
        instruction = """
你是一个专业的问答系统专家。请为给定的知识内容生成高度多样化的问题，确保：
1. 问题类型多样化：直接问、间接问、对比问、条件问、假设问、推理问等
2. 表达方式多样化：使用不同的句式、词汇、语气
3. 难度层次多样化：简单、中等、困难的问题都要有
4. 角度多样化：从不同角度和维度提问
5. 确保问题不超出知识内容范围

请返回JSON格式：
{
    "questions": [
        {
            "question": "问题内容",
            "question_type": "问题类型",
            "difficulty": "难度等级",
            "perspective": "提问角度",
            "is_answerable": "给出的知识能否回答该问题",
            "answer": "基于该知识的回答"
        }
    ]
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 知识内容 ###
{knowledge_chunk}

### 生成问题数量 ###
{num_questions}

### 生成结果 ###
"""
        
        response = self.get_completion(prompt)
        response = preprocess_json_response(response)
        
        try:
            result = json.loads(response)
            return result.get('questions', [])
        except json.JSONDecodeError as e:
            logger.error(f"多样化问题生成JSON解析失败: {e}")
            return []
    
    def build_knowledge_index(self, knowledge_base: List[Dict[str, Any]]) -> None:
        """构建知识库的BM25索引（包括原文和问题）"""
        logger.info("正在构建知识库索引...")
        
        self.knowledge_base = knowledge_base
        content_documents = []
        question_documents = []
        content_metadata = []
        question_metadata = []
        
        for i, chunk in enumerate(knowledge_base):
            text = chunk.get('content', '')
            if not text.strip():
                continue
            
            # 原文文档
            content_words = preprocess_text(text, config.stop_words)
            if content_words:
                content_documents.append(content_words)
                content_metadata.append({
                    "id": chunk.get('id', f"chunk_{i}"),
                    "content": text,
                    "category": chunk.get('category', ''),
                    "chunk": chunk,
                    "type": "content"
                })
            
            # 问题文档（如果存在生成的问题）
            if 'generated_questions' in chunk and chunk['generated_questions']:
                for j, question_data in enumerate(chunk['generated_questions']):
                    question = question_data.get('question', '')
                    if question.strip():
                        combined_text = f"内容：{text} 问题：{question}"
                        question_words = preprocess_text(combined_text, config.stop_words)
                        
                        if question_words:
                            question_documents.append(question_words)
                            question_metadata.append({
                                "id": f"{chunk.get('id', f'chunk_{i}')}_q{j}",
                                "content": question,
                                "combined_content": combined_text,
                                "category": chunk.get('category', ''),
                                "chunk": chunk,
                                "type": "question",
                                "question_data": question_data
                            })
        
        # 创建BM25索引
        if content_documents:
            self.content_bm25 = BM25Okapi(content_documents)
            self.content_documents = content_documents
            self.content_metadata = content_metadata
            logger.info(f"原文索引构建完成，共索引 {len(content_documents)} 个知识切片")
        
        if question_documents:
            self.question_bm25 = BM25Okapi(question_documents)
            self.question_documents = question_documents
            self.question_metadata = question_metadata
            logger.info(f"问题索引构建完成，共索引 {len(question_documents)} 个问题")
        
        if not content_documents and not question_documents:
            logger.warning("没有有效的内容可以索引")
    
    def search_similar_chunks(
        self, 
        query: str, 
        k: int = 3, 
        search_type: str = "content"
    ) -> List[Dict[str, Any]]:
        """使用BM25搜索相似的内容（原文或问题）"""
        if search_type == "content":
            if not self.content_bm25:
                return []
            bm25 = self.content_bm25
            metadata_store = self.content_metadata
        elif search_type == "question":
            if not self.question_bm25:
                return []
            bm25 = self.question_bm25
            metadata_store = self.question_metadata
        else:
            return []
        
        try:
            query_words = preprocess_text(query, config.stop_words)
            if not query_words:
                return []
            
            scores = bm25.get_scores(query_words)
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    metadata = metadata_store[idx]
                    similarity = min(1.0, scores[idx] / 10.0)
                    results.append({
                        "metadata": metadata,
                        "score": float(scores[idx]),
                        "similarity": similarity
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def calculate_similarity(self, query: str, knowledge_chunk: str) -> float:
        """计算查询与知识切片的相似度（使用BM25）"""
        try:
            query_words = preprocess_text(query, config.stop_words)
            chunk_words = preprocess_text(knowledge_chunk, config.stop_words)
            
            if not query_words or not chunk_words:
                return 0.0
            
            temp_bm25 = BM25Okapi([chunk_words])
            scores = temp_bm25.get_scores(query_words)
            
            max_score = max(scores) if scores else 0.0
            return min(1.0, max_score / 10.0)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def evaluate_retrieval_methods(
        self, 
        knowledge_base: List[Dict[str, Any]], 
        test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """评估两种检索方法的准确度"""
        self.build_knowledge_index(knowledge_base)
        
        results = {
            'content_similarity': [],
            'question_similarity': [],
            'improvement': [],
            'content_scores': [],
            'question_scores': [],
            'query_details': []
        }
        
        for query_info in test_queries:
            user_query = query_info['query']
            correct_chunk = query_info['correct_chunk']
            
            # 方法1：BM25原文检索
            content_results = self.search_similar_chunks(user_query, k=1, search_type="content")
            content_correct = False
            content_score = 0.0
            content_chunk_id = None
            if content_results:
                best_match = content_results[0]['metadata']['chunk']
                content_correct = best_match['content'] == correct_chunk
                content_score = content_results[0]['similarity']
                content_chunk_id = best_match['id']
            
            # 方法2：BM25问题检索
            question_results = self.search_similar_chunks(user_query, k=1, search_type="question")
            question_correct = False
            question_score = 0.0
            question_chunk_id = None
            if question_results:
                best_match = question_results[0]['metadata']['chunk']
                question_correct = best_match['content'] == correct_chunk
                question_score = question_results[0]['similarity']
                question_chunk_id = best_match['id']
            
            results['content_similarity'].append(content_correct)
            results['question_similarity'].append(question_correct)
            results['improvement'].append(question_correct and not content_correct)
            results['content_scores'].append(content_score)
            results['question_scores'].append(question_score)
            
            results['query_details'].append({
                'query': user_query,
                'content_score': content_score,
                'question_score': question_score,
                'content_correct': content_correct,
                'question_correct': question_correct,
                'score_diff': question_score - content_score,
                'content_chunk_id': content_chunk_id,
                'question_chunk_id': question_chunk_id
            })
        
        return results
