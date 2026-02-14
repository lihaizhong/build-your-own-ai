"""
对话知识提取与沉淀模块
从对话中提取有价值的信息并沉淀为知识
"""

import json
from collections import Counter
from typing import Any, Dict, List, Optional

from openai import OpenAI
from loguru import logger

try:
    from .config import config
    from .utils import preprocess_json_response, get_api_key
except ImportError:
    from config import config
    from utils import preprocess_json_response, get_api_key

class ConversationKnowledgeExtractor:
    """对话知识提取器 - 从对话中提取和沉淀知识"""
    
    def __init__(self, model: Optional[str] = None):
        self.model = model or config.llm_model
        self.extracted_knowledge: List[Dict[str, Any]] = []
        self.knowledge_frequency: Counter = Counter()
        
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
    
    def extract_knowledge_from_conversation(
        self, 
        conversation: str
    ) -> Dict[str, Any]:
        """从单次对话中提取知识"""
        instruction = """
你是一个专业的知识提取专家。请从给定的对话中提取有价值的知识点，包括：
1. 事实性信息（地点、时间、价格、规则等）
2. 用户需求和偏好
3. 常见问题和解答
4. 操作流程和步骤
5. 注意事项和提醒

请返回JSON格式：
{
    "extracted_knowledge": [
        {
            "knowledge_type": "知识类型（事实/需求/问题/流程/注意）",
            "content": "知识内容",
            "confidence": "置信度(0-1)",
            "source": "来源（用户/AI/对话）",
            "keywords": ["关键词1", "关键词2"],
            "category": "分类"
        }
    ],
    "conversation_summary": "对话摘要",
    "user_intent": "用户意图"
}
"""
        
        prompt = f"""
### 指令 ###
{instruction}

### 对话内容 ###
{conversation}

### 提取结果 ###
"""
        
        response = self.get_completion(prompt)
        response = preprocess_json_response(response)
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"对话知识提取JSON解析失败: {e}")
            return {
                "extracted_knowledge": [],
                "conversation_summary": "无法解析对话",
                "user_intent": "未知"
            }
    
    def batch_extract_knowledge(
        self, 
        conversations: List[str]
    ) -> List[Dict[str, Any]]:
        """批量提取知识"""
        all_knowledge = []
        
        for i, conversation in enumerate(conversations):
            logger.info(f"正在处理对话 {i+1}/{len(conversations)}...")
            
            result = self.extract_knowledge_from_conversation(conversation)
            all_knowledge.extend(result.get('extracted_knowledge', []))
            
            # 更新频率统计
            for knowledge in result.get('extracted_knowledge', []):
                key = f"{knowledge['knowledge_type']}:{knowledge['content'][:50]}"
                self.knowledge_frequency[key] += 1
        
        return all_knowledge
    
    def merge_similar_knowledge(
        self, 
        knowledge_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """使用LLM合并相似的知识点，过滤掉需求和问题类型"""
        # 过滤掉需求和问题类型的知识
        filtered_knowledge = [
            knowledge for knowledge in knowledge_list 
            if knowledge.get('knowledge_type') not in ['需求', '问题']
        ]
        
        logger.info(f"过滤前知识点数量: {len(knowledge_list)}")
        logger.info(f"过滤后知识点数量: {len(filtered_knowledge)}")
        logger.info(f"过滤掉的'需求'和'问题'类型知识点: {len(knowledge_list) - len(filtered_knowledge)}")
        
        # 按知识类型分组
        knowledge_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for knowledge in filtered_knowledge:
            knowledge_type = knowledge.get('knowledge_type', '其他')
            if knowledge_type not in knowledge_by_type:
                knowledge_by_type[knowledge_type] = []
            knowledge_by_type[knowledge_type].append(knowledge)
        
        merged_knowledge = []
        
        # 对每个知识类型分别进行LLM合并
        for knowledge_type, knowledge_group in knowledge_by_type.items():
            if len(knowledge_group) == 1:
                merged_knowledge.append(knowledge_group[0])
            else:
                merged = self._merge_knowledge_with_llm(knowledge_group, knowledge_type)
                merged_knowledge.append(merged)
        
        return merged_knowledge
    
    def _merge_knowledge_with_llm(
        self, 
        knowledge_group: List[Dict[str, Any]], 
        knowledge_type: str
    ) -> Dict[str, Any]:
        """使用LLM合并同类型的知识组"""
        knowledge_contents = []
        all_keywords: set = set()
        all_sources: List[str] = []
        
        for i, knowledge in enumerate(knowledge_group, 1):
            content = knowledge.get('content', '')
            confidence = knowledge.get('confidence', 0.5)
            keywords = knowledge.get('keywords', [])
            source = knowledge.get('source', '')
            category = knowledge.get('category', '')
            
            knowledge_contents.append(f"{i}. 内容: {content}")
            knowledge_contents.append(f"   置信度: {confidence}")
            knowledge_contents.append(f"   分类: {category}")
            knowledge_contents.append(f"   来源: {source}")
            knowledge_contents.append(f"   关键词: {', '.join(keywords)}")
            knowledge_contents.append("")
            
            all_keywords.update(keywords)
            if source and source not in all_sources:
                all_sources.append(source)
        
        prompt = f"""
你是一个专业的知识整理专家。请将以下{knowledge_type}类型的知识点进行智能合并，生成一个更完整、准确的知识点。

### 合并要求：
1. 保留所有重要信息，避免信息丢失
2. 消除重复内容，整合相似表述
3. 提高内容的准确性和完整性
4. 保持逻辑清晰，结构合理
5. 合并后的置信度取所有知识点中的最高值

### 待合并的知识点：
{chr(10).join(knowledge_contents)}

### 请返回JSON格式：
{{
    "knowledge_type": "{knowledge_type}",
    "content": "合并后的知识内容",
    "confidence": 最高置信度值,
    "keywords": ["合并后的关键词列表"],
    "category": "合并后的分类",
    "sources": ["所有来源"],
    "frequency": {len(knowledge_group)}
}}

### 合并结果：
"""
        
        response = self.get_completion(prompt)
        response = preprocess_json_response(response)
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"知识合并JSON解析失败: {e}")
            
            # 解析失败时，使用简单的合并策略
            best_knowledge = max(knowledge_group, key=lambda x: x.get('confidence', 0))
            return {
                "knowledge_type": knowledge_type,
                "content": best_knowledge['content'],
                "confidence": best_knowledge.get('confidence', 0.5),
                "frequency": len(knowledge_group),
                "keywords": list(all_keywords),
                "category": best_knowledge['category'],
                "sources": all_sources
            }
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """获取知识提取统计信息"""
        return {
            "total_knowledge": len(self.extracted_knowledge),
            "unique_knowledge": len(self.knowledge_frequency),
            "top_knowledge": self.knowledge_frequency.most_common(10),
            "type_distribution": dict(Counter(
                k.split(':')[0] for k in self.knowledge_frequency.keys()
            ))
        }
