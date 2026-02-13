"""
知识库健康度检查模块
检查知识库的完整性、时效性和一致性
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI
from loguru import logger

from .config import config
from .utils import preprocess_json_response, get_api_key, get_datetime_str


class KnowledgeBaseHealthChecker:
    """知识库健康度检查器 - 检查完整性、时效性和一致性"""
    
    def __init__(self, model: Optional[str] = None):
        self.model = model or config.llm_model
        self.health_report: Dict[str, Any] = {}
        
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
    
    def check_missing_knowledge(
        self, 
        knowledge_base: List[Dict[str, Any]], 
        test_queries: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """使用LLM检查缺少的知识"""
        instruction = """
你是一个知识库完整性检查专家。请分析给定的测试查询和知识库内容，判断知识库中是否缺少相关的知识。

检查标准：
1. 查询是否能在知识库中找到相关答案
2. 知识是否完整、准确
3. 是否覆盖了用户的主要需求
4. 是否存在知识空白

请返回JSON格式：
{
    "missing_knowledge": [
        {
            "query": "测试查询",
            "missing_aspect": "缺少的知识方面",
            "importance": "重要性（高/中/低）",
            "suggested_content": "建议的知识内容",
            "category": "知识分类"
        }
    ],
    "coverage_score": "覆盖率评分(0-1)",
    "completeness_analysis": "完整性分析"
}
"""
        
        # 构建知识库内容摘要
        knowledge_summary = []
        for chunk in knowledge_base:
            knowledge_summary.append(
                f"ID: {chunk.get('id', 'unknown')} - {chunk.get('content', '')}"
            )
        
        knowledge_text = "\n".join(knowledge_summary)
        
        # 构建测试查询列表
        queries_text = []
        for query_info in test_queries:
            query = query_info['query']
            expected = query_info.get('expected_answer', '')
            queries_text.append(f"查询: {query} | 期望答案: {expected}")
        
        queries_text = "\n".join(queries_text)
        
        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 测试查询 ###
{queries_text}

### 分析结果 ###
"""
        
        try:
            response = self.get_completion(prompt)
            response = preprocess_json_response(response)
            result = json.loads(response)
            return result
            
        except Exception as e:
            logger.error(f"LLM检查缺少知识失败: {e}")
            return None
    
    def check_outdated_knowledge(
        self, 
        knowledge_base: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """使用LLM检查过期的知识"""
        instruction = """
你是一个知识时效性检查专家。请分析给定的知识内容，判断是否存在过期或需要更新的信息。

检查标准：
1. 时间相关信息是否过期（年份、日期、时间范围）
2. 价格信息是否最新（价格、费用、票价等）
3. 政策规则是否更新（政策、规定、规则等）
4. 活动信息是否有效（活动、节日、特殊安排等）
5. 联系方式是否准确（电话、地址、网址等）
6. 技术信息是否过时（版本、技术标准等）

请返回JSON格式：
{
    "outdated_knowledge": [
        {
            "chunk_id": "知识切片ID",
            "content": "知识内容",
            "outdated_aspect": "过期方面",
            "severity": "严重程度（高/中/低）",
            "suggested_update": "建议更新内容",
            "last_verified": "最后验证时间"
        }
    ],
    "freshness_score": "新鲜度评分(0-1)",
    "update_recommendations": "更新建议"
}
"""
        
        # 构建知识库内容
        knowledge_text = []
        for chunk in knowledge_base:
            content = chunk.get('content', '')
            chunk_id = chunk.get('id', 'unknown')
            last_updated = chunk.get('last_updated', 'unknown')
            knowledge_text.append(
                f"ID: {chunk_id} | 更新时间: {last_updated} | 内容: {content}"
            )
        
        knowledge_text = "\n".join(knowledge_text)
        
        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 当前时间 ###
{get_datetime_str()}

### 分析结果 ###
"""
        
        try:
            response = self.get_completion(prompt)
            response = preprocess_json_response(response)
            result = json.loads(response)
            return result
            
        except Exception as e:
            logger.error(f"LLM检查过期知识失败: {e}")
            return None
    
    def check_conflicting_knowledge(
        self, 
        knowledge_base: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """使用LLM检查冲突的知识"""
        instruction = """
你是一个知识一致性检查专家。请分析给定的知识库，找出可能存在冲突或矛盾的信息。

检查标准：
1. 同一主题的不同说法（地点、名称、描述等）
2. 价格信息的差异（价格、费用、收费标准等）
3. 时间信息的不一致（营业时间、开放时间、活动时间等）
4. 规则政策的冲突（规定、政策、要求等）
5. 操作流程的差异（步骤、方法、流程等）
6. 联系方式的差异（地址、电话、网址等）

请返回JSON格式：
{
    "conflicting_knowledge": [
        {
            "conflict_type": "冲突类型",
            "chunk_ids": ["相关切片ID"],
            "conflicting_content": ["冲突内容"],
            "severity": "严重程度（高/中/低）",
            "resolution_suggestion": "解决建议"
        }
    ],
    "consistency_score": "一致性评分(0-1)",
    "conflict_analysis": "冲突分析"
}
"""
        
        # 构建知识库内容
        knowledge_text = []
        for chunk in knowledge_base:
            content = chunk.get('content', '')
            chunk_id = chunk.get('id', 'unknown')
            knowledge_text.append(f"ID: {chunk_id} | 内容: {content}")
        
        knowledge_text = "\n".join(knowledge_text)
        
        prompt = f"""
### 指令 ###
{instruction}

### 知识库内容 ###
{knowledge_text}

### 分析结果 ###
"""
        
        try:
            response = self.get_completion(prompt)
            response = preprocess_json_response(response)
            result = json.loads(response)
            return result
            
        except Exception as e:
            logger.error(f"LLM检查冲突知识失败: {e}")
            return None
    
    def _calculate_overall_health_score(
        self, 
        missing_result: Dict[str, Any], 
        outdated_result: Dict[str, Any], 
        conflicting_result: Dict[str, Any]
    ) -> float:
        """计算整体健康度评分"""
        coverage_score = missing_result.get('coverage_score', 0)
        freshness_score = outdated_result.get('freshness_score', 0)
        consistency_score = conflicting_result.get('consistency_score', 0)
        
        # 加权计算
        overall_score = (
            coverage_score * config.coverage_weight +
            freshness_score * config.freshness_weight +
            consistency_score * config.consistency_weight
        )
        
        return float(overall_score)
    
    def _get_health_level(self, score: float) -> str:
        """根据评分确定健康等级"""
        if score >= 0.8:
            return "优秀"
        elif score >= 0.6:
            return "良好"
        elif score >= 0.4:
            return "一般"
        else:
            return "需要改进"
    
    def _generate_recommendations(
        self, 
        missing_result: Dict[str, Any], 
        outdated_result: Dict[str, Any], 
        conflicting_result: Dict[str, Any]
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        missing_count = len(missing_result.get('missing_knowledge', []))
        if missing_count > 0:
            recommendations.append(f"补充{missing_count}个缺少的知识点，提高覆盖率")
        
        outdated_count = len(outdated_result.get('outdated_knowledge', []))
        if outdated_count > 0:
            recommendations.append(f"更新{outdated_count}个过期知识点，确保信息时效性")
        
        conflicting_count = len(conflicting_result.get('conflicting_knowledge', []))
        if conflicting_count > 0:
            recommendations.append(f"解决{conflicting_count}个知识冲突，提高一致性")
        
        if not recommendations:
            recommendations.append("知识库状态良好，建议定期维护")
        
        return recommendations
    
    def generate_health_report(
        self, 
        knowledge_base: List[Dict[str, Any]], 
        test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成完整的健康度报告"""
        logger.info("正在检查知识库健康度...")
        
        # 1. 检查缺少的知识
        logger.info("1. 检查缺少的知识...")
        missing_result = self.check_missing_knowledge(knowledge_base, test_queries)
        
        # 2. 检查过期的知识
        logger.info("2. 检查过期的知识...")
        outdated_result = self.check_outdated_knowledge(knowledge_base)
        
        # 3. 检查冲突的知识
        logger.info("3. 检查冲突的知识...")
        conflicting_result = self.check_conflicting_knowledge(knowledge_base)
        
        # 确保所有结果都不为None
        if missing_result is None:
            missing_result = {"missing_knowledge": [], "coverage_score": 0.5, "completeness_analysis": "检查失败"}
        if outdated_result is None:
            outdated_result = {"outdated_knowledge": [], "freshness_score": 0.5, "update_recommendations": "检查失败"}
        if conflicting_result is None:
            conflicting_result = {"conflicting_knowledge": [], "consistency_score": 0.5, "conflict_analysis": "检查失败"}
        
        # 4. 计算整体健康度
        overall_score = self._calculate_overall_health_score(
            missing_result, outdated_result, conflicting_result
        )
        
        # 5. 生成报告
        report = {
            "overall_health_score": overall_score,
            "health_level": self._get_health_level(overall_score),
            "missing_knowledge": missing_result,
            "outdated_knowledge": outdated_result,
            "conflicting_knowledge": conflicting_result,
            "recommendations": self._generate_recommendations(
                missing_result, outdated_result, conflicting_result
            ),
            "check_date": datetime.now().isoformat()
        }
        
        self.health_report = report
        return report
