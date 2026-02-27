#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
===========================================
深思熟虑智能体（Deliberative Agent）- 智能投研助手
===========================================

【学习目标】
1. 理解深思熟虑型智能体的核心流程（感知→建模→推理→决策→报告）
2. 掌握 LangGraph 状态图的构建方法
3. 学习 LangChain >= 1.0 的最佳实践

【核心概念】
- 深思熟虑型智能体：不同于简单的"感知-行动"循环，它会：
  1. 构建内部世界模型
  2. 生成多个候选方案
  3. 模拟和评估结果
  4. 选择最优行动方案

【LangChain >= 1.0 主要变化】
1. pydantic_v1 兼容层已废弃，直接使用 pydantic v2
2. LLM 类统一使用 langchain_openai 或 langchain_community 的聊天模型
3. 输出解析器更推荐使用 with_structured_output() 方法
4. LangGraph 状态管理使用 TypedDict + Annotated

基于LangGraph实现的深思熟虑型智能体，适用于投资研究场景，能够整合数据，
进行多步骤分析和推理，生成投资观点和研究报告。

核心流程：
1. 感知：收集市场数据和信息
2. 建模：构建内部世界模型，理解市场状态
3. 推理：生成多个候选分析方案并模拟结果
4. 决策：选择最优投资观点并形成报告
5. 报告：生成完整研究报告
"""

import os
import json
from typing import Dict, List, Any, Literal, Optional, Annotated, TypedDict
from datetime import datetime

# ============================================================================
# 【导入说明 - LangChain >= 1.0 最佳实践】
# ============================================================================
# 1. pydantic 直接导入，不再使用 langchain_core.pydantic_v1 兼容层
#    原因：LangChain 1.0+ 已经完全适配 Pydantic v2
from pydantic import BaseModel, Field

# 2. 提示模板从 langchain_core.prompts 导入（标准位置）
from langchain_core.prompts import ChatPromptTemplate

# 3. 输出解析器从 langchain_core.output_parsers 导入
from langchain_core.output_parsers import StrOutputParser

# 4. 【重要变化】LLM 导入方式
#    LangChain >= 1.0 推荐使用 Chat Models 而非传统 LLM
#    - 阿里云通义千问：使用 langchain_openai.ChatOpenAI 兼容模式
#    - 或者使用 langchain_community.chat_models 中的 ChatTongyi
from langchain_openai import ChatOpenAI

# 5. LangGraph 核心组件
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
# 消息聚合器：用于在状态中累积消息列表
from langgraph.graph.message import add_messages

# ============================================================================
# 【配置说明 - API 密钥和模型设置】
# ============================================================================
# 环境变量获取最佳实践：
# 1. 使用 os.getenv() 而非 os.environ[]，避免未设置时报错
# 2. 提供默认值或在使用前检查
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# ============================================================================
# 【LLM 实例创建 - LangChain >= 1.0 推荐方式】
# ============================================================================
# 阿里云通义千问的 OpenAI 兼容接口
# 优点：与 OpenAI API 格式完全兼容，迁移成本低
# 
# 【对比】LangChain < 1.0 的旧方式：
#   from langchain_community.llms import Tongyi
#   llm = Tongyi(model_name="qwen-turbo-latest", dashscope_api_key=...)
#
# 【新方式】使用 ChatOpenAI 兼容模式：
#   base_url: 阿里云的 OpenAI 兼容端点
#   api_key: 阿里云的 API Key
#   model: 模型名称（不需要 qwen- 前缀时直接用 turbo、plus 等）
#
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=DASHSCOPE_API_KEY,  # type: ignore[assignment]
    model="qwen-turbo-latest",  # 或 "qwen-plus-latest", "qwen-max-latest"
    temperature=0.7,  # 控制输出随机性，0-1之间，越高越随机
)


# ============================================================================
# 【Pydantic 模型定义 - 输出结构化数据】
# ============================================================================
# 【学习要点】
# 1. BaseModel：Pydantic 的基础类，用于数据验证和序列化
# 2. Field()：定义字段属性，包括描述、默认值、验证规则等
# 3. 类型提示：使用 Python 类型提示（str, int, float, List, Dict 等）
#
# 【LangChain >= 1.0 变化】
# 旧版：from langchain_core.pydantic_v1 import BaseModel, Field
# 新版：from pydantic import BaseModel, Field
#
# 原因：LangChain 1.0+ 不再需要 pydantic_v1 兼容层
#

class PerceptionOutput(BaseModel):
    """
    感知阶段输出的市场数据和信息
    
    【Pydantic 模型作用】
    1. 数据验证：自动验证字段类型和约束
    2. 序列化：轻松转换为 JSON/dict
    3. 文档化：Field(description=...) 提供 API 文档
    4. 结构化输出：配合 LLM 的 with_structured_output() 使用
    """
    market_overview: str = Field(..., description="市场概况和最新动态")
    # 【Field 参数说明】
    # ... (Ellipsis)：表示该字段必填
    # description：字段描述，用于生成 JSON Schema
    key_indicators: Dict[str, str] = Field(
        ..., 
        description="关键经济和市场指标"
    )
    recent_news: List[str] = Field(
        ..., 
        description="近期重要新闻",
        min_length=3,  # 【Pydantic v2 新增】支持列表长度验证
    )
    industry_trends: Dict[str, str] = Field(..., description="行业趋势分析")


class ModelingOutput(BaseModel):
    """建模阶段输出的内部世界模型"""
    market_state: str = Field(..., description="当前市场状态评估")
    economic_cycle: str = Field(..., description="经济周期判断")
    risk_factors: List[str] = Field(
        ..., 
        description="主要风险因素",
        min_length=3,
    )
    opportunity_areas: List[str] = Field(
        ..., 
        description="潜在机会领域",
        min_length=3,
    )
    market_sentiment: str = Field(..., description="市场情绪分析")


class ReasoningPlan(BaseModel):
    """推理阶段生成的候选分析方案"""
    plan_id: str = Field(..., description="方案ID")
    hypothesis: str = Field(..., description="投资假设")
    analysis_approach: str = Field(..., description="分析方法")
    expected_outcome: str = Field(..., description="预期结果")
    confidence_level: float = Field(
        ..., 
        description="置信度(0-1)",
        ge=0.0,  # 【Pydantic v2】greater than or equal
        le=1.0,  # 【Pydantic v2】less than or equal
    )
    pros: List[str] = Field(..., description="方案优势", min_length=3)
    cons: List[str] = Field(..., description="方案劣势", min_length=2)


class DecisionOutput(BaseModel):
    """决策阶段选择的最优投资观点"""
    selected_plan_id: str = Field(..., description="选中的方案ID")
    investment_thesis: str = Field(..., description="投资论点")
    supporting_evidence: List[str] = Field(..., description="支持证据")
    risk_assessment: str = Field(..., description="风险评估")
    recommendation: str = Field(..., description="投资建议")
    timeframe: str = Field(..., description="时间框架")


# ============================================================================
# 【LangGraph 状态定义 - 使用 TypedDict】
# ============================================================================
# 【核心概念】
# StateGraph 的状态是一个字典，在节点之间传递和更新
# TypedDict 提供类型提示，但不进行运行时验证
#
# 【LangGraph 状态更新规则】
# 1. 节点函数返回的字典会与当前状态 **合并**（merge）
# 2. 相同 key 的值会被覆盖
# 3. 使用 Annotated 可以自定义合并行为（如 add_messages 用于累积消息）
#
# 【状态设计原则】
# 1. 包含所有节点需要访问的数据
# 2. 包含控制流所需的字段（如 current_phase）
# 3. 使用 Optional 标记可能为空的字段
#
class ResearchAgentState(TypedDict):
    """
    研究智能体的状态
    
    【TypedDict vs Pydantic BaseModel】
    - TypedDict：轻量级类型提示，不进行运行时验证，适合 LangGraph 状态
    - BaseModel：完整的数据验证，适合输入输出结构化
    
    【状态字段分类】
    1. 输入字段：研究主题、行业焦点、时间范围
    2. 处理状态：各阶段的中间结果
    3. 输出字段：最终研究报告
    4. 控制流：当前阶段、错误信息
    """
    # ==================== 输入字段 ====================
    research_topic: str   # 研究主题
    industry_focus: str   # 行业焦点
    time_horizon: str     # 时间范围(短期/中期/长期)
    
    # ==================== 处理状态 ====================
    # Optional 表示字段可以为 None
    # Dict[str, Any] 表示键为字符串，值为任意类型
    perception_data: Optional[Dict[str, Any]]    # 感知阶段收集的数据
    world_model: Optional[Dict[str, Any]]        # 内部世界模型
    reasoning_plans: Optional[List[Dict[str, Any]]]  # 候选分析方案
    selected_plan: Optional[Dict[str, Any]]      # 选中的最优方案
    
    # ==================== 输出字段 ====================
    final_report: Optional[str]  # 最终研究报告
    
    # ==================== 控制流字段 ====================
    # Literal 类型限制字段只能是指定的字符串值之一
    current_phase: Literal["perception", "modeling", "reasoning", "decision", "report", "completed"]
    error: Optional[str]  # 错误信息


# ============================================================================
# 【提示模板定义】
# ============================================================================
# 【ChatPromptTemplate vs PromptTemplate】
# - ChatPromptTemplate：用于聊天模型，支持 system/user/assistant 角色
# - PromptTemplate：用于传统 LLM（已不推荐）
#
# 【模板变量】
# 使用 {variable_name} 格式定义变量，invoke 时传入具体值
#

PERCEPTION_PROMPT = """你是一个专业的投资研究分析师，请收集和整理关于以下研究主题的市场数据和信息：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

请从以下几个方面进行市场感知：
1. 市场概况和最新动态
2. 关键经济和市场指标
3. 近期重要新闻（至少3条）
4. 行业趋势分析（至少针对3个细分领域）

根据你的专业知识和经验，提供尽可能详细和准确的信息。

输出格式要求为JSON，包含以下字段：
- market_overview: 字符串
- key_indicators: 字典，键为指标名称，值为指标值和简要解释
- recent_news: 字符串列表，每项为一条重要新闻
- industry_trends: 字典，键为细分领域，值为趋势分析
"""

MODELING_PROMPT = """你是一个资深投资策略师，请根据以下市场数据和信息，构建市场内部模型，进行深度分析：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场数据和信息:
{perception_data}

请构建一个全面的市场内部模型，包括：
1. 当前市场状态评估
2. 经济周期判断
3. 主要风险因素（至少3个）
4. 潜在机会领域（至少3个）
5. 市场情绪分析

输出格式要求为JSON，包含以下字段：
- market_state: 字符串
- economic_cycle: 字符串
- risk_factors: 字符串列表
- opportunity_areas: 字符串列表
- market_sentiment: 字符串
"""

REASONING_PROMPT = """你是一个战略投资顾问，请根据以下市场模型，生成3个不同的投资分析方案：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场内部模型:
{world_model}

请为每个方案提供：
1. 方案ID（简短标识符）
2. 投资假设
3. 分析方法
4. 预期结果
5. 置信度（0-1之间的小数）
6. 方案优势（至少3点）
7. 方案劣势（至少2点）

这些方案应该有明显的差异，代表不同的投资思路或分析角度。

输出格式要求为JSON数组，每个元素包含以下字段：
- plan_id: 字符串
- hypothesis: 字符串
- analysis_approach: 字符串
- expected_outcome: 字符串
- confidence_level: 浮点数
- pros: 字符串列表
- cons: 字符串列表
"""

DECISION_PROMPT = """你是一个投资决策委员会主席，请评估以下候选分析方案，选择最优方案并形成投资决策：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场内部模型:
{world_model}

候选分析方案:
{reasoning_plans}

请基于方案的假设、分析方法、预期结果、置信度以及优缺点，选择最优的投资方案，并给出详细的决策理由。
你的决策应该综合考虑投资潜力、风险水平和时间框架的匹配度。

输出格式要求为JSON，包含以下字段：
- selected_plan_id: 字符串
- investment_thesis: 字符串
- supporting_evidence: 字符串列表
- risk_assessment: 字符串
- recommendation: 字符串
- timeframe: 字符串
"""

REPORT_PROMPT = """你是一个专业的投资研究报告撰写人，请根据以下信息生成一份完整的投资研究报告：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场数据和信息:
{perception_data}

市场内部模型:
{world_model}

选定的投资决策:
{selected_plan}

请生成一份结构完整、逻辑清晰的投研报告，包括但不限于：
1. 报告标题和摘要
2. 市场和行业背景
3. 核心投资观点
4. 详细分析论证
5. 风险因素
6. 投资建议
7. 时间框架和预期回报

报告应当专业、客观，同时提供足够的分析深度和洞见。
"""


# ============================================================================
# 【节点函数定义 - LangGraph 的核心】
# ============================================================================
# 【节点函数规范】
# 1. 输入：接收当前状态（TypedDict 定义的类型）
# 2. 输出：返回状态更新（部分状态的字典）
# 3. 返回的字典会与当前状态合并
#
# 【状态更新模式】
# 方式一（推荐）：返回更新字段
#   return {"perception_data": result, "current_phase": "modeling"}
#
# 方式二（旧版，不推荐）：返回完整状态
#   return {**state, "perception_data": result, ...}
#
# 【LangChain >= 1.0 Chain 调用方式】
# 旧版：chain = prompt | llm | parser; result = chain.invoke(input)
# 新版：同上，但推荐使用 with_structured_output() 替代 JsonOutputParser
#

# ============================================================================
# 【辅助函数 - JSON 解析】
# ============================================================================
def parse_json_output(text: str) -> Any:
    """
    解析 LLM 输出的 JSON 字符串
    
    【为什么需要这个函数？】
    1. LLM 可能输出带 markdown 代码块的 JSON（```json ... ```）
    2. LLM 可能输出不合法的 JSON
    3. 需要统一的错误处理
    
    【LangChain >= 1.0 更好的方式】
    使用 with_structured_output() 直接获取结构化数据：
        llm.with_structured_output(PerceptionOutput).invoke(prompt)
    但这里保持原有逻辑以便学习 JSON 解析过程
    """
    import re
    
    # 尝试提取 markdown 代码块中的 JSON
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1)
    
    # 清理常见问题
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试修复常见问题
        # 移除尾部逗号
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return json.loads(text)


# ============================================================================
# 第一阶段：感知 - 收集市场数据和信息
# ============================================================================
def perception(state: ResearchAgentState) -> Dict[str, Any]:
    """
    感知阶段：收集和整理市场数据和信息
    
    【节点函数返回值说明】
    返回的字典会与当前状态合并，所以只需要返回需要更新的字段
    
    【状态更新示例】
    当前状态: {"research_topic": "新能源", "current_phase": "perception", ...}
    返回: {"perception_data": {...}, "current_phase": "modeling"}
    合并后: {"research_topic": "新能源", "perception_data": {...}, "current_phase": "modeling", ...}
    """
    
    print("1. 感知阶段：收集市场数据和信息...")
    
    try:
        # ==================== 创建提示模板 ====================
        # ChatPromptTemplate.from_template() 创建单消息模板（默认角色为 user）
        prompt = ChatPromptTemplate.from_template(PERCEPTION_PROMPT)
        
        # ==================== 准备输入数据 ====================
        input_data = {
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"]
        }
        
        # ==================== 构建 Chain 并调用 ====================
        # 【LangChain LCEL 语法】
        # | 操作符用于连接组件，形成处理链
        # prompt | llm | parser 等价于：
        #   1. prompt.invoke(input) -> 格式化提示
        #   2. llm.invoke(formatted_prompt) -> 调用 LLM
        #   3. parser.invoke(llm_output) -> 解析输出
        #
        # 【LangChain >= 1.0 变化】
        # StrOutputParser 仍然可用，但更推荐：
        # - 结构化输出：llm.with_structured_output(Model)
        # - JSON 输出：llm.with_structured_output(dict)
        
        # 先获取字符串输出
        chain = prompt | llm | StrOutputParser()
        result_text = chain.invoke(input_data)
        
        # 解析 JSON
        result = parse_json_output(result_text)
        
        # ==================== 返回状态更新 ====================
        # 只返回需要更新的字段，LangGraph 会自动合并
        return {
            "perception_data": result,
            "current_phase": "modeling"
        }
        
    except Exception as e:
        # 错误处理：返回错误信息，保持在当前阶段
        return {
            "error": f"感知阶段出错: {str(e)}",
            "current_phase": "perception"
        }


# ============================================================================
# 第二阶段：建模 - 构建内部世界模型
# ============================================================================
def modeling(state: ResearchAgentState) -> Dict[str, Any]:
    """
    建模阶段：构建内部世界模型，理解市场状态
    
    【数据依赖】
    此阶段依赖 perception_data，需要确保感知阶段已完成
    """
    
    print("2. 建模阶段：构建内部世界模型...")
    
    try:
        # ==================== 数据校验 ====================
        # 使用 state.get() 而非 state["key"]，避免 KeyError
        # 如果 perception_data 不存在，返回 None
        perception_data = state.get("perception_data")
        if not perception_data:
            return {
                "error": "建模阶段缺少感知数据",
                "current_phase": "perception"  # 回退到感知阶段
            }
        
        # ==================== 创建 Chain ====================
        prompt = ChatPromptTemplate.from_template(MODELING_PROMPT)
        
        input_data = {
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "perception_data": json.dumps(perception_data, ensure_ascii=False, indent=2)
        }
        
        chain = prompt | llm | StrOutputParser()
        result_text = chain.invoke(input_data)
        result = parse_json_output(result_text)
        
        return {
            "world_model": result,
            "current_phase": "reasoning"
        }
        
    except Exception as e:
        return {
            "error": f"建模阶段出错: {str(e)}",
            "current_phase": "modeling"
        }


# ============================================================================
# 第三阶段：推理 - 生成候选分析方案
# ============================================================================
def reasoning(state: ResearchAgentState) -> Dict[str, Any]:
    """
    推理阶段：生成多个候选分析方案并模拟结果
    
    【核心思想】
    深思熟虑型智能体的关键特点：
    不是直接给出一个答案，而是生成多个候选方案进行比较
    """
    
    print("3. 推理阶段：生成候选分析方案...")
    
    try:
        world_model = state.get("world_model")
        if not world_model:
            return {
                "error": "推理阶段缺少世界模型",
                "current_phase": "modeling"
            }
        
        prompt = ChatPromptTemplate.from_template(REASONING_PROMPT)
        
        input_data = {
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "world_model": json.dumps(world_model, ensure_ascii=False, indent=2)
        }
        
        chain = prompt | llm | StrOutputParser()
        result_text = chain.invoke(input_data)
        result = parse_json_output(result_text)
        
        return {
            "reasoning_plans": result,
            "current_phase": "decision"
        }
        
    except Exception as e:
        return {
            "error": f"推理阶段出错: {str(e)}",
            "current_phase": "reasoning"
        }


# ============================================================================
# 第四阶段：决策 - 选择最优方案
# ============================================================================
def decision(state: ResearchAgentState) -> Dict[str, Any]:
    """
    决策阶段：评估候选方案并选择最优投资观点
    
    【决策依据】
    综合考虑：
    1. 投资潜力（预期收益）
    2. 风险水平（风险因素）
    3. 时间框架匹配度
    4. 置信度
    """
    
    print("4. 决策阶段：选择最优投资观点...")
    
    try:
        reasoning_plans = state.get("reasoning_plans")
        if not reasoning_plans:
            return {
                "error": "决策阶段缺少候选方案",
                "current_phase": "reasoning"
            }
        
        prompt = ChatPromptTemplate.from_template(DECISION_PROMPT)
        
        input_data = {
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "world_model": json.dumps(state["world_model"], ensure_ascii=False, indent=2),
            "reasoning_plans": json.dumps(reasoning_plans, ensure_ascii=False, indent=2)
        }
        
        chain = prompt | llm | StrOutputParser()
        result_text = chain.invoke(input_data)
        result = parse_json_output(result_text)
        
        return {
            "selected_plan": result,
            "current_phase": "report"
        }
        
    except Exception as e:
        return {
            "error": f"决策阶段出错: {str(e)}",
            "current_phase": "decision"
        }


# ============================================================================
# 第五阶段：报告 - 生成完整研究报告
# ============================================================================
def report_generation(state: ResearchAgentState) -> Dict[str, Any]:
    """
    报告阶段：生成完整的投资研究报告
    
    【报告整合】
    将前面所有阶段的结果整合成一份完整报告：
    - 感知数据 → 市场背景
    - 世界模型 → 分析框架
    - 选定方案 → 核心观点
    """
    
    print("5. 报告阶段：生成完整研究报告...")
    
    try:
        selected_plan = state.get("selected_plan")
        if not selected_plan:
            return {
                "error": "报告阶段缺少选定方案",
                "current_phase": "decision"
            }
        
        prompt = ChatPromptTemplate.from_template(REPORT_PROMPT)
        
        input_data = {
            "research_topic": state["research_topic"],
            "industry_focus": state["industry_focus"],
            "time_horizon": state["time_horizon"],
            "perception_data": json.dumps(state["perception_data"], ensure_ascii=False, indent=2),
            "world_model": json.dumps(state["world_model"], ensure_ascii=False, indent=2),
            "selected_plan": json.dumps(selected_plan, ensure_ascii=False, indent=2)
        }
        
        # 报告阶段直接输出文本，不需要 JSON 解析
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke(input_data)
        
        return {
            "final_report": result,
            "current_phase": "completed"
        }
        
    except Exception as e:
        return {
            "error": f"报告生成阶段出错: {str(e)}",
            "current_phase": "report"
        }


# ============================================================================
# 【LangGraph 工作流构建】
# ============================================================================
def create_research_agent_workflow() -> CompiledStateGraph:
    """
    创建深思熟虑型研究智能体工作流图
    
    【LangGraph 核心概念】
    1. StateGraph：状态图，管理节点之间的状态传递
    2. Node：节点，执行具体操作的函数
    3. Edge：边，定义节点之间的转换关系
    4. START/END：特殊的入口和结束节点
    
    【工作流构建步骤】
    1. 创建 StateGraph，指定状态类型
    2. 添加节点（add_node）
    3. 设置入口点（set_entry_point 或 add_edge(START, node)）
    4. 添加边（add_edge）
    5. 编译（compile）
    
    【边的类型】
    1. 普通边：add_edge("A", "B") - A 完成后必定到 B
    2. 条件边：add_conditional_edges("A", router) - A 完成后根据条件选择下一步
    """
    
    # ==================== 创建状态图 ====================
    # StateGraph 需要指定状态类型（TypedDict）
    workflow = StateGraph(ResearchAgentState)
    
    # ==================== 添加节点 ====================
    # add_node(节点名称, 节点函数)
    # 节点函数接收状态，返回状态更新
    workflow.add_node("perception", perception)
    workflow.add_node("modeling", modeling)
    workflow.add_node("reasoning", reasoning)
    workflow.add_node("decision", decision)
    workflow.add_node("report", report_generation)
    
    # ==================== 设置入口点 ====================
    # 【LangGraph >= 0.2 变化】
    # 旧版：workflow.set_entry_point("perception")
    # 新版：workflow.add_edge(START, "perception")
    # 推荐使用新方式，更清晰明确
    workflow.add_edge(START, "perception")
    
    # ==================== 添加边 ====================
    # 普通边：线性流程，A → B
    # 这里我们使用简单的线性流程：感知 → 建模 → 推理 → 决策 → 报告 → 结束
    workflow.add_edge("perception", "modeling")
    workflow.add_edge("modeling", "reasoning")
    workflow.add_edge("reasoning", "decision")
    workflow.add_edge("decision", "report")
    workflow.add_edge("report", END)
    
    # ==================== 条件边示例（已注释）====================
    # 如果需要根据状态决定下一步，可以使用条件边：
    #
    # def should_continue(state: ResearchAgentState) -> str:
    #     """根据当前状态决定下一步"""
    #     if state.get("error"):
    #         return "error_handler"  # 有错误，跳转到错误处理
    #     return state["current_phase"]  # 正常流程
    #
    # workflow.add_conditional_edges(
    #     "perception",
    #     should_continue,
    #     {
    #         "modeling": "modeling",
    #         "error_handler": "error_handler",
    #     }
    # )
    
    # ==================== 编译工作流 ====================
    # compile() 返回可执行的工作流
    # 编译后可以调用 .invoke(state) 运行
    return workflow.compile()


# ============================================================================
# 【主函数 - 运行智能体】
# ============================================================================
def run_research_agent(topic: str, industry: str, horizon: str) -> Dict[str, Any]:
    """
    运行研究智能体并返回结果
    
    【使用流程】
    1. 创建工作流
    2. 准备初始状态
    3. 调用 invoke() 运行
    """
    
    # 创建工作流
    agent = create_research_agent_workflow()
    
    # ==================== 准备初始状态 ====================
    # 初始状态必须包含 TypedDict 中所有非 Optional 的字段
    # Optional 字段可以设置为 None
    initial_state: ResearchAgentState = {
        # 输入（必填）
        "research_topic": topic,
        "industry_focus": industry,
        "time_horizon": horizon,
        # 处理状态（可选，初始化为 None）
        "perception_data": None,
        "world_model": None,
        "reasoning_plans": None,
        "selected_plan": None,
        # 输出（可选）
        "final_report": None,
        # 控制流
        "current_phase": "perception",
        "error": None,
    }
    
    # ==================== 打印流程图 ====================
    # get_graph() 获取工作流图结构
    # draw_mermaid() 生成 Mermaid 格式的流程图
    # 可以在支持 Mermaid 的编辑器中查看
    print("LangGraph Mermaid流程图：")
    print(agent.get_graph().draw_mermaid())
    
    # ==================== 运行智能体 ====================
    # invoke() 是同步执行方式
    # ainvoke() 是异步执行方式（需要 await）
    result = agent.invoke(initial_state)
    
    return result


# ============================================================================
# 【程序入口】
# ============================================================================
if __name__ == "__main__":
    print("=== 深思熟虑智能体 - 智能投研助手 ===\n")
    print("【使用模型】qwen-turbo-latest（阿里云通义千问）\n")
    print("【核心流程】感知 → 建模 → 推理 → 决策 → 报告\n")
    
    # 检查 API Key
    if not DASHSCOPE_API_KEY:
        print("【错误】请设置环境变量 DASHSCOPE_API_KEY")
        print("示例：export DASHSCOPE_API_KEY='your-api-key'\n")
        exit(1)
    
    # 用户输入
    topic = input("请输入研究主题 (例如: 新能源汽车行业投资机会): ").strip()
    industry = input("请输入行业焦点 (例如: 电动汽车制造、电池技术): ").strip()
    horizon = input("请输入时间范围 [短期/中期/长期]: ").strip()
    
    # 默认值
    if not topic:
        topic = "新能源汽车行业投资机会"
    if not industry:
        industry = "电动汽车制造、电池技术"
    if not horizon:
        horizon = "中期"
    
    print(f"\n研究主题: {topic}")
    print(f"行业焦点: {industry}")
    print(f"时间范围: {horizon}")
    print("\n智能投研助手开始工作...\n")
    
    try:
        # 运行智能体
        result = run_research_agent(topic, industry, horizon)
        
        # 处理结果
        if result.get("error"):
            print(f"\n【发生错误】{result['error']}")
        else:
            print("\n" + "=" * 60)
            print("=== 最终研究报告 ===")
            print("=" * 60 + "\n")
            print(result.get("final_report", "未生成报告"))
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_report_{timestamp}.txt"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result.get("final_report", "未生成报告"))
            
            print(f"\n【报告已保存】{filename}")
            
            # 打印中间结果（调试用）
            print("\n" + "=" * 60)
            print("【调试信息】各阶段输出概览")
            print("=" * 60)
            
            if result.get("perception_data"):
                print("\n1. 感知阶段关键指标:")
                indicators = result["perception_data"].get("key_indicators", {})
                for key, value in list(indicators.items())[:3]:
                    print(f"   - {key}: {value}")
            
            if result.get("world_model"):
                print("\n2. 建模阶段市场状态:")
                print(f"   {result['world_model'].get('market_state', 'N/A')}")
            
            if result.get("reasoning_plans"):
                print(f"\n3. 推理阶段生成 {len(result['reasoning_plans'])} 个候选方案")
            
            if result.get("selected_plan"):
                print("\n4. 决策阶段选定方案:")
                print(f"   方案ID: {result['selected_plan'].get('selected_plan_id', 'N/A')}")
                print(f"   投资建议: {result['selected_plan'].get('recommendation', 'N/A')}")
            
    except Exception as e:
        print(f"\n【运行错误】{str(e)}")
        import traceback
        traceback.print_exc()