#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合智能体（Hybrid Agent）- 财富管理投顾AI助手

基于LangGraph实现的混合型智能体，结合反应式架构的即时响应能力和深思熟虑架构的长期规划能力，
通过协调层动态切换处理模式，提供智能化财富管理咨询服务。

三层架构：
1. 底层（反应式）：即时响应客户查询，提供快速反馈
2. 中层（协调）：评估任务类型和优先级，动态选择处理模式
3. 顶层（深思熟虑）：进行复杂的投资分析和长期财务规划

【LangChain 1.0+ 重大变化说明】
========================================
1. Agent API 变化：
   - AgentExecutor、LLMSingleActionAgent 已被移除
   - 新增 create_agent() 作为创建 Agent 的主要入口
   - Agent 返回 CompiledStateGraph，支持流式输出

2. 模型 API 变化：
   - 推荐 ChatTongyi 替代 Tongyi（Chat 模型更适合 Agent 场景）
   - 调用方式改为 invoke() 方法
   - 返回 AIMessage 对象，需通过 .content 获取内容

3. 工具定义变化：
   - 推荐使用 @tool 装饰器替代 Tool 类
   - 工具描述从 docstring 自动提取

4. 类型系统变化：
   - AgentOutputParser 移至 langchain_core.output_parsers.BaseOutputParser
   - AgentAction/AgentFinish 移至 langchain_core.agents
   - 需要使用泛型类型参数

5. Pydantic 变化：
   - langchain_core.pydantic_v1 已废弃
   - 直接使用 pydantic v2 的 BaseModel, Field, field_validator
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Literal, TypedDict, Optional, Union, cast

from pydantic import BaseModel, Field, field_validator, SecretStr

# ==================== LangChain 1.0+ 核心导入 ====================
# 【关键知识点】LangChain 1.0+ 采用模块化架构
# - langchain_core: 核心接口和基础类
# - langchain_community: 第三方集成（如 Tongyi）
# - langchain: 高级 API（如 create_agent）

# ChatPromptTemplate: 支持角色对话的提示模板
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser: 将 LLM 输出转为字符串
# JsonOutputParser: 将 LLM 输出解析为 JSON
# PydanticOutputParser: 将 LLM 输出解析为 Pydantic 模型
# BaseOutputParser: 所有输出解析器的基类
# 【LangChain 1.0+ 变化】所有输出解析器都在 langchain_core.output_parsers 中
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    BaseOutputParser,
)

# AgentAction: Agent 决定执行的动作（调用哪个工具）
# AgentFinish: Agent 完成任务，返回最终答案
# 【变化】从 langchain.schema 移至 langchain_core.agents
from langchain_core.agents import AgentAction, AgentFinish

# @tool 装饰器: 推荐的工具定义方式
from langchain_core.tools import tool

# HumanMessage: 用户消息类型
from langchain_core.messages import HumanMessage

# 【推荐】Chat 模型比普通 LLM 更适合 Agent 场景
from langchain_community.chat_models import ChatTongyi

# LangGraph 状态图
from langgraph.graph import StateGraph, END

# 【LangChain 1.0+ 新 API】create_agent 替代旧的 AgentExecutor
from langchain.agents import create_agent

import requests
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
# 【最佳实践】敏感信息从环境变量获取，使用 SecretStr 包装防止意外泄露
_api_key_value = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_API_KEY: SecretStr = SecretStr(_api_key_value) if _api_key_value else SecretStr("")

# 创建 LLM 实例
# 【LangChain 1.0+ 变化】使用 ChatTongyi 替代 Tongyi
# Chat 模型支持角色对话，更适合 Agent 场景
llm = ChatTongyi(
    model="qwen-turbo",
    api_key=DASHSCOPE_API_KEY,
    model_kwargs={"temperature": 0.7},
)


# ==================== 数据模型定义 ====================
# 【Pydantic v2 变化】
# - validator 装饰器改为 field_validator
# - 使用 mode="before" 或 mode="after" 指定验证时机

class CustomerProfile(BaseModel):
    """客户画像信息
    
    【设计思考】使用 Pydantic 模型的好处：
    1. 数据验证：自动验证字段类型和约束
    2. 序列化：支持 JSON/dict 转换
    3. 文档生成：自动生成 OpenAPI schema
    """
    customer_id: str = Field(..., description="客户ID")
    risk_tolerance: Literal["保守型", "稳健型", "平衡型", "成长型", "进取型"] = Field(
        ..., description="风险承受能力"
    )
    investment_horizon: Literal["短期", "中期", "长期"] = Field(..., description="投资期限")
    financial_goals: List[str] = Field(..., description="财务目标")
    investment_preferences: List[str] = Field(..., description="投资偏好")
    portfolio_value: float = Field(..., description="投资组合总价值")
    current_allocations: Dict[str, float] = Field(..., description="当前资产配置")


class EmergencyResponseOutput(BaseModel):
    """紧急查询的即时响应"""
    response_type: str = Field(..., description="响应类型")
    direct_answer: str = Field(..., description="直接回答")
    data_points: Optional[Dict[str, Any]] = Field(None, description="相关数据点")
    suggested_actions: Optional[List[str]] = Field(None, description="建议操作")


class InvestmentAnalysisOutput(BaseModel):
    """深度投资分析结果"""
    market_assessment: str = Field(..., description="市场评估")
    portfolio_analysis: Dict[str, Any] = Field(..., description="投资组合分析")
    recommendations: List[Dict[str, Any]] = Field(..., description="投资建议")
    risk_analysis: Dict[str, Any] = Field(..., description="风险分析")
    expected_outcomes: Dict[str, Any] = Field(..., description="预期结果")


# ==================== 状态定义 ====================
# 【LangGraph 核心概念】State 是节点间传递数据的容器
# TypedDict 定义状态的结构，提供类型安全

class WealthAdvisorState(TypedDict):
    """财富顾问智能体的状态
    
    【设计原则】
    1. 状态是不可变的：每个节点返回新的状态，不修改原状态
    2. 状态是累加的：新状态合并旧状态的字段
    3. 状态驱动流程：根据状态字段决定分支走向
    """
    # 输入
    user_query: str  # 用户查询
    customer_profile: Optional[Dict[str, Any]]  # 客户画像
    
    # 处理状态
    query_type: Optional[Literal["emergency", "informational", "analytical"]]  # 查询类型
    processing_mode: Optional[Literal["reactive", "deliberative"]]  # 处理模式
    emergency_response: Optional[Dict[str, Any]]  # 紧急响应结果
    market_data: Optional[Dict[str, Any]]  # 市场数据
    analysis_results: Optional[Dict[str, Any]]  # 分析结果
    
    # 输出
    final_response: Optional[str]  # 最终响应
    
    # 控制流
    current_phase: Optional[str]
    error: Optional[str]  # 错误信息


# ==================== 提示模板 ====================
# 【Prompt Engineering 最佳实践】
# 1. 明确角色和任务
# 2. 提供清晰的输出格式要求
# 3. 给出示例（Few-shot Learning）

ASSESSMENT_PROMPT = """你是一个财富管理投顾AI助手的协调层。请评估以下用户查询，确定其类型和应该采用的处理模式。

用户查询: {user_query}

请判断:
1. 查询类型: 
   - "emergency": 紧急的或直接的查询，需要立即响应（如市场状况、账户信息、产品信息等）
   - "informational": 信息性的查询，需要特定领域知识（如税务政策、投资工具介绍等）
   - "analytical": 需要深度分析的查询（如投资组合优化、长期理财规划等）

2. 建议的处理模式:
   - "reactive": 适用于需要快速反应的查询
   - "deliberative": 适用于需要深度思考和分析的查询

请以JSON格式返回结果，包含以下字段:
- query_type: 查询类型（上述三种类型之一）
- processing_mode: 处理模式（上述两种模式之一）
- reasoning: 决策理由的简要说明
"""

REACTIVE_PROMPT = """你是一个财富管理投顾AI助手，专注于提供快速准确的响应。请针对用户的查询提供直接的回答。

用户查询: {user_query}

客户信息:
{customer_profile}

请提供:
1. 直接回答用户问题
2. 相关的关键数据点（如适用）
3. 建议的后续操作（如适用）

以JSON格式返回响应，包含以下字段:
- response_type: 响应类型
- direct_answer: 直接回答
- data_points: 相关数据点（可选）
- suggested_actions: 建议操作（可选）
"""

DATA_COLLECTION_PROMPT = """你是一个财富管理投顾AI助手的数据收集模块。基于以下用户查询，确定需要收集哪些市场和财务数据进行深入分析。

用户查询: {user_query}

客户信息:
{customer_profile}

请确定需要收集的数据类型，例如:
- 资产类别表现数据
- 经济指标
- 行业趋势
- 历史回报率
- 风险指标
- 税收信息
- 其他相关数据

以JSON格式返回结果，包含以下字段:
- required_data_types: 需要收集的数据类型列表
- data_sources: 建议的数据来源列表
- collected_data: 模拟收集的数据（为简化示例，请生成合理的模拟数据）
"""

ANALYSIS_PROMPT = """你是一个财富管理投顾AI助手的分析引擎。请根据收集的数据对用户的投资情况进行深入分析。

用户查询: {user_query}

客户信息:
{customer_profile}

市场数据:
{market_data}

请提供全面的投资分析，包括:
1. 当前市场状况评估
2. 客户投资组合分析
3. 个性化投资建议
4. 风险评估
5. 预期结果和回报预测

以JSON格式返回分析结果，包含以下字段:
- market_assessment: 市场评估
- portfolio_analysis: 投资组合分析
- recommendations: 投资建议列表
- risk_analysis: 风险分析
- expected_outcomes: 预期结果
"""

RECOMMENDATION_PROMPT = """你是一个财富管理投顾AI助手。请根据深入分析结果，为客户准备最终的咨询建议。

用户查询: {user_query}

客户信息:
{customer_profile}

分析结果:
{analysis_results}

请提供专业、个性化且详细的投资建议，语言应友好易懂，避免过多专业术语。建议应包括:
1. 总体投资策略
2. 具体行动步骤
3. 资产配置建议
4. 风险管理策略
5. 时间框架
6. 预期收益
7. 后续跟进计划

返回格式应为自然语言文本，适合直接呈现给客户。
"""


# ==================== 工具定义 ====================
def query_shanghai_index(_: str = "") -> str:
    """上证指数实时查询工具（模拟版），返回固定的行情数据
    
    【实际应用】在生产环境中，这里应该调用真实的行情 API
    例如：新浪财经、东方财富、Wind 等数据源
    """
    # 直接返回模拟数据，避免外部 API 不可用导致报错
    name = "上证指数"
    price = "3125.62"
    change = "6.32"
    pct = "0.20"
    result = f"{name} 当前点位: {price}，涨跌: {change}，涨跌幅: {pct}%"
    print('result=', result)
    return result


# ==================== 节点函数定义 ====================
# 【LangGraph 核心概念】每个节点是一个纯函数
# 输入：当前状态
# 输出：状态的更新部分（会与原状态合并）

def assess_query(state: WealthAdvisorState) -> WealthAdvisorState:
    """评估用户查询，确定类型和处理模式
    
    【协调层核心功能】
    这是混合智能体的"大脑"，负责：
    1. 理解用户意图
    2. 判断查询复杂度
    3. 选择合适的处理路径
    
    【LangChain 1.0+ 调用方式】
    - 使用 | 操作符构建链：prompt | llm | parser
    - 使用 invoke() 方法执行
    """
    print("[DEBUG] 进入节点: assess_query")
    
    try:
        # 准备提示模板
        prompt = ChatPromptTemplate.from_template(ASSESSMENT_PROMPT)
        
        # 构建输入数据
        input_data = {
            "user_query": state["user_query"],
        }
        
        # 【LangChain 1.0+ LCEL 语法】
        # 使用管道操作符 | 连接组件
        # 等价于：prompt.invoke(input) -> llm.invoke(prompt_output) -> parser.invoke(llm_output)
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke(input_data)
        
        print("[DEBUG] LLM评估输出:", result)
        print(f"[DEBUG] 分支判断: processing_mode={result.get('processing_mode', '未知')}, query_type={result.get('query_type', '未知')}")
        
        # 获取处理模式，确保有值
        processing_mode = result.get("processing_mode", "reactive")
        if processing_mode not in ["reactive", "deliberative"]:
            processing_mode = "reactive"  # 默认使用反应式处理
        
        # 获取查询类型，确保有值
        query_type = result.get("query_type", "emergency")
        if query_type not in ["emergency", "informational", "analytical"]:
            query_type = "emergency"  # 默认为紧急查询
        
        # 返回状态更新
        # 【重要】返回的是要更新的字段，会与原状态合并
        return {
            **state,
            "query_type": query_type,
            "processing_mode": processing_mode,
        }
    except Exception as e:
        return {
            **state,
            "error": f"评估阶段出错: {str(e)}",
            "final_response": "评估查询时发生错误，无法处理您的请求。"
        }


def reactive_processing(state: WealthAdvisorState) -> WealthAdvisorState:
    """反应式处理模式，提供快速响应
    
    【反应式架构特点】
    1. 快速响应：简单查询直接处理，不经过复杂的推理链
    2. 工具调用：可以调用外部工具获取实时数据
    3. 直接输出：生成可直接呈现给用户的答案
    
    【LangChain 1.0+ Agent 创建方式】
    使用 create_agent() 替代旧的 AgentExecutor + LLMSingleActionAgent
    新方式更简洁，基于 LangGraph 的 StateGraph 实现
    """
    print("[DEBUG] 进入节点: reactive_processing")
    
    try:
        # ========== 1. 定义工具 ==========
        # 【LangChain 1.0+ 推荐方式】使用 @tool 装饰器
        # 相比 Tool 类，这种方式更简洁，且自动生成工具描述
        
        @tool
        def shanghai_index_query(query: str) -> str:
            """用于查询上证指数的最新行情，输入内容可为空或任意字符串
            
            【@tool 装饰器说明】
            - 函数名成为工具名
            - docstring 成为工具描述
            - 参数类型提示用于生成工具 schema
            """
            return query_shanghai_index(query)
        
        tools = [shanghai_index_query]
        
        # ========== 2. 创建 Agent ==========
        # 【LangChain 1.0+ 新 API】
        # create_agent 返回 CompiledStateGraph，支持流式输出
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""你是一个财富管理投顾AI助手，请根据用户的问题选择合适的工具来回答。

注意：
1. 如果用户询问上证指数相关信息，请使用"shanghai_index_query"工具
2. 回答要专业、简洁、准确
3. 如果不需要使用工具，可以直接给出答案""",
            debug=True,
        )
        
        # ========== 3. 运行 Agent ==========
        # 【LangChain 1.0+ 新的调用方式】
        # 输入格式：{"messages": [HumanMessage(content="...")]}
        user_query = state["user_query"]
        inputs = {"messages": [HumanMessage(content=user_query)]}
        
        # 使用 invoke 方法同步执行
        result = agent.invoke(inputs)  # type: ignore[arg-type]
        
        # 提取最终响应
        # 【LangChain 1.0+ 变化】结果格式为 {"messages": [...]}
        # 最后一条消息是 AI 的回复
        last_message = result["messages"][-1]
        final_response = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        return {
            **state,
            "final_response": final_response
        }
    except Exception as e:
        return {
            **state,
            "error": f"反应式处理出错: {str(e)}",
            "final_response": "处理您的查询时发生错误，无法提供响应。"
        }


def collect_data(state: WealthAdvisorState) -> WealthAdvisorState:
    """收集市场数据和客户信息进行深入分析
    
    【深思熟虑架构 - 第一阶段】
    这是深思熟虑模式的起点，负责收集分析所需的数据
    
    【数据收集策略】
    1. 识别查询涉及的数据类型
    2. 从多个数据源获取信息
    3. 数据清洗和预处理
    """
    print("[DEBUG] 进入节点: collect_data")
    
    try:
        # 准备提示
        prompt = ChatPromptTemplate.from_template(DATA_COLLECTION_PROMPT)
        
        # 构建输入
        input_data = {
            "user_query": state["user_query"],
            "customer_profile": json.dumps(state.get("customer_profile", {}), ensure_ascii=False, indent=2)
        }
        
        # 【LCEL 链式调用】
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke(input_data)
        
        # 更新状态
        return {
            **state,
            "market_data": result.get("collected_data", {}),
            "current_phase": "analyze"
        }
    except Exception as e:
        return {
            **state,
            "error": f"数据收集阶段出错: {str(e)}",
            "current_phase": "collect_data"  # 保持在当前阶段
        }


def analyze_data(state: WealthAdvisorState) -> WealthAdvisorState:
    """进行深度投资分析
    
    【深思熟虑架构 - 第二阶段】
    对收集的数据进行深入分析，生成投资建议
    
    【分析方法】
    1. 市场状况评估
    2. 投资组合分析
    3. 风险评估
    4. 收益预测
    """
    print("[DEBUG] 进入节点: analyze_data")
    
    try:
        # 确保必要数据已收集
        if not state.get("market_data"):
            return {
                **state,
                "error": "分析阶段缺少市场数据",
                "current_phase": "collect_data"  # 回到数据收集阶段
            }
        
        # 准备提示
        prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
        
        # 构建输入
        input_data = {
            "user_query": state["user_query"],
            "customer_profile": json.dumps(state.get("customer_profile", {}), ensure_ascii=False, indent=2),
            "market_data": json.dumps(state.get("market_data", {}), ensure_ascii=False, indent=2)
        }
        
        # 调用 LLM
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke(input_data)
        
        # 更新状态
        return {
            **state,
            "analysis_results": result,
            "current_phase": "recommend"
        }
    except Exception as e:
        return {
            **state,
            "error": f"分析阶段出错: {str(e)}",
            "current_phase": "analyze"  # 保持在当前阶段
        }


def generate_recommendations(state: WealthAdvisorState) -> WealthAdvisorState:
    """生成投资建议和行动计划
    
    【深思熟虑架构 - 第三阶段】
    将分析结果转化为客户可理解的建议
    
    【建议生成原则】
    1. 个性化：基于客户画像定制
    2. 可操作性：给出明确的行动步骤
    3. 风险提示：充分披露风险
    """
    print("[DEBUG] 进入节点: generate_recommendations")
    
    try:
        # 确保分析结果已存在
        if not state.get("analysis_results"):
            return {
                **state,
                "error": "建议生成阶段缺少分析结果",
                "current_phase": "analyze"  # 回到分析阶段
            }
        
        # 准备提示
        prompt = ChatPromptTemplate.from_template(RECOMMENDATION_PROMPT)
        
        # 构建输入
        input_data = {
            "user_query": state["user_query"],
            "customer_profile": json.dumps(state.get("customer_profile", {}), ensure_ascii=False, indent=2),
            "analysis_results": json.dumps(state.get("analysis_results", {}), ensure_ascii=False, indent=2)
        }
        
        # 调用 LLM
        # 【注意】这里使用 StrOutputParser，因为我们需要自然语言输出
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke(input_data)
        
        # 更新状态
        return {
            **state,
            "final_response": result,
            "current_phase": "respond"
        }
    except Exception as e:
        return {
            **state,
            "error": f"建议生成阶段出错: {str(e)}",
            "current_phase": "recommend"  # 保持在当前阶段
        }


# ==================== 工作流构建 ====================
def create_wealth_advisor_workflow() -> StateGraph:
    """创建财富顾问混合智能体工作流
    
    【LangGraph 核心概念】
    1. StateGraph：状态图，定义节点和边
    2. Node：节点，处理状态的函数
    3. Edge：边，定义节点间的转移
    4. Conditional Edge：条件边，根据状态决定下一节点
    
    【工作流设计模式】
    这是典型的"路由器模式"：
    1. 入口节点评估请求
    2. 根据评估结果路由到不同分支
    3. 各分支处理后汇聚到出口
    """
    
    # 创建状态图
    # 【泛型参数】指定状态类型，提供类型安全
    workflow = StateGraph(WealthAdvisorState)
    
    # 添加节点
    # 【节点命名】使用动词或名词短语，表示节点的作用
    workflow.add_node("assess", assess_query)
    workflow.add_node("reactive", reactive_processing)
    workflow.add_node("collect_data", collect_data)
    workflow.add_node("analyze", analyze_data)
    workflow.add_node("recommend", generate_recommendations)
    
    # 定义响应节点
    def respond_function(state: WealthAdvisorState) -> WealthAdvisorState:
        """最终响应生成节点，原样返回状态
        
        【设计思考】这个节点的作用是：
        1. 作为所有路径的汇聚点
        2. 可以在这里添加后处理逻辑（如日志记录）
        """
        if not state.get("final_response"):
            state = {
                **state,
                "final_response": "无法生成响应。请检查处理流程。",
                "error": state.get("error", "未知错误")
            }
        return state
    
    workflow.add_node("respond", respond_function)
    
    # 设置入口点
    # 【流程起点】所有请求从 assess 节点开始
    workflow.set_entry_point("assess")
    
    # 添加条件边
    # 【路由逻辑】根据 processing_mode 决定下一节点
    # lambda 函数接收状态，返回目标节点名称
    workflow.add_conditional_edges(
        "assess",
        lambda x: "reactive" if x.get("processing_mode") == "reactive" else "collect_data",
        {
            "reactive": "reactive",
            "collect_data": "collect_data"
        }
    )
    
    # 添加固定路径边
    # 【反应式路径】reactive -> respond -> END
    workflow.add_edge("reactive", "respond")
    
    # 【深思熟虑路径】collect_data -> analyze -> recommend -> respond -> END
    workflow.add_edge("collect_data", "analyze")
    workflow.add_edge("analyze", "recommend")
    workflow.add_edge("recommend", "respond")
    
    # 所有路径最终都到达 END
    workflow.add_edge("respond", END)
    
    # 编译工作流
    # 【编译过程】将图定义转换为可执行的状态机
    return workflow.compile()


# ==================== 示例数据 ====================
SAMPLE_CUSTOMER_PROFILES = {
    "customer1": {
        "customer_id": "C10012345",
        "risk_tolerance": "平衡型",
        "investment_horizon": "中期",
        "financial_goals": ["退休规划", "子女教育金"],
        "investment_preferences": ["ESG投资", "科技行业"],
        "portfolio_value": 1500000.0,
        "current_allocations": {
            "股票": 0.40,
            "债券": 0.30,
            "现金": 0.10,
            "另类投资": 0.20
        }
    },
    "customer2": {
        "customer_id": "C10067890",
        "risk_tolerance": "进取型",
        "investment_horizon": "长期",
        "financial_goals": ["财富增长", "资产配置多元化"],
        "investment_preferences": ["新兴市场", "高成长行业"],
        "portfolio_value": 3000000.0,
        "current_allocations": {
            "股票": 0.65,
            "债券": 0.15,
            "现金": 0.05,
            "另类投资": 0.15
        }
    }
}


# ==================== 运行函数 ====================
def run_wealth_advisor(user_query: str, customer_id: str = "customer1") -> Dict[str, Any]:
    """运行财富顾问智能体并返回结果
    
    【执行流程】
    1. 创建工作流实例
    2. 准备初始状态
    3. 调用 invoke 执行
    4. 返回最终状态
    """
    
    # 创建工作流
    agent = create_wealth_advisor_workflow()
    
    # 获取客户画像
    customer_profile = SAMPLE_CUSTOMER_PROFILES.get(customer_id, SAMPLE_CUSTOMER_PROFILES["customer1"])
    
    # 准备初始状态
    # 【状态初始化】所有可选字段初始化为 None
    initial_state: WealthAdvisorState = {
        "user_query": user_query,
        "customer_profile": customer_profile,
        "query_type": None,
        "processing_mode": None,
        "emergency_response": None,
        "market_data": None,
        "analysis_results": None,
        "final_response": None,
        "current_phase": "assess",
        "error": None
    }
    
    try:
        # 打印流程图（用于学习调试）
        print("LangGraph Mermaid流程图：")
        print(agent.get_graph().draw_mermaid())

        # 运行智能体
        result = agent.invoke(initial_state)
        return result
    except Exception as e:
        error_msg = str(e)
        print(f"捕获异常: {error_msg}")
        return {
            **initial_state,
            "error": f"执行过程中发生错误: {error_msg}",
            "final_response": "很抱歉，处理您的请求时出现了问题。"
        }


# ==================== 主函数 ====================
if __name__ == "__main__":
    print("=== 混合智能体 - 财富管理投顾AI助手 ===\n")
    print("使用模型：Qwen-Turbo (LangChain 1.0+)\n")
    print("【架构说明】")
    print("- 反应式模式：快速响应简单查询（如行情查询）")
    print("- 深思熟虑模式：深度分析复杂问题（如投资规划）\n")
    print("-" * 50 + "\n")
    
    # 示例查询
    SAMPLE_QUERIES = [
        # 紧急/简单查询 - 适合反应式处理
        "今天上证指数的表现如何？",
        "我的投资组合中科技股占比是多少？",
        "请解释一下什么是ETF？",
        
        # 分析性查询 - 适合深思熟虑处理
        "根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？",
        "考虑到我的退休目标，请评估我当前的投资策略并提供优化建议。",
        "我想为子女准备教育金，请帮我设计一个10年期的投资计划。"
    ]
    
    # 用户选择查询示例或输入自定义查询
    print("请选择一个示例查询或输入您自己的查询:\n")
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"{i}. {query}")
    print("0. 输入自定义查询")
    
    choice = input("\n请输入选项数字(0-6): ")
    
    if choice == "0":
        user_query = input("请输入您的查询: ")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(SAMPLE_QUERIES):
                user_query = SAMPLE_QUERIES[idx]
            else:
                print("无效选择，使用默认查询")
                user_query = SAMPLE_QUERIES[0]
        except ValueError:
            print("无效输入，使用默认查询")
            user_query = SAMPLE_QUERIES[0]
    
    # 选择客户
    customer_id = "customer1"  # 默认客户
    customer_choice = input("\n选择客户 (1: 平衡型投资者, 2: 进取型投资者): ")
    if customer_choice == "2":
        customer_id = "customer2"
    
    print(f"\n用户查询: {user_query}")
    print(f"选择客户: {SAMPLE_CUSTOMER_PROFILES[customer_id]['risk_tolerance']} 投资者")
    print("\n正在处理...\n")
    
    try:
        # 运行智能体
        start_time = datetime.now()
        result = run_wealth_advisor(user_query, customer_id)
        end_time = datetime.now()
        
        # 如果有错误，显示错误信息并退出
        if result.get("error"):
            print(f"处理过程中发生错误: {result['error']}")
            print(f"\n最终响应: {result.get('final_response', '未能生成响应')}")
            process_time = (end_time - start_time).total_seconds()
            print(f"\n处理用时: {process_time:.2f}秒")
            exit(1)
        
        # 显示处理模式
        process_mode = result.get("processing_mode", "未知")
        if process_mode == "reactive":
            print("【处理模式: 反应式】- 快速响应简单查询")
        else:
            print("【处理模式: 深思熟虑】- 深度分析复杂查询")
        
        # 显示结果
        print("\n=== 响应结果 ===\n")
        print(result.get("final_response", "未生成响应"))
        
        # 显示处理时间
        process_time = (end_time - start_time).total_seconds()
        print(f"\n处理用时: {process_time:.2f}秒")
        
    except Exception as e:
        print(f"\n运行过程中发生意外错误: {str(e)}")


# ==================== 附录：LangChain 0.x vs 1.0+ API 对比 ====================
"""
【旧版 API（LangChain 0.x）】- 已废弃，仅作学习参考

# 旧版导入
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain_community.llms import Tongyi

# 旧版 LLM
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=API_KEY)

# 旧版工具定义
tools = [
    Tool(
        name="上证指数查询",
        func=query_shanghai_index,
        description="用于查询上证指数的最新行情"
    ),
]

# 旧版 Agent 创建
agent = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=llm, prompt=prompt),
    output_parser=CustomOutputParser(),
    stop=["\nObservation:"],
    allowed_tools=["上证指数查询"],
)

# 旧版执行器
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)

# 旧版调用
result = agent_executor.run(user_query)


【新版 API（LangChain 1.0+）】

# 新版导入
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatTongyi

# 新版 LLM（推荐 Chat 模型）
llm = ChatTongyi(model="qwen-turbo", api_key=API_KEY)

# 新版工具定义（@tool 装饰器）
@tool
def shanghai_index_query(query: str) -> str:
    '''用于查询上证指数的最新行情'''
    return query_shanghai_index(query)

# 新版 Agent 创建
agent = create_agent(
    model=llm,
    tools=[shanghai_index_query],
    system_prompt="你是一个投顾助手...",
)

# 新版调用
inputs = {"messages": [HumanMessage(content=user_query)]}
result = agent.invoke(inputs)
final_response = result["messages"][-1].content


【核心变化总结】
1. LLM: Tongyi -> ChatTongyi（Chat 模型更适合 Agent）
2. 工具: Tool 类 -> @tool 装饰器
3. Agent: LLMSingleActionAgent + AgentExecutor -> create_agent()
4. 调用: .run() -> .invoke({"messages": [...]})
5. 结果: 直接字符串 -> {"messages": [...]} 结构
"""