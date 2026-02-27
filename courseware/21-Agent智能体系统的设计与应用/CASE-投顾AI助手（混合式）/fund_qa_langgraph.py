#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
私募基金运作指引问答助手 - 反应式智能体实现

适合反应式架构的私募基金问答助手，使用Agent模式实现主动思考和工具选择。
适配 LangChain 1.0+ 和 LangGraph 0.3+ 版本。

【LangChain 1.0+ 重大变化】
1. AgentExecutor 已被移除，改用 LangGraph StateGraph
2. 推荐使用 @tool 装饰器定义工具
3. 新架构基于 StateGraph，更加灵活和可扩展

【LangGraph 核心概念】
- StateGraph: 状态图，定义 Agent 的执行流程
- Node: 节点，执行具体操作（如调用 LLM、执行工具）
- Edge: 边，定义节点间的跳转逻辑
- Conditional Edge: 条件边，根据状态决定下一步
"""

import os
import re
from typing import Annotated, Dict
from pydantic import SecretStr

# ==================== LangChain 1.0+ 核心导入 ====================
# 【关键知识点】LangChain 1.0+ 将核心组件迁移到 langchain_core 模块

# HumanMessage: 用户消息类型
# AIMessage: AI 消息类型
# 【类型安全】使用消息对象而非字典，满足类型检查器要求
from langchain_core.messages import HumanMessage

# PromptTemplate: 标准提示模板，支持变量替换
from langchain_core.prompts import PromptTemplate

# 【变化】@tool 装饰器替代 Tool 类，更简洁且自动生成工具描述
from langchain_core.tools import tool

# BaseLanguageModel: 语言模型的基类，用于类型注解
from langchain_core.language_models import BaseLanguageModel

# ==================== LangGraph 导入 ====================
# 【LangGraph】LangChain 官方的状态机框架，用于构建复杂的 Agent 工作流

# StateGraph: 状态图的核心类
from langgraph.graph import StateGraph, END

# add_messages: 消息累加器，用于状态中的消息列表
# 【关键】Annotated[List, add_messages] 表示消息列表会自动累加
from langgraph.graph.message import add_messages

# ToolNode: 预置的工具执行节点
# 【便捷】封装了工具调用逻辑，开箱即用
from langgraph.prebuilt import ToolNode

# 【变化】推荐使用 Chat 模型而非普通 LLM
from langchain_community.chat_models import ChatTongyi

# ==================== 配置 ====================
# 【最佳实践】敏感信息应从环境变量获取
# 【类型安全】使用 SecretStr 包装，防止敏感信息被意外打印
_api_key_value = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_API_KEY: SecretStr = SecretStr(_api_key_value) if _api_key_value else SecretStr("")

# ==================== 知识库数据 ====================
# 【设计思考】这是一个简化的知识库，实际项目中可以接入向量数据库
FUND_RULES_DB = [
    {
        "id": "rule001",
        "category": "设立与募集",
        "question": "私募基金的合格投资者标准是什么？",
        "answer": "合格投资者是指具备相应风险识别能力和风险承担能力，投资于单只私募基金的金额不低于100万元且符合下列条件之一的单位和个人：\n1. 净资产不低于1000万元的单位\n2. 金融资产不低于300万元或者最近三年个人年均收入不低于50万元的个人"
    },
    {
        "id": "rule002",
        "category": "设立与募集",
        "question": "私募基金的最低募集规模要求是多少？",
        "answer": "私募证券投资基金的最低募集规模不得低于人民币1000万元。对于私募股权基金、创业投资基金等其他类型的私募基金，监管规定更加灵活，通常需符合基金合同的约定。"
    },
    {
        "id": "rule014",
        "category": "监管规定",
        "question": "私募基金管理人的风险准备金要求是什么？",
        "answer": "私募证券基金管理人应当按照管理费收入的10%计提风险准备金，主要用于赔偿因管理人违法违规、违反基金合同、操作错误等给基金财产或者投资者造成的损失。"
    }
]

# ==================== 提示词模板 ====================
# 【核心概念】Prompt Engineering 是 AI 应用的基础技能

# 上下文问答模板 - 用于基于检索到的信息回答问题
CONTEXT_QA_TMPL = """
你是私募基金问答助手。请根据以下信息回答问题：

信息：{context}
问题：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

# 超出知识库范围模板 - 处理知识边界问题
# 【重要】这是 RAG 系统的关键设计：明确告知用户知识的边界，避免幻觉
OUTSIDE_KNOWLEDGE_TMPL = """
你是私募基金问答助手。用户的问题是关于私募基金的，但我们的知识库中没有直接相关的信息。
请首先明确告知用户"对不起，在我的知识库中没有关于[具体主题]的详细信息"，
然后，如果你有相关知识，可以以"根据我的经验"或"一般来说"等方式提供一些通用信息，
并建议用户查阅官方资料或咨询专业人士获取准确信息。

用户问题：{query}
缺失的知识主题：{missing_topic}
"""
OUTSIDE_KNOWLEDGE_PROMPT = PromptTemplate(
    input_variables=["query", "missing_topic"],
    template=OUTSIDE_KNOWLEDGE_TMPL,
)


# ==================== 数据源类 ====================
class FundRulesDataSource:
    """
    私募基金规则数据源，提供多种查询工具
    
    【设计模式】这是工具层的核心，封装了所有数据访问逻辑
    Agent 通过调用这些方法来获取信息
    """
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.rules_db = FUND_RULES_DB

    def search_rules_by_keywords(self, keywords: str) -> str:
        """
        通过关键词搜索相关私募基金规则
        
        【算法思路】
        1. 分词：将输入按分隔符拆分成关键词列表
        2. 匹配：遍历所有规则，计算每个规则的匹配分数
        3. 排序：按匹配分数降序排列
        4. 返回：返回得分最高的前2条规则
        """
        keywords = keywords.strip().lower()
        keyword_list = re.split(r'[,，\s]+', keywords)
        
        matched_rules = []
        for rule in self.rules_db:
            rule_text = (rule["category"] + " " + rule["question"]).lower()
            match_count = sum(1 for kw in keyword_list if kw in rule_text)
            if match_count > 0:
                matched_rules.append((rule, match_count))
        
        matched_rules.sort(key=lambda x: x[1], reverse=True)
        
        if not matched_rules:
            return "未找到与关键词相关的规则。"
        
        result = []
        for rule, _ in matched_rules[:2]:
            result.append(f"类别: {rule['category']}\n问题: {rule['question']}\n答案: {rule['answer']}")
        
        return "\n\n".join(result)

    def search_rules_by_category(self, category: str) -> str:
        """根据规则类别查询私募基金规则"""
        category = category.strip()
        matched_rules = []
        
        for rule in self.rules_db:
            if category.lower() in rule["category"].lower():
                matched_rules.append(rule)
        
        if not matched_rules:
            return f"未找到类别为 '{category}' 的规则。"
        
        result = []
        for rule in matched_rules:
            result.append(f"问题: {rule['question']}\n答案: {rule['answer']}")
        
        return "\n\n".join(result)

    def answer_question(self, query: str) -> str:
        """
        直接回答用户关于私募基金的问题
        
        【核心算法】基于词重叠的相似度匹配
        1. 将问题和规则分词
        2. 计算词集合的交集大小
        3. 计算相似度得分 = 交集大小 / 问题词数
        4. 选择得分最高的规则
        """
        query = query.strip()
        
        best_rule = None
        best_score = 0
        
        for rule in self.rules_db:
            query_words = set(query.lower().split())
            rule_words = set((rule["question"] + " " + rule["category"]).lower().split())
            common_words = query_words.intersection(rule_words)
            
            score = len(common_words) / max(1, len(query_words))
            if score > best_score:
                best_score = score
                best_rule = rule
        
        if best_score < 0.2 or best_rule is None:
            # 识别问题主题
            missing_topic = self._identify_missing_topic(query)
            prompt = OUTSIDE_KNOWLEDGE_PROMPT.format(
                query=query,
                missing_topic=missing_topic
            )
            # 【LangChain 1.0+ 变化】使用 invoke 方法
            response = self.llm.invoke(prompt)
            # Chat 模型返回 AIMessage，需要提取 content
            content = response.content if hasattr(response, 'content') else str(response)
            return f"这个问题超出了知识库范围。\n\n{content}"
        
        context = best_rule["answer"]
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        
        # 【LangChain 1.0+ 变化】使用 invoke 方法
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def _identify_missing_topic(self, query: str) -> str:
        """识别查询中缺失的知识主题"""
        query = query.lower()
        if "投资" in query and "资产" in query:
            return "私募基金可投资的资产类别"
        elif "公募" in query and "区别" in query:
            return "私募基金与公募基金的区别"
        elif "退出" in query and ("机制" in query or "方式" in query):
            return "创业投资基金的退出机制"
        elif "费用" in query and "结构" in query:
            return "私募基金的费用结构"
        elif "托管" in query:
            return "私募基金资产托管"
        return "您所询问的具体主题"


# ==================== Agent 提示词模板 ====================
# 【ReAct 模式】Reasoning + Acting
# Agent 通过 Thought-Action-Observation 循环来解决问题
AGENT_TMPL = """你是一个私募基金问答助手，请根据用户的问题选择合适的工具来回答。

你可以使用以下工具：

{tools}

按照以下格式回答问题：

---
Question: 用户的问题
Thought: 我需要思考如何回答这个问题
Action: 工具名称
Action Input: 工具的输入
Observation: 工具返回的结果
...（这个思考/行动/行动输入/观察可以重复几次）
Thought: 现在我知道答案了
Final Answer: 给用户的最终答案
---

注意：
1. 如果知识库中没有相关信息，请明确告知用户"对不起，在我的知识库中没有关于[具体主题]的详细信息"
2. 如果你基于自己的知识提供补充信息，请用"根据我的经验"或"一般来说"等前缀明确标识
3. 回答要专业、简洁、准确

Question: {input}
{agent_scratchpad}
"""


# ==================== Agent 状态定义 ====================
# 【LangGraph 核心】StateGraph 需要一个状态类来传递数据
# 【关键】Annotated[List, add_messages] 表示消息列表会自动累加新消息
class AgentState(Dict):
    """
    Agent 状态定义
    
    【设计思路】
    - messages: 消息历史，包含用户问题和 AI 回复
    - add_messages: 自动累加消息，而不是覆盖
    """
    messages: Annotated[list, add_messages]


def create_fund_qa_agent():
    """
    创建私募基金问答 Agent（LangGraph 版本）
    
    【LangGraph 工作流程】
    1. 定义 LLM 和工具
    2. 创建 StateGraph，定义状态
    3. 添加节点（agent, tools）
    4. 设置入口点和条件边
    5. 编译图
    
    【架构图】
    用户输入 → agent（思考+选择工具）→ tools（执行）→ agent（观察结果）
             ↑                                              ↓
             ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
                           循环直到没有工具调用
    """
    
    # ========== 1. 创建 LLM ==========
    # 【变化】使用 ChatTongyi 替代 Tongyi
    # Chat 模型更适合 Agent 场景，支持 tool_calls
    llm = ChatTongyi(
        model="qwen-turbo",
        api_key=DASHSCOPE_API_KEY,  # SecretStr 类型
        # 【注意】ChatTongyi 不直接支持 temperature 参数
        # 需要通过 model_kwargs 传递
        model_kwargs={"temperature": 0.7},
    )
    
    # ========== 2. 创建数据源 ==========
    fund_rules_source = FundRulesDataSource(llm)
    
    # ========== 3. 定义工具（@tool 装饰器）==========
    # 【LangChain 1.0+ 推荐方式】使用 @tool 装饰器
    # 相比 Tool 类，这种方式更简洁，且自动生成工具描述
    
    @tool
    def keywords_search(keywords: str) -> str:
        """当需要通过关键词搜索私募基金规则时使用，输入应为相关关键词"""
        return fund_rules_source.search_rules_by_keywords(keywords)
    
    @tool
    def category_query(category: str) -> str:
        """当需要查询特定类别的私募基金规则时使用，输入应为类别名称。类别名称有两种：设立与募集, 监管规定"""
        return fund_rules_source.search_rules_by_category(category)
    
    @tool
    def answer_question(query: str) -> str:
        """当能够直接回答用户问题时使用，输入应为完整的用户问题"""
        return fund_rules_source.answer_question(query)
    
    tools = [keywords_search, category_query, answer_question]
    
    # ========== 4. 绑定工具到 LLM ==========
    # 【关键】bind_tools 让 LLM 知道有哪些工具可用
    # LLM 会自动生成 tool_calls 而不是文本输出
    llm_with_tools = llm.bind_tools(tools)
    
    # ========== 5. 构建系统提示 ==========
    system_prompt = """你是一个私募基金问答助手，请根据用户的问题选择合适的工具来回答。

你可以使用以下工具：
1. keywords_search: 当需要通过关键词搜索私募基金规则时使用，输入应为相关关键词
2. category_query: 当需要查询特定类别的私募基金规则时使用，输入应为类别名称。类别名称有两种：设立与募集, 监管规定
3. answer_question: 当能够直接回答用户问题时使用，输入应为完整的用户问题

注意：
1. 如果知识库中没有相关信息，请明确告知用户"对不起，在我的知识库中没有关于[具体主题]的详细信息"
2. 如果你基于自己的知识提供补充信息，请用"根据我的经验"或"一般来说"等前缀明确标识
3. 回答要专业、简洁、准确
"""
    
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    # 【LCEL 语法】使用 ChatPromptTemplate 创建提示
    # MessagesPlaceholder 用于插入消息历史
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # 【LCEL 链式调用】prompt | llm_with_tools
    # 等价于 llm_with_tools.invoke(prompt.invoke(state))
    agent_chain = prompt | llm_with_tools
    
    # ========== 6. 创建工具节点 ==========
    # ToolNode 是预置的工具执行节点，封装了工具调用逻辑
    tool_node = ToolNode(tools)
    
    # ========== 7. 定义 Agent 节点函数 ==========
    def agent_node(state: AgentState) -> dict:
        """
        Agent 推理节点
        
        【工作流程】
        1. 调用 LLM（带工具绑定）
        2. 返回新的消息（可能是 AIMessage 或带 tool_calls 的消息）
        """
        response = agent_chain.invoke(state)
        return {"messages": [response]}
    
    # ========== 8. 定义路由函数 ==========
    def should_continue(state: AgentState) -> str:
        """
        判断是否需要继续调用工具
        
        【路由逻辑】
        - 如果 LLM 返回了 tool_calls → 跳转到 tools 节点
        - 否则 → 结束（END）
        """
        messages = state.get("messages", [])
        if not messages:
            return END
        
        last_message = messages[-1]
        
        # 检查是否有工具调用
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        return END
    
    # ========== 9. 构建状态图 ==========
    # 【LangGraph 核心】StateGraph 定义 Agent 的执行流程
    # type: ignore: StateGraph 泛型类型推断问题，实际运行正常
    workflow = StateGraph(AgentState)  # type: ignore[arg-type]
    
    # 添加节点
    # type: ignore: add_node 泛型类型推断问题，实际运行正常
    workflow.add_node("agent", agent_node)  # type: ignore[arg-type]
    workflow.add_node("tools", tool_node)  # type: ignore[arg-type]
    
    # 设置入口点：从 agent 节点开始
    workflow.set_entry_point("agent")
    
    # 【条件边】根据 should_continue 的返回值决定下一步
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # 需要调用工具 → tools 节点
            END: END,          # 不需要调用工具 → 结束
        }
    )
    
    # 【普通边】tools 节点执行后返回 agent 节点
    workflow.add_edge("tools", "agent")
    
    # 【编译】将图编译为可执行的应用
    app = workflow.compile()
    
    return app, llm


def run_agent(app, user_input: str) -> str:
    """
    运行 Agent 并返回结果
    
    【调用方式】
    - stream: 流式输出，逐步返回结果
    - invoke: 同步调用，一次性返回结果
    """
    inputs = {"messages": [HumanMessage(content=user_input)]}
    
    # 收集最终响应
    final_response = ""
    
    # 【流式输出】stream 方法逐步返回执行结果
    # stream_mode="updates" 表示每次节点更新时返回
    # type: ignore: 泛型类型推断问题，实际运行正常
    for chunk in app.stream(inputs, stream_mode="updates"):  # type: ignore[arg-type]
        for node_name, node_output in chunk.items():
            if node_name == "agent":
                messages = node_output.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "content") and msg.content:
                        final_response = msg.content
    
    return final_response


if __name__ == "__main__":
    # 创建 Agent
    fund_qa_agent, llm = create_fund_qa_agent()
    
    print("=== 私募基金运作指引问答助手（反应式智能体）===\n")
    print("使用模型：qwen-turbo (LangChain 1.0+ / LangGraph)\n")
    print("您可以提问关于私募基金的各类问题，输入'退出'结束对话\n")
    
    # 主循环
    while True:
        try:
            user_input = input("请输入您的问题：")
            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("感谢使用，再见！")
                break
            
            response = run_agent(fund_qa_agent, user_input)
            print(f"回答: {response}\n")
            print("-" * 40)
        except KeyboardInterrupt:
            print("\n程序已中断，感谢使用！")
            break
        except Exception as e:
            print(f"发生错误：{e}")
            print("请尝试重新提问或更换提问方式。")


# ==================== 附录：旧版 API 对比 ====================
"""
【LangChain 0.x 旧版 API】（已废弃）

from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from langchain.schema import AgentOutputParser
from langchain_community.llms import Tongyi

# 创建 LLM
llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=KEY)

# 定义工具
tools = [
    Tool(name="关键词搜索", func=search_func, description="..."),
]

# 创建 Agent
agent = LLMSingleActionAgent(llm_chain=chain, output_parser=parser, ...)

# 创建执行器
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)

# 调用
response = agent_executor.run(user_input)


【LangChain 1.0+ / LangGraph 新版 API】

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi

# 创建 LLM
llm = ChatTongyi(model="qwen-turbo", api_key=KEY)

# 定义工具
@tool
def my_tool(input: str) -> str:
    '''工具描述'''
    return result

# 构建 StateGraph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {...})
workflow.add_edge("tools", "agent")
app = workflow.compile()

# 调用
for chunk in app.stream({"messages": [HumanMessage(content="...")]}):
    print(chunk)
"""
