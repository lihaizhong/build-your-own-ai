#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
私募基金运作指引问答助手 - 反应式智能体实现

适合反应式架构的私募基金问答助手，使用Agent模式实现主动思考和工具选择。
适配 LangChain 1.0+ 和 LangGraph 0.3+ 版本。

【LangChain 1.0+ 重大变化】
1. AgentExecutor 已被移除，改用 create_agent 或 LangGraph
2. AgentOutputParser 已被移除，使用 BaseOutputParser
3. Tool 类仍可用，但推荐使用 @tool 装饰器
4. 新架构基于 StateGraph，更加灵活和可扩展
"""

import re
import os
from typing import Union
from pydantic import SecretStr

# ==================== LangChain 1.0+ 核心导入 ====================
# 【关键知识点】LangChain 1.0+ 将核心组件迁移到 langchain_core 模块
# 这种模块化设计使得依赖更清晰，也方便其他框架集成

# 【变化】@tool 装饰器替代 Tool 类，更简洁且自动生成工具描述
from langchain_core.tools import tool

# PromptTemplate: 标准提示模板，支持变量替换
from langchain_core.prompts import PromptTemplate

# BaseLanguageModel: 语言模型的基类，用于类型注解
from langchain_core.language_models import BaseLanguageModel

# BaseOutputParser: 输出解析器基类
# 【注意】AgentOutputParser 在 LangChain 1.0+ 中已被移除
from langchain_core.output_parsers import BaseOutputParser

# AgentAction: 表示 Agent 决定执行的动作（调用哪个工具）
# AgentFinish: 表示 Agent 完成任务，返回最终答案
from langchain_core.agents import AgentAction, AgentFinish

# HumanMessage: 用户消息类型，用于构建对话输入
# 【类型安全】使用消息对象而非字典，满足类型检查器要求
from langchain_core.messages import HumanMessage

# 【变化】推荐使用 Chat 模型而非普通 LLM
# Chat 模型支持角色对话，更适合 Agent 场景
from langchain_community.chat_models import ChatTongyi

# 【LangChain 1.0+ 新 API】create_agent: 创建 Agent 的主要入口
from langchain.agents import create_agent

# ==================== 配置 ====================
# 【最佳实践】敏感信息（如 API Key）应从环境变量获取
# 【类型安全】使用 SecretStr 包装，防止敏感信息被意外打印或记录
_api_key_value = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_API_KEY: SecretStr = SecretStr(_api_key_value) if _api_key_value else SecretStr("")

# ==================== 知识库数据 ====================
# 【设计思考】这是一个简化的知识库，实际项目中可以接入：
# - 向量数据库（如 Milvus、Pinecone）
# - 关系数据库（如 MySQL、PostgreSQL）
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
# 好的提示词模板能显著提升 LLM 的输出质量

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
# 【重要】这是 RAG 系统的关键设计：明确告知用户知识的边界
# 避免 LLM 编造信息（幻觉问题）
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

    # 工具1：通过关键词搜索相关规则
    def search_rules_by_keywords(self, keywords: str) -> str:
        """通过关键词搜索相关私募基金规则
        
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

    # 工具2：根据规则类别查询
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

    # 工具3：直接回答用户问题
    def answer_question(self, query: str) -> str:
        """直接回答用户关于私募基金的问题
        
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
            # 【LangChain 1.0+ 变化】使用 invoke 方法替代直接调用
            # Chat 模型返回 AIMessage 对象，需要提取 content 属性
            response = self.llm.invoke(prompt)
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


# ==================== 自定义输出解析器 ====================
class CustomOutputParser(BaseOutputParser[Union[AgentAction, AgentFinish]]):
    """
    自定义输出解析器 - 处理 LLM 输出的各种情况
    
    【这是 Agent 最关键也是最复杂的部分！】
    
    【为什么输出解析如此重要？】
    1. LLM 的输出是不可控的，可能不遵循预期格式
    2. 需要将 LLM 的文本输出转换为程序可处理的对象
    3. 边界情况处理决定了 Agent 的鲁棒性
    
    【LangChain 1.0+ 变化】
    - AgentOutputParser 已被移除
    - 需要继承 BaseOutputParser[Union[AgentAction, AgentFinish]]
    - 泛型参数指定 parse 方法的返回类型
    """
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """解析 LLM 输出，返回 Agent 动作或最终结果"""
        
        # ========== 边界情况1：Final Answer 格式 ==========
        # 这是最理想的结束状态，LLM 正确遵循了格式
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
            
        # ========== 边界情况2：直接道歉型回答 ==========
        # 【重要】LLM 有时会跳过工具调用，直接给出回答
        if llm_output.strip().startswith("对不起") or llm_output.strip().startswith("抱歉"):
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=f"Direct response detected: {llm_output}"
            )
            
        # ========== 边界情况3：知识边界声明 ==========
        # 【关键】这是防止幻觉的重要机制
        knowledge_boundary_phrases = [
            "在我的知识库中没有",
            "超出了我的知识范围",
            "我没有相关信息",
            "根据我的经验"
        ]
        
        for phrase in knowledge_boundary_phrases:
            if phrase in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.strip()},
                    log=f"Knowledge boundary response detected: {llm_output}"
                )

        # ========== 正常解析：Action + Action Input ==========
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        if not match:
            # ========== 边界情况4：长非结构化响应 ==========
            # 如果 LLM 输出了较长的内容但不符合格式，可能是直接给出了详细回答
            if len(llm_output.strip()) > 50:
                return AgentFinish(
                    return_values={"output": llm_output.strip()},
                    log=f"Long unstructured response detected: {llm_output}"
                )
            # ========== 边界情况5：真正无法解析 ==========
            raise ValueError(f"无法解析LLM输出: `{llm_output}`")
        
        action = match.group(1).strip()
        action_input = match.group(2)
        
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
    @property
    def _type(self) -> str:
        return "custom_agent_output_parser"


# ==================== Agent 创建函数 ====================
def create_fund_qa_agent():
    """
    创建私募基金问答助手 Agent
    
    【LangChain 1.0+ 新架构】
    使用 create_agent 替代旧的 AgentExecutor
    新架构基于 LangGraph 的 StateGraph 实现
    
    【核心变化】
    1. 使用 @tool 装饰器定义工具
    2. 使用 create_agent 创建 Agent
    3. Agent 返回 CompiledStateGraph，支持流式输出
    """
    
    # ========== 1. 创建 LLM ==========
    # 【变化】使用 ChatTongyi 替代 Tongyi
    # Chat 模型更适合 Agent 场景，支持角色对话
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
    
    # ========== 4. 创建 Agent（LangChain 1.0+ 新 API）==========
    # 【create_agent 参数说明】
    # - model: 语言模型实例
    # - tools: 工具列表
    # - system_prompt: 系统提示词
    # - debug: 是否打印调试信息
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""你是一个私募基金问答助手，请根据用户的问题选择合适的工具来回答。

注意：
1. 如果知识库中没有相关信息，请明确告知用户"对不起，在我的知识库中没有关于[具体主题]的详细信息"
2. 如果你基于自己的知识提供补充信息，请用"根据我的经验"或"一般来说"等前缀明确标识
3. 回答要专业、简洁、准确""",
        debug=True,  # 打印详细执行过程，便于学习调试
    )
    
    return agent, tools


if __name__ == "__main__":
    # 创建 Agent
    agent, tools = create_fund_qa_agent()
    
    print("=== 私募基金运作指引问答助手（反应式智能体）===\n")
    print("使用模型：qwen-turbo (LangChain 1.0+)\n")
    print("您可以提问关于私募基金的各类问题，输入'退出'结束对话\n")
    
    # 主循环
    while True:
        try:
            user_input = input("请输入您的问题：")
            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("感谢使用，再见！")
                break
            
            # ========== LangChain 1.0+ 新的调用方式 ==========
            # agent 是 CompiledStateGraph，使用 stream 或 invoke 方法
            # 输入格式：{"messages": [HumanMessage(content="...")]}
            print("\n--- Agent 执行过程 ---")
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            # 【流式输出】stream 方法逐步返回执行结果
            # stream_mode="updates" 表示每次节点更新时返回
            # type: ignore: 泛型类型推断问题，实际运行正常
            for chunk in agent.stream(inputs, stream_mode="updates"):  # type: ignore[arg-type]
                for node_name, node_output in chunk.items():
                    print(f"\n[{node_name}]")
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            msg_type = type(msg).__name__
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                            if len(content) > 200:
                                content = content[:200] + "..."
                            print(f"  {msg_type}: {content}")
            
            # 【同步调用】invoke 方法一次性返回完整结果
            print("\n--- 最终回答 ---")
            result = agent.invoke(inputs)  # type: ignore[arg-type]
            last_message = result["messages"][-1]
            print(f"回答: {last_message.content}\n")
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
llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=DASHSCOPE_API_KEY)

# 定义工具
tools = [
    Tool(
        name="关键词搜索",
        func=fund_rules_source.search_rules_by_keywords,
        description="当需要通过关键词搜索私募基金规则时使用",
    ),
]

# 创建 Agent
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

# 创建执行器
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# 调用
response = agent_executor.run(user_input)


【LangChain 1.0+ 新版 API】

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi

# 创建 LLM
llm = ChatTongyi(model="qwen-turbo", api_key=DASHSCOPE_API_KEY)

# 定义工具
@tool
def my_tool(input: str) -> str:
    '''工具描述'''
    return result

# 创建 Agent
agent = create_agent(model=llm, tools=[my_tool], system_prompt="...")

# 调用（流式）
for chunk in agent.stream({"messages": [HumanMessage(content="...")]}):
    print(chunk)

# 调用（同步）
result = agent.invoke({"messages": [HumanMessage(content="...")]})
"""
