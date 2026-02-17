"""
2-LLMChain.py - LangChain 1.29 版本

主要变更:
1. initialize_agent 已完全移除，改用 langchain.agents.create_agent
2. 工具从 langchain_community.tools 导入
3. 使用 Chat Model
4. agent 返回的是 Graph 的执行结果
"""
import os
import dashscope
from langchain_community.chat_models import ChatTongyi  # 使用 Chat Model
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun  # DuckDuckGo 搜索工具
from langchain.agents import create_agent

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY', '')
dashscope.api_key = api_key

# 加载 Chat 模型（LangGraph 需要使用 Chat Model）
llm = ChatTongyi(model="qwen-turbo", api_key=api_key)  # type: ignore[arg-type]


# 使用 DuckDuckGo 搜索工具（免费，无需 API Key）
search = DuckDuckGoSearchRun()

# 创建工具列表
tools = [search]

# 创建 ReAct Agent - LangChain 1.29 方式
agent = create_agent(llm, tools=tools)  # type: ignore[arg-type]

# 运行 agent
# LangGraph 使用 invoke 方法，传入 messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "今天是几月几号?历史上的今天有哪些名人出生"}]
})

# 提取最终回复
for message in result["messages"]:
    if hasattr(message, 'content'):
        print(f"{message.__class__.__name__}: {message.content}")
