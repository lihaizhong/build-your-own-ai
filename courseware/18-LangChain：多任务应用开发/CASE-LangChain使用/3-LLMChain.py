"""
3-LLMChain.py - LangChain 1.29 版本

主要变更:
1. 使用 langchain.agents.create_agent
2. 使用 DuckDuckGo 搜索工具（免费）
3. 使用 Chat Model
"""
import os
import math
import dashscope
from langchain_community.chat_models import ChatTongyi
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun  # DuckDuckGo 搜索工具
from langchain_core.tools import tool
from langchain.agents import create_agent

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key
 
# 加载 Chat 模型
llm = ChatTongyi(model="qwen-turbo", api_key=api_key)  # type: ignore[arg-type]


# DuckDuckGo 搜索工具（免费，无需 API Key）
search = DuckDuckGoSearchRun()


# 计算器工具 - 使用 @tool 装饰器自定义
@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式。
    支持: sqrt, sin, cos, tan, log, exp, pi, e 等
    输入: 数学表达式，如 '2 + 2', 'sqrt(16)', '(74 - 32) * 5/9'
    输出: 计算结果
    """
    try:
        safe_dict = {
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
            'tan': math.tan, 'log': math.log, 'log10': math.log10,
            'exp': math.exp, 'pi': math.pi, 'e': math.e,
            'abs': abs, 'round': round, 'pow': pow,
        }
        expression = expression.replace('^', '**')
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 创建工具列表
tools = [search, calculator]

# 创建 Agent
agent = create_agent(llm, tools=tools)  # type: ignore[arg-type]
 
# 运行 agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "当前北京的温度是多少摄氏度？这个温度的1/4是多少"}]
})

# 打印结果
for message in result["messages"]:
    if hasattr(message, 'content'):
        print(f"{message.__class__.__name__}: {message.content}")
