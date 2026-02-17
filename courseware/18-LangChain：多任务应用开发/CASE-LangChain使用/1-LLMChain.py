"""
1-LLMChain.py - LangChain >= 1.0 版本

主要变更:
1. PromptTemplate 仍然可用，但推荐使用 ChatPromptTemplate
2. LLMChain 已弃用，改用 LCEL 管道语法 (prompt | llm)
3. 使用 invoke() 方法而不是 run()
"""
import os
import dashscope
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key
 
# 加载 Tongyi 模型
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)  # 使用通义千问qwen-turbo模型

# 创建Prompt Template (LangChain 1.0 仍然支持)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# LangChain 1.0 推荐用法：LCEL 管道语法
# 将 prompt 和 llm 组合成一个"可运行序列"
chain = prompt | llm

# 使用 invoke 方法传入输入
result1 = chain.invoke({"product": "colorful socks"})
print(result1)

result2 = chain.invoke({"product": "广告设计"})
print(result2)