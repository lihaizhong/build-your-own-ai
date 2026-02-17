"""
5-product_llm.py - LangChain 1.29 版本

主要变更:
1. LLMChain 已弃用，改用 LCEL 管道语法
2. Agent 使用 langchain.agents.create_agent
3. Tool 使用 @tool 装饰器
4. BaseLLM 从 langchain_core.language_models 导入
"""
import textwrap
import time
import os

from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM
from langchain_community.llms import Tongyi
from langchain_community.chat_models import ChatTongyi
from langchain.agents import create_agent

# 定义了LLM的Prompt Template
CONTEXT_QA_TMPL = """
根据以下提供的信息，回答用户的问题
信息：{context}

问题：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

# 输出结果显示，每行最多60字符，每个字符显示停留0.1秒（动态显示效果）
def output_response(response: str) -> None:
    if not response:
        exit(0)
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)
            print(" ", end="", flush=True)
        print()
    print("----------------------------------------------------------------")


# 模拟公司产品和公司介绍的数据源
class TeslaDataSource:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.qa_chain = CONTEXT_QA_PROMPT | llm

    def find_product_description(self, product_name: str) -> str:
        """模拟公司产品的数据库"""
        product_info = {
            "Model 3": "具有简洁、动感的外观设计，流线型车身和现代化前脸。定价23.19-33.19万",
            "Model Y": "在外观上与Model 3相似，但采用了更高的车身和更大的后备箱空间。定价26.39-36.39万",
            "Model X": "拥有独特的翅子门设计和更加大胆的外观风格。定价89.89-105.89万",
        }
        return product_info.get(product_name, "没有找到这个产品")

    def find_company_info(self, query: str) -> str:
        """模拟公司介绍文档数据库"""
        context = """
        特斯拉最知名的产品是电动汽车，其中包括Model S、Model 3、Model X和Model Y等多款车型。
        特斯拉以其技术创新、高性能和领先的自动驾驶技术而闻名。
        """
        return self.qa_chain.invoke({"query": query, "context": context})


# 设置通义千问API密钥（从环境变量获取）
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')

if __name__ == "__main__":
    # 定义 LLM（用于工具内部）
    llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)
    # 定义 Chat Model（用于 Agent）
    chat_llm = ChatTongyi(model="qwen-turbo", api_key=DASHSCOPE_API_KEY)  # type: ignore[arg-type]
    
    # 自有数据
    tesla_data_source = TeslaDataSource(llm)
    
    # 定义工具 - 使用 @tool 装饰器
    @tool
    def query_product(product_name: str) -> str:
        """
        通过产品名称找到产品描述。
        输入: 产品名称（如 Model 3, Model Y, Model X）
        输出: 产品描述信息
        """
        return tesla_data_source.find_product_description(product_name)
    
    @tool
    def query_company_info(query: str) -> str:
        """
        查询公司相关信息。
        输入: 关于公司的问题
        输出: 公司信息回答
        """
        return tesla_data_source.find_company_info(query)
    
    tools = [query_product, query_company_info]
    
    # 创建 Agent - LangChain 1.29 方式
    agent = create_agent(chat_llm, tools=tools)  # type: ignore[arg-type]

    # 主过程：可以一直提问下去，直到Ctrl+C
    while True:
        try:
            user_input = input("请输入您的问题：")
            result = agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            # 获取最后一条消息（AI 的回复）
            last_message = result["messages"][-1]
            output_response(last_message.content)
        except KeyboardInterrupt:
            break