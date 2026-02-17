"""
4-ConversationChain.py - LangChain >= 1.0 版本

主要变更:
1. ConversationChain 已弃用，改用 RunnableWithMessageHistory
2. 需要显式配置消息历史存储
3. 使用 langchain_core 中的组件
"""
import os
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# 设置通义千问API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 加载 Tongyi 模型
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)

# 创建提示模板 - 包含消息历史占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 创建基础链
chain = prompt | llm

# 消息历史存储（使用内存存储作为示例）
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取或创建会话历史"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 创建带消息历史的链
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 第一轮对话
print("=" * 50)
print("第一轮对话:")
output1 = conversation.invoke(
    {"input": "Hi there!"},
    config={"configurable": {"session_id": "test-session"}}
)
print(output1)

# 第二轮对话 - 可以看到 AI 记住了之前的对话
print("=" * 50)
print("第二轮对话:")
output2 = conversation.invoke(
    {"input": "I'm doing well! Just having a conversation with an AI."},
    config={"configurable": {"session_id": "test-session"}}
)
print(output2)