# Text2SQL：自助式数据报表开发

步骤

- 自然语言理解
- 模式链接
- SQL 生成
- SQL 执行

---

|  | SQLDatabase | coding for self |
| ---- | ---- | ---- |
| 方式 | LangChain 框架 | LLM + RAG |
| 具体方式 | 提供 sql chain、prompt、retriever、tools、agent，让用户通过自然语言，执行 SQL 查询 | 选择合适的 LLM、RAG，可以分成：向量数据库检索 + 固定文件 |
| 优点 | 使用方便，自动通过数据库连接，获取数据库的 metadata | 重心在于 RAG 的提供上，准确性高，配置灵活 |
| 缺点 | 执行不灵活，需要多次判断哪个表适合；复杂查询很难胜任，对于复杂查询通过率低 | 需要设置的条件规则多 |

---

```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# 需要设置LLM
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# Task: 描述数据表
agent_executor.run("描述与订单相关的表及其关系")

# 为提高准确率
# 可以给 Agent 配备专有的知识库，在 prompt 中动态完善和 query 相关的 context
```
