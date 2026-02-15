# LangChain：多任务应用开发

LangChain 组成：

- Models：模型 -- **langchain_community.llms**
- Prompts：提示词，包括提示词管理、提示词优化和提示词序列化 -- **langchain.prompts**
- Memory：记忆，用来保存和模型交互时的上下文
- Indexes：索引，用于结构化文档，方便和模型交互
- Chains：链，一系列对各种组件的调用
- Agents：代理，决定模型采用哪些行动，执行并且观察流程，直到完成为止

---

