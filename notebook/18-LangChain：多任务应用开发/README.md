# LangChain：多任务应用开发

LangChain 组成：

- Models：模型 -- `langchain_community.llms`
- Prompts：提示词，包括提示词管理、提示词优化和提示词序列化 -- `langchain.prompts`
- Memory：记忆，用来保存和模型交互时的上下文 -- `langchain.memory`
- Indexes：索引，用于结构化文档，方便和模型交互
- Chains：链，一系列对各种组件的调用
- Agents：代理，决定模型采用哪些行动，执行并且观察流程，直到完成为止 -- `langchian.agents`

---

常用 Tools

- Apify
- ArXiv API Tool
- AWS Lambda API
- Shell Tool
- Bing Search
- ChatGPT Plugins
- DuckDuckGo Search
- File System Tools
- Google Places
- Google Places
- Google Search
- Google Serper API
- Gradio Tools
- GraphQL tool
- HuggingFace Tools
- Human as a tool
- IFTTT WebHooks
- Metaphor Search
- Call the API
- Use Metaphor as a tool
- OpenWeatherMap API
- Python REPL
- SceneXplain
- Search Tools
- SearxNG Search API
- SerpAPI
- Twilio
- Wikipedia
- Wolfram Alpha
- YouTubeSearchTool
- Zapier Natural Language Actions API
- Example with SimpleSequentialChain

---

ZERO_SHOT_REACT_DESCRIPTION 表示：
   - Zero-Shot: 零样本学习，不需要示例就能执行任务
   - ReAct: 推理+行动范式，Agent 会先"思考"再"行动"
   - Description: 基于工具的描述来决定使用哪个工具

> 简单说：这个 Agent 会根据工具的描述，自主决定调用哪个工具来完成任务。

---

LangChain 中常用的 AgentType 值

| AgentType | 说明 | 适用场景 |
| ---- | ---- | ---- |
| ZERO_SHOT_REACT_DESCRIPTION | 基于工具描述的 ReAct Agent | 通用场景，最常用 |
| CHAT_ZERO_SHOT_REACT_DESCRIPTION | 针对聊天模型优化的 ReAct | 使用 ChatGPT 等聊天模型时 |
| CONVERSATIONAL_REACT_DESCRIPTION | 带对话记忆的 ReAct Agent | 需要多轮对话的场景 |
| STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION | 支持多输入参数的结构化 Agent | 工具需要复杂参数时 |
| OPENAI_FUNCTIONS | 使用 OpenAI Function Calling | OpenAI 模型专用，效果更好 |
| OPENAI_MULTI_FUNCTIONS | 支持多函数调用的 OpenAI Agent | 需要并行调用多个函数时 |

使用建议

```python
from langchain.agents import AgentType, initialize_agent

# 通用场景（推荐）
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# 使用 OpenAI 模型时（更推荐）
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)

# 需要对话记忆时
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory)
```

✦ 选择建议：如果使用 OpenAI 模型，优先选择 **OPENAI_FUNCTIONS**；其他模型用 **ZERO_SHOT_REACT_DESCRIPTION** 或 **CHAT_ZERO_SHOT_REACT_DESCRIPTION**。

---

几种短期记忆的方式：

- BufferMemory
  - 将之前的对话完全存储下来，传给 LLM
- BufferWindowMemory
  - 最近的 K 组对话存储下来，传给 LLM
- ConversionMemory
  - 对对话进行摘要，将摘要存储在内存中，相当于将压缩过的历史对话传递给 LLM
- VectorStore-backed Memory
  - 将之前所有对话通过向量存储到 VectorDB（向量数据库）中，每次对话，会根据用户的输入信息，匹配向量数据库中最相似的 K 组对话

---

ReAct 范式

将推理和动作相结合，克服 LLM 胡言乱语的问题，同时提高了结果的可解释性和可信赖度。

> ReAct 是大模型的一种工作模式。目的是为了降低大模型的幻觉。

---

Agent 的工作模式主要包括以下几类：

1. ReAct（推理+行动）
   - 核心思想：交替进行推理和行动
   - 流程：思考→ 行动→ 观察→ 循环
   - 代表：LangChain ReAct Agent、原始 ReAct 论文
2. Plan-and-Execute（规划-执行）
   - 核心思想：先规划完整步骤，再依次执行
   - 流程：制定计划 → 拆分子任务 → 逐步执行
   - 代表：BabyAGI、AutoGPT、LangChain Plan-and-Execute Agent
3. Reflection（反思模式）
   - 核心思想：执行后自我反思，迭代改进
   - 流程：生成结果 → 反思 critique → 改进优化 → 输出
   - 代表：Reflexion、Self-Refine
4. Multi-Agent（多智能体协作）
   - 核心思想：多个 Agent 扮演不同角色协作完成任务
   - 模式：
     - 对话协作
     - 角色分工
     - 层级管理
   - 代表：AutoGen、MetaGPT、CrewAI
5. CoT（思维链）
   - 核心思想：显式展示推理步骤
   - 流程：问题 → 中间推理步骤 → 最终答案
   - 适用：数学、逻辑推理任务
6. ToT（思维树）
   - 核心思想：探索多条推理路径，择优选择
   - 流程：生成多个候选 → 评估 → 搜索最优路径
   - 支持：回溯、分支探索
7. Tool Use / Function Calling（工具调用）
   - 核心思想：通过调用外部工具扩展能力边界
   - 能力：搜索、计算、API调用、数据库查询
   - 代表：OpenAI Function Calling、LangChain Tools
8. ReWOO（Reasoning Without Observation）
   - 核心思想：规划和执行解耦，减少观察依赖
   - 优势：提高执行效率，降低 token 消耗
9. RAFT
   - 核心思想：结合前瞻性推理的行动模式
   - 特点：在行动前预判结果

### 模式对比总结

| 模式 | 适用场景 | 优势 | 劣势 |
| --- | --- | --- | --- |
| ReAct | 动态决策任务 | 灵活、适应性强 | 可能陷入循环 |
| Plan-and-Execute | 结构化任务 | 有序、可控 | 缺乏灵活性 |
| Reflection | 高质量输出 | 结果质量高 | 迭代成本高 |
| Multi-Agent | 复杂协作任务 | 专业分工 | 协调复杂 |
| CoT/ToT | 推理任务 | 推理透明 | token 消耗大 |
| Tool Use | 工具依赖任务 | 能力扩展强 | 依赖工具质量 |
| ReWOO | 高效执行场景 | 降低 token 消耗 | 适用场景有限 |
| RAFT | 预判决策任务 | 前瞻性强 | 实现复杂 |

---

> LCEL 是 LangChain 推出的链式表达式语言，支持用 `|` 操作符将各类单元组合。

优势：

- 代码简洁，逻辑清晰，易于多步任务编排。
- 支持多分支、条件、并行等复杂链路。
- 易于插拔、复用和调试每个字任务。

---



