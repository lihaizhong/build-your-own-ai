# iFlow 模型特性与使用建议

本文档整理了 iFlow CLI 支持的各类 AI 模型的特性，用于指导 `.iflow/agents` 配置中的模型选择。

---

## 模型概览

| 模型名称 | 类型 | 状态 | 适用场景 |
|---------|------|------|---------|
| GLM-4.7 | 通用大模型 | 推荐 | 综合任务、代码编写、文档处理 |
| iFlow-ROME-30BA3B | 专用模型 | 预览版 | Agent 专用、复杂推理 |
| DeepSeek-V3.2 | 代码与推理 | 稳定版 | 深度编码、算法实现 |
| Qwen3-Coder-Plus | 代码专用 | 稳定版 | 代码生成、代码审查 |
| Kimi-K2-Thinking | 思维链模型 | 稳定版 | 复杂推理、问题拆解 |
| MiniMax-M2.1 | 轻量通用 | 稳定版 | 快速响应、简单任务 |
| Kimi-K2.5 | 通用大模型 | 稳定版 | 多模态、长文本处理 |
| Kimi-K2-0905 | 通用大模型 | 稳定版 | 通用对话、文档问答 |

---

## 详细特性分析

### 1. GLM-4.7 (推荐)

**核心优势：**
- **综合能力强**：在代码生成、文档处理、数据分析等多个领域表现均衡
- **中文优化**：对中文语境理解深入，适合中文项目
- **长文本支持**：支持长上下文窗口，适合处理大文件和复杂项目
- **推理能力**：具备较强的逻辑推理和任务拆解能力

**适用场景：**
- 软件工程任务的通用模型
- 代码编写、重构和优化
- 文档编写和维护
- 数据分析和可视化
- Agent 系统的核心推理引擎

**推荐配置的 Agent：**
- `general-purpose`：通用任务处理
- `code-reviewer`：代码审查
- `docs-architect`：技术文档编写
- `python-pro`：Python 代码编写
- `tutorial-engineer`：教程和文档创建

---

### 2. iFlow-ROME-30BA3B (预览版)

**核心优势：**
- **Agent 专用设计**：专为多 Agent 协作场景优化
- **30B 参数规模**：在复杂任务处理上表现出色
- **上下文理解**：对长期对话和任务状态理解能力强
- **协作能力**：支持 Agent 间的高效协作和任务分配

**适用场景：**
- 复杂多步骤任务的规划
- 多 Agent 系统的协调
- 项目级架构设计
- 长期项目的上下文管理

**推荐配置的 Agent：**
- `context-manager`：跨 Agent 上下文管理
- `architect-reviewer`：架构审查

**注意：** 预览版，建议在非生产环境中使用，或作为备用模型。

---

### 3. DeepSeek-V3.2

**核心优势：**
- **代码专精**：在代码生成、算法实现方面表现优异
- **深度推理**：适合处理复杂的算法和逻辑问题
- **多语言支持**：支持 Python、Java、C++、JavaScript 等多种编程语言
- **问题解决**：在调试、Bug 修复方面能力强

**适用场景：**
- 代码实现和算法开发
- 复杂业务逻辑编写
- 性能优化和代码重构
- 机器学习和深度学习项目

**推荐配置的 Agent：**
- `ml-engineer`：机器学习工程
- `mlops-engineer`：MLOps 流程
- `python-pro`：Python 高级特性
- `code-reviewer`：深度代码审查

---

### 4. Qwen3-Coder-Plus

**核心优势：**
- **代码专注**：专为代码生成和优化设计
- **代码审查**：在代码质量检测、安全扫描方面表现出色
- **最佳实践**：熟悉各类编程规范和设计模式
- **快速响应**：响应速度快，适合实时代码建议

**适用场景：**
- 代码质量检查
- 代码风格规范化
- 安全漏洞检测
- 代码补全和建议

**推荐配置的 Agent：**
- `code-reviewer`：代码审查（主要推荐）
- `python-pro`：Python 代码规范检查

---

### 5. Kimi-K2-Thinking

**核心优势：**
- **思维链能力**：擅长将复杂问题拆解为可执行的步骤
- **逻辑推理**：在需要深度思考的任务上表现优异
- **问题分析**：能够深入分析问题本质
- **规划能力**：适合任务规划和方案设计

**适用场景：**
- 复杂问题的分析和拆解
- 技术方案设计
- 需求分析和转化
- Bug 根因分析

**推荐配置的 Agent：**
- `architect-reviewer`：架构设计审查

---

### 6. MiniMax-M2.1

**核心优势：**
- **轻量快速**：响应速度快，资源占用低
- **简单任务**：在简单、明确的任务上效率高
- **成本效益**：适合高频次的简单查询
- **稳定性**：成熟稳定，适合生产环境

**适用场景：**
- 快速代码片段生成
- 简单的文档编辑
- 常见问题解答
- 小型任务的快速处理

**推荐配置的 Agent：**
- 不建议用于复杂 Agent，可作为辅助工具或快速查询使用

---

### 7. Kimi-K2.5

**核心优势：**
- **多模态能力**：支持文本、代码、图像等多种输入
- **长文本处理**：支持超长上下文，适合文档处理
- **文档理解**：在文档解析和内容提取方面能力强
- **通用性强**：在多个领域都有不错表现

**适用场景：**
- 文档分析和处理
- 代码与文档结合的任务
- 图像内容理解
- 长文档问答

**推荐配置的 Agent：**
- `intelli-doc-writer`：智能文档编写
- `docs-architect`：技术文档生成
- `data-analysis-agent`：数据分析（如果涉及文档）

---

### 8. Kimi-K2-0905

**核心优势：**
- **通用对话**：适合日常对话和问答
- **中文友好**：对中文语言特性理解好
- **稳定性好**：成熟版本，适合稳定使用
- **成本较低**：适合大量简单对话场景

**适用场景：**
- 通用问答
- 日常对话
- 简单任务处理
- 知识查询

**推荐配置的 Agent：**
- 不建议用于复杂 Agent 任务，适合作为基础问答模型

---

## Agent 模型配置建议

### 通用原则

1. **核心 Agent 使用 GLM-4.7**：作为默认推荐模型，覆盖大部分场景
2. **代码相关使用 DeepSeek-V3.2 或 Qwen3-Coder-Plus**：代码任务优先选择
3. **复杂规划使用 Kimi-K2-Thinking 或 iFlow-ROME-30BA3B**：需要深度思考的任务
4. **文档处理使用 Kimi-K2.5**：涉及文档和多模态的任务
5. **预览版谨慎使用**：iFlow-ROME-30BA3B 仅在特定场景下使用

### 具体配置建议

| Agent | 推荐模型 | 备选模型 | 说明 |
|-------|---------|---------|------|
| `ai-engineer` | DeepSeek-V3.2 | GLM-4.7 | LLM 应用和 RAG 系统开发 |
| `architect-reviewer` | Kimi-K2-Thinking | GLM-4.7 | 架构审查需要深度思考 |
| `code-reviewer` | Qwen3-Coder-Plus | DeepSeek-V3.2 | 代码审查是核心能力 |
| `context-manager` | iFlow-ROME-30BA3B | GLM-4.7 | 长期上下文管理 |
| `data-analysis-agent` | GLM-4.7 | Kimi-K2.5 | 数据分析需要综合能力 |
| `data-collection-agent` | GLM-4.7 | MiniMax-M2.1 | 数据采集和信息搜索 |
| `docs-architect` | GLM-4.7 | Kimi-K2.5 | 文档生成和结构化 |
| `frond-master` | GLM-4.7 | Kimi-K2.5 | 前端设计与工程实现 |
| `intelli-doc-writer` | Kimi-K2.5 | GLM-4.7 | 文档编写和多模态 |
| `ml-engineer` | DeepSeek-V3.2 | GLM-4.7 | 机器学习需要代码能力 |
| `mlops-engineer` | DeepSeek-V3.2 | GLM-4.7 | MLOps 流程和部署 |
| `perception-agent` | GLM-4.7 | Kimi-K2.5 | 内容感知和分析 |
| `prompt-engineer` | GLM-4.7 | Kimi-K2-Thinking | Prompt 优化和设计 |
| `python-pro` | Qwen3-Coder-Plus | DeepSeek-V3.2 | Python 专家级任务 |
| `tutorial-engineer` | GLM-4.7 | - | 教程编写需要表达能力 |
| `ui-ux-designer` | GLM-4.7 | Kimi-K2.5 | 界面设计和用户体验 |

---

## 性能与成本对比

### 响应速度排序（从快到慢）
1. MiniMax-M2.1
2. Qwen3-Coder-Plus
3. GLM-4.7
4. Kimi-K2-0905
5. Kimi-K2.5
6. DeepSeek-V3.2
7. Kimi-K2-Thinking
8. iFlow-ROME-30BA3B

### 推理能力排序（从强到弱）
1. iFlow-ROME-30BA3B
2. Kimi-K2-Thinking
3. DeepSeek-V3.2
4. GLM-4.7
5. Qwen3-Coder-Plus
6. Kimi-K2.5
7. Kimi-K2-0905
8. MiniMax-M2.1

### 代码能力排序（从强到弱）
1. DeepSeek-V3.2
2. Qwen3-Coder-Plus
3. GLM-4.7
4. Kimi-K2-Thinking
5. iFlow-ROME-30BA3B
6. Kimi-K2.5
7. Kimi-K2-0905
8. MiniMax-M2.1

---

## 使用建议

### 开发环境
- 优先使用 **GLM-4.7** 作为默认模型
- 代码相关任务使用 **DeepSeek-V3.2** 或 **Qwen3-Coder-Plus**
- 复杂规划和探索使用 **Kimi-K2-Thinking**

### 生产环境
- 使用稳定版模型（GLM-4.7、DeepSeek-V3.2、Qwen3-Coder-Plus）
- 避免使用预览版模型（iFlow-ROME-30BA3B）
- 考虑成本和性能的平衡

### 特殊场景
- **多模态任务**：使用 Kimi-K2.5
- **Agent 协作**：考虑 iFlow-ROME-30BA3B（预览版）
- **快速原型**：使用 MiniMax-M2.1
- **深度学习**：使用 DeepSeek-V3.2

---

## 配置示例

### `.iflow/settings.json` 基础配置
`settings.json` 主要用于配置认证、主题、工具等 CLI 行为，不包含 Agent 模型配置。Agent 模型需要在各自的 `.md` 配置文件中设置。

```json
{
  "selectedAuthType": "iflow",
  "apiKey": "sk-xxx",
  "baseUrl": "https://apis.iflow.cn/v1",
  "modelName": "GLM-4.7",
  "theme": "Default",
  "vimMode": false,
  "maxSessionTurns": -1,
  "telemetry": {
    "enabled": false,
    "target": "local"
  }
}
```

### Agent 模型配置

每个 Agent 的模型配置位于 `.iflow/agents/<agent-name>.md` 文件的 YAML frontmatter 中。例如：

**`.iflow/agents/ai-engineer.md`**（示例配置）：
```markdown
---
agent-type: ai-engineer
name: ai-engineer
description: Build LLM applications, RAG systems, and prompt pipelines.
when-to-use: Build LLM applications, RAG systems, and prompt pipelines.
allowed-tools: 
model: DeepSeek-V3.2
inherit-tools: true
inherit-mcps: true
color: red
---

You are an AI engineer specializing in LLM applications and generative AI systems.
...
```

**`.iflow/agents/code-reviewer.md`**（示例配置）：
```markdown
---
agent-type: code-reviewer
name: code-reviewer
description: Expert code review specialist.
when-to-use: Expert code review specialist.
allowed-tools: 
model: Qwen3-Coder-Plus
inherit-tools: true
inherit-mcps: true
color: red
---

You are a senior code reviewer with deep expertise in configuration security.
...
```

### 环境变量配置

所有 `settings.json` 中的配置项都支持通过环境变量设置，使用 `IFLOW_` 前缀：

```bash
export IFLOW_apiKey="your_api_key"
export IFLOW_baseUrl="https://apis.iflow.cn/v1"
export IFLOW_modelName="GLM-4.7"
export IFLOW_theme="Default"
```

---

## 更新日志

- **2026-02-09**：初始化文档，整理 8 个模型的特性
- **2026-02-09**：添加新增 Agent 配置建议（ai-engineer、data-collection-agent、frond-master、perception-agent、prompt-engineer、ui-ux-designer）
- **2026-02-09**：移除未安装的 Agent（plan-agent、explore-agent），仅保留已安装的 16 个 Agent
- **2026-02-09**：修正配置示例，移除不存在的 agentModels 字段，更新为正确的 Agent 配置方式
- **2026-02-09**：修正 Agent 模型配置示例，使用 YAML frontmatter 格式的 `model` 字段
- 基于当前支持的模型版本

---

## 参考资料

- iFlow CLI 官方文档
- 各模型官方文档和特性说明
- AGENTS.md - 代理开发指南

---

*文档维护：建议定期更新模型特性和配置建议*