# iFlow Agents & Skills 汇总

> 本文档整理了 `.iflow/` 目录下所有 agents 和 skills 的作用、描述和触发关键字。

---

# 📋 Agents 汇总

## 1. ai-engineer

**作用**：构建 LLM 应用、RAG 系统和提示词管道

**描述**：实现向量搜索、智能体编排和 AI API 集成。主动用于 LLM 功能、聊天机器人或 AI 驱动的应用。

**触发关键字**：
- LLM applications / 大模型应用
- RAG systems / RAG系统
- Chatbots / 聊天机器人
- AI-powered applications / AI应用
- Prompt pipelines / 提示词管道
- Vector search / 向量搜索
- Agent orchestration / 智能体编排

**模型**：DeepSeek-V3.2

**颜色标识**：[红色](#color-identifier)

---

## 2. architect-reviewer

**作用**：架构一致性审查

**描述**：审查代码更改的架构一致性和模式。在任何结构更改、新服务或 API 修改后主动使用。确保 SOLID 原则、适当的分层和可维护性。

**触发关键字**：
- Architectural review / 架构审查
- Structural changes / 结构更改
- API modifications / API修改
- New services / 新服务
- SOLID principles
- Pattern compliance / 模式符合性
- System design / 系统设计

**模型**：Kimi-K2-Thinking

**颜色标识**：[蓝色](#color-identifier)

---

## 3. code-reviewer

**作用**：代码质量审查

**描述**：代码审查专家。主动审查代码的质量、安全性和可维护性。在编写或修改代码后立即使用。

**触发关键字**：
- Code review / 代码审查
- Quality check / 质量检查
- Security review / 安全审查
- Refactoring / 重构
- Best practices / 最佳实践
- Configuration changes / 配置更改

**模型**：Qwen3-Coder-Plus

**颜色标识**：[红色](#color-identifier)

---

## 4. context-manager

**作用**：上下文管理

**描述**：管理多个智能体和长期任务的上下文。用于协调复杂的多智能体工作流或需要在多个会话间保留上下文时。对于超过 10k tokens 的项目必须使用。

**触发关键字**：
- Multi-agent workflows / 多智能体工作流
- Long-running tasks / 长期任务
- Context preservation / 上下文保留
- Large projects / 大型项目 (>10k tokens)
- Session management / 会话管理

**模型**：iFlow-ROME-30A3B

**颜色标识**：[棕色](#color-identifier)

---

## 5. data-analysis-agent

**作用**：数据分析与可视化

**描述**：专业数据分析师，精通数据标注与可视化。支持聚类、指标计算与统计分析，并按数据类型生成专业图表。

**触发关键字**：
- Data analysis / 数据分析
- Data visualization / 数据可视化
- Chart generation / 图表生成
- Clustering / 聚类
- Data labeling / 数据标注
- Statistical analysis / 统计分析
- CSV/Excel/JSON analysis

**模型**：GLM-4.7

**颜色标识**：[绿色](#color-identifier)

**可用工具**：memory-system, python-execute-server, sequential-thinking

---

## 6. data-collection-agent

**作用**：数据收集与信息采集

**描述**：专注于从全网搜集、甄别与整合信息的采集代理，支持多源检索、去重汇总与来源标注。

**触发关键字**：
- Web search / 网络搜索
- Data collection / 数据收集
- Information gathering / 信息收集
- Cross-source comparison / 跨来源比对
- Deep search / 深度搜索
- Material compilation / 资料汇编

**模型**：GLM-4.7

**颜色标识**：[绿色](#color-identifier)

**可用工具**：web_search, web_fetch, memory-system, sequential-thinking

---

## 7. docs-architect

**作用**：技术文档架构

**描述**：从现有代码库创建全面的技术文档。分析架构、设计模式和实现细节，生成长篇技术手册和电子书。

**触发关键字**：
- Technical documentation / 技术文档
- Architecture guides / 架构指南
- System documentation / 系统文档
- Technical deep-dives / 技术深入
- Codebase analysis / 代码库分析
- Design patterns / 设计模式

**模型**：GLM-4.7

**颜色标识**：[黄色](#color-identifier)

---

## 8. frond-master

**作用**：前端设计与工程化

**描述**：像 Claude Code 的 Agent Skills 一样，专注前端设计 + 审美 + 工程可落地性。

**触发关键字**：
- Frontend design / 前端设计
- UI/UX systems / UI/UX系统
- Visual engineering / 视觉工程
- Interface design / 界面设计
- Component design / 组件设计
- Responsive design / 响应式设计

**模型**：GLM-4.7

**颜色标识**：[蓝色](#color-identifier)

---

## 9. intelli-doc-writer

**作用**：智能文档与代码助手

**描述**：辅助用户创建、生成、编辑、优化文档和代码。记住创建、修改或者删除的文件，在下一次操作同一文件前，使用版本控制或者重新读取来防止后续修改错误的问题。

**触发关键字**：
- Document creation / 文档创建
- Code generation / 代码生成
- File editing / 文件编辑
- Document optimization / 文档优化
- Version control / 版本控制
- File management / 文件管理

**模型**：Kimi-K2.5

**颜色标识**：[橙色](#color-identifier)

---

## 10. ml-engineer

**作用**：机器学习工程

**描述**：实现 ML 管道、模型服务和特征工程。处理 TensorFlow/PyTorch 部署、A/B 测试和监控。

**触发关键字**：
- ML pipelines / ML管道
- Model serving / 模型服务
- Feature engineering / 特征工程
- A/B testing / A/B测试
- Model monitoring / 模型监控
- TensorFlow/PyTorch deployment
- Production deployment / 生产部署

**模型**：DeepSeek-V3.2

**颜色标识**：[紫色](#color-identifier)

---

## 11. mlops-engineer

**作用**：MLOps 工程与基础设施

**描述**：构建 ML 管道、实验跟踪和模型注册。实现 MLflow、Kubeflow 和自动重训练。处理数据版本控制和可重现性。

**触发关键字**：
- MLOps / MLOps
- ML infrastructure / ML基础设施
- Experiment tracking / 实验跟踪
- Model registry / 模型注册
- Automated retraining / 自动重训练
- Data versioning / 数据版本控制
- Pipeline automation / 管道自动化

**模型**：DeepSeek-V3.2

**颜色标识**：[橙色](#color-identifier)

---

## 12. perception-agent

**作用**：内容感知和分析

**描述**：专业的内容感知和分析专家，擅长读取和理解用户输入的查询和文件内容，提取关键信息并结合用户需求进行综合分析。

**触发关键字**：
- Content analysis / 内容分析
- File analysis / 文件分析
- Query understanding / 查询理解
- Information extraction / 信息提取
- Constraint extraction / 约束提取
- Directory analysis / 目录分析

**模型**：GLM-4.7

**颜色标识**：[橙色](#color-identifier)

**可用工具**：*, memory-system, excel-edit-server

---

## 13. prompt-engineer

**作用**：提示词工程优化

**描述**：优化 LLM 和 AI 系统的提示词。用于构建 AI 功能、改进智能体性能或制作系统提示词。

**触发关键字**：
- Prompt engineering / 提示词工程
- Prompt optimization / 提示词优化
- System prompts / 系统提示词
- Agent performance / 智能体性能
- AI features / AI功能
- Prompt patterns / 提示词模式

**模型**：GLM-4.7

**颜色标识**：[棕色](#color-identifier)

---

## 14. python-pro

**作用**：Python 专家编程

**描述**：编写符合 Python 习惯的代码，使用高级功能如装饰器、生成器和 async/await。优化性能、实现设计模式并确保全面测试。

**触发关键字**：
- Python refactoring / Python重构
- Python optimization / Python优化
- Advanced Python features / 高级Python功能
- Decorators / 装饰器
- Async/await
- Design patterns / 设计模式
- Python testing / Python测试

**模型**：Qwen3-Coder-Plus

**颜色标识**：[红色](#color-identifier)

---

## 15. tutorial-engineer

**作用**：教程工程与教育内容

**描述**：从代码创建分步教程和教育内容。将复杂概念转化为渐进式学习体验，包含动手实践示例。

**触发关键字**：
- Tutorials / 教程
- Educational content / 教育内容
- Onboarding guides / 入门指南
- Feature tutorials / 功能教程
- Concept explanations / 概念解释
- Step-by-step learning / 分步学习

**模型**：GLM-4.7

**颜色标识**：[蓝色](#color-identifier)

---

## 16. ui-ux-designer

**作用**：UI/UX 设计

**描述**：创建界面设计、线框图和设计系统。精通用户研究、原型设计和可访问性标准。

**触发关键字**：
- Interface design / 界面设计
- Wireframes / 线框图
- Design systems / 设计系统
- User flows / 用户流程
- User research / 用户研究
- Accessibility / 可访问性
- Interface optimization / 界面优化

**模型**：GLM-4.7

**颜色标识**：[棕色](#color-identifier)

---

# 🛠️ Skills 汇总

## 1. doc-coauthoring

**作用**：文档协作工作流

**描述**：引导用户通过结构化工作流程进行文档协作。当用户想要编写文档、提案、技术规范、决策文档或类似结构化内容时使用。此工作流程帮助用户高效传递上下文、通过迭代优化内容，并验证文档对读者有效。

**触发关键字**：
- Write documentation / 编写文档
- Create proposals / 创建提案
- Draft specs / 起草规范
- Decision docs / 决策文档
- Technical writing / 技术写作
- Collaborative writing / 协作写作
- RFC (Request for Comments)

**工作流程**：
1. 上下文收集：用户提供所有相关上下文，Claude 提出澄清问题
2. 优化与结构：通过头脑风暴和编辑迭代构建每个部分
3. 读者测试：用全新的 Claude（无上下文）测试文档，发现盲点

---

## 2. docx

**作用**：Word 文档处理

**描述**：全面的文档创建、编辑和分析，支持修订、注释、格式保留和文本提取。当 Claude 需要处理专业文档（.docx 文件）时使用。

**触发关键字**：
- .docx files / Word文档
- Document creation / 文档创建
- Document editing / 文档编辑
- Tracked changes / 修订
- Comments / 注释
- Format preservation / 格式保留
- Text extraction / 文本提取

**主要功能**：
- 读取/分析内容：文本提取、XML 访问
- 创建新文档：使用 docx-js
- 编辑现有文档：使用 Document 库（Python）
- 修订工作流：支持全面修订跟踪
- 文档转图片：用于视觉分析

**依赖**：
- pandoc（文本提取）
- docx（npm，创建文档）
- LibreOffice（PDF 转换）
- Poppler（PDF 转图片）
- defusedxml（XML 解析）

---

## 3. pdf

**作用**：PDF 处理工具包

**描述**：全面的 PDF 处理工具包，用于提取文本和表格、创建新 PDF、合并/拆分文档以及处理表单。当 Claude 需要填充 PDF 表单或以编程方式处理、生成或分析 PDF 文档时使用。

**触发关键字**：
- PDF processing / PDF处理
- Extract text from PDF / PDF文本提取
- Extract tables from PDF / PDF表格提取
- Create PDFs / 创建PDF
- Merge PDFs / 合并PDF
- Split PDFs / 拆分PDF
- Fill PDF forms / 填充PDF表单

**主要工具**：
- **pypdf**：基本操作（合并、拆分、旋转、加密）
- **pdfplumber**：文本和表格提取
- **reportlab**：创建 PDF
- **pdftotext**：命令行文本提取
- **qpdf**：命令行 PDF 操作
- **pytesseract**：OCR 扫描 PDF

---

## 4. pptx

**作用**：PowerPoint 演示文稿处理

**描述**：演示文稿创建、编辑和分析。当 Claude 需要处理演示文稿（.pptx 文件）时使用。

**触发关键字**：
- .pptx files / PowerPoint文档
- Presentation creation / 演示文稿创建
- Slide editing / 幻灯片编辑
- Layouts / 布局
- Speaker notes / 演讲者备注
- Comments / 注释
- Template-based creation / 基于模板创建

**主要工作流程**：
- **创建新演示文稿（无模板）**：使用 html2pptx 工作流
- **编辑现有演示文稿**：使用 OOXML 直接编辑
- **使用模板创建**：提取模板、分析布局、重新排列、替换文本

**设计原则**：
- 根据内容选择合适的调色板
- 使用 Web 安全字体
- 创建清晰的视觉层次
- 确保可读性和对比度
- 保持一致性

**依赖**：
- markitdown（文本提取）
- pptxgenjs（创建演示文稿）
- playwright（HTML 渲染）
- sharp（图像处理）
- LibreOffice（PDF 转换）
- Poppler（PDF 转图片）

---

## 5. xlsx

**作用**：Excel 电子表格处理

**描述**：全面的电子表格创建、编辑和分析，支持公式、格式、数据分析和可视化。当 Claude 需要处理电子表格时使用。

**触发关键字**：
- .xlsx/.xlsm/.csv files / Excel文件
- Spreadsheet creation / 电子表格创建
- Data analysis / 数据分析
- Formulas / 公式
- Formatting / 格式
- Financial models / 财务模型
- Recalculate formulas / 重新计算公式

**关键要求**：
- **零公式错误**：交付的 Excel 模型必须无任何公式错误
- **使用公式而非硬编码**：始终使用 Excel 公式，不要在 Python 中计算并硬编码值
- **保留现有模板**：修改文件时精确匹配现有格式、样式和约定

**财务模型标准**：
- **颜色编码**：蓝色（输入）、黑色（公式）、绿色（内部链接）、红色（外部链接）、黄色（关键假设）
- **数字格式**：年份为文本、货币带单位、零显示为"-"、百分比默认一位小数、负数使用括号

**主要工具**：
- **pandas**：数据分析和批量操作
- **openpyxl**：复杂格式、公式和 Excel 特定功能
- **recalc.py**：重新计算公式并检测错误

---

# 📊 统计总结

| 类型 | 数量 | 说明 |
|------|------|------|
| Agents | 16 | 涵盖 AI、ML、代码、文档、设计等领域的专业智能体 |
| Skills | 5 | 处理文档、PDF、演示文稿和电子表格的专业技能 |

**模型分布**：
- GLM-4.7：5 个
- DeepSeek-V3.2：3 个
- Qwen3-Coder-Plus：2 个
- Kimi-K2.5：1 个
- Kimi-K2-Thinking：1 个
- iFlow-ROME-30A3B：1 个

**颜色标识**：
- 红色：3 个（ai-engineer, code-reviewer, python-pro）
- 蓝色：3 个（architect-reviewer, frond-master, tutorial-engineer）
- 棕色：3 个（context-manager, prompt-engineer, ui-ux-designer）
- 橙色：2 个（intelli-doc-writer, mlops-engineer, perception-agent）
- 绿色：2 个（data-analysis-agent, data-collection-agent）
- 黄色：1 个（docs-architect）
- 紫色：1 个（ml-engineer）

---

<a id="color-identifier"></a>

## 关于颜色标识

**颜色标识**是 iFlow 系统中为每个 Agent 分配的视觉标识，用于在用户界面中区分不同的 Agent。这些颜色标签帮助用户：

- 快速识别当前正在使用的 Agent
- 在多个 Agent 协作工作时进行视觉区分
- 通过颜色联想记忆不同 Agent 的功能特性
- 在日志输出和界面显示中提供清晰的分类

当前系统中使用的颜色包括：红色、蓝色、棕色、橙色、绿色、黄色、紫色，每种颜色代表一类功能相似的 Agent。

---

# 📖 使用建议

1. **根据任务类型选择合适的 Agent**：参考触发关键字快速定位
2. **组合使用多个 Agent**：对于复杂任务，可以按顺序调用多个 Agent
3. **优先使用主动推荐的 Agent**：文档中标注 "Use PROACTIVELY" 的 Agent 应优先考虑
4. **注意 Agent 的可用工具**：某些 Agent 有特定的工具限制或专有工具
5. **模型选择**：不同 Agent 使用不同的基础模型，根据任务需求选择

---

*文档创建日期：2026年2月10日*
*版本：1.0*