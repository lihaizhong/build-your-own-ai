# 🚀 Build Your Own AI

> AI/ML 全栈学习项目，涵盖大模型、机器学习、深度学习、Agent 开发、RAG、向量数据库等 28 个课程模块

## 📖 项目简介

本项目是一个系统性的 AI/ML 学习资源库，包含：

- **28 个完整课程模块** - 从入门到进阶的完整学习路径
- **26 个实战项目** - 涵盖竞赛级案例和企业级应用
- **统一依赖管理** - 基于 uv 的现代化 Python 开发环境
- **丰富的中文文档** - 详尽的代码注释和学习笔记

## 🏃 快速开始

```bash
# 克隆项目
git clone https://github.com/lihaizhong/build-your-own-ai.git
cd build-your-own-ai

# 安装依赖（需要先安装 uv）
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填写必要的 API 密钥
```

> 💡 **提示**: 首次使用请先安装 [uv](https://docs.astral.sh/uv/)

## 📁 目录结构

```
build-your-own-ai/
├── courseware/        # 📚 课程材料（29个模块）
├── practice/          # 💻 实战项目（26个子项目）
├── practice-py/       # 🐍 Python 基础练习
├── notebook/          # 📝 学习笔记
├── docs/              # 📄 项目文档
├── public/            # 🖼️ 公共资源
└── .iflow/            # ⚙️ iFlow CLI 配置
```

## 🗺️ 课程模块

| 阶段 | 模块 | 内容 |
|------|------|------|
| **基础入门** | 01-06 | 大模型原理、API 使用、Prompt 工程、Cursor 编程、Coze/Dify 平台 |
| **机器学习** | 07-11 | 分析式 AI、算法原理、Scikit-learn、时间序列预测 |
| **深度学习** | 12-14 | 神经网络、TensorFlow、PyTorch、目标检测 |
| **RAG 技术** | 15-17 | 向量数据库、RAG 原理与实战、Text2SQL |
| **Agent 开发** | 18-21 | LangChain、Function Calling、MCP、智能体设计 |
| **高级应用** | 22-28 | 多模态大模型、Fine-tuning、企业级项目实战 |

> 📥 [课程资料下载](https://pan.baidu.com/s/1MfjQwHba-dHav67tYVAWAw?pwd=8888)

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **语言** | Python 3.11+ |
| **包管理** | uv（现代 Python 包管理器） |
| **类型检查** | basedpyright |
| **深度学习** | PyTorch、TensorFlow、Transformers |
| **大模型** | OpenAI、通义千问、DeepSeek、Ollama |
| **向量数据库** | FAISS、ChromaDB |
| **Agent 框架** | LangChain、Qwen-Agent、MCP |
| **Web 框架** | FastAPI、Flask、Gradio |

## 📚 文档资料

### 包管理工具
- [uv 官方文档](https://docs.astral.sh/uv/) | [uv 中文文档](https://hellowac.github.io/uv-zh-cn/)

### 数据科学
- [NumPy 中文文档](https://numpy.com.cn/doc/2.3/index.html) | [Pandas 官方文档](https://pandas.pydata.org/docs/)
- [Matplotlib 教程](https://www.runoob.com/matplotlib/matplotlib-tutorial.html) | [SciPy 教程](https://www.runoob.com/scipy/scipy-tutorial.html)

### 机器学习
- [Scikit-learn 官方文档](https://scikit-learn.org/stable/user_guide.html)
- [PyTorch 官方文档](https://pytorch.org) | [TensorFlow 官方文档](https://www.tensorflow.org/?hl=zh-cn)

### 大模型与 Agent
- [Transformers 教程](https://huggingface.co/docs/transformers/v4.56.0/zh/index)
- [LangChain 官方文档](https://docs.langchain.com)
- [AI Agent 教程](https://www.runoob.com/ai-agent/ai-agent-tutorial.html)

### 学习资源
- [x] [大模型 RAG 基础](https://arthurchiao.art/blog/rag-basis-bge-zh/)
- [ ] [Transformer 工作原理](https://arthurchiao.art/blog/transformers-from-scratch-zh/)
- [ ] [GPT 极简实现](https://arthurchiao.art/blog/gpt-as-a-finite-state-markov-chain-zh/)
- [ ] [如何训练企业级 GPT 助手](https://arthurchiao.art/blog/how-to-train-a-gpt-assistant-zh/)

## ⚙️ 环境配置

### uv 查找 Python 环境的顺序

1. 当前目录下的 `.python-version` 文件设定的版本
2. 当前启用的虚拟环境
3. 当前目录下的 `.venv` 目录
4. uv 自己安装的 Python 环境
5. 系统环境设定的 Python 环境

### 常用命令

```bash
# 添加新依赖
uv add package_name

# 添加开发依赖
uv add --group dev package_name

# 运行 Python 脚本
uv run python script.py

# 运行 Jupyter
uv run jupyter notebook

# 类型检查
basedpyright
```

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

<p align="center">
  如果这个项目对你有帮助，欢迎 ⭐ Star 支持一下！
</p>