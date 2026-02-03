# RAG问答系统

基于LangChain的检索增强生成(RAG)问答系统，用于从文档中智能回答问题。

## 功能特性

- 文档向量化处理
- 向量数据库检索
- 智能问答生成
- 支持多种文档格式
- 中英文双语支持

## 安装

```bash
# 创建虚拟环境
uv venv --python 3.11

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv sync
```

## 配置

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的API密钥：
```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# DashScope API Key (可选，用于中文embedding)
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

## 使用方法

```bash
# 运行示例
uv run python code/rag_example.py
```

## 项目结构

```
RAG-问答系统/
├── code/              # 核心代码
│   ├── rag_example.py    # RAG示例代码
│   └── rag_system.py     # RAG系统核心类
├── data/              # 数据目录
├── docs/              # 文档
├── output/            # 输出结果
├── pyproject.toml     # 项目配置
└── README.md          # 项目说明
```

## 技术栈

- **LangChain**: RAG框架
- **OpenAI**: 大语言模型
- **FAISS**: 向量数据库
- **Sentence Transformers**: 文本向量化
- **PyPDF2**: PDF处理

## 许可证

MIT License