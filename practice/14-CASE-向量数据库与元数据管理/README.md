# 向量数据库与元数据管理

向量数据库与元数据管理项目，用于演示如何使用向量数据库进行相似度搜索和元数据管理。

## 功能特性

- 向量嵌入生成
- 向量数据库索引构建
- 元数据管理
- 相似度搜索

## 安装

```bash
# 创建虚拟环境
uv venv --python 3.11

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv sync
```

## 使用方法

```bash
# 运行示例代码
uv run python code/example.py
```

## 技术栈

- Python 3.11+
- FAISS
- ChromaDB
- Sentence Transformers
- NumPy
- Pandas