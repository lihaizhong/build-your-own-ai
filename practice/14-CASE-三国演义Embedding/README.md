# CASE-三国演义Embedding

## 项目概述

本项目专注于三国演义文本的 Embedding（词嵌入）和向量检索技术研究。通过将三国演义的文本内容转换为向量表示，实现高效的语义搜索和文本相似度分析。

## 技术栈

- **Python**: 3.11.13
- **包管理器**: UV
- **核心库**:
  - sentence-transformers: 文本 Embedding 生成
  - faiss-cpu: 高效向量检索
  - jieba: 中文分词
  - pandas: 数据处理
  - numpy: 数值计算
  - scikit-learn: 机器学习工具
  - matplotlib/seaborn: 数据可视化

## 项目结构

```
CASE-三国演义Embedding/
├── code/                    # 核心代码文件
├── data/                    # 原始数据文件
├── model/                   # 训练好的模型文件
├── prediction_result/       # 预测结果
├── user_data/               # 用户生成内容
├── docs/                    # 项目文档
├── feature/                 # 分析工具和特征工程
├── .venv/                   # Python 虚拟环境 (Python 3.11.13)
├── pyproject.toml           # 项目依赖配置
└── README.md                # 项目说明
```

## 环境设置

### 激活虚拟环境

```bash
cd /Users/lihaizhong/Documents/Project/build-your-own-x/build-your-own-ai/practice/CASE-三国演义Embedding
source .venv/bin/activate
```

### 安装依赖

```bash
uv sync
```

### 验证安装

```bash
python --version  # 应该显示 Python 3.11.13
python -c "import faiss; print(f'FAISS 版本: {faiss.__version__}')"
python -c "import sentence_transformers; print(f'Sentence-Transformers 版本: {sentence_transformers.__version__}')"
```

## 使用方法

### 基础使用

```python
# 导入必要的库
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 加载预训练模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 生成文本 Embedding
text = "三国演义是中国古典四大名著之一"
embedding = model.encode(text)

# 创建 FAISS 索引
index = faiss.IndexFlatL2(embedding.shape[0])
index.add(embedding.reshape(1, -1))

# 搜索相似文本
k = 5
distances, indices = index.search(embedding.reshape(1, -1), k)
```

## 应用场景

1. **语义搜索**: 根据含义而非关键词搜索文本
2. **文本相似度**: 计算文本之间的语义相似度
3. **问答系统**: 基于向量检索的问答系统
4. **推荐系统**: 基于内容相似度的推荐
5. **文本聚类**: 将相似文本分组

## 注意事项

- 虚拟环境使用 Python 3.11.13
- 使用 `faiss-cpu` 而非 `faiss`（支持 Python 3.11+）
- 中文文本建议使用支持中文的 Embedding 模型
- 大规模文本处理时注意内存使用

## 扩展计划

- [ ] 添加三国演义文本数据集
- [ ] 实现完整的文本 Embedding 流程
- [ ] 构建向量检索系统
- [ ] 添加可视化分析
- [ ] 实现问答系统演示

## 联系方式

- 项目仓库: https://github.com/lihaizhong/build-your-own-ai
- 问题反馈: 通过 GitHub Issues 提交

---

*最后更新: 2026年1月19日*
*版本: v0.1.0*