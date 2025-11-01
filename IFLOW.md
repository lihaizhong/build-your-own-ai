# build your own ai - IFLOW 指南

## 项目概述

这是一个全面的AI学习和实践项目，专注于构建从基础到高级的人工智能应用。项目包含完整的机器学习课程材料、实践案例和优化实验，涵盖大模型API使用、传统机器学习、深度学习、Agent开发、RAG技术、Text2SQL、向量数据库等多个AI领域。项目已发展成为一个成熟的AI学习生态系统，包含19个核心课程模块和丰富的实战案例，形成了从基础到专家的完整学习路径。项目以实战为导向，通过大量真实项目和竞赛案例，帮助学习者掌握AI技术的实际应用能力。

## 核心技术栈

- **Python环境管理**: uv (现代Python包管理器)
- **机器学习**: scikit-learn, pandas, numpy, scipy, statsmodels, mlxtend
- **深度学习**: PyTorch, Transformers, accelerate, peft
- **大模型集成**: OpenAI, Dashscope, ModelScope, cozepy
- **可视化**: matplotlib, seaborn, plotly
- **Web服务**: FastAPI, Flask, uvicorn
- **数据分析**: ydata-profiling, missingno
- **优化算法**: XGBoost, LightGBM, CatBoost, bayesian-optimization
- **时间序列**: Prophet
- **RAG技术**: LangChain, FAISS向量数据库
- **Text2SQL**: vanna, SQL Copilot
- **开发工具**: loguru, tqdm, ipywidgets, basedpyright, joblib

## 环境配置

### Python版本要求
- Python >= 3.11
- 推荐使用 uv 管理虚拟环境和依赖

### 快速设置

```bash
# 克隆项目后，设置环境
cd build-your-own-ai

# 使用 uv 创建虚拟环境并安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

## 项目结构

```
build-your-own-ai/
├── courseware/                    # 课程材料和讲义
│   ├── 0-开营直播/               # 开营直播内容
│   ├── 1-AI大模型原理与API使用/    # 大模型基础
│   ├── 2-DeepSeek使用与Prompt工程/ # Prompt工程
│   ├── 3-Cursor编程-从入门到精通/   # AI编程工具
│   ├── 4-Coze工作原理与应用实例/   # Agent平台
│   ├── 5-Agent进阶实战与插件开发/   # Agent开发
│   ├── 6-Dify本地化部署和应用/     # 工作流平台
│   ├── 7-分析式AI基础/           # 传统机器学习
│   ├── 8-不同领域的AI算法/        # 算法应用
│   ├── 9-机器学习神器/           # 工具集
│   ├── 10-时间序列模型/          # 时间序列
│   ├── 11-时间序列AI大赛/         # 竞赛项目
│   ├── 12-神经网络基础与Tensorflow实战/ # TensorFlow
│   ├── 13-Pytorch与视觉检测/      # PyTorch视觉
│   ├── 14-Embeddings和向量数据库/   # 向量数据库
│   ├── 15-RAG技术与应用/          # RAG基础与应用
│   ├── 16-RAG高级技术与最佳实践/    # RAG高级技术
│   ├── 17-Text2SQL：自助式数据报表开发/ # Text2SQL开发
│   ├── 18-LangChain：多任务应用开发/   # LangChain框架
│   └── 19-Function Calling与协作/    # 函数调用与协作
├── practice/                       # 实践练习代码
│   ├── 1-AI大模型原理与API使用/    # API调用实践
│   ├── 2-DeepSeek使用与Prompt工程/  # 本地模型部署
│   ├── 3-Cursor编程-从入门到精通/   # 编程工具实战
│   ├── 4-Coze工作原理与应用实例/   # 平台应用案例
│   ├── 5-Agent进阶实战与插件开发/   # Agent开发实践
│   ├── 6-Dify本地化部署和应用/     # 工作流平台实践
│   ├── 7-分析式AI基础/           # 机器学习基础
│   ├── 8-不同领域的AI算法/        # 算法应用案例
│   ├── 9-机器学习神器/           # 工具集实践
│   ├── 10-时间序列模型/          # 时间序列实践
│   └── 11-时间序列AI大赛/         # 竞赛项目实践
├── projects/                       # 核心项目案例
│   ├── Case-二手车价格预测-DataWhale/ # 原始版本
│   ├── Case-二手车价格预测-P1/    # 优化版本
│   └── CASE-资金流入流出预测-P1/   # 时间序列项目
└── public/                         # 公共资源
```

## 核心项目案例

### 🚗 二手车价格预测 (projects/Case-二手车价格预测-P1)

这是一个完整的机器学习项目，展示了从数据预处理到模型优化的全流程：

**项目特点**:
- 完整的数据预处理和特征工程
- 29个版本的模型迭代优化 (modeling_v1-v29)
- 多种算法集成 (RandomForest, LightGBM, XGBoost, CatBoost, ExtraTrees)
- 高级技术：分层建模、Stacking集成、智能校准、贝叶斯优化
- 全面的数据分析和可视化
- 特征分析和模型验证模块
- 竞赛级性能优化和验证

**运行方式**:
```bash
cd projects/Case-二手车价格预测-P1
python model/modeling_v28.py  # 最佳性能版本
```

**性能成果**:
- **最佳成绩**: MAE = 487.7112 (modeling_v28.py)
- **目标达成**: 成功突破MAE < 500的目标
- **竞赛排名**: 多个版本进入竞赛前列
- **版本迭代**: 从v1的1000+分优化到v28的487分

### 📈 资金流入流出预测 (projects/CASE-资金流入流出预测-P1)

时间序列预测项目，应用Prophet等先进模型：

**项目特点**:
- 时间序列特征工程
- 多种预测模型对比
- 季节性和趋势分析
- 商业场景应用

### 🤖 Dify平台集成 (courseware/6-Dify本地化部署和应用)

Dify工作流平台本地化部署和应用案例：

**项目特点**:
- Agent API客户端实现
- 工作流自动化配置
- 多种应用类型支持
- 实际业务场景集成

**运行方式**:
```bash
cd courseware/6-Dify本地化部署和应用
python dify_agent_client.py
```

### 📚 RAG技术与应用 (courseware/15-RAG技术与应用)

RAG(Retrieval-Augmented Generation)技术实战应用：

**项目特点**:
- ChatPDF应用实现
- 迪士尼RAG助手案例
- Embedding技术深入应用
- 向量检索与生成结合

**运行方式**:
```bash
cd courseware/15-RAG技术与应用/CASE-ChatPDF-Faiss
python chat_pdf.py
```

### 🔍 Text2SQL开发 (courseware/17-Text2SQL：自助式数据报表开发)

Text2SQL技术实现自助式数据报表开发：

**项目特点**:
- Vanna框架集成应用
- LangChain SQL工具链
- SQL Copilot智能助手
- 自然语言转SQL实战

**运行方式**:
```bash
cd courseware/17-Text2SQL：自助式数据报表开发/CASE-SQL-vanna
python sql_assistant.py
```

### ⚙️ LangChain多任务开发 (courseware/18-LangChain：多任务应用开发)

基于LangChain框架的多任务应用开发：

**项目特点**:
- 故障诊断Agent实现
- 工具链组合应用
- LangChain最佳实践
- 复杂任务自动化

**运行方式**:
```bash
cd courseware/18-LangChain：多任务应用开发/CASE-搭建故障诊断Agent
python diagnostic_agent.py
```

## 学习路径

### 初学者路径
1. **0-开营直播**: 了解课程规划和学习目标
2. **1-AI大模型原理与API使用**: 掌握大模型基础API调用
3. **7-分析式AI基础**: 学习传统机器学习方法
4. **9-机器学习神器**: 熟悉常用工具和库

### 进阶路径
1. **2-DeepSeek使用与Prompt工程**: 深入Prompt优化
2. **5-Agent进阶实战与插件开发**: 构建智能Agent
3. **6-Dify本地化部署和应用**: 工作流自动化
4. **10-时间序列模型**: 专业领域应用

### 专家路径
1. **11-时间序列AI大赛**: 竞赛级项目实践
2. **12-神经网络基础与Tensorflow实战**: 深度学习框架
3. **13-Pytorch与视觉检测**: 计算机视觉应用
4. **14-Embeddings和向量数据库**: 向量检索技术

### 高级应用路径
1. **15-RAG技术与应用**: 掌握检索增强生成技术
2. **16-RAG高级技术与最佳实践**: RAG系统优化与高级应用
3. **17-Text2SQL：自助式数据报表开发**: 自然语言转SQL实战
4. **18-LangChain：多任务应用开发**: 复杂AI应用构建
5. **19-Function Calling与协作**: 函数调用与多Agent协作

## 开发规范

### 代码规范
- 使用 Python 3.11+ 类型注解
- 遵循 PEP 8 代码风格
- 函数和类添加详细文档字符串
- 使用 uv 管理项目依赖
- 使用 basedpyright 进行类型检查

### 项目结构规范
- 每个课程模块包含完整的材料、案例和实践代码
- 使用模块化设计，功能分离清晰
- 配置文件统一管理 (pyproject.toml)
- 结果输出标准化 (prediction_result/)
- 文档和代码分离管理

### 实验规范
- 每个项目包含README说明和性能指标
- 代码版本控制 (modeling_v1, v2...)
- 结果可重现和可验证
- 包含完整的错误处理和日志
- 竞赛级性能验证和优化

## 常用命令

### 环境管理
```bash
# 安装依赖
uv sync

# 添加新依赖
uv add package_name

# 运行代码
uv run python script.py

# 激活环境
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 开发环境安装
uv sync --group dev
```

### Jupyter Notebook
```bash
# 启动Jupyter
uv run jupyter notebook

# 或使用JupyterLab
uv run jupyter lab
```

### Web服务
```bash
# 启动FastAPI服务
uv run uvicorn main:app --reload

# 启动Flask服务
uv run python app.py
```

## 数据处理最佳实践

### 数据加载
```python
import pandas as pd
import numpy as np
from pathlib import Path

# 标准数据加载模式
def load_data(file_path):
    """标准数据加载函数"""
    return pd.read_csv(file_path, sep=' ', na_values=['-'])

def get_project_path(*paths):
    """获取项目路径的统一方法"""
    current_dir = Path(__file__).parent
    return current_dir.joinpath(*paths)
```

### 特征工程
```python
from sklearn.preprocessing import LabelEncoder, RobustScaler, QuantileTransformer
import pandas as pd
import numpy as np

# 标准特征工程模式
def create_features(df):
    """标准特征工程函数"""
    df = df.copy()
    # 添加特征工程逻辑
    return df

def enhanced_preprocessing():
    """增强数据预处理 - 基于项目最佳实践"""
    # 数据清洗、特征编码、异常值处理
    pass
```

### 模型训练
```python
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

def train_model(X, y, model, cv=5):
    """标准模型训练函数"""
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    return -scores.mean()

def ensemble_models(models, X, weights=None):
    """集成模型训练"""
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    if weights is None:
        weights = [1/len(models)] * len(models)
    
    return np.average(predictions, axis=0, weights=weights)
```

## 性能优化建议

### 1. 数据优化
- 使用适当的数据类型 (category, float32)
- 及时释放不需要的变量
- 批量处理大数据集
- 使用内存映射文件处理超大数据集
- 实施增量学习策略

### 2. 模型优化
- 使用交叉验证防止过拟合
- 尝试多种算法集成 (Stacking, Blending)
- 超参数自动调优 (贝叶斯优化)
- 特征重要性分析和选择
- 模型校准和后处理

### 3. 代码优化
- 向量化操作替代循环
- 使用并行处理 (joblib, multiprocessing)
- 缓存重复计算结果
- 使用JIT编译 (numba)
- 异步处理I/O密集型任务

### 4. 高级优化技术
- 分层建模策略
- 动态权重调整
- 分布式训练
- 模型压缩和量化

## 故障排除

### 常见问题
1. **依赖冲突**: 使用 `uv sync --refresh` 重新同步
2. **内存不足**: 减少数据样本或使用分块处理
3. **模型过拟合**: 增加正则化或减少模型复杂度
4. **环境问题**: 重新创建虚拟环境

### 调试技巧
```python
# 添加调试日志
import logging
from loguru import logger
logger.add("app.log", rotation="1 day")

# 使用断点调试
import pdb; pdb.set_trace()

# 性能分析
import cProfile
import pstats
cProfile.run('function_to_profile()', 'profile_output')
stats = pstats.Stats('profile_output')
stats.sort_stats('cumulative').print_stats(10)

# 内存分析
import tracemalloc
tracemalloc.start()
# 代码执行
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

## 贡献指南

### 提交代码
1. Fork 项目仓库
2. 创建功能分支
3. 编写测试用例
4. 提交Pull Request

### 文档更新
- 更新相关README文件
- 添加代码注释和文档字符串
- 更新API文档

## 新增功能与更新

### 最新技术集成
- **大模型本地部署**: Ollama集成，支持本地运行DeepSeek等模型
- **Agent开发**: Coze和Dify平台API集成
- **向量数据库**: Faiss集成，支持ChatPDF等应用
- **时间序列**: Prophet模型集成，专门处理时间序列预测
- **自动化工作流**: Dify工作流引擎集成
- **RAG技术**: 完整的检索增强生成技术栈，包括ChatPDF和知识库处理
- **Text2SQL**: Vanna和LangChain SQL工具链，支持自然语言转SQL
- **Function Calling**: 函数调用与多Agent协作框架
- **GraphRAG**: 图数据库增强的RAG技术
- **Query改写**: 智能查询重写和优化技术

### 项目优化成果
- **二手车价格预测**: 从v1迭代到v29，MAE从1000+优化到487.71，竞赛级性能
- **特征工程**: 建立了完整的特征分析和自动化流程
- **模型集成**: 实现了多模型Stacking和动态权重调整
- **校准技术**: 开发了智能校准算法提升预测精度
- **RAG应用**: 构建了多个生产级RAG应用案例
- **Text2SQL系统**: 实现了完整的自助式数据报表开发平台
- **LangChain应用**: 开发了故障诊断、工具链组合等复杂应用
- **Function Calling**: 构建了多Agent协作和工单处理系统
- **竞赛优化**: 多个版本在阿里云天池等竞赛中获得优异成绩

## 联系支持

- 项目仓库: https://github.com/lihaizhong/build-your-own-ai.git
- 问题反馈: 通过GitHub Issues提交
- 技术讨论: 项目相关技术社区

---

*本IFLOW指南将随项目发展持续更新，建议定期查看最新版本。*
*最后更新: 2025年11月1日*