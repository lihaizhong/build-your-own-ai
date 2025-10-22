# build your own ai - IFLOW 指南

## 项目概述

这是一个全面的AI学习和实践项目，专注于构建从基础到高级的人工智能应用。项目包含完整的机器学习课程材料、实践案例和优化实验，涵盖大模型API使用、传统机器学习、深度学习、Agent开发等多个AI领域。

## 核心技术栈

- **Python环境管理**: uv (现代Python包管理器)
- **机器学习**: scikit-learn, pandas, numpy, scipy
- **深度学习**: PyTorch, Transformers
- **大模型集成**: OpenAI, Dashscope, ModelScope
- **可视化**: matplotlib, seaborn, plotly
- **Web服务**: FastAPI, Flask, uvicorn
- **数据分析**: pandas-profiling, missingno
- **优化算法**: XGBoost, LightGBM, CatBoost

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
│   └── 14-Embeddings和向量数据库/   # 向量数据库
├── experiment/                     # 实验代码和案例
│   ├── 1-AI大模型原理与API使用/
│   ├── 2-DeepSeek使用与Prompt工程/
│   ├── 3-Cursor编程-从入门到精通/
│   ├── 4-Coze工作原理与应用实例/
│   ├── 5-Agent进阶实战与插件开发/
│   ├── 6-Dify本地化部署和应用/
│   ├── 7-分析式AI基础/
│   │   └── Case-二手车价格预测/    # 核心案例项目
│   ├── 8-不同领域的AI算法/
│   └── 9-机器学习神器/
└── public/                         # 公共资源
```

## 核心项目案例

### 🚗 二手车价格预测 (experiment/7-分析式AI基础/Case-二手车价格预测)

这是一个完整的机器学习项目，展示了从数据预处理到模型优化的全流程：

**项目特点**:
- 完整的数据预处理和特征工程
- 15个版本的模型迭代优化 (modeling_v1-v15)
- 多种算法集成 (RandomForest, LightGBM, XGBoost, CatBoost)
- 高级技术：分层建模、Stacking集成、智能校准
- 全面的数据分析和可视化

**运行方式**:
```bash
cd experiment/7-分析式AI基础/Case-二手车价格预测
python model/modeling_v15.py  # 最新优化版本
```

**性能目标**: MAE < 500 (目前最佳版本v12达到605分)

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

## 开发规范

### 代码规范
- 使用 Python 3.11+ 类型注解
- 遵循 PEP 8 代码风格
- 函数和类添加详细文档字符串
- 使用 uv 管理项目依赖

### 项目结构规范
- 每个实验目录包含完整的数据、代码、文档
- 使用模块化设计，功能分离清晰
- 配置文件统一管理 (pyproject.toml)
- 结果输出标准化 (prediction_result/)

### 实验规范
- 每个实验包含README说明
- 代码版本控制 (modeling_v1, v2...)
- 结果可重现和可验证
- 包含完整的错误处理和日志

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
source .venv/bin/activate
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

# 标准数据加载模式
def load_data(file_path):
    """标准数据加载函数"""
    return pd.read_csv(file_path, sep=' ', na_values=['-'])
```

### 特征工程
```python
# 标准特征工程模式
def create_features(df):
    """标准特征工程函数"""
    df = df.copy()
    # 添加特征工程逻辑
    return df
```

### 模型训练
```python
# 标准模型训练模式
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

def train_model(X, y, model):
    """标准模型训练函数"""
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    return -scores.mean()
```

## 性能优化建议

### 1. 数据优化
- 使用适当的数据类型 (category, float32)
- 及时释放不需要的变量
- 批量处理大数据集

### 2. 模型优化
- 使用交叉验证防止过拟合
- 尝试多种算法集成
- 超参数自动调优

### 3. 代码优化
- 向量化操作替代循环
- 使用并行处理
- 缓存重复计算结果

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 使用断点调试
import pdb; pdb.set_trace()

# 性能分析
import cProfile
cProfile.run('function_to_profile()')
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

## 联系支持

- 项目仓库: [GitHub Repository]
- 问题反馈: 通过Issues提交
- 技术讨论: [相关论坛或群组]

---

*本IFLOW指南将随项目发展持续更新，建议定期查看最新版本。*