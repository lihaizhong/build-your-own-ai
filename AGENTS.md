# AGENTS.md - Coding Guidelines for build-your-own-ai

## Project Overview

AI/ML学习项目，包含28个课程模块，涵盖大模型、传统机器学习、深度学习、Agent开发、RAG、Text2SQL、向量数据库等全面内容。项目基于Python开发，包含丰富的中文文档和实践案例。

**项目特色：**
- 28个完整课程模块，从入门到进阶
- 实战项目驱动，涵盖竞赛级案例
- 多种算法实现和对比分析
- 完整的MLOps流程实践
- 统一的依赖管理环境

**项目结构：**
- `courseware/` - 课程材料（28个模块），包含课件PDF和参考代码
- `practice/` - 实践项目和案例研究
- `practice-py/` - Python基础练习册
- `notebook/` - 学习笔记目录（与课程模块对应）
- `public/` - 公共资源（图片等）
- `docs/` - 项目文档
- `.iflow/` - iFlow CLI配置和技能

## Build/Lint/Test Commands

### Environment Setup

**根目录环境：**
```bash
# 安装依赖（使用uv包管理器）
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 验证环境
python --version  # 应显示 Python 3.11+
uv --version      # 验证uv已安装
```

**子项目说明：**
- 本项目采用统一依赖管理，所有子项目共享根目录的虚拟环境
- 子项目不再维护独立的 `pyproject.toml` 文件
- 如需添加新依赖，在根目录执行 `uv add package_name`

### Type Checking
```bash
# 运行basedpyright类型检查器
basedpyright

# 检查特定文件
basedpyright path/to/file.py

# 检查特定目录
basedpyright courseware/
basedpyright practice/
```

### Testing
```bash
# 运行所有测试（如果存在）
python -m unittest discover -s . -p "*_test.py"

# 运行特定测试文件
python -m unittest path/to/test_file.py

# 运行特定测试类
python -m unittest path.to.test_file.TestClassName

# 运行特定测试方法
python -m unittest path.to.test_file.TestClassName.test_method_name

# 详细输出模式
python -m unittest -v path/to/test_file.py
```

### Running Python Scripts

**根目录执行：**
```bash
# 标准Python执行
python path/to/script.py

# 使用uv执行（推荐，自动使用虚拟环境）
uv run python path/to/script.py

# 运行Jupyter Notebook
uv run jupyter notebook

# 运行JupyterLab
uv run jupyter lab
```

**实践项目执行（RAG问答系统示例）：**
```bash
# 进入子项目目录
cd practice/15-CASE-创建RAG问答

# 运行主程序
python code/main.py

# 或使用uv运行
uv run python code/main.py

# 运行测试
python test_rag.py
python test_system.py
```

**课程模块示例：**
```bash
# 进入课程目录
cd courseware/02-DeepSeek使用与Prompt工程

# 运行Python脚本
python 1-情感分析-Deepseek-阿里代理.py

# 运行Jupyter Notebook
jupyter notebook 1-情感分析-Deepseek-阿里代理.ipynb
```

## Code Style Guidelines

### Formatting
- **缩进**: 4空格（由.editorconfig强制）
- **行结束符**: LF（Unix风格）
- **字符集**: UTF-8
- **尾随空格**: 不修剪（默认）
- **最终换行**: 不要求

### Imports
```python
# 1. 标准库导入
import os
import json
import sys
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

# 2. 第三方库导入
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger
import matplotlib.pyplot as plt

# 3. 本地导入
from my_module import my_function
# 相对导入需要在包内使用，脚本直接运行时需添加路径或使用绝对导入
try:
    from .local_utils import helper_function
except ImportError:
    from local_utils import helper_function
```

### Type Hints
- 为函数参数和返回值使用类型提示
- 使用`Optional[T]`表示可空类型
- 使用`List[T]`, `Dict[K, V]`等从typing模块导入
- 需要时内联禁用pyright检查：
```python
# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
from some_untyped_module import something
```

### Naming Conventions
- **函数**: `snake_case` (如 `load_model`, `get_response`)
- **变量**: `snake_case` (如 `model_path`, `api_key`)
- **常量**: `UPPER_CASE` (如 `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- **类**: `PascalCase` (如 `DataLoader`, `ModelTrainer`)
- **私有成员**: `_leading_underscore` 表示内部使用
- **模块**: `lowercase` (如 `data_preprocessing.py`)
- **测试文件**: `*_test.py` 或 `test_*.py`
- **测试类**: `Test*` (如 `TestDataLoader`)
- **测试方法**: `test_*` (如 `test_load_data`)

### Documentation
- 为模块、类和函数使用docstrings
- 用户文档使用中文
- 使用三引号表示多行docstrings
```python
def analyze_data(data: pd.DataFrame) -> Dict[str, Any]:
    """
    分析数据并返回统计结果
    
    Args:
        data: 输入的数据框
        
    Returns:
        包含统计结果的字典，包括均值、标准差等
        
    Raises:
        ValueError: 当输入数据为空时
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> analyze_data(df)
        {'mean': 2.0, 'std': 1.0}
    """
```

### Error Handling
- 使用特定异常，而非裸`except:`
- 使用loguru记录错误：`logger.error(f"Failed: {e}")`
- 优雅处理API失败，使用try-except
- 使用前检查环境变量是否存在
```python
from loguru import logger

try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    return None
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    return None
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### Environment Variables
- 使用`python-dotenv`从`.env`文件加载
- 始终检查环境变量是否存在并提供默认值
```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get('API_KEY')
if not api_key:
    logger.warning("API_KEY not set, using default")
    api_key = "default_key"
```

### Common Patterns

#### Project Path Helper
```python
def get_project_path(*paths: str) -> Path:
    """获取项目路径的统一方法"""
    try:
        current_dir = Path(__file__).parent
        project_dir = current_dir.parent
        return project_dir.joinpath(*paths)
    except NameError:
        # 在交互式环境中
        return Path.cwd().joinpath(*paths)
```

#### Main Function Pattern
```python
def main():
    """主函数"""
    print("=" * 60)
    print("项目标题")
    print("=" * 60)
    
    # 实现代码
    logger.info("Starting data processing...")
    # ...
    
    logger.info("Processing completed successfully")
    
if __name__ == "__main__":
    main()
```

#### Configuration Pattern
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """配置类"""
    model_path: str
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    random_seed: int = 42
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """从字典创建配置"""
        return cls(**config_dict)
```

### ML/DL Specific Guidelines
- 使用`np.random.seed()`确保可重现性
- 使用`tqdm`记录模型训练进度
- 将模型保存到`model/`目录
- 使用`joblib`保存sklearn模型
- 使用`torch.save()`保存PyTorch模型
- 使用`pickle`或`joblib`保存训练好的模型
- 记录超参数和性能指标

```python
import numpy as np
from joblib import dump, load
from torch import save as torch_save
from tqdm import tqdm

# 设置随机种子
np.random.seed(42)

# 训练时使用进度条
for epoch in tqdm(range(num_epochs), desc="Training"):
    # 训练代码
    pass

# 保存模型
dump(model, 'model/trained_model.joblib')
torch_save(model.state_dict(), 'model/pytorch_model.pth')
```

### Dependencies
- 将新依赖添加到根目录`pyproject.toml`的dependencies数组
- 使用`uv add package_name`添加依赖
- 为稳定性固定版本（如 `numpy>=2.4.0`）
- 使用版本范围而不是精确版本

```bash
# 添加新依赖
uv add pandas>=3.0.0

# 添加开发依赖
uv add --group dev pytest

# 更新现有依赖
uv add package_name@latest
```

### Git
- **不要提交**: `.env`, `__pycache__/`, `.venv/`, 模型文件, 数据文件, `catboost_info/`, `*.pyc`, `.ipynb_checkpoints/`
- **应该提交**: `.env.example`, 文档, 代码文件, 配置文件, `pyproject.toml`, `uv.lock`
- **大文件**: 模型文件和数据文件应该使用Git LFS或外部存储
- **分支管理**: 当前开发分支为 `develop`，主分支为 `main`

## Project Structure

```
build-your-own-ai/
├── courseware/              # 课程材料（28个模块）
│   ├── 00-开营直播/
│   ├── 01-AI大模型原理与API使用/
│   │   └── CASE-API使用/
│   ├── 02-DeepSeek使用与Prompt工程/
│   ├── 03-Cursor编程-从入门到精通/
│   ├── 04-Coze工作原理与应用实例/
│   ├── 05-Agent进阶实战与插件开发/
│   ├── 06-Dify本地化部署和应用/
│   ├── 07-分析式AI基础/
│   ├── 08-不同领域的AI算法/
│   ├── 09-机器学习神器/
│   ├── 10-时间序列模型/
│   ├── 11-时间序列AI大赛/
│   ├── 12-神经网络基础与Tensorflow实战/
│   ├── 13-Pytorch与视觉检测/
│   ├── 14-Embeddings和向量数据库/
│   ├── 15-RAG技术与应用/
│   ├── 16-RAG高级技术与最佳实践/
│   ├── 17-Text2SQL：自助式数据报表开发/
│   ├── 18-LangChain：多任务应用开发/
│   ├── 19-Function Calling与协作/
│   ├── 20-MCP与A2A的应用/
│   ├── 21-Agent智能体系统的设计与应用/
│   ├── 22-视觉大模型与多模态理解/
│   ├── 23-Fine-tuning微调艺术/
│   ├── 24-Fine-tuning实操/
│   ├── 25-项目实战：企业知识库/
│   ├── 26-项目实战：交互式BI报表/
│   ├── 27-项目实战：AI运营助手/
│   └── 28-项目实战：AI搜索类应用/
├── practice/                # 实践项目和案例研究
│   ├── 01-CASE-大模型原理与API使用/
│   ├── 02-CASE-DeepSeek使用与Prompt工程/
│   ├── 03-CASE-Cursor编程-从入门到精通/
│   ├── 06-CASE-Dify本地化部署和应用/
│   ├── 08-CASE-二手车价格预测-DataWhale/
│   ├── 08-CASE-二手车价格预测-P1/
│   ├── 09-CASE-员工离职预测分析-P1/
│   ├── 11-CASE-资金流入流出预测-P1/
│   ├── 12-CASE-波士顿房价预测-P1/
│   ├── 12-CASE-激活函数示例/
│   ├── 13-CASE-钢铁缺陷检测-P1/
│   ├── 14-CASE-三国演义Embedding/
│   ├── 14-CASE-向量数据库与元数据管理/
│   ├── 15-CASE-创建RAG问答/
│   ├── 15-CASE-迪士尼RAG助手/
│   ├── 16-CASE-知识库处理/
│   ├── 16-CASE-Query改写/
│   ├── 16-Query+联网搜索/
│   ├── 17-CASE-Text2SQL/
│   ├── 项目实战：交互式BI报表/
│   ├── 项目实战：企业知识库/
│   ├── 项目实战：AI搜索类应用/
│   ├── 项目实战：AI运营助手/
│   └── shared/              # 共享资源
├── practice-py/             # Python基础练习
│   ├── 作用域与作用域链.ipynb
│   └── LangChain管道语法.ipynb
├── notebook/                # Jupyter笔记本（学习笔记）
│   ├── 00-开营直播/ ~ 28-项目实战：AI搜索类应用/
│   ├── 笔记-20260130.md
│   └── 作业-20260208.md
├── docs/                    # 项目文档
│   ├── IFlow 配置区别.md
│   ├── iflow-agents-skills汇总.md
│   ├── iFlow模型特性与使用建议.md
│   └── npm/
├── public/                  # 公共资源（图片等）
├── .iflow/                  # iFlow CLI配置
│   ├── agents/              # iFlow代理配置（16个）
│   ├── settings.json
│   └── skills/              # iFlow技能（5个）
├── .venv/                   # 根目录虚拟环境（不提交）
├── pyproject.toml           # 项目配置和依赖
├── pyrightconfig.json       # PyRight类型检查配置
├── .editorconfig            # 编辑器配置
├── .gitignore               # Git忽略配置
├── .gitattributes           # Git属性配置
├── AGENTS.md                # 代理开发指南（本文件）
├── README.md                # 项目说明
└── LICENSE                  # 许可证
```

## Technology Stack

### 核心语言
- **Python**: >=3.11（当前系统：3.14.3，类型检查：3.12）

### 包管理
- **uv**: 现代Python包管理器（替代pip/venv）
- **统一依赖管理**: 所有子项目共享根目录虚拟环境
- **依赖查找顺序**:
  1. `.python-version` 文件设定的版本
  2. 当前启用的虚拟环境
  3. 当前目录的 `.venv` 目录
  4. uv 自己安装的 Python 环境
  5. 系统环境设定的 Python 环境

### 类型检查
- **basedpyright**: 基于PyRight的Python类型检查器

### 核心依赖（按功能分类）

#### 数据处理与科学计算
- **numpy**: >=2.4.2 - 数值计算
- **pandas**: >=3.0.0 - 数据分析
- **scipy**: 科学计算（间接依赖）

#### 机器学习
- **scikit-learn**: >=1.8.0 - 机器学习算法
- **xgboost**: >=3.1.3 - 梯度提升
- **lightgbm**: >=4.6.0 - 轻量级梯度提升
- **catboost**: >=1.2.8 - 分类提升
- **prophet**: >=1.3.0 - 时间序列预测
- **statsmodels**: >=0.14.6 - 统计模型

#### 深度学习
- **torch**: >=2.10.0 - PyTorch深度学习框架
- **torchvision**: >=0.25.0 - 计算机视觉
- **transformers**: >=5.1.0 - 预训练模型
- **ultralytics**: >=8.4.12 - YOLO目标检测

#### 大模型API
- **openai**: >=2.17.0 - OpenAI API
- **dashscope**: >=1.25.11 - 阿里云通义千问
- **cozepy**: >=0.20.0 - Coze平台API

#### 向量数据库与RAG
- **faiss-cpu**: >=1.13.2 - 向量相似度搜索
- **langchain**: >=1.2.9 - LLM应用开发框架
- **langchain-community**: >=0.4.1 - LangChain社区组件
- **langchain-openai**: >=1.1.7 - OpenAI集成
- **chromadb**: >=1.5.0 - 向量数据库
- **sentence-transformers**: >=5.2.2 - 句子向量化
- **rank-bm25**: >=0.2.2 - BM25检索
- **jieba**: >=0.42.1 - 中文分词
- **gensim**: >=4.4.0 - 文本相似度计算

#### Web框架
- **fastapi**: >=0.128.4 - 高性能Web框架
- **flask**: >=3.1.2 - 轻量级Web框架

#### 数据可视化
- **matplotlib**: >=3.10.8 - 基础绘图
- **seaborn**: >=0.13.2 - 统计数据可视化

#### 文档处理
- **python-docx**: >=1.2.0 - Word文档处理
- **pypdf**: >=6.6.2 - PDF处理
- **pypdf2**: >=3.0.1 - PDF文本提取
- **pymupdf**: >=1.26.7 - PDF高级处理
- **pillow**: >=12.1.0 - 图像处理
- **pytesseract**: >=0.3.13 - OCR文字识别

#### 数据库
- **sqlalchemy**: >=2.0.0 - SQL工具包

#### 其他工具
- **loguru**: >=0.7.3 - 日志记录
- **pydantic**: >=2.12.5 - 数据验证
- **python-dotenv**: >=1.2.1 - 环境变量管理
- **requests**: >=2.32.5 - HTTP请求
- **rich**: >=13.0.0 - 终端美化输出
- **tabulate**: >=0.9.0 - 表格格式化

## Best Practices

### 统一依赖管理

本项目采用统一依赖管理策略，所有子项目共享根目录的虚拟环境。

**工作流程：**

1. **开始开发时：**
   ```bash
   # 在项目根目录
   uv sync  # 安装所有依赖
   source .venv/bin/activate  # 激活虚拟环境
   ```

2. **添加新依赖：**
   ```bash
   # 在项目根目录执行
   uv add package_name
   uv add --group dev package_name  # 开发依赖
   ```

3. **环境变量管理：**
   - 项目根目录维护统一的 `.env` 文件
   - 从 `.env.example` 复制并填写配置
   ```bash
   cp .env.example .env
   # 编辑 .env 文件
   ```

4. **配置的API密钥：**
   - OpenAI API
   - DashScope API（阿里云通义千问）
   - Ollama API
   - 高德 API
   - Coze API
   - Dify API
   - MySQL 数据库

### 项目实践建议

1. **依赖管理**
   - 所有依赖在根目录 `pyproject.toml` 中统一管理
   - 使用 `uv add` 添加新依赖
   - 定期运行 `uv sync` 保持依赖同步

2. **代码组织**
   - 每个课程模块有独立的结构，包含课件和参考代码
   - 实践项目包含完整的ML/DL流程
   - 使用标准化的目录结构（code/data/docs）

3. **数据处理**
   - 原始数据放在子项目的 `data/` 目录
   - 处理后的数据放在子项目的 `output/` 目录
   - 大数据集使用压缩格式

4. **模型管理**
   - 训练好的模型放在子项目的 `model/` 目录（如果存在）
   - 使用版本号标识模型文件
   - 记录模型训练参数和性能

5. **可视化输出**
   - 图表保存到子项目的 `output/` 目录
   - 使用高分辨率（300 DPI）
   - 包含清晰的标题和标签

6. **文档编写**
   - 使用中文编写用户文档
   - 子项目包含独立的 README.md
   - 记录依赖和环境要求

7. **环境变量管理**
   - 使用统一的 `.env` 文件
   - 使用 `.env.example` 作为模板
   - 不要提交包含真实密钥的 `.env` 文件

### iFlow Agent 使用

项目配置了 16 个专业 Agent 和 5 个技能，详见 `docs/iflow-agents-skills汇总.md`。

**常用 Agent：**
- `ai-engineer`: LLM 应用开发、RAG 系统
- `code-reviewer`: 代码审查
- `python-pro`: Python 高级编程
- `data-analysis-agent`: 数据分析与可视化
- `docs-architect`: 技术文档编写

**模型选择建议：**
- 通用任务：GLM-4.7
- 代码任务：DeepSeek-V3.2、Qwen3-Coder-Plus
- 复杂推理：Kimi-K2-Thinking
- 文档处理：Kimi-K2.5

详见 `docs/iFlow模型特性与使用建议.md`。

### 性能优化

1. **数据加载**
   - 使用分块处理大数据集
   - 考虑使用HDF5或Parquet格式
   - 利用缓存机制

2. **训练优化**
   - 使用GPU加速（如可用）
   - 实现早停策略
   - 使用学习率调度

3. **内存管理**
   - 及时释放不再使用的变量
   - 使用生成器处理大型数据
   - 监控内存使用情况

### 调试技巧

1. **日志记录**
   ```python
   from loguru import logger
   
   logger.add("app.log", rotation="1 day")
   logger.info("Starting processing")
   logger.debug(f"Data shape: {data.shape}")
   ```

2. **性能分析**
   ```python
   import cProfile
   
   cProfile.run('function_to_profile()', 'profile_output')
   ```

3. **内存分析**
   ```python
   import tracemalloc
   
   tracemalloc.start()
   # 执行代码
   snapshot = tracemalloc.take_snapshot()
   ```

## 常见问题

### 依赖问题
```bash
# 解决依赖冲突
uv sync --refresh

# 重新创建虚拟环境
rm -rf .venv
uv venv --python 3.11
uv sync
```

### 类型检查错误
```bash
# 查看详细错误信息
basedpyright --verbose

# 检查特定文件
basedpyright path/to/problematic_file.py

# 检查特定目录
basedpyright courseware/02-DeepSeek使用与Prompt工程/
basedpyright practice/
```

### GPU相关问题
```bash
# 检查PyTorch GPU支持
python -c "import torch; print(torch.cuda.is_available())"

# 检查系统Python版本
python3 --version  # 当前系统: 3.14.3
```

### Jupyter Notebook问题
```bash
# 确保在正确的虚拟环境中
source .venv/bin/activate

# 安装Jupyter（如果未安装）
uv add jupyter

# 启动Jupyter
jupyter notebook
```

## 扩展资源

### 官方文档
- [uv官方文档](https://docs.astral.sh/uv/)
- [uv中文文档](https://hellowac.github.io/uv-zh-cn/)
- [NumPy中文文档](https://numpy.org.cn/doc/2.3/index.html)
- [Pandas官方文档](https://pandas.pydata.org/docs/)
- [PyTorch官方文档](https://pytorch.org/docs/)
- [Transformers文档](https://huggingface.co/docs/transformers)
- [LangChain文档](https://python.langchain.com/docs/)

### 项目文档
- `README.md`: 项目概述和快速开始
- `AGENTS.md`: 代理开发指南（本文件）
- `docs/IFlow 配置区别.md`: iFlow配置说明
- `docs/iflow-agents-skills汇总.md`: Agent和Skill汇总
- `docs/iFlow模型特性与使用建议.md`: 模型选择指南

### 学习资源
- [课程资料下载](https://pan.baidu.com/s/1MfjQwHba-dHav67tYVAWAw?pwd=8888)
- [Python3教程](https://www.runoob.com/python3/python3-tutorial.html)
- [大模型RAG基础](https://arthurchiao.art/blog/rag-basis-bge-zh/)
- [NLP教程](https://www.runoob.com/nlp/nlp-tutorial.html)

### 课程模块结构
- 每个模块包含课件PDF和参考代码
- 实践项目提供完整的代码实现
- 笔记目录包含学习笔记（与课程模块对应）

## 贡献指南

1. **代码贡献**
   - 遵循项目的代码风格
   - 为子项目添加适当的文档字符串和README
   - 确保代码通过类型检查
   - 更新相关文档

2. **文档贡献**
   - 使用中文编写文档
   - 包含完整的示例和环境配置说明
   - 保持文档的及时更新

3. **问题报告**
   - 提供清晰的错误描述
   - 说明问题发生的子项目和环境
   - 包含复现步骤
   - 附上相关的日志信息

## 许可证

本项目遵循项目根目录下的LICENSE文件中定义的许可证。

---

*最后更新: 2026年2月15日*
*版本: 4.0*
*项目根目录 Python 版本: 3.14.3*
*项目要求 Python 版本: >=3.11*
*类型检查 Python 版本: 3.12*
