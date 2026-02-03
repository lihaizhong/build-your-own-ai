# AGENTS.md - Coding Guidelines for build-your-own-ai

## Project Overview

AI/ML学习项目，包含28个课程模块，涵盖大模型、传统机器学习、深度学习、Agent开发、RAG、Text2SQL、向量数据库等全面内容。项目基于Python开发，包含丰富的中文文档和实践案例。

**项目特色：**
- 28个完整课程模块，从入门到进阶
- 实战项目驱动，涵盖竞赛级案例
- 多种算法实现和对比分析
- 完整的MLOps流程实践

## Build/Lint/Test Commands

### Environment Setup
```bash
# 安装依赖（使用uv包管理器）
uv sync

# 安装开发依赖
uv sync --group dev

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 验证环境
python --version  # 应显示 Python 3.11.x
uv --version      # 验证uv已安装
```

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

## Code Style Guidelines

### Formatting
- **缩进**: 4空格（由.editorconfig强制）
- **行结束符**: LF（Unix风格）
- **字符集**: UTF-8
- **最大行长度**: 100字符（推荐）
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
from .local_utils import helper_function
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
- 将新依赖添加到`pyproject.toml`的dependencies数组
- 使用`uv add package_name`添加依赖
- 为稳定性固定版本（如 `numpy>=1.24,<2.0`）
- 使用版本范围而不是精确版本

```bash
# 添加新依赖
uv add pandas>=2.0.0

# 添加开发依赖
uv add --group dev pytest

# 更新现有依赖
uv add package_name@latest
```

### Git
- **不要提交**: `.env`, `__pycache__/`, `.venv/`, 模型文件, 数据文件, `catboost_info/`
- **应该提交**: `.env.example`, 文档, 代码文件, 配置文件
- **大文件**: 模型文件和数据文件应该使用Git LFS或外部存储

## Project Structure

```
build-your-own-ai/
├── courseware/              # 课程材料（28个模块）
│   ├── 00-开营直播/
│   ├── 01-AI大模型原理与API使用/
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
│   ├── 08-CASE-二手车价格预测-P1/
│   ├── 09-CASE-员工离职预测分析-P1/
│   ├── 11-CASE-资金流入流出预测-P1/
│   ├── 12-CASE-波士顿房价预测-P1/
│   ├── 12-CASE-激活函数示例/
│   ├── 13-CASE-钢铁缺陷检测-P1/
│   ├── 14-CASE-向量数据库与元数据管理/
│   ├── 15-CASE-创建你的RAG问答（LangChain）/
│   ├── 16-CASE-知识库处理/
│   └── 项目实战系列（4个项目）
├── notebook/                # Jupyter笔记本（学习笔记）
├── docs/                    # 文档
├── public/                  # 公共资源（图片等）
├── database/                # 数据库文件
├── .iflow/                  # iFlow配置
│   ├── agents/              # iFlow代理配置
│   ├── commands/            # iFlow命令配置
│   └── skills/              # iFlow技能
├── sub_modules/             # 子模块
├── typings/                 # 类型定义
├── .venv/                   # 虚拟环境（不提交）
├── pyproject.toml           # 项目配置和依赖
├── pyrightconfig.json       # PyRight类型检查配置
└── .editorconfig            # 编辑器配置
```

## Technology Stack

### 核心语言
- **Python**: 3.11+

### 包管理
- **uv**: 现代Python包管理器（替代pip/venv）
- **虚拟环境**: 自动管理的.venv目录

### 类型检查
- **basedpyright**: 基于PyRight的Python类型检查器

### 数据处理与科学计算
- **numpy**: >=1.24,<2.0
- **pandas**: >=2.3.2
- **scipy**: >=1.4.1,<1.14

### 机器学习
- **scikit-learn**: >=1.7.2
- **xgboost**: >=3.0.5
- **lightgbm**: >=4.6.0
- **catboost**: >=1.2.8
- **mlxtend**: >=0.23.4

### 深度学习
- **torch**: >=2.8.0
- **tensorflow**: >=2.18.1
- **transformers**: >=4.56.0
- **accelerate**: >=1.10.1
- **peft**: >=0.17.1

### 大模型API
- **openai**: >=1.102.0
- **dashscope**: >=1.24.2
- **cozepy**: >=0.19.0
- **modelscope**: >=1.29.1

### 时间序列
- **prophet**: >=1.1.7
- **statsmodels**: >=0.13.2,<0.14

### 向量数据库与RAG
- **faiss-cpu**: >=1.13.2
- **langchain**: >=0.1.0,<0.2.0
- **jieba**: >=0.42.1
- **gensim**: >=4.4.0
- **flagembedding**: >=1.3.5

### Web框架
- **fastapi**: >=0.116.1
- **flask**: >=3.1.2
- **uvicorn**: >=0.35.0

### 数据可视化
- **matplotlib**: >=3.10.6
- **seaborn**: ==0.12.2
- **plotly**: >=6.3.1
- **ydata-profiling**: >=4.6.0
- **missingno**: >=0.5.2

### 文档处理
- **python-docx**: >=1.2.0
- **pypdf2**: >=3.0.1
- **pymupdf**: >=1.26.7
- **pillow**: >=11.3.0
- **lxml**: >=6.0.2

### 其他工具
- **loguru**: >=0.7.3（日志）
- **tqdm**: >=4.67.1（进度条）
- **joblib**: >=1.5.2（模型序列化）
- **ipywidgets**: >=8.1.5（交互式组件）
- **python-dotenv**: >=1.1.1（环境变量）
- **requests**: >=2.32.5（HTTP请求）
- **tiktoken**: >=0.12.0（Token计数）

## Best Practices

### 项目实践建议

1. **虚拟环境管理**
   - 始终使用uv管理虚拟环境
   - 项目级别的依赖在`pyproject.toml`中定义
   - 不要提交.venv目录

2. **代码组织**
   - 每个课程模块有独立的结构
   - 实践项目包含完整的ML/DL流程
   - 使用标准化的目录结构

3. **数据处理**
   - 原始数据放在`data/`目录
   - 处理后的数据放在`processed/`目录
   - 大数据集使用压缩格式

4. **模型管理**
   - 训练好的模型放在`model/`目录
   - 使用版本号标识模型文件
   - 记录模型训练参数和性能

5. **可视化输出**
   - 图表保存到`user_data/`目录
   - 使用高分辨率（300 DPI）
   - 包含清晰的标题和标签

6. **文档编写**
   - 使用中文编写用户文档
   - 包含完整的使用示例
   - 记录依赖和环境要求

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
uv sync
```

### 类型检查错误
```bash
# 查看详细错误信息
basedpyright --verbose

# 检查特定文件
basedpyright path/to/problematic_file.py
```

### GPU相关问题
```bash
# 检查PyTorch GPU支持
python -c "import torch; print(torch.cuda.is_available())"

# 检查TensorFlow GPU支持
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 扩展资源

### 官方文档
- [uv官方文档](https://docs.astral.sh/uv/)
- [NumPy中文文档](https://numpy.org.cn/doc/2.3/index.html)
- [Pandas官方文档](https://pandas.pydata.org/docs/)
- [PyTorch官方文档](https://pytorch.org/docs/)
- [Transformers文档](https://huggingface.co/docs/transformers)

### 项目文档
- `README.md`: 项目概述和快速开始
- `IFLOW.md`: 详细的项目指南
- 各课程模块的README: 具体模块说明

### 学习资源
- 课程资料下载链接在README.md中
- 每个模块包含完整的学习材料
- 实践项目提供完整的代码实现

## 贡献指南

1. **代码贡献**
   - 遵循项目的代码风格
   - 添加适当的文档字符串
   - 确保代码通过类型检查
   - 更新相关文档

2. **文档贡献**
   - 使用中文编写文档
   - 包含完整的示例
   - 保持文档的及时更新

3. **问题报告**
   - 提供清晰的错误描述
   - 包含复现步骤
   - 附上相关的日志信息

## 许可证

本项目遵循项目根目录下的LICENSE文件中定义的许可证。

---

*最后更新: 2026年2月4日*
*版本: 2.0*