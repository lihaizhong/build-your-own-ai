# AGENTS.md - Coding Guidelines for build-your-own-ai

## Project Overview

AI/ML learning project with 28 course modules covering LLMs, traditional ML, deep learning, Agent development, RAG, Text2SQL, vector databases, and more. Python-based with extensive Chinese documentation.

## Build/Lint/Test Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --group dev

# Activate virtual environment
source .venv/bin/activate
```

### Type Checking
```bash
# Run basedpyright type checker
basedpyright

# Check specific file
basedpyright path/to/file.py
```

### Testing
```bash
# Run all tests
python -m unittest discover -s . -p "*_test.py"

# Run specific test file
python -m unittest path/to/test_file.py

# Run specific test class
python -m unittest path.to.test_file.TestClassName

# Run specific test method
python -m unittest path.to.test_file.TestClassName.test_method_name

# Run test with verbosity
python -m unittest -v path/to/test_file.py
```

### Running Python Scripts
```bash
# Run a script
python path/to/script.py

# Run with uv
uv run python path/to/script.py
```

## Code Style Guidelines

### Formatting
- **Indentation**: 4 spaces (enforced by .editorconfig)
- **Line Endings**: LF (Unix-style)
- **Charset**: UTF-8
- **Max Line Length**: 100 characters (recommended)
- **Trailing Whitespace**: Not trimmed by default
- **Final Newline**: Not required

### Imports
```python
# 1. Standard library imports first
import os
import json
import sys
from typing import List, Dict, Optional

# 2. Third-party imports second
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger

# 3. Local imports last
from my_module import my_function
```

### Type Hints
- Use type hints for function parameters and return values
- Use `Optional[T]` for nullable types
- Use `List[T]`, `Dict[K, V]` from typing module
- Disable pyright checks inline when needed:
```python
# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
```

### Naming Conventions
- **Functions**: `snake_case` (e.g., `load_model`, `get_response`)
- **Variables**: `snake_case` (e.g., `model_path`, `api_key`)
- **Constants**: `UPPER_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- **Classes**: `PascalCase` (e.g., `DataLoader`, `ModelTrainer`)
- **Private**: `_leading_underscore` for internal use
- **Modules**: `lowercase` (e.g., `data_preprocessing.py`)
- **Test Files**: `*_test.py` or `test_*.py`
- **Test Classes**: `Test*` (e.g., `TestDataLoader`)
- **Test Methods**: `test_*` (e.g., `test_load_data`)

### Documentation
- Use docstrings for modules, classes, and functions
- Use Chinese for user-facing documentation
- Use triple quotes for multi-line docstrings
```python
def analyze_data(data: pd.DataFrame) -> Dict:
    """
    分析数据并返回统计结果
    
    Args:
        data: 输入的数据框
        
    Returns:
        包含统计结果的字典
    """
```

### Error Handling
- Use specific exceptions, not bare `except:`
- Log errors with loguru: `logger.error(f"Failed: {e}")`
- Handle API failures gracefully with try-except
- Check environment variables exist before use
```python
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    return None
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### Environment Variables
- Load from `.env` file using `python-dotenv`
- Always check if env var exists with fallback
```python
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get('API_KEY')
if not api_key:
    raise ValueError("API_KEY environment variable not set")
```

### Common Patterns

#### Project Path Helper
```python
def get_project_path(*paths):
    """获取项目路径的统一方法"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        return os.path.join(project_dir, *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)
```

#### Main Function Pattern
```python
def main():
    """主函数"""
    print("=" * 60)
    print("标题")
    print("=" * 60)
    # Implementation
    
if __name__ == "__main__":
    main()
```

### ML/DL Specific Guidelines
- Use `np.random.seed()` for reproducibility
- Log model training progress with `tqdm`
- Save models to `model/` directory
- Use `joblib` for sklearn model persistence
- Use `torch.save()` for PyTorch models

### Dependencies
- Add new deps to `pyproject.toml` dependencies array
- Use `uv add package_name` to add dependencies
- Pin versions for stability (e.g., `numpy>=1.24,<2.0`)

### Git
- Don't commit: `.env`, `__pycache__/`, `.venv/`, model files, data files
- Do commit: `.env.example`, documentation, code files

## Project Structure

```
build-your-own-ai/
├── courseware/          # Course materials (28 modules)
├── practice/           # Practice projects and case studies
├── notebook/           # Jupyter notebooks
├── docs/              # Documentation
├── public/            # Public assets
├── .iflow/            # iFlow configuration
└── .venv/             # Virtual environment (not committed)
```

## Technology Stack

- Python 3.11+
- uv (package manager)
- basedpyright (type checking)
- PyTorch, TensorFlow, Transformers
- scikit-learn, pandas, numpy
- FastAPI, Flask
- FAISS, LangChain
- loguru (logging)
