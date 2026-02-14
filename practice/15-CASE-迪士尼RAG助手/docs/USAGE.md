# Disney RAG问答助手 - 使用指南

## 项目概述

Disney RAG问答助手是一个基于RAG（Retrieval-Augmented Generation）技术的智能问答系统，能够基于迪士尼相关文档和图像进行准确的问题回答。

### 核心功能

1. **文档处理** - 支持Word文档(.docx)、文本文件(.txt, .md)的解析和处理
2. **图像处理** - 支持图片OCR识别和CLIP视觉特征提取
3. **向量化** - 使用阿里云百炼text-embedding-v4和CLIP模型进行向量化
4. **混合检索** - 支持文本和图像的混合检索，关键词触发图像搜索
5. **智能问答** - 基于检索结果生成准确的答案

## 项目结构

```
15-CASE-迪士尼RAG助手/
├── code/                       # 核心代码
│   ├── __init__.py            # 模块初始化
│   ├── config.py              # 配置管理
│   ├── utils.py               # 工具函数
│   ├── data_processor.py      # 数据处理层（Step1）
│   ├── embedding.py           # 向量化层（Step2）
│   ├── retrieval.py           # 检索层（Step3）
│   ├── generator.py           # 生成层（Step4）
│   └── main.py                # 主程序入口
├── data/                       # 数据目录
│   ├── documents/             # 文档数据
│   └── images/                # 图像数据
├── docs/                       # 文档目录
│   └── USAGE.md               # 使用指南（本文件）
├── user_data/                  # 用户数据目录
│   ├── indexes/               # FAISS索引文件
│   └── cache/                 # 缓存目录
├── output/                     # 输出目录
│   └── logs/                  # 日志文件
├── tests/                      # 测试目录
└── README.md                   # 项目说明
```

## 环境要求

- Python >= 3.11
- uv 包管理器
- 根目录的虚拟环境

## 快速开始

### 1. 激活虚拟环境

```bash
# 激活根目录的虚拟环境
cd /Users/lihaizhong/Documents/Project/build-your-own-x/build-your-own-ai
source .venv/bin/activate
```

### 2. 验证依赖

根目录的 `pyproject.toml` 已包含所有必需的依赖，包括：
- `langchain>=1.2.9` - 大模型应用开发框架
- `faiss-cpu>=1.13.2` - 向量相似度搜索
- `dashscope>=1.25.11` - 阿里云通义千问API
- `transformers>=5.1.0` - CLIP模型（图像Embedding）
- `pytesseract>=0.3.13` - OCR文本识别
- `python-docx>=1.2.0` - Word文档处理
- `torch>=2.10.0` - 深度学习框架
- 以及其他相关依赖

### 3. 配置环境变量

确保根目录的 `.env` 文件包含：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

项目会自动从项目根目录的 `.env` 文件加载环境变量。

### 4. 准备数据

将文档和图像放入相应目录：

```bash
# 文档放入 data/documents/
cp your_document.docx data/documents/
cp your_text.md data/documents/

# 图像放入 data/images/
cp your_image.png data/images/
```

项目已包含示例文档和Word文件：
- `data/documents/` - 文档目录
- `data/images/` - 图像目录
- `data/1-上海迪士尼门票规则.docx`
- `data/2-迪士尼老人票价规定.docx`
- `data/3-迪士尼乐园游玩攻略清单.docx`
- `data/4-上海迪士尼乐园酒店会员制度.docx`

### 5. 构建索引

```bash
cd practice/15-CASE-迪士尼RAG助手
python -m code.main --build
```

### 6. 运行问答

#### 交互模式

```bash
python -m code.main --interactive
```

#### 单次查询

```bash
python -m code.main --query "迪士尼有哪些经典动画电影？"
```

#### 查看帮助

```bash
python -m code.main --help
```

## 使用示例

### 示例1: 文本问答

```bash
$ python -m code.main -q "迪士尼有哪些经典动画电影？"

问题: 迪士尼有哪些经典动画电影？

答案:
迪士尼有许多经典动画电影，包括：

1. 米老鼠系列 - 1928年创造的经典角色，代表作包括《威利汽船》和《糊涂交响曲》
2. 《白雪公主与七个小矮人》(1937) - 迪士尼第一部动画长片
3. 《木偶奇遇记》(1940) - 讲述木偶匹诺曹变成真正小男孩的故事
4. 《灰姑娘》(1950) - 经典的灰姑娘童话
5. 《睡美人》(1959) - 公主沉睡的故事
6. 《小美人鱼》(1989) - 标志迪士尼动画复兴
7. 《狮子王》(1994) - 最成功的迪士尼电影之一
```

### 示例2: 图像触发检索

```bash
$ python -m code.main -q "展示一下迪士尼的海报"

问题: 展示一下迪士尼的海报

答案:
迪士尼有众多经典电影的海报，包括：

1. 《白雪公主与七个小矮人》海报 - 展示白雪公主和七个小矮人的经典形象
2. 《狮子王》海报 - 辛巴站在荣耀岩上的震撼画面
3. 《冰雪奇缘》海报 - 艾莎和安娜的姐妹情深
4. 《小美人鱼》海报 - 爱丽儿在大海中的美丽形象

相关图像信息：
- 图像1 (来源: data/images/lion_king_poster.jpg, 相似度: 0.856)
  图像: data/images/lion_king_poster.jpg
- 图像2 (来源: data/images/snow_white_poster.jpg, 相似度: 0.832)
  图像: data/images/snow_white_poster.jpg
```

### 示例3: 交互模式

```bash
$ python -m code.main --interactive

╔══════════════════════════════════════════════════════════╗
║                                                          ║
║            Disney RAG 问答助手 v1.0.0                   ║
║                                                          ║
║       基于向量数据库的智能问答系统                       ║
║       支持文档处理、图像检索和混合问答                   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

输入 'quit' 或 'exit' 退出
输入 'help' 查看帮助信息
═══════════════════════════════════════════════════════════════

请输入您的问题: 迪士尼乐园在哪里？

═══════════════════════════════════════════════════════════════
正在查询: 迪士尼乐园在哪里？
═══════════════════════════════════════════════════════════════

问题: 迪士尼乐园在哪里？

答案:
迪士尼乐园遍布全球，共有六个主要度假区：

1. 加州迪士尼乐园 (美国阿纳海姆) - 1955年开幕，世界第一座迪士尼乐园
2. 华特迪士尼世界 (美国佛罗里达奥兰多) - 1971年开幕，最大的迪士尼度假村
3. 东京迪士尼度假区 (日本东京) - 1983年开幕，第一座美国以外的迪士尼乐园
4. 巴黎迪士尼乐园 (法国巴黎) - 1992年开幕，欧洲唯一的迪士尼乐园
5. 香港迪士尼乐园 (中国香港) - 2005年开幕
6. 上海迪士尼度假区 (中国上海) - 2016年开幕，最新开幕的迪士尼乐园

请输入您的问题: quit
再见！
```

## 技术架构

### Step1: 数据层

- **文档处理**: `data_processor.py` - 解析Word文档，提取文本段落和表格
- **图像处理**: `data_processor.py` - 支持图片OCR（Tesseract）和图像提取

### Step2: 向量化层

- **文本Embedding**: `embedding.py` - 使用阿里云百炼text-embedding-v4（1024维）
- **图像Embedding**: `embedding.py` - 使用CLIP模型提取特征（512维）
- **双索引系统**: `embedding.py` - FAISS分别构建文本和图像向量索引

### Step3: 检索层

- **混合检索**: `retrieval.py` - 文本查询使用语义相似度，图像查询使用CLIP编码器
- **关键词触发**: `retrieval.py` - 检测"海报"、"图片"等关键词触发图像检索

### Step4: 生成层

- **上下文组织**: `generator.py` - 将检索结果组织成结构化提示
- **答案生成**: `generator.py` - 使用大语言模型生成准确答案
- **流式输出**: `generator.py` - 支持流式生成答案

## 配置说明

主要配置在 `code/config.py` 中：

```python
@dataclass
class Config:
    # 路径配置
    project_root: Path = get_project_root()
    data_dir: Path | None = None
    documents_dir: Path | None = None  # 默认: data/documents
    images_dir: Path | None = None     # 默认: data/images
    indexes_dir: Path | None = None    # 默认: user_data/indexes
    cache_dir: Path | None = None      # 默认: user_data/cache
    output_dir: Path | None = None     # 默认: output
    
    # Embedding配置
    text_embedding_model: str = "text-embedding-v4"
    text_embedding_dim: int = 1024
    image_embedding_dim: int = 512
    clip_model_name: str = "ViT-B/32"
    
    # FAISS配置
    index_type: str = "IndexFlatL2"
    nlist: int = 100  # IVF索引的聚类中心数
    
    # 检索配置
    top_k: int = 5
    score_threshold: float = 0.7
    
    # LLM配置
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # OCR配置
    tesseract_config: str = "--psm 6"
    ocr_language: str = "chi_sim+eng"
    
    # 检索关键词触发
    image_keywords: list | None = None
```

默认图像检索关键词为：`["海报", "图片", "图像", "照片", "截图", "展示"]`

## 常见问题

### Q1: 如何添加新文档？

将文档文件放入 `data/documents/` 目录，然后重新构建索引：

```bash
python -m code.main --build
```

### Q2: 如何添加新图像？

将图像文件放入 `data/images/` 目录，然后重新构建索引：

```bash
python -m code.main --build
```

### Q3: 如何更换大语言模型？

可以通过环境变量设置或修改 `code/config.py` 中的 `llm_model` 配置：
- 通过环境变量：`LLM_MODEL=qwen-max`
- 或使用支持DashScope的其他模型（如 deepseek-chat、qwen-turbo 等）

### Q4: 索引文件保存在哪里？

索引文件保存在 `user_data/indexes/` 目录：
- `text_index.faiss` - 文本向量索引
- `text_documents.pkl` - 文本文档数据
- `image_index.faiss` - 图像向量索引
- `image_documents.pkl` - 图像数据

### Q5: 如何清理缓存？

删除 `user_data/cache/` 目录下的所有文件：

```bash
rm -rf user_data/cache/*
```

### Q6: 支持哪些文档格式？

支持 `.docx`、`.txt`、`.md` 格式。

### Q7: 支持哪些图像格式？

支持 `.png`、`.jpg`、`.jpeg`、`.gif`、`.bmp`、`.webp` 格式。

## 进阶使用

### 自定义关键词触发

修改 `code/config.py` 中的 `image_keywords` 列表：

```python
config.image_keywords = ["海报", "图片", "图像", "照片", "截图", "展示", "剧照"]
```

### 调整检索参数

修改 `code/config.py` 中的检索参数：

```python
config.top_k = 10  # 返回更多结果
config.score_threshold = 0.6  # 降低分数阈值
```

### 自定义LLM参数

修改 `code/config.py` 中的LLM参数：

```python
config.llm_temperature = 0.5  # 降低随机性
config.llm_max_tokens = 3000  # 增加输出长度
```

### 使用代码接口

```python
from pathlib import Path
from code.config import config, load_env_config
from code.data_processor import DocumentProcessor, ImageProcessor
from code.embedding import VectorStore
from code.retrieval import HybridRetriever
from code.generator import AnswerGenerator, RAGPipeline

# 加载环境变量
load_env_config()

# 构建索引
vector_store = VectorStore()
doc_processor = DocumentProcessor()
chunks = doc_processor.process_directory()
vector_store.build_text_index(chunks)

# 执行查询
retriever = HybridRetriever(vector_store)
generator = AnswerGenerator()
pipeline = RAGPipeline(retriever, generator)
result = pipeline.query("迪士尼有哪些经典动画电影？")
print(pipeline.format_response(result))
```

## 许可证

本项目遵循项目根目录下的LICENSE文件中定义的许可证。