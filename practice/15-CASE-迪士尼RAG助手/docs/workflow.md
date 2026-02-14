# Disney RAG问答助手 - 工作流程图

本文档描述了 `main.py` 的整体工作流程和系统架构。

## 系统架构概览

```mermaid
graph TB
    subgraph "入口层"
        MAIN[main.py<br/>主程序入口]
    end
    
    subgraph "配置层"
        CONFIG[config.py<br/>配置管理]
        ENV[.env<br/>环境变量]
    end
    
    subgraph "数据处理层"
        DOC_PROC[DocumentProcessor<br/>文档处理器]
        IMG_PROC[ImageProcessor<br/>图像处理器]
        DATA_DIR[(data/<br/>文档&图像)]
    end
    
    subgraph "向量化层"
        TEXT_EMB[TextEmbeddingModel<br/>文本向量化]
        IMG_EMB[ImageEmbeddingModel<br/>CLIP图像向量化]
        VECTOR_STORE[VectorStore<br/>向量存储管理]
        FAISS[(FAISS索引<br/>text_index.faiss<br/>image_index.faiss)]
    end
    
    subgraph "检索层"
        TEXT_RET[TextRetriever<br/>文本检索器]
        IMG_RET[ImageRetriever<br/>图像检索器]
        HYBRID_RET[HybridRetriever<br/>混合检索器]
    end
    
    subgraph "生成层"
        PROMPT_BUILDER[PromptBuilder<br/>提示词构建]
        ANSWER_GEN[AnswerGenerator<br/>答案生成器]
        RAG_PIPELINE[RAGPipeline<br/>RAG管道]
    end
    
    subgraph "LLM服务"
        DASHSCOPE[DashScope API<br/>通义千问]
    end

    %% 配置层连接
    CONFIG --> DOC_PROC
    CONFIG --> IMG_PROC
    CONFIG --> VECTOR_STORE
    CONFIG --> HYBRID_RET
    CONFIG --> ANSWER_GEN
    ENV --> CONFIG

    %% 入口层连接
    MAIN --> CONFIG
    MAIN --> DOC_PROC
    MAIN --> IMG_PROC
    MAIN --> VECTOR_STORE
    MAIN --> HYBRID_RET
    MAIN --> RAG_PIPELINE

    %% 数据处理层连接
    DATA_DIR --> DOC_PROC
    DATA_DIR --> IMG_PROC
    DOC_PROC --> |TextChunk| VECTOR_STORE
    IMG_PROC --> |ImageData| VECTOR_STORE

    %% 向量化层连接
    TEXT_EMB --> VECTOR_STORE
    IMG_EMB --> VECTOR_STORE
    VECTOR_STORE --> FAISS

    %% 检索层连接
    FAISS --> TEXT_RET
    FAISS --> IMG_RET
    TEXT_RET --> HYBRID_RET
    IMG_RET --> HYBRID_RET
    VECTOR_STORE --> TEXT_RET
    VECTOR_STORE --> IMG_RET

    %% 生成层连接
    HYBRID_RET --> |RetrievalResult| RAG_PIPELINE
    PROMPT_BUILDER --> ANSWER_GEN
    ANSWER_GEN --> RAG_PIPELINE
    ANSWER_GEN --> DASHSCOPE
```

## 模块详细说明

### 1. 入口层 (main.py)

主程序入口，提供三种运行模式：

| 命令参数 | 功能 | 描述 |
|---------|------|------|
| `--build` | 构建索引 | 处理文档和图像，构建FAISS向量索引 |
| `--build-text-only` | 仅构建文本索引 | 跳过图像处理，仅构建文本索引 |
| `--interactive` / `-i` | 交互式问答 | 启动交互式问答会话 |
| `--query` / `-q` | 单次查询 | 执行单次查询并返回结果 |

### 2. 配置层 (config.py)

管理项目所有配置参数：

```python
@dataclass
class Config:
    # 路径配置
    project_root: Path
    data_dir: Path          # 数据目录
    documents_dir: Path     # 文档目录
    images_dir: Path        # 图像目录
    indexes_dir: Path       # 索引目录
    
    # Embedding配置
    text_embedding_model: str    # 文本向量化模型
    text_embedding_dim: int      # 文本向量维度 (1024)
    image_embedding_dim: int     # 图像向量维度 (512)
    
    # 检索配置
    top_k: int              # 返回结果数量 (5)
    score_threshold: float  # 相似度阈值 (0.7)
    
    # LLM配置
    llm_model: str          # 模型名称 (qwen-max)
    llm_temperature: float  # 温度参数 (0.7)
```

### 3. 数据处理层 (data_processor.py)

#### DocumentProcessor - 文档处理器

```mermaid
flowchart LR
    A[文档目录] --> B[扫描文件]
    B --> C{文件类型?}
    C -->|.docx| D[解析Word文档]
    C -->|.txt/.md| E[读取文本文件]
    D --> F[提取段落&表格]
    E --> F
    F --> G[文本分块<br/>chunk_size=500<br/>overlap=50]
    G --> H[TextChunk列表]
```

#### ImageProcessor - 图像处理器

```mermaid
flowchart LR
    A[图像目录] --> B[扫描图像文件]
    B --> C[加载图像]
    C --> D[OCR识别<br/>Tesseract]
    D --> E[提取文本]
    E --> F[ImageData列表]
```

### 4. 向量化层 (embedding.py)

#### 文本向量化流程

```mermaid
flowchart LR
    A[TextChunk] --> B[TextEmbeddingModel]
    B --> C[DashScope API<br/>text-embedding-v4]
    C --> D[1024维向量]
    D --> E[FAISS索引]
```

#### 图像向量化流程

```mermaid
flowchart LR
    A[ImageData] --> B[ImageEmbeddingModel]
    B --> C[CLIP模型<br/>clip-vit-base-patch32]
    C --> D[512维向量]
    D --> E[FAISS索引]
```

### 5. 检索层 (retrieval.py)

#### 混合检索流程

```mermaid
flowchart TB
    A[用户查询] --> B[HybridRetriever]
    B --> C{关键词触发?}
    C -->|是| D[图像检索]
    C -->|否| E[仅文本检索]
    D --> F[TextRetriever]
    D --> G[ImageRetriever]
    E --> F
    F --> H[文本检索结果]
    G --> I[图像检索结果]
    H --> J[合并&排序]
    I --> J
    J --> K[返回Top-K结果]
```

**关键词触发机制**：当查询包含以下关键词时触发图像检索：
- 海报、图片、图像、照片、截图、展示

### 6. 生成层 (generator.py)

#### RAG管道流程

```mermaid
flowchart TB
    A[用户问题] --> B[HybridRetriever检索]
    B --> C[获取相关上下文]
    C --> D[PromptBuilder构建提示词]
    D --> E[系统提示词<br/>+ 用户提示词]
    E --> F[AnswerGenerator]
    F --> G[DashScope API<br/>qwen-max]
    G --> H[生成答案]
    H --> I[返回结果]
```


## 完整工作流程图

### 模式一：构建索引流程 (`--build`)

```mermaid
sequenceDiagram
    participant U as 用户
    participant M as main.py
    participant C as Config
    participant DP as DocumentProcessor
    participant IP as ImageProcessor
    participant VS as VectorStore
    participant TE as TextEmbeddingModel
    participant IE as ImageEmbeddingModel
    participant FS as FAISS索引

    U->>M: python main.py --build
    M->>C: 加载配置
    C-->>M: 返回配置

    M->>DP: process_directory()
    DP->>DP: 扫描文档目录
    DP->>DP: 解析文档(.docx/.txt/.md)
    DP->>DP: 文本分块(chunk_size=500)
    DP-->>M: 返回TextChunk列表

    M->>VS: build_text_index(chunks)
    VS->>TE: embed_text_chunks(chunks)
    TE->>TE: 调用DashScope API向量化
    TE-->>VS: 返回向量数组
    VS->>FS: 创建文本FAISS索引

    M->>IP: process_directory()
    IP->>IP: 扫描图像目录
    IP->>IP: OCR识别图像文本
    IP-->>M: 返回ImageData列表

    M->>VS: build_image_index(images)
    VS->>IE: embed_image_data_list(images)
    IE->>IE: CLIP模型向量化
    IE-->>VS: 返回向量数组
    VS->>FS: 创建图像FAISS索引

    M->>VS: save_indexes()
    VS->>FS: 保存索引文件
    M-->>U: 索引构建完成
```


### 模式二：交互式问答流程 (`--interactive`)

```mermaid
sequenceDiagram
    participant U as 用户
    participant M as main.py
    participant VS as VectorStore
    participant HR as HybridRetriever
    participant TR as TextRetriever
    participant IR as ImageRetriever
    participant PB as PromptBuilder
    participant AG as AnswerGenerator
    participant LLM as DashScope API

    U->>M: python main.py -i
    M->>VS: load_indexes()
    VS-->>M: 加载FAISS索引

    M->>HR: 创建HybridRetriever
    M->>AG: 创建AnswerGenerator
    M->>M: 创建RAGPipeline

    loop 交互循环
        U->>M: 输入问题
        M->>HR: retrieve(question)
        
        HR->>HR: 检查关键词触发
        HR->>TR: retrieve(query)
        TR->>TR: 文本向量化
        TR->>VS: FAISS搜索
        VS-->>TR: 相似文本块
        TR-->>HR: 文本检索结果

        alt 包含图像关键词
            HR->>IR: retrieve_by_text(query)
            IR->>IR: CLIP文本编码
            IR->>VS: FAISS搜索
            VS-->>IR: 相似图像
            IR-->>HR: 图像检索结果
        end

        HR-->>M: 混合检索结果

        M->>PB: build_context(results)
        PB-->>M: 上下文文本

        M->>AG: generate(query, results)
        AG->>LLM: 调用通义千问API
        LLM-->>AG: 生成答案
        AG-->>M: 返回答案

        M-->>U: 显示答案和上下文
    end
```


### 模式三：单次查询流程 (`--query`)

```mermaid
sequenceDiagram
    participant U as 用户
    participant M as main.py
    participant RP as RAGPipeline
    participant HR as HybridRetriever
    participant AG as AnswerGenerator
    participant LLM as DashScope API

    U->>M: python main.py -q "问题"
    M->>M: 加载索引
    M->>RP: 创建RAGPipeline

    M->>RP: query(question)
    
    Note over RP: Step 1: 检索
    RP->>HR: retrieve(query)
    HR->>HR: 混合检索(文本+图像)
    HR-->>RP: 检索结果

    Note over RP: Step 2: 构建上下文
    RP->>RP: build_context(results)

    Note over RP: Step 3: 生成答案
    RP->>AG: generate(query, results)
    AG->>LLM: API调用
    LLM-->>AG: 答案
    AG-->>RP: 答案

    RP-->>M: 返回结果字典
    M-->>U: 格式化输出
```

## 数据流图

### 数据流向概览

```mermaid
flowchart TB
    subgraph 输入数据
        DOCS[文档文件<br/>.docx/.txt/.md]
        IMGS[图像文件<br/>.png/.jpg/.jpeg]
        QUERY[用户查询]
    end

    subgraph 处理层
        DOC_PROC[DocumentProcessor]
        IMG_PROC[ImageProcessor]
    end

    subgraph 中间数据
        CHUNKS[TextChunk<br/>text + metadata]
        IMG_DATA[ImageData<br/>image_path + ocr_text]
    end

    subgraph 向量化
        TEXT_VEC[文本向量<br/>1024维]
        IMG_VEC[图像向量<br/>512维]
    end

    subgraph 索引存储
        TEXT_IDX[文本索引<br/>text_index.faiss]
        IMG_IDX[图像索引<br/>image_index.faiss]
    end

    subgraph 检索结果
        RET_RESULT[RetrievalResult<br/>content + score + metadata]
    end

    subgraph 输出
        ANSWER[生成的答案]
        CONTEXT[相关上下文]
    end

    DOCS --> DOC_PROC
    DOC_PROC --> CHUNKS
    CHUNKS --> TEXT_VEC
    TEXT_VEC --> TEXT_IDX

    IMGS --> IMG_PROC
    IMG_PROC --> IMG_DATA
    IMG_DATA --> IMG_VEC
    IMG_VEC --> IMG_IDX

    QUERY --> RET_RESULT
    TEXT_IDX --> RET_RESULT
    IMG_IDX --> RET_RESULT

    RET_RESULT --> CONTEXT
    RET_RESULT --> ANSWER
```


## 核心类关系图

```mermaid
classDiagram
    class Config {
        +Path project_root
        +Path data_dir
        +Path documents_dir
        +Path images_dir
        +Path indexes_dir
        +str text_embedding_model
        +int text_embedding_dim
        +int image_embedding_dim
        +int top_k
        +float score_threshold
        +str llm_model
        +from_dict() Config
        +to_dict() Dict
    }

    class TextChunk {
        +str text
        +str source
        +int page
        +str chunk_id
        +Dict metadata
    }

    class ImageData {
        +Path image_path
        +str ocr_text
        +List embedding
        +Dict metadata
    }

    class DocumentProcessor {
        +Path documents_dir
        +List~TextChunk~ chunks
        +process_directory() List~TextChunk~
        +process_file() List~TextChunk~
    }

    class ImageProcessor {
        +Path images_dir
        +List~ImageData~ images
        +process_directory() List~ImageData~
        +process_image() ImageData
    }

    class TextEmbeddingModel {
        +str api_key
        +str model_name
        +int embedding_dim
        +embed_text() List~float~
        +embed_texts() List~List~float~~
        +embed_text_chunks() Tuple
    }

    class ImageEmbeddingModel {
        +str model_name
        +int embedding_dim
        +CLIPModel model
        +embed_image() List~float~
        +embed_image_path() List~float~
        +embed_text_for_image_search() List~float~
    }

    class FAISSIndex {
        +int embedding_dim
        +str index_type
        +faiss.Index index
        +List documents
        +add_vectors()
        +search() List~Tuple~
        +save()
        +load()
    }

    class VectorStore {
        +FAISSIndex text_index
        +FAISSIndex image_index
        +TextEmbeddingModel text_embedding_model
        +ImageEmbeddingModel image_embedding_model
        +build_text_index() FAISSIndex
        +build_image_index() FAISSIndex
        +save_indexes()
        +load_indexes()
    }

    class RetrievalResult {
        +str content
        +str source
        +float score
        +Dict metadata
        +str result_type
        +to_dict() Dict
    }

    class TextRetriever {
        +VectorStore vector_store
        +FAISSIndex text_index
        +retrieve() List~RetrievalResult~
    }

    class ImageRetriever {
        +VectorStore vector_store
        +FAISSIndex image_index
        +retrieve_by_text() List~RetrievalResult~
        +retrieve_by_image() List~RetrievalResult~
    }

    class HybridRetriever {
        +VectorStore vector_store
        +TextRetriever text_retriever
        +ImageRetriever image_retriever
        +List image_keywords
        +retrieve() Dict
        +retrieve_unified() List~RetrievalResult~
    }

    class PromptBuilder {
        +str system_prompt
        +build_context() str
        +build_prompt() str
        +build_stream_prompt() Dict
    }

    class AnswerGenerator {
        +str api_key
        +str model
        +OpenAI client
        +PromptBuilder prompt_builder
        +generate() str
        +generate_stream() Iterator
    }

    class RAGPipeline {
        +HybridRetriever retriever
        +AnswerGenerator generator
        +PromptBuilder prompt_builder
        +query() Dict
        +query_stream() Iterator
        +format_response() str
    }

    Config <.. DocumentProcessor : 使用
    Config <.. ImageProcessor : 使用
    Config <.. VectorStore : 使用
    Config <.. HybridRetriever : 使用
    Config <.. AnswerGenerator : 使用
    
    DocumentProcessor --> TextChunk : 生成
    ImageProcessor --> ImageData : 生成
    
    TextEmbeddingModel <.. VectorStore : 使用
    ImageEmbeddingModel <.. VectorStore : 使用
    FAISSIndex <.. VectorStore : 管理
    
    VectorStore --> TextRetriever : 依赖
    VectorStore --> ImageRetriever : 依赖
    TextRetriever --> HybridRetriever : 组合
    ImageRetriever --> HybridRetriever : 组合
    
    HybridRetriever --> RetrievalResult : 返回
    RetrievalResult --> PromptBuilder : 输入
    PromptBuilder --> AnswerGenerator : 组合
    HybridRetriever --> RAGPipeline : 组合
    AnswerGenerator --> RAGPipeline : 组合
```


## 技术栈

| 层级 | 技术组件 | 用途 |
|------|---------|------|
| **数据处理** | python-docx, pytesseract, PIL | 文档解析、OCR识别 |
| **文本向量化** | DashScope TextEmbedding API | 文本转向量 (1024维) |
| **图像向量化** | CLIP (clip-vit-base-patch32) | 图像转向量 (512维) |
| **向量存储** | FAISS (IndexFlatL2) | 向量相似度搜索 |
| **LLM生成** | DashScope (qwen-max) | 答案生成 |
| **配置管理** | python-dotenv, dataclass | 环境变量、配置类 |
| **日志系统** | loguru | 日志记录 |

## 文件结构

```
practice/15-CASE-迪士尼RAG助手/
├── code/
│   ├── main.py           # 主程序入口
│   ├── config.py         # 配置管理
│   ├── data_processor.py # 数据处理层
│   ├── embedding.py      # 向量化层
│   ├── retrieval.py      # 检索层
│   ├── generator.py      # 生成层
│   └── utils.py          # 工具函数
├── data/
│   ├── documents/        # 文档目录
│   └── images/           # 图像目录
├── user_data/
│   └── indexes/          # FAISS索引存储
│       ├── text_index.faiss
│       ├── text_documents.pkl
│       ├── image_index.faiss
│       └── image_documents.pkl
├── docs/
│   ├── workflow.md       # 本文档
│   └── USAGE.md          # 使用说明
├── output/               # 输出目录
├── .env                  # 环境变量
└── pyproject.toml        # 项目配置
```

## 关键参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `text_embedding_dim` | 1024 | 文本向量维度 |
| `image_embedding_dim` | 512 | 图像向量维度 |
| `top_k` | 5 | 检索返回结果数 |
| `score_threshold` | 0.7 | 相似度阈值 |
| `llm_temperature` | 0.7 | LLM温度参数 |
| `llm_max_tokens` | 2000 | 最大生成token数 |
| `chunk_size` | 500 | 文本分块大小 |
| `chunk_overlap` | 50 | 分块重叠字符数 |

---

*文档生成时间: 2026-02-12*
