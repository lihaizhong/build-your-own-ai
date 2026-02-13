# Text2SQL 智能查询系统

基于 Vanna + 大语言模型实现的自然语言转 SQL 查询系统。

## 功能特点

- 🤖 支持多种 LLM 提供商（通义千问、OpenAI、Ollama）
- 📊 内置王者荣耀英雄数据集（heros 数据库）
- 💾 支持 SQLite 和 MySQL 数据库
- 🎯 Few-shot 学习，提供示例问答对
- 💻 交互式命令行界面
- 🔧 可扩展训练数据

## 快速开始

### 1. 环境准备

项目使用根目录的虚拟环境，确保已安装依赖：

```bash
# 在项目根目录
cd /Users/lihaizhong/Documents/Project/build-your-own-x/build-your-own-ai
source .venv/bin/activate

# 安装额外依赖
uv pip install vanna[chromadb] rich tabulate
```

### 2. 配置 API Key

在根目录的 `.env` 文件中配置：

```bash
# 通义千问（推荐）
DASHSCOPE_API_KEY=your_api_key

# 或 OpenAI
OPENAI_API_KEY=your_api_key
```

### 3. 准备数据

```bash
cd practice/17-CASE-Text2SQL/code
python prepare_data.py
```

### 4. 运行

```bash
# 交互式界面
python cli.py

# 或指定 LLM 提供商
python cli.py --provider dashscope

# 运行演示
python __init__.py
```

## 使用示例

### 交互式查询

```
请输入问题: 查询所有战士类英雄

🔍 正在处理问题...

生成的 SQL:
SELECT * FROM heros WHERE role = '战士'

📊 执行查询...
┌──────────┬──────────┬──────┬──────┬─────────┐
│ hero_id  │ hero_name│ role │ ... │ health │
├──────────┼──────────┼──────┼──────┼─────────┤
│ 1        │ 亚瑟     │ 战士 │ 近战 │ 3500   │
│ 2        │ 吕布     │ 战士 │ 近战 │ 3800   │
│ ...      │ ...      │ ...  │ ...  │ ...    │
└──────────┴──────────┴──────┴──────┴─────────┘
```

### 示例问题

| 问题 | 说明 |
|------|------|
| 查询所有战士类英雄 | 简单条件查询 |
| 查询生命值最高的前5个英雄 | 排序+限制 |
| 统计每个定位有多少个英雄 | 分组统计 |
| 查询周免英雄有哪些 | 布尔条件 |
| 查询击杀数最高的3场比赛记录 | 多表关联 |

## 数据库结构

### heros 表（英雄信息）

| 字段 | 类型 | 说明 |
|------|------|------|
| hero_id | INTEGER | 英雄ID |
| hero_name | VARCHAR | 英雄名称 |
| role | VARCHAR | 定位（战士/法师/射手/辅助/坦克/刺客） |
| health | INTEGER | 生命值 |
| attack_damage | INTEGER | 攻击力 |
| ... | ... | ... |

### match_records 表（比赛记录）

| 字段 | 类型 | 说明 |
|------|------|------|
| match_id | INTEGER | 比赛ID |
| hero_id | INTEGER | 英雄ID |
| kill_count | INTEGER | 击杀数 |
| win | BOOLEAN | 是否获胜 |
| ... | ... | ... |

## 核心代码说明

### 1. 创建 Vanna 实例

```python
from text2sql_vanna import create_vanna

# 使用默认配置（通义千问）
vanna = create_vanna()

# 指定 LLM 提供商
vanna = create_vanna(llm_provider="openai")
```

### 2. 生成 SQL

```python
sql = vanna.generate_sql("查询所有战士类英雄")
print(sql)  # SELECT * FROM heros WHERE role = '战士'
```

### 3. 执行查询

```python
results = vanna.run_sql(sql)
for row in results:
    print(row)
```

### 4. 完整问答流程

```python
result = vanna.ask("查询生命值最高的5个英雄")
print(result["sql"])      # SQL 语句
print(result["results"])  # 查询结果
```

### 5. 添加训练数据

```python
vanna.train(
    question="查询法师类英雄的平均法术强度",
    sql="SELECT AVG(magic_damage) FROM heros WHERE role = '法师'"
)
```

## 项目结构

```
17-CASE-Text2SQL/
├── code/
│   ├── prepare_data.py      # 数据准备脚本
│   ├── text2sql_vanna.py    # 核心模块
│   ├── cli.py               # 命令行界面
│   └── __init__.py          # 示例代码
├── data/
│   ├── heros.db             # SQLite 数据库
│   └── chroma/              # 向量存储（可选）
├── docs/
│   └── database_schema.md   # 数据库文档
└── README.md
```

## 扩展使用

### 连接 MySQL 数据库

```python
# 修改 text2sql_vanna.py 中的数据库连接
import sqlalchemy

engine = sqlalchemy.create_engine(
    "mysql+pymysql://user:password@localhost/dbname"
)
```

### 使用 Ollama 本地模型

```bash
# 启动 Ollama
ollama serve

# 运行 CLI
python cli.py --provider ollama
```

## 注意事项

1. 确保配置了正确的 API Key
2. 首次使用需运行 `prepare_data.py` 创建数据库
3. 生成的 SQL 可能需要人工校验
4. 复杂查询建议添加更多训练示例

## 参考资料

- [Vanna 官方文档](https://vanna.ai/)
- [LangChain SQL Agent](https://python.langchain.com/docs/use_cases/sql/)
- [通义千问 API](https://help.aliyun.com/zh/dashscope/)
