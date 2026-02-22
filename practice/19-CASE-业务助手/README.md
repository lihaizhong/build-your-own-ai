# 销售数据业务助手

使用 Qwen Agent + Function Calling 实现的销售数据查询助手。

## 功能特性

- 查询某个月份的销量和销售额
- 计算销量环比增长
- 查询各省份销售额
- 查询销售金额 Top N 渠道
- 支持自定义 SQL 查询

## 快速开始

### 1. 安装依赖

```bash
# 在项目根目录
uv sync
source .venv/bin/activate
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，配置 DashScope API Key
DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 初始化示例数据

```bash
# 进入项目目录
cd practice/19-CASE-业务助手

# 初始化 SQLite 数据库（默认）
python code/init_db.py

# 或指定参数
python code/init_db.py --start-date 2024-01-01 --end-date 2024-12-31 --num-records 3000
```

### 4. 设置数据库连接

```bash
# SQLite（使用刚才生成的数据库）
export DATABASE_URL="sqlite:///$(pwd)/data/sales.db"

# MySQL
export DATABASE_URL="mysql+mysqlconnector://user:password@localhost:3306/sales_db"
```

### 5. 运行业务助手

```bash
# Web 界面模式（默认）
python code/main.py

# 或终端交互模式
python code/main.py --mode tui
```

## 项目结构

```
19-CASE-业务助手/
├── code/
│   ├── __init__.py      # 模块初始化
│   ├── main.py          # 主程序入口
│   ├── db.py            # 数据库连接
│   ├── tools.py         # Function Calling 工具
│   └── init_db.py       # 示例数据初始化
├── data/
│   ├── sales.db         # SQLite 数据库
│   └── sales_sample.csv # 示例数据 CSV
├── output/              # 输出目录
└── README.md
```

## 使用示例

### Web 界面

启动后访问 Web 界面，可以输入自然语言问题：

- "2024年1月的销量是多少？"
- "2024年2月相比1月，销量环比增长多少？"
- "不同省份的销售额是多少？"
- "2024年第一季度销售金额Top3的渠道是哪些？"

### 终端模式

```bash
python code/main.py --mode tui

# 交互示例
用户: 2024年3月的销量是多少？
助手: | 月份   | 总销量 | 总销售额    |
|--------|--------|------------|
| 2024-03| 1,234  | ¥123,456.78|
```

## 自定义扩展

### 添加新工具

1. 在 `tools.py` 中添加工具函数：

```python
def get_product_sales(product_name: str, engine=None) -> str:
    """查询某产品的销售额"""
    sql = f"""
        SELECT product_name, SUM(total_amount) as total
        FROM sales
        WHERE product_name LIKE '%{product_name}%'
        GROUP BY product_name
    """
    df = execute_sql(sql, engine)
    return df.to_markdown(index=False)
```

2. 在 `main.py` 中注册工具：

```python
@register_tool("get_product_sales")
class GetProductSalesTool(BaseTool):
    description = "查询某产品的销售额"
    parameters = [...]

    def call(self, params: str, **kwargs) -> str:
        ...
```

3. 更新 `SYSTEM_PROMPT` 和 `function_list`

### 使用自己的 MySQL 数据库

1. 创建数据库和表：

```sql
CREATE DATABASE sales_db;
USE sales_db;

CREATE TABLE sales (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_date DATE NOT NULL,
    order_month VARCHAR(7) NOT NULL,
    province VARCHAR(50) NOT NULL,
    city VARCHAR(50),
    channel VARCHAR(100) NOT NULL,
    product_name VARCHAR(200),
    quantity INT NOT NULL DEFAULT 0,
    unit_price DECIMAL(10, 2) NOT NULL DEFAULT 0,
    total_amount DECIMAL(12, 2) NOT NULL DEFAULT 0,
    customer_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_order_date (order_date),
    INDEX idx_order_month (order_month),
    INDEX idx_province (province),
    INDEX idx_channel (channel)
);
```

2. 配置环境变量：

```bash
export DATABASE_URL="mysql+mysqlconnector://user:password@host:3306/sales_db"
```

## 技术栈

- **LLM**: 通义千问 (Qwen) via DashScope API
- **Agent 框架**: qwen-agent
- **数据库**: SQLite / MySQL
- **数据处理**: pandas, SQLAlchemy

## 注意事项

1. 确保 `DASHSCOPE_API_KEY` 已正确配置
2. 数据库连接需要正确设置 `DATABASE_URL`
3. 自定义 SQL 只允许 SELECT 查询，禁止修改操作
