"""
门票助手 - 基于 Qwen-Agent 框架的 SQL 查询 Agent

【学习要点】
本文件使用阿里巴巴的 Qwen-Agent 框架，而非 LangChain。
Qwen-Agent 是专为通义千问设计的 Agent 框架，提供：
- Assistant: 核心助手类，支持工具调用
- BaseTool: 工具基类，用于自定义工具
- WebUI: 内置的 Web 图形界面

【框架对比】
| 功能 | LangChain | Qwen-Agent |
|------|-----------|------------|
| Agent 类 | create_agent() | Assistant |
| 工具定义 | @tool 装饰器 | @register_tool + BaseTool |
| LLM 配置 | 初始化参数 | llm_cfg 字典 |
| 状态管理 | StateGraph | 内置 messages |

【注意事项】
1. dashscope 是阿里云 SDK，用于调用通义千问 API
2. 超时配置在 llm_cfg 中，不是 dashscope.timeout
3. API Key 建议通过环境变量 DASHSCOPE_API_KEY 配置
"""

import os
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine
from qwen_agent.tools.base import BaseTool, register_tool
import matplotlib.pyplot as plt
import numpy as np

# ====== Matplotlib 中文显示配置 ======
# 【学习要点】中文字体配置是数据可视化的常见问题
# SimHei: Windows 常用中文字体
# Arial Unicode MS: macOS 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# ====== DashScope API 配置 ======
# 【学习要点】dashscope 是阿里云的 LLM SDK
# API Key 优先从环境变量读取，避免硬编码
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
# 注意：dashscope 模块没有 timeout 属性
# 超时配置应在 llm_cfg 中设置，如：llm_cfg = {'timeout': 30}

# ====== 门票助手 system prompt 和函数描述 ======
system_prompt = """我是门票助手，以下是关于门票订单表相关的字段，我可能会编写对应的SQL，对数据进行查询
-- 门票订单表
CREATE TABLE tkt_orders (
    order_time DATETIME,             -- 订单日期
    account_id INT,                  -- 预定用户ID
    gov_id VARCHAR(18),              -- 商品使用人ID（身份证号）
    gender VARCHAR(10),              -- 使用人性别
    age INT,                         -- 年龄
    province VARCHAR(30),           -- 使用人省份
    SKU VARCHAR(100),                -- 商品SKU名
    product_serial_no VARCHAR(30),  -- 商品ID
    eco_main_order_id VARCHAR(20),  -- 订单ID
    sales_channel VARCHAR(20),      -- 销售渠道
    status VARCHAR(30),             -- 商品状态
    order_value DECIMAL(10,2),       -- 订单金额
    quantity INT                     -- 商品数量
);
一日门票，对应多种SKU：
Universal Studios Beijing One-Day Dated Ticket-Standard
Universal Studios Beijing One-Day Dated Ticket-Child
Universal Studios Beijing One-Day Dated Ticket-Senior
二日门票，对应多种SKU：
USB 1.5-Day Dated Ticket Standard
USB 1.5-Day Dated Ticket Discounted
一日门票、二日门票查询
SUM(CASE WHEN SKU LIKE 'Universal Studios Beijing One-Day%' THEN quantity ELSE 0 END) AS one_day_ticket_sales,
SUM(CASE WHEN SKU LIKE 'USB%' THEN quantity ELSE 0 END) AS two_day_ticket_sales
我将回答用户关于门票相关的问题

每当 exc_sql 工具返回 markdown 表格和图片时，你必须原样输出工具返回的全部内容（包括图片 markdown），不要只总结表格，也不要省略图片。这样用户才能直接看到表格和图片。
"""

functions_desc = [
    {
        "name": "exc_sql",
        "description": "对于生成的SQL，进行SQL查询",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_input": {
                    "type": "string",
                    "description": "生成的SQL语句",
                }
            },
            "required": ["sql_input"],
        },
    },
]

# ====== 会话隔离 DataFrame 存储 ======
# 用于存储每个会话的 DataFrame，避免多用户数据串扰
_last_df_dict = {}

def get_session_id(kwargs):
    """根据 kwargs 获取当前会话的唯一 session_id，这里用 messages 的 id"""
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    return None

# ====== exc_sql 工具类实现 ======
@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    """
    SQL查询工具，执行传入的SQL语句并返回结果，并自动进行可视化。
    
    【学习要点】Qwen-Agent 自定义工具开发
    
    1. @register_tool('tool_name') 装饰器注册工具
    2. 继承 BaseTool 基类
    3. 必须定义：
       - description: 工具描述（LLM 会根据此描述决定是否调用）
       - parameters: 参数列表（定义工具输入）
       - call() 方法: 实际执行逻辑
    
    【与 LangChain 对比】
    | 特性 | Qwen-Agent | LangChain |
    |------|------------|-----------|
    | 装饰器 | @register_tool | @tool |
    | 基类 | BaseTool | BaseTool |
    | 参数定义 | parameters 列表 | args_schema (Pydantic) |
    | 执行方法 | call(self, params: str) | _run(self, **kwargs) |
    """
    description = '对于生成的SQL，进行SQL查询，并自动可视化'
    parameters = [{
        'name': 'sql_input',
        'type': 'string',
        'description': '生成的SQL语句',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        """
        执行 SQL 查询并返回结果
        
        【学习要点】工具返回值格式
        - 返回字符串，LLM 会将其作为工具结果处理
        - 可以返回 Markdown 格式，包括表格和图片链接
        - 图片使用相对路径，WebUI 会自动渲染
        
        Args:
            params: JSON 字符串，包含 sql_input 参数
            **kwargs: 额外参数（如 messages 历史）
            
        Returns:
            Markdown 格式的查询结果 + 图片
        """
        import json
        import matplotlib.pyplot as plt
        import io, os, time
        import numpy as np
        args = json.loads(params)
        sql_input = args['sql_input']
        database = args.get('database', 'ubr')
        
        # 创建数据库连接
        # 【学习要点】SQLAlchemy 是 Python 最流行的 ORM
        engine = create_engine(
            f'mysql+mysqlconnector://student123:student321@rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306/{database}?charset=utf8mb4',
            connect_args={'connect_timeout': 10},
            pool_size=10,
            max_overflow=20
        )
        try:
            # 执行 SQL 查询
            df = pd.read_sql(sql_input, engine)
            
            # 转换为 Markdown 表格（只显示前 10 行）
            md = df.head(10).to_markdown(index=False)
            
            # 自动创建图片保存目录
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成唯一文件名（使用时间戳避免冲突）
            filename = f'bar_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            
            # 生成可视化图表
            generate_chart_png(df, save_path)
            
            # 返回 Markdown 格式的结果（表格 + 图片）
            # 【学习要点】WebUI 会自动渲染 Markdown 图片
            img_path = os.path.join('image_show', filename)
            img_md = f'![柱状图]({img_path})'
            return f"{md}\n\n{img_md}"
        except Exception as e:
            return f"SQL执行或可视化出错: {str(e)}"

# ========== 通用可视化函数 ========== 
def generate_chart_png(df_sql, save_path):
    """
    根据查询结果自动生成柱状图
    
    【学习要点】智能可视化策略
    1. 自动检测数据类型（数值型 vs 分类型）
    2. 如果有分类列，自动创建堆积柱状图
    3. 支持透视表转换，适配复杂查询结果
    
    Args:
        df_sql: pandas DataFrame，SQL 查询结果
        save_path: 图片保存路径
    """
    columns = df_sql.columns
    x = np.arange(len(df_sql))
    
    # 获取分类列（object 类型）
    object_columns = df_sql.select_dtypes(include='O').columns.tolist()
    if columns[0] in object_columns:
        object_columns.remove(columns[0])
    
    # 获取数值列
    num_columns = df_sql.select_dtypes(exclude='O').columns.tolist()
    
    if len(object_columns) > 0:
        # 有分类列：创建堆积柱状图
        # 【学习要点】pivot_table 是数据分析利器
        pivot_df = df_sql.pivot_table(
            index=columns[0],       # 第一列作为 X 轴
            columns=object_columns, # 分类列作为分组
            values=num_columns,     # 数值列作为 Y 轴
            fill_value=0
        )
        
        # 绘制堆积柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        bottoms = None
        for col in pivot_df.columns:
            ax.bar(pivot_df.index, pivot_df[col], bottom=bottoms, label=str(col))
            if bottoms is None:
                bottoms = pivot_df[col].copy()
            else:
                bottoms += pivot_df[col]
    else:
        # 无分类列：简单柱状图
        bottom = np.zeros(len(df_sql))
        for column in columns[1:]:
            plt.bar(x, df_sql[column], bottom=bottom, label=column)
            bottom += df_sql[column]
        plt.xticks(x, df_sql[columns[0]])
    
    plt.legend()
    plt.title("销售统计")
    plt.xlabel(columns[0])
    plt.ylabel("门票数量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ====== 初始化门票助手服务 ======
def init_agent_service():
    """初始化门票助手服务
    
    【学习要点】Qwen-Agent 的配置方式
    
    llm_cfg 参数说明：
    - model: 模型名称，如 'qwen-turbo', 'qwen-plus', 'qwen-max'
    - timeout: API 调用超时时间（秒）
    - retry_count: 失败重试次数
    
    Assistant 参数说明：
    - llm: LLM 配置字典
    - name: 助手名称
    - description: 助手描述
    - system_message: 系统提示词（定义 Agent 的角色和行为）
    - function_list: 工具名称列表
    """
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,      # API 调用超时时间（秒）
        'retry_count': 3,   # 失败重试次数
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='门票助手',
            description='门票查询与订单分析',
            system_message=system_prompt,
            function_list=['exc_sql'],  # 移除绘图工具
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def app_tui():
    """终端交互模式（Text User Interface）
    
    【学习要点】Agent 交互模式
    
    提供命令行交互界面，支持：
    - 连续对话：messages 列表保存对话历史
    - 文件输入：支持多模态输入
    - 流式响应：bot.run() 返回生成器
    
    【与 LangChain 对比】
    | 特性 | Qwen-Agent | LangChain |
    |------|------------|-----------|
    | 对话历史 | messages 列表 | ChatMessageHistory |
    | 流式输出 | bot.run() 生成器 | .stream() 方法 |
    | 多模态 | content 列表格式 | HumanMessage(content=...) |
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史（关键：保持上下文）
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('user question: ')
                # 获取可选的文件输入
                file = input('file url (press enter if no file): ').strip()
                
                # 输入验证
                if not query:
                    print('user question cannot be empty！')
                    continue
                    
                # 构建消息
                # 【学习要点】Qwen-Agent 消息格式
                # 纯文本: {'role': 'user', 'content': '文本'}
                # 多模态: {'role': 'user', 'content': [{'text': '文本'}, {'file': '文件URL'}]}
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("正在处理您的请求...")
                
                # 运行助手并处理响应
                # 【学习要点】bot.run() 是生成器，支持流式输出
                response = []
                for response in bot.run(messages):
                    print('bot response:', response)
                messages.extend(response)  # 将响应添加到历史
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式（Graphical User Interface）
    
    【学习要点】Qwen-Agent WebUI
    
    WebUI 是 Qwen-Agent 内置的 Web 界面：
    - 基于 Gradio 构建
    - 支持预设问题建议
    - 自动处理流式响应
    - 支持文件上传
    
    Args:
        chatbot_config: 界面配置
            - prompt.suggestions: 预设问题列表
    """
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        
        # 配置聊天界面，列举 3 个典型门票查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '2023年4、5、6月一日门票，二日门票的销量多少？帮我按照周进行统计',
                '2023年7月的不同省份的入园人数统计',
                '帮我查看2023年10月1-7日销售渠道订单金额排名',
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 运行模式选择
    app_gui()          # 图形界面模式（默认）