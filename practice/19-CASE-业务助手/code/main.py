"""
业务助手主程序

使用 Qwen Agent + Function Calling 实现销售数据查询助手
支持：
1. 查询某个月份的销量
2. 计算销量环比增长
3. 查询各省销售额
4. 查询 Top N 销售渠道
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional

import dashscope
from dotenv import load_dotenv
from loguru import logger
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

# 支持直接运行
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from code.tools import (
        execute_custom_sql,
        get_monthly_sales,
        get_monthly_sales_growth,
        get_province_sales,
        get_top_channels,
    )
else:
    from .tools import (
        execute_custom_sql,
        get_monthly_sales,
        get_monthly_sales_growth,
        get_province_sales,
        get_top_channels,
    )

# 加载环境变量
load_dotenv()

# 配置 DashScope
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
dashscope.timeout = 60 # type: ignore

# 系统提示词
SYSTEM_PROMPT = """你是企业销售数据分析助手，可以帮助用户查询和分析销售数据。

## 数据库表结构

以下是销售数据表 `sales` 的字段说明：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT | 主键ID |
| order_date | DATE | 订单日期 |
| order_month | VARCHAR(7) | 订单月份，格式 YYYY-MM |
| province | VARCHAR(50) | 省份 |
| city | VARCHAR(50) | 城市 |
| channel | VARCHAR(100) | 销售渠道 |
| product_name | VARCHAR(200) | 产品名称 |
| quantity | INT | 销售数量 |
| unit_price | DECIMAL(10,2) | 单价 |
| total_amount | DECIMAL(12,2) | 销售金额 |
| customer_id | VARCHAR(50) | 客户ID |

## 可用工具

1. `get_monthly_sales`: 查询某个月份的销量和销售额
   - 参数: month (格式: YYYY-MM)
   
2. `get_monthly_sales_growth`: 计算某月相比上月的销量环比增长
   - 参数: month (格式: YYYY-MM)
   
3. `get_province_sales`: 查询各省份的销售额
   - 参数: start_month, end_month (可选，格式: YYYY-MM)
   
4. `get_top_channels`: 查询某时间段销售金额 Top N 的渠道
   - 参数: start_date, end_date (可选，格式: YYYY-MM-DD), top_n (默认3)
   
5. `execute_custom_sql`: 执行自定义 SELECT 查询
   - 参数: sql (SELECT 语句)

## 回答规范

1. 使用中文回答
2. 销售金额保留两位小数，使用 ¥ 符号
3. 大数字使用千分位分隔符
4. 增长率使用百分比格式，正数显示 + 号
5. 如果用户问题不明确，主动询问澄清
"""

# Function Calling 工具定义
FUNCTIONS_DESC = [
    {
        "name": "get_monthly_sales",
        "description": "查询某个月份的销量和销售额",
        "parameters": {
            "type": "object",
            "properties": {
                "month": {
                    "type": "string",
                    "description": "月份，格式 YYYY-MM，例如 2024-01",
                }
            },
            "required": ["month"],
        },
    },
    {
        "name": "get_monthly_sales_growth",
        "description": "计算某个月份相比上个月的销量环比增长",
        "parameters": {
            "type": "object",
            "properties": {
                "month": {
                    "type": "string",
                    "description": "当前月份，格式 YYYY-MM，例如 2024-02",
                }
            },
            "required": ["month"],
        },
    },
    {
        "name": "get_province_sales",
        "description": "查询不同省份的销售额，可指定时间范围",
        "parameters": {
            "type": "object",
            "properties": {
                "start_month": {
                    "type": "string",
                    "description": "开始月份，格式 YYYY-MM（可选）",
                },
                "end_month": {
                    "type": "string",
                    "description": "结束月份，格式 YYYY-MM（可选）",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_top_channels",
        "description": "查询某时间段销售金额 Top N 的渠道",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "开始日期，格式 YYYY-MM-DD（可选）",
                },
                "end_date": {
                    "type": "string",
                    "description": "结束日期，格式 YYYY-MM-DD（可选）",
                },
                "top_n": {
                    "type": "integer",
                    "description": "返回前几名，默认 3",
                },
            },
            "required": [],
        },
    },
    {
        "name": "execute_custom_sql",
        "description": "执行自定义 SQL SELECT 查询（仅限 SELECT 语句）",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SELECT 查询语句",
                }
            },
            "required": ["sql"],
        },
    },
]


# ====== 注册工具类 ======

# 保护性注册：检查工具是否已存在
def _safe_register_tool(name: str):
    """安全注册工具的装饰器，避免重复注册"""
    from qwen_agent.tools.base import TOOL_REGISTRY
    if name in TOOL_REGISTRY:
        def decorator(cls):
            return cls
        return decorator
    return register_tool(name)


@_safe_register_tool("get_monthly_sales")
class GetMonthlySalesTool(BaseTool):
    """查询月度销量工具"""
    
    description = "查询某个月份的销量和销售额"
    parameters = [{
        "name": "month",
        "type": "string",
        "description": "月份，格式 YYYY-MM",
        "required": True,
    }]
    
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        month = args.get("month")
        if not month:
            return "请提供月份参数"
        return get_monthly_sales(month)


@_safe_register_tool("get_monthly_sales_growth")
class GetMonthlySalesGrowthTool(BaseTool):
    """计算销量环比增长工具"""
    
    description = "计算某个月份相比上个月的销量环比增长"
    parameters = [{
        "name": "month",
        "type": "string",
        "description": "当前月份，格式 YYYY-MM",
        "required": True,
    }]
    
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        month = args.get("month")
        if not month:
            return "请提供月份参数"
        return get_monthly_sales_growth(month)


@_safe_register_tool("get_province_sales")
class GetProvinceSalesTool(BaseTool):
    """查询省份销售额工具"""
    
    description = "查询不同省份的销售额"
    parameters = [
        {
            "name": "start_month",
            "type": "string",
            "description": "开始月份（可选）",
            "required": False,
        },
        {
            "name": "end_month",
            "type": "string",
            "description": "结束月份（可选）",
            "required": False,
        },
    ]
    
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        start_month = args.get("start_month")
        end_month = args.get("end_month")
        return get_province_sales(start_month, end_month)


@_safe_register_tool("get_top_channels")
class GetTopChannelsTool(BaseTool):
    """查询 Top 渠道工具"""
    
    description = "查询某时间段销售金额 Top N 的渠道"
    parameters = [
        {
            "name": "start_date",
            "type": "string",
            "description": "开始日期（可选）",
            "required": False,
        },
        {
            "name": "end_date",
            "type": "string",
            "description": "结束日期（可选）",
            "required": False,
        },
        {
            "name": "top_n",
            "type": "integer",
            "description": "返回前几名，默认 3",
            "required": False,
        },
    ]
    
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        start_date = args.get("start_date")
        end_date = args.get("end_date")
        top_n = args.get("top_n", 3)
        return get_top_channels(start_date, end_date, top_n)


@_safe_register_tool("execute_custom_sql")
class ExecuteCustomSQLTool(BaseTool):
    """执行自定义 SQL 工具"""
    
    description = "执行自定义 SQL SELECT 查询"
    parameters = [{
        "name": "sql",
        "type": "string",
        "description": "SELECT 查询语句",
        "required": True,
    }]
    
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        sql = args.get("sql")
        if not sql:
            return "请提供 SQL 查询语句"
        return execute_custom_sql(sql)


# ====== 初始化助手 ======

def init_agent_service() -> Assistant:
    """
    初始化业务助手服务
    
    Returns:
        Assistant 实例
    """
    llm_cfg = {
        "model": "qwen-turbo-latest",
        "timeout": 60,
        "retry_count": 3,
    }
    
    try:
        bot = Assistant(
            llm=llm_cfg,
            name="销售数据助手",
            description="企业销售数据查询与分析助手",
            system_message=SYSTEM_PROMPT,
            function_list=[
                "get_monthly_sales",
                "get_monthly_sales_growth",
                "get_province_sales",
                "get_top_channels",
                "execute_custom_sql",
            ],
        )
        logger.info("业务助手初始化成功")
        return bot
    except Exception as e:
        logger.error(f"业务助手初始化失败: {e}")
        raise


# ====== 终端交互模式 ======

def run_tui():
    """终端交互模式"""
    logger.info("启动终端交互模式")
    
    bot = init_agent_service()
    messages: List[Dict] = []
    
    print("=" * 60)
    print("销售数据业务助手")
    print("=" * 60)
    print("可用查询:")
    print("  1. 查询某月销量: 2024年1月的销量是多少？")
    print("  2. 环比增长: 2024年2月的销量环比增长多少？")
    print("  3. 省份销售额: 各省的销售额是多少？")
    print("  4. Top 渠道: 2024年第一季度销售金额Top3的渠道")
    print("  输入 'quit' 退出")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n用户: ").strip()
            
            if query.lower() == "quit":
                print("再见！")
                break
            
            if not query:
                continue
            
            messages.append({"role": "user", "content": query})
            
            print("\n助手: ", end="")
            response_text = ""
            
            for response in bot.run(messages): # type: ignore
                if response:
                    for msg in response:
                        if msg.get("role") == "assistant" and msg.get("content"):
                            content = msg["content"]
                            if isinstance(content, str):
                                response_text = content
            
            print(response_text)
            messages.append({"role": "assistant", "content": response_text})
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            logger.error(f"处理请求出错: {e}")
            print(f"处理出错: {e}")


# ====== Web 界面模式 ======

def run_gui():
    """Web 图形界面模式"""
    # 延迟导入 WebUI，避免在没有 GUI 依赖时影响终端模式
    try:
        from qwen_agent.gui import WebUI
    except ImportError as e:
        logger.error(f"GUI 依赖未安装: {e}")
        print("GUI 依赖未安装，请运行: pip install 'qwen-agent[gui]'")
        print("或使用终端模式: python main.py --mode tui")
        return
    
    logger.info("启动 Web 界面模式")
    
    bot = init_agent_service()
    
    chatbot_config = {
        "prompt.suggestions": [
            "2024年1月的销量是多少？",
            "2024年2月相比1月，销量环比增长多少？",
            "不同省份的销售额是多少？",
            "2024年第一季度销售金额Top3的渠道是哪些？",
        ]
    }
    
    WebUI(bot, chatbot_config=chatbot_config).run()


# ====== 主程序入口 ======

def main():
    """主程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="销售数据业务助手")
    parser.add_argument(
        "--mode",
        choices=["tui", "gui"],
        default="gui",
        help="运行模式: tui(终端) 或 gui(Web界面)",
    )
    
    args = parser.parse_args()
    
    if args.mode == "tui":
        run_tui()
    else:
        run_gui()


if __name__ == "__main__":
    main()
