"""
业务助手模块

使用 Qwen Agent + Function Calling 实现销售数据查询
"""
from .db import execute_sql, get_engine, test_connection
from .main import init_agent_service, run_gui, run_tui
from .tools import (
    call_tool,
    execute_custom_sql,
    get_monthly_sales,
    get_monthly_sales_growth,
    get_province_sales,
    get_top_channels,
)

__all__ = [
    # 数据库
    "get_engine",
    "execute_sql",
    "test_connection",
    # 工具函数
    "get_monthly_sales",
    "get_monthly_sales_growth",
    "get_province_sales",
    "get_top_channels",
    "execute_custom_sql",
    "call_tool",
    # 主程序
    "init_agent_service",
    "run_tui",
    "run_gui",
]
