"""
业务助手工具函数

实现销售数据查询的 Function Calling 工具
"""
import json
import os
import sys
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

# 支持直接运行
try:
    from .db import execute_sql, get_engine
except ImportError:
    from db import execute_sql, get_engine


def get_monthly_sales(month: str, engine: Optional[Any] = None) -> str:
    """
    查询某个月份的销量和销售额
    
    Args:
        month: 月份，格式 YYYY-MM
        engine: 数据库引擎
        
    Returns:
        查询结果的 Markdown 表格
    """
    sql = f"""
        SELECT 
            order_month as 月份,
            SUM(quantity) as 总销量,
            ROUND(SUM(total_amount), 2) as 总销售额
        FROM sales
        WHERE order_month = '{month}'
        GROUP BY order_month
    """
    
    try:
        df = execute_sql(sql, engine)
        if df.empty:
            return f"未找到 {month} 的销售数据"
        return df.to_markdown(index=False)
    except Exception as e:
        logger.error(f"查询月度销量失败: {e}")
        return f"查询失败: {str(e)}"


def get_monthly_sales_growth(month: str, engine: Optional[Any] = None) -> str:
    """
    计算某个月份相比上个月的销量环比增长
    
    Args:
        month: 当前月份，格式 YYYY-MM
        engine: 数据库引擎
        
    Returns:
        环比增长分析结果
    """
    # 计算上个月
    year, m = map(int, month.split("-"))
    prev_month = f"{year}-{m-1:02d}" if m > 1 else f"{year-1}-12"
    
    sql = f"""
        SELECT 
            order_month as 月份,
            SUM(quantity) as 总销量,
            ROUND(SUM(total_amount), 2) as 总销售额
        FROM sales
        WHERE order_month IN ('{month}', '{prev_month}')
        GROUP BY order_month
        ORDER BY order_month
    """
    
    try:
        df = execute_sql(sql, engine)
        
        if df.empty:
            return f"未找到 {month} 和 {prev_month} 的销售数据"
        
        # 转换为字典方便查询
        data = df.set_index("月份").to_dict("index")
        
        if month not in data:
            return f"未找到 {month} 的销售数据"
        
        current = data.get(month, {})
        previous = data.get(prev_month, {})
        
        curr_qty = current.get("总销量", 0) or 0
        prev_qty = previous.get("总销量", 0) or 0
        curr_amount = current.get("总销售额", 0) or 0
        prev_amount = previous.get("总销售额", 0) or 0
        
        # 计算环比增长率
        if prev_qty > 0:
            qty_growth = (curr_qty - prev_qty) / prev_qty * 100
        else:
            qty_growth = 0 if curr_qty == 0 else 100
        
        if prev_amount > 0:
            amount_growth = (curr_amount - prev_amount) / prev_amount * 100
        else:
            amount_growth = 0 if curr_amount == 0 else 100
        
        result = f"""
## {month} 销售环比分析

| 指标 | {prev_month} | {month} | 环比增长 |
|------|-------------|---------|---------|
| 总销量 | {prev_qty:,} | {curr_qty:,} | {qty_growth:+.2f}% |
| 总销售额 | ¥{prev_amount:,.2f} | ¥{curr_amount:,.2f} | {amount_growth:+.2f}% |
"""
        return result
        
    except Exception as e:
        logger.error(f"计算环比增长失败: {e}")
        return f"查询失败: {str(e)}"


def get_province_sales(
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    engine: Optional[Any] = None,
) -> str:
    """
    查询不同省份的销售额
    
    Args:
        start_month: 开始月份（可选）
        end_month: 结束月份（可选）
        engine: 数据库引擎
        
    Returns:
        各省销售额的 Markdown 表格
    """
    where_clauses = []
    if start_month:
        where_clauses.append(f"order_month >= '{start_month}'")
    if end_month:
        where_clauses.append(f"order_month <= '{end_month}'")
    
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    sql = f"""
        SELECT 
            province as 省份,
            COUNT(*) as 订单数,
            SUM(quantity) as 总销量,
            ROUND(SUM(total_amount), 2) as 总销售额
        FROM sales
        WHERE {where_sql}
        GROUP BY province
        ORDER BY 总销售额 DESC
    """
    
    try:
        df = execute_sql(sql, engine)
        if df.empty:
            return "未找到符合条件的销售数据"
        
        # 计算占比
        total_amount = df["总销售额"].sum()
        df["销售额占比"] = (df["总销售额"] / total_amount * 100).round(2)
        
        return df.to_markdown(index=False)
    except Exception as e:
        logger.error(f"查询省份销售额失败: {e}")
        return f"查询失败: {str(e)}"


def get_top_channels(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    top_n: int = 3,
    engine: Optional[Any] = None,
) -> str:
    """
    查询某时间段销售金额 Top N 的渠道
    
    Args:
        start_date: 开始日期（可选）
        end_date: 结束日期（可选）
        top_n: 返回前几名
        engine: 数据库引擎
        
    Returns:
        Top N 渠道的 Markdown 表格
    """
    where_clauses = []
    if start_date:
        where_clauses.append(f"order_date >= '{start_date}'")
    if end_date:
        where_clauses.append(f"order_date <= '{end_date}'")
    
    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    sql = f"""
        SELECT 
            channel as 渠道,
            COUNT(*) as 订单数,
            SUM(quantity) as 总销量,
            ROUND(SUM(total_amount), 2) as 总销售额
        FROM sales
        WHERE {where_sql}
        GROUP BY channel
        ORDER BY 总销售额 DESC
        LIMIT {top_n}
    """
    
    try:
        df = execute_sql(sql, engine)
        if df.empty:
            return "未找到符合条件的销售数据"
        
        # 计算占比
        total_amount = df["总销售额"].sum()
        df["销售额占比"] = (df["总销售额"] / total_amount * 100).round(2)
        
        result = f"## 销售金额 Top{top_n} 渠道\n\n"
        result += df.to_markdown(index=False)
        return result
    except Exception as e:
        logger.error(f"查询 Top 渠道失败: {e}")
        return f"查询失败: {str(e)}"


def execute_custom_sql(sql: str, engine: Optional[Any] = None) -> str:
    """
    执行自定义 SQL 查询
    
    Args:
        sql: SQL 查询语句
        engine: 数据库引擎
        
    Returns:
        查询结果的 Markdown 表格
    """
    # 安全检查：只允许 SELECT 语句
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "只允许执行 SELECT 查询语句"
    
    # 禁止危险操作
    dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE"]
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return f"禁止执行包含 {keyword} 的语句"
    
    try:
        df = execute_sql(sql, engine)
        if df.empty:
            return "查询结果为空"
        
        # 限制返回行数
        if len(df) > 50:
            df = df.head(50)
            return df.to_markdown(index=False) + "\n\n*注：结果已截断，仅显示前 50 条*"
        
        return df.to_markdown(index=False)
    except Exception as e:
        logger.error(f"执行自定义 SQL 失败: {e}")
        return f"查询失败: {str(e)}"


# 工具函数映射
TOOL_FUNCTIONS: Dict[str, callable] = {
    "get_monthly_sales": get_monthly_sales,
    "get_monthly_sales_growth": get_monthly_sales_growth,
    "get_province_sales": get_province_sales,
    "get_top_channels": get_top_channels,
    "execute_custom_sql": execute_custom_sql,
}


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    调用工具函数
    
    Args:
        tool_name: 工具名称
        arguments: 工具参数
        
    Returns:
        工具执行结果
    """
    if tool_name not in TOOL_FUNCTIONS:
        return f"未知的工具: {tool_name}"
    
    func = TOOL_FUNCTIONS[tool_name]
    
    try:
        return func(**arguments)
    except Exception as e:
        logger.error(f"工具调用失败: {tool_name}, 错误: {e}")
        return f"工具执行失败: {str(e)}"
