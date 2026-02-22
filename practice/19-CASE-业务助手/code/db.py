"""
数据库连接和销售数据模型

支持 MySQL 数据库连接，提供销售数据的查询功能
"""
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# 加载环境变量
load_dotenv()


def get_engine() -> Engine:
    """
    获取数据库连接引擎
    
    优先使用 DATABASE_URL 环境变量，否则使用单独的配置项
    """
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        logger.info("使用 DATABASE_URL 连接数据库")
        # SQLite 不支持 connect_timeout 和 pool 参数
        if database_url.startswith("sqlite"):
            return create_engine(database_url)
        return create_engine(
            database_url,
            connect_args={"connect_timeout": 10},
            pool_size=10,
            max_overflow=20,
        )
    
    # 单独配置项（用于本地开发）
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "sales_db")
    
    connection_url = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
    logger.info(f"连接数据库: {db_host}:{db_port}/{db_name}")
    
    return create_engine(
        connection_url,
        connect_args={"connect_timeout": 10},
        pool_size=10,
        max_overflow=20,
    )


def execute_sql(sql: str, engine: Optional[Engine] = None) -> pd.DataFrame:
    """
    执行 SQL 查询并返回 DataFrame
    
    Args:
        sql: SQL 查询语句
        engine: 数据库引擎（可选）
        
    Returns:
        查询结果的 DataFrame
    """
    if engine is None:
        engine = get_engine()
    
    try:
        df = pd.read_sql(sql, engine)
        logger.info(f"SQL 执行成功，返回 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"SQL 执行失败: {e}")
        raise


def test_connection() -> bool:
    """测试数据库连接"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("数据库连接测试成功")
        return True
    except Exception as e:
        logger.error(f"数据库连接测试失败: {e}")
        return False


# 销售数据表的建表语句
SALES_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS sales (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_date DATE NOT NULL COMMENT '订单日期',
    order_month VARCHAR(7) NOT NULL COMMENT '订单月份，格式 YYYY-MM',
    province VARCHAR(50) NOT NULL COMMENT '省份',
    city VARCHAR(50) COMMENT '城市',
    channel VARCHAR(100) NOT NULL COMMENT '销售渠道',
    product_name VARCHAR(200) COMMENT '产品名称',
    quantity INT NOT NULL DEFAULT 0 COMMENT '销售数量',
    unit_price DECIMAL(10, 2) NOT NULL DEFAULT 0 COMMENT '单价',
    total_amount DECIMAL(12, 2) NOT NULL DEFAULT 0 COMMENT '销售金额',
    customer_id VARCHAR(50) COMMENT '客户ID',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_order_date (order_date),
    INDEX idx_order_month (order_month),
    INDEX idx_province (province),
    INDEX idx_channel (channel)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='销售数据表';
"""
