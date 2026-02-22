"""
示例数据初始化脚本

创建销售数据表并插入示例数据
支持 SQLite（默认）和 MySQL
"""
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

# 加载环境变量
load_dotenv()

# 示例数据配置
PROVINCES = ["北京", "上海", "广东", "浙江", "江苏", "山东", "四川", "湖北", "河南", "福建"]
CITIES = {
    "北京": ["北京"],
    "上海": ["上海"],
    "广东": ["广州", "深圳", "东莞", "佛山"],
    "浙江": ["杭州", "宁波", "温州", "嘉兴"],
    "江苏": ["南京", "苏州", "无锡", "常州"],
    "山东": ["济南", "青岛", "烟台", "潍坊"],
    "四川": ["成都", "绵阳", "德阳"],
    "湖北": ["武汉", "宜昌", "襄阳"],
    "河南": ["郑州", "洛阳", "开封"],
    "福建": ["福州", "厦门", "泉州"],
}
CHANNELS = ["线上商城", "天猫旗舰店", "京东旗舰店", "抖音直播", "微信小程序", "线下门店", "代理商", "批发商"]
PRODUCTS = [
    ("产品A-标准版", 199),
    ("产品A-高级版", 399),
    ("产品B-基础版", 99),
    ("产品B-专业版", 299),
    ("产品C-入门版", 49),
    ("产品C-完整版", 599),
]


def generate_sample_data(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    num_records: int = 2000,
) -> pd.DataFrame:
    """
    生成示例销售数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        num_records: 记录数量
        
    Returns:
        销售数据 DataFrame
    """
    logger.info(f"生成示例数据: {start_date} 到 {end_date}, {num_records} 条记录")
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = (end - start).days
    
    records = []
    
    for _ in range(num_records):
        # 随机日期
        random_days = random.randint(0, date_range)
        order_date = start + timedelta(days=random_days)
        order_month = order_date.strftime("%Y-%m")
        
        # 随机省份和城市
        province = random.choice(PROVINCES)
        city = random.choice(CITIES[province])
        
        # 随机渠道
        channel = random.choice(CHANNELS)
        
        # 随机产品
        product_name, unit_price = random.choice(PRODUCTS)
        
        # 随机数量（1-50）
        quantity = random.randint(1, 50)
        
        # 计算总金额（加一些随机折扣）
        discount = random.uniform(0.9, 1.0)
        total_amount = round(unit_price * quantity * discount, 2)
        
        # 客户ID
        customer_id = f"C{random.randint(10000, 99999)}"
        
        records.append({
            "order_date": order_date.strftime("%Y-%m-%d"),
            "order_month": order_month,
            "province": province,
            "city": city,
            "channel": channel,
            "product_name": product_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "total_amount": total_amount,
            "customer_id": customer_id,
        })
    
    df = pd.DataFrame(records)
    logger.info(f"生成完成，共 {len(df)} 条记录")
    
    return df


def init_sqlite_db(db_path: str, df: pd.DataFrame):
    """
    初始化 SQLite 数据库
    
    Args:
        db_path: 数据库文件路径
        df: 销售数据 DataFrame
    """
    engine = create_engine(f"sqlite:///{db_path}")
    
    # 建表
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_date TEXT NOT NULL,
                order_month TEXT NOT NULL,
                province TEXT NOT NULL,
                city TEXT,
                channel TEXT NOT NULL,
                product_name TEXT,
                quantity INTEGER NOT NULL DEFAULT 0,
                unit_price REAL NOT NULL DEFAULT 0,
                total_amount REAL NOT NULL DEFAULT 0,
                customer_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
    
    # 插入数据
    df.to_sql("sales", engine, if_exists="append", index=False)
    logger.info(f"SQLite 数据库初始化完成: {db_path}")


def init_mysql_db(df: pd.DataFrame):
    """
    初始化 MySQL 数据库
    
    Args:
        df: 销售数据 DataFrame
    """
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "3306")
        db_user = os.getenv("DB_USER", "root")
        db_password = os.getenv("DB_PASSWORD", "")
        db_name = os.getenv("DB_NAME", "sales_db")
        database_url = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
    
    engine = create_engine(database_url)
    
    # 建表
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sales (
                id INT AUTO_INCREMENT PRIMARY KEY,
                order_date DATE NOT NULL COMMENT '订单日期',
                order_month VARCHAR(7) NOT NULL COMMENT '订单月份',
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
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='销售数据表'
        """))
        conn.commit()
    
    # 插入数据
    df.to_sql("sales", engine, if_exists="append", index=False)
    logger.info("MySQL 数据库初始化完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="初始化示例数据")
    parser.add_argument(
        "--db-type",
        choices=["sqlite", "mysql"],
        default="sqlite",
        help="数据库类型",
    )
    parser.add_argument(
        "--db-path",
        default="data/sales.db",
        help="SQLite 数据库路径（仅 sqlite 模式）",
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="数据开始日期",
    )
    parser.add_argument(
        "--end-date",
        default="2024-12-31",
        help="数据结束日期",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=2000,
        help="生成记录数量",
    )
    
    args = parser.parse_args()
    
    # 生成示例数据
    df = generate_sample_data(
        start_date=args.start_date,
        end_date=args.end_date,
        num_records=args.num_records,
    )
    
    # 保存 CSV 副本
    csv_path = Path(__file__).parent.parent / "data" / "sales_sample.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"CSV 文件已保存: {csv_path}")
    
    # 初始化数据库
    if args.db_type == "sqlite":
        db_path = Path(__file__).parent.parent / args.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        init_sqlite_db(str(db_path), df)
        
        print(f"\n数据库文件: {db_path}")
        print("设置环境变量:")
        print(f'  export DATABASE_URL="sqlite:///{db_path}"')
    else:
        init_mysql_db(df)
        print("\nMySQL 数据库初始化完成")


if __name__ == "__main__":
    main()
