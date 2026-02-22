"""
测试业务助手工具函数
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置数据库路径
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/data/sales.db"

from tools import (
    get_monthly_sales,
    get_monthly_sales_growth,
    get_province_sales,
    get_top_channels,
)


def test_tools():
    """测试所有工具函数"""
    print("=" * 60)
    print("测试业务助手工具函数")
    print("=" * 60)
    
    # 1. 测试月度销量查询
    print("\n1. 查询 2024年1月 的销量:")
    print("-" * 40)
    result = get_monthly_sales("2024-01")
    print(result)
    
    # 2. 测试销量环比增长
    print("\n2. 查询 2024年2月 销量环比增长:")
    print("-" * 40)
    result = get_monthly_sales_growth("2024-02")
    print(result)
    
    # 3. 测试省份销售额
    print("\n3. 查询各省销售额:")
    print("-" * 40)
    result = get_province_sales()
    print(result)
    
    # 4. 测试 Top 渠道
    print("\n4. 查询 2024年第一季度 销售金额 Top3 渠道:")
    print("-" * 40)
    result = get_top_channels(start_date="2024-01-01", end_date="2024-03-31", top_n=3)
    print(result)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_tools()