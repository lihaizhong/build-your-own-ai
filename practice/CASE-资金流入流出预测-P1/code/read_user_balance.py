#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from get_project_path import get_project_path


def read_user_balance_data():
    """
    读取user_balance_table.csv文件的前5行数据，显示全部列
    """
    # 数据文件路径
    data_file = get_project_path('..', 'data', 'user_balance_table.csv')
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"错误：文件 {data_file} 不存在")
        return
    
    try:
        # 读取CSV文件的前5行数据
        df = pd.read_csv(data_file, nrows=5)
        
        print("=== user_balance_table.csv 前5行数据 ===")
        print(f"数据形状: {df.shape}")
        print(f"列数: {len(df.columns)}")
        print("\n=== 全部列名 ===")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\n=== 前5行数据 ===")
        print(df.to_string())
        
        print("\n=== 数据类型 ===")
        print(df.dtypes)
        
        return df
        
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None

if __name__ == "__main__":
    read_user_balance_data()
