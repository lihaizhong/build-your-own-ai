#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车数据读取脚本
读取训练数据集的前5行，显示所有列
"""

import pandas as pd
import numpy as np

def read_car_data():
    """读取二手车训练数据的前5行"""
    # 读取CSV文件
    data_path = 'experiment/7-分析式AI基础/Case-二手车价格预测/used_car_train_20200313.csv'
    
    try:
        # 读取数据，设置显示选项以显示所有列
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        # 读取前5行数据，使用空格作为分隔符
        df = pd.read_csv(data_path, nrows=5, sep=' ')
        
        print("=== 二手车训练数据前5行 ===")
        print(f"数据形状: {df.shape}")
        print("\n数据预览:")
        print(df)
        
        print("\n=== 列信息 ===")
        print(f"总列数: {len(df.columns)}")
        print("列名列表:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\n=== 数据类型 ===")
        print(df.dtypes)
        
        return df
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {data_path}")
        return None
    except Exception as e:
        print(f"读取数据时发生错误: {e}")
        return None

if __name__ == "__main__":
    df = read_car_data()