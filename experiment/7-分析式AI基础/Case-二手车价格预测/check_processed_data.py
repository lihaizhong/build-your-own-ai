#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查预处理后的数据
"""

import pandas as pd
import numpy as np

def check_processed_data():
    """检查预处理后的数据质量"""
    print("="*50)
    print("检查预处理后的数据质量")
    print("="*50)
    
    # 读取处理后的训练集
    train_df = pd.read_csv('processed_train_data.csv')
    print(f"训练集形状: {train_df.shape}")
    print(f"训练集列名: {list(train_df.columns)}")
    
    # 读取处理后的测试集
    test_df = pd.read_csv('processed_test_data.csv')
    print(f"测试集形状: {test_df.shape}")
    print(f"测试集列名: {list(test_df.columns)}")
    
    print("\n训练集前5行:")
    print(train_df.head())
    
    print("\n测试集前5行:")
    print(test_df.head())
    
    print("\n训练集缺失值检查:")
    missing_train = train_df.isnull().sum()
    print(missing_train[missing_train > 0])
    if missing_train.sum() == 0:
        print("无缺失值")
    
    print("\n测试集缺失值检查:")
    missing_test = test_df.isnull().sum()
    print(missing_test[missing_test > 0])
    if missing_test.sum() == 0:
        print("无缺失值")
    
    print("\n训练集数据类型:")
    print(train_df.dtypes)
    
    print("\n价格统计信息:")
    if 'price' in train_df.columns:
        print(train_df['price'].describe())
    
    print("\npower字段检查:")
    if 'power' in train_df.columns:
        print(f"训练集power最大值: {train_df['power'].max()}")
        print(f"训练集power超过600的数量: {(train_df['power'] > 600).sum()}")
    
    if 'power' in test_df.columns:
        print(f"测试集power最大值: {test_df['power'].max()}")
        print(f"测试集power超过600的数量: {(test_df['power'] > 600).sum()}")
    
    print("\nnotRepairedDamage字段检查:")
    if 'notRepairedDamage' in train_df.columns:
        print("训练集notRepairedDamage值分布:")
        print(train_df['notRepairedDamage'].value_counts())
    
    if 'notRepairedDamage' in test_df.columns:
        print("测试集notRepairedDamage值分布:")
        print(test_df['notRepairedDamage'].value_counts().head())

if __name__ == "__main__":
    check_processed_data()