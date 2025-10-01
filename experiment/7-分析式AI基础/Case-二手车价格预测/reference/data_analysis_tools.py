#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析工具合集
整合各种数据分析和特征分析工具
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_categorical_features(data, target_col='price'):
    """分析分类特征与目标变量的关系"""
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    results = {}
    for col in categorical_cols:
        if col != target_col:
            # 计算每个类别的统计信息
            stats = data.groupby(col)[target_col].agg(['count', 'mean', 'std']).round(2)
            results[col] = stats
            
            print(f"\n{col} 特征分析:")
            print(stats.head(10))
    
    return results

def check_data_features(train_data, test_data):
    """检查训练集和测试集的特征一致性"""
    print("训练集特征:", train_data.columns.tolist())
    print("测试集特征:", test_data.columns.tolist())
    
    train_features = set(train_data.columns)
    test_features = set(test_data.columns)
    
    common_features = train_features & test_features
    only_train = train_features - test_features
    only_test = test_features - train_features
    
    print(f"\n共同特征数: {len(common_features)}")
    print(f"仅训练集特征数: {len(only_train)}")
    print(f"仅测试集特征数: {len(only_test)}")
    
    if only_train:
        print("仅训练集特征:", list(only_train))
    if only_test:
        print("仅测试集特征:", list(only_test))
    
    return common_features, only_train, only_test

def read_data_columns(file_path):
    """读取数据文件的列信息"""
    try:
        data = pd.read_csv(file_path, nrows=5)
        print(f"文件: {file_path}")
        print(f"形状: {data.shape}")
        print("列名:", data.columns.tolist())
        print("数据类型:")
        print(data.dtypes)
        return data.columns.tolist()
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def validate_preprocessed_data(train_data, test_data):
    """验证预处理后的数据质量"""
    print("=== 数据质量验证 ===")
    
    # 检查缺失值
    print("\n训练集缺失值:")
    print(train_data.isnull().sum().sum())
    
    print("\n测试集缺失值:")
    print(test_data.isnull().sum().sum())
    
    # 检查数据类型
    print("\n数据类型一致性:")
    common_cols = set(train_data.columns) & set(test_data.columns)
    for col in common_cols:
        if train_data[col].dtype != test_data[col].dtype:
            print(f"警告: {col} 类型不一致")
    
    # 检查数值范围
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in test_data.columns:
            train_range = (train_data[col].min(), train_data[col].max())
            test_range = (test_data[col].min(), test_data[col].max())
            print(f"{col}: 训练集{train_range}, 测试集{test_range}")

if __name__ == "__main__":
    print("数据分析工具合集已加载")