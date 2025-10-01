#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证预处理数据的质量和完整性
"""

import pandas as pd
import numpy as np

def validate_preprocessed_data():
    """验证预处理后的数据"""
    print("验证预处理后的数据质量")
    print("="*50)
    
    try:
        # 读取预处理后的数据
        train_df = pd.read_csv("临时数据/used_car_train_preprocess.csv")
        test_df = pd.read_csv("临时数据/used_car_testB_preprocess.csv")
        
        print(f"训练集形状: {train_df.shape}")
        print(f"测试集形状: {test_df.shape}")
        
        # 检查缺失值
        print(f"\n训练集缺失值: {train_df.isnull().sum().sum()}")
        print(f"测试集缺失值: {test_df.isnull().sum().sum()}")
        
        # 检查新增特征
        print("\n新增的时间特征:")
        time_features = ['car_age', 'reg_year', 'reg_month', 'reg_season']
        for feature in time_features:
            if feature in train_df.columns:
                print(f"✅ {feature}: 范围 {train_df[feature].min()}-{train_df[feature].max()}")
        
        # 检查编码特征
        print("\n编码特征:")
        encoded_features = ['brand_encoded', 'model_encoded', 'bodyType_encoded']
        for feature in encoded_features:
            if feature in train_df.columns:
                print(f"✅ {feature}: {train_df[feature].nunique()} 个唯一值")
        
        # 检查目标变量变换
        if 'price_log' in train_df.columns:
            print(f"\n目标变量变换:")
            print(f"✅ price_log: 范围 {train_df['price_log'].min():.2f}-{train_df['price_log'].max():.2f}")
            
        # 检查删除的特征
        removed_features = ['SaleID', 'name', 'offerType', 'seller', 'v_1', 'v_7', 'v_4', 'v_8', 'v_2', 'v_12']
        print(f"\n已删除的特征检查:")
        for feature in removed_features:
            if feature not in train_df.columns:
                print(f"✅ {feature}: 已删除")
            else:
                print(f"❌ {feature}: 未删除")
        
        print(f"\n✅ 数据验证完成！")
        
    except Exception as e:
        print(f"❌ 验证过程中出错: {str(e)}")

if __name__ == "__main__":
    validate_preprocessed_data()