#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析分类特征的基数，为编码方法选择提供依据
"""

import pandas as pd
import numpy as np

def analyze_categorical_features():
    """
    分析分类特征的基数和分布
    """
    # 读取训练数据
    try:
        df = pd.read_csv("训练数据/used_car_train_20200313.csv", sep=' ')
        print("数据读取成功！")
        print(f"数据形状: {df.shape}")
        
        # 定义分类特征
        categorical_features = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
        
        print("\n" + "="*80)
        print("分类特征基数分析")
        print("="*80)
        
        for feature in categorical_features:
            if feature in df.columns:
                # 计算唯一值数量（排除缺失值）
                non_null_values = df[feature].dropna()
                unique_count = non_null_values.nunique()
                total_count = len(df)
                null_count = df[feature].isnull().sum()
                null_percentage = (null_count / total_count) * 100
                
                print(f"\n【{feature}】")
                print(f"  唯一值数量: {unique_count}")
                print(f"  缺失值数量: {null_count} ({null_percentage:.2f}%)")
                print(f"  数据类型: {df[feature].dtype}")
                
                # 显示频次分布（前10个）
                value_counts = non_null_values.value_counts().head(10)
                print(f"  前10个高频值:")
                for idx, (value, count) in enumerate(value_counts.items(), 1):
                    percentage = (count / len(non_null_values)) * 100
                    print(f"    {idx}. {value}: {count} ({percentage:.2f}%)")
                
                # 基于基数给出编码建议
                if unique_count <= 10:
                    encoding_suggestion = "One-Hot编码 或 标签编码"
                    reason = "低基数特征，One-Hot不会产生过多维度"
                elif unique_count <= 50:
                    encoding_suggestion = "标签编码"
                    reason = "中等基数特征，标签编码更合适"
                else:
                    encoding_suggestion = "标签编码 或 频次编码"
                    reason = "高基数特征，需要降维处理"
                
                print(f"  编码建议: {encoding_suggestion}")
                print(f"  建议原因: {reason}")
                
        # 分析 notRepairedDamage 特殊情况
        if 'notRepairedDamage' in df.columns:
            print(f"\n【notRepairedDamage 特殊分析】")
            print(f"  数据类型: {df['notRepairedDamage'].dtype}")
            print(f"  唯一值: {df['notRepairedDamage'].unique()}")
            value_counts = df['notRepairedDamage'].value_counts()
            print(f"  值分布:")
            for value, count in value_counts.items():
                percentage = (count / len(df)) * 100
                print(f"    '{value}': {count} ({percentage:.2f}%)")
            
    except Exception as e:
        print(f"分析过程中出错: {str(e)}")

if __name__ == "__main__":
    analyze_categorical_features()