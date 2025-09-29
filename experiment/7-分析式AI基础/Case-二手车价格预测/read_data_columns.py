#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - 数据列信息查看脚本
读取训练数据中的 used_car_train_20200313.csv 文件，显示全部列信息
"""

import pandas as pd
import os

def read_and_show_columns():
    """
    读取训练数据文件并显示列信息
    """
    # 数据文件路径
    data_path = "训练数据/used_car_train_20200313.csv"
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误：文件 {data_path} 不存在！")
        return
    
    try:
        # 读取CSV文件（使用空格作为分隔符）
        print("正在读取数据文件...")
        df = pd.read_csv(data_path, sep=' ')
        
        # 显示数据基本信息
        print("\n" + "="*60)
        print("数据文件基本信息")
        print("="*60)
        print(f"文件路径: {data_path}")
        print(f"数据形状: {df.shape}")
        print(f"总行数: {df.shape[0]:,}")
        print(f"总列数: {df.shape[1]}")
        
        # 显示所有列名
        print("\n" + "="*60)
        print("所有列名信息")
        print("="*60)
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        # 显示每列的数据类型和基本统计信息
        print("\n" + "="*60)
        print("列数据类型信息")
        print("="*60)
        print(df.dtypes)
        
        # 显示前几行数据预览
        print("\n" + "="*60)
        print("数据预览（前5行）")
        print("="*60)
        print(df.head())
        
        # 显示数据信息摘要
        print("\n" + "="*60)
        print("数据信息摘要")
        print("="*60)
        print(df.info())
        
        # 显示缺失值统计
        print("\n" + "="*60)
        print("缺失值统计")
        print("="*60)
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        missing_summary = pd.DataFrame({
            '缺失值数量': missing_values,
            '缺失值比例(%)': missing_percent.round(2)
        })
        # 只显示有缺失值的列
        missing_summary = missing_summary[missing_summary['缺失值数量'] > 0]
        if not missing_summary.empty:
            print(missing_summary)
        else:
            print("所有列都没有缺失值")
        
        # 显示数值型特征的描述性统计
        print("\n" + "="*60)
        print("数值型特征描述性统计")
        print("="*60)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        print(df[numeric_cols].describe())
        
        # 显示分类特征的唯一值统计
        print("\n" + "="*60)
        print("分类特征唯一值统计")
        print("="*60)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_values = df[col].nunique()
            print(f"{col}: {unique_values} 个唯一值")
            if unique_values < 20:  # 如果唯一值较少，显示所有唯一值
                print(f"  唯一值: {df[col].unique()}")
            else:
                print(f"  前10个唯一值: {df[col].unique()[:10]}")
        
        print("\n数据列信息读取完成！")
        
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")

if __name__ == "__main__":
    read_and_show_columns()