#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车数据探索性数据分析（EDA）
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载数据"""
    data_path = 'experiment/7-分析式AI基础/Case-二手车价格预测/used_car_train_20200313.csv'
    df = pd.read_csv(data_path, sep=' ')
    print("数据加载完成！")
    print(f"数据形状: {df.shape}")
    return df

def basic_info_analysis(df):
    """基本信息分析"""
    print("\n" + "="*50)
    print("1. 基本信息分析")
    print("="*50)
    
    print(f"数据集大小: {df.shape}")
    print(f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n数据类型分布:")
    print(df.dtypes.value_counts())
    
    print("\n数据基本统计信息:")
    print(df.describe())
    
    return df

def missing_value_analysis(df):
    """缺失值分析"""
    print("\n" + "="*50)
    print("2. 缺失值分析")
    print("="*50)
    
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100
    
    missing_df = pd.DataFrame({
        '缺失数量': missing_counts,
        '缺失百分比': missing_percent
    })
    missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
    
    if len(missing_df) > 0:
        print("存在缺失值的字段:")
        print(missing_df)
    else:
        print("未发现标准缺失值（NaN）")
    
    # 检查特殊值'-'
    print("\n检查notRepairedDamage字段的特殊值:")
    print(df['notRepairedDamage'].value_counts())
    
    return missing_df

def target_analysis(df):
    """目标变量分析"""
    print("\n" + "="*50)
    print("3. 目标变量（price）分析")
    print("="*50)
    
    price = df['price']
    
    print("价格基本统计:")
    print(f"均值: {price.mean():.2f}")
    print(f"中位数: {price.median():.2f}")
    print(f"标准差: {price.std():.2f}")
    print(f"最小值: {price.min():.2f}")
    print(f"最大值: {price.max():.2f}")
    print(f"偏度: {price.skew():.2f}")
    print(f"峰度: {price.kurtosis():.2f}")
    
    # 价格分布区间统计
    print("\n价格分布区间:")
    bins = [0, 5000, 10000, 20000, 50000, float('inf')]
    labels = ['0-5k', '5k-10k', '10k-20k', '20k-50k', '50k+']
    price_ranges = pd.cut(price, bins=bins, labels=labels)
    print(price_ranges.value_counts())
    
    # 检查异常值
    Q1 = price.quantile(0.25)
    Q3 = price.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = price[(price < lower_bound) | (price > upper_bound)]
    print(f"\n异常值数量: {len(outliers)} ({len(outliers)/len(price)*100:.2f}%)")
    print(f"异常值范围: 小于{lower_bound:.2f}或大于{upper_bound:.2f}")
    
    return price

def categorical_analysis(df):
    """分类变量分析"""
    print("\n" + "="*50)
    print("4. 分类变量分析")
    print("="*50)
    
    categorical_cols = ['brand', 'bodyType', 'fuelType', 'gearbox', 'seller', 'offerType']
    
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col}字段分析:")
            value_counts = df[col].value_counts()
            print(f"唯一值数量: {df[col].nunique()}")
            print(f"前5个最频繁值:")
            print(value_counts.head())
            
            # 计算价格均值
            if 'price' in df.columns:
                avg_price = df.groupby(col)['price'].mean().sort_values(ascending=False)
                print(f"各类别平均价格（前5）:")
                print(avg_price.head())

def numerical_analysis(df):
    """数值变量分析"""
    print("\n" + "="*50)
    print("5. 数值变量分析")
    print("="*50)
    
    numerical_cols = ['power', 'kilometer', 'model']
    
    for col in numerical_cols:
        if col in df.columns:
            print(f"\n{col}字段分析:")
            series = df[col].dropna()
            print(f"均值: {series.mean():.2f}")
            print(f"中位数: {series.median():.2f}")
            print(f"标准差: {series.std():.2f}")
            print(f"最小值: {series.min()}")
            print(f"最大值: {series.max()}")
            
            # 检查异常值
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
            print(f"异常值数量: {len(outliers)} ({len(outliers)/len(series)*100:.2f}%)")

def anonymous_features_analysis(df):
    """匿名特征分析"""
    print("\n" + "="*50)
    print("6. 匿名特征（v_0到v_14）分析")
    print("="*50)
    
    v_cols = [col for col in df.columns if col.startswith('v_')]
    
    print(f"匿名特征数量: {len(v_cols)}")
    
    v_df = df[v_cols]
    print("\n匿名特征基本统计:")
    print(v_df.describe())
    
    # 计算与价格的相关性
    if 'price' in df.columns:
        correlations = df[v_cols + ['price']].corr()['price'].drop('price').sort_values(key=abs, ascending=False)
        print("\n与价格相关性最强的匿名特征（前10）:")
        print(correlations.head(10))

def correlation_analysis(df):
    """相关性分析"""
    print("\n" + "="*50)
    print("7. 特征相关性分析")
    print("="*50)
    
    # 选择数值型特征进行相关性分析
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if 'price' in numeric_cols:
        correlations = df[numeric_cols].corr()['price'].drop('price').sort_values(key=abs, ascending=False)
        print("与价格相关性最强的特征（前15）:")
        print(correlations.head(15))
        
        print("\n与价格相关性最弱的特征（后10）:")
        print(correlations.tail(10))

def data_quality_check(df):
    """数据质量检查"""
    print("\n" + "="*50)
    print("8. 数据质量检查")
    print("="*50)
    
    # 检查重复值
    duplicates = df.duplicated().sum()
    print(f"重复行数: {duplicates}")
    
    # 检查ID唯一性
    if 'SaleID' in df.columns:
        unique_ids = df['SaleID'].nunique()
        total_rows = len(df)
        print(f"SaleID唯一性: {unique_ids}/{total_rows} = {unique_ids/total_rows*100:.2f}%")
    
    # 检查日期字段合理性
    if 'regDate' in df.columns:
        reg_dates = df['regDate'].dropna()
        print(f"注册日期范围: {reg_dates.min()} - {reg_dates.max()}")
        
        # 转换为年份进行分析
        reg_years = reg_dates // 10000
        print(f"注册年份范围: {reg_years.min()} - {reg_years.max()}")
    
    if 'creatDate' in df.columns:
        create_dates = df['creatDate'].dropna()
        print(f"创建日期范围: {create_dates.min()} - {create_dates.max()}")
    
    # 检查逻辑一致性
    if 'regDate' in df.columns and 'creatDate' in df.columns:
        # 理论上创建日期应该晚于注册日期
        inconsistent = df[df['creatDate'] < df['regDate']]
        print(f"创建日期早于注册日期的记录数: {len(inconsistent)}")

def run_eda():
    """运行完整EDA分析"""
    print("开始二手车数据探索性数据分析...")
    
    # 加载数据
    df = load_data()
    
    # 执行各项分析
    basic_info_analysis(df)
    missing_value_analysis(df)
    target_analysis(df)
    categorical_analysis(df)
    numerical_analysis(df)
    anonymous_features_analysis(df)
    correlation_analysis(df)
    data_quality_check(df)
    
    print("\n" + "="*50)
    print("EDA分析完成！")
    print("="*50)
    
    return df

if __name__ == "__main__":
    df = run_eda()