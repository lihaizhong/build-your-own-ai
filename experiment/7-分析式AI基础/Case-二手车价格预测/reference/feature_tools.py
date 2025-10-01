#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程工具合集
整合特征相关性分析等特征工程工具
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def feature_correlation_analysis(data, target_col='price', save_plot=True):
    """特征相关性分析"""
    print("=== 特征相关性分析 ===")
    
    # 只分析数值特征
    numeric_data = data.select_dtypes(include=[np.number])
    
    if target_col not in numeric_data.columns:
        print(f"警告: 目标变量 {target_col} 不在数值特征中")
        return None
    
    # 计算相关性矩阵
    correlation_matrix = numeric_data.corr()
    
    # 与目标变量的相关性
    target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
    
    print("与目标变量相关性最高的10个特征:")
    print(target_corr.head(10))
    
    # 可视化相关性矩阵
    if save_plot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig('docs/特征相关性热力图.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("相关性热力图已保存到 docs/特征相关性热力图.png")
    
    # 找出高相关性特征对
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = abs(correlation_matrix.iloc[i, j])
            if corr_val > 0.8:  # 高相关性阈值
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        print(f"\n发现 {len(high_corr_pairs)} 对高相关性特征 (>0.8):")
        for pair in high_corr_pairs:
            print(f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
    
    return {
        'correlation_matrix': correlation_matrix,
        'target_correlation': target_corr,
        'high_corr_pairs': high_corr_pairs
    }

def create_rf_friendly_features(train_data, test_data):
    """创建随机森林友好的特征"""
    print("创建RF友好特征...")
    
    # 复制数据避免修改原始数据
    train_processed = train_data.copy()
    test_processed = test_data.copy()
    
    # 获取数值特征
    numeric_features = []
    for col in train_processed.columns:
        if col != 'price' and col in test_processed.columns:
            if train_processed[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
    
    # 1. 基础交互特征
    if 'kilometer' in numeric_features and 'regDate' in numeric_features:
        # 年份里程比
        train_processed['km_per_year'] = train_processed['kilometer'] / (train_processed['regDate'] + 1)
        test_processed['km_per_year'] = test_processed['kilometer'] / (test_processed['regDate'] + 1)
    
    if 'power' in numeric_features and 'kilometer' in numeric_features:
        # 功率里程比
        train_processed['power_km_ratio'] = train_processed['power'] / (train_processed['kilometer'] + 1)
        test_processed['power_km_ratio'] = test_processed['power'] / (test_processed['kilometer'] + 1)
    
    # 2. 分箱特征
    if 'kilometer' in numeric_features:
        km_bins = [0, 1, 5, 10, 15, 20, float('inf')]
        train_processed['km_bin'] = pd.cut(train_processed['kilometer'], bins=km_bins, labels=False)
        test_processed['km_bin'] = pd.cut(test_processed['kilometer'], bins=km_bins, labels=False)
    
    if 'power' in numeric_features:
        power_bins = [0, 80, 120, 160, 200, float('inf')]
        train_processed['power_bin'] = pd.cut(train_processed['power'], bins=power_bins, labels=False)
        test_processed['power_bin'] = pd.cut(test_processed['power'], bins=power_bins, labels=False)
    
    # 3. 品牌统计特征（基于训练集）
    if 'brand' in train_processed.columns and 'price' in train_processed.columns:
        brand_stats = train_processed.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
        brand_stats.columns = ['brand', 'brand_mean_price', 'brand_count']
        
        # 合并到训练集和测试集
        train_processed = train_processed.merge(brand_stats, on='brand', how='left')
        test_processed = test_processed.merge(brand_stats, on='brand', how='left')
        
        # 填充缺失值
        overall_mean = train_processed['price'].mean()
        train_processed['brand_mean_price'] = train_processed['brand_mean_price'].fillna(overall_mean)
        test_processed['brand_mean_price'] = test_processed['brand_mean_price'].fillna(overall_mean)
        
        train_processed['brand_count'] = train_processed['brand_count'].fillna(1)
        test_processed['brand_count'] = test_processed['brand_count'].fillna(1)
    
    print(f"特征工程前训练集: {train_data.shape}")
    print(f"特征工程后训练集: {train_processed.shape}")
    print(f"特征工程前测试集: {test_data.shape}")
    print(f"特征工程后测试集: {test_processed.shape}")
    
    return train_processed, test_processed

def encode_categorical_features(train_data, test_data, encoding_strategy=None):
    """编码分类特征"""
    if encoding_strategy is None:
        # 默认编码策略
        encoding_strategy = {
            'brand': 'label',
            'model': 'label', 
            'bodyType': 'label',
            'fuelType': 'onehot',
            'gearbox': 'label',
            'notRepairedDamage': 'onehot'
        }
    
    train_encoded = train_data.copy()
    test_encoded = test_data.copy()
    
    for col, method in encoding_strategy.items():
        if col in train_encoded.columns and col in test_encoded.columns:
            if method == 'label':
                # 标签编码
                all_values = pd.concat([
                    train_encoded[col].astype(str), 
                    test_encoded[col].astype(str)
                ]).unique()
                
                le = LabelEncoder()
                le.fit(all_values)
                
                train_encoded[col] = le.transform(train_encoded[col].astype(str))
                test_encoded[col] = le.transform(test_encoded[col].astype(str))
                
            elif method == 'onehot':
                # One-hot编码
                train_dummies = pd.get_dummies(train_encoded[col], prefix=col)
                test_dummies = pd.get_dummies(test_encoded[col], prefix=col)
                
                # 确保训练集和测试集有相同的虚拟变量列
                all_cols = set(train_dummies.columns) | set(test_dummies.columns)
                
                for dummy_col in all_cols:
                    if dummy_col not in train_dummies.columns:
                        train_dummies[dummy_col] = 0
                    if dummy_col not in test_dummies.columns:
                        test_dummies[dummy_col] = 0
                
                # 删除原列并添加虚拟变量
                train_encoded = train_encoded.drop(col, axis=1)
                test_encoded = test_encoded.drop(col, axis=1)
                
                train_encoded = pd.concat([train_encoded, train_dummies], axis=1)
                test_encoded = pd.concat([test_encoded, test_dummies], axis=1)
    
    return train_encoded, test_encoded

def select_important_features(X_train, y_train, model, top_k=None):
    """基于模型选择重要特征"""
    # 训练模型获取特征重要性
    model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("特征重要性排序:")
    print(feature_importance.head(20))
    
    if top_k:
        selected_features = feature_importance.head(top_k)['feature'].tolist()
        print(f"选择前 {top_k} 个重要特征")
        return selected_features
    
    return feature_importance

if __name__ == "__main__":
    print("特征工程工具合集已加载")