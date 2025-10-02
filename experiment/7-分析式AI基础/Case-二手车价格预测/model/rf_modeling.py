#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专注随机森林优化脚本
解决训练验证与考试结果差异问题，专注探索随机森林能力
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

def get_project_path(*paths):
    """
    获取项目路径的统一方法
    Args:
        *paths: 相对于项目根目录的路径组件
    Returns:
        str: 绝对路径
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录
    project_root = os.path.dirname(current_dir)
    # 构建目标路径
    return os.path.join(project_root, *paths)

def load_and_preprocess_data():
    """加载并预处理数据"""
    print("正在加载数据...")
    
    # 加载原始数据
    train_raw = pd.read_csv(get_project_path('data', 'used_car_train_20200313.csv'), sep=' ')
    test_raw = pd.read_csv(get_project_path('data', 'used_car_testB_20200421.csv'), sep=' ')
    
    print(f"原始训练集: {train_raw.shape}")
    print(f"原始测试集: {test_raw.shape}")
    
    # 确保特征完全一致
    common_features = set(train_raw.columns) & set(test_raw.columns)
    feature_cols = [col for col in common_features if col != 'price']
    
    train_data = train_raw[feature_cols + ['price']].copy()
    test_data = test_raw[feature_cols].copy()
    
    # 处理缺失值
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    categorical_cols = train_data.select_dtypes(exclude=[np.number]).columns
    
    # 数值型特征用中位数填充
    for col in numeric_cols:
        if col != 'price':
            median_val = train_data[col].median()
            train_data[col] = train_data[col].fillna(median_val)
            if col in test_data.columns:
                test_data[col] = test_data[col].fillna(median_val)
    
    # 分类特征用众数填充
    for col in categorical_cols:
        if col != 'price':
            mode_val = 'unknown'
            if len(train_data[col].mode()) > 0:
                mode_val = train_data[col].mode().iloc[0]
            train_data[col] = train_data[col].fillna(mode_val)
            if col in test_data.columns:
                test_data[col] = test_data[col].fillna(mode_val)
    
    # 分类特征编码
    categorical_features = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    
    for col in categorical_features:
        if col in train_data.columns and col in test_data.columns:
            # 合并训练和测试集进行统一编码
            all_values = pd.concat([
                train_data[col].astype(str), 
                test_data[col].astype(str)
            ]).unique()
            
            le = LabelEncoder()
            le.fit(all_values)
            
            train_data[col] = le.transform(train_data[col].astype(str))
            test_data[col] = le.transform(test_data[col].astype(str))
    
    # 处理价格异常值
    if 'price' in train_data.columns:
        price_q01 = train_data['price'].quantile(0.01)
        price_q99 = train_data['price'].quantile(0.99)
        
        valid_idx = (train_data['price'] >= price_q01) & (train_data['price'] <= price_q99)
        train_data = train_data[valid_idx].reset_index(drop=True)
        
        print(f"价格范围: {train_data['price'].min():.2f} - {train_data['price'].max():.2f}")
    
    print(f"预处理后训练集: {train_data.shape}")
    print(f"预处理后测试集: {test_data.shape}")
    
    return train_data, test_data

def create_rf_features(train_data, test_data):
    """创建随机森林友好的特征"""
    print("创建特征工程...")
    
    # 获取数值特征
    numeric_features = []
    for col in train_data.columns:
        if col != 'price' and col in test_data.columns:
            if train_data[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
    
    # 简单特征交互
    if 'kilometer' in numeric_features and 'regDate' in numeric_features:
        # 先将regDate转换为车龄年数（以2020年为基准）
        current_year = 2020
        train_data['car_age'] = current_year - (train_data['regDate'] // 10000)
        test_data['car_age'] = current_year - (test_data['regDate'] // 10000)
        
        # 确保车龄为正数
        train_data['car_age'] = np.maximum(train_data['car_age'], 1)
        test_data['car_age'] = np.maximum(test_data['car_age'], 1)
        
        # 年均里程数（每年平均跑多少公里）
        train_data['km_per_year'] = train_data['kilometer'] / train_data['car_age']
        test_data['km_per_year'] = test_data['kilometer'] / test_data['car_age']
    
    if 'power' in numeric_features and 'kilometer' in numeric_features:
        # 功率里程比
        train_data['power_km_ratio'] = train_data['power'] / (train_data['kilometer'] + 1)
        test_data['power_km_ratio'] = test_data['power'] / (test_data['kilometer'] + 1)
    
    # 分箱特征
    if 'kilometer' in numeric_features:
        km_bins = [0, 1, 5, 10, 15, 20, float('inf')]
        train_data['km_bin'] = pd.cut(train_data['kilometer'], bins=km_bins, labels=False)
        test_data['km_bin'] = pd.cut(test_data['kilometer'], bins=km_bins, labels=False)
    
    if 'power' in numeric_features:
        power_bins = [0, 80, 120, 160, 200, float('inf')]
        train_data['power_bin'] = pd.cut(train_data['power'], bins=power_bins, labels=False)
        test_data['power_bin'] = pd.cut(test_data['power'], bins=power_bins, labels=False)
    
    print(f"特征工程后训练集: {train_data.shape}")
    print(f"特征工程后测试集: {test_data.shape}")
    
    return train_data, test_data

def rf_cross_validation(X, y):
    """随机森林交叉验证"""
    print("进行随机森林交叉验证...")
    
    # 不同复杂度的随机森林配置
    rf_configs = {
        'simple': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),
        'balanced': RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=15,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1
        ),
        'complex': RandomForestRegressor(
            n_estimators=300,
            max_depth=25,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    }
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    best_mae = float('inf')
    best_model = None
    
    for name, model in rf_configs.items():
        print(f"测试 {name} 配置...")
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"  交叉验证MAE: {cv_mae:.4f} ± {cv_std:.4f}")
        
        if cv_mae < best_mae:
            best_mae = cv_mae
            best_model = model
    
    print(f"最佳交叉验证MAE: {best_mae:.4f}")
    return best_model

def create_rf_ensemble(X_train, y_train, X_test):
    """创建随机森林集成"""
    print("创建随机森林集成...")
    
    # 多个随机森林变体
    models = [
        RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
        RandomForestRegressor(n_estimators=250, max_depth=22, random_state=123),
        RandomForestRegressor(n_estimators=180, max_depth=18, random_state=456),
        ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=789)
    ]
    
    predictions = []
    for i, model in enumerate(models):
        print(f"训练模型 {i+1}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.maximum(pred, 0)  # 确保非负
        predictions.append(pred)
    
    # 简单平均集成
    ensemble_pred = np.mean(predictions, axis=0)
    
    print(f"集成预测统计:")
    print(f"  均值: {ensemble_pred.mean():.2f}")
    print(f"  中位数: {np.median(ensemble_pred):.2f}")
    print(f"  范围: {ensemble_pred.min():.2f} - {ensemble_pred.max():.2f}")
    
    return ensemble_pred

def main():
    """主函数"""
    print("开始随机森林专项优化...")
    
    # 1. 加载数据
    train_data, test_data = load_and_preprocess_data()
    
    # 2. 特征工程
    train_data, test_data = create_rf_features(train_data, test_data)
    
    # 3. 准备建模数据
    feature_cols = [col for col in train_data.columns if col != 'price']
    feature_cols = [col for col in feature_cols if col in test_data.columns]
    
    X_train = train_data[feature_cols]
    y_train = train_data['price']
    X_test = test_data[feature_cols]
    
    print(f"最终特征数量: {len(feature_cols)}")
    print(f"训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")
    
    # 4. 交叉验证
    best_model = rf_cross_validation(X_train, y_train)
    
    # 5. 集成预测
    ensemble_pred = create_rf_ensemble(X_train, y_train, X_test)
    
    # 6. 保存结果
    submission = pd.DataFrame({
        'SaleID': range(len(ensemble_pred)),
        'price': ensemble_pred
    })
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = get_project_path('prediction_result', f'rf_focused_{timestamp}.csv')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    submission.to_csv(filename, index=False)
    print(f"结果已保存到: {filename}")
    
    return filename

if __name__ == "__main__":
    result_file = main()
    print("随机森林专项优化完成！")