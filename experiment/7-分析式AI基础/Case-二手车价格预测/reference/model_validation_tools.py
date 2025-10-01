#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型验证工具合集
整合MAE分析、预测问题分析等工具
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

def analyze_mae_issue(y_true, y_pred, title="MAE分析"):
    """深入分析MAE问题"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"=== {title} ===")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # 分析残差
    residuals = y_true - y_pred
    print(f"\n残差统计:")
    print(f"  均值: {residuals.mean():.4f}")
    print(f"  标准差: {residuals.std():.4f}")
    print(f"  最大正残差: {residuals.max():.4f}")
    print(f"  最大负残差: {residuals.min():.4f}")
    
    # 分价格区间分析
    price_bins = [0, 1000, 5000, 10000, 20000, float('inf')]
    bin_labels = ['<1k', '1k-5k', '5k-10k', '10k-20k', '>20k']
    
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'residual': residuals})
    df['price_bin'] = pd.cut(df['true'], bins=price_bins, labels=bin_labels)
    
    bin_analysis = df.groupby('price_bin').agg({
        'residual': ['count', 'mean', 'std']
    }).round(4)
    
    print(f"\n分价格区间MAE分析:")
    print(bin_analysis)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'residuals': residuals,
        'bin_analysis': bin_analysis
    }

def analyze_prediction_issues(train_data, test_data, model, feature_cols):
    """分析预测问题"""
    print("=== 预测问题分析 ===")
    
    X_train = train_data[feature_cols]
    y_train = train_data['price']
    X_test = test_data[feature_cols]
    
    # 训练集预测
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    
    # 测试集预测
    test_pred = model.predict(X_test)
    
    print(f"训练集MAE: {train_mae:.4f}")
    print(f"测试集预测统计:")
    print(f"  均值: {test_pred.mean():.2f}")
    print(f"  标准差: {test_pred.std():.2f}")
    print(f"  最小值: {test_pred.min():.2f}")
    print(f"  最大值: {test_pred.max():.2f}")
    
    # 检查异常预测
    negative_preds = test_pred[test_pred < 0]
    if len(negative_preds) > 0:
        print(f"警告: {len(negative_preds)} 个负值预测")
        test_pred = np.maximum(test_pred, 0)
    
    return test_pred

def comprehensive_mae_analysis(models_dict, X_train, y_train, cv_folds=5):
    """综合MAE分析"""
    print("=== 综合MAE分析 ===")
    
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models_dict.items():
        print(f"\n分析模型: {name}")
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, 
                                   cv=kfold, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"  交叉验证MAE: {cv_mae:.4f} ± {cv_std:.4f}")
        
        # 训练后评估
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        
        print(f"  训练集MAE: {train_mae:.4f}")
        
        results[name] = {
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'train_mae': train_mae,
            'model': model
        }
    
    # 排序显示
    sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_mae'])
    print(f"\n模型排序 (按交叉验证MAE):")
    for name, result in sorted_results:
        print(f"  {name}: {result['cv_mae']:.4f}")
    
    return results

def check_rf_results(result_file_path):
    """检查随机森林结果文件"""
    try:
        results = pd.read_csv(result_file_path)
        print(f"结果文件: {result_file_path}")
        print(f"形状: {results.shape}")
        print(f"预测统计:")
        print(f"  均值: {results['price'].mean():.2f}")
        print(f"  中位数: {results['price'].median():.2f}")
        print(f"  标准差: {results['price'].std():.2f}")
        print(f"  范围: {results['price'].min():.2f} - {results['price'].max():.2f}")
        
        # 检查异常值
        negative_count = (results['price'] < 0).sum()
        if negative_count > 0:
            print(f"警告: {negative_count} 个负值预测")
        
        return results
    except Exception as e:
        print(f"读取结果文件失败: {e}")
        return None

if __name__ == "__main__":
    print("模型验证工具合集已加载")