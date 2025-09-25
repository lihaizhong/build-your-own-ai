#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析原始价格分布与预测结果对比
"""

import pandas as pd
import numpy as np

def analyze_price_distribution():
    """分析原始价格分布"""
    print("="*50)
    print("分析原始训练集价格分布")
    print("="*50)
    
    # 读取原始训练集
    train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    print(f"原始训练集形状: {train_df.shape}")
    
    # 分析原始价格分布
    price = train_df['price']
    print("\n原始价格统计:")
    print(f"最小值: {price.min()}")
    print(f"最大值: {price.max()}")
    print(f"均值: {price.mean():.2f}")
    print(f"中位数: {price.median():.2f}")
    print(f"标准差: {price.std():.2f}")
    
    # 价格分位数
    print("\n价格分位数:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(price, p)
        print(f"{p:2d}%: {value:.2f}")
    
    # 价格区间分布
    print("\n价格区间分布:")
    bins = [0, 1000, 3000, 5000, 10000, 20000, 50000, float('inf')]
    labels = ['0-1k', '1k-3k', '3k-5k', '5k-10k', '10k-20k', '20k-50k', '50k+']
    price_ranges = pd.cut(price, bins=bins, labels=labels)
    counts = price_ranges.value_counts().sort_index()
    for label, count in counts.items():
        percentage = count / len(price) * 100
        print(f"{label:>8}: {count:6d} ({percentage:5.1f}%)")
    
    return price

def analyze_predictions():
    """分析预测结果分布"""
    print("\n" + "="*50)
    print("分析预测结果分布")
    print("="*50)
    
    # 读取最新的预测结果
    try:
        xgb_result = pd.read_csv('xgb_result_20250926_023816.csv')
        print(f"XGBoost预测结果形状: {xgb_result.shape}")
        
        pred_price = xgb_result['price']
        print("\nXGBoost预测价格统计:")
        print(f"最小值: {pred_price.min():.2f}")
        print(f"最大值: {pred_price.max():.2f}")
        print(f"均值: {pred_price.mean():.2f}")
        print(f"中位数: {pred_price.median():.2f}")
        print(f"标准差: {pred_price.std():.2f}")
        
        # 预测价格分位数
        print("\n预测价格分位数:")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(pred_price, p)
            print(f"{p:2d}%: {value:.2f}")
        
        # 预测价格区间分布
        print("\n预测价格区间分布:")
        bins = [0, 1000, 3000, 5000, 10000, 20000, 50000, float('inf')]
        labels = ['0-1k', '1k-3k', '3k-5k', '5k-10k', '10k-20k', '20k-50k', '50k+']
        pred_ranges = pd.cut(pred_price, bins=bins, labels=labels)
        pred_counts = pred_ranges.value_counts().sort_index()
        for label, count in pred_counts.items():
            percentage = count / len(pred_price) * 100
            print(f"{label:>8}: {count:6d} ({percentage:5.1f}%)")
            
        return pred_price
        
    except FileNotFoundError:
        print("未找到XGBoost预测结果文件")
        return None

def compare_distributions(original_price, pred_price):
    """对比分布差异"""
    if pred_price is None:
        return
        
    print("\n" + "="*50)
    print("分布对比分析")
    print("="*50)
    
    print("\n关键统计量对比:")
    print(f"{'指标':<10} {'原始':<12} {'预测':<12} {'差异':<12}")
    print("-" * 50)
    
    metrics = [
        ('最小值', original_price.min(), pred_price.min()),
        ('最大值', original_price.max(), pred_price.max()),
        ('均值', original_price.mean(), pred_price.mean()),
        ('中位数', original_price.median(), pred_price.median()),
        ('标准差', original_price.std(), pred_price.std()),
    ]
    
    for name, orig, pred in metrics:
        diff = pred - orig
        diff_pct = (diff / orig) * 100 if orig != 0 else 0
        print(f"{name:<10} {orig:<12.2f} {pred:<12.2f} {diff:>+8.2f} ({diff_pct:>+5.1f}%)")
    
    # 分位数对比
    print("\n分位数对比:")
    print(f"{'分位数':<6} {'原始':<12} {'预测':<12} {'差异':<12}")
    print("-" * 44)
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        orig_val = np.percentile(original_price, p)
        pred_val = np.percentile(pred_price, p)
        diff = pred_val - orig_val
        print(f"{p:>3d}%   {orig_val:<12.2f} {pred_val:<12.2f} {diff:>+8.2f}")

def suggest_improvements():
    """建议改进方向"""
    print("\n" + "="*50)
    print("改进建议")
    print("="*50)
    
    print("\n1. 价格分布校准:")
    print("   - 当前预测可能偏离了原始价格分布")
    print("   - 需要调整模型使预测分布更接近训练集分布")
    
    print("\n2. 可能的改进方法:")
    print("   - 对数变换: 对价格进行log变换后建模")
    print("   - 分位数回归: 使用分位数损失函数")
    print("   - 后处理校准: 对预测结果进行分布校准")
    print("   - 特征工程: 创建更好的价格相关特征")
    
    print("\n3. 模型调优:")
    print("   - 调整XGBoost参数（学习率、树深度、正则化）")
    print("   - 增加模型复杂度以更好拟合价格分布")
    print("   - 使用更多的树或更深的树")

def main():
    """主函数"""
    # 分析原始价格分布
    original_price = analyze_price_distribution()
    
    # 分析预测结果分布
    pred_price = analyze_predictions()
    
    # 对比分布差异
    compare_distributions(original_price, pred_price)
    
    # 提供改进建议
    suggest_improvements()

if __name__ == "__main__":
    main()