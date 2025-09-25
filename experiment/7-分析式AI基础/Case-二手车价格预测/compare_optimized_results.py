#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查优化后的预测结果分布
"""

import pandas as pd
import numpy as np

def compare_optimized_results():
    """对比优化后的预测结果"""
    print("="*60)
    print("对比优化前后的预测结果")
    print("="*60)
    
    # 原始训练集价格分布
    train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    original_price = train_df['price']
    
    # 读取不同版本的预测结果
    try:
        # 第一版XGBoost结果
        xgb_v1 = pd.read_csv('xgb_result_20250926_023816.csv')
        print(f"XGBoost v1加载完成: {xgb_v1.shape}")
    except:
        print("未找到XGBoost v1结果")
        xgb_v1 = None
    
    try:
        # 优化版XGBoost结果
        xgb_opt = pd.read_csv('optimized_xgb_result_20250926_024549.csv')
        print(f"优化XGBoost加载完成: {xgb_opt.shape}")
    except:
        print("未找到优化XGBoost结果")
        xgb_opt = None
    
    # 对比分析
    print("\n" + "="*60)
    print("分位数对比（原始 vs 优化版）")
    print("="*60)
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"{'分位数':<6} {'原始':<10} {'优化版':<10} {'差异':<10} {'改进%':<8}")
    print("-" * 50)
    
    for p in percentiles:
        orig_val = np.percentile(original_price, p)
        
        if xgb_opt is not None:
            opt_val = np.percentile(xgb_opt['price'], p)
            diff = opt_val - orig_val
            improve_pct = abs(diff) / orig_val * 100
            print(f"{p:>3d}%   {orig_val:<10.0f} {opt_val:<10.0f} {diff:>+8.0f} {improve_pct:<6.1f}%")
    
    # 价格区间分布对比
    print("\n" + "="*60)
    print("价格区间分布对比")
    print("="*60)
    
    bins = [0, 1000, 3000, 5000, 10000, 20000, 50000, float('inf')]
    labels = ['0-1k', '1k-3k', '3k-5k', '5k-10k', '10k-20k', '20k-50k', '50k+']
    
    # 原始分布
    orig_ranges = pd.cut(original_price, bins=bins, labels=labels)
    orig_counts = orig_ranges.value_counts().sort_index()
    
    print(f"{'区间':<8} {'原始':<8} {'占比%':<6}", end="")
    if xgb_opt is not None:
        print(f" {'优化版':<8} {'占比%':<6} {'差异':<6}")
        print("-" * 48)
        
        opt_ranges = pd.cut(xgb_opt['price'], bins=bins, labels=labels)
        opt_counts = opt_ranges.value_counts().sort_index()
        
        for label in labels:
            orig_count = orig_counts.get(label, 0)
            orig_pct = orig_count / len(original_price) * 100
            
            opt_count = opt_counts.get(label, 0)
            opt_pct = opt_count / len(xgb_opt) * 100
            
            diff_pct = opt_pct - orig_pct
            
            print(f"{label:<8} {orig_count:<8} {orig_pct:<6.1f} {opt_count:<8} {opt_pct:<6.1f} {diff_pct:>+5.1f}")
    else:
        print()
        print("-" * 22)
        for label in labels:
            orig_count = orig_counts.get(label, 0)
            orig_pct = orig_count / len(original_price) * 100
            print(f"{label:<8} {orig_count:<8} {orig_pct:<6.1f}")

    # 关键统计量对比
    if xgb_opt is not None:
        print("\n" + "="*60)
        print("关键统计量对比")
        print("="*60)
        
        opt_price = xgb_opt['price']
        
        stats_comparison = [
            ('均值', original_price.mean(), opt_price.mean()),
            ('中位数', original_price.median(), opt_price.median()),
            ('标准差', original_price.std(), opt_price.std()),
            ('最小值', original_price.min(), opt_price.min()),
            ('最大值', original_price.max(), opt_price.max()),
        ]
        
        print(f"{'统计量':<8} {'原始':<10} {'优化版':<10} {'差异':<10} {'差异%':<8}")
        print("-" * 48)
        
        for name, orig, opt in stats_comparison:
            diff = opt - orig
            diff_pct = (diff / orig) * 100 if orig != 0 else 0
            print(f"{name:<8} {orig:<10.0f} {opt:<10.0f} {diff:>+8.0f} {diff_pct:>+6.1f}%")

if __name__ == "__main__":
    compare_optimized_results()