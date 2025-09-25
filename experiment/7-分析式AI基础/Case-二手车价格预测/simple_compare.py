#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的预测结果对比分析
"""

import csv

def read_csv_simple(filename, delimiter=' '):
    """简单读取CSV文件"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader)
        for row in reader:
            data.append(row)
    return headers, data

def get_price_stats(prices):
    """计算价格统计信息"""
    prices = [float(p) for p in prices if p and p != 'price']
    prices.sort()
    n = len(prices)
    
    return {
        'count': n,
        'min': min(prices),
        'max': max(prices),
        'mean': sum(prices) / n,
        'median': prices[n//2],
        'q25': prices[n//4],
        'q75': prices[3*n//4],
        'std': (sum((p - sum(prices)/n)**2 for p in prices) / n)**0.5
    }

def main():
    print("="*60)
    print("简单预测结果对比分析")
    print("="*60)
    
    # 读取原始训练数据
    try:
        headers, train_data = read_csv_simple('used_car_train_20200313.csv', ' ')
        price_idx = headers.index('price')
        original_prices = [row[price_idx] for row in train_data]
        orig_stats = get_price_stats(original_prices)
        print(f"原始训练数据加载完成: {len(train_data)} 条记录")
    except Exception as e:
        print(f"读取训练数据失败: {e}")
        return
    
    # 读取优化后的预测结果
    try:
        headers, opt_data = read_csv_simple('optimized_xgb_result_20250926_024549.csv', ',')
        price_idx = headers.index('price')
        opt_prices = [row[price_idx] for row in opt_data]
        opt_stats = get_price_stats(opt_prices)
        print(f"优化预测结果加载完成: {len(opt_data)} 条记录")
    except Exception as e:
        print(f"读取优化预测结果失败: {e}")
        opt_stats = None
    
    # 读取集成结果
    try:
        headers, ens_data = read_csv_simple('ensemble_result_smart_weighted_20250926_025722.csv', ',')
        price_idx = headers.index('price')
        ens_prices = [row[price_idx] for row in ens_data]
        ens_stats = get_price_stats(ens_prices)
        print(f"集成预测结果加载完成: {len(ens_data)} 条记录")
    except Exception as e:
        print(f"读取集成预测结果失败: {e}")
        ens_stats = None
    
    # 对比统计信息
    print("\n" + "="*60)
    print("统计信息对比")
    print("="*60)
    
    stats_names = ['count', 'min', 'max', 'mean', 'median', 'q25', 'q75', 'std']
    print(f"{'统计量':<8} {'原始':<10} {'优化版':<10} {'集成版':<10}")
    print("-" * 50)
    
    for stat in stats_names:
        orig_val = orig_stats[stat]
        line = f"{stat:<8} {orig_val:<10.0f}"
        
        if opt_stats:
            opt_val = opt_stats[stat]
            line += f" {opt_val:<10.0f}"
        else:
            line += f" {'N/A':<10}"
        
        if ens_stats:
            ens_val = ens_stats[stat]
            line += f" {ens_val:<10.0f}"
        else:
            line += f" {'N/A':<10}"
        
        print(line)
    
    # 价格分布分析
    print("\n" + "="*60)
    print("价格区间分布分析")
    print("="*60)
    
    bins = [0, 1000, 3000, 5000, 10000, 20000, 50000, float('inf')]
    labels = ['0-1k', '1k-3k', '3k-5k', '5k-10k', '10k-20k', '20k-50k', '50k+']
    
    def get_distribution(prices):
        """计算价格在各区间的分布"""
        dist = {label: 0 for label in labels}
        prices = [float(p) for p in prices if p and p != 'price']
        
        for price in prices:
            for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
                if low <= price < high:
                    dist[labels[i]] += 1
                    break
        
        total = len(prices)
        return {k: (v, v/total*100) for k, v in dist.items()}
    
    orig_dist = get_distribution(original_prices)
    
    print(f"{'区间':<8} {'原始':<12} {'优化版':<12} {'集成版':<12}")
    print("-" * 50)
    
    for label in labels:
        orig_count, orig_pct = orig_dist[label]
        line = f"{label:<8} {orig_count:>4}({orig_pct:>4.1f}%)"
        
        if opt_stats:
            opt_dist = get_distribution(opt_prices)
            opt_count, opt_pct = opt_dist[label]
            line += f" {opt_count:>4}({opt_pct:>4.1f}%)"
        else:
            line += f" {'N/A':<11}"
        
        if ens_stats:
            ens_dist = get_distribution(ens_prices)
            ens_count, ens_pct = ens_dist[label]
            line += f" {ens_count:>4}({ens_pct:>4.1f}%)"
        else:
            line += f" {'N/A':<11}"
        
        print(line)

if __name__ == "__main__":
    main()