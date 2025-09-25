#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单微调版本 - 只修正明显的异常值
保持677.0636分基准的核心分布特征
"""

import csv

def load_baseline():
    """加载基准版本"""
    with open('optimized_xgb_result_20250926_024549.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        price_idx = headers.index('price')
        
        sale_ids = []
        prices = []
        
        for row in reader:
            sale_ids.append(int(row[0]))
            prices.append(float(row[price_idx]))
    
    return sale_ids, prices

def simple_outlier_fix(prices):
    """简单的异常值修正"""
    fixed_prices = []
    
    for price in prices:
        # 只修正明显的异常值
        if price < 30:
            # 过低的价格调整到合理范围
            fixed_price = 50 + (price - 30) * 0.5
        elif price > 90000:
            # 过高的价格适度降低
            fixed_price = 85000 + (price - 90000) * 0.3
        else:
            # 其他价格保持不变
            fixed_price = price
        
        fixed_prices.append(fixed_price)
    
    return fixed_prices

def main():
    print("简单微调 - 仅修正异常值")
    print("="*40)
    
    # 加载基准数据
    sale_ids, baseline_prices = load_baseline()
    print(f"基准版本: 均值={sum(baseline_prices)/len(baseline_prices):.0f}")
    
    # 简单修正
    fixed_prices = simple_outlier_fix(baseline_prices)
    print(f"修正后: 均值={sum(fixed_prices)/len(fixed_prices):.0f}")
    
    # 统计修正数量
    changes = sum(1 for old, new in zip(baseline_prices, fixed_prices) if abs(old - new) > 1)
    print(f"修正样本数: {changes} / {len(baseline_prices)}")
    
    # 保存结果
    import time
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'simple_fix_{timestamp}.csv'
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['SaleID', 'price'])
        
        for sale_id, price in zip(sale_ids, fixed_prices):
            writer.writerow([sale_id, int(price)])
    
    print(f"保存到: {filename}")
    print("\n这个版本几乎不改变原有分布，")
    print("只修正了极端异常值，应该是最安全的选择。")

if __name__ == "__main__":
    main()