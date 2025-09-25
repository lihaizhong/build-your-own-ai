#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正预测结果，确保价格为正数
"""

import pandas as pd
import numpy as np
from datetime import datetime

def fix_predictions():
    """修正预测结果"""
    print("修正预测结果...")
    
    # 读取预测结果
    result_df = pd.read_csv('rf_result_20250926_023210.csv')
    
    print(f"原始预测结果统计:")
    print(f"总记录数: {len(result_df)}")
    print(f"负价格数量: {(result_df['price'] < 0).sum()}")
    print(f"价格范围: {result_df['price'].min():.2f} - {result_df['price'].max():.2f}")
    
    # 修正负价格：将负价格设为正数的绝对值或最小正价格
    negative_mask = result_df['price'] < 0
    if negative_mask.sum() > 0:
        # 将负价格设为绝对值，但最小为100元
        result_df.loc[negative_mask, 'price'] = np.maximum(
            np.abs(result_df.loc[negative_mask, 'price']), 
            100
        )
        print(f"已修正 {negative_mask.sum()} 个负价格")
    
    # 确保价格在合理范围内
    result_df['price'] = np.clip(result_df['price'], 100, 100000)
    
    print(f"修正后预测结果统计:")
    print(f"价格范围: {result_df['price'].min():.2f} - {result_df['price'].max():.2f}")
    print(f"价格均值: {result_df['price'].mean():.2f}")
    print(f"价格中位数: {result_df['price'].median():.2f}")
    
    # 保存修正后的结果
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    fixed_filename = f"rf_result_{current_time}_fixed.csv"
    result_df.to_csv(fixed_filename, index=False)
    
    print(f"修正后的预测结果已保存至: {fixed_filename}")
    
    return fixed_filename, result_df

if __name__ == "__main__":
    fix_predictions()