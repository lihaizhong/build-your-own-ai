#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证香港疫情数据读取
"""

import pandas as pd
from pathlib import Path

def verify_hk_covid_data():
    current_dir = Path(__file__).parent
    excel_file = current_dir / "香港各区疫情数据_20250322.xlsx"
    
    print("=" * 100)
    print("验证香港疫情数据读取")
    print("=" * 100)
    
    if not excel_file.exists():
        print("❌ 香港疫情数据文件不存在")
        return
    
    try:
        # 读取前20行数据
        df = pd.read_excel(excel_file, nrows=20)
        total_rows = len(pd.read_excel(excel_file))
        
        print(f"✅ 文件读取成功")
        print(f"📊 文件总行数: {total_rows}")
        print(f"📊 读取行数: {len(df)}")
        print(f"📊 列数: {df.shape[1]}")
        
        print(f"\n📋 列信息:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\n📅 前5行数据预览:")
        print("-" * 100)
        # 选择关键列显示
        key_columns = ['日期', '地区', '新增确诊', '累计确诊', '疫苗接种率']
        available_columns = [col for col in key_columns if col in df.columns]
        if available_columns:
            print(df[available_columns].head().to_string(index=True))
        else:
            print(df.head().to_string(index=True))
        
        # 简单统计
        if '新增确诊' in df.columns:
            total_cases = df['新增确诊'].sum()
            avg_cases = df['新增确诊'].mean()
            print(f"\n📊 前20行统计:")
            print(f"  总新增确诊: {total_cases} 例")
            print(f"  平均新增确诊: {avg_cases:.1f} 例")
        
        if '地区' in df.columns:
            unique_districts = df['地区'].nunique()
            print(f"  涉及地区数: {unique_districts} 个")
        
        print("\n✅ 验证完成!")
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")

if __name__ == "__main__":
    verify_hk_covid_data()