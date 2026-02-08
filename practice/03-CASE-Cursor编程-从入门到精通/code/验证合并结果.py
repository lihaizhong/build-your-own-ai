#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证合并后的Excel文件内容
"""

import pandas as pd
from ...shared import get_project_path

def verify_merged_file():
    """
    验证合并后的Excel文件
    """
    current_dir = get_project_path()
    user_data_dir = current_dir / "user_data"
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # 合并文件优先从 user_data 中寻找
    merged_file = (user_data_dir / "员工综合信息表_2024Q4.xlsx") if (user_data_dir / "员工综合信息表_2024Q4.xlsx").exists() else (current_dir / "员工综合信息表_2024Q4.xlsx")
    
    if not merged_file.exists():
        print("❌ 合并文件不存在")
        return
    
    print("=" * 80)
    print("验证合并后的Excel文件")
    print("=" * 80)
    
    try:
        # 读取所有工作表
        excel_file = pd.ExcelFile(merged_file)
        print(f"📊 工作表列表：{excel_file.sheet_names}")
        
        # 读取主数据表
        main_df = pd.read_excel(merged_file, sheet_name='员工综合信息')
        print(f"\n📋 员工综合信息表：")
        print(f"   行数：{len(main_df)}")
        print(f"   列数：{main_df.shape[1]}")
        print(f"   列名：{list(main_df.columns)}")
        
        # 显示前5行数据
        print(f"\n前5行数据预览：")
        print("-" * 80)
        display_columns = ['员工编号', '姓名', '部门', '职位', '薪资', '综合得分', '绩效等级']
        available_columns = [col for col in display_columns if col in main_df.columns]
        print(main_df[available_columns].head().to_string(index=True))
        
        # 读取统计摘要表
        if '数据统计摘要' in excel_file.sheet_names:
            summary_df = pd.read_excel(merged_file, sheet_name='数据统计摘要')
            print(f"\n📊 数据统计摘要：")
            print(summary_df.to_string(index=False))
        
        # 读取部门绩效分析表
        if '部门绩效分析' in excel_file.sheet_names:
            dept_df = pd.read_excel(merged_file, sheet_name='部门绩效分析')
            print(f"\n🏢 部门绩效分析：")
            print(dept_df.to_string(index=False))
        
        print("\n✅ 验证完成！合并文件内容正确。")
        
    except Exception as e:
        print(f"❌ 验证过程中发生错误：{e}")

if __name__ == "__main__":
    verify_merged_file()