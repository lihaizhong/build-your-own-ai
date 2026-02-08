#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版的双Excel文件读取脚本，用于验证功能
"""

import pandas as pd
from ...shared import get_project_path

def main():
    current_dir = get_project_path()
    user_data_dir = current_dir / "user_data"
    user_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("双Excel文件读取验证程序")
    print("=" * 60)
    
    # 文件路径
    # 优先从 user_data 读取（便于临时文件管理），否则回退到脚本目录
    info_file = (user_data_dir / "员工基本信息表.xlsx") if (user_data_dir / "员工基本信息表.xlsx").exists() else (current_dir / "员工基本信息表.xlsx")
    perf_file = (user_data_dir / "员工绩效表.xlsx") if (user_data_dir / "员工绩效表.xlsx").exists() else (current_dir / "员工绩效表.xlsx")
    
    # 读取员工基本信息表前5行
    print("\n1. 读取员工基本信息表前5行：")
    print("-" * 40)
    try:
        info_df = pd.read_excel(info_file, nrows=5)
        print(f"成功读取 {len(info_df)} 行，{info_df.shape[1]} 列")
        print("\n数据内容：")
        for i, row in info_df.iterrows():
            print(f"第{i+1}行: {row['员工编号']} - {row['姓名']} - {row['部门']} - {row['职位']}") # type: ignore
    except Exception as e:
        print(f"读取失败: {e}")
    
    # 读取员工绩效表前5行
    print("\n2. 读取员工绩效表前5行：")
    print("-" * 40)
    try:
        perf_df = pd.read_excel(perf_file, nrows=5)
        print(f"成功读取 {len(perf_df)} 行，{perf_df.shape[1]} 列")
        print("\n数据内容：")
        for i, row in perf_df.iterrows():
            print(f"第{i+1}行: {row['员工编号']} - {row['姓名']} - 综合得分: {row['综合得分']} - {row['绩效等级']}") # type: ignore
    except Exception as e:
        print(f"读取失败: {e}")
    
    print("\n" + "=" * 60)
    print("程序执行完成!")

if __name__ == "__main__":
    main()