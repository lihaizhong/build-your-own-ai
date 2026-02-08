#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from ...shared import get_project_path

print("开始测试...")

# 使用 get_project_path 并优先从 user_data 读取临时文件
base_dir = get_project_path()
user_data_dir = base_dir / "user_data"
user_data_dir.mkdir(parents=True, exist_ok=True)

# 测试读取员工基本信息表
try:
    info_candidate = user_data_dir / "员工基本信息表.xlsx"
    info_path = info_candidate if info_candidate.exists() else (base_dir / "员工基本信息表.xlsx")
    info_df = pd.read_excel(info_path, nrows=5)
    print("✅ 员工基本信息表读取成功")
    print(f"行数: {len(info_df)}, 列数: {info_df.shape[1]}")
    print(info_df)
except Exception as e:
    print(f"❌ 员工基本信息表读取失败: {e}")

print("\n" + "="*50 + "\n")

# 测试读取员工绩效表
try:
    perf_candidate = user_data_dir / "员工绩效表.xlsx"
    perf_path = perf_candidate if perf_candidate.exists() else (base_dir / "员工绩效表.xlsx")
    perf_df = pd.read_excel(perf_path, nrows=5)
    print("✅ 员工绩效表读取成功")
    print(f"行数: {len(perf_df)}, 列数: {perf_df.shape[1]}")
    print(perf_df)
except Exception as e:
    print(f"❌ 员工绩效表读取失败: {e}")

print("\n测试完成！")