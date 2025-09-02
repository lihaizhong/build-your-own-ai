#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

print("开始测试...")

# 测试读取员工基本信息表
try:
    info_df = pd.read_excel("员工基本信息表.xlsx", nrows=5)
    print("✅ 员工基本信息表读取成功")
    print(f"行数: {len(info_df)}, 列数: {info_df.shape[1]}")
    print(info_df)
except Exception as e:
    print(f"❌ 员工基本信息表读取失败: {e}")

print("\n" + "="*50 + "\n")

# 测试读取员工绩效表
try:
    perf_df = pd.read_excel("员工绩效表.xlsx", nrows=5)
    print("✅ 员工绩效表读取成功")
    print(f"行数: {len(perf_df)}, 列数: {perf_df.shape[1]}")
    print(perf_df)
except Exception as e:
    print(f"❌ 员工绩效表读取失败: {e}")

print("\n测试完成！")