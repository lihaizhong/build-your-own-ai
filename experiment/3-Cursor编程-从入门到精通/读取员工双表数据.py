#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取员工基本信息表.xlsx和员工绩效表.xlsx文件的前5行数据
"""

import pandas as pd
import os
from pathlib import Path

def read_multiple_excel_files():
    """
    读取员工基本信息表.xlsx和员工绩效表.xlsx文件的前5行数据
    """
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # Excel文件路径
    employee_info_file = current_dir / "员工基本信息表.xlsx"
    employee_performance_file = current_dir / "员工绩效表.xlsx"
    
    print("=" * 80)
    print("员工信息与绩效数据Excel文件读取程序")
    print("=" * 80)
    
    # 处理员工基本信息表
    print("\n📊 处理员工基本信息表...")
    employee_info_df = process_excel_file(employee_info_file, "员工基本信息表", create_employee_info_sample)
    
    # 处理员工绩效表
    print("\n📈 处理员工绩效表...")
    employee_performance_df = process_excel_file(employee_performance_file, "员工绩效表", create_employee_performance_sample)
    
    # 数据分析与对比
    if employee_info_df is not None and employee_performance_df is not None:
        print("\n" + "=" * 80)
        print("📋 数据分析汇总")
        print("=" * 80)
        analyze_data(employee_info_df, employee_performance_df)
    
    return employee_info_df, employee_performance_df

def process_excel_file(file_path, file_description, create_sample_func):
    """
    处理单个Excel文件的读取
    """
    # 检查文件是否存在
    if not file_path.exists():
        print(f"⚠️  错误：找不到文件 {file_path}")
        print(f"正在创建示例{file_description}文件...")
        create_sample_func(file_path)
        print(f"✅ 已创建示例文件：{file_path}")
    
    try:
        # 读取Excel文件的前5行数据
        print(f"📖 正在读取文件：{file_path}")
        df = pd.read_excel(file_path, nrows=5)
        
        # 显示文件信息
        total_rows = len(pd.read_excel(file_path))
        print(f"   文件总行数（不包括表头）：{total_rows}")
        print(f"   文件列数：{df.shape[1]}")
        print(f"   读取的行数：{len(df)}")
        
        # 显示前5行数据
        print(f"\n📋 {file_description}前5行数据：")
        print("-" * 70)
        print(df.to_string(index=True))
        
        # 显示列信息
        print(f"\n📝 {file_description}列信息：")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
            
        return df
        
    except Exception as e:
        print(f"❌ 读取{file_description}文件时发生错误：{e}")
        return None

def create_employee_info_sample(file_path):
    """
    创建示例的员工基本信息表Excel文件
    """
    # 示例员工数据
    employee_data = {
        '员工编号': ['E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008', 'E009', 'E010'],
        '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十', '郑十一', '王十二'],
        '性别': ['男', '女', '男', '女', '男', '女', '男', '女', '男', '女'],
        '年龄': [28, 32, 25, 29, 35, 27, 31, 26, 33, 30],
        '部门': ['技术部', '人事部', '技术部', '财务部', '市场部', '技术部', '人事部', '技术部', '财务部', '市场部'],
        '职位': ['软件工程师', '人事专员', '前端工程师', '会计师', '市场专员', '后端工程师', '招聘主管', '测试工程师', '财务经理', '市场经理'],
        '入职日期': ['2022-01-15', '2021-03-20', '2023-06-10', '2020-11-05', '2022-08-12', '2023-02-28', '2021-07-18', '2023-04-03', '2019-12-01', '2022-05-25'],
        '薪资': [12000, 8000, 10000, 9000, 7500, 13000, 11000, 9500, 15000, 12500]
    }
    
    # 创建DataFrame
    df = pd.DataFrame(employee_data)
    
    # 保存为Excel文件
    df.to_excel(file_path, index=False, engine='openpyxl')

def create_employee_performance_sample(file_path):
    """
    创建示例的员工绩效表Excel文件
    """
    # 示例绩效数据
    performance_data = {
        '员工编号': ['E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008', 'E009', 'E010'],
        '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十', '郑十一', '王十二'],
        '考核月份': ['2024-12', '2024-12', '2024-12', '2024-12', '2024-12', '2024-12', '2024-12', '2024-12', '2024-12', '2024-12'],
        '工作质量得分': [92, 88, 95, 85, 78, 90, 87, 93, 89, 91],
        '工作效率得分': [89, 92, 87, 90, 82, 94, 88, 91, 86, 93],
        '团队协作得分': [95, 85, 90, 88, 75, 92, 89, 87, 91, 94],
        '创新能力得分': [88, 79, 92, 83, 71, 89, 84, 90, 85, 87],
        '综合得分': [91, 86, 91, 86.5, 76.5, 91.25, 87, 90.25, 87.75, 91.25],
        '绩效等级': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'A', 'B', 'A']
    }
    
    # 创建DataFrame
    df = pd.DataFrame(performance_data)
    
    # 保存为Excel文件
    df.to_excel(file_path, index=False, engine='openpyxl')

def analyze_data(info_df, performance_df):
    """
    分析员工信息和绩效数据
    """
    print("🔍 数据关联分析：")
    print(f"   员工基本信息表记录数：{len(info_df)}")
    print(f"   员工绩效表记录数：{len(performance_df)}")
    
    # 检查共同员工
    common_employees = set(info_df['员工编号']) & set(performance_df['员工编号'])
    print(f"   共同员工数量：{len(common_employees)}")
    print(f"   共同员工编号：{', '.join(sorted(common_employees))}")
    
    # 绩效统计
    if '综合得分' in performance_df.columns:
        avg_score = performance_df['综合得分'].mean()
        max_score = performance_df['综合得分'].max()
        min_score = performance_df['综合得分'].min()
        print(f"\n📊 绩效得分统计（前5名员工）：")
        print(f"   平均综合得分：{avg_score:.2f}")
        print(f"   最高综合得分：{max_score}")
        print(f"   最低综合得分：{min_score}")
    
    # 部门统计
    if '部门' in info_df.columns:
        dept_count = info_df['部门'].value_counts()
        print(f"\n🏢 部门分布统计（前5名员工）：")
        for dept, count in dept_count.items():
            print(f"   {dept}：{count}人")

if __name__ == "__main__":
    # 读取两个Excel文件
    info_df, performance_df = read_multiple_excel_files()
    
    if info_df is not None and performance_df is not None:
        print("\n✅ 程序执行完成！所有文件读取成功！")
    else:
        print("\n❌ 程序执行过程中遇到问题！")