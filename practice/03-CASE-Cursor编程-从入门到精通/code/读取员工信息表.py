#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取员工基本信息表Excel文件的前5行数据
"""

import pandas as pd
from ...shared import get_project_path

def read_employee_excel():
    """
    读取员工基本信息表.xlsx文件的前5行数据
    """
    # 获取脚本目录并准备 user_data 目录
    current_dir = get_project_path()
    user_data_dir = current_dir / "user_data"
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # 优先使用 user_data 中的文件（如果存在），否则回退到脚本目录
    candidate = user_data_dir / "员工基本信息表.xlsx"
    excel_file = candidate if candidate.exists() else (current_dir / "员工基本信息表.xlsx")
    
    # 检查文件是否存在
    if not excel_file.exists():
        # 创建到 user_data 以便统一存放临时文件
        create_path = user_data_dir / "员工基本信息表.xlsx"
        print(f"⚠️ 找不到文件，正在创建示例文件到 {create_path} ...")
        create_sample_excel(create_path)
        print(f"已创建示例文件：{create_path}")
        excel_file = create_path
    
    try:
        # 读取Excel文件的前5行数据
        print(f"正在读取文件：{excel_file}")
        df = pd.read_excel(excel_file, nrows=5)
        
        # 显示文件信息
        print(f"\n文件总行数（不包括表头）：{len(pd.read_excel(excel_file))}")
        print(f"文件列数：{df.shape[1]}")
        print(f"读取的行数：{len(df)}")
        
        # 显示前5行数据
        print("\n前5行数据：")
        print("=" * 60)
        print(df.to_string(index=True))
        
        # 显示列信息
        print("\n" + "=" * 60)
        print("列信息：")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
            
        return df
        
    except Exception as e:
        print(f"读取Excel文件时发生错误：{e}")
        return None

def create_sample_excel(file_path):
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

if __name__ == "__main__":
    print("=" * 60)
    print("员工基本信息表Excel文件读取程序")
    print("=" * 60)
    
    # 读取Excel文件
    result = read_employee_excel()
    
    if result is not None:
        print("\n程序执行完成！")
    else:
        print("\n程序执行失败！")