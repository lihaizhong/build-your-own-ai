#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并员工基本信息表和员工绩效表，生成包含2024年第4季度绩效的综合Excel文件
"""

import pandas as pd
from ...shared import get_project_path
from datetime import datetime

def merge_employee_data():
    """
    合并员工基本信息表和员工绩效表数据
    """
    # 使用 shared 的 get_project_path 获取脚本目录
    current_dir = get_project_path()

    # user_data 用于存放临时/生成文件（优先使用）
    user_data_dir = current_dir / "user_data"
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # 输入文件路径（优先从 user_data 读取）
    info_candidate = user_data_dir / "员工基本信息表.xlsx"
    perf_candidate = user_data_dir / "员工绩效表.xlsx"

    employee_info_file = info_candidate if info_candidate.exists() else (current_dir / "员工基本信息表.xlsx")
    employee_performance_file = perf_candidate if perf_candidate.exists() else (current_dir / "员工绩效表.xlsx")

    # 输出文件路径（写入 user_data）
    merged_file = user_data_dir / "员工综合信息表_2024Q4.xlsx"
    
    print("=" * 80)
    print("员工基本信息与绩效数据合并程序")
    print("=" * 80)
    
    # 检查输入文件是否存在
    if not employee_info_file.exists():
        print(f"❌ 错误：找不到文件 {employee_info_file}")
        print("请确保员工基本信息表.xlsx文件存在")
        return None
    
    if not employee_performance_file.exists():
        print(f"❌ 错误：找不到文件 {employee_performance_file}")
        print("请确保员工绩效表.xlsx文件存在")
        return None
    
    try:
        # 读取员工基本信息表
        print("📊 正在读取员工基本信息表...")
        info_df = pd.read_excel(employee_info_file)
        print(f"   成功读取员工基本信息 {len(info_df)} 条记录")
        
        # 读取员工绩效表
        print("📈 正在读取员工绩效表...")
        performance_df = pd.read_excel(employee_performance_file)
        print(f"   成功读取员工绩效信息 {len(performance_df)} 条记录")
        
        # 显示数据概览
        print("\n📋 数据概览：")
        print(f"   员工基本信息表列数：{info_df.shape[1]}")
        print(f"   员工绩效表列数：{performance_df.shape[1]}")
        print(f"   员工基本信息表列名：{list(info_df.columns)}")
        print(f"   员工绩效表列名：{list(performance_df.columns)}")
        
        # 基于员工编号进行左连接合并
        print("\n🔄 正在合并数据...")
        merged_df = pd.merge(
            info_df, 
            performance_df, 
            on='员工编号', 
            how='left',
            suffixes=('', '_绩效')
        )
        
        # 处理重复列名（如姓名列）
        if '姓名_绩效' in merged_df.columns:
            # 检查姓名是否一致
            name_mismatch = merged_df[merged_df['姓名'] != merged_df['姓名_绩效']]
            if not name_mismatch.empty:
                print("⚠️  警告：发现姓名不匹配的记录：")
                for idx, row in name_mismatch.iterrows():
                    print(f"   员工编号 {row['员工编号']}: {row['姓名']} vs {row['姓名_绩效']}")
            
            # 删除重复的姓名列
            merged_df = merged_df.drop('姓名_绩效', axis=1)
        
        # 重新排列列的顺序，将绩效相关列放在最后
        basic_columns = list(info_df.columns)
        performance_columns = [col for col in performance_df.columns if col not in ['员工编号', '姓名']]
        
        # 构建最终列顺序
        final_columns = basic_columns + performance_columns
        merged_df = merged_df[final_columns]
        
        # 添加合并时间戳
        merged_df['数据合并时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"✅ 数据合并完成！共合并 {len(merged_df)} 条记录")
        
        # 显示合并后的统计信息
        print("\n📊 合并后数据统计：")
        print(f"   总记录数：{len(merged_df)}")
        print(f"   总列数：{merged_df.shape[1]}")
        
        # 统计有绩效数据的员工数量
        has_performance = merged_df['综合得分'].notna().sum()
        print(f"   有绩效数据的员工：{has_performance} 人")
        print(f"   无绩效数据的员工：{len(merged_df) - has_performance} 人")
        
        # 显示前5行合并后的数据
        print("\n📋 合并后数据前5行预览：")
        print("-" * 100)
        display_columns = ['员工编号', '姓名', '部门', '职位', '薪资', '综合得分', '绩效等级']
        available_columns = [col for col in display_columns if col in merged_df.columns]
        print(merged_df[available_columns].head().to_string(index=True))
        
        # 保存合并后的数据到新Excel文件
        print(f"\n💾 正在保存合并数据到 {merged_file.name}...")
        
        # 使用ExcelWriter进行更精细的格式控制
        with pd.ExcelWriter(merged_file, engine='openpyxl') as writer:
            # 保存主数据表
            merged_df.to_excel(writer, sheet_name='员工综合信息', index=False)
            
            # 创建数据统计表
            create_summary_sheet(writer, merged_df, info_df, performance_df)
            
            # 创建绩效分析表
            create_performance_analysis_sheet(writer, merged_df)
        
        print(f"✅ 文件保存成功：{merged_file}")
        
        return merged_df
        
    except Exception as e:
        print(f"❌ 合并过程中发生错误：{e}")
        import traceback
        traceback.print_exc()
        return None

def create_summary_sheet(writer, merged_df, info_df, performance_df):
    """
    创建数据统计摘要表
    """
    summary_data = {
        '统计项目': [
            '员工基本信息总数',
            '员工绩效记录总数', 
            '合并后总记录数',
            '有绩效数据员工数',
            '无绩效数据员工数',
            '平均综合得分',
            '最高综合得分',
            '最低综合得分',
            'A级绩效员工数',
            'B级绩效员工数',
            'C级绩效员工数'
        ],
        '数值': [
            len(info_df),
            len(performance_df),
            len(merged_df),
            merged_df['综合得分'].notna().sum(),
            merged_df['综合得分'].isna().sum(),
            round(merged_df['综合得分'].mean(), 2) if merged_df['综合得分'].notna().any() else 0,
            merged_df['综合得分'].max() if merged_df['综合得分'].notna().any() else 0,
            merged_df['综合得分'].min() if merged_df['综合得分'].notna().any() else 0,
            len(merged_df[merged_df['绩效等级'] == 'A']) if '绩效等级' in merged_df.columns else 0,
            len(merged_df[merged_df['绩效等级'] == 'B']) if '绩效等级' in merged_df.columns else 0,
            len(merged_df[merged_df['绩效等级'] == 'C']) if '绩效等级' in merged_df.columns else 0
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='数据统计摘要', index=False)

def create_performance_analysis_sheet(writer, merged_df):
    """
    创建绩效分析表
    """
    if '部门' in merged_df.columns and '综合得分' in merged_df.columns:
        # 按部门统计绩效
        dept_performance = merged_df.groupby('部门').agg({
            '综合得分': ['count', 'mean', 'max', 'min'],
            '员工编号': 'count'
        }).round(2)
        
        # 重命名列
        dept_performance.columns = ['绩效记录数', '平均得分', '最高得分', '最低得分', '员工总数']
        dept_performance = dept_performance.reset_index()
        
        dept_performance.to_excel(writer, sheet_name='部门绩效分析', index=False)

def main():
    """
    主函数
    """
    print("🚀 开始执行员工数据合并程序...")
    
    # 执行合并操作
    result = merge_employee_data()
    
    if result is not None:
        print("\n" + "=" * 80)
        print("✅ 程序执行成功！")
        print("📁 生成的文件：员工综合信息表_2024Q4.xlsx")
        print("📊 包含工作表：")
        print("   1. 员工综合信息 - 完整的合并数据")
        print("   2. 数据统计摘要 - 数据统计信息")
        print("   3. 部门绩效分析 - 按部门的绩效分析")
        print("=" * 80)
    else:
        print("\n❌ 程序执行失败，请检查错误信息并重试。")

if __name__ == "__main__":
    main()