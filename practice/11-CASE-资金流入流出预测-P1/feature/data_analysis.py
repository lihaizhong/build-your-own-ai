#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from datetime import datetime
from collections import defaultdict
from ...shared import get_project_path


def analyze_daily_flow():
    """
    分析每日申购总额和赎回总额的趋势（使用标准库实现）
    """
    data_file = get_project_path('..', 'data', 'user_balance_table.csv')
    
    print("正在读取用户余额数据...")
    
    try:
        # 使用csv模块读取数据
        daily_data = defaultdict(lambda: {'purchase': 0, 'redeem': 0})
        
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            
            for row in reader:
                count += 1
                if count % 10000 == 0:
                    print(f"已处理 {count:,} 条记录...")
                
                date = row['report_date']
                try:
                    purchase_amt = float(row['total_purchase_amt']) if row['total_purchase_amt'] else 0
                    redeem_amt = float(row['total_redeem_amt']) if row['total_redeem_amt'] else 0
                    
                    daily_data[date]['purchase'] += purchase_amt
                    daily_data[date]['redeem'] += redeem_amt
                except (ValueError, KeyError):
                    continue
        
        print(f"数据处理完成，共处理 {count:,} 条记录")
        
        # 转换数据并排序
        daily_summary = []
        for date, data in daily_data.items():
            try:
                # 将YYYYMMDD格式转换为日期
                date_obj = datetime.strptime(date, '%Y%m%d')
                daily_summary.append({
                    'date': date,
                    'date_obj': date_obj,
                    'purchase': data['purchase'],
                    'redeem': data['redeem'],
                    'net_flow': data['purchase'] - data['redeem']
                })
            except ValueError:
                continue
        
        # 按日期排序
        daily_summary.sort(key=lambda x: x['date'])
        
        print(f"\n=== 分析结果 ===")
        print(f"分析时间范围: {daily_summary[0]['date'][:4]}-{daily_summary[0]['date'][4:6]}-{daily_summary[0]['date'][6:8]} 至 {daily_summary[-1]['date'][:4]}-{daily_summary[-1]['date'][4:6]}-{daily_summary[-1]['date'][6:8]}")
        print(f"总计分析天数: {len(daily_summary)} 天")
        
        total_purchase = sum(item['purchase'] for item in daily_summary)
        total_redeem = sum(item['redeem'] for item in daily_summary)
        
        print(f"累计申购总额: ¥{total_purchase:,.0f}")
        print(f"累计赎回总额: ¥{total_redeem:,.0f}")
        print(f"平均每日申购额: ¥{total_purchase/len(daily_summary):,.0f}")
        print(f"平均每日赎回额: ¥{total_redeem/len(daily_summary):,.0f}")
        
        # 保存汇总数据到CSV（考试格式：无表头，只包含date, purchase_amt, redeem_amt）
        output_file = get_project_path('..', 'user_data', 'daily_summary.csv')
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for item in daily_summary:
                # 考试格式：YYYYMMDD（无连字符）
                formatted_date = item['date']  # 已经是YYYYMMDD格式
                writer.writerow([formatted_date, f"{item['purchase']:.0f}", f"{item['redeem']:.0f}"])
        
        print(f"\n数据已保存到: {output_file}")
        
        # 详细统计
        max_purchase_idx = max(range(len(daily_summary)), key=lambda i: daily_summary[i]['purchase'])
        max_redeem_idx = max(range(len(daily_summary)), key=lambda i: daily_summary[i]['redeem'])
        max_net_flow_idx = max(range(len(daily_summary)), key=lambda i: daily_summary[i]['net_flow'])
        min_net_flow_idx = min(range(len(daily_summary)), key=lambda i: daily_summary[i]['net_flow'])
        
        print(f"\n=== 详细统计信息 ===")
        print(f"最大单日申购额: ¥{daily_summary[max_purchase_idx]['purchase']:,.0f} (日期: {daily_summary[max_purchase_idx]['date'][:4]}-{daily_summary[max_purchase_idx]['date'][4:6]}-{daily_summary[max_purchase_idx]['date'][6:8]})")
        print(f"最大单日赎回额: ¥{daily_summary[max_redeem_idx]['redeem']:,.0f} (日期: {daily_summary[max_redeem_idx]['date'][:4]}-{daily_summary[max_redeem_idx]['date'][4:6]}-{daily_summary[max_redeem_idx]['date'][6:8]})")
        print(f"最大单日净流入: ¥{daily_summary[max_net_flow_idx]['net_flow']:,.0f} (日期: {daily_summary[max_net_flow_idx]['date'][:4]}-{daily_summary[max_net_flow_idx]['date'][4:6]}-{daily_summary[max_net_flow_idx]['date'][6:8]})")
        print(f"最大单日净流出: ¥{daily_summary[min_net_flow_idx]['net_flow']:,.0f} (日期: {daily_summary[min_net_flow_idx]['date'][:4]}-{daily_summary[min_net_flow_idx]['date'][4:6]}-{daily_summary[min_net_flow_idx]['date'][6:8]})")
        
        # 计算流入天数和流出天数
        positive_days = sum(1 for item in daily_summary if item['net_flow'] > 0)
        negative_days = sum(1 for item in daily_summary if item['net_flow'] < 0)
        print(f"净流入天数: {positive_days} 天 ({positive_days/len(daily_summary)*100:.1f}%)")
        print(f"净流出天数: {negative_days} 天 ({negative_days/len(daily_summary)*100:.1f}%)")
        
        # 生成简单的ASCII图表数据（用于后续可视化）
        chart_data = []
        max_value = max(max(item['purchase'], item['redeem']) for item in daily_summary)
        
        print(f"\n=== 数据趋势概览 ===")
        print("日期           申购总额        赎回总额        净流入")
        print("-" * 60)
        
        for item in daily_summary[::max(1, len(daily_summary)//30)]:  # 显示约30个点
            formatted_date = f"{item['date'][:4]}-{item['date'][4:6]}-{item['date'][6:8]}"
            purchase_str = f"¥{item['purchase']/1e6:.1f}M" if item['purchase'] >= 1e6 else f"¥{item['purchase']/1e3:.0f}K"
            redeem_str = f"¥{item['redeem']/1e6:.1f}M" if item['redeem'] >= 1e6 else f"¥{item['redeem']/1e3:.0f}K"
            net_flow_str = f"¥{item['net_flow']/1e6:.1f}M" if item['net_flow'] >= 1e6 else f"¥{item['net_flow']/1e3:.0f}K"
            
            print(f"{formatted_date:<12} {purchase_str:<12} {redeem_str:<12} {net_flow_str}")
            chart_data.append({
                'date': formatted_date,
                'purchase': item['purchase'],
                'redeem': item['redeem'],
                'net_flow': item['net_flow']
            })
        
        # 保存图表数据
        chart_file = get_project_path('..', 'user_data', 'chart_data.json')
        with open(chart_file, 'w', encoding='utf-8') as f:
            json.dump(chart_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n图表数据已保存到: {chart_file}")
        print(f"\n可用工具:")
        print(f"1. 查看CSV文件: {output_file}")
        print(f"2. 使用Excel打开CSV文件进行图表制作")
        print(f"3. 或使用在线工具导入JSON数据制作交互式图表")
        
        return daily_summary
        
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        return None


if __name__ == "__main__":
    print("=== 每日申购赎回金额分析 ===")
    analyze_daily_flow()