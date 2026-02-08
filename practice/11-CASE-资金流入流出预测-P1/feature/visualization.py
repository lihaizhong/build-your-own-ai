#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt
from ...shared import get_project_path

# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def create_ascii_chart(data_points, width=80, height=20):
    """创建ASCII折线图"""
    if len(data_points) < 2:
        return "数据点不足，无法生成图表"
    
    # 找到最大和最小值
    max_val = max(data_points)
    min_val = min(data_points)
    val_range = max_val - min_val if max_val != min_val else 1
    
    # 计算每个数据点在图表中的位置
    chart_lines = []
    for i in range(height, -1, -1):
        line = ""
        threshold = min_val + (i * val_range / height)
        
        for j, val in enumerate(data_points):
            if j == 0:
                # 第一个点，用左边界标记
                if val >= threshold:
                    line += "●"
                else:
                    line += " "
            else:
                # 检查是否需要连接线
                prev_threshold = min_val + ((i + 1) * val_range / height)
                current_threshold = threshold
                prev_val = data_points[j-1]
                
                # 判断是否绘制连接线
                if (prev_val >= prev_threshold and val >= current_threshold) or \
                   (prev_val < prev_threshold and val < current_threshold):
                    if val >= threshold:
                        line += "─●"
                    else:
                        line += "  "
                else:
                    # 检查是否跨越了当前阈值线
                    if prev_val >= threshold and val < threshold:
                        line += "┘ "
                    elif prev_val < threshold and val >= threshold:
                        line += "┐ "
                    else:
                        line += "  "
        
        # 添加右侧数值标签（简化版）
        if i % 4 == 0:  # 每4行显示一次数值
            value = min_val + (i * val_range / height)
            if value >= 1e9:
                label = f"{value/1e9:.1f}B"
            elif value >= 1e6:
                label = f"{value/1e6:.1f}M"
            else:
                label = f"{value/1e3:.0f}K"
            line = f"{label:<8}{line}"
        else:
            line = f"         {line}"
        
        chart_lines.append(line)
    
    # 添加时间轴
    time_axis = "         "
    for i in range(0, len(data_points), max(1, len(data_points)//10)):
        time_axis += f"│{i:3d}"
    chart_lines.append(time_axis)
    
    return "\n".join(chart_lines)


def generate_chart_report():
    """生成图表报告"""
    csv_file = get_project_path('..', 'user_data', 'daily_summary.csv')
    
    try:
        # 读取CSV数据（无表头格式）
        dates = []
        purchases = []
        redeems = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                dates.append(row[0])  # date
                purchases.append(float(row[1]))  # purchase_amt
                redeems.append(float(row[2]))  # redeem_amt
        
        # 计算净流入
        net_flows = [p - r for p, r in zip(purchases, redeems)]
        
        print("=== 每日申购赎回金额趋势图表 ===\n")
        
        # 生成申购总额趋势图（只显示每7天的数据点以简化显示）
        step = max(1, len(purchases) // 50)
        sample_purchases = purchases[::step]
        sample_dates = dates[::step]
        
        print("每日申购总额趋势 (每7天取样):")
        print("-" * 80)
        # 显示部分日期用于参考
        print("日期样本:", " ".join([f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in sample_dates[:5]]), "...")
        print(create_ascii_chart(sample_purchases))
        print("\n")
        
        # 生成赎回总额趋势图
        sample_redeems = redeems[::step]
        print("每日赎回总额趋势 (每7天取样):")
        print("-" * 80)
        print(create_ascii_chart(sample_redeems))
        print("\n")
        
        # 生成净流入趋势图
        sample_net_flows = net_flows[::step]
        print("每日净流入趋势 (申购-赎回) (每7天取样):")
        print("-" * 80)
        print(create_ascii_chart(sample_net_flows))
        print("\n")
        
        # 生成更详细的关键指标表格
        print("=== 关键指标汇总 ===")
        print("时间范围: 2013年7月1日 - 2014年8月31日")
        print("总天数: 427天")
        print()
        
        # 按月份统计
        monthly_data = {}
        for i, date in enumerate(dates):
            year_month = f"{date[:4]}-{date[4:6]}"  # 从YYYYMMDD提取YYYY-MM
            if year_month not in monthly_data:
                monthly_data[year_month] = {'purchase': 0, 'redeem': 0}
            monthly_data[year_month]['purchase'] += purchases[i]
            monthly_data[year_month]['redeem'] += redeems[i]
        
        print("月份         申购总额(M)    赎回总额(M)    净流入(M)")
        print("-" * 50)
        for month, data in sorted(monthly_data.items()):
            net = data['purchase'] - data['redeem']
            print(f"{month}      {data['purchase']/1e6:8.1f}      {data['redeem']/1e6:8.1f}      {net/1e6:8.1f}")
        
        print()
        print("=== 重要发现 ===")
        
        # 找到极值点
        max_purchase_idx = purchases.index(max(purchases))
        max_redeem_idx = redeems.index(max(redeems))
        max_net_idx = net_flows.index(max(net_flows))
        min_net_idx = net_flows.index(min(net_flows))
        
        # 格式化日期显示（YYYY-MM-DD格式用于显示）
        max_purchase_date = f"{dates[max_purchase_idx][:4]}-{dates[max_purchase_idx][4:6]}-{dates[max_purchase_idx][6:8]}"
        max_redeem_date = f"{dates[max_redeem_idx][:4]}-{dates[max_redeem_idx][4:6]}-{dates[max_redeem_idx][6:8]}"
        max_net_date = f"{dates[max_net_idx][:4]}-{dates[max_net_idx][4:6]}-{dates[max_net_idx][6:8]}"
        min_net_date = f"{dates[min_net_idx][:4]}-{dates[min_net_idx][4:6]}-{dates[min_net_idx][6:8]}"
        
        print(f"• 最大单日申购额: {max(purchases)/1e6:.1f}M ({max_purchase_date})")
        print(f"• 最大单日赎回额: {max(redeems)/1e6:.1f}M ({max_redeem_date})")
        print(f"• 最大单日净流入: {max(net_flows)/1e6:.1f}M ({max_net_date})")
        print(f"• 最大单日净流出: {min(net_flows)/1e6:.1f}M ({min_net_date})")
        
        # 计算流入流出天数
        positive_days = sum(1 for flow in net_flows if flow > 0)
        negative_days = sum(1 for flow in net_flows if flow < 0)
        
        print(f"• 净流入天数: {positive_days}天 ({positive_days/len(net_flows)*100:.1f}%)")
        print(f"• 净流出天数: {negative_days}天 ({negative_days/len(net_flows)*100:.1f}%)")
        
        # 计算总金额
        total_purchase = sum(purchases)
        total_redeem = sum(redeems)
        total_net = total_purchase - total_redeem
        
        print(f"• 累计申购总额: {total_purchase/1e9:.2f}B")
        print(f"• 累计赎回总额: {total_redeem/1e9:.2f}B")
        print(f"• 累计净流入: {total_net/1e9:.2f}B")
        
        # 年度对比
        print(f"\n=== 年度表现对比 ===")
        print("2013年下半年 vs 2014年上半年:")
        
        # 2013年下半年 (7-12月)
        h2_2013_purchase = sum(purchases[:184])  # 约半年数据
        h2_2013_redeem = sum(redeems[:184])
        
        # 2014年上半年 (1-6月)
        h1_2014_purchase = sum(purchases[184:365])
        h1_2014_redeem = sum(redeems[184:365])
        
        print(f"2013年H2 申购: {h2_2013_purchase/1e9:.2f}B, 赎回: {h2_2013_redeem/1e9:.2f}B")
        print(f"2014年H1 申购: {h1_2014_purchase/1e9:.2f}B, 赎回: {h1_2014_redeem/1e9:.2f}B")
        
        return True
        
    except Exception as e:
        print(f"生成图表报告时发生错误: {e}")
        return False


if __name__ == "__main__":
    generate_chart_report()
