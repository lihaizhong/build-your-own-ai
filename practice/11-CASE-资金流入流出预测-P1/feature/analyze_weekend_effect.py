#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析周末效应对资金流入流出的影响
检查工作日vs周末的交易模式差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from ...shared import get_project_path

warnings.filterwarnings('ignore')


def analyze_weekend_effect():
    """分析周末效应"""
    print("=== 分析周末效应 ===")
    
    # 读取每日汇总数据
    data_file = get_project_path('..', 'user_data', 'daily_summary.csv')
    df = pd.read_csv(data_file, header=None, names=['date', 'purchase', 'redeem'])
    
    # 转换日期格式
    df['ds'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['weekday'] = df['ds'].dt.dayofweek  # 0=周一, 6=周日
    df['is_weekend'] = df['weekday'].isin([5, 6])  # 周六和周日
    df['day_name'] = df['ds'].dt.day_name()
    
    print(f"数据时间范围: {df['ds'].min()} 至 {df['ds'].max()}")
    print(f"总数据量: {len(df)} 天")
    
    # 1. 按星期分析
    weekday_stats = df.groupby('day_name')[['purchase', 'redeem']].agg(['mean', 'std', 'count']).round(0)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_stats = weekday_stats.reindex(weekday_order)
    
    print(f"\n📊 按星期统计 (单位: 元):")
    print(f"{'星期':<8} {'申购均值':<12} {'申购标准差':<12} {'赎回均值':<12} {'赎回标准差':<12} {'样本数':<8}")
    print("-" * 70)
    
    for day in weekday_order:
        if day in weekday_stats.index:
            purchase_mean = weekday_stats.loc[day, ('purchase', 'mean')]
            purchase_std = weekday_stats.loc[day, ('purchase', 'std')]
            redeem_mean = weekday_stats.loc[day, ('redeem', 'mean')]
            redeem_std = weekday_stats.loc[day, ('redeem', 'std')]
            count = weekday_stats.loc[day, ('purchase', 'count')]
            
            print(f"{day:<8} {purchase_mean:>10,.0f} {purchase_std:>10,.0f} {redeem_mean:>10,.0f} {redeem_std:>10,.0f} {count:>6.0f}")
    
    # 2. 工作日 vs 周末对比
    workday_weekend = df.groupby('is_weekend')[['purchase', 'redeem']].agg(['mean', 'std', 'count'])
    workday_weekend.index = ['工作日', '周末']
    
    print(f"\n🏢 工作日 vs 周末对比:")
    print(f"{'类型':<6} {'申购均值':<12} {'申购标准差':<12} {'赎回均值':<12} {'赎回标准差':<12} {'样本数':<8}")
    print("-" * 70)
    
    for idx in workday_weekend.index:
        purchase_mean = workday_weekend.loc[idx, ('purchase', 'mean')]
        purchase_std = workday_weekend.loc[idx, ('purchase', 'std')]
        redeem_mean = workday_weekend.loc[idx, ('redeem', 'mean')]
        redeem_std = workday_weekend.loc[idx, ('redeem', 'std')]
        count = workday_weekend.loc[idx, ('purchase', 'count')]
        
        print(f"{idx:<6} {purchase_mean:>10,.0f} {purchase_std:>10,.0f} {redeem_mean:>10,.0f} {redeem_std:>10,.0f} {count:>6.0f}")
    
    # 3. 计算周末效应
    workday_purchase = df[~df['is_weekend']]['purchase'].mean()
    weekend_purchase = df[df['is_weekend']]['purchase'].mean()
    workday_redeem = df[~df['is_weekend']]['redeem'].mean()
    weekend_redeem = df[df['is_weekend']]['redeem'].mean()
    
    purchase_weekend_effect = ((weekend_purchase - workday_purchase) / workday_purchase) * 100
    redeem_weekend_effect = ((weekend_redeem - workday_redeem) / workday_redeem) * 100
    
    print(f"\n🎯 周末效应分析:")
    print(f"- 申购: 周末比工作日 {purchase_weekend_effect:+.1f}%")
    print(f"- 赎回: 周末比工作日 {redeem_weekend_effect:+.1f}%")
    
    # 4. 统计显著性检验
    from scipy import stats
    
    workday_purchase_data = df[~df['is_weekend']]['purchase']
    weekend_purchase_data = df[df['is_weekend']]['purchase']
    workday_redeem_data = df[~df['is_weekend']]['redeem']
    weekend_redeem_data = df[df['is_weekend']]['redeem']
    
    # t检验
    purchase_tstat, purchase_pvalue = stats.ttest_ind(workday_purchase_data, weekend_purchase_data)
    redeem_tstat, redeem_pvalue = stats.ttest_ind(workday_redeem_data, weekend_redeem_data)
    
    print(f"\n📈 统计显著性检验 (t-test):")
    print(f"- 申购: t-statistic={purchase_tstat:.3f}, p-value={purchase_pvalue:.4f}")
    print(f"- 赎回: t-statistic={redeem_tstat:.3f}, p-value={redeem_pvalue:.4f}")
    
    if purchase_pvalue < 0.05:
        print("✅ 申购的周末效应在统计上显著")
    else:
        print("⚠️  申购的周末效应在统计上不显著")
        
    if redeem_pvalue < 0.05:
        print("✅ 赎回的周末效应在统计上显著")
    else:
        print("⚠️  赎回的周末效应在统计上不显著")
    
    # 5. 创建可视化
    create_weekend_visualization(df, weekday_stats)
    
    return {
        'purchase_weekend_effect': purchase_weekend_effect,
        'redeem_weekend_effect': redeem_weekend_effect,
        'purchase_pvalue': purchase_pvalue,
        'redeem_pvalue': redeem_pvalue,
        'weekday_stats': weekday_stats,
        'workday_weekend': workday_weekend
    }


def create_weekend_visualization(df, weekday_stats):
    """创建周末效应可视化"""
    print(f"\n=== 生成周末效应可视化图表 ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('周末效应对资金流入流出的影响分析', fontsize=16, fontweight='bold')
    
    # 1. 星期趋势图
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekdays_chinese = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    
    purchase_by_weekday = df.groupby('day_name')['purchase'].mean().reindex(weekday_order)
    redeem_by_weekday = df.groupby('day_name')['redeem'].mean().reindex(weekday_order)
    
    ax1 = axes[0, 0]
    x_pos = range(len(weekdays_chinese))
    ax1.bar(x_pos, purchase_by_weekday.values / 1e8, alpha=0.7, color='lightblue', label='申购')
    ax1.plot(x_pos, redeem_by_weekday.values / 1e8, 'ro-', linewidth=2, label='赎回')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(weekdays_chinese)
    ax1.set_title('各星期资金流动均值')
    ax1.set_ylabel('金额 (亿元)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 工作日 vs 周末对比
    workday_data = df[~df['is_weekend']]
    weekend_data = df[df['is_weekend']]
    
    ax2 = axes[0, 1]
    categories = ['申购', '赎回']
    workday_values = [workday_data['purchase'].mean() / 1e8, workday_data['redeem'].mean() / 1e8]
    weekend_values = [weekend_data['purchase'].mean() / 1e8, weekend_data['redeem'].mean() / 1e8]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    ax2.bar(x_pos - width/2, workday_values, width, label='工作日', alpha=0.7, color='lightgreen')
    ax2.bar(x_pos + width/2, weekend_values, width, label='周末', alpha=0.7, color='orange')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.set_title('工作日 vs 周末对比')
    ax2.set_ylabel('金额 (亿元)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 波动性分析
    ax3 = axes[1, 0]
    purchase_std_by_weekday = df.groupby('day_name')['purchase'].std().reindex(weekday_order)
    redeem_std_by_weekday = df.groupby('day_name')['redeem'].std().reindex(weekday_order)
    
    x_pos_std = range(len(weekdays_chinese))
    ax3.bar([x - 0.2 for x in x_pos_std], purchase_std_by_weekday.values / 1e8, 
            width=0.4, alpha=0.7, color='lightblue', label='申购标准差')
    ax3.bar([x + 0.2 for x in x_pos_std], redeem_std_by_weekday.values / 1e8, 
            width=0.4, alpha=0.7, color='orange', label='赎回标准差')
    ax3.set_xticks(x_pos_std)
    ax3.set_xticklabels(weekdays_chinese)
    ax3.set_title('各星期资金流动波动性')
    ax3.set_ylabel('标准差 (亿元)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 箱型图
    ax4 = axes[1, 1]
    
    # 创建箱型图数据
    workday_purchase = df[~df['is_weekend']]['purchase'] / 1e8
    weekend_purchase = df[df['is_weekend']]['purchase'] / 1e8
    workday_redeem = df[~df['is_weekend']]['redeem'] / 1e8
    weekend_redeem = df[df['is_weekend']]['redeem'] / 1e8
    
    box_data = [workday_purchase, weekend_purchase, workday_redeem, weekend_redeem]
    box_labels = ['申购-工作日', '申购-周末', '赎回-工作日', '赎回-周末']
    
    ax4.boxplot(box_data, labels=box_labels)
    ax4.set_title('工作日 vs 周末分布对比')
    ax4.set_ylabel('金额 (亿元)')
    ax4.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    chart_file = get_project_path('..', 'user_data', 'weekend_effect_analysis.png')
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"周末效应分析图表已保存到: {chart_file}")


def main():
    """主函数"""
    print("=== 周末效应分析工具 ===\n")
    
    try:
        results = analyze_weekend_effect()
        
        print(f"\n=== 分析结论 ===")
        
        if results['purchase_pvalue'] < 0.05:
            print("✅ 申购具有显著周末效应")
        else:
            print("⚠️  申购周末效应不显著")
            
        if results['redeem_pvalue'] < 0.05:
            print("✅ 赎回具有显著周末效应")
        else:
            print("⚠️  赎回周末效应不显著")
        
        print(f"\n💡 建议:")
        if results['purchase_pvalue'] < 0.05 or results['redeem_pvalue'] < 0.05:
            print("- 建议在Prophet模型中添加显式的周末节假日")
            print("- 周末和工作日可能存在不同的交易模式")
            print("- 考虑添加周末效应对模型预测的提升")
        else:
            print("- 当前Prophet的weekly_seasonality可能已经足够")
            print("- 可以考虑不添加显式周末节假日")
            print("- 或者尝试其他特征工程方法")
        
        return True
        
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()