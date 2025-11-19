#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stationarity Analysis Module
Performs Augmented Dickey-Fuller (ADF) tests on time series data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


def get_project_path(*paths):
    """获取项目路径的统一方法"""
    import os
    try:
        return os.path.join(os.path.dirname(__file__), *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)


def load_daily_data():
    """加载每日汇总数据"""
    csv_file = get_project_path('..', 'user_data', 'daily_summary.csv')
    
    try:
        # 读取CSV数据（无表头格式）
        dates = []
        purchases = []
        redeems = []
        
        df = pd.read_csv(csv_file, header=None, names=['date', 'purchase_amt', 'redeem_amt'])
        
        print(f"成功加载数据，共 {len(df)} 行记录")
        print(f"数据时间范围: {df['date'].iloc[0]} 到 {df['date'].iloc[-1]}")
        
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None


def adf_test(series, series_name):
    """执行ADF平稳性检验"""
    result = adfuller(series, autolag='AIC')
    
    print(f"\n=== {series_name} ADF检验结果 ===")
    print(f"ADF统计量: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"使用的滞后期: {result[2]}")
    print(f"观测值数量: {result[3]}")
    
    print("\n临界值:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.6f}")
    
    # 判断平稳性
    if result[1] <= 0.05:
        print(f"\n结论: {series_name} 是平稳的 (拒绝原假设，p-value <= 0.05)")
        return True
    else:
        print(f"\n结论: {series_name} 是非平稳的 (接受原假设，p-value > 0.05)")
        return False


def filter_data_by_date(df, start_date, end_date):
    """按日期范围筛选数据"""
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    # 将date列转换为整数以便比较
    df['date_int'] = df['date'].astype(int)
    start_int = int(start_str)
    end_int = int(end_str)
    
    filtered_df = df[(df['date_int'] >= start_int) & (df['date_int'] <= end_int)].copy()
    filtered_df['date'] = pd.to_datetime(filtered_df['date'], format='%Y%m%d')
    filtered_df.set_index('date', inplace=True)
    
    print(f"\n筛选后数据范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    print(f"筛选后数据量: {len(filtered_df)} 天")
    
    return filtered_df


def create_stationarity_plots(filtered_df, output_dir):
    """创建平稳性分析图表"""
    # 设置中文字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 使用中文标签
    title_text = '时间序列平稳性分析 (2014-03-01 到 2014-08-31)'
    purchase_title = '每日申购总额时间序列'
    redeem_title = '每日赎回总额时间序列'
    purchase_ylabel = '申购金额'
    redeem_ylabel = '赎回金额'
    date_xlabel = '日期'
    purchase_dist_title = '申购金额分布'
    redeem_dist_title = '赎回金额分布'
    amount_xlabel = '申购金额'
    amount_xlabel2 = '赎回金额'
    freq_ylabel = '频次'
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title_text, fontsize=16)
    
    # 原始时间序列
    axes[0, 0].plot(filtered_df.index, filtered_df['purchase_amt'], 'b-', linewidth=1)
    axes[0, 0].set_title(purchase_title)
    axes[0, 0].set_ylabel(purchase_ylabel)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 0].plot(filtered_df.index, filtered_df['redeem_amt'], 'r-', linewidth=1)
    axes[1, 0].set_title(redeem_title)
    axes[1, 0].set_ylabel(redeem_ylabel)
    axes[1, 0].set_xlabel(date_xlabel)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 分布图
    axes[0, 1].hist(filtered_df['purchase_amt'], bins=30, alpha=0.7, color='blue')
    axes[0, 1].set_title(purchase_dist_title)
    axes[0, 1].set_xlabel(amount_xlabel)
    axes[0, 1].set_ylabel(freq_ylabel)
    
    axes[1, 1].hist(filtered_df['redeem_amt'], bins=30, alpha=0.7, color='red')
    axes[1, 1].set_title(redeem_dist_title)
    axes[1, 1].set_xlabel(amount_xlabel2)
    axes[1, 1].set_ylabel(freq_ylabel)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = f"{output_dir}/stationarity_analysis_{filtered_df.index[0].strftime('%Y%m%d')}_{filtered_df.index[-1].strftime('%Y%m%d')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"平稳性分析图表已保存到: {plot_file}")
    return plot_file


def generate_descriptive_stats(filtered_df):
    """生成描述性统计"""
    print(f"\n=== 描述性统计 (2014-03-01 到 2014-08-31) ===")
    
    stats_data = []
    for column in ['purchase_amt', 'redeem_amt']:
        series = filtered_df[column]
        stats = {
            '指标': '申购总额' if column == 'purchase_amt' else '赎回总额',
            '均值': f"{series.mean():,.0f}",
            '标准差': f"{series.std():,.0f}",
            '最小值': f"{series.min():,.0f}",
            '最大值': f"{series.max():,.0f}",
            '中位数': f"{series.median():,.0f}",
            '变异系数': f"{series.std()/series.mean():.3f}"
        }
        stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    # 保存统计结果
    stats_file = get_project_path('..', 'user_data', 'stationarity_descriptive_stats.csv')
    stats_df.to_csv(stats_file, index=False, encoding='utf-8')
    print(f"\n描述性统计已保存到: {stats_file}")
    
    return stats_df


def save_filtered_data(filtered_df):
    """保存筛选后的数据"""
    output_file = get_project_path('..', 'user_data', 
                                  f"filtered_data_{filtered_df.index[0].strftime('%Y%m%d')}_{filtered_df.index[-1].strftime('%Y%m%d')}.csv")
    
    # 保存时保留原始日期格式
    output_df = filtered_df.reset_index()
    output_df['date'] = output_df['date'].dt.strftime('%Y%m%d')
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"筛选后数据已保存到: {output_file}")
    return output_file


def perform_stationarity_analysis():
    """执行完整的平稳性分析"""
    print("=== 时间序列平稳性分析 ===")
    
    # 加载数据
    df = load_daily_data()
    if df is None:
        return
    
    # 筛选数据范围：2014-03-01 到 2014-08-31
    start_date = pd.to_datetime('2014-03-01')
    end_date = pd.to_datetime('2014-08-31')
    
    filtered_df = filter_data_by_date(df, start_date, end_date)
    
    # 保存筛选后的数据
    save_filtered_data(filtered_df)
    
    # 生成描述性统计
    generate_descriptive_stats(filtered_df)
    
    # 执行ADF检验
    purchase_stationary = adf_test(filtered_df['purchase_amt'], '申购总额')
    redeem_stationary = adf_test(filtered_df['redeem_amt'], '赎回总额')
    
    # 创建可视化图表
    output_dir = get_project_path('..', 'user_data')
    create_stationarity_plots(filtered_df, output_dir)
    
    # 如果赎回总额非平稳，进行差分分析
    if not redeem_stationary:
        print("\n由于赎回总额是非平稳的，现在进行差分分析...")
        diff_results = perform_differencing_analysis(filtered_df)
    else:
        diff_results = None
    
    # 总结分析结果
    print(f"\n=== 平稳性分析总结 ===")
    print(f"分析时间段: {filtered_df.index[0].strftime('%Y-%m-%d')} 到 {filtered_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"总天数: {len(filtered_df)} 天")
    print(f"申购总额 (total_purchase_amt): {'平稳' if purchase_stationary else '非平稳'}")
    print(f"赎回总额 (total_redeem_amt): {'平稳' if redeem_stationary else '非平稳'}")
    
    if not redeem_stationary:
        print(f"一阶差分后赎回总额: {'平稳' if diff_results['diff_stationary'] else '仍非平稳'}")
        if diff_results['diff_stationary']:
            print("建议: 对赎回总额进行一阶差分处理后可进行时间序列建模")
        else:
            print("建议: 可能需要更高阶差分或其他平稳化方法")
    
    return {
        'purchase_stationary': purchase_stationary,
        'redeem_stationary': redeem_stationary,
        'total_days': len(filtered_df),
        'date_range': f"{filtered_df.index[0].strftime('%Y-%m-%d')} 到 {filtered_df.index[-1].strftime('%Y-%m-%d')}",
        'differencing_results': diff_results
    }


def perform_differencing_analysis(filtered_df):
    """执行差分分析和ADF检验"""
    print(f"\n=== 差分平稳性分析 ===")
    
    # 对赎回总额进行一阶差分
    redeem_diff = filtered_df['redeem_amt'].diff().dropna()
    
    print(f"原始赎回总额ADF检验结果:")
    original_adf = adfuller(filtered_df['redeem_amt'].dropna())
    print(f"  ADF统计量: {original_adf[0]:.6f}")
    print(f"  p-value: {original_adf[1]:.6f}")
    print(f"  结论: {'平稳' if original_adf[1] <= 0.05 else '非平稳'}")
    
    # 对差分后的数据进行ADF检验
    print(f"\n赎回总额一阶差分后ADF检验结果:")
    diff_adf = adfuller(redeem_diff)
    print(f"  ADF统计量: {diff_adf[0]:.6f}")
    print(f"  p-value: {diff_adf[1]:.6f}")
    print(f"  使用的滞后期: {diff_adf[2]}")
    print(f"  观测值数量: {diff_adf[3]}")
    
    print("\n临界值:")
    for key, value in diff_adf[4].items():
        print(f"\t{key}: {value:.6f}")
    
    # 判断差分后的平稳性
    if diff_adf[1] <= 0.05:
        print(f"\n结论: 赎回总额一阶差分后是平稳的 (拒绝原假设，p-value <= 0.05)")
        diff_stationary = True
    else:
        print(f"\n结论: 赎回总额一阶差分后仍是非平稳的 (接受原假设，p-value > 0.05)")
        diff_stationary = False
    
    # 生成差分后的统计信息
    print(f"\n=== 差分后描述性统计 ===")
    diff_stats = {
        '均值': f"{redeem_diff.mean():,.0f}",
        '标准差': f"{redeem_diff.std():,.0f}",
        '最小值': f"{redeem_diff.min():,.0f}",
        '最大值': f"{redeem_diff.max():,.0f}",
        '中位数': f"{redeem_diff.median():,.0f}"
    }
    
    for key, value in diff_stats.items():
        print(f"{key}: {value}")
    
    # 保存差分数据
    diff_data = pd.DataFrame({
        'date': filtered_df.index[1:],  # 差分后减少一个观测值
        'redeem_diff': redeem_diff.values
    })
    
    diff_file = get_project_path('..', 'user_data', 
                                f"redeem_diff_{filtered_df.index[0].strftime('%Y%m%d')}_{filtered_df.index[-1].strftime('%Y%m%d')}.csv")
    diff_data.to_csv(diff_file, index=False, encoding='utf-8')
    print(f"\n差分数据已保存到: {diff_file}")
    
    # 创建差分后图表
    create_differencing_plots(filtered_df, redeem_diff)
    
    return {
        'original_adf': original_adf,
        'diff_adf': diff_adf,
        'diff_stationary': diff_stationary,
        'diff_data': redeem_diff
    }


def create_differencing_plots(original_df, diff_series):
    """创建差分分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('赎回总额差分分析 (2014-03-01 到 2014-08-31)', fontsize=16)
    
    # 原始赎回时间序列
    axes[0, 0].plot(original_df.index, original_df['redeem_amt'], 'r-', linewidth=1)
    axes[0, 0].set_title('原始赎回总额时间序列')
    axes[0, 0].set_ylabel('赎回金额')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 一阶差分时间序列
    axes[0, 1].plot(diff_series.index, diff_series.values, 'g-', linewidth=1)
    axes[0, 1].set_title('赎回总额一阶差分')
    axes[0, 1].set_ylabel('差分值')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 原始数据分布
    axes[1, 0].hist(original_df['redeem_amt'], bins=30, alpha=0.7, color='red')
    axes[1, 0].set_title('原始赎回金额分布')
    axes[1, 0].set_xlabel('赎回金额')
    axes[1, 0].set_ylabel('频次')
    
    # 差分后分布
    axes[1, 1].hist(diff_series.values, bins=30, alpha=0.7, color='green')
    axes[1, 1].set_title('差分后赎回金额分布')
    axes[1, 1].set_xlabel('差分值')
    axes[1, 1].set_ylabel('频次')
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = f"{get_project_path('..', 'user_data')}/differencing_analysis_{original_df.index[0].strftime('%Y%m%d')}_{original_df.index[-1].strftime('%Y%m%d')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"差分分析图表已保存到: {plot_file}")
    return plot_file


if __name__ == "__main__":
    perform_stationarity_analysis()