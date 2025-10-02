#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - 探索性数据分析（EDA）
对训练数据进行全面的探索性分析，包括数据分布、相关性分析、异常值检测等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

# 设置中文字体和图表样式（参考腾讯云文章解决方案）
# 支持多种中文字体，按优先级尝试
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 尝试重建字体缓存（如果可用）
try:
    from matplotlib.font_manager import _rebuild
    _rebuild()  # reload一下
except:
    pass

# 设置图形显示参数
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 10
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

def load_data():
    """加载数据"""
    data_path = "../data/used_car_train_20200313.csv"
    print("正在加载数据...")
    df = pd.read_csv(data_path, sep=' ')
    print(f"数据加载完成，形状: {df.shape}")
    return df

def basic_info_analysis(df):
    """基本信息分析"""
    print("\n" + "="*80)
    print("1. 数据基本信息分析")
    print("="*80)
    
    print(f"数据集形状: {df.shape}")
    print(f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 数据类型统计
    print("\n数据类型分布:")
    dtype_counts = df.dtypes.value_counts()
    print(dtype_counts)
    
    # 缺失值统计
    print("\n缺失值统计:")
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
    if len(missing_stats) > 0:
        missing_df = pd.DataFrame({
            '缺失数量': missing_stats,
            '缺失比例(%)': (missing_stats / len(df) * 100).round(2)
        })
        print(missing_df)
    else:
        print("没有缺失值")

def target_variable_analysis(df):
    """目标变量分析"""
    print("\n" + "="*80)
    print("2. 目标变量（价格）分析")
    print("="*80)
    
    # 在绘图前重新设置字体（避免seaborn干扰）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    price = df['price']
    
    # 基本统计信息
    print("价格基本统计信息:")
    print(f"均值: {price.mean():,.2f}")
    print(f"中位数: {price.median():,.2f}")
    print(f"标准差: {price.std():,.2f}")
    print(f"最小值: {price.min():,.2f}")
    print(f"最大值: {price.max():,.2f}")
    print(f"偏度: {price.skew():.2f}")
    print(f"峰度: {price.kurtosis():.2f}")
    
    # 分位数分析
    print("\n价格分位数分析:")
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for q in quantiles:
        print(f"{q*100}%分位数: {price.quantile(q):,.2f}")
    
    # 价格分布可视化（优化中文显示）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 直方图
    axes[0,0].hist(price, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title(u'价格分布直方图', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel(u'价格（元）', fontsize=12)
    axes[0,0].set_ylabel(u'频次', fontsize=12)
    axes[0,0].grid(True, alpha=0.3)
    
    # 对数变换后的直方图
    log_price = np.log1p(price)
    axes[0,1].hist(log_price, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,1].set_title(u'价格对数变换分布', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel(u'log(价格+1)', fontsize=12)
    axes[0,1].set_ylabel(u'频次', fontsize=12)
    axes[0,1].grid(True, alpha=0.3)
    
    # 箱型图
    box_plot = axes[1,0].boxplot(price, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightcoral')
    axes[1,0].set_title(u'价格箱型图', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel(u'价格（元）', fontsize=12)
    axes[1,0].grid(True, alpha=0.3)
    
    # QQ图
    stats.probplot(price, dist="norm", plot=axes[1,1])
    axes[1,1].set_title(u'价格Q-Q图（正态分布）', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel(u'理论分位数', fontsize=12)
    axes[1,1].set_ylabel(u'样本分位数', fontsize=12)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(u'../docs/价格分布分析.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(u"价格分布图表已保存")
    plt.show()

def categorical_features_analysis(df):
    """分类特征分析"""
    print("\n" + "="*80)
    print("3. 分类特征分析")
    print("="*80)
    
    # 识别分类特征（包括编码后的分类特征）
    categorical_cols = ['brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 
                       'seller', 'offerType', 'regionCode']
    
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col} 特征分析:")
            unique_count = df[col].nunique()
            print(f"唯一值数量: {unique_count}")
            
            if unique_count < 20:  # 如果唯一值较少，显示详细统计
                value_counts = df[col].value_counts().head(10)
                print("前10个最频繁的值:")
                print(value_counts)
                
                # 计算与价格的关系
                if col != 'price':
                    price_by_category = df.groupby(col)['price'].agg(['mean', 'median', 'count'])
                    print(f"\n按{col}分组的价格统计:")
                    print(price_by_category.head(10))

def numerical_features_analysis(df):
    """数值特征分析"""
    print("\n" + "="*80)
    print("4. 数值特征分析")
    print("="*80)
    
    # 选择数值型特征
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'price' in numerical_cols:
        numerical_cols.remove('price')  # 移除目标变量
    
    print(f"数值型特征数量: {len(numerical_cols)}")
    
    # 描述性统计
    print("\n数值特征描述性统计:")
    desc_stats = df[numerical_cols].describe()
    print(desc_stats.round(2))
    
    # 异常值检测（使用IQR方法）
    print("\n异常值检测（IQR方法）:")
    outlier_summary = []
    
    for col in numerical_cols[:10]:  # 只分析前10个特征
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        outlier_summary.append({
            '特征': col,
            '异常值数量': outlier_count,
            '异常值比例(%)': round(outlier_pct, 2),
            '下界': round(lower_bound, 2),
            '上界': round(upper_bound, 2)
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df)

def correlation_analysis(df):
    """相关性分析"""
    print("\n" + "="*80)
    print("5. 特征相关性分析")
    print("="*80)
    
    # 选择数值型特征
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 计算相关性矩阵
    corr_matrix = df[numerical_cols].corr()
    
    # 与价格的相关性
    price_corr = corr_matrix['price'].abs().sort_values(ascending=False)
    print("与价格最相关的特征（按绝对值排序）:")
    print(price_corr.head(15))
    
    # 绘制相关性热力图（优化中文显示）
    # 重新设置字体（避免seaborn干扰）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(20, 16))
    
    # 只显示与价格相关性较高的特征
    top_features = price_corr.head(15).index.tolist()
    corr_subset = corr_matrix.loc[top_features, top_features]
    
    sns.heatmap(corr_subset, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, fmt='.2f', cbar_kws={"shrink": .8},
                linewidths=0.5, linecolor='white')
    plt.title(u'特征相关性热力图（Top 15 与价格相关的特征）', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.savefig(u'../docs/特征相关性热力图.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(u"相关性热力图已保存")
    plt.show()
    
    # 高相关性特征对识别
    print("\n高相关性特征对（|相关系数| > 0.8）:")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    '特征1': corr_matrix.columns[i],
                    '特征2': corr_matrix.columns[j],
                    '相关系数': round(corr_val, 3)
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        print(high_corr_df)
    else:
        print("没有发现高相关性特征对")

def time_features_analysis(df):
    """时间特征分析"""
    print("\n" + "="*80)
    print("6. 时间特征分析")
    print("="*80)
    
    # 转换日期格式
    df_time = df.copy()
    
    # 注册日期分析
    if 'regDate' in df.columns:
        df_time['regDate_str'] = df_time['regDate'].astype(str)
        df_time['reg_year'] = df_time['regDate_str'].str[:4].astype(int)
        df_time['reg_month'] = df_time['regDate_str'].str[4:6].astype(int)
        
        print("注册年份分布:")
        reg_year_stats = df_time['reg_year'].value_counts().sort_index()
        print(reg_year_stats.head(10))
        
        print("\n注册月份分布:")
        reg_month_stats = df_time['reg_month'].value_counts().sort_index()
        print(reg_month_stats)
        
        # 计算车龄（假设当前年份为2020）
        current_year = 2020
        df_time['car_age'] = current_year - df_time['reg_year']
        
        print(f"\n车龄统计:")
        print(f"平均车龄: {df_time['car_age'].mean():.1f} 年")
        print(f"车龄中位数: {df_time['car_age'].median():.1f} 年")
        print(f"最大车龄: {df_time['car_age'].max()} 年")
        print(f"最小车龄: {df_time['car_age'].min()} 年")
        
        # 车龄与价格关系
        age_price_corr = df_time[['car_age', 'price']].corr().iloc[0,1]
        print(f"车龄与价格相关系数: {age_price_corr:.3f}")
    
    # 创建日期分析可视化
    if 'regDate' in df.columns:
        # 重新设置字体（避免seaborn干扰）
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 注册年份分布
        reg_year_stats.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title(u'注册年份分布')
        axes[0,0].set_xlabel(u'年份')
        axes[0,0].set_ylabel(u'数量')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 注册月份分布
        reg_month_stats.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title(u'注册月份分布')
        axes[0,1].set_xlabel(u'月份')
        axes[0,1].set_ylabel(u'数量')
        
        # 车龄分布
        axes[1,0].hist(df_time['car_age'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1,0].set_title(u'车龄分布')
        axes[1,0].set_xlabel(u'车龄（年）')
        axes[1,0].set_ylabel(u'频次')
        
        # 车龄与价格散点图
        sample_data = df_time.sample(n=min(5000, len(df_time)))  # 采样避免过度绘制
        axes[1,1].scatter(sample_data['car_age'], sample_data['price'], alpha=0.5, s=1)
        axes[1,1].set_title(u'车龄vs价格 (相关系数: {:.3f})'.format(age_price_corr))
        axes[1,1].set_xlabel(u'车龄（年）')
        axes[1,1].set_ylabel(u'价格')
        
        plt.tight_layout()
        plt.savefig(u'../docs/时间特征分析.png', dpi=300, bbox_inches='tight')
        plt.show()

def feature_importance_analysis(df):
    """特征重要性分析（基于简单的统计方法）"""
    print("\n" + "="*80)
    print("7. 特征重要性分析")
    print("="*80)
    
    # 计算数值特征与价格的相关性
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'price' in numerical_cols:
        numerical_cols.remove('price')
    
    feature_importance = []
    
    for col in numerical_cols:
        # 计算相关系数
        corr = df[col].corr(df['price'])
        
        # 计算互信息（简化版本：使用分箱后的互信息）
        try:
            # 将连续变量分箱
            col_binned = pd.cut(df[col].dropna(), bins=10, labels=False)
            price_binned = pd.cut(df['price'], bins=10, labels=False)
            
            # 计算条件熵的简化版本
            mutual_info = 0
            for bin_val in range(10):
                if sum(col_binned == bin_val) > 0:
                    prob = sum(col_binned == bin_val) / len(col_binned)
                    conditional_entropy = 0
                    subset_price = price_binned[col_binned == bin_val]
                    for price_bin in range(10):
                        if len(subset_price) > 0:
                            cond_prob = sum(subset_price == price_bin) / len(subset_price)
                            if cond_prob > 0:
                                conditional_entropy -= cond_prob * np.log2(cond_prob)
                    mutual_info += prob * conditional_entropy
        except:
            mutual_info = 0
        
        feature_importance.append({
            '特征': col,
            '相关系数': abs(corr),
            '互信息': mutual_info
        })
    
    # 按相关系数排序
    importance_df = pd.DataFrame(feature_importance)
    importance_df = importance_df.sort_values('相关系数', ascending=False)
    
    print("特征重要性排序（前20个）:")
    print(importance_df.head(20))
    
    # 可视化特征重要性
    # 重新设置字体（避免seaborn干扰）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    
    plt.barh(range(len(top_features)), top_features['相关系数'])
    plt.yticks(range(len(top_features)), top_features['特征'])
    plt.xlabel(u'与价格的相关系数（绝对值）')
    plt.title(u'特征重要性排序（Top 15）')
    plt.gca().invert_yaxis()
    
    for i, v in enumerate(top_features['相关系数']):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(u'../docs/特征重要性分析.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_eda_report(df):
    """生成最完整的EDA分析报告"""
    print("\n" + "="*80)
    print("8. 生成最完整版EDA分析报告")
    print("="*80)
    
    # 确保文档报告目录存在
    os.makedirs('../docs', exist_ok=True)
    
    # 计算详细统计信息
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numerical_cols].corr()
    price_corr = corr_matrix['price'].abs().sort_values(ascending=False)
    
    # 计算异常值统计
    price_Q1 = df['price'].quantile(0.25)
    price_Q3 = df['price'].quantile(0.75)
    price_IQR = price_Q3 - price_Q1
    price_outliers = df[(df['price'] < price_Q1 - 1.5 * price_IQR) | 
                       (df['price'] > price_Q3 + 1.5 * price_IQR)]
    
    power_outliers = df[df['power'] > 600] if 'power' in df.columns else pd.DataFrame()
    
    # 计算时间特征
    df_temp = df.copy()
    if 'regDate' in df.columns:
        df_temp['regDate_str'] = df_temp['regDate'].astype(str)
        df_temp['reg_year'] = df_temp['regDate_str'].str[:4].astype(int)
        current_year = 2020
        df_temp['car_age'] = current_year - df_temp['reg_year']
        age_price_corr = df_temp[['car_age', 'price']].corr().iloc[0,1]
    
    report_content = f"""# 二手车价格预测 - 最完整版EDA分析报告

## 数据集概览
- **数据集大小**: {df.shape[0]:,} 行 × {df.shape[1]} 列
- **内存使用**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- **数据时间**: 2020年3月13日
- **分析时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 核心发现总结

### 1. 目标变量（价格）核心特征
- **平均价格**: {df['price'].mean():,.2f} 元
- **价格中位数**: {df['price'].median():,.2f} 元
- **价格范围**: {df['price'].min():,.2f} - {df['price'].max():,.2f} 元
- **标准差**: {df['price'].std():,.2f} 元
- **分布特征**: 左偏分布（偏度 {df['price'].skew():.2f}），尖峰分布（峰度 {df['price'].kurtosis():.2f}）

### 2. 数据质量评估
"""
    
    # 添加缺失值信息
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
    
    if len(missing_stats) > 0:
        report_content += "\n**缺失值情况**:\n"
        for col, missing_count in missing_stats.items():
            missing_pct = (missing_count / len(df)) * 100
            report_content += f"- {col}: {missing_count:,} 个缺失值 ({missing_pct:.2f}%)\n"
    else:
        report_content += "\n**数据完整性**: 大部分字段数据完整\n"
    
    # 添加异常值信息
    report_content += f"\n**异常值统计**:\n"
    report_content += f"- 价格异常值: {len(price_outliers):,} 条记录 ({len(price_outliers)/len(df)*100:.2f}%)\n"
    if len(power_outliers) > 0:
        report_content += f"- power异常值 (>600): {len(power_outliers):,} 条记录 ({len(power_outliers)/len(df)*100:.2f}%)\n"
    
    # 添加特征类型统计
    dtype_counts = df.dtypes.value_counts()
    report_content += f"\n### 3. 特征类型分布\n"
    for dtype, count in dtype_counts.items():
        report_content += f"- {dtype}: {count} 个特征\n"
    
    # 添加相关性分析结果
    report_content += f"\n### 4. 特征重要性排序（与价格相关性）\n"
    count = 0
    for feature, corr_val in price_corr.items():
        if feature != 'price' and count < 15:
            report_content += f"{count+1:2d}. **{feature}**: {corr_val:.4f}\n"
            count += 1
    
    # 添加时间特征分析
    if 'regDate' in df.columns:
        report_content += f"\n### 5. 时间特征洞察\n"
        report_content += f"- 平均车龄: {df_temp['car_age'].mean():.1f} 年\n"
        report_content += f"- 车龄中位数: {df_temp['car_age'].median():.1f} 年\n"
        report_content += f"- 车龄范围: {df_temp['car_age'].min()} - {df_temp['car_age'].max()} 年\n"
        report_content += f"- 车龄与价格相关系数: {age_price_corr:.4f} (强负相关)\n"
    
    # 添加分位数分析
    report_content += f"\n### 6. 价格分位数分析\n"
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for q in quantiles:
        report_content += f"- {q*100:4.0f}%分位数: {df['price'].quantile(q):8,.2f} 元\n"
    
    # 添加高相关性特征对
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    '特征1': corr_matrix.columns[i],
                    '特征2': corr_matrix.columns[j],
                    '相关系数': corr_val
                })
    
    report_content += f"\n### 7. 高相关性特征对 (|相关系数| > 0.8)\n"
    if high_corr_pairs:
        for pair in high_corr_pairs[:10]:  # 只显示前10个
            report_content += f"- {pair['特征1']} vs {pair['特征2']}: {pair['相关系数']:.4f}\n"
    else:
        report_content += "- 无高相关性特征对\n"
    
    # 添加项目规范要求的异常值处理建议
    report_content += f"\n## 数据预处理建议\n\n### 异常值处理规范（项目要求）\n1. **价格异常值**: 直接删除 {len(price_outliers):,} 条异常记录\n2. **power异常值**: 超过600的记录统一设置为600"
    if len(power_outliers) > 0:
        report_content += f" ({len(power_outliers):,} 条记录需要处理)"
    
    report_content += f"\n\n### 缺失值处理建议"
    if len(missing_stats) > 0:
        report_content += "\n"
        for col, missing_count in missing_stats.items():
            missing_pct = (missing_count / len(df)) * 100
            if missing_pct < 5:
                report_content += f"- **{col}**: 缺失比例较低 ({missing_pct:.2f}%)，建议使用众数/均值填充\n"
            else:
                report_content += f"- **{col}**: 缺失比例较高 ({missing_pct:.2f}%)，建议创建缺失值指示变量\n"
    else:
        report_content += "\n- 数据完整性良好，无需特殊处理\n"
    
    # 添加特征工程建议
    report_content += f"\n### 特征工程建议\n"
    report_content += "1. **时间特征提取**: 从 regDate 提取车龄、季节、月份等特征\n"
    report_content += "2. **分类特征编码**: 对 brand、bodyType 等分类特征进行标签编码\n"
    report_content += "3. **目标变量变换**: 由于价格呈偏态分布，建议使用对数变换\n"
    report_content += "4. **特征选择**: 重点关注 v_3, v_12, v_8, v_0, regDate 等高相关性特征\n"
    report_content += "5. **数据标准化**: 对数值特征进行标准化或归一化处理\n"
    
    # 添加建模建议
    report_content += f"\n## 建模策略建议\n"
    report_content += "\n### 模型选择\n"
    report_content += "1. **集成学习方法**: 随机森林、XGBoost、LightGBM、CatBoost\n"
    report_content += "2. **模型融合**: 结合多个模型的预测结果，提高泛化性能\n"
    report_content += "3. **交叉验证**: 使用K折交叉验证评估模型性能\n"
    
    report_content += "\n### 评估指标\n"
    report_content += "1. **主要指标**: MAE (Mean Absolute Error)、RMSE (Root Mean Square Error)\n"
    report_content += "2. **辅助指标**: MAPE (Mean Absolute Percentage Error)、R² Score\n"
    report_content += "3. **分布校准**: 关注预测结果在不同价格区间的准确性\n"
    
    report_content += "\n### 模型优化方向\n"
    report_content += "1. **超参数调优**: 使用网格搜索或贝叶斯优化\n"
    report_content += "2. **特征重要性分析**: 利用模型输出的特征重要性进行特征筛选\n"
    report_content += "3. **集成学习权重优化**: 针对不同模型调整融合权重\n"
    
    report_content += f"\n## 附录：生成文件\n"
    report_content += "- 📈 `价格分布分析.png` - 价格分布可视化图表\n"
    report_content += "- 🔥 `特征相关性热力图.png` - 特征间相关性可视化\n"
    report_content += "- ⏰ `时间特征分析.png` - 时间相关特征分析\n"
    report_content += "- 📊 `特征重要性分析.png` - 特征重要性排序图表\n"
    
    report_content += f"\n---\n*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    
    # 保存报告
    with open(u'../docs/最完整版EDA分析报告.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(u"最完整版EDA分析报告已保存到: 文档报告/最完整版EDA分析报告.md")

def main():
    """主函数"""
    print("开始进行探索性数据分析（EDA）...")
    
    # 创建文档报告目录
    os.makedirs('../docs', exist_ok=True)
    
    # 加载数据
    df = load_data()
    
    # 执行各项分析
    basic_info_analysis(df)
    target_variable_analysis(df)
    categorical_features_analysis(df)
    numerical_features_analysis(df)
    correlation_analysis(df)
    time_features_analysis(df)
    feature_importance_analysis(df)
    generate_comprehensive_eda_report(df)  # 使用新的完整报告函数
    
    print("\n" + "="*80)
    print("EDA分析完成！")
    print("生成的文件:")
    print("- 文档报告/最完整版EDA分析报告.md")
    print("- 文档报告/价格分布分析.png")
    print("- 文档报告/特征相关性热力图.png")
    print("- 文档报告/时间特征分析.png")
    print("- 文档报告/特征重要性分析.png")
    print("="*80)

if __name__ == "__main__":
    main()