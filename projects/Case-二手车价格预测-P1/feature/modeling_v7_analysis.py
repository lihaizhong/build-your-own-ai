#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V7模型分析脚本
用于分析训练集和测试集的预测分布，帮助诊断模型性能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model.modeling_v7 import create_advanced_features, v7_optimize_model, get_project_path, get_user_data_path, load_and_preprocess_data

def load_data():
    """加载训练集和测试集数据"""
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_data = pd.read_csv(train_path, sep=' ')
    test_data = pd.read_csv(test_path, sep=' ')
    
    return train_data, test_data

def load_predictions():
    """加载模型预测结果"""
    # 加载最新的V7模型预测结果
    pred_paths = [
        get_project_path('prediction_result', 'modeling_v7_result_20251021_*.csv')
    ]
    
    import glob
    for pattern in pred_paths:
        matched_files = glob.glob(pattern)
        if matched_files:
            # 选择最新的文件
            latest_file = max(matched_files, key=os.path.getctime)
            predictions = pd.read_csv(latest_file)
            print(f"成功加载预测文件: {latest_file}")
            return predictions
    
    print("未找到任何预测文件")
    return None

def plot_prediction_distribution(train_data, test_predictions, save_path):
    """绘制训练集真实值和测试集预测值的分布对比图"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 训练集价格分布
    axes[0].hist(train_data['price'], bins=100, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    axes[0].set_title('训练集真实价格分布')
    axes[0].set_xlabel('价格')
    axes[0].set_ylabel('频次')
    axes[0].grid(True, alpha=0.3)
    
    # 测试集预测价格分布
    axes[1].hist(test_predictions['price'], bins=100, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
    axes[1].set_title('测试集预测价格分布')
    axes[1].set_xlabel('价格')
    axes[1].set_ylabel('频次')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '训练集与测试集预测分布对比.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_scatter(train_data, test_predictions, save_path):
    """绘制预测值散点图"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建价格分段用于颜色区分
    train_data['price_segment'] = pd.qcut(train_data['price'], q=10, labels=False, duplicates='drop')
    
    plt.figure(figsize=(12, 8))
    
    # 绘制训练集价格分布
    plt.scatter(range(len(train_data)), train_data['price'], 
                c=train_data['price_segment'], cmap='viridis', alpha=0.6, s=1, label='训练集真实价格')
    
    # 绘制测试集预测价格（为了对齐，我们使用相同数量的点）
    sample_size = min(len(train_data), len(test_predictions))
    test_sample = test_predictions.sample(n=sample_size, random_state=42)
    
    plt.scatter(range(len(test_sample)), test_sample['price'], 
                alpha=0.6, s=1, color='red', label='测试集预测价格')
    
    plt.title('训练集真实价格 vs 测试集预测价格（抽样对比）')
    plt.xlabel('样本索引')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '训练集与测试集预测散点图.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_statistics(train_data, test_predictions):
    """计算训练集和测试集预测值的统计信息"""
    train_stats = {
        'mean': train_data['price'].mean(),
        'median': train_data['price'].median(),
        'std': train_data['price'].std(),
        'min': train_data['price'].min(),
        'max': train_data['price'].max()
    }
    
    test_stats = {
        'mean': test_predictions['price'].mean(),
        'median': test_predictions['price'].median(),
        'std': test_predictions['price'].std(),
        'min': test_predictions['price'].min(),
        'max': test_predictions['price'].max()
    }
    
    # 计算差异
    diff_stats = {
        'mean_diff': abs(train_stats['mean'] - test_stats['mean']),
        'median_diff': abs(train_stats['median'] - test_stats['median']),
        'std_diff': abs(train_stats['std'] - test_stats['std'])
    }
    
    return train_stats, test_stats, diff_stats

def plot_learning_curve(train_df, save_path):
    """绘制学习曲线"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备特征
    train_df = create_advanced_features(train_df)
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    
    # 为了加快计算速度，我们使用部分数据进行学习曲线分析
    # 随机采样10%的数据
    sample_size = min(15000, len(X_train))  # 最多使用15000个样本
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train.iloc[indices]
    y_sample = y_train.iloc[indices]
    
    # 创建一个简单的随机森林模型用于学习曲线分析
    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    
    # 计算学习曲线
    learning_curve_result = learning_curve(
        model, X_sample, y_sample, 
        cv=3,  # 减少交叉验证折数以提高速度
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),  # 减少训练尺寸点数
        scoring='neg_mean_absolute_error',
        random_state=42
    )
    
    # 解包学习曲线结果
    train_sizes = learning_curve_result[0]
    train_scores = learning_curve_result[1]
    val_scores = learning_curve_result[2]
    
    # 转换为正的MAE值
    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = -train_scores.std(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    val_scores_std = -val_scores.std(axis=1)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.title('V7模型学习曲线')
    plt.xlabel('训练样本数')
    plt.ylabel('平均绝对误差 (MAE)')
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="训练集得分")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g",
             label="验证集得分")
    
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '模型学习曲线.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存学习曲线数据
    learning_curve_data = pd.DataFrame({
        'train_sizes': train_sizes,
        'train_scores_mean': train_scores_mean,
        'train_scores_std': train_scores_std,
        'val_scores_mean': val_scores_mean,
        'val_scores_std': val_scores_std
    })
    
    learning_curve_data.to_csv(os.path.join(save_path, '学习曲线数据.csv'), index=False)
    
    return learning_curve_data

def save_statistics_report(train_stats, test_stats, diff_stats, save_path):
    """保存统计分析报告"""
    report = f"""
# V7模型预测分布分析报告

## 训练集统计信息
- 均值: {train_stats['mean']:.2f}
- 中位数: {train_stats['median']:.2f}
- 标准差: {train_stats['std']:.2f}
- 最小值: {train_stats['min']:.2f}
- 最大值: {train_stats['max']:.2f}

## 测试集预测统计信息
- 均值: {test_stats['mean']:.2f}
- 中位数: {test_stats['median']:.2f}
- 标准差: {test_stats['std']:.2f}
- 最小值: {test_stats['min']:.2f}
- 最大值: {test_stats['max']:.2f}

## 分布差异分析
- 均值差异: {diff_stats['mean_diff']:.2f}
- 中位数差异: {diff_stats['median_diff']:.2f}
- 标准差差异: {diff_stats['std_diff']:.2f}

## 分析结论
"""
    
    # 根据差异大小给出分析结论
    if diff_stats['mean_diff'] > 200:
        report += "- 警告：训练集与测试集预测均值差异较大，可能存在数据分布偏移问题\n"
    else:
        report += "- 训练集与测试集预测均值差异在合理范围内\n"
        
    if diff_stats['std_diff'] > 1000:
        report += "- 警告：训练集与测试集预测标准差差异较大，模型可能存在过拟合或欠拟合问题\n"
    else:
        report += "- 训练集与测试集预测标准差差异在合理范围内\n"
    
    with open(os.path.join(save_path, '预测分布分析报告.md'), 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    """主函数"""
    # 创建保存目录
    # 使用绝对路径确保文件保存在正确位置
    save_path = get_project_path('user_data', 'modeling_v7')
    os.makedirs(save_path, exist_ok=True)
    
    # 加载数据
    print("正在加载数据...")
    train_data, test_data = load_data()
    
    # 如果预测结果不存在，则重新训练模型生成预测结果
    print("正在加载预测结果...")
    predictions = load_predictions()
    
    if predictions is None:
        print("预测文件不存在，重新训练模型生成预测结果...")
        # 训练模型并生成预测结果
        test_predictions = v7_optimize_model()
        
        # 重新加载测试数据以获取SaleID
        _, test_data = load_data()
        predictions = pd.DataFrame({
            'SaleID': test_data['SaleID'],
            'price': test_predictions
        })
        
        # 保存预测结果供后续使用
        result_dir = get_user_data_path('prediction_result')
        os.makedirs(result_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(result_dir, f"modeling_v7_result_{timestamp}.csv")
        predictions.to_csv(result_file, index=False)
        print(f"新生成的预测结果已保存到: {result_file}")
    
    # 绘制分布对比图
    print("正在绘制分布对比图...")
    plot_prediction_distribution(train_data, predictions, save_path)
    
    # 绘制散点图
    print("正在绘制散点图...")
    plot_prediction_scatter(train_data, predictions, save_path)
    
    # 计算统计信息
    print("正在计算统计信息...")
    train_stats, test_stats, diff_stats = calculate_statistics(train_data, predictions)
    
    # 绘制学习曲线
    print("正在绘制学习曲线...")
    # 重新加载训练数据用于学习曲线分析
    train_df, _ = load_and_preprocess_data()
    learning_curve_data = plot_learning_curve(train_df, save_path)
    print(f"学习曲线数据已保存到: {os.path.join(save_path, '学习曲线数据.csv')}")
    
    # 保存统计报告
    print("正在保存统计报告...")
    save_statistics_report(train_stats, test_stats, diff_stats, save_path)
    
    print(f"分析完成，结果已保存到: {save_path}")

if __name__ == "__main__":
    main()