"""
V4版本模型深入分析脚本 - 分析V4模型表现异常的原因

目标:
1. 深入分析V4模型预测结果异常的原因
2. 对比V3和V4模型的预测差异
3. 分析不同价格区间的预测表现
4. 提供针对性的优化建议
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_project_path(*paths):
    """获取项目路径的统一方法"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        return os.path.join(project_dir, *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)

def get_user_data_path(*paths):
    """获取用户数据路径"""
    return get_project_path('user_data', *paths)

def load_data():
    """加载训练集和测试集数据"""
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    return train_df, test_df

def load_model_predictions():
    """加载各版本模型的预测结果"""
    # 查找预测文件
    pred_dir = get_project_path('prediction_result')
    model_files = {}
    
    if os.path.exists(pred_dir):
        for file in os.listdir(pred_dir):
            if file.endswith('.csv'):
                if 'modeling_v1' in file:
                    model_files['v1'] = os.path.join(pred_dir, file)
                elif 'modeling_v2' in file:
                    model_files['v2'] = os.path.join(pred_dir, file)
                elif 'modeling_v3' in file:
                    model_files['v3'] = os.path.join(pred_dir, file)
                elif 'modeling_v4' in file:
                    model_files['v4'] = os.path.join(pred_dir, file)
    
    predictions = {}
    for version, file_path in model_files.items():
        try:
            pred_df = pd.read_csv(file_path)
            predictions[version] = pred_df
            print(f"加载{version.upper()}预测文件: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"无法加载{version.upper()}预测文件: {e}")
    
    return predictions

def analyze_prediction_distributions(train_df, predictions):
    """分析各版本模型预测分布"""
    print("\n=== 预测分布分析 ===")
    
    # 训练集价格分布
    train_stats = {
        'mean': train_df['price'].mean(),
        'median': train_df['price'].median(),
        'std': train_df['price'].std()
    }
    print(f"训练集价格 - 均值: {train_stats['mean']:.2f}, 中位数: {train_stats['median']:.2f}, 标准差: {train_stats['std']:.2f}")
    
    # 各版本模型预测分布
    for version, pred_df in predictions.items():
        pred_stats = {
            'mean': pred_df['price'].mean(),
            'median': pred_df['price'].median(),
            'std': pred_df['price'].std()
        }
        print(f"{version.upper()}预测 - 均值: {pred_stats['mean']:.2f}, 中位数: {pred_stats['median']:.2f}, 标准差: {pred_stats['std']:.2f}")
        
        # 与训练集的差异
        mean_diff = abs(pred_stats['mean'] - train_stats['mean'])
        std_diff = abs(pred_stats['std'] - train_stats['std'])
        print(f"  与训练集均值差异: {mean_diff:.2f}, 标准差差异: {std_diff:.2f}")

def analyze_price_segments(train_df, predictions):
    """分析不同价格区间的预测表现"""
    print("\n=== 价格区间分析 ===")
    
    # 将训练集价格分段
    train_df['price_segment'] = pd.cut(train_df['price'], 
                                      bins=[0, 5000, 10000, 15000, 20000, 30000, 50000, 100000],
                                      labels=['<5K', '5K-10K', '10K-15K', '15K-20K', '20K-30K', '30K-50K', '>50K'])
    
    # 计算每个价格区间的样本数
    segment_stats = train_df['price_segment'].value_counts().sort_index()
    print("训练集价格区间分布:")
    for segment, count in segment_stats.items():
        print(f"  {segment}: {count} 样本 ({count/len(train_df)*100:.1f}%)")
    
    # 分析各版本模型在各价格区间的预测分布
    for version, pred_df in predictions.items():
        if 'price' in pred_df.columns:
            pred_df['price_segment'] = pd.cut(pred_df['price'], 
                                             bins=[0, 5000, 10000, 15000, 20000, 30000, 50000, 100000],
                                             labels=['<5K', '5K-10K', '10K-15K', '15K-20K', '20K-30K', '30K-50K', '>50K'])
            
            pred_segment_stats = pred_df['price_segment'].value_counts().sort_index()
            print(f"\n{version.upper()}预测价格区间分布:")
            for segment, count in pred_segment_stats.items():
                print(f"  {segment}: {count} 样本 ({count/len(pred_df)*100:.1f}%)")

def compare_model_performance(predictions):
    """比较各版本模型性能"""
    print("\n=== 模型性能比较 ===")
    
    for version, pred_df in predictions.items():
        if 'price' in pred_df.columns:
            pred_mean = pred_df['price'].mean()
            pred_std = pred_df['price'].std()
            print(f"{version.upper()} - 均值: {pred_mean:.2f}, 标准差: {pred_std:.2f}")

def generate_visualizations(train_df, predictions):
    """生成可视化图表"""
    print("\n=== 生成可视化图表 ===")
    
    # 创建保存目录
    viz_dir = get_user_data_path('modeling_v4')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. 价格分布对比图
    plt.figure(figsize=(15, 10))
    
    # 训练集价格分布
    plt.subplot(2, 3, 1)
    plt.hist(train_df['price'], bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.title('训练集价格分布', fontsize=14)
    plt.xlabel('价格')
    plt.ylabel('频数')
    
    # 各版本模型预测分布
    versions = list(predictions.keys())
    for i, version in enumerate(versions):
        if i >= 5:  # 最多显示5个子图
            break
        plt.subplot(2, 3, i+2)
        plt.hist(predictions[version]['price'], bins=100, alpha=0.7, color='green', edgecolor='black')
        plt.title(f'{version.upper()}预测分布', fontsize=14)
        plt.xlabel('价格')
        plt.ylabel('频数')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '价格分布对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存价格分布对比图: {os.path.join(viz_dir, '价格分布对比.png')}")
    
    # 2. 均值和标准差对比
    plt.figure(figsize=(12, 6))
    
    versions = ['train'] + list(predictions.keys())
    means = [train_df['price'].mean()]
    stds = [train_df['price'].std()]
    
    for version in predictions.keys():
        means.append(predictions[version]['price'].mean())
        stds.append(predictions[version]['price'].std())
    
    x = range(len(versions))
    plt.subplot(1, 2, 1)
    plt.bar(x, means, color=['blue'] + ['green']*len(predictions.keys()))
    plt.xticks(x, ['train'] + [v.upper() for v in predictions.keys()], rotation=45)
    plt.title('各版本均值对比')
    plt.ylabel('价格均值')
    
    plt.subplot(1, 2, 2)
    plt.bar(x, stds, color=['blue'] + ['green']*len(predictions.keys()))
    plt.xticks(x, ['train'] + [v.upper() for v in predictions.keys()], rotation=45)
    plt.title('各版本标准差对比')
    plt.ylabel('价格标准差')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '统计指标对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存统计指标对比图: {os.path.join(viz_dir, '统计指标对比.png')}")

def diagnose_v4_issues(train_df, predictions):
    """诊断V4模型问题"""
    print("\n=== V4模型问题诊断 ===")
    
    if 'v4' not in predictions:
        print("未找到V4模型预测结果")
        return
    
    v4_pred = predictions['v4']
    train_mean = train_df['price'].mean()
    v4_mean = v4_pred['price'].mean()
    
    print(f"训练集均值: {train_mean:.2f}")
    print(f"V4预测均值: {v4_mean:.2f}")
    print(f"均值差异: {abs(v4_mean - train_mean):.2f}")
    
    # 检查是否出现极端预测值
    extreme_high = (v4_pred['price'] > 100000).sum()
    extreme_low = (v4_pred['price'] < 0).sum()
    
    print(f"极端高价预测 (>100K): {extreme_high} 个")
    print(f"负值预测: {extreme_low} 个")
    
    # 分析V4相对于V3的差异
    if 'v3' in predictions:
        v3_pred = predictions['v3']
        v3_mean = v3_pred['price'].mean()
        mean_diff = abs(v4_mean - v3_mean)
        print(f"V4相对于V3均值差异: {mean_diff:.2f}")
        
        # 计算版本间预测差异
        if len(v4_pred) == len(v3_pred):
            pred_diff = np.abs(v4_pred['price'] - v3_pred['price'])
            print(f"V4与V3预测差异 - 均值: {pred_diff.mean():.2f}, 最大值: {pred_diff.max():.2f}")

def generate_detailed_report():
    """生成详细分析报告"""
    print("\n" + "="*60)
    print("V4模型深入分析报告")
    print("="*60)
    
    # 加载数据
    train_df, test_df = load_data()
    
    # 加载各版本模型预测结果
    predictions = load_model_predictions()
    
    # 分析预测分布
    analyze_prediction_distributions(train_df, predictions)
    
    # 分析价格区间
    analyze_price_segments(train_df, predictions)
    
    # 比较模型性能
    compare_model_performance(predictions)
    
    # 诊断V4问题
    diagnose_v4_issues(train_df, predictions)
    
    # 生成可视化图表
    generate_visualizations(train_df, predictions)
    
    print("\n" + "="*60)
    print("V4模型问题诊断结论")
    print("="*60)
    print("1. V4模型预测均值与训练集差异较大，可能存在以下问题:")
    print("   - 分段策略不当，导致某些区间的模型表现异常")
    print("   - 特征工程引入了噪声特征")
    print("   - 模型集成权重设置不合理")
    print("   - 校准方法过度调整")
    print("")
    print("2. 可能的优化方向:")
    print("   - 重新设计分段策略，确保各区间样本分布合理")
    print("   - 简化特征工程，去除噪声特征")
    print("   - 调整模型集成权重")
    print("   - 优化校准方法，避免过度调整")
    print("="*60)

def main():
    """主函数"""
    print("开始V4模型深入分析...")
    
    # 生成详细报告
    generate_detailed_report()
    
    print("\n分析完成!")

if __name__ == "__main__":
    main()