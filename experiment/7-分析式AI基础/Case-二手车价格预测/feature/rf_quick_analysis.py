# -*- coding: utf-8 -*-
"""
随机森林快速分析 - 轻量版本
快速生成核心诊断图表，判断训练状态
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_project_path(*paths):
    """获取项目路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, *paths)

def load_data_fast():
    """快速加载数据 - 使用采样减少计算量"""
    print("🔄 快速加载数据...")
    
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    train_data = pd.read_csv(train_path, sep=' ')
    
    # 随机采样减少数据量，加快分析速度
    sample_size = min(20000, len(train_data))  # 最多使用2万样本
    train_data = train_data.sample(n=sample_size, random_state=42)
    print(f"采样数据: {train_data.shape}")
    
    # 快速预处理
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'price':
            train_data[col] = train_data[col].fillna(train_data[col].median())
    
    # 分类特征简单处理
    categorical_cols = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    for col in categorical_cols:
        if col in train_data.columns:
            train_data[col] = train_data[col].fillna(0)
            # 简单的标签编码
            train_data[col] = pd.Categorical(train_data[col]).codes
    
    # 处理异常值
    if 'power' in train_data.columns:
        train_data.loc[train_data['power'] > 600, 'power'] = 600
    
    # 价格异常值
    train_data = train_data[(train_data['price'] >= 500) & (train_data['price'] <= 50000)]
    
    # 简单车龄特征
    if 'regDate' in train_data.columns:
        train_data['car_age'] = 2020 - (train_data['regDate'] // 10000)
        train_data['car_age'] = np.clip(train_data['car_age'], 1, 30)
    
    feature_cols = [col for col in train_data.columns if col != 'price']
    X = train_data[feature_cols]
    y = train_data['price']
    
    print(f"预处理完成: {X.shape}")
    return X, y

def analyze_rf_performance(X, y):
    """快速分析随机森林性能"""
    print("🎯 快速性能分析...")
    
    results = {}
    
    # 1. 简单的学习曲线 - 不同训练集大小
    print("1️⃣ 学习曲线分析...")
    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    train_mae_list = []
    val_mae_list = []
    
    for size in train_sizes:
        n_samples = int(len(X) * size)
        X_subset = X.iloc[:n_samples]
        y_subset = y.iloc[:n_samples]
        
        # 简单模型
        rf = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=2)
        
        # 训练集性能
        rf.fit(X_subset, y_subset)
        train_pred = rf.predict(X_subset)
        train_mae = mean_absolute_error(y_subset, train_pred)
        train_mae_list.append(train_mae)
        
        # 验证集性能（交叉验证）
        val_scores = cross_val_score(rf, X_subset, y_subset, cv=3, 
                                   scoring='neg_mean_absolute_error', n_jobs=2)
        val_mae = -val_scores.mean()
        val_mae_list.append(val_mae)
    
    results['learning_curve'] = (train_sizes, train_mae_list, val_mae_list)
    
    # 2. 收敛分析 - 不同树数量
    print("2️⃣ 收敛分析...")
    n_estimators_range = [10, 25, 50, 75, 100, 150]
    convergence_scores = []
    
    for n_est in n_estimators_range:
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=15, random_state=42, n_jobs=2)
        scores = cross_val_score(rf, X, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=2)
        convergence_scores.append(-scores.mean())
    
    results['convergence'] = (n_estimators_range, convergence_scores)
    
    # 3. 特征重要性
    print("3️⃣ 特征重要性...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=2)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    results['feature_importance'] = importance_df
    
    # 4. 参数对比
    print("4️⃣ 参数对比...")
    depth_range = [8, 12, 15, 18, 20]
    depth_scores = []
    
    for depth in depth_range:
        rf = RandomForestRegressor(n_estimators=50, max_depth=depth, random_state=42, n_jobs=2)
        scores = cross_val_score(rf, X, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=2)
        depth_scores.append(-scores.mean())
    
    results['parameter_analysis'] = (depth_range, depth_scores)
    
    return results

def create_analysis_plots(results):
    """创建分析图表"""
    print("📊 生成分析图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('随机森林训练状态快速分析', fontsize=16, fontweight='bold')
    
    # 1. 学习曲线
    train_sizes, train_mae_list, val_mae_list = results['learning_curve']
    
    axes[0,0].plot([int(s*20000) for s in train_sizes], train_mae_list, 'o-', 
                   label='训练集MAE', color='blue', linewidth=2)
    axes[0,0].plot([int(s*20000) for s in train_sizes], val_mae_list, 'o-', 
                   label='验证集MAE', color='red', linewidth=2)
    axes[0,0].set_title('🔍 学习曲线 - 训练vs验证Gap')
    axes[0,0].set_xlabel('训练样本数')
    axes[0,0].set_ylabel('MAE')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 计算Gap
    final_gap = val_mae_list[-1] - train_mae_list[-1]
    gap_status = "✅良好" if final_gap < 80 else "⚠️注意" if final_gap < 150 else "❌过拟合"
    axes[0,0].text(0.05, 0.95, f'Gap: {final_gap:.1f}\n状态: {gap_status}', 
                   transform=axes[0,0].transAxes, fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
                   verticalalignment='top')
    
    # 2. 收敛分析
    n_estimators_range, convergence_scores = results['convergence']
    
    axes[0,1].plot(n_estimators_range, convergence_scores, 'o-', color='green', linewidth=2)
    axes[0,1].set_title('🌳 收敛分析 - 树数量优化')
    axes[0,1].set_xlabel('树数量')
    axes[0,1].set_ylabel('验证集MAE')
    axes[0,1].grid(True, alpha=0.3)
    
    # 标记最佳点
    best_idx = np.argmin(convergence_scores)
    best_n_est = n_estimators_range[best_idx]
    best_mae = convergence_scores[best_idx]
    axes[0,1].axvline(x=best_n_est, color='red', linestyle='--', alpha=0.7)
    axes[0,1].text(best_n_est + 5, best_mae + 10, f'最佳: {best_n_est}棵树', 
                   fontsize=9, color='red', fontweight='bold')
    
    # 3. 特征重要性 Top10
    importance_df = results['feature_importance']
    top_features = importance_df.head(10)
    
    axes[1,0].barh(range(len(top_features)), top_features['importance'][::-1], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
    axes[1,0].set_yticks(range(len(top_features)))
    axes[1,0].set_yticklabels(top_features['feature'][::-1])
    axes[1,0].set_title('📊 Top10 重要特征')
    axes[1,0].set_xlabel('重要性')
    
    # 4. 参数分析
    depth_range, depth_scores = results['parameter_analysis']
    
    axes[1,1].plot(depth_range, depth_scores, 'o-', color='orange', linewidth=2)
    axes[1,1].set_title('⚖️ 深度参数分析')
    axes[1,1].set_xlabel('max_depth')
    axes[1,1].set_ylabel('验证集MAE')
    axes[1,1].grid(True, alpha=0.3)
    
    # 标记最佳深度
    best_depth_idx = np.argmin(depth_scores)
    best_depth = depth_range[best_depth_idx]
    axes[1,1].axvline(x=best_depth, color='green', linestyle='--', alpha=0.7)
    axes[1,1].text(best_depth + 0.5, min(depth_scores) + 5, f'最佳: {best_depth}', 
                   fontsize=9, color='green', fontweight='bold')
    
    return fig, final_gap, best_n_est, best_depth

def main():
    """主函数"""
    print("🎯 随机森林快速训练状态分析")
    print("="*50)
    
    # 加载数据
    X, y = load_data_fast()
    
    # 分析性能
    results = analyze_rf_performance(X, y)
    
    # 创建图表
    fig, final_gap, best_n_est, best_depth = create_analysis_plots(results)
    
    # 保存图表
    results_dir = get_project_path('prediction_result')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(results_dir, f'rf_quick_analysis_{timestamp}.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 快速分析图表已保存: {save_path}")
    
    # 显示图表
    plt.show()
    
    # 生成简要报告
    print("\n" + "="*50)
    print("🎯 快速分析报告")
    print("="*50)
    
    print(f"📈 学习曲线分析:")
    print(f"   训练/验证Gap: {final_gap:.1f}")
    if final_gap > 150:
        print("   ❌ 过拟合严重，需要增加正则化")
    elif final_gap > 80:
        print("   ⚠️ 轻微过拟合，建议调参")
    else:
        print("   ✅ 泛化能力良好")
    
    print(f"\n🌳 收敛分析:")
    print(f"   建议树数量: {best_n_est}")
    if best_n_est >= 100:
        print("   💡 可考虑增加更多树")
    else:
        print("   ✅ 树数量已基本够用")
    
    print(f"\n⚖️ 参数优化:")
    print(f"   建议最大深度: {best_depth}")
    
    print(f"\n📊 特征重要性:")
    importance_df = results['feature_importance']
    print("   Top3特征:")
    for _, row in importance_df.head(3).iterrows():
        print(f"   • {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n🎯 总体建议:")
    if final_gap > 100:
        print("  优先解决过拟合问题:")
        print("  • 增加min_samples_split (10→15)")
        print("  • 增加min_samples_leaf (5→8)")
        print(f"  • 调整max_depth到{max(best_depth-2, 10)}")
    else:
        print("  模型状态良好，可进行微调:")
        print(f"  • 使用{best_n_est}棵树")
        print(f"  • 设置max_depth={best_depth}")
        print("  • 可尝试特征工程优化")
    
    print(f"\n💡 这些图表提供了类似深度学习'训练损失vs验证损失'的判断依据")
    print(f"✅ 可以据此判断随机森林是否达到最佳训练状态")
    
    return save_path

if __name__ == "__main__":
    result_path = main()