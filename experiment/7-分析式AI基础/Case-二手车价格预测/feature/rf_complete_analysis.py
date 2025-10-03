# -*- coding: utf-8 -*-
"""
随机森林完整训练状态分析脚本
生成5个核心诊断图表：学习曲线、收敛分析、特征重要性、参数验证、残差分析
判断模型是否达到最佳训练状态
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (learning_curve, validation_curve, cross_val_score, 
                                   cross_val_predict, KFold)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")

def get_project_path(*paths):
    """获取项目路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, *paths)

def load_and_preprocess_data():
    """加载并预处理数据 - 基于项目规范"""
    print("🔄 加载并预处理数据...")
    
    # 加载训练数据
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    train_data = pd.read_csv(train_path, sep=' ')
    print(f"原始数据: {train_data.shape}")
    
    # 处理异常值 - power字段
    if 'power' in train_data.columns:
        power_outliers = (train_data['power'] > 600).sum()
        train_data.loc[train_data['power'] > 600, 'power'] = 600
        print(f"修正 {power_outliers} 个power异常值")
    
    # 处理缺失值
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'price':
            train_data[col] = train_data[col].fillna(train_data[col].median())
    
    # 分类特征编码
    categorical_cols = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    for col in categorical_cols:
        if col in train_data.columns:
            train_data[col] = train_data[col].fillna('unknown')
            le = LabelEncoder()
            train_data[col] = le.fit_transform(train_data[col].astype(str))
    
    # 价格异常值处理 - 使用保守策略
    price_q005 = train_data['price'].quantile(0.005)
    price_q995 = train_data['price'].quantile(0.995)
    valid_idx = (train_data['price'] >= price_q005) & (train_data['price'] <= price_q995)
    removed_count = len(train_data) - valid_idx.sum()
    train_data = train_data[valid_idx].reset_index(drop=True)
    print(f"移除 {removed_count} 个价格异常样本")
    
    # 创建车龄特征
    if 'regDate' in train_data.columns:
        current_year = 2020
        train_data['reg_year'] = train_data['regDate'] // 10000
        train_data['car_age'] = current_year - train_data['reg_year']
        train_data['car_age'] = np.maximum(train_data['car_age'], 1)
        
        # 年均里程数特征
        if 'kilometer' in train_data.columns:
            train_data['km_per_year'] = train_data['kilometer'] / train_data['car_age']
    
    # 准备建模数据
    feature_cols = [col for col in train_data.columns if col != 'price']
    X = train_data[feature_cols]
    y = train_data['price']
    
    print(f"预处理完成: X={X.shape}, y={y.shape}")
    return X, y

def plot_learning_curve_analysis(X, y, ax):
    """🔍 学习曲线分析 - 判断训练/验证Gap"""
    print("1️⃣ 生成学习曲线...")
    
    # 控制模型复杂度 - 基于记忆规范
    rf_model = RandomForestRegressor(
        n_estimators=150,  # 控制在300以内
        max_depth=18,      # 控制在20以内
        min_samples_split=12,
        min_samples_leaf=6,
        random_state=42,
        n_jobs=-1
    )
    
    # 计算学习曲线
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        rf_model, X, y, 
        cv=3,  # 加快计算速度
        train_sizes=train_sizes,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    # 转换为MAE
    train_mae = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mae = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # 绘制曲线
    ax.plot(train_sizes, train_mae, 'o-', color='#2E86AB', linewidth=2.5, 
            label='训练集MAE', markersize=6)
    ax.fill_between(train_sizes, train_mae - train_std, train_mae + train_std, 
                    alpha=0.2, color='#2E86AB')
    
    ax.plot(train_sizes, val_mae, 'o-', color='#F24236', linewidth=2.5, 
            label='验证集MAE', markersize=6)
    ax.fill_between(train_sizes, val_mae - val_std, val_mae + val_std, 
                    alpha=0.2, color='#F24236')
    
    # 样式设置
    ax.set_xlabel('训练样本数量', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE (平均绝对误差)', fontsize=11, fontweight='bold')
    ax.set_title('🔍 学习曲线分析 - 训练vs验证Gap', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 分析结果
    final_gap = val_mae[-1] - train_mae[-1]
    final_train_mae = train_mae[-1]
    final_val_mae = val_mae[-1]
    
    # 添加分析文本
    status_color = '#28a745' if final_gap < 80 else '#ffc107' if final_gap < 150 else '#dc3545'
    status_text = '✅ 良好' if final_gap < 80 else '⚖️ 适中' if final_gap < 150 else '⚠️ 过拟合'
    
    ax.text(0.05, 0.95, f'Gap: {final_gap:.1f}', transform=ax.transAxes, 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax.text(0.05, 0.88, f'状态: {status_text}', transform=ax.transAxes, 
            fontsize=10, color=status_color, fontweight='bold')
    
    return final_train_mae, final_val_mae, final_gap

def plot_convergence_analysis(X, y, ax):
    """🌳 收敛分析 - 模拟迭代过程"""
    print("2️⃣ 生成收敛分析...")
    
    # 树数量范围 - 控制在300以内
    n_estimators_range = [10, 25, 50, 75, 100, 125, 150, 200, 250, 300]
    
    train_scores = []
    val_scores = []
    
    for n_est in n_estimators_range:
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=18,
            min_samples_split=12,
            min_samples_leaf=6,
            random_state=42,
            n_jobs=-1
        )
        
        # 交叉验证得分
        cv_scores = cross_val_score(rf, X, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        val_scores.append(-cv_scores.mean())
        
        # 训练集得分
        rf.fit(X, y)
        train_pred = rf.predict(X)
        train_mae = mean_absolute_error(y, train_pred)
        train_scores.append(train_mae)
    
    # 绘制收敛曲线
    ax.plot(n_estimators_range, train_scores, 'o-', color='#2E86AB', 
            linewidth=2.5, label='训练集MAE', markersize=6)
    ax.plot(n_estimators_range, val_scores, 'o-', color='#F24236', 
            linewidth=2.5, label='验证集MAE', markersize=6)
    
    # 找到最佳点
    best_idx = np.argmin(val_scores)
    best_n_est = n_estimators_range[best_idx]
    best_val_mae = val_scores[best_idx]
    
    # 标记最佳点
    ax.axvline(x=best_n_est, color='#28a745', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(best_n_est + 15, best_val_mae + 10, f'最佳: {best_n_est}棵树', 
            fontsize=9, color='#28a745', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    ax.set_xlabel('树的数量 (模拟训练迭代)', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
    ax.set_title('🌳 收敛分析 - 找到最佳树数量', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return best_n_est, best_val_mae, val_scores

def plot_feature_importance(X, y, ax):
    """📊 特征重要性分析"""
    print("3️⃣ 生成特征重要性分析...")
    
    # 训练模型获取特征重要性
    rf = RandomForestRegressor(n_estimators=150, max_depth=18, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # 创建重要性数据框
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True).tail(15)  # 取前15个
    
    # 绘制水平条形图
    bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))
    
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('特征重要性', fontsize=11, fontweight='bold')
    ax.set_title('📊 Top15 重要特征分析', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    return importance_df

def plot_parameter_validation(X, y, ax):
    """⚖️ 参数验证曲线 - 找最优参数"""
    print("4️⃣ 生成参数验证分析...")
    
    # 测试max_depth参数
    depth_range = [8, 12, 15, 18, 20, 25, 30]
    
    train_scores, val_scores = validation_curve(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        X, y, param_name='max_depth', param_range=depth_range,
        cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    
    train_mae = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mae = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # 绘制验证曲线
    ax.plot(depth_range, train_mae, 'o-', color='#2E86AB', 
            linewidth=2.5, label='训练集MAE', markersize=6)
    ax.fill_between(depth_range, train_mae - train_std, train_mae + train_std, 
                    alpha=0.2, color='#2E86AB')
    
    ax.plot(depth_range, val_mae, 'o-', color='#F24236', 
            linewidth=2.5, label='验证集MAE', markersize=6)
    ax.fill_between(depth_range, val_mae - val_std, val_mae + val_std, 
                    alpha=0.2, color='#F24236')
    
    # 找到最佳深度
    best_idx = np.argmin(val_mae)
    best_depth = depth_range[best_idx]
    best_mae = val_mae[best_idx]
    
    ax.axvline(x=best_depth, color='#28a745', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(best_depth + 1, best_mae + 5, f'最佳深度: {best_depth}', 
            fontsize=9, color='#28a745', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    ax.set_xlabel('最大深度', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
    ax.set_title('⚖️ 参数验证 - 最优深度搜索', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return best_depth, best_mae

def plot_residual_analysis(X, y, ax):
    """📈 残差分析 - 预测质量检查"""
    print("5️⃣ 生成残差分析...")
    
    # 使用交叉验证预测
    rf = RandomForestRegressor(n_estimators=150, max_depth=18, random_state=42, n_jobs=-1)
    y_pred = cross_val_predict(rf, X, y, cv=3)
    
    # 计算残差
    residuals = y - y_pred
    
    # 绘制残差vs预测值散点图
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color='#A663CC')
    ax.axhline(y=0, color='#F24236', linestyle='--', alpha=0.8, linewidth=2)
    
    ax.set_xlabel('预测值', fontsize=11, fontweight='bold')
    ax.set_ylabel('残差 (真实值 - 预测值)', fontsize=11, fontweight='bold')
    ax.set_title('📈 残差分析 - 预测质量检查', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 计算关键指标
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    # 添加指标文本
    metrics_text = f'MAE: {mae:.1f}\nRMSE: {rmse:.1f}\nR²: {r2:.3f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
            verticalalignment='top')
    
    return mae, rmse, r2, residuals

def generate_analysis_report(results):
    """生成分析报告"""
    print("\n" + "="*60)
    print("🎯 随机森林训练状态分析报告")
    print("="*60)
    
    # 解析结果
    final_train_mae, final_val_mae, gap = results['learning_curve']
    best_n_est, best_val_mae, _ = results['convergence']
    importance_df = results['feature_importance']
    best_depth, _ = results['parameter_validation']
    mae, rmse, r2, _ = results['residual_analysis']
    
    print(f"\n1️⃣ 📈 学习曲线分析:")
    print(f"   训练MAE: {final_train_mae:.1f}")
    print(f"   验证MAE: {final_val_mae:.1f}")
    print(f"   Gap: {gap:.1f}")
    
    if gap > 150:
        print("   ❌ 模型过拟合严重！建议:")
        print("      • 增加min_samples_split (15→20)")
        print("      • 减少max_depth (18→15)")
        print("      • 增加min_samples_leaf")
    elif gap > 80:
        print("   ⚠️  存在轻微过拟合，建议微调参数")
    else:
        print("   ✅ 模型泛化能力良好")
    
    print(f"\n2️⃣ 🌳 收敛分析:")
    print(f"   最佳树数量: {best_n_est}")
    print(f"   最佳验证MAE: {best_val_mae:.1f}")
    
    if best_n_est >= 250:
        print("   💡 可考虑增加更多树来进一步优化")
    else:
        print("   ✅ 树数量已基本收敛")
    
    print(f"\n3️⃣ 📊 特征重要性:")
    print("   Top5重要特征:")
    for _, row in importance_df.tail(5).iloc[::-1].iterrows():
        print(f"   • {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n4️⃣ ⚖️  参数优化:")
    print(f"   建议最大深度: {best_depth}")
    
    print(f"\n5️⃣ 📈 模型性能:")
    print(f"   MAE: {mae:.1f}")
    print(f"   RMSE: {rmse:.1f}")
    print(f"   R²: {r2:.3f}")
    
    # 总体建议
    print(f"\n🎯 总体建议:")
    if gap > 100 and mae > 600:
        print("  ❌ 优先级: 解决过拟合 > 提升性能")
        print("  📝 具体行动: 增加正则化，减少模型复杂度")
    elif mae > 600:
        print("  🚀 优先级: 提升模型性能")
        print("  📝 具体行动: 优化特征工程，尝试参数调优")
    else:
        print("  ✅ 模型状态良好，可进行最后的微调")
        print("  📝 具体行动: 保存模型，准备集成")

def main():
    """主函数 - 生成完整的训练状态分析"""
    print("🎯 开始随机森林完整训练状态分析")
    print("="*60)
    
    # 加载数据
    X, y = load_and_preprocess_data()
    
    # 创建结果保存目录
    results_dir = get_project_path('prediction_result')
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建大图 - 2x3布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🎯 随机森林训练状态完整分析', fontsize=16, fontweight='bold', y=0.95)
    
    # 存储分析结果
    results = {}
    
    # 1. 学习曲线分析
    results['learning_curve'] = plot_learning_curve_analysis(X, y, axes[0, 0])
    
    # 2. 收敛分析
    results['convergence'] = plot_convergence_analysis(X, y, axes[0, 1])
    
    # 3. 特征重要性
    results['feature_importance'] = plot_feature_importance(X, y, axes[0, 2])
    
    # 4. 参数验证
    results['parameter_validation'] = plot_parameter_validation(X, y, axes[1, 0])
    
    # 5. 残差分析
    results['residual_analysis'] = plot_residual_analysis(X, y, axes[1, 1])
    
    # 6. 添加总结文本
    axes[1, 2].axis('off')
    summary_text = """
    🎯 训练状态判断标准
    
    ✅ 模型已达最佳状态:
    • 训练/验证Gap < 80
    • 验证曲线趋于平缓
    • 残差随机分布
    
    ⚠️ 需要调优:
    • Gap 80-150: 微调参数
    • Gap > 150: 过拟合，增加正则化
    
    🚀 可继续优化:
    • 验证性能还在提升
    • 收敛曲线未达平台期
    • 特征工程有改进空间
    
    💡 类似深度学习的早停:
    观察验证MAE是否到达最低点
    """
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    
    # 保存图表
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(results_dir, f'rf_complete_analysis_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 完整分析图表已保存: {save_path}")
    
    plt.show()
    
    # 生成分析报告
    generate_analysis_report(results)
    
    print(f"\n🎉 分析完成！已生成5个核心诊断图表")
    print(f"📁 图表保存位置: {save_path}")
    
    return results

if __name__ == "__main__":
    analysis_results = main()