# -*- coding: utf-8 -*-
"""
随机森林训练状态分析 - 简化版本
专注于核心5个诊断图表，判断模型是否达到最佳状态
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (learning_curve, validation_curve, cross_val_score, 
                                   cross_val_predict)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

def load_data():
    """快速加载数据"""
    print("🔄 加载数据...")
    
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    train_data = pd.read_csv(train_path, sep=' ')
    print(f"数据大小: {train_data.shape}")
    
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
            le = LabelEncoder()
            train_data[col] = le.fit_transform(train_data[col].astype(str))
    
    # 处理异常值
    if 'power' in train_data.columns:
        train_data.loc[train_data['power'] > 600, 'power'] = 600
    
    # 价格异常值
    price_q01 = train_data['price'].quantile(0.01)
    price_q99 = train_data['price'].quantile(0.99)
    train_data = train_data[(train_data['price'] >= price_q01) & 
                           (train_data['price'] <= price_q99)]
    
    # 车龄特征
    if 'regDate' in train_data.columns:
        train_data['car_age'] = 2020 - (train_data['regDate'] // 10000)
        train_data['car_age'] = np.maximum(train_data['car_age'], 1)
    
    feature_cols = [col for col in train_data.columns if col != 'price']
    X = train_data[feature_cols]
    y = train_data['price']
    
    print(f"预处理完成: {X.shape}")
    return X, y

def main():
    """主函数"""
    print("🎯 随机森林训练状态分析")
    print("="*50)
    
    # 加载数据
    X, y = load_data()
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('随机森林训练状态完整分析', fontsize=16, fontweight='bold')
    
    results = {}
    
    # 1. 学习曲线
    print("1️⃣ 学习曲线分析...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    train_sizes = np.linspace(0.1, 1.0, 8)
    
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            rf_model, X, y, cv=3, train_sizes=train_sizes,
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        train_mae = -train_scores.mean(axis=1)
        val_mae = -val_scores.mean(axis=1)
        
        axes[0,0].plot(train_sizes, train_mae, 'o-', label='训练集MAE', color='blue')
        axes[0,0].plot(train_sizes, val_mae, 'o-', label='验证集MAE', color='red')
        axes[0,0].set_title('学习曲线 - 判断Gap')
        axes[0,0].set_xlabel('训练样本数')
        axes[0,0].set_ylabel('MAE')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        gap = val_mae[-1] - train_mae[-1]
        results['gap'] = gap
        
        print(f"   训练MAE: {train_mae[-1]:.1f}")
        print(f"   验证MAE: {val_mae[-1]:.1f}")
        print(f"   Gap: {gap:.1f}")
        
    except Exception as e:
        print(f"学习曲线计算错误: {e}")
        axes[0,0].text(0.5, 0.5, '学习曲线计算失败', ha='center', va='center')
    
    # 2. 收敛分析
    print("2️⃣ 收敛分析...")
    n_estimators_range = [25, 50, 75, 100, 150, 200, 250]
    val_scores = []
    
    try:
        for n_est in n_estimators_range:
            rf = RandomForestRegressor(n_estimators=n_est, max_depth=15, random_state=42, n_jobs=-1)
            scores = cross_val_score(rf, X, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            val_scores.append(-scores.mean())
        
        axes[0,1].plot(n_estimators_range, val_scores, 'o-', color='green')
        axes[0,1].set_title('收敛分析 - 树数量')
        axes[0,1].set_xlabel('树数量')
        axes[0,1].set_ylabel('验证集MAE')
        axes[0,1].grid(True, alpha=0.3)
        
        best_n = n_estimators_range[np.argmin(val_scores)]
        best_mae = min(val_scores)
        axes[0,1].axvline(x=best_n, color='red', linestyle='--')
        
        results['best_n_estimators'] = best_n
        print(f"   最佳树数量: {best_n}")
        print(f"   最佳MAE: {best_mae:.1f}")
        
    except Exception as e:
        print(f"收敛分析错误: {e}")
        axes[0,1].text(0.5, 0.5, '收敛分析失败', ha='center', va='center')
    
    # 3. 特征重要性
    print("3️⃣ 特征重要性...")
    try:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)
        
        axes[0,2].barh(range(len(importance_df)), importance_df['importance'])
        axes[0,2].set_yticks(range(len(importance_df)))
        axes[0,2].set_yticklabels(importance_df['feature'])
        axes[0,2].set_title('Top15 重要特征')
        axes[0,2].set_xlabel('重要性')
        
        print("   Top5特征:")
        for _, row in importance_df.tail(5).iterrows():
            print(f"   • {row['feature']}: {row['importance']:.4f}")
            
    except Exception as e:
        print(f"特征重要性错误: {e}")
        axes[0,2].text(0.5, 0.5, '特征重要性失败', ha='center', va='center')
    
    # 4. 参数验证
    print("4️⃣ 参数验证...")
    try:
        depth_range = [8, 12, 15, 18, 20, 25]
        train_scores, val_scores = validation_curve(
            RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            X, y, param_name='max_depth', param_range=depth_range,
            cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        train_mae = -train_scores.mean(axis=1)
        val_mae = -val_scores.mean(axis=1)
        
        axes[1,0].plot(depth_range, train_mae, 'o-', label='训练集', color='blue')
        axes[1,0].plot(depth_range, val_mae, 'o-', label='验证集', color='red')
        axes[1,0].set_title('参数验证 - 最大深度')
        axes[1,0].set_xlabel('max_depth')
        axes[1,0].set_ylabel('MAE')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        best_depth = depth_range[np.argmin(val_mae)]
        axes[1,0].axvline(x=best_depth, color='green', linestyle='--')
        
        print(f"   最佳深度: {best_depth}")
        
    except Exception as e:
        print(f"参数验证错误: {e}")
        axes[1,0].text(0.5, 0.5, '参数验证失败', ha='center', va='center')
    
    # 5. 残差分析
    print("5️⃣ 残差分析...")
    try:
        rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        y_pred = cross_val_predict(rf, X, y, cv=3)
        residuals = y - y_pred
        
        axes[1,1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[1,1].axhline(y=0, color='red', linestyle='--')
        axes[1,1].set_title('残差分析')
        axes[1,1].set_xlabel('预测值')
        axes[1,1].set_ylabel('残差')
        axes[1,1].grid(True, alpha=0.3)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        print(f"   MAE: {mae:.1f}")
        print(f"   RMSE: {rmse:.1f}")
        print(f"   R²: {r2:.3f}")
        
    except Exception as e:
        print(f"残差分析错误: {e}")
        axes[1,1].text(0.5, 0.5, '残差分析失败', ha='center', va='center')
    
    # 6. 总结
    axes[1,2].axis('off')
    summary_text = """
训练状态判断:

✅ 良好状态:
• Gap < 80
• 曲线趋于平缓
• 残差随机分布

⚠️ 需要调优:
• Gap 80-150
• 验证MAE还在下降

❌ 过拟合:
• Gap > 150
• 训练误差远小于验证误差

💡 优化建议:
• 观察验证MAE最低点
• 类似深度学习早停原理
"""
    
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
    
    # 保存图表
    results_dir = get_project_path('prediction_result')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(results_dir, f'rf_analysis_{timestamp}.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 分析图表已保存: {save_path}")
    
    # 显示图表
    plt.show()
    
    # 生成总结报告
    print("\n" + "="*50)
    print("🎯 训练状态分析总结")
    print("="*50)
    
    if 'gap' in results:
        gap = results['gap']
        if gap > 150:
            print("❌ 模型过拟合严重，需要:")
            print("   • 增加正则化（min_samples_split, min_samples_leaf）")
            print("   • 减少模型复杂度（max_depth）")
        elif gap > 80:
            print("⚠️ 轻微过拟合，建议微调参数")
        else:
            print("✅ 模型泛化能力良好")
    
    if 'best_n_estimators' in results:
        best_n = results['best_n_estimators']
        if best_n >= 200:
            print(f"💡 建议树数量: {best_n}，可考虑进一步增加")
        else:
            print(f"✅ 树数量已收敛: {best_n}")
    
    print(f"\n🎉 分析完成！")
    print(f"图表包含了类似深度学习'训练损失vs验证损失'的判断逻辑")
    print(f"可以据此判断随机森林是否达到最佳训练状态")
    
    return save_path

if __name__ == "__main__":
    result_path = main()