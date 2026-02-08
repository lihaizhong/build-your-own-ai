"""
模型诊断脚本 - 分析本地验证与线上提交差异的原因
主要功能:
1. 训练集/验证集/测试集的价格分布对比
2. 各个特征在三个数据集上的分布对比
3. 模型预测值与真实值的散点图
4. 残差分析
5. 特征重要性分析
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import warnings
from ...shared import get_project_path

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']  # macOS使用Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def get_user_data_path(*paths):
    """获取用户数据路径"""
    return get_project_path('user_data', *paths)

def load_and_preprocess_data():
    """加载并预处理数据(与原模型保持一致)"""
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 合并数据进行统一预处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 处理power异常值
    if 'power' in all_df.columns:
        all_df['power'] = np.clip(all_df['power'], 0, 600)
    
    # 分类特征缺失值处理
    for col in ['fuelType', 'gearbox', 'bodyType']:
        mode_value = all_df[col].mode()
        if len(mode_value) > 0:
            all_df[col] = all_df[col].fillna(mode_value.iloc[0])
        all_df[f'{col}_missing'] = (all_df[col].isnull()).astype(int)

    model_mode = all_df['model'].mode()
    if len(model_mode) > 0:
        all_df['model'] = all_df['model'].fillna(model_mode.iloc[0])

    # 特征工程
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year
    all_df['car_age'] = all_df['car_age'].fillna(0).astype(int)
    all_df.drop(columns=['regDate'], inplace=True)
    
    if 'power' in all_df.columns:
        all_df['power_age_ratio'] = all_df['power'] / (all_df['car_age'] + 1)
    else:
        all_df['power_age_ratio'] = 0
    
    # 品牌统计特征
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
        brand_stats['smooth_mean'] = (brand_stats['mean'] * brand_stats['count'] + all_df['price'].mean() * 10) / (brand_stats['count'] + 10)
        brand_map: dict = brand_stats.set_index('brand')['smooth_mean'].to_dict()  # type: ignore
        all_df['brand_avg_price'] = all_df['brand'].map(brand_map)  # type: ignore
    
    # 标签编码
    categorical_cols = ['model', 'brand', 'fuelType', 'gearbox', 'bodyType']
    for col in categorical_cols:
        if col in all_df.columns:
            le = LabelEncoder()
            all_df[col] = le.fit_transform(all_df[col].astype(str))
    
    # 填充数值型缺失值
    numeric_cols = all_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        has_null = bool(all_df[col].isnull().any())  # type: ignore[arg-type]
        if has_null:
            median_value = all_df[col].median()
            all_df[col] = all_df[col].fillna(median_value)
    
    # 重新分离
    train_df = all_df.iloc[:len(train_df)].copy()
    test_df = all_df.iloc[len(train_df):].copy()
    
    # 删除训练集价格异常值
    Q1 = train_df['price'].quantile(0.25)
    Q3 = train_df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train_df = train_df[(train_df['price'] >= lower_bound) & (train_df['price'] <= upper_bound)]
    
    print(f"清理后训练集: {train_df.shape}")
    print(f"清理后测试集: {test_df.shape}")
    
    return train_df, test_df

def create_features(df):
    """创建高级特征"""
    df = df.copy()
    
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 3, 6, 10, float('inf')], labels=['new', 'medium', 'old', 'vintage'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    if 'power' in df.columns:
        df['power_segment'] = pd.cut(df['power'], bins=[-1, 50, 100, 150, 200, float('inf')], labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        df['power_segment'] = df['power_segment'].cat.codes
    else:
        df['power_segment'] = 0
    
    return df

def diagnose_model():
    """诊断模型问题"""
    print("=" * 60)
    print("开始模型诊断...")
    print("=" * 60)
    
    # 加载数据
    train_df, test_df = load_and_preprocess_data()
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    y_train_log = np.log1p(y_train)
    
    # 分割验证集
    X_tr, X_val, y_tr_log, y_val_log = train_test_split(
        X_train, y_train_log, test_size=0.2, random_state=42
    )
    
    # 训练模型
    print("\n训练模型...")
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_split=50, 
        min_samples_leaf=30, max_features='sqrt', random_state=42, n_jobs=-1  # type: ignore[arg-type]
    )
    
    lgb_params = {
        'objective': 'mae', 'metric': 'mae', 'boosting_type': 'gbdt',
        'num_leaves': 64, 'max_depth': 6, 'min_data_in_leaf': 50,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'random_state': 42
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=300)
    ridge_model = Ridge(alpha=1.0)
    
    # 训练基模型
    rf_model.fit(X_tr, y_tr_log)
    lgb_model.fit(X_tr, y_tr_log)
    ridge_model.fit(X_tr, y_tr_log)
    
    # 验证集预测
    rf_pred_val = np.expm1(rf_model.predict(X_val))
    lgb_pred_val = np.expm1(np.array(lgb_model.predict(X_val)))
    ridge_pred_val = np.expm1(ridge_model.predict(X_val))
    
    # Stacking
    stack_train = np.column_stack([rf_pred_val, lgb_pred_val, ridge_pred_val])
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stack_train, np.expm1(y_val_log))
    final_pred_val = meta_model.predict(stack_train)
    
    # 全量训练
    rf_model.fit(X_train, y_train_log)
    lgb_model.fit(X_train, y_train_log)
    ridge_model.fit(X_train, y_train_log)
    
    # 测试集预测
    rf_pred_test = np.expm1(rf_model.predict(X_test))
    lgb_pred_test = np.expm1(np.array(lgb_model.predict(X_test)))
    ridge_pred_test = np.expm1(ridge_model.predict(X_test))
    
    final_stack_test = np.column_stack([rf_pred_test, lgb_pred_test, ridge_pred_test])
    final_pred_test = meta_model.predict(final_stack_test)
    
    # 计算MAE
    val_mae = mean_absolute_error(np.expm1(y_val_log), final_pred_val)
    print(f"\n本地验证 MAE: {val_mae:.2f}")
    
    # 创建诊断图表目录
    diagnosis_dir = get_project_path('user_data', 'modeling_v1')
    os.makedirs(diagnosis_dir, exist_ok=True)
    
    # ========== 图表1: 价格分布对比 ==========
    print("\n绘制图表1: 价格分布对比...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 训练集价格分布
    axes[0, 0].hist(y_train, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title(f'训练集价格分布 (n={len(y_train)})', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('价格', fontsize=12)
    axes[0, 0].set_ylabel('频数', fontsize=12)
    axes[0, 0].axvline(y_train.mean(), color='red', linestyle='--', label=f'均值: {y_train.mean():.0f}')
    axes[0, 0].axvline(y_train.median(), color='green', linestyle='--', label=f'中位数: {y_train.median():.0f}')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 验证集价格分布
    y_val = np.expm1(y_val_log)
    axes[0, 1].hist(y_val, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_title(f'验证集价格分布 (n={len(y_val)})', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('价格', fontsize=12)
    axes[0, 1].set_ylabel('频数', fontsize=12)
    axes[0, 1].axvline(y_val.mean(), color='red', linestyle='--', label=f'均值: {y_val.mean():.0f}')
    axes[0, 1].axvline(np.median(y_val), color='green', linestyle='--', label=f'中位数: {np.median(y_val):.0f}')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 测试集预测值分布
    axes[1, 0].hist(final_pred_test, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].set_title(f'测试集预测值分布 (n={len(final_pred_test)})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('预测价格', fontsize=12)
    axes[1, 0].set_ylabel('频数', fontsize=12)
    axes[1, 0].axvline(final_pred_test.mean(), color='red', linestyle='--', label=f'均值: {final_pred_test.mean():.0f}')
    axes[1, 0].axvline(np.median(final_pred_test), color='green', linestyle='--', label=f'中位数: {np.median(final_pred_test):.0f}')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 三者对比（箱线图）
    data_to_plot = [y_train, y_val, final_pred_test]
    axes[1, 1].boxplot(data_to_plot, labels=['训练集', '验证集', '测试集预测'], patch_artist=True)
    axes[1, 1].set_title('价格分布箱线图对比', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('价格', fontsize=12)
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(diagnosis_dir, '1_价格分布对比.png'), dpi=300, bbox_inches='tight')
    print(f"已保存: {os.path.join(diagnosis_dir, '1_价格分布对比.png')}")
    plt.close()
    
    # ========== 图表2: 预测值vs真实值 ==========
    print("\n绘制图表2: 预测值vs真实值...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 验证集散点图
    axes[0].scatter(y_val, final_pred_val, alpha=0.5, s=10)
    axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='理想拟合线')
    axes[0].set_xlabel('真实价格', fontsize=12)
    axes[0].set_ylabel('预测价格', fontsize=12)
    axes[0].set_title(f'验证集: 预测vs真实 (MAE={val_mae:.2f})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 残差分布
    residuals = final_pred_val - y_val
    axes[1].hist(residuals, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1].axvline(0, color='black', linestyle='--', lw=2)
    axes[1].set_xlabel('残差 (预测值 - 真实值)', fontsize=12)
    axes[1].set_ylabel('频数', fontsize=12)
    axes[1].set_title(f'验证集残差分布 (均值={residuals.mean():.2f})', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(diagnosis_dir, '2_预测质量分析.png'), dpi=300, bbox_inches='tight')
    print(f"已保存: {os.path.join(diagnosis_dir, '2_预测质量分析.png')}")
    plt.close()
    
    # ========== 图表3: 关键特征分布对比 ==========
    print("\n绘制图表3: 关键特征分布对比...")
    key_features = ['power', 'car_age', 'kilometer', 'v_0', 'v_1', 'v_2']
    existing_features = [f for f in key_features if f in X_train.columns]
    
    if len(existing_features) >= 6:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
    else:
        n_cols = min(3, len(existing_features))
        n_rows = (len(existing_features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if len(existing_features) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
    
    for idx, feature in enumerate(existing_features):
        train_vals = X_train[feature]
        test_vals = X_test[feature]
        
        axes[idx].hist(train_vals, bins=50, alpha=0.5, label='训练集', color='blue', density=True)
        axes[idx].hist(test_vals, bins=50, alpha=0.5, label='测试集', color='red', density=True)
        axes[idx].set_title(f'{feature} 分布对比', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('密度', fontsize=10)
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
        
        # 添加统计信息
        train_mean = train_vals.mean()
        test_mean = test_vals.mean()
        axes[idx].text(0.02, 0.98, f'训练均值: {train_mean:.2f}\n测试均值: {test_mean:.2f}', 
                      transform=axes[idx].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    # 隐藏多余的子图
    for idx in range(len(existing_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(diagnosis_dir, '3_关键特征分布对比.png'), dpi=300, bbox_inches='tight')
    print(f"已保存: {os.path.join(diagnosis_dir, '3_关键特征分布对比.png')}")
    plt.close()
    
    # ========== 图表4: 特征重要性 ==========
    print("\n绘制图表4: 特征重要性...")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(feature_importance)), feature_importance['importance'], color='steelblue')
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'])
    ax.set_xlabel('重要性', fontsize=12)
    ax.set_title('LightGBM Top 20 特征重要性', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(diagnosis_dir, '4_特征重要性.png'), dpi=300, bbox_inches='tight')
    print(f"已保存: {os.path.join(diagnosis_dir, '4_特征重要性.png')}")
    plt.close()
    
    # ========== 图表5: 预测误差分析 ==========
    print("\n绘制图表5: 预测误差分析...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 按真实价格区间分析误差
    price_bins = pd.cut(y_val, bins=10)
    error_by_price = pd.DataFrame({
        'price_bin': price_bins,
        'error': np.abs(residuals)
    })
    error_stats = error_by_price.groupby('price_bin')['error'].mean().sort_index()
    
    axes[0].bar(range(len(error_stats)), error_stats.values, color='coral')
    axes[0].set_xticks(range(len(error_stats)))
    axes[0].set_xticklabels([str(interval) for interval in error_stats.index], rotation=45, ha='right')
    axes[0].set_xlabel('价格区间', fontsize=12)
    axes[0].set_ylabel('平均绝对误差', fontsize=12)
    axes[0].set_title('不同价格区间的预测误差', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    
    # 残差vs真实价格
    axes[1].scatter(y_val, residuals, alpha=0.5, s=10, c=np.abs(residuals), cmap='coolwarm')
    axes[1].axhline(0, color='black', linestyle='--', lw=2)
    axes[1].set_xlabel('真实价格', fontsize=12)
    axes[1].set_ylabel('残差', fontsize=12)
    axes[1].set_title('残差 vs 真实价格', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(diagnosis_dir, '5_误差分析.png'), dpi=300, bbox_inches='tight')
    print(f"已保存: {os.path.join(diagnosis_dir, '5_误差分析.png')}")
    plt.close()
    
    # ========== 统计报告 ==========
    print("\n" + "=" * 60)
    print("诊断统计报告")
    print("=" * 60)
    print(f"\n【价格统计】")
    print(f"训练集价格 - 均值: {y_train.mean():.2f}, 中位数: {y_train.median():.2f}, 标准差: {y_train.std():.2f}")
    print(f"验证集价格 - 均值: {y_val.mean():.2f}, 中位数: {np.median(y_val):.2f}, 标准差: {y_val.std():.2f}")
    print(f"测试集预测 - 均值: {final_pred_test.mean():.2f}, 中位数: {np.median(final_pred_test):.2f}, 标准差: {final_pred_test.std():.2f}")
    
    print(f"\n【预测质量】")
    print(f"验证集 MAE: {val_mae:.2f}")
    print(f"残差均值: {residuals.mean():.2f} (应接近0)")
    print(f"残差标准差: {residuals.std():.2f}")
    
    print(f"\n【可能的问题】")
    train_test_mean_diff = abs(y_train.mean() - final_pred_test.mean())
    train_test_std_diff = abs(y_train.std() - final_pred_test.std())
    
    if train_test_mean_diff > 1000:
        print(f"⚠️  训练集与测试集预测的均值差异过大: {train_test_mean_diff:.2f}")
    if train_test_std_diff > 500:
        print(f"⚠️  训练集与测试集预测的标准差差异过大: {train_test_std_diff:.2f}")
    if val_mae < 500 and abs(residuals.mean()) > 50:
        print(f"⚠️  残差均值偏离0较多,可能存在系统性偏差")
    
    print(f"\n所有诊断图表已保存至: {diagnosis_dir}")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_model()
