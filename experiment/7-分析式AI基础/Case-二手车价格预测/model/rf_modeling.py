# -*- coding: utf-8 -*-
"""
针对性优化脚本 - 基于深度分析结果
目标: 将MAE从698降到500以内
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold, learning_curve, validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_project_path(*paths):
    """获取项目路径的统一方法"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, *paths)

def load_and_optimize_data():
    """基于分析结果优化数据加载"""
    print("正在加载并优化数据...")
    
    # 加载原始数据 - 使用绝对路径
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    print(f"训练数据路径: {train_path}")
    print(f"测试数据路径: {test_path}")
    
    try:
        train_raw = pd.read_csv(train_path, sep=' ')
        test_raw = pd.read_csv(test_path, sep=' ')
    except Exception as e:
        print(f"数据加载错误: {e}")
        print("尝试使用逗号分隔符...")
        train_raw = pd.read_csv(train_path)
        test_raw = pd.read_csv(test_path)
    
    print(f"原始训练集: {train_raw.shape}")
    print(f"原始测试集: {test_raw.shape}")
    
    # 确保特征完全一致
    common_features = set(train_raw.columns) & set(test_raw.columns)
    feature_cols = [col for col in common_features if col != 'price']
    
    train_data = train_raw[feature_cols + ['price']].copy()
    test_data = test_raw[feature_cols].copy()
    
    # 1. 按规范处理power异常值 (发现143个>600的记录)
    print("处理power异常值...")
    if 'power' in train_data.columns:
        power_outliers_train = (train_data['power'] > 600).sum()
        train_data.loc[train_data['power'] > 600, 'power'] = 600
        print(f"训练集修正了 {power_outliers_train} 个power异常值")
    
    if 'power' in test_data.columns:
        power_outliers_test = (test_data['power'] > 600).sum()
        test_data.loc[test_data['power'] > 600, 'power'] = 600
        print(f"测试集修正了 {power_outliers_test} 个power异常值")
    
    # 2. 按规范处理缺失值
    print("按规范处理缺失值...")
    
    # 数值型特征用中位数填充
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'price':
            median_val = train_data[col].median()
            train_data[col] = train_data[col].fillna(median_val)
            if col in test_data.columns:
                test_data[col] = test_data[col].fillna(median_val)
    
    # 分类特征用众数填充
    categorical_cols = train_data.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if col != 'price':
            if len(train_data[col].mode()) > 0:
                mode_val = train_data[col].mode().iloc[0]
            else:
                mode_val = 'unknown'
            train_data[col] = train_data[col].fillna(mode_val)
            if col in test_data.columns:
                test_data[col] = test_data[col].fillna(mode_val)
    
    # 3. 处理分类特征编码问题 (model字段训练集独有3个值)
    print("优化分类特征编码...")
    categorical_features = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    
    for col in categorical_features:
        if col in train_data.columns and col in test_data.columns:
            # 合并训练和测试集进行统一编码，处理训练集独有值的问题
            all_values = pd.concat([
                train_data[col].astype(str), 
                test_data[col].astype(str)
            ]).unique()
            
            le = LabelEncoder()
            le.fit(all_values)
            
            train_data[col] = le.transform(train_data[col].astype(str))
            test_data[col] = le.transform(test_data[col].astype(str))
    
    # 4. 保守的价格异常值处理（避免过度删除高价样本）
    if 'price' in train_data.columns:
        # 使用更宽松的0.5%-99.5%范围
        price_q005 = train_data['price'].quantile(0.005)
        price_q995 = train_data['price'].quantile(0.995)
        
        valid_idx = (train_data['price'] >= price_q005) & (train_data['price'] <= price_q995)
        removed_count = len(train_data) - valid_idx.sum()
        train_data = train_data[valid_idx].reset_index(drop=True)
        
        print(f"价格范围: {train_data['price'].min():.2f} - {train_data['price'].max():.2f}")
        print(f"移除了 {removed_count} 个价格异常样本 ({removed_count/len(train_raw)*100:.2f}%)")
    
    print(f"优化后训练集: {train_data.shape}")
    print(f"优化后测试集: {test_data.shape}")
    
    return train_data, test_data

def create_targeted_features(train_data, test_data):
    """基于特征重要性分析创建针对性特征"""
    print("创建针对性特征...")
    
    # 5. 正确处理regDate - 按规范提取年份计算车龄
    if 'regDate' in train_data.columns:
        print("正确处理regDate时间特征...")
        current_year = 2020
        
        # 提取年份
        train_data['reg_year'] = train_data['regDate'] // 10000
        test_data['reg_year'] = test_data['regDate'] // 10000
        
        # 计算车龄
        train_data['car_age'] = current_year - train_data['reg_year']
        test_data['car_age'] = current_year - test_data['reg_year']
        
        # 确保车龄为正数
        train_data['car_age'] = np.maximum(train_data['car_age'], 1)
        test_data['car_age'] = np.maximum(test_data['car_age'], 1)
        
        # 提取月份（季节性特征）
        train_data['reg_month'] = (train_data['regDate'] % 10000) // 100
        test_data['reg_month'] = (test_data['regDate'] % 10000) // 100
        
        # 车龄分档（基于业务理解）
        age_bins = [0, 3, 7, 12, 20, 50]
        train_data['age_group'] = pd.cut(train_data['car_age'], bins=age_bins, labels=False).fillna(4).astype(int)
        test_data['age_group'] = pd.cut(test_data['car_age'], bins=age_bins, labels=False).fillna(4).astype(int)
    
    # 6. 基于重要特征创建交互特征
    if 'kilometer' in train_data.columns and 'car_age' in train_data.columns:
        # 年均里程数 (分析中发现这个特征有效)
        train_data['km_per_year'] = train_data['kilometer'] / train_data['car_age']
        test_data['km_per_year'] = test_data['kilometer'] / test_data['car_age']
        
        # 里程使用强度分类
        km_year_bins = [0, 8000, 18000, 35000, np.inf]
        train_data['usage_intensity'] = pd.cut(train_data['km_per_year'], bins=km_year_bins, labels=False).fillna(0).astype(int)
        test_data['usage_intensity'] = pd.cut(test_data['km_per_year'], bins=km_year_bins, labels=False).fillna(0).astype(int)
    
    # 7. 功率效率特征 (分析中排名第5)
    if 'power' in train_data.columns and 'kilometer' in train_data.columns:
        train_data['power_efficiency'] = train_data['power'] / (train_data['kilometer'] + 1)
        test_data['power_efficiency'] = test_data['power'] / (test_data['kilometer'] + 1)
        
        # 功率分档
        power_bins = [0, 75, 110, 150, 200, 600]
        train_data['power_level'] = pd.cut(train_data['power'], bins=power_bins, labels=False).fillna(0).astype(int)
        test_data['power_level'] = pd.cut(test_data['power'], bins=power_bins, labels=False).fillna(0).astype(int)
    
    # 8. 基于Top特征的交互 (v_0, v_12, v_3是最重要的)
    important_features = ['v_0', 'v_12', 'v_3']
    for feat in important_features:
        if feat in train_data.columns:
            # 与车龄的交互
            if 'car_age' in train_data.columns:
                train_data[f'{feat}_age_ratio'] = train_data[feat] / (train_data['car_age'] + 1)
                test_data[f'{feat}_age_ratio'] = test_data[feat] / (test_data['car_age'] + 1)
    
    # 9. 组合特征 - 基于业务理解
    if 'v_0' in train_data.columns and 'v_12' in train_data.columns:
        train_data['v0_v12_combo'] = train_data['v_0'] * train_data['v_12']
        test_data['v0_v12_combo'] = test_data['v_0'] * test_data['v_12']
    
    print(f"特征工程后训练集: {train_data.shape}")
    print(f"特征工程后测试集: {test_data.shape}")
    
    return train_data, test_data

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """绘制特征重要性图"""
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title(f'前{top_n}个重要特征的重要性分析', fontsize=14, fontweight='bold')
    plt.xlabel('特征重要性', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {save_path}")
    
    plt.show()
    return feature_importance

def plot_learning_curve(model, X, y, cv=5, save_path=None):
    """绘制学习曲线分析模型是否过拟合或欠拟合"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, 
        scoring='neg_mean_absolute_error', n_jobs=-1
    )
    
    train_mae = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mae = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mae, 'o-', color='blue', label='训练集MAE')
    plt.fill_between(train_sizes, train_mae - train_std, train_mae + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mae, 'o-', color='red', label='验证集MAE')
    plt.fill_between(train_sizes, val_mae - val_std, val_mae + val_std, alpha=0.1, color='red')
    
    plt.xlabel('训练样本数量', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('学习曲线分析 - 判断过拟合/欠拟合', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习曲线图已保存到: {save_path}")
    
    plt.show()
    
    # 分析结果
    final_gap = val_mae[-1] - train_mae[-1]
    print(f"\n学习曲线分析结果:")
    print(f"最终训练MAE: {train_mae[-1]:.4f}")
    print(f"最终验证MAE: {val_mae[-1]:.4f}")
    print(f"Gap: {final_gap:.4f}")
    
    if final_gap > 50:
        print("⚠️  模型可能存在过拟合，建议增加正则化或减少模型复杂度")
    elif final_gap < 20:
        print("🚀 模型泛化能力较好，可以考虑增加模型复杂度")
    else:
        print("✅ 模型复杂度较为合适")
    
    return train_mae, val_mae

def plot_validation_curve_analysis(X, y, param_name='max_depth', param_range=None, save_path=None):
    """绘制验证曲线分析参数优化空间"""
    if param_range is None:
        if param_name == 'max_depth':
            param_range = [5, 10, 15, 20, 25, 30, 35]
        elif param_name == 'n_estimators':
            param_range = [50, 100, 150, 200, 250, 300]
        elif param_name == 'min_samples_split':
            param_range = [2, 5, 10, 15, 20, 25]
        else:
            param_range = [1, 2, 3, 4, 5]
    
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    
    train_mae = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mae = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mae, 'o-', color='blue', label='训练集MAE')
    plt.fill_between(param_range, train_mae - train_std, train_mae + train_std, alpha=0.1, color='blue')
    plt.plot(param_range, val_mae, 'o-', color='red', label='验证集MAE')
    plt.fill_between(param_range, val_mae - val_std, val_mae + val_std, alpha=0.1, color='red')
    
    plt.xlabel(f'{param_name} 参数值', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title(f'{param_name} 参数验证曲线 - 寻找最优参数', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"验证曲线图已保存到: {save_path}")
    
    plt.show()
    
    # 找到最优参数
    best_idx = np.argmin(val_mae)
    best_param = param_range[best_idx]
    best_mae = val_mae[best_idx]
    
    print(f"\n{param_name} 参数优化结果:")
    print(f"最优参数: {best_param}")
    print(f"最优MAE: {best_mae:.4f}")
    
    return best_param, best_mae

def plot_residual_analysis(y_true, y_pred, save_path=None):
    """绘制残差分析图"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 残差 vs 预测值
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[0, 0].set_xlabel('预测值', fontsize=12)
    axes[0, 0].set_ylabel('残差 (真实值 - 预测值)', fontsize=12)
    axes[0, 0].set_title('残差 vs 预测值', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 残差分布直方图
    axes[0, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
    axes[0, 1].set_xlabel('残差', fontsize=12)
    axes[0, 1].set_ylabel('频次', fontsize=12)
    axes[0, 1].set_title('残差分布直方图', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. QQ图 - 检验残差是否符合正态分布
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('QQ图 - 残差正态性检验', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 真实值 vs 预测值
    axes[1, 1].scatter(y_true, y_pred, alpha=0.6, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1, 1].set_xlabel('真实值', fontsize=12)
    axes[1, 1].set_ylabel('预测值', fontsize=12)
    axes[1, 1].set_title('真实值 vs 预测值', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"残差分析图已保存到: {save_path}")
    
    plt.show()
    
    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n残差分析结果:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"残差平均值: {residuals.mean():.4f}")
    print(f"残差标准差: {residuals.std():.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'residuals': residuals}

def plot_price_distribution_comparison(y_train, ensemble_pred, save_path=None):
    """对比真实价格与预测价格的分布"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_train, bins=50, alpha=0.7, label='训练集真实价格', color='blue', edgecolor='black')
    plt.hist(ensemble_pred, bins=50, alpha=0.7, label='测试集预测价格', color='red', edgecolor='black')
    plt.xlabel('价格', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.title('价格分布对比', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([y_train, ensemble_pred], labels=['训练集真实价格', '测试集预测价格'])
    plt.ylabel('价格', fontsize=12)
    plt.title('价格分布箱线图', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"价格分布对比图已保存到: {save_path}")
    
    plt.show()
    
    print(f"\n价格分布统计对比:")
    print(f"训练集 - 平均: {y_train.mean():.2f}, 中位: {y_train.median():.2f}, 标准差: {y_train.std():.2f}")
    print(f"预测集 - 平均: {ensemble_pred.mean():.2f}, 中位: {np.median(ensemble_pred):.2f}, 标准差: {ensemble_pred.std():.2f}")

def create_optimized_rf_ensemble(X_train, y_train, X_test, enable_analysis=True):
    """创建优化的随机森林集成"""
    print("创建优化随机森林集成...")
    
    # 基于特征数量调整参数
    n_features = X_train.shape[1]
    
    # 优化的随机森林配置
    rf_models = [
        # 保守配置 - 防止过拟合
        RandomForestRegressor(
            n_estimators=200,
            max_depth=min(18, max(10, n_features//3)),
            min_samples_split=15,
            min_samples_leaf=8,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),
        # 平衡配置
        RandomForestRegressor(
            n_estimators=250,
            max_depth=min(22, max(12, n_features//2)),
            min_samples_split=12,
            min_samples_leaf=6,
            max_features='log2',
            bootstrap=True,
            random_state=123,
            n_jobs=-1
        ),
        # 复杂配置 - 但仍然保守
        RandomForestRegressor(
            n_estimators=180,
            max_depth=min(25, max(15, n_features//2)),
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.6,
            bootstrap=True,
            random_state=456,
            n_jobs=-1
        ),
        # ExtraTrees - 增加多样性
        ExtraTreesRegressor(
            n_estimators=200,
            max_depth=min(20, max(12, n_features//3)),
            min_samples_split=12,
            min_samples_leaf=6,
            max_features='sqrt',
            bootstrap=False,
            random_state=789,
            n_jobs=-1
        )
    ]
    
    # 训练集成模型
    predictions = []
    trained_models = []
    
    for i, model in enumerate(rf_models):
        print(f"训练模型 {i+1}/4...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.maximum(pred, 0)  # 确保非负
        predictions.append(pred)
        trained_models.append(model)
        
        # 打印预测范围
        print(f"  模型{i+1}预测范围: {pred.min():.2f} - {pred.max():.2f}")
    
    # 加权集成 - 给表现更好的模型更高权重
    weights = [0.3, 0.3, 0.25, 0.15]  # 根据经验调整
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    
    print(f"集成预测统计:")
    print(f"  均值: {ensemble_pred.mean():.2f}")
    print(f"  中位数: {np.median(ensemble_pred):.2f}")
    print(f"  范围: {ensemble_pred.min():.2f} - {ensemble_pred.max():.2f}")
    
    # 如果启用分析，进行详细分析
    if enable_analysis:
        print("\n开始模型分析...")
        
        # 创建结果保存目录
        results_dir = get_project_path('results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 使用第一个模型进行分析
        main_model = trained_models[0]
        feature_names = X_train.columns.tolist()
        
        # 1. 特征重要性分析
        print("1. 绘制特征重要性图...")
        importance_path = os.path.join(results_dir, f'feature_importance_{timestamp}.png')
        feature_importance = plot_feature_importance(
            main_model, feature_names, top_n=20, save_path=importance_path
        )
        
        # 2. 学习曲线分析
        print("2. 绘制学习曲线...")
        learning_path = os.path.join(results_dir, f'learning_curve_{timestamp}.png')
        train_mae, val_mae = plot_learning_curve(
            main_model, X_train, y_train, cv=3, save_path=learning_path
        )
        
        # 3. 参数验证曲线
        print("3. 绘制参数验证曲线...")
        validation_path = os.path.join(results_dir, f'validation_curve_{timestamp}.png')
        best_depth, best_mae = plot_validation_curve_analysis(
            X_train, y_train, param_name='max_depth', 
            param_range=[10, 15, 20, 25, 30, 35, 40], 
            save_path=validation_path
        )
        
        # 4. 残差分析（使用交叉验证预测）
        print("4. 进行残差分析...")
        from sklearn.model_selection import cross_val_predict
        cv_pred = cross_val_predict(main_model, X_train, y_train, cv=3)
        residual_path = os.path.join(results_dir, f'residual_analysis_{timestamp}.png')
        residual_stats = plot_residual_analysis(
            y_train, cv_pred, save_path=residual_path
        )
        
        # 5. 价格分布对比
        print("5. 绘制价格分布对比...")
        distribution_path = os.path.join(results_dir, f'price_distribution_{timestamp}.png')
        plot_price_distribution_comparison(
            y_train, ensemble_pred, save_path=distribution_path
        )
        
        print(f"\n📊 所有分析图表已保存到: {results_dir}")
        
        # 返回分析结果
        analysis_results = {
            'feature_importance': feature_importance,
            'best_depth': best_depth,
            'residual_stats': residual_stats,
            'final_cv_mae': val_mae[-1] if len(val_mae) > 0 else None
        }
        
        return ensemble_pred, analysis_results
    
    return ensemble_pred

def quick_cv_evaluation(X_train, y_train):
    """快速交叉验证评估"""
    print("快速交叉验证评估...")
    
    # 使用一个代表性模型进行快速评估
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=12,
        min_samples_leaf=6,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # 使用3折加快速度
    cv_scores = cross_val_score(rf, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"交叉验证MAE: {cv_mae:.4f} ± {cv_std:.4f}")
    return cv_mae

def main():
    """主函数"""
    print("开始针对性优化...")
    print("目标: MAE从698降到500以内")
    
    # 1. 优化数据加载
    train_data, test_data = load_and_optimize_data()
    
    # 2. 创建针对性特征
    train_data, test_data = create_targeted_features(train_data, test_data)
    
    # 3. 准备建模数据
    feature_cols = [col for col in train_data.columns if col != 'price']
    feature_cols = [col for col in feature_cols if col in test_data.columns]
    
    X_train = train_data[feature_cols]
    y_train = train_data['price']
    X_test = test_data[feature_cols]
    
    print(f"\n最终特征数量: {len(feature_cols)}")
    print(f"训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")
    
    # 4. 快速交叉验证评估
    cv_mae = quick_cv_evaluation(X_train, y_train)
    
    # 5. 创建优化集成（启用分析功能）
    result = create_optimized_rf_ensemble(X_train, y_train, X_test, enable_analysis=True)
    
    # 检查返回值类型
    if isinstance(result, tuple):
        ensemble_pred, analysis_results = result
        print(f"\n📊 模型分析完成！")
        print(f"最优深度建议: {analysis_results.get('best_depth', 'N/A')}")
        if analysis_results.get('final_cv_mae'):
            print(f"学习曲线最终MAE: {analysis_results['final_cv_mae']:.4f}")
    else:
        ensemble_pred = result
        print(f"\n⚠️ 跳过了模型分析")
    
    # 6. 保存结果 - 按规范保存
    submission = pd.DataFrame({
        'SaleID': range(len(ensemble_pred)),
        'price': ensemble_pred
    })
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 按规范保存到results目录
    results_dir = get_project_path('results')
    os.makedirs(results_dir, exist_ok=True)
    
    filename = os.path.join(results_dir, f'rf_result_{timestamp}.csv')
    submission.to_csv(filename, index=False)
    
    print(f"\n=== 针对性优化完成 ===")
    print(f"交叉验证MAE: {cv_mae:.4f}")
    print(f"结果已保存到: {filename}")
    print(f"预测均值: {ensemble_pred.mean():.2f}")
    print(f"预测范围: {ensemble_pred.min():.2f} - {ensemble_pred.max():.2f}")
    
    # 预测性能评估
    if cv_mae < 500:
        print("🎉 目标达成！交叉验证MAE < 500")
    else:
        print(f"⚠️  还需优化，当前MAE {cv_mae:.0f}，目标 < 500")
    
    return filename

if __name__ == "__main__":
    result_file = main()
    print("针对性优化完成！")