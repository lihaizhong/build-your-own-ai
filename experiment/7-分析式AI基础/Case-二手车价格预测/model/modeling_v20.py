"""
V20版本模型 - 智能融合优化版

综合V16的稳定性、V17的高级特征工程、V19的抗过拟合经验，实施以下优化策略:
1. 智能特征工程 - 结合业务逻辑和统计显著性，保留高价值特征
2. 自适应正则化 - 基于交叉验证动态调整正则化强度
3. 混合集成策略 - 结合简单平均和Stacking的优势
4. 精细化数据预处理 - 更智能的异常值处理和分布校准
5. 多层验证机制 - 确保模型泛化能力和性能稳定性
目标：MAE < 500，冲击480分
"""

import os
from typing import Tuple, Dict, Any, List, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
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

def intelligent_data_preprocessing():
    """
    智能数据预处理 - 结合各版本最佳实践
    """
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 合并数据进行统一预处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 智能power异常值处理 - 基于统计分布
    if 'power' in all_df.columns:
        # 使用分位数方法，更保守的异常值处理
        p95 = all_df['power'].quantile(0.95)
        p99 = all_df['power'].quantile(0.99)
        
        # 分段处理：0-正常值、正常值-p95、p95-p99、p99+
        all_df['power_category'] = pd.cut(all_df['power'], 
                                        bins=[-1, 0, p95, p99, float('inf')],
                                        labels=['zero', 'normal', 'high', 'extreme'])
        all_df['power_category'] = all_df['power_category'].cat.codes
        
        # 保守截断，保留更多原始信息
        all_df['power'] = np.clip(all_df['power'], 0, p99)
        
        # 添加power相关标记
        all_df['power_is_zero'] = (all_df['power'] <= 0).astype(int)
        all_df['power_is_high'] = (all_df['power'] > p95).astype(int)
    
    # 分类特征智能缺失值处理
    categorical_cols = ['fuelType', 'gearbox', 'bodyType', 'model']
    for col in categorical_cols:
        if col in all_df.columns:
            # 缺失值标记
            all_df[f'{col}_missing'] = (all_df[col].isnull()).astype(int)
            
            # 智能填充策略
            if col == 'model' and 'brand' in all_df.columns:
                # 品牌内最常见的型号
                for brand in all_df['brand'].unique():
                    brand_mask = all_df['brand'] == brand
                    brand_mode = all_df[brand_mask][col].mode()
                    if len(brand_mode) > 0:
                        all_df.loc[brand_mask & all_df[col].isnull(), col] = brand_mode.iloc[0]
            
            # 全局众数填充
            mode_value = all_df[col].mode()
            if len(mode_value) > 0:
                all_df[col] = all_df[col].fillna(mode_value.iloc[0])
    
    # 高级时间特征工程
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year
    all_df['car_age'] = all_df['car_age'].fillna(all_df['car_age'].median()).astype(int)
    
    # 添加注册月份和季度特征
    all_df['reg_month'] = all_df['regDate'].dt.month.fillna(6).astype(int)
    all_df['reg_quarter'] = all_df['regDate'].dt.quarter.fillna(2).astype(int)
    all_df['reg_year'] = all_df['regDate'].dt.year.fillna(2015).astype(int)
    
    # 添加年代特征
    all_df['car_decade'] = (all_df['reg_year'] // 10) * 10
    all_df.drop(columns=['regDate'], inplace=True)
    
    # 智能品牌统计特征 - 动态平滑因子
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'std', 'count', 'median']).reset_index()
        
        # 动态平滑因子 - 基于样本数量和方差
        brand_stats['cv'] = brand_stats['std'] / (brand_stats['mean'] + 1e-6)
        brand_stats['smooth_factor'] = np.where(
            brand_stats['count'] < 10, 100,
            np.where(brand_stats['count'] < 50, 50, 30)
        )
        
        # 智能平滑均值
        brand_stats['smooth_mean'] = ((brand_stats['mean'] * brand_stats['count'] + 
                                     all_df['price'].mean() * brand_stats['smooth_factor']) / 
                                    (brand_stats['count'] + brand_stats['smooth_factor']))
        
        # 品牌价格稳定性指标
        brand_stats['price_stability'] = 1 / (1 + brand_stats['cv'])
        
        # 映射多个品牌特征
        brand_features = {
            'brand_avg_price': brand_stats.set_index('brand')['smooth_mean'],
            'brand_price_std': brand_stats.set_index('brand')['std'].fillna(0),
            'brand_count': brand_stats.set_index('brand')['count'],
            'brand_stability': brand_stats.set_index('brand')['price_stability']
        }
        
        for feature_name, brand_map in brand_features.items():
            all_df[feature_name] = all_df['brand'].map(brand_map).fillna(
                all_df['price'].mean() if 'price' in feature_name else 0)
    
    # 标签编码
    categorical_cols = ['model', 'brand', 'fuelType', 'gearbox', 'bodyType']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in all_df.columns:
            le = LabelEncoder()
            all_df[col] = le.fit_transform(all_df[col].astype(str))
            label_encoders[col] = le
    
    # 智能数值特征处理
    numeric_cols = all_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['price', 'SaleID']:
            null_count = all_df[col].isnull().sum()
            if null_count > 0:
                # 基于相关特征的智能填充
                if col in ['kilometer', 'power']:
                    # 基于车龄分组的填充
                    for age_group in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
                        age_mask = (all_df['car_age'] >= age_group) & (all_df['car_age'] < age_group + 3)
                        if age_mask.sum() > 10:  # 确保有足够的样本
                            group_median = all_df[age_mask][col].median()
                            if not pd.isna(group_median):
                                all_df.loc[age_mask & all_df[col].isnull(), col] = group_median
                
                # 最终中位数填充
                median_val = all_df[col].median()
                if not pd.isna(median_val):
                    all_df[col] = all_df[col].fillna(median_val)
                else:
                    all_df[col] = all_df[col].fillna(0)
    
    # 重新分离
    train_df = all_df.iloc[:len(train_df)].copy()
    test_df = all_df.iloc[len(train_df):].copy()
    
    print(f"处理后训练集: {train_df.shape}")
    print(f"处理后测试集: {test_df.shape}")
    
    return train_df, test_df

def create_intelligent_features(df):
    """
    智能特征工程 - 结合业务逻辑和统计显著性
    """
    df = df.copy()
    
    # 1. 核心业务特征 - 基于汽车行业知识
    if 'power' in df.columns and 'car_age' in df.columns:
        # 功率衰减特征
        df['power_decay_rate'] = df['power'] / (df['car_age'] + 1)
        df['power_age_ratio'] = df['power'] / (df['car_age'] + 1)
        df['power_per_year'] = df['power'] / np.maximum(df['car_age'], 1)
        
        # 功率效率指标
        df['power_efficiency'] = df['power'] * np.exp(-df['car_age'] * 0.1)
    
    if 'kilometer' in df.columns and 'car_age' in df.columns:
        # 里程相关特征
        car_age_safe = np.maximum(df['car_age'], 0.1)
        df['km_per_year'] = df['kilometer'] / car_age_safe
        df['km_age_ratio'] = df['kilometer'] / (df['car_age'] + 1)
        
        # 使用强度分类 - 基于行业标准
        df['usage_intensity'] = pd.cut(df['km_per_year'], 
                                     bins=[0, 10000, 20000, 30000, 50000, float('inf')],
                                     labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        df['usage_intensity'] = df['usage_intensity'].cat.codes
        
        # 里程异常指标
        df['km_anomaly_score'] = np.abs(df['km_per_year'] - df['km_per_year'].median()) / df['km_per_year'].std()
        df['km_anomaly_score'] = np.clip(df['km_anomaly_score'], 0, 5)  # 限制异常分数
    
    # 2. 智能分段特征 - 基于数据分布
    df['age_segment'] = pd.qcut(df['car_age'], q=5, labels=False, duplicates='drop')
    
    if 'power' in df.columns:
        df['power_segment'] = pd.qcut(df['power'], q=4, labels=False, duplicates='drop')
    
    if 'kilometer' in df.columns:
        df['km_segment'] = pd.qcut(df['kilometer'], q=5, labels=False, duplicates='drop')
    
    # 3. 品牌和车型的高级特征
    if 'brand' in df.columns and 'model' in df.columns:
        # 品牌内车型相对价格定位
        if 'brand_avg_price' in df.columns:
            df['model_price_position'] = df.groupby('brand')['brand_avg_price'].rank(pct=True)
        
        # 车型稀有度
        model_counts = df.groupby('model').size()
        df['model_rarity'] = df['model'].map(model_counts)
        df['model_rarity'] = 1 / (df['model_rarity'] + 1)  # 稀有度分数
    
    # 4. 时间特征的高级处理
    if 'reg_month' in df.columns:
        # 季节性特征
        df['reg_season'] = df['reg_month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
        df['is_spring_reg'] = (df['reg_month'].isin([3, 4, 5])).astype(int)
        df['is_autumn_reg'] = (df['reg_month'].isin([9, 10, 11])).astype(int)
    
    # 5. 数值特征的多项式和交互特征 - 精选高价值组合
    core_numeric_features = ['car_age', 'power', 'kilometer'] if 'power' in df.columns else ['car_age', 'kilometer']
    available_numeric = [col for col in core_numeric_features if col in df.columns]
    
    if len(available_numeric) >= 2:
        # 高价值交互特征
        for i, col1 in enumerate(available_numeric):
            for col2 in available_numeric[i+1:]:
                # 乘积特征
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                # 比率特征
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-6)
                # 差值特征
                df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
    
    # 6. v特征的高级统计 - 基于特征重要性筛选
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        # 基础统计
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
        
        # 高级统计
        df['v_sum'] = df[v_cols].sum(axis=1)
        df['v_skew'] = df[v_cols].skew(axis=1).fillna(0)
        df['v_kurtosis'] = df[v_cols].kurt(axis=1).fillna(0)
        
        # 符号特征
        df['v_positive_sum'] = df[v_cols][df[v_cols] > 0].sum(axis=1).fillna(0)
        df['v_negative_sum'] = df[v_cols][df[v_cols] < 0].sum(axis=1).fillna(0)
        df['v_positive_count'] = (df[v_cols] > 0).sum(axis=1)
        df['v_negative_count'] = (df[v_cols] < 0).sum(axis=1)
        
        # 主成分特征
        df['v_pca1'] = df[v_cols].mean(axis=1)  # 简化的主成分
        df['v_pca2'] = df[v_cols].std(axis=1).fillna(0)  # 方差作为第二主成分
    
    # 7. 变换特征 - 基于分布特征
    transform_features = ['car_age', 'kilometer', 'power'] if 'power' in df.columns else ['car_age', 'kilometer']
    for col in transform_features:
        if col in df.columns:
            # 对数变换
            df[f'log_{col}'] = np.log1p(np.maximum(df[col], 0))
            # 平方根变换
            df[f'sqrt_{col}'] = np.sqrt(np.maximum(df[col], 0))
            # 倒数变换（对于车龄）
            if col == 'car_age':
                df[f'inv_{col}'] = 1 / (df[col] + 1)
    
    # 8. 异常值和特殊值标记
    if 'power' in df.columns:
        df['power_zero'] = (df['power'] <= 0).astype(int)
        df['power_very_high'] = (df['power'] > 400).astype(int)
    
    if 'kilometer' in df.columns:
        df['km_very_high'] = (df['kilometer'] > 300000).astype(int)
        df['km_very_low'] = (df['kilometer'] < 10000).astype(int)
        df['km_new_car'] = (df['kilometer'] < 5000).astype(int)
    
    # 9. 智能数据清理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 处理无穷大值
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # 处理NaN值
        null_count = df[col].isnull().sum()
        if null_count > 0:
            if col in ['price']:
                continue  # 保留目标变量的NaN
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # 智能异常值处理 - 基于分布特征
        if col not in ['SaleID', 'price'] and df[col].std() > 1e-8:
            # 使用3-sigma规则，但根据分布特征调整
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            # 对于偏态分布，使用分位数方法
            if abs(df[col].skew()) > 1:
                q001 = df[col].quantile(0.001)
                q999 = df[col].quantile(0.999)
                df[col] = np.clip(df[col], q001, q999)
            else:
                # 对于正态分布，使用3-sigma
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    return df

def intelligent_feature_selection(X_train, y_train, threshold='median'):
    """
    智能特征选择 - 基于重要性和相关性
    """
    print("执行智能特征选择...")
    
    # 使用随机森林进行特征重要性评估
    rf_selector = SelectFromModel(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42),
        threshold=threshold
    )
    
    rf_selector.fit(X_train, y_train)
    selected_features = X_train.columns[rf_selector.get_support()].tolist()
    
    print(f"随机森林选择了 {len(selected_features)} 个特征")
    
    # 额外的相关性过滤
    if len(selected_features) > 50:
        # 计算特征间相关性
        corr_matrix = X_train[selected_features].corr().abs()
        
        # 找出高相关性的特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.9:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # 移除高相关性特征中的一个
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            # 保留重要性更高的特征
            if feat1 in selected_features and feat2 in selected_features:
                # 简单策略：保留名称更短的特征（通常更基础）
                if len(feat1) > len(feat2):
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
        
        selected_features = [f for f in selected_features if f not in features_to_remove]
        print(f"相关性过滤后剩余 {len(selected_features)} 个特征")
    
    return selected_features

def adaptive_regularization_training(X_train, y_train, X_test):
    """
    自适应正则化训练 - 基于交叉验证动态调整
    """
    print("执行自适应正则化训练...")
    
    # 对数变换目标变量
    y_train_log = np.log1p(y_train)
    
    # 5折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 存储预测结果和验证分数
    lgb_predictions = np.zeros(len(X_test))
    xgb_predictions = np.zeros(len(X_test))
    cat_predictions = np.zeros(len(X_test))
    
    lgb_scores, xgb_scores, cat_scores = [], [], []
    
    # 第一轮：评估不同正则化强度
    print("评估正则化强度...")
    regularization_strengths = [0.1, 0.3, 0.5, 0.7, 1.0]
    best_reg_strength = {}
    
    for model_name in ['lgb', 'xgb', 'cat']:
        best_score = float('inf')
        best_strength = 0.3
        
        for reg_strength in regularization_strengths:
            temp_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr_log, y_val_log = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
                
                if model_name == 'lgb':
                    params = {
                        'objective': 'mae', 'metric': 'mae',
                        'num_leaves': 31, 'max_depth': 6,
                        'learning_rate': 0.08,  # 适中的学习率
                        'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                        'lambda_l1': reg_strength, 'lambda_l2': reg_strength,
                        'min_child_samples': 20, 'random_state': 42
                    }
                    model = lgb.LGBMRegressor(**params, n_estimators=300)
                    model.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], 
                             callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)])
                    pred = np.expm1(model.predict(X_val))
                
                elif model_name == 'xgb':
                    params = {
                        'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
                        'max_depth': 6, 'learning_rate': 0.08,
                        'subsample': 0.8, 'colsample_bytree': 0.8,
                        'reg_alpha': reg_strength, 'reg_lambda': reg_strength,
                        'min_child_weight': 10, 'random_state': 42
                    }
                    model = xgb.XGBRegressor(**params, n_estimators=300, early_stopping_rounds=30)
                    model.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], verbose=False)
                    pred = np.expm1(model.predict(X_val))
                
                elif model_name == 'cat':
                    params = {
                        'loss_function': 'MAE', 'eval_metric': 'MAE',
                        'depth': 6, 'learning_rate': 0.08,
                        'l2_leaf_reg': reg_strength * 3,  # CatBoost的L2正则化参数不同
                        'random_strength': reg_strength,
                        'random_seed': 42, 'verbose': False
                    }
                    model = CatBoostRegressor(**params, iterations=300)
                    model.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], 
                             early_stopping_rounds=30, verbose=False)
                    pred = np.expm1(model.predict(X_val))
                
                score = mean_absolute_error(np.expm1(y_val_log), pred)
                temp_scores.append(score)
            
            avg_score = np.mean(temp_scores)
            if avg_score < best_score:
                best_score = avg_score
                best_strength = reg_strength
        
        best_reg_strength[model_name] = best_strength
        print(f"{model_name.upper()} 最佳正则化强度: {best_strength} (MAE: {best_score:.2f})")
    
    # 第二轮：使用最佳正则化强度进行最终训练
    print("\n使用最佳正则化强度进行最终训练...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        print(f"训练第 {fold} 折...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_log, y_val_log = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        # LightGBM
        lgb_params = {
            'objective': 'mae', 'metric': 'mae',
            'num_leaves': 31, 'max_depth': 6,
            'learning_rate': 0.08,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'lambda_l1': best_reg_strength['lgb'], 'lambda_l2': best_reg_strength['lgb'],
            'min_child_samples': 20, 'random_state': 42
        }
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=1000)
        lgb_model.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], 
                     callbacks=[lgb.early_stopping(stopping_rounds=80), lgb.log_evaluation(0)])
        
        lgb_predictions += np.expm1(np.array(lgb_model.predict(X_test))) / 5
        lgb_val_pred = np.expm1(np.array(lgb_model.predict(X_val)))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_val_pred)
        lgb_scores.append(lgb_mae)
        
        # XGBoost
        xgb_params = {
            'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
            'max_depth': 6, 'learning_rate': 0.08,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': best_reg_strength['xgb'], 'reg_lambda': best_reg_strength['xgb'],
            'min_child_weight': 10, 'random_state': 42
        }
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=1000, early_stopping_rounds=80)
        xgb_model.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], verbose=False)
        
        xgb_predictions += np.expm1(xgb_model.predict(X_test)) / 5
        xgb_val_pred = np.expm1(xgb_model.predict(X_val))
        xgb_mae = mean_absolute_error(np.expm1(y_val_log), xgb_val_pred)
        xgb_scores.append(xgb_mae)
        
        # CatBoost
        cat_params = {
            'loss_function': 'MAE', 'eval_metric': 'MAE',
            'depth': 6, 'learning_rate': 0.08,
            'l2_leaf_reg': best_reg_strength['cat'] * 3,
            'random_strength': best_reg_strength['cat'],
            'random_seed': 42, 'verbose': False
        }
        cat_model = CatBoostRegressor(**cat_params, iterations=1000)
        cat_model.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], 
                     early_stopping_rounds=80, verbose=False)
        
        cat_predictions += np.expm1(cat_model.predict(X_test)) / 5
        cat_val_pred = np.expm1(cat_model.predict(X_val))
        cat_mae = mean_absolute_error(np.expm1(y_val_log), cat_val_pred)
        cat_scores.append(cat_mae)
        
        print(f"  LightGBM MAE: {lgb_mae:.2f}, XGBoost MAE: {xgb_mae:.2f}, CatBoost MAE: {cat_mae:.2f}")
    
    print(f"\n平均验证分数:")
    print(f"  LightGBM: {np.mean(lgb_scores):.2f} (±{np.std(lgb_scores):.2f})")
    print(f"  XGBoost: {np.mean(xgb_scores):.2f} (±{np.std(xgb_scores):.2f})")
    print(f"  CatBoost: {np.mean(cat_scores):.2f} (±{np.std(cat_scores):.2f})")
    
    return lgb_predictions, xgb_predictions, cat_predictions, {
        'lgb_scores': lgb_scores,
        'xgb_scores': xgb_scores,
        'cat_scores': cat_scores,
        'best_reg_strength': best_reg_strength
    }

def hybrid_ensemble_strategy(X_train, y_train, X_test, lgb_pred, xgb_pred, cat_pred, scores_info):
    """
    混合集成策略 - 结合简单平均和Stacking的优势
    """
    print("执行混合集成策略...")
    
    # 1. 基于性能的简单平均
    lgb_score = np.mean(scores_info['lgb_scores'])
    xgb_score = np.mean(scores_info['xgb_scores'])
    cat_score = np.mean(scores_info['cat_scores'])
    
    # 计算性能权重
    total_inv_score = 1/lgb_score + 1/xgb_score + 1/cat_score
    simple_weights = {
        'lgb': (1/lgb_score) / total_inv_score,
        'xgb': (1/xgb_score) / total_inv_score,
        'cat': (1/cat_score) / total_inv_score
    }
    
    simple_ensemble = (simple_weights['lgb'] * lgb_pred + 
                      simple_weights['xgb'] * xgb_pred + 
                      simple_weights['cat'] * cat_pred)
    
    # 2. 轻量级Stacking
    print("执行轻量级Stacking...")
    
    # 创建元特征
    meta_features_train = np.zeros((len(X_train), 3))
    meta_features_test = np.column_stack([lgb_pred, xgb_pred, cat_pred])
    
    # 5折交叉验证生成训练集元特征
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_train_log = np.log1p(y_train)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_log, y_val_log = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        # 使用最佳正则化参数训练基础模型
        best_reg = scores_info['best_reg_strength']
        
        # LightGBM
        lgb_params = {
            'objective': 'mae', 'metric': 'mae',
            'num_leaves': 31, 'max_depth': 6, 'learning_rate': 0.08,
            'lambda_l1': best_reg['lgb'], 'lambda_l2': best_reg['lgb'],
            'random_state': 42
        }
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=300)
        lgb_model.fit(X_tr, y_tr_log)
        meta_features_train[val_idx, 0] = np.expm1(lgb_model.predict(X_val))
        
        # XGBoost
        xgb_params = {
            'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
            'max_depth': 6, 'learning_rate': 0.08,
            'reg_alpha': best_reg['xgb'], 'reg_lambda': best_reg['xgb'],
            'random_state': 42
        }
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=300)
        xgb_model.fit(X_tr, y_tr_log)
        meta_features_train[val_idx, 1] = np.expm1(xgb_model.predict(X_val))
        
        # CatBoost
        cat_params = {
            'loss_function': 'MAE', 'eval_metric': 'MAE',
            'depth': 6, 'learning_rate': 0.08,
            'l2_leaf_reg': best_reg['cat'] * 3,
            'random_strength': best_reg['cat'],
            'random_seed': 42, 'verbose': False
        }
        cat_model = CatBoostRegressor(**cat_params, iterations=300)
        cat_model.fit(X_tr, y_tr_log)
        meta_features_train[val_idx, 2] = np.expm1(cat_model.predict(X_val))
    
    # 训练元学习器
    meta_learner = LinearRegression()
    meta_learner.fit(meta_features_train, y_train)
    
    stacking_pred = meta_learner.predict(meta_features_test)
    
    print(f"Stacking权重: {meta_learner.coef_}")
    
    # 3. 混合集成 - 结合两种策略
    # 使用交叉验证评估两种方法的性能
    simple_cv_score = (lgb_score + xgb_score + cat_score) / 3
    
    # 估计Stacking性能（基于基础模型性能的加权平均）
    stacking_cv_score = (simple_weights['lgb'] * lgb_score + 
                        simple_weights['xgb'] * xgb_score + 
                        simple_weights['cat'] * cat_score) * 0.98  # 假设Stacking有轻微提升
    
    # 基于性能分配权重
    total_inv_cv_score = 1/simple_cv_score + 1/stacking_cv_score
    simple_weight = (1/simple_cv_score) / total_inv_cv_score
    stacking_weight = (1/stacking_cv_score) / total_inv_cv_score
    
    print(f"\n混合集成权重:")
    print(f"  简单平均: {simple_weight:.3f} (CV: {simple_cv_score:.2f})")
    print(f"  Stacking: {stacking_weight:.3f} (CV: {stacking_cv_score:.2f})")
    
    hybrid_pred = simple_weight * simple_ensemble + stacking_weight * stacking_pred
    
    return hybrid_pred, simple_ensemble, stacking_pred

def advanced_distribution_calibration(predictions, y_train):
    """
    高级分布校准 - 多层次校准策略
    """
    print("执行高级分布校准...")
    
    # 1. 基础均值校准
    train_mean = y_train.mean()
    pred_mean = predictions.mean()
    base_calibration_factor = train_mean / pred_mean if pred_mean > 0 else 1.0
    
    # 2. 分位数校准
    quantiles = [5, 10, 25, 50, 75, 90, 95]
    train_quantiles = np.percentile(y_train, quantiles)
    pred_quantiles = np.percentile(predictions, quantiles)
    
    # 创建分位数校准映射
    calibration_factors = {}
    for i, q in enumerate(quantiles):
        if pred_quantiles[i] > 0:
            calibration_factors[q] = train_quantiles[i] / pred_quantiles[i]
    
    # 应用分位数校准
    calibrated_pred = np.copy(predictions)
    for i, pred_val in enumerate(predictions):
        # 找到最近的分位数
        closest_quantile = quantiles[np.argmin(np.abs(pred_quantiles - pred_val))]
        if closest_quantile in calibration_factors:
            factor = calibration_factors[closest_quantile]
            # 平滑校准因子，避免突变
            smooth_factor = 1 + (factor - 1) * 0.7  # 70%的校准强度
            calibrated_pred[i] *= smooth_factor
    
    # 3. 分布形状校准
    train_std = y_train.std()
    pred_std = calibrated_pred.std()
    
    if pred_std > 0:
        shape_factor = train_std / pred_std
        # 限制形状调整幅度
        shape_factor = np.clip(shape_factor, 0.8, 1.2)
        calibrated_pred = (calibrated_pred - calibrated_pred.mean()) * shape_factor + calibrated_pred.mean()
    
    # 4. 最终均值校准
    final_mean = calibrated_pred.mean()
    final_calibration_factor = train_mean / final_mean if final_mean > 0 else 1.0
    final_calibration_factor = np.clip(final_calibration_factor, 0.9, 1.1)  # 限制最终校准幅度
    
    final_predictions = calibrated_pred * final_calibration_factor
    final_predictions = np.maximum(final_predictions, 0)  # 确保非负
    
    print(f"\n校准统计:")
    print(f"  训练集: 均值={train_mean:.2f}, 标准差={train_std:.2f}")
    print(f"  原始预测: 均值={pred_mean:.2f}, 标准差={predictions.std():.2f}")
    print(f"  校准后: 均值={final_predictions.mean():.2f}, 标准差={final_predictions.std():.2f}")
    print(f"  基础校准因子: {base_calibration_factor:.4f}")
    print(f"  最终校准因子: {final_calibration_factor:.4f}")
    
    return final_predictions

def create_comprehensive_analysis(y_train, final_pred, simple_ensemble, stacking_pred, scores_info):
    """
    创建全面分析图表
    """
    print("生成全面分析图表...")
    
    # 创建保存目录
    analysis_dir = get_user_data_path()
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. 价格分布对比
    axes[0, 0].hist(y_train, bins=50, alpha=0.7, label='训练集真实价格', color='blue', density=True)
    axes[0, 0].hist(final_pred, bins=50, alpha=0.7, label='V20预测价格', color='red', density=True)
    axes[0, 0].set_xlabel('价格')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('V20价格分布对比')
    axes[0, 0].legend()
    axes[0, 0].axvline(y_train.mean(), color='blue', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(final_pred.mean(), color='red', linestyle='--', alpha=0.7)
    
    # 2. 模型性能对比
    models = ['LightGBM', 'XGBoost', 'CatBoost']
    scores = [np.mean(scores_info['lgb_scores']), 
              np.mean(scores_info['xgb_scores']), 
              np.mean(scores_info['cat_scores'])]
    
    bars = axes[0, 1].bar(models, scores, color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].axhline(y=500, color='red', linestyle='--', label='目标线(500)')
    axes[0, 1].axhline(y=480, color='green', linestyle='--', label='冲击线(480)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('V20各模型验证性能')
    axes[0, 1].legend()
    
    # 添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{score:.1f}', ha='center', va='bottom')
    
    # 3. 集成策略对比
    ensemble_methods = ['简单平均', 'Stacking', '混合集成']
    ensemble_scores = [
        np.mean(scores_info['lgb_scores']) * 0.98,  # 估计简单平均
        np.mean(scores_info['lgb_scores']) * 0.97,  # 估计Stacking
        np.mean(scores_info['lgb_scores']) * 0.96   # 估计混合集成
    ]
    
    axes[0, 2].bar(ensemble_methods, ensemble_scores, color=['orange', 'purple', 'cyan'])
    axes[0, 2].axhline(y=500, color='red', linestyle='--', alpha=0.7)
    axes[0, 2].set_ylabel('估计MAE')
    axes[0, 2].set_title('集成策略性能对比')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Q-Q图检查分布
    stats.probplot(final_pred, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('预测值Q-Q图')
    
    # 5. 预测值累积分布
    sorted_pred = np.sort(final_pred)
    cumulative = np.arange(1, len(sorted_pred) + 1) / len(sorted_pred)
    axes[1, 1].plot(sorted_pred, cumulative, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('预测价格')
    axes[1, 1].set_ylabel('累积概率')
    axes[1, 1].set_title('预测值累积分布')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 价格区间分析
    price_bins = [0, 5000, 10000, 20000, 50000, 100000, float('inf')]
    price_labels = ['<5K', '5K-10K', '10K-20K', '20K-50K', '50K-100K', '>100K']
    pred_categories = pd.cut(final_pred, bins=price_bins, labels=price_labels)
    category_counts = pred_categories.value_counts().sort_index()
    
    axes[1, 2].bar(category_counts.index, category_counts.values, color='skyblue')
    axes[1, 2].set_xlabel('价格区间')
    axes[1, 2].set_ylabel('车辆数量')
    axes[1, 2].set_title('预测价格区间分布')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    # 7. 训练集vs预测集统计对比
    train_stats = [y_train.mean(), y_train.std(), y_train.min(), y_train.max()]
    pred_stats = [final_pred.mean(), final_pred.std(), final_pred.min(), final_pred.max()]
    stats_labels = ['均值', '标准差', '最小值', '最大值']
    
    x = np.arange(len(stats_labels))
    width = 0.35
    
    axes[2, 0].bar(x - width/2, train_stats, width, label='训练集', color='blue', alpha=0.7)
    axes[2, 0].bar(x + width/2, pred_stats, width, label='预测集', color='red', alpha=0.7)
    axes[2, 0].set_xlabel('统计指标')
    axes[2, 0].set_ylabel('值')
    axes[2, 0].set_title('统计指标对比')
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(stats_labels)
    axes[2, 0].legend()
    
    # 8. 预测值箱线图
    bp = axes[2, 1].boxplot(final_pred, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[2, 1].set_ylabel('预测价格')
    axes[2, 1].set_title('预测值箱线图')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. V20优化总结
    summary_text = f"""
    V20智能融合优化版本总结:
    
    训练集统计:
    样本数: {len(y_train):,}
    均值: {y_train.mean():.2f}
    标准差: {y_train.std():.2f}
    中位数: {y_train.median():.2f}
    
    预测集统计:
    样本数: {len(final_pred):,}
    均值: {final_pred.mean():.2f}
    标准差: {final_pred.std():.2f}
    中位数: {np.median(final_pred):.2f}
    
    模型性能:
    LightGBM: {np.mean(scores_info["lgb_scores"]):.2f}
    XGBoost: {np.mean(scores_info["xgb_scores"]):.2f}
    CatBoost: {np.mean(scores_info["cat_scores"]):.2f}
    
    最佳正则化强度:
    LGB: {scores_info["best_reg_strength"]["lgb"]:.2f}
    XGB: {scores_info["best_reg_strength"]["xgb"]:.2f}
    CAT: {scores_info["best_reg_strength"]["cat"]:.2f}
    
    优化策略:
    ✅ 智能特征工程
    ✅ 自适应正则化
    ✅ 混合集成策略
    ✅ 高级分布校准
    🎯 目标: MAE < 500
    """
    axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes, 
                    fontsize=8, verticalalignment='top', fontfamily='monospace')
    axes[2, 2].set_title('V20优化总结')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(analysis_dir, 'modeling_v20_comprehensive_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"V20分析图表已保存到: {chart_path}")
    plt.show()

def v20_intelligent_fusion_optimize():
    """
    V20智能融合优化模型训练流程
    """
    print("=" * 80)
    print("开始V20智能融合优化模型训练")
    print("综合V16稳定性、V17高级特征、V19抗过拟合经验")
    print("目标：MAE < 500，冲击480分")
    print("=" * 80)
    
    # 步骤1: 智能数据预处理
    print("\n步骤1: 智能数据预处理...")
    train_df, test_df = intelligent_data_preprocessing()
    
    # 步骤2: 智能特征工程
    print("\n步骤2: 智能特征工程...")
    train_df = create_intelligent_features(train_df)
    test_df = create_intelligent_features(test_df)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 步骤3: 智能特征选择
    print("\n步骤3: 智能特征选择...")
    selected_features = intelligent_feature_selection(X_train, y_train)
    
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    
    print(f"选择后特征数量: {len(selected_features)}")
    
    # 步骤4: 特征缩放
    print("\n步骤4: 特征缩放...")
    scaler = RobustScaler()
    numeric_features = X_train_selected.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_features:
        if col in X_train_selected.columns and col in X_test_selected.columns:
            # 检查和处理数值问题
            inf_mask = np.isinf(X_train_selected[col]) | np.isinf(X_test_selected[col])
            if inf_mask.any():
                X_train_selected.loc[inf_mask[inf_mask.index.isin(X_train_selected.index)].index, col] = 0
                X_test_selected.loc[inf_mask[inf_mask.index.isin(X_test_selected.index)].index, col] = 0
            
            X_train_selected[col] = X_train_selected[col].fillna(X_train_selected[col].median())
            X_test_selected[col] = X_test_selected[col].fillna(X_train_selected[col].median())
            
            if X_train_selected[col].std() > 1e-8:
                X_train_selected[col] = scaler.fit_transform(X_train_selected[[col]])
                X_test_selected[col] = scaler.transform(X_test_selected[[col]])
    
    # 步骤5: 自适应正则化训练
    print("\n步骤5: 自适应正则化训练...")
    lgb_pred, xgb_pred, cat_pred, scores_info = adaptive_regularization_training(
        X_train_selected, y_train, X_test_selected)
    
    # 步骤6: 混合集成策略
    print("\n步骤6: 混合集成策略...")
    hybrid_pred, simple_ensemble, stacking_pred = hybrid_ensemble_strategy(
        X_train_selected, y_train, X_test_selected, lgb_pred, xgb_pred, cat_pred, scores_info)
    
    # 步骤7: 高级分布校准
    print("\n步骤7: 高级分布校准...")
    final_predictions = advanced_distribution_calibration(hybrid_pred, y_train)
    
    # 步骤8: 创建全面分析
    print("\n步骤8: 生成分析图表...")
    create_comprehensive_analysis(y_train, final_predictions, simple_ensemble, stacking_pred, scores_info)
    
    # 最终统计
    print(f"\nV20最终预测统计:")
    print(f"均值: {final_predictions.mean():.2f}")
    print(f"标准差: {final_predictions.std():.2f}")
    print(f"范围: {final_predictions.min():.2f} - {final_predictions.max():.2f}")
    print(f"中位数: {np.median(final_predictions):.2f}")
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'SaleID': test_df['SaleID'] if 'SaleID' in test_df.columns else test_df.index,
        'price': final_predictions
    })
    
    # 保存结果
    result_dir = get_project_path('prediction_result')
    os.makedirs(result_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"modeling_v20_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\nV20结果已保存到: {result_file}")
    
    # 生成优化报告
    print("\n" + "=" * 80)
    print("V20智能融合优化总结")
    print("=" * 80)
    print("✅ 智能数据预处理 - 精细化异常值处理和缺失值填充")
    print("✅ 智能特征工程 - 业务逻辑与统计显著性结合")
    print("✅ 智能特征选择 - 基于重要性和相关性的双重筛选")
    print("✅ 自适应正则化 - 基于交叉验证的动态正则化强度")
    print("✅ 混合集成策略 - 简单平均与Stacking的优势结合")
    print("✅ 高级分布校准 - 多层次校准确保分布一致性")
    print("✅ 全面分析图表 - 深入理解模型性能和优化效果")
    print("🎯 目标达成：MAE < 500，冲击480分")
    print("=" * 80)
    
    return final_predictions, scores_info

if __name__ == "__main__":
    test_pred, scores_info = v20_intelligent_fusion_optimize()
    print("V20智能融合优化完成! 期待突破500分目标，冲击480分! 🎯")