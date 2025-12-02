"""
V24简化版本模型 - 核心优化测试版

基于V23的497.6048分基础，测试核心优化策略:
1. 增强特征工程 - 目标编码、更多交互特征
2. 优化模型参数 - 基于V23的微调
3. 改进集成策略 - 更智能的权重调整
4. 增强校准 - 分位数校准的改进版本
目标：测试核心优化效果，快速验证改进空间
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
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

def simplified_preprocessing():
    """
    简化的数据预处理 - 基于V23但增加目标编码
    """
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 合并数据进行预处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 增强的power处理
    if 'power' in all_df.columns:
        all_df['power'] = np.clip(all_df['power'], 0, 600)
        all_df['power_is_zero'] = (all_df['power'] <= 0).astype(int)
        all_df['power_is_high'] = (all_df['power'] > 400).astype(int)
        
        # V24新增：power的多种变换
        all_df['log_power'] = np.log1p(np.maximum(all_df['power'], 1))
        all_df['sqrt_power'] = np.sqrt(np.maximum(all_df['power'], 0))
    
    # 高级分类特征处理 - 增加目标编码
    categorical_cols = ['fuelType', 'gearbox', 'bodyType', 'model', 'brand']
    for col in categorical_cols:
        if col in all_df.columns:
            all_df[f'{col}_missing'] = (all_df[col].isnull()).astype(int)
            
            # 智能填充
            if col == 'model' and 'brand' in all_df.columns:
                for brand in all_df['brand'].unique():
                    brand_mask = all_df['brand'] == brand
                    brand_mode = all_df[brand_mask][col].mode()
                    if len(brand_mode) > 0:
                        all_df.loc[brand_mask & all_df[col].isnull(), col] = brand_mode.iloc[0]
            
            mode_value = all_df[col].mode()
            if len(mode_value) > 0:
                all_df[col] = all_df[col].fillna(mode_value.iloc[0])
            
            # 频率编码
            freq_map = all_df[col].value_counts().to_dict()
            all_df[f'{col}_freq'] = all_df[col].map(freq_map)
            
            # V24新增：目标编码（仅对训练集有效）
            if 'price' in all_df.columns and col != 'brand':  # brand太多类别
                target_mean = all_df.groupby(col)['price'].mean()
                global_mean = all_df['price'].mean()
                count = all_df[col].value_counts()
                smooth_factor = 100
                
                smooth_encoding = (target_mean * count + global_mean * smooth_factor) / (count + smooth_factor)
                all_df[f'{col}_target_enc'] = all_df[col].map(smooth_encoding).fillna(global_mean)
    
    # 高级时间特征工程
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year
    all_df['car_age'] = all_df['car_age'].fillna(0).astype(int)
    all_df['reg_month'] = all_df['regDate'].dt.month.fillna(6).astype(int)
    all_df['reg_quarter'] = all_df['regDate'].dt.quarter.fillna(2).astype(int)
    all_df['reg_dayofweek'] = all_df['regDate'].dt.dayofweek.fillna(3).astype(int)
    
    # V23的季节特征
    all_df['reg_season'] = all_df['reg_month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
    all_df['is_winter_reg'] = all_df['reg_month'].isin([12, 1, 2]).astype(int)
    all_df['is_summer_reg'] = all_df['reg_month'].isin([6, 7, 8]).astype(int)
    
    # V24新增：周期性时间特征
    all_df['reg_month_sin'] = np.sin(2 * np.pi * all_df['reg_month'] / 12)
    all_df['reg_month_cos'] = np.cos(2 * np.pi * all_df['reg_month'] / 12)
    
    all_df.drop(columns=['regDate'], inplace=True)
    
    # 高级品牌统计特征
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count', 'std', 'median']).reset_index()
        brand_stats['smooth_mean'] = ((brand_stats['mean'] * brand_stats['count'] + 
                                     all_df['price'].mean() * 40) / (brand_stats['count'] + 40))
        
        # V24新增：更多品牌统计特征
        brand_stats['cv'] = brand_stats['std'] / brand_stats['mean']
        brand_stats['cv'] = brand_stats['cv'].fillna(brand_stats['cv'].median())
        brand_stats['skewness'] = (brand_stats['mean'] - brand_stats['median']) / brand_stats['std']
        brand_stats['skewness'] = brand_stats['skewness'].fillna(0)
        
        # 映射特征
        all_df['brand_avg_price'] = all_df['brand'].map(brand_stats.set_index('brand')['smooth_mean']).fillna(all_df['price'].mean())
        all_df['brand_price_stability'] = all_df['brand'].map(brand_stats.set_index('brand')['cv']).fillna(brand_stats['cv'].median())
        all_df['brand_skewness'] = all_df['brand'].map(brand_stats.set_index('brand')['skewness']).fillna(0)
    
    # 标签编码
    categorical_cols = ['model', 'brand', 'fuelType', 'gearbox', 'bodyType']
    for col in categorical_cols:
        if col in all_df.columns:
            le = LabelEncoder()
            all_df[col] = le.fit_transform(all_df[col].astype(str))
    
    # 数值特征处理
    numeric_cols = all_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['price', 'SaleID']:
            null_count = all_df[col].isnull().sum()
            if null_count > 0:
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

def create_simplified_features(df):
    """
    简化的增强特征工程 - 基于V23但更精准
    """
    df = df.copy()
    
    # V23的核心业务特征
    if 'power' in df.columns and 'car_age' in df.columns:
        df['power_age_ratio'] = df['power'] / (df['car_age'] + 1)
        df['power_decay'] = df['power'] * np.exp(-df['car_age'] * 0.05)
    
    if 'kilometer' in df.columns and 'car_age' in df.columns:
        car_age_safe = np.maximum(df['car_age'], 0.1)
        df['km_per_year'] = df['kilometer'] / car_age_safe
        df['km_per_year'] = np.clip(df['km_per_year'], 0, 40000)
    
    # V24新增：更多业务逻辑特征
    if 'power' in df.columns and 'kilometer' in df.columns:
        df['power_km_ratio'] = df['power'] / (df['kilometer'] + 1)
    
    if 'car_age' in df.columns and 'kilometer' in df.columns:
        df['age_km_interaction'] = df['car_age'] * df['kilometer'] / 1000
    
    # V24新增：多项式特征
    if 'power' in df.columns:
        df['power_squared'] = df['power'] ** 2 / 1000  # 归一化
    
    if 'car_age' in df.columns:
        df['car_age_squared'] = df['car_age'] ** 2
    
    # V23的分段特征
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 3, 6, 10, float('inf')], 
                              labels=['very_new', 'new', 'medium', 'old'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    # V24新增：更精细的年龄分段
    df['age_segment_fine'] = pd.cut(df['car_age'], bins=[-1, 1, 3, 5, 7, 10, 15, float('inf')], 
                                   labels=['brand_new', 'very_new', 'new', 'medium', 'old', 'very_old', 'ancient'])
    df['age_segment_fine'] = df['age_segment_fine'].cat.codes
    
    # V24新增：里程分段
    if 'kilometer' in df.columns:
        df['km_segment'] = pd.cut(df['kilometer'], bins=[-1, 30000, 80000, 120000, 160000, 200000, float('inf')], 
                                 labels=['very_low', 'low', 'medium_low', 'medium', 'high', 'very_high'])
        df['km_segment'] = df['km_segment'].cat.codes
    
    # V23的变换特征
    if 'car_age' in df.columns:
        df['log_car_age'] = np.log1p(df['car_age'])
        df['sqrt_car_age'] = np.sqrt(df['car_age'])
    
    if 'kilometer' in df.columns:
        df['log_kilometer'] = np.log1p(df['kilometer'])
        df['sqrt_kilometer'] = np.sqrt(df['kilometer'])
    
    # V23的v特征统计
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
        df['v_skew'] = df[v_cols].skew(axis=1).fillna(0)
        df['v_kurt'] = df[v_cols].kurtosis(axis=1).fillna(0)
        
        # V24新增：v特征的更多统计
        df['v_sum'] = df[v_cols].sum(axis=1)
        df['v_median'] = df[v_cols].median(axis=1)
        df['v_mean_to_std_ratio'] = df['v_mean'] / (df['v_std'] + 1e-6)
    
    # V24新增：高阶交互特征
    interaction_features = [
        ('power', 'car_age'),
        ('power', 'kilometer'),
        ('car_age', 'kilometer'),
        ('power_age_ratio', 'km_per_year'),
        ('brand_avg_price', 'car_age'),
        ('brand_avg_price', 'power'),
    ]
    
    for feat1, feat2 in interaction_features:
        if feat1 in df.columns and feat2 in df.columns:
            df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1)
    
    # 数据清理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 处理无穷大值
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # 填充NaN值
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
        
        # 异常值处理
        if col not in ['SaleID', 'price'] and df[col].std() > 1e-8:
            q999 = df[col].quantile(0.999)
            q001 = df[col].quantile(0.001)
            if q999 > q001 and q999 > 0:
                df[col] = np.clip(df[col], q001, q999)
    
    return df

def train_optimized_models(X_train, y_train, X_test):
    """
    训练优化模型 - 基于V23参数的进一步优化
    """
    print("训练优化模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 分层交叉验证
    y_bins = pd.qcut(y_train, q=10, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # V24优化参数 - 基于V23的微调
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'num_leaves': 37,        # V24微调：从33增加到37
        'max_depth': 8,          # V24微调：从7增加到8
        'learning_rate': 0.07,   # V24微调：从0.075降低到0.07
        'feature_fraction': 0.9, # V24微调：从0.85增加到0.9
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'lambda_l1': 0.2,        # V24微调：从0.25降低到0.2
        'lambda_l2': 0.2,
        'min_child_samples': 15, # V24微调：从18降低到15
        'random_state': 42,
    }
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': 8,          # V24微调：从7增加到8
        'learning_rate': 0.07,   # V24微调：从0.075降低到0.07
        'subsample': 0.9,        # V24微调：从0.85增加到0.9
        'colsample_bytree': 0.9,
        'reg_alpha': 0.6,        # V24微调：从0.7降低到0.6
        'reg_lambda': 0.6,
        'min_child_weight': 7,   # V24微调：从9降低到7
        'random_state': 42
    }
    
    catboost_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'depth': 8,              # V24微调：从7增加到8
        'learning_rate': 0.07,   # V24微调：从0.075降低到0.07
        'iterations': 1100,      # V24微调：从900增加到1100
        'l2_leaf_reg': 1.1,      # V24微调：从1.3降低到1.1
        'random_strength': 0.3,  # V24微调：从0.4降低到0.3
        'random_seed': 42,
        'verbose': False
    }
    
    # 存储预测结果
    lgb_predictions = np.zeros(len(X_test))
    xgb_predictions = np.zeros(len(X_test))
    cat_predictions = np.zeros(len(X_test))
    
    # 存储验证分数
    lgb_scores, xgb_scores, cat_scores = [], [], []
    
    # 交叉验证训练
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_bins), 1):
        print(f"训练第 {fold} 折...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_log, y_val_log = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=1800)
        lgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)])
        
        lgb_predictions += np.expm1(np.array(lgb_model.predict(X_test))) / 5
        lgb_val_pred = np.expm1(np.array(lgb_model.predict(X_val)))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_val_pred)
        lgb_scores.append(lgb_mae)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=1800, early_stopping_rounds=100)
        xgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     verbose=False)
        
        xgb_predictions += np.expm1(xgb_model.predict(X_test)) / 5
        xgb_val_pred = np.expm1(xgb_model.predict(X_val))
        xgb_mae = mean_absolute_error(np.expm1(y_val_log), xgb_val_pred)
        xgb_scores.append(xgb_mae)
        
        # CatBoost
        cat_model = CatBoostRegressor(**catboost_params)
        cat_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     early_stopping_rounds=100, 
                     verbose=False)
        
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
        'cat_scores': cat_scores
    }

def optimized_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info):
    """
    优化集成策略 - 基于V23但更智能
    """
    print("执行优化集成策略...")
    
    # 基于性能的自适应权重
    lgb_score = np.mean(scores_info['lgb_scores'])
    xgb_score = np.mean(scores_info['xgb_scores'])
    cat_score = np.mean(scores_info['cat_scores'])
    
    # 计算权重，但使用更智能的调整
    total_inv_score = 1/lgb_score + 1/xgb_score + 1/cat_score
    raw_weights = {
        'lgb': (1/lgb_score) / total_inv_score,
        'xgb': (1/xgb_score) / total_inv_score,
        'cat': (1/cat_score) / total_inv_score
    }
    
    # 基于分数稳定性的权重调整
    lgb_std = np.std(scores_info['lgb_scores'])
    xgb_std = np.std(scores_info['xgb_scores'])
    cat_std = np.std(scores_info['cat_scores'])
    
    # 稳定性惩罚因子
    stability_factor = {
        'lgb': 1 / (1 + lgb_std),
        'xgb': 1 / (1 + xgb_std),
        'cat': 1 / (1 + cat_std)
    }
    
    # 应用稳定性调整
    for model in raw_weights:
        raw_weights[model] *= stability_factor[model]
    
    # 重新归一化并限制权重
    total_weight = sum(raw_weights.values())
    balanced_weights = {}
    for model, weight in raw_weights.items():
        balanced_weights[model] = (weight / total_weight) * 0.85 + 0.05  # 确保最小权重0.05
        balanced_weights[model] = np.clip(balanced_weights[model], 0.1, 0.8)  # V24微调：从0.15-0.7调整为0.1-0.8
    
    # 最终归一化
    total_weight = sum(balanced_weights.values())
    final_weights = {model: weight/total_weight for model, weight in balanced_weights.items()}
    
    print(f"优化集成权重:")
    for model, weight in final_weights.items():
        print(f"  {model.upper()}: {weight:.3f}")
    
    ensemble_pred = (final_weights['lgb'] * lgb_pred + 
                    final_weights['xgb'] * xgb_pred + 
                    final_weights['cat'] * cat_pred)
    
    return ensemble_pred

def optimized_calibration(predictions, y_train):
    """
    优化校准 - 改进版分位数校准
    """
    train_mean = y_train.mean()
    train_median = y_train.median()
    pred_mean = predictions.mean()
    pred_median = np.median(predictions)
    
    print(f"\n优化校准:")
    print(f"  训练集均值: {train_mean:.2f}, 中位数: {train_median:.2f}")
    print(f"  预测均值: {pred_mean:.2f}, 中位数: {pred_median:.2f}")
    
    # V24改进：更精细的分位数校准
    quantiles = [2, 5, 10, 25, 50, 75, 90, 95, 98]
    train_quantiles = np.percentile(y_train, quantiles)
    pred_quantiles = np.percentile(predictions, quantiles)
    
    # 计算分位数校准因子
    quantile_factors = train_quantiles / pred_quantiles
    quantile_factors = np.clip(quantile_factors, 0.7, 1.3)
    
    # 应用分位数校准
    calibrated_predictions = predictions.copy()
    for i in range(len(predictions)):
        pred_val = predictions[i]
        
        # 找到对应的分位数区间
        for j in range(len(quantiles) - 1):
            if pred_val <= pred_quantiles[j + 1]:
                # 线性插值
                if j == 0:
                    factor = quantile_factors[0]
                else:
                    t = (pred_val - pred_quantiles[j]) / (pred_quantiles[j + 1] - pred_quantiles[j])
                    factor = quantile_factors[j] * (1 - t) + quantile_factors[j + 1] * t
                break
        else:
            factor = quantile_factors[-1]
        
        calibrated_predictions[i] *= factor
    
    # V23的基础校准作为后备
    calibration_factor = train_mean / pred_mean if pred_mean > 0 else 1.0
    calibration_factor = np.clip(calibration_factor, 0.85, 1.15)
    
    # V24改进：混合校准 - 80%分位数校准 + 20%均值校准
    final_predictions = calibrated_predictions * 0.8 + predictions * calibration_factor * 0.2
    final_predictions = np.maximum(final_predictions, 0)
    
    print(f"  分位数校准因子范围: {quantile_factors.min():.3f} - {quantile_factors.max():.3f}")
    print(f"  均值校准因子: {calibration_factor:.4f}")
    
    return final_predictions

def create_simplified_analysis(y_train, predictions, scores_info):
    """
    创建简化分析图表
    """
    print("生成简化分析图表...")
    
    # 创建保存目录
    analysis_dir = get_user_data_path()
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 价格分布对比
    axes[0, 0].hist(y_train, bins=50, alpha=0.7, label='训练集真实价格', color='blue', density=True)
    axes[0, 0].hist(predictions, bins=50, alpha=0.7, label='V24预测价格', color='red', density=True)
    axes[0, 0].set_xlabel('价格')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('V24价格分布对比')
    axes[0, 0].legend()
    
    # 2. 模型性能对比
    models = ['LightGBM', 'XGBoost', 'CatBoost']
    scores = [np.mean(scores_info['lgb_scores']), 
              np.mean(scores_info['xgb_scores']), 
              np.mean(scores_info['cat_scores'])]
    
    bars = axes[0, 1].bar(models, scores, color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].axhline(y=497.6, color='orange', linestyle='--', label='V23基准(497.6)')
    axes[0, 1].axhline(y=490, color='red', linestyle='--', label='目标线(490)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('V24各模型验证性能')
    axes[0, 1].legend()
    
    # 3. 预测值分布
    axes[1, 0].hist(predictions, bins=50, alpha=0.7, color='red')
    axes[1, 0].set_xlabel('预测价格')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('V24预测值分布')
    
    # 4. 版本对比总结
    comparison_text = f"""
    V24简化优化版本总结:
    
    基于V23的497.6048分基础:
    ✅ V23: 精准突破500分
    ✅ V22: 平衡策略和502分基础
    
    V24新增核心优化:
    🚀 目标编码: 增加分类特征信息
    🚀 增强特征: 更多交互和多项式
    🚀 参数优化: 基于V23的精细调优
    🚀 改进校准: 更精细的分位数校准
    
    训练集统计:
    样本数: {len(y_train):,}
    均值: {y_train.mean():.2f}
    标准差: {y_train.std():.2f}
    
    预测集统计:
    样本数: {len(predictions):,}
    均值: {predictions.mean():.2f}
    标准差: {predictions.std():.2f}
    
    验证性能:
    LightGBM: {np.mean(scores_info["lgb_scores"]):.2f}
    XGBoost: {np.mean(scores_info["xgb_scores"]):.2f}
    CatBoost: {np.mean(scores_info["cat_scores"]):.2f}
    
    🎯 目标: 优化核心策略，冲击更好成绩!
    """
    axes[1, 1].text(0.05, 0.95, comparison_text, transform=axes[1, 1].transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('V24简化优化总结')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(analysis_dir, 'modeling_v24_simplified_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"V24分析图表已保存到: {chart_path}")
    plt.show()

def v24_simplified_optimize():
    """
    V24简化优化模型训练流程
    """
    print("=" * 80)
    print("开始V24简化优化模型训练")
    print("基于V23的497.6048分基础，测试核心优化策略")
    print("目标：优化核心策略，冲击更好成绩")
    print("=" * 80)
    
    # 步骤1: 简化数据预处理
    print("\n步骤1: 简化数据预处理...")
    train_df, test_df = simplified_preprocessing()
    
    # 步骤2: 简化特征工程
    print("\n步骤2: 简化特征工程...")
    train_df = create_simplified_features(train_df)
    test_df = create_simplified_features(test_df)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 步骤3: 特征缩放
    print("\n步骤3: 特征缩放...")
    scaler = RobustScaler()
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_features:
        if col in X_train.columns and col in X_test.columns:
            # 检查无穷大值
            inf_mask = np.isinf(X_train[col]) | np.isinf(X_test[col])
            if inf_mask.any():
                X_train.loc[inf_mask[inf_mask.index.isin(X_train.index)].index, col] = 0
                X_test.loc[inf_mask[inf_mask.index.isin(X_test.index)].index, col] = 0
            
            X_train[col] = X_train[col].fillna(X_train[col].median())
            X_test[col] = X_test[col].fillna(X_train[col].median())
            
            if X_train[col].std() > 1e-8:
                X_train[col] = scaler.fit_transform(X_train[[col]])
                X_test[col] = scaler.transform(X_test[[col]])
    
    # 步骤4: 训练优化模型
    print("\n步骤4: 训练优化模型...")
    lgb_pred, xgb_pred, cat_pred, scores_info = train_optimized_models(
        X_train, y_train, X_test)
    
    # 步骤5: 优化集成
    print("\n步骤5: 优化集成...")
    ensemble_pred = optimized_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info)
    
    # 步骤6: 优化校准
    print("\n步骤6: 优化校准...")
    final_predictions = optimized_calibration(ensemble_pred, y_train)
    
    # 步骤7: 创建分析图表
    print("\n步骤7: 生成分析图表...")
    create_simplified_analysis(y_train, final_predictions, scores_info)
    
    # 最终统计
    print(f"\nV24最终预测统计:")
    print(f"均值: {final_predictions.mean():.2f}")
    print(f"标准差: {final_predictions.std():.2f}")
    print(f"范围: {final_predictions.min():.2f} - {final_predictions.max():.2f}")
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'SaleID': test_df['SaleID'] if 'SaleID' in test_df.columns else test_df.index,
        'price': final_predictions
    })
    
    # 保存结果
    result_dir = get_project_path('prediction_result')
    os.makedirs(result_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"modeling_v24_simplified_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\nV24结果已保存到: {result_file}")
    
    # 生成优化报告
    print("\n" + "=" * 80)
    print("V24简化优化总结")
    print("=" * 80)
    print("✅ 基于V23的497.6048分优秀基础")
    print("✅ 目标编码 - 增加分类特征信息")
    print("✅ 增强特征 - 更多交互和多项式特征")
    print("✅ 参数优化 - 基于V23的精细调优")
    print("✅ 改进校准 - 更精细的分位数校准")
    print("🚀 目标：优化核心策略，冲击更好成绩!")
    print("=" * 80)
    
    return final_predictions, scores_info

if __name__ == "__main__":
    test_pred, scores_info = v24_simplified_optimize()
    print("V24简化优化完成! 期待优化效果! 🚀")
