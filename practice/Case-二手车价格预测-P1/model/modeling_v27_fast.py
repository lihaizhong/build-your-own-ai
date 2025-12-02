"""
V27快速测试版本模型 - 精准突破450分核心策略测试

基于V26的497.9590分基础，快速测试核心优化策略:
1. 增强特征工程 - 增加高价值特征到40个
2. 优化模型参数 - 在V26基础上微调
3. 改进集成策略 - 动态权重调整
4. 精细化校准 - 分位数校准

目标：快速验证优化效果，为完整版提供参考
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

def enhanced_preprocessing():
    """
    增强的数据预处理 - 基于V26但增加目标编码
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
        all_df['power_is_low'] = (all_df['power'] <= 100).astype(int)
        all_df['power_is_high'] = (all_df['power'] > 200).astype(int)
        
        # 多种变换
        all_df['log_power'] = np.log1p(np.maximum(all_df['power'], 1))
        all_df['sqrt_power'] = np.sqrt(np.maximum(all_df['power'], 0))
    
    # 增强的分类特征处理
    categorical_cols = ['fuelType', 'gearbox', 'bodyType', 'model', 'brand']
    for col in categorical_cols:
        if col in all_df.columns:
            # 基础处理
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
    
    # 增强的时间特征工程
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year
    all_df['car_age'] = all_df['car_age'].fillna(0).astype(int)
    all_df['reg_month'] = all_df['regDate'].dt.month.fillna(6).astype(int)
    
    # 季节特征
    all_df['reg_season'] = all_df['reg_month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
    
    all_df.drop(columns=['regDate'], inplace=True)
    
    # 增强的品牌统计特征
    if 'price' in all_df.columns:
        # 品牌统计
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'std', 'median', 'count']).reset_index()
        global_mean = all_df['price'].mean()
        global_std = all_df['price'].std()
        
        # 平滑处理
        smooth_factor = 50
        brand_stats['smooth_mean'] = ((brand_stats['mean'] * brand_stats['count'] + 
                                     global_mean * smooth_factor) / (brand_stats['count'] + smooth_factor))
        
        # 映射特征
        brand_mean_map = brand_stats.set_index('brand')['smooth_mean']
        brand_median_map = brand_stats.set_index('brand')['median']
        
        all_df['brand_avg_price'] = all_df['brand'].map(brand_mean_map).fillna(global_mean)
        all_df['brand_median_price'] = all_df['brand'].map(brand_median_map).fillna(global_mean)
        
        # 价格偏差特征
        if 'model' in all_df.columns:
            model_stats = all_df.groupby('model')['price'].agg(['mean', 'count']).reset_index()
            smooth_factor = 30
            model_stats['smooth_mean'] = ((model_stats['mean'] * model_stats['count'] + 
                                         global_mean * smooth_factor) / (model_stats['count'] + smooth_factor))
            model_mean_map = model_stats.set_index('model')['smooth_mean']
            all_df['model_avg_price'] = all_df['model'].map(model_mean_map).fillna(global_mean)
            all_df['price_vs_brand'] = all_df['model_avg_price'] - all_df['brand_avg_price']
    
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

def create_enhanced_features(df):
    """
    创建增强特征 - 增加到40个核心特征
    """
    df = df.copy()
    
    # 核心业务特征
    if 'power' in df.columns and 'car_age' in df.columns:
        df['power_age_ratio'] = df['power'] / (df['car_age'] + 1)
        df['power_age_diff'] = df['power'] - df['car_age'] * 10
    
    if 'kilometer' in df.columns and 'car_age' in df.columns:
        car_age_safe = np.maximum(df['car_age'], 0.1)
        df['km_per_year'] = df['kilometer'] / car_age_safe
        df['km_per_year'] = np.clip(df['km_per_year'], 0, 40000)
        df['km_age_ratio'] = df['kilometer'] / (df['car_age'] + 1)
    
    # 价格相关特征
    if 'brand_avg_price' in df.columns and 'model_avg_price' in df.columns:
        df['price_ratio_brand_model'] = df['model_avg_price'] / (df['brand_avg_price'] + 1)
        df['price_diff_brand_model'] = df['model_avg_price'] - df['brand_avg_price']
    
    # 增强的分段特征
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 2, 5, 8, 12, float('inf')], 
                              labels=['new', 'young', 'medium', 'old', 'very_old'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    if 'kilometer' in df.columns:
        df['km_segment'] = pd.cut(df['kilometer'], bins=[-1, 30000, 80000, 120000, 160000, float('inf')], 
                                 labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        df['km_segment'] = df['km_segment'].cat.codes
    
    if 'power' in df.columns:
        df['power_segment'] = pd.cut(df['power'], bins=[-1, 50, 100, 150, 200, 300, float('inf')], 
                                    labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme'])
        df['power_segment'] = df['power_segment'].cat.codes
    
    # 多种变换特征
    for col in ['car_age', 'kilometer', 'power']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(np.maximum(df[col], 0))
            df[f'sqrt_{col}'] = np.sqrt(np.maximum(df[col], 0))
    
    # v特征统计增强
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 5:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_range'] = df[v_cols].max(axis=1) - df[v_cols].min(axis=1)
        df['v_skew'] = df[v_cols].skew(axis=1).fillna(0)
        df['v_sum'] = df[v_cols].sum(axis=1)
        
        # v特征分组
        v_positive = [col for col in v_cols if df[col].mean() > 0]
        v_negative = [col for col in v_cols if df[col].mean() <= 0]
        
        if v_positive:
            df['v_pos_mean'] = df[v_positive].mean(axis=1)
        if v_negative:
            df['v_neg_mean'] = df[v_negative].mean(axis=1)
    
    # 交互特征
    if 'brand' in df.columns and 'bodyType' in df.columns:
        df['brand_bodyType'] = df['brand'].astype(str) + '_' + df['bodyType'].astype(str)
        le = LabelEncoder()
        df['brand_bodyType'] = le.fit_transform(df['brand_bodyType'])
    
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
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            df[col] = np.clip(df[col], q01, q99)
    
    return df

def train_enhanced_ensemble(X_train, y_train, X_test):
    """
    训练增强集成模型 - 优化参数，动态权重
    """
    print("训练增强集成模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 5折交叉验证（快速版本）
    y_bins = pd.qcut(y_train, q=5, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # V27优化参数 - 平衡正则化与拟合能力
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'num_leaves': 43,        # 适度增加叶子节点
        'max_depth': 9,          # 适度增加深度
        'learning_rate': 0.06,   # 适度提高学习率
        'feature_fraction': 0.85, # 适度提高特征采样
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'lambda_l1': 0.2,        # 适度降低L1正则化
        'lambda_l2': 0.2,        # 适度降低L2正则化
        'min_child_samples': 18, # 适度降低最小样本数
        'random_state': 42,
    }
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': 9,          # 适度增加深度
        'learning_rate': 0.06,   # 适度提高学习率
        'subsample': 0.85,       # 适度提高采样
        'colsample_bytree': 0.85,
        'reg_alpha': 0.6,        # 适度降低L1正则化
        'reg_lambda': 0.6,       # 适度降低L2正则化
        'min_child_weight': 8,   # 适度降低最小权重
        'random_state': 42
    }
    
    catboost_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'depth': 9,              # 适度增加深度
        'learning_rate': 0.06,   # 适度提高学习率
        'iterations': 1000,      # 减少迭代次数（快速版本）
        'l2_leaf_reg': 2.5,      # 适度降低L2正则化
        'random_strength': 0.3,  # 适度降低随机性
        'random_seed': 42,
        'verbose': False
    }
    
    # 存储预测
    lgb_predictions = np.zeros(len(X_test))
    xgb_predictions = np.zeros(len(X_test))
    cat_predictions = np.zeros(len(X_test))
    
    lgb_cv_pred = np.zeros(len(X_train))
    xgb_cv_pred = np.zeros(len(X_train))
    cat_cv_pred = np.zeros(len(X_train))
    
    lgb_scores, xgb_scores, cat_scores = [], [], []
    
    # 交叉验证训练
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_bins), 1):
        print(f"训练第 {fold} 折...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_log, y_val_log = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=2000)
        lgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)])
        
        lgb_predictions += np.expm1(np.array(lgb_model.predict(X_test))) / 5
        lgb_cv_pred[val_idx] = np.expm1(np.array(lgb_model.predict(X_val)))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_cv_pred[val_idx])
        lgb_scores.append(lgb_mae)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=2000, early_stopping_rounds=100)
        xgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     verbose=False)
        
        xgb_predictions += np.expm1(xgb_model.predict(X_test)) / 5
        xgb_cv_pred[val_idx] = np.expm1(xgb_model.predict(X_val))
        xgb_mae = mean_absolute_error(np.expm1(y_val_log), xgb_cv_pred[val_idx])
        xgb_scores.append(xgb_mae)
        
        # CatBoost
        cat_model = CatBoostRegressor(**catboost_params)
        cat_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     early_stopping_rounds=100, 
                     verbose=False)
        
        cat_predictions += np.expm1(cat_model.predict(X_test)) / 5
        cat_cv_pred[val_idx] = np.expm1(cat_model.predict(X_val))
        cat_mae = mean_absolute_error(np.expm1(y_val_log), cat_cv_pred[val_idx])
        cat_scores.append(cat_mae)
        
        print(f"  LGB: {lgb_mae:.2f}, XGB: {xgb_mae:.2f}, CAT: {cat_mae:.2f}")
    
    print(f"\n验证分数:")
    print(f"  LightGBM: {np.mean(lgb_scores):.2f} (±{np.std(lgb_scores):.2f})")
    print(f"  XGBoost: {np.mean(xgb_scores):.2f} (±{np.std(xgb_scores):.2f})")
    print(f"  CatBoost: {np.mean(cat_scores):.2f} (±{np.std(cat_scores):.2f})")
    
    # 动态权重调整
    model_scores = {
        'lgb': np.mean(lgb_scores),
        'xgb': np.mean(xgb_scores),
        'cat': np.mean(cat_scores)
    }
    
    # 计算权重（分数越低权重越高）
    inv_scores = {model: 1/score for model, score in model_scores.items()}
    total_inv = sum(inv_scores.values())
    weights = {model: inv_score/total_inv for model, inv_score in inv_scores.items()}
    
    # 考虑模型稳定性
    stability = {
        'lgb': 1 / (1 + np.std(lgb_scores)),
        'xgb': 1 / (1 + np.std(xgb_scores)),
        'cat': 1 / (1 + np.std(cat_scores))
    }
    
    # 综合权重
    final_weights = {}
    total_weight = 0
    for model in model_scores.keys():
        final_weights[model] = weights[model] * stability[model]
        total_weight += final_weights[model]
    
    for model in final_weights:
        final_weights[model] /= total_weight
    
    # 最终集成
    final_predictions = (
        final_weights['lgb'] * lgb_predictions +
        final_weights['xgb'] * xgb_predictions +
        final_weights['cat'] * cat_predictions
    )
    
    print(f"\n模型权重:")
    for model, weight in final_weights.items():
        print(f"  {model.upper()}: {weight:.3f}")
    
    return final_predictions, {
        'lgb_scores': lgb_scores,
        'xgb_scores': xgb_scores,
        'cat_scores': cat_scores,
        'weights': final_weights
    }

def enhanced_calibration(predictions, y_train):
    """
    增强校准 - 分位数校准+动态调整
    """
    print("执行增强校准...")
    
    # 基础统计
    train_mean = y_train.mean()
    train_std = y_train.std()
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    
    print(f"\n校准前:")
    print(f"  训练集均值: {train_mean:.2f}, 标准差: {train_std:.2f}")
    print(f"  预测均值: {pred_mean:.2f}, 标准差: {pred_std:.2f}")
    
    # 第一阶段：均值校准
    if pred_mean > 0:
        mean_calibration = train_mean / pred_mean
        mean_calibration = np.clip(mean_calibration, 0.8, 1.2)
    else:
        mean_calibration = 1.0
    
    calibrated_predictions = predictions * mean_calibration
    
    # 第二阶段：标准差校准
    calib_std = calibrated_predictions.std()
    if calib_std > 0:
        std_calibration = train_std / calib_std
        std_calibration = np.clip(std_calibration, 0.8, 1.2)
    else:
        std_calibration = 1.0
    
    # 应用标准差校准
    calibrated_predictions = (calibrated_predictions - calibrated_predictions.mean()) * std_calibration + train_mean
    
    # 第三阶段：分位数校准（简化版）
    try:
        # 使用训练集的分位数进行调整
        train_quantiles = np.percentile(y_train, [10, 25, 50, 75, 90])
        pred_quantiles = np.percentile(calibrated_predictions, [10, 25, 50, 75, 90])
        
        # 简化的分段映射
        def quantile_mapping(x):
            if x <= pred_quantiles[0]:
                return train_quantiles[0]
            elif x <= pred_quantiles[2]:
                ratio = (x - pred_quantiles[0]) / (pred_quantiles[2] - pred_quantiles[0])
                return train_quantiles[0] + ratio * (train_quantiles[2] - train_quantiles[0])
            elif x <= pred_quantiles[4]:
                ratio = (x - pred_quantiles[2]) / (pred_quantiles[4] - pred_quantiles[2])
                return train_quantiles[2] + ratio * (train_quantiles[4] - train_quantiles[2])
            else:
                return train_quantiles[4]
        
        # 应用分位数校准
        quantile_calibrated = np.array([quantile_mapping(x) for x in calibrated_predictions])
        
        # 混合原始预测和分位数校准结果
        calibrated_predictions = 0.8 * calibrated_predictions + 0.2 * quantile_calibrated
        
    except Exception as e:
        print(f"分位数校准失败，使用基础校准: {e}")
    
    # 确保预测值为正
    calibrated_predictions = np.maximum(calibrated_predictions, 0)
    
    print(f"\n校准后:")
    print(f"  均值校准因子: {mean_calibration:.3f}")
    print(f"  标准差校准因子: {std_calibration:.3f}")
    print(f"  预测均值: {calibrated_predictions.mean():.2f}")
    print(f"  预测标准差: {calibrated_predictions.std():.2f}")
    
    return calibrated_predictions

def v27_fast_test():
    """
    V27快速测试版本训练流程
    """
    print("=" * 80)
    print("开始V27快速测试模型训练")
    print("基于V26的497.96分基础，快速测试核心优化策略")
    print("目标：验证优化效果，为完整版提供参考")
    print("=" * 80)
    
    # 步骤1: 增强数据预处理
    print("\n步骤1: 增强数据预处理...")
    train_df, test_df = enhanced_preprocessing()
    
    # 步骤2: 创建增强特征
    print("\n步骤2: 创建增强特征...")
    train_df = create_enhanced_features(train_df)
    test_df = create_enhanced_features(test_df)
    
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
    
    # 使用相关性筛选
    correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    top_features = correlations.head(40).index.tolist()
    
    # 确保包含重要的业务特征
    business_features = ['car_age', 'power', 'kilometer', 'brand', 'model']
    for feature in business_features:
        if feature in X_train.columns and feature not in top_features:
            top_features.append(feature)
    
    X_train = X_train[top_features]
    X_test = X_test[top_features]
    
    print(f"筛选后特征数量: {len(top_features)}")
    
    # 步骤4: 特征缩放
    print("\n步骤4: 特征缩放...")
    
    # 对数值特征进行Robust缩放
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X_train[col].std() > 1e-8:
            # 检查无穷大值
            inf_mask = np.isinf(X_train[col]) | np.isinf(X_test[col])
            if inf_mask.any():
                X_train.loc[inf_mask[inf_mask.index.isin(X_train.index)].index, col] = 0
                X_test.loc[inf_mask[inf_mask.index.isin(X_test.index)].index, col] = 0
            
            X_train[col] = X_train[col].fillna(X_train[col].median())
            X_test[col] = X_test[col].fillna(X_train[col].median())
            
            scaler = RobustScaler()
            X_train[col] = scaler.fit_transform(X_train[[col]])
            X_test[col] = scaler.transform(X_test[[col]])
    
    # 步骤5: 训练增强集成模型
    print("\n步骤5: 训练增强集成模型...")
    ensemble_pred, scores_info = train_enhanced_ensemble(X_train, y_train, X_test)
    
    # 步骤6: 增强校准
    print("\n步骤6: 增强校准...")
    final_predictions = enhanced_calibration(ensemble_pred, y_train)
    
    # 最终统计
    print(f"\nV27快速测试最终预测统计:")
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
    result_file = os.path.join(result_dir, f"modeling_v27_fast_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\nV27快速测试结果已保存到: {result_file}")
    
    # 生成优化报告
    print("\n" + "=" * 80)
    print("V27快速测试总结")
    print("=" * 80)
    print("✅ 基于V26的497.96分基础，快速测试核心优化策略")
    print("✅ 增强特征工程 - 40个核心特征，目标编码")
    print("✅ 优化模型参数 - 平衡正则化与拟合能力")
    print("✅ 改进集成策略 - 动态权重调整")
    print("✅ 精细化校准 - 分位数校准+动态调整")
    print("✅ 智能特征选择 - 基于相关性的动态筛选")
    print("🚀 快速测试完成，期待优化效果!")
    print("=" * 80)
    
    return final_predictions, scores_info

if __name__ == "__main__":
    test_pred, scores_info = v27_fast_test()
    print("V27快速测试完成! 🚀")
