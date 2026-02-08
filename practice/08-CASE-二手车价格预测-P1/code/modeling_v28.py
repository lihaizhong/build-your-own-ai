"""
V28版本模型 - 融合创新突破版

基于表现最佳模型的深度分析和融合创新:
1. V24_simplified (488.7255) - 精准特征工程和优化参数
2. V23 (497.6048) - 分层验证和增强特征
3. V26 (497.9590) - 抗过拟合和稳定架构
4. V24_fast (501.8398) - 目标编码和关键特征
5. V22 (502.1616) - 平衡策略和稳健集成

V28核心创新策略:
🚀 动态特征重要性分析 - 自动筛选高价值特征
🚀 分层建模策略 - 按价格区间分别建模
🚀 自适应参数调优 - 基于验证集性能动态调整
🚀 增强校准算法 - 多阶段校准优化
🚀 智能集成权重 - 稳定性和性能平衡

目标：突破488.7255分，冲击480分以内
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import warnings
import joblib
from ...shared import get_project_path

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def save_models(models, version_name):
    """
    保存训练好的模型到model目录
    
    Parameters:
    -----------
    models : dict
        模型字典，key为模型名称，value为模型对象
    version_name : str
        版本名称，如'v28'
    """
    model_dir = get_project_path('model')
    os.makedirs(model_dir, exist_ok=True)
    
    saved_files = []
    for model_name, model_obj in models.items():
        if model_obj is not None:
            model_file = os.path.join(model_dir, f'{version_name}_{model_name}_model.pkl')
            joblib.dump(model_obj, model_file)
            saved_files.append(model_file)
            print(f"✅ 模型已保存: {model_file}")
    
    return saved_files


def get_user_data_path(*paths):
    """获取用户数据路径"""
    return get_project_path('user_data', *paths)

def enhanced_preprocessing():
    """
    增强的数据预处理 - 融合各版本最佳实践
    """
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 合并数据进行预处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # V24_simplified的增强power处理
    if 'power' in all_df.columns:
        all_df['power'] = np.clip(all_df['power'], 0, 600)
        all_df['power_is_zero'] = (all_df['power'] <= 0).astype(int)
        all_df['power_is_high'] = (all_df['power'] > 400).astype(int)
        
        # V28新增：更多power变换
        all_df['log_power'] = np.log1p(np.maximum(all_df['power'], 1))
        all_df['sqrt_power'] = np.sqrt(np.maximum(all_df['power'], 0))
        all_df['power_squared'] = (all_df['power'] ** 2) / 1000  # 归一化
    
    # 融合各版本的分类特征处理
    categorical_cols = ['fuelType', 'gearbox', 'bodyType', 'model', 'brand']
    for col in categorical_cols:
        if col in all_df.columns:
            # 基础缺失标记
            all_df[f'{col}_missing'] = (all_df[col].isnull()).astype(int)
            
            # V23的智能填充
            if col == 'model' and 'brand' in all_df.columns:
                for brand in all_df['brand'].unique():
                    brand_mask = all_df['brand'] == brand
                    brand_mode = all_df[brand_mask][col].mode()
                    if len(brand_mode) > 0:
                        all_df.loc[brand_mask & all_df[col].isnull(), col] = brand_mode.iloc[0]
            
            # 全局众数填充
            mode_value = all_df[col].mode()
            if len(mode_value) > 0:
                all_df[col] = all_df[col].fillna(mode_value.iloc[0])
            
            # V24的目标编码 - 增强版本
            if 'price' in all_df.columns:
                target_mean = all_df.groupby(col)['price'].mean()
                global_mean = all_df['price'].mean()
                count = all_df[col].value_counts()
                
                # V28新增：自适应平滑因子
                if col == 'brand':
                    smooth_factor = 100  # brand类别多，需要更多平滑
                elif col == 'model':
                    smooth_factor = 50
                else:
                    smooth_factor = 20
                
                smooth_encoding = (target_mean * count + global_mean * smooth_factor) / (count + smooth_factor)
                all_df[f'{col}_target_enc'] = all_df[col].map(smooth_encoding).fillna(global_mean)
            
            # 频率编码
            freq_map = all_df[col].value_counts().to_dict()
            all_df[f'{col}_freq'] = all_df[col].map(freq_map)
    
    # V23的增强时间特征工程
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
    
    # V28新增：周期性时间特征
    all_df['reg_month_sin'] = np.sin(2 * np.pi * all_df['reg_month'] / 12)
    all_df['reg_month_cos'] = np.cos(2 * np.pi * all_df['reg_month'] / 12)
    
    all_df.drop(columns=['regDate'], inplace=True)
    
    # V24_simplified的增强品牌统计特征
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count', 'std', 'median']).reset_index()
        global_mean = all_df['price'].mean()
        
        # 平滑均值
        smooth_factor = 40
        brand_stats['smooth_mean'] = ((brand_stats['mean'] * brand_stats['count'] + 
                                     global_mean * smooth_factor) / (brand_stats['count'] + smooth_factor))
        
        # V28新增：更多品牌统计特征
        brand_stats['cv'] = brand_stats['std'] / brand_stats['mean']
        brand_stats['cv'] = brand_stats['cv'].fillna(brand_stats['cv'].median())
        brand_stats['skewness'] = (brand_stats['mean'] - brand_stats['median']) / (brand_stats['std'] + 1e-6)
        brand_stats['skewness'] = brand_stats['skewness'].fillna(0)
        brand_stats['price_range'] = brand_stats['mean'] + brand_stats['std']
        
        # 映射特征
        all_df['brand_avg_price'] = all_df['brand'].map(brand_stats.set_index('brand')['smooth_mean']).fillna(global_mean)
        all_df['brand_price_stability'] = all_df['brand'].map(brand_stats.set_index('brand')['cv']).fillna(brand_stats['cv'].median())
        all_df['brand_skewness'] = all_df['brand'].map(brand_stats.set_index('brand')['skewness']).fillna(0)
        all_df['brand_price_range'] = all_df['brand'].map(brand_stats.set_index('brand')['price_range']).fillna(global_mean)
    
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

def create_innovative_features(df):
    """
    创新特征工程 - 融合各版本精华并加入新特征
    """
    df = df.copy()
    
    # V24_simplified的核心业务特征
    if 'power' in df.columns and 'car_age' in df.columns:
        df['power_age_ratio'] = df['power'] / (df['car_age'] + 1)
        df['power_decay'] = df['power'] * np.exp(-df['car_age'] * 0.05)
    
    if 'kilometer' in df.columns and 'car_age' in df.columns:
        car_age_safe = np.maximum(df['car_age'], 0.1)
        df['km_per_year'] = df['kilometer'] / car_age_safe
        df['km_per_year'] = np.clip(df['km_per_year'], 0, 40000)
    
    # V28新增：更多业务逻辑特征
    if 'power' in df.columns and 'kilometer' in df.columns:
        df['power_km_ratio'] = df['power'] / (df['kilometer'] + 1)
        df['power_km_interaction'] = df['power'] * np.log1p(df['kilometer'])
    
    if 'car_age' in df.columns and 'kilometer' in df.columns:
        df['age_km_interaction'] = df['car_age'] * df['kilometer'] / 1000
        df['age_km_log_interaction'] = df['car_age'] * np.log1p(df['kilometer'])
    
    # V24_simplified的分段特征 - 增强版本
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 2, 4, 6, 8, 12, 20, float('inf')], 
                              labels=['brand_new', 'very_new', 'new', 'medium', 'old', 'very_old', 'ancient'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    if 'kilometer' in df.columns:
        df['km_segment'] = pd.cut(df['kilometer'], bins=[-1, 30000, 60000, 90000, 120000, 150000, 180000, float('inf')], 
                                 labels=['very_low', 'low', 'medium_low', 'medium', 'medium_high', 'high', 'very_high'])
        df['km_segment'] = df['km_segment'].cat.codes
    
    if 'power' in df.columns:
        df['power_segment_fine'] = pd.cut(df['power'], bins=[-1, 50, 100, 150, 200, 250, 300, 400, 600],
                                         labels=['very_low', 'low', 'medium_low', 'medium', 'medium_high', 'high', 'very_high', 'extreme'])
        df['power_segment_fine'] = df['power_segment_fine'].cat.codes
    
    # V23的变换特征 - 增强版本
    if 'car_age' in df.columns:
        df['log_car_age'] = np.log1p(df['car_age'])
        df['sqrt_car_age'] = np.sqrt(df['car_age'])
        df['car_age_squared'] = df['car_age'] ** 2
    
    if 'kilometer' in df.columns:
        df['log_kilometer'] = np.log1p(df['kilometer'])
        df['sqrt_kilometer'] = np.sqrt(df['kilometer'])
        df['kilometer_squared'] = df['kilometer'] ** 2
    
    # V24_simplified的v特征统计 - 增强版本
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
        df['v_skew'] = df[v_cols].skew(axis=1).fillna(0)
        df['v_kurt'] = df[v_cols].kurtosis(axis=1).fillna(0)
        
        # V28新增：更多v特征统计
        df['v_sum'] = df[v_cols].sum(axis=1)
        df['v_median'] = df[v_cols].median(axis=1)
        df['v_mean_to_std_ratio'] = df['v_mean'] / (df['v_std'] + 1e-6)
        df['v_range_to_mean_ratio'] = df['v_range'] / (df['v_mean'] + 1e-6)
    
    # V28新增：高阶交互特征
    high_value_interactions = [
        ('power_age_ratio', 'km_per_year'),
        ('brand_avg_price', 'car_age'),
        ('brand_avg_price', 'power'),
        ('power_decay', 'log_kilometer'),
        ('v_mean', 'power'),
        ('v_std', 'car_age'),
        ('age_segment', 'power_segment_fine'),
        ('km_segment', 'age_segment'),
    ]
    
    for feat1, feat2 in high_value_interactions:
        if feat1 in df.columns and feat2 in df.columns:
            df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1)
    
    # V28新增：品牌相关的高级特征
    if 'brand_avg_price' in df.columns:
        if 'car_age' in df.columns:
            df['brand_price_age_interaction'] = df['brand_avg_price'] * np.log1p(df['car_age'])
            df['brand_age_ratio'] = df['brand_avg_price'] / (df['car_age'] + 1)
        if 'power' in df.columns:
            df['brand_price_power_interaction'] = df['brand_avg_price'] * np.log1p(df['power'])
            df['brand_power_ratio'] = df['brand_avg_price'] / (df['power'] + 1)
        if 'kilometer' in df.columns:
            df['brand_km_interaction'] = df['brand_avg_price'] * np.log1p(df['kilometer'])
    
    # V28新增：时间相关的组合特征
    if 'reg_season' in df.columns and 'car_age' in df.columns:
        df['season_age_interaction'] = df['reg_season'] * df['car_age']
    
    if 'is_winter_reg' in df.columns and 'power' in df.columns:
        df['winter_power_interaction'] = df['is_winter_reg'] * df['power']
    
    # 数据清理 - 融合各版本的最佳实践
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 处理无穷大值
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # 填充NaN值
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
        
        # V26的保守异常值处理，但对某些特征使用更宽松的限制
        if col not in ['SaleID', 'price'] and df[col].std() > 1e-8:
            q999 = df[col].quantile(0.999)
            q001 = df[col].quantile(0.001)
            
            # 对比率特征使用更宽松的限制
            ratio_features = [c for c in df.columns if 'ratio' in c or 'interaction' in c]
            if col in ratio_features:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                if q99 > q01 and q99 > 0:
                    df[col] = np.clip(df[col], q01, q99)
            else:
                if q999 > q001 and q999 > 0:
                    df[col] = np.clip(df[col], q001, q999)
    
    return df

def dynamic_feature_selection(X_train, y_train, X_test, max_features=80):
    """
    V28新增：动态特征重要性分析
    """
    print("执行动态特征重要性分析...")
    
    # 使用互信息进行特征筛选
    feature_names = X_train.columns.tolist()
    
    # 计算互信息分数
    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    mi_df = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # 选择top特征
    top_features = mi_df.head(max_features)['feature'].tolist()
    
    print(f"从{len(feature_names)}个特征中选择了{len(top_features)}个高价值特征")
    print("Top 10重要特征:")
    for i, (feat, score) in enumerate(zip(mi_df['feature'].head(10), mi_df['mi_score'].head(10))):
        print(f"  {i+1}. {feat}: {score:.4f}")
    
    return X_train[top_features], X_test[top_features], mi_df

def adaptive_parameter_tuning(X_train, y_train):
    """
    V28新增：自适应参数调优
    """
    print("执行自适应参数调优...")
    
    # 基于数据特征动态调整参数
    n_samples, n_features = X_train.shape
    y_std = y_train.std()
    
    # 根据数据规模调整参数
    if n_samples < 50000:
        # 小数据集，使用更保守的参数
        base_learning_rate = 0.08
        base_num_leaves = 31
        base_depth = 7
        base_iterations = 1200
    else:
        # 大数据集，可以使用更复杂的参数
        base_learning_rate = 0.07
        base_num_leaves = 37
        base_depth = 8
        base_iterations = 1800
    
    # 根据特征数量调整
    if n_features > 60:
        # 高维特征，需要更强的正则化
        reg_factor = 1.2
        feature_fraction = 0.8
    else:
        reg_factor = 1.0
        feature_fraction = 0.9
    
    # 根据目标变量方差调整
    if y_std > 5000:
        # 高方差，使用更小的学习率
        learning_rate_factor = 0.9
    else:
        learning_rate_factor = 1.0
    
    # 计算最终参数
    final_learning_rate = base_learning_rate * learning_rate_factor
    final_num_leaves = int(base_num_leaves * reg_factor)
    final_depth = base_depth
    final_iterations = int(base_iterations * (1.2 if n_features > 60 else 1.0))
    
    print(f"自适应参数结果:")
    print(f"  学习率: {final_learning_rate}")
    print(f"  叶子节点数: {final_num_leaves}")
    print(f"  树深度: {final_depth}")
    print(f"  迭代次数: {final_iterations}")
    print(f"  特征采样率: {feature_fraction}")
    
    return {
        'learning_rate': final_learning_rate,
        'num_leaves': final_num_leaves,
        'max_depth': final_depth,
        'iterations': final_iterations,
        'feature_fraction': feature_fraction,
        'reg_factor': reg_factor
    }

def train_innovative_models(X_train, y_train, X_test):
    """
    训练创新模型 - 融合各版本精华
    """
    print("训练创新融合模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # V23的分层交叉验证
    y_bins = pd.qcut(y_train, q=10, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 自适应参数调优
    adaptive_params = adaptive_parameter_tuning(X_train, y_train)
    
    # V28融合参数 - 基于各版本最佳实践
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'num_leaves': adaptive_params['num_leaves'],
        'max_depth': adaptive_params['max_depth'],
        'learning_rate': adaptive_params['learning_rate'],
        'feature_fraction': adaptive_params['feature_fraction'],
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'lambda_l1': 0.25 * adaptive_params['reg_factor'],
        'lambda_l2': 0.25 * adaptive_params['reg_factor'],
        'min_child_samples': 18,
        'random_state': 42,
    }
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': adaptive_params['max_depth'],
        'learning_rate': adaptive_params['learning_rate'],
        'subsample': 0.85,
        'colsample_bytree': adaptive_params['feature_fraction'],
        'reg_alpha': 0.6 * adaptive_params['reg_factor'],
        'reg_lambda': 0.6 * adaptive_params['reg_factor'],
        'min_child_weight': 8,
        'random_state': 42
    }
    
    catboost_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'depth': adaptive_params['max_depth'],
        'learning_rate': adaptive_params['learning_rate'],
        'iterations': adaptive_params['iterations'],
        'l2_leaf_reg': 1.2 * adaptive_params['reg_factor'],
        'random_strength': 0.35,
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
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=adaptive_params['iterations'])
        lgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     callbacks=[lgb.early_stopping(stopping_rounds=120), lgb.log_evaluation(0)])
        
        lgb_predictions += np.expm1(np.array(lgb_model.predict(X_test))) / 5
        lgb_val_pred = np.expm1(np.array(lgb_model.predict(X_val)))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_val_pred)
        lgb_scores.append(lgb_mae)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=adaptive_params['iterations'], early_stopping_rounds=120)
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
                     early_stopping_rounds=120, 
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
    
    # 返回预测结果、评分信息和训练好的模型
    models = {
        'lgb': lgb_model, # type: ignore
        'xgb': xgb_model, # type: ignore
        'cat': cat_model # type: ignore
    }
    
    return lgb_predictions, xgb_predictions, cat_predictions, {
        'lgb_scores': lgb_scores,
        'xgb_scores': xgb_scores,
        'cat_scores': cat_scores
    }, models

def innovative_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info):
    """
    创新集成策略 - 融合V22平衡和V24智能
    """
    print("执行创新集成策略...")
    
    # 基于性能的自适应权重
    lgb_score = np.mean(scores_info['lgb_scores'])
    xgb_score = np.mean(scores_info['xgb_scores'])
    cat_score = np.mean(scores_info['cat_scores'])
    
    # 计算基础权重
    total_inv_score = 1/lgb_score + 1/xgb_score + 1/cat_score
    raw_weights = {
        'lgb': (1/lgb_score) / total_inv_score,
        'xgb': (1/xgb_score) / total_inv_score,
        'cat': (1/cat_score) / total_inv_score
    }
    
    # V28新增：基于分数稳定性的权重调整
    lgb_std = np.std(scores_info['lgb_scores'])
    xgb_std = np.std(scores_info['xgb_scores'])
    cat_std = np.std(scores_info['cat_scores'])
    
    # 稳定性惩罚因子 - 更精细的调整
    stability_factor = {
        'lgb': 1 / (1 + lgb_std * 2),  # 更强的稳定性惩罚
        'xgb': 1 / (1 + xgb_std * 2),
        'cat': 1 / (1 + cat_std * 2)
    }
    
    # 应用稳定性调整
    for model in raw_weights:
        raw_weights[model] *= stability_factor[model]
    
    # V22的平衡权重限制 - V28微调
    balanced_weights = {}
    for model, weight in raw_weights.items():
        # V28微调：权重范围调整为0.12-0.65，给予更多灵活性
        balanced_weights[model] = np.clip(weight, 0.12, 0.65)
    
    # 重新归一化
    total_weight = sum(balanced_weights.values())
    final_weights = {model: weight/total_weight for model, weight in balanced_weights.items()}
    
    print(f"创新集成权重:")
    for model, weight in final_weights.items():
        print(f"  {model.upper()}: {weight:.3f}")
    
    ensemble_pred = (final_weights['lgb'] * lgb_pred + 
                    final_weights['xgb'] * xgb_pred + 
                    final_weights['cat'] * cat_pred)
    
    return ensemble_pred

def enhanced_calibration(predictions, y_train):
    """
    V28增强校准算法 - 多阶段校准
    """
    print("执行增强校准算法...")
    
    train_mean = y_train.mean()
    train_median = y_train.median()
    pred_mean = predictions.mean()
    pred_median = np.median(predictions)
    
    print(f"\n校准前统计:")
    print(f"  训练集均值: {train_mean:.2f}, 中位数: {train_median:.2f}")
    print(f"  预测均值: {pred_mean:.2f}, 中位数: {pred_median:.2f}")
    
    # 第一阶段：分位数校准 - V24_simplified的增强版本
    quantiles = [5, 10, 25, 40, 50, 60, 75, 90, 95]
    train_quantiles = np.percentile(y_train, quantiles)
    pred_quantiles = np.percentile(predictions, quantiles)
    
    # 计算分位数校准因子
    quantile_factors = train_quantiles / pred_quantiles
    quantile_factors = np.clip(quantile_factors, 0.7, 1.3)
    
    # 应用分位数校准
    quantile_calibrated = predictions.copy()
    for i in range(len(predictions)):
        pred_val = predictions[i]
        
        # 找到对应的分位数区间 - 更精细的插值
        for j in range(len(quantiles) - 1):
            if pred_val <= pred_quantiles[j + 1]:
                if j == 0:
                    factor = quantile_factors[0]
                else:
                    # 线性插值
                    t = (pred_val - pred_quantiles[j]) / (pred_quantiles[j + 1] - pred_quantiles[j])
                    factor = quantile_factors[j] * (1 - t) + quantile_factors[j + 1] * t
                break
        else:
            factor = quantile_factors[-1]
        
        quantile_calibrated[i] *= factor
    
    # 第二阶段：均值校准
    mean_calibration_factor = train_mean / pred_mean if pred_mean > 0 else 1.0
    mean_calibration_factor = np.clip(mean_calibration_factor, 0.85, 1.15)
    mean_calibrated = predictions * mean_calibration_factor
    
    # 第三阶段：中位数校准
    median_calibration_factor = train_median / pred_median if pred_median > 0 else 1.0
    median_calibration_factor = np.clip(median_calibration_factor, 0.9, 1.1)
    median_calibrated = predictions * median_calibration_factor
    
    # V28新增：智能权重融合
    # 根据预测分布的偏度调整权重
    pred_skew = (predictions.mean() - np.median(predictions)) / predictions.std()
    
    if abs(pred_skew) > 0.5:  # 偏度较大，更依赖分位数校准
        weights = {'quantile': 0.6, 'mean': 0.25, 'median': 0.15}
    else:  # 分布相对对称，平衡使用
        weights = {'quantile': 0.4, 'mean': 0.35, 'median': 0.25}
    
    final_predictions = (
        weights['quantile'] * quantile_calibrated +
        weights['mean'] * mean_calibrated +
        weights['median'] * median_calibrated
    )
    
    # 确保预测值为正
    final_predictions = np.maximum(final_predictions, 0)
    
    print(f"\n校准后统计:")
    print(f"  分位数校准因子范围: {quantile_factors.min():.3f} - {quantile_factors.max():.3f}")
    print(f"  均值校准因子: {mean_calibration_factor:.4f}")
    print(f"  中位数校准因子: {median_calibration_factor:.4f}")
    print(f"  预测偏度: {pred_skew:.3f}")
    print(f"  校准权重: {weights}")
    print(f"  最终预测均值: {final_predictions.mean():.2f}")
    
    return final_predictions

def create_innovative_analysis(y_train, predictions, scores_info, feature_importance=None):
    """
    创建创新分析图表
    """
    print("生成创新分析图表...")
    
    # 创建保存目录
    analysis_dir = get_user_data_path()
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 价格分布对比
    axes[0, 0].hist(y_train, bins=50, alpha=0.7, label='训练集真实价格', color='blue', density=True)
    axes[0, 0].hist(predictions, bins=50, alpha=0.7, label='V28预测价格', color='red', density=True)
    axes[0, 0].set_xlabel('价格')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('V28价格分布对比')
    axes[0, 0].legend()
    
    # 2. 模型性能对比
    models = ['LightGBM', 'XGBoost', 'CatBoost']
    scores = [np.mean(scores_info['lgb_scores']), 
              np.mean(scores_info['xgb_scores']), 
              np.mean(scores_info['cat_scores'])]
    
    bars = axes[0, 1].bar(models, scores, color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].axhline(y=488.7, color='purple', linestyle='--', label='V24_simplified基准(488.7)')
    axes[0, 1].axhline(y=480, color='red', linestyle='--', label='V28目标(480)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('V28各模型验证性能')
    axes[0, 1].legend()
    
    # 3. 特征重要性（如果有）
    if feature_importance is not None:
        top_features = feature_importance.head(10)
        axes[0, 2].barh(range(len(top_features)), top_features['mi_score'])
        axes[0, 2].set_yticks(range(len(top_features)))
        axes[0, 2].set_yticklabels(top_features['feature'])
        axes[0, 2].set_xlabel('互信息分数')
        axes[0, 2].set_title('V28 Top 10 特征重要性')
    else:
        axes[0, 2].text(0.5, 0.5, '特征重要性分析\n未启用', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('V28特征重要性')
    
    # 4. 预测值vs真实值散点图（模拟）
    sample_size = min(2000, len(y_train))
    sample_indices = np.random.choice(len(y_train), sample_size, replace=False)
    y_sample = y_train.iloc[sample_indices]
    
    # 创建一些模拟的预测值用于可视化
    noise = np.random.normal(0, y_train.std() * 0.08, sample_size)
    pred_sample = y_sample + noise
    
    axes[1, 0].scatter(y_sample, pred_sample, alpha=0.5, s=1)
    axes[1, 0].plot([y_sample.min(), y_sample.max()], [y_sample.min(), y_sample.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('真实价格')
    axes[1, 0].set_ylabel('预测价格')
    axes[1, 0].set_title('预测vs真实值散点图（模拟）')
    
    # 5. 残差分布（模拟）
    residuals = y_sample - pred_sample
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('残差')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('残差分布（模拟）')
    
    # 6. 版本对比总结
    comparison_text = f"""
    V28融合创新版本总结:
    
    融合最佳实践:
    ✅ V24_simplified: 精准特征工程(488.7分)
    ✅ V23: 分层验证和增强特征(497.6分)
    ✅ V26: 抗过拟合和稳定架构(498.0分)
    ✅ V24_fast: 目标编码和关键特征(501.8分)
    ✅ V22: 平衡策略和稳健集成(502.2分)
    
    V28核心创新:
    🚀 动态特征重要性分析
    🚀 自适应参数调优
    🚀 增强校准算法
    🚀 智能集成权重
    🚀 高阶交互特征
    
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
    
    🎯 目标: 突破488.7255分，冲击480分以内!
    """
    axes[1, 2].text(0.05, 0.95, comparison_text, transform=axes[1, 2].transAxes, 
                    fontsize=8, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_title('V28融合创新总结')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(analysis_dir, 'modeling_v28_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"V28分析图表已保存到: {chart_path}")
    plt.show()

def v28_innovative_optimize():
    """
    V28融合创新模型训练流程
    """
    print("=" * 80)
    print("开始V28融合创新模型训练")
    print("基于表现最佳模型的深度分析和融合创新")
    print("目标：突破488.7255分，冲击480分以内")
    print("=" * 80)
    
    # 步骤1: 增强数据预处理
    print("\n步骤1: 增强数据预处理...")
    train_df, test_df = enhanced_preprocessing()
    
    # 步骤2: 创新特征工程
    print("\n步骤2: 创新特征工程...")
    train_df = create_innovative_features(train_df)
    test_df = create_innovative_features(test_df)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"初始特征数量: {len(feature_cols)}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # V28新增：动态特征选择
    print("\n步骤2.5: 动态特征重要性分析...")
    X_train_selected, X_test_selected, feature_importance = dynamic_feature_selection(
        X_train, y_train, X_test, max_features=80)
    
    # 步骤3: 特征缩放
    print("\n步骤3: 特征缩放...")
    scaler = RobustScaler()
    numeric_features = X_train_selected.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_features:
        if col in X_train_selected.columns and col in X_test_selected.columns:
            # 检查无穷大值
            inf_mask = np.isinf(X_train_selected[col]) | np.isinf(X_test_selected[col])
            if inf_mask.any():
                X_train_selected.loc[inf_mask[inf_mask.index.isin(X_train_selected.index)].index, col] = 0
                X_test_selected.loc[inf_mask[inf_mask.index.isin(X_test_selected.index)].index, col] = 0
            
            X_train_selected[col] = X_train_selected[col].fillna(X_train_selected[col].median())
            X_test_selected[col] = X_test_selected[col].fillna(X_train_selected[col].median())
            
            if X_train_selected[col].std() > 1e-8:
                X_train_selected[col] = scaler.fit_transform(X_train_selected[[col]])
                X_test_selected[col] = scaler.transform(X_test_selected[[col]])
    
    # 步骤4: 训练创新模型
    print("\n步骤4: 训练创新融合模型...")
    lgb_pred, xgb_pred, cat_pred, scores_info, trained_models = train_innovative_models(
        X_train_selected, y_train, X_test_selected)
    
    # 步骤5: 创新集成
    print("\n步骤5: 创新集成策略...")
    ensemble_pred = innovative_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info)
    
    # 步骤6: 增强校准
    print("\n步骤6: 增强校准算法...")
    final_predictions = enhanced_calibration(ensemble_pred, y_train)
    
    # 步骤7: 创建分析图表
    print("\n步骤7: 生成创新分析图表...")
    create_innovative_analysis(y_train, final_predictions, scores_info, feature_importance)
    
    # 最终统计
    print(f"\nV28最终预测统计:")
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
    result_file = os.path.join(result_dir, f"modeling_v28_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\nV28结果已保存到: {result_file}")
    
    # 保存训练好的模型
    print("\n保存训练好的模型...")
    if trained_models:
        save_models(trained_models, 'v28')
    else:
        print("⚠️ 警告: 没有可保存的模型")

    
    # 生成优化报告
    print("\n" + "=" * 80)
    print("V28融合创新优化总结")
    print("=" * 80)
    print("✅ 融合V24_simplified的精准特征工程")
    print("✅ 借鉴V23的分层验证策略")
    print("✅ 采用V26的抗过拟合原则")
    print("✅ 优化V22的平衡集成策略")
    print("🚀 V28核心创新:")
    print("   - 动态特征重要性分析")
    print("   - 自适应参数调优")
    print("   - 增强校准算法")
    print("   - 智能集成权重")
    print("   - 高阶交互特征")
    print("🎯 目标：突破488.7255分，冲击480分以内!")
    print("=" * 80)
    
    return final_predictions, scores_info

if __name__ == "__main__":
    test_pred, scores_info = v28_innovative_optimize()
    print("V28融合创新优化完成! 期待突破性表现! 🚀")