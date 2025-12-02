"""
V24版本模型 - 极限优化探索版

基于V23的497.6048分优秀基础，实施以下极限优化策略:
1. 高级特征工程 - 目标编码、组合特征、非线性变换
2. 模型架构优化 - 多层Stacking、神经网络集成
3. 超参数精调 - 贝叶斯优化、网格搜索
4. 数据增强技术 - 样本权重、噪声注入
5. 集成学习进阶 - 动态权重、模型选择
6. 后处理优化 - 多阶段校准、分布匹配
目标：冲击490分以下，探索模型极限潜力
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler, QuantileTransformer, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
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

def advanced_preprocessing():
    """
    高级数据预处理 - 基于V23但更深入
    """
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 合并数据进行预处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # V24新增：异常值检测和处理
    def detect_outliers_iqr(series):
        """IQR异常值检测"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    # 高级power处理
    if 'power' in all_df.columns:
        # V23的基础处理
        all_df['power'] = np.clip(all_df['power'], 0, 600)
        all_df['power_is_zero'] = (all_df['power'] <= 0).astype(int)
        all_df['power_is_high'] = (all_df['power'] > 400).astype(int)
        
        # V24新增：power异常值标记
        power_outliers = detect_outliers_iqr(all_df['power'])
        all_df['power_is_outlier'] = power_outliers.astype(int)
        
        # V24新增：更精细的power分段
        all_df['power_segment_fine'] = pd.cut(all_df['power'], 
                                            bins=[-1, 30, 60, 90, 120, 160, 200, 250, 300, 400, 600],
                                            labels=['extreme_low', 'very_low', 'low', 'medium_low', 'medium', 
                                                   'medium_high', 'high', 'very_high', 'extreme_high', 'super_high'])
        all_df['power_segment_fine'] = all_df['power_segment_fine'].cat.codes
        
        # V24新增：power的多种变换
        all_df['log_power'] = np.log1p(np.maximum(all_df['power'], 1))
        all_df['sqrt_power'] = np.sqrt(np.maximum(all_df['power'], 0))
        all_df['power_cubed'] = all_df['power'] ** 3 / 1000000  # 归一化立方
    
    # 高级分类特征处理
    categorical_cols = ['fuelType', 'gearbox', 'bodyType', 'model', 'brand']
    for col in categorical_cols:
        if col in all_df.columns:
            # V23的基础处理
            all_df[f'{col}_missing'] = (all_df[col].isnull()).astype(int)
            
            # V23的智能填充
            if col == 'model' and 'brand' in all_df.columns:
                for brand in all_df['brand'].unique():
                    brand_mask = all_df['brand'] == brand
                    brand_mode = all_df[brand_mask][col].mode()
                    if len(brand_mode) > 0:
                        all_df.loc[brand_mask & all_df[col].isnull(), col] = brand_mode.iloc[0]
            
            mode_value = all_df[col].mode()
            if len(mode_value) > 0:
                all_df[col] = all_df[col].fillna(mode_value.iloc[0])
            
            # V23的频率编码
            freq_map = all_df[col].value_counts().to_dict()
            all_df[f'{col}_freq'] = all_df[col].map(freq_map)
            
            # V24新增：目标编码（仅对训练集有效）
            if 'price' in all_df.columns and col != 'brand':  # brand太多类别，容易过拟合
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
    all_df['reg_day'] = all_df['regDate'].dt.day.fillna(15).astype(int)
    
    # V23的季节特征
    all_df['reg_season'] = all_df['reg_month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
    all_df['is_winter_reg'] = all_df['reg_month'].isin([12, 1, 2]).astype(int)
    all_df['is_summer_reg'] = all_df['reg_month'].isin([6, 7, 8]).astype(int)
    
    # V24新增：更复杂的时间特征
    all_df['is_weekend_reg'] = (all_df['reg_dayofweek'] >= 5).astype(int)
    all_df['is_month_start'] = (all_df['reg_day'] <= 5).astype(int)
    all_df['is_month_end'] = (all_df['reg_day'] >= 25).astype(int)
    all_df['reg_month_sin'] = np.sin(2 * np.pi * all_df['reg_month'] / 12)
    all_df['reg_month_cos'] = np.cos(2 * np.pi * all_df['reg_month'] / 12)
    
    all_df.drop(columns=['regDate'], inplace=True)
    
    # 高级品牌统计特征
    if 'price' in all_df.columns:
        # V23的基础统计
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count', 'std', 'median', 'min', 'max']).reset_index()
        brand_stats['smooth_mean'] = ((brand_stats['mean'] * brand_stats['count'] + 
                                     all_df['price'].mean() * 40) / (brand_stats['count'] + 40))
        
        # V24新增：更多品牌统计特征
        brand_stats['price_range'] = brand_stats['max'] - brand_stats['min']
        brand_stats['price_iqr'] = brand_stats['75%'] - brand_stats['25%'] if '75%' in brand_stats.columns else 0
        brand_stats['cv'] = brand_stats['std'] / brand_stats['mean']
        brand_stats['cv'] = brand_stats['cv'].fillna(brand_stats['cv'].median())
        brand_stats['skewness'] = (brand_stats['mean'] - brand_stats['median']) / brand_stats['std']
        brand_stats['skewness'] = brand_stats['skewness'].fillna(0)
        
        # 映射特征
        brand_maps = {
            'avg_price': brand_stats.set_index('brand')['smooth_mean'],
            'price_stability': brand_stats.set_index('brand')['cv'],
            'price_range': brand_stats.set_index('brand')['price_range'],
            'brand_skewness': brand_stats.set_index('brand')['skewness']
        }
        
        for feature_name, brand_map in brand_maps.items():
            all_df[f'brand_{feature_name}'] = all_df['brand'].map(brand_map).fillna(brand_map.median())
    
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

def create_advanced_features(df):
    """
    高级特征工程 - 基于V23但更深入
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
        df['power_km_product'] = df['power'] * df['kilometer'] / 100000  # 归一化
    
    if 'car_age' in df.columns and 'kilometer' in df.columns:
        df['age_km_interaction'] = df['car_age'] * df['kilometer'] / 1000
        df['depreciation_rate'] = df['kilometer'] / (df['car_age'] + 1) / 10000
    
    # V24新增：多项式特征
    if 'power' in df.columns:
        df['power_squared'] = df['power'] ** 2 / 1000  # 归一化
        df['power_cubed'] = df['power'] ** 3 / 1000000  # 归一化
    
    if 'car_age' in df.columns:
        df['car_age_squared'] = df['car_age'] ** 2
        df['car_age_cubed'] = df['car_age'] ** 3
    
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
        df['v_q25'] = df[v_cols].quantile(0.25, axis=1)
        df['v_q75'] = df[v_cols].quantile(0.75, axis=1)
        df['v_iqr'] = df['v_q75'] - df['v_q25']
        
        # V24新增：v特征的组合
        df['v_mean_to_std_ratio'] = df['v_mean'] / (df['v_std'] + 1e-6)
        df['v_range_to_mean_ratio'] = df['v_range'] / (df['v_mean'] + 1e-6)
    
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
            df[f'{feat1}_add_{feat2}'] = df[feat1] + df[feat2]
            df[f'{feat1}_sub_{feat2}'] = df[feat1] - df[feat2]
    
    # V24新增：聚类特征（简化版）
    if len(v_cols) >= 5:
        from sklearn.cluster import KMeans
        v_data = df[v_cols].fillna(0)
        
        # 使用KMeans创建聚类特征
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df['v_cluster'] = kmeans.fit_predict(v_data)
        
        # 计算到聚类中心的距离
        centers = kmeans.cluster_centers_
        distances = []
        for i in range(len(df)):
            cluster_id = int(df.iloc[i]['v_cluster'])
            center = centers[cluster_id]
            point = v_data.iloc[i].values
            distance = np.linalg.norm(point - center)
            distances.append(distance)
        
        df['v_cluster_distance'] = distances
    
    # 数据清理 - 更精细的处理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 处理无穷大值
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # 填充NaN值
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
        
        # V24新增：更智能的异常值处理
        if col not in ['SaleID', 'price', 'v_cluster'] and df[col].std() > 1e-8:
            # 根据特征类型选择不同的截断策略
            ratio_features = [col for col in df.columns if 'ratio' in col or 'rate' in col]
            power_features = [col for col in df.columns if 'power' in col]
            age_features = [col for col in df.columns if 'age' in col]
            km_features = [col for col in df.columns if 'km' in col or 'kilometer' in col]
            
            if col in ratio_features:
                # 比例特征使用更保守的截断
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = np.clip(df[col], q01, q99)
            elif col in power_features:
                # 功率特征使用物理合理范围
                df[col] = np.clip(df[col], df[col].quantile(0.005), df[col].quantile(0.995))
            else:
                # 其他特征使用标准截断
                q999 = df[col].quantile(0.999)
                q001 = df[col].quantile(0.001)
                df[col] = np.clip(df[col], q001, q999)
    
    return df

def train_advanced_ensemble(X_train, y_train, X_test):
    """
    训练高级集成模型 - 多层Stacking
    """
    print("训练高级集成模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 分层交叉验证
    y_bins = pd.qcut(y_train, q=10, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 第一层模型 - 基础模型
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'num_leaves': 35,
        'max_depth': 8,
        'learning_rate': 0.07,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'lambda_l1': 0.2,
        'lambda_l2': 0.2,
        'min_child_samples': 15,
        'random_state': 42,
    }
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': 8,
        'learning_rate': 0.07,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.6,
        'reg_lambda': 0.6,
        'min_child_weight': 8,
        'random_state': 42
    }
    
    catboost_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'depth': 8,
        'learning_rate': 0.07,
        'iterations': 1000,
        'l2_leaf_reg': 1.2,
        'random_strength': 0.3,
        'random_seed': 42,
        'verbose': False
    }
    
    rf_params = {
        'n_estimators': 500,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    et_params = {
        'n_estimators': 500,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 存储第一层预测
    lgb_predictions = np.zeros(len(X_test))
    xgb_predictions = np.zeros(len(X_test))
    cat_predictions = np.zeros(len(X_test))
    rf_predictions = np.zeros(len(X_test))
    et_predictions = np.zeros(len(X_test))
    
    # 存储训练集的交叉验证预测（用于第二层训练）
    lgb_cv_pred = np.zeros(len(X_train))
    xgb_cv_pred = np.zeros(len(X_train))
    cat_cv_pred = np.zeros(len(X_train))
    rf_cv_pred = np.zeros(len(X_train))
    et_cv_pred = np.zeros(len(X_train))
    
    # 存储验证分数
    lgb_scores, xgb_scores, cat_scores, rf_scores, et_scores = [], [], [], [], []
    
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
        lgb_cv_pred[val_idx] = np.expm1(np.array(lgb_model.predict(X_val)))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_cv_pred[val_idx])
        lgb_scores.append(lgb_mae)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=1800, early_stopping_rounds=100)
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
        
        # RandomForest
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_tr, np.expm1(y_tr_log))
        rf_predictions += rf_model.predict(X_test) / 5
        rf_cv_pred[val_idx] = rf_model.predict(X_val)
        rf_mae = mean_absolute_error(np.expm1(y_val_log), rf_cv_pred[val_idx])
        rf_scores.append(rf_mae)
        
        # ExtraTrees
        et_model = ExtraTreesRegressor(**et_params)
        et_model.fit(X_tr, np.expm1(y_tr_log))
        et_predictions += et_model.predict(X_test) / 5
        et_cv_pred[val_idx] = et_model.predict(X_val)
        et_mae = mean_absolute_error(np.expm1(y_val_log), et_cv_pred[val_idx])
        et_scores.append(et_mae)
        
        print(f"  LGB: {lgb_mae:.2f}, XGB: {xgb_mae:.2f}, CAT: {cat_mae:.2f}, RF: {rf_mae:.2f}, ET: {et_mae:.2f}")
    
    print(f"\n第一层平均验证分数:")
    print(f"  LightGBM: {np.mean(lgb_scores):.2f} (±{np.std(lgb_scores):.2f})")
    print(f"  XGBoost: {np.mean(xgb_scores):.2f} (±{np.std(xgb_scores):.2f})")
    print(f"  CatBoost: {np.mean(cat_scores):.2f} (±{np.std(cat_scores):.2f})")
    print(f"  RandomForest: {np.mean(rf_scores):.2f} (±{np.std(rf_scores):.2f})")
    print(f"  ExtraTrees: {np.mean(et_scores):.2f} (±{np.std(et_scores):.2f})")
    
    # 第二层：Stacking
    print("\n训练第二层Stacking模型...")
    
    # 构建第二层特征
    stack_train = pd.DataFrame({
        'lgb': lgb_cv_pred,
        'xgb': xgb_cv_pred,
        'cat': cat_cv_pred,
        'rf': rf_cv_pred,
        'et': et_cv_pred
    })
    
    stack_test = pd.DataFrame({
        'lgb': lgb_predictions,
        'xgb': xgb_predictions,
        'cat': cat_predictions,
        'rf': rf_predictions,
        'et': et_predictions
    })
    
    # 第二层模型
    ridge_params = {'alpha': 0.1, 'random_state': 42}
    enet_params = {'alpha': 0.1, 'l1_ratio': 0.5, 'random_state': 42}
    
    # 训练第二层模型
    ridge_model = Ridge(**ridge_params)
    ridge_model.fit(stack_train, y_train)
    ridge_pred = ridge_model.predict(stack_test)
    
    enet_model = ElasticNet(**enet_params)
    enet_model.fit(stack_train, y_train)
    enet_pred = enet_model.predict(stack_test)
    
    # 第三层：最终集成
    print("\n第三层：最终集成...")
    
    # 计算各模型的权重
    model_scores = {
        'lgb': np.mean(lgb_scores),
        'xgb': np.mean(xgb_scores),
        'cat': np.mean(cat_scores),
        'rf': np.mean(rf_scores),
        'et': np.mean(et_scores)
    }
    
    # 基于性能的权重
    inv_scores = {model: 1/score for model, score in model_scores.items()}
    total_inv = sum(inv_scores.values())
    base_weights = {model: inv_score/total_inv for model, inv_score in inv_scores.items()}
    
    # 稳定性调整
    model_stds = {
        'lgb': np.std(lgb_scores),
        'xgb': np.std(xgb_scores),
        'cat': np.std(cat_scores),
        'rf': np.std(rf_scores),
        'et': np.std(et_scores)
    }
    
    stability_factors = {model: 1/(1+std) for model, std in model_stds.items()}
    
    # 应用稳定性调整
    for model in base_weights:
        base_weights[model] *= stability_factors[model]
    
    # 重新归一化
    total_weight = sum(base_weights.values())
    first_layer_weights = {model: weight/total_weight for model, weight in base_weights.items()}
    
    # 第一层集成
    first_layer_ensemble = (
        first_layer_weights['lgb'] * lgb_predictions +
        first_layer_weights['xgb'] * xgb_predictions +
        first_layer_weights['cat'] * cat_predictions +
        first_layer_weights['rf'] * rf_predictions +
        first_layer_weights['et'] * et_predictions
    )
    
    # 最终集成：第一层 + 第二层
    # 使用简单的加权平均，第二层模型权重稍高
    final_ensemble = 0.4 * first_layer_ensemble + 0.35 * ridge_pred + 0.25 * enet_pred
    
    print(f"第一层权重:")
    for model, weight in first_layer_weights.items():
        print(f"  {model.upper()}: {weight:.3f}")
    
    print(f"最终集成权重: 第一层(40%), Ridge(35%), ElasticNet(25%)")
    
    return final_ensemble, {
        'lgb_scores': lgb_scores,
        'xgb_scores': xgb_scores,
        'cat_scores': cat_scores,
        'rf_scores': rf_scores,
        'et_scores': et_scores,
        'first_layer_weights': first_layer_weights
    }

def advanced_calibration(predictions, y_train):
    """
    高级校准 - 多阶段校准
    """
    print("执行高级多阶段校准...")
    
    train_mean = y_train.mean()
    train_median = y_train.median()
    pred_mean = predictions.mean()
    pred_median = np.median(predictions)
    
    print(f"\n校准前统计:")
    print(f"  训练集均值: {train_mean:.2f}, 中位数: {train_median:.2f}")
    print(f"  预测均值: {pred_mean:.2f}, 中位数: {pred_median:.2f}")
    
    # 第一阶段：分位数校准
    quantiles = [5, 10, 25, 50, 75, 90, 95]
    train_quantiles = np.percentile(y_train, quantiles)
    pred_quantiles = np.percentile(predictions, quantiles)
    
    # 计算分位数校准因子
    quantile_factors = train_quantiles / pred_quantiles
    quantile_factors = np.clip(quantile_factors, 0.7, 1.3)
    
    # 应用分位数校准
    calibrated_pred = predictions.copy()
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
        
        calibrated_pred[i] *= factor
    
    # 第二阶段：分布匹配
    from scipy.stats import norm
    
    # 使用Box-Cox变换进行分布匹配
    def boxcox_optimize(y):
        """优化Box-Cox变换参数"""
        from scipy.stats import boxcox
        y_positive = y[y > 0]
        if len(y_positive) == 0:
            return y, 0
        
        try:
            transformed, lambda_param = boxcox(y_positive)
            return transformed, lambda_param
        except:
            return np.log1p(y_positive), 0
    
    # 对训练集和预测集进行Box-Cox变换
    train_transformed, train_lambda = boxcox_optimize(y_train)
    pred_transformed, pred_lambda = boxcox_optimize(calibrated_pred)
    
    # 匹配分布参数
    train_mean, train_std = np.mean(train_transformed), np.std(train_transformed)
    pred_mean, pred_std = np.mean(pred_transformed), np.std(pred_transformed)
    
    # 调整预测集分布
    if pred_std > 1e-8:
        pred_adjusted = (pred_transformed - pred_mean) * train_std / pred_std + train_mean
    else:
        pred_adjusted = pred_transformed
    
    # 逆变换
    def inv_boxcox(transformed, lambda_param):
        """Box-Cox逆变换"""
        if lambda_param == 0:
            return np.exp(transformed) - 1
        else:
            return np.exp(transformed) * lambda_param ** (1/lambda_param) - 1
    
    try:
        distribution_matched = inv_boxcox(pred_adjusted, pred_lambda)
        # 确保长度匹配
        if len(distribution_matched) != len(calibrated_pred):
            distribution_matched = calibrated_pred
    except:
        distribution_matched = calibrated_pred
    
    # 第三阶段：局部校准
    # 基于价格区间的局部校准
    price_bins = [0, 5000, 10000, 20000, 40000, 60000, 100000, float('inf')]
    local_factors = {}
    
    for i in range(len(price_bins) - 1):
        lower, upper = price_bins[i], price_bins[i + 1]
        
        # 训练集中的样本
        train_mask = (y_train >= lower) & (y_train < upper)
        if train_mask.sum() == 0:
            continue
        
        train_local_mean = y_train[train_mask].mean()
        
        # 预测集中的样本
        pred_mask = (calibrated_pred >= lower) & (calibrated_pred < upper)
        if pred_mask.sum() == 0:
            continue
        
        pred_local_mean = calibrated_pred[pred_mask].mean()
        
        # 计算局部校准因子
        if pred_local_mean > 0:
            local_factor = train_local_mean / pred_local_mean
            local_factor = np.clip(local_factor, 0.8, 1.2)
            local_factors[i] = local_factor
    
    # 应用局部校准
    final_pred = distribution_matched.copy()
    for i in range(len(final_pred)):
        pred_val = final_pred[i]
        
        # 找到对应的价格区间
        for j in range(len(price_bins) - 1):
            if price_bins[j] <= pred_val < price_bins[j + 1]:
                if j in local_factors:
                    final_pred[i] *= local_factors[j]
                break
    
    # 确保预测值为正
    final_pred = np.maximum(final_pred, 0)
    
    print(f"\n校准后统计:")
    print(f"  预测均值: {final_pred.mean():.2f}, 中位数: {np.median(final_pred):.2f}")
    print(f"  分位数校准因子: {quantile_factors}")
    print(f"  局部校准因子数量: {len(local_factors)}")
    
    return final_pred

def create_advanced_analysis(y_train, predictions, scores_info):
    """
    创建高级分析图表
    """
    print("生成高级分析图表...")
    
    # 创建保存目录
    analysis_dir = get_user_data_path()
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. 价格分布对比
    axes[0, 0].hist(y_train, bins=50, alpha=0.7, label='训练集真实价格', color='blue', density=True)
    axes[0, 0].hist(predictions, bins=50, alpha=0.7, label='V24预测价格', color='red', density=True)
    axes[0, 0].set_xlabel('价格')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('V24价格分布对比')
    axes[0, 0].legend()
    
    # 2. 模型性能对比
    models = ['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest', 'ExtraTrees']
    scores = [np.mean(scores_info['lgb_scores']), 
              np.mean(scores_info['xgb_scores']), 
              np.mean(scores_info['cat_scores']),
              np.mean(scores_info['rf_scores']),
              np.mean(scores_info['et_scores'])]
    
    bars = axes[0, 1].bar(models, scores, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
    axes[0, 1].axhline(y=497.6, color='orange', linestyle='--', label='V23基准(497.6)')
    axes[0, 1].axhline(y=490, color='red', linestyle='--', label='目标线(490)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('V24各模型验证性能')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 权重分析
    weights = scores_info['first_layer_weights']
    model_names = list(weights.keys())
    weight_values = list(weights.values())
    
    axes[0, 2].pie(weight_values, labels=[name.upper() for name in model_names], autopct='%1.3f')
    axes[0, 2].set_title('V24第一层模型权重分布')
    
    # 4. 预测值vs真实值散点图（模拟）
    sample_size = min(1000, len(y_train))
    sample_indices = np.random.choice(len(y_train), sample_size, replace=False)
    y_sample = y_train.iloc[sample_indices]
    
    # 创建一些模拟的预测值用于可视化
    noise = np.random.normal(0, y_train.std() * 0.1, sample_size)
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
    
    # 6. 分位数-分位数图
    stats.probplot(residuals, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q图（模拟）')
    
    # 7. 价格区间分析
    price_bins = [0, 5000, 10000, 20000, 40000, 60000, 100000, float('inf')]
    bin_labels = ['<5K', '5K-10K', '10K-20K', '20K-40K', '40K-60K', '60K-100K', '>100K']
    
    train_bin_counts = []
    pred_bin_counts = []
    
    for i in range(len(price_bins) - 1):
        lower, upper = price_bins[i], price_bins[i + 1]
        train_count = ((y_train >= lower) & (y_train < upper)).sum()
        pred_count = ((predictions >= lower) & (predictions < upper)).sum()
        train_bin_counts.append(train_count)
        pred_bin_counts.append(pred_count)
    
    x = np.arange(len(bin_labels))
    width = 0.35
    
    axes[2, 0].bar(x - width/2, train_bin_counts, width, label='训练集', alpha=0.7)
    axes[2, 0].bar(x + width/2, pred_bin_counts, width, label='预测集', alpha=0.7)
    axes[2, 0].set_xlabel('价格区间')
    axes[2, 0].set_ylabel('样本数量')
    axes[2, 0].set_title('价格区间分布对比')
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(bin_labels, rotation=45)
    axes[2, 0].legend()
    
    # 8. 特征重要性（模拟）
    top_features = ['power_age_ratio', 'km_per_year', 'brand_avg_price', 'car_age', 'power']
    importance = [0.25, 0.20, 0.18, 0.15, 0.12]
    
    axes[2, 1].barh(top_features, importance, color='lightblue')
    axes[2, 1].set_xlabel('重要性')
    axes[2, 1].set_title('Top 5 特征重要性（模拟）')
    
    # 9. 版本对比总结
    comparison_text = f"""
    V24极限优化版本总结:
    
    基于V23的497.6048分基础:
    ✅ V23: 精准突破500分
    ✅ V22: 平衡策略和502分基础
    ✅ V16: 稳定基线和自适应集成
    
    V24新增极限优化:
    🚀 高级特征工程: 目标编码、聚类特征
    🚀 多层Stacking: 5个基础模型+2个元模型
    🚀 高级校准: 分位数+分布+局部三阶段
    🚀 智能权重: 基于性能和稳定性
    🚀 异常处理: 更精细的数据清理
    
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
    RandomForest: {np.mean(scores_info["rf_scores"]):.2f}
    ExtraTrees: {np.mean(scores_info["et_scores"]):.2f}
    
    🎯 目标: 冲击490分以下!
    """
    axes[2, 2].text(0.05, 0.95, comparison_text, transform=axes[2, 2].transAxes, 
                    fontsize=8, verticalalignment='top', fontfamily='monospace')
    axes[2, 2].set_title('V24极限优化总结')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(analysis_dir, 'modeling_v24_advanced_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"V24分析图表已保存到: {chart_path}")
    plt.show()

def v24_advanced_optimize():
    """
    V24极限优化模型训练流程
    """
    print("=" * 80)
    print("开始V24极限优化模型训练")
    print("基于V23的497.6048分基础，极限优化探索模型潜力")
    print("目标：冲击490分以下，探索模型极限")
    print("=" * 80)
    
    # 步骤1: 高级数据预处理
    print("\n步骤1: 高级数据预处理...")
    train_df, test_df = advanced_preprocessing()
    
    # 步骤2: 高级特征工程
    print("\n步骤2: 高级特征工程...")
    train_df = create_advanced_features(train_df)
    test_df = create_advanced_features(test_df)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 步骤3: 高级特征缩放
    print("\n步骤3: 高级特征缩放...")
    
    # 对不同类型的特征使用不同的缩放方法
    ratio_features = [c for c in X_train.columns if 'ratio' in c or 'rate' in c]
    power_features = [c for c in X_train.columns if 'power' in c]
    age_features = [c for c in X_train.columns if 'age' in c]
    km_features = [c for c in X_train.columns if 'km' in c or 'kilometer' in c]
    v_features = [c for c in X_train.columns if c.startswith('v_')]
    other_features = [c for c in X_train.columns if c not in ratio_features + power_features + age_features + km_features + v_features]
    
    # 对不同特征使用不同的缩放器
    scalers = {}
    
    for feature_list, scaler_type in [
        (ratio_features + power_features + age_features + km_features, RobustScaler()),
        (v_features + other_features, StandardScaler())
    ]:
        if feature_list:
            for col in feature_list:
                if col in X_train.columns and col in X_test.columns:
                    # 检查无穷大值
                    inf_mask = np.isinf(X_train[col]) | np.isinf(X_test[col])
                    if inf_mask.any():
                        X_train.loc[inf_mask[inf_mask.index.isin(X_train.index)].index, col] = 0
                        X_test.loc[inf_mask[inf_mask.index.isin(X_test.index)].index, col] = 0
                    
                    X_train[col] = X_train[col].fillna(X_train[col].median())
                    X_test[col] = X_test[col].fillna(X_train[col].median())
                    
                    if X_train[col].std() > 1e-8:
                        if scaler_type == 'robust':
                            scaler = RobustScaler()
                        else:
                            scaler = StandardScaler()
                        
                        X_train[col] = scaler.fit_transform(X_train[[col]])
                        X_test[col] = scaler.transform(X_test[[col]])
                        scalers[col] = scaler
    
    # 步骤4: 训练高级集成模型
    print("\n步骤4: 训练高级集成模型...")
    ensemble_pred, scores_info = train_advanced_ensemble(X_train, y_train, X_test)
    
    # 步骤5: 高级校准
    print("\n步骤5: 高级校准...")
    final_predictions = advanced_calibration(ensemble_pred, y_train)
    
    # 步骤6: 创建分析图表
    print("\n步骤6: 生成分析图表...")
    create_advanced_analysis(y_train, final_predictions, scores_info)
    
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
    result_file = os.path.join(result_dir, f"modeling_v24_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\nV24结果已保存到: {result_file}")
    
    # 生成优化报告
    print("\n" + "=" * 80)
    print("V24极限优化总结")
    print("=" * 80)
    print("✅ 基于V23的497.6048分优秀基础")
    print("✅ 高级特征工程 - 目标编码、聚类特征、多项式特征")
    print("✅ 多层Stacking - 5个基础模型+2个元模型")
    print("✅ 高级校准 - 分位数+分布+局部三阶段校准")
    print("✅ 智能权重 - 基于性能和稳定性的动态调整")
    print("✅ 精细异常处理 - 更智能的数据清理策略")
    print("🚀 目标：冲击490分以下，探索模型极限!")
    print("=" * 80)
    
    return final_predictions, scores_info

if __name__ == "__main__":
    test_pred, scores_info = v24_advanced_optimize()
    print("V24极限优化完成! 期待冲击490分! 🚀")