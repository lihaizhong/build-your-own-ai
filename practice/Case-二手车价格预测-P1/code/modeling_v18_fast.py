"""
V18 Fast版本模型 - 快速验证革命性优化

基于V18的革命性策略，实施以下快速验证优化:
1. 核心革命性特征工程 - 目标编码、关键多项式特征
2. 智能超参数优化 - 基于V17最佳参数的快速搜索
3. 简化多层Stacking - 3个核心模型的快速集成
4. 高级校准融合 - 分布校准和自适应权重
目标：快速验证V18策略的有效性
"""

import os
from typing import Tuple, Dict, Any, List, Union, Optional
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.model_selection import KFold, train_test_split, cross_val_score  # type: ignore
from sklearn.ensemble import RandomForestRegressor, IsolationForest  # type: ignore
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet  # type: ignore
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.feature_selection import SelectFromModel  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
import lightgbm as lgb  # type: ignore
import xgboost as xgb  # type: ignore
from catboost import CatBoostRegressor  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from scipy import stats  # type: ignore
import warnings
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
import time
import joblib  # type: ignore
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def get_project_path(*paths: str) -> str:
    """获取项目路径的统一方法"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        return os.path.join(project_dir, *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)

def get_user_data_path(*paths: str) -> str:
    """获取用户数据路径"""
    return get_project_path('user_data', *paths)

class TargetEncoder(BaseEstimator, TransformerMixin):
    """简化版目标编码器"""
    def __init__(self, smoothing: float = 5.0):
        self.smoothing = smoothing
        self.encodings = {}
        self.global_mean = 0
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean = y.mean()
        for col in X.columns:
            if X[col].nunique() < 50:  # 只对低基数特征编码
                temp = pd.concat([X[col], y], axis=1)
                averages = temp.groupby(col)[y.name].agg(['mean', 'count'])
                smooth = (averages['count'] * averages['mean'] + 
                         self.smoothing * self.global_mean) / (averages['count'] + self.smoothing)
                self.encodings[col] = smooth
        return self
    
    def transform(self, X: pd.DataFrame):
        X_encoded = X.copy()
        for col in X.columns:
            if col in self.encodings:
                X_encoded[f'{col}_target_enc'] = X[col].map(self.encodings[col]).fillna(self.global_mean)
        return X_encoded

def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    快速智能数据预处理
    """
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 合并数据进行统一预处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 1. 快速异常值检测
    print("执行快速异常值检测...")
    numeric_cols = ['power', 'kilometer']
    numeric_cols = [col for col in numeric_cols if col in all_df.columns]
    
    for col in numeric_cols:
        if col in all_df.columns:
            # 使用分位数截断
            Q1, Q3 = all_df[col].quantile([0.01, 0.99])
            all_df[col] = np.clip(all_df[col], Q1, Q3)
    
    # 2. 缺失值处理
    print("处理缺失值...")
    categorical_cols = ['fuelType', 'gearbox', 'bodyType', 'model']
    
    for col in categorical_cols:
        if col in all_df.columns:
            # 缺失值标记
            all_df[f'{col}_missing'] = all_df[col].isnull().astype(int)
            
            # 智能填充
            if col == 'model' and 'brand' in all_df.columns:
                for brand in all_df['brand'].unique():
                    brand_mask = all_df['brand'] == brand
                    brand_mode = all_df[brand_mask][col].mode()
                    if len(brand_mode) > 0:
                        all_df.loc[brand_mask & all_df[col].isnull(), col] = brand_mode.iloc[0]
            
            # 最终众数填充
            mode_value = all_df[col].mode()
            if len(mode_value) > 0:
                all_df[col] = all_df[col].fillna(mode_value.iloc[0])
    
    # 3. 时间特征工程
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year
    all_df['car_age'] = all_df['car_age'].fillna(all_df['car_age'].median()).astype(int)
    
    all_df['reg_month'] = all_df['regDate'].dt.month.fillna(6).astype(int)
    all_df['reg_quarter'] = all_df['regDate'].dt.quarter.fillna(2).astype(int)
    
    # 季节性特征
    all_df['reg_season'] = all_df['reg_month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
    
    all_df.drop(columns=['regDate'], inplace=True)
    
    # 4. 品牌统计特征
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'std', 'count']).reset_index()
        
        brand_stats['smooth_factor'] = np.where(brand_stats['count'] < 10, 100, 
                                              np.where(brand_stats['count'] < 50, 50, 30))
        brand_stats['smooth_mean'] = ((brand_stats['mean'] * brand_stats['count'] + 
                                     all_df['price'].mean() * brand_stats['smooth_factor']) / 
                                    (brand_stats['count'] + brand_stats['smooth_factor']))
        
        brand_maps = {
            'brand_avg_price': brand_stats.set_index('brand')['smooth_mean'],
            'brand_price_std': brand_stats.set_index('brand')['std'].fillna(0),
            'brand_count': brand_stats.set_index('brand')['count']
        }
        
        for feature_name, brand_map in brand_maps.items():
            all_df[feature_name] = all_df['brand'].map(brand_map).fillna(all_df['price'].mean())
    
    # 5. 标签编码
    categorical_cols = ['model', 'brand', 'fuelType', 'gearbox', 'bodyType']
    
    for col in categorical_cols:
        if col in all_df.columns:
            le = LabelEncoder()
            all_df[col] = le.fit_transform(all_df[col].astype(str))
    
    # 6. 数值特征处理
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

def create_core_revolutionary_features(df: pd.DataFrame, is_train: bool = True, target_encoder: Optional[TargetEncoder] = None) -> Tuple[pd.DataFrame, Optional[TargetEncoder]]:
    """
    核心革命性特征工程 - 快速版
    """
    df = df.copy()
    
    # 1. 目标编码特征
    if is_train and 'price' in df.columns:
        categorical_for_te = ['brand', 'fuelType', 'gearbox']
        cat_cols_te = [col for col in categorical_for_te if col in df.columns]
        
        if cat_cols_te:
            target_encoder = TargetEncoder(smoothing=5.0)
            target_encoder.fit(df[cat_cols_te], df['price'])
            df = target_encoder.transform(df)
    elif target_encoder is not None:
        categorical_for_te = ['brand', 'fuelType', 'gearbox']
        cat_cols_te = [col for col in categorical_for_te if col in df.columns]
        if cat_cols_te:
            df = target_encoder.transform(df)
    
    # 2. 核心多项式特征
    core_features = ['car_age', 'power', 'kilometer']
    core_features = [col for col in core_features if col in df.columns]
    
    if len(core_features) >= 2:
        for i, col1 in enumerate(core_features):
            for col2 in core_features[i+1:]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-6)
    
    # 3. 聚类特征（简化版）
    if len(core_features) >= 2:
        scaler_for_clustering = StandardScaler()
        clustering_data = scaler_for_clustering.fit_transform(df[core_features].fillna(df[core_features].median()))
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=5)
        cluster_labels = kmeans.fit_predict(clustering_data)
        df['cluster'] = cluster_labels
    
    # 4. 业务逻辑特征
    if 'power' in df.columns and 'car_age' in df.columns:
        df['power_decay'] = df['power'] / (df['car_age'] + 1)
        df['power_per_year'] = df['power'] / np.maximum(df['car_age'], 1)
    
    if 'kilometer' in df.columns and 'car_age' in df.columns:
        car_age_safe = np.maximum(df['car_age'], 0.1)
        df['km_per_year'] = df['kilometer'] / car_age_safe
        
        # 使用强度分类
        df['usage_intensity'] = pd.cut(df['km_per_year'], 
                                     bins=[0, 10000, 20000, 30000, float('inf')],
                                     labels=['low', 'medium', 'high', 'very_high'])
        df['usage_intensity'] = df['usage_intensity'].cat.codes
    
    # 5. v特征统计
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
    
    # 6. 时间特征增强
    if 'reg_month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['reg_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['reg_month'] / 12)
    
    # 7. 异常值标记
    if 'power' in df.columns:
        df['power_zero'] = (df['power'] <= 0).astype(int)
        df['power_very_high'] = (df['power'] > df['power'].quantile(0.95)).astype(int)
    
    if 'kilometer' in df.columns:
        df['km_very_high'] = (df['kilometer'] > df['kilometer'].quantile(0.95)).astype(int)
    
    # 8. 数据清理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if col not in ['price', 'SaleID']:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
            
            if df[col].std() > 1e-8:
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = np.clip(df[col], mean_val - 4*std_val, mean_val + 4*std_val)
    
    return df, target_encoder

def fast_hyperparameter_optimization(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    快速超参数优化 - 基于V17最佳参数
    """
    print("开始快速超参数优化...")
    
    y_train_log = np.log1p(y_train)
    
    # 基于V17最佳参数的搜索空间
    lgb_param_grid = [
        {'num_leaves': 31, 'max_depth': 8, 'learning_rate': 0.1, 'feature_fraction': 0.8},
        {'num_leaves': 50, 'max_depth': 10, 'learning_rate': 0.05, 'feature_fraction': 0.9},
        {'num_leaves': 40, 'max_depth': 9, 'learning_rate': 0.08, 'feature_fraction': 0.85}
    ]
    
    xgb_param_grid = [
        {'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'max_depth': 10, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'max_depth': 9, 'learning_rate': 0.08, 'subsample': 0.85, 'colsample_bytree': 0.85}
    ]
    
    cat_param_grid = [
        {'depth': 8, 'learning_rate': 0.1, 'l2_leaf_reg': 1.0},
        {'depth': 10, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0},
        {'depth': 9, 'learning_rate': 0.08, 'l2_leaf_reg': 2.0}
    ]
    
    best_params = {}
    
    # LightGBM优化
    best_lgb_score = float('inf')
    for params in lgb_param_grid:
        lgb_model = lgb.LGBMRegressor(objective='mae', metric='mae', random_state=42, 
                                     n_estimators=100, **params)
        scores = cross_val_score(lgb_model, X_train, y_train_log, cv=3, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)
        avg_score = -scores.mean()
        
        if avg_score < best_lgb_score:
            best_lgb_score = avg_score
            best_params['lgb'] = params
    
    # XGBoost优化
    best_xgb_score = float('inf')
    for params in xgb_param_grid:
        xgb_model = xgb.XGBRegressor(objective='reg:absoluteerror', eval_metric='mae', 
                                    random_state=42, n_estimators=100, **params)
        scores = cross_val_score(xgb_model, X_train, y_train_log, cv=3, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)
        avg_score = -scores.mean()
        
        if avg_score < best_xgb_score:
            best_xgb_score = avg_score
            best_params['xgb'] = params
    
    # CatBoost优化
    best_cat_score = float('inf')
    for params in cat_param_grid:
        cat_model = CatBoostRegressor(loss_function='MAE', eval_metric='MAE', 
                                     random_seed=42, iterations=100, verbose=False, **params)
        scores = cross_val_score(cat_model, X_train, y_train_log, cv=3, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)
        avg_score = -scores.mean()
        
        if avg_score < best_cat_score:
            best_cat_score = avg_score
            best_params['cat'] = params
    
    print(f"快速优化完成 - LGB: {best_lgb_score:.4f}, XGB: {best_xgb_score:.4f}, CAT: {best_cat_score:.4f}")
    
    return best_params

def fast_multi_stacking(X_train: pd.DataFrame, y_train: Union[pd.Series, pd.DataFrame], 
                       X_test: pd.DataFrame, best_params: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    快速多层Stacking集成
    """
    print("执行快速多层Stacking集成...")
    
    y_train_log = np.log1p(y_train)
    
    # 基础模型参数
    lgb_params = best_params['lgb']
    lgb_params.update({'objective': 'mae', 'metric': 'mae', 'random_state': 42})
    
    xgb_params = best_params['xgb']
    xgb_params.update({'objective': 'reg:absoluteerror', 'eval_metric': 'mae', 'random_state': 42})
    
    cat_params = best_params['cat']
    cat_params.update({'loss_function': 'MAE', 'eval_metric': 'MAE', 'random_seed': 42, 'verbose': False})
    
    # 3折交叉验证
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # 元特征
    meta_features_train = np.zeros((len(X_train), 3))
    meta_features_test = np.zeros((len(X_test), 3))
    
    base_models = [
        ('lgb', lgb.LGBMRegressor(**lgb_params, n_estimators=500)),
        ('xgb', xgb.XGBRegressor(**xgb_params, n_estimators=500)),
        ('cat', CatBoostRegressor(**{**cat_params, 'iterations': 500}))
    ]
    
    test_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"快速Stacking第 {fold} 折...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_log, y_val_log = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        fold_test_preds = []
        
        for i, (name, model) in enumerate(base_models):
            if name == 'lgb':
                model.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], 
                         callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)])
            elif name == 'xgb':
                model.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], verbose=False)
            else:  # catboost
                model.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], early_stopping_rounds=30, verbose=False)
            
            val_pred = np.expm1(model.predict(X_val))
            meta_features_train[val_idx, i] = val_pred
            
            test_pred = np.expm1(model.predict(X_test))
            fold_test_preds.append(test_pred)
        
        test_predictions.append(fold_test_preds)
    
    # 平均测试集预测
    for i in range(len(base_models)):
        meta_features_test[:, i] = np.mean([fold[i] for fold in test_predictions], axis=0)
    
    # 元学习器
    meta_learners = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lgb_meta': lgb.LGBMRegressor(objective='mae', random_state=42, n_estimators=300)
    }
    
    meta_predictions = {}
    for name, meta_model in meta_learners.items():
        meta_model.fit(meta_features_train, y_train)
        meta_predictions[name] = meta_model.predict(meta_features_test)
    
    # 动态权重
    meta_scores = {}
    for name, meta_model in meta_learners.items():
        meta_pred_val = meta_model.predict(meta_features_train)
        meta_mae = mean_absolute_error(y_train, meta_pred_val)
        meta_scores[name] = meta_mae
    
    total_inv_score = sum(1/score for score in meta_scores.values())
    dynamic_weights = {name: (1/score) / total_inv_score for name, score in meta_scores.items()}
    
    # 加权融合
    final_stacking_pred = sum(dynamic_weights[name] * pred for name, pred in meta_predictions.items())
    
    return final_stacking_pred, meta_predictions

def fast_calibration_and_fusion(meta_predictions: Dict[str, np.ndarray], y_train: pd.Series) -> np.ndarray:
    """
    快速校准和融合
    """
    print("执行快速校准和融合...")
    
    # 分布校准
    train_quantiles = np.percentile(y_train, [10, 25, 50, 75, 90])
    
    # 使用最佳预测进行校准
    best_pred_name = list(meta_predictions.keys())[0]  # 简化：使用第一个
    base_pred = meta_predictions[best_pred_name]
    
    pred_quantiles = np.percentile(base_pred, [10, 25, 50, 75, 90])
    
    # 简化的分位数映射
    calibrated_pred = np.copy(base_pred)
    for i in range(len(train_quantiles)):
        if pred_quantiles[i] > 0:
            scale_factor = train_quantiles[i] / pred_quantiles[i]
            mask = np.abs(base_pred - pred_quantiles[i]) < (pred_quantiles[min(i+1, len(pred_quantiles)-1)] - pred_quantiles[max(i-1, 0)]) / 2
            calibrated_pred[mask] *= scale_factor
    
    # 自适应融合
    pred_matrix = np.column_stack(list(meta_predictions.values()))
    pred_weights = np.ones(len(meta_predictions)) / len(meta_predictions)
    
    # 基于方差的权重调整
    pred_variances = np.var(pred_matrix, axis=0)
    pred_weights = pred_weights / (pred_variances + 1e-6)
    pred_weights = pred_weights / np.sum(pred_weights)
    
    final_pred = sum(weight * pred for weight, pred in zip(pred_weights, meta_predictions.values()))
    
    # 最终处理
    final_pred = np.maximum(final_pred, 0)
    
    train_mean, train_std = y_train.mean(), y_train.std()
    pred_mean, pred_std = final_pred.mean(), final_pred.std()
    
    if pred_std > 0:
        final_pred = (final_pred - pred_mean) * (train_std / pred_std) + train_mean
    
    final_pred = np.maximum(final_pred, 0)
    final_pred = np.clip(final_pred, y_train.quantile(0.01), y_train.quantile(0.99))
    
    print(f"校准完成 - 均值: {final_pred.mean():.2f}, 标准差: {final_pred.std():.2f}")
    
    return final_pred

def v18_fast_optimize() -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    V18 Fast版本优化流程
    """
    print("=" * 80)
    print("开始V18 Fast版本优化 - 快速验证革命性策略")
    print("=" * 80)
    
    # 步骤1: 数据预处理
    print("步骤1: 快速数据预处理...")
    train_df, test_df = load_and_preprocess_data()
    
    # 步骤2: 核心革命性特征工程
    print("步骤2: 核心革命性特征工程...")
    train_df, target_encoder = create_core_revolutionary_features(train_df, is_train=True)
    test_df, _ = create_core_revolutionary_features(test_df, is_train=False, target_encoder=target_encoder)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 步骤3: 特征选择
    print("\n步骤3: 特征选择...")
    selector = SelectFromModel(estimator=RandomForestRegressor(n_estimators=50, random_state=42), 
                             threshold='median')
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"选择的特征数量: {len(selected_features)}")
    
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    
    # 步骤4: 特征缩放
    print("\n步骤4: 特征缩放...")
    scaler = RobustScaler()
    numeric_features = X_train_selected.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_features:
        if col in X_train_selected.columns and col in X_test_selected.columns:
            X_train_selected[col] = X_train_selected[col].fillna(X_train_selected[col].median())
            X_test_selected[col] = X_test_selected[col].fillna(X_train_selected[col].median())
            
            if X_train_selected[col].std() > 1e-8:
                X_train_selected[col] = scaler.fit_transform(X_train_selected[[col]])
                X_test_selected[col] = scaler.transform(X_test_selected[[col]])
    
    # 步骤5: 快速超参数优化
    print("\n步骤5: 快速超参数优化...")
    best_params = fast_hyperparameter_optimization(X_train_selected, y_train)
    
    # 步骤6: 快速多层Stacking
    print("\n步骤6: 快速多层Stacking...")
    stacking_pred, meta_predictions = fast_multi_stacking(
        X_train_selected, y_train, X_test_selected, best_params)
    
    # 步骤7: 快速校准和融合
    print("\n步骤7: 快速校准和融合...")
    final_predictions = fast_calibration_and_fusion(meta_predictions, y_train)
    
    # 最终统计
    print(f"\nV18 Fast最终预测统计:")
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
    result_file = os.path.join(result_dir, f"modeling_v18_fast_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\nV18 Fast结果已保存到: {result_file}")
    
    # 保存训练好的模型
    print("\n保存训练好的模型...")
    models_to_save = {}
    
    # 收集所有已训练的模型
    if 'lgb_model' in locals():
        models_to_save['lgb'] = lgb_model
    if 'xgb_model' in locals():
        models_to_save['xgb'] = xgb_model
    if 'cat_model' in locals():
        models_to_save['cat'] = cat_model
    if 'rf_model' in locals():
        models_to_save['rf'] = rf_model
    if 'ridge_model' in locals():
        models_to_save['ridge'] = ridge_model
    if 'meta_model' in locals():
        models_to_save['meta'] = meta_model
    
    # 保存模型
    if models_to_save:
        save_models(models_to_save, 'v18_fast')

    
    # 生成报告
    print("\n" + "=" * 80)
    print("V18 Fast版本优化总结")
    print("=" * 80)
    print("✅ 快速智能数据预处理 - 异常值检测+缺失值处理")
    print("✅ 核心革命性特征工程 - 目标编码+关键多项式+聚类特征")
    print("✅ 快速超参数优化 - 基于V17最佳参数的智能搜索")
    print("✅ 快速多层Stacking - 3个核心模型+动态权重")
    print("✅ 快速校准和融合 - 分布校准+自适应权重")
    print("🎯 快速验证V18革命性策略的有效性")
    print("=" * 80)
    
    return final_predictions, {
        'best_params': best_params,
        'meta_predictions': meta_predictions,
        'selected_features': selected_features
    }

if __name__ == "__main__":
    test_pred, model_info = v18_fast_optimize()
    print("V18 Fast版本优化完成! 验证革命性策略! 🚀")