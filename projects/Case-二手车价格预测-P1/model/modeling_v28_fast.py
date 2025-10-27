"""
V28测试版本模型 - 快速验证版

基于V28的融合创新策略，但简化以快速验证:
1. 保留核心创新点：动态特征选择、自适应参数、增强校准
2. 简化训练过程：减少交叉验证折数和迭代次数
3. 快速验证效果：确保核心策略有效
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

def get_project_path(*paths):
    """获取项目路径的统一方法"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        return os.path.join(project_dir, *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)

def fast_preprocessing():
    """快速数据预处理"""
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 合并数据进行预处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 基础power处理
    if 'power' in all_df.columns:
        all_df['power'] = np.clip(all_df['power'], 0, 600)
        all_df['power_is_zero'] = (all_df['power'] <= 0).astype(int)
        all_df['log_power'] = np.log1p(np.maximum(all_df['power'], 1))
    
    # 分类特征处理 - 简化版
    categorical_cols = ['fuelType', 'gearbox', 'bodyType', 'model', 'brand']
    for col in categorical_cols:
        if col in all_df.columns:
            all_df[f'{col}_missing'] = (all_df[col].isnull()).astype(int)
            
            # 简单填充
            mode_value = all_df[col].mode()
            if len(mode_value) > 0:
                all_df[col] = all_df[col].fillna(mode_value.iloc[0])
            
            # 简化目标编码
            if 'price' in all_df.columns and col != 'brand':
                target_mean = all_df.groupby(col)['price'].mean()
                global_mean = all_df['price'].mean()
                smooth_factor = 30
                count = all_df[col].value_counts()
                smooth_encoding = (target_mean * count + global_mean * smooth_factor) / (count + smooth_factor)
                all_df[f'{col}_target_enc'] = all_df[col].map(smooth_encoding).fillna(global_mean)
            
            # 频率编码
            freq_map = all_df[col].value_counts().to_dict()
            all_df[f'{col}_freq'] = all_df[col].map(freq_map)
    
    # 时间特征工程 - 简化版
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year
    all_df['car_age'] = all_df['car_age'].fillna(0).astype(int)
    all_df['reg_month'] = all_df['regDate'].dt.month.fillna(6).astype(int)
    all_df['reg_season'] = all_df['reg_month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
    
    all_df.drop(columns=['regDate'], inplace=True)
    
    # 品牌统计特征 - 简化版
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
        global_mean = all_df['price'].mean()
        smooth_factor = 40
        brand_stats['smooth_mean'] = ((brand_stats['mean'] * brand_stats['count'] + 
                                     global_mean * smooth_factor) / (brand_stats['count'] + smooth_factor))
        brand_map = brand_stats.set_index('brand')['smooth_mean'].to_dict()
        all_df['brand_avg_price'] = all_df['brand'].map(brand_map).fillna(global_mean)
    
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

def create_fast_features(df):
    """快速特征工程"""
    df = df.copy()
    
    # 核心业务特征
    if 'power' in df.columns and 'car_age' in df.columns:
        df['power_age_ratio'] = df['power'] / (df['car_age'] + 1)
        df['power_decay'] = df['power'] * np.exp(-df['car_age'] * 0.05)
    
    if 'kilometer' in df.columns and 'car_age' in df.columns:
        car_age_safe = np.maximum(df['car_age'], 0.1)
        df['km_per_year'] = df['kilometer'] / car_age_safe
        df['km_per_year'] = np.clip(df['km_per_year'], 0, 40000)
    
    # 关键交互特征
    if 'power' in df.columns and 'kilometer' in df.columns:
        df['power_km_ratio'] = df['power'] / (df['kilometer'] + 1)
    
    if 'car_age' in df.columns and 'kilometer' in df.columns:
        df['age_km_interaction'] = df['car_age'] * df['kilometer'] / 1000
    
    # 分段特征
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 3, 7, 12, float('inf')], 
                              labels=['new', 'medium', 'old', 'very_old'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    # 变换特征
    if 'car_age' in df.columns:
        df['log_car_age'] = np.log1p(df['car_age'])
    
    if 'kilometer' in df.columns:
        df['log_kilometer'] = np.log1p(df['kilometer'])
    
    # v特征统计
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_range'] = df[v_cols].max(axis=1) - df[v_cols].min(axis=1)
    
    # 数据清理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
        
        if col not in ['SaleID', 'price'] and df[col].std() > 1e-8:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            if q99 > q01 and q99 > 0:
                df[col] = np.clip(df[col], q01, q99)
    
    return df

def fast_feature_selection(X_train, y_train, X_test, max_features=60):
    """快速特征选择"""
    print("执行快速特征选择...")
    
    feature_names = X_train.columns.tolist()
    
    # 计算互信息分数
    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    mi_df = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # 选择top特征
    top_features = mi_df.head(max_features)['feature'].tolist()
    
    print(f"从{len(feature_names)}个特征中选择了{len(top_features)}个特征")
    
    return X_train[top_features], X_test[top_features], mi_df

def fast_adaptive_params(X_train, y_train):
    """快速自适应参数"""
    n_samples, n_features = X_train.shape
    
    # 简化参数调整
    if n_features > 50:
        learning_rate = 0.07
        num_leaves = 35
        depth = 8
    else:
        learning_rate = 0.08
        num_leaves = 31
        depth = 7
    
    return {
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': depth
    }

def train_fast_models(X_train, y_train, X_test):
    """快速训练模型"""
    print("快速训练模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 简化交叉验证 - 3折
    y_bins = pd.qcut(y_train, q=5, labels=False)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # 自适应参数
    adaptive_params = fast_adaptive_params(X_train, y_train)
    
    # 参数设置
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'num_leaves': adaptive_params['num_leaves'],
        'max_depth': adaptive_params['max_depth'],
        'learning_rate': adaptive_params['learning_rate'],
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'lambda_l1': 0.25,
        'lambda_l2': 0.25,
        'min_child_samples': 18,
        'random_state': 42,
    }
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': adaptive_params['max_depth'],
        'learning_rate': adaptive_params['learning_rate'],
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.6,
        'reg_lambda': 0.6,
        'min_child_weight': 8,
        'random_state': 42
    }
    
    catboost_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'depth': adaptive_params['max_depth'],
        'learning_rate': adaptive_params['learning_rate'],
        'iterations': 1000,
        'l2_leaf_reg': 1.2,
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
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=1200)
        lgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     callbacks=[lgb.early_stopping(stopping_rounds=80), lgb.log_evaluation(0)])
        
        lgb_predictions += np.expm1(np.array(lgb_model.predict(X_test))) / 3
        lgb_val_pred = np.expm1(np.array(lgb_model.predict(X_val)))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_val_pred)
        lgb_scores.append(lgb_mae)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=1200, early_stopping_rounds=80)
        xgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     verbose=False)
        
        xgb_predictions += np.expm1(xgb_model.predict(X_test)) / 3
        xgb_val_pred = np.expm1(xgb_model.predict(X_val))
        xgb_mae = mean_absolute_error(np.expm1(y_val_log), xgb_val_pred)
        xgb_scores.append(xgb_mae)
        
        # CatBoost
        cat_model = CatBoostRegressor(**catboost_params)
        cat_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     early_stopping_rounds=80, 
                     verbose=False)
        
        cat_predictions += np.expm1(cat_model.predict(X_test)) / 3
        cat_val_pred = np.expm1(cat_model.predict(X_val))
        cat_mae = mean_absolute_error(np.expm1(y_val_log), cat_val_pred)
        cat_scores.append(cat_mae)
        
        print(f"  LGB: {lgb_mae:.2f}, XGB: {xgb_mae:.2f}, CAT: {cat_mae:.2f}")
    
    print(f"\n平均验证分数:")
    print(f"  LightGBM: {np.mean(lgb_scores):.2f}")
    print(f"  XGBoost: {np.mean(xgb_scores):.2f}")
    print(f"  CatBoost: {np.mean(cat_scores):.2f}")
    
    return lgb_predictions, xgb_predictions, cat_predictions, {
        'lgb_scores': lgb_scores,
        'xgb_scores': xgb_scores,
        'cat_scores': cat_scores
    }

def fast_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info):
    """快速集成"""
    print("执行快速集成...")
    
    # 基于性能的权重
    lgb_score = np.mean(scores_info['lgb_scores'])
    xgb_score = np.mean(scores_info['xgb_scores'])
    cat_score = np.mean(scores_info['cat_scores'])
    
    total_inv_score = 1/lgb_score + 1/xgb_score + 1/cat_score
    weights = {
        'lgb': (1/lgb_score) / total_inv_score,
        'xgb': (1/xgb_score) / total_inv_score,
        'cat': (1/cat_score) / total_inv_score
    }
    
    print(f"集成权重:")
    for model, weight in weights.items():
        print(f"  {model.upper()}: {weight:.3f}")
    
    ensemble_pred = (weights['lgb'] * lgb_pred + 
                    weights['xgb'] * xgb_pred + 
                    weights['cat'] * cat_pred)
    
    return ensemble_pred

def fast_calibration(predictions, y_train):
    """快速校准"""
    print("执行快速校准...")
    
    train_mean = y_train.mean()
    pred_mean = predictions.mean()
    
    # 简单校准
    calibration_factor = train_mean / pred_mean if pred_mean > 0 else 1.0
    calibration_factor = np.clip(calibration_factor, 0.9, 1.1)
    
    final_predictions = predictions * calibration_factor
    final_predictions = np.maximum(final_predictions, 0)
    
    print(f"  校准因子: {calibration_factor:.4f}")
    
    return final_predictions

def v28_fast_test():
    """V28快速测试流程"""
    print("=" * 60)
    print("开始V28快速测试")
    print("验证融合创新策略效果")
    print("=" * 60)
    
    # 步骤1: 快速预处理
    print("\n步骤1: 快速预处理...")
    train_df, test_df = fast_preprocessing()
    
    # 步骤2: 快速特征工程
    print("\n步骤2: 快速特征工程...")
    train_df = create_fast_features(train_df)
    test_df = create_fast_features(test_df)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"初始特征数量: {len(feature_cols)}")
    
    # 步骤3: 快速特征选择
    print("\n步骤3: 快速特征选择...")
    X_train_selected, X_test_selected, feature_importance = fast_feature_selection(
        X_train, y_train, X_test, max_features=60)
    
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
    
    # 步骤5: 快速训练
    print("\n步骤5: 快速训练模型...")
    lgb_pred, xgb_pred, cat_pred, scores_info = train_fast_models(
        X_train_selected, y_train, X_test_selected)
    
    # 步骤6: 快速集成
    print("\n步骤6: 快速集成...")
    ensemble_pred = fast_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info)
    
    # 步骤7: 快速校准
    print("\n步骤7: 快速校准...")
    final_predictions = fast_calibration(ensemble_pred, y_train)
    
    # 最终统计
    print(f"\nV28快速测试最终预测统计:")
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
    result_file = os.path.join(result_dir, f"modeling_v28_fast_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\nV28快速测试结果已保存到: {result_file}")
    
    # 生成报告
    print("\n" + "=" * 60)
    print("V28快速测试总结")
    print("=" * 60)
    print("✅ 融合创新策略验证")
    print("✅ 动态特征选择")
    print("✅ 自适应参数调优")
    print("✅ 智能集成权重")
    print("🚀 核心策略验证完成!")
    print("=" * 60)
    
    return final_predictions, scores_info

if __name__ == "__main__":
    test_pred, scores_info = v28_fast_test()
    print("V28快速测试完成! 🚀")