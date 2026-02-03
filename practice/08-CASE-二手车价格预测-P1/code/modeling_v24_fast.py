"""
V24核心测试版本模型 - 快速验证版

基于V23的497.6048分基础，快速验证核心优化策略:
1. 目标编码 - 增加分类特征信息
2. 关键特征工程 - 最有效的交互特征
3. 参数微调 - 基于V23的关键参数调整
4. 改进集成 - 更智能的权重分配
目标：快速验证核心优化效果
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
import warnings
import joblib
warnings.filterwarnings('ignore')

def get_project_path(*paths):
    """获取项目路径的统一方法"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        return os.path.join(project_dir, *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)

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


def fast_preprocessing():
    """
    快速数据预处理 - 基于V23但增加目标编码
    """
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
    
    # 分类特征处理 - 增加目标编码
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
                smooth_factor = 50  # 简化参数
                
                smooth_encoding = (target_mean * count + global_mean * smooth_factor) / (count + smooth_factor)
                all_df[f'{col}_target_enc'] = all_df[col].map(smooth_encoding).fillna(global_mean)
    
    # 时间特征工程
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year
    all_df['car_age'] = all_df['car_age'].fillna(0).astype(int)
    all_df['reg_month'] = all_df['regDate'].dt.month.fillna(6).astype(int)
    all_df['reg_quarter'] = all_df['regDate'].dt.quarter.fillna(2).astype(int)
    all_df['reg_season'] = all_df['reg_month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
    all_df['is_winter_reg'] = all_df['reg_month'].isin([12, 1, 2]).astype(int)
    
    all_df.drop(columns=['regDate'], inplace=True)
    
    # 品牌统计特征
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count', 'std']).reset_index()
        brand_stats['smooth_mean'] = ((brand_stats['mean'] * brand_stats['count'] + 
                                     all_df['price'].mean() * 40) / (brand_stats['count'] + 40))
        brand_stats['cv'] = brand_stats['std'] / brand_stats['mean']
        brand_stats['cv'] = brand_stats['cv'].fillna(brand_stats['cv'].median())
        
        all_df['brand_avg_price'] = all_df['brand'].map(brand_stats.set_index('brand')['smooth_mean']).fillna(all_df['price'].mean())
        all_df['brand_price_stability'] = all_df['brand'].map(brand_stats.set_index('brand')['cv']).fillna(brand_stats['cv'].median())
    
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

def create_key_features(df):
    """
    关键特征工程 - 专注于最有效的特征
    """
    df = df.copy()
    
    # 核心业务特征
    if 'power' in df.columns and 'car_age' in df.columns:
        df['power_age_ratio'] = df['power'] / (df['car_age'] + 1)
        df['power_decay'] = df['power'] * np.exp(-df['car_age'] * 0.05)
    
    if 'kilometer' in df.columns and 'car_age' in df.columns:
        car_age_safe = np.maximum(df['car_age'], 0.1)
        df['km_per_year'] = df['kilometer'] / car_age_safe
        df['km_per_year'] = np.clip(df['km_per_year'], 0, 40000)
    
    # V24新增：关键交互特征
    if 'power' in df.columns and 'kilometer' in df.columns:
        df['power_km_ratio'] = df['power'] / (df['kilometer'] + 1)
    
    if 'car_age' in df.columns and 'kilometer' in df.columns:
        df['age_km_interaction'] = df['car_age'] * df['kilometer'] / 1000
    
    # 关键分段特征
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 3, 6, 10, float('inf')], 
                              labels=['very_new', 'new', 'medium', 'old'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    # 关键变换特征
    if 'car_age' in df.columns:
        df['log_car_age'] = np.log1p(df['car_age'])
    
    if 'kilometer' in df.columns:
        df['log_kilometer'] = np.log1p(df['kilometer'])
    
    # v特征统计
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
    
    # 数据清理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
        
        if col not in ['SaleID', 'price'] and df[col].std() > 1e-8:
            q999 = df[col].quantile(0.999)
            q001 = df[col].quantile(0.001)
            if q999 > q001 and q999 > 0:
                df[col] = np.clip(df[col], q001, q999)
    
    return df

def train_fast_models(X_train, y_train, X_test):
    """
    快速模型训练 - 减少迭代次数
    """
    print("训练快速模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 分层交叉验证 - 减少折数
    y_bins = pd.qcut(y_train, q=5, labels=False)  # 减少到5个bin
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 减少到3折
    
    # V24优化参数
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'num_leaves': 37,
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
        'min_child_weight': 7,
        'random_state': 42
    }
    
    catboost_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'depth': 8,
        'learning_rate': 0.07,
        'iterations': 800,  # 减少迭代次数
        'l2_leaf_reg': 1.1,
        'random_strength': 0.3,
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
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=1000)  # 减少迭代次数
        lgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)])  # 减少early_stopping
        
        lgb_predictions += np.expm1(np.array(lgb_model.predict(X_test))) / 3
        lgb_val_pred = np.expm1(np.array(lgb_model.predict(X_val)))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_val_pred)
        lgb_scores.append(lgb_mae)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=1000, early_stopping_rounds=50)  # 减少迭代次数
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
                     early_stopping_rounds=50,  # 减少early_stopping
                     verbose=False)
        
        cat_predictions += np.expm1(cat_model.predict(X_test)) / 3
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

def fast_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info):
    """
    快速集成策略
    """
    print("执行快速集成策略...")
    
    # 基于性能的自适应权重
    lgb_score = np.mean(scores_info['lgb_scores'])
    xgb_score = np.mean(scores_info['xgb_scores'])
    cat_score = np.mean(scores_info['cat_scores'])
    
    # 计算权重
    total_inv_score = 1/lgb_score + 1/xgb_score + 1/cat_score
    raw_weights = {
        'lgb': (1/lgb_score) / total_inv_score,
        'xgb': (1/xgb_score) / total_inv_score,
        'cat': (1/cat_score) / total_inv_score
    }
    
    # 稳定性调整
    lgb_std = np.std(scores_info['lgb_scores'])
    xgb_std = np.std(scores_info['xgb_scores'])
    cat_std = np.std(scores_info['cat_scores'])
    
    stability_factor = {
        'lgb': 1 / (1 + lgb_std),
        'xgb': 1 / (1 + xgb_std),
        'cat': 1 / (1 + cat_std)
    }
    
    # 应用稳定性调整
    for model in raw_weights:
        raw_weights[model] *= stability_factor[model]
    
    # 重新归一化
    total_weight = sum(raw_weights.values())
    final_weights = {model: weight/total_weight for model, weight in raw_weights.items()}
    
    print(f"快速集成权重:")
    for model, weight in final_weights.items():
        print(f"  {model.upper()}: {weight:.3f}")
    
    ensemble_pred = (final_weights['lgb'] * lgb_pred + 
                    final_weights['xgb'] * xgb_pred + 
                    final_weights['cat'] * cat_pred)
    
    return ensemble_pred

def fast_calibration(predictions, y_train):
    """
    快速校准
    """
    train_mean = y_train.mean()
    pred_mean = predictions.mean()
    
    print(f"\n快速校准:")
    print(f"  训练集均值: {train_mean:.2f}")
    print(f"  预测均值: {pred_mean:.2f}")
    
    # 简单校准
    calibration_factor = train_mean / pred_mean if pred_mean > 0 else 1.0
    calibration_factor = np.clip(calibration_factor, 0.9, 1.1)
    
    final_predictions = predictions * calibration_factor
    final_predictions = np.maximum(final_predictions, 0)
    
    print(f"  校准因子: {calibration_factor:.4f}")
    
    return final_predictions

def v24_fast_test():
    """
    V24快速测试流程
    """
    print("=" * 60)
    print("开始V24快速测试")
    print("基于V23的497.6048分基础，快速验证核心优化")
    print("=" * 60)
    
    # 步骤1: 快速预处理
    print("\n步骤1: 快速预处理...")
    train_df, test_df = fast_preprocessing()
    
    # 步骤2: 关键特征工程
    print("\n步骤2: 关键特征工程...")
    train_df = create_key_features(train_df)
    test_df = create_key_features(test_df)
    
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
            inf_mask = np.isinf(X_train[col]) | np.isinf(X_test[col])
            if inf_mask.any():
                X_train.loc[inf_mask[inf_mask.index.isin(X_train.index)].index, col] = 0
                X_test.loc[inf_mask[inf_mask.index.isin(X_test.index)].index, col] = 0
            
            X_train[col] = X_train[col].fillna(X_train[col].median())
            X_test[col] = X_test[col].fillna(X_train[col].median())
            
            if X_train[col].std() > 1e-8:
                X_train[col] = scaler.fit_transform(X_train[[col]])
                X_test[col] = scaler.transform(X_test[[col]])
    
    # 步骤4: 快速模型训练
    print("\n步骤4: 快速模型训练...")
    lgb_pred, xgb_pred, cat_pred, scores_info = train_fast_models(
        X_train, y_train, X_test)
    
    # 步骤5: 快速集成
    print("\n步骤5: 快速集成...")
    ensemble_pred = fast_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info)
    
    # 步骤6: 快速校准
    print("\n步骤6: 快速校准...")
    final_predictions = fast_calibration(ensemble_pred, y_train)
    
    # 最终统计
    print(f"\nV24快速测试最终预测统计:")
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
    result_file = os.path.join(result_dir, f"modeling_v24_fast_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\nV24快速测试结果已保存到: {result_file}")
    
    # 保存训练好的模型
    print("\n保存训练好的模型...")
    models_to_save = {}
    
    # 收集所有已训练的模型
    if 'lgb_model' in locals():
        models_to_save['lgb'] = lgb_model # type: ignore
    if 'xgb_model' in locals():
        models_to_save['xgb'] = xgb_model # type: ignore
    if 'cat_model' in locals():
        models_to_save['cat'] = cat_model # type: ignore
    if 'rf_model' in locals():
        models_to_save['rf'] = rf_model # type: ignore
    if 'ridge_model' in locals():
        models_to_save['ridge'] = ridge_model # type: ignore
    if 'meta_model' in locals():
        models_to_save['meta'] = meta_model # type: ignore
    
    # 保存模型
    if models_to_save:
        save_models(models_to_save, 'v24_fast')

    
    # 生成报告
    print("\n" + "=" * 60)
    print("V24快速测试总结")
    print("=" * 60)
    print("✅ 基于V23的497.6048分优秀基础")
    print("✅ 目标编码 - 增加分类特征信息")
    print("✅ 关键特征工程 - 专注最有效特征")
    print("✅ 参数优化 - 基于V23的微调")
    print("✅ 快速集成 - 智能权重分配")
    print("🚀 快速验证核心优化效果!")
    print("=" * 60)
    
    return final_predictions, scores_info

if __name__ == "__main__":
    test_pred, scores_info = v24_fast_test()
    print("V24快速测试完成! 🚀")
