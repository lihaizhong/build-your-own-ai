# -*- coding: utf-8 -*-
"""
V19 Verification版本模型 - 极简抗过拟合验证版

基于V16的成功经验，实施极简抗过拟合策略:
1. 最简特征工程 - 只保留V16验证有效的核心特征
2. 强正则化参数 - 基于V16增加正则化
3. 保守集成策略 - 等权重平均
4. 严格早停机制 - 防止过拟合
目标：快速验证MAE < 500的可行性
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
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

def load_and_preprocess_data():
    """极简数据预处理 - 基于V16成功经验"""
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 合并数据进行统一预处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 保守的异常值处理
    if 'power' in all_df.columns:
        all_df['power'] = np.clip(all_df['power'], 0, 600)
    
    # 简化的缺失值处理
    for col in ['fuelType', 'gearbox', 'bodyType']:
        mode_value = all_df[col].mode()
        if len(mode_value) > 0:
            all_df[col] = all_df[col].fillna(mode_value.iloc[0])

    model_mode = all_df['model'].mode()
    if len(model_mode) > 0:
        all_df['model'] = all_df['model'].fillna(model_mode.iloc[0])

    # 简化的时间特征工程
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year
    all_df['car_age'] = all_df['car_age'].fillna(0).astype(int)
    all_df.drop(columns=['regDate'], inplace=True)
    
    # 核心有效特征 - 只保留V16验证有效的
    if 'power' in all_df.columns:
        all_df['power_age_ratio'] = all_df['power'] / (all_df['car_age'] + 1)
    
    # 保守的品牌统计特征 - 基于V16的平滑因子
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
        # 使用V16的保守平滑因子40
        brand_stats['smooth_mean'] = (brand_stats['mean'] * brand_stats['count'] + 
                                     all_df['price'].mean() * 40) / (brand_stats['count'] + 40)
        brand_map = brand_stats.set_index('brand')['smooth_mean'].to_dict()
        all_df['brand_avg_price'] = all_df['brand'].map(brand_map)
    
    # 标签编码
    categorical_cols = ['model', 'brand', 'fuelType', 'gearbox', 'bodyType']
    for col in categorical_cols:
        if col in all_df.columns:
            le = LabelEncoder()
            all_df[col] = le.fit_transform(all_df[col].astype(str))
    
    # 填充数值型缺失值
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

def create_simple_features(df):
    """创建极简特征 - 基于V16的核心特征"""
    df = df.copy()
    
    # 基础分段特征 - V16的保守分段
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 3, 6, 10, float('inf')], 
                              labels=['very_new', 'new', 'medium', 'old'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    if 'power' in df.columns:
        df['power_segment'] = pd.cut(df['power'], bins=[-1, 100, 200, 300, float('inf')], 
                                    labels=['low', 'medium', 'high', 'very_high'])
        df['power_segment'] = df['power_segment'].cat.codes
    else:
        df['power_segment'] = 0
    
    # 核心业务特征 - 保守计算，避免极值
    if 'kilometer' in df.columns and 'car_age' in df.columns:
        # 年均里程 - V16的保守限制
        car_age_safe = np.maximum(df['car_age'], 0.1)
        df['km_per_year'] = df['kilometer'] / car_age_safe
        df['km_per_year'] = np.clip(df['km_per_year'], 0, 50000)
    
    # 简化的时间特征
    if 'car_age' in df.columns:
        df['log_car_age'] = np.log1p(df['car_age'])
    
    if 'kilometer' in df.columns:
        df['log_kilometer'] = np.log1p(df['kilometer'])
    
    # 只保留最有效的交叉特征
    if 'power' in df.columns and 'car_age' in df.columns:
        df['power_age_interaction'] = df['power'] * df['car_age']
    
    # 简化的v特征统计 - 基于V16
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
    
    # 保守的数据清理 - 基于V16
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 只处理明显的无穷大值
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # 填充NaN值
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
        
        # 保守的极值处理 - V16的分位数截断
        if col not in ['SaleID', 'price']:
            q999 = df[col].quantile(0.999)
            q001 = df[col].quantile(0.001)
            if q999 > q001 and q999 > 0:
                df[col] = np.clip(df[col], q001, q999)
    
    return df

def train_simple_models(X_train, y_train, X_test):
    """训练强正则化模型 - 基于V16但增加正则化"""
    print("训练强正则化模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 严格的3折交叉验证
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # 强正则化的模型参数 - 基于V16但增加正则化
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'num_leaves': 31,        # 保持保守
        'max_depth': 6,          # 降低深度
        'learning_rate': 0.05,   # 降低学习率
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.5,        # 增加L1正则化
        'lambda_l2': 0.5,        # 增加L2正则化
        'min_child_samples': 20,
        'random_state': 42,
    }
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': 6,          # 降低深度
        'learning_rate': 0.05,   # 降低学习率
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,        # 增加L1正则化
        'reg_lambda': 1.0,       # 增加L2正则化
        'min_child_weight': 10,
        'random_state': 42
    }
    
    catboost_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'depth': 6,              # 降低深度
        'learning_rate': 0.05,   # 降低学习率
        'iterations': 300,       # 减少迭代次数
        'l2_leaf_reg': 3.0,      # 增加L2正则化
        'random_strength': 0.5,
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
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        print(f"训练第 {fold} 折...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_log, y_val_log = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        # 训练LightGBM - 强正则化
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=500)  # 减少轮数
        lgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)])
        
        # 预测测试集
        lgb_predictions += np.expm1(np.array(lgb_model.predict(X_test))) / 3
        
        # 计算验证分数
        lgb_val_pred = np.expm1(np.array(lgb_model.predict(X_val)))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_val_pred)
        lgb_scores.append(lgb_mae)
        
        # 训练XGBoost - 强正则化
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=500, early_stopping_rounds=50)
        xgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     verbose=False)
        
        # 预测测试集
        xgb_predictions += np.expm1(xgb_model.predict(X_test)) / 3
        
        # 计算验证分数
        xgb_val_pred = np.expm1(xgb_model.predict(X_val))
        xgb_mae = mean_absolute_error(np.expm1(y_val_log), xgb_val_pred)
        xgb_scores.append(xgb_mae)
        
        # 训练CatBoost - 强正则化
        cat_model = CatBoostRegressor(**catboost_params)
        cat_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     early_stopping_rounds=50, 
                     verbose=False)
        
        # 预测测试集
        cat_predictions += np.expm1(cat_model.predict(X_test)) / 3
        
        # 计算验证分数
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

def simple_ensemble(lgb_pred, xgb_pred, cat_pred):
    """极简集成策略 - 等权重平均，避免过度优化"""
    print("执行极简集成策略...")
    
    # 等权重平均 - 保守策略
    ensemble_pred = (lgb_pred + xgb_pred + cat_pred) / 3
    
    print(f"使用等权重平均集成: LGB={1/3:.3f}, XGB={1/3:.3f}, CAT={1/3:.3f}")
    
    return ensemble_pred

def simple_calibration(predictions, y_train):
    """极简校准 - 基于V16的保守校准策略"""
    train_mean = y_train.mean()
    pred_mean = predictions.mean()
    
    print(f"\n极简校准:")
    print(f"  训练集均值: {train_mean:.2f}")
    print(f"  预测均值: {pred_mean:.2f}")
    
    # 保守的校准因子
    calibration_factor = train_mean / pred_mean if pred_mean > 0 else 1.0
    calibration_factor = np.clip(calibration_factor, 0.8, 1.2)  # 限制校准幅度
    print(f"  校准因子(限制后): {calibration_factor:.4f}")
    
    # 应用校准
    calibrated_predictions = predictions * calibration_factor
    
    # 确保预测值非负
    calibrated_predictions = np.maximum(calibrated_predictions, 0)
    
    return calibrated_predictions

def v19_simple_optimize():
    """V19极简抗过拟合优化模型训练流程"""
    print("=" * 80)
    print("开始V19极简抗过拟合优化模型训练")
    print("基于V16成功经验，避免V18过拟合问题")
    print("目标：MAE < 500")
    print("=" * 80)
    
    # 加载数据
    train_df, test_df = load_and_preprocess_data()
    train_df = create_simple_features(train_df)
    test_df = create_simple_features(test_df)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 特征缩放
    print("\n应用鲁棒特征缩放...")
    scaler = RobustScaler()
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_features:
        if col in X_train.columns and col in X_test.columns:
            # 检查无穷大值
            inf_mask = np.isinf(X_train[col]) | np.isinf(X_test[col])
            if inf_mask.any():
                X_train.loc[inf_mask[inf_mask.index.isin(X_train.index)].index, col] = 0
                X_test.loc[inf_mask[inf_mask.index.isin(X_test.index)].index, col] = 0
            
            # 检查NaN值
            X_train[col] = X_train[col].fillna(X_train[col].median())
            X_test[col] = X_test[col].fillna(X_train[col].median())
            
            # 跳过常数列
            if X_train[col].std() > 1e-8:
                X_train[col] = scaler.fit_transform(X_train[[col]])
                X_test[col] = scaler.transform(X_test[[col]])
    
    # 训练强正则化模型
    lgb_pred, xgb_pred, cat_pred, scores_info = train_simple_models(X_train, y_train, X_test)
    
    # 极简集成
    ensemble_pred = simple_ensemble(lgb_pred, xgb_pred, cat_pred)
    
    # 极简校准
    final_predictions = simple_calibration(ensemble_pred, y_train)
    
    print(f"\nV19 Verification最终预测统计:")
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
    result_file = os.path.join(result_dir, f"modeling_v19_simple_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\nV19 Verification结果已保存到: {result_file}")
    
    # 生成抗过拟合报告
    print("\n" + "=" * 80)
    print("V19 Verification抗过拟合优化总结")
    print("=" * 80)
    print("✅ 极简特征工程 - 只保留V16验证有效的核心特征")
    print("✅ 强正则化策略 - 增加L1/L2正则化和早停机制")
    print("✅ 极简集成策略 - 等权重平均，避免过度优化")
    print("✅ 严格早停机制 - 防止过拟合")
    print("🎯 目标：MAE < 500，避免过拟合")
    print("=" * 80)
    
    # 计算平均分数
    avg_score = np.mean([np.mean(scores_info['lgb_scores']), 
                         np.mean(scores_info['xgb_scores']), 
                         np.mean(scores_info['cat_scores'])])
    
    print(f"\n预期线上分数: {avg_score:.2f} (基于3折交叉验证)")
    if avg_score < 500:
        print("🎉 预期可以达到目标 (< 500)!")
    else:
        print("⚠️  预期分数仍高于目标，需要进一步优化")
    
    return final_predictions, scores_info

if __name__ == "__main__":
    test_pred, scores_info = v19_simple_optimize()
    print("V19 Verification抗过拟合优化完成! 期待稳定在500分以下!")