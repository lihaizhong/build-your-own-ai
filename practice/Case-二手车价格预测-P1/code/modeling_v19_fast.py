# -*- coding: utf-8 -*-
"""
V19 Fast版本模型 - 快速抗过拟合验证版

基于V19的抗过拟合策略，实施以下快速验证优化:
1. 极简特征工程 - 只保留V16核心特征
2. 快速正则化调优 - 基于V16参数增加正则化
3. 简化数据增强 - 轻量级噪声注入
4. 快速保守集成 - 等权重平均
5. 快速验证流程 - 3折交叉验证
目标：快速验证抗过拟合策略的有效性
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import warnings
import joblib
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

def load_and_preprocess_data():
    """
    快速保守的数据预处理 - 基于V16成功经验
    """
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
    
    # 核心有效特征
    if 'power' in all_df.columns:
        all_df['power_age_ratio'] = all_df['power'] / (all_df['car_age'] + 1)
    
    # 保守的品牌统计特征
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
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

def create_simple_robust_features(df):
    """
    创建简单鲁棒特征 - 基于V16的核心特征
    """
    df = df.copy()
    
    # 基础分段特征
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 3, 6, 10, float('inf')], 
                              labels=['very_new', 'new', 'medium', 'old'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    if 'power' in df.columns:
        df['power_segment'] = pd.cut(df['power'], bins=[-1, 100, 200, 300, float('inf')], 
                                    labels=['low', 'medium', 'high', 'very_high'])
        df['power_segment'] = df['power_segment'].cat.codes
    else:
        df['power_segment'] = 0
    
    # 核心业务特征
    if 'kilometer' in df.columns and 'car_age' in df.columns:
        car_age_safe = np.maximum(df['car_age'], 0.1)
        df['km_per_year'] = df['kilometer'] / car_age_safe
        df['km_per_year'] = np.clip(df['km_per_year'], 0, 50000)
    
    # 简化的时间特征
    if 'car_age' in df.columns:
        df['log_car_age'] = np.log1p(df['car_age'])
    
    if 'kilometer' in df.columns:
        df['log_kilometer'] = np.log1p(df['kilometer'])
    
    # 最有效的交叉特征
    if 'power' in df.columns and 'car_age' in df.columns:
        df['power_age_interaction'] = df['power'] * df['car_age']
    
    # 简化的v特征统计
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
    
    # 保守的数据清理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
        
        if col not in ['SaleID', 'price']:
            q999 = df[col].quantile(0.999)
            q001 = df[col].quantile(0.001)
            if q999 > q001 and q999 > 0:
                df[col] = np.clip(df[col], q001, q999)
    
    return df

def fast_noise_injection(X_train, y_train, noise_level=0.005):
    """
    快速数据增强 - 轻量级噪声注入
    """
    print(f"快速噪声注入 (噪声水平: {noise_level})...")
    
    X_train_noisy = X_train.copy()
    y_train_noisy = y_train.copy()
    
    # 只对关键特征添加噪声
    key_features = ['car_age', 'power', 'kilometer', 'brand_avg_price']
    key_features = [col for col in key_features if col in X_train.columns]
    
    for col in key_features:
        if X_train[col].std() > 1e-8:
            noise = np.random.normal(0, X_train[col].std() * noise_level, size=len(X_train))
            X_train_noisy[col] = X_train[col] + noise
    
    # 轻量级目标变量噪声
    y_noise = np.random.normal(0, y_train.std() * noise_level, size=len(y_train))
    y_train_noisy = y_train + y_noise
    y_train_noisy = np.maximum(y_train_noisy, 0)
    
    # 合并数据
    X_train_augmented = pd.concat([X_train, X_train_noisy], ignore_index=True)
    y_train_augmented = pd.concat([y_train, y_train_noisy], ignore_index=True)
    
    print(f"快速数据增强: {len(X_train)} -> {len(X_train_augmented)}")
    
    return X_train_augmented, y_train_augmented

def train_fast_regularized_models(X_train, y_train, X_test):
    """
    快速训练强正则化模型 - 3折交叉验证
    """
    print("快速训练强正则化模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 快速3折交叉验证
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # 基于V16的强正则化参数
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'num_leaves': 31,
        'max_depth': 6,          # 保守深度
        'learning_rate': 0.05,   # 保守学习率
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.5,        # 增强正则化
        'lambda_l2': 0.5,
        'min_child_samples': 20,
        'random_state': 42,
    }
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,        # 增强正则化
        'reg_lambda': 1.0,
        'min_child_weight': 10,
        'random_state': 42
    }
    
    catboost_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'depth': 6,
        'learning_rate': 0.05,
        'iterations': 300,       # 减少迭代次数
        'l2_leaf_reg': 3.0,      # 增强正则化
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
    
    # 快速交叉验证训练
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        print(f"快速训练第 {fold} 折...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_log, y_val_log = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        # 训练LightGBM
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=800)
        lgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)])
        
        lgb_predictions += np.expm1(np.array(lgb_model.predict(X_test))) / 3
        
        lgb_val_pred = np.expm1(np.array(lgb_model.predict(X_val)))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_val_pred)
        lgb_scores.append(lgb_mae)
        
        # 训练XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=800, early_stopping_rounds=50)
        xgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     verbose=False)
        
        xgb_predictions += np.expm1(xgb_model.predict(X_test)) / 3
        
        xgb_val_pred = np.expm1(xgb_model.predict(X_val))
        xgb_mae = mean_absolute_error(np.expm1(y_val_log), xgb_val_pred)
        xgb_scores.append(xgb_mae)
        
        # 训练CatBoost
        cat_model = CatBoostRegressor(**catboost_params)
        cat_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     early_stopping_rounds=50, 
                     verbose=False)
        
        cat_predictions += np.expm1(cat_model.predict(X_test)) / 3
        
        cat_val_pred = np.expm1(cat_model.predict(X_val))
        cat_mae = mean_absolute_error(np.expm1(y_val_log), cat_val_pred)
        cat_scores.append(cat_mae)
        
        print(f"  LightGBM MAE: {lgb_mae:.2f}, XGBoost MAE: {xgb_mae:.2f}, CatBoost MAE: {cat_mae:.2f}")
    
    print(f"\n快速验证平均分数:")
    print(f"  LightGBM: {np.mean(lgb_scores):.2f} (±{np.std(lgb_scores):.2f})")
    print(f"  XGBoost: {np.mean(xgb_scores):.2f} (±{np.std(xgb_scores):.2f})")
    print(f"  CatBoost: {np.mean(cat_scores):.2f} (±{np.std(cat_scores):.2f})")
    
    return lgb_predictions, xgb_predictions, cat_predictions, {
        'lgb_scores': lgb_scores,
        'xgb_scores': xgb_scores,
        'cat_scores': cat_scores
    }

def fast_conservative_ensemble(lgb_pred, xgb_pred, cat_pred):
    """
    快速保守集成 - 等权重平均
    """
    print("执行快速保守集成...")
    
    # 等权重平均
    ensemble_pred = (lgb_pred + xgb_pred + cat_pred) / 3
    
    print(f"快速等权重平均集成: LGB={1/3:.3f}, XGB={1/3:.3f}, CAT={1/3:.3f}")
    
    return ensemble_pred

def fast_robust_calibration(predictions, y_train):
    """
    快速鲁棒校准
    """
    train_mean = y_train.mean()
    pred_mean = predictions.mean()
    
    print(f"\n快速鲁棒校准:")
    print(f"  训练集均值: {train_mean:.2f}")
    print(f"  预测均值: {pred_mean:.2f}")
    
    # 保守校准
    calibration_factor = train_mean / pred_mean if pred_mean > 0 else 1.0
    calibration_factor = np.clip(calibration_factor, 0.8, 1.2)
    print(f"  校准因子: {calibration_factor:.4f}")
    
    calibrated_predictions = predictions * calibration_factor
    calibrated_predictions = np.maximum(calibrated_predictions, 0)
    
    return calibrated_predictions

def create_fast_analysis_plots(y_train, predictions, scores_info, model_name="V19_Fast"):
    """
    创建快速分析图表
    """
    print("生成快速分析图表...")
    
    # 创建保存目录
    analysis_dir = get_user_data_path()
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 价格分布对比
    axes[0, 0].hist(y_train, bins=50, alpha=0.7, label='训练集真实价格', color='blue')
    axes[0, 0].hist(predictions, bins=50, alpha=0.7, label='V19 Fast预测价格', color='red')
    axes[0, 0].set_xlabel('价格')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('V19 Fast价格分布对比')
    axes[0, 0].legend()
    
    # 2. 模型性能对比
    models = ['LightGBM', 'XGBoost', 'CatBoost']
    scores = [np.mean(scores_info['lgb_scores']), 
              np.mean(scores_info['xgb_scores']), 
              np.mean(scores_info['cat_scores'])]
    
    axes[0, 1].bar(models, scores, color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].axhline(y=500, color='red', linestyle='--', label='目标线(500)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('V19 Fast各模型验证性能')
    axes[0, 1].legend()
    
    # 3. 预测值分布
    axes[1, 0].hist(predictions, bins=50, alpha=0.7, color='red')
    axes[1, 0].set_xlabel('预测价格')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('V19 Fast预测值分布')
    
    # 4. 快速统计信息
    avg_score = np.mean([np.mean(scores_info['lgb_scores']), 
                         np.mean(scores_info['xgb_scores']), 
                         np.mean(scores_info['cat_scores'])])
    
    stats_text = f"""
    V19 Fast抗过拟合统计:
    
    训练集统计:
    均值: {y_train.mean():.2f}
    标准差: {y_train.std():.2f}
    范围: {y_train.min():.2f} - {y_train.max():.2f}
    
    预测集统计:
    均值: {predictions.mean():.2f}
    标准差: {predictions.std():.2f}
    范围: {predictions.min():.2f} - {predictions.max():.2f}
    
    快速验证结果:
    平均MAE: {avg_score:.2f}
    目标MAE: < 500
    
    抗过拟合策略:
    极简特征工程
    强正则化
    快速数据增强
    保守集成
    """
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center')
    axes[1, 1].set_title('V19 Fast快速统计')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(analysis_dir, f'{model_name}_fast_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"V19 Fast分析图表已保存到: {chart_path}")
    plt.show()

def v19_fast_anti_overfitting_optimize():
    """
    V19 Fast抗过拟合优化模型训练流程
    """
    print("=" * 80)
    print("开始V19 Fast抗过拟合优化模型训练")
    print("快速验证抗过拟合策略")
    print("=" * 80)
    
    # 加载数据
    train_df, test_df = load_and_preprocess_data()
    train_df = create_simple_robust_features(train_df)
    test_df = create_simple_robust_features(test_df)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 快速数据增强
    X_train_augmented, y_train_augmented = fast_noise_injection(X_train, y_train, noise_level=0.005)
    
    # 特征缩放
    print("\n应用快速特征缩放...")
    scaler = RobustScaler()
    numeric_features = X_train_augmented.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_features:
        if col in X_train_augmented.columns and col in X_test.columns:
            X_train_augmented[col] = X_train_augmented[col].fillna(X_train_augmented[col].median())
            X_test[col] = X_test[col].fillna(X_train_augmented[col].median())
            
            if X_train_augmented[col].std() > 1e-8:
                X_train_augmented[col] = scaler.fit_transform(X_train_augmented[[col]])
                X_test[col] = scaler.transform(X_test[[col]])
    
    # 快速训练强正则化模型
    lgb_pred, xgb_pred, cat_pred, scores_info = train_fast_regularized_models(
        X_train_augmented, y_train_augmented, X_test)
    
    # 快速保守集成
    ensemble_pred = fast_conservative_ensemble(lgb_pred, xgb_pred, cat_pred)
    
    # 快速鲁棒校准
    final_predictions = fast_robust_calibration(ensemble_pred, y_train)
    
    # 创建快速分析图表
    create_fast_analysis_plots(y_train, final_predictions, scores_info, "V19_Fast")
    
    print(f"\nV19 Fast最终预测统计:")
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
    result_file = os.path.join(result_dir, f"modeling_v19_fast_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\nV19 Fast结果已保存到: {result_file}")
    
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
        save_models(models_to_save, 'v19_fast')

    
    # 生成快速报告
    print("\n" + "=" * 80)
    print("V19 Fast抗过拟合优化总结")
    print("=" * 80)
    print("极简特征工程 - 只保留V16核心特征")
    print("快速正则化调优 - 基于V16参数增加正则化")
    print("简化数据增强 - 轻量级噪声注入")
    print("快速保守集成 - 等权重平均")
    print("快速验证流程 - 3折交叉验证")
    print("快速验证抗过拟合策略有效性")
    print("=" * 80)
    
    return final_predictions, scores_info

if __name__ == "__main__":
    test_pred, scores_info = v19_fast_anti_overfitting_optimize()
    print("V19 Fast抗过拟合优化完成! 快速验证策略有效性!")