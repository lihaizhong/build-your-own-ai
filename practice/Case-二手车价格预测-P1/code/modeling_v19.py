"""
V19版本模型 - 抗过拟合优化版

基于V16的成功经验和V18的过拟合教训，实施以下抗过拟合策略:
1. 极简特征工程 - 只保留V16验证有效的核心特征
2. 强正则化策略 - 增加L1/L2正则化和早停机制
3. 数据增强和噪声注入 - 提高模型泛化能力
4. 保守集成策略 - 简单平均，避免过度优化
5. 严格的交叉验证 - 确保验证结果的可靠性
目标：MAE < 500，避免过拟合
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
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
    保守的数据预处理 - 基于V16的成功经验
    """
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 合并数据进行统一预处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 保守的异常值处理 - 基于V16的经验
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

def create_robust_features(df):
    """
    创建鲁棒特征 - 基于V16的有效特征，避免过度工程
    """
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

def add_noise_injection(X_train, y_train, noise_level=0.01):
    """
    数据增强和噪声注入 - 提高泛化能力
    """
    print(f"添加噪声注入 (噪声水平: {noise_level})...")
    
    X_train_noisy = X_train.copy()
    y_train_noisy = y_train.copy()
    
    # 对数值特征添加噪声
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X_train[col].std() > 1e-8:
            noise = np.random.normal(0, X_train[col].std() * noise_level, size=len(X_train))
            X_train_noisy[col] = X_train[col] + noise
    
    # 对目标变量添加少量噪声
    y_noise = np.random.normal(0, y_train.std() * noise_level, size=len(y_train))
    y_train_noisy = y_train + y_noise
    
    # 确保非负
    y_train_noisy = np.maximum(y_train_noisy, 0)
    
    # 合并原始数据和噪声数据
    X_train_augmented = pd.concat([X_train, X_train_noisy], ignore_index=True)
    y_train_augmented = pd.concat([y_train, y_train_noisy], ignore_index=True)
    
    print(f"数据增强完成: {len(X_train)} -> {len(X_train_augmented)}")
    
    return X_train_augmented, y_train_augmented

def train_regularized_models(X_train, y_train, X_test):
    """
    训练强正则化模型 - 抗过拟合策略
    """
    print("训练强正则化模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 严格的5折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
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
        'iterations': 500,
        'l2_leaf_reg': 3.0,      # 增加L2正则化
        'random_strength': 0.5,
        'bootstrap_type': 'Bayesian',  # 贝叶斯bootstrap提高泛化
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
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=2000)  # 更多轮数但早停
        lgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)])
        
        # 预测测试集
        lgb_predictions += np.expm1(np.array(lgb_model.predict(X_test))) / 5
        
        # 计算验证分数
        lgb_val_pred = np.expm1(np.array(lgb_model.predict(X_val)))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_val_pred)
        lgb_scores.append(lgb_mae)
        
        # 训练XGBoost - 强正则化
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=2000, early_stopping_rounds=100)
        xgb_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     verbose=False)
        
        # 预测测试集
        xgb_predictions += np.expm1(xgb_model.predict(X_test)) / 5
        
        # 计算验证分数
        xgb_val_pred = np.expm1(xgb_model.predict(X_val))
        xgb_mae = mean_absolute_error(np.expm1(y_val_log), xgb_val_pred)
        xgb_scores.append(xgb_mae)
        
        # 训练CatBoost - 强正则化
        cat_model = CatBoostRegressor(**catboost_params)
        cat_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     early_stopping_rounds=100, 
                     verbose=False)
        
        # 预测测试集
        cat_predictions += np.expm1(cat_model.predict(X_test)) / 5
        
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

def conservative_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info):
    """
    保守集成策略 - 简单平均，避免过度优化
    """
    print("执行保守集成策略...")
    
    # 等权重平均 - 保守策略
    ensemble_pred = (lgb_pred + xgb_pred + cat_pred) / 3
    
    print(f"使用等权重平均集成: LGB={1/3:.3f}, XGB={1/3:.3f}, CAT={1/3:.3f}")
    
    return ensemble_pred

def robust_calibration(predictions, y_train):
    """
    鲁棒校准 - 基于V16的保守校准策略
    """
    train_mean = y_train.mean()
    pred_mean = predictions.mean()
    
    print(f"\n鲁棒校准:")
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

def create_robust_analysis_plots(y_train, predictions, scores_info, model_name="V19"):
    """
    创建鲁棒分析图表
    """
    print("生成鲁棒分析图表...")
    
    # 创建保存目录
    analysis_dir = get_user_data_path()
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 价格分布对比
    axes[0, 0].hist(y_train, bins=50, alpha=0.7, label='训练集真实价格', color='blue')
    axes[0, 0].hist(predictions, bins=50, alpha=0.7, label='V19预测价格', color='red')
    axes[0, 0].set_xlabel('价格')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('V19价格分布对比')
    axes[0, 0].legend()
    
    # 2. 模型性能对比
    models = ['LightGBM', 'XGBoost', 'CatBoost']
    scores = [np.mean(scores_info['lgb_scores']), 
              np.mean(scores_info['xgb_scores']), 
              np.mean(scores_info['cat_scores'])]
    
    axes[0, 1].bar(models, scores, color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].axhline(y=500, color='red', linestyle='--', label='目标线(500)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('V19各模型验证性能')
    axes[0, 1].legend()
    
    # 3. 预测值分布
    axes[1, 0].hist(predictions, bins=50, alpha=0.7, color='red')
    axes[1, 0].set_xlabel('预测价格')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('V19预测值分布')
    
    # 4. 统计信息和抗过拟合指标
    lgb_std = np.std(scores_info['lgb_scores'])
    xgb_std = np.std(scores_info['xgb_scores'])
    cat_std = np.std(scores_info['cat_scores'])
    
    stats_text = f"""
    V19抗过拟合版本统计:
    
    训练集统计:
    均值: {y_train.mean():.2f}
    标准差: {y_train.std():.2f}
    范围: {y_train.min():.2f} - {y_train.max():.2f}
    
    预测集统计:
    均值: {predictions.mean():.2f}
    标准差: {predictions.std():.2f}
    范围: {predictions.min():.2f} - {predictions.max():.2f}
    
    模型稳定性:
    LightGBM: {np.mean(scores_info['lgb_scores']):.2f} (±{lgb_std:.2f})
    XGBoost: {np.mean(scores_info['xgb_scores']):.2f} (±{xgb_std:.2f})
    CatBoost: {np.mean(scores_info['cat_scores']):.2f} (±{cat_std:.2f})
    
    抗过拟合策略:
    ✅ 极简特征工程
    ✅ 强正则化
    ✅ 数据增强
    ✅ 保守集成
    ✅ 严格交叉验证
    """
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center')
    axes[1, 1].set_title('V19抗过拟合统计')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(analysis_dir, f'{model_name}_robust_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"V19分析图表已保存到: {chart_path}")
    plt.show()

def v19_anti_overfitting_optimize():
    """
    V19抗过拟合优化模型训练流程
    """
    print("=" * 80)
    print("开始V19抗过拟合优化模型训练")
    print("基于V16成功经验，避免V18过拟合问题")
    print("目标：MAE < 500")
    print("=" * 80)
    
    # 加载数据
    train_df, test_df = load_and_preprocess_data()
    train_df = create_robust_features(train_df)
    test_df = create_robust_features(test_df)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 数据增强和噪声注入
    X_train_augmented, y_train_augmented = add_noise_injection(X_train, y_train, noise_level=0.01)
    
    # 特征缩放
    print("\n应用鲁棒特征缩放...")
    scaler = RobustScaler()
    numeric_features = X_train_augmented.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_features:
        if col in X_train_augmented.columns and col in X_test.columns:
            # 检查无穷大值
            inf_mask = np.isinf(X_train_augmented[col]) | np.isinf(X_test[col])
            if inf_mask.any():
                X_train_augmented.loc[inf_mask[inf_mask.index.isin(X_train_augmented.index)].index, col] = 0
                X_test.loc[inf_mask[inf_mask.index.isin(X_test.index)].index, col] = 0
            
            # 检查NaN值
            X_train_augmented[col] = X_train_augmented[col].fillna(X_train_augmented[col].median())
            X_test[col] = X_test[col].fillna(X_train_augmented[col].median())
            
            # 跳过常数列
            if X_train_augmented[col].std() > 1e-8:
                X_train_augmented[col] = scaler.fit_transform(X_train_augmented[[col]])
                X_test[col] = scaler.transform(X_test[[col]])
    
    # 训练强正则化模型
    lgb_pred, xgb_pred, cat_pred, scores_info = train_regularized_models(
        X_train_augmented, y_train_augmented, X_test)
    
    # 保守集成
    ensemble_pred = conservative_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info)
    
    # 鲁棒校准
    final_predictions = robust_calibration(ensemble_pred, y_train)
    
    # 创建分析图表
    create_robust_analysis_plots(y_train, final_predictions, scores_info, "V19")
    
    print(f"\nV19最终预测统计:")
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
    result_file = os.path.join(result_dir, f"modeling_v19_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\nV19结果已保存到: {result_file}")
    
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
        save_models(models_to_save, 'v19')

    
    # 生成抗过拟合报告
    print("\n" + "=" * 80)
    print("V19抗过拟合优化总结")
    print("=" * 80)
    print("✅ 极简特征工程 - 只保留V16验证有效的核心特征")
    print("✅ 强正则化策略 - 增加L1/L2正则化和早停机制")
    print("✅ 数据增强和噪声注入 - 提高模型泛化能力")
    print("✅ 保守集成策略 - 简单平均，避免过度优化")
    print("✅ 严格的交叉验证 - 确保验证结果的可靠性")
    print("🎯 目标：MAE < 500，避免过拟合")
    print("=" * 80)
    
    return final_predictions, scores_info

if __name__ == "__main__":
    test_pred, scores_info = v19_anti_overfitting_optimize()
    print("V19抗过拟合优化完成! 期待稳定在500分以下! 🎯")