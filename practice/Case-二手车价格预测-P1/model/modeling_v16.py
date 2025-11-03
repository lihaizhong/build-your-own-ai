"""
V16版本模型 - 修复性能下降问题

基于V12的成功经验和V15的教训，实施以下优化策略:
1. 简化特征工程，保留有效特征
2. 添加早停机制和验证曲线监控
3. 使用适度的数值处理，避免过度截断
4. 简化集成策略，提高泛化能力
5. 保持数值稳定性，但不过度处理
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
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_project_path(*paths: str):
    """获取项目路径的统一方法"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        return os.path.join(project_dir, *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)

def get_user_data_path(*paths: str):
    """获取用户数据路径"""
    return get_project_path('user_data', *paths)

def load_and_preprocess_data():
    """
    加载并预处理数据 - 基于V12的成功经验
    """
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 合并数据进行统一预处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 处理power异常值 - 适度的截断
    if 'power' in all_df.columns:
        all_df['power'] = np.clip(all_df['power'], 0, 600)
    
    # 分类特征缺失值处理
    for col in ['fuelType', 'gearbox', 'bodyType']:
        mode_value = all_df[col].mode()
        if len(mode_value) > 0:
            all_df[col] = all_df[col].fillna(mode_value.iloc[0])
        all_df[f'{col}_missing'] = (all_df[col].isnull()).astype(int)

    model_mode = all_df['model'].mode()
    if len(model_mode) > 0:
        all_df['model'] = all_df['model'].fillna(model_mode.iloc[0])

    # 特征工程 - 简化版本，保留V12的有效特征
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year
    all_df['car_age'] = all_df['car_age'].fillna(0).astype(int)
    all_df.drop(columns=['regDate'], inplace=True)
    
    # 核心有效特征 - 基于V12的经验
    if 'power' in all_df.columns:
        all_df['power_age_ratio'] = all_df['power'] / (all_df['car_age'] + 1)
    
    # 品牌统计特征（适度平滑）
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
        # 适度平滑因子40 - V12的成功参数
        brand_stats['smooth_mean'] = (brand_stats['mean'] * brand_stats['count'] + all_df['price'].mean() * 40) / (brand_stats['count'] + 40)
        brand_map: dict[str, float] = brand_stats.set_index('brand')['smooth_mean'].to_dict()
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
        null_count = all_df[col].isnull().sum()
        if col not in ['price', 'SaleID'] and null_count > 0:
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

def create_effective_features(df: pd.DataFrame):
    """
    创建有效特征 - 基于V12的成功经验，避免过度工程
    """
    df = df.copy()
    
    # 基础分段特征 - 适度分段
    df['age_segment'] = pd.cut(
        df['car_age'],
        bins=[-1, 3, 6, 10, float('inf')], 
        labels=['very_new', 'new', 'medium', 'old']
    )
    df['age_segment'] = df['age_segment'].cat.codes
    
    if 'power' in df.columns:
        df['power_segment'] = pd.cut(
            df['power'],
            bins=[-1, 100, 200, 300, float('inf')], 
            labels=['low', 'medium', 'high', 'very_high']
        )
        df['power_segment'] = df['power_segment'].cat.codes
    else:
        df['power_segment'] = 0
    
    # 核心业务特征 - 适度计算，避免极值
    if 'kilometer' in df.columns and 'car_age' in df.columns:
        # 年均里程 - 适度限制
        car_age_safe = np.maximum(df['car_age'], 0.1)
        df['km_per_year'] = df['kilometer'] / car_age_safe
        # 温和的极值处理，保留更多原始信息
        df['km_per_year'] = np.clip(df['km_per_year'], 0, 50000)
    
    # 时间特征 - 保持简单有效
    if 'car_age' in df.columns:
        df['car_age_squared'] = df['car_age'] ** 2
        df['log_car_age'] = np.log1p(df['car_age'])
    
    if 'kilometer' in df.columns:
        df['log_kilometer'] = np.log1p(df['kilometer'])
    
    # 交叉特征 - 只保留最有效的
    if 'power' in df.columns and 'car_age' in df.columns:
        df['power_age_interaction'] = df['power'] * df['car_age']
    
    # 统计特征 - 简化版本
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
    
    # 温和的数值检查 - 只处理明显问题
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 只处理明显的无穷大值
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # 填充NaN值
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
        
        # 温和的极值处理 - 只处理极端异常值
        if col not in ['SaleID', 'price']:
            q999 = df[col].quantile(0.999)
            q001 = df[col].quantile(0.001)
            if q999 > q001 and q999 > 0:
                df[col] = np.clip(df[col], q001, q999)
    
    return df

def train_models_with_early_stopping(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    使用早停机制训练模型 - 基于V12的成功经验
    """
    print("使用早停机制训练模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 5折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 模型参数 - 基于V12的成功配置
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_child_samples': 20,
        'random_state': 42,
    }
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'min_child_weight': 10,
        'random_state': 42
    }
    
    catboost_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'depth': 6,
        'learning_rate': 0.1,
        'iterations': 500,  # 增加迭代次数，让早停机制发挥作用
        'l2_leaf_reg': 0.5,
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
        
        # 训练LightGBM
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=1000) # type: ignore
        lgb_model.fit(
            X_tr,
            y_tr_log, 
            eval_set=[(X_val, y_val_log)], 
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0)
            ]
        )
        
        # 预测测试集
        lgb_predictions += np.expm1(np.array(lgb_model.predict(X_test))) / 5
        
        # 计算验证分数
        lgb_val_pred = np.expm1(np.array(lgb_model.predict(X_val)))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_val_pred)
        lgb_scores.append(lgb_mae)
        
        # 训练XGBoost - 简化版本，不使用早停
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=200)  # 减少迭代次数避免过拟合
        xgb_model.fit(X_tr, y_tr_log, verbose=False)
        
        # 预测测试集
        xgb_predictions += np.expm1(xgb_model.predict(X_test)) / 5
        
        # 计算验证分数
        xgb_val_pred = np.expm1(xgb_model.predict(X_val))
        xgb_mae = mean_absolute_error(np.expm1(y_val_log), xgb_val_pred)
        xgb_scores.append(xgb_mae)
        
        # 训练CatBoost
        cat_model = CatBoostRegressor(**catboost_params)
        cat_model.fit(X_tr, y_tr_log, 
                     eval_set=[(X_val, y_val_log)], 
                     early_stopping_rounds=50, 
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

def adaptive_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info):
    """
    自适应集成 - 基于验证性能调整权重
    """
    lgb_score = np.mean(scores_info['lgb_scores'])
    xgb_score = np.mean(scores_info['xgb_scores'])
    cat_score = np.mean(scores_info['cat_scores'])
    
    # 根据性能计算权重（性能越好权重越大）
    total_inv_score = 1/lgb_score + 1/xgb_score + 1/cat_score
    lgb_weight = (1/lgb_score) / total_inv_score
    xgb_weight = (1/xgb_score) / total_inv_score
    cat_weight = (1/cat_score) / total_inv_score
    
    print(f"\n自适应集成权重:")
    print(f"  LightGBM: {lgb_weight:.3f}")
    print(f"  XGBoost: {xgb_weight:.3f}")
    print(f"  CatBoost: {cat_weight:.3f}")
    
    ensemble_pred = lgb_weight * lgb_pred + xgb_weight * xgb_pred + cat_weight * cat_pred
    return ensemble_pred

def smart_calibration(predictions, y_train):
    """
    智能校准 - 基于分布的动态调整
    """
    train_mean = y_train.mean()
    pred_mean = predictions.mean()
    
    print(f"\n智能校准:")
    print(f"  训练集均值: {train_mean:.2f}")
    print(f"  预测均值: {pred_mean:.2f}")
    
    # 计算校准因子
    calibration_factor = train_mean / pred_mean if pred_mean > 0 else 1.0
    print(f"  校准因子: {calibration_factor:.4f}")
    
    # 应用校准
    calibrated_predictions = predictions * calibration_factor
    
    # 确保预测值非负
    calibrated_predictions = np.maximum(calibrated_predictions, 0)
    
    return calibrated_predictions

def create_simple_analysis_plots(y_train, predictions, scores_info, model_name="V16"):
    """
    创建简化的分析图表
    """
    print("生成简化分析图表...")
    
    # 创建保存目录
    analysis_dir = get_user_data_path()
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 价格分布对比
    axes[0, 0].hist(y_train, bins=50, alpha=0.7, label='训练集真实价格', color='blue')
    axes[0, 0].hist(predictions, bins=50, alpha=0.7, label='测试集预测价格', color='red')
    axes[0, 0].set_xlabel('价格')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('价格分布对比')
    axes[0, 0].legend()
    
    # 2. 模型性能对比
    models = ['LightGBM', 'XGBoost', 'CatBoost']
    scores = [np.mean(scores_info['lgb_scores']), 
              np.mean(scores_info['xgb_scores']), 
              np.mean(scores_info['cat_scores'])]
    
    axes[0, 1].bar(models, scores, color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('各模型验证性能')
    
    # 3. 预测值分布
    axes[1, 0].hist(predictions, bins=50, alpha=0.7, color='red')
    axes[1, 0].set_xlabel('预测价格')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('预测值分布')
    
    # 4. 统计信息
    stats_text = f"""
    训练集统计:
    均值: {y_train.mean():.2f}
    标准差: {y_train.std():.2f}
    范围: {y_train.min():.2f} - {y_train.max():.2f}
    
    预测集统计:
    均值: {predictions.mean():.2f}
    标准差: {predictions.std():.2f}
    范围: {predictions.min():.2f} - {predictions.max():.2f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='center')
    axes[1, 1].set_title('统计信息')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = os.path.join(analysis_dir, f'{model_name}_simple_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"分析图表已保存到: {chart_path}")
    plt.show()

def v16_optimize_model():
    """
    V16优化模型训练流程 - 修复性能下降问题
    """
    print("=" * 60)
    print("开始V16优化模型训练（修复版本）...")
    print("=" * 60)
    
    # 加载数据
    train_df, test_df = load_and_preprocess_data()
    train_df = create_effective_features(train_df)
    test_df = create_effective_features(test_df)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 特征缩放 - 温和的缩放
    print("\n应用特征缩放...")
    scaler = RobustScaler()
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # 检查并处理数值问题
    for col in numeric_features:
        if col in X_train.columns and col in X_test.columns:
            # 检查无穷大值
            inf_mask = np.isinf(X_train[col]) | np.isinf(X_test[col])
            if inf_mask.any():  # type: ignore[truthy-bool]
                X_train.loc[inf_mask[inf_mask.index.isin(X_train.index)].index, col] = 0
                X_test.loc[inf_mask[inf_mask.index.isin(X_test.index)].index, col] = 0
            
            # 检查NaN值
            X_train[col] = X_train[col].fillna(X_train[col].median())
            X_test[col] = X_test[col].fillna(X_train[col].median())
            
            # 跳过常数列
            if X_train[col].std() > 1e-8:
                X_train[col] = scaler.fit_transform(X_train[[col]])
                X_test[col] = scaler.transform(X_test[[col]])
    
    # 使用早停机制训练模型
    lgb_pred, xgb_pred, cat_pred, scores_info = train_models_with_early_stopping(X_train, y_train, X_test)
    
    # 自适应集成
    ensemble_pred = adaptive_ensemble(lgb_pred, xgb_pred, cat_pred, scores_info)
    
    # 智能校准
    final_predictions = smart_calibration(ensemble_pred, y_train)
    
    # 创建分析图表
    create_simple_analysis_plots(y_train, final_predictions, scores_info, "V16")
    
    print(f"\n最终预测统计:")
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
    result_file = os.path.join(result_dir, f"modeling_v16_result_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\n结果已保存到: {result_file}")
    
    # 生成优化报告
    print("\n" + "=" * 60)
    print("V16优化总结")
    print("=" * 60)
    print("✅ 简化特征工程 - 保留有效特征，避免过度工程")
    print("✅ 早停机制 - 防止过拟合，提高泛化能力")
    print("✅ 温和数值处理 - 保持数据分布，避免过度截断")
    print("✅ 自适应集成 - 基于性能动态调整权重")
    print("✅ 智能校准 - 基于分布的动态调整")
    print("✅ 简化分析 - 专注核心性能指标")
    print("=" * 60)
    
    return final_predictions, scores_info

if __name__ == "__main__":
    test_pred, scores_info = v16_optimize_model()
    print("V16优化完成! 期待线上分数有显著改善。")