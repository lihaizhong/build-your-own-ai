"""
V4版本模型 - 进一步提升预测性能

基于V3分析结果，实施以下优化策略:
1. 更精细的价格分段策略
2. 增强特征工程 - 添加更多交叉特征和时间序列特征
3. 模型集成优化 - 调整权重并添加CatBoost
4. 高级校准技术
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
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

def get_user_data_path(*paths):
    """获取用户数据路径"""
    return get_project_path('user_data', *paths)

def load_and_preprocess_data():
    """
    加载并预处理数据
    """
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 合并数据进行统一预处理
    all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    # 处理power异常值
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

    # 特征工程
    all_df['regDate'] = pd.to_datetime(all_df['regDate'], format='%Y%m%d', errors='coerce')
    current_year = 2020
    all_df['car_age'] = current_year - all_df['regDate'].dt.year
    all_df['car_age'] = all_df['car_age'].fillna(0).astype(int)
    all_df.drop(columns=['regDate'], inplace=True)
    
    if 'power' in all_df.columns:
        all_df['power_age_ratio'] = all_df['power'] / (all_df['car_age'] + 1)
    else:
        all_df['power_age_ratio'] = 0
    
    # 品牌统计特征（使用更大的平滑因子增强鲁棒性）
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
        # 增加平滑因子从10到80,进一步降低对极端值的敏感度
        brand_stats['smooth_mean'] = (brand_stats['mean'] * brand_stats['count'] + all_df['price'].mean() * 80) / (brand_stats['count'] + 80)
        brand_map: dict = brand_stats.set_index('brand')['smooth_mean'].to_dict()  # type: ignore
        all_df['brand_avg_price'] = all_df['brand'].map(brand_map)  # type: ignore
    
    # 标签编码
    categorical_cols = ['model', 'brand', 'fuelType', 'gearbox', 'bodyType']
    for col in categorical_cols:
        if col in all_df.columns:
            le = LabelEncoder()
            all_df[col] = le.fit_transform(all_df[col].astype(str))
    
    # 填充数值型缺失值
    numeric_cols = all_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        has_null = bool(all_df[col].isnull().any())  # type: ignore[arg-type]
        if has_null:
            median_value = all_df[col].median()
            all_df[col] = all_df[col].fillna(median_value)
    
    # 重新分离
    train_df = all_df.iloc[:len(train_df)].copy()
    test_df = all_df.iloc[len(train_df):].copy()
    
    print(f"优化版训练集: {train_df.shape}")
    print(f"训练集价格统计: 均值={train_df['price'].mean():.2f}, 中位数={train_df['price'].median():.2f}, 标准差={train_df['price'].std():.2f}")
    
    return train_df, test_df

def create_advanced_features(df):
    """创建高级特征"""
    df = df.copy()
    
    # 基础分段特征
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 2, 5, 8, 12, float('inf')], labels=['very_new', 'new', 'medium', 'old', 'very_old'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    if 'power' in df.columns:
        df['power_segment'] = pd.cut(df['power'], bins=[-1, 40, 80, 120, 160, 200, float('inf')], labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme_high'])
        df['power_segment'] = df['power_segment'].cat.codes
    else:
        df['power_segment'] = 0
    
    # 添加更多交叉特征
    if 'power' in df.columns and 'kilometer' in df.columns:
        df['power_km_ratio'] = df['power'] / (df['kilometer'] + 1)
        df['power_km_product'] = df['power'] * df['kilometer']
        df['power_div_km'] = df['power'] / (df['kilometer'] / 10000 + 1)  # 调整比例
    
    # 添加更多统计特征
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
        df['v_skew'] = df[v_cols].skew(axis=1)
    
    # 时间序列特征
    if 'car_age' in df.columns and 'kilometer' in df.columns:
        df['km_per_year'] = df['kilometer'] / (df['car_age'] + 1)
        df['log_kilometer'] = np.log1p(df['kilometer'])
        df['log_car_age'] = np.log1p(df['car_age'])
    
    return df

def train_multi_segment_models(X_train, y_train, X_test):
    """
    多分段建模策略 - 针对不同价格区间分别训练模型
    """
    print("\n实施多分段建模策略...")
    
    # 定义多个价格阈值
    segments = [
        ('very_low', 0, 5000),
        ('low', 5000, 10000),
        ('medium', 10000, 20000),
        ('high', 20000, 35000),
        ('very_high', 35000, float('inf'))
    ]
    
    models = {}
    predictions = np.zeros(len(X_test))
    
    # 对每个价格区间分别训练模型
    for segment_name, min_price, max_price in segments:
        # 分离数据
        segment_mask = (y_train >= min_price) & (y_train < max_price)
        X_segment = X_train[segment_mask]
        y_segment = y_train[segment_mask]
        
        print(f"{segment_name}价格区间 [{min_price}, {max_price}): {len(X_segment)} 样本")
        
        if len(X_segment) == 0:
            continue
            
        # 对数变换
        y_segment_log = np.log1p(y_segment)
        
        # 定义模型参数
        lgb_params = {
            'objective': 'mae',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 12,          # 进一步降低复杂度
            'max_depth': 3,            # 进一步降低深度
            'min_data_in_leaf': 250,   # 增加最小样本数
            'feature_fraction': 0.5,   # 降低特征采样
            'bagging_fraction': 0.5,   # 降低数据采样
            'bagging_freq': 5,
            'lambda_l1': 1.5,          # 增加L1正则化
            'lambda_l2': 1.5,          # 增加L2正则化
            'verbose': -1,
            'random_state': 42
        }
        
        xgb_params = {
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'max_depth': 3,
            'learning_rate': 0.03,
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'reg_alpha': 1.5,
            'reg_lambda': 1.5,
            'random_state': 42
        }
        
        catboost_params = {
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'depth': 3,
            'learning_rate': 0.03,
            'iterations': 150,
            'l2_leaf_reg': 1.5,
            'random_seed': 42,
            'verbose': False
        }
        
        # 训练模型
        print(f"训练{segment_name}价格区间模型...")
        lgb_model = lgb.LGBMRegressor(**lgb_params) # type: ignore
        xgb_model = xgb.XGBRegressor(**xgb_params)
        cat_model = CatBoostRegressor(**catboost_params)
        
        lgb_model.fit(X_segment, y_segment_log)
        xgb_model.fit(X_segment, y_segment_log)
        cat_model.fit(X_segment, y_segment_log)
        
        # 保存模型
        models[segment_name] = {
            'lgb': lgb_model,
            'xgb': xgb_model,
            'cat': cat_model
        }
        
        # 预测测试集（这里简化处理，实际应该使用更复杂的分类方法）
        # 在实际应用中，我们会先预测价格区间，然后使用对应区间的模型
    
    # 训练整体模型用于初步分类
    print("训练整体模型用于初步分类...")
    lgb_overall = lgb.LGBMRegressor(
        objective='mae',
        metric='mae',
        boosting_type='gbdt',
        num_leaves=15,
        max_depth=4,
        min_data_in_leaf=200,
        feature_fraction=0.6,
        bagging_fraction=0.6,
        bagging_freq=5,
        lambda_l1=1.0,
        lambda_l2=1.0,
        verbose=-1,
        random_state=42,
        n_estimators=150
    )
    lgb_overall.fit(X_train, np.log1p(y_train))
    
    # 初步预测
    pred_overall = np.expm1(np.array(lgb_overall.predict(X_test)))
    
    # 根据初步预测选择模型并进行预测
    for segment_name, min_price, max_price in segments:
        if segment_name in models:
            segment_mask = (pred_overall >= min_price) & (pred_overall < max_price)
            if np.sum(segment_mask) > 0:
                # 使用集成预测
                lgb_pred = np.expm1(models[segment_name]['lgb'].predict(X_test[segment_mask]))
                xgb_pred = np.expm1(models[segment_name]['xgb'].predict(X_test[segment_mask]))
                cat_pred = np.expm1(models[segment_name]['cat'].predict(X_test[segment_mask]))
                
                # 加权平均 (可以根据验证集表现调整权重)
                ensemble_pred = 0.4 * lgb_pred + 0.3 * xgb_pred + 0.3 * cat_pred
                predictions[segment_mask] = ensemble_pred
    
    return predictions

def advanced_calibration(predictions, y_train):
    """
    高级校准技术
    """
    print("\n应用高级校准技术...")
    
    # 分位数校准
    train_mean = y_train.mean()
    pred_mean = predictions.mean()
    
    print(f"训练集均值: {train_mean:.2f}")
    print(f"预测均值: {pred_mean:.2f}")
    
    # 基础校准
    calibrated_predictions = predictions - 0.6 * (pred_mean - train_mean)
    
    # 分段校准
    segments = [
        ('very_low', 0, 5000),
        ('low', 5000, 10000),
        ('medium', 10000, 20000),
        ('high', 20000, 35000),
        ('very_high', 35000, float('inf'))
    ]
    
    for segment_name, min_price, max_price in segments:
        segment_mask = (calibrated_predictions >= min_price) & (calibrated_predictions < max_price)
        if np.sum(segment_mask) > 0:
            segment_mean = calibrated_predictions[segment_mask].mean()
            train_segment_mean = y_train[(y_train >= min_price) & (y_train < max_price)].mean() if len(y_train[(y_train >= min_price) & (y_train < max_price)]) > 0 else segment_mean
            # 分段校准
            calibrated_predictions[segment_mask] = calibrated_predictions[segment_mask] - 0.3 * (segment_mean - train_segment_mean)
    
    # 确保预测值非负
    calibrated_predictions = np.maximum(calibrated_predictions, 0)
    
    return calibrated_predictions

def v4_optimize_model():
    """
    V4优化模型训练流程
    """
    print("=" * 60)
    print("开始V4优化模型训练...")
    print("=" * 60)
    
    # 加载数据
    train_df, test_df = load_and_preprocess_data()
    train_df = create_advanced_features(train_df)
    test_df = create_advanced_features(test_df)
    
    # 准备特征
    y_col = 'price'
    feature_cols = [c for c in train_df.columns if c not in [y_col, 'SaleID']]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[y_col].copy()
    X_test = test_df[feature_cols].copy()
    
    # 特征缩放
    print("\n应用特征缩放...")
    scaler = RobustScaler()
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # 使用多分段建模策略
    predictions = train_multi_segment_models(X_train, y_train, X_test)
    
    # 应用高级校准技术
    final_predictions = advanced_calibration(predictions, y_train)
    
    print(f"\n最终预测统计:")
    print(f"均值: {final_predictions.mean():.2f}")
    print(f"标准差: {final_predictions.std():.2f}")
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'SaleID': test_df['SaleID'] if 'SaleID' in test_df.columns else test_df.index,
        'price': final_predictions
    })
    
    # 保存结果
    result_dir = get_project_path('prediction_result')
    os.makedirs(result_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"modeling_v4_result_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\n结果已保存到: {result_file}")
    
    # 生成优化报告
    print("\n" + "=" * 60)
    print("V4优化总结")
    print("=" * 60)
    print("✅ 实施多分段建模策略，针对不同价格区间分别训练模型")
    print("✅ 进一步降低模型复杂度，增强正则化")
    print("✅ 添加更多交叉特征和时间序列特征")
    print("✅ 使用更强大的校准方法")
    print("✅ 集成LightGBM、XGBoost和CatBoost模型")
    print("=" * 60)
    
    return final_predictions

if __name__ == "__main__":
    test_pred = v4_optimize_model()
    print("V4优化完成!请将生成的CSV文件提交到平台验证效果。")