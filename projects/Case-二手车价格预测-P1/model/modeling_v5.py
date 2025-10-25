"""
V5版本模型 - 解决V4模型线上MAE异常问题

基于V4分析结果，实施以下优化策略:
1. 简化分段策略，避免过度复杂化
2. 优化特征工程，去除可能引入噪声的特征
3. 调整模型集成权重
4. 改进校准方法
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
    
    # 品牌统计特征（适度平滑）
    if 'price' in all_df.columns:
        brand_stats = all_df.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
        # 适度平滑因子30
        brand_stats['smooth_mean'] = (brand_stats['mean'] * brand_stats['count'] + all_df['price'].mean() * 30) / (brand_stats['count'] + 30)
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

def create_features(df):
    """创建特征（简化版）"""
    df = df.copy()
    
    # 基础分段特征（简化）
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 5, 10, float('inf')], labels=['new', 'medium', 'old'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    if 'power' in df.columns:
        df['power_segment'] = pd.cut(df['power'], bins=[-1, 100, 150, float('inf')], labels=['low', 'medium', 'high'])
        df['power_segment'] = df['power_segment'].cat.codes
    else:
        df['power_segment'] = 0
    
    # 简化的交叉特征
    if 'power' in df.columns and 'kilometer' in df.columns:
        df['power_km_ratio'] = df['power'] / (df['kilometer'] + 1)
    
    # 简化的时间序列特征
    if 'car_age' in df.columns and 'kilometer' in df.columns:
        df['km_per_year'] = df['kilometer'] / (df['car_age'] + 1)
    
    return df

def train_models(X_train, y_train, X_test):
    """
    训练模型（简化版集成）
    """
    print("\n训练模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 简化的模型参数
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'random_state': 42
    }
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    }
    
    catboost_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'depth': 6,
        'learning_rate': 0.1,
        'iterations': 200,
        'l2_leaf_reg': 0.1,
        'random_seed': 42,
        'verbose': False
    }
    
    # 训练模型
    print("训练LightGBM...")
    lgb_model = lgb.LGBMRegressor(**lgb_params) # type: ignore
    lgb_model.fit(X_train, y_train_log)
    
    print("训练XGBoost...")
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train_log)
    
    print("训练CatBoost...")
    cat_model = CatBoostRegressor(**catboost_params)
    cat_model.fit(X_train, y_train_log)
    
    # 预测
    print("生成预测...")
    lgb_pred = np.expm1(np.array(lgb_model.predict(X_test)))
    xgb_pred = np.expm1(xgb_model.predict(X_test))
    cat_pred = np.expm1(cat_model.predict(X_test))
    
    # 简化集成（等权重）
    ensemble_pred = (lgb_pred + xgb_pred + cat_pred) / 3
    
    return ensemble_pred

def calibration(predictions, y_train):
    """
    简化校准方法
    """
    print("\n应用简化校准...")
    
    train_mean = y_train.mean()
    pred_mean = predictions.mean()
    
    print(f"训练集均值: {train_mean:.2f}")
    print(f"预测均值: {pred_mean:.2f}")
    
    # 简化校准（温和调整）
    calibrated_predictions = predictions - 0.2 * (pred_mean - train_mean)
    
    # 确保预测值非负
    calibrated_predictions = np.maximum(calibrated_predictions, 0)
    
    return calibrated_predictions

def v5_optimize_model():
    """
    V5优化模型训练流程
    """
    print("=" * 60)
    print("开始V5优化模型训练...")
    print("=" * 60)
    
    # 加载数据
    train_df, test_df = load_and_preprocess_data()
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
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
    
    # 训练模型
    predictions = train_models(X_train, y_train, X_test)
    
    # 应用简化校准
    final_predictions = calibration(predictions, y_train)
    
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
    result_file = os.path.join(result_dir, f"modeling_v5_result_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\n结果已保存到: {result_file}")
    
    # 生成优化报告
    print("\n" + "=" * 60)
    print("V5优化总结")
    print("=" * 60)
    print("✅ 简化分段策略，避免过度复杂化")
    print("✅ 优化特征工程，去除噪声特征")
    print("✅ 使用等权重集成，避免权重设置不当")
    print("✅ 简化校准方法，避免过度调整")
    print("=" * 60)
    
    return final_predictions

if __name__ == "__main__":
    test_pred = v5_optimize_model()
    print("V5优化完成!请将生成的CSV文件提交到平台验证效果。")