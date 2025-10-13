"""
高级优化模型 - 进一步提升预测性能

基于进一步分析的结果，实施以下优化策略:
1. 分层建模策略 - 针对高价车和低价车分别建模
2. 增强正则化 - 进一步降低模型复杂度
3. 特征工程优化 - 添加更多交叉特征
4. 集成方法改进 - 调整模型权重和添加更多基模型
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
        # 增加平滑因子从10到50,进一步降低对极端值的敏感度
        brand_stats['smooth_mean'] = (brand_stats['mean'] * brand_stats['count'] + all_df['price'].mean() * 50) / (brand_stats['count'] + 50)
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
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 3, 6, 10, float('inf')], labels=['new', 'medium', 'old', 'vintage'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    if 'power' in df.columns:
        df['power_segment'] = pd.cut(df['power'], bins=[-1, 50, 100, 150, 200, float('inf')], labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        df['power_segment'] = df['power_segment'].cat.codes
    else:
        df['power_segment'] = 0
    
    # 添加更多交叉特征
    if 'power' in df.columns and 'kilometer' in df.columns:
        df['power_km_ratio'] = df['power'] / (df['kilometer'] + 1)
        df['power_km_product'] = df['power'] * df['kilometer']
    
    # 添加更多统计特征
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
    
    return df

def train_high_low_models(X_train, y_train, X_test):
    """
    分层建模策略 - 针对高价车和低价车分别训练模型
    """
    print("\n实施分层建模策略...")
    
    # 定义价格阈值
    price_threshold = 10000
    
    # 分离高价车和低价车数据
    high_price_mask = y_train > price_threshold
    low_price_mask = ~high_price_mask
    
    X_train_high = X_train[high_price_mask]
    y_train_high = y_train[high_price_mask]
    X_train_low = X_train[low_price_mask]
    y_train_low = y_train[low_price_mask]
    
    print(f"高价车样本数: {len(X_train_high)} ({len(X_train_high)/len(X_train)*100:.1f}%)")
    print(f"低价车样本数: {len(X_train_low)} ({len(X_train_low)/len(X_train)*100:.1f}%)")
    
    # 对数变换
    y_train_high_log = np.log1p(y_train_high)
    y_train_low_log = np.log1p(y_train_low)
    
    # 定义更保守的模型参数
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 15,          # 进一步降低复杂度
        'max_depth': 4,            # 进一步降低深度
        'min_data_in_leaf': 200,   # 增加最小样本数
        'feature_fraction': 0.6,   # 降低特征采样
        'bagging_fraction': 0.6,   # 降低数据采样
        'bagging_freq': 5,
        'lambda_l1': 1.0,          # 增加L1正则化
        'lambda_l2': 1.0,          # 增加L2正则化
        'verbose': -1,
        'random_state': 42
    }
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'random_state': 42
    }
    
    # 训练高价车模型
    print("训练高价车模型...")
    lgb_high = lgb.LGBMRegressor(**lgb_params, n_estimators=150)
    xgb_high = xgb.XGBRegressor(**xgb_params, n_estimators=150)
    
    lgb_high.fit(X_train_high, y_train_high_log)
    xgb_high.fit(X_train_high, y_train_high_log)
    
    # 训练低价车模型
    print("训练低价车模型...")
    lgb_low = lgb.LGBMRegressor(**lgb_params, n_estimators=150)
    xgb_low = xgb.XGBRegressor(**xgb_params, n_estimators=150)
    
    lgb_low.fit(X_train_low, y_train_low_log)
    xgb_low.fit(X_train_low, y_train_low_log)
    
    # 预测
    # 对于测试集，我们需要先预测价格，然后根据预测价格决定使用哪个模型
    # 这里我们使用一个简单的方法：先用整体模型预测，然后根据阈值选择模型
    
    # 训练整体模型用于初步分类
    print("训练整体模型用于初步分类...")
    lgb_overall = lgb.LGBMRegressor(**lgb_params, n_estimators=150)
    lgb_overall.fit(X_train, np.log1p(y_train))
    
    # 初步预测
    pred_overall = np.expm1(np.array(lgb_overall.predict(X_test)))
    
    # 根据初步预测选择模型
    high_mask = pred_overall > price_threshold
    low_mask = ~high_mask
    
    print(f"测试集中预测为高价车的样本数: {sum(high_mask)}")
    print(f"测试集中预测为低价车的样本数: {sum(low_mask)}")
    
    # 分别预测
    pred_high = np.zeros(len(X_test))
    pred_low = np.zeros(len(X_test))
    
    if sum(high_mask) > 0:
        pred_high[high_mask] = 0.5 * np.expm1(np.array(lgb_high.predict(X_test[high_mask]))) + \
                              0.5 * np.expm1(xgb_high.predict(X_test[high_mask]))
    
    if sum(low_mask) > 0:
        pred_low[low_mask] = 0.5 * np.expm1(np.array(lgb_low.predict(X_test[low_mask]))) + \
                            0.5 * np.expm1(xgb_low.predict(X_test[low_mask]))
    
    # 合并预测结果
    final_pred = pred_high + pred_low
    
    return final_pred

def advanced_optimize_model():
    """
    高级优化模型训练流程
    """
    print("=" * 60)
    print("开始高级优化模型训练...")
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
    
    # 使用分层建模策略
    final_pred_test = train_high_low_models(X_train, y_train, X_test)
    
    # 预测值校准 - 增强校准
    train_mean = y_train.mean()
    pred_mean = final_pred_test.mean()
    
    print(f"\n预测值统计:")
    print(f"训练集 - 均值: {train_mean:.2f}")
    print(f"预测前 - 均值: {pred_mean:.2f}")
    
    # 增强校准(更强的校准强度)
    calibration_factor = 0.5  # 校准强度50%
    final_pred_test_calibrated = final_pred_test - calibration_factor * (pred_mean - train_mean)
    
    print(f"预测后 - 均值: {final_pred_test_calibrated.mean():.2f}")
    
    # 确保预测值非负
    final_pred_test_calibrated = np.maximum(final_pred_test_calibrated, 0)
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'SaleID': test_df['SaleID'] if 'SaleID' in test_df.columns else test_df.index,
        'price': final_pred_test_calibrated
    })
    
    # 保存结果
    result_dir = get_project_path('prediction_result')
    os.makedirs(result_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"modeling_v3_result_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\n结果已保存到: {result_file}")
    
    # 生成优化报告
    print("\n" + "=" * 60)
    print("高级优化总结")
    print("=" * 60)
    print("✅ 实施分层建模策略，针对高价车和低价车分别训练模型")
    print("✅ 进一步降低模型复杂度，增强正则化")
    print("✅ 添加更多交叉特征和统计特征")
    print("✅ 使用更强大的校准方法")
    print("✅ 集成LightGBM和XGBoost模型")
    print("=" * 60)
    
    return final_pred_test_calibrated

if __name__ == "__main__":
    test_pred = advanced_optimize_model()
    print("高级优化完成!请将生成的CSV文件提交到平台验证效果。")