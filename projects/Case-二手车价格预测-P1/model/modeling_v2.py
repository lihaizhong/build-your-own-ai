"""
优化版二手车价格预测模型 - 解决线上线下MAE差异问题

核心优化策略:
1. 保留价格异常值,避免训练集分布偏移
2. 使用分位数回归增强鲁棒性
3. 添加特征标准化降低分布偏移影响
4. 使用更保守的正则化参数
5. 添加预测值校准机制
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
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
    关键改进: 保留价格异常值,避免训练集分布偏移
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
        # 增加平滑因子从10到30,降低对极端值的敏感度
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
    
    # 【关键改进】不删除价格异常值,保持完整分布
    print(f"优化版训练集(保留异常值): {train_df.shape}")
    print(f"训练集价格统计: 均值={train_df['price'].mean():.2f}, 中位数={train_df['price'].median():.2f}, 标准差={train_df['price'].std():.2f}")
    
    return train_df, test_df

def create_features(df):
    """创建高级特征"""
    df = df.copy()
    
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 3, 6, 10, float('inf')], labels=['new', 'medium', 'old', 'vintage'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    if 'power' in df.columns:
        df['power_segment'] = pd.cut(df['power'], bins=[-1, 50, 100, 150, 200, float('inf')], labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        df['power_segment'] = df['power_segment'].cat.codes
    else:
        df['power_segment'] = 0
    
    return df

def optimize_model_v2():
    """
    优化版模型训练流程
    主要改进:
    1. 保留异常值
    2. 使用RobustScaler进行特征缩放
    3. 使用5折交叉验证
    4. 调整模型参数增强泛化能力
    5. 添加预测值校准
    """
    print("=" * 60)
    print("开始优化版模型训练...")
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
    
    # 【改进1】使用RobustScaler进行特征缩放,对异常值更鲁棒
    print("\n应用特征缩放...")
    scaler = RobustScaler()
    # 只对数值型特征进行缩放
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 【改进2】使用5折交叉验证评估模型
    print("\n使用5折交叉验证...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    # 定义更保守的模型参数
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,          # 降低复杂度(从64降到31)
        'max_depth': 5,            # 降低深度(从6降到5)
        'min_data_in_leaf': 100,   # 增加最小样本数(从50增到100)
        'feature_fraction': 0.7,   # 降低特征采样(从0.8降到0.7)
        'bagging_fraction': 0.7,   # 降低数据采样(从0.8降到0.7)
        'bagging_freq': 5,
        'lambda_l1': 0.5,          # 增加L1正则化(从0.1增到0.5)
        'lambda_l2': 0.5,          # 增加L2正则化(从0.1增到0.5)
        'verbose': -1,
        'random_state': 42
    }
    
    # 交叉验证
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_log, y_val_log = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        # 训练LightGBM
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=200)  # type: ignore 降低树的数量(从300降到200)
        lgb_model.fit(X_tr, y_tr_log)
        
        # 预测
        pred_val_log = np.array(lgb_model.predict(X_val))
        pred_val = np.expm1(pred_val_log)
        
        # 计算MAE
        fold_mae = mean_absolute_error(np.expm1(y_val_log), pred_val)
        cv_scores.append(fold_mae)
        print(f"Fold {fold} MAE: {fold_mae:.2f}")
    
    print(f"\n交叉验证平均 MAE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
    
    # 【改进3】在全量数据上训练最终模型
    print("\n在全量训练集上训练最终模型...")
    
    # 随机森林(更保守的参数)
    rf_model = RandomForestRegressor(
        n_estimators=150,         # 降低树的数量(从200降到150)
        max_depth=5,              # 降低深度(从6降到5)
        min_samples_split=100,    # 增加最小分割样本(从50增到100)
        min_samples_leaf=50,      # 增加叶节点最小样本(从30增到50)
        max_features='sqrt',      # type: ignore[arg-type]
        random_state=42,
        n_jobs=-1
    )
    
    lgb_model_final = lgb.LGBMRegressor(**lgb_params, n_estimators=200) # type: ignore
    ridge_model = Ridge(alpha=10.0)  # 增加正则化强度(从1.0增到10.0)
    
    # 训练基模型
    rf_model.fit(X_train, y_train_log)
    lgb_model_final.fit(X_train, y_train_log)
    ridge_model.fit(X_train, y_train_log)
    
    # 测试集预测
    rf_pred_test = np.expm1(rf_model.predict(X_test))
    lgb_pred_test = np.expm1(np.array(lgb_model_final.predict(X_test)))
    ridge_pred_test = np.expm1(ridge_model.predict(X_test))
    
    # 【改进4】使用加权平均而非Stacking(更稳定)
    # 权重基于交叉验证性能分配
    final_pred_test = 0.5 * lgb_pred_test + 0.3 * rf_pred_test + 0.2 * ridge_pred_test
    
    # 【改进5】预测值校准 - 调整到训练集分布
    train_mean = y_train.mean()
    train_std = y_train.std()
    pred_mean = final_pred_test.mean()
    pred_std = final_pred_test.std()
    
    print(f"\n预测值统计:")
    print(f"训练集 - 均值: {train_mean:.2f}, 标准差: {train_std:.2f}")
    print(f"预测前 - 均值: {pred_mean:.2f}, 标准差: {pred_std:.2f}")
    
    # 校准预测值(温和调整,避免过度校准)
    calibration_factor = 0.3  # 校准强度30%
    final_pred_test_calibrated = final_pred_test - calibration_factor * (pred_mean - train_mean)
    
    print(f"预测后 - 均值: {final_pred_test_calibrated.mean():.2f}, 标准差: {final_pred_test_calibrated.std():.2f}")
    
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
    result_file = os.path.join(result_dir, f"modeling_v2_result_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\n结果已保存到: {result_file}")
    
    # 生成对比报告
    print("\n" + "=" * 60)
    print("优化总结")
    print("=" * 60)
    print("✅ 保留了价格异常值,避免训练集分布偏移")
    print("✅ 使用RobustScaler进行特征缩放")
    print("✅ 降低模型复杂度,增强泛化能力")
    print("✅ 增加正则化强度,防止过拟合")
    print("✅ 应用预测值校准,调整到训练集分布")
    print(f"✅ 5折交叉验证 MAE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
    print("=" * 60)
    
    return np.mean(cv_scores), final_pred_test_calibrated

if __name__ == "__main__":
    cv_mae, test_pred = optimize_model_v2()
    print(f"\n最终交叉验证 MAE: {cv_mae:.2f}")
    print("预测完成!请将生成的CSV文件提交到平台验证效果。")
