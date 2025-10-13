"""
V11版本模型 - 平衡正则化和模型复杂度

基于V10分析结果，实施以下优化策略:
1. 平衡正则化参数，避免过度正则化
2. 优化模型复杂度，在过拟合和欠拟合之间找到平衡
3. 改进特征工程，保留有效特征
4. 优化集成策略，平衡各模型权重
5. 精细化数据增强，避免引入噪声
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

def augment_data_balanced(train_df, y_train, n_augment=1):
    """
    精细化数据增强技术
    通过添加适度噪声来增加训练数据的多样性，避免引入过多噪声
    """
    print(f"原始训练集大小: {train_df.shape}")
    
    # 创建增强数据
    augmented_features = []
    augmented_targets = []
    
    # 添加原始数据
    augmented_features.append(train_df.copy())
    augmented_targets.append(y_train.copy())
    
    # 为数值特征添加适度噪声
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    for i in range(n_augment):
        # 复制原始数据
        augmented_df = train_df.copy()
        augmented_y = y_train.copy()
        
        # 为数值特征添加适度噪声（标准差的0.5%）
        for col in numeric_cols:
            noise = np.random.normal(0, 0.005 * augmented_df[col].std(), size=len(augmented_df))
            augmented_df[col] = augmented_df[col] + noise
        
        # 为目标变量添加适度噪声（标准差的0.5%）
        y_noise = np.random.normal(0, 0.005 * augmented_y.std(), size=len(augmented_y))
        augmented_y = augmented_y + y_noise
        augmented_y = np.maximum(augmented_y, 0)  # 确保价格非负
        
        augmented_features.append(augmented_df)
        augmented_targets.append(augmented_y)
    
    # 合并所有数据
    final_features = pd.concat(augmented_features, ignore_index=True)
    final_targets = np.concatenate(augmented_targets)
    
    print(f"增强后训练集大小: {final_features.shape}")
    return final_features, final_targets

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
        # 适度平滑因子40
        brand_stats['smooth_mean'] = (brand_stats['mean'] * brand_stats['count'] + all_df['price'].mean() * 40) / (brand_stats['count'] + 40)
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

def create_balanced_features(df):
    """创建平衡的特征工程"""
    df = df.copy()
    
    # 基础分段特征
    df['age_segment'] = pd.cut(df['car_age'], bins=[-1, 3, 6, 10, float('inf')], labels=['very_new', 'new', 'medium', 'old'])
    df['age_segment'] = df['age_segment'].cat.codes
    
    # 功率分段特征
    if 'power' in df.columns:
        df['power_segment'] = pd.cut(df['power'], bins=[-1, 50, 100, 150, 200, float('inf')], labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        df['power_segment'] = df['power_segment'].cat.codes
    else:
        df['power_segment'] = 0
    
    # 交叉特征
    if 'power' in df.columns and 'kilometer' in df.columns:
        df['power_km_ratio'] = df['power'] / (df['kilometer'] + 1)
        df['power_km_product'] = df['power'] * df['kilometer']
    
    # 统计特征
    v_cols = [col for col in df.columns if col.startswith('v_')]
    if len(v_cols) >= 3:
        df['v_mean'] = df[v_cols].mean(axis=1)
        df['v_std'] = df[v_cols].std(axis=1)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_min'] = df[v_cols].min(axis=1)
    
    # 时间序列特征
    if 'car_age' in df.columns and 'kilometer' in df.columns:
        df['km_per_year'] = df['kilometer'] / (df['car_age'] + 1)
        df['log_kilometer'] = np.log1p(df['kilometer'])
        df['log_car_age'] = np.log1p(df['car_age'])
    
    return df

def train_models_with_cv(X_train, y_train, X_test):
    """
    使用交叉验证训练模型，平衡正则化和模型复杂度
    """
    print("\n使用交叉验证训练模型...")
    
    # 对数变换
    y_train_log = np.log1p(y_train)
    
    # 5折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 模型参数 - 平衡正则化和模型复杂度
    lgb_params = {
        'objective': 'mae',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 10,  # 平衡复杂度
        'max_depth': 3,    # 平衡深度
        'min_data_in_leaf': 400,  # 平衡最小样本数
        'feature_fraction': 0.6,  # 平衡特征采样
        'bagging_fraction': 0.6,  # 平衡数据采样
        'bagging_freq': 5,
        'lambda_l1': 3.0,  # 平衡L1正则化
        'lambda_l2': 3.0,  # 平衡L2正则化
        'min_gain_to_split': 0.3,  # 平衡分裂最小增益
        'verbose': -1,
        'random_state': 42
    }
    
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': 3,
        'learning_rate': 0.02,  # 平衡学习率
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 3.0,
        'reg_lambda': 3.0,
        'min_child_weight': 50,  # 平衡子节点最小权重
        'random_state': 42
    }
    
    catboost_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'depth': 3,
        'learning_rate': 0.02,  # 平衡学习率
        'iterations': 60,  # 平衡迭代次数
        'l2_leaf_reg': 3.0,
        'random_strength': 1.5,  # 平衡随机强度
        'random_seed': 42,
        'verbose': False
    }
    
    # 存储各模型的预测结果
    lgb_predictions = np.zeros(len(X_test))
    xgb_predictions = np.zeros(len(X_test))
    cat_predictions = np.zeros(len(X_test))
    
    # 交叉验证训练
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        print(f"训练第 {fold} 折...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_log, y_val_log = y_train_log[train_idx], y_train_log[val_idx]
        
        # 训练LightGBM
        lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=60)
        lgb_model.fit(X_tr, y_tr_log)
        lgb_pred_log = lgb_model.predict(X_val)
        lgb_pred = np.expm1(np.array(lgb_pred_log))
        lgb_mae = mean_absolute_error(np.expm1(y_val_log), lgb_pred)
        
        # 训练XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params, n_estimators=60)
        xgb_model.fit(X_tr, y_tr_log)
        xgb_pred_log = xgb_model.predict(X_val)
        xgb_pred = np.expm1(xgb_pred_log)
        xgb_mae = mean_absolute_error(np.expm1(y_val_log), xgb_pred)
        
        # 训练CatBoost
        cat_model = CatBoostRegressor(**catboost_params)
        cat_model.fit(X_tr, y_tr_log, verbose=False)
        cat_pred_log = cat_model.predict(X_val)
        cat_pred = np.expm1(cat_pred_log)
        cat_mae = mean_absolute_error(np.expm1(y_val_log), cat_pred)
        
        print(f"  LightGBM MAE: {lgb_mae:.2f}")
        print(f"  XGBoost MAE: {xgb_mae:.2f}")
        print(f"  CatBoost MAE: {cat_mae:.2f}")
        
        cv_scores.append((lgb_mae, xgb_mae, cat_mae))
        
        # 累加测试集预测结果
        lgb_predictions += np.expm1(np.array(lgb_model.predict(X_test)))
        xgb_predictions += np.expm1(xgb_model.predict(X_test))
        cat_predictions += np.expm1(cat_model.predict(X_test))
    
    # 计算平均CV分数
    avg_lgb_mae = np.mean([score[0] for score in cv_scores])
    avg_xgb_mae = np.mean([score[1] for score in cv_scores])
    avg_cat_mae = np.mean([score[2] for score in cv_scores])
    
    print(f"\n交叉验证平均MAE:")
    print(f"  LightGBM: {avg_lgb_mae:.2f}")
    print(f"  XGBoost: {avg_xgb_mae:.2f}")
    print(f"  CatBoost: {avg_cat_mae:.2f}")
    
    # 计算平均预测结果
    lgb_predictions /= 5
    xgb_predictions /= 5
    cat_predictions /= 5
    
    return lgb_predictions, xgb_predictions, cat_predictions

def advanced_ensemble(lgb_pred, xgb_pred, cat_pred, y_train):
    """
    高级集成方法
    平衡各模型权重
    """
    print("\n应用高级集成方法...")
    
    # 平衡各模型权重
    # LightGBM表现最好，给予较高权重
    # XGBoost表现中等，给予中等权重
    # CatBoost表现较差，给予较低权重
    ensemble_pred = (0.65 * lgb_pred + 0.25 * xgb_pred + 0.1 * cat_pred)
    
    return ensemble_pred

def advanced_calibration(predictions, y_train):
    """
    高级校准方法
    """
    print("\n应用高级校准方法...")
    
    train_mean = y_train.mean()
    pred_mean = predictions.mean()
    
    print(f"训练集均值: {train_mean:.2f}")
    print(f"预测均值: {pred_mean:.2f}")
    
    # 计算校准因子
    calibration_factor = train_mean / pred_mean if pred_mean > 0 else 1.0
    print(f"校准因子: {calibration_factor:.4f}")
    
    # 应用校准
    calibrated_predictions = predictions * calibration_factor
    
    # 确保预测值非负
    calibrated_predictions = np.maximum(calibrated_predictions, 0)
    
    return calibrated_predictions

def v11_optimize_model():
    """
    V11优化模型训练流程
    """
    print("=" * 60)
    print("开始V11优化模型训练...")
    print("=" * 60)
    
    # 加载数据
    train_df, test_df = load_and_preprocess_data()
    train_df = create_balanced_features(train_df)
    test_df = create_balanced_features(test_df)
    
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
    
    # 精细化数据增强
    print("\n应用精细化数据增强...")
    X_train_aug, y_train_aug = augment_data_balanced(X_train, y_train, n_augment=1)
    
    # 使用交叉验证训练模型
    lgb_pred, xgb_pred, cat_pred = train_models_with_cv(X_train_aug, y_train_aug, X_test)
    
    # 高级集成
    ensemble_pred = advanced_ensemble(lgb_pred, xgb_pred, cat_pred, y_train)
    
    # 高级校准
    final_predictions = advanced_calibration(ensemble_pred, y_train)
    
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
    result_file = os.path.join(result_dir, f"modeling_v11_result_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    print(f"\n结果已保存到: {result_file}")
    
    # 生成优化报告
    print("\n" + "=" * 60)
    print("V11优化总结")
    print("=" * 60)
    print("✅ 平衡正则化参数，避免过度正则化")
    print("✅ 优化模型复杂度，在过拟合和欠拟合之间找到平衡")
    print("✅ 改进特征工程，保留有效特征")
    print("✅ 优化集成策略，平衡各模型权重")
    print("✅ 精细化数据增强，避免引入噪声")
    print("=" * 60)
    
    return final_predictions

if __name__ == "__main__":
    test_pred = v11_optimize_model()
    print("V11优化完成!请将生成的CSV文件提交到平台验证效果。")