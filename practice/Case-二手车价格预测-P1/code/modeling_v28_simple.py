"""
V28极简测试版本 - 核心策略验证

最简化版本，只验证核心创新策略：
1. 基础特征工程
2. 简单模型训练
3. 快速结果验证
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
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


def simple_test():
    """极简测试"""
    print("开始V28极简测试...")
    
    # 加载数据
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')
    
    train_df = pd.read_csv(train_path, sep=' ', na_values=['-'])
    test_df = pd.read_csv(test_path, sep=' ', na_values=['-'])
    
    print(f"训练集: {train_df.shape}")
    print(f"测试集: {test_df.shape}")
    
    # 基础预处理
    # 处理power
    train_df['power'] = np.clip(train_df['power'], 0, 600)
    test_df['power'] = np.clip(test_df['power'], 0, 600)
    
    # 处理时间
    def process_date(df):
        df['regDate'] = pd.to_datetime(df['regDate'], format='%Y%m%d', errors='coerce')
        df['car_age'] = 2020 - df['regDate'].dt.year
        df['car_age'] = df['car_age'].fillna(0)
        df = df.drop('regDate', axis=1)
        return df
    
    train_df = process_date(train_df)
    test_df = process_date(test_df)
    
    # 基础特征工程
    def create_features(df):
        df = df.copy()
        
        # 基础特征
        if 'power' in df.columns and 'car_age' in df.columns:
            df['power_age_ratio'] = df['power'] / (df['car_age'] + 1)
        
        if 'kilometer' in df.columns and 'car_age' in df.columns:
            df['km_per_year'] = df['kilometer'] / (df['car_age'] + 0.1)
        
        # 处理分类特征
        categorical_cols = ['fuelType', 'gearbox', 'bodyType', 'model', 'brand']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # 填充数值特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['price', 'SaleID']:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # 准备特征
    feature_cols = [c for c in train_df.columns if c not in ['price', 'SaleID']]
    X_train = train_df[feature_cols]
    y_train = train_df['price']
    X_test = test_df[feature_cols]
    
    print(f"特征数量: {len(feature_cols)}")
    
    # 简单验证
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 训练简单模型
    print("训练LightGBM模型...")
    model = lgb.LGBMRegressor(
        objective='mae',
        num_leaves=31,
        learning_rate=0.08,
        n_estimators=1000,
        random_state=42
    )
    
    model.fit(X_tr, np.log1p(y_tr))
    val_pred = np.expm1(model.predict(X_val))
    val_mae = mean_absolute_error(y_val, val_pred)
    
    print(f"验证MAE: {val_mae:.2f}")
    
    # 预测测试集
    test_pred = np.expm1(model.predict(X_test))
    
    # 简单校准
    train_mean = y_train.mean()
    pred_mean = test_pred.mean()
    calibration_factor = train_mean / pred_mean
    test_pred_calibrated = test_pred * calibration_factor
    
    print(f"校准因子: {calibration_factor:.4f}")
    print(f"预测均值: {test_pred_calibrated.mean():.2f}")
    
    # 保存结果
    submission_df = pd.DataFrame({
        'SaleID': test_df['SaleID'],
        'price': test_pred_calibrated
    })
    
    result_dir = get_project_path('prediction_result')
    os.makedirs(result_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(result_dir, f"modeling_v28_simple_{timestamp}.csv")
    submission_df.to_csv(result_file, index=False)
    
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
        save_models(models_to_save, 'v28_simple')

    
    print(f"结果已保存到: {result_file}")
    print("V28极简测试完成!")
    
    return val_mae

if __name__ == "__main__":
    mae = simple_test()
    print(f"验证MAE: {mae:.2f}")