import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建数据文件的相对路径
data_dir = os.path.join(current_dir, '..', 'processed_data')
pred_dir = os.path.join(current_dir, '..', 'prediction_result')
user_data_dir = os.path.join(current_dir, '..', 'user_data')

# 加载处理好的数据
print("Loading processed data...")
X_train = joblib.load(os.path.join(data_dir, 'X_train.joblib'))
y_train = joblib.load(os.path.join(data_dir, 'y_train.joblib'))
X_test = joblib.load(os.path.join(data_dir, 'test_data.joblib'))

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 简化版LightGBM参数
lgb_params = {
    'objective': 'mae',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 6,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'min_gain_to_split': 0.0,
    'verbose': -1,
    'random_state': 42
}

# 3折交叉验证（减少计算时间）
kf = KFold(n_splits=3, shuffle=True, random_state=42)
lgb_oof = np.zeros(len(X_train))
lgb_preds = np.zeros(len(X_test))
lgb_scores = []

print("Training LightGBM model...")

# 训练模型
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}")
    
    # 准备训练和验证数据
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    lgb_val_pred = lgb_model.predict(X_val)
    lgb_oof[val_idx] = lgb_val_pred
    lgb_score = mean_absolute_error(y_val, lgb_val_pred)
    lgb_scores.append(lgb_score)
    lgb_test_pred = np.array(lgb_model.predict(X_test))
    lgb_preds = lgb_preds + lgb_test_pred / 3

# 计算平均分数
lgb_mean_score = np.mean(lgb_scores)
print(f"LightGBM CV Score: {lgb_mean_score:.4f}")

# 生成最终预测
final_prediction = lgb_preds

# 保存预测结果
submission = pd.DataFrame({
    'price': final_prediction
})

submission.to_csv(os.path.join(pred_dir, 'predictions_v12.csv'), index=False)
print("V12 predictions saved to '../../prediction_result/predictions_v12.csv'")

# 保存验证集预测结果用于后续分析
val_predictions = pd.DataFrame({
    'true_price': y_train,
    'lgb_pred': lgb_oof
})

val_predictions.to_csv(os.path.join(user_data_dir, 'val_predictions_v12.csv'), index=False)
print("V12 validation predictions saved to '../../user_data/val_predictions_v12.csv'")