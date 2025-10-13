import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建数据文件的相对路径
data_dir = os.path.join(current_dir, '..', 'processed_data')

# 加载处理好的数据
print("Loading processed data...")
X_train = joblib.load(os.path.join(data_dir, 'X_train.joblib'))
y_train = joblib.load(os.path.join(data_dir, 'y_train.joblib'))
X_test = joblib.load(os.path.join(data_dir, 'test_data.joblib'))

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=pd.Index(X_train.columns))
X_test_scaled = pd.DataFrame(X_test_scaled, columns=pd.Index(X_test.columns))

# 更激进的LightGBM参数
lgb_params = {
    'objective': 'mae',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,  # 大幅增加复杂度
    'max_depth': 6,    # 大幅增加深度
    'min_data_in_leaf': 20,  # 大幅减少最小样本数
    'feature_fraction': 0.8,  # 减少特征采样
    'bagging_fraction': 0.8,  # 减少数据采样
    'bagging_freq': 5,
    'lambda_l1': 0.0,  # 几乎无L1正则化
    'lambda_l2': 0.0,  # 几乎无L2正则化
    'min_gain_to_split': 0.0,  # 几乎无分裂最小增益
    'verbose': -1,
    'random_state': 42
}

# 更激进的XGBoost参数
xgb_params = {
    'objective': 'reg:absoluteerror',
    'eval_metric': 'mae',
    'booster': 'gbtree',
    'max_depth': 6,  # 大幅增加深度
    'learning_rate': 0.05,
    'n_estimators': 2000,
    'subsample': 0.8,  # 减少数据采样
    'colsample_bytree': 0.8,  # 减少特征采样
    'reg_alpha': 0.0,  # 几乎无L1正则化
    'reg_lambda': 0.0,  # 几乎无L2正则化
    'min_child_weight': 1,
    'random_state': 42
}

# 更激进的CatBoost参数
cb_params = {
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'iterations': 2000,
    'depth': 6,  # 大幅增加深度
    'learning_rate': 0.05,
    'l2_leaf_reg': 0.0,  # 几乎无L2正则化
    'bootstrap_type': 'Bayesian',
    'random_seed': 42,
    'verbose': False
}

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_oof = np.zeros(len(X_train))
xgb_oof = np.zeros(len(X_train))
cb_oof = np.zeros(len(X_train))
lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

lgb_scores = []
xgb_scores = []
cb_scores = []

print("Training models with aggressive parameters...")

# 训练模型
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}")
    
    # 准备训练和验证数据
    X_tr_lgb, X_val_lgb = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr_lgb, y_val_lgb = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    X_tr_xgb, X_val_xgb = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
    y_tr_xgb, y_val_xgb = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    X_tr_cb, X_val_cb = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
    y_tr_cb, y_val_cb = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr_lgb, y_tr_lgb,
        eval_set=[(X_val_lgb, y_val_lgb)],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)]
    )
    lgb_val_pred = lgb_model.predict(X_val_lgb)
    lgb_oof[val_idx] = lgb_val_pred
    lgb_score = mean_absolute_error(y_val_lgb, lgb_val_pred)
    lgb_scores.append(lgb_score)
    lgb_test_pred = np.array(lgb_model.predict(X_test))
    for i in range(len(X_test)):
        lgb_preds[i] += lgb_test_pred[i] / 5.0
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_tr_xgb, y_tr_xgb,
        eval_set=[(X_val_xgb, y_val_xgb)],
        verbose=0
    )
    xgb_val_pred = xgb_model.predict(X_val_xgb)
    xgb_oof[val_idx] = xgb_val_pred
    xgb_score = mean_absolute_error(y_val_xgb, xgb_val_pred)
    xgb_scores.append(xgb_score)
    xgb_test_pred = np.array(xgb_model.predict(X_test_scaled))
    for i in range(len(X_test)):
        xgb_preds[i] += xgb_test_pred[i] / 5.0
    
    # CatBoost
    cb_model = cb.CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_tr_cb, y_tr_cb,
        eval_set=[(X_val_cb, y_val_cb)],
        early_stopping_rounds=100,
        verbose=0
    )
    cb_val_pred = cb_model.predict(X_val_cb)
    cb_oof[val_idx] = cb_val_pred
    cb_score = mean_absolute_error(y_val_cb, cb_val_pred)
    cb_scores.append(cb_score)
    cb_test_pred = np.array(cb_model.predict(X_test_scaled))
    for i in range(len(X_test)):
        cb_preds[i] += cb_test_pred[i] / 5.0

# 计算平均分数
lgb_mean_score = np.mean(lgb_scores)
xgb_mean_score = np.mean(xgb_scores)
cb_mean_score = np.mean(cb_scores)

print(f"LightGBM CV Score: {lgb_mean_score:.4f}")
print(f"XGBoost CV Score: {xgb_mean_score:.4f}")
print(f"CatBoost CV Score: {cb_mean_score:.4f}")

# 使用加权平均进行集成（基于验证集表现分配权重）
# 由于我们采用了更激进的参数，这里我们给三个模型分配相对均衡的权重
lgb_weight = 0.4
xgb_weight = 0.3
cb_weight = 0.3

ensemble_oof = lgb_weight * lgb_oof + xgb_weight * xgb_oof + cb_weight * cb_oof
ensemble_score = mean_absolute_error(y_train, ensemble_oof)
print(f"Ensemble CV Score: {ensemble_score:.4f}")

# 生成最终预测
final_prediction = lgb_weight * lgb_preds + xgb_weight * xgb_preds + cb_weight * cb_preds

# 保存预测结果
submission = pd.DataFrame({
    'price': final_prediction
})

submission.to_csv('../../prediction_result/predictions_v12.csv', index=False)
print("V12 predictions saved to '../../prediction_result/predictions_v12.csv'")

# 保存验证集预测结果用于后续分析
val_predictions = pd.DataFrame({
    'true_price': y_train,
    'lgb_pred': lgb_oof,
    'xgb_pred': xgb_oof,
    'cb_pred': cb_oof,
    'ensemble_pred': ensemble_oof
})

val_predictions.to_csv('../../user_data/val_predictions_v12.csv', index=False)
print("V12 validation predictions saved to '../../user_data/val_predictions_v12.csv'")