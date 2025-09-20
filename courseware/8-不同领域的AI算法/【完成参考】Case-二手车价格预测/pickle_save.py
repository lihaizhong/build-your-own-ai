#!/usr/bin/env python
# coding: utf-8
"""
使用pickle保存特征工程数据和CatBoost模型
"""

import pickle
import joblib
from catboost import CatBoostRegressor

# 1. 加载特征工程后的数据（用joblib加载已有文件）
X_train = joblib.load('processed_data/fe_X_train.joblib')
X_val = joblib.load('processed_data/fe_X_val.joblib')
y_train = joblib.load('processed_data/fe_y_train.joblib')
y_val = joblib.load('processed_data/fe_y_val.joblib')
X_test = joblib.load('processed_data/fe_test_data.joblib')
test_ids = joblib.load('processed_data/fe_sale_ids.joblib')
cat_features = joblib.load('processed_data/fe_cat_features.joblib')

# 2. 加载CatBoost模型
model = CatBoostRegressor()
model.load_model('processed_data/fe_catboost_model.cbm')

# 3. 使用pickle保存数据
with open('processed_data/fe_X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('processed_data/fe_X_val.pkl', 'wb') as f:
    pickle.dump(X_val, f)
with open('processed_data/fe_y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('processed_data/fe_y_val.pkl', 'wb') as f:
    pickle.dump(y_val, f)
with open('processed_data/fe_test_data.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('processed_data/fe_sale_ids.pkl', 'wb') as f:
    pickle.dump(test_ids, f)
with open('processed_data/fe_cat_features.pkl', 'wb') as f:
    pickle.dump(cat_features, f)

# 4. 使用pickle保存CatBoost模型
with open('processed_data/fe_catboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("所有数据和模型已使用pickle保存为.pkl文件！") 