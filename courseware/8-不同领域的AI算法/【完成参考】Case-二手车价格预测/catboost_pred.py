#!/usr/bin/env python
# coding: utf-8
"""
CatBoost模型加载与测试集预测脚本
"""

import joblib
import pandas as pd
from catboost import CatBoostRegressor, Pool

# 1. 加载特征工程后的数据
X_train = joblib.load('processed_data/fe_X_train.joblib')
X_val = joblib.load('processed_data/fe_X_val.joblib')
y_train = joblib.load('processed_data/fe_y_train.joblib')
y_val = joblib.load('processed_data/fe_y_val.joblib')
X_test = joblib.load('processed_data/fe_test_data.joblib')
test_ids = joblib.load('processed_data/fe_sale_ids.joblib')
cat_features = joblib.load('processed_data/fe_cat_features.joblib')

# 2. 加载已保存的CatBoost模型
print("加载CatBoost模型...")
model = CatBoostRegressor()
model.load_model('processed_data/fe_catboost_model.cbm')
print("模型加载完成！")

# 3. 用模型对测试集进行预测
print("对测试集进行预测...")
test_pool = Pool(X_test, cat_features=cat_features)
predictions = model.predict(test_pool)

# 4. 生成提交文件
submit_data = pd.DataFrame({
    'SaleID': test_ids,
    'price': predictions
})
submit_data.to_csv('catboost_submit_result.csv', index=False)
print("预测结果已保存到 catboost_submit_result.csv")
