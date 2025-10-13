#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V12模型分析脚本
用于分析训练集和测试集的预测分布，帮助诊断模型性能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
train_df = pd.read_csv('../../data/processed_train.csv')
val_predictions = pd.read_csv('../../user_data/val_predictions_v12.csv')

# 计算各个模型的MAE
lgb_mae = mean_absolute_error(val_predictions['true_price'], val_predictions['lgb_pred'])
xgb_mae = mean_absolute_error(val_predictions['true_price'], val_predictions['xgb_pred'])
cb_mae = mean_absolute_error(val_predictions['true_price'], val_predictions['cb_pred'])
ensemble_mae = mean_absolute_error(val_predictions['true_price'], val_predictions['ensemble_pred'])

print(f"LightGBM MAE: {lgb_mae:.4f}")
print(f"XGBoost MAE: {xgb_mae:.4f}")
print(f"CatBoost MAE: {cb_mae:.4f}")
print(f"Ensemble MAE: {ensemble_mae:.4f}")

# 绘制预测值vs真实值散点图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# LightGBM
axes[0, 0].scatter(val_predictions['true_price'], val_predictions['lgb_pred'], alpha=0.5)
axes[0, 0].plot([val_predictions['true_price'].min(), val_predictions['true_price'].max()], 
                [val_predictions['true_price'].min(), val_predictions['true_price'].max()], 'r--', lw=2)
axes[0, 0].set_xlabel('True Price')
axes[0, 0].set_ylabel('Predicted Price')
axes[0, 0].set_title(f'LightGBM Predictions vs True Prices (MAE: {lgb_mae:.4f})')

# XGBoost
axes[0, 1].scatter(val_predictions['true_price'], val_predictions['xgb_pred'], alpha=0.5)
axes[0, 1].plot([val_predictions['true_price'].min(), val_predictions['true_price'].max()], 
                [val_predictions['true_price'].min(), val_predictions['true_price'].max()], 'r--', lw=2)
axes[0, 1].set_xlabel('True Price')
axes[0, 1].set_ylabel('Predicted Price')
axes[0, 1].set_title(f'XGBoost Predictions vs True Prices (MAE: {xgb_mae:.4f})')

# CatBoost
axes[1, 0].scatter(val_predictions['true_price'], val_predictions['cb_pred'], alpha=0.5)
axes[1, 0].plot([val_predictions['true_price'].min(), val_predictions['true_price'].max()], 
                [val_predictions['true_price'].min(), val_predictions['true_price'].max()], 'r--', lw=2)
axes[1, 0].set_xlabel('True Price')
axes[1, 0].set_ylabel('Predicted Price')
axes[1, 0].set_title(f'CatBoost Predictions vs True Prices (MAE: {cb_mae:.4f})')

# Ensemble
axes[1, 1].scatter(val_predictions['true_price'], val_predictions['ensemble_pred'], alpha=0.5)
axes[1, 1].plot([val_predictions['true_price'].min(), val_predictions['true_price'].max()], 
                [val_predictions['true_price'].min(), val_predictions['true_price'].max()], 'r--', lw=2)
axes[1, 1].set_xlabel('True Price')
axes[1, 1].set_ylabel('Predicted Price')
axes[1, 1].set_title(f'Ensemble Predictions vs True Prices (MAE: {ensemble_mae:.4f})')

plt.tight_layout()
plt.savefig('../../user_data/v12_predictions_scatter.png')
plt.show()

# 绘制残差图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# LightGBM残差
lgb_residuals = val_predictions['true_price'] - val_predictions['lgb_pred']
axes[0, 0].scatter(val_predictions['lgb_pred'], lgb_residuals, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted Price')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('LightGBM Residuals Plot')

# XGBoost残差
xgb_residuals = val_predictions['true_price'] - val_predictions['xgb_pred']
axes[0, 1].scatter(val_predictions['xgb_pred'], xgb_residuals, alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted Price')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('XGBoost Residuals Plot')

# CatBoost残差
cb_residuals = val_predictions['true_price'] - val_predictions['cb_pred']
axes[1, 0].scatter(val_predictions['cb_pred'], cb_residuals, alpha=0.5)
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Predicted Price')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('CatBoost Residuals Plot')

# Ensemble残差
ensemble_residuals = val_predictions['true_price'] - val_predictions['ensemble_pred']
axes[1, 1].scatter(val_predictions['ensemble_pred'], ensemble_residuals, alpha=0.5)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Predicted Price')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Ensemble Residuals Plot')

plt.tight_layout()
plt.savefig('../../user_data/v12_residuals.png')
plt.show()

# 绘制MAE对比柱状图
models = ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble']
maes = [lgb_mae, xgb_mae, cb_mae, ensemble_mae]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, maes, color=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
plt.ylabel('Mean Absolute Error')
plt.title('Model Performance Comparison (V12)')
plt.ylim(0, max(maes) * 1.1)

# 在柱状图上添加数值标签
for bar, mae in zip(bars, maes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'{mae:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('../../user_data/v12_model_comparison.png')
plt.show()

# 分析价格区间的预测性能
val_predictions['price_range'] = pd.cut(val_predictions['true_price'], 
                                       bins=[0, 50000, 100000, 150000, 200000, np.inf], 
                                       labels=['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k+'])

# 计算各价格区间的MAE
price_range_mae = {}
for price_range in val_predictions['price_range'].unique():
    if pd.isna(price_range):
        continue
    subset = val_predictions[val_predictions['price_range'] == price_range]
    mae = mean_absolute_error(subset['true_price'], subset['ensemble_pred'])
    price_range_mae[price_range] = mae

# 绘制各价格区间的MAE
plt.figure(figsize=(10, 6))
ranges = list(price_range_mae.keys())
maes = list(price_range_mae.values())
bars = plt.bar(ranges, maes, color='lightblue')
plt.ylabel('Mean Absolute Error')
plt.title('Ensemble Model Performance by Price Range (V12)')
plt.xticks(rotation=45)

# 在柱状图上添加数值标签
for bar, mae in zip(bars, maes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'{mae:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('../../user_data/v12_price_range_performance.png')
plt.show()

print("V12模型分析完成，图表已保存到user_data目录下")
