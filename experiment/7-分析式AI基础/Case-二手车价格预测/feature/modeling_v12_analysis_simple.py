import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建数据文件的相对路径
data_dir = os.path.join(current_dir, '..', 'processed_data')
user_data_dir = os.path.join(current_dir, '..', 'user_data')

# 加载数据
print("Loading data...")
y_train = joblib.load(os.path.join(data_dir, 'y_train.joblib'))
val_predictions = pd.read_csv(os.path.join(user_data_dir, 'val_predictions_v12.csv'))

# 计算LightGBM的MAE
lgb_mae = mean_absolute_error(val_predictions['true_price'], val_predictions['lgb_pred'])

print(f"LightGBM MAE: {lgb_mae:.4f}")

# 绘制预测值vs真实值散点图
plt.figure(figsize=(10, 8))
plt.scatter(val_predictions['true_price'], val_predictions['lgb_pred'], alpha=0.5)
plt.plot([val_predictions['true_price'].min(), val_predictions['true_price'].max()], 
         [val_predictions['true_price'].min(), val_predictions['true_price'].max()], 'r--', lw=2)
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.title(f'LightGBM Predictions vs True Prices (MAE: {lgb_mae:.4f})')
plt.tight_layout()
plt.savefig(os.path.join(user_data_dir, 'v12_predictions_scatter.png'))
plt.close()  # 关闭图形以释放内存

# 绘制残差图
plt.figure(figsize=(10, 8))
lgb_residuals = val_predictions['true_price'] - val_predictions['lgb_pred']
plt.scatter(val_predictions['lgb_pred'], lgb_residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('LightGBM Residuals Plot')
plt.tight_layout()
plt.savefig(os.path.join(user_data_dir, 'v12_residuals.png'))
plt.close()  # 关闭图形以释放内存

# 绘制MAE对比柱状图
models = ['LightGBM']
maes = [lgb_mae]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, maes, color=['skyblue'])
plt.ylabel('Mean Absolute Error')
plt.title('Model Performance Comparison (V12)')
plt.ylim(0, max(maes) * 1.1)

# 在柱状图上添加数值标签
for bar, mae in zip(bars, maes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'{mae:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(user_data_dir, 'v12_model_comparison.png'))
plt.close()  # 关闭图形以释放内存

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
    mae = mean_absolute_error(subset['true_price'], subset['lgb_pred'])
    price_range_mae[price_range] = mae

# 绘制各价格区间的MAE
plt.figure(figsize=(10, 6))
ranges = list(price_range_mae.keys())
maes = list(price_range_mae.values())
bars = plt.bar(ranges, maes, color='lightblue')
plt.ylabel('Mean Absolute Error')
plt.title('LightGBM Model Performance by Price Range (V12)')
plt.xticks(rotation=45)

# 在柱状图上添加数值标签
for bar, mae in zip(bars, maes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'{mae:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(user_data_dir, 'v12_price_range_performance.png'))
plt.close()  # 关闭图形以释放内存

print("V12模型分析完成，图表已保存到user_data目录下")