#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA Time Series Prediction
使用ARIMA模型预测申购总额和赎回总额
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pickle
import warnings
warnings.filterwarnings('ignore')


def get_project_path(*paths):
    """获取项目路径的统一方法"""
    import os
    try:
        return os.path.join(os.path.dirname(__file__), *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)


# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_training_data():
    """加载训练数据（2014-03-01 到 2014-08-31）"""
    csv_file = get_project_path('..', 'user_data', 'filtered_data_20140301_20140831.csv')
    
    try:
        df = pd.read_csv(csv_file)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df.set_index('date', inplace=True)
        
        print(f"成功加载训练数据，共 {len(df)} 行记录")
        print(f"训练数据时间范围: {df.index[0].strftime('%Y-%m-%d')} 到 {df.index[-1].strftime('%Y-%m-%d')}")
        
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None


def train_arima_model(series, series_name, order, save_path):
    """训练ARIMA模型"""
    print(f"\n=== 训练 {series_name} ARIMA{order} 模型 ===")
    
    try:
        # 创建ARIMA模型
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        # 打印模型摘要
        print(f"ARIMA{order} 模型训练完成")
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")
        print(f"Log Likelihood: {fitted_model.llf:.2f}")
        
        # 保存模型
        with open(save_path, 'wb') as f:
            pickle.dump(fitted_model, f)
        print(f"模型已保存到: {save_path}")
        
        # 进行训练集预测和残差分析
        train_predictions = fitted_model.fittedvalues
        residuals = series - train_predictions
        
        print(f"\n训练集预测评估:")
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"MAE (平均绝对误差): {mae:,.0f}")
        print(f"RMSE (均方根误差): {rmse:,.0f}")
        
        return fitted_model
        
    except Exception as e:
        print(f"模型训练失败: {e}")
        return None


def make_predictions(model, start_date, end_date, predict_periods=30):
    """进行预测"""
    try:
        # 生成预测日期
        prediction_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 进行预测
        forecast = model.forecast(steps=predict_periods)
        forecast_ci = model.get_forecast(steps=predict_periods).conf_int()
        
        print(f"\n预测结果:")
        print(f"预测时间段: {start_date} 到 {end_date}")
        print(f"预测天数: {predict_periods} 天")
        
        # 创建预测结果DataFrame
        prediction_df = pd.DataFrame({
            'report_date': prediction_dates.strftime('%Y%m%d'),
            'predicted_value': forecast.values,
            'lower_bound': forecast_ci.iloc[:, 0].values,
            'upper_bound': forecast_ci.iloc[:, 1].values
        })
        
        # 显示前5天预测结果
        print(f"\n前5天预测结果:")
        for i in range(min(5, len(prediction_df))):
            row = prediction_df.iloc[i]
            print(f"{row['report_date']}: {row['predicted_value']:,.0f} "
                  f"(区间: {row['lower_bound']:,.0f} - {row['upper_bound']:,.0f})")
        
        return prediction_df
        
    except Exception as e:
        print(f"预测失败: {e}")
        return None


def create_prediction_plots(train_data, purchase_predictions, redeem_predictions):
    """创建预测结果图表"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('ARIMA模型预测结果 (2014-09-01 到 2014-09-30)', fontsize=16)
    
    # 预测申购总额
    axes[0].plot(train_data.index, train_data['purchase_amt'], 'b-', linewidth=1, label='历史数据')
    axes[0].plot(purchase_predictions['predicted_date'], purchase_predictions['predicted_value'], 'r-', linewidth=2, label='预测值')
    axes[0].fill_between(purchase_predictions['predicted_date'], 
                         purchase_predictions['lower_bound'], 
                         purchase_predictions['upper_bound'], 
                         alpha=0.3, color='red', label='置信区间')
    axes[0].set_title('申购总额预测')
    axes[0].set_ylabel('申购金额')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 预测赎回总额
    axes[1].plot(train_data.index, train_data['redeem_amt'], 'b-', linewidth=1, label='历史数据')
    axes[1].plot(redeem_predictions['predicted_date'], redeem_predictions['predicted_value'], 'r-', linewidth=2, label='预测值')
    axes[1].fill_between(redeem_predictions['predicted_date'], 
                         redeem_predictions['lower_bound'], 
                         redeem_predictions['upper_bound'], 
                         alpha=0.3, color='red', label='置信区间')
    axes[1].set_title('赎回总额预测')
    axes[1].set_ylabel('赎回金额')
    axes[1].set_xlabel('日期')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = get_project_path('..', 'user_data', 'arima_predictions_201409.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"预测图表已保存到: {plot_file}")
    return plot_file


def save_prediction_results(purchase_predictions, redeem_predictions, output_file):
    """保存最终预测结果"""
    # 合并预测结果
    final_predictions = pd.DataFrame({
        'report_date': purchase_predictions['report_date'],
        'purchase': purchase_predictions['predicted_value'].round(0).astype(int),
        'redeem': redeem_predictions['predicted_value'].round(0).astype(int)
    })
    
    # 显示原始浮点值示例（前3天）
    print(f"\n原始浮点预测结果示例（前3天）:")
    for i in range(min(3, len(purchase_predictions))):
        row_p = purchase_predictions.iloc[i]
        row_r = redeem_predictions.iloc[i]
        print(f"{row_p['report_date']}: 申购={row_p['predicted_value']:.2f}, 赎回={row_r['predicted_value']:.2f}")
        print(f"           整数化后: 申购={final_predictions.iloc[i]['purchase']}, 赎回={final_predictions.iloc[i]['redeem']}")
        print()
    
    # 保存为CSV（不包含表头）
    final_predictions.to_csv(output_file, index=False, header=False, encoding='utf-8')
    
    print(f"预测结果已保存到: {output_file}")
    print(f"预测数据格式: 不包含表头，格式为 report_date,purchase,redeem")
    print(f"说明: 原始浮点预测结果已四舍五入为整数")
    
    return final_predictions


def main():
    """主函数"""
    print("=== ARIMA时间序列预测 ===")
    
    # 创建model目录
    model_dir = get_project_path('..', 'model')
    import os
    os.makedirs(model_dir, exist_ok=True)
    
    # 加载训练数据
    train_data = load_training_data()
    if train_data is None:
        return
    
    # 训练申购总额的ARIMA(5,0,5)模型（不需要差分）
    purchase_model_path = get_project_path('..', 'model', 'purchase_arima_model.pkl')
    purchase_model = train_arima_model(
        train_data['purchase_amt'], 
        '申购总额', 
        (5, 0, 5), 
        purchase_model_path
    )
    
    if purchase_model is None:
        print("申购模型训练失败，程序退出")
        return
    
    # 训练赎回总额的ARIMA(5,1,5)模型（需要差分）
    redeem_model_path = get_project_path('..', 'model', 'redeem_arima_model.pkl')
    redeem_model = train_arima_model(
        train_data['redeem_amt'], 
        '赎回总额', 
        (5, 1, 5), 
        redeem_model_path
    )
    
    if redeem_model is None:
        print("赎回模型训练失败，程序退出")
        return
    
    # 进行预测（2014-09-01 到 2014-09-30）
    start_date = '2014-09-01'
    end_date = '2014-09-30'
    predict_periods = 30
    
    print(f"\n=== 开始预测 ===")
    
    # 预测申购总额
    purchase_predictions = make_predictions(purchase_model, start_date, end_date, predict_periods)
    if purchase_predictions is None:
        print("申购预测失败")
        return
    
    # 预测赎回总额
    redeem_predictions = make_predictions(redeem_model, start_date, end_date, predict_periods)
    if redeem_predictions is None:
        print("赎回预测失败")
        return
    
    # 准备绘图数据
    purchase_plot_data = pd.DataFrame({
        'report_date': purchase_predictions['report_date'],
        'predicted_date': pd.to_datetime(purchase_predictions['report_date'], format='%Y%m%d'),
        'predicted_value': purchase_predictions['predicted_value'],
        'lower_bound': purchase_predictions['lower_bound'],
        'upper_bound': purchase_predictions['upper_bound']
    })
    
    redeem_plot_data = pd.DataFrame({
        'report_date': redeem_predictions['report_date'],
        'predicted_date': pd.to_datetime(redeem_predictions['report_date'], format='%Y%m%d'),
        'predicted_value': redeem_predictions['predicted_value'],
        'lower_bound': redeem_predictions['lower_bound'],
        'upper_bound': redeem_predictions['upper_bound']
    })
    
    # 创建预测图表
    create_prediction_plots(train_data, purchase_plot_data, redeem_plot_data)
    
    # 保存最终预测结果到CSV
    output_file = get_project_path('..', 'prediction_result', 'arima_predictions_201409.csv')
    save_prediction_results(purchase_predictions, redeem_predictions, output_file)
    
    print(f"\n=== 预测完成 ===")
    print(f"模型文件已保存到: model/")
    print(f"预测图表已保存到: user_data/arima_predictions_201409.png")
    print(f"预测结果已保存到: {output_file}")
    
    # 显示最终预测统计
    total_predicted_purchase = purchase_predictions['predicted_value'].sum()
    total_predicted_redeem = redeem_predictions['predicted_value'].sum()
    
    print(f"\n=== 2014年9月预测汇总 ===")
    print(f"申购总额预测: {total_predicted_purchase/1e9:.2f}B")
    print(f"赎回总额预测: {total_predicted_redeem/1e9:.2f}B")
    print(f"净流入预测: {(total_predicted_purchase - total_predicted_redeem)/1e9:.2f}B")


if __name__ == "__main__":
    main()