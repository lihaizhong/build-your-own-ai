#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def get_project_path(*paths):
    """获取项目路径的统一方法"""
    import os
    try:
        return os.path.join(os.path.dirname(__file__), *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)


def load_and_prepare_data():
    """加载并准备Prophet模型的数据"""
    print("=== 加载数据并准备Prophet格式 ===")
    
    # 读取每日汇总数据
    data_file = get_project_path('..', 'user_data', 'daily_summary.csv')
    df = pd.read_csv(data_file, header=None, names=['date', 'purchase', 'redeem'])
    
    # 转换日期格式
    df['ds'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    # 创建申购数据框
    purchase_df = df[['ds', 'purchase']].copy()
    purchase_df.rename(columns={'purchase': 'y'}, inplace=True)
    
    # 创建赎回数据框
    redeem_df = df[['ds', 'redeem']].copy()
    redeem_df.rename(columns={'redeem': 'y'}, inplace=True)
    
    print(f"数据加载完成:")
    print(f"- 数据时间范围: {df['ds'].min()} 至 {df['ds'].max()}")
    print(f"- 总天数: {len(df)} 天")
    print(f"- 申购数据平均: ¥{purchase_df['y'].mean():,.0f}")
    print(f"- 赎回数据平均: ¥{redeem_df['y'].mean():,.0f}")
    
    return purchase_df, redeem_df


def train_prophet_model(df, model_name, target_column):
    """训练Prophet模型"""
    print(f"\n=== 训练{model_name}Prophet模型 ===")
    
    # 创建Prophet模型
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        mcmc_samples=0
    )
    
    # 添加自定义假日（可选）
    # model.add_country_holidays(country_name='China')
    
    # 训练模型
    model.fit(df)
    
    # 创建未来日期
    future = model.make_future_dataframe(periods=30)  # 预测未来30天
    
    # 生成预测
    forecast = model.predict(future)
    
    # 保存模型
    model_path = get_project_path('..', 'model', f'{target_column}_prophet_model.pkl')
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"模型已保存到: {model_path}")
    
    return model, forecast


def generate_predictions(purchase_model, redeem_model, forecast_purchase, forecast_redeem):
    """生成预测结果"""
    print("\n=== 生成预测结果 ===")
    
    # 获取未来30天的预测
    future_predictions = forecast_purchase.tail(30)
    future_redeem = forecast_redeem.tail(30)
    
    # 创建预测结果数据框
    predictions = pd.DataFrame({
        'date': future_predictions['ds'],
        'purchase_forecast': future_predictions['yhat'],
        'redeem_forecast': future_redeem['yhat']
    })
    
    # 计算置信区间
    predictions['purchase_lower'] = future_predictions['yhat_lower']
    predictions['purchase_upper'] = future_predictions['yhat_upper']
    predictions['redeem_lower'] = future_redeem['yhat_lower']
    predictions['redeem_upper'] = future_redeem['yhat_upper']
    
    # 保存预测结果（考试格式）
    prediction_file = get_project_path('..', 'prediction_result', 'prophet_predictions_201409.csv')
    exam_format = predictions[['date']].copy()
    exam_format['date'] = exam_format['date'].dt.strftime('%Y%m%d')
    exam_format['purchase'] = predictions['purchase_forecast'].round(0).astype(int)
    exam_format['redeem'] = predictions['redeem_forecast'].round(0).astype(int)
    
    # 保存为CSV（考试格式：YYYYMMDD,申购金额,赎回金额）
    exam_format.to_csv(prediction_file, header=False, index=False)
    
    print(f"预测结果已保存到: {prediction_file}")
    print(f"预测期间: {predictions['date'].min()} 至 {predictions['date'].max()}")
    print(f"预测平均申购额: ¥{predictions['purchase_forecast'].mean():,.0f}")
    print(f"预测平均赎回额: ¥{predictions['redeem_forecast'].mean():,.0f}")
    
    return predictions


def create_visualization(purchase_df, redeem_df, forecast_purchase, forecast_redeem, predictions):
    """创建可视化图表"""
    print("\n=== 生成可视化图表 ===")
    
    # 设置图表风格（会影响全局字体设置）
    # plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prophet时间序列预测分析', fontsize=16, fontweight='bold')
    
    # 1. 申购趋势预测
    ax1 = axes[0, 0]
    # 历史数据
    ax1.plot(purchase_df['ds'], purchase_df['y'], 'b-', alpha=0.7, label='历史申购数据')
    # 预测数据
    forecast_purchase_future = forecast_purchase.tail(30)
    ax1.plot(forecast_purchase_future['ds'], forecast_purchase_future['yhat'], 'r-', label='预测申购额')
    # 置信区间
    ax1.fill_between(forecast_purchase_future['ds'], 
                    forecast_purchase_future['yhat_lower'],
                    forecast_purchase_future['yhat_upper'],
                    alpha=0.2, color='red', label='95%置信区间')
    
    ax1.set_title('申购金额预测趋势')
    ax1.set_ylabel('申购金额')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 赎回趋势预测
    ax2 = axes[0, 1]
    # 历史数据
    ax2.plot(redeem_df['ds'], redeem_df['y'], 'g-', alpha=0.7, label='历史赎回数据')
    # 预测数据
    forecast_redeem_future = forecast_redeem.tail(30)
    ax2.plot(forecast_redeem_future['ds'], forecast_redeem_future['yhat'], 'orange', label='预测赎回额')
    # 置信区间
    ax2.fill_between(forecast_redeem_future['ds'], 
                    forecast_redeem_future['yhat_lower'],
                    forecast_redeem_future['yhat_upper'],
                    alpha=0.2, color='orange', label='95%置信区间')
    
    ax2.set_title('赎回金额预测趋势')
    ax2.set_ylabel('赎回金额')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 组件分析（申购）
    components = purchase_model.plot_components(forecast_purchase)
    components.suptitle('申购金额预测组件分析', fontsize=14)
    
    # 4. 组件分析（赎回）
    components = redeem_model.plot_components(forecast_redeem)
    components.suptitle('赎回金额预测组件分析', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    chart_file = get_project_path('..', 'user_data', 'prophet_forecast_analysis.png')
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到: {chart_file}")
    
    # 创建单独的预测对比图
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 绘制预测期间的申购赎回对比
    pred_dates = pd.to_datetime(predictions['date'])
    ax.plot(pred_dates, predictions['purchase_forecast'], 'r-', linewidth=2, label='预测申购额')
    ax.plot(pred_dates, predictions['redeem_forecast'], 'b-', linewidth=2, label='预测赎回额')
    
    # 添加置信区间
    ax.fill_between(pred_dates, 
                   predictions['purchase_lower'],
                   predictions['purchase_upper'],
                   alpha=0.2, color='red', label='申购额置信区间')
    ax.fill_between(pred_dates, 
                   predictions['redeem_lower'],
                   predictions['redeem_upper'],
                   alpha=0.2, color='blue', label='赎回额置信区间')
    
    ax.set_title('未来30天申购赎回预测对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('金额')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加净流入线
    net_flow = predictions['purchase_forecast'] - predictions['redeem_forecast']
    ax2 = ax.twinx()
    ax2.plot(pred_dates, net_flow, 'g--', linewidth=2, alpha=0.7, label='净流入')
    ax2.set_ylabel('净流入', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_file = get_project_path('..', 'user_data', 'prophet_forecast_comparison.png')
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"对比图表已保存到: {comparison_file}")


def analyze_model_performance(forecast_purchase, forecast_redeem, purchase_df, redeem_df):
    """分析模型性能"""
    print("\n=== 模型性能分析 ===")
    
    # 分离训练期和预测期
    train_size = len(purchase_df)
    test_purchase = forecast_purchase.iloc[:train_size]
    test_redeem = forecast_redeem.iloc[:train_size]
    
    # 计算误差指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # 申购模型误差
    purchase_mae = mean_absolute_error(purchase_df['y'], test_purchase['yhat'])
    purchase_rmse = np.sqrt(mean_squared_error(purchase_df['y'], test_purchase['yhat']))
    purchase_mape = np.mean(np.abs((purchase_df['y'] - test_purchase['yhat']) / purchase_df['y'])) * 100
    
    # 赎回模型误差
    redeem_mae = mean_absolute_error(redeem_df['y'], test_redeem['yhat'])
    redeem_rmse = np.sqrt(mean_squared_error(redeem_df['y'], test_redeem['yhat']))
    redeem_mape = np.mean(np.abs((redeem_df['y'] - test_redeem['yhat']) / redeem_df['y'])) * 100
    
    print(f"申购模型性能:")
    print(f"  MAE: ¥{purchase_mae:,.0f}")
    print(f"  RMSE: ¥{purchase_rmse:,.0f}")
    print(f"  MAPE: {purchase_mape:.2f}%")
    
    print(f"\n赎回模型性能:")
    print(f"  MAE: ¥{redeem_mae:,.0f}")
    print(f"  RMSE: ¥{redeem_rmse:,.0f}")
    print(f"  MAPE: {redeem_mape:.2f}%")
    
    return {
        'purchase_mae': purchase_mae,
        'purchase_rmse': purchase_rmse,
        'purchase_mape': purchase_mape,
        'redeem_mae': redeem_mae,
        'redeem_rmse': redeem_rmse,
        'redeem_mape': redeem_mape
    }


def main():
    """主函数"""
    print("=== Prophet资金流入流出预测分析 ===")
    
    try:
        # 1. 加载数据
        purchase_df, redeem_df = load_and_prepare_data()
        
        # 2. 训练模型
        global purchase_model, redeem_model
        purchase_model, forecast_purchase = train_prophet_model(purchase_df, "申购", "purchase")
        redeem_model, forecast_redeem = train_prophet_model(redeem_df, "赎回", "redeem")
        
        # 3. 生成预测
        predictions = generate_predictions(purchase_model, redeem_model, forecast_purchase, forecast_redeem)
        
        # 4. 分析模型性能
        performance = analyze_model_performance(forecast_purchase, forecast_redeem, purchase_df, redeem_df)
        
        # 5. 创建可视化
        create_visualization(purchase_df, redeem_df, forecast_purchase, forecast_redeem, predictions)
        
        print(f"\n=== 预测完成 ===")
        print(f"模型训练成功，预测结果已保存")
        print(f"可查看文件:")
        print(f"- 预测结果: prediction_result/prophet_predictions_201409.csv")
        print(f"- 分析图表: user_data/prophet_forecast_analysis.png")
        print(f"- 对比图表: user_data/prophet_forecast_comparison.png")
        
        return True
        
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()