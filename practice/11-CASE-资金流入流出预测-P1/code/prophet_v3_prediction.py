#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prophet预测模型 v3.0 - 高级优化版本
基于Prophet算法的综合优化版本，包含多重特征工程和参数调优
版本特性：外部变量集成、多重季节性、异常值处理、参数优化
演进：从v2基础+节假日升级到综合优化版本
核心创新：利率回归变量、异常值检测、模型参数自动调优
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import pickle


def get_project_path(*paths):
    """获取项目路径的统一方法"""
    import os
    try:
        return os.path.join(os.path.dirname(__file__), *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)


def load_external_features():
    """加载外部特征数据（利率、收益率等）"""
    print("=== 加载外部特征数据 ===")
    
    # 加载货币基金收益率数据
    interest_file = get_project_path('..', 'data', 'mfd_day_share_interest.csv')
    interest_df = pd.read_csv(interest_file)
    interest_df['ds'] = pd.to_datetime(interest_df['mfd_date'], format='%Y%m%d')
    interest_df = interest_df[['ds', 'mfd_daily_yield', 'mfd_7daily_yield']].copy()
    
    # 加载银行拆借利率数据
    shibor_file = get_project_path('..', 'data', 'mfd_bank_shibor.csv')
    shibor_df = pd.read_csv(shibor_file)
    shibor_df['ds'] = pd.to_datetime(shibor_df['mfd_date'], format='%Y%m%d')
    # 选择主要期限利率
    shibor_df = shibor_df[['ds', 'Interest_O_N', 'Interest_1_W', 'Interest_1_M']].copy()
    
    print(f"外部特征加载完成:")
    print(f"- 收益率数据: {len(interest_df)} 天")
    print(f"- 拆借利率数据: {len(shibor_df)} 天")
    
    return interest_df, shibor_df


def detect_and_handle_outliers(data, column, method='iqr', threshold=3):
    """检测和处理异常值"""
    print(f"=== 检测和处理{column}异常值 ===")
    
    original_data = data.copy()
    
    if method == 'iqr':
        # 使用IQR方法检测异常值
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
        
    elif method == 'zscore':
        # 使用Z-score方法检测异常值
        z_scores = np.abs(stats.zscore(data[column]))
        outlier_mask = z_scores > threshold
    
    outlier_count = outlier_mask.sum()
    outlier_percentage = (outlier_count / len(data)) * 100
    
    print(f"检测到异常值: {outlier_count} 个 ({outlier_percentage:.1f}%)")
    
    if outlier_count > 0:
        # 使用中位数替换异常值
        median_value = data[column].median()
        data.loc[outlier_mask, column] = median_value
        
        print(f"异常值处理完成，使用中位数({median_value:,.0f})替换")
    
    return data, outlier_mask


def create_enhanced_features(df, interest_df, shibor_df):
    """创建增强特征"""
    print("=== 创建增强特征 ===")
    
    # 合并外部数据
    enhanced_df = df.merge(interest_df, on='ds', how='left')
    enhanced_df = enhanced_df.merge(shibor_df, on='ds', how='left')
    
    # 填充缺失值（使用前向填充）
    enhanced_df[['mfd_daily_yield', 'mfd_7daily_yield', 'Interest_O_N', 'Interest_1_W', 'Interest_1_M']] = \
        enhanced_df[['mfd_daily_yield', 'mfd_7daily_yield', 'Interest_O_N', 'Interest_1_W', 'Interest_1_M']].fillna(method='ffill')
    
    # 创建滞后特征（前1-7天）
    for lag in range(1, 8):
        enhanced_df[f'y_lag_{lag}'] = enhanced_df['y'].shift(lag)
    
    # 创建滚动统计特征
    enhanced_df['y_ma_7'] = enhanced_df['y'].rolling(window=7, min_periods=1).mean()
    enhanced_df['y_ma_30'] = enhanced_df['y'].rolling(window=30, min_periods=1).mean()
    enhanced_df['y_std_7'] = enhanced_df['y'].rolling(window=7, min_periods=1).std()
    
    # 创建趋势特征
    enhanced_df['day_of_year'] = enhanced_df['ds'].dt.dayofyear
    enhanced_df['day_of_month'] = enhanced_df['ds'].dt.day
    enhanced_df['week_of_year'] = enhanced_df['ds'].dt.isocalendar().week
    enhanced_df['is_month_end'] = (enhanced_df['day_of_month'] >= 28).astype(int)
    enhanced_df['is_month_start'] = (enhanced_df['day_of_month'] <= 3).astype(int)
    
    # 创建周末效应特征
    enhanced_df['is_weekend'] = (enhanced_df['ds'].dt.weekday >= 5).astype(int)
    enhanced_df['is_friday'] = (enhanced_df['ds'].dt.weekday == 4).astype(int)
    enhanced_df['is_monday'] = (enhanced_df['ds'].dt.weekday == 0).astype(int)
    
    print(f"增强特征创建完成: {len(enhanced_df.columns)} 个特征")
    
    return enhanced_df


def create_china_holidays_v3():
    """创建v3版本的节假日数据（更精确的节假日效应）"""
    holidays = []
    
    # 主要节假日（带权重）
    main_holidays = [
        # 春节效应最强（7天假期）
        {'holiday': '春节假期', 'ds': '2013-02-09', 'lower_window': -3, 'upper_window': 4},
        {'holiday': '春节假期', 'ds': '2014-01-30', 'lower_window': -3, 'upper_window': 4},
        
        # 国庆节（7天）
        {'holiday': '国庆假期', 'ds': '2013-10-01', 'lower_window': 0, 'upper_window': 6},
        {'holiday': '国庆假期', 'ds': '2014-10-01', 'lower_window': 0, 'upper_window': 6},
        
        # 劳动节（3天）
        {'holiday': '劳动节', 'ds': '2013-05-01', 'lower_window': 0, 'upper_window': 2},
        {'holiday': '劳动节', 'ds': '2014-05-01', 'lower_window': 0, 'upper_window': 2},
        
        # 清明节（3天）
        {'holiday': '清明节', 'ds': '2013-04-04', 'lower_window': 0, 'upper_window': 2},
        {'holiday': '清明节', 'ds': '2014-04-05', 'lower_window': 0, 'upper_window': 2},
        
        # 端午节（3天）
        {'holiday': '端午节', 'ds': '2013-06-12', 'lower_window': 0, 'upper_window': 2},
        {'holiday': '端午节', 'ds': '2014-05-31', 'lower_window': 0, 'upper_window': 2},
        
        # 中秋节（3天）
        {'holiday': '中秋节', 'ds': '2013-09-19', 'lower_window': 0, 'upper_window': 2},
        {'holiday': '中秋节', 'ds': '2014-09-06', 'lower_window': 0, 'upper_window': 2},
        
        # 元旦
        {'holiday': '元旦', 'ds': '2013-01-01', 'lower_window': 0, 'upper_window': 0},
        {'holiday': '元旦', 'ds': '2014-01-01', 'lower_window': 0, 'upper_window': 0},
    ]
    
    holidays.extend(main_holidays)
    
    return pd.DataFrame(holidays)


def optimize_prophet_parameters(df):
    """优化Prophet模型参数"""
    print("=== Prophet参数优化 ===")
    
    # 参数候选值
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1],
        'seasonality_prior_scale': [1, 10, 100],
        'holidays_prior_scale': [1, 10, 100],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    
    best_score = float('inf')
    best_params = None
    
    # 简化的网格搜索（避免过长时间）
    for changepoint_prior_scale in [0.01, 0.05, 0.1]:
        for seasonality_prior_scale in [1, 10]:
            for seasonality_mode in ['additive', 'multiplicative']:
                try:
                    # 创建模型
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        seasonality_mode=seasonality_mode,
                        changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_prior_scale=seasonality_prior_scale,
                        holidays_prior_scale=10.0,
                        interval_width=0.95
                    )
                    
                    # 训练模型
                    model.fit(df.iloc[:-30])  # 使用除最后30天外的数据
                    
                    # 预测最后30天
                    future = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future)
                    predictions = forecast['yhat'].iloc[-30:]
                    actual = df['y'].iloc[-30:]
                    
                    # 计算MAPE
                    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
                    
                    if mape < best_score:
                        best_score = mape
                        best_params = {
                            'changepoint_prior_scale': changepoint_prior_scale,
                            'seasonality_prior_scale': seasonality_prior_scale,
                            'seasonality_mode': seasonality_mode
                        }
                        
                    print(f"参数组合测试: MAPE={mape:.2f}%")
                    
                except Exception as e:
                    print(f"参数组合测试失败: {e}")
                    continue
    
    print(f"最优参数: {best_params}, 最佳MAPE: {best_score:.2f}%")
    return best_params


def train_optimized_prophet_model(df, model_name, target_column, interest_df, shibor_df):
    """训练优化版Prophet模型"""
    print(f"\n=== 训练{model_name}优化版Prophet模型 ===")
    
    # 数据预处理
    processed_df, outlier_mask = detect_and_handle_outliers(df, 'y')
    
    # 创建增强特征
    enhanced_df = create_enhanced_features(processed_df, interest_df, shibor_df)
    
    # 创建节假日
    holidays_df = create_china_holidays_v3()
    
    # 参数优化
    best_params = optimize_prophet_parameters(enhanced_df)
    
    # 创建优化版Prophet模型
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=best_params['seasonality_mode'],
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        holidays_prior_scale=10.0,
        mcmc_samples=0,
        holidays=holidays_df,
        interval_width=0.95,
        uncertainty_samples=1000
    )
    
    # 添加外部回归变量
    if not enhanced_df['mfd_daily_yield'].isna().all():
        model.add_regressor('mfd_daily_yield')
    if not enhanced_df['Interest_O_N'].isna().all():
        model.add_regressor('Interest_O_N')
    
    # 添加自定义季节性
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
    
    # 训练模型
    model.fit(enhanced_df)
    
    # 创建未来日期并预测
    future = model.make_future_dataframe(periods=30)
    
    # 为未来日期添加外部特征（使用最后已知值）
    for col in ['mfd_daily_yield', 'mfd_7daily_yield', 'Interest_O_N', 'Interest_1_W', 'Interest_1_M']:
        if col in enhanced_df.columns:
            last_value = enhanced_df[col].iloc[-1]
            future[col] = last_value
    
    forecast = model.predict(future)
    
    # 保存模型
    model_path = get_project_path('..', 'model', f'{target_column}_prophet_v3_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"优化版模型已保存到: {model_path}")
    print(f"模型特征数: {len(enhanced_df.columns)}")
    
    return model, forecast, enhanced_df, outlier_mask


def generate_optimized_predictions(purchase_model, redeem_model, forecast_purchase, forecast_redeem):
    """生成优化版预测结果"""
    print("\n=== 生成优化版预测结果 ===")
    
    # 获取未来30天的预测
    future_predictions = forecast_purchase.tail(30)
    future_redeem = forecast_redeem.tail(30)
    
    # 创建预测结果数据框
    predictions = pd.DataFrame({
        'date': future_predictions['ds'],
        'purchase_forecast': future_predictions['yhat'],
        'redeem_forecast': future_redeem['yhat'],
        'purchase_lower': future_predictions['yhat_lower'],
        'purchase_upper': future_predictions['yhat_upper'],
        'redeem_lower': future_redeem['yhat_lower'],
        'redeem_upper': future_redeem['yhat_upper']
    })
    
    # 添加日期特征
    predictions['weekday'] = predictions['date'].dt.dayofweek
    predictions['is_weekend'] = predictions['weekday'].isin([5, 6])
    predictions['is_friday'] = predictions['weekday'] == 4
    predictions['is_monday'] = predictions['weekday'] == 0
    predictions['day_name'] = predictions['date'].dt.day_name()
    
    # 计算净流入
    predictions['net_flow'] = predictions['purchase_forecast'] - predictions['redeem_forecast']
    predictions['net_flow_lower'] = predictions['purchase_lower'] - predictions['redeem_upper']
    predictions['net_flow_upper'] = predictions['purchase_upper'] - predictions['redeem_lower']
    
    # 保存优化版预测结果（考试格式）
    prediction_file = get_project_path('..', 'prediction_result', 'prophet_v3_predictions_201409.csv')
    exam_format = predictions[['date']].copy()
    exam_format['date'] = exam_format['date'].dt.strftime('%Y%m%d')
    exam_format['purchase'] = predictions['purchase_forecast'].round(0).astype(int)
    exam_format['redeem'] = predictions['redeem_forecast'].round(0).astype(int)
    
    exam_format.to_csv(prediction_file, header=False, index=False)
    
    print(f"优化版预测结果已保存到: {prediction_file}")
    
    # 统计预测结果
    total_purchase = predictions['purchase_forecast'].sum()
    total_redeem = predictions['redeem_forecast'].sum()
    net_flow = total_purchase - total_redeem
    
    print(f"\n📊 预测结果统计:")
    print(f"- 总申购预测: ¥{total_purchase:,.0f}")
    print(f"- 总赎回预测: ¥{total_redeem:,.0f}")
    print(f"- 净流入预测: ¥{net_flow:,.0f}")
    print(f"- 平均日申购: ¥{predictions['purchase_forecast'].mean():,.0f}")
    print(f"- 平均日赎回: ¥{predictions['redeem_forecast'].mean():,.0f}")
    
    return predictions


def create_optimized_visualization(purchase_df, redeem_df, forecast_purchase, forecast_redeem, predictions):
    """创建优化版可视化图表"""
    print("\n=== 生成优化版可视化图表 ===")
    
    # 创建综合分析图表
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('优化版Prophet时间序列预测分析 (v3.0)', fontsize=16, fontweight='bold')
    
    # 1. 申购预测与置信区间
    ax1 = axes[0, 0]
    future_pred = forecast_purchase.tail(30)
    ax1.plot(purchase_df['ds'], purchase_df['y'], 'b-', alpha=0.7, label='历史申购数据')
    ax1.plot(future_pred['ds'], future_pred['yhat'], 'r-', linewidth=2, label='预测申购额')
    ax1.fill_between(future_pred['ds'], future_pred['yhat_lower'], future_pred['yhat_upper'],
                    alpha=0.2, color='red', label='95%置信区间')
    ax1.set_title('申购金额预测（含置信区间）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 赎回预测与置信区间
    ax2 = axes[0, 1]
    future_redeem = forecast_redeem.tail(30)
    ax2.plot(redeem_df['ds'], redeem_df['y'], 'g-', alpha=0.7, label='历史赎回数据')
    ax2.plot(future_redeem['ds'], future_redeem['yhat'], 'orange', linewidth=2, label='预测赎回额')
    ax2.fill_between(future_redeem['ds'], future_redeem['yhat_lower'], future_redeem['yhat_upper'],
                    alpha=0.2, color='orange', label='95%置信区间')
    ax2.set_title('赎回金额预测（含置信区间）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 净流入分析
    ax3 = axes[0, 2]
    ax3.plot(predictions['date'], predictions['net_flow'], 'purple', linewidth=2, label='净流入')
    ax3.fill_between(predictions['date'], predictions['net_flow_lower'], predictions['net_flow_upper'],
                    alpha=0.2, color='purple', label='95%置信区间')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('净流入预测分析')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 周末效应分析
    ax4 = axes[1, 0]
    weekend_data = predictions[predictions['is_weekend']]
    workday_data = predictions[~predictions['is_weekend']]
    
    if len(weekend_data) > 0 and len(workday_data) > 0:
        categories = ['工作日', '周末']
        purchase_means = [workday_data['purchase_forecast'].mean(), weekend_data['purchase_forecast'].mean()]
        redeem_means = [workday_data['redeem_forecast'].mean(), weekend_data['redeem_forecast'].mean()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x - width/2, purchase_means, width, label='申购', alpha=0.8)
        ax4.bar(x + width/2, redeem_means, width, label='赎回', alpha=0.8)
        ax4.set_title('工作日 vs 周末效应')
        ax4.set_ylabel('平均金额')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. 预测分布分析
    ax5 = axes[1, 1]
    ax5.hist(predictions['purchase_forecast'], bins=15, alpha=0.7, label='申购预测', color='red')
    ax5.hist(predictions['redeem_forecast'], bins=15, alpha=0.7, label='赎回预测', color='blue')
    ax5.set_title('预测金额分布')
    ax5.set_xlabel('金额')
    ax5.set_ylabel('频次')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 时间序列分解
    ax6 = axes[1, 2]
    # 显示趋势组件
    trend = forecast_purchase['trend'].iloc[-60:]  # 最近60天
    ax6.plot(trend.index, trend.values, 'green', linewidth=2, label='趋势')
    ax6.set_title('长期趋势分析')
    ax6.set_xlabel('时间')
    ax6.set_ylabel('趋势值')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    chart_file = get_project_path('..', 'user_data', 'optimized_prophet_forecast_analysis.png')
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"优化版分析图表已保存到: {chart_file}")


def analyze_optimized_model_performance(forecast_purchase, forecast_redeem, purchase_df, redeem_df):
    """分析优化版模型性能"""
    print("\n=== 优化版模型性能分析 ===")
    
    # 分离训练期和预测期
    train_size = len(purchase_df)
    test_purchase = forecast_purchase.iloc[:train_size]
    test_redeem = forecast_redeem.iloc[:train_size]
    
    # 计算误差指标
    purchase_mae = mean_absolute_error(purchase_df['y'], test_purchase['yhat'])
    purchase_rmse = np.sqrt(mean_squared_error(purchase_df['y'], test_purchase['yhat']))
    purchase_mape = np.mean(np.abs((purchase_df['y'] - test_purchase['yhat']) / purchase_df['y'])) * 100
    
    redeem_mae = mean_absolute_error(redeem_df['y'], test_redeem['yhat'])
    redeem_rmse = np.sqrt(mean_squared_error(redeem_df['y'], test_redeem['yhat']))
    redeem_mape = np.mean(np.abs((redeem_df['y'] - test_redeem['yhat']) / redeem_df['y'])) * 100
    
    # 计算预测稳定性（方差）
    purchase_residuals = purchase_df['y'] - test_purchase['yhat']
    redeem_residuals = redeem_df['y'] - test_redeem['yhat']
    
    purchase_stability = np.std(purchase_residuals)
    redeem_stability = np.std(redeem_residuals)
    
    print(f"优化版申购模型性能:")
    print(f"  MAE: ¥{purchase_mae:,.0f}")
    print(f"  RMSE: ¥{purchase_rmse:,.0f}")
    print(f"  MAPE: {purchase_mape:.2f}%")
    print(f"  稳定性(标准差): ¥{purchase_stability:,.0f}")
    
    print(f"\n优化版赎回模型性能:")
    print(f"  MAE: ¥{redeem_mae:,.0f}")
    print(f"  RMSE: ¥{redeem_rmse:,.0f}")
    print(f"  MAPE: {redeem_mape:.2f}%")
    print(f"  稳定性(标准差): ¥{redeem_stability:,.0f}")
    
    return {
        'purchase_mae': purchase_mae,
        'purchase_rmse': purchase_rmse,
        'purchase_mape': purchase_mape,
        'purchase_stability': purchase_stability,
        'redeem_mae': redeem_mae,
        'redeem_rmse': redeem_rmse,
        'redeem_mape': redeem_mape,
        'redeem_stability': redeem_stability
    }


def save_detailed_results(predictions, performance):
    """保存详细结果"""
    print("\n=== 保存详细结果 ===")
    
    # 保存详细预测结果
    detailed_file = get_project_path('..', 'user_data', 'prophet_v3_detailed_201409.csv')
    predictions.to_csv(detailed_file, index=False)
    
    # 保存性能指标
    performance_file = get_project_path('..', 'user_data', 'prophet_v3_performance.csv')
    performance_df = pd.DataFrame([performance])
    performance_df.to_csv(performance_file, index=False)
    
    print(f"详细预测结果已保存到: {detailed_file}")
    print(f"性能指标已保存到: {performance_file}")


def main():
    """主函数"""
    print("=== 优化版Prophet资金流入流出预测分析 ===")
    print("🎯 本版本特性: 外部变量 + 异常值处理 + 参数优化 + 多重季节性")
    
    try:
        # 1. 加载基础数据
        data_file = get_project_path('..', 'user_data', 'daily_summary.csv')
        df = pd.read_csv(data_file, header=None, names=['date', 'purchase', 'redeem'])
        df['ds'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # 2. 加载外部特征
        interest_df, shibor_df = load_external_features()
        
        # 3. 创建Prophet格式数据
        purchase_df = df[['ds', 'purchase']].copy()
        purchase_df.rename(columns={'purchase': 'y'}, inplace=True)
        redeem_df = df[['ds', 'redeem']].copy()
        redeem_df.rename(columns={'redeem': 'y'}, inplace=True)
        
        print(f"\n数据概况:")
        print(f"- 数据时间范围: {df['ds'].min()} 至 {df['ds'].max()}")
        print(f"- 总天数: {len(df)} 天")
        print(f"- 申购数据平均: ¥{purchase_df['y'].mean():,.0f}")
        print(f"- 赎回数据平均: ¥{redeem_df['y'].mean():,.0f}")
        
        # 4. 训练优化版模型
        global purchase_model, redeem_model
        purchase_model, forecast_purchase, purchase_enhanced, purchase_outliers = \
            train_optimized_prophet_model(purchase_df, "申购", "purchase", interest_df, shibor_df)
        
        redeem_model, forecast_redeem, redeem_enhanced, redeem_outliers = \
            train_optimized_prophet_model(redeem_df, "赎回", "redeem", interest_df, shibor_df)
        
        # 5. 生成优化版预测
        predictions = generate_optimized_predictions(purchase_model, redeem_model, forecast_purchase, forecast_redeem)
        
        # 6. 创建优化版可视化
        create_optimized_visualization(purchase_df, redeem_df, forecast_purchase, forecast_redeem, predictions)
        
        # 7. 分析优化版模型性能
        performance = analyze_optimized_model_performance(forecast_purchase, forecast_redeem, purchase_df, redeem_df)
        
        # 8. 保存详细结果
        save_detailed_results(predictions, performance)
        
        print(f"\n=== 优化版预测完成 ===")
        print(f"✅ 综合优化Prophet模型训练成功")
        print(f"📊 预测结果已保存")
        print(f"📈 可查看文件:")
        print(f"   - 优化版预测结果: prediction_result/prophet_v3_predictions_201409.csv")
        print(f"   - 优化版分析图表: user_data/optimized_prophet_forecast_analysis.png")
        print(f"   - 详细预测数据: user_data/prophet_v3_detailed_201409.csv")
        print(f"   - 性能指标: user_data/prophet_v3_performance.csv")
        print(f"   - 训练好的模型: model/purchase_prophet_v3_model.pkl")
        print(f"                 model/redeem_prophet_v3_model.pkl")
        
        return True
        
    except Exception as e:
        print(f"优化版预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
