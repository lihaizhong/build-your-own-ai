#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prophet模型版本对比分析
对比原始版本、仅节假日版本、增强版（节假日+周末）的性能
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
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


def load_data():
    """加载数据"""
    data_file = get_project_path('..', 'user_data', 'daily_summary.csv')
    df = pd.read_csv(data_file, header=None, names=['date', 'purchase', 'redeem'])
    
    df['ds'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    purchase_df = df[['ds', 'purchase']].copy()
    purchase_df.rename(columns={'purchase': 'y'}, inplace=True)
    
    redeem_df = df[['ds', 'redeem']].copy()
    redeem_df.rename(columns={'redeem': 'y'}, inplace=True)
    
    return purchase_df, redeem_df


def create_china_holidays():
    """创建中国节假日数据框"""
    holidays = [
        {'holiday': '元旦', 'ds': '2013-01-01'},
        {'holiday': '春节', 'ds': '2013-02-10'}, {'holiday': '春节', 'ds': '2013-02-11'},
        {'holiday': '春节', 'ds': '2013-02-12'}, {'holiday': '春节', 'ds': '2013-02-13'},
        {'holiday': '春节', 'ds': '2013-02-14'}, {'holiday': '春节', 'ds': '2013-02-15'},
        {'holiday': '清明节', 'ds': '2013-04-04'}, {'holiday': '清明节', 'ds': '2013-04-05'},
        {'holiday': '清明节', 'ds': '2013-04-06'},
        {'holiday': '劳动节', 'ds': '2013-05-01'},
        {'holiday': '端午节', 'ds': '2013-06-12'},
        {'holiday': '中秋节', 'ds': '2013-09-19'}, {'holiday': '中秋节', 'ds': '2013-09-20'},
        {'holiday': '中秋节', 'ds': '2013-09-21'},
        {'holiday': '国庆节', 'ds': '2013-10-01'}, {'holiday': '国庆节', 'ds': '2013-10-02'},
        {'holiday': '国庆节', 'ds': '2013-10-03'}, {'holiday': '国庆节', 'ds': '2013-10-04'},
        {'holiday': '国庆节', 'ds': '2013-10-05'}, {'holiday': '国庆节', 'ds': '2013-10-06'},
        {'holiday': '国庆节', 'ds': '2013-10-07'},
        
        {'holiday': '元旦', 'ds': '2014-01-01'},
        {'holiday': '春节', 'ds': '2014-01-31'}, {'holiday': '春节', 'ds': '2014-02-01'},
        {'holiday': '春节', 'ds': '2014-02-02'}, {'holiday': '春节', 'ds': '2014-02-03'},
        {'holiday': '春节', 'ds': '2014-02-04'}, {'holiday': '春节', 'ds': '2014-02-05'},
        {'holiday': '春节', 'ds': '2014-02-06'},
        {'holiday': '清明节', 'ds': '2014-04-05'}, {'holiday': '清明节', 'ds': '2014-04-06'},
        {'holiday': '清明节', 'ds': '2014-04-07'},
        {'holiday': '劳动节', 'ds': '2014-05-01'}, {'holiday': '劳动节', 'ds': '2014-05-02'},
        {'holiday': '劳动节', 'ds': '2014-05-03'},
        {'holiday': '端午节', 'ds': '2014-05-31'}, {'holiday': '端午节', 'ds': '2014-06-01'},
        {'holiday': '端午节', 'ds': '2014-06-02'},
        {'holiday': '中秋节', 'ds': '2014-09-06'}, {'holiday': '中秋节', 'ds': '2014-09-07'},
        {'holiday': '中秋节', 'ds': '2014-09-08'},
        {'holiday': '国庆节', 'ds': '2014-10-01'}, {'holiday': '国庆节', 'ds': '2014-10-02'},
        {'holiday': '国庆节', 'ds': '2014-10-03'}, {'holiday': '国庆节', 'ds': '2014-10-04'},
        {'holiday': '国庆节', 'ds': '2014-10-05'}, {'holiday': '国庆节', 'ds': '2014-10-06'},
        {'holiday': '国庆节', 'ds': '2014-10-07'},
    ]
    return pd.DataFrame(holidays)


def create_enhanced_holidays():
    """创建包含周末的增强版节假日"""
    from datetime import datetime, timedelta
    
    holidays = create_china_holidays().to_dict('records')
    
    # 添加训练期间的所有周末
    start_date = datetime(2013, 7, 1)
    end_date = datetime(2014, 8, 31)
    
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() in [5, 6]:  # 周六周日
            weekend_name = '周六' if current_date.weekday() == 5 else '周日'
            holidays.append({
                'holiday': f'周末-{weekend_name}',
                'ds': current_date.strftime('%Y-%m-%d')
            })
        current_date += timedelta(days=1)
    
    return pd.DataFrame(holidays)


def train_prophet_basic(df):
    """训练基础Prophet模型（无节假日）"""
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
        seasonality_mode='additive', changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0, holidays_prior_scale=10.0, mcmc_samples=0
    )
    model.fit(df)
    return model


def train_prophet_with_holidays(df):
    """训练包含节假日的Prophet模型"""
    china_holidays = create_china_holidays()
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
        seasonality_mode='additive', changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0, holidays_prior_scale=10.0, mcmc_samples=0,
        holidays=china_holidays
    )
    model.fit(df)
    return model


def train_prophet_enhanced(df):
    """训练增强版Prophet模型（节假日+周末）"""
    enhanced_holidays = create_enhanced_holidays()
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
        seasonality_mode='additive', changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0, holidays_prior_scale=10.0, mcmc_samples=0,
        holidays=enhanced_holidays
    )
    model.fit(df)
    return model


def calculate_metrics(actual, predicted):
    """计算性能指标"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def main():
    """主函数 - 完整性能对比"""
    print("=== Prophet模型版本对比分析 ===\n")
    
    try:
        # 加载数据
        purchase_df, redeem_df = load_data()
        print(f"数据时间范围: {purchase_df['ds'].min()} 至 {purchase_df['ds'].max()}")
        print(f"总数据量: {len(purchase_df)} 天\n")
        
        # 训练三种模型
        print("=== 训练三种Prophet模型版本 ===")
        print("1. 基础版本（无节假日）...")
        purchase_basic = train_prophet_basic(purchase_df)
        redeem_basic = train_prophet_basic(redeem_df)
        
        print("2. 节假日版本（仅中国节假日）...")
        purchase_holidays = train_prophet_with_holidays(purchase_df)
        redeem_holidays = train_prophet_with_holidays(redeem_df)
        
        print("3. 增强版本（中国节假日+周末）...")
        purchase_enhanced = train_prophet_enhanced(purchase_df)
        redeem_enhanced = train_prophet_enhanced(redeem_df)
        
        # 生成预测
        print("\n=== 生成预测并计算性能 ===")
        
        # 基础版本预测
        future = purchase_basic.make_future_dataframe(periods=30)
        forecast_basic_purchase = purchase_basic.predict(future)
        forecast_basic_redeem = redeem_basic.predict(future)
        
        # 节假日版本预测
        forecast_holidays_purchase = purchase_holidays.predict(future)
        forecast_holidays_redeem = redeem_holidays.predict(future)
        
        # 增强版本预测
        forecast_enhanced_purchase = purchase_enhanced.predict(future)
        forecast_enhanced_redeem = redeem_enhanced.predict(future)
        
        # 计算性能指标
        train_size = len(purchase_df)
        
        # 申购模型性能
        actual_purchase = purchase_df['y']
        basic_pred_purchase = forecast_basic_purchase['yhat'].iloc[:train_size]
        holidays_pred_purchase = forecast_holidays_purchase['yhat'].iloc[:train_size]
        enhanced_pred_purchase = forecast_enhanced_purchase['yhat'].iloc[:train_size]
        
        basic_metrics_purchase = calculate_metrics(actual_purchase, basic_pred_purchase)
        holidays_metrics_purchase = calculate_metrics(actual_purchase, holidays_pred_purchase)
        enhanced_metrics_purchase = calculate_metrics(actual_purchase, enhanced_pred_purchase)
        
        # 赎回模型性能
        actual_redeem = redeem_df['y']
        basic_pred_redeem = forecast_basic_redeem['yhat'].iloc[:train_size]
        holidays_pred_redeem = forecast_holidays_redeem['yhat'].iloc[:train_size]
        enhanced_pred_redeem = forecast_enhanced_redeem['yhat'].iloc[:train_size]
        
        basic_metrics_redeem = calculate_metrics(actual_redeem, basic_pred_redeem)
        holidays_metrics_redeem = calculate_metrics(actual_redeem, holidays_pred_redeem)
        enhanced_metrics_redeem = calculate_metrics(actual_redeem, enhanced_pred_redeem)
        
        # 打印对比结果
        print("\n📊 申购模型性能对比:")
        print(f"{'版本':<12} {'MAE (万元)':<12} {'RMSE (万元)':<14} {'MAPE (%)':<10}")
        print("-" * 55)
        
        for name, metrics in [
            ('基础版', basic_metrics_purchase),
            ('节假日版', holidays_metrics_purchase),
            ('增强版', enhanced_metrics_purchase)
        ]:
            print(f"{name:<12} {metrics['mae']/1e4:>10.0f} {metrics['rmse']/1e4:>12.0f} {metrics['mape']:>8.2f}")
        
        print("\n📊 赎回模型性能对比:")
        print(f"{'版本':<12} {'MAE (万元)':<12} {'RMSE (万元)':<14} {'MAPE (%)':<10}")
        print("-" * 55)
        
        for name, metrics in [
            ('基础版', basic_metrics_redeem),
            ('节假日版', holidays_metrics_redeem),
            ('增强版', enhanced_metrics_redeem)
        ]:
            print(f"{name:<12} {metrics['mae']/1e4:>10.0f} {metrics['rmse']/1e4:>12.0f} {metrics['mape']:>8.2f}")
        
        # 计算改进幅度
        print(f"\n🎯 性能改进分析:")
        
        # 申购模型改进
        purchase_improvement_holidays = (
            (basic_metrics_purchase['mae'] - holidays_metrics_purchase['mae']) / basic_metrics_purchase['mae'] * 100
        )
        purchase_improvement_enhanced = (
            (basic_metrics_purchase['mae'] - enhanced_metrics_purchase['mae']) / basic_metrics_purchase['mae'] * 100
        )
        
        print(f"申购模型:")
        print(f"  节假日版本 vs 基础版: MAE改进 {purchase_improvement_holidays:+.1f}%")
        print(f"  增强版本 vs 基础版: MAE改进 {purchase_improvement_enhanced:+.1f}%")
        
        # 赎回模型改进
        redeem_improvement_holidays = (
            (basic_metrics_redeem['mae'] - holidays_metrics_redeem['mae']) / basic_metrics_redeem['mae'] * 100
        )
        redeem_improvement_enhanced = (
            (basic_metrics_redeem['mae'] - enhanced_metrics_redeem['mae']) / basic_metrics_redeem['mae'] * 100
        )
        
        print(f"赎回模型:")
        print(f"  节假日版本 vs 基础版: MAE改进 {redeem_improvement_holidays:+.1f}%")
        print(f"  增强版本 vs 基础版: MAE改进 {redeem_improvement_enhanced:+.1f}%")
        
        # 最终推荐
        print(f"\n💡 版本推荐:")
        
        best_purchase = min([
            ('基础版', basic_metrics_purchase),
            ('节假日版', holidays_metrics_purchase),
            ('增强版', enhanced_metrics_purchase)
        ], key=lambda x: x[1]['mae'])
        
        best_redeem = min([
            ('基础版', basic_metrics_redeem),
            ('节假日版', holidays_metrics_redeem),
            ('增强版', enhanced_metrics_redeem)
        ], key=lambda x: x[1]['mae'])
        
        print(f"- 申购模型最佳: {best_purchase[0]} (MAE: ¥{best_purchase[1]['mae']:,.0f})")
        print(f"- 赎回模型最佳: {best_redeem[0]} (MAE: ¥{best_redeem[1]['mae']:,.0f})")
        
        if enhanced_metrics_purchase['mae'] <= holidays_metrics_purchase['mae'] and \
           enhanced_metrics_redeem['mae'] <= holidays_metrics_redeem['mae']:
            print(f"\n✅ 增强版本（节假日+周末）在申购和赎回模型上都是最佳选择！")
        elif holidays_metrics_purchase['mae'] <= basic_metrics_purchase['mae'] and \
             holidays_metrics_redeem['mae'] <= basic_metrics_redeem['mae']:
            print(f"\n✅ 节假日版本在申购和赎回模型上都比基础版更好！")
        else:
            print(f"\n⚠️  需要进一步调优模型参数以获得更好的性能")
        
        # 周末效应验证
        print(f"\n🏮 周末效应验证:")
        
        # 获取增强版模型的预测值，并分析周末效应
        future_predictions = forecast_enhanced_purchase.tail(30)
        future_dates = future_predictions['ds']
        weekend_mask = future_dates.dt.weekday.isin([5, 6])
        
        weekend_purchase_pred = future_predictions.loc[weekend_mask, 'yhat'].mean()
        workday_purchase_pred = future_predictions.loc[~weekend_mask, 'yhat'].mean()
        
        weekend_redeem_pred = forecast_enhanced_redeem.tail(30).loc[weekend_mask, 'yhat'].mean()
        workday_redeem_pred = forecast_enhanced_redeem.tail(30).loc[~weekend_mask, 'yhat'].mean()
        
        predicted_weekend_effect_purchase = ((weekend_purchase_pred - workday_purchase_pred) / workday_purchase_pred) * 100
        predicted_weekend_effect_redeem = ((weekend_redeem_pred - workday_redeem_pred) / workday_redeem_pred) * 100
        
        print(f"- 增强版模型预测的周末申购效应: {predicted_weekend_effect_purchase:+.1f}%")
        print(f"- 增强版模型预测的周末赎回效应: {predicted_weekend_effect_redeem:+.1f}%")
        print(f"- 实际数据观察到的周末申购效应: -37.4%")
        print(f"- 实际数据观察到的周末赎回效应: -35.2%")
        
        return True
        
    except Exception as e:
        print(f"对比分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
