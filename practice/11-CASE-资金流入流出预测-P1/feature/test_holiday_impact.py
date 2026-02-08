#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试中国节假日对Prophet模型性能的影响
对比添加中国节假日前后的模型性能差异
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from ...shared import get_project_path

warnings.filterwarnings('ignore')


def create_china_holidays():
    """创建中国节假日数据框"""
    holidays = [
        # 2013年节假日
        {'holiday': '元旦', 'ds': '2013-01-01'},
        {'holiday': '春节', 'ds': '2013-02-10'},
        {'holiday': '春节', 'ds': '2013-02-11'},
        {'holiday': '春节', 'ds': '2013-02-12'},
        {'holiday': '春节', 'ds': '2013-02-13'},
        {'holiday': '春节', 'ds': '2013-02-14'},
        {'holiday': '春节', 'ds': '2013-02-15'},
        {'holiday': '清明节', 'ds': '2013-04-04'},
        {'holiday': '清明节', 'ds': '2013-04-05'},
        {'holiday': '清明节', 'ds': '2013-04-06'},
        {'holiday': '劳动节', 'ds': '2013-05-01'},
        {'holiday': '端午节', 'ds': '2013-06-12'},
        {'holiday': '中秋节', 'ds': '2013-09-19'},
        {'holiday': '中秋节', 'ds': '2013-09-20'},
        {'holiday': '中秋节', 'ds': '2013-09-21'},
        {'holiday': '国庆节', 'ds': '2013-10-01'},
        {'holiday': '国庆节', 'ds': '2013-10-02'},
        {'holiday': '国庆节', 'ds': '2013-10-03'},
        {'holiday': '国庆节', 'ds': '2013-10-04'},
        {'holiday': '国庆节', 'ds': '2013-10-05'},
        {'holiday': '国庆节', 'ds': '2013-10-06'},
        {'holiday': '国庆节', 'ds': '2013-10-07'},
        
        # 2014年节假日
        {'holiday': '元旦', 'ds': '2014-01-01'},
        {'holiday': '春节', 'ds': '2014-01-31'},
        {'holiday': '春节', 'ds': '2014-02-01'},
        {'holiday': '春节', 'ds': '2014-02-02'},
        {'holiday': '春节', 'ds': '2014-02-03'},
        {'holiday': '春节', 'ds': '2014-02-04'},
        {'holiday': '春节', 'ds': '2014-02-05'},
        {'holiday': '春节', 'ds': '2014-02-06'},
        {'holiday': '清明节', 'ds': '2014-04-05'},
        {'holiday': '清明节', 'ds': '2014-04-06'},
        {'holiday': '清明节', 'ds': '2014-04-07'},
        {'holiday': '劳动节', 'ds': '2014-05-01'},
        {'holiday': '劳动节', 'ds': '2014-05-02'},
        {'holiday': '劳动节', 'ds': '2014-05-03'},
        {'holiday': '端午节', 'ds': '2014-05-31'},
        {'holiday': '端午节', 'ds': '2014-06-01'},
        {'holiday': '端午节', 'ds': '2014-06-02'},
        {'holiday': '中秋节', 'ds': '2014-09-06'},
        {'holiday': '中秋节', 'ds': '2014-09-07'},
        {'holiday': '中秋节', 'ds': '2014-09-08'},
        {'holiday': '国庆节', 'ds': '2014-10-01'},
        {'holiday': '国庆节', 'ds': '2014-10-02'},
        {'holiday': '国庆节', 'ds': '2014-10-03'},
        {'holiday': '国庆节', 'ds': '2014-10-04'},
        {'holiday': '国庆节', 'ds': '2014-10-05'},
        {'holiday': '国庆节', 'ds': '2014-10-06'},
        {'holiday': '国庆节', 'ds': '2014-10-07'},
    ]
    
    return pd.DataFrame(holidays)


def train_model_without_holidays(df, model_name):
    """训练不包含节假日的Prophet模型"""
    print(f"训练{model_name}模型（无节假日）...")
    
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
    
    model.fit(df)
    return model


def train_model_with_holidays(df, model_name):
    """训练包含节假日的Prophet模型"""
    print(f"训练{model_name}模型（包含中国节假日）...")
    
    china_holidays = create_china_holidays()
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        mcmc_samples=0,
        holidays=china_holidays
    )
    
    model.fit(df)
    return model


def load_data():
    """加载并准备数据"""
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
    
    return purchase_df, redeem_df


def calculate_performance_metrics(actual, predicted):
    """计算性能指标"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


def main():
    """主函数 - 对比有无节假日的模型性能"""
    print("=== 中国节假日对Prophet模型性能影响分析 ===\n")
    
    try:
        # 1. 加载数据
        purchase_df, redeem_df = load_data()
        
        print(f"数据时间范围: {purchase_df['ds'].min()} 至 {purchase_df['ds'].max()}")
        print(f"总数据量: {len(purchase_df)} 天\n")
        
        # 2. 训练模型（无节假日）
        print("=== 无节假日模型训练 ===")
        purchase_model_no_holiday = train_model_without_holidays(purchase_df, "申购")
        redeem_model_no_holiday = train_model_without_holidays(redeem_df, "赎回")
        
        # 3. 训练模型（包含节假日）
        print("\n=== 包含节假日模型训练 ===")
        purchase_model_with_holiday = train_model_with_holidays(purchase_df, "申购")
        redeem_model_with_holiday = train_model_with_holidays(redeem_df, "赎回")
        
        # 4. 生成预测
        print("\n=== 生成预测结果 ===")
        
        # 无节假日预测
        future = purchase_model_no_holiday.make_future_dataframe(periods=30)
        forecast_purchase_no_holiday = purchase_model_no_holiday.predict(future)
        forecast_redeem_no_holiday = redeem_model_no_holiday.predict(future)
        
        # 包含节假日预测
        forecast_purchase_with_holiday = purchase_model_with_holiday.predict(future)
        forecast_redeem_with_holiday = redeem_model_with_holiday.predict(future)
        
        # 5. 性能对比分析
        print("\n=== 性能对比分析 ===")
        
        # 分离训练期数据
        train_size = len(purchase_df)
        
        # 申购模型性能对比
        actual_purchase = purchase_df['y']
        pred_purchase_no_holiday = forecast_purchase_no_holiday['yhat'].iloc[:train_size]
        pred_purchase_with_holiday = forecast_purchase_with_holiday['yhat'].iloc[:train_size]
        
        perf_purchase_no_holiday = calculate_performance_metrics(actual_purchase, pred_purchase_no_holiday)
        perf_purchase_with_holiday = calculate_performance_metrics(actual_purchase, pred_purchase_with_holiday)
        
        # 赎回模型性能对比
        actual_redeem = redeem_df['y']
        pred_redeem_no_holiday = forecast_redeem_no_holiday['yhat'].iloc[:train_size]
        pred_redeem_with_holiday = forecast_redeem_with_holiday['yhat'].iloc[:train_size]
        
        perf_redeem_no_holiday = calculate_performance_metrics(actual_redeem, pred_redeem_no_holiday)
        perf_redeem_with_holiday = calculate_performance_metrics(actual_redeem, pred_redeem_with_holiday)
        
        # 6. 打印对比结果
        print("\n📊 申购模型性能对比:")
        print(f"{'指标':<8} {'无节假日':<15} {'包含节假日':<15} {'改进幅度':<10}")
        print("-" * 55)
        
        for metric in ['mae', 'rmse', 'mape']:
            no_holiday = perf_purchase_no_holiday[metric]
            with_holiday = perf_purchase_with_holiday[metric]
            improvement = ((no_holiday - with_holiday) / no_holiday) * 100
            
            if metric == 'mae':
                print(f"{metric.upper():<8} ¥{no_holiday:>12,.0f} ¥{with_holiday:>12,.0f} {improvement:>7.1f}%")
            elif metric == 'rmse':
                print(f"{metric.upper():<8} ¥{no_holiday:>12,.0f} ¥{with_holiday:>12,.0f} {improvement:>7.1f}%")
            else:
                print(f"{metric.upper():<8} {no_holiday:>12.2f}% {with_holiday:>12.2f}% {improvement:>7.1f}%")
        
        print("\n📊 赎回模型性能对比:")
        print(f"{'指标':<8} {'无节假日':<15} {'包含节假日':<15} {'改进幅度':<10}")
        print("-" * 55)
        
        for metric in ['mae', 'rmse', 'mape']:
            no_holiday = perf_redeem_no_holiday[metric]
            with_holiday = perf_redeem_with_holiday[metric]
            improvement = ((no_holiday - with_holiday) / no_holiday) * 100
            
            if metric == 'mae':
                print(f"{metric.upper():<8} ¥{no_holiday:>12,.0f} ¥{with_holiday:>12,.0f} {improvement:>7.1f}%")
            elif metric == 'rmse':
                print(f"{metric.upper():<8} ¥{no_holiday:>12,.0f} ¥{with_holiday:>12,.0f} {improvement:>7.1f}%")
            else:
                print(f"{metric.upper():<8} {no_holiday:>12.2f}% {with_holiday:>12.2f}% {improvement:>7.1f}%")
        
        # 7. 总体评估
        print(f"\n🎯 总体评估:")
        
        # 申购模型总体改进
        purchase_improvement = (
            (perf_purchase_no_holiday['mae'] - perf_purchase_with_holiday['mae']) / perf_purchase_no_holiday['mae'] +
            (perf_purchase_no_holiday['rmse'] - perf_purchase_with_holiday['rmse']) / perf_purchase_no_holiday['rmse'] +
            (perf_purchase_no_holiday['mape'] - perf_purchase_with_holiday['mape']) / perf_purchase_no_holiday['mape']
        ) / 3 * 100
        
        # 赎回模型总体改进
        redeem_improvement = (
            (perf_redeem_no_holiday['mae'] - perf_redeem_with_holiday['mae']) / perf_redeem_no_holiday['mae'] +
            (perf_redeem_no_holiday['rmse'] - perf_redeem_with_holiday['rmse']) / perf_redeem_no_holiday['rmse'] +
            (perf_redeem_no_holiday['mape'] - perf_redeem_with_holiday['mape']) / perf_redeem_no_holiday['mape']
        ) / 3 * 100
        
        print(f"- 申购模型总体改进: {purchase_improvement:+.1f}%")
        print(f"- 赎回模型总体改进: {redeem_improvement:+.1f}%")
        
        if purchase_improvement > 0 and redeem_improvement > 0:
            print("✅ 添加中国节假日显著提升了模型性能！")
        elif purchase_improvement > 0 or redeem_improvement > 0:
            print("⚠️  添加中国节假日对部分模型有改进效果")
        else:
            print("❌ 添加中国节假日对模型性能提升有限")
        
        # 8. 节假日影响分析
        print(f"\n🏮 节假日覆盖分析:")
        china_holidays = create_china_holidays()
        china_holidays['ds'] = pd.to_datetime(china_holidays['ds'])
        
        training_period_start = purchase_df['ds'].min()
        training_period_end = purchase_df['ds'].max()
        
        covered_holidays = china_holidays[
            (china_holidays['ds'] >= training_period_start) & 
            (china_holidays['ds'] <= training_period_end)
        ]
        
        print(f"- 训练期内覆盖的节假日: {len(covered_holidays)} 天")
        print(f"- 主要节假日类型: {covered_holidays['holiday'].value_counts().to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()