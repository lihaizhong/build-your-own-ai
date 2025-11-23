#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prophet预测模型 v2.0 - 中国节假日+周末效应版本
基于Prophet算法，包含中国节假日和周末效应优化
版本特性：节假日+周末效应建模，预测周末效应-29.6%/-19.5%
实验版本：探索周末显式建模效果，性能与v1相近
演进：从v1基础版添加节假日+周末效应
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def get_project_path(*paths):
    """获取项目路径的统一方法"""
    import os
    try:
        return os.path.join(os.path.dirname(__file__), *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)


def create_china_holidays_with_weekends():
    """创建包含中国节假日和周末的数据框"""
    holidays = []
    
    # 2013-2014年中国主要节假日
    main_holidays = [
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
    
    holidays.extend(main_holidays)
    
    # 添加训练期间的所有周末（周六周日）
    # 基于数据分析结果：周末效应显著，应在模型中显式处理
    start_date = datetime(2013, 7, 1)
    end_date = datetime(2014, 8, 31)
    
    current_date = start_date
    while current_date <= end_date:
        # 如果是周六(5)或周日(6)
        if current_date.weekday() in [5, 6]:
            weekend_name = '周六' if current_date.weekday() == 5 else '周日'
            holidays.append({
                'holiday': f'周末-{weekend_name}',
                'ds': current_date.strftime('%Y-%m-%d')
            })
        current_date += timedelta(days=1)
    
    return pd.DataFrame(holidays)


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


def train_enhanced_prophet_model(df, model_name, target_column):
    """训练增强版Prophet模型（包含中国节假日+周末）"""
    print(f"\n=== 训练{model_name}增强版Prophet模型（中国节假日+周末效应） ===")
    
    # 创建包含节假日和周末的数据框
    enhanced_holidays = create_china_holidays_with_weekends()
    
    print(f"- 节假日总数: {len(enhanced_holidays)} 天")
    print(f"- 主要节假日: {len([h for h in enhanced_holidays['holiday'] if not h.startswith('周末')])} 天")
    print(f"- 周末天数: {len([h for h in enhanced_holidays['holiday'] if h.startswith('周末')])} 天")
    
    # 创建增强版Prophet模型
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        mcmc_samples=0,
        holidays=enhanced_holidays  # 添加包含周末的节假日
    )
    
    # 训练模型
    model.fit(df)
    
    # 创建未来日期
    future = model.make_future_dataframe(periods=30)  # 预测未来30天
    
    # 生成预测
    forecast = model.predict(future)
    
    # 保存模型
    model_path = get_project_path('..', 'model', f'{target_column}_prophet_v2_model.pkl')
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"增强版模型已保存到: {model_path}")
    
    return model, forecast


def generate_enhanced_predictions(purchase_model, redeem_model, forecast_purchase, forecast_redeem):
    """生成增强版预测结果"""
    print("\n=== 生成增强版预测结果 ===")
    
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
    
    # 标记预测期间的周末
    predictions['weekday'] = predictions['date'].dt.dayofweek
    predictions['is_weekend'] = predictions['weekday'].isin([5, 6])
    predictions['day_name'] = predictions['date'].dt.day_name()
    
    # 保存增强版预测结果（考试格式）
    prediction_file = get_project_path('..', 'prediction_result', 'prophet_v2_predictions_201409.csv')
    exam_format = predictions[['date']].copy()
    exam_format['date'] = exam_format['date'].dt.strftime('%Y%m%d')
    exam_format['purchase'] = predictions['purchase_forecast'].round(0).astype(int)
    exam_format['redeem'] = predictions['redeem_forecast'].round(0).astype(int)
    
    # 保存为CSV（考试格式：YYYYMMDD,申购金额,赎回金额）
    exam_format.to_csv(prediction_file, header=False, index=False)
    
    print(f"增强版预测结果已保存到: {prediction_file}")
    print(f"预测期间: {predictions['date'].min()} 至 {predictions['date'].max()}")
    print(f"预测平均申购额: ¥{predictions['purchase_forecast'].mean():,.0f}")
    print(f"预测平均赎回额: ¥{predictions['redeem_forecast'].mean():,.0f}")
    
    # 周末vs工作日预测分析
    weekend_predictions = predictions[predictions['is_weekend']]
    workday_predictions = predictions[~predictions['is_weekend']]
    
    if len(weekend_predictions) > 0 and len(workday_predictions) > 0:
        print(f"\n🏮 预测期间周末效应分析:")
        print(f"- 工作日预测平均申购: ¥{workday_predictions['purchase_forecast'].mean():,.0f}")
        print(f"- 周末预测平均申购: ¥{weekend_predictions['purchase_forecast'].mean():,.0f}")
        print(f"- 工作日预测平均赎回: ¥{workday_predictions['redeem_forecast'].mean():,.0f}")
        print(f"- 周末预测平均赎回: ¥{weekend_predictions['redeem_forecast'].mean():,.0f}")
        
        weekend_purchase_effect = ((weekend_predictions['purchase_forecast'].mean() - 
                                  workday_predictions['purchase_forecast'].mean()) / 
                                 workday_predictions['purchase_forecast'].mean()) * 100
        weekend_redeem_effect = ((weekend_predictions['redeem_forecast'].mean() - 
                                workday_predictions['redeem_forecast'].mean()) / 
                               workday_predictions['redeem_forecast'].mean()) * 100
        
        print(f"- 模型预测的周末申购效应: {weekend_purchase_effect:+.1f}%")
        print(f"- 模型预测的周末赎回效应: {weekend_redeem_effect:+.1f}%")
    
    return predictions


def create_enhanced_visualization(purchase_df, redeem_df, forecast_purchase, forecast_redeem, predictions):
    """创建增强版可视化图表"""
    print("\n=== 生成增强版可视化图表 ===")
    
    # 创建增强版分析图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('增强版Prophet时间序列预测分析 (中国节假日+周末效应)', fontsize=16, fontweight='bold')
    
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
    
    # 标记预测期间的周末
    weekend_dates = predictions[predictions['is_weekend']]['date']
    for date in weekend_dates:
        ax1.axvline(x=date, color='orange', alpha=0.3, linestyle='--')
    
    ax1.set_title('申购金额预测趋势（橙色虚线标记周末）')
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
    
    # 标记预测期间的周末
    for date in weekend_dates:
        ax2.axvline(x=date, color='red', alpha=0.3, linestyle='--')
    
    ax2.set_title('赎回金额预测趋势（橙色虚线标记周末）')
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
    chart_file = get_project_path('..', 'user_data', 'enhanced_prophet_forecast_analysis.png')
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"增强版可视化图表已保存到: {chart_file}")
    
    # 创建增强版对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('增强版预测对比分析（含周末效应）', fontsize=14, fontweight='bold')
    
    # 上图：预测期间的申购赎回对比
    pred_dates = pd.to_datetime(predictions['date'])
    ax1.plot(pred_dates, predictions['purchase_forecast'], 'r-', linewidth=2, label='预测申购额')
    ax1.plot(pred_dates, predictions['redeem_forecast'], 'b-', linewidth=2, label='预测赎回额')
    
    # 标记周末
    weekend_mask = predictions['is_weekend']
    ax1.scatter(pred_dates[weekend_mask], predictions.loc[weekend_mask, 'purchase_forecast'], 
               color='red', s=50, alpha=0.7, marker='s', label='周末申购')
    ax1.scatter(pred_dates[weekend_mask], predictions.loc[weekend_mask, 'redeem_forecast'], 
               color='blue', s=50, alpha=0.7, marker='s', label='周末赎回')
    
    ax1.set_title('未来30天申购赎回预测（含周末标记）')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('金额')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：净流入分析
    net_flow = predictions['purchase_forecast'] - predictions['redeem_forecast']
    ax2.plot(pred_dates, net_flow, 'g-', linewidth=2, label='净流入')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 标记周末的净流入
    ax2.scatter(pred_dates[weekend_mask], net_flow[weekend_mask], 
               color='orange', s=50, alpha=0.7, marker='s', label='周末净流入')
    
    ax2.set_title('预测期间净流入分析（含周末效应）')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('净流入')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_file = get_project_path('..', 'user_data', 'enhanced_prophet_forecast_comparison.png')
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"增强版对比图表已保存到: {comparison_file}")


def analyze_enhanced_model_performance(forecast_purchase, forecast_redeem, purchase_df, redeem_df):
    """分析增强版模型性能"""
    print("\n=== 增强版模型性能分析 ===")
    
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
    
    print(f"增强版申购模型性能:")
    print(f"  MAE: ¥{purchase_mae:,.0f}")
    print(f"  RMSE: ¥{purchase_rmse:,.0f}")
    print(f"  MAPE: {purchase_mape:.2f}%")
    
    print(f"\n增强版赎回模型性能:")
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
    print("=== 增强版Prophet资金流入流出预测分析 ===")
    print("🎯 本版本包含：中国节假日 + 周末效应优化")
    
    try:
        # 1. 加载数据
        purchase_df, redeem_df = load_and_prepare_data()
        
        # 2. 训练增强版模型
        global purchase_model, redeem_model
        purchase_model, forecast_purchase = train_enhanced_prophet_model(purchase_df, "申购", "purchase")
        redeem_model, forecast_redeem = train_enhanced_prophet_model(redeem_df, "赎回", "redeem")
        
        # 3. 生成增强版预测
        predictions = generate_enhanced_predictions(purchase_model, redeem_model, forecast_purchase, forecast_redeem)
        
        # 4. 分析增强版模型性能
        performance = analyze_enhanced_model_performance(forecast_purchase, forecast_redeem, purchase_df, redeem_df)
        
        # 5. 创建增强版可视化
        create_enhanced_visualization(purchase_df, redeem_df, forecast_purchase, forecast_redeem, predictions)
        
        print(f"\n=== 增强版预测完成 ===")
        print(f"✅ 包含中国节假日和周末效应的增强版模型训练成功")
        print(f"📊 预测结果已保存")
        print(f"📈 可查看文件:")
        print(f"   - 增强版预测结果: prediction_result/prophet_v2_predictions_201409.csv")
        print(f"   - 增强版分析图表: user_data/enhanced_prophet_forecast_analysis.png")
        print(f"   - 增强版对比图表: user_data/enhanced_prophet_forecast_comparison.png")
        print(f"   - 训练好的模型: model/purchase_prophet_v2_model.pkl")
        print(f"                 model/redeem_prophet_v2_model.pkl")
        
        return True
        
    except Exception as e:
        print(f"增强版预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
