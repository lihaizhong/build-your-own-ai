#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prophet预测模型 v6.0 - 最终版
基于v3-v5版本经验教训，回归最成功的配置
版本特性：参考v1/v2的成功参数 + 最小化干预 + 防止过拟合
演进：v3(90分) → v4(过拟合76分) → v5(欠拟合) → v6(平衡)
核心理念：Less is More + 基于成功经验的优化
关键改进：回归v1/v2成功配置 + 轻度数据处理 + 保持简单
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from ...shared import get_project_path

def load_and_prepare_data():
    """加载并准备数据 - 参考v1/v2成功模式"""
    print("=== 加载数据并准备Prophet格式（参考v1/v2成功模式） ===")
    
    # 读取每日汇总数据
    data_file = get_project_path('..', 'user_data', 'daily_summary.csv')
    df = pd.read_csv(data_file, header=None, names=['date', 'purchase', 'redeem'])
    
    # 转换日期格式
    df['ds'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    print(f"数据概况:")
    print(f"- 数据时间范围: {df['ds'].min()} 至 {df['ds'].max()}")
    print(f"- 总天数: {len(df)} 天")
    print(f"- 申购数据平均: ¥{df['purchase'].mean():,.0f}")
    print(f"- 赎回数据平均: ¥{df['redeem'].mean():,.0f}")
    
    return df


def create_china_holidays_v6():
    """创建v6版本节假日（参考v1/v2的成功配置）"""
    print("=== 创建中国节假日（v1/v2成功模式） ===")
    
    holidays = []
    
    # 主要节假日（简化版，参考成功经验）
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
    
    holidays_df = pd.DataFrame(holidays)
    
    print(f"v1/v2成功模式节假日建模完成: {len(holidays_df)} 天")
    
    return holidays_df


def train_final_prophet_model(df, model_name, target_column):
    """训练最终版Prophet模型（参考v1/v2成功参数）"""
    print(f"\n=== 训练{model_name}最终版Prophet模型（v1/v2成功配置） ===")
    
    # 创建节假日
    holidays_df = create_china_holidays_v6()
    
    # 参考v1/v2的成功配置
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        
        # 回到v1/v2的成功参数
        changepoint_prior_scale=0.05,      # 标准敏感度
        seasonality_prior_scale=10.0,      # 标准季节性权重
        holidays_prior_scale=10.0,         # 标准节假日权重
        interval_width=0.95,               # 宽置信区间
        
        # 简化配置
        mcmc_samples=0,
        holidays=holidays_df
    )
    
    # 创建Prophet格式数据
    prophet_df = df[['ds', target_column]].copy()
    prophet_df.rename(columns={target_column: 'y'}, inplace=True)
    
    # 训练模型
    model.fit(prophet_df)
    
    # 创建未来日期
    future = model.make_future_dataframe(periods=30)
    
    # 生成预测
    forecast = model.predict(future)
    
    # 保存模型
    model_path = get_project_path('..', 'model', f'{target_column}_prophet_v6_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"最终版模型已保存到: {model_path}")
    
    return model, forecast


def generate_final_predictions(purchase_model, redeem_model, forecast_purchase, forecast_redeem):
    """生成最终版预测结果"""
    print("\n=== 生成最终版预测结果 ===")
    
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
    predictions['day_name'] = predictions['date'].dt.day_name()
    
    # 计算净流入
    predictions['net_flow'] = predictions['purchase_forecast'] - predictions['redeem_forecast']
    
    # 保存最终版预测结果（考试格式）
    prediction_file = get_project_path('..', 'prediction_result', 'prophet_v6_predictions_201409.csv')
    exam_format = predictions[['date']].copy()
    exam_format['date'] = exam_format['date'].dt.strftime('%Y%m%d')
    exam_format['purchase'] = predictions['purchase_forecast'].round(0).astype(int)
    exam_format['redeem'] = predictions['redeem_forecast'].round(0).astype(int)
    
    exam_format.to_csv(prediction_file, header=False, index=False)
    
    print(f"最终版预测结果已保存到: {prediction_file}")
    
    # 统计预测结果
    total_purchase = predictions['purchase_forecast'].sum()
    total_redeem = predictions['redeem_forecast'].sum()
    net_flow = total_purchase - total_redeem
    
    print(f"\n📊 最终版预测结果统计:")
    print(f"- 总申购预测: ¥{total_purchase:,.0f}")
    print(f"- 总赎回预测: ¥{total_redeem:,.0f}")
    print(f"- 净流入预测: ¥{net_flow:,.0f}")
    print(f"- 平均日申购: ¥{predictions['purchase_forecast'].mean():,.0f}")
    print(f"- 平均日赎回: ¥{predictions['redeem_forecast'].mean():,.0f}")
    
    return predictions


def analyze_final_performance(forecast_purchase, forecast_redeem, purchase_df, redeem_df):
    """分析最终版模型性能"""
    print("\n=== 最终版模型性能分析 ===")
    
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
    
    print(f"最终版申购模型性能:")
    print(f"  MAE: ¥{purchase_mae:,.0f}")
    print(f"  RMSE: ¥{purchase_rmse:,.0f}")
    print(f"  MAPE: {purchase_mape:.2f}%")
    
    print(f"\n最终版赎回模型性能:")
    print(f"  MAE: ¥{redeem_mae:,.0f}")
    print(f"  RMSE: ¥{redeem_rmse:,.0f}")
    print(f"  MAPE: {redeem_mape:.2f}%")
    
    # 与历史版本对比
    try:
        v1_perf = pd.read_csv(get_project_path('..', 'user_data', 'prophet_v1_performance.csv'))
        v2_perf = pd.read_csv(get_project_path('..', 'user_data', 'prophet_v2_performance.csv'))
        v3_perf = pd.read_csv(get_project_path('..', 'user_data', 'prophet_v3_performance.csv'))
        v4_perf = pd.read_csv(get_project_path('..', 'user_data', 'prophet_v4_performance.csv'))
        
        print(f"\n📈 完整版本对比:")
        print(f"申购MAPE: v1({v1_perf['purchase_mape'].iloc[0]:.2f}%) → v2({v2_perf['purchase_mape'].iloc[0]:.2f}%) → v3({v3_perf['purchase_mape'].iloc[0]:.2f}%) → v4({v4_perf['purchase_mape'].iloc[0]:.2f}%) → v6({purchase_mape:.2f}%)")
        print(f"赎回MAPE: v1({v1_perf['redeem_mape'].iloc[0]:.2f}%) → v2({v2_perf['redeem_mape'].iloc[0]:.2f}%) → v3({v3_perf['redeem_mape'].iloc[0]:.2f}%) → v4({v4_perf['redeem_mape'].iloc[0]:.2f}%) → v6({redeem_mape:.2f}%)")
        
    except Exception as e:
        print(f"部分版本性能数据加载失败: {e}")
    
    return {
        'purchase_mae': purchase_mae,
        'purchase_rmse': purchase_rmse,
        'purchase_mape': purchase_mape,
        'redeem_mae': redeem_mae,
        'redeem_rmse': redeem_rmse,
        'redeem_mape': redeem_mape
    }


def save_final_results(predictions, performance):
    """保存最终版详细结果"""
    print("\n=== 保存最终版详细结果 ===")
    
    # 保存详细预测结果
    detailed_file = get_project_path('..', 'user_data', 'prophet_v6_detailed_201409.csv')
    predictions.to_csv(detailed_file, index=False)
    
    # 保存性能指标
    performance_file = get_project_path('..', 'user_data', 'prophet_v6_performance.csv')
    performance_df = pd.DataFrame([performance])
    performance_df.to_csv(performance_file, index=False)
    
    # 保存版本总结
    version_summary = {
        'version': 'prophet_v6',
        'strategy': '回归v1/v2成功配置',
        'key_features': [
            '参考v1/v2的参数配置',
            '简化节假日建模',
            '避免过度工程化',
            '保持Prophet的简洁性'
        ],
        'expected_score': '90-100分',
        'main_improvement': '解决v4过拟合和v5欠拟合问题'
    }
    
    summary_file = get_project_path('..', 'user_data', 'prophet_v6_summary.csv')
    pd.DataFrame([version_summary]).to_csv(summary_file, index=False)
    
    print(f"详细预测结果已保存到: {detailed_file}")
    print(f"性能指标已保存到: {performance_file}")
    print(f"版本总结已保存到: {summary_file}")


def main():
    """主函数 - 最终版，回归成功经验"""
    print("=== 最终版Prophet资金流入流出预测分析 ===")
    print("🎯 最终策略：回归v1/v2成功配置，解决过拟合和欠拟合问题")
    print("💡 核心理念：基于成功经验的平衡优化")
    
    try:
        # 1. 加载数据（参考v1/v2模式）
        df = load_and_prepare_data()
        
        # 2. 创建Prophet格式数据
        purchase_df = df[['ds', 'purchase']].copy()
        purchase_df.rename(columns={'purchase': 'y'}, inplace=True)
        redeem_df = df[['ds', 'redeem']].copy()
        redeem_df.rename(columns={'redeem': 'y'}, inplace=True)
        
        # 3. 训练最终版模型
        global purchase_model, redeem_model
        purchase_model, forecast_purchase = train_final_prophet_model(df, "申购", "purchase")
        redeem_model, forecast_redeem = train_final_prophet_model(df, "赎回", "redeem")
        
        # 4. 生成最终版预测
        predictions = generate_final_predictions(purchase_model, redeem_model, forecast_purchase, forecast_redeem)
        
        # 5. 分析最终版模型性能
        performance = analyze_final_performance(forecast_purchase, forecast_redeem, purchase_df, redeem_df)
        
        # 6. 保存最终版详细结果
        save_final_results(predictions, performance)
        
        print(f"\n=== 最终版预测完成 ===")
        print(f"✅ 回归v1/v2成功配置的最终版模型训练成功")
        print(f"📊 预测结果已保存")
        print(f"🏆 预期解决所有问题，分数回归90+分")
        print(f"📈 可查看文件:")
        print(f"   - 最终版预测结果: prediction_result/prophet_v6_predictions_201409.csv")
        print(f"   - 详细预测数据: user_data/prophet_v6_detailed_201409.csv")
        print(f"   - 性能指标: user_data/prophet_v6_performance.csv")
        print(f"   - 版本总结: user_data/prophet_v6_summary.csv")
        print(f"   - 训练好的模型: model/purchase_prophet_v6_model.pkl")
        print(f"                   model/redeem_prophet_v6_model.pkl")
        
        return True
        
    except Exception as e:
        print(f"最终版预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()