#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prophet预测模型 v5.0 - 防过拟合版本
基于v4过拟合问题分析，重新校准模型复杂度
版本特性：回归基础配置 + 精准特征选择 + 防止过拟合
演进：从v4过度简化回归理性平衡
核心理念：找到复杂度的最佳平衡点，避免欠拟合和过拟合
关键改进：恢复关键参数 + 最小化预处理 + 特征重要性分析
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


def load_raw_data():
    """加载原始数据 - 最小化预处理"""
    print("=== 加载原始数据（最小化预处理） ===")
    
    # 读取每日汇总数据
    data_file = get_project_path('..', 'user_data', 'daily_summary.csv')
    df = pd.read_csv(data_file, header=None, names=['date', 'purchase', 'redeem'])
    df['ds'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    print(f"原始数据概况:")
    print(f"- 数据时间范围: {df['ds'].min()} 至 {df['ds'].max()}")
    print(f"- 总天数: {len(df)} 天")
    
    return df


def minimal_outlier_handling(data, column, method='extreme_only'):
    """最小化异常值处理 - 只处理极端异常值"""
    print(f"=== 最小化异常值处理{column}（仅处理极端值） ===")
    
    if method == 'extreme_only':
        # 只处理极端异常值（超过5σ）
        mean_val = data[column].mean()
        std_val = data[column].std()
        
        # 定义极端异常值阈值
        lower_extreme = mean_val - 5 * std_val
        upper_extreme = mean_val + 5 * std_val
        
        # 只替换真正的极端值
        extreme_mask = (data[column] < lower_extreme) | (data[column] > upper_extreme)
        extreme_count = extreme_mask.sum()
        
        print(f"极端异常值: {extreme_count} 个")
        
        if extreme_count > 0:
            # 用99%和1%分位数替换
            p99 = data[column].quantile(0.99)
            p1 = data[column].quantile(0.01)
            
            too_high = data[column] > upper_extreme
            too_low = data[column] < lower_extreme
            
            data.loc[too_high, column] = p99
            data.loc[too_low, column] = p1
            
            print(f"极端异常值已替换")
    
    return data


def create_simple_holidays():
    """创建简化节假日 - 只保留主要节假日"""
    print("=== 创建简化节假日（主要节假日） ===")
    
    holidays = []
    
    # 只保留最重要的节假日（避免过度建模）
    main_holidays = [
        # 春节（影响最大）
        {'holiday': '春节', 'ds': '2013-02-10', 'lower_window': -1, 'upper_window': 2},
        {'holiday': '春节', 'ds': '2014-01-31', 'lower_window': -1, 'upper_window': 2},
        
        # 国庆节
        {'holiday': '国庆节', 'ds': '2013-10-01', 'lower_window': 0, 'upper_window': 6},
        {'holiday': '国庆节', 'ds': '2014-10-01', 'lower_window': 0, 'upper_window': 6},
        
        # 劳动节
        {'holiday': '劳动节', 'ds': '2013-05-01', 'lower_window': 0, 'upper_window': 2},
        {'holiday': '劳动节', 'ds': '2014-05-01', 'lower_window': 0, 'upper_window': 2},
        
        # 清明节
        {'holiday': '清明节', 'ds': '2013-04-04', 'lower_window': 0, 'upper_window': 2},
        {'holiday': '清明节', 'ds': '2014-04-05', 'lower_window': 0, 'upper_window': 2},
        
        # 其他重要节假日
        {'holiday': '元旦', 'ds': '2013-01-01', 'lower_window': 0, 'upper_window': 0},
        {'holiday': '元旦', 'ds': '2014-01-01', 'lower_window': 0, 'upper_window': 0},
    ]
    
    holidays.extend(main_holidays)
    holidays_df = pd.DataFrame(holidays)
    
    print(f"简化节假日建模完成: {len(holidays_df)} 天")
    
    return holidays_df


def feature_importance_analysis(df):
    """特征重要性分析 - 识别真正重要的特征"""
    print("=== 特征重要性分析 ===")
    
    # 创建基础特征
    features = {
        'basic': ['ds'],  # 最基础的时间特征
        'with_regressors': ['ds', 'purchase_lag1', 'purchase_lag2'],  # 添加简单滞后特征
        'with_external': ['ds', 'purchase_lag1', 'purchase_ma7']     # 添加移动平均
    }
    
    # 测试不同特征组合的性能
    results = {}
    
    for feature_name, feature_cols in features.items():
        try:
            print(f"测试特征组合: {feature_name}")
            
            # 准备数据
            test_data = df[['ds', 'purchase']].copy()
            test_data.rename(columns={'purchase': 'y'}, inplace=True)
            
            # 添加简单特征
            if feature_name == 'with_regressors':
                test_data['purchase_lag1'] = test_data['y'].shift(1)
                test_data['purchase_lag2'] = test_data['y'].shift(2)
                
            elif feature_name == 'with_external':
                test_data['purchase_lag1'] = test_data['y'].shift(1)
                test_data['purchase_ma7'] = test_data['y'].rolling(7).mean()
            
            # 创建简单模型
            simple_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,  # 回到标准参数
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                interval_width=0.95
            )
            
            # 训练模型
            simple_model.fit(test_data.iloc[:-30])  # 留最后30天验证
            
            # 预测验证
            future = simple_model.make_future_dataframe(periods=30)
            forecast = simple_model.predict(future)
            predictions = forecast['yhat'].iloc[-30:]
            actual = test_data['y'].iloc[-30:]
            
            # 计算MAPE
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            results[feature_name] = mape
            
            print(f"  {feature_name} MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"  {feature_name} 测试失败: {e}")
            results[feature_name] = float('inf')
    
    # 选择最佳特征组合
    best_feature = min(results, key=results.get)
    print(f"\n最佳特征组合: {best_feature} (MAPE: {results[best_feature]:.2f}%)")
    
    return best_feature, results


def train_balanced_prophet_model(df, model_name, target_column):
    """训练平衡版Prophet模型"""
    print(f"\n=== 训练{model_name}平衡版Prophet模型（防过拟合） ===")
    
    # 最小化异常值处理
    processed_data = minimal_outlier_handling(df.copy(), target_column)
    
    # 创建Prophet格式数据
    prophet_df = processed_data[['ds', target_column]].copy()
    prophet_df.rename(columns={target_column: 'y'}, inplace=True)
    
    # 特征重要性分析
    best_feature, feature_results = feature_importance_analysis(processed_data)
    
    # 创建简化节假日
    holidays_df = create_simple_holidays()
    
    # 平衡的Prophet配置（参数适中，不过度简化也不过度复杂）
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        
        # 平衡的参数配置
        changepoint_prior_scale=0.05,    # 适中敏感度
        seasonality_prior_scale=10,      # 标准季节性权重
        holidays_prior_scale=10,         # 标准节假日权重
        interval_width=0.95,             # 宽置信区间
        
        # 简化配置
        mcmc_samples=0,
        uncertainty_samples=1000,
        holidays=holidays_df
    )
    
    # 只在确定有益的情况下添加回归变量
    if best_feature == 'with_regressors':
        model.add_regressor('purchase_lag1')
        model.add_regressor('purchase_lag2')
        print("添加滞后特征回归变量")
    elif best_feature == 'with_external':
        model.add_regressor('purchase_lag1')
        model.add_regressor('purchase_ma7')
        print("添加滞后+移动平均特征")
    
    # 训练模型
    model.fit(prophet_df)
    
    # 创建未来日期
    future = model.make_future_dataframe(periods=30)
    
    # 为未来日期添加回归变量
    if best_feature in ['with_regressors', 'with_external']:
        # 使用最后已知值填充
        future['purchase_lag1'] = prophet_df['y'].iloc[-1]
        if best_feature == 'with_regressors':
            future['purchase_lag2'] = prophet_df['y'].iloc[-2] if len(prophet_df) >= 2 else prophet_df['y'].iloc[-1]
        elif best_feature == 'with_external':
            future['purchase_ma7'] = prophet_df['y'].tail(7).mean()
    
    # 生成预测
    forecast = model.predict(future)
    
    # 保存模型
    model_path = get_project_path('..', 'model', f'{target_column}_prophet_v5_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"平衡版模型已保存到: {model_path}")
    print(f"特征重要性分析结果: {feature_results}")
    
    return model, forecast


def generate_balanced_predictions(purchase_model, redeem_model, forecast_purchase, forecast_redeem):
    """生成平衡版预测结果"""
    print("\n=== 生成平衡版预测结果 ===")
    
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
    
    # 保存平衡版预测结果（考试格式）
    prediction_file = get_project_path('..', 'prediction_result', 'prophet_v5_predictions_201409.csv')
    exam_format = predictions[['date']].copy()
    exam_format['date'] = exam_format['date'].dt.strftime('%Y%m%d')
    exam_format['purchase'] = predictions['purchase_forecast'].round(0).astype(int)
    exam_format['redeem'] = predictions['redeem_forecast'].round(0).astype(int)
    
    exam_format.to_csv(prediction_file, header=False, index=False)
    
    print(f"平衡版预测结果已保存到: {prediction_file}")
    
    # 统计预测结果
    total_purchase = predictions['purchase_forecast'].sum()
    total_redeem = predictions['redeem_forecast'].sum()
    net_flow = total_purchase - total_redeem
    
    print(f"\n📊 平衡版预测结果统计:")
    print(f"- 总申购预测: ¥{total_purchase:,.0f}")
    print(f"- 总赎回预测: ¥{total_redeem:,.0f}")
    print(f"- 净流入预测: ¥{net_flow:,.0f}")
    print(f"- 平均日申购: ¥{predictions['purchase_forecast'].mean():,.0f}")
    print(f"- 平均日赎回: ¥{predictions['redeem_forecast'].mean():,.0f}")
    
    return predictions


def analyze_balanced_performance(forecast_purchase, forecast_redeem, purchase_df, redeem_df):
    """分析平衡版模型性能"""
    print("\n=== 平衡版模型性能分析 ===")
    
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
    
    print(f"平衡版申购模型性能:")
    print(f"  MAE: ¥{purchase_mae:,.0f}")
    print(f"  RMSE: ¥{purchase_rmse:,.0f}")
    print(f"  MAPE: {purchase_mape:.2f}%")
    
    print(f"\n平衡版赎回模型性能:")
    print(f"  MAE: ¥{redeem_mae:,.0f}")
    print(f"  RMSE: ¥{redeem_rmse:,.0f}")
    print(f"  MAPE: {redeem_mape:.2f}%")
    
    # 与v3、v4版本对比
    try:
        v3_perf = pd.read_csv(get_project_path('..', 'user_data', 'prophet_v3_performance.csv'))
        v4_perf = pd.read_csv(get_project_path('..', 'user_data', 'prophet_v4_performance.csv'))
        
        print(f"\n📈 版本对比:")
        print(f"申购MAPE: v3({v3_perf['purchase_mape'].iloc[0]:.2f}%) → v4({v4_perf['purchase_mape'].iloc[0]:.2f}%) → v5({purchase_mape:.2f}%)")
        print(f"赎回MAPE: v3({v3_perf['redeem_mape'].iloc[0]:.2f}%) → v4({v4_perf['redeem_mape'].iloc[0]:.2f}%) → v5({redeem_mape:.2f}%)")
        
    except Exception as e:
        print(f"无法加载历史版本性能数据: {e}")
    
    return {
        'purchase_mae': purchase_mae,
        'purchase_rmse': purchase_rmse,
        'purchase_mape': purchase_mape,
        'redeem_mae': redeem_mae,
        'redeem_rmse': redeem_rmse,
        'redeem_mape': redeem_mape
    }


def save_balanced_results(predictions, performance):
    """保存平衡版详细结果"""
    print("\n=== 保存平衡版详细结果 ===")
    
    # 保存详细预测结果
    detailed_file = get_project_path('..', 'user_data', 'prophet_v5_detailed_201409.csv')
    predictions.to_csv(detailed_file, index=False)
    
    # 保存性能指标
    performance_file = get_project_path('..', 'user_data', 'prophet_v5_performance.csv')
    performance_df = pd.DataFrame([performance])
    performance_df.to_csv(performance_file, index=False)
    
    print(f"详细预测结果已保存到: {detailed_file}")
    print(f"性能指标已保存到: {performance_file}")


def main():
    """主函数 - 防过拟合平衡版本"""
    print("=== 防过拟合Prophet资金流入流出预测分析 ===")
    print("🎯 防过拟合策略：回归基础 + 精准优化 + 特征重要性分析")
    print("💡 核心理念：找到模型复杂度的最佳平衡点")
    
    try:
        # 1. 加载原始数据（最小化预处理）
        df = load_raw_data()
        
        # 2. 创建Prophet格式数据
        purchase_df = df[['ds', 'purchase']].copy()
        purchase_df.rename(columns={'purchase': 'y'}, inplace=True)
        redeem_df = df[['ds', 'redeem']].copy()
        redeem_df.rename(columns={'redeem': 'y'}, inplace=True)
        
        # 3. 训练平衡版模型
        global purchase_model, redeem_model
        purchase_model, forecast_purchase = train_balanced_prophet_model(df, "申购", "purchase")
        redeem_model, forecast_redeem = train_balanced_prophet_model(df, "赎回", "redeem")
        
        # 4. 生成平衡版预测
        predictions = generate_balanced_predictions(purchase_model, redeem_model, forecast_purchase, forecast_redeem)
        
        # 5. 分析平衡版模型性能
        performance = analyze_balanced_performance(forecast_purchase, forecast_redeem, purchase_df, redeem_df)
        
        # 6. 保存平衡版详细结果
        save_balanced_results(predictions, performance)
        
        print(f"\n=== 防过拟合预测完成 ===")
        print(f"✅ 平衡版Prophet模型训练成功")
        print(f"📊 预测结果已保存")
        print(f"🏆 预期解决过拟合问题，分数回升")
        print(f"📈 可查看文件:")
        print(f"   - 平衡版预测结果: prediction_result/prophet_v5_predictions_201409.csv")
        print(f"   - 详细预测数据: user_data/prophet_v5_detailed_201409.csv")
        print(f"   - 性能指标: user_data/prophet_v5_performance.csv")
        print(f"   - 训练好的模型: model/purchase_prophet_v5_model.pkl")
        print(f"                   model/redeem_prophet_v5_model.pkl")
        
        return True
        
    except Exception as e:
        print(f"防过拟合预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
