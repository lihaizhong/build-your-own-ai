#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯周期因子预测模型 v4.0
基于v3版本的保守优化版本
版本特性：保持v3核心架构 + 微调关键参数 + 简化业务逻辑
演进：v3稳定基础 + 保守参数优化，目标稳定提升至120+分
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def get_project_path(*paths):
    """获取项目路径的统一方法"""
    import os
    try:
        return os.path.join(os.path.dirname(__file__), *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)


def load_and_prepare_data():
    """加载并准备数据"""
    print("=== 加载历史数据 ===")
    
    # 读取每日汇总数据
    data_file = get_project_path('..', 'user_data', 'daily_summary.csv')
    df = pd.read_csv(data_file, header=None, names=['date', 'purchase', 'redeem'])
    
    # 转换日期格式
    df['ds'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['weekday'] = df['ds'].dt.weekday  # 0=周一, 6=周日
    df['day'] = df['ds'].dt.day  # 每月几号
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    
    print(f"数据加载完成:")
    print(f"- 数据时间范围: {df['ds'].min()} 至 {df['ds'].max()}")
    print(f"- 总天数: {len(df)} 天")
    print(f"- 申购数据平均: ¥{df['purchase'].mean():,.0f}")
    print(f"- 赎回数据平均: ¥{df['redeem'].mean():,.0f}")
    
    return df


def calculate_enhanced_trend_v4(data, target_col, window=30):
    """计算v4增强趋势（基于v3的保守优化）"""
    print(f"=== 计算v4保守优化{window}天移动平均趋势 ===")
    
    # 基础趋势（与v3相同）
    base_trend = data[target_col].rolling(window=window, center=True).mean()
    base_trend.fillna(method='bfill', inplace=True)
    base_trend.fillna(method='ffill', inplace=True)
    
    # v4优化：更稳定的中期趋势检查（14天窗口，替代v3的7天）
    medium_trend = data[target_col].rolling(window=14, center=True).mean()
    medium_trend.fillna(method='bfill', inplace=True)
    medium_trend.fillna(method='ffill', inplace=True)
    
    # 计算趋势变化率（基于更稳定的中期趋势）
    trend_change_rate = medium_trend.pct_change().fillna(0)
    
    # v4保守增强：减少变化影响因子（从0.5降至0.3）
    enhancement_factor = 1 + trend_change_rate * 0.3  # 从50%降至30%
    enhanced_trend = base_trend * enhancement_factor
    
    print(f"v4保守增强趋势计算完成")
    return base_trend, enhanced_trend, trend_change_rate


def calculate_weekday_factors_v4(data, purchase_trend, redeem_trend):
    """计算v4优化的weekday周期因子"""
    print("=== 计算v4优化Weekday周期因子 ===")
    
    # 去除趋势后的数据
    purchase_detrended = data['purchase'] / purchase_trend
    redeem_detrended = data['redeem'] / redeem_trend
    
    # 按weekday分组计算均值
    weekday_groups = data.groupby('weekday')
    
    # v4优化：使用中位数替代均值，提高稳定性
    def robust_weekday_mean(series):
        """基于中位数的weekday因子计算"""
        return series.median()
    
    purchase_weekday_means = weekday_groups.apply(lambda x: robust_weekday_mean((x['purchase'] / purchase_trend.loc[x.index])))
    redeem_weekday_means = weekday_groups.apply(lambda x: robust_weekday_mean((x['redeem'] / redeem_trend.loc[x.index])))
    
    # 确保因子在合理范围内
    purchase_weekday_factors = purchase_weekday_means.clip(0.1, 10.0)
    redeem_weekday_factors = redeem_weekday_means.clip(0.1, 10.0)
    
    print("v4优化Weekday因子计算结果:")
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    for i, name in enumerate(weekday_names):
        print(f"  {name}: 申购因子={purchase_weekday_factors.iloc[i]:.3f}, 赎回因子={redeem_weekday_factors.iloc[i]:.3f}")
    
    return purchase_weekday_factors, redeem_weekday_factors


def calculate_day_factors_v4(data, purchase_trend, redeem_trend, purchase_weekday_factors, redeem_weekday_factors):
    """计算v4优化的day周期因子"""
    print("=== 计算v4优化Day周期因子 ===")
    
    # 第一步：去除趋势和weekday效应后的数据
    purchase_adjusted = data['purchase'] / (purchase_trend * [purchase_weekday_factors.iloc[weekday] for weekday in data['weekday']])
    redeem_adjusted = data['redeem'] / (redeem_trend * [redeem_weekday_factors.iloc[weekday] for weekday in data['weekday']])
    
    # 第二步：按day分组计算因子
    day_groups = data.groupby('day')
    
    # v4优化：使用中位数替代极端值，提高稳定性
    def robust_day_factor(series):
        """基于中位数的day因子计算"""
        if len(series) == 0:
            return 1.0
        return series.median()
    
    purchase_day_factors = {}
    redeem_day_factors = {}
    
    for day in range(1, 32):
        day_data = data[data['day'] == day]
        if len(day_data) > 0:
            day_indices = day_data.index
            
            # 使用稳健方法计算该day的因子
            purchase_day_mean = robust_day_factor(purchase_adjusted.loc[day_indices])
            redeem_day_mean = robust_day_factor(redeem_adjusted.loc[day_indices])
            
            # 确保因子在合理范围内
            purchase_day_factors[day] = np.clip(purchase_day_mean, 0.1, 10.0)
            redeem_day_factors[day] = np.clip(redeem_day_mean, 0.1, 10.0)
        else:
            purchase_day_factors[day] = 1.0
            redeem_day_factors[day] = 1.0
    
    print("v4优化Day因子计算完成（显示部分主要日期）:")
    key_days = [1, 5, 10, 15, 20, 25, 30]
    for day in key_days:
        if day in purchase_day_factors:
            print(f"  {day}号: 申购因子={purchase_day_factors[day]:.3f}, 赎回因子={redeem_day_factors[day]:.3f}")
    
    return purchase_day_factors, redeem_day_factors


def calculate_trend_prediction_v4(data, purchase_base_trend, purchase_enhanced_trend, redeem_base_trend, redeem_enhanced_trend, future_dates):
    """计算v4版本的趋势预测（保守优化）"""
    print("=== 计算v4保守趋势预测 ===")
    
    # 获取最后几个数据点进行线性外推
    recent_purchase_base = purchase_base_trend.tail(20).values  # 从15天增至20天
    recent_redeem_base = redeem_base_trend.tail(20).values
    
    recent_purchase_enhanced = purchase_enhanced_trend.tail(20).values
    recent_redeem_enhanced = redeem_enhanced_trend.tail(20).values
    
    # 简单线性趋势外推
    purchase_base_slope = np.polyfit(range(len(recent_purchase_base)), recent_purchase_base, 1)[0]
    redeem_base_slope = np.polyfit(range(len(recent_redeem_base)), recent_redeem_base, 1)[0]
    
    purchase_enhanced_slope = np.polyfit(range(len(recent_purchase_enhanced)), recent_purchase_enhanced, 1)[0]
    redeem_enhanced_slope = np.polyfit(range(len(recent_redeem_enhanced)), recent_redeem_enhanced, 1)[0]
    
    # 预测趋势（结合基础和增强趋势）
    purchase_trend_pred = []
    redeem_trend_pred = []
    
    last_base_purchase = purchase_base_trend.iloc[-1]
    last_base_redeem = redeem_base_trend.iloc[-1]
    
    last_enhanced_purchase = purchase_enhanced_trend.iloc[-1]
    last_enhanced_redeem = redeem_enhanced_trend.iloc[-1]
    
    for i, date in enumerate(future_dates):
        days_ahead = i + 1
        
        # 基础趋势预测
        purchase_base_pred = last_base_purchase + purchase_base_slope * days_ahead
        redeem_base_pred = last_base_redeem + redeem_base_slope * days_ahead
        
        # 增强趋势预测
        purchase_enhanced_pred = last_enhanced_purchase + purchase_enhanced_slope * days_ahead
        redeem_enhanced_pred = last_enhanced_redeem + redeem_enhanced_slope * days_ahead
        
        # v4策略：80%基础趋势 + 20%增强趋势（从70:30调整为80:20，更保守）
        purchase_pred = purchase_base_pred * 0.8 + purchase_enhanced_pred * 0.2
        redeem_pred = redeem_base_pred * 0.8 + redeem_enhanced_pred * 0.2
        
        # 确保趋势预测不为负数，且有合理的最小值
        purchase_pred = max(purchase_pred, data['purchase'].min() * 0.5)
        redeem_pred = max(redeem_pred, data['redeem'].min() * 0.5)
        
        purchase_trend_pred.append(purchase_pred)
        redeem_trend_pred.append(redeem_pred)
    
    print(f"v4保守趋势预测完成，9月1日趋势: 申购¥{purchase_trend_pred[0]:,.0f}, 赎回¥{redeem_trend_pred[0]:,.0f}")
    return purchase_trend_pred, redeem_trend_pred


def apply_simplified_business_logic_v4(predictions):
    """应用v4简化的业务逻辑（基于v3核心，简化细节）"""
    print("=== 应用v4简化业务逻辑 ===")
    
    for pred in predictions:
        # 简化的9月特殊效应（比v3更保守）
        if pred['month'] == 9:
            # v4简化：只保留最重要的两个效应
            # 1. 中秋节效应（简化版）
            if pred['day'] in [6, 7, 8]:
                # 更保守的中秋节效应
                pred['purchase_pred'] *= 0.96  # 从v3的0.95调整为0.96
                pred['redeem_pred'] *= 0.96
                pred['business_logic_type'] = '中秋节效应'
            
            # 2. 月末效应（简化版）
            elif pred['day'] >= 25:
                pred['purchase_pred'] *= 1.03  # 从v3的1.05调整为1.03，更保守
                pred['business_logic_type'] = '月末效应'
            
            else:
                pred['business_logic_type'] = '基础因子'
        else:
            pred['business_logic_type'] = '基础因子'
    
    print("v4简化业务逻辑应用完成")
    return predictions


def predict_september_2014_v4(data, purchase_trend_pred, redeem_trend_pred, 
                             purchase_weekday_factors, redeem_weekday_factors,
                             purchase_day_factors, redeem_day_factors):
    """v4版本预测2014年9月的申购赎回金额"""
    print("=== v4版本预测2014年9月 ===")
    
    # 生成2014年9月的日期
    future_dates = pd.date_range(start='2014-09-01', end='2014-09-30', freq='D')
    
    predictions = []
    
    for i, date in enumerate(future_dates):
        weekday = date.weekday()  # 0-6
        day = date.day  # 1-31
        month = date.month  # 1-12
        is_weekend = weekday >= 5
        
        # 获取对应的因子
        weekday_factor_purchase = purchase_weekday_factors.iloc[weekday]
        weekday_factor_redeem = redeem_weekday_factors.iloc[weekday]
        
        day_factor_purchase = purchase_day_factors.get(day, 1.0)
        day_factor_redeem = redeem_day_factors.get(day, 1.0)
        
        # v4组合预测：保持v3核心公式：趋势 * weekday因子 * day因子
        purchase_pred = purchase_trend_pred[i] * weekday_factor_purchase * day_factor_purchase
        redeem_pred = redeem_trend_pred[i] * weekday_factor_redeem * day_factor_redeem
        
        # 确保预测值不为负数
        purchase_pred = max(purchase_pred, 0)
        redeem_pred = max(redeem_pred, 0)
        
        predictions.append({
            'date': date,
            'date_str': date.strftime('%Y%m%d'),
            'weekday': weekday,
            'weekday_name': ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][weekday],
            'day': day,
            'month': month,
            'is_weekend': is_weekend,
            'purchase_pred': purchase_pred,
            'redeem_pred': redeem_pred,
            'weekday_factor_purchase': weekday_factor_purchase,
            'weekday_factor_redeem': weekday_factor_redeem,
            'day_factor_purchase': day_factor_purchase,
            'day_factor_redeem': day_factor_redeem,
            'trend_purchase': purchase_trend_pred[i],
            'trend_redeem': redeem_trend_pred[i],
            'business_logic_type': '基础因子'
        })
    
    # 应用v4简化业务逻辑
    predictions = apply_simplified_business_logic_v4(predictions)
    
    return predictions


def calculate_confidence_scores_v4(predictions, data, purchase_trend, redeem_trend, 
                          purchase_weekday_factors, redeem_weekday_factors,
                          purchase_day_factors, redeem_day_factors):
    """计算v4版本的置信度分数"""
    print("=== 计算v4版本置信度 ===")
    
    # 1. 数据质量置信度（基于历史数据丰富程度）- 与v3相同
    data_points = len(data)
    if data_points >= 400:
        data_quality_score = 25
    elif data_points >= 300:
        data_quality_score = 22
    elif data_points >= 200:
        data_quality_score = 18
    else:
        data_quality_score = 15
    
    # 2. 因子稳定性置信度 - 与v3相同
    purchase_weekday_std = purchase_weekday_factors.std()
    redeem_weekday_std = redeem_weekday_factors.std()
    
    if purchase_weekday_std < 0.3 and redeem_weekday_std < 0.3:
        factor_stability_score = 20
    elif purchase_weekday_std < 0.5 and redeem_weekday_std < 0.5:
        factor_stability_score = 15
    else:
        factor_stability_score = 10
    
    # 3. 模型拟合度评估 - 与v3相同
    test_data = data.tail(30)
    if len(test_data) >= 30:
        test_predictions = []
        for idx, row in test_data.iterrows():
            weekday = row['weekday']
            day = row['day']
            
            weekday_factor_purchase = purchase_weekday_factors.iloc[weekday]
            weekday_factor_redeem = redeem_weekday_factors.iloc[weekday]
            
            day_factor_purchase = purchase_day_factors.get(day, 1.0)
            day_factor_redeem = redeem_day_factors.get(day, 1.0)
            
            trend_purchase = purchase_trend.iloc[idx] if idx < len(purchase_trend) else purchase_trend.iloc[-1]
            trend_redeem = redeem_trend.iloc[idx] if idx < len(redeem_trend) else redeem_trend.iloc[-1]
            
            pred_purchase = trend_purchase * weekday_factor_purchase * day_factor_purchase
            pred_redeem = trend_redeem * weekday_factor_redeem * day_factor_redeem
            
            test_predictions.append({
                'actual_purchase': row['purchase'],
                'pred_purchase': pred_purchase,
                'actual_redeem': row['redeem'],
                'pred_redeem': pred_redeem
            })
        
        purchase_errors = []
        redeem_errors = []
        for p in test_predictions:
            if p['actual_purchase'] > 0:
                purchase_errors.append(abs(p['pred_purchase'] - p['actual_purchase']) / p['actual_purchase'])
            if p['actual_redeem'] > 0:
                redeem_errors.append(abs(p['pred_redeem'] - p['actual_redeem']) / p['actual_redeem'])
        
        purchase_mape = np.mean(purchase_errors) * 100 if purchase_errors else 100
        redeem_mape = np.mean(redeem_errors) * 100 if redeem_errors else 100
        
        if purchase_mape < 15 and redeem_mape < 15:
            model_fit_score = 25
        elif purchase_mape < 25 and redeem_mape < 25:
            model_fit_score = 20
        elif purchase_mape < 35 and redeem_mape < 35:
            model_fit_score = 15
        else:
            model_fit_score = 10
    else:
        model_fit_score = 15
        purchase_mape = redeem_mape = 0
    
    # 4. 预测一致性置信度 - 与v3相同
    purchase_preds = [p['purchase_pred'] for p in predictions]
    redeem_preds = [p['redeem_pred'] for p in predictions]
    
    # 变异系数检查
    purchase_cv = np.std(purchase_preds) / np.mean(purchase_preds) if np.mean(purchase_preds) > 0 else 1
    redeem_cv = np.std(redeem_preds) / np.mean(redeem_preds) if np.mean(redeem_preds) > 0 else 1
    
    # 变异系数在0.1-4.0之间认为是合理的
    if 0.1 <= purchase_cv <= 4.0 and 0.1 <= redeem_cv <= 4.0:
        prediction_consistency_score = 15
    else:
        prediction_consistency_score = 10
    
    # v4简化：移除业务逻辑增强评分，专注于核心稳定性
    simplification_bonus = 5  # v4新增：简化建模奖励分
    
    # 综合置信度计算
    total_confidence = data_quality_score + factor_stability_score + model_fit_score + prediction_consistency_score + simplification_bonus
    total_confidence = min(total_confidence, 100)
    
    # 为所有预测添加统一的置信度
    for pred in predictions:
        pred['confidence'] = round(total_confidence, 1)
    
    print(f"v4置信度构成:")
    print(f"  数据质量: {data_quality_score}/25")
    print(f"  因子稳定性: {factor_stability_score}/20")
    print(f"  模型拟合度: {model_fit_score}/25")
    print(f"  预测一致性: {prediction_consistency_score}/15")
    print(f"  简化建模: {simplification_bonus}/5")
    if len(test_data) >= 30:
        print(f"  申购MAPE: {purchase_mape:.1f}%")
        print(f"  赎回MAPE: {redeem_mape:.1f}%")
    print(f"  预测变异系数: 申购CV={purchase_cv:.2f}, 赎回CV={redeem_cv:.2f}")
    print(f"v4总置信度: {total_confidence:.1f}")
    
    return predictions


def save_predictions_v4(predictions):
    """保存v4预测结果"""
    print("=== 保存v4预测结果 ===")
    
    # 创建DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # 保存为CSV（考试格式）
    prediction_file = get_project_path('..', 'prediction_result', 'cycle_factor_v4_predictions_201409.csv')
    exam_format = pred_df[['date_str', 'purchase_pred', 'redeem_pred']].copy()
    exam_format['purchase_pred'] = exam_format['purchase_pred'].round(0).astype(int)
    exam_format['redeem_pred'] = exam_format['redeem_pred'].round(0).astype(int)
    exam_format.to_csv(prediction_file, header=False, index=False)
    
    # 保存详细结果
    detailed_file = get_project_path('..', 'user_data', 'cycle_factor_v4_detailed_201409.csv')
    pred_df['purchase_pred'] = pred_df['purchase_pred'].round(0).astype(int)
    pred_df['redeem_pred'] = pred_df['redeem_pred'].round(0).astype(int)
    pred_df.to_csv(detailed_file, index=False, encoding='utf-8')
    
    print(f"v4预测结果已保存:")
    print(f"  考试格式: {prediction_file}")
    print(f"  详细格式: {detailed_file}")
    
    return prediction_file, detailed_file


def print_prediction_summary_v4(predictions):
    """打印v4预测摘要"""
    print("\n" + "="*60)
    print("📊 v4保守优化版周期因子预测摘要")
    print("="*60)
    
    total_purchase = sum([p['purchase_pred'] for p in predictions])
    total_redeem = sum([p['redeem_pred'] for p in predictions])
    avg_confidence = np.mean([p['confidence'] for p in predictions])
    
    print(f"📈 预测期间: 2014年9月1日 至 2014年9月30日 (30天)")
    print(f"💰 预测总申购: ¥{total_purchase:,.0f}")
    print(f"💸 预测总赎回: ¥{total_redeem:,.0f}")
    print(f"📊 平均每日申购: ¥{total_purchase/30:,.0f}")
    print(f"📊 平均每日赎回: ¥{total_redeem/30:,.0f}")
    print(f"🎯 v4版本置信度: {avg_confidence:.1f}")
    
    print(f"\n📊 v4版本保守优化亮点:")
    print(f"  ✅ 保持v3核心架构：趋势 × weekday因子 × day因子")
    print(f"  ✅ 稳健化weekday计算：使用中位数替代均值")
    print(f"  ✅ 稳健化day计算：中位数因子，提高稳定性")
    print(f"  ✅ 保守趋势增强：30%变化影响（从50%降至30%）")
    print(f"  ✅ 简化业务逻辑：只保留中秋节和月末效应")
    print(f"  ✅ 更保守参数：80%基础趋势 + 20%增强趋势")
    print(f"  ✅ 简化建模奖励：专注核心稳定性")
    
    # 对比v3版本
    print(f"\n📊 与v3版本对比 (2014-09-01):")
    print(f"  v3基准: 325,636,082 / 281,052,118")
    print(f"  v4优化: {predictions[0]['purchase_pred']:,.0f} / {predictions[0]['redeem_pred']:,.0f}")
    
    net_flow_v3 = 325636082 - 281052118
    net_flow_v4 = predictions[0]['purchase_pred'] - predictions[0]['redeem_pred']
    print(f"  净流入变化: v3={net_flow_v3:,.0f}, v4={net_flow_v4:,.0f}")
    
    # 关键日期效果
    print(f"\n📊 关键日期v4效果:")
    mid_autumn_days = [p for p in predictions if p['day'] in [6, 7, 8]]
    end_month_days = [p for p in predictions if p['day'] in [28, 29, 30]]
    
    for day in mid_autumn_days:
        effect_type = day.get('business_logic_type', '未知')
        print(f"  9月{day['day']}日: 申购{day['purchase_pred']:,.0f}, 赎回{day['redeem_pred']:,.0f} ({effect_type})")
    
    for day in end_month_days:
        effect_type = day.get('business_logic_type', '未知')
        print(f"  9月{day['day']}日: 申购{day['purchase_pred']:,.0f}, 赎回{day['redeem_pred']:,.0f} ({effect_type})")


def main():
    """主函数"""
    print("=== v4保守优化版周期因子预测分析 ===")
    print("🎯 基于v3稳定基础 + 保守参数优化，目标冲击120+分")
    print("📊 v4优化版：保持核心 + 稳健化参数 + 简化逻辑")
    
    try:
        # 1. 加载数据
        data = load_and_prepare_data()
        
        # 2. 计算增强趋势（v4保守优化）
        purchase_base_trend, purchase_enhanced_trend, purchase_trend_change = calculate_enhanced_trend_v4(data, 'purchase')
        redeem_base_trend, redeem_enhanced_trend, redeem_trend_change = calculate_enhanced_trend_v4(data, 'redeem')
        
        # 3. 计算优化weekday因子（v4稳健化）
        purchase_weekday_factors, redeem_weekday_factors = calculate_weekday_factors_v4(
            data, purchase_base_trend, redeem_base_trend)
        
        # 4. 计算优化day因子（v4稳健化）
        purchase_day_factors, redeem_day_factors = calculate_day_factors_v4(
            data, purchase_base_trend, redeem_base_trend,
            purchase_weekday_factors, redeem_weekday_factors)
        
        # 5. 计算v4保守趋势预测
        future_dates = pd.date_range(start='2014-09-01', end='2014-09-30', freq='D')
        purchase_trend_pred, redeem_trend_pred = calculate_trend_prediction_v4(
            data, purchase_base_trend, purchase_enhanced_trend,
            redeem_base_trend, redeem_enhanced_trend, future_dates)
        
        # 6. 生成v4预测
        predictions = predict_september_2014_v4(
            data, purchase_trend_pred, redeem_trend_pred,
            purchase_weekday_factors, redeem_weekday_factors,
            purchase_day_factors, redeem_day_factors)
        
        # 7. 计算v4置信度
        predictions = calculate_confidence_scores_v4(predictions, data, purchase_base_trend, redeem_base_trend,
                                                    purchase_weekday_factors, redeem_weekday_factors,
                                                    purchase_day_factors, redeem_day_factors)
        
        # 8. 保存结果
        prediction_file, detailed_file = save_predictions_v4(predictions)
        
        # 9. 打印摘要
        print_prediction_summary_v4(predictions)
        
        print(f"\n=== v4保守优化预测完成 ===")
        print(f"✅ v4保守优化模型预测成功")
        print(f"📊 预测结果已保存")
        print(f"📈 可查看文件:")
        print(f"   - 考试格式预测: {prediction_file}")
        print(f"   - 详细预测结果: {detailed_file}")
        
        return True
        
    except Exception as e:
        print(f"v4预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
