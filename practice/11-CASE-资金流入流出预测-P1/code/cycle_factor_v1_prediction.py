#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯周期因子预测模型 v1.0
基础版本：weekday和day周期因子建模
基于weekday周期因子和day周期因子的时间序列分解预测
"""

import pandas as pd
import numpy as np
import warnings
from ...shared import get_project_path

warnings.filterwarnings('ignore')


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


def calculate_trend(data, window=30):
    """计算趋势（使用移动平均）"""
    print(f"=== 计算{window}天移动平均趋势 ===")
    
    # 对申购和赎回分别计算趋势
    purchase_trend = data['purchase'].rolling(window=window, center=True).mean()
    redeem_trend = data['redeem'].rolling(window=window, center=True).mean()
    
    # 处理首尾的NaN值
    purchase_trend.fillna(method='bfill', inplace=True)
    purchase_trend.fillna(method='ffill', inplace=True)
    redeem_trend.fillna(method='bfill', inplace=True)
    redeem_trend.fillna(method='ffill', inplace=True)
    
    print(f"趋势计算完成")
    return purchase_trend, redeem_trend


def calculate_weekday_factors(data, purchase_trend, redeem_trend):
    """计算weekday周期因子（7个因子）"""
    print("=== 计算Weekday周期因子 ===")
    
    # 去除趋势后的数据
    purchase_detrended = data['purchase'] / purchase_trend  # 使用比率而非差值
    redeem_detrended = data['redeem'] / redeem_trend  # 使用比率而非差值
    
    # 按weekday分组计算均值
    weekday_groups = data.groupby('weekday')
    
    # 申购weekday因子：计算每个weekday的平均比率
    purchase_weekday_means = weekday_groups.apply(lambda x: (x['purchase'] / purchase_trend.loc[x.index]).mean())
    redeem_weekday_means = weekday_groups.apply(lambda x: (x['redeem'] / redeem_trend.loc[x.index]).mean())
    
    # 确保因子在合理范围内（0.1到10之间）
    purchase_weekday_factors = purchase_weekday_means.clip(0.1, 10.0)
    redeem_weekday_factors = redeem_weekday_means.clip(0.1, 10.0)
    
    print("Weekday因子计算结果:")
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    for i, name in enumerate(weekday_names):
        print(f"  {name}: 申购因子={purchase_weekday_factors.iloc[i]:.3f}, 赎回因子={redeem_weekday_factors.iloc[i]:.3f}")
    
    return purchase_weekday_factors, redeem_weekday_factors


def calculate_day_factors(data, purchase_trend, redeem_trend, purchase_weekday_factors, redeem_weekday_factors):
    """计算day周期因子（1-31号的因子）"""
    print("=== 计算Day周期因子 ===")
    
    # 第一步：去除趋势和weekday效应后的数据
    purchase_adjusted = data['purchase'] / (purchase_trend * [purchase_weekday_factors.iloc[weekday] for weekday in data['weekday']])
    redeem_adjusted = data['redeem'] / (redeem_trend * [redeem_weekday_factors.iloc[weekday] for weekday in data['weekday']])
    
    # 第二步：按day分组计算因子
    day_groups = data.groupby('day')
    
    # 创建day因子字典，默认值为1.0
    purchase_day_factors = {}
    redeem_day_factors = {}
    
    # 计算每个day的因子（只考虑在历史数据中出现过的day）
    for day in range(1, 32):  # 1-31号
        day_data = data[data['day'] == day]
        if len(day_data) > 0:
            day_indices = day_data.index
            
            # 计算该day的平均调整后比率
            purchase_day_mean = purchase_adjusted.loc[day_indices].mean()
            redeem_day_mean = redeem_adjusted.loc[day_indices].mean()
            
            # 确保因子在合理范围内
            purchase_day_factors[day] = np.clip(purchase_day_mean, 0.1, 10.0)
            redeem_day_factors[day] = np.clip(redeem_day_mean, 0.1, 10.0)
        else:
            # 如果某一天在历史数据中没有，使用默认值
            purchase_day_factors[day] = 1.0
            redeem_day_factors[day] = 1.0
    
    print("Day因子计算完成（显示部分主要日期）:")
    key_days = [1, 5, 10, 15, 20, 25, 30]
    for day in key_days:
        if day in purchase_day_factors:
            print(f"  {day}号: 申购因子={purchase_day_factors[day]:.3f}, 赎回因子={redeem_day_factors[day]:.3f}")
    
    return purchase_day_factors, redeem_day_factors


def calculate_trend_prediction(data, purchase_trend, redeem_trend, future_dates):
    """计算趋势预测"""
    print("=== 计算趋势预测 ===")
    
    # 获取最后几个数据点进行线性外推
    recent_purchase_trend = purchase_trend.tail(15).values
    recent_redeem_trend = redeem_trend.tail(15).values
    
    # 简单线性趋势外推
    purchase_trend_slope = np.polyfit(range(len(recent_purchase_trend)), recent_purchase_trend, 1)[0]
    redeem_trend_slope = np.polyfit(range(len(recent_redeem_trend)), recent_redeem_trend, 1)[0]
    
    # 预测趋势
    purchase_trend_pred = []
    redeem_trend_pred = []
    
    last_trend_purchase = purchase_trend.iloc[-1]
    last_trend_redeem = redeem_trend.iloc[-1]
    
    for i, date in enumerate(future_dates):
        days_ahead = i + 1
        purchase_pred = last_trend_purchase + purchase_trend_slope * days_ahead
        redeem_pred = last_trend_redeem + redeem_trend_slope * days_ahead
        
        # 确保趋势预测不为负数，且有合理的最小值
        purchase_pred = max(purchase_pred, data['purchase'].min() * 0.5)
        redeem_pred = max(redeem_pred, data['redeem'].min() * 0.5)
        
        purchase_trend_pred.append(purchase_pred)
        redeem_trend_pred.append(redeem_pred)
    
    print(f"趋势预测完成，9月1日趋势: 申购¥{purchase_trend_pred[0]:,.0f}, 赎回¥{redeem_trend_pred[0]:,.0f}")
    return purchase_trend_pred, redeem_trend_pred


def predict_september_2014(data, purchase_trend_pred, redeem_trend_pred, 
                          purchase_weekday_factors, redeem_weekday_factors,
                          purchase_day_factors, redeem_day_factors):
    """预测2014年9月的申购赎回金额"""
    print("=== 预测2014年9月 ===")
    
    # 生成2014年9月的日期
    future_dates = pd.date_range(start='2014-09-01', end='2014-09-30', freq='D')
    
    predictions = []
    
    for i, date in enumerate(future_dates):
        weekday = date.weekday()  # 0-6
        day = date.day  # 1-31
        
        # 获取对应的因子
        weekday_factor_purchase = purchase_weekday_factors.iloc[weekday]
        weekday_factor_redeem = redeem_weekday_factors.iloc[weekday]
        
        day_factor_purchase = purchase_day_factors.get(day, 1.0)
        day_factor_redeem = redeem_day_factors.get(day, 1.0)
        
        # 组合预测：趋势 * weekday因子 * day因子
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
            'purchase_pred': purchase_pred,
            'redeem_pred': redeem_pred,
            'weekday_factor_purchase': weekday_factor_purchase,
            'weekday_factor_redeem': weekday_factor_redeem,
            'day_factor_purchase': day_factor_purchase,
            'day_factor_redeem': day_factor_redeem
        })
    
    return predictions


def calculate_confidence_scores_v1(predictions, data, purchase_trend, redeem_trend, 
                          purchase_weekday_factors, redeem_weekday_factors,
                          purchase_day_factors, redeem_day_factors):
    """计算整体方案的置信度分数（0-100）- 基础版"""
    print("=== 计算整体方案置信度 ===")
    
    # 1. 数据质量置信度（基于历史数据丰富程度）
    # 基于数据量、完整性和质量
    data_points = len(data)
    if data_points >= 400:  # 400天以上为优秀
        data_quality_score = 25
    elif data_points >= 300:  # 300天以上为良好
        data_quality_score = 22
    elif data_points >= 200:  # 200天以上为中等
        data_quality_score = 18
    else:
        data_quality_score = 15
    
    # 2. 因子稳定性置信度（检查weekday和day因子的合理性）
    purchase_weekday_std = purchase_weekday_factors.std()
    redeem_weekday_std = redeem_weekday_factors.std()
    
    if purchase_weekday_std < 0.3 and redeem_weekday_std < 0.3:
        factor_stability_score = 20
    elif purchase_weekday_std < 0.5 and redeem_weekday_std < 0.5:
        factor_stability_score = 15
    else:
        factor_stability_score = 10
    
    # 3. 模型拟合度评估（用历史数据进行交叉验证）
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
        purchase_mape = redeem_mape = 0  # 用于显示
    
    # 4. 预测一致性置信度（基础版）
    purchase_preds = [p['purchase_pred'] for p in predictions]
    redeem_preds = [p['redeem_pred'] for p in predictions]
    
    # 预测值的变化幅度应在合理范围内
    purchase_cv = np.std(purchase_preds) / np.mean(purchase_preds) if np.mean(purchase_preds) > 0 else 1
    redeem_cv = np.std(redeem_preds) / np.mean(redeem_preds) if np.mean(redeem_preds) > 0 else 1
    
    # 变异系数在0.5-2.0之间认为是合理的（v1版本标准）
    if 0.5 <= purchase_cv <= 2.0 and 0.5 <= redeem_cv <= 2.0:
        prediction_consistency_score = 15
    else:
        prediction_consistency_score = 10
    
    # 综合置信度计算
    total_confidence = data_quality_score + factor_stability_score + model_fit_score + prediction_consistency_score
    total_confidence = min(total_confidence, 100)  # 最高100分
    
    # 为所有预测添加统一的置信度
    for pred in predictions:
        pred['confidence'] = round(total_confidence, 1)
    
    print(f"置信度构成:")
    print(f"  数据质量: {data_quality_score}/25")
    print(f"  因子稳定性: {factor_stability_score}/20")
    print(f"  模型拟合度: {model_fit_score}/25")
    print(f"  预测一致性: {prediction_consistency_score}/15")
    if len(test_data) >= 30:
        print(f"  申购MAPE: {purchase_mape:.1f}%")
        print(f"  赎回MAPE: {redeem_mape:.1f}%")
    print(f"总置信度: {total_confidence:.1f}")
    
    return predictions


def save_predictions(predictions):
    """保存预测结果"""
    print("=== 保存预测结果 ===")
    
    # 创建DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # 保存为CSV（考试格式）
    prediction_file = get_project_path('..', 'prediction_result', 'cycle_factor_v1_predictions_201409.csv')
    exam_format = pred_df[['date_str', 'purchase_pred', 'redeem_pred']].copy()
    exam_format['purchase_pred'] = exam_format['purchase_pred'].round(0).astype(int)
    exam_format['redeem_pred'] = exam_format['redeem_pred'].round(0).astype(int)
    exam_format.to_csv(prediction_file, header=False, index=False)
    
    # 保存详细结果
    detailed_file = get_project_path('..', 'user_data', 'cycle_factor_v1_detailed_201409.csv')
    pred_df['purchase_pred'] = pred_df['purchase_pred'].round(0).astype(int)
    pred_df['redeem_pred'] = pred_df['redeem_pred'].round(0).astype(int)
    pred_df.to_csv(detailed_file, index=False, encoding='utf-8')
    
    print(f"预测结果已保存:")
    print(f"  考试格式: {prediction_file}")
    print(f"  详细格式: {detailed_file}")
    
    return prediction_file, detailed_file


def print_prediction_summary(predictions):
    """打印预测摘要"""
    print("\n" + "="*60)
    print("📊 2014年9月周期因子预测摘要（v1.0基础版）")
    print("="*60)
    
    total_purchase = sum([p['purchase_pred'] for p in predictions])
    total_redeem = sum([p['redeem_pred'] for p in predictions])
    avg_confidence = np.mean([p['confidence'] for p in predictions])
    
    print(f"📈 预测期间: 2014年9月1日 至 2014年9月30日 (30天)")
    print(f"💰 预测总申购: ¥{total_purchase:,.0f}")
    print(f"💸 预测总赎回: ¥{total_redeem:,.0f}")
    print(f"📊 平均每日申购: ¥{total_purchase/30:,.0f}")
    print(f"📊 平均每日赎回: ¥{total_redeem/30:,.0f}")
    print(f"🎯 方案置信度: {avg_confidence:.1f}")
    
    print(f"\n📅 详细预测结果:")
    print("-" * 80)
    print(f"{'日期':<10} {'星期':<6} {'申购金额':<15} {'赎回金额':<15} {'置信度':<8} {'主要因子'}")
    print("-" * 80)
    
    for pred in predictions:
        weekday_name = pred['weekday_name']
        purchase = pred['purchase_pred']
        redeem = pred['redeem_pred']
        confidence = pred['confidence']
        
        # 找出主要因子
        if pred['weekday_factor_purchase'] > 1.2:
            weekday_factor = "周高"
        elif pred['weekday_factor_purchase'] < 0.8:
            weekday_factor = "周低"
        else:
            weekday_factor = "周平"
            
        if pred['day_factor_purchase'] > 1.2:
            day_factor = "日高"
        elif pred['day_factor_purchase'] < 0.8:
            day_factor = "日低"
        else:
            day_factor = "日平"
        
        main_factor = f"{weekday_factor}+{day_factor}"
        
        print(f"{pred['date_str']:<10} {weekday_name:<6} ¥{purchase:<14,.0f} ¥{redeem:<14,.0f} {confidence:<7.1f} {main_factor}")
    
    # 分析weekday模式
    print(f"\n📊 Weekday模式分析:")
    weekday_analysis = {}
    for pred in predictions:
        weekday = pred['weekday_name']
        if weekday not in weekday_analysis:
            weekday_analysis[weekday] = {'purchase': [], 'redeem': [], 'count': 0}
        weekday_analysis[weekday]['purchase'].append(pred['purchase_pred'])
        weekday_analysis[weekday]['redeem'].append(pred['redeem_pred'])
        weekday_analysis[weekday]['count'] += 1
    
    for weekday, data in weekday_analysis.items():
        avg_purchase = np.mean(data['purchase'])
        avg_redeem = np.mean(data['redeem'])
        print(f"  {weekday}: 平均申购 ¥{avg_purchase:,.0f}, 平均赎回 ¥{avg_redeem:,.0f}")


def main():
    """主函数"""
    print("=== 纯周期因子资金流入流出预测分析 v1.0 ===")
    print("🎯 基于weekday和day周期因子的时间序列分解预测")
    print("📊 基础版：纯周期因子建模")
    
    try:
        # 1. 加载数据
        data = load_and_prepare_data()
        
        # 2. 计算趋势
        purchase_trend, redeem_trend = calculate_trend(data)
        
        # 3. 计算weekday因子
        purchase_weekday_factors, redeem_weekday_factors = calculate_weekday_factors(
            data, purchase_trend, redeem_trend)
        
        # 4. 计算day因子
        purchase_day_factors, redeem_day_factors = calculate_day_factors(
            data, purchase_trend, redeem_trend, 
            purchase_weekday_factors, redeem_weekday_factors)
        
        # 5. 计算趋势预测
        future_dates = pd.date_range(start='2014-09-01', end='2014-09-30', freq='D')
        purchase_trend_pred, redeem_trend_pred = calculate_trend_prediction(
            data, purchase_trend, redeem_trend, future_dates)
        
        # 6. 生成预测
        predictions = predict_september_2014(
            data, purchase_trend_pred, redeem_trend_pred,
            purchase_weekday_factors, redeem_weekday_factors,
            purchase_day_factors, redeem_day_factors)
        
        # 7. 计算置信度（基础版）
        predictions = calculate_confidence_scores_v1(predictions, data, purchase_trend, redeem_trend,
                                                    purchase_weekday_factors, redeem_weekday_factors,
                                                    purchase_day_factors, redeem_day_factors)
        
        # 8. 保存结果
        prediction_file, detailed_file = save_predictions(predictions)
        
        # 9. 打印摘要
        print_prediction_summary(predictions)
        
        print(f"\n=== 预测完成 ===")
        print(f"✅ 基础版纯周期因子模型预测成功")
        print(f"📊 预测结果已保存")
        print(f"📈 可查看文件:")
        print(f"   - 考试格式预测: {prediction_file}")
        print(f"   - 详细预测结果: {detailed_file}")
        
        return True
        
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()