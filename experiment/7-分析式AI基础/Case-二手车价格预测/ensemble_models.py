#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成模型方案 - 融合多个优化模型的预测结果
目标：通过模型融合进一步提升精度，冲击500分
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def ensemble_predictions():
    """集成多个模型的预测结果"""
    print("="*60)
    print("模型集成方案 - 融合预测结果")
    print("="*60)
    
    # 收集所有可用的预测结果
    prediction_files = []
    prediction_data = {}
    
    # 尝试加载不同版本的预测结果
    models_to_try = [
        ('RandomForest', 'rf_result_20250926_023237_fixed.csv'),
        ('XGBoost_v1', 'xgb_result_20250926_023816.csv'),
        ('XGBoost_Optimized', 'optimized_xgb_result_20250926_024549.csv'),
    ]
    
    loaded_models = []
    
    for model_name, filename in models_to_try:
        try:
            df = pd.read_csv(filename)
            prediction_data[model_name] = df
            loaded_models.append(model_name)
            print(f"✅ 成功加载 {model_name}: {df.shape}")
        except FileNotFoundError:
            print(f"❌ 未找到 {model_name} 文件: {filename}")
    
    if len(loaded_models) < 2:
        print("❌ 可用模型数量不足，无法进行集成")
        return None
    
    print(f"\n成功加载 {len(loaded_models)} 个模型，开始集成...")
    
    # 获取SaleID（使用第一个模型的SaleID）
    sale_ids = prediction_data[loaded_models[0]]['SaleID'].values
    
    # 收集所有预测价格
    all_predictions = {}
    for model_name in loaded_models:
        all_predictions[model_name] = prediction_data[model_name]['price'].values
    
    # 分析各模型的预测特点
    print("\n各模型预测统计:")
    for model_name in loaded_models:
        prices = all_predictions[model_name]
        print(f"{model_name:>15}: 均值={prices.mean():7.1f}, 中位数={np.median(prices):7.1f}, 标准差={prices.std():7.1f}")
    
    return ensemble_different_strategies(all_predictions, sale_ids, loaded_models)

def ensemble_different_strategies(all_predictions, sale_ids, model_names):
    """尝试不同的集成策略"""
    print("\n" + "="*60)
    print("尝试不同的集成策略")
    print("="*60)
    
    strategies = {}
    
    # 策略1: 简单平均
    predictions_array = np.array([all_predictions[name] for name in model_names])
    strategies['simple_average'] = np.mean(predictions_array, axis=0)
    
    # 策略2: 加权平均（基于模型复杂度）
    if len(model_names) >= 2:
        # 给更复杂的模型更高权重
        weights = []
        for name in model_names:
            if 'RandomForest' in name:
                weights.append(0.2)
            elif 'XGBoost_v1' in name:
                weights.append(0.3)
            elif 'Optimized' in name:
                weights.append(0.5)
            else:
                weights.append(0.25)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 归一化
        
        strategies['weighted_average'] = np.average(predictions_array, axis=0, weights=weights)
        print(f"加权策略权重: {dict(zip(model_names, weights))}")
    
    # 策略3: 中位数集成（对异常值更鲁棒）
    strategies['median_ensemble'] = np.median(predictions_array, axis=0)
    
    # 策略4: 分段集成（不同价格区间使用不同策略）
    if len(model_names) >= 2:
        # 计算每个样本在所有模型中的平均预测
        avg_predictions = np.mean(predictions_array, axis=0)
        
        segmented_result = np.zeros_like(avg_predictions)
        
        # 低价区间（<2000）：使用表现较好的模型
        low_price_mask = avg_predictions < 2000
        if np.any(low_price_mask):
            # 对于低价车，可能某个模型表现更好，这里使用加权平均
            segmented_result[low_price_mask] = strategies['weighted_average'][low_price_mask]
        
        # 中价区间（2000-15000）：使用平均
        mid_price_mask = (avg_predictions >= 2000) & (avg_predictions <= 15000)
        if np.any(mid_price_mask):
            segmented_result[mid_price_mask] = strategies['simple_average'][mid_price_mask]
        
        # 高价区间（>15000）：使用中位数（更保守）
        high_price_mask = avg_predictions > 15000
        if np.any(high_price_mask):
            segmented_result[high_price_mask] = strategies['median_ensemble'][high_price_mask]
        
        strategies['segmented_ensemble'] = segmented_result
    
    # 策略5: 智能加权（基于预测一致性）
    if len(model_names) >= 2:
        # 计算每个样本的预测标准差（作为不确定性指标）
        prediction_std = np.std(predictions_array, axis=0)
        
        # 对于一致性高的预测，使用简单平均；对于一致性低的，使用中位数
        consistency_threshold = np.percentile(prediction_std, 75)
        
        smart_result = np.zeros_like(avg_predictions)
        consistent_mask = prediction_std <= consistency_threshold
        inconsistent_mask = ~consistent_mask
        
        smart_result[consistent_mask] = strategies['simple_average'][consistent_mask]
        smart_result[inconsistent_mask] = strategies['median_ensemble'][inconsistent_mask]
        
        strategies['smart_weighted'] = smart_result
    
    # 分析各策略的统计特点
    print("\n各集成策略统计:")
    for strategy_name, predictions in strategies.items():
        print(f"{strategy_name:>18}: 均值={predictions.mean():7.1f}, 中位数={np.median(predictions):7.1f}, 标准差={predictions.std():7.1f}")
    
    # 选择最佳策略（这里选择智能加权，如果可用的话）
    if 'smart_weighted' in strategies:
        best_strategy = 'smart_weighted'
        best_predictions = strategies['smart_weighted']
    elif 'weighted_average' in strategies:
        best_strategy = 'weighted_average'
        best_predictions = strategies['weighted_average']
    else:
        best_strategy = 'simple_average'
        best_predictions = strategies['simple_average']
    
    print(f"\n选择策略: {best_strategy}")
    
    # 保存集成结果
    return save_ensemble_result(sale_ids, best_predictions, best_strategy, strategies)

def analyze_ensemble_distribution(predictions, strategy_name):
    """分析集成结果的分布"""
    print(f"\n{strategy_name} 策略分布分析:")
    
    # 基本统计
    print(f"均值: {predictions.mean():.2f}")
    print(f"中位数: {np.median(predictions):.2f}")
    print(f"标准差: {predictions.std():.2f}")
    print(f"范围: {predictions.min():.2f} - {predictions.max():.2f}")
    
    # 关键分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\n分位数分布:")
    for p in percentiles:
        value = np.percentile(predictions, p)
        print(f"{p:2d}%: {value:.2f}")
    
    # 价格区间分布
    bins = [0, 1000, 3000, 5000, 10000, 20000, 50000, float('inf')]
    labels = ['0-1k', '1k-3k', '3k-5k', '5k-10k', '10k-20k', '20k-50k', '50k+']
    
    digitized = np.digitize(predictions, bins)
    print("\n价格区间分布:")
    for i, label in enumerate(labels):
        count = np.sum(digitized == i + 1)
        percentage = count / len(predictions) * 100
        print(f"{label:>8}: {count:5d} ({percentage:5.1f}%)")

def save_ensemble_result(sale_ids, best_predictions, strategy_name, all_strategies):
    """保存集成结果"""
    print(f"\n保存集成结果...")
    
    # 最终的后处理
    final_predictions = post_process_ensemble(best_predictions)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'SaleID': sale_ids,
        'price': final_predictions
    })
    
    # 生成文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ensemble_result_{strategy_name}_{current_time}.csv"
    
    # 保存结果
    result_df.to_csv(filename, index=False)
    
    print(f"集成预测结果已保存至: {filename}")
    print(f"预测了 {len(result_df)} 条记录")
    
    # 分析最终结果
    analyze_ensemble_distribution(final_predictions, f"最终集成({strategy_name})")
    
    # 保存所有策略的结果供分析
    if len(all_strategies) > 1:
        all_results_df = pd.DataFrame({'SaleID': sale_ids})
        for strategy, predictions in all_strategies.items():
            all_results_df[f'price_{strategy}'] = predictions
        
        analysis_filename = f"ensemble_analysis_{current_time}.csv"
        all_results_df.to_csv(analysis_filename, index=False)
        print(f"所有策略分析结果已保存至: {analysis_filename}")
    
    return filename, result_df

def post_process_ensemble(predictions):
    """集成结果的后处理"""
    print("对集成结果进行后处理...")
    
    # 确保价格为正
    predictions = np.maximum(predictions, 50)
    
    # 平滑极值
    # 对于过低的预测进行轻微调整
    low_threshold = np.percentile(predictions, 2)
    if low_threshold < 100:
        low_mask = predictions < low_threshold
        predictions[low_mask] = np.maximum(predictions[low_mask], 100)
    
    # 对于过高的预测进行轻微压缩
    high_threshold = np.percentile(predictions, 98)
    if high_threshold > 50000:
        high_mask = predictions > high_threshold
        # 轻微压缩而不是硬截断
        predictions[high_mask] = high_threshold + (predictions[high_mask] - high_threshold) * 0.8
    
    print("集成结果后处理完成")
    return predictions

def compare_with_original():
    """与原始分布对比"""
    try:
        # 读取原始训练集
        train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
        original_price = train_df['price']
        
        print("\n与原始分布对比:")
        print("="*40)
        
        # 原始分布关键统计
        print("原始训练集:")
        print(f"  均值: {original_price.mean():.2f}")
        print(f"  中位数: {original_price.median():.2f}")
        print(f"  标准差: {original_price.std():.2f}")
        
        # 关键分位数
        print("\n原始分位数参考:")
        key_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in key_percentiles:
            value = np.percentile(original_price, p)
            print(f"  {p:2d}%: {value:.2f}")
            
    except Exception as e:
        print(f"无法加载原始分布进行对比: {e}")

def main():
    """主函数"""
    try:
        # 显示原始分布参考
        compare_with_original()
        
        # 执行集成
        result = ensemble_predictions()
        
        if result:
            filename, result_df = result
            print("\n" + "="*60)
            print("🎯 模型集成完成！期待突破500分！")
            print("="*60)
            print(f"📁 最终文件: {filename}")
            return filename, result_df
        else:
            print("❌ 集成失败，请检查预测文件")
            return None, None
        
    except Exception as e:
        print(f"集成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()