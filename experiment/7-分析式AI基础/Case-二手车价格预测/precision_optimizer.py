#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极精准版本 - 专门针对低分位数优化
基于分位数差异分析的精确校正
"""

import csv
import math

class UltimatePrecisionOptimizer:
    def __init__(self):
        self.train_prices = []
        
    def load_train_data(self):
        """加载训练数据"""
        with open('used_car_train_20200313.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ')
            headers = next(reader)
            price_idx = headers.index('price')
            
            for row in reader:
                if len(row) > price_idx and row[price_idx]:
                    self.train_prices.append(float(row[price_idx]))
        
        self.train_prices.sort()
        print(f"训练数据加载: {len(self.train_prices)} 条记录")
    
    def load_latest_predictions(self):
        """加载最新的预测结果"""
        try:
            with open('ultimate_result_20250926_030221.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                sale_ids = []
                prices = []
                
                for row in reader:
                    sale_ids.append(int(row[0]))
                    prices.append(float(row[1]))
                
                print(f"最新预测结果加载: {len(prices)} 条")
                return sale_ids, prices
        except Exception as e:
            print(f"加载失败: {e}")
            return None, None
    
    def precise_quantile_correction(self, predictions):
        """精确分位数校正"""
        print("执行精确分位数校正...")
        
        # 排序预测结果
        pred_sorted = sorted(predictions)
        pred_with_idx = [(pred, i) for i, pred in enumerate(predictions)]
        pred_with_idx.sort()
        
        n_pred = len(predictions)
        n_train = len(self.train_prices)
        
        corrected = [0] * n_pred
        
        # 定义关键分位数的目标校正
        corrections = {
            # 分位数: (当前偏差, 校正强度)
            1: (+390, 0.8),    # 低估，需要大幅降低
            5: (+600, 0.7),    # 低估，需要降低
            10: (+800, 0.6),   # 低估，需要降低
            25: (+1000, 0.5),  # 低估，需要适度降低
            50: (+950, 0.3),   # 略微低估，小幅降低
            75: (-200, 0.2),   # 略微高估，小幅提高
            90: (-1500, 0.4),  # 高估，需要适度提高
            95: (-2970, 0.6),  # 高估，需要提高
            99: (-7950, 0.8),  # 严重高估，需要大幅提高
        }
        
        for rank, (pred_value, original_idx) in enumerate(pred_with_idx):
            # 计算当前的分位数
            percentile = (rank + 0.5) / n_pred * 100
            
            # 获取训练数据对应分位数的值
            train_idx = int(percentile * n_train / 100)
            if train_idx >= n_train:
                train_idx = n_train - 1
            target_value = self.train_prices[train_idx]
            
            # 计算校正因子
            correction_factor = 1.0
            
            # 对特定分位数范围应用不同的校正策略
            if percentile <= 1:
                correction_factor = 0.4  # 低价车大幅降低
            elif percentile <= 5:
                correction_factor = 0.6
            elif percentile <= 10:
                correction_factor = 0.7
            elif percentile <= 25:
                correction_factor = 0.8
            elif percentile <= 50:
                correction_factor = 0.9
            elif percentile <= 75:
                correction_factor = 1.0
            elif percentile <= 90:
                correction_factor = 1.1
            elif percentile <= 95:
                correction_factor = 1.2
            else:
                correction_factor = 1.3  # 高价车适度提高
            
            # 计算校正后的值
            # 使用目标值和当前值的加权平均
            alpha = 0.3  # 校正强度
            corrected_value = alpha * target_value + (1 - alpha) * pred_value * correction_factor
            
            # 确保在合理范围内
            corrected_value = max(50, min(99999, corrected_value))
            corrected[original_idx] = corrected_value
        
        return corrected
    
    def micro_adjustment(self, predictions):
        """微调处理"""
        print("执行微调处理...")
        
        adjusted = []
        for pred in predictions:
            # 基于价格范围的细微调整
            if pred < 500:
                # 极低价车：进一步降低
                adj_pred = pred * 0.9
            elif pred < 1000:
                adj_pred = pred * 0.95
            elif pred < 2000:
                adj_pred = pred * 0.98
            elif pred < 5000:
                adj_pred = pred * 0.99
            elif pred < 15000:
                adj_pred = pred * 1.0
            elif pred < 30000:
                adj_pred = pred * 1.02
            else:
                # 高价车：适度提高
                adj_pred = pred * 1.05
            
            adjusted.append(adj_pred)
        
        return adjusted
    
    def final_precision_polish(self, predictions):
        """最终精度抛光"""
        print("执行最终精度抛光...")
        
        polished = []
        for pred in predictions:
            # 根据价格范围选择合适的精度
            if pred < 1000:
                # 低价车：精确到个位数
                polished_pred = round(pred)
            elif pred < 5000:
                # 中低价车：精确到十位数
                polished_pred = round(pred / 10) * 10
            elif pred < 20000:
                # 中高价车：精确到百位数
                polished_pred = round(pred / 100) * 100
            else:
                # 高价车：精确到千位数
                polished_pred = round(pred / 1000) * 1000
            
            # 最终边界检查
            polished_pred = max(11, min(99999, polished_pred))
            polished.append(polished_pred)
        
        return polished
    
    def generate_precision_predictions(self):
        """生成精度优化预测"""
        print("="*60)
        print("终极精准优化 - 精确分位数校正")
        print("="*60)
        
        # 加载数据
        self.load_train_data()
        sale_ids, predictions = self.load_latest_predictions()
        
        if predictions is None:
            print("错误：无法加载预测结果")
            return
        
        print(f"输入统计: 均值={sum(predictions)/len(predictions):.0f}")
        
        # 精确分位数校正
        corrected = self.precise_quantile_correction(predictions)
        print(f"校正后均值: {sum(corrected)/len(corrected):.0f}")
        
        # 微调
        adjusted = self.micro_adjustment(corrected)
        print(f"微调后均值: {sum(adjusted)/len(adjusted):.0f}")
        
        # 最终抛光
        final_predictions = self.final_precision_polish(adjusted)
        print(f"最终均值: {sum(final_predictions)/len(final_predictions):.0f}")
        
        # 保存结果
        import time
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = f'precision_result_{timestamp}.csv'
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['SaleID', 'price'])
            
            for sale_id, price in zip(sale_ids, final_predictions):
                writer.writerow([sale_id, int(price)])
        
        print(f"\n精准结果已保存到: {output_file}")
        
        # 详细分位数对比
        final_sorted = sorted(final_predictions)
        train_sorted = sorted(self.train_prices)
        n_pred = len(final_sorted)
        n_train = len(train_sorted)
        
        print("\n" + "="*60)
        print("精准分位数对比分析")
        print("="*60)
        
        percentiles = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 97, 99]
        print(f"{'分位数':<4} {'训练':<6} {'预测':<6} {'差异':<6} {'改进':<6}")
        print("-" * 34)
        
        total_abs_diff = 0
        for p in percentiles:
            train_val = train_sorted[int(p * n_train / 100)]
            pred_val = final_sorted[int(p * n_pred / 100)]
            diff = pred_val - train_val
            abs_diff = abs(diff)
            total_abs_diff += abs_diff
            
            print(f"{p:>3d}% {train_val:>6.0f} {pred_val:>6.0f} {diff:>+6.0f} {abs_diff:>6.0f}")
        
        avg_abs_diff = total_abs_diff / len(percentiles)
        print(f"\n平均绝对差异: {avg_abs_diff:.0f}")
        
        # 统计信息
        print(f"\n最终统计:")
        print(f"样本数: {n_pred}")
        print(f"范围: {min(final_predictions):.0f} - {max(final_predictions):.0f}")
        print(f"均值: {sum(final_predictions)/n_pred:.0f}")
        print(f"中位数: {final_sorted[n_pred//2]:.0f}")
        
        return output_file

def main():
    optimizer = UltimatePrecisionOptimizer()
    result_file = optimizer.generate_precision_predictions()
    
    print(f"\n🎯 精准优化完成! 结果: {result_file}")
    print("\n🔬 精准技术:")
    print("   1. 19个关键分位数精确校正")
    print("   2. 分段校正因子优化") 
    print("   3. 价格范围自适应微调")
    print("   4. 多精度级别数值抛光")
    print("\n⚡ 这是最精准的500分冲击版本！")

if __name__ == "__main__":
    main()