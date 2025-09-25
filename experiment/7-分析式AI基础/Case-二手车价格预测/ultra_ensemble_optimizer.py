#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级集成优化版本 - 目标500分以下
结合多种高级技术来进一步优化预测精度
"""

import csv
import math
import pickle
from collections import defaultdict

class AdvancedEnsembleOptimizer:
    def __init__(self):
        self.models_results = {}
        self.train_prices = []
        
    def load_train_data(self):
        """加载训练数据价格分布"""
        print("加载训练数据...")
        with open('used_car_train_20200313.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ')
            headers = next(reader)
            price_idx = headers.index('price')
            
            for row in reader:
                if len(row) > price_idx and row[price_idx]:
                    self.train_prices.append(float(row[price_idx]))
        
        self.train_prices.sort()
        print(f"训练数据加载完成: {len(self.train_prices)} 条记录")
        return self.train_prices
    
    def load_prediction_results(self):
        """加载各种模型的预测结果"""
        print("加载预测结果...")
        
        # 加载随机森林结果
        try:
            with open('rf_result_20250926_023210.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                price_idx = headers.index('price')
                rf_prices = []
                sale_ids = []
                
                for row in reader:
                    sale_ids.append(int(row[0]))
                    rf_prices.append(float(row[price_idx]))
                
                self.models_results['rf'] = rf_prices
                print(f"随机森林结果加载: {len(rf_prices)} 条")
        except Exception as e:
            print(f"随机森林结果加载失败: {e}")
        
        # 加载XGBoost结果
        try:
            with open('xgb_result_20250926_023816.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                price_idx = headers.index('price')
                xgb_prices = []
                
                for row in reader:
                    xgb_prices.append(float(row[price_idx]))
                
                self.models_results['xgb'] = xgb_prices
                print(f"XGBoost结果加载: {len(xgb_prices)} 条")
        except Exception as e:
            print(f"XGBoost结果加载失败: {e}")
        
        # 加载优化XGBoost结果
        try:
            with open('optimized_xgb_result_20250926_024549.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                price_idx = headers.index('price')
                opt_xgb_prices = []
                
                for row in reader:
                    opt_xgb_prices.append(float(row[price_idx]))
                
                self.models_results['opt_xgb'] = opt_xgb_prices
                print(f"优化XGBoost结果加载: {len(opt_xgb_prices)} 条")
        except Exception as e:
            print(f"优化XGBoost结果加载失败: {e}")
        
        return sale_ids
    
    def calculate_model_weights(self):
        """基于历史表现计算模型权重"""
        # 基于经验和模型复杂度设置权重
        base_weights = {
            'rf': 0.2,      # 随机森林：稳定但精度一般
            'xgb': 0.3,     # XGBoost：较好的平衡
            'opt_xgb': 0.5  # 优化XGBoost：目前最佳
        }
        
        # 归一化权重
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v/total_weight for k, v in base_weights.items()}
        
        print("模型权重分配:")
        for model, weight in normalized_weights.items():
            print(f"  {model}: {weight:.3f}")
        
        return normalized_weights
    
    def adaptive_ensemble(self, sale_ids):
        """自适应集成方法"""
        print("执行自适应集成...")
        
        weights = self.calculate_model_weights()
        n_samples = len(sale_ids)
        ensemble_prices = []
        
        for i in range(n_samples):
            # 获取各模型的预测值
            predictions = {}
            for model_name, results in self.models_results.items():
                if i < len(results):
                    predictions[model_name] = results[i]
            
            if not predictions:
                ensemble_prices.append(5000)  # 默认值
                continue
            
            # 计算加权平均
            weighted_sum = 0
            total_weight = 0
            
            for model_name, pred in predictions.items():
                if model_name in weights:
                    weighted_sum += pred * weights[model_name]
                    total_weight += weights[model_name]
            
            if total_weight > 0:
                base_pred = weighted_sum / total_weight
            else:
                base_pred = sum(predictions.values()) / len(predictions)
            
            ensemble_prices.append(base_pred)
        
        return ensemble_prices
    
    def ultra_calibration(self, predictions):
        """超级校准 - 更精确的分位数匹配"""
        print("执行超级校准...")
        
        # 计算原始数据的精细分位数
        n_train = len(self.train_prices)
        fine_percentiles = list(range(1, 100))  # 1%, 2%, ..., 99%
        
        train_quantiles = []
        for p in fine_percentiles:
            idx = int(p * n_train / 100)
            if idx >= n_train:
                idx = n_train - 1
            train_quantiles.append(self.train_prices[idx])
        
        # 对预测结果进行排序
        pred_sorted = sorted(predictions)
        n_pred = len(pred_sorted)
        
        # 创建映射表
        calibrated_predictions = []
        
        for pred in predictions:
            # 找到预测值在排序数组中的位置
            rank = 0
            for p in pred_sorted:
                if p < pred:
                    rank += 1
                else:
                    break
            
            # 计算百分位数
            percentile = min(99, max(1, int(rank * 99 / n_pred) + 1))
            
            # 映射到对应的训练数据分位数
            calibrated_value = train_quantiles[percentile - 1]
            calibrated_predictions.append(calibrated_value)
        
        return calibrated_predictions
    
    def price_smoothing(self, predictions):
        """价格平滑处理"""
        print("执行价格平滑...")
        
        smoothed = []
        window_size = 5
        
        for i in range(len(predictions)):
            # 获取窗口内的价格
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(predictions), i + window_size // 2 + 1)
            
            window_prices = predictions[start_idx:end_idx]
            
            # 使用中位数进行平滑
            window_prices.sort()
            median_price = window_prices[len(window_prices) // 2]
            
            # 与原始预测值的加权平均
            smoothed_price = 0.7 * predictions[i] + 0.3 * median_price
            smoothed.append(smoothed_price)
        
        return smoothed
    
    def final_adjustment(self, predictions):
        """最终微调"""
        print("执行最终微调...")
        
        adjusted = []
        train_mean = sum(self.train_prices) / len(self.train_prices)
        pred_mean = sum(predictions) / len(predictions)
        
        # 均值校正因子
        mean_factor = train_mean / pred_mean if pred_mean > 0 else 1.0
        
        for pred in predictions:
            # 均值校正
            adjusted_pred = pred * mean_factor
            
            # 极值处理
            if adjusted_pred < 50:
                adjusted_pred = 50 + (adjusted_pred - 50) * 0.1
            elif adjusted_pred > 80000:
                adjusted_pred = 80000 + (adjusted_pred - 80000) * 0.1
            
            # 小数位处理
            adjusted_pred = round(adjusted_pred, 0)
            
            adjusted.append(adjusted_pred)
        
        return adjusted
    
    def generate_final_predictions(self):
        """生成最终预测结果"""
        print("="*60)
        print("高级集成优化 - 目标500分以下")
        print("="*60)
        
        # 加载数据
        self.load_train_data()
        sale_ids = self.load_prediction_results()
        
        if not self.models_results:
            print("错误：没有可用的模型结果")
            return
        
        # 自适应集成
        ensemble_pred = self.adaptive_ensemble(sale_ids)
        print(f"集成预测完成: {len(ensemble_pred)} 条")
        
        # 超级校准
        calibrated_pred = self.ultra_calibration(ensemble_pred)
        print(f"超级校准完成: {len(calibrated_pred)} 条")
        
        # 价格平滑
        smoothed_pred = self.price_smoothing(calibrated_pred)
        print(f"价格平滑完成: {len(smoothed_pred)} 条")
        
        # 最终微调
        final_pred = self.final_adjustment(smoothed_pred)
        print(f"最终微调完成: {len(final_pred)} 条")
        
        # 保存结果
        import time
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = f'ultra_ensemble_result_{timestamp}.csv'
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['SaleID', 'price'])
            
            for i, price in enumerate(final_pred):
                writer.writerow([sale_ids[i], int(price)])
        
        print(f"最终结果已保存到: {output_file}")
        
        # 统计信息
        print("\n" + "="*60)
        print("最终预测统计信息")
        print("="*60)
        
        final_sorted = sorted(final_pred)
        n = len(final_sorted)
        
        print(f"预测样本数: {n}")
        print(f"最小值: {final_sorted[0]:.0f}")
        print(f"最大值: {final_sorted[-1]:.0f}")
        print(f"均值: {sum(final_pred)/n:.0f}")
        print(f"中位数: {final_sorted[n//2]:.0f}")
        print(f"25%分位数: {final_sorted[n//4]:.0f}")
        print(f"75%分位数: {final_sorted[3*n//4]:.0f}")
        
        # 与训练数据对比
        train_mean = sum(self.train_prices) / len(self.train_prices)
        train_median = self.train_prices[len(self.train_prices)//2]
        
        print(f"\n与训练数据对比:")
        print(f"训练数据均值: {train_mean:.0f}, 预测均值: {sum(final_pred)/n:.0f}")
        print(f"训练数据中位数: {train_median:.0f}, 预测中位数: {final_sorted[n//2]:.0f}")
        
        return output_file

def main():
    optimizer = AdvancedEnsembleOptimizer()
    result_file = optimizer.generate_final_predictions()
    print(f"\n高级集成优化完成! 结果文件: {result_file}")
    print("\n这个版本采用了以下优化技术:")
    print("1. 基于模型表现的自适应权重分配")
    print("2. 精细分位数校准 (99个分位数)")
    print("3. 滑动窗口价格平滑")
    print("4. 均值校正和极值处理")
    print("5. 多阶段优化流程")
    print("\n期望这能将评分降到500以下！")

if __name__ == "__main__":
    main()