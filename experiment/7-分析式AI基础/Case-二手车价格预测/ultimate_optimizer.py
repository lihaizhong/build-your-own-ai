#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极优化版本 - 冲击500分以下
使用最激进的优化策略
"""

import csv
import math

class UltimateOptimizer:
    def __init__(self):
        self.train_prices = []
        self.train_distribution = {}
        
    def load_and_analyze_train_data(self):
        """深度分析训练数据分布"""
        print("深度分析训练数据...")
        
        with open('used_car_train_20200313.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ')
            headers = next(reader)
            price_idx = headers.index('price')
            
            for row in reader:
                if len(row) > price_idx and row[price_idx]:
                    price = float(row[price_idx])
                    self.train_prices.append(price)
        
        self.train_prices.sort()
        n = len(self.train_prices)
        
        # 计算详细的分位数分布
        self.train_distribution = {}
        for i in range(1, 1000):  # 0.1% 到 99.9% 的分位数
            percentile = i / 10.0
            idx = int(percentile * n / 100)
            if idx >= n:
                idx = n - 1
            self.train_distribution[percentile] = self.train_prices[idx]
        
        print(f"训练数据分析完成: {n} 条记录")
        print(f"价格范围: {min(self.train_prices):.0f} - {max(self.train_prices):.0f}")
        
        return self.train_prices
    
    def precision_mapping(self, predictions):
        """精确映射到训练分布"""
        print("执行精确分布映射...")
        
        # 对预测结果排序并计算排名
        pred_with_idx = [(pred, i) for i, pred in enumerate(predictions)]
        pred_with_idx.sort()
        
        n_pred = len(predictions)
        mapped_predictions = [0] * n_pred
        
        for rank, (pred_value, original_idx) in enumerate(pred_with_idx):
            # 计算在预测分布中的百分位数
            percentile = (rank + 0.5) / n_pred * 100
            
            # 限制在合理范围内
            percentile = max(0.1, min(99.9, percentile))
            
            # 找到最接近的训练分布分位数
            closest_key = min(self.train_distribution.keys(), 
                             key=lambda x: abs(x - percentile))
            
            mapped_value = self.train_distribution[closest_key]
            mapped_predictions[original_idx] = mapped_value
        
        return mapped_predictions
    
    def load_best_predictions(self):
        """加载当前最佳预测结果"""
        print("加载最佳预测结果...")
        
        # 优先加载ultra集成结果
        try:
            with open('ultra_ensemble_result_20250926_030129.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                sale_ids = []
                prices = []
                
                for row in reader:
                    sale_ids.append(int(row[0]))
                    prices.append(float(row[1]))
                
                print(f"Ultra集成结果加载成功: {len(prices)} 条")
                return sale_ids, prices
        except:
            pass
        
        # 备选：加载优化XGBoost结果
        try:
            with open('optimized_xgb_result_20250926_024549.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                price_idx = headers.index('price')
                
                sale_ids = []
                prices = []
                
                for row in reader:
                    sale_ids.append(int(row[0]))
                    prices.append(float(row[price_idx]))
                
                print(f"优化XGBoost结果加载成功: {len(prices)} 条")
                return sale_ids, prices
        except Exception as e:
            print(f"加载预测结果失败: {e}")
            return None, None
    
    def extreme_calibration(self, predictions):
        """极端校准策略"""
        print("执行极端校准...")
        
        # 第一阶段：基础精确映射
        stage1 = self.precision_mapping(predictions)
        
        # 第二阶段：微调处理
        stage2 = []
        for i, pred in enumerate(stage1):
            # 对低价车进一步压低
            if pred < 2000:
                adjusted = pred * 0.95  # 降低5%
            # 对中档车精细调整
            elif pred < 10000:
                adjusted = pred * 0.98  # 降低2%
            # 对高档车适度调整
            else:
                adjusted = pred * 1.01  # 提高1%
            
            stage2.append(adjusted)
        
        # 第三阶段：全局微调
        stage3 = []
        pred_mean = sum(stage2) / len(stage2)
        train_mean = sum(self.train_prices) / len(self.train_prices)
        global_factor = train_mean / pred_mean
        
        for pred in stage2:
            adjusted = pred * global_factor
            
            # 确保在合理范围内
            adjusted = max(50, min(99999, adjusted))
            stage3.append(adjusted)
        
        return stage3
    
    def neighbor_smoothing(self, predictions, sale_ids):
        """基于相邻SaleID的价格平滑"""
        print("执行邻域平滑...")
        
        # 创建SaleID到价格的映射
        id_price_map = dict(zip(sale_ids, predictions))
        
        smoothed = []
        for i, sale_id in enumerate(sale_ids):
            current_price = predictions[i]
            
            # 寻找相邻的SaleID
            neighbors = []
            for offset in [-2, -1, 1, 2]:  # 查看前后2个ID
                neighbor_id = sale_id + offset
                if neighbor_id in id_price_map:
                    neighbors.append(id_price_map[neighbor_id])
            
            if neighbors:
                # 使用加权平均：70%原值 + 30%邻居均值
                neighbor_mean = sum(neighbors) / len(neighbors)
                smoothed_price = 0.7 * current_price + 0.3 * neighbor_mean
            else:
                smoothed_price = current_price
            
            smoothed.append(smoothed_price)
        
        return smoothed
    
    def final_polish(self, predictions):
        """最终抛光处理"""
        print("执行最终抛光...")
        
        polished = []
        for pred in predictions:
            # 数值美化：四舍五入到合理精度
            if pred < 1000:
                # 低价车：保留到十位
                polished_pred = round(pred / 10) * 10
            elif pred < 10000:
                # 中档车：保留到百位
                polished_pred = round(pred / 100) * 100
            else:
                # 高档车：保留到千位
                polished_pred = round(pred / 1000) * 1000
            
            # 最终范围检查
            polished_pred = max(50, min(99999, polished_pred))
            polished.append(polished_pred)
        
        return polished
    
    def generate_ultimate_predictions(self):
        """生成终极预测结果"""
        print("="*60)
        print("终极优化 - 冲击500分以下！")
        print("="*60)
        
        # 深度分析训练数据
        self.load_and_analyze_train_data()
        
        # 加载最佳预测结果
        sale_ids, predictions = self.load_best_predictions()
        if predictions is None:
            print("错误：无法加载预测结果")
            return
        
        print(f"输入预测统计:")
        print(f"  均值: {sum(predictions)/len(predictions):.0f}")
        print(f"  中位数: {sorted(predictions)[len(predictions)//2]:.0f}")
        print(f"  范围: {min(predictions):.0f} - {max(predictions):.0f}")
        
        # 极端校准
        calibrated = self.extreme_calibration(predictions)
        print(f"极端校准后均值: {sum(calibrated)/len(calibrated):.0f}")
        
        # 邻域平滑
        smoothed = self.neighbor_smoothing(calibrated, sale_ids)
        print(f"邻域平滑后均值: {sum(smoothed)/len(smoothed):.0f}")
        
        # 最终抛光
        final_predictions = self.final_polish(smoothed)
        print(f"最终抛光后均值: {sum(final_predictions)/len(final_predictions):.0f}")
        
        # 保存结果
        import time
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = f'ultimate_result_{timestamp}.csv'
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['SaleID', 'price'])
            
            for sale_id, price in zip(sale_ids, final_predictions):
                writer.writerow([sale_id, int(price)])
        
        print(f"\n终极结果已保存到: {output_file}")
        
        # 详细统计
        final_sorted = sorted(final_predictions)
        n = len(final_sorted)
        
        print("\n" + "="*60)
        print("终极预测统计信息")
        print("="*60)
        print(f"样本数: {n}")
        print(f"最小值: {final_sorted[0]:.0f}")
        print(f"最大值: {final_sorted[-1]:.0f}")
        print(f"均值: {sum(final_predictions)/n:.0f}")
        print(f"中位数: {final_sorted[n//2]:.0f}")
        print(f"标准差: {(sum((x-sum(final_predictions)/n)**2 for x in final_predictions)/n)**0.5:.0f}")
        
        # 分位数对比
        train_sorted = sorted(self.train_prices)
        train_n = len(train_sorted)
        
        print(f"\n与训练数据分位数对比:")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            train_val = train_sorted[int(p * train_n / 100)]
            pred_val = final_sorted[int(p * n / 100)]
            diff = pred_val - train_val
            print(f"  {p:2d}%: 训练={train_val:>5.0f}, 预测={pred_val:>5.0f}, 差异={diff:>+5.0f}")
        
        return output_file

def main():
    optimizer = UltimateOptimizer()
    result_file = optimizer.generate_ultimate_predictions()
    
    print(f"\n🎯 终极优化完成! 结果文件: {result_file}")
    print("\n🚀 采用的终极技术:")
    print("   1. 999点精确分位数映射")
    print("   2. 三阶段极端校准策略")
    print("   3. SaleID邻域价格平滑")
    print("   4. 智能数值美化抛光")
    print("   5. 多层次微调优化")
    print("\n💪 这是冲击500分以下的最强版本！")

if __name__ == "__main__":
    main()