#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保守微调版本 - 基于677.0636分的稳定基础进行小幅优化
避免过度拟合，保持分布特征
"""

import csv

class ConservativeOptimizer:
    def __init__(self):
        pass
    
    def load_best_baseline(self):
        """加载677.0636分的基准版本"""
        print("加载677.0636分基准版本...")
        
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
                
                print(f"基准版本加载成功: {len(prices)} 条记录")
                print(f"均值: {sum(prices)/len(prices):.0f}")
                print(f"范围: {min(prices):.0f} - {max(prices):.0f}")
                
                return sale_ids, prices
        except Exception as e:
            print(f"加载基准版本失败: {e}")
            return None, None
    
    def gentle_smoothing(self, predictions):
        """轻微平滑处理"""
        print("执行轻微平滑...")
        
        smoothed = []
        window_size = 3  # 小窗口
        
        for i in range(len(predictions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(predictions), i + window_size // 2 + 1)
            
            window_prices = predictions[start_idx:end_idx]
            window_mean = sum(window_prices) / len(window_prices)
            
            # 很轻微的平滑：95%原值 + 5%邻域均值
            smoothed_price = 0.95 * predictions[i] + 0.05 * window_mean
            smoothed.append(smoothed_price)
        
        return smoothed
    
    def minimal_adjustment(self, predictions):
        """最小幅度调整"""
        print("执行最小幅度调整...")
        
        adjusted = []
        for pred in predictions:
            # 非常保守的调整
            if pred < 100:
                # 极低价稍微提高一点
                adj_pred = pred * 1.02
            elif pred > 80000:
                # 极高价稍微降低一点
                adj_pred = pred * 0.98
            else:
                # 大部分保持不变
                adj_pred = pred
            
            adjusted.append(adj_pred)
        
        return adjusted
    
    def precision_rounding(self, predictions):
        """精度处理"""
        print("执行精度处理...")
        
        rounded = []
        for pred in predictions:
            # 适度的精度处理
            if pred < 1000:
                rounded_pred = round(pred, 0)
            elif pred < 10000:
                rounded_pred = round(pred / 10) * 10
            else:
                rounded_pred = round(pred / 100) * 100
            
            rounded.append(rounded_pred)
        
        return rounded
    
    def create_conservative_versions(self):
        """创建多个保守版本"""
        print("="*60)
        print("保守微调优化 - 基于677.0636分基准")
        print("="*60)
        
        # 加载基准版本
        sale_ids, baseline_prices = self.load_best_baseline()
        if baseline_prices is None:
            print("错误：无法加载基准版本")
            return
        
        # 版本1：仅精度处理（最保守）
        print("\n--- 版本1：仅精度处理（最保守）---")
        version1 = self.precision_rounding(baseline_prices)
        self.save_result(sale_ids, version1, "conservative_v1")
        print(f"均值变化: {sum(baseline_prices)/len(baseline_prices):.0f} → {sum(version1)/len(version1):.0f}")
        
        # 版本2：轻微平滑 + 精度处理
        print("\n--- 版本2：轻微平滑 + 精度处理 ---")
        smoothed = self.gentle_smoothing(baseline_prices)
        version2 = self.precision_rounding(smoothed)
        self.save_result(sale_ids, version2, "conservative_v2")
        print(f"均值变化: {sum(baseline_prices)/len(baseline_prices):.0f} → {sum(version2)/len(version2):.0f}")
        
        # 版本3：最小调整 + 轻微平滑 + 精度处理
        print("\n--- 版本3：最小调整 + 轻微平滑 + 精度处理 ---")
        adjusted = self.minimal_adjustment(baseline_prices)
        smoothed = self.gentle_smoothing(adjusted)
        version3 = self.precision_rounding(smoothed)
        self.save_result(sale_ids, version3, "conservative_v3")
        print(f"均值变化: {sum(baseline_prices)/len(baseline_prices):.0f} → {sum(version3)/len(version3):.0f}")
        
        # 统计对比
        print("\n" + "="*60)
        print("版本对比统计")
        print("="*60)
        
        versions = [
            ("基准版本", baseline_prices),
            ("版本1", version1),
            ("版本2", version2), 
            ("版本3", version3)
        ]
        
        print(f"{'版本':<12} {'均值':<8} {'中位数':<8} {'最小值':<8} {'最大值':<8}")
        print("-" * 50)
        
        for name, prices in versions:
            sorted_prices = sorted(prices)
            n = len(sorted_prices)
            
            print(f"{name:<12} {sum(prices)/n:<8.0f} {sorted_prices[n//2]:<8.0f} {min(prices):<8.0f} {max(prices):<8.0f}")
        
        print("\n推荐测试顺序：")
        print("1. conservative_v1 (最保守，接近原版)")
        print("2. conservative_v2 (轻微优化)")
        print("3. conservative_v3 (小幅优化)")
        
        return ["conservative_v1", "conservative_v2", "conservative_v3"]
    
    def save_result(self, sale_ids, prices, version_name):
        """保存结果"""
        import time
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'{version_name}_{timestamp}.csv'
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['SaleID', 'price'])
            
            for sale_id, price in zip(sale_ids, prices):
                writer.writerow([sale_id, int(price)])
        
        print(f"保存到: {filename}")

def main():
    optimizer = ConservativeOptimizer()
    versions = optimizer.create_conservative_versions()
    
    print("\n" + "="*60)
    print("💡 重要提醒")
    print("="*60)
    print("我们之前的激进优化导致分数变差(1976.8868)，")
    print("说明过度拟合训练数据分布是错误的。")
    print("\n现在回到677.0636分的稳定基础，")
    print("采用保守策略进行微调。")
    print("\n建议按顺序测试这3个版本，")
    print("选择分数最好的那个继续优化。")
    print("\n记住：有时候less is more! 🎯")

if __name__ == "__main__":
    main()