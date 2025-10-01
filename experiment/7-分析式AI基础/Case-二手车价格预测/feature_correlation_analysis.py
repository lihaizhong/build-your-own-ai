#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多重共线性分析和特征选择建议
分析强相关特征对，给出最优特征保留建议
"""

import pandas as pd
import numpy as np

def analyze_multicollinearity():
    """
    分析多重共线性并给出特征选择建议
    """
    try:
        # 读取数据
        df = pd.read_csv("训练数据/used_car_train_20200313.csv", sep=' ')
        print("数据读取成功！")
        print(f"数据形状: {df.shape}")
        
        # 计算与目标变量price的相关性
        numeric_features = df.select_dtypes(include=[np.number]).columns
        target_corr = df[numeric_features].corr()['price'].abs().sort_values(ascending=False)
        
        print("\n" + "="*80)
        print("特征与目标变量price的相关性排序")
        print("="*80)
        for feature, corr in target_corr.items():
            if feature != 'price':
                print(f"{feature:12s}: {corr:.4f}")
        
        # 分析强相关特征对
        corr_matrix = df[numeric_features].corr()
        
        # 定义强相关特征对（根据EDA报告）- 按相关性强度排序
        strong_corr_pairs = [
            ('v_1', 'v_6', 0.9994),    # 最强相关
            ('v_2', 'v_7', 0.9737),
            ('v_4', 'v_9', 0.9629),
            ('v_5', 'v_7', -0.9394),   # 注意v_7已在上面处理
            ('v_4', 'v_13', 0.9346),   # 注意v_4已在上面处理
            ('v_3', 'v_8', -0.9332),
            ('v_1', 'v_10', -0.9219),  # 注意v_1已在第一个处理
            ('v_2', 'v_5', -0.9219),   # 注意v_2已在第二个处理
            ('v_3', 'v_12', -0.8113),
            ('v_2', 'v_11', 0.8009)    # 注意v_2已在前面处理
        ]
        
        print("\n" + "="*80)
        print("强相关特征对分析与选择建议")
        print("="*80)
        
        selected_features = set()
        removed_features = set()
        
        for feat1, feat2, pair_corr in strong_corr_pairs:
            if feat1 in removed_features or feat2 in removed_features:
                print(f"\n【特征对: {feat1} vs {feat2}】- 跳过（其中一个特征已被删除）")
                continue
                
            # 获取与目标变量的相关性
            corr1_target = target_corr.get(feat1, 0)
            corr2_target = target_corr.get(feat2, 0)
            
            # 验证特征对的相关性
            if feat1 in corr_matrix.columns and feat2 in corr_matrix.columns:
                actual_corr = corr_matrix.loc[feat1, feat2]
            else:
                actual_corr = pair_corr
            
            print(f"\n【特征对: {feat1} vs {feat2}】")
            print(f"  特征对相关性: {actual_corr:.4f}")
            print(f"  {feat1} vs price: {corr1_target:.4f}")
            print(f"  {feat2} vs price: {corr2_target:.4f}")
            
            # 选择与目标变量相关性更高的特征
            if corr1_target >= corr2_target:
                selected_features.add(feat1)
                removed_features.add(feat2)
                print(f"  ✅ 保留: {feat1} (与price相关性更高: {corr1_target:.4f})")
                print(f"  ❌ 删除: {feat2} (与price相关性较低: {corr2_target:.4f})")
            else:
                selected_features.add(feat2)
                removed_features.add(feat1)
                print(f"  ✅ 保留: {feat2} (与price相关性更高: {corr2_target:.4f})")
                print(f"  ❌ 删除: {feat1} (与price相关性较低: {corr1_target:.4f})")
        
        # 汇总建议
        print("\n" + "="*80)
        print("多重共线性处理总结")
        print("="*80)
        
        print(f"\n【建议保留的特征】({len(selected_features)}个):")
        for feature in sorted(selected_features):
            if feature not in removed_features:  # 确保不在删除列表中
                corr_val = target_corr.get(feature, 0)
                print(f"  ✅ {feature:8s} (与price相关性: {corr_val:.4f})")
        
        print(f"\n【建议删除的特征】({len(removed_features)}个):")
        for feature in sorted(removed_features):
            corr_val = target_corr.get(feature, 0)
            print(f"  ❌ {feature:8s} (与price相关性: {corr_val:.4f})")
        
        # 分析删除特征后的效果
        original_features = set(numeric_features) - {'price'}
        # 正确计算保留的特征（从原始特征中删除被移除的特征）
        remaining_features = original_features - removed_features
        
        print(f"\n【处理前后对比】:")
        print(f"  原始特征数量: {len(original_features)}")
        print(f"  删除特征数量: {len(removed_features)}")
        print(f"  保留特征数量: {len(remaining_features)}")
        print(f"  特征降维比例: {len(removed_features)/len(original_features)*100:.1f}%")
        
        # 保留特征的相关性分析
        remaining_features_list = list(remaining_features)
        if len(remaining_features_list) > 0:
            remaining_corr = target_corr[remaining_features_list].sort_values(ascending=False)
            print(f"\n【保留特征与price的相关性排序】:")
            for i, (feature, corr) in enumerate(remaining_corr.items(), 1):
                print(f"  {i:2d}. {feature:12s}: {corr:.4f}")
        
        # 输出最终的特征选择建议
        print("\n" + "="*80)
        print("最终特征选择建议")
        print("="*80)
        
        print("\n删除以下高相关特征（避免多重共线性）:")
        for feature in sorted(removed_features):
            print(f"  - {feature}")
        
        print(f"\n保留{len(remaining_features)}个有效特征，包括:")
        print("  - 所有低相关性的独立特征")
        print("  - 强相关特征对中与price相关性更高的特征")
        
        return list(remaining_features), list(removed_features)
        
    except Exception as e:
        print(f"分析过程中出错: {str(e)}")
        return [], []

if __name__ == "__main__":
    remaining_features, removed_features = analyze_multicollinearity()
