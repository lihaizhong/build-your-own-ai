#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型版本对比可视化脚本
用于绘制各版本模型的学习曲线对比图
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def load_learning_curve_data(version):
    """加载指定版本的学习曲线数据"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'user_data', f'modeling_{version}', '学习曲线数据.csv')
    
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        print(f"未找到{version}版本的学习曲线数据: {data_path}")
        return None

def plot_learning_curve_comparison():
    """绘制各版本模型学习曲线对比图"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 定义颜色和标记
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', 'D', '^', 'v']
    versions = ['v6', 'v7', 'v8', 'v9', 'v10']
    
    # 绘制训练集MAE对比
    axes[0].set_title('各版本模型训练集MAE对比')
    axes[0].set_xlabel('训练样本数')
    axes[0].set_ylabel('平均绝对误差 (MAE)')
    axes[0].grid(True, alpha=0.3)
    
    # 绘制验证集MAE对比
    axes[1].set_title('各版本模型验证集MAE对比')
    axes[1].set_xlabel('训练样本数')
    axes[1].set_ylabel('平均绝对误差 (MAE)')
    axes[1].grid(True, alpha=0.3)
    
    # 加载各版本数据并绘制
    for i, version in enumerate(versions):
        data = load_learning_curve_data(version)
        if data is not None:
            # 绘制训练集MAE
            axes[0].plot(data['train_sizes'], data['train_scores_mean'], 
                        marker=markers[i], color=colors[i], 
                        label=f'{version.upper()}训练集', alpha=0.7)
            
            # 绘制验证集MAE
            axes[1].plot(data['train_sizes'], data['val_scores_mean'], 
                        marker=markers[i], color=colors[i], 
                        label=f'{version.upper()}验证集', alpha=0.7)
    
    # 添加图例
    axes[0].legend()
    axes[1].legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(project_root, 'user_data')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, '模型版本学习曲线对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"模型版本学习曲线对比图已保存到: {os.path.join(save_path, '模型版本学习曲线对比.png')}")

def generate_gap_analysis():
    """生成训练集与验证集MAE差距分析"""
    versions = ['v6', 'v7', 'v8', 'v9', 'v10']
    gaps = []
    
    for version in versions:
        data = load_learning_curve_data(version)
        if data is not None:
            # 计算最后一点的差距
            final_train_mae = data['train_scores_mean'].iloc[-1]
            final_val_mae = data['val_scores_mean'].iloc[-1]
            gap = final_val_mae - final_train_mae
            gaps.append(gap)
            print(f"{version.upper()}版本最后一点MAE差距: {gap:.2f}")
    
    # 绘制差距对比图
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(versions, gaps, color=['red', 'blue', 'green', 'orange', 'purple'])
    plt.title('各版本模型训练集与验证集MAE差距对比')
    plt.xlabel('模型版本')
    plt.ylabel('MAE差距 (验证集 - 训练集)')
    plt.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, gap in zip(bars, gaps):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{gap:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图像
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(project_root, 'user_data')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, '模型版本MAE差距对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"模型版本MAE差距对比图已保存到: {os.path.join(save_path, '模型版本MAE差距对比.png')}")

def main():
    """主函数"""
    print("开始生成模型版本对比图...")
    
    # 绘制学习曲线对比
    plot_learning_curve_comparison()
    
    # 生成差距分析
    generate_gap_analysis()
    
    print("模型版本对比分析完成!")

if __name__ == "__main__":
    main()