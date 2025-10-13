#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
学习曲线可视化脚本
用于绘制V6/V7模型的学习曲线图
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_learning_curve_from_data(csv_path, save_path, model_version="V6"):
    """从CSV数据绘制学习曲线"""
    # 读取学习曲线数据
    data = pd.read_csv(csv_path)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.title(f'{model_version}模型学习曲线')
    plt.xlabel('训练样本数')
    plt.ylabel('平均绝对误差 (MAE)')
    
    plt.grid()
    
    # 绘制训练集和验证集曲线
    plt.fill_between(data['train_sizes'], 
                     data['train_scores_mean'] - data['train_scores_std'],
                     data['train_scores_mean'] + data['train_scores_std'], 
                     alpha=0.1, color="r")
    plt.fill_between(data['train_sizes'], 
                     data['val_scores_mean'] - data['val_scores_std'],
                     data['val_scores_mean'] + data['val_scores_std'], 
                     alpha=0.1, color="g")
    
    plt.plot(data['train_sizes'], data['train_scores_mean'], 'o-', color="r",
             label="训练集得分")
    plt.plot(data['train_sizes'], data['val_scores_mean'], 'o-', color="g",
             label="验证集得分")
    
    plt.legend(loc="best")
    plt.tight_layout()
    
    # 保存图像
    filename = f'{model_version}模型学习曲线.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"学习曲线图已保存到: {os.path.join(save_path, filename)}")

def main():
    # 检查命令行参数
    model_version = "V6"
    if len(sys.argv) > 1:
        model_version = sys.argv[1].upper()
    
    # 设置路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'user_data', f'modeling_{model_version.lower()}')
    data_path = os.path.join(data_dir, '学习曲线数据.csv')
    save_path = data_dir
    
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 绘制学习曲线
    if os.path.exists(data_path):
        plot_learning_curve_from_data(data_path, save_path, model_version)
    else:
        print(f"未找到学习曲线数据文件: {data_path}")

if __name__ == "__main__":
    main()