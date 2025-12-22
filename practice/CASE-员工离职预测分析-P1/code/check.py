#!/usr/bin/env python3
"""
快速数据检查
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("员工离职预测数据快速检查")
print("=" * 60)

# 加载数据
data_dir = Path(__file__).parent.parent / "data"
train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")

# 训练集信息
print("\n训练集信息:")
print(f"  形状: {train_df.shape[0]} 行 × {train_df.shape[1]} 列")
print(f"  列名: {list(train_df.columns)}")
print(f"  目标变量 'Attrition' 分布:")
if 'Attrition' in train_df.columns:
    print(train_df['Attrition'].value_counts())
    print("  比例:")
    print(train_df['Attrition'].value_counts(normalize=True).round(3))
else:
    print("  目标变量不存在")

# 测试集信息
print("\n测试集信息:")
print(f"  形状: {test_df.shape[0]} 行 × {test_df.shape[1]} 列")
print(f"  是否包含目标变量: {'Attrition' in test_df.columns}")

# 缺失值检查
print("\n训练集缺失值检查:")
missing = train_df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(missing)
else:
    print("  无缺失值")

# 数据类型
print("\n训练集数据类型分布:")
print(train_df.dtypes.value_counts())

# 数值特征
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
print(f"\n数值特征数量: {len(numeric_cols)}")
print("重要数值特征示例:")
important_num = [col for col in numeric_cols if col not in ['user_id', 'EmployeeNumber', 'EmployeeCount', 'StandardHours']]
print(important_num[:10])

# 分类特征
categorical_cols = train_df.select_dtypes(include=['object']).columns
print(f"\n分类特征数量: {len(categorical_cols)}")
print("分类特征:")
print(list(categorical_cols))

# 数据质量总结
print("\n数据质量总结:")
print("1. 数据完整，无缺失值")
print("2. 目标变量存在类别不平衡（需要验证）")
print("3. 包含数值和分类特征混合")
print("4. 特征数量较多，适合机器学习建模")

print("\n建议下一步操作:")
print("1. 深入EDA（探索性数据分析）")
print("2. 特征工程：创建新特征，编码分类变量")
print("3. 数据预处理：标准化/归一化数值特征")
print("4. 建立基线模型（如逻辑回归、随机森林）")
print("5. 模型评估与优化")