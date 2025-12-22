#!/usr/bin/env python3
"""
员工离职预测数据探索性分析（EDA）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文显示和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_data():
    """加载训练和测试数据"""
    data_dir = Path(__file__).parent.parent / "data"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, test_df

def basic_info(df, name="数据集"):
    """输出数据集基本信息"""
    print(f"\n{'='*60}")
    print(f"{name}基本信息")
    print(f"{'='*60}")
    print(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"\n列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\n数据类型:")
    print(df.dtypes.value_counts())
    
    print(f"\n前5行数据:")
    print(df.head())
    
    print(f"\n缺失值统计:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'缺失数量': missing, '缺失百分比%': missing_pct})
    print(missing_df[missing_df['缺失数量'] > 0])

def target_analysis(df, target_col='Attrition'):
    """目标变量分析"""
    print(f"\n{'='*60}")
    print("目标变量分析")
    print(f"{'='*60}")
    
    if target_col in df.columns:
        target_counts = df[target_col].value_counts()
        target_pct = df[target_col].value_counts(normalize=True) * 100
        
        print(f"目标变量 '{target_col}' 分布:")
        for value, count in target_counts.items():
            print(f"  {value}: {count} ({target_pct[value]:.2f}%)")
        
        # 可视化目标变量分布
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        target_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title(f'{target_col} 分布')
        plt.xlabel(target_col)
        plt.ylabel('数量')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
                colors=['lightblue', 'lightcoral'])
        plt.title(f'{target_col} 占比')
        
        plt.tight_layout()
        plt.savefig('../docs/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print(f"目标变量 '{target_col}' 不存在于数据中")

def numeric_feature_analysis(df):
    """数值特征分析"""
    print(f"\n{'='*60}")
    print("数值特征分析")
    print(f"{'='*60}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"数值特征数量: {len(numeric_cols)}")
    print(f"数值特征: {list(numeric_cols)}")
    
    if len(numeric_cols) > 0:
        print(f"\n数值特征描述性统计:")
        print(df[numeric_cols].describe().T.round(2))
        
        # 相关性分析（仅限部分重要特征）
        important_num_cols = [col for col in numeric_cols if col not in ['user_id', 'EmployeeNumber', 'EmployeeCount', 'StandardHours']]
        if len(important_num_cols) > 1:
            correlation = df[important_num_cols].corr()
            print(f"\n数值特征相关性 (前10个特征):")
            print(correlation.iloc[:10, :10])
            
            # 相关性热图
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title('数值特征相关性热图')
            plt.tight_layout()
            plt.savefig('../docs/numeric_correlation.png', dpi=300, bbox_inches='tight')
            plt.show()

def categorical_feature_analysis(df):
    """分类特征分析"""
    print(f"\n{'='*60}")
    print("分类特征分析")
    print(f"{'='*60}")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"分类特征数量: {len(categorical_cols)}")
    print(f"分类特征: {list(categorical_cols)}")
    
    for col in categorical_cols:
        print(f"\n特征 '{col}' 的分布:")
        value_counts = df[col].value_counts()
        print(f"  唯一值数量: {df[col].nunique()}")
        for value, count in value_counts.head(10).items():
            pct = count / len(df) * 100
            print(f"  {value}: {count} ({pct:.2f}%)")
        if len(value_counts) > 10:
            print(f"  ... 还有 {len(value_counts) - 10} 个其他值")

def feature_target_relationship(df, target_col='Attrition'):
    """特征与目标变量关系分析"""
    if target_col not in df.columns:
        return
    
    print(f"\n{'='*60}")
    print("特征与目标变量关系分析")
    print(f"{'='*60}")
    
    # 分类特征与目标变量的关系
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != target_col]
    
    for col in categorical_cols[:5]:  # 只分析前5个分类特征
        print(f"\n特征 '{col}' 与 '{target_col}' 的关系:")
        cross_tab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        print(cross_tab.round(2))
        
        # 可视化
        plt.figure(figsize=(10, 6))
        cross_tab.plot(kind='bar', stacked=True)
        plt.title(f'{col} 与 {target_col} 的关系')
        plt.xlabel(col)
        plt.ylabel('百分比 (%)')
        plt.legend(title=target_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'../docs/{col}_vs_{target_col}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("员工离职预测数据探索性分析 (EDA)")
    print("=" * 60)
    
    # 加载数据
    train_df, test_df = load_data()
    
    # 训练集分析
    basic_info(train_df, "训练集")
    target_analysis(train_df)
    numeric_feature_analysis(train_df)
    categorical_feature_analysis(train_df)
    feature_target_relationship(train_df)
    
    # 测试集分析
    basic_info(test_df, "测试集")
    
    # 数据质量总结
    print(f"\n{'='*60}")
    print("数据质量总结")
    print(f"{'='*60}")
    print("1. 训练集大小:", train_df.shape)
    print("2. 测试集大小:", test_df.shape)
    print("3. 目标变量分布:")
    if 'Attrition' in train_df.columns:
        print("   - 离职 (Yes):", train_df['Attrition'].value_counts().get('Yes', 0))
        print("   - 未离职 (No):", train_df['Attrition'].value_counts().get('No', 0))
    print("4. 建议下一步:")
    print("   - 特征工程: 创建新特征，如工作满意度综合评分")
    print("   - 数据预处理: 编码分类变量，标准化数值特征")
    print("   - 建模: 尝试逻辑回归、随机森林、XGBoost等算法")
    print("   - 评估: 使用交叉验证和ROC曲线评估模型性能")

if __name__ == "__main__":
    main()