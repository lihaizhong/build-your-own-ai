# -*- coding: utf-8 -*-
"""
深度特征分析脚本
分析当前模型特征处理的问题，进一步优化到MAE < 500
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def get_project_path(*paths):
    """获取项目路径的统一方法"""
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, *paths)

def load_and_analyze_data():
    """加载数据并进行深度分析"""
    print("=== 深度数据分析 ===")
    
    # 加载原始数据
    train_raw = pd.read_csv(get_project_path('data', 'used_car_train_20200313.csv'), sep=' ')
    test_raw = pd.read_csv(get_project_path('data', 'used_car_testB_20200421.csv'), sep=' ')
    
    print(f"原始训练集: {train_raw.shape}")
    print(f"原始测试集: {test_raw.shape}")
    
    return train_raw, test_raw

def analyze_potential_issues(train_data, test_data):
    """分析潜在的特征处理问题"""
    print("\n=== 特征问题分析 ===")
    
    issues = []
    
    # 1. 检查异常值处理是否合理
    print("1. 异常值分析...")
    
    # power字段分析
    if 'power' in train_data.columns:
        power_stats = train_data['power'].describe()
        power_outliers = train_data[train_data['power'] > 600]['power'].count()
        print(f"Power异常值 (>600): {power_outliers} 条记录")
        
        if power_outliers > 0:
            issues.append("power字段存在超过600的异常值，需要按规范处理为600")
    
    # 2. 检查缺失值处理
    print("2. 缺失值分析...")
    train_missing = train_data.isnull().sum()
    test_missing = test_data.isnull().sum()
    
    for col in train_missing[train_missing > 0].index:
        train_pct = train_missing[col] / len(train_data) * 100
        test_pct = test_missing.get(col, 0) / len(test_data) * 100
        print(f"{col}: 训练集{train_pct:.2f}%, 测试集{test_pct:.2f}%")
        
        if abs(train_pct - test_pct) > 5:
            issues.append(f"{col}字段训练集和测试集缺失比例差异较大")
    
    # 3. 检查分类特征编码
    print("3. 分类特征分析...")
    categorical_cols = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    
    for col in categorical_cols:
        if col in train_data.columns and col in test_data.columns:
            train_unique = train_data[col].nunique()
            test_unique = test_data[col].nunique()
            
            # 检查训练集和测试集的类别是否一致
            train_values = set(train_data[col].dropna().astype(str))
            test_values = set(test_data[col].dropna().astype(str))
            
            only_in_train = train_values - test_values
            only_in_test = test_values - train_values
            
            if only_in_train or only_in_test:
                print(f"{col}: 训练集独有{len(only_in_train)}个值, 测试集独有{len(only_in_test)}个值")
                if len(only_in_test) > 0:
                    issues.append(f"{col}字段测试集存在训练集未见过的类别")
    
    # 4. 检查数值特征分布差异
    print("4. 数值特征分布分析...")
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'price' and col in test_data.columns]
    
    for col in numeric_cols[:10]:  # 检查前10个数值特征
        train_mean = train_data[col].mean()
        test_mean = test_data[col].mean()
        diff_pct = abs(train_mean - test_mean) / train_mean * 100 if train_mean != 0 else 0
        
        if diff_pct > 10:
            print(f"{col}: 均值差异{diff_pct:.2f}% (训练:{train_mean:.2f}, 测试:{test_mean:.2f})")
            issues.append(f"{col}字段训练集和测试集分布差异较大")
    
    return issues

def create_enhanced_features(train_data, test_data):
    """创建增强特征，解决发现的问题"""
    print("\n=== 增强特征工程 ===")
    
    # 复制数据
    train_enhanced = train_data.copy()
    test_enhanced = test_data.copy()
    
    # 1. 按规范处理power异常值
    if 'power' in train_enhanced.columns:
        print("处理power异常值...")
        original_power_outliers = (train_enhanced['power'] > 600).sum()
        train_enhanced.loc[train_enhanced['power'] > 600, 'power'] = 600
        test_enhanced.loc[test_enhanced['power'] > 600, 'power'] = 600
        print(f"修正了 {original_power_outliers} 个power异常值")
    
    # 2. 改进缺失值处理 - 按规范使用中位数和众数
    print("改进缺失值处理...")
    
    # 数值型特征用中位数填充
    numeric_cols = train_enhanced.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'price':
            median_val = train_enhanced[col].median()
            train_enhanced[col] = train_enhanced[col].fillna(median_val)
            if col in test_enhanced.columns:
                test_enhanced[col] = test_enhanced[col].fillna(median_val)
    
    # 分类特征用众数填充
    categorical_cols = train_enhanced.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if col != 'price':
            mode_val = train_enhanced[col].mode().iloc[0] if len(train_enhanced[col].mode()) > 0 else 'unknown'
            train_enhanced[col] = train_enhanced[col].fillna(mode_val)
            if col in test_enhanced.columns:
                test_enhanced[col] = test_enhanced[col].fillna(mode_val)
    
    # 3. 正确处理regDate - 按规范提取年份计算车龄
    if 'regDate' in train_enhanced.columns:
        print("正确处理regDate...")
        current_year = 2020
        
        # 提取年份
        train_enhanced['reg_year'] = train_enhanced['regDate'] // 10000
        test_enhanced['reg_year'] = test_enhanced['regDate'] // 10000
        
        # 计算车龄
        train_enhanced['car_age'] = current_year - train_enhanced['reg_year']
        test_enhanced['car_age'] = current_year - test_enhanced['reg_year']
        
        # 确保车龄为正数
        train_enhanced['car_age'] = np.maximum(train_enhanced['car_age'], 1)
        test_enhanced['car_age'] = np.maximum(test_enhanced['car_age'], 1)
        
        # 提取月份和日期
        train_enhanced['reg_month'] = (train_enhanced['regDate'] % 10000) // 100
        test_enhanced['reg_month'] = (test_enhanced['regDate'] % 10000) // 100
        
        train_enhanced['reg_day'] = train_enhanced['regDate'] % 100
        test_enhanced['reg_day'] = test_enhanced['regDate'] % 100
    
    # 4. 创建更多有意义的交互特征
    if 'kilometer' in train_enhanced.columns and 'car_age' in train_enhanced.columns:
        # 年均里程数
        train_enhanced['km_per_year'] = train_enhanced['kilometer'] / train_enhanced['car_age']
        test_enhanced['km_per_year'] = test_enhanced['kilometer'] / test_enhanced['car_age']
        
        # 里程使用强度分类
        train_usage = pd.cut(
            train_enhanced['km_per_year'], 
            bins=[0, 5000, 15000, 30000, np.inf], 
            labels=[0, 1, 2, 3]
        )
        train_enhanced['usage_intensity'] = train_usage.fillna(0).astype(int)
        
        test_usage = pd.cut(
            test_enhanced['km_per_year'], 
            bins=[0, 5000, 15000, 30000, np.inf], 
            labels=[0, 1, 2, 3]
        )
        test_enhanced['usage_intensity'] = test_usage.fillna(0).astype(int)
    
    # 5. 功率相关特征
    if 'power' in train_enhanced.columns:
        # 功率分档
        train_power = pd.cut(
            train_enhanced['power'], 
            bins=[0, 60, 100, 150, 200, 600], 
            labels=[0, 1, 2, 3, 4]
        )
        train_enhanced['power_level'] = train_power.fillna(0).astype(int)
        
        test_power = pd.cut(
            test_enhanced['power'], 
            bins=[0, 60, 100, 150, 200, 600], 
            labels=[0, 1, 2, 3, 4]
        )
        test_enhanced['power_level'] = test_power.fillna(0).astype(int)
        
        if 'kilometer' in train_enhanced.columns:
            # 功率里程效率
            train_enhanced['power_efficiency'] = train_enhanced['power'] / (train_enhanced['kilometer'] + 1)
            test_enhanced['power_efficiency'] = test_enhanced['power'] / (test_enhanced['kilometer'] + 1)
    
    # 6. 改进分类特征编码
    categorical_features = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    
    for col in categorical_features:
        if col in train_enhanced.columns and col in test_enhanced.columns:
            # 合并训练和测试集进行统一编码
            all_values = pd.concat([
                train_enhanced[col].astype(str), 
                test_enhanced[col].astype(str)
            ]).unique()
            
            le = LabelEncoder()
            le.fit(all_values)
            
            train_enhanced[col] = le.transform(train_enhanced[col].astype(str))
            test_enhanced[col] = le.transform(test_enhanced[col].astype(str))
    
    # 7. 处理价格异常值 - 更保守的处理策略
    if 'price' in train_enhanced.columns:
        # 使用更宽松的范围，避免过度删除高价样本
        price_q005 = train_enhanced['price'].quantile(0.005)
        price_q995 = train_enhanced['price'].quantile(0.995)
        
        valid_idx = (train_enhanced['price'] >= price_q005) & (train_enhanced['price'] <= price_q995)
        train_enhanced = train_enhanced[valid_idx].reset_index(drop=True)
        
        print(f"价格范围: {train_enhanced['price'].min():.2f} - {train_enhanced['price'].max():.2f}")
        print(f"保留样本比例: {len(train_enhanced)/len(train_data)*100:.1f}%")
    
    print(f"增强特征后训练集: {train_enhanced.shape}")
    print(f"增强特征后测试集: {test_enhanced.shape}")
    
    return train_enhanced, test_enhanced

def analyze_feature_importance(X_train, y_train):
    """分析特征重要性"""
    print("\n=== 特征重要性分析 ===")
    
    # 使用随机森林分析特征重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 获取特征重要性
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 重要特征:")
    print(importance_df.head(15).to_string(index=False))
    
    # 使用互信息分析
    print("\n计算互信息...")
    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    mi_df = pd.DataFrame({
        'feature': X_train.columns,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("Top 15 互信息特征:")
    print(mi_df.head(15).to_string(index=False))
    
    return importance_df, mi_df

def create_optimized_model(X_train, y_train, X_test):
    """创建优化的随机森林模型"""
    print("\n=== 创建优化模型 ===")
    
    # 更精细的随机森林参数
    rf_configs = {
        'conservative': RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'balanced': RandomForestRegressor(
            n_estimators=200,
            max_depth=22,
            min_samples_split=12,
            min_samples_leaf=6,
            max_features='sqrt',
            random_state=123,
            n_jobs=-1
        ),
        'deeper': RandomForestRegressor(
            n_estimators=180,
            max_depth=25,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='log2',
            random_state=456,
            n_jobs=-1
        )
    }
    
    # 交叉验证选择最佳配置
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    best_mae = float('inf')
    best_config = None
    
    for name, model in rf_configs.items():
        print(f"测试 {name} 配置...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"  交叉验证MAE: {cv_mae:.4f} ± {cv_std:.4f}")
        
        if cv_mae < best_mae:
            best_mae = cv_mae
            best_config = name
    
    print(f"最佳配置: {best_config}, MAE: {best_mae:.4f}")
    
    # 使用最佳配置创建集成
    best_model = rf_configs[best_config]
    
    # 创建集成模型
    models = [
        best_model,
        RandomForestRegressor(n_estimators=220, max_depth=20, min_samples_split=14, random_state=789, n_jobs=-1),
        RandomForestRegressor(n_estimators=160, max_depth=24, min_samples_split=8, random_state=999, n_jobs=-1)
    ]
    
    predictions = []
    for i, model in enumerate(models):
        print(f"训练模型 {i+1}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.maximum(pred, 0)  # 确保非负
        predictions.append(pred)
    
    # 集成预测
    ensemble_pred = np.mean(predictions, axis=0)
    
    print(f"集成预测统计:")
    print(f"  均值: {ensemble_pred.mean():.2f}")
    print(f"  中位数: {np.median(ensemble_pred):.2f}")
    print(f"  范围: {ensemble_pred.min():.2f} - {ensemble_pred.max():.2f}")
    
    return ensemble_pred, best_mae

def main():
    """主函数"""
    print("开始深度特征分析...")
    
    # 1. 加载数据
    train_raw, test_raw = load_and_analyze_data()
    
    # 2. 分析潜在问题
    issues = analyze_potential_issues(train_raw, test_raw)
    
    print(f"\n发现的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    # 3. 创建增强特征
    train_enhanced, test_enhanced = create_enhanced_features(train_raw, test_raw)
    
    # 4. 准备建模数据
    feature_cols = [col for col in train_enhanced.columns if col != 'price']
    feature_cols = [col for col in feature_cols if col in test_enhanced.columns]
    
    X_train = train_enhanced[feature_cols]
    y_train = train_enhanced['price']
    X_test = test_enhanced[feature_cols]
    
    print(f"\n最终特征数量: {len(feature_cols)}")
    print(f"训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")
    
    # 5. 特征重要性分析
    importance_df, mi_df = analyze_feature_importance(X_train, y_train)
    
    # 6. 创建优化模型
    ensemble_pred, cv_mae = create_optimized_model(X_train, y_train, X_test)
    
    # 7. 保存结果 - 按规范保存到结果报告目录
    submission = pd.DataFrame({
        'SaleID': range(len(ensemble_pred)),
        'price': ensemble_pred
    })
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 按规范命名和保存路径
    results_dir = get_project_path('prediction_result')
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    filename = os.path.join(results_dir, f'rf_result_{timestamp}.csv')
    submission.to_csv(filename, index=False)
    
    print(f"\n=== 分析完成 ===")
    print(f"交叉验证MAE: {cv_mae:.4f}")
    print(f"结果已保存到: {filename}")
    print(f"预测范围: {ensemble_pred.min():.2f} - {ensemble_pred.max():.2f}")
    print(f"预测均值: {ensemble_pred.mean():.2f}")
    
    return filename

if __name__ == "__main__":
    result_file = main()
    print("深度特征分析完成！")