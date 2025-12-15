# -*- coding: utf-8 -*-
"""
针对性优化脚本 - 基于深度分析结果
目标: 将MAE从698降到500以内
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold, learning_curve, validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
import joblib

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def get_project_path(*paths):
    """获取项目路径的统一方法"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)

        return os.path.join(project_dir, *paths)
    except NameError:
        return os.path.join(os.getcwd(), *paths)

def get_user_data_path(*paths):
    """获取用户数据路径"""
    return get_project_path('user_data', *paths)

def save_models(models, version_name):
    """
    保存训练好的模型到model目录
    
    Parameters:
    -----------
    models : dict
        模型字典，key为模型名称，value为模型对象
    version_name : str
        版本名称，如'rf_baseline'
    """
    model_dir = get_project_path('model')
    os.makedirs(model_dir, exist_ok=True)
    
    saved_files = []
    for model_name, model_obj in models.items():
        if model_obj is not None:
            model_file = os.path.join(model_dir, f'{version_name}_{model_name}_model.pkl')
            joblib.dump(model_obj, model_file)
            saved_files.append(model_file)
            print(f"✅ 模型已保存: {model_file}")
    
    return saved_files

def advanced_missing_value_handler(train_data, test_data):
    """
    高级缺失值处理策略 - 专为随机森林优化

    策略1: 分组填充 - 根据相关特征分组计算统计值
    策略2: 缺失值指示变量 - 为高缺失率特征创建指示变量
    策略3: 多重插值 - 对关键特征使用多种方法
    策略4: 业务逻辑填充 - 基于领域知识的填充

    Args:
        train_data: 训练数据集
        test_data: 测试数据集
    
    Returns:
        处理后的训练集和测试集
    """
    print("\n=== 高级缺失值处理策略 ===")

    # 创建数据副本避免修改原始数据
    train_enhanced = train_data.copy()
    test_enhanced = test_data.copy()

    missing_report = {}

    # 策略1: 关键数值特征的智能分组填充
    key_features_config = {
        'power': {
            'group_by': ['brand', 'bodyType'],  # 按品牌和车身类型分组
            'fallback_groups': ['brand'],  # 如果组合分组样本不足，回退到单一分组
            'method': 'median',
            'create_missing_indicator': True
        },
        'kilometer': {
            'group_by': ['brand', 'car_age'] if 'car_age' in train_data.columns else ['brand'],
            'fallback_groups': ['brand'],
            'method': 'median',
            'create_missing_indicator': True
        }
    }

    for feature, config in key_features_config.items():
        if feature in train_enhanced.columns:
            missing_count = train_enhanced[feature].isnull().sum()
            if missing_count > 0:
                print(f"\n处理关键特征 {feature} ({missing_count} 个缺失值)...")

                # 创建缺失值指示变量
                if config['create_missing_indicator']:
                    train_enhanced[f'{feature}_was_missing'] = train_enhanced[feature].isnull().astype(int)
                    if feature in test_enhanced.columns:
                        test_enhanced[f'{feature}_was_missing'] = test_enhanced[feature].isnull().astype(int)

                # 智能分组填充
                filled_count = 0

                # 尝试主要分组策略
                if all(col in train_enhanced.columns for col in config['group_by']):
                    group_stats = train_enhanced.groupby(config['group_by'])[feature].agg(['median', 'count'])

                    for group_key, group_data in group_stats.iterrows():
                        if group_data['count'] >= 3:  # 至少3个样本才用分组统计
                            fill_value = group_data['median']

                            # 构建筛选条件
                            if len(config['group_by']) == 2:
                                mask_train = (train_enhanced[config['group_by'][0]] == group_key[0]) & \
                                           (train_enhanced[config['group_by'][1]] == group_key[1]) & \
                                           train_enhanced[feature].isnull()
                                mask_test = (test_enhanced[config['group_by'][0]] == group_key[0]) & \
                                          (test_enhanced[config['group_by'][1]] == group_key[1]) & \
                                          test_enhanced[feature].isnull()
                            else:
                                mask_train = (train_enhanced[config['group_by'][0]] == group_key) & \
                                           train_enhanced[feature].isnull()
                                mask_test = (test_enhanced[config['group_by'][0]] == group_key) & \
                                          test_enhanced[feature].isnull()

                            # 填充
                            count_filled = mask_train.sum()
                            if count_filled > 0:
                                train_enhanced.loc[mask_train, feature] = fill_value
                                filled_count += count_filled
                            
                            if feature in test_enhanced.columns:
                                test_enhanced.loc[mask_test, feature] = fill_value
                
                # 处理剩余缺失值 - 使用回退策略
                remaining_missing = train_enhanced[feature].isnull().sum()
                if remaining_missing > 0:
                    print(f"  使用回退策略处理剩余 {remaining_missing} 个缺失值...")
                    global_median = train_enhanced[feature].median()
                    train_enhanced[feature] = train_enhanced[feature].fillna(global_median)
                    if feature in test_enhanced.columns:
                        test_enhanced[feature] = test_enhanced[feature].fillna(global_median)
                
                missing_report[feature] = {
                    'original_missing': missing_count,
                    'group_filled': filled_count,
                    'global_filled': remaining_missing,
                    'strategy': '智能分组填充+缺失指示'
                }

    # 策略2: 分类特征的高级处理
    categorical_features = ['fuelType', 'gearbox', 'bodyType', 'model']

    for feature in categorical_features:
        if feature in train_enhanced.columns:
            missing_count = train_enhanced[feature].isnull().sum()
            missing_rate = missing_count / len(train_enhanced)

            if missing_count > 0:
                print(f"\n处理分类特征 {feature} ({missing_count} 个缺失值, {missing_rate:.2%})...")

                # 高缺失率特征创建指示变量
                if missing_rate >= 0.03:  # 3%以上创建指示变量
                    train_enhanced[f'{feature}_was_missing'] = train_enhanced[feature].isnull().astype(int)
                    if feature in test_enhanced.columns:
                        test_enhanced[f'{feature}_was_missing'] = test_enhanced[feature].isnull().astype(int)
                    strategy = '众数填充+缺失指示'
                else:
                    strategy = '众数填充'

                # 众数填充
                if len(train_enhanced[feature].mode()) > 0:
                    mode_val = train_enhanced[feature].mode().iloc[0]
                else:
                    mode_val = 'unknown'

                train_enhanced[feature] = train_enhanced[feature].fillna(mode_val)
                if feature in test_enhanced.columns:
                    test_enhanced[feature] = test_enhanced[feature].fillna(mode_val)

                missing_report[feature] = {
                    'original_missing': missing_count,
                    'missing_rate': missing_rate,
                    'strategy': strategy
                }

    # 策略3: 其他数值特征的标准处理
    numeric_cols = train_enhanced.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'price' and col not in key_features_config and not col.endswith('_missing') and not col.endswith('_was_missing'):
            missing_count = train_enhanced[col].isnull().sum()
            if missing_count > 0:
                median_val = train_enhanced[col].median()
                train_enhanced[col] = train_enhanced[col].fillna(median_val)
                if col in test_enhanced.columns:
                    test_enhanced[col] = test_enhanced[col].fillna(median_val)
                
                missing_report[col] = {
                    'original_missing': missing_count,
                    'strategy': '中位数填充'
                }
    
    # 输出处理报告
    print("\n=== 缺失值处理报告 ===")
    for feature, report in missing_report.items():
        print(f"{feature}: {report['original_missing']} 个缺失值 - {report['strategy']}")
    
    print(f"\n处理完成! 创建了 {sum(1 for col in train_enhanced.columns if col.endswith('_was_missing'))} 个缺失值指示变量")
    
    return train_enhanced, test_enhanced

def load_and_optimize_data():
    """基于分析结果优化数据加载 - 添加高级缺失值处理"""
    print("正在加载并优化数据...")

    # 加载原始数据 - 使用绝对路径
    train_path = get_project_path('data', 'used_car_train_20200313.csv')
    test_path = get_project_path('data', 'used_car_testB_20200421.csv')

    print(f"训练数据路径: {train_path}")
    print(f"测试数据路径: {test_path}")

    try:
        train_raw = pd.read_csv(train_path, sep=' ')
        test_raw = pd.read_csv(test_path, sep=' ')
    except Exception as e:
        print(f"数据加载错误: {e}")
        print("尝试使用逗号分隔符...")
        train_raw = pd.read_csv(train_path)
        test_raw = pd.read_csv(test_path)

    print(f"原始训练集: {train_raw.shape}")
    print(f"原始测试集: {test_raw.shape}")

    # 确保特征完全一致
    common_features = set(train_raw.columns) & set(test_raw.columns)
    feature_cols = [col for col in common_features if col != 'price']

    train_data = train_raw[feature_cols + ['price']].copy()
    test_data = test_raw[feature_cols].copy()

    # 1. 按规范处理power异常值 (发现143个>600的记录)
    print("处理power异常值...")
    power_outliers_train = 0
    power_outliers_test = 0

    if 'power' in train_data.columns:
        power_outliers_train = (train_data['power'] > 600).sum()
        train_data.loc[train_data['power'] > 600, 'power'] = 600
        print(f"训练集修正了 {power_outliers_train} 个power异常值")

    if 'power' in test_data.columns:
        power_outliers_test = (test_data['power'] > 600).sum()
        test_data.loc[test_data['power'] > 600, 'power'] = 600
        print(f"测试集修正了 {power_outliers_test} 个power异常值")

    # 2. 高级缺失值处理 - 替换原有的简单处理
    train_data, test_data = advanced_missing_value_handler(train_data, test_data)

    # 2. 优化缺失值处理 - 分组填充 + 缺失值指示变量
    print("优化缺失值处理...")

    # 创建缺失值统计
    missing_stats = {}

    # 优先处理关键数值特征 - 使用分组填充
    key_numeric_features = ['power', 'kilometer']
    for col in key_numeric_features:
        if col in train_data.columns:
            missing_count_train = train_data[col].isnull().sum()
            missing_count_test = test_data[col].isnull().sum() if col in test_data.columns else 0

            if missing_count_train > 0 or missing_count_test > 0:
                print(f"  处理 {col} 的 {missing_count_train + missing_count_test} 个缺失值...")

                # 创建缺失值指示变量
                train_data[f'{col}_missing'] = train_data[col].isnull().astype(int)
                if col in test_data.columns:
                    test_data[f'{col}_missing'] = test_data[col].isnull().astype(int)

                # 按品牌分组填充（如果brand可用）
                if 'brand' in train_data.columns and train_data[col].isnull().sum() > 0:
                    # 计算每个品牌的中位数
                    brand_medians = train_data.groupby('brand')[col].median().to_dict()

                    # 填充训练集
                    for brand, median_val in brand_medians.items():
                        mask = (train_data['brand'] == brand) & train_data[col].isnull()
                        train_data.loc[mask, col] = median_val

                    # 处理剩余的缺失值（用全局中位数）
                    global_median = train_data[col].median()
                    train_data[col] = train_data[col].fillna(global_median)

                    # 填充测试集
                    if col in test_data.columns:
                        for brand, median_val in brand_medians.items():
                            mask = (test_data['brand'] == brand) & test_data[col].isnull()
                            test_data.loc[mask, col] = median_val
                        test_data[col] = test_data[col].fillna(global_median)
                else:
                    # 回退到简单中位数填充
                    median_val = train_data[col].median()
                    train_data[col] = train_data[col].fillna(median_val)
                    if col in test_data.columns:
                        test_data[col] = test_data[col].fillna(median_val)

                missing_stats[col] = {'train': missing_count_train, 'test': missing_count_test, 'method': '分组中位数+指示变量'}

    # 处理其他数值特征 - 简单中位数填充
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'price' and col not in key_numeric_features and not col.endswith('_missing'):
            missing_count_train = train_data[col].isnull().sum()
            missing_count_test = test_data[col].isnull().sum() if col in test_data.columns else 0

            if missing_count_train > 0 or missing_count_test > 0:
                median_val = train_data[col].median()
                train_data[col] = train_data[col].fillna(median_val)
                if col in test_data.columns:
                    test_data[col] = test_data[col].fillna(median_val)
                missing_stats[col] = {'train': missing_count_train, 'test': missing_count_test, 'method': '中位数填充'}

    # 优化分类特征处理 - 创建高缺失率特征的指示变量
    categorical_cols = train_data.select_dtypes(exclude=[np.number]).columns
    high_missing_threshold = 0.05  # 5%以上缺失率创建指示变量

    for col in categorical_cols:
        if col != 'price':
            missing_count_train = train_data[col].isnull().sum()
            missing_count_test = test_data[col].isnull().sum() if col in test_data.columns else 0
            missing_rate = missing_count_train / len(train_data)

            if missing_count_train > 0 or missing_count_test > 0:
                # 高缺失率特征创建指示变量
                if missing_rate >= high_missing_threshold:
                    print(f"  为 {col} 创建缺失值指示变量 (缺失率: {missing_rate:.2%})")
                    train_data[f'{col}_missing'] = train_data[col].isnull().astype(int)
                    if col in test_data.columns:
                        test_data[f'{col}_missing'] = test_data[col].isnull().astype(int)
                    method = '众数填充+指示变量'
                else:
                    method = '众数填充'

                # 众数填充
                if len(train_data[col].mode()) > 0:
                    mode_val = train_data[col].mode().iloc[0]
                else:
                    mode_val = 'unknown'

                train_data[col] = train_data[col].fillna(mode_val)
                if col in test_data.columns:
                    test_data[col] = test_data[col].fillna(mode_val)

                missing_stats[col] = {'train': missing_count_train, 'test': missing_count_test, 'method': method}

    # 输出缺失值处理统计
    if missing_stats:
        print("\n缺失值处理统计:")
        for col, stats in missing_stats.items():
            print(f"  {col}: 训练集{stats['train']}个, 测试集{stats['test']}个 - {stats['method']}")
    else:
        print("  无缺失值需要处理")
    # 3. 分类特征编码优化
    categorical_features = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']

    for col in categorical_features:
        if col in train_data.columns and col in test_data.columns:
            # 合并训练和测试集进行统一编码，处理训练集独有值的问题
            all_values = pd.concat([
                train_data[col].astype(str), 
                test_data[col].astype(str)
            ]).unique()

            le = LabelEncoder()
            le.fit(all_values)

            train_data[col] = le.transform(train_data[col].astype(str))
            test_data[col] = le.transform(test_data[col].astype(str))

    # 4. 保守的价格异常值处理（避免过度删除高价样本）
    if 'price' in train_data.columns:
        # 使用更宽松的0.5%-99.5%范围
        price_q005 = train_data['price'].quantile(0.005)
        price_q995 = train_data['price'].quantile(0.995)

        valid_idx = (train_data['price'] >= price_q005) & (train_data['price'] <= price_q995)
        removed_count = len(train_data) - valid_idx.sum()
        train_data = train_data[valid_idx].reset_index(drop=True)

        print(f"价格范围: {train_data['price'].min():.2f} - {train_data['price'].max():.2f}")
        print(f"移除了 {removed_count} 个价格异常样本 ({removed_count/len(train_raw)*100:.2f}%)")

    print(f"优化后训练集: {train_data.shape}")
    print(f"优化后测试集: {test_data.shape}")

    return train_data, test_data

def create_targeted_features(train_data, test_data):
    """基于特征重要性分析创建针对性特征"""
    print("创建针对性特征...")
    
    # 5. 正确处理regDate - 按规范提取年份计算车龄
    if 'regDate' in train_data.columns:
        print("正确处理regDate时间特征...")

        current_year = datetime.now().year
        
        # 提取注册年份
        train_data['reg_year'] = train_data['regDate'] // 10000
        test_data['reg_year'] = test_data['regDate'] // 10000

        # 提取上线年份
        train_data['create_year'] = train_data['creatDate'] // 10000
        test_data['create_year'] = test_data['creatDate'] // 10000
        
        # 计算车龄
        train_data['car_age'] = current_year - train_data['reg_year']
        test_data['car_age'] = current_year - test_data['reg_year']
        
        # 确保车龄为正数
        train_data['car_age'] = np.maximum(train_data['car_age'], 1)
        test_data['car_age'] = np.maximum(test_data['car_age'], 1)
        
        # 提取月份（季节性特征）
        train_data['reg_month'] = (train_data['regDate'] % 10000) // 100
        test_data['reg_month'] = (test_data['regDate'] % 10000) // 100
        
        # 车龄分档（基于业务理解）
        age_bins = [0, 3, 7, 12, 20, 50]
        train_data['age_group'] = pd.Series(pd.cut(train_data['car_age'], bins=age_bins, labels=False)).fillna(4).astype(int)
        test_data['age_group'] = pd.Series(pd.cut(test_data['car_age'], bins=age_bins, labels=False)).fillna(4).astype(int)

        train_data.drop(columns=['name', 'offerType', 'seller', 'regDate', 'creatDate'], inplace=True)
        test_data.drop(columns=['name', 'offerType', 'seller', 'regDate', 'creatDate'], inplace=True)
    
    # 6. 基于重要特征创建交互特征
    if 'kilometer' in train_data.columns and 'car_age' in train_data.columns:
        # 年均里程数 (分析中发现这个特征有效)
        train_data['km_per_year'] = train_data['kilometer'] / train_data['car_age']
        test_data['km_per_year'] = test_data['kilometer'] / test_data['car_age']
        
        # 里程使用强度分类
        km_year_bins = [0, 8000, 18000, 35000, np.inf]
        train_data['usage_intensity'] = pd.Series(pd.cut(train_data['km_per_year'], bins=km_year_bins, labels=False)).fillna(0).astype(int)
        test_data['usage_intensity'] = pd.Series(pd.cut(test_data['km_per_year'], bins=km_year_bins, labels=False)).fillna(0).astype(int)

    # 7. 功率效率特征 (分析中排名第5)
    if 'power' in train_data.columns and 'kilometer' in train_data.columns:
        train_data['power_efficiency'] = train_data['power'] / (train_data['kilometer'] + 1)
        test_data['power_efficiency'] = test_data['power'] / (test_data['kilometer'] + 1)
        
        # 功率分档
        power_bins = [0, 75, 110, 150, 200, 600]
        train_data['power_level'] = pd.Series(pd.cut(train_data['power'], bins=power_bins, labels=False)).fillna(0).astype(int)
        test_data['power_level'] = pd.Series(pd.cut(test_data['power'], bins=power_bins, labels=False)).fillna(0).astype(int)
    
    # 8. 基于Top特征的交互 (v_0, v_12, v_3是最重要的)
    important_features = ['v_0', 'v_12', 'v_3']
    for feat in important_features:
        if feat in train_data.columns:
            # 与车龄的交互
            if 'car_age' in train_data.columns:
                train_data[f'{feat}_age_ratio'] = train_data[feat] / (train_data['car_age'] + 1)
                test_data[f'{feat}_age_ratio'] = test_data[feat] / (test_data['car_age'] + 1)
    
    # 9. 组合特征 - 基于业务理解
    if 'v_0' in train_data.columns and 'v_12' in train_data.columns:
        train_data['v0_v12_combo'] = train_data['v_0'] * train_data['v_12']
        test_data['v0_v12_combo'] = test_data['v_0'] * test_data['v_12']
    
    # 10. 新增重要特征组合（基于727分数进一步优化）
    if 'v_0' in train_data.columns and 'v_3' in train_data.columns:
        train_data['v0_v3_interaction'] = train_data['v_0'] * train_data['v_3']
        test_data['v0_v3_interaction'] = test_data['v_0'] * test_data['v_3']
    
    # 11. 基于车辆使用状况的综合评分
    if 'kilometer' in train_data.columns and 'car_age' in train_data.columns and 'power' in train_data.columns:
        # 综合车辆状况评分：里程 + 车龄 + 功率的综合考虑
        train_data['vehicle_condition'] = (
            (train_data['power'] / 100) / 
            (np.log1p(train_data['kilometer'] / 1000) * np.log1p(train_data['car_age']))
        )
        test_data['vehicle_condition'] = (
            (test_data['power'] / 100) / 
            (np.log1p(test_data['kilometer'] / 1000) * np.log1p(test_data['car_age']))
        )
    
    # 12. 品牌-车型组合特征（如果数据支持）
    if 'brand' in train_data.columns and 'model' in train_data.columns:
        train_data['brand_model'] = train_data['brand'].astype(str) + '_' + train_data['model'].astype(str)
        test_data['brand_model'] = test_data['brand'].astype(str) + '_' + test_data['model'].astype(str)
        
        # 对组合特征进行编码
        from sklearn.preprocessing import LabelEncoder
        le_brand_model = LabelEncoder()
        
        # 合并所有可能的组合值
        all_brand_model = pd.concat([
            train_data['brand_model'],
            test_data['brand_model']
        ]).unique()
        
        le_brand_model.fit(all_brand_model)
        train_data['brand_model_encoded'] = le_brand_model.transform(train_data['brand_model'])
        test_data['brand_model_encoded'] = le_brand_model.transform(test_data['brand_model'])
        
        # 删除原始字符串列
        train_data = train_data.drop(['brand_model'], axis=1)
        test_data = test_data.drop(['brand_model'], axis=1)
    
    print(f"特征工程后训练集: {train_data.shape}")
    print(f"特征工程后测试集: {test_data.shape}")
    
    return train_data, test_data

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """绘制特征重要性图"""
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title(f'前{top_n}个重要特征的重要性分析', fontsize=14, fontweight='bold')
    plt.xlabel('特征重要性', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {save_path}")
    
    plt.show()
    return feature_importance

def plot_learning_curve(model, X, y, cv=2, save_path=None):  # 减少CV折数
    """绘制学习曲线分析模型是否过拟合或欠拟合"""
    # 使用采样数据加速计算
    sample_size = min(12000, len(X))  # 限制样本大小
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
        # 使用索引对齐选择对应标签，避免将索引标签误用为整数位置
        y_sample = y.iloc[X_sample.index]
    else:
        X_sample, y_sample = X, y
    
    train_sizes = np.linspace(0.2, 1.0, 6)  # 减少训练大小点数
    train_sizes, train_scores, val_scores = learning_curve(  # type: ignore
        model, X_sample, y_sample, cv=cv, train_sizes=train_sizes, 
        scoring='neg_mean_absolute_error', n_jobs=1  # 减少并行度
    )
    
    train_mae = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mae = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mae, 'o-', color='blue', label='训练集MAE')
    plt.fill_between(train_sizes, train_mae - train_std, train_mae + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mae, 'o-', color='red', label='验证集MAE')
    plt.fill_between(train_sizes, val_mae - val_std, val_mae + val_std, alpha=0.1, color='red')
    
    plt.xlabel('训练样本数量', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('学习曲线分析 - 判断过拟合/欠拟合', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习曲线图已保存到: {save_path}")
    
    plt.show()
    
    # 分析结果
    final_gap = val_mae[-1] - train_mae[-1]
    print(f"\n学习曲线分析结果:")
    print(f"最终训练MAE: {train_mae[-1]:.4f}")
    print(f"最终验证MAE: {val_mae[-1]:.4f}")
    print(f"Gap: {final_gap:.4f}")
    
    if final_gap > 50:
        print("⚠️  模型可能存在过拟合，建议增加正则化或减少模型复杂度")
    elif final_gap < 20:
        print("🚀 模型泛化能力较好，可以考虑增加模型复杂度")
    else:
        print("✅ 模型复杂度较为合适")
    
    return train_mae, val_mae

def plot_validation_curve_analysis(X, y, param_name='max_depth', param_range=None, save_path=None):
    """绘制验证曲线分析参数优化空间"""
    if param_range is None:
        if param_name == 'max_depth':
            param_range = [5, 10, 15, 20, 25, 30, 35]
        elif param_name == 'n_estimators':
            param_range = [50, 100, 150, 200, 250, 300]
        elif param_name == 'min_samples_split':
            param_range = [2, 5, 10, 15, 20, 25]
        else:
            param_range = [1, 2, 3, 4, 5]
    
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # 使用更快的交叉验证配置
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=2, scoring='neg_mean_absolute_error', n_jobs=1  # 减少CV折数和线程数
    )
    
    train_mae = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mae = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mae, 'o-', color='blue', label='训练集MAE')
    plt.fill_between(param_range, train_mae - train_std, train_mae + train_std, alpha=0.1, color='blue')
    plt.plot(param_range, val_mae, 'o-', color='red', label='验证集MAE')
    plt.fill_between(param_range, val_mae - val_std, val_mae + val_std, alpha=0.1, color='red')
    
    plt.xlabel(f'{param_name} 参数值', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title(f'{param_name} 参数验证曲线 - 寻找最优参数', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"验证曲线图已保存到: {save_path}")
    
    plt.show()
    
    # 找到最优参数
    best_idx = np.argmin(val_mae)
    best_param = param_range[best_idx]
    best_mae = val_mae[best_idx]
    
    print(f"\n{param_name} 参数优化结果:")
    print(f"最优参数: {best_param}")
    print(f"最优MAE: {best_mae:.4f}")
    
    return best_param, best_mae

def plot_residual_analysis(y_true, y_pred, save_path=None):
    """绘制残差分析图"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 残差 vs 预测值
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[0, 0].set_xlabel('预测值', fontsize=12)
    axes[0, 0].set_ylabel('残差 (真实值 - 预测值)', fontsize=12)
    axes[0, 0].set_title('残差 vs 预测值', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 残差分布直方图
    axes[0, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
    axes[0, 1].set_xlabel('残差', fontsize=12)
    axes[0, 1].set_ylabel('频次', fontsize=12)
    axes[0, 1].set_title('残差分布直方图', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. QQ图 - 检验残差是否符合正态分布
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('QQ图 - 残差正态性检验', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 真实值 vs 预测值
    axes[1, 1].scatter(y_true, y_pred, alpha=0.6, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1, 1].set_xlabel('真实值', fontsize=12)
    axes[1, 1].set_ylabel('预测值', fontsize=12)
    axes[1, 1].set_title('真实值 vs 预测值', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"残差分析图已保存到: {save_path}")
    
    plt.show()
    
    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n残差分析结果:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"残差平均值: {residuals.mean():.4f}")
    print(f"残差标准差: {residuals.std():.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'residuals': residuals}

def plot_price_distribution_comparison(y_train, ensemble_pred, save_path=None):
    """对比真实价格与预测价格的分布"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_train, bins=50, alpha=0.7, label='训练集真实价格', color='blue', edgecolor='black')
    plt.hist(ensemble_pred, bins=50, alpha=0.7, label='测试集预测价格', color='red', edgecolor='black')
    plt.xlabel('价格', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.title('价格分布对比', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([y_train, ensemble_pred], label=['训练集真实价格', '测试集预测价格'])
    plt.ylabel('价格', fontsize=12)
    plt.title('价格分布箱线图', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"价格分布对比图已保存到: {save_path}")
    
    plt.show()
    
    print(f"\n价格分布统计对比:")
    print(f"训练集 - 平均: {y_train.mean():.2f}, 中位: {y_train.median():.2f}, 标准差: {y_train.std():.2f}")
    print(f"预测集 - 平均: {ensemble_pred.mean():.2f}, 中位: {np.median(ensemble_pred):.2f}, 标准差: {ensemble_pred.std():.2f}")

def create_optimized_rf_ensemble(X_train, y_train, X_test, enable_analysis=True):
    """创建优化的随机森林集成 - 重点解决过拟合问题"""
    print("创建抗过拟合随机森林集成...")
    
    # 基于特征数量调整参数
    n_features = X_train.shape[1]
    
    # 超保守的防过拟合配置 - 针对727分数进一步优化
    rf_models = [
        # 模型1：极度保守 - 最强正则化
        RandomForestRegressor(
            n_estimators=120,      # 减少树数量
            max_depth=8,           # 大幅减少深度
            min_samples_split=30,  # 显著增加最小分裂样本
            min_samples_leaf=20,   # 显著增加最小叶子样本
            max_features=0.4,      # 大幅减少特征采样
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            min_impurity_decrease=0.001  # 添加纯度阈值
        ),
        # 模型2：中等保守
        RandomForestRegressor(
            n_estimators=150,
            max_depth=10,          # 从15减到10
            min_samples_split=25,  # 从20增加到25
            min_samples_leaf=15,   # 从12增加到15
            max_features=0.5,      # 从sqrt改为固定比例
            bootstrap=True,
            random_state=123,
            n_jobs=-1,
            min_impurity_decrease=0.0005
        ),
        # 模型3：适度保守
        RandomForestRegressor(
            n_estimators=180,
            max_depth=12,          # 从18减到12
            min_samples_split=20,
            min_samples_leaf=12,
            max_features=0.6,      # 控制特征比例
            bootstrap=True,
            random_state=456,
            n_jobs=-1
        ),
        # 模型4：新增极简模型
        RandomForestRegressor(
            n_estimators=100,
            max_depth=6,           # 最浅深度
            min_samples_split=40,  # 最大分裂要求
            min_samples_leaf=25,   # 最大叶子要求
            max_features=0.3,      # 最少特征
            bootstrap=True,
            random_state=789,
            n_jobs=-1,
            min_impurity_decrease=0.002
        ),
        # 模型5：Bagging风格的简化模型
        RandomForestRegressor(
            n_estimators=200,
            max_depth=9,
            min_samples_split=35,
            min_samples_leaf=18,
            max_features=0.3,
            bootstrap=True,
            max_samples=0.8,       # 样本采样比例
            random_state=999,
            n_jobs=-1
        )
    ]
    
    # 训练集成模型并计算OOB分数
    predictions = []
    trained_models = []
    oob_scores = []
    
    for i, model in enumerate(rf_models):
        print(f"训练抗过拟合模型 {i+1}/{len(rf_models)}...")
        
        # 设置OOB评分
        model.oob_score = True
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.maximum(pred, 0)  # 确保非负
        
        predictions.append(pred)
        trained_models.append(model)
        oob_scores.append(model.oob_score_)
        
        print(f"  模型{i+1} - 预测范围: {pred.min():.2f} - {pred.max():.2f}, OOB Score: {model.oob_score_:.4f}")
    
    # 基于OOB分数的智能加权集成
    oob_weights = np.array(oob_scores)
    # 对负的OOB分数进行处理（转换为正数）
    if np.any(oob_weights < 0):
        oob_weights = oob_weights - np.min(oob_weights) + 0.01
    
    # 归一化权重
    oob_weights = oob_weights / oob_weights.sum()
    
    # 额外给简单模型（第1和第4个）加权
    simple_model_boost = [1.2, 1.0, 1.0, 1.3, 1.1]  # 简单模型权重提升
    final_weights = oob_weights * np.array(simple_model_boost)
    final_weights = final_weights / final_weights.sum()
    
    ensemble_pred = np.average(predictions, axis=0, weights=final_weights)
    
    print(f"\n集成预测统计:")
    print(f"  OOB分数: {[f'{score:.4f}' for score in oob_scores]}")
    print(f"  最终权重: {[f'{w:.3f}' for w in final_weights]}")
    print(f"  均值: {ensemble_pred.mean():.2f}")
    print(f"  中位数: {np.median(ensemble_pred):.2f}")
    print(f"  范围: {ensemble_pred.min():.2f} - {ensemble_pred.max():.2f}")
    # 如果启用分析，进行详细分析
    if enable_analysis:
        print("\n开始模型分析...")
        
        # 创建结果保存目录
        results_dir = get_project_path('prediction_result')
        os.makedirs(results_dir, exist_ok=True)
        
        # 使用最保守的模型进行分析
        main_model = trained_models[0]  # 使用第一个最保守的模型
        feature_names = X_train.columns.tolist()
        
        # 1. 特征重要性分析
        print("1. 绘制特征重要性图...")
        importance_path = get_user_data_path('feature_importance.png')
        feature_importance = plot_feature_importance(
            main_model, feature_names, top_n=20, save_path=importance_path
        )
        
        # 2. 学习曲线分析 - 重点关注过拟合
        print("2. 绘制学习曲线...")
        learning_path = get_user_data_path('learning_curve.png')
        train_mae, val_mae = plot_learning_curve(
            main_model, X_train, y_train, cv=3, save_path=learning_path
        )
        
        # 3. 参数验证曲线 - 验证当前参数是否合适
        print("3. 绘制参数验证曲线...")
        validation_path = get_user_data_path('validation_curve.png')
        # 验证max_depth参数
        best_depth, best_mae = plot_validation_curve_analysis(
            X_train.sample(n=min(8000, len(X_train)), random_state=42),
            y_train.iloc[:min(8000, len(y_train))],
            param_name='max_depth', 
            param_range=[6, 8, 10, 12, 15],  # 验证更保守的深度范围
            save_path=validation_path
        )
        
        # 4. 残差分析（使用交叉验证预测）
        print("4. 进行残差分析...")
        from sklearn.model_selection import cross_val_predict
        cv_pred = cross_val_predict(main_model, X_train, y_train, cv=3)
        residual_path = get_user_data_path('residual_analysis.png')
        residual_stats = plot_residual_analysis(
            y_train, cv_pred, save_path=residual_path
        )
        
        # 5. 价格分布对比
        print("5. 绘制价格分布对比...")
        distribution_path = get_user_data_path('price_distribution.png')
        plot_price_distribution_comparison(
            y_train, ensemble_pred, save_path=distribution_path
        )
        
        print(f"\n📊 所有分析图表已保存到: {get_user_data_path()}")
        
        # 返回分析结果
        analysis_results = {
            'feature_importance': feature_importance,
            'best_depth': best_depth,
            'residual_stats': residual_stats,
            'final_cv_mae': val_mae[-1] if len(val_mae) > 0 else None,
            'oob_scores': oob_scores,
            'model_weights': final_weights.tolist()
        }
        
        return ensemble_pred, analysis_results
    
    return ensemble_pred

def quick_cv_evaluation(X_train, y_train):
    """快速交叉验证评估 - 使用更保守的参数"""
    print("快速交叉验证评估...")
    
    # 使用极度保守的参数 - 与集成中最保守模型保持一致
    rf = RandomForestRegressor(
        n_estimators=100,      # 减少树数量
        max_depth=8,           # 与最保守模型一致
        min_samples_split=30,  # 大幅增加最小分裂样本
        min_samples_leaf=20,   # 大幅增加最小叶子样本
        max_features=0.4,      # 与最保守模型一致
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        min_impurity_decrease=0.001  # 添加纯度阈值防过拟合
    )
    
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # 使用3折保证稳定性
    cv_scores = cross_val_score(rf, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"交叉验证MAE: {cv_mae:.4f} ± {cv_std:.4f}")
    print(f"使用极度保守参数：max_depth=8, min_samples_split=30, min_samples_leaf=20")
    return cv_mae

def main():
    """主函数"""
    print("开始针对性优化...")
    print("目标: MAE从698降到500以内")
    
    # 1. 优化数据加载
    train_data, test_data = load_and_optimize_data()
    
    # 2. 创建针对性特征
    train_data, test_data = create_targeted_features(train_data, test_data)
    
    # 3. 准备建模数据
    feature_cols = [col for col in train_data.columns if col != 'price']
    feature_cols = [col for col in feature_cols if col in test_data.columns]
    
    X_train = train_data[feature_cols]
    y_train = train_data['price']
    X_test = test_data[feature_cols]
    
    print(f"\n最终特征数量: {len(feature_cols)}")
    print(f"训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")
    
    # 4. 快速交叉验证评估 - 使用采样加速
    print("4. 快速交叉验证评估...")
    sample_size = min(15000, len(X_train))  # 采样减少计算量
    X_sample = X_train.sample(n=sample_size, random_state=42)
    y_sample = y_train.iloc[X_sample.index]
    cv_mae = quick_cv_evaluation(X_sample, y_sample)
    
    # 5. 创建优化集成（启用分析功能）
    result = create_optimized_rf_ensemble(X_train, y_train, X_test, enable_analysis=True)
    
    # 检查返回值类型
    if isinstance(result, tuple):
        ensemble_pred, analysis_results = result
        print(f"\n📊 模型分析完成！")
        print(f"最优深度建议: {analysis_results.get('best_depth', 'N/A')}")
        if analysis_results.get('final_cv_mae'):
            print(f"学习曲线最终MAE: {analysis_results['final_cv_mae']:.4f}")
    else:
        ensemble_pred = result
        print(f"\n⚠️ 跳过了模型分析")
    
    # 6. 保存结果 - 按规范保存
    submission = pd.DataFrame({
        'SaleID': range(len(ensemble_pred)),
        'price': ensemble_pred
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 按规范保存到results目录
    results_dir = get_project_path('prediction_result')
    os.makedirs(results_dir, exist_ok=True)
    
    filename = os.path.join(results_dir, f'rf_result_{timestamp}.csv')
    submission.to_csv(filename, index=False)
    
    # 保存训练好的模型
    print("\n保存训练好的模型...")
    models_to_save = {}
    
    # 保存所有训练好的RF模型
    for i, model in enumerate(trained_models):
        models_to_save[f'rf_{i+1}'] = model
    
    if models_to_save:
        save_models(models_to_save, 'rf_baseline')
    
    print(f"\n=== 针对性优化完成 ===")
    print(f"交叉验证MAE: {cv_mae:.4f}")
    print(f"结果已保存到: {filename}")
    print(f"预测均值: {ensemble_pred.mean():.2f}")
    print(f"预测范围: {ensemble_pred.min():.2f} - {ensemble_pred.max():.2f}")
    
    # 预测性能评估
    if cv_mae < 500:
        print("🎉 目标达成！交叉验证MAE < 500")
    else:
        print(f"⚠️  还需优化，当前MAE {cv_mae:.0f}，目标 < 500")
    
    return filename

if __name__ == "__main__":
    result_file = main()
    print("针对性优化完成！")