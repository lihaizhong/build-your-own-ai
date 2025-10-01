#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测 - 数据特征预处理脚本
对训练集和测试集进行全面的特征预处理，保存预处理结果供后续模型训练使用

处理流程：
1. 异常值处理（价格、power、notRepairedDamage）
2. 缺失值处理（众数填充 + 指示变量）
3. 时间特征提取（车龄、年份、季节、月份）
4. 分类特征编码（多种编码方法）
5. 目标变量变换（对数变换）
6. 多重共线性处理（删除强相关特征）
7. 数据质量验证

作者: AI助手
日期: 2025-10-01
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    print("警告：sklearn未安装，将使用简化版本的编码方法")
    SKLEARN_AVAILABLE = False
    
    # 简化版LabelEncoder
    class LabelEncoder:
        def __init__(self):
            self.classes_ = {}
            self.class_to_label = {}
            
        def fit_transform(self, data):
            unique_values = pd.Series(data).unique()
            self.classes_ = {i: val for i, val in enumerate(unique_values)}
            self.class_to_label = {val: i for i, val in enumerate(unique_values)}
            return pd.Series(data).map(self.class_to_label)
            
        def transform(self, data):
            return pd.Series(data).map(self.class_to_label).fillna(-1)
    
    # 简化版train_test_split
    def train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
        np.random.seed(random_state)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return (X.iloc[train_indices], X.iloc[test_indices], 
                y.iloc[train_indices], y.iloc[test_indices])

import warnings
warnings.filterwarnings('ignore')

class CarPricePreprocessor:
    """二手车价格预测数据预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        self.label_encoders = {}
        self.freq_encoders = {}
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        self.feature_stats = {}
        self.processing_report = {
            'outliers_removed': 0,
            'missing_filled': {},
            'features_created': 0,
            'features_removed': 0,
            'final_shape': None
        }
    
    def detect_price_outliers(self, df, method='iqr'):
        """
        检测价格异常值
        Args:
            df: 数据框
            method: 检测方法，默认使用IQR方法
        Returns:
            outlier_mask: 异常值掩码
        """
        if 'price' not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
            
        prices = df['price']
        
        if method == 'iqr':
            Q1 = prices.quantile(0.25)
            Q3 = prices.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (prices < lower_bound) | (prices > upper_bound)
        
        print(f"价格异常值检测完成：发现 {outlier_mask.sum()} 个异常值")
        return outlier_mask
    
    def handle_outliers(self, df):
        """
        处理异常值
        1. 价格异常值：直接删除
        2. power异常值：截断到600
        3. notRepairedDamage异常值：'-' 设置为 -1
        """
        print("\n=== 开始异常值处理 ===")
        original_shape = df.shape
        
        # 1. 处理价格异常值（直接删除）
        if 'price' in df.columns:
            outlier_mask = self.detect_price_outliers(df)
            df = df[~outlier_mask].copy()
            removed_count = outlier_mask.sum()
            self.processing_report['outliers_removed'] = removed_count
            print(f"删除价格异常值：{removed_count} 条记录")
        
        # 2. 处理power异常值（截断到600）
        if 'power' in df.columns:
            power_outliers = df['power'] > 600
            outlier_count = power_outliers.sum()
            df.loc[power_outliers, 'power'] = 600
            print(f"power异常值处理：{outlier_count} 条记录被截断到600")
        
        # 3. 处理notRepairedDamage异常值（'-' 设置为 -1）
        if 'notRepairedDamage' in df.columns:
            dash_count = (df['notRepairedDamage'] == '-').sum()
            df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', -1)
            print(f"notRepairedDamage异常值处理：{dash_count} 条记录从'-'转换为-1")
        
        print(f"异常值处理完成：{original_shape} -> {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """
        处理缺失值
        - fuelType: 众数填充 + 缺失值指示变量
        - gearbox, bodyType, model: 众数填充
        """
        print("\n=== 开始缺失值处理 ===")
        
        missing_info = {}
        
        # 处理fuelType（众数填充 + 缺失值指示变量）
        if 'fuelType' in df.columns:
            missing_count = df['fuelType'].isnull().sum()
            if missing_count > 0:
                # 创建缺失值指示变量
                df['fuelType_missing'] = df['fuelType'].isnull().astype(int)
                # 众数填充
                mode_value = df['fuelType'].mode().iloc[0] if not df['fuelType'].mode().empty else 0
                df['fuelType'].fillna(mode_value, inplace=True)
                missing_info['fuelType'] = missing_count
                print(f"fuelType缺失值处理：{missing_count} 个缺失值，已填充并创建指示变量")
        
        # 处理其他分类特征（众数填充）
        categorical_features = ['gearbox', 'bodyType', 'model']
        for feature in categorical_features:
            if feature in df.columns:
                missing_count = df[feature].isnull().sum()
                if missing_count > 0:
                    mode_value = df[feature].mode().iloc[0] if not df[feature].mode().empty else 0
                    df[feature].fillna(mode_value, inplace=True)
                    missing_info[feature] = missing_count
                    print(f"{feature}缺失值处理：{missing_count} 个缺失值已用众数填充")
        
        self.processing_report['missing_filled'] = missing_info
        print("缺失值处理完成")
        return df
    
    def extract_time_features(self, df):
        """
        提取时间特征
        从regDate字段提取：车龄、注册年份、注册季节、注册月份
        """
        print("\n=== 开始时间特征提取 ===")
        
        if 'regDate' not in df.columns:
            print("警告：未找到regDate字段，跳过时间特征提取")
            return df
        
        # 将regDate转换为日期格式
        df['regDate_str'] = df['regDate'].astype(str)
        
        # 解析日期（假设格式为YYYYMMDD）
        try:
            df['reg_year'] = df['regDate_str'].str[:4].astype(int)
            df['reg_month'] = df['regDate_str'].str[4:6].astype(int)
            df['reg_day'] = df['regDate_str'].str[6:8].astype(int)
            
            # 计算车龄（以2020年为基准）
            current_year = 2020
            df['car_age'] = current_year - df['reg_year']
            
            # 注册季节
            df['reg_season'] = df['reg_month'].apply(
                lambda x: 1 if x in [3,4,5] else 
                         2 if x in [6,7,8] else 
                         3 if x in [9,10,11] else 4
            )
            
            # 保留原始时间特征
            time_features = ['car_age', 'reg_year', 'reg_month', 'reg_season']
            self.processing_report['features_created'] += len(time_features)
            
            print(f"时间特征提取完成：{time_features}")
            
        except Exception as e:
            print(f"时间特征提取失败：{str(e)}")
        
        return df
    
    def encode_categorical_features(self, df, is_training=True):
        """
        分类特征编码
        - brand: 标签编码
        - model: 频次编码 + 标签编码
        - bodyType: 标签编码
        - fuelType: One-Hot编码
        - gearbox: 二进制编码
        - notRepairedDamage: One-Hot编码
        """
        print("\n=== 开始分类特征编码 ===")
        
        # 1. brand标签编码
        if 'brand' in df.columns:
            if is_training:
                self.label_encoders['brand'] = LabelEncoder()
                df['brand_encoded'] = self.label_encoders['brand'].fit_transform(df['brand'])
            else:
                df['brand_encoded'] = self.label_encoders['brand'].transform(df['brand'])
            print("brand标签编码完成")
        
        # 2. model频次编码 + 标签编码
        if 'model' in df.columns:
            if is_training:
                # 频次编码
                model_freq = df['model'].value_counts().to_dict()
                self.freq_encoders['model'] = model_freq
                df['model_freq'] = df['model'].map(model_freq)
                
                # 标签编码
                self.label_encoders['model'] = LabelEncoder()
                df['model_encoded'] = self.label_encoders['model'].fit_transform(df['model'])
            else:
                df['model_freq'] = df['model'].map(self.freq_encoders['model']).fillna(0)
                df['model_encoded'] = self.label_encoders['model'].transform(df['model'])
            print("model频次编码+标签编码完成")
        
        # 3. bodyType标签编码
        if 'bodyType' in df.columns:
            if is_training:
                self.label_encoders['bodyType'] = LabelEncoder()
                df['bodyType_encoded'] = self.label_encoders['bodyType'].fit_transform(df['bodyType'])
            else:
                df['bodyType_encoded'] = self.label_encoders['bodyType'].transform(df['bodyType'])
            print("bodyType标签编码完成")
        
        # 4. fuelType One-Hot编码
        if 'fuelType' in df.columns:
            fuel_dummies = pd.get_dummies(df['fuelType'], prefix='fuelType')
            df = pd.concat([df, fuel_dummies], axis=1)
            print("fuelType One-Hot编码完成")
        
        # 5. gearbox二进制编码
        if 'gearbox' in df.columns:
            df['gearbox_binary'] = df['gearbox'].astype(int)
            print("gearbox二进制编码完成")
        
        # 6. notRepairedDamage One-Hot编码
        if 'notRepairedDamage' in df.columns:
            damage_dummies = pd.get_dummies(df['notRepairedDamage'], prefix='notRepairedDamage')
            df = pd.concat([df, damage_dummies], axis=1)
            print("notRepairedDamage One-Hot编码完成")
        
        return df
    
    def apply_target_transformation(self, df):
        """
        目标变量变换
        对price进行对数变换以改善分布
        """
        print("\n=== 开始目标变量变换 ===")
        
        if 'price' in df.columns:
            # 检查是否有零值或负值
            if (df['price'] <= 0).any():
                print("警告：price中存在零值或负值，将使用log1p变换")
                df['price_log'] = np.log1p(df['price'])
            else:
                df['price_log'] = np.log(df['price'])
            
            print("price对数变换完成")
        
        return df
    
    def remove_multicollinearity_features(self, df):
        """
        删除多重共线性特征
        删除：v_1, v_7, v_4, v_8, v_2, v_12
        """
        print("\n=== 开始多重共线性处理 ===")
        
        features_to_remove = ['v_1', 'v_7', 'v_4', 'v_8', 'v_2', 'v_12']
        
        removed_features = []
        for feature in features_to_remove:
            if feature in df.columns:
                df.drop(feature, axis=1, inplace=True)
                removed_features.append(feature)
        
        self.processing_report['features_removed'] = len(removed_features)
        print(f"删除多重共线性特征：{removed_features}")
        
        return df
    
    def remove_irrelevant_features(self, df):
        """
        删除无关特征
        删除：SaleID, name, offerType, seller
        """
        print("\n=== 开始删除无关特征 ===")
        
        irrelevant_features = ['SaleID', 'name', 'offerType', 'seller']
        
        removed_features = []
        for feature in irrelevant_features:
            if feature in df.columns:
                df.drop(feature, axis=1, inplace=True)
                removed_features.append(feature)
        
        print(f"删除无关特征：{removed_features}")
        
        return df
    
    def validate_data_quality(self, df, dataset_name="数据集"):
        """
        数据质量验证
        """
        print(f"\n=== {dataset_name}质量验证 ===")
        
        # 1. 完整性检查
        missing_count = df.isnull().sum().sum()
        completeness_rate = (1 - missing_count / (df.shape[0] * df.shape[1])) * 100
        print(f"数据完整率: {completeness_rate:.2f}%")
        
        # 2. 异常值比例检查
        if 'price' in df.columns:
            outlier_mask = self.detect_price_outliers(df)
            outlier_rate = (outlier_mask.sum() / len(df)) * 100
            print(f"价格异常值比例: {outlier_rate:.2f}%")
        
        # 3. 数据类型检查
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"数值特征数量: {len(numeric_cols)}")
        
        # 4. 特征有效性评分（简化版）
        if 'price' in df.columns:
            correlations = df[numeric_cols].corr()['price'].abs()
            avg_correlation = correlations.drop('price').mean()
            print(f"特征有效性评分: {avg_correlation:.3f}")
        
        # 质量指标评估
        quality_checks = {
            '数据完整率 > 99%': completeness_rate > 99,
            '异常值比例 < 1%': outlier_rate < 1 if 'price' in df.columns else True,
            '特征有效性评分 > 0.8': avg_correlation > 0.8 if 'price' in df.columns else True
        }
        
        print("质量指标检查:")
        for check, passed in quality_checks.items():
            status = "✅ 通过" if passed else "❌ 未通过"
            print(f"  {check}: {status}")
        
        return df
    
    def prepare_modeling_data(self, df):
        """
        建模数据准备
        分层抽样划分训练集和验证集
        """
        print("\n=== 建模数据准备 ===")
        
        if 'price' not in df.columns:
            print("测试集无需划分")
            return df, None
        
        # 分层抽样（基于价格分位数）
        df['price_quartile'] = pd.qcut(df['price'], q=4, labels=False)
        
        # 准备特征和目标变量
        feature_cols = [col for col in df.columns if col not in ['price', 'price_log', 'price_quartile']]
        X = df[feature_cols]
        y = df['price_log'] if 'price_log' in df.columns else df['price']
        
        # 分层划分
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['price_quartile']
        )
        
        # 重新组合数据
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        
        print(f"训练集大小: {train_data.shape}")
        print(f"验证集大小: {val_data.shape}")
        
        return train_data, val_data
    
    def process_dataset(self, df, is_training=True):
        """
        完整的数据预处理流程
        """
        print(f"\n{'='*60}")
        print(f"开始处理{'训练集' if is_training else '测试集'}")
        print(f"原始数据形状: {df.shape}")
        print(f"{'='*60}")
        
        # 1. 异常值处理
        df = self.handle_outliers(df)
        
        # 2. 缺失值处理
        df = self.handle_missing_values(df)
        
        # 3. 时间特征提取
        df = self.extract_time_features(df)
        
        # 4. 分类特征编码
        df = self.encode_categorical_features(df, is_training)
        
        # 5. 目标变量变换（仅训练集）
        if is_training and 'price' in df.columns:
            df = self.apply_target_transformation(df)
        
        # 6. 删除无关特征
        df = self.remove_irrelevant_features(df)
        
        # 7. 多重共线性处理
        df = self.remove_multicollinearity_features(df)
        
        # 8. 数据质量验证
        df = self.validate_data_quality(df, "训练集" if is_training else "测试集")
        
        self.processing_report['final_shape'] = df.shape
        
        print(f"\n预处理完成！最终数据形状: {df.shape}")
        
        return df

def main():
    """主函数"""
    print("二手车价格预测 - 数据特征预处理")
    print("="*60)
    
    # 创建临时数据目录
    temp_dir = "临时数据"
    model_dir = "最终模型"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"创建临时数据目录: {temp_dir}")
    
    # 初始化预处理器
    preprocessor = CarPricePreprocessor()
    
    try:
        # 读取训练集
        print("\n读取训练集...")
        train_df = pd.read_csv("训练数据/used_car_train_20200313.csv", sep=' ')
        print(f"训练集加载成功: {train_df.shape}")
        
        # 处理训练集
        train_processed = preprocessor.process_dataset(train_df, is_training=True)
        
        # 准备建模数据（划分训练集和验证集）
        train_final, val_final = preprocessor.prepare_modeling_data(train_processed)
        
        # 保存训练集
        train_save_path = os.path.join(temp_dir, "used_car_train_preprocess.csv")
        train_processed.to_csv(train_save_path, index=False)
        print(f"训练集保存成功: {train_save_path}")
        
        # 读取测试集
        print("\n读取测试集...")
        test_df = pd.read_csv("训练数据/used_car_testB_20200421.csv", sep=' ')
        print(f"测试集加载成功: {test_df.shape}")
        
        # 处理测试集
        test_processed = preprocessor.process_dataset(test_df, is_training=False)
        
        # 保存测试集
        test_save_path = os.path.join(temp_dir, "used_car_testB_preprocess.csv")
        test_processed.to_csv(test_save_path, index=False)
        print(f"测试集保存成功: {test_save_path}")
        
        # 保存预处理器对象
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"预处理器保存成功: {preprocessor_path}")
        
        # 生成处理报告
        print("\n" + "="*60)
        print("数据预处理完成报告")
        print("="*60)
        print(f"异常值删除数量: {preprocessor.processing_report['outliers_removed']}")
        print(f"缺失值填充情况: {preprocessor.processing_report['missing_filled']}")
        print(f"新增特征数量: {preprocessor.processing_report['features_created']}")
        print(f"删除特征数量: {preprocessor.processing_report['features_removed']}")
        print(f"最终数据形状: {preprocessor.processing_report['final_shape']}")
        
        print(f"\n✅ 数据预处理完成！")
        print(f"📁 训练集: {train_save_path}")
        print(f"📁 测试集: {test_save_path}")
        print(f"🔧 预处理器: {preprocessor_path}")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()