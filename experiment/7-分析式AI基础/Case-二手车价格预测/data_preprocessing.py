#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车数据预处理脚本
处理训练集和测试集，包括缺失值处理、异常值处理、特征编码等
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleLabelEncoder:
    """简单的标签编码器"""
    
    def __init__(self):
        self.classes_ = None
        self.class_to_index = {}
    
    def fit_transform(self, y):
        """拟合并转换"""
        unique_values = pd.Series(y).unique()
        self.classes_ = sorted([str(val) for val in unique_values if pd.notna(val)])
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self.transform(y)
    
    def transform(self, y):
        """转换"""
        result = []
        for val in y:
            str_val = str(val)
            if str_val in self.class_to_index:
                result.append(self.class_to_index[str_val])
            else:
                result.append(-1)  # 未知类别
        return result

class CarDataPreprocessor:
    """二手车数据预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        self.label_encoders = {}  # 存储标签编码器
        self.median_values = {}   # 存储中位数值
        self.is_fitted = False    # 是否已拟合
        
    def load_data(self, train_path, test_path):
        """加载训练集和测试集"""
        print("正在加载数据...")
        
        # 加载训练集
        self.train_df = pd.read_csv(train_path, sep=' ')
        print(f"训练集加载完成，形状: {self.train_df.shape}")
        
        # 加载测试集
        self.test_df = pd.read_csv(test_path, sep=' ')
        print(f"测试集加载完成，形状: {self.test_df.shape}")
        
        return self.train_df, self.test_df
    
    def remove_irrelevant_features(self, df):
        """删除无关特征"""
        print("删除无关特征...")
        
        # 要删除的特征列表
        features_to_remove = ['SaleID', 'name', 'offerType', 'seller']
        
        # 检查哪些特征实际存在于数据中
        existing_features = [col for col in features_to_remove if col in df.columns]
        
        if existing_features:
            df = df.drop(columns=existing_features)
            print(f"已删除特征: {existing_features}")
        else:
            print("未找到需要删除的特征")
            
        return df
    
    def handle_missing_values(self, df, is_train=True):
        """处理缺失值，采用中位数补充"""
        print("处理缺失值...")
        
        # 数值型特征列表
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 如果有price列且不是训练阶段，则排除price列
        if 'price' in numeric_features and not is_train:
            numeric_features.remove('price')
        
        if is_train:
            # 训练阶段：计算并保存中位数
            for col in numeric_features:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    self.median_values[col] = median_val
                    df[col].fillna(median_val, inplace=True)
                    print(f"  {col}: 用中位数 {median_val:.2f} 填充 {df[col].isnull().sum()} 个缺失值")
        else:
            # 测试阶段：使用保存的中位数
            for col in numeric_features:
                if col in self.median_values and df[col].isnull().sum() > 0:
                    df[col].fillna(self.median_values[col], inplace=True)
                    print(f"  {col}: 用训练集中位数 {self.median_values[col]:.2f} 填充缺失值")
        
        return df
    
    def handle_outliers(self, df, is_train=True):
        """处理异常值"""
        print("处理异常值...")
        
        # 处理price异常值（仅训练集）
        if 'price' in df.columns and is_train:
            original_count = len(df)
            # 删除价格异常值（小于等于0或过大的值）
            df = df[(df['price'] > 0) & (df['price'] <= 100000)]
            removed_count = original_count - len(df)
            print(f"  price异常值: 删除了 {removed_count} 条记录")
        
        # 处理power异常值
        if 'power' in df.columns:
            original_max = df['power'].max()
            outlier_count = (df['power'] > 600).sum()
            df.loc[df['power'] > 600, 'power'] = 600
            print(f"  power异常值: 将 {outlier_count} 个超过600的值设置为600 (原最大值: {original_max})")
        
        return df
    
    def convert_data_types(self, df):
        """数据类型转换"""
        print("进行数据类型转换...")
        
        # 转换日期字段
        date_columns = ['regDate', 'creatDate']
        for col in date_columns:
            if col in df.columns:
                # 将数字日期转换为日期格式 (YYYYMMDD -> YYYY-MM-DD)
                df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors='coerce')
                print(f"  {col}: 转换为日期类型")
        
        # 处理notRepairedDamage字段
        if 'notRepairedDamage' in df.columns:
            # 将'-'替换为-1
            original_dash_count = (df['notRepairedDamage'] == '-').sum()
            df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', -1)
            
            # 转换为数值类型
            df['notRepairedDamage'] = pd.to_numeric(df['notRepairedDamage'], errors='coerce')
            print(f"  notRepairedDamage: 将 {original_dash_count} 个'-'值设置为-1")
        
        return df
    
    def encode_categorical_features(self, df, is_train=True):
        """分类特征标签编码"""
        print("进行分类特征标签编码...")
        
        # 需要编码的分类特征
        categorical_features = ['brand', 'bodyType', 'fuelType', 'gearbox', 'model', 'regionCode']
        
        # 过滤出实际存在的特征
        existing_categorical = [col for col in categorical_features if col in df.columns]
        
        for col in existing_categorical:
            if is_train:
                # 训练阶段：创建并拟合标签编码器
                le = SimpleLabelEncoder()
                # 处理缺失值
                df[col] = df[col].fillna(-999)  # 用特殊值标记缺失值
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  {col}: 标签编码完成，共 {len(le.classes_)} 个类别")
            else:
                # 测试阶段：使用已训练的编码器
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = df[col].fillna(-999)
                    df[col] = le.transform(df[col].astype(str))
                    print(f"  {col}: 使用训练集编码器进行转换")
        
        return df
    
    def create_date_features(self, df):
        """创建日期相关特征"""
        print("创建日期相关特征...")
        
        # 从注册日期提取年份、月份等特征
        if 'regDate' in df.columns:
            df['regYear'] = df['regDate'].dt.year
            df['regMonth'] = df['regDate'].dt.month
            # 处理日期解析失败的情况
            df['regYear'] = df['regYear'].fillna(df['regYear'].median())
            df['regMonth'] = df['regMonth'].fillna(df['regMonth'].median())
            print("  从regDate提取年份和月份特征")
        
        if 'creatDate' in df.columns:
            df['creatYear'] = df['creatDate'].dt.year
            df['creatMonth'] = df['creatDate'].dt.month
            # 处理日期解析失败的情况
            df['creatYear'] = df['creatYear'].fillna(df['creatYear'].median())
            df['creatMonth'] = df['creatMonth'].fillna(df['creatMonth'].median())
            print("  从creatDate提取年份和月份特征")
        
        # 计算车龄（基于2016年）
        if 'regYear' in df.columns:
            df['car_age'] = 2016 - df['regYear']
            # 确保车龄为正数
            df['car_age'] = df['car_age'].clip(lower=0)
            print("  计算车龄特征")
        
        # 删除原始日期列（可选）
        date_cols_to_drop = ['regDate', 'creatDate']
        existing_date_cols = [col for col in date_cols_to_drop if col in df.columns]
        if existing_date_cols:
            df = df.drop(columns=existing_date_cols)
            print(f"  删除原始日期列: {existing_date_cols}")
        
        return df
    
    def fit_transform_train(self, train_path):
        """拟合并转换训练集"""
        print("="*50)
        print("开始处理训练集")
        print("="*50)
        
        # 加载训练数据
        self.train_df = pd.read_csv(train_path, sep=' ')
        print(f"训练集原始形状: {self.train_df.shape}")
        
        # 数据预处理流程
        self.train_df = self.remove_irrelevant_features(self.train_df)
        self.train_df = self.handle_outliers(self.train_df, is_train=True)
        self.train_df = self.convert_data_types(self.train_df)
        self.train_df = self.create_date_features(self.train_df)
        self.train_df = self.handle_missing_values(self.train_df, is_train=True)
        self.train_df = self.encode_categorical_features(self.train_df, is_train=True)
        
        self.is_fitted = True
        print(f"训练集处理完成，最终形状: {self.train_df.shape}")
        
        return self.train_df
    
    def transform_test(self, test_path):
        """转换测试集"""
        if not self.is_fitted:
            raise ValueError("必须先拟合训练集再处理测试集")
        
        print("="*50)
        print("开始处理测试集")
        print("="*50)
        
        # 加载测试数据
        self.test_df = pd.read_csv(test_path, sep=' ')
        print(f"测试集原始形状: {self.test_df.shape}")
        
        # 数据预处理流程（注意不处理price相关的异常值）
        self.test_df = self.remove_irrelevant_features(self.test_df)
        self.test_df = self.handle_outliers(self.test_df, is_train=False)
        self.test_df = self.convert_data_types(self.test_df)
        self.test_df = self.create_date_features(self.test_df)
        self.test_df = self.handle_missing_values(self.test_df, is_train=False)
        self.test_df = self.encode_categorical_features(self.test_df, is_train=False)
        
        print(f"测试集处理完成，最终形状: {self.test_df.shape}")
        
        return self.test_df
    
    def save_processed_data(self, train_output_path, test_output_path, preprocessor_path):
        """保存处理后的数据和预处理器"""
        print("="*50)
        print("保存处理后的数据")
        print("="*50)
        
        # 保存处理后的数据集
        self.train_df.to_csv(train_output_path, index=False)
        print(f"训练集已保存至: {train_output_path}")
        
        self.test_df.to_csv(test_output_path, index=False)
        print(f"测试集已保存至: {test_output_path}")
        
        # 保存预处理器
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'median_values': self.median_values,
            'is_fitted': self.is_fitted
        }
        
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        print(f"预处理器已保存至: {preprocessor_path}")
    
    def get_data_summary(self):
        """获取数据摘要信息"""
        print("="*50)
        print("数据处理摘要")
        print("="*50)
        
        if hasattr(self, 'train_df'):
            print(f"训练集最终形状: {self.train_df.shape}")
            print(f"训练集特征列表: {list(self.train_df.columns)}")
            
            if 'price' in self.train_df.columns:
                print(f"目标变量统计:")
                print(f"  均值: {self.train_df['price'].mean():.2f}")
                print(f"  中位数: {self.train_df['price'].median():.2f}")
                print(f"  标准差: {self.train_df['price'].std():.2f}")
        
        if hasattr(self, 'test_df'):
            print(f"测试集最终形状: {self.test_df.shape}")
            print(f"测试集特征列表: {list(self.test_df.columns)}")

def main():
    """主函数"""
    print("二手车数据预处理开始...")
    
    # 文件路径
    train_path = 'used_car_train_20200313.csv'
    test_path = 'used_car_testB_20200421.csv'
    
    # 输出路径
    train_output_path = 'processed_train_data.csv'
    test_output_path = 'processed_test_data.csv'
    preprocessor_path = 'data_preprocessor.pkl'
    
    try:
        # 创建预处理器
        preprocessor = CarDataPreprocessor()
        
        # 处理训练集
        train_processed = preprocessor.fit_transform_train(train_path)
        
        # 处理测试集
        test_processed = preprocessor.transform_test(test_path)
        
        # 保存处理后的数据
        preprocessor.save_processed_data(
            train_output_path, 
            test_output_path, 
            preprocessor_path
        )
        
        # 输出摘要信息
        preprocessor.get_data_summary()
        
        print("="*50)
        print("数据预处理完成！")
        print("="*50)
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()