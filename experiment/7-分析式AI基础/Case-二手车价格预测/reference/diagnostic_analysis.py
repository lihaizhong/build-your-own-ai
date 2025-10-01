#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测诊断脚本
分析训练验证结果与实际考试结果的差异
专注于随机森林模型的深度诊断
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RFDiagnosticAnalyzer:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        try:
            # 尝试加载预处理过的数据
            if pd.io.common.file_exists('train_processed.csv') and pd.io.common.file_exists('test_processed.csv'):
                self.train_data = pd.read_csv('train_processed.csv')
                self.test_data = pd.read_csv('test_processed.csv')
                print("加载预处理数据成功")
            else:
                # 如果没有预处理数据，加载原始数据并进行简单预处理
                print("未找到预处理数据，加载原始数据...")
                train_raw = pd.read_csv('训练数据/used_car_train_20200313.csv', sep=' ')
                test_raw = pd.read_csv('训练数据/used_car_testB_20200421.csv', sep=' ')
                
                # 简单的数据预处理
                self.train_data, self.test_data = self._simple_preprocessing(train_raw, test_raw)
                print("简单预处理完成")
            
            print(f"训练集形状: {self.train_data.shape}")
            print(f"测试集形状: {self.test_data.shape}")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def _simple_preprocessing(self, train_raw, test_raw):
        """简单的数据预处理"""
        print("进行简单数据预处理...")
        
        # 基本清理
        train_data = train_raw.copy()
        test_data = test_raw.copy()
        
        # 处理缺失值
        for col in train_data.columns:
            if train_data[col].dtype == 'object':
                train_data[col] = train_data[col].fillna('unknown')
                if col in test_data.columns:
                    test_data[col] = test_data[col].fillna('unknown')
            else:
                train_data[col] = train_data[col].fillna(train_data[col].median())
                if col in test_data.columns:
                    test_data[col] = test_data[col].fillna(train_data[col].median())
        
        # 基本特征编码
        from sklearn.preprocessing import LabelEncoder
        
        categorical_cols = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
        
        for col in categorical_cols:
            if col in train_data.columns and col in test_data.columns:
                le = LabelEncoder()
                # 合并训练和测试集的类别
                all_values = pd.concat([train_data[col], test_data[col]]).astype(str)
                le.fit(all_values)
                train_data[col] = le.transform(train_data[col].astype(str))
                test_data[col] = le.transform(test_data[col].astype(str))
        
        return train_data, test_data
    
    def analyze_data_consistency(self):
        """分析训练集和测试集的一致性"""
        print("\n=== 数据一致性分析 ===")
        
        # 检查特征一致性
        train_cols = set(self.train_data.columns)
        test_cols = set(self.test_data.columns)
        
        # 排除目标变量
        exclude_cols = ['price', 'price_log', 'price_quartile']
        train_features = train_cols - set(exclude_cols)
        test_features = test_cols - set(exclude_cols)
        
        only_in_train = train_features - test_features
        only_in_test = test_features - train_features
        common_features = train_features & test_features
        
        print(f"训练集独有特征: {len(only_in_train)}")
        if only_in_train:
            print(f"  {list(only_in_train)[:10]}")  # 只显示前10个
        
        print(f"测试集独有特征: {len(only_in_test)}")
        if only_in_test:
            print(f"  {list(only_in_test)[:10]}")
            
        print(f"共同特征数量: {len(common_features)}")
        
        # 分析特征分布差异
        print("\n=== 特征分布分析 ===")
        common_numeric_features = []
        for col in list(common_features)[:20]:  # 分析前20个共同特征
            if col in self.train_data.columns and col in self.test_data.columns:
                if self.train_data[col].dtype in ['int64', 'float64']:
                    common_numeric_features.append(col)
        
        distribution_diffs = {}
        for col in common_numeric_features:
            train_mean = self.train_data[col].mean()
            test_mean = self.test_data[col].mean()
            train_std = self.train_data[col].std()
            test_std = self.test_data[col].std()
            
            mean_diff = abs(train_mean - test_mean) / (train_mean + 1e-8)
            std_diff = abs(train_std - test_std) / (train_std + 1e-8)
            
            distribution_diffs[col] = {
                'mean_diff_ratio': mean_diff,
                'std_diff_ratio': std_diff
            }
        
        # 显示分布差异最大的特征
        sorted_features = sorted(distribution_diffs.items(), 
                               key=lambda x: x[1]['mean_diff_ratio'], reverse=True)
        
        print("分布差异最大的10个特征:")
        for col, diff in sorted_features[:10]:
            print(f"  {col}: 均值差异比率={diff['mean_diff_ratio']:.4f}, 标准差差异比率={diff['std_diff_ratio']:.4f}")
        
        return common_features, distribution_diffs
    
    def prepare_data_for_modeling(self, common_features):
        """准备建模数据"""
        print("\n=== 准备建模数据 ===")
        
        # 确保使用完全相同的特征
        feature_cols = list(common_features)
        
        # 检查目标变量
        if 'price' in self.train_data.columns:
            X_train = self.train_data[feature_cols]
            y_train = self.train_data['price']
            X_test = self.test_data[feature_cols]
            
            print(f"建模特征数量: {len(feature_cols)}")
            print(f"训练集样本数: {len(X_train)}")
            print(f"测试集样本数: {len(X_test)}")
            print(f"目标变量范围: {y_train.min():.2f} - {y_train.max():.2f}")
            
            return X_train, y_train, X_test, feature_cols
        else:
            print("错误: 训练集中未找到price列")
            return None, None, None, None
    
    def cross_validation_analysis(self, X_train, y_train):
        """交叉验证分析"""
        print("\n=== 交叉验证分析 ===")
        
        # 定义多种随机森林配置
        rf_configs = {
            'conservative': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'moderate': RandomForestRegressor(
                n_estimators=200, 
                max_depth=20, 
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'aggressive': RandomForestRegressor(
                n_estimators=300, 
                max_depth=30, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        
        cv_results = {}
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in rf_configs.items():
            print(f"测试 {name} 配置...")
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=kfold, scoring='neg_mean_absolute_error')
            
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            cv_results[name] = {
                'cv_mae_mean': cv_mae,
                'cv_mae_std': cv_std,
                'cv_scores': cv_scores
            }
            
            print(f"  交叉验证MAE: {cv_mae:.4f} ± {cv_std:.4f}")
        
        return cv_results, rf_configs
    
    def holdout_validation_analysis(self, X_train, y_train):
        """留出验证分析"""
        print("\n=== 留出验证分析 ===")
        
        # 分割训练集进行留出验证
        split_idx = int(len(X_train) * 0.8)
        
        # 按照价格排序后分割，模拟真实考试数据分布
        price_order = y_train.argsort()
        train_idx = price_order[:split_idx]
        val_idx = price_order[split_idx:]
        
        X_train_split = X_train.iloc[train_idx]
        y_train_split = y_train.iloc[train_idx]
        X_val_split = X_train.iloc[val_idx]
        y_val_split = y_train.iloc[val_idx]
        
        print(f"训练子集大小: {len(X_train_split)}")
        print(f"验证子集大小: {len(X_val_split)}")
        print(f"训练子集价格范围: {y_train_split.min():.2f} - {y_train_split.max():.2f}")
        print(f"验证子集价格范围: {y_val_split.min():.2f} - {y_val_split.max():.2f}")
        
        # 测试不同模型
        models_to_test = {
            'simple_rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'deep_rf': RandomForestRegressor(n_estimators=300, max_depth=30, random_state=42),
            'ensemble_rf': None  # 将在后面实现
        }
        
        holdout_results = {}
        for name, model in models_to_test.items():
            if model is None:
                continue
                
            print(f"测试 {name}...")
            model.fit(X_train_split, y_train_split)
            y_pred = model.predict(X_val_split)
            
            mae = mean_absolute_error(y_val_split, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
            r2 = r2_score(y_val_split, y_pred)
            
            holdout_results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
        
        return holdout_results
    
    def prediction_distribution_analysis(self, X_train, y_train, X_test):
        """预测分布分析"""
        print("\n=== 预测分布分析 ===")
        
        # 训练一个基础模型
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=25, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # 生成预测
        train_pred = rf_model.predict(X_train)
        test_pred = rf_model.predict(X_test)
        
        # 分析预测分布
        print("训练集真实值分布:")
        print(f"  均值: {y_train.mean():.2f}")
        print(f"  标准差: {y_train.std():.2f}")
        print(f"  分位数: {np.percentile(y_train, [25, 50, 75])}")
        
        print("训练集预测值分布:")
        print(f"  均值: {train_pred.mean():.2f}")
        print(f"  标准差: {train_pred.std():.2f}")
        print(f"  分位数: {np.percentile(train_pred, [25, 50, 75])}")
        
        print("测试集预测值分布:")
        print(f"  均值: {test_pred.mean():.2f}")
        print(f"  标准差: {test_pred.std():.2f}")
        print(f"  分位数: {np.percentile(test_pred, [25, 50, 75])}")
        
        # 检查异常值
        train_mae = mean_absolute_error(y_train, train_pred)
        print(f"\n训练集MAE: {train_mae:.4f}")
        
        # 检查负值预测
        negative_preds = test_pred[test_pred < 0]
        if len(negative_preds) > 0:
            print(f"警告: 测试集中有 {len(negative_preds)} 个负值预测")
            test_pred = np.maximum(test_pred, 0)  # 修正负值
        
        return rf_model, train_pred, test_pred
    
    def feature_importance_analysis(self, model, feature_cols):
        """特征重要性分析"""
        print("\n=== 特征重要性分析 ===")
        
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Top 20 重要特征:")
        for idx, row in feature_importance.head(20).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    
    def generate_robust_predictions(self, X_train, y_train, X_test):
        """生成稳健的预测结果"""
        print("\n=== 生成稳健预测 ===")
        
        # 多模型集成策略
        models = {
            'rf1': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
            'rf2': RandomForestRegressor(n_estimators=300, max_depth=25, random_state=123),
            'rf3': RandomForestRegressor(n_estimators=250, max_depth=22, random_state=456)
        }
        
        predictions = []
        for name, model in models.items():
            print(f"训练 {name}...")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pred = np.maximum(pred, 0)  # 确保非负
            predictions.append(pred)
        
        # 集成预测 - 使用中位数减少异常值影响
        ensemble_pred = np.median(predictions, axis=0)
        
        print(f"集成预测统计:")
        print(f"  均值: {ensemble_pred.mean():.2f}")
        print(f"  标准差: {ensemble_pred.std():.2f}")
        print(f"  最小值: {ensemble_pred.min():.2f}")
        print(f"  最大值: {ensemble_pred.max():.2f}")
        
        return ensemble_pred
    
    def save_diagnostic_results(self, test_pred):
        """保存诊断结果"""
        print("\n=== 保存诊断结果 ===")
        
        # 创建提交文件
        submission = pd.DataFrame({
            'SaleID': range(len(test_pred)),
            'price': test_pred
        })
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'结果报告/rf_diagnostic_{timestamp}.csv'
        
        submission.to_csv(filename, index=False)
        print(f"诊断结果已保存到: {filename}")
        
        return filename
    
    def run_full_diagnostic(self):
        """运行完整诊断"""
        print("开始随机森林诊断分析...")
        
        # 1. 加载数据
        if not self.load_data():
            return
        
        # 2. 数据一致性分析
        common_features, dist_diffs = self.analyze_data_consistency()
        
        # 3. 准备建模数据
        X_train, y_train, X_test, feature_cols = self.prepare_data_for_modeling(common_features)
        if X_train is None:
            return
        
        # 4. 交叉验证分析
        cv_results, rf_configs = self.cross_validation_analysis(X_train, y_train)
        
        # 5. 留出验证分析
        holdout_results = self.holdout_validation_analysis(X_train, y_train)
        
        # 6. 预测分布分析
        model, train_pred, test_pred = self.prediction_distribution_analysis(X_train, y_train, X_test)
        
        # 7. 特征重要性分析
        feature_importance = self.feature_importance_analysis(model, feature_cols)
        
        # 8. 生成稳健预测
        robust_pred = self.generate_robust_predictions(X_train, y_train, X_test)
        
        # 9. 保存结果
        filename = self.save_diagnostic_results(robust_pred)
        
        # 10. 总结建议
        self.print_diagnostic_summary(cv_results, holdout_results)
        
        return filename
    
    def print_diagnostic_summary(self, cv_results, holdout_results):
        """打印诊断总结"""
        print("\n" + "="*50)
        print("诊断总结和建议")
        print("="*50)
        
        print("\n1. 交叉验证结果总结:")
        for name, result in cv_results.items():
            print(f"   {name}: MAE = {result['cv_mae_mean']:.4f} ± {result['cv_mae_std']:.4f}")
        
        print("\n2. 留出验证结果总结:")
        for name, result in holdout_results.items():
            print(f"   {name}: MAE = {result['mae']:.4f}")
        
        print("\n3. 可能的问题分析:")
        print("   - 如果交叉验证结果很好但考试结果差，可能存在数据泄漏")
        print("   - 如果留出验证结果与交叉验证差异很大，可能存在过拟合")
        print("   - 训练集和测试集分布差异可能导致泛化性能下降")
        
        print("\n4. 建议的改进策略:")
        print("   - 使用更保守的模型参数减少过拟合")
        print("   - 采用集成方法提高预测稳定性")
        print("   - 重点关注特征工程而非模型复杂度")
        print("   - 使用分位数校准确保预测分布合理")

if __name__ == "__main__":
    analyzer = RFDiagnosticAnalyzer()
    result_file = analyzer.run_full_diagnostic()
    if result_file:
        print(f"\n诊断完成！结果文件: {result_file}")