#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的随机森林建模脚本
使用简单的集成模型来模拟随机森林的效果
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleRegressionModel:
    """简单回归模型类"""
    
    def __init__(self, n_models=20, random_state=42):
        """初始化模型集成"""
        self.n_models = n_models
        self.random_state = random_state
        self.models = []
        self.feature_coefficients = {}
        self.is_trained = False
        np.random.seed(random_state)
    
    def simple_linear_regression(self, X, y):
        """简单线性回归实现"""
        # 添加偏置项
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # 正规方程求解：theta = (X^T * X)^(-1) * X^T * y
        try:
            XtX = np.dot(X_with_bias.T, X_with_bias)
            XtX_inv = np.linalg.inv(XtX + np.eye(XtX.shape[0]) * 1e-6)  # 添加正则化
            Xty = np.dot(X_with_bias.T, y)
            theta = np.dot(XtX_inv, Xty)
            return theta
        except:
            # 如果矩阵奇异，使用岭回归
            XtX = np.dot(X_with_bias.T, X_with_bias)
            ridge_term = np.eye(XtX.shape[0]) * 0.1
            XtX_reg = XtX + ridge_term
            Xty = np.dot(X_with_bias.T, y)
            theta = np.linalg.solve(XtX_reg, Xty)
            return theta
    
    def predict_with_model(self, X, theta):
        """使用模型进行预测"""
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        return np.dot(X_with_bias, theta)
    
    def fit(self, X, y):
        """训练集成模型"""
        print(f"开始训练简化随机森林模型（{self.n_models}个基模型）...")
        
        n_samples, n_features = X.shape
        self.feature_names = X.columns.tolist()
        self.models = []
        
        # 训练多个基模型
        for i in range(self.n_models):
            if (i + 1) % 5 == 0:
                print(f"正在训练第 {i + 1}/{self.n_models} 个模型...")
            
            # 随机特征子集选择
            n_features_subset = max(5, int(np.sqrt(n_features)))
            feature_indices = np.random.choice(n_features, n_features_subset, replace=False)
            
            # Bootstrap采样
            bootstrap_indices = np.random.choice(n_samples, int(0.8 * n_samples), replace=True)
            
            # 选择特征子集和样本子集
            X_subset = X.iloc[bootstrap_indices, feature_indices]
            y_subset = y.iloc[bootstrap_indices]
            
            # 训练简单线性回归模型
            theta = self.simple_linear_regression(X_subset.values, y_subset.values)
            
            # 保存模型信息
            model_info = {
                'theta': theta,
                'feature_indices': feature_indices,
                'feature_names': [self.feature_names[idx] for idx in feature_indices]
            }
            self.models.append(model_info)
        
        self.is_trained = True
        print("模型训练完成！")
    
    def predict(self, X):
        """集成预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练！")
        
        print("开始预测...")
        predictions = []
        
        for i, model_info in enumerate(self.models):
            if (i + 1) % 5 == 0:
                print(f"正在使用第 {i + 1}/{len(self.models)} 个模型预测...")
            
            # 获取特征子集
            X_subset = X.iloc[:, model_info['feature_indices']]
            
            # 使用模型预测
            pred = self.predict_with_model(X_subset.values, model_info['theta'])
            predictions.append(pred)
        
        # 平均预测结果
        final_predictions = np.mean(predictions, axis=0)
        print("预测完成！")
        return final_predictions
    
    def get_feature_importance(self):
        """计算特征重要性"""
        if not self.is_trained:
            raise ValueError("模型尚未训练！")
        
        # 统计每个特征被使用的次数和权重
        feature_importance = {}
        for feature_name in self.feature_names:
            feature_importance[feature_name] = 0
        
        for model_info in self.models:
            theta = model_info['theta'][1:]  # 排除偏置项
            feature_names = model_info['feature_names']
            
            for i, feature_name in enumerate(feature_names):
                feature_importance[feature_name] += abs(theta[i])
        
        # 归一化
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            for feature_name in feature_importance:
                feature_importance[feature_name] /= total_importance
        
        # 转换为DataFrame
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df

def load_processed_data():
    """加载预处理后的数据"""
    print("加载预处理后的数据...")
    
    # 加载训练集
    train_df = pd.read_csv('processed_train_data.csv')
    print(f"训练集加载完成，形状: {train_df.shape}")
    
    # 加载测试集
    test_df = pd.read_csv('processed_test_data.csv')
    print(f"测试集加载完成，形状: {test_df.shape}")
    
    # 加载原始测试集以获取SaleID
    original_test = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    sale_ids = original_test['SaleID'].values
    print(f"获取到 {len(sale_ids)} 个SaleID")
    
    return train_df, test_df, sale_ids

def prepare_features(train_df, test_df):
    """准备特征数据"""
    print("准备特征数据...")
    
    # 分离特征和目标变量
    X_train = train_df.drop('price', axis=1)
    y_train = train_df['price']
    X_test = test_df
    
    print(f"训练特征形状: {X_train.shape}")
    print(f"训练目标形状: {y_train.shape}")
    print(f"测试特征形状: {X_test.shape}")
    
    # 确保训练集和测试集有相同的特征
    common_features = list(set(X_train.columns) & set(X_test.columns))
    common_features.sort()  # 保持顺序一致
    
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    print(f"使用 {len(common_features)} 个共同特征")
    
    return X_train, y_train, X_test

def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    # 计算RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 计算MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 计算R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

def train_and_predict():
    """训练模型并预测"""
    print("="*50)
    print("简化随机森林建模开始")
    print("="*50)
    
    # 加载数据
    train_df, test_df, sale_ids = load_processed_data()
    
    # 准备特征
    X_train, y_train, X_test = prepare_features(train_df, test_df)
    
    # 创建并训练模型
    model = SimpleRegressionModel(n_models=20, random_state=42)
    model.fit(X_train, y_train)
    
    # 训练集预测
    print("\n评估模型性能...")
    train_pred = model.predict(X_train)
    
    # 评估模型
    metrics = evaluate_model(y_train, train_pred)
    print("\n模型性能评估:")
    for metric, value in metrics.items():
        print(f"训练集 {metric}: {value:.4f}")
    
    # 特征重要性
    print("\n计算特征重要性...")
    feature_importance = model.get_feature_importance()
    print("前10个重要特征:")
    print(feature_importance.head(10))
    
    # 测试集预测
    print("\n开始预测测试集...")
    test_pred = model.predict(X_test)
    
    return test_pred, sale_ids, model, feature_importance

def save_predictions(test_pred, sale_ids):
    """保存预测结果"""
    print("\n保存预测结果...")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'SaleID': sale_ids,
        'price': test_pred
    })
    
    # 生成文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rf_result_{current_time}.csv"
    
    # 保存结果
    result_df.to_csv(filename, index=False)
    
    print(f"预测结果已保存至: {filename}")
    print(f"预测了 {len(result_df)} 条记录")
    
    # 显示预测结果统计
    print("\n预测结果统计:")
    print(f"预测价格均值: {test_pred.mean():.2f}")
    print(f"预测价格中位数: {np.median(test_pred):.2f}")
    print(f"预测价格标准差: {test_pred.std():.2f}")
    print(f"预测价格范围: {test_pred.min():.2f} - {test_pred.max():.2f}")
    
    return filename, result_df

def save_model_and_results(model, feature_importance):
    """保存模型和结果"""
    print("\n保存模型和特征重要性...")
    
    # 保存模型
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"rf_model_{current_time}.pkl"
    
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存至: {model_filename}")
    except Exception as e:
        print(f"模型保存失败: {e}")
    
    # 保存特征重要性
    if feature_importance is not None:
        importance_filename = f"feature_importance_{current_time}.csv"
        feature_importance.to_csv(importance_filename, index=False)
        print(f"特征重要性已保存至: {importance_filename}")

def main():
    """主函数"""
    try:
        # 训练模型并预测
        test_pred, sale_ids, model, feature_importance = train_and_predict()
        
        # 保存预测结果
        filename, result_df = save_predictions(test_pred, sale_ids)
        
        # 保存模型和其他结果
        save_model_and_results(model, feature_importance)
        
        print("="*50)
        print("简化随机森林建模完成！")
        print("="*50)
        
        return filename, result_df
        
    except Exception as e:
        print(f"建模过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()